#!/usr/bin/env python
# Evaluate target panels, prior construction and cross-context transfer.
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
R = __import__("os").environ["GEOBRIDGE_ROOT"]


def _sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


SCRIPT_SHA0 = _sha(__file__)
KS = (50, 100, 200)
MIN_RESP_GATE = 20                                               # gated TF = >= 20 testable responders (same floor as the lattice gate)
BLOCK = {"GATA1": "ERYTHROID", "KLF1": "ERYTHROID", "PANC": "PANC", "BM_CHIPATLAS": "BM_CHIPATLAS", "BM_BREADTH": "BM_BREADTH", "K562_EMP_ALLNULL": "K562"}
DOMINANT = {"BM_BREADTH": {"GATA1", "GFI1B", "RUNX1"}, "BM_CHIPATLAS": {"GATA1", "RUNX1"}}     # dominant-TF exclusion sensitivity (BM)
SIMPLE_EXCL = ("P_pairU", "P_pairM", "P_cellU", "P_cellM", "FIT_GB")                # GB family / fitted model are not 'simple panels'
OWN_AXIS = {"GATA1": "GATA1", "KLF1": "GATA1", "PANC": "PANC", "BM_CHIPATLAS": "BONEMARROW", "BM_BREADTH": "BONEMARROW", "K562_EMP_ALLNULL": "K562"}
DONORS = ["GATA1", "PANC", "BONEMARROW", "K562", "DENTATEGYRUS", "GASTRULATION"]    # catbank axes with priors.npz (zebrafish: no 1:1 ortholog table here -> not used)
MOUSE_AXES = {"GATA1", "PANC", "DENTATEGYRUS", "GASTRULATION"}


# canonical method name: GeoBRIDGE -> P_pairU; '|Pearson|' -> pearson; 'RegVelo |J|' -> regvelo
def cname(m):
    if m == "GeoBRIDGE":
        return "P_pairU"
    return m.replace("|", " ").split()[0].lower()


# pure numerics
# deterministic top-K indices of v among its finite entries: descending value, ties by ascending index
def topk(v, K):
    v = np.asarray(v, float)
    fin = np.flatnonzero(np.isfinite(v))
    return fin[np.lexsort((fin, -v[fin]))][:K]


# rank decile 0..9 of x over `mask` (average ranks); -1 outside
def decile(x, mask):
    from scipy.stats import rankdata
    out = np.full(len(x), -1, int)
    idx = np.flatnonzero(mask)
    r = rankdata(np.asarray(x, float)[idx])
    out[idx] = np.minimum(((r - 1) / len(idx) * 10).astype(int), 9)
    return out


# B random panels with the same (abundance decile x detection decile) composition as the panel ref_idx, drawn from the strata pools
def matched_random(ref_idx, strata, rng, B):
    out = np.empty((B, len(ref_idx)), int)
    s_ref = strata[ref_idx]
    col = 0
    for s in np.unique(s_ref):
        c = int((s_ref == s).sum())
        pool = np.flatnonzero(strata == s)
        if c <= len(pool):
            draw = pool[rng.random((B, len(pool))).argsort(1)[:, :c]]
        else:
            draw = pool[rng.integers(0, len(pool), (B, c))]
        out[:, col:col + c] = draw
        col += c
    return out


# sel = panel gene indices. untestable slots are counted and never refilled
def panel_metrics(sel, resp, test):
    n = len(sel)
    tt = test[sel]
    hit = resp[sel] & tt
    return dict(n_slots=n, n_untestable=int((~tt).sum()), n_testable_slots=int(tt.sum()), captured=int(hit.sum()))


# gene -> shared 'human-symbol' key
# Human axes: upper-case symbol
# Mouse axes: 1:1 mouse<->human orthologs of HOM_MouseHumanSequence.rpt (as gb_power_match_v2 PANC); unmapped -> None
def keymap_factory():
    hom = pd.read_csv(f"{R}/data/orthology/HOM_MouseHumanSequence.rpt", sep="\t", low_memory=False)
    kk = hom.columns[0]
    org = hom["Common Organism Name"]
    mm = hom[org.str.startswith("mouse")].groupby(kk)["Symbol"].apply(list)
    hs = hom[org.str.startswith("human")].groupby(kk)["Symbol"].apply(list)
    m2h = {mm[k][0]: hs[k][0].upper() for k in set(mm.index) & set(hs.index) if len(mm[k]) == 1 and len(hs[k]) == 1}

    def keys(genes, mouse):
        g = np.asarray(genes).astype(str)
        return np.array([m2h.get(x) for x in g], object) if mouse else np.array([x.upper() for x in g], object)
    return keys


def is_mouse_genes(genes):
    g = np.asarray(genes).astype(str)
    return bool(np.mean([any(c.islower() for c in x) for x in g]) > 0.5)


# rank-normal(v) residual after OLS on rank-normal(X) columns over mask (NaN outside); truth-free control-expression adjustment
def rank_resid(v, X, mask):
    from scipy.stats import norm, rankdata
    idx = np.flatnonzero(mask)
    f = lambda a: norm.ppf((rankdata(np.asarray(a, float)[idx]) - 0.5) / len(idx))
    y = f(v)
    A_ = np.column_stack([np.ones(len(idx))] + [f(X[:, j]) for j in range(X.shape[1])])
    out = np.full(len(v), np.nan)
    out[idx] = y - A_ @ np.linalg.lstsq(A_, y, rcond=None)[0]
    return out


# unit cache
def save_unit(d, name, arrs, meta):
    os.makedirs(d, exist_ok=True)
    tmp = f"{d}/{name}.tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta)), **arrs)
    os.replace(tmp, f"{d}/{name}.npz")


# -> list of unit dicts (system, tf, genes, test, resp, Q, vec{name: arr}, abund, det, meta)
def load_units(d, systems=None):
    out = []
    for f in sorted(glob.glob(f"{d}/*.npz")):
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        if systems and meta["system"] not in systems:
            continue
        vec = {k[2:]: np.asarray(z[k], float) for k in z.files if k.startswith("v|")}
        base = dict(system=meta["system"], genes=z["genes"].astype(str), vec=vec, abund=np.asarray(z["abund"], float), det=np.asarray(z["det"], float), Q=np.asarray(z["Q"], bool), meta=meta)
        if meta["system"] == "BM_BREADTH":
            T, Rr = np.asarray(z["T"], bool), np.asarray(z["Rr"], bool)
            for i, p in enumerate(z["perts"].astype(str)):
                out.append(dict(base, tf=p, test=T[i], resp=Rr[i] & T[i]))
        else:
            te = np.asarray(z["test"], bool)
            out.append(dict(base, tf=meta["tf"], test=te, resp=np.asarray(z["resp"], bool) & te))
    return out


def cov_seed(system):
    return int(np.random.SeedSequence([20260930, zlib.crc32(system.encode())]).generate_state(1)[0])


# units: lattice systems
def build_lattice_unit(system, tf, L, blk, aud, A, AD, GD):
    cfg = L.SYSTEMS[system]
    bank = A.load_bank(cfg["axis"])
    ep = L.build_endpoint(system, tf, blk, False)                                   # truth is joined inside build_endpoint after every score artifact
    V, genes = ep["V"], np.asarray(ep["genes"]).astype(str)
    use = [m for m in A.METHODS if m in V and m in bank.X]
    if A.GB not in use:
        raise L.Blocked(f"GB_ROW_OR_BANK_MISSING {system}/{tf}")
    r = bank.row_of(tf)
    if r is None:
        raise L.Blocked(f"FOCAL_NOT_IN_BANK {system}/{tf}")
    finV = np.all([np.isfinite(V[m]) for m in use], 0)
    ai, has, ai0, tc, prior, Qcore = AD.masks_and_priors(bank, genes, V, use, r, finV)        # control-only universe: no truth (M0 = finite scores only)
    abl = GD.ablation_priors(cfg["axis"], bank, r, tc, ai0)
    refs = list(L.REFS)
    Xr = np.column_stack([np.asarray(V[n], float) for n in refs])
    Q = Qcore & np.isfinite(Xr).all(1) & np.all([np.isfinite(v) for v in abl.values()], 0)
    vec, dropped = {}, []
    for m in use:
        vec[cname(m)] = np.where(Q, prior[m], np.nan)
    for n, v in abl.items():
        vec[n] = np.where(Q, v, np.nan)
    for c in ("COMP_Z", "COMP_RANK"):
        if c in V and np.isfinite(np.asarray(V[c], float)[Q]).all():
            vec[c] = np.where(Q, np.asarray(V[c], float), np.nan)
        else:
            dropped.append(c)
    for n, j in zip(refs, range(len(refs))):
        vec[n] = np.where(Q, Xr[:, j], np.nan)
    fit, _, st = AD.hgb_cross_fit(GD.zrow(prior[A.GB], Q), Xr, np.asarray(ep["fold"]), Q)
    vec["FIT_GB"] = np.where(Q, fit, np.nan)
    # parity: PRIOR AUROC on Qcore & U vs audit_decomp_v2 (priors do not depend on the mask -> exact)
    y, U = np.asarray(ep["resp"], bool), np.asarray(ep["U"], bool)
    M1 = Qcore & U
    a = aud[(aud.system == system) & (aud.tf == tf) & (aud.kind == "lattice") & (aud.target == "resp") & aud.status.isin(["OK", "DESCRIPTIVE_ONLY"])]
    worst, n = 0.0, 0
    for _, q in a.iterrows():
        if q.method in use and M1.any() and y[M1].any() and (~y[M1]).any():
            got = AD.auroc_rows(prior[q.method][None, :][:, M1], y[M1])[0]
            worst, n = max(worst, abs(got - q.AUROC_prior)), n + 1
    if worst > 1e-9:
        raise SystemExit(f"REFUSING_PARITY {system}/{tf}: PRIOR AUROC differs from audit_decomp_v2 by {worst:.3e} over {n} cells")
    meta = dict(system=system, tf=tf, n_axis=len(genes), n_Qcore=int(Qcore.sum()), n_Q=int(Q.sum()), n_U=int(U.sum()), n_E=int((Q & U).sum()), n_resp_E=int((Q & U & y).sum()), methods=use,
                dropped=dropped, parity=dict(n_cells=n, worst_absdiff=worst, note="PRIOR vectors vs audit_decomp_v2 audit_per_tf on Qcore&U; FIT is trained on Q (control-only), not on U-restricted M1, so FIT parity is not asserted"),
                hgb_R2=st["R2_heldout"], U_definition="lattice ep['U'] = truth resp_ok & occ_ok (the frozen evaluation universe)")
    return dict(genes=genes, test=U, resp=y & U, Q=Q, abund=-np.asarray(V["REF -abundance"], float), det=-np.asarray(V["REF -raw_det"], float), **{f"v|{k}": v for k, v in vec.items()}), meta


def cmd_units(a):
    import gb_attribution as A
    import gb_audit_decomp as AD
    import gb_decompose as GD
    import gb_lattice_extra as LE
    import gb_layer_lattice as L
    LE.register()
    os.makedirs(a.out, exist_ok=True)
    aud = pd.read_csv(f"{R}/results/real_uniform/audit_decomp_v2/audit_per_tf.tsv", sep="\t", dtype={"tf": str})
    for system in [s.strip() for s in a.systems.split(",") if s.strip()]:
        if system == "BM_BREADTH":
            build_bm_breadth(a, A, AD)
            continue
        cfg = L.SYSTEMS[system]
        allowed = sorted(aud[(aud.system == system) & (aud.kind == "lattice")].tf.unique())
        tfs = [t.strip() for t in a.tfs.split(",") if t.strip()] or allowed
        bad = [t for t in tfs if t not in allowed]
        if bad:
            raise SystemExit(f"REFUSING: --tfs {bad} not audited for {system}: {allowed}")
        if a.max_tfs:
            tfs = tfs[:a.max_tfs]
        if a.nslice > 1:
            tfs = [t for i, t in enumerate(tfs) if i % a.nslice == a.slice]
        seed = int(np.random.SeedSequence([L.BOOT_SEED, L.BLOCKS.index(cfg["block"])]).generate_state(1)[0])
        blk = L.BlockAxis(L.axis_of(cfg["axis"]), 10, seed, None, "gene")              # only gi / fold_of are used; counts are not
        for tf in tfs:
            t0 = time.time()
            try:
                arrs, meta = build_lattice_unit(system, tf, L, blk, aud, A, AD, GD)
            except (L.Blocked, SystemExit) as e:
                if isinstance(e, SystemExit) and "REFUSING_PARITY" in str(e):
                    raise
                print(f"[unit] {system}/{tf}: NOT_RUN {e}", flush=True)
                continue
            meta.update(script_sha256=SCRIPT_SHA0, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
            save_unit(a.out, f"{system}__{tf}", arrs, meta)
            print(f"[unit] {system}/{tf}: Q={meta['n_Q']} E={meta['n_E']} resp={meta['n_resp_E']} parity={meta['parity']['worst_absdiff']:.1e}/{meta['parity']['n_cells']} {time.time()-t0:.0f}s", flush=True)
    assert _sha(__file__) == SCRIPT_SHA0


# units: BM breadth (gb_ladder)
def build_bm_breadth(a, A, AD):
    import gb_ladder as LD
    bank = LD.load_bank("BONEMARROW", "", "T7", False, roots=False, tier="F1")
    comps, cdiag = LD.comp_vectors(bank)
    tr = LD.load_breadth_truth("BM", "TOTAL", "BM19")                              # truth joined after every bank object exists; calibration verdict recorded below
    bi = LD.axis_index(bank["genes"], tr["genes"])
    okg = bi >= 0
    if len(np.unique(bi[okg])) != okg.sum():
        raise SystemExit("BM_BREADTH: truth genes map non-injectively onto the bank axis")
    perts = tr["perts"].astype(str)
    if a.max_tfs:
        perts = perts[:a.max_tfs]
    keep = np.isin(tr["perts"].astype(str), perts)
    rows_ex = [bank["members"].index(p) for p in np.char.upper(tr["perts"].astype(str)) if p in bank["src_idx"]]       # all perturbed members out of the prior (ladder 'excl' variant)
    G = len(bank["genes"])
    names = LD.GB_PRIORS + bank["gens"]
    vec = {cname("GeoBRIDGE") if m == "P_pairU" else (m if m.startswith("P_") else cname(m)): LD.prior_excl(bank, m, rows_ex) for m in names}
    sota = {cname(m): LD.prior_excl(bank, m, rows_ex, zsc=bank["ZSC_SOTA"]) for m in bank["sota"]}
    refs = LD.ref_vectors(bank["feat"])
    Q = np.all([np.isfinite(v) for v in vec.values()], 0) & np.all([np.isfinite(v) for v in refs.values()], 0) & np.all([np.isfinite(v) for v in comps.values()], 0)
    for k, v in {**refs, **comps}.items():
        vec[k] = v
    for k, v in sota.items():
        vec[k] = v                                                                 # partial universe (optional SOTA priors): ranked within their own finite set
    zs = lambda v: (v - v[Q].mean()) / v[Q].std()
    fit, _, st = AD.hgb_cross_fit(zs(vec["P_pairU"]), np.column_stack(list(refs.values())), LD.fold_of(bank["genes"]), Q)
    vec["FIT_GB"] = fit
    for k in list(vec):
        if k not in sota:
            vec[k] = np.where(Q, vec[k], np.nan)
    T, Rr = np.zeros((len(perts), G), bool), np.zeros((len(perts), G), bool)
    T[:, bi[okg]] = tr["T"][keep][:, okg]
    Rr[:, bi[okg]] = tr["R"][keep][:, okg]
    meta = dict(system="BM_BREADTH", tf="ALL", n_axis=G, n_Q=int(Q.sum()), n_perts=len(perts), perts=list(perts), n_excluded_members=len(rows_ex), calibration=tr["calibration"], efficacy_status=tr["efficacy_status"],
                truth_receipt_sha256=tr["truth_receipt_sha256"], bank_files=bank["files"], composites=str(cdiag), hgb_R2=st["R2_heldout"], script_sha256=SCRIPT_SHA0,
                prior_rule="bank with ALL perturbed members removed (ladder 'excl' variant): ONE frozen prior/panel per method for every held-out perturbation",
                U_definition="per-perturbation testable pairs (breadth truth T matrix), genes with n_testable>0", created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    save_unit(a.out, "BM_BREADTH__ALL", dict(genes=np.asarray(bank["genes"]).astype(str), Q=Q, abund=bank["feat"]["abundance"], det=bank["feat"]["raw_det"], T=T, Rr=Rr, perts=perts.astype(str),
                                             **{f"v|{k}": v for k, v in vec.items()}), meta)
    print(f"[unit] BM_BREADTH: {len(perts)} perts, Q={int(Q.sum())}/{G}, calib={tr['calibration']}", flush=True)


# (2) panels
def tf_boot_ci(d, rng, n=4000):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return np.nan, np.nan
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def cmd_panel(a):
    import gb_audit_decomp as AD
    units = load_units(a.units, set(a.systems.split(",")) if a.systems else None)
    if not units:
        raise SystemExit("REFUSING: no units")
    os.makedirs(a.out, exist_ok=True)
    # Select panels from control-only vectors.
    panels, rows_p = {}, []
    for u in units:
        for K in KS:
            for nm, v in u["vec"].items():
                sel = topk(v, K)
                panels[(u["system"], u["tf"], K, nm)] = sel
                if nm in ("P_pairU",) or nm.startswith("COMP") or nm.startswith("REF") or nm in ("pearson", "P_pairM", "P_cellU", "P_cellM"):
                    g = u["genes"][sel]
                    rows_p.append(dict(system=u["system"], tf=u["tf"], K=K, vector=nm, n=len(sel), panel_sha256=hashlib.sha256("\n".join(g).encode()).hexdigest(), genes=",".join(g)))
    pf = f"{a.out}/panels_frozen.tsv.gz"
    pd.DataFrame(rows_p).to_csv(pf, sep="\t", index=False)
    panel_sha = _sha(pf)
    # Evaluate panels on held-out perturbations.
    rng = np.random.default_rng(cov_seed("panel"))
    rows, captured = [], {}
    for u in units:
        resp, test, Q = u["resp"], u["test"], u["Q"]
        E = Q & test
        n_resp, n_E = int((resp & E).sum()), int(E.sum())
        strata = np.where(Q, decile(u["abund"], Q) * 10 + decile(u["det"], Q), -1)
        for K in KS:
            for nm in u["vec"]:
                sel = panels[(u["system"], u["tf"], K, nm)]
                m = panel_metrics(sel, resp, test)
                hit = sel[resp[sel] & test[sel]]
                captured[(u["system"], u["tf"], K, nm)] = hit
                rows.append(dict(system=u["system"], block=BLOCK[u["system"]], tf=u["tf"], K=K, vector=nm, n_resp=n_resp, n_E=n_E, prevalence=n_resp / max(n_E, 1),
                                 recall=m["captured"] / n_resp if n_resp else np.nan, prec_testable=m["captured"] / m["n_testable_slots"] if m["n_testable_slots"] else np.nan,
                                 any1=float(m["captured"] >= 1), ge3=float(m["captured"] >= 3), **m))
            # expression-matched random panels (matched to the GB prior panel composition):
            # expectation over B draws
            ref = panels[(u["system"], u["tf"], K, "P_pairU")]
            Rp = matched_random(ref, strata, rng, a.B_rand)
            cap = (resp & test)[Rp].sum(1)
            unt = (~test[Rp]).sum(1)
            gbc = panel_metrics(ref, resp, test)["captured"]
            rows.append(dict(system=u["system"], block=BLOCK[u["system"]], tf=u["tf"], K=K, vector="RANDOM_matched", n_resp=n_resp, n_E=n_E, prevalence=n_resp / max(n_E, 1), n_slots=len(ref),
                             n_untestable=float(unt.mean()), n_testable_slots=float(len(ref) - unt.mean()), captured=float(cap.mean()), recall=float(cap.mean() / n_resp) if n_resp else np.nan,
                             prec_testable=float(cap.sum() / max((len(ref) * len(cap) - unt.sum()), 1)), any1=float((cap >= 1).mean()), ge3=float((cap >= 3).mean()),
                             rand_q95_captured=float(np.percentile(cap, 95)), p_ge_gb=float((1 + (cap >= gbc).sum()) / (len(cap) + 1)), n_draws=len(cap)))
    D = pd.DataFrame(rows)
    D["gated"] = D.n_resp >= MIN_RESP_GATE
    D.to_csv(f"{a.out}/tf_metrics.tsv", sep="\t", index=False)
    # aggregates: block x variant x K x vector (TF equal weight); union coverage from the captured sets
    def union_cov(sys_, tfs, K, nm):
        us = [u for u in units if u["system"] == sys_ and u["tf"] in tfs]
        allr = set()
        cap = set()
        for u in us:
            allr |= set(np.flatnonzero(u["resp"] & u["Q"] & u["test"]))
            cap |= set(captured[(sys_, u["tf"], K, nm)]) if nm != "RANDOM_matched" else set()
        return len(cap) / len(allr) if allr and nm != "RANDOM_matched" else np.nan
    agg, gains, gains_r = [], [], []
    rng2 = np.random.default_rng(cov_seed("panel_agg"))
    variants = [("all_tfs", lambda d, b: d), ("gated", lambda d, b: d[d.gated]),
                ("excl_dominant", lambda d, b: d[~d.tf.isin(DOMINANT.get(b, set()))] if b in DOMINANT else None)]
    for blk_name, Db in D.groupby("block"):
        for vname, fn in variants:
            Dv = fn(Db, blk_name)
            if Dv is None or not len(Dv):
                continue
            for K, DK in Dv.groupby("K"):
                for nm in DK.vector.unique():
                    g = DK[DK.vector == nm]
                    sysn = g.system.iloc[0]
                    agg.append(dict(block=blk_name, variant=vname, K=K, vector=nm, n_tf=len(g), n_tf_with_resp=int((g.n_resp > 0).sum()), mean_captured=g.captured.mean(), macro_recall=g.recall.mean(),
                                    frac_tf_any1=g.any1.mean(), frac_tf_ge3=g.ge3.mean(), mean_prec_testable=g.prec_testable.mean(), mean_untestable_slots=g.n_untestable.mean(),
                                    micro_recall=g.captured.sum() / g.n_resp.sum() if g.n_resp.sum() else np.nan, union_coverage=union_cov(sysn, set(g.tf), K, nm)))
                simple = [v for v in DK.vector.unique() if v.startswith("REF") or v in ("pearson", "COMP_Z", "COMP_RANK")]
                mr = {v: DK[DK.vector == v].set_index("tf").recall.mean() for v in simple}
                best = max(mr, key=lambda k: (mr[k] if np.isfinite(mr[k]) else -1, k))
                gbv = DK[DK.vector == "P_pairU"].set_index("tf")
                for comp_name, comp_lab in ((best, "strongest_simple(hindsight)"), ("RANDOM_matched", "expression_matched_random"), ("pearson", "control_covariance"), ("COMP_Z", "composite_Z"),
                                            ("P_pairM", "mature_only"), ("P_cellU", "cell_U"), ("P_cellM", "cell_M")):
                    cv = DK[DK.vector == comp_name].set_index("tf")
                    if not len(cv):
                        continue
                    for met in ("recall", "any1", "captured"):
                        d = (gbv[met] - cv[met].reindex(gbv.index)).to_numpy(float)
                        d = d[np.isfinite(d)]
                        lo, hi = tf_boot_ci(d, rng2)
                        p = AD.signflip(d)[0] if len(d) >= 2 else np.nan
                        gains.append(dict(block=blk_name, variant=vname, K=K, comparator=comp_name, comparator_role=comp_lab, metric=met, n_tf=len(d), gb=float(gbv[met].mean()), comp=float(cv[met].mean()),
                                          gain=float(d.mean()) if len(d) else np.nan, lo=lo, hi=hi, p_signflip=p, frac_tf_gain_pos=float((d > 0).mean()) if len(d) else np.nan, inference="TF-level; n_tf=1 -> descriptive only"))
    pd.DataFrame(agg).to_csv(f"{a.out}/block_summary.tsv", sep="\t", index=False)
    G = pd.DataFrame(gains)
    G.to_csv(f"{a.out}/gains.tsv", sep="\t", index=False)
    # cross-block (one vote per block)
    cb_rows = []
    for (vname, K, role, met), g in G[G.comparator_role.isin(["strongest_simple(hindsight)", "expression_matched_random", "control_covariance"])].groupby(["variant", "K", "comparator_role", "metric"]):
        d = g.gain.to_numpy(float)
        cb_rows.append(dict(variant=vname, K=K, comparator_role=role, metric=met, n_blocks=len(d), mean_gain=float(np.nanmean(d)), n_blocks_pos=int((d > 0).sum()), blocks=",".join(g.block),
                            p_signflip=AD.signflip(d)[0] if len(d) >= 2 else np.nan, min_possible_p=2.0 / 2 ** len(d) if len(d) else np.nan))
    pd.DataFrame(cb_rows).to_csv(f"{a.out}/cross_block.tsv", sep="\t", index=False)
    assert _sha(__file__) == SCRIPT_SHA0
    json.dump(dict(script=__file__, script_sha256=SCRIPT_SHA0, argv=sys.argv, panels_frozen_sha256=panel_sha, panels_frozen_written_before_labels_read=True, units=a.units,
                   unit_files={os.path.basename(f): _sha(f) for f in sorted(glob.glob(f"{a.units}/*.npz"))}, Ks=list(KS), B_rand=a.B_rand, gate_min_resp=MIN_RESP_GATE, dominant=str({k: sorted(v) for k, v in DOMINANT.items()}),
                   rules=dict(panel="top-K among the control-only universe Q of each unit, deterministic ties; untestable slots counted not refilled; LOO prior per TF (lattice) / all-perturbed-removed prior (BM_BREADTH)",
                              strongest_simple="hindsight argmax over {9 REF single-feature rankers, COMP_Z, COMP_RANK, pearson} of block macro recall per (variant,K) -> conservative comparator (max over 12)",
                              random="B_rand expression-matched (abundance decile x detection decile of the GB panel) random panels; expectation reported"),
                   created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(f"{a.out}/receipt.json", "w"), indent=1)
    print(f"[panel] {len(units)} units -> {a.out}")


# (3) construction
CONSTRUCT_VEC = ["P_pairU", "P_pairM", "P_cellU", "P_cellM", "FIT_GB", "pearson"]
PAIRS = [("P_pairU", "P_pairM"), ("P_pairU", "P_cellU"), ("P_pairU", "P_cellM"), ("P_pairU", "FIT_GB"), ("P_pairU", "pearson"), ("P_pairM", "pearson"), ("P_cellM", "pearson"),
         ("P_cellU", "P_cellM"), ("P_pairM", "P_cellM"), ("FIT_GB", "pearson")]
MET = ("AUROC", "APLIFT", "P50")


def cmd_construct(a):
    import gb_audit_decomp as AD
    import gb_canon_table as CT
    units = load_units(a.units, set(a.systems.split(",")) if a.systems else None)
    os.makedirs(a.out, exist_ok=True)
    rows, prow, cache = [], [], {}
    for u in units:
        n = len(u["genes"])
        key = (u["system"], n)
        if key not in cache:
            cache[key] = CT.make_counts(n, a.B, cov_seed("construct:" + u["system"]))
        counts, gi = cache[key], np.arange(n)
        E = u["Q"] & u["test"]
        y = u["resp"]
        n_resp, n_non = int((y & E).sum()), int((~y & E).sum())
        res = {}
        for nm in CONSTRUCT_VEC:
            if nm not in u["vec"]:
                continue
            if n_resp == 0 or n_non == 0:
                continue
            res[nm] = CT.eval_rows(u["vec"][nm], y, E, gi, counts)
            pt, bs = res[nm]
            lo, hi = np.nanpercentile(bs["AUROC"], [2.5, 97.5])
            rows.append(dict(system=u["system"], block=BLOCK[u["system"]], tf=u["tf"], vector=nm, n_E=int(E.sum()), n_resp=n_resp, gated=n_resp >= MIN_RESP_GATE, AUROC=pt["AUROC"], AUROC_lo=lo, AUROC_hi=hi,
                             APLIFT=pt["APLIFT"], AP=pt["AUPRC"], PREV=pt["PREV"], P50=pt["P50"], hits50=pt["P50"] * 50 if np.isfinite(pt["P50"]) else np.nan))
        for x, yv in PAIRS:
            if x in res and yv in res:
                for met in MET:
                    d0 = res[x][0][met] - res[yv][0][met]
                    bs = res[x][1][met] - res[yv][1][met]
                    bs = bs[np.isfinite(bs)]
                    prow.append(dict(system=u["system"], block=BLOCK[u["system"]], tf=u["tf"], a=x, b=yv, metric=met, n_resp=n_resp, gated=n_resp >= MIN_RESP_GATE, diff=d0,
                                     lo=float(np.percentile(bs, 2.5)) if len(bs) else np.nan, hi=float(np.percentile(bs, 97.5)) if len(bs) else np.nan,
                                     p_boot=float(min(1.0, 2 * min(1 + (bs <= 0).sum(), 1 + (bs >= 0).sum()) / (len(bs) + 1))) if len(bs) else np.nan))
    U = pd.DataFrame(rows)
    P = pd.DataFrame(prow)
    U.to_csv(f"{a.out}/unit_metrics.tsv", sep="\t", index=False)
    P.to_csv(f"{a.out}/paired_unit.tsv", sep="\t", index=False)
    rng = np.random.default_rng(cov_seed("construct_agg"))
    brow, vrow = [], []
    for (b, vname), Ub in [((b, v), (g if v == "all_tfs" else g[g.gated])) for b, g in U.groupby("block") for v in ("all_tfs", "gated")]:
        for nm, g in Ub.groupby("vector"):
            vrow.append(dict(block=b, variant=vname, vector=nm, n_tf=len(g), **{f"mean_{m}": g[m].mean() for m in ("AUROC", "APLIFT", "P50", "hits50")}))
    for (b, vname), Pb in [((b, v), (g if v == "all_tfs" else g[g.gated])) for b, g in P.groupby("block") for v in ("all_tfs", "gated")]:
        for (x, yv, met), g in Pb.groupby(["a", "b", "metric"]):
            d = g["diff"].to_numpy(float)
            d = d[np.isfinite(d)]
            lo, hi = tf_boot_ci(d, rng)
            if len(d) == 1:
                lo, hi = float(g.lo.iloc[0]), float(g.hi.iloc[0])                     # single-unit block: gene-bootstrap interval (conditional)
            brow.append(dict(block=b, variant=vname, a=x, b=yv, metric=met, n_tf=len(d), mean_diff=float(d.mean()) if len(d) else np.nan, lo=lo, hi=hi, frac_pos=float((d > 0).mean()) if len(d) else np.nan,
                             p_signflip=AD.signflip(d)[0] if len(d) >= 2 else np.nan, ci_type="TF bootstrap" if len(d) >= 2 else "gene bootstrap (single unit, conditional)"))
    pd.DataFrame(vrow).to_csv(f"{a.out}/block_vector_means.tsv", sep="\t", index=False)
    pd.DataFrame(brow).to_csv(f"{a.out}/block_paired.tsv", sep="\t", index=False)
    assert _sha(__file__) == SCRIPT_SHA0
    json.dump(dict(script=__file__, script_sha256=SCRIPT_SHA0, argv=sys.argv, units=a.units, unit_files={os.path.basename(f): _sha(f) for f in sorted(glob.glob(f"{a.units}/*.npz"))}, B=a.B,
                   note="pearson = mean over LOO catalogue bank of row-z |Pearson| (control cells only) = the catbank control-covariance prior; same bank/members/universe as the GB priors; FIT_GB = 9-REF-feature HGB reconstruction of the GB prior, cross-fitted on the fixed gene folds (truth-free)",
                   created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(f"{a.out}/receipt.json", "w"), indent=1)
    print(f"[construct] {len(units)} units -> {a.out}")


# (4) context specificity
def cmd_context(a):
    import gb_audit_decomp as AD
    import gb_canon_table as CT
    units = load_units(a.units, set(a.systems.split(",")) if a.systems else None)
    keys = keymap_factory()
    CB = f"{R}/results/real_uniform/catbank"
    donors = {}
    for ax in DONORS:
        z = np.load(f"{CB}/{ax}_priors.npz", allow_pickle=True)
        g, P = z["T7|genes"].astype(str), np.asarray(z["T7|P_pairU|Pfull"], float)
        k = keys(g, ax in MOUSE_AXES)
        d = {}
        for kk, p in zip(k, P):
            if kk is not None and np.isfinite(p) and kk not in d:
                d[kk] = p                                                              # first occurrence; duplicated keys are rare 1:1-mapped symbols
        donors[ax] = d
    os.makedirs(a.out, exist_ok=True)
    rows, prow, cache = [], [], {}
    for u in units:
        axis = OWN_AXIS[u["system"]]
        mouse = axis in MOUSE_AXES
        assert is_mouse_genes(u["genes"]) == mouse, (u["system"], axis)
        uk = keys(u["genes"], mouse)
        n = len(u["genes"])
        ck = (u["system"], n)
        if ck not in cache:
            cache[ck] = CT.make_counts(n, a.B, cov_seed("context:" + u["system"]))
        counts, gi = cache[ck], np.arange(n)
        E = u["Q"] & u["test"]
        y = u["resp"]
        dv = {}
        for ax, d in donors.items():
            if ax == axis:
                continue
            dv[ax] = np.array([d.get(k, np.nan) if k is not None else np.nan for k in uk], float)
        own = u["vec"]["P_pairU"]
        ex = u["vec"].get("REF -abundance")
        dt = u["vec"].get("REF -raw_det")
        allv = dict(own=own, **{f"donor:{ax}": v for ax, v in dv.items()}, ctl_expr=ex, ctl_det=dt)
        allv = {k: v for k, v in allv.items() if v is not None}
        common = E.copy()
        for k, v in allv.items():
            if k.startswith("donor:"):
                common &= np.isfinite(v)
        X = np.column_stack([u["abund"], u["det"]])
        for scope, mask_fn in (("common", lambda k: common), ("pairwise", lambda k: E & np.isfinite(allv[k]) & (np.isfinite(allv["own"])) if k.startswith("donor:") else E)):
            res = {}
            for k, v in allv.items():
                m = mask_fn(k)
                if int((y & m).sum()) == 0 or int((~y & m).sum()) == 0 or m.sum() < (50 if scope == 'common' else 100):
                    continue
                for adj in ("raw", "expr_resid"):
                    vv = v if adj == "raw" else rank_resid(np.where(m, v, 0.0), X, m)
                    if adj == "expr_resid" and k in ("ctl_expr", "ctl_det"):
                        continue
                    pt, bs = CT.eval_rows(vv, y, m, gi, counts)
                    res[(k, adj)] = (pt, bs, m)
                    lo, hi = np.nanpercentile(bs["AUROC"], [2.5, 97.5])
                    rows.append(dict(system=u["system"], block=BLOCK[u["system"]], tf=u["tf"], scope=scope, source=k, adj=adj, n_rows=int(m.sum()), n_resp=int((y & m).sum()), gated=int((y & m).sum()) >= MIN_RESP_GATE,
                                     AUROC=pt["AUROC"], AUROC_lo=lo, AUROC_hi=hi, APLIFT=pt["APLIFT"], P50=pt["P50"]))
            if scope == "pairwise":
                for (k, adj), (pt, bs, m) in res.items():
                    if not k.startswith("donor:"):
                        continue
                    o = res.get(("own", adj))
                    if o is None:
                        continue
                    # paired on the donor's pairwise rows: re-evaluate own on the same rows
                    po, bo = CT.eval_rows(allv["own"] if adj == "raw" else rank_resid(np.where(m, allv["own"], 0.0), X, m), y, m, gi, counts)
                    for met in ("AUROC", "APLIFT"):
                        d = po[met] - pt[met]
                        b = bo[met] - bs[met]
                        b = b[np.isfinite(b)]
                        prow.append(dict(system=u["system"], block=BLOCK[u["system"]], tf=u["tf"], donor=k.split(":")[1], adj=adj, metric=met, n_rows=int(m.sum()), n_resp=int((y & m).sum()),
                                         gated=int((y & m).sum()) >= MIN_RESP_GATE, own=po[met], donor_val=pt[met], diff=d, lo=float(np.percentile(b, 2.5)) if len(b) else np.nan,
                                         hi=float(np.percentile(b, 97.5)) if len(b) else np.nan, own_wins=float(d > 0)))
    U = pd.DataFrame(rows)
    P = pd.DataFrame(prow)
    U.to_csv(f"{a.out}/unit_metrics.tsv", sep="\t", index=False)
    P.to_csv(f"{a.out}/paired_own_vs_donor.tsv", sep="\t", index=False)
    rng = np.random.default_rng(cov_seed("context_agg"))
    srow = []
    for (sys_, adj, met, donor), g in P.groupby(["system", "adj", "metric", "donor"]):
        for vname, gg in (("all_tfs", g), ("gated", g[g.gated])):
            d = gg["diff"].to_numpy(float)
            d = d[np.isfinite(d)]
            if not len(d):
                continue
            lo, hi = tf_boot_ci(d, rng)
            srow.append(dict(system=sys_, variant=vname, adj=adj, metric=met, donor=donor, n_tf=len(d), own=float(gg.own.mean()), donor_val=float(gg.donor_val.mean()), mean_diff=float(d.mean()), lo=lo, hi=hi,
                             frac_own_wins=float((d > 0).mean()), p_signflip=AD.signflip(d)[0] if len(d) >= 2 else np.nan, n_rows_median=float(gg.n_rows.median())))
    S = pd.DataFrame(srow)
    S.to_csv(f"{a.out}/recipient_summary.tsv", sep="\t", index=False)
    # matching-tissue verdict on the common gene set: own vs max / mean over donors, per recipient
    C = U[(U.scope == "common") & (U.adj == "raw")]
    vrow = []
    for (sys_, tf), g in C.groupby(["system", "tf"]):
        dn = g[g.source.str.startswith("donor:")]
        if not len(dn) or not (g.source == "own").any():
            continue
        o = float(g[g.source == "own"].AUROC.iloc[0])
        ex = g[g.source == "ctl_expr"].AUROC
        vrow.append(dict(system=sys_, tf=tf, gated=bool(g.gated.iloc[0]), n_rows=int(g.n_rows.iloc[0]), n_resp=int(g.n_resp.iloc[0]), own=o, donor_mean=float(dn.AUROC.mean()), donor_max=float(dn.AUROC.max()),
                         best_donor=dn.loc[dn.AUROC.idxmax(), "source"], n_donors=len(dn), ctl_expr=float(ex.iloc[0]) if len(ex) else np.nan, own_beats_all_donors=bool(o > dn.AUROC.max()),
                         own_minus_mean_donor=o - float(dn.AUROC.mean())))
    V = pd.DataFrame(vrow, columns=["system","tf","gated","n_rows","n_resp","own","donor_mean","donor_max","best_donor","n_donors","ctl_expr","own_beats_all_donors","own_minus_mean_donor"]) if not vrow else pd.DataFrame(vrow)
    V.to_csv(f"{a.out}/matching_tissue_verdict.tsv", sep="\t", index=False)
    assert _sha(__file__) == SCRIPT_SHA0
    json.dump(dict(script=__file__, script_sha256=SCRIPT_SHA0, argv=sys.argv, units=a.units, donors=DONORS, B=a.B, unit_files={os.path.basename(f): _sha(f) for f in sorted(glob.glob(f"{a.units}/*.npz"))},
                   donor_files={ax: _sha(f"{CB}/{ax}_priors.npz") for ax in DONORS}, note="donor prior = catbank T7|P_pairU|Pfull (full-bank GB prior of the donor tissue; no truth); shared genes via 1:1 orthologs (mouse axes) / upper-case symbol (human axes); "
                                                                                          "'common' = intersection over ALL donors, 'pairwise' = own vs one donor on that donor's rows; expr_resid = rank-normal residual after OLS on control abundance+detection; ZEBRAFISH not used",
                   created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())), open(f"{a.out}/receipt.json", "w"), indent=1)
    print(f"[context] {len(units)} units -> {a.out}")


# selftest
def selftest():
    rng = np.random.default_rng(0)
    v = np.array([3., 1, 3, np.nan, 2, 5])
    assert list(topk(v, 3)) == [5, 0, 2] and list(topk(v, 10)) == [5, 0, 2, 4, 1]                       # ties by index, NaN skipped
    n = 1000
    mask = np.ones(n, bool)
    ab, de = rng.normal(size=n), rng.normal(size=n)
    st = decile(ab, mask) * 10 + decile(de, mask)
    ref = np.argsort(-ab)[:50]
    Rp = matched_random(ref, st, rng, 200)
    assert Rp.shape == (200, 50) and all(sorted(st[Rp[i]]) == sorted(st[ref]) for i in range(0, 200, 37))
    assert all(len(set(Rp[i])) == 50 for i in range(20))                                                # no duplicates where the pool allows
    resp = np.zeros(n, bool)
    resp[:100] = True
    test = np.ones(n, bool)
    test[10:20] = False                                                                                 # 10 untestable genes
    sel = np.arange(0, 30)
    m = panel_metrics(sel, resp, test)
    assert m == dict(n_slots=30, n_untestable=10, n_testable_slots=20, captured=20)                      # untestable counted, never refilled
    # matched random has expectation ~ prevalence * testable slots when labels are independent of
    # the strata
    y = rng.random(n) < 0.2
    Rp = matched_random(np.arange(50), st, rng, 2000)
    assert abs(y[Rp].sum(1).mean() - 10) < 0.6
    # rank_resid removes the covariate exactly
    x = rng.normal(size=(n, 2))
    vv = 2 * x[:, 0] + rng.normal(size=n) * .1
    rr = rank_resid(vv, x, mask)
    assert abs(np.corrcoef(rr, x[:, 0])[0, 1]) < 0.05
    assert cname("|Pearson|") == "pearson" and cname("RegVelo |J|") == "regvelo" and cname("GeoBRIDGE") == "P_pairU" and cname("GENIE3") == "genie3"
    assert is_mouse_genes(["Gata1", "Klf1", "Mybl1"]) and not is_mouse_genes(["GATA1", "KLF1"])
    print("gb_readout selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    u = sp.add_parser("units")
    u.add_argument("--systems", required=True)
    u.add_argument("--tfs", default="")
    u.add_argument("--out", required=True)
    u.add_argument("--max-tfs", type=int, default=0)
    u.add_argument("--slice", type=int, default=0)
    u.add_argument("--nslice", type=int, default=1)
    for nm in ("panel", "construct", "context"):
        p = sp.add_parser(nm)
        p.add_argument("--units", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--systems", default="")
        p.add_argument("--B", type=int, default=500)
        if nm == "panel":
            p.add_argument("--B-rand", dest="B_rand", type=int, default=1000)
    a = ap.parse_args()
    {"selftest": lambda: selftest(), "units": lambda: cmd_units(a), "panel": lambda: cmd_panel(a), "construct": lambda: cmd_construct(a), "context": lambda: cmd_context(a)}[a.cmd]()


if __name__ == "__main__":
    main()
