#!/usr/bin/env python
"""D9 (method-defense) -- DEPTH-RESOLVED REGULATOR-SPECIFIC UTILITY of D = Z - P.  EVALUATION ONLY; production frozen, everything imported.

For depth sets ds in {k=0..8 single, 0:2, 0:8} (pair-U channel, same aggregator gb_primary.zcons(ks=ds) as D3):
  X_ds   = per-source PRIMARY-type row (bank order, axis columns, NaN outside the source's observational universe).
  T      = the FROZEN T7 target set of the catbank priors (PANC/BM_priors.npz 'T7|genes'); never changed by a depth set (catbank.sota_prior_arrays precedent:
           a row may be NaN on T and then contributes nothing).
  Z_ds   = row-z over T (gb_catbank.sota_prior_arrays = build_set's row_consts / Z / prior_sums), S,C = prior sums.
  P_ds^(-r) = gb_catbank.loo_prior(Z,S,C,r,2) (leave-focal-out); D_ds = Z_ds[r] - P_ds^(-r)  (so Z = P + D exactly).
EVALUATION (truth opened after every score vector is digested): AUROC of D alone, P alone, Z = P + D, paired increment AUROC(Z) - AUROC(P) (paired gene-level
resample draws, DESCRIPTIVE), on the lattice truth endpoints (resp, occ_prim, A>B, C>D) exactly as D3 (L.build_endpoint / KP.targets_of / CT.eval_rows).
MATCHED-SOURCE CONTROL: the identical statistic (leave-self-out P, D = Z_s - P^(-s) on the same T) for the N_MATCH nearest NON-cognate catalogue sources on
(spliced detection, log1p mean Ms), standardised Euclidean (decile bins leave <50 sources per cell, so nearest-neighbour matching is used), evaluated on the FOCAL
truth.  Reported: cognate percentile = frac(matched < cognate), p = (1 + #{matched >= cognate}) / (1 + n) (pool rank, pointwise, unadjusted).
PRE-DECLARED READING (coordinator, D9): shallow depth (k0, k1, k2, 0:2): percentile >= .95 AND AUROC(D) > .5 -> SPECIFIC_SHALLOW_D; otherwise
GENERIC_OR_NULL.  Labelled DESCRIPTIVE (gene-level resampling; no multiplicity correction over 11 depth sets x 4 endpoints in the pilot).
GATES (fail closed): X_0:8 == shard P|PU; X_0:8 == production GB vector; sota_prior_arrays(X_0:8) reproduces the STORED T7|P_pairU Z, S, C and loo prior.

  python scratch/gb_defense_dutil.py selftest
  python scratch/gb_defense_dutil.py run --units PANC:RFX6,BONEMARROW:RUNX1 --B 999 --out results/real_uniform/defense/D9/pilot
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gb_canon_table as CT  # noqa: E402
import gb_catbank as CB  # noqa: E402
import gb_layer_lattice as L  # noqa: E402
import gb_lattice_kprofile as KP  # noqa: E402
import gb_primary as GP  # noqa: E402

LABEL = "DESCRIPTIVE_D_UTILITY"
DSETS = {**{str(k): (k, k) for k in range(9)}, "0:2": (0, 2), "0:8": (0, 8)}
SHALLOW = ("0", "1", "2", "0:2")
EPS = ("resp", "occ_prim", "AB", "CD")
N_MATCH, MIN_CONTRIB, PARITY_TOL = 50, 2, 1e-5
OUT_DEFAULT = f"{L.R}/results/real_uniform/defense/D9"


def load_all(system):
    """raw pairU |S| at both roots for EVERY bank member + universe + stored P|PU, in bank order (validated shards, receipts)"""
    S = L.SYSTEMS[system]["axis"]
    bank, where = KP.open_bank(S)
    files = sorted({f for f, _ in where.values()})
    nb = len(bank["members"])
    raw, uni, P0, genes, recs = None, None, None, None, []
    for f in files:
        L.receipt_of(f)
        recs.append(L.frec(f))
        with np.load(f, allow_pickle=True) as z:
            idx = np.asarray(z["bank_idx"], int)
            g = z["genes"].astype(str)
            if raw is None:
                genes, G = g, len(g)
                raw = {h: np.full((nb, G, 31), np.nan, np.float32) for h in GP.ROOTS}
                uni, P0 = np.zeros((nb, G), bool), np.full((nb, G), np.nan)
            elif not np.array_equal(genes, g):
                raise L.Blocked(f"AXIS_DIFFERS_BETWEEN_SHARDS {f}")
            for h in GP.ROOTS:
                raw[h][idx] = z[f"raw|{h}|PU"]
            uni[idx], P0[idx] = z["universe"], z["P|PU"]
    return bank, raw, uni, P0, genes, recs


def ds_matrix(raw, uni, ks):
    nb, G = uni.shape
    X = np.full((nb, G), np.nan)
    for s in range(nb):
        u = uni[s]
        X[s, u] = GP.zcons({h: np.asarray(raw[h][s])[u] for h in GP.ROOTS}, ks=ks)
    return X


def stat_auroc(v, y, rows):
    r = rows & np.isfinite(v)
    return KP.auroc_pt(v, y, r)


def pool_rank(cog, matched):
    m = np.asarray([x for x in matched if np.isfinite(x)], float)
    if not np.isfinite(cog) or len(m) == 0:
        return np.nan, np.nan, len(m)
    return float((m < cog).mean() + 0.5 * (m == cog).mean()), float((1 + (m >= cog).sum()) / (1 + len(m))), len(m)


def reading(pct, auc_d, shallow):
    if not shallow:
        return "NOT_SHALLOW"
    if not (np.isfinite(pct) and np.isfinite(auc_d)):
        return "NA"
    return "SPECIFIC_SHALLOW_D" if (pct >= 0.95 and auc_d > 0.5) else "GENERIC_OR_NULL"


def nearest_matched(bank, det_lms_ctx, r, n=N_MATCH):
    """n nearest non-cognate sources; if fewer than n exist, ALL available non-cognate sources (focal excluded)"""
    members = bank["members"]
    det = np.asarray([bank["detection"][m] for m in members], float)
    ax = {g: i for i, g in enumerate(det_lms_ctx["axis"])}
    lms = np.asarray([det_lms_ctx["lms"][ax[m]] for m in members], float)
    F = np.column_stack([det, lms])
    F = (F - F.mean(0)) / F.std(0)
    d = np.sqrt(((F - F[r]) ** 2).sum(1))
    d[r] = np.inf
    o = np.argsort(d, kind="stable")[:n]
    o = o[np.isfinite(d[o])]
    dec = lambda a: np.minimum((np.argsort(np.argsort(a)) * 10 // len(a)), 9)       # noqa: E731
    return o, dict(mean_dist=float(d[o].mean()), frac_same_det_decile=float((dec(det)[o] == dec(det)[r]).mean()), frac_same_expr_decile=float((dec(lms)[o] == dec(lms)[r]).mean()))


def run_unit(system, tf, st, B, n_match=N_MATCH):
    t0 = time.time()
    cfg = L.SYSTEMS[system]
    S = cfg["axis"]
    if st.get("load") is None:
        st["load"] = load_all(system)
        pri = np.load(f"{CB.OUT}/{S}_priors.npz", allow_pickle=True)
        st["pri"] = {k: np.asarray(pri[k]) for k in pri.files if k.startswith("T7|P_pairU") or k in ("members", "axis", "T7|genes")}
        print(f"[D9] {system}: loaded {len(st['load'][0]['members'])} sources {time.time()-t0:.0f}s", flush=True)
    bank, raw, uni, P0, genes, recs = st["load"]
    pri = st["pri"]
    members = bank["members"]
    if not (np.array_equal(pri["axis"].astype(str), genes) and [str(m) for m in pri["members"]] == list(members)):
        raise L.Blocked("PRIORS_AXIS_OR_MEMBERS_DIFFER_FROM_SHARDS")
    r = next(i for i, m in enumerate(members) if m.upper() == tf.upper())
    Tm = np.isin(genes, pri["T7|genes"].astype(str))
    bank_cols = np.asarray([int(np.flatnonzero(genes == m)[0]) for m in members])
    # ---- all depth sets: frozen-T7 row-z, prior sums, leave-self-out P, D
    V, Zs = {}, {}
    G = len(genes)
    for name, (lo, hi) in DSETS.items():
        X = ds_matrix(raw, uni, tuple(range(lo, hi + 1)))
        if name == "0:8":                                                         # ---- gates
            fa, fb = np.isfinite(X), np.isfinite(P0)
            d = float(np.abs(X[fa & fb] - P0[fa & fb]).max())
            if not (np.array_equal(fa, fb) and d <= PARITY_TOL):
                raise L.Blocked(f"DS_0:8_VS_SHARD_PARITY_FAILED max|diff|={d} finiteness_equal={np.array_equal(fa, fb)}")
        PR = CB.sota_prior_arrays(X, bank_cols, Tm, MIN_CONTRIB)
        if name == "0:8":
            for k_, a_ in (("Z", PR["Z"]), ("S", PR["S"]), ("C", PR["C"]), ("Pfull", PR["Pfull"])):
                b_ = pri[f"T7|P_pairU|{k_}"]
                if not (np.array_equal(np.isfinite(a_), np.isfinite(b_)) and np.allclose(a_[np.isfinite(a_)], b_[np.isfinite(b_)], atol=1e-9, rtol=0)):
                    raise L.Blocked(f"DS_0:8_PRIOR_PARITY_FAILED vs stored T7|P_pairU|{k_}")
        Zs[name] = PR
    ctx = KP.axis_ctx(S)
    if not np.array_equal(ctx["axis"], genes):
        raise L.Blocked("CTX_AXIS_DIFFERS")
    mo, minfo = nearest_matched(bank, ctx, r, n_match)

    def vec(PR, row):
        Z = PR["Z"][row]
        P = CB.loo_prior(PR["Z"], PR["S"], PR["C"], row, MIN_CONTRIB)
        out = {}
        for nm, a in (("Z", Z), ("P", P), ("D", Z - P)):
            full = np.full(G, np.nan)
            full[Tm] = a
            out[nm] = full
        return out
    focal = {n: vec(PR, r) for n, PR in Zs.items()}
    matched = {n: [vec(PR, s) for s in mo] for n, PR in Zs.items()}
    flat = {(n, "F", k): v for n, d in focal.items() for k, v in d.items()}
    flat.update({(n, f"M{i}", k): v for n, ms in matched.items() for i, d in enumerate(ms) for k, v in d.items()})
    pre = KP.score_digest(flat)
    # ---- truth opened here
    if st.get("blk") is None:
        seed = int(np.random.SeedSequence([L.BOOT_SEED, L.BLOCKS.index(cfg["block"])]).generate_state(1)[0])
        units, scheme = L.load_modules(S, genes)
        st.update(blk=L.BlockAxis(genes, B, seed, units, scheme), scheme=scheme, seed=seed)
    ep = L.build_endpoint(system, tf, st["blk"], False)
    if KP.score_digest(flat) != pre:
        raise SystemExit("REFUSING: scores changed across truth join")
    pos = {g: i for i, g in enumerate(genes)}
    sel = np.asarray([pos.get(g, -1) for g in ep["genes"]])
    onep = lambda v: np.where(sel >= 0, np.asarray(v)[np.maximum(sel, 0)], np.nan)      # noqa: E731
    a_, b_ = onep(X0 := ds_matrix(raw, uni, tuple(range(0, 9)))[r]), np.asarray(ep["V"][L.GB], float)
    fa, fb = np.isfinite(a_), np.isfinite(b_)
    d_gb = float(np.abs(a_[fa & fb] - b_[fa & fb]).max()) if (fa & fb).any() else np.nan
    if not (np.array_equal(fa, fb) and d_gb <= PARITY_TOL):
        raise L.Blocked(f"DS_0:8_VS_PRODUCTION_GB_FAILED finiteness_equal={np.array_equal(fa, fb)} max|diff|={d_gb}")
    U = ep["U"]
    counts = ep["blk"].counts[:B]
    rows = []
    for name in DSETS:
        F = {k: onep(v) for k, v in focal[name].items()}
        M = [{k: onep(v) for k, v in d.items()} for d in matched[name]]
        common = U & np.isfinite(F["D"]) & np.isfinite(F["P"]) & np.isfinite(F["Z"])
        T = KP.targets_of(ep, common)
        for e in EPS:
            y, rws = T[e]
            yy = np.asarray(y, bool)[rws]
            base = dict(system=system, tf=tf, ds=name, shallow=name in SHALLOW, endpoint=e, n=int(rws.sum()), n_pos=int(yy.sum()), n_T=int(Tm.sum()), n_common_U=int(common.sum()),
                        n_lost_vs_U=int(U.sum() - common.sum()), label=LABEL)
            if not (yy.any() and (~yy).any()):
                rows.append(dict(base, status="TOO_FEW"))
                continue
            gate = bool(L.gate_of(ep, y, rws, common))
            ev = {k: CT.eval_rows(F[k], y, rws, ep["gi"], counts) for k in ("D", "P", "Z")}
            inc = np.asarray(ev["Z"][1]["AUROC"], float) - np.asarray(ev["P"][1]["AUROC"], float)
            ilo, ihi, _ = L.pct(inc)
            dlo, dhi, _ = L.pct(ev["D"][1]["AUROC"])
            mD, mI = [], []
            for m in M:
                aD, aP, aZ = (stat_auroc(m[k], y, rws) for k in ("D", "P", "Z"))
                mD.append(aD)
                mI.append(aZ - aP)
            cD, cI = ev["D"][0]["AUROC"], ev["Z"][0]["AUROC"] - ev["P"][0]["AUROC"]
            pD, ppD, nm_ = pool_rank(cD, mD)
            pI, ppI, _ = pool_rank(cI, mI)
            ok = np.isfinite(F["D"]) & np.isfinite(F["P"]) & rws
            from scipy.stats import spearmanr
            rows.append(dict(base, status="OK" if gate else "DESCRIPTIVE_ONLY", gate=gate, AUROC_D=cD, D_lo=dlo, D_hi=dhi, AUROC_P=ev["P"][0]["AUROC"], AUROC_Z=ev["Z"][0]["AUROC"],
                             inc_Z_minus_P=cI, inc_lo=ilo, inc_hi=ihi, AUPRC_D=ev["D"][0]["AUPRC"], prevalence=ev["D"][0]["PREV"], spearman_D_P=float(spearmanr(F["D"][ok], F["P"][ok])[0]),
                             n_matched=nm_, matched_D_median=float(np.nanmedian(mD)), matched_D_q25=float(np.nanpercentile(mD, 25)), matched_D_q75=float(np.nanpercentile(mD, 75)),
                             matched_D_max=float(np.nanmax(mD)), cognate_pct_D=pD, p_pool_D=ppD, matched_inc_median=float(np.nanmedian(mI)), cognate_pct_inc=pI, p_pool_inc=ppI,
                             reading=reading(pD, cD, name in SHALLOW) if gate else "DESCRIPTIVE_ONLY_gate_fail"))
    rc = dict(system=system, tf=tf, n_bank=len(members), n_T7=int(Tm.sum()), n_U=int(U.sum()), matched=[members[i] for i in mo], match_info=minfo, parity_gb_vector=d_gb, scores_sha256=pre,
              shards=recs, bank_sha256=bank["sha256_members"], priors_sha256=L.sha_file(f"{CB.OUT}/{S}_priors.npz"), gb_sha256=ep["meta"]["gb_sha256"], truth_sha256=ep["meta"]["truth_sha256"],
              wall_s=time.time() - t0)
    return pd.DataFrame(rows), rc


def cmd_run(a):
    Path(a.out).mkdir(parents=True, exist_ok=True)
    states, allr, rcs = {}, [], []
    for u in a.units.split(","):
        system, tf = u.split(":")
        try:
            df, rc = run_unit(system, tf, states.setdefault(system, {}), a.B, a.n_match)
        except L.Blocked as e:
            print(f"[D9] {system}/{tf}: NOT_RUN {e}", flush=True)
            rcs.append(dict(system=system, tf=tf, status=f"NOT_RUN: {e}"))
            continue
        allr.append(df)
        rcs.append(dict(rc, status="OK"))
        print(f"[D9] {system}/{tf} done {rc['wall_s']:.0f}s", flush=True)
    if not allr:
        raise SystemExit("REFUSING: no unit ran")
    P = pd.concat(allr, ignore_index=True)
    L.atomic_tsv(P, f"{a.out}/D9_dutil.tsv")
    sch = {s: dict(scheme=v["scheme"], seed=v["seed"], fp=v["blk"].fingerprint()) for s, v in states.items() if v.get("blk") is not None}
    L.atomic_json(f"{a.out}/receipt.json", dict(created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), label=LABEL, script_sha256=L.sha_file(__file__), B=a.B, blocks=sch, units=rcs,
                                                depth_sets=DSETS, shallow=SHALLOW, n_match=a.n_match, resampling="gene-level (no module map) => DESCRIPTIVE",
                                                module_shas={n: L.sha_file(m.__file__) for n, m in sys.modules.items() if n in ("gb_primary", "gb_catbank", "gb_layer_lattice", "gb_lattice_kprofile", "gb_canon_table")}))
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", "{:.3f}".format)
    print(P[P.status.isin(["OK", "DESCRIPTIVE_ONLY"])][["system", "tf", "ds", "endpoint", "n", "AUROC_D", "D_lo", "D_hi", "AUROC_P", "AUROC_Z", "inc_Z_minus_P", "inc_lo", "inc_hi",
                                                           "matched_D_median", "matched_D_max", "cognate_pct_D", "p_pool_D", "cognate_pct_inc", "reading"]].to_string(index=False))


def selftest(_a=None):
    assert pool_rank(0.7, [0.5, 0.6, 0.65])[0] == 1.0 and pool_rank(0.7, [0.5, 0.6, 0.65])[1] == 0.25
    assert pool_rank(0.5, [0.5, 0.5])[0] == 0.5
    assert reading(0.96, 0.55, True) == "SPECIFIC_SHALLOW_D" and reading(0.96, 0.45, True) == "GENERIC_OR_NULL" and reading(0.5, 0.6, True) == "GENERIC_OR_NULL"
    assert reading(0.99, 0.6, False) == "NOT_SHALLOW"
    rng = np.random.default_rng(0)
    # Z = P + D identity through the production prior algebra, and leave-self-out excludes the focal row
    X = rng.normal(size=(30, 200))
    T = np.ones(200, bool)
    PR = CB.sota_prior_arrays(X, np.arange(30), T, 2)
    P5 = CB.loo_prior(PR["Z"], PR["S"], PR["C"], 5, 2)
    ref = np.nanmean(np.delete(PR["Z"], 5, 0), 0)
    assert np.allclose(P5[np.isfinite(P5)], ref[np.isfinite(P5)])
    # depth-set aggregation is the unchanged zcons: ds 0:8 over raw == zcons default
    R = {h: np.abs(rng.standard_t(3, size=(40, 3, 31))) for h in GP.ROOTS}
    u = np.ones((3, 40), bool)
    Xa = ds_matrix({h: np.transpose(R[h], (1, 0, 2)) for h in GP.ROOTS}, u, tuple(range(9)))
    assert np.allclose(Xa[1], GP.zcons({h: R[h][:, 1, :] for h in GP.ROOTS}), equal_nan=True)
    print("D9 synthetic selftest OK")
    st = {}
    df, rc = run_unit("PANC", "RFX6", st, 20)               # fail-closed gates inside: shard parity, GB-vector parity, stored-prior parity
    assert rc["parity_gb_vector"] <= PARITY_TOL and (df.ds == "0:8").any()
    print(f"D9 real-data production parity OK (PANC/RFX6): gb vector {rc['parity_gb_vector']:.1e}; X_0:8 == shard P|PU; Z,S,C,Pfull == stored T7|P_pairU")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    r = sp.add_parser("run")
    r.add_argument("--units", default="PANC:RFX6,BONEMARROW:RUNX1")
    r.add_argument("--B", type=int, default=999)
    r.add_argument("--n-match", type=int, default=N_MATCH)
    r.add_argument("--out", default=OUT_DEFAULT)
    a = ap.parse_args()
    {"selftest": selftest, "run": cmd_run}[a.cmd](a)


if __name__ == "__main__":
    main()
