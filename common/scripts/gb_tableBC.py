#!/usr/bin/env python
"""TABLE B / TABLE C post-hoc layer (coordinator 2026-10-02; internal review internal_review_log sections 1, 2, 7).  Reads the MERGED output of gb_prior_adjusted.py (table2_cells / table2_decoys / draws parts);
NO re-scoring: every item is a filter / summary of stored quantities (decoy-median draws `DM` are stored, so E is recomputed inside every draw).

TABLE B  (a) a (method, unit) needs >= 50 usable decoys else NO_CALL (matching never widened); ungated / TOO_FEW cells are UNAVAILABLE (p = 1).
         (b) per cell: coverage, class counts, AUROC S/P/M1, raw Delta, decoy median + 5-95%, mid-percentile, E = Delta - median decoy Delta, pool p = (1+#{decoy>=cog})/(J+1)  (AUROC inferential;
             AP / AP-lift / P50 descriptive).
         (c) ONE Holm family over methods x ALL declared focal units (response, occupancy, four-class, regulon; sensitivity panels KLF1_B / BM_GATA1_ACUTE excluded; BONEMARROW duplicate already
             dropped); unavailable = p 1; attainable floor 1/(J+1); NOT_RESOLVABLE when J+1 < 20 H.
         (d) macro: equal TF weight within substrate, equal substrate weight overall, separately per endpoint, on the roster of units evaluable for ALL methods; gene-identity draws + module-block (G/50)
             draws, decoy median recomputed inside every draw; methods compared by paired differences in macro E on that identical roster.
         (e) decoy exchangeability: pseudo-focal mid-percentile among the remaining decoys (KS vs uniform + histogram table; WITHIN a unit this is uniform by construction except for ties, so a
             non-vacuous companion is added: within-unit Spearman of decoy increment vs match distance rank), and matching balance (SMD on detection, log1p mean Ms; cognate vs matched decoys vs all bank).
TABLE C  one row per method: substrate-balanced response prior strength AUROC(P)-.5, macro response E, frozen section-360 F/O (SOTA10_v2) after an exact sorted-gene-ID-hash verification of its
         eligibility masks (`fo_verify`), coverage / exclusions, erythroid two-TF range; source tables for the faceted maps.  No ranking columns.

  python scratch/gb_tableBC.py tableB --in DIR --out OUT
  python scratch/gb_tableBC.py fo_verify --out OUT
  python scratch/gb_tableBC.py tableC --out OUT
  python scratch/gb_tableBC.py selftest
"""
from __future__ import annotations

import os

os.environ.setdefault("OMP_NUM_THREADS", "2")
import argparse
import glob
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = __import__("os").environ["GEOBRIDGE_ROOT"]
for _p in (str(Path(__file__).resolve().parent), f"{R}/scratch"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gb_prior_adjusted as PA  # noqa: E402  (imports only; its scoring code is not re-run)
import gb_ptable as PT  # noqa: E402

METHODS = PA.METHODS
J_MIN = 50
ALPHA = 0.05
SENS_SYSTEMS = {"KLF1_B", "BM_GATA1_ACUTE"}                                    # alternative / acute panels: sensitivity rows, never in the family or macros
SUB = {"BM_CHIPATLAS": "BONEMARROW", "GATA1": "ERYTHROID", "KLF1": "ERYTHROID", "PANC": "PANC", "K562_EMP_ALLNULL": "K562"}
LAT_T = {"resp": "response", "occ_prim": "occupancy", "AB_prim": "four-class", "CD_prim": "four-class", "AC_prim": "four-class", "BD_prim": "four-class"}
FO_DIR = f"{R}/results/real_uniform/layer_table/SOTA10_v2"
LAYER_REF = {"base": f"{R}/results/real_uniform/layer_lattice_sota_final/endpoints.tsv", "extra": f"{R}/results/real_uniform/layer_lattice_extra_sota/{{system}}/endpoints.tsv"}


# --------------------------------------------------------------------------------------------------------------------- pure numerics (selftested)
def holm(p, alpha=ALPHA):
    """Holm step-down adjusted p-values (monotone) and rejection flags"""
    p = np.asarray(p, float)
    H = len(p)
    o = np.argsort(p, kind="stable")
    adj = np.empty(H)
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (H - r) * p[i]))
        adj[i] = run
    return adj, adj <= alpha


def pseudo_midpct(d):
    """each decoy as pseudo-focal: mid-percentile = (#{others < d} + .5 #{others == d}) / (J-1) among the REMAINING decoys"""
    d = np.asarray(d, float)
    out = np.empty(len(d))
    for i in range(len(d)):
        o = np.delete(d, i)
        out[i] = ((o < d[i]).sum() + 0.5 * (o == d[i]).sum()) / len(o)
    return out


def substrate_of(kind, system):
    if system in SENS_SYSTEMS:
        return None
    return SUB.get(system) if kind == "lattice" else system


def macro_mean(units_by_sub, fn):
    """equal TF weight within substrate, equal substrate weight overall: mean over substrates of the mean of fn(unit) over that substrate's units"""
    return np.mean([np.mean([fn(u) for u in us], 0) for us in units_by_sub.values()], 0)


# --------------------------------------------------------------------------------------------------------------------- loading
def load(inp):
    c = pd.read_csv(f"{inp}/table2_cells.tsv", sep="\t", dtype={"tf": str})
    d = pd.read_csv(f"{inp}/table2_decoys.tsv", sep="\t", dtype={"tf": str})
    dr = {}
    for f in sorted(glob.glob(f"{inp}/part_*.draws.npz")):
        with np.load(f) as z:
            for k in z.files:
                dr[k] = z[k]
    return c, d, dr


def cell_flags(c):
    """Table B status per cell: OK | NO_CALL (decoys < 50 / not run) | UNAVAILABLE (ungated, TOO_FEW)"""
    c = c.copy()
    c["sensitivity"] = c.system.isin(SENS_SYSTEMS)
    nd = c.get("n_decoy", pd.Series(0, index=c.index)).fillna(0)
    st = np.where(c.status != "OK", "UNAVAILABLE", np.where((nd >= J_MIN) & c.decoy_status.fillna("").str.startswith("OK"), "OK", "NO_CALL"))
    c["B_status"] = st
    c["endpoint_class"] = np.where(c.kind == "regulon", "regulon", c.target.map(LAT_T))
    c["coverage"] = c.n_rows / c.n_M1
    c["J"] = nd.astype(int)
    p = c.p_pool_AUROC.where(c.B_status == "OK", 1.0).fillna(1.0)
    c["p_B"] = p
    return c


def tableB_cells(c):
    keep = ["kind", "system", "tf", "target", "endpoint_class", "method", "sensitivity", "B_status", "n_M1", "n_rows", "coverage", "n_pos", "n_neg", "J", "S_AUROC", "P_AUROC", "M1_AUROC", "d_AUROC",
            "dec_med_AUROC", "dec_p05_AUROC", "dec_p95_AUROC", "cog_pct_AUROC", "calD_AUROC", "p_pool_AUROC", "p_B",
            "S_AUPRC", "P_AUPRC", "M1_AUPRC", "d_AUPRC", "calD_AUPRC", "S_APLIFT", "d_APLIFT", "calD_APLIFT", "S_P50", "d_P50", "calD_P50"]
    t = c[[k for k in keep if k in c]].copy()
    off = t.B_status != "OK"
    for k in ("dec_med_AUROC", "dec_p05_AUROC", "dec_p95_AUROC", "cog_pct_AUROC", "calD_AUROC", "p_pool_AUROC"):
        if k in t:
            t.loc[off, k] = np.nan
    return t


def holm_family(c):
    f = c[~c.sensitivity].copy()
    H = len(f)
    adj, rej = holm(f.p_B.to_numpy())
    f["holm_p"], f["holm_reject"] = adj, rej
    J = int(f[f.B_status == "OK"].J.min()) if (f.B_status == "OK").any() else 0
    floor = 1.0 / (J + 1) if J else np.nan
    summ = dict(H=H, n_OK=int((f.B_status == "OK").sum()), n_NO_CALL=int((f.B_status == "NO_CALL").sum()), n_UNAVAILABLE=int((f.B_status == "UNAVAILABLE").sum()), J_min=J, p_floor=floor,
                first_holm_threshold=ALPHA / H, floor_le_first_threshold=bool(floor <= ALPHA / H) if J else False, required_J_plus_1=20 * H,
                NOT_RESOLVABLE=bool((J + 1) < 20 * H), n_holm_reject=int(rej.sum()), n_raw_p_le_05=int((f[f.B_status == "OK"].p_pool_AUROC <= 0.05).sum()),
                by_class=f.groupby("endpoint_class").size().to_dict())
    return f, summ


# --------------------------------------------------------------------------------------------------------------------- macro (d)
def macro_tables(c, dr):
    """per endpoint: roster = units OK for every method; macro quantities with gene / module draws, paired E differences"""
    ok = c[(~c.sensitivity) & (c.B_status == "OK")]
    rows, pairs, subst = [], [], []
    for target, tc in ok.groupby("target"):
        cnt = tc.groupby(["kind", "system", "tf"]).method.nunique()
        full = set(cnt[cnt == len(METHODS)].index)
        ep = "regulon" if tc.kind.iloc[0] == "regulon" else LAT_T[target]
        by_sub = {}
        for k, s, t in sorted(full):
            sb = substrate_of(k, s)
            if sb is not None:
                by_sub.setdefault(sb, []).append((k, s, t))
        if not by_sub:
            continue
        sched = {}                                                            # (sch, quantity, method) -> macro draws

        def u_draw(u, m, q, sch):
            k, s, t = u
            g = lambda nm: dr[f"{k}|{s}|{t}|{target}|{m}|AUROC|{nm}|{sch}"].astype(float)
            return {"E": lambda: g("M1") - g("P") - g("DM"), "d": lambda: g("M1") - g("P"), "P": lambda: g("P") - 0.5, "S": lambda: g("S")}[q]()

        def u_pt(u, m, q):
            k, s, t = u
            r = tc[(tc.kind == k) & (tc.system == s) & (tc.tf == t) & (tc.method == m)].iloc[0]
            return {"E": r.calD_AUROC, "d": r.d_AUROC, "P": r.P_AUROC - 0.5, "S": r.S_AUROC}[q]
        for m in METHODS:
            for q in ("E", "d", "P", "S"):
                pt = macro_mean(by_sub, lambda u: u_pt(u, m, q))
                rec = dict(endpoint=ep, target=target, method=m, quantity=q, n_substrates=len(by_sub), n_units=sum(len(v) for v in by_sub.values()), point=float(pt))
                for sch, tag in (("g", "gene"), ("m", "module")):
                    arr = macro_mean(by_sub, lambda u: u_draw(u, m, q, sch))
                    sched[(sch, q, m)] = arr
                    rec[f"{tag}_lo"], rec[f"{tag}_hi"] = PT.pct_ci(arr)
                rows.append(rec)
        for q in ("E",):
            for i, a in enumerate(METHODS):
                for b in METHODS[i + 1:]:
                    rec = dict(endpoint=ep, target=target, quantity=q, method_a=a, method_b=b, n_units=sum(len(v) for v in by_sub.values()),
                               point=float(next(r["point"] for r in rows[::-1] if r["target"] == target and r["method"] == a and r["quantity"] == q) -
                                           next(r["point"] for r in rows[::-1] if r["target"] == target and r["method"] == b and r["quantity"] == q)))
                    for sch, tag in (("g", "gene"), ("m", "module")):
                        dd = sched[(sch, q, a)] - sched[(sch, q, b)]
                        rec[f"{tag}_lo"], rec[f"{tag}_hi"] = PT.pct_ci(dd)
                        rec[f"p_{tag}"] = PT.boot_p(dd)
                    pairs.append(rec)
        for sb, us in by_sub.items():                                          # substrate-level values (source of the faceted map)
            for m in METHODS:
                rec = dict(endpoint=ep, target=target, substrate=sb, method=m, n_units=len(us), units=";".join(f"{s}:{t}" for _k, s, t in us))
                for q in ("E", "P"):
                    rec[f"{q}_point"] = float(np.mean([u_pt(u, m, q) for u in us]))
                    for sch, tag in (("g", "gene"), ("m", "module")):
                        arr = np.mean([u_draw(u, m, q, sch) for u in us], 0)
                        rec[f"{q}_{tag}_lo"], rec[f"{q}_{tag}_hi"] = PT.pct_ci(arr)
                subst.append(rec)
    return pd.DataFrame(rows), pd.DataFrame(pairs), pd.DataFrame(subst)


# --------------------------------------------------------------------------------------------------------------------- exchangeability (e)
def exchangeability(c, d):
    ok = c[(~c.sensitivity) & (c.B_status == "OK")][["kind", "system", "tf", "target", "method"]]
    d = d.merge(ok, on=["kind", "system", "tf", "target", "method"])
    pc, trend = [], []
    from scipy.stats import spearmanr
    for (k, s, t, tg, m), g in d.groupby(["kind", "system", "tf", "target", "method"]):
        v = g.d_AUROC.to_numpy(float)
        if np.isfinite(v).sum() < J_MIN:
            continue
        g = g[np.isfinite(v)]
        pm = pseudo_midpct(g.d_AUROC.to_numpy(float))
        pc.append(pd.DataFrame(dict(method=m, endpoint_class="regulon" if k == "regulon" else LAT_T[tg], pct=pm)))
        rho = spearmanr(g.match_rank, g.d_AUROC)[0]
        trend.append(dict(kind=k, system=s, tf=t, target=tg, method=m, rho_matchrank_delta=float(rho), n=len(g)))
    pc = pd.concat(pc, ignore_index=True)
    from scipy.stats import kstest, wilcoxon
    ks = []
    for nm, g in [("ALL", pc)] + [(f"method={m}", x) for m, x in pc.groupby("method")] + [(f"class={e}", x) for e, x in pc.groupby("endpoint_class")]:
        r = kstest(g.pct, "uniform")
        ks.append(dict(subset=nm, n_pseudo_focal=len(g), ks_stat=float(r.statistic), ks_p=float(r.pvalue), mean_pct=float(g.pct.mean())))
    hist = pd.concat([pd.DataFrame(dict(subset="ALL", **dict(zip(["bin_lo", "count"], [np.histogram(pc.pct, bins=10, range=(0, 1))[1][:-1], np.histogram(pc.pct, bins=10, range=(0, 1))[0]]))))] +
                     [pd.DataFrame(dict(subset=f"method={m}", **dict(zip(["bin_lo", "count"], [np.histogram(x.pct, bins=10, range=(0, 1))[1][:-1], np.histogram(x.pct, bins=10, range=(0, 1))[0]])))) for m, x in pc.groupby("method")],
                     ignore_index=True)
    tr = pd.DataFrame(trend)
    tsum = []
    for m, g in tr.groupby("method"):
        r = g.rho_matchrank_delta.dropna()
        tsum.append(dict(method=m, n_units=len(r), mean_rho=float(r.mean()), frac_rho_gt0=float((r > 0).mean()), wilcoxon_p=float(wilcoxon(r).pvalue) if len(r) > 5 and (r != 0).any() else np.nan))
    return pd.DataFrame(ks), hist, tr, pd.DataFrame(tsum)


def balance_with_decoys(c, d):
    import gb_layer_lattice as L
    import gb_catbank as CB
    import gb_lattice_extra as LE
    LE.register()
    ok = c[(~c.sensitivity) & (c.B_status == "OK")][["kind", "system", "tf", "target", "method"]]
    d = d.merge(ok, on=["kind", "system", "tf", "target", "method"])
    out, ctx = [], {}
    for (k, s, t, tg, m), g in d.groupby(["kind", "system", "tf", "target", "method"]):
        axis = L.SYSTEMS[s]["axis"] if k == "lattice" else s
        if axis not in ctx:
            bj = json.load(open(f"{CB.OUT}/{axis}_bank.json"))
            stub = type("B", (), {"members": bj["members"]})
            F = PA.match_ctx(axis, stub)["F"]
            ctx[axis] = (F, {mm: i for i, mm in enumerate(bj["members"])}, np.array([x.upper() for x in bj["members"]]))
        F, ix, mu = ctx[axis]
        r = int(np.flatnonzero(mu == t.upper())[0])
        di = np.array([ix[x] for x in g.decoy])
        allo = np.delete(np.arange(len(F)), r)
        for j, nm in enumerate(("detection", "log1p_mean_Ms")):
            out.append(dict(kind=k, system=s, tf=t, target=tg, method=m, feature=nm, smd_matched=float(F[di, j].mean() - F[r, j]), smd_allbank=float(F[allo, j].mean() - F[r, j]), n=len(di)))
    b = pd.DataFrame(out)
    summ = b.groupby(["method", "feature"]).agg(units=("smd_matched", "size"), mean_smd_matched=("smd_matched", "mean"), mean_abs_smd_matched=("smd_matched", lambda x: float(np.abs(x).mean())),
                                                max_abs_smd_matched=("smd_matched", lambda x: float(np.abs(x).max())), mean_abs_smd_allbank=("smd_allbank", lambda x: float(np.abs(x).mean()))).reset_index()
    return b, summ


def cmd_tableB(a):
    os.makedirs(a.out, exist_ok=True)
    c, d, dr = load(a.inp)
    c = cell_flags(c)
    PT.atomic_tsv(tableB_cells(c), f"{a.out}/tableB_cells.tsv")
    f, summ = holm_family(c)
    PT.atomic_tsv(f[["kind", "system", "tf", "target", "endpoint_class", "method", "B_status", "J", "p_B", "holm_p", "holm_reject"]], f"{a.out}/tableB_holm_family.tsv")
    PT.atomic_json(f"{a.out}/tableB_holm_summary.json", summ)
    cls = f[f.B_status == "OK"].groupby(["method", "endpoint_class"]).agg(cells=("tf", "size"), raw_p_le_05=("p_B", lambda x: int((x <= 0.05).sum()))).reset_index()
    cnt = c[~c.sensitivity].groupby(["method", "endpoint_class", "B_status"]).size().unstack(fill_value=0).reset_index()
    PT.atomic_tsv(cnt, f"{a.out}/tableB_class_counts.tsv")
    PT.atomic_tsv(cls, f"{a.out}/tableB_raw_p_counts.tsv")
    mac, pairs, subst = macro_tables(c, dr)
    PT.atomic_tsv(mac, f"{a.out}/tableB_macro.tsv")
    PT.atomic_tsv(pairs, f"{a.out}/tableB_macro_paired_E.tsv")
    PT.atomic_tsv(subst, f"{a.out}/tableB_substrate_values.tsv")
    ks, hist, tr, tsum = exchangeability(c, d)
    PT.atomic_tsv(ks, f"{a.out}/tableB_exch_ks.tsv")
    PT.atomic_tsv(hist, f"{a.out}/tableB_exch_hist.tsv")
    PT.atomic_tsv(tr, f"{a.out}/tableB_exch_trend_units.tsv")
    PT.atomic_tsv(tsum, f"{a.out}/tableB_exch_trend_summary.tsv")
    b, bs = balance_with_decoys(c, d)
    PT.atomic_tsv(b, f"{a.out}/tableB_balance_units.tsv")
    PT.atomic_tsv(bs, f"{a.out}/tableB_balance_summary.tsv")
    print("HOLM", json.dumps(summ, default=str))
    print("KS\n", ks.round(4).to_string(index=False))
    print("TREND\n", tsum.round(3).to_string(index=False))
    print("BALANCE\n", bs.round(3).to_string(index=False))


# --------------------------------------------------------------------------------------------------------------------- F/O mask verification (section 7)
def gene_hash(genes):
    return hashlib.sha256("\n".join(sorted(map(str, genes))).encode()).hexdigest()


def cmd_fo_verify(a):
    """rebuild the upstream ALL6+SOTA masks, hash the sorted gene IDs of every eligible F/O component, check (i) all ten methods are finite on exactly those genes, (ii) n / n_pos / n_neg and
    every method's AUROC recomputed on the hashed rows equal the frozen layer-lattice rows that section 360 consumed"""
    import gb_layer_lattice as L
    import gb_lattice_extra as LE
    from sklearn.metrics import roc_auc_score
    LE.register()
    ro = pd.read_csv(f"{FO_DIR}/roster_v2.tsv", sep="\t", dtype={"tf": str})
    ro = ro[ro.eligible & ro.system.isin(list(SUB))]
    M10 = ["GeoBRIDGE", "Velorama", "RegVelo |J|", "GENIE3", "GRNBoost2", "|Pearson|", "RegDiffusion", "DeepSEM", "PIDC", "FastTENET"]
    ref = {}
    out = []
    for (system, tf), g in ro.groupby(["system", "tf"]):
        path = LAYER_REF["base"] if system in ("GATA1", "KLF1", "PANC", "BONEMARROW") else LAYER_REF["extra"].format(system=system)
        if path not in ref:
            ref[path] = pd.read_csv(path, sep="\t", dtype={"tf": str})
        E = ref[path]
        E = E[(E.system == system) & (E.tf == tf) & (E.scope == "ALL6+SOTA")]
        dr_, _ = PT.lattice_draws(system, 10, sizes=(50,))
        ep = L.build_endpoint(system, tf, dr_.gene, False)
        V, U, genes = ep["V"], np.asarray(ep["U"], bool), np.asarray(ep["genes"]).astype(str)
        fin = {m: np.isfinite(np.asarray(V[m], float)) for m in M10}
        common = U & np.all([fin[m] for m in M10], 0)
        T = L.targets(ep, common)
        for comp in g.component:
            tg = {"resp": "resp", "occ": "occ_prim", "AB": "AB_prim", "CD": "CD_prim", "AC": "AC_prim", "BD": "BD_prim"}[comp]
            y, rows = T[tg]
            y = np.asarray(y, bool)
            h = gene_hash(genes[rows])
            per_method_same = all(gene_hash(genes[rows & fin[m]]) == h for m in M10)          # every method finite on exactly the hashed rows
            e = E[E.target == tg]
            for m in M10:
                r = e[e.method == m]
                if not len(r):
                    out.append(dict(system=system, tf=tf, component=comp, method=m, hash=h, status="MISSING_UPSTREAM_ROW"))
                    continue
                r = r.iloc[0]
                auc = roc_auc_score(y[rows], np.asarray(V[m], float)[rows])
                cnt_ok = (int(rows.sum()), int(y[rows].sum()), int((~y[rows]).sum())) == (int(r.n), int(r.n_pos), int(r.n_neg))
                out.append(dict(system=system, tf=tf, component=comp, method=m, hash=h, all_methods_finite_on_hashed_rows=per_method_same, counts_match=cnt_ok,
                                auroc_recomputed=float(auc), auroc_frozen=float(r.AUROC), auroc_absdiff=abs(auc - float(r.AUROC)), n=int(rows.sum()),
                                status="PASS" if (per_method_same and cnt_ok and abs(auc - float(r.AUROC)) < 1e-6) else "MISMATCH"))
        print(f"[fo_verify] {system}/{tf}: {len(g)} components", flush=True)
    t = pd.DataFrame(out)
    os.makedirs(a.out, exist_ok=True)
    PT.atomic_tsv(t, f"{a.out}/tableC_FO_mask_verification.tsv")
    print(f"[fo_verify] rows {len(t)}; PASS {int((t.status == 'PASS').sum())}; MISMATCH {int((t.status == 'MISMATCH').sum())}; MISSING {int((t.status == 'MISSING_UPSTREAM_ROW').sum())}")


# --------------------------------------------------------------------------------------------------------------------- Table C
def cmd_tableC(a):
    mac = pd.read_csv(f"{a.out}/tableB_macro.tsv", sep="\t")
    sub = pd.read_csv(f"{a.out}/tableB_substrate_values.tsv", sep="\t")
    cells = pd.read_csv(f"{a.out}/tableB_cells.tsv", sep="\t", dtype={"tf": str})
    resp = mac[mac.target == "resp"]
    ph = pd.read_csv(f"{FO_DIR}/phenotype_v2.tsv", sep="\t")
    ov = pd.read_csv(f"{FO_DIR}/overall_v2.tsv", sep="\t")
    ver = f"{a.out}/tableC_FO_mask_verification.tsv"
    if os.path.exists(ver):
        v = pd.read_csv(ver, sep="\t")
        vst = "VERIFIED_ALL_PASS" if (v.status == "PASS").all() else f"MISMATCH n={int((v.status != 'PASS').sum())} of {len(v)}"
    else:
        vst = "NOT_RUN"
    rc = cells[(cells.target == "resp") & (~cells.sensitivity)]
    rows = []
    ru = resp[resp.quantity == "E"]
    n_sub, n_units = (int(ru.n_substrates.iloc[0]), int(ru.n_units.iloc[0])) if len(ru) else (0, 0)
    decl = rc[rc.method == "GeoBRIDGE"]
    excl = ";".join(f"{r.system}:{r.tf}({r.B_status})" for r in decl.itertuples() if r.B_status != "OK") or "none"
    for m in METHODS:
        r = {"method": m}
        for q, nm in (("P", "prior_strength_AUROC_minus_.5"), ("E", "calibrated_E_response")):
            x = resp[(resp.method == m) & (resp.quantity == q)]
            if len(x):
                x = x.iloc[0]
                r.update({nm: x.point, f"{nm}_gene_lo": x.gene_lo, f"{nm}_gene_hi": x.gene_hi, f"{nm}_module_lo": x.module_lo, f"{nm}_module_hi": x.module_hi})
        r.update(response_substrates=n_sub, response_units=n_units, declared_response_units=decl.groupby(["system", "tf"]).ngroups, excluded_response_units=excl,
                 sensitivity_panels_excluded=";".join(sorted(SENS_SYSTEMS)))
        o = ov[ov.method == m]
        if len(o):
            o = o.iloc[0]
            r.update(F_overall=o.F, F_lo=o.F_lo, F_hi=o.F_hi, O_overall=o.O, O_lo=o.O_lo, O_hi=o.O_hi, FO_substrates=o.substrates)
        for sb in ("BONEMARROW", "ERYTHROID", "PANC", "K562"):
            p = ph[(ph.group == f"SUBSTRATE_MAIN:{sb}") & (ph.method == m)]
            if len(p):
                p = p.iloc[0]
                r.update({f"F_{sb}": p.F, f"F_{sb}_lo": p.F_lo, f"F_{sb}_hi": p.F_hi, f"O_{sb}": p.O, f"O_{sb}_lo": p.O_lo, f"O_{sb}_hi": p.O_hi, f"n_roster_{sb}": p.n_roster,
                          f"incomplete_{sb}": p.incomplete_detail})
        g1, k1 = (ph[(ph.group == s) & (ph.method == m)] for s in ("GATA1", "KLF1"))
        if len(g1) and len(k1):
            r.update(F_ERYTHROID_range=f"{min(g1.F.iloc[0], k1.F.iloc[0]):.3f}..{max(g1.F.iloc[0], k1.F.iloc[0]):.3f} (two-TF range GATA1,KLF1; bootstrap interval not meaningful)",
                     O_ERYTHROID_range=f"{min(g1.O.iloc[0], k1.O.iloc[0]):.3f}..{max(g1.O.iloc[0], k1.O.iloc[0]):.3f}")
        r["FO_mask_hash_verification"] = vst
        rows.append(r)
    PT.atomic_tsv(pd.DataFrame(rows), f"{a.out}/tableC.tsv")
    mp = sub[sub.target == "resp"][["substrate", "method", "n_units", "units", "P_point", "P_gene_lo", "P_gene_hi", "E_point", "E_gene_lo", "E_gene_hi", "E_module_lo", "E_module_hi"]].rename(
        columns={"P_point": "x_prior_strength_AUROC_minus_.5", "E_point": "y_calibrated_E"})
    mp["x_prior_strength_AUROC_minus_.5"] = mp["x_prior_strength_AUROC_minus_.5"]
    PT.atomic_tsv(mp, f"{a.out}/tableC_map_response_source.tsv")
    fo = ph[ph.group.str.startswith("SUBSTRATE_MAIN:")][["group", "method", "n_roster", "roster", "incomplete_detail", "F", "F_lo", "F_hi", "O", "O_lo", "O_hi"]].rename(columns={"O": "x_O", "F": "y_F"})
    PT.atomic_tsv(fo, f"{a.out}/tableC_map_FO_source.tsv")
    print(pd.DataFrame(rows)[[c for c in ("method", "prior_strength_AUROC_minus_.5", "calibrated_E_response", "F_overall", "O_overall", "FO_mask_hash_verification") if c in pd.DataFrame(rows)]].round(3).to_string(index=False))


# --------------------------------------------------------------------------------------------------------------------- selftest
def selftest(_a=None):
    adj, rej = holm([0.01, 0.04, 0.03, 0.5])
    assert np.allclose(adj, [0.04, 0.09, 0.09, 0.5]) and list(rej) == [True, False, False, False], (adj, rej)
    assert holm([1.0, 1.0])[0].tolist() == [1.0, 1.0]
    pm = pseudo_midpct([0.1, 0.2, 0.3, 0.4])
    assert np.allclose(pm, [0, 1 / 3, 2 / 3, 1]) and abs(pseudo_midpct([1, 1, 1])[0] - 0.5) < 1e-12        # no ties -> exactly uniform on {0..1}; all ties -> .5
    ub = {"A": [1, 2], "B": [3]}
    assert np.allclose(macro_mean(ub, lambda u: np.array([float(u)])), [(1.5 + 3) / 2])                    # equal TF weight within substrate, equal substrate weight overall
    assert gene_hash(["b", "a"]) == gene_hash(["a", "b"]) and gene_hash(["a"]) != gene_hash(["a", "b"])
    c = pd.DataFrame(dict(kind="lattice", system=["PANC"] * 4 + ["KLF1_B"], tf="T", target="resp", method="GeoBRIDGE", status=["OK", "OK", "OK", "DESCRIPTIVE_ONLY", "OK"],
                          decoy_status=["OK", "OK", "OK_FEWER_THAN_50", "NOT_RUN_UNGATED", "OK"], n_decoy=[50, 50, 30, 0, 50], n_rows=1, n_M1=2, p_pool_AUROC=[0.02, 0.5, 0.01, np.nan, 0.01]))
    f = cell_flags(c)
    assert f.B_status.tolist() == ["OK", "OK", "NO_CALL", "UNAVAILABLE", "OK"] and f.p_B.tolist() == [0.02, 0.5, 1.0, 1.0, 0.01]      # 30 decoys -> NO_CALL, ungated -> p 1
    fam, s = holm_family(f)
    assert s["H"] == 4 and s["NOT_RESOLVABLE"] and abs(s["p_floor"] - 1 / 51) < 1e-12 and s["n_holm_reject"] == 0 and s["required_J_plus_1"] == 80
    print("[tableBC] selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("tableB")
    p.add_argument("--in", dest="inp", required=True)
    p.add_argument("--out", required=True)
    p = sp.add_parser("fo_verify")
    p.add_argument("--out", required=True)
    p = sp.add_parser("tableC")
    p.add_argument("--out", required=True)
    sp.add_parser("selftest")
    a = ap.parse_args()
    {"tableB": cmd_tableB, "fo_verify": cmd_fo_verify, "tableC": cmd_tableC, "selftest": selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
