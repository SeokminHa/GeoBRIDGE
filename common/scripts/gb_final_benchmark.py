#!/usr/bin/env python
"""CANONICAL FINAL six-method benchmark (internal directive 2026-09-27 items 7-8).  Evaluation only; frozen endpoints (RU.trace_endpoints),
GB = PRIMARY (zcons), comparators = latest v2 files (same loaders as gb_interim_fig1).  A missing method is MISSING, never imputed.
Two explicitly different evaluations, never mixed:
  PAIRWISE   rows = eval ∩ finite(GB) ∩ finite(method)          (one universe per GB-vs-method pair)
  COMMON     rows = eval ∩ finite(every AVAILABLE method)        (one universe per endpoint; "common universe of available methods")
Effect = ΔAUROC = GB − comparator, paired class-stratified bootstrap (B draws; the same resampled genes for both methods).
System level: endpoint -> TF mean -> system mean per family; its CI uses the same per-endpoint draws (conditional on the TF set).
Across systems: median Δ, #systems Δ > 0, #systems with CI > 0, range.  TFs are never treated as independent systems.
  python scratch/gb_final_benchmark.py shard SYSTEM   -> results/real_uniform/final_benchmark/shard_SYSTEM.pkl  (parallel per system)
  python scratch/gb_final_benchmark.py merge          -> results/real_uniform/final_benchmark/*.tsv (every shard present)
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score as APS

sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
import gb_interim_fig1 as IF
import gb_killtests as KT  # noqa: F401  (keeps RAWH)
import gb_primary as GP
import gb_real_uniform as RU

B = 500
METHODS = ["GeoBRIDGE", "|Pearson|", "GENIE3", "GRNBoost2", "RegVelo |J|", "RegVelo +J", "RegVelo -J", "Velorama"]
COMPS = METHODS[1:]
REFS = ["REF -detection", "REF -pair_support", "REF +dynamics"]   # truth-free, regulator-INDEPENDENT observability rankings (reference only)


def obs_cov(system):
    """gene -> (detection, pair_support, dynamics) on control cells, same definitions as gb_fig3_mech.covariates (no source column)"""
    D = RU.canonical_inputs(system)
    S = D["S"]
    X = np.asarray(S["X"], np.float64)
    dyn, sup = [], []
    for root in RU.ROOTS:
        I, J = (np.asarray(a, int) for a in S["pairs"][root])
        dX = X[J] - X[I]
        dyn.append(np.std(dX, 0) / np.std(X, 0))
        sup.append(np.mean(dX != 0, 0))
    g = list(D["fixture_genes"])
    return {k: dict(zip(g, v)) for k, v in (("REF -detection", -np.mean(X > 0, 0)), ("REF -pair_support", -np.mean(sup, 0)),
                                            ("REF +dynamics", np.mean(dyn, 0)))}


def auc(y, s):
    r = rankdata(s)
    n1 = y.sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * (len(y) - n1))


def draws(y, rng):
    p, n = np.flatnonzero(y), np.flatnonzero(~y)
    return [np.r_[rng.choice(p, len(p)), rng.choice(n, len(n))] for _ in range(B)]


OUTD = RU.OUT / "final_benchmark"
SEED = {"PANC": 1, "GATA1": 2, "BONEMARROW": 3, "DENTATEGYRUS": 4, "GASTRULATION": 5, "K562": 6, "ZEBRAFISH": 7}


def shard(system):
    import pickle
    OUTD.mkdir(parents=True, exist_ok=True)
    system = system.upper()
    rng = np.random.default_rng([20260927, SEED[system]])
    recs, dd, cov = [], [], []
    for system in [system]:
        d = RU.system_dir(system)
        L = {"|Pearson|": IF.loader(d / "pearson.npz", "score"), "GENIE3": IF.loader(d / "genie3_v2.npz", "score"),
             "GRNBoost2": IF.loader(d / "grnboost2_v2.npz", "score"), "RegVelo": IF.loader(d / "regvelo_v2_full.npz", "score_signed"),
             "Velorama": IF.loader(d / "velorama_v2_native.npz", "score")}
        OC = obs_cov(system)
        print(f"[final] {system}: available {[m for m, v in L.items() if v is not None]}", flush=True)
        for name, family, _s, src, eps, n, npos in RU.trace_endpoints(systems=[system]):
            src = src.upper()
            E = eps["U"]
            V = {"GeoBRIDGE": GP.zcons(E.RAWH)}
            for m, l in L.items():
                v = IF.row(l, src, E.obs_genes)
                if v is None:
                    continue
                if m == "RegVelo":
                    V["RegVelo |J|"], V["RegVelo +J"], V["RegVelo -J"] = np.abs(v), v, -v
                else:
                    V[m] = v
            for r in REFS:
                V[r] = np.array([OC[r].get(g, np.nan) for g in E.obs_genes], float)
            ev = E.eval_mask.copy()
            y = np.zeros(len(ev), bool)
            y[ev] = np.asarray(E.y, bool)
            fam = "regulon" if family == "F1" else "functional"
            base = dict(system=system, fam=fam, endpoint=name, source=src)
            cov.append(dict(**base, n_eval=int(ev.sum()), **{f"cov|{m}": (float(np.isfinite(V[m][ev]).mean()) if m in V else np.nan)
                                                               for m in METHODS + REFS}))
            six = ev.copy()
            for m_, v in V.items():
                if m_ not in REFS:
                    six &= np.isfinite(v)
            scopes = [("COMMON_AVAILABLE", m, six) for m in COMPS + REFS if m in V]
            scopes += [("PAIRWISE", m, ev & np.isfinite(V["GeoBRIDGE"]) & np.isfinite(V[m])) for m in COMPS + REFS if m in V]
            for m in COMPS:
                if m not in V:
                    recs.append(dict(**base, scope="BOTH", comparator=m, status="MISSING"))
            for scope, m, rows in scopes:
                yy = y[rows]
                if yy.sum() < 5 or (~yy).sum() < 5:
                    recs.append(dict(**base, scope=scope, comparator=m, status="TOO_FEW", n=int(rows.sum()), n_pos=int(yy.sum())))
                    continue
                g, c = V["GeoBRIDGE"][rows], V[m][rows]
                dr = np.array([auc(yy[i], g[i]) - auc(yy[i], c[i]) for i in draws(yy, rng)])
                ag, ac = auc(yy, g), auc(yy, c)
                recs.append(dict(**base, scope=scope, comparator=m, status="OK", n=int(rows.sum()), n_pos=int(yy.sum()),
                                 AUROC_GB=ag, AUROC_comp=ac, delta=ag - ac, lo=np.percentile(dr, 2.5), hi=np.percentile(dr, 97.5),
                                 APlift_GB=APS(yy, g) / yy.mean(), APlift_comp=APS(yy, c) / yy.mean()))
                dd.append(dict(**base, scope=scope, comparator=m, draws=dr, delta=ag - ac))
    pickle.dump(dict(recs=recs, dd=dd, cov=cov), open(OUTD / f"shard_{system}.pkl", "wb"))
    print(f"[final] shard {system}: {len(recs)} rows", flush=True)


def merge():
    import glob
    import pickle
    out = OUTD
    recs, dd, cov = [], [], []
    for f in sorted(glob.glob(str(OUTD / "shard_*.pkl"))):
        z = pickle.load(open(f, "rb"))
        recs += z["recs"]; dd += z["dd"]; cov += z["cov"]
    R_ = pd.DataFrame(recs)
    C = pd.DataFrame(cov)
    R_.to_csv(out / "endpoints.tsv", sep="\t", index=False)
    C.to_csv(out / "coverage.tsv", sep="\t", index=False)
    # system level: endpoint -> TF mean -> system mean, CI from the aligned draws
    DD = pd.DataFrame(dd)
    sysrows = []
    for (system, fam, scope, m), g in DD.groupby(["system", "fam", "scope", "comparator"]):
        tf_d = g.groupby("source").delta.mean()
        tf_draw = np.vstack([np.mean(np.vstack(x.draws.to_list()), 0) for _, x in g.groupby("source")])
        sd = tf_draw.mean(0)
        ok = R_[(R_.system == system) & (R_.fam == fam) & (R_.scope == scope) & (R_.comparator == m) & (R_.status == "OK")]
        sysrows.append(dict(system=system, fam=fam, scope=scope, comparator=m, n_tf=len(tf_d), n_endpoints=len(g),
                            AUROC_GB=ok.groupby("source").AUROC_GB.mean().mean(), AUROC_comp=ok.groupby("source").AUROC_comp.mean().mean(),
                            APlift_GB=ok.groupby("source").APlift_GB.mean().mean(), APlift_comp=ok.groupby("source").APlift_comp.mean().mean(),
                            delta=tf_d.mean(), lo=np.percentile(sd, 2.5), hi=np.percentile(sd, 97.5)))
    S = pd.DataFrame(sysrows)
    S.to_csv(out / "system_forest.tsv", sep="\t", index=False)
    X = S.groupby(["fam", "scope", "comparator"]).agg(n_systems=("delta", "size"), median_delta=("delta", "median"),
                                                      n_pos=("delta", lambda x: int((x > 0).sum())),
                                                      n_ci_pos=("lo", lambda x: int((x > 0).sum())),
                                                      n_ci_neg=("hi", lambda x: int((x < 0).sum())),
                                                      min_delta=("delta", "min"), max_delta=("delta", "max")).reset_index()
    X.to_csv(out / "cross_system_summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 400)
    print(S.round(3).to_string(index=False))
    print(X.round(3).to_string(index=False))
    print(C.groupby("system")[[c for c in C.columns if c.startswith("cov|")]].mean().round(2).to_string())


if __name__ == "__main__":
    shard(sys.argv[2]) if sys.argv[1] == "shard" else merge()
