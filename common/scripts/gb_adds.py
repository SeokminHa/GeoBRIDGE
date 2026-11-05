#!/usr/bin/env python
"""WHAT DOES GEOBRIDGE ADD — analyses #1 (incremental value), #2 (component factorial), #8 (k regret).
Design: internal design note (not shipped) (brief internal design note (not shipped)). Evaluation only; no score changes.
Endpoints = the frozen KT endpoints (RU.trace_endpoints); rows per endpoint = eval_mask ∩ finite for every quantity used.
Per-target truth-free features (explicit definitions, identical in every system):
  abundance  = log1p mean Ms;  detection = fraction of cells with unsmoothed log-normalised spliced (fixture .X) > 0;
  dynamics   = mean over roots SD(ΔMs over production pairs) / SD(Ms over cells);  pair_support = mean over roots frac(ΔMs != 0);
  |r|        = |corr(Ms_source, Ms_target)|;  PCp = |corr(R8 Ms_source, R8 Mu_target)|, R8 = residual on intercept + PC1..8.
#1  OOF L2 logistic (C = 1, lbfgs), 5 folds fixed per system by gene (seed 0), training-fold scaling.
    Delta = AP(B + Z) - AP(B) with B = the six features, Z = GB PRIMARY. NO_CALL if < 10 positives.
#2  2x2x2 {pair, cell} x {U, M} x {Z0:8, k0}; pair U/M = production |S| (E.RAWH / EX.RAWH); cell twin = same concordance
    |sum x y| / sum |x y| over CELLS with x = residual source Ms, y = residual target Mu (U) or Ms (M), residual on intercept +
    PC1..k (k = 0 -> centring only); Z = zcons over k 0..8. Contrasts: channel = pairUZ - pairMZ; conditioning = pairUZ - pairU0;
    pairspace = pairUZ - cellUZ.
#8  AP of every single k 0..30 (pair U) vs Z; regret R* = max_k AP_k - AP_Z (descriptive oracle), tolerance .01 frozen.
Aggregation: endpoint -> TF (source) -> system (equal weight) per family; CIs by resampling TFs within system (descriptive);
cross-system by resampling systems.
  python scratch/gb_adds.py --systems PANC DENTATEGYRUS [--out adds_smalln]
"""
import argparse
import sys
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
import gb_killtests as KT          # noqa: F401  (patches CA.Endpoint to keep RAWH)
import gb_primary as GP
import gb_real_uniform as RU

ROOTS = GP.ROOTS
FEAT = ["abundance", "detection", "dynamics", "pair_support", "abs_r", "PCp8"]
KMAX_CELL = 8


def dn(X):
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def context(system):
    import anndata as ad
    import gb_substrate as GS
    rows, eg = RU.gb_raw_rows(system)
    eg = np.asarray(eg).astype(str)
    fx, _, _ = GS.paths(system, canonical=True)
    A = ad.read_h5ad(fx)
    fm = {g: i for i, g in enumerate(np.asarray(A.var_names).astype(str))}
    cols = np.asarray([fm[g] for g in eg])
    Ms = dn(A.layers["Ms"])[:, cols].astype(np.float64)
    Mu = dn(A.layers["Mu"])[:, cols].astype(np.float64)
    X = dn(A.X)[:, cols]
    S = GS.load(system, canonical=True)
    dyn, sup = [], []
    for h in ROOTS:
        I, J = (np.asarray(a, int) for a in S["pairs"][h])
        d = Ms[J] - Ms[I]
        dyn.append(d.std(0) / np.where(Ms.std(0) > 0, Ms.std(0), np.nan))
        sup.append((d != 0).mean(0))
    feat = pd.DataFrame({"abundance": np.log1p(Ms.mean(0)), "detection": (X > 0).mean(0),
                         "dynamics": np.mean(dyn, 0), "pair_support": np.mean(sup, 0)}, index=eg)
    P = np.asarray(A.obsm["X_pca"], float)[:, :KMAX_CELL]
    fold = np.random.default_rng(0).permutation(len(eg)) % 5
    return dict(eg=eg, gi={g: i for i, g in enumerate(eg)}, Ms=Ms, Mu=Mu, P=P, feat=feat, fold=fold)


def cell_scores(ctx, src):
    """cell-space concordance twin for U (Mu) and M (Ms) at k 0..8; plus |r| and PC-partial(8)."""
    Ms, Mu, P = ctx["Ms"], ctx["Mu"], ctx["P"]
    s = Ms[:, ctx["gi"][src]]
    out = {"U": np.empty((Ms.shape[1], KMAX_CELL + 1)), "M": np.empty((Ms.shape[1], KMAX_CELL + 1))}
    pcp8 = None
    for k in range(KMAX_CELL + 1):
        Q, _ = np.linalg.qr(np.column_stack([np.ones(len(s)), P[:, :k]]))
        res = lambda Y: Y - Q @ (Q.T @ Y)
        x = res(s[:, None])
        for ch, Y in (("U", Mu), ("M", Ms)):
            prod = x * res(Y)
            den = np.abs(prod).sum(0)
            out[ch][:, k] = np.where(den > 1e-12, np.abs(prod.sum(0)) / den, np.nan)
            if k == KMAX_CELL and ch == "U":
                ry = res(Y)
                nx, ny = np.linalg.norm(x), np.linalg.norm(ry, axis=0)
                pcp8 = np.abs((x[:, 0] @ ry) / (nx * np.where(ny > 0, ny, np.nan)))
    z = lambda a: (a - a.mean()) / a.std()
    r = np.abs((z(s)[:, None] * ((Ms - Ms.mean(0)) / np.where(Ms.std(0) > 0, Ms.std(0), np.nan))).mean(0))
    return out, r, pcp8


def oof_ap(Xf, y, fold):
    p = np.zeros(len(y))
    for f in range(5):
        tr, te = fold != f, fold == f
        if y[tr].sum() < 2 or (~y[tr]).sum() < 2 or te.sum() == 0:
            return np.nan
        mu, sd = Xf[tr].mean(0), Xf[tr].std(0) + 1e-12
        p[te] = LogisticRegression(C=1.0, max_iter=2000).fit((Xf[tr] - mu) / sd, y[tr]).predict_proba((Xf[te] - mu) / sd)[:, 1]
    return average_precision_score(y, p)


def run(systems, outdir):
    out = RU.OUT / outdir
    out.mkdir(parents=True, exist_ok=True)
    rows_ep, rows_k = [], []
    for system in systems:
        t0 = time.time()
        ctx = context(system)
        cache = {}
        for name, family, _s, src, eps, n, npos in RU.trace_endpoints(systems=[system]):
            src = src.upper()
            E, EX = eps["U"], eps["X"]
            ix = np.asarray([ctx["gi"][g] for g in E.obs_genes])
            if src not in cache:
                srcname = [g for g in ctx["eg"] if g.upper() == src][0]
                cache[src] = cell_scores(ctx, srcname)
            cell, r, pcp8 = cache[src]
            nk = E.RAWH[ROOTS[0]].shape[1]
            sc = {"pairUZ": GP.zcons(E.RAWH), "pairMZ": GP.zcons(EX.RAWH),
                  "pairU0": GP.zcons(E.RAWH, ks=(0,)), "pairM0": GP.zcons(EX.RAWH, ks=(0,))}
            for ch in "UM":
                c = cell[ch][ix]
                sc[f"cell{ch}Z"] = GP.zcons({h: c for h in ROOTS})
                sc[f"cell{ch}0"] = GP.zcons({h: c for h in ROOTS}, ks=(0,))
            single = {k: GP.zcons(E.RAWH, ks=(k,)) for k in range(nk)}
            F = ctx["feat"].iloc[ix].copy()
            F["abs_r"], F["PCp8"] = r[ix], pcp8[ix]
            F.iloc[np.flatnonzero(ix == ctx["gi"][[g for g in ctx["eg"] if g.upper() == src][0]])] = np.nan
            common = E.eval_mask & np.isfinite(F[FEAT].to_numpy()).all(1)
            for v in sc.values():
                common &= np.isfinite(v)
            y = np.zeros(len(ix), bool)
            y[E.eval_mask] = np.asarray(E.y, bool)
            yy = y[common]
            if yy.sum() < 10:
                rows_ep.append(dict(system=system, family=family, endpoint=name, source=src, status="NO_CALL_POS<10",
                                    n=int(common.sum()), n_pos=int(yy.sum())))
                continue
            rec = dict(system=system, family=family, endpoint=name, source=src, status="OK", n=int(common.sum()),
                       n_pos=int(yy.sum()), prevalence=float(yy.mean()))
            for m, v in sc.items():
                rec[f"AP_{m}"] = average_precision_score(yy, v[common])
                rec[f"AUC_{m}"] = roc_auc_score(yy, v[common])
            B = F[FEAT].to_numpy()[common]
            fold = ctx["fold"][ix][common]
            rec["AP_base"] = oof_ap(B, yy, fold)
            rec["AP_base+GB"] = oof_ap(np.column_stack([B, sc["pairUZ"][common]]), yy, fold)
            rec["AP_base+M"] = oof_ap(np.column_stack([B, sc["pairMZ"][common]]), yy, fold)
            rec["d_incr_GB"] = rec["AP_base+GB"] - rec["AP_base"]
            rec["d_incr_M"] = rec["AP_base+M"] - rec["AP_base"]
            rec["d_channel"] = rec["AP_pairUZ"] - rec["AP_pairMZ"]
            rec["d_conditioning"] = rec["AP_pairUZ"] - rec["AP_pairU0"]
            rec["d_pairspace"] = rec["AP_pairUZ"] - rec["AP_cellUZ"]
            ok_k = common.copy()
            for v in single.values():
                ok_k &= np.isfinite(v)
            yk = y[ok_k]
            if yk.sum() >= 10:
                apz = average_precision_score(yk, sc["pairUZ"][ok_k])
                aps = {k: average_precision_score(yk, v[ok_k]) for k, v in single.items()}
                rec["regret"] = max(aps.values()) - apz
                rec["best_k"] = int(max(aps, key=aps.get))
                rec["n_lost_regret"] = int(common.sum() - ok_k.sum())
                rows_k += [dict(system=system, family=family, endpoint=name, source=src, k=k, AP=a, AP_Z=apz) for k, a in aps.items()]
            rows_ep.append(rec)
        print(f"[adds] {system} done {time.time()-t0:.0f}s", flush=True)
    D, K = pd.DataFrame(rows_ep), pd.DataFrame(rows_k)
    D.to_csv(out / "endpoints.tsv", sep="\t", index=False)
    K.to_csv(out / "kprofile.tsv", sep="\t", index=False)
    summarize(D, out)


def summarize(D, out, nb=2000):
    rng = np.random.default_rng(0)
    D = D[D.status == "OK"].copy()
    D["fam"] = np.where(D.family == "F1", "regulon", "functional")
    cols = ["d_incr_GB", "d_incr_M", "d_channel", "d_conditioning", "d_pairspace", "regret", "AP_base", "AP_base+GB",
            "AP_pairUZ", "AP_pairMZ", "AP_pairU0", "AP_cellUZ", "AP_cellMZ", "prevalence"]
    res = []
    for (sy, fam), g in D.groupby(["system", "fam"]):
        T = g.groupby("source")[cols].mean()
        for c in cols:
            b = [T[c].iloc[rng.integers(0, len(T), len(T))].mean() for _ in range(nb)]
            res.append(dict(system=sy, fam=fam, stat=c, n_TF=len(T), n_endpoints=len(g), mean=T[c].mean(),
                            lo=np.nanpercentile(b, 2.5), hi=np.nanpercentile(b, 97.5), frac_pos=float((T[c] > 0).mean())))
    R_ = pd.DataFrame(res)
    R_.to_csv(out / "summary_system.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    for c in ("d_incr_GB", "d_incr_M", "d_channel", "d_conditioning", "d_pairspace", "regret"):
        print(f"== {c}")
        print(R_[R_.stat == c][["system", "fam", "n_TF", "n_endpoints", "mean", "lo", "hi", "frac_pos"]].round(4).to_string(index=False))
    print("== AP levels")
    print(R_[R_.stat.isin(["AP_base", "AP_base+GB", "AP_pairUZ", "AP_pairMZ", "AP_pairU0", "AP_cellUZ", "AP_cellMZ", "prevalence"])]
          .pivot_table(index=["system", "fam"], columns="stat", values="mean").round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    ap.add_argument("--out", default="adds_smalln")
    a = ap.parse_args()
    run([s.upper() for s in a.systems], a.out)
