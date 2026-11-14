#!/usr/bin/env python
# Post-hoc conditional feature-ablation
import hashlib
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
import gb_killtests as KT  # noqa: F401  (keeps RAWH)
import gb_primary as GP
import gb_real_uniform as RU

B, K, MINPOS = 200, 5, 10
MODELS = {"N": ["raw_det", "sm_det", "abundance", "pair_support"], "N+D": ["raw_det", "sm_det", "abundance", "pair_support", "dynamics"],
          "N+GB": ["raw_det", "sm_det", "abundance", "pair_support", "GB"],
          "N+D+GB": ["raw_det", "sm_det", "abundance", "pair_support", "dynamics", "GB"]}
MODELS.update({"N+D+PRIOR": MODELS["N+D"] + ["PRIOR"], "N+D+PRIOR+GB": MODELS["N+D"] + ["PRIOR", "GB"]})
CONTRASTS = [("N+D", "N"), ("N+GB", "N"), ("N+D+GB", "N+D"), ("N+D+PRIOR", "N+D"), ("N+D+PRIOR+GB", "N+D+PRIOR")]


def feats(system):
    D = RU.canonical_inputs(system)
    S, A = D["S"], D["A"]
    X = np.asarray(S["X"], np.float64)
    sp = A.layers["spliced"]
    raw = np.asarray((sp > 0).mean(0)).ravel()
    dyn, sup = [], []
    for root in RU.ROOTS:
        I, J = (np.asarray(a, int) for a in S["pairs"][root])
        dX = X[J] - X[I]
        dyn.append(np.std(dX, 0) / np.std(X, 0))
        sup.append(np.mean(dX != 0, 0))
    return pd.DataFrame(dict(raw_det=raw, sm_det=np.mean(X > 0, 0), abundance=np.log1p(X.mean(0)), pair_support=np.mean(sup, 0),
                             dynamics=np.mean(dyn, 0)), index=list(D["fixture_genes"]))


def fold_of(genes):
    return np.array([int(hashlib.sha1(g.upper().encode()).hexdigest(), 16) % K for g in genes])


def oof(Xf, y, folds):
    p = np.full(len(y), np.nan)
    for k in range(K):
        tr, te = folds != k, folds == k
        if te.sum() == 0 or y[tr].sum() < 2:
            continue
        m = make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=3), LogisticRegression(C=1.0, max_iter=2000))
        m.fit(Xf[tr], y[tr])
        p[te] = m.predict_proba(Xf[te])[:, 1]
    return p


def endpoints(system):
    if system == "KLF1":
        import gb_klf1_eval as KE
        genes, methods, S, _ = KE.load_pred()
        T = KE.ko_truth(f"{KE.OUT}/truth/GSE94351_gene_exp.diff.gz")
        gu = np.char.upper(genes.astype(str))
        ev = np.isin(gu, T.index)
        y = np.array([bool(T.resp.get(g, False)) for g in gu])
        yield "KLF1 KO (held-out)", "functional", "KLF1", genes.astype(str), ev, y, S[list(methods).index("GeoBRIDGE")]
        return
    for name, family, _s, src, eps, n, npos in RU.trace_endpoints(systems=[system]):
        E = eps["U"]
        y = np.zeros(len(E.eval_mask), bool)
        y[E.eval_mask] = np.asarray(E.y, bool)
        yield name, "regulon" if family == "F1" else "functional", src.upper(), np.asarray(E.obs_genes).astype(str), E.eval_mask.copy(), y, \
            GP.zcons(E.RAWH)


def main(system):
    out = RU.OUT / "decomp"
    out.mkdir(parents=True, exist_ok=True)
    F = feats("GATA1" if system == "KLF1" else system)
    pz = np.load(RU.OUT / "neutral" / f"{'GATA1' if system == 'KLF1' else system}_prior.npz", allow_pickle=True)
    pr = dict(zip(pz["genes"].astype(str), np.where(pz["common"], pz["prior|GeoBRIDGE"], np.nan)))
    F["PRIOR"] = [pr.get(g, np.nan) for g in F.index]
    rng = np.random.default_rng(20261002)
    universe = list(F.index)
    uidx = {g: i for i, g in enumerate(universe)}
    W = rng.multinomial(len(universe), np.full(len(universe), 1 / len(universe)), size=B).astype(float)   # shared gene weights
    rows, keep = [], []
    for name, fam, src, genes, ev, y, gb in endpoints(system):
        fi = np.array([uidx.get(g, -1) for g in genes])
        Fv = F.reindex(genes)   # includes PRIOR (NaN outside the neutral common-target set ⇒ those rows drop)
        ok = ev & np.isfinite(gb) & (fi >= 0) & np.isfinite(Fv.to_numpy()).all(1)
        yy = y[ok]
        if yy.sum() < MINPOS or (~yy).sum() < MINPOS:
            rows.append(dict(system=system, endpoint=name, fam=fam, source=src, status="TOO_FEW", n=int(ok.sum()), n_pos=int(yy.sum())))
            continue
        Z = Fv[ok].copy()
        Z["GB"] = gb[ok]
        folds = fold_of(genes[ok])
        P = {m: oof(Z[c].to_numpy(), yy, folds) for m, c in MODELS.items()}
        good = np.all([np.isfinite(p) for p in P.values()], 0)
        r = dict(system=system, endpoint=name, fam=fam, source=src, status="OK", n=int(good.sum()), n_pos=int(yy[good].sum()),
                 AUROC_GB_raw=roc_auc_score(yy[good], Z["GB"].to_numpy()[good]))
        for m, p in P.items():
            r[f"AUROC|{m}"] = roc_auc_score(yy[good], p[good])
            r[f"APlift|{m}"] = average_precision_score(yy[good], p[good]) / yy[good].mean()
        rows.append(r)
        w = W[:, fi[ok][good]]                                   # (B, n) gene weights shared across endpoints
        bd = {}
        for a, b in CONTRASTS:
            for met, f in (("AUROC", roc_auc_score), ("AP", average_precision_score)):
                vals = []
                for wb in w:
                    k = wb > 0
                    if yy[good][k].sum() < 2 or (~yy[good][k]).sum() < 2:
                        vals.append(np.nan)
                        continue
                    vals.append(f(yy[good][k], P[a][good][k], sample_weight=wb[k]) - f(yy[good][k], P[b][good][k], sample_weight=wb[k]))
                bd[f"{met}|{a}-{b}"] = np.asarray(vals)
        keep.append(dict(fam=fam, source=src, point={f"{met}|{a}-{b}": r[f"{'AUROC' if met == 'AUROC' else 'APlift'}|{a}"]
                                                     - r[f"{'AUROC' if met == 'AUROC' else 'APlift'}|{b}"] for a, b in CONTRASTS
                                                     for met in ("AUROC", "AP")}, boot=bd, yrate=yy[good].mean()))
        print(name, {k: round(v, 3) for k, v in r.items() if k.startswith("AUROC")}, flush=True)
    R_ = pd.DataFrame(rows)
    R_.to_csv(out / f"{system}_endpoints.tsv", sep="\t", index=False)
    srows = []
    for fam in sorted({k["fam"] for k in keep}):
        ks = [k for k in keep if k["fam"] == fam]
        srcs = sorted({k["source"] for k in ks})
        for key in ks[0]["point"]:
            # point: AP contrasts use AP lift difference
            pt = np.mean([np.mean([k["point"][key] for k in ks if k["source"] == s]) for s in srcs])
            scale = (lambda k: 1 / k["yrate"]) if key.startswith("AP") else (lambda k: 1.0)
            bmat = np.vstack([np.nanmean(np.vstack([k["boot"][key] * scale(k) for k in ks if k["source"] == s]), 0) for s in srcs])
            bs = np.nanmean(bmat, 0)
            srows.append(dict(system=system, fam=fam, contrast=key, n_tf=len(srcs), n_endpoints=len(ks), delta=pt,
                              lo=np.nanpercentile(bs, 2.5), hi=np.nanpercentile(bs, 97.5)))
    S_ = pd.DataFrame(srows)
    S_.to_csv(out / f"{system}_system.tsv", sep="\t", index=False)
    ok = R_[R_.status == "OK"]
    pd.set_option("display.width", 220)
    print(ok.groupby("fam")[[c for c in ok.columns if c.startswith("AUROC") or c.startswith("APlift")]].mean().round(3).to_string())
    print(S_.round(3).to_string(index=False))
    print("skipped (TOO_FEW):", int((R_.status == "TOO_FEW").sum()))


if __name__ == "__main__":
    main(sys.argv[1].upper())
