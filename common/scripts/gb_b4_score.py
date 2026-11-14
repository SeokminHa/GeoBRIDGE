#!/usr/bin/env python
# Aggregate stored U-channel profiles with gb_primary.zcons.
import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
import gb_primary as GP
import gb_sergio_band_eval as E
import gb_sergio_drift_v5 as V

R = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/results/sergio_drift/confirm_P")
KALL = 31


def desc(direct, s):
    seen, st = set(), [s]
    while st:
        u = st.pop()
        for a, b in direct:
            if a == u and b not in seen:
                seen.add(b)
                st.append(b)
    seen.discard(s)
    return seen


# per_src_roots: list over sources of {root: (ntarg, 31)} -> (nsrc, ntarg) via the frozen gb_primary.zcons
def zc(per_src_roots, ks):
    return np.vstack([GP.zcons(p, ks) for p in per_src_roots])


def pcpart_matched(ms, mu, pcs, src, cols):
    out = []
    for k in range(9):
        X = np.column_stack([np.ones(len(ms))] + ([pcs[:, :k]] if k else []))
        res = lambda M: M - X @ np.linalg.lstsq(X, M, rcond=None)[0]
        a, b = res(ms[:, src]), res(mu)
        a /= np.linalg.norm(a, axis=0, keepdims=True) + 1e-12
        b /= np.linalg.norm(b, axis=0, keepdims=True) + 1e-12
        C = np.abs(a.T @ b)                                          # (nsrc, ngenes)
        out.append(np.take_along_axis(C, cols, 1))
    per = []
    for i in range(len(src)):
        Zk = np.stack([GP.zcol(out[k][i]) for k in range(9)])
        n_ok = np.isfinite(Zk).sum(0)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(Zk, 0)
        m[n_ok < 6] = np.nan
        per.append(m)
    return np.vstack(per)


DEV = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/results/sergio_drift/core_P")
SEEDS = (1, 2)


# smoke=True: only the exposed GRN100 canary -> B4_SMOKE_GRN100.tsv (code check)
def score(smoke=False, dev=False):
    rows = []
    pat = f"{DEV}/*/*_P_q3_G*_g*_s0/record.json" if dev else f"{R}/*/*_P_q3_G*_g1[0-9][0-9]_s[12]/record.json"
    for rj in sorted(glob.glob(pat)):
        dd = os.path.dirname(rj)
        rec = json.load(open(rj))
        g = int(rec["grn_seed"])
        if dev:
            if g >= 100:
                continue
        elif (g != 100) if smoke else (g == 100 or not 101 <= g <= 109):
            continue
        if not os.path.exists(f"{dd}/raw/geobridge_raw.npz"):
            print("MISSING raw", dd, flush=True)                  # recorded; summary fails closed on the expected grid
            continue
        Z = np.load(f"{dd}/raw/geobridge_raw.npz")
        C = np.load(f"{dd}/raw/comparators.npz")
        G = V.generate_grn(g)
        src = np.asarray(Z["source_gene_index"], int)
        cols = np.asarray(Z["cols_PI"], int)
        prim = np.array([s in G["regs"] for s in src])
        direct = {(int(s), int(t)) for s, t in G["gt"]}
        two = E.strict_two_hop(direct, V.N_GENES)
        Yd = np.array([[(s, int(t)) in direct for t in cols[i]] for i, s in enumerate(src)])[prim]
        Yt = np.array([[int(t) in desc(direct, int(s)) for t in cols[i]] for i, s in enumerate(src) if prim[i]])
        L = np.array([[1 if (s, int(t)) in direct else (0 if (s, int(t)) in two else -1) for t in cols[i]] for i, s in enumerate(src)])[prim]
        A = {r: np.asarray(Z[f"TRUE|U|{r}|absS"], float) for r in E.ROOTS}          # (31, nsrc, ngenes)
        per = [{r: A[r][:, i, cols[i]].T for r in E.ROOTS} for i in range(len(src))]   # (ntarg, 31) per source
        Ak = {r: [np.take_along_axis(A[r][k], cols, 1) for k in range(KALL)] for r in E.ROOTS}

        def metr(sc):
            sc = sc[prim]
            ok = np.isfinite(sc)
            di = ok & (L >= 0)
            return dict(AP=average_precision_score(Yd[ok], sc[ok]), AP_desc=average_precision_score(Yt[ok], sc[ok]),
                        macroAP=np.mean([average_precision_score(Yd[i][ok[i]], sc[i][ok[i]]) for i in range(len(Yd)) if Yd[i][ok[i]].any()]),
                        DvI=roc_auc_score(L[di] == 1, sc[di]), cov=float(ok.mean()))
        import anndata as ad
        a = ad.read_h5ad(glob.glob(f"{dd}/*.h5ad")[0])
        dense = lambda M: np.asarray(M.todense()) if hasattr(M, "todense") else np.asarray(M)
        pcs = np.asarray(np.load(f"{dd}/raw/geometry_raw.npz")["production_pca30"], float)
        views = {"PRIMARY": zc(per, range(0, 9)), "Z_1_8": zc(per, range(1, 9)),
                 "RAW": E.root_consensus({r: E.rank_percentile(Ak[r][0]) for r in E.ROOTS}),
                 "PCT_1_8": E.root_consensus({r: np.mean([E.rank_percentile(Ak[r][k]) for k in range(1, 9)], 0) for r in E.ROOTS}),
                 "GENIE3_pct": E.rank_percentile(np.asarray(C["GENIE3"], float)),
                 "PCPART_matched": pcpart_matched(dense(a.layers["Ms"]).astype(float), dense(a.layers["Mu"]).astype(float), pcs, src, cols)}
        out = dict(dataset=os.path.basename(dd), obs=rec["obs"], topo=rec["topo"], grn=g, G=rec["G_target"], sim=int(rec["sim_seed"]))
        for nm, sc in views.items():
            for m, v in metr(sc).items():
                out[f"{nm}|{m}"] = v
        out["ORACLE|AP"] = max(metr(zc(per, range(a_, b_ + 1)))["AP"] for a_ in range(KALL) for b_ in range(a_, KALL))
        out["PRIMARY|regret"] = out["ORACLE|AP"] - out["PRIMARY|AP"]
        rows.append(out)
        print(out["dataset"], {k: round(v, 4) for k, v in out.items() if k.endswith("|AP")}, flush=True)
    D = pd.DataFrame(rows)
    D.to_csv(f"{R}/B4_DEVREF.tsv" if dev else f"{R}/B4_SMOKE_GRN100.tsv" if smoke else f"{R}/B4_DATASETS.tsv", sep="\t", index=False)


def hier(D, num):
    D = D.copy()
    D["w"] = 1.0 / (D.groupby("grn").topo.transform("nunique") * D.groupby(["grn", "topo"]).G.transform("nunique")
                    * D.groupby(["grn", "topo", "G"]).obs.transform("size"))
    D["w"] = D.w / D.w.groupby(D.grn).transform("sum")
    return D.groupby("grn").apply(lambda x: pd.Series({c: np.average(x[c], weights=x.w) for c in num}))


def summary():
    from scipy.stats import t as T
    D = pd.read_csv(f"{R}/B4_DATASETS.tsv", sep="\t")
    num = [c for c in D.columns if "|" in c]
    need = [c for c in num if c.split("|")[1] in ("AP", "DvI", "regret")]
    D["G"] = D.G.round(2)
    gen = []
    for rj in glob.glob(f"{R}/*/*_P_q3_G*_g1[0-9][0-9]_s[12]/record.json"):
        rec = json.load(open(rj))
        if 101 <= int(rec["grn_seed"]) <= 109:
            gen.append((int(rec["grn_seed"]), rec["topo"], round(float(rec["G_target"]), 2), rec["obs"], int(rec["sim_seed"])))
    gen = pd.DataFrame(gen, columns=["grn", "topo", "G", "obs", "sim"])
    ns = gen.groupby(["grn", "topo", "G", "obs"]).sim.nunique()
    inset = ns[ns == 2].index                                         # stratum in set iff both seeds generated
    grid = pd.MultiIndex.from_product([range(101, 110), E.TOPOLOGIES, (0.0, .15, .3, .45, .6, .75), ("UMI_DEPTH", "CONC")],
                                      names=["grn", "topo", "G", "obs"])                    # frozen expected grid (contract)
    ns = ns.reindex(grid, fill_value=0)
    excluded = ns[ns < 2]
    have = D.set_index(["grn", "topo", "G", "obs", "sim"])
    need_idx = pd.MultiIndex.from_tuples([(*k, sd) for k in inset for sd in SEEDS], names=have.index.names)
    missing = need_idx.difference(have.index)
    sub = have.loc[have.index.intersection(need_idx)]
    nonfinite = sub[~np.isfinite(sub[need]).all(1)].index
    D = sub.reset_index()
    contrib = D.groupby("grn").topo.nunique()
    contrib = contrib[contrib >= 3]
    D = D[D.grn.isin(contrib.index)]
    complete = len(missing) == 0 and len(nonfinite) == 0 and len(contrib) >= 7
    print(f"generated strata {len(ns)} (in-set {len(inset)}, excluded one-seed {len(excluded)}: {dict(excluded.groupby(level=0).size())}); "
          f"in-set datasets missing {len(missing)}, nonfinite {len(nonfinite)}; contributing GRNs {len(contrib)}/9; status: "
          f"{'COMPLETE' if complete else 'INCOMPLETE: all contrasts UNCONFIRMED'}")
    D = D.groupby(["grn", "topo", "G", "obs"], as_index=False)[num].mean()           # Both seeds are present and finite; take their mean.
    Gm = hier(D, num)
    Gm.to_csv(f"{R}/B4_GRN.tsv", sep="\t")
    Dv = pd.read_csv(f"{R}/B4_DEVREF.tsv", sep="\t")
    dev_regret = float(hier(Dv, ["PRIMARY|regret"])["PRIMARY|regret"].mean())      # dev: GRN-level hierarchical mean, same 0..30 oracle

    def bound(d):
        n = len(d)
        return d.mean(), (d.mean() - T.ppf(.95, n - 1) * d.std(ddof=1) / np.sqrt(n)) if n > 1 else np.nan
    res = []
    for name, a, b, m, margin in (("C1 PRIMARY - PCT_1_8", "PRIMARY", "PCT_1_8", "AP", 0), ("C2 PRIMARY - GENIE3_pct", "PRIMARY", "GENIE3_pct", "AP", 0),
                                  ("C3a PRIMARY - PCPART_matched", "PRIMARY", "PCPART_matched", "AP", 0),
                                  ("C3b PRIMARY - PCPART_matched", "PRIMARY", "PCPART_matched", "DvI", 0),
                                  ("C4 PRIMARY - Z_1_8 (>= -.01)", "PRIMARY", "Z_1_8", "AP", -.01)):
        mean, lo = bound(Gm[f"{a}|{m}"] - Gm[f"{b}|{m}"])
        ok = (lo >= margin) if name.startswith("C4") else (lo > margin)   # C4 is non-inferiority (>=), C1-C3 superiority (>)
        res.append(dict(contrast=name, metric=m, n_grn=len(Gm), mean=mean, bound95=lo, bound_side="lower",
                        verdict="CONFIRMED" if (complete and ok) else "UNCONFIRMED"))
    rg = Gm["PRIMARY|regret"]
    up = rg.mean() + T.ppf(.95, len(rg) - 1) * rg.std(ddof=1) / np.sqrt(len(rg)) if len(rg) > 1 else np.nan
    res.append(dict(contrast=f"C5 PRIMARY regret <= dev {dev_regret:.4f} + .01 (upper 95 %)", metric="AP", n_grn=len(Gm), mean=rg.mean(),
                    bound95=up, bound_side="upper", verdict="CONFIRMED" if (complete and up <= dev_regret + .01) else "UNCONFIRMED"))
    Rz = pd.DataFrame(res)
    Rz.to_csv(f"{R}/B4_CONTRASTS.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print(Gm[[c for c in num if c.endswith("|AP") or c.endswith("|DvI")]].round(4).to_string())
    print(Rz.round(4).to_string(index=False))


if __name__ == "__main__":
    {"smoke": lambda: score(smoke=True), "devref": lambda: score(dev=True), "score": score, "summary": summary}[sys.argv[1]]()
