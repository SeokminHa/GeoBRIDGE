"""E8 (EVALUATION-ONLY): does conditioning remove the ENDOGENOUS master-regulator (MR) programme? m_g = mean over 2 roots of R2 of OLS(dMu_g ~ dMu[MRs\\g] + intercept),
dMu = Mu[J]-Mu[I] (h5ad layer Mu). Statistic: per source Spearman(score, m_g) over NON-DIRECT targets (stratum), mean over sources. MR ids used only for evaluation.
usage: python gb_e8_mr_alignment_2026_11_02.py OUTSUB [--grns ..] [--topos ..] [--conds ..] [--sims ..]"""
import sys, os, json, itertools, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
import gb_tf32rep_fast_2026_10_22 as T
import gb_b4_score as B
import anndata
R = T.R; OUT = f"{R}/results/signed/r2fix_2026_11_01/E8_mr_alignment/{sys.argv[1]}"; os.makedirs(OUT, exist_ok=True)
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1].split(",") if k in sys.argv else d
GRNS = [int(x) for x in arg("--grns", T.GRNS)]; TOPOS = arg("--topos", T.TOPOS); CONDS = arg("--conds", list(T.COND)); SIMS = [int(x) for x in arg("--sims", T.SIMS)]
ROOTS = B.GP.ROOTS; V = T.V_(); TAB = pd.read_csv(f"{T.MAIN_ROOT}/TABLE.tsv", sep="\t"); MINT = 5
STRATA = {"d2": lambda d: d == 2, "d3": lambda d: d == 3, "d4p": lambda d: (d >= 4) & (d < 99), "unreach": lambda d: d >= 99, "pooled": lambda d: d >= 2}


def mr_r2(dM, mrs):
    out = np.zeros(dM.shape[1])
    for g in range(dM.shape[1]):
        x = [m for m in mrs if m != g]; y = dM[:, g] - dM[:, g].mean(); sst = y @ y
        if sst <= 1e-300 or not x: out[g] = np.nan; continue
        X = dM[:, x] - dM[:, x].mean(0); b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b; out[g] = 1 - (r @ r) / sst
    return out


def one_ds(g, t, c, s, G, D, pre):
    dd = T.dsdir(g, t, c, s); rec = json.load(open(f"{dd}/record.json")); assert rec["status"] == "OK", rec["status"]
    Gm = np.load(f"{dd}/raw/geometry_raw.npz"); Zr = np.load(f"{dd}/raw/geobridge_raw.npz")
    a = anndata.read_h5ad(T.h5path(g, t, c, s)); Mu = a.layers["Mu"]; Mu = np.asarray(Mu.toarray() if hasattr(Mu, "toarray") else Mu, float); assert Mu.shape[1] == 100
    mrs = [int(m) for m in G["mrs"]]
    mg = np.mean([mr_r2(Mu[Gm[f"pairs_{r}_J"]] - Mu[Gm[f"pairs_{r}_I"]], mrs) for r in ROOTS], 0)
    src, cols = np.asarray(Zr["source_gene_index"], int), np.asarray(Zr["cols_PI"], int); assert list(src) == pre["cands"]
    A_ = {r: np.asarray(Zr[f"TRUE|U|{r}|absS"], float) for r in ROOTS}; per = [{r: A_[r][:, i, cols[i]].T for r in ROOTS} for i in range(len(src))]
    S = {"GB": B.zc(per, range(0, 9)), "GB_k0": B.zc(per, [0])}
    gd = T.gdir(g, t); tag = T.tagof(t, g, s); vf = f"{gd}/velorama_rt0/{c}/{tag}.npz"
    if os.path.exists(vf): z = np.load(vf); assert np.array_equal(z["src"], src) and np.array_equal(z["cols"], cols); S["VEL"] = np.asarray(z["score"], float)
    for m in T.EXT_ALL:
        p = T.score_path(m, ("rep", g, t, c, s))
        if os.path.exists(p): z = np.load(p); assert np.array_equal(z["src"], src) and np.array_equal(z["cols"], cols); S[m] = np.asarray(z["score"], float)
    Dm = np.array([[D[s_, int(x)] for x in cols[i]] for i, s_ in enumerate(src)]); Y = Dm == 1
    ap = T.metrics(S["GB"], Y, Dm)["AP"]; row = TAB[(TAB.grn == g) & (TAB.topo == t) & (TAB.cond == c) & (TAB.sim == s) & (TAB.method == "GB")]
    assert len(row) == 1 and abs(row.AP.iloc[0] - ap) < 1e-9, f"GB AP parity {g}{t}{c}{s}: {ap} vs {row.AP.values}"
    out = []
    for m, sc in S.items():
        for sn, f in STRATA.items():
            rs, skip = [], 0
            for i in range(len(src)):
                al = mg[cols[i]]; nd = f(Dm[i]) & np.isfinite(sc[i]) & np.isfinite(al)
                if nd.sum() < MINT or np.ptp(sc[i][nd]) == 0 or np.ptp(al[nd]) == 0: skip += 1; continue
                rs.append(stats.spearmanr(sc[i][nd], al[nd])[0])
            out.append(dict(grn=g, topo=t, cond=c, sim=s, method=m, stratum=sn, spearman=np.mean(rs) if rs else np.nan, n_src_used=len(rs), n_src_skipped=skip, GB_AP=ap,
                            n_mr=len(mrs), mean_mg=float(np.nanmean(mg))))
    return out, ap


def t95(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]; n = len(d)
    if n < 2: return dict(mean=np.nan, lo95=np.nan, hi95=np.nan, n_grn=n, n_neg=np.nan, p_signflip=np.nan)
    m = d.mean(); h = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
    sg = np.array(list(itertools.product([1, -1], repeat=n))); p = float((np.abs((sg * d).mean(1)) >= abs(m) - 1e-15).mean())  # exact two-sided sign-flip
    return dict(mean=m, lo95=m - h, hi95=m + h, n_grn=n, n_neg=int((d < 0).sum()), p_signflip=p)


if __name__ == "__main__":
    ROWS = []; PAR = []
    for g in GRNS:
        G = V.generate_grn(g); D = T.dist_matrix(sorted((int(a), int(b)) for a, b in G["gt"]))
        for t in TOPOS:
            pre = json.load(open(f"{T.gdir(g, t)}/roster_pre.json"))
            for c in CONDS:
                for s in SIMS:
                    o, ap = one_ds(g, t, c, s, G, D, pre); ROWS += o; PAR.append(dict(grn=g, topo=t, cond=c, sim=s, GB_AP=ap)); print("done", g, t, c, s, ap, flush=True)
    E = pd.DataFrame(ROWS); E.to_csv(f"{OUT}/E8_per_dataset.tsv", sep="\t", index=False); pd.DataFrame(PAR).to_csv(f"{OUT}/parity_per_dataset.tsv", sep="\t", index=False)
    rows = []
    for c in CONDS:
        for tp in ["all"] + list(TOPOS):
            for sn in STRATA:
                sub = E[(E.cond == c) & (E.stratum == sn) & ((E.topo == tp) if tp != "all" else True)]
                W = sub.groupby(["grn", "method"]).spearman.mean().unstack()
                for m in W.columns: rows.append(dict(cond=c, topo=tp, stratum=sn, contrast=m, **t95(W[m])))
                for ref in [m for m in W.columns if m != "GB"]: rows.append(dict(cond=c, topo=tp, stratum=sn, contrast=f"GB-{ref}", **t95(W["GB"] - W[ref])))
    SUM = pd.DataFrame(rows); SUM.to_csv(f"{OUT}/E8_summary.tsv", sep="\t", index=False)
    print(SUM[(SUM.contrast.isin(["GB", "GB_k0", "GB-GB_k0"])) & (SUM.topo == "all")].to_string()); print("wrote", OUT)
