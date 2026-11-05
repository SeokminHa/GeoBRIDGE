"""T13b: E4 sensitivity by distance stratum (EVALUATION-ONLY; copy of gb_t13_story_E4E5_2026_11_02.py one_ds logic, E5 dropped).
E4 = within-source Spearman(method NON-DIRECT score, a_rg) with a_rg = mean over 2 roots of |corr| (or |cov|) over pairs of (dH_r,dH_g), dH = reconstructed injected hidden production.
Strata of non-direct pairs: d2, d3, d4p (4<=d<99), unreach (d==99), pooled(all d>=2). Sources/strata with <MINT targets skipped (counted).
usage: python gb_t13b_E4_strat_2026_11_02.py OUTSUB [--grns ..] [--topos ..] [--conds C1,C2,C3] [--sims ..]"""
import sys, os, json, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
import gb_tf32rep_fast_2026_10_22 as T
import gb_b4_score as B
R = T.R; OUT = f"{R}/results/signed/r2fix_2026_11_01/T13b_E4_strat/{sys.argv[1]}"; os.makedirs(OUT, exist_ok=True)
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1].split(",") if k in sys.argv else d
GRNS = [int(x) for x in arg("--grns", T.GRNS)]; TOPOS = arg("--topos", T.TOPOS); CONDS = [c for c in arg("--conds", list(T.COND)) if c != "C0"]; SIMS = [int(x) for x in arg("--sims", T.SIMS)]
ROOTS = B.GP.ROOTS; V = T.V_(); TAB = pd.read_csv(f"{T.MAIN_ROOT}/TABLE.tsv", sep="\t"); MINT = 5
STRATA = {"d2": lambda d: d == 2, "d3": lambda d: d == 3, "d4p": lambda d: (d >= 4) & (d < 99), "unreach": lambda d: d >= 99, "pooled": lambda d: d >= 2}


def assoc_rows(Dh, rows, cov):
    Z = Dh - Dh.mean(0)
    if cov: return np.abs(Z[:, rows].T @ Z) / (len(Z) - 1)
    Z = Z / (np.linalg.norm(Z, axis=0) + 1e-300); return np.abs(Z[:, rows].T @ Z)


def one_ds(g, t, c, s, G, D, pre):
    dd = T.dsdir(g, t, c, s); rec = json.load(open(f"{dd}/record.json")); assert rec["status"] == "OK", rec["status"]
    Gm = np.load(f"{dd}/raw/geometry_raw.npz"); Zr = np.load(f"{dd}/raw/geobridge_raw.npz"); amp = rec["ab"]["amp"]
    Ul = np.asarray(Gm["conc_unspliced_latent"], float); Ed = Gm["edges"]; cm = np.asarray(Gm["component_edge_mask"], bool); bb = np.asarray(Gm["sampled_bin"], int)
    hA, hB = V.ab_hn(t); cA, cB = V.ab_loadings(g); n = len(Ul)
    iA, iB = np.zeros((n, 100)), np.zeros((n, 100))
    for m in G["mrs"]: iA[:, m] = amp[0] * cA[m] * hA[bb]; iB[:, m] = amp[1] * cB[m] * hB[bb]
    V.PILOT["mr_inj"] = iA + iB; Pg, Ph = V.production_parts(Ul, Ed, component_mask=cm); V.PILOT["mr_inj"] = None
    PhA = V.production_parts(Ul, Ed, component_mask=cm & (np.asarray(Ed["regulator"]) == 100))[1] + iA
    PhB = V.production_parts(Ul, Ed, component_mask=cm & (np.asarray(Ed["regulator"]) == 101))[1] + iB
    assert np.abs(PhA + PhB - Ph).max() <= 1e-9 * max(1, np.abs(Ph).max()), "A+B != total hidden"
    src, cols = np.asarray(Zr["source_gene_index"], int), np.asarray(Zr["cols_PI"], int); assert list(src) == pre["cands"]
    ah = {"corr": [], "cov": []}; par = {}
    for r in ROOTS:
        I, J = Gm[f"pairs_{r}_I"], Gm[f"pairs_{r}_J"]; Q = np.linalg.qr(Gm["production_pca30"][J] - Gm["production_pca30"][I])[0]; assert Q.shape[1] == 30
        dH = Ph[J] - Ph[I]
        par[r] = float(np.linalg.norm(Q[:, :3].T @ dH, "fro") ** 2 / np.maximum(np.linalg.norm(dH, "fro") ** 2, V.EPS)); st = rec[f"geometry_{r}"]["driver_pair_energy_prefix3"]
        assert abs(par[r] - st) < 1e-9, f"driver_pair_energy_prefix3 parity {g}{t}{c}{s}{r}: {par[r]} vs {st}"
        for k_, cv in (("corr", False), ("cov", True)): ah[k_].append(assoc_rows(dH, src, cv))
    a = {k_: np.mean(v, 0) for k_, v in ah.items()}
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
    e4 = []
    for m, sc in S.items():
        for ak in ("corr", "cov"):
            for sn, f in STRATA.items():
                rs, nt, skip = [], [], 0
                for i in range(len(src)):
                    al = a[ak][i, cols[i]]; nd = f(Dm[i]) & np.isfinite(sc[i]) & np.isfinite(al)
                    if nd.sum() < MINT or np.ptp(sc[i][nd]) == 0 or np.ptp(al[nd]) == 0: skip += 1; continue
                    rs.append(stats.spearmanr(sc[i][nd], al[nd])[0]); nt.append(nd.sum())
                e4.append(dict(grn=g, topo=t, cond=c, sim=s, method=m, assoc=ak, stratum=sn, spearman=np.mean(rs) if rs else np.nan,
                               n_src_used=len(rs), n_src_skipped=skip, mean_n_targets=np.mean(nt) if nt else np.nan, GB_AP=ap))
    return e4, dict(grn=g, topo=t, cond=c, sim=s, par_min=par["dc1_min"], par_max=par["dc1_max"], GB_AP=ap)


def t95(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]; n = len(d)
    if n < 2: return np.nan, np.nan, np.nan, n, np.nan
    m = d.mean(); h = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n); return m, m - h, m + h, n, int((d < 0).sum())


if __name__ == "__main__":
    E4, PAR = [], []
    for g in GRNS:
        G = V.generate_grn(g); D = T.dist_matrix(sorted((int(a), int(b)) for a, b in G["gt"]))
        for t in TOPOS:
            pre = json.load(open(f"{T.gdir(g, t)}/roster_pre.json"))
            for c in CONDS:
                for s in SIMS:
                    e4, p = one_ds(g, t, c, s, G, D, pre); E4 += e4; PAR.append(p); print("done", g, t, c, s, p["GB_AP"], p["par_min"], flush=True)
    E4 = pd.DataFrame(E4); E4.to_csv(f"{OUT}/E4strat_per_dataset.tsv", sep="\t", index=False); pd.DataFrame(PAR).to_csv(f"{OUT}/parity_per_dataset.tsv", sep="\t", index=False)
    cnt = E4.groupby(["cond", "assoc", "stratum", "method"])[["n_src_used", "n_src_skipped"]].sum().reset_index(); cnt.to_csv(f"{OUT}/E4strat_source_counts.tsv", sep="\t", index=False)
    rows = []; kw = ("mean", "lo95", "hi95", "n_grn", "n_neg")
    for c in CONDS:
        for ak in ("corr", "cov"):
            for sn in STRATA:
                sub = E4[(E4.cond == c) & (E4.assoc == ak) & (E4.stratum == sn)]
                W = sub.groupby(["grn", "method"]).spearman.mean().unstack()      # GRN x method (mean over topos, sims)
                for m in W.columns: rows.append(dict(cond=c, assoc=ak, stratum=sn, contrast=m, **dict(zip(kw, t95(W[m])))))
                for ref in [m for m in W.columns if m != "GB"]:
                    rows.append(dict(cond=c, assoc=ak, stratum=sn, contrast=f"GB-{ref}", **dict(zip(kw, t95(W["GB"] - W[ref])))))
    SUM = pd.DataFrame(rows); SUM.to_csv(f"{OUT}/E4strat_summary_t95.tsv", sep="\t", index=False)
    print(SUM[(SUM.contrast.isin(["GB", "GB_k0", "GB-GB_k0"]))].to_string()); print("wrote", OUT)
