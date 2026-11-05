"""T13 story E4/E5' (EVALUATION-ONLY; reads stored artefacts, no scoring pipeline re-run, no truth fed to any method).
  E5': fraction of pair-difference energy captured by the first k (0..30) pair-space PCs of production_pca30 for hidden-driver injected
       production (A, B, A+B), GRN production, and observed Mu.
  E4 : alignment of each method's NON-DIRECT score with the TRUE shared-driver co-variation a_rg = mean_root |corr_pairs(dH_r, dH_g)|.
Hidden production dH is reconstructed EXACTLY as the generator does: production_parts(conc_unspliced_latent, edges, component_mask) (+ mr_inj
for master regulators = amp*loading*Hn[bin]); A/B split by hidden regulator id 100/101 (+ iA/iB for MRs), as in gb_tf32rep_fast DIAG.
usage: python gb_t13_story_E4E5_2026_11_02.py OUTSUB [--grns 501,..] [--topos bifurcation,tree] [--conds C0,..] [--sims 1,2]"""
import sys, os, json, numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
import gb_tf32rep_fast_2026_10_22 as T
import gb_b4_score as B
R = T.R; OUT = f"{R}/results/signed/r2fix_2026_11_01/T13_story_E4E5/{sys.argv[1]}"; os.makedirs(OUT, exist_ok=True)
arg = lambda k, d: sys.argv[sys.argv.index(k) + 1].split(",") if k in sys.argv else d
GRNS = [int(x) for x in arg("--grns", T.GRNS)]; TOPOS = arg("--topos", T.TOPOS); CONDS = arg("--conds", list(T.COND)); SIMS = [int(x) for x in arg("--sims", T.SIMS)]
ROOTS = B.GP.ROOTS; V = T.V_(); TAB = pd.read_csv(f"{T.MAIN_ROOT}/TABLE.tsv", sep="\t")
PCTOP = 0.05


def energy_curve(Q, D):
    """D (npairs,100) -> (31,) total-energy fraction and (31,) gene-mean fraction for k=0..30 (k=0 -> 0)"""
    W = (Q.T @ D) ** 2; cum = np.vstack([np.zeros((1, D.shape[1])), np.cumsum(W, 0)]); e = (D ** 2).sum(0)
    ok = e > 1e-12 * max(e.max(), 1e-300)
    return cum.sum(1) / e.sum(), (cum[:, ok] / e[ok]).mean(1)


def corr_rows(Dh, rows):
    Z = Dh - Dh.mean(0); Z = Z / (np.linalg.norm(Z, axis=0) + 1e-300)
    return Z[:, rows].T @ Z          # (nsrc, 100) correlation over pairs (zero-variance genes -> 0)


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
    import anndata
    a = anndata.read_h5ad(T.h5path(g, t, c, s)); Mu = a.layers["Mu"]; Mu = np.asarray(Mu.toarray() if hasattr(Mu, "toarray") else Mu, float); assert Mu.shape == (n, 100), Mu.shape
    src, cols = np.asarray(Zr["source_gene_index"], int), np.asarray(Zr["cols_PI"], int); assert list(src) == pre["cands"]
    inj = dict(A=amp[0] != 0, B=amp[1] != 0); inj["AB"] = inj["A"] or inj["B"]
    e5, ah, par = [], [], {}
    for r in ROOTS:
        I, J = Gm[f"pairs_{r}_I"], Gm[f"pairs_{r}_J"]; Q = np.linalg.qr(Gm["production_pca30"][J] - Gm["production_pca30"][I])[0]; assert Q.shape[1] == 30
        dH = Ph[J] - Ph[I]
        # parity: stored driver_pair_energy_prefix3 (geometry_stats, EPS clamp as in the generator)
        par[r] = float(np.linalg.norm(Q[:, :3].T @ dH, "fro") ** 2 / np.maximum(np.linalg.norm(dH, "fro") ** 2, V.EPS)); st = rec[f"geometry_{r}"]["driver_pair_energy_prefix3"]
        assert abs(par[r] - st) < 1e-9, f"driver_pair_energy_prefix3 parity {g}{t}{c}{s}{r}: {par[r]} vs {st}"
        for q, M, ok in (("A", PhA, inj["A"]), ("B", PhB, inj["B"]), ("AB", Ph, inj["AB"]), ("GRN", Pg, True), ("Mu", Mu, True)):
            Dq = M[J] - M[I]
            if ok: ft, fg = energy_curve(Q, Dq)
            else: ft = fg = np.full(31, np.nan)
            e5 += [dict(grn=g, topo=t, cond=c, sim=s, root=r, qty=q, k=k, frac_total=ft[k], frac_genemean=fg[k]) for k in range(31)]
        ah.append(np.abs(corr_rows(dH, src)))
    a_rg = np.mean(ah, 0)                                   # (nsrc,100) mean over the 2 roots of |corr|
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
        rs, tf, nn = [], [], []
        for i in range(len(src)):
            al = a_rg[i, cols[i]]; nd = (Dm[i] >= 2) & np.isfinite(sc[i]) & np.isfinite(al)
            if nd.sum() < 10 or np.ptp(sc[i][nd]) == 0 or np.ptp(al[nd]) == 0: continue
            x, y = sc[i][nd], al[nd]; rs.append(stats.spearmanr(x, y)[0]); kk = max(1, int(np.ceil(PCTOP * nd.sum())))
            top = np.argsort(-x, kind="stable")[:kk]; tert = y >= np.quantile(y, 2 / 3); tf.append(tert[top].mean()); nn.append(nd.sum())
        e4.append(dict(grn=g, topo=t, cond=c, sim=s, method=m, spearman=np.mean(rs) if rs else np.nan, top5_frac_in_top_tertile=np.mean(tf) if tf else np.nan,
                       n_src_used=len(rs), mean_n_nondirect=np.mean(nn) if nn else np.nan, GB_AP=ap,
                       mean_a_nondirect=float(np.mean([a_rg[i, cols[i]][(Dm[i] >= 2)].mean() for i in range(len(src))]))))
    return e5, e4, dict(grn=g, topo=t, cond=c, sim=s, par_min=par["dc1_min"], par_max=par["dc1_max"], GB_AP=ap,
                        hidden_over_GRN_pair_energy=float(((Ph[J] - Ph[I]) ** 2).sum() / ((Pg[J] - Pg[I]) ** 2).sum()))


def t95(d):
    d = np.asarray(d, float); d = d[np.isfinite(d)]; n = len(d)
    if n < 2: return np.nan, np.nan, np.nan, n
    m = d.mean(); h = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n); return m, m - h, m + h, n


if __name__ == "__main__":
    E5, E4, PAR = [], [], []
    for g in GRNS:
        G = V.generate_grn(g); D = T.dist_matrix(sorted((int(a), int(b)) for a, b in G["gt"]))
        for t in TOPOS:
            pre = json.load(open(f"{T.gdir(g, t)}/roster_pre.json"))
            for c in CONDS:
                for s in SIMS:
                    e5, e4, p = one_ds(g, t, c, s, G, D, pre); E5 += e5; E4 += e4; PAR.append(p); print("done", g, t, c, s, p["GB_AP"], p["par_min"], flush=True)
    E5 = pd.DataFrame(E5); E4 = pd.DataFrame(E4)
    E5.to_csv(f"{OUT}/E5_energy_by_k_per_dataset.tsv", sep="\t", index=False); E4.to_csv(f"{OUT}/E4_per_dataset.tsv", sep="\t", index=False)
    pd.DataFrame(PAR).to_csv(f"{OUT}/parity_per_dataset.tsv", sep="\t", index=False)
    E5s = E5.groupby(["cond", "topo", "qty", "k"])[["frac_total", "frac_genemean"]].mean().reset_index(); E5s.to_csv(f"{OUT}/E5_energy_by_k_summary.tsv", sep="\t", index=False)
    rows = []
    for c in [x for x in CONDS if x != "C0"]:
        for tp in ["all"] + list(TOPOS):
            sub = E4[(E4.cond == c) & ((E4.topo == tp) if tp != "all" else True)]
            for met in ("spearman", "top5_frac_in_top_tertile"):
                W = sub.groupby(["grn", "method"])[met].mean().unstack()      # GRN x method (mean over sims, and topos for 'all')
                for m in W.columns: rows.append(dict(cond=c, topo=tp, metric=met, contrast=m, **dict(zip(("mean", "lo95", "hi95", "n_grn"), t95(W[m])))))
                for ref in [m for m in W.columns if m != "GB"]:
                    rows.append(dict(cond=c, topo=tp, metric=met, contrast=f"GB-{ref}", **dict(zip(("mean", "lo95", "hi95", "n_grn"), t95(W["GB"] - W[ref])))))
    pd.DataFrame(rows).to_csv(f"{OUT}/E4_summary_t95.tsv", sep="\t", index=False)
    print("wrote", OUT)
