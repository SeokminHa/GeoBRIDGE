#!/usr/bin/env python
"""TRUTH-FREE k-rule geometry audit — internal directive (results doc §183) with internal review operator-faithful definitions
(internal design note (not shipped)). Reads NO truth. FETAL CORTEX excluded by construction (not in SYSTEMS).
usage: gb_geometry_audit.py <SYSTEM> | selftest | table        -> results/geometry_audit/<SYS>.{npz,json}, TABLE.tsv
PRODUCTION OPERATOR: gb_target_extension.py:160 P30 = V1 fixture obsm X_pca[:, :30] (gb_canonical.py:166 sc.pp.pca(n_comps=30,
zero_center) on log .X over HVG(seurat top-2000, log view) ∪ full TF catalogue, gb_canonical.py:148-160); :168 Q_h = qr(B_h P30), nested
prefixes Q_hk; :60-84 project_scores residualises BOTH source Ms and target Mu pair differences on Q_hk, |S| = |Σ x u| / Σ|x u|.
DEFINITIONS (internal review):
 D_h = B_h Ms_Gref diag(1/s_g); G_ref = every canonical gene with positive Ms variance (source- and truth-independent); s_g = cell-level
 Ms SD. C^w_h(k) = 1 − ||W^½(I−Π_hk)D_h||² / ||W^½ D_h||², weights w_pair = 1/m, w_cell(i,j) = (1/d_i + 1/d_j)/(2 n_active), d = incident
 pair count. Production projector kept; NO weighted QR; C^w may be non-monotone (kept). Rank-deficient prefix (|R_kk| < 1e-10·|R_11|)
 flagged, never pivoted.
 Audit basis: own 80-PC PCA of the same log .X / gene mask; first 30 compared with stored P30; Kmax prefixes 20/30/50/80 on production
 pairs (graph, Ms/Mu fixed). ADEQUATE(K) iff max_{h,w}|C^w(80) − C^w(K)| ≤ .02 on full data AND at the 90th percentile over the 20
 resplits (probe pairs, fitting-set basis); else BASIS_LIMITED.
 Reproducibility: 20 partitions = two disjoint 40 % fitting sets + 20 % probe, stratified by truth-free state metadata (obs label if present ×
 dc1_min DPT decile); PCA refit per fitting set (per-cell normalisation is cell-local; gene mask held fixed = declared simplification).
 Loadings compared in GENE coordinates: prefix k passes a partition iff min cos²θ ≥ .80 and mean cos²θ ≥ .90; k ∈ R_load iff ≥ 18/20.
 Probe-projector check: both bases evaluated on the same probe pairs, QR, same criterion -> R^probe_h. Block 31–50 reported.
 Bands (K = 30 registered capacity; K = 80 audit reported): B_A = {1..5}; B_B = {k ∈ 1..K : .20 ≤ C^w_h(k) ≤ .50 ∀ h, w};
 B_C = B_B ∩ R_load ∩ ⋂_h R^probe_h. < 2 depths -> BAND_UNRESOLVED.
 Source guard: R^w_rh(k) = ||W^½(I−Π_hk)B_h Ms_r||² / ||W^½ B_h Ms_r||²; UNSUPPORTED if min_{h,w,k∈band} R < ρ, ρ = .20 (sens .10/.30);
 zero baseline energy -> UNSUPPORTED. Eligible sources = catalogue TFs on the axis with spliced detection ≥ .10 (production floor).
 Collapse diagnostics (all eligible sources, 1000 fixed-random non-source targets, rng 7, U channel, production Q30, k 0..30):
 scoreable fraction (den > EPS), median den, median PR = den²/z2, |S| IQR, root disagreement 1 − Spearman(min, max); k×k Spearman for focal.
RED FLAGS (declared before computation): BASIS_LIMITED (above); WEIGHTING_DEPENDENT |k.5(cell) − k.5(pair)| ≥ 2 or max_k|C^cell − C^pair| > .05;
 ROOT_INCOMPATIBLE |k.5(min) − k.5(max)| ≥ 2 or |d_rep(min) − d_rep(max)| ≥ 2; RETENTION_COLLAPSE median eligible-source R^pair(k.5) < .30;
 SUBSAMPLE_UNSTABLE range of k.5 over 5 × 80 % draws ≥ 2; STABILITY_BY_COLLAPSE median neighbouring-k Spearman > .90 while median
 IQR(k30)/IQR(k0) < .5 or median scoreable(k30) < .8; SOURCE_EXCLUSION (catalogue-excluded basis) min cos θ(k=10) < .90 or |Δk.5| ≥ 2;
 NO_COMMON_COORDINATE (table) k.5 spread across systems ≥ 3 AND C^pair(5) spread ≥ .2. k.5 = first k with C^pair ≥ .5 (level table
 .3/.5/.7 reported for both weights)."""
import sys, os, json, time
import numpy as np, pandas as pd

R = __import__("os").environ["GEOBRIDGE_ROOT"]; sys.path.insert(0, f"{R}/scratch")
OUT = f"{R}/results/geometry_audit"; K_LIST = (20, 30, 50, 80); KB = 80; LEVELS = (.3, .5, .7); NT = 1000; GCH = 150
SYSTEMS = ("GATA1", "PANC", "BONEMARROW", "K562", "DENTATEGYRUS", "GASTRULATION", "ZEBRAFISH", "RPE1")
LABEL_COLS = ("celltype", "clusters", "cell_type", "cluster"); NPART = 20; RHO = (.10, .20, .30)


def pca(X, n, mean=None):
    mu = X.mean(0) if mean is None else mean; Xc = X - mu
    w, V = np.linalg.eigh(Xc.T @ Xc); V = V[:, ::-1][:, :n]
    return Xc @ V, V.T, mu


def cos_angles(A, B):
    qa, _ = np.linalg.qr(A); qb, _ = np.linalg.qr(B)
    return np.clip(np.linalg.svd(qa.T @ qb, compute_uv=False), 0, 1)


def qr_flag(M):
    Q, Rr = np.linalg.qr(M); d = np.abs(np.diag(Rr))
    return Q, [int(i + 1) for i in np.flatnonzero(d < 1e-10 * d[0])]


def weights(I, J, n):
    d = np.bincount(np.r_[I, J], minlength=n).astype(float); na = int((d > 0).sum())
    return dict(pair=np.full(len(I), 1.0 / len(I)), cell=(1 / d[I] + 1 / d[J]) / (2 * na))


def cw_curve(Q, Y, I, J, w, scale=None, chunk=GCH):
    """C^w(k) = 1 − Σ_p w_p||r_kp||² / Σ_p w_p||D_p||², k = 0..Q.shape[1], r_k = (I−Π_k)D (production projector, unweighted QR).
    Exact recursion S_k = S_{k−1} − 2 c_k·a_k + ||c_k||² M_kk, c = QᵀD, a_k = (Q∘w)_kᵀD − M[k,:k] c[:k], M = QᵀWQ."""
    K = Q.shape[1]; Qw = Q * w[:, None]; M = Q.T @ Qw; S = np.zeros(K + 1); S0 = 0.0
    for g0 in range(0, Y.shape[1], chunk):
        D = Y[J, g0:g0 + chunk] - Y[I, g0:g0 + chunk]
        if scale is not None:
            D = D / scale[g0:g0 + chunk]
        c = Q.T @ D; G = Qw.T @ D; s = float((w[:, None] * D * D).sum()); S0 += s; S[0] += s; run_ = s
        for k in range(K):
            a = G[k] - M[k, :k] @ c[:k]
            run_ = run_ - 2 * (c[k] * a).sum() + (c[k] ** 2).sum() * M[k, k]; S[k + 1] += run_
    return 1 - S / S0


def ret_curve(Q, Dsrc, w):
    """R^w_r(k) per source column (same exact recursion per column)"""
    K = Q.shape[1]; Qw = Q * w[:, None]; M = Q.T @ Qw; c = Q.T @ Dsrc; G = Qw.T @ Dsrc
    s0 = (w[:, None] * Dsrc ** 2).sum(0); out = np.empty((K + 1, Dsrc.shape[1])); run_ = s0.copy(); out[0] = run_
    for k in range(K):
        a = G[k] - M[k, :k] @ c[:k]; run_ = run_ - 2 * c[k] * a + c[k] ** 2 * M[k, k]; out[k + 1] = run_
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(s0 > 0, out / s0, np.nan)


def k_at(C, lev):
    ok = np.flatnonzero(C >= lev); return int(ok[0]) if len(ok) else -1


def cos2_pass(VA, VB, k):
    c2 = np.linalg.svd(VA[:k] @ VB[:k].T, compute_uv=False) ** 2
    return bool(c2.min() >= .80 and c2.mean() >= .90), float(c2.min()), float(c2.mean())


def band_B(Cs, K):
    return [k for k in range(1, K + 1) if all(.20 <= C[k] <= .50 for C in Cs)]


def load(system):
    import gb_geometry_matrix as GM
    S, A, Mu, Xs, V, P30 = GM.load_sys(system)
    f = np.load(f"{R}/results/boundary/RPE1/canon/RPE1_canonical_pairs.npz") if system == "RPE1" else np.load(S["pairs_file"])
    Ms = np.asarray(S["X"], np.float64); sd = Ms.std(0); gref = sd > 0
    det = np.asarray((A.layers["spliced"] > 0).mean(0)).ravel()
    lab = next((c for c in LABEL_COLS if c in A.obs.columns), None)
    return dict(S=S, X=Xs.astype(np.float64), Ms=Ms, Mu=Mu.astype(np.float64), sd=sd, gref=gref, det=det, P30=P30, f=f,
                labels=np.asarray(A.obs[lab]).astype(str) if lab else None, label_col=lab, genes=np.asarray(S["genes"]).astype(str),
                focal=list(np.load(f"{R}/results/geometry_matrix/{system}.npz", allow_pickle=True)["focal"].astype(str)))


def run(system):
    from gb_target_extension import project_scores
    from scipy.stats import spearmanr
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); D = load(system); X, Ms, n = D["X"], D["Ms"], D["X"].shape[0]
    gu = np.char.upper(D["genes"]); gi = {g: i for i, g in enumerate(gu)}; cat = np.asarray(D["S"]["is_tf"], bool)
    elig = np.flatnonzero(cat & (D["det"] >= .10) & (D["sd"] > 0)); esrc = list(gu[elig])
    Yref, scl = Ms[:, D["gref"]], D["sd"][D["gref"]]
    P80, V80, _ = pca(X, KB)
    js = dict(system=system, n_cells=n, n_gref=int(D["gref"].sum()), n_eligible_sources=len(elig), truth_read=False,
              parity_P30_cos_min=float(cos_angles(D["P30"], P80[:, :30]).min()))
    B80 = np.hstack([D["P30"], P80[:, 30:KB]]); out = {}
    js["basis"] = dict(n_genes=int(X.shape[1]), n_catalogue_tfs_in_basis=int(cat.sum()),
                       loading_energy_on_catalogue_TFs_top30=float((V80[:30][:, cat] ** 2).sum() / 30),
                       loading_energy_on_eligible_sources_top30=float((V80[:30][:, elig] ** 2).sum() / 30),
                       E2="P30 unchanged by E2: gb_target_extension.py:114 copies the V1 obsm X_pca; :160 reads the V1 fixture X_pca")
    Pex, _, _ = pca(X[:, ~cat], 30)
    ptc = {r: np.asarray(D["f"][f"{r}_pt"], float) for r in ("dc1_min", "dc1_max")}
    dec = np.digitize(ptc["dc1_min"], np.quantile(ptc["dc1_min"], np.linspace(.1, .9, 9)))
    strata = dec.astype(str) if D["labels"] is None else np.char.add(dec.astype(str), D["labels"])
    parts = []
    for p in range(NPART):
        rng = np.random.default_rng(1000 + p); lab = np.empty(n, int)
        for s_ in np.unique(strata):
            ix = rng.permutation(np.flatnonzero(strata == s_)); a, b = int(round(.4 * len(ix))), int(round(.8 * len(ix)))
            lab[ix[:a]] = 0; lab[ix[a:b]] = 1; lab[ix[b:]] = 2
        _, VA, muA = pca(X[lab == 0], KB); _, VB, muB = pca(X[lab == 1], KB)
        parts.append((lab, VA, muA, VB, muB))
    loadpass = np.array([[cos2_pass(VA, VB, k)[0] for k in range(1, 51)] for _, VA, _, VB, _ in parts])
    js["loading_block31_50_min_mean_cos2"] = [cos2_pass(VA[30:50], VB[30:50], 20)[1:] for _, VA, _, VB, _ in parts]
    R_load = [k for k in range(1, 51) if loadpass[:, k - 1].sum() >= 18]; js["R_loading"] = R_load; js["roots"] = {}
    absU = {}
    for root in ("dc1_min", "dc1_max"):
        I, J = D["S"]["pairs"][root]; r = {}; W = weights(I, J, n)
        Q30, rd30 = qr_flag(D["P30"][J] - D["P30"][I]); QB, rdB = qr_flag(B80[J] - B80[I])
        r["rank_deficient_prefixes"] = dict(P30=rd30, B80=rdB); r["nested_prefix_cos_min_k30"] = float(cos_angles(Q30, QB[:, :30]).min())
        C = {w: cw_curve(QB, Yref, I, J, W[w], scl) for w in W}
        for w in W:
            out[f"{root}|C_{w}"] = C[w]
        out[f"{root}|C_trivial"] = Ctriv = cw_curve(Q30, D["P30"], I, J, W["pair"])
        r.update(C_at={w: {k: float(C[w][k]) for k in (1, 3, 5, 10, 20, 30, 50, 80)} for w in W},
                 k_at={w: {str(l): k_at(C[w], l) for l in LEVELS} for w in W}, C_trivial_at5=float(Ctriv[5]),
                 kmax_gap_full={str(K): float(max(abs(C[w][80] - C[w][K]) for w in W)) for K in K_LIST},
                 cell_vs_pair_maxabs=float(np.abs(C["cell"] - C["pair"]).max()))
        pp, gaps, ends = [], [], []
        for lab, VA, muA, VB, muB in parts:
            m = (lab[I] == 2) & (lab[J] == 2); Ip, Jp = I[m], J[m]
            if m.sum() < 200:
                pp.append([False] * 50); continue
            PA = (X - muA) @ VA.T; PB = (X - muB) @ VB.T
            QA, _ = np.linalg.qr(PA[Jp][:, :50] - PA[Ip][:, :50]); QBb, _ = np.linalg.qr(PB[Jp][:, :50] - PB[Ip][:, :50])
            row = []
            for k in range(1, 51):
                c2 = np.linalg.svd(QA[:, :k].T @ QBb[:, :k], compute_uv=False) ** 2; row.append(bool(c2.min() >= .8 and c2.mean() >= .9))
            pp.append(row)
            Qp, _ = np.linalg.qr(PA[Jp] - PA[Ip]); Wp = weights(Ip, Jp, n)
            Cp = {w: cw_curve(Qp, Yref, Ip, Jp, Wp[w], scl) for w in Wp}
            gaps.append({str(K): max(abs(Cp[w][80] - Cp[w][K]) for w in Wp) for K in K_LIST})
            bb = band_B([Cp["pair"], Cp["cell"]], 30); ends.append((bb[0], bb[-1]) if bb else (-1, -1))
        pp = np.array(pp); R_probe = [k for k in range(1, 51) if pp[:, k - 1].sum() >= 18]
        r["R_probe"] = R_probe; r["d_rep"] = next((k - 1 for k in range(1, 51) if not (k in R_load and k in R_probe)), 50)
        r["kmax_gap_resplit_p90"] = {K: float(np.percentile([g[K] for g in gaps], 90)) for K in map(str, K_LIST)} if gaps else {}
        r["band_endpoints_resplit"] = ends
        br = {}
        if D["labels"] is not None:
            same = D["labels"][I] == D["labels"][J]
            for lb in np.unique(D["labels"][I][same]):
                mm = same & (D["labels"][I] == lb)
                if mm.sum() >= .08 * len(I):
                    Cb = cw_curve(Q30, Yref, I, J, mm.astype(float), scl)
                    br[lb] = dict(n_pairs=int(mm.sum()), k_at={str(l): k_at(Cb, l) for l in LEVELS}, C5=float(Cb[5]), C10=float(Cb[10]))
            r["branch_basis"] = f"obs['{D['label_col']}'] within-label pairs ≥8 %; global production projector"
        else:
            tb = np.digitize(ptc[root][I], np.quantile(ptc[root][I], [1 / 3, 2 / 3]))
            for b in range(3):
                Cb = cw_curve(Q30, Yref, I, J, (tb == b).astype(float), scl)
                br[f"dpt_tertile_{b}"] = dict(n_pairs=int((tb == b).sum()), k_at={str(l): k_at(Cb, l) for l in LEVELS}, C5=float(Cb[5]), C10=float(Cb[10]))
            r["branch_basis"] = "no truth-free branch label: DPT tertiles of the source cell (NOT branches)"
        r["branches"] = br
        QE, _ = qr_flag(Pex[J] - Pex[I]); CE = cw_curve(QE, Yref, I, J, W["pair"], scl)
        r["catalogue_excluded"] = dict(cos_min_k={str(k): float(cos_angles(D["P30"][:, :k], Pex[:, :k]).min()) for k in (5, 10, 20, 30)},
                                       k_at={str(l): k_at(CE, l) for l in LEVELS}, C5=float(CE[5]))
        ks = []
        for dr in range(5):
            rr = np.random.default_rng(200 + dr); keep = np.zeros(n, bool)
            for s_ in np.unique(dec):
                ix = np.flatnonzero(dec == s_); keep[rr.choice(ix, int(round(.8 * len(ix))), replace=False)] = True
            _, Vs, ms = pca(X[keep], 30); Pf = (X - ms) @ Vs.T; mm = keep[I] & keep[J]
            Qs, _ = np.linalg.qr(Pf[J[mm]] - Pf[I[mm]]); Cs = cw_curve(Qs, Yref, I[mm], J[mm], np.ones(int(mm.sum())), scl)
            ks.append({str(l): k_at(Cs, l) for l in LEVELS})
        r["subsample80_k_at"] = ks
        Dsrc = Ms[J][:, elig] - Ms[I][:, elig]
        Ret = {w: ret_curve(Q30, Dsrc, W[w]) for w in W}; out[f"{root}|Ret_pair"] = Ret["pair"]; out[f"{root}|Ret_cell"] = Ret["cell"]
        foc_ix = [esrc.index(s) for s in D["focal"] if s in esrc]
        r["retention_median_eligible"] = {k: float(np.nanmedian(Ret["pair"][k])) for k in (1, 3, 5, 10, 20, 30)}
        r["retention_median_focal"] = {k: float(np.nanmedian(Ret["pair"][k][foc_ix])) for k in (1, 3, 5, 10, 20, 30)}
        rng = np.random.default_rng(7); tpool = np.setdiff1d(np.arange(len(gu)), elig); tg = np.sort(rng.choice(tpool, min(NT, len(tpool)), replace=False))
        dx = Dsrc.astype(np.float32); A_ = np.full((31, len(elig), len(tg)), np.nan, np.float32); Dn = A_.copy(); Z2 = A_.copy()
        for g0 in range(0, len(tg), 250):
            sl = tg[g0:g0 + 250]; y = (D["Mu"][J][:, sl] - D["Mu"][I][:, sl]).astype(np.float32)
            for k, num, den, z2, a in project_scores(dx, y, Q30, kmax=30):
                A_[k, :, g0:g0 + len(sl)] = a; Dn[k, :, g0:g0 + len(sl)] = den; Z2[k, :, g0:g0 + len(sl)] = z2
        absU[root] = A_
        with np.errstate(invalid="ignore", divide="ignore"):
            PR = Dn.astype(np.float64) ** 2 / Z2
        out[f"{root}|scoreable"] = np.isfinite(A_).mean(2); out[f"{root}|den_med"] = np.nanmedian(Dn, 2); out[f"{root}|PR_med"] = np.nanmedian(PR, 2)
        out[f"{root}|iqr"] = np.nanpercentile(A_, 75, axis=2) - np.nanpercentile(A_, 25, axis=2)
        out[f"{root}|nb_spearman"] = np.array([[spearmanr(A_[k, s], A_[k + 1, s], nan_policy="omit")[0] for k in range(30)] for s in range(len(elig))])
        out[f"{root}|kxk_focal"] = np.array([[[spearmanr(A_[a1, s], A_[a2, s], nan_policy="omit")[0] for a2 in range(31)] for a1 in range(31)] for s in foc_ix])
        js["roots"][root] = r
        print(f"{system} {root}: k.3/.5/.7 pair {r['k_at']['pair']} cell {r['k_at']['cell']} C(30) {C['pair'][30]:.3f} C(80) {C['pair'][80]:.3f} "
              f"d_rep {r['d_rep']} {time.time()-t0:.0f}s", flush=True)
    out["root_disagree"] = np.array([[1 - spearmanr(absU["dc1_min"][k, s], absU["dc1_max"][k, s], nan_policy="omit")[0] for s in range(len(elig))] for k in range(31)])
    rr = js["roots"]; Rrep = sorted(set(R_load) & set(rr["dc1_min"]["R_probe"]) & set(rr["dc1_max"]["R_probe"]))
    for Kc in (30, 80):
        BA = list(range(1, 6)); BB = band_B([out[f"{h}|C_{w}"] for h in rr for w in ("pair", "cell")], Kc); BC = [k for k in BB if k in Rrep]
        bands = {}
        for nm, Bd in (("A_fixed_1to5", BA), ("B_operator_energy", BB), ("C_reproducible", BC)):
            if len(Bd) < 2:
                bands[nm] = dict(band=Bd, status="BAND_UNRESOLVED"); continue
            kk = [k for k in Bd if k <= 30]
            mins = np.nanmin(np.stack([out[f"{h}|Ret_{w}"][kk] for h in rr for w in ("pair", "cell")]).reshape(-1, len(elig)), 0)
            bands[nm] = dict(band=Bd, status="OK", retention_eval_depths=kk,
                             unsupported={str(rho): [esrc[i] for i in np.flatnonzero(~(mins >= rho))] for rho in RHO},
                             n_unsupported={str(rho): int((~(mins >= rho)).sum()) for rho in RHO},
                             focal_unsupported_rho20=[s for s in D["focal"] if s in esrc and not (mins[esrc.index(s)] >= .20)])
        js[f"bands_K{Kc}"] = bands
    js.update(R_reproducible=Rrep, eligible_sources=esrc, focal=D["focal"], targets_sampled=NT, wall_s=round(time.time() - t0),
              script_sha256=__import__("hashlib").sha256(open(__file__, "rb").read()).hexdigest())
    js["flags"] = flags(js, out)
    np.savez_compressed(f"{OUT}/{system}.npz", **out); json.dump(js, open(f"{OUT}/{system}.json", "w"), indent=1, default=float)
    print(system, "FLAGS", js["flags"], "bands K30", {k: (v["band"], v["status"], v.get("n_unsupported")) for k, v in js["bands_K30"].items()},
          f"{time.time()-t0:.0f}s", flush=True)


def flags(js, out):
    F = []; rr = js["roots"]
    full = max(r["kmax_gap_full"]["30"] for r in rr.values()); rs = max((r["kmax_gap_resplit_p90"].get("30", 0) for r in rr.values()), default=0)
    if full > .02 or rs > .02: F.append(f"BASIS_LIMITED(K30: full {full:.3f}, resplit p90 {rs:.3f})")
    for h, r in rr.items():
        if abs(r["k_at"]["cell"]["0.5"] - r["k_at"]["pair"]["0.5"]) >= 2 or r["cell_vs_pair_maxabs"] > .05: F.append(f"WEIGHTING_DEPENDENT[{h}]")
        k5 = max(r["k_at"]["pair"]["0.5"], 0)
        if np.nanmedian(out[f"{h}|Ret_pair"][min(k5, 30)]) < .30: F.append(f"RETENTION_COLLAPSE[{h}]")
        ks = [d["0.5"] for d in r["subsample80_k_at"]]
        if max(ks) - min(ks) >= 2: F.append(f"SUBSAMPLE_UNSTABLE[{h}]")
        nb = np.nanmedian(out[f"{h}|nb_spearman"]); iq = out[f"{h}|iqr"]
        with np.errstate(invalid="ignore", divide="ignore"):
            if nb > .90 and (np.nanmedian(iq[30] / iq[0]) < .5 or np.nanmedian(out[f"{h}|scoreable"][30]) < .8): F.append(f"STABILITY_BY_COLLAPSE[{h}]")
        ce = r["catalogue_excluded"]
        if ce["cos_min_k"]["10"] < .90 or abs(ce["k_at"]["0.5"] - r["k_at"]["pair"]["0.5"]) >= 2: F.append(f"SOURCE_EXCLUSION[{h}]")
    a, b = rr["dc1_min"], rr["dc1_max"]
    if abs(a["k_at"]["pair"]["0.5"] - b["k_at"]["pair"]["0.5"]) >= 2 or abs(a["d_rep"] - b["d_rep"]) >= 2: F.append("ROOT_INCOMPATIBLE")
    return F


def table():
    rows = []
    for s in SYSTEMS:
        p = f"{OUT}/{s}.json"
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        for h, r in j["roots"].items():
            b = j["bands_K30"]
            rows.append(dict(system=s, root=h, **{f"k{l}_{w}": r["k_at"][w][str(l)] for w in ("pair", "cell") for l in LEVELS},
                             C5=r["C_at"]["pair"]["5"], C10=r["C_at"]["pair"]["10"], C30=r["C_at"]["pair"]["30"], C80=r["C_at"]["pair"]["80"],
                             gap30_full=r["kmax_gap_full"]["30"], gap30_resplit90=r["kmax_gap_resplit_p90"].get("30"), d_rep=r["d_rep"],
                             Ret_elig_k5=r["retention_median_eligible"]["5"], Ret_focal_k5=r["retention_median_focal"]["5"],
                             bandB=str(b["B_operator_energy"]["band"]), bandC=str(b["C_reproducible"]["band"]),
                             unsup_rho20_A_B_C="/".join(str(b[x].get("n_unsupported", {}).get("0.2", "UNRES")) for x in ("A_fixed_1to5", "B_operator_energy", "C_reproducible"))
                             + f" of {j['n_eligible_sources']}", flags=";".join(j["flags"])))
    T = pd.DataFrame(rows); T.to_csv(f"{OUT}/TABLE.tsv", sep="\t", index=False)
    k5 = T.groupby("system")["k0.5_pair"].mean(); c5 = T.groupby("system").C5.mean()
    pd.set_option("display.width", 320); print(T.round(3).to_string(index=False))
    print("NO_COMMON_COORDINATE" if (k5.max() - k5.min() >= 3 and c5.max() - c5.min() >= .2) else "common coordinate not rejected",
          f"(k.5 spread {k5.max()-k5.min():.1f}; C(5) spread {c5.max()-c5.min():.2f})")


def selftest():
    rng = np.random.default_rng(3); n = 400
    Y = rng.normal(size=(n, 30)); I = rng.integers(0, n, 3000); J = rng.integers(0, n, 3000); Q, _ = np.linalg.qr(rng.normal(size=(3000, 6)))
    for w in (np.ones(3000), rng.random(3000)):
        C = cw_curve(Q, Y, I, J, w, chunk=7); Dm = Y[J] - Y[I]
        ref = [1 - (w[:, None] * (Dm - Q[:, :k] @ (Q[:, :k].T @ Dm)) ** 2).sum() / (w[:, None] * Dm ** 2).sum() for k in range(7)]
        assert np.allclose(C, ref, atol=1e-10), (C, ref)
        Rt = ret_curve(Q, Dm[:, :3], w)
        rf = [(w[:, None] * (Dm[:, :3] - Q[:, :k] @ (Q[:, :k].T @ Dm[:, :3])) ** 2).sum(0) / (w[:, None] * Dm[:, :3] ** 2).sum(0) for k in range(7)]
        assert np.allclose(Rt, rf, atol=1e-10)
    Z = rng.normal(size=(n, 3)); Xs = Z @ rng.normal(size=(3, 40)) + .01 * rng.normal(size=(n, 40)); P, V, _ = pca(Xs, 5)
    assert cos2_pass(V, V, 5)[0] and cos_angles(P[:, :3], Z - Z.mean(0)).min() > .999
    w_ = weights(np.array([0, 0, 1]), np.array([1, 2, 2]), 3); assert np.isclose(w_["cell"].sum(), .5)
    assert band_B([np.r_[0, .1, .25, .4, .6]], 4) == [2, 3]
    print("gb_geometry_audit selftest PASS (exact weighted residual recursion; retention; PCA; weights; band)")


if __name__ == "__main__":
    c = sys.argv[1]
    selftest() if c == "selftest" else table() if c == "table" else run(c.upper())
