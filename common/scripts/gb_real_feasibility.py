#!/usr/bin/env python
# Truth-blind real-data feasibility
import hashlib, json, math, os, sys, time
import numpy as np, scipy.sparse as sp, anndata as ad, pandas as pd
from sklearn.decomposition import PCA

ROOT = __import__("os").environ["GEOBRIDGE_ROOT"]
CFG = dict(
    PANC=dict(raw=f"{ROOT}/data/Pancreas/endocrinogenesis_day15.h5ad", sub=f"{ROOT}/real_1_Pancreas/cache/pancreas_processed__default.h5ad",
              focal="Rfx6", wt=None),
    GATA1=dict(raw=f"{ROOT}/results/canonical_velocity_builder_fixture_2026_08_15/anchor_rebuild_input.h5ad",
               sub=f"{ROOT}/results/published_reference_fixture_2026_08_16/published_reference_out.h5ad", focal="Gata1", wt="WT"))
LAMBERT = f"{ROOT}/scratch/capacity_verify/lambert_TF_v1.01.csv"
HOM = f"{ROOT}/data/orthology/HOM_MouseHumanSequence.rpt"
SEED, NANCH, NPC, NRIV, BOOT = 20260913, 50, 10, 8, 200
POS_CTRL = ["Mki67", "Top2a", "Cdk1", "Hmgb2", "Ccnb1", "Hist1h2ap"]   # within-state-variable (cell cycle) power controls; not scored
FORBIDDEN = ("label", "_de.", "chip", "truth", "aldous")


def fsha(p, n=16):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""): h.update(b)
    return h.hexdigest()[:n]


def mouse_catalogue():
    lam = pd.read_csv(LAMBERT)
    human = set(lam.loc[lam["Is TF?"] == "Yes", "HGNC symbol"])
    hom = pd.read_csv(HOM, sep="\t", usecols=["DB Class Key", "Common Organism Name", "Symbol"])
    grp = hom.groupby("DB Class Key")
    out = set()
    for _, g in grp:
        h = g.loc[g["Common Organism Name"] == "human", "Symbol"]; m = g.loc[g["Common Organism Name"] == "mouse, laboratory", "Symbol"]
        if len(h) == 1 and len(m) == 1 and h.iloc[0] in human: out.add(m.iloc[0])     # one-to-one orthologs only
    return out


def q2_basis(D, tol=1e-10):
    k = D.shape[1]
    cols = [np.ones(len(D))] + [D[:, j] for j in range(k)] + [D[:, i] * D[:, j] for i in range(k) for j in range(i, k)]
    Z = np.column_stack(cols)
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, s > tol * s.max()]


def resid(Q, Y):
    return Y - Q @ (Q.T @ Y)


def ess(r):
    r2 = r ** 2; return float(r2.sum() ** 2 / max((r2 ** 2).sum(), 1e-300))


# column-wise split-half residual correlation and one-sided cell-bootstrap lower bound (percentile)
def split_corr_lb(R1, R2, rng, alpha):
    def corr(a, b):
        a = a - a.mean(0); b = b - b.mean(0)
        return (a * b).sum(0) / np.sqrt((a * a).sum(0) * (b * b).sum(0) + 1e-300)
    c = corr(R1, R2)
    idx = rng.integers(0, len(R1), size=(BOOT, len(R1)))
    bs = np.stack([corr(R1[i], R2[i]) for i in idx])
    return c, np.quantile(bs, alpha, axis=0)


def twonn(X, rng, n=2000):
    X = X[rng.choice(len(X), min(n, len(X)), replace=False)]
    D = np.sqrt(((X[:, None] - X[None]) ** 2).sum(2)); np.fill_diagonal(D, np.inf)
    s = np.sort(D, 1)[:, :2]; mu = s[:, 1] / np.maximum(s[:, 0], 1e-12); mu = mu[np.isfinite(mu) & (mu > 1)]
    return float(len(mu) / np.log(mu).sum())


def main(outdir, ds, max_src=None):
    t0 = time.time(); cfg = CFG[ds]
    for p in (cfg["raw"], cfg["sub"]): assert not any(f in os.path.basename(p).lower() for f in FORBIDDEN), p
    raw = ad.read_h5ad(cfg["raw"]); sub = ad.read_h5ad(cfg["sub"], backed="r")
    if cfg["wt"]: raw = raw[raw.obs["genotype"] == cfg["wt"]].copy()
    assert list(raw.obs_names) == list(sub.obs_names), "raw control cells must equal production substrate cells, same order"
    n = raw.n_obs; m = math.ceil(4 * math.sqrt(n))
    S = sp.csr_matrix(raw.layers["spliced"]).astype(np.float64)
    assert np.allclose(S.data, np.round(S.data)), "raw spliced must be integer"
    lib = np.asarray(S.sum(1)).ravel()
    cat = mouse_catalogue(); names = np.array(raw.var_names)
    # geometry: production HVG axis minus catalogue TFs, log CP10k, PCA10 on half A / half B
    feat = [g for g in sub.var_names if g not in cat and g in set(names)]
    fi = pd.Index(names).get_indexer(feat)
    Xf = np.log1p(S[:, fi].toarray() / lib[:, None] * 1e4)
    rng = np.random.default_rng(SEED)
    halfA = np.zeros(n, bool); halfA[rng.permutation(n)[: n // 2]] = True
    def pca_fit(mask):
        mu, sd = Xf[mask].mean(0), Xf[mask].std(0) + 1e-8
        return PCA(NPC, random_state=SEED).fit((Xf[mask] - mu) / sd).transform((Xf - mu) / sd)
    WA, WB = pca_fit(halfA), pca_fit(~halfA)
    anchors = np.sort(np.random.default_rng(SEED + 1).choice(n, NANCH, replace=False))
    def nbhd(W):
        Ws = (W - W.mean(0)) / (W.std(0) + 1e-12); out = []
        for a in anchors:
            d = ((Ws - Ws[a]) ** 2).sum(1); d[a] = np.inf; out.append(np.argsort(d, kind="stable")[:m])
        return out, Ws
    NA, WsA = nbhd(WA); NB, _ = nbhd(WB)
    jac = [len(set(a) & set(b)) / len(set(a) | set(b)) for a, b in zip(NA, NB)]
    dall = np.sqrt(((WsA[rng.choice(n, 500)] [:, None] - WsA[rng.choice(n, 500)][None]) ** 2).sum(2))
    radius = [float(np.sqrt(((WsA[N] - WsA[a]) ** 2).sum(1)).max()) for a, N in zip(anchors, NA)]
    geometry = dict(n_cells=n, m=m, n_features=len(feat), jaccard_median=float(np.median(jac)), jaccard_q10=float(np.quantile(jac, .1)),
                    anchors_jaccard_ge_05=int(np.sum(np.array(jac) >= .5)), twonn_dim=twonn(WsA, rng),
                    radius_median=float(np.median(radius)), dist_concentration=float(np.median(radius) / np.median(dall)),
                    pca_var_explained_halfA=None)
    # sources: catalogue TFs with >=1% global detection + focal
    det_g = np.asarray((S > 0).mean(0)).ravel()
    src = sorted({g for g in names[det_g >= .01] if g in cat} | {cfg["focal"]})
    if max_src: src = [cfg["focal"]] + [g for g in src if g != cfg["focal"]][: int(max_src) - 1]
    ctrl = [g for g in POS_CTRL if g in set(names) and g not in src]; src = src + ctrl
    si = pd.Index(names).get_indexer(src); assert (si >= 0).all()
    Xs = S[:, si].toarray()
    # Seed thinning by gene name so the split is independent of roster order.
    X1 = np.column_stack([np.random.default_rng([SEED + 2, int(hashlib.sha256(g.encode()).hexdigest()[:8], 16)]).binomial(Xs[:, j].astype(np.int64), 0.5)
                          for j, g in enumerate(src)]).astype(float); X2 = Xs - X1
    A = np.log1p(Xs / lib[:, None] * 1e4); A1 = np.log1p(X1 / (lib[:, None] / 2) * 1e4); A2 = np.log1p(X2 / (lib[:, None] / 2) * 1e4)
    alpha = 0.05 / NANCH
    K = len(src); fcol = src.index(cfg["focal"])
    per = {k: np.full((NANCH, K), np.nan) for k in ("det", "ndet", "c", "lb", "infoess", "maxlev", "npairs", "pair_ess", "frac_above_noise", "dA_med")}
    rival = []; brng = np.random.default_rng(SEED + 3); pair_len = []
    for ai, (a, N) in enumerate(zip(anchors, NA)):
        D = WsA[N] - WsA[a]; Q = q2_basis(D); H0 = (Q ** 2).sum(1)
        per["det"][ai] = (Xs[N] > 0).mean(0); per["ndet"][ai] = (Xs[N] > 0).sum(0)
        R, R1, R2 = resid(Q, A[N]), resid(Q, A1[N]), resid(Q, A2[N])
        c, lb = split_corr_lb(R1, R2, brng, alpha); per["c"][ai] = c; per["lb"][ai] = lb
        nr = np.sqrt((R ** 2).sum(0)) + 1e-300
        per["infoess"][ai] = [ess(R[:, j]) for j in range(K)]
        per["maxlev"][ai] = (H0[:, None] + (R / nr) ** 2).max(0)
        pairs = []
        free = np.ones(m, bool); D2 = ((D[:, None] - D[None]) ** 2).sum(2); np.fill_diagonal(D2, np.inf)
        for i in range(m):                                                     # Greedy matching on geometry features.
            if not free[i]: continue
            dd = np.where(free, D2[i], np.inf); dd[i] = np.inf; j = int(np.argmin(dd))
            if not np.isfinite(dd[j]): break
            pairs.append((i, j)); free[i] = free[j] = False
        P = np.array(pairs); pair_len += np.sqrt(D2[P[:, 0], P[:, 1]]).tolist()
        dA = A[N][P[:, 0]] - A[N][P[:, 1]]
        noise_sd = np.sqrt(2 * ((A1[N] - A2[N]) ** 2).mean(0) / 4) + 1e-12
        per["npairs"][ai] = len(P); per["pair_ess"][ai] = [ess(dA[:, j]) for j in range(K)]
        per["frac_above_noise"][ai] = (np.abs(dA) > 2 * noise_sd).mean(0); per["dA_med"][ai] = np.median(np.abs(dA), 0)
        elig = [j for j in range(K) if j != fcol and src[j] not in ctrl and per["det"][ai, j] >= .10 and per["ndet"][ai, j] >= 20]
        rr = sorted(elig, key=lambda j: (-abs(np.corrcoef(R[:, fcol], R[:, j])[0, 1]) if R[:, j].std() > 0 else 0, src[j]))[:NRIV]
        if rr:
            U_, s_, _ = np.linalg.svd(np.column_stack([Q, A[N][:, rr]]), full_matrices=False); Qc = U_[:, s_ > 1e-10 * s_.max()]
        else:
            Qc = Q
        Rc, Rc1, Rc2 = resid(Qc, A[N][:, [fcol]]), resid(Qc, A1[N][:, [fcol]]), resid(Qc, A2[N][:, [fcol]])
        cc, lbc = split_corr_lb(Rc1, Rc2, brng, alpha)
        rival.append(dict(anchor=int(a), rivals=[src[j] for j in rr], n_rivals=len(rr), resid_df=int(m - Qc.shape[1] - 1),
                          infoess=ess(Rc[:, 0]), maxlev=float(((Qc ** 2).sum(1) + Rc[:, 0] ** 2 / ((Rc ** 2).sum() + 1e-300)).max()),
                          c=float(cc[0]), lb=float(lbc[0]),
                          resid_var_frac=float((Rc ** 2).sum() / ((R[:, fcol] ** 2).sum() + 1e-300))))
    resid_df = int(m - q2_basis(WsA[NA[0]] - WsA[anchors[0]]).shape[1] - 1)
    qual = lambda j: ((per["det"][:, j] >= .10) & (per["ndet"][:, j] >= 20) & (per["lb"][:, j] > .10))
    e2ok = lambda j: (per["infoess"][:, j] >= 30) & (per["maxlev"][:, j] <= .5) & (resid_df >= 50)
    pairok = lambda j: (per["npairs"][:, j] >= 30) & (per["pair_ess"][:, j] >= 30)
    summ = {}
    for j, g in enumerate(src):
        summ[g] = dict(global_det=float(det_g[si[j]]), det_median=float(np.median(per["det"][:, j])),
                       split_corr_median=float(np.nanmedian(per["c"][:, j])), n_anchor_source_qualify=int(qual(j).sum()),
                       n_anchor_e2_ok=int((qual(j) & e2ok(j)).sum()), n_anchor_pair_ok=int((qual(j) & pairok(j)).sum()),
                       frac_above_noise_median=float(np.median(per["frac_above_noise"][:, j])),
                       n_anchor_det10=int((per["det"][:, j] >= .10).sum()),
                       spearman_brown_full_median=float(np.nanmedian(2 * per["c"][:, j] / (1 + np.abs(per["c"][:, j])))), infoess_median=float(np.median(per["infoess"][:, j])))
    ranks = np.array([summ[g]["n_anchor_source_qualify"] for g in src if g not in ctrl])
    focal = dict(summary=summ[cfg["focal"]], per_anchor={k: v[:, fcol].tolist() for k, v in per.items()},
                 catalogue_percentile_of_qualifying_anchors=float((ranks < summ[cfg["focal"]]["n_anchor_source_qualify"]).mean()),
                 rival_conditioned=dict(n_anchor_qualify=int(sum((r["lb"] > .10) and (r["infoess"] >= 30) and (r["maxlev"] <= .5) and (r["resid_df"] >= 50)
                                                                 for r, ok in zip(rival, qual(fcol)) if ok)), per_anchor=rival))
    early_kill = dict(source_route_to_lane_M=bool(summ[cfg["focal"]]["n_anchor_source_qualify"] < 10),
                      geometry_unstable=bool(geometry["jaccard_median"] < .5), status="UNCALIBRATED_DIAGNOSTIC")
    out = dict(dataset=ds, truth_blind=True, n_sources=K, resid_df_q2=resid_df, geometry=geometry, pair_length_quantiles=np.quantile(pair_len, [.1, .5, .9]).tolist(),
               focal=focal, early_kill=early_kill,
               catalogue_n_sources_ge10_anchors=int((ranks >= 10).sum()), catalogue_qualify_quantiles=np.quantile(ranks, [.1, .5, .9]).tolist(),
               sources=summ, power_controls={g: summ[g] for g in ctrl}, config=dict(SEED=SEED, NANCH=NANCH, NPC=NPC, NRIV=NRIV, BOOT=BOOT, alpha=alpha, max_src=max_src),
               inputs=dict(raw=cfg["raw"], raw_sha=fsha(cfg["raw"]), sub=cfg["sub"], sub_sha=fsha(cfg["sub"]), lambert_sha=fsha(LAMBERT), hom_sha=fsha(HOM)),
               code_sha=fsha(__file__), runtime_s=time.time() - t0)
    os.makedirs(outdir, exist_ok=True)
    json.dump(out, open(f"{outdir}/feasibility_{ds}{'_smoke' if max_src else ''}.json", "x"), indent=1)
    s = summ[cfg["focal"]]
    print(f"{ds} n={n} m={m} df={resid_df} srcs={K} jaccard={geometry['jaccard_median']:.2f} twonn={geometry['twonn_dim']:.1f} | focal {cfg['focal']}: "
          f"det {s['det_median']:.3f} splitc {s['split_corr_median']:.3f} qualify {s['n_anchor_source_qualify']}/50 e2ok {s['n_anchor_e2_ok']} pairok {s['n_anchor_pair_ok']} "
          f"rivalcond {focal['rival_conditioned']['n_anchor_qualify']} | catalogue median qualify {np.median(ranks):.0f}, >=10 anchors {out['catalogue_n_sources_ge10_anchors']} | {out['runtime_s']:.0f}s")
    print("power controls:", {g: (round(v["det_median"], 2), round(v["split_corr_median"], 3), v["n_anchor_source_qualify"]) for g, v in out["power_controls"].items()})


def selftest():
    rng = np.random.default_rng(0)
    D = rng.normal(size=(300, 3)); Q = q2_basis(D)
    y = 1 + D[:, 0] - 2 * D[:, 1] * D[:, 2] + D[:, 2] ** 2
    assert np.allclose(resid(Q, y[:, None]), 0, atol=1e-8)                                     # quadratic nuisance removed exactly
    z = rng.normal(size=(300, 1)); R1 = z + .3 * rng.normal(size=(300, 1)); R2 = z + .3 * rng.normal(size=(300, 1))
    c, lb = split_corr_lb(np.column_stack([R1, rng.normal(size=(300, 1))]), np.column_stack([R2, rng.normal(size=(300, 1))]), rng, .001)
    assert c[0] > .8 and lb[0] > .7 and abs(c[1]) < .2 and lb[1] < .1                        # shared signal vs independent noise
    x = rng.poisson(3, size=(5000, 1)); x1 = np.random.default_rng(1).binomial(x, .5)
    assert abs(np.corrcoef(x1[:, 0], (x - x1)[:, 0])[0, 1]) < .05                            # Poisson thinning -> independent halves
    assert 1.6 < twonn(rng.normal(size=(3000, 2)), rng) < 2.5
    print("feasibility selftest PASS")


if __name__ == "__main__":
    if sys.argv[1] == "--selftest": selftest()
    else: main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
