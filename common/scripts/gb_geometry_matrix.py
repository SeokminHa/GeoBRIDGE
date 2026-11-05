#!/usr/bin/env python
"""TRUTH-FREE geometry / ranking-stability matrix for every canonical real substrate (results doc §180 directive). Reads NO truth.
usage: gb_geometry_matrix.py <SYSTEM> [--threads 24]      -> results/geometry_matrix/<SYS>.npz + <SYS>.tsv
       gb_geometry_matrix.py selftest
       gb_geometry_matrix.py merge                        -> results/geometry_matrix/MATRIX.tsv
Pair space = EXACTLY the conditioning operator's coordinates (gb_target_extension.project_scores): dP = P30[J] - P30[I], Qf = QR(dP).
Per system x root: singular spectrum of dP, C(k), participation r_eff = (sum s^2)^2/sum s^4, entropy r_eff, Gavish-Donoho rank on dP;
FULL-GENE pair differences of the state matrix (.X, subsampled pairs) -> spectrum, C_full(k), parallel-analysis rank (each gene column
permuted independently across pairs, B=50, 95th pct) and GD rank — the PA is done in full-gene space because the 30 PC columns are
already (near-)decorrelated, so a column-permutation null in PC space reproduces the observed spectrum and carries no information.
Per source (Ms source layer, Mu targets, production scorer): Q_r(k), E^U = |S_U| target ranking stability across neighbouring k (Spearman,
top-20/50 Jaccard), Jaccard vs the v1 summary (mean within-source percentile over k 8..30, min over roots), root agreement per k.
Velocity coherence per root = median over pairs of cos(dMs, V[I])."""
import sys, os, json, time, argparse
import numpy as np, pandas as pd

R = __import__("os").environ["GEOBRIDGE_ROOT"]; sys.path.insert(0, f"{R}/scratch")
OUT = f"{R}/results/geometry_matrix"; KMAX = 30; KADM = list(range(8, 31)); B_PA = 50; NSUB = 20000; CHUNK = 800
FOCAL = {"GATA1": ["GATA1", "KLF1"],
         "PANC": ["RFX6", "NKX2-2", "PDX1", "SOX9", "NEUROD1", "NEUROG3", "PAX6", "INSM1", "ISL1", "ARX", "MAFB", "HNF1B", "FOXA2",
                  "HES1", "NKX6-1", "PAX4", "MNX1", "FEV"],
         "BONEMARROW": ["SPI1", "GATA1", "RUNX1", "TAL1", "KLF1", "MYB", "NFE2", "FOSL1", "BCL11A"],
         "DENTATEGYRUS": ["TCF4", "PROX1"],
         "GASTRULATION": ["T", "MESP1", "FOXA2", "SOX17", "GATA6", "TAL1", "GATA1"],
         "K562": ["GATA1", "MYC", "MAX", "NRF1", "YY1", "GABPA"],
         "ZEBRAFISH": ["MITFA", "TFAP2C", "TFAP2B"],
         "RPE1": ["CLOCK", "SON", "SMARCB1"]}


def gd_omega(beta):
    return 0.56 * beta ** 3 - 0.95 * beta ** 2 + 1.82 * beta + 1.43


def gd_rank(s, m, n):
    """Gavish-Donoho optimal hard threshold, unknown noise; s singular values of an m x n matrix (m >= n)"""
    beta = min(m, n) / max(m, n)
    return int((s > gd_omega(beta) * np.median(s)).sum())


def spectrum_stats(s):
    e = s ** 2; tot = e.sum(); C = np.r_[0, np.cumsum(e) / tot]
    p = e / tot; p = p[p > 0]
    return dict(C=C, reff=float(tot ** 2 / (e ** 2).sum()), reff_ent=float(np.exp(-(p * np.log(p)).sum())))


def pa_rank(D, B=B_PA, seed=0, ntop=60):
    """parallel analysis: leading eigenvalues of D^T D above the 95th pct of column-permuted nulls (consecutive from the top)"""
    from scipy.sparse.linalg import eigsh
    rng = np.random.default_rng(seed); ntop = min(ntop, D.shape[1] - 2)
    top = lambda M: np.sort(eigsh(M.T @ M, k=ntop, which="LA", return_eigenvectors=False))[::-1]
    obs = top(D); null = np.empty((B, ntop))
    for b in range(B):
        null[b] = top(np.column_stack([rng.permutation(D[:, j]) for j in range(D.shape[1])]))
    thr = np.percentile(null, 95, axis=0); above = obs > thr
    return int(np.argmin(above) if not above.all() else ntop), obs, thr


def load_sys(system):
    import anndata as ad
    from gb_substrate import load, d_
    if system == "RPE1":                        # rebuilt canonical pairs (boundary lane); the CANON copy predates the pair repair
        C = f"{R}/results/boundary/RPE1/canon"; A = ad.read_h5ad(f"{C}/RPE1_canonical.h5ad"); f = np.load(f"{C}/RPE1_canonical_pairs.npz")
        A = A[:, [str(g) for g in f["genes"]]].copy()
        cat = {l.strip().upper() for l in open(f"{R}/GB_real_v2/lambert2018_human_tfs.txt") if l.strip()}
        genes = np.asarray(A.var_names).astype(str)
        S = dict(genes=genes, X=d_(A.layers["Ms"]).astype(np.float64), is_tf=np.isin(np.char.upper(genes), list(cat)),
                 pairs={r: (f[f"{r}_I"].astype(np.int64), f[f"{r}_J"].astype(np.int64)) for r in ("dc1_min", "dc1_max")})
    else:
        S = load(system, canonical=True); A = ad.read_h5ad(S["fixture"]); A = A[:, list(S["genes"])].copy()
    return S, A, d_(A.layers["Mu"]).astype(np.float32), d_(A.X).astype(np.float32), \
        np.nan_to_num(d_(A.layers["velocity"]).astype(np.float64)), np.asarray(A.obsm["X_pca"], float)[:, :30]


def pct(v):
    o = np.full(v.shape, np.nan); ok = np.isfinite(v)
    o[ok] = pd.Series(v[ok]).rank(method="average").to_numpy() / ok.sum(); return o


def jac(a, b, K):
    A, B_ = set(np.argsort(-np.nan_to_num(a, nan=-1))[:K]), set(np.argsort(-np.nan_to_num(b, nan=-1))[:K])
    return len(A & B_) / len(A | B_)


def run(system):
    from gb_target_extension import project_scores
    from scipy.stats import spearmanr
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    S, A, Mu, Xs, V, P30 = load_sys(system)
    genes = np.asarray(S["genes"]).astype(str); gu = np.char.upper(genes); gi = {g: i for i, g in enumerate(gu)}
    foc = [t for t in FOCAL[system] if t in gi]; miss = [t for t in FOCAL[system] if t not in gi]
    rng = np.random.default_rng(0)
    pool = sorted(set(gu[S["is_tf"]]) - set(foc))
    rnd = list(rng.choice(pool, size=min(20, len(pool)), replace=False))
    srcs = foc + rnd; si = [gi[t] for t in srcs]; Xsrc = np.asarray(S["X"], np.float64)[:, si]
    uni = ~np.isin(gu, srcs); tg = np.flatnonzero(uni)
    res = dict(system=system, sources=np.array(srcs), focal=np.array(foc), missing_focal=np.array(miss), genes_universe=genes[tg])
    absS = {}
    rows = []
    for root in ("dc1_min", "dc1_max"):
        I, J = S["pairs"][root]
        dP = P30[J] - P30[I]; Qf, _ = np.linalg.qr(dP)
        s = np.linalg.svd(dP, compute_uv=False); st = spectrum_stats(s)
        sub = rng.choice(len(I), size=min(NSUB, len(I)), replace=False)
        Dfull = (Xs[J[sub]] - Xs[I[sub]]).astype(np.float64); Dfull -= Dfull.mean(0)
        sf = np.sqrt(np.clip(np.linalg.eigvalsh(Dfull.T @ Dfull)[::-1], 0, None)); stf = spectrum_stats(sf)
        kpa, pa_obs, pa_thr = pa_rank(Dfull)
        dM = np.asarray(S["X"], np.float64)[J[sub]] - np.asarray(S["X"], np.float64)[I[sub]]; Vi = V[I[sub]]
        with np.errstate(invalid="ignore", divide="ignore"):
            cosv = (dM * Vi).sum(1) / (np.linalg.norm(dM, axis=1) * np.linalg.norm(Vi, axis=1))
        vcoh = float(np.nanmedian(cosv))
        dx = (Xsrc[J] - Xsrc[I]).astype(np.float32)
        Cx = Qf[:, :KMAX].T @ dx.astype(np.float64); n0 = (dx.astype(np.float64) ** 2).sum(0)
        Qr = np.vstack([np.ones(len(srcs)), 1 - np.cumsum(Cx ** 2, 0) / n0])        # (31, ns)
        E = np.full((KMAX + 1, len(srcs), len(tg)), np.nan, np.float32)
        for g0 in range(0, len(tg), CHUNK):
            sl = tg[g0:g0 + CHUNK]; Ms_ = Mu[:, sl]; y = (Ms_[J] - Ms_[I]).astype(np.float32)
            for k, _n, _d, _z, a in project_scores(dx, y, Qf):
                E[k, :, g0:g0 + len(sl)] = a
        absS[root] = E
        k50, k80 = int(np.argmax(st["C"] >= .5)), int(np.argmax(st["C"] >= .8))
        kf50, kf80 = int(np.argmax(stf["C"] >= .5)), int(np.argmax(stf["C"] >= .8))
        res.update({f"{root}|s_dP": s, f"{root}|C_dP": st["C"], f"{root}|s_full": sf[:200], f"{root}|C_full": stf["C"][:201],
                    f"{root}|pa_obs": pa_obs, f"{root}|pa_thr": pa_thr, f"{root}|Qr": Qr, f"{root}|cosv": cosv})
        res[f"{root}|sys"] = np.array([st["reff"], st["reff_ent"], gd_rank(s, *dP.shape), kpa, gd_rank(sf, *Dfull.shape),
                                       stf["reff"], stf["reff_ent"], vcoh, k50, k80, kf50, kf80, stf["C"][30], stf["C"][10], stf["C"][5]])
        res[f"{root}|n_pairs"] = len(I)
        print(f"{system} {root}: pairs {len(I)} reff {st['reff']:.2f} kGD_dP {gd_rank(s, *dP.shape)} kPA_full {kpa} "
              f"kGD_full {gd_rank(sf, *Dfull.shape)} Cfull(30) {stf['C'][30]:.3f} vcoh {vcoh:+.3f} {time.time()-t0:.0f}s", flush=True)
    # per-source stability
    v1 = {}
    for root in absS:
        P = np.stack([np.vstack([pct(absS[root][k, s_]) for k in KADM]) for s_ in range(len(srcs))])   # (ns, |KADM|, T)
        v1[root] = np.nanmean(P, 1)
    v1s = np.minimum(v1["dc1_min"], v1["dc1_max"])
    for s_, src in enumerate(srcs):
        for root in absS:
            E = absS[root][:, s_]
            sp = [spearmanr(E[k], E[k + 1], nan_policy="omit")[0] for k in range(KMAX)]
            j20 = [jac(E[k], E[k + 1], 20) for k in range(KMAX)]; j50 = [jac(E[k], E[k + 1], 50) for k in range(KMAX)]
            jv = [jac(E[k], v1s[s_], 50) for k in range(KMAX + 1)]
            ra = [spearmanr(absS["dc1_min"][k, s_], absS["dc1_max"][k, s_], nan_policy="omit")[0] for k in range(KMAX + 1)]
            res[f"{root}|{src}|sp_nb"] = np.array(sp); res[f"{root}|{src}|j20_nb"] = np.array(j20)
            res[f"{root}|{src}|j50_nb"] = np.array(j50); res[f"{root}|{src}|j50_v1"] = np.array(jv); res[f"{root}|{src}|root_agree"] = np.array(ra)
            sy = res[f"{root}|sys"]; Qr = res[f"{root}|Qr"][:, s_]
            rows.append(dict(system=system, root=root, source=src, focal=src in foc, n_pairs=res[f"{root}|n_pairs"],
                             reff_dP=sy[0], reff_ent_dP=sy[1], kGD_dP=int(sy[2]), kPA_full=int(sy[3]), kGD_full=int(sy[4]),
                             reff_full=sy[5], reff_ent_full=sy[6], vel_coh=sy[7],
                             C5=res[f"{root}|C_dP"][5], C10=res[f"{root}|C_dP"][10], C30=res[f"{root}|C_dP"][30],
                             k_C50=int(sy[8]), k_C80=int(sy[9]), k_Cfull50=int(sy[10]), k_Cfull80=int(sy[11]),
                             Cfull5=sy[14], Cfull10=sy[13], Cfull30=sy[12],
                             Qr5=Qr[5], Qr10=Qr[10], Qr30=Qr[30],
                             sp_nb_mean=float(np.nanmean(sp)), sp_nb_min=float(np.nanmin(sp)),
                             j20_nb_mean=float(np.mean(j20)), j50_nb_mean=float(np.mean(j50)),
                             j50_v1_k0=jv[0], j50_v1_k5=jv[5], j50_v1_mean_KADM=float(np.mean([jv[k] for k in KADM])),
                             root_agree_k0=ra[0], root_agree_k5=ra[5], root_agree_mean=float(np.nanmean(ra))))
    np.savez_compressed(f"{OUT}/{system}.npz", **{k: (v if isinstance(v, np.ndarray) else np.asarray(v)) for k, v in res.items()},
                        absS_min=absS["dc1_min"], absS_max=absS["dc1_max"])
    pd.DataFrame(rows).to_csv(f"{OUT}/{system}.tsv", sep="\t", index=False)
    json.dump(dict(system=system, script_sha256=__import__("hashlib").sha256(open(__file__, "rb").read()).hexdigest(),
                   truth_read=False, missing_focal=miss, wall_s=round(time.time() - t0)), open(f"{OUT}/{system}.receipt.json", "w"), indent=1)
    print(f"{system} done {time.time()-t0:.0f}s missing focal {miss}", flush=True)


def selftest():
    rng = np.random.default_rng(1); n, p, r = 4000, 200, 5
    D = rng.normal(size=(n, r)) @ (rng.normal(size=(r, p)) * 3) + rng.normal(size=(n, p))
    D -= D.mean(0)
    kpa, _, _ = pa_rank(D, B=20)
    s = np.linalg.svd(D, compute_uv=False)
    assert kpa == r, kpa
    assert gd_rank(s, n, p) == r, gd_rank(s, n, p)
    st = spectrum_stats(np.ones(10)); assert abs(st["reff"] - 10) < 1e-9 and abs(st["reff_ent"] - 10) < 1e-9
    assert jac(np.arange(100.), np.arange(100.), 20) == 1.0
    print("gb_geometry_matrix selftest PASS (PA and GD recover rank 5; r_eff exact on flat spectrum)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("cmd"); ap.add_argument("--threads", type=int, default=24); a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "merge":
        import glob
        pd.concat([pd.read_csv(f, sep="\t") for f in sorted(glob.glob(f"{OUT}/*.tsv")) if not f.endswith("MATRIX.tsv")]).to_csv(
            f"{OUT}/MATRIX.tsv", sep="\t", index=False); print("merged")
    else:
        run(a.cmd.upper())
