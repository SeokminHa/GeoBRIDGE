"""Signed per-depth GeoBRIDGE concordance cache (U channel). Identical algebra to gb_real_uniform.score_gb_sources but stores
num/den (signed) instead of |num/den|. usage: gb_signed_cache.py SYSTEM SRC,SRC,... ; writes results/signed/<SYSTEM>.npz.
Parity: |signed| must equal the production cache absS (float32) for the same source rows. No truth is read."""
import sys, time, numpy as np
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch'), (__import__("os").environ["GEOBRIDGE_ROOT"] + '/pairspace')]
import gb_real_uniform as RU, gb_cop_alignment as CA
from gb_target_extension import project_scores, CHUNK, NPC, EPS
import gb_substrate as GS, anndata as ad
system, srcs = sys.argv[1].upper(), [s.upper() for s in sys.argv[2].split(',')]
t0 = time.time()
rows, eg = RU.gb_raw_rows(system, srcs=srcs); axis = list(eg.astype(str))
S = GS.load(system, canonical=True); fixture, _, _ = GS.paths(system, canonical=True); F = ad.read_h5ad(fixture)
fi = {g: i for i, g in enumerate(np.asarray(F.var_names).astype(str))}; fc = np.array([fi[g] for g in axis])
Mu = RU.dense(F.layers["Mu"][:, fc], np.float32); au = {g.upper(): i for i, g in enumerate(axis)}
si = np.array([au[s] for s in srcs]); Xsrc = np.asarray(S["X"], np.float64)[:, fc[si]]
P30 = np.asarray(F.obsm["X_pca"], float)[:, :NPC]; G = len(axis); nk = len(RU.GB_KS)
out = {r: np.full((len(srcs), G, nk), np.nan, np.float32) for r in RU.ROOTS}
for r in RU.ROOTS:
    I, J = (np.asarray(a, int) for a in S["pairs"][r]); Qf, _ = np.linalg.qr(P30[J] - P30[I]); dx = (Xsrc[J] - Xsrc[I]).astype(np.float32)
    for g0 in range(0, G, CHUNK):
        sl = slice(g0, min(g0 + CHUNK, G)); dy = (Mu[J, sl] - Mu[I, sl]).astype(np.float32)
        for k, num, den, _z, absS in project_scores(dx, dy, Qf):
            sup = den > EPS; out[r][:, sl, k] = np.where(sup, num / np.where(sup, den, 1.0), np.nan)
    print(system, r, len(I), f"{time.time()-t0:.0f}s", flush=True)
uni = []; par = {}
for j, s in enumerate(srcs):
    for r in RU.ROOTS: out[r][j, si[j], :] = np.nan
    Rw = rows[s]; u, _ = CA.observational_universe(s, Rw, eg); uni.append(np.asarray(u))
    for r in RU.ROOTS:
        a, b = np.abs(out[r][j]), np.asarray(Rw[(r, "U")]); m = np.isfinite(a) & np.isfinite(b)
        par[f"{s}|{r}"] = (float(np.abs(a[m] - b[m]).max()), int((np.isfinite(a) != np.isfinite(b)).sum()))
print("PARITY (max|abs-signed - cache|, nan-mismatch):", par)
import os; os.makedirs((__import__("os").environ["GEOBRIDGE_ROOT"] + "/results/signed"), exist_ok=True)
np.savez_compressed(f"{__import__('os').environ['GEOBRIDGE_ROOT']}/results/signed/{system}.npz", genes=np.array(axis), sources=np.array(srcs), uni=np.array(uni, dtype=object),
                    **{f"{r}|U": out[r] for r in RU.ROOTS})
print("done", f"{time.time()-t0:.0f}s")
