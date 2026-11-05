"""Deterministic synthetic smoke fixture (NOT World-P): 400 cells, 60 genes, 2 roots x 1200 pairs. python make_fixture.py"""
import numpy as np, anndata as ad, hashlib
rng = np.random.default_rng(2026)
n, g = 400, 60
t = rng.random(n); X = np.outer(t, rng.normal(size=g)) + rng.normal(scale=.5, size=(n, g))
Mu = X * .7 + rng.normal(scale=.3, size=(n, g)); pca = np.hstack([t[:, None], rng.normal(size=(n, 39))])
A = ad.AnnData(X.astype(np.float32), var=__import__("pandas").DataFrame(index=[f"G{i}" for i in range(g)]))
A.layers["Ms"], A.layers["Mu"] = X.astype(np.float32), Mu.astype(np.float32); A.obsm["X_pca"] = pca.astype(np.float32)
pairs = {}
for r, o in (("dc1_min", t), ("dc1_max", -t)):
    I = rng.integers(0, n, 1200); J = rng.integers(0, n, 1200)
    sw = o[I] > o[J]; I, J = np.where(sw, J, I), np.where(sw, I, J); pairs[r] = {"I": I.astype(np.int64), "J": J.astype(np.int64)}
A.obs["embryo"] = np.where(np.arange(n) % 2 == 0, "e0", "e1")
pairs["dc1_min"]["pt"], pairs["dc1_max"]["pt"] = t.astype(np.float64), (1 - t).astype(np.float64)   # pseudotime per root (iso window 0.02)
A.uns["gb_pairs"] = pairs
A.write_h5ad("fixture/synthetic_smoke.h5ad")
