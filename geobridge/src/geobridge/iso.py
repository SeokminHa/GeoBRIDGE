# Sign estimate d_iso and signed score T = d_iso * F_r(Z)
# S0 is averaged over seeds and both roots (NaN if any is NaN) and d_iso = sign(S0)
import os, sys, hashlib, importlib.util, numpy as np
from . import primary

VI = os.path.join(os.path.dirname(__file__), "vendor", "iso")
SEEDS_N = 3


def verify_vendor():
    for l in open(f"{VI}/SHA256SUMS"):
        h, p = l.split(); assert hashlib.sha256(open(f"{VI}/{p}", "rb").read()).hexdigest() == h, f"vendored file modified: {p}"


_IL = None


# Import the vendored libs under the module names they import each other by, without leaking into sys.modules/sys.path
def load():
    global _IL
    if _IL is not None: return _IL
    verify_vendor(); saved_path = list(sys.path); names = ("gb_primary", "gb_iso_lib", "gb_isofull_lib"); saved = {n: sys.modules.get(n) for n in names}
    try:
        sys.modules["gb_primary"] = primary
        for n in names[1:]:
            sp = importlib.util.spec_from_file_location(n, f"{VI}/{n}.py"); m = importlib.util.module_from_spec(sp); sys.modules[n] = m; sp.loader.exec_module(m)
        _IL = sys.modules["gb_isofull_lib"]
    finally:
        sys.path[:] = saved_path
        for n, v in saved.items():
            if v is None: sys.modules.pop(n, None)
            else: sys.modules[n] = v
    return _IL


# Return (S0, d_iso, info); score and sign arrays have shape (n_regulators, n_targets)
def iso_polarity(adata, regulators, pairs, stratum_key=None, window=0.02, seed=42, genes_idx=None):
    IL = load(); L = IL.L; ROOTS = IL.ROOTS
    allg = np.asarray(adata.var_names).astype(str); gi = np.arange(len(allg)) if genes_idx is None else np.asarray(genes_idx)
    up = np.char.upper(allg); si = [int(np.where(up == r.upper())[0][0]) for r in regulators]; n = adata.n_obs
    Xs = np.asarray(adata.layers["Ms"].todense() if hasattr(adata.layers["Ms"], "todense") else adata.layers["Ms"], np.float64)[:, si]
    Mu = np.asarray(adata.layers["Mu"].todense() if hasattr(adata.layers["Mu"], "todense") else adata.layers["Mu"], np.float32)[:, gi]
    P30 = np.asarray(adata.obsm["X_pca"], np.float64)[:, :IL.NPC]
    st = np.zeros(n, np.int64) if stratum_key in (None, "NONE") else IL.strata_codes(adata.obs, [stratum_key])[0]
    self_idx = np.array([int(np.where(gi == s)[0][0]) if s in set(gi) else -1 for s in si])
    S = {}; info = dict(window=window, seeds=[seed + i for i in range(SEEDS_N)], n_strata=int(len(np.unique(st))), n_pairs={}, unsupported_roots=[])
    for ri, r in enumerate(ROOTS):
        pt = np.asarray(pairs[r]["pt"], np.float64); per = []
        for sd in info["seeds"]:
            I, J = L.iso_pairs(pt, st, window, L.CAP, sd, ri); info["n_pairs"][f"{r}|{sd}"] = int(len(I))
            if len(I) < IL.NPC: per.append(np.full((len(si), len(gi)), np.nan)); info["unsupported_roots"].append(f"{r}|{sd}"); continue
            e = IL.engine(Xs, Mu, P30, I, J, km=0, self_idx=self_idx); per.append(IL.ratio(e["N"][0], e["D"][0]))
        S[r] = IL.seed_mean(per)
    S0 = IL.root_mean(S[ROOTS[0]], S[ROOTS[1]])
    return S0, IL.sgn(S0), info


# T = d_iso * F_r(Z)
# F_r = (midrank - .5)/m over the m finite entries of Z in that regulator's row (vendored IL.frac_rank)
# NaN if either factor NaN
def signed_composite(Z, d_iso):
    IL = load(); F = np.full(Z.shape, np.nan)
    for j in range(Z.shape[0]): F[j] = IL.frac_rank(Z[j], np.isfinite(Z[j]))
    return np.where(np.isfinite(F) & np.isfinite(d_iso), d_iso * F, np.nan), F
