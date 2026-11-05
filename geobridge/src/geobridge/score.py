"""Frozen production score (PRODUCTION_SPEC_v2) from a canonical fixture + frozen pairs. Algebra = VENDORED
gb_target_extension.project_scores (numexpr) so U/X |S| are bitwise the production arrays; universe = vendored
CA.observational_universe logic (gb_cop_alignment.py:188), score() channels U (Mu) and X (Ms) as gb_cop_alignment.score."""
import numpy as np
from .primary import zcons, ROOTS, KS
from . import pipeline

NPC, CHUNK = 30, 1000


def _dense(M):
    return np.asarray(M.todense() if hasattr(M, "todense") else M)


def channel_absS(adata, pairs, regulators, genes_idx):
    """{(root, 'U'|'X'): (n_reg, n_tgt, 31) float32 |S|} with each regulator's own gene NaN (gb_cop_alignment.score)."""
    pipeline._load(); from gb_target_extension import project_scores
    up = np.char.upper(np.asarray(adata.var_names).astype(str))
    si = [int(np.where(up == r.upper())[0][0]) for r in regulators]
    Mu = _dense(adata.layers["Mu"])[:, genes_idx].astype(np.float32); Ms = _dense(adata.layers["Ms"])[:, genes_idx].astype(np.float32)
    Xsrc = _dense(adata.layers["Ms"]).astype(np.float64)[:, si]
    P30 = np.asarray(adata.obsm["X_pca"], float)[:, :NPC]; G = len(genes_idx)
    raw = {(r, c): np.full((len(si), G, 31), np.nan, np.float32) for r in ROOTS for c in ("U", "X")}
    pos = {g: i for i, g in enumerate(genes_idx)}
    for r in ROOTS:
        I, J = (np.asarray(pairs[r][c], int) for c in ("I", "J")); Qf, _ = np.linalg.qr(P30[J] - P30[I]); dx = (Xsrc[J] - Xsrc[I]).astype(np.float32)
        for g0 in range(0, G, CHUNK):
            sl = slice(g0, min(g0 + CHUNK, G))
            for c, M in (("U", Mu), ("X", Ms)):
                for k, _n, _d, _z, absS in project_scores(dx, (M[J, sl] - M[I, sl]).astype(np.float32), Qf):
                    raw[(r, c)][:, sl, k] = absS
    for j, s in enumerate(si):
        if s in pos:
            for a in raw.values(): a[j, pos[s], :] = np.nan
    return raw


def observational_universe(src, rows, eg):
    """verbatim logic of gb_cop_alignment.observational_universe: not self; finite for both roots x {U,X} x all depths."""
    uni = np.char.upper(np.asarray(eg).astype(str)) != src.upper()
    for h in ROOTS:
        for c in ("U", "X"): uni &= np.isfinite(rows[(h, c)]).all(1)
    return uni


def score_fixture(adata, regulators, pairs, targets=None):
    """-> (PRIMARY (n_reg, n_tgt) NaN=UNSUPPORTED, standardised over each regulator's own observational universe; raw dict; universes)."""
    allg = np.asarray(adata.var_names).astype(str)
    gi = np.arange(len(allg)) if targets is None else np.array([int(np.where(np.char.upper(allg) == t.upper())[0][0]) for t in targets])
    raw = channel_absS(adata, pairs, regulators, gi); eg = allg[gi]
    prim = np.full((len(regulators), len(gi)), np.nan); unis = []
    for j, s in enumerate(regulators):
        rows = {k: v[j] for k, v in raw.items()}; u = observational_universe(s, rows, eg); unis.append(u)
        p = zcons({h: rows[(h, "U")][u] for h in ROOTS}, KS); prim[j, u] = p
    return prim, raw, unis
