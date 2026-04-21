"""Baseline: Pearson correlation over Ms (smoothed expression)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import issparse


def run(adata, tfs: list[str], use_abs: bool = False, layer: str = "Ms", **_) -> pd.DataFrame:
    genes = adata.var_names.tolist()
    X = adata.layers[layer] if layer in adata.layers else adata.X
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    # z-score per gene
    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-8
    Z = (X - mu) / sd

    tf_idx = [i for i, g in enumerate(genes) if g in set(tfs)]
    tf_names = [genes[i] for i in tf_idx]
    C = (Z[:, tf_idx].T @ Z) / Z.shape[0]   # (n_tf, n_gene)
    if use_abs:
        C = np.abs(C)

    rows = []
    for i, tf in enumerate(tf_names):
        for j, tgt in enumerate(genes):
            if tgt == tf:
                continue
            rows.append((tf, tgt, float(C[i, j])))
    return pd.DataFrame(rows, columns=["regulator", "target", "score"])
