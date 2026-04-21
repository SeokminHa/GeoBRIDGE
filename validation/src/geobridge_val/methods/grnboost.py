"""GRNBoost2 via arboreto."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import issparse


def run(adata, tfs: list[str], n_estimators: int = 1000, n_jobs: int = 8,
        layer: str = "Ms", **_) -> pd.DataFrame:
    from arboreto.algo import grnboost2   # lazy import

    X = adata.layers[layer] if layer in adata.layers else adata.X
    if issparse(X):
        X = X.toarray()
    genes = adata.var_names.tolist()
    expr = pd.DataFrame(np.asarray(X), columns=genes)
    tfs_in = [g for g in tfs if g in expr.columns]

    net = grnboost2(
        expression_data=expr,
        tf_names=tfs_in,
        verbose=True,
        seed=42,
    )
    # arboreto returns columns: TF, target, importance
    net = net.rename(columns={"TF": "regulator", "importance": "score"})
    return net[["regulator", "target", "score"]]
