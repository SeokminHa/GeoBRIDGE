"""GRNBoost2 via arboreto with an explicit LocalCluster.

Note: arboreto requires dask <2025 (needs legacy dask.dataframe API). Install with:
  pip install "dask<2025" "distributed<2025"
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import issparse


def run(adata, tfs: list[str], n_estimators: int = 1000, n_jobs: int = 8,
        layer: str = "Ms", **_) -> pd.DataFrame:
    from arboreto.algo import grnboost2   # lazy import
    from distributed import Client, LocalCluster

    X = adata.layers[layer] if layer in adata.layers else adata.X
    if issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    genes = adata.var_names.tolist()
    expr = pd.DataFrame(X, columns=genes)
    tfs_in = [g for g in tfs if g in expr.columns]
    if not tfs_in:
        raise ValueError("no TFs overlap with var_names")

    with LocalCluster(
        n_workers=n_jobs,
        threads_per_worker=1,
        memory_limit="8GB",
        dashboard_address=None,
        processes=True,
    ) as cluster:
        with Client(cluster) as client:
            net = grnboost2(
                expression_data=expr,
                tf_names=tfs_in,
                client_or_address=client,
                seed=42,
                verbose=True,
            )

    net = net.rename(columns={"TF": "regulator", "importance": "score"})
    return net[["regulator", "target", "score"]]
