"""GeoBRIDGE wrapper.

Calls into the existing geobridge_pairspace_experiments_indirect/geobridge_model
package if installed on PYTHONPATH, otherwise raises with instructions.

Returns long-format TF->target scores (DataFrame).
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def _resolve_geobridge():
    try:
        from geobridge_model import pairspace, scoring  # type: ignore
        return pairspace, scoring
    except ImportError as e:
        raise ImportError(
            "geobridge_model not importable. Either:\n"
            "  export PYTHONPATH=/path/to/geobridge_pairspace_experiments_indirect:$PYTHONPATH\n"
            "or install it via pip install -e . in that directory."
        ) from e


def run(
    adata,
    tfs: list[str],
    pair_knn: int = 30,
    k: int | str = "auto",
    lambda_: float = 1.0,
    mode: str = "dual",
    residualize: bool = True,
    layer_X: str = "Ms",
    layer_V: str = "velocity",
    **_,
) -> pd.DataFrame:
    pairspace, scoring = _resolve_geobridge()
    genes = adata.var_names.tolist()
    tf_set = set(tfs)
    tf_idx = np.array([i for i, g in enumerate(genes) if g in tf_set])
    if len(tf_idx) == 0:
        raise ValueError("no TFs found in var_names")

    # This is the hook. Match it to your actual API. If the current repo exposes
    # `pairspace.run(adata, ...)` returning a score matrix (n_tf, n_gene) with
    # row/col names, adapt here.
    score_mat = pairspace.run(  # type: ignore[attr-defined]
        adata,
        tf_idx=tf_idx,
        pair_knn=pair_knn,
        k=k,
        lambda_=lambda_,
        mode=mode,
        residualize=residualize,
        layer_X=layer_X,
        layer_V=layer_V,
    )
    score_mat = np.asarray(score_mat)
    assert score_mat.shape == (len(tf_idx), len(genes)), \
        f"expected ({len(tf_idx)}, {len(genes)}) got {score_mat.shape}"

    rows = []
    for i, ti in enumerate(tf_idx):
        tf = genes[ti]
        for j, tgt in enumerate(genes):
            if tgt == tf:
                continue
            rows.append((tf, tgt, float(score_mat[i, j])))
    return pd.DataFrame(rows, columns=["regulator", "target", "score"])
