"""GeoBRIDGE wrapper.

Calls GeoBridgeModel.fit(X, V, pseudotime) from the parent repo.
Auto-adds the repo root to sys.path if needed.
"""
from __future__ import annotations
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd


def _add_parent_to_sys_path():
    """Walk up from this file to find a dir containing geobridge_model/, add to sys.path."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "geobridge_model" / "__init__.py").exists():
            p = str(parent)
            if p not in sys.path:
                sys.path.insert(0, p)
            return p
    return None


def _resolve_geobridge():
    try:
        from geobridge_model.config import GeoBridgeConfig
        from geobridge_model.model import GeoBridgeModel
        return GeoBridgeConfig, GeoBridgeModel
    except ImportError:
        _add_parent_to_sys_path()
        try:
            from geobridge_model.config import GeoBridgeConfig
            from geobridge_model.model import GeoBridgeModel
            return GeoBridgeConfig, GeoBridgeModel
        except ImportError as e:
            raise ImportError(
                "geobridge_model not importable. Repo root (containing geobridge_model/) "
                "must be on PYTHONPATH."
            ) from e


def _get_pseudotime(adata) -> np.ndarray:
    for key in ("velocity_pseudotime", "dpt_pseudotime", "palantir_pseudotime", "pseudotime"):
        if key in adata.obs.columns:
            pt = np.asarray(adata.obs[key]).astype(np.float64)
            if np.all(np.isfinite(pt)):
                return pt
    import scvelo as scv
    scv.tl.velocity_pseudotime(adata)
    return np.asarray(adata.obs["velocity_pseudotime"]).astype(np.float64)


def run(
    adata,
    tfs: list[str],
    max_k: int = 40,
    tail_fraction: float = 0.2,
    tail_min_start: int = 20,
    slope_window: int = 3,
    stable_m: int = 3,
    tau: float = 2.0,
    project_y: bool = True,
    normalize_score: bool = True,
    layer_X: str = "Ms",
    layer_V: str = "velocity",
    **_,
) -> pd.DataFrame:
    GeoBridgeConfig, GeoBridgeModel = _resolve_geobridge()

    X = np.asarray(adata.layers[layer_X] if layer_X in adata.layers else adata.X).astype(np.float64)
    V = np.asarray(adata.layers[layer_V]).astype(np.float64)
    pt = _get_pseudotime(adata)

    # Sort cells by pseudotime so pair-forward windows work
    order = np.argsort(pt, kind="mergesort")
    X = X[order]; V = V[order]; pt = pt[order]
    # Drop NaN velocity rows (scvelo can produce NaN for genes that failed; handle as zero)
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    cfg = GeoBridgeConfig(
        max_k=max_k,
        tail_fraction=tail_fraction,
        tail_min_start=tail_min_start,
        slope_window=slope_window,
        stable_m=stable_m,
        tau=tau,
        project_y=project_y,
        normalize_score=normalize_score,
    )
    model = GeoBridgeModel(cfg)
    result = model.fit(X, V, pt)
    scores = np.asarray(result["scores"])  # (G, G)

    genes = adata.var_names.tolist()
    G = len(genes)
    assert scores.shape == (G, G), f"expected ({G},{G}) got {scores.shape}"

    tf_set = set(tfs)
    tf_idx = [i for i, g in enumerate(genes) if g in tf_set]
    if not tf_idx:
        raise ValueError("no TFs found in var_names")

    rows = []
    for i in tf_idx:
        tf = genes[i]
        for j, tgt in enumerate(genes):
            if j == i:
                continue
            rows.append((tf, tgt, float(scores[i, j])))
    return pd.DataFrame(rows, columns=["regulator", "target", "score"])
