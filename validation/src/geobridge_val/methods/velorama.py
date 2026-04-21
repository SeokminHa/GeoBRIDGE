"""Velorama wrapper.

Velorama is a separate package; install in its own env (GPU) and call via CLI,
or install in-process if available. This module is a thin shim so that
scripts/run_methods.py can dispatch uniformly.

If `velorama` import fails, we raise with install hints instead of silently
substituting a fake baseline.
"""
from __future__ import annotations
import pandas as pd


def run(adata, tfs: list[str], dag_source: str = "velocity",
        n_lags: int = 5, layer_X: str = "Ms", layer_V: str = "velocity",
        **_) -> pd.DataFrame:
    try:
        import velorama   # type: ignore
    except ImportError as e:
        raise ImportError(
            "velorama not installed. Install via:\n"
            "  pip install git+https://github.com/rs239/velorama\n"
            "or run velorama in a separate conda env and invoke its CLI, then "
            "feed the output TSV through geobridge_val.io.save_ranking()."
        ) from e

    # Adapt to velorama's actual API. Expected: returns per-TF-per-target scores.
    net = velorama.run(           # type: ignore[attr-defined]
        adata,
        tfs=tfs,
        dag_source=dag_source,
        n_lags=n_lags,
        layer_X=layer_X,
        layer_V=layer_V,
    )
    return net.rename(columns={"TF": "regulator"})[["regulator", "target", "score"]]
