"""Velorama wrapper (NOT YET INTEGRATED).

Velorama's API is CLI-oriented (`velorama.execute_cmdline`) or low-level
(`estimate_interactions`, `construct_dag`, `train_model`). Integrating it
requires writing adata → DAG → Granger-MLP pipeline explicitly. Skipped for now.
"""
from __future__ import annotations
import pandas as pd


def run(adata, tfs: list[str], **_) -> pd.DataFrame:
    raise NotImplementedError(
        "velorama wrapper not yet integrated; use its CLI directly and feed "
        "the output TSV through geobridge_val.io.save_ranking()."
    )
