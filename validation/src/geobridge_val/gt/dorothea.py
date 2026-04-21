"""DoRothEA / CollecTRI loader. Tier A is the default; never rely on C/D as GT."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def load_dorothea(path: str | Path, tiers: tuple[str, ...] = ("A",)) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    # Expected cols: tf, target, confidence, mor (optional)
    cols_lower = {c.lower(): c for c in df.columns}
    df = df.rename(columns={cols_lower.get("source", "tf"): "tf",
                            cols_lower.get("target", "target"): "target"})
    if "confidence" in df.columns:
        df = df[df["confidence"].isin(tiers)]
    return df.reset_index(drop=True)


def gt_sets(df: pd.DataFrame, min_size: int = 5) -> dict[str, set[str]]:
    d = df.groupby("tf")["target"].apply(set).to_dict()
    return {k: v for k, v in d.items() if len(v) >= min_size}


def gt_edges(df: pd.DataFrame) -> set[tuple[str, str]]:
    return set(zip(df["tf"].astype(str), df["target"].astype(str)))
