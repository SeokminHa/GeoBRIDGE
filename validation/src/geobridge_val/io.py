"""Unified ranking I/O. All methods produce this format; all evaluators consume it."""
from __future__ import annotations
from pathlib import Path
import pandas as pd

RANKING_COLS = ["regulator", "target", "score", "rank", "method", "dataset"]


def save_ranking(df: pd.DataFrame, path: str | Path, method: str, dataset: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    assert {"regulator", "target", "score"}.issubset(out.columns), \
        f"missing cols in ranking; got {out.columns.tolist()}"
    out["method"] = method
    out["dataset"] = dataset
    out["rank"] = out["score"].rank(ascending=False, method="first").astype("int64")
    out[RANKING_COLS].to_parquet(path, index=False)


def load_rankings(glob_pattern: str) -> pd.DataFrame:
    paths = sorted(Path().glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"no rankings matching {glob_pattern}")
    return pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)


def load_edges_set(path: str | Path, reg_col: str = "tf", tgt_col: str = "target") -> set[tuple[str, str]]:
    df = pd.read_csv(path, sep=None, engine="python")
    return set(zip(df[reg_col].astype(str), df[tgt_col].astype(str)))
