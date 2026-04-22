"""Compact key-TF NES readout for the drift-projection experiment.

Pivots per_tf.csv (written by run_v1_enrichment.py) to a wide table with
one row per diagnostic TF and one column per method. Method order is
fixed so lineage/stress patterns are easy to scan.

Usage:
  python scripts/summarize_drift_eval.py \
      --v1 results/v1_day5/bm/per_tf.csv \
      --v2 results/v2_day5/bm/per_tf.csv \
      --out results/summary_day5/bm/
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd

KEY_TFS = ["GATA2", "KLF1", "NFE2", "NFKB1", "HSF1", "KLF2", "KLF4"]
TF_CLASS = {
    "GATA2": "lineage", "KLF1": "lineage", "NFE2": "lineage",
    "NFKB1": "stress", "HSF1": "stress",
    "KLF2": "ambiguous", "KLF4": "ambiguous",
}


def _method_sort_key(m: str):
    """Fix column order: correlation, gb_drift_none, then onto/out by order."""
    if m == "correlation":
        return (0, 0, 0, m)
    if m == "gb_drift_none":
        return (1, 0, 0, m)
    match = re.match(r"gb_drift_(onto|out)_o(\d+)d(\d+)", m)
    if match:
        mode, order, dim = match.groups()
        mode_rank = {"onto": 2, "out": 3}[mode]
        return (mode_rank, int(order), int(dim), m)
    return (9, 0, 0, m)


def _summarize(per_tf_path: Path, tag: str) -> pd.DataFrame | None:
    if not per_tf_path.exists():
        print(f"[summary] skip {tag}: {per_tf_path} not found")
        return None
    df = pd.read_csv(per_tf_path)
    df = df[df["tf"].isin(KEY_TFS)].copy()
    if df.empty:
        print(f"[summary] skip {tag}: no key TFs in {per_tf_path}")
        return None
    methods = sorted(df["method"].unique(), key=_method_sort_key)
    wide = (df.pivot_table(index="tf", columns="method", values="nes",
                           aggfunc="first")
              .reindex(index=KEY_TFS, columns=methods))
    wide.insert(0, "class", [TF_CLASS.get(t, "") for t in wide.index])
    return wide


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", default=None, help="path to V1 per_tf.csv")
    ap.add_argument("--v2", default=None, help="path to V2 per_tf.csv")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    for tag, path in (("v1", args.v1), ("v2", args.v2)):
        if path is None:
            continue
        wide = _summarize(Path(path), tag)
        if wide is None:
            continue
        out_csv = out / f"key_tf_{tag}.csv"
        wide.to_csv(out_csv, float_format="%.3f")
        print(f"\n=== {tag.upper()} (NES) ===")
        with pd.option_context("display.max_columns", None,
                               "display.width", 200,
                               "display.float_format", "{:+.2f}".format):
            print(wide)
        print(f"[summary] wrote {out_csv}")


if __name__ == "__main__":
    main()
