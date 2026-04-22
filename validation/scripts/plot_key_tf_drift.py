"""Per-TF NES barplot across drift-experiment methods.

One subplot per diagnostic TF. Bars colored by mode (correlation/none/onto/out).
Dashed lines at ±0.5 mark the strict decision threshold.

Usage:
  python scripts/plot_key_tf_drift.py \
      --v2 results/v2_day5/bm/per_tf.csv \
      --out results/summary_day5/bm/key_tf_v2.png
"""
from __future__ import annotations
import argparse
import re
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KEY_TFS = ["GATA2", "KLF1", "NFE2", "NFKB1", "HSF1", "KLF2", "KLF4"]
TF_CLASS = {
    "GATA2": "lineage", "KLF1": "lineage", "NFE2": "lineage",
    "NFKB1": "stress", "HSF1": "stress",
    "KLF2": "ambiguous", "KLF4": "ambiguous",
}


def _method_meta(m: str):
    """Return (sort_key, short_label, color) for a method name."""
    if m == "correlation":
        return (0, "corr", "#888888")
    if m == "gb_drift_none":
        return (1, "none", "#222222")
    match = re.match(r"gb_drift_(onto|out)_o(\d+)d(\d+)", m)
    if match:
        mode, order, _ = match.groups()
        color = {"onto": "#d95f02", "out": "#1b9e77"}[mode]
        return ({"onto": 2, "out": 3}[mode] * 10 + int(order),
                f"{mode}_o{order}", color)
    return (99, m, "#cccccc")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tf", required=True,
                    help="path to per_tf.csv (V1 or V2)")
    ap.add_argument("--out", required=True, help="output PNG path")
    ap.add_argument("--title", default=None)
    args = ap.parse_args()

    df = pd.read_csv(args.per_tf)
    df = df[df["tf"].isin(KEY_TFS)].copy()
    if df.empty:
        raise SystemExit(f"no key TFs found in {args.per_tf}")

    methods_in = sorted(df["method"].unique(), key=lambda m: _method_meta(m)[0])
    method_meta = {m: _method_meta(m) for m in methods_in}

    fig, axes = plt.subplots(2, 4, figsize=(14, 7), sharey=True)
    axes = axes.ravel()
    for i, tf in enumerate(KEY_TFS):
        ax = axes[i]
        sub = df[df["tf"] == tf].set_index("method").reindex(methods_in)
        vals = sub["nes"].values
        colors = [method_meta[m][2] for m in methods_in]
        labels = [method_meta[m][1] for m in methods_in]
        xs = range(len(methods_in))
        ax.bar(xs, vals, color=colors, edgecolor="black", linewidth=0.3)
        ax.axhline(0, color="black", linewidth=0.6)
        ax.axhline(+0.5, color="red", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.axhline(-0.5, color="red", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_xticks(list(xs))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
        cls = TF_CLASS.get(tf, "")
        ax.set_title(f"{tf} ({cls})", fontsize=10)
        if i % 4 == 0:
            ax.set_ylabel("NES")

    # hide unused subplot (7 TFs in 2x4)
    for k in range(len(KEY_TFS), len(axes)):
        axes[k].axis("off")

    if args.title:
        fig.suptitle(args.title, fontsize=11)
    fig.tight_layout()
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[plot] wrote {out_path}")


if __name__ == "__main__":
    main()
