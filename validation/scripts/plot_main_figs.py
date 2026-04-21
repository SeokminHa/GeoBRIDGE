"""Minimal figure generation for V1 + V2 summary."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_v1(csv: Path, out: Path):
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    sns.barplot(df, x="method", y="nes", errorbar="se", ax=ax)
    ax.axhline(0, lw=0.5, color="k")
    ax.set_title(f"V1 NES — {csv.parent.name}")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def plot_v2(csv: Path, out: Path):
    df = pd.read_csv(csv)
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    sns.boxplot(df, x="method", y="auprc", ax=ax)
    sns.stripplot(df, x="method", y="auprc", color="k", size=3, ax=ax)
    ax.set_title(f"V2 AUPRC per TF — {csv.parent.name}")
    fig.tight_layout(); fig.savefig(out, bbox_inches="tight"); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--fig-dir", default="results/figures")
    args = ap.parse_args()
    fig_dir = Path(args.fig_dir); fig_dir.mkdir(parents=True, exist_ok=True)

    v1 = Path("results/v1_enrichment") / args.dataset / "per_tf.csv"
    v2 = Path("results/v2_chip")       / args.dataset / "per_tf.csv"
    if v1.exists():
        plot_v1(v1, fig_dir / f"fig_v1_{args.dataset}.pdf")
        print(f"[fig] {fig_dir / f'fig_v1_{args.dataset}.pdf'}")
    if v2.exists():
        plot_v2(v2, fig_dir / f"fig_v2_{args.dataset}.pdf")
        print(f"[fig] {fig_dir / f'fig_v2_{args.dataset}.pdf'}")


if __name__ == "__main__":
    main()
