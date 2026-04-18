"""
Task 5 — Per-TF AUROC distribution (GT-vs-BG lite diagnostic).

Uses existing per-TF AUROC from all_ablation.csv. Each per-TF AUROC is
a rank-based GT-vs-BG separation statistic: AUROC = P(score_GT > score_BG).
The distribution of per-TF AUROCs across TFs tells us whether the
separation is systematic across the TF population.

Role: fast diagnostic filter, NOT final evidence. Magnitude info is lost
(rank-only). If this is ambiguous, escalate to Task 6 (gene-level density).

Usage:
  python -u -m benchmark.post_analysis.post_tf_auroc_density
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

IN = Path("results/ablation_projection_production/all_ablation.csv")
OUT = Path("results/post_analysis"); OUT.mkdir(parents=True, exist_ok=True)

K_FOCUS = 5


def load_and_pivot():
    df = pd.read_csv(IN)
    df = df[df["k"].isin([0, K_FOCUS])].copy()
    return df


def test_shift(values, baseline=0.5):
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) < 3:
        return np.nan, np.nan
    try:
        _, p = stats.wilcoxon(values - baseline, alternative="greater")
    except ValueError:
        p = np.nan
    return np.median(values), p


def summary(df):
    rows = []
    for ds in sorted(df["dataset"].unique()):
        for method in ["Identity", "SVD", "SVD_projectY", "Random"]:
            if method == "Identity":
                sub = df[(df["dataset"] == ds) & (df["method"] == method) & (df["k"] == 0)]
            else:
                sub = df[(df["dataset"] == ds) & (df["method"] == method) & (df["k"] == K_FOCUS)]
            if len(sub) == 0:
                continue
            if method == "Random":
                # Per-TF mean across seeds, then distribution
                values = sub.groupby("tf")["auroc"].mean().values
            else:
                values = sub["auroc"].values
            median, p = test_shift(values)
            rows.append({
                "dataset": ds,
                "method": method,
                "n_tfs": len(values),
                "median_auroc": median,
                "mean_auroc": float(np.nanmean(values)) if len(values) else np.nan,
                "frac_above_0.5": float(np.mean(values > 0.5)) if len(values) else np.nan,
                "wilcoxon_p_above_0.5": p,
            })
    return pd.DataFrame(rows)


def paired_stats(df):
    rows = []
    for ds in sorted(df["dataset"].unique()):
        svd = df[(df["dataset"] == ds) & (df["method"] == "SVD") & (df["k"] == K_FOCUS)].set_index("tf")
        rnd = df[(df["dataset"] == ds) & (df["method"] == "Random") & (df["k"] == K_FOCUS)]
        if len(svd) == 0 or len(rnd) == 0:
            continue
        rnd_per_tf = rnd.groupby("tf")["auroc"].mean()
        common = svd.index.intersection(rnd_per_tf.index)
        if len(common) < 3:
            continue
        s = svd.loc[common, "auroc"].values
        r = rnd_per_tf.loc[common].values
        mask = ~(np.isnan(s) | np.isnan(r))
        s, r = s[mask], r[mask]
        if len(s) < 3:
            continue
        try:
            _, p = stats.wilcoxon(s, r, alternative="greater")
        except ValueError:
            p = np.nan
        rows.append({
            "dataset": ds,
            "svd_median_auroc": float(np.median(s)),
            "random_median_auroc": float(np.median(r)),
            "delta_median": float(np.median(s) - np.median(r)),
            "wilcoxon_p_svd_gt_rand": p,
            "n_tfs": len(s),
        })
    return pd.DataFrame(rows)


def plot_density(df):
    import matplotlib.pyplot as plt
    datasets = sorted(df["dataset"].unique())
    n = len(datasets)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    axes = axes.flatten()
    color = {"SVD": "#1f77b4", "Random": "#d62728",
             "SVD_projectY": "#2ca02c", "Identity": "gray"}

    bins = np.linspace(0, 1, 21)
    for ax, ds in zip(axes, datasets):
        for method in ["SVD", "Random"]:
            if method == "Identity":
                sub = df[(df["dataset"] == ds) & (df["method"] == method) & (df["k"] == 0)]
            else:
                sub = df[(df["dataset"] == ds) & (df["method"] == method) & (df["k"] == K_FOCUS)]
            if len(sub) == 0:
                continue
            if method == "Random":
                values = sub.groupby("tf")["auroc"].mean().values
            else:
                values = sub["auroc"].values
            values = values[~np.isnan(values)]
            if len(values) == 0:
                continue
            ax.hist(values, bins=bins, alpha=0.45, color=color[method],
                    label=f"{method} (median={np.median(values):.3f})",
                    density=True, edgecolor="black", linewidth=0.3)
            ax.axvline(np.median(values), color=color[method], lw=1.5, ls="--")
        ax.axvline(0.5, color="k", lw=0.7, ls=":")
        ax.set_xlabel("per-TF AUROC")
        ax.set_ylabel("density")
        ax.set_title(ds)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    for ax in axes[len(datasets):]:
        ax.axis("off")

    fig.suptitle(f"per-TF AUROC density  (SVD vs Random, k={K_FOCUS})  [rank-only GT-vs-BG proxy]")
    fig.tight_layout()
    fig.savefig(OUT / "figure_tf_auroc_density.png", dpi=150, bbox_inches="tight")


def main():
    df = load_and_pivot()
    s_summary = summary(df)
    s_paired = paired_stats(df)
    s_summary.to_csv(OUT / "tf_auroc_dist_summary.csv", index=False)
    s_paired.to_csv(OUT / "tf_auroc_dist_paired.csv", index=False)

    print("=== per-TF AUROC SUMMARY ===")
    print(s_summary.round(4).to_string(index=False))
    print("\n=== SVD vs Random paired Wilcoxon (median AUROC across TFs) ===")
    print(s_paired.round(4).to_string(index=False))

    try:
        plot_density(df)
        print(f"\n[SAVE] {OUT}/figure_tf_auroc_density.png")
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    # Diagnostic verdict per dataset
    print(f"\n=== per-TF AUROC DIAGNOSTIC VERDICT (k={K_FOCUS}) ===")
    for ds in ["A", "C", "B", "D", "bonemarrow", "pancreas"]:
        r = s_paired[s_paired["dataset"] == ds]
        if len(r) == 0:
            continue
        row = r.iloc[0]
        svd_med = row["svd_median_auroc"]
        d_med = row["delta_median"]
        p = row["wilcoxon_p_svd_gt_rand"]

        if ds in ("A", "C"):
            threshold_med = 0.60
            verdict = ("strong" if svd_med >= threshold_med and d_med >= 0.05
                       else "weak" if svd_med >= 0.55
                       else "null")
        elif ds == "pancreas":
            if svd_med >= 0.55 and p < 0.05:
                verdict = "confirmatory"
            elif svd_med >= 0.51:
                verdict = "ambiguous (escalate to Task 6)"
            else:
                verdict = "abandon-indicator"
        elif ds == "bonemarrow":
            if svd_med >= 0.52 and p < 0.05:
                verdict = "confirmatory"
            elif svd_med >= 0.50:
                verdict = "ambiguous (escalate to Task 6)"
            else:
                verdict = "abandon-indicator"
        else:  # B, D
            verdict = ("rescue" if svd_med >= 0.55 else "weak")
        print(f"  {ds:12s}  SVD median={svd_med:.3f}  Δ median={d_med:+.3f}  p={p:.3g}  → {verdict}")


if __name__ == "__main__":
    main()
