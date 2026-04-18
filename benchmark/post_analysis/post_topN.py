"""
Task 2 — Top-N precision post-processing from existing all_ablation.csv.

Uses existing `topk_precision_{10,20,50}` columns computed per TF.
Produces per-(dataset, method, N) aggregate plus paired Wilcoxon SVD vs Random.

Usage:
  python -u -m benchmark.post_analysis.post_topN
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

N_VALUES = [10, 20, 50]
K_FOCUS = 5


def aggregate():
    df = pd.read_csv(IN)

    rows = []
    for ds in sorted(df["dataset"].unique()):
        identity_rows = df[(df["dataset"] == ds) & (df["method"] == "Identity") & (df["k"] == 0)]

        for method in ["Identity", "SVD", "SVD_projectY", "Random"]:
            if method == "Identity":
                sub = identity_rows
            else:
                sub = df[(df["dataset"] == ds) & (df["method"] == method) & (df["k"] == K_FOCUS)]
            if len(sub) == 0:
                continue
            for N in N_VALUES:
                col = f"topk_precision_{N}"
                if col not in sub.columns:
                    continue
                if method == "Random":
                    # Aggregate per seed first (mean across TFs), then across seeds
                    per_seed = sub.groupby("seed")[col].mean()
                    mean_val = per_seed.mean()
                    sd_val = per_seed.std()
                else:
                    mean_val = sub[col].mean()
                    sd_val = sub[col].std() / np.sqrt(max(sub[col].count(), 1))

                rows.append({
                    "dataset": ds, "method": method, "N": N,
                    "precision_mean": mean_val,
                    "precision_se_or_sd": sd_val,
                    "n_tfs": sub["tf"].nunique(),
                })

    agg = pd.DataFrame(rows)
    agg.to_csv(OUT / "topN_aggregate.csv", index=False)
    return df, agg


def paired_stats(df):
    """Paired Wilcoxon SVD vs Random per (dataset, N) at k=K_FOCUS."""
    rows = []
    for ds in sorted(df["dataset"].unique()):
        for N in N_VALUES:
            col = f"topk_precision_{N}"
            svd = df[(df["dataset"] == ds) & (df["method"] == "SVD") & (df["k"] == K_FOCUS)]
            rnd = df[(df["dataset"] == ds) & (df["method"] == "Random") & (df["k"] == K_FOCUS)]
            if col not in svd.columns or len(svd) == 0 or len(rnd) == 0:
                continue
            svd_per_tf = svd.set_index("tf")[col]
            rnd_per_tf = rnd.groupby("tf")[col].mean()
            common = svd_per_tf.index.intersection(rnd_per_tf.index)
            if len(common) < 3:
                continue
            s = svd_per_tf.loc[common].values
            r = rnd_per_tf.loc[common].values
            mask = ~(np.isnan(s) | np.isnan(r))
            s, r = s[mask], r[mask]
            if len(s) < 3:
                continue
            try:
                _, p = stats.wilcoxon(s, r, alternative="greater")
            except ValueError:
                p = np.nan
            # Baseline ≈ N / (G - 1) — approximated from n_targets / n_candidates per TF
            mean_baseline = (svd["n_targets"] / svd["n_candidates"]).mean() * (N / N)  # just use per-TF baseline col if available
            if "baseline_auprc" in svd.columns:
                mean_baseline = svd["baseline_auprc"].mean()

            rows.append({
                "dataset": ds, "N": N,
                "svd_precision_mean": s.mean(),
                "random_precision_mean": r.mean(),
                "delta": s.mean() - r.mean(),
                "wilcoxon_p": p,
                "n_tfs": len(s),
                "baseline_auprc_mean": mean_baseline,
                "svd_over_baseline": s.mean() / max(mean_baseline, 1e-12),
            })
    return pd.DataFrame(rows)


def plot_panels(agg):
    import matplotlib.pyplot as plt
    datasets = sorted(agg["dataset"].unique())
    n = len(datasets)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    axes = axes.flatten()

    color = {
        "Identity": "gray",
        "SVD": "#1f77b4",
        "SVD_projectY": "#2ca02c",
        "Random": "#d62728",
    }
    marker = {"Identity": "s", "SVD": "o", "SVD_projectY": "^", "Random": "D"}

    for ax, ds in zip(axes, datasets):
        for method in ["Identity", "SVD", "SVD_projectY", "Random"]:
            sub = agg[(agg["dataset"] == ds) & (agg["method"] == method)].sort_values("N")
            if len(sub) == 0:
                continue
            yerr = 1.96 * sub["precision_se_or_sd"] if method == "Random" else None
            ax.errorbar(sub["N"], sub["precision_mean"],
                        yerr=yerr,
                        marker=marker[method], color=color[method],
                        label=method, lw=1.5,
                        capsize=3 if yerr is not None else 0)
        ax.set_xscale("log")
        ax.set_xlabel("N (top predictions)")
        ax.set_ylabel(f"precision@N (mean across TFs)  [k={K_FOCUS}]")
        ax.set_title(ds)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)

    for ax in axes[len(datasets):]:
        ax.axis("off")

    fig.suptitle(f"Top-N precision (k={K_FOCUS}), real data baseline ≪ SERGIO baseline")
    fig.tight_layout()
    fig.savefig(OUT / "figure_topN.png", dpi=150, bbox_inches="tight")


def main():
    df, agg = aggregate()
    stats_df = paired_stats(df)
    stats_df.to_csv(OUT / "topN_stats.csv", index=False)

    print("=== TOP-N AGGREGATE ===")
    print(agg.round(4).to_string(index=False))
    print("\n=== TOP-N STATS (SVD vs Random) ===")
    print(stats_df.round(4).to_string(index=False))

    try:
        plot_panels(agg)
        print(f"\n[SAVE] {OUT}/figure_topN.png")
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    print(f"\n[SAVE] {OUT}/topN_aggregate.csv")
    print(f"[SAVE] {OUT}/topN_stats.csv")

    # Kill-criteria check
    print("\n=== KILL CRITERIA (Top-N @ N=20, k=5) ===")
    k20 = stats_df[stats_df["N"] == 20].set_index("dataset")
    for ds in ["A", "C", "B", "D", "bonemarrow", "pancreas"]:
        if ds not in k20.index:
            continue
        r = k20.loc[ds]
        delta = r["delta"]
        ratio = r["svd_over_baseline"]
        p = r["wilcoxon_p"]
        if ds in ("A", "C"):
            strong = (delta >= 0.05) and (ratio >= 2.0)
            status = "PRIMARY PASS" if strong else "PRIMARY FAIL"
        elif ds in ("B", "D"):
            strong = delta >= 0.05
            status = "rescue OK" if strong else "rescue weak"
        elif ds == "pancreas":
            strong = (delta >= 0.005) and (p < 0.05) and (ratio >= 2.0)
            status = "PANCREAS PASS" if strong else "PANCREAS FAIL"
        elif ds == "bonemarrow":
            strong = delta >= 0.002
            status = "bonemarrow weak positive" if strong else "bonemarrow flat/negative"
        print(f"  {ds:12s}  Δ={delta:+.4f}  ratio={ratio:.2f}×  p={p:.3g}  → {status}")


if __name__ == "__main__":
    main()
