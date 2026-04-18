"""
Task 3 — stdAUPRC vs k figure.

Uses existing `std_auprc` column in all_ablation.csv.

Usage:
  python -u -m benchmark.post_analysis.post_stdauprc_vs_k
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


def aggregate():
    df = pd.read_csv(IN)

    rows = []
    for ds in sorted(df["dataset"].unique()):
        for method in ["Identity", "SVD", "SVD_projectY", "Random"]:
            sub = df[(df["dataset"] == ds) & (df["method"] == method)]
            if len(sub) == 0:
                continue
            for k in sorted(sub["k"].unique()):
                s_k = sub[sub["k"] == k]
                if method == "Random":
                    per_seed = s_k.groupby("seed")["std_auprc"].mean()
                    mean_val = per_seed.mean()
                    sd_val = per_seed.std()
                    se_val = sd_val / np.sqrt(max(len(per_seed), 1))
                else:
                    mean_val = s_k["std_auprc"].mean()
                    sd_val = s_k["std_auprc"].std()
                    se_val = sd_val / np.sqrt(max(len(s_k), 1))
                rows.append({
                    "dataset": ds, "method": method, "k": int(k),
                    "std_auprc_mean": mean_val,
                    "std_auprc_sd": sd_val,
                    "std_auprc_se": se_val,
                    "n_tfs": s_k["tf"].nunique(),
                })
    return df, pd.DataFrame(rows)


def paired_stats_at_k(df, k=K_FOCUS):
    rows = []
    for ds in sorted(df["dataset"].unique()):
        svd = df[(df["dataset"] == ds) & (df["method"] == "SVD") & (df["k"] == k)]
        rnd = df[(df["dataset"] == ds) & (df["method"] == "Random") & (df["k"] == k)]
        if len(svd) == 0 or len(rnd) == 0:
            continue
        svd_per_tf = svd.set_index("tf")["std_auprc"]
        rnd_per_tf = rnd.groupby("tf")["std_auprc"].mean()
        common = svd_per_tf.index.intersection(rnd_per_tf.index)
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
        rows.append({
            "dataset": ds,
            "svd_std_auprc_mean": s.mean(),
            "random_std_auprc_mean": r.mean(),
            "delta": s.mean() - r.mean(),
            "wilcoxon_p": p,
            "n_tfs": len(s),
        })
    return pd.DataFrame(rows)


def plot(agg):
    import matplotlib.pyplot as plt
    datasets = sorted(agg["dataset"].unique())
    n = len(datasets)
    ncol = 3
    nrow = int(np.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 4 * nrow), squeeze=False)
    axes = axes.flatten()
    color = {"Identity": "gray", "SVD": "#1f77b4",
             "SVD_projectY": "#2ca02c", "Random": "#d62728"}
    marker = {"Identity": "s", "SVD": "o", "SVD_projectY": "^", "Random": "D"}

    for ax, ds in zip(axes, datasets):
        for method in ["Identity", "SVD", "SVD_projectY", "Random"]:
            sub = agg[(agg["dataset"] == ds) & (agg["method"] == method)].sort_values("k")
            if len(sub) == 0:
                continue
            yerr = 1.96 * sub["std_auprc_se"] if method == "Random" else None
            ax.errorbar(sub["k"], sub["std_auprc_mean"], yerr=yerr,
                        marker=marker[method], color=color[method],
                        label=method, lw=1.5,
                        capsize=3 if yerr is not None else 0)
        ax.axhline(1.0, ls="--", color="k", alpha=0.5, lw=0.7)
        ax.set_xscale("symlog", linthresh=1)
        ax.set_xlabel("k (rank of projected subspace)")
        ax.set_ylabel("stdAUPRC (mean across TFs)")
        ax.set_title(ds)
        ax.legend(fontsize=7, loc="best")
        ax.grid(alpha=0.3)

    for ax in axes[len(datasets):]:
        ax.axis("off")

    fig.suptitle("stdAUPRC vs k  (stdAUPRC = AUPRC / baseline_AUPRC; random → 1.0)")
    fig.tight_layout()
    fig.savefig(OUT / "figure_stdauprc_vs_k.png", dpi=150, bbox_inches="tight")


def main():
    df, agg = aggregate()
    agg.to_csv(OUT / "stdauprc_aggregate.csv", index=False)

    print("=== stdAUPRC AGGREGATE (at each k) ===")
    piv = agg.pivot_table(index=["dataset", "method"], columns="k",
                           values="std_auprc_mean").round(3)
    print(piv.to_string())

    stats_df = paired_stats_at_k(df, k=K_FOCUS)
    stats_df.to_csv(OUT / "stdauprc_stats_k5.csv", index=False)

    print(f"\n=== stdAUPRC STATS at k={K_FOCUS} (SVD vs Random) ===")
    print(stats_df.round(4).to_string(index=False))

    try:
        plot(agg)
        print(f"\n[SAVE] {OUT}/figure_stdauprc_vs_k.png")
    except Exception as e:
        print(f"[WARN] plot failed: {e}")

    # Kill-criteria check at k=5
    print(f"\n=== KILL CRITERIA (stdAUPRC @ k={K_FOCUS}) ===")
    for ds in ["A", "C", "B", "D", "bonemarrow", "pancreas"]:
        row = stats_df[stats_df["dataset"] == ds]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        svd = r["svd_std_auprc_mean"]
        delta = r["delta"]
        p = r["wilcoxon_p"]
        if ds in ("A", "C"):
            strong = (svd >= 2.5) and (delta >= 0.5)
            tag = "PRIMARY PASS" if strong else "PRIMARY FAIL"
        elif ds in ("B", "D"):
            tag = "rescue present" if delta >= 0.5 else "rescue weak"
        elif ds == "pancreas":
            strong = (svd >= 2.0) and (delta >= 0.3) and (p < 0.05)
            tag = "PANCREAS PASS" if strong else "PANCREAS FAIL"
        elif ds == "bonemarrow":
            strong = (svd >= 2.0) and (delta >= 0.3)
            tag = "bonemarrow weak positive" if strong else "bonemarrow flat/negative"
        else:
            tag = "?"
        print(f"  {ds:12s}  SVD={svd:.2f}×  Δ={delta:+.2f}  p={p:.3g}  → {tag}")


if __name__ == "__main__":
    main()
