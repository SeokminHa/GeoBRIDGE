"""
Task 6 — Gene-level GT vs BG score density (final decision step).

Motivation:
  Pancreas has AUROC-level SVD > Random signal (Wilcoxon p=0.001) but NO
  Top-N or stdAUPRC advantage. Determine whether this AUROC-level signal
  corresponds to a real distributional separation at the score level.

Pipeline:
  For dataset ∈ {bonemarrow, pancreas}, at k=5:
    1. Run production precompute_pairspace_stats_streaming
    2. Compute SVD scores (Vk = pre["Vt"][:5].T)
    3. Compute Random scores (single seed=0; distribution-shape not CI)
    4. For each TF: split |scores[tf_idx, :]| into GT vs BG
    5. Compute Cohen's d, KS, tail ratios; pooled + per-TF

Compute: ~20 min (bonemarrow ~13 min precompute, pancreas ~5 min). No other
reruns required.

Usage:
  python -u -m benchmark.post_analysis.task6_gt_vs_bg_density
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from benchmark.utils_projection import (
    load_for_production, precompute_pre, svd_vk, random_orthonormal_vk,
    scores_with_custom_vk,
)

OUT = Path("results/post_analysis/task6")
OUT.mkdir(parents=True, exist_ok=True)

K_FOCUS = 5
TARGETS = ["bonemarrow", "pancreas"]
RANDOM_SEED = 0  # single seed; distribution shape not uncertainty


# ------------------------------------------------------------------
# Record collection
# ------------------------------------------------------------------

def collect_records(scores, gt_dict, gene_names, method_label, ds_label,
                    min_targets=5):
    """
    Returns DataFrame with columns: dataset, method, tf, gene, role, score.
    score is ABS to match AUROC/Top-N convention.
    """
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    rows = []
    for tf, targets_container in gt_dict.items():
        tf_idx = None
        for cand in (tf, tf.title(), tf.upper(), tf.lower()):
            if cand in name_to_idx:
                tf_idx = name_to_idx[cand]
                break
        if tf_idx is None:
            continue
        if isinstance(targets_container, dict):
            gt_set = set(targets_container.keys())
        else:
            gt_set = set(targets_container)
        if len(gt_set) < min_targets:
            continue

        for j, g in enumerate(gene_names):
            if j == tf_idx:
                continue
            role = "GT" if (g in gt_set or g.upper() in gt_set) else "BG"
            rows.append({
                "dataset": ds_label,
                "method": method_label,
                "tf": tf,
                "gene": g,
                "role": role,
                "score": abs(float(scores[tf_idx, j])),
            })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------

def cohens_d(gt, bg):
    gt = np.asarray(gt)
    bg = np.asarray(bg)
    if len(gt) < 2 or len(bg) < 2:
        return np.nan
    vg = gt.var(ddof=1)
    vb = bg.var(ddof=1)
    pooled = np.sqrt((vg * (len(gt) - 1) + vb * (len(bg) - 1))
                     / (len(gt) + len(bg) - 2))
    if pooled == 0:
        return np.nan
    return (gt.mean() - bg.mean()) / pooled


def pooled_stats(df_subset):
    gt = df_subset[df_subset.role == "GT"]["score"].values
    bg = df_subset[df_subset.role == "BG"]["score"].values
    if len(gt) == 0 or len(bg) == 0:
        return {"n_gt": len(gt), "n_bg": len(bg)}
    # alternative="less" means CDF_GT is less (→ GT distribution is
    # stochastically greater than BG, which is what we want)
    try:
        ks_stat, ks_p = stats.ks_2samp(gt, bg, alternative="less")
    except Exception:
        ks_stat, ks_p = np.nan, np.nan
    q95_bg = np.quantile(bg, 0.95)
    frac_gt_above_bg95 = float((gt > q95_bg).mean())
    return {
        "n_gt": len(gt), "n_bg": len(bg),
        "gt_mean": float(gt.mean()), "bg_mean": float(bg.mean()),
        "gt_median": float(np.median(gt)), "bg_median": float(np.median(bg)),
        "cohens_d_pooled": cohens_d(gt, bg),
        "ks_statistic": float(ks_stat),
        "ks_p": float(ks_p),
        "gt_q90": float(np.quantile(gt, 0.9)),
        "bg_q90": float(np.quantile(bg, 0.9)),
        "tail_ratio_q90": float(np.quantile(gt, 0.9) / max(np.quantile(bg, 0.9), 1e-12)),
        "frac_gt_above_bg_q95": frac_gt_above_bg95,
    }


def per_tf_stats(df_subset):
    """
    Per-TF Cohen's d, AUROC, AUPRC computed from ABSOLUTE |score|.
    Uses sklearn; labels: 1=GT target, 0=BG.
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    rows = []
    for tf, sub in df_subset.groupby("tf"):
        gt = sub[sub.role == "GT"]["score"].values
        bg = sub[sub.role == "BG"]["score"].values
        if len(gt) < 2 or len(bg) < 2:
            continue
        labels = np.concatenate([np.ones(len(gt)), np.zeros(len(bg))])
        preds = np.concatenate([gt, bg])
        try:
            auroc = roc_auc_score(labels, preds)
        except Exception:
            auroc = np.nan
        try:
            auprc = average_precision_score(labels, preds)
        except Exception:
            auprc = np.nan
        rows.append({
            "tf": tf,
            "n_gt": len(gt), "n_bg": len(bg),
            "cohens_d": cohens_d(gt, bg),
            "auroc": auroc,
            "auprc": auprc,
            "baseline_auprc": len(gt) / max(len(gt) + len(bg), 1),
            "gt_mean": float(gt.mean()),
            "bg_mean": float(bg.mean()),
            "gt_over_bg_mean_ratio": float(gt.mean() / max(bg.mean(), 1e-12)),
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# Plotting
# ------------------------------------------------------------------

def plot_ecdf(df_ds, ds, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, scale in zip(axes, ["linear", "log"]):
        for method, color_gt, color_bg in [("SVD", "#1f77b4", "#9ecae1"),
                                             ("Random", "#d62728", "#fc9272")]:
            d = df_ds[df_ds.method == method]
            for role, color, ls in [("GT", color_gt, "-"), ("BG", color_bg, "--")]:
                s = d[d.role == role]["score"].values
                if len(s) == 0:
                    continue
                s_sorted = np.sort(s)
                y = np.arange(1, len(s_sorted) + 1) / len(s_sorted)
                ax.plot(s_sorted, y, color=color, ls=ls, lw=1.5,
                        label=f"{method} {role} (n={len(s_sorted)})")
        ax.set_xscale(scale)
        ax.set_xlabel("|score|")
        ax.set_ylabel("ECDF")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f"{ds}: ECDF of |score| by (method, GT/BG) at k={K_FOCUS}")
    fig.tight_layout()
    fig.savefig(out / f"ecdf_{ds}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_density(df_ds, ds, out):
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), squeeze=False)
    axes = axes.flatten()
    for ax, method in zip(axes, ["SVD", "Random"]):
        d = df_ds[df_ds.method == method]
        gt = d[d.role == "GT"]["score"].values
        bg = d[d.role == "BG"]["score"].values
        if len(gt) == 0 or len(bg) == 0:
            continue
        lo = max(min(gt.min(), bg.min()), 1e-6)
        hi = max(gt.max(), bg.max())
        bins = np.logspace(np.log10(lo), np.log10(hi), 50)
        ax.hist(bg, bins=bins, alpha=0.55, density=True,
                label=f"BG (n={len(bg)})", color="gray")
        ax.hist(gt, bins=bins, alpha=0.7, density=True,
                label=f"GT (n={len(gt)})", color="orange")
        ax.set_xscale("log")
        ax.set_xlabel("|score|")
        ax.set_ylabel("density")
        ax.set_title(method)
        ax.legend(fontsize=8)
    fig.suptitle(f"{ds}: |score| density by method at k={K_FOCUS}")
    fig.tight_layout()
    fig.savefig(out / f"density_{ds}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_cohens_d_per_tf(per_tf_df, ds, out):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    data = []
    labels = []
    for method in ["SVD", "Random"]:
        sub = per_tf_df[(per_tf_df.dataset == ds) & (per_tf_df.method == method)]
        data.append(sub["cohens_d"].dropna().values)
        labels.append(f"{method}\n(n={len(sub)})")
    bp = ax.boxplot(data, tick_labels=labels, showmeans=True)
    ax.axhline(0, ls="--", color="k", alpha=0.4, lw=0.7)
    ax.axhline(0.2, ls=":", color="orange", alpha=0.7, label="d = 0.2")
    ax.set_ylabel("per-TF Cohen's d (GT vs BG |score|)")
    ax.set_title(f"{ds}: per-TF Cohen's d distribution at k={K_FOCUS}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out / f"cohens_d_{ds}.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------------
# Verdict
# ------------------------------------------------------------------

def classify(svd_pooled, rand_pooled, per_tf_svd):
    """Classify SVD's signal strength on this dataset."""
    if not svd_pooled or "cohens_d_pooled" not in svd_pooled:
        return "NO DATA"
    d_svd = svd_pooled.get("cohens_d_pooled", np.nan)
    ks_p = svd_pooled.get("ks_p", np.nan)
    d_rand = rand_pooled.get("cohens_d_pooled", np.nan) if rand_pooled else np.nan
    delta_d = (d_svd - d_rand) if not (np.isnan(d_svd) or np.isnan(d_rand)) else np.nan

    if len(per_tf_svd) > 0:
        med_d = float(np.nanmedian(per_tf_svd["cohens_d"].values))
        frac_strong = float((per_tf_svd["cohens_d"] > 0.2).mean())
    else:
        med_d, frac_strong = np.nan, 0.0

    strong = (
        (d_svd >= 0.15) and
        (ks_p < 1e-6) and
        (med_d >= 0.1) and
        (frac_strong >= 0.3) and
        (delta_d >= 0.05)
    )
    weak = (
        (d_svd >= 0.05 and ks_p < 0.01) or
        (med_d >= 0.05 and frac_strong >= 0.2) or
        (delta_d >= 0.02)
    )
    if strong:
        return "STRONG"
    if weak:
        return "WEAK"
    return "NO SIGNAL"


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    all_records = []
    all_pooled = []
    all_per_tf = []

    for ds in TARGETS:
        print(f"\n=== {ds} ===")
        X, V, pt, gene_names, gt_dict = load_for_production(ds)
        G = len(gene_names)
        print(f"  G={G}, cells={X.shape[0]}, n_TFs(GT)={len(gt_dict)}")

        t0 = time.perf_counter()
        pre = precompute_pre(X, V, pt, pair_seed=42)
        print(f"  precompute_pre done in {time.perf_counter() - t0:.1f}s")

        for method in ["SVD", "Random"]:
            if method == "SVD":
                Vk = svd_vk(pre, K_FOCUS)
            else:
                Vk = random_orthonormal_vk(G, K_FOCUS, seed=RANDOM_SEED)
            scores = scores_with_custom_vk(pre, Vk, normalize_score=True)

            df_rec = collect_records(scores, gt_dict, gene_names,
                                      method_label=method, ds_label=ds)
            all_records.append(df_rec)

            p = pooled_stats(df_rec)
            p["dataset"] = ds
            p["method"] = method
            all_pooled.append(p)

            ptf = per_tf_stats(df_rec)
            ptf["dataset"] = ds
            ptf["method"] = method
            all_per_tf.append(ptf)

            print(f"  [{method}] Cohen's d={p.get('cohens_d_pooled', np.nan):.4f}  "
                  f"KS p={p.get('ks_p', np.nan):.3g}  "
                  f"tail_q90_ratio={p.get('tail_ratio_q90', np.nan):.3f}  "
                  f"frac_GT>BG_q95={p.get('frac_gt_above_bg_q95', np.nan):.3f}")

    # Concatenate and save
    df_recs_full = pd.concat(all_records, ignore_index=True)
    df_pooled = pd.DataFrame(all_pooled)
    df_per_tf = pd.concat(all_per_tf, ignore_index=True)

    df_recs_full.to_csv(OUT / "records.csv.gz", index=False, compression="gzip")
    df_pooled.to_csv(OUT / "pooled_stats.csv", index=False)
    df_per_tf.to_csv(OUT / "per_tf_stats.csv", index=False)

    # Plots per dataset
    for ds in TARGETS:
        df_ds = df_recs_full[df_recs_full.dataset == ds]
        plot_ecdf(df_ds, ds, OUT)
        plot_density(df_ds, ds, OUT)
        plot_cohens_d_per_tf(df_per_tf, ds, OUT)
        print(f"  [plots saved for {ds}]")

    # Verdict
    lines = ["# Task 6 — Gene-level GT vs BG density\n"]
    lines.append("Compare SVD vs Random at k=5. SVD must beat Random by measurable "
                 "margin (Δ Cohen's d ≥ 0.05) to claim distributional separation.\n")
    lines.append(
        "| dataset | method | n_gt | Cohen's d | KS p | median per-TF d | "
        "frac TF d>0.2 | per-TF AUROC | per-TF AUPRC | baseline | "
        "AUPRC / baseline | tail_ratio_q90 | GT>BG_q95 |"
    )
    lines.append("|" + "---|" * 13)

    verdicts = {}
    for ds in TARGETS:
        svd_p = next((x for x in all_pooled if x["dataset"] == ds and x["method"] == "SVD"), None)
        rand_p = next((x for x in all_pooled if x["dataset"] == ds and x["method"] == "Random"), None)
        per_tf_svd = df_per_tf[(df_per_tf.dataset == ds) & (df_per_tf.method == "SVD")]
        per_tf_rand = df_per_tf[(df_per_tf.dataset == ds) & (df_per_tf.method == "Random")]

        for method, p, ptf in [("SVD", svd_p, per_tf_svd), ("Random", rand_p, per_tf_rand)]:
            if p is None:
                continue
            med_d = float(np.nanmedian(ptf["cohens_d"].values)) if len(ptf) else np.nan
            frac = float((ptf["cohens_d"] > 0.2).mean()) if len(ptf) else np.nan
            auroc_mean = float(np.nanmean(ptf["auroc"].values)) if len(ptf) else np.nan
            auprc_mean = float(np.nanmean(ptf["auprc"].values)) if len(ptf) else np.nan
            baseline_mean = float(np.nanmean(ptf["baseline_auprc"].values)) if len(ptf) else np.nan
            auprc_over_baseline = auprc_mean / max(baseline_mean, 1e-12) if not np.isnan(auprc_mean) else np.nan
            lines.append(
                f"| {ds} | {method} | {p.get('n_gt', '?')} | "
                f"{p.get('cohens_d_pooled', np.nan):.4f} | "
                f"{p.get('ks_p', np.nan):.3g} | "
                f"{med_d:.4f} | {frac:.2%} | "
                f"{auroc_mean:.4f} | {auprc_mean:.4f} | {baseline_mean:.4f} | "
                f"{auprc_over_baseline:.2f}× | "
                f"{p.get('tail_ratio_q90', np.nan):.3f} | "
                f"{p.get('frac_gt_above_bg_q95', np.nan):.3f} |"
            )

        verdicts[ds] = classify(svd_p, rand_p, per_tf_svd)

    lines.append("\n## Per-dataset verdicts\n")
    for ds, v in verdicts.items():
        lines.append(f"- **{ds}**: {v}")

    # Global
    pancreas_v = verdicts.get("pancreas", "NO DATA")
    bonemarrow_v = verdicts.get("bonemarrow", "NO DATA")
    lines.append(
        f"\n## Global paper verdict\n"
        f"- bonemarrow: **{bonemarrow_v}**\n"
        f"- pancreas:   **{pancreas_v}**\n"
    )
    if pancreas_v == "STRONG":
        lines.append("\n**→ FULL PUBLISHABLE**: pancreas is co-primary real-data evidence.")
    elif pancreas_v == "WEAK":
        lines.append("\n**→ INTERMEDIATE PUBLISHABLE**: pancreas supporting with caveat.")
    else:
        lines.append("\n**→ CONSERVATIVE PUBLISHABLE**: pancreas demoted to supplementary; "
                     "paper leads with SERGIO + bonemarrow.")

    (OUT / "verdict.md").write_text("\n".join(lines))
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
