"""
Unified analysis + figures for GeoBRIDGE vs Velorama on real scRNA-seq data.

Reads CSV outputs from bm_real_data_comparison.py and produces all figures.

Figures produced:
  Fig A — Overall metrics (AUROC/AUPRC/EPR bars) per dataset × method × k
  Fig B — Top-K precision curves per dataset
  Fig C — k-sweep line plot (AUROC/AUPRC/EPR vs k)
  Fig D — Runtime bar chart (parsed from log files)
  Fig E — Featured TF heatmaps (top-20 predicted targets vs GT)
  Fig F — Featured TF score distribution (GT vs BG histograms)
  Fig G — Featured TF Venn diagrams (GeoBRIDGE ∩ Velorama ∩ GT)
  Fig H — Featured TF GO enrichment (bubble plot)
  Fig I — Stream plots (UMAP + velocity)

Usage:
  # Uses existing CSVs + Velorama TSVs + prepared h5ads
  python -u -m benchmark.analyze_real_comparison --outdir results/bm_real_comparison/figures
"""

import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from benchmark.bm_real_data_comparison import (
    _load_collectri_regulons,
    _build_ground_truth,
    _evaluate_tf,
    DATASET_ORGANISM,
    K_VALUES,
    TOP_K_VALUES,
)

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

OUTDIR = Path("results/bm_real_comparison")
PREPARED_DIR = OUTDIR / "prepared_h5ad"
CSV_DIR = OUTDIR / "csv"
VELO_DIR = OUTDIR / "velorama_output"

DATASETS = ["bonemarrow", "pancreas"]

# Featured TFs per dataset for case studies
FEATURED_TFS = {
    "bonemarrow": ["GATA1", "GATA2", "SPI1", "FLI1", "ERG"],
    "pancreas": ["NEUROG3", "PDX1", "NKX6-1", "PAX4", "ARX"],
}

# K values to focus on in comparison plots
FOCUS_K = [0, 5, 50]

# Color palette
COLOR_VELORAMA = "#d62728"   # red
COLOR_GB_K0 = "#1f77b4"      # blue
COLOR_GB_K5 = "#2ca02c"      # green
COLOR_GB_K50 = "#9467bd"     # purple
COLOR_GT = "#ff7f0e"          # orange
COLOR_BG = "#7f7f7f"          # gray


# ----------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------

def load_geobridge_results():
    """Load per-TF GeoBRIDGE results CSV."""
    path = CSV_DIR / "geobridge_real_results.csv"
    if not path.exists():
        print(f"[WARN] Not found: {path}")
        return None
    return pd.read_csv(path)


def load_velorama_results():
    """
    Load per-TF Velorama results. Either from `real_comparison_full.csv`
    (after `phase compare`) or by re-evaluating TSVs.
    """
    full_path = CSV_DIR / "real_comparison_full.csv"
    if full_path.exists():
        df = pd.read_csv(full_path)
        return df[df["method"] == "Velorama"].copy()

    # Fallback: re-evaluate from TSV
    rows = []
    import scanpy as sc

    for ds in DATASETS:
        tsv_candidates = list(VELO_DIR.glob(f"*{ds}*interactions*.tsv"))
        if not tsv_candidates:
            print(f"[WARN] No Velorama TSV for {ds}")
            continue
        tsv_path = tsv_candidates[0]
        print(f"  [load] {ds}: {tsv_path.name}")

        df_tsv = pd.read_csv(tsv_path, sep="\t", index_col=0)
        df_tsv.index = df_tsv.index.astype(str)
        df_tsv.columns = df_tsv.columns.astype(str)

        h5ad = PREPARED_DIR / f"{ds}.h5ad"
        if not h5ad.exists():
            print(f"  [SKIP] {ds}: prepared h5ad missing")
            continue
        adata = sc.read_h5ad(str(h5ad))
        gene_names = list(adata.var_names)

        organism = DATASET_ORGANISM.get(ds, "human")
        uppercase = (organism == "mouse")
        net = _load_collectri_regulons(organism=organism)
        gt_dict = _build_ground_truth(net, gene_names, uppercase_match=uppercase)

        for tf, targets in gt_dict.items():
            if tf not in df_tsv.columns:
                continue
            scores_tf, labels_tf = [], []
            for gene_j in gene_names:
                if gene_j == tf or gene_j not in df_tsv.index:
                    continue
                scores_tf.append(abs(df_tsv.loc[gene_j, tf]))
                labels_tf.append(1 if gene_j in targets else 0)
            labels_tf = np.array(labels_tf)
            scores_tf = np.array(scores_tf)
            if labels_tf.sum() == 0 or labels_tf.sum() == len(labels_tf):
                continue
            m = _evaluate_tf(scores_tf, labels_tf)
            rows.append({"dataset": ds, "method": "Velorama", "k": -1, "tf": tf, **m})

    return pd.DataFrame(rows) if rows else None


def load_score_matrices():
    """
    Load per-dataset score matrices (gene × gene) for both methods at all k.

    Returns:
      scores[dataset][method_key] = (gene_names, score_matrix)
        method_key: "velorama" or "geobridge_k{K}"
    """
    import scanpy as sc
    scores = {}

    for ds in DATASETS:
        h5ad = PREPARED_DIR / f"{ds}.h5ad"
        if not h5ad.exists():
            continue
        adata = sc.read_h5ad(str(h5ad))
        gene_names = list(adata.var_names)
        scores[ds] = {"gene_names": gene_names}

        # Velorama TSV
        tsv_candidates = list(VELO_DIR.glob(f"*{ds}*interactions*.tsv"))
        if tsv_candidates:
            df_tsv = pd.read_csv(tsv_candidates[0], sep="\t", index_col=0)
            df_tsv.index = df_tsv.index.astype(str)
            df_tsv.columns = df_tsv.columns.astype(str)
            # Align to gene_names: Velorama TSV has rows=targets, cols=regulators
            G = len(gene_names)
            M = np.zeros((G, G))
            name_to_idx = {g: i for i, g in enumerate(gene_names)}
            for tgt_name in df_tsv.index:
                if tgt_name not in name_to_idx:
                    continue
                j = name_to_idx[tgt_name]
                for reg_name in df_tsv.columns:
                    if reg_name not in name_to_idx:
                        continue
                    i = name_to_idx[reg_name]
                    M[i, j] = abs(df_tsv.loc[tgt_name, reg_name])
            scores[ds]["velorama"] = M

        # GeoBRIDGE at each k — need to recompute OR load precomputed
        # For now, recompute from h5ad (~15min per dataset)
        scores[ds]["geobridge_cache"] = _compute_geobridge_scores(adata, gene_names)

    return scores


def _compute_geobridge_scores(adata, gene_names):
    """Compute GeoBRIDGE scores at all K_VALUES (cached across calls)."""
    from geobridge_model import GeoBridgeModel

    X = np.asarray(adata.layers["Ms"] if "Ms" in adata.layers
                   else (adata.X.toarray() if hasattr(adata.X, "toarray")
                         else adata.X),
                   dtype=float)
    V = np.asarray(adata.layers["velocity"], dtype=float)
    V = np.nan_to_num(V, nan=0.0)
    pt = np.asarray(adata.obs["pseudotime"], dtype=float)
    pt = np.nan_to_num(pt, nan=float(np.nanmedian(pt)))

    order = np.argsort(pt)
    X_sorted, V_sorted, pt_sorted = X[order], V[order], pt[order]

    print(f"  Computing pairspace stats (may take 10+ min)...")
    t0 = time.perf_counter()
    pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
        X_sorted, V_sorted, pt_sorted, pair_seed=42,
    )
    print(f"    done in {time.perf_counter()-t0:.1f}s")

    scores_k = {}
    for k in K_VALUES:
        if k >= len(pre["S"]):
            continue
        scores_k[k] = GeoBridgeModel.scores_from_streaming_stats(
            pre, k=k, project_y=True, normalize_score=True, eps=1e-12,
        )
    return scores_k


# ----------------------------------------------------------------------
# Figure A: Overall metrics bar chart
# ----------------------------------------------------------------------

def fig_overall_metrics(gb_df, vr_df, outdir):
    """Bar plot of AUROC / AUPRC / EPR for each (dataset, method, k)."""
    metrics = ["auroc", "auprc", "epr"]
    titles = ["AUROC", "AUPRC", "EPR (Early Precision Ratio)"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Combine into one dataframe with 'method_label'
    rows = []
    if vr_df is not None:
        for ds in DATASETS:
            sub = vr_df[vr_df["dataset"] == ds]
            if len(sub) == 0:
                continue
            for m in metrics:
                rows.append({
                    "dataset": ds, "method_label": "Velorama",
                    "metric": m,
                    "mean": sub[m].mean(), "std": sub[m].std(),
                    "n": len(sub),
                })
    if gb_df is not None:
        for ds in DATASETS:
            for k in FOCUS_K:
                sub = gb_df[(gb_df["dataset"] == ds) & (gb_df["k"] == k)]
                if len(sub) == 0:
                    continue
                for m in metrics:
                    rows.append({
                        "dataset": ds, "method_label": f"GeoBRIDGE k={k}",
                        "metric": m,
                        "mean": sub[m].mean(), "std": sub[m].std(),
                        "n": len(sub),
                    })

    if not rows:
        print("  [SKIP] fig_overall_metrics: no data")
        return
    df = pd.DataFrame(rows)

    methods_order = ["Velorama"] + [f"GeoBRIDGE k={k}" for k in FOCUS_K]
    colors = [COLOR_VELORAMA, COLOR_GB_K0, COLOR_GB_K5, COLOR_GB_K50]
    method_to_color = dict(zip(methods_order, colors))

    for ax, metric, title in zip(axes, metrics, titles):
        sub = df[df["metric"] == metric]
        # Grouped bar
        x_labels = DATASETS
        x = np.arange(len(x_labels))
        width = 0.8 / len(methods_order)

        for i, m in enumerate(methods_order):
            means, stds = [], []
            for ds in x_labels:
                r = sub[(sub["dataset"] == ds) & (sub["method_label"] == m)]
                if len(r) == 0:
                    means.append(np.nan)
                    stds.append(0)
                else:
                    means.append(r["mean"].values[0])
                    stds.append(r["std"].values[0])
            offset = (i - len(methods_order) / 2 + 0.5) * width
            ax.bar(x + offset, means, width, yerr=stds, label=m,
                   color=method_to_color[m], capsize=3, alpha=0.85)

        ax.set_xticks(x)
        ax.set_xticklabels(x_labels)
        ax.set_title(title)
        ax.set_ylabel(title)
        if metric == "auroc":
            ax.axhline(0.5, ls="--", c="k", lw=0.5, alpha=0.5)
            ax.set_ylim(0.3, max(0.7, ax.get_ylim()[1]))
        if metric == "epr":
            ax.axhline(1.0, ls="--", c="k", lw=0.5, alpha=0.5)
        if ax is axes[-1]:
            ax.legend(loc="best", fontsize=8)

    fig.suptitle("GeoBRIDGE vs Velorama: Per-TF metrics (mean ± SD across TFs)")
    fig.tight_layout()
    path = outdir / "figA_overall_metrics.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {path}")


# ----------------------------------------------------------------------
# Figure B: Top-K precision curves
# ----------------------------------------------------------------------

def fig_topk_precision(gb_df, vr_df, outdir):
    """Line plot of precision@K for K in TOP_K_VALUES."""
    fig, axes = plt.subplots(1, len(DATASETS), figsize=(6 * len(DATASETS), 5),
                             squeeze=False)
    axes = axes.flatten()

    for ax, ds in zip(axes, DATASETS):
        # Build (K, mean_precision) for each method
        curves = {}  # method_label -> {K: mean_prec}
        if vr_df is not None:
            sub = vr_df[vr_df["dataset"] == ds]
            if len(sub) > 0:
                curves["Velorama"] = {
                    k: sub.get(f"topk_precision_{k}", pd.Series([np.nan])).mean()
                    for k in TOP_K_VALUES
                }
        if gb_df is not None:
            for k_val in FOCUS_K:
                sub = gb_df[(gb_df["dataset"] == ds) & (gb_df["k"] == k_val)]
                if len(sub) > 0:
                    curves[f"GeoBRIDGE k={k_val}"] = {
                        kk: sub.get(f"topk_precision_{kk}", pd.Series([np.nan])).mean()
                        for kk in TOP_K_VALUES
                    }

        method_to_color = {
            "Velorama": COLOR_VELORAMA,
            f"GeoBRIDGE k=0": COLOR_GB_K0,
            f"GeoBRIDGE k=5": COLOR_GB_K5,
            f"GeoBRIDGE k=50": COLOR_GB_K50,
        }
        for label, d in curves.items():
            xs = sorted(d.keys())
            ys = [d[k] for k in xs]
            ax.plot(xs, ys, "o-", label=label,
                    color=method_to_color.get(label, "gray"), lw=2)

        # Baseline (random)
        if gb_df is not None:
            sub = gb_df[(gb_df["dataset"] == ds) & (gb_df["k"] == 0)]
            if len(sub) > 0:
                baseline = sub["baseline_auprc"].mean()
                ax.axhline(baseline, ls="--", c="k", lw=0.7, alpha=0.5,
                           label=f"Random baseline ({baseline:.4f})")

        ax.set_xlabel("Top-K")
        ax.set_ylabel("Precision@K (mean across TFs)")
        ax.set_title(f"{ds}: Top-K precision")
        ax.set_xscale("log")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    path = outdir / "figB_topk_precision.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {path}")


# ----------------------------------------------------------------------
# Figure C: k-sweep line plot
# ----------------------------------------------------------------------

def fig_k_sweep(gb_df, outdir):
    """For GeoBRIDGE: AUROC / AUPRC / EPR as a function of k."""
    if gb_df is None:
        return

    metrics = ["auroc", "auprc", "epr", "gt_vs_bg_ratio"]
    titles = ["AUROC", "AUPRC", "EPR", "GT/BG score ratio"]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

    for ax, m, t in zip(axes, metrics, titles):
        for ds in DATASETS:
            sub = gb_df[gb_df["dataset"] == ds]
            if len(sub) == 0:
                continue
            by_k = sub.groupby("k")[m].agg(["mean", "std", "count"]).reset_index()
            ax.errorbar(by_k["k"], by_k["mean"],
                        yerr=by_k["std"] / np.sqrt(by_k["count"]),
                        marker="o", label=ds, lw=2, capsize=3)
        ax.set_xlabel("k (pair-space projection dim)")
        ax.set_ylabel(f"{t} (mean ± SEM)")
        ax.set_title(t)
        ax.grid(alpha=0.3)
        if m == "auroc":
            ax.axhline(0.5, ls="--", c="k", lw=0.5, alpha=0.5)
        if m in ("epr", "gt_vs_bg_ratio"):
            ax.axhline(1.0, ls="--", c="k", lw=0.5, alpha=0.5)
        ax.legend(fontsize=8)

    fig.suptitle("GeoBRIDGE: effect of pair-space projection dimension k")
    fig.tight_layout()
    path = outdir / "figC_k_sweep.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {path}")


# ----------------------------------------------------------------------
# Figure D: Runtime comparison
# ----------------------------------------------------------------------

def fig_runtime(outdir, runtime_data):
    """Bar chart of wall times per (dataset, method).

    runtime_data: dict like {(ds, method): seconds}
    """
    if not runtime_data:
        return

    methods = sorted({m for (_, m) in runtime_data.keys()})
    datasets = sorted({d for (d, _) in runtime_data.keys()})

    fig, ax = plt.subplots(figsize=(7, 5))
    x = np.arange(len(datasets))
    width = 0.8 / len(methods)
    color_map = {"Velorama": COLOR_VELORAMA, "GeoBRIDGE": COLOR_GB_K0}

    for i, m in enumerate(methods):
        ys = [runtime_data.get((d, m), 0) / 60.0 for d in datasets]  # to minutes
        offset = (i - len(methods) / 2 + 0.5) * width
        ax.bar(x + offset, ys, width, label=m,
               color=color_map.get(m, "gray"), alpha=0.85)
        for xi, yi in zip(x + offset, ys):
            if yi > 0:
                ax.text(xi, yi + 2, f"{yi:.0f}m", ha="center", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylabel("Wall time (minutes)")
    ax.set_title("Runtime comparison")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    path = outdir / "figD_runtime.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [SAVE] {path}")


# ----------------------------------------------------------------------
# Figure E: Featured TF case study heatmaps
# ----------------------------------------------------------------------

def fig_tf_heatmaps(scores, outdir, top_n=20):
    """
    For each (dataset, featured_tf):
      Heatmap of top-N predicted targets by method (rows) x genes (cols),
      with GT annotation on top.
    """
    import scanpy as sc

    for ds in DATASETS:
        if ds not in scores:
            continue
        gene_names = scores[ds]["gene_names"]
        name_to_idx = {g: i for i, g in enumerate(gene_names)}

        organism = DATASET_ORGANISM.get(ds, "human")
        uppercase = (organism == "mouse")
        net = _load_collectri_regulons(organism=organism)
        gt_dict = _build_ground_truth(net, gene_names, uppercase_match=uppercase)

        featured = FEATURED_TFS.get(ds, [])
        # For mouse: also try title-case variant of featured names
        featured_present = []
        for f in featured:
            if f in name_to_idx:
                featured_present.append(f)
            elif uppercase and f.title() in name_to_idx:
                # Gene stored as 'Neurog3' but GT key is 'NEUROG3'
                featured_present.append(f.title())

        if not featured_present:
            print(f"  [SKIP] {ds}: no featured TFs found in gene_names")
            continue

        fig, axes = plt.subplots(len(featured_present), 1,
                                 figsize=(top_n * 0.35, 2.2 * len(featured_present)),
                                 squeeze=False)
        axes = axes.flatten()

        for ax, tf in zip(axes, featured_present):
            tf_idx = name_to_idx[tf]
            # GT key — if uppercase mode, GT keys are uppercased already by _build_ground_truth
            tf_key = tf.upper() if uppercase else tf
            targets_set = gt_dict.get(tf_key, set())

            # Collect top-N from each available method
            method_rows = []   # list of (label, top_genes_ordered_by_score, scores)
            if "velorama" in scores[ds]:
                M = scores[ds]["velorama"]
                method_rows.append(("Velorama", M[tf_idx]))
            if "geobridge_cache" in scores[ds]:
                for k in FOCUS_K:
                    M = scores[ds]["geobridge_cache"].get(k)
                    if M is not None:
                        method_rows.append((f"GB k={k}", np.abs(M[tf_idx])))

            # Union of top-N from each method
            top_genes = set()
            per_method_top = {}
            for label, s in method_rows:
                s_copy = s.copy()
                s_copy[tf_idx] = -np.inf  # exclude self
                order = np.argsort(-s_copy)
                top_idx = order[:top_n]
                top_genes.update(top_idx)
                per_method_top[label] = (s_copy, top_idx)

            # Order columns by GeoBRIDGE-best k (or first method) score desc
            primary_label = method_rows[-1][0] if method_rows else None
            if primary_label is None:
                continue
            primary_scores = per_method_top[primary_label][0]
            col_order = sorted(top_genes, key=lambda i: -primary_scores[i])
            col_genes = [gene_names[i] for i in col_order]

            # Build matrix: rows = methods, cols = genes; value = normalized score
            mat = np.zeros((len(method_rows), len(col_order)))
            for ri, (label, _) in enumerate(method_rows):
                s, _ = per_method_top[label]
                row_scores = s[col_order]
                # Normalize per row to [0,1]
                if row_scores.max() > row_scores.min():
                    mat[ri] = (row_scores - row_scores.min()) / (row_scores.max() - row_scores.min())

            # GT annotation row
            gt_mask = np.array([(col_genes[j] in targets_set or
                                col_genes[j].upper() in targets_set)
                                for j in range(len(col_genes))])

            # Plot
            im = ax.imshow(mat, aspect="auto", cmap="viridis", vmin=0, vmax=1)
            ax.set_yticks(range(len(method_rows)))
            ax.set_yticklabels([m[0] for m in method_rows], fontsize=8)
            ax.set_xticks(range(len(col_genes)))
            ax.set_xticklabels(col_genes, rotation=90, fontsize=7)

            # Mark GT targets with red boxes
            for j, is_gt in enumerate(gt_mask):
                if is_gt:
                    for ri in range(len(method_rows)):
                        ax.add_patch(plt.Rectangle((j - 0.5, ri - 0.5), 1, 1,
                                                   fill=False, edgecolor="red", lw=1.5))

            n_gt_in_top = int(gt_mask.sum())
            ax.set_title(f"{ds} | TF = {tf}  (GT targets in union: {n_gt_in_top}/{len(col_genes)})",
                         fontsize=9, loc="left")

        fig.suptitle(f"Featured TF case study — {ds}", fontsize=11)
        fig.tight_layout()
        path = outdir / f"figE_{ds}_tf_heatmaps.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVE] {path}")


# ----------------------------------------------------------------------
# Figure F: Score distribution (GT vs BG)
# ----------------------------------------------------------------------

def fig_score_distribution(scores, outdir):
    """For featured TFs: histogram of GT vs non-GT scores."""
    for ds in DATASETS:
        if ds not in scores:
            continue
        gene_names = scores[ds]["gene_names"]
        name_to_idx = {g: i for i, g in enumerate(gene_names)}

        organism = DATASET_ORGANISM.get(ds, "human")
        uppercase = (organism == "mouse")
        net = _load_collectri_regulons(organism=organism)
        gt_dict = _build_ground_truth(net, gene_names, uppercase_match=uppercase)

        featured = FEATURED_TFS.get(ds, [])
        featured_present = [f for f in featured
                            if f in name_to_idx or
                            (uppercase and f.title() in name_to_idx)]
        if not featured_present:
            continue

        # Methods to compare
        methods_to_show = []
        if "velorama" in scores[ds]:
            methods_to_show.append(("Velorama", scores[ds]["velorama"]))
        if "geobridge_cache" in scores[ds]:
            gb_cache = scores[ds]["geobridge_cache"]
            if 0 in gb_cache:
                methods_to_show.append(("GB k=0", np.abs(gb_cache[0])))
            if 5 in gb_cache:
                methods_to_show.append(("GB k=5", np.abs(gb_cache[5])))

        n_rows = len(featured_present)
        n_cols = len(methods_to_show)
        if n_cols == 0 or n_rows == 0:
            continue

        fig, axes = plt.subplots(n_rows, n_cols,
                                 figsize=(4 * n_cols, 2.2 * n_rows),
                                 squeeze=False)

        for ri, tf in enumerate(featured_present):
            tf_key = tf.upper() if uppercase else tf
            if uppercase and tf in name_to_idx:
                tf_name_in_data = tf
            elif uppercase:
                tf_name_in_data = tf.title()
            else:
                tf_name_in_data = tf

            tf_idx = name_to_idx.get(tf_name_in_data)
            if tf_idx is None:
                continue
            targets_set = gt_dict.get(tf_key, set())

            for ci, (label, M) in enumerate(methods_to_show):
                ax = axes[ri, ci]
                s = M[tf_idx].copy()
                s[tf_idx] = np.nan
                labels = np.array([
                    1 if (gene_names[j] in targets_set or
                          gene_names[j].upper() in targets_set) else 0
                    for j in range(len(gene_names))
                ])
                labels[tf_idx] = -1

                gt_scores = s[labels == 1]
                bg_scores = s[labels == 0]
                gt_scores = gt_scores[~np.isnan(gt_scores)]
                bg_scores = bg_scores[~np.isnan(bg_scores)]

                if len(gt_scores) == 0:
                    ax.text(0.5, 0.5, "No GT targets", ha="center", va="center",
                            transform=ax.transAxes)
                    continue

                bins = np.linspace(
                    min(gt_scores.min(), bg_scores.min()),
                    max(gt_scores.max(), bg_scores.max()),
                    30,
                )
                ax.hist(bg_scores, bins=bins, alpha=0.5, color=COLOR_BG,
                        label=f"BG (n={len(bg_scores)})", density=True)
                ax.hist(gt_scores, bins=bins, alpha=0.7, color=COLOR_GT,
                        label=f"GT (n={len(gt_scores)})", density=True)
                ax.axvline(gt_scores.mean(), color=COLOR_GT, ls="--", lw=1)
                ax.axvline(bg_scores.mean(), color=COLOR_BG, ls="--", lw=1)
                ax.set_title(f"{tf} | {label}", fontsize=9)
                ax.set_xlabel("|score|", fontsize=8)
                if ci == 0:
                    ax.set_ylabel("density", fontsize=8)
                ax.legend(fontsize=6)

        fig.suptitle(f"GT vs Background score distribution — {ds}")
        fig.tight_layout()
        path = outdir / f"figF_{ds}_score_distribution.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVE] {path}")


# ----------------------------------------------------------------------
# Figure G: Venn diagrams (GB ∩ Velorama ∩ GT) per featured TF
# ----------------------------------------------------------------------

def fig_venn_diagrams(scores, outdir, top_n=50):
    """Per featured TF: Venn of GB top-N, Velorama top-N, and GT target set."""
    try:
        from matplotlib_venn import venn3
    except ImportError:
        print("  [SKIP] fig_venn_diagrams: matplotlib_venn not installed")
        return

    for ds in DATASETS:
        if ds not in scores:
            continue
        gene_names = scores[ds]["gene_names"]
        name_to_idx = {g: i for i, g in enumerate(gene_names)}

        organism = DATASET_ORGANISM.get(ds, "human")
        uppercase = (organism == "mouse")
        net = _load_collectri_regulons(organism=organism)
        gt_dict = _build_ground_truth(net, gene_names, uppercase_match=uppercase)

        featured = FEATURED_TFS.get(ds, [])
        featured_present = [f for f in featured
                            if f in name_to_idx or
                            (uppercase and f.title() in name_to_idx)]
        if not featured_present:
            continue

        n = len(featured_present)
        fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
        axes = axes.flatten()

        for ax, tf in zip(axes, featured_present):
            tf_key = tf.upper() if uppercase else tf
            tf_name_in_data = tf if (tf in name_to_idx) else tf.title()
            tf_idx = name_to_idx[tf_name_in_data]
            gt_targets = gt_dict.get(tf_key, set())

            # Velorama top-N
            velo_top = set()
            if "velorama" in scores[ds]:
                M = scores[ds]["velorama"]
                s = M[tf_idx].copy()
                s[tf_idx] = -np.inf
                velo_top = set(gene_names[i] for i in np.argsort(-s)[:top_n])

            # GeoBRIDGE top-N (k=0 as baseline representative)
            gb_top = set()
            if "geobridge_cache" in scores[ds]:
                M = scores[ds]["geobridge_cache"].get(0)
                if M is not None:
                    s = np.abs(M[tf_idx])
                    s[tf_idx] = -np.inf
                    gb_top = set(gene_names[i] for i in np.argsort(-s)[:top_n])

            # GT set (may use uppercase keys — map back to gene name case)
            gt_set_cased = set()
            for t in gt_targets:
                if t in name_to_idx:
                    gt_set_cased.add(t)
                elif uppercase:
                    # try title-case
                    if t.title() in name_to_idx:
                        gt_set_cased.add(t.title())

            v = venn3([gb_top, velo_top, gt_set_cased],
                      set_labels=("GeoBRIDGE", "Velorama", "GT"), ax=ax)
            ax.set_title(f"{tf}  (top-{top_n})", fontsize=10)

        fig.suptitle(f"Predicted vs GT targets — {ds}")
        fig.tight_layout()
        path = outdir / f"figG_{ds}_venn.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVE] {path}")


# ----------------------------------------------------------------------
# Figure H: GO enrichment (optional — requires gseapy)
# ----------------------------------------------------------------------

def fig_go_enrichment(scores, outdir, top_n=50):
    """GO BP enrichment for top-N predicted targets per featured TF.

    Requires gseapy: pip install gseapy
    """
    try:
        import gseapy as gp
    except ImportError:
        print("  [SKIP] fig_go_enrichment: gseapy not installed")
        return

    for ds in DATASETS:
        if ds not in scores:
            continue
        gene_names = scores[ds]["gene_names"]
        name_to_idx = {g: i for i, g in enumerate(gene_names)}

        organism = DATASET_ORGANISM.get(ds, "human")
        uppercase = (organism == "mouse")
        featured = FEATURED_TFS.get(ds, [])
        featured_present = [f for f in featured
                            if f in name_to_idx or
                            (uppercase and f.title() in name_to_idx)]

        all_enrich = []  # rows of (tf, method, term, pval, nes)

        for tf in featured_present:
            tf_name_in_data = tf if (tf in name_to_idx) else tf.title()
            tf_idx = name_to_idx[tf_name_in_data]

            for method_label, M in [
                ("Velorama", scores[ds].get("velorama")),
                ("GB k=0", np.abs(scores[ds].get("geobridge_cache", {}).get(0,
                    np.zeros((len(gene_names), len(gene_names))))) if
                    scores[ds].get("geobridge_cache", {}).get(0) is not None else None),
            ]:
                if M is None:
                    continue
                s = M[tf_idx].copy()
                s[tf_idx] = -np.inf
                top_genes = [gene_names[i] for i in np.argsort(-s)[:top_n]]
                if uppercase:
                    top_genes = [g.upper() for g in top_genes]

                try:
                    enr = gp.enrichr(gene_list=top_genes,
                                     gene_sets=["GO_Biological_Process_2023"],
                                     organism="Human",
                                     outdir=None, no_plot=True)
                    for _, r in enr.results.head(5).iterrows():
                        all_enrich.append({
                            "tf": tf, "method": method_label,
                            "term": r["Term"],
                            "adj_pval": r["Adjusted P-value"],
                            "combined_score": r["Combined Score"],
                        })
                except Exception as e:
                    print(f"    [WARN] enrichr failed for {tf} / {method_label}: {e}")

        if not all_enrich:
            continue

        enr_df = pd.DataFrame(all_enrich)
        enr_df.to_csv(outdir / f"figH_{ds}_go_enrichment.csv", index=False)

        # Bubble plot (simple)
        import seaborn as sns
        fig, ax = plt.subplots(figsize=(10, max(4, 0.4 * len(enr_df))))
        enr_df["-log10_padj"] = -np.log10(enr_df["adj_pval"].clip(lower=1e-20))
        sns.scatterplot(
            data=enr_df, x="method", y="term",
            size="-log10_padj", hue="tf", ax=ax, sizes=(20, 300), alpha=0.7,
        )
        ax.set_title(f"GO BP enrichment — {ds} (top-{top_n} targets)")
        ax.tick_params(axis='y', labelsize=7)
        fig.tight_layout()
        path = outdir / f"figH_{ds}_go_enrichment.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVE] {path}")


# ----------------------------------------------------------------------
# Figure I: Stream plot (UMAP + velocity)
# ----------------------------------------------------------------------

def fig_stream_plots(outdir):
    import scanpy as sc
    import scvelo as scv

    for ds in DATASETS:
        h5ad = PREPARED_DIR / f"{ds}.h5ad"
        if not h5ad.exists():
            continue
        adata = sc.read_h5ad(str(h5ad))

        # Need UMAP
        if "X_umap" not in adata.obsm:
            sc.pp.pca(adata)
            sc.pp.neighbors(adata, use_rep="X_pca")
            sc.tl.umap(adata)
        # Velocity graph for stream
        try:
            scv.tl.velocity_graph(adata)
        except Exception:
            pass

        fig, ax = plt.subplots(figsize=(6, 5))
        try:
            scv.pl.velocity_embedding_stream(
                adata, basis="umap", color="clusters" if "clusters" in adata.obs
                else "pseudotime",
                ax=ax, show=False, title=ds, legend_loc="right margin",
            )
        except Exception as e:
            print(f"    [WARN] stream plot failed for {ds}: {e}")
            sc.pl.umap(adata, color="pseudotime", ax=ax, show=False, title=ds)

        path = outdir / f"figI_{ds}_stream.png"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  [SAVE] {path}")


# ----------------------------------------------------------------------
# Runtime parsing from logs
# ----------------------------------------------------------------------

def parse_runtime_from_logs():
    """Scan log files in OUTDIR and extract total wall time per run."""
    runtimes = {}  # (dataset, method) -> seconds

    for log_name, (ds, method) in [
        ("velorama_bm.log", ("bonemarrow", "Velorama")),
        ("velorama_pan.log", ("pancreas", "Velorama")),
        ("geobridge.log", (None, "GeoBRIDGE")),
    ]:
        log_path = OUTDIR / log_name
        if not log_path.exists():
            continue
        text = log_path.read_text(errors="ignore")

        if method == "Velorama":
            # Look for "Total time: <sec>"
            import re
            m = re.search(r"Total time:\s*([\d.]+)", text)
            if m:
                runtimes[(ds, method)] = float(m.group(1))
        elif method == "GeoBRIDGE":
            # Extract "Stats: Xs, ..." per dataset
            import re
            lines = text.split("\n")
            current_ds = None
            total_per_ds = {}
            for line in lines:
                if "GEOBRIDGE on " in line:
                    current_ds = line.split("GEOBRIDGE on ")[-1].strip()
                if current_ds and "Stats:" in line:
                    m = re.search(r"Stats:\s*([\d.]+)s", line)
                    if m:
                        total_per_ds[current_ds] = float(m.group(1))
            for dsname, secs in total_per_ds.items():
                runtimes[(dsname, "GeoBRIDGE")] = secs

    return runtimes


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/bm_real_comparison/figures")
    ap.add_argument("--skip-heavy", action="store_true",
                    help="Skip figures requiring GeoBRIDGE score recomputation "
                         "(heatmaps, Venn, score dist, GO, stream)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print("ANALYZE REAL DATA COMPARISON")
    print(f"{'=' * 70}\n")

    # Load tabular data
    print("[load] GeoBRIDGE results...")
    gb_df = load_geobridge_results()
    print(f"  {len(gb_df) if gb_df is not None else 0} rows")

    print("[load] Velorama results...")
    vr_df = load_velorama_results()
    print(f"  {len(vr_df) if vr_df is not None else 0} rows")

    # Fig A: Overall metrics
    print("\n[FIG A] Overall metrics bar...")
    fig_overall_metrics(gb_df, vr_df, outdir)

    # Fig B: Top-K precision
    print("[FIG B] Top-K precision curves...")
    fig_topk_precision(gb_df, vr_df, outdir)

    # Fig C: k-sweep
    print("[FIG C] k-sweep line plot...")
    fig_k_sweep(gb_df, outdir)

    # Fig D: Runtime
    print("[FIG D] Runtime bar...")
    runtimes = parse_runtime_from_logs()
    print(f"  Parsed runtimes: {runtimes}")
    fig_runtime(outdir, runtimes)

    # Heavy figures: need score matrices
    if not args.skip_heavy:
        print("\n[load] Score matrices (computing GeoBRIDGE from h5ads)...")
        scores = load_score_matrices()

        print("\n[FIG E] TF case-study heatmaps...")
        fig_tf_heatmaps(scores, outdir)

        print("[FIG F] Score distribution (GT vs BG)...")
        fig_score_distribution(scores, outdir)

        print("[FIG G] Venn diagrams...")
        fig_venn_diagrams(scores, outdir)

        print("[FIG H] GO enrichment...")
        fig_go_enrichment(scores, outdir)

        print("[FIG I] Stream plots...")
        fig_stream_plots(outdir)
    else:
        print("\n[SKIP] Heavy figures (use without --skip-heavy to generate)")

    print(f"\n[DONE] Figures in {outdir}/")


if __name__ == "__main__":
    main()
