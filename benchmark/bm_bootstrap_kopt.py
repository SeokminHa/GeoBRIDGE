"""
Bootstrap stability of per-TF k_opt on bone marrow data.

Null hypothesis: "Most apparent per-TF k_opt heterogeneity is noise,
and a global k in a reasonable low-to-moderate range is sufficient."

Design:
  - 20 bootstrap replicates, each subsampling 80% of cells
  - Reuse existing velocity (recomputing velocity per bootstrap is
    prohibitively expensive and not the variable under test — the
    question is whether k_opt is stable given fixed biology but
    variable cell sampling)
  - Recompute streaming pairspace stats per subsample
  - Evaluate per-TF AUPRC across k grid
  - Report k_opt distributions, stability metrics, and null hypothesis test

Output:
  results/bm_bootstrap_k/csv/bootstrap_kopt_per_tf.csv
  results/bm_bootstrap_k/csv/bootstrap_auprc_full.csv
  results/bm_bootstrap_k/csv/bootstrap_summary.csv
  results/bm_bootstrap_k/figures/bootstrap_kopt_distributions.png
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from geobridge_model import GeoBridgeModel

# ===================================================================

K_GRID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]

KEY_TFS = ["ERG", "FLI1", "GATA2", "SPI1", "KLF1", "MYB", "MEIS1"]

N_BOOTSTRAP = 20
SUBSAMPLE_FRAC = 0.80


def load_and_preprocess(data_dir):
    """Load bone marrow, preprocess, compute velocity, return X, V, pt, genes."""
    import warnings
    warnings.filterwarnings("ignore")
    import scanpy as sc
    import scvelo as scv

    cache = Path(data_dir) / "human_cd34_bone_marrow.h5ad" if data_dir else None
    if cache and cache.exists():
        adata = sc.read_h5ad(cache)
    else:
        adata = scv.datasets.bonemarrow()
        if cache:
            adata.write_h5ad(cache)

    scv.pp.filter_genes(adata, min_shared_counts=20)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    print("[BM] Computing stochastic velocity...")
    t0 = time.perf_counter()
    scv.tl.velocity(adata, mode="stochastic")
    print(f"[BM] Velocity done in {time.perf_counter()-t0:.0f}s")

    X = np.asarray(adata.layers["Ms"], dtype=float)
    V = np.asarray(adata.layers["velocity"], dtype=float)
    V = np.nan_to_num(V, nan=0.0)
    genes = list(adata.var_names)

    # Pseudotime
    if "palantir_pseudotime" in adata.obs.columns:
        pt = np.asarray(adata.obs["palantir_pseudotime"], dtype=float)
    else:
        sc.tl.diffmap(adata)
        if "CD34" in genes:
            adata.uns["iroot"] = np.argmax(X[:, genes.index("CD34")])
        else:
            adata.uns["iroot"] = 0
        sc.tl.dpt(adata)
        pt = np.asarray(adata.obs["dpt_pseudotime"], dtype=float)
    pt = np.nan_to_num(pt, nan=float(np.nanmedian(pt)))

    print(f"[BM] {X.shape[0]} cells x {X.shape[1]} genes")
    return X, V, pt, genes


def load_ground_truth(genes):
    """Load CollecTRI edges, return tf_targets dict."""
    import decoupler as dc

    if hasattr(dc, "op") and hasattr(dc.op, "collectri"):
        net = dc.op.collectri(organism="human")
    elif hasattr(dc, "get_collectri"):
        net = dc.get_collectri(organism="human", split_complexes=False)
    else:
        raise RuntimeError("Cannot load CollecTRI")

    gene_set = set(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    tf_targets = {}
    for tf in net["source"].unique():
        if tf not in gene_set:
            continue
        targets = net[net["source"] == tf]["target"].values
        matched = [gene_to_idx[t] for t in targets if t in gene_set and t != tf]
        if len(matched) >= 5:
            tf_targets[tf] = matched

    print(f"[BM] {len(tf_targets)} evaluable TFs, {len(net)} CollecTRI edges")
    return tf_targets, gene_to_idx


def compute_per_tf_auprc(S_score, tf_idx, target_idxs, G):
    """Compute AUPRC for one TF row."""
    from sklearn.metrics import average_precision_score

    target_set = set(target_idxs)
    scores = []
    labels = []
    for j in range(G):
        if j == tf_idx:
            continue
        scores.append(abs(S_score[tf_idx, j]))
        labels.append(1 if j in target_set else 0)
    scores = np.array(scores)
    labels = np.array(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan
    return average_precision_score(labels, scores)


def run_one_bootstrap(X, V, pt, genes, tf_targets, gene_to_idx, rng, rep_id,
                      subsample_frac=0.80):
    """Run one bootstrap replicate: subsample cells, recompute stats, evaluate."""
    n_cells = X.shape[0]
    G = X.shape[1]
    n_sub = int(n_cells * subsample_frac)

    # Subsample cells without replacement
    cell_idx = rng.choice(n_cells, size=n_sub, replace=False)
    X_sub = X[cell_idx]
    V_sub = V[cell_idx]
    pt_sub = pt[cell_idx]

    # Sort by pseudotime so streaming stats can use fast monotonic path
    sort_order = np.argsort(pt_sub)
    X_sub = X_sub[sort_order]
    V_sub = V_sub[sort_order]
    pt_sub = pt_sub[sort_order]

    # Recompute pairspace stats on subsampled cells
    # No max_pairs_per_cell so the parallel (threaded) path is used
    pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
        X_sub, V_sub, pt_sub, pair_seed=42 + rep_id,
    )

    rows = []
    for k in K_GRID:
        if k >= len(pre["S"]):
            continue

        S_k = GeoBridgeModel.scores_from_streaming_stats(
            pre, k=k, project_y=True, normalize_score=True, eps=1e-12,
        )

        for tf, target_idxs in tf_targets.items():
            tf_idx = gene_to_idx[tf]
            auprc = compute_per_tf_auprc(S_k, tf_idx, target_idxs, G)
            rows.append({
                "rep": rep_id,
                "k": k,
                "tf": tf,
                "auprc": auprc,
                "n_targets": len(target_idxs),
                "is_key_tf": tf in KEY_TFS,
            })

    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="E:/data_cache")
    ap.add_argument("--outdir", default="results/bm_bootstrap_k")
    ap.add_argument("--n-bootstrap", type=int, default=N_BOOTSTRAP)
    ap.add_argument("--subsample-frac", type=float, default=SUBSAMPLE_FRAC)
    args = ap.parse_args()

    n_boot = args.n_bootstrap
    sub_frac = args.subsample_frac

    outdir = Path(args.outdir)
    csvdir = outdir / "csv"
    figdir = outdir / "figures"
    csvdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # Load data (velocity computed once, reused across bootstraps)
    X, V, pt, genes = load_and_preprocess(args.data_dir)
    tf_targets, gene_to_idx = load_ground_truth(genes)
    G = len(genes)

    print(f"\n[BOOTSTRAP] {n_boot} replicates, {sub_frac:.0%} cell subsample")
    print(f"[BOOTSTRAP] {len(tf_targets)} TFs, {len(K_GRID)} k values per replicate")

    rng = np.random.default_rng(seed=2024)

    all_rows = []
    for rep in range(n_boot):
        t_rep = time.perf_counter()
        rows = run_one_bootstrap(X, V, pt, genes, tf_targets, gene_to_idx, rng, rep,
                                 subsample_frac=sub_frac)
        all_rows.extend(rows)
        elapsed = time.perf_counter() - t_rep
        # Brief progress
        if (rep + 1) % 5 == 0 or rep == 0:
            n_rows = len(rows)
            print(f"  rep {rep+1:2d}/{n_boot}: {n_rows} rows, {elapsed:.1f}s")

    df = pd.DataFrame(all_rows)
    df.to_csv(csvdir / "bootstrap_auprc_full.csv", index=False)
    print(f"\n[SAVE] bootstrap_auprc_full.csv ({len(df)} rows)")

    # ===================================================================
    # Compute k_opt per TF per replicate
    # ===================================================================
    kopt_rows = []
    for (rep, tf), grp in df.groupby(["rep", "tf"]):
        best_row = grp.loc[grp["auprc"].idxmax()]
        kopt_rows.append({
            "rep": rep,
            "tf": tf,
            "k_opt": int(best_row["k"]),
            "best_auprc": best_row["auprc"],
            "n_targets": int(best_row["n_targets"]),
            "is_key_tf": best_row["is_key_tf"],
            "auprc_at_k0": float(grp.loc[grp["k"] == 0, "auprc"].values[0])
                if 0 in grp["k"].values else np.nan,
        })
    kopt_df = pd.DataFrame(kopt_rows)
    kopt_df.to_csv(csvdir / "bootstrap_kopt_per_tf.csv", index=False)
    print(f"[SAVE] bootstrap_kopt_per_tf.csv ({len(kopt_df)} rows)")

    # ===================================================================
    # Summary statistics per TF
    # ===================================================================
    summary_rows = []
    for tf, grp in kopt_df.groupby("tf"):
        kopt_vals = grp["k_opt"].values
        summary_rows.append({
            "tf": tf,
            "n_targets": int(grp["n_targets"].iloc[0]),
            "is_key_tf": bool(grp["is_key_tf"].iloc[0]),
            "k_opt_median": float(np.median(kopt_vals)),
            "k_opt_mean": float(np.mean(kopt_vals)),
            "k_opt_std": float(np.std(kopt_vals)),
            "k_opt_iqr_lo": float(np.percentile(kopt_vals, 25)),
            "k_opt_iqr_hi": float(np.percentile(kopt_vals, 75)),
            "k_opt_min": int(np.min(kopt_vals)),
            "k_opt_max": int(np.max(kopt_vals)),
            "k_opt_iqr": float(np.percentile(kopt_vals, 75) - np.percentile(kopt_vals, 25)),
            "k_opt_range": int(np.max(kopt_vals) - np.min(kopt_vals)),
            "frac_k_leq_5": float(np.mean(kopt_vals <= 5)),
            "frac_k_leq_8": float(np.mean(kopt_vals <= 8)),
            "best_auprc_mean": float(grp["best_auprc"].mean()),
            "best_auprc_std": float(grp["best_auprc"].std()),
            "k0_auprc_mean": float(grp["auprc_at_k0"].mean()),
            "k0_auprc_std": float(grp["auprc_at_k0"].std()),
        })
    summary_df = pd.DataFrame(summary_rows)
    summary_df = summary_df.sort_values("n_targets", ascending=False)
    summary_df.to_csv(csvdir / "bootstrap_summary.csv", index=False)
    print(f"[SAVE] bootstrap_summary.csv ({len(summary_df)} rows)")

    # ===================================================================
    # Compute per-TF AUPRC mean and std at each k (for curve smoothness analysis)
    # ===================================================================
    auprc_stats = df.groupby(["tf", "k"])["auprc"].agg(["mean", "std"]).reset_index()
    auprc_stats.to_csv(csvdir / "bootstrap_auprc_stats_per_k.csv", index=False)
    print(f"[SAVE] bootstrap_auprc_stats_per_k.csv")

    # ===================================================================
    # Print results
    # ===================================================================
    print("\n" + "=" * 90)
    print("BOOTSTRAP k_opt STABILITY: KEY HEMATOPOIETIC TFs")
    print("=" * 90)

    key_summary = summary_df[summary_df["is_key_tf"]].sort_values("n_targets", ascending=False)
    print(f"\n  {'TF':<8s}  {'n_tgt':>5s}  {'k_med':>5s}  {'k_mean':>6s}  "
          f"{'k_std':>5s}  {'IQR':>10s}  {'range':>5s}  "
          f"{'%k≤5':>5s}  {'%k≤8':>5s}  {'best_auprc':>10s}")
    print(f"  {'':8s}  {'':5s}  {'':5s}  {'':6s}  "
          f"{'':5s}  {'':10s}  {'':5s}  "
          f"{'':5s}  {'':5s}  {'(mean±std)':>10s}")
    print("  " + "-" * 86)

    for _, r in key_summary.iterrows():
        iqr_str = f"[{r['k_opt_iqr_lo']:.0f}, {r['k_opt_iqr_hi']:.0f}]"
        ap_str = f"{r['best_auprc_mean']:.4f}±{r['best_auprc_std']:.4f}"
        print(f"  {r['tf']:<8s}  {r['n_targets']:5d}  {r['k_opt_median']:5.0f}  "
              f"{r['k_opt_mean']:6.1f}  {r['k_opt_std']:5.1f}  "
              f"{iqr_str:>10s}  {r['k_opt_range']:5d}  "
              f"{r['frac_k_leq_5']:5.0%}  {r['frac_k_leq_8']:5.0%}  "
              f"{ap_str:>10s}")

    # ===================================================================
    # Full k_opt distribution for key TFs
    # ===================================================================
    print("\n" + "=" * 90)
    print("k_opt DISTRIBUTION (histogram) FOR KEY TFs")
    print("=" * 90)

    for tf in KEY_TFS:
        tf_kopt = kopt_df[kopt_df["tf"] == tf]
        if len(tf_kopt) == 0:
            continue
        kvals = tf_kopt["k_opt"].values
        n_tgt = tf_kopt["n_targets"].iloc[0]
        counts = {}
        for kv in kvals:
            counts[kv] = counts.get(kv, 0) + 1
        hist_str = "  ".join(f"k={k}:{c}" for k, c in sorted(counts.items()))
        print(f"\n  {tf} (n_targets={n_tgt}):")
        print(f"    {hist_str}")
        print(f"    median={np.median(kvals):.0f}, IQR=[{np.percentile(kvals,25):.0f}, "
              f"{np.percentile(kvals,75):.0f}], range=[{min(kvals)}, {max(kvals)}]")

    # ===================================================================
    # Relationship: target count vs k_opt stability
    # ===================================================================
    print("\n" + "=" * 90)
    print("TARGET COUNT vs k_opt STABILITY (all 63 TFs)")
    print("=" * 90)

    # Bin TFs by target count
    all_summary = summary_df.copy()
    bins = [0, 10, 20, 40, 100, 1000]
    labels = ["5-9", "10-19", "20-39", "40-99", "100+"]
    all_summary["tgt_bin"] = pd.cut(all_summary["n_targets"], bins=bins, labels=labels)
    bin_stats = all_summary.groupby("tgt_bin", observed=True).agg({
        "k_opt_std": "mean",
        "k_opt_iqr": "mean",
        "k_opt_range": "mean",
        "tf": "count",
    }).rename(columns={"tf": "n_tfs"})
    print(f"\n  {'Bin':<8s}  {'n_TFs':>5s}  {'mean_k_std':>10s}  {'mean_IQR':>8s}  {'mean_range':>10s}")
    print("  " + "-" * 46)
    for idx, r in bin_stats.iterrows():
        print(f"  {idx:<8s}  {r['n_tfs']:5.0f}  {r['k_opt_std']:10.1f}  "
              f"{r['k_opt_iqr']:8.1f}  {r['k_opt_range']:10.1f}")

    # ===================================================================
    # Relationship: target count vs AUPRC curve smoothness
    # (smoothness = mean std of AUPRC across bootstrap at each k)
    # ===================================================================
    print("\n" + "=" * 90)
    print("TARGET COUNT vs AUPRC CURVE SMOOTHNESS (all TFs)")
    print("=" * 90)

    # For each TF, compute the mean bootstrap std across all k values
    smooth_df = auprc_stats.groupby("tf")["std"].mean().reset_index()
    smooth_df.columns = ["tf", "mean_auprc_std"]
    smooth_df = smooth_df.merge(summary_df[["tf", "n_targets"]], on="tf")
    smooth_df["tgt_bin"] = pd.cut(smooth_df["n_targets"], bins=bins, labels=labels)
    smooth_stats = smooth_df.groupby("tgt_bin", observed=True).agg({
        "mean_auprc_std": "mean",
        "tf": "count",
    }).rename(columns={"tf": "n_tfs"})
    print(f"\n  {'Bin':<8s}  {'n_TFs':>5s}  {'mean_auprc_bootstrap_std':>25s}")
    print("  " + "-" * 38)
    for idx, r in smooth_stats.iterrows():
        print(f"  {idx:<8s}  {r['n_tfs']:5.0f}  {r['mean_auprc_bootstrap_std']:25.6f}")

    # ===================================================================
    # Null hypothesis test: global k adequacy
    # ===================================================================
    print("\n" + "=" * 90)
    print("NULL HYPOTHESIS TEST: IS GLOBAL k ADEQUATE?")
    print("=" * 90)

    # For each replicate, compute the mean AUPRC across all TFs at each k
    # Then find global k_opt per replicate
    global_kopt_per_rep = []
    for rep in range(n_boot):
        rep_df = df[df["rep"] == rep]
        mean_by_k = rep_df.groupby("k")["auprc"].mean()
        best_k = mean_by_k.idxmax()
        global_kopt_per_rep.append(best_k)
    global_kopt_per_rep = np.array(global_kopt_per_rep)

    print(f"\n  Global k_opt across {n_boot} replicates:")
    print(f"    median={np.median(global_kopt_per_rep):.0f}, "
          f"mean={np.mean(global_kopt_per_rep):.1f}, "
          f"std={np.std(global_kopt_per_rep):.1f}")
    g_counts = {}
    for k in global_kopt_per_rep:
        g_counts[k] = g_counts.get(k, 0) + 1
    print(f"    distribution: {', '.join(f'k={k}:{c}' for k, c in sorted(g_counts.items()))}")

    # For each key TF, compute: "regret" of using global k_opt=median
    # vs the per-TF k_opt
    print(f"\n  Regret analysis: AUPRC loss from using global k vs per-TF k_opt")
    global_k_median = int(np.median(global_kopt_per_rep))
    print(f"  (global k = {global_k_median}, the median of bootstrap global k_opt)")

    print(f"\n  {'TF':<8s}  {'n_tgt':>5s}  {'AUPRC@global':>13s}  {'AUPRC@tfopt':>12s}  "
          f"{'regret':>8s}  {'regret%':>8s}  {'tf_k_med':>8s}")
    print("  " + "-" * 72)

    for tf in KEY_TFS:
        tf_data = df[df["tf"] == tf]
        if len(tf_data) == 0:
            continue
        # Mean AUPRC at global k across bootstraps
        at_global = tf_data[tf_data["k"] == global_k_median]["auprc"]
        # Mean best AUPRC (at per-TF k_opt each rep)
        tf_kopts = kopt_df[kopt_df["tf"] == tf]
        best_per_rep = tf_kopts["best_auprc"]
        n_tgt = int(tf_data["n_targets"].iloc[0])
        tf_k_med = float(tf_kopts["k_opt"].median())

        auprc_global_mean = at_global.mean()
        auprc_best_mean = best_per_rep.mean()
        regret = auprc_best_mean - auprc_global_mean
        regret_pct = regret / auprc_best_mean * 100 if auprc_best_mean > 0 else 0

        print(f"  {tf:<8s}  {n_tgt:5d}  {auprc_global_mean:13.4f}  "
              f"{auprc_best_mean:12.4f}  {regret:8.4f}  {regret_pct:7.1f}%  "
              f"{tf_k_med:8.0f}")

    # Also compute: what fraction of all-TF variance in AUPRC is explained by k?
    # vs by bootstrap replicate? (if most variance is from replicate, not k, then
    # k choice matters less)
    print(f"\n  Variance decomposition: how much does k matter vs cell sampling?")
    for tf in KEY_TFS:
        tf_data = df[df["tf"] == tf]
        if len(tf_data) == 0:
            continue
        n_tgt = int(tf_data["n_targets"].iloc[0])
        # Pivot: rows=rep, cols=k, values=auprc
        piv = tf_data.pivot_table(index="rep", columns="k", values="auprc")
        total_var = piv.values.var()
        # Variance across k (mean over reps, then var across k)
        var_across_k = piv.mean(axis=0).var()
        # Variance across reps (mean over k, then var across reps)
        var_across_reps = piv.mean(axis=1).var()
        pct_k = var_across_k / total_var * 100 if total_var > 0 else 0
        pct_rep = var_across_reps / total_var * 100 if total_var > 0 else 0
        print(f"    {tf:<8s} (n={n_tgt:2d}): total_var={total_var:.2e}, "
              f"var_from_k={pct_k:5.1f}%, var_from_rep={pct_rep:5.1f}%")

    # ===================================================================
    # FLI1 specific analysis
    # ===================================================================
    print("\n" + "=" * 90)
    print("FLI1 ANOMALY CHECK")
    print("=" * 90)

    fli1_data = df[df["tf"] == "FLI1"]
    if len(fli1_data) > 0:
        # Mean AUPRC at each k across bootstraps
        fli1_means = fli1_data.groupby("k")["auprc"].agg(["mean", "std"])
        print(f"\n  FLI1 bootstrap AUPRC (mean ± std across {n_boot} reps):")
        print(f"  {'k':>3s}  {'mean':>8s}  {'std':>8s}  {'CV':>8s}")
        for k, row in fli1_means.iterrows():
            cv = row["std"] / row["mean"] if row["mean"] > 0 else np.nan
            marker = ""
            if k == fli1_means["mean"].idxmax():
                marker = " <-- best"
            print(f"  {k:3d}  {row['mean']:8.4f}  {row['std']:8.4f}  {cv:8.2f}{marker}")

        # Does k=1 consistently beat k=0?
        fli1_k0 = fli1_data[fli1_data["k"] == 0].set_index("rep")["auprc"]
        fli1_k1 = fli1_data[fli1_data["k"] == 1].set_index("rep")["auprc"]
        common_reps = fli1_k0.index.intersection(fli1_k1.index)
        wins = (fli1_k1.loc[common_reps] > fli1_k0.loc[common_reps]).sum()
        print(f"\n  FLI1 k=1 > k=0 in {wins}/{len(common_reps)} replicates "
              f"({wins/len(common_reps):.0%})")

    # ===================================================================
    # SPI1 k-insensitivity check
    # ===================================================================
    print("\n" + "=" * 90)
    print("SPI1 k-INSENSITIVITY CHECK")
    print("=" * 90)

    spi1_data = df[df["tf"] == "SPI1"]
    if len(spi1_data) > 0:
        spi1_means = spi1_data.groupby("k")["auprc"].agg(["mean", "std"])
        spi1_range = spi1_means["mean"].max() - spi1_means["mean"].min()
        spi1_cv_across_k = spi1_means["mean"].std() / spi1_means["mean"].mean()
        print(f"\n  SPI1 AUPRC range across k: {spi1_range:.4f}")
        print(f"  SPI1 CV of mean AUPRC across k: {spi1_cv_across_k:.3f}")
        print(f"  SPI1 best k (bootstrap mean): {spi1_means['mean'].idxmax()}")
        print(f"  SPI1 mean AUPRC at best k: {spi1_means['mean'].max():.4f}")
        print(f"  SPI1 mean AUPRC at k=0: {spi1_means.loc[0, 'mean']:.4f}")
        rel_gain = (spi1_means["mean"].max() - spi1_means.loc[0, "mean"]) / spi1_means.loc[0, "mean"]
        print(f"  SPI1 relative gain from best k over k=0: {rel_gain:.1%}")

    # ===================================================================
    # Plots
    # ===================================================================
    print("\n[PLOT] Generating figures...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        # Figure 1: k_opt distributions for key TFs (violin/strip plot)
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for ti, tf in enumerate(KEY_TFS):
            ax = axes[ti]
            tf_kopt = kopt_df[kopt_df["tf"] == tf]["k_opt"].values
            if len(tf_kopt) == 0:
                continue

            # Histogram
            k_bins = sorted(set(K_GRID))
            counts = [np.sum(tf_kopt == k) for k in k_bins]
            ax.bar(range(len(k_bins)), counts, color="C0", alpha=0.7, edgecolor="black")
            ax.set_xticks(range(len(k_bins)))
            ax.set_xticklabels([str(k) for k in k_bins], fontsize=7, rotation=45)
            ax.set_xlabel("k_opt")
            ax.set_ylabel("count")
            n_tgt = int(kopt_df[kopt_df["tf"] == tf]["n_targets"].iloc[0])
            med = np.median(tf_kopt)
            iqr = np.percentile(tf_kopt, 75) - np.percentile(tf_kopt, 25)
            ax.set_title(f"{tf} (n={n_tgt}, med={med:.0f}, IQR={iqr:.0f})", fontsize=9)

        # Last panel: all key TFs boxplot
        ax = axes[7]
        data_for_box = []
        labels_for_box = []
        for tf in KEY_TFS:
            tf_kopt = kopt_df[kopt_df["tf"] == tf]["k_opt"].values
            if len(tf_kopt) > 0:
                data_for_box.append(tf_kopt)
                n_tgt = int(kopt_df[kopt_df["tf"] == tf]["n_targets"].iloc[0])
                labels_for_box.append(f"{tf}\n(n={n_tgt})")
        ax.boxplot(data_for_box, labels=labels_for_box)
        ax.set_ylabel("k_opt")
        ax.set_title("k_opt distributions (all key TFs)", fontsize=9)
        ax.tick_params(axis='x', labelsize=7)

        fig.suptitle(f"Bootstrap k_opt Stability ({n_boot} reps, {sub_frac:.0%} cells)",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(figdir / "bootstrap_kopt_distributions.png", dpi=150)
        plt.close(fig)
        print(f"[FIG] bootstrap_kopt_distributions.png")

        # Figure 2: Bootstrap AUPRC curves with error bands for key TFs
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for ti, tf in enumerate(KEY_TFS):
            ax = axes[ti]
            tf_stats = auprc_stats[auprc_stats["tf"] == tf].sort_values("k")
            if len(tf_stats) == 0:
                continue

            ks = tf_stats["k"].values
            means = tf_stats["mean"].values
            stds = tf_stats["std"].values

            ax.plot(ks, means, "o-", color="C0", markersize=4)
            ax.fill_between(ks, means - stds, means + stds, alpha=0.2, color="C0")
            ax.axhline(means[0], color="gray", ls="--", alpha=0.5, label="k=0 mean")

            n_tgt = int(df[df["tf"] == tf]["n_targets"].iloc[0])
            ax.set_title(f"{tf} (n={n_tgt})", fontsize=10)
            ax.set_xlabel("k")
            ax.set_ylabel("AUPRC")
            if ti == 0:
                ax.legend(fontsize=7)

        # Last panel: mean across key TFs
        ax = axes[7]
        key_tf_set = set(KEY_TFS)
        key_stats = auprc_stats[auprc_stats["tf"].isin(key_tf_set)]
        key_mean_by_k = key_stats.groupby("k")["mean"].mean()
        # Also get all-TF mean
        all_mean_by_k = auprc_stats.groupby("k")["mean"].mean()

        ax.plot(key_mean_by_k.index, key_mean_by_k.values, "o-", color="C0",
                markersize=4, label="key TF mean")
        ax.plot(all_mean_by_k.index, all_mean_by_k.values, "s--", color="C5",
                markersize=3, label="all TF mean")
        ax.set_title("Mean AUPRC (bootstrap)", fontsize=10)
        ax.set_xlabel("k")
        ax.set_ylabel("AUPRC")
        ax.legend(fontsize=7)

        fig.suptitle(f"Bootstrap AUPRC Curves ± 1 std ({n_boot} reps)", fontsize=12)
        fig.tight_layout()
        fig.savefig(figdir / "bootstrap_auprc_curves.png", dpi=150)
        plt.close(fig)
        print(f"[FIG] bootstrap_auprc_curves.png")

    except Exception as e:
        print(f"[PLOT ERROR] {e}")
        import traceback
        traceback.print_exc()

    wall = time.perf_counter() - t0
    print(f"\n[DONE] {wall:.1f}s total")


if __name__ == "__main__":
    main()
