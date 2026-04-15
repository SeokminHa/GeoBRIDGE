"""
Ground truth robustness and per-k discriminative signal analysis.

Two analyses in one script:

  (1) Ground truth robustness: Compare k_opt across CollecTRI vs DoRothEA
      to test whether k_opt heterogeneity is a property of the biology
      or the evaluation setup.

  (2) Per-k discriminative signal: Track target vs non-target score
      separation as k increases, testing the "confounding removal"
      hypothesis — that high-k TFs need dominant spectral components
      removed to expose discriminative signal.

Decision hinge:
  - If k_opt is NOT stable across ground truths -> stop mechanism analysis
  - If k_opt IS stable -> confounding removal mechanism is worth pursuing

Inputs:
  - Bone marrow data (same as bootstrap)
  - Bootstrap results: results/bm_bootstrap_k/csv/bootstrap_summary.csv

Outputs:
  results/bm_gt_robustness/csv/kopt_per_gt.csv
  results/bm_gt_robustness/csv/auprc_per_gt_k_tf.csv
  results/bm_gt_robustness/csv/discriminative_signal.csv
  results/bm_gt_robustness/figures/...
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from geobridge_model import GeoBridgeModel

KEY_TFS = ["ERG", "FLI1", "GATA2", "SPI1", "KLF1", "MYB", "MEIS1"]
K_GRID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]


# =====================================================================
# Data loading (shared with bootstrap / spectral alignment scripts)
# =====================================================================

def load_and_preprocess(data_dir):
    """Load bone marrow, preprocess, compute velocity."""
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


# =====================================================================
# Ground truth loading: multiple databases
# =====================================================================

def load_ground_truths(genes, min_targets=5):
    """
    Load multiple TF-target databases via decoupler.
    Returns dict: {db_name: (tf_targets, gene_to_idx)}
    """
    import decoupler as dc

    gene_set = set(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    def _build_tf_targets(net, name):
        tf_targets = {}
        for tf in net["source"].unique():
            if tf not in gene_set:
                continue
            targets = net[net["source"] == tf]["target"].values
            matched = [gene_to_idx[t] for t in targets if t in gene_set and t != tf]
            if len(matched) >= min_targets:
                tf_targets[tf] = matched
        print(f"  [{name}] {len(net)} edges -> {len(tf_targets)} evaluable TFs "
              f"(min {min_targets} targets)")
        return tf_targets

    databases = {}

    # CollecTRI
    try:
        if hasattr(dc, "op") and hasattr(dc.op, "collectri"):
            ct_net = dc.op.collectri(organism="human")
        elif hasattr(dc, "get_collectri"):
            ct_net = dc.get_collectri(organism="human", split_complexes=False)
        else:
            ct_net = None
        if ct_net is not None:
            databases["CollecTRI"] = _build_tf_targets(ct_net, "CollecTRI")
    except Exception as e:
        print(f"  [CollecTRI] FAILED: {e}")

    # DoRothEA (all confidence levels: A+B+C)
    try:
        if hasattr(dc, "op") and hasattr(dc.op, "dorothea"):
            do_net = dc.op.dorothea(organism="human")
        elif hasattr(dc, "get_dorothea"):
            do_net = dc.get_dorothea(organism="human")
        else:
            do_net = None
        if do_net is not None:
            databases["DoRothEA_ABC"] = _build_tf_targets(do_net, "DoRothEA_ABC")
            # High confidence only: A+B
            do_ab = do_net[do_net["confidence"].isin(["A", "B"])]
            databases["DoRothEA_AB"] = _build_tf_targets(do_ab, "DoRothEA_AB")
    except Exception as e:
        print(f"  [DoRothEA] FAILED: {e}")

    return databases, gene_to_idx


# =====================================================================
# AUPRC computation
# =====================================================================

def compute_per_tf_auprc(S_score, tf_idx, target_idxs, G):
    """Compute AUPRC for one TF."""
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


# =====================================================================
# Per-k discriminative signal analysis
# =====================================================================

def compute_discriminative_metrics(S_score, tf_idx, target_idxs, G):
    """
    Compute target vs non-target separation metrics for one TF at one k.

    Returns dict with:
      - mean_target_score, mean_nontarget_score, mean_diff
      - cohens_d: standardized effect size
      - ks_stat, ks_pval: Kolmogorov-Smirnov test
      - auroc: area under ROC (target vs non-target classification)
    """
    target_set = set(target_idxs)
    target_scores = []
    nontarget_scores = []
    for j in range(G):
        if j == tf_idx:
            continue
        s = abs(S_score[tf_idx, j])
        if j in target_set:
            target_scores.append(s)
        else:
            nontarget_scores.append(s)

    target_scores = np.array(target_scores)
    nontarget_scores = np.array(nontarget_scores)

    if len(target_scores) == 0 or len(nontarget_scores) == 0:
        return None

    mean_tgt = target_scores.mean()
    mean_nontgt = nontarget_scores.mean()
    mean_diff = mean_tgt - mean_nontgt

    # Cohen's d
    pooled_std = np.sqrt(
        (target_scores.var() * len(target_scores)
         + nontarget_scores.var() * len(nontarget_scores))
        / (len(target_scores) + len(nontarget_scores))
    )
    cohens_d = mean_diff / pooled_std if pooled_std > 1e-15 else 0.0

    # KS test
    ks_stat, ks_pval = sp_stats.ks_2samp(target_scores, nontarget_scores)

    # AUROC (Mann-Whitney U based)
    from sklearn.metrics import roc_auc_score
    labels = np.concatenate([np.ones(len(target_scores)),
                              np.zeros(len(nontarget_scores))])
    scores = np.concatenate([target_scores, nontarget_scores])
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = np.nan

    return {
        "mean_target_score": float(mean_tgt),
        "mean_nontarget_score": float(mean_nontgt),
        "mean_diff": float(mean_diff),
        "cohens_d": float(cohens_d),
        "ks_stat": float(ks_stat),
        "ks_pval": float(ks_pval),
        "auroc": float(auroc),
    }


# =====================================================================
# Main analysis
# =====================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Ground truth robustness + discriminative signal analysis")
    ap.add_argument("--data-dir", default="E:/data_cache")
    ap.add_argument("--bootstrap-dir", default="results/bm_bootstrap_k/csv")
    ap.add_argument("--outdir", default="results/bm_gt_robustness")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    csvdir = outdir / "csv"
    figdir = outdir / "figures"
    csvdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    X, V, pt, genes = load_and_preprocess(args.data_dir)
    G = len(genes)

    print("\n[GT] Loading ground truth databases...")
    databases, gene_to_idx = load_ground_truths(genes)

    # Load bootstrap summary for comparison
    bs_path = Path(args.bootstrap_dir) / "bootstrap_summary.csv"
    if bs_path.exists():
        bootstrap_summary = pd.read_csv(bs_path)
        print(f"[GT] Loaded bootstrap summary: {len(bootstrap_summary)} TFs")
    else:
        bootstrap_summary = None
        print(f"[GT] No bootstrap summary found at {bs_path}")

    # ------------------------------------------------------------------
    # Compute full-dataset streaming stats
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("COMPUTING FULL-DATASET PAIRSPACE STATS")
    print("=" * 90)

    sort_order = np.argsort(pt)
    X_sorted = X[sort_order]
    V_sorted = V[sort_order]
    pt_sorted = pt[sort_order]

    t_pre = time.perf_counter()
    pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
        X_sorted, V_sorted, pt_sorted, pair_seed=42,
    )
    print(f"[PRE] Streaming stats: {time.perf_counter()-t_pre:.1f}s, "
          f"{pre['n_pairs']} pairs")

    # ------------------------------------------------------------------
    # Compute score matrices at each k (once, reuse across databases)
    # ------------------------------------------------------------------
    print("\n[SCORES] Computing score matrices at each k...")
    score_matrices = {}
    for k in K_GRID:
        if k >= len(pre["S"]):
            continue
        S_k = GeoBridgeModel.scores_from_streaming_stats(
            pre, k=k, project_y=True, normalize_score=True, eps=1e-12,
        )
        score_matrices[k] = S_k
    print(f"[SCORES] {len(score_matrices)} score matrices computed")

    # ------------------------------------------------------------------
    # Analysis 1: AUPRC per TF per k per database
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("ANALYSIS 1: GROUND TRUTH ROBUSTNESS")
    print("=" * 90)

    auprc_rows = []
    kopt_rows = []

    for db_name, tf_targets in databases.items():
        print(f"\n  [{db_name}] Evaluating {len(tf_targets)} TFs...")

        for tf, target_idxs in tf_targets.items():
            tf_idx = gene_to_idx[tf]
            best_auprc = -1
            best_k = 0
            tf_auprcs = {}

            for k, S_k in score_matrices.items():
                auprc = compute_per_tf_auprc(S_k, tf_idx, target_idxs, G)
                auprc_rows.append({
                    "database": db_name,
                    "tf": tf,
                    "k": k,
                    "auprc": auprc,
                    "n_targets": len(target_idxs),
                    "is_key_tf": tf in KEY_TFS,
                })
                tf_auprcs[k] = auprc
                if not np.isnan(auprc) and auprc > best_auprc:
                    best_auprc = auprc
                    best_k = k

            kopt_rows.append({
                "database": db_name,
                "tf": tf,
                "k_opt": best_k,
                "best_auprc": best_auprc,
                "n_targets": len(target_idxs),
                "is_key_tf": tf in KEY_TFS,
            })

    auprc_df = pd.DataFrame(auprc_rows)
    kopt_df = pd.DataFrame(kopt_rows)
    auprc_df.to_csv(csvdir / "auprc_per_gt_k_tf.csv", index=False)
    kopt_df.to_csv(csvdir / "kopt_per_gt.csv", index=False)
    print(f"\n[SAVE] auprc_per_gt_k_tf.csv ({len(auprc_df)} rows)")
    print(f"[SAVE] kopt_per_gt.csv ({len(kopt_df)} rows)")

    # ------------------------------------------------------------------
    # Report: k_opt comparison across databases
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("k_opt COMPARISON ACROSS GROUND TRUTH DATABASES")
    print("=" * 90)

    db_names = sorted(databases.keys())

    # Key TFs table
    print(f"\n  Key TFs k_opt by database:")
    header = f"  {'TF':<8s}"
    for db in db_names:
        header += f"  {db:>16s}"
    if bootstrap_summary is not None:
        header += f"  {'Bootstrap_med':>14s}"
    print(header)
    print("  " + "-" * (8 + 18 * len(db_names) + 16))

    for tf in KEY_TFS:
        row_str = f"  {tf:<8s}"
        for db in db_names:
            sub = kopt_df[(kopt_df["database"] == db) & (kopt_df["tf"] == tf)]
            if len(sub) > 0:
                k = int(sub["k_opt"].values[0])
                n = int(sub["n_targets"].values[0])
                row_str += f"  {k:>3d} (n={n:>3d})"
            else:
                row_str += f"  {'N/A':>16s}"
        if bootstrap_summary is not None:
            bs_row = bootstrap_summary[bootstrap_summary["tf"] == tf]
            if len(bs_row) > 0:
                row_str += f"  {bs_row['k_opt_median'].values[0]:>14.0f}"
            else:
                row_str += f"  {'N/A':>14s}"
        print(row_str)

    # All TFs: cross-database k_opt agreement
    print(f"\n  Cross-database k_opt agreement (TFs present in all databases):")
    for i, db1 in enumerate(db_names):
        for db2 in db_names[i+1:]:
            k1_df = kopt_df[kopt_df["database"] == db1][["tf", "k_opt"]].rename(
                columns={"k_opt": f"k_{db1}"})
            k2_df = kopt_df[kopt_df["database"] == db2][["tf", "k_opt"]].rename(
                columns={"k_opt": f"k_{db2}"})
            merged = k1_df.merge(k2_df, on="tf")
            if len(merged) > 0:
                rho, pval = sp_stats.spearmanr(
                    merged[f"k_{db1}"], merged[f"k_{db2}"])
                exact_match = (merged[f"k_{db1}"] == merged[f"k_{db2}"]).mean()
                within_5 = (abs(merged[f"k_{db1}"] - merged[f"k_{db2}"]) <= 5).mean()
                print(f"    {db1} vs {db2}: "
                      f"n={len(merged)} shared TFs, "
                      f"Spearman rho={rho:+.3f} (p={pval:.4f}), "
                      f"exact match={exact_match:.1%}, "
                      f"within-5={within_5:.1%}")

    # Compare with bootstrap k_opt
    if bootstrap_summary is not None:
        print(f"\n  Database k_opt vs bootstrap k_opt_median (CollecTRI):")
        for db in db_names:
            db_kopt = kopt_df[kopt_df["database"] == db][["tf", "k_opt"]].rename(
                columns={"k_opt": "k_db"})
            bs_kopt = bootstrap_summary[["tf", "k_opt_median"]].rename(
                columns={"k_opt_median": "k_bs"})
            merged = db_kopt.merge(bs_kopt, on="tf")
            if len(merged) > 0:
                rho, pval = sp_stats.spearmanr(merged["k_db"], merged["k_bs"])
                exact = (merged["k_db"] == merged["k_bs"]).mean()
                within_5 = (abs(merged["k_db"] - merged["k_bs"]) <= 5).mean()
                print(f"    {db}: n={len(merged)}, rho={rho:+.3f} (p={pval:.4f}), "
                      f"exact={exact:.1%}, within-5={within_5:.1%}")

    # ------------------------------------------------------------------
    # k_opt distribution per database
    # ------------------------------------------------------------------
    print(f"\n  k_opt distribution by database:")
    for db in db_names:
        sub = kopt_df[kopt_df["database"] == db]
        kvals = sub["k_opt"].values
        print(f"    {db}: median={np.median(kvals):.0f}, "
              f"mean={np.mean(kvals):.1f}, "
              f"std={np.std(kvals):.1f}, "
              f"IQR=[{np.percentile(kvals,25):.0f}, {np.percentile(kvals,75):.0f}]")

    # ------------------------------------------------------------------
    # Analysis 2: Per-k discriminative signal (CollecTRI only, key TFs)
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("ANALYSIS 2: PER-k DISCRIMINATIVE SIGNAL (CollecTRI)")
    print("=" * 90)

    if "CollecTRI" in databases:
        tf_targets_ct = databases["CollecTRI"]
        disc_rows = []

        for tf in list(KEY_TFS) + [t for t in tf_targets_ct if t not in KEY_TFS]:
            if tf not in tf_targets_ct:
                continue
            target_idxs = tf_targets_ct[tf]
            tf_idx = gene_to_idx[tf]

            for k, S_k in score_matrices.items():
                metrics = compute_discriminative_metrics(
                    S_k, tf_idx, target_idxs, G)
                if metrics is None:
                    continue
                metrics["tf"] = tf
                metrics["k"] = k
                metrics["n_targets"] = len(target_idxs)
                metrics["is_key_tf"] = tf in KEY_TFS
                disc_rows.append(metrics)

        disc_df = pd.DataFrame(disc_rows)
        disc_df.to_csv(csvdir / "discriminative_signal.csv", index=False)
        print(f"[SAVE] discriminative_signal.csv ({len(disc_df)} rows)")

        # Report: per-k metrics for key TFs
        print(f"\n  Per-k discriminative signal for key TFs:")
        for tf in KEY_TFS:
            tf_disc = disc_df[disc_df["tf"] == tf].sort_values("k")
            if len(tf_disc) == 0:
                continue
            n_tgt = tf_disc["n_targets"].iloc[0]
            # Find k_opt from CollecTRI
            ct_kopt = kopt_df[(kopt_df["database"] == "CollecTRI") &
                              (kopt_df["tf"] == tf)]
            kopt_val = int(ct_kopt["k_opt"].values[0]) if len(ct_kopt) > 0 else "?"

            print(f"\n  {tf} (n_targets={n_tgt}, k_opt={kopt_val}):")
            print(f"    {'k':>3s}  {'Cohen_d':>8s}  {'AUROC':>7s}  "
                  f"{'KS_stat':>8s}  {'mean_tgt':>10s}  {'mean_nontgt':>10s}  "
                  f"{'diff':>10s}")
            print(f"    " + "-" * 70)
            for _, r in tf_disc.iterrows():
                marker = " <-- k_opt" if r["k"] == kopt_val else ""
                print(f"    {r['k']:3.0f}  {r['cohens_d']:8.3f}  "
                      f"{r['auroc']:7.4f}  {r['ks_stat']:8.4f}  "
                      f"{r['mean_target_score']:10.4f}  "
                      f"{r['mean_nontarget_score']:10.4f}  "
                      f"{r['mean_diff']:10.4f}{marker}")

        # Summary: how does discriminative signal change with k?
        print(f"\n  Discriminative signal trajectory patterns:")
        for tf in KEY_TFS:
            tf_disc = disc_df[disc_df["tf"] == tf].sort_values("k")
            if len(tf_disc) == 0:
                continue
            # Find k where AUROC peaks
            auroc_peak_k = int(tf_disc.loc[tf_disc["auroc"].idxmax(), "k"])
            # Find k where Cohen's d peaks
            d_peak_k = int(tf_disc.loc[tf_disc["cohens_d"].idxmax(), "k"])
            # AUROC at k=0 vs peak
            auroc_k0 = tf_disc[tf_disc["k"] == 0]["auroc"].values[0]
            auroc_peak = tf_disc["auroc"].max()
            d_k0 = tf_disc[tf_disc["k"] == 0]["cohens_d"].values[0]
            d_peak = tf_disc["cohens_d"].max()

            ct_kopt = kopt_df[(kopt_df["database"] == "CollecTRI") &
                              (kopt_df["tf"] == tf)]
            kopt_val = int(ct_kopt["k_opt"].values[0]) if len(ct_kopt) > 0 else -1

            print(f"    {tf:<8s}: k_opt(AUPRC)={kopt_val:2d}, "
                  f"k_peak(AUROC)={auroc_peak_k:2d}, "
                  f"k_peak(d)={d_peak_k:2d}, "
                  f"AUROC: {auroc_k0:.3f}->{auroc_peak:.3f}, "
                  f"d: {d_k0:.3f}->{d_peak:.3f}")
    else:
        disc_df = pd.DataFrame()
        print("  [SKIP] CollecTRI not loaded")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n[PLOT] Generating figures...")
    try:
        _make_plots(kopt_df, auprc_df, disc_df, databases, bootstrap_summary,
                    figdir)
    except Exception as e:
        print(f"[PLOT ERROR] {e}")
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------------------
    # Summary verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SUMMARY VERDICT")
    print("=" * 90)

    # Cross-database agreement for key TFs
    if len(db_names) >= 2:
        db1, db2 = db_names[0], db_names[1]
        k1 = kopt_df[kopt_df["database"] == db1][["tf", "k_opt"]].rename(
            columns={"k_opt": "k1"})
        k2 = kopt_df[kopt_df["database"] == db2][["tf", "k_opt"]].rename(
            columns={"k_opt": "k2"})
        merged = k1.merge(k2, on="tf")
        if len(merged) > 0:
            rho_all, p_all = sp_stats.spearmanr(merged["k1"], merged["k2"])
            print(f"\n  Cross-database k_opt agreement ({db1} vs {db2}):")
            print(f"    All shared TFs (n={len(merged)}): "
                  f"rho={rho_all:+.3f}, p={p_all:.4f}")

            # Key TFs only
            key_merged = merged[merged["tf"].isin(KEY_TFS)]
            if len(key_merged) >= 3:
                rho_key, p_key = sp_stats.spearmanr(
                    key_merged["k1"], key_merged["k2"])
                print(f"    Key TFs only (n={len(key_merged)}): "
                      f"rho={rho_key:+.3f}, p={p_key:.4f}")

    print(f"\n  Interpretation:")
    if len(db_names) >= 2:
        if rho_all > 0.6 and p_all < 0.05:
            print(f"    k_opt shows SIGNIFICANT cross-database agreement "
                  f"(rho={rho_all:+.3f})")
            print(f"    -> k_opt heterogeneity is likely a real property "
                  f"of the data, not just CollecTRI")
            print(f"    -> confounding removal mechanism is worth pursuing")
        elif rho_all > 0.3:
            print(f"    k_opt shows MODERATE cross-database agreement "
                  f"(rho={rho_all:+.3f})")
            print(f"    -> partial overlap; some heterogeneity is data-driven, "
                  f"some is database-specific")
        else:
            print(f"    k_opt shows WEAK cross-database agreement "
                  f"(rho={rho_all:+.3f})")
            print(f"    -> k_opt heterogeneity is substantially "
                  f"database-dependent")
            print(f"    -> stop mechanism hunting; use moderate global k")

    wall = time.perf_counter() - t0
    print(f"\n[DONE] {wall:.1f}s total")


# =====================================================================
# Plots
# =====================================================================

def _make_plots(kopt_df, auprc_df, disc_df, databases, bootstrap_summary,
                figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    db_names = sorted(databases.keys())

    # ----- Figure 1: k_opt scatter across databases -----
    if len(db_names) >= 2:
        n_pairs = len(db_names) * (len(db_names) - 1) // 2
        fig, axes = plt.subplots(1, max(n_pairs, 1),
                                 figsize=(6 * max(n_pairs, 1), 5))
        if n_pairs == 1:
            axes = [axes]

        idx = 0
        for i, db1 in enumerate(db_names):
            for db2 in db_names[i+1:]:
                ax = axes[idx]
                k1_df = kopt_df[kopt_df["database"] == db1][["tf", "k_opt", "is_key_tf"]].rename(
                    columns={"k_opt": "k1"})
                k2_df = kopt_df[kopt_df["database"] == db2][["tf", "k_opt"]].rename(
                    columns={"k_opt": "k2"})
                merged = k1_df.merge(k2_df, on="tf")

                # Regular TFs
                regular = merged[~merged["is_key_tf"]]
                ax.scatter(regular["k1"], regular["k2"],
                           c="C0", alpha=0.3, s=20, label="other TFs")
                # Key TFs
                key = merged[merged["is_key_tf"]]
                ax.scatter(key["k1"], key["k2"],
                           c="C3", s=50, zorder=5, label="key TFs")
                for _, r in key.iterrows():
                    ax.annotate(r["tf"], (r["k1"], r["k2"]),
                                fontsize=8, ha="left", va="bottom")

                # Diagonal
                lims = [-1, 26]
                ax.plot(lims, lims, "k--", alpha=0.3)
                rho, pval = sp_stats.spearmanr(merged["k1"], merged["k2"])
                ax.set_xlabel(f"k_opt ({db1})")
                ax.set_ylabel(f"k_opt ({db2})")
                ax.set_title(f"rho={rho:+.3f}, p={pval:.4f}, n={len(merged)}")
                ax.legend(fontsize=8)
                ax.set_xlim(lims)
                ax.set_ylim(lims)
                # Add jitter for overlapping points
                idx += 1

        fig.suptitle("k_opt Agreement Across Ground Truth Databases", fontsize=12)
        fig.tight_layout()
        fig.savefig(figdir / "kopt_cross_database.png", dpi=150)
        plt.close(fig)
        print(f"[FIG] kopt_cross_database.png")

    # ----- Figure 2: AUPRC curves by database for key TFs -----
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    colors = {"CollecTRI": "C0", "DoRothEA_ABC": "C1", "DoRothEA_AB": "C2"}

    for ti, tf in enumerate(KEY_TFS):
        ax = axes[ti]
        for db_name in db_names:
            tf_data = auprc_df[(auprc_df["database"] == db_name) &
                               (auprc_df["tf"] == tf)].sort_values("k")
            if len(tf_data) == 0:
                continue
            n_tgt = int(tf_data["n_targets"].iloc[0])
            ax.plot(tf_data["k"], tf_data["auprc"],
                    "o-", color=colors.get(db_name, "C5"),
                    markersize=3, label=f"{db_name} (n={n_tgt})", lw=1.2)

        ax.set_xlabel("k")
        ax.set_ylabel("AUPRC")
        ax.set_title(tf, fontsize=10)
        if ti == 0:
            ax.legend(fontsize=6)

    # Summary panel
    ax = axes[7]
    for db_name in db_names:
        sub = kopt_df[kopt_df["database"] == db_name]
        kvals = sub["k_opt"].values
        ax.hist(kvals, bins=np.arange(-0.5, 26.5, 1),
                alpha=0.4, label=db_name, color=colors.get(db_name, "C5"))
    ax.set_xlabel("k_opt")
    ax.set_ylabel("count (TFs)")
    ax.set_title("k_opt distribution", fontsize=10)
    ax.legend(fontsize=7)

    fig.suptitle("AUPRC Curves by Ground Truth Database", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "auprc_curves_by_database.png", dpi=150)
    plt.close(fig)
    print(f"[FIG] auprc_curves_by_database.png")

    # ----- Figure 3: Discriminative signal trajectories for key TFs -----
    if len(disc_df) > 0:
        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for ti, tf in enumerate(KEY_TFS):
            ax = axes[ti]
            tf_disc = disc_df[disc_df["tf"] == tf].sort_values("k")
            if len(tf_disc) == 0:
                continue

            ax2 = ax.twinx()
            ax.plot(tf_disc["k"], tf_disc["cohens_d"],
                    "o-", color="C0", markersize=4, label="Cohen's d")
            ax2.plot(tf_disc["k"], tf_disc["auroc"],
                     "s--", color="C3", markersize=3, label="AUROC")

            # Mark k_opt
            ct_kopt = kopt_df[(kopt_df["database"] == "CollecTRI") &
                              (kopt_df["tf"] == tf)]
            if len(ct_kopt) > 0:
                kopt_val = int(ct_kopt["k_opt"].values[0])
                ax.axvline(kopt_val, color="gray", ls=":", alpha=0.5)

            n_tgt = int(tf_disc["n_targets"].iloc[0])
            ax.set_title(f"{tf} (n={n_tgt}, k_opt={kopt_val})", fontsize=9)
            ax.set_xlabel("k")
            ax.set_ylabel("Cohen's d", color="C0")
            ax2.set_ylabel("AUROC", color="C3")
            if ti == 0:
                ax.legend(fontsize=7, loc="upper left")
                ax2.legend(fontsize=7, loc="upper right")

        # Summary: all TFs Cohen's d at k_opt vs k=0
        ax = axes[7]
        if "CollecTRI" in databases:
            d_at_0 = []
            d_at_kopt = []
            for tf in disc_df["tf"].unique():
                tf_d = disc_df[disc_df["tf"] == tf]
                d0 = tf_d[tf_d["k"] == 0]["cohens_d"].values
                ct_k = kopt_df[(kopt_df["database"] == "CollecTRI") &
                               (kopt_df["tf"] == tf)]
                if len(ct_k) > 0 and len(d0) > 0:
                    kopt_v = int(ct_k["k_opt"].values[0])
                    dk = tf_d[tf_d["k"] == kopt_v]["cohens_d"].values
                    if len(dk) > 0:
                        d_at_0.append(d0[0])
                        d_at_kopt.append(dk[0])
            if d_at_0:
                ax.scatter(d_at_0, d_at_kopt, alpha=0.5, s=20)
                lims = [min(min(d_at_0), min(d_at_kopt)) - 0.05,
                        max(max(d_at_0), max(d_at_kopt)) + 0.05]
                ax.plot(lims, lims, "k--", alpha=0.3)
                ax.set_xlabel("Cohen's d at k=0")
                ax.set_ylabel("Cohen's d at k_opt")
                ax.set_title("Effect size: k=0 vs k_opt", fontsize=9)

        fig.suptitle("Discriminative Signal vs Projection Depth", fontsize=12)
        fig.tight_layout()
        fig.savefig(figdir / "discriminative_signal_trajectories.png", dpi=150)
        plt.close(fig)
        print(f"[FIG] discriminative_signal_trajectories.png")


if __name__ == "__main__":
    main()
