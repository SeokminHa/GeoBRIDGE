"""
Spectral alignment analysis (5B): does TF-specific spectral structure
explain k_opt heterogeneity beyond simpler baseline predictors?

Staged design:
  Stage A  — baseline predictors (target count, mean target variance,
             mean target velocity magnitude). Test whether these alone
             explain bootstrap k_opt.
  Stage B  — spectral alignment descriptors from pair-space eigen-
             decomposition. Test incremental explanatory power.
  (Stage C — target-label permutation, run only with --run-permutation)

Null to challenge:
  "k_opt heterogeneity is fully explained by simpler factors like target
   count / variance, without needing TF-specific spectral structure."

Inputs:
  - Bone marrow data (scVelo, same as bootstrap)
  - Bootstrap results: results/bm_bootstrap_k/csv/csv/bootstrap_summary.csv

Outputs:
  results/bm_spectral_alignment/csv/stage_a_predictors.csv
  results/bm_spectral_alignment/csv/stage_b_alignment.csv
  results/bm_spectral_alignment/csv/stage_b_alignment_profiles.csv
  results/bm_spectral_alignment/figures/...
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


# =====================================================================
# Data loading (reused from bm_bootstrap_kopt.py)
# =====================================================================

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
    """Load CollecTRI edges, return tf_targets dict and gene_to_idx."""
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


# =====================================================================
# Stage A: baseline predictors
# =====================================================================

def compute_stage_a(X, V, genes, tf_targets, gene_to_idx, bootstrap_summary):
    """
    For each evaluable TF, compute simple baseline predictors and merge
    with bootstrap k_opt results.
    """
    G = X.shape[1]

    # Per-gene statistics (cell-level)
    gene_var = np.var(X, axis=0)             # (G,)  expression variance
    gene_vel_mag = np.sqrt(np.mean(V**2, axis=0))  # (G,)  RMS velocity magnitude

    # Rank genes by variance (descending: rank 0 = highest variance)
    gene_var_rank = np.argsort(np.argsort(-gene_var)).astype(float)  # (G,)

    rows = []
    for tf, target_idxs in tf_targets.items():
        tf_idx = gene_to_idx[tf]
        tidx = np.array(target_idxs)

        rows.append({
            "tf": tf,
            "tf_idx": tf_idx,
            "n_targets": len(tidx),
            "is_key_tf": tf in KEY_TFS,
            # Target-set statistics
            "mean_target_var": float(gene_var[tidx].mean()),
            "median_target_var": float(np.median(gene_var[tidx])),
            "mean_target_var_rank": float(gene_var_rank[tidx].mean()),
            "mean_target_vel_mag": float(gene_vel_mag[tidx].mean()),
            "median_target_vel_mag": float(np.median(gene_vel_mag[tidx])),
            # TF's own statistics (for reference)
            "tf_var": float(gene_var[tf_idx]),
            "tf_vel_mag": float(gene_vel_mag[tf_idx]),
        })

    df_a = pd.DataFrame(rows)

    # Merge with bootstrap k_opt results
    bs = bootstrap_summary[["tf", "k_opt_median", "k_opt_mean", "k_opt_std",
                            "k_opt_iqr", "k_opt_range", "k_opt_iqr_lo",
                            "k_opt_iqr_hi", "best_auprc_mean"]].copy()
    df_a = df_a.merge(bs, on="tf", how="inner")

    # Derived stability measures
    df_a["k_opt_iqr_width"] = df_a["k_opt_iqr_hi"] - df_a["k_opt_iqr_lo"]
    df_a["k_opt_cv"] = df_a["k_opt_std"] / (df_a["k_opt_mean"] + 1e-6)
    # Bimodality indicator: fraction of bootstrap reps where k_opt <= 5
    # (will be loaded from kopt_per_tf if needed)

    return df_a


def report_stage_a(df_a):
    """Report Stage A results: correlations between baseline predictors and k_opt."""
    print("\n" + "=" * 90)
    print("STAGE A: BASELINE PREDICTORS vs k_opt")
    print("=" * 90)

    response_vars = [
        ("k_opt_median", "median k_opt"),
        ("k_opt_std", "k_opt std (stability)"),
        ("k_opt_iqr_width", "k_opt IQR width"),
    ]
    predictors = [
        ("n_targets", "target count"),
        ("mean_target_var", "mean target variance"),
        ("mean_target_var_rank", "mean target var rank"),
        ("mean_target_vel_mag", "mean target velocity mag"),
    ]

    print(f"\n  Spearman correlations (n = {len(df_a)} TFs):")
    print(f"  {'predictor':<28s}", end="")
    for _, rname in response_vars:
        print(f"  {rname:>22s}", end="")
    print()
    print("  " + "-" * 96)

    for pcol, pname in predictors:
        print(f"  {pname:<28s}", end="")
        for rcol, _ in response_vars:
            rho, pval = sp_stats.spearmanr(df_a[pcol], df_a[rcol])
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {rho:+.3f} (p={pval:.3f}){sig:>3s}", end="")
        print()

    # Simple OLS: k_opt_median ~ n_targets + mean_target_var
    print(f"\n  --- OLS: k_opt_median ~ n_targets + mean_target_var ---")
    _report_ols(df_a, "k_opt_median", ["n_targets", "mean_target_var"])

    # Also report with velocity
    print(f"\n  --- OLS: k_opt_median ~ n_targets + mean_target_var + mean_target_vel_mag ---")
    _report_ols(df_a, "k_opt_median",
                ["n_targets", "mean_target_var", "mean_target_vel_mag"])

    # Key TF table
    print(f"\n  Key TFs baseline predictors:")
    key = df_a[df_a["is_key_tf"]].sort_values("k_opt_median")
    print(f"  {'TF':<8s}  {'n_tgt':>5s}  {'k_med':>5s}  {'k_std':>5s}  "
          f"{'tgt_var':>10s}  {'var_rank':>8s}  {'vel_mag':>10s}")
    print("  " + "-" * 65)
    for _, r in key.iterrows():
        print(f"  {r['tf']:<8s}  {r['n_targets']:5d}  {r['k_opt_median']:5.0f}  "
              f"{r['k_opt_std']:5.1f}  {r['mean_target_var']:10.4f}  "
              f"{r['mean_target_var_rank']:8.1f}  {r['mean_target_vel_mag']:10.4f}")

    return


def _report_ols(df, y_col, x_cols):
    """Simple OLS with R^2 and coefficient report. No heavy dependencies."""
    y = df[y_col].values.astype(float)
    X = np.column_stack([df[c].values.astype(float) for c in x_cols])
    # Standardize for coefficient comparison
    X_std = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-12)
    # Add intercept
    X_aug = np.column_stack([np.ones(len(y)), X_std])
    # OLS fit
    beta, residuals, rank, sv = np.linalg.lstsq(X_aug, y, rcond=None)
    y_hat = X_aug @ beta
    ss_res = np.sum((y - y_hat)**2)
    ss_tot = np.sum((y - y.mean())**2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    n = len(y)
    p = len(x_cols)
    r2_adj = 1 - (1 - r2) * (n - 1) / (n - p - 1) if n > p + 1 else r2

    print(f"    R^2 = {r2:.3f},  R^2_adj = {r2_adj:.3f},  n = {n}")
    print(f"    intercept = {beta[0]:.2f}")
    for i, c in enumerate(x_cols):
        print(f"    beta_{c} (standardized) = {beta[i+1]:+.2f}")

    # Residual Spearman with y (to check if residuals still correlate with y)
    resid = y - y_hat
    rho_resid, p_resid = sp_stats.spearmanr(resid, y)
    print(f"    residual ~ y Spearman: rho={rho_resid:+.3f}, p={p_resid:.3f}")


# =====================================================================
# Stage B: spectral alignment
# =====================================================================

def compute_stage_b(pre, genes, tf_targets, gene_to_idx, df_a):
    """
    Compute spectral alignment profiles for each TF's target set.

    For each target gene j, the cross-covariance signal vector is M[:, j].
    Its projection onto eigenvector l of C is alpha_l(j) = v_l^T M[:, j].

    Per-TF profile: average |alpha_l|^2 over targets.
    """
    Vt = pre["Vt"]   # (G, G) — rows are eigenvectors
    M = pre["M"]     # (G, G) — cross-covariance ΔXc^T ΔV
    C = pre["C"]     # (G, G) — ΔXc^T ΔXc
    S = pre["S"]     # (G,)   — singular values (sqrt of eigenvalues)
    G = pre["G"]

    # Spectral projections of cross-covariance: (G_components, G_genes)
    # alpha[l, j] = v_l^T M[:, j] = amount of gene j's cross-cov signal in component l
    alpha = Vt @ M   # (G, G)

    # Also compute pair-space variance per gene (diagonal of C)
    pairspace_var = np.diag(C)  # (G,)
    pairspace_var_rank = np.argsort(np.argsort(-pairspace_var)).astype(float)

    # Per-TF alignment descriptors
    K_MAX_PROFILE = min(50, G)  # only store first 50 components in profile CSV
    rows = []
    profile_rows = []

    for tf, target_idxs in tf_targets.items():
        tf_idx = gene_to_idx[tf]
        tidx = np.array(target_idxs)

        # Per-target alpha^2, averaged over targets: shape (G,)
        alpha_sq_targets = np.mean(alpha[:, tidx]**2, axis=1)  # (G,)
        total_signal = alpha_sq_targets.sum()

        if total_signal < 1e-20:
            continue

        # Normalized profile (fraction of signal per component)
        profile = alpha_sq_targets / total_signal  # (G,)

        # Descriptors
        component_idx = np.arange(G, dtype=float)

        spectral_centroid = float(np.sum(component_idx * profile))
        frac_top5 = float(profile[:5].sum())
        frac_top10 = float(profile[:10].sum())

        # Concentration: 1 - normalized entropy
        # Clip for log stability
        p_nonzero = profile[profile > 1e-20]
        entropy = -np.sum(p_nonzero * np.log(p_nonzero))
        max_entropy = np.log(G)
        concentration = 1.0 - entropy / max_entropy if max_entropy > 0 else 0.0

        # Peak component
        peak_component = int(np.argmax(profile))

        # Mean pair-space variance of targets (for tautology check)
        mean_target_pairvar = float(pairspace_var[tidx].mean())
        mean_target_pairvar_rank = float(pairspace_var_rank[tidx].mean())

        rows.append({
            "tf": tf,
            "spectral_centroid": spectral_centroid,
            "frac_top5": frac_top5,
            "frac_top10": frac_top10,
            "concentration": concentration,
            "peak_component": peak_component,
            "total_signal": total_signal,
            "mean_target_pairvar": mean_target_pairvar,
            "mean_target_pairvar_rank": mean_target_pairvar_rank,
        })

        # Store profile (first K_MAX_PROFILE components) for visualization
        for l in range(K_MAX_PROFILE):
            profile_rows.append({
                "tf": tf,
                "component": l,
                "frac_signal": float(profile[l]),
                "cumul_signal": float(profile[:l+1].sum()),
                "alpha_sq_mean": float(alpha_sq_targets[l]),
            })

    df_b = pd.DataFrame(rows)
    df_profiles = pd.DataFrame(profile_rows)

    # Merge with Stage A
    df_b = df_b.merge(df_a, on="tf", how="inner")

    return df_b, df_profiles


def report_stage_b(df_b):
    """Report Stage B: spectral alignment vs k_opt, tautology check."""
    print("\n" + "=" * 90)
    print("STAGE B: SPECTRAL ALIGNMENT vs k_opt")
    print("=" * 90)

    # --- Tautology check ---
    print(f"\n  --- Tautology check ---")
    print(f"  Does spectral_centroid just track mean target variance rank?")

    rho_taut, p_taut = sp_stats.spearmanr(
        df_b["spectral_centroid"], df_b["mean_target_var_rank"])
    print(f"  spectral_centroid vs mean_target_var_rank:  "
          f"rho = {rho_taut:+.3f}, p = {p_taut:.4f}")

    rho_taut2, p_taut2 = sp_stats.spearmanr(
        df_b["spectral_centroid"], df_b["mean_target_pairvar_rank"])
    print(f"  spectral_centroid vs mean_target_pairvar_rank: "
          f"rho = {rho_taut2:+.3f}, p = {p_taut2:.4f}")

    if abs(rho_taut) > 0.9 or abs(rho_taut2) > 0.9:
        print(f"  >>> WARNING: high correlation (rho > 0.9) — tautology risk is HIGH")
    elif abs(rho_taut) > 0.7 or abs(rho_taut2) > 0.7:
        print(f"  >>> CAUTION: moderate correlation — spectral centroid partially "
              f"reflects variance ordering")
    else:
        print(f"  >>> Spectral centroid is NOT dominated by variance ordering")

    # --- Spectral descriptors vs k_opt ---
    print(f"\n  --- Spectral descriptors vs k_opt (Spearman) ---")
    response_vars = [
        ("k_opt_median", "median k_opt"),
        ("k_opt_std", "k_opt std"),
    ]
    spectral_preds = [
        ("spectral_centroid", "spectral centroid"),
        ("frac_top5", "frac signal in top-5"),
        ("frac_top10", "frac signal in top-10"),
        ("concentration", "spectral concentration"),
        ("peak_component", "peak component"),
        ("mean_target_pairvar", "mean target pairspace var"),
        ("mean_target_pairvar_rank", "mean target pairvar rank"),
    ]

    print(f"\n  {'descriptor':<28s}", end="")
    for _, rname in response_vars:
        print(f"  {rname:>22s}", end="")
    print()
    print("  " + "-" * 76)

    for pcol, pname in spectral_preds:
        print(f"  {pname:<28s}", end="")
        for rcol, _ in response_vars:
            rho, pval = sp_stats.spearmanr(df_b[pcol], df_b[rcol])
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {rho:+.3f} (p={pval:.3f}){sig:>3s}", end="")
        print()

    # --- Incremental R^2: baseline-only vs baseline + spectral ---
    print(f"\n  --- Incremental R^2 test ---")
    print(f"  Model 1: k_opt_median ~ n_targets + mean_target_var")
    _report_ols(df_b, "k_opt_median", ["n_targets", "mean_target_var"])

    print(f"\n  Model 2: k_opt_median ~ n_targets + mean_target_var + spectral_centroid")
    _report_ols(df_b, "k_opt_median",
                ["n_targets", "mean_target_var", "spectral_centroid"])

    print(f"\n  Model 3: k_opt_median ~ n_targets + mean_target_var + spectral_centroid "
          f"+ frac_top5")
    _report_ols(df_b, "k_opt_median",
                ["n_targets", "mean_target_var", "spectral_centroid", "frac_top5"])

    # --- Check: does spectral centroid alone beat baseline? ---
    print(f"\n  Model 4 (spectral only): k_opt_median ~ spectral_centroid")
    _report_ols(df_b, "k_opt_median", ["spectral_centroid"])

    print(f"\n  Model 5 (spectral only): k_opt_median ~ spectral_centroid + frac_top5 "
          f"+ concentration")
    _report_ols(df_b, "k_opt_median",
                ["spectral_centroid", "frac_top5", "concentration"])

    # --- Key TF spectral profiles ---
    print(f"\n  Key TFs spectral alignment:")
    key = df_b[df_b["is_key_tf"]].sort_values("k_opt_median")
    print(f"  {'TF':<8s}  {'k_med':>5s}  {'centroid':>8s}  {'frac5':>6s}  "
          f"{'frac10':>6s}  {'concen':>6s}  {'peak':>4s}  "
          f"{'pvar_rnk':>8s}  {'var_rnk':>8s}")
    print("  " + "-" * 75)
    for _, r in key.iterrows():
        print(f"  {r['tf']:<8s}  {r['k_opt_median']:5.0f}  "
              f"{r['spectral_centroid']:8.1f}  {r['frac_top5']:6.3f}  "
              f"{r['frac_top10']:6.3f}  {r['concentration']:6.3f}  "
              f"{r['peak_component']:4.0f}  "
              f"{r['mean_target_pairvar_rank']:8.1f}  "
              f"{r['mean_target_var_rank']:8.1f}")


# =====================================================================
# Plotting
# =====================================================================

def make_plots(df_b, df_profiles, figdir):
    """Generate diagnostic figures."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir.mkdir(parents=True, exist_ok=True)

    # ----- Figure 1: baseline predictors vs k_opt -----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    ax.scatter(df_b["n_targets"], df_b["k_opt_median"],
               c="C0", alpha=0.5, s=25)
    for _, r in df_b[df_b["is_key_tf"]].iterrows():
        ax.annotate(r["tf"], (r["n_targets"], r["k_opt_median"]),
                    fontsize=7, ha="left")
    ax.set_xlabel("n_targets")
    ax.set_ylabel("median k_opt")
    rho, p = sp_stats.spearmanr(df_b["n_targets"], df_b["k_opt_median"])
    ax.set_title(f"Target count (rho={rho:+.2f}, p={p:.3f})")

    ax = axes[1]
    ax.scatter(df_b["mean_target_var"], df_b["k_opt_median"],
               c="C1", alpha=0.5, s=25)
    for _, r in df_b[df_b["is_key_tf"]].iterrows():
        ax.annotate(r["tf"], (r["mean_target_var"], r["k_opt_median"]),
                    fontsize=7, ha="left")
    ax.set_xlabel("mean target variance")
    ax.set_ylabel("median k_opt")
    rho, p = sp_stats.spearmanr(df_b["mean_target_var"], df_b["k_opt_median"])
    ax.set_title(f"Target variance (rho={rho:+.2f}, p={p:.3f})")

    ax = axes[2]
    ax.scatter(df_b["mean_target_vel_mag"], df_b["k_opt_median"],
               c="C2", alpha=0.5, s=25)
    for _, r in df_b[df_b["is_key_tf"]].iterrows():
        ax.annotate(r["tf"], (r["mean_target_vel_mag"], r["k_opt_median"]),
                    fontsize=7, ha="left")
    ax.set_xlabel("mean target velocity magnitude")
    ax.set_ylabel("median k_opt")
    rho, p = sp_stats.spearmanr(df_b["mean_target_vel_mag"], df_b["k_opt_median"])
    ax.set_title(f"Target velocity (rho={rho:+.2f}, p={p:.3f})")

    fig.suptitle("Stage A: Baseline Predictors vs k_opt", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "stage_a_baseline_vs_kopt.png", dpi=150)
    plt.close(fig)
    print(f"[FIG] stage_a_baseline_vs_kopt.png")

    # ----- Figure 2: spectral centroid vs k_opt + tautology check -----
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))

    ax = axes[0]
    ax.scatter(df_b["spectral_centroid"], df_b["k_opt_median"],
               c="C3", alpha=0.5, s=25)
    for _, r in df_b[df_b["is_key_tf"]].iterrows():
        ax.annotate(r["tf"], (r["spectral_centroid"], r["k_opt_median"]),
                    fontsize=7, ha="left")
    rho, p = sp_stats.spearmanr(df_b["spectral_centroid"], df_b["k_opt_median"])
    ax.set_xlabel("spectral centroid")
    ax.set_ylabel("median k_opt")
    ax.set_title(f"Centroid vs k_opt (rho={rho:+.2f}, p={p:.3f})")

    ax = axes[1]
    ax.scatter(df_b["frac_top5"], df_b["k_opt_median"],
               c="C4", alpha=0.5, s=25)
    for _, r in df_b[df_b["is_key_tf"]].iterrows():
        ax.annotate(r["tf"], (r["frac_top5"], r["k_opt_median"]),
                    fontsize=7, ha="left")
    rho, p = sp_stats.spearmanr(df_b["frac_top5"], df_b["k_opt_median"])
    ax.set_xlabel("fraction of signal in top-5 SVs")
    ax.set_ylabel("median k_opt")
    ax.set_title(f"frac_top5 vs k_opt (rho={rho:+.2f}, p={p:.3f})")

    # Tautology: spectral centroid vs mean target var rank
    ax = axes[2]
    ax.scatter(df_b["spectral_centroid"], df_b["mean_target_var_rank"],
               c="C5", alpha=0.5, s=25)
    for _, r in df_b[df_b["is_key_tf"]].iterrows():
        ax.annotate(r["tf"], (r["spectral_centroid"], r["mean_target_var_rank"]),
                    fontsize=7, ha="left")
    rho, p = sp_stats.spearmanr(df_b["spectral_centroid"], df_b["mean_target_var_rank"])
    ax.set_xlabel("spectral centroid")
    ax.set_ylabel("mean target var rank (cell-level)")
    ax.set_title(f"Tautology check (rho={rho:+.2f}, p={p:.3f})")

    fig.suptitle("Stage B: Spectral Alignment vs k_opt", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "stage_b_spectral_vs_kopt.png", dpi=150)
    plt.close(fig)
    print(f"[FIG] stage_b_spectral_vs_kopt.png")

    # ----- Figure 3: spectral alignment heatmap (key TFs) -----
    key_tfs_present = [tf for tf in KEY_TFS
                       if tf in df_b["tf"].values]
    # Sort by k_opt_median
    key_sorted = (df_b[df_b["tf"].isin(key_tfs_present)]
                  .sort_values("k_opt_median")["tf"].tolist())

    n_components_show = 30
    fig, ax = plt.subplots(figsize=(12, 4))
    heatmap_data = []
    for tf in key_sorted:
        prof = df_profiles[(df_profiles["tf"] == tf) &
                           (df_profiles["component"] < n_components_show)]
        heatmap_data.append(prof["frac_signal"].values)

    if heatmap_data:
        hm = np.array(heatmap_data)
        im = ax.imshow(hm, aspect="auto", cmap="viridis",
                       interpolation="nearest")
        ax.set_yticks(range(len(key_sorted)))
        k_meds = [df_b[df_b["tf"] == tf]["k_opt_median"].values[0]
                   for tf in key_sorted]
        ylabels = [f"{tf} (k={km:.0f})" for tf, km in zip(key_sorted, k_meds)]
        ax.set_yticklabels(ylabels, fontsize=9)
        ax.set_xlabel("singular vector index")
        ax.set_title("Cross-covariance signal distribution across SVs (key TFs)")
        plt.colorbar(im, ax=ax, label="fraction of signal", shrink=0.8)

    fig.tight_layout()
    fig.savefig(figdir / "stage_b_alignment_heatmap_key.png", dpi=150)
    plt.close(fig)
    print(f"[FIG] stage_b_alignment_heatmap_key.png")

    # ----- Figure 4: alignment heatmap (ALL TFs, sorted by k_opt) -----
    all_sorted = df_b.sort_values("k_opt_median")["tf"].tolist()
    fig, ax = plt.subplots(figsize=(12, max(6, len(all_sorted) * 0.15)))
    heatmap_all = []
    for tf in all_sorted:
        prof = df_profiles[(df_profiles["tf"] == tf) &
                           (df_profiles["component"] < n_components_show)]
        if len(prof) == n_components_show:
            heatmap_all.append(prof["frac_signal"].values)
        else:
            heatmap_all.append(np.zeros(n_components_show))

    if heatmap_all:
        hm = np.array(heatmap_all)
        im = ax.imshow(hm, aspect="auto", cmap="viridis",
                       interpolation="nearest")
        ax.set_yticks(range(len(all_sorted)))
        k_meds = [df_b[df_b["tf"] == tf]["k_opt_median"].values[0]
                   for tf in all_sorted]
        ylabels = [f"{tf} (k={km:.0f})" for tf, km in zip(all_sorted, k_meds)]
        ax.set_yticklabels(ylabels, fontsize=5)
        ax.set_xlabel("singular vector index")
        ax.set_title("Cross-covariance signal distribution across SVs (all TFs, sorted by k_opt)")
        plt.colorbar(im, ax=ax, label="fraction of signal", shrink=0.6)

    fig.tight_layout()
    fig.savefig(figdir / "stage_b_alignment_heatmap_all.png", dpi=150)
    plt.close(fig)
    print(f"[FIG] stage_b_alignment_heatmap_all.png")

    # ----- Figure 5: cumulative signal curves for key TFs -----
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(key_sorted)))
    for i, tf in enumerate(key_sorted):
        prof = df_profiles[(df_profiles["tf"] == tf) &
                           (df_profiles["component"] < n_components_show)]
        if len(prof) > 0:
            km = df_b[df_b["tf"] == tf]["k_opt_median"].values[0]
            ax.plot(prof["component"].values, prof["cumul_signal"].values,
                    label=f"{tf} (k_opt={km:.0f})", color=colors[i], lw=1.5)
            # Mark k_opt on the curve
            k_int = int(km)
            if k_int < len(prof):
                y_at_k = prof[prof["component"] == k_int]["cumul_signal"].values
                if len(y_at_k) > 0:
                    ax.plot(k_int, y_at_k[0], "o", color=colors[i], markersize=6)

    ax.set_xlabel("number of SVs included")
    ax.set_ylabel("cumulative fraction of cross-cov signal")
    ax.set_title("Cumulative spectral alignment (key TFs)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(-0.5, n_components_show - 0.5)
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(figdir / "stage_b_cumulative_signal.png", dpi=150)
    plt.close(fig)
    print(f"[FIG] stage_b_cumulative_signal.png")


# =====================================================================
# Main
# =====================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(description="5B: Spectral alignment analysis")
    ap.add_argument("--data-dir", default="E:/data_cache",
                    help="Directory with cached bone marrow h5ad")
    ap.add_argument("--bootstrap-dir",
                    default="results/bm_bootstrap_k/csv",
                    help="Directory with bootstrap CSV results")
    ap.add_argument("--outdir", default="results/bm_spectral_alignment")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    csvdir = outdir / "csv"
    figdir = outdir / "figures"
    csvdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Load bootstrap results
    # ------------------------------------------------------------------
    bs_dir = Path(args.bootstrap_dir)
    bootstrap_summary = pd.read_csv(bs_dir / "bootstrap_summary.csv")
    print(f"[LOAD] bootstrap_summary: {len(bootstrap_summary)} TFs")

    # ------------------------------------------------------------------
    # Load and preprocess data
    # ------------------------------------------------------------------
    X, V, pt, genes = load_and_preprocess(args.data_dir)
    tf_targets, gene_to_idx = load_ground_truth(genes)

    # ------------------------------------------------------------------
    # STAGE A
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("COMPUTING STAGE A: BASELINE PREDICTORS")
    print("=" * 90)

    df_a = compute_stage_a(X, V, genes, tf_targets, gene_to_idx,
                           bootstrap_summary)
    df_a.to_csv(csvdir / "stage_a_predictors.csv", index=False)
    print(f"[SAVE] stage_a_predictors.csv ({len(df_a)} TFs)")

    report_stage_a(df_a)

    # ------------------------------------------------------------------
    # Compute full-dataset streaming pairspace stats
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("COMPUTING FULL-DATASET PAIRSPACE STATS (one pass)")
    print("=" * 90)

    # Sort by pseudotime for fast monotonic parallel path
    sort_order = np.argsort(pt)
    X_sorted = X[sort_order]
    V_sorted = V[sort_order]
    pt_sorted = pt[sort_order]

    t_pre = time.perf_counter()
    pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
        X_sorted, V_sorted, pt_sorted, pair_seed=42,
    )
    print(f"[PRE] Streaming stats: {time.perf_counter()-t_pre:.1f}s, "
          f"{pre['n_pairs']} pairs, G={pre['G']}")

    # Quick spectrum summary
    S = pre["S"]
    print(f"[PRE] Singular value range: {S[0]:.1f} ... {S[-1]:.4f}")
    print(f"[PRE] Top-5 SV explain {np.sum(S[:5]**2)/np.sum(S**2)*100:.1f}% of variance")
    print(f"[PRE] Top-10 SV explain {np.sum(S[:10]**2)/np.sum(S**2)*100:.1f}% of variance")
    print(f"[PRE] Top-25 SV explain {np.sum(S[:25]**2)/np.sum(S**2)*100:.1f}% of variance")

    # ------------------------------------------------------------------
    # STAGE B
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("COMPUTING STAGE B: SPECTRAL ALIGNMENT")
    print("=" * 90)

    df_b, df_profiles = compute_stage_b(pre, genes, tf_targets, gene_to_idx,
                                        df_a)
    df_b.to_csv(csvdir / "stage_b_alignment.csv", index=False)
    df_profiles.to_csv(csvdir / "stage_b_alignment_profiles.csv", index=False)
    print(f"[SAVE] stage_b_alignment.csv ({len(df_b)} TFs)")
    print(f"[SAVE] stage_b_alignment_profiles.csv ({len(df_profiles)} rows)")

    report_stage_b(df_b)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n[PLOT] Generating figures...")
    try:
        make_plots(df_b, df_profiles, figdir)
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

    # Compute key statistics for verdict
    rho_ntgt, p_ntgt = sp_stats.spearmanr(df_b["n_targets"], df_b["k_opt_median"])
    rho_var, p_var = sp_stats.spearmanr(df_b["mean_target_var"], df_b["k_opt_median"])
    rho_cent, p_cent = sp_stats.spearmanr(df_b["spectral_centroid"], df_b["k_opt_median"])
    rho_frac5, p_frac5 = sp_stats.spearmanr(df_b["frac_top5"], df_b["k_opt_median"])
    rho_taut, p_taut = sp_stats.spearmanr(
        df_b["spectral_centroid"], df_b["mean_target_var_rank"])

    # Baseline R^2
    y = df_b["k_opt_median"].values
    X_base = np.column_stack([
        (df_b["n_targets"].values - df_b["n_targets"].mean()) / (df_b["n_targets"].std() + 1e-12),
        (df_b["mean_target_var"].values - df_b["mean_target_var"].mean()) / (df_b["mean_target_var"].std() + 1e-12),
    ])
    X_base_aug = np.column_stack([np.ones(len(y)), X_base])
    beta_base = np.linalg.lstsq(X_base_aug, y, rcond=None)[0]
    r2_base = 1 - np.sum((y - X_base_aug @ beta_base)**2) / np.sum((y - y.mean())**2)

    # Baseline + spectral R^2
    X_spec = np.column_stack([
        X_base,
        (df_b["spectral_centroid"].values - df_b["spectral_centroid"].mean()) / (df_b["spectral_centroid"].std() + 1e-12),
    ])
    X_spec_aug = np.column_stack([np.ones(len(y)), X_spec])
    beta_spec = np.linalg.lstsq(X_spec_aug, y, rcond=None)[0]
    r2_spec = 1 - np.sum((y - X_spec_aug @ beta_spec)**2) / np.sum((y - y.mean())**2)
    delta_r2 = r2_spec - r2_base

    print(f"\n  Stage A baseline:    R^2 = {r2_base:.3f}")
    print(f"    n_targets vs k_opt:    rho = {rho_ntgt:+.3f} (p={p_ntgt:.3f})")
    print(f"    mean_tgt_var vs k_opt: rho = {rho_var:+.3f} (p={p_var:.3f})")

    print(f"\n  Stage B spectral:    R^2 = {r2_spec:.3f}  (delta = +{delta_r2:.3f})")
    print(f"    centroid vs k_opt:     rho = {rho_cent:+.3f} (p={p_cent:.3f})")
    print(f"    frac_top5 vs k_opt:    rho = {rho_frac5:+.3f} (p={p_frac5:.3f})")

    print(f"\n  Tautology:  centroid vs var_rank: rho = {rho_taut:+.3f} (p={p_taut:.3f})")
    if abs(rho_taut) > 0.9:
        print(f"    ==> TAUTOLOGICAL: spectral centroid is just variance ranking")
    elif abs(rho_taut) > 0.7:
        print(f"    ==> PARTIALLY TAUTOLOGICAL: substantial overlap with variance")
    else:
        print(f"    ==> NON-TAUTOLOGICAL: spectral centroid captures something beyond variance")

    print(f"\n  Interpretation:")
    if r2_base > 0.7:
        print(f"    Baseline R^2 = {r2_base:.3f} > 0.7 — trivial predictors dominate.")
        print(f"    Spectral alignment may not add substantial explanatory power.")
    elif r2_base > 0.3:
        print(f"    Baseline R^2 = {r2_base:.3f} — partial explanation.")
        if delta_r2 > 0.1:
            print(f"    Spectral alignment adds delta_R^2 = {delta_r2:.3f} — meaningful.")
        else:
            print(f"    Spectral alignment adds delta_R^2 = {delta_r2:.3f} — modest.")
    else:
        print(f"    Baseline R^2 = {r2_base:.3f} < 0.3 — trivial predictors fail.")
        if delta_r2 > 0.2:
            print(f"    Spectral alignment adds delta_R^2 = {delta_r2:.3f} — substantial contribution.")
        elif delta_r2 > 0.1:
            print(f"    Spectral alignment adds delta_R^2 = {delta_r2:.3f} — moderate contribution.")
        else:
            print(f"    Spectral alignment adds delta_R^2 = {delta_r2:.3f} — limited contribution.")

    wall = time.perf_counter() - t0
    print(f"\n[DONE] {wall:.1f}s total")


if __name__ == "__main__":
    main()
