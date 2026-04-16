"""
Benchmark: GeoBRIDGE vs Velorama on real scRNA-seq datasets.

Datasets:
  - Bone marrow (scVelo, 5780 cells, Setty 2019)
  - Pancreas (scVelo, ~3700 cells, Bastidas-Ponce 2019)

Design:
  Phase 1 (CPU/GPU): Prepare h5ad files with velocity + annotations
  Phase 2 (GPU): Run Velorama on prepared h5ads
  Phase 3 (CPU): Run GeoBRIDGE (k=0, k=5) and compare

Usage:
  # Phase 1: Prepare data
  python -u -m benchmark.bm_real_data_comparison --phase prepare

  # Phase 2: Run Velorama (GPU, on charybdis)
  python -u -m benchmark.bm_real_data_comparison --phase velorama --device cuda

  # Phase 3: Run GeoBRIDGE + compare
  python -u -m benchmark.bm_real_data_comparison --phase geobridge
  python -u -m benchmark.bm_real_data_comparison --phase compare
"""

import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

OUTDIR = Path("results/bm_real_comparison")
PREPARED_DIR = OUTDIR / "prepared_h5ad"

# Ground truth: CollecTRI via decoupler
MIN_TARGETS_PER_TF = 5

# GeoBRIDGE k values
K_VALUES = [0, 1, 3, 5, 10, 20, 50]

# Top-K values for precision evaluation
TOP_K_VALUES = [10, 20, 50, 100]

# Organism per dataset (pancreas = mouse, bonemarrow = human)
DATASET_ORGANISM = {
    "bonemarrow": "human",
    "pancreas": "mouse",
}


# =====================================================================
# Evaluation helpers
# =====================================================================

def _evaluate_tf(scores_tf, labels_tf, top_k_values=TOP_K_VALUES):
    """
    Compute all metrics for one TF.

    Args:
      scores_tf: 1D array of abs scores for each candidate gene
      labels_tf: 1D binary array (1 = true target)

    Returns dict with:
      auroc, auprc, n_targets, n_candidates
      top_k_precision_<k>, top_k_recall_<k>
      epr  = precision@n_targets / baseline
      gt_score_mean, gt_score_median, gt_score_rank_pct
      bg_score_mean, bg_score_median
      gt_vs_bg_ratio (mean GT score / mean BG score)
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    n_pos = int(labels_tf.sum())
    n_total = len(labels_tf)
    baseline = n_pos / n_total

    out = {
        "n_targets": n_pos,
        "n_candidates": n_total,
        "baseline_auprc": baseline,
        "auroc": roc_auc_score(labels_tf, scores_tf),
        "auprc": average_precision_score(labels_tf, scores_tf),
    }

    # Rank-based metrics
    order = np.argsort(-scores_tf)  # descending
    ranks = np.empty_like(order)
    ranks[order] = np.arange(n_total)

    # Top-K precision and recall
    for k in top_k_values:
        if k > n_total:
            continue
        top_k_labels = labels_tf[order[:k]]
        out[f"topk_precision_{k}"] = float(top_k_labels.mean())
        out[f"topk_recall_{k}"] = float(top_k_labels.sum() / max(n_pos, 1))

    # EPR: Early Precision Ratio (Velorama/BEELINE standard)
    # precision @ k = n_pos  divided by baseline
    k_epr = n_pos
    if k_epr > 0 and k_epr <= n_total:
        prec_at_npos = float(labels_tf[order[:k_epr]].mean())
        out["precision_at_npos"] = prec_at_npos
        out["epr"] = prec_at_npos / max(baseline, 1e-12)

    # GT score distribution
    gt_scores = scores_tf[labels_tf == 1]
    bg_scores = scores_tf[labels_tf == 0]
    out["gt_score_mean"] = float(gt_scores.mean())
    out["gt_score_median"] = float(np.median(gt_scores))
    out["bg_score_mean"] = float(bg_scores.mean())
    out["bg_score_median"] = float(np.median(bg_scores))
    out["gt_vs_bg_ratio"] = out["gt_score_mean"] / max(out["bg_score_mean"], 1e-12)

    # Mean percentile rank of GT targets (0 = best rank, 1 = worst)
    gt_ranks = ranks[labels_tf == 1]
    out["gt_rank_pct_mean"] = float(gt_ranks.mean() / n_total)
    out["gt_rank_pct_median"] = float(np.median(gt_ranks) / n_total)

    return out


# =====================================================================
# Data preparation
# =====================================================================

def _load_collectri_regulons(organism="human"):
    """Load CollecTRI regulons via decoupler.

    Falls back to human if mouse ortholog download fails (EBI 404).
    """
    import decoupler as dc

    for org in ([organism, "human"] if organism != "human" else ["human"]):
        for fn in [
            lambda: dc.op.collectri(organism=org),
            lambda: dc.get_collectri(organism=org),
            lambda: dc.op.get_collectri(organism=org),
        ]:
            try:
                net = fn()
                if org != organism:
                    print(f"  [WARN] Using human CollecTRI (mouse ortholog unavailable)")
                return net
            except (AttributeError, TypeError):
                continue
            except Exception as e:
                if "404" in str(e) or "HTTPError" in str(type(e).__name__):
                    print(f"  [WARN] {org} CollecTRI download failed: {e}")
                    break  # try next organism
                continue

    print(f"  [ERROR] Cannot load CollecTRI (decoupler {dc.__version__})")
    return None


def _build_ground_truth(net, gene_names, min_targets=MIN_TARGETS_PER_TF,
                        uppercase_match=False):
    """
    Build ground truth from regulon network.

    If uppercase_match=True, upper-cases gene_names for matching against a
    human-style regulon network (useful when dataset is mouse but only human
    CollecTRI is available: mouse 'Ins1' -> 'INS1' ~ human).

    Returns:
      gt_dict: {tf: set(target_genes_in_original_case)}
    """
    if uppercase_match:
        name_upper_to_orig = {g.upper(): g for g in gene_names}
        gene_set = set(name_upper_to_orig.keys())
    else:
        gene_set = set(gene_names)
        name_upper_to_orig = {g: g for g in gene_names}

    # Filter to genes in our dataset (after case-matching)
    net_filtered = net[
        net["source"].isin(gene_set) & net["target"].isin(gene_set)
    ].copy()

    # Count targets per TF
    tf_counts = net_filtered.groupby("source")["target"].nunique()
    valid_tfs = tf_counts[tf_counts >= min_targets].index.tolist()

    # Map back to original-case names
    gt_dict = {}
    for tf in valid_tfs:
        tf_orig = name_upper_to_orig[tf]
        targets_net = set(net_filtered[net_filtered["source"] == tf]["target"].unique())
        targets_orig = {name_upper_to_orig[t] for t in targets_net if t in name_upper_to_orig}
        targets_orig.discard(tf_orig)  # no self-loops
        if len(targets_orig) >= min_targets:
            gt_dict[tf_orig] = targets_orig

    print(f"  GT: {len(gt_dict)} TFs with >= {min_targets} targets"
          f"{' (uppercase match)' if uppercase_match else ''}")
    return gt_dict


def prepare_bonemarrow():
    """Prepare bone marrow dataset for both methods."""
    print("\n" + "=" * 70)
    print("PREPARING: Bone Marrow")
    print("=" * 70)

    adata = scv.datasets.bonemarrow()
    print(f"  Raw: {adata.shape}")

    # Preprocess
    scv.pp.filter_genes(adata, min_shared_counts=20)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3",
                                layer="spliced")
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"  After HVG: {adata.shape}")

    # Velocity
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(adata, mode="stochastic")
    print(f"  Velocity computed")

    # Pseudotime
    if "palantir_pseudotime" in adata.obs:
        adata.obs["pseudotime"] = adata.obs["palantir_pseudotime"]
        print(f"  Using Palantir pseudotime")
    else:
        sc.pp.pca(adata, n_comps=30)
        sc.pp.neighbors(adata, n_pcs=30)
        sc.tl.diffmap(adata)
        # Root: highest CD34
        if "CD34" in adata.var_names:
            adata.uns["iroot"] = int(np.argmax(adata[:, "CD34"].X.toarray().flatten()))
        else:
            adata.uns["iroot"] = 0
        sc.tl.dpt(adata)
        adata.obs["pseudotime"] = adata.obs["dpt_pseudotime"]
        print(f"  Using DPT pseudotime")

    # Velorama annotations: all genes are candidates
    adata.var["is_reg"] = True
    adata.var["is_target"] = True

    # Save
    out_path = PREPARED_DIR / "bonemarrow.h5ad"
    adata.write_h5ad(str(out_path))
    print(f"  Saved: {out_path} ({adata.shape})")
    return adata


def prepare_pancreas():
    """Prepare pancreas dataset for both methods."""
    print("\n" + "=" * 70)
    print("PREPARING: Pancreas")
    print("=" * 70)

    adata = scv.datasets.pancreas()
    print(f"  Raw: {adata.shape}")

    # Preprocess (split steps to avoid scVelo API compat issues)
    scv.pp.filter_genes(adata, min_shared_counts=20)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor="seurat_v3",
                                layer="spliced")
    adata = adata[:, adata.var["highly_variable"]].copy()
    print(f"  After filter: {adata.shape}")

    # Velocity
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(adata, mode="stochastic")
    print(f"  Velocity computed")

    # Pseudotime via DPT
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=30)
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_pcs=30)
    sc.tl.diffmap(adata)

    # Root: use Ngn3 high EP or ductal progenitor
    if "clusters" in adata.obs:
        # Find a stem-like cluster
        for root_cluster in ["Ductal", "Ngn3 high EP", "Pre-endocrine"]:
            mask = adata.obs["clusters"] == root_cluster
            if mask.any():
                adata.uns["iroot"] = int(np.where(mask)[0][0])
                print(f"  Root cell from cluster '{root_cluster}'")
                break
        else:
            adata.uns["iroot"] = 0
    else:
        adata.uns["iroot"] = 0

    sc.tl.dpt(adata)
    adata.obs["pseudotime"] = adata.obs["dpt_pseudotime"]
    print(f"  Using DPT pseudotime")

    # Velorama annotations
    adata.var["is_reg"] = True
    adata.var["is_target"] = True

    # Save
    out_path = PREPARED_DIR / "pancreas.h5ad"
    adata.write_h5ad(str(out_path))
    print(f"  Saved: {out_path} ({adata.shape})")
    return adata


def phase_prepare():
    """Phase 1: Prepare all datasets."""
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)

    prepare_bonemarrow()
    prepare_pancreas()

    print(f"\n[DONE] Prepared h5ads in {PREPARED_DIR}/")
    print(f"  Next: scp to charybdis and run --phase velorama")


# =====================================================================
# Velorama
# =====================================================================

def phase_velorama(device="cuda"):
    """Phase 2: Run Velorama on prepared h5ads."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    csvdir = OUTDIR / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)

    velorama_outdir = OUTDIR / "velorama_output"
    velorama_outdir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "bonemarrow": PREPARED_DIR / "bonemarrow.h5ad",
        "pancreas": PREPARED_DIR / "pancreas.h5ad",
    }

    for ds_name, h5ad_path in datasets.items():
        if not h5ad_path.exists():
            print(f"  [SKIP] {ds_name}: {h5ad_path} not found")
            continue

        print(f"\n{'='*70}")
        print(f"VELORAMA on {ds_name}")
        print(f"{'='*70}")

        # Velorama needs root_dir containing the h5ad
        # Use the prepared dir as root, dataset name must match filename
        # IMPORTANT: use absolute paths — Ray Tune workers run from a different cwd
        rd_abs = str(PREPARED_DIR.resolve())
        sd_abs = str(velorama_outdir.resolve())
        cmd = [
            "velorama",
            "-n", ds_name,
            "-ds", ds_name,
            "-dyn", "rna_velocity",
            "-dev", device,
            "-l", "5",
            "-hd", "32",
            "-mi", "1000",
            "-rd", rd_abs,
            "-sd", sd_abs,
            # Narrow lambda range (informed by SERGIO diagnosis)
            "-ls", "-2.3",
            "-le", "-0.7",
            "-nl", "19",
        ]
        print(f"  Running: {' '.join(cmd)}")
        t0 = time.perf_counter()

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=14400)
            elapsed = time.perf_counter() - t0
            print(f"  Completed in {elapsed:.1f}s, returncode={result.returncode}")
            if result.returncode != 0:
                print(f"  STDERR (last 500): {result.stderr[-500:]}")

                # Try default lambda range as fallback
                print(f"  Retrying with default lambda range...")
                cmd_default = [
                    "velorama",
                    "-n", ds_name,
                    "-ds", ds_name,
                    "-dyn", "rna_velocity",
                    "-dev", device,
                    "-l", "5", "-hd", "32", "-mi", "1000",
                    "-rd", rd_abs,
                    "-sd", sd_abs,
                ]
                result2 = subprocess.run(cmd_default, capture_output=True, text=True,
                                         timeout=14400)
                print(f"  Fallback completed, rc={result2.returncode}")
                if result2.returncode != 0:
                    print(f"  Fallback STDERR: {result2.stderr[-500:]}")

        except subprocess.TimeoutExpired:
            print(f"  TIMEOUT after 4h")
        except FileNotFoundError:
            print(f"  ERROR: 'velorama' not found")
            return

    print(f"\n[DONE] Velorama outputs in {velorama_outdir}/")


# =====================================================================
# GeoBRIDGE
# =====================================================================

def phase_geobridge():
    """Phase 3: Run GeoBRIDGE on prepared h5ads."""
    from geobridge_model import GeoBridgeModel
    from sklearn.metrics import average_precision_score, roc_auc_score

    csvdir = OUTDIR / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "bonemarrow": PREPARED_DIR / "bonemarrow.h5ad",
        "pancreas": PREPARED_DIR / "pancreas.h5ad",
    }

    results = []

    for ds_name, h5ad_path in datasets.items():
        if not h5ad_path.exists():
            print(f"  [SKIP] {ds_name}: {h5ad_path} not found")
            continue

        print(f"\n{'='*70}")
        print(f"GEOBRIDGE on {ds_name}")
        print(f"{'='*70}")

        adata = sc.read_h5ad(str(h5ad_path))
        gene_names = list(adata.var_names)
        G = len(gene_names)
        print(f"  {adata.shape[0]} cells x {G} genes")

        # Ground truth (organism-specific)
        organism = DATASET_ORGANISM.get(ds_name, "human")
        net = _load_collectri_regulons(organism=organism)
        if net is None:
            print(f"  [ERROR] Cannot load CollecTRI for {organism}. Skipping.")
            continue
        # If the organism we wanted wasn't the one we got (mouse->human fallback),
        # uppercase mouse gene names to match human CollecTRI.
        uppercase = (organism == "mouse")
        gt_dict = _build_ground_truth(net, gene_names, uppercase_match=uppercase)
        if not gt_dict:
            print(f"  [SKIP] No TFs with sufficient targets")
            continue

        # Expression and velocity
        X = np.asarray(adata.layers["Ms"] if "Ms" in adata.layers
                       else (adata.X.toarray() if hasattr(adata.X, "toarray")
                             else adata.X),
                       dtype=float)
        V = np.asarray(adata.layers["velocity"], dtype=float)
        V = np.nan_to_num(V, nan=0.0)
        pt = np.asarray(adata.obs["pseudotime"], dtype=float)
        pt = np.nan_to_num(pt, nan=float(np.nanmedian(pt)))

        # Sort by pseudotime
        order = np.argsort(pt)
        X_sorted, V_sorted, pt_sorted = X[order], V[order], pt[order]

        # Pairspace stats
        print(f"  Computing pairspace stats...")
        t0 = time.perf_counter()
        pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
            X_sorted, V_sorted, pt_sorted, pair_seed=42,
        )
        print(f"  Stats: {time.perf_counter()-t0:.1f}s, {pre['n_pairs']} pairs")

        # Evaluate at each k
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        for k in K_VALUES:
            if k >= len(pre["S"]):
                continue

            S_k = GeoBridgeModel.scores_from_streaming_stats(
                pre, k=k, project_y=True, normalize_score=True, eps=1e-12,
            )

            # Per-TF evaluation
            tf_results = []
            for tf, targets in gt_dict.items():
                tf_idx = gene_to_idx.get(tf)
                if tf_idx is None:
                    continue

                scores_tf = []
                labels_tf = []
                for j in range(G):
                    if j == tf_idx:
                        continue
                    gene_j = gene_names[j]
                    scores_tf.append(abs(S_k[tf_idx, j]))
                    labels_tf.append(1 if gene_j in targets else 0)

                labels_tf = np.array(labels_tf)
                scores_tf = np.array(scores_tf)

                if labels_tf.sum() == 0 or labels_tf.sum() == len(labels_tf):
                    continue

                metrics = _evaluate_tf(scores_tf, labels_tf)
                row = {
                    "dataset": ds_name,
                    "method": "GeoBRIDGE",
                    "k": k,
                    "tf": tf,
                    **metrics,
                }
                tf_results.append(row)

            if tf_results:
                tf_df = pd.DataFrame(tf_results)
                print(f"  k={k:3d}: "
                      f"AUROC={tf_df['auroc'].mean():.4f}±{tf_df['auroc'].std():.3f}  "
                      f"AUPRC={tf_df['auprc'].mean():.4f}±{tf_df['auprc'].std():.3f}  "
                      f"EPR={tf_df['epr'].mean():.2f}  "
                      f"GT/BG={tf_df['gt_vs_bg_ratio'].mean():.3f}  "
                      f"({len(tf_df)} TFs)")

                results.extend(tf_results)

        # Incremental save after each dataset (so a pancreas crash doesn't lose bonemarrow)
        if results:
            df = pd.DataFrame(results)
            df.to_csv(csvdir / "geobridge_real_results.csv", index=False)
            print(f"  [SAVE] geobridge_real_results.csv ({len(df)} rows so far)")

    if results:
        print(f"\n[DONE] {len(results)} total rows saved to geobridge_real_results.csv")


# =====================================================================
# Compare
# =====================================================================

def phase_compare():
    """Compare GeoBRIDGE and Velorama on real data."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    csvdir = OUTDIR / "csv"
    velorama_outdir = OUTDIR / "velorama_output"

    # Load GeoBRIDGE results
    gb_path = csvdir / "geobridge_real_results.csv"
    if not gb_path.exists():
        print("[ERROR] Run --phase geobridge first")
        return
    gb_df = pd.read_csv(gb_path)

    datasets = {
        "bonemarrow": PREPARED_DIR / "bonemarrow.h5ad",
        "pancreas": PREPARED_DIR / "pancreas.h5ad",
    }

    all_results = [gb_df]

    for ds_name, h5ad_path in datasets.items():
        # Find Velorama TSV
        tsv_path = velorama_outdir / f"{ds_name}.rna_velocity.velorama.interactions.tsv"
        if not tsv_path.exists():
            # Search alternatives
            for f in velorama_outdir.glob(f"*{ds_name}*.interactions.tsv"):
                tsv_path = f
                break

        if not tsv_path.exists():
            print(f"  [SKIP] No Velorama output for {ds_name}")
            continue

        print(f"\nEvaluating Velorama on {ds_name}...")

        # Load adata for gene names
        adata = sc.read_h5ad(str(h5ad_path))
        gene_names = list(adata.var_names)
        G = len(gene_names)
        gene_to_idx = {g: i for i, g in enumerate(gene_names)}

        # Load Velorama TSV
        interaction_df = pd.read_csv(tsv_path, sep="\t", index_col=0)
        interaction_df.index = interaction_df.index.astype(str)
        interaction_df.columns = interaction_df.columns.astype(str)
        print(f"  Velorama TSV: {interaction_df.shape}")

        # Ground truth (organism-specific)
        organism = DATASET_ORGANISM.get(ds_name, "human")
        net = _load_collectri_regulons(organism=organism)
        if net is None:
            print(f"  [ERROR] Cannot load CollecTRI for {organism}. Skipping.")
            continue
        # If the organism we wanted wasn't the one we got (mouse->human fallback),
        # uppercase mouse gene names to match human CollecTRI.
        uppercase = (organism == "mouse")
        gt_dict = _build_ground_truth(net, gene_names, uppercase_match=uppercase)

        vr_results = []
        for tf, targets in gt_dict.items():
            if tf not in interaction_df.columns:
                continue

            scores_tf = []
            labels_tf = []
            for gene_j in gene_names:
                if gene_j == tf:
                    continue
                if gene_j not in interaction_df.index:
                    continue
                scores_tf.append(abs(interaction_df.loc[gene_j, tf]))
                labels_tf.append(1 if gene_j in targets else 0)

            labels_tf = np.array(labels_tf)
            scores_tf = np.array(scores_tf)

            if labels_tf.sum() == 0 or labels_tf.sum() == len(labels_tf):
                continue

            metrics = _evaluate_tf(scores_tf, labels_tf)
            vr_results.append({
                "dataset": ds_name,
                "method": "Velorama",
                "k": -1,
                "tf": tf,
                **metrics,
            })

        if vr_results:
            vr_df = pd.DataFrame(vr_results)
            print(f"  Velorama: "
                  f"AUROC={vr_df['auroc'].mean():.4f}±{vr_df['auroc'].std():.3f}  "
                  f"AUPRC={vr_df['auprc'].mean():.4f}±{vr_df['auprc'].std():.3f}  "
                  f"EPR={vr_df['epr'].mean():.2f}  "
                  f"({len(vr_df)} TFs)")
            all_results.append(vr_df)

    # Combine and summarize
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(csvdir / "real_comparison_full.csv", index=False)

    # Summary table (per dataset × method × k)
    print(f"\n{'='*100}")
    print(f"SUMMARY: Real Data Comparison")
    print(f"{'='*100}")

    header = (f"{'Dataset':<12s} {'Method':<22s} "
              f"{'AUROC':>8s}±{'sd':<5s} {'AUPRC':>8s}±{'sd':<5s} "
              f"{'EPR':>6s} {'GT/BG':>7s} {'P@20':>6s} {'P@100':>6s} {'N':>4s}")
    print(header)
    print("-" * 100)

    for ds_name in ["bonemarrow", "pancreas"]:
        # Velorama row(s)
        for method, k in [("Velorama", -1)] + [("GeoBRIDGE", kv) for kv in K_VALUES]:
            mask = (combined["dataset"] == ds_name) & (combined["method"] == method)
            if k >= 0:
                mask = mask & (combined["k"] == k)
            sub = combined[mask]
            if len(sub) == 0:
                continue
            label = f"{method}" if k < 0 else f"{method} k={k}"
            row = (
                f"  {ds_name:<10s} {label:<22s} "
                f"{sub['auroc'].mean():8.4f}±{sub['auroc'].std():5.3f} "
                f"{sub['auprc'].mean():8.4f}±{sub['auprc'].std():5.3f} "
                f"{sub['epr'].mean():6.2f} "
                f"{sub['gt_vs_bg_ratio'].mean():7.3f} "
                f"{sub.get('topk_precision_20', pd.Series([np.nan])).mean():6.3f} "
                f"{sub.get('topk_precision_100', pd.Series([np.nan])).mean():6.3f} "
                f"{len(sub):4d}"
            )
            print(row)
        print()

    # Per-dataset aggregate summary CSV (one row per method×k)
    agg_rows = []
    metric_cols = [c for c in combined.columns
                   if c not in ("dataset", "method", "k", "tf")]
    for (ds, method, k), grp in combined.groupby(["dataset", "method", "k"]):
        row = {"dataset": ds, "method": method, "k": k, "n_tfs": len(grp)}
        for col in metric_cols:
            if col in grp.columns:
                row[f"{col}_mean"] = grp[col].mean()
                row[f"{col}_std"] = grp[col].std()
                row[f"{col}_median"] = grp[col].median()
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows)
    agg_df.to_csv(csvdir / "real_comparison_aggregate.csv", index=False)

    print(f"\n[SAVE] real_comparison_full.csv ({len(combined)} rows)")
    print(f"[SAVE] real_comparison_aggregate.csv ({len(agg_df)} rows)")


# =====================================================================
# Main
# =====================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["prepare", "velorama", "geobridge", "compare"],
                    help="Which phase to run")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.phase == "prepare":
        phase_prepare()
    elif args.phase == "velorama":
        phase_velorama(device=args.device)
    elif args.phase == "geobridge":
        phase_geobridge()
    elif args.phase == "compare":
        phase_compare()


if __name__ == "__main__":
    main()
