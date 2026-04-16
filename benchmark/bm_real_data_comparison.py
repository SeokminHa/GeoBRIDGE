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

# GeoBRIDGE k values (minimal, per user directive)
K_VALUES = [0, 5]

# Organism per dataset (pancreas = mouse, bonemarrow = human)
DATASET_ORGANISM = {
    "bonemarrow": "human",
    "pancreas": "mouse",
}


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


def _build_ground_truth(net, gene_names, min_targets=MIN_TARGETS_PER_TF):
    """
    Build ground truth from regulon network.

    Returns:
      tf_list: list of TF names with >= min_targets in gene_names
      gt_dict: {tf: set(target_genes)} for evaluation
    """
    gene_set = set(gene_names)

    # Filter to genes in our dataset
    net_filtered = net[
        net["source"].isin(gene_set) & net["target"].isin(gene_set)
    ].copy()

    # Count targets per TF
    tf_counts = net_filtered.groupby("source")["target"].nunique()
    valid_tfs = tf_counts[tf_counts >= min_targets].index.tolist()

    gt_dict = {}
    for tf in valid_tfs:
        targets = set(net_filtered[net_filtered["source"] == tf]["target"].unique())
        targets.discard(tf)  # no self-loops
        if len(targets) >= min_targets:
            gt_dict[tf] = targets

    print(f"  GT: {len(gt_dict)} TFs with >= {min_targets} targets")
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
        gt_dict = _build_ground_truth(net, gene_names)
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

                auroc = roc_auc_score(labels_tf, scores_tf)
                auprc = average_precision_score(labels_tf, scores_tf)
                n_tgt = int(labels_tf.sum())

                tf_results.append({
                    "dataset": ds_name,
                    "method": "GeoBRIDGE",
                    "k": k,
                    "tf": tf,
                    "n_targets": n_tgt,
                    "auroc": auroc,
                    "auprc": auprc,
                })

            if tf_results:
                tf_df = pd.DataFrame(tf_results)
                mean_auroc = tf_df["auroc"].mean()
                mean_auprc = tf_df["auprc"].mean()
                print(f"  k={k}: mean AUROC={mean_auroc:.4f}, "
                      f"mean AUPRC={mean_auprc:.4f} ({len(tf_df)} TFs)")

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
        gt_dict = _build_ground_truth(net, gene_names)

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

            auroc = roc_auc_score(labels_tf, scores_tf)
            auprc = average_precision_score(labels_tf, scores_tf)

            vr_results.append({
                "dataset": ds_name,
                "method": "Velorama",
                "k": -1,
                "tf": tf,
                "n_targets": int(labels_tf.sum()),
                "auroc": auroc,
                "auprc": auprc,
            })

        if vr_results:
            vr_df = pd.DataFrame(vr_results)
            print(f"  Velorama: mean AUROC={vr_df['auroc'].mean():.4f}, "
                  f"mean AUPRC={vr_df['auprc'].mean():.4f} ({len(vr_df)} TFs)")
            all_results.append(vr_df)

    # Combine and summarize
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(csvdir / "real_comparison_full.csv", index=False)

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY: Real Data Comparison")
    print(f"{'='*70}")
    print(f"\n{'Dataset':<14s} {'Method':<22s} {'Mean AUROC':>10s} {'Mean AUPRC':>10s} "
          f"{'N TFs':>6s}")
    print("-" * 70)

    for ds_name in ["bonemarrow", "pancreas"]:
        for method_k in [("Velorama", -1), ("GeoBRIDGE", 0), ("GeoBRIDGE", 5)]:
            method, k = method_k
            mask = (combined["dataset"] == ds_name) & (combined["method"] == method)
            if k >= 0:
                mask = mask & (combined["k"] == k)
            sub = combined[mask]
            if len(sub) == 0:
                continue
            label = f"{method}" if k < 0 else f"{method} (k={k})"
            print(f"  {ds_name:<12s} {label:<22s} "
                  f"{sub['auroc'].mean():10.4f} {sub['auprc'].mean():10.4f} "
                  f"{len(sub):6d}")
        print()

    print(f"\n[SAVE] real_comparison_full.csv ({len(combined)} rows)")


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
