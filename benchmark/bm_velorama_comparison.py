"""
Benchmark: GeoBRIDGE vs Velorama on Velorama's synthetic datasets.

Two-phase design (can run on different machines):
  Phase 1 (CPU, e.g. dragon): Run GeoBRIDGE on all datasets
  Phase 2 (GPU, e.g. charybdis): Run Velorama on all datasets
  Phase 3 (CPU): Evaluate and compare both methods

Usage:
  # Phase 1: GeoBRIDGE (CPU)
  python -m benchmark.bm_velorama_comparison --phase geobridge --data-dir results/bm_velorama/velorama_data --outdir results/bm_velorama

  # Phase 2: Velorama (GPU)
  python -m benchmark.bm_velorama_comparison --phase velorama --data-dir results/bm_velorama/velorama_data --outdir results/bm_velorama --device cuda

  # Phase 3: Compare
  python -m benchmark.bm_velorama_comparison --phase compare --outdir results/bm_velorama
"""

import sys
import time
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

DATASETS = ["dataset_A", "dataset_B", "dataset_C", "dataset_D"]
KEY_K_VALUES = [0, 1, 2, 3, 5, 7, 10, 15, 20, 25]

# All four Velorama synthetic datasets share the same SERGIO GRN topology.
# 137 directed edges among 100 genes (IDs 0-99).
# Source: https://github.com/PayamDiba/SERGIO/tree/master/data_sets/
SERGIO_GT_URL = (
    "https://raw.githubusercontent.com/PayamDiba/SERGIO/master/"
    "data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/gt_GRN.csv"
)


# =====================================================================
# Ground truth extraction
# =====================================================================

def download_sergio_gt(data_dir):
    """Download the shared SERGIO ground truth GRN edge list."""
    gt_path = Path(data_dir) / "gt_GRN.csv"
    if gt_path.exists():
        print(f"  [GT] Already downloaded: {gt_path}")
        return gt_path
    print(f"  [GT] Downloading SERGIO ground truth from {SERGIO_GT_URL}")
    urllib.request.urlretrieve(SERGIO_GT_URL, gt_path)
    print(f"  [GT] Saved to {gt_path}")
    return gt_path


def load_ground_truth(gt_path, n_genes=100):
    """
    Load SERGIO ground truth GRN from edge list CSV.

    Format: regulator_id,target_id  (0-indexed, no header)
    Returns (n_genes, n_genes) binary adjacency matrix where
    gt[reg, tgt] = 1 means reg → tgt edge exists.
    """
    edges = pd.read_csv(gt_path, header=None, names=["reg", "tgt"])
    gt = np.zeros((n_genes, n_genes), dtype=float)
    for _, row in edges.iterrows():
        gt[int(row["reg"]), int(row["tgt"])] = 1
    n_edges = int(gt.sum())
    print(f"  [GT] Loaded {n_edges} edges among {n_genes} genes from {gt_path.name}")
    return gt


# =====================================================================
# Phase 1: GeoBRIDGE
# =====================================================================

def run_geobridge(data_dir, outdir):
    """Run GeoBRIDGE on all Velorama synthetic datasets."""
    import scanpy as sc
    from geobridge_model import GeoBridgeModel

    csvdir = Path(outdir) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)

    # Download and load shared ground truth
    gt_path = download_sergio_gt(data_dir)
    n_genes = 100  # all SERGIO datasets use 100 genes
    gt_matrix = load_ground_truth(gt_path, n_genes=n_genes)

    results = []

    for ds_name in DATASETS:
        fpath = Path(data_dir) / f"{ds_name}.h5ad"
        if not fpath.exists():
            print(f"  [SKIP] {ds_name} not found")
            continue

        print(f"\n{'='*70}")
        print(f"GeoBRIDGE on {ds_name}")
        print(f"{'='*70}")

        adata = sc.read_h5ad(str(fpath))
        n_cells, G = adata.shape
        print(f"  {n_cells} cells x {G} genes")
        assert G == n_genes, f"Expected {n_genes} genes, got {G}"

        # Get expression
        X = np.asarray(adata.X.toarray() if hasattr(adata.X, 'toarray')
                       else adata.X, dtype=float)

        # Compute RNA velocity from spliced/unspliced
        V = None
        for vel_key in ['velocity', 'velocities']:
            if vel_key in adata.layers:
                V = np.asarray(adata.layers[vel_key], dtype=float)
                print(f"  Using precomputed velocity from layers['{vel_key}']")
                break

        if V is None:
            if 'spliced' in adata.layers and 'unspliced' in adata.layers:
                import scvelo as scv
                print(f"  Computing velocity from spliced/unspliced...")
                scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
                scv.tl.velocity(adata, mode="stochastic")
                V = np.asarray(adata.layers['velocity'], dtype=float)
                # Use smoothed expression for X
                if 'Ms' in adata.layers:
                    X = np.asarray(adata.layers['Ms'], dtype=float)
                    print(f"  Using smoothed expression (Ms) for X")
            else:
                print(f"  [ERROR] No velocity and no spliced/unspliced for {ds_name}")
                continue

        V = np.nan_to_num(V, nan=0.0)

        # Pseudotime: use 'time' column if available (SERGIO simulation time)
        pt = None
        for pt_key in ['time', 'pseudotime', 'latent_time', 'dpt_pseudotime']:
            if pt_key in adata.obs:
                pt = np.asarray(adata.obs[pt_key], dtype=float)
                print(f"  Using pseudotime from obs['{pt_key}']")
                break

        if pt is None:
            print(f"  Computing pseudotime via DPT...")
            if 'X_pca' not in adata.obsm:
                sc.pp.pca(adata, n_comps=min(30, G - 1))
            if 'neighbors' not in adata.uns:
                sc.pp.neighbors(adata, n_pcs=min(30, adata.obsm['X_pca'].shape[1]))
            sc.tl.diffmap(adata)
            adata.uns['iroot'] = int(adata.uns.get('iroot', 0))
            sc.tl.dpt(adata)
            pt = np.asarray(adata.obs['dpt_pseudotime'], dtype=float)

        pt = np.nan_to_num(pt, nan=float(np.nanmedian(pt)))

        # Sort by pseudotime
        sort_order = np.argsort(pt)
        X_sorted = X[sort_order]
        V_sorted = V[sort_order]
        pt_sorted = pt[sort_order]

        # Compute streaming stats
        print(f"  Computing pairspace stats...")
        t0 = time.perf_counter()
        pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
            X_sorted, V_sorted, pt_sorted, pair_seed=42,
        )
        print(f"  Stats: {time.perf_counter()-t0:.1f}s, {pre['n_pairs']} pairs")

        # Evaluate at each k
        from sklearn.metrics import average_precision_score, roc_auc_score

        for k in KEY_K_VALUES:
            if k >= len(pre["S"]):
                continue

            S_k = GeoBridgeModel.scores_from_streaming_stats(
                pre, k=k, project_y=True, normalize_score=True, eps=1e-12,
            )

            # All genes are both regulators and targets.
            # Build score/label vectors for all (i, j) pairs where i != j.
            scores = []
            labels = []
            for i in range(G):
                for j in range(G):
                    if i == j:
                        continue
                    scores.append(abs(S_k[i, j]))
                    labels.append(int(gt_matrix[i, j] > 0))

            scores = np.array(scores)
            labels = np.array(labels)

            if labels.sum() == 0 or labels.sum() == len(labels):
                continue

            auroc = roc_auc_score(labels, scores)
            auprc = average_precision_score(labels, scores)
            baseline_auprc = labels.mean()  # random baseline

            results.append({
                "dataset": ds_name,
                "method": "GeoBRIDGE",
                "k": k,
                "auroc": auroc,
                "auprc": auprc,
                "baseline_auprc": baseline_auprc,
                "n_genes": G,
                "n_edges": int(labels.sum()),
                "n_pairs": len(labels),
            })

            print(f"    k={k:2d}: AUROC={auroc:.4f}, AUPRC={auprc:.4f} "
                  f"(baseline={baseline_auprc:.4f})")

    df = pd.DataFrame(results)
    df.to_csv(csvdir / "geobridge_results.csv", index=False)
    print(f"\n[SAVE] geobridge_results.csv ({len(df)} rows)")
    return df


# =====================================================================
# Phase 2: Velorama
# =====================================================================

def _find_velorama_tsv(ds_name, dynamics, search_dirs):
    """Search for Velorama output TSV across multiple directories and names.

    Prefers per-dataset files ({ds_name}.*) over the generic fallback
    (velorama_run.*), which may be stale from a previous run where all
    datasets overwrote the same file (only the last dataset survives).
    """
    # Per-dataset name first (preferred)
    for d in search_dirs:
        d = Path(d)
        f = d / f"{ds_name}.{dynamics}.velorama.interactions.tsv"
        if f.exists():
            return f
    # Fallback: generic name (WARNING: may be from a different dataset)
    for d in search_dirs:
        d = Path(d)
        f = d / f"velorama_run.{dynamics}.velorama.interactions.tsv"
        if f.exists():
            return f
    return None


def run_velorama(data_dir, outdir, device="cuda"):
    """Run Velorama on all datasets via CLI, then evaluate against GT.

    Velorama CLI saves output as:
      {save_dir}/{name}.{dynamics}.velorama.interactions.tsv
    We use -n {ds_name} so each dataset gets a unique file.
    -rd (root_dir) = directory containing {ds_name}.h5ad
    -sd (save_dir) = directory where output TSVs go (MUST differ from root_dir
                     because Velorama also uses os.path.join(root_dir, save_dir)
                     for Ray Tune local_dir).
    """
    import shutil
    csvdir = Path(outdir) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)

    # Velorama's root_dir: where the h5ad files live
    velorama_rootdir = Path(outdir) / "velorama_data"
    velorama_rootdir.mkdir(parents=True, exist_ok=True)

    # Velorama's save_dir: where output TSVs + checkpoints go
    velorama_savedir = Path(outdir) / "velorama_results"
    velorama_savedir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Run Velorama CLI on each dataset ----
    for ds_name in DATASETS:
        fpath = Path(data_dir) / f"{ds_name}.h5ad"
        if not fpath.exists():
            print(f"  [SKIP] {ds_name} not found")
            continue

        print(f"\n{'='*70}")
        print(f"Velorama on {ds_name}")
        print(f"{'='*70}")

        # Copy h5ad to Velorama root_dir
        dst = velorama_rootdir / f"{ds_name}.h5ad"
        if not dst.exists():
            shutil.copy2(fpath, dst)

        for dynamics in ["pseudotime", "rna_velocity"]:
            # Check if already computed (search multiple locations)
            existing = _find_velorama_tsv(
                ds_name, dynamics,
                [velorama_savedir, Path(outdir) / "velorama_outputs",
                 Path("results")]
            )
            if existing is not None:
                print(f"  [SKIP] {ds_name}/{dynamics} found at {existing}")
                continue

            cmd = [
                "velorama",
                "-n", ds_name,
                "-ds", ds_name,
                "-dyn", dynamics,
                "-dev", device,
                "-l", "5",
                "-hd", "32",
                "-rd", str(velorama_rootdir),
                "-sd", str(velorama_savedir),
            ]
            print(f"  Running: {' '.join(cmd)}")
            t0 = time.perf_counter()

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=7200,
                )
                elapsed = time.perf_counter() - t0
                print(f"  Completed in {elapsed:.1f}s, returncode={result.returncode}")
                if result.returncode != 0:
                    print(f"  STDERR (last 500): {result.stderr[-500:]}")
                    print(f"  STDOUT (last 500): {result.stdout[-500:]}")
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT after 7200s")
            except FileNotFoundError:
                print(f"  ERROR: 'velorama' command not found. "
                      f"Install via: pip install velorama")
                return

    # ---- Step 2: Evaluate Velorama results against GT ----
    print(f"\n{'='*70}")
    print(f"EVALUATING VELORAMA RESULTS")
    print(f"{'='*70}")

    gt_path = download_sergio_gt(data_dir)
    n_genes = 100
    gt_matrix = load_ground_truth(gt_path, n_genes=n_genes)

    from sklearn.metrics import average_precision_score, roc_auc_score

    # Search all likely directories for Velorama output
    search_dirs = [
        velorama_savedir,
        Path(outdir) / "velorama_outputs",
        Path("results"),   # Velorama default save_dir
    ]

    results = []
    for ds_name in DATASETS:
        for dynamics in ["pseudotime", "rna_velocity"]:
            result_file = _find_velorama_tsv(ds_name, dynamics, search_dirs)

            if result_file is None:
                print(f"  [SKIP] No Velorama output for {ds_name}/{dynamics}")
                continue

            print(f"  Loading {result_file}")
            # Load interaction matrix: rows=targets, cols=regulators
            interaction_df = pd.read_csv(result_file, sep='\t', index_col=0)
            # Ensure index/columns are strings (pandas may parse "0","1" as int)
            interaction_df.index = interaction_df.index.astype(str)
            interaction_df.columns = interaction_df.columns.astype(str)
            print(f"  {ds_name}/{dynamics}: {interaction_df.shape} (targets x regs)")

            # Gene names are "0" through "99"
            scores = []
            labels = []
            for tgt_idx in range(n_genes):
                tgt_name = str(tgt_idx)
                if tgt_name not in interaction_df.index:
                    continue
                for reg_idx in range(n_genes):
                    if reg_idx == tgt_idx:
                        continue
                    reg_name = str(reg_idx)
                    if reg_name not in interaction_df.columns:
                        continue
                    scores.append(abs(interaction_df.loc[tgt_name, reg_name]))
                    labels.append(int(gt_matrix[reg_idx, tgt_idx] > 0))

            scores = np.array(scores)
            labels = np.array(labels)

            if labels.sum() == 0 or labels.sum() == len(labels):
                print(f"    WARNING: degenerate labels (sum={labels.sum()})")
                continue

            auroc = roc_auc_score(labels, scores)
            auprc = average_precision_score(labels, scores)
            baseline_auprc = labels.mean()

            results.append({
                "dataset": ds_name,
                "method": f"Velorama_{dynamics}",
                "k": -1,  # N/A for Velorama
                "auroc": auroc,
                "auprc": auprc,
                "baseline_auprc": baseline_auprc,
                "n_genes": n_genes,
                "n_edges": int(labels.sum()),
                "n_pairs": len(labels),
            })

            print(f"    AUROC={auroc:.4f}, AUPRC={auprc:.4f} "
                  f"(baseline={baseline_auprc:.4f})")

    df = pd.DataFrame(results)
    df.to_csv(csvdir / "velorama_results.csv", index=False)
    print(f"\n[SAVE] velorama_results.csv ({len(df)} rows)")
    return df


# =====================================================================
# Phase 3: Compare
# =====================================================================

def compare_results(outdir):
    """Compare GeoBRIDGE and Velorama results."""
    csvdir = Path(outdir) / "csv"
    figdir = Path(outdir) / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # Load results
    gb_path = csvdir / "geobridge_results.csv"
    vr_path = csvdir / "velorama_results.csv"

    if not gb_path.exists():
        print(f"  [ERROR] {gb_path} not found. Run --phase geobridge first.")
        return

    gb_df = pd.read_csv(gb_path)

    # Velorama results may be empty or missing
    if vr_path.exists() and vr_path.stat().st_size > 10:
        vr_df = pd.read_csv(vr_path)
    else:
        print(f"  [WARN] No Velorama results yet. Showing GeoBRIDGE only.")
        vr_df = pd.DataFrame(columns=gb_df.columns)

    print(f"\n{'='*70}")
    print(f"COMPARISON: GeoBRIDGE vs Velorama")
    print(f"{'='*70}")

    # For GeoBRIDGE, pick the best k per dataset and also show k=0, k=5
    for ds_name in DATASETS:
        gb_ds = gb_df[gb_df["dataset"] == ds_name]
        vr_ds = vr_df[vr_df["dataset"] == ds_name]

        if len(gb_ds) == 0 and len(vr_ds) == 0:
            continue

        print(f"\n  --- {ds_name} ---")

        if len(gb_ds) > 0:
            best_auroc_row = gb_ds.loc[gb_ds["auroc"].idxmax()]
            best_auprc_row = gb_ds.loc[gb_ds["auprc"].idxmax()]
            baseline = gb_ds["baseline_auprc"].iloc[0]

            print(f"  GeoBRIDGE:")
            print(f"    Best AUROC: {best_auroc_row['auroc']:.4f} (k={best_auroc_row['k']:.0f})")
            print(f"    Best AUPRC: {best_auprc_row['auprc']:.4f} (k={best_auprc_row['k']:.0f})")
            for k_show in [0, 3, 5, 10]:
                row = gb_ds[gb_ds["k"] == k_show]
                if len(row) > 0:
                    print(f"    k={k_show:2d}: AUROC={row['auroc'].values[0]:.4f}, "
                          f"AUPRC={row['auprc'].values[0]:.4f}")

        if len(vr_ds) > 0:
            print(f"  Velorama:")
            for _, row in vr_ds.iterrows():
                print(f"    {row['method']}: AUROC={row['auroc']:.4f}, "
                      f"AUPRC={row['auprc']:.4f}")

        if len(gb_ds) > 0 and len(vr_ds) > 0:
            print(f"  Baseline AUPRC (random): {baseline:.4f}")

    # Summary table
    print(f"\n{'='*70}")
    print(f"SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Dataset':<12s}  {'Method':<25s}  {'AUROC':>7s}  {'AUPRC':>7s}  "
          f"{'Baseline':>8s}")
    print("  " + "-" * 65)

    for ds_name in DATASETS:
        gb_ds = gb_df[gb_df["dataset"] == ds_name]
        vr_ds = vr_df[vr_df["dataset"] == ds_name]

        if len(gb_ds) > 0:
            best = gb_ds.loc[gb_ds["auroc"].idxmax()]
            print(f"  {ds_name:<12s}  {'GeoBRIDGE (best k)':25s}  "
                  f"{best['auroc']:7.4f}  {best['auprc']:7.4f}  "
                  f"{best['baseline_auprc']:8.4f}")
            k0 = gb_ds[gb_ds["k"] == 0]
            if len(k0) > 0:
                print(f"  {'':12s}  {'GeoBRIDGE (k=0)':25s}  "
                      f"{k0['auroc'].values[0]:7.4f}  {k0['auprc'].values[0]:7.4f}")

        for _, row in vr_ds.iterrows():
            print(f"  {'':12s}  {row['method']:25s}  "
                  f"{row['auroc']:7.4f}  {row['auprc']:7.4f}")

    # Plots
    try:
        _make_comparison_plots(gb_df, vr_df, figdir)
    except Exception as e:
        print(f"[PLOT ERROR] {e}")


def _make_comparison_plots(gb_df, vr_df, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    datasets = sorted(set(gb_df["dataset"].unique()) |
                      set(vr_df["dataset"].unique()))

    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        # GeoBRIDGE curve
        gb_ds = gb_df[gb_df["dataset"] == ds].sort_values("k")
        if len(gb_ds) > 0:
            ax.plot(gb_ds["k"], gb_ds["auroc"], "o-", color="C0",
                    label="GeoBRIDGE", markersize=5)

        # Velorama horizontal lines
        vr_ds = vr_df[vr_df["dataset"] == ds]
        colors_vr = {"Velorama_pseudotime": "C3", "Velorama_rna_velocity": "C1"}
        for _, row in vr_ds.iterrows():
            ax.axhline(row["auroc"], color=colors_vr.get(row["method"], "C5"),
                       ls="--", label=row["method"])

        ax.axhline(0.5, color="gray", ls=":", alpha=0.3, label="random")
        ax.set_xlabel("k (GeoBRIDGE projection depth)")
        ax.set_ylabel("AUROC")
        ax.set_title(ds)
        ax.legend(fontsize=7)

    fig.suptitle("GeoBRIDGE vs Velorama: AUROC on Synthetic Datasets", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "geobridge_vs_velorama_auroc.png", dpi=150)
    plt.close(fig)
    print(f"[FIG] geobridge_vs_velorama_auroc.png")

    # AUPRC version
    fig, axes = plt.subplots(1, len(datasets), figsize=(5 * len(datasets), 5))
    if len(datasets) == 1:
        axes = [axes]

    for ax, ds in zip(axes, datasets):
        gb_ds = gb_df[gb_df["dataset"] == ds].sort_values("k")
        if len(gb_ds) > 0:
            ax.plot(gb_ds["k"], gb_ds["auprc"], "o-", color="C0",
                    label="GeoBRIDGE", markersize=5)
            ax.axhline(gb_ds["baseline_auprc"].iloc[0], color="gray",
                       ls=":", alpha=0.3, label="random baseline")

        vr_ds = vr_df[vr_df["dataset"] == ds]
        for _, row in vr_ds.iterrows():
            ax.axhline(row["auprc"],
                       color=colors_vr.get(row["method"], "C5"),
                       ls="--", label=row["method"])

        ax.set_xlabel("k")
        ax.set_ylabel("AUPRC")
        ax.set_title(ds)
        ax.legend(fontsize=7)

    fig.suptitle("GeoBRIDGE vs Velorama: AUPRC on Synthetic Datasets", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "geobridge_vs_velorama_auprc.png", dpi=150)
    plt.close(fig)
    print(f"[FIG] geobridge_vs_velorama_auprc.png")


# =====================================================================
# Main
# =====================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True,
                    choices=["geobridge", "velorama", "compare", "all"],
                    help="Which phase to run")
    ap.add_argument("--data-dir", default="results/bm_velorama/velorama_data")
    ap.add_argument("--outdir", default="results/bm_velorama")
    ap.add_argument("--device", default="cuda",
                    help="Device for Velorama (cuda or cpu)")
    args = ap.parse_args()

    if args.phase in ("geobridge", "all"):
        run_geobridge(args.data_dir, args.outdir)

    if args.phase in ("velorama", "all"):
        run_velorama(args.data_dir, args.outdir, device=args.device)

    if args.phase in ("compare", "all"):
        compare_results(args.outdir)


if __name__ == "__main__":
    main()
