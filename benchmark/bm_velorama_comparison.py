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
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

DATASETS = ["dataset_A", "dataset_B", "dataset_C", "dataset_D"]
KEY_K_VALUES = [0, 1, 2, 3, 5, 7, 10, 15, 20, 25]


# =====================================================================
# Ground truth extraction
# =====================================================================

def extract_ground_truth(adata):
    """
    Extract ground truth regulatory edges from a Velorama synthetic h5ad.

    Tries multiple possible locations. Returns:
      gt_matrix: (n_reg, n_target) binary or weighted adjacency
      reg_genes: list of regulator gene names
      target_genes: list of target gene names

    Raises RuntimeError if ground truth cannot be found.
    """
    import scipy.sparse as sp

    reg_mask = adata.var['is_reg'].values if 'is_reg' in adata.var else None
    tgt_mask = adata.var['is_target'].values if 'is_target' in adata.var else None

    if reg_mask is None or tgt_mask is None:
        raise RuntimeError("Missing is_reg / is_target in adata.var")

    reg_genes = list(adata.var_names[reg_mask])
    target_genes = list(adata.var_names[tgt_mask])
    all_genes = list(adata.var_names)

    # Strategy 1: check varp for gene-gene matrix
    for key in ['ground_truth', 'true_network', 'GRN', 'adjacency',
                'W', 'regulatory_network']:
        if key in adata.varp:
            mat = adata.varp[key]
            if sp.issparse(mat):
                mat = mat.toarray()
            mat = np.asarray(mat)
            # Extract reg x target submatrix
            reg_idx = [all_genes.index(g) for g in reg_genes]
            tgt_idx = [all_genes.index(g) for g in target_genes]
            gt = mat[np.ix_(reg_idx, tgt_idx)]
            print(f"  [GT] Found in varp['{key}']: {gt.shape}, "
                  f"{np.count_nonzero(gt)} nonzero edges")
            return gt, reg_genes, target_genes

    # Strategy 2: check uns for ground truth matrix
    for key in ['ground_truth', 'true_network', 'GRN', 'W',
                'regulatory_network', 'gt_adj', 'true_adj',
                'ground_truth_network']:
        if key in adata.uns:
            v = adata.uns[key]
            if isinstance(v, np.ndarray) and v.ndim == 2:
                print(f"  [GT] Found in uns['{key}']: shape={v.shape}, "
                      f"{np.count_nonzero(v)} nonzero edges")
                # Determine if this is (reg x target) or (all x all)
                if v.shape == (len(reg_genes), len(target_genes)):
                    return v, reg_genes, target_genes
                elif v.shape[0] == v.shape[1] == len(all_genes):
                    reg_idx = [all_genes.index(g) for g in reg_genes]
                    tgt_idx = [all_genes.index(g) for g in target_genes]
                    gt = v[np.ix_(reg_idx, tgt_idx)]
                    return gt, reg_genes, target_genes
                else:
                    print(f"    WARNING: shape {v.shape} doesn't match "
                          f"reg={len(reg_genes)}, tgt={len(target_genes)}, "
                          f"all={len(all_genes)}")
                    return v, reg_genes, target_genes

    # Strategy 3: check uns for dict with edges
    for key in ['ground_truth', 'edges', 'true_edges', 'regulatory_edges']:
        if key in adata.uns and isinstance(adata.uns[key], dict):
            d = adata.uns[key]
            if 'source' in d and 'target' in d:
                sources = d['source']
                targets = d['target']
                print(f"  [GT] Found edge list in uns['{key}']: "
                      f"{len(sources)} edges")
                gt = np.zeros((len(reg_genes), len(target_genes)))
                reg_to_idx = {g: i for i, g in enumerate(reg_genes)}
                tgt_to_idx = {g: i for i, g in enumerate(target_genes)}
                for s, t in zip(sources, targets):
                    if s in reg_to_idx and t in tgt_to_idx:
                        gt[reg_to_idx[s], tgt_to_idx[t]] = 1
                return gt, reg_genes, target_genes

    # Strategy 4: check for separate columns in var
    gt_cols = [c for c in adata.var.columns
               if 'target_of' in c.lower() or 'regulated_by' in c.lower()
               or 'ground' in c.lower()]
    if gt_cols:
        print(f"  [GT] Found potential GT columns in var: {gt_cols}")

    # Strategy 5: check all uns keys for any 2D array
    for key in adata.uns:
        v = adata.uns[key]
        if isinstance(v, np.ndarray) and v.ndim == 2:
            nr, nc = v.shape
            if (nr == len(reg_genes) or nr == len(all_genes)) and \
               (nc == len(target_genes) or nc == len(all_genes)):
                print(f"  [GT] Possible GT in uns['{key}']: shape={v.shape}")

    raise RuntimeError(
        f"Could not find ground truth. "
        f"uns keys: {list(adata.uns.keys())}, "
        f"varp keys: {list(adata.varp.keys()) if hasattr(adata, 'varp') else []}, "
        f"var columns: {list(adata.var.columns)}"
    )


# =====================================================================
# Phase 1: GeoBRIDGE
# =====================================================================

def run_geobridge(data_dir, outdir):
    """Run GeoBRIDGE on all Velorama synthetic datasets."""
    import scanpy as sc
    from geobridge_model import GeoBridgeModel

    csvdir = Path(outdir) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)

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
        print(f"  {adata.shape[0]} cells x {adata.shape[1]} genes")

        # Extract ground truth
        try:
            gt_matrix, reg_genes, target_genes = extract_ground_truth(adata)
        except RuntimeError as e:
            print(f"  [ERROR] {e}")
            continue

        all_genes = list(adata.var_names)
        gene_to_idx = {g: i for i, g in enumerate(all_genes)}

        # Get expression and velocity
        X = np.asarray(adata.X.toarray() if hasattr(adata.X, 'toarray')
                       else adata.X, dtype=float)

        # Check for velocity in layers
        V = None
        for vel_key in ['velocity', 'velocities', 'Ms_velocity']:
            if vel_key in adata.layers:
                V = np.asarray(adata.layers[vel_key], dtype=float)
                print(f"  Using velocity from layers['{vel_key}']")
                break

        if V is None:
            # Try computing velocity if spliced/unspliced available
            if 'spliced' in adata.layers and 'unspliced' in adata.layers:
                import scvelo as scv
                print(f"  Computing velocity from spliced/unspliced...")
                scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
                scv.tl.velocity(adata, mode="stochastic")
                V = np.asarray(adata.layers['velocity'], dtype=float)
            else:
                # Last resort: use smoothed expression as proxy
                if 'Ms' in adata.layers:
                    X = np.asarray(adata.layers['Ms'], dtype=float)
                print(f"  WARNING: No velocity found. Available layers: "
                      f"{list(adata.layers.keys())}")
                print(f"  Will try to compute from data...")
                # If no velocity at all, skip
                if V is None:
                    print(f"  [ERROR] Cannot compute velocity for {ds_name}")
                    continue

        V = np.nan_to_num(V, nan=0.0)

        # Pseudotime
        pt = None
        for pt_key in ['pseudotime', 'latent_time', 'dpt_pseudotime',
                        'palantir_pseudotime']:
            if pt_key in adata.obs:
                pt = np.asarray(adata.obs[pt_key], dtype=float)
                print(f"  Using pseudotime from obs['{pt_key}']")
                break

        if pt is None:
            # Compute DPT
            print(f"  Computing pseudotime via DPT...")
            import scanpy as sc_compute
            if 'X_pca' not in adata.obsm:
                sc_compute.pp.pca(adata, n_comps=min(30, adata.shape[1] - 1))
            if 'neighbors' not in adata.uns:
                sc_compute.pp.neighbors(adata, n_pcs=min(30, adata.obsm['X_pca'].shape[1]))
            sc_compute.tl.diffmap(adata)
            if 'iroot' in adata.uns:
                adata.uns['iroot'] = int(adata.uns['iroot'])
            else:
                adata.uns['iroot'] = 0
            sc_compute.tl.dpt(adata)
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

        G = len(all_genes)
        reg_idxs = [gene_to_idx[g] for g in reg_genes]
        tgt_idxs = [gene_to_idx[g] for g in target_genes]

        for k in KEY_K_VALUES:
            if k >= len(pre["S"]):
                continue

            S_k = GeoBridgeModel.scores_from_streaming_stats(
                pre, k=k, project_y=True, normalize_score=True, eps=1e-12,
            )

            # Build score vector and label vector for all (reg, target) pairs
            scores = []
            labels = []
            for ri, reg_i in enumerate(reg_idxs):
                for ti, tgt_i in enumerate(tgt_idxs):
                    if reg_i == tgt_i:
                        continue
                    scores.append(abs(S_k[reg_i, tgt_i]))
                    labels.append(1 if abs(gt_matrix[ri, ti]) > 0 else 0)

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
                "n_reg": len(reg_genes),
                "n_target": len(target_genes),
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

def run_velorama(data_dir, outdir, device="cuda"):
    """Run Velorama on all datasets via CLI."""
    csvdir = Path(outdir) / "csv"
    csvdir.mkdir(parents=True, exist_ok=True)
    velorama_outdir = Path(outdir) / "velorama_outputs"
    velorama_outdir.mkdir(parents=True, exist_ok=True)

    for ds_name in DATASETS:
        fpath = Path(data_dir) / f"{ds_name}.h5ad"
        if not fpath.exists():
            print(f"  [SKIP] {ds_name} not found")
            continue

        print(f"\n{'='*70}")
        print(f"Velorama on {ds_name}")
        print(f"{'='*70}")

        # Copy h5ad to velorama output dir (Velorama expects it in root_dir)
        import shutil
        dst = velorama_outdir / f"{ds_name}.h5ad"
        if not dst.exists():
            shutil.copy2(fpath, dst)

        # Run Velorama CLI
        # Try both dynamics modes
        for dynamics in ["pseudotime", "rna_velocity"]:
            result_file = (velorama_outdir /
                           f"{ds_name}.{dynamics}.velorama.interactions.tsv")
            if result_file.exists():
                print(f"  [SKIP] {ds_name}/{dynamics} already computed")
                continue

            cmd = [
                "velorama",
                "-ds", ds_name,
                "-dyn", dynamics,
                "-dev", device,
                "-l", "5",
                "-hd", "32",
                "-rd", str(velorama_outdir),
            ]
            print(f"  Running: {' '.join(cmd)}")
            t0 = time.perf_counter()

            try:
                result = subprocess.run(
                    cmd, capture_output=True, text=True, timeout=3600,
                )
                elapsed = time.perf_counter() - t0
                print(f"  Completed in {elapsed:.1f}s")
                if result.returncode != 0:
                    print(f"  STDERR: {result.stderr[:500]}")
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT after 3600s")
            except FileNotFoundError:
                print(f"  ERROR: 'velorama' command not found. "
                      f"Install via: pip install velorama")
                return

    # Parse Velorama results
    print(f"\n{'='*70}")
    print(f"PARSING VELORAMA RESULTS")
    print(f"{'='*70}")

    import scanpy as sc
    from sklearn.metrics import average_precision_score, roc_auc_score

    results = []
    for ds_name in DATASETS:
        fpath = Path(data_dir) / f"{ds_name}.h5ad"
        if not fpath.exists():
            continue

        adata = sc.read_h5ad(str(fpath))
        try:
            gt_matrix, reg_genes, target_genes = extract_ground_truth(adata)
        except RuntimeError:
            continue

        for dynamics in ["pseudotime", "rna_velocity"]:
            result_file = (velorama_outdir /
                           f"{ds_name}.{dynamics}.velorama.interactions.tsv")
            if not result_file.exists():
                print(f"  [SKIP] {result_file.name} not found")
                continue

            # Load interaction matrix (rows=targets, cols=regulators)
            interaction_df = pd.read_csv(result_file, sep='\t', index_col=0)
            print(f"  {ds_name}/{dynamics}: "
                  f"{interaction_df.shape} (targets x regs)")

            # Build score and label vectors
            scores = []
            labels = []
            reg_to_ri = {g: i for i, g in enumerate(reg_genes)}
            tgt_to_ti = {g: i for i, g in enumerate(target_genes)}

            for tgt_gene in interaction_df.index:
                if tgt_gene not in tgt_to_ti:
                    continue
                ti = tgt_to_ti[tgt_gene]
                for reg_gene in interaction_df.columns:
                    if reg_gene not in reg_to_ri:
                        continue
                    ri = reg_to_ri[reg_gene]
                    if reg_gene == tgt_gene:
                        continue
                    scores.append(abs(interaction_df.loc[tgt_gene, reg_gene]))
                    labels.append(1 if abs(gt_matrix[ri, ti]) > 0 else 0)

            scores = np.array(scores)
            labels = np.array(labels)

            if labels.sum() == 0 or labels.sum() == len(labels):
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
                "n_reg": len(reg_genes),
                "n_target": len(target_genes),
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
    if not vr_path.exists():
        print(f"  [ERROR] {vr_path} not found. Run --phase velorama first.")
        return

    gb_df = pd.read_csv(gb_path)
    vr_df = pd.read_csv(vr_path)

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
