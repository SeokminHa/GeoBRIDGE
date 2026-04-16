"""
Full Velorama diagnostic: resolve AUPRC discrepancy with paper.

Runs sequentially on charybdis:
  1) Check h5ad annotations (is_reg, is_target)
  2) GENIE3 sanity check (determines paper's evaluation method)
  3) Velorama with narrow lambda grid (0.01-0.2, 19 values)
  4) Velorama with paper-like lambda grid for comparison
  5) Final comparison table

Usage:
  cd /data/cb/scratch/seokmin/geobridge
  source venv_gpu/bin/activate
  python -u benchmark/velorama_full_diagnosis.py 2>&1 | tee velorama_diagnosis_log.txt
"""

import sys
import time
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import average_precision_score, roc_auc_score

N_GENES = 100
DATA_DIR = Path("results/bm_velorama/velorama_data")
GT_PATH = DATA_DIR / "gt_GRN.csv"


# =====================================================================
# Helpers
# =====================================================================

def load_gt():
    edges = pd.read_csv(GT_PATH, header=None, names=["reg", "tgt"])
    gt = np.zeros((N_GENES, N_GENES))
    for _, r in edges.iterrows():
        gt[int(r["reg"]), int(r["tgt"])] = 1
    master_regs = np.where(gt.sum(axis=1) > 0)[0]
    return gt, master_regs


def eval_score_matrix(S, gt, master_regs):
    """Compute all three AUPRC variants."""
    N = N_GENES

    # 1) Global AUPRC (all 9900 pairs)
    sc_g, lb_g = [], []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            sc_g.append(abs(S[i, j]))
            lb_g.append(int(gt[i, j] > 0))
    sc_g, lb_g = np.array(sc_g), np.array(lb_g)
    g_auroc = roc_auc_score(lb_g, sc_g)
    g_auprc = average_precision_score(lb_g, sc_g)

    # 2) Per-target averaged, all 100 candidate TFs
    pt_aurocs, pt_auprcs = [], []
    for j in range(N):
        sj, lj = [], []
        for i in range(N):
            if i == j:
                continue
            sj.append(abs(S[i, j]))
            lj.append(int(gt[i, j] > 0))
        lj, sj = np.array(lj), np.array(sj)
        if lj.sum() == 0 or lj.sum() == len(lj):
            continue
        pt_aurocs.append(roc_auc_score(lj, sj))
        pt_auprcs.append(average_precision_score(lj, sj))

    # 3) Per-target averaged, 10 master regs only
    mr_aurocs, mr_auprcs = [], []
    for j in range(N):
        sj, lj = [], []
        for i in master_regs:
            if i == j:
                continue
            sj.append(abs(S[i, j]))
            lj.append(int(gt[i, j] > 0))
        lj, sj = np.array(lj), np.array(sj)
        if lj.sum() == 0 or lj.sum() == len(lj):
            continue
        mr_aurocs.append(roc_auc_score(lj, sj))
        mr_auprcs.append(average_precision_score(lj, sj))

    return {
        "global_auroc": g_auroc,
        "global_auprc": g_auprc,
        "pt100_auroc": np.mean(pt_aurocs) if pt_aurocs else np.nan,
        "pt100_auprc": np.mean(pt_auprcs) if pt_auprcs else np.nan,
        "mr_auroc": np.mean(mr_aurocs) if mr_aurocs else np.nan,
        "mr_auprc": np.mean(mr_auprcs) if mr_auprcs else np.nan,
    }


def load_velorama_tsv(tsv_path):
    """Load Velorama TSV into (N, N) score matrix."""
    df = pd.read_csv(tsv_path, sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    S = np.zeros((N_GENES, N_GENES))
    for j in range(N_GENES):
        for i in range(N_GENES):
            if str(j) in df.index and str(i) in df.columns:
                S[i, j] = df.loc[str(j), str(i)]
    return S


# =====================================================================
# Step 1: Check h5ad annotations
# =====================================================================

def step1_check_annotations():
    print("\n" + "=" * 70)
    print("STEP 1: Check h5ad annotations (is_reg, is_target)")
    print("=" * 70)

    for ds in ["dataset_A", "dataset_B", "dataset_C", "dataset_D"]:
        fpath = DATA_DIR / f"{ds}.h5ad"
        if not fpath.exists():
            print(f"  {ds}: NOT FOUND")
            continue
        adata = sc.read_h5ad(str(fpath))
        print(f"\n  {ds}: {adata.shape[0]} cells x {adata.shape[1]} genes")

        # Check var columns
        print(f"    var columns: {list(adata.var.columns)}")
        for key in ["is_reg", "is_target", "is_tf", "regulator", "target"]:
            if key in adata.var.columns:
                vals = adata.var[key]
                n_true = vals.sum() if vals.dtype == bool else (vals > 0).sum()
                print(f"    >>> {key}: {n_true} genes flagged")

        # Check uns
        uns_keys = list(adata.uns.keys())
        print(f"    uns keys: {uns_keys}")

        # Check obs columns
        obs_cols = list(adata.obs.columns)
        print(f"    obs columns: {obs_cols}")

        # Check layers
        layer_keys = list(adata.layers.keys())
        print(f"    layers: {layer_keys}")

        # Check varp
        varp_keys = list(adata.varp.keys()) if hasattr(adata, "varp") else []
        print(f"    varp: {varp_keys}")


# =====================================================================
# Step 2: GENIE3 sanity check
# =====================================================================

def step2_genie3():
    print("\n" + "=" * 70)
    print("STEP 2: GENIE3 sanity check (determines paper's eval method)")
    print("=" * 70)

    gt, master_regs = load_gt()
    print(f"  GT: {int(gt.sum())} edges, {len(master_regs)} master regs")

    adata = sc.read_h5ad(str(DATA_DIR / "dataset_D.h5ad"))
    X = np.asarray(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        dtype=float,
    )

    print(f"  Running GENIE3 (RF, 500 trees) on dataset_D ({X.shape})...")
    t0 = time.perf_counter()
    S = np.zeros((N_GENES, N_GENES))
    for j in range(N_GENES):
        mask = np.ones(N_GENES, dtype=bool)
        mask[j] = False
        rf = RandomForestRegressor(
            n_estimators=500, max_features="sqrt", random_state=42, n_jobs=-1
        )
        rf.fit(X[:, mask], X[:, j])
        feat_genes = np.where(mask)[0]
        for k, g in enumerate(feat_genes):
            S[g, j] = rf.feature_importances_[k]
        if (j + 1) % 25 == 0:
            print(f"    {j+1}/{N_GENES} targets done")

    elapsed = time.perf_counter() - t0
    print(f"  GENIE3 done in {elapsed:.1f}s")

    metrics = eval_score_matrix(S, gt, master_regs)

    print(f"\n  GENIE3 dataset_D results:")
    print(f"    Global AUROC:       {metrics['global_auroc']:.4f}")
    print(f"    Global AUPRC:       {metrics['global_auprc']:.4f}")
    print(f"    Per-target(100) AUPRC: {metrics['pt100_auprc']:.4f}")
    print(f"    Per-target(MR) AUPRC:  {metrics['mr_auprc']:.4f}")
    print(f"\n  Paper reports GENIE3 AUPRC = 0.129 for dataset D")
    print(f"    Global AUPRC match:       {abs(metrics['global_auprc'] - 0.129) < 0.03}")
    print(f"    Per-target(100) match:    {abs(metrics['pt100_auprc'] - 0.129) < 0.03}")
    print(f"    Per-target(MR) match:     {abs(metrics['mr_auprc'] - 0.129) < 0.03}")

    # Which is closest?
    diffs = {
        "global": abs(metrics["global_auprc"] - 0.129),
        "pt100": abs(metrics["pt100_auprc"] - 0.129),
        "mr": abs(metrics["mr_auprc"] - 0.129),
    }
    closest = min(diffs, key=diffs.get)
    print(f"\n  >>> CLOSEST MATCH: {closest} (diff={diffs[closest]:.4f})")
    print(f"  >>> This suggests the paper uses '{closest}' evaluation")

    return metrics


# =====================================================================
# Step 3: Velorama with narrow lambda grid
# =====================================================================

def step3_velorama_narrow_lambda():
    print("\n" + "=" * 70)
    print("STEP 3: Velorama with narrow lambda grid (0.005-0.2)")
    print("=" * 70)
    print("  Rationale: original grid 0.01-10 has most lambdas in the")
    print("  'dead zone' (variable usage=0%). Narrower grid concentrates")
    print("  on the informative range.")

    outdir = Path("/tmp/velo_narrow")
    outdir.mkdir(parents=True, exist_ok=True)

    ds = "dataset_D"
    cmd = [
        "velorama",
        "-n", f"{ds}_narrow",
        "-ds", ds,
        "-dyn", "rna_velocity",
        "-dev", "cuda",
        "-l", "5",
        "-hd", "32",
        "-mi", "1000",
        "-rd", str(DATA_DIR),
        "-sd", str(outdir),
        # Narrow lambda range: 10^(-2.3) to 10^(-0.7) ≈ 0.005 to 0.2
        "-ls", "-2.3",
        "-le", "-0.7",
        "-nl", "19",
    ]
    print(f"  Running: {' '.join(cmd)}")
    t0 = time.perf_counter()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.perf_counter() - t0
        print(f"  Completed in {elapsed:.1f}s, returncode={result.returncode}")

        # Show lambda-specific training info
        for line in result.stdout.split("\n"):
            if "Variable usage" in line or "TERMINATED" in line:
                print(f"    {line.strip()}")

        if result.returncode != 0:
            print(f"  STDERR (last 300): {result.stderr[-300:]}")
    except subprocess.TimeoutExpired:
        print("  TIMEOUT after 3600s")
        return None

    # Evaluate
    tsv_path = outdir / f"{ds}_narrow.rna_velocity.velorama.interactions.tsv"
    if not tsv_path.exists():
        print(f"  TSV not found: {tsv_path}")
        # Search for it
        for f in outdir.glob("*.interactions.tsv"):
            print(f"  Found: {f}")
            tsv_path = f
            break

    if tsv_path.exists():
        gt, master_regs = load_gt()
        S = load_velorama_tsv(tsv_path)
        metrics = eval_score_matrix(S, gt, master_regs)
        print(f"\n  Narrow-lambda Velorama results:")
        print(f"    Global AUROC:       {metrics['global_auroc']:.4f}")
        print(f"    Global AUPRC:       {metrics['global_auprc']:.4f}")
        print(f"    Per-target(MR) AUPRC: {metrics['mr_auprc']:.4f}")

        # Compare with default lambda
        print(f"\n  vs Default lambda (0.01-10):")
        print(f"    Global AUPRC: 0.0345 -> {metrics['global_auprc']:.4f}")
        print(f"    AUPRC(mr):    0.4224 -> {metrics['mr_auprc']:.4f}")
        return metrics
    return None


# =====================================================================
# Step 4: Velorama with is_reg annotation (if applicable)
# =====================================================================

def step4_velorama_with_isreg():
    """If h5ad files DON'T have is_reg, create a version that does."""
    print("\n" + "=" * 70)
    print("STEP 4: Velorama with is_reg=master_regulators annotation")
    print("=" * 70)

    gt, master_regs = load_gt()

    # Check if dataset_D already has is_reg
    adata = sc.read_h5ad(str(DATA_DIR / "dataset_D.h5ad"))
    if "is_reg" in adata.var.columns:
        n_reg = adata.var["is_reg"].sum()
        print(f"  dataset_D already has is_reg: {n_reg} regulators")
        if n_reg == 10:
            print("  Already set to master regulators, skipping.")
            return None

    # Create annotated version
    annotated_dir = Path("/tmp/velo_annotated")
    annotated_dir.mkdir(parents=True, exist_ok=True)

    ds = "dataset_D"
    adata = sc.read_h5ad(str(DATA_DIR / f"{ds}.h5ad"))

    # Set is_reg: only master regulators
    adata.var["is_reg"] = False
    for mr in master_regs:
        gene_name = str(mr)
        if gene_name in adata.var_names:
            adata.var.loc[gene_name, "is_reg"] = True
        elif mr < len(adata.var_names):
            adata.var.iloc[mr, adata.var.columns.get_loc("is_reg")] = True

    # Set is_target: all genes
    adata.var["is_target"] = True

    n_reg = adata.var["is_reg"].sum()
    n_tgt = adata.var["is_target"].sum()
    print(f"  Annotated {ds}: is_reg={n_reg}, is_target={n_tgt}")

    annotated_path = annotated_dir / f"{ds}.h5ad"
    adata.write_h5ad(str(annotated_path))

    # Run Velorama on annotated version
    outdir_ann = Path("/tmp/velo_annotated_out")
    outdir_ann.mkdir(parents=True, exist_ok=True)

    cmd = [
        "velorama",
        "-n", f"{ds}_annotated",
        "-ds", ds,
        "-dyn", "rna_velocity",
        "-dev", "cuda",
        "-l", "5",
        "-hd", "32",
        "-mi", "1000",
        "-rd", str(annotated_dir),
        "-sd", str(outdir_ann),
    ]
    print(f"  Running: {' '.join(cmd)}")
    t0 = time.perf_counter()

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        elapsed = time.perf_counter() - t0
        print(f"  Completed in {elapsed:.1f}s, returncode={result.returncode}")
        if result.returncode != 0:
            print(f"  STDERR (last 500): {result.stderr[-500:]}")
            print(f"  STDOUT (last 500): {result.stdout[-500:]}")
    except subprocess.TimeoutExpired:
        print("  TIMEOUT")
        return None

    # Evaluate
    tsv_path = outdir_ann / f"{ds}_annotated.rna_velocity.velorama.interactions.tsv"
    if not tsv_path.exists():
        for f in outdir_ann.glob("*.interactions.tsv"):
            tsv_path = f
            break

    if tsv_path.exists():
        S = load_velorama_tsv(tsv_path)
        metrics = eval_score_matrix(S, gt, master_regs)
        print(f"\n  Annotated Velorama (is_reg=master_regs) results:")
        print(f"    Global AUROC:       {metrics['global_auroc']:.4f}")
        print(f"    Global AUPRC:       {metrics['global_auprc']:.4f}")
        print(f"    Per-target(MR) AUPRC: {metrics['mr_auprc']:.4f}")
        return metrics
    return None


# =====================================================================
# Step 5: Final summary
# =====================================================================

def step5_summary(genie3_metrics, narrow_metrics, annotated_metrics):
    print("\n" + "=" * 70)
    print("FINAL SUMMARY: EVALUATION METHOD DIAGNOSIS")
    print("=" * 70)

    gt, master_regs = load_gt()

    # Load existing results
    # Default Velorama (1000 iter, lambda 0.01-10)
    default_tsv = Path("/tmp/velo_bench/output/dataset_D.rna_velocity.velorama.interactions.tsv")
    if default_tsv.exists():
        S_default = load_velorama_tsv(default_tsv)
        default_metrics = eval_score_matrix(S_default, gt, master_regs)
    else:
        default_metrics = {
            "global_auroc": 0.7863, "global_auprc": 0.0345, "mr_auprc": 0.4224,
            "pt100_auprc": 0.0930,
        }

    print(f"\n  Paper reference (dataset_D):")
    print(f"    Velorama AUPRC:  0.656")
    print(f"    GENIE3 AUPRC:    0.129")

    print(f"\n  {'Method':<40s} {'Global':>8s} {'PT(100)':>8s} {'PT(MR)':>8s}")
    print(f"  {'-'*68}")

    print(f"  {'Velorama default (lam 0.01-10)':<40s} "
          f"{default_metrics['global_auprc']:8.4f} "
          f"{default_metrics.get('pt100_auprc', float('nan')):8.4f} "
          f"{default_metrics['mr_auprc']:8.4f}")

    if narrow_metrics:
        print(f"  {'Velorama narrow (lam 0.005-0.2)':<40s} "
              f"{narrow_metrics['global_auprc']:8.4f} "
              f"{narrow_metrics.get('pt100_auprc', float('nan')):8.4f} "
              f"{narrow_metrics['mr_auprc']:8.4f}")

    if annotated_metrics:
        print(f"  {'Velorama annotated (is_reg=10)':<40s} "
              f"{annotated_metrics['global_auprc']:8.4f} "
              f"{annotated_metrics.get('pt100_auprc', float('nan')):8.4f} "
              f"{annotated_metrics['mr_auprc']:8.4f}")

    if genie3_metrics:
        print(f"  {'GENIE3 (RF, 500 trees)':<40s} "
              f"{genie3_metrics['global_auprc']:8.4f} "
              f"{genie3_metrics.get('pt100_auprc', float('nan')):8.4f} "
              f"{genie3_metrics['mr_auprc']:8.4f}")

    print(f"\n  {'Paper reference':<40s} {'???':>8s} {'???':>8s} {'???':>8s}")
    print(f"  {'  Velorama':<40s} {'':>8s} {'':>8s} {'0.656':>8s}")
    print(f"  {'  GENIE3':<40s} {'':>8s} {'':>8s} {'0.129':>8s}")

    # Diagnosis
    if genie3_metrics:
        diffs = {
            "global": abs(genie3_metrics["global_auprc"] - 0.129),
            "pt100": abs(genie3_metrics["pt100_auprc"] - 0.129),
            "mr": abs(genie3_metrics["mr_auprc"] - 0.129),
        }
        closest = min(diffs, key=diffs.get)
        print(f"\n  DIAGNOSIS: GENIE3 closest to 0.129 under '{closest}' metric")
        print(f"    (diff = {diffs[closest]:.4f})")

        if closest == "mr":
            print(f"  ==> Paper uses MASTER-REG RESTRICTED per-target AUPRC")
            print(f"  ==> Our Velorama AUPRC(mr) = {default_metrics['mr_auprc']:.4f} "
                  f"vs paper 0.656")
            print(f"  ==> Gap of {0.656 - default_metrics['mr_auprc']:.3f} likely due to "
                  f"is_reg annotation and/or code version")
        elif closest == "pt100":
            print(f"  ==> Paper uses PER-TARGET (all 100 TFs) AUPRC")
        else:
            print(f"  ==> Paper uses GLOBAL pooled AUPRC")


# =====================================================================
# Main
# =====================================================================

def main():
    print("=" * 70)
    print("VELORAMA FULL DIAGNOSIS")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    # Step 1: Check annotations
    step1_check_annotations()

    # Step 2: GENIE3
    genie3_metrics = step2_genie3()

    # Step 3: Narrow lambda
    narrow_metrics = step3_velorama_narrow_lambda()

    # Step 4: Annotated is_reg
    annotated_metrics = step4_velorama_with_isreg()

    # Step 5: Summary
    step5_summary(genie3_metrics, narrow_metrics, annotated_metrics)

    print(f"\nDone at {time.strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
