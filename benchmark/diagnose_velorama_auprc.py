"""
Diagnose Velorama AUPRC discrepancy with published paper (Figure 3C).

Paper reports AUPRC ~0.4-0.6 for velocity+probabilistic.
Our runs give global AUPRC ~0.03-0.08.

Hypothesis: the paper uses **per-target averaged AUPRC** (BEELINE/DREAM standard),
while we compute **global AUPRC** (all 9900 pairs pooled).

This script evaluates a Velorama TSV using BOTH metrics to confirm.

Usage (on charybdis):
  python -m benchmark.diagnose_velorama_auprc \
    --tsv /tmp/velo_test/output/test_D.rna_velocity.velorama.interactions.tsv \
    --gt-csv results/bm_velorama/velorama_data/gt_GRN.csv

  # Or evaluate all TSVs in a directory:
  python -m benchmark.diagnose_velorama_auprc \
    --tsv-dir results/bm_velorama/velorama_results \
    --gt-csv results/bm_velorama/velorama_data/gt_GRN.csv
"""

import sys
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

N_GENES = 100
SERGIO_GT_URL = (
    "https://raw.githubusercontent.com/PayamDiba/SERGIO/master/"
    "data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/gt_GRN.csv"
)


def load_ground_truth(gt_path):
    """Load SERGIO ground truth into (100,100) binary adjacency matrix."""
    edges = pd.read_csv(gt_path, header=None, names=["reg", "tgt"])
    gt = np.zeros((N_GENES, N_GENES), dtype=float)
    for _, row in edges.iterrows():
        gt[int(row["reg"]), int(row["tgt"])] = 1
    print(f"GT: {int(gt.sum())} edges, {len(edges['reg'].unique())} regulators, "
          f"{len(edges['tgt'].unique())} targets")
    return gt


def ensure_gt(gt_csv):
    """Download GT if not present."""
    gt_path = Path(gt_csv)
    if not gt_path.exists():
        gt_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading GT from {SERGIO_GT_URL}")
        urllib.request.urlretrieve(SERGIO_GT_URL, gt_path)
    return gt_path


def global_metrics(scores_matrix, gt_matrix):
    """Global AUPRC/AUROC: pool all (i,j) pairs with i!=j."""
    scores, labels = [], []
    for i in range(N_GENES):
        for j in range(N_GENES):
            if i == j:
                continue
            scores.append(abs(scores_matrix[i, j]))
            labels.append(int(gt_matrix[i, j] > 0))
    scores = np.array(scores)
    labels = np.array(labels)
    auroc = roc_auc_score(labels, scores)
    auprc = average_precision_score(labels, scores)
    return auroc, auprc, labels.mean()


def per_target_metrics(scores_matrix, gt_matrix):
    """
    Per-target averaged AUPRC/AUROC (BEELINE/DREAM standard).

    For each target gene j:
      - Collect scores from all candidate regulators i (i != j)
      - Compute AUPRC for that target
      - Average across targets that have at least one true regulator

    This is what most GRN inference papers report.
    """
    aurocs, auprcs = [], []
    targets_with_edges = []
    targets_skipped = 0

    for j in range(N_GENES):
        scores_j = []
        labels_j = []
        for i in range(N_GENES):
            if i == j:
                continue
            scores_j.append(abs(scores_matrix[i, j]))
            labels_j.append(int(gt_matrix[i, j] > 0))
        labels_j = np.array(labels_j)
        scores_j = np.array(scores_j)

        n_pos = labels_j.sum()
        if n_pos == 0 or n_pos == len(labels_j):
            targets_skipped += 1
            continue

        aurocs.append(roc_auc_score(labels_j, scores_j))
        auprcs.append(average_precision_score(labels_j, scores_j))
        targets_with_edges.append(j)

    mean_auroc = np.mean(aurocs)
    mean_auprc = np.mean(auprcs)
    median_auprc = np.median(auprcs)

    return {
        "mean_auroc": mean_auroc,
        "mean_auprc": mean_auprc,
        "median_auprc": median_auprc,
        "n_targets_evaluated": len(targets_with_edges),
        "n_targets_skipped": targets_skipped,
        "per_target_aurocs": aurocs,
        "per_target_auprcs": auprcs,
    }


def per_regulator_metrics(scores_matrix, gt_matrix):
    """
    Per-regulator averaged AUPRC (alternative: evaluate per source gene).

    For each regulator i:
      - Collect scores for all candidate targets j (j != i)
      - Compute AUPRC for that regulator
    """
    aurocs, auprcs = [], []
    regs_skipped = 0

    for i in range(N_GENES):
        scores_i = []
        labels_i = []
        for j in range(N_GENES):
            if i == j:
                continue
            scores_i.append(abs(scores_matrix[i, j]))
            labels_i.append(int(gt_matrix[i, j] > 0))
        labels_i = np.array(labels_i)
        scores_i = np.array(scores_i)

        n_pos = labels_i.sum()
        if n_pos == 0 or n_pos == len(labels_i):
            regs_skipped += 1
            continue

        aurocs.append(roc_auc_score(labels_i, scores_i))
        auprcs.append(average_precision_score(labels_i, scores_i))

    return {
        "mean_auroc": np.mean(aurocs),
        "mean_auprc": np.mean(auprcs),
        "median_auprc": np.median(auprcs),
        "n_regs_evaluated": len(aurocs),
        "n_regs_skipped": regs_skipped,
    }


def master_reg_per_target_metrics(scores_matrix, gt_matrix):
    """
    Per-target AUPRC restricted to the 10 SERGIO master regulators as candidates.

    SERGIO GRN has exactly 10 master regulators (genes that regulate others).
    If the paper evaluates only among these 10 as candidate TFs, the positive
    rate jumps from ~1.4% to ~15% per target, explaining AUPRC ~0.4-0.6.

    The 10 master regulators are identified from gt_matrix: genes with
    outgoing edges (rows with at least one 1).
    """
    master_regs = np.where(gt_matrix.sum(axis=1) > 0)[0]
    print(f"  Master regulators ({len(master_regs)}): {list(master_regs)}")

    aurocs, auprcs = [], []
    for j in range(N_GENES):
        scores_j, labels_j = [], []
        for i in master_regs:
            if i == j:
                continue
            scores_j.append(scores_matrix[i, j])
            labels_j.append(int(gt_matrix[i, j] > 0))
        labels_j = np.array(labels_j)
        scores_j = np.array(scores_j)

        n_pos = labels_j.sum()
        if n_pos == 0 or n_pos == len(labels_j):
            continue
        aurocs.append(roc_auc_score(labels_j, scores_j))
        auprcs.append(average_precision_score(labels_j, scores_j))

    return {
        "mean_auroc": np.mean(aurocs),
        "mean_auprc": np.mean(auprcs),
        "median_auprc": np.median(auprcs),
        "n_targets_evaluated": len(auprcs),
        "n_master_regs": len(master_regs),
        "baseline_per_target": f"~{gt_matrix[master_regs].sum() / (len(master_regs) * (N_GENES - 1)):.3f}",
    }


def load_velorama_tsv(tsv_path):
    """Load Velorama interaction TSV into (100,100) score matrix.

    Velorama output format: rows=targets, cols=regulators.
    So interaction_df.loc[tgt, reg] = score for reg->tgt.
    We want scores_matrix[reg, tgt] = score.
    """
    df = pd.read_csv(tsv_path, sep='\t', index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    print(f"Loaded {tsv_path}: shape {df.shape}")

    scores = np.zeros((N_GENES, N_GENES))
    for tgt_idx in range(N_GENES):
        tgt_name = str(tgt_idx)
        if tgt_name not in df.index:
            continue
        for reg_idx in range(N_GENES):
            reg_name = str(reg_idx)
            if reg_name not in df.columns:
                continue
            scores[reg_idx, tgt_idx] = df.loc[tgt_name, reg_name]

    # Print score statistics
    flat = scores[np.eye(N_GENES) == 0]
    print(f"  Score stats: min={flat.min():.6f}, max={flat.max():.6f}, "
          f"mean={flat.mean():.6f}, std={flat.std():.6f}")
    print(f"  Non-zero scores: {(flat != 0).sum()}/{len(flat)}")
    print(f"  Score range [5%, 95%]: [{np.percentile(flat, 5):.6f}, "
          f"{np.percentile(flat, 95):.6f}]")

    return scores


def evaluate_tsv(tsv_path, gt_matrix):
    """Full evaluation of one Velorama TSV."""
    scores_matrix = load_velorama_tsv(tsv_path)

    print(f"\n  --- Global metrics (all 9900 pairs pooled) ---")
    g_auroc, g_auprc, baseline = global_metrics(scores_matrix, gt_matrix)
    print(f"    AUROC:    {g_auroc:.4f}")
    print(f"    AUPRC:    {g_auprc:.4f}  (random baseline: {baseline:.4f})")

    print(f"\n  --- Per-target averaged metrics (BEELINE standard) ---")
    pt = per_target_metrics(scores_matrix, gt_matrix)
    print(f"    Mean AUROC:   {pt['mean_auroc']:.4f}")
    print(f"    Mean AUPRC:   {pt['mean_auprc']:.4f}")
    print(f"    Median AUPRC: {pt['median_auprc']:.4f}")
    print(f"    Targets evaluated: {pt['n_targets_evaluated']} "
          f"(skipped {pt['n_targets_skipped']} with no true edges)")

    print(f"\n  --- Per-regulator averaged metrics ---")
    pr = per_regulator_metrics(scores_matrix, gt_matrix)
    print(f"    Mean AUROC:   {pr['mean_auroc']:.4f}")
    print(f"    Mean AUPRC:   {pr['mean_auprc']:.4f}")
    print(f"    Median AUPRC: {pr['median_auprc']:.4f}")
    print(f"    Regulators evaluated: {pr['n_regs_evaluated']} "
          f"(skipped {pr['n_regs_skipped']} with no true edges)")

    print(f"\n  --- Per-target AUPRC (master regulators only as candidates) ---")
    mr = master_reg_per_target_metrics(scores_matrix, gt_matrix)
    print(f"    Mean AUROC:   {mr['mean_auroc']:.4f}")
    print(f"    Mean AUPRC:   {mr['mean_auprc']:.4f}")
    print(f"    Median AUPRC: {mr['median_auprc']:.4f}")
    print(f"    Targets evaluated: {mr['n_targets_evaluated']} "
          f"({mr['n_master_regs']} master regs, "
          f"baseline ~{mr['baseline_per_target']})")

    # Top-k precision (another common metric in GRN literature)
    flat_scores = []
    flat_labels = []
    for i in range(N_GENES):
        for j in range(N_GENES):
            if i == j:
                continue
            flat_scores.append(abs(scores_matrix[i, j]))
            flat_labels.append(int(gt_matrix[i, j] > 0))
    flat_scores = np.array(flat_scores)
    flat_labels = np.array(flat_labels)
    order = np.argsort(-flat_scores)
    n_true = int(flat_labels.sum())

    print(f"\n  --- Top-k precision ---")
    for top_k in [n_true, n_true * 2, 500, 1000]:
        if top_k > len(order):
            continue
        prec_at_k = flat_labels[order[:top_k]].mean()
        print(f"    Top-{top_k}: precision={prec_at_k:.4f}")

    return {
        "global_auroc": g_auroc,
        "global_auprc": g_auprc,
        "pertarget_mean_auroc": pt["mean_auroc"],
        "pertarget_mean_auprc": pt["mean_auprc"],
        "perreg_mean_auroc": pr["mean_auroc"],
        "perreg_mean_auprc": pr["mean_auprc"],
        "masterreg_mean_auprc": mr["mean_auprc"],
        "masterreg_mean_auroc": mr["mean_auroc"],
        "baseline": baseline,
    }


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--tsv", help="Single Velorama TSV to evaluate")
    ap.add_argument("--tsv-dir", help="Directory to search for Velorama TSVs")
    ap.add_argument("--gt-csv", default="results/bm_velorama/velorama_data/gt_GRN.csv")
    args = ap.parse_args()

    gt_path = ensure_gt(args.gt_csv)
    gt_matrix = load_ground_truth(gt_path)

    if args.tsv:
        print(f"\n{'='*70}")
        print(f"Evaluating: {args.tsv}")
        print(f"{'='*70}")
        evaluate_tsv(args.tsv, gt_matrix)

    if args.tsv_dir:
        tsv_dir = Path(args.tsv_dir)
        tsv_files = sorted(tsv_dir.glob("*.interactions.tsv"))
        if not tsv_files:
            print(f"No TSV files found in {tsv_dir}")
            return

        all_results = []
        for tsv_path in tsv_files:
            print(f"\n{'='*70}")
            print(f"Evaluating: {tsv_path.name}")
            print(f"{'='*70}")
            result = evaluate_tsv(tsv_path, gt_matrix)
            result["file"] = tsv_path.name
            all_results.append(result)

        if all_results:
            df = pd.DataFrame(all_results)
            print(f"\n{'='*70}")
            print(f"SUMMARY")
            print(f"{'='*70}")
            print(df.to_string(index=False))

    if not args.tsv and not args.tsv_dir:
        print("Provide --tsv or --tsv-dir")


if __name__ == "__main__":
    main()
