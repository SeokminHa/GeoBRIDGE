"""
Scribe (RDI) vs GeoBRIDGE on SERGIO synthetic data.

Uses the same SERGIO datasets as bm_velorama_comparison.py so results
can be merged into a single comparison table.

Scribe API (as of master branch):
  from Scribe.Scribe import causal_net_dynamics_coupling, CLR
  adata_result = causal_net_dynamics_coupling(
      adata, TFs=None, Targets=None,
      t0_key='Ms', t1_key='velocity',
      normalize=True, drop_zero_cells=True, copy=True,
  )
  rdi_matrix = adata_result.uns['causal_net']['RDI']
  clr_matrix = CLR(rdi_matrix)  # background-corrected scores

Installation:
  pip install git+https://github.com/aristoteleo/Scribe-py

Usage:
  python -u -m benchmark.bm_scribe_sergio
"""
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))


OUTDIR = Path("results/bm_scribe")
CSV_DIR = OUTDIR / "csv"

# Path to SERGIO data (reused from Velorama benchmark)
SERGIO_DATA = Path("results/bm_velorama/velorama_data")
SERGIO_H5ADS = {
    "A": SERGIO_DATA / "test_A.h5ad",
    "B": SERGIO_DATA / "test_B.h5ad",
    "C": SERGIO_DATA / "test_C.h5ad",
    "D": SERGIO_DATA / "test_D.h5ad",
}
SERGIO_GT = SERGIO_DATA / "gt_GRN.csv"
N_GENES = 100


def load_ground_truth():
    edges = pd.read_csv(SERGIO_GT, header=None, names=["reg", "tgt"])
    gt = np.zeros((N_GENES, N_GENES), dtype=float)
    for _, r in edges.iterrows():
        gt[int(r["reg"]), int(r["tgt"])] = 1
    return gt


def run_scribe_on_adata(adata):
    """Run Scribe (RDI + CLR). Returns (N_GENES, N_GENES) score matrix.

    Convention: scores[reg, tgt] = strength of reg -> tgt influence.

    Scribe's RDI output layout (checked from source):
      adata.uns['causal_net']['RDI']  is a DataFrame with shape
      (n_targets × n_regs); rows = targets, cols = regulators.
    """
    from Scribe.Scribe import causal_net_dynamics_coupling, CLR

    # Ensure needed layers
    if "Ms" not in adata.layers:
        if "spliced" in adata.layers:
            adata.layers["Ms"] = adata.layers["spliced"]
        else:
            adata.layers["Ms"] = adata.X

    # Scribe requires velocity layer
    if "velocity" not in adata.layers:
        raise RuntimeError(
            "adata has no 'velocity' layer; compute with scVelo first")

    adata_result = causal_net_dynamics_coupling(
        adata,
        TFs=None, Targets=None,
        t0_key="Ms", t1_key="velocity",
        normalize=True, drop_zero_cells=True,
        copy=True,
    )
    rdi_df = adata_result.uns["causal_net"]["RDI"]
    # CLR background correction
    try:
        clr_df = CLR(rdi_df)
    except Exception as e:
        print(f"    [WARN] CLR failed: {e}; using raw RDI")
        clr_df = rdi_df

    # rdi_df rows=targets, cols=regulators → convert to scores[reg, tgt]
    gene_names = list(adata.var_names)
    G = len(gene_names)
    name_to_idx = {g: i for i, g in enumerate(gene_names)}

    rdi_raw = np.zeros((G, G))
    clr_m = np.zeros((G, G))
    for tgt_name in clr_df.index:
        if tgt_name not in name_to_idx:
            continue
        j = name_to_idx[tgt_name]
        for reg_name in clr_df.columns:
            if reg_name not in name_to_idx:
                continue
            i = name_to_idx[reg_name]
            clr_m[i, j] = float(clr_df.loc[tgt_name, reg_name])
            if isinstance(rdi_df, pd.DataFrame):
                rdi_raw[i, j] = float(rdi_df.loc[tgt_name, reg_name])

    return {"rdi": rdi_raw, "clr": clr_m}


def global_metrics(scores, gt):
    flat_s, flat_l = [], []
    for i in range(N_GENES):
        for j in range(N_GENES):
            if i == j:
                continue
            flat_s.append(abs(scores[i, j]))
            flat_l.append(int(gt[i, j] > 0))
    s = np.array(flat_s)
    l = np.array(flat_l)
    return {
        "auroc": roc_auc_score(l, s),
        "auprc": average_precision_score(l, s),
        "baseline": l.mean(),
    }


def per_target_metrics(scores, gt):
    aurocs, auprcs = [], []
    for j in range(N_GENES):
        s, l = [], []
        for i in range(N_GENES):
            if i == j:
                continue
            s.append(abs(scores[i, j]))
            l.append(int(gt[i, j] > 0))
        s = np.array(s); l = np.array(l)
        if l.sum() == 0 or l.sum() == len(l):
            continue
        aurocs.append(roc_auc_score(l, s))
        auprcs.append(average_precision_score(l, s))
    return {
        "auroc_mean": float(np.mean(aurocs)),
        "auroc_std": float(np.std(aurocs)),
        "auprc_mean": float(np.mean(auprcs)),
        "auprc_std": float(np.std(auprcs)),
        "n_targets": len(aurocs),
    }


def main():
    import scanpy as sc
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    gt = load_ground_truth()
    print(f"GT: {int(gt.sum())} edges")

    rows = []
    for name, h5ad_path in SERGIO_H5ADS.items():
        if not h5ad_path.exists():
            print(f"[SKIP] {name}: {h5ad_path} missing")
            continue

        print(f"\n=== Dataset {name} ===")
        adata = sc.read_h5ad(str(h5ad_path))
        print(f"  {adata.shape}, layers={list(adata.layers.keys())}")

        t0 = time.perf_counter()
        out = run_scribe_on_adata(adata)
        elapsed = time.perf_counter() - t0
        print(f"  Scribe done in {elapsed:.1f}s")

        for score_type, scores in out.items():
            g_m = global_metrics(scores, gt)
            pt_m = per_target_metrics(scores, gt)
            row = {
                "dataset": name,
                "method": f"Scribe_{score_type.upper()}",
                "elapsed_sec": elapsed,
                "auroc_global": g_m["auroc"],
                "auprc_global": g_m["auprc"],
                "baseline": g_m["baseline"],
                "auroc_pertarget_mean": pt_m["auroc_mean"],
                "auroc_pertarget_std": pt_m["auroc_std"],
                "auprc_pertarget_mean": pt_m["auprc_mean"],
                "auprc_pertarget_std": pt_m["auprc_std"],
                "n_targets_evaluated": pt_m["n_targets"],
            }
            rows.append(row)
            print(f"    [{score_type.upper()}] global AUROC={g_m['auroc']:.4f} "
                  f"AUPRC={g_m['auprc']:.4f}  "
                  f"per-target AUROC={pt_m['auroc_mean']:.4f}±"
                  f"{pt_m['auroc_std']:.3f}")

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(CSV_DIR / "scribe_results.csv", index=False)
        print(f"\n[SAVE] {CSV_DIR}/scribe_results.csv ({len(df)} rows)")

        # Combined summary
        print(f"\n{'='*70}")
        print("SCRIBE SERGIO SUMMARY")
        print(f"{'='*70}")
        print(f"{'DS':<3s} {'method':<14s} {'t(s)':>7s} "
              f"{'AUROCg':>7s} {'AUPRCg':>7s} "
              f"{'AUROCpt':>8s} {'AUPRCpt':>8s}")
        for _, r in df.iterrows():
            print(f"{r['dataset']:<3s} {r['method']:<14s} "
                  f"{r['elapsed_sec']:7.1f} "
                  f"{r['auroc_global']:7.4f} {r['auprc_global']:7.4f} "
                  f"{r['auroc_pertarget_mean']:8.4f} "
                  f"{r['auprc_pertarget_mean']:8.4f}")


if __name__ == "__main__":
    main()
