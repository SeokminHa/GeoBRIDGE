"""
Velocity provenance diagnostic for SERGIO datasets.

Question:
  Does the SERGIO h5ad already contain a 'velocity' layer (from prior
  preparation), or does our `utils.load_sergio` always compute it on-the-fly
  via scVelo stochastic? Does the on-the-fly velocity differ from any
  pre-computed version? And what is production GB's identity (k=0) AUROC
  per velocity choice?

Approach:
  For each SERGIO dataset:
    1. Load raw h5ad
    2. Inspect layers: record which velocity keys exist (velocity,
       velocity_stochastic, velocity_deterministic, etc.)
    3. For each available velocity variant (precomputed if exists, plus
       scVelo stochastic and scVelo deterministic), run production GB at
       k=0 (identity) and k=5.
    4. Report per-TF AUROC/AUPRC.

This isolates whether the weak/anti-signal on SERGIO A/B/D in the earlier
surrogate ablation is due to velocity estimation rather than projection
logic.

Usage:
  python -u -m benchmark.diagnose_sergio_velocity --datasets A B C D
"""
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from benchmark.utils_projection import (
    scores_with_custom_vk, svd_vk, precompute_pre, evaluate_scores_per_tf,
)
from benchmark.utils import SERGIO_BASE

OUTDIR = Path("results/velocity_diagnostic")


def load_sergio_raw(name):
    """Return the h5ad, trying both naming conventions."""
    import scanpy as sc
    for cand in [SERGIO_BASE / f"dataset_{name}.h5ad",
                 SERGIO_BASE / f"test_{name}.h5ad"]:
        if cand.exists():
            return sc.read_h5ad(str(cand)), cand.name
    raise FileNotFoundError(f"No SERGIO h5ad for {name}")


def load_sergio_gt(gene_names, min_targets=5):
    """SERGIO ground truth → gt_dict filtered to genes in gene_names."""
    gt_csv = SERGIO_BASE / "gt_GRN.csv"
    edges = pd.read_csv(gt_csv, header=None, names=["reg", "tgt"])
    gt = {}
    for _, r in edges.iterrows():
        reg_s = str(int(r["reg"])); tgt_s = str(int(r["tgt"]))
        gt.setdefault(reg_s, set()).add(tgt_s)
    filtered = {}
    for tf, tgts in gt.items():
        if tf not in gene_names:
            continue
        valid = {t for t in tgts if t in gene_names and t != tf}
        if len(valid) >= min_targets:
            filtered[tf] = valid
    return filtered


def compute_velocity(adata, mode, copy=True):
    """Compute scVelo velocity with the given mode; return (adata, V)."""
    import scvelo as scv
    a = adata.copy() if copy else adata
    if "Ms" not in a.layers:
        scv.pp.moments(a, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(a, mode=mode)
    return a, np.nan_to_num(np.asarray(a.layers["velocity"], dtype=float), nan=0.0)


def ensure_pseudotime(adata):
    """Return pseudotime; derive DPT if absent."""
    for key in ("pseudotime", "dpt_pseudotime"):
        if key in adata.obs:
            return np.nan_to_num(
                np.asarray(adata.obs[key], dtype=float),
                nan=float(np.nanmedian(adata.obs[key])),
            )
    import scanpy as sc
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=30)
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_pcs=30)
    sc.tl.diffmap(adata)
    adata.uns["iroot"] = 0
    sc.tl.dpt(adata)
    return np.asarray(adata.obs["dpt_pseudotime"], dtype=float)


def run_gb_identity_and_k5(X, V, pt, gene_names, gt_dict):
    """Run production GB at k=0 (identity) and k=5. Return summary dict."""
    pre = precompute_pre(X, V, pt, pair_seed=42)
    out = {}
    for k in (0, 5):
        Vk = svd_vk(pre, k) if k > 0 else None
        scores = scores_with_custom_vk(pre, Vk, normalize_score=True)
        df = evaluate_scores_per_tf(scores, gt_dict, gene_names)
        out[k] = {
            "n_tfs": len(df),
            "auroc_mean": float(df["auroc"].mean()) if len(df) else float("nan"),
            "auprc_mean": float(df["auprc"].mean()) if len(df) else float("nan"),
            "baseline_auprc_mean": float(df["baseline_auprc"].mean()) if len(df) else float("nan"),
        }
    return out


def diagnose(name):
    import scanpy as sc
    print(f"\n=== SERGIO {name} ===")
    adata, fname = load_sergio_raw(name)
    print(f"  file: {fname}  shape: {adata.shape}  layers: {list(adata.layers.keys())}")

    # Which velocity variants exist in the h5ad as-is?
    preexisting = [k for k in adata.layers.keys() if "velocity" in k.lower()]
    print(f"  pre-existing velocity layers: {preexisting}")

    gene_names = list(adata.var_names)
    X = np.asarray(
        adata.layers["Ms"] if "Ms" in adata.layers
        else (adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X),
        dtype=float,
    )
    pt = ensure_pseudotime(adata)
    gt_dict = load_sergio_gt(gene_names, min_targets=5)
    print(f"  TFs in GT (≥5 targets): {len(gt_dict)}")

    rows = []

    # Variant 0: precomputed velocity (if present)
    if "velocity" in adata.layers and "velocity" in preexisting:
        V_pre = np.nan_to_num(np.asarray(adata.layers["velocity"], dtype=float), nan=0.0)
        r = run_gb_identity_and_k5(X, V_pre, pt, gene_names, gt_dict)
        for k, s in r.items():
            rows.append({"dataset": name, "velocity": "precomputed",
                         "k": k, **s})
        print(f"  [precomputed] k=0 AUROC={r[0]['auroc_mean']:.4f}  "
              f"k=5 AUROC={r[5]['auroc_mean']:.4f}")

    # Variant 1: scVelo stochastic (recompute fresh)
    try:
        a_s, V_s = compute_velocity(adata, mode="stochastic")
        r = run_gb_identity_and_k5(X, V_s, pt, gene_names, gt_dict)
        for k, s in r.items():
            rows.append({"dataset": name, "velocity": "scvelo_stochastic",
                         "k": k, **s})
        print(f"  [stochastic] k=0 AUROC={r[0]['auroc_mean']:.4f}  "
              f"k=5 AUROC={r[5]['auroc_mean']:.4f}")
    except Exception as e:
        print(f"  [stochastic] FAILED: {e}")

    # Variant 2: scVelo deterministic
    try:
        a_d, V_d = compute_velocity(adata, mode="deterministic")
        r = run_gb_identity_and_k5(X, V_d, pt, gene_names, gt_dict)
        for k, s in r.items():
            rows.append({"dataset": name, "velocity": "scvelo_deterministic",
                         "k": k, **s})
        print(f"  [deterministic] k=0 AUROC={r[0]['auroc_mean']:.4f}  "
              f"k=5 AUROC={r[5]['auroc_mean']:.4f}")
    except Exception as e:
        print(f"  [deterministic] FAILED: {e}")

    # Variant 3: scVelo dynamical (if available; heavier)
    try:
        import scvelo as scv
        a = adata.copy()
        if "Ms" not in a.layers:
            scv.pp.moments(a, n_pcs=30, n_neighbors=30)
        scv.tl.recover_dynamics(a, n_jobs=4)
        scv.tl.velocity(a, mode="dynamical")
        V_dyn = np.nan_to_num(np.asarray(a.layers["velocity"], dtype=float), nan=0.0)
        r = run_gb_identity_and_k5(X, V_dyn, pt, gene_names, gt_dict)
        for k, s in r.items():
            rows.append({"dataset": name, "velocity": "scvelo_dynamical",
                         "k": k, **s})
        print(f"  [dynamical] k=0 AUROC={r[0]['auroc_mean']:.4f}  "
              f"k=5 AUROC={r[5]['auroc_mean']:.4f}")
    except Exception as e:
        print(f"  [dynamical] SKIP: {e}")

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--no-dynamical", action="store_true",
                    help="Skip dynamical model (slow)")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for ds in args.datasets:
        try:
            df = diagnose(ds)
            df.to_csv(OUTDIR / f"{ds}_velocity_diag.csv", index=False)
            frames.append(df)
        except Exception as e:
            print(f"[ERROR] {ds}: {e}")
            import traceback; traceback.print_exc()

    if frames:
        full = pd.concat(frames, ignore_index=True)
        full.to_csv(OUTDIR / "all_velocity_diag.csv", index=False)

        # Compact report
        lines = ["# SERGIO velocity provenance diagnostic\n"]
        lines.append("| dataset | velocity | k | n_TFs | AUROC | AUPRC | baseline AUPRC |")
        lines.append("|---|---|---|---|---|---|---|")
        for _, r in full.iterrows():
            lines.append(
                f"| {r['dataset']} | {r['velocity']} | {r['k']} | "
                f"{r['n_tfs']} | {r['auroc_mean']:.4f} | "
                f"{r['auprc_mean']:.4f} | {r['baseline_auprc_mean']:.4f} |"
            )
        (OUTDIR / "REPORT.md").write_text("\n".join(lines))
        print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
