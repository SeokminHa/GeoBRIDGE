"""
Velocity provenance diagnostic for SERGIO datasets.

Question:
  Does the SERGIO h5ad already contain a 'velocity' layer (from prior
  preparation), or does our `utils.load_sergio` always compute it on-the-fly
  via scVelo stochastic? Does the on-the-fly velocity differ from any
  pre-computed version? And what is production GB's identity (k=0) AUROC
  per velocity choice?

Design invariants:
  - X (expression) is held FIXED across velocity variants. Only V changes.
  - Pseudotime is also held fixed; its source (obs column vs freshly-computed
    DPT) is recorded in the output under `pseudotime_source`.
  - GT and TF set are fixed across variants.

Approach:
  For each SERGIO dataset:
    1. Load raw h5ad
    2. Record existing velocity layers (precomputed inputs)
    3. For each velocity variant {precomputed if present, scVelo stochastic,
       scVelo deterministic, (optional) scVelo dynamical} run production GB
       at k=0 (identity) and k=5.
    4. Report AUROC/AUPRC/stdAUPRC + gate_pass flag (identity AUROC > 0.5).

This isolates whether the weak/anti-signal on SERGIO A/B/D in the earlier
surrogate ablation is due to velocity estimation rather than projection
logic.

Usage:
  python -u -m benchmark.diagnose_sergio_velocity --datasets A B C D
  python -u -m benchmark.diagnose_sergio_velocity --datasets B --no-dynamical
  python -u -m benchmark.diagnose_sergio_velocity --datasets B --save-per-tf
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


def compute_velocity(adata, mode, copy=True, force_recompute_moments=True):
    """
    Compute scVelo velocity with the given mode.

    For a diagnostic script we want reproducible behavior: by default we
    FORCE-recompute moments (ignoring any cached `Ms`/`Mu`) so that each
    velocity variant is built from the same preprocessing pipeline.

    Returns (adata_copy, V_dense_array).
    """
    import scvelo as scv
    a = adata.copy() if copy else adata
    if force_recompute_moments or "Ms" not in a.layers:
        scv.pp.moments(a, n_pcs=30, n_neighbors=30)
    if mode == "dynamical":
        scv.tl.recover_dynamics(a, n_jobs=4)
    scv.tl.velocity(a, mode=mode)
    V = np.nan_to_num(np.asarray(a.layers["velocity"], dtype=float), nan=0.0)
    return a, V


def ensure_pseudotime(adata):
    """
    Return (pt, source) where source ∈ {'obs:pseudotime', 'obs:dpt_pseudotime',
    'derived:dpt'} indicating pseudotime provenance.
    """
    for key in ("pseudotime", "dpt_pseudotime"):
        if key in adata.obs:
            pt = np.nan_to_num(
                np.asarray(adata.obs[key], dtype=float),
                nan=float(np.nanmedian(adata.obs[key])),
            )
            return pt, f"obs:{key}"
    import scanpy as sc
    if "X_pca" not in adata.obsm:
        sc.pp.pca(adata, n_comps=30)
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, n_pcs=30)
    sc.tl.diffmap(adata)
    adata.uns["iroot"] = 0
    sc.tl.dpt(adata)
    pt = np.asarray(adata.obs["dpt_pseudotime"], dtype=float)
    return pt, "derived:dpt"


def run_gb_identity_and_k5(X, V, pt, gene_names, gt_dict, return_per_tf=False):
    """
    Run production GB at k=0 (identity) and k=5 with the fair-comparison
    scoring (project_y=False).  Return summary dict + optionally per-TF DFs.
    """
    pre = precompute_pre(X, V, pt, pair_seed=42)
    out = {}
    per_tf = {}
    for k in (0, 5):
        Vk = svd_vk(pre, k) if k > 0 else None
        scores = scores_with_custom_vk(pre, Vk, normalize_score=True)
        df = evaluate_scores_per_tf(scores, gt_dict, gene_names)
        per_tf[k] = df
        out[k] = {
            "n_tfs": len(df),
            "auroc_mean": float(df["auroc"].mean()) if len(df) else float("nan"),
            "auprc_mean": float(df["auprc"].mean()) if len(df) else float("nan"),
            "std_auprc_mean": float(df["std_auprc"].mean()) if (len(df) and "std_auprc" in df.columns) else float("nan"),
            "baseline_auprc_mean": float(df["baseline_auprc"].mean()) if len(df) else float("nan"),
        }
    if return_per_tf:
        return out, per_tf
    return out


def diagnose(name, no_dynamical=False, save_per_tf_dir=None):
    """
    Run the velocity provenance diagnostic for one SERGIO dataset.

    Design: X (expression) and pseudotime are computed ONCE and held fixed
    across all velocity variants — only V changes per variant. This isolates
    the effect of velocity estimation on downstream GB scoring.
    """
    import scanpy as sc
    print(f"\n=== SERGIO {name} ===")
    adata, fname = load_sergio_raw(name)
    print(f"  file: {fname}  shape: {adata.shape}  layers: {list(adata.layers.keys())}")

    # Which velocity layers are already in the file (precomputed input)?
    preexisting = [k for k in adata.layers.keys() if "velocity" in k.lower()]
    has_precomputed_velocity = "velocity" in adata.layers
    print(f"  pre-existing velocity layers: {preexisting}  "
          f"(precomputed 'velocity' usable: {has_precomputed_velocity})")

    # X and pseudotime are HELD FIXED across all velocity variants below.
    gene_names = list(adata.var_names)
    X = np.asarray(
        adata.layers["Ms"] if "Ms" in adata.layers
        else (adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X),
        dtype=float,
    )
    pt, pt_source = ensure_pseudotime(adata)
    gt_dict = load_sergio_gt(gene_names, min_targets=5)
    print(f"  TFs in GT (≥5 targets): {len(gt_dict)}  "
          f"pseudotime source: {pt_source}")

    rows = []

    def _run_variant(vel_name, V_arr):
        result = run_gb_identity_and_k5(
            X, V_arr, pt, gene_names, gt_dict, return_per_tf=bool(save_per_tf_dir),
        )
        if save_per_tf_dir:
            summary, per_tf = result
        else:
            summary, per_tf = result, None
        # Gate pass = identity (k=0) AUROC > 0.5
        gate_pass = bool(summary[0]["auroc_mean"] > 0.5)
        for k, s in summary.items():
            rows.append({
                "dataset": name,
                "velocity": vel_name,
                "pseudotime_source": pt_source,
                "k": k,
                "gate_pass": gate_pass,
                **s,
            })
        # Save per-TF if requested
        if save_per_tf_dir is not None and per_tf is not None:
            out = Path(save_per_tf_dir)
            out.mkdir(parents=True, exist_ok=True)
            for k, df in per_tf.items():
                if len(df) == 0:
                    continue
                df_out = df.copy()
                df_out["dataset"] = name
                df_out["velocity"] = vel_name
                df_out["k"] = k
                df_out["pseudotime_source"] = pt_source
                df_out.to_csv(
                    out / f"{name}_{vel_name}_k{k}_per_tf.csv",
                    index=False,
                )
        print(f"  [{vel_name}] gate_pass={gate_pass}  "
              f"k=0 AUROC={summary[0]['auroc_mean']:.4f} "
              f"stdAUPRC={summary[0]['std_auprc_mean']:.2f}×  |  "
              f"k=5 AUROC={summary[5]['auroc_mean']:.4f} "
              f"stdAUPRC={summary[5]['std_auprc_mean']:.2f}×")

    # Variant 0: precomputed velocity (single, clean check)
    if has_precomputed_velocity:
        V_pre = np.nan_to_num(
            np.asarray(adata.layers["velocity"], dtype=float), nan=0.0,
        )
        _run_variant("precomputed", V_pre)

    # Variant 1: scVelo stochastic
    try:
        _, V_s = compute_velocity(adata, mode="stochastic")
        _run_variant("scvelo_stochastic", V_s)
    except Exception as e:
        print(f"  [scvelo_stochastic] FAILED: {e}")

    # Variant 2: scVelo deterministic
    try:
        _, V_d = compute_velocity(adata, mode="deterministic")
        _run_variant("scvelo_deterministic", V_d)
    except Exception as e:
        print(f"  [scvelo_deterministic] FAILED: {e}")

    # Variant 3: scVelo dynamical (heaviest; opt-out)
    if no_dynamical:
        print(f"  [scvelo_dynamical] SKIP (--no-dynamical)")
    else:
        try:
            _, V_dyn = compute_velocity(adata, mode="dynamical")
            _run_variant("scvelo_dynamical", V_dyn)
        except Exception as e:
            print(f"  [scvelo_dynamical] FAILED: {e}")

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["A", "B", "C", "D"])
    ap.add_argument("--no-dynamical", action="store_true",
                    help="Skip scVelo dynamical mode (slow)")
    ap.add_argument("--save-per-tf", action="store_true",
                    help="Also save per-TF CSV per (dataset, velocity, k)")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    per_tf_dir = (OUTDIR / "per_tf") if args.save_per_tf else None

    frames = []
    for ds in args.datasets:
        try:
            df = diagnose(
                ds,
                no_dynamical=args.no_dynamical,
                save_per_tf_dir=per_tf_dir,
            )
            df.to_csv(OUTDIR / f"{ds}_velocity_diag.csv", index=False)
            frames.append(df)
        except Exception as e:
            print(f"[ERROR] {ds}: {e}")
            import traceback; traceback.print_exc()

    if frames:
        full = pd.concat(frames, ignore_index=True)
        full.to_csv(OUTDIR / "all_velocity_diag.csv", index=False)

        # Report with gate_pass + stdAUPRC columns
        lines = ["# SERGIO velocity provenance diagnostic\n"]
        lines.append(
            "X and pseudotime are held fixed per dataset; only V changes.\n"
        )
        lines.append(
            "| dataset | velocity | pseudotime | k | n_TFs | "
            "AUROC | AUPRC | stdAUPRC | baseline AUPRC | gate_pass |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for _, r in full.iterrows():
            lines.append(
                f"| {r['dataset']} | {r['velocity']} | "
                f"{r.get('pseudotime_source', '?')} | {r['k']} | "
                f"{r['n_tfs']} | {r['auroc_mean']:.4f} | "
                f"{r['auprc_mean']:.4f} | "
                f"{r.get('std_auprc_mean', float('nan')):.2f}× | "
                f"{r['baseline_auprc_mean']:.4f} | "
                f"{r.get('gate_pass', False)} |"
            )
        (OUTDIR / "REPORT.md").write_text("\n".join(lines))
        print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
