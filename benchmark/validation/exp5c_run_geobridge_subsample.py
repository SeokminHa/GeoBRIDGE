"""
EXP 5c. Run GeoBRIDGE on each subsampled h5ad.

In-process (no subprocess), since GeoBRIDGE has no CLI and is fast.

Saves per-subsample score matrix (gene × gene) as .npz, keyed by k=K_FOCUS.
Also saves gene_names for alignment.

Usage:
  python -u -m benchmark.validation.exp5c_run_geobridge_subsample [--dataset X]
"""
import sys
import argparse
import time
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from benchmark.validation.config import (
    EXP5_DIR, DATASETS, STABILITY_SEEDS, K_FOCUS, ensure_dirs,
)


def run_one(ds, seed, sub_dir, out_dir):
    import scanpy as sc
    from geobridge_model import GeoBridgeModel

    h5ad = sub_dir / f"{ds}_seed{seed}.h5ad"
    if not h5ad.exists():
        print(f"  [SKIP] {h5ad} missing")
        return

    out_npz = out_dir / f"{ds}_seed{seed}_k{K_FOCUS}.npz"
    if out_npz.exists():
        print(f"  [SKIP] already: {out_npz}")
        return

    t0 = time.perf_counter()
    adata = sc.read_h5ad(str(h5ad))
    gene_names = list(adata.var_names)

    X = np.asarray(adata.layers["Ms"] if "Ms" in adata.layers
                   else (adata.X.toarray() if hasattr(adata.X, "toarray")
                         else adata.X), dtype=float)
    V = np.nan_to_num(np.asarray(adata.layers["velocity"], dtype=float), nan=0.0)
    pt = np.nan_to_num(np.asarray(adata.obs["pseudotime"], dtype=float),
                       nan=float(np.nanmedian(adata.obs["pseudotime"])))
    order = np.argsort(pt)

    pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
        X[order], V[order], pt[order], pair_seed=42)
    if K_FOCUS >= len(pre["S"]):
        print(f"  [SKIP] k={K_FOCUS} >= {len(pre['S'])}")
        return
    S_k = GeoBridgeModel.scores_from_streaming_stats(
        pre, k=K_FOCUS, project_y=True, normalize_score=True, eps=1e-12)

    np.savez_compressed(out_npz, scores=S_k, genes=np.array(gene_names))
    elapsed = time.perf_counter() - t0
    print(f"  [OK] {ds} seed={seed}: {elapsed:.1f}s → {out_npz.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    args = ap.parse_args()

    ensure_dirs()
    sub_dir = EXP5_DIR / "subsamples"
    out_dir = EXP5_DIR / "geobridge_out"
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        print(f"\n=== {ds} ===")
        for seed in STABILITY_SEEDS:
            run_one(ds, seed, sub_dir, out_dir)


if __name__ == "__main__":
    main()
