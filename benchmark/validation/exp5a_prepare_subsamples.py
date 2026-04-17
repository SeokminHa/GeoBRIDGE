"""
EXP 5a. Prepare subsampled h5ads for stability pilot.

Creates N_SUBSAMPLES x DATASETS h5ad files by subsampling cells.
Saves cell ID lists + the h5ads themselves.

Usage:
  python -u -m benchmark.validation.exp5a_prepare_subsamples [--dataset bonemarrow]

Output:
  results/bm_real_comparison/validation/exp5_stability/subsamples/
    {ds}_seed{seed}.h5ad        (subsampled h5ad)
    {ds}_seed{seed}_cells.txt   (cell IDs for reproducibility)
"""
import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from benchmark.validation.config import (
    PREPARED_DIR, EXP5_DIR, DATASETS,
    STABILITY_SEEDS, STABILITY_SUBSAMPLE_FRAC, ensure_dirs,
)


def subsample(ds, seed, outdir):
    import scanpy as sc
    h5ad_in = PREPARED_DIR / f"{ds}.h5ad"
    adata = sc.read_h5ad(str(h5ad_in))
    n_cells = adata.shape[0]
    n_keep = int(n_cells * STABILITY_SUBSAMPLE_FRAC)

    rng = np.random.default_rng(seed)
    idx = rng.choice(n_cells, size=n_keep, replace=False)
    idx = np.sort(idx)
    sub = adata[idx].copy()

    # Re-run velocity graph computation on subset (needed by Velorama and
    # any method using knn graphs)
    import scvelo as scv
    try:
        scv.pp.moments(sub, n_pcs=30, n_neighbors=30)
        scv.tl.velocity(sub, mode="stochastic")
    except Exception as e:
        print(f"    [WARN] velocity recompute failed: {e}")

    # Pseudotime: preserve original pt column but only for kept cells
    if "pseudotime" in adata.obs:
        sub.obs["pseudotime"] = adata.obs["pseudotime"].values[idx]

    # Ensure Velorama annotations
    sub.var["is_reg"] = True
    sub.var["is_target"] = True

    out_h5ad = outdir / f"{ds}_seed{seed}.h5ad"
    sub.write_h5ad(str(out_h5ad))

    cell_ids = list(adata.obs.index[idx])
    (outdir / f"{ds}_seed{seed}_cells.txt").write_text(
        "\n".join(cell_ids)
    )
    print(f"    [OK] seed={seed}: {sub.shape[0]} cells → {out_h5ad}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    args = ap.parse_args()

    ensure_dirs()
    outdir = EXP5_DIR / "subsamples"
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    for ds in datasets:
        print(f"\n=== {ds} ===")
        for seed in STABILITY_SEEDS:
            subsample(ds, seed, outdir)

    print(f"\n[DONE] Subsamples in {outdir}/")


if __name__ == "__main__":
    main()
