"""
Step 0: Download Velorama synthetic datasets and inspect their structure.
Find where ground truth regulatory edges are stored.

Run this FIRST before bm_velorama_comparison.py.

Usage:
  python -m benchmark.bm_velorama_inspect --outdir results/bm_velorama
"""

import sys
import os
from pathlib import Path
import urllib.request

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

DATASET_BASE_URL = "https://github.com/rs239/velorama/raw/main/datasets"
DATASETS = ["dataset_A", "dataset_B", "dataset_C", "dataset_D"]


def download_datasets(outdir):
    """Download Velorama synthetic h5ad files."""
    data_dir = Path(outdir) / "velorama_data"
    data_dir.mkdir(parents=True, exist_ok=True)

    for ds in DATASETS:
        fname = f"{ds}.h5ad"
        fpath = data_dir / fname
        if fpath.exists():
            print(f"  [SKIP] {fname} already exists")
            continue
        url = f"{DATASET_BASE_URL}/{fname}"
        print(f"  [DOWNLOAD] {fname} from {url}")
        try:
            urllib.request.urlretrieve(url, fpath)
            print(f"    -> {fpath} ({fpath.stat().st_size / 1e6:.1f} MB)")
        except Exception as e:
            print(f"    -> FAILED: {e}")

    return data_dir


def inspect_h5ad(fpath):
    """Thoroughly inspect an h5ad file to find ground truth."""
    import scanpy as sc
    import numpy as np

    print(f"\n{'='*70}")
    print(f"INSPECTING: {fpath.name}")
    print(f"{'='*70}")

    adata = sc.read_h5ad(str(fpath))
    print(f"\n  Shape: {adata.shape} (cells x genes)")
    print(f"  X type: {type(adata.X)}, dtype: {adata.X.dtype if hasattr(adata.X, 'dtype') else 'N/A'}")

    # --- obs (cell metadata) ---
    print(f"\n  obs columns: {list(adata.obs.columns)}")
    for col in adata.obs.columns:
        vals = adata.obs[col]
        if vals.dtype in ['float64', 'float32', 'int64', 'int32']:
            print(f"    {col}: {vals.dtype}, range=[{vals.min():.4f}, {vals.max():.4f}]")
        else:
            print(f"    {col}: {vals.dtype}, nunique={vals.nunique()}")

    # --- var (gene metadata) ---
    print(f"\n  var columns: {list(adata.var.columns)}")
    for col in adata.var.columns:
        vals = adata.var[col]
        if vals.dtype == bool:
            print(f"    {col}: {vals.sum()}/{len(vals)} True")
        elif vals.dtype in ['float64', 'float32', 'int64', 'int32']:
            print(f"    {col}: {vals.dtype}, range=[{vals.min():.4f}, {vals.max():.4f}]")
        else:
            print(f"    {col}: {vals.dtype}, nunique={vals.nunique()}")

    # --- layers ---
    print(f"\n  layers: {list(adata.layers.keys())}")
    for k, v in adata.layers.items():
        if hasattr(v, 'shape'):
            print(f"    {k}: shape={v.shape}")

    # --- obsm (embeddings) ---
    print(f"\n  obsm: {list(adata.obsm.keys())}")
    for k in adata.obsm:
        print(f"    {k}: shape={adata.obsm[k].shape}")

    # --- varm ---
    if hasattr(adata, 'varm') and adata.varm:
        print(f"\n  varm: {list(adata.varm.keys())}")
        for k in adata.varm:
            print(f"    {k}: shape={adata.varm[k].shape}")

    # --- obsp ---
    if hasattr(adata, 'obsp') and adata.obsp:
        print(f"\n  obsp: {list(adata.obsp.keys())}")
        for k in adata.obsp:
            v = adata.obsp[k]
            print(f"    {k}: shape={v.shape}, nnz={v.nnz if hasattr(v, 'nnz') else 'dense'}")

    # --- varp (gene-gene pairwise — MOST LIKELY for ground truth) ---
    if hasattr(adata, 'varp') and adata.varp:
        print(f"\n  varp: {list(adata.varp.keys())}")
        for k in adata.varp:
            v = adata.varp[k]
            print(f"    {k}: shape={v.shape}, "
                  f"nnz={v.nnz if hasattr(v, 'nnz') else 'dense'}")
            if hasattr(v, 'toarray'):
                arr = v.toarray()
            else:
                arr = np.asarray(v)
            print(f"    {k}: min={arr.min():.4f}, max={arr.max():.4f}, "
                  f"nonzero={np.count_nonzero(arr)}")
    else:
        print(f"\n  varp: (empty)")

    # --- uns (unstructured — check for ground truth) ---
    print(f"\n  uns keys: {list(adata.uns.keys())}")
    for k in adata.uns:
        v = adata.uns[k]
        if isinstance(v, np.ndarray):
            print(f"    {k}: ndarray shape={v.shape}, dtype={v.dtype}")
            if v.ndim == 2:
                print(f"      min={v.min():.4f}, max={v.max():.4f}, "
                      f"nonzero={np.count_nonzero(v)}")
        elif isinstance(v, dict):
            print(f"    {k}: dict with keys {list(v.keys())[:10]}")
            for kk, vv in list(v.items())[:5]:
                if isinstance(vv, np.ndarray):
                    print(f"      {kk}: ndarray shape={vv.shape}")
                else:
                    print(f"      {kk}: {type(vv).__name__} = {str(vv)[:80]}")
        elif isinstance(v, (str, int, float, bool)):
            print(f"    {k}: {type(v).__name__} = {v}")
        else:
            print(f"    {k}: {type(v).__name__}")

    # --- Gene names ---
    genes = list(adata.var_names)
    reg_genes = list(adata.var_names[adata.var['is_reg']]) if 'is_reg' in adata.var else []
    tgt_genes = list(adata.var_names[adata.var['is_target']]) if 'is_target' in adata.var else []
    print(f"\n  Gene names (first 20): {genes[:20]}")
    print(f"  Regulator genes ({len(reg_genes)}): {reg_genes[:15]}")
    print(f"  Target genes ({len(tgt_genes)}): {tgt_genes[:15]}")

    # Check if reg and target overlap
    overlap = set(reg_genes) & set(tgt_genes)
    print(f"  Reg-Target overlap: {len(overlap)} genes")

    return adata


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default="results/bm_velorama")
    args = ap.parse_args()

    print("=" * 70)
    print("VELORAMA SYNTHETIC DATA: DOWNLOAD AND INSPECT")
    print("=" * 70)

    data_dir = download_datasets(args.outdir)

    for ds in DATASETS:
        fpath = data_dir / f"{ds}.h5ad"
        if fpath.exists():
            adata = inspect_h5ad(fpath)
        else:
            print(f"\n  [SKIP] {ds} not found")

    print("\n" + "=" * 70)
    print("DONE — inspect output above to find ground truth location")
    print("=" * 70)


if __name__ == "__main__":
    main()
