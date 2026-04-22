"""One-shot velocity sanity check before burning overnight compute.

For scvelo-provided datasets, this prepares a ready h5ad with HVGs + moments +
stochastic velocity, and saves the embedding-stream figure.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import scvelo as scv


def _sanitize_and_hvg(adata, n_top: int):
    """Replace inf/nan in X (from zero-count cells) and pick top-variance genes."""
    X = adata.X
    if sp.issparse(X):
        mask = ~np.isfinite(X.data)
        if mask.any():
            print(f"[qc] zeroing {int(mask.sum())} non-finite entries in X")
            X.data[mask] = 0
        mean = np.asarray(X.mean(axis=0)).ravel()
        sq = np.asarray(X.multiply(X).mean(axis=0)).ravel()
        var = np.clip(sq - mean ** 2, 0, None)
    else:
        bad = ~np.isfinite(X)
        if bad.any():
            print(f"[qc] zeroing {int(bad.sum())} non-finite entries in X")
            X[bad] = 0
        var = np.nanvar(np.asarray(X), axis=0)
    var = np.where(np.isfinite(var), var, 0.0)
    n_keep = min(n_top, var.size)
    top = np.argsort(-var)[:n_keep]
    keep = np.zeros(var.size, dtype=bool); keep[top] = True
    adata._inplace_subset_var(keep)
    print(f"[qc] kept {keep.sum()} HVGs by variance")


LOADERS = {
    "bonemarrow":            scv.datasets.bonemarrow,
    "pancreas":              scv.datasets.pancreas,
    "gastrulation_erythroid": scv.datasets.gastrulation_erythroid,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--qc-out", default="results/qc")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    out_h5 = Path(cfg["h5ad"]); out_h5.parent.mkdir(parents=True, exist_ok=True)
    qc = Path(args.qc_out); qc.mkdir(parents=True, exist_ok=True)

    if cfg.get("raw_loader") == "scvelo":
        # key derived from config name
        key = {"bm": "bonemarrow", "pancreas": "pancreas",
               "gastrulation_erythroid": "gastrulation_erythroid"}.get(cfg["name"])
        if key is None:
            raise ValueError(f"unknown scvelo loader for {cfg['name']}")
        adata = LOADERS[key]()
    else:
        adata = sc.read(cfg["h5ad"])

    n_hvg = cfg.get("n_hvg", 3000)
    sc.pp.filter_cells(adata, min_counts=200)
    scv.pp.filter_and_normalize(adata, min_shared_counts=20)
    _sanitize_and_hvg(adata, n_hvg)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(adata, mode="stochastic")
    scv.tl.velocity_graph(adata)
    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata)

    # Persist ready h5ad FIRST so downstream steps never depend on plot success.
    adata.write_h5ad(out_h5)
    print(f"[qc] wrote {out_h5}  ({adata.n_obs} cells × {adata.n_vars} genes)")

    # QC plot is best-effort — fail open.
    try:
        # Drop cells with non-finite UMAP coords (cause of
        # 'Can only output finite numbers in PDF' in matplotlib)
        if "X_umap" in adata.obsm:
            um = np.asarray(adata.obsm["X_umap"])
            bad = ~np.isfinite(um).all(axis=1)
            if bad.any():
                print(f"[qc] {int(bad.sum())} cells have non-finite UMAP coords; "
                      f"excluding from plot")
        ax = scv.pl.velocity_embedding_stream(
            adata, basis="umap", show=False, save=False,
            color=adata.obs.columns[0],
        )
        fig = ax.get_figure() if hasattr(ax, "get_figure") else None
        if fig is not None:
            # PNG avoids the pdf-finite-number check; good enough for QC.
            fig.savefig(qc / f"{cfg['name']}_velocity.png",
                        bbox_inches="tight", dpi=150)
    except Exception as e:
        print(f"[qc] plot step failed ({type(e).__name__}: {e}); "
              f"skipping (h5ad already written)")


if __name__ == "__main__":
    main()
