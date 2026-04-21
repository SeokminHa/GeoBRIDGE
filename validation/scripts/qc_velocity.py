"""One-shot velocity sanity check before burning overnight compute.

For scvelo-provided datasets, this prepares a ready h5ad with HVGs + moments +
stochastic velocity, and saves the embedding-stream figure.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import yaml
import scanpy as sc
import scvelo as scv


LOADERS = {
    "bonemarrow": scv.datasets.bonemarrow,
    "pancreas":   scv.datasets.pancreas,
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
        key = {"bm": "bonemarrow", "pancreas": "pancreas"}.get(cfg["name"])
        if key is None:
            raise ValueError(f"unknown scvelo loader for {cfg['name']}")
        adata = LOADERS[key]()
    else:
        adata = sc.read(cfg["h5ad"])

    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=cfg.get("n_hvg", 3000))
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(adata, mode="stochastic")
    scv.tl.velocity_graph(adata)
    scv.tl.umap(adata) if "X_umap" not in adata.obsm else None

    # QC plot
    ax = scv.pl.velocity_embedding_stream(
        adata, basis="umap", show=False, save=False, color=adata.obs.columns[0]
    )
    fig = ax.get_figure() if hasattr(ax, "get_figure") else None
    if fig is not None:
        fig.savefig(qc / f"{cfg['name']}_velocity.pdf", bbox_inches="tight")

    adata.write_h5ad(out_h5)
    print(f"[qc] wrote {out_h5}  ({adata.n_obs} cells × {adata.n_vars} genes)")


if __name__ == "__main__":
    main()
