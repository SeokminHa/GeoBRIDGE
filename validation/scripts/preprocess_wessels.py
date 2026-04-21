"""Wessels 2023 CRISPRi (GSE150062) preprocessing skeleton.

Manual steps (document what to run on oni):
  1) prefetch SRA accessions from data/raw/wessels/SRR_list.txt
  2) fasterq-dump -> fastq
  3) cellranger count (or starsolo) -> BAM + filtered_feature_bc_matrix
  4) velocyto run10x -> .loom
  5) THIS script: assemble into ready h5ad with velocity

This script handles (5) only. Steps 1-4 are in scripts/server/preprocess_wessels.sh.
"""
from __future__ import annotations
import argparse, yaml
from pathlib import Path
import scanpy as sc
import scvelo as scv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out-wt", default=None)
    ap.add_argument("--out-pert", default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    loom = cfg["raw_loom"]
    adata = scv.read_loom(loom)

    # TODO: merge sgRNA metadata — wire to whatever column matches in cfg["perturb"]
    # Typically sgRNA info is in a separate CSV per cell barcode:
    # meta = pd.read_csv("data/raw/wessels/sgrna_per_cell.csv")
    # adata.obs = adata.obs.merge(meta.set_index("barcode"), left_index=True, right_index=True, how="left")

    scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=cfg.get("n_hvg", 3000))
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(adata, mode="stochastic")
    scv.tl.velocity_graph(adata)

    p = cfg.get("perturb", {})
    sk = p.get("sgrna_key", "perturbation")
    ctrl = set(p.get("control_labels", ["non-targeting", "NT", "safe-harbor"]))

    if sk in adata.obs.columns:
        is_wt = adata.obs[sk].astype(str).isin(ctrl)
        wt = adata[is_wt].copy()
        pert = adata[~is_wt].copy()
    else:
        print(f"[warn] no sgRNA key '{sk}' in obs; writing full object as WT only")
        wt, pert = adata, None

    out_wt = Path(args.out_wt or cfg["h5ad"])
    out_wt.parent.mkdir(parents=True, exist_ok=True)
    wt.write_h5ad(out_wt)
    print(f"[wt]   {out_wt}  ({wt.n_obs} × {wt.n_vars})")

    if pert is not None:
        out_p = Path(args.out_pert or cfg["h5ad_pert"])
        pert.write_h5ad(out_p)
        print(f"[pert] {out_p}  ({pert.n_obs} × {pert.n_vars})")


if __name__ == "__main__":
    main()
