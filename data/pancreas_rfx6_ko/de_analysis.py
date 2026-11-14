# DESeq2-equivalent DE analysis on Aldous salmon quants
from __future__ import annotations

import gzip
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ALDOUS = Path((__import__("os").environ["GEOBRIDGE_ROOT"] + "/real_1_Pancreas/external_validation/aldous_rfx6"))
QUANTS = ALDOUS / "quants"
GTF = ALDOUS / "refs" / "gencode.v44.annotation.gtf.gz"
OUT_DIR = ALDOUS / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# Parse gencode GTF to build transcript_id -> (gene_id, gene_symbol) map
def parse_gtf_tx2gene(gtf_path: Path):
    rows = []
    open_func = gzip.open if str(gtf_path).endswith(".gz") else open
    with open_func(gtf_path, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 9 or parts[2] != "transcript":
                continue
            attrs = {}
            for f in parts[8].split(";"):
                f = f.strip()
                if not f:
                    continue
                k, _, v = f.partition(" ")
                attrs[k] = v.strip('"')
            tx_id = attrs.get("transcript_id", "")
            gene_id = attrs.get("gene_id", "")
            gene_name = attrs.get("gene_name", "")
            gene_type = attrs.get("gene_type", "")
            if tx_id and gene_id:
                rows.append({"tx_id": tx_id, "gene_id": gene_id,
                             "gene_name": gene_name, "gene_type": gene_type})
    return pd.DataFrame(rows)


# Aggregate salmon transcript NumReads to gene-level counts
def load_salmon_to_gene_counts(quants_dir: Path, samples: list[str],
                                 tx2gene: pd.DataFrame, log) -> pd.DataFrame:
    tx2g = dict(zip(tx2gene["tx_id"], tx2gene["gene_id"]))
    g2name = dict(zip(tx2gene["gene_id"], tx2gene["gene_name"]))

    gene_counts = {}
    for sname in samples:
        quant_path = quants_dir / sname / "quant.sf"
        if not quant_path.is_file():
            log(f"[WARN] missing {quant_path}")
            continue
        df = pd.read_csv(quant_path, sep="\t")
        df["gene_id"] = df["Name"].map(tx2g)
        df = df.dropna(subset=["gene_id"])
        gene_sum = df.groupby("gene_id")["NumReads"].sum()
        gene_counts[sname] = gene_sum
    counts = pd.DataFrame(gene_counts).fillna(0).astype(int)
    counts.index.name = "gene_id"
    counts["gene_name"] = counts.index.map(g2name)
    return counts


# Run pyDESeq2 DE analysis
def run_de(counts_df: pd.DataFrame, samples_meta: pd.DataFrame,
           contrast_col: str, contrast_levels: tuple, log) -> pd.DataFrame:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    # counts: samples (rows) x genes (cols), pyDESeq2 expects raw int counts
    counts = counts_df.drop(columns="gene_name").T  # samples x genes
    counts = counts.astype(int)

    dds = DeseqDataSet(
        counts=counts,
        metadata=samples_meta,
        design=f"~{contrast_col}",
        refit_cooks=True,
        n_cpus=8,
    )
    dds.deseq2()
    log(f"  DESeq2 fit complete")

    stats = DeseqStats(dds, contrast=[contrast_col,
                                       contrast_levels[0],
                                       contrast_levels[1]],
                       n_cpus=8)
    stats.summary()
    df = stats.results_df.copy()
    df = df.reset_index().rename(columns={"index": "gene_id"})
    df["gene_name"] = df["gene_id"].map(
        dict(zip(counts_df.index, counts_df["gene_name"])))
    return df


def main():
    log = lambda m: print(m, flush=True)
    log("Aldous DESeq2: EPs WT vs KO + PPs WT vs KO")

    log(f"[GTF] parsing {GTF}")
    tx2gene = parse_gtf_tx2gene(GTF)
    log(f"  {len(tx2gene)} transcript entries, "
        f"{tx2gene['gene_id'].nunique()} genes")
    pc = tx2gene[tx2gene["gene_type"] == "protein_coding"]
    log(f"  {pc['gene_id'].nunique()} protein-coding genes")

    # EPs analysis
    log("\nEPs WT vs KO")
    eps_samples = ["EPs-WT-Rep1", "EPs-WT-Rep2", "EPs-KO-Rep1", "EPs-KO-Rep2"]
    eps_meta = pd.DataFrame({
        "condition": ["WT", "WT", "KO", "KO"],
    }, index=eps_samples)
    counts_eps = load_salmon_to_gene_counts(QUANTS, eps_samples, tx2gene, log)
    log(f"  counts shape: {counts_eps.shape}")
    if len(counts_eps) > 0:
        de_eps = run_de(counts_eps, eps_meta, "condition", ("KO", "WT"), log)
        out = OUT_DIR / "aldous_EPs_WT_vs_KO_DE.tsv"
        de_eps.to_csv(out, sep="\t", index=False)
        log(f"  [SAVE] {out}")
        log(f"  KO vs WT: log2FC>1 & padj<0.05 -> "
            f"up={((de_eps['log2FoldChange']>1)&(de_eps['padj']<0.05)).sum()}, "
            f"down={((de_eps['log2FoldChange']<-1)&(de_eps['padj']<0.05)).sum()}")

    # PPs analysis
    log("\nPPs WT vs KO")
    pps_samples = ["PPs-WT-Rep1", "PPs-WT-Rep2", "PPs-WT-Rep3",
                   "PPs-KO-Rep1", "PPs-KO-Rep2", "PPs-KO-Rep3"]
    pps_meta = pd.DataFrame({
        "condition": ["WT", "WT", "WT", "KO", "KO", "KO"],
    }, index=pps_samples)
    counts_pps = load_salmon_to_gene_counts(QUANTS, pps_samples, tx2gene, log)
    log(f"  counts shape: {counts_pps.shape}")
    if len(counts_pps) > 0:
        de_pps = run_de(counts_pps, pps_meta, "condition", ("KO", "WT"), log)
        out = OUT_DIR / "aldous_PPs_WT_vs_KO_DE.tsv"
        de_pps.to_csv(out, sep="\t", index=False)
        log(f"  [SAVE] {out}")
        log(f"  KO vs WT: log2FC>1 & padj<0.05 -> "
            f"up={((de_pps['log2FoldChange']>1)&(de_pps['padj']<0.05)).sum()}, "
            f"down={((de_pps['log2FoldChange']<-1)&(de_pps['padj']<0.05)).sum()}")

    log("[DONE]")


if __name__ == "__main__":
    main()
