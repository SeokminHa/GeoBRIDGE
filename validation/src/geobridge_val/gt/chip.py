"""ChIP-seq GT builder. Input: peak BEDs per TF. Output: (tf, target) edge set.

Peak-to-gene: nearest TSS within ±window. Good-enough for benchmarking; swap to
ABC or cicero if you have matched scATAC.
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def build_chip_edges_from_bed(
    bed_by_tf: dict[str, str | Path],
    tss_table: pd.DataFrame,          # columns: gene, chrom, tss, strand
    window: int = 10_000,
) -> pd.DataFrame:
    """Assign each peak to genes whose TSS is within ±window. Return long-format edges."""
    try:
        import pybedtools            # lazy
    except ImportError as e:
        raise ImportError("pip install pybedtools (requires bedtools on PATH)") from e

    tss = tss_table.copy()
    tss["start"] = (tss["tss"] - window).clip(lower=0)
    tss["end"]   = tss["tss"] + window
    tss_bed = pybedtools.BedTool.from_dataframe(
        tss[["chrom", "start", "end", "gene"]]
    )

    rows = []
    for tf, bed in bed_by_tf.items():
        peaks = pybedtools.BedTool(str(bed))
        hits = peaks.intersect(tss_bed, wa=True, wb=True).to_dataframe()
        if hits.empty:
            continue
        genes = hits.iloc[:, -1].unique().tolist()
        rows.extend((tf, g) for g in genes)
    return pd.DataFrame(rows, columns=["tf", "target"]).drop_duplicates()


def load_chip_edges(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    return df.rename(columns={c: c.lower() for c in df.columns})[["tf", "target"]]
