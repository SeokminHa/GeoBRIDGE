#!/usr/bin/env bash
# Wessels GSE150062 preprocessing on oni.
# Run inside a tmux or screen — this is a multi-hour pipeline.
#
# Prereqs: sra-tools, cellranger (or starsolo), velocyto, samtools in PATH or modules.
set -euo pipefail

RAW=data/raw/wessels
SRR_LIST=${SRR_LIST:-$RAW/SRR_list.txt}
REF=${CELLRANGER_REF:?"set CELLRANGER_REF to a refdata-cellranger-* directory"}
THREADS=${THREADS:-16}

if [ ! -f "$SRR_LIST" ]; then
    echo "Populate $SRR_LIST with one SRR accession per line (from GSE150062)." >&2
    exit 1
fi

mkdir -p "$RAW"/{sra,fastq,counts,loom}

# 1) prefetch
while read -r srr; do
    [ -z "$srr" ] && continue
    [ -d "$RAW/sra/$srr" ] && { echo "[skip] $srr"; continue; }
    prefetch --max-size 500g -O "$RAW/sra" "$srr"
done < "$SRR_LIST"

# 2) fasterq-dump
while read -r srr; do
    [ -z "$srr" ] && continue
    [ -f "$RAW/fastq/${srr}_1.fastq.gz" ] && { echo "[skip dump] $srr"; continue; }
    fasterq-dump --split-files --threads "$THREADS" --outdir "$RAW/fastq" \
        "$RAW/sra/$srr/$srr.sra"
    gzip -f "$RAW/fastq/${srr}"_*.fastq
done < "$SRR_LIST"

# 3) cellranger count — adapt the fastq naming convention to 10x format before this step
# (Wessels used 10x 3' v3; follow GEO's sample sheet to rename {srr}_1 -> sample_S1_L001_R1_001.fastq.gz etc.)
echo "[note] cellranger step requires manual fastq renaming per sample. See GSE150062 metadata." >&2

# 4) velocyto per sample
for sample_dir in "$RAW"/counts/*/; do
    sample=$(basename "$sample_dir")
    [ -f "$RAW/loom/${sample}.loom" ] && { echo "[skip loom] $sample"; continue; }
    velocyto run10x -@ "$THREADS" "$sample_dir" "$REF"/genes/genes.gtf
    mv "$sample_dir"/velocyto/*.loom "$RAW/loom/${sample}.loom"
done

echo "[next] run: python scripts/preprocess_wessels.py --config configs/wessels.yaml"
