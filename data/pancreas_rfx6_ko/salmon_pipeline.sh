#!/bin/bash
# Salmon quantification for Aldous RFX6 bulk RNA-seq.
# Builds index, quantifies 10 paired-end samples, writes DONE marker.
set -euo pipefail

ENV=${GB_JUPYTER_ENV_BIN%/bin}
SALMON=$ENV/bin/salmon
if [ ! -x "$SALMON" ]; then
    echo "[ERROR] salmon not found at $SALMON"
    exit 2
fi
ALDOUS=${GEOBRIDGE_ROOT}/real_1_Pancreas/external_validation/aldous_rfx6
DATA=$ALDOUS/data
REFS=$ALDOUS/refs
QUANTS=$ALDOUS/quants
mkdir -p "$REFS" "$QUANTS"

echo "[$(date '+%H:%M:%S')] Salmon pipeline"

# Build index from GENCODE v44 transcriptome
INDEX=$REFS/salmon_index_v44
if [ ! -d "$INDEX" ] || [ ! -f "$INDEX/info.json" ]; then
    echo "[$(date '+%H:%M:%S')] Building salmon index..."
    "$SALMON" index --threads 32 -t "$REFS/gencode.v44.transcripts.fa.gz" -i "$INDEX" --gencode -k 31 2>&1 | tail -5
    echo "[$(date '+%H:%M:%S')] Index done"
else
    echo "[$(date '+%H:%M:%S')] Index exists"
fi

# Sample manifest
declare -A SAMPLES=(
    ["EPs-WT-Rep1"]="EPs-WT-Rep1_S7_R1_001.fastq.gz EPs-WT-Rep1_S7_R2_001.fastq.gz"
    ["EPs-WT-Rep2"]="EPs-WT-Rep2_S8_R1_001.fastq.gz EPs-WT-Rep2_S8_R2_001.fastq.gz"
    ["EPs-KO-Rep1"]="EPs-KO-Rep1_S9_R1_001.fastq.gz EPs-KO-Rep1_S9_R2_001.fastq.gz"
    ["EPs-KO-Rep2"]="EPs-KO-Rep2_S10_R1_001.fastq.gz EPs-KO-Rep2_S10_R2_001.fastq.gz"
    ["PPs-WT-Rep1"]="PPs-WT-Rep1_S1_R1_001.fastq.gz PPs-WT-Rep1_S1_R2_001.fastq.gz"
    ["PPs-WT-Rep2"]="PPs-WT-Rep2_S2_R1_001.fastq.gz PPs-WT-Rep2_S2_R2_001.fastq.gz"
    ["PPs-WT-Rep3"]="PPs-WT-Rep3_S3_R1_001.fastq.gz PPs-WT-Rep3_S3_R2_001.fastq.gz"
    ["PPs-KO-Rep1"]="PPs-KO-Rep1_S4_R1_001.fastq.gz PPs-KO-Rep1_S4_R2_001.fastq.gz"
    ["PPs-KO-Rep2"]="PPs-KO-Rep2_S5_R1_001.fastq.gz PPs-KO-Rep2_S5_R2_001.fastq.gz"
    ["PPs-KO-Rep3"]="PPs-KO-Rep3_S6_R1_001.fastq.gz PPs-KO-Rep3_S6_R2_001.fastq.gz"
)

# Quantify each sample
for SNAME in "${!SAMPLES[@]}"; do
    OUTDIR="$QUANTS/$SNAME"
    if [ -f "$OUTDIR/quant.sf" ]; then
        echo "[$(date '+%H:%M:%S')] $SNAME: already quantified"
        continue
    fi
    read R1 R2 <<< "${SAMPLES[$SNAME]}"
    echo "[$(date '+%H:%M:%S')] Quantifying $SNAME..."
    if ! "$SALMON" quant -i "$INDEX" -l A \
        -1 "$DATA/$R1" -2 "$DATA/$R2" \
        -p 32 --gcBias --validateMappings \
        -o "$OUTDIR" 2>&1 | tail -3; then
        echo "[ERROR] salmon failed for $SNAME"
        exit 3
    fi
    if [ ! -f "$OUTDIR/quant.sf" ]; then
        echo "[ERROR] quant.sf not produced for $SNAME"
        exit 4
    fi
    echo "[$(date '+%H:%M:%S')] $SNAME done"
done

touch "$ALDOUS/SALMON_DONE"
echo "[$(date '+%H:%M:%S')] Salmon all done"
ls -la "$QUANTS/"
