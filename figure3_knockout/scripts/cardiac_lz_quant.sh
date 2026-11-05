#!/usr/bin/env bash
# cardiac_lz_quant.sh <SRR> [threads]: salmon quant (decoy-aware GRCm38 Ensembl-102 index, same annotation as the kb nac reference)
set -euo pipefail
RUN=$1; T=${2:-12}
B=${GEOBRIDGE_ROOT}/data/raw/mouse_cardiac/lz_rnaseq
P=${GB_JUPYTER_ENV_BIN}
[ -f $B/fastq/${RUN}_1.fastq.gz.ok ] && [ -f $B/fastq/${RUN}_2.fastq.gz.ok ] || { echo "fastq not verified"; exit 1; }
[ -s $B/quant/$RUN/quant.sf ] && { echo "exists"; exit 0; }
mkdir -p $B/quant
$P/salmon quant -i $B/salmon_index_GRCm38_e102/idx -l A -1 $B/fastq/${RUN}_1.fastq.gz -2 $B/fastq/${RUN}_2.fastq.gz \
   -p $T --validateMappings --gcBias --seqBias -o $B/quant/$RUN
