#!/usr/bin/env bash
# cardiac_bam_worker.sh <SRR> <bam_basename> [threads] [workroot]
# de Soysa GSE126128 WT 10x v2 cellranger BAM -> bamtofastq -> kb count nac (10xv2) -> post (GEO-passed cells + gem verification)
# BAM/FASTQ/bus are INTERMEDIATE and are deleted only after cardiac_bam_post.py succeeds.  Idempotent: skips a run whose receipt exists.
set -euo pipefail
RUN=$1; NAME=$2; T=${3:-32}; WR=${4:-/var/tmp/geobridge_cardiac}
G=${GEOBRIDGE_ROOT}
OUT=$G/data/raw/mouse_cardiac
PY=${GB_JUPYTER_ENV_BIN}
REF=$G/refs/kb_nac_GRCm38
B2FQ=$G/GB_real_v2/third_systems/bcl11b_bam/bamtofastq
W=$WR/$RUN; mkdir -p $W $OUT/kb/$RUN $OUT/logs
LOG=$OUT/logs/$RUN.worker.log
exec >> $LOG 2>&1
echo "[$(date -Is)] START $RUN $NAME on $(hostname)"
[ -s $OUT/receipts/$RUN.receipt.json ] && { echo "receipt exists, skip"; exit 0; }
URL=https://sra-pub-src-1.s3.amazonaws.com/$RUN/${NAME}_possorted_genome_bam.bam.1
SZ=$(curl -sI $URL | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
echo "expected bytes $SZ"
for i in $(seq 1 30); do
  cur=$(stat -c %s $W/x.bam 2>/dev/null || echo 0)
  [ "$cur" = "$SZ" ] && break
  curl -fsSL -C - --retry 5 --retry-delay 5 -o $W/x.bam $URL || sleep 10
done
[ "$(stat -c %s $W/x.bam)" = "$SZ" ] || { echo "DOWNLOAD FAILED"; exit 1; }
echo "[$(date -Is)] downloaded"
if [ ! -f $W/fq.done ]; then
  rm -rf $W/fq; $B2FQ --nthreads=16 $W/x.bam $W/fq && touch $W/fq.done
fi
echo "[$(date -Is)] bamtofastq done"
FQ=""
for d in $W/fq/*/; do for r1 in $(ls $d*_R1_001.fastq.gz | sort); do FQ="$FQ $r1 ${r1/_R1_/_R2_}"; done; done
if [ ! -f $W/kb/counts_unfiltered/cells_x_genes.mature.mtx ]; then
  rm -rf $W/kb
  $PY/kb count -i $REF/index.idx -g $REF/t2g.txt -x 10xv2 --workflow nac -c1 $REF/cdna_t2c.txt -c2 $REF/nascent_t2c.txt \
     --h5ad -o $W/kb -t $T -m 100G $FQ
fi
echo "[$(date -Is)] kb count done"
$PY/python $G/scratch/cardiac_bam_post.py $RUN $NAME $W/kb $OUT
echo "[$(date -Is)] post done; cleaning intermediates"
rm -rf $W
echo "[$(date -Is)] DONE $RUN"
