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
$PY/samtools view -H $W/x.bam 2>/dev/null | grep -E "^@RG|^@CO" | cut -c1-300 > $OUT/receipts/$RUN.bamheader.txt || true
if [ ! -f $W/fq.done ]; then
  rm -rf $W/fq; $B2FQ --nthreads=16 $W/x.bam $W/fq && touch $W/fq.done
fi
echo "[$(date -Is)] bamtofastq done"
# ALL bamtofastq chunks (bamtofastq splits a read group into *_R1_001, _002, ... at 50M reads: the first version of this worker globbed only *_R1_001 and silently lost reads on the deep NextSeq runs)
FQ=""; n1=0; n2=0
for d in $W/fq/*/; do for r1 in $(ls $d*_R1_[0-9][0-9][0-9].fastq.gz | sort); do r2=${r1/_R1_/_R2_}; [ -s "$r2" ] || { echo "MISSING MATE $r2"; exit 1; }; FQ="$FQ $r1 $r2"; n1=$((n1+1)); done; n2=$((n2+$(ls $d*_R2_[0-9][0-9][0-9].fastq.gz | wc -l))); done
[ "$n1" = "$n2" ] || { echo "R1/R2 file count differs $n1 $n2"; exit 1; }
EXPECT=$(grep -h "Observed" $LOG | tail -1 | awk '{print $4}')
echo "fastq pairs of files: $n1 ; bamtofastq observed read pairs: ${EXPECT:-UNKNOWN}"
if [ ! -f $W/kb/counts_unfiltered/cells_x_genes.mature.mtx ]; then
  rm -rf $W/kb
  $PY/kb count -i $REF/index.idx -g $REF/t2g.txt -x 10xv2 --workflow nac -c1 $REF/cdna_t2c.txt -c2 $REF/nascent_t2c.txt \
     --h5ad -o $W/kb -t $T -m 100G $FQ
fi
echo "[$(date -Is)] kb count done"
$PY/python $G/scratch/cardiac_bam_post.py $RUN $NAME $W/kb $OUT ${EXPECT:-0}
echo "[$(date -Is)] post done; cleaning intermediates"
rm -rf $W
echo "[$(date -Is)] DONE $RUN"
