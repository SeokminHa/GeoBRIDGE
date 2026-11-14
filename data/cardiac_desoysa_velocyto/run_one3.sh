#!/usr/bin/env bash
# run_one3.sh <SRR> <bam_basename> <gem> [threads] [workroot]     env: NCHUNK (default 4), GATE (file that must exist before the velocyto stage)
# Wait for GATE if set, then count disjoint barcode chunks with velocyto.
set -euo pipefail
RUN=$1; NAME=$2; GEM=$3; T=${4:-8}; WR=${5:-/var/tmp/geobridge_cardiacv}; NCH=${NCHUNK:-4}; GATE=${GATE:-}
G=${GEOBRIDGE_ROOT}; D=$G/data/raw/mouse_cardiac_velocyto
J=${GB_JUPYTER_ENV_BIN}
VEL=${GB_BMDCUS_BIN}/velocyto
SLOTS=${SLOTS:-5}; SD=/var/tmp/geobridge_slots; mkdir -p $SD
# Limit concurrent counting jobs; deep libraries can use about 10 GB per process.
acquire_slot() { while true; do for k in $(seq 1 $SLOTS); do exec {SFD}>$SD/$(hostname -s).$k; if flock -n $SFD; then return 0; fi; exec {SFD}>&-; done; sleep 20; done; }
W=$WR/$RUN; mkdir -p $W $D/loom $D/logs
LOG=$D/logs/$RUN.run.log; exec >> $LOG 2>&1
echo "[$(date -Is)] START $RUN $NAME gem $GEM on $(hostname) NCHUNK=$NCH"
[ -s $D/receipts/$RUN.receipt.json ] && { echo "receipt exists, skip"; exit 0; }
awk -v g="-$GEM" '{n=length($1); if(substr($1,n-length(g)+1)==g){print substr($1,1,n-length(g)) "-1"}}' $G/data/raw/mouse_cardiac/geo_cells.txt | sort > $W/bc.tsv
echo "barcodes: $(wc -l < $W/bc.tsv)"
$D/scripts/dl.sh $RUN $NAME $WR | tail -1
[ "$(stat -c %s $W/x.bam)" -gt 1000000000 ]
SB=$W/cellsorted_x.bam
if [ ! -f $W/sort.done ]; then
  $J/samtools view -@ $T -D CB:$W/bc.tsv -u $W/x.bam 2>/dev/null | $J/samtools sort -t CB -m 3G -@ $T -o $SB - 2>/dev/null
  touch $W/sort.done
fi
echo "[$(date -Is)] sorted: $(stat -c %s $SB) bytes"
if [ -n "$GATE" ]; then until [ -f "$GATE" ]; do sleep 30; done; echo "[$(date -Is)] gate open"; fi
if [ ! -f $W/vel.done ]; then
  rm -rf $W/vel $W/chunk_*; mkdir -p $W/vel
  split -n l/$NCH -d $W/bc.tsv $W/bcpart_
  PIDS=""
  for p in $W/bcpart_*; do
    i=${p##*_}; C=$W/chunk_$i; mkdir -p $C
    ( $J/samtools view -@ 2 -D CB:$p -b -o $C/cellsorted_x.bam $SB 2>/dev/null; ln $W/x.bam $C/x.bam   # Use a hard link: velocyto resolves symlinks before finding the cell-sorted BAM.
      acquire_slot; $VEL run -b $p -o $C/vel -e ${RUN}c$i -m $D/ref/mm10_rmsk.gtf -l Default -vv $C/x.bam $D/ref/genes_chr.gtf > $C/vel.log 2>&1 && cp $C/vel/*.loom $D/loom/${RUN}_c$i.loom && touch $C/ok ) &
    PIDS="$PIDS $!"
  done
  for pid in $PIDS; do wait $pid || true; done
  for p in $W/bcpart_*; do [ -f $W/chunk_${p##*_}/ok ] || { echo "CHUNK FAILED ${p##*_}"; exit 1; }; done
  touch $W/vel.done
fi
echo "[$(date -Is)] velocyto done"
LOOMS=$(ls $D/loom/${RUN}_c*.loom | tr "\n" "," | sed 's/,$//')
$J/python $D/scripts/post.py $RUN $NAME $GEM $LOOMS $D
echo "[$(date -Is)] post done; cleaning intermediates"
rm -rf $W
echo "[$(date -Is)] DONE $RUN"
