#!/usr/bin/env bash
# waits for the salmon index and each run's verified FASTQ, then quantifies (3 at a time, 16 threads each)
B=${GEOBRIDGE_ROOT}/data/raw/mouse_cardiac
S=${GEOBRIDGE_ROOT}/scratch
until grep -q INDEX_DONE $B/logs/salmon_index.log; do sleep 20; done
RUNS="SRR2177623 SRR2177624 SRR2177625 SRR2177629 SRR2177630 SRR2177631 SRR2177635 SRR2177636 SRR2177637"
running=0
for r in $RUNS; do
  until [ -f $B/lz_rnaseq/fastq/${r}_1.fastq.gz.ok ] && [ -f $B/lz_rnaseq/fastq/${r}_2.fastq.gz.ok ]; do sleep 20; done
  bash $S/cardiac_lz_quant.sh $r 16 > $B/logs/quant_$r.log 2>&1 &
  running=$((running+1)); [ $((running % 3)) -eq 0 ] && wait
done
wait
echo QUANT_ALL_DONE > $B/logs/quant_all.done
