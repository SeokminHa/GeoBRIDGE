#!/usr/bin/env bash
# cardiac_lz_dl.sh <SRR> : download ENA paired FASTQ for a Luna-Zurita 2016 RNA-seq run (retained raw input), verify by gzip -t + read count vs ENA
set -uo pipefail
RUN=$1; RC=$2   # expected read pairs
D=${GEOBRIDGE_ROOT}/data/raw/mouse_cardiac/lz_rnaseq/fastq; mkdir -p $D
sub=00${RUN: -1}; base=https://ftp.sra.ebi.ac.uk/vol1/fastq/${RUN:0:6}/$sub/$RUN
for m in 1 2; do
  f=$D/${RUN}_$m.fastq.gz
  for i in $(seq 1 40); do
    if [ -f $f.ok ]; then break; fi
    curl -fsSL -C - --retry 5 --retry-delay 5 -o $f $base/${RUN}_$m.fastq.gz && gzip -t $f && touch $f.ok || sleep 10
  done &
done
wait
for m in 1 2; do f=$D/${RUN}_$m.fastq.gz; [ -f $f.ok ] || { echo "FAIL $f"; exit 1; }; n=$(zcat $f | wc -l); echo "$RUN mate$m reads $((n/4)) expected $RC"; done > $D/$RUN.verify.txt
echo "DONE $RUN" >> $D/$RUN.verify.txt
