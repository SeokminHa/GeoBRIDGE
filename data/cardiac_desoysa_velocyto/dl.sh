#!/usr/bin/env bash
# dl.sh <SRR> <name> [root] [suffix]: resumable stream download of the authors' cellranger BAM (AWS SRA original-format mirror)
set -uo pipefail
RUN=$1; NAME=$2; WR=${3:-/var/tmp/geobridge_cardiacv}; SUF=${4:-_possorted_genome_bam.bam.1}; W=$WR/$RUN; mkdir -p $W
URL=https://sra-pub-src-1.s3.amazonaws.com/$RUN/${NAME}${SUF}
SZ=$(curl -sI $URL | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
for i in $(seq 1 30); do
  [ "$(stat -c %s $W/x.bam 2>/dev/null || echo 0)" = "$SZ" ] && break
  curl -fsSL -C - --retry 5 --retry-delay 5 -o $W/x.bam $URL || sleep 10
done
[ "$(stat -c %s $W/x.bam)" = "$SZ" ] && echo "DL_OK $RUN $SZ" || echo "DL_FAIL $RUN"
