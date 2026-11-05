#!/usr/bin/env bash
# cardiac_bam_dl.sh <SRR> <bam_basename> [workroot]: resumable AWS download only (the worker skips the download when the file is complete)
set -uo pipefail
RUN=$1; NAME=$2; WR=${3:-/var/tmp/geobridge_cardiac}; W=$WR/$RUN; mkdir -p $W
URL=https://sra-pub-src-1.s3.amazonaws.com/$RUN/${NAME}_possorted_genome_bam.bam.1
SZ=$(curl -sI $URL | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
for i in $(seq 1 30); do
  [ "$(stat -c %s $W/x.bam 2>/dev/null || echo 0)" = "$SZ" ] && break
  curl -fsSL -C - --retry 5 --retry-delay 5 -o $W/x.bam $URL || sleep 10
done
[ "$(stat -c %s $W/x.bam)" = "$SZ" ] && echo "DL_OK $RUN $SZ" || echo "DL_FAIL $RUN"
