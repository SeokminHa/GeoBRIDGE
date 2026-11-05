#!/bin/bash
# queue worker: gb_velorama_worker.sh <gpu> <listfile>   (claims datasets via atomic mkdir lock)
export PYTHONNOUSERSITE=1
R=${GEOBRIDGE_ROOT}; O=$R/results/sergio_drift/topo_adv/velorama; mkdir -p $O/.lock $O/logs
while read d; do
  t=$(basename $d); [ -e $O/$t.npz ] && continue
  mkdir $O/.lock/$t 2>/dev/null || continue
  ${GB_VELORAMA_BIN}/python $R/scratch/gb_velorama_official.py run $R/$d $1 > $O/logs/$t.log 2>&1 || echo "FAIL $t" >> $O/logs/FAILURES.txt
done < $2
