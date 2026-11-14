#!/bin/bash
# Launch CPU and GPU workers on GB_CPU_HOSTS and GB_GPU_HOSTS.
# Usage: GEOBRIDGE_ROOT=... GB_CPU_HOSTS="hostA:5 hostB:5" GB_GPU_HOSTS="hostC:0 hostC:1" bash run_tf32rep.sh  (host 'localhost' runs without ssh)
# CPU hosts run GeoBRIDGE before external comparators; GPU workers run concurrently.
# Workers detach with setsid; grn_seeds.json is required and STOP must be absent.
R=${GEOBRIDGE_ROOT}; ROOT=$R/results/sergio_tf32rep_2026_10_22
PY="${GB_JUPYTER_ENV_BIN}/python -s"; D=$R/scratch/gb_tf32rep.py
ENVV="MPLCONFIGDIR=/tmp PYTHONNOUSERSITE=1"; EXTSLOTS=${EXTSLOTS:-2}; CPUEXT=genie3,grnboost2,pidc,pearson; GPUM=velorama,deepsem,regdiffusion,fasttenet
launch() {  # host tag "command"
  C="mkdir -p $ROOT/_logs && cd $R && setsid nohup bash -c '$3' > $ROOT/_logs/launcher_fast_$2.log 2>&1 < /dev/null & disown; echo launched $1 $2"
  if [ "$1" = localhost ]; then bash -c "$C"; else ssh "$1" "$C"; fi   # host "localhost": run directly, no ssh
}
cpu() { launch "$1" "cpu_$1" "env $ENVV $PY $D worker $2; env $ENVV $PY $D cpuext $EXTSLOTS $CPUEXT"; }
gpu() { launch "$1" "gpu_$1_$2" "env $ENVV $PY $D gpuworker $2 $GPUM"; }
( test -f $ROOT/grn_seeds.json && ! test -f $ROOT/STOP ) || { echo "no grn_seeds.json or STOP present; abort"; exit 1; }
for hs in ${GB_CPU_HOSTS:?set GB_CPU_HOSTS="host:slots ..."}; do cpu "${hs%%:*}" "${hs##*:}"; done
for hg in ${GB_GPU_HOSTS:?set GB_GPU_HOSTS="host:gpu_index ..."}; do gpu "${hg%%:*}" "${hg##*:}"; done
# monitoring:  $PY $D status      retry failed:  add --retry-failed to a worker command      stale locks after a crash (no workers alive):  $D unlock
