#!/usr/bin/env bash
# Single-command launcher on oni: activates env, runs Phase-1, keeps a log.
#
# Usage (on oni):
#   bash scripts/server/run_on_oni.sh bm
#   bash scripts/server/run_on_oni.sh pancreas
#
# Detached:
#   nohup bash scripts/server/run_on_oni.sh bm > logs/launch.log 2>&1 &
set -euo pipefail

DATASET=${1:?"usage: $0 {bm|pancreas}"}
ENV_NAME=${ENV_NAME:-geobridge}

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

export PYTHONPATH="$(pwd)/src:$(pwd)/../geobridge_pairspace_experiments_indirect:${PYTHONPATH:-}"

# cap parallelism so we don't nuke shared resources
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

bash scripts/overnight_phase1.sh "$DATASET"
