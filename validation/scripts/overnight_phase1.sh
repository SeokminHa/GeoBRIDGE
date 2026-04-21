#!/usr/bin/env bash
# Overnight Phase-1 job: QC + all methods + V1 + V2 + figures for one dataset.
#
# Usage:
#   bash scripts/overnight_phase1.sh bm
#   bash scripts/overnight_phase1.sh pancreas
#
# Launch detached:
#   nohup bash scripts/overnight_phase1.sh bm > logs/bm_$(date +%Y%m%d_%H%M).log 2>&1 &
set -euo pipefail

DATASET=${1:?"usage: $0 {bm|pancreas}"}
CFG="configs/${DATASET}.yaml"
METHODS=${METHODS:-"correlation,geobridge,grnboost,velorama"}
BOOT=${BOOT:-1000}

mkdir -p logs results
STAMP=$(date +%Y%m%d_%H%M)
LOG="logs/${DATASET}_${STAMP}.log"

echo "=== $(date) :: Phase-1 on $DATASET ===" | tee -a "$LOG"

echo "=== [0/4] QC ===" | tee -a "$LOG"
python scripts/qc_velocity.py --config "$CFG" 2>&1 | tee -a "$LOG"

echo "=== [1/4] methods ===" | tee -a "$LOG"
python scripts/run_methods.py \
    --config "$CFG" \
    --methods "$METHODS" \
    --out results/rankings/ 2>&1 | tee -a "$LOG"

echo "=== [2/4] V1 DoRothEA ===" | tee -a "$LOG"
python scripts/run_v1_enrichment.py \
    --rankings "results/rankings/*_${DATASET}.parquet" \
    --gt "$(python -c "import yaml; print(yaml.safe_load(open('${CFG}'))['gt']['dorothea'])")" \
    --out "results/v1_enrichment/${DATASET}/" 2>&1 | tee -a "$LOG"

echo "=== [3/4] V2 ChIP ===" | tee -a "$LOG"
python scripts/run_v2_chip_eval.py \
    --rankings "results/rankings/*_${DATASET}.parquet" \
    --gt "$(python -c "import yaml; print(yaml.safe_load(open('${CFG}'))['gt']['chip_edges'])")" \
    --bootstrap "$BOOT" \
    --out "results/v2_chip/${DATASET}/" 2>&1 | tee -a "$LOG"

echo "=== [4/4] figures ===" | tee -a "$LOG"
python scripts/plot_main_figs.py --dataset "$DATASET" 2>&1 | tee -a "$LOG"

echo "=== $(date) :: DONE ($DATASET) ===" | tee -a "$LOG"
