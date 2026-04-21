#!/usr/bin/env bash
# rsync the project (code only, not data/) to oni.csail.mit.edu.
#
# Usage:
#   bash scripts/server/sync_to_oni.sh [remote-path]
#
# Set these env vars (or defaults below):
#   ONI_USER   your CSAIL username
#   ONI_HOST   oni.csail.mit.edu
#   ONI_PATH   ~/geobridge_validation
set -euo pipefail

ONI_USER=${ONI_USER:-seokmin}
ONI_HOST=${ONI_HOST:-oni.csail.mit.edu}
ONI_PATH=${1:-${ONI_PATH:-~/geobridge_validation}}

REMOTE="${ONI_USER}@${ONI_HOST}:${ONI_PATH}"

echo "[sync] -> ${REMOTE}"
rsync -avh --delete \
    --exclude='.git/' \
    --exclude='data/raw/' \
    --exclude='data/interim/' \
    --exclude='results/' \
    --exclude='logs/' \
    --exclude='__pycache__/' \
    --exclude='*.egg-info/' \
    ./ "${REMOTE}/"

echo "[sync] done. On oni:"
echo "  ssh ${ONI_USER}@${ONI_HOST}"
echo "  cd ${ONI_PATH} && bash scripts/server/setup_env.sh"
echo "  nohup bash scripts/overnight_phase1.sh bm > logs/bm_\$(date +%Y%m%d_%H%M).log 2>&1 &"
