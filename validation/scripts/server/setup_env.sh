#!/usr/bin/env bash
# One-time conda env setup on oni.csail.mit.edu.
#
# Usage (on oni):
#   bash scripts/server/setup_env.sh
set -euo pipefail

ENV_NAME=${ENV_NAME:-geobridge}
PY=${PY:-3.11}

if ! command -v conda >/dev/null; then
    echo "conda not found. Load a module or install miniforge first." >&2
    exit 1
fi

source "$(conda info --base)/etc/profile.d/conda.sh"

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    conda create -n "$ENV_NAME" python="$PY" -y
fi
conda activate "$ENV_NAME"

pip install --upgrade pip
pip install -r requirements.txt
pip install -e .

# geobridge core package — in the sibling directory
if [ -d "../geobridge_pairspace_experiments_indirect" ]; then
    pip install -e ../geobridge_pairspace_experiments_indirect || \
        echo "[warn] geobridge core package has no setup; add it to PYTHONPATH instead."
fi

echo "env '$ENV_NAME' ready. To activate:"
echo "  conda activate $ENV_NAME"
