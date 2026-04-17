#!/usr/bin/env bash
# EXP 5b. Run Velorama on each subsampled h5ad.
#
# Usage:
#   bash benchmark/validation/exp5b_run_velorama_subsample.sh bonemarrow
#   bash benchmark/validation/exp5b_run_velorama_subsample.sh pancreas
#
# Runtime per run (bonemarrow, 4 GPUs, -cp 4): ~45 min
# Total: 5 subsamples × ~45 min = ~4h on bonemarrow
#
# Assumes Velorama ngpu patch is applied (ngpu=1) and -cp 4 works.

# Note: do NOT use set -e here — Ray workers may error on individual trials
# while the aggregate TSV still gets produced, and we want to attempt all
# seeds rather than aborting at the first failure. Check per-seed TSV
# existence after.
set +e

DS=${1:-bonemarrow}
SEEDS=(42 43 44 45 46)
ROOT=/data/cb/scratch/seokmin/geobridge
SUB_DIR=$ROOT/results/bm_real_comparison/validation/exp5_stability/subsamples
OUT_DIR=$ROOT/results/bm_real_comparison/validation/exp5_stability/velorama_out
mkdir -p $OUT_DIR

# Honor CUDA_VISIBLE_DEVICES if already set in env; else default to 0,1,2,3
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
echo "[info] Using CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

for SEED in "${SEEDS[@]}"; do
    NAME="${DS}_seed${SEED}"
    H5AD="$SUB_DIR/${NAME}.h5ad"
    if [ ! -f "$H5AD" ]; then
        echo "[SKIP] $H5AD not found"
        continue
    fi
    SEED_OUT="$OUT_DIR/$NAME"
    mkdir -p "$SEED_OUT"

    EXPECTED="$SEED_OUT/${NAME}.rna_velocity.velorama.interactions.tsv"
    if [ -f "$EXPECTED" ]; then
        echo "[SKIP] already done: $EXPECTED"
        continue
    fi

    echo "=== Running Velorama on $NAME ==="
    # NOTE: narrow lambda range (-ls -2.3 -le -0.7 -nl 19) caused
    # estimate_interactions to fail on cell-subsampled data (all trials
    # filtered out by nnz_percent threshold). Using default lambda range
    # (velorama picks a broad grid) for robustness.
    velorama \
        -n "$NAME" -ds "$NAME" -dyn rna_velocity -dev cuda \
        -l 5 -hd 32 -mi 1000 -cp 4 \
        -rd "$SUB_DIR" \
        -sd "$SEED_OUT" \
        2>&1 | tee "$OUT_DIR/${NAME}.log"
done

echo "[DONE] Velorama subsample runs for $DS"
