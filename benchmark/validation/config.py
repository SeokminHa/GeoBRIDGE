"""Shared configuration for validation experiments."""
from pathlib import Path

# Root of results directory
RESULTS_ROOT = Path("results/bm_real_comparison")
PREPARED_DIR = RESULTS_ROOT / "prepared_h5ad"
CSV_DIR = RESULTS_ROOT / "csv"
VELO_DIR = RESULTS_ROOT / "velorama_output"

# Validation outputs
VALIDATION_DIR = RESULTS_ROOT / "validation"
EXP1_DIR = VALIDATION_DIR / "exp1_sign_flip"
EXP2_DIR = VALIDATION_DIR / "exp2_stats"
EXP3_DIR = VALIDATION_DIR / "exp3_pancreas_ortholog"
EXP4_DIR = VALIDATION_DIR / "exp4_motif"
EXP5_DIR = VALIDATION_DIR / "exp5_stability"

DATASETS = ["bonemarrow", "pancreas"]

DATASET_ORGANISM = {
    "bonemarrow": "human",
    "pancreas": "mouse",
}

# Per-dataset featured TFs (used for motif enrichment and heatmaps)
FEATURED_TFS = {
    "bonemarrow": ["GATA1", "GATA2", "SPI1", "FLI1", "ERG"],
    "pancreas": ["NEUROG3", "PDX1", "NKX6-1", "PAX4", "ARX"],
}

# GeoBRIDGE k values
K_VALUES = [0, 1, 3, 5, 10, 20, 50]
K_FOCUS = 5  # primary operating point for statistical tests

# Stability pilot
STABILITY_N_SUBSAMPLES = 5
STABILITY_SUBSAMPLE_FRAC = 0.80
STABILITY_TOP_K = 50
STABILITY_SEEDS = [42, 43, 44, 45, 46]

# Minimum TF size for GT
MIN_TARGETS_PER_TF = 5


def ensure_dirs():
    for d in [VALIDATION_DIR, EXP1_DIR, EXP2_DIR, EXP3_DIR, EXP4_DIR, EXP5_DIR]:
        d.mkdir(parents=True, exist_ok=True)
