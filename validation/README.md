# GeoBRIDGE validation

Real-data validation for GeoBRIDGE (velocity-based causal GRN inference).

## Plan

| Figure | Validation | Dataset | Status |
|---|---|---|---|
| V1 | DoRothEA / CollecTRI enrichment | BM (primary), Pancreas (secondary) | Main |
| V2 | Cell-type-matched ChIP-seq AUPRC / EPR | BM, Pancreas | Main |
| V3 | Perturb-seq target-set enrichment | Wessels CRISPRi (GSE150062) | Main (needs preprocessing) |
| V4 | Motif enrichment at targets (non-circular) | BM | Secondary |
| V5 | Known lineage-driver TF recovery | BM, Pancreas | Supplement |
| V6 | Dynamic-consistency diagnostic | all | Interpretability (not validation) |

Baselines: `correlation`, `grnboost` (GRNBoost2/arboreto), `velorama` (critical dynamic-GRN comparator), `geobridge`.

**Success criteria**: GeoBRIDGE beats correlation + GRNBoost + Velorama on V1+V2 (Δ-lift with bootstrap CIs), plus per-TF lift > ~1.5× and significant pooled enrichment on V3.

## Layout

```
configs/       # per-dataset YAMLs
data/raw/      # scvelo downloads, ChIP beds, DoRothEA tsv
data/interim/  # preprocessed h5ads
data/gt/       # DoRothEA tierA, ChIP-derived edge lists, Lambert TF list
src/geobridge_val/
  io.py        # unified ranking parquet: (regulator, target, score, rank, method, dataset)
  methods/     # one .run(adata, tfs, **cfg) -> DataFrame per method
  gt/          # loaders for DoRothEA, ChIP
  eval/        # enrichment (GSEA preranked), auprc, nulls
scripts/       # CLI entry points
scripts/server/# oni.csail setup + launchers
results/       # rankings, v1_enrichment, v2_chip, figures
logs/
```

All methods output the same long-format parquet. All evaluators consume that parquet. Adding a new method = writing one file in `src/geobridge_val/methods/`, not touching evaluators.

## Quickstart (local)

```bash
pip install -e .
python scripts/download_gt.py
python scripts/qc_velocity.py --config configs/bm.yaml
python scripts/run_methods.py --config configs/bm.yaml \
    --methods correlation,geobridge,grnboost,velorama \
    --out results/rankings/
python scripts/run_v1_enrichment.py \
    --rankings "results/rankings/*_bm.parquet" \
    --gt data/gt/dorothea_tierA.tsv \
    --out results/v1_enrichment/bm/
```

## Running on oni.csail.mit.edu

```bash
# one-time setup
bash scripts/server/setup_env.sh
# sync from local
bash scripts/server/sync_to_oni.sh
# launch overnight job on server
ssh oni "cd ~/geobridge_validation && bash scripts/overnight_phase1.sh bm"
```

See `scripts/server/` for details.

## Status

- Phase 1 (BM + pancreas V1/V2): scripts ready, configs ready, awaiting first overnight run.
- Phase 2 (Wessels V3): `preprocess_wessels.py` skeleton; BAM download is the rate-limiting step.
