# Validation experiments — README

Five experiments that progressively validate the real-data comparison
between GeoBRIDGE and Velorama.

## Execution order

```
EXP1 ──┐
EXP2   ├── serial, cheap (~3h total)
EXP3 ──┘
        │
        ▼
EXP4 ──┐
EXP5a  ├── parallel (different resources)
EXP5b──│
EXP5c──│
EXP5d ─┘── after 5a,b,c complete
```

## Scripts

| Script | Purpose | Runtime | Deps |
|---|---|---|---|
| `exp1_sign_flip.py` | Check whether Velorama AUROC<0.5 is a sign bug | ~2 min | existing CSVs + Velorama TSVs |
| `exp2_statistical_tests.py` | Wilcoxon p-values (GB>0.5, Velo<0.5, GB>Velo) | ~1 min | `real_comparison_full.csv` |
| `exp3_pancreas_ortholog.py` | Replace uppercase hack with real mouse GT via mygene | ~1-2 h | `pip install mygene` |
| `exp4_motif_enrichment.py` | Motif enrichment via MSigDB TFT (Fisher's exact) | ~30 min | `pip install gseapy` |
| `exp5a_prepare_subsamples.py` | Generate 5 × 80% cell subsample h5ads per dataset | ~5 min | scvelo |
| `exp5b_run_velorama_subsample.sh` | Run Velorama on each subsample | ~5-10 GPU-h | GPU + patched velorama |
| `exp5c_run_geobridge_subsample.py` | Run GeoBRIDGE on each subsample at K_FOCUS | ~1-2 h | CPU |
| `exp5d_aggregate_stability.py` | Jaccard stability per TF, paired Wilcoxon | ~5 min | none |

## Outputs

All written under `results/bm_real_comparison/validation/`:

```
validation/
  exp1_sign_flip/       REPORT.md + CSVs
  exp2_stats/           REPORT.md + statistical_tests.csv
  exp3_pancreas_ortholog/
                        human_to_mouse_map.json, collectri_mouse.csv,
                        pancreas_results_new_gt.csv, old_vs_new_gt_comparison.csv,
                        REPORT.md
  exp4_motif/           motif_enrichment.csv, REPORT.md, top_lists/
  exp5_stability/       subsamples/, velorama_out/, geobridge_out/,
                        stability_per_tf.csv, REPORT.md
```

## Execution commands (charybdis)

```bash
cd /data/cb/scratch/seokmin/geobridge
source venv_gpu/bin/activate
git pull

# Immediate (serial, ~2-3h)
python -u -m benchmark.validation.exp1_sign_flip
python -u -m benchmark.validation.exp2_statistical_tests
pip install --no-cache-dir mygene
python -u -m benchmark.validation.exp3_pancreas_ortholog

# In parallel (two tmux sessions)
# Session A: motif (CPU, ~30 min)
python -u -m benchmark.validation.exp4_motif_enrichment

# Session B: stability (GPU + CPU, ~8 h)
python -u -m benchmark.validation.exp5a_prepare_subsamples
bash benchmark/validation/exp5b_run_velorama_subsample.sh bonemarrow
python -u -m benchmark.validation.exp5c_run_geobridge_subsample --dataset bonemarrow
python -u -m benchmark.validation.exp5d_aggregate_stability
```

## Success / failure criteria

### EXP 1 — sign flip
- **POSITIVE for paper (sign flip fails to rescue Velorama)**: Velorama
  signed AUROC ≈ 0.45 AND −signed AUROC ≈ 0.55 with neither far from 0.5
  → Velorama scores have no directional meaning at this hyperparameter.
- **BLOCKING FAILURE**: Velorama signed AUROC ≈ 0.45 AND −signed AUROC > 0.55
  → sign convention is backwards. Must flip all Velorama scores
  downstream and rewrite Section 4.

### EXP 2 — statistical tests
- **POSITIVE**: one-sample Wilcoxon p(GB>0.5) < 0.05 for both datasets
  AND paired Wilcoxon p(GB > Velo) < 0.05.
- **WEAK**: p(GB>0.5) < 0.05 but paired p > 0.05 → GB is above random
  but not significantly better than Velorama.
- **NEGATIVE**: neither test significant → cannot claim superiority;
  rework Section 4 to emphasize efficiency only.

### EXP 3 — pancreas ortholog fix
- **POSITIVE**: new GT has ≥30 mapped TFs AND GB AUROC trend consistent
  with bonemarrow (mean > 0.5).
- **MIXED**: new GT smaller/larger than before; metrics shift
  significantly → report both old and new as supplementary; discuss.
- **NEGATIVE**: mapping yields <15 usable TFs → too few for per-TF
  testing; treat pancreas as supplementary and base claims on bonemarrow.

### EXP 4 — motif enrichment
- **POSITIVE**: GB top-50 shows Fisher p_adj < 0.05 for the TF's own
  motif-target set for ≥3/5 featured TFs per dataset AND GB odds ratio
  > Velorama odds ratio for ≥2/5 TFs.
- **MIXED**: GB is enriched for some TFs but not systematically better
  than Velorama → motif is a noise-reducing step common to both.
  Report but don't lead with this.
- **NEGATIVE**: no systematic enrichment for either method → drop the
  motif claim. Keep top lists in Supplementary for transparency.

### EXP 5 — stability pilot
- **SCALE-UP**: paired (GB_Jaccard − Velo_Jaccard) ≥ 0.05, p(GB>V) < 0.1
  in ≥1 dataset → expand to 20 subsamples on both datasets; stability
  becomes a paper claim.
- **DROP**: paired difference < 0.05 or p > 0.1 → both methods equally
  stable (or unstable); drop the stability language from the paper.

## What is deliberately NOT included

- Go enrichment: conditionally deferred. Re-evaluate only if EXP 4
  shows motif enrichment is positive.
- TF activity / regulon inference via decoupler: depends on EXP 4
  passing.
- Additional datasets (dentate gyrus, corticogenesis): skip.
- Extended k sweep or finer k grid: skip.
- Network topology (scale-free, feed-forward loops): skip.

## Decision tree

```
After EXP 1–3:
  if EXP 1 FAILS (flip rescues Velorama):
      → rewrite Section 4 with flipped Velorama; all downstream redo
  if EXP 2 NEGATIVE:
      → paper fallback: methods + synthetic + efficiency only
  if EXP 3 NEGATIVE:
      → use bonemarrow only; pancreas as supplementary caveat

After EXP 4:
  if POSITIVE:  add motif section, lead real-data claim
  if NEGATIVE:  drop featured-TF biology language entirely

After EXP 5 pilot:
  if SCALE-UP: expand then add stability claim to abstract
  if DROP:      remove all "stable/consistent" language from draft
```
