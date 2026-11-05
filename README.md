# GeoBRIDGE

Code accompanying *Inferring Direct Causal Regulation in Branching Lineages via Geometric Deconfounding* (S. Ha and B. Berger, RECOMB 2027 submission).

GeoBRIDGE (**Geo**metric **B**ranch-**R**esolved **I**nference of **D**irect **G**ene-regulatory **E**dges) prioritizes candidate regulator–target interactions in branching differentiation lineages from **unperturbed** single-cell RNA-seq data with spliced and unspliced counts.

## Method in brief

1. **Cell pairs.** Cells are ordered by diffusion pseudotime from two data-defined roots, and forward cell pairs are sampled within a fixed pseudotime window.
2. **Association.** For each pair, the change in a regulator's smoothed *spliced* abundance is compared with the change in a target's smoothed *unspliced* abundance.
3. **Conditioning in pair space.** Differences of the leading 30 expression principal components are QR-factorized, and both difference vectors are projected onto the orthogonal complement of the first *k* directions. This removes shared developmental variation in the same pair geometry used for scoring and gives a concordance S(k) in [−1, 1].
4. **Aggregation.** Concordance magnitudes are standardized within each regulator, root and depth, averaged over depths k = 0–8, and the minimum over the two roots gives the unsigned score Z.
5. **Sign (optional).** Regulatory sign is estimated separately from near-isochronic cell pairs; the signed score T is the sign times the within-regulator percentile of Z.

Perturbation, ChIP and RNA-velocity estimates are **not** inputs of the score; knockout and ChIP data are used only to evaluate frozen scores.

## Repository layout

```
geobridge/            reference implementation of the score (Z, signed T), command-line interface, tests and a small fixture   <- start here
data/                 data accessions and download scripts
common/               shared preprocessing and evaluation modules used by several figures
figure2_synthetic/    Fig. 2: SERGIO benchmark (simulation, comparator runs, evaluation, plots) and result tables
figure3_knockout/     Fig. 3 and Table 1: real-data preprocessing, knockout labels, evaluation, plots and result tables
figure4_signed/       Fig. 4: signed score, knockout x ChIP classes, trend test, plots and result tables
table1/               generation of Table 1 from the shipped result table
supplementary/        Supplementary figures: scripts and result tables
patches/              compatibility patch for the SERGIO simulator
setup_workspace.sh    builds the workspace layout expected by the figure scripts
MANIFEST.md           every main-text panel -> producing script -> input tables
```

## Installation

```bash
conda env create -f environment.yml
conda activate geobridge          # Python 3.10; scVelo 0.3.4, Scanpy 1.11, NumPy 1.26 (exact versions in environment.yml)
pip install -e ./geobridge        # already included in environment.yml
```

scVelo 0.3.4 is required, because the preprocessing uses its moment computation. Comparator methods (Velorama, FastTENET, RegDiffusion, DeepSEM) need separate environments; see the header of `environment.yml`.

## Quick start

```bash
cd geobridge
PYTHONPATH=src pytest tests       # tests on the bundled synthetic fixture (seconds)

# score the bundled fixture (400 cells x 60 genes)
geobridge score fixture/synthetic_smoke.h5ad --canonical --pairs pairs.npz \
    --regulators G0,G1,G2 --stratum-key embryo --out out.tsv

# score a dataset from raw spliced/unspliced counts
geobridge score raw.h5ad --species mouse --regulators RFX6 --stratum-key clusters --out o.tsv
# -> o.tsv (unsigned score Z), o.d_iso.tsv (sign), o.T.tsv (signed score T)
```

Input requirements, output columns and missing-score rules are described in `geobridge/README.md`. Scoring runs on CPU; the pancreatic dataset (3,696 cells) takes about 6 min and 4.4 GB of memory.

## Reproducing the figures

```bash
export GEOBRIDGE_ROOT=/path/to/workspace
bash setup_workspace.sh           # links the scripts and the shipped result tables into $GEOBRIDGE_ROOT
```

Each figure folder has a README describing its pipeline. The quantitative panels listed below can be re-plotted directly from the result tables shipped in this repository. Recomputing the tables requires the public raw data (`data/README.md`) and, for the synthetic benchmark, regenerating the simulated datasets from the provided seeds; the comparator methods additionally require GPUs.

| Figure | Folder | Re-plot from shipped tables | Full recomputation |
|---|---|---|---|
| Fig. 2 d–f | `figure2_synthetic/` | yes | simulate 6 networks x 4 conditions x 2 replicates, run all methods |
| Fig. 3 b–g, Table 1 | `figure3_knockout/` | yes | pipeline scripts from public single-cell and knockout data (folder README) |
| Fig. 4 | `figure4_signed/` | yes | pipeline scripts from public single-cell, knockout and ChIP data (folder README) |
| Supplementary figures | `supplementary/` | yes | as above |

Internal evaluation scripts check SHA-256 pins of source files. Set `GB_SKIP_PREFLIGHT=1` to skip the pins that refer to files not distributed here; all data-integrity checks remain active.

## Data availability

All data are public (details in `data/README.md`):

- Pancreatic endocrinogenesis (GSE132188) and human bone marrow (Setty et al. 2019), as distributed with scVelo
- Mouse cardiac development (de Soysa et al. 2019; GSE126128)
- RFX6 knockout (Zenodo 10.5281/zenodo.10656891), GATA1 knockout (GSE274110), TBX5 and NKX2-5 knockouts (GSE77547)
- TBX5 and NKX2-5 ChIP-exo (GSE72223), RFX6 ChIP-seq (GSE62844), CellMarker 2.0

## Citation

If you use GeoBRIDGE, please cite:

> S. Ha and B. Berger. Inferring Direct Causal Regulation in Branching Lineages via Geometric Deconfounding. RECOMB 2027 (submitted).

## License

See `LICENSE`. Third-party software (SERGIO, Velorama, FastTENET, RegDiffusion, DeepSEM, arboreto) is installed from upstream and retains its own license.
