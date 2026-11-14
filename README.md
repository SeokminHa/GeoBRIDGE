# GeoBRIDGE

GeoBRIDGE (Geometric Branch-Resolved Inference of Direct Gene-regulatory Edges) is Python code for prioritizing direct regulator-target interactions in branching differentiation lineages from unperturbed single-cell RNA-seq data with spliced and unspliced counts.

This repository contains the code for the paper "Inferring Direct Causal Regulation in Branching Lineages via Geometric Deconfounding" (S. Ha and B. Berger).

Notation: the paper's $G_{rg}$ is `Z` (`PRIMARY`) in the code; $T_{rg}$ is `T`.

The forward cell pairs that define Z are ordered pairs of cells $(u, v)$ with diffusion pseudotime difference $0.02 < t_v - t_u \le 0.25$ and at most 40 partners per cell, computed separately from two data-defined roots. Forward pairing ignores branch identity and can connect cells from different branches. The near-isochronic pairs used only for the sign are restricted to the same `--stratum-key` group.

## Requirements

- Python 3.10, scVelo 0.3.4, Scanpy 1.11, NumPy 1.26 (full list in `environment.yml`)
- Tested on Linux. Scoring needs only a CPU; no GPU or other special hardware is needed.
- The comparator methods used in the benchmark (Velorama, FastTENET, RegDiffusion, DeepSEM) need separate environments; in the paper they were run on GPUs where the method supports it.

## Installation

```bash
conda env create -f environment.yml
conda activate geobridge
```

This also installs the `geobridge` package in `geobridge/`.

## Tutorial

1. Prepare the data. GeoBRIDGE takes an AnnData file (`.h5ad`) with integer `spliced` and `unspliced` count layers from an unperturbed differentiation dataset.

2. Choose the regulators. Give a comma-separated list of transcription factors. In the paper, the regulator list consisted of catalogue transcription factors with nonzero spliced counts in at least 5% of cells; the package takes the list as input.

3. Run GeoBRIDGE.

   ```bash
   geobridge score data.h5ad --species mouse --regulators RFX6 --stratum-key clusters --out result.tsv
   ```

   `--stratum-key` is the cell annotation used for the signed score (for example, cluster or sample). The command preprocesses the data, builds pseudotime and cell pairs, and computes the scores.

4. Read the output.

   - `result.tsv`: unsigned score Z ($G_{rg}$ in the paper) for each regulator-target pair. Higher Z means a stronger conditioned association.
   - `result.d_iso.tsv`: estimated sign of the observational association (+1 positive, -1 negative).
   - `result.T.tsv`: signed score T, the sign times the within-regulator percentile of Z.

   The signed score does not identify direct activation or repression.

   Pairs without enough support are left empty rather than set to zero.

A small example dataset (already preprocessed, with its cell pairs) is included. It runs in a few seconds:

```bash
cd geobridge
geobridge score fixture/synthetic_smoke.h5ad --canonical --pairs fixture/synthetic_smoke_pairs.npz \
    --regulators G0,G1,G2 --stratum-key embryo --out out.tsv
PYTHONPATH=src pytest tests
```

This writes `out.tsv`, `out.d_iso.tsv` and `out.T.tsv`. Without the installed package, use `PYTHONPATH=src python -m geobridge.cli score ...` instead of `geobridge score ...`.

The pancreatic dataset in the paper (3,696 cells) takes about 7 minutes and 4.4 GB of memory. More details are in `geobridge/README.md`.

## Figures

The code used to make the figures in the paper is in the following folders. Each folder has its own README and the result tables needed to redraw the figure.

- `figure2_synthetic/`: synthetic benchmark (Fig. 2); use `fig2f.py` for panel 2f.
- `figure3_knockout/`: knockout-response evaluation (Fig. 3, Table 1)
- `figure4_signed/`: signed score with knockout and ChIP data (Fig. 4)
- `supplementary/`: supplementary figures

Run `bash setup_workspace.sh` once after setting `GEOBRIDGE_ROOT` to a working directory. It copies the shipped tables into that directory, so recomputations there never overwrite the tables in this repository; plotting scripts write to the output folder you give them. `MANIFEST.md` lists the script used for each panel.

See `MANIFEST.md` for source-hash compatibility and preflight limitations.

## Data

All datasets are public. Download scripts and accession numbers are in `data/`.

- Single-cell data: mouse pancreas (GSE132188), human bone marrow (Setty et al., 2019), mouse heart (GSE126128)
- Knockout data: RFX6 (Zenodo 10.5281/zenodo.10656891), GATA1 (GSE274110), TBX5 and NKX2-5 (GSE77547)
- ChIP data: TBX5 and NKX2-5 (GSE72223), RFX6 (GSE62844)
