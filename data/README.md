# Data availability and download

Raw datasets are not bundled; small result tables and a test fixture are included. All inputs are public. Every path below is relative to `$GEOBRIDGE_ROOT` (see the top-level README).
Status column: **traced** = the exact file name / accession was read from the scripts that consume it; **unverified** = the download command was written
from the standard URL pattern and was NOT executed by the release builder.

## Single-cell datasets (score inputs; unperturbed cells only)

| System | Source | Accession / file used by the scripts | How obtained here |
|---|---|---|---|
| Mouse pancreas (E15.5 endocrinogenesis) | Bastidas-Ponce et al. 2019; GEO GSE132188 | `data/Pancreas/endocrinogenesis_day15.h5ad` (= `scvelo.datasets.pancreas()`, 3,696 cells) | `download_scvelo_datasets.py` (unverified) |
| Human CD34+ bone marrow | Setty et al. 2019 (Palantir) | `data/BoneMarrow/human_cd34_bone_marrow.h5ad` (= `scvelo.datasets.bonemarrow()`, 5,780 cells) | `download_scvelo_datasets.py` (unverified) |
| Mouse heart (cardiogenesis, WT cells) | de Soysa et al. 2019; GEO GSE126128 | `data/raw/mouse_cardiac_velocyto/cardiacv_raw_velocyto.h5ad` (primary build `CARDIACV`, velocyto counts) | rebuilt from the authors' BAMs: `download_cardiac_desoysa.md`, `cardiac_desoysa_velocyto/` (traced) |
| (non-primary) heart, kb-nac build `CARDIAC` | same BAMs | `data/raw/mouse_cardiac/cardiac_raw_nac.h5ad` | `figure3_knockout/scripts/cardiac_bam_worker3.sh` (traced; supplementary use only) |

Note: scVelo does not ship the heart data, so the velocity-compatible (spliced/unspliced) heart counts had to be generated from the original 10x BAMs.
The score itself uses only spliced `Ms` and unspliced `Mu` moments and a 30-PC embedding; RNA velocity is computed by the canonical preprocessing but is not read by the score.

## Knockout (evaluation labels only; never score inputs)

| Endpoint | Source | Accession | Processing (script) |
|---|---|---|---|
| RFX6 KO vs WT, pancreatic progenitors (human iPSC, 3 vs 3) | Aldous et al. 2024 | Zenodo 10.5281/zenodo.10656891 (FASTQ; 20/20 md5 matched the record) | `pancreas_rfx6_ko/salmon_pipeline.sh` (Salmon 1.10.3, GENCODE v44) -> `pancreas_rfx6_ko/de_analysis.py` (pyDESeq2, KO vs WT); human->mouse by one-to-one MGI orthologs (`HOM_MouseHumanSequence.rpt`) |
| GATA1 KO vs AAVS1, 24 h, PRO-seq | Martin-Rufino et al. 2025 | GEO GSE274110 supplementary `GSE274110_Sankaran003_gga_deseq_24h_GATA1vs24h_AAVS1_results_table.txt.gz` (authors' DESeq2 table, used as is) | `figure3_knockout/scripts/gb_truth_tables_2026_10_28.py` |
| Tbx5 KO, Nkx2-5 KO vs WT cardiomyocytes (mouse ESC-derived, 3 vs 3) | Luna-Zurita et al. 2016 | GEO GSE77547 (SubSeries of GSE72223); ENA runs SRR2177623-25 (WT), SRR2177629-31 (Tbx5 KO), SRR2177635-37 (Nkx2-5 KO) | `figure3_knockout/scripts/cardiac_lz_dl.sh`, `cardiac_lz_quant.sh`, `cardiac_lz_quant_all.sh` (Salmon, Ensembl-102 GRCm38) -> `cardiac_truth_build.py de/table` (pyDESeq2; responders padj<0.05 and abs(log2FC)>=0.5) |

## ChIP evidence (Fig. 4 only; evaluation labels only)

| TF | Source | Accession / file | Processing |
|---|---|---|---|
| TBX5, NKX2-5 (heart) | Luna-Zurita et al. 2016 ChIP-exo, WT cardiomyocytes | GEO GSE72223: `GSM2054345_Tbx5_WT_CM_avgfootprints.txt.gz`, `GSM2054346_Nkx_WT_CM_avgfootprints.txt.gz`; refGene mm9 | `cardiac_truth_build.py bound` (footprints within +-2 kb of any TSS; +-10 kb sensitivity) |
| RFX6 (pancreas) | Piccand et al. 2014 (HA-Rfx6 ChIP-seq, mouse Min6b1 cells) | GEO GSE62844, sample GSM1534425 `GSM1534425_PSR-06_HA-Rfx6.peaks.txt.gz` (MACS, mm9, FDR<=5%) | `figure4_signed/scripts/gb_endpoints_v2.py` (each summit -> single nearest canonical TSS, ties unassigned; occupancy window as defined in that script) -> `results/endpoints_v2/rfx6_occupancy.tsv` (shipped in `figure3_knockout/results/truth/`) |

Orthology: MGI `HOM_MouseHumanSequence.rpt`. Annotation sets for Table 1: CellMarker 2.0 (Hu et al. 2023) via the Enrichr library `CellMarker_2024` (primary) and `PanglaoDB_Augmented_2021` (sensitivity).
TF catalogues: Lambert 2018 human TF list (shipped, `common/GB_real_v2/`); mouse catalogue as referenced in `common/GEOBRIDGE_PREPROCESS_V1.json`.

`download_ko_chip_annotation.sh` collects the GEO/Zenodo/UCSC/MGI/Enrichr files above (unverified commands; check md5/sha against the sha256 values in the run receipts if available; the receipts are not included).

## Synthetic data

Not stored (3.1 GB). All 96 SERGIO datasets are regenerated from seeds: see `figure2_synthetic/README.md`.

## Heavy intermediates not shipped

`results/canonical_2026_09_18/*_canonical.h5ad` (+ `*_pairs.npz`), the signed per-depth caches `results/signed/<SYSTEM>.npz`, comparator score caches (Velorama, FastTENET,
GENIE3, GRNBoost2, PIDC, RegDiffusion, DeepSEM on real data), and the SERGIO datasets are rebuilt by the commands in the figure READMEs.
