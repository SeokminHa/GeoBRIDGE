# Cardiac spliced and unspliced counts (GSE126128)

The main cardiac build is `CARDIACV` (velocyto spliced/unspliced counts; gb_canonical_cardiac.py registers it as `CARDIACV`).
Nine WT 10x libraries (E7.75 WT1-5, E8.25 WT1-2, E9.25 WT1-2); 36,753 GEO-passed cells.

| run | library name (BAM file prefix) | GEM group | cells |
|---|---|---|---|
| SRR8535078 | E775_WT1 | 1 | 3069 |
| SRR8535079 | E775_WT2 | 2 | 1673 |
| SRR8535080 | E775_WT3 | 3 | 2728 |
| SRR8535081 | E775_WT4 | 4 | 1311 |
| SRR8535082 | E775_WT5 | 5 | 2265 |
| SRR8535083 | E825_WT1 | 6 | 2316 |
| SRR8535084 | E825_WT2 | 7 | 7368 |
| SRR8535085 | E925_WT1 | 8 | 4691 |
| SRR8535086 | E925_WT2 | 9 | 11332 |

Pipeline (scripts in this directory; run once per library; each needs the authors' cellranger BAM, ~tens of GB):
1. `geo_cells.txt` = the 36,753 GEO barcodes with gem suffix, from the GEO count matrix `GSE126128_AllTimePoints_WT_10X.csv.gz`
   (regenerate this file from the GEO matrix column names; it is not included).
2. `dl.sh`  downloads the BAM from the SRA original-format mirror (`https://sra-pub-src-1.s3.amazonaws.com/<SRR>/<name>_possorted_genome_bam.bam.1`).
3. `run_one3.sh <SRR> <name> <gem>`: keeps GEO-passed barcodes (samtools), cell-sorts, runs `velocyto run` (Default logic, chunked over barcode sets) with the
   Ensembl-102 GRCm38 GTF chr-prefixed to match the BAM contigs (`ref/genes_chr.gtf`) and the UCSC mm10 repeat mask (`ref/mm10_rmsk.gtf`) -> loom;
   `post.py` validates the loom and writes a per-run receipt.
4. `figure3_knockout/scripts/cardiacv_build_raw_h5ad.py` merges the looms into `data/raw/mouse_cardiac_velocyto/cardiacv_raw_velocyto.h5ad`
   (X = spliced; layers spliced/unspliced; velocyto "ambiguous" excluded).
5. `python gb_canonical_cardiac.py build $GEOBRIDGE_ROOT/results/canonical_2026_09_18 --system CARDIACV`, then `... pairs ...` (see figure3_knockout/README.md).
The non-primary kallisto-bustools `nac` build (`CARDIAC`) is produced by `figure3_knockout/scripts/cardiac_bam_worker3.sh` / `cardiac_build_raw_h5ad.py`;
it is used only by supplementary diagnostics.
