#!/usr/bin/env bash
# Public inputs for KO truth, ChIP truth and annotation. Everything lands under $GEOBRIDGE_ROOT/data (paths as expected by the scripts).
# Download commands are untested; verify URLs and checksums before use.
set -euo pipefail
: "${GEOBRIDGE_ROOT:?}"; D="$GEOBRIDGE_ROOT/data"; mkdir -p "$D"
geo_s() { # geo_s GSMxxxxxxx filename dest_dir   (GEO sample supplementary file)
  local gsm=$1 f=$2 dest=$3; mkdir -p "$dest"; local stub="${gsm:0:${#gsm}-3}nnn"
  curl -fsSL -C - -o "$dest/$f" "https://ftp.ncbi.nlm.nih.gov/geo/samples/$stub/$gsm/suppl/$f"; }
geo_series() { local gse=$1 f=$2 dest=$3; mkdir -p "$dest"; local stub="${gse:0:${#gse}-3}nnn"
  curl -fsSL -C - -o "$dest/$f" "https://ftp.ncbi.nlm.nih.gov/geo/series/$stub/$gse/suppl/$f"; }

# Bone marrow GATA1 KO (Martin-Rufino et al. 2025): authors' DESeq2 table, GATA1 vs AAVS1, 24 h (PRO-seq), GEO GSE274110
geo_series GSE274110 GSE274110_Sankaran003_gga_deseq_24h_GATA1vs24h_AAVS1_results_table.txt.gz "$D/gse274110_proseq"

# Pancreas RFX6 KO (Aldous et al. 2024): FASTQ at Zenodo record 10.5281/zenodo.10656891 (10 libraries: 4 EP + 6 PP; PP = 3 WT vs 3 KO used)
mkdir -p "$D/../real_1_Pancreas/external_validation/aldous_rfx6/data"
curl -fsSL https://zenodo.org/api/records/10656891 | python3 -c "
import sys,json
for f in json.load(sys.stdin)['files']: print(f['key'], f['links']['self'])" > "$D/zenodo_10656891_files.txt"
while read -r name url; do curl -fsSL -C - -o "$D/../real_1_Pancreas/external_validation/aldous_rfx6/data/$name" "$url"; done < "$D/zenodo_10656891_files.txt"
#   then (see data/pancreas_rfx6_ko/): GENCODE v44 transcripts -> salmon_pipeline.sh (Salmon 1.10.3) -> de_analysis.py (pyDESeq2)

# Heart TBX5 / NKX2-5 KO RNA-seq (Luna-Zurita et al. 2016), GSE77547 (SubSeries of GSE72223): ENA FASTQ, runs in cardiac_truth_build.py SAMPLES
#   WT CM SRR2177623-25 ; Tbx5 KO CM SRR2177629-31 ; Nkx2-5 KO CM SRR2177635-37   (scripts: figure3_knockout/scripts/cardiac_lz_dl.sh, cardiac_lz_quant*.sh)
# Heart ChIP-exo footprints (Luna-Zurita 2016; GSE72223): GSM2054345 (Tbx5 WT CM), GSM2054346 (Nkx2-5 WT CM), (GSM2054347 Gata4, not used in figures)
geo_s GSM2054345 GSM2054345_Tbx5_WT_CM_avgfootprints.txt.gz "$D/raw/mouse_cardiac_feasibility/chipexo_footprints"
geo_s GSM2054346 GSM2054346_Nkx_WT_CM_avgfootprints.txt.gz  "$D/raw/mouse_cardiac_feasibility/chipexo_footprints"
curl -fsSL -o "$D/raw/mouse_cardiac_feasibility/annotation/refGene_mm9.txt.gz" --create-dirs https://hgdownload.soe.ucsc.edu/goldenPath/mm9/database/refGene.txt.gz

# Pancreas RFX6 ChIP-seq: GSE62844, sample GSM1534425 (HA-Rfx6, mouse Min6b1 cells), MACS peaks, mm9 (cited as the pancreatic ChIP evidence in Fig 4)
geo_s GSM1534425 GSM1534425_PSR-06_HA-Rfx6.peaks.txt.gz "$D/rfx6_binding/GSE62844"
cp "$D/raw/mouse_cardiac_feasibility/annotation/refGene_mm9.txt.gz" "$D/rfx6_binding/refGene_mm9.txt.gz"

# Orthology (human->mouse 1:1, MGI) and marker sets
mkdir -p "$D/orthology" "$D/genesets"
curl -fsSL -o "$D/orthology/HOM_MouseHumanSequence.rpt" https://www.informatics.jax.org/downloads/reports/HOM_MouseHumanSequence.rpt
# CellMarker 2.0 / 2024 and PanglaoDB through the Enrichr library endpoint (Table 1: PRIMARY CellMarker_2024; SENSITIVITY PanglaoDB_Augmented_2021)
for lib in CellMarker_2024 PanglaoDB_Augmented_2021; do
  curl -fsSL -o "$D/genesets/$lib.txt" "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName=$lib"; done
echo done
