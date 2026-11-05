#!/usr/bin/env bash
# Build a $GEOBRIDGE_ROOT workspace that mirrors the directory layout the (unmodified-logic) scripts expect:
#   $GEOBRIDGE_ROOT/scratch/*.py   <- every script from common/scripts and */scripts  (flat; scripts import each other by bare module name)
#   $GEOBRIDGE_ROOT/{pairspace,benchmark,geobridge_model}  <- common/<same>
#   $GEOBRIDGE_ROOT/results/...    <- the small final tables shipped in */results/ (so plotting scripts run without recomputation)
# Large inputs/intermediates (data/, results/canonical_*, comparator caches, SERGIO datasets) are NOT shipped: see data/README.md and each figure README.
# Usage:  export GEOBRIDGE_ROOT=/path/to/workspace ; bash setup_workspace.sh
set -euo pipefail
: "${GEOBRIDGE_ROOT:?set GEOBRIDGE_ROOT to the workspace directory (created if missing)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
R="$GEOBRIDGE_ROOT"; mkdir -p "$R/scratch" "$R/results/signed"
for f in "$HERE"/common/scripts/* "$HERE"/figure*/scripts/* "$HERE"/supplementary/scripts/*; do
  [ -f "$f" ] && ln -sfn "$f" "$R/scratch/$(basename "$f")"
done
for d in pairspace benchmark geobridge_model; do ln -sfn "$HERE/common/$d" "$R/$d"; done
ln -sfn "$HERE/common/GEOBRIDGE_PREPROCESS_V1.json" "$R/GEOBRIDGE_PREPROCESS_V1.json"
mkdir -p "$R/GB_real_v2" "$R/data/LaraAstiaso_raw/ext_ref"
ln -sfn "$HERE/common/GB_real_v2/lambert2018_human_tfs.txt" "$R/GB_real_v2/lambert2018_human_tfs.txt"
ln -sfn "$HERE/common/data/LaraAstiaso_raw/ext_ref/external_regulators.txt" "$R/data/LaraAstiaso_raw/ext_ref/external_regulators.txt"
ln_res() {  # ln_res <repo-relative file or dir> <workspace-relative target>
  mkdir -p "$(dirname "$R/$2")"; ln -sfn "$HERE/$1" "$R/$2"; }
# Fig 2 plotting inputs
ln_res figure2_synthetic/results/TABLE.tsv results/sergio_tf32rep_2026_10_22/TABLE.tsv
ln_res figure2_synthetic/results/grn_seeds.json results/sergio_tf32rep_2026_10_22/grn_seeds.json
ln_res figure2_synthetic/results/D1_vs_D2D3_datasets.tsv results/signed/dvi_audit3_2026_10_29/full/agg/D1_vs_D2D3_datasets.tsv
# Fig 3 plotting inputs
ln_res figure3_knockout/results/fig3d_rebuild results/signed/fig3d_rebuild_2026_10_29
ln_res figure3_knockout/results/table1 results/signed/table1_panc_markers_2026_10_29/final
ln_res figure3_knockout/results/truth/rfx6_occupancy.tsv results/endpoints_v2/rfx6_occupancy.tsv
ln_res figure3_knockout/results_scripts/bm_acute_design_audit_2026_10_26 results/signed/bm_acute_design_audit_2026_10_26
ln_res figure3_knockout/results_scripts/fig3_ugb_2026_10_26 results/signed/fig3_ugb_2026_10_26
ln_res figure3_knockout/results/evaluator_inputs results/real_uniform/meta_nascent_sutil/full/parts
# Fig 4 plotting inputs
for d in grand_v3d_dumpT_2026_10_15 iso_pilot_2026_10_15 chip5_sens_2026_10_15 beta_rp_2026_10_15; do ln_res figure4_signed/results/$d results/signed/$d; done   # per-gene tables (TBX5, NKX2-5, RFX6)
ln_res figure4_signed/results/kochip_audit_2026_10_15 results/signed/kochip_audit_2026_10_15
ln_res figure4_signed/results/fig4b_trend.tsv manuscript/figures_2026_10_28/fig4_dvi/style_2026_10_29/fig4b_trend.tsv
echo "workspace ready at $R  (scripts: $(ls "$R/scratch" | wc -l) files). Now: export GEOBRIDGE_ROOT=$R and follow the figure READMEs."
