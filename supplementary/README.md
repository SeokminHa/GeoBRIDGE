# Supplementary figures

This folder contains the code and the summary tables used to make the supplementary figures of the paper. Every figure can be redrawn in a few seconds from the tables in `data/`. The scripts that produced those tables are in `scripts/` as well.

## Requirements

- Python 3.10 with NumPy, pandas, SciPy, statsmodels, Matplotlib and Pillow (all included in `../environment.yml`)
- Redrawing the figures needs no other data and no GPU.
- Recomputing the tables needs the simulated benchmark datasets (see `../figure2_synthetic/README.md`) and the knockout evaluation inputs (see `../figure3_knockout/README.md`).

## Tutorial

1. Install the environment from the repository root.

   ```bash
   conda env create -f environment.yml
   conda activate geobridge
   ```

2. Draw the figures. Each script reads its tables from `supplementary/data/` and writes PNG files to the folder given by `--out`.

   ```bash
   python supplementary/scripts/plot_figS1.py --out out/figS1
   python supplementary/scripts/plot_figS2.py --out out/figS2
   python supplementary/scripts/plot_figS3.py --out out/figS3
   python supplementary/scripts/plot_figS4.py --out out/figS4
   python supplementary/scripts/plot_figS5_S6.py --out out/figS5_S6
   python supplementary/scripts/plot_figS7.py --out out/figS7
   ```

   Figures S1 to S4 are written as one PNG per panel. Figures S5 and S6 are written as whole figures, with the single panels in `panels/`; Figure S7 is written as one PNG. The scripts also write the plotted values (means, confidence intervals, p-values) as TSV files next to the PNGs.

3. (Optional) Recompute the tables. Set up the workspace as described in the repository README (`export GEOBRIDGE_ROOT=/path/to/workspace; bash setup_workspace.sh`), run the compute scripts listed below from `$GEOBRIDGE_ROOT`, then collect the tables:

   ```bash
   python supplementary/scripts/make_data_tables.py out/supp_data
   ```

   This only selects and merges rows from the compute outputs and writes them to `out/supp_data/`. Without an argument it writes to `supplementary/data/` and overwrites the shipped tables, so pass a separate folder and compare.

## Figures

All synthetic results use the bifurcation benchmark of Fig. 2: 6 GRNs, 4 conditions (no shared effect, temporal effect A, branch effect B, both), 2 simulations per GRN averaged before taking the mean and 95% t interval over GRNs. "Direct vs indirect" compares direct edges with all indirect pairs at graph distance 2 to 5.

- Fig. S1, direct-versus-indirect AUPRC and conditioning depth. `plot_figS1.py`, tables `figS1_dvi_auprc.tsv` (panel a and legend; paired t-test of GeoBRIDGE against each of 7 methods, Holm correction per condition) and `figS1_kcurve.tsv` (panels b to e; GeoBRIDGE AUPRC, AUROC, direct-vs-indirect AUROC and AUPRC against the maximum depth K = 0 to 30). Computed by `gb_si_dviap_panels_2026_11_05.py` (modes `pairs` and `kc`), `gb_si_recompute_dviall_2026_11_05.py` (mode `kc_e1`) and `gb_fig2_kcurve_2026_11_01.py`.
- Fig. S2, choice of target channel and conditioning operator. `plot_figS2.py`, tables `figS2_channel_operator.tsv` (AUPRC per GRN for each operator, channel and depth window) and `figS2_permutation_null.tsv` (label-permutation reference line). Computed by `gb_fig2_ablation_2026_10_27.py`, `gb_fig2_ablation_aggregate_2026_10_27.py` and `gb_si_figS2_permutation_null.py`.
- Fig. S3, cell-pair window and number of pairs. `plot_figS3.py`, table `figS3_pairwindow.tsv` (heatmaps of AUPRC over the lower and upper bound of the pseudotime difference for the 4 conditions; AUPRC for 10, 20, 40 and 80 pairs per cell). Computed by `gb_e11_pairwindow_2026_11_02.py` and `gb_e11_summarize_2026_11_02.py`.
- Fig. S4, strength of each shared effect. `plot_figS4.py`, table `figS4_amplitude_summary.tsv` (AUPRC and direct-vs-indirect AUROC for the temporal-only and branch-only amplitude series 0, 0.25, 0.5 and 1.0). Paired comparisons are in `figS4_amplitude_paired.tsv`. Computed by `gb_dose_sep_2026_11_05.py` (simulation of the new amplitudes, all methods, then `eval` and `agg`).
- Fig. S5 and S6, knockout-response prioritization by response direction (AUROC in S5, AUPRC lift in S6). `plot_figS5_S6.py`, tables `figS5_S6_ko_direction.tsv` (KO-upregulated and KO-downregulated responders against non-responders, 95% module-bootstrap intervals with 9,999 draws, Holm correction over the 7 comparators) and `figS5_S6_k0.tsv` (GeoBRIDGE without conditioning, k = 0). These rows come from the Fig. 3 evaluation (`../figure3_knockout/results/fig3_common_v2_UOLD.tsv` and `fig3d_rebuild/full/fig3d_rebuild_table.tsv`); see `../figure3_knockout/README.md` for how they were computed.
- Fig. S7, sign estimators. `plot_figS7.py`, tables `figS7_sign_synthetic.tsv` (balanced accuracy of each sign estimator against the true edge sign on the synthetic benchmark; the figure uses the bifurcation rows) and `figS7_sign_knockout.tsv` (agreement with knockout direction among responders, equal-weight mean over five TFs). Computed by `gb_sign_synth_E13_2026_11_03.py` and `gb_sign_real_E14_2026_11_03.py`. The knockout endpoints overlap those of Fig. 4, so this panel is not independent validation.
- Table S1 (single-cell datasets) and Table S2 (knockout datasets) list public data only. Accession numbers and download scripts are in `../data/README.md`.

## Notes

- `gb_si_recompute_dviall_2026_11_05.py` has other modes (`dose`, `plot`) that are not needed for these figures.
- `gb_si_dviap_panels_2026_11_05.py pairs` reads the per-pair tables written by `../figure2_synthetic/scripts/gb_dvi_audit3_2026_10_29.py`. It also checks its AUROC values against `dviall_per_dataset.tsv`, written by `../figure2_synthetic/scripts/gb_fig2f_dviall_2026_11_05.py` (shipped in `../figure2_synthetic/results/fig2f_dviall/`).
