# MANIFEST -- main-text figure panels and Table 1

Levels (how far each panel can be reproduced): **R1** = re-plot from tables shipped in this repository; **R2** = recompute from intermediate files produced by the listed upstream scripts (mostly not shipped); **R3** = recompute from raw public data.
Paths are relative to the repository; scripts are run from `$GEOBRIDGE_ROOT` after `setup_workspace.sh` as `python scratch/<name>`.

## Figure 1 (schematic)
| Panel | Script chain | Level / note |
|---|---|---|

## Figure 2 (SERGIO TF32REP, `figure2_synthetic/`)
| Panel | Script chain | Level |
|---|---|---|
| 2a, 2b, 2c schematic / hidden inputs A(t), B(t) | `gb_fig2_schematic_2026_10_28.py` <- `raw/counts_truth.npz` of the C3 datasets <- generator `gb_pilot_AB_tf32rep_fast_2026_10_22.py` (`ab_profiles`, `ab_hn`) | R3 (regenerate datasets); single-dataset recipe (C3, GRN 501) in `figure2_synthetic/README.md`, design in `figure2_synthetic/DESIGN.md` |
| 2d AUPRC | `gb_fig2_nopearson_2026_10_30.py` <- `results/TABLE.tsv` <- `gb_tf32rep_fast_2026_10_22.py eval` <- {GeoBRIDGE: `gb_tf32rep_fast ... run` (generator + `gb_canonical` + `gb_primary`/`gb_target_extension`), Velorama: `gb_velorama_official_rt0.py`, GENIE3/GRNBoost2/PIDC: `ext` in the driver (`gb_sota_run.arboreto_fit`, `gb_sota_ext.pidc_weights`), RegDiffusion/DeepSEM/FastTENET: `gb_sota_ext.py`, `gb_topo_adv_gpu.run_tenet`} | R1 for the plot; R3 for the data |
| 2e AUROC | same | R1 / R3 |
| 2f direct-vs-indirect AUROC (direct vs ALL indirect pairs, graph distance 2 to 5) | `gb_fig2f_dviall_2026_11_05.py OUT figure2_synthetic/results/fig2f_dviall/Fig2f_DvIall_values.tsv` (re-plot) <- `results/fig2f_dviall/Fig2f_DvIall_values.tsv`, `dviall_per_dataset.tsv` <- `gb_fig2f_dviall_2026_11_05.py OUT` (recompute) <- per-pair tables of `gb_dvi_audit3_2026_10_29.py` (not shipped) <- score caches of the 96 datasets. The 2-/3-hop panel drawn by `gb_fig2_nopearson_2026_10_30.py` (via `results/D1_vs_D2D3_datasets.tsv`, `gb_dvi_audit3_agg_2026_10_29.py`) is an earlier version, used only as a parity reference | R1 / R3 |
| Holm / t-test values | `results/Fig2_values_holm7.tsv` (reference values from the final run) | R1 |

## Figure 3 and Table 1 (`figure3_knockout/`)
| Panel | Script chain | Level |
|---|---|---|
| 3a UMAP + velocity streams | `gb_fig3a_umap_labels_2026_10_30.py` <- `results/canonical_2026_09_18/{PANC,BONEMARROW,CARDIACV}_canonical.h5ad` <- `gb_canonical.py build` / `gb_canonical_cardiac.py build` <- raw data (`data/README.md`) | R3 |
| 3b AUROC, 3c AUPRC lift (4 endpoints + equal-system mean; k=0 bar) | `gb_fig3bc_panels_2026_10_30.py` <- `results/fig3_common_v2_UOLD.tsv` (<- `gb_fig3_common_v2_2026_10_28.py --B 9999 --uold` <- `gb_fig3_ugb_v2_velalltargets_2026_10_26.py` + `results_scripts/.../fig3_ugb_v2_noF2_velalltargets.py` <- cached `ep_<stem>.vec.npz` (written by `gb_meta_nascent_sutil.py`; the 4 paper endpoints are shipped in `figure3_knockout/results/evaluator_inputs/`), truth tables `gb_truth_tables_2026_10_28.py`) and `results/fig3d_rebuild/full/fig3d_rebuild_table.tsv` (<- `gb_fig3d_rebuild_2026_10_29.py --B 9999`) | R1; R2 recompute (needs the remaining server-side inputs; internal pins skippable with `GB_SKIP_PREFLIGHT=1`) |
| 3d CV R^2 (pseudotime; terminal identity beyond pseudotime) | `gb_fig3d_panels_2026_10_30.py` <- `results/fig3d_rebuild/devstructure/panc_devstructure.tsv` <- `gb_fig3d_devstructure_2026_10_29.py` <- canonical PANC fixture + stored pairs | R1; R2 |
| 3e pseudotime-difference energy captured (k=1..10) | same plot script <- `results/fig3d_rebuild/geom30/geom30_pooled.tsv` <- `gb_fig3d_geom30_2026_10_29.py` | R1; R2 |
| 3f terminal-identity energy captured | same | R1; R2 |
| 3g RFX6 AUROC vs aggregation depth | same plot script <- `results/fig3d_rebuild/depth_multi_B9999/panc/depth_multi.tsv` <- `gb_fig3d_depth_multi_2026_10_29.py --B 9999` <- `results/signed/PANC.npz` (`gb_signed_cache.py`) + frozen evaluator rows | R1; R2 |
| Table 1 (RFX6 endocrine marker rank percentiles, 10 KO-responsive of 53 markers) | `gb_table1_panc_markers_2026_10_29.py --B 9999` -> `results/table1/table1_main_rows.tsv`, `table1_wholeset_metrics.tsv`; marker sets: CellMarker_2024 via Enrichr; typeset `results/table1/TABLE1_PANC_endocrine_markers_2026_10_30.tex` (hand-typeset; `table1/make_table1_tex.py` regenerates the rows from the TSVs, verified to agree with the .tex on all values) | R1 for numbers and rows (`python table1/make_table1_tex.py`); R2 recompute |
| Table (KO datasets) | counts from `results/fig3_bc_holm7.tsv`; labels: `gb_truth_tables_2026_10_28.py`, `cardiac_truth_build.py`, `data/pancreas_rfx6_ko/*` | R1 (counts) / R3 |

## Figure 4 (`figure4_signed/`)
| Panel | Script chain | Level |
|---|---|---|
| 4a class schematic | `gb_fig4ab_polish_2026_10_30.py` (drawn) | R1 |
| 4b class means, gene-bootstrap CI, adjacent MWU (Holm-6), trend P (Holm-3) | `gb_fig4ab_polish_2026_10_30.py` <- `results/kochip_audit_2026_10_15/{p1_levels,fig19_contrast_pvalues}.tsv` (<- `gb_kochip_audit.py` stages `aux,p1,p1b`, `gb_kochip_fig_p.py` <- `T_<unit>.tsv` from `gb_grand_v3d_dumpT.py`, `features_<unit>.tsv` from `gb_iso_pilot.py score`, labels from `gb_chip5_sens.py`) and `results/fig4b_trend.tsv` (<- `gb_fig4_style_2026_10_29.py` <- `gb_fig4_bind_prod_k0_2026_10_29.py`) | R1; R2 for stages other than `p1`; stage `p1` (-> `p1_levels.tsv`) re-run from the shipped per-gene tables and reproduces the shipped table exactly (R1.5) |

## Package (`geobridge/`)
Z (`primary.py`, `score.py`), signed score (`iso.py`), CLI, tests; parity against stored internal artifacts is documented in `geobridge/README.md` (PANC RFX6: S0 max abs diff 1.6e-15; T Spearman 1.0000). The real-data parity tests need `GB_REPO` and are skipped otherwise.

## Supplementary figures
| Figure | Plot script (`supplementary/scripts/`) | Tables (`supplementary/data/`) | Level |
|---|---|---|---|
| S1 | `plot_figS1.py` | `figS1_dvi_auprc.tsv`, `figS1_kcurve.tsv` | R1; R2 via `gb_si_dviap_panels_2026_11_05.py`, `gb_si_recompute_dviall_2026_11_05.py`, `gb_fig2_kcurve_2026_11_01.py` |
| S2 | `plot_figS2.py` | `figS2_channel_operator.tsv`, `figS2_permutation_null.tsv` | R1; R2 via `gb_fig2_ablation_2026_10_27.py`, `gb_fig2_ablation_aggregate_2026_10_27.py`, `gb_si_figS2_permutation_null.py` |
| S3 | `plot_figS3.py` | `figS3_pairwindow.tsv` | R1; R2 via `gb_e11_pairwindow_2026_11_02.py`, `gb_e11_summarize_2026_11_02.py` |
| S4 | `plot_figS4.py` | `figS4_amplitude_summary.tsv` | R1; R3 via `gb_dose_sep_2026_11_05.py` |
| S5, S6 | `plot_figS5_S6.py` | `figS5_S6_ko_direction.tsv`, `figS5_S6_k0.tsv` | R1; R2 via the Fig. 3 chain above |

Details in `supplementary/README.md`; `supplementary/scripts/make_data_tables.py` collects the tables from the compute outputs.
