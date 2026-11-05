# MANIFEST -- main-text figure panels and Table 1

Evidence levels: **R1** = re-plot from tables shipped in this repository; **R2** = recomputation needs server-side cached intermediates (scripts traced to the producing step, intermediates NOT shipped);
**R3** = from raw public data (chain partly traced). "Traced" means the script chain and file names were read from the code; the original release build executed nothing (only `ast.parse`); the 2026-11-05 revision executed only the items listed in `CHANGES_FROM_INTERNAL.md` section 6.
Paths are relative to the repository; scripts are run from `$GEOBRIDGE_ROOT` after `setup_workspace.sh` as `python scratch/<name>`. Manuscript = `RECOMB2027_2` (2026-11-04).

## Figure 1 (schematic)
| Panel | Script chain | Level / note |
|---|---|---|

## Figure 2 (SERGIO TF32REP, `figure2_synthetic/`)
| Panel | Script chain | Level |
|---|---|---|
| 2a, 2b, 2c schematic / hidden inputs A(t), B(t) | `gb_fig2_schematic_2026_10_28.py` <- `raw/counts_truth.npz` of the C3 datasets <- generator `gb_pilot_AB_tf32rep_fast_2026_10_22.py` (`ab_profiles`, `ab_hn`) | R3 (regenerate datasets); single-dataset recipe (C3, GRN 501) in `figure2_synthetic/README.md`, design in `figure2_synthetic/DESIGN.md` |
| 2d AUPRC | `gb_fig2_nopearson_2026_10_30.py` <- `results/TABLE.tsv` <- `gb_tf32rep_fast_2026_10_22.py eval` <- {GeoBRIDGE: `gb_tf32rep_fast ... run` (generator + `gb_canonical` + `gb_primary`/`gb_target_extension`), Velorama: `gb_velorama_official_rt0.py`, GENIE3/GRNBoost2/PIDC: `ext` in the driver (`gb_sota_run.arboreto_fit`, `gb_sota_ext.pidc_weights`), RegDiffusion/DeepSEM/FastTENET: `gb_sota_ext.py`, `gb_topo_adv_gpu.run_tenet`} | R1 for the plot; R3 for the data |
| 2e AUROC | same | R1 / R3 |
| 2f direct-vs-indirect AUROC (2- and 3-hop indirect) | `gb_fig2_nopearson_2026_10_30.py` (metric per `gb_fig2f_dvi23_2026_10_29.py`) <- `results/D1_vs_D2D3_datasets.tsv` <- `gb_dvi_audit3_agg_2026_10_29.py` <- `gb_dvi_audit3_2026_10_29.py` <- score caches of the 96 datasets | R1 / R3 |
| Holm / t-test values | `results/Fig2_values_holm7.tsv` (reference values from the final run) | R1 |

## Figure 3 and Table 1 (`figure3_knockout/`)
| Panel | Script chain | Level |
|---|---|---|
| 3a UMAP + velocity streams | `gb_fig3a_umap_labels_2026_10_30.py` <- `results/canonical_2026_09_18/{PANC,BONEMARROW,CARDIACV}_canonical.h5ad` <- `gb_canonical.py build` / `gb_canonical_cardiac.py build` <- raw data (`data/README.md`) | R3 |
| 3b AUROC, 3c AUPRC lift (4 endpoints + equal-system mean; k=0 bar) | `gb_fig3bc_panels_2026_10_30.py` <- `results/fig3_common_v2_UOLD.tsv` (<- `gb_fig3_common_v2_2026_10_28.py --B 9999 --uold` <- `gb_fig3_ugb_v2_velalltargets_2026_10_26.py` + `results_scripts/.../fig3_ugb_v2_noF2_velalltargets.py` <- cached `ep_<stem>.vec.npz` (written by `gb_meta_nascent_sutil.py`; the 4 paper endpoints are shipped in `figure3_knockout/results/evaluator_inputs/`), truth tables `gb_truth_tables_2026_10_28.py`) and `results/fig3d_rebuild/full/fig3d_rebuild_table.tsv` (<- `gb_fig3d_rebuild_2026_10_29.py --B 9999`) | R1; R2 recompute (needs the remaining server-side inputs; internal pins skippable with `GB_SKIP_PREFLIGHT=1`, OPEN_ISSUES #1) |
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
Z (`primary.py`, `score.py`), signed score (`iso.py`), CLI, tests; parity against stored internal artifacts is documented in `geobridge/README.md` (PANC RFX6: S0 max abs diff 1.6e-15; T Spearman 1.0000). Package tests re-run on a server copy of this release on 2026-11-05 (see CHANGES_FROM_INTERNAL.md section 6); the real-data parity part needs `GB_REPO`.

## Supplementary figures
See `supplementary/README.md` (script -> table directory map; SI figure source notes). Level: R1 from shipped `supplementary/results/` TSVs (< 5 MB each); the large per-dataset tables were not copied.
