# Figure 3 and Table 1 -- knockout-response prioritisation on real developmental systems

Systems / endpoints: mouse pancreas (RFX6 KO, 1,353 common genes), human bone marrow (GATA1 KO, acute 24 h PRO-seq), mouse heart (TBX5 KO, NKX2-5 KO; shared WT libraries).
Methods compared: GeoBRIDGE, GeoBRIDGE k=0 (ablation, outside the Holm family), Velorama, FastTENET, GENIE3, GRNBoost2, PIDC, RegDiffusion, DeepSEM. Labels (KO) are evaluation-only.
Inference: AUROC / AUPRC-lift on one common finite row set per endpoint (the evaluator "U_OLD" universe: rows scored by GeoBRIDGE, all comparators and the original cached Velorama column),
paired module bootstrap (G/50 modules, B = 9,999 draws, synchronised across methods; cardiac draws shared by the two heart TFs), Holm over the 7 external comparators per endpoint x metric.
Evidence level of each panel is stated in `../MANIFEST.md` (R1 = re-plot from shipped table; R2 = needs server-side cached intermediates, scripts traced; R3 = from raw data, chain partially traced).

Workspace: `export GEOBRIDGE_ROOT=/path/ws; bash ../setup_workspace.sh`; commands run from `$GEOBRIDGE_ROOT` as `python scratch/<script>`.

## Panel -> script -> input -> output
(Manuscript numbering. File names in the internal figure folders differ for 3b/3c: the files `Fig3b_*_APLIFT.png` are AUPRC-lift = manuscript **3c**, `Fig3c_*_AUROC.png` are AUROC = manuscript **3b**.)

| Panel | Script (in `scripts/`) | Input | Output |
|---|---|---|---|
| 3a embeddings + velocity streams (pancreas, bone marrow, heart) | `gb_fig3a_umap_labels_2026_10_30.py <PANC\|BONEMARROW\|CARDIACV> OUT` (identical pipeline to `gb_fig3_umap_stream_2026_10_23.py`: neighbours on stored 30 PCs seed 0 -> UMAP seed 0 -> Leiden res 1.0 for colour only -> scVelo stream; labels = median UMAP position of the dataset's own `clusters` annotation) | `results/canonical_2026_09_18/{PANC,BONEMARROW,CARDIACV}_canonical.h5ad` (rebuilt, R3) | `{SYS}_umap_stream_labeled.png` |
| 3b AUROC, 3c AUPRC-lift (equal-system mean + 4 endpoints) | `gb_fig3bc_panels_2026_10_30.py TABLE OUTDIR --k0 FIG3D_REBUILD_TABLE` | `results/fig3_common_v2_UOLD.tsv` (shipped; the B=9,999 table) and `results/fig3d_rebuild/full/fig3d_rebuild_table.tsv` (shipped; supplies the k=0 bar and asserts its arm D == GeoBRIDGE row) | `Fig3{b,c}_<endpoint>_{APLIFT,AUROC}.png`, `fig3_bc_holm7.tsv` (shipped for comparison; recomputed here) |
| 3d CV R^2 of 30-PC embedding explained by pseudotime / terminal identity | `gb_fig3d_panels_2026_10_30.py OUTDIR` (plot) <- `gb_fig3d_devstructure_2026_10_29.py OUT` (compute; 5-fold CV, 20 permutations, seeds `default_rng([20261029, ...])`) | `results/fig3d_rebuild/devstructure/panc_devstructure.tsv` (shipped) | `Fig3d_devstructure.png` |
| 3e pseudotime-difference energy captured by first k directions | same plot script <- `gb_fig3d_geom30_2026_10_29.py OUT` | `results/fig3d_rebuild/geom30/geom30_pooled.tsv` (shipped) | `Fig3e1_capture_pseudotime.png` |
| 3f terminal-identity energy beyond pseudotime | same | same | `Fig3e2_capture_identity.png` |
| 3g RFX6 AUROC vs aggregation depth (0..k), k=8 marked | same plot script <- `gb_fig3d_depth_multi_2026_10_29.py OUT --endpoints PANC_RFX6,HEART_TBX5,HEART_NKX2-5,BM_GATA1_acute --B 9999` | `results/fig3d_rebuild/depth_multi_B9999/{panc,heart,bm}/depth_multi.tsv` (shipped) | `Fig3f_PANC_depth_curve.png` (internal v7), SI: all systems, forest of production minus k=0 |
| Table 1 RFX6 endocrine-marker rank percentiles | `gb_table1_panc_markers_2026_10_29.py OUT --B 9999` | CellMarker_2024 (Enrichr) marker sets, MGI orthologs, frozen Fig 3 scores/truth (R2) | `results/table1/table1_main_rows.tsv` (shipped), `table1_wholeset_metrics.tsv`, `table1_receipt.json`; typeset in `results/table1/TABLE1_PANC_endocrine_markers_2026_10_30.tex` (producer of the .tex from the .tsv not found) |
| Table (KO datasets; counts) | `gb_truth_tables_2026_10_28.py` (+ `cardiac_truth_build.py`, `data/pancreas_rfx6_ko/de_analysis.py`) | see `../data/README.md` | evaluated-gene / responder counts per endpoint |

### Re-plot (R1: no recomputation)
```
python scratch/gb_fig3bc_panels_2026_10_30.py <repo>/figure3_knockout/results/fig3_common_v2_UOLD.tsv out_fig3 \
       --k0 $GEOBRIDGE_ROOT/results/signed/fig3d_rebuild_2026_10_29/full/fig3d_rebuild_table.tsv
python scratch/gb_fig3d_panels_2026_10_30.py out_fig3d          # d, e, f, g (+SI) from the shipped devstructure/geom30/depth_multi tables
```

## Full chain, in order (R2/R3; server-scale)
1. Inputs: `../data/README.md` (scVelo pancreas and bone marrow, heart velocyto rebuild, KO and ChIP tables).
2. Canonical preprocessing (frozen contract `common/GEOBRIDGE_PREPROCESS_V1.json`): `python scratch/gb_canonical.py build results/canonical_2026_09_18 --system PANC` (and `BONEMARROW`);
   heart: `python scratch/gb_canonical_cardiac.py build results/canonical_2026_09_18 --system CARDIACV`; pairs for both roots (`dc1_min`, `dc1_max`): `... pairs ...` (`gb_canonical.py`/`gb_canonical_cardiac.py` CLI).
   The same scoring for a single dataset is available through `../geobridge` (`geobridge score ...`).
3a. Evaluator input vectors (the four paper endpoints are SHIPPED in `results/evaluator_inputs/`: `ep_PANC__RFX6__hist`, `ep_BM_GATA1_ACUTE__GATA1__hist`, `ep_CARDIACV__TBX5__hist`, `ep_CARDIACV__NKX2-5__hist` `.vec.npz`, 0.45-0.65 MB each, linked by `setup_workspace.sh` to `results/real_uniform/meta_nascent_sutil/full/parts/`; they are *necessary but not sufficient* inputs of the evaluator, which also reads truth tables, draws and comparator caches, and a point-AUROC computed from the vectors alone is NOT the table value (checked: differences up to 0.05 for BM GATA1).).  Original location: `results/real_uniform/meta_nascent_sutil/full/parts/ep_<system>__<TF>__<truth>.vec.npz` (labels `y`, mask `MR`, one `cog::<method>` score column per method incl. `cog::GeoBRIDGE`): written by `gb_meta_nascent_sutil.py` (A4/A5 meta-analysis stage; needs the stored catbank / comparator caches below). Its provenance was traced from the reader (`gb_fig3d_magnitude.VEC`), not from a run.
3. GeoBRIDGE focal-TF scores per depth: `gb_signed_cache.py SYSTEM SRC,SRC` -> `results/signed/<SYSTEM>.npz`; production cache rows for the evaluator come from `gb_real_uniform.py` (`gbcache`) / `gb_catbank.py`.
4. Comparators on the same canonical fixtures (truth-free, same TF/target universe rules): Velorama `gb_velorama_v2_rt0_alltargets.py` (official code, SHA-bound, GPU), FastTENET / RegDiffusion / DeepSEM `gb_sota_ext.py` via `gb_real_uniform.py gpu`, GENIE3 / GRNBoost2 `gb_sota_run.py` (arboreto), PIDC (`gb_sota_ext.pidc_weights`), |Pearson| (dropped from the final figure).
5. Truth: `cardiac_truth_build.py {de,bound,table}`; `data/pancreas_rfx6_ko/{salmon_pipeline.sh,de_analysis.py}`; BM table as deposited; `gb_truth_tables_2026_10_28.py` assembles the four endpoint truth tables.
6. Depth-matched control arms (spline-of-pseudotime and random-direction arms used by the rebuild): `gb_pt_comparator_compute_2026_10_21.py SYSTEM OUTDIR --nseeds 20` (PANC, BONEMARROW, CARDIACV).
7. Evaluator: `python scratch/gb_fig3_common_v2_2026_10_28.py OUT --B 9999 --uold` -> `fig3_common_v2_UOLD.tsv`; `python scratch/gb_fig3d_rebuild_2026_10_29.py OUT --B 9999` -> `full/fig3d_rebuild_table.tsv`; then the d/e/f/g compute scripts above; Table 1 script.
   The evaluator executes `results_scripts/bm_acute_design_audit_2026_10_26/noF2/fig3_ugb_v2_noF2_velalltargets.py` (bone-marrow GATA1 acute adapter) and reads `gb_fig3_ugb_v2_velalltargets_2026_10_26.py` source (via `exec`).
8. Plots (previous section).

**Preflight pins:** the evaluator prefix calls `gb_calbm_propagate.preflight()`, which refuses to run unless four source files match pinned SHA-256 prefixes and internal pre-declaration documents are present. These documents are not part of this repository. Set `GB_SKIP_PREFLIGHT=1` to skip ONLY those brief/sha pins (prints `preflight skipped (public release)`; required side effects such as `register_cal_arm` still run). The remaining data-integrity checks (receipts, parity, finiteness) are unchanged and still need the server-side intermediates.

## Seeding
Bootstrap draws: `gb_ptable.lattice_draws` seeds each module draw from `SeedSequence(mod_seed([BOOT_SEED, ...]))` (fixed constants; identical across methods by construction). Fig 3d permutations/null seeds `default_rng([20261029, k, seed])`. UMAP/neighbours/Leiden `random_state=0`. Velorama on real data: `-s` seed 0 in `gb_velorama_v2_rt0_alltargets.py` (a `_long` and a seed-check variant exist for the training-sensitivity audit).
