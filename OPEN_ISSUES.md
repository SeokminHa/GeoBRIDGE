# OPEN ISSUES (updated 2026-11-05)

Status legend: **RESOLVED** (changed in the 2026-11-05 revision), **PARTLY**, **OPEN**. Details of every change: `CHANGES_FROM_INTERNAL.md` section 6.
Nothing was resolved by guessing; the full figure pipelines were NOT re-run end to end.

## A. Running the real-data evaluation (Fig. 3 b, c, g; Table 1)
1. **Fail-closed integrity checks (source-hash and pre-declaration pins) -- PARTLY RESOLVED.** The checks are unchanged by default (they refuse), because the pinned bytes/documents are internal and not shipped.
   New opt-in `GB_SKIP_PREFLIGHT=1` skips only those brief/sha pins in every script that has them (`gb_calbm_propagate.preflight`, `gb_calbm_reeval.preflight`, `gb_nullp_challenge.pins_ok`, `gb_meta_nascent_sutil` (2 sites),
   `gb_sergio_band_eval.assert_prereg`, `gb_real_uniform` (2 sites), `gb_cop_alignment.evaluate`, `gb_canon_table.pin_check`, `gb_decompose`, `gb_ptable._audit_control`, `gb_truth_v2_load.load_verified_gata1_v2`, `gb_m1_calib3` (2 sites)); required side effects still run.
   Still OPEN: the evaluator also needs the server-side intermediates (comparator caches, catbank, truth tables) -- see 3.
   Not touched because they are data-integrity checks, not brief pins: receipt/output-hash checks (`STALE_OR_ALTERED`, `REFUSING_GBCACHE_*`), the Fig. 2 pilot-design hash checks (they only compare against design files that exist from an earlier run), `gb_klf1_eval` predictions hash.
2. `TF32REP_SPEC.md` (frozen Fig. 2 spec) does not exist on the server -- **PARTLY RESOLVED:** `figure2_synthetic/DESIGN.md` documents the design as implemented (assembled from docstrings/README; original spec not archived). The generator file's module docstring is inherited from an older benchmark version and does not describe Fig. 2 (noted in DESIGN.md).
3. Evaluator input vectors -- **PARTLY RESOLVED:** the four per-endpoint `ep_*.vec.npz` (PANC_RFX6, BM_GATA1_acute, HEART_TBX5, HEART_NKX2-5) are shipped in `figure3_knockout/results/evaluator_inputs/`. They are necessary but not sufficient: the evaluator also reads truth tables, shared draws, comparator caches and `results/signed/*.npz`, which remain server-only (R2).
   A point-AUROC computed from the vectors alone differs from `fig3_common_v2_UOLD.tsv` (up to 0.05, BM GATA1; <= 0.01 elsewhere), so the vectors are NOT a stand-alone reproduction of Fig. 3b.

## B. Seeding
4. Fig. 2 data generation is seeded (see `figure2_synthetic/README.md`); "regenerates bit-identically" is **OPEN / not verified** (depends on numpy/SERGIO versions and the patched SERGIO; the 96 datasets came from a generator file with a different sha256 than the public copy; a `_fastparity` check on the server was not read).
5. Comparator seeding -- **RESOLVED for the public code, shipped numbers unchanged:** FastTENET now seeded (`np.random.seed(0)`, `torch.manual_seed(0)`) in `gb_topo_adv_gpu.run_tenet` and `gb_sota_ext.run_fasttenet`; the shipped numbers were produced without it (re-running FastTENET will not reproduce them exactly).
   GPU runs still do not set `cudnn.deterministic`; bitwise identity across devices/drivers is **OPEN**.
6. GeoBRIDGE preprocessing: a raw PANC rebuild differs from the stored pairs at ~1.8e-5 relative (PCA; thread/BLAS), ranks agree (Spearman 0.99997) -- **OPEN** (bitwise reproduction not claimed).

## C. Not traced / missing / server-only
7. Fig. 2 a-c final assembly (`Figure2_assembled.jpg`, `Figures_v2.pptx`) done by hand -- **OPEN**. A single-dataset recipe for the a-c inputs is now documented (untested end to end).
8. Fig. 3 panel naming (internal `Fig3b_*_APLIFT` vs manuscript 3b = AUROC; `_v7` suffix) -- **OPEN** (documented in `figure3_knockout/README.md`).
9. Table 1 .tex producer -- **RESOLVED (equivalent):** `table1/make_table1_tex.py` generates the rows from `table1_main_rows.tsv` / `table1_wholeset_metrics.tsv`; its output matches the hand-typeset values (only the SST subtype order differs). The original typesetting script was never found. The "KO datasets" table counts remain hand-taken from `fig3_bc_holm7.tsv`.
10. Fig. 1 is a schematic with no data; its renderer is not shipped (removed 2026-11-04 per author decision).
11. Fig. 4 per-gene inputs -- **PARTLY RESOLVED:** `T_<unit>.tsv`, `features_<unit>.tsv`, `<unit>_labels.tsv`, `rp_<unit>.tsv` for TBX5, NKX2-5 and RFX6 are shipped (3.4 MB); stage `p1` reproduces the shipped `p1_levels.tsv` exactly. Stages `aux`/`p1b`, the trend table and `iso_pilot` score recomputation still need control-Ms h5ad files / `iso_freeze.json` (server-only).
    The ChIP occupancy window of the RFX6 class labels (1/10/50 kb in `gb_endpoints_v2.py`) is still not determined from the code -- **OPEN** (check `gb_chip5_sens.py`).
12. **Data download verification -- OPEN.** The scVelo/GEO/Zenodo/Enrichr/UCSC/MGI commands in `data/` were written from standard URL patterns and never executed or checksum-compared; the heart reference files (`geo_cells.txt`, chr-prefixed Ensembl-102 GTF, mm10 repeat mask, `refs/kb_nac_GRCm38`) have no build script; third-party code under `$GEOBRIDGE_ROOT/external/` and `$GEOBRIDGE_ROOT/SERGIO` is not shipped.
13. `geobridge/` package -- **RESOLVED (paths):** vendored files no longer contain the internal absolute path (`GB_REPO` env var, empty default); `SHA256SUMS` regenerated; tests needing internal files skip when `GB_REPO` is unset. The real-data parity tests still need stored PANC artifacts (server-only). The vendored modules are no longer byte-identical to the internal ones (documented).
14. Reviewer/model names, `collab/briefs/...` references, host names and account paths in scripts and docs -- **RESOLVED** (mechanical text scrub, section 6 of `CHANGES_FROM_INTERNAL.md`). Comment text now says "internal review" / "internal design note (not shipped)"; a few comments therefore read awkwardly. Re-grep before publication.
15. Static dependency closure (AST imports + quoted script names) not tested by importing -- **OPEN.** The workspace builder `setup_workspace.sh` was run on the server copy (202 scripts linked); scripts were not import-tested one by one.
16. Local vs server: server copies were used; two closure members come from `legacy/` -- unchanged.
17. Excluded on purpose: `legacy/`, governance/collab files, `.env`, internal logs, supplementary tables > 5 MB, KO truth tables (heart 9.4 MB, pancreas 6.1 MB) and the 16 MB bootstrap draw file -- unchanged.
18. **License / copyright holder -- OPEN** (`LICENSE` is a placeholder).
19. Statistical-reporting caveats stated by the manuscript still hold (sign estimator selected with KO directions; conditional bootstraps; coarse 6-GRN resolution) -- unchanged.
20. **Unpublished systems in `common/` modules -- OPEN.** Several shared modules (e.g. KLF1 / K562 / retina / dentate-gyrus / gastrulation code paths) cover systems that are not in the paper; decide whether to prune or keep before publication.
