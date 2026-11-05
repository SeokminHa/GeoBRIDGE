# Changes relative to the internal code

Principle: scripts were copied from the lab server repository (which is newer than the local working copy; server versions were used throughout, local `scratch/` was only used to classify files). No scientific logic was rewritten.
The per-file sha256 of every copied script before and after the edits below is in `ORIGINAL_SHA256.tsv` (218 scripts; 45 byte-identical, 173 edited by the rules below).

## 1. Mechanical path substitution (all `.py`, `.sh`)
* `"<internal-root>/geobridge..."` string literals (231 occurrences in code):
  plain strings -> `(__import__("os").environ["GEOBRIDGE_ROOT"] + "<rest>")`; f-strings -> `f"{__import__('os').environ['GEOBRIDGE_ROOT']}<rest>"`; shell scripts -> `${GEOBRIDGE_ROOT}`.
  The variable must be set (a missing variable raises `KeyError` immediately; there is no silent default).
* The same absolute path inside comments and docstrings was replaced textually by `$GEOBRIDGE_ROOT`.
* Interpreter / environment binaries `<internal-root>/(miniconda3/)envs/<NAME>/bin/<exe>` (8 occurrences) -> `os.path.join(os.environ.get("GB_<NAME>_BIN", <dir of sys.executable>), "<exe>")`
  (shell: `${GB_<NAME>_BIN}`): `GB_JUPYTER_ENV_BIN` (main env), `GB_VELORAMA_BIN` (velorama env), `GB_GB_SOTA_BIN` (FastTENET / RegDiffusion env).
* Files that contained only path-bearing comments are counted among the 173 "edited" files.
* Consequence: the SHA-256 of these scripts differs from the internal originals. Internal fail-closed checks that pin **source hashes** therefore no longer match (OPEN_ISSUES #1).

## 2. Host names (documentation-level edits, no logic)
* `figure2_synthetic/scripts/gb_tf32rep_launch_2026_10_22.sh` and `..._fast_...sh`: the hard-coded `ssh <lab node>` calls and node lists were replaced by two environment lists
  (`GB_CPU_HOSTS="host:slots ..."`, `GB_GPU_HOSTS="host:gpu_index ..."`); the guard `ssh <node> "test -f grn_seeds.json && ! test -f STOP"` became a local `test`; the header's WSL/`/mnt/d` hint was removed.
  The queue semantics (`worker`, `cpuext`, `gpuworker`) are unchanged. `localhost` is not special-cased: the launcher still uses `ssh <host>` for every entry.
* Comments/labels naming lab nodes were replaced by generic wording in: `common/scripts/gb_defense_recompute_v1_89c4c21b.py`, `common/scripts/gb_regvelo_noprior_gpu.py`,
  `figure1_overview/scripts/gb_main_fig1_draft_2026_11_03.py`, `supplementary/scripts/gb_runtime_common_agg_2026_11_04.py` (the last one changes hardware strings written into the runtime table: "<node name> 72 cores" -> "CPU node 72 cores").
* The word `kappa` in `common/scripts/gb_gata1_truth_v2.py` is Cohen's kappa (statistic), not a host name, and was left unchanged.

## 3. Files added / moved (not copies of internal files)
* New: `README.md` files, `MANIFEST.md`, `OPEN_ISSUES.md`, `setup_workspace.sh`, `environment.yml`, `LICENSE` (placeholder), `.gitignore`, `data/download_*` (written for this release, not executed), `data/download_cardiac_desoysa.md`.
* Taken from the server without edit: `patches/sergio_numpy_compat.patch` (= `git diff` of the lab's SERGIO checkout: only `np.int`/`np.float` -> `int`/`float`), `common/GEOBRIDGE_PREPROCESS_V1.json` (frozen preprocessing contract, hash-bound by artifacts, unchanged),
  `common/GB_real_v2/lambert2018_human_tfs.txt`, `common/data/LaraAstiaso_raw/ext_ref/external_regulators.txt` (both from `package/geobridge/.../vendor/R`).
* Evaluator helper scripts that live in result directories on the server (path-substituted): `figure3_knockout/results_scripts/bm_acute_design_audit_2026_10_26/noF2/fig3_ugb_v2_noF2_velalltargets.py`, `.../fig3_ugb_2026_10_26/fig3_ugb_v2.py` (+`SPEC.md`).
* `geobridge/` = `package/geobridge` of the internal repository, byte-identical to the server copy (checked by md5), minus `.pytest_cache`, `__pycache__` and `manifest/` (the manifest listed internal draft figures of 2026-10-07 and the `build_manifest.py`/`hash_inputs.py` helpers for internal hashing).
* Final summary tables in `*/results/` are unmodified copies (sizes < 5 MB each). `supplementary/results/` keeps only TSVs < 5 MB of the internal `r2fix_2026_11_01` tree (200 of its files; larger tables and all non-TSV artifacts were left out).
* Placement: a script that is the entry point of exactly one figure sits in that figure's `scripts/`; modules imported by two or more figures sit in `common/`. Flat bare-name imports are preserved by `setup_workspace.sh`.

## 4. Deliberately NOT changed
* No logic, constants, seeds, pins, thresholds or statistics. Source-hash pins and pre-declaration checks are left in place (they now fail; see OPEN_ISSUES #1).
* (superseded by section 6) The original release left reviewer/model names in docstrings and comments; they were scrubbed on 2026-11-05.

## 5. Additional scrubs
* Account-name fragments in work-directory defaults (`/var/tmp/<account-name>_*` -> `/var/tmp/geobridge_*`) in the cardiac download scripts; conda-environment paths in `data/pancreas_rfx6_ko/salmon_pipeline.sh` (now `${GB_JUPYTER_ENV_BIN%/bin}`) and one docstring in `common/scripts/gb_real_feasibility.py`.
* `figure3_knockout/results/table1/table1_receipt.json` and `figure3_knockout/results/truth/rfx6_occupancy.receipt.json`: the absolute server path prefix was replaced by `$GEOBRIDGE_ROOT` (hashes untouched; these receipts are provenance records, not read by code).
* (superseded by section 6) Originally NOT scrubbed: the whole `geobridge/` package, byte-identical to the internal one. Its vendored modules (`src/geobridge/vendor/iso/gb_iso_lib.py`, `gb_isofull_lib.py`, `vendor/R/scratch/gb_canonical.py`, `gb_target_extension.py`, `vendor/R/pairspace/gb_gata1_preflight.py`) are SHA-256-verified at import (`vendor/*/SHA256SUMS`)
  and contain `R = '<internal-root>/geobridge'` defaults; `tests/*.py` default `GB_REPO` to the same path. Set `GB_REPO` for the tests; editing the vendored files requires regenerating `SHA256SUMS` -- left for the maintainers (OPEN_ISSUES #13).

## 6. 2026-11-05 revision (reviewer-requested fixes)
Every file changed in this revision, with sha256 before/after, is listed in `PUBLIC_EDITS_2026_11_05_SHA256.tsv`; `ORIGINAL_SHA256.tsv` has a fifth column with the sha256 after this revision (`note` = new path of the seven files moved to `archive/superseded/`).
1. **Scrub (text only, no logic change).** In all scripts and docs: reviewer/model names (the four internal reviewer names, "Claude", "Codex"), "PI directive" and "reviewed by ..." phrases -> "internal review" / "internal directive";
   every `collab/...` reference -> "internal design note (not shipped)" (in the 9 places where it was a path string inside executable code, e.g. the `PREDECL`/`PREREG` path constants, the path now points to a non-existent
   `internal_design_note_not_shipped*` / `design_notes_not_shipped/` name -- these checks failed before and still fail by default); references to internal project-rules / lock-amendment files -> neutral wording;
   host names and the absolute account path removed everywhere (`geobridge/README.md` "measured on a shared CPU node"; this file). Syntax of all 233 `.py` files re-checked with `ast.parse`.
   The scrub is regex-based, so some comments read awkwardly (e.g. "internal review r99"); no executable statement was changed except the path constants named above.
   Vendored package: `R0`/`R`/`GEOBRIDGE_ROOT` defaults in `vendor/iso/gb_iso_lib.py`, `gb_isofull_lib.py`, `vendor/R/scratch/gb_canonical.py`, `gb_target_extension.py`, `vendor/R/pairspace/gb_gata1_preflight.py` are now `os.environ.get("GB_REPO", "")`
   (`sys.path` inserts guarded by `if R`), tests default `GB_REPO` to `""`; `vendor/iso/SHA256SUMS` and `vendor/R/SHA256SUMS` regenerated with the package's own method (sha256 of each listed file). The golden hashes
   (`GOLDEN_SHA256`, `fixture/expected.json`) hash numerical outputs, not source bytes, and were NOT changed; they still pass. The vendored modules are therefore no longer byte-identical to the internal ones.
2. **Tests.** `test_parity.py::test_vendored_equals_pinned` skips when `$GB_REPO/scratch/gb_primary.py` is absent; the real-data tests already skipped without the stored artifacts. Results are in the section 6 test log below.
3. **`GB_SKIP_PREFLIGHT=1`** (opt-in; default unchanged): skips only the brief/sha pins, prints `preflight skipped (public release)`, side effects (e.g. `register_cal_arm`) still run. Sites: `gb_calbm_propagate.preflight`, `gb_calbm_reeval.preflight`,
   `gb_nullp_challenge.pins_ok`, `gb_meta_nascent_sutil.cmd_run` / `cmd_breadth`, `gb_sergio_band_eval.assert_prereg`, `gb_real_uniform.frozen_fig1_roster` / `trace_endpoints`, `gb_cop_alignment.evaluate`, `gb_canon_table.pin_check`,
   `gb_decompose` (receipt-vs-script sha), `gb_ptable._audit_control` (receipt-vs-script sha), `gb_truth_v2_load.load_verified_gata1_v2` (manifest sha), `gb_m1_calib3` (both runtime-hash checks). Verified on a server copy: with the variable `pins_ok()` and `gb_calbm_reeval.preflight()` return after printing the message; without it they refuse as before (`REFUSING: gb_ptable.py sha ... != pinned 5db6f77d`).
4. **Shipped small evaluator inputs.** `figure3_knockout/results/evaluator_inputs/ep_{PANC__RFX6,BM_GATA1_ACUTE__GATA1,CARDIACV__TBX5,CARDIACV__NKX2-5}__hist.vec.npz` (copied unchanged from the lab server, 0.45-0.65 MB each);
   Fig. 4 per-gene inputs for CARDIACV_TBX5, CARDIACV_NKX2-5, PANC_RFX6 (`T_<unit>.tsv`, `features_<unit>.tsv`, `<unit>_labels.tsv`, `rp_<unit>.tsv`; 3.4 MB; copied unchanged). `setup_workspace.sh` links them to the paths the scripts read.
   Verified: `gb_kochip_audit.py p1` on these inputs reproduces the three units' rows of the shipped `p1_levels.tsv` exactly (max abs difference 0.0, KB=2000, ~10 s).
5. **FastTENET seeding** (`np.random.seed(0)`, `torch.manual_seed(0)`, `torch.cuda.manual_seed_all(0)` before `run`) added to `gb_topo_adv_gpu.run_tenet` AND to `gb_sota_ext.run_fasttenet` (the function the Fig. 2 driver actually calls). Added in release; shipped numbers were produced without it.
6. **Fig. 2**: `figure2_synthetic/DESIGN.md` ("design as implemented (original spec document not archived)"); single-dataset quick reproduction recipe (README, untested end to end); both launchers accept host `localhost` (runs the worker command with `bash -c`, no ssh; `bash -n` checked, not executed).
7. **`table1/make_table1_tex.py`** (new): TSV -> LaTeX rows; run on a server copy; all numbers equal the hand-typeset .tex (gene column uses the `gene` field, e.g. INS1; SST subtype order differs).
8. **Superseded renderers moved** (no edits) to `archive/superseded/`: `gb_fig4_final_2026_10_29.py`, `gb_fig4_render_v6.py`, `gb_fig2_bars_v2_2026_10_27.py`, `gb_fig2_layout_2026_10_28.py`, and the three 2026-11-03 Fig. 1 scripts (`gb_fig1_overview_2026_11_03.py`, `gb_fig1_panel_a_2026_11_03.py`, `gb_main_fig1_draft_2026_11_03.py`).
   Nothing imports them (checked by grep); READMEs and MANIFEST updated.
9. `OPEN_ISSUES.md` rewritten with status per item.

### Test log (server copy of this release, `python -m pytest tests`, jupyter_env python; copy in a scratch directory, nothing written to the internal repository)
* Without `GB_REPO`: **5 passed, 6 skipped** (final tree; 8 s). Skips = internal-artifact tests.
* With `GB_REPO` = internal workspace: **10 passed, 1 skipped** (the raw-rebuild end-to-end test, which needs `GB_E2E=1`), 16.5 min. This run used the tree one revision before the last two edits (a comment text and the regenerated `SHA256SUMS`); the final tree was re-tested only without `GB_REPO`.
* Also run on the copy: `setup_workspace.sh` (202 scripts linked), `gb_kochip_audit.py p1` (matches shipped table exactly), `GB_SKIP_PREFLIGHT` on/off check, `table1/make_table1_tex.py`.
