# Figure 2 -- SERGIO benchmark (TF32REP): GeoBRIDGE vs seven GRN methods under shared developmental effects

Design: 6 GRNs (seeds 501-506; 100 genes, 32 candidate regulators, exactly 240 direct edges, 3,168 regulator-target pairs, prevalence 0.0758) x 2 topologies (bifurcation = Fig. 2; tree = SI)
x 2 simulations x 4 conditions (C0 none, C1 temporal input A, C2 branch input B, C3 A+B) = **96 datasets**. 1,200 cells per bifurcation dataset (300 per cell type, 4 types), UMI counts at depth 10.
Methods: GeoBRIDGE (the unsigned score Z, called PRIMARY in the scripts; k=0..8, minimum over the two roots), Velorama (rt0 configuration), FastTENET, GENIE3, GRNBoost2, PIDC, RegDiffusion, DeepSEM (+ |Pearson|, which was dropped from the final figure).
Statistics (in `scripts/gb_fig2_nopearson_2026_10_30.py`): per-dataset metric -> mean over 2 sims -> per GRN; mean over 6 GRNs; 95% t interval; two-sided paired t-test over the 6 GRNs, Holm over the 7 comparators per condition and metric.

Design (as implemented; original spec not archived): `DESIGN.md`.

Workspace: `export GEOBRIDGE_ROOT=/path/ws; bash ../setup_workspace.sh` (links all scripts to `$GEOBRIDGE_ROOT/scratch`, as the scripts import each other by bare name and run "from repo root").
All commands are run with `cd $GEOBRIDGE_ROOT` and `python scratch/<script>`.

## Panel -> script -> input -> output

| Panel | Script | Input | Output |
|---|---|---|---|
| 2a topology schematic, 2b temporal input A(t), 2c branch input B(t) | `gb_fig2_schematic_2026_10_28.py OUT` (b, c are the binned means of the *simulated* latent driver concentrations: `raw/counts_truth.npz` of the C3 datasets) | regenerated SERGIO datasets (`results/sergio_tf32rep_2026_10_22/g50*/`) | `Fig2b_clean.png`, `Fig2c_clean.png`, `Fig2_top_schematic_draft.png` (a/b/c). |
| 2d AUPRC (bifurcation) | `gb_fig2_nopearson_2026_10_30.py OUT` | `results/TABLE.tsv` (176 KB, shipped), `results/D1_vs_D2D3_datasets.tsv` (shipped) | `Fig2d_AP_bifurcation.png`, `Fig2_values_holm7.tsv` (shipped for comparison) |
| 2e AUROC | same script | same | `Fig2e_AUROC_bifurcation.png` |
| **2f** direct-vs-indirect AUROC (direct edges vs ALL indirect pairs, graph distance 2 to 5) | `gb_fig2f_dviall_2026_11_05.py OUT VALUES_TSV` (re-plot) or `gb_fig2f_dviall_2026_11_05.py OUT` (recompute from the per-pair tables of step 4, not shipped) | re-plot: `results/fig2f_dviall/Fig2f_DvIall_values.tsv` (shipped); recompute: per-pair tables + `OUT/REF_Fig2_values_holm7.tsv` (a copy of `results/Fig2_values_holm7.tsv`, used for a parity check) | `Fig2f_DvIall_bifurcation.{png,jpg}`; recompute also writes `Fig2f_DvIall_values.tsv`, `dviall_per_dataset.tsv` (shipped copies in `results/fig2f_dviall/`) |

`gb_fig2_nopearson_2026_10_30.py` also draws a panel `Fig2f_DvI23_*.png` (direct vs 2- and 3-hop pairs only). That is an earlier version of panel f (as is the standalone `gb_fig2f_dvi23_2026_10_29.py`, kept for provenance only) and is **not** the paper's Fig. 2f; its values (`DvI23` rows of `Fig2_values_holm7.tsv`) are only used as the parity reference of `gb_fig2f_dviall_2026_11_05.py`.

### Re-plot only (no recomputation; uses shipped tables)
```
export GEOBRIDGE_ROOT=$PWD/ws; bash setup_workspace.sh
python $GEOBRIDGE_ROOT/scratch/gb_fig2_nopearson_2026_10_30.py out_fig2     # d, e (and the old 2-3-hop f, see above) for both topologies + Fig2_values_holm7.tsv
python $GEOBRIDGE_ROOT/scratch/gb_fig2f_dviall_2026_11_05.py out_fig2f figure2_synthetic/results/fig2f_dviall/Fig2f_DvIall_values.tsv   # paper Fig. 2f
```
Both write only into the given output folder.
(Panels a-c need the simulated datasets, see below.)

### Quick reproduction of Fig. 2a-c from ONE dataset (C3, GRN 501, bifurcation, sim 1; one CPU host, no GPU, no comparators) -- untested end-to-end in this release
```
export GEOBRIDGE_ROOT=$PWD/ws; bash setup_workspace.sh          # needs SERGIO (patched, see below) at $GEOBRIDGE_ROOT/SERGIO and GB_JUPYTER_ENV_BIN (bin directory of the main environment)
cd $GEOBRIDGE_ROOT
python scratch/gb_tf32rep_fast_2026_10_22.py ph 501 bifurcation         # roster + hidden-driver design (prep + harvest); grn_seeds.json is linked by setup_workspace.sh
python scratch/gb_tf32rep_fast_2026_10_22.py run 501 bifurcation C3 1    # SERGIO -> UMI -> canonical preprocessing -> GeoBRIDGE; writes raw/counts_truth.npz
python scratch/gb_fig2_schematic_2026_10_28.py out_fig2abc              # panels a-c pool every C3 dataset found under results/sergio_tf32rep_2026_10_22/g*/bifurcation/C3/ (here: just this one, so b/c are noisier than the paper's 12-dataset mean)
```
Cost depends on the machine (GeoBRIDGE run uses 8 threads). The schematic script also draws d, e and an earlier version of f (direct vs 2- and 3-hop pairs) from the shipped `TABLE.tsv`; the paper's Fig. 2f comes from `gb_fig2f_dviall_2026_11_05.py`.

### Launching the full queue without ssh
`GB_CPU_HOSTS="localhost:2" GB_GPU_HOSTS="localhost:0" bash scripts/gb_tf32rep_launch_fast_2026_10_22.sh` -- the host name `localhost` runs the worker directly (no ssh); any other name still uses `ssh <host>`.

## Full recomputation, in order
Dependencies: main env + `patches/sergio_numpy_compat.patch` applied to https://github.com/PayamDiba/SERGIO (commit a6190b7; the only change is `np.int/np.float` -> `int/float` for modern numpy) placed at `$GEOBRIDGE_ROOT/SERGIO`;
velorama env (`GB_VELORAMA_BIN`), gb_sota env (`GB_GB_SOTA_BIN`, for FastTENET / RegDiffusion); GPUs for Velorama, DeepSEM, RegDiffusion, FastTENET; CPU for GENIE3/GRNBoost2/PIDC (arboreto uses 8 dask workers per task).

1. Cohort selection (graph-only, no simulation): `python scratch/gb_tf32rep_fast_2026_10_22.py audit 501 60` -> `FEASIBILITY.md`, `grn_seeds.json` = first six GRN seeds that pass the structural criteria = **501-506**
   (shipped copies: `results/FEASIBILITY.md`, `results/grn_seeds.json`).
2. Generation + GeoBRIDGE scoring + comparators, via the persistent queue (task locks in `_locks/`, completion in `_done/`):
   `scripts/gb_tf32rep_launch_fast_2026_10_22.sh` (see header; set `GB_CPU_HOSTS`, `GB_GPU_HOSTS`) runs, per host,
   `gb_tf32rep_fast_2026_10_22.py worker <slots>` (= `ph`: GRN roster + fixed half orientation; `run`: SERGIO simulation -> UMI counts -> canonical preprocessing -> GeoBRIDGE),
   `gpuworker <gpu> velorama,deepsem,regdiffusion,fasttenet`, and `cpuext <slots> genie3,grnboost2,pidc,pearson`.
   Single-dataset form: `python scratch/gb_tf32rep_fast_2026_10_22.py run 501 bifurcation C3 1`.
   Generator: `gb_pilot_AB_tf32rep_fast_2026_10_22.py` (the records of the 96 datasets carry its sha256 as `script_sha256`; the non-"fast" `gb_pilot_AB_tf32rep_2026_10_22.py` is imported for `generate_grn`/graph statistics only).
3. `python scratch/gb_tf32rep_fast_2026_10_22.py eval` -> `TABLE.tsv` (864 rows = 96 datasets x 9 methods), `COMPLETENESS.md`, `INVALID.tsv`; `... analyze` -> `ANALYSIS.md` (GRN-level contrasts, exact sign-flip, Holm-7).
4. Direct-vs-indirect per-pair tables: `python scratch/gb_dvi_audit3_2026_10_29.py results/signed/dvi_audit3_2026_10_29/full` then `python scratch/gb_dvi_audit3_agg_2026_10_29.py results/signed/dvi_audit3_2026_10_29/full` -> `agg/D1_vs_D2D3_datasets.tsv`.
5. Fig. 2f values from the per-pair tables: `mkdir -p out_fig2f && cp <repo>/figure2_synthetic/results/Fig2_values_holm7.tsv out_fig2f/REF_Fig2_values_holm7.tsv && python scratch/gb_fig2f_dviall_2026_11_05.py out_fig2f` (asserts that its 2-/3-hop values equal the reference, then writes `Fig2f_DvIall_values.tsv`, `dviall_per_dataset.tsv` and the panel).
6. Plots: commands in the table above.

Output locations: steps 1, 3 and 4 write into `$GEOBRIDGE_ROOT/results/...` by design (e.g. step 3 rewrites `results/sergio_tf32rep_2026_10_22/TABLE.tsv`). `setup_workspace.sh` places *copies* of the shipped tables there, so this replaces the workspace copy, never the files in this repository; use a fresh `GEOBRIDGE_ROOT` if you want to keep the shipped copies next to your recomputed ones. Steps 5 and 6 write only into the folder given on the command line.

## Seeding
* GRN: `generate_grn(g)` uses `np.random.RandomState(g)` for every structural draw (layer assignment, edges, signs, strengths K in U(1,5)); GRN seeds 501-506 are fixed by the audit (first six passing seeds).
* SERGIO simulation: `np.random.seed(1000*grn + sim)` immediately before `sim.simulate()` (sim in {1,2}); master-regulator basal rates `RandomState(grn+5000)`.
* Hidden-input loadings: A `RandomState(777+grn)`, B `RandomState(31337+grn)`.
* UMI rendering (depth 10): dedicated `np.random.Generator(PCG64([20260924, grn, sim, round(1e6*lam)]))` (not the global RNG).
* Preprocessing/pairs: scanpy PCA/neighbors `random_state` and pair sampling `pair_seed` are read from the fixed `GEOBRIDGE_PREPROCESS_V1.json` (forward-pair sampling uses its `pair_seed`; near-isochronic sign pairs use seeds 42/43/44 with `default_rng([seed, root])`, per `geobridge/README.md`).
* Comparators: Velorama CLI `-s 0` (numpy+torch seeded in `train.py`; `-vm stochastic` scVelo velocity; deterministic given input); RegDiffusion `seed=0`; DeepSEM `np/torch/cuda seed 0`; GENIE3/GRNBoost2 arboreto `seed=0`; PIDC has no RNG; FastTENET: the paper's FastTENET results (the shipped numbers) came from runs without an explicit seed. The released wrapper (`gb_sota_ext.run_fasttenet`, `gb_topo_adv_gpu.run_tenet`) sets seed = 0 (`np.random.seed(0)`, `torch.manual_seed(0)`) before `run`, so reruns are repeatable but may differ slightly from the paper's FastTENET numbers.
* Not guaranteed: bitwise identity on other hardware (BLAS thread count, GPU atomics), and scanpy-PCA differences (relative ~1e-5 in the PANC rebuild, see `geobridge/README.md`).
