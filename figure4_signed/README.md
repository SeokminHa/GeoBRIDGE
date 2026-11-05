# Figure 4 -- signed score T across KO x ChIP classes (TBX5, NKX2-5 heart; RFX6 pancreas)

Classes (D+, D-, N, U-, U+; called levels L1 to L5 in the scripts): KO-down with / without ChIP binding, non-responders, KO-up without / with binding.
Signed score T = d_iso x F_r(Z): d_iso = sign of the **unconditioned (k=0) near-isochronic concordance** (pairs with |dpseudotime| <= 0.02 inside the same annotated stratum;
3 sampled pair sets, seeds 42/43/44, both roots; stratum = CARDIACV `embryo`, PANC `clusters`), F_r = within-regulator percentile of the unsigned score Z, called PRIMARY in the scripts (`geobridge/src/geobridge/iso.py` is the package form; production scripts: `gb_iso_pilot.py`, `gb_iso_lib.py`).
In the paper this score is $T_{rg}=\operatorname{sign}(S^{(0),\mathrm{iso}}_{rg})\,F_r(G_{rg})$. In code, the Fig 4 score is `T_iso = sign(iso_C) * abs(Z|d4)` (`iso_C` = $S^{(0),\mathrm{iso}}_{rg}$, `Z` = $G_{rg}$) (`gb_kochip_audit_lib.load_unit`); `Z|d4` is a stored column whose absolute value equals F_r (the code asserts this; its sign comes from an alternative sign estimate, `d4`, that is not used in Fig. 4), so `T_iso` = sign(iso_C) x F_r. Comparisons with alternative sign estimators are in Supplementary Fig. S7; the signed analysis is exploratory (manuscript S1.5).
Statistics shown: class means with descriptive 95% **gene**-bootstrap intervals; one-sided Mann-Whitney between adjacent classes, Holm over the 6 contrasts per TF (4 drawn);
ordered-trend test over the five classes with a null-imposed **control-module** cluster bootstrap (B = 9,999), Holm over the 3 TFs.

Workspace: `export GEOBRIDGE_ROOT=/path/ws; bash ../setup_workspace.sh`; commands from `$GEOBRIDGE_ROOT`.

## Panel -> script -> input -> output
| Panel | Script (in `scripts/`) | Input | Output |
|---|---|---|---|
| 4a class schematic | `gb_fig4ab_polish_2026_10_30.py OUTDIR [--w 6.2]` | none (drawn) | `Fig4a.png` |
| 4b class means + brackets + trend P | same script | `results/kochip_audit_2026_10_15/p1_levels.tsv` (class means / gene-bootstrap CI), `.../fig19_contrast_pvalues.tsv` (adjacent MWU, Holm-6), `results/fig4b_trend.tsv` (trend slope, Holm-3 P) -- all shipped | `Fig4b.png` |

### Re-plot (R1): `python scratch/gb_fig4ab_polish_2026_10_30.py out_fig4` (reads the three shipped tables through the workspace links).

## Shipped per-gene inputs (TBX5, NKX2-5, RFX6) -- stage `p1` now runs from this repository
`results/{grand_v3d_dumpT_2026_10_15/parts/T_<unit>.tsv, iso_pilot_2026_10_15/scores/<unit>/features_<unit>.tsv, chip5_sens_2026_10_15/parts/<unit>_labels.tsv, beta_rp_2026_10_15/rp_<unit>.tsv}`
for units `CARDIACV_TBX5`, `CARDIACV_NKX2-5`, `PANC_RFX6` (3.4 MB; the four files `load_unit` in `gb_kochip_audit_lib.py` joins; `setup_workspace.sh` links them to the paths the scripts read).
```
export GEOBRIDGE_ROOT=$PWD/ws; bash setup_workspace.sh; cd $GEOBRIDGE_ROOT
KB=2000 python scratch/gb_kochip_audit.py p1 CARDIACV_TBX5,CARDIACV_NKX2-5,PANC_RFX6    # recomputes p1_levels.tsv (class means, gene-bootstrap CI)
```
`gb_kochip_audit.py` writes into `$GEOBRIDGE_ROOT/results/signed/kochip_audit_2026_10_15/` by design. `setup_workspace.sh` puts a *copy* of the shipped tables there, so the recomputation replaces the workspace copy, not the files in this repository; compare it with `figure4_signed/results/kochip_audit_2026_10_15/p1_levels.tsv`.
Stages `aux`, `p1b` and the trend (`gb_fig4_bind_prod_k0_2026_10_29.py`) additionally need control-Ms h5ad files / `iso_freeze.json` (not included).

## Upstream chain that produced the three tables (R2: stages beyond `p1` need intermediate files that are not included; scripts traced)
1. Per-gene signed-score dumps for each unit (`T_<unit>.tsv`): `gb_grand_v3d_dumpT.py` (modes `ko` / `chip`; stored Z and the candidate sign estimates, including `d4` and the near-isochronic sign `iso`).
2. Near-isochronic polarity features `iso_C` per unit: `python scratch/gb_iso_pilot.py score <SYSTEM> <FOCAL>` (e.g. `CARDIACV TBX5`, `CARDIACV NKX2-5`, `PANC RFX6`) -> `results/signed/iso_pilot_2026_10_15/scores/<unit>/features_<unit>.tsv` (no truth read).
3. KO x ChIP labels per unit: `python scratch/gb_chip5_sens.py unit <UNIT> OUTDIR` (KO tables + ChIP occupancy; `cardiac_truth_build.py`, `gb_endpoints_v2.py`); regulatory potential `gb_beta_rp.py unit <UNIT> OUTDIR`.
4. Audit stages: `python scratch/gb_kochip_audit.py aux|p1|p1b <units>` -> `p1_levels.tsv`, `p1b_module.tsv`, `aux_<unit>.tsv` (50 control-module average-linkage clusters); `python scratch/gb_kochip_fig_p.py` -> `fig19_contrast_pvalues.tsv`.
5. Trend: `python scratch/gb_fig4_bind_prod_k0_2026_10_29.py OUT --B 2000` (per-gene tables `genes_<UNIT>.tsv`), then `python scratch/gb_fig4_style_2026_10_29.py RUN_DIR OUTDIR --B 9999 --minmod 5` -> `fig4b_trend.tsv`.
6. Plot (above).

## Package CLI versus the Fig. 4 tables
The `geobridge` CLI (`../geobridge`) computes the same signed score T. On pancreatic RFX6 it reproduces the paper's ranking exactly (Spearman 1.0000, 0 sign mismatches against `sign(iso_C) * F` of the shipped `features_PANC_RFX6.tsv`), but the values can differ (max abs diff 0.077): the CLI computes the percentile F over its own target universe (every gene with a finite Z), whereas Fig. 4 used the evaluation gene universe (the stored evaluation gene mask, 1,345 genes for PANC RFX6; Fig. 3 uses 1,353 genes, the genes scored by every method). The Fig. 4 tables were produced by `scripts/gb_iso_pilot.py score` (`features_<unit>.tsv`: `iso_C`, `F`) and `scripts/gb_grand_v3d_dumpT.py` (`T_<unit>.tsv`). The parity test is `geobridge/tests/test_iso.py::test_panc_rfx6_d_iso_and_T_vs_stored`.

## Seeding
Trend bootstrap `default_rng([20261029, 41, n_modules])`; gene-bootstrap intervals reproduce `gb_kochip_audit.py` (`rng_for(unit, 'lev', i)`, global `SEED = 20261015`); near-isochronic pair sampling seeds 42/43/44 with `default_rng([seed, root_idx])`.

## Caveats stated in the manuscript (keep them when reusing)
No direct-regulation claim on real data; KO and ChIP only label genes for evaluation; the class ordering is descriptive. The sign is the unconditioned near-isochronic concordance; its comparison with alternative sign estimators on synthetic and knockout data is in Supplementary Fig. S7 (`../supplementary/`). Because those knockout endpoints overlap the Fig. 4 endpoints, the signed analyses are exploratory.
