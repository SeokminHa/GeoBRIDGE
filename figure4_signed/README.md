# Figure 4: signed scores and KO/ChIP classes

Signed score T across KO x ChIP classes (TBX5, NKX2-5 heart; RFX6 pancreas).

Classes (D+, D-, N, U-, U+; called levels L1 to L5 in the scripts): KO-down with / without ChIP binding, non-responders, KO-up without / with binding.
Signed score T = d_iso x F_r(Z), where Z is the unsigned score ($G_{rg}$ in the paper): d_iso = sign of the **unconditioned (k=0) near-isochronic concordance** (pairs with |dpseudotime| <= 0.02 inside the same annotated stratum;
3 sampled pair sets, seeds 42/43/44, both roots; stratum = CARDIACV `embryo`, PANC `clusters`), F_r = within-regulator percentile of the unsigned score Z, called PRIMARY in the scripts (`geobridge/src/geobridge/iso.py` is the package form; production scripts: `gb_iso_pilot.py`, `gb_iso_lib.py`).
Statistics shown: class means with descriptive 95% **gene**-bootstrap intervals; one-sided Mann-Whitney between adjacent classes, Holm over the 6 contrasts per TF (4 drawn);
ordered-trend test over the five classes with a null-imposed **control-module** cluster bootstrap (B = 9,999), Holm over the 3 TFs.

Workspace: `export GEOBRIDGE_ROOT=/path/ws; bash ../setup_workspace.sh`; commands from `$GEOBRIDGE_ROOT`.

## Panels and scripts
| Panel | Script (in `scripts/`) | Input | Output |
|---|---|---|---|
| 4a class schematic | `fig4ab.py OUTDIR [--w 6.2]` | none (drawn) | `Fig4a.png` |
| 4b class means + brackets + trend P | same script | `results/kochip_audit_2026_10_15/p1_levels.tsv` (class means / gene-bootstrap CI), `.../fig19_contrast_pvalues.tsv` (adjacent MWU, Holm-6), `results/fig4b_trend.tsv` (trend slope, Holm-3 P) -- all shipped | `Fig4b.png` |

### Re-plot
`python scratch/fig4ab.py out_fig4` (reads the three shipped tables through the workspace links).

## Included per-gene inputs
TBX5, NKX2-5 and RFX6; stage `p1` runs from this repository.
`results/{grand_v3d_dumpT_2026_10_15/parts/T_<unit>.tsv, iso_pilot_2026_10_15/scores/<unit>/features_<unit>.tsv, chip5_sens_2026_10_15/parts/<unit>_labels.tsv, beta_rp_2026_10_15/rp_<unit>.tsv}`
for units `CARDIACV_TBX5`, `CARDIACV_NKX2-5`, `PANC_RFX6` (3.4 MB). `gb_kochip_audit_lib.load_unit` joins these four files per unit; `setup_workspace.sh` supplies the expected paths.
```
export GEOBRIDGE_ROOT=$PWD/ws; bash setup_workspace.sh; cd $GEOBRIDGE_ROOT
KB=2000 python scratch/gb_kochip_audit.py p1 CARDIACV_TBX5,CARDIACV_NKX2-5,PANC_RFX6    # recomputes p1_levels.tsv (class means, gene-bootstrap CI)
```
Recomputation replaces the workspace copy of `p1_levels.tsv`. Compare it with `figure4_signed/results/kochip_audit_2026_10_15/p1_levels.tsv`.
Stages `aux`, `p1b` and the trend (`gb_fig4_gene_tables.py`) additionally need control-Ms h5ad files / `iso_freeze.json` (not included).

## Recompute the tables
R2: stages beyond `p1` need intermediate files that are not included.
1. Per-gene signed-score dumps for each unit (`T_<unit>.tsv`): `gb_signed_tables.py` (modes `ko` / `chip`; stored Z and the candidate sign estimates, including `d4` and the near-isochronic sign `iso`).
2. Near-isochronic polarity features `iso_C` per unit: `python scratch/gb_iso_pilot.py score <SYSTEM> <FOCAL>` (e.g. `CARDIACV TBX5`, `CARDIACV NKX2-5`, `PANC RFX6`) -> `results/signed/iso_pilot_2026_10_15/scores/<unit>/features_<unit>.tsv` (no truth read).
3. KO x ChIP labels per unit: `python scratch/gb_chip5_sens.py unit <UNIT> OUTDIR` (KO tables + ChIP occupancy; `cardiac_truth_build.py`, `gb_signed_endpoints.py`); regulatory potential `gb_beta_rp.py unit <UNIT> OUTDIR`.
4. Audit stages: `python scratch/gb_kochip_audit.py aux|p1|p1b <units>` -> `p1_levels.tsv`, `p1b_module.tsv`, `aux_<unit>.tsv` (50 control-module average-linkage clusters); `python scratch/gb_kochip_fig_p.py` -> `fig19_contrast_pvalues.tsv`.
5. Trend: `python scratch/gb_fig4_gene_tables.py OUT --B 2000` (per-gene tables `genes_<UNIT>.tsv`), then `python scratch/gb_fig4_trend.py RUN_DIR OUTDIR --B 9999 --minmod 5` -> `fig4b_trend.tsv`.
6. Plot (above).

## Package CLI versus the Fig. 4 tables
The CLI and Figure 4 use different target universes for the percentile F. To reproduce the class means, calculate F over the Figure 4 evaluation genes (1,345 for PANC RFX6). See `../geobridge/README.md` for the parity results and scripts. The five-class analysis then uses 1,339 of these genes, because six responders without a defined knockout fold change cannot be assigned a direction.

In the paper, T is written $T_{rg}=\operatorname{sign}(S^{(0),\mathrm{iso}}_{rg})\,F_r(G_{rg})$. In code, the Fig 4 score is `T_iso = sign(iso_C) * abs(Z|d4)` (`iso_C` = $S^{(0),\mathrm{iso}}_{rg}$, `Z` = $G_{rg}$) (`gb_kochip_audit_lib.load_unit`); `Z|d4` is a stored column whose absolute value equals F_r (the code asserts this; its sign comes from an alternative sign estimate, `d4`, that is not used in Fig. 4), so `T_iso` = sign(iso_C) x F_r. Comparisons with alternative sign estimators are in Supplementary Fig. S7; the signed analysis is exploratory (manuscript S1.5).

## Seeding
Trend bootstrap `default_rng([20261029, 41, n_modules])`; gene-bootstrap intervals reproduce `gb_kochip_audit.py` (`rng_for(unit, 'lev', i)`, global `SEED = 20261015`); near-isochronic pair sampling seeds 42/43/44 with `default_rng([seed, root_idx])`.

## Interpretation
No direct-regulation claim on real data; KO and ChIP only label genes for evaluation; the class ordering is descriptive. The sign is the unconditioned near-isochronic concordance; its comparison with alternative sign estimators on synthetic and knockout data is in Supplementary Fig. S7 (`../supplementary/`). Because those knockout endpoints overlap the Fig. 4 endpoints, the signed analyses are exploratory.

## Other scripts

- `gb_d1d4.py`: k=0 full-data S_0 and centred Sc_0 for every observational-universe target, both roots
- `gb_fourclass_signed_k0.py`: four-class split by KO direction at k=0
- `gb_geom_phase1.py`: cached score components and axis helpers (diagnostic only)
- `gb_polarity_compare.py`: polarity x method comparison on identical masks
- `gb_signed_decomp.py`: decompose signed R1 concordance S_k = N_k/D_k into component contributions
- `gb_signed_effects.py`: KO effect helpers: group ids and weighted ranks
