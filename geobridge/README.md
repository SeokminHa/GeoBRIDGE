# geobridge reproduction package

## What the score consumes (precise)
- Raw input: an untreated h5ad with integer **spliced** and **unspliced** count layers (+ species).
- Vendored canonical preprocessing (scVelo `moments` on the frozen kNN graph) turns these into **smoothed moments Ms (spliced) and Mu (unspliced)**, a 30-PC state embedding `X_pca`, and frozen cell pairs (two pseudotime roots `dc1_min`, `dc1_max`, with per-cell pseudotime `pt`).
- The score uses **only**: source regulator `Ms`, target `Mu`, the 30-PC embedding (for nuisance projection) and the pairs/pseudotime. **RNA velocity is NOT an input to the score** (scVelo velocity is computed by the preprocessing and stored, but no output here reads it). Do not describe velocity as a required input.
- Perturbation/KO/ChIP data are never read.

## Outputs of `geobridge score`
| output | file | definition |
|---|---|---|
| Z (PRIMARY, unsigned) | `--out` | min over the 2 roots of mean over depths k=0..8 of z(abs signed-concordance S_U(k)), z within each regulator's observational universe (forward pairs). NaN = UNSUPPORTED / outside universe / own gene |
| d_iso (sign of the observational association) | `<out-stem>.d_iso.tsv` (needs `--stratum-key`) | sign of S0 = N0/D0 (Ms of regulator vs Mu of target, no projection) on **isochronic pairs**: same stratum, abs(dpt) <= `--iso-window`; S0 averaged over 3 seeds then over the 2 roots. +1/-1; **0 = S0 exactly 0 (no sign); NaN = UNSUPPORTED** |
| T (signed composite) | `<out-stem>.T.tsv` | T = d_iso * F_r(Z), F_r = (midrank-0.5)/m over the m finite Z of that regulator's row. NaN if Z or d_iso is NaN; 0 where d_iso = 0 |

d_iso = +1 means a positive and -1 a negative observational association between the regulator's spliced and the target's unspliced moments. Neither d_iso nor T identifies direct activation or repression.

Missing-support rules (all NaN, never 0): own regulator gene; D <= 1e-12 at any seed/root; any of the 3 seeds or either root NaN; a root/seed with < 30 iso pairs (all genes NaN for that regulator; listed in the printed `unsupported_roots`); Z collapsed/unsupported (rules in `primary.py`). d_iso is computed for every gene; T additionally requires finite Z.

`--seed S` (default 42): first iso-pair sampling seed; seeds S, S+1, S+2 are averaged (production 42,43,44). Pair sampling uses `default_rng([seed, root_idx])`, so output is bitwise deterministic given `--seed`, `--iso-window` and the pairs file. Z itself uses no randomness. `--stratum-key NONE` = one stratum; otherwise an `obs` column (production: CARDIACV `embryo`, GATA1 `sample`, PANC `clusters`).

## Install / run
`pip install -e .` (env: environment.yml, versions from server jupyter_env).
```
# synthetic fixture (seconds; run from geobridge/). The pairs file has keys {root}_I, {root}_J, {root}_pt for both roots
# (fixture/synthetic_smoke_pairs.npz is written by make_fixture.py together with the h5ad). Without installing: PYTHONPATH=src python -m geobridge.cli score ...
geobridge score fixture/synthetic_smoke.h5ad --canonical --pairs fixture/synthetic_smoke_pairs.npz --regulators G0,G1,G2 --stratum-key embryo --out out.tsv
# -> out.tsv (Z), out.d_iso.tsv, out.T.tsv
# real unit from a canonical fixture + pairs (bitwise paper route), PANC RFX6; these files are not shipped: build them with figure3_knockout/README.md step 2 ($GB = that workspace)
geobridge score $GB/results/canonical_2026_09_18/PANC_canonical.h5ad --canonical --pairs $GB/results/canonical_2026_09_18/PANC_canonical_pairs.npz \
   --regulators RFX6 --stratum-key clusters --out panc_rfx6.tsv      # -> panc_rfx6.tsv, panc_rfx6.d_iso.tsv, panc_rfx6.T.tsv
# raw input (rank-level, not bitwise, reproduction; builds its own pairs incl. pt)
geobridge score raw.h5ad --species mouse --regulators RFX6 --stratum-key clusters --out o.tsv
```

## Resources (measured on a shared CPU node, 72 cores shared, jupyter_env)
- Synthetic fixture (400 cells x 60 genes, 3 regulators; Z + d_iso + T): 2.1 s wall, 0.58 GB peak RSS.
- Real unit PANC RFX6 (3,696 cells x 3,879 genes, 1 regulator): Z 364 s, d_iso (6 iso-pair engine runs, ~120k pairs each) 63 s, 4.4 GB peak RSS.

## Tests (`PYTHONPATH=src pytest tests`; `GB_REPO` = internal repo root for stored-artifact parity, optional: those tests are skipped when it is unset)
`PYTHONPATH=src pytest tests`: golden PRIMARY hash; vendored == pinned `zcons` (bitwise); fixture hashes (Z, d_iso, T); fixture CLI end-to-end; PANC real-data parity.
- PANC RFX6 d_iso vs stored pilot `iso_C` (`results/signed/iso_pilot_2026_10_15/scores/PANC_RFX6/features_PANC_RFX6.tsv`): S0 max abs diff 1.6e-15 (n=1345 genes), 0 sign mismatches.
- T vs stored `sign(iso_C) * F`: Spearman 1.0000, 0 sign mismatches, max abs diff 0.077. The CLI's T reproduces the paper's ranking exactly, but its values can differ: the percentile F in T is computed over the CLI's target universe (every gene with finite Z), whereas the Fig. 4 tables computed it over the evaluation gene universe (the evaluation mask with duplicate symbols removed; 1345 genes for PANC RFX6). The Fig. 4 per-gene tables were produced by `figure4_signed/scripts/gb_iso_pilot.py score` (`features_<unit>.tsv`, columns `iso_C`, `F`) and `figure4_signed/scripts/gb_grand_v3d_dumpT.py` (`T_<unit>.tsv`); see `figure4_signed/README.md`.
- `fixture/expected.json` primary hash was regenerated 2026-10-17: the Oct-9 value did not reproduce from `make_fixture.py` in the current env (also for the unmodified generator; thread-count independent). The fixture now also carries `obs['embryo']` and per-root `pt`.

## Figures
The figure scripts and their result tables are in `../figure2_synthetic`, `../figure3_knockout`, `../figure4_signed` and `../supplementary` (panel -> script list: `../MANIFEST.md`). The package reproduces the per-regulator Z / d_iso / T that feed those tables (PANC parity above).

## Vendored code
`src/geobridge/vendor/R` (canonical preprocessing/pairs; `SHA256SUMS`) and `src/geobridge/vendor/iso` (`gb_iso_lib.py`, `gb_isofull_lib.py`: byte-identical copies of the paper's scripts, `SHA256SUMS`); both verified at import. `iso.py` is plumbing only.

## Scope limits / known gaps
- Targets default to all genes in the fixture (observational universe applied per regulator); no custom `--universe` option in this CLI.
- Raw-PANC rebuild does not reproduce stored pairs bitwise (first divergence: scanpy PCA, rel 1.8e-5; PRIMARY Spearman 0.99997 vs stored). Use `--canonical --pairs` for bitwise production reproduction.
- Evaluation (nulls, KO/ChIP truth, ordinal classes, bootstraps) is not packaged; the scripts are in the figure folders of this repository (`../MANIFEST.md`).
- d_iso/T parity is shown for PANC RFX6 only (CARDIAC/GATA1 canonical inputs not wired into tests).
