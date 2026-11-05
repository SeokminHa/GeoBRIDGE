# Fig 3 re-evaluation on U_GB with pairwise comparisons — FROZEN SPEC (2026-10-26, before any new metric)
Origin: internal review universe audit (internal design note (not shipped)), approved 2026-10-26. Evaluation-only; no score recomputed or re-normalized.

Endpoints: the 9 SET9 endpoints (scratch/gb_ms_fig3_set9_2026_10_19.py EP9) with the B1000_fix PANC effect repair (scratch/gb_ms_fig3_set9_sens_2026_10_20.py effect_fixed). Truth definitions unchanged (current project rules).
PRIMARY universe per endpoint: U_GB = stored vector axis ∩ resp_ok ∩ unique uppercase symbol (drop all duplicate members) ∩ not the TF ∩ finite frozen GeoBRIDGE score.
PRIMARY comparison: for each comparator m in the 8 Fig 3b methods (|Pearson|, GENIE3, GRNBoost2, Velorama, RegDiffusion, DeepSEM, PIDC, FastTENET): pair set P_m = U_GB ∩ finite(m). On P_m compute GB and m: AUROC and AP lift (AP/prevalence), directions ALL / KO-up / KO-down (opposite-direction responders excluded, as canonical), raw scheme; paired bootstrap (genes resampled, B=1000, same draws for GB and m, seed as canonical) for GB−m difference with 95% CI. Report n(P_m), n+ per direction, coverage of m on U_GB. Also report GB alone on U_GB.
No cross-population ranking table: raw metrics of different m are not ranked against each other.
Aggregation: per-endpoint primary; summary = count of endpoints where GB−m CI>0 / CI<0 / includes 0, per m and direction; plus the canonical hierarchical aggregation (assay→TF→system, equal system weights, gb_ms_fig3_set9_2026_10_19.py:138) applied to GB−m differences.
Endpoints with <20 positives in a direction: reported, flagged descriptive.
SENSITIVITY: (S1) current all-9 intersection — must reproduce B1000_fix per_endpoint.tsv exactly (harness check); (S2) all-8 without Velorama — must reproduce B1000_fullcov where available.
Small-N gate: PANC_RFX6 and HEART_NKX2-5, B=200: verify S1 parity, U_GB sizes, label/order equality asserts; only then full.
Not changed: scores, z-normalization, truth rules, method set, metrics. Exploratory until internal review code review.
