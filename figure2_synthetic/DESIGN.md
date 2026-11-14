# Figure 2 benchmark design

This summary follows `scripts/gb_tf32rep_simulator.py` and `scripts/gb_tf32rep.py`.
The generator header refers to an earlier benchmark (20 regulators, 193 edges, 39 candidates); `generate_grn` defines the Figure 2 network.

## Question
Does a score computed from unperturbed spliced/unspliced counts identify direct regulator -> target edges when cells share developmental variation
(a time-shared and a branch-shared hidden input) that makes non-regulatory pairs co-vary? Truth = the simulated GRN, evaluated only after scoring.

## Simulator
* SERGIO (steady-state differentiation graphs, patched for modern numpy: `../patches/`), topologies `bifurcation` (Fig. 2) and `tree` (SI);
  4 cell types x 300 cells = 1,200 cells (bifurcation), then UMI rendering at depth 10.
* GRN (`generate_grn(g)`): 100 genes; 32 candidate regulators = 32 active regulators (5 master regulators + 27 intermediates in layers of size 7, 7, 7, 6), 0 decoy regulators;
  edges regulator -> any strictly higher layer gene (DAG; only master regulators receive no regulation); source degrees heterogeneous (per-regulator lognormal weight);
  exactly E = 240 direct edges (mandatory cascade edges first, remaining edges sampled without replacement with probability proportional to source weight);
  universe = 32 regulators x 99 non-self targets = 3,168 pairs, prevalence 240/3168 = 0.0758; edge strengths K ~ U(1,5), random sign.
* Cohort: GRN seeds 501..560 are audited graph-only (`audit`): valid DAG, E = 240, every non-master gene has in-degree >= 1, every regulator out-degree >= 1, >= 100 strictly 2-hop and >= 50 strictly 3-hop regulator-target pairs;
  the first six passing seeds (501-506) are the cohort (no other criterion; `results/grn_seeds.json`, produced by the `audit` step of the driver). A distortion report compares the exact-240 generator with its Bernoulli-density predecessor.
* Hidden inputs ("AB" design, `q = 2` hidden driver genes, loading lambda = 1): A(t) = time-shared input, rate 0.3 + 3.5 d (d = normalised pseudotime depth of the cell-type bin, monotone);
  B = branch-shared input, rate 0.3 + 3.5 (0.5 + 0.5 b g) with g = 0 on the trunk and 1 on branches, b = +1/-1 per branch (signed after centring).
  Hidden driver genes are removed from the output and from the truth. Each of the 100 genes receives dense mixed-sign loadings U(1,3) x (+-1) from A (`RandomState(777+grn)`) and B (`RandomState(31337+grn)`).
* Conditions (strength of A, strength of B): C0 (0, 0) none, C1 (0.5, 0) temporal only, C2 (0, 0.5) branch only, C3 (0.5, 0.5) both.
* Replicates: 2 simulations per (GRN, topology, condition), SERGIO `np.random.seed(1000*grn + sim)`. Total 6 GRNs x 2 topologies x 2 sims x 4 conditions = 96 datasets (48 bifurcation).
* Observation: spliced and unspliced UMI counts rendered with a dedicated PCG64 stream per (grn, sim, lambda); no velocity is used as input.

## Methods scored on each dataset (same 3,168 pairs)
GeoBRIDGE unsigned score Z ($G_{rg}$ in the paper, called PRIMARY in the scripts; k = 0..8 depth average, minimum over the two roots; canonical preprocessing `common/GEOBRIDGE_PREPROCESS_V1.json`), Velorama (rt0 configuration), FastTENET, GENIE3, GRNBoost2, PIDC, RegDiffusion, DeepSEM
(and |Pearson|, dropped from the final figure). Comparator inputs and configurations are in `scripts/gb_sota_ext.py`, `gb_velorama_official_rt0.py`, `gb_tf32rep.py` (`ext`).

## Metrics and inference (`scripts/fig2_panels.py`)
* Per dataset: AUPRC and AUROC of the score against the 240 direct edges over the 3,168 pairs; direct-vs-indirect AUROC (direct edges vs ALL indirect pairs, i.e. regulator-target pairs at graph distance 2 to 5; computed by `scripts/fig2f.py` from the per-pair tables of `scripts/gb_dvi_pairs.py`). An earlier variant compared direct edges with 2- and 3-hop pairs only; its tables are kept as the parity reference.
* Aggregation: mean over the 2 simulations -> per GRN; mean over 6 GRNs; 95% t interval (n = 6); two-sided paired t-test over the 6 GRNs for GeoBRIDGE minus each comparator;
  Holm correction over the 7 comparators per condition and metric. Six GRNs give coarse resolution (smallest exact sign-flip p = 0.031).
* The reference output of the final run is `results/TABLE.tsv` (864 rows = 96 datasets x 9 methods), `results/D1_vs_D2D3_datasets.tsv`, `results/Fig2_values_holm7.tsv`, `results/fig2f_dviall/Fig2f_DvIall_values.tsv`.

## Seeds
All generator and comparator seeds are listed in `README.md` ("Seeding"). The 96 datasets were produced by the generator file whose sha256 is recorded in each dataset's `record.json`;
the copy here differs only in file paths and comments.
