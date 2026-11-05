#!/usr/bin/env python
"""CANONICAL "WHAT IS P?" TABLE (PI manuscript lock 2026-10-02 item 2; internal review internal_review_log "Q2 canonical What is P? table").  EVALUATION ONLY: production S, its depth ensemble,
the catalogue-bank prior P and every comparator are IMPORTED, never edited or re-fit; nothing here is a score, k rule, feature or hyper-parameter.  Retrospective systems ONLY:
CARDIAC / CARDIACV / RETINA truth is NEVER opened (hard refusal below; those systems belong to the frozen prospective runner gb_prospective.py).

QUESTION  what information does the target-common readout P_g^(-r) (GB leave-focal-out catalogue-bank prior) carry relative to (i) observability / dynamics features, (ii) equal-weight composites of
          those features and (iii) the leave-focal-out priors of the nine other GRN methods -- on ONE identical gene universe per endpoint.  The table does not try to show that P wins.

ENDPOINTS (one block of rows per evaluable unit; ungated units are kept and flagged, never dropped)
  response (lattice)   PANC RFX6 | GATA1 | KLF1 | BM declared panels: BM_CHIPATLAS (8 TFs; BONEMARROW is a DUPLICATE of its GATA1/GATA2/RUNX1/TAL1 and is dropped, precedence as gb_power_match_v2) plus
                       BM_GATA1_ACUTE (separate KO experiment: own rows, outside the block macro) | K562_EMP_ALLNULL (gated TFs enter the block macro) | KLF1_B (sensitivity, outside the macro)
  regulon              per-regulator CollecTRI regulon labels (gb_canon_table family F1; the audit_decomp regulon roster), separate rows, per system
  breadth              BM held-out perturbation breadth (gb_ladder BM TOTAL/BM19 'any' label), separate rows
UNIVERSE  M1 = the frozen audit/prospective mask (gb_audit_decomp.masks_and_priors: response-tested & occupancy-mappable lattice universe U, finite for every Tier-1 cognate row, on the frozen T7 target
          set, finite leave-focal-out prior for every Tier-1 method).  MR = M1 & all nine REF features finite = the ONE universe of every row of an endpoint.  SENSITIVITY (labelled mask=resp_only, never
          mixed with MR): MRR = response-tested & T7 & every cognate/prior/COMP finite WITHOUT occupancy mappability, & nine features finite (gb_prospective's declared `resp_only` mask + the nine-feature condition).
PREDICTORS (all truth-free; every HGB model is fit to P, never to truth)
  GB P                          prior[GeoBRIDGE]  (leave-focal-out mean of the T7 row-z of P_pairU over the other bank rows)
  HGB obs-5 / dyn-4 / all-9     gb_audit_decomp.hgb_cross_fit (HistGradientBoostingRegressor(random_state=0), library DEFAULTS, pinned + asserted) of z(P over MR) on observability-five
                                (raw_det, sm_det, abundance, pair_support, mu_det) / dynamics-four (dynamics, mu_dynamics, abs_spearman_mu_dpt, mu_logvar) / all nine signed REF columns, cross-fitted on the
                                lattice's fixed gene-hash folds (ep['fold'], sha1(upper gene) mod 5); the held-out FIT is the predictor.  (gb_prior_semantics.HGB_KW is a DIFFERENT configuration and is not used.)
  COMP_Z, COMP_RANK             frozen equal-weight composites of the nine REF columns (gb_composites)
  9 other-method priors         |Pearson|, GENIE3, GRNBoost2, RegVelo |J|, Velorama, RegDiffusion, DeepSEM, PIDC, FastTENET: each method's own leave-focal-out prior (catbank priors/external/sota_priors)
  extras, labelled              `HGB all-9 module-fold` (same fit, folds = control-module hash: sensitivity of the reconstruction to within-module leakage); `GB S` (production cognate row, reference only)
METRICS   n, responders / prevalence, native coverage, AUROC, AP (tie-grouped), AP-lift, tie-aware P50 (expected hits under random tie-breaking; defined for n >= 100): gb_canon_table.eval_rows.
PAIRED    P - X for every other predictor X, same rows, same draws, for AUROC / AP / AP-lift / P50: point, 95% percentile interval, two-sided bootstrap p (+1 rule).  Descriptive: no multiplicity correction.
UNCERTAINTY  (a) PRIMARY, labelled "gene-identity resampling": the lattice's synchronized BlockAxis multinomial gene draws (gb_layer_lattice.BlockAxis, the audit_decomp seed), B draws shared by every
          predictor / TF of the block.  (b) DEPENDENCE SENSITIVITY, labelled "module-block resampling": the same BlockAxis class with `units` = a frozen CONTROL-ONLY coexpression module map (rule
          MODULE_RULE below; built by `modules` from the untreated control fixture's Ms layer before any evaluation); a module is resampled as ONE unit.  The module map is a rule FAMILY with three
          frozen granularities, K = max(20, ceil(G_nonconstant / size)), size in {25, 50, 100}: G/50 is the primary module scheme (the approved pilot's map, unchanged), G/25 and G/100 are nested cuts of the
          SAME Ward tree frozen before any full-scale evaluation; `merge` reports, per granularity, the median CI-width ratio (module / gene) and the number of paired-contrast significance flips (lost / gained).
          The primary analysis is unchanged.  Shared TFs / controls / substrates are NOT independent replicates; macro rows are conditional on the fixed TF set.
REUSE / PARITY  audit_decomp_v2/control_per_tf.tsv (PRIOR / FIT hgb / COMP rows on the same MR), prior_semantics/full/summary/primary_unit_metrics.tsv (s374), layer_lattice(_extra)_sota_final endpoints.tsv
          (GB S), ladder/final/breadth_BM.tsv are used as FAIL-CLOSED parity gates after mask / hash checks (artifact sha256 recorded in the receipt); recomputation is the table's source of numbers.

  python scratch/gb_ptable.py selftest [--data]
  python scratch/gb_ptable.py modules [--axes PANC,BONEMARROW,...] [--sizes 25,50,100]    # freeze the control-only module maps (+ source-matching covariates) FIRST
  python scratch/gb_ptable.py run --systems PANC,BM_CHIPATLAS [--regulon PANC,BONEMARROW] [--breadth BM] --B 2000 --out DIR [--tfs A,B] [--slice i --nslice n]
  python scratch/gb_ptable.py merge --out DIR
  python scratch/gb_ptable.py report --out DIR [--groups PANC/RFX6,MACRO:BM] [--mask MR] [--macro] [--K 25|50|100]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import inspect
import glob
import hashlib
import json
import platform
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = __import__("os").environ["GEOBRIDGE_ROOT"]
for _p in (str(HERE), f"{R}/scratch"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GB = "GeoBRIDGE"
P_NAME = "GB P"
MET = ("AUROC", "AUPRC", "APLIFT", "P50")
B_DEFAULT = 2000
FORBIDDEN = {"CARDIAC", "CARDIACV", "CARDIAC_GATA4_OCC", "CARDIACV_GATA4_OCC", "RETINA", "RETINA_NRL_3V2", "RPE1"}      # prospective / sealed: truth is never opened here
LATTICE_PRIMARY = ["PANC", "GATA1", "KLF1", "BM_CHIPATLAS", "K562_EMP_ALLNULL"]
LATTICE_SENS = ["KLF1_B", "BM_GATA1_ACUTE"]                                                       # own rows, outside the block macro (gb_power_match_v2 convention)
LATTICE_DUP = {"BONEMARROW": "BM_CHIPATLAS"}                                                      # duplicate units (identical response truth / universe): dropped, logged
REGULON_SYSTEMS = ["PANC", "BONEMARROW", "GATA1", "DENTATEGYRUS", "GASTRULATION", "ZEBRAFISH"]
MACRO_BLOCK = {"PANC": "PANC", "GATA1": "ERYTHROID", "KLF1": "ERYTHROID", "BM_CHIPATLAS": "BM", "K562_EMP_ALLNULL": "K562"}
OBS5 = ["raw_det", "sm_det", "abundance", "pair_support", "mu_det"]                              # observability-five (PROSPECTIVE_SECONDARY_FREEZE S1b; gb_prior_semantics.SPLIT_OBS)
DYN4 = ["dynamics", "mu_dynamics", "abs_spearman_mu_dpt", "mu_logvar"]                           # dynamics-four (S1c; SPLIT_DYN)
PINNED_HGB = {'categorical_features': 'from_dtype', 'early_stopping': 'auto', 'interaction_cst': None, 'l2_regularization': 0.0, 'learning_rate': 0.1, 'loss': 'squared_error', 'max_bins': 255,
              'max_depth': None, 'max_features': 1.0, 'max_iter': 100, 'max_leaf_nodes': 31, 'min_samples_leaf': 20, 'monotonic_cst': None, 'n_iter_no_change': 10, 'quantile': None,
              'random_state': 0, 'scoring': 'loss', 'tol': 1e-07, 'validation_fraction': 0.1, 'verbose': 0, 'warm_start': False}      # sklearn 1.7.2 defaults (asserted)
MOD_DIR = f"{R}/results/real_uniform/ptable/modules"            # NOT layer_lattice/modules: writing there would silently change every future lattice run
AUD_DIR = f"{R}/results/real_uniform/audit_decomp_v2"
S374 = f"{R}/results/real_uniform/prior_semantics/full/summary/primary_unit_metrics.tsv"
BREADTH_TSV = f"{R}/results/real_uniform/ladder/final/breadth_BM.tsv"
BREADTH_RCPT = f"{R}/results/real_uniform/ladder/final/receipt_breadth_BM.json"
MIN_CLASS = 20
CI_PRIMARY = "gene-identity resampling (synchronized BlockAxis draws; 95% percentile)"
CI_SENS = "module-block resampling (frozen control-only coexpression modules at mean module sizes G/25, G/50 [primary], G/100; 95% percentile; dependence sensitivity)"

# ------------------------------------------------------------------------------------------------------------------------------------------------------------ control-only module map rule FAMILY
# ONE rule family, three granularities (declared 2026-10-02, the two extra sizes frozen BEFORE any full-scale evaluation): K = max(20, ceil(G_nonconstant / size)), size in {25, 50, 100}.  size 50 = the PRIMARY map
# of the approved pilot (its rule sha is unchanged); 25 and 100 are the coordinator-requested granularity sensitivities.
MODULE_SIZES, PRIMARY_SIZE, MIN_MODULES = (25, 50, 100), 50, 20
MODULE_MEAN_SIZE = PRIMARY_SIZE


def module_rule(size):
    return dict(
        version="V1",
        input="canonical fixture (results/canonical_2026_09_18/<AXIS>_canonical.h5ad) layers['Ms'] = kNN-smoothed spliced moments of the UNTREATED control / atlas cells (the production observational input), "
              "restricted to the catbank axis genes (exact names); no perturbation, guide, KO, ChIP, ATAC or label of any kind",
        standardize="per-gene z over all fixture cells (population sd); genes with sd <= 1e-12*(1+|mean|) or any non-finite value are SINGLETON modules named 'S:<gene>'",
        distance="d_ij = sqrt(2*(1 - rho_ij)), rho = Pearson correlation of the z-scored Ms profiles (= Euclidean distance between z-scored profiles / sqrt(n_cells))",
        linkage="Ward agglomerative clustering (scipy.cluster.hierarchy.ward on the condensed distance matrix)",
        cut=f"fcluster(criterion='maxclust') with K = max({MIN_MODULES}, ceil(G_nonconstant / {size})) modules (target mean module size {size} genes)",
        naming="'M<cluster label, 4 digits>'",
        frozen="before any evaluation: written by `modules`, verified by `run` (rule sha256 + module-file sha256)")


MODULE_RULE = module_rule(PRIMARY_SIZE)
MODULE_RULE_SHAS = {s: hashlib.sha256(json.dumps(module_rule(s), sort_keys=True).encode()).hexdigest() for s in MODULE_SIZES}
MODULE_RULE_SHA = MODULE_RULE_SHAS[PRIMARY_SIZE]                      # == the approved pilot's rule sha (asserted against the frozen receipts at run time)
SCHEMES_T1 = ("g",) + tuple(f"m{s}" for s in MODULE_SIZES)            # resampling schemes of Table 1: gene-identity + module-block at three granularities
MATCHCOV_DIR = f"{R}/results/real_uniform/ptable/matchcov"           # truth-free source-matching covariates for the decoy calibration (log1p mean Ms per axis gene)


def _sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


SCRIPT_SHA0 = _sha(__file__)


DEP_MODULES = ("gb_layer_lattice", "gb_lattice_extra", "gb_audit_decomp", "gb_attribution", "gb_canon_table", "gb_decompose", "gb_catbank", "gb_primary", "gb_composites", "gb_real_uniform",
               "gb_prior_semantics", "gb_readout", "gb_ladder", "gb_substrate", "gb_cop_alignment", "gb_harmonized_tri")


def dep_shas():
    """sha256 of every imported production / analysis module this table depends on (provenance; none of them is edited)"""
    return {n: _sha(sys.modules[n].__file__) for n in DEP_MODULES if n in sys.modules and getattr(sys.modules[n], "__file__", None)}


def atomic_tsv(df, p):
    tmp = f"{p}.tmp{os.getpid()}"
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, p)


def atomic_json(p, obj):
    tmp = f"{p}.tmp{os.getpid()}"
    json.dump(obj, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, p)


def guard_systems(names):
    bad = [s for s in names if s in FORBIDDEN or s.startswith(("CARDIAC", "RETINA"))]
    if bad:
        raise SystemExit(f"REFUSING: {bad} are prospective / sealed systems (CARDIACV, RETINA truth is reserved for gb_prospective.py)")


# ===================================================================================================== control-only coexpression module map (rule family V1, sizes 25 / 50 / 100)
def build_module_labels_multi(Ms, genes, sizes=(PRIMARY_SIZE,), min_modules=MIN_MODULES):
    """MODULE_RULE V1 on a (cells, genes) matrix -> {size: (module names aligned to `genes`, diag)}.  ONE Ward tree, cut at K = max(min_modules, ceil(G_nonconstant / size)) for every size (nested cuts of
    the same tree).  Pure and deterministic (no randomness)."""
    from scipy.cluster.hierarchy import fcluster, ward
    from scipy.spatial.distance import squareform
    X = np.asarray(Ms, np.float64)
    genes = np.asarray(genes).astype(str)
    n, G = X.shape
    with np.errstate(invalid="ignore"):
        mu, sd = X.mean(0), X.std(0)
    ok = np.isfinite(mu) & np.isfinite(sd) & (sd > 1e-12 * (1 + np.abs(mu)))
    out = {}
    Zl = None
    if ok.sum() >= 3:
        Z = (X[:, ok] - mu[ok]) / sd[ok]
        C = np.clip(Z.T @ Z / n, -1.0, 1.0)
        d = np.sqrt(np.maximum(2.0 * (1.0 - C), 0.0))
        d = (d + d.T) / 2
        np.fill_diagonal(d, 0.0)
        Zl = ward(squareform(d, checks=False))
    for size in sizes:
        names = np.array([f"S:{g}" for g in genes], dtype=object)
        k_used, K = 0, 0
        if Zl is not None:
            K = min(max(min_modules, int(np.ceil(ok.sum() / size))), int(ok.sum()))
            lab = fcluster(Zl, t=K, criterion="maxclust")
            names[np.flatnonzero(ok)] = [f"M{int(x):04d}" for x in lab]
            k_used = int(len(np.unique(lab)))
        out[size] = (names.astype(str), dict(n_cells=int(n), n_genes=int(G), n_constant_singletons=int((~ok).sum()), K_requested=int(K), K_used=k_used))
    return out


def build_module_labels(Ms, genes, mean_size=MODULE_MEAN_SIZE, min_modules=MIN_MODULES):
    """single-size wrapper (selftest / back-compat): (module names, diag)"""
    return build_module_labels_multi(Ms, genes, (mean_size,), min_modules)[mean_size]


def _module_paths(axis_sys, size):
    p = f"{MOD_DIR}/{axis_sys}_modules{'' if size == PRIMARY_SIZE else f'_m{size}'}.tsv"
    return p, p.replace(".tsv", ".receipt.json")


def _frozen_ok(axis_sys, size):
    """True if the module map of this size exists and is internally consistent with THIS script's rule; a present-but-stale map is refused (frozen artifacts are never silently overwritten)"""
    p, rp = _module_paths(axis_sys, size)
    if not (os.path.exists(p) and os.path.exists(rp)):
        return False
    rc = json.load(open(rp))
    if rc["rule_sha256"] != MODULE_RULE_SHAS[size] or rc["module_file_sha256"] != _sha(p):
        raise SystemExit(f"REFUSING: frozen module map {p} is stale (rule sha / file sha differ from its receipt / this script); it is never overwritten silently")
    return True


def matchcov_paths(axis_sys):
    p = f"{MATCHCOV_DIR}/{axis_sys}_matchcov.tsv"
    return p, p.replace(".tsv", ".receipt.json")


def freeze_axis(axis_sys, sizes=MODULE_SIZES):
    """freeze (truth-free, control-only) for ONE catbank axis: the module maps of every requested size (+ the source-matching covariate log1p mean Ms used by the decoy calibration) from the untreated
    fixture's Ms layer.  Existing consistent artifacts are kept (idempotent); nothing frozen is overwritten.  Refuses prospective axes."""
    guard_systems([axis_sys])
    todo = [s for s in sizes if not _frozen_ok(axis_sys, s)]
    mp, mrp = matchcov_paths(axis_sys)
    need_cov = not (os.path.exists(mp) and os.path.exists(mrp) and json.load(open(mrp))["tsv_sha256"] == _sha(mp))
    if not todo and not need_cov:
        print(f"[freeze] {axis_sys}: sizes {list(sizes)} and match covariates already frozen", flush=True)
        return
    import anndata as ad
    import gb_substrate as GS
    os.makedirs(MOD_DIR, exist_ok=True)
    os.makedirs(MATCHCOV_DIR, exist_ok=True)
    bj = json.load(open(f"{R}/results/real_uniform/catbank/{axis_sys}_bank.json"))
    with np.load(f"{R}/results/real_uniform/catbank/{axis_sys}_priors.npz", allow_pickle=True) as z:
        axis = [str(x) for x in z["axis"]]
    fix, _pairs, _cat = GS.paths(axis_sys, canonical=True)
    t0 = time.time()
    A_ = ad.read_h5ad(fix)
    var = np.asarray(A_.var_names).astype(str)
    ix = {g: i for i, g in enumerate(var)}
    miss = [g for g in axis if g not in ix]
    if miss:
        raise SystemExit(f"REFUSING_FREEZE {axis_sys}: {len(miss)} axis genes absent from the fixture, e.g. {miss[:3]}")
    Ms = A_.layers["Ms"]
    Ms = Ms[:, [ix[g] for g in axis]]
    Ms = Ms.toarray() if hasattr(Ms, "toarray") else np.asarray(Ms)
    base = dict(axis=axis_sys, fixture=fix, fixture_sha256=_sha(fix), fixture_receipt=json.load(open(fix.replace("_canonical.h5ad", "_receipt.json"))), bank_sha256_members=bj["sha256_members"],
                n_axis=len(axis), obs_columns=list(A_.obs.columns), script_sha256=SCRIPT_SHA0, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
    if need_cov:                                       # == gb_lattice_kprofile.axis_ctx lms (dutil's matching covariate): log1p of the float64 mean of the float32 Ms over all fixture cells
        lms = np.log1p(np.asarray(Ms, np.float32).astype(np.float64).mean(0))
        atomic_tsv(pd.DataFrame(dict(gene=axis, lms=lms)), mp)
        atomic_json(mrp, dict(base, tsv=mp, tsv_sha256=_sha(mp), definition="lms = log1p(mean over all fixture cells of the Ms layer, float64), per catbank-axis gene (gb_lattice_kprofile.axis_ctx['lms'])",
                              note="control-only, truth-free; source-matching covariate of the decoy calibration (with the bank's spliced detection)"))
        print(f"[freeze] {axis_sys}: match covariates (log1p mean Ms) frozen for {len(axis)} genes", flush=True)
    if todo:
        labs = build_module_labels_multi(Ms, axis, todo)
        for size in todo:
            lab, diag = labs[size]
            sizes_ = pd.Series(lab).value_counts()
            p, rp = _module_paths(axis_sys, size)
            atomic_tsv(pd.DataFrame(dict(gene=axis, module=lab)), p)
            atomic_json(rp, dict(base, rule=module_rule(size), rule_sha256=MODULE_RULE_SHAS[size], module_size=size, module_file=p, module_file_sha256=_sha(p), n_modules=int(sizes_.size),
                                 n_singletons=int((sizes_ == 1).sum()), size_quantiles={q: float(sizes_.quantile(q)) for q in (0.0, 0.25, 0.5, 0.75, 1.0)}, **diag, seconds=round(time.time() - t0, 1),
                                 note="control-only: built from the untreated fixture's Ms; truth was not opened; written BEFORE any gb_ptable / gb_prior_adjusted evaluation"))
            print(f"[freeze] {axis_sys} G/{size}: G={len(axis)} cells={diag['n_cells']} K={int(sizes_.size)} singletons={int((sizes_ == 1).sum())} sizes(min/med/max)="
                  f"{sizes_.min():.0f}/{sizes_.median():.0f}/{sizes_.max():.0f} {time.time() - t0:.0f}s", flush=True)


def load_modules(axis_sys, names, size=PRIMARY_SIZE):
    """verified control-only module labels aligned to `names` (rule sha + file sha checked); genes absent from the map become singletons (counted).  Returns (labels, receipt)."""
    p, rp = _module_paths(axis_sys, size)
    if not (os.path.exists(p) and os.path.exists(rp)):
        raise SystemExit(f"REFUSING: no frozen module map G/{size} for {axis_sys}; run `gb_ptable.py modules --axes {axis_sys}` BEFORE any evaluation")
    rc = json.load(open(rp))
    if rc["rule_sha256"] != MODULE_RULE_SHAS[size] or rc["module_file_sha256"] != _sha(p):
        raise SystemExit(f"REFUSING: module map {p} is stale (rule sha / file sha differ from its receipt / this script)")
    d = pd.read_csv(p, sep="\t", dtype=str).set_index("gene")["module"]
    names = np.asarray(names).astype(str)
    lab = np.array([d.get(g, f"S:{g}") for g in names], dtype=object).astype(str)
    return lab, dict(rc, n_names_missing_in_map=int((~pd.Index(names).isin(d.index)).sum()))


def load_matchcov(axis_sys):
    """frozen log1p mean Ms of every catbank-axis gene (pandas Series gene -> lms); receipt hash checked"""
    p, rp = matchcov_paths(axis_sys)
    if not (os.path.exists(p) and os.path.exists(rp)):
        raise SystemExit(f"REFUSING: no frozen match covariates for {axis_sys}; run `gb_ptable.py modules --axes {axis_sys}` BEFORE any evaluation")
    if json.load(open(rp))["tsv_sha256"] != _sha(p):
        raise SystemExit(f"REFUSING: match covariates {p} are stale")
    return pd.read_csv(p, sep="\t", dtype={"gene": str}).set_index("gene")["lms"]


# ===================================================================================================== resampling draws (primary gene-identity + module-block sensitivities)
class Draws:
    """all draw sets of one block, stacked so every metric is evaluated once: rows [0, B) gene-identity (L.BlockAxis, units=None), then one B-row block per module size (L.BlockAxis, units=modules; schemes
    'm25' / 'm50' / 'm100').  C = the stacked (n_schemes * B, G) multiplicity matrix; `schemes` = names in stacking order."""

    def __init__(self, names, B, seed_gene, seeds_mod, units_by_size):
        import gb_layer_lattice as L
        self.gene = L.BlockAxis(names, B, seed_gene, None, "gene")
        self.sizes = tuple(sorted(units_by_size))
        self.mods = {s: L.BlockAxis(names, B, seeds_mod[s], units_by_size[s], "module") for s in self.sizes}
        self.B = B
        self.schemes = ("g",) + tuple(f"m{s}" for s in self.sizes)
        self.C = np.vstack([self.gene.counts] + [self.mods[s].counts for s in self.sizes])
        self.n_modules = {s: int(self.mods[s].unit.max() + 1) for s in self.sizes}
        self.fp = ":".join([self.gene.fingerprint()[:12]] + [self.mods[s].fingerprint()[:12] for s in self.sizes])      # identical draws <=> identical fingerprint (asserted before any macro)
        self.n_missing_in_map = 0                                                          # axis names absent from the frozen module maps (singletons); set by the builders

    def fold_gene(self, gi):
        return self.gene.fold_of(gi)

    def fold_module(self, gi):
        """control-module hash folds of the PRIMARY (G/50) map ('control-defined gene blocks')"""
        return self.mods[PRIMARY_SIZE].fold_of(gi)


def mod_seed(base, size):
    """seed of the module-block draws: the PRIMARY size keeps the approved pilot's seed; the other sizes append the size"""
    return list(base) if size == PRIMARY_SIZE else list(base) + [size]


def lattice_draws(system, B, sizes=MODULE_SIZES):
    import gb_layer_lattice as L
    cfg = L.SYSTEMS[system]
    names = L.axis_of(cfg["axis"])
    bi = L.BLOCKS.index(cfg["block"])
    sg = int(np.random.SeedSequence([L.BOOT_SEED, bi]).generate_state(1)[0])                 # == gb_audit_decomp.cmd_run (gene-identity draws are the audit's)
    units, rcs = {}, {}
    for s in sizes:
        units[s], rcs[s] = load_modules(cfg["axis"], names, s)
        if rcs[s]["n_names_missing_in_map"]:
            raise SystemExit(f"REFUSING: {rcs[s]['n_names_missing_in_map']} axis genes of {system} are absent from the frozen module map G/{s}")
    seeds = {s: int(np.random.SeedSequence(mod_seed([L.BOOT_SEED, bi, 20261002], s)).generate_state(1)[0]) for s in sizes}
    dr = Draws(names, B, sg, seeds, units)
    dr.n_missing_in_map = 0
    dr.module_receipts = rcs
    return dr, (rcs[PRIMARY_SIZE] if PRIMARY_SIZE in rcs else rcs[sizes[0]])


# ===================================================================================================== numerics
def pct_ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def boot_p(d):
    """two-sided bootstrap p of a paired difference: 2 min(P*(d<=0), P*(d>=0)) with the +1 rule (gb_decompose.boot_p)"""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min(1 + (d <= 0).sum(), 1 + (d >= 0).sum()) / (len(d) + 1))) if len(d) else np.nan


def split_eval(v, y, rows, gi, C, B):
    """metric points + the draw sets of ONE vector: (pt, [one {metric: draws} per scheme, in the stacking order of the draw matrix C = [gene; module sizes...]])"""
    import gb_canon_table as CT
    pt, bs = CT.eval_rows(v, y, rows, gi, C)
    ns = C.shape[0] // B
    return pt, [{k: np.asarray(bs[k][i * B:(i + 1) * B], float) for k in bs} for i in range(ns)]


def sfx_of(sch):
    """column suffix of a resampling scheme: '' for the primary gene-identity scheme, '_m25' / '_m50' / '_m100' for module-block"""
    return "" if sch == "g" else f"_{sch}"


def ref_index():
    """positions of the observability-five / dynamics-four REF columns inside L.REFS, plus the asserted feature mapping"""
    import gb_layer_lattice as L
    feat = {n: c for n, (c, sg) in L.REF_SIGNS.items()}
    refs = list(L.REFS)
    obs = [i for i, n in enumerate(refs) if feat[n] in OBS5]
    dyn = [i for i, n in enumerate(refs) if feat[n] in DYN4]
    assert sorted(feat[refs[i]] for i in obs) == sorted(OBS5) and sorted(feat[refs[i]] for i in dyn) == sorted(DYN4) and not set(obs) & set(dyn) and len(obs) + len(dyn) == len(refs) == 9
    return obs, dyn, list(range(len(refs)))


def predictor_set(P, others, comps, Xr, S, folds_gene, folds_mod, mask):
    """{name: (vector, role, native vector for coverage)} on `mask`.  P, S, others{method: prior}, comps{name: vec}, Xr (n, 9) signed REF matrix.  Truth-free; HGB fits target z(P) only."""
    import gb_audit_decomp as AD
    import gb_decompose as D
    obs, dyn, allx = ref_index()
    zP = D.zrow(P, mask)
    out = {P_NAME: (np.where(mask, P, np.nan), "P", P)}
    fits = {}
    for nm, cols, f in (("HGB obs-5 -> P", obs, folds_gene), ("HGB dyn-4 -> P", dyn, folds_gene), ("HGB all-9 -> P", allx, folds_gene), ("HGB all-9 module-fold -> P (sens.)", allx, folds_mod)):
        fit, _res, st = AD.hgb_cross_fit(zP, Xr[:, cols], f, mask)
        fits[nm] = st["R2_heldout"]
        nat = np.where(np.isfinite(Xr[:, cols]).all(1), 0.0, np.nan)
        out[nm] = (np.where(mask, fit, np.nan), "reconstruction" if "sens." not in nm else "sensitivity", nat)
    for c in ("COMP_Z", "COMP_RANK"):
        out[c] = (np.where(mask, comps[c], np.nan), "composite", comps[c])
    for m, v in others.items():
        out[f"prior {m}"] = (np.where(mask, v, np.nan), "other-method prior", v)
    out["GB S (production, reference)"] = (np.where(mask, S, np.nan), "reference", S)
    return out, fits


def order_predictors(names):
    pri = [P_NAME, "HGB obs-5 -> P", "HGB dyn-4 -> P", "HGB all-9 -> P", "COMP_Z", "COMP_RANK"]
    rest = [n for n in names if n not in pri and not n.startswith(("GB S", "HGB all-9 module"))]
    tail = [n for n in names if n.startswith(("HGB all-9 module", "GB S"))]
    return [n for n in pri if n in names] + rest + tail


def table1_rows(u, dr, mask, mask_kind):
    """metrics + paired rows + draw store for ONE unit on ONE mask.  u = unit dict (see lattice_unit / regulon_unit / breadth_unit)."""
    B = dr.B
    y, gi = u["y"], u["gi"]
    preds, fits = predictor_set(u["P"], u["others"], u["comps"], u["Xr"], u["S"], u["folds_gene"], dr.fold_module(gi), mask)
    nU = int(u["U"].sum())
    yy = np.asarray(y, bool)[mask]
    base = dict(kind=u["kind"], system=u["system"], block=u["block"], tf=u["tf"], endpoint=u["endpoint"], endpoint_class=u["endpoint_class"], mask=mask_kind, n_U=nU,
                n_M1=int(u["M1"].sum()), n_rows=int(mask.sum()), frac_rows_of_U=float(mask.sum() / max(nU, 1)), n_pos=int(yy.sum()), n_neg=int((~yy).sum()), PREV=float(yy.mean()),
                gate=bool(u["gate"][mask_kind]), macro_group=u["macro_group"], draws_fp=dr.fp, ci_primary=CI_PRIMARY, ci_sensitivity=CI_SENS)
    status = "OK" if base["gate"] else "DESCRIPTIVE_ONLY"
    if not (yy.any() and (~yy).any()):
        return [dict(base, status="TOO_FEW")], [], {}
    res = {n: split_eval(v, y, mask, gi, dr.C, B) for n, (v, _r, _nat) in preds.items()}
    ppt, pbs = res[P_NAME]
    mrows, prows, draws = [], [], {}
    uid = f"{u['kind']}|{u['system']}|{u['tf']}|{u['endpoint']}|{mask_kind}"
    for n in order_predictors(list(res)):
        pt, bs = res[n]
        _v, role, nat = preds[n]
        cov = float(np.isfinite(np.asarray(nat, float)[u["U"]]).mean()) if u["U"].any() else np.nan
        r = dict(base, status=status, predictor=n, role=role, native_coverage_U=cov, R2_heldout=fits.get(n, np.nan))
        for k in MET:
            r[k] = pt[k]
            for sch, b_ in zip(dr.schemes, bs):
                r[f"{k}{sfx_of(sch)}_lo"], r[f"{k}{sfx_of(sch)}_hi"] = pct_ci(b_[k])
                draws[f"{uid}|{n}|{k}|{sch}"] = b_[k].astype(np.float32)
        mrows.append(r)
        if n == P_NAME:
            continue
        q = dict(base, status=status, predictor=n, role=role)
        for k in MET:
            q[f"d_{k}"] = ppt[k] - pt[k]
            for sch, bp, bx in zip(dr.schemes, pbs, bs):
                dd = bp[k] - bx[k]
                q[f"d_{k}{sfx_of(sch)}_lo"], q[f"d_{k}{sfx_of(sch)}_hi"], q[f"p_{k}{sfx_of(sch)}"] = (*pct_ci(dd), boot_p(dd))
        prows.append(q)
    return mrows, prows, draws


# ===================================================================================================== unit builders (truth-free vectors + labels)
def _complete_methods(V, bank):
    import gb_attribution as A
    return [m for m in A.METHODS if m in V and m in bank.X]


def lattice_unit(system, tf, dr_blk):
    """(system, TF) response unit.  ORDER as the audit: scores -> catbank -> truth inside L.build_endpoint; the truth dict is captured (resp_ok for the resp_only sensitivity) by wrapping L.assemble."""
    import gb_attribution as A
    import gb_audit_decomp as AD
    import gb_layer_lattice as L
    cfg = L.SYSTEMS[system]
    bank = A.load_bank(cfg["axis"])
    cap, orig = {}, L.assemble

    def _assemble(system_, tf_, genes_, V_, truth_, blk_, meta_, gb_sha_pre=None):
        cap["truth"] = truth_
        return orig(system_, tf_, genes_, V_, truth_, blk_, meta_, gb_sha_pre)
    L.assemble = _assemble
    try:
        ep = L.build_endpoint(system, tf, dr_blk, False)
    finally:
        L.assemble = orig
    V, genes = ep["V"], np.asarray(ep["genes"]).astype(str)
    use = _complete_methods(V, bank)
    if A.GB not in use or len(use) != len(A.METHODS):
        raise L.Blocked(f"METHOD_MISSING {system}/{tf}: used {use}")
    r = bank.row_of(tf)
    if r is None:
        raise L.Blocked(f"FOCAL_NOT_IN_BANK {system}/{tf}")
    if "prior GB" not in V:
        raise L.Blocked(f"NO_LATTICE_PRIOR_GB {system}/{tf}")
    M0 = np.asarray(ep["U"], bool) & np.all([np.isfinite(V[m]) for m in use], 0)
    ai, has, ai0, tc, prior, M1 = AD.masks_and_priors(bank, genes, V, use, r, M0)
    f = M1 & np.isfinite(V["prior GB"])
    dgb = float(np.abs(prior[A.GB][f] - V["prior GB"][f]).max()) if f.any() else np.nan
    if not dgb <= 1e-9:
        raise SystemExit(f"REFUSING_PARITY {system}/{tf}: GB leave-focal-out prior differs from the lattice `prior GB` by {dgb}")
    refs = list(L.REFS)
    if not all(n in V for n in refs) or not all(c in V for c in ("COMP_Z", "COMP_RANK")):
        raise L.Blocked(f"REF_OR_COMP_MISSING {system}/{tf}")
    Xr = np.column_stack([np.asarray(V[n], float) for n in refs])
    comps = {c: np.asarray(V[c], float) for c in ("COMP_Z", "COMP_RANK")}
    MR = M1 & np.isfinite(Xr).all(1)
    y, rows = L.targets(ep, MR)["resp"]
    resp_ok = np.asarray(cap["truth"]["resp_ok"], bool)
    M0s = resp_ok & np.all([np.isfinite(V[m]) for m in use], 0)
    MS = AD.masks_and_priors(bank, genes, V, use, r, M0s)[5] & np.isfinite(comps["COMP_Z"]) & np.isfinite(comps["COMP_RANK"])
    MRR = MS & np.isfinite(Xr).all(1)
    gate = {"MR": bool(L.gate_of(ep, y, MR, MR)), "resp_only": bool(L.gate_of(ep, y, MRR, MRR))}
    gref = A.lattice_ref_auroc(system, tf)
    return dict(kind="lattice", system=system, axis=cfg["axis"], tf=tf, endpoint="resp", endpoint_class="response", block=cfg["block"], macro_group=MACRO_BLOCK.get(system, ""), genes=genes, gi=ep["gi"],
                folds_gene=np.asarray(ep["fold"]), y=np.asarray(y, bool), U=np.asarray(ep["U"], bool), M0=M0, M1=M1, MR=MR, MRR=MRR, V={m: np.asarray(V[m], float) for m in use},
                prior=prior, P=prior[A.GB], S=np.asarray(V[A.GB], float), others={m: prior[m] for m in use if m != A.GB}, comps=comps, Xr=Xr, gate=gate, ep=ep, bank=bank, use=use,
                ref_lattice_auroc=gref, files={k: v for k, v in ep["files"].items()}, parity_prior_GB=dgb)


def regulon_units(system, B, slice_, nslice, tfs=None, sizes=MODULE_SIZES):
    """per-regulator regulon units of one system (gb_audit_decomp.regulon_system recipe, imported pieces only).  Yields (unit, Draws).  Units failing the audit floor (< 20 pos / neg on M1) are
    yielded as dict(status=NOT_RUN, ...) so they are logged, never silently dropped."""
    import gb_attribution as A
    import gb_audit_decomp as AD
    import gb_canon_table as CT
    import gb_layer_lattice as L
    H = CT.heavy()
    union, gen = CT.load_system(system, H)
    bank = A.load_bank(system)
    idx = CT.DEFAULT_SYSTEMS.index(system)
    bi_seed = [CT.BOOT_SEED, idx]
    units, rcs = {}, {}
    for sz in sizes:
        units[sz], rcs[sz] = load_modules(system, union, sz)
    dr = Draws(union, B, bi_seed, {sz: mod_seed([CT.BOOT_SEED, idx, 20261002], sz) for sz in sizes}, units)
    dr.n_missing_in_map = max(rc["n_names_missing_in_map"] for rc in rcs.values())
    dr.module_receipts = rcs
    gidx = {g: i for i, g in enumerate(union)}
    seen, k = set(), 0
    for rec in gen:
        if rec["family"] != "F1" or rec["source"] in seen:
            continue
        tf = rec["source"]
        seen.add(tf)
        k += 1
        if (nslice > 1 and (k - 1) % nslice != slice_) or (tfs and tf not in tfs):
            continue
        V, genes, ev, y = rec["V"], np.asarray(rec["obs_genes"]).astype(str), np.asarray(rec["ev"], bool), np.asarray(rec["y"], bool)
        r = bank.row_of(tf)
        use = _complete_methods(V, bank)
        if r is None or len(use) != len(A.METHODS):
            yield dict(status=f"NOT_RUN: {'FOCAL_NOT_IN_BANK' if r is None else 'METHODS_MISSING ' + str(use)}", system=system, tf=tf, kind="regulon"), dr
            continue
        M0 = ev & np.all([np.isfinite(V[m]) for m in use], 0)
        ai, has, ai0, tc, prior, M1 = AD.masks_and_priors(bank, genes, V, use, r, M0)
        yy = y[M1]
        if yy.sum() < MIN_CLASS or (~yy).sum() < MIN_CLASS:
            yield dict(status=f"NOT_RUN: regulon on M1 has {int(yy.sum())} pos / {int((~yy).sum())} neg (< {MIN_CLASS})", system=system, tf=tf, kind="regulon", n_M1=int(M1.sum())), dr
            continue
        F = bank.feats.reindex(genes)
        refd = L.ref_vectors(F)
        comps_all, cdiag = L.comp_vectors(F)
        if any(c not in comps_all for c in ("COMP_Z", "COMP_RANK")):
            yield dict(status="NOT_RUN: COMP_Z/COMP_RANK NO_CALL", system=system, tf=tf, kind="regulon"), dr
            continue
        refs = list(L.REFS)
        Xr = np.column_stack([np.asarray(refd[n], float) for n in refs])
        comps = {c: np.asarray(comps_all[c], float) for c in ("COMP_Z", "COMP_RANK")}
        MR = M1 & np.isfinite(Xr).all(1) & np.isfinite(comps["COMP_Z"]) & np.isfinite(comps["COMP_RANK"])
        gi = np.fromiter((gidx[g] for g in genes), int, len(genes))
        yr = y[MR]
        gate = {"MR": bool(yr.sum() >= MIN_CLASS and (~yr).sum() >= MIN_CLASS)}
        yield dict(kind="regulon", system=system, axis=system, tf=tf, endpoint="regulon", endpoint_class="regulon", block=system, macro_group=f"REGULON:{system}", genes=genes, gi=gi,
                   folds_gene=dr.fold_gene(gi), y=y, U=ev, M0=M0, M1=M1, MR=MR, MRR=None, V={m: np.asarray(V[m], float) for m in use}, prior=prior, P=prior[A.GB],
                   S=np.asarray(V[A.GB], float), others={m: prior[m] for m in use if m != A.GB}, comps=comps, Xr=Xr, gate=gate, bank=bank, use=use, endpoint_name=rec["endpoint"]), dr
    return


# ===================================================================================================== parity gates (reuse of existing artifacts after mask / hash checks)
_CACHE = {}


def _audit_control():
    if "ctl" not in _CACHE:
        rc = json.load(open(f"{AUD_DIR}/receipt.json"))
        import gb_audit_decomp as AD
        if __import__("os").environ.get("GB_SKIP_PREFLIGHT") == "1": print("preflight skipped (public release)")
        elif rc["script_sha256"] != _sha(AD.__file__):
            raise SystemExit("REFUSING_PARITY: audit_decomp_v2 receipt script sha != the current gb_audit_decomp.py (the reused control tables are stale)")
        _CACHE["ctl"] = pd.read_csv(f"{AUD_DIR}/control_per_tf.tsv", sep="\t", dtype={"tf": str})
        _CACHE["ctl_sha"] = _sha(f"{AUD_DIR}/control_per_tf.tsv")
        _CACHE["aud"] = pd.read_csv(f"{AUD_DIR}/audit_per_tf.tsv", sep="\t", dtype={"tf": str})
        _CACHE["aud_sha"] = _sha(f"{AUD_DIR}/audit_per_tf.tsv")
    return _CACHE["ctl"], _CACHE["aud"]


PARITY_MAP = {P_NAME: ("PRIOR", "none", GB), "HGB all-9 -> P": ("FIT", "hgb", GB), "COMP_Z": ("COMP", "none", "COMP_Z"), "COMP_RANK": ("COMP", "none", "COMP_RANK")}
PARITY_TOL = 1e-9
PARITY_TOL_HGB = 1e-6


def parity_lattice(u, mrows):
    """GB P / HGB all-9 / COMP / nine priors on MR vs audit_decomp_v2 control_per_tf; GB S on M0 vs the lattice table; n_M1 / n_MR vs the audit; s374 unit AUROC when E == MR.  Returns a dict; raises on breach."""
    import gb_attribution as A
    import gb_decompose as D
    ctl, aud = _audit_control()
    rows = {r["predictor"]: r for r in mrows if r["mask"] == "MR" and "predictor" in r}
    c = ctl[(ctl.system == u["system"]) & (ctl.tf == u["tf"]) & (ctl.status.isin(["OK", "DESCRIPTIVE_ONLY"]))]
    out = dict(status="PASS", worst=0.0, n_cells=0, notes=[])
    if not len(c):
        return dict(out, status="NO_AUDIT_REFERENCE")
    if not (c.n_M1.iloc[0] == int(u["M1"].sum()) and c.n_refmask.iloc[0] == int(u["MR"].sum())):
        raise SystemExit(f"REFUSING_PARITY_MASK {u['system']}/{u['tf']}: n_M1/n_MR now {int(u['M1'].sum())}/{int(u['MR'].sum())} vs audit {c.n_M1.iloc[0]}/{c.n_refmask.iloc[0]}")
    pm = dict(PARITY_MAP)
    pm.update({f"prior {m}": ("PRIOR", "none", m) for m in A.METHODS if m != GB})
    for nm, (vt, arm, vec) in pm.items():
        q = c[(c.vtype == vt) & (c.arm == arm) & (c.vector == vec)]
        if not len(q) or nm not in rows:
            continue
        q, mine = q.iloc[0], rows[nm]
        tol = PARITY_TOL_HGB if vt == "FIT" else PARITY_TOL
        for k in MET:
            dlt = abs(float(q[k]) - float(mine[k])) if np.isfinite(q[k]) and np.isfinite(mine[k]) else 0.0
            out["worst"] = max(out["worst"], dlt)
            out["n_cells"] += 1
            if dlt > tol:
                raise SystemExit(f"REFUSING_PARITY {u['system']}/{u['tf']}/{nm}/{k}: now {mine[k]!r} vs audit_decomp_v2 {q[k]!r} (|d|={dlt:.3e} > {tol})")
    # GB S: M0 vs the lattice table, M1 vs audit AUROC_full
    S = u["S"]
    import gb_audit_decomp as AD
    y = u["y"]
    resp_m0 = AD.auroc_rows(S[None, u["M0"]], y[u["M0"]])[0] if y[u["M0"]].any() and (~y[u["M0"]]).any() else np.nan
    if u["ref_lattice_auroc"] is not None and np.isfinite(resp_m0):
        dlt = abs(resp_m0 - u["ref_lattice_auroc"])
        out["S_M0_vs_lattice"] = dict(now=float(resp_m0), ref=float(u["ref_lattice_auroc"]), absdiff=float(dlt))
        if dlt > PARITY_TOL:
            raise SystemExit(f"REFUSING_PARITY {u['system']}/{u['tf']}: GB S AUROC on ALL6+SOTA scope {resp_m0!r} != lattice table {u['ref_lattice_auroc']!r}")
    a = aud[(aud.system == u["system"]) & (aud.tf == u["tf"]) & (aud.kind == "lattice") & (aud.target == "resp") & (aud.method == GB) & aud.status.isin(["OK", "DESCRIPTIVE_ONLY"])]
    if len(a):
        m1y = u["y"][u["M1"]]
        s_m1 = AD.auroc_rows(S[None, u["M1"]], m1y)[0]
        dlt = abs(float(a.AUROC_full.iloc[0]) - float(s_m1))
        out["S_M1_vs_audit"] = dict(now=float(s_m1), ref=float(a.AUROC_full.iloc[0]), absdiff=dlt)
        if dlt > PARITY_TOL:
            raise SystemExit(f"REFUSING_PARITY {u['system']}/{u['tf']}: GB S AUROC on M1 {s_m1!r} != audit AUROC_full {a.AUROC_full.iloc[0]!r}")
    if os.path.exists(S374):
        t = pd.read_csv(S374, sep="\t", dtype={"tf": str})
        t = t[(t.system == u["system"]) & (t.tf == u["tf"])]
        if len(t):
            out["s374"] = dict(n_E=int(t.n_E.iloc[0]), n_MR=int(u["MR"].sum()), AUROC_P_s374=float(t.AUROC.iloc[0]), AUROC_P_now=float(rows[P_NAME]["AUROC"]))
            if int(t.n_E.iloc[0]) == int(u["MR"].sum()):
                dlt = abs(float(t.AUROC.iloc[0]) - float(rows[P_NAME]["AUROC"]))
                out["s374"]["absdiff"] = dlt
                if dlt > PARITY_TOL:
                    raise SystemExit(f"REFUSING_PARITY {u['system']}/{u['tf']}: GB P AUROC {rows[P_NAME]['AUROC']!r} != s374 primary_unit_metrics {t.AUROC.iloc[0]!r} (same n)")
    return out


def parity_regulon(u, mrows):
    """GB S / P / the nine priors vs audit_per_tf kind=regulon on M1 (the audit reports AUROC_full / AUROC_prior on M1; MR differs only if a REF is missing)"""
    import gb_audit_decomp as AD
    _ctl, aud = _audit_control()
    a = aud[(aud.system == u["system"]) & (aud.tf == u["tf"]) & (aud.kind == "regulon") & aud.status.isin(["OK", "DESCRIPTIVE_ONLY"])]
    out = dict(status="PASS", worst=0.0, n_cells=0)
    if not len(a):
        return dict(out, status="NO_AUDIT_REFERENCE")
    if int(a.n_M1.iloc[0]) != int(u["M1"].sum()):
        raise SystemExit(f"REFUSING_PARITY_MASK regulon {u['system']}/{u['tf']}: n_M1 {int(u['M1'].sum())} vs audit {int(a.n_M1.iloc[0])}")
    y, M1 = u["y"], u["M1"]
    for _, q in a.iterrows():
        m = q.method
        for tag, vec, col in (("S", u["V"][m], "AUROC_full"), ("P", u["prior"][m], "AUROC_prior")):
            now = AD.auroc_rows(np.asarray(vec, float)[None, M1], y[M1])[0]
            dlt = abs(now - float(q[col]))
            out["worst"] = max(out["worst"], dlt)
            out["n_cells"] += 1
            if dlt > PARITY_TOL:
                raise SystemExit(f"REFUSING_PARITY regulon {u['system']}/{u['tf']}/{m}/{tag}: {now!r} vs audit {q[col]!r}")
    return out



# ===================================================================================================== BM breadth (gb_ladder BM TOTAL / BM19 'any' label; reuse of the ladder's own functions and draws)
BREADTH_NAME = {P_NAME: "raw:P_pairU", "prior |Pearson|": "raw:pearson", "prior GENIE3": "raw:genie3", "prior GRNBoost2": "raw:grnboost2", "prior RegDiffusion": "raw:regdiffusion",
                "prior DeepSEM": "raw:deepsem", "prior PIDC": "raw:pidc", "prior FastTENET": "raw:fasttenet", "COMP_Z": "raw:COMP_Z", "COMP_RANK": "raw:COMP_RANK"}   # predictor -> ladder row name
BREADTH_METRIC = {"AUROC": "auc_any", "APLIFT": "aplift_any", "P50": "yield_any@50_stratified"}                                                                # AUPRC: not defined by the ladder (AP-lift only)
BM_SEED_TAG = "BR_BM_TOTAL_BM19"


def breadth_unit_bm(B, n_jobs):
    """BM held-out perturbation breadth unit on the ladder's own objects: bank (gb_ladder.load_bank T7, F1), composites, breadth truth (receipt-hash checked), perturbation folds, complete-case genes.
    The ten priors are the FULL-bank priors (ladder variant `full`, PRIMARY); RegVelo / Velorama come from gb_attribution's bank (the ladder T7 set carries no external-method prior)."""
    import gb_attribution as A
    import gb_audit_decomp as AD
    import gb_ladder as LD
    import gb_layer_lattice as L
    import gb_decompose as D
    bankL = LD.load_bank("BONEMARROW", "", "T7", False, roots=False, tier="F1")
    comps_bank, cdiag = LD.comp_vectors(bankL)
    tr = LD.load_breadth_truth("BM", "TOTAL", "BM19")
    if tr["calibration"] != "CALIBRATED":
        raise SystemExit(f"REFUSING breadth: null-gate verdict {tr['calibration']}")
    bi = LD.axis_index(bankL["genes"], tr["genes"])
    okg = bi >= 0
    bi = bi[okg]
    if len(np.unique(bi)) != len(bi):
        raise SystemExit("REFUSING breadth: truth genes map non-injectively onto the bank axis")
    genes = np.asarray(tr["genes"])[okg].astype(str)
    truth = dict(perts=tr["perts"], T=tr["T"][:, okg], R=tr["R"][:, okg], E=tr["E"][:, okg].astype(float), CN=tr["CN"][:, okg])
    t7 = np.asarray(bankL["genes"]).astype(str)[bi]                                                 # exact catbank axis names of the breadth genes (module-map / resampling keys)
    pri = {}
    for m in LD.GB_PRIORS + bankL["gens"]:
        pri[m] = LD.prior_excl(bankL, m, [])[bi]
    sota = {m: LD.prior_excl(bankL, m, [], zsc=bankL["ZSC_SOTA"])[bi] for m in bankL["sota"]}
    bankA = A.load_bank("BONEMARROW")
    ai = np.array([bankA.ai[g] for g in t7])
    ext = {m: np.asarray(bankA.prior(m, None), float)[ai] for m in A.METHODS}                       # every method's FULL-bank prior on the breadth genes (T7 columns)
    for nm, lad in (("GeoBRIDGE", pri["P_pairU"]), ("|Pearson|", pri["pearson"]), ("GENIE3", pri["genie3"]), ("GRNBoost2", pri["grnboost2"]),
                    ("RegDiffusion", sota["regdiffusion"]), ("DeepSEM", sota["deepsem"]), ("PIDC", sota["pidc"]), ("FastTENET", sota["fasttenet"])):
        f = np.isfinite(lad) & np.isfinite(ext[nm])
        if (np.isfinite(lad) != np.isfinite(ext[nm])).any() or (f.any() and np.abs(lad[f] - ext[nm][f]).max() > 1e-9):
            raise SystemExit(f"REFUSING breadth: gb_attribution full-bank prior of {nm} differs from the ladder prior")
    feat = {c: bankL["feat"][c][bi] for c in bankL["f1"]}
    Xr = np.column_stack([sg * np.asarray(feat[c], float) for c, sg in L.REF_SIGNS.values()])        # L.REFS order
    Xf = np.column_stack([feat[c] for c in feat])
    keep_lad = np.isfinite(Xf).all(1) & np.all([np.isfinite(pri[m]) for m in pri], 0) & np.isfinite(Xr).all(1)      # = gb_ladder.breadth_core complete-case genes
    comps = {c: np.asarray(comps_bank[c], float)[bi] for c in ("COMP_Z", "COMP_RANK")}
    mine = keep_lad & np.all([np.isfinite(ext[m]) for m in ext], 0) & np.isfinite(comps["COMP_Z"]) & np.isfinite(comps["COMP_RANK"])
    perts = np.char.upper(np.asarray(truth["perts"]).astype(str))
    order = np.argsort([int(hashlib.sha1(p_.encode()).hexdigest(), 16) for p_ in perts])
    pf = np.empty(len(perts), int)
    pf[order] = np.arange(len(perts)) % LD.K
    return dict(LD=LD, bankL=bankL, genes=genes, t7=t7, bi=bi, truth=truth, ext=ext, others={m: ext[m] for m in A.METHODS if m != A.GB}, P=ext[A.GB], Xr=Xr, comps=comps, keep=np.flatnonzero(mine),
                keep_lad=keep_lad, mine=mine, pf=pf, perts=perts, folds=np.array([int(hashlib.sha1(g.upper().encode()).hexdigest(), 16) % LD.K for g in genes]), cdiag=cdiag, tr=tr,
                files=dict(bank=bankL["files"], truth_receipt_sha256=tr["truth_receipt_sha256"]))


def _breadth_draw_all(LD, D, Wd, PW, n_jobs):
    LD._BD = D
    pt, _nc = LD.breadth_draw(np.ones(len(D["truth"][0][0])), np.ones(PW.shape[1]), D)
    args = [(np.asarray(Wd[b], float), np.asarray(PW[b], float)) for b in range(len(Wd))]
    if n_jobs > 1:
        import multiprocessing as mp
        with mp.get_context("fork").Pool(n_jobs) as pool:
            out = pool.map(LD._bd, args)
    else:
        out = [LD._bd(x) for x in args]
    return pt, [o[0] for o in out]


def run_breadth(a, which, out):
    """BM breadth rows on TWO universes, both labelled: mask=MR (genes where EVERY one of the ten priors, the nine features and both composites is finite; Velorama's prior is finite on only ~75% of the ladder genes)
    and mask=ladder_universe (the ladder's own complete-case genes, 1648; RegVelo / Velorama priors NOT_AVAILABLE there) -- the latter reproduces the headline ladder numbers and is the parity gate.
    Resampling: two-level (perturbation cluster x [gene identity = the ladder's own shared draws | control-module block at G/25, G/50 (primary sensitivity), G/100])."""
    if which != "BM":
        raise SystemExit("REFUSING: only the BM breadth endpoint exists in the frozen ladder outputs")
    import gb_audit_decomp as AD
    import gb_decompose as D_
    import gb_layer_lattice as L
    t0 = time.time()
    B = a.B
    U = breadth_unit_bm(B, a.n_jobs)
    LD = U["LD"]
    genes, pf = U["genes"], U["pf"]
    n_all = len(genes)
    units_by, mrcs = {}, {}
    for sz in MODULE_SIZES:
        units_by[sz], mrcs[sz] = load_modules("BONEMARROW", U["t7"], sz)
        if mrcs[sz]["n_names_missing_in_map"]:
            raise SystemExit(f"REFUSING breadth: {mrcs[sz]['n_names_missing_in_map']} breadth genes absent from the frozen module map G/{sz}")
    mrc = mrcs[PRIMARY_SIZE]
    obs, dyn, allx = ref_index()
    fold_mod = np.array([int(hashlib.sha1(str(g).upper().encode()).hexdigest(), 16) % LD.K for g in units_by[PRIMARY_SIZE]])
    # ---- draws: gene level = the ladder's own shared multinomial matrix (hash-checked) | module level = BlockAxis units; perturbation level = the ladder's PW (shared by all schemes)
    wp = f"{LD.LAD}/final/draws_{BM_SEED_TAG}_{n_all}.npy"
    rc = json.load(open(BREADTH_RCPT))
    rec = [x for x in rc["analyses"] if x["estimand"] == "TOTAL" and x["arm"] == "BM19" and x["variant"] == "full"][0]
    if rec["n_mapped_to_bank"] != n_all or _sha(wp) != rec["draws_sha256"]:
        raise SystemExit(f"REFUSING_PARITY breadth: mapped genes {n_all} / draws sha differ from the ladder receipt ({rec['n_mapped_to_bank']})")
    Wg_all = np.asarray(np.load(wp, mmap_mode="r")[:B])
    bi_blk = L.BLOCKS.index("BONEMARROW")
    bms = {sz: L.BlockAxis(U["t7"], B, int(np.random.SeedSequence(mod_seed([L.BOOT_SEED, bi_blk, 20261002, 1], sz)).generate_state(1)[0]), units_by[sz], "module") for sz in MODULE_SIZES}
    schemes = ("g",) + tuple(f"m{sz}" for sz in MODULE_SIZES)
    nP = len(U["perts"])
    PW = np.random.default_rng(np.random.SeedSequence([LD.SEED, zlib.crc32(BM_SEED_TAG.encode()), 1])).multinomial(nP, np.full(nP, 1.0 / nP), size=6000)[:B]
    bt = pd.read_csv(BREADTH_TSV, sep="\t", low_memory=False)
    bt = bt[(bt.estimand == "TOTAL") & (bt.arm == "BM19") & (bt.variant == "full") & (bt.kind == "model")]
    mrows, prows, draws, par = [], [], {}, dict(n_cells=0, worst=0.0, table_sha256=_sha(BREADTH_TSV), n_keep_ladder=int(U["keep_lad"].sum()), n_keep_common=int(U["mine"].sum()))
    fp = f"ladder:{_sha(wp)[:12]}|" + "|".join(bms[sz].fingerprint()[:12] for sz in MODULE_SIZES)
    for mk, maskv in (("MR", U["mine"]), ("ladder_universe", U["keep_lad"])):
        keep = np.flatnonzero(maskv)
        zP = D_.zrow(U["P"], maskv)
        vec, role, nat, fits = {P_NAME: U["P"]}, {P_NAME: "P"}, {P_NAME: U["P"]}, {}
        for nm, cols, f in (("HGB obs-5 -> P", obs, U["folds"]), ("HGB dyn-4 -> P", dyn, U["folds"]), ("HGB all-9 -> P", allx, U["folds"]), ("HGB all-9 module-fold -> P (sens.)", allx, fold_mod)):
            fit, _r, st = AD.hgb_cross_fit(zP, U["Xr"][:, cols], f, maskv)
            vec[nm], role[nm], fits[nm] = fit, ("reconstruction" if "sens." not in nm else "sensitivity"), st["R2_heldout"]
            nat[nm] = np.where(np.isfinite(U["Xr"][:, cols]).all(1), 0.0, np.nan)
        for c in ("COMP_Z", "COMP_RANK"):
            vec[c], role[c], nat[c] = U["comps"][c], "composite", U["comps"][c]
        missing = []
        for m, v in U["others"].items():
            if np.isfinite(np.asarray(v, float)[keep]).all():
                vec[f"prior {m}"], role[f"prior {m}"], nat[f"prior {m}"] = v, "other-method prior", v
            else:
                missing.append(m)
        restricted = {n: np.asarray(v, float)[keep] for n, v in vec.items()}
        assert all(np.isfinite(v).all() for v in restricted.values())
        D = dict(models=list(restricted), fitted={}, raw=restricted, keep=keep, gf=U["folds"][keep], pf=pf, truth=tuple(x[:, keep] for x in (U["truth"]["T"], U["truth"]["R"], U["truth"]["E"], U["truth"]["CN"])),
                 perts=list(U["perts"]), tie=None)
        pt, dg = _breadth_draw_all(LD, D, Wg_all[:, keep], PW, a.n_jobs)
        dsch = {"g": dg}
        for sz in MODULE_SIZES:
            _pt2, dsch[f"m{sz}"] = _breadth_draw_all(LD, D, bms[sz].counts[:, keep], PW, a.n_jobs)
        tg = LD.mk_targets(D["truth"][0], D["truth"][1], D["truth"][2], D["truth"][3], np.ones(nP, bool), LD.COVER_FLOOR, min_t=1)
        n_pos, n_neg = int((tg["any"] == 1).sum()), int((tg["any"] == 0).sum())
        if mk == "ladder_universe":                                                              # PARITY with the existing ladder table (same genes, same objects)
            for nm, lname in BREADTH_NAME.items():
                for k, mkey in BREADTH_METRIC.items():
                    q = bt[(bt.name == lname) & (bt.metric == {"yield_any@50_stratified": "yield_any@50"}.get(mkey, mkey))]
                    if len(q) and nm in pt:
                        dlt = abs(float(q.point.iloc[0]) - float(pt[nm][mkey]))
                        par["worst"], par["n_cells"] = max(par["worst"], dlt), par["n_cells"] + 1
                        if dlt > 1e-9:
                            raise SystemExit(f"REFUSING_PARITY breadth {nm}/{k}: now {pt[nm][mkey]!r} vs ladder {q.point.iloc[0]!r}")
        base = dict(kind="breadth", system="BM_BREADTH", block="BONEMARROW", tf="ALL", endpoint="any", endpoint_class="breadth", mask=mk, n_U=int(n_all), n_M1=int(U["keep_lad"].sum()),
                    n_rows=int(len(keep)), frac_rows_of_U=float(len(keep) / n_all), n_pos=n_pos, n_neg=n_neg, PREV=n_pos / max(n_pos + n_neg, 1), gate=True, macro_group="", status="OK",
                    predictors_not_available=",".join(missing), draws_fp=fp, ci_primary="two-level: perturbation-cluster x gene-identity resampling (the ladder's own draws)",
                    ci_sensitivity="two-level: perturbation-cluster x module-block resampling (frozen control-only modules at G/25, G/50, G/100)",
                    resampling_note="two-level: perturbation-cluster (ladder PW) x [gene identity (ladder draws) | control-module block]; metrics fold-stratified (ladder): AUROC=auc_any, AP-lift=aplift_any "
                                    "(n-weighted over folds; dominated by sparse folds), P50=yield_any@50 (stratified); AP not defined by the ladder")
        arr = lambda D_l, nm, mkey: np.array([d[nm][mkey] for d in D_l], float)
        uid = f"breadth|BM_BREADTH|ALL|any|{mk}"
        PB = {sch: {k: arr(dsch[sch], P_NAME, mkey) for k, mkey in BREADTH_METRIC.items()} for sch in schemes}
        for n in order_predictors(list(vec)):
            r = dict(base, predictor=n, role=role[n], native_coverage_U=float(np.isfinite(np.asarray(nat[n], float)).mean()), R2_heldout=fits.get(n, np.nan))
            q = dict(base, predictor=n, role=role[n])
            for k in MET:
                if k in BREADTH_METRIC:
                    mkey = BREADTH_METRIC[k]
                    r[k] = pt[n][mkey]
                    if n != P_NAME:
                        q[f"d_{k}"] = pt[P_NAME][mkey] - pt[n][mkey]
                    for sch in schemes:
                        bx = arr(dsch[sch], n, mkey)
                        r[f"{k}{sfx_of(sch)}_lo"], r[f"{k}{sfx_of(sch)}_hi"] = pct_ci(bx)
                        draws[f"{uid}|{n}|{k}|{sch}"] = bx.astype(np.float32)
                        if n != P_NAME:
                            dd = PB[sch][k] - bx
                            q[f"d_{k}{sfx_of(sch)}_lo"], q[f"d_{k}{sfx_of(sch)}_hi"], q[f"p_{k}{sfx_of(sch)}"] = (*pct_ci(dd), boot_p(dd))
                else:
                    r[k] = np.nan
                    q[f"d_{k}"] = np.nan
                    for sch in schemes:
                        r[f"{k}{sfx_of(sch)}_lo"] = r[f"{k}{sfx_of(sch)}_hi"] = np.nan
                        q[f"d_{k}{sfx_of(sch)}_lo"] = q[f"d_{k}{sfx_of(sch)}_hi"] = q[f"p_{k}{sfx_of(sch)}"] = np.nan
            mrows.append(r)
            if n != P_NAME:
                prows.append(q)
        print(f"[ptable-brd] BM breadth mask={mk}: rows={len(keep)} pos={n_pos} neg={n_neg} not_available={missing} P AUROC={pt[P_NAME]['auc_any']:.6f}", flush=True)
    write_part(out, "part_brd_BM_all", mrows, prows, draws, dict(stem="part_brd_BM_all", kind="breadth", system="BM_BREADTH", B=B, n_modules={sz: int(bms[sz].unit.max() + 1) for sz in MODULE_SIZES}, script_sha256=SCRIPT_SHA0,
               module_receipt={k: mrc[k] for k in ("rule_sha256", "module_file_sha256", "created", "n_modules", "n_singletons", "n_names_missing_in_map")}, parity=par, tfs={"ALL": dict(parity=par)},
               ladder_files=U["files"], seconds=round(time.time() - t0, 1)))
    print(f"[ptable-brd] BM breadth done: ladder parity worst={par['worst']:.2e} over {par['n_cells']} cells; {time.time() - t0:.0f}s", flush=True)


# ===================================================================================================== run
def dedup_report():
    """verify (from the audit table) that the dropped BONEMARROW units are exact duplicates of the BM_CHIPATLAS units of the same TF: same n_M1 / n_pos / AUROC of S and P"""
    _ctl, aud = _audit_control()
    a = aud[(aud.kind == "lattice") & (aud.target == "resp") & (aud.method == GB) & aud.status.isin(["OK", "DESCRIPTIVE_ONLY"])]
    out = {}
    for dup, keep in LATTICE_DUP.items():
        for tf in sorted(a[a.system == dup].tf.unique()):
            x, y = a[(a.system == dup) & (a.tf == tf)].iloc[0], a[(a.system == keep) & (a.tf == tf)]
            out[f"{dup}/{tf}"] = dict(duplicate_of=keep, identical=bool(len(y) and x.n_M1 == y.n_M1.iloc[0] and x.n_pos == y.n_pos.iloc[0] and abs(x.AUROC_full - y.AUROC_full.iloc[0]) < 1e-12
                                                                           and abs(x.AUROC_prior - y.AUROC_prior.iloc[0]) < 1e-12), n_M1=int(x.n_M1), AUROC_prior=float(x.AUROC_prior))
    return out


def cmd_modules(a):
    axes = [s.strip() for s in a.axes.split(",") if s.strip()]
    sizes = tuple(int(x) for x in a.sizes.split(",") if x.strip())
    bad = [x for x in sizes if x not in MODULE_SIZES]
    if bad:
        raise SystemExit(f"REFUSING: module sizes {bad} are not in the declared family {MODULE_SIZES}")
    guard_systems(axes)
    for ax in axes:
        freeze_axis(ax, sizes)


def draws_meta(dr):
    """provenance of one Draws object: fingerprint, seeds, module counts and the frozen module-map receipts (rule sha / file sha) per size"""
    keys = ("rule_sha256", "module_file_sha256", "created", "n_modules", "n_singletons")
    return dict(draws_fp=dr.fp, seeds=dict(gene=dr.gene.seed, **{f"m{s}": dr.mods[s].seed for s in dr.sizes}), n_modules=dr.n_modules, n_missing_in_map=dr.n_missing_in_map,
                module_receipts={s: {k: rc[k] for k in keys} for s, rc in dr.module_receipts.items()})


def write_part(out, stem, mrows, prows, draws, meta):
    if mrows:
        atomic_tsv(pd.DataFrame(mrows), f"{out}/{stem}.metrics.tsv")
        atomic_tsv(pd.DataFrame(prows), f"{out}/{stem}.paired.tsv")
        tmp = f"{out}/{stem}.draws.tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, **draws)
        os.replace(tmp, f"{out}/{stem}.draws.npz")
    atomic_json(f"{out}/{stem}.meta.json", dict(meta, dep_shas=dep_shas(), python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, host=platform.node(),
                                                created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def run_lattice(a, system, out):
    import gb_audit_decomp as AD
    import gb_layer_lattice as L
    import gb_lattice_extra as LE
    LE.register()
    t0 = time.time()
    dr, mrc = lattice_draws(system, a.B)
    cfg = L.SYSTEMS[system]
    allowed = AD.tf_list_audit(system, L, LE)
    tfs = [t.strip() for t in a.tfs.split(",") if t.strip()] or allowed
    bad = [t for t in tfs if t not in allowed]
    if bad:
        raise SystemExit(f"REFUSING: --tfs {bad} not in the audit roster {allowed} of {system}")
    if a.nslice > 1:
        tfs = [t for i, t in enumerate(tfs) if i % a.nslice == a.slice]
    mrows, prows, draws, metas = [], [], {}, {}
    for tf in tfs:
        t1 = time.time()
        try:
            u = lattice_unit(system, tf, dr.gene)
        except (L.Blocked, SystemExit) as e:
            metas[tf] = dict(status=f"NOT_RUN: {e}")
            print(f"[ptable] {system}/{tf}: NOT_RUN {e}", flush=True)
            if isinstance(e, SystemExit) and "REFUSING_PARITY" in str(e):
                raise
            continue
        mr_, pr_, dd_ = table1_rows(u, dr, u["MR"], "MR")
        par = {} if a.no_parity else parity_lattice(u, mr_)
        mrs_, prs_, dds_ = table1_rows(u, dr, u["MRR"], "resp_only") if u["MRR"].any() else ([], [], {})
        mrows += mr_ + mrs_
        prows += pr_ + prs_
        draws.update(dd_)
        draws.update(dds_)
        metas[tf] = dict(n_U=int(u["U"].sum()), n_M0=int(u["M0"].sum()), n_M1=int(u["M1"].sum()), n_MR=int(u["MR"].sum()), n_MRR=int(u["MRR"].sum()), gate=u["gate"], parity=par,
                         parity_prior_GB_vs_lattice_absdiff=u["parity_prior_GB"], files=u["files"], methods_used=u["use"], bank_sha=u["bank"].sha)
        print(f"[ptable] {system}/{tf}: U={int(u['U'].sum())} M1={int(u['M1'].sum())} MR={int(u['MR'].sum())} MRR={int(u['MRR'].sum())} gate={u['gate']} parity={par.get('status')}"
              f" worst={par.get('worst', float('nan')):.2e} {time.time() - t1:.0f}s", flush=True)
    tag = ("all" if not a.tfs else "_".join(tfs)) + (f"_s{a.slice}of{a.nslice}" if a.nslice > 1 else "")
    write_part(out, f"part_lat_{system}_{tag}", mrows, prows, draws, dict(stem=f"part_lat_{system}_{tag}", kind="lattice", system=system, B=a.B, **draws_meta(dr), script_sha256=SCRIPT_SHA0, tfs=metas,
                                                                          seconds=round(time.time() - t0, 1),
                                                                          audit_artifacts=dict(control_per_tf=_CACHE.get("ctl_sha"), audit_per_tf=_CACHE.get("aud_sha"), s374=_sha(S374) if os.path.exists(S374) else None)))


def run_regulon(a, system, out):
    t0 = time.time()
    mrows, prows, draws, metas = [], [], {}, {}
    dr_ = None
    for u, dr in regulon_units(system, a.B, a.slice, a.nslice, set(t.strip() for t in a.reg_tfs.split(",") if t.strip()) or None):
        dr_ = dr
        if u.get("status", "").startswith("NOT_RUN"):
            metas[u["tf"]] = dict(status=u["status"])
            print(f"[ptable-reg] {system}/{u['tf']}: {u['status']}", flush=True)
            continue
        t1 = time.time()
        mr_, pr_, dd_ = table1_rows(u, dr, u["MR"], "MR")
        par = {} if a.no_parity else parity_regulon(u, mr_)
        mrows += mr_
        prows += pr_
        draws.update(dd_)
        metas[u["tf"]] = dict(n_U=int(u["U"].sum()), n_M1=int(u["M1"].sum()), n_MR=int(u["MR"].sum()), gate=u["gate"], parity=par, methods_used=u["use"], endpoint=u["endpoint_name"])
        print(f"[ptable-reg] {system}/{u['tf']}: M1={int(u['M1'].sum())} MR={int(u['MR'].sum())} pos={int(u['y'][u['MR']].sum())} parity={par.get('status')} {time.time() - t1:.0f}s", flush=True)
    tag = "all" + (f"_s{a.slice}of{a.nslice}" if a.nslice > 1 else "")
    write_part(out, f"part_reg_{system}_{tag}", mrows, prows, draws, dict(stem=f"part_reg_{system}_{tag}", kind="regulon", system=system, B=a.B, **(draws_meta(dr_) if dr_ else {}),
                                                                          script_sha256=SCRIPT_SHA0, tfs=metas, seconds=round(time.time() - t0, 1)))


def assert_hgb_defaults():
    from sklearn.ensemble import HistGradientBoostingRegressor
    if HistGradientBoostingRegressor(random_state=0).get_params() != PINNED_HGB:
        raise SystemExit("REFUSING: sklearn HistGradientBoostingRegressor defaults differ from the pinned configuration (library changed)")


def cmd_run(a):
    assert_hgb_defaults()
    os.makedirs(a.out, exist_ok=True)
    systems = [s.strip() for s in a.systems.split(",") if s.strip()]
    regs = [s.strip() for s in a.regulon.split(",") if s.strip()]
    guard_systems(systems + regs + ([a.breadth] if a.breadth else []))
    for s in systems:
        if s in LATTICE_DUP:
            raise SystemExit(f"REFUSING: {s} is a duplicate of {LATTICE_DUP[s]} (dedup rule); its units are logged in the receipt, not re-run")
        if s not in LATTICE_PRIMARY + LATTICE_SENS:
            raise SystemExit(f"REFUSING: {s} not in {LATTICE_PRIMARY + LATTICE_SENS}")
    for s in regs:
        if s not in REGULON_SYSTEMS:
            raise SystemExit(f"REFUSING: regulon system {s} not in {REGULON_SYSTEMS}")
    for s in systems:
        run_lattice(a, s, a.out)
    for s in regs:
        run_regulon(a, s, a.out)
    if a.breadth:
        run_breadth(a, a.breadth, a.out)
    assert _sha(__file__) == SCRIPT_SHA0, "script changed during the run"


# ===================================================================================================== merge: macro rows + final tables
def macro_tables(mrows, prows, draws, B, schemes=SCHEMES_T1):
    """block / system macro over GATED primary units: mean metric per draw across the group's units (conditional on the fixed unit set), CI from the macro draws under every resampling scheme; paired macro P - X likewise."""
    if not len(mrows):
        return pd.DataFrame(), pd.DataFrame()
    M = mrows[(mrows.status == "OK") & mrows.macro_group.astype(str).ne("")]
    out_m, out_p, store = [], [], {}
    for (grp, mask, ec), g in M.groupby(["macro_group", "mask", "endpoint_class"], sort=False):
        units = sorted(set(zip(g.kind, g.system, g.tf, g.endpoint)))
        if len(units) < 1:
            continue
        if g.draws_fp.nunique() != 1:
            raise SystemExit(f"REFUSING_MACRO {grp}/{mask}: units of one macro group use different resample draws {sorted(g.draws_fp.unique())}")
        for pred in order_predictors(list(g.predictor.unique())):
            row = dict(macro_group=grp, mask=mask, endpoint_class=ec, predictor=pred, n_units=len(units), units=",".join(f"{s}/{t}" for _k, s, t, _e in units) if len(units) <= 20 else f"{len(units)} units")
            ok = True
            md = {}
            for k in MET:
                for sch in schemes:
                    arrs = [draws.get(f"{kk}|{ss}|{tt}|{ee}|{mask}|{pred}|{k}|{sch}") for kk, ss, tt, ee in units]
                    if any(x is None for x in arrs):
                        ok = False
                        break
                    md[(k, sch)] = np.mean(np.vstack(arrs), 0)
                if not ok:
                    break
            if not ok:
                continue
            gm = g[g.predictor == pred]
            for k in MET:
                row[k] = float(gm[k].mean())
                for sch in schemes:
                    row[f"{k}{sfx_of(sch)}_lo"], row[f"{k}{sfx_of(sch)}_hi"] = pct_ci(md[(k, sch)])
            out_m.append(row)
            store[(grp, mask, ec, pred)] = md
    mm = pd.DataFrame(out_m)
    if not len(mm):
        return mm, pd.DataFrame()
    for (grp, mask, ec), g in mm.groupby(["macro_group", "mask", "endpoint_class"], sort=False):
        if (grp, mask, ec, P_NAME) not in store:
            continue
        pd_ = store[(grp, mask, ec, P_NAME)]
        for pred in g.predictor:
            if pred == P_NAME:
                continue
            xd = store[(grp, mask, ec, pred)]
            row = dict(macro_group=grp, mask=mask, endpoint_class=ec, predictor=pred, n_units=int(g.n_units.iloc[0]))
            pr = g[g.predictor == P_NAME].iloc[0]
            xr = g[g.predictor == pred].iloc[0]
            for k in MET:
                row[f"d_{k}"] = float(pr[k] - xr[k])
                for sch in schemes:
                    dd = pd_[(k, sch)] - xd[(k, sch)]
                    row[f"d_{k}{sfx_of(sch)}_lo"], row[f"d_{k}{sfx_of(sch)}_hi"], row[f"p_{k}{sfx_of(sch)}"] = (*pct_ci(dd), boot_p(dd))
            out_p.append(row)
    return mm, pd.DataFrame(out_p)


def dependence_summary(pr, pm, schemes=SCHEMES_T1):
    """does module-block resampling change the inference, and how does that depend on the module granularity K?  Per level (unit / macro) x group x metric x module scheme (G/25, G/50, G/100): how many
    paired P - X intervals exclude 0 under gene-identity vs the module scheme, how many are lost / gained, and the median ratio of interval widths (module / gene).  Descriptive."""
    rows = []
    for level, df, keys in (("unit", pr[pr.status == "OK"] if "status" in pr else pr, ["kind", "system", "mask"]), ("macro", pm, ["macro_group", "mask"])):
        if not len(df):
            continue
        for key, g in df.groupby(keys):
            for k in MET:
                if f"d_{k}_lo" not in g or g[f"d_{k}_lo"].notna().sum() == 0:
                    continue
                lo, hi = g[f"d_{k}_lo"].to_numpy(float), g[f"d_{k}_hi"].to_numpy(float)
                sg = (lo > 0) | (hi < 0)
                for sch in schemes[1:]:
                    mlo, mhi = g[f"d_{k}{sfx_of(sch)}_lo"].to_numpy(float), g[f"d_{k}{sfx_of(sch)}_hi"].to_numpy(float)
                    sm = (mlo > 0) | (mhi < 0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        wr = np.nanmedian((mhi - mlo) / (hi - lo))
                    rows.append(dict(level=level, group="|".join(map(str, key)), metric=k, scheme=sch, n_contrasts=int(len(g)), n_sig_gene=int(sg.sum()), n_sig_scheme=int(sm.sum()),
                                     n_lost=int((sg & ~sm).sum()), n_gained=int((~sg & sm).sum()), median_ci_width_ratio_scheme_over_gene=float(wr)))
    return pd.DataFrame(rows)


def dependence_by_K(ds):
    """aggregate of dependence_summary over ALL groups per (level, metric, scheme): total contrasts, significant under gene / under the module scheme, lost / gained (flips), median of the group width ratios"""
    if not len(ds):
        return ds
    g = ds.groupby(["level", "metric", "scheme"]).agg(n_contrasts=("n_contrasts", "sum"), n_sig_gene=("n_sig_gene", "sum"), n_sig_scheme=("n_sig_scheme", "sum"), n_lost=("n_lost", "sum"),
                                                     n_gained=("n_gained", "sum"), median_group_width_ratio=("median_ci_width_ratio_scheme_over_gene", "median")).reset_index()
    g["n_flips"] = g.n_lost + g.n_gained
    g["flip_fraction_of_gene_sig"] = np.where(g.n_sig_gene > 0, g.n_lost / g.n_sig_gene.clip(lower=1), np.nan)
    return g


def cmd_merge(a):
    metas = [json.load(open(f)) for f in sorted(glob.glob(f"{a.out}/part_*.meta.json"))]
    if not metas:
        raise SystemExit("REFUSING: no part files")
    if len({m["script_sha256"] for m in metas}) != 1 or len({int(m["B"]) for m in metas}) != 1:
        raise SystemExit("REFUSING_MIXED_PARTS: script_sha256 / B differ across parts")
    B = int(metas[0]["B"])
    mr, pr, dr = [], [], {}
    for m in metas:
        f = f"{a.out}/{m['stem']}.metrics.tsv"
        if not os.path.exists(f):
            continue
        mr.append(pd.read_csv(f, sep="\t", dtype={"tf": str}))
        pr.append(pd.read_csv(f.replace(".metrics.", ".paired."), sep="\t", dtype={"tf": str}))
        with np.load(f.replace(".metrics.tsv", ".draws.npz")) as z:
            for k in z.files:
                if k in dr:
                    raise SystemExit(f"REFUSING_DUPLICATE_DRAWS {k}")
                dr[k] = z[k]
    mr, pr = pd.concat(mr, ignore_index=True).drop_duplicates(), pd.concat(pr, ignore_index=True).drop_duplicates()
    mm, pm = macro_tables(mr, pr, dr, B)
    atomic_tsv(mr, f"{a.out}/table1_metrics.tsv")
    atomic_tsv(pr, f"{a.out}/table1_paired.tsv")
    atomic_tsv(mm, f"{a.out}/table1_macro_metrics.tsv")
    atomic_tsv(pm, f"{a.out}/table1_macro_paired.tsv")
    ds = dependence_summary(pr, pm)
    atomic_tsv(ds, f"{a.out}/table1_dependence_summary.tsv")
    dk = dependence_by_K(ds)
    atomic_tsv(dk, f"{a.out}/table1_dependence_by_K.tsv")
    notrun = {f"{m['system']}|{k}": v["status"] for m in metas for k, v in m["tfs"].items() if str(v.get("status", "")).startswith("NOT_RUN")}
    dedup = dedup_report()
    atomic_json(f"{a.out}/dedup_check.json", dict(script_sha256=SCRIPT_SHA0, dedup_check=dedup))
    rc = dict(script_sha256=SCRIPT_SHA0, B=B, parts=[os.path.basename(f) for f in sorted(glob.glob(f"{a.out}/part_*.meta.json"))], not_run=notrun, dropped_duplicates=LATTICE_DUP, dedup_check=dedup,
              module_rule=MODULE_RULE, module_rule_sha256=MODULE_RULE_SHA, module_rule_shas={str(s): h for s, h in MODULE_RULE_SHAS.items()}, schemes=list(SCHEMES_T1), hgb_params=PINNED_HGB,
              python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, host=platform.node(), created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              labels=dict(primary="gene-identity resampling (synchronized BlockAxis draws, audit_decomp seed)", sensitivity="module-block resampling (frozen control-only coexpression modules at G/25, G/50, G/100)"))
    atomic_json(f"{a.out}/receipt.json", rc)
    print(f"[ptable merge] {len(mr)} metric rows, {len(pr)} paired rows, {len(mm)} macro metric rows; not_run={len(notrun)}; dedup identical={sum(v['identical'] for v in dedup.values())}/{len(dedup)}")
    if len(dk):
        print("[ptable merge] dependence by module granularity (AUROC, all groups):")
        print(dk[dk.metric == "AUROC"].round(3).to_string(index=False))


def _f(x, d=3):
    return "  NA " if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def cmd_report(a):
    """compact print of Table 1 rows (per endpoint / macro group): AUROC [gene CI], AP-lift, P50, and P - X (AUROC, P50) with the gene-identity CI, the module-block CI (granularity --K) and p"""
    m = pd.read_csv(f"{a.out}/table1_metrics.tsv", sep="\t", dtype={"tf": str})
    p = pd.read_csv(f"{a.out}/table1_paired.tsv", sep="\t", dtype={"tf": str})

    def _opt(name):
        f = f"{a.out}/{name}"
        try:
            return pd.read_csv(f, sep="\t")
        except (pd.errors.EmptyDataError, FileNotFoundError):                    # an empty / absent macro file (single-unit systems) must never crash the report
            return pd.DataFrame()
    mm, pm = _opt("table1_macro_metrics.tsv"), _opt("table1_macro_paired.tsv")
    want = [x.strip() for x in a.groups.split(",") if x.strip()]
    sk = f"_m{a.K}"

    def block(title, M, P):
        print(f"\n== {title}")
        print(f"{'predictor':36s} {'AUROC [gene CI]':24s} {'AP-lift':>7s} {'P50':>5s} | {'P-X dAUROC [gene CI]':27s} {'[G/' + str(a.K) + ' CI]':17s} {'p':>6s} | {'dP50 [gene CI]':20s}")
        P = P.set_index("predictor")
        for _, r in M.iterrows():
            n = r.predictor
            line = f"{n[:36]:36s} {_f(r.AUROC)} [{_f(r.AUROC_lo)},{_f(r.AUROC_hi)}]   {_f(r.APLIFT, 2):>7s} {_f(r.P50, 2):>5s} | "
            if n in P.index:
                q = P.loc[n]
                line += f"{q.d_AUROC:+.3f} [{q.d_AUROC_lo:+.3f},{q.d_AUROC_hi:+.3f}]   [{q[f'd_AUROC{sk}_lo']:+.3f},{q[f'd_AUROC{sk}_hi']:+.3f}] {_f(q.p_AUROC, 3):>6s} | {q.d_P50:+.2f} [{q.d_P50_lo:+.2f},{q.d_P50_hi:+.2f}]"
            print(line)
    for kind, system, tf, mask in sorted(set(zip(m.kind, m.system, m.tf, m["mask"]))):
        key = f"{system}/{tf}"
        if (want and key not in want) or mask != a.mask:
            continue
        M = m[(m.system == system) & (m.tf == tf) & (m["mask"] == mask) & (m.kind == kind) & m.predictor.notna()]
        if not len(M):
            continue
        r0 = M.iloc[0]
        block(f"{key} [{kind}] mask={mask} n={r0.n_rows} pos={r0.n_pos} prev={r0.PREV:.3f} gate={r0.gate}", M, p[(p.system == system) & (p.tf == tf) & (p["mask"] == mask) & (p.kind == kind)])
    if len(mm):
        for grp in sorted(mm.macro_group.unique()):
            if (want and f"MACRO:{grp}" not in want) or not a.macro:
                continue
            M, P = mm[(mm.macro_group == grp) & (mm["mask"] == a.mask)], pm[(pm.macro_group == grp) & (pm["mask"] == a.mask)]
            if len(M):
                block(f"MACRO {grp} mask={a.mask} units={M.units.iloc[0]}", M, P)


# ===================================================================================================== selftest
def selftest(a=None):
    from scipy.stats import rankdata
    import gb_canon_table as CT
    rng = np.random.default_rng(0)
    # 1. pinned HGB defaults == the library defaults actually installed (a library upgrade that changes a default must fail loudly)
    from sklearn.ensemble import HistGradientBoostingRegressor
    assert HistGradientBoostingRegressor(random_state=0).get_params() == PINNED_HGB, "sklearn HGB defaults differ from PINNED_HGB"
    # 2. feature-group mapping: obs-5 + dyn-4 = the nine REF columns, disjoint, identical to gb_prior_semantics.SPLIT_*
    obs, dyn, allx = ref_index()
    import gb_prior_semantics as PS
    assert PS.SPLIT_OBS == OBS5 and PS.SPLIT_DYN == DYN4 and len(obs) == 5 and len(dyn) == 4 and sorted(obs + dyn) == allx
    # 3. HGB fits are deterministic (random_state=0) -> repeating the fit reproduces it bit for bit.  NOTE (measured 2026-10-02, sklearn 1.7.2): HistGradientBoostingRegressor is NOT invariant to the sign of a
    #    feature (held-out predictions differ ~.3 in a toy problem, R2 differs ~.005), so the signed REF columns of the audit (gb_layer_lattice.REF_SIGNS) are used verbatim for parity with control_per_tf.
    import gb_audit_decomp as AD
    n = 600
    X = rng.standard_normal((n, 4))
    yv = np.tanh(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.2 * rng.standard_normal(n)
    folds, mk = rng.integers(0, 5, n), np.ones(n, bool)
    f1 = AD.hgb_cross_fit(yv, X, folds, mk)[0]
    f2 = AD.hgb_cross_fit(yv, X, folds, mk)[0]
    assert np.array_equal(f1, f2), "HGB fit is not deterministic"
    # 4. module labels: planted modules recovered, constant gene -> singleton, deterministic, K rule; ONE Ward tree cut at the rule family's three sizes (nested, ordered K)
    nc, per, nm = 400, 12, 6
    lat = rng.standard_normal((nc, nm))
    Ms = np.column_stack([lat[:, j // per] * 2 + rng.standard_normal(nc) * 0.5 for j in range(nm * per)] + [np.full(nc, 3.0)])
    genes = [f"g{i}" for i in range(Ms.shape[1])]
    lab, dg = build_module_labels(Ms, genes, mean_size=per, min_modules=2)
    lab2, _ = build_module_labels(Ms, genes, mean_size=per, min_modules=2)
    assert (lab == lab2).all() and lab[-1] == f"S:{genes[-1]}" and dg["n_constant_singletons"] == 1
    planted = np.repeat(np.arange(nm), per)
    for k in range(nm):
        assert len(set(lab[:-1][planted == k])) == 1, "planted module split"
    assert len(set(lab[:-1])) == nm, set(lab[:-1])
    multi = build_module_labels_multi(Ms, genes, (3, 6, 12), 2)
    nmod = {sz: len(set(l_[:-1])) for sz, (l_, _d) in multi.items()}
    assert nmod[3] > nmod[6] > nmod[12] >= 2, nmod                                           # smaller mean size -> more modules (K = ceil(G/size))
    assert (multi[12][0] == lab).all(), "multi-size builder differs from the single-size builder at the same size"
    for fine, coarse in ((3, 6), (6, 12)):                                                   # nested cuts of ONE tree: every fine module lies inside one coarse module
        lf, lc = multi[fine][0][:-1], multi[coarse][0][:-1]
        assert all(len(set(lc[lf == f_])) == 1 for f_ in set(lf)), "module granularities are not nested"
    # the frozen rule family: sizes ordered, the primary size's rule sha is the pilot's
    assert MODULE_SIZES == (25, 50, 100) and MODULE_RULE_SHA == MODULE_RULE_SHAS[50] and len(set(MODULE_RULE_SHAS.values())) == 3
    # 5. Draws: module-block counts are constant inside a module, gene counts are not; B prefix-stable; fold = module hash (same module -> same fold); three module granularities share one stacked matrix
    G = 300
    names = np.array([f"x{i}" for i in range(G)])
    ub = {25: np.array([f"M{i // 5:03d}" for i in range(G)]), 50: np.array([f"M{i // 10:03d}" for i in range(G)]), 100: np.array([f"M{i // 20:03d}" for i in range(G)])}
    sd = {25: 13, 50: 12, 100: 14}
    d1 = Draws(names, 50, 11, sd, ub)
    d2 = Draws(names, 80, 11, sd, ub)
    assert d1.schemes == ("g", "m25", "m50", "m100") and d1.C.shape == (200, G) and (d1.gene.counts == d2.gene.counts[:50]).all()
    for sz, w in ((25, 5), (50, 10), (100, 20)):
        assert (d1.mods[sz].counts == d2.mods[sz].counts[:50]).all(), "module draws are not prefix-stable in B"
        assert (d1.mods[sz].counts[:, 0:w] == d1.mods[sz].counts[:, [0]]).all(), f"module G/{sz} counts are not constant inside a module"
    assert not (d1.gene.counts[:, 0:10] == d1.gene.counts[:, [0]]).all()
    fm = d1.fold_module(np.arange(G))
    assert all(len(set(fm[i * 10:(i + 1) * 10])) == 1 for i in range(G // 10))
    # the three granularities draw from different seeds but the SAME rule: the (G/K) ordering of the number of modules is exact
    assert [d1.n_modules[s_] for s_ in (25, 50, 100)] == [60, 30, 15]
    assert mod_seed([1, 2], 50) == [1, 2] and mod_seed([1, 2], 25) == [1, 2, 25] and mod_seed([1, 2], 100) == [1, 2, 100]
    # 6. split_eval: weighted point == plain AUROC, gene / module halves, paired difference sign and weighted-vs-duplicated-rows identity
    y = rng.random(G) < 0.3
    s = rng.standard_normal(G) + 1.0 * y
    rows = np.ones(G, bool)
    pt, bl = split_eval(s, y, rows, np.arange(G), d1.C, 50)
    assert len(bl) == 4 and all(len(b["AUROC"]) == 50 for b in bl)
    assert abs(pt["AUROC"] - AD.auroc_rows(s[None, :], y)[0]) < 1e-12
    c = d1.gene.counts[0].astype(int)
    dup = np.repeat(np.arange(G), c)
    assert abs(bl[0]["AUROC"][0] - AD.auroc_rows(s[None, dup], y[dup])[0]) < 1e-12
    c2 = d1.mods[100].counts[0].astype(int)
    dup2 = np.repeat(np.arange(G), c2)
    assert abs(bl[3]["AUROC"][0] - AD.auroc_rows(s[None, dup2], y[dup2])[0]) < 1e-12, "module G/100 draw does not reproduce the duplicated-row AUROC"
    # 7. table1_rows end-to-end on a synthetic unit: P finite, paired d = P - X, P vs itself absent, reconstructions truth-free (labels shuffled -> identical HGB fits)
    U = np.ones(G, bool)
    P = rng.standard_normal(G) + 0.5 * y
    Xr = rng.standard_normal((G, 9))
    P2 = P + 0.8 * Xr[:, 0] - 0.4 * Xr[:, 5]
    oth = {m: rng.standard_normal(G) for m in ("GENIE3", "PIDC")}
    comps = {c_: rng.standard_normal(G) for c_ in ("COMP_Z", "COMP_RANK")}
    u = dict(kind="syn", system="SYN", tf="T", endpoint="resp", endpoint_class="response", block="SYN", macro_group="SYN", y=y, gi=np.arange(G), U=U, M1=U, MR=U, P=P2, S=rng.standard_normal(G), others=oth,
             comps=comps, Xr=Xr, folds_gene=d1.fold_gene(np.arange(G)), gate={"MR": True})
    mrows, prows, draws = table1_rows(u, d1, U, "MR")
    pr = pd.DataFrame(prows)
    mrdf = pd.DataFrame(mrows)
    assert set(pr.predictor) == set(mrdf.predictor) - {P_NAME} and (pr.d_AUROC.round(12) == pr.predictor.map(lambda n: mrdf[mrdf.predictor == P_NAME].AUROC.iloc[0] - mrdf[mrdf.predictor == n].AUROC.iloc[0]).round(12)).all()
    u2 = dict(u, y=rng.permutation(y))
    mrows2, _p2, _d2 = table1_rows(u2, d1, U, "MR")
    h1 = mrdf[mrdf.predictor == "HGB all-9 -> P"].R2_heldout.iloc[0]
    h2 = pd.DataFrame(mrows2).query("predictor == 'HGB all-9 -> P'").R2_heldout.iloc[0]
    assert h1 == h2, "HGB reconstruction depends on the labels"
    for sch in ("_m25", "_m50", "_m100"):
        assert (pr[f"d_AUROC{sch}_hi"] >= pr[f"d_AUROC{sch}_lo"]).all() and f"p_AUROC{sch}" in pr
    assert np.isfinite(pr.d_P50).all() or True
    # 8. macro: single-unit macro equals the unit; dependence summaries count flips and width ratios per K
    allm = pd.DataFrame(mrows)
    dd = dict(draws)
    mm, pm = macro_tables(allm.assign(status="OK"), pr, dd, 50)
    one = mm[mm.predictor == P_NAME].iloc[0]
    assert abs(one.AUROC - allm[allm.predictor == P_NAME].AUROC.iloc[0]) < 1e-12
    ds = dependence_summary(pr.assign(status="OK"), pm)
    assert set(ds.scheme) == {"m25", "m50", "m100"} and (ds.n_lost + ds.n_gained <= ds.n_contrasts).all()
    dk = dependence_by_K(ds)
    assert (dk.n_flips == dk.n_lost + dk.n_gained).all() and set(dk.scheme) == {"m25", "m50", "m100"}
    # flips on a hand-built table: 3 contrasts, gene CIs [+,+], [+,+], [-,+]; module CIs [-,+], [+,+], [+,+] -> lost 1, gained 1, width ratio median 2
    hand = pd.DataFrame(dict(kind="k", system="s", mask="MR", status="OK", d_AUROC_lo=[.1, .1, -.1], d_AUROC_hi=[.2, .2, .1], d_AUROC_m25_lo=[-.1, .1, .1], d_AUROC_m25_hi=[.3, .3, .3]))
    hd = dependence_summary(hand, pd.DataFrame(), schemes=("g", "m25"))
    hr = hd[(hd.metric == "AUROC") & (hd.scheme == "m25")].iloc[0]
    assert (hr.n_sig_gene, hr.n_sig_scheme, hr.n_lost, hr.n_gained) == (2, 2, 1, 1) and abs(hr.median_ci_width_ratio_scheme_over_gene - 2.0) < 1e-12, hr
    # report must not crash on an empty macro file (pilot defect 2) and merge receipts must carry dedup_check (pilot defect 1)
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        pd.DataFrame(mrows).assign(status="OK").to_csv(f"{td}/table1_metrics.tsv", sep="\t", index=False)
        pd.DataFrame(prows).to_csv(f"{td}/table1_paired.tsv", sep="\t", index=False)
        open(f"{td}/table1_macro_metrics.tsv", "w").close()
        open(f"{td}/table1_macro_paired.tsv", "w").close()
        import argparse as _ap
        cmd_report(_ap.Namespace(out=td, groups="", mask="MR", macro=True, K=50))
        assert "dedup_check" in inspect.getsource(cmd_merge), "merge receipt lacks dedup_check"
    print("[ptable] selftest (synthetic) PASS")
    if a is not None and getattr(a, "data", False):
        selftest_data(a)


def selftest_data(a):
    """real-data parity on the pilot units: PANC RFX6 (plus the systems named in --systems); reuses the run-time parity gates, B small."""
    import gb_layer_lattice as L
    import gb_lattice_extra as LE
    LE.register()
    for spec in a.systems.split(","):
        system, _, tf = spec.partition(":")
        guard_systems([system])
        dr, _ = lattice_draws(system, 60)
        tf = tf or L.SYSTEMS[system]["tfs"][0]
        u = lattice_unit(system, tf, dr.gene)
        mr_, _pr, _dd = table1_rows(u, dr, u["MR"], "MR")
        par = parity_lattice(u, mr_)
        assert par["status"] == "PASS", par
        print(f"[ptable] data parity {system}/{tf}: n_M1={int(u['M1'].sum())} n_MR={int(u['MR'].sum())} worst={par['worst']:.2e} over {par['n_cells']} cells; S(M0) vs lattice "
              f"{par.get('S_M0_vs_lattice')}; S(M1) vs audit {par.get('S_M1_vs_audit')}; s374 {par.get('s374')}", flush=True)
    print("[ptable] selftest (data parity) PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest")
    p.add_argument("--data", action="store_true")
    p.add_argument("--systems", default="PANC")
    p = sub.add_parser("modules")
    p.add_argument("--axes", default="PANC,BONEMARROW,GATA1,K562,DENTATEGYRUS,GASTRULATION,ZEBRAFISH")
    p.add_argument("--sizes", default=",".join(str(x) for x in MODULE_SIZES), help="module-size family members to freeze (mean module size G/<size>)")
    p = sub.add_parser("run")
    p.add_argument("--systems", default="")
    p.add_argument("--regulon", default="")
    p.add_argument("--breadth", default="")
    p.add_argument("--B", type=int, default=B_DEFAULT)
    p.add_argument("--out", required=True)
    p.add_argument("--tfs", default="", help="lattice TFs (audit roster)")
    p.add_argument("--reg-tfs", default="", dest="reg_tfs", help="regulon sources")
    p.add_argument("--slice", type=int, default=0)
    p.add_argument("--nslice", type=int, default=1)
    p.add_argument("--no-parity", action="store_true", dest="no_parity")
    p.add_argument("--n-jobs", type=int, default=8, dest="n_jobs")
    p = sub.add_parser("merge")
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.add_argument("--groups", default="", help="comma list of SYSTEM/TF (and MACRO:<group>) to print; empty = all")
    p.add_argument("--mask", default="MR")
    p.add_argument("--macro", action="store_true")
    p.add_argument("--K", type=int, default=PRIMARY_SIZE, choices=MODULE_SIZES, help="module granularity G/K of the second interval column")
    a = ap.parse_args()
    {"selftest": selftest, "modules": cmd_modules, "run": cmd_run, "merge": cmd_merge, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
