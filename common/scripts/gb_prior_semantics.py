#!/usr/bin/env python
# Target-prior supplemental analyses (evaluation only; the production score Z is unchanged)
# Input: catalogue-bank priors and features (<AXIS>_priors.npz, <AXIS>_features.tsv) and endpoint truth tables
# Output: axis/unit variant tables, bootstrap tables and receipts (tsv, json)
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = __import__("os").environ["GEOBRIDGE_ROOT"]
for _p in (str(HERE), f"{R}/scratch"):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gb_readout as RO  # noqa: E402


def _sha(p):
    return RO._sha(p)


SCRIPT_SHA0 = _sha(__file__)
CB = f"{R}/results/real_uniform/catbank"
OUTROOT = f"{R}/results/real_uniform/prior_semantics"
UNIT_DIRS = [f"{R}/results/real_uniform/readout_units", f"{OUTROOT}/units_extra"]
LAMBERT = f"{R}/scratch/capacity_verify/lambert_TF_v1.01.csv"
SEED0 = 20261001
NFOLD = 5
MIN_CLASS = 20
HGB_KW = dict(learning_rate=0.1, max_iter=100, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, early_stopping=False, random_state=0)
N_SUB, N_FAMSPLIT, MIN_FAM, GATE_MED, GATE_P5 = 20, 10, 5, 0.90, 0.80
N_DECOY, MIN_DECOY, MIN_COMMON_TFS, MIN_COMMON_ROWS = 100, 50, 50, 200
CALIPER, MIN_CALIPER_DECOY = 0.5, 20          # decoy caliper: every one of the four covariates within 0.5 pooled-SD of the focal (Chebyshev); NO_CALL below 20 in-caliper decoys
GO_FILE = f"{R}/data/collectri/GO_Biological_Process_2023.txt"
SPLIT_OBS = ["raw_det", "sm_det", "abundance", "pair_support", "mu_det"]                 # observability / detection nuisance (confounding by detection)
SPLIT_DYN = ["dynamics", "mu_dynamics", "abs_spearman_mu_dpt", "mu_logvar"]             # dynamics nuisance (mediation via dynamics)
B_REFIT = 100
SPEC_DEPARTURES = [
    "FOLDS: 5 gene-HASH folds (sha1 of the upper-case gene name mod 5), not 'control-defined gene-module folds': no module map exists (results/real_uniform/layer_lattice/modules absent); gene bootstrap blocks are therefore single genes",
    "COVERAGE COVARIATES: ChIP indegree uses ONE assay-opportunity covariate (log1p neighbouring TSS count within +-5 kb of the assigned TSS); no peak-calling depth / assay-quality covariate exists on disk; assay coverage is otherwise reported only as TF counts. "
    "STUDY-BIAS covariate (all indegree sources) = log1p(# GO Biological Process 2023 terms containing the gene; upper-case symbol match, mouse genes by upper-case symbol = approximate): a proxy for annotation/study intensity, not citation counts (none on disk)",
    "DECOY OVERLAP: decoys are bank rows of the SAME bank that defines P (the focal row is inside every decoy's leave-out prior and each decoy inside the focal's); decoys of different TFs of one system are drawn from the same pool and share cells, so decoy "
    "increments are not independent across TFs; matched on raw_det, abundance, |rho(Ms,DPT)| and loading with a per-covariate caliper of 0.5 pooled SD (NO_CALL < 20 in caliper), at most 100 nearest decoys; "
    "the empirical decoy p has a floor 1/(n_decoy+1)",
    "SCALE SENSITIVITY: the logistic beta*S~ term is affine-invariant, so the E-z vs T7-z choice cannot change the increment; it is tabulated for D = S~ - P only",
    "Analysis constants: margins AUROC .02 / AP-lift .10 / P50 .05; spline (5 quantile knots, degree 3) + L2 logistic C=1 with label-free feature transforms fit on E; MIN_CLASS 20; Lambert DBD families >= 5 bank members; 10 family partitions; "
    "chip floor 10 TFs; common-bank floors 50 TFs / 200 genes / 20 per class; 20 seeded 50% subsamples",
    "INFERENCE: TF-level bootstrap / exact sign-flip treat TFs as independent although TFs of one system share cells, controls and response programs; single-unit cells use a conditional gene bootstrap; BM_CHIPATLAS / BONEMARROW / BM_GATA1_ACUTE share cells and "
    "GATA1, RUNX1, TAL1 appear in both BM_CHIPATLAS and BONEMARROW; verdict SUPERIOR/INFERIOR needs Holm p < .05 AND the Bonferroni-corrected interval to clear 0; cells whose Holm floor (family size x smallest attainable p) >= .05 are UNTESTABLE",
    "INDEGREE: ChIP indegree exists only for human contexts (K562 ENCODE 33 TFs; BONEMARROW ChIP-Atlas+MK 10 TFs, at the floor); residual-Spearman CI is the gene bootstrap CONDITIONAL on the cross-fit, plus (primary HGB arm only) a refit-inside-bootstrap CI with B_refit=100",
    "EXCLUDED: BM_BREADTH, ZEBRAFISH, NO_CALL units (n_resp < 20)"]
CHIP_FLOOR = 10
DELTA = {"AUROC": 0.02, "APLIFT": 0.10, "P50": 0.05}
SYS_AXIS = {"GATA1": "GATA1", "KLF1": "GATA1", "KLF1_B": "GATA1", "PANC": "PANC", "BM_CHIPATLAS": "BONEMARROW", "BONEMARROW": "BONEMARROW", "BM_GATA1_ACUTE": "BONEMARROW",
            "K562_EMP_ALLNULL": "K562"}
BLOCK2 = {"GATA1": "ERYTHROID", "KLF1": "ERYTHROID", "KLF1_B": "ERYTHROID", "PANC": "PANC", "BM_CHIPATLAS": "BM", "BONEMARROW": "BM", "BM_GATA1_ACUTE": "BM", "K562_EMP_ALLNULL": "K562"}
AXIS_SPECIES = {"GATA1": "mouse", "PANC": "mouse", "DENTATEGYRUS": "mouse", "GASTRULATION": "mouse", "BONEMARROW": "human", "K562": "human"}
REFS9 = ["REF -raw_det", "REF -sm_det", "REF -abundance", "REF +dynamics", "REF -pair_support", "REF -mu_det", "REF +mu_dynamics", "REF +abs_spearman_mu_dpt", "REF +mu_logvar"]
REF_COLS = ["raw_det", "sm_det", "abundance", "dynamics", "pair_support", "mu_det", "mu_dynamics", "abs_spearman_mu_dpt", "mu_logvar"]
ZKEYS = ("P_pairU", "P_pairM", "P_cellU", "P_cellM")
ARMS = ("raw", "obs_cov", "obs_resid")
METS = ("AUROC", "APLIFT", "AUPRC", "P50", "P100", "PREV")


def seed_of(*parts):
    return int(np.random.SeedSequence([SEED0] + [zlib.crc32(str(p).encode()) for p in parts]).generate_state(1)[0])


# fixed gene-hash folds (sha1 of the upper-case gene name mod 5) == gb_layer_lattice.BlockAxis.fold_of for the gene-level scheme (no module map on disk)
def gene_folds(genes):
    return np.array([int(hashlib.sha1(str(g).upper().encode()).hexdigest(), 16) % NFOLD for g in genes])


# pure numerics (selftested)
def agg_mean(Zs):
    f = np.isfinite(Zs)
    c = f.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c >= 2, np.where(f, Zs, 0.0).sum(0) / np.maximum(c, 1), np.nan)


def agg_median(Zs):
    c = np.isfinite(Zs).sum(0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = np.nanmedian(Zs, axis=0)
    return np.where(c >= 2, m, np.nan)


# column mean after dropping floor(q*c) finite values from each tail (== scipy.stats.trim_mean per column)
def agg_trim(Zs, q=0.10):
    f = np.isfinite(Zs)
    c = f.sum(0)
    S = np.sort(np.where(f, Zs, np.inf), axis=0)
    cs0 = np.vstack([np.zeros((1, S.shape[1])), np.cumsum(np.where(np.isfinite(S), S, 0.0), axis=0)])
    k = np.floor(q * c).astype(int)
    col = np.arange(S.shape[1])
    n = c - 2 * k
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((c >= 2) & (n >= 1), (cs0[c - k, col] - cs0[k, col]) / np.maximum(n, 1), np.nan)


# within-row percentile (rank - .5)/n_finite of every finite entry; rows with < 3 finite or constant -> all NaN
def row_pct(Z):
    from scipy.stats import rankdata
    out = np.full(Z.shape, np.nan)
    for i in range(len(Z)):
        f = np.isfinite(Z[i])
        if f.sum() >= 3 and np.ptp(Z[i][f]) > 0:
            out[i, f] = (rankdata(Z[i][f]) - 0.5) / f.sum()
    return out


# population row-z over the finite entries of every row (NaN stays NaN); constant / < 3 finite rows -> all NaN
def rowz(X):
    X = np.asarray(X, float)
    out = np.full(X.shape, np.nan)
    for i in range(len(X)):
        f = np.isfinite(X[i])
        if f.sum() >= 3:
            s = X[i][f].std()
            if s > 0:
                out[i, f] = (X[i][f] - X[i][f].mean()) / s
    return out


def zvec(v, mask):
    out = np.full(len(v), np.nan)
    x = np.asarray(v, float)[mask]
    out[mask] = (x - x.mean()) / x.std()
    return out


def spear(a, b):
    from scipy.stats import rankdata
    a, b = np.asarray(a, float), np.asarray(b, float)
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < 5:
        return np.nan
    ra, rb = rankdata(a[f]) - rankdata(a[f]).mean(), rankdata(b[f]) - rankdata(b[f]).mean()
    d = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / d) if d > 0 else np.nan


def topk_overlap(a, b, K=50):
    f = np.isfinite(a) & np.isfinite(b)
    if f.sum() < K:
        return np.nan
    ia, ib = np.flatnonzero(f), np.flatnonzero(f)
    ta, tb = ia[np.lexsort((ia, -np.asarray(a)[f]))][:K], ib[np.lexsort((ib, -np.asarray(b)[f]))][:K]
    return len(set(ta) & set(tb)) / K


# tie-grouped point metrics (AUROC, AUPRC, APLIFT, PREV, P20/P50/P100) of `score` on `rows` (all finite there)
def pt_metrics(score, y, rows):
    import gb_canon_table as CT
    idx = np.flatnonzero(rows)
    sc = np.asarray(score, float)[idx]
    if not np.isfinite(sc).all():
        raise RuntimeError("pt_metrics: non-finite score inside rows")
    o, st = CT.prep_scores(sc)
    m = CT.weighted_metrics(np.ones((1, len(idx))), np.asarray(y, bool)[idx][o], st, n_rows=len(idx))
    return {k: float(v[0]) for k, v in m.items()}


def boot_q(d, qs=(2.5, 97.5)):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return [float(x) for x in np.percentile(d, qs)] if len(d) > 20 else [np.nan] * len(qs)


def boot_p(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min(1 + (d <= 0).sum(), 1 + (d >= 0).sum()) / (len(d) + 1))) if len(d) else np.nan


# cross-fitted frozen hgb of v on X over `mask`, fixed folds -> held-out fit (NaN outside mask)
def hgb_cf(v, X, folds, mask):
    from sklearn.ensemble import HistGradientBoostingRegressor
    v, X, folds, mask = np.asarray(v, float), np.asarray(X, float), np.asarray(folds), np.asarray(mask, bool)
    fit = np.full(len(v), np.nan)
    for f in np.unique(folds[mask]):
        te, tr = mask & (folds == f), mask & (folds != f)
        fit[te] = HistGradientBoostingRegressor(**HGB_KW).fit(X[tr], v[tr]).predict(X[te])
    return fit


def spline_basis(v, mask):
    from sklearn.preprocessing import SplineTransformer
    out = np.zeros((len(v), 6))
    x = np.asarray(v, float)[mask][:, None]
    out[mask] = SplineTransformer(n_knots=5, degree=3, knots="quantile", extrapolation="constant", include_bias=False).fit_transform(x)
    return out


def lin_block(M, mask):
    M = np.atleast_2d(np.asarray(M, float).T).T if np.ndim(M) == 1 else np.asarray(M, float)
    out = np.zeros(M.shape)
    mu, sd = M[mask].mean(0), M[mask].std(0)
    out[mask] = (M[mask] - mu) / np.where(sd > 0, sd, 1.0)
    return out


# held-out predicted probability (linear score) of a fixed L2 logistic (C=1) by `folds`, fitted on the training folds of `mask` only
# NaN outside mask
def oof_logit(X, y, folds, mask):
    from sklearn.linear_model import LogisticRegression
    X, y, folds, mask = np.asarray(X, float), np.asarray(y, bool), np.asarray(folds), np.asarray(mask, bool)
    out = np.full(len(y), np.nan)
    for f in np.unique(folds[mask]):
        te, tr = mask & (folds == f), mask & (folds != f)
        if y[tr].sum() < 5 or (~y[tr]).sum() < 5:
            return None
        m = LogisticRegression(C=1.0, max_iter=5000).fit(X[tr], y[tr])
        out[te] = m.decision_function(X[te])
    return out


# axis context (catbank priors; truth-free)
# one catbank axis: members, T7 genes, row-z matrices Z[key] (nb, nT7), bank features, Lambert families, frozen subsamples / family partitions
class AxisCtx:
    _cache = {}

    @classmethod
    def get(cls, ax):
        if ax not in cls._cache:
            cls._cache[ax] = cls(ax)
        return cls._cache[ax]

    def __init__(self, ax):
        self.ax = ax
        pf = f"{CB}/{ax}_priors.npz"
        bj = json.load(open(f"{CB}/{ax}_bank.json"))
        rc = json.load(open(f"{CB}/{ax}_priors.receipt.json"))
        if rc["config"]["bank_sha256"] != bj["sha256_members"]:
            raise SystemExit(f"REFUSING_PRIORS_BANK_MISMATCH {ax}")
        z = np.load(pf, allow_pickle=True)
        self.members = [str(x) for x in z["members"]]
        if self.members != bj["members"]:
            raise SystemExit(f"REFUSING_MEMBERS_DIFFER {ax}")
        self.axis = [str(x) for x in z["axis"]]
        self.t7 = [str(x) for x in z["T7|genes"]]
        self.t7col = {g: i for i, g in enumerate(self.t7)}
        self.Z = {k: np.asarray(z[f"T7|{k}|Z"], float) for k in ZKEYS}
        self.nb, self.nT = self.Z["P_pairU"].shape
        self.mu = np.array([m.upper() for m in self.members])
        self.shas = dict(priors=_sha(pf), bank=_sha(f"{CB}/{ax}_bank.json"), features=_sha(f"{CB}/{ax}_features.tsv"))
        F = pd.read_csv(f"{CB}/{ax}_features.tsv", sep="\t", index_col="gene")
        self.F = F
        self.mouse = ax in RO.MOUSE_AXES
        self.pct = row_pct(self.Z["P_pairU"])
        self._fam()
        rg = [np.random.default_rng(seed_of("sub", ax, i)) for i in range(N_SUB)]
        self.subs = []
        for g in rg:
            h = np.zeros(self.nb, bool)
            h[g.permutation(self.nb)[: self.nb // 2]] = True
            self.subs.append(h)
        self._famsplits()

    def row_of(self, name):
        w = np.flatnonzero(self.mu == name.upper())
        return int(w[0]) if len(w) else None

    def _fam(self):
        L_ = pd.read_csv(LAMBERT, usecols=["HGNC symbol", "DBD", "Is TF?"])
        L_ = L_[L_["Is TF?"] == "Yes"].dropna(subset=["HGNC symbol", "DBD"])
        fam = dict(zip(L_["HGNC symbol"].str.upper(), L_["DBD"]))
        keys = RO.keymap_factory()(self.members, self.mouse)
        self.fam_of = np.array([fam.get(k) if k is not None else None for k in keys], object)
        self.keys = keys
        sz = pd.Series([f for f in self.fam_of if f is not None]).value_counts()
        self.fam_sizes = sz
        self.fam_masks = {f: np.array([x == f for x in self.fam_of]) for f in sz.index if sz[f] >= MIN_FAM}
        self.fam_cov = dict(n_members=self.nb, n_annotated=int(sum(x is not None for x in self.fam_of)), n_families=int(len(sz)), n_families_ge_min=int(len(self.fam_masks)),
                            source=os.path.basename(LAMBERT), min_family_size=MIN_FAM)

    def _famsplits(self):
        ann = np.array([x is not None for x in self.fam_of])
        fams = sorted(self.fam_sizes.index)
        self.famsplit, self.randsplit = [], []
        for i in range(N_FAMSPLIT):
            g = np.random.default_rng(seed_of("famsplit", self.ax, i))
            # balance: assign families in random order to the currently smaller half (disjoint by
            # construction)
            order = g.permutation(len(fams))
            A_, B_, na, nb_ = set(), set(), 0, 0
            for j in order:
                if na <= nb_:
                    A_.add(fams[j]),
                    na += int(self.fam_sizes[fams[j]])
                else:
                    B_.add(fams[j])
                    nb_ += int(self.fam_sizes[fams[j]])
            mA = np.array([x in A_ for x in self.fam_of]) & ann
            mB = np.array([x in B_ for x in self.fam_of]) & ann
            self.famsplit.append((mA, mB))
            idx = np.flatnonzero(ann)
            g2 = np.random.default_rng(seed_of("randsplit", self.ax, i))
            p = g2.permutation(idx)
            ra, rb = np.zeros(self.nb, bool), np.zeros(self.nb, bool)
            ra[p[: len(p) // 2]], rb[p[len(p) // 2: 2 * (len(p) // 2)]] = True, True
            self.randsplit.append((ra, rb))

    # {name: P-vector over T7 columns}
    def variants(self, drop=None, which=("agg", "sub", "fam", "famsplit", "randsplit")):
        keep = np.ones(self.nb, bool)
        if drop is not None:
            keep[drop] = False
        Z = self.Z["P_pairU"]
        out = {}
        if "agg" in which:
            Zk = Z[keep]
            out["mean"], out["median"], out["trim10"] = agg_mean(Zk), agg_median(Zk), agg_trim(Zk)
            out["pctmean"] = agg_mean(self.pct[keep])
        if "sub" in which:
            for i, h in enumerate(self.subs):
                out[f"sub{i:02d}"] = agg_mean(Z[h & keep])
        if "fam" in which:
            for f, m in self.fam_masks.items():
                out[f"fam:{f}"] = agg_mean(Z[keep & ~m])
        if "famsplit" in which:
            for i, (mA, mB) in enumerate(self.famsplit):
                out[f"famA{i:02d}"], out[f"famB{i:02d}"] = agg_mean(Z[keep & mA]), agg_mean(Z[keep & mB])
        if "randsplit" in which:
            for i, (mA, mB) in enumerate(self.randsplit):
                out[f"rndA{i:02d}"], out[f"rndB{i:02d}"] = agg_mean(Z[keep & mA]), agg_mean(Z[keep & mB])
        return out

    # leave-focal-out prior of Z[key] over T7 (the catbank loo_prior algebra, >= 2 contributors)
    def p_loo(self, key, r):
        keep = np.ones(self.nb, bool)
        if r is not None:
            keep[r] = False
        return agg_mean(self.Z[key][keep])

    def to_genes(self, v_t7, genes):
        tc = np.array([self.t7col.get(g, -1) for g in genes])
        return np.where(tc >= 0, np.asarray(v_t7, float)[np.maximum(tc, 0)], np.nan), tc


def vclass(name):
    return {"mean": "primary", "median": "agg", "trim10": "agg", "pctmean": "agg"}.get(name) or ("sub" if name.startswith("sub") else "fam" if name.startswith("fam:") else
                                                                                                "famsplit" if name[:4] in ("famA", "famB") else "randsplit")


# unit preparation
class Unit:
    def __init__(self, u):
        self.u = u
        self.system, self.tf, self.genes = u["system"], u["tf"], np.asarray(u["genes"]).astype(str)
        self.ax = SYS_AXIS[self.system]
        self.ctx = AxisCtx.get(self.ax)
        self.n = len(self.genes)
        self.E = np.asarray(u["Q"], bool) & np.asarray(u["test"], bool)
        self.y = np.asarray(u["resp"], bool)
        self.n_resp, self.n_non = int((self.y & self.E).sum()), int((~self.y & self.E).sum())
        self.r = self.ctx.row_of(self.tf)
        if self.r is None:
            raise RuntimeError(f"focal {self.tf} not in bank {self.ax}")
        self.folds = gene_folds(self.genes)
        self.O = np.column_stack([np.asarray(u["vec"][c], float) for c in REFS9])
        pu, self.tc = self.ctx.to_genes(self.ctx.p_loo("P_pairU", self.r), self.genes)
        vu = np.asarray(u["vec"]["P_pairU"], float)
        both = np.isfinite(pu) & np.isfinite(vu)
        if np.abs(pu[both] - vu[both]).max() > 1e-9 or (np.isfinite(vu) & ~np.isfinite(pu)).any():
            raise SystemExit(f"REFUSING_PARITY {self.system}/{self.tf}: recomputed leave-focal-out P differs from the unit cache")
        self.P = pu
        self.block = BLOCK2[self.system]
        self.parity = float(np.abs(pu[both] - vu[both]).max())
        self.gi = np.arange(self.n)

    def counts(self, B):
        import gb_canon_table as CT
        k = (self.system, self.n, B)
        if k not in Unit._counts:
            Unit._counts[k] = CT.make_counts(self.n, B, RO.cov_seed("prior_semantics:" + self.system))
        return Unit._counts[k]

    _counts = {}

    def eval_boot(self, score, rows, B):
        import gb_canon_table as CT
        return CT.eval_rows(score, self.y, rows, self.gi, self.counts(B))

    def base(self):
        return dict(system=self.system, block=self.block, axis=self.ax, tf=self.tf, n_E=int(self.E.sum()), n_resp=self.n_resp, gated=self.n_resp >= MIN_CLASS)


def load_units(a):
    dirs = [d for d in UNIT_DIRS if os.path.isdir(d)]
    out = []
    want = set(a.systems.split(",")) if a.systems else None
    tfs = set(t.strip() for t in a.tfs.split(",") if t.strip()) if getattr(a, "tfs", "") else None
    seen = set()
    for d in dirs:
        for u in RO.load_units(d, want):
            if u["system"] == "BM_BREADTH" or u["system"] not in SYS_AXIS:
                continue
            if tfs and u["tf"] not in tfs:
                continue
            if (u["system"], u["tf"]) in seen:
                continue
            seen.add((u["system"], u["tf"]))
            out.append(u)
    out.sort(key=lambda u: (u["system"], u["tf"]))
    if getattr(a, "max_units", 0):
        out = out[: a.max_units]
    if getattr(a, "nslice", 1) > 1:
        out = [u for i, u in enumerate(out) if i % a.nslice == a.slice]
    return out


def tag_of(a):
    return (a.tag or "all") + (f"_s{a.slice}of{a.nslice}" if getattr(a, "nslice", 1) > 1 else "")


def write_tsv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def receipt(a, out, part, extra=None):
    assert _sha(__file__) == SCRIPT_SHA0, "script changed during the run"
    ax = {k: v.shas for k, v in AxisCtx._cache.items()}
    rec = dict(script=os.path.abspath(__file__), script_sha256=SCRIPT_SHA0, part=part, spec_departures=SPEC_DEPARTURES, argv=sys.argv, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
               axis_artifact_shas=ax, constants=dict(HGB_KW=HGB_KW, NFOLD=NFOLD, MIN_CLASS=MIN_CLASS, N_SUB=N_SUB, N_FAMSPLIT=N_FAMSPLIT, MIN_FAM=MIN_FAM, GATE=[GATE_MED, GATE_P5],
                                                    N_DECOY=N_DECOY, DELTA=DELTA, B=getattr(a, "B", None), SEED0=SEED0, SYS_AXIS=SYS_AXIS),
               gate_note="folds = gene-hash (no module map on disk)", **(extra or {}))
    json.dump(rec, open(f"{out}/receipt_{part}_{tag_of(a)}.json", "w"), indent=1, default=str)


def used_unit_shas(units):
    sh = {}
    for d in UNIT_DIRS:
        for f in sorted(glob.glob(f"{d}/*.npz")):
            b = os.path.basename(f)[:-4]
            if any(b == f"{u['system']}__{u['tf']}" for u in units):
                sh[os.path.relpath(f, R)] = _sha(f)
    return sh


# (3-4) robustness
def cmd_robust(a):
    units = [Unit(u) for u in load_units(a)]
    out = f"{a.out}/robust"
    axes = sorted({u.ax for u in units})
    # AXIS level: full-bank P (no focal), T7 columns. one instance per (axis, variant): no pseudo-replication over TFs
    rows = []
    for ax in axes:
        ctx = AxisCtx.get(ax)
        V = ctx.variants(None)
        prim = V["mean"]
        for nm, v in V.items():
            if nm == "mean":
                continue
            f = np.isfinite(v) & np.isfinite(prim)
            rows.append(dict(axis=ax, variant=nm, vclass=vclass(nm), n_T7=int(f.sum()), spearman=spear(v, prim), top50=topk_overlap(v, prim, 50)))
        for i in range(N_FAMSPLIT):                                    # A vs B comparisons (disjoint families) and the random split-half noise floor
            for pre, lab in (("fam", "famsplit_AvsB"), ("rnd", "randsplit_AvsB")):
                va, vb = V[f"{pre}A{i:02d}"], V[f"{pre}B{i:02d}"]
                f = np.isfinite(va) & np.isfinite(vb)
                rows.append(dict(axis=ax, variant=f"{lab}{i:02d}", vclass=lab, n_T7=int(f.sum()), spearman=spear(va, vb), top50=topk_overlap(va, vb, 50)))
        rows.append(dict(axis=ax, variant="__coverage__", vclass="coverage", n_T7=ctx.nT, spearman=np.nan, top50=np.nan, **{f"fam_{k}": v for k, v in ctx.fam_cov.items()}))
    write_tsv(rows, f"{out}/axis_variants_{tag_of(a)}.tsv")
    # UNIT level: leave-focal-out P on E; response metrics per variant (no selection)
    rows, brows = [], []
    for u in units:
        t0 = time.time()
        V = u.ctx.variants(u.r)
        Pu = {nm: u.ctx.to_genes(v, u.genes)[0] for nm, v in V.items()}
        prim = Pu["mean"]
        for nm, v in Pu.items():
            f = u.E & np.isfinite(v) & np.isfinite(prim)
            ok = int((f & u.y).sum()) > 0 and int((f & ~u.y).sum()) > 0
            mv = pt_metrics(v, u.y, f) if ok else {}
            mp = pt_metrics(prim, u.y, f) if ok else {}
            rows.append(dict(**u.base(), variant=nm, vclass=vclass(nm), n_common=int(f.sum()), frac_E=float(f.sum() / max(u.E.sum(), 1)), spearman=spear(v[f], prim[f]) if f.any() else np.nan,
                             top50=topk_overlap(np.where(f, v, np.nan), np.where(f, prim, np.nan), 50),
                             **{f"{k}": mv.get(k, np.nan) for k in METS}, **{f"prim_{k}": mp.get(k, np.nan) for k in METS}))
        com = u.E.copy()
        for nm in ("mean", "median", "trim10", "pctmean"):
            com &= np.isfinite(Pu[nm])
        if int((com & u.y).sum()) > 0 and int((~u.y & com).sum()) > 0:
            p0, b0 = u.eval_boot(prim, com, a.B)
            for nm in ("median", "trim10", "pctmean"):
                p1, b1 = u.eval_boot(Pu[nm], com, a.B)
                for met in ("AUROC", "APLIFT", "P50"):
                    d = b1[met] - b0[met]
                    lo, hi = boot_q(d)
                    brows.append(dict(**u.base(), variant=nm, metric=met, primary=p0[met], alt=p1[met], diff=p1[met] - p0[met], lo=lo, hi=hi, p_boot=boot_p(d), n_common=int(com.sum())))
        print(f"[robust] {u.system}/{u.tf}: {len(Pu)} variants parity={u.parity:.1e} {time.time()-t0:.0f}s", flush=True)
    write_tsv(rows, f"{out}/unit_variants_{tag_of(a)}.tsv")
    write_tsv(brows, f"{out}/unit_boot_{tag_of(a)}.tsv")
    receipt(a, out, "robust", dict(unit_shas=used_unit_shas([u.u for u in units]), family_coverage={k: v.fam_cov for k, v in AxisCtx._cache.items()}))


# (7) breadth / intensity
def cmd_breadth(a):
    units = [Unit(u) for u in load_units(a)]
    out = f"{a.out}/breadth"
    rows = []
    for u in units:
        Z = u.ctx.Z["P_pairU"]
        keep = np.ones(u.ctx.nb, bool)
        keep[u.r] = False
        Zk = Z[keep]
        f = np.isfinite(Zk)
        tau = np.full(len(Zk), np.nan)
        for i in range(len(Zk)):
            if f[i].sum() >= 3:
                tau[i] = np.percentile(Zk[i][f[i]], 95)                  # outcome-free per-row threshold: the row's own 95th percentile over T7
        okr = np.isfinite(tau)
        Zk, f, tau = Zk[okr], f[okr], tau[okr]
        contrib = f & (Zk > tau[:, None])
        exc = np.where(contrib, Zk - tau[:, None], 0.0)
        nfin, ncon = f.sum(0), contrib.sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            breadth = np.where(nfin >= 2, ncon / np.maximum(nfin, 1), np.nan)
            inten_c = np.where(ncon >= 1, exc.sum(0) / np.maximum(ncon, 1), np.nan)               # mean standardized excess among contributors
            inten_z = np.where(nfin >= 2, exc.sum(0) / np.maximum(nfin, 1), np.nan)               # zeros included (non-contributors count as 0 excess) = breadth * intensity_c
        vecs = {"P": u.P}
        for nm, v in (("breadth", breadth), ("intensity_contrib", inten_c), ("intensity_zeros", inten_z)):
            vecs[nm] = u.ctx.to_genes(v, u.genes)[0]
        n_nocon = int((u.E & (np.nan_to_num(vecs["breadth"], nan=-1) == 0)).sum())
        vecs["intensity_contrib0"] = np.where(np.isfinite(vecs["intensity_contrib"]), vecs["intensity_contrib"], np.where(np.isfinite(vecs["breadth"]), 0.0, np.nan))   # no contributor -> 0 (reported variant)
        for nm, v in vecs.items():
            com = u.E & np.isfinite(v) & np.isfinite(vecs["P"])                # each vector against P on its own finite rows (identical mask for the pair)
            if int((com & u.y).sum()) < 5 or int((~u.y & com).sum()) < 5:
                continue
            pP, bP = u.eval_boot(vecs["P"], com, a.B)
            p1, b1 = u.eval_boot(v, com, a.B)
            d = b1["AUROC"] - bP["AUROC"]
            lo, hi = boot_q(b1["AUROC"])
            dl, dh = boot_q(d)
            rows.append(dict(**u.base(), vector=nm, n_common=int(com.sum()), n_genes_no_contributor=n_nocon, spearman_with_P=spear(v[com], vecs["P"][com]),
                             spearman_breadth_intensity=spear(vecs["breadth"][com], vecs["intensity_contrib"][com]), AUROC=p1["AUROC"], AUROC_lo=lo, AUROC_hi=hi, APLIFT=p1["APLIFT"], AUPRC=p1["AUPRC"],
                             PREV=p1["PREV"], P50=p1["P50"], P100=p1["P100"], dAUROC_vs_P=p1["AUROC"] - pP["AUROC"], dlo=dl, dhi=dh, p_boot=boot_p(d)))
        print(f"[breadth] {u.system}/{u.tf} done", flush=True)
    write_tsv(rows, f"{out}/unit_breadth_{tag_of(a)}.tsv")
    receipt(a, out, "breadth", dict(unit_shas=used_unit_shas([u.u for u in units]), rule="tau_b = 95th percentile of the T7 row-z of bank row b (focal row dropped); contributor = Z > tau_b; breadth = contributors / finite rows; "
                                    "intensity_contrib = mean (Z - tau) over contributors; intensity_zeros = mean over all finite rows of max(Z - tau, 0)"))


# (8-9) incremental value
# (M0 design, M1 design) for an arm; transforms fit on `mask` (label-free)
def arm_design(arm, P, S, O, Pres, Sres, mask, full):
    if arm == "obs_resid":
        b0 = [spline_basis(Pres, mask)]
        b1 = b0 + [lin_block(Sres, mask)]
    else:
        b0 = [spline_basis(P, mask)] + ([lin_block(O, mask)] if arm == "obs_cov" else [])
        b1 = [spline_basis(P, mask), lin_block(S, mask)] + ([lin_block(O, mask)] if arm == "obs_cov" else [])
    return np.hstack(b0), np.hstack(b1)


# S~ on the unit's genes: primary = z over E (current-D scale)
def inc_vectors(u, S_T7_genes):
    return S_T7_genes


def cv_pair(X0, X1, y, folds, mask):
    o0, o1 = oof_logit(X0, y, folds, mask), oof_logit(X1, y, folds, mask)
    return o0, o1


def inc_metrics(o, y, mask):
    return pt_metrics(o, y, mask)


# eligible bank rows: finite S~ on E, not the focal, not a perturbed-panel TF, complete features
def decoy_pool(u, S_all, Pm_all, Fz, excl_names):
    ctx = u.ctx
    E = u.E
    tcE = u.tc[E]
    idx, cov = [], []
    for b in range(ctx.nb):
        if b == u.r or ctx.mu[b] in excl_names:
            continue
        s = S_all[b][tcE]
        if not np.isfinite(s).all() or s.std() == 0:
            continue
        pm = Pm_all[b][tcE]
        if not np.isfinite(pm).all():
            continue
        fe = Fz[b]
        if not np.isfinite(fe).all():
            continue
        idx.append(b)
        cov.append(list(fe) + [spear(s, pm)])
    return np.array(idx, int), np.array(cov, float)


def cmd_increment(a):
    import gb_audit_decomp as AD
    import gb_attribution as A
    import gb_layer_lattice as L
    import gb_lattice_extra as LE
    LE.register()
    units = [Unit(u) for u in load_units(a)]
    out = f"{a.out}/increment"
    rows, drows, srows, store = [], [], [], {}
    by_sys = {}
    for u in units:
        t0 = time.time()
        ctx, E, y = u.ctx, u.E, u.y
        base = u.base()
        if u.n_resp < MIN_CLASS or u.n_non < MIN_CLASS:
            rows.append(dict(**base, arm="ALL", status="NO_CALL", reason=f"n_resp={u.n_resp} n_non={u.n_non} < {MIN_CLASS}"))
            continue
        Zg = ctx.Z["P_pairU"]
        S_T7 = u.ctx.to_genes(Zg[u.r], u.genes)[0]
        Em = E & np.isfinite(S_T7) & np.isfinite(u.P)
        if Em.sum() < 0.9 * E.sum():
            rows.append(dict(**base, arm="ALL", status="NO_CALL", reason="S~/P finite on < 90% of E"))
            continue
        S_M1 = zvec(S_T7, Em)
        Pr = np.where(Em, u.P, np.nan)
        Pres = np.where(Em, Pr - hgb_cf(Pr, u.O, u.folds, Em), np.nan)
        Sres = np.where(Em, S_M1 - hgb_cf(S_M1, u.O, u.folds, Em), np.nan)
        cache = dict(Em=Em, S_M1=S_M1, S_T7=S_T7, Pres=Pres, Sres=Sres)
        by_sys.setdefault(u.system, []).append((u, cache))
        if a.mode == "loto":                                              # leave-one-TF-out needs every unit of the system in one process (never sliced)
            continue
        # cognate increments, all arms
        oofs = {}
        for arm in ARMS:
            X0, X1 = arm_design(arm, Pr, S_M1, u.O, Pres, Sres, Em, None)
            o0, o1 = cv_pair(X0, X1, y, u.folds, Em)
            if o0 is None or o1 is None:
                rows.append(dict(**base, arm=arm, status="NO_CALL", reason="training fold lacks a class"))
                continue
            oofs[arm] = (o0, o1)
            p0, b0 = u.eval_boot(o0, Em, a.B)
            p1, b1 = u.eval_boot(o1, Em, a.B)
            rec = dict(**base, arm=arm, status="OK", n_eval=int(Em.sum()), n_resp_eval=int((y & Em).sum()))
            for m in ("AUROC", "AUPRC", "APLIFT", "P50"):
                d = b1[m] - b0[m]
                lo, hi = boot_q(d)
                rec.update({f"M0_{m}": p0[m], f"M1_{m}": p1[m], f"d{m}": p1[m] - p0[m], f"d{m}_lo": lo, f"d{m}_hi": hi, f"d{m}_p": boot_p(d)})
            rec["PREV"] = p0["PREV"]
            rows.append(rec)
        # scale sensitivity (named): the logistic term beta*S~ is affine-invariant, so the E-z vs T7-z scale matters only for the bookkeeping deviation D = S~ - P: rank by D on both scales
        for nm, Sv in (("D_Ez(current)", S_M1), ("D_T7z", S_T7)):
            Dv = np.where(Em, Sv - Pr, np.nan)
            pD, bD = u.eval_boot(Dv, Em, a.B)
            lo, hi = boot_q(bD["AUROC"])
            srows.append(dict(**base, vector=nm, n_eval=int(Em.sum()), AUROC=pD["AUROC"], AUROC_lo=lo, AUROC_hi=hi, APLIFT=pD["APLIFT"], AUPRC=pD["AUPRC"], PREV=pD["PREV"], P50=pD["P50"]))
        # matched decoy increments
        try:
            perturbed, _ = AD.perturbed_of(u.system, L, LE)
        except (Exception, SystemExit) as e:                             # recorded, never silent
            perturbed = {u.tf.upper()}
            print(f"[increment] perturbed_of failed for {u.system}: {e}", flush=True)
        perturbed = set(perturbed) | {u.tf.upper()}
        Fz = ctx.F.reindex(ctx.members)[["raw_det", "abundance", "abs_spearman_ms_dpt"]].to_numpy(float)
        Sfull = np.where(np.isfinite(Zg), Zg, 0.0).sum(0)
        Cfull = np.isfinite(Zg).sum(0)
        S_all, Pm_all = Zg, np.full((ctx.nb, ctx.nT), np.nan)
        for b in range(ctx.nb):
            fb = np.isfinite(Zg[b])
            with np.errstate(invalid="ignore", divide="ignore"):
                c_ = Cfull - fb
                Pm_all[b] = np.where(c_ >= 2, (Sfull - np.where(fb, Zg[b], 0.0)) / np.maximum(c_, 1), np.nan)
        idx, cov = decoy_pool(u, S_all, Pm_all, Fz, perturbed)
        cov_f = np.array(list(Fz[u.r]) + [spear(S_T7[Em], u.P[Em])], float)
        if len(idx) < MIN_DECOY or not np.isfinite(cov_f).all():
            for arm in ARMS[:3]:
                drows.append(dict(**base, arm=arm, status="NO_CALL", reason=f"matched-decoy pool {len(idx)} < {MIN_DECOY} or focal covariates missing"))
            continue
        mu, sd = cov.mean(0), np.where(cov.std(0) > 0, cov.std(0), 1.0)
        zd = np.abs((cov - mu) / sd - (cov_f - mu) / sd)
        dist = np.sqrt((zd ** 2).sum(1))
        incal = np.flatnonzero((zd <= CALIPER).all(1))                   # CALIPER: all four covariates within 0.5 pooled SD of the focal
        if len(incal) < MIN_CALIPER_DECOY:
            for arm in ARMS:
                drows.append(dict(**base, arm=arm, status="NO_CALL", reason=f"only {len(incal)} decoys within the {CALIPER} SD caliper (< {MIN_CALIPER_DECOY})", pool=len(idx), n_in_caliper=len(incal)))
            continue
        sel = incal[np.lexsort((idx[incal], dist[incal]))][:N_DECOY]
        mdist = float(dist[sel].max())
        for arm in ARMS[:3]:
            if arm not in oofs:
                continue
            dd = {m: [] for m in ("AUROC", "APLIFT", "AUPRC", "P50")}
            for j in sel:
                b = idx[j]
                Sg, Pg = ctx.to_genes(S_all[b], u.genes)[0], ctx.to_genes(Pm_all[b], u.genes)[0]
                if not (np.isfinite(Sg[Em]).all() and np.isfinite(Pg[Em]).all()):
                    continue
                Sb, Pb = zvec(Sg, Em), np.where(Em, Pg, np.nan)
                Pres_b = Sres_b = None
                if arm == "obs_resid":
                    Pres_b = np.where(Em, Pb - hgb_cf(Pb, u.O, u.folds, Em), np.nan)
                    Sres_b = np.where(Em, Sb - hgb_cf(Sb, u.O, u.folds, Em), np.nan)
                X0, X1 = arm_design(arm, Pb, Sb, u.O, Pres_b, Sres_b, Em, None)
                o0, o1 = cv_pair(X0, X1, y, u.folds, Em)
                if o0 is None or o1 is None:
                    continue
                m0, m1 = pt_metrics(o0, y, Em), pt_metrics(o1, y, Em)
                for m in dd:
                    dd[m].append(m1[m] - m0[m])
            n_d = len(dd["AUROC"])
            if n_d < MIN_CALIPER_DECOY:
                drows.append(dict(**base, arm=arm, status="NO_CALL", reason=f"only {n_d} usable matched decoys"))
                continue
            rec = dict(**base, arm=arm, status="OK", n_decoy=n_d, pool=len(idx), n_in_caliper=len(incal), caliper=CALIPER, match_max_dist=mdist, focal_cov=json.dumps([float(x) for x in cov_f]))
            o0, o1 = oofs[arm]
            m0, m1 = pt_metrics(o0, y, Em), pt_metrics(o1, y, Em)
            for m in dd:
                d = np.array(dd[m], float)
                cg = m1[m] - m0[m]
                rec.update({f"cog_d{m}": cg, f"decoy_med_d{m}": float(np.median(d)), f"decoy_q95_d{m}": float(np.percentile(d, 95)), f"p_upper_d{m}": float((1 + (d >= cg).sum()) / (len(d) + 1)),
                            f"pct_decoy_below_d{m}": float((d < cg).mean()), f"cog_minus_decoy_med_d{m}": cg - float(np.median(d))})
            drows.append(rec)
            store[f"{u.system}|{u.tf}|{arm}|dAUROC"] = np.array(dd["AUROC"])
        print(f"[increment] {u.system}/{u.tf}: arms={len(oofs)} pool={len(idx)} {time.time()-t0:.0f}s", flush=True)
    if a.mode == "loto":
        rows, drows, srows, store = [], [], [], {}
    write_tsv(rows, f"{out}/unit_increment_{tag_of(a)}.tsv")
    write_tsv(drows, f"{out}/unit_decoy_{tag_of(a)}.tsv")
    write_tsv(srows, f"{out}/unit_Dscale_{tag_of(a)}.tsv")
    np.savez_compressed(f"{out}/decoy_draws_{tag_of(a)}.npz", **store)
    # multi-TF systems: leave-one-TF-out x gene-fold (held-out TFs and held-out gene modules)
    lrows = []
    for sysn, lst in by_sys.items():
        if a.mode != "loto" or len(lst) < 3:
            continue
        for arm in ("raw", "obs_cov"):
            D_ = []
            for u, c in lst:
                Em = c["Em"]
                Pr = np.where(Em, u.P, np.nan)
                X0, X1 = arm_design(arm, Pr, c["S_M1"], u.O, None, None, Em, None)
                D_.append((u, c, X0, X1))
            for ti, (u, c, X0t, X1t) in enumerate(D_):
                Em = c["Em"]
                o0, o1 = np.full(u.n, np.nan), np.full(u.n, np.nan)
                ok = True
                for f in range(NFOLD):
                    te = Em & (u.folds == f)
                    trX0, trX1, trY = [], [], []
                    for tj, (v, cv, X0v, X1v) in enumerate(D_):
                        if tj == ti:
                            continue
                        tr = cv["Em"] & (v.folds != f)
                        trX0.append(X0v[tr]), trX1.append(X1v[tr]), trY.append(v.y[tr])
                    Yt = np.concatenate(trY)
                    if Yt.sum() < 5 or (~Yt).sum() < 5:
                        ok = False
                        break
                    from sklearn.linear_model import LogisticRegression
                    o0[te] = LogisticRegression(C=1.0, max_iter=5000).fit(np.vstack(trX0), Yt).decision_function(X0t[te])
                    o1[te] = LogisticRegression(C=1.0, max_iter=5000).fit(np.vstack(trX1), Yt).decision_function(X1t[te])
                if not ok:
                    continue
                p0, b0 = u.eval_boot(o0, Em, a.B)
                p1, b1 = u.eval_boot(o1, Em, a.B)
                rec = dict(**u.base(), arm=arm, scheme="leave-one-TF-out x gene-fold", n_tfs_train=len(D_) - 1)
                for m in ("AUROC", "APLIFT", "P50"):
                    d = b1[m] - b0[m]
                    lo, hi = boot_q(d)
                    rec.update({f"M0_{m}": p0[m], f"M1_{m}": p1[m], f"d{m}": p1[m] - p0[m], f"d{m}_lo": lo, f"d{m}_hi": hi, f"d{m}_p": boot_p(d)})
                lrows.append(rec)
            print(f"[increment] {sysn}/{arm}: leave-one-TF-out done", flush=True)
    write_tsv(lrows, f"{out}/unit_loto_{tag_of(a)}.tsv")
    receipt(a, out, "increment", dict(unit_shas=used_unit_shas([u.u for u in units]), models="spline(P) [+ beta*S~] logistic C=1, 5 gene-hash folds, transforms fit on E (label-free); arms raw / obs_cov / obs_resid; D-scale table"))


# (10) construction ablation
CONTRASTS = [("P_pairU", "P_pairM"), ("P_cellU", "P_cellM"), ("P_pairU", "P_cellU"), ("P_pairM", "P_cellM")]


def cmd_construct(a):
    units = [Unit(u) for u in load_units(a)]
    out = f"{a.out}/construct"
    rows, prow, draws = [], [], {}
    for u in units:
        com = u.E.copy()
        for k in ZKEYS:
            com &= np.isfinite(u.u["vec"][k])
        if int((com & u.y).sum()) < 5 or int((~u.y & com).sum()) < 5:
            continue
        res = {k: u.eval_boot(np.asarray(u.u["vec"][k], float), com, a.B) for k in ZKEYS}
        for k, (p, b) in res.items():
            lo, hi = boot_q(b["AUROC"])
            rows.append(dict(**u.base(), vector=k, n_common=int(com.sum()), **{m: p[m] for m in METS}, AUROC_lo=lo, AUROC_hi=hi, hits50=p["P50"] * 50))
        for x, yv in CONTRASTS:
            for m in ("AUROC", "APLIFT", "P50"):
                d = res[x][1][m] - res[yv][1][m]
                lo, hi = boot_q(d)
                prow.append(dict(**u.base(), a=x, b=yv, metric=m, diff=res[x][0][m] - res[yv][0][m], lo=lo, hi=hi, p_boot=boot_p(d)))
                draws[f"{u.system}|{u.tf}|{x}|{yv}|{m}"] = d.astype(np.float32)
        print(f"[construct] {u.system}/{u.tf} done", flush=True)
    write_tsv(rows, f"{out}/unit_vectors_{tag_of(a)}.tsv")
    write_tsv(prow, f"{out}/unit_contrasts_{tag_of(a)}.tsv")
    np.savez_compressed(f"{out}/draws_{tag_of(a)}.npz", **draws)
    receipt(a, out, "construct", dict(unit_shas=used_unit_shas([u.u for u in units]), note="identical Q & testable rows, identical frozen aggregation (leave-focal-out mean of T7 row-z), common mask over the four vectors"))


# (11) own vs every donor on the COMMON bank
DONORS = ["GATA1", "PANC", "BONEMARROW", "K562", "DENTATEGYRUS", "GASTRULATION"]


# P over the common genes from the common TF bank: both Z matrices are restricted to the common TF rows and common gene columns
def common_prior(own, don, own_key, don_key, focal_key, gene_sel_own, gene_sel_don):
    ko = {k: i for i, k in enumerate(own_key) if k is not None}
    kd = {k: i for i, k in enumerate(don_key) if k is not None}
    com = sorted((set(ko) & set(kd)) - {focal_key})
    ro, rd = [ko[k] for k in com], [kd[k] for k in com]
    Zo = rowz(own.Z["P_pairU"][np.ix_(ro, gene_sel_own)])
    Zd = rowz(don.Z["P_pairU"][np.ix_(rd, gene_sel_don)])
    ok = np.isfinite(Zo).any(1) & np.isfinite(Zd).any(1)                      # a row must be usable in both contexts (identical TF bank)
    return agg_mean(Zo[ok]), agg_mean(Zd[ok]), int(ok.sum())


def cmd_transfer(a):
    units = [Unit(u) for u in load_units(a)]
    out = f"{a.out}/transfer"
    keys = RO.keymap_factory()
    rows, prow, draws = [], [], {}
    dctx = {d: AxisCtx.get(d) for d in DONORS}
    for u in units:
        own = u.ctx
        okeys = keys(u.genes, own.mouse)
        own_tkey = own.keys
        for d, dc in dctx.items():
            if d == u.ax:
                continue
            base = dict(**u.base(), donor=d)
            # unique donor keys only
            kk = keys(dc.t7, dc.mouse)
            cnt = pd.Series([k for k in kk if k is not None]).value_counts()
            gk_d = {k: i for i, k in enumerate(kk) if k is not None and cnt[k] == 1}
            cols_u = [(i, own.t7col.get(g, -1), gk_d.get(k, -1)) for i, (g, k) in enumerate(zip(u.genes, okeys)) if k is not None and own.t7col.get(g, -1) >= 0]
            cols_u = [c for c in cols_u if c[2] >= 0 and u.E[c[0]]]
            if len(cols_u) < MIN_COMMON_ROWS:
                rows.append(dict(**base, status="NO_CALL", reason=f"common genes {len(cols_u)} < {MIN_COMMON_ROWS}"))
                continue
            gi_u = np.array([c[0] for c in cols_u])
            Po, Pd, ntf = common_prior(own, dc, own_tkey, dc.keys, own_tkey[u.r], [c[1] for c in cols_u], [c[2] for c in cols_u])
            if ntf < MIN_COMMON_TFS:
                rows.append(dict(**base, status="NO_CALL", reason=f"common TFs {ntf} < {MIN_COMMON_TFS}"))
                continue
            f = np.isfinite(Po) & np.isfinite(Pd)
            m = np.zeros(u.n, bool)
            m[gi_u[f]] = True
            so, sd_ = np.full(u.n, np.nan), np.full(u.n, np.nan)
            so[gi_u[f]], sd_[gi_u[f]] = Po[f], Pd[f]
            if int((m & u.y).sum()) < MIN_CLASS or int((m & ~u.y).sum()) < MIN_CLASS:
                rows.append(dict(**base, status="NO_CALL", reason=f"common rows resp {int((m&u.y).sum())} / non {int((m&~u.y).sum())} < {MIN_CLASS}"))
                continue
            po, bo = u.eval_boot(so, m, a.B)
            pd_, bd = u.eval_boot(sd_, m, a.B)
            rec = dict(**base, status="OK", n_common_rows=int(m.sum()), n_resp_common=int((m & u.y).sum()), n_common_tfs=ntf, spearman_own_donor=spear(so[m], sd_[m]))
            for met in ("AUROC", "APLIFT", "P50"):
                dif = bo[met] - bd[met]
                lo, hi = boot_q(dif)
                rec.update({f"own_{met}": po[met], f"donor_{met}": pd_[met], f"d{met}": po[met] - pd_[met], f"d{met}_lo": lo, f"d{met}_hi": hi, f"d{met}_p": boot_p(dif)})
                draws[f"{u.system}|{u.tf}|{d}|{met}"] = dif.astype(np.float32)
            rec["PREV"] = po["PREV"]
            rows.append(rec)
        print(f"[transfer] {u.system}/{u.tf} done", flush=True)
    write_tsv(rows, f"{out}/unit_transfer_{tag_of(a)}.tsv")
    np.savez_compressed(f"{out}/draws_{tag_of(a)}.npz", **draws)
    receipt(a, out, "transfer", dict(unit_shas=used_unit_shas([u.u for u in units]), donors=DONORS,
                                     note="own and donor priors both rebuilt on the COMMON TF bank (1:1 ortholog symbols for mouse axes) and COMMON genes (unique shared keys), rows re-z-scored on the common genes, focal excluded"))


# (5-6) indegree
def neighbor_counts(chrom, pos, win=5000):
    chrom, pos = np.asarray(chrom), np.asarray(pos, np.int64)
    out = np.zeros(len(pos), int)
    for c in np.unique(chrom):
        ii = np.flatnonzero(chrom == c)
        p = np.sort(pos[ii])
        lo, hi = np.searchsorted(p, pos[ii] - win, "left"), np.searchsorted(p, pos[ii] + win, "right")
        out[ii] = hi - lo - 1
    return out


# -> dict(genes_upper -> d, cov covariate (neighbour TSS), sources, coverage) or dict(status='NOT_AVAILABLE', reason)
def chip_indegree(ax, ctx):
    import gb_layer_lattice as L
    import gb_lattice_extra as LE
    LE.register()
    members = {m.upper() for m in ctx.members}
    if ax == "K562":
        tss, amb = L.k562_tss_from_gtf(L.K562_GTF)
        bl = pd.read_csv(L.K562_BLACKLIST, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"], comment="#")
        srcs, sets, files = [], {}, {}
        for p in sorted(glob.glob(f"{L.K562_CHIP_DIR}/*_ENC*.bed.gz")):
            tf = os.path.basename(p).split("_ENC")[0].upper()
            if tf not in members or tf in sets:
                continue
            pk = pd.read_csv(p, sep="\t", header=None, usecols=[0, 1, 2, 6, 9], names=["chrom", "start", "end", "sig", "off"])
            prim, _s, _st, _d, _k = L.k562_assign(pk, tss, bl)
            sets[tf], files[tf] = set(prim), dict(path=os.path.relpath(p, R), sha256=_sha(p))
        mappable = set(tss.sym)
        nb = dict(zip(tss.sym, neighbor_counts(tss.chrom.to_numpy(), tss.tss0.to_numpy())))
        src = "ENCODE K562 TF ChIP-seq (data/encode/k562_v2), assignment rule: unique-nearest MANE/longest TSS (GENCODE v44) within +-2kb of a valid summit, blacklist-filtered"
    elif ax == "BONEMARROW":
        tss, dup = L.bm_tss()
        sets, files = {}, {}
        J = json.load(open(LE.CA_SEL))
        for tf in sorted(J["selection"]):
            if tf.upper() not in members or tf in J.get("excluded_tfs", {}):
                continue
            try:
                sets[tf.upper()] = set(LE.chipatlas_occ(tf, tss)["prim"])
                files[tf.upper()] = dict(selection=os.path.relpath(LE.CA_SEL, R), tier=J["selection"][tf]["tier"], n_experiments=len(J["selection"][tf]["srx"]))
            except Exception as e:
                files[tf.upper()] = dict(error=str(e)[:200])
        if "FLI1" in members and "FLI1" not in sets:                      # MK ChIP (GSE24674) for the TF with no ChIP-Atlas selection
            pk, pkf = L.peak_mids("FLI1")
            if pk is not None:
                sets["FLI1"] = set(LE.occ_from_peaks(pk, tss)["prim"])
                files["FLI1"] = dict(path=os.path.relpath(pkf, R), sha256=_sha(pkf))
        mappable = set(tss) - set(dup)
        ch = np.array([tss[s][0] for s in tss])
        ps = np.array([tss[s][1] for s in tss])
        nb = dict(zip(list(tss), neighbor_counts(ch, ps)))
        src = "ChIP-Atlas hg38 bed05 (selection.json rule R1, TSS+-2kb consensus) + MK ChIP GSE24674 for FLI1; GENCODE v32 gene-level TSS"
    else:
        return dict(status="NOT_AVAILABLE", reason=f"no catalogue-scale peak sets on disk for {ax} (mouse ChIP on disk: GATA1 (2 ENCODE beds), EKLF, Rfx6 only)")
    sets = {k: v for k, v in sets.items() if v}
    cov = dict(n_tfs_with_peaks=len(sets), n_catalogue_tfs=len(members), frac=len(sets) / max(len(members), 1), tfs=sorted(sets), source=src, files=files)
    if len(sets) < CHIP_FLOOR:
        return dict(status="NOT_AVAILABLE", reason=f"only {len(sets)} catalogue TFs with peaks < floor {CHIP_FLOOR}", coverage=cov)
    return dict(status="OK", sets=sets, mappable=mappable, neighbors=nb, coverage=cov)


def regulon_indegree(ax, ctx):
    from gb_fig1_eligibility import regulons
    sp = AXIS_SPECIES[ax]
    reg = regulons(sp)[0]
    members = {m.upper() for m in ctx.members}
    d_cat, d_all = {}, {}
    n_src_cat = 0
    for s, tg in reg.items():
        for t in tg:
            d_all[t] = d_all.get(t, 0) + 1
            if s in members:
                d_cat[t] = d_cat.get(t, 0) + 1
        n_src_cat += s in members
    cov = dict(species=sp, n_sources_all=len(reg), n_sources_in_bank=int(n_src_cat), n_catalogue_tfs=len(members), source="CollecTRI (data/collectri), human; mouse via strict 1:1 orthologs (gb_fig1_eligibility.regulons)",
               collectri_sha256=_sha(f"{R}/data/collectri/collectri_9606.tsv"))
    return d_cat, d_all, cov


# cross-fitted OLS (intercept + label-free standardised X) on the fixed folds -> held-out fit (NaN outside mask)
def ols_cf(v, X, folds, mask):
    v, X, folds, mask = np.asarray(v, float), np.asarray(X, float), np.asarray(folds), np.asarray(mask, bool)
    mu, sd = X[mask].mean(0), X[mask].std(0)
    A_ = np.column_stack([np.ones(len(v)), (X - mu) / np.where(sd > 0, sd, 1.0)])
    fit = np.full(len(v), np.nan)
    for f in np.unique(folds[mask]):
        te, tr = mask & (folds == f), mask & (folds != f)
        fit[te] = A_[te] @ np.linalg.lstsq(A_[tr], v[tr], rcond=None)[0]
    return fit


# study-bias proxy: number of GO Biological Process 2023 terms (Enrichr library on disk) that contain the gene (upper-case symbols)
def go_counts():
    cnt = {}
    for ln in open(GO_FILE):
        for g in {x.upper() for x in ln.rstrip("\n").split("\t")[2:] if x}:
            cnt[g] = cnt.get(g, 0) + 1
    return cnt


def cmd_indegree(a):
    axes = sorted(set(a.axes.split(",")))
    out = f"{a.out}/indegree"
    rows = []
    GO = go_counts()
    for ax in axes:
        ctx = AxisCtx.get(ax)
        Pfull = ctx.p_loo("P_pairU", None)
        genes = np.array(ctx.t7)
        gu = np.char.upper(genes)
        F = ctx.F.reindex(genes)
        O9, OBS, DYN = F[REF_COLS].to_numpy(float), F[SPLIT_OBS].to_numpy(float), F[SPLIT_DYN].to_numpy(float)
        study = np.log1p(np.array([GO.get(g, 0) for g in gu], float))
        folds = gene_folds(genes)
        srcs = []
        cd, da, rcov = regulon_indegree(ax, ctx)
        srcs.append(("regulon_bank", np.array([cd.get(g, 0) for g in gu], float), rcov, None))
        srcs.append(("regulon_all", np.array([da.get(g, 0) for g in gu], float), rcov, None))
        ci = chip_indegree(ax, ctx)
        if ci["status"] != "OK":
            rows.append(dict(axis=ax, source="chip", arm="ALL", status=ci["status"], reason=ci["reason"], coverage=json.dumps(ci.get("coverage", {}), default=str)))
        else:
            d = np.zeros(len(genes))
            mapp = np.array([g in ci["mappable"] for g in gu])
            for tf, s_ in ci["sets"].items():
                d += np.array([g in s_ for g in gu], float)
            nbc = np.array([np.log1p(ci["neighbors"].get(g, np.nan)) if g in ci["neighbors"] else np.nan for g in gu])
            srcs.append(("chip", np.where(mapp, d, np.nan), ci["coverage"], nbc))
        for name, d, cov, extra in srcs:
            # (arm, fitter, nuisance matrix); 'raw' = no adjustment
            arms = [("raw", None, None), ("lin_O9", "lin", O9), ("hgb_O9", "hgb", O9), ("lin_obs5", "lin", OBS), ("hgb_obs5", "hgb", OBS), ("lin_dyn4", "lin", DYN), ("hgb_dyn4", "hgb", DYN),
                    ("lin_O9+study", "lin", np.column_stack([O9, study])), ("hgb_O9+study", "hgb", np.column_stack([O9, study]))]
            if extra is not None:
                arms += [("hgb_O9+coverage", "hgb", np.column_stack([O9, extra])), ("hgb_O9+coverage+study", "hgb", np.column_stack([O9, extra, study]))]
            ld = np.log1p(d)
            for armn, kind, X in arms:
                m = np.isfinite(Pfull) & np.isfinite(O9).all(1) & np.isfinite(d) & (np.isfinite(X).all(1) if X is not None else True)
                if m.sum() < 200 or np.ptp(d[m]) == 0:
                    rows.append(dict(axis=ax, source=name, arm=armn, status="NO_CALL", reason=f"n={int(m.sum())} or constant indegree", coverage=json.dumps(cov, default=str)))
                    continue
                if kind is None:
                    fP = fd = np.zeros(len(Pfull))
                    rP, rd_ = np.where(m, Pfull, np.nan), np.where(m, ld, np.nan)
                else:
                    fit = hgb_cf if kind == "hgb" else ols_cf
                    fP, fd = fit(Pfull, X, folds, m), fit(ld, X, folds, m)
                    rP, rd_ = np.where(m, Pfull - fP, np.nan), np.where(m, ld - fd, np.nan)
                idx = np.flatnonzero(m)
                rg = np.random.default_rng(seed_of("indeg", ax, name, armn))
                bs = np.array([spear(rP[b], rd_[b]) for b in (idx[rg.integers(0, len(idx), len(idx))] for _ in range(a.B))])
                lo, hi = boot_q(bs)
                rec = dict(axis=ax, source=name, arm=armn, status="OK", kind=kind or "none", n_genes=int(m.sum()), n_sources=cov.get("n_tfs_with_peaks", cov.get("n_sources_in_bank")),
                           frac_zero_indegree=float((d[m] == 0).mean()), rho_raw=spear(Pfull[m], ld[m]), rho_resid=spear(rP, rd_), rho_lo=lo, rho_hi=hi, p_boot=boot_p(bs), B=a.B)
                if kind is not None:
                    r2 = lambda v, f_: float(1 - ((v[m] - f_[m]) ** 2).sum() / ((v[m] - v[m].mean()) ** 2).sum())
                    rec.update(R2_nuis_on_P=r2(Pfull, fP), R2_nuis_on_logd=r2(ld, fd), rho_fits=spear(fP[m], fd[m]))
                if armn == "hgb_O9":                                      # refit-inside-bootstrap CI (the cross-fit is redone on every gene resample; duplicates share a fold by hash)
                    rb = []
                    for _ in range(B_REFIT):
                        b = idx[rg.integers(0, len(idx), len(idx))]
                        allm = np.ones(len(b), bool)
                        rb.append(spear(Pfull[b] - hgb_cf(Pfull[b], O9[b], folds[b], allm), ld[b] - hgb_cf(ld[b], O9[b], folds[b], allm)))
                    rec.update(rho_lo_refit=boot_q(rb)[0], rho_hi_refit=boot_q(rb)[1], B_refit=B_REFIT)
                rec["coverage"] = json.dumps({k: v for k, v in cov.items() if k != "files"}, default=str)
                rows.append(rec)
                print(f"[indegree] {ax}/{name}/{armn}: n={int(m.sum())} rho_resid={rec['rho_resid']:.3f} [{lo:.3f},{hi:.3f}]", flush=True)
    write_tsv(rows, f"{out}/axis_indegree_{tag_of(a)}.tsv")
    receipt(a, out, "indegree", dict(note="P = full-bank prior Pfull (mean T7 row-z, T7 genes); nuisance sets: O9 = nine REF columns, obs5 = raw_det/sm_det/abundance/pair_support/mu_det (confounding by detection), dyn4 = dynamics/mu_dynamics/"
                                          "abs_spearman_mu_dpt/mu_logvar (mediation via dynamics); linear = cross-fit OLS, hgb = frozen HGB; study = log1p GO-BP term count; d = count of catalogue-bank TFs with an assigned peak / regulon edge; "
                                          "chip only for human contexts; CI = gene bootstrap conditional on the cross-fit (+ refit bootstrap for hgb_O9)"))


# summarize
def tf_boot(d, B=4000, qs=(2.5, 97.5), seed=0):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return [np.nan] * len(qs)
    rg = np.random.default_rng(seed)
    return [float(x) for x in np.percentile(d[rg.integers(0, len(d), (B, len(d)))].mean(1), qs)]


# TF-equal-weight aggregate of unit differences: mean, CI (TF bootstrap if >= 2 units else the unit's gene-bootstrap), p
def agg_block(g, val, lo, hi, p_unit, draws=None, key=None, m_family=1):
    import gb_audit_decomp as AD
    d = g[val].to_numpy(float)
    d = d[np.isfinite(d)]
    n = len(d)
    qa = 100 * 0.025 / max(m_family, 1)
    qs = (qa, 100 - qa)
    if n >= 2:
        lo_, hi_ = tf_boot(d)
        clo, chi = tf_boot(d, qs=qs)
        p, ct = AD.signflip(d)[0], "TF bootstrap / exact sign-flip over TFs"
    elif n == 1:
        lo_, hi_ = float(g[lo].iloc[0]), float(g[hi].iloc[0])
        dr = draws.get(key) if draws is not None else None
        clo, chi = (float(np.percentile(dr, qs[0])), float(np.percentile(dr, qs[1]))) if dr is not None and len(dr) > 20 else (np.nan, np.nan)
        p, ct = float(g[p_unit].iloc[0]), "gene bootstrap (single unit, conditional)"
    else:
        return None
    return dict(n_units=n, mean_diff=float(d.mean()), lo=lo_, hi=hi_, corr_lo=clo, corr_hi=chi, p=p, ci_type=ct, frac_pos=float((d > 0).mean()))


def verdict(row, metric):
    dl = DELTA[metric]
    if not np.isfinite(row.get("corr_lo", np.nan)):
        return "NA"
    sig = row.get("p_holm", 0.0) < 0.05                                   # Holm-significant and the Bonferroni-corrected interval clears 0 (the exact sign-flip floor 2/2^n can make Holm unreachable for small n)
    if row["mean_diff"] >= dl and row["corr_lo"] > 0 and sig:
        return "SUPERIOR"
    if row["mean_diff"] <= -dl and row["corr_hi"] < 0 and sig:
        return "INFERIOR"
    if row["corr_lo"] > -dl and row["corr_hi"] < dl:
        return "EQUIVALENT"
    return "INCONCLUSIVE"


def holm_col(df, pcol, by=None):
    import gb_attribution as A
    out = np.full(len(df), np.nan)
    groups = [df.index] if by is None else [g.index for _, g in df.groupby(by)]
    for ix in groups:
        out[df.index.get_indexer(ix)] = A.holm(df.loc[ix, pcol].to_numpy(float))
    return out


# Holm floor: family size m x the smallest p the test can attain
def floor_cols(df, by, B, single_floor=None):
    n = df["n_units"].to_numpy(float) if "n_units" in df else np.ones(len(df))
    fl = np.where(n >= 2, np.where(n <= 20, 2.0 / 2.0 ** np.minimum(n, 20), 2.0 / (2e5 + 1)), 2.0 / (B + 1))
    if single_floor is not None:
        fl = np.where(n < 2, np.asarray(single_floor, float), fl)
    df["floor_p"] = fl
    ok = np.isfinite(df["p"] if "p" in df else df["p_signflip"])
    key = df[by] if isinstance(by, list) else df[[by]]
    df["family_m"] = ok.groupby([key[c] for c in key.columns]).transform("sum").to_numpy()
    df["holm_floor"] = df["family_m"] * df["floor_p"]
    df["testable"] = df["holm_floor"] < 0.05
    return df


def load_part(out, part, name):
    fs = [f for f in sorted(glob.glob(f"{out}/{part}/{name}_*.tsv")) if os.path.getsize(f) > 2]
    return pd.concat([pd.read_csv(f, sep="\t") for f in fs], ignore_index=True) if fs else pd.DataFrame()


def load_draws(out, part):
    d = {}
    for f in sorted(glob.glob(f"{out}/{part}/draws_*.npz")):
        z = np.load(f)
        d.update({k: z[k] for k in z.files})
    return d


def cmd_summarize(a):
    import gb_audit_decomp as AD
    out = a.out
    sm = f"{out}/summary"
    os.makedirs(sm, exist_ok=True)
    rep = {}
    # robust
    ax = load_part(out, "robust", "axis_variants")
    if len(ax):
        axv = ax[ax.vclass.isin(["agg", "sub", "fam"])].dropna(subset=["spearman"])
        fs = ax[ax.vclass == "famsplit_AvsB"].dropna(subset=["spearman"])
        rs = ax[ax.vclass == "randsplit_AvsB"].dropna(subset=["spearman"])
        G = []
        for nm, d in (("variants_pooled(agg+sub+fam)", axv), ("agg_only", axv[axv.vclass == "agg"]), ("sub_only", axv[axv.vclass == "sub"]), ("fam_only", axv[axv.vclass == "fam"]),
                      ("famdisjoint_AvsB", fs), ("random_splithalf_AvsB(noise floor)", rs)):
            if len(d):
                G.append(dict(set=nm, n_instances=len(d), n_axes=d.axis.nunique(), median=d.spearman.median(), p05=d.spearman.quantile(0.05), min=d.spearman.min(),
                              top50_median=d.top50.median(), gate_pass=bool(d.spearman.median() >= GATE_MED and d.spearman.quantile(0.05) >= GATE_P5)))
        for ax_, d in axv.groupby("axis"):
            G.append(dict(set=f"variants_pooled | axis={ax_}", n_instances=len(d), n_axes=1, median=d.spearman.median(), p05=d.spearman.quantile(0.05), min=d.spearman.min(), top50_median=d.top50.median(),
                          gate_pass=bool(d.spearman.median() >= GATE_MED and d.spearman.quantile(0.05) >= GATE_P5)))
        for ax_, d in fs.groupby("axis"):
            G.append(dict(set=f"famdisjoint_AvsB | axis={ax_}", n_instances=len(d), n_axes=1, median=d.spearman.median(), p05=d.spearman.quantile(0.05), min=d.spearman.min(), top50_median=d.top50.median(),
                          gate_pass=bool(d.spearman.median() >= GATE_MED and d.spearman.quantile(0.05) >= GATE_P5)))
        write_tsv(G, f"{sm}/robust_gate_axis.tsv")
        write_tsv(ax.sort_values(["axis", "variant"]).to_dict("records"), f"{sm}/robust_axis_all.tsv")
    uv = load_part(out, "robust", "unit_variants")
    if len(uv):
        v = uv[uv.vclass != "primary"]
        G = []
        for nm, d in (("unit_variants_pooled", v[v.vclass.isin(["agg", "sub", "fam"])]), ("famsplit_halves(P_A,P_B vs primary)", v[v.vclass == "famsplit"]), ("randsplit_halves", v[v.vclass == "randsplit"])):
            d = d.dropna(subset=["spearman"])
            if len(d):
                G.append(dict(set=nm, n_instances=len(d), n_units=d.groupby(["system", "tf"]).ngroups, median=d.spearman.median(), p05=d.spearman.quantile(0.05), top50_median=d.top50.median(),
                              gate_pass=bool(d.spearman.median() >= GATE_MED and d.spearman.quantile(0.05) >= GATE_P5)))
        write_tsv(G, f"{sm}/robust_gate_unit.tsv")
        v2 = uv.copy()
        v2["dAUROC"], v2["dAPLIFT"], v2["dP50"] = v2.AUROC - v2.prim_AUROC, v2.APLIFT - v2.prim_APLIFT, v2.P50 - v2.prim_P50
        cls = v2[v2.vclass != "primary"].groupby(["system", "vclass"]).agg(n_variants=("variant", "nunique"), mean_dAUROC=("dAUROC", "mean"), min_dAUROC=("dAUROC", "min"), max_dAUROC=("dAUROC", "max"),
                                                                           mean_dAPLIFT=("dAPLIFT", "mean"), min_dAPLIFT=("dAPLIFT", "min"), max_dAPLIFT=("dAPLIFT", "max")).reset_index()
        cls.to_csv(f"{sm}/robust_endpoint_change_by_class.tsv", sep="\t", index=False)
        prim = uv[uv.variant == "mean"][["system", "block", "axis", "tf", "n_E", "n_resp", "gated"] + list(METS)]
        prim.to_csv(f"{sm}/primary_unit_metrics.tsv", sep="\t", index=False)
        ub = load_part(out, "robust", "unit_boot")
        if len(ub):
            rb = []
            for (sysn, var, met), g in ub.groupby(["system", "variant", "metric"]):
                r_ = agg_block(g, "diff", "lo", "hi", "p_boot")
                if r_:
                    rb.append(dict(system=sysn, variant=var, metric=met, **r_))
            RB = pd.DataFrame(rb)
            RB["p_holm"] = holm_col(RB, "p", by="metric")
            floor_cols(RB, "metric", a.B)
            RB.to_csv(f"{sm}/robust_alt_aggregators.tsv", sep="\t", index=False)
    # breadth
    br = load_part(out, "breadth", "unit_breadth")
    if len(br):
        rows = []
        for (sysn, vec), g in br.groupby(["system", "vector"]):
            r_ = dict(system=sysn, vector=vec, n_units=len(g), mean_AUROC=g.AUROC.mean(), mean_APLIFT=g.APLIFT.mean(), mean_P50=g.P50.mean(), mean_spearman_with_P=g.spearman_with_P.mean(),
                      mean_spearman_breadth_intensity=g.spearman_breadth_intensity.mean())
            if vec != "P":
                q = agg_block(g, "dAUROC_vs_P", "dlo", "dhi", "p_boot")
                if q:
                    r_.update({f"dAUROC_vs_P_{k}": v for k, v in q.items() if k in ("mean_diff", "lo", "hi", "p")})
            rows.append(r_)
        BR = pd.DataFrame(rows)
        BR.to_csv(f"{sm}/breadth_system.tsv", sep="\t", index=False)
    # increment
    inc = load_part(out, "increment", "unit_increment")
    if len(inc):
        ok = inc[inc.status == "OK"]
        rows = []
        for (sysn, arm), g in ok.groupby(["system", "arm"]):
            for m in ("AUROC", "APLIFT", "P50"):
                r_ = agg_block(g, f"d{m}", f"d{m}_lo", f"d{m}_hi", f"d{m}_p")
                if r_:
                    rows.append(dict(system=sysn, arm=arm, metric=m, mean_M0=g[f"M0_{m}"].mean(), mean_M1=g[f"M1_{m}"].mean(), **r_))
        IS = pd.DataFrame(rows)
        IS["p_holm"] = holm_col(IS, "p", by=["arm", "metric"])
        floor_cols(IS, ["arm", "metric"], a.B)
        IS.to_csv(f"{sm}/increment_system.tsv", sep="\t", index=False)
        inc[inc.status != "OK"].to_csv(f"{sm}/increment_no_call.tsv", sep="\t", index=False)
    dec = load_part(out, "increment", "unit_decoy")
    if len(dec):
        d_ok = dec[dec.status == "OK"]
        rows = []
        for (sysn, arm), g in d_ok.groupby(["system", "arm"]):
            for m in ("AUROC", "APLIFT"):
                diff = g[f"cog_minus_decoy_med_d{m}"].to_numpy(float)
                rr = dict(system=sysn, arm=arm, metric=m, n_units=len(g), n_decoy_min=int(g.n_decoy.min()), mean_cognate_inc=g[f"cog_d{m}"].mean(), mean_decoy_median_inc=g[f"decoy_med_d{m}"].mean(),
                          mean_cog_minus_decoy_med=float(np.mean(diff)), frac_p_le_05=float((g[f"p_upper_d{m}"] <= 0.05).mean()), median_p_upper=float(g[f"p_upper_d{m}"].median()))
                if len(diff) >= 2:
                    rr["p_signflip"], rr["lo"], rr["hi"] = AD.signflip(diff)[0], *tf_boot(diff)
                else:
                    rr["p_signflip"], rr["lo"], rr["hi"] = float(g[f"p_upper_d{m}"].iloc[0]), np.nan, np.nan
                rows.append(rr)
        DS = pd.DataFrame(rows)
        DS["p"] = DS["p_signflip"]
        DS["p_holm"] = holm_col(DS, "p_signflip", by=["arm", "metric"])
        floor_cols(DS, ["arm", "metric"], a.B, single_floor=1.0 / (DS["n_decoy_min"] + 1.0))
        DS.to_csv(f"{sm}/decoy_system.tsv", sep="\t", index=False)
        dec[dec.status != "OK"].to_csv(f"{sm}/decoy_no_call.tsv", sep="\t", index=False)
    ds = load_part(out, "increment", "unit_Dscale")
    if len(ds):
        ds.groupby(["system", "vector"])[["AUROC", "APLIFT", "P50", "PREV"]].mean().reset_index().to_csv(f"{sm}/Dscale_system_means.tsv", sep="\t", index=False)
    lo_ = load_part(out, "increment", "unit_loto")
    if len(lo_):
        rows = []
        for (sysn, arm), g in lo_.groupby(["system", "arm"]):
            for m in ("AUROC", "APLIFT", "P50"):
                r_ = agg_block(g, f"d{m}", f"d{m}_lo", f"d{m}_hi", f"d{m}_p")
                if r_:
                    rows.append(dict(system=sysn, arm=arm, metric=m, scheme="leave-one-TF-out x gene-fold", **r_))
        LS = pd.DataFrame(rows)
        LS["p_holm"] = holm_col(LS, "p", by=["arm", "metric"])
        floor_cols(LS, ["arm", "metric"], a.B)
        LS.to_csv(f"{sm}/increment_loto_system.tsv", sep="\t", index=False)
    # construct
    cu = load_part(out, "construct", "unit_contrasts")
    if len(cu):
        dr = load_draws(out, "construct")
        rows = []
        for (sysn, x, yv, m), g in cu.groupby(["system", "a", "b", "metric"]):
            key = f"{sysn}|{g.tf.iloc[0]}|{x}|{yv}|{m}"
            rows.append(dict(system=sysn, a=x, b=yv, metric=m, _g=g, _key=key))
        fam_m = {m: sum(1 for r in rows if r["metric"] == m) for m in ("AUROC", "APLIFT", "P50")}
        res = []
        for r in rows:
            q = agg_block(r["_g"], "diff", "lo", "hi", "p_boot", dr, r["_key"], fam_m[r["metric"]])
            if q:
                res.append(dict(system=r["system"], a=r["a"], b=r["b"], metric=r["metric"], **q))
        CS = pd.DataFrame(res)
        CS["p_holm"] = holm_col(CS, "p", by="metric")
        floor_cols(CS, "metric", a.B)
        CS["verdict"] = [verdict(r, r["metric"]) for r in CS.to_dict("records")]
        CS.loc[(~CS.testable) & (CS.verdict != "EQUIVALENT"), "verdict"] = "UNTESTABLE"
        CS.to_csv(f"{sm}/construct_system.tsv", sep="\t", index=False)
        cv = load_part(out, "construct", "unit_vectors")
        cv.groupby(["system", "vector"])[list(METS) + ["hits50"]].mean().reset_index().to_csv(f"{sm}/construct_vector_means.tsv", sep="\t", index=False)
    # transfer
    tr = load_part(out, "transfer", "unit_transfer")
    if len(tr):
        dr = load_draws(out, "transfer")
        ok = tr[tr.status == "OK"]
        rows = []
        fam_m = {m: ok.groupby(["system", "donor"]).ngroups for m in ("AUROC", "APLIFT", "P50")}
        for (sysn, d), g in ok.groupby(["system", "donor"]):
            for m in ("AUROC", "APLIFT", "P50"):
                q = agg_block(g, f"d{m}", f"d{m}_lo", f"d{m}_hi", f"d{m}_p", dr, f"{sysn}|{g.tf.iloc[0]}|{d}|{m}", fam_m[m])
                if q:
                    rows.append(dict(system=sysn, donor=d, metric=m, mean_own=g[f"own_{m}"].mean(), mean_donor=g[f"donor_{m}"].mean(), mean_common_tfs=g.n_common_tfs.mean(), **q))
        TS = pd.DataFrame(rows)
        TS["p_holm"] = holm_col(TS, "p", by="metric")
        floor_cols(TS, "metric", a.B)
        TS["verdict"] = [verdict(r, r["metric"]) for r in TS.to_dict("records")]
        TS.loc[(~TS.testable) & (TS.verdict != "EQUIVALENT"), "verdict"] = "UNTESTABLE"
        TS.to_csv(f"{sm}/transfer_system.tsv", sep="\t", index=False)
        own_all = []
        for (sysn, tf), g in ok.groupby(["system", "tf"]):
            own_all.append(dict(system=sysn, tf=tf, n_donors=len(g), own_beats_all=bool((g.dAUROC > 0).all()), n_donors_own_wins=int((g.dAUROC > 0).sum()), mean_dAUROC=float(g.dAUROC.mean())))
        pd.DataFrame(own_all).to_csv(f"{sm}/transfer_own_vs_all.tsv", sep="\t", index=False)
        tr[tr.status != "OK"].to_csv(f"{sm}/transfer_no_call.tsv", sep="\t", index=False)
    # indegree
    ind = load_part(out, "indegree", "axis_indegree")
    if len(ind):
        import gb_attribution as A_
        ok = ind[(ind.status == "OK") & (ind.arm != "raw")].reset_index(drop=True)          # 'raw' has no adjustment: reported, not part of the adjusted-association families
        ok["p_holm_allarms"] = A_.holm(ok.p_boot.to_numpy(float))
        pm = (ok.arm == "hgb_O9").to_numpy()
        ok["p_holm_primary"] = np.nan
        ok.loc[pm, "p_holm_primary"] = A_.holm(ok.loc[pm, "p_boot"].to_numpy(float))
        ok["floor_p"] = 2.0 / (ok.B + 1.0)
        ok["holm_floor_primary"], ok["holm_floor_allarms"] = pm.sum() * ok.floor_p, len(ok) * ok.floor_p
        ok["testable_primary"], ok["testable_allarms"] = ok.holm_floor_primary < 0.05, ok.holm_floor_allarms < 0.05
        ok = pd.concat([ok, ind[(ind.status == "OK") & (ind.arm == "raw")]], ignore_index=True)
        ind = pd.concat([ok, ind[ind.status != "OK"]], ignore_index=True)
        ind.to_csv(f"{sm}/indegree_axis.tsv", sep="\t", index=False)
        wide = ok.pivot_table(index=["axis", "source"], columns="arm", values="rho_resid")
        wide["raw"] = ok[ok.arm == "raw"].set_index(["axis", "source"]).rho_resid
        wide.insert(0, "n_genes", ok.groupby(["axis", "source"]).n_genes.first())
        wide.to_csv(f"{sm}/indegree_side_by_side_rho.tsv", sep="\t")
    assert _sha(__file__) == SCRIPT_SHA0
    json.dump(dict(script_sha256=SCRIPT_SHA0, argv=sys.argv, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), DELTA=DELTA,
                   tables=sorted(os.listdir(sm)), table_shas={f: _sha(f"{sm}/{f}") for f in sorted(os.listdir(sm)) if f.endswith(".tsv")}), open(f"{sm}/receipt_summarize.json", "w"), indent=1)
    print(f"[summarize] -> {sm}: {sorted(os.listdir(sm))}")


# selftest
def selftest():
    from scipy.stats import trim_mean
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(60, 40))
    Z[rng.random(Z.shape) < 0.2] = np.nan
    Z[:, 3] = np.nan
    Z[:1, 5] = 1.0
    Z[1:, 5] = np.nan                                              # a single contributor -> NaN
    f = np.isfinite(Z)
    tm = agg_trim(Z, 0.10)
    for j in range(40):
        c = f[:, j].sum()
        if c >= 2:
            assert abs(tm[j] - trim_mean(Z[f[:, j], j], 0.10)) < 1e-12, j
        else:
            assert np.isnan(tm[j])
    assert np.isnan(agg_mean(Z)[3]) and np.isnan(agg_mean(Z)[5]) and np.isnan(agg_median(Z)[5])
    assert np.allclose(agg_mean(Z)[0], np.nanmean(Z[:, 0])) and np.allclose(agg_median(Z)[1], np.nanmedian(Z[:, 1]))
    pc = row_pct(Z)
    assert np.nanmin(pc) > 0 and np.nanmax(pc) < 1 and np.allclose(np.nanmean(pc, axis=1), 0.5)       # average-rank percentiles are centred
    rz = rowz(Z)
    assert np.allclose(np.nanmean(rz, 1), 0, atol=1e-12) and np.allclose(np.nanstd(rz, 1), 1)
    # leave-one-out identity: mean over rows != r equals (S - Z_r)/(C - f_r)
    S, C = np.where(f, Z, 0.0).sum(0), f.sum(0)
    keep = np.ones(60, bool)
    keep[7] = False
    lo_ = agg_mean(Z[keep])
    with np.errstate(invalid="ignore", divide="ignore"):
        ident = np.where(C - f[7] >= 2, (S - np.where(f[7], Z[7], 0.0)) / np.maximum(C - f[7], 1), np.nan)
    assert np.allclose(lo_, ident, equal_nan=True)
    # spearman / topk
    a_ = rng.normal(size=300)
    assert abs(spear(a_, a_ ** 3) - 1) < 1e-12 and abs(topk_overlap(a_, a_ ** 3, 50) - 1) < 1e-12
    # gene folds deterministic and balanced
    fo = gene_folds([f"G{i}" for i in range(2000)])
    assert set(fo) == set(range(NFOLD)) and fo.min() == 0 and np.all(fo == gene_folds([f"g{i}" for i in range(2000)]))
    # pt_metrics = rank AUROC
    y = rng.random(500) < 0.2
    s = rng.normal(size=500) + 1.5 * y
    m = pt_metrics(s, y, np.ones(500, bool))
    import gb_audit_decomp as AD
    assert abs(m["AUROC"] - AD.auroc_rows(s[None, :], y)[0]) < 1e-12 and m["APLIFT"] > 1.5 and abs(m["PREV"] - y.mean()) < 1e-12
    # oof logit recovers signal
    n = 1500
    msk = np.ones(n, bool)
    fo = gene_folds([f"X{i}" for i in range(n)])
    Pn = rng.normal(size=n)
    Ssig = rng.normal(size=n)
    yy = (Ssig + rng.normal(size=n)) > 1.0
    X0 = np.hstack([spline_basis(Pn, msk)])
    X1 = np.hstack([spline_basis(Pn, msk), lin_block(Ssig, msk)])
    o0, o1 = oof_logit(X0, yy, fo, msk), oof_logit(X1, yy, fo, msk)
    a0, a1 = pt_metrics(o0, yy, msk)["AUROC"], pt_metrics(o1, yy, msk)["AUROC"]
    assert abs(a0 - 0.5) < 0.06 and a1 > 0.70, (a0, a1)
    Sjunk = rng.normal(size=n)
    o2 = oof_logit(np.hstack([spline_basis(Pn, msk), lin_block(Sjunk, msk)]), yy, fo, msk)
    assert abs(pt_metrics(o2, yy, msk)["AUROC"] - a0) < 0.06
    # hgb cross-fit removes a nonlinear function of O and is label-free
    O_ = rng.normal(size=(n, 3))
    v = np.sin(2 * O_[:, 0]) + 0.1 * rng.normal(size=n)
    fit = hgb_cf(v, O_, fo, msk)
    assert np.corrcoef(fit, v)[0, 1] > 0.9
    # common_prior: identical Z in 'own' and 'donor' -> identical priors
    class _C:
        pass
    own, don = _C(), _C()
    own.Z = {"P_pairU": rng.normal(size=(40, 120))}
    don.Z = {"P_pairU": own.Z["P_pairU"] * 3.0 + 1.0}                  # affine -> identical after row re-z
    keys_ = [f"T{i}" for i in range(40)]
    Po, Pd, nt = common_prior(own, don, keys_, keys_, "T3", np.arange(100), np.arange(100))
    assert nt == 39 and np.allclose(Po, Pd)
    # neighbour counts
    assert list(neighbor_counts(["c"] * 4, [0, 1000, 20000, 20500], 5000)) == [1, 1, 1, 1]
    # axis machinery on the synthetic path: family splits are disjoint
    print("gb_prior_semantics selftest PASS")


# CLI
def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    for nm in ("robust", "breadth", "increment", "construct", "transfer", "indegree", "summarize"):
        p = sp.add_parser(nm)
        p.add_argument("--out", required=True)
        p.add_argument("--B", type=int, default=500)
        p.add_argument("--tag", default="")
        if nm == "indegree":
            p.add_argument("--axes", default="GATA1,PANC,BONEMARROW,K562,DENTATEGYRUS,GASTRULATION")
        elif nm != "summarize":
            p.add_argument("--systems", default="")
            p.add_argument("--tfs", default="")
            p.add_argument("--max-units", dest="max_units", type=int, default=0)
            p.add_argument("--slice", type=int, default=0)
            p.add_argument("--nslice", type=int, default=1)
            p.add_argument("--mode", default="main", choices=["main", "loto"])
    a = ap.parse_args()
    if a.cmd != "summarize" and a.cmd != "selftest":
        os.makedirs(a.out, exist_ok=True)
    {"selftest": lambda: selftest(), "robust": lambda: cmd_robust(a), "breadth": lambda: cmd_breadth(a), "increment": lambda: cmd_increment(a), "construct": lambda: cmd_construct(a),
     "transfer": lambda: cmd_transfer(a), "indegree": lambda: cmd_indegree(a), "summarize": lambda: cmd_summarize(a)}[a.cmd]()


if __name__ == "__main__":
    main()
