#!/usr/bin/env python
# Evaluate target priors and perturbation-response breadth.
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse
import glob
import hashlib
import itertools
import json
import multiprocessing as mp
import sys
import tempfile
import time
import warnings
import zlib
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer, StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_composites as GC  # noqa: E402

warnings.filterwarnings("ignore", category=ConvergenceWarning)

# Loader interfaces for gb_catbank.py and gb_breadth_truth.py.
R_ = __import__("os").environ["GEOBRIDGE_ROOT"]
OUT = f"{R_}/results/real_uniform"
LAD = f"{OUT}/ladder"
CATBANK = f"{OUT}/catbank"
BTRUTH = f"{R_}/results/breadth_truth"
# gb_catbank outputs (stem = system + tag, tag "" or "_smalln"; bank.json carries no tag)
BANK_JSON = CATBANK + "/{s}_bank.json"                       # system, members, sha256_members, detection, n_members
BANK_JSON_KEYS = ("system", "members", "sha256_members", "detection", "n_members")
PRIORS_NPZ = CATBANK + "/{s}_priors.npz"                     # members, axis, "{set}|genes|relaxed_genes|relaxed_is_extra", "{set}|{method}|{key}"
PRIORS_RECEIPT = CATBANK + "/{s}_priors.receipt.json"        # output_sha256, config.bank_sha256, sets.{set}.status
FEATURES_TSV = CATBANK + "/{s}_features.tsv"                 # index gene; F0/F1/F1x columns; F0_complete, F1_complete, F1x_complete
FEATURES_RECEIPT = CATBANK + "/{s}_features.receipt.json"    # config.F0, config.F1, config.F1x
SOTA_NPZ = CATBANK + "/{s}_sota_priors.npz"                  # gb_catbank SOTA: Tier-1 SOTA target priors on the frozen T7 axis (optional; raw comparators only, never in the complete-case mask)
SOTA_RECEIPT = CATBANK + "/{s}_sota_priors.receipt.json"
SOTA_METHODS = ("regdiffusion", "deepsem", "pidc", "fasttenet")
GB_PRIORS = ["P_pairU", "P_pairM", "P_cellU", "P_cellM"]
PGB = "P_pairU"                                               # P_GB = production representation/channel
GEN_T7 = ["pearson", "genie3", "grnboost2"]                   # generic priors of the strict set T7 (catbank seven minus the GB four)
SETS = ("T7", "T7+regvelo", "T7+velorama")                    # catbank sets: strict over the seven (+ one extra method); root diagnostics ride on T7 only
SET_KEYS = ("mu", "sd", "Z", "S", "C", "Sr", "Cr", "Zx", "Pfull", "Pfull_relaxed")
MIN_CONTRIB, SMALLN_MEMBERS = 2, 5
# root diagnostics = catbank diagnostic methods of the T7 set (row-z'd on T7): per-root H1/H2 and
# the root mean of the pair variants
ROOT_MAP = {f"{m}@{t}": f"{c}_{m[2:]}" for m in ("P_pairU", "P_pairM") for t, c in (("root0", "H1"), ("root1", "H2"), ("mean", "Pmean"))}
F0_COLS = ["raw_det", "sm_det", "abundance", "pair_support", "dynamics"]
F1_EXTRA = ["raw_det_u", "mu_det", "mu_abundance", "mu_logvar", "mu_pair_support", "mu_dynamics", "abs_spearman_ms_dpt", "abs_spearman_mu_dpt"]
F1X_EXTRA = ["hvg_disp_norm", "cv2_spliced_norm", "us_ratio_norm"]                                                                        # auxiliary tier F1x
# gb_breadth_truth v2 outputs: BTRUTH/{BM|K562}{ttag}/{per_gene.tsv, per_pair.tsv.gz,
# perturbations.tsv, receipt.json}
PER_GENE_COLS = ["estimand", "arm", "gene", "n_perts_in_arm", "cover_min", "n_testable", "n_responder", "n_conf_nonresp", "n_ambiguous", "b_g", "burden",
                 "label_any_vs_confnon", "label_broad_vs_single"]
PER_PAIR_COLS = ["estimand", "pert", "gene", "T", "status", "e_crossfit"]   # e_rg = log2FC_A * log2FC_B
PERT_COLS = ["estimand", "pert", "arms"]                                    # optional: efficacy_status, null_calibration
STATUSES = {"UNTESTED", "RESPONDER", "CONFIDENT_NONRESPONDER", "AMBIGUOUS"}
TRUTH_HASHED = ("per_gene.tsv", "per_pair.tsv.gz", "perturbations.tsv")     # must equal receipt.json output_sha256
# declared analyses (each estimand x arm is a separate analysis): (estimand, arm, primary)
ANALYSES = {"BM": [("TOTAL", "BM19", True), ("WITHIN_STATE", "BM19", False)],
            "K562": [("TOTAL", "LAMBERT_TF", True), ("TOTAL", "F2_36", False), ("TOTAL", "ALL_PERT", False)]}
BANK_OF = {"BM": "BONEMARROW", "K562": "K562"}
# truth directory per breadth context
TRUTH_DIR = {"BM": "BM", "K562": "K562_EMP_SHARD"}
REF_SPEC = [("raw_det", -1), ("sm_det", -1), ("abundance", -1), ("pair_support", -1), ("mu_det", -1), ("dynamics", 1), ("mu_dynamics", 1),
            ("abs_spearman_mu_dpt", 1), ("mu_logvar", 1)]
ref_name = lambda col, sg: f"REF_{'-' if sg < 0 else '+'}{col}"
REF_NAMES = [ref_name(c, g) for c, g in REF_SPEC]


# {REF name: sign * feature} from the catbank feature columns (fail-closed if a declared column is absent)
def ref_vectors(feat, take=lambda v: v):
    miss = [c for c, _ in REF_SPEC if c not in feat]
    if miss:
        die(f"REF feature columns missing from the bank features: {miss}")
    return {ref_name(c, g): g * take(feat[c]) for c, g in REF_SPEC}


# Frozen truth-free composites (gb_composites: COMP_Z/rank/PC1/FA) fit once on U* = bank-axis genes with all nine REF columns finite
def comp_vectors(bank):
    if GC.SPEC != REF_SPEC:
        die("gb_composites.SPEC != REF_SPEC")
    return GC.comp_from_features(bank["feat"])
BLOCK = {"KLF1": "GATA1"}                          # KLF1 held-out lives on the GATA1 substrate: one block
EXPECT_BLOCKS = {"regulon": {"GATA1", "PANC", "BONEMARROW", "DENTATEGYRUS", "ZEBRAFISH", "GASTRULATION"},
                 "functional": {"BONEMARROW", "PANC", "GATA1", "K562"}}
_CB = {}


# gb_catbank (row_consts, prior_sums, loo_prior, members_sha)
def cb():
    if "m" not in _CB:
        sys.path.insert(0, f"{R_}/scratch")
        import gb_catbank
        _CB["m"] = gb_catbank
    return _CB["m"]


def outdir(stage, sset="T7", relaxed=False, tier="F1"):
    suf = ("" if sset == "T7" else "_" + sset.replace("+", "_")) + ("_relaxed" if relaxed else "") + ("" if tier == "F1" else "_" + tier)
    return f"{LAD}/{stage}{suf}"


def gens_of(sset):
    if sset not in SETS:
        die(f"unknown set {sset}")
    return GEN_T7 + ([sset.split("+")[1]] if "+" in sset else [])


# statistical constants
K, MINPOS, SEED = 5, 10, 20261101
DELTA_AUROC = 0.010
AP_LIFT_DELTA = 0.10               # PROVISIONAL declared margin for AP-lift differences (secondary readings)
ALPHA, N_FAM, TAIL_MIN = 0.05, 2, 5
STAGES = {"smalln": dict(Bfit=50, Braw=50), "pilot": dict(Bfit=200, Braw=2000), "final": dict(Bfit=6000, Braw=10000)}   # final Bfit: percentile tails usable at alpha=.05/28
B_INFO, MIN_CROSS_PERTS = 200, 3      # information-only fitted models run at pilot B only; crossed-context test needs >= this many common perturbed TFs
KS = (20, 50, 100)
BROAD_FRAC, MIN_RESP_BROAD, MIN_T = 0.25, 2, 2
COVER_FRAC, COVER_FLOOR, COVER_FLOOR_FOLD, CONF_FRAC = 0.5, 5, 2, 0.8   # = gb_breadth_truth v2 rules (full arm; consistency check); fold floor is PROVISIONAL
NFIT = [0]                                         # fit counter (selftest: proves the bootstrap refits)


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def fold_of(genes):   # identical to gb_decomp.py:54-55
    return np.array([int(hashlib.sha1(str(g).upper().encode()).hexdigest(), 16) % K for g in genes])


def die(msg):
    raise SystemExit("BLOCKED_BY_SPEC_LOADER: " + msg)


def need(z, keys, what):
    miss = [k for k in keys if k not in (z.files if hasattr(z, "files") else z)]
    if miss:
        die(f"{what} missing keys {miss}")


# metrics
# Weighted AUROC / AP / prevalence of one score vector s for every row of a (B,n) weight matrix (== sklearn sample_weight metrics)
def wraw(y, s, Wm):
    o = np.argsort(-s, kind="stable")
    s, y, Wm = s[o], y[o], Wm[:, o]
    st = np.r_[0, np.flatnonzero(np.diff(s) != 0) + 1]
    P = np.add.reduceat(Wm * y, st, axis=1)
    N = np.add.reduceat(Wm * (~y), st, axis=1)
    cP, cN = P.cumsum(1), N.cumsum(1)
    TP, TN = cP[:, -1], cN[:, -1]
    with np.errstate(all="ignore"):
        auc = (P * (TN[:, None] - cN + 0.5 * N)).sum(1) / (TP * TN)
        ap = (P / TP[:, None] * np.where(cP + cN > 0, cP / (cP + cN), 0.0)).sum(1)
        prev = TP / (TP + TN)
    bad = (TP <= 0) | (TN <= 0)
    auc[bad] = ap[bad] = np.nan
    return auc, ap, prev


def pt_metrics(y, s):
    a, ap, pv = wraw(y, s, np.ones((1, len(y))))
    o = np.argsort(-s, kind="stable")   # ties -> original order
    d = dict(AUROC=a[0], AP=ap[0], prev=pv[0], APlift=ap[0] / pv[0])
    d.update({f"P@{k}": (float(y[o[:k]].mean()) if len(y) >= k else np.nan) for k in KS})
    return d


# (auc[B], APlift[B]) for each raw score under gene-multiplicity draws W[:B][:, cols]
def raw_draws(W, cols, y, scores, B):
    out = {k: ([], []) for k in scores}
    for i in range(0, B, 256):
        Wc = np.asarray(W[i:min(i + 256, B)])[:, cols].astype(float)
        for k, s in scores.items():
            a, ap, pv = wraw(y, s, Wc)
            out[k][0].append(a)
            out[k][1].append(ap / pv)
    return {k: (np.concatenate(v[0]), np.concatenate(v[1])) for k, v in out.items()}


# Shared multinomial gene-multiplicity matrix (B,N) int16, deterministic in (tag, N); created atomically, prefix-stable in B
def gene_draws(d, tag, N, B):
    os.makedirs(d, exist_ok=True)
    p = f"{d}/draws_{tag}_{N}.npy"
    if os.path.exists(p) and np.load(p, mmap_mode="r").shape[0] >= B:
        return p
    rng = np.random.default_rng(np.random.SeedSequence([SEED, zlib.crc32(tag.encode()), N]))
    Wm = np.empty((B, N), np.int16)
    for i in range(0, B, 256):
        Wm[i:i + 256] = rng.multinomial(N, np.full(N, 1.0 / N), size=min(256, B - i))
    tmp = p + f".tmp{os.getpid()}.npy"
    np.save(tmp, Wm)
    os.replace(tmp, p)
    return p


# evaluator: StandardScaler -> SplineTransformer(n_knots=4, degree=3) -> L2 logistic (C=1) / Ridge(1);  training-only fits
NCONV = [0]                                        # ConvergenceWarnings seen in fits (reported in the receipts; never silenced silently)


# a point-estimate fit failed: the endpoint is marked FAILED (logged), the pool keeps running
class PointFitFailure(RuntimeError):
    ...


def fit_pred(kind, Xtr, ytr, w, Xte):
    NFIT[0] += 1
    est = Ridge(alpha=1.0) if kind == "reg" else LogisticRegression(C=1.0, max_iter=2000)
    m = make_pipeline(StandardScaler(), SplineTransformer(n_knots=4, degree=3), est)
    nm = type(est).__name__.lower()
    with warnings.catch_warnings(record=True) as wl:
        warnings.simplefilter("always", ConvergenceWarning)
        m.fit(Xtr, ytr, standardscaler__sample_weight=w, **{f"{nm}__sample_weight": w})
    NCONV[0] += sum(issubclass(x.category, ConvergenceWarning) for x in wl)
    return m.predict(Xte) if kind == "reg" else m.predict_proba(Xte)[:, 1]


# out-of-fold predictions; genes with multiplicity 0 are not used for training; folds are by original gene identity
def oof_w(X, y, folds, c):
    p = np.full(len(y), np.nan)
    for k in range(K):
        te = folds == k
        tr = (folds != k) & (c > 0)
        if te.sum() == 0 or y[tr].sum() < 2:
            continue
        p[te] = fit_pred("clf", X[tr], y[tr], c[tr], X[te])
    return p


# bank loader
# Catbank -> in-memory bank
def load_bank(system, tag="", sset="T7", relaxed=False, roots=True, tier="F1"):
    C = cb()
    stem = system + tag
    bj = json.load(open(BANK_JSON.format(s=system)))
    need(bj, BANK_JSON_KEYS, "bank.json")
    if bj["system"] != system or bj["sha256_members"] != C.members_sha(bj["members"]) or bj["n_members"] != len(bj["members"]):
        die("bank.json inconsistent (system / sha256_members / n_members)")
    members = list(bj["members"][:SMALLN_MEMBERS] if tag == "_smalln" else bj["members"])
    pf = PRIORS_NPZ.format(s=stem)
    rec = json.load(open(PRIORS_RECEIPT.format(s=stem)))
    if rec.get("output_sha256") != sha256(pf) or rec.get("config", {}).get("bank_sha256") != bj["sha256_members"]:
        die("priors.npz receipt does not match the file / the bank")
    if rec.get("sets", {}).get(sset, {}).get("status") != "OK":
        die(f"priors set {sset} status != OK in the receipt")
    z = np.load(pf, allow_pickle=True)
    need(z, ("members", "axis", f"{sset}|genes", f"{sset}|relaxed_genes", f"{sset}|relaxed_is_extra"), "priors.npz")
    if [str(x) for x in z["members"]] != members:
        die("priors.npz members != bank.json members")
    Tg = z[f"{sset}|genes"].astype(str)
    genes = z[f"{sset}|relaxed_genes"].astype(str) if relaxed else Tg
    extra = np.asarray(z[f"{sset}|relaxed_is_extra"], bool)
    if len(extra) != len(z[f"{sset}|relaxed_genes"]) or (~extra).sum() != len(Tg):
        die("relaxed_is_extra inconsistent with the T / relaxed gene lists")
    if len(set(genes)) != len(genes):
        die("gene axis not unique")
    gens = gens_of(sset)
    roots = bool(roots and sset == "T7")           # catbank stores the root diagnostics on the frozen T7 only
    meths = GB_PRIORS + gens + (sorted(set(ROOT_MAP.values())) if roots else [])
    nb, ZSC, RAW = len(members), {}, {}
    for m in meths:
        need(z, [f"{sset}|{m}|{k}" for k in SET_KEYS], "priors.npz")
        g = lambda k: np.asarray(z[f"{sset}|{m}|{k}"], float)
        Z, S, Cc = g("Z"), g("S"), g("C")
        if Z.shape != (nb, len(Tg)) or S.shape != (len(Tg),) or Cc.shape != (len(Tg),) or g("Zx").shape != (nb, int(extra.sum())):
            die(f"shapes of {sset}|{m}")
        if not np.isfinite(Z).all():
            die(f"{sset}|{m}: Z not finite on the strict set T")
        if not np.allclose(C.loo_prior(Z, S, Cc, None, MIN_CONTRIB), g("Pfull"), atol=1e-9, equal_nan=True):
            die(f"{sset}|{m}: Pfull != loo_prior(Z,S,C,None)")
        RAW[m] = Z * g("sd")[:, None] + g("mu")[:, None]
        if relaxed:
            Zr = np.empty((nb, len(genes)))
            Zr[:, ~extra], Zr[:, extra] = Z, g("Zx")
            s2, c2 = C.prior_sums(Zr)
            if not (np.allclose(s2, g("Sr")) and np.array_equal(c2, g("Cr"))):
                die(f"{sset}|{m}: relaxed Sr/Cr do not reproduce from [Z | Zx]")
            ZSC[m] = (Zr, g("Sr"), g("Cr"))
        else:
            ZSC[m] = (Z, S, Cc)
    if roots:
        for v in ("pairU", "pairM"):
            P = RAW[f"P_{v}"]
            if not (np.allclose(np.minimum(RAW[f"H1_{v}"], RAW[f"H2_{v}"]), P, atol=1e-5) and np.allclose((RAW[f"H1_{v}"] + RAW[f"H2_{v}"]) / 2, RAW[f"Pmean_{v}"], atol=1e-5)):
                die(f"root diagnostics {v}: min(H1,H2) != production prior or Pmean != (H1+H2)/2 (raw units)")
        for name, src in ROOT_MAP.items():
            ZSC[name] = ZSC[src]
        for src in set(ROOT_MAP.values()):
            del ZSC[src]
    f0 = list(F0_COLS)
    f1 = f0 + list(F1_EXTRA) + (list(F1X_EXTRA) if tier == "F1x" else [])
    F = pd.read_csv(FEATURES_TSV.format(s=stem), sep="\t", index_col="gene")
    fr = json.load(open(FEATURES_RECEIPT.format(s=stem)))
    cf = fr.get("config", {})
    if fr.get("output_sha256") != sha256(FEATURES_TSV.format(s=stem)) or cf.get("F0") != f0 or cf.get("F1") != f0 + list(F1_EXTRA) or \
            cf.get("F1x") != f0 + list(F1_EXTRA) + list(F1X_EXTRA):
        die("features receipt: file sha or F0/F1/F1x names differ from the frozen names")
    need({c: 1 for c in F.columns}, f0 + F1_EXTRA + F1X_EXTRA + ["F0_complete", "F1_complete", "F1x_complete"], "features.tsv")
    for flag, cols in (("F0_complete", f0), ("F1_complete", f0 + F1_EXTRA), ("F1x_complete", f0 + F1_EXTRA + F1X_EXTRA)):
        if not np.array_equal(F[flag].to_numpy(bool), np.isfinite(F[cols].to_numpy(float)).all(1)):
            die(f"features.tsv {flag} disagrees with the finiteness of its columns")
    if not set(genes) <= set(F.index):
        die("features.tsv does not cover the prior gene axis")
    F = F.reindex(genes)
    zs, sfiles = ({}, {}) if relaxed else load_sota(stem, bj, pf, Tg, members)
    return dict(genes=genes, members=[m.upper() for m in members], src_idx={m.upper(): i for i, m in enumerate(members)}, ZSC=ZSC, gens=gens, ZSC_SOTA=zs, sota=list(zs),
                extra=sorted(k for k in ZSC if "@" in k), f0=f0, f1=f1, feat={c: F[c].to_numpy(float) for c in f1}, sset=sset, relaxed=relaxed,
                system=system, tier=tier, roots=roots, files={"bank": sha256(BANK_JSON.format(s=system)), "priors": sha256(pf),
                                                                  "features": sha256(FEATURES_TSV.format(s=stem)), **sfiles})


# Tier-1 SOTA target priors {method: (Z, S, C)} from gb_catbank SOTA (optional; {} if the artifact is absent)
def load_sota(stem, bj, priors_file, Tg, members):
    sf = SOTA_NPZ.format(s=stem)
    if not os.path.exists(sf):
        return {}, {}
    rec = json.load(open(SOTA_RECEIPT.format(s=stem)))
    if rec.get("output_sha256") != sha256(sf) or rec.get("bank_sha256") != bj["sha256_members"] or rec.get("priors_sha256") != sha256(priors_file):
        die("sota_priors receipt does not match the file / the bank / the priors.npz")
    z = np.load(sf, allow_pickle=True)
    need(z, ("members", "genes", "methods"), "sota_priors.npz")
    if [str(x) for x in z["genes"]] != [str(x) for x in Tg] or [str(x) for x in z["members"]] != [str(x) for x in members]:
        die("sota_priors gene axis / members differ from the frozen T7 axis / bank members")
    out = {}
    for m in (str(x) for x in z["methods"]):
        need(z, [f"{m}|{k}" for k in ("Z", "S", "C", "Pfull")], f"sota_priors.npz {m}")
        Z, S, C = (np.asarray(z[f"{m}|{k}"], float) for k in ("Z", "S", "C"))
        if Z.shape != (len(members), len(Tg)) or not np.isfinite(Z).all():
            die(f"sota {m}: Z shape / non-finite on the frozen T7 axis")
        if not np.allclose(cb().loo_prior(Z, S, C, None, MIN_CONTRIB), z[f"{m}|Pfull"], atol=1e-9, equal_nan=True):
            die(f"sota {m}: Pfull != loo_prior(Z,S,C,None)")
        out[m] = (Z, S, C)
    return out, {"sota": sha256(sf)}


def axis_index(bank_genes, genes):
    ex = {g: i for i, g in enumerate(bank_genes)}
    gi = np.array([ex.get(g, -1) for g in genes])
    if (gi < 0).any():
        up, dup = {}, set()
        for i, g in enumerate(bank_genes):
            u = g.upper()
            if u in up:
                dup.add(u)
            up[u] = i
        gi = np.array([i if i >= 0 else (up.get(g.upper(), -1) if g.upper() not in dup else -1) for i, g in zip(gi, genes)])
    return gi


# Frozen endpoints via gb_decomp.endpoints (RU.trace_endpoints / KLF1 held-out)
def real_endpoints(system):
    sys.path.insert(0, f"{R_}/scratch")
    import gb_decomp as DC
    for name, fam, src, genes, ev, y, gb in DC.endpoints(system):
        if name.startswith("F3|"):
            continue
        yield dict(name=name, fam=fam, tf=str(src).upper(), genes=np.asarray(genes).astype(str), ev=np.asarray(ev, bool),
                   y=np.asarray(y, bool), G=np.asarray(gb, float))


# one endpoint: common frozen mask, ladder models, refit bootstrap


def _is_info(k):
    n = k.split("|")[-1]
    return n == "F0" or (n.startswith("F1+") and n not in ("F1+ALLGEN", "F1+PGB", "F1+ALLGEN+PGB", "F1+G", "F1+PGB+G", "F1+ALLGEN+PGB+G"))


def eval_endpoint(ep, bank, W, Bfit, Braw):
    if isinstance(W, str):
        W = np.load(W, mmap_mode="r")
    n_fit0, n_conv0 = NFIT[0], NCONV[0]
    Binfo = min(Bfit, B_INFO)
    gi = axis_index(bank["genes"], ep["genes"])
    ok = gi >= 0
    j = np.where(ok, gi, 0)
    fi = bank["src_idx"].get(ep["tf"])
    gens, f0, f1 = bank["gens"], bank["f0"], bank["f1"]
    pri, ex = {}, bank["extra"]
    for m in GB_PRIORS + gens + ex:   # leave-focal-out always: catbank.loo_prior(Z,S,C,r) (focal not in bank => r=None => full prior; flagged)
        Zm, Sm, Cm = bank["ZSC"][m]
        pri[m] = np.where(ok, cb().loo_prior(Zm, Sm, Cm, fi, MIN_CONTRIB)[j], np.nan)
    feat = {c: np.where(ok, bank["feat"][c][j], np.nan) for c in f1}
    G, ev, y = ep["G"], ep["ev"], ep["y"]
    fin = lambda vs: np.all([np.isfinite(v) for v in vs], 0)
    C = ev & ok & fin([G, *feat.values(), *pri.values()])          # one frozen common evaluation mask (no label used)
    Tm = ev & ok & fin([pri[m] for m in GB_PRIORS])                 # full-T mask for the standalone factorial
    yc, yt = y[C], y[Tm]
    row = dict(endpoint=ep["name"], family=ep["fam"], tf=ep["tf"], n_eval=int(ev.sum()), n_C=int(C.sum()), npos_C=int(yc.sum()),
               n_T=int(Tm.sum()), npos_T=int(yt.sum()), focal_in_bank=fi is not None, lo_used=fi is not None, generic_set=",".join(gens))
    if yc.sum() < MINPOS or (~yc).sum() < MINPOS:
        row["status"] = "TOO_FEW"
        return dict(row=row, ok=False)
    if len(np.unique(gi[C])) != C.sum():
        raise SystemExit(f"REFUSING: endpoint genes map non-injectively onto the bank axis ({ep['name']})")
    folds = fold_of(ep["genes"][C])
    cols = {**{c: feat[c][C] for c in f1}, **{m: pri[m][C] for m in GB_PRIORS + gens + ex}, "G": G[C]}
    raw = {f"C|raw|{m}": cols[m] for m in GB_PRIORS + gens + ex}   # ex = root diagnostics P@root0/root1/mean (pair U and M)
    raw["C|raw|G"] = cols["G"]
    raw.update({f"C|raw|{k}": v for k, v in ref_vectors(cols).items()})   # REF rankers: unsupervised, fixed signs, same mask
    rawT = {f"T|raw|{m}": pri[m][Tm] for m in GB_PRIORS}
    specs = {"F0": f0, "F1": f1, **{f"F1+{g}": f1 + [g] for g in gens}, "F1+ALLGEN": f1 + gens, "F1+PGB": f1 + [PGB], "F1+G": f1 + ["G"],
             "F1+ALLGEN+PGB": f1 + gens + [PGB], "F1+PGB+G": f1 + [PGB, "G"], "F1+ALLGEN+PGB+G": f1 + gens + [PGB, "G"]}
    specs.update({f"F1+PGB@{t}": f1 + [f"{PGB}@{t}"] for t in ("root0", "root1", "mean") if f"{PGB}@{t}" in cols})
    Xs = {f"C|fit|{k}": np.column_stack([cols[c] for c in v]) for k, v in specs.items()}
    pm = {}
    for k, s in raw.items():
        pm[k] = pt_metrics(yc, s)
    for k, s in rawT.items():
        pm[k] = pt_metrics(yt, s)
    one = np.ones(len(yc))
    for k, X in Xs.items():
        p = oof_w(X, yc, folds, one)
        if np.isnan(p).any():
            raise PointFitFailure(f"point fit failed: {ep['name']} {k}")
        pm[k] = pt_metrics(yc, p)
    if fi is not None:   # residual R_rg = Z_GB(r,g) - P_GB^(-r)(g) (raw, label-free score)
        Rv = np.where(ok, (bank["ZSC"][PGB][0][fi] - cb().loo_prior(*bank["ZSC"][PGB], fi, MIN_CONTRIB))[j], np.nan)
        mR = C & np.isfinite(Rv)
        if mR.sum() >= 2 * MINPOS and y[mR].sum() >= MINPOS and (~y[mR]).sum() >= MINPOS:
            for mk, v in pt_metrics(y[mR], Rv[mR]).items():
                row[f"R|raw|R|{mk}"] = v
            row["n_R"] = int(mR.sum())
    for k, d in pm.items():
        for mk, v in d.items():
            row[f"{k}|{mk}"] = v
    pt = {k: (d["AUROC"], d["APlift"]) for k, d in pm.items()}
    dr_raw = raw_draws(W, gi[C], yc, raw, Braw)
    dr_raw.update(raw_draws(W, gi[Tm], yt, rawT, Braw))
    dr = {k: ([], []) for k in Xs}
    for b in range(Bfit):
        c = np.asarray(W[b])[gi[C]].astype(float)      # gene multiplicities of this resample
        v = c > 0
        for k, X in Xs.items():
            if b >= Binfo and _is_info(k):
                continue                                 # information-only models: pilot B only
            p = oof_w(X, yc, folds, c)                   # full refit on the resample
            if np.isnan(p[v]).any():
                a, l = np.nan, np.nan
            else:
                aa, ap, pv = wraw(yc[v], p[v], c[v][None, :])
                a, l = aa[0], ap[0] / pv[0]              # resample's own prevalence
            dr[k][0].append(a)
            dr[k][1].append(l)
    dr = {k: (np.array(a), np.array(l)) for k, (a, l) in dr.items()}
    dr_fit = {k: v for k, v in dr.items() if not _is_info(k)}
    dr_info = {k: v for k, v in dr.items() if _is_info(k)}
    row.update(status="OK", n_fits=NFIT[0] - n_fit0, n_convergence_warnings=NCONV[0] - n_conv0)
    return dict(row=row, ok=True, pt=pt, dr_raw=dr_raw, dr_fit=dr_fit, dr_info=dr_info, gens=gens, nconv=NCONV[0] - n_conv0)


# endpoint -> TF (mean over the TF's endpoints) -> block sums (per family). cnt = endpoints per (family, TF) expected
class Acc:

    def __init__(s, eps):
        s.cnt = Counter((e["fam"], e["tf"]) for e in eps)
        s.buf = defaultdict(list)
        s.F = {}
        s.nconv = 0

    def add(s, ep, res):
        k = (ep["fam"], ep["tf"])
        s.cnt[k] -= 1
        if res["ok"]:
            s.buf[k].append(res)
            s.nconv += res.get("nconv", 0)
        if s.cnt[k] == 0 and s.buf[k]:
            rs = sorted(s.buf.pop(k), key=lambda r: r["row"]["endpoint"])
            kr, kf, ki = sorted(rs[0]["dr_raw"]), sorted(rs[0]["dr_fit"]), sorted(rs[0]["dr_info"])
            allk = kr + kf + ki
            mean = lambda f: np.mean([f(r) for r in rs], 0)
            pt = mean(lambda r: np.array([r["pt"][q] for q in allk]))
            raw = mean(lambda r: np.array([r["dr_raw"][q] for q in kr]))
            fit = mean(lambda r: np.array([r["dr_fit"][q] for q in kf]))
            inf = mean(lambda r: np.array([r["dr_info"][q] for q in ki]))
            F = s.F.setdefault(ep["fam"], dict(n=0, tfs=[], pt=0, raw=0, fit=0, inf=0, kr=kr, kf=kf, ki=ki, gens=rs[0]["gens"]))
            F["n"] += 1
            F["tfs"].append(ep["tf"])
            F["pt"], F["raw"], F["fit"], F["inf"] = F["pt"] + pt, F["raw"] + raw, F["fit"] + fit, F["inf"] + inf

    def dump(s, path, meta):
        d = {"meta": np.array(json.dumps(meta))}
        for fam, F in s.F.items():
            d[f"{fam}|info"] = np.array(json.dumps(dict(n=F["n"], tfs=sorted(F["tfs"]), kr=F["kr"], kf=F["kf"], ki=F["ki"], gens=F["gens"])))
            d[f"{fam}|pt"], d[f"{fam}|raw"], d[f"{fam}|fit"], d[f"{fam}|inf"] = F["pt"], F["raw"], F["fit"], F["inf"]
        np.savez_compressed(path, **d)


def _work(i):
    S = _S
    try:
        return i, eval_endpoint(S["eps"][i], S["bank"], S["W"], S["Bfit"], S["Braw"])
    except PointFitFailure as e:
        ep = S["eps"][i]
        return i, dict(ok=False, row=dict(endpoint=ep["name"], family=ep["fam"], tf=ep["tf"], status=f"FAILED: {e}"))


_S = {}


def run_endpoints(system, stage, n_jobs, limit=None, tag="", sset="T7", relaxed=False, roots=True, tier="F1"):
    Bf, Br = STAGES[stage]["Bfit"], STAGES[stage]["Braw"]
    outd = outdir(stage, sset, relaxed, tier)
    os.makedirs(outd, exist_ok=True)
    block = BLOCK.get(system, system)
    t0 = time.time()
    bank = load_bank(block, tag, sset, relaxed, roots, tier)
    W = gene_draws(outd, block, len(bank["genes"]), max(Bf, Br))
    eps = sorted(real_endpoints(system), key=lambda e: e["name"])[:limit]
    if not eps:
        raise SystemExit(f"no endpoints for {system}")
    _S.update(eps=eps, bank=bank, W=W, Bfit=Bf, Braw=Br)
    acc, rows = Acc(eps), []
    if n_jobs > 1:
        with mp.get_context("fork").Pool(n_jobs) as pool:
            for n, (i, res) in enumerate(pool.imap_unordered(_work, range(len(eps)))):
                acc.add(eps[i], res)
                rows.append(res["row"])
                print(f"[{n + 1}/{len(eps)}] {eps[i]['name']} {res['row']['status']}", flush=True)
    else:
        for i in range(len(eps)):
            _, res = _work(i)
            acc.add(eps[i], res)
            rows.append(res["row"])
            print(f"[{i + 1}/{len(eps)}] {eps[i]['name']} {res['row']['status']}", flush=True)
    pd.DataFrame(rows).sort_values("endpoint").to_csv(f"{outd}/{system}_endpoints.tsv", sep="\t", index=False)
    meta = dict(system=system, block=block, stage=stage, Bfit=Bf, Braw=Br, Binfo=min(Bf, B_INFO), limit=limit, sset=sset, relaxed=relaxed, roots=bank["roots"],
                bank_tag=tag, tier=tier, script_sha256=sha256(__file__))
    acc.dump(f"{outd}/{system}_blockstats.npz", meta)
    receipt(f"{outd}/{system}_receipt.json", dict(meta, inputs=bank["files"], n_endpoints=len(eps), n_ok=sum(r["status"] == "OK" for r in rows),
                                                  too_few=[r["endpoint"] for r in rows if r["status"] == "TOO_FEW"],
                                                  failed=[(r["endpoint"], r["status"]) for r in rows if str(r["status"]).startswith("FAILED")],
                                                  generic_set=bank["gens"], convergence_warnings=acc.nconv, seconds=round(time.time() - t0, 1),
                                                  draws=W, draws_sha256=sha256(W),
                                                  note="point estimates use the original genes; bootstrap replicates re-weight/duplicate genes, so a percentile "
                                                       "interval can exclude the point estimate (selftest B=3 shows it); intervals are working-bootstrap"))


def receipt(path, d):
    import sklearn
    d.update(script_sha256=sha256(__file__), sklearn=sklearn.__version__, numpy=np.__version__, time=time.strftime("%Y-%m-%dT%H:%M:%S"),
             DELTA_AUROC=DELTA_AUROC, AP_LIFT_DELTA=AP_LIFT_DELTA, AP_LIFT_DELTA_STATUS="PROVISIONAL", SEED=SEED, ALPHA=ALPHA, N_FAM=N_FAM,
             TAIL_MIN=TAIL_MIN, K=K, MINPOS=MINPOS, cmd=" ".join(sys.argv))
    json.dump(d, open(path, "w"), indent=1, default=str)


# aggregation: family medians, readings, cases
def ci(d, pt, a):
    d = d[np.isfinite(d)]
    if len(d) < 20:
        return np.nan, np.nan, "insufficient_draws"
    if len(d) * a / 2 >= TAIL_MIN:
        return float(np.quantile(d, a / 2)), float(np.quantile(d, 1 - a / 2)), "percentile"
    z, s = norm.ppf(1 - a / 2), d.std(ddof=1)
    return float(pt - z * s), float(pt + z * s), "normal_approx"      # tails hold < TAIL_MIN draws: normal approximation (labelled)


# '~=' = CI within [-delta, delta]
def reading(pt, lo, hi, delta, maj):
    if not np.isfinite([pt, lo, hi]).all():
        return "?"
    if pt >= delta and lo > 0 and maj:
        return ">"
    if pt <= -delta and hi < 0 and maj:
        return "<"
    if lo >= -delta and hi <= delta:
        return "≈"
    return "?"


def build_specs(gens_all, gens_case):
    p = lambda m, t: f"{m}|raw|{t}"
    S = []
    for m in "CT":
        for a, b in itertools.combinations(GB_PRIORS, 2):
            S.append((f"fx|{m}|{a}-{b}", [(p(m, a), 1), (p(m, b), -1)], "fx"))
    S.append(("fx|C|interaction", [(p("C", "P_pairU"), 1), (p("C", "P_pairM"), -1), (p("C", "P_cellU"), -1), (p("C", "P_cellM"), 1)], "info"))
    for g in gens_all:
        S.append((f"PGB-gen:{g}", [(p("C", PGB), 1), (p("C", g), -1)], "case" if g in gens_case else "info"))
        S.append((f"G-gen:{g}", [(p("C", "G"), 1), (p("C", g), -1)], "case" if g in gens_case else "info"))
    S.append(("G-PGB", [(p("C", "G"), 1), (p("C", PGB), -1)], "case"))
    for rn in REF_NAMES:
        S.append((f"REF|PGB-{rn}", [(p("C", PGB), 1), (p("C", rn), -1)], "ref"))
    f = lambda t: f"C|fit|{t}"
    S.append(("F1-F0", [(f("F1"), 1), (f("F0"), -1)], "info"))
    for m in ("P_pairU", "P_pairM"):
        for t in ("root0", "root1", "mean"):
            S.append((f"root|{m}:min-{t}", [(p("C", m), 1), (p("C", f"{m}@{t}"), -1)], "info"))
    for t in ("root0", "root1", "mean"):
        S.append((f"root|F1+PGB:min-{t}", [(f("F1+PGB"), 1), (f(f"F1+PGB@{t}"), -1)], "info"))
    for g in gens_all:
        S.append((f"gen:{g}|F1", [(f(f"F1+{g}"), 1), (f("F1"), -1)], "info"))
    for nm, a, b in (("ALLGEN|F1", "F1+ALLGEN", "F1"), ("PGB|F1", "F1+PGB", "F1"), ("PGB|ALLGEN", "F1+ALLGEN+PGB", "F1+ALLGEN"),
                     ("G|PGB", "F1+PGB+G", "F1+PGB"), ("G|ALLGEN+PGB", "F1+ALLGEN+PGB+G", "F1+ALLGEN+PGB"),
                     ("G|F1", "F1+G", "F1"), ("ALLGEN+PGB+G|F1", "F1+ALLGEN+PGB+G", "F1")):
        S.append((nm, [(f(a), 1), (f(b), -1)], "case"))
    return S


def fact_label(rd):
    E, Gt = (lambda k: rd[k] == "≈"), (lambda k: rd[k] == ">")
    n = lambda a, b: f"{a}-{b}"
    pu, pm, cu, cm = GB_PRIORS
    six = [n(a, b) for a, b in itertools.combinations(GB_PRIORS, 2)]
    if all(E(k) for k in six):
        return "D"
    if Gt(n(pu, pm)) and Gt(n(pu, cu)) and Gt(n(pu, cm)):
        return "A"
    if E(n(pu, cu)) and E(n(pm, cm)) and Gt(n(pu, pm)) and Gt(n(cu, cm)):
        return "B"
    if E(n(pu, pm)) and E(n(cu, cm)) and Gt(n(pu, cu)) and Gt(n(pm, cm)):
        return "C"
    return "UNRESOLVED"


D_KEYS = ("ALLGEN|F1", "PGB|F1", "PGB|ALLGEN", "G|PGB", "G|ALLGEN+PGB", "G|F1", "ALLGEN+PGB+G|F1")


# Returns (label, A_conditions_met)
def case_label(rd, gens, family="functional"):
    if not gens:
        return "UNRESOLVED", False
    E, Gt, Lt = (lambda k: rd.get(k) == "≈"), (lambda k: rd.get(k) == ">"), (lambda k: rd.get(k) == "<")
    pg = [f"PGB-gen:{g}" for g in gens]
    gg = [f"G-gen:{g}" for g in gens]
    a_met = bool(Gt("G-PGB") and all(Gt(k) for k in gg) and all(E(k) for k in pg))
    if all(E(k) for k in D_KEYS):
        return "D", a_met
    if a_met and family == "functional":
        return "A", a_met
    if all(Gt(k) for k in pg) and Gt("PGB|F1") and Gt("PGB|ALLGEN") and E("G|PGB") and E("G|ALLGEN+PGB"):
        return "B", a_met
    if all(E(k) for k in pg):
        return "C", a_met
    if any(Lt(k) for k in pg):
        return "GENERIC_BEATS_GB", a_met
    return "UNRESOLVED", a_met


BLOCK_NOTE = ("with 4-6 substrate blocks a family-median '≈' needs (nearly) every block within +-delta and '>' needs a strict majority of blocks with the "
              "same sign; intervals are gene-conditional/two-level working-bootstrap, not biological replication")


# parts: list of dicts {meta, fams} (loaded blockstats). returns (models_df, contrasts_df, cases_df, notes)
def aggregate_blocks(parts, allow_partial, seed=SEED):
    same = ("limit", "bank_tag", "roots", "sset", "relaxed", "tier", "Bfit", "Braw", "Binfo", "script_sha256", "stage")
    for kx in same:
        vals = {json.dumps(P["meta"].get(kx)) for P in parts}
        if len(vals) != 1:
            die(f"blockstats disagree on {kx}: {sorted(vals)} (refusing to mix runs)")
    meta0 = parts[0]["meta"]
    Bfit, Braw, Binfo = meta0["Bfit"], meta0["Braw"], meta0["Binfo"]
    BL = defaultdict(dict)
    for P in sorted(parts, key=lambda P: (P["meta"]["system"], P["meta"]["block"])):
        for fam, F in P["fams"].items():
            b = P["meta"]["block"]
            if b in BL[fam]:
                E = BL[fam][b]
                if (E["kr"], E["kf"], E["ki"]) != (F["kr"], F["kf"], F["ki"]):
                    die(f"blockstats key mismatch within block {b}")
                E["n"] += F["n"]
                E["tfs"] = sorted(E["tfs"] + F["tfs"])
                for q in ("pt", "raw", "fit", "inf"):
                    E[q] = E[q] + F[q]
            else:
                BL[fam][b] = dict(F)
    notes = []
    for fam, exp in EXPECT_BLOCKS.items():
        have = set(BL.get(fam, {}))
        if have != exp:
            msg = f"{fam}: blocks present {sorted(have)} != expected {sorted(exp)}"
            if not allow_partial:
                die(msg + " (use --allow-partial)")
            notes.append("PARTIAL " + msg)
    if "GATA1" in BL.get("functional", {}) and "KLF1" not in BL["functional"]["GATA1"]["tfs"]:
        msg = "functional/GATA1 block lacks the KLF1 held-out TF"
        if not allow_partial:
            die(msg)
        notes.append("PARTIAL " + msg)
    if meta0.get("limit"):
        notes.append(f"SMOKE: endpoints limited to the first {meta0['limit']} per system")
    for fam in BL:
        for b, E in BL[fam].items():   # sums -> block means
            E["ptm"], E["rawm"], E["fitm"], E["infm"] = E["pt"] / E["n"], E["raw"] / E["n"], E["fit"] / E["n"], E["inf"] / E["n"]
            E["kall"] = E["kr"] + E["kf"] + E["ki"]
    mrows, crows, lrows = [], [], []
    for fam in sorted(BL):
        blocks = BL[fam]
        names = sorted(blocks)
        gsets = [set(blocks[b]["gens"]) for b in names]
        gens_case = [g for g in blocks[names[0]]["gens"] if all(g in s for s in gsets)]
        gens_all = sorted(set().union(*gsets), key=lambda g: (g not in GEN_T7, g))
        specs = build_specs(gens_all, gens_case)
        n_fx = sum(1 for s in specs if s[2] == "fx" and s[0].startswith("fx|C|"))
        n_case = sum(1 for s in specs if s[2] == "case")
        alpha = {"fx": ALPHA / (n_fx * N_FAM), "case": ALPHA / (n_case * N_FAM), "ref": ALPHA / (len(REF_NAMES) * N_FAM), "info": ALPHA, "model": ALPHA}
        nb = len(names)
        picks = {}

        def pick(L):    # shared block-resampling indices per draw length (same across every contrast)
            if L not in picks:
                picks[L] = np.random.default_rng(np.random.SeedSequence([seed, zlib.crc32(fam.encode()), L])).integers(0, nb, size=(L, nb))
            return picks[L]
        kinds = {}
        for b in names:
            E = blocks[b]
            kinds.update({k: "raw" for k in E["kr"]})
            kinds.update({k: "fit" for k in E["kf"]})
            kinds.update({k: "info" for k in E["ki"]})
        allspecs = specs + [(f"MODEL:{k}", [(k, 1)], "model") for k in sorted(kinds)]
        rd = {"AUROC": {}, "APlift": {}}
        for name, terms, grp in allspecs:
            miss = [b for b in names if not all(t in blocks[b]["kall"] for t, _ in terms)]
            if miss and grp in ("fx", "case", "ref"):
                die(f"{fam}: blocks {miss} lack keys for the declared contrast {name} (refusing to drop blocks silently)")
            use_b = [b for b in names if b not in miss]
            if not use_b:
                continue
            tk = [kinds[t] for t, _ in terms]
            kind = "raw" if all(k == "raw" for k in tk) else ("info" if "info" in tk else "fit")
            arr = lambda E, t: {"raw": E["rawm"], "fit": E["fitm"], "info": E["infm"]}[kinds[t]][{"raw": E["kr"], "fit": E["kf"], "info": E["ki"]}[kinds[t]].index(t)]
            L = min(arr(blocks[use_b[0]], t).shape[-1] for t, _ in terms)
            pts = np.array([sum(c * blocks[b]["ptm"][blocks[b]["kall"].index(t)] for t, c in terms) for b in use_b])                 # (nb',2)
            drs = np.array([sum(c * arr(blocks[b], t)[:, :L] for t, c in terms) for b in use_b])                                        # (nb',2,L)
            for mi, (mname, delta) in enumerate((("AUROC", DELTA_AUROC), ("APlift", AP_LIFT_DELTA))):
                pv, D = pts[:, mi], drs[:, mi, :]
                point = float(np.median(pv))
                cond = np.median(D, 0)                       # a failed block fails the draw (NaN propagates; counted in valid_draws)
                pk = pick(L)[:, :len(pv)] % len(pv)
                two = np.median(D[pk, np.arange(L)[:, None]], 1)
                loo = [np.median(np.delete(pv, i)) for i in range(len(pv))] if len(pv) > 1 else [point]
                maj = float(np.mean(np.sign(pv) == np.sign(point))) > 0.5 if point != 0 else False
                a = alpha[grp]
                clo, chi, cm = ci(cond, point, a)
                tlo, thi, tm = ci(two, point, a)
                if kind == "raw":
                    lo, hi, meth, ctype = tlo, thi, tm, "two_level"
                else:                                         # fitted contrasts: wider of gene-conditional and two-level
                    both = bool(np.isfinite([clo, tlo, chi, thi]).all())
                    lo, hi, meth, ctype = (min(clo, tlo) if both else np.nan), (max(chi, thi) if both else np.nan), f"{cm}|{tm}", "wider_of_gene_conditional_and_two_level"
                r = reading(point, lo, hi, delta, maj) if grp in ("fx", "case", "info", "ref") else ""
                if grp in ("fx", "case", "info"):
                    rd[mname][(fam, name)] = r
                row = dict(family=fam, contrast=name, metric=mname, kind=kind, group=grp, n_blocks=len(pv), point=point, ci_lo=lo, ci_hi=hi,
                           ci_type=ctype, ci_method=meth, alpha_each=a, cond_lo=clo, cond_hi=chi, two_lo=tlo, two_hi=thi, delta=delta, reading=r,
                           maj_blocks=maj, loo_min=min(loo), loo_max=max(loo), valid_draws_cond=int(np.isfinite(cond).sum()),
                           valid_draws_two=int(np.isfinite(two).sum()), B=L)
                (mrows if grp == "model" else crows).append(row)
        for mname in ("AUROC", "APlift"):     # descriptive HINDSIGHT_UPPER_BOUND: best REF ranker per metric (max over the 9; no inference, not a decision quantity)
            ref_rows = [r for r in mrows if r["family"] == fam and r["metric"] == mname and r["contrast"].startswith("MODEL:C|raw|REF_")]
            if ref_rows:
                best = max(ref_rows, key=lambda r: r["point"])
                mrows.append(dict(family=fam, contrast="HINDSIGHT_UPPER_BOUND", metric=mname, kind="raw", group="hindsight", n_blocks=best["n_blocks"],
                                  point=best["point"], best_ref=best["contrast"].replace("MODEL:C|raw|", ""), note="max over REF models; descriptive, no interval"))
        for mname in ("AUROC", "APlift"):
            r_ = {k[1]: v for k, v in rd[mname].items() if k[0] == fam}
            fl = lambda m: fact_label({f"{a}-{b}": r_.get(f"fx|{m}|{a}-{b}", "?") for a, b in itertools.combinations(GB_PRIORS, 2)})
            lab, a_met = case_label(r_, gens_case, fam)
            lrows.append(dict(family=fam, metric=mname, delta=DELTA_AUROC if mname == "AUROC" else AP_LIFT_DELTA, case=lab, A_conditions_met=a_met,
                              factorial_C=fl("C"), factorial_T=fl("T"), n_blocks=nb, blocks=",".join(names), generic_case_set=",".join(gens_case),
                              generic_all=",".join(gens_all), primary=mname == "AUROC",
                              readings=json.dumps({k: v for k, v in r_.items() if k.startswith(("fx|C", "PGB", "G", "ALLGEN"))}, ensure_ascii=False),
                              notes="; ".join(notes) or "complete", block_note=BLOCK_NOTE))
    return pd.DataFrame(mrows), pd.DataFrame(crows), pd.DataFrame(lrows), notes


def load_parts(outd, systems=None):
    parts = []
    for f in sorted(glob.glob(f"{outd}/*_blockstats.npz")):
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        if systems and meta["system"] not in systems:
            continue
        fams = {}
        for k in z.files:
            if k.endswith("|info"):
                fam = k[:-5]
                info = json.loads(str(z[k]))
                fams[fam] = dict(info, pt=z[f"{fam}|pt"], raw=z[f"{fam}|raw"], fit=z[f"{fam}|fit"], inf=z[f"{fam}|inf"])
        parts.append(dict(meta=meta, fams=fams))
    if not parts:
        die(f"no blockstats in {outd}")
    return parts


def run_aggregate(stage, allow_partial, sset="T7", relaxed=False, tier="F1"):
    outd = outdir(stage, sset, relaxed, tier)
    parts = load_parts(outd)
    M, C, L, notes = aggregate_blocks(parts, allow_partial)
    M.to_csv(f"{outd}/family_models.tsv", sep="\t", index=False)
    C.to_csv(f"{outd}/family_contrasts.tsv", sep="\t", index=False)
    L.to_csv(f"{outd}/case_labels.tsv", sep="\t", index=False)
    receipt(f"{outd}/receipt_aggregate.json", dict(stage=stage, systems=sorted(p["meta"]["system"] for p in parts), notes=notes,
                                                   blockstats_sha256={os.path.basename(f): sha256(f) for f in sorted(glob.glob(f"{outd}/*_blockstats.npz"))}))
    print(L[["family", "metric", "case", "A_conditions_met", "factorial_C", "factorial_T", "n_blocks", "notes"]].to_string(index=False))


# breadth
# Breadth quantities from the perturbation subset m (bool over perts, optional integer multiplicities pw = perturbation-cluster bootstrap)
def mk_targets(T, Rr, E, CN, m, floor, min_t=MIN_T, pw=None):
    w = (np.ones(T.shape[0]) if pw is None else np.asarray(pw, float)) * m
    W_ = w[:, None]
    nT, nR, nC = (W_ * T).sum(0), (W_ * (Rr & T)).sum(0), (W_ * (CN & T)).sum(0)
    cover = max(floor, int(np.ceil(COVER_FRAC * w.sum())))
    okg = nT >= min_t
    with np.errstate(all="ignore"):
        b = np.where(okg, nR / np.maximum(nT, 1), np.nan)
        bur = np.where(okg, np.sqrt(np.maximum((W_ * E * T).sum(0) / np.maximum(nT, 1), 0.0)), np.nan)
    anyl = np.full(len(nT), np.nan)
    anyl[okg & (nR >= 1)] = 1
    anyl[okg & (nR == 0) & (nC >= CONF_FRAC * nT) & (nT >= cover)] = 0
    brl = np.full(len(nT), np.nan)
    brl[okg & (b >= BROAD_FRAC) & (nR >= MIN_RESP_BROAD)] = 1
    brl[okg & (nR == 1) & (nC >= cover - 1) & (nT >= cover)] = 0
    return dict(b=b, bur=bur, any=anyl, broad=brl, resp=np.where(okg, (nR >= 1).astype(float), np.nan))


# (AUROC, AP-lift, n_pos, n_neg) over rows with a finite label
# NaN metrics if fewer than 2 rows of a class (AP-lift needs >=1 positive)
def bin_metrics(lab, s):
    m = np.isfinite(lab)
    y = lab[m] == 1
    if y.sum() < 1 or (~y).sum() < 1:
        return np.nan, np.nan, int(y.sum()), int((~y).sum())
    a, ap, pv = wraw(y, s[m], np.ones((1, len(y))))
    return (a[0] if (y.sum() >= 2 and (~y).sum() >= 2) else np.nan), ap[0] / pv[0], int(y.sum()), int((~y).sum())


# Within one row set (a held-out fold, or the pooled rows)
def fold_metrics(pred, tg, idx):
    o = {}
    fail = lambda p, t: bool(np.any(np.isfinite(t) & ~np.isfinite(p)))

    def sp(x, y):
        m = np.isfinite(x) & np.isfinite(y)
        return (spearmanr(x[m], y[m])[0], int(m.sum())) if m.sum() > 5 else (np.nan, int(m.sum()))
    for nm, pk, tk in (("rho_breadth", "b", "b"), ("rho_burden", "bur", "bur")):
        pp, tt = pred[pk][idx], tg[tk][idx]
        f = fail(pp, tt)
        v, w = sp(pp, tt)
        o[nm] = (np.nan if f else v, w, f)
    for nm, k in (("any", "any"), ("broad", "broad")):
        pp, tt = pred[k][idx], tg[k][idx]
        f = fail(pp, tt)
        a, apl, npos, nneg = bin_metrics(tt, np.nan_to_num(pp))
        o[f"auc_{nm}"] = (np.nan if f else a, npos * nneg, f)
        o[f"aplift_{nm}"] = (np.nan if f else apl, npos + nneg, f)
    s, resp, b = pred["any"][idx], tg["resp"][idx], tg["b"][idx]
    f = fail(s, resp)
    m = np.isfinite(s) & np.isfinite(resp)
    order = np.argsort(-s[m], kind="stable")
    for k in KS:
        sel = order[:k]
        ok = m.sum() >= k
        o[f"yield_any@{k}"] = (float(resp[m][sel].mean()) if (ok and not f) else np.nan, 1, f)
        o[f"meanbreadth@{k}"] = (float(b[m][sel].mean()) if (ok and not f) else np.nan, 1, f)
    return o


# fold-stratified combination of within-fold metrics (calibration differs by fold because each fold's model is trained without its own perturbations)
def combine_folds(recs):
    out = {}
    for k in recs[0]:
        vw = [(r[k][0], r[k][1]) for r in recs]
        failed = any(r[k][2] for r in recs)
        good = [(v, w) for v, w in vw if np.isfinite(v) and w > 0]
        if k.startswith(("yield_any", "meanbreadth")):
            out[k] = float(np.mean([v for v, _ in vw])) if not failed else np.nan                       # legacy per-fold mean (NaN if any fold undefined)
            out[k + "_stratified"] = float(np.mean([v for v, _ in good])) if (good and not failed) else np.nan
            out[k + "_folds_used"] = len(good)
            continue
        out[k] = (sum(v * w for v, w in good) / sum(w for _, w in good)) if (good and not failed) else np.nan
        if k in POOLED_KEYS and k.startswith(("rho", "auc")):
            out[k + "_folds_used"] = len(good)
    return out


# One (re)fit + evaluation for gene multiplicities c and perturbation multiplicities pw (ones = point estimate)
def breadth_draw(c, pw, D):
    n0 = NCONV[0]
    n = len(c)
    idx = np.repeat(np.arange(n), c.astype(int))
    T, Rr, E, CN = D["truth"]
    res = {m: [] for m in D["models"]}
    pool = {m: ({q: [] for q in ("b", "bur", "any", "broad")}, {q: [] for q in ("b", "bur", "any", "broad", "resp")}) for m in D["models"]}
    for kf in range(K):
        tg_te = mk_targets(T, Rr, E, CN, D["pf"] == kf, COVER_FLOOR_FOLD, MIN_T, pw)
        tg_tr = mk_targets(T, Rr, E, CN, D["pf"] != kf, COVER_FLOOR, MIN_T, pw)
        preds = {m: {q: np.full(n, np.nan) for q in ("b", "bur", "any", "broad")} for m in D["fitted"]}
        for m, X in D["fitted"].items():
            for jf in range(K):
                te = D["gf"] == jf
                trn = (D["gf"] != jf) & (c > 0)
                r = trn & np.isfinite(tg_tr["b"]) & np.isfinite(tg_tr["bur"])
                if r.sum() > 20:
                    Y = np.column_stack([tg_tr["b"][r], tg_tr["bur"][r]])
                    P = fit_pred("reg", X[r], Y, c[r], X[te])
                    preds[m]["b"][te], preds[m]["bur"][te] = P[:, 0], P[:, 1]
                for q in ("any", "broad"):
                    r = trn & np.isfinite(tg_tr[q])
                    if (tg_tr[q][r] == 1).sum() >= 2 and (tg_tr[q][r] == 0).sum() >= 2:
                        preds[m][q][te] = fit_pred("clf", X[r], tg_tr[q][r] == 1, c[r], X[te])
        for m in D["models"]:
            p = preds[m] if m in D["fitted"] else {q: D["raw"][m] for q in ("b", "bur", "any", "broad")}
            res[m].append(fold_metrics(p, tg_te, idx))
            for q in ("b", "bur", "any", "broad"):       # pooled oof rows: prediction from the fold's training perts, target from its held-out perts
                pool[m][0][q].append(p[q][idx])
                pool[m][1][q].append(tg_te[q][idx])
            pool[m][1]["resp"].append(tg_te["resp"][idx])
    out = {}
    for m, fs in res.items():
        d = combine_folds(fs)                                                          # PRIMARY: fold-stratified
        pp, tt = ({q: np.concatenate(v) for q, v in x.items()} for x in pool[m])
        pooled = fold_metrics(pp, tt, np.arange(len(pp["b"])))
        d.update({k + "_pooled_LEGACY": pooled[k][0] for k in POOLED_KEYS})           # legacy pooled-oof version: biased for fitted models (fold-specific calibration)
        out[m] = d
    return out, NCONV[0] - n0


POOLED_KEYS = ("rho_breadth", "rho_burden", "auc_any", "aplift_any", "auc_broad", "aplift_broad")   # primary = fold-stratified; *_pooled_LEGACY = old pooled rows


# per held-out fold: number of genes with a finite b / distinct b values / any-response 1,0 / broad 1,0 / responders (diagnostic for NA metrics)
def fold_label_counts(truth, pf, keep):
    T, Rr, E, CN = (x[:, keep] for x in (truth["T"], truth["R"], truth["E"], truth["CN"]))
    rows = []
    for kf in range(K):
        tg = mk_targets(T, Rr, E, CN, pf == kf, COVER_FLOOR_FOLD, MIN_T)
        b = tg["b"][np.isfinite(tg["b"])]
        rows.append(dict(fold=kf, n_perts=int((pf == kf).sum()), n_genes_b_finite=int(len(b)), n_distinct_b=int(len(np.unique(b))), n_responder_genes=int((tg["resp"] == 1).sum()),
                         any1=int((tg["any"] == 1).sum()), any0=int((tg["any"] == 0).sum()), broad1=int((tg["broad"] == 1).sum()), broad0=int((tg["broad"] == 0).sum())))
    return rows


_BD = {}


def _bd(a):
    return breadth_draw(a[0], a[1], _BD)


# propensity metrics (raw label-free predictors only; no fitting)
# Primary inference uses tf_macro with tf_only bootstrap; two_level and fold_macro are sensitivities.
PROP_LEVELS = [f"fold{k}" for k in range(K)] + ["fold_macro", "tf_macro"]
PROP_PRIMARY_LEVEL, PROP_PRIMARY_MODE = "tf_macro", "tf_only"
PROP_GROUPS = ("REF", "COMP", "EXTERNAL", "TIER1_SOTA")             # comparator groups (TIER1_SOTA = RNA-only regdiffusion / deepsem / pidc / fasttenet target priors)
PROP_PRIMARY = ("APlift", "FB50_resp")                        # AP lift and top-50 yield
PROP_SECONDARY = ("P@50", "R@50", "AP", "prev", "FB50_resp_allTF")
PROP_DECLARATION = dict(
    primary=dict(level=PROP_PRIMARY_LEVEL, mode=PROP_PRIMARY_MODE, metrics=list(PROP_PRIMARY)), secondary_metrics=list(PROP_SECONDARY),
    sensitivities="mode two_level (gene x TF cluster), level fold_macro / fold0..4, FB50_resp_allTF (all-TF top-50 yield), FB50U_* (top-50 chosen on the full bank universe U*)",
    equal_TF_macro="both primary metrics are averaged over the SAME TF set = TFs with AP defined (>=1 responder AND >=1 confident non-responder among the evaluable genes); n_TF per row",
    fixed_budget="top-50 by predictor over the truth-testable complete-case genes (FB50_*) chosen without looking at labels; ambiguous / untested slots are counted (FB50_unlab), never refilled; "
                 "sensitivity FB50U_* takes the top-50 over the FULL bank universe (all raw predictors finite), genes without a truth row count as unlabeled",
    tie_break="ties in a predictor are broken by upper-case gene symbol (lexicographic), deterministic",
    testability="gb_breadth_truth.py:432 T = est & det & ~self_mask; det = NT detection >= MIN_DET=.05 (gb_breadth_truth.py:81,743 -> gsel); est = estimable contrast (perturbed/NT cell and unit counts, "
                "arm size; header lines 38-42); self_mask = the perturbed gene itself; genes with zero null SD are UNTESTED in every mode (gb_breadth_truth.py:378-380).  Testability therefore "
                "depends on control-cell detection, cell counts / arm size and the self gene, NOT on the size of the response; AMBIGUOUS (T but neither RESPONDER nor CONFIDENT) does depend on the response and is not refilled.")


# Simultaneous max-|t| interval over the C columns of V (B, C) around the points p (C,), available-case as gb_layer_lattice.maxT
def simul_ci(V, p, a=ALPHA):
    V, p = np.asarray(V, float), np.asarray(p, float)
    fin = np.isfinite(V)
    nv = fin.sum(0)
    sd = np.array([V[fin[:, j], j].std(ddof=1) if nv[j] >= 20 else np.nan for j in range(V.shape[1])])
    good = np.isfinite(sd) & (sd > 0)
    if not good.any():
        return np.full(len(p), np.nan), np.full(len(p), np.nan), np.nan, nv, len(V)
    with np.errstate(all="ignore"):
        t = np.where(fin & good, np.abs(V - p) / np.where(good, sd, np.nan), -np.inf)
    mx = t.max(1)
    q = float(np.quantile(mx[np.isfinite(mx)], 1 - a)) if np.isfinite(mx).any() else np.nan
    return np.where(good, p - q * sd, np.nan), np.where(good, p + q * sd, np.nan), q, nv, len(V)


# Propensity metrics of one score s over gene rows with label lab in {1, 0, NaN}; ties in s broken by tie (integer rank of the gene symbol)
def prop_one(lab, s, tie):
    fb = lab[np.lexsort((tie, -s))[:50]]
    d = dict(FB50_resp=float((fb == 1).sum()), FB50_nonresp=float((fb == 0).sum()), FB50_unlab=float(np.isnan(fb).sum()))
    m = np.isfinite(lab)
    y = lab[m] == 1
    d.update(n_lab=float(m.sum()), n_pos=float(y.sum()))
    d.update(dict.fromkeys(["AP", "prev", "APlift"] + [f"{q}@{k}" for q in "PR" for k in KS], np.nan))
    if y.sum() < 1 or (~y).sum() < 1:
        return d
    a, ap, pv = wraw(y, s[m], np.ones((1, len(y))))
    o = np.lexsort((tie[m], -s[m]))
    d.update(AP=ap[0], prev=pv[0], APlift=ap[0] / pv[0])
    for k in KS:
        if len(y) >= k:
            d[f"P@{k}"], d[f"R@{k}"] = float(y[o[:k]].mean()), float(y[o[:k]].sum() / y.sum())
    return d


def _wmean(v, w):
    v, w = np.asarray(v, float), np.asarray(w, float)
    g = np.isfinite(v)
    return float((v[g] * w[g]).sum() / w[g].sum()) if g.any() else np.nan


# Evaluate all raw predictors for one gene and perturbation resample.
def prop_draw(c, pw, D, genes_fixed=False):
    idx = np.repeat(np.arange(len(c)), c.astype(int))
    T, Rr, E, CN = D["truth"]
    tie = D["tie"][idx]
    labs = {f"fold{k}": mk_targets(T, Rr, E, CN, D["pf"] == k, COVER_FLOOR_FOLD, MIN_T, pw)["any"][idx] for k in range(K)}
    pw = np.asarray(pw, float)
    tfs = np.flatnonzero(pw > 0)
    F = D.get("full") if genes_fixed else None
    out = {}
    for m, s0 in D["raw"].items():
        s = s0[idx]
        per = {lv: prop_one(lb, s, tie) for lv, lb in labs.items()}
        pt = [(i, prop_one(D["lab_tf"][i][idx], s, tie)) for i in tfs]
        use = [(i, r) for i, r in pt if np.isfinite(r["AP"])]
        wu = pw[[i for i, _ in use]]
        for lv, r in per.items():
            out.update({(lv, m, q): v for q, v in r.items()})
        for q in per["fold0"]:
            out[("fold_macro", m, q)] = float(np.nanmean([per[f"fold{k}"][q] for k in range(K)])) if any(np.isfinite(per[f"fold{k}"][q]) for k in range(K)) else np.nan
            out[("tf_macro", m, q)] = _wmean([r[q] for _, r in use], wu)
        out[("tf_macro", m, "n_TF")] = float(len(use))
        out[("tf_macro", m, "FB50_resp_allTF")] = _wmean([r["FB50_resp"] for _, r in pt], pw[tfs])
        if F is not None:
            t50 = F["top50"][m]
            fb = [(F["lab"][i][t50], pw[i]) for i, _ in use]
            for nm, f in (("FB50U_resp", lambda x: (x == 1).sum()), ("FB50U_nonresp", lambda x: (x == 0).sum()), ("FB50U_unlab", lambda x: np.isnan(x).sum())):
                out[("tf_macro", m, nm)] = _wmean([float(f(x)) for x, _ in fb], [w for _, w in fb])
    return out


def _pd(a):
    return prop_draw(a[0], a[1], _BD, a[2])


# Point + two bootstrap modes for the propensity metrics of the raw predictors D['raw']
def propensity_core(D, W, PW, B, groups, n_jobs=1):
    global _BD
    T, Rr, E, CN = D["truth"]
    D = dict(D, lab_tf=np.where(T & Rr, 1.0, np.where(T & CN, 0.0, np.nan)))
    Fu = D.get("full")
    if Fu is not None:                                                   # label-free top-50 on the full bank universe (once), labels of the truth rows only
        ok = np.all([np.isfinite(v) for v in Fu["vec"].values()], 0)
        keepU = np.flatnonzero(ok)
        labU = np.full((T.shape[0], len(ok)), np.nan)
        tt, rr, cc = Fu["truth"]
        labU[:, Fu["bi"]] = np.where(tt & rr, 1.0, np.where(tt & cc, 0.0, np.nan))
        D["full"] = dict(lab=labU[:, keepU], top50={m: np.lexsort((Fu["tie"][keepU], -Fu["vec"][m][keepU]))[:50] for m in D["raw"]}, n_universe=int(len(keepU)))
    _BD = D
    n, P = len(D["gf"]), T.shape[0]
    pt = prop_draw(np.ones(n), np.ones(P), D, True)
    rows, draws = [], {}
    isprim = lambda mode, lv, q: mode == PROP_PRIMARY_MODE and lv == PROP_PRIMARY_LEVEL and q in PROP_PRIMARY
    for mode in ("two_level", "tf_only"):
        args = [((np.asarray(W[b])[D["keep"]].astype(float) if mode == "two_level" else np.ones(n)), np.asarray(PW[b], float), mode == "tf_only") for b in range(B)]
        if n_jobs > 1:
            with mp.get_context("fork").Pool(n_jobs) as pool:
                dr = pool.map(_pd, args)
        else:
            dr = [_pd(a) for a in args]
        for k, p0 in pt.items():
            if mode == "two_level" and k[2].startswith("FB50U"):
                continue                                                   # universe genes without a gene draw: point / tf_only only
            v = np.array([d[k] for d in dr])
            draws[mode, k] = v
            lo, hi, meth = ci(v, p0, ALPHA)
            rows.append(dict(kind="model", mode=mode, level=k[0], name=k[1], metric=k[2], point=p0, lo=lo, hi=hi, ci_method=meth, n_valid=int(np.isfinite(v).sum()),
                             prop_primary=isprim(mode, k[0], k[2]), role=("GB_LABEL_FREE" if k[1][4:] in GB_PRIORS else [g for g in PROP_GROUPS if k[1] in groups[g]][0])))
    comp = [m for g in PROP_GROUPS for m in groups[g]]
    gb = f"raw:{PGB}"
    simlb = {}
    cfam = lambda m: [g for g in PROP_GROUPS if m in groups[g]][0]
    for mode in ("two_level", "tf_only"):
        for lv in ("tf_macro", "fold_macro"):
            mets = [q for q in PROP_PRIMARY + PROP_SECONDARY if (lv, gb, q) in pt]
            p0 = {(q, m): pt[lv, gb, q] - pt[lv, m, q] for q in mets for m in comp}
            V = {(q, m): draws[mode, (lv, gb, q)] - draws[mode, (lv, m, q)] for q in mets for m in comp}
            sk = [(q, m) for q in PROP_PRIMARY for m in comp]
            lo, hi, qq, nvv, nB = simul_ci(np.column_stack([V[k] for k in sk]), np.array([p0[k] for k in sk]))
            sim = {k: (lo[i], hi[i]) for i, k in enumerate(sk)}
            nval = {k: int(nvv[i]) for i, k in enumerate(sk)}
            for k in sk:
                simlb[mode, lv, k[0], k[1]] = sim[k][0]
            for q in mets:
                for m in comp:
                    plo, phi, meth = ci(V[q, m], p0[q, m], ALPHA)
                    rows.append(dict(kind="contrast", mode=mode, level=lv, name=f"{gb} - {m}", metric=q, point=p0[q, m], lo=plo, hi=phi, ci_method=meth,
                                     lo_simul=sim.get((q, m), (np.nan, np.nan))[0], hi_simul=sim.get((q, m), (np.nan, np.nan))[1], q_maxT=qq if (q, m) in sim else np.nan,
                                     n_valid_draws=nval.get((q, m), int(np.isfinite(V[q, m]).sum())), n_dropped_draws=nB - nval.get((q, m), int(np.isfinite(V[q, m]).sum())),
                                     n_family=len(sk), comparator_family=cfam(m), prop_primary=isprim(mode, lv, q),
                                     family=f"PROP|{mode}|{lv}|simul_maxT(primary metrics x comparators)" if (q, m) in sim else "pointwise"))
    for lv in ("fold_macro", "tf_macro"):                                                    # descriptive best-in-hindsight comparator (point only)
        for q in PROP_PRIMARY:
            ok = {m: pt[lv, m, q] for m in comp if np.isfinite(pt[lv, m, q])}
            if ok:
                bk = max(ok, key=ok.get)
                rows.append(dict(kind="hindsight", mode="point", level=lv, name="HINDSIGHT_BEST_COMPARATOR", metric=q, point=ok[bk], ci_method="none", best_comparator=bk,
                                 prop_primary=isprim(PROP_PRIMARY_MODE, lv, q), note="max over REF + COMP + external priors, selected on the observed metric; descriptive only"))
    tf_rows = []                                                                             # per-TF point rows
    for i, tf in enumerate(D["perts"]):
        for m, s0 in D["raw"].items():
            r = prop_one(D["lab_tf"][i], s0, D["tie"])
            tf_rows.append(dict(tf=tf, name=m, n_pos=r["n_pos"], n_eval=r["n_lab"], AP=r["AP"], APlift=r["APlift"], FB50_resp=r["FB50_resp"], FB50_unlab=r["FB50_unlab"],
                                in_macro_set=bool(np.isfinite(r["AP"]))))
    for r in rows:                                                                           # combined flag: simultaneous LB > 0 in both bootstrap modes
        if r["kind"] == "contrast" and r["family"] != "pointwise":
            m = r["name"].split(" - ", 1)[1]
            lbs = [simlb.get((md, r["level"], r["metric"], m), np.nan) for md in ("tf_only", "two_level")]
            r["superiority_both_modes"] = bool(all(np.isfinite(x) and x > 0 for x in lbs))
    tfd = pd.DataFrame(tf_rows)
    return pd.DataFrame(rows + robustness_rows(tfd, gb, comp)), tfd


DOMINANT_TFS = ("GATA1", "GFI1B", "RUNX1")


# post-hoc robustness of the tf_macro point estimates, computed from the per-TF rows (macro set = TFs with AP defined)
def robustness_rows(tfd, gb, comp):
    out = []
    d = tfd[tfd.in_macro_set]
    tfs = sorted(d.tf.unique())
    for q, col in (("APlift", "APlift"), ("FB50_resp", "FB50_resp")):
        W = d.pivot(index="name", columns="tf", values=col)
        for tag, drop in [(f"LOTO:{t}", [t]) for t in tfs] + [("DROP_DOMINANT:" + "+".join(x for x in DOMINANT_TFS if x in tfs), [x for x in DOMINANT_TFS if x in tfs])]:
            keep = [t for t in tfs if t not in drop]
            if not keep or (tag.startswith("DROP") and not drop):
                continue
            mac = W[keep].mean(1)
            for m in [gb] + comp:
                out.append(dict(kind="robustness", mode="point", level="tf_macro", name=m, metric=q, point=float(mac[m]), robustness=tag, n_TF=len(keep)))
                if m != gb:
                    out.append(dict(kind="robustness", mode="point", level="tf_macro", name=f"{gb} - {m}", metric=q, point=float(mac[gb] - mac[m]), robustness=tag, n_TF=len(keep)))
    return out


PRIMARY_BREADTH_METRICS = ("rho_breadth", "auc_any", "aplift_any", "yield_any@50")
# the stacked GB-vs-comparator simultaneous family uses the fold-stratified yield
ALLCOMP_BREADTH_METRICS = ("rho_breadth", "auc_any", "aplift_any", "yield_any@50_stratified")


# Genes (n,), cols {feature: (n,)}, priors {name: (n,)}, truth {T,R,E,CN,perts} on the same gene axis
def breadth_core(genes, cols, priors, truth, B, W, PW, n_jobs=1, refs=None, comps=None, full=None, strict_family=False, sota=None):
    perts = np.char.upper(np.asarray(truth["perts"]).astype(str))
    order = np.argsort([int(hashlib.sha1(p.encode()).hexdigest(), 16) for p in perts])
    pf = np.empty(len(perts), int)
    pf[order] = np.arange(len(perts)) % K
    if len(perts) < K:
        die(f"only {len(perts)} perturbations")
    pn = list(priors)
    gens = [p for p in pn if p not in GB_PRIORS]
    Xf = np.column_stack([cols[c] for c in cols])
    fitted = {"F1": Xf}
    for p in pn:
        fitted[f"F1+{p}"] = np.column_stack([Xf, priors[p]])
    fitted["F1+ALLGEN"] = np.column_stack([Xf] + [priors[g] for g in gens])
    fitted["F1+ALLGEN+PGB"] = np.column_stack([Xf] + [priors[g] for g in gens] + [priors[PGB]])
    raw = {f"raw:{p}": priors[p] for p in pn}
    refs, comps, sota = refs or {}, comps or {}, sota or {}
    raw.update({f"raw:{k}": v for k, v in refs.items()})                # REF rankers (unsupervised; same complete-case gene set, no extra exclusions)
    raw.update({f"raw:{k}": v for k, v in comps.items()})               # COMP composites (frozen, label-free, fit on U*; never enter the complete-case filter)
    raw.update({f"raw:{k}": v for k, v in sota.items()})                # Tier-1 SOTA target priors (label-free raw comparators; never enter the complete-case mask or the fitted models)
    keep = np.flatnonzero(np.isfinite(Xf).all(1) & np.all([np.isfinite(priors[p]) for p in pn], 0) & np.all([np.isfinite(v) for v in refs.values()], 0)) if refs else \
        np.flatnonzero(np.isfinite(Xf).all(1) & np.all([np.isfinite(priors[p]) for p in pn], 0))   # complete-case genes
    D = dict(models=list(fitted) + list(raw), fitted={k: v[keep] for k, v in fitted.items()}, raw={k: v[keep] for k, v in raw.items()}, keep=keep,
             gf=fold_of(genes)[keep], pf=pf, truth=tuple(x[:, keep] for x in (truth["T"], truth["R"], truth["E"], truth["CN"])), perts=list(perts),
             tie=np.unique(np.char.upper(np.asarray(genes).astype(str))[keep], return_inverse=True)[1])      # tie-break rank = upper-case gene symbol
    if full is not None:                                                # label-free bank-universe vectors for the FB50U sensitivity (top-50 on U*, unlabeled counted)
        if not set(D["raw"]) <= set(full["vec"]):
            die(f"full-universe vectors missing for {sorted(set(D['raw']) - set(full['vec']))}")
        D["full"] = dict(vec={k: np.asarray(full["vec"][k], float) for k in D["raw"]}, bi=full["bi"], truth=(truth["T"], truth["R"], truth["CN"]),
                         tie=np.unique(np.char.upper(np.asarray(full["genes"]).astype(str)), return_inverse=True)[1])
    if not all(np.isfinite(D["raw"][f"raw:{k}"]).all() for k in list(comps) + list(sota)):
        die("a COMP composite / SOTA prior is non-finite on the complete-case genes (U* / the frozen T7 axis must contain them)")
    if isinstance(W, str):
        W = np.load(W, mmap_mode="r")
    global _BD
    _BD = D
    pt, nconv = breadth_draw(np.ones(len(keep)), np.ones(len(perts)), D)
    args = [(np.asarray(W[b])[keep].astype(float), np.asarray(PW[b], float)) for b in range(B)]
    if n_jobs > 1:
        with mp.get_context("fork").Pool(n_jobs) as pool:
            out = pool.map(_bd, args)
    else:
        out = [_bd(a) for a in args]
    dr, nconv = [o[0] for o in out], nconv + sum(o[1] for o in out)
    pname = lambda p: f"F1+{p}"
    declared = [("F1+PGB - F1", f"F1+{PGB}", "F1"), ("F1+ALLGEN+PGB - F1+ALLGEN", "F1+ALLGEN+PGB", "F1+ALLGEN")] + \
               [(f"raw:PGB - raw:{g}", f"raw:{PGB}", f"raw:{g}") for g in gens]
    info = [(f"F1+{p} - F1", f"F1+{p}", "F1") for p in pn if p != PGB]
    ref_c = [(f"raw:PGB - raw:{k}", f"raw:{PGB}", f"raw:{k}") for k in refs]
    comp_c = [(f"raw:PGB - raw:{k}", f"raw:{PGB}", f"raw:{k}") for k in comps]
    a_decl = ALPHA / (len(declared) * len(PRIMARY_BREADTH_METRICS))
    a_ref = ALPHA / max(len(ref_c) * len(PRIMARY_BREADTH_METRICS), 1)   # Bonferroni within the declared family BREADTH|REF
    a_comp = ALPHA / max(len(comp_c) * len(PRIMARY_BREADTH_METRICS), 1)   # ... and within the composite family BREADTH|COMP
    sota_c = [(f"raw:PGB - raw:{k}", f"raw:{PGB}", f"raw:{k}") for k in sota]
    a_sota = ALPHA / max(len(sota_c) * len(PRIMARY_BREADTH_METRICS), 1)   # ... and within BREADTH|TIER1_SOTA
    # role: F1-containing fitted models are supplementary
    role = lambda m: ("SUPPLEMENTARY_SUPERVISED" if not m.startswith("raw:") else "GB_LABEL_FREE" if m[4:] in GB_PRIORS else "REF" if m[4:] in refs else "COMP" if m[4:] in comps
                      else "TIER1_SOTA" if m[4:] in sota else "EXTERNAL")
    rows = []
    for m in D["models"]:
        for q in pt[m]:
            v = np.array([d[m][q] for d in dr])
            lo, hi, meth = ci(v, pt[m][q], ALPHA)
            rows.append(dict(kind="model", name=m, role=role(m), metric=q, point=pt[m][q], lo=lo, hi=hi, ci_method=meth, alpha_each=ALPHA, declared=False,
                             n_valid=int(np.isfinite(v).sum()), n_failed=int((~np.isfinite(v)).sum()), n_genes=len(keep)))
    for fam_, lst in (("BREADTH|declared", declared), ("BREADTH|REF", ref_c), ("BREADTH|COMP", comp_c), ("BREADTH|TIER1_SOTA", sota_c), ("info", info)):
        for nm, a, b in lst:
            for q in pt[a]:
                v = np.array([d[a][q] - d[b][q] for d in dr])
                dec = fam_ != "info" and q in PRIMARY_BREADTH_METRICS
                al = (a_ref if fam_ == "BREADTH|REF" else a_comp if fam_ == "BREADTH|COMP" else a_sota if fam_ == "BREADTH|TIER1_SOTA" else a_decl) if dec else ALPHA
                p0 = pt[a][q] - pt[b][q]
                lo, hi, meth = ci(v, p0, al)
                rows.append(dict(kind="contrast", family=fam_, name=nm, metric=q, point=p0, lo=lo, hi=hi, ci_method=meth, alpha_each=al, declared=dec,
                                 role=("COMPARATOR" if a.startswith("raw:") and b.startswith("raw:") else "SUPPLEMENTARY_SUPERVISED"),
                                 n_valid=int(np.isfinite(v).sum()), n_failed=int((~np.isfinite(v)).sum()), n_genes=len(keep)))
    for q in pt[f"raw:{PGB}"]:                                          # descriptive HINDSIGHT_UPPER_BOUND (max over the REF rankers; no inference)
        if refs:
            vals = {k: pt[f"raw:{k}"][q] for k in refs}
            ok = {k: x for k, x in vals.items() if np.isfinite(x)}
            if ok:
                bk = max(ok, key=ok.get)
                rows.append(dict(kind="hindsight", family="BREADTH|REF", name="HINDSIGHT_UPPER_BOUND", metric=q, point=ok[bk], ci_method="none", declared=False,
                                 best_ref=bk, note="max over the REF rankers; descriptive only", n_genes=len(keep)))
    fam = declared[2:] + ref_c + comp_c + sota_c                           # GB - comparator: one simultaneous max-|t| family over {external + REF + COMP} x PRIMARY_BREADTH_METRICS
    for m_ in pt:                                                         # the stratified yield must reproduce the legacy value whenever the legacy value is defined
        lg, st = pt[m_].get("yield_any@50", np.nan), pt[m_].get("yield_any@50_stratified", np.nan)
        if np.isfinite(lg) and not abs(lg - st) < 1e-12:
            die(f"{m_}: yield_any@50_stratified {st} != legacy yield_any@50 {lg} at the point estimate")
    sk = [(q, x) for q in ALLCOMP_BREADTH_METRICS for x in fam]
    lo, hi, qq, nvv, nB = simul_ci(np.column_stack([[d[x[1]][q] - d[x[2]][q] for d in dr] for q, x in sk]), np.array([pt[x[1]][q] - pt[x[2]][q] for q, x in sk]))
    if strict_family and nB >= 20 and (nB - nvv).max() > 0.01 * nB:
        w_ = int(np.argmax(nB - nvv))
        die(f"TOTAL primary family missing {(nB - nvv[w_]) / nB:.1%} of draws (> 1%) for {sk[w_][0]} / {sk[w_][1][0]}")
    for i, (q, (nm, a, b)) in enumerate(sk):
        rows.append(dict(kind="contrast", family="BREADTH|ALLCOMP_simul_maxT", name=nm, metric=q, point=pt[a][q] - pt[b][q], lo=lo[i], hi=hi[i], ci_method="simul_maxT", q_maxT=qq,
                         n_valid_draws=int(nvv[i]), n_dropped_draws=int(nB - nvv[i]), n_family=len(sk), role="COMPARATOR", alpha_each=ALPHA, declared=False, n_genes=len(keep)))
    for q in ALLCOMP_BREADTH_METRICS:
        ok = {b: pt[b][q] for _, _, b in fam if np.isfinite(pt[b][q])}    # descriptive best observed comparator (labelled hindsight)
        if ok:
            bk = max(ok, key=ok.get)
            rows.append(dict(kind="hindsight", family="BREADTH|ALLCOMP", name="HINDSIGHT_BEST_COMPARATOR", metric=q, point=ok[bk], ci_method="none", declared=False,
                             best_ref=bk, note="max over REF + COMP + external priors; descriptive only", n_genes=len(keep)))
    df = pd.DataFrame(rows)
    df.attrs["propensity"], df.attrs["propensity_per_tf"] = propensity_core(D, W, PW, B, dict(GB=[f"raw:{p}" for p in GB_PRIORS], REF=[f"raw:{k}" for k in refs],
                                                                                              COMP=[f"raw:{k}" for k in comps], EXTERNAL=[f"raw:{g}" for g in gens], TIER1_SOTA=[f"raw:{k}" for k in sota]), n_jobs)
    df.attrs["convergence_warnings"] = nconv
    df.attrs["fold_label_counts"] = fold_label_counts(truth, pf, keep)
    return df


# bank prior recomputed without the bank members (rows) that are perturbed in the panel: (S - sum Z[rows]) / (C - n finite[rows])
def prior_excl(bank, m, rows, min_rows=MIN_CONTRIB, zsc=None):
    Zm, S, C = (zsc or bank["ZSC"])[m]
    if len(rows) == 0:
        return cb().loo_prior(Zm, S, C, None, min_rows)
    f = np.isfinite(Zm[rows])
    s, c = S - np.where(f, Zm[rows], 0.0).sum(0), C - f.sum(0)
    with np.errstate(all="ignore"):
        return np.where(c >= min_rows, s / np.maximum(c, 1), np.nan)


# per-perturbation x gene matrices built in memory from gb_breadth_truth's per_pair.tsv.gz, restricted to genes with n_testable > 0
def load_breadth_truth(ctx, est, arm, ttag=""):
    d = f"{BTRUTH}/{TRUTH_DIR[ctx]}{ttag}"
    rc = json.load(open(f"{d}/receipt.json"))
    if rc.get("phase") != "COMPLETE":
        die(f"{d}/receipt.json phase is {rc.get('phase')!r}, not COMPLETE")
    hs = rc.get("output_sha256", {})
    for fn in TRUTH_HASHED:
        if hs.get(fn) != sha256(f"{d}/{fn}"):
            die(f"{d}/{fn} does not match receipt.json output_sha256")
    calib = rc.get("null_gate_result", {}).get(est, {}).get("label", "NOT_RUN")
    pg = pd.read_csv(f"{d}/per_gene.tsv", sep="\t")
    need({c: 1 for c in pg.columns}, PER_GENE_COLS, "per_gene.tsv")
    pg = pg[(pg.estimand == est) & (pg.arm == arm)].reset_index(drop=True)
    if not len(pg):
        die(f"per_gene.tsv has no rows for estimand={est} arm={arm}")
    n_axis = len(pg)
    pt = pd.read_csv(f"{d}/perturbations.tsv", sep="\t")
    need({c: 1 for c in pt.columns}, PERT_COLS, "perturbations.tsv")
    sel = pt[(pt.estimand == est) & pt.arms.astype(str).apply(lambda a: arm in a.split(","))]
    perts = sorted(set(sel.pert.astype(str)))
    eff = ",".join(sorted(set(sel.efficacy_status.astype(str)))) if "efficacy_status" in sel.columns else "UNKNOWN"
    if len(perts) != int(pg.n_perts_in_arm.iloc[0]):
        die(f"perturbations.tsv arm size {len(perts)} != per_gene.n_perts_in_arm {int(pg.n_perts_in_arm.iloc[0])}")
    # symbols that occur more than once within (estimand, arm) (10x symbols are not unique) are
    # UNKNOWN: all their rows are dropped, one is never picked
    sym = pg.gene.astype(str)
    dup = set(sym[sym.duplicated(keep=False)])
    n_dup_rows = int(sym.isin(dup).sum())
    ids_unique = (bool(pg.gene_id.astype(str).is_unique) if "gene_id" in pg.columns else None)
    pg = pg[~sym.isin(dup)].reset_index(drop=True)
    pg = pg[pg.n_testable > 0].reset_index(drop=True)                    # filter first: only testable genes enter the breadth analysis
    genes = pg.gene.astype(str).to_numpy()
    if len(set(genes)) != len(genes):
        die("per_gene.tsv genes not unique after dropping duplicated symbols")
    gix, pix = {g: i for i, g in enumerate(genes)}, {p: i for i, p in enumerate(perts)}
    P, G = len(perts), len(genes)
    T, Rr, CN, E = np.zeros((P, G), bool), np.zeros((P, G), bool), np.zeros((P, G), bool), np.zeros((P, G), np.float32)
    n_pairs = 0
    for ch in pd.read_csv(f"{d}/per_pair.tsv.gz", sep="\t", usecols=PER_PAIR_COLS, chunksize=2_000_000):
        ch = ch[(ch.estimand == est) & ch.pert.astype(str).isin(pix)]
        if not len(ch):
            continue
        st = ch.status.astype(str).to_numpy()
        if not set(st) <= STATUSES:
            die(f"per_pair.tsv unknown status {set(st) - STATUSES}")
        t = ch["T"].to_numpy() == 1
        if not np.array_equal(t, st != "UNTESTED"):
            die("per_pair.tsv T flag disagrees with status")
        gs = ch.gene.astype(str)
        c = gs.map(gix)
        if c[t & ~gs.isin(dup).to_numpy()].isna().any():
            die("per_pair.tsv testable pair on a gene absent from the testable per_gene list")
        keepr = ~c.isna().to_numpy()                                       # untested pairs of non-testable genes / all rows of duplicated symbols are dropped
        r, cc, st, ee = ch.pert.astype(str).map(pix).to_numpy()[keepr], c.to_numpy()[keepr].astype(int), st[keepr], ch.e_crossfit.to_numpy(float)[keepr]
        T[r, cc] = st != "UNTESTED"
        Rr[r, cc] = st == "RESPONDER"
        CN[r, cc] = st == "CONFIDENT_NONRESPONDER"
        E[r, cc] = np.nan_to_num(ee)
        n_pairs += int(keepr.sum())
    tg = mk_targets(T, Rr, E, CN, np.ones(P, bool), COVER_FLOOR, min_t=1)
    if int(pg.cover_min.iloc[0]) != max(COVER_FLOOR, int(np.ceil(COVER_FRAC * P))):
        die("cover_min of per_gene.tsv differs from the declared coverage rule")
    same = lambda a, b, tol: np.array_equal(np.isnan(a), np.isnan(b)) and np.allclose(a[~np.isnan(a)], b[~np.isnan(b)], rtol=tol, atol=tol)
    ok = (np.array_equal(T.sum(0), pg.n_testable.to_numpy()) and np.array_equal((Rr & T).sum(0), pg.n_responder.to_numpy())
          and same(tg["b"], pg.b_g.to_numpy(float), 1e-5) and same(tg["bur"], pg.burden.to_numpy(float), 2e-3)
          and same(tg["any"], pg.label_any_vs_confnon.to_numpy(float), 0) and same(tg["broad"], pg.label_broad_vs_single.to_numpy(float), 0))
    if not ok:
        die("per_pair.tsv.gz does not reproduce per_gene.tsv (n_testable / n_responder / b_g / burden / labels)")
    return dict(truth_dir=d, truth_receipt_sha256=sha256(f"{d}/receipt.json"), truth_phase=rc["phase"], truth_sharding=rc.get("sharding", {}).get("n"),
                genes=genes, T=T, R=Rr, E=E, CN=CN, perts=np.array(perts), n_pairs=n_pairs, arm_size=P, calibration=calib, efficacy_status=eff,
                n_axis_genes=n_axis, n_testable_genes=G, n_dup_symbols=len(dup), n_dup_rows_dropped=n_dup_rows, gene_id_unique=ids_unique, files={fn: hs[fn] for fn in TRUTH_HASHED})


def run_breadth(ctx, stage, n_jobs, variants=("full", "excl"), tag="", ttag="", sset="T7", relaxed=False, tier="F1", allow_uncal=False, arms=None):
    outd = outdir(stage, sset, relaxed, tier)
    os.makedirs(outd, exist_ok=True)
    if ctx == "CROSS":
        return run_cross(stage, sset, relaxed, tier)
    Bf = STAGES[stage]["Bfit"]
    system = BANK_OF[ctx]
    bank = load_bank(system, tag, sset, relaxed, roots=False, tier=tier)
    comps_bank, cdiag = comp_vectors(bank)               # frozen composites on the bank axis (truth-free, computed before any truth is joined)
    print(f"COMP composites: U*={cdiag['n_Ustar']}/{cdiag['n_universe']} genes (excluded {cdiag['n_excluded']}), NO_CALL={cdiag.get('no_call')}")
    allrows, allprop, alltf, info, nconv = [], [], [], [], 0
    role = {"full": "PRIMARY", "excl": "MANDATORY_SENSITIVITY (bank members perturbed in the panel removed)"}
    for est, arm, primary in ANALYSES[ctx]:
        if arms and arm not in arms:                                 # --arms: restrict the analyses (other arms are simply not run; recorded in the receipt via cmd)
            continue
        tr = load_breadth_truth(ctx, est, "BM_SMALLN" if (ttag and arm == "BM19") else arm, ttag)
        if tr["calibration"] != "CALIBRATED" and not allow_uncal:
            die(f"REFUSING {ctx} {est}/{arm}: null-gate verdict {tr['calibration']} (use --allow-uncalibrated to label and continue)")
        bi = axis_index(bank["genes"], tr["genes"])
        okg = bi >= 0
        if okg.sum() < 3 * K or len(np.unique(bi[okg])) != okg.sum():
            die(f"{ctx} {est}/{arm}: truth genes map (almost) nowhere / non-injectively onto the bank axis ({int(okg.sum())}/{len(bi)})")
        bi = bi[okg]
        truth = dict(perts=tr["perts"], T=tr["T"][:, okg], R=tr["R"][:, okg], E=tr["E"][:, okg].astype(float), CN=tr["CN"][:, okg])
        genes = tr["genes"][okg]
        cols = {c: bank["feat"][c][bi] for c in bank["f1"]}
        tag_w = f"BR_{ctx}_{est}_{arm}"
        W = gene_draws(outd, tag_w, len(bi), Bf)
        rp = np.random.default_rng(np.random.SeedSequence([SEED, zlib.crc32(tag_w.encode()), 1]))
        PW = rp.multinomial(len(tr["perts"]), np.full(len(tr["perts"]), 1.0 / len(tr["perts"])), size=Bf)     # perturbation-cluster draws (shared by all models)
        rows_ex = [bank["members"].index(p) for p in np.char.upper(tr["perts"]) if p in bank["src_idx"]]
        b_all = mk_targets(truth["T"], truth["R"], truth["E"], truth["CN"], np.ones(len(tr["perts"]), bool), COVER_FLOOR, min_t=1)["b"]   # = per_gene b_g
        save = dict(genes=genes, b=b_all, F1=np.column_stack([cols[c] for c in cols]), f1=np.array(list(cols)), perts=tr["perts"], T=truth["T"], R=truth["R"])
        for v in variants:
            names = GB_PRIORS + bank["gens"]
            pri_full = {m: prior_excl(bank, m, [] if v == "full" else rows_ex) for m in names}                      # bank axis (FB50U sensitivity)
            pri = {m: x[bi] for m, x in pri_full.items()}
            sota_full = {m: prior_excl(bank, m, [] if v == "full" else rows_ex, zsc=bank["ZSC_SOTA"]) for m in bank["sota"]}
            ref_full = ref_vectors(bank["feat"])
            full = dict(vec={**{f"raw:{m}": x for m, x in pri_full.items()}, **{f"raw:{k}": x for k, x in ref_full.items()}, **{f"raw:{k}": x for k, x in comps_bank.items()},
                             **{f"raw:{k}": x for k, x in sota_full.items()}}, bi=bi, genes=bank["genes"])
            save.update({f"{v}|{m}": x for m, x in pri.items()})
            t0 = time.time()
            df = breadth_core(genes, cols, pri, truth, Bf, W, PW, n_jobs, refs=ref_vectors(bank["feat"], lambda v: v[bi]), comps={k: x[bi] for k, x in comps_bank.items()}, full=full,
                              strict_family=(est == "TOTAL" and primary), sota={k: x[bi] for k, x in sota_full.items()})
            nconv += df.attrs["convergence_warnings"]
            fold_counts = df.attrs["fold_label_counts"]
            dfp, dft = df.attrs.pop("propensity"), df.attrs.pop("propensity_per_tf")               # (a DataFrame in attrs breaks pd.concat)
            for d_ in (df, dfp, dft):
                for k_, v_ in (("efficacy_status", tr["efficacy_status"]), ("decision_grade", tr["calibration"] == "CALIBRATED" and est != "WITHIN_STATE"), ("grade_note", "NOT_DECISION_GRADE: only 2 held-out folds ever contain both classes" if est == "WITHIN_STATE" else ""), ("calibration", tr["calibration"]),
                               ("variant_role", role[v]), ("gb_prior_label", PGB if v == "full" else PGB + "_noPanel"), ("variant", v), ("primary", primary), ("arm", arm),
                               ("estimand", est), ("context", ctx)):
                    d_.insert(0, k_, v_)
            allrows.append(df)
            allprop.append(dfp)
            alltf.append(dft)
            info.append(dict(estimand=est, arm=arm, variant=v, n_bank_excluded=len(rows_ex) if v == "excl" else 0, n_perts=len(tr["perts"]),
                             n_axis_genes=tr["n_axis_genes"], n_truth_testable_genes=tr["n_testable_genes"], n_dup_symbols_dropped=tr["n_dup_symbols"], n_dup_rows_dropped=tr["n_dup_rows_dropped"],
                             gene_id_unique=tr["gene_id_unique"], n_mapped_to_bank=int(okg.sum()),
                             mapped_fraction_of_testable=float(okg.mean()), n_pairs=tr["n_pairs"], calibration=tr["calibration"],
                             efficacy_status=tr["efficacy_status"], truth_files_sha256=tr["files"], truth_dir=tr["truth_dir"], truth_receipt_sha256=tr["truth_receipt_sha256"],
                             truth_phase=tr["truth_phase"], truth_shards=tr["truth_sharding"], draws_sha256=sha256(W), seconds=round(time.time() - t0, 1),
                             held_out_fold_label_counts=fold_counts))
        np.savez_compressed(f"{outd}/breadth_{ctx}_{est}_{arm}_context.npz", **save)
    D = pd.concat(allrows)
    D.to_csv(f"{outd}/breadth_{ctx}.tsv", sep="\t", index=False)
    pd.concat(allprop).to_csv(f"{outd}/breadth_{ctx}_propensity.tsv", sep="\t", index=False)
    pd.concat(alltf).to_csv(f"{outd}/breadth_{ctx}_propensity_per_tf.tsv", sep="\t", index=False)
    receipt(f"{outd}/receipt_breadth_{ctx}.json", dict(context=ctx, stage=stage, B=Bf, inputs=bank["files"], analyses=info, sset=sset, relaxed=relaxed, tier=tier, composites=cdiag, propensity_declaration=PROP_DECLARATION,
                                                       convergence_warnings=nconv, variant_roles=role, allow_uncalibrated=allow_uncal,
                                                       provisional=dict(BROAD_FRAC=BROAD_FRAC, MIN_RESP_BROAD=MIN_RESP_BROAD, MIN_T=MIN_T,
                                                                        COVER_FLOOR_FOLD=COVER_FLOOR_FOLD, CONF_FRAC=CONF_FRAC),
                                                       uncertainty="two-level refit bootstrap (perturbation cluster x gene, shared across models); Bonferroni over "
                                                                   "declared contrasts x primary metrics; the point estimate uses original genes/perts so a "
                                                                   "percentile interval can exclude it",
                                                       truth_self_check="per_pair.tsv.gz reproduces per_gene.tsv; truth hashes == receipt.json"))
    print(D[D.metric.isin(["rho_breadth", "auc_any", "yield_any@50"]) & (D.primary) & (D.kind == "model")].drop(columns=["variant_role", "efficacy_status"]).round(3).to_string(index=False))


# Pearson correlation of the residuals of (rx, ry) after regressing out Xc (rows idx); rx, ry, Xc are ranks (rank-then-resample, declared)
def _partial(rx, ry, Xc, idx):
    A = np.column_stack([np.ones(len(idx)), Xc[idx]])
    res = lambda v: v[idx] - A @ np.linalg.lstsq(A, v[idx], rcond=None)[0]
    a, b = res(rx), res(ry)
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


# [rho(P_B,b_B)-rho(P_K,b_B)]
def run_cross(stage, sset="T7", relaxed=False, tier="F1"):
    from scipy.stats import rankdata
    outd = outdir(stage, sset, relaxed, tier)
    Bf = STAGES[stage]["Bfit"]
    prim = {c: [a for a in ANALYSES[c] if a[2]][0] for c in ("BM", "K562")}
    Z = {c: np.load(f"{outd}/breadth_{c}_{prim[c][0]}_{prim[c][1]}_context.npz", allow_pickle=True) for c in prim}
    u = {c: dict(zip(np.char.upper(Z[c]["genes"].astype(str)), range(len(Z[c]["genes"])))) for c in Z}
    cg = sorted(set(u["BM"]) & set(u["K562"]))
    iB, iK = np.array([u["BM"][g] for g in cg]), np.array([u["K562"][g] for g in cg])
    pB, pK = np.char.upper(Z["BM"]["perts"].astype(str)), np.char.upper(Z["K562"]["perts"].astype(str))
    common = sorted(set(pB) & set(pK))
    def bpan(Zc, pl, ii, sub):
        m = np.isin(pl, sub)
        nT = Zc["T"][m].sum(0)[ii]
        return np.where(nT > 0, (Zc["R"][m] & Zc["T"][m]).sum(0)[ii] / np.maximum(nT, 1), np.nan)
    if len(common) >= MIN_CROSS_PERTS:
        bB, bK, restr = bpan(Z["BM"], pB, iB, common), bpan(Z["K562"], pK, iK, common), "common_perturbed_TFs"
    else:
        bB, bK, restr = Z["BM"]["b"][iB], Z["K562"]["b"][iK], "full_panels_NOT_DECISION_GRADE"
    Fc = np.column_stack([Z["BM"]["F1"][iB], Z["K562"]["F1"][iK]])
    jobs = [(v, m) for v in ("full", "excl") for m in GB_PRIORS + GEN_T7 + ["regvelo", "velorama"] if f"{v}|{m}" in Z["BM"].files and f"{v}|{m}" in Z["K562"].files]
    a_each = ALPHA / max(len(jobs), 1)
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 77]))
    rows = []
    for v, m in jobs:
        PB, PK = Z["BM"][f"{v}|{m}"][iB], Z["K562"][f"{v}|{m}"][iK]
        ok = np.isfinite(np.column_stack([PB, PK, bB, bK, Fc])).all(1)
        rk = lambda a: rankdata(a[ok])
        RB, RK, rb, rkk = rk(PB), rk(PK), rk(bB), rk(bK)
        Xc = np.column_stack([rankdata(Fc[ok][:, i]) for i in range(Fc.shape[1])])
        st = lambda idx: (lambda a, b, c, d: (a, b, c, d, (a - b) - (c - d)))(_partial(RB, rb, Xc, idx), _partial(RK, rb, Xc, idx),
                                                                           _partial(RB, rkk, Xc, idx), _partial(RK, rkk, Xc, idx))
        n = int(ok.sum())
        pt = st(np.arange(n))
        dr = np.array([st(rng.integers(0, n, n)) for _ in range(Bf)])
        lo, hi, meth = ci(dr[:, 4], pt[4], a_each)
        rows.append(dict(variant=v, prior=m, n_common_genes=n, n_common_perts=len(common), panel_restriction=restr, controls="rank-partial on F1_BM + F1_K562",
                         decision_grade=restr == "common_perturbed_TFs", rho_PB_bB=pt[0], rho_PK_bB=pt[1], rho_PB_bK=pt[2], rho_PK_bK=pt[3], interaction=pt[4],
                         lo=lo, hi=hi, ci_method=meth, alpha_each=a_each, B=Bf))
    pd.DataFrame(rows).to_csv(f"{outd}/breadth_CROSS.tsv", sep="\t", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


# selftest (synthetic only)
# verbatim copies of gb_catbank.row_consts / prior_sums / loo_prior / members_sha (gb_catbank.py:454-479, 45-47) so the selftest needs no data import
class _LocalCB:
    @staticmethod
    def members_sha(members):
        return hashlib.sha256("\n".join(members).encode()).hexdigest()

    @staticmethod
    def row_consts(x, T):
        mu, sd = np.full(len(x), np.nan), np.full(len(x), np.nan)
        for r in range(len(x)):
            v = x[r, T]
            if np.isfinite(v).sum() > 2 and np.nanstd(v) > 0:
                mu[r], sd[r] = np.nanmean(v), np.nanstd(v)
        return mu, sd

    @staticmethod
    def prior_sums(Z):
        f = np.isfinite(Z)
        return np.where(f, Z, 0.0).sum(0), f.sum(0)

    @staticmethod
    def loo_prior(Z, S, C, r, min_contrib=2):
        if r is None:
            s, c = S, C
        else:
            f = np.isfinite(Z[r])
            s, c = S - np.where(f, Z[r], 0.0), C - f
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c >= min_contrib, s / np.maximum(c, 1), np.nan)


# Synthetic catbank directory in the actual gb_catbank layout
def _write_catbank(dirp, system, rng, kind, v, nb=14, mutate=None, leak_y=None, sota=False):
    L = _LocalCB
    n = len(v)
    axis = np.array([f"g{i}" for i in range(n)])
    extra = (np.arange(n) % 15 == 7)
    Tm = ~extra
    members = [f"TF{i}" for i in range(nb)]
    names = GB_PRIORS + GEN_T7
    coef = {m: (1.0 if kind == "C" else {"P_pairU": 1.0, "P_pairM": 0.3}.get(m, 0.0)) for m in names}
    sig = 0.8 if kind == "C" else 1.0
    X = {m: coef[m] * v[None, :] + sig * rng.normal(size=(nb, n)) for m in names}
    if leak_y is not None:                         # focal row TF0 of P_pairU contains the labels (on T genes)
        X["P_pairU"][0, Tm] = 10.0 * (leak_y.astype(float) - 0.5) + 0.1 * rng.normal(size=int(Tm.sum()))
    for vv, m in (("pairU", "P_pairU"), ("pairM", "P_pairM")):     # root diagnostics: min(H1, H2) == X exactly
        d = rng.normal(size=X[m].shape)
        H1, H2 = X[m] + np.where(d > 0, d, 0.0), X[m] + np.where(d <= 0, -d, 0.0)
        if mutate == "stale_shard" and vv == "pairU":
            H2 = H2 + 0.5
        X[f"H1_{vv}"], X[f"H2_{vv}"], X[f"Pmean_{vv}"] = H1, H2, (H1 + H2) / 2
    X["regvelo"] = rng.normal(size=(nb, n))
    arrays = {"members": np.array(members[::-1] if mutate == "members_order" else members), "axis": axis}
    for sset, ms in (("T7", [k for k in X if k != "regvelo"]), ("T7+regvelo", names + ["regvelo"])):
        arrays.update({f"{sset}|genes": axis[Tm], f"{sset}|relaxed_genes": axis, f"{sset}|relaxed_is_extra": extra})
        for m in ms:
            mu, sd = L.row_consts(X[m], Tm)
            Zr = (X[m] - mu[:, None]) / sd[:, None]
            Z = Zr[:, Tm]
            S, C = L.prior_sums(Z)
            Sr, Cr = L.prior_sums(Zr)
            for k, val in dict(mu=mu, sd=sd, Z=Z, S=S, C=C, Sr=Sr, Cr=Cr, Zx=Zr[:, extra], Pfull=L.loo_prior(Z, S, C, None),
                               Pfull_relaxed=L.loo_prior(None, Sr, Cr, None)).items():
                if mutate == "bad_pfull" and m == "P_pairM" and k == "Pfull":
                    val = val + 1.0
                if mutate == "drop_key" and m == "P_pairM" and k == "Zx":
                    continue
                arrays[f"{sset}|{m}|{k}"] = val
    os.makedirs(dirp, exist_ok=True)
    pf = f"{dirp}/{system}_priors.npz"
    np.savez(pf, **arrays)
    bank_sha = L.members_sha(members)
    json.dump(dict(system=system, members=members, sha256_members=bank_sha, detection={m: 0.5 for m in members}, n_members=nb), open(f"{dirp}/{system}_bank.json", "w"))
    json.dump(dict(output_sha256=("0" * 64 if mutate == "bad_receipt" else sha256(pf)), config=dict(bank_sha256=bank_sha),
                   sets={"T7": dict(status="OK"), "T7+regvelo": dict(status="OK"), "T7+velorama": dict(status="FAILED: empty")}),
              open(f"{dirp}/{system}_priors.receipt.json", "w"))
    f0, f1, f1x = list(F0_COLS), F0_COLS + F1_EXTRA, F0_COLS + F1_EXTRA + F1X_EXTRA
    F = pd.DataFrame(rng.normal(size=(n, len(f1x))), index=pd.Index(axis, name="gene"), columns=f1x)
    if mutate == "bad_flag":
        F.iloc[0, len(f1) - 1] = np.nan
    F["F0_complete"] = np.isfinite(F[f0].to_numpy()).all(1)
    F["F1_complete"] = np.isfinite(F[f1].to_numpy()).all(1) | (mutate == "bad_flag")
    F["F1x_complete"] = np.isfinite(F[f1x].to_numpy()).all(1)
    F.to_csv(f"{dirp}/{system}_features.tsv", sep="\t")
    json.dump(dict(output_sha256=sha256(f"{dirp}/{system}_features.tsv"),
                   config=dict(F0=f0, F1=(f1[:-1] + ["oops"] if mutate == "f1_name" else f1), F1x=f1x)), open(f"{dirp}/{system}_features.receipt.json", "w"))
    if sota:                                                        # gb_catbank SOTA layout: members, genes (= T7 axis), methods, {m}|{mu,sd,Z,S,C,Pfull}
        sarr = {"members": np.array(members), "genes": axis[Tm], "methods": np.array(SOTA_METHODS)}
        for j, m in enumerate(SOTA_METHODS):
            Xs = (0.5 if j % 2 == 0 else 0.0) * v[None, :] + rng.normal(size=(nb, n))
            mu, sd = L.row_consts(Xs, Tm)
            Z = ((Xs - mu[:, None]) / sd[:, None])[:, Tm]
            S, C = L.prior_sums(Z)
            sarr.update({f"{m}|{k}": val for k, val in dict(mu=mu, sd=sd, Z=Z, S=S, C=C, Pfull=L.loo_prior(Z, S, C, None)).items()})
        sf = f"{dirp}/{system}_sota_priors.npz"
        np.savez(sf, **sarr)
        json.dump(dict(output_sha256=sha256(sf), bank_sha256=bank_sha, priors_sha256=sha256(pf)), open(f"{dirp}/{system}_sota_priors.receipt.json", "w"))
    return dict(axis=axis, Tm=Tm, members=members)


def _use_dir(d):
    globals().update(BANK_JSON=d + "/{s}_bank.json", PRIORS_NPZ=d + "/{s}_priors.npz", PRIORS_RECEIPT=d + "/{s}_priors.receipt.json",
                     FEATURES_TSV=d + "/{s}_features.tsv", FEATURES_RECEIPT=d + "/{s}_features.receipt.json",
                     SOTA_NPZ=d + "/{s}_sota_priors.npz", SOTA_RECEIPT=d + "/{s}_sota_priors.receipt.json")


# Synthetic gb_breadth_truth v2 outputs written by its own build_outputs / PairWriter
def _write_truth(dirp, ctx, rng, v_T, genes_T, perts, est_arms, calib="CALIBRATED", n_untestable=20, dup_pairs=0):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import gb_breadth_truth as BT
    os.makedirs(dirp, exist_ok=True)
    genes_all = list(genes_T) + [f"zz{i}" for i in range(n_untestable)]
    for i in range(dup_pairs):                                             # duplicated symbols (10x): rows 2i and 2i+1 share a symbol, distinct gene_id
        genes_all[2 * i + 1] = genes_all[2 * i]
    G, Gt = len(genes_all), len(genes_T)
    vv = np.r_[v_T, np.zeros(n_untestable)]
    results, w = {}, BT.PairWriter(f"{dirp}/per_pair.tsv.gz")
    for est in {e for e, _ in est_arms}:
        results[est] = {}
        for p in perts:
            Tt = (rng.random(G) < 0.9) & (np.arange(G) < Gt)
            resp = Tt & (rng.random(G) < 1 / (1 + np.exp(-(1.5 * vv - 1.5))))
            conf = Tt & ~resp & (rng.random(G) < 0.7)
            status = np.zeros(G, np.int8)
            status[Tt] = 3
            status[conf] = 2
            status[resp] = 1
            lfc = np.where(resp, np.sign(rng.normal(size=G)) * (0.5 + np.abs(rng.normal(size=G))), 0.3 * rng.normal(size=G))
            lA, lB = lfc + 0.1 * rng.normal(size=G), lfc + 0.1 * rng.normal(size=G)
            cons = resp | (Tt & (rng.random(G) < 0.3))
            res = dict(lfc=lfc, lfcA=lA, lfcB=lB, z=lfc, p=np.full(G, 0.01), se=np.full(G, 0.2), n1=np.full(G, 100.0), n0=np.full(G, 100.0), nu=6)
            o = dict(status=status, T=Tt, e=np.where(Tt, lA * lB, 0.0), e_legacy=np.zeros(G), q=np.full(G, 0.02), cons=cons, strict=conf, se=np.full(G, 0.2),
                     res=res, n_strata=1, lo=lfc - 0.1, hi=lfc + 0.1)
            results[est][p] = o
            w.add(est, p, genes_all, o)
    w.close()
    arms = {arm: list(perts) for _, arm in est_arms}
    BT.build_outputs(dirp, np.asarray(genes_all), np.arange(G), np.ones(G), np.arange(G), list(perts), results, arms, {},
                     lambda est, p: dict(efficacy_status="EFFICACY_UNVERIFIED", null_calibration=calib))
    _write_truth_receipt(dirp, {e: calib for e, _ in est_arms})


def _write_truth_receipt(dirp, calib, phase="COMPLETE"):
    json.dump(dict(phase=phase, null_gate_result={e: dict(label=c) for e, c in calib.items()},
                   output_sha256={fn: sha256(f"{dirp}/{fn}") for fn in TRUTH_HASHED}), open(f"{dirp}/receipt.json", "w"))


# F0 / F1 / F1x names parsed (ast, no execution) from the sibling sources: gb_decomp.feats (F0) and gb_catbank.cmd_features (f1cols / f1xcols)
def _sibling_feature_lists():
    import ast
    here = os.path.dirname(os.path.abspath(__file__))

    def lst(node):
        return [e.value for e in node.elts]
    f0 = f1 = f1x = None
    for n in ast.walk(ast.parse(open(f"{here}/gb_catbank.py").read())):
        if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and isinstance(n.value, ast.BinOp) and isinstance(n.value.right, ast.List):
            if n.targets[0].id == "f1cols":
                f1 = lst(n.value.right)
            if n.targets[0].id == "f1xcols":
                f1x = lst(n.value.right)
    for fn in ast.walk(ast.parse(open(f"{here}/gb_decomp.py").read())):
        if isinstance(fn, ast.FunctionDef) and fn.name == "feats":
            for c in ast.walk(fn):
                if isinstance(c, ast.Call) and isinstance(c.func, ast.Attribute) and c.func.attr == "DataFrame":
                    f0 = [k.arg for k in c.args[0].keywords]
    return f0, f1, f1x


def selftest():
    global DELTA_AUROC, AP_LIFT_DELTA, F1_EXTRA, ANALYSES, BTRUTH, LAD, B_INFO
    tmp = tempfile.mkdtemp()
    rng = np.random.default_rng(1)
    from sklearn.metrics import average_precision_score, roc_auc_score
    f0s, f1s, f1xs = _sibling_feature_lists()
    assert f0s == F0_COLS and f1s == F1_EXTRA and f1xs == F1X_EXTRA, ("F0/F1/F1x differ from the sibling sources", f0s, f1s, f1xs)
    print("ok 0: F0 / F1 (8 spec columns) / F1x names equal the names parsed from gb_decomp.feats and gb_catbank.cmd_features")
    _CB["m"] = _LocalCB
    F1_EXTRA = ["raw_det_u", "mu_det", "mu_dynamics", "abs_spearman_mu_dpt", "mu_logvar"]     # speed only (names verified above; keeps the REF columns)
    # 1. weighted vectorised metrics == sklearn
    n = 400
    y = rng.random(n) < 0.2
    s = np.round(rng.normal(size=n), 1) + y * 0.3
    c = rng.poisson(1, size=(5, n)).astype(float)
    a, ap, pv = wraw(y, s, c)
    for i in range(5):
        assert abs(a[i] - roc_auc_score(y, s, sample_weight=c[i])) < 1e-12
        assert abs(ap[i] - average_precision_score(y, s, sample_weight=c[i])) < 1e-12
        assert abs(pv[i] - (c[i] * y).sum() / c[i].sum()) < 1e-12
    assert abs(pt_metrics(y, np.zeros(n))["AUROC"] - 0.5) < 1e-12, "constant prior must give AUROC .5"
    print("ok 1: weighted metrics == sklearn; constant prior AUROC .5")
    # 2. margin rules
    assert reading(0.02, 0.005, 0.03, 0.01, True) == ">"
    assert reading(0.02, -0.001, 0.05, 0.01, True) == "?", "positive point but lower bound < 0 -> UNRESOLVED"
    assert reading(0.02, 0.005, 0.03, 0.01, False) == "?", "no majority of blocks"
    assert reading(0.001, -0.005, 0.006, 0.01, True) == "≈"
    assert reading(0.001, -0.05, 0.06, 0.01, True) == "?", "wide CI -> UNRESOLVED, never equivalence"
    assert reading(-0.03, -0.05, -0.02, 0.01, True) == "<"
    assert ci(np.random.default_rng(0).normal(size=50), 0.0, 0.05 / 24)[2] == "normal_approx" and ci(np.random.default_rng(0).normal(size=5000), 0, 0.05)[2] == "percentile"
    print("ok 2: margin rules")
    # 3. case logic on fabricated readings
    g = GEN_T7
    base = {f"PGB-gen:{x}": "?" for x in g}
    base.update({f"G-gen:{x}": "?" for x in g})
    base.update({k: "?" for k in ("G-PGB",) + D_KEYS})
    B_ = dict(base, **{f"PGB-gen:{x}": ">" for x in g}, **{"PGB|F1": ">", "PGB|ALLGEN": ">", "G|PGB": "≈", "G|ALLGEN+PGB": "≈"})
    C_ = dict(base, **{f"PGB-gen:{x}": "≈" for x in g})
    A_ = dict(C_, **{"G-PGB": ">"}, **{f"G-gen:{x}": ">" for x in g})
    D_ = dict(C_, **{k: "≈" for k in D_KEYS})
    D_missing = dict(D_, **{"G|F1": "?"})                               # D needs the explicit F1+G model contrast
    A_noG = dict(A_, **{f"G-gen:{g[0]}": "?"})                          # A needs G > each generic
    Lt_ = dict(base, **{f"PGB-gen:{g[0]}": "<", f"PGB-gen:{g[1]}": "≈", f"PGB-gen:{g[2]}": "≈"})
    assert [case_label(x, g, "functional")[0] for x in (B_, C_, A_, D_, base, D_missing, A_noG, Lt_)] == \
        ["B", "C", "A", "D", "UNRESOLVED", "C", "C", "GENERIC_BEATS_GB"]
    assert case_label(A_, g, "regulon") == ("C", True), "A is labelled only for the multi-TF functional family"
    assert case_label(C_, [], "functional")[0] == "UNRESOLVED", "empty generic list must not be a vacuous case"
    print("ok 3: case logic A/B/C/D/GENERIC_BEATS_GB/UNRESOLVED (D needs F1+G; A needs G>each generic; A functional-only; empty list)")
    # 4. loader against a synthetic catbank in the actual layout + fail-closed on every mutation
    nT = 600
    v0 = rng.normal(size=nT + nT // 14 + 2)
    cbdir = f"{tmp}/cb"
    _use_dir(cbdir)
    _write_catbank(cbdir, "SYN", rng, "B", v0)
    bank = load_bank("SYN")
    assert len(bank["genes"]) == int((~(np.arange(len(v0)) % 15 == 7)).sum()) and "P_pairU@root0" in bank["ZSC"] and len(bank["extra"]) == 6
    assert load_bank("SYN", relaxed=True, roots=True)["genes"].shape[0] == len(v0)
    assert "regvelo" in load_bank("SYN", sset="T7+regvelo")["gens"] and not load_bank("SYN", sset="T7+regvelo")["extra"]
    tf1 = load_bank("SYN", tier="F1x")
    assert tf1["f1"][-3:] == F1X_EXTRA
    for mut in ("members_order", "drop_key", "bad_flag", "f1_name", "stale_shard", "bad_pfull", "bad_receipt"):
        _write_catbank(f"{tmp}/neg_{mut}", "NEG", np.random.default_rng(2), "B", v0, mutate=mut)
        _use_dir(f"{tmp}/neg_{mut}")
        try:
            load_bank("NEG")
        except SystemExit as e:
            assert str(e).startswith("BLOCKED_BY_SPEC_LOADER"), (mut, str(e))
        else:
            raise AssertionError(f"loader did not fail closed on mutation {mut}")
    _use_dir(cbdir)
    try:
        load_bank("SYN", sset="T7+velorama")
    except SystemExit as e:
        assert "status != OK" in str(e)
    else:
        raise AssertionError("failed set must be refused")
    _use_dir(cbdir)
    print("ok 4: loader reads the catbank layout (T7 / T7+regvelo / relaxed / roots / F1x tier) and fails closed on 7 mutations + a failed set")
    # 5. evaluator: leave-focal-out via catbank.loo_prior, refit count, per-resample prevalence,
    # info models at pilot B, FAILED endpoint
    lg = lambda x: 1 / (1 + np.exp(-x))
    yv = rng.random(len(v0)) < lg(1.6 * v0 - 1.2)
    msk0 = ~(np.arange(len(v0)) % 15 == 7)
    _write_catbank(f"{tmp}/cbleak", "LEAK", rng, "B", np.zeros(len(v0)), leak_y=yv[msk0])   # no latent signal: only the leak
    _use_dir(f"{tmp}/cbleak")
    bk = load_bank("LEAK")
    Tg = bk["genes"]
    yT = yv[msk0]
    ep = dict(name="E", fam="regulon", tf="TF0", genes=Tg, ev=np.ones(len(Tg), bool), y=yT, G=rng.normal(size=len(Tg)))
    Wp = gene_draws(tmp, "T", len(Tg), 6)
    r_lo = eval_endpoint(ep, bk, Wp, 2, 6)
    r_out = eval_endpoint(dict(ep, tf="NOTINBANK"), bk, Wp, 2, 6)
    assert r_lo["row"]["lo_used"] and r_lo["row"]["C|raw|P_pairU|AUROC"] < 0.6, r_lo["row"]["C|raw|P_pairU|AUROC"]
    assert (not r_out["row"]["lo_used"]) and r_out["row"]["C|raw|P_pairU|AUROC"] > r_lo["row"]["C|raw|P_pairU|AUROC"] + 0.05
    print("ok 5a: leave-focal-out prior used for the regulator endpoint (leaky-row AUROC out-of-bank %.3f vs leave-out %.3f)" %
          (r_out["row"]["C|raw|P_pairU|AUROC"], r_lo["row"]["C|raw|P_pairU|AUROC"]))
    _use_dir(cbdir)
    bank = load_bank("SYN")
    Tg = bank["genes"]
    yT = yv[msk0]
    ep = dict(name="E", fam="regulon", tf="TF0", genes=Tg, ev=np.ones(len(Tg), bool), y=yT, G=rng.normal(size=len(Tg)))
    NFIT[0] = 0
    res = eval_endpoint(ep, bank, Wp, 3, 6)
    nmod = 2 + len(bank["gens"]) + 6 + 3
    assert NFIT[0] == nmod * K * (1 + 3), f"bootstrap must REFIT: {NFIT[0]} != {nmod * K * 4}"
    print("ok 5b: refit count", NFIT[0], "== models*folds*(1+B)")
    Wm = np.load(Wp, mmap_mode="r")
    c0 = np.asarray(Wm[0]).astype(float)
    pri_u = cb().loo_prior(*bank["ZSC"][PGB], 0, MIN_CONTRIB)
    assert abs(res["dr_raw"]["C|raw|P_pairU"][1][0] - average_precision_score(yT, pri_u, sample_weight=c0) / ((c0 * yT).sum() / c0.sum())) < 1e-9
    assert len({round(float((np.asarray(Wm[b]) * yT).sum() / np.asarray(Wm[b]).sum()), 6) for b in range(6)}) > 1
    print("ok 5c: AP lift uses each resample's own prevalence")
    B_INFO = 2
    res2 = eval_endpoint(ep, bank, Wp, 4, 6)
    assert len(res2["dr_info"]["C|fit|F0"][0]) == 2 and len(res2["dr_fit"]["C|fit|F1"][0]) == 4 and "C|fit|F1+G" in res2["dr_fit"] and "C|fit|F1+pearson" in res2["dr_info"]
    B_INFO = 200
    _S.update(eps=[ep], bank=bank, W=Wp, Bfit=2, Braw=2)
    _orig = eval_endpoint
    globals()["eval_endpoint"] = lambda *a, **k: (_ for _ in ()).throw(PointFitFailure("forced"))
    try:
        i_, r_ = _work(0)
    finally:
        globals()["eval_endpoint"] = _orig
    assert (not r_["ok"]) and r_["row"]["status"].startswith("FAILED"), "a point-fit failure must mark the endpoint FAILED, not kill the pool"
    print("ok 5d: information-only models run at pilot B only; point-fit failure -> endpoint FAILED (pool survives)")
    # 6. planted worlds end-to-end (margins scaled to the synthetic n)
    DELTA_AUROC, AP_LIFT_DELTA = 0.04, 0.5
    metaf = lambda blk: dict(system=blk, block=blk, stage="t", Bfit=24, Braw=24, Binfo=24, limit=None, sset="T7", relaxed=False, roots=True, bank_tag="", tier="F1",
                             script_sha256="x")
    for kind, want in (("B", "B"), ("C", "C")):
        parts = []
        for bi_ in range(3):
            d = f"{tmp}/w{kind}{bi_}"
            rg = np.random.default_rng(100 * (kind == "C") + bi_)
            vv = rg.normal(size=nT + nT // 14 + 2)
            _write_catbank(d, f"BLK{bi_}", rg, kind, vv)
            _use_dir(d)
            bk = load_bank(f"BLK{bi_}")
            msk = ~(np.arange(len(vv)) % 15 == 7)
            eps = [dict(name=f"E{bi_}", fam="regulon", tf="TF0", genes=bk["genes"], ev=np.ones(msk.sum(), bool), G=rg.normal(size=int(msk.sum())),
                        y=rg.random(int(msk.sum())) < lg(1.6 * vv[msk] - 1.2))]
            Wp = gene_draws(tmp, f"{kind}{bi_}", len(bk["genes"]), 24)
            acc = Acc(eps)
            for e in eps:
                acc.add(e, eval_endpoint(e, bk, Wp, 24, 24))
            f = f"{tmp}/bs{kind}{bi_}.npz"
            acc.dump(f, metaf(f"BLK{bi_}"))
            parts.append(load_parts_from(f))
        M, Cn, Lb, _ = aggregate_blocks(parts, True)
        lab = Lb[(Lb.family == "regulon") & (Lb.metric == "AUROC")].iloc[0]
        rd = Cn[(Cn.metric == "AUROC") & Cn.group.isin(["case"])].set_index("contrast")
        print(f"  world {kind}: case={lab.case} factorial_C={lab.factorial_C}")
        print(rd[["point", "ci_lo", "ci_hi", "reading"]].round(3).to_string())
        assert lab.case == want, f"planted world {kind} expected case {want}, got {lab.case}"
        assert len(Cn[(Cn.metric == "AUROC") & Cn.contrast.str.startswith("root|")]) == 9, "root diagnostics contrasts missing"
        refc = Cn[(Cn.metric == "AUROC") & (Cn.group == "ref")]
        assert len(refc) == 9 and np.allclose(refc.alpha_each, 0.05 / (9 * N_FAM)) and set(refc.contrast) == {f"REF|PGB-{n_}" for n_ in REF_NAMES}, "REF contrast family"
        hs = M[(M.contrast == "HINDSIGHT_UPPER_BOUND") & (M.metric == "AUROC")]
        refm = M[(M.metric == "AUROC") & M.contrast.str.startswith("MODEL:C|raw|REF_")]
        assert len(refm) == 9 and len(hs) == 1 and abs(hs.point.iloc[0] - refm.point.max()) < 1e-12 and "ci_lo" not in hs.dropna(axis=1).columns, "hindsight row"
        assert rd.loc["G|F1"].ci_type.startswith("wider") and "wider" in Cn[Cn.contrast == "PGB|F1"].ci_type.iloc[0], "fitted '≈' must use the wider interval"
    # aggregator refusals
    bad = [dict(p, meta=dict(p["meta"])) for p in parts]
    bad[1]["meta"]["roots"] = False
    for mk, part_list in (("mixed roots", bad),):
        try:
            aggregate_blocks(part_list, True)
        except SystemExit as e:
            assert "refusing to mix" in str(e)
        else:
            raise AssertionError("aggregator must refuse mixed runs")
    miss = [dict(p, fams={k: dict(v) for k, v in p["fams"].items()}) for p in parts]
    E0 = miss[2]["fams"]["regulon"]
    E0["kr"] = [k for k in E0["kr"] if k != "C|raw|G"]
    E0["raw"] = E0["raw"][[i for i, k in enumerate(parts[2]["fams"]["regulon"]["kr"]) if k != "C|raw|G"]]
    E0["pt"] = E0["pt"][[i for i, k in enumerate(parts[2]["fams"]["regulon"]["kr"] + parts[2]["fams"]["regulon"]["kf"] + parts[2]["fams"]["regulon"]["ki"]) if k != "C|raw|G"]]
    try:
        aggregate_blocks(miss, True)
    except SystemExit as e:
        assert "lack keys" in str(e)
    else:
        raise AssertionError("aggregator must refuse a block missing a declared key")
    print("ok 6: planted target-only signal -> Case B; shared-latent priors -> Case C; wider interval for fitted contrasts; aggregator refuses mixed runs / missing keys")
    # 6b. the real-BM NA case: only a few perturbations respond, so several held-out folds have no
    # responders (constant b, no positive class)
    rg = np.random.default_rng(11)
    n, P = 300, 10
    u = rg.normal(size=n)
    Tm_ = np.ones((P, n), bool)
    Rm_ = np.zeros((P, n), bool)
    Rm_[:3] = rg.random((3, n)) < 1 / (1 + np.exp(-(2.0 * u - 1.0)))      # only TF0..TF2 ever respond
    truth = dict(perts=[f"TF{i}" for i in range(P)], T=Tm_, R=Rm_, E=np.where(Rm_, 1.0, 0.1), CN=Tm_ & ~Rm_)
    cols_ = {"a": rg.normal(size=n), "b": rg.normal(size=n)}
    pri_ = {PGB: u + 0.7 * rg.normal(size=n), "P_pairM": rg.normal(size=n), "P_cellU": rg.normal(size=n), "P_cellM": rg.normal(size=n),
            "pearson": rg.normal(size=n), "genie3": rg.normal(size=n), "grnboost2": rg.normal(size=n)}
    Wb = gene_draws(tmp, "BRna", n, 2)
    PWb = np.random.default_rng(3).multinomial(P, np.full(P, 1.0 / P), size=2)
    dfb = breadth_core(np.array([f"g{i}" for i in range(n)]), cols_, pri_, truth, 2, Wb, PWb)
    fc = pd.DataFrame(dfb.attrs["fold_label_counts"])
    assert (fc.n_distinct_b == 1).any() and (fc.any1 == 0).any(), "the synthetic case must contain held-out folds without responders"
    g = lambda q, kind="model": dfb[(dfb.name == f"raw:{PGB}") & (dfb.metric == q) & (dfb.kind == kind)].point.iloc[0]
    assert np.isfinite(g("rho_breadth")) and g("rho_breadth") > 0.2 and np.isfinite(g("auc_any")) and g("auc_any") > 0.6, "fold-stratified metrics must be finite"
    assert g("rho_breadth_folds_used") < 5 and g("rho_breadth_folds_used") >= 1, "folds without responders are skipped and counted"
    assert np.isfinite(g("rho_breadth_pooled_LEGACY")) and np.isfinite(g("yield_any@50_stratified"))
    print("ok 6b: NA case reproduced (folds without responders: %s); fold-stratified rho/auc finite (rho %.3f from %d folds, auc_any %.3f); legacy pooled kept" %
          (fc[fc.any1 == 0].fold.tolist(), g("rho_breadth"), int(g("rho_breadth_folds_used")), g("auc_any")))
    # 6c. base rate varies by fold and F1 is as informative as the raw prior: fold-stratified F1 ~
    # raw, legacy pooled F1 < raw (fold-specific calibration)
    rg = np.random.default_rng(21)
    n, P = 400, 10
    u = rg.normal(size=n)
    off = np.array([-0.3] * 5 + [-5.0] * 5)                                      # responsive vs silent perts -> held-out base rate differs by fold
    Tm_ = np.ones((P, n), bool)
    Rm_ = rg.random((P, n)) < 1 / (1 + np.exp(-(1.5 * u[None, :] + off[:, None])))
    truth = dict(perts=[f"TF{i}" for i in range(P)], T=Tm_, R=Rm_, E=np.where(Rm_, 1.0, 0.1), CN=Tm_ & ~Rm_)
    sig = lambda: u + 0.7 * rg.normal(size=n)
    cols_ = {"a": sig(), "b": rg.normal(size=n)}
    pri_ = {PGB: sig(), "P_pairM": rg.normal(size=n), "P_cellU": rg.normal(size=n), "P_cellM": rg.normal(size=n),
            "pearson": rg.normal(size=n), "genie3": rg.normal(size=n), "grnboost2": rg.normal(size=n)}
    dfc = breadth_core(np.array([f"g{i}" for i in range(n)]), cols_, pri_, truth, 1, gene_draws(tmp, "x6c", n, 1), np.ones((1, P), int))
    gc = lambda nm, q: dfc[(dfc.name == nm) & (dfc.metric == q) & (dfc.kind == "model")].point.iloc[0]
    assert abs(gc("F1", "auc_any") - gc(f"raw:{PGB}", "auc_any")) < 0.05, "stratified F1 must ~ raw when equally informative"
    assert gc("F1", "auc_any_pooled_LEGACY") < gc(f"raw:{PGB}", "auc_any_pooled_LEGACY") - 0.05, "pooled F1 must be biased below raw"
    assert gc("F1", "auc_any") > gc("F1", "auc_any_pooled_LEGACY") + 0.1
    # combination rules + NaN-prediction failure
    rec = lambda a, w, f=False: {"auc_any": (a, w, f)}
    assert abs(combine_folds([rec(0.6, 10), rec(0.8, 30), rec(np.nan, 0)])["auc_any"] - 0.75) < 1e-12, "pair-weighted mean; folds without positives contribute nothing"
    assert np.isnan(combine_folds([rec(0.6, 10), rec(0.8, 30, True)])["auc_any"]), "a failed fold makes the metric NaN"
    pr_ = {q: np.array([0.1, np.nan, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]) for q in ("b", "bur", "any", "broad")}
    tg_ = {"b": np.arange(8.0), "bur": np.arange(8.0), "any": np.array([0, 1, 0, 1, 0, 1, 0, 1.0]), "broad": np.full(8, np.nan), "resp": np.array([0, 1, 0, 1, 0, 1, 0, 1.0])}
    fm = fold_metrics(pr_, tg_, np.arange(8))
    assert fm["auc_any"][2] and np.isnan(fm["auc_any"][0]) and fm["rho_breadth"][2], "NaN prediction on a labelled row must fail the metric, not be dropped"
    assert not fm["auc_broad"][2] and np.isnan(fm["auc_broad"][0]), "no labelled rows is undefined, not a failure"
    print("ok 6c: base-rate-varying folds: stratified F1 %.3f ~ raw %.3f, LEGACY pooled F1 %.3f < raw %.3f; combination weights; NaN-prediction rows fail the metric" %
          (gc("F1", "auc_any"), gc(f"raw:{PGB}", "auc_any"), gc("F1", "auc_any_pooled_LEGACY"), gc(f"raw:{PGB}", "auc_any_pooled_LEGACY")))
    # 6d. propensity units: ties broken by gene-symbol rank (not input order)
    s_ = np.zeros(120)
    lab_ = np.where(np.arange(120) % 3 == 0, np.nan, (np.arange(120) % 2).astype(float))
    tie_ = np.random.default_rng(0).permutation(120)
    o1 = prop_one(lab_, s_, tie_)
    assert np.array_equal(np.lexsort((tie_, -s_))[:50], np.argsort(tie_)[:50]) and o1["FB50_resp"] + o1["FB50_nonresp"] + o1["FB50_unlab"] == 50
    assert o1["FB50_unlab"] == np.isnan(lab_[np.argsort(tie_)[:50]]).sum(), "unlabeled slots are counted, not refilled"
    Vs = np.random.default_rng(1).normal(size=(60, 3))
    Vs[:7, 1] = np.nan
    lo_, hi_, q_, nv_, nB_ = simul_ci(Vs, np.zeros(3))
    assert nv_.tolist() == [60, 53, 60] and nB_ == 60 and np.isfinite(lo_).all() and (hi_ > lo_).all(), "available-case: only the column with NaN draws loses draws"
    Vz = np.c_[Vs[:, 0], np.zeros(60), Vs[:, 2]]
    lz, hz, qz, nz, _ = simul_ci(Vz, np.zeros(3))
    assert np.isnan(lz[1]) and np.isfinite(lz[[0, 2]]).all(), "zero-SD column: NaN interval, excluded from the max"
    rg6 = np.random.default_rng(31)                                # adding Tier-1 SOTA comparators must leave every existing row identical
    n6, P6 = 300, 10
    u6 = rg6.normal(size=n6)
    T6 = np.ones((P6, n6), bool)
    R6 = rg6.random((P6, n6)) < 1 / (1 + np.exp(-(1.5 * u6[None, :] - 1.0)))
    tr6 = dict(perts=[f"TF{i}" for i in range(P6)], T=T6, R=R6, E=np.where(R6, 1.0, 0.1), CN=T6 & ~R6)
    c6 = {"a": rg6.normal(size=n6), "b": rg6.normal(size=n6)}
    p6 = {PGB: u6 + 0.7 * rg6.normal(size=n6), "P_pairM": rg6.normal(size=n6), "P_cellU": rg6.normal(size=n6), "P_cellM": rg6.normal(size=n6),
          "pearson": rg6.normal(size=n6), "genie3": rg6.normal(size=n6), "grnboost2": rg6.normal(size=n6)}
    W6, PW6 = gene_draws(tmp, "x6d", n6, 3), np.random.default_rng(4).multinomial(P6, np.full(P6, 0.1), size=3)
    g6 = np.array([f"g{i}" for i in range(n6)])
    d0 = breadth_core(g6, c6, p6, tr6, 3, W6, PW6)
    d1 = breadth_core(g6, c6, p6, tr6, 3, W6, PW6, sota={m_: rg6.normal(size=n6) for m_ in SOTA_METHODS})
    k6 = ["kind", "name", "metric"]
    j6 = d0[(d0.kind == "model")].merge(d1[d1.kind == "model"], on=k6, suffixes=("_0", "_1"))
    assert len(j6) == (d0.kind == "model").sum() and np.allclose(j6.point_0, j6.point_1, equal_nan=True), "existing model rows unchanged by adding SOTA comparators"
    j7 = d0[(d0.kind == "contrast") & d0.declared].merge(d1[(d1.kind == "contrast") & d1.declared], on=k6 + ["family"], suffixes=("_0", "_1"))
    assert len(j7) == ((d0.kind == "contrast") & d0.declared).sum() and np.allclose(j7.point_0, j7.point_1, equal_nan=True) and np.allclose(j7.alpha_each_0, j7.alpha_each_1)
    print("ok 6d: propensity tie-break by symbol, fixed budget without refill, simul_ci reports n_complete/n_dropped")
    # 7. breadth against gb_breadth_truth v2's own output format + receipt gating + crossed context
    DELTA_AUROC, AP_LIFT_DELTA = 0.010, 0.10
    BTRUTH, LAD = f"{tmp}/bt", f"{tmp}/lad"
    ANALYSES = {"BM": [("TOTAL", "BM19", True)], "K562": [("TOTAL", "LAMBERT_TF", True)]}
    STAGES["smalln"] = dict(Bfit=3, Braw=3)
    perts = [f"TF{i}" for i in range(10)]
    res_b = {}
    for ctx, sysn in (("BM", "BONEMARROW"), ("K562", "K562")):
        rg = np.random.default_rng(5 if ctx == "BM" else 6)
        vv = rg.normal(size=nT + nT // 14 + 2)
        d = f"{tmp}/cb_{ctx}"
        _write_catbank(d, sysn, rg, "B", vv, sota=True)
        _use_dir(d)
        bk = load_bank(sysn, roots=False)
        msk = ~(np.arange(len(vv)) % 15 == 7)
        _write_truth(f"{BTRUTH}/{TRUTH_DIR[ctx]}", ctx, rg, vv[msk], list(bk["genes"]), perts, [(a[0], a[1]) for a in ANALYSES[ctx]], dup_pairs=(2 if ctx == "BM" else 0))
        if ctx == "BM":
            tr = load_breadth_truth("BM", "TOTAL", "BM19")
            assert tr["n_dup_symbols"] == 2 and tr["n_dup_rows_dropped"] == 4 and tr["gene_id_unique"] is True, (tr["n_dup_symbols"], tr["n_dup_rows_dropped"])
            assert tr["n_axis_genes"] - tr["n_testable_genes"] == 20 + 4 and tr["calibration"] == "CALIBRATED", "duplicated symbols (all rows) + untestable genes dropped"
            assert not (set(tr["genes"]) & {"g0", "g2"}), "a duplicated symbol must be UNKNOWN: never resolved to one of its rows"
            _write_truth_receipt(f"{BTRUTH}/BM", {"TOTAL": "UNCALIBRATED"})
            try:
                run_breadth("BM", "smalln", 1, variants=("full",))
            except SystemExit as e:
                assert "null-gate verdict UNCALIBRATED" in str(e)
            else:
                raise AssertionError("UNCALIBRATED truth must be refused")
            _write_truth_receipt(f"{BTRUTH}/BM", {"TOTAL": "CALIBRATED"})
        run_breadth(ctx, "smalln", 1, variants=("full", "excl") if ctx == "BM" else ("full",))
        res_b[ctx] = pd.read_csv(f"{LAD}/smalln/breadth_{ctx}.tsv", sep="\t")
    r = res_b["BM"]
    g_ = r[(r.name == f"raw:{PGB}") & (r.metric == "rho_breadth") & (r.variant == "full") & (r.kind == "model")].iloc[0]
    h_ = r[(r.name == "raw:pearson") & (r.metric == "rho_breadth") & (r.variant == "full") & (r.kind == "model")].iloc[0]
    assert g_.point > 0.25 and abs(h_.point) < 0.25, (g_.point, h_.point)
    assert set(r.variant) == {"full", "excl"} and set(r.variant_role) >= {"PRIMARY"} and (r.efficacy_status == "EFFICACY_UNVERIFIED").all() and r.decision_grade.all()
    dec = r[(r.kind == "contrast") & r.declared & (r.family == "BREADTH|declared")]
    assert len(dec) == 4 * 5 * 2 and np.allclose(dec.alpha_each, 0.05 / 20), "Bonferroni over declared contrasts x primary metrics"
    dr_ = r[(r.kind == "contrast") & r.declared & (r.family == "BREADTH|REF")]
    assert len(dr_) == 9 * 4 * 2 and np.allclose(dr_.alpha_each, 0.05 / 36), "Bonferroni within the BREADTH|REF family"
    assert all(f"raw:{n_}" in set(r.name) for n_ in REF_NAMES), "REF models present"
    hb = r[(r.kind == "hindsight") & (r.variant == "full") & (r.name == "HINDSIGHT_UPPER_BOUND")]
    assert len(hb) == r[(r.kind == "model") & (r.name == f"raw:{PGB}") & (r.variant == "full")].shape[0] and hb.ci_method.eq("none").all(), "hindsight rows"
    # composites: 4 COMP models per variant, own family BREADTH|COMP, simultaneous family over REF +
    # COMP + external, best-comparator hindsight
    assert {f"raw:{n_}" for n_ in GC.COMP_NAMES} <= set(r.name) and (r[r.name.isin([f"raw:{n_}" for n_ in GC.COMP_NAMES]) & (r.kind == "model")].role == "COMP").all()
    dc_ = r[(r.kind == "contrast") & r.declared & (r.family == "BREADTH|COMP")]
    assert len(dc_) == 4 * 4 * 2 and np.allclose(dc_.alpha_each, 0.05 / 16), "Bonferroni within BREADTH|COMP"
    sm_ = r[(r.kind == "contrast") & (r.family == "BREADTH|ALLCOMP_simul_maxT") & (r.variant == "full") & (r.metric == "auc_any")]
    assert len(sm_) == 3 + 9 + 4 + 4 and (r[(r.kind == "model") & r.name.str.startswith("F1")].role == "SUPPLEMENTARY_SUPERVISED").all()
    assert (r[(r.kind == "hindsight") & (r.name == "HINDSIGHT_BEST_COMPARATOR")].shape[0] > 0)
    ds_ = r[(r.kind == "contrast") & r.declared & (r.family == "BREADTH|TIER1_SOTA")]
    assert len(ds_) == 4 * 4 * 2 and np.allclose(ds_.alpha_each, 0.05 / 16) and (r[r.name.isin([f"raw:{m_}" for m_ in SOTA_METHODS]) & (r.kind == "model")].role == "TIER1_SOTA").all()
    prr_ = set(pd.read_csv(f"{LAD}/smalln/breadth_BM_propensity.tsv", sep="\t", usecols=["role"]).role.dropna())
    assert prr_ >= {"TIER1_SOTA", "COMP", "REF", "EXTERNAL", "GB_LABEL_FREE"}, prr_
    assert (r[(r.kind == "contrast") & r.name.str.contains("F1")].role == "SUPPLEMENTARY_SUPERVISED").all() and (r[(r.kind == "contrast") & r.name.str.startswith("raw:PGB")].role == "COMPARATOR").all()
    fa_ = r[(r.kind == "contrast") & (r.family == "BREADTH|ALLCOMP_simul_maxT")]
    assert fa_.n_family.eq(4 * 20).all() and {"n_valid_draws", "n_dropped_draws"} <= set(fa_.columns), "ONE simultaneous family across metrics x comparators"
    # propensity: both bootstrap modes, fixed-budget slots sum to 50 (unlabeled slots not refilled),
    # P_pairU_noPanel label on the sensitivity variant
    pr_ = pd.read_csv(f"{LAD}/smalln/breadth_BM_propensity.tsv", sep="\t")
    assert set(pr_["mode"]) >= {"two_level", "tf_only", "point"} and {"fold_macro", "tf_macro", "fold0"} <= set(pr_.level)
    fb_ = pr_[(pr_.kind == "model") & (pr_["mode"] == "two_level")].pivot_table(index=["variant", "level", "name"], columns="metric", values="point")
    fbf_ = fb_.xs("fold0", level="level")
    assert np.allclose(fbf_.FB50_resp + fbf_.FB50_nonresp + fbf_.FB50_unlab, 50), "fixed budget: 50 slots, unlabeled counted not refilled"
    assert np.allclose(fbf_.AP / fbf_.prev, fbf_.APlift, equal_nan=True)
    assert set(pr_[pr_.variant == "excl"].gb_prior_label) == {PGB + "_noPanel"} and set(pr_[pr_.variant == "full"].gb_prior_label) == {PGB}
    ct_ = pr_[(pr_.kind == "contrast") & (pr_.variant == "full")]
    assert (ct_.n_family == 2 * 20).all() and ct_.family.str.startswith(("PROP|", "pointwise")).all() and {"n_valid_draws", "n_dropped_draws"} <= set(ct_.columns), "one family per (mode, level): primary metrics x 20 comparators (REF + COMP + external + TIER1_SOTA)"
    assert set(ct_[ct_.family != "pointwise"].metric) == set(PROP_PRIMARY) and set(ct_[ct_.family == "pointwise"].metric) >= set(PROP_SECONDARY) - {"FB50_resp_allTF"}
    pm_ = pr_[(pr_.kind == "model") & pr_.prop_primary]
    assert set(pm_["mode"]) == {PROP_PRIMARY_MODE} and set(pm_.level) == {PROP_PRIMARY_LEVEL} and set(pm_.metric) == set(PROP_PRIMARY), "declared primary = tf_macro x tf_only x (APlift, FB50_resp)"
    tm_ = pr_[(pr_.kind == "model") & (pr_["mode"] == "tf_only") & (pr_.level == "tf_macro") & (pr_.variant == "full")].pivot_table(index="name", columns="metric", values="point")
    assert (tm_.n_TF >= 1).all() and tm_.n_TF.nunique() == 1, "the SAME TF set for every predictor"
    assert np.allclose(tm_.FB50U_resp + tm_.FB50U_nonresp + tm_.FB50U_unlab, 50), "FB50U: 50 slots on the full universe, unlabeled counted"
    ro_ = pr_[(pr_.kind == "robustness") & (pr_.variant == "full")]
    assert {"APlift", "FB50_resp"} <= set(ro_.metric) and ro_.robustness.str.startswith("LOTO:").any() and "superiority_both_modes" in pr_.columns
    lo1 = ro_[(ro_.robustness == ro_.robustness.iloc[0]) & (ro_.metric == "APlift") & (ro_.name == f"raw:{PGB}")].point.iloc[0]
    assert np.isfinite(lo1), "leave-one-TF-out macro from the per-TF rows"
    assert set(fa_.metric) == set(ALLCOMP_BREADTH_METRICS), "stacked family uses the stratified yield"
    pt_ = pd.read_csv(f"{LAD}/smalln/breadth_BM_propensity_per_tf.tsv", sep="\t")
    assert {"tf", "n_pos", "n_eval", "AP", "APlift", "FB50_resp", "in_macro_set"} <= set(pt_.columns) and (pt_[pt_.variant == "full"].groupby("name").in_macro_set.sum() == tm_.n_TF.iloc[0]).all()
    assert (r[(r.kind == "contrast") & (r.name == "raw:PGB - raw:pearson")].shape[0] > 0)
    _write_truth_receipt(f"{BTRUTH}/{TRUTH_DIR['K562']}", {"TOTAL": "CALIBRATED"}, phase="RUNNING")          # a K562 truth that is not COMPLETE is refused
    try:
        load_breadth_truth("K562", "TOTAL", "LAMBERT_TF")
    except SystemExit as e:
        assert "not COMPLETE" in str(e)
    else:
        raise AssertionError("an incomplete K562 truth receipt must be refused")
    _write_truth_receipt(f"{BTRUTH}/{TRUTH_DIR['K562']}", {"TOTAL": "CALIBRATED"})
    assert TRUTH_DIR["K562"] == "K562_EMP_SHARD" and load_breadth_truth("K562", "TOTAL", "LAMBERT_TF")["truth_dir"].endswith("K562_EMP_SHARD")
    run_cross("smalln")
    cr = pd.read_csv(f"{LAD}/smalln/breadth_CROSS.tsv", sep="\t")
    assert len(cr) >= 4 and np.isfinite(cr.interaction).all() and cr.decision_grade.all() and (cr.panel_restriction == "common_perturbed_TFs").all()
    # truth self-check fails closed when per_gene.tsv is inconsistent with per_pair.tsv.gz (receipt
    # hash refreshed so the b_g check itself is what fires)
    pgf = f"{BTRUTH}/BM/per_gene.tsv"
    pgd = pd.read_csv(pgf, sep="\t")
    pgd.loc[:, "b_g"] = pgd.b_g.iloc[::-1].to_numpy()
    pgd.to_csv(pgf, sep="\t", index=False)
    try:
        load_breadth_truth("BM", "TOTAL", "BM19")
    except SystemExit as e:
        assert str(e).startswith("BLOCKED_BY_SPEC_LOADER") and "receipt.json" in str(e)
    else:
        raise AssertionError("truth hash check did not fail closed")
    _write_truth_receipt(f"{BTRUTH}/BM", {"TOTAL": "CALIBRATED"})
    try:
        load_breadth_truth("BM", "TOTAL", "BM19")
    except SystemExit as e:
        assert "does not reproduce per_gene.tsv" in str(e)
    else:
        raise AssertionError("truth self-check did not fail closed")
    print("ok 7: breadth: testable-gene filter + coverage; receipt gating (UNCALIBRATED refused); per_pair reproduces per_gene; rho(P_GB,b)=%.3f vs rho(Pearson,b)=%.3f; "
          "two-level bootstrap + Bonferroni; partial-Spearman crossed test on common perts" % (g_.point, h_.point))
    print("SELFTEST PASSED")


def load_parts_from(path):
    z = np.load(path, allow_pickle=True)
    fams = {}
    for k in z.files:
        if k.endswith("|info"):
            fam = k[:-5]
            fams[fam] = dict(json.loads(str(z[k])), pt=z[f"{fam}|pt"], raw=z[f"{fam}|raw"], fit=z[f"{fam}|fit"], inf=z[f"{fam}|inf"])
    return dict(meta=json.loads(str(z["meta"])), fams=fams)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", nargs="?", choices=["endpoints", "aggregate", "breadth"])
    ap.add_argument("arg", nargs="?")
    ap.add_argument("--stage", default="pilot", choices=list(STAGES))
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--smalln", action="store_true")
    ap.add_argument("--limit", type=int, default=None, help="first N endpoints by name (smoke only; recorded in the receipt)")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--bank-tag", default="", choices=["", "_smalln"], help="use gb_catbank --smalln outputs (5 members)")
    ap.add_argument("--truth-tag", default="", choices=["", "_smalln"], help="use gb_breadth_truth --smalln outputs")
    ap.add_argument("--set", default="T7", choices=list(SETS), help="catbank strict set: T7 (GB4 + pearson/genie3/grnboost2) or T7+regvelo / T7+velorama")
    ap.add_argument("--tier", default="F1", choices=["F1", "F1x"], help="Feature tier: F1; F1x adds normalised dispersion, CV2 and the U-S ratio.")
    ap.add_argument("--relaxed", action="store_true", help="relaxed-target sensitivity (regulator genes kept as targets; T-derived normalisation)")
    ap.add_argument("--arms", default="", help="breadth: comma list of arms to run (e.g. LAMBERT_TF,F2_36); default = every declared arm")
    ap.add_argument("--no-roots", action="store_true", help="skip the per-root / root-mean diagnostics (T7 only)")
    ap.add_argument("--allow-uncalibrated", action="store_true", help="breadth: continue on a null-gate verdict != CALIBRATED (rows are labelled, not decision-grade)")
    a = ap.parse_args()
    kw = dict(tag=a.bank_tag, sset=a.set, relaxed=a.relaxed, tier=a.tier)
    if a.selftest:
        return selftest()
    if a.smalln:
        run_endpoints("PANC", "smalln", a.n_jobs, a.limit, roots=not a.no_roots, **kw)
        run_aggregate("smalln", True, a.set, a.relaxed, a.tier)
        run_breadth("BM", "smalln", a.n_jobs, ttag=a.truth_tag, allow_uncal=a.allow_uncalibrated, **kw)
        return
    if a.mode == "endpoints":
        run_endpoints(a.arg.upper(), a.stage, a.n_jobs, a.limit, roots=not a.no_roots, **kw)
    elif a.mode == "aggregate":
        run_aggregate(a.stage, a.allow_partial, a.set, a.relaxed, a.tier)
    elif a.mode == "breadth":
        run_breadth(a.arg.upper(), a.stage, a.n_jobs, ttag=a.truth_tag, allow_uncal=a.allow_uncalibrated, arms=[x for x in a.arms.split(",") if x] or None, **kw)
    else:
        ap.error("mode required")


if __name__ == "__main__":
    main()
