#!/usr/bin/env python
"""COGNATE-ROW ATTRIBUTION -- does the focal regulator's OWN row carry information beyond (a) the same method's leave-focal-out target prior and (b) observationally matched
pseudo-regulator rows?  EVALUATION ONLY (internal review decisive unknown 2026-09-29, "GB signal is target-level?").  No score, feature or hyper-parameter is changed; truth is read
only inside gb_layer_lattice.build_endpoint AFTER every score / prior / feature artifact (the lattice's own order), and matching uses UNTREATED observational features only.

  METHODS (10)   GeoBRIDGE (production PRIMARY == catbank P_pairU), |Pearson|, GENIE3, GRNBoost2, RegVelo |J|, Velorama, RegDiffusion, DeepSEM, PIDC, FastTENET
  SYSTEMS        GATA1, KLF1, KLF1_B, PANC, BONEMARROW, BM_CHIPATLAS, BM_GATA1_ACUTE, K562_EMP_ALLNULL  (lattice truth builders imported, gb_lattice_extra.register())
  UNIVERSE       M1 = lattice ALL6+SOTA mask (U & the 10 cognate rows finite)  AND  T7 (catbank strict common target set; regulator genes of the bank drop out, exactly as in the
                 lattice's `prior GB`)  AND  every method's leave-focal-out prior finite.  ONE mask per (system, TF) for cognate / prior / pseudos of EVERY method.
  THREE VECTORS  (i) cognate row S_m[r,.] (= lattice V[m], i.e. production row); (ii) prior P_m^(-r) = gb_catbank.loo_prior over the catalogue bank (row-z on T7, self entries NaN,
                 built by gb_catbank.sota_prior_arrays == build_set algebra; asserted equal to the stored catbank Z and to the lattice `prior GB`); (iii) the WHOLE ELIGIBLE BANK as the
                 pseudo-regulator pool (all bank rows except the focal, the system's perturbed-panel TFs [K562: every TF of the state file], rows with missing features or non-finite on
                 the mask for any method).  The bank = species TF catalogue with detection >= 5%; it includes cofactors / chromatin regulators, not only sequence-specific TFs.
                 Eligible-bank size (N_bank) is recorded next to every null.  Truth never used.  The old K=20 feature-matched subset is kept as a LEGACY side-by-side column.
  METRICS        resp AUROC / AUPRC / AP-lift; AUROC of AB, CD, occ, AC, BD (primary window) on the SAME rows; gate >= 20/class (L.gate_of; K562 prereg support rule where it applies).
  PRIMARY NULL   E_prior = cognate - prior is BIASED (a single noisy row scores below a bank average; anti-predictive priors inflate it), so the attribution statistic is the CALIBRATED
                 excess E_cal = cognate - OLS prediction from the whole eligible bank (metric units), regressing each bank row's metric on truth-free covariates [loading = Spearman(row,
                 its own leave-out prior) on the mask, raw_det, abundance, |rho(Ms,DPT)|].  The p-value is the percentile of the focal's STANDARDIZED residual among the bank's standardized
                 residuals (scale model: OLS of log residual^2 on the same covariates, normalised to mean 1) -> robust to heteroscedasticity; p_cal_global (unstandardized) and p_cal_knn
                 (k=200 covariate-nearest bank rows) are reported beside it, with the unadjusted whole-bank percentile p_bank.  The +-0.20 AP-lift margin flag applies to the mean E_cal.
  UNCERTAINTY    ONE gene multinomial resample matrix per block (lattice seed), shared by every method / vector / target; E_prior CI = paired percentile; E_cal CI is CONDITIONAL on the
                 point-estimate bank fit (bank not refit per resample) -- the percentile p is the null calibration.
  SUBSTRATE      one declared PRIMARY cell: GeoBRIDGE x resp AUROC per substrate (= system; systems are never pooled), p = BRACKET [aligned null, independent-draw null] of the mean
                 standardized excess over the substrate's gated TFs, Holm across substrates on each bracket end.  Everything else is DESCRIPTIVE.  Regulator pools overlap across TFs.

  python scratch/gb_attribution.py --selftest
  python scratch/gb_attribution.py tfs --systems BM_CHIPATLAS --slice 0 --nslice 2
  python scratch/gb_attribution.py run --systems GATA1,PANC --K 20 --B 200 --out results/real_uniform/attribution_smalln [--tfs A,B] [--assert-parity]
  python scratch/gb_attribution.py merge --out DIR          # per_tf_method.tsv, summary.tsv, receipt.json
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gb_canon_table as CT  # noqa: E402
import gb_catbank as CB  # noqa: E402

R = __import__("os").environ["GEOBRIDGE_ROOT"]
OUT_DEFAULT = f"{R}/results/real_uniform/attribution"
GB = "GeoBRIDGE"
NAME2KEY = {"|Pearson|": "pearson", "GENIE3": "genie3", "GRNBoost2": "grnboost2", "RegVelo |J|": "regvelo", "Velorama": "velorama",
            "RegDiffusion": "regdiffusion", "DeepSEM": "deepsem", "PIDC": "pidc", "FastTENET": "fasttenet"}
METHODS = [GB] + list(NAME2KEY)
SOTA_M = ["RegDiffusion", "DeepSEM", "PIDC", "FastTENET"]
FEATS = ["raw_det", "abundance", "abs_spearman_ms_dpt"]        # spliced detection, mean abundance, |Spearman(Ms, DPT)| (catbank features.tsv; untreated fixture only)
SYSTEMS = ["GATA1", "KLF1", "KLF1_B", "PANC", "BONEMARROW", "BM_CHIPATLAS", "BM_GATA1_ACUTE", "K562_EMP_ALLNULL"]
TF_OVERRIDE = {"BONEMARROW": ["GATA1", "GATA2", "RUNX1", "TAL1"]}           # spec: RUNX1, GATA1, GATA2, TAL1 (FLI1 has no KO panel)
PERTURBED_FIXED = {"GATA1": {"GATA1", "KLF1"}, "KLF1": {"GATA1", "KLF1"}, "KLF1_B": {"GATA1", "KLF1"}, "PANC": {"RFX6"}}
TARGETS = ["resp", "AB_prim", "CD_prim", "occ_prim", "AC_prim", "BD_prim"]
METRICS = {"resp": ("AUROC", "AUPRC", "APLIFT")}                              # every other target: AUROC only
MARGIN = 0.20
LATTICE_REF = {"base": f"{R}/results/real_uniform/layer_lattice_sota_final/endpoints.tsv", "extra": f"{R}/results/real_uniform/layer_lattice_extra_sota/{{system}}/endpoints.tsv"}


# ===================================================================================================== pure numerics (unit-tested by --selftest)
def point(score, y, rows):
    """point metrics of `score` on `rows` (bool over genes) against `y`; None if a class is empty.  Same engine (CT.weighted_metrics, tie-grouped) as the lattice."""
    idx = np.flatnonzero(rows)
    s = np.asarray(score, float)[idx]
    if not np.isfinite(s).all():
        raise RuntimeError("attribution.point: non-finite score inside the universe")
    yy = np.asarray(y, bool)[idx]
    if not (yy.any() and (~yy).any()):
        return None
    o, starts = CT.prep_scores(s)
    return {k: float(v[0]) for k, v in CT.weighted_metrics(np.ones((1, len(idx))), yy[o], starts, n_rows=len(idx)).items()}


def pseudo_p(cog, pseudo):
    """(1 + #{pseudo >= cognate}) / (K + 1): the cognate's tail rank among the pseudo rows (ties count against the cognate)"""
    pseudo = np.asarray(pseudo, float)
    pseudo = pseudo[np.isfinite(pseudo)]
    return float((1 + (pseudo >= cog).sum()) / (len(pseudo) + 1)) if np.isfinite(cog) else np.nan


def pick_pseudos(F, f_r, eligible, K, names):
    """K nearest rows of F (nb, f; standardized features) to f_r among `eligible`; Euclidean, ties broken by name.  Returns (indices, distances)."""
    d = np.sqrt(((np.asarray(F, float) - np.asarray(f_r, float)) ** 2).sum(1))
    d = np.where(np.asarray(eligible, bool) & np.isfinite(d), d, np.inf)
    o = np.lexsort((np.asarray(names).astype(str), d))
    o = [i for i in o if np.isfinite(d[i])][:K]
    return np.asarray(o, int), d[o]


def block_stats(cog, prior, PS, PP, y, rows, mets):
    """cognate / prior / pseudo point metrics and the excess statistics for one target.  PS (K, n_genes) pseudo rows, PP (K, n_genes) their own leave-out priors (row NaN -> skipped)."""
    c, p = point(cog, y, rows), point(prior, y, rows)
    if c is None or p is None:
        return None
    ps = [point(v, y, rows) for v in PS]
    pp = [point(v, y, rows) if np.isfinite(np.asarray(v)[rows]).all() else None for v in PP]
    out = {}
    for m in mets:
        pv = np.array([q[m] for q in ps], float)
        nul = np.array([a[m] - b[m] for a, b in zip(ps, pp) if b is not None], float)
        e_prior = c[m] - p[m]
        out[m] = dict(cognate=c[m], prior=p[m], pseudo_median=float(np.median(pv)), pseudo_q25=float(np.percentile(pv, 25)), pseudo_q75=float(np.percentile(pv, 75)),
                      E_prior=e_prior, E_pseudo=c[m] - float(np.median(pv)), pseudo_p=pseudo_p(c[m], pv), K_eff=int(len(pv)),
                      null_E_prior_median=float(np.median(nul)) if len(nul) else np.nan, null_E_prior_sd=float(np.std(nul)) if len(nul) else np.nan, n_pseudo_focal=int(len(nul)),
                      pseudo_focal_p=float((1 + (nul >= e_prior).sum()) / (len(nul) + 1)) if len(nul) else np.nan)
    return out


def margin_flag(x):
    return "NA" if not np.isfinite(x) else ("cognate>ref beyond margin" if x > MARGIN else "ref>cognate beyond margin" if x < -MARGIN else "within +-0.20")


def spearman_rows(A, B):
    """row-wise Spearman correlation of A (N, n) with B (N, n) (average ranks); NaN if a row is constant / non-finite"""
    from scipy.stats import rankdata
    A, B = np.asarray(A, float), np.asarray(B, float)
    ok = np.isfinite(A).all(1) & np.isfinite(B).all(1)
    out = np.full(len(A), np.nan)
    if ok.any():
        ra, rb = rankdata(A[ok], axis=1), rankdata(B[ok], axis=1)
        ra, rb = ra - ra.mean(1, keepdims=True), rb - rb.mean(1, keepdims=True)
        with np.errstate(invalid="ignore", divide="ignore"):
            out[ok] = (ra * rb).sum(1) / np.sqrt((ra ** 2).sum(1) * (rb ** 2).sum(1))
    return out


def bank_calibration(yb, Cb, yf, cf, k_nn=200):
    """PRIMARY loading-adjusted, HETEROSCEDASTICITY-standardized null.  yb (N,) metric of every eligible bank row; Cb (N, q) its truth-free covariates [loading, raw_det, abundance,
    |rho(Ms,DPT)|]; yf, cf the focal's.  (1) OLS mean fit with intercept over the bank rows -> residuals res_i, focal E_cal = yf - predicted(cf) (metric units).  (2) Scale model:
    OLS of log(res_i^2 + tiny) on the same covariates; scale_i = sqrt(exp(fit_i) * c) with c set so that mean(res_i^2 / scale_i^2) = 1 over the bank (the focal's fitted log-variance is
    clipped to the bank's fitted range: no extrapolated scale).  z_i = res_i / scale_i, z_focal = E_cal / scale_focal.  PRIMARY p_cal = (1 + #{z_i >= z_focal}) / (N + 1).
    Side by side: p_cal_global (unstandardized residual percentile), p_cal_knn (focal residual among the k_nn bank rows nearest in standardized covariate space), and the unadjusted p_bank.
    Returns (dict, raw residuals aligned with yb, standardized residuals aligned with yb; NaN where the row was unusable)."""
    yb, Cb, cf = np.asarray(yb, float), np.asarray(Cb, float), np.asarray(cf, float)
    ok = np.isfinite(yb) & np.isfinite(Cb).all(1)
    res_all, z_all = np.full(len(yb), np.nan), np.full(len(yb), np.nan)
    if ok.sum() < Cb.shape[1] + 5 or not (np.isfinite(yf) and np.isfinite(cf).all()):
        return dict(N_bank=int(ok.sum())), res_all, z_all
    n = int(ok.sum())
    A = np.c_[np.ones(n), Cb[ok]]
    af = np.r_[1.0, cf]
    beta = np.linalg.lstsq(A, yb[ok], rcond=None)[0]
    res = yb[ok] - A @ beta
    res_all[ok] = res
    e_cal = float(yf - af @ beta)
    lv = np.log(res ** 2 + 1e-12)
    g = np.linalg.lstsq(A, lv, rcond=None)[0]
    fit_b = A @ g
    fit_f = float(np.clip(af @ g, fit_b.min(), fit_b.max()))
    c = float(np.mean(res ** 2 / np.exp(fit_b)))
    z = res / np.sqrt(np.exp(fit_b) * c)
    z_all[ok] = z
    scale_f = float(np.sqrt(np.exp(fit_f) * c))
    z_f = e_cal / scale_f
    mu_c, sd_c = Cb[ok].mean(0), Cb[ok].std(0)
    sd_c = np.where(sd_c > 0, sd_c, 1.0)
    d = np.sqrt((((Cb[ok] - mu_c) / sd_c - (cf - mu_c) / sd_c) ** 2).sum(1))
    nn = np.argsort(d, kind="mergesort")[:min(k_nn, n)]
    tss = ((yb[ok] - yb[ok].mean()) ** 2).sum()
    return dict(N_bank=n, pred_focal=float(af @ beta), E_cal=e_cal, p_cal=float((1 + (z >= z_f).sum()) / (n + 1)), z_cal=float(z_f), scale_focal=scale_f,
                p_cal_global=float((1 + (res >= e_cal).sum()) / (n + 1)), p_cal_knn=float((1 + (res[nn] >= e_cal).sum()) / (len(nn) + 1)), k_nn=int(len(nn)),
                sd_resid=float(res.std(ddof=A.shape[1])), R2_bank=float(1 - (res ** 2).sum() / tss) if tss > 0 else np.nan, bank_median=float(np.median(yb[ok])),
                E_bank=float(yf - np.median(yb[ok])), p_bank=float((1 + (yb[ok] >= yf).sum()) / (n + 1)), beta_loading=float(beta[1])), res_all, z_all


def holm(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    f = np.flatnonzero(np.isfinite(p))
    o = f[np.argsort(p[f])]
    run = 0.0
    for i, j in enumerate(o):
        run = max(run, min(1.0, (len(o) - i) * p[j]))
        out[j] = run
    return out


def is_primary(method, target, metric):
    """the ONE declared primary cell: GeoBRIDGE x resp AUROC, one p per substrate (bracket), Holm across substrates; every other cell is descriptive"""
    return method == GB and target == "resp" and metric == "AUROC"


def substrate_p(focal_z, tf_z, n_mc=20000, seed=0):
    """substrate-level p of the MEAN standardized calibrated excess over the substrate's gated TFs, reported as a BRACKET of two nulls built from the bank rows' standardized residuals
    (tf_z = [{bank name: z}], one dict per TF):
      p_aligned = null_p = mean over TFs of the SAME bank row's z (keeps the cross-TF dependence of re-used regulator rows; exact only if the focal regulators' rows were identical across TFs,
                  otherwise a conservative bound);
      p_indep   = one random bank row per TF, independently per TF (exact only if the TFs' rows are unrelated; ignores shared-row dependence).
    Returns (p_aligned, p_indep); NaN if there is no common bank row."""
    common = sorted(set.intersection(*[set(d) for d in tf_z])) if tf_z else []
    if not common:
        return np.nan, np.nan
    obs = float(np.mean(focal_z))
    M = np.array([[d[n_] for n_ in common] for d in tf_z], float)                 # (n_tf, n_common)
    M = M[:, np.isfinite(M).all(0)]
    if M.shape[1] == 0:
        return np.nan, np.nan
    nul_a = M.mean(0)
    rg = np.random.default_rng(seed)
    nul_i = np.array([M[t][rg.integers(0, M.shape[1], n_mc)] for t in range(M.shape[0])]).mean(0)
    return float((1 + (nul_a >= obs).sum()) / (len(nul_a) + 1)), float((1 + (nul_i >= obs).sum()) / (n_mc + 1))


def summarize(pt, resid=None):
    """DESCRIPTIVE except the ONE declared primary cell (GeoBRIDGE x resp AUROC per substrate; Holm across substrates on each bracket end).  Macro (equal-weight over gated TFs) per
    (system, method, target, metric) of the CALIBRATED excess (metric units); p_substrate_* = bracket of substrate nulls from the STANDARDIZED bank residuals (see substrate_p).
    Regulator pools overlap across TFs, so TF-level p-values are not independent.  N_bank_min = eligible-bank size behind every null.
    resid[(system, tf)] = {(method, target, metric): {bank name: standardized residual}} (optional)."""
    d = pt[(pt.status == "OK") & pt.gate.astype(bool)]
    rows = []
    for (s, m, t, k), g in d.groupby(["system", "method", "target", "metric"], sort=False):
        r = dict(system=s, method=m, target=t, metric=k, primary_cell=is_primary(m, t, k), n_tf=len(g), N_bank_min=int(g.N_bank.min()), N_bank_max=int(g.N_bank.max()),
                 mean_cognate=g.cognate.mean(), mean_prior=g.prior.mean(), mean_bank_median=g.bank_median.mean(),
                 mean_E_cal=g.E_cal.mean(), sd_E_cal=g.E_cal.std(ddof=1) if len(g) > 1 else np.nan, mean_z_cal=g.z_cal.mean(), median_p_cal=g.p_cal.median(),
                 n_tf_p_cal_le_05=int((g.p_cal <= 0.05).sum()), median_p_cal_global=g.p_cal_global.median(), median_p_cal_knn=g.p_cal_knn.median(), median_p_bank_unadjusted=g.p_bank.median(),
                 mean_E_prior_raw=g.E_prior.mean(), mean_E_pseudo20_legacy=g.E_pseudo.mean(), median_pseudo20_p_legacy=g.pseudo_p.median(), median_pseudo_focal_p_diag=g.pseudo_focal_p.median(),
                 p_substrate_aligned=np.nan, p_substrate_indep=np.nan)
        if resid is not None:
            lst = [resid[(s, tf)][(m, t, k)] for tf in g.tf if (s, tf) in resid and (m, t, k) in resid[(s, tf)]]
            if len(lst) == len(g):
                r["p_substrate_aligned"], r["p_substrate_indep"] = substrate_p(g.z_cal.to_numpy(), lst)
        if k == "APLIFT":
            r["margin_E_cal"] = margin_flag(r["mean_E_cal"])
        rows.append(r)
    S = pd.DataFrame(rows)
    S["p_holm_aligned"] = S["p_holm_indep"] = np.nan
    pc = S[S.primary_cell]
    if len(pc):
        S.loc[pc.index, "p_holm_aligned"] = holm(pc.p_substrate_aligned.to_numpy())
        S.loc[pc.index, "p_holm_indep"] = holm(pc.p_substrate_indep.to_numpy())
    S["note"] = ("PRIMARY only on primary_cell (GB x resp AUROC; Holm across substrates); all else DESCRIPTIVE.  E_cal = focal - loading-adjusted whole-bank prediction (metric units); p from the "
                 "heteroscedasticity-standardized residual; substrate p is a bracket [aligned, independent]; regulator pools overlap across TFs (non-independent)")
    return S


# ===================================================================================================== bank context (per catbank axis system)
class Bank:
    """raw bank score matrices (nb x G_axis, bank order) + row-z constants of the frozen T7 set, per method"""

    def __init__(self, members, axis, T, X, feats=None, sha=None):
        self.members, self.axis, self.T, self.X, self.feats, self.sha = list(members), list(axis), np.asarray(T, bool), X, feats, sha or {}
        self.mu = np.array([m.upper() for m in self.members])
        self.ai = {g: i for i, g in enumerate(self.axis)}
        self.bank_cols = np.asarray([self.ai[g] for g in self.members])
        self.tcol = np.full(len(self.axis), -1)
        self.tcol[self.T] = np.arange(int(self.T.sum()))
        self.P = {m: CB.sota_prior_arrays(x, self.bank_cols, self.T, 2) for m, x in X.items()}          # == build_set algebra, NaN tolerant, T never changed
        self.row_ok = {m: np.isfinite(p["mu"]) for m, p in self.P.items()}                              # rows that are usable (not all-NaN / constant on T)

    def row_of(self, name):
        w = np.flatnonzero(self.mu == name.upper())
        return int(w[0]) if len(w) else None

    def prior_rows(self, m, rows):
        """leave-own-row-out priors of MANY bank rows at once, (len(rows), n_T7); identical formula to CB.loo_prior"""
        p = self.P[m]
        Z = p["Z"][rows]
        f = np.isfinite(Z)
        s_, c_ = p["S"][None, :] - np.where(f, Z, 0.0), p["C"][None, :] - f
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(c_ >= 2, s_ / np.maximum(c_, 1), np.nan)

    def prior(self, m, r):
        """leave-r-out prior over the axis (NaN outside T7); r = bank row or None (focal not in the bank -> full-bank prior)"""
        p = self.P[m]
        v = CB.loo_prior(p["Z"], p["S"], p["C"], r, 2)
        out = np.full(len(self.axis), np.nan)
        out[self.T] = v
        return out


def sota_tolerant(z, key, nk, members, axis):
    """rows for `members` and columns for `axis` from a catalogue-wide artifact; a missing bank row -> all-NaN row (RECORDED), a missing axis gene -> NaN column"""
    rn = {str(x).upper(): i for i, x in enumerate(z[nk].astype(str))}
    cp = {str(g): i for i, g in enumerate(z["genes"].astype(str))}
    M = np.asarray(z[key], np.float64)
    cols = np.array([cp.get(g, -1) for g in axis])
    out = np.full((len(members), len(axis)), np.nan)
    miss = []
    for k, g in enumerate(members):
        i = rn.get(g.upper())
        if i is None:
            miss.append(g)
            continue
        out[k, cols >= 0] = M[i, cols[cols >= 0]]
    return out, miss, int((cols < 0).sum())


_BANKS = {}


def load_bank(axis_sys):
    """validated catbank artifacts (bank sha, priors/external/shard receipts) -> Bank.  Fails closed on any stale artifact."""
    if axis_sys in _BANKS:
        return _BANKS[axis_sys]
    import gb_layer_lattice as L
    import gb_real_uniform as RU
    B = json.load(open(f"{CB.OUT}/{axis_sys}_bank.json"))
    if B["sha256_members"] != CB.members_sha(B["members"]):
        raise SystemExit(f"REFUSING_BANK_INCONSISTENT {axis_sys}")
    members = B["members"]
    pf = f"{CB.OUT}/{axis_sys}_priors.npz"
    if L.receipt_of(pf)["config"]["bank_sha256"] != B["sha256_members"]:
        raise SystemExit(f"REFUSING_PRIORS_BANK_MISMATCH {pf}")
    with np.load(pf, allow_pickle=True) as z:
        axis, Tg = [str(x) for x in z["axis"]], [str(x) for x in z["T7|genes"]]
        if [str(x) for x in z["members"]] != members:
            raise SystemExit("REFUSING_PRIORS_MEMBERS_DIFFER")
        zgb, zpe = np.asarray(z["T7|P_pairU|Z"]), np.asarray(z["T7|pearson|Z"])
    T = np.isin(axis, Tg)
    if [g for g, t in zip(axis, T) if t] != Tg:
        raise SystemExit("REFUSING_T7_NOT_A_SUBSEQUENCE_OF_THE_AXIS")
    sha, X, notes = {"bank": L.sha_file(f"{CB.OUT}/{axis_sys}_bank.json"), "priors": L.sha_file(pf)}, {}, {}
    genes, gbm, _dg, used, _meta = CB.load_gb(axis_sys, False, members, B)                               # validated shards (receipt, bank sha, production_universe, parity_verified)
    if [str(g) for g in genes] != axis:
        raise SystemExit("REFUSING_SHARD_AXIS_DIFFERS_FROM_PRIORS")
    X[GB] = np.asarray(gbm["P_pairU"], float)
    sha["scores_shards"] = used
    ep_ = f"{CB.OUT}/{axis_sys}_external.npz"
    er = L.receipt_of(ep_)
    if er.get("bank_sha256") != B["sha256_members"] or er.get("fixture_sha256") != B["inputs"]["fixture_sha256"]:
        raise SystemExit("REFUSING_EXTERNAL_RECEIPT_STALE")
    sha["external"] = L.sha_file(ep_)
    with np.load(ep_, allow_pickle=True) as z:
        if [str(x) for x in z["members"]] != members or [str(x) for x in z["axis"]] != axis:
            raise SystemExit("REFUSING_EXTERNAL_BANK_OR_AXIS_MISMATCH")
        have = [str(m) for m in z["methods"]]
        for nm, key in NAME2KEY.items():
            if key in have:
                X[nm] = np.asarray(z[f"M|{key}"], float)
    d = RU.system_dir(axis_sys)
    for nm in SOTA_M:                                                                                    # SOTA artifacts (score_all over catalogue TFs); tolerant of missing bank rows
        fn = f"{NAME2KEY[nm]}_v1.npz"
        prob = "file missing" if not (d / fn).exists() else CB.ext_artifact_problem(d / fn, B["inputs"]["fixture_sha256"])
        if prob:
            notes[nm] = f"NOT_AVAILABLE: {prob}"
            continue
        with np.load(d / fn, allow_pickle=True) as z:
            X[nm], miss, ncol = sota_tolerant(z, "score_all", "tfs", members, axis)
        sha[nm] = L.sha_file(d / fn)
        notes[nm] = dict(missing_bank_rows=miss, n_axis_cols_missing=ncol)
    ff = f"{CB.OUT}/{axis_sys}_features.tsv"
    L.receipt_of(ff)
    F = pd.read_csv(ff, sep="\t", index_col="gene")
    sha["features"] = L.sha_file(ff)
    bank = Bank(members, axis, T, X, F, sha)
    bank.notes = notes
    for nm, z_ in (("P_pairU", zgb), ("pearson", zpe)):                                                  # parity of MY prior construction vs the stored catbank Z (same algebra)
        mine = bank.P[GB if nm == "P_pairU" else "|Pearson|"]["Z"]
        if not np.allclose(mine, z_, atol=1e-9, equal_nan=True):
            raise SystemExit(f"REFUSING_PRIOR_CONSTRUCTION_DIFFERS_FROM_CATBANK {nm}")
    _BANKS[axis_sys] = bank
    return bank


# ===================================================================================================== one endpoint (system, TF)
def perturbed_set(system, L, LE):
    if system in PERTURBED_FIXED:
        return set(PERTURBED_FIXED[system])
    if system.startswith("BM"):
        return {t.upper() for t in LE.PANEL_TFS} | {"GATA1"}
    if system.startswith("K562"):                                                    # ALL perturbed TFs of the state file (86), not only those with ChIP (34)
        sp = {"K562_EMP_ALLNULL": LE.K562_EMP_ALLNULL_STATE, "K562_EMP": LE.K562_EMP_STATE}.get(system, L.K562_STATE)
        return {str(t).upper() for t in np.load(sp, allow_pickle=True)["tfs"]}
    raise SystemExit(f"no perturbed-panel definition for {system}")


def lattice_ref_auroc(system, tf):
    """GeoBRIDGE resp AUROC on the ALL6+SOTA scope from the lattice run, or None"""
    p = LATTICE_REF["base"] if system in ("GATA1", "KLF1", "PANC", "BONEMARROW") else LATTICE_REF["extra"].format(system=system)
    if not os.path.exists(p):
        return None
    d = pd.read_csv(p, sep="\t", dtype={"tf": str})
    d = d[(d.system == system) & (d.tf == tf) & (d.scope == "ALL6+SOTA") & (d.method == GB) & (d.target == "resp")]
    return float(d.AUROC.iloc[0]) if len(d) else None


def run_tf(system, tf, blk, K, B, assert_parity, L, outdir):
    cfg = L.SYSTEMS[system]
    bank = load_bank(cfg["axis"])
    ep = L.build_endpoint(system, tf, blk, False)                                                        # scores -> catbank -> truth (lattice order)
    V, genes = ep["V"], np.asarray(ep["genes"]).astype(str)
    use = [m for m in METHODS if m in V and m in bank.X]
    lost = {m: ("no cognate row" if m not in V else "no bank matrix") for m in METHODS if m not in use}
    if GB not in use:
        raise L.Blocked(f"GB_ROW_OR_BANK_MISSING {system}/{tf}")
    meta = dict(system=system, tf=tf, methods_used=use, methods_missing=lost, bank_size=len(bank.members), n_genes=len(genes))
    ai = np.array([bank.ai.get(g, -1) for g in genes])
    if (ai < 0).mean() > 0.001:
        raise L.Blocked(f"LATTICE_GENES_NOT_ON_CATBANK_AXIS {system}/{tf}: {(ai < 0).sum()}")
    has = ai >= 0
    tc = np.where(has, bank.tcol[np.where(has, ai, 0)], -1)
    r = bank.row_of(tf)
    meta["focal_in_bank"] = r is not None
    # ---- universe: lattice ALL6+SOTA mask, then T7, then every method's prior finite
    M0 = np.asarray(ep["U"], bool) & np.all([np.isfinite(V[m]) for m in use], 0)
    prior = {m: np.where(tc >= 0, bank.prior(m, r)[np.where(has, ai, 0)], np.nan) for m in use}
    M1 = M0 & (tc >= 0) & np.all([np.isfinite(prior[m]) for m in use], 0)
    meta.update(n_M0_all6sota=int(M0.sum()), n_M0_not_T7=int((M0 & (tc < 0)).sum()), n_M1=int(M1.sum()))
    # ---- parity: lattice GB AUROC(resp) on M0 and GB cognate == catbank P_pairU row / lattice `prior GB`
    par = {}
    q0 = point(V[GB], ep["resp"], M0)
    a_ref, a_now = lattice_ref_auroc(system, tf), (q0["AUROC"] if q0 else None)
    par["lattice_resp_AUROC_ref"], par["resp_AUROC_M0_now"] = a_ref, a_now
    if assert_parity and (a_ref is None or a_now is None):
        raise L.Blocked(f"REFUSING_PARITY_NO_LATTICE_REFERENCE: {system}/{tf}: lattice ALL6+SOTA GB resp AUROC reference {a_ref!r} / recomputed {a_now!r} (file or row missing)")
    if a_ref is not None and a_now is not None:
        par["resp_AUROC_absdiff"] = abs(a_ref - a_now)
        if assert_parity and abs(a_ref - a_now) > 1e-9:
            raise L.Blocked(f"REFUSING_PARITY: {system}/{tf} GB resp AUROC on ALL6+SOTA {a_now!r} != lattice {a_ref!r}")
    if assert_parity and r is None:
        raise L.Blocked(f"REFUSING_PARITY_FOCAL_NOT_IN_BANK: {system}/{tf}: no bank row to check the cognate GB row against")
    if r is not None:
        for m in use:
            v = np.where(has, bank.X[m][r][np.where(has, ai, 0)], np.nan)
            f = np.isfinite(v) & np.isfinite(V[m])
            par[f"row_absdiff|{m}"] = float(np.abs(v[f] - V[m][f]).max()) if f.any() else None
            par[f"row_finite_mismatch|{m}"] = int((np.isfinite(v) != np.isfinite(V[m])).sum())
        if assert_parity and (par[f"row_absdiff|{GB}"] or 0) > 1e-5:
            raise L.Blocked(f"REFUSING_PARITY: {system}/{tf} GB cognate row differs from the catbank P_pairU bank row by {par[f'row_absdiff|{GB}']:.3e}")
    if "prior GB" in V:
        f = M1 & np.isfinite(V["prior GB"])
        par["prior_GB_absdiff_vs_lattice"] = float(np.abs(prior[GB][f] - V["prior GB"][f]).max()) if f.any() else None
        if assert_parity and (par["prior_GB_absdiff_vs_lattice"] or 0) > 1e-9:
            raise L.Blocked(f"REFUSING_PARITY: {system}/{tf} GB leave-focal-out prior differs from the lattice `prior GB`")
    meta["parity"] = par
    # ---- pseudo-regulators: observational matching only
    F = bank.feats[~bank.feats.index.str.upper().duplicated(keep=False)]              # upper-case join (mouse symbols); symbols shared by >1 fixture gene are dropped -> focal blocked, never guessed
    F = F.set_axis(F.index.str.upper())
    Fb = F.reindex(bank.mu)[FEATS].to_numpy(float)
    fcomp = np.isfinite(Fb).all(1)
    mu, sd = Fb[fcomp].mean(0), Fb[fcomp].std(0)
    if tf.upper() not in F.index or not np.isfinite(F.loc[tf.upper(), FEATS].to_numpy(float)).all():
        raise L.Blocked(f"FOCAL_FEATURES_MISSING {system}/{tf}")
    f_r = (F.loc[tf.upper(), FEATS].to_numpy(float) - mu) / sd
    Fz = (Fb - mu) / sd
    perturbed = perturbed_set(system, L, __import__("gb_lattice_extra"))
    sel = np.flatnonzero(M1)
    rows_ok = np.all([np.isfinite(bank.X[m][:, ai[sel]]).all(1) & bank.row_ok[m] for m in use], 0)
    is_r = (bank.mu == tf.upper())
    is_pert = np.isin(bank.mu, list(perturbed))
    elig = fcomp & rows_ok & ~is_r & ~is_pert
    meta["pseudo_exclusions"] = dict(n_bank=len(bank.members), focal=int(is_r.sum()), perturbed_panel=int(is_pert.sum()), missing_features=int((~fcomp).sum()),
                                     row_not_finite_on_M1_any_method=int((fcomp & ~rows_ok & ~is_r & ~is_pert).sum()), eligible=int(elig.sum()))
    pi, pd_ = pick_pseudos(Fz, f_r, elig, K, bank.members)
    meta.update(pseudos=[bank.members[i] for i in pi], pseudo_dist=[float(x) for x in pd_], K_eff=int(len(pi)),
                match_absdiff_std=[float(x) for x in np.abs(Fz[pi] - f_r).mean(0)] if len(pi) else None, match_features=FEATS)
    if len(pi) < 5:
        raise L.Blocked(f"TOO_FEW_PSEUDOS {system}/{tf}: {len(pi)} eligible (need >= 5)")
    PS = {m: np.where(has[None, :], bank.X[m][pi][:, np.where(has, ai, 0)], np.nan) for m in use}          # legacy K=20 feature-matched subset
    PP = {m: np.stack([np.where(tc >= 0, bank.prior(m, int(p))[np.where(has, ai, 0)], np.nan) for p in pi]) for m in use}
    # ---- WHOLE eligible bank: per-row metric vectors, truth-free loading on its own leave-out prior, covariates
    E = np.flatnonzero(elig)
    ai0 = np.where(has, ai, 0)
    Xe = {m: np.where(has[None, :], bank.X[m][E][:, ai0], np.nan) for m in use}
    tcz = np.where(tc >= 0, tc, 0)
    load_b, load_f = {}, {}
    for m in use:
        Pe = np.where((tc >= 0)[None, :], bank.prior_rows(m, E)[:, tcz], np.nan)
        load_b[m] = spearman_rows(Xe[m][:, M1], Pe[:, M1])
        load_f[m] = float(spearman_rows(V[m][M1][None, :], prior[m][M1][None, :])[0])
    Fraw = F.reindex(bank.mu)[FEATS].to_numpy(float)
    f_raw = F.loc[tf.upper(), FEATS].to_numpy(float)
    meta.update(n_eligible_bank=int(len(E)), loading_focal=load_f, loading_bank_median={m: float(np.nanmedian(load_b[m])) for m in use},
                loading_bank_q05_q95={m: [float(np.nanpercentile(load_b[m], 5)), float(np.nanpercentile(load_b[m], 95))] for m in use})
    # ---- metrics
    T = L.targets(ep, M1)
    counts = blk.counts[:B]
    rows_out = []
    npz = {"bank_names": np.asarray([bank.members[i] for i in E]), **{f"loading|{m}": load_b[m] for m in use}}
    for tname in TARGETS:
        if tname not in T:
            continue
        y, rows = T[tname]
        gate = bool(L.gate_of(ep, y, rows, M1))
        yy = np.asarray(y, bool)[rows]
        mets = METRICS.get(tname, ("AUROC",))
        for m in use:
            base = dict(system=system, block=ep["block"], tf=tf, method=m, target=tname, n=int(rows.sum()), n_pos=int(yy.sum()), n_neg=int((~yy).sum()), gate=gate,
                        n_M1=int(M1.sum()), K_eff=len(pi), focal_in_bank=r is not None, loading_focal=load_f[m])
            st = block_stats(V[m], prior[m], PS[m], PP[m], y, rows, mets)
            if st is None:
                rows_out += [dict(base, metric=k, status="TOO_FEW") for k in mets]
                continue
            bm = [point(Xe[m][i], y, rows) for i in range(len(E))]                                             # every eligible bank row on the SAME rows / labels
            bc = CT.eval_rows(V[m], y, rows, ep["gi"], counts)                                             # paired resamples: identical `counts` for cognate and prior
            bp = CT.eval_rows(prior[m], y, rows, ep["gi"], counts)
            Cb, cf = np.c_[load_b[m], Fraw[E]], np.r_[load_f[m], f_raw]
            for k in mets:
                d = bc[1][k] - bp[1][k]
                lo, hi, nb = L.pct(d)
                yb = np.array([q[k] if q else np.nan for q in bm], float)
                cal, res, zres = bank_calibration(yb, Cb, st[k]["cognate"], cf)
                npz[f"val|{m}|{tname}|{k}"], npz[f"resid|{m}|{tname}|{k}"], npz[f"zres|{m}|{tname}|{k}"] = yb, res, zres
                dc = bc[1][k] - cal.get("pred_focal", np.nan)                                                  # CONDITIONAL on the point-estimate bank fit (bank not re-fit per resample)
                clo, chi, _ = L.pct(dc)
                rows_out.append(dict(base, metric=k, status="OK" if gate else "DESCRIPTIVE_ONLY", **st[k], **cal, E_prior_lo=lo, E_prior_hi=hi, E_prior_boot_se=float(np.nanstd(d)) if nb else np.nan,
                                     nboot=nb, E_cal_lo_cond=clo, E_cal_hi_cond=chi))
    os.makedirs(f"{outdir}/bank_metrics", exist_ok=True)
    np.savez_compressed(f"{outdir}/bank_metrics/{system}_{tf}.npz", **npz)
    meta["files"] = {k: v for k, v in ep["files"].items()}
    return rows_out, meta


def tf_list(system, L, LE):
    if system in TF_OVERRIDE:
        return TF_OVERRIDE[system]
    t = L.SYSTEMS[system]["tfs"]
    return list(t) if t is not None else L.k562_tfs()


def cmd_run(a):
    import gb_lattice_extra as LE
    import gb_layer_lattice as L
    LE.register()
    os.makedirs(a.out, exist_ok=True)
    for system in [s.strip() for s in a.systems.split(",") if s.strip()]:
        if system not in SYSTEMS:
            raise SystemExit(f"REFUSING: {system} not in {SYSTEMS}")
        cfg = L.SYSTEMS[system]
        allowed = tf_list(system, L, LE)
        tfs = [t.strip() for t in a.tfs.split(",") if t.strip()] or allowed
        bad = [t for t in tfs if t not in allowed]
        if bad:
            raise SystemExit(f"REFUSING: --tfs {bad} not in the declared list of {system}: {allowed}")
        seed = int(np.random.SeedSequence([L.BOOT_SEED, L.BLOCKS.index(cfg["block"])]).generate_state(1)[0])
        blk = L.BlockAxis(L.axis_of(cfg["axis"]), a.B, seed, None, "gene")
        rows, metas = [], {}
        for tf in tfs:
            t0 = time.time()
            try:
                r_, m_ = run_tf(system, tf, blk, a.K, a.B, a.assert_parity, L, a.out)
            except L.Blocked as e:
                rows.append(dict(system=system, tf=tf, status=f"NOT_RUN: {e}"))
                metas[tf] = dict(status=f"NOT_RUN: {e}")
                print(f"[attrib] {system}/{tf}: NOT_RUN {e}", flush=True)
                continue
            rows += r_
            metas[tf] = m_
            print(f"[attrib] {system}/{tf}: M1={m_['n_M1']} N_bank={m_['n_eligible_bank']} K_eff={m_['K_eff']} {time.time()-t0:.0f}s", flush=True)
        tag = "all" if not a.tfs else "_".join(tfs)
        pd.DataFrame(rows).to_csv(f"{a.out}/part_{system}_{tag}.tsv", sep="\t", index=False)
        L.atomic_json(f"{a.out}/part_{system}_{tag}.meta.json", dict(system=system, K=a.K, B=a.B, seed=seed, script_sha256=L.sha_file(__file__), module_shas=L.module_shas(),
                                                                     bank_sha=(load_bank(cfg["axis"]).sha if metas else None),
                                                                     bank_notes=(load_bank(cfg["axis"]).notes if metas else None), tfs=metas))


def check_parts(metas, ok):
    """merge refuses: mismatched K / B / script sha across parts, and duplicate (system, tf, method, target, metric) keys"""
    for k in ("K", "B", "script_sha256"):
        if len({m[k] for m in metas}) != 1:
            raise SystemExit(f"REFUSING_MERGE_MISMATCHED_{k.upper()}: {sorted({str(m[k]) for m in metas})}")
    dup = ok.duplicated(["system", "tf", "method", "target", "metric"], keep=False)
    if dup.any():
        raise SystemExit(f"REFUSING_MERGE_DUPLICATE_KEYS: {ok[dup][['system','tf','method','target','metric']].drop_duplicates().head(10).to_dict('records')}")


def cmd_merge(a):
    import gb_layer_lattice as L
    parts = sorted(glob.glob(f"{a.out}/part_*.tsv"))
    if not parts:
        raise SystemExit("REFUSING: no part_*.tsv")
    pt = pd.concat([pd.read_csv(p, sep="\t", dtype={"tf": str}) for p in parts], ignore_index=True)
    ok = pt[pt.status.isin(["OK", "DESCRIPTIVE_ONLY", "TOO_FEW"])]
    notrun = pt[pt.status.astype(str).str.startswith("NOT_RUN")]
    metas = [json.load(open(p)) for p in sorted(glob.glob(f"{a.out}/part_*.meta.json"))]
    if len(metas) != len(parts):
        raise SystemExit("REFUSING_MERGE: part_*.tsv and part_*.meta.json counts differ")
    check_parts(metas, ok)
    resid = {}
    for (s_, tf), g in ok.groupby(["system", "tf"]):
        f = f"{a.out}/bank_metrics/{s_}_{tf}.npz"
        if not os.path.exists(f):
            raise SystemExit(f"REFUSING_MERGE: missing {f}")
        z = np.load(f, allow_pickle=True)
        names = [str(x) for x in z["bank_names"]]
        resid[(s_, tf)] = {(m, t, k): dict(zip(names, z[f"zres|{m}|{t}|{k}"])) for m, t, k in g[["method", "target", "metric"]].itertuples(index=False) if f"zres|{m}|{t}|{k}" in z.files}
    ok.to_csv(f"{a.out}/per_tf_method.tsv", sep="\t", index=False)
    S = summarize(ok, resid)
    S.to_csv(f"{a.out}/summary.tsv", sep="\t", index=False)
    L.atomic_json(f"{a.out}/receipt.json", dict(
        script_sha256=L.sha_file(__file__), parts={os.path.basename(p): L.sha_file(p) for p in parts}, n_rows=len(ok), n_summary=len(S), margin_aplift=MARGIN,
        not_run=notrun[["system", "tf", "status"]].to_dict("records"), K=sorted({m["K"] for m in metas}), B=sorted({m["B"] for m in metas}),
        module_shas=[m["module_shas"] for m in metas][:1], bank_input_sha256={m["system"]: m.get("bank_sha") for m in metas},
        coverage={m["system"]: {tf: {k: v for k, v in x.items() if k in ("n_genes", "n_M0_all6sota", "n_M0_not_T7", "n_M1", "K_eff", "n_eligible_bank", "methods_missing", "pseudo_exclusions", "focal_in_bank", "status",
                                                                             "loading_focal", "loading_bank_median", "loading_bank_q05_q95")}
                                for tf, x in m["tfs"].items()} for m in metas},
        parity={m["system"]: {tf: x.get("parity") for tf, x in m["tfs"].items()} for m in metas},
        pseudos_k20_legacy={m["system"]: {tf: x.get("pseudos") for tf, x in m["tfs"].items()} for m in metas},
        method="PRIMARY null = whole eligible bank, OLS-adjusted for truth-free covariates [loading = Spearman(row, own leave-out prior) on the mask, raw_det, abundance, |rho(Ms,DPT)|]; "
               "E_cal = focal - predicted, p_cal = (1+#{bank residual >= E_cal})/(N+1).  Reported beside it: unadjusted whole-bank percentile and the legacy K=20 feature-matched values.  "
               "Bootstrap CI of E_cal is CONDITIONAL on the point-estimate bank fit (bank not re-fit per resample); percentile p_cal is the null calibration.  Summary is DESCRIPTIVE ONLY; "
               "p_substrate uses an aligned bank null (mean over the substrate's gated TFs of each bank row's residual); regulator pools overlap across TFs (non-independent).",
        bank_note="the catbank bank = species TF catalogue ∩ detected (raw spliced detection >= 5%); it includes cofactors / chromatin regulators / non-DNA-binding genes, not only sequence-specific TFs",
        pseudo_exclusions_note="focal, ALL perturbed-panel TFs of the system (K562: every TF in the state file), genes with missing features, rows non-finite on the mask for any method",
        note="one mask per (system, TF) shared by all methods and vectors; regulator genes of the catbank bank are not targets (T7); truth read only inside gb_layer_lattice.build_endpoint after scores"))
    print(f"[attrib] merged {len(parts)} parts: {len(ok)} rows, {len(S)} summary rows, {len(notrun)} NOT_RUN -> {a.out}")


def cmd_tfs(a):
    import gb_lattice_extra as LE
    import gb_layer_lattice as L
    LE.register()
    allt = []
    for s in a.systems.split(","):
        allt += tf_list(s.strip(), L, LE)
    sel = allt[a.slice::a.nslice]
    if not sel:
        raise SystemExit("REFUSING: empty TF slice (an empty --tfs would run ALL TFs)")
    print(",".join(sel))


# ===================================================================================================== selftest (synthetic only)
def _world(seed, nb=80, G=1800, a=0.0, noise=1.0, c=1.0, frac=0.25):
    """bank rows S[r,g] = c*t_g + noise*eps ; labels y_g ~ Bernoulli(sigmoid(t_g - 1)) [target-level propensity] ; focal row r0 gets + a*y_g (regulator-specific)"""
    rg = np.random.default_rng(seed)
    t = rg.normal(size=G)
    y = rg.random(G) < 1 / (1 + np.exp(-(1.5 * t - 1.2)))
    X = c * t[None, :] + noise * rg.normal(size=(nb, G))
    X[0] += a * y
    return X, y


def _attribute_synthetic(X, y, K=20, seed=0):
    """full pipeline on a synthetic bank via the SAME prior / pseudo / stats code: focal = bank row 0; bank = all rows; T = all columns (self entries of the first nb columns
    are NaN in the bank, so pass T over the columns nb..G, as the real T7 excludes bank genes)."""
    nb, G = X.shape
    T = np.zeros(G, bool)
    T[nb:] = True
    cols = np.arange(nb)
    Pz = CB.sota_prior_arrays(X, cols, T, 2)
    idx = np.flatnonzero(T)
    prior = lambda r: CB.loo_prior(Pz["Z"], Pz["S"], Pz["C"], r, 2)
    rows = np.ones(len(idx), bool)
    ys = y[idx]
    ps = np.random.default_rng(seed).choice(np.arange(1, nb), K, replace=False)             # random (feature-free) pseudo rows: matching itself is tested separately
    st = block_stats(X[0, idx], prior(0), X[ps][:, idx], np.stack([prior(int(p)) for p in ps]), ys, rows, ("AUROC", "AUPRC", "APLIFT"))
    return st


def selftest():
    # 1 prior construction == brute force mean of OTHER rows' population-z on T
    rg = np.random.default_rng(1)
    X = rg.normal(size=(12, 60))
    T = np.zeros(60, bool)
    T[12:] = True
    Pz = CB.sota_prior_arrays(X, np.arange(12), T, 2)
    Xc = X.copy()
    Xc[np.arange(12), np.arange(12)] = np.nan
    Zb = np.stack([(Xc[i, T] - np.nanmean(Xc[i, T])) / np.nanstd(Xc[i, T]) for i in range(12)])
    for r in (0, 5, 11):
        bf = np.delete(Zb, r, 0).mean(0)
        assert np.allclose(CB.loo_prior(Pz["Z"], Pz["S"], Pz["C"], r, 2), bf, atol=1e-12), "loo prior != brute force"
    assert np.allclose(CB.loo_prior(Pz["Z"], Pz["S"], Pz["C"], None, 2), Zb.mean(0), atol=1e-12)
    # 2 pseudo_p tie handling and margin flag
    assert pseudo_p(0.7, [0.7, 0.6, 0.5]) == (1 + 1) / 4, "ties must count against the cognate"
    assert pseudo_p(0.9, [0.1] * 19) == 1 / 20 and pseudo_p(0.0, [0.5] * 19) == 1.0
    assert margin_flag(0.3).startswith("cognate") and margin_flag(-0.3).startswith("ref") and margin_flag(0.0) == "within +-0.20"
    # 3 nearest-neighbour pseudo picking: known geometry, ties by name, ineligible rows never picked
    F = np.array([[0.0, 0], [1, 0], [0.5, 0], [-1, 0], [1, 0]])
    i, d = pick_pseudos(F, [0.0, 0], [False, True, True, True, True], 3, ["a", "b", "c", "d", "e"])
    assert list(i) == [2, 1, 3], f"got {i}"                                                        # 0.5 first; the distance-1 tie among b, d, e -> alphabetical b, d
    assert 0 not in i
    # 4 point() = brute-force AUROC with ties, and AP-lift of a perfect ranking
    s, y = np.array([3.0, 2, 2, 1, 0]), np.array([1, 1, 0, 0, 1], bool)
    pa = point(s, y, np.ones(5, bool))["AUROC"]
    bf = np.mean([(a > b) + 0.5 * (a == b) for a in s[y] for b in s[~y]])
    assert abs(pa - bf) < 1e-12, (pa, bf)
    assert abs(point(np.array([3.0, 2, 1, 0]), np.array([1, 1, 0, 0], bool), np.ones(4, bool))["APLIFT"] - 2.0) < 1e-12
    # 5 PLANTED regulator-specific signal: E_prior > 0 and the cognate is extreme against both nulls
    ep_, pp_, pf_ = [], [], []
    for sd in range(12):
        X, y = _world(sd, a=1.5)
        s = _attribute_synthetic(X, y, seed=sd)["AUROC"]
        ep_.append(s["E_prior"]), pp_.append(s["pseudo_p"]), pf_.append(s["pseudo_focal_p"])
    assert np.mean(ep_) > 0.05 and np.mean(np.array(pp_) <= 0.05) > 0.9 and np.mean(np.array(pf_) <= 0.05) > 0.9, (np.mean(ep_), pp_, pf_)
    # 6 PURE target-level signal, low row noise: E_prior ~ 0 (|mean| < 0.02) and the cognate is NOT extreme against the nulls (uniform-ish p)
    ep0, pp0, pf0 = [], [], []
    for sd in range(40):
        X, y = _world(100 + sd, a=0.0, noise=0.3)
        s = _attribute_synthetic(X, y, seed=sd)["AUROC"]
        ep0.append(s["E_prior"]), pp0.append(s["pseudo_p"]), pf0.append(s["pseudo_focal_p"])
    assert abs(np.mean(ep0)) < 0.02, np.mean(ep0)
    assert np.mean(np.array(pp0) <= 0.05) < 0.15 and np.mean(np.array(pf0) <= 0.05) < 0.15 and 0.3 < np.mean(pf0) < 0.7, (np.mean(np.array(pp0) <= 0.05), np.mean(pf0))
    # 7 pure target-level with NOISY rows: E_prior is NEGATIVE (one row vs a bank average) -- the pseudo-focal null absorbs it, so the calibrated call stays null
    en, pfn = [], []
    for sd in range(40):
        X, y = _world(200 + sd, a=0.0, noise=1.5)
        s = _attribute_synthetic(X, y, seed=sd)["AUROC"]
        en.append(s["E_prior"] - s["null_E_prior_median"]), pfn.append(s["pseudo_focal_p"])
        assert s["E_prior"] < 0.02
    assert abs(np.mean(en)) < 0.03 and np.mean(np.array(pfn) <= 0.05) < 0.15, (np.mean(en), np.mean(np.array(pfn) <= 0.05))
    # 8 loading-adjusted whole-bank null: spearman_rows == scipy; Bank.prior_rows == loo_prior; OLS calibration exact on a linear toy; Holm; aligned substrate p; merge guards
    from scipy.stats import spearmanr
    A_, B_ = rg.normal(size=(5, 40)), rg.normal(size=(5, 40))
    B_[0] += 2 * A_[0]
    assert np.allclose(spearman_rows(A_, B_), [spearmanr(A_[i], B_[i])[0] for i in range(5)], atol=1e-12)
    X12 = np.random.default_rng(1).normal(size=(12, 60))
    bk = Bank([f"g{i}" for i in range(12)], [f"g{j}" for j in range(60)], T, {"m": X12})
    assert np.allclose(bk.prior_rows("m", np.arange(12))[[0, 5, 11]], np.stack([CB.loo_prior(bk.P["m"]["Z"], bk.P["m"]["S"], bk.P["m"]["C"], r, 2) for r in (0, 5, 11)]), atol=1e-12)
    xl = np.random.default_rng(3).uniform(size=(200, 1))
    yb_ = 0.5 + 0.3 * xl[:, 0] + 0.01 * np.random.default_rng(4).normal(size=200)
    cal, res, _z = bank_calibration(yb_, xl, 0.5 + 0.3 * 0.4 + 0.25, [0.4])
    assert abs(cal["E_cal"] - 0.25) < 0.01 and abs(cal["p_cal"] - 1 / 201) < 1e-12 and cal["R2_bank"] > 0.98 and abs(cal["beta_loading"] - 0.3) < 0.01, cal
    assert np.allclose(holm([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06]) and np.isnan(holm([np.nan, .5])[0])
    assert substrate_p([0.5], [{"a": 0.1, "b": 0.6, "c": 0.0}])[0] == (1 + 1) / 4
    assert substrate_p([0.5, 0.5], [{"a": 0.1, "b": 0.6}, {"a": 0.1, "b": 0.6}])[0] == (1 + 1) / 3          # aligned: row b is high in BOTH TFs
    okd = pd.DataFrame([dict(system="S", tf="t", method="m", target="resp", metric="AUROC", K=20)] * 2)
    try:
        check_parts([dict(K=20, B=200, script_sha256="x")], okd)
        raise AssertionError("duplicate keys not refused")
    except SystemExit:
        pass
    try:
        check_parts([dict(K=20, B=200, script_sha256="x"), dict(K=20, B=200, script_sha256="y")], okd.iloc[:1])
        raise AssertionError("mismatched sha not refused")
    except SystemExit:
        pass
    # 9 LOADING confound: a PURE target-level focal with an EXTREME loading looks specific against the unadjusted whole bank (p_bank small) but not after adjustment;
    #   a planted regulator-specific focal with a typical loading stays extreme after adjustment.
    def loading_world(sd, lam_focal, a):
        r_ = np.random.default_rng(sd)
        nb_, G_ = 120, 1500
        t = r_.normal(size=G_)
        y = r_.random(G_) < 1 / (1 + np.exp(-(1.5 * t - 1.2)))
        lam = np.r_[lam_focal, r_.uniform(0.1, 1.2, nb_ - 1)]
        Xw = lam[:, None] * t[None, :] + 0.8 * r_.normal(size=(nb_, G_))
        Xw[0] += a * y
        Tw = np.zeros(G_, bool)
        Tw[nb_:] = True
        bkw = Bank([f"g{i}" for i in range(nb_)], [f"g{j}" for j in range(G_)], Tw, {"m": Xw})
        idx, E = np.flatnonzero(Tw), np.arange(1, nb_)
        ones = np.ones(len(idx), bool)
        pf = CB.loo_prior(bkw.P["m"]["Z"], bkw.P["m"]["S"], bkw.P["m"]["C"], 0, 2)
        lb = spearman_rows(Xw[E][:, idx], bkw.prior_rows("m", E))
        lf = spearman_rows(Xw[0, idx][None], pf[None])[0]
        yb = np.array([point(Xw[p_, idx], y[idx], ones)["AUROC"] for p_ in E])
        yf = point(Xw[0, idx], y[idx], ones)["AUROC"]
        cb_, cf_ = np.c_[lb, r_.normal(size=(len(E), 2))], np.r_[lf, r_.normal(size=2)]
        return bank_calibration(yb, cb_, yf, cf_)[0]
    conf = [loading_world(300 + sd, 1.15, 0.0) for sd in range(30)]
    assert np.mean([c["p_bank"] <= 0.1 for c in conf]) > 0.6 and np.mean([c["p_cal"] <= 0.05 for c in conf]) < 0.15 and 0.3 < np.mean([c["p_cal"] for c in conf]) < 0.7, \
        (np.mean([c["p_bank"] <= 0.1 for c in conf]), np.mean([c["p_cal"] <= 0.05 for c in conf]), np.mean([c["p_cal"] for c in conf]))
    spec = [loading_world(400 + sd, 0.6, 1.5) for sd in range(30)]
    assert np.mean([c["p_cal"] <= 0.05 for c in spec]) > 0.85, np.mean([c["p_cal"] <= 0.05 for c in spec])
    # 10 HETEROSCEDASTICITY: residual spread grows with a covariate and the focal sits in the high-spread region (pure target-level): the standardized p keeps its size, the
    #    unstandardized global p is inflated (proves the case is sensitive)
    def het_case(sd_):
        r_ = np.random.default_rng(sd_)
        N_ = 400
        C_ = r_.uniform(size=(N_, 3))
        sdv = lambda c0: 0.02 + 0.1 * c0
        yb = 0.5 + 0.2 * C_[:, 0] + 0.1 * C_[:, 1] + sdv(C_[:, 0]) * r_.normal(size=N_)
        cf = np.r_[0.97, r_.uniform(size=2)]
        yf = 0.5 + 0.2 * cf[0] + 0.1 * cf[1] + sdv(cf[0]) * r_.normal()
        return bank_calibration(yb, C_, yf, cf)[0]
    het = [het_case(500 + sd_) for sd_ in range(300)]
    size_std, size_glob = np.mean([c["p_cal"] <= 0.05 for c in het]), np.mean([c["p_cal_global"] <= 0.05 for c in het])
    assert size_std <= 0.08 and size_glob > 0.08, (size_std, size_glob, np.mean([c["p_cal_knn"] <= 0.05 for c in het]))
    assert all(c["k_nn"] == 200 for c in het[:3])
    # 11 substrate_p bracket: independent rows -> both nulls have size <= .08; SHARED regulator-row effect with an identical focal row across TFs -> the aligned null is exact
    #    (size <= .08) while the independent-draw null is anti-conservative there
    def sub_sim(sd_, shared):
        r_ = np.random.default_rng(sd_)
        nt, N_ = 3, 150
        if shared:
            u, uf = r_.normal(size=N_), r_.normal()
            tfz = [dict(zip(map(str, range(N_)), u + 0.5 * r_.normal(size=N_))) for _ in range(nt)]
            fz = uf + 0.5 * r_.normal(size=nt)
        else:
            tfz = [dict(zip(map(str, range(N_)), r_.normal(size=N_))) for _ in range(nt)]
            fz = r_.normal(size=nt)
        return substrate_p(fz, tfz, n_mc=1000, seed=sd_)
    ind = np.array([sub_sim(600 + i, False) for i in range(300)])
    shr = np.array([sub_sim(900 + i, True) for i in range(300)])
    assert (ind[:, 0] <= 0.05).mean() <= 0.08 and (ind[:, 1] <= 0.05).mean() <= 0.08, ((ind[:, 0] <= 0.05).mean(), (ind[:, 1] <= 0.05).mean())
    assert (shr[:, 0] <= 0.05).mean() <= 0.08, (shr[:, 0] <= 0.05).mean()
    # 12 summarize(): macro over gated TFs only, calibrated-excess margin flag for AP-lift only, ONE primary cell (GB x resp AUROC) with Holm across substrates
    rows_ = []
    for sysn in ("S1", "S2"):
        for meth in (GB, "m"):
            for k in ("APLIFT", "AUROC"):
                for g, e in ((True, .5), (True, .3), (False, 9.0)):
                    rows_.append(dict(system=sysn, tf=f"t{e}", method=meth, target="resp", metric=k, status="OK", gate=g, cognate=1.0, prior=.5, bank_median=.5, E_cal=e, E_prior=e, E_pseudo=e,
                                      p_cal=.04, p_cal_global=.04, p_cal_knn=.04, z_cal=1.0, N_bank=100, p_bank=.04, pseudo_p=.04, pseudo_focal_p=.04))
    Sm = summarize(pd.DataFrame(rows_))
    assert len(Sm) == 8 and (Sm.n_tf == 2).all() and abs(Sm.mean_E_cal.iloc[0] - .4) < 1e-12 and Sm.primary_cell.sum() == 2 and (Sm[Sm.primary_cell].metric == "AUROC").all()
    assert Sm[(Sm.metric == "APLIFT") & (Sm.system == "S1")].margin_E_cal.iloc[0].startswith("cognate") and pd.isna(Sm[Sm.metric == "AUROC"].margin_E_cal.iloc[0]) and (Sm.N_bank_min == 100).all()
    print("[attrib] selftest OK: prior==brute force, ties, NN picking, AUROC/APLIFT, planted specific, pure target-level (+ noisy), loading-adjusted bank null, heteroscedastic size "
          f"(std {size_std:.3f} vs global {size_glob:.3f}), substrate bracket size (indep {(ind[:, 0] <= .05).mean():.3f}/{(ind[:, 1] <= .05).mean():.3f}; shared-row aligned {(shr[:, 0] <= .05).mean():.3f}, "
          f"indep {(shr[:, 1] <= .05).mean():.3f}), merge guards, summary")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", choices=["run", "merge", "tfs"])
    ap.add_argument("--systems", default=",".join(SYSTEMS))
    ap.add_argument("--tfs", default="")
    ap.add_argument("--K", type=int, default=20)
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--nslice", type=int, default=1)
    ap.add_argument("--no-assert-parity", dest="assert_parity", action="store_false", help="parity checks are ON by default (a failure / missing lattice reference marks the TF NOT_RUN)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "merge":
        cmd_merge(a)
    elif a.cmd == "tfs":
        cmd_tfs(a)
    else:
        ap.error("need a command or --selftest")


if __name__ == "__main__":
    main()
