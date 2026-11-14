#!/usr/bin/env python
"""Evaluate full scores, target priors and residual components."""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
import types
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gb_attribution as A  # noqa: E402
import gb_canon_table as CT  # noqa: E402
import gb_decompose as D  # noqa: E402

R = A.R
V3 = f"{R}/results/real_uniform/attribution_v3"
GB = A.GB
TARGETS = A.TARGETS                                 # resp, AB_prim, CD_prim, occ_prim, AC_prim, BD_prim
PRETTY = {"resp": "KO response", "occ_prim": "ChIP occupancy", "AB_prim": "A>B", "CD_prim": "C>D", "AC_prim": "A>C", "BD_prim": "B>D", "regulon": "regulon"}
COMPS_ = ("full", "prior", "resid")
MIN_CLASS = 20
EPS_CLASS = 0.01
N_TFBOOT = 10000
REGULON_SYSTEMS = ["PANC", "BONEMARROW", "GATA1", "DENTATEGYRUS", "GASTRULATION", "ZEBRAFISH"]
F_TARGETS, O_TARGETS = ("resp", "AB_prim", "CD_prim"), ("occ_prim", "AC_prim", "BD_prim")


# pure numerics (unit-tested by --selftest)
# AUROC (Mann-Whitney, average ranks for ties == the tie-grouped AUROC of gb_canon_table.weighted_metrics) of every row of M (N, n) against y (n,) bool
# NaN if a class is empty
def auroc_rows(M, y, chunk=256):
    from scipy.stats import rankdata
    M, y = np.asarray(M, float), np.asarray(y, bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    out = np.full(len(M), np.nan)
    if n1 == 0 or n0 == 0 or len(M) == 0:
        return out
    for a in range(0, len(M), chunk):
        r = rankdata(M[a:a + chunk], axis=1)
        out[a:a + chunk] = (r[:, y].sum(1) - n1 * (n1 + 1) / 2.0) / (n1 * n0)
    return out


# frozen decomposition: S = row-z(v) over `mask` (population sd), P = prior (as given), R = S - P
# Returns (S, P, R) with NaN outside the mask
def decomp_vectors(v, prior, mask):
    S = D.zrow(v, mask)
    P = np.where(mask, np.asarray(prior, float), np.nan)
    if not np.isfinite(P[mask]).all():
        raise RuntimeError("decomp_vectors: non-finite prior inside the mask")
    return S, P, S - P


# Descriptive class of one regulator from AUC_full, AUC_prior and whether the residual exceeds the decoy 95th percentile
def classify(auc_full, auc_prior, resid_above95):
    if not (np.isfinite(auc_full) and np.isfinite(auc_prior)) or resid_above95 is None or (isinstance(resid_above95, float) and not np.isfinite(resid_above95)):
        return "NA"
    if auc_prior >= auc_full - EPS_CLASS and not resid_above95:
        return "target-common-dominated"
    if resid_above95 and auc_full - auc_prior > EPS_CLASS:
        return "regulator-specific"
    return "mixed"


# two-sided sign-flip p of the mean of per-unit differences d: exact over all 2^n flips for n <= 20 (floor 2/2^n), Monte-Carlo (+1 rule) otherwise
def signflip(d, n_mc=200000, seed=0):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 2:
        return np.nan, "n<2"
    if n <= 20:
        return D.sign_flip_p(d)["p_two"], f"exact_2^{n}"
    rg = np.random.default_rng(seed)
    obs = abs(d.mean())
    sc = max(np.abs(d).max(), 1e-300) * 1e-12
    cnt = 0
    for a in range(0, n_mc, 20000):
        s = rg.choice([-1.0, 1.0], size=(min(20000, n_mc - a), n))
        cnt += int((np.abs(s @ d / n) >= obs - sc).sum())
    return float((1 + cnt) / (n_mc + 1)), f"mc_{n_mc}"


# substrate-level upper-tail p of the TF-macro residual AUROC
def macro_decoy_p(focal, decoys, n_mc=20000, seed=0):
    if not decoys:
        return np.nan, np.nan, 0
    common = sorted(set.intersection(*[set(d) for d in decoys]))
    if not common:
        return np.nan, np.nan, 0
    M = np.array([[d[n_] for n_ in common] for d in decoys], float)
    M = M[:, np.isfinite(M).all(0)]
    if M.shape[1] == 0:
        return np.nan, np.nan, 0
    obs = float(np.mean(focal))
    na = M.mean(0)
    rg = np.random.default_rng(seed)
    ni = np.array([M[t][rg.integers(0, M.shape[1], n_mc)] for t in range(M.shape[0])]).mean(0)
    return float((1 + (na >= obs).sum()) / (len(na) + 1)), float((1 + (ni >= obs).sum()) / (n_mc + 1)), int(M.shape[1])


# rank-normal scores of x over `mask` (NaN outside): Phi^-1((rank - .5)/n)
def rank_normal(x, mask):
    from scipy.stats import norm, rankdata
    out = np.full(len(x), np.nan)
    v = np.asarray(x, float)[mask]
    out[mask] = norm.ppf((rankdata(v) - 0.5) / len(v))
    return out


# cross-fitted HistGradientBoostingRegressor(random_state=0) defaults on the fixed gene folds: held-out fit and residual over `mask`
def hgb_cross_fit(y, X, folds, mask):
    from sklearn.ensemble import HistGradientBoostingRegressor
    y, X, folds, mask = np.asarray(y, float), np.asarray(X, float), np.asarray(folds), np.asarray(mask, bool)
    fit = np.full(len(y), np.nan)
    for f in np.unique(folds[mask]):
        te, tr = mask & (folds == f), mask & (folds != f)
        fit[te] = HistGradientBoostingRegressor(random_state=0).fit(X[tr], y[tr]).predict(X[te])
    res = np.where(mask, y - fit, np.nan)
    tot = ((y[mask] - y[mask].mean()) ** 2).sum()
    return fit, res, dict(R2_heldout=float(1 - (res[mask] ** 2).sum() / tot))


def pct(x):
    return D.pct(x)


# the audit core (plain arrays; used by lattice and regulon endpoints)
# row-wise transform over the columns of X (N, n), all finite: 'z' = population z; 'rank' = normal scores Phi^-1((average rank - .5)/n)
def rownorm(X, scale):
    from scipy.stats import norm, rankdata
    X = np.asarray(X, float)
    if scale == "z":
        with np.errstate(invalid="ignore", divide="ignore"):
            return (X - X.mean(1, keepdims=True)) / X.std(1, keepdims=True)
    return norm.ppf((rankdata(X, axis=1) - 0.5) / X.shape[1])


# Decompose focal and decoy scores on M1 genes with finite priors.
def make_dec(Vm, priorm, Xb_m, Pb_m, M1, scale="z"):
    cols = np.flatnonzero(M1)
    P = np.asarray(priorm, float)[cols]
    keep = np.isfinite(P)
    v = np.asarray(Vm, float)[cols][keep]
    xb, pb = np.asarray(Xb_m, float)[:, keep], np.asarray(Pb_m, float)[:, keep]
    S = rownorm(v[None, :], scale)[0]
    with np.errstate(invalid="ignore", divide="ignore"):
        rb = rownorm(xb, scale) - pb
    okb = np.isfinite(rb).all(1) & (xb.std(1) > 0)
    rb[~okb] = np.nan
    return dict(keep=keep, S=v, P=P[keep], R=S - P[keep], rb=rb, ok=okb, load_b=A.spearman_rows(xb, pb), load_f=float(A.spearman_rows(v[None, :], P[keep][None, :])[0]),
                rho_SP=float(A.spearman_rows(S[None, :], P[keep][None, :])[0]), sd_P=float(P[keep].std()))


# AUROC of full / prior / resid of one (method, target) plus the decoy calibration
def dec_eval(dd, y, rows, cols, cov_b, cov_f):
    rows_M = np.asarray(rows, bool)[cols][dd["keep"]]
    yM = np.asarray(y, bool)[cols][dd["keep"]][rows_M]
    f_full, f_prior, f_res = (auroc_rows(dd[k][None, :][:, rows_M], yM)[0] for k in ("S", "P", "R"))
    au = np.full(len(dd["ok"]), np.nan)
    if dd["ok"].any():
        au[dd["ok"]] = auroc_rows(dd["rb"][dd["ok"]][:, rows_M], yM)
    fin = au[np.isfinite(au)]
    out = dict(AUROC_full=f_full, AUROC_prior=f_prior, AUROC_resid=f_res, N_decoy=int(len(fin)))
    z_all = np.full(len(au), np.nan)
    if len(fin):
        out.update(decoy_median=float(np.median(fin)), decoy_q95=float(np.percentile(fin, 95)), decoy_q99=float(np.percentile(fin, 99)),
                   p_decoy=float((1 + (fin >= f_res).sum()) / (len(fin) + 1)), pct_decoy_below=float((fin < f_res).mean()), resid_above_q95=bool(f_res > np.percentile(fin, 95)))
        if cov_f is not None:
            cal, _, z_all = A.bank_calibration(au, np.c_[dd["load_b"], np.asarray(cov_b, float)], f_res, np.r_[dd["load_f"], np.asarray(cov_f, float)])
            out.update(p_cal=cal.get("p_cal", np.nan), z_cal=cal.get("z_cal", np.nan), E_cal=cal.get("E_cal", np.nan), p_cal_knn=cal.get("p_cal_knn", np.nan), N_cal=cal.get("N_bank", 0))
    return out, au, z_all


# (primary class = adjusted decoy calibration p_cal <= .05, sensitivity class = whole-bank decoy 95th percentile)
def class_pair(auc_full, auc_prior, p_cal, resid_above_q95):
    pc = None if p_cal is None or not np.isfinite(p_cal) else bool(p_cal <= 0.05)
    return classify(auc_full, auc_prior, pc), classify(auc_full, auc_prior, resid_above_q95)


# One focal regulator x one label family
def core(system, tf, kind, use, V, prior, M1, T, gate_fn, gi, counts, Xb, Pb, names_E, Cov_b, cov_f, meta_row):
    cols = np.flatnonzero(M1)
    vec, dec = {}, {}
    for m in use:
        dd = make_dec(V[m], prior[m], Xb[m], Pb[m], M1, "z")
        assert dd["keep"].all(), "core: the main run requires the prior finite on M1"
        S, P, Rm = D.zrow(V[m], M1), np.where(M1, np.asarray(prior[m], float), np.nan), np.full(len(M1), np.nan)
        Rm[cols] = dd["R"]
        vec[m] = dict(full=np.where(M1, np.asarray(V[m], float), np.nan), prior=P, resid=Rm)
        dec[m] = dd
    rows_out, npz = [], {"names": np.asarray(names_E), **{f"load|{m}": dec[m]["load_b"] for m in use}}
    for tname, (y, rows) in T.items():
        y, rows = np.asarray(y, bool), np.asarray(rows, bool)
        yy = y[rows]
        gate = bool(gate_fn(y, rows))
        for m in use:
            dd = dec[m]
            base = dict(system=system, tf=tf, kind=kind, target=tname, method=m, n_M1=int(M1.sum()), n=int(rows.sum()), n_pos=int(yy.sum()), n_neg=int((~yy).sum()), gate=gate,
                        n_decoy_pool=len(names_E), loading_focal=dd["load_f"], sd_prior_on_M1=dd["sd_P"], rho_S_P=dd["rho_SP"], **meta_row)
            if not (yy.any() and (~yy).any()):
                rows_out.append(dict(base, status="TOO_FEW"))
                continue
            pts, bss = {}, {}
            for c in COMPS_:
                v = vec[m][c]
                if not np.isfinite(v[rows]).all():
                    raise RuntimeError(f"core: non-finite {c} inside the universe {system}/{tf}/{m}/{tname}")
                pts[c], bss[c] = CT.eval_rows(v, y, rows, gi, counts)
            r = dict(base, status="OK" if gate else "DESCRIPTIVE_ONLY", PREV=pts["full"]["PREV"])
            for c in COMPS_:
                for k, nm in (("AUROC", "AUROC"), ("AUPRC", "AP"), ("APLIFT", "APL")):
                    lo, hi, _ = pct(bss[c][k])
                    r[f"{nm}_{c}"], r[f"{nm}_{c}_lo"], r[f"{nm}_{c}_hi"] = pts[c][k], lo, hi
            for k, nm in (("AUROC", "AUROC"), ("APLIFT", "APL")):                           # transparent differences (paired draws)
                d = bss["full"][k] - bss["prior"][k]
                lo, hi, _ = pct(d)
                r[f"d_{nm}_fp"], r[f"d_{nm}_fp_lo"], r[f"d_{nm}_fp_hi"], r[f"p_{nm}_fp"] = pts["full"][k] - pts["prior"][k], lo, hi, D.boot_p(d)
                v0 = 0.5 if k == "AUROC" else 1.0
                d = bss["resid"][k] - v0
                lo, hi, _ = pct(d)
                r[f"r_{nm}"], r[f"r_{nm}_lo"], r[f"r_{nm}_hi"], r[f"p_{nm}_r"] = pts["resid"][k] - v0, lo, hi, D.boot_p(d)
            de, au, z_all = dec_eval(dd, y, rows, cols, Cov_b, cov_f)
            for c in COMPS_:                                                              # the fast AUROC used for the decoys must equal the tie-grouped point AUROC
                assert abs(de[f"AUROC_{c}"] - pts[c]["AUROC"]) < 1e-9, (c, de[f"AUROC_{c}"], pts[c]["AUROC"])
            r.update({k: v for k, v in de.items() if not k.startswith("AUROC_")})
            npz[f"dec|{m}|{tname}"], npz[f"zres|{m}|{tname}"] = au, z_all.astype(np.float32)
            r["class"], r["class_wb"] = class_pair(r["AUROC_full"], r["AUROC_prior"], r.get("p_cal"), r.get("resid_above_q95"))
            rows_out.append(r)
    return rows_out, npz


N_SEEDS, BANK_SEED = 20, 20260929
VARIANTS = [f"sub{s:02d}" for s in range(N_SEEDS)] + ["rank"]
_RZ = {}


# rank-normal row scores (nb, nT7) of the bank matrix of method m on the frozen T7
def rank_bank(bank, m):
    from scipy.stats import norm, rankdata
    k = (id(bank), m)
    if k not in _RZ:
        X = np.array(bank.X[m][:, bank.T], float)
        sc = bank.tcol[bank.bank_cols]
        X[np.flatnonzero(sc >= 0), sc[sc >= 0]] = np.nan
        Z = np.full(X.shape, np.nan)
        for i in range(len(X)):
            f = np.isfinite(X[i])
            if f.sum() >= 3 and np.ptp(X[i][f]) > 0:
                Z[i, f] = norm.ppf((rankdata(X[i][f]) - 0.5) / f.sum())
        _RZ[k] = Z
    return _RZ[k]


# leave-out priors on T7 from the bank rows H (bool over bank rows; the focal is not in H): focal prior (nT7,) and the decoy priors of rows E_sel
def prior_from(Zm, H, E_sel, focal_only=True):
    ZH = Zm[H]
    f = np.isfinite(ZH)
    S, C = np.where(f, ZH, 0.0).sum(0), f.sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        pf = np.where(C >= 2, S / np.maximum(C, 1), np.nan)
        Zs = Zm[E_sel]
        fs = np.isfinite(Zs)
        c_ = C[None, :] - fs
        pd_ = np.where(c_ >= 2, (S[None, :] - np.where(fs, Zs, 0.0)) / np.maximum(c_, 1), np.nan)
    return pf, pd_


# Robustness checks
# Returns (rows, npz arrays)
def robust(system, tf, kind, use, V, prior, M1, T, gate_fn, bank, r, tc, E, Xb, Pb, names_E, Cov_b, cov_f, meta_row):
    cols = np.flatnonzero(M1)
    nb, tcm = len(bank.members), tc[cols]
    rows_out, zs = [], {}
    for m in use:                                                                   # exact-reproduction check of the prior construction used below
        H = np.ones(nb, bool)
        H[r] = False
        pf, _ = prior_from(bank.P[m]["Z"], H, E[:0])
        assert np.allclose(pf[tcm], np.asarray(prior[m], float)[cols], atol=1e-9, equal_nan=True), f"robust: prior reconstruction differs from the main prior ({m})"
    for vi, v in enumerate(VARIANTS):
        if v == "rank":
            H = np.ones(nb, bool)
            H[r] = False
            scale, Zsrc = "rank", {m: rank_bank(bank, m) for m in use}
        else:
            rg = np.random.default_rng([BANK_SEED, int(v[3:])])
            H = np.zeros(nb, bool)
            H[rg.permutation(nb)[:nb // 2]] = True
            H[r] = False
            scale, Zsrc = "z", {m: bank.P[m]["Z"] for m in use}
        sel = np.flatnonzero(H[E])
        Vv, Pv, dds = {}, {}, {}
        for m in use:
            pf, pdv = prior_from(Zsrc[m], H, E[sel])
            pv = np.full(len(M1), np.nan)
            pv[cols] = pf[tcm]
            dds[m] = make_dec(V[m], pv, np.asarray(Xb[m])[sel], pdv[:, tcm], M1, scale)
        for tname, (y, rows) in T.items():
            y, rows = np.asarray(y, bool), np.asarray(rows, bool)
            if not (y[rows].any() and (~y[rows]).any()):
                continue
            gate = bool(gate_fn(y, rows))
            for m in use:
                de, au, z_all = dec_eval(dds[m], y, rows, cols, np.asarray(Cov_b)[sel], cov_f)
                zf = np.full(len(names_E), np.nan, np.float32)
                zf[sel] = z_all
                zs.setdefault(f"robz|{m}|{tname}", np.full((len(VARIANTS), len(names_E)), np.nan, np.float32))[vi] = zf
                cl, cl_wb = class_pair(de["AUROC_full"], de["AUROC_prior"], de.get("p_cal"), de.get("resid_above_q95"))
                rows_out.append(dict(system=system, tf=tf, kind=kind, variant=v, target=tname, method=m, gate=gate, n_decoy=de["N_decoy"], n_keep=int(dds[m]["keep"].sum()),
                                     AUROC_full=de["AUROC_full"], AUROC_prior=de["AUROC_prior"], AUROC_resid=de["AUROC_resid"], p_cal=de.get("p_cal", np.nan), z_cal=de.get("z_cal", np.nan),
                                     p_decoy=de.get("p_decoy", np.nan), loading_focal=dds[m]["load_f"], class_tf=cl, class_wb_tf=cl_wb))
    zs["variants"] = np.asarray(VARIANTS)
    return rows_out, zs


# bank-side context shared by the lattice and regulon paths
# genes -> bank axis / T7 columns; leave-focal-out prior per method
# M1 = M0 & T7 & every prior finite (attribution_v3's construction)
def masks_and_priors(bank, genes, V, use, r, M0):
    ai = np.array([bank.ai.get(g, -1) for g in genes])
    has = ai >= 0
    ai0 = np.where(has, ai, 0)
    tc = np.where(has, bank.tcol[ai0], -1)
    prior = {m: np.where(tc >= 0, bank.prior(m, r)[ai0], np.nan) for m in use}
    M1 = np.asarray(M0, bool) & (tc >= 0) & np.all([np.isfinite(prior[m]) for m in use], 0)
    return ai, has, ai0, tc, prior, M1


# gb_attribution's eligible bank (features complete, rows finite on M1 for every method, not the focal, not a perturbed-panel TF)
def decoy_pool(bank, use, M1, ai, tc, tf, perturbed):
    F = bank.feats[~bank.feats.index.str.upper().duplicated(keep=False)]
    F = F.set_axis(F.index.str.upper())
    Fraw = F.reindex(bank.mu)[A.FEATS].to_numpy(float)
    fcomp = np.isfinite(Fraw).all(1)
    cols = np.flatnonzero(M1)
    rows_ok = np.all([np.isfinite(bank.X[m][:, ai[cols]]).all(1) & bank.row_ok[m] for m in use], 0)
    is_r = bank.mu == tf.upper()
    is_pert = np.isin(bank.mu, list(perturbed)) if perturbed else np.zeros(len(bank.mu), bool)
    elig = fcomp & rows_ok & ~is_r & ~is_pert
    E = np.flatnonzero(elig)
    Xb = {m: bank.X[m][np.ix_(E, ai[cols])] for m in use}
    Pb = {m: bank.prior_rows(m, E)[:, tc[cols]] for m in use}
    cov_f = F.loc[tf.upper(), A.FEATS].to_numpy(float) if tf.upper() in F.index and np.isfinite(F.loc[tf.upper(), A.FEATS].to_numpy(float)).all() else None
    notes = dict(n_bank=len(bank.members), focal=int(is_r.sum()), perturbed_panel=int(is_pert.sum()), missing_features=int((~fcomp).sum()),
                 row_not_finite_on_M1_any_method=int((fcomp & ~rows_ok & ~is_r & ~is_pert).sum()), eligible=int(elig.sum()))
    return E, Xb, Pb, [bank.members[i] for i in E], Fraw[E], cov_f, notes


# Include perturbations for the bone-marrow base system.
def perturbed_of(system, L, LE):
    try:
        return A.perturbed_set(system, L, LE), "A.perturbed_set"
    except SystemExit:
        if system == "BONEMARROW":
            return {t.upper() for t in LE.PANEL_TFS} | {"GATA1", "GATA2", "RUNX1", "TAL1", "FLI1"}, "BONEMARROW: LE.PANEL_TFS + GATA1/GATA2/RUNX1/TAL1/FLI1 (audit extension)"
        raise


_V3 = {}


def v3_table():
    if "t" not in _V3:
        _V3["t"] = pd.read_csv(f"{V3}/per_tf_method.tsv", sep="\t", dtype={"tf": str})
    return _V3["t"]


# assert parity with attribution_v3 (same n_M1, cognate and prior AUROC for every method x target present there, tol 1e-9)
def v3_parity(system, tf, rows_out, M1):
    d = v3_table()
    d = d[(d.system == system) & (d.tf == tf) & (d.metric == "AUROC") & d.status.isin(["OK", "DESCRIPTIVE_ONLY"])]
    if not len(d):
        return dict(status="NO_V3_REFERENCE")
    if not (d.n_M1 == int(M1.sum())).all():
        raise SystemExit(f"REFUSING_PARITY_MASK {system}/{tf}: n_M1 now {int(M1.sum())} vs attribution_v3 {sorted(d.n_M1.unique())}")
    mine = {(r["method"], r["target"]): r for r in rows_out if r.get("status") in ("OK", "DESCRIPTIVE_ONLY")}
    worst, n = 0.0, 0
    for _, q in d.iterrows():
        r = mine.get((q.method, q.target))
        if r is None:
            continue
        for a, b in ((r["AUROC_full"], q.cognate), (r["AUROC_prior"], q.prior)):
            worst = max(worst, abs(a - b))
        n += 1
    if worst > 1e-9:
        raise SystemExit(f"REFUSING_PARITY_AUROC {system}/{tf}: full/prior AUROC differ from attribution_v3 by {worst:.3e}")
    return dict(status="PASS", n_cells=n, worst_absdiff=worst)


# nonlinear simple-feature control (response only)
def control(system, tf, ep, L, use, V, prior, M1, gi, counts):
    refs = list(L.REFS)
    if not all(n in V for n in refs):
        return [dict(system=system, tf=tf, status="NOT_RUN_REF_MISSING")]
    Xr = np.column_stack([np.asarray(V[n], float) for n in refs])
    mr = M1 & np.isfinite(Xr).all(1)
    folds = np.asarray(ep["fold"])
    y, rows = L.targets(ep, mr)["resp"]
    gate = bool(L.gate_of(ep, y, rows, mr))
    Xrn = np.column_stack([rank_normal(Xr[:, j], mr) for j in range(Xr.shape[1])])
    vecs = []                                                       # (vtype, arm, method, vec, extra)
    for m in use:
        zt = D.zrow(prior[m], mr)
        fh, rh, sh = hgb_cross_fit(zt, Xr, folds, mr)
        fo, ro, so = D.fold_ols(zt, Xrn, folds, mr)
        vecs += [("PRIOR", "none", m, np.where(mr, prior[m], np.nan), {}), ("FIT", "hgb", m, fh, sh), ("RESID", "hgb", m, rh, sh), ("FIT", "ols_ranknormal", m, fo, so),
                 ("RESID", "ols_ranknormal", m, ro, so)]
    for c in ("COMP_Z", "COMP_RANK"):
        if c in V and np.isfinite(np.asarray(V[c], float)[mr]).all():
            vecs.append(("COMP", "none", c, np.where(mr, np.asarray(V[c], float), np.nan), {}))
    for n in refs:
        vecs.append(("REF", "none", n, np.where(mr, Xr[:, refs.index(n)], np.nan), {}))
    out = []
    for vt, arm, name, v, ex in vecs:
        e = D.eval_vec(v, y, rows, gi, counts, null={"AUROC": 0.5, "APLIFT": 1.0})
        base = dict(system=system, block=ep["block"], tf=tf, vtype=vt, arm=arm, vector=name, target="resp", gate=gate, n_M1=int(M1.sum()), n_refmask=int(mr.sum()), **ex)
        out.append(dict(base, status="TOO_FEW") if e is None else dict(base, status="OK" if gate else "DESCRIPTIVE_ONLY", **e))
    return out


# lattice systems (KO response / ChIP occupancy / four-class contrasts)
def lattice_tf(system, tf, blk, B, L, LE):
    cfg = L.SYSTEMS[system]
    bank = A.load_bank(cfg["axis"])
    ep = L.build_endpoint(system, tf, blk, False)
    V, genes = ep["V"], np.asarray(ep["genes"]).astype(str)
    use = [m for m in A.METHODS if m in V and m in bank.X]
    if GB not in use:
        raise L.Blocked(f"GB_ROW_OR_BANK_MISSING {system}/{tf}")
    r = bank.row_of(tf)
    if r is None:
        raise L.Blocked(f"FOCAL_NOT_IN_BANK {system}/{tf}")
    M0 = np.asarray(ep["U"], bool) & np.all([np.isfinite(V[m]) for m in use], 0)
    ai, has, ai0, tc, prior, M1 = masks_and_priors(bank, genes, V, use, r, M0)
    if "prior GB" in V:                                                             # exact parity with the lattice prior
        f = M1 & np.isfinite(V["prior GB"])
        dgb = float(np.abs(prior[GB][f] - V["prior GB"][f]).max()) if f.any() else np.nan
        if not dgb <= 1e-9:
            raise SystemExit(f"REFUSING_PARITY {system}/{tf}: GB leave-focal-out prior differs from lattice `prior GB` by {dgb}")
    else:
        dgb = np.nan
    pert, pnote = perturbed_of(system, L, LE)
    E, Xb, Pb, names_E, Cov_b, cov_f, notes = decoy_pool(bank, use, M1, ai, tc, tf, pert)
    Tt = L.targets(ep, M1)
    T = {k: Tt[k] for k in TARGETS if k in Tt}
    counts = blk.counts[:B]
    meta_row = dict(block=ep["block"], focal_in_bank=True)
    gate_fn = lambda y, rows: L.gate_of(ep, y, rows, M1)
    rows_out, npz = core(system, tf, "lattice", use, V, prior, M1, T, gate_fn, ep["gi"], counts, Xb, Pb, names_E, Cov_b, cov_f, meta_row)
    rob_rows, rz = robust(system, tf, "lattice", use, V, prior, M1, T, gate_fn, bank, r, tc, E, Xb, Pb, names_E, Cov_b, cov_f, meta_row)
    npz.update(rz)
    par = v3_parity(system, tf, rows_out, M1)
    ctrl = control(system, tf, ep, L, use, V, prior, M1, ep["gi"], counts)
    coverage = {m: (m in use) for m in A.METHODS}
    meta = dict(system=system, tf=tf, methods_used=use, methods_missing=[m for m in A.METHODS if m not in use], n_M1=int(M1.sum()), n_M0=int(M0.sum()), parity_v3=par,
                parity_prior_GB_vs_lattice_absdiff=dgb, pool=notes, perturbed_source=pnote, cov_f_missing=cov_f is None, coverage=coverage, files={k: v for k, v in ep["files"].items()},
                gate={t: bool(L.gate_of(ep, *T[t], M1)) for t in T})
    return rows_out, ctrl, npz, meta, rob_rows


def v3_tfs(system):
    d = v3_table()
    d = d[(d.system == system) & d.status.isin(["OK", "DESCRIPTIVE_ONLY", "TOO_FEW"])]
    return sorted(d.tf.unique())


def tf_list_audit(system, L, LE):
    t = v3_tfs(system)
    return t if t else list(A.tf_list(system, L, LE))


def cmd_run(a):
    import gb_lattice_extra as LE
    import gb_layer_lattice as L
    LE.register()
    os.makedirs(a.out, exist_ok=True)
    for system in [s.strip() for s in a.systems.split(",") if s.strip()]:
        if system not in A.SYSTEMS:
            raise SystemExit(f"REFUSING: {system} not in {A.SYSTEMS}")
        cfg = L.SYSTEMS[system]
        allowed = tf_list_audit(system, L, LE)
        tfs = [t.strip() for t in a.tfs.split(",") if t.strip()] or allowed
        bad = [t for t in tfs if t not in set(A.tf_list(system, L, LE)) | set(allowed)]
        if bad:
            raise SystemExit(f"REFUSING: --tfs {bad} not declared for {system}")
        if a.nslice > 1:
            tfs = [t for i, t in enumerate(tfs) if i % a.nslice == a.slice]
        seed = int(np.random.SeedSequence([L.BOOT_SEED, L.BLOCKS.index(cfg["block"])]).generate_state(1)[0])
        blk = L.BlockAxis(L.axis_of(cfg["axis"]), a.B, seed, None, "gene")
        wide, ctrl, rob, metas, npz_all = [], [], [], {}, {}
        for tf in tfs:
            t0 = time.time()
            try:
                w, c, nz, mt, rb = lattice_tf(system, tf, blk, a.B, L, LE)
            except (L.Blocked, SystemExit) as e:
                metas[tf] = dict(status=f"NOT_RUN: {e}")
                print(f"[audit] {system}/{tf}: NOT_RUN {e}", flush=True)
                continue
            wide += w
            ctrl += c
            rob += rb
            metas[tf] = mt
            npz_all.update({f"{tf}::{k}": v for k, v in nz.items()})
            print(f"[audit] {system}/{tf}: M1={mt['n_M1']} pool={mt['pool']['eligible']} parity={mt['parity_v3']['status']} methods={len(mt['methods_used'])} {time.time()-t0:.0f}s", flush=True)
        tag = ("all" if not a.tfs else "_".join(tfs)) + (f"_s{a.slice}of{a.nslice}" if a.nslice > 1 else "")
        if wide:
            pd.DataFrame(wide).to_csv(f"{a.out}/lat_{system}_{tag}.wide.tsv", sep="\t", index=False)
            pd.DataFrame(ctrl).to_csv(f"{a.out}/lat_{system}_{tag}.ctrl.tsv", sep="\t", index=False)
            pd.DataFrame(rob).to_csv(f"{a.out}/lat_{system}_{tag}.rob.tsv", sep="\t", index=False)
            np.savez_compressed(f"{a.out}/lat_{system}_{tag}.dec.npz", **npz_all)
        L.atomic_json(f"{a.out}/lat_{system}_{tag}.meta.json", dict(system=system, kind="lattice", B=a.B, seed=seed, script_sha256=L.sha_file(__file__), attribution_sha256=L.sha_file(A.__file__),
                                                                   decompose_sha256=L.sha_file(D.__file__), module_shas=L.module_shas(), tfs=metas))


# regulon endpoints (gb_canon_table families F1, per-regulator regulon labels)
def regulon_system(system, B, max_tfs, sl, nsl, out):
    import gb_layer_lattice as L
    H = CT.heavy()
    union, gen = CT.load_system(system, H)
    bank = A.load_bank(system)
    gidx = {g: i for i, g in enumerate(union)}
    counts = CT.make_counts(len(union), B, [CT.BOOT_SEED, CT.DEFAULT_SYSTEMS.index(system)])[:B]
    wide, rob, metas, npz_all = [], [], {}, {}
    k, seen = 0, set()
    for rec in gen:
        if rec["family"] != "F1" or rec["source"] in seen:
            continue
        tf = rec["source"]
        seen.add(tf)
        k += 1
        if (nsl > 1 and (k - 1) % nsl != sl) or (max_tfs and k > max_tfs):
            continue
        t0 = time.time()
        V, genes, ev, y = rec["V"], np.asarray(rec["obs_genes"]).astype(str), np.asarray(rec["ev"], bool), np.asarray(rec["y"], bool)
        r = bank.row_of(tf)
        use = [m for m in A.METHODS if m in V and m in bank.X]
        if r is None or GB not in use:
            metas[tf] = dict(status=f"NOT_RUN: {'FOCAL_NOT_IN_BANK' if r is None else 'GB_ROW_MISSING'}")
            print(f"[audit-reg] {system}/{tf}: {metas[tf]['status']}", flush=True)
            continue
        M0 = ev & np.all([np.isfinite(V[m]) for m in use], 0)
        ai, has, ai0, tc, prior, M1 = masks_and_priors(bank, genes, V, use, r, M0)
        yy = y[M1]
        if yy.sum() < MIN_CLASS or (~yy).sum() < MIN_CLASS:
            metas[tf] = dict(status=f"NOT_RUN: regulon on M1 has {int(yy.sum())} pos / {int((~yy).sum())} neg (< {MIN_CLASS})", n_M1=int(M1.sum()))
            print(f"[audit-reg] {system}/{tf}: {metas[tf]['status']}", flush=True)
            continue
        par = {}
        for m in use:                                                                # canon cognate row vs bank row (recorded, not asserted)
            v = np.where(has, bank.X[m][r][ai0], np.nan)
            f = np.isfinite(v) & np.isfinite(V[m])
            par[m] = float(np.abs(v[f] - V[m][f]).max()) if f.any() else None
        E, Xb, Pb, names_E, Cov_b, cov_f, notes = decoy_pool(bank, use, M1, ai, tc, tf, set())
        gi = np.fromiter((gidx[g] for g in genes), int, len(genes))
        T = {"regulon": (y, M1)}
        gate_fn = lambda yy_, rows: (yy_[rows].sum() >= MIN_CLASS and (~yy_[rows]).sum() >= MIN_CLASS)
        mrow = dict(block=system, focal_in_bank=True)
        w, nz = core(system, tf, "regulon", use, V, prior, M1, T, gate_fn, gi, counts, Xb, Pb, names_E, Cov_b, cov_f, mrow)
        rb, rz = robust(system, tf, "regulon", use, V, prior, M1, T, gate_fn, bank, r, tc, E, Xb, Pb, names_E, Cov_b, cov_f, mrow)
        nz.update(rz)
        rob += rb
        wide += w
        npz_all.update({f"{tf}::{k_}": v_ for k_, v_ in nz.items()})
        metas[tf] = dict(methods_used=use, methods_missing=[m for m in A.METHODS if m not in use], n_M1=int(M1.sum()), n_pos=int(yy.sum()), pool=notes, cov_f_missing=cov_f is None,
                         canon_vs_bank_row_absdiff=par, endpoint=rec["endpoint"])
        print(f"[audit-reg] {system}/{tf}: M1={int(M1.sum())} pos={int(yy.sum())} pool={notes['eligible']} methods={len(use)} {time.time()-t0:.0f}s", flush=True)
    tag = "all" + (f"_max{max_tfs}" if max_tfs else "") + (f"_s{sl}of{nsl}" if nsl > 1 else "")
    if wide:
        pd.DataFrame(wide).to_csv(f"{out}/reg_{system}_{tag}.wide.tsv", sep="\t", index=False)
        pd.DataFrame(rob).to_csv(f"{out}/reg_{system}_{tag}.rob.tsv", sep="\t", index=False)
        np.savez_compressed(f"{out}/reg_{system}_{tag}.dec.npz", **npz_all)
    L.atomic_json(f"{out}/reg_{system}_{tag}.meta.json", dict(system=system, kind="regulon", B=B, script_sha256=L.sha_file(__file__), attribution_sha256=L.sha_file(A.__file__),
                                                              decompose_sha256=L.sha_file(D.__file__), module_shas=L.module_shas(), tfs=metas))


def cmd_regulon(a):
    os.makedirs(a.out, exist_ok=True)
    for system in [s.strip() for s in a.systems.split(",") if s.strip()]:
        if system not in REGULON_SYSTEMS:
            raise SystemExit(f"REFUSING: {system} not in {REGULON_SYSTEMS}")
        regulon_system(system, a.B, a.max_tfs, a.slice, a.nslice, a.out)


# merge
def _macro_vals(g, col):
    return g[col].to_numpy(float)


# system x kind x target x method: TF-macro (gated TFs) of every component + transparent differences + decoy bracket + class. decs[(system, kind)]
def summarize(wide, decs):
    d = wide[wide.status == "OK"]
    out = []
    for (s, kd, t, m), g in d.groupby(["system", "kind", "target", "method"], sort=False):
        n = len(g)
        idx = np.random.default_rng(zlib.crc32(f"{s}{kd}".encode())).integers(0, n, (N_TFBOOT, n)) if n >= 2 else None
        r = dict(system=s, kind=kd, target=t, method=m, n_tf=n, tfs=",".join(sorted(g.tf)) if n <= 12 else f"{n} TFs")
        for c in COMPS_:
            for nm in ("AUROC", "AP", "APL"):
                v = _macro_vals(g, f"{nm}_{c}")
                r[f"{nm}_{c}"] = float(v.mean())
                if n == 1:
                    r[f"{nm}_{c}_lo"], r[f"{nm}_{c}_hi"] = float(g[f"{nm}_{c}_lo"].iloc[0]), float(g[f"{nm}_{c}_hi"].iloc[0])
                else:
                    _, r[f"{nm}_{c}_lo"], r[f"{nm}_{c}_hi"] = D.tf_boot(v, idx)
        for nm, cols in (("AUROC", ("d_AUROC_fp", "r_AUROC")), ("APL", ("d_APL_fp", "r_APL"))):
            for col in cols:
                v = _macro_vals(g, col)
                r[col] = float(v.mean())
                if n == 1:
                    r[f"{col}_lo"], r[f"{col}_hi"] = float(g[f"{col}_lo"].iloc[0]), float(g[f"{col}_hi"].iloc[0])
                    r[f"p_{col}"], r[f"p_{col}_method"] = float(g[f"p_{nm}_fp" if col.startswith("d_") else f"p_{nm}_r"].iloc[0]), "gene_bootstrap_conditional"
                else:
                    _, r[f"{col}_lo"], r[f"{col}_hi"] = D.tf_boot(v, idx)
                    r[f"p_{col}"], r[f"p_{col}_method"] = signflip(v, seed=zlib.crc32(f"{s}{t}{m}{col}".encode()))
        r["n_tf_full_gt_prior"] = int((g.d_AUROC_fp > 0).sum())
        r["n_tf_resid_gt_half"] = int((g.r_AUROC > 0).sum())
        r["PREV_mean"] = float(g.PREV.mean())
        r["N_decoy_min"] = int(g.N_decoy.min()) if "N_decoy" in g else np.nan
        r["median_p_decoy_tf"] = float(g.p_decoy.median()) if "p_decoy" in g else np.nan
        r["median_p_cal_tf"] = float(g.p_cal.median()) if "p_cal" in g and g.p_cal.notna().any() else np.nan
        r["n_tf_resid_above_q95"] = int(g.resid_above_q95.fillna(False).astype(bool).sum()) if "resid_above_q95" in g else 0
        if n == 1:
            r["p_decoy_aligned"] = r["p_decoy_indep"] = float(g.p_decoy.iloc[0])
            r["n_decoy_common"] = int(g.N_decoy.iloc[0])
        else:
            dl = []
            for tf in g.tf:
                z = decs.get((s, kd), {}).get(tf)
                if z is None or f"dec|{m}|{t}" not in z:
                    dl = None
                    break
                au = z[f"dec|{m}|{t}"]
                dl.append({nme: v for nme, v in zip(z["names"], au)})
            if dl:
                pa, pi, nc = macro_decoy_p(g.AUROC_resid.to_numpy(float), dl, seed=zlib.crc32(f"{s}{t}{m}".encode()))
                r["p_decoy_aligned"], r["p_decoy_indep"], r["n_decoy_common"] = pa, pi, nc
        p_cons = np.nanmax([r.get("p_decoy_aligned", np.nan), r.get("p_decoy_indep", np.nan)]) if np.isfinite([r.get("p_decoy_aligned", np.nan), r.get("p_decoy_indep", np.nan)]).any() else np.nan
        r["p_decoy_conservative"] = p_cons                                                              # descriptive whole-bank (loading-confounded)
        above = None if not np.isfinite(p_cons) else bool(p_cons <= 0.05)
        r["resid_above_decoy_q95"] = above
        r["class_wb"] = classify(r["AUROC_full"], r["AUROC_prior"], above)                              # sensitivity: whole-bank decoy 95th percentile
        # PRIMARY calibration: loading + heteroscedasticity-adjusted standardized residual (gb_attribution.bank_calibration); multi-TF = MACRO test of the mean adjusted z
        zs = g.z_cal.to_numpy(float) if "z_cal" in g else np.array([np.nan])
        r["z_cal_macro"] = float(np.mean(zs)) if np.isfinite(zs).all() else np.nan
        r["signflip_floor"] = float(2 / 2 ** n) if 2 <= n <= 20 else np.nan
        if n == 1:
            p_al = p_in = float(g.p_cal.iloc[0]) if "p_cal" in g else np.nan
            r["n_adj_common"] = int(g.N_cal.iloc[0]) if "N_cal" in g and np.isfinite(g.N_cal.iloc[0]) else 0
        else:
            lst = []
            for tf in g.tf:
                z = decs.get((s, kd), {}).get(tf)
                if z is None or f"zres|{m}|{t}" not in z:
                    lst = None
                    break
                lst.append((z["names"], z[f"zres|{m}|{t}"]))
            p_al, p_in, r["n_adj_common"] = macro_z_p(zs, lst, seed=zlib.crc32(f"{s}{t}{m}z".encode())) if lst else (np.nan, np.nan, 0)
        p_adj = np.nanmax([p_al, p_in]) if np.isfinite([p_al, p_in]).any() else np.nan
        r["p_adj_aligned"], r["p_adj_indep"], r["p_adj_conservative"] = p_al, p_in, p_adj
        f_, pr_ = r["AUROC_full"], r["AUROC_prior"]
        r["class"] = classify(f_, pr_, None if not np.isfinite(p_adj) else bool(p_adj <= 0.05))          # primary (conservative end of the bracket)
        r["class_aligned"] = classify(f_, pr_, None if not np.isfinite(p_al) else bool(p_al <= 0.05))
        r["class_indep"] = classify(f_, pr_, None if not np.isfinite(p_in) else bool(p_in <= 0.05))
        out.append(r)
    return pd.DataFrame(out)


# Macro test of the adjusted calibration (gb_attribution.substrate_p semantics, array form)
def macro_z_p(focal_z, lst, n_mc=20000, seed=0):
    focal_z = np.asarray(focal_z, float)
    if not lst or not np.isfinite(focal_z).all():
        return np.nan, np.nan, 0
    ser = [pd.Series(np.asarray(z, float), index=np.asarray(nm).astype(str)) for nm, z in lst]
    common = ser[0].index
    for s_ in ser[1:]:
        common = common.intersection(s_.index)
    M = np.array([s_.reindex(common).to_numpy() for s_ in ser])
    M = M[:, np.isfinite(M).all(0)]
    if M.shape[1] == 0:
        return np.nan, np.nan, 0
    obs = float(focal_z.mean())
    na = M.mean(0)
    rg = np.random.default_rng(seed)
    ni = np.array([M[t][rg.integers(0, M.shape[1], n_mc)] for t in range(M.shape[0])]).mean(0)
    return float((1 + (na >= obs).sum()) / (len(na) + 1)), float((1 + (ni >= obs).sum()) / (n_mc + 1)), int(M.shape[1])


# Class stability per cell: for every system x kind x target x method, the class under each of the 20 bank-subsample seeds and under rank scale
def robust_summary(S, rob, decs):
    out = []
    rob = rob[rob.gate.astype(bool)]
    for _, c in S.iterrows():
        s, kd, t, m = c.system, c.kind, c.target, c.method
        g = rob[(rob.system == s) & (rob.kind == kd) & (rob.target == t) & (rob.method == m)]
        tfs = [x for x in (c.tfs.split(",") if c.n_tf <= 12 else sorted(g.tf.unique()))]
        gm = g[g.tf.isin(tfs)] if c.n_tf <= 12 else g
        tfs = sorted(gm.tf.unique())
        if len(tfs) != c.n_tf:
            continue
        cls = {}
        for vi, v in enumerate(VARIANTS):
            gv = gm[gm.variant == v].set_index("tf").loc[tfs]
            zs = gv.z_cal.to_numpy(float)
            if len(tfs) == 1:
                p = float(gv.p_cal.iloc[0])
            else:
                lst = []
                for tf in tfs:
                    zz = decs.get((s, kd), {}).get(tf, {})
                    if f"robz|{m}|{t}" not in zz:
                        lst = None
                        break
                    lst.append((zz["names"], zz[f"robz|{m}|{t}"][vi]))
                pa, pi, _ = macro_z_p(zs, lst, n_mc=4000, seed=zlib.crc32(f"{s}{t}{m}{v}".encode())) if lst else (np.nan, np.nan, 0)
                p = np.nanmax([pa, pi]) if np.isfinite([pa, pi]).any() else np.nan
            cls[v] = classify(float(gv.AUROC_full.mean()), float(gv.AUROC_prior.mean()), None if not np.isfinite(p) else bool(p <= 0.05))
        sub = [cls[v] for v in VARIANTS if v.startswith("sub")]
        r = dict(system=s, kind=kd, target=t, method=m, n_tf=len(tfs), class_main=c["class"], stability_sub=float(np.mean([x == c["class"] for x in sub])), class_rank=cls["rank"],
                 rank_same=bool(cls["rank"] == c["class"]), **{f"n_sub_{k}": sum(x == k for x in sub) for k in ("target-common-dominated", "mixed", "regulator-specific", "NA")})
        out.append(r)
    return pd.DataFrame(out)


# per (system, kind, TF, method): the bank loading distribution
def loading_summary(wide, decs):
    rows = []
    for (s, kd), d in decs.items():
        for tf, z in d.items():
            for k in z:
                if not k.startswith("load|"):
                    continue
                m = k[5:]
                lb = np.asarray(z[k], float)
                lb = lb[np.isfinite(lb)]
                f = wide[(wide.system == s) & (wide.kind == kd) & (wide.tf == tf) & (wide.method == m)]
                lf = float(f.loading_focal.iloc[0]) if len(f) else np.nan
                rows.append(dict(system=s, kind=kd, tf=tf, method=m, n_decoy=len(lb), q05=np.percentile(lb, 5), q25=np.percentile(lb, 25), median=np.median(lb), q75=np.percentile(lb, 75),
                                 q95=np.percentile(lb, 95), focal_loading=lf, focal_percentile=float((lb < lf).mean()) if np.isfinite(lf) else np.nan))
    return pd.DataFrame(rows)


# Evaluate the KO prediction (a majority, at least 6 of 10 methods, is target-common-dominated on KO response per substrate) and its falsifiers
def claims(S):
    blk = {"GATA1": "ERYTHROID", "KLF1": "ERYTHROID", "KLF1_B": "ERYTHROID", "PANC": "PANC", "BM_CHIPATLAS": "BONEMARROW", "BONEMARROW": "BONEMARROW", "BM_GATA1_ACUTE": "BONEMARROW", "K562_EMP_ALLNULL": "K562"}
    rows = []
    for cal in ("class", "class_wb"):
        for s in sorted(S[S.kind == "lattice"].system.unique()):
            ko = S[(S.system == s) & (S.target == "resp") & (S.kind == "lattice")]
            oc = S[(S.system == s) & (S.target == "occ_prim") & (S.kind == "lattice")]
            rows.append(dict(calibration=cal, system=s, block=blk.get(s), n_methods_ko=len(ko), ko_TC=int((ko[cal] == "target-common-dominated").sum()), ko_MX=int((ko[cal] == "mixed").sum()),
                             ko_RS=int((ko[cal] == "regulator-specific").sum()), prediction_met=bool((ko[cal] == "target-common-dominated").sum() >= 6),
                             falsifier_a_majority_RS=bool((ko[cal] == "regulator-specific").sum() >= 6), n_methods_occ=len(oc), occ_TC=int((oc[cal] == "target-common-dominated").sum()),
                             occ_MX=int((oc[cal] == "mixed").sum()), occ_RS=int((oc[cal] == "regulator-specific").sum()),
                             occ_n_methods_full_gt_prior_01=int((oc.AUROC_full - oc.AUROC_prior > EPS_CLASS).sum()), falsifier_b_prior_reproduces_all=bool(len(oc) > 0 and (oc.AUROC_prior >= oc.AUROC_full - EPS_CLASS).all())))
    C = pd.DataFrame(rows)
    C["falsifier_a_substrates"] = ""
    for cal in ("class", "class_wb"):
        m = C.calibration == cal
        nsub = C[m & C.falsifier_a_majority_RS].block.nunique()
        C.loc[m, "falsifier_a_substrates"] = f"{nsub} substrate(s) with a system where a majority of methods is RS (falsifier a needs >= 2)"
    return C


# F = mean(resp, A>B, C>D AUROC - .5), O = mean(occ, A>C, B>D AUROC - .5) per TF over the available gated targets of that TF
def phenotype(wide):
    d = wide[(wide.status == "OK") & (wide.kind == "lattice")]
    rows = []
    for (s, tf, m), g in d.groupby(["system", "tf", "method"], sort=False):
        r = dict(system=s, tf=tf, method=m)
        for c in COMPS_:
            f = g[g.target.isin(F_TARGETS)][f"AUROC_{c}"] - 0.5
            o = g[g.target.isin(O_TARGETS)][f"AUROC_{c}"] - 0.5
            r[f"F_{c}"], r[f"O_{c}"], r["n_F"], r["n_O"] = (float(f.mean()) if len(f) else np.nan), (float(o.mean()) if len(o) else np.nan), len(f), len(o)
        rows.append(r)
    P = pd.DataFrame(rows)
    if not len(P):
        return P, P
    val = [f"{x}_{c}" for c in COMPS_ for x in "FO"]
    S = P.groupby(["system", "method"], sort=False)[val].mean().reset_index()
    S["n_tf"] = P.groupby(["system", "method"], sort=False).size().to_numpy()
    for c in COMPS_:
        S[f"D_{c}"] = S[f"F_{c}"] - S[f"O_{c}"]
    return P, S


# Per system x vector (vtype, arm, name): TF-macro response AUROC / AP-lift, test vs .5 / 1
def control_summary(ctrl):
    d = ctrl[ctrl.status == "OK"]
    out = []
    for (s, vt, arm, nm), g in d.groupby(["system", "vtype", "arm", "vector"], sort=False):
        n = len(g)
        idx = np.random.default_rng(zlib.crc32(s.encode())).integers(0, n, (N_TFBOOT, n)) if n >= 2 else None
        r = dict(system=s, vtype=vt, arm=arm, vector=nm, n_tf=n, tfs=",".join(sorted(g.tf)) if n <= 12 else f"{n} TFs", mean_R2_heldout=float(g.R2_heldout.mean()) if "R2_heldout" in g and g.R2_heldout.notna().any() else np.nan)
        for k, v0 in (("AUROC", 0.5), ("APLIFT", 1.0)):
            v = g[k].to_numpy(float)
            r[f"mean_{k}"] = float(v.mean())
            if n == 1:
                r[f"{k}_lo"], r[f"{k}_hi"], r[f"p_{k}_vs_null"] = float(g[f"{k}_lo"].iloc[0]), float(g[f"{k}_hi"].iloc[0]), float(g[f"p_{k}_null"].iloc[0])
            else:
                _, r[f"{k}_lo"], r[f"{k}_hi"] = D.tf_boot(v, idx)
                r[f"p_{k}_vs_null"], _ = signflip(v - v0, seed=zlib.crc32(f"{s}{nm}{k}".encode()))
        out.append(r)
    S = pd.DataFrame(out)
    con = []
    for s, g in d.groupby("system", sort=False):
        piv = g.pivot_table(index="tf", columns=["vtype", "arm", "vector"], values="AUROC")
        for arm in ("hgb", "ols_ranknormal"):
            for m in A.METHODS:
                if ("PRIOR", "none", m) in piv and ("FIT", arm, m) in piv:
                    dv = (piv[("PRIOR", "none", m)] - piv[("FIT", arm, m)]).to_numpy(float)
                    p, pm = signflip(dv, seed=zlib.crc32(f"{s}{arm}{m}".encode())) if len(dv) >= 2 else (np.nan, "n=1")
                    con.append(dict(system=s, arm=arm, method=m, contrast="PRIOR_minus_FIT(feature model)", n_tf=len(dv), mean_diff=float(np.mean(dv)), p_two=p, p_method=pm, n_tf_prior_gt_fit=int((dv > 0).sum())))
        refs = [c for c in piv.columns if c[0] == "REF"]
        if refs:
            best = max(refs, key=lambda c: piv[c].mean())
            con.append(dict(system=s, arm="none", method=f"HINDSIGHT best single REF = {best[2]}", contrast="best-in-hindsight (LABEL: chosen with truth; never inference)", n_tf=len(piv),
                            mean_diff=float(piv[best].mean()), p_two=np.nan, p_method="descriptive", n_tf_prior_gt_fit=np.nan))
    return S, pd.DataFrame(con)


def cmd_merge(a):
    W = [pd.read_csv(f, sep="\t", dtype={"tf": str}) for f in sorted(glob.glob(f"{a.out}/lat_*.wide.tsv")) + sorted(glob.glob(f"{a.out}/reg_*.wide.tsv"))]
    if not W:
        raise SystemExit("REFUSING: no part files")
    wide = pd.concat(W, ignore_index=True)
    dup = wide.duplicated(["system", "kind", "tf", "target", "method"])
    if dup.any():
        raise SystemExit(f"REFUSING_DUPLICATE_KEYS: {int(dup.sum())}")
    shas = set()
    for f in glob.glob(f"{a.out}/*.meta.json"):
        shas.add(json.load(open(f))["script_sha256"])
    if len(shas) != 1:
        raise SystemExit(f"REFUSING: parts come from {len(shas)} different script shas")
    decs = {}                                                       # system names may contain underscores (KLF1_B, ...): recover (system, kind) from the part's own wide table
    for f in sorted(glob.glob(f"{a.out}/lat_*.dec.npz")) + sorted(glob.glob(f"{a.out}/reg_*.dec.npz")):
        wf = f.replace(".dec.npz", ".wide.tsv")
        wp = pd.read_csv(wf, sep="\t", dtype={"tf": str}, usecols=["system", "kind", "tf"]).drop_duplicates()
        tf2sys = {(r.tf): (r.system, r.kind) for r in wp.itertuples()}
        with np.load(f, allow_pickle=True) as z:
            for k in z.files:
                tf, key = k.split("::", 1)
                decs.setdefault(tf2sys[tf], {}).setdefault(tf, {})[key] = z[k]
    S = summarize(wide, decs)
    P, PS = phenotype(wide)
    cs = [pd.read_csv(f, sep="\t", dtype={"tf": str}) for f in sorted(glob.glob(f"{a.out}/lat_*.ctrl.tsv"))]
    CS = CC = pd.DataFrame()
    if cs:
        ctrl = pd.concat(cs, ignore_index=True)
        ctrl.to_csv(f"{a.out}/control_per_tf.tsv", sep="\t", index=False)
        CS, CC = control_summary(ctrl)
    wide.to_csv(f"{a.out}/audit_per_tf.tsv", sep="\t", index=False)
    S.to_csv(f"{a.out}/audit_macro.tsv", sep="\t", index=False)
    P.to_csv(f"{a.out}/phenotype_per_tf.tsv", sep="\t", index=False)
    PS.to_csv(f"{a.out}/phenotype_substrate.tsv", sep="\t", index=False)
    CS.to_csv(f"{a.out}/control_macro.tsv", sep="\t", index=False)
    CC.to_csv(f"{a.out}/control_contrasts.tsv", sep="\t", index=False)
    rb = [pd.read_csv(f, sep="\t", dtype={"tf": str}) for f in sorted(glob.glob(f"{a.out}/lat_*.rob.tsv")) + sorted(glob.glob(f"{a.out}/reg_*.rob.tsv"))]
    RS_ = robust_summary(S, pd.concat(rb, ignore_index=True), decs) if rb else pd.DataFrame()
    RS_.to_csv(f"{a.out}/robust_cells.tsv", sep="\t", index=False)
    if len(RS_):
        S = S.merge(RS_[["system", "kind", "target", "method", "stability_sub", "class_rank", "rank_same"]], on=["system", "kind", "target", "method"], how="left")
        S.to_csv(f"{a.out}/audit_macro.tsv", sep="\t", index=False)
    loading_summary(wide, decs).to_csv(f"{a.out}/loading_summary.tsv", sep="\t", index=False)
    CL = claims(S)
    CL.to_csv(f"{a.out}/claims_prediction_falsifiers.tsv", sep="\t", index=False)
    cnt = []
    for tgt, g in S.groupby("target"):
        for cal in ("class", "class_wb") + (("class_aligned", "class_indep") if tgt == "regulon" else ()):
            vc = g[cal].value_counts()
            n_ = int(vc.sum())
            cnt.append(dict(target=tgt, calibration=cal, n_cells=n_, TC=int(vc.get("target-common-dominated", 0)), MX=int(vc.get("mixed", 0)), RS=int(vc.get("regulator-specific", 0)),
                            NA=int(vc.get("NA", 0)), RS_fraction=float(vc.get("regulator-specific", 0) / max(n_, 1))))
    pd.DataFrame(cnt).to_csv(f"{a.out}/class_counts.tsv", sep="\t", index=False)
    cls = S[S.n_tf >= 1].assign(cell=lambda x: x.system + "|" + x.target).pivot_table(index="method", columns="cell", values="class", aggfunc="first")
    cls.to_csv(f"{a.out}/class_matrix.tsv", sep="\t")
    meta = {"script_sha256": next(iter(shas)), "n_rows": len(wide), "n_macro": len(S), "not_run": {}, "parity": {}, "coverage": {}}
    for f in sorted(glob.glob(f"{a.out}/*.meta.json")):
        m_ = json.load(open(f))
        for tf, mt in m_["tfs"].items():
            key = f"{m_['system']}|{m_['kind']}|{tf}"
            if str(mt.get("status", "")).startswith("NOT_RUN"):
                meta["not_run"][key] = mt["status"]
            else:
                meta["parity"][key] = mt.get("parity_v3", mt.get("canon_vs_bank_row_absdiff"))
                meta["coverage"][key] = dict(used=mt["methods_used"], missing=mt["methods_missing"])
    json.dump(meta, open(f"{a.out}/receipt.json", "w"), indent=1, default=str)
    print(f"[audit merge] {len(wide)} per-TF rows, {len(S)} macro rows; not_run={len(meta['not_run'])}; classes:", S["class"].value_counts().to_dict())


# selftest (synthetic; assert-based)
# Synthetic axis: common target-level latent c (responsiveness), each bank row = load*c + noise
def _synth(kind, rng, n_genes=3000, n_bank=300, sig=1.4):
    c = rng.standard_normal(n_genes)
    load = rng.uniform(0.5, 1.0, n_bank)
    Xb = load[:, None] * c[None, :] + rng.standard_normal((n_bank, n_genes)) * 1.0
    s = rng.standard_normal(n_genes)
    focal = 0.5 * c + 1.0 * s + 0.3 * rng.standard_normal(n_genes)
    lat = c if kind == "common" else s
    y = (lat + 0.3 * rng.standard_normal(n_genes)) > np.quantile(lat, 0.8)
    return c, Xb, focal, y


def selftest():
    from scipy.stats import rankdata
    rng = np.random.default_rng(0)
    # 1. auroc_rows == tie-grouped AUROC of gb_canon_table (with ties)
    M = np.round(rng.standard_normal((5, 400)), 1)
    y = rng.random(400) < 0.3
    got = auroc_rows(M, y)
    for i in range(5):
        pt = A.point(M[i], y, np.ones(400, bool))
        assert abs(pt["AUROC"] - got[i]) < 1e-12, (pt["AUROC"], got[i])
    assert np.isnan(auroc_rows(M, np.zeros(400, bool))).all()
    # 2. decomp_vectors: R = S - P exactly, no coefficient
    v, p = rng.standard_normal(100), rng.standard_normal(100)
    mk = rng.random(100) < 0.8
    S, P, Rm = decomp_vectors(v, p, mk)
    assert np.allclose(Rm[mk], S[mk] - p[mk]) and abs(S[mk].mean()) < 1e-12 and abs(S[mk].std() - 1) < 1e-12 and np.isnan(Rm[~mk]).all()
    # 3. classify rule
    assert classify(.60, .60, False) == "target-common-dominated" and classify(.60, .592, False) == "target-common-dominated" and classify(.60, .585, False) == "mixed" and classify(.70, .55, True) == "regulator-specific"
    assert classify(.70, .55, False) == "mixed" and classify(.55, .60, True) == "mixed" and classify(np.nan, .5, False) == "NA" and classify(.6, .5, None) == "NA"
    # 4. signflip exact / MC agree
    d = rng.standard_normal(9) + .7
    p1, m1 = signflip(d)
    assert m1 == "exact_2^9" and 2 / 512 - 1e-12 <= p1 <= 1
    assert signflip(np.ones(21))[1].startswith("mc_") and signflip(np.ones(3))[0] == 0.25
    # 5. end-to-end synthetic through core(): target-common truth -> prior ~ full, residual ~ .5
    for kind in ("common", "specific"):
        res = []
        for rep in range(3):
            c, Xb, focal, y = _synth(kind, np.random.default_rng(10 + rep))
            n_g, n_b = len(c), len(Xb)
            allrows = np.vstack([focal[None, :], Xb])                                     # focal is bank row 0 in this toy
            z = (allrows - allrows.mean(1, keepdims=True)) / allrows.std(1, keepdims=True)
            Ssum = z.sum(0)
            prior = (Ssum - z[0]) / n_b                                                   # leave-focal-out mean of the others' row-z
            E = np.arange(1, n_b + 1)
            Pb = (Ssum[None, :] - z[E]) / (n_b)                                           # each decoy's own leave-out prior (bank incl. the focal)
            M1 = np.ones(n_g, bool)
            counts = np.random.default_rng(1).multinomial(n_g, np.full(n_g, 1.0 / n_g), size=100).astype(np.int16)
            rows, npz = core("SYN", "F", "lattice", ["GeoBRIDGE"], {"GeoBRIDGE": focal}, {"GeoBRIDGE": prior}, M1, {"resp": (y, M1)}, lambda yy, rr: True, np.arange(n_g), counts,
                             {"GeoBRIDGE": Xb}, {"GeoBRIDGE": Pb}, [f"d{i}" for i in E], np.random.default_rng(5).standard_normal((len(E), 1)), np.zeros(1), dict(block="SYN", focal_in_bank=True))
            r = rows[0]
            # robustness gates through robust() on a toy bank (rows: focal = 0, decoys 1..n_b)
            bank_ = types.SimpleNamespace(members=[f"m{i}" for i in range(n_b + 1)], P={"GeoBRIDGE": {"Z": z}}, X={"GeoBRIDGE": allrows}, T=np.ones(n_g, bool), tcol=np.full(n_g, -1),
                                          bank_cols=np.zeros(n_b + 1, int))
            rrows, rz = robust("SYN", "F", "lattice", ["GeoBRIDGE"], {"GeoBRIDGE": focal}, {"GeoBRIDGE": prior}, M1, {"resp": (y, M1)}, lambda yy, rr: True, bank_, 0, np.arange(n_g), E,
                               {"GeoBRIDGE": Xb}, {"GeoBRIDGE": Pb}, [f"d{i}" for i in E], np.random.default_rng(5).standard_normal((len(E), 1)), np.zeros(1), dict(block="SYN"))
            rr = pd.DataFrame(rrows)
            assert len(rr) == len(VARIANTS) and rz["robz|GeoBRIDGE|resp"].shape == (len(VARIANTS), n_b) and set(rr.variant) == set(VARIANTS)
            sub = rr[rr.variant.str.startswith("sub")]
            stab_rs = float((sub.class_tf == "regulator-specific").mean())
            if kind == "specific":
                assert stab_rs >= 0.9 and (rr.AUROC_resid > 0.7).all(), (stab_rs, rr.AUROC_resid.min())
            else:
                assert stab_rs == 0.0, stab_rs
            assert r["status"] == "OK" and np.isfinite(r["AUROC_full"]) and np.isfinite(r["AUROC_resid"]) and r["N_decoy"] == n_b
            # residual AUROC from core == auroc_rows on the same vector
            S_, P_, R_ = decomp_vectors(focal, prior, M1)
            assert abs(auroc_rows(R_[None, :], y)[0] - r["AUROC_resid"]) < 1e-12
            assert abs(auroc_rows(prior[None, :], y)[0] - r["AUROC_prior"]) < 1e-12
            res.append(r)
        if kind == "common":
            for r in res:
                assert r["AUROC_prior"] >= r["AUROC_full"] - 0.05 and abs(r["AUROC_resid"] - 0.5) < 0.15 and r["class"] in ("target-common-dominated", "mixed") and r["class_wb"] != "regulator-specific" and not r["resid_above_q95"], r
        else:
            for r in res:
                assert r["AUROC_resid"] > 0.7 and r["AUROC_full"] - r["AUROC_prior"] > 0.05 and r["resid_above_q95"] and r["class"] == "regulator-specific", r
    # 6. macro_decoy_p: extreme focal -> small p
    dl = [{f"d{i}": float(rng.normal(.5, .02)) for i in range(200)} for _ in range(3)]
    pa, pi, nc = macro_decoy_p(np.array([.6, .6, .6]), dl)
    assert pa < .05 and pi < .05 and nc == 200
    pa, pi, nc = macro_decoy_p(np.array([.5, .5, .5]), dl)
    assert pa > .2 and pi > .2
    # 7. hgb / rank-normal / fold OLS on a feature-determined prior: R2 high, residual ~ noise
    n = 2000
    X = rng.standard_normal((n, 9))
    yv = np.tanh(X[:, 0]) + 0.5 * X[:, 1] * X[:, 2] + 0.2 * rng.standard_normal(n)
    folds = rng.integers(0, 5, n)
    mk = np.ones(n, bool)
    fit, res_, st = hgb_cross_fit(yv, X, folds, mk)
    assert st["R2_heldout"] > 0.5 and np.isfinite(fit).all() and abs(np.corrcoef(res_, yv - fit)[0, 1] - 1) < 1e-9
    xn = np.column_stack([rank_normal(X[:, j], mk) for j in range(9)])
    assert abs(xn[:, 0].mean()) < 1e-3 and abs(xn[:, 0].std() - 1) < 0.02
    f2, r2, s2 = D.fold_ols(yv, xn, folds, mk)
    assert np.isfinite(f2).all() and s2["R2_heldout"] < st["R2_heldout"] + 0.05
    # 8. macro_z_p == gb_attribution.substrate_p (aligned end) ; summarize(): primary class from the
    # macro test of the adjusted z, assertions
    names = [f"d{i}" for i in range(300)]
    zt = [rng.standard_normal(300) for _ in range(4)]
    pa, pi, nc = macro_z_p(np.array([.1, -.2, .3, 0.0]), [(names, z_) for z_ in zt])
    pa2, pi2 = A.substrate_p(np.array([.1, -.2, .3, 0.0]), [dict(zip(names, z_)) for z_ in zt])
    assert abs(pa - pa2) < 1e-12 and nc == 300 and abs(pi - pi2) < 0.02
    assert macro_z_p(np.array([5., 5, 5, 5]), [(names, z_) for z_ in zt])[0] < .01
    rows, decs = [], {}
    for tf in "ABCD":
        decs.setdefault(("S", "lattice"), {})[tf] = {"names": np.asarray(names), "zres|M|resp": zt["ABCD".index(tf)].astype(np.float32), "zres|R|resp": zt["ABCD".index(tf)].astype(np.float32)}
        for m, zc, au_p in (("M", 0.0, .6), ("R", 6.0, .5)):
            rows.append(dict(system="S", tf=tf, kind="lattice", target="resp", method=m, status="OK", gate=True, PREV=.2, N_decoy=300, N_cal=300, p_decoy=.5, resid_above_q95=False, p_cal=.5, z_cal=zc,
                             **{f"{nm}_{c}": v for c, v in (("full", .7 if m == "R" else .6), ("prior", au_p), ("resid", .5)) for nm in ("AUROC", "AP", "APL")},
                             **{f"{nm}_{c}_{e}": 0.0 for c in COMPS_ for nm in ("AUROC", "AP", "APL") for e in ("lo", "hi")}, d_AUROC_fp=.0, r_AUROC=.0, d_APL_fp=.0, r_APL=.0))
    Sm = summarize(pd.DataFrame(rows), decs).set_index("method")
    assert len(Sm) == 2 and Sm.loc["M", "class"] == "target-common-dominated" and Sm.loc["R", "class"] == "regulator-specific" and Sm.loc["R", "signflip_floor"] == 2 / 16, Sm[["class", "p_adj_aligned"]]
    Cl = claims(Sm.reset_index().assign(kind="lattice", target="resp"))
    assert len(Cl) == 2 and set(Cl.calibration) == {"class", "class_wb"}
    print("[audit] selftest PASS")


# main
def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("run")
    p.add_argument("--systems", required=True)
    p.add_argument("--tfs", default="")
    p.add_argument("--B", type=int, default=2000)
    p.add_argument("--out", required=True)
    p.add_argument("--slice", type=int, default=0)
    p.add_argument("--nslice", type=int, default=1)
    p = sub.add_parser("regulon")
    p.add_argument("--systems", required=True)
    p.add_argument("--B", type=int, default=2000)
    p.add_argument("--out", required=True)
    p.add_argument("--max-tfs", type=int, default=0, dest="max_tfs")
    p.add_argument("--slice", type=int, default=0)
    p.add_argument("--nslice", type=int, default=1)
    p = sub.add_parser("merge")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"run": cmd_run, "regulon": cmd_regulon, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
