#!/usr/bin/env python
"""PRIOR-ADJUSTED BENCHMARK (PI manuscript lock 2026-10-02 item 3; internal review internal_review_log "Q3 prior-adjusted benchmark").  EVALUATION ONLY: production S, P and every method's score are IMPORTED
and never edited; the PRIMARY quantity has NO fitted weight and NO truth-optimised element.  Retrospective systems only (CARDIAC / CARDIACV / RETINA are refused).

QUESTION  how much of each method's benchmark performance survives after its own target-common component, and does the method ranking change?  The primary increment measures fixed-combination
          utility of adding the regulator row to the method's own leave-focal-out prior; it does NOT establish regulator specificity (the null of the increment is not zero: adding an uninformative
          row to P tends to DILUTE it).  Winner changes are descriptive only.

SCOPE     each of the 10 methods (GeoBRIDGE, |Pearson|, GENIE3, GRNBoost2, RegVelo |J|, Velorama, RegDiffusion, DeepSEM, PIDC, FastTENET) x every endpoint of the 10-method audit lattice
          (gb_audit_decomp: lattice targets resp / AB_prim / CD_prim / occ_prim / AC_prim / BD_prim per TF unit, plus the per-regulator regulon endpoints), on the SAME frozen universe M1 the audit uses
          (gb_ptable.lattice_unit / regulon_units; target rows = the audit's rows).  Ungated cells are kept and flagged (DESCRIPTIVE_ONLY), never dropped.  BM breadth is not a lattice endpoint.
PRIMARY   per (unit, target, method m), on identical genes: S = the method's cognate row of the focal regulator r; P = the method's OWN leave-focal-out prior P_m^(-r);
              M0 = midrank(P_m)                              (== P_m: AUROC / AP / P50 are rank-based -> Perf(M0) = Perf(P_m) EXACTLY; asserted)
              M1 = [midrank(P_m) + midrank(S_m,r)] / 2        fixed 1:1 rank average, ranks over the target rows, ties = average ranks
          reported: Perf(S), Perf(P), Perf(M1) and the RAW increment Delta = Perf(M1) - Perf(M0) for AUROC, AP (+ AP-lift) and tie-aware P50 (gb_canon_table.eval_rows), with the paired 95% percentile
          interval and two-sided bootstrap p; plus rho_SP = Spearman(S, P) on the rows.  The raw Delta is DESCRIPTIVE: it is confounded by each method's own prior quality.
DECOY-CALIBRATED INCREMENT (coordinator 2026-10-02; the quantity the manuscript interprets)  for every GATED (unit, target, method m):
          DECOYS = the N_DECOY = 50 nearest NON-cognate source rows of THAT method's own bank, standardised Euclidean on (spliced detection, log1p mean Ms of the untreated control fixture) exactly as
          gb_defense_dutil.nearest_matched, restricted to rows whose raw score row and leave-decoy-out prior are finite on the cell's rows (so a decoy is evaluated on the IDENTICAL genes as the cognate;
          the audit's own decoy-pool eligibility).  For decoy d:  M0_d = midrank(P_m^(-d)),  M1_d = [midrank(P_m^(-d)) + midrank(S_m,d)] / 2,  Delta_d = Perf(M1_d) - Perf(M0_d)  evaluated on the FOCAL
          truth, same rows, same resampling draws (P_m^(-d) = gb_attribution.Bank.prior_rows = the method's leave-decoy-out prior).
          reported per (cell, method, metric): cognate percentile among decoys (dutil pool_rank: frac(Delta_d < Delta_cog) + .5 frac(=)), pool p = (1 + #{Delta_d >= Delta_cog}) / (1 + n_decoy), calibrated
          Delta = Delta_cog - median_d Delta_d (point AND per paired draw -> interval + bootstrap p, gene-identity and module-block).  Cells with < MIN_DECOY (20) usable decoys are NO_CALL (never zero);
          < 50 is flagged.  Sensitivity (point level only): the same statistics with perturbed-panel TFs removed from the decoy set (a perturbed TF may be a genuine regulator of the focal truth).
RANKING   Kendall tau-b between the raw-performance method ranking (Perf(S_m)) and (i) the RAW incremental ranking (Delta_m) and (ii) the CALIBRATED incremental ranking (calibrated Delta_m) per endpoint and
          per block / system macro (mean over GATED units, per draw), with the paired-resampling interval of tau-b (every draw uses ONE resample for all methods), the rank of GeoBRIDGE under every ranking,
          the winner under every ranking and the fraction of draws in which the winner changes; plus tau-b of each increment ranking against the PRIOR-QUALITY ranking Perf(P_m) (the confounding the
          calibration removes).  Descriptive.
SECONDARY (DIAGNOSTIC, supervised, never a method component)  fixed-hyperparameter logistic  Y ~ f(P)  vs  Y ~ f(P) + S  with f = 5-knot cubic quantile spline (gb_prior_semantics.spline_basis, fit on the
          rows, label-free), S standardised, L2 logistic C = 1 (max_iter 5000); cross-fitted over 5 control-module-hash folds (BlockAxis.fold_of with the frozen control-only module map =
          "control-defined gene blocks"); the model is REFIT inside every resample (module-block multinomial draws, sample weights = multiplicities, training rows with multiplicity 0 dropped, folds fixed
          by original module identity).  Cells with < 95% valid refit draws are FLAGGED (low_valid_draws).  LEAVE-ONE-TF-OUT x module-fold arm for blocks with >= 3 gated TFs; B_loto refit draws by block
          size (<= 12 units: B_loto, <= 40: min(B_loto, 30), larger: point estimate only).
UNCERTAINTY  PRIMARY = "gene-identity resampling" (the synchronized BlockAxis draws of gb_ptable / the audit); SENSITIVITY = "module-block resampling" (frozen control-only coexpression modules, G/50);
          the secondary uses module-block draws only (its folds are module folds).  Macro rows are conditional on the fixed unit set; shared TFs / controls are not independent replicates.
PARITY    S / P AUROC, AP, AP-lift per (unit, target, method) vs audit_decomp_v2/audit_per_tf.tsv (1e-9); GB S vs the lattice table; M0 == P; the cognate S / leave-focal-out prior reproduce the bank row /
          Bank.prior_rows of the focal source (the decoy pipeline run on the focal row IS the cognate pipeline); fail-closed (--no-parity to disable).

  python scratch/gb_prior_adjusted.py selftest [--data]
  python scratch/gb_prior_adjusted.py run --systems PANC,BM_CHIPATLAS [--regulon PANC,BONEMARROW] --B 2000 --out DIR [--tfs A,B] [--secondary resp,regulon] [--B-secondary N] [--loto | --loto-only]
                                          [--n-decoy 50 | --no-decoy] [--n-jobs 16]
  python scratch/gb_prior_adjusted.py merge --out DIR
  python scratch/gb_prior_adjusted.py report --out DIR --groups PANC/RFX6,MACRO:BM [--targets resp]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")                       # LR fits are tiny; parallelism is across (cell, method) tasks

import argparse
import glob
import json
import multiprocessing as mp
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
import gb_ptable as PT  # noqa: E402  (NOT edited by this script; unit builders / draws / modules / guards are shared so both tables use literally the same universes)

GB, MET, B_DEFAULT = PT.GB, PT.MET, PT.B_DEFAULT
SCRIPT_SHA0 = PT._sha(__file__)
LAT_TARGETS = ["resp", "AB_prim", "CD_prim", "occ_prim", "AC_prim", "BD_prim"]      # gb_attribution.TARGETS / audit_decomp
METHODS = ["GeoBRIDGE", "|Pearson|", "GENIE3", "GRNBoost2", "RegVelo |J|", "Velorama", "RegDiffusion", "DeepSEM", "PIDC", "FastTENET"]
B_SEC_DEFAULT, B_LOTO_DEFAULT = 999, 100
LOGIT_C, LOGIT_ITER = 1.0, 5000
N_DECOY, MIN_DECOY = 50, 20                              # decoys per (cell, method): target / minimum usable (below -> NO_CALL)
MIN_VALID_FRAC = 0.95                                    # refit draws: below this fraction a logistic cell is FLAGGED
SIZES2 = (PT.PRIMARY_SIZE,)                              # Table 2 resamples genes (primary) and the G/50 control modules (sensitivity); G/25, G/100 are Table 1's dependence sweep
PARITY_BANK_TOL = 1e-5                                   # |bank row - cognate vector| <= tol * max(1, max|score|) (float32 shards)


# ===================================================================================================== numerics (unit-tested by selftest)
def midrank_on(v, rows):
    """average (mid)ranks of v over `rows` placed back on the full axis (NaN elsewhere)"""
    from scipy.stats import rankdata
    out = np.full(len(v), np.nan)
    out[rows] = rankdata(np.asarray(v, float)[rows])
    return out


def combine(P, S, rows):
    """(M0, M1) on `rows`: M0 = midrank(P); M1 = [midrank(P) + midrank(S)] / 2 (fixed 1:1, ties = average ranks, identical genes)"""
    rP, rS = midrank_on(P, rows), midrank_on(S, rows)
    return rP, (rP + rS) / 2.0


def tau_b(a, b):
    from scipy.stats import kendalltau
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return np.nan
    return float(kendalltau(a, b, variant="b")[0])


def rank_change(Sd, Dd, methods):
    """Sd, Dd: {method: (n,) values or (B, ) draws} -> arrays.  Returns dict(tau, rank_gb_raw, rank_gb_incr, winner_raw, winner_incr) for point (1-D) or per-draw (2-D) inputs."""
    A_ = np.column_stack([np.atleast_1d(Sd[m]) for m in methods]) if np.ndim(Sd[methods[0]]) else np.array([[Sd[m] for m in methods]])
    Bm = np.column_stack([np.atleast_1d(Dd[m]) for m in methods]) if np.ndim(Dd[methods[0]]) else np.array([[Dd[m] for m in methods]])
    ok = np.isfinite(A_).all(1) & np.isfinite(Bm).all(1)
    g = methods.index(GB)
    return dict(ok=ok, tau=np.array([tau_b(A_[i], Bm[i]) if ok[i] else np.nan for i in range(len(A_))]), rank_gb_raw=np.where(ok, 1 + (A_ > A_[:, [g]]).sum(1), np.nan),
                rank_gb_incr=np.where(ok, 1 + (Bm > Bm[:, [g]]).sum(1), np.nan), winner_raw=np.where(ok, A_.argmax(1), -1), winner_incr=np.where(ok, Bm.argmax(1), -1))


def wmetrics(score, y, w, n_rows):
    """tie-grouped weighted metrics of one score vector (rows already restricted) under multiplicities w (gb_canon_table.weighted_metrics, expanded-sample identity)"""
    import gb_canon_table as CT
    o, st = CT.prep_scores(np.asarray(score, float))
    m = CT.weighted_metrics(np.asarray(w, float)[None, o], np.asarray(y, bool)[o], st, n_rows=n_rows)
    return {k: float(m[k][0]) for k in MET}


def oof_logit(X, y, fold, w, nconv=None):
    """cross-fitted fixed L2 logistic (C=1): held-out decision function by `fold`; training rows weighted by multiplicity w (w == 0 dropped); None if a training fold lacks >= 5 of a class.
    ConvergenceWarnings are COUNTED into nconv[0] (never silently dropped)."""
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    out = np.full(len(y), np.nan)
    for f in np.unique(fold):
        te, tr = fold == f, (fold != f) & (w > 0)
        if y[tr].sum() < 5 or (~y[tr]).sum() < 5:
            return None
        with warnings.catch_warnings(record=True) as wl:
            warnings.simplefilter("always", ConvergenceWarning)
            out[te] = LogisticRegression(C=LOGIT_C, max_iter=LOGIT_ITER).fit(X[tr], y[tr], sample_weight=w[tr]).decision_function(X[te])
        if nconv is not None:
            nconv[0] += sum(issubclass(x.category, ConvergenceWarning) for x in wl)
    return out


def design(P, S, rows):
    """(X0, X1): spline(P) | spline(P) + standardised S, transforms fit on the rows (label-free) -- gb_prior_semantics.spline_basis / lin_block"""
    import gb_prior_semantics as PS
    Xp = PS.spline_basis(np.asarray(P, float), rows)[rows]
    z = PS.lin_block(np.asarray(S, float), rows)[rows]
    return Xp, np.hstack([Xp, z.reshape(len(Xp), -1)])


# ===================================================================================================== decoy calibration: matching, per-decoy increment, calibrated statistics
_MATCH = {}


def match_ctx(axis, bank):
    """truth-free source-matching covariates of every bank member, standardised over ALL members (gb_defense_dutil.nearest_matched): spliced detection (bank.json) and log1p mean Ms of the untreated
    control fixture (frozen by gb_ptable `modules`, before any evaluation).  Fails closed on a missing / non-finite covariate."""
    if axis not in _MATCH:
        import gb_catbank as CB
        bj = json.load(open(f"{CB.OUT}/{axis}_bank.json"))
        if list(bj["members"]) != list(bank.members) or bj["sha256_members"] != CB.members_sha(bj["members"]):
            raise SystemExit(f"REFUSING: bank.json of {axis} differs from the loaded bank")
        det = np.asarray([bj["detection"][m] for m in bank.members], float)
        lms_s = PT.load_matchcov(axis)
        lms = np.asarray([lms_s[m] for m in bank.members], float)
        if not (np.isfinite(det).all() and np.isfinite(lms).all()):
            raise SystemExit(f"REFUSING: non-finite matching covariate for {axis}: detection {int((~np.isfinite(det)).sum())}, log1p mean Ms {int((~np.isfinite(lms)).sum())}")
        F = np.column_stack([det, lms])
        _MATCH[axis] = dict(F=(F - F.mean(0)) / F.std(0), det=det, lms=lms)
    return _MATCH[axis]


def pick_decoys(F, r, elig, n):
    """the n nearest ELIGIBLE non-focal sources: standardised Euclidean distance to the focal source r, stable ties; returns (indices in match order, distances).  With every source eligible this is
    exactly gb_defense_dutil.nearest_matched."""
    d = np.sqrt(((F - F[r]) ** 2).sum(1))
    d[~np.asarray(elig, bool)] = np.inf
    d[r] = np.inf
    o = np.argsort(d, kind="stable")[:n]
    o = o[np.isfinite(d[o])]
    return o, d[o]


def unit_ctx(u):
    """per-unit decoy context: gene -> bank-axis / T7 column maps (gb_audit_decomp.masks_and_priors recipe), the focal bank row, the matching features"""
    bank = u["bank"]
    genes = np.asarray(u["genes"]).astype(str)
    ai = np.array([bank.ai.get(g, -1) for g in genes])
    has = ai >= 0
    ai0 = np.where(has, ai, 0)
    tc = np.where(has, bank.tcol[ai0], -1)
    r = bank.row_of(u["tf"])
    if r is None:
        raise SystemExit(f"REFUSING: focal {u['system']}/{u['tf']} not in the bank")
    return dict(ai0=ai0, has=has, tc=tc, r=r, F=match_ctx(u["axis"], bank)["F"])


def perturbed_names(u):
    """upper-case perturbed-panel TF names of a lattice unit (gb_attribution.perturbed_set), None where undefined (regulon units: the label is not a perturbation panel)"""
    if u["kind"] != "lattice":
        return None
    import gb_attribution as A
    import gb_layer_lattice as L
    import gb_lattice_extra as LE
    try:
        return {x.upper() for x in A.perturbed_set(u["system"], L, LE)}
    except SystemExit:
        return None


def bank_cognate_parity(u, ctx, m, rows):
    """the focal source's bank row / leave-focal-out prior must equal the cognate vector / prior the unit carries (so that the decoy pipeline on the focal row IS the cognate pipeline).  -> (d_score, d_prior, tol)"""
    bank, r, cols = u["bank"], ctx["r"], np.flatnonzero(rows)
    a, b = bank.X[m][r][ctx["ai0"][cols]], np.asarray(u["V"][m], float)[cols]
    p_b = bank.prior_rows(m, np.array([r]))[0][ctx["tc"][cols]]
    p_u = np.asarray(u["prior"][m], float)[cols]
    with np.errstate(invalid="ignore"):
        return float(np.abs(a - b).max()), float(np.abs(p_b - p_u).max()), PARITY_BANK_TOL * max(1.0, float(np.abs(b).max()))


def prepare_decoys(u, ctx, m, rows, n_decoy=N_DECOY):
    """matched decoys of (unit, method, cell rows): dict(status, sel, dist, Sd, Pd, cols, n_eligible).  Eligibility = the audit's decoy-pool rule restricted to THIS cell: the source is not the focal (any
    case variant of its name), its raw score row is finite on every row of the cell and usable (bank.row_ok), and its leave-self-out prior is finite on every row."""
    bank, r = u["bank"], ctx["r"]
    cols = np.flatnonzero(rows)
    if not (ctx["has"][cols].all() and (ctx["tc"][cols] >= 0).all()):
        return dict(status="NO_CALL: cell rows outside the bank axis / T7 set", n_eligible=0, sel=np.array([], int))
    Xc = bank.X[m][:, ctx["ai0"][cols]]
    cand = np.flatnonzero(np.isfinite(Xc).all(1) & bank.row_ok[m] & (bank.mu != u["tf"].upper()))
    Pc = bank.prior_rows(m, cand)[:, ctx["tc"][cols]] if len(cand) else np.zeros((0, len(cols)))
    ok2 = np.isfinite(Pc).all(1)
    cand, Pc = cand[ok2], Pc[ok2]
    elig = np.zeros(len(bank.members), bool)
    elig[cand] = True
    sel, dist = pick_decoys(ctx["F"], r, elig, n_decoy)
    out = dict(sel=sel, dist=dist, n_eligible=int(len(cand)), cols=cols)
    if len(sel) < MIN_DECOY:
        return dict(out, status=f"NO_CALL: {len(sel)} usable decoys (< {MIN_DECOY})")
    pos = {int(c): i for i, c in enumerate(cand)}
    ii = np.array([pos[int(s)] for s in sel])
    return dict(out, status="OK" if len(sel) >= n_decoy else f"OK_FEWER_THAN_{n_decoy}", Sd=Xc[sel].astype(float), Pd=Pc[ii])


_D = {}


def _decoy_task(key):
    """Delta_d = Perf(M1_d) - Perf(M0_d) for every decoy of one (cell, method): point + B gene-identity draws + B module-block draws.  Runs in a forked worker; arrays come from the module-global _D."""
    c, C, B = _D[key], _D["__C__"], _D["__B__"]
    n, cols, rows, y, gi = c["n"], c["cols"], c["rows"], c["y"], c["gi"]
    nd = len(c["Sd"])
    pt = {k: np.full(nd, np.nan) for k in MET}
    dg = {k: np.full((nd, B), np.nan, np.float32) for k in MET}
    dm = {k: np.full((nd, B), np.nan, np.float32) for k in MET}
    for j in range(nd):
        Pf, Sf = np.full(n, np.nan), np.full(n, np.nan)
        Pf[cols], Sf[cols] = c["Pd"][j], c["Sd"][j]
        M0, M1 = combine(Pf, Sf, rows)
        p0, (g0, m0) = PT.split_eval(M0, y, rows, gi, C, B)
        p1, (g1, m1) = PT.split_eval(M1, y, rows, gi, C, B)
        for k in MET:
            pt[k][j] = p1[k] - p0[k]
            dg[k][j], dm[k][j] = g1[k] - g0[k], m1[k] - m0[k]
    return key, dict(pt=pt, dg=dg, dm=dm)


def decoy_stats(c_pt, c_dg, c_dm, d_pt, d_dg, d_dm, min_n=MIN_DECOY):
    """calibrated statistics of ONE metric.  c_*: the cognate increment (point, gene draws (B,), module draws (B,)); d_*: the decoy increments (n,), (n, B), (n, B).  Returns dict with the point
    summaries (median / 5-95% of the decoy increments, cognate percentile = frac(decoy < cog) + .5 frac(=), pool p = (1 + #{decoy >= cog}) / (1 + n), calibrated = cog - median) and, per scheme, the
    per-draw decoy-median (dm_g / dm_m), the calibrated interval and the bootstrap p of calibrated <= 0.  NO_CALL (NaN) when fewer than min_n finite decoys."""
    d_pt = np.asarray(d_pt, float)
    fin = np.isfinite(d_pt)
    n = int(fin.sum())
    nan = np.nan
    out = dict(n=n, med=nan, p05=nan, p95=nan, pct=nan, p_pool=nan, cal=nan)
    if n >= min_n and np.isfinite(c_pt):
        dv = d_pt[fin]
        out.update(med=float(np.median(dv)), p05=float(np.percentile(dv, 5)), p95=float(np.percentile(dv, 95)), pct=float((dv < c_pt).mean() + 0.5 * (dv == c_pt).mean()),
                   p_pool=float((1 + (dv >= c_pt).sum()) / (1 + n)), cal=float(c_pt - np.median(dv)))
    for tag, cd, dd in (("g", c_dg, d_dg), ("m", c_dm, d_dm)):
        dd = np.asarray(dd, float)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med_b = np.nanmedian(dd, axis=0)
        med_b = np.where(np.isfinite(dd).sum(0) >= min_n, med_b, np.nan)
        cal_b = np.asarray(cd, float) - med_b
        out[f"dm_{tag}"] = med_b
        out[f"cal_{tag}_lo"], out[f"cal_{tag}_hi"] = PT.pct_ci(cal_b)
        out[f"p_cal_{tag}"] = PT.boot_p(cal_b)
    return out


def decoy_columns(stats, meta, nopert=None):
    """flatten per-metric decoy_stats into cells-table columns (+ meta columns, + the perturbed-panel-excluded point-level sensitivity)"""
    row = dict(meta)
    for k in MET:
        s = stats[k]
        row.update({f"dec_med_{k}": s["med"], f"dec_p05_{k}": s["p05"], f"dec_p95_{k}": s["p95"], f"cog_pct_{k}": s["pct"], f"p_pool_{k}": s["p_pool"], f"calD_{k}": s["cal"],
                    f"calD_{k}_lo": s["cal_g_lo"], f"calD_{k}_hi": s["cal_g_hi"], f"p_calD_{k}": s["p_cal_g"], f"calD_{k}_mod_lo": s["cal_m_lo"], f"calD_{k}_mod_hi": s["cal_m_hi"],
                    f"p_calD_{k}_mod": s["p_cal_m"]})
        if nopert is not None:
            row[f"calD_{k}_noPert"], row[f"p_pool_{k}_noPert"] = nopert[k]["cal"], nopert[k]["p_pool"]
    return row


# ===================================================================================================== primary: one (unit, target)
def cell_primary(u, target, y, rows, dr, B, gate):
    """all methods on one (unit, target).  Returns (rows, draw store keyed uid|method|metric|{S,P,M1}|{g,m}, keep) with keep[(method, metric)] = the cognate increment (point, gene draws, module draws; float64)."""
    gi, n = u["gi"], len(u["genes"])
    yy = np.asarray(y, bool)[rows]
    uid = f"{u['kind']}|{u['system']}|{u['tf']}|{target}"
    base = dict(kind=u["kind"], system=u["system"], block=u["block"], tf=u["tf"], target=target, macro_group=u["macro_group"], n_M1=int(u["M1"].sum()), n_rows=int(rows.sum()),
                n_pos=int(yy.sum()), n_neg=int((~yy).sum()), PREV=float(yy.mean()) if len(yy) else np.nan, gate=bool(gate), draws_fp=dr.fp)
    if not (yy.any() and (~yy).any()):
        return [dict(base, status="TOO_FEW", method=m) for m in u["use"]], {}, {}
    if dr.C.shape[0] != 2 * B:
        raise SystemExit(f"REFUSING: Table 2 draws must be [gene; module G/{PT.PRIMARY_SIZE}] (got {dr.C.shape[0]} draw rows for B={B})")
    status = "OK" if gate else "DESCRIPTIVE_ONLY"
    out, store, keep = [], {}, {}
    for m in u["use"]:
        S, P = np.where(rows, u["V"][m], np.nan), np.where(rows, u["prior"][m], np.nan)
        M0, M1 = combine(u["prior"][m], u["V"][m], rows)
        res = {nm: PT.split_eval(v, y, rows, gi, dr.C, B) for nm, v in (("S", S), ("P", P), ("M0", M0), ("M1", M1))}
        for k in MET:                                                              # IDENTITY: Perf(M0) == Perf(P) (rank-based metrics; ties preserved) -- point AND draws
            if not (abs(res["M0"][0][k] - res["P"][0][k]) < 1e-12 or (np.isnan(res["M0"][0][k]) and np.isnan(res["P"][0][k]))):
                raise SystemExit(f"REFUSING: Perf(M0) != Perf(P) for {uid}/{m}/{k}: {res['M0'][0][k]!r} vs {res['P'][0][k]!r}")
            for i in (0, 1):
                if not np.allclose(res["M0"][1][i][k], res["P"][1][i][k], atol=1e-12, equal_nan=True):
                    raise SystemExit(f"REFUSING: draws of Perf(M0) != Perf(P) for {uid}/{m}/{k}")
        from scipy.stats import spearmanr
        r = dict(base, status=status, method=m, rho_SP=float(spearmanr(S[rows], P[rows])[0]))
        for nm, lab in (("S", "S"), ("P", "P"), ("M1", "M1")):
            pt, (bg, bm) = res[nm]
            for k in MET:
                r[f"{lab}_{k}"] = pt[k]
                r[f"{lab}_{k}_lo"], r[f"{lab}_{k}_hi"] = PT.pct_ci(bg[k])
                r[f"{lab}_{k}_mod_lo"], r[f"{lab}_{k}_mod_hi"] = PT.pct_ci(bm[k])
        for k in MET:
            dg, dm = res["M1"][1][0][k] - res["M0"][1][0][k], res["M1"][1][1][k] - res["M0"][1][1][k]
            r[f"d_{k}"] = res["M1"][0][k] - res["M0"][0][k]
            r[f"d_{k}_lo"], r[f"d_{k}_hi"], r[f"p_{k}"] = (*PT.pct_ci(dg), PT.boot_p(dg))
            r[f"d_{k}_mod_lo"], r[f"d_{k}_mod_hi"], r[f"p_{k}_mod"] = (*PT.pct_ci(dm), PT.boot_p(dm))
            keep[(m, k)] = dict(pt=float(r[f"d_{k}"]), dg=dg, dm=dm)
            for nm in ("S", "P", "M1"):
                store[f"{uid}|{m}|{k}|{nm}|g"], store[f"{uid}|{m}|{k}|{nm}|m"] = res[nm][1][0][k].astype(np.float32), res[nm][1][1][k].astype(np.float32)
        out.append(r)
    return out, store, keep


# ===================================================================================================== secondary: refit-inside-resampling logistic diagnostic
_G = {}


def _logit_task(key):
    """one (cell, method): point + B refit draws of the M0 / M1 logistic models (module-block multiplicities of the genes of the rows).  Runs in a forked worker; arrays come from the module-global _G
    (the draw matrix is shared, not copied)."""
    c, C = _G[key], _G["__counts__"]
    X0, X1, y, fold, gi, n = c["X0"], c["X1"], c["y"], c["fold"], c["gi"], len(c["y"])
    nc = [0]
    o0, o1 = oof_logit(X0, y, fold, np.ones(n), nc), oof_logit(X1, y, fold, np.ones(n), nc)
    if o0 is None or o1 is None:
        return key, None
    pt0, pt1 = wmetrics(o0, y, np.ones(n), n), wmetrics(o1, y, np.ones(n), n)
    d0 = {k: np.full(len(C), np.nan) for k in MET}
    d1 = {k: np.full(len(C), np.nan) for k in MET}
    for b in range(len(C)):
        w = C[b][gi].astype(float)
        a0, a1 = oof_logit(X0, y, fold, w, nc), oof_logit(X1, y, fold, w, nc)
        if a0 is None or a1 is None:
            continue
        m0, m1 = wmetrics(a0, y, w, n), wmetrics(a1, y, w, n)
        for k in MET:
            d0[k][b], d1[k][b] = m0[k], m1[k]
    return key, dict(pt0=pt0, pt1=pt1, d0=d0, d1=d1, nconv=nc[0])


def secondary_rows(cells, dr, B, n_jobs):
    """cells: list of dict(key=(uid, method), base=row dict, P, S, y, rows, gi) -> secondary rows (module-block refit draws)."""
    _G.clear()
    _G["__counts__"] = dr.mods[PT.PRIMARY_SIZE].counts[:B]
    for c in cells:
        rows = c["rows"]
        X0, X1 = design(c["P"], c["S"], rows)
        _G[c["key"]] = dict(X0=X0, X1=X1, y=np.asarray(c["y"], bool)[rows], fold=dr.fold_module(c["gi"])[rows], gi=c["gi"][rows])
    keys = [k for k in _G if k != "__counts__"]
    if n_jobs > 1 and len(keys) > 1:
        with mp.get_context("fork").Pool(min(n_jobs, len(keys))) as pool:
            res = dict(pool.map(_logit_task, keys, chunksize=1))
    else:
        res = dict(_logit_task(k) for k in keys)
    out = []
    for c in cells:
        r = res[c["key"]]
        row = dict(c["base"], scheme="module-fold logistic, refit inside module-block resampling (DIAGNOSTIC)", B_refit=int(B))
        if r is None:
            out.append(dict(row, status_logit="NO_CALL: a training fold lacks a class", low_valid_draws=True))
            continue
        row["status_logit"] = "OK"
        for k in MET:
            row[f"M0logit_{k}"], row[f"M1logit_{k}"] = r["pt0"][k], r["pt1"][k]
            row[f"M0logit_{k}_lo"], row[f"M0logit_{k}_hi"] = PT.pct_ci(r["d0"][k])
            row[f"M1logit_{k}_lo"], row[f"M1logit_{k}_hi"] = PT.pct_ci(r["d1"][k])
            dd = r["d1"][k] - r["d0"][k]
            row[f"dlogit_{k}"] = r["pt1"][k] - r["pt0"][k]
            row[f"dlogit_{k}_lo"], row[f"dlogit_{k}_hi"], row[f"p_dlogit_{k}"] = (*PT.pct_ci(dd), PT.boot_p(dd))
        row["n_valid_draws"] = int(np.isfinite(r["d0"]["AUROC"] + r["d1"]["AUROC"]).sum())
        row["frac_valid_draws"] = row["n_valid_draws"] / max(int(B), 1)
        row["low_valid_draws"] = bool(row["frac_valid_draws"] < MIN_VALID_FRAC)                  # < 95% valid refit draws: interval is conditional on the draws that could be fit -> FLAGGED
        row["n_convergence_warnings"] = int(r["nconv"])
        out.append(row)
    _G.clear()
    return out


_L = {}


def loto_B(n_units, B_loto):
    """refit draws of a LOTO block by size (frozen rule): <= 12 units -> B_loto, <= 40 -> min(B_loto, 30), larger -> 0 (point estimate only; every draw refits n_units x 5 pooled models)"""
    return int(B_loto) if n_units <= 12 else (min(int(B_loto), 30) if n_units <= 40 else 0)


def _loto_task(key):
    """leave-one-TF-out x module-fold for one method of one block: held-out TF AND held-out module.  Training = every OTHER TF's rows of the training folds; refit per draw (shared multiplicities)."""
    c, C = _L[key], _L["__counts__"][:_L[key]["B"]]
    units = c["units"]
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression
    nc = [0]

    def run(w_gene):
        res = []
        for t, ut in enumerate(units):
            o0, o1 = np.full(len(ut["y"]), np.nan), np.full(len(ut["y"]), np.nan)
            for f in np.unique(ut["fold"]):
                te = ut["fold"] == f
                tr0, tr1, ytr, wtr = [], [], [], []
                for j, uj in enumerate(units):
                    if j == t:
                        continue
                    w = (w_gene[uj["gi"]] if w_gene is not None else np.ones(len(uj["y"])))
                    m = (uj["fold"] != f) & (w > 0)
                    tr0.append(uj["X0"][m]), tr1.append(uj["X1"][m]), ytr.append(uj["y"][m]), wtr.append(w[m])
                Y = np.concatenate(ytr)
                if Y.sum() < 5 or (~Y).sum() < 5:
                    return None
                Wt = np.concatenate(wtr)
                with warnings.catch_warnings(record=True) as wl:
                    warnings.simplefilter("always", ConvergenceWarning)
                    o0[te] = LogisticRegression(C=LOGIT_C, max_iter=LOGIT_ITER).fit(np.vstack(tr0), Y, sample_weight=Wt).decision_function(ut["X0"][te])
                    o1[te] = LogisticRegression(C=LOGIT_C, max_iter=LOGIT_ITER).fit(np.vstack(tr1), Y, sample_weight=Wt).decision_function(ut["X1"][te])
                nc[0] += sum(issubclass(x.category, ConvergenceWarning) for x in wl)
            wt = (w_gene[ut["gi"]] if w_gene is not None else np.ones(len(ut["y"])))
            res.append((wmetrics(o0, ut["y"], wt, len(ut["y"])), wmetrics(o1, ut["y"], wt, len(ut["y"]))))
        return res
    pt = run(None)
    draws = [run(C[b].astype(float)) for b in range(len(C))]
    return key, dict(pt=pt, draws=draws, nconv=nc[0])


def loto_rows(blocks, dr, B_loto, n_jobs):
    """blocks: {(group, target): [unit cell dicts]} each with per-method design.  Returns rows (one per held-out TF x method).  B per block by loto_B."""
    _L.clear()
    meta = {}
    for (grp, target), cells in blocks.items():
        Bk = loto_B(len(cells), B_loto)
        for m in cells[0]["methods"]:
            units = []
            for c in cells:
                rows = c["rows"]
                X0, X1 = design(c["P"][m], c["S"][m], rows)
                units.append(dict(X0=X0, X1=X1, y=np.asarray(c["y"], bool)[rows], fold=dr.fold_module(c["gi"])[rows], gi=c["gi"][rows]))
            _L[(grp, target, m)] = dict(units=units, B=Bk)
            meta[(grp, target, m)] = [c["base"] for c in cells]
    _L["__counts__"] = dr.mods[PT.PRIMARY_SIZE].counts[:max(1, int(B_loto))]
    keys = [k for k in _L if k != "__counts__"]
    if n_jobs > 1 and len(keys) > 1:
        with mp.get_context("fork").Pool(min(n_jobs, len(keys))) as pool:
            res = dict(pool.map(_loto_task, keys, chunksize=1))
    else:
        res = dict(_loto_task(k) for k in keys)
    out = []
    for key, bases in meta.items():
        r, Bk = res[key], _L[key]["B"]
        for t, base in enumerate(bases):
            row = dict(base, method=key[2], scheme="LOTO x module-fold logistic, refit inside module-block resampling (DIAGNOSTIC)", B_refit=int(Bk), n_tfs_train=len(bases) - 1,
                       n_convergence_warnings=int(r["nconv"]))
            if r["pt"] is None:
                out.append(dict(row, status_logit="NO_CALL", low_valid_draws=True))
                continue
            row["status_logit"] = "OK"
            for k in MET:
                p0, p1 = r["pt"][t][0][k], r["pt"][t][1][k]
                d0 = np.array([d[t][0][k] if d else np.nan for d in r["draws"]], float)
                d1 = np.array([d[t][1][k] if d else np.nan for d in r["draws"]], float)
                row[f"M0logit_{k}"], row[f"M1logit_{k}"], row[f"dlogit_{k}"] = p0, p1, p1 - p0
                row[f"dlogit_{k}_lo"], row[f"dlogit_{k}_hi"], row[f"p_dlogit_{k}"] = (*PT.pct_ci(d1 - d0), PT.boot_p(d1 - d0))
                if k == "AUROC":
                    row["n_valid_draws"] = int(np.isfinite(d0 + d1).sum())
            row["frac_valid_draws"] = (row["n_valid_draws"] / Bk) if Bk else np.nan
            row["low_valid_draws"] = bool(Bk == 0 or row["frac_valid_draws"] < MIN_VALID_FRAC)      # Bk == 0: point estimate only (no interval) -> flagged as such
            out.append(row)
    _L.clear()
    return out


# ===================================================================================================== parity against the audit
def parity_cells(u, target, rows, y, prim_rows):
    """S / P AUROC, AP, AP-lift of every (method, target) vs audit_per_tf (lattice or regulon); 1e-9"""
    _ctl, aud = PT._audit_control()
    a = aud[(aud.system == u["system"]) & (aud.tf == u["tf"]) & (aud.kind == u["kind"]) & (aud.target == target) & aud.status.isin(["OK", "DESCRIPTIVE_ONLY"])]
    out = dict(n=0, worst=0.0)
    if not len(a):
        return dict(out, status="NO_AUDIT_REFERENCE")
    mine = {r["method"]: r for r in prim_rows if "S_AUROC" in r}
    for _, q in a.iterrows():
        r = mine.get(q.method)
        if r is None:
            continue
        for ours, theirs in (("S_AUROC", "AUROC_full"), ("P_AUROC", "AUROC_prior"), ("S_AUPRC", "AP_full"), ("P_AUPRC", "AP_prior"), ("S_APLIFT", "APL_full"), ("P_APLIFT", "APL_prior")):
            dlt = abs(float(r[ours]) - float(q[theirs]))
            out["worst"], out["n"] = max(out["worst"], dlt), out["n"] + 1
            if dlt > 1e-9:
                raise SystemExit(f"REFUSING_PARITY {u['system']}/{u['tf']}/{target}/{q.method}/{ours}: {r[ours]!r} vs audit {q[theirs]!r}")
    return dict(out, status="PASS")


# ===================================================================================================== run
def unit_cells(u):
    """[(target, y, rows, gate)] of one unit (the audit's rows)"""
    import gb_layer_lattice as L
    if u["kind"] == "lattice":
        T = L.targets(u["ep"], u["M1"])
        return [(t, T[t][0], T[t][1], bool(L.gate_of(u["ep"], T[t][0], T[t][1], u["M1"]))) for t in LAT_TARGETS if t in T]
    yr = u["y"][u["M1"]]
    return [("regulon", u["y"], u["M1"], bool(yr.sum() >= PT.MIN_CLASS and (~yr).sum() >= PT.MIN_CLASS))]


def run_decoys(u, dr, a, jobs, keeps, rows_by, store, dec_rows):
    """decoy-calibrated increment for every gated (target, method) of ONE unit.  jobs = {(target, m): (y, rows)}; keeps[(target)] = cell_primary's keep; rows_by[(target, m)] = the cells-table row (mutated)."""
    ctx = unit_ctx(u)
    pert = perturbed_names(u)
    _D.clear()
    _D["__C__"], _D["__B__"] = dr.C, a.B
    meta = {}
    for (target, m), (y, rows) in jobs.items():
        d1, d2, tol = bank_cognate_parity(u, ctx, m, rows)
        if not (d1 <= tol and d2 <= 1e-9):
            raise SystemExit(f"REFUSING_PARITY_BANK_COGNATE {u['system']}/{u['tf']}/{target}/{m}: |bank row - cognate| = {d1:.3e} (tol {tol:.1e}), |bank prior - cognate prior| = {d2:.3e}")
        dd = prepare_decoys(u, ctx, m, rows, a.n_decoy)
        meta[(target, m)] = dict(dd, parity_bank_score=d1, parity_bank_prior=d2)
        if dd["status"].startswith("NO_CALL"):
            continue
        _D[(target, m)] = dict(n=len(u["genes"]), cols=dd["cols"], rows=rows, y=np.asarray(y, bool), gi=u["gi"], Sd=dd["Sd"], Pd=dd["Pd"])
    keys = [k for k in _D if k not in ("__C__", "__B__")]
    if a.n_jobs > 1 and len(keys) > 1:
        with mp.get_context("fork").Pool(min(a.n_jobs, len(keys))) as pool:
            res = dict(pool.map(_decoy_task, keys, chunksize=1))
    else:
        res = dict(_decoy_task(k) for k in keys)
    bank = u["bank"]
    for (target, m), mt in meta.items():
        uid = f"{u['kind']}|{u['system']}|{u['tf']}|{target}"
        row = rows_by[(target, m)]
        base = dict(decoy_status=mt["status"], n_decoy=int(len(mt["sel"])) if not mt["status"].startswith("NO_CALL") else 0, decoy_n_eligible=mt["n_eligible"],
                    decoy_parity_bank_score=mt["parity_bank_score"], decoy_parity_bank_prior=mt["parity_bank_prior"])
        if mt["status"].startswith("NO_CALL"):
            row.update(base)
            continue
        rs = res[(target, m)]
        names = [bank.members[int(s)] for s in mt["sel"]]
        isp = np.array([(bank.mu[int(s)] in pert) if pert else False for s in mt["sel"]], bool)
        stats = {k: decoy_stats(keeps[target][(m, k)]["pt"], keeps[target][(m, k)]["dg"], keeps[target][(m, k)]["dm"], rs["pt"][k], rs["dg"][k], rs["dm"][k]) for k in MET}
        nop = None
        if pert and (~isp).sum() >= MIN_DECOY:
            nop = {k: decoy_stats(keeps[target][(m, k)]["pt"], keeps[target][(m, k)]["dg"], keeps[target][(m, k)]["dm"], rs["pt"][k][~isp], rs["dg"][k][~isp], rs["dm"][k][~isp]) for k in MET}
        row.update(decoy_columns(stats, dict(base, decoy_mean_dist=float(np.mean(mt["dist"])), decoy_n_perturbed=int(isp.sum()), decoy_sha=zlib.crc32("|".join(names).encode()) & 0xFFFFFFFF), nop))
        if nop is not None:
            row["n_decoy_noPert"] = int((~isp).sum())
        for k in MET:
            store[f"{uid}|{m}|{k}|DM|g"], store[f"{uid}|{m}|{k}|DM|m"] = stats[k]["dm_g"].astype(np.float32), stats[k]["dm_m"].astype(np.float32)
        for j, nm in enumerate(names):
            dec_rows.append(dict(kind=u["kind"], system=u["system"], tf=u["tf"], target=target, method=m, decoy=nm, match_rank=j, dist=float(mt["dist"][j]), is_perturbed=bool(isp[j]),
                                 **{f"d_{k}": float(rs["pt"][k][j]) for k in MET}))
    _D.clear()


def process_unit(u, dr, a, sec_cells, loto_blocks, draws, dec_rows=None):
    rows_out, par_tot = [], dict(n=0, worst=0.0)
    dec_rows = [] if dec_rows is None else dec_rows
    jobs, keeps, rows_by = {}, {}, {}
    for target, y, rows, gate in unit_cells(u):
        if getattr(a, "loto_only", False):                                       # LOTO-only stage: no primary / decoy / logistic work (those run in the sliced main stage)
            if gate and target in a.secondary and u["macro_group"]:
                loto_blocks.setdefault((u["macro_group"], target), []).append(dict(rows=rows, y=y, gi=u["gi"], methods=list(u["use"]), P=u["prior"], S=u["V"],
                                                                                   base=dict(kind=u["kind"], system=u["system"], block=u["block"], tf=u["tf"], target=target,
                                                                                             macro_group=u["macro_group"], n_rows=int(rows.sum()), gate=True)))
            continue
        pr, store, keep = cell_primary(u, target, y, rows, dr, a.B, gate)
        rows_out += pr
        draws.update(store)
        if not a.no_parity and store:
            p = parity_cells(u, target, rows, y, pr)
            par_tot = dict(n=par_tot["n"] + p["n"], worst=max(par_tot["worst"], p["worst"]))
        if store and gate and not a.no_decoy:
            keeps[target] = keep
            for r_ in pr:
                if r_["status"] == "OK":
                    jobs[(target, r_["method"])] = (y, rows)
                    rows_by[(target, r_["method"])] = r_
        elif store and not gate:
            for r_ in pr:
                r_["decoy_status"] = "NOT_RUN_UNGATED"
        if store and target in a.secondary and gate:
            for m in u["use"]:
                base = dict(kind=u["kind"], system=u["system"], block=u["block"], tf=u["tf"], target=target, macro_group=u["macro_group"], method=m, n_rows=int(rows.sum()),
                            n_pos=int(np.asarray(y, bool)[rows].sum()), gate=True)
                sec_cells.append(dict(key=(f"{u['kind']}|{u['system']}|{u['tf']}|{target}", m), base=base, P=u["prior"][m], S=u["V"][m], y=y, rows=rows, gi=u["gi"]))
            if a.loto and u["macro_group"]:
                loto_blocks.setdefault((u["macro_group"], target), []).append(dict(rows=rows, y=y, gi=u["gi"], methods=list(u["use"]), P=u["prior"], S=u["V"],
                                                                                   base=dict(kind=u["kind"], system=u["system"], block=u["block"], tf=u["tf"], target=target,
                                                                                             macro_group=u["macro_group"], n_rows=int(rows.sum()), gate=True)))
    if jobs:
        run_decoys(u, dr, a, jobs, keeps, rows_by, draws, dec_rows)
    return rows_out, par_tot


def run_system(a, system, kind, out):
    t0 = time.time()
    rows_all, draws, sec_cells, loto_blocks, metas, dec_rows = [], {}, [], {}, {}, []
    dr_used = None
    if kind == "lattice":
        import gb_audit_decomp as AD
        import gb_layer_lattice as L
        import gb_lattice_extra as LE
        LE.register()
        dr, mrc = PT.lattice_draws(system, a.B, sizes=SIZES2)
        allowed = AD.tf_list_audit(system, L, LE)
        tfs = [t.strip() for t in a.tfs.split(",") if t.strip()] or allowed
        bad = [t for t in tfs if t not in allowed]
        if bad:
            raise SystemExit(f"REFUSING: --tfs {bad} not in the audit roster {allowed} of {system}")
        if a.nslice > 1:
            tfs = [t for i, t in enumerate(tfs) if i % a.nslice == a.slice]
        dr_used = dr
        for tf in tfs:
            t1 = time.time()
            try:
                u = PT.lattice_unit(system, tf, dr.gene)
            except (L.Blocked, SystemExit) as e:
                metas[tf] = dict(status=f"NOT_RUN: {e}")
                print(f"[prior-adj] {system}/{tf}: NOT_RUN {e}", flush=True)
                if isinstance(e, SystemExit) and "REFUSING_PARITY" in str(e):
                    raise
                continue
            r_, par = process_unit(u, dr, a, sec_cells, loto_blocks, draws, dec_rows)
            rows_all += r_
            metas[tf] = dict(n_M1=int(u["M1"].sum()), parity=par, gate_resp=bool(u["gate"]["MR"]), files=u["files"])
            print(f"[prior-adj] {system}/{tf}: M1={int(u['M1'].sum())} cells={len({(r['target']) for r in r_})} parity worst={par['worst']:.2e}/{par['n']} {time.time() - t1:.0f}s", flush=True)
    else:
        mrc = None
        for u, dr in PT.regulon_units(system, a.B, a.slice, a.nslice, set(t.strip() for t in a.reg_tfs.split(",") if t.strip()) or None, sizes=SIZES2):
            dr_used = dr
            if u.get("status", "").startswith("NOT_RUN"):
                metas[u["tf"]] = dict(status=u["status"])
                print(f"[prior-adj-reg] {system}/{u['tf']}: {u['status']}", flush=True)
                continue
            t1 = time.time()
            r_, par = process_unit(u, dr, a, sec_cells, loto_blocks, draws, dec_rows)
            rows_all += r_
            metas[u["tf"]] = dict(n_M1=int(u["M1"].sum()), parity=par)
            print(f"[prior-adj-reg] {system}/{u['tf']}: M1={int(u['M1'].sum())} pos={int(u['y'][u['M1']].sum())} parity worst={par['worst']:.2e}/{par['n']} {time.time() - t1:.0f}s", flush=True)
        if dr_used is not None:
            mrc = dr_used.module_receipts[PT.PRIMARY_SIZE]
    sec_rows, loto_out = [], []
    if sec_cells and dr_used is not None:
        t2 = time.time()
        sec_rows = secondary_rows(sec_cells, dr_used, min(a.B_secondary, a.B), a.n_jobs)
        nlow = sum(bool(r.get("low_valid_draws")) for r in sec_rows)
        print(f"[prior-adj] {system}: secondary logistic {len(sec_cells)} cells x refit B={min(a.B_secondary, a.B)} ({nlow} flagged < {MIN_VALID_FRAC:.0%} valid draws) {time.time() - t2:.0f}s", flush=True)
    if loto_blocks and dr_used is not None:
        blocks = {k: v for k, v in loto_blocks.items() if len(v) >= 3}
        if blocks:
            t2 = time.time()
            loto_out = loto_rows(blocks, dr_used, min(a.B_loto, a.B), a.n_jobs)
            print(f"[prior-adj] {system}: LOTO {len(blocks)} blocks (units {sorted(len(v) for v in blocks.values())}) B_loto={a.B_loto} {time.time() - t2:.0f}s", flush=True)
    sel = (a.tfs if kind == "lattice" else a.reg_tfs).replace(",", "_").replace(" ", "")
    tag = (sel or "all") + (f"_s{a.slice}of{a.nslice}" if a.nslice > 1 else "") + ("_loto" if a.loto_only else "")
    stem = f"part_{'lat' if kind == 'lattice' else 'reg'}_{system}_{tag}"
    if rows_all:
        PT.atomic_tsv(pd.DataFrame(rows_all), f"{out}/{stem}.cells.tsv")
        tmp = f"{out}/{stem}.draws.tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, **draws)
        os.replace(tmp, f"{out}/{stem}.draws.npz")
    if dec_rows:
        PT.atomic_tsv(pd.DataFrame(dec_rows), f"{out}/{stem}.decoys.tsv")
    if sec_rows or loto_out:
        PT.atomic_tsv(pd.DataFrame(sec_rows + loto_out), f"{out}/{stem}.logit.tsv")
    ncal = sum(1 for r in rows_all if str(r.get("decoy_status", "")).startswith("OK"))
    PT.atomic_json(f"{out}/{stem}.meta.json", dict(stem=stem, kind=kind, system=system, B=a.B, B_secondary=min(a.B_secondary, a.B), B_loto=min(a.B_loto, a.B), n_decoy=a.n_decoy, no_decoy=bool(a.no_decoy),
                                                   loto_only=bool(a.loto_only), script_sha256=SCRIPT_SHA0, ptable_sha256=PT.SCRIPT_SHA0,
                                                   module_receipt=(None if mrc is None else {k: mrc[k] for k in ("rule_sha256", "module_file_sha256", "created", "n_modules", "n_singletons")}),
                                                   tfs=metas, secondary_targets=list(a.secondary), seconds=round(time.time() - t0, 1), audit_artifacts=dict(audit_per_tf=PT._CACHE.get("aud_sha")),
                                                   n_cells_decoy_calibrated=ncal, n_cells_decoy_nocall=sum(1 for r in rows_all if str(r.get("decoy_status", "")).startswith("NO_CALL")),
                                                   dep_shas=PT.dep_shas(), draws_fp=(dr_used.fp if dr_used is not None else None), n_missing_in_map=(dr_used.n_missing_in_map if dr_used is not None else None),
                                                   seeds=(dict(gene=dr_used.gene.seed, module=dr_used.mods[PT.PRIMARY_SIZE].seed) if dr_used is not None else None),
                                                   python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, host=PT.platform.node(), created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def cmd_run(a):
    os.makedirs(a.out, exist_ok=True)
    a.secondary = [s.strip() for s in a.secondary.split(",") if s.strip()]
    if a.loto_only:
        a.loto = True
    systems = [s.strip() for s in a.systems.split(",") if s.strip()]
    regs = [s.strip() for s in a.regulon.split(",") if s.strip()]
    PT.guard_systems(systems + regs)
    for s in systems:
        if s in PT.LATTICE_DUP:
            raise SystemExit(f"REFUSING: {s} duplicates {PT.LATTICE_DUP[s]} (dedup)")
        if s not in PT.LATTICE_PRIMARY + PT.LATTICE_SENS:
            raise SystemExit(f"REFUSING: {s} not in {PT.LATTICE_PRIMARY + PT.LATTICE_SENS}")
    for s in regs:
        if s not in PT.REGULON_SYSTEMS:
            raise SystemExit(f"REFUSING: regulon system {s} not in {PT.REGULON_SYSTEMS}")
    if a.loto_only and a.nslice > 1:
        raise SystemExit("REFUSING: --loto-only needs every unit of a block in one process (no --nslice)")
    for s in systems:
        run_system(a, s, "lattice", a.out)
    for s in regs:
        run_system(a, s, "regulon", a.out)
    assert PT._sha(__file__) == SCRIPT_SHA0, "script changed during the run"


# ===================================================================================================== merge: macro + ranking change
def _stack(draws, units, target, m, metric, nm, sch):
    return np.mean([draws[f"{k}|{s}|{t}|{target}|{m}|{metric}|{nm}|{sch}"] for k, s, t in units], 0)


def _have(draws, units, target, m, metric, nm):
    return all(f"{k}|{s}|{t}|{target}|{m}|{metric}|{nm}|g" in draws for k, s, t in units)


def macro_and_rank(cells, draws, B):
    """per (group, target[, unit]) rank-change rows for BOTH increments (delta_kind raw | calibrated).  Units: each gated cell on its own (draws of that cell) and each macro group (mean over its
    gated cells per draw).  The calibrated ranking uses only the methods whose decoy calibration exists for EVERY unit of the group (missing = excluded, never zero)."""
    rk, macro_rows = [], []
    ok = cells[cells.status == "OK"]

    def one(units, kind, grp, target, metric):
        methods = [m for m in METHODS if all(_have(draws, units, target, m, metric, nm) for nm in ("S", "P", "M1"))]
        if len(methods) < 3 or GB not in methods:
            return None
        sub = ok[(ok.target == target)]
        mask = np.zeros(len(sub), bool)
        for k, s, t in units:
            mask |= ((sub.kind == k) & (sub.system == s) & (sub.tf == t)).to_numpy()
        sub = sub[mask]
        if "draws_fp" in sub and sub.draws_fp.nunique() != 1:                      # per-draw means over units are only valid when every unit used the SAME resamples
            raise SystemExit(f"REFUSING_MACRO {grp}/{target}: units use different resample draws {sorted(sub.draws_fp.unique())}")

        def pt_mean(m, col):
            v = sub[sub.method == m][col].to_numpy(float) if col in sub else np.array([])
            return float(v.mean()) if len(v) == len(units) and np.isfinite(v).all() else np.nan
        pt = {nm: {m: pt_mean(m, f"{nm}_{metric}") for m in methods} for nm in ("S", "P", "M1")}
        res = {}
        for sch in ("g", "m"):
            Sd = {m: _stack(draws, units, target, m, metric, "S", sch) for m in methods}
            Pd = {m: _stack(draws, units, target, m, metric, "P", sch) for m in methods}
            Md = {m: _stack(draws, units, target, m, metric, "M1", sch) for m in methods}
            res[sch] = (Sd, Pd, Md)
        # ---- the two increments: raw (all methods) and calibrated (methods with decoys for every unit)
        kinds = {"raw": dict(methods=methods, pt={m: pt["M1"][m] - pt["P"][m] for m in methods},
                             dd={sch: {m: res[sch][2][m] - res[sch][1][m] for m in methods} for sch in ("g", "m")})}
        mc = [m for m in methods if _have(draws, units, target, m, metric, "DM") and np.isfinite(pt_mean(m, f"calD_{metric}"))]
        if len(mc) >= 3 and GB in mc:
            kinds["calibrated"] = dict(methods=mc, pt={m: pt_mean(m, f"calD_{metric}") for m in mc},
                                       dd={sch: {m: res[sch][2][m] - res[sch][1][m] - _stack(draws, units, target, m, metric, "DM", sch) for m in mc} for sch in ("g", "m")})
        for dk, info in kinds.items():
            ms = info["methods"]
            S_pt = {m: pt["S"][m] for m in ms}
            P_pt = {m: pt["P"][m] for m in ms}
            r_pt = rank_change(S_pt, info["pt"], ms)
            rP_pt = rank_change(P_pt, info["pt"], ms)
            row = dict(delta_kind=dk, kind=kind, group=grp, target=target, metric=metric, n_units=len(units), n_methods=len(ms), tau_b=float(r_pt["tau"][0]), tau_vsP=float(rP_pt["tau"][0]),
                       rank_GB_raw=float(r_pt["rank_gb_raw"][0]), rank_GB_incr=float(r_pt["rank_gb_incr"][0]), winner_raw=ms[int(r_pt["winner_raw"][0])] if r_pt["ok"][0] else "",
                       winner_incr=ms[int(r_pt["winner_incr"][0])] if r_pt["ok"][0] else "")
            for sch, sfx in (("g", ""), ("m", "_mod")):
                Sd = {m: res[sch][0][m] for m in ms}
                Pd = {m: res[sch][1][m] for m in ms}
                Dd = info["dd"][sch]
                rc, rcP = rank_change(Sd, Dd, ms), rank_change(Pd, Dd, ms)
                okb = rc["ok"]
                row[f"tau_b{sfx}_lo"], row[f"tau_b{sfx}_hi"] = PT.pct_ci(rc["tau"][okb])
                row[f"tau_vsP{sfx}_lo"], row[f"tau_vsP{sfx}_hi"] = PT.pct_ci(rcP["tau"][rcP["ok"]])
                row[f"rank_GB_raw{sfx}_lo"], row[f"rank_GB_raw{sfx}_hi"] = PT.pct_ci(rc["rank_gb_raw"][okb])
                row[f"rank_GB_incr{sfx}_lo"], row[f"rank_GB_incr{sfx}_hi"] = PT.pct_ci(rc["rank_gb_incr"][okb])
                row[f"frac_draws_winner_changes{sfx}"] = float((rc["winner_raw"][okb] != rc["winner_incr"][okb]).mean()) if okb.any() else np.nan
                row[f"frac_draws_GB_rank_changes{sfx}"] = float((rc["rank_gb_raw"][okb] != rc["rank_gb_incr"][okb]).mean()) if okb.any() else np.nan
                row[f"n_valid_draws{sfx}"] = int(okb.sum())
            rk.append(row)
        if kind == "macro":
            for m in methods:
                mr = dict(group=grp, target=target, metric=metric, method=m, n_units=len(units), S=pt["S"][m], P=pt["P"][m], M1=pt["M1"][m], d=pt["M1"][m] - pt["P"][m])
                for sch, sfx in (("g", ""), ("m", "_mod")):
                    Sd, Pd, Md = res[sch]
                    for nm, dd in (("S", Sd[m]), ("P", Pd[m]), ("M1", Md[m]), ("d", Md[m] - Pd[m])):
                        mr[f"{nm}{sfx}_lo"], mr[f"{nm}{sfx}_hi"] = PT.pct_ci(dd)
                    mr[f"p_d{sfx}"] = PT.boot_p(Md[m] - Pd[m])
                if "calibrated" in kinds and m in kinds["calibrated"]["methods"]:
                    mr["calD"] = kinds["calibrated"]["pt"][m]
                    for sch, sfx in (("g", ""), ("m", "_mod")):
                        cd = kinds["calibrated"]["dd"][sch][m]
                        mr[f"calD{sfx}_lo"], mr[f"calD{sfx}_hi"] = PT.pct_ci(cd)
                        mr[f"p_calD{sfx}"] = PT.boot_p(cd)
                    mm_ = sub[sub.method == m]
                    mr["mean_cog_pct"] = float(mm_[f"cog_pct_{metric}"].mean()) if f"cog_pct_{metric}" in mm_ else np.nan
                    mr["frac_units_p_pool_le_05"] = float((mm_[f"p_pool_{metric}"] <= 0.05).mean()) if f"p_pool_{metric}" in mm_ else np.nan
                macro_rows.append(mr)
        return True

    for target in sorted(ok.target.unique()):
        t = ok[ok.target == target]
        for metric in ("AUROC", "AUPRC", "P50"):
            for (k, s, tf), _g in t.groupby(["kind", "system", "tf"]):
                one([(k, s, tf)], "unit", f"{s}/{tf}", target, metric)
            for grp, g in t.groupby("macro_group"):
                if not grp:
                    continue
                units = sorted(set(zip(g.kind, g.system, g.tf)))
                if len(units) >= 2 or str(grp).startswith("REGULON"):
                    one(units, "macro", grp, target, metric)
    return pd.DataFrame(rk), pd.DataFrame(macro_rows)


def _read(f, **kw):
    try:
        return pd.read_csv(f, sep="\t", **kw)
    except (pd.errors.EmptyDataError, FileNotFoundError):                    # an empty / absent table (e.g. no macro group in a single-unit run) must never crash merge / report
        return pd.DataFrame()


def cmd_merge(a):
    metas = [json.load(open(f)) for f in sorted(glob.glob(f"{a.out}/part_*.meta.json"))]
    if not metas:
        raise SystemExit("REFUSING: no part files")
    if len({m["script_sha256"] for m in metas}) != 1 or len({int(m["B"]) for m in metas}) != 1 or len({m["ptable_sha256"] for m in metas}) != 1 or len({int(m["n_decoy"]) for m in metas}) != 1:
        raise SystemExit("REFUSING_MIXED_PARTS: script shas / B / n_decoy differ")
    B = int(metas[0]["B"])
    cells, logit, decoys, draws = [], [], [], {}
    for m in metas:
        f = f"{a.out}/{m['stem']}.cells.tsv"
        if os.path.exists(f):
            cells.append(pd.read_csv(f, sep="\t", dtype={"tf": str}))
            with np.load(f.replace(".cells.tsv", ".draws.npz")) as z:
                for k in z.files:
                    if k in draws:
                        raise SystemExit(f"REFUSING_DUPLICATE_DRAWS {k}")
                    draws[k] = z[k]
        for suf, lst in ((".logit.tsv", logit), (".decoys.tsv", decoys)):
            g = f"{a.out}/{m['stem']}{suf}"
            if os.path.exists(g):
                lst.append(pd.read_csv(g, sep="\t", dtype={"tf": str}))
    cells = pd.concat(cells, ignore_index=True)
    dup = cells.duplicated(["kind", "system", "tf", "target", "method"]).sum()
    if dup:
        raise SystemExit(f"REFUSING_DUPLICATE_CELLS {dup} (kind, system, tf, target, method) rows appear in more than one part")
    PT.atomic_tsv(cells, f"{a.out}/table2_cells.tsv")
    if logit:
        lg = pd.concat(logit, ignore_index=True)
        PT.atomic_tsv(lg, f"{a.out}/table2_logit_diagnostic.tsv")
    if decoys:
        PT.atomic_tsv(pd.concat(decoys, ignore_index=True), f"{a.out}/table2_decoys.tsv")
    rk, mac = macro_and_rank(cells, draws, B)
    PT.atomic_tsv(rk, f"{a.out}/table2_rankchange.tsv")
    dsum = []
    for level, df in (("unit", cells[cells.status == "OK"]),):
        for (kind, target), g in df.groupby(["kind", "target"]):
            for k in MET:
                for quantity, pre in (("raw_delta", f"d_{k}"), ("calibrated_delta", f"calD_{k}")):
                    cols = [f"{pre}_lo", f"{pre}_hi", f"{pre}_mod_lo", f"{pre}_mod_hi"]
                    if not all(c in g for c in cols):
                        continue
                    gg = g.dropna(subset=cols)
                    if not len(gg):
                        continue
                    lo, hi, mlo, mhi = (gg[c].to_numpy(float) for c in cols)
                    sg, sm = (lo > 0) | (hi < 0), (mlo > 0) | (mhi < 0)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        wr = np.nanmedian((mhi - mlo) / (hi - lo))
                    dsum.append(dict(level=level, quantity=quantity, kind=kind, target=target, metric=k, n_contrasts=int(len(gg)), n_sig_gene=int(sg.sum()), n_sig_module=int(sm.sum()),
                                     n_lost_under_module=int((sg & ~sm).sum()), n_gained_under_module=int((~sg & sm).sum()), median_ci_width_ratio_module_over_gene=float(wr)))
    PT.atomic_tsv(pd.DataFrame(dsum), f"{a.out}/table2_dependence_summary.tsv")
    PT.atomic_tsv(mac, f"{a.out}/table2_macro_methods.tsv")
    notrun = {f"{m['system']}|{k}": v["status"] for m in metas for k, v in m["tfs"].items() if str(v.get("status", "")).startswith("NOT_RUN")}
    dstat = {}
    if "decoy_status" in cells:
        dstat = {str(k): int(v) for k, v in cells[cells.status == "OK"].decoy_status.fillna("NA").str.slice(0, 40).value_counts().items()}
    nlow = int(pd.concat(logit).low_valid_draws.fillna(False).astype(bool).sum()) if logit else 0
    dedup = PT.dedup_report()
    PT.atomic_json(f"{a.out}/receipt.json", dict(script_sha256=SCRIPT_SHA0, ptable_sha256=PT.SCRIPT_SHA0, B=B, n_decoy=int(metas[0]["n_decoy"]), min_decoy=MIN_DECOY, min_valid_frac=MIN_VALID_FRAC,
                                                 parts=[m["stem"] for m in metas], not_run=notrun, dropped_duplicates=PT.LATTICE_DUP, dedup_check=dedup, decoy_status_counts_gated_cells=dstat,
                                                 n_logit_rows_low_valid_draws=nlow, module_rule_sha256=PT.MODULE_RULE_SHA, python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                                                 created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                                 labels=dict(primary="gene-identity resampling (synchronized BlockAxis draws)", sensitivity="module-block resampling (frozen control-only modules, G/50)",
                                                             calibrated="decoy-calibrated increment = cognate Delta - median Delta of the 50 matched non-cognate sources of the method's own bank (quantity to interpret)",
                                                             raw="raw increment Delta (descriptive; confounded by the method's own prior quality)",
                                                             secondary="DIAGNOSTIC supervised logistic (module-fold, refit inside module-block resampling)")))
    print(f"[prior-adj merge] cells={len(cells)} rank rows={len(rk)} macro method rows={len(mac)} logit rows={sum(len(x) for x in logit)} (low-valid {nlow}) decoy rows={sum(len(x) for x in decoys)} not_run={len(notrun)} decoy status={dstat}")


def _f(x, d=3):
    return "  NA " if x is None or not np.isfinite(x) else f"{x:.{d}f}"


def cmd_report(a):
    """compact print of Table 2: per unit (SYSTEM/TF) and macro group, one block per target: Perf(S), Perf(P), Perf(M1), raw Delta and DECOY-CALIBRATED Delta (AUROC) with the gene-identity and
    module-block intervals, the cognate decoy percentile / pool p, the logistic diagnostic when present, and the tau-b rows for both increments"""
    cells = _read(f"{a.out}/table2_cells.tsv", dtype={"tf": str})
    rk = _read(f"{a.out}/table2_rankchange.tsv")
    mac = _read(f"{a.out}/table2_macro_methods.tsv")
    lg = _read(f"{a.out}/table2_logit_diagnostic.tsv", dtype={"tf": str})
    want = [x.strip() for x in a.groups.split(",") if x.strip()]
    for grp in want:
        for target in [t.strip() for t in a.targets.split(",") if t.strip()]:
            if grp.startswith("MACRO:"):
                g = mac[(mac.group == grp[6:]) & (mac.target == target)] if len(mac) else mac
                if not len(g):
                    continue
                A_ = g[g.metric == "AUROC"].set_index("method")
                print(f"\n== {grp} target={target} units={int(A_.n_units.iloc[0])}  (macro over gated units; gene-identity CI | module-block CI)")
                print(f"{'method':14s} {'S':>6s} {'P':>6s} {'M1':>6s} | {'raw dAUROC [gene CI]':26s} {'[module CI]':16s} | {'calibrated dAUROC [gene CI]':28s} {'[module CI]':16s} {'p':>6s} {'pct':>5s}")
                for m_, r in A_.iterrows():
                    cal = (f"{r.calD:+.3f} [{r.calD_lo:+.3f},{r.calD_hi:+.3f}]   [{r.calD_mod_lo:+.3f},{r.calD_mod_hi:+.3f}] {_f(r.p_calD, 3):>6s} {_f(r.mean_cog_pct, 2):>5s}"
                           if "calD" in r and np.isfinite(r.get("calD", np.nan)) else "      no calibration")
                    print(f"{m_:14s} {_f(r.S):>6s} {_f(r.P):>6s} {_f(r.M1):>6s} | {r.d:+.3f} [{r.d_lo:+.3f},{r.d_hi:+.3f}]   [{r.d_mod_lo:+.3f},{r.d_mod_hi:+.3f}] | {cal}")
                for met in ("AUROC", "AUPRC", "P50"):
                    q = rk[(rk.kind == "macro") & (rk.group == grp[6:]) & (rk.target == target) & (rk.metric == met)]
                    for _, r in q.iterrows():
                        print(f"   tau-b[{met}] {r.delta_kind:10s} raw-vs-incremental = {r.tau_b:+.3f} [{r.tau_b_lo:+.3f},{r.tau_b_hi:+.3f}] (module {r.tau_b_mod_lo:+.3f},{r.tau_b_mod_hi:+.3f}); vs prior-quality "
                              f"{r.tau_vsP:+.3f}; GB rank {r.rank_GB_raw:.0f} -> {r.rank_GB_incr:.0f}; winner {r.winner_raw} -> {r.winner_incr} (changes in {100 * r.frac_draws_winner_changes:.0f}% of draws)")
            else:
                sysn, _, tf = grp.partition("/")
                g = cells[(cells.system == sysn) & (cells.tf == tf) & (cells.target == target)] if len(cells) else cells
                if not len(g):
                    continue
                r0 = g.iloc[0]
                print(f"\n== {grp} target={target} n={int(r0.n_rows)} pos={int(r0.n_pos)} prev={r0.PREV:.3f} gate={r0.gate} status={r0.status}")
                print(f"{'method':14s} {'rhoSP':>6s} {'S':>6s} {'P':>6s} {'M1':>6s} | {'raw dAUROC [gene CI]':26s} {'[module CI]':16s} {'p':>6s} | {'calibrated dAUROC [gene CI]':28s} {'pct':>5s} {'p_pool':>6s} {'nDec':>4s}", end="")
                L_ = lg[(lg.system == sysn) & (lg.tf == tf) & (lg.target == target) & lg.scheme.fillna("").str.startswith("module")] if len(lg) else lg
                print(" | dlogitAUROC [module-block refit CI]" if len(L_) else "")
                for _, r in g.iterrows():
                    cal = (f"{r.calD_AUROC:+.3f} [{r.calD_AUROC_lo:+.3f},{r.calD_AUROC_hi:+.3f}] {_f(r.cog_pct_AUROC, 2):>5s} {_f(r.p_pool_AUROC, 3):>6s} {int(r.n_decoy):>4d}"
                           if "calD_AUROC" in r and np.isfinite(r.get("calD_AUROC", np.nan)) else f"{'no calibration (' + str(r.get('decoy_status', 'NA'))[:20] + ')':<44s}")
                    line = (f"{r.method:14s} {_f(r.rho_SP):>6s} {_f(r.S_AUROC):>6s} {_f(r.P_AUROC):>6s} {_f(r.M1_AUROC):>6s} | {r.d_AUROC:+.3f} [{r.d_AUROC_lo:+.3f},{r.d_AUROC_hi:+.3f}]   "
                            f"[{r.d_AUROC_mod_lo:+.3f},{r.d_AUROC_mod_hi:+.3f}] {_f(r.p_AUROC, 3):>6s} | {cal}")
                    if len(L_):
                        q = L_[L_.method == r.method]
                        if len(q) and q.status_logit.iloc[0] == "OK":
                            q = q.iloc[0]
                            line += f" | {q.dlogit_AUROC:+.3f} [{q.dlogit_AUROC_lo:+.3f},{q.dlogit_AUROC_hi:+.3f}]" + (" LOW-VALID" if bool(q.get("low_valid_draws", False)) else "")
                    print(line)
                for met in ("AUROC", "AUPRC", "P50"):
                    q = rk[(rk.kind == "unit") & (rk.group == grp) & (rk.target == target) & (rk.metric == met)] if len(rk) else rk
                    for _, r in q.iterrows():
                        print(f"   tau-b[{met}] {r.delta_kind:10s} = {r.tau_b:+.3f} [{r.tau_b_lo:+.3f},{r.tau_b_hi:+.3f}]; vs prior-quality {r.tau_vsP:+.3f}; GB rank {r.rank_GB_raw:.0f} -> {r.rank_GB_incr:.0f}; "
                              f"winner {r.winner_raw} -> {r.winner_incr}")


# ===================================================================================================== selftest
def _fake_bank(rng, nb=60, G=300, nt=200, methods=("GeoBRIDGE", "GENIE3"), nan_rows=(5, 6), nan_cols=()):
    """a real gb_attribution.Bank (production prior algebra) on synthetic matrices: members = the first nb axis genes, T = the first nt axis genes"""
    import gb_attribution as A
    axis = [f"g{i}" for i in range(G)]
    members = axis[:nb]
    T = np.zeros(G, bool)
    T[:nt] = True
    X = {}
    for m in methods:
        x = rng.standard_normal((nb, G))
        x[list(nan_rows), :] = np.where(rng.random((len(nan_rows), G)) < 0.5, np.nan, x[list(nan_rows), :])
        X[m] = x
    return A.Bank(members, axis, T, X)


def selftest(a=None):
    from scipy.stats import rankdata
    import gb_canon_table as CT  # noqa: F401
    rng = np.random.default_rng(0)
    G = 400
    # 1. combine(): M0 == midrank(P); M1 == (midrank(P) + midrank(S)) / 2 incl. ties; NaN outside rows; identical genes
    P = np.round(rng.standard_normal(G), 1)
    S = np.round(rng.standard_normal(G), 1)
    rows = rng.random(G) < 0.7
    M0, M1 = combine(P, S, rows)
    assert np.array_equal(M0[rows], rankdata(P[rows])) and np.array_equal(M1[rows], (rankdata(P[rows]) + rankdata(S[rows])) / 2) and np.isnan(M0[~rows]).all() and np.isnan(M1[~rows]).all()
    # 2. identity: AUROC / AP / P50 of M0 equal those of P (ties preserved), point and draws (gene and module schemes of the [gene; module G/50] draw matrix)
    names = np.array([f"x{i}" for i in range(G)])
    units = np.array([f"M{i // 8:03d}" for i in range(G)])
    dr = PT.Draws(names, 40, 3, {50: 4}, {50: units})
    assert dr.C.shape == (80, G)
    y = rng.random(G) < 0.3
    rr = np.ones(G, bool)
    a_ = PT.split_eval(P, y, rr, np.arange(G), dr.C, 40)
    c0, _ = combine(P, S, rr)
    b_ = PT.split_eval(c0, y, rr, np.arange(G), dr.C, 40)
    for k in MET:
        assert abs(a_[0][k] - b_[0][k]) < 1e-12 or (np.isnan(a_[0][k]) and np.isnan(b_[0][k]))
        for i in (0, 1):
            assert np.allclose(a_[1][i][k], b_[1][i][k], atol=1e-12, equal_nan=True)
    # 3. S == P  =>  M1 == M0 exactly  =>  Delta == 0 in every draw
    m0, m1 = combine(P, P, rr)
    assert np.array_equal(m0, m1)
    # 4. planted signal: S carries label information beyond P => Delta(AUROC) > 0;  S pure noise => Delta <= ~0 (dilution) -- fixed seed
    n = 1500
    lat = rng.standard_normal(n)
    yy = (lat + rng.standard_normal(n)) > 0.8
    Pp = lat + rng.standard_normal(n) * 1.5
    S_sig = lat * 0.0 + (yy * 1.2 + rng.standard_normal(n))             # independent of P given labels, informative
    S_noise = rng.standard_normal(n)
    allr = np.ones(n, bool)
    from sklearn.metrics import roc_auc_score
    for Sx, sign in ((S_sig, +1), (S_noise, -1)):
        _m0, _m1 = combine(Pp, Sx, allr)
        d = roc_auc_score(yy, _m1) - roc_auc_score(yy, _m0)
        assert (d > 0.02) if sign > 0 else (d < 0.01), (sign, d)
    # 5. Kendall tau-b equals scipy, handles ties, permutation-invariant over methods, perfect / reversed
    x = np.round(rng.standard_normal(10), 1)
    z = np.round(x + rng.standard_normal(10) * 0.5, 1)
    from scipy.stats import kendalltau
    assert abs(tau_b(x, z) - kendalltau(x, z, variant="b")[0]) < 1e-15 and abs(tau_b(x, x) - 1) < 1e-12 and abs(tau_b(x, -x) + 1) < 1e-12
    perm = rng.permutation(10)
    assert abs(tau_b(x[perm], z[perm]) - tau_b(x, z)) < 1e-12
    Sd = {m: float(v) for m, v in zip(["GeoBRIDGE", "B", "C"], [0.6, 0.5, 0.4])}
    Dd = {m: float(v) for m, v in zip(["GeoBRIDGE", "B", "C"], [0.0, 0.2, 0.1])}
    rc = rank_change(Sd, Dd, ["GeoBRIDGE", "B", "C"])
    assert rc["rank_gb_raw"][0] == 1 and rc["rank_gb_incr"][0] == 3 and rc["winner_raw"][0] == 0 and rc["winner_incr"][0] == 1 and abs(rc["tau"][0] - (-1 / 3)) < 1e-12, rc
    # 6. weighted logistic == the same logistic on duplicated rows (integer multiplicities); weights 0 drop rows; refit really depends on the draw
    X = rng.standard_normal((60, 3))
    yb = (X[:, 0] + rng.standard_normal(60)) > 0
    w = rng.integers(0, 4, 60).astype(float)
    from sklearn.linear_model import LogisticRegression
    m_w = LogisticRegression(C=1.0, max_iter=5000).fit(X[w > 0], yb[w > 0], sample_weight=w[w > 0])
    idx = np.repeat(np.arange(60), w.astype(int))
    m_d = LogisticRegression(C=1.0, max_iter=5000).fit(X[idx], yb[idx])
    assert np.abs(m_w.coef_ - m_d.coef_).max() < 1e-4 and abs(m_w.intercept_[0] - m_d.intercept_[0]) < 1e-4, (m_w.coef_, m_d.coef_)
    # 7. oof_logit: genes of a fold are predicted by a model that never saw that fold (label perturbation in the test fold leaves its predictions unchanged); fold = module hash
    nn = 500
    Xd = rng.standard_normal((nn, 4))
    yd = (Xd[:, 0] + rng.standard_normal(nn)) > 0.2
    fold = dr.fold_module(np.arange(G))[:G]
    fold = np.resize(fold, nn)
    o1 = oof_logit(Xd, yd, fold, np.ones(nn))
    y2 = yd.copy()
    te = fold == 2
    y2[te] = ~y2[te]
    o2 = oof_logit(Xd, y2, fold, np.ones(nn))
    assert np.allclose(o1[te], o2[te]) and not np.allclose(o1[~te], o2[~te])
    # 8. wmetrics with integer multiplicities == the metric on the expanded sample; ones == plain
    sc = rng.standard_normal(80)
    yl = rng.random(80) < 0.4
    ww = rng.integers(0, 3, 80)
    e = wmetrics(sc, yl, ww.astype(float), 80)
    dup = np.repeat(np.arange(80), ww)
    e2 = wmetrics(sc[dup], yl[dup], np.ones(len(dup)), 80)
    for k in ("AUROC", "AUPRC"):
        assert abs(e[k] - e2[k]) < 1e-12
    # 9. design(): label-free (changing y cannot change X), 6 spline columns for P and 7 with S
    import gb_prior_semantics as PS  # noqa: F401
    rws = np.ones(300, bool)
    X0, X1 = design(rng.standard_normal(300), rng.standard_normal(300), rws)
    assert X0.shape == (300, 6) and X1.shape == (300, 7)
    # 10. secondary_rows end-to-end on a planted problem (serial, tiny B): informative S gives dlogit > 0, refit draws differ across draws, identical inputs give identical outputs; valid-draw flag
    Gs = 500
    names2 = np.array([f"y{i}" for i in range(Gs)])
    units2 = np.array([f"M{i // 10:03d}" for i in range(Gs)])
    dr2 = PT.Draws(names2, 12, 5, {50: 6}, {50: units2})
    lat2 = rng.standard_normal(Gs)
    y2_ = (lat2 + rng.standard_normal(Gs)) > 0.7
    P2 = lat2 + rng.standard_normal(Gs)
    S2 = y2_ * 1.5 + rng.standard_normal(Gs)
    cell = dict(key=("syn", "m"), base=dict(system="SYN"), P=P2, S=S2, y=y2_, rows=np.ones(Gs, bool), gi=np.arange(Gs))
    sr = secondary_rows([cell], dr2, 12, 1)[0]
    assert sr["status_logit"] == "OK" and sr["dlogit_AUROC"] > 0.05 and sr["n_valid_draws"] >= 10 and sr["low_valid_draws"] == (sr["n_valid_draws"] / 12 < MIN_VALID_FRAC) and "frac_valid_draws" in sr
    sr2 = secondary_rows([cell], dr2, 12, 1)[0]
    assert sr["dlogit_AUROC"] == sr2["dlogit_AUROC"] and sr["dlogit_AUROC_lo"] == sr2["dlogit_AUROC_lo"]
    # a cell whose positives are so rare that most resamples cannot be refit is FLAGGED (< 95% valid draws), never silently averaged over the survivors
    ys = np.zeros(Gs, bool)
    ys[rng.choice(Gs, 8, replace=False)] = True
    cell_rare = dict(key=("syn", "rare"), base=dict(system="SYN"), P=P2, S=S2, y=ys, rows=np.ones(Gs, bool), gi=np.arange(Gs))
    sr3 = secondary_rows([cell_rare], dr2, 12, 1)[0]
    assert sr3["status_logit"] != "OK" or sr3["low_valid_draws"], sr3
    # LOTO block-size rule
    assert [loto_B(n_, 100) for n_ in (3, 12, 13, 40, 41, 200)] == [100, 100, 30, 30, 0, 0] and loto_B(20, 10) == 10
    # 11. cell_primary end-to-end on a synthetic unit (10 methods): identity asserts pass, rows complete, ranking pipeline runs
    meths = list(METHODS)
    Gu = 400
    namesu = np.array([f"z{i}" for i in range(Gu)])
    unitsu = np.array([f"M{i // 8:03d}" for i in range(Gu)])
    dru = PT.Draws(namesu, 30, 7, {50: 8}, {50: unitsu})
    latu = rng.standard_normal(Gu)
    yu = (latu + rng.standard_normal(Gu)) > 0.5
    V = {m: latu * rng.uniform(0.2, 1.0) + rng.standard_normal(Gu) for m in meths}
    Pr = {m: latu * rng.uniform(0.2, 1.0) + rng.standard_normal(Gu) for m in meths}
    M1u = np.ones(Gu, bool)
    u = dict(kind="syn", system="SYN", tf="T", block="SYN", macro_group="SYN", genes=namesu, gi=np.arange(Gu), M1=M1u, use=meths, V=V, prior=Pr)
    rws, st, keep = cell_primary(u, "resp", yu, M1u, dru, 30, True)
    assert len(rws) == 10 and all(r["status"] == "OK" for r in rws) and all(k in st for k in [f"syn|SYN|T|resp|GeoBRIDGE|AUROC|{x}|{s}" for x in ("S", "P", "M1") for s in ("g", "m")])
    assert abs(keep[("GeoBRIDGE", "AUROC")]["pt"] - rws[0]["d_AUROC"]) < 1e-15 and keep[("GeoBRIDGE", "AUROC")]["dg"].shape == (30,)
    # ---- 12. decoy machinery on a REAL gb_attribution.Bank (production prior algebra) with synthetic matrices
    selftest_decoy(rng)
    print("[prior-adj] selftest (synthetic) PASS")
    if a is not None and getattr(a, "data", False):
        selftest_data(a)


def selftest_decoy(rng):
    """matching == the reference nearest-neighbour rule; eligibility; per-decoy increment == an independent brute force (scipy rank + sklearn AUROC on duplicated rows); calibrated statistics arithmetic"""
    from scipy.stats import rankdata
    from sklearn.metrics import roc_auc_score
    nb, G, nt = 60, 300, 200
    bank = _fake_bank(rng, nb=nb, G=G, nt=nt, methods=("GeoBRIDGE", "GENIE3"), nan_rows=(5, 6))
    # (a) pick_decoys == a plain re-statement of gb_defense_dutil.nearest_matched on random covariates; focal excluded; eligibility honoured; ties stable
    det, lms = rng.random(nb), rng.standard_normal(nb)
    Fm = np.column_stack([det, lms])
    Fm = (Fm - Fm.mean(0)) / Fm.std(0)
    r = 17
    d_ref = np.sqrt(((Fm - Fm[r]) ** 2).sum(1))
    d_ref[r] = np.inf
    o_ref = np.argsort(d_ref, kind="stable")[:25]
    o, dist = pick_decoys(Fm, r, np.ones(nb, bool), 25)
    assert np.array_equal(o, o_ref) and r not in o and np.allclose(dist, d_ref[o_ref])
    el = np.ones(nb, bool)
    el[o_ref[:5]] = False
    o2, _ = pick_decoys(Fm, r, el, 25)
    assert not set(o2) & set(o_ref[:5]) and np.array_equal(o2[:20], o_ref[5:25]) and len(o2) == 25
    o3, _ = pick_decoys(Fm, r, np.arange(nb) < 20, 25)
    assert len(o3) == 19 and r not in o3 and set(o3) <= set(range(20))                    # only 19 eligible non-focal sources exist -> all of them, never padded
    # (b) a unit on top of the bank: genes g0..g249, rows = a random subset of the T7 columns; the focal source is g17
    genes = np.array([f"g{i}" for i in range(250)])
    ctx = dict(ai0=np.arange(250), has=np.ones(250, bool), tc=np.where(np.arange(250) < nt, np.arange(250), -1), r=r, F=Fm)
    rows = (np.arange(250) < nt) & (rng.random(250) < 0.8)
    yv = rng.random(250) < 0.25
    yv[rows] = rng.random(int(rows.sum())) < 0.25
    m = "GeoBRIDGE"
    Vm = {mm: np.where(ctx["tc"] >= 0, bank.X[mm][r][np.arange(250)], np.nan) for mm in bank.X}
    prior = {mm: np.where(ctx["tc"] >= 0, bank.prior(mm, r)[np.arange(250)], np.nan) for mm in bank.X}
    u = dict(kind="syn", system="SYN", tf=bank.members[r], genes=genes, bank=bank, V=Vm, prior=prior, axis="SYN")
    d1, d2, tol = bank_cognate_parity(u, ctx, m, rows)
    assert d1 == 0 and d2 < 1e-12, (d1, d2)                                                # the focal bank row / prior IS the cognate vector / prior
    bad = dict(u, V={m: Vm[m] + 0.1 * rng.standard_normal(250)})
    assert bank_cognate_parity(bad, ctx, m, rows)[0] > tol                                 # a mismatching cognate vector is detected
    dd = prepare_decoys(u, ctx, m, rows, 30)
    assert dd["status"] == "OK" and len(dd["sel"]) == 30 and r not in dd["sel"] and not {5, 6} & set(dd["sel"].tolist())        # NaN-row sources 5, 6 are ineligible
    assert dd["Sd"].shape == (30, int(rows.sum())) and np.isfinite(dd["Sd"]).all() and np.isfinite(dd["Pd"]).all()
    cols = dd["cols"]
    for j, s_ in enumerate(dd["sel"][:5]):                                                 # decoy score row / leave-decoy-out prior == bank.X / bank.prior (CB.loo_prior) of that source
        assert np.array_equal(dd["Sd"][j], bank.X[m][s_][cols]) and np.allclose(dd["Pd"][j], bank.prior(m, int(s_))[cols], atol=1e-12)
    few = prepare_decoys(u, ctx, m, rows, 100)
    assert few["status"].startswith("OK_FEWER") and len(few["sel"]) == nb - 3              # fewer than requested usable -> every usable source (not the focal, not the 2 NaN rows), flagged
    # (c) the worker == an independent brute force: scipy ranks + sklearn roc_auc_score on the multiplicity-expanded sample, for several decoys and several draws of both schemes
    B = 6
    unitsx = np.array([f"M{i // 10:03d}" for i in range(250)])
    drx = PT.Draws(genes, B, 21, {50: 22}, {50: unitsx})
    _D.clear()
    _D["__C__"], _D["__B__"] = drx.C, B
    _D["k"] = dict(n=250, cols=cols, rows=rows, y=yv, gi=np.arange(250), Sd=dd["Sd"][:4], Pd=dd["Pd"][:4])
    _k, rs = _decoy_task("k")
    _D.clear()
    for j in range(4):
        sj, pj = dd["Sd"][j], dd["Pd"][j]
        m0, m1 = rankdata(pj), (rankdata(pj) + rankdata(sj)) / 2
        yr = yv[cols]
        ref_pt = roc_auc_score(yr, m1) - roc_auc_score(yr, m0)
        assert abs(rs["pt"]["AUROC"][j] - ref_pt) < 1e-12, (rs["pt"]["AUROC"][j], ref_pt)
        for si, ct in enumerate((drx.gene.counts, drx.mods[50].counts)):
            for b in range(B):
                w = ct[b][cols].astype(int)
                keepi = w > 0
                if yr[keepi].all() or (~yr[keepi]).all():
                    continue
                ref = roc_auc_score(yr[keepi], m1[keepi], sample_weight=w[keepi]) - roc_auc_score(yr[keepi], m0[keepi], sample_weight=w[keepi])
                got = (rs["dg"] if si == 0 else rs["dm"])["AUROC"][j, b]
                assert abs(got - ref) < 1e-6, (j, si, b, got, ref)                          # float32 storage of the draws
    # (d) calibrated statistics: hand-computed pool rank / percentile / calibrated value; NO_CALL below min_n; draws pair cognate with the per-draw decoy median
    d_pt = np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    s = decoy_stats(0.35, np.full(5, 0.35), np.full(5, 0.35), d_pt, np.tile(d_pt[:, None], (1, 5)), np.tile(d_pt[:, None], (1, 5)), min_n=3)
    assert s["n"] == 6 and abs(s["pct"] - 4 / 6) < 1e-12 and abs(s["p_pool"] - (1 + 2) / 7) < 1e-12 and abs(s["med"] - 0.25) < 1e-12 and abs(s["cal"] - 0.10) < 1e-12
    assert np.allclose(s["dm_g"], 0.25) and abs(s["cal_g_lo"] - 0.10) < 1e-12 and abs(s["cal_m_hi"] - 0.10) < 1e-12
    s_tie = decoy_stats(0.2, np.full(3, 0.2), np.full(3, 0.2), d_pt, np.tile(d_pt[:, None], (1, 3)), np.tile(d_pt[:, None], (1, 3)), min_n=3)
    assert abs(s_tie["pct"] - (2 + 0.5) / 6) < 1e-12 and abs(s_tie["p_pool"] - (1 + 4) / 7) < 1e-12                     # ties count as >= in the pool p, half in the percentile
    s_nc = decoy_stats(0.35, np.full(3, 0.35), np.full(3, 0.35), d_pt, np.tile(d_pt[:, None], (1, 3)), np.tile(d_pt[:, None], (1, 3)), min_n=10)
    assert np.isnan(s_nc["cal"]) and np.isnan(s_nc["p_pool"]) and np.isnan(s_nc["cal_g_lo"]) and np.isnan(s_nc["dm_g"]).all()
    nanp = decoy_stats(0.35, np.full(3, 0.35), np.full(3, 0.35), np.r_[d_pt, np.nan, np.nan], np.vstack([np.tile(d_pt[:, None], (1, 3)), np.full((2, 3), np.nan)]), np.vstack([np.tile(d_pt[:, None], (1, 3)), np.full((2, 3), np.nan)]), min_n=3)
    assert nanp["n"] == 6 and abs(nanp["cal"] - 0.10) < 1e-12                                                          # NaN decoys are dropped, not zero-filled
    # (e) calibrated ranking: a method with a high prior has a high RAW increment only through its prior; the calibration removes it (planted)
    ms = ["GeoBRIDGE", "GENIE3", "PIDC", "DeepSEM"]                                                                    # real method names (macro_and_rank only ranks METHODS)
    draws, cells = {}, []
    nd = 40
    for i, mm in enumerate(ms):                                                                                        # 3 units, deltas rise with the prior quality; decoys share the prior-driven part
        for ui in range(3):
            uid = f"syn|SYN|T{ui}|resp"
            raw = 0.02 * i + 0.01 * rng.standard_normal(nd)
            coh = 0.0 if mm == "GeoBRIDGE" else 0.0
            for k in ("AUROC",):
                for nm, base_ in (("S", 0.6 - 0.05 * i), ("P", 0.5 + 0.03 * i), ("M1", 0.5 + 0.03 * i + 0.02 * i)):
                    arr = (base_ + 0.005 * rng.standard_normal(nd)).astype(np.float32)
                    draws[f"{uid}|{mm}|{k}|{nm}|g"], draws[f"{uid}|{mm}|{k}|{nm}|m"] = arr, arr
                dm_ = (0.02 * i + 0.0 * coh + 0.002 * rng.standard_normal(nd)).astype(np.float32)
                draws[f"{uid}|{mm}|{k}|DM|g"], draws[f"{uid}|{mm}|{k}|DM|m"] = dm_, dm_
            cells.append(dict(kind="syn", system="SYN", tf=f"T{ui}", target="resp", method=mm, status="OK", macro_group="SYN", **{f"{nm}_AUROC": v for nm, v in (("S", 0.6 - 0.05 * i), ("P", 0.5 + 0.03 * i), ("M1", 0.5 + 0.05 * i))},
                              calD_AUROC=0.0, cog_pct_AUROC=0.5, p_pool_AUROC=0.5))
    rk, mac = macro_and_rank(pd.DataFrame(cells), draws, nd)
    m_ = rk[(rk.group == "SYN") & (rk.metric == "AUROC")].set_index("delta_kind")
    assert set(m_.index) == {"raw", "calibrated"} and m_.loc["raw", "n_methods"] == 4 and m_.loc["calibrated", "n_methods"] == 4
    assert m_.loc["raw", "tau_vsP"] > 0.99 and (np.isnan(m_.loc["calibrated", "tau_vsP"]) or abs(m_.loc["calibrated", "tau_vsP"]) < 1.0)          # raw increment ordered exactly like the prior quality
    assert m_.loc["raw", "winner_incr"] == "DeepSEM" and abs(float(mac[(mac.method == "DeepSEM")].calD.iloc[0])) < 1e-12
    # a method lacking decoys for ONE unit is excluded from the calibrated ranking (never zero-filled) but kept in the raw one
    d2 = {k: v for k, v in draws.items() if not k.startswith("syn|SYN|T1|resp|DeepSEM|AUROC|DM")}
    c2 = pd.DataFrame(cells)
    c2.loc[(c2.method == "DeepSEM") & (c2.tf == "T1"), "calD_AUROC"] = np.nan
    rk2, _m2 = macro_and_rank(c2, d2, nd)
    q = rk2[(rk2.group == "SYN") & (rk2.metric == "AUROC")].set_index("delta_kind")
    assert q.loc["raw", "n_methods"] == 4 and q.loc["calibrated", "n_methods"] == 3, q[["n_methods"]]


def selftest_data(a):
    """real-data parity on PANC RFX6 (and the systems named in --systems): S / P AUROC / AP / AP-lift of all 10 methods x 6 targets vs the audit (1e-9), GB S vs the lattice table, M0 == P identity
    (asserted inside cell_primary); decoy matching == gb_defense_dutil.nearest_matched on the real bank; bank row / prior == cognate vector / prior; independent brute force of decoy increments."""
    import gb_layer_lattice as L
    import gb_lattice_extra as LE
    LE.register()
    for spec in a.systems.split(","):
        system, _, tf = spec.partition(":")
        PT.guard_systems([system])
        dr, _ = PT.lattice_draws(system, 40, sizes=SIZES2)
        tf = tf or L.SYSTEMS[system]["tfs"][0]
        u = PT.lattice_unit(system, tf, dr.gene)
        ns = argparse.Namespace(B=40, no_parity=False, secondary=[], loto=False, loto_only=False, no_decoy=False, n_decoy=N_DECOY, n_jobs=1)
        draws, sec, loto, decs = {}, [], {}, []
        rows, par = process_unit(u, dr, ns, sec, loto, draws, decs)
        gb = [r for r in rows if r["method"] == GB and r["target"] == "resp"][0]
        print(f"[prior-adj] data parity {system}/{tf}: {len(rows)} method-target rows, audit parity worst={par['worst']:.2e} over {par['n']} comparisons; GB resp S={gb['S_AUROC']:.6f} P={gb['P_AUROC']:.6f} "
              f"M1={gb['M1_AUROC']:.6f} dAUROC={gb['d_AUROC']:+.4f}", flush=True)
        assert par["n"] > 0 and par["worst"] < 1e-9
        # ---- decoy path on real data
        import gb_defense_dutil as DU
        import gb_lattice_kprofile as KP
        ctx = unit_ctx(u)
        bankd, _w = KP.open_bank(u["axis"])
        cx = KP.axis_ctx(u["axis"])
        o_ref, _info = DU.nearest_matched(bankd, cx, ctx["r"], N_DECOY)
        o_mine, _d = pick_decoys(ctx["F"], ctx["r"], np.r_[np.ones(len(u["bank"].members), bool)], N_DECOY)
        assert np.array_equal(o_ref, o_mine), "decoy matching differs from gb_defense_dutil.nearest_matched"
        mc = match_ctx(u["axis"], u["bank"])
        lms_ref = np.asarray([cx["lms"][{g: i for i, g in enumerate(cx["axis"])}[m_]] for m_ in u["bank"].members], float)
        print(f"[prior-adj] decoy matching == dutil on {len(o_ref)} sources; frozen log1p mean Ms vs KP.axis_ctx max|diff| {np.abs(mc['lms'] - lms_ref).max():.2e}", flush=True)
        assert np.abs(mc["lms"] - lms_ref).max() < 1e-4
        calibrated = [r for r in rows if str(r.get("decoy_status", "")).startswith("OK")]
        assert calibrated, "no cell was decoy-calibrated"
        print(f"[prior-adj] decoy cells: {len(calibrated)} (method x target) calibrated; n_decoy min/median {min(r['n_decoy'] for r in calibrated)}/{int(np.median([r['n_decoy'] for r in calibrated]))}; "
              f"max bank-cognate parity score {max(r['decoy_parity_bank_score'] for r in calibrated):.2e} prior {max(r['decoy_parity_bank_prior'] for r in calibrated):.2e}", flush=True)
        # ---- independent brute force of one real decoy: scipy ranks + sklearn AUROC, rows of the cell
        from scipy.stats import rankdata
        from sklearn.metrics import roc_auc_score
        dec = pd.DataFrame(decs)
        for m in (GB, "GENIE3"):
            T = {t_: (y_, rw_) for t_, y_, rw_, g_ in unit_cells(u)}
            y_, rw_ = T["resp"]
            sub = dec[(dec.target == "resp") & (dec.method == m)].iloc[:3]
            cols = np.flatnonzero(rw_)
            for _, q in sub.iterrows():
                s_ = u["bank"].members.index(q.decoy)
                sj = u["bank"].X[m][s_][ctx["ai0"][cols]]
                pj = u["bank"].prior(m, s_)[ctx["ai0"][cols]]
                yr = np.asarray(y_, bool)[cols]
                ref = roc_auc_score(yr, (rankdata(pj) + rankdata(sj)) / 2) - roc_auc_score(yr, rankdata(pj))
                assert abs(ref - q.d_AUROC) < 1e-9, (m, q.decoy, ref, q.d_AUROC)
        print("[prior-adj] decoy increments reproduce an independent scipy/sklearn brute force", flush=True)
    print("[prior-adj] selftest (data parity) PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest")
    p.add_argument("--data", action="store_true")
    p.add_argument("--systems", default="PANC")
    p = sub.add_parser("run")
    p.add_argument("--systems", default="")
    p.add_argument("--regulon", default="")
    p.add_argument("--B", type=int, default=B_DEFAULT)
    p.add_argument("--out", required=True)
    p.add_argument("--tfs", default="", help="lattice TFs (audit roster)")
    p.add_argument("--reg-tfs", default="", dest="reg_tfs", help="regulon sources")
    p.add_argument("--slice", type=int, default=0)
    p.add_argument("--nslice", type=int, default=1)
    p.add_argument("--secondary", default="resp,regulon", help="targets that also get the logistic diagnostic ('' = none)")
    p.add_argument("--B-secondary", type=int, default=B_SEC_DEFAULT, dest="B_secondary")
    p.add_argument("--loto", action="store_true")
    p.add_argument("--loto-only", action="store_true", dest="loto_only", help="LOTO stage only (whole blocks in one process; no primary / decoy / logistic work, no cells file)")
    p.add_argument("--B-loto", type=int, default=B_LOTO_DEFAULT, dest="B_loto")
    p.add_argument("--n-decoy", type=int, default=N_DECOY, dest="n_decoy")
    p.add_argument("--no-decoy", action="store_true", dest="no_decoy")
    p.add_argument("--n-jobs", type=int, default=8, dest="n_jobs")
    p.add_argument("--no-parity", action="store_true", dest="no_parity")
    p = sub.add_parser("merge")
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--out", required=True)
    p.add_argument("--groups", required=True, help="comma list: SYSTEM/TF or MACRO:<group>")
    p.add_argument("--targets", default="resp")
    a = ap.parse_args()
    {"selftest": selftest, "run": cmd_run, "merge": cmd_merge, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    main()
