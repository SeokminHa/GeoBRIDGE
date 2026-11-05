#!/usr/bin/env python
"""gb_prospective.py -- FROZEN PROSPECTIVE TEST of the GB target prior (results doc s358 + FROZEN-SPEC AMENDMENT (s359 v2 block), s363, s366, s367 Opus-review declarations).  EVALUATION ONLY: no score /
prior / feature / k is changed; truth is read only inside gb_layer_lattice.build_endpoint (after every score artifact; plus one re-read of the SAME truth builder for resp_ok in the declared sensitivity)
and every vector below is observational.  Written by Sonnet 2026-09-29, revised 2026-09-30 after the Opus FIX_REQUIRED review (declared in s367 BEFORE any prospective result); Opus re-review pending.

  SYSTEMS  PRIMARY: CARDIACV (TBX5, NKX2-5), RETINA (NRL, CRX).  SENSITIVITY (never pooled, never mixed into RETINA's draws): RETINA_NRL_3V2 (declared s366 R3).  PARITY (regression check of THIS script
           against the published audit numbers; NOT part of the prospective test): PANC.  The merge step hard-codes the roster of each system (ROSTER).
  MASK     M1 = the audit/attribution mask (gb_audit_decomp.masks_and_priors): response-tested & occupancy-mappable fixture-axis genes (lattice universe U), finite for EVERY Tier-1 cognate row, on the
           frozen T7 target set, with a finite leave-focal-out prior for every Tier-1 method -- ONE mask for every vector (identical to s357-s365).  Reduction arms use MR = M1 & all nine REF features finite.
           Declared NON-DECISIVE sensitivities: `pairwise_U` (each comparator on its own finite mask within U) and `resp_only` (response-tested & T7 & all-finite, WITHOUT occupancy mappability).
  VECTORS  GBP = GB leave-focal-out prior P_pairU^(-r).  PRIMARY COMPARATORS (21 per TF): every OTHER Tier-1 method's leave-focal-out prior (9), every Tier-1 method's cognate row (10, INCLUDING GB's own
           cognate row: the stricter reading), COMP_Z, COMP_RANK.  A missing method / composite / TOO_FEW cell makes the system INCOMPLETE (never zero-filled; a missing cell counts as p = 1 in the Holm family).
           DESCRIPTIVE: COMP_PC1, COMP_FA, ablation priors P_pairM / P_cellU / P_cellM, pcorr (= |Pearson| prior), the nine REF rankers (best single REF = HINDSIGHT).
  REDUCTION (PRIMARY, truth-free): z-row of GBP over MR regressed on the 9 REF features, cross-fitted on the fixed gene-hash folds, by (a) HistGradientBoostingRegressor(random_state 0) and (b) OLS on
           rank-normal features; residual response AUROC / AP-lift vs .5 / 1 (gene bootstrap B = 2000 = counts[:2000], the frozen spec; the full-B version is reported alongside), Holm over the TFs of the system
           per arm; direction + supports_residual = Holm-reject AND residual AUROC > .5.  Secondary: linear OLS.
  METRICS  AUROC (primary), AUPRC, prevalence, AP-lift, tie-aware top-50 yield P50.  Gene bootstrap = the lattice's shared multinomial draws: B = 9999 by default; results are ALSO reported at B = 2000
           (counts[:2000], prefix-stable).  A family with 1/(B+1) > alpha/m is NOT_EVALUABLE_RESOLUTION; p1 Monte-Carlo SE and a near-threshold flag are reported.
  SUCCESS  per (system, TF, primary comparator): d_AUROC = AUROC(GBP) - AUROC(comparator) >= .01 AND Holm rejects AND the Holm-step lower bound > 0.  PRIMARY Holm family = ALL primary cells of the SYSTEM
           (TF x comparator, m = 42 for two TFs); per-TF families and the family without GB's own cognate row are sensitivities.  Top-50: strict = d_P50 > 0 with the same system-wide Holm on P50 (primary);
           lenient (d_P50 > 0) and the unadjusted one-sided 95% lower bound are descriptors.  Gate = the BASE lattice response gate (>= 20 responders, >= 20 non-responders, >= 20 units); the K562-style support rule
           (>= 100 bound, >= 20 A, >= 20 B) is reported as a descriptor.  VERDICT tiers: SUCCESS / AUROC_SUCCESS_TOP50_NOT_SUPPORTED / NOT_SUCCESS / INCOMPLETE / NOT_EVALUABLE_GATE / NOT_EVALUABLE_RESOLUTION.
           Equivalence (report only): the 95% percentile interval of d_AUROC lies within +-.01.  All outcomes are reported; no criterion is changed.

  python scratch/gb_prospective.py --selftest
  python scratch/gb_prospective.py run --systems CARDIACV,RETINA --out results/real_uniform/prospective [--B 9999] [--tfs A,B]
  python scratch/gb_prospective.py merge --out DIR
"""
from __future__ import annotations

import argparse
import glob
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
import gb_attribution as A  # noqa: E402  (NOT edited)
import gb_audit_decomp as AD  # noqa: E402  (NOT edited; helpers only: masks_and_priors, rank_normal, hgb_cross_fit)
import gb_canon_table as CT  # noqa: E402
import gb_catbank as CB  # noqa: E402
import gb_decompose as D  # noqa: E402  (NOT edited; helpers only: zrow, fold_ols, eval_vec, boot_p)

R = A.R
GB = A.GB
PRIMARY_SYSTEMS = ("CARDIACV", "RETINA")
SENS_SYSTEMS = ("RETINA_NRL_3V2",)
PARITY_SYSTEMS = ("PANC",)
ROSTER = {"CARDIACV": ["TBX5", "NKX2-5"], "RETINA": ["NRL", "CRX"], "RETINA_NRL_3V2": ["NRL"], "PANC": ["RFX6"]}
ROLE = {**{s: "PRIMARY" for s in PRIMARY_SYSTEMS}, **{s: "SENSITIVITY" for s in SENS_SYSTEMS}, **{s: "PARITY_CHECK" for s in PARITY_SYSTEMS}}
EPS_GAIN, ALPHA = 0.01, 0.05
B_DEFAULT, B_SPEC = 9999, 2000
MET = D.MET                                                       # AUROC, AUPRC, APLIFT, P50
DESC_COMPS = ("COMP_PC1", "COMP_FA")
ABL = ("P_pairM", "P_cellU", "P_cellM")
GB_COG = f"cognate::{GB}"
PRIM_COMPS = [f"prior::{m}" for m in A.METHODS if m != GB] + [f"cognate::{m}" for m in A.METHODS] + ["COMP_Z", "COMP_RANK"]     # 9 + 10 + 2 = 21
RED_PRIMARY_ARMS = ("hgb", "ols_ranknormal")


# ===================================================================================================== pure numerics (unit-tested by --selftest)
def holm_reject(p, alpha=ALPHA):
    """Holm step-down on a vector of p-values (NaN = not testable, never rejected, does not count toward m).  Returns (reject bool array, adjusted p, step alpha per hypothesis; NaN when not testable)."""
    p = np.asarray(p, float)
    ok = np.isfinite(p)
    m = int(ok.sum())
    rej, adj, stepa = np.zeros(len(p), bool), np.full(len(p), np.nan), np.full(len(p), np.nan)
    idx = np.flatnonzero(ok)
    order = idx[np.argsort(p[idx], kind="stable")]
    run, alive = 0.0, True
    for i, j in enumerate(order):
        a_i = alpha / (m - i)
        stepa[j] = a_i
        run = max(run, min(1.0, (m - i) * p[j]))
        adj[j] = run
        alive = alive and p[j] <= a_i
        rej[j] = alive
    return rej, adj, stepa


def boot_p1(d, nb=None):
    """one-sided (greater) bootstrap p with the +1 rule over the first nb draws: (1 + #{d <= 0}) / (n + 1)"""
    d = np.asarray(d, float)[:nb]
    d = d[np.isfinite(d)]
    return float((1 + (d <= 0).sum()) / (len(d) + 1)) if len(d) else np.nan


def p1_se(p, nb):
    return float(np.sqrt(max(p * (1 - p), 0.0) / nb)) if np.isfinite(p) else np.nan


def lower_bound(d, alpha, nb=None):
    d = np.asarray(d, float)[:nb]
    d = d[np.isfinite(d)]
    return float(np.percentile(d, 100 * alpha)) if len(d) and np.isfinite(alpha) else np.nan


def contrast_draws(a, c, y, rows, gi, counts):
    """paired difference (a - c) on rows & finite(a) & finite(c): {metric: (a point, c point, float64 draws of the difference)} plus n / n_pos; None if a class is empty"""
    a, c, rows = np.asarray(a, float), np.asarray(c, float), np.asarray(rows, bool)
    rj = rows & np.isfinite(a) & np.isfinite(c)
    yy = np.asarray(y, bool)[rj]
    if not (yy.any() and (~yy).any()):
        return None
    pa, ba = CT.eval_rows(a, y, rj, gi, counts)
    pc, bc = CT.eval_rows(c, y, rj, gi, counts)
    return dict(n=int(rj.sum()), n_pos=int(yy.sum()), n_neg=int((~yy).sum()), frac_mask=float(rj.sum() / max(int(rows.sum()), 1)),
                m={k: (pa[k], pc[k], np.asarray(ba[k], np.float64) - np.asarray(bc[k], np.float64)) for k in MET})


def family_rows(cons, alpha=ALPHA, nb=None, sfx=""):
    """cons: list of contrast_draws() results (None = TOO_FEW).  Per-metric d, CI, one-sided p (+MC SE), Holm reject / step alpha / lower bound WITHIN this family (AUROC and P50), computed on the first nb
    draws; keys carry `sfx`.  A TOO_FEW member counts toward m with p = 1 (never dropped from the family)."""
    out = [dict() for _ in cons]
    m_all = len(cons)
    for k in ("AUROC", "P50"):
        p1 = np.array([boot_p1(c["m"][k][2], nb) if c is not None else 1.0 for c in cons])
        rej, adj, stepa = holm_reject(p1, alpha)
        for i, c in enumerate(cons):
            if c is None:
                continue
            a_pt, c_pt, dd = c["m"][k]
            dd = dd[:nb]
            n = len(dd)
            out[i].update({f"d_{k}{sfx}": a_pt - c_pt, f"a_{k}{sfx}": a_pt, f"c_{k}{sfx}": c_pt, f"d_{k}_lo{sfx}": float(np.percentile(dd, 2.5)), f"d_{k}_hi{sfx}": float(np.percentile(dd, 97.5)),
                           f"p1_{k}{sfx}": p1[i], f"p1_se_{k}{sfx}": p1_se(p1[i], n), f"p2_{k}{sfx}": D.boot_p(dd), f"holm_p_{k}{sfx}": adj[i], f"holm_reject_{k}{sfx}": bool(rej[i]),
                           f"holm_alpha_step_{k}{sfx}": stepa[i], f"holm_lo_{k}{sfx}": lower_bound(dd, stepa[i]), f"lb95_unadj_{k}{sfx}": lower_bound(dd, ALPHA),
                           f"resolution_ok_{k}{sfx}": bool(1.0 / (n + 1) <= alpha / m_all), f"m_family{sfx}": m_all})
    for i, c in enumerate(cons):
        if c is None:
            continue
        for k in ("APLIFT", "AUPRC"):
            a_pt, c_pt, dd = c["m"][k]
            out[i].update({f"d_{k}{sfx}": a_pt - c_pt, f"d_{k}_lo{sfx}": float(np.percentile(dd[:nb], 2.5)), f"d_{k}_hi{sfx}": float(np.percentile(dd[:nb], 97.5))})
        out[i].update(n=c["n"], n_pos=c["n_pos"], n_neg=c["n_neg"], frac_mask=c["frac_mask"])
        o = out[i]
        o[f"cell_success_auroc{sfx}"] = bool(o[f"d_AUROC{sfx}"] >= EPS_GAIN and o[f"holm_reject_AUROC{sfx}"] and o[f"holm_lo_AUROC{sfx}"] > 0)
        o[f"cell_top50_strict{sfx}"] = bool(o[f"d_P50{sfx}"] > 0 and o[f"holm_reject_P50{sfx}"] and o[f"holm_lo_P50{sfx}"] > 0)
        o[f"cell_top50_lenient{sfx}"] = bool(o[f"d_P50{sfx}"] > 0)
        o[f"equivalent_pm01{sfx}"] = bool(o[f"d_AUROC_lo{sfx}"] > -EPS_GAIN and o[f"d_AUROC_hi{sfx}"] < EPS_GAIN)
    return out


def both_B(cons, nb_full):
    """family_rows at the full B and (when B > 2000) at the prefix B = 2000 with suffix `_b2000`"""
    f = family_rows(cons, nb=None)
    if nb_full > B_SPEC:
        g = family_rows(cons, nb=B_SPEC, sfx="_b2000")
        f = [dict(x, **y) for x, y in zip(f, g)]
    return f


def sys_eval(draws, points, tfs, comps, met, nb, family="sw", exclude=(), alpha=ALPHA):
    """SYSTEM-level Holm evaluation from stored draws.  draws[(tf, comp, met)] = float64 draws; points[(tf, comp, met)] = point difference.  Cells = tfs x comps (minus `exclude`); a cell without draws is
    MISSING and counts as p = 1 (never dropped).  family = 'sw' (one family over all cells, PRIMARY) | 'pertf' (one family per TF, sensitivity).  Returns a DataFrame."""
    cells = [(t, c) for t in tfs for c in comps if c not in exclude]
    have = [(t, c, met) in draws for t, c in cells]
    p1 = np.array([boot_p1(draws[(t, c, met)], nb) if h else 1.0 for (t, c), h in zip(cells, have)])
    grp = [t if family == "pertf" else "ALL" for t, c in cells]
    rej, adj, stepa = np.zeros(len(cells), bool), np.full(len(cells), np.nan), np.full(len(cells), np.nan)
    for g in sorted(set(grp)):
        ix = [i for i, x in enumerate(grp) if x == g]
        r_, a_, s_ = holm_reject(p1[ix], alpha)
        rej[ix], adj[ix], stepa[ix] = r_, a_, s_
    rows = []
    for i, (t, c) in enumerate(cells):
        m_g = sum(1 for x in grp if x == grp[i])
        d = points.get((t, c, met), np.nan)
        lo = lower_bound(draws[(t, c, met)], stepa[i], nb) if have[i] else np.nan
        n = len(np.asarray(draws[(t, c, met)])[:nb]) if have[i] else nb
        thr = (d >= EPS_GAIN) if met == "AUROC" else (d > 0)
        rows.append(dict(tf=t, comparator=c, metric=met, family=family, nb=nb, available=have[i], d=d, p1=p1[i], p1_se=p1_se(p1[i], n), holm_p=adj[i], holm_reject=bool(rej[i]), alpha_step=stepa[i],
                         holm_lo=lo, lb95_unadj=lower_bound(draws[(t, c, met)], alpha, nb) if have[i] else np.nan, m_family=m_g, resolution_ok=bool(1.0 / (n + 1) <= alpha / m_g),
                         near_threshold=bool(have[i] and np.isfinite(stepa[i]) and abs(p1[i] - stepa[i]) < 2 * p1_se(p1[i], n)),
                         success=bool(have[i] and np.isfinite(d) and thr and rej[i] and np.isfinite(lo) and lo > 0)))
    return pd.DataFrame(rows)


def select_parts(metas, system):
    """parts (stems) of exactly this system, by the meta's own `system` field (never by file-name prefix: RETINA must not absorb RETINA_NRL_3V2)"""
    return [m for m in metas if m["system"] == system]


# ===================================================================================================== one (system, TF)
def ablation_priors(axis, bank, r, tc, ai0):
    """leave-focal-out priors of P_pairM / P_cellU / P_cellM on the frozen T7 (gb_decompose.ablation_priors algebra, reproduced: loo_prior on the stored Z, S, C)"""
    with np.load(f"{CB.OUT}/{axis}_priors.npz", allow_pickle=True) as z:
        Z3 = {n: (np.asarray(z[f"T7|{n}|Z"]), np.asarray(z[f"T7|{n}|S"]), np.asarray(z[f"T7|{n}|C"])) for n in ABL}
    tcz = np.where(tc >= 0, tc, 0)
    return {n: np.where(tc >= 0, CB.loo_prior(Z, S, C, r)[tcz], np.nan) for n, (Z, S, C) in Z3.items()}


def eval_two(v, y, rows, gi, counts, null=None):
    """D.eval_vec at B = 2000 (counts[:2000]; the frozen spec / audit_decomp_v2 parity) with the full-B CI / p alongside (`_fullB`)"""
    e = D.eval_vec(v, y, rows, gi, counts[:B_SPEC], null=null)
    if e is not None and len(counts) > B_SPEC:
        f = D.eval_vec(v, y, rows, gi, counts, null=null)
        e.update({f"{k}_fullB": val for k, val in f.items() if k.endswith(("_lo", "_hi")) or k.startswith(("p_", "p1_"))})
    return e


def run_tf(system, tf, blk, B, L, LE):
    cfg = L.SYSTEMS[system]
    bank = A.load_bank(cfg["axis"])
    ep = L.build_endpoint(system, tf, blk, False)
    V, genes = ep["V"], np.asarray(ep["genes"]).astype(str)
    use = [m for m in A.METHODS if m in V and m in bank.X]
    missing = [m for m in A.METHODS if m not in use]
    if GB not in use:
        raise L.Blocked(f"GB_ROW_OR_BANK_MISSING {system}/{tf}")
    r = bank.row_of(tf)
    if r is None:
        raise L.Blocked(f"FOCAL_NOT_IN_BANK {system}/{tf}")
    if "prior GB" not in V:
        raise L.Blocked(f"NO_LATTICE_PRIOR_GB {system}/{tf}")
    M0 = np.asarray(ep["U"], bool) & np.all([np.isfinite(V[m]) for m in use], 0)
    ai, has, ai0, tc, prior, M1 = AD.masks_and_priors(bank, genes, V, use, r, M0)
    f = M1 & np.isfinite(V["prior GB"])                                        # exact parity with the lattice prior (as audit lattice_tf)
    dgb = float(np.abs(prior[GB][f] - V["prior GB"][f]).max()) if f.any() else np.nan
    if not dgb <= 1e-9:
        raise SystemExit(f"REFUSING_PARITY {system}/{tf}: GB leave-focal-out prior differs from lattice `prior GB` by {dgb}")
    y, rows = L.targets(ep, M1)["resp"]
    gate = bool(LE._ORIG["gate_of"](ep, y, rows, M1))                            # BASE response gate (>= 20 resp, >= 20 non-resp, >= 20 units)
    gate_k562 = bool(L.gate_of(ep, y, rows, M1))                                # K562-style support rule for the gated systems: a DESCRIPTOR
    gi, counts, folds = ep["gi"], blk.counts[:B], np.asarray(ep["fold"])
    gbp = np.where(M1, prior[GB], np.nan)

    def comp_dict(mask, gbv_prior, priors, kinds):
        """comparator vectors on `mask` (prior finite already enforced by the mask builder where required)"""
        prim_, desc_ = {}, {}
        for m in use:
            if m != GB:
                prim_[f"prior::{m}"] = np.where(mask, priors[m], np.nan)
            prim_[f"cognate::{m}"] = np.where(mask, np.asarray(V[m], float), np.nan)
        nc = []
        for c in ("COMP_Z", "COMP_RANK"):
            if c in V and np.isfinite(np.asarray(V[c], float)[mask]).all():
                prim_[c] = np.where(mask, np.asarray(V[c], float), np.nan)
            else:
                nc.append(c)
        for c in DESC_COMPS:
            if c in V and np.isfinite(np.asarray(V[c], float)[mask]).all():
                desc_[c] = np.where(mask, np.asarray(V[c], float), np.nan)
        return prim_, desc_, nc

    prim, desc, nocall = comp_dict(M1, gbp, prior, None)
    for n, v in ablation_priors(cfg["axis"], bank, r, tc, ai0).items():
        desc[f"ablation::{n}"] = np.where(M1 & np.isfinite(v), v, np.nan)
    if "|Pearson|" in use:
        desc["ablation::pcorr"] = prim["prior::|Pearson|"]
    refs = list(L.REFS)
    Xr = np.column_stack([np.asarray(V[n], float) for n in refs]) if all(n in V for n in refs) else None
    mr = (M1 & np.isfinite(Xr).all(1)) if Xr is not None else None
    vec_rows = []

    def add_vec(kind, name, v, rws, extra=None):
        e = eval_two(v, y, rws, gi, counts)
        base = dict(system=system, tf=tf, kind=kind, vector=name, gate=gate, n_M1=int(M1.sum()), n_MR=int(mr.sum()) if mr is not None else -1, **(extra or {}))
        vec_rows.append(dict(base, status="TOO_FEW") if e is None else dict(base, status="OK" if gate else "DESCRIPTIVE_ONLY", **e))

    add_vec("GBP", "GB_prior", gbp, M1)
    for k, v in {**prim, **desc}.items():
        add_vec("primary" if k in prim else "descriptive", k, v, M1)
    con, draws = [], {}

    def contrasts(gb_vec, cdict, rws, mask_kind, fam, store):
        names = list(cdict)
        cons = [contrast_draws(gb_vec, cdict[k], y, rws, gi, counts) for k in names]
        for k, c, fr in zip(names, cons, both_B(cons, len(counts))):
            base = dict(system=system, tf=tf, comparator=k, mask_kind=mask_kind, family=fam, gate=gate, gate_k562_support=gate_k562)
            con.append(dict(base, status="TOO_FEW") if c is None else dict(base, status="OK" if gate else "DESCRIPTIVE_ONLY", **fr))
            if c is not None and store:
                for met in ("AUROC", "P50"):
                    draws[f"{tf}::{mask_kind}::{k}::{met}"] = np.asarray(c["m"][met][2], np.float64)

    contrasts(gbp, prim, M1, "M1", "primary_perTF", True)                      # per-TF Holm columns = SENSITIVITY; the PRIMARY system-wide family is recomputed at merge from these draws
    contrasts(gbp, desc, M1, "M1", "descriptive_perTF", True)
    # ---- non-decisive sensitivities: an exception here is RECORDED (SENSITIVITY_NOT_RUN) and never aborts the primary analysis (partial rows rolled back)
    sens_status, n_resp_only = {}, None

    def sens_block(name, fn):
        n0, d0 = len(con), set(draws)
        try:
            fn()
            sens_status.setdefault(name, "OK")
        except (Exception, SystemExit) as e:
            del con[n0:]
            for k in set(draws) - d0:
                draws.pop(k)
            sens_status[name] = f"SENSITIVITY_NOT_RUN: {type(e).__name__}: {e}"
            print(f"[prosp] {system}/{tf}: {name} {sens_status[name]}", flush=True)

    def pairwise_U():
        # pairwise-finite mask on U (each comparator on its own finite set; comparators NOT re-z'd)
        Ug = np.asarray(ep["U"], bool)
        gb_u = np.where(Ug & (tc >= 0) & np.isfinite(prior[GB]), prior[GB], np.nan)
        pw = {}
        for m in use:
            if m != GB:
                pw[f"prior::{m}"] = np.where(Ug & np.isfinite(prior[m]), prior[m], np.nan)
            pw[f"cognate::{m}"] = np.where(Ug & np.isfinite(np.asarray(V[m], float)), np.asarray(V[m], float), np.nan)
        for c in ("COMP_Z", "COMP_RANK"):
            if c in V:
                pw[c] = np.where(Ug & np.isfinite(np.asarray(V[c], float)), np.asarray(V[c], float), np.nan)
        contrasts(gb_u, pw, Ug, "pairwise_U", "primary_perTF", False)

    def resp_only():
        # (s367 item 10) WITHOUT occupancy mappability = response-tested & T7 & all-finite
        nonlocal n_resp_only
        X = LE.EXTRA.get(system)
        if X is None:
            sens_status["resp_only"] = "N/A (base system: no truth builder in LE.EXTRA)"
            return
        tr, _mf = X["truth"](tf, genes)                                          # the same frozen truth builder (already read by build_endpoint); exposes resp_ok
        if not np.array_equal(np.asarray(tr["resp"], bool), np.asarray(ep["resp"], bool)):
            raise RuntimeError(f"truth builder resp differs from the endpoint resp {system}/{tf}")
        M0s = np.asarray(tr["resp_ok"], bool) & np.all([np.isfinite(V[m]) for m in use], 0)
        _a, _b, _c, tcs, priors_s, MS = AD.masks_and_priors(bank, genes, V, use, r, M0s)
        for c in ("COMP_Z", "COMP_RANK"):
            if c in V:
                MS = MS & np.isfinite(np.asarray(V[c], float))
        prim_s, _d, _nc = comp_dict(MS, None, priors_s, None)
        contrasts(np.where(MS, prior[GB], np.nan), prim_s, MS, "resp_only", "primary_perTF", False)
        n_resp_only = int(MS.sum())

    sens_block("pairwise_U", pairwise_U)
    sens_block("resp_only", resp_only)
    # ---- PRIMARY truth-free reduction on MR (HGB + rank-normal OLS), secondary linear OLS; best single REF (hindsight)
    red = []
    if Xr is None:
        red.append(dict(system=system, tf=tf, vtype="RESID", arm="NOT_RUN_REF_MISSING", status="NOT_RUN_REF_MISSING"))
    else:
        yr, rr = L.targets(ep, mr)["resp"]
        zt = D.zrow(prior[GB], mr)
        Xrn = np.column_stack([AD.rank_normal(Xr[:, j], mr) for j in range(Xr.shape[1])])
        fh, rh, sh = AD.hgb_cross_fit(zt, Xr, folds, mr)
        fo, ro, so = D.fold_ols(zt, Xrn, folds, mr)
        fl, rl, sl = D.fold_ols(zt, Xr, folds, mr)
        for vt, arm, v, ex, prim_arm in (("PRIOR_ON_MR", "none", np.where(mr, prior[GB], np.nan), {}, False), ("FIT", "hgb", fh, sh, True), ("RESID", "hgb", rh, sh, True),
                                         ("FIT", "ols_ranknormal", fo, so, True), ("RESID", "ols_ranknormal", ro, so, True), ("FIT", "ols_linear", fl, sl, False),
                                         ("RESID", "ols_linear", rl, sl, False)):
            e = eval_two(v, yr, rr, gi, counts, null={"AUROC": 0.5, "APLIFT": 1.0})
            base = dict(system=system, tf=tf, vtype=vt, arm=arm, primary_arm=prim_arm, gate=gate, n_M1=int(M1.sum()), n_MR=int(mr.sum()), **ex)
            if e is not None and vt == "RESID":
                base["direction"] = "above_0.5" if e["AUROC"] > 0.5 else "below_0.5"
            red.append(dict(base, status="TOO_FEW") if e is None else dict(base, status="OK" if gate else "DESCRIPTIVE_ONLY", **e))
        refv = {n: np.where(mr, Xr[:, i], np.nan) for i, n in enumerate(refs)}                    # signed rankers (higher = predicted responder), as lattice ref_vectors
        best, best_auc = None, -1.0
        for n, v in refv.items():
            e = eval_two(v, yr, rr, gi, counts)
            if e is not None:
                red.append(dict(system=system, tf=tf, vtype="REF", arm="hindsight_single", vector=n, primary_arm=False, gate=gate, status="OK" if gate else "DESCRIPTIVE_ONLY", **e))
                if e["AUROC"] > best_auc:
                    best, best_auc = n, e["AUROC"]
        if best is not None:
            c = contrast_draws(np.where(mr, prior[GB], np.nan), refv[best], yr, rr, gi, counts)
            fr = family_rows([c])[0]
            red.append(dict(system=system, tf=tf, vtype="GBP_minus_bestREF", arm="hindsight_single", vector=best, primary_arm=False, gate=gate, status="OK" if gate else "DESCRIPTIVE_ONLY",
                            **{k: v for k, v in fr.items() if k.startswith(("d_", "a_", "c_")) and "holm" not in k}))
    meta = dict(system=system, tf=tf, methods_used=use, methods_missing=missing, composites_no_call=nocall, n_M0=int(M0.sum()), n_M1=int(M1.sum()), n_MR=int(mr.sum()) if mr is not None else None,
                n_resp_only=n_resp_only, sensitivity_status=sens_status, n_resp_M1=int(np.asarray(y)[M1].sum()), prevalence_M1=float(np.asarray(y)[M1].mean()), gate=gate, gate_k562_support=gate_k562,
                parity_prior_GB_vs_lattice_absdiff=dgb, n_primary_comparators=len(prim), primary_comparators=list(prim), files={k: v for k, v in ep["files"].items()}, focal_in_bank=True,
                fold_sizes=[int((folds[M1] == f).sum()) for f in np.unique(folds[M1])])
    return vec_rows, con, red, draws, meta


def cmd_run(a):
    import gb_lattice_extra as LE
    import gb_layer_lattice as L
    LE.register()
    os.makedirs(a.out, exist_ok=True)
    for system in [s.strip() for s in a.systems.split(",") if s.strip()]:
        if system not in ROSTER:
            raise SystemExit(f"REFUSING: {system} not in {sorted(ROSTER)}")
        cfg = L.SYSTEMS[system]
        tfs = [t.strip() for t in a.tfs.split(",") if t.strip()] or list(ROSTER[system])
        seed = int(np.random.SeedSequence([L.BOOT_SEED, L.BLOCKS.index(cfg["block"])]).generate_state(1)[0])
        blk = L.BlockAxis(L.axis_of(cfg["axis"]), a.B, seed, None, "gene")
        vec, con, red, draws, metas = [], [], [], {}, {}
        for tf in tfs:
            t0 = time.time()
            try:
                v_, c_, r_, d_, m_ = run_tf(system, tf, blk, a.B, L, LE)
            except (L.Blocked, SystemExit, Exception) as e:                    # fail-closed per TF (recorded, never silently dropped; other TFs' rows are kept)
                metas[tf] = dict(status=f"NOT_RUN: {e}")
                print(f"[prosp] {system}/{tf}: NOT_RUN {e}", flush=True)
                continue
            vec, con, red = vec + v_, con + c_, red + r_
            draws.update(d_)
            metas[tf] = m_
            print(f"[prosp] {system}/{tf}: M1={m_['n_M1']} MR={m_['n_MR']} resp_only={m_['n_resp_only']} gate={m_['gate']} missing={m_['methods_missing']} nocall={m_['composites_no_call']} {time.time()-t0:.0f}s", flush=True)
        tag = "all" if not a.tfs else "_".join(tfs)
        stem = f"part_{system}_{tag}"
        if vec:
            for nm, rows in (("vec", vec), ("con", con), ("red", red)):
                pd.DataFrame(rows).to_csv(f"{a.out}/{stem}.{nm}.tsv", sep="\t", index=False)
            np.savez_compressed(f"{a.out}/{stem}.draws.npz", **draws)
        L.atomic_json(f"{a.out}/{stem}.meta.json", dict(system=system, stem=stem, B=a.B, seed=seed, script_sha256=L.sha_file(__file__), attribution_sha256=L.sha_file(A.__file__),
                                                        audit_sha256=L.sha_file(AD.__file__), decompose_sha256=L.sha_file(D.__file__), module_shas=L.module_shas(), tfs=metas, role=ROLE.get(system)))


# ===================================================================================================== merge
def load_system(out, system, metas):
    """tables + draws of exactly this system's parts; duplicate (tf, comparator, mask_kind, family) rows or draw keys are refused"""
    mm = select_parts(metas, system)
    tabs = {}
    for nm in ("vec", "con", "red"):
        fr = [pd.read_csv(f"{out}/{m['stem']}.{nm}.tsv", sep="\t") for m in mm if os.path.exists(f"{out}/{m['stem']}.{nm}.tsv")]
        tabs[nm] = pd.concat(fr, ignore_index=True) if fr else pd.DataFrame()
    c = tabs["con"]
    if len(c) and c.duplicated(["system", "tf", "comparator", "mask_kind", "family"]).any():
        raise SystemExit(f"REFUSING_DUPLICATE_CONTRAST_ROWS {system}")
    dr = {}
    for m in mm:
        p = f"{out}/{m['stem']}.draws.npz"
        if os.path.exists(p):
            with np.load(p) as z:
                for k in z.files:
                    tf, mk, rest = k.split("::", 2)
                    comp, met = rest.rsplit("::", 1)                                  # comparator names contain "::"
                    key = (tf, comp, met) if mk == "M1" else (tf, comp, met, mk)
                    if key in dr:
                        raise SystemExit(f"REFUSING_DUPLICATE_DRAWS {system} {key}")
                    dr[key] = z[k]
    return mm, tabs, dr


def cmd_merge(a):
    metas = [json.load(open(f)) for f in sorted(glob.glob(f"{a.out}/part_*.meta.json"))]
    allt = {nm: [] for nm in ("vec", "con", "red", "cells")}
    verdict = {}
    for system in sorted({m["system"] for m in metas}):
        mm, tabs, dr = load_system(a.out, system, metas)
        for nm in ("vec", "con", "red"):
            allt[nm].append(tabs[nm])
        tfs = ROSTER[system]
        if len({m["script_sha256"] for m in mm}) != 1 or len({int(m["B"]) for m in mm}) != 1:
            raise SystemExit(f"REFUSING_MIXED_PARTS {system}: script_sha256 / B differ across parts {sorted({(m['script_sha256'][:8], m['B']) for m in mm})}")
        Bfull = int(mm[0]["B"])
        con, red = tabs["con"], tabs["red"]
        reasons = []
        tfmeta = {t: m["tfs"][t] for m in mm for t in m["tfs"]}
        for t in tfs:
            if t not in tfmeta or "methods_missing" not in tfmeta[t]:
                reasons.append(f"TF {t} not run: {tfmeta.get(t, {}).get('status', 'absent')}")
        run_meta = {t: v for t, v in tfmeta.items() if "methods_missing" in v}
        for t, v in run_meta.items():
            if v["methods_missing"]:
                reasons.append(f"{t}: methods missing {v['methods_missing']}")
            if v["composites_no_call"]:
                reasons.append(f"{t}: composites NO_CALL/non-finite {v['composites_no_call']}")
        prim = con[(con.mask_kind == "M1") & (con.family == "primary_perTF")] if len(con) else con
        points = {}
        for r_ in prim.itertuples():
            if r_.status in ("OK", "DESCRIPTIVE_ONLY"):
                points[(r_.tf, r_.comparator, "AUROC")], points[(r_.tf, r_.comparator, "P50")] = r_.d_AUROC, r_.d_P50
        toofew = prim[~prim.status.isin(["OK", "DESCRIPTIVE_ONLY"])] if len(prim) else prim
        for r_ in toofew.itertuples():
            reasons.append(f"{r_.tf}/{r_.comparator}: {r_.status}")
        draws = {k: v for k, v in dr.items() if len(k) == 3}
        for t in tfs:
            miss = [c for c in PRIM_COMPS if (t, c, "AUROC") not in draws]
            if miss and t in run_meta:
                reasons.append(f"{t}: primary cells missing {miss}")
        # reduction: both primary arms present (OK) for every roster TF
        redsum = []
        if len(red):
            rs = red[(red.vtype == "RESID") & red.arm.isin(RED_PRIMARY_ARMS)]
            for arm in RED_PRIMARY_ARMS:
                g = rs[(rs.arm == arm) & (rs.status.isin(["OK", "DESCRIPTIVE_ONLY"]))]
                for t in tfs:
                    if t not in set(g.tf):
                        reasons.append(f"{t}: reduction arm {arm} missing / TOO_FEW")
                if len(g):
                    rej, adj, _ = holm_reject(g["p_AUROC_null"].to_numpy(float))
                    rej_f, adj_f, _ = holm_reject(g["p_AUROC_null_fullB"].to_numpy(float)) if "p_AUROC_null_fullB" in g else (rej, adj, None)
                    for (_, q), rj, ad_, rjf, adf in zip(g.iterrows(), rej, adj, rej_f, adj_f):
                        redsum.append(dict(system=system, arm=arm, tf=q["tf"], resid_AUROC=q["AUROC"], lo=q["AUROC_lo"], hi=q["AUROC_hi"], p_two_sided_B2000=q["p_AUROC_null"], holm_p_over_TFs_B2000=ad_,
                                           holm_reject_B2000=bool(rj), p_two_sided_fullB=q.get("p_AUROC_null_fullB"), holm_p_over_TFs_fullB=adf, holm_reject_fullB=bool(rjf),
                                           direction="above_0.5" if q["AUROC"] > 0.5 else "below_0.5", supports_residual=bool(rj and q["AUROC"] > 0.5), resid_APLIFT=q["APLIFT"], R2_heldout=q.get("R2_heldout")))
        # system-wide (PRIMARY) + sensitivities, at the full B and at B = 2000
        cell_tabs, summary = [], {}
        for nb in sorted({Bfull, min(Bfull, B_SPEC)}):
            sw = {met: sys_eval(draws, points, tfs, PRIM_COMPS, met, nb, "sw") for met in ("AUROC", "P50")}
            pt = {met: sys_eval(draws, points, tfs, PRIM_COMPS, met, nb, "pertf") for met in ("AUROC", "P50")}
            ng = {met: sys_eval(draws, points, tfs, PRIM_COMPS, met, nb, "sw", exclude=(GB_COG,)) for met in ("AUROC", "P50")}
            for nm, d_ in (("sw", sw), ("pertf", pt), ("noGBcog", ng)):
                for met, df in d_.items():
                    cell_tabs.append(df.assign(system=system, variant=nm))
            res_ok = bool(sw["AUROC"].resolution_ok.all() and sw["P50"].resolution_ok.all())
            au, t50 = sw["AUROC"], sw["P50"]
            summary[f"B{nb}"] = dict(nb=nb, resolution_ok=res_ok, m_primary=int(len(au)), n_cells_available=int(au.available.sum()),
                                     auroc_success=bool(au.success.all()), n_auroc_success=int(au.success.sum()), top50_strict=bool(t50.success.all()), n_top50_strict=int(t50.success.sum()),
                                     top50_lenient=bool((t50.d > 0).all()), top50_lb95_unadj_pos=bool((t50.lb95_unadj > 0).all()),
                                     auroc_success_by_tf_perTF_sensitivity={t: bool(g.success.all()) for t, g in pt["AUROC"].groupby("tf")},
                                     auroc_success_without_GB_cognate=bool(ng["AUROC"].success.all()), top50_strict_without_GB_cognate=bool(ng["P50"].success.all()),
                                     n_success_cells_near_threshold=int((au.success & au.near_threshold).sum()) + int((t50.success & t50.near_threshold).sum()),
                                     success_depends_on_near_threshold=bool((au.success & au.near_threshold).any() or (t50.success & t50.near_threshold).any()), n_near_threshold=int(au.near_threshold.sum()), min_d_AUROC=float(au.d.min()), max_p1_se=float(au.p1_se.max()),
                                     equivalent_pm01_cells=int(con[(con.mask_kind == "M1") & (con.family == "primary_perTF") & (con.get("equivalent_pm01", False) == True)].shape[0]) if len(con) else 0)
        cells = pd.concat(cell_tabs, ignore_index=True)
        allt["cells"].append(cells)

        def tier(s):
            if reasons:
                return "INCOMPLETE"
            if not all(v["gate"] for v in run_meta.values()):
                return "NOT_EVALUABLE_GATE"
            if not s["resolution_ok"]:
                return "NOT_EVALUABLE_RESOLUTION"
            if s["auroc_success"] and s["top50_strict"]:
                return "SUCCESS"
            return "AUROC_SUCCESS_TOP50_NOT_SUPPORTED" if s["auroc_success"] else "NOT_SUCCESS"
        verdict[system] = dict(role=ROLE.get(system), roster=tfs, B=Bfull, status=tier(summary[f"B{Bfull}"]), success_depends_on_near_threshold=summary[f"B{Bfull}"]["success_depends_on_near_threshold"],
                               sensitivity_status_by_tf={t: v.get("sensitivity_status") for t, v in run_meta.items()}, status_B2000=tier(summary[f"B{min(Bfull, B_SPEC)}"]), reasons=sorted(set(reasons)),
                               gate_by_tf={t: v["gate"] for t, v in run_meta.items()}, gate_k562_support_by_tf={t: v["gate_k562_support"] for t, v in run_meta.items()},
                               summary=summary, reduction=redsum,
                               resp_only_sensitivity_n={t: v.get("n_resp_only") for t, v in run_meta.items()})
    for nm, fr in allt.items():
        pd.concat([f for f in fr if len(f)], ignore_index=True).to_csv(f"{a.out}/prospective_{nm}.tsv", sep="\t", index=False) if any(len(f) for f in fr) else None
    pd.DataFrame([r for v in verdict.values() for r in v["reduction"]]).to_csv(f"{a.out}/prospective_reduction_summary.tsv", sep="\t", index=False)
    json.dump(verdict, open(f"{a.out}/prospective_verdict.json", "w"), indent=1, default=str)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk in ("role", "status", "status_B2000", "reasons")} for k, v in verdict.items()}, indent=1, default=str))


# ===================================================================================================== selftest
def selftest():
    rej, adj, st = holm_reject([0.001, 0.02, 0.5, np.nan])
    assert list(rej) == [True, True, False, False] and np.isclose(adj[0], 0.003) and np.isclose(adj[1], 0.04) and np.isnan(adj[3]) and np.isclose(st[0], 0.05 / 3), (rej, adj, st)
    assert holm_reject([0.01, 0.02, 0.03])[0].all()
    assert not holm_reject([0.02, 0.021, 0.03])[0].any()                 # step-down stops at the first failure
    d = np.r_[np.full(1990, 0.05), np.full(10, -0.01)]
    assert np.isclose(boot_p1(d), 11 / 2001) and lower_bound(d, 0.05) > 0 and np.isclose(boot_p1(d, 1000), 1 / 1001)
    # synthetic contrast: a informative, c noise -> success; a vs identical copy -> equivalent, not a success
    rng = np.random.default_rng(0)
    n = 1500
    y = rng.random(n) < 0.3
    a = y * 0.8 + rng.normal(size=n)
    c1, c2 = rng.normal(size=n), a.copy()
    counts = rng.multinomial(n, np.full(n, 1.0 / n), size=300).astype(np.int16)
    gi, rows = np.arange(n), np.ones(n, bool)
    cons = [contrast_draws(a, c, y, rows, gi, counts) for c in (c1, c2)]
    f = family_rows(cons)
    assert f[0]["cell_success_auroc"] and f[0]["d_AUROC"] > 0.1, f[0]
    assert (not f[1]["cell_success_auroc"]) and abs(f[1]["d_AUROC"]) < 1e-12 and f[1]["equivalent_pm01"], f[1]
    assert cons[0]["m"]["AUROC"][2].dtype == np.float64
    # system-wide Holm: 2 TFs x 3 comps, one cell MISSING (counts toward m with p = 1), resolution guard, MC SE
    dd = {}
    pts = {}
    for t in ("T1", "T2"):
        for c in ("c1", "c2", "c3"):
            if (t, c) == ("T2", "c3"):
                continue
            dd[(t, c, "AUROC")] = np.full(9999, 0.05) + rng.normal(0, 0.005, 9999)
            pts[(t, c, "AUROC")] = 0.05
    ev = sys_eval(dd, pts, ["T1", "T2"], ["c1", "c2", "c3"], "AUROC", 9999, "sw")
    assert len(ev) == 6 and int((~ev.available).sum()) == 1 and ev.m_family.eq(6).all() and int(ev.success.sum()) == 5 and not ev.success.all()
    assert ev.resolution_ok.all()
    assert not sys_eval(dd, pts, ["T1", "T2"], ["c1", "c2", "c3"], "AUROC", 20, "sw").resolution_ok.all()      # 1/21 > .05/6
    evp = sys_eval(dd, pts, ["T1", "T2"], ["c1", "c2", "c3"], "AUROC", 9999, "pertf")
    assert evp[evp.tf == "T1"].m_family.eq(3).all()
    # system-keyed parts: RETINA must not absorb RETINA_NRL_3V2
    ms = [dict(system="RETINA", stem="part_RETINA_all"), dict(system="RETINA_NRL_3V2", stem="part_RETINA_NRL_3V2_all")]
    assert [m["stem"] for m in select_parts(ms, "RETINA")] == ["part_RETINA_all"]
    print("gb_prospective selftest OK")


def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--systems", required=True)
    r.add_argument("--tfs", default="")
    r.add_argument("--B", type=int, default=B_DEFAULT)
    r.add_argument("--out", required=True)
    m = sub.add_parser("merge")
    m.add_argument("--out", required=True)
    a = ap.parse_args()
    cmd_run(a) if a.cmd == "run" else cmd_merge(a)


if __name__ == "__main__":
    main()
