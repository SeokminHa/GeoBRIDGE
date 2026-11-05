#!/usr/bin/env python
"""A2 CALIBRATED BM PER-TF RE-EVALUATION (pre-declared 2026-10-03: internal design note (not shipped), section A2).  EVALUATION ONLY.
Existing frozen scores / priors / universe logic (gb_ptable sha 5db6f77d, IMPORTED, never edited) are evaluated against the null-gate-CALIBRATED BM per-perturbation responder sets of gb_breadth_truth
(results/breadth_truth/BM, GSE274113 KO panel, receipt-verified through gb_ladder.load_breadth_truth) in place of the historical pooled-cell Welch/Stouffer lattice lists
(gb_lattice_extra.resp_ko_panel).  No estimator, depth, universe rule or score is changed; ONLY the (responder, tested) sets handed to the unchanged BM_CHIPATLAS truth builder differ.

ARMS (declared here BEFORE any calibrated number was inspected; the brief fixes the truth source but not the negative rule, so the defaults below are DECLARED, not tuned)
  HIST            = the historical lattice system BM_CHIPATLAS (pooled-cell stratified-panel responders; tested = NTC detect >= 5%)            [the side-by-side reference]
  CAL_TOTAL       = PRIMARY: gb_breadth_truth TOTAL estimand; responder = RESPONDER; negatives = ALL other TESTED genes (CONFIDENT_NONRESPONDER + AMBIGUOUS), the same
                    "tested minus responders" semantics as HIST and as the K562 PRIMARY (K562_EMP_ALLNULL, declared before results in results doc s344); UNTESTED -> UNKNOWN
  CAL_TOTAL_CONF  = SENSITIVITY: as CAL_TOTAL but AMBIGUOUS excluded (confident non-responders only; the K562_EMP sensitivity convention)
  CAL_WS          = SENSITIVITY: WITHIN_STATE estimand (direct standardisation to the fixed NT cell-type composition), negatives as CAL_TOTAL
  Each arm uses the UNCHANGED BM_CHIPATLAS truth builder (gb_lattice_extra.bm_truth: same ChIP-Atlas occupancy mappability / TSS / ambiguity rules) so U, M0, M1, MR differ between arms ONLY through the response-tested set.
UNIVERSE   gb_ptable.lattice_unit: M1 = response-tested & occupancy-mappable lattice universe U, finite for all 10 cognate rows, on the frozen T7 target set (T7 = strict common finite target set over the bank
           matrices, which EXCLUDES bank-member genes: their own column is NaN in their own bank row), finite leave-focal-out priors for all 10 methods; MR = M1 & nine REF features finite (the ONE universe of every row).
           T7 coverage / bank-TF exclusion are DISCLOSED per unit (n_pos at every funnel step; responders lost to bank-TF exclusion).
VECTORS    S rows : the 10 methods' cognate rows (GeoBRIDGE production S, |Pearson|, GENIE3, GRNBoost2, RegVelo |J|, Velorama, RegDiffusion, DeepSEM, PIDC, FastTENET)
           P rows : GB P (leave-focal-out catbank prior of P_pairU) and the 9 other methods' leave-focal-out priors
METRICS    AUROC, AP-lift, tie-aware P50 (gb_canon_table.eval_rows via gb_ptable.split_eval); CIs = gene-identity resampling (primary; the audit's synchronized BlockAxis draws) and G/50 control-module block resampling
           (the frozen control-only module map); paired GB - X on the same rows and the same draws; two-sided bootstrap p (+1 rule).  Descriptive: no multiplicity correction (as gb_ptable Table 1).
GATE       a TF whose MR universe has < 20 responders or < 20 non-responders (gb_layer_lattice.gate_of: also >= 20 resampling units) is reported GATE_FAILED (descriptive numbers kept, never macro-averaged, never dropped).
MACRO      equal-weight mean over GATED TFs per draw (conditional on the fixed TF set), per arm; plus the like-for-like set gated under BOTH HIST and CAL_TOTAL.
SELFTEST   on historical truth the GB S AUROC of BM_CHIPATLAS/RUNX1 must reproduce the lattice table (layer_lattice_extra_sota/BM_CHIPATLAS/endpoints.tsv, ALL6+SOTA scope) to <= 1e-9; the truth-free vectors of every
           calibrated arm must be bit-identical to HIST (only the truth differs); the HIST arm must reproduce results/real_uniform/ptable/full_20261002 P / prior / GB S rows (merge, fail closed).

  python scratch/gb_calbm_reeval.py selftest [--data]
  python scratch/gb_calbm_reeval.py run --tfs RUNX1,GATA1 --B 200 --out DIR          # small-N (2 TFs) first
  python scratch/gb_calbm_reeval.py run --B 2000 --out DIR [--tfs ...] [--slice i --nslice n]
  python scratch/gb_calbm_reeval.py merge --out DIR
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import glob
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = __import__("os").environ["GEOBRIDGE_ROOT"]
for _p in (str(HERE), f"{R}/scratch"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PREDECL = f"{R}/internal_design_note_not_shipped_NULLP.md"
PREDECL_SHA256 = "e8afeeedf73274a16e7a43991532ee478538218a87997c410c35450481806b9e"
PT_SHA8 = "5db6f77d"
PT_FULL = f"{R}/results/real_uniform/ptable/full_20261002"
BM_TRUTH_DIR = f"{R}/results/breadth_truth/BM"
GB = "GeoBRIDGE"
MET = ("AUROC", "APLIFT", "P50")
SCHEMES = ("g", "m50")                                    # gene-identity (primary) | G/50 control-module block
MIN_CLASS = 20
PANEL19 = ["ATF4", "BCL11A", "FOSL1", "GATA1", "GATA2", "GFI1B", "IRF1", "IRF9", "KLF1", "LDB1", "LMO2", "MYB", "NFE2", "PRDM16", "RUNX1", "SNAI2", "SOX12", "SPI1", "TAL1"]   # perturbations.tsv of results/breadth_truth/BM

ARMS = {
    "HIST": dict(system="BM_CHIPATLAS", est=None, rule=None, role="REFERENCE: historical pooled-cell stratified-panel responders (no null gate)"),
    "CAL_TOTAL": dict(system="BM_CHIPATLAS_CALTOT", est="TOTAL", rule="ALLNULL", role="PRIMARY: calibrated TOTAL; negatives = all tested non-responders"),
    "CAL_TOTAL_CONF": dict(system="BM_CHIPATLAS_CALTOT_CONF", est="TOTAL", rule="CONFNON", role="SENSITIVITY: calibrated TOTAL; AMBIGUOUS excluded (confident non-responders only)"),
    "CAL_WS": dict(system="BM_CHIPATLAS_CALWS", est="WITHIN_STATE", rule="ALLNULL", role="SENSITIVITY: calibrated WITHIN_STATE; negatives = all tested non-responders"),
}
CAL_ARMS = [a for a, c in ARMS.items() if c["est"]]


def _sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def atomic_tsv(df, p):
    tmp = f"{p}.tmp{os.getpid()}"
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, p)


def atomic_json(p, obj):
    tmp = f"{p}.tmp{os.getpid()}"
    json.dump(obj, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, p)


SCRIPT_SHA0 = _sha(__file__)


def preflight():
    """fail closed: the pre-declaration on disk is the one this script was written against; gb_ptable is the pinned version; imports only."""
    import gb_ptable as PT
    if __import__("os").environ.get("GB_SKIP_PREFLIGHT") == "1":      # opt-in (public release): skip only the brief/sha pins
        print("preflight skipped (public release)")
        return PT
    if not os.path.exists(PREDECL) or _sha(PREDECL) != PREDECL_SHA256:
        raise SystemExit(f"REFUSING: pre-declaration {PREDECL} missing or its sha256 != {PREDECL_SHA256}")
    if not PT.SCRIPT_SHA0.startswith(PT_SHA8):
        raise SystemExit(f"REFUSING: gb_ptable.py sha {PT.SCRIPT_SHA0[:8]} != pinned {PT_SHA8}")
    return PT


# ===================================================================================================== calibrated truth -> the UNCHANGED BM_CHIPATLAS truth builder
_TR = {}


def cal_truth(est):
    """receipt-verified calibrated matrices (gb_ladder.load_breadth_truth: receipt hashes, per_pair reproduces per_gene) and the null-gate verdict; fail closed unless CALIBRATED."""
    if est not in _TR:
        import gb_ladder as LD
        tr = LD.load_breadth_truth("BM", est, "BM19")
        if tr["calibration"] != "CALIBRATED":
            raise SystemExit(f"REFUSING: null-gate verdict of {est} is {tr['calibration']}")
        _TR[est] = tr
    return _TR[est]


def cal_sets(tr, tf, rule):
    """(responder set, tested set) of UPPER-case symbols for perturbation `tf`.  ALLNULL: tested = every testable gene of that perturbation (negatives = tested non-responders);
    CONFNON: tested = RESPONDER or CONFIDENT_NONRESPONDER (AMBIGUOUS dropped to UNKNOWN).  Symbols colliding after upper-casing are UNKNOWN."""
    perts = np.char.upper(np.asarray(tr["perts"]).astype(str))
    w = np.flatnonzero(perts == tf.upper())
    if not len(w):
        raise KeyError(f"no calibrated perturbation {tf}")
    p = int(w[0])
    Tt, Rr, CN = np.asarray(tr["T"][p], bool), np.asarray(tr["R"][p], bool), np.asarray(tr["CN"][p], bool)
    tested = Tt if rule == "ALLNULL" else (Rr | CN)
    up = np.char.upper(np.asarray(tr["genes"]).astype(str))
    dup = set(pd.Series(up)[pd.Series(up).duplicated(keep=False)])
    resp, tst = set(up[Rr & tested]) - dup, set(up[tested]) - dup
    return resp & tst, tst


def truth_cal(tf, genes, est, rule):
    import gb_layer_lattice as L
    import gb_lattice_extra as LE
    tr = cal_truth(est)
    try:
        resp, tested = cal_sets(tr, tf, rule)
    except KeyError as e:
        raise L.Blocked(f"NO_CALIBRATED_PERTURBATION_FOR_{tf}: {e}")
    mf = dict(response=f"gb_breadth_truth BM {est} (null gate CALIBRATED), rule {rule}: responder = RESPONDER; tested = "
                       + ("all testable genes" if rule == "ALLNULL" else "RESPONDER or CONFIDENT_NONRESPONDER (AMBIGUOUS excluded)") + "; UNTESTED -> UNKNOWN",
              files=[L.frec(f"{BM_TRUTH_DIR}/receipt.json"), L.frec(f"{BM_TRUTH_DIR}/per_pair.tsv.gz"), L.frec(f"{BM_TRUTH_DIR}/per_gene.tsv")],
              calibrated_truth_receipt_sha256=tr["truth_receipt_sha256"], calibration=tr["calibration"], efficacy_status=tr["efficacy_status"])
    return LE.bm_truth(genes, resp, tested, LE.chipatlas_occ(tf, LE.bm_ctx()["tss"]), mf)


def register_arms():
    import gb_lattice_extra as LE
    for arm in CAL_ARMS:
        c = ARMS[arm]
        LE.EXTRA[c["system"]] = dict(block="BONEMARROW", axis="BONEMARROW", tfs=list(LE.CA_TFS), base="BONEMARROW", win=LE.WIN_BM, role=c["role"],
                                     truth=(lambda e, r: (lambda tf, g: truth_cal(tf, g, e, r)))(c["est"], c["rule"]))
    LE.register()


# ===================================================================================================== evaluation
def vecs(u):
    import gb_attribution as A
    S = {m: np.asarray(u["V"][m], float) for m in A.METHODS}
    P = {GB: np.asarray(u["P"], float), **{m: np.asarray(u["others"][m], float) for m in A.METHODS if m != GB}}
    return {"S": S, "P": P}


def evaluate(PT, arm, u, dr, mask, mkind, tf):
    y, gi, B = u["y"], u["gi"], dr.B
    yy = np.asarray(y, bool)[mask]
    n, pos = int(mask.sum()), int(yy.sum())
    gate = bool(u["gate"][mkind])
    base = dict(arm=arm, system=u["system"], mask=mkind, tf=tf, n_rows=n, n_pos=pos, n_neg=n - pos, PREV=pos / max(n, 1), gate=gate)
    if not (pos and n - pos):
        return [dict(base, status="TOO_FEW")], [], {}
    status = "OK" if gate else "GATE_FAILED"
    res = {(k, m): PT.split_eval(v, y, mask, gi, dr.C, B) for k, vs in vecs(u).items() for m, v in vs.items()}
    rows, prow, draws = [], [], {}
    for (kind, m), (pt, bs) in res.items():
        r = dict(base, status=status, kind=kind, method=m)
        for k in MET:
            r[k] = pt[k]
            for sch, b in zip(SCHEMES, bs):
                r[f"{k}_{sch}_lo"], r[f"{k}_{sch}_hi"] = PT.pct_ci(b[k])
                draws[f"{arm}|{mkind}|{tf}|{kind}|{m}|{k}|{sch}"] = b[k].astype(np.float32)
        rows.append(r)
        if m == GB:
            continue
        gpt, gbs = res[(kind, GB)]
        q = dict(base, status=status, kind=kind, method=m)
        for k in MET:
            q[f"d_{k}"] = gpt[k] - pt[k]
            for sch, bg, bx in zip(SCHEMES, gbs, bs):
                dd = bg[k] - bx[k]
                q[f"d_{k}_{sch}_lo"], q[f"d_{k}_{sch}_hi"] = PT.pct_ci(dd)
                q[f"p_{k}_{sch}"] = PT.boot_p(dd)
        prow.append(q)
    return rows, prow, draws


def funnel(u, tr, tf, rule):
    """disclosure: responders at every step of the universe funnel, T7 coverage and the bank-TF exclusion"""
    import gb_attribution as A
    ep, bank = u["ep"], u["bank"]
    genes = np.asarray(u["genes"]).astype(str)
    up = np.char.upper(genes)
    resp = np.asarray(ep["resp"], bool)
    ai = np.array([bank.ai.get(g, -1) for g in genes])
    tc = np.where(ai >= 0, bank.tcol[np.where(ai >= 0, ai, 0)], -1)
    isb = np.isin(up, bank.mu)
    M0, M1, MR, MRR, U = u["M0"], u["M1"], u["MR"], u["MRR"], u["U"]
    d = dict(n_axis=len(genes), n_resp_tested_axis=int(len(genes) - ep["meta"]["n_resp_unknown"]), n_U=int(U.sum()), n_M0=int(M0.sum()), n_M1=int(M1.sum()), n_MR=int(MR.sum()), n_MRR=int(MRR.sum()),
             n_pos_resp_tested_axis=int(resp.sum()), n_pos_U=int((resp & U).sum()), n_pos_M0=int((resp & M0).sum()), n_pos_M1=int((resp & M1).sum()), n_pos_MR=int((resp & MR).sum()),
             n_pos_MRR=int((resp & MRR).sum()), n_bank=len(bank.members), n_axis_genes_that_are_bank_TFs=int(isb.sum()), n_axis_in_T7=int((tc >= 0).sum()),
             n_axis_bankTF_in_T7=int((isb & (tc >= 0)).sum()), n_pos_M0_bankTF_excluded_from_T7=int((resp & M0 & isb & (tc < 0)).sum()),
             n_pos_M0_not_T7_nonbank=int((resp & M0 & ~isb & (tc < 0)).sum()), n_pos_M0_not_M1_other=int((resp & M0 & ~M1 & (tc >= 0)).sum()),
             n_U_bankTF=int((U & isb).sum()), n_M0_bankTF=int((M0 & isb).sum()), n_pos_U_bankTF=int((resp & U & isb).sum()), n_pos_U_not_M0=int((resp & U & ~M0).sum()),
             n_pos_U_not_M0_bankTF=int((resp & U & ~M0 & isb).sum()), n_pos_U_not_M0_nonbank=int((resp & U & ~M0 & ~isb).sum()))
    if tr is not None:
        perts = np.char.upper(np.asarray(tr["perts"]).astype(str))
        p = int(np.flatnonzero(perts == tf.upper())[0])
        d.update(n_testable_panel=int(np.asarray(tr["T"][p]).sum()), n_resp_panel=int(np.asarray(tr["R"][p]).sum()), n_confnon_panel=int(np.asarray(tr["CN"][p]).sum()))
    return d


def s_lattice_parity(u, tf):
    """GB S AUROC on the ALL6+SOTA scope (= M0 of the unit) vs the lattice table; fail closed above 1e-9.  Only meaningful on historical truth."""
    import gb_attribution as A
    import gb_audit_decomp as AD
    y, M0 = u["y"], u["M0"]
    now = float(AD.auroc_rows(np.asarray(u["S"], float)[None, M0], np.asarray(y, bool)[M0])[0]) if y[M0].any() and (~y[M0]).any() else np.nan
    ref = A.lattice_ref_auroc("BM_CHIPATLAS", tf)
    out = dict(S_AUROC_M0_now=now, S_AUROC_lattice_table=ref, absdiff=(abs(now - ref) if ref is not None and np.isfinite(now) else None))
    if out["absdiff"] is not None and out["absdiff"] > 1e-9:
        raise SystemExit(f"REFUSING_PARITY BM_CHIPATLAS/{tf}: GB S AUROC {now!r} != lattice table {ref!r}")
    return out


def truth_free_equal(u0, u1):
    """the truth-free vectors of two units of the same TF must be bit-identical (only the truth differs between arms)"""
    ok = True
    for k, a in vecs(u0).items():
        for m, v in a.items():
            ok &= bool(np.array_equal(v, vecs(u1)[k][m], equal_nan=True))
    ok &= bool(np.array_equal(u0["Xr"], u1["Xr"], equal_nan=True)) and all(np.array_equal(u0["comps"][c], u1["comps"][c], equal_nan=True) for c in u0["comps"])
    ok &= bool(np.array_equal(np.asarray(u0["genes"]).astype(str), np.asarray(u1["genes"]).astype(str)))
    return ok


def overlap(u0, u1):
    """HIST vs calibrated responder overlap inside the intersection of the two MR universes (descriptive)"""
    m = u0["MR"] & u1["MR"]
    a, b = np.asarray(u0["y"], bool)[m], np.asarray(u1["y"], bool)[m]
    both, uni = int((a & b).sum()), int((a | b).sum())
    return dict(n_common_MR=int(m.sum()), pos_hist=int(a.sum()), pos_cal=int(b.sum()), pos_both=both, jaccard=(both / uni if uni else np.nan))


# ===================================================================================================== run (one TF = one job: all arms)
def run_tf(PT, tf, dr, a, out):
    import gb_layer_lattice as L
    t0 = time.time()
    mrows, prows, draws, meta, units = [], [], {}, dict(tf=tf, arms={}), {}
    for arm, c in ARMS.items():
        t1 = time.time()
        try:
            u = PT.lattice_unit(c["system"], tf, dr.gene)
        except (L.Blocked, SystemExit) as e:
            if isinstance(e, SystemExit) and "REFUSING_PARITY" in str(e):
                raise
            meta["arms"][arm] = dict(status=f"NOT_RUN: {type(e).__name__}: {e}")
            print(f"[calbm] {tf}/{arm}: NOT_RUN {e}", flush=True)
            continue
        units[arm] = u
        info = dict(status="RUN", system=c["system"], role=c["role"], gate=u["gate"], **funnel(u, cal_truth(c["est"]) if c["est"] else None, tf, c["rule"]))
        if arm == "HIST":
            info["parity_S_vs_lattice"] = s_lattice_parity(u, tf)
        for mk, mask in (("MR", u["MR"]), ("resp_only", u["MRR"])):
            r_, q_, d_ = evaluate(PT, arm, u, dr, mask, mk, tf)
            mrows += r_
            prows += q_
            draws.update(d_)
        meta["arms"][arm] = info
        print(f"[calbm] {tf}/{arm}: U={info['n_U']} M1={info['n_M1']} MR={info['n_MR']} pos(MR)={info['n_pos_MR']} gate={u['gate']['MR']} {time.time() - t1:.0f}s", flush=True)
    if "HIST" in units:
        for arm in CAL_ARMS:
            if arm in units:
                meta["arms"][arm]["truth_free_vectors_identical_to_HIST"] = truth_free_equal(units["HIST"], units[arm])
                meta["arms"][arm]["overlap_with_HIST_on_common_MR"] = overlap(units["HIST"], units[arm])
                if not meta["arms"][arm]["truth_free_vectors_identical_to_HIST"]:
                    raise SystemExit(f"REFUSING: truth-free vectors of {arm}/{tf} differ from HIST")
    stem = f"part_{tf}"
    if mrows:
        atomic_tsv(pd.DataFrame(mrows), f"{out}/{stem}.metrics.tsv")
        atomic_tsv(pd.DataFrame(prows), f"{out}/{stem}.paired.tsv")
        tmp = f"{out}/{stem}.draws.tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, **draws)
        os.replace(tmp, f"{out}/{stem}.draws.npz")
    atomic_json(f"{out}/{stem}.meta.json", dict(meta, B=a.B, draws_fp=dr.fp, schemes=list(SCHEMES), predeclaration_sha256=PREDECL_SHA256, script_sha256=SCRIPT_SHA0, gb_ptable_sha256=PT.SCRIPT_SHA0,
                                                dep_shas=PT.dep_shas(), host=platform.node(), python=sys.version.split()[0], seconds=round(time.time() - t0, 1),
                                                created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


def cmd_run(a):
    PT = preflight()
    import gb_layer_lattice as L
    register_arms()
    os.makedirs(a.out, exist_ok=True)
    tfs = [t.strip().upper() for t in a.tfs.split(",") if t.strip()] or PANEL19
    bad = [t for t in tfs if t not in PANEL19]
    if bad:
        raise SystemExit(f"REFUSING: {bad} are not calibrated BM perturbations {PANEL19}")
    if a.nslice > 1:
        tfs = [t for i, t in enumerate(tfs) if i % a.nslice == a.slice]
    dr, _ = PT.lattice_draws("BM_CHIPATLAS", a.B, sizes=(50,))
    assert dr.schemes == ("g", "m50")
    for tf in tfs:
        run_tf(PT, tf, dr, a, a.out)
    assert _sha(__file__) == SCRIPT_SHA0, "script changed during the run"
    print(f"[calbm] done {tfs}", flush=True)


# ===================================================================================================== merge: ranks, macro, side-by-side, parity vs the existing ptable run
def rank_of(vals, name=GB):
    v = vals.dropna()
    if name not in v.index:
        return np.nan, np.nan, ""
    return int(1 + (v > v[name]).sum()), int((v == v[name]).sum() - 1), str(v.idxmax())


def rank_table(M):
    rows = []
    for (arm, mk, tf, kind), g in M.groupby(["arm", "mask", "tf", "kind"], sort=False):
        r = dict(arm=arm, mask=mk, tf=tf, kind=kind, n_methods=len(g), gate=bool(g.gate.iloc[0]), n_pos=int(g.n_pos.iloc[0]), n_neg=int(g.n_neg.iloc[0]))
        for k in MET:
            rk, ties, best = rank_of(g.set_index("method")[k])
            r[f"{k}_rank"], r[f"{k}_n_tied_with_GB"], r[f"{k}_best_method"] = rk, ties, best
            r[f"{k}_best_value"] = float(g[k].max())
            r[f"{k}_GB"] = float(g.set_index("method")[k].get(GB, np.nan))
        rows.append(r)
    return pd.DataFrame(rows)


def macro(PT, M, draws, tfset_of_arm, label):
    """equal-weight macro over the TFs of tfset_of_arm[arm] (per draw), metrics + paired GB - X, per arm / mask / kind"""
    out_m, out_p = [], []
    for arm, tfs in tfset_of_arm.items():
        if not tfs:
            continue
        for mk in ("MR", "resp_only"):
            for kind in ("S", "P"):
                sub = M[(M.arm == arm) & (M["mask"] == mk) & (M.kind == kind) & M.tf.isin(tfs)]
                if sub.tf.nunique() != len(tfs):
                    continue
                store = {}
                for m in sub.method.unique():
                    row = dict(tfset=label, arm=arm, mask=mk, kind=kind, method=m, n_tfs=len(tfs), tfs=",".join(sorted(tfs)))
                    for k in MET:
                        row[k] = float(sub[(sub.method == m)][k].mean())
                        for sch in SCHEMES:
                            md = np.mean(np.vstack([draws[f"{arm}|{mk}|{t}|{kind}|{m}|{k}|{sch}"] for t in sorted(tfs)]), 0)
                            store[(m, k, sch)] = md
                            row[f"{k}_{sch}_lo"], row[f"{k}_{sch}_hi"] = PT.pct_ci(md)
                    out_m.append(row)
                for m in sub.method.unique():
                    if m == GB:
                        continue
                    q = dict(tfset=label, arm=arm, mask=mk, kind=kind, method=m, n_tfs=len(tfs))
                    for k in MET:
                        q[f"d_{k}"] = float(sub[sub.method == GB][k].mean() - sub[sub.method == m][k].mean())
                        for sch in SCHEMES:
                            dd = store[(GB, k, sch)] - store[(m, k, sch)]
                            q[f"d_{k}_{sch}_lo"], q[f"d_{k}_{sch}_hi"] = PT.pct_ci(dd)
                            q[f"p_{k}_{sch}"] = PT.boot_p(dd)
                    out_p.append(q)
    return pd.DataFrame(out_m), pd.DataFrame(out_p)


def parity_vs_ptable_full(M):
    """HIST arm / MR vs the existing gb_ptable full run (B-independent point estimates): GB P, nine priors, GB S.  Fail closed above 1e-9."""
    f = f"{PT_FULL}/part_lat_BM_CHIPATLAS_all.metrics.tsv"
    ref = pd.read_csv(f, sep="\t", dtype={"tf": str})
    ref = ref[(ref["mask"] == "MR")]
    out = dict(file=f, sha256=_sha(f), n_cells=0, worst=0.0, n_tf=0)
    mine = M[(M.arm == "HIST") & (M["mask"] == "MR")]
    for tf in sorted(mine.tf.unique()):
        r = ref[ref.tf == tf]
        if not len(r):
            continue
        out["n_tf"] += 1
        for kind, m, name in [("S", GB, "GB S (production, reference)"), ("P", GB, "GB P")] + [("P", mm, f"prior {mm}") for mm in mine.method.unique() if mm != GB]:
            x, y_ = mine[(mine.tf == tf) & (mine.kind == kind) & (mine.method == m)], r[r.predictor == name]
            if not len(x) or not len(y_):
                continue
            for k, kr in (("AUROC", "AUROC"), ("APLIFT", "APLIFT"), ("P50", "P50")):
                d = abs(float(x[k].iloc[0]) - float(y_[kr].iloc[0]))
                out["worst"], out["n_cells"] = max(out["worst"], d), out["n_cells"] + 1
                if d > 1e-9:
                    raise SystemExit(f"REFUSING_PARITY HIST vs ptable full run {tf}/{kind}/{m}/{k}: {x[k].iloc[0]!r} vs {y_[kr].iloc[0]!r}")
    return out


def _f(x, d=3, sgn=False):
    return "  NA " if x is None or not np.isfinite(x) else (f"{x:+.{d}f}" if sgn else f"{x:.{d}f}")


def _sig(r, k="AUROC"):
    """'' / '*' (gene CI excludes 0) / '#' (gene AND G/50 module CIs exclude 0)"""
    g = (r[f"d_{k}_g_lo"] > 0) or (r[f"d_{k}_g_hi"] < 0)
    m = (r[f"d_{k}_m50_lo"] > 0) or (r[f"d_{k}_m50_hi"] < 0)
    return "#" if g and m else ("*" if g else "")


def cmd_merge(a):
    PT = preflight()
    metas = [json.load(open(f)) for f in sorted(glob.glob(f"{a.out}/part_*.meta.json"))]
    if not metas:
        raise SystemExit("REFUSING: no part files")
    if len({m["script_sha256"] for m in metas}) != 1 or len({m["B"] for m in metas}) != 1 or len({m["draws_fp"] for m in metas}) != 1:
        raise SystemExit("REFUSING_MIXED_PARTS: script sha / B / draws differ across parts")
    B = metas[0]["B"]
    M, Pp, draws = [], [], {}
    for m in metas:
        f = f"{a.out}/part_{m['tf']}.metrics.tsv"
        if not os.path.exists(f):
            continue
        M.append(pd.read_csv(f, sep="\t", dtype={"tf": str}))
        Pp.append(pd.read_csv(f.replace(".metrics.", ".paired."), sep="\t", dtype={"tf": str}))
        with np.load(f.replace(".metrics.tsv", ".draws.npz")) as z:
            for k in z.files:
                draws[k] = z[k]
    M, Pp = pd.concat(M, ignore_index=True), pd.concat(Pp, ignore_index=True)
    par = parity_vs_ptable_full(M)
    atomic_tsv(M, f"{a.out}/calbm_metrics.tsv")
    atomic_tsv(Pp, f"{a.out}/calbm_paired.tsv")
    rk = rank_table(M)
    atomic_tsv(rk, f"{a.out}/calbm_ranks.tsv")
    # gate / funnel / overlap table (every one of the 19 calibrated perturbations: evaluated, gate-failed or NOT_RUN with the reason)
    gt = []
    for m in metas:
        for arm, info in m["arms"].items():
            r = dict(tf=m["tf"], arm=arm, status=info["status"])
            if info["status"] == "RUN":
                r.update({k: v for k, v in info.items() if k not in ("status", "gate", "role", "system", "parity_S_vs_lattice", "overlap_with_HIST_on_common_MR")})
                r["gate_MR"], r["gate_resp_only"] = info["gate"]["MR"], info["gate"]["resp_only"]
                r["verdict"] = "EVALUATED" if info["gate"]["MR"] else "GATE_FAILED"
                for k, v in (info.get("overlap_with_HIST_on_common_MR") or {}).items():
                    r[f"ovl_{k}"] = v
                r["S_parity"] = json.dumps(info.get("parity_S_vs_lattice")) if arm == "HIST" else ""
            else:
                r["verdict"] = "NOT_RUN"
            gt.append(r)
    gt = pd.DataFrame(gt)
    atomic_tsv(gt, f"{a.out}/calbm_gate_funnel.tsv")
    # macro: own gated TF sets per arm; like-for-like = gated under BOTH HIST and CAL_TOTAL
    gated = {arm: sorted(gt[(gt.arm == arm) & (gt.verdict == "EVALUATED")].tf) for arm in ARMS}
    both = sorted(set(gated["HIST"]) & set(gated["CAL_TOTAL"]))
    mm1, pm1 = macro(PT, M, draws, gated, "own_gated")
    mm2, pm2 = macro(PT, M, draws, {"HIST": both, "CAL_TOTAL": both}, "common_gated_HIST_and_CAL_TOTAL")
    MM, PM = pd.concat([mm1, mm2], ignore_index=True), pd.concat([pm1, pm2], ignore_index=True)
    atomic_tsv(MM, f"{a.out}/calbm_macro_metrics.tsv")
    atomic_tsv(PM, f"{a.out}/calbm_macro_paired.tsv")
    # side by side: HIST vs CAL_TOTAL per TF / kind / method
    sb = []
    for tf in sorted(Pp.tf.unique()):
        for kind in ("S", "P"):
            for meth in sorted(Pp[Pp.kind == kind].method.unique()):
                row = dict(tf=tf, kind=kind, method=meth)
                for arm in ("HIST", "CAL_TOTAL"):
                    q = Pp[(Pp.arm == arm) & (Pp["mask"] == "MR") & (Pp.tf == tf) & (Pp.kind == kind) & (Pp.method == meth)]
                    x = M[(M.arm == arm) & (M["mask"] == "MR") & (M.tf == tf) & (M.kind == kind) & (M.method == meth)]
                    if len(q):
                        for k in MET:
                            row[f"{arm}_d_{k}"] = q[f"d_{k}"].iloc[0]
                        row[f"{arm}_sig_AUROC"] = _sig(q.iloc[0])
                        row[f"{arm}_gate"] = bool(q.gate.iloc[0])
                    if len(x):
                        for k in MET:
                            row[f"{arm}_{k}"] = x[k].iloc[0]
                sb.append(row)
    atomic_tsv(pd.DataFrame(sb), f"{a.out}/calbm_side_by_side.tsv")
    nr = {f"{m['tf']}|{arm}": i["status"] for m in metas for arm, i in m["arms"].items() if i["status"] != "RUN"}
    rc = dict(task="A2 calibrated BM per-TF re-evaluation", predeclaration=PREDECL, predeclaration_sha256=PREDECL_SHA256, script_sha256=metas[0]["script_sha256"], gb_ptable_sha256=PT.SCRIPT_SHA0,
              B=B, schemes=list(SCHEMES), draws_fp=metas[0]["draws_fp"], arms={k: dict(v) for k, v in ARMS.items()}, panel19=PANEL19, parts=[os.path.basename(f) for f in sorted(glob.glob(f"{a.out}/part_*.meta.json"))],
              not_run=nr, gated_tfs=gated, common_gated_tfs=both, parity_hist_vs_ptable_full=par, truth_source=BM_TRUTH_DIR,
              truth_receipt_sha256=_sha(f"{BM_TRUTH_DIR}/receipt.json"), truth_receipt_null_gate=json.load(open(f"{BM_TRUTH_DIR}/receipt.json"))["null_gate_result"],
              truth_script_sha256_in_receipt=json.load(open(f"{BM_TRUTH_DIR}/receipt.json"))["script_sha256"], dep_shas=metas[0]["dep_shas"], host=platform.node(),
              python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              notes=["macro rows are conditional on the fixed gated TF set (genes resampled, TFs not); descriptive, no multiplicity correction",
                     "CALIBRATED responder lists are not conditioned on KO efficacy (EFFICACY_UNVERIFIED for all perturbations in the truth receipt)",
                     "HIST arm point estimates reproduce the existing gb_ptable full run (parity_hist_vs_ptable_full)"])
    atomic_json(f"{a.out}/receipt.json", rc)
    report(a.out, M, Pp, rk, gt, MM, PM, par, B)


def report(out, M, Pp, rk, gt, MM, PM, par, B):
    print(f"\n[A2] parity HIST vs ptable full run: worst |d| = {par['worst']:.2e} over {par['n_cells']} cells, {par['n_tf']} TFs")
    print("\n== GATE / FUNNEL (MR universe; n_pos at each funnel step)")
    cols = ["tf", "arm", "verdict", "n_testable_panel", "n_resp_panel", "n_pos_U", "n_pos_M0", "n_pos_M1", "n_pos_MR", "n_pos_M0_bankTF_excluded_from_T7", "n_MR", "ovl_jaccard"]
    g = gt[[c for c in cols if c in gt.columns]].copy()
    print(g.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print("\n== PER TF (MR): GB S rank among 10 cognate rows | GB P rank among 10 priors   [HIST -> CAL_TOTAL]")
    for tf in sorted(rk.tf.unique()):
        for arm in ("HIST", "CAL_TOTAL", "CAL_TOTAL_CONF", "CAL_WS"):
            s = rk[(rk.tf == tf) & (rk.arm == arm) & (rk["mask"] == "MR") & (rk.kind == "S")]
            p = rk[(rk.tf == tf) & (rk.arm == arm) & (rk["mask"] == "MR") & (rk.kind == "P")]
            if not len(s):
                continue
            s, p = s.iloc[0], p.iloc[0]
            print(f"{tf:6s} {arm:15s} pos/neg={s.n_pos:4d}/{s.n_neg:4d} gate={str(s.gate):5s} | S AUROC={s.AUROC_GB:.3f} rank {s.AUROC_rank}/10 (best {s.AUROC_best_method} {s.AUROC_best_value:.3f}); "
                  f"APL rank {s.APLIFT_rank}; P50 rank {s.P50_rank} | P AUROC={p.AUROC_GB:.3f} rank {p.AUROC_rank}/10 (best {p.AUROC_best_method} {p.AUROC_best_value:.3f}); APL rank {p.APLIFT_rank}; P50 rank {p.P50_rank}")
    print("\n== PER TF contrasts GB - X, AUROC [* gene CI excludes 0; # gene AND G/50 CI exclude 0]   CAL_TOTAL (HIST)")
    for tf in sorted(Pp.tf.unique()):
        for kind, lab in (("S", "S - cognate"), ("P", "P - prior")):
            parts = []
            for meth in sorted(Pp[Pp.kind == kind].method.unique()):
                c = Pp[(Pp.arm == "CAL_TOTAL") & (Pp["mask"] == "MR") & (Pp.tf == tf) & (Pp.kind == kind) & (Pp.method == meth)]
                h = Pp[(Pp.arm == "HIST") & (Pp["mask"] == "MR") & (Pp.tf == tf) & (Pp.kind == kind) & (Pp.method == meth)]
                if not len(c) and not len(h):
                    continue
                cs = f"{c.d_AUROC.iloc[0]:+.3f}{_sig(c.iloc[0])}" if len(c) else "  NA "
                hs = f"{h.d_AUROC.iloc[0]:+.3f}{_sig(h.iloc[0])}" if len(h) else "  NA "
                parts.append(f"{meth.replace('RegVelo |J|', 'RegVelo').replace('|Pearson|', 'Pearson')[:9]} {cs}({hs})")
            if parts:
                print(f"{tf:6s} {lab:11s} " + "  ".join(parts))
    print("\n== BM MACRO (MR): equal-weight over gated TFs")
    if not len(MM):
        print("   (no gated TF in any arm)")
        return
    for (ts, arm, kind), g in MM[MM["mask"] == "MR"].groupby(["tfset", "arm", "kind"], sort=False):
        ranks = {k: (1 + (g[k] > g[g.method == GB][k].iloc[0]).sum()) for k in MET}
        print(f"-- {ts} | {arm} | {kind} | TFs={g.tfs.iloc[0]} | GB rank AUROC {ranks['AUROC']}/{len(g)} APL {ranks['APLIFT']}/{len(g)} P50 {ranks['P50']}/{len(g)}")
        gg = g.sort_values("AUROC", ascending=False)
        pm = PM[(PM.tfset == ts) & (PM.arm == arm) & (PM["mask"] == "MR") & (PM.kind == kind)].set_index("method")
        for _, r in gg.iterrows():
            ln = f"   {r.method:14s} AUROC {r.AUROC:.3f} [{r.AUROC_g_lo:.3f},{r.AUROC_g_hi:.3f}]  APL {r.APLIFT:.2f}  P50 {r.P50:.2f}"
            if r.method in pm.index:
                q = pm.loc[r.method]
                ln += f" | GB-X dAUROC {q.d_AUROC:+.3f} g[{q.d_AUROC_g_lo:+.3f},{q.d_AUROC_g_hi:+.3f}] G50[{q.d_AUROC_m50_lo:+.3f},{q.d_AUROC_m50_hi:+.3f}] dAPL {q.d_APLIFT:+.2f} dP50 {q.d_P50:+.2f}"
            print(ln)


# ===================================================================================================== selftest
def selftest(a):
    # 1. cal_sets: ALLNULL vs CONFNON semantics, upper-case collisions -> UNKNOWN, perturbation lookup
    tr = dict(perts=np.array(["AAA", "BBB"]), genes=np.array(["g1", "g2", "g3", "g4", "g5", "G5"]),
              T=np.array([[1, 1, 1, 1, 0, 1], [1, 1, 1, 1, 1, 1]], bool), R=np.array([[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]], bool),
              CN=np.array([[0, 1, 1, 0, 0, 0], [1, 0, 0, 1, 0, 0]], bool))
    r, t = cal_sets(tr, "aaa", "ALLNULL")
    assert r == {"G1"} and t == {"G1", "G2", "G3", "G4"}, (r, t)                      # g5 untested for AAA -> UNKNOWN; g5/G5 collide upper-case -> dropped everywhere
    r, t = cal_sets(tr, "AAA", "CONFNON")
    assert r == {"G1"} and t == {"G1", "G2", "G3"}, (r, t)                            # g4 is AMBIGUOUS (tested, neither responder nor confident) -> UNKNOWN under CONFNON
    r, t = cal_sets(tr, "BBB", "ALLNULL")
    assert r == {"G2"} and t == {"G1", "G2", "G3", "G4"}, (r, t)                      # g5/G5 collision dropped even when tested
    try:
        cal_sets(tr, "ZZZ", "ALLNULL")
        raise AssertionError("missing perturbation not refused")
    except KeyError:
        pass
    # 2. rank_of: strict-greater rank, ties counted
    v = pd.Series({"GeoBRIDGE": 0.6, "a": 0.7, "b": 0.6, "c": 0.5})
    assert rank_of(v) == (2, 1, "a"), rank_of(v)
    # 3. rank_table / macro algebra on a toy (PT stub for pct_ci / boot_p)
    class _PT:
        pct_ci = staticmethod(lambda x: (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))))
        boot_p = staticmethod(lambda d: 1.0)
    rng = np.random.default_rng(0)
    rows, dr = [], {}
    for tf in ("T1", "T2"):
        for m, base in ((GB, 0.7), ("X", 0.5)):
            r_ = dict(arm="CAL_TOTAL", mask="MR", tf=tf, kind="S", method=m, n_rows=10, n_pos=5, n_neg=5, gate=True, status="OK")
            for k in MET:
                r_[k] = base + (0.1 if tf == "T2" else 0.0)
                for sch in SCHEMES:
                    d_ = rng.normal(r_[k], 0.01, 50)
                    dr[f"CAL_TOTAL|MR|{tf}|S|{m}|{k}|{sch}"] = d_
            rows.append(r_)
    Mt = pd.DataFrame(rows)
    mm, pm = macro(_PT, Mt, dr, {"CAL_TOTAL": ["T1", "T2"]}, "toy")
    assert abs(mm[(mm["mask"] == "MR") & (mm.method == GB)].AUROC.iloc[0] - 0.75) < 1e-12 and abs(pm[pm["mask"] == "MR"].d_AUROC.iloc[0] - 0.2) < 1e-12
    assert rank_table(Mt).AUROC_rank.eq(1).all()
    print("[calbm] selftest (synthetic) PASS")
    if getattr(a, "data", False):
        selftest_data(a)


def selftest_data(a):
    PT = preflight()
    register_arms()
    dr, _ = PT.lattice_draws("BM_CHIPATLAS", 60, sizes=(50,))
    tf = "RUNX1"
    uh = PT.lattice_unit("BM_CHIPATLAS", tf, dr.gene)
    sp = s_lattice_parity(uh, tf)
    assert sp["S_AUROC_lattice_table"] is not None and sp["absdiff"] <= 1e-9, sp
    mr_, _pr, _dd = PT.table1_rows(uh, dr, uh["MR"], "MR")
    par = PT.parity_lattice(uh, mr_)
    assert par["status"] == "PASS", par
    print(f"[calbm] HIST BM_CHIPATLAS/{tf}: GB S AUROC (ALL6+SOTA scope, M0) = {sp['S_AUROC_M0_now']!r} vs lattice table {sp['S_AUROC_lattice_table']!r}  |d|={sp['absdiff']:.2e}  "
          f"exact={sp['S_AUROC_M0_now'] == sp['S_AUROC_lattice_table']}; ptable parity worst={par['worst']:.2e} over {par['n_cells']} cells")
    for arm in CAL_ARMS:
        uc = PT.lattice_unit(ARMS[arm]["system"], tf, dr.gene)
        assert truth_free_equal(uh, uc), f"truth-free vectors differ {arm}"
        f_ = funnel(uc, cal_truth(ARMS[arm]["est"]), tf, ARMS[arm]["rule"])
        print(f"[calbm] {arm}/{tf}: n_U={f_['n_U']} n_M1={f_['n_M1']} n_MR={f_['n_MR']} n_pos(MR)={f_['n_pos_MR']} gate={uc['gate']['MR']} panel responders={f_['n_resp_panel']}/{f_['n_testable_panel']}; "
              f"overlap {overlap(uh, uc)}")
    # independent re-derivation of the calibrated truth join and of the metric (pandas straight from per_pair.tsv.gz + sklearn), GATA1 and RUNX1
    from sklearn.metrics import roc_auc_score
    pp = pd.read_csv(f"{BM_TRUTH_DIR}/per_pair.tsv.gz", sep="\t", usecols=["estimand", "pert", "gene", "status"])
    for t_ in ("GATA1", "RUNX1"):
        uc = PT.lattice_unit(ARMS["CAL_TOTAL"]["system"], t_, dr.gene)
        q = pp[(pp.estimand == "TOTAL") & (pp.pert == t_)]
        q = q[~q.gene.str.upper().duplicated(keep=False)]
        up = np.char.upper(np.asarray(uc["genes"]).astype(str))
        resp, tst = set(q.gene[q.status == "RESPONDER"].str.upper()), set(q.gene[q.status != "UNTESTED"].str.upper())
        mk = uc["MR"]
        y_ind = np.isin(up, list(resp))[mk]
        assert (y_ind == np.asarray(uc["y"], bool)[mk]).all() and np.isin(up[mk], list(tst)).all(), f"calibrated truth join differs from the independent read ({t_})"
        pt, _bs = PT.split_eval(np.asarray(uc["S"], float), uc["y"], mk, uc["gi"], dr.C, 60)
        a_ = roc_auc_score(y_ind, np.asarray(uc["S"], float)[mk])
        assert abs(a_ - pt["AUROC"]) < 1e-12, (a_, pt["AUROC"])
        print(f"[calbm] independent check {t_}: n_pos(MR)={int(y_ind.sum())} responders/tested joined from per_pair.tsv.gz identical; GB S AUROC sklearn={a_:.12f} == engine={pt['AUROC']:.12f}")
    print("[calbm] selftest (data) PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("selftest")
    p.add_argument("--data", action="store_true")
    p = sub.add_parser("run")
    p.add_argument("--tfs", default="")
    p.add_argument("--B", type=int, default=2000)
    p.add_argument("--out", required=True)
    p.add_argument("--slice", type=int, default=0)
    p.add_argument("--nslice", type=int, default=1)
    p = sub.add_parser("merge")
    p.add_argument("--out", required=True)
    a = ap.parse_args()
    {"selftest": selftest, "run": cmd_run, "merge": cmd_merge}[a.cmd](a)


if __name__ == "__main__":
    main()
