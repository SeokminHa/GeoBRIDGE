#!/usr/bin/env python
"""OBSERVATIONAL SCORE DECOMPOSITION (internal review r99 spec items 3-8; reviewer request 2026-09-29).  EVALUATION ONLY: no score / prior / feature / k is changed; truth is read only inside
gb_layer_lattice.build_endpoint after every score artifact (attribution_v3 order) and the decomposition below uses observational scores ONLY.

  For each (system, focal TF) of attribution_v3 and each of its 10 Tier-1 methods m, on the attribution mask M1 (one mask for every method / vector):
      Zf = row-z over M1 of the cognate row S_m[r,.]      Zp = row-z over M1 of the leave-focal-out own prior P_m^(-r)         (population sd, ddof = 0)
      Zf_g = a + b Zp_g + R_g   OLS, FIXED 5 gene folds (ep['fold'] = lattice hash folds): (a, b) fit on 4 folds, the held-out fold is predicted and residualised
  and FULL (= cognate row), PRIOR (= P^(-r)), FIT (= held-out a + b Zp, sign-carrying) and RESID (= held-out R) are evaluated SEPARATELY on the same labels (no truth-based sign flip).
  Additional PRIOR-type vectors on the same masks (mask = M1 and finite): frozen composites COMP_Z/COMP_RANK/COMP_PC1/COMP_FA (gb_composites, unchanged; NO_CALL absent, preserved) and
  the catbank ablation priors P_pairM / P_cellU / P_cellM (leave-focal-out, gb_catbank.loo_prior on the stored (Z, S, C)).
  CAVEAT: RESID is a STATISTICAL remainder, not a regulator-specific score: the leave-focal-out prior is a noisy proxy of the target-level axis, so RESID retains some target-level
  signal (errors-in-variables; selftest 2 uses a near-exact proxy to isolate the algebra).  It acquires a specificity reading only through the calibrated attribution audit.
  METRICS per (vector, target): AUROC, AUPRC, AP-lift, tie-aware top-50 yield P50 (gb_canon_table.weighted_metrics), targets resp / AB / CD / occ / AC / BD; gene-bootstrap 95% CI
  (the lattice's shared multinomial draws, CONDITIONAL on the fixed folds / fitted a,b).
  CONTRASTS (declared Holm family per substrate, resp AUROC): GB PRIOR minus each other method's PRIOR and minus COMP_Z / COMP_RANK, paired on the joint finite mask.  Descriptive extras (not in the
  family): COMP_PC1, COMP_FA, P_pairM, P_cellU, P_cellM.  Substrate = system (never pooled); multi-TF substrates: TF-macro over the gated TFs, TF-bootstrap 10k draws (paired across vectors); single-TF
  substrates: gene bootstrap.  Holm over the declared contrasts of the substrate.

  v2 (after internal review): (1) multi-TF substrate p = EXACT SIGN-FLIP permutation of the per-TF paired differences (all 2^n flips of the mean; two-sided primary, one-sided reported; floor 2/2^n);
  the TF bootstrap CI and the paired t are DESCRIPTIVE sensitivity columns only; n_tf == 1 keeps the (conditional) gene bootstrap.  (2) panel_dependence.tsv: responder Jaccard between the TFs of a
  substrate + composite / REF identity across TFs (TFs of one substrate share a perturbation panel, so the effective n is below n_tf and the flip test is conditional on the panel).
  (3) TRUTH-FREE REDUCTION TEST: per (system, TF) the GB prior (on M1 & the 9 REF features finite) is regressed on the 9 REF features by OLS (no labels, same fixed gene folds, cross-fitted);
  the RESIDUAL prior's response AUROC / AP-lift is tested against 0.5 / 1 with the same exact machinery, Holm over substrates.

  python scratch/gb_decompose.py --selftest
  python scratch/gb_decompose.py tfs --systems BM_CHIPATLAS --slice 0 --nslice 2
  python scratch/gb_decompose.py run --systems GATA1,PANC --B 2000 --out results/real_uniform/attribution_decomp_smalln [--tfs A,B]
  python scratch/gb_decompose.py merge --out DIR
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gb_attribution as A  # noqa: E402  (NOT edited; sha bound to attribution_v3)
import gb_canon_table as CT  # noqa: E402
import gb_catbank as CB  # noqa: E402

R = A.R
V3 = f"{R}/results/real_uniform/attribution_v3"
OUT_DEFAULT = f"{R}/results/real_uniform/attribution_decomp"
GB = A.GB
TARGETS = A.TARGETS
MET = ("AUROC", "AUPRC", "APLIFT", "P50")
COMPS = ["COMP_Z", "COMP_RANK", "COMP_PC1", "COMP_FA"]
ABL = {"P_pairM": "P_pairM", "P_cellU": "P_cellU", "P_cellM": "P_cellM"}
DECL = ["COMP_Z", "COMP_RANK"]                      # + every other method's own prior
N_TFBOOT = 10000


# ===================================================================================================== pure numerics (unit-tested by --selftest)
def zrow(v, mask):
    """row-z of v over `mask` (population sd); NaN outside the mask; constant -> refuse"""
    v = np.asarray(v, float)
    m = np.asarray(mask, bool)
    x = v[m]
    if not np.isfinite(x).all():
        raise RuntimeError("zrow: non-finite value inside the mask")
    sd = x.std()
    if not sd > 0:
        raise RuntimeError("zrow: constant vector on the mask")
    out = np.full(len(v), np.nan)
    out[m] = (x - x.mean()) / sd
    return out


def fold_decompose(zf, zp, folds, mask):
    """Zf = a + b Zp + R by OLS with FIXED gene folds: (a, b) fitted on the other folds, the held-out fold predicted and residualised.  Returns (fit, resid, stats); NaN outside `mask`."""
    zf, zp, folds, mask = np.asarray(zf, float), np.asarray(zp, float), np.asarray(folds), np.asarray(mask, bool)
    fit, res = np.full(len(zf), np.nan), np.full(len(zf), np.nan)
    ab = []
    for f in np.unique(folds[mask]):
        te, tr = mask & (folds == f), mask & (folds != f)
        if tr.sum() < 20 or te.sum() < 1:
            raise RuntimeError(f"fold_decompose: fold {f} has {tr.sum()} train / {te.sum()} test genes")
        x, y = zp[tr], zf[tr]
        vx = ((x - x.mean()) ** 2).sum()
        if not vx > 0:
            raise RuntimeError("fold_decompose: constant prior in the training folds")
        b = ((x - x.mean()) * (y - y.mean())).sum() / vx
        a = y.mean() - b * x.mean()
        fit[te] = a + b * zp[te]
        res[te] = zf[te] - fit[te]
        ab.append((a, b))
    ab = np.array(ab)
    from scipy.stats import spearmanr
    tot = ((zf[mask] - zf[mask].mean()) ** 2).sum()
    return fit, res, dict(a_mean=float(ab[:, 0].mean()), b_mean=float(ab[:, 1].mean()), b_sd_folds=float(ab[:, 1].std()), n_folds=int(len(ab)),
                          R2_heldout=float(1 - (res[mask] ** 2).sum() / tot), corr_pearson=float(np.corrcoef(zf[mask], zp[mask])[0, 1]),
                          corr_spearman=float(spearmanr(zf[mask], zp[mask])[0]))


def pct(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)), len(x)) if len(x) else (np.nan, np.nan, 0)


def boot_p(d):
    """two-sided bootstrap p of a difference: 2 min(P*(d<=0), P*(d>=0)) with the +1 rule, capped at 1; NaN if no finite draw"""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min(1 + (d <= 0).sum(), 1 + (d >= 0).sum()) / (len(d) + 1))) if len(d) else np.nan


def boot_p1(d):
    """one-sided (greater) bootstrap p: (1 + #{d <= 0}) / (n + 1)"""
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float((1 + (d <= 0).sum()) / (len(d) + 1)) if len(d) else np.nan


def eval_vec(vec, y, rows, gi, counts, null=None):
    """point metrics + gene-bootstrap 95% CI of `vec` on rows & finite(vec); None if a class is empty.  null = {metric: value}: also the two-sided / one-sided-greater gene-bootstrap p
    of each such metric against its null value (e.g. AUROC vs .5, AP-lift vs 1)"""
    vec = np.asarray(vec, float)
    rv = np.asarray(rows, bool) & np.isfinite(vec)
    yy = np.asarray(y, bool)[rv]
    if not (yy.any() and (~yy).any()):
        return None
    pt, bs = CT.eval_rows(vec, y, rv, gi, counts)
    out = dict(n=int(rv.sum()), n_pos=int(yy.sum()), n_neg=int((~yy).sum()), frac_mask=float(rv.sum() / max(int(np.asarray(rows).sum()), 1)), PREV=pt["PREV"])
    for k in MET:
        lo, hi, nb = pct(bs[k])
        out.update({k: pt[k], f"{k}_lo": lo, f"{k}_hi": hi})
    for k, v0 in (null or {}).items():
        d = np.asarray(bs[k], float) - v0
        out.update({f"p_{k}_null": boot_p(d), f"p1_{k}_null": boot_p1(d)})
    return out


def contrast(a, c, y, rows, gi, counts):
    """paired difference (a - c) on rows & finite(a) & finite(c): point, gene-bootstrap CI and p for each MET metric"""
    rj = np.asarray(rows, bool) & np.isfinite(a) & np.isfinite(c)
    yy = np.asarray(y, bool)[rj]
    if not (yy.any() and (~yy).any()):
        return None
    pa, ba = CT.eval_rows(a, y, rj, gi, counts)
    pc, bc = CT.eval_rows(c, y, rj, gi, counts)
    out = dict(n=int(rj.sum()), n_pos=int(yy.sum()), n_neg=int((~yy).sum()), frac_mask=float(rj.sum() / max(int(np.asarray(rows).sum()), 1)))
    for k in MET:
        d = ba[k] - bc[k]
        lo, hi, _ = pct(d)
        out.update({f"a_{k}": pa[k], f"c_{k}": pc[k], f"d_{k}": pa[k] - pc[k], f"d_{k}_lo": lo, f"d_{k}_hi": hi, f"p_{k}": boot_p(d), f"p1_{k}": boot_p1(d)})
    return out


def tf_boot(vals, idx):
    """TF-bootstrap of the macro mean: idx (n_boot, n_tf) resample indices.  Returns (mean, lo, hi)"""
    v = np.asarray(vals, float)
    m = v[idx].mean(1)
    return float(v.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def tf_boot_p(d, idx):
    return boot_p(np.asarray(d, float)[idx].mean(1))


def sign_test_p(d):
    from scipy.stats import binomtest
    d = np.asarray(d, float)
    d = d[np.isfinite(d) & (d != 0)]
    return float(binomtest(int((d > 0).sum()), len(d), 0.5).pvalue) if len(d) else np.nan


def sign_flip_p(d, tol=1e-12):
    """EXACT sign-flip permutation test of H0: the per-TF differences are symmetric about 0.  Statistic = mean(d); all 2^n sign vectors enumerated.  Returns dict(p_two, p_greater, n_flips,
    min_p): p_two = P(|mean*| >= |obs|), p_greater = P(mean* >= obs) (ties count against the observation, floating tolerance)."""
    d = np.asarray(d, float)
    n = len(d)
    if n < 2 or n > 20 or not np.isfinite(d).all():
        return dict(p_two=np.nan, p_greater=np.nan, n_flips=int(2 ** n) if 0 < n <= 20 else 0, min_p=np.nan)
    sg = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1) * 2.0 - 1.0
    m = sg @ d / n
    obs = d.mean()
    sc = max(np.abs(d).max(), 1e-300) * tol
    return dict(p_two=float((np.abs(m) >= abs(obs) - sc).mean()), p_greater=float((m >= obs - sc).mean()), n_flips=int(2 ** n), min_p=float(2 / 2 ** n))


def ttest_p(d):
    from scipy.stats import ttest_1samp
    d = np.asarray(d, float)
    return float(ttest_1samp(d, 0.0).pvalue) if len(d) >= 2 and np.ptp(d) > 0 else np.nan


def fold_ols(y, X, folds, mask):
    """cross-fitted multi-regressor OLS (intercept + standardized columns of X, standardisation and coefficients from the TRAINING folds only): held-out fit and residual over `mask` (NaN
    outside), plus the held-out R2 and the training design condition number.  No labels."""
    y, X, folds, mask = np.asarray(y, float), np.asarray(X, float), np.asarray(folds), np.asarray(mask, bool)
    fit, res = np.full(len(y), np.nan), np.full(len(y), np.nan)
    cond = []
    for f in np.unique(folds[mask]):
        te, tr = mask & (folds == f), mask & (folds != f)
        if tr.sum() < 5 * (X.shape[1] + 1) or te.sum() < 1:
            raise RuntimeError(f"fold_ols: fold {f} has {tr.sum()} train / {te.sum()} test genes")
        mu, sd = X[tr].mean(0), X[tr].std(0)
        sd = np.where(sd > 0, sd, 1.0)
        Atr, Ate = np.c_[np.ones(tr.sum()), (X[tr] - mu) / sd], np.c_[np.ones(te.sum()), (X[te] - mu) / sd]
        beta = np.linalg.lstsq(Atr, y[tr], rcond=None)[0]
        fit[te] = Ate @ beta
        res[te] = y[te] - fit[te]
        cond.append(float(np.linalg.cond(Atr)))
    tot = ((y[mask] - y[mask].mean()) ** 2).sum()
    return fit, res, dict(R2_heldout=float(1 - (res[mask] ** 2).sum() / tot), cond_max=float(max(cond)), n_regressors=int(X.shape[1]))


# ===================================================================================================== one endpoint
def v3_table():
    return pd.read_csv(f"{V3}/per_tf_method.tsv", sep="\t", dtype={"tf": str})


def v3_tfs(system):
    d = v3_table()
    d = d[(d.system == system) & d.status.isin(["OK", "DESCRIPTIVE_ONLY", "TOO_FEW"])]
    return sorted(d.tf.unique())


_PZ = {}


def ablation_priors(axis, bank, r, tc, ai0):
    """leave-focal-out priors of P_pairM / P_cellU / P_cellM on the frozen T7 (loo_prior on the stored Z, S, C), mapped to the lattice genes"""
    if axis not in _PZ:
        with np.load(f"{CB.OUT}/{axis}_priors.npz", allow_pickle=True) as z:
            _PZ[axis] = {n: (np.asarray(z[f"T7|{n}|Z"]), np.asarray(z[f"T7|{n}|S"]), np.asarray(z[f"T7|{n}|C"])) for n in ABL}
    out = {}
    tcz = np.where(tc >= 0, tc, 0)
    for n, (Z, S, C) in _PZ[axis].items():
        v = CB.loo_prior(Z, S, C, r)
        out[n] = np.where(tc >= 0, v[tcz], np.nan)
    return out


def run_tf(system, tf, blk, B, L):
    cfg = L.SYSTEMS[system]
    bank = A.load_bank(cfg["axis"])
    ep = L.build_endpoint(system, tf, blk, False)
    V, genes = ep["V"], np.asarray(ep["genes"]).astype(str)
    use = [m for m in A.METHODS if m in V and m in bank.X]
    if GB not in use:
        raise L.Blocked(f"GB_ROW_OR_BANK_MISSING {system}/{tf}")
    ai = np.array([bank.ai.get(g, -1) for g in genes])
    has = ai >= 0
    ai0 = np.where(has, ai, 0)
    tc = np.where(has, bank.tcol[ai0], -1)
    r = bank.row_of(tf)
    if r is None:
        raise L.Blocked(f"FOCAL_NOT_IN_BANK {system}/{tf}")
    M0 = np.asarray(ep["U"], bool) & np.all([np.isfinite(V[m]) for m in use], 0)
    prior = {m: np.where(tc >= 0, bank.prior(m, r)[ai0], np.nan) for m in use}
    M1 = M0 & (tc >= 0) & np.all([np.isfinite(prior[m]) for m in use], 0)
    # ---- parity with attribution_v3: same mask size, same GB cognate AUROC on M1
    d3 = v3_table()
    d3 = d3[(d3.system == system) & (d3.tf == tf) & (d3.method == GB) & (d3.target == "resp") & (d3.metric == "AUROC")]
    if len(d3) != 1 or int(d3.n_M1.iloc[0]) != int(M1.sum()):
        raise SystemExit(f"REFUSING_PARITY_MASK {system}/{tf}: n_M1 now {int(M1.sum())} vs attribution_v3 {d3.n_M1.tolist()}")
    a_now = A.point(V[GB], ep["resp"], M1)["AUROC"]
    if abs(a_now - float(d3.cognate.iloc[0])) > 1e-9:
        raise SystemExit(f"REFUSING_PARITY_AUROC {system}/{tf}: GB cognate resp AUROC {a_now!r} != attribution_v3 {float(d3.cognate.iloc[0])!r}")
    folds = np.asarray(ep["fold"])
    T = L.targets(ep, M1)
    counts = blk.counts[:B]
    gi = ep["gi"]
    # ---- vectors
    vecs, dec, nocall = [], [], []                                   # (vtype, kind, name, vector)
    for m in use:
        zf, zp = zrow(V[m], M1), zrow(prior[m], M1)
        fit, res, st = fold_decompose(zf, zp, folds, M1)
        dec.append(dict(system=system, tf=tf, method=m, n_M1=int(M1.sum()), **st))
        vecs += [("FULL", "method", m, np.where(M1, V[m], np.nan)), ("PRIOR", "method", m, np.where(M1, prior[m], np.nan)), ("FIT", "method", m, fit), ("RESID", "method", m, res)]
    for c in COMPS:
        if c in V:
            vecs.append(("PRIOR", "composite", c, np.where(M1 & np.isfinite(V[c]), V[c], np.nan)))
        else:
            nocall.append(c)
    for n, v in ablation_priors(cfg["axis"], bank, r, tc, ai0).items():
        vecs.append(("PRIOR", "ablation", n, np.where(M1 & np.isfinite(v), v, np.nan)))
    rows_out, con_out = [], []
    for tname in TARGETS:
        if tname not in T:
            continue
        y, rows = T[tname]
        gate = bool(L.gate_of(ep, y, rows, M1))
        for vt, kind, name, v in vecs:
            e = eval_vec(v, y, rows, gi, counts)
            base = dict(system=system, block=ep["block"], tf=tf, vtype=vt, kind=kind, vector=name, target=tname, gate=gate)
            rows_out.append(dict(base, status="TOO_FEW") if e is None else dict(base, status="OK" if gate else "DESCRIPTIVE_ONLY", **e))
        if tname == "resp":
            gbp = dict((n, v) for vt, k, n, v in vecs if vt == "PRIOR" and k == "method")[GB]
            for vt, kind, name, v in vecs:
                if vt != "PRIOR" or (kind == "method" and name == GB):
                    continue
                c = contrast(gbp, v, y, rows, gi, counts)
                declared = (kind == "method") or name in DECL
                base = dict(system=system, block=ep["block"], tf=tf, comparator=name, kind=kind, declared=declared, target=tname, gate=gate)
                con_out.append(dict(base, status="TOO_FEW") if c is None else dict(base, status="OK" if gate else "DESCRIPTIVE_ONLY", **c))
    # ---- (3) truth-free REDUCTION test: GB prior on the 9 REF features (no labels), evaluate the residual prior's response information
    red_out, refs = [], [n for n in L.REFS]
    y, rows = T["resp"]
    gate = bool(L.gate_of(ep, y, rows, M1))
    if all(n in V for n in refs):
        Xr = np.column_stack([np.asarray(V[n], float) for n in refs])
        mr = M1 & np.isfinite(Xr).all(1)
        zg = zrow(prior[GB], mr)
        fitr, resr, sr = fold_ols(zg, Xr, folds, mr)
        for vt, v in (("PRIOR_ON_REFMASK", np.where(mr, prior[GB], np.nan)), ("FIT_REF", fitr), ("RESID_REF", resr)):
            e = eval_vec(v, y, rows, gi, counts, null={"AUROC": 0.5, "APLIFT": 1.0})
            base = dict(system=system, block=ep["block"], tf=tf, vtype=vt, vector=GB, target="resp", gate=gate, n_refmask=int(mr.sum()), n_M1=int(M1.sum()), **sr)
            red_out.append(dict(base, status="TOO_FEW") if e is None else dict(base, status="OK" if gate else "DESCRIPTIVE_ONLY", **e))
    else:
        red_out.append(dict(system=system, tf=tf, vtype="RESID_REF", vector=GB, target="resp", status="NOT_RUN_REF_MISSING"))
    # ---- (2) shared-panel dependence inputs: responders / mask / composite + REF vectors on M1 (compared across the TFs of a substrate at merge)
    sv = {"genes": genes[M1], "resp": np.asarray(ep["resp"], bool)[M1], **{f"vec::{n}": np.asarray(V[n], float)[M1] for n in COMPS + refs if n in V}}
    meta = dict(system=system, tf=tf, methods_used=use, composites_no_call=nocall, n_M1=int(M1.sum()), n_genes=len(genes), files={k: v for k, v in ep["files"].items()},
                fold_sizes=[int((folds[M1] == f).sum()) for f in np.unique(folds[M1])])
    return rows_out, con_out, dec, red_out, sv, meta


def cmd_run(a):
    import gb_lattice_extra as LE
    import gb_layer_lattice as L
    LE.register()
    os.makedirs(a.out, exist_ok=True)
    v3sha = json.load(open(f"{V3}/receipt.json"))["script_sha256"]
    if __import__("os").environ.get("GB_SKIP_PREFLIGHT") == "1": print("preflight skipped (public release)")
    elif L.sha_file(A.__file__) != v3sha:
        raise SystemExit(f"REFUSING: gb_attribution.py sha {L.sha_file(A.__file__)[:8]} != attribution_v3 receipt {v3sha[:8]}")
    for system in [s.strip() for s in a.systems.split(",") if s.strip()]:
        if system not in A.SYSTEMS:
            raise SystemExit(f"REFUSING: {system} not in {A.SYSTEMS}")
        cfg = L.SYSTEMS[system]
        allowed = v3_tfs(system)
        tfs = [t.strip() for t in a.tfs.split(",") if t.strip()] or allowed
        bad = [t for t in tfs if t not in allowed]
        if bad:
            raise SystemExit(f"REFUSING: --tfs {bad} not among the attribution_v3 TFs of {system}: {allowed}")
        seed = int(np.random.SeedSequence([L.BOOT_SEED, L.BLOCKS.index(cfg["block"])]).generate_state(1)[0])
        blk = L.BlockAxis(L.axis_of(cfg["axis"]), a.B, seed, None, "gene")
        vec, con, dec, red, svs, metas = [], [], [], [], {}, {}
        for tf in tfs:
            t0 = time.time()
            try:
                v_, c_, d_, r_, sv_, m_ = run_tf(system, tf, blk, a.B, L)
            except (L.Blocked, SystemExit) as e:                        # fail-closed per TF (recorded in the meta / receipt, never silently dropped); other TFs continue
                metas[tf] = dict(status=f"NOT_RUN: {e}")
                print(f"[decomp] {system}/{tf}: NOT_RUN {e}", flush=True)
                continue
            vec, con, dec, red, metas[tf] = vec + v_, con + c_, dec + d_, red + r_, m_
            svs.update({f"{tf}::{k}": v for k, v in sv_.items()})
            print(f"[decomp] {system}/{tf}: M1={m_['n_M1']} nocall={m_['composites_no_call']} {time.time()-t0:.0f}s", flush=True)
        tag = "all" if not a.tfs else "_".join(tfs)
        if vec:                                                                    # a fully blocked system writes only its meta (NOT_RUN reasons); merge reports it
            for nm, rows in (("vec", vec), ("con", con), ("dec", dec), ("red", red)):
                pd.DataFrame(rows).to_csv(f"{a.out}/part_{system}_{tag}.{nm}.tsv", sep="\t", index=False)
            np.savez_compressed(f"{a.out}/part_{system}_{tag}.panel.npz", **svs)
        L.atomic_json(f"{a.out}/part_{system}_{tag}.meta.json", dict(system=system, B=a.B, seed=seed, script_sha256=L.sha_file(__file__), attribution_sha256=L.sha_file(A.__file__),
                                                                     module_shas=L.module_shas(), tfs=metas))


# ===================================================================================================== merge: substrate tables
def summarize_vectors(vec, seed_tag=""):
    """substrate x vector x target: macro over the gated TFs with TF-bootstrap (n_tf >= 2) or the TF's own gene-bootstrap CI (n_tf == 1)"""
    d = vec[vec.status == "OK"]
    out = []
    for (s, vt, kind, name, t), g in d.groupby(["system", "vtype", "kind", "vector", "target"], sort=False):
        r = dict(system=s, vtype=vt, kind=kind, vector=name, target=t, n_tf=len(g), tfs=",".join(sorted(g.tf)))
        idx = np.random.default_rng(zlib.crc32(s.encode())).integers(0, len(g), (N_TFBOOT, len(g))) if len(g) >= 2 else None
        for k in MET:
            v = g[k].to_numpy(float)
            ok = np.isfinite(v)
            if ok.sum() == 0:
                r.update({f"mean_{k}": np.nan, f"{k}_lo": np.nan, f"{k}_hi": np.nan})
            elif len(g) == 1:
                r.update({f"mean_{k}": float(v[0]), f"{k}_lo": float(g[f"{k}_lo"].iloc[0]), f"{k}_hi": float(g[f"{k}_hi"].iloc[0])})
            elif ok.all():
                m_, lo, hi = tf_boot(v, idx)
                r.update({f"mean_{k}": m_, f"{k}_lo": lo, f"{k}_hi": hi})
            else:                                                            # P50 undefined on some TF (n < 100): macro over the defined TFs only, its own resample
                ii = np.random.default_rng(zlib.crc32(s.encode())).integers(0, int(ok.sum()), (N_TFBOOT, int(ok.sum())))
                m_, lo, hi = tf_boot(v[ok], ii)
                r.update({f"mean_{k}": m_, f"{k}_lo": lo, f"{k}_hi": hi})
        out.append(r)
    return pd.DataFrame(out)


def _tf_stats(dv, idx, null_p=None, null_p1=None):
    """per-substrate inference on the per-TF differences dv (n_tf,): n >= 2 -> EXACT sign-flip permutation (primary, two-sided + one-sided greater) with the TF-bootstrap CI / bootstrap p / paired t
    as DESCRIPTIVE sensitivity columns; n == 1 -> the TF's own (conditional) gene-bootstrap p (null_p, null_p1)."""
    dv = np.asarray(dv, float)
    if len(dv) == 1:
        return dict(p_two=null_p, p_greater=null_p1, p_method="gene_bootstrap_conditional", n_flips=np.nan, min_p=np.nan, p_tfboot=np.nan, p_ttest=np.nan, tf_lo=np.nan, tf_hi=np.nan)
    if not np.isfinite(dv).all():
        return dict(p_two=np.nan, p_greater=np.nan, p_method="undefined", n_flips=np.nan, min_p=np.nan, p_tfboot=np.nan, p_ttest=np.nan, tf_lo=np.nan, tf_hi=np.nan)
    sf = sign_flip_p(dv)
    _, lo, hi = tf_boot(dv, idx)
    return dict(p_two=sf["p_two"], p_greater=sf["p_greater"], p_method="signflip_exact", n_flips=sf["n_flips"], min_p=sf["min_p"], p_tfboot=tf_boot_p(dv, idx), p_ttest=ttest_p(dv), tf_lo=lo, tf_hi=hi)


def summarize_contrasts(con):
    """declared Holm family per substrate = GB PRIOR vs every other method's PRIOR + COMP_Z + COMP_RANK on resp AUROC (paired, gated TFs only); a comparator absent (NO_CALL) on any
    gated TF of the substrate leaves the family and is REPORTED.  Every comparator (declared or not) gets its row; Holm is over the declared ones.  p_* = exact sign-flip (n_tf >= 2) or
    conditional gene bootstrap (n_tf == 1); *_tfboot / *_ttest / d_*_lo|hi (TF-bootstrap CI) are descriptive."""
    d = con[con.status == "OK"]
    out = []
    for s, gs in d.groupby("system", sort=False):
        tfs_all = sorted(gs.tf.unique())
        idx = np.random.default_rng(zlib.crc32(s.encode())).integers(0, len(tfs_all), (N_TFBOOT, len(tfs_all))) if len(tfs_all) >= 2 else None
        for (name, kind, decl), g in gs.groupby(["comparator", "kind", "declared"], sort=False):
            r = dict(system=s, comparator=name, kind=kind, declared=bool(decl), n_tf=len(g), n_tf_gated=len(tfs_all), tfs=",".join(sorted(g.tf)))
            if len(g) < len(tfs_all):
                r.update(status="NO_CALL_IN_SUBSTRATE" if len(g) == 0 else "MISSING_ON_SOME_GATED_TFS")
                out.append(r)
                continue
            r["status"] = "OK"
            g = g.set_index("tf").loc[tfs_all]
            for k in MET:
                dv = g[f"d_{k}"].to_numpy(float)
                st = _tf_stats(dv, idx, float(g[f"p_{k}"].iloc[0]) if len(g) == 1 else None, float(g[f"p1_{k}"].iloc[0]) if len(g) == 1 else None)
                if len(g) == 1:
                    lo, hi = float(g[f"d_{k}_lo"].iloc[0]), float(g[f"d_{k}_hi"].iloc[0])
                else:
                    lo, hi = st["tf_lo"], st["tf_hi"]
                r.update({f"d_{k}": float(np.mean(dv)), f"d_{k}_lo": lo, f"d_{k}_hi": hi, f"p_{k}": st["p_two"], f"p1_{k}": st["p_greater"], f"p_{k}_tfboot": st["p_tfboot"], f"p_{k}_ttest": st["p_ttest"]})
                if k == "AUROC":
                    r.update(p_method=st["p_method"], n_signflips=st["n_flips"], min_attainable_p=st["min_p"])
            r["mean_gb_AUROC"], r["mean_comp_AUROC"] = float(g["a_AUROC"].mean()), float(g["c_AUROC"].mean())
            r["n_tf_gb_better_AUROC"] = int((g["d_AUROC"] > 0).sum())
            r["sign_test_p_AUROC"] = sign_test_p(g["d_AUROC"]) if len(g) >= 2 else np.nan
            out.append(r)
    S = pd.DataFrame(out)
    S["p_holm_AUROC"] = np.nan
    for s in S.system.unique():
        f = np.flatnonzero((S.system == s) & S.declared & (S.status == "OK"))
        S.loc[S.index[f], "p_holm_AUROC"] = A.holm(S.iloc[f]["p_AUROC"].to_numpy(float))
        S.loc[S.index[f], "holm_family_size"] = len(f)
    S["p_floor_exceeds_alpha"] = S.get("min_attainable_p", pd.Series(np.nan, index=S.index)) > 0.05
    return S


def summarize_reduction(red):
    """truth-free reduction test: per substrate x vtype (PRIOR_ON_REFMASK / FIT_REF / RESID_REF): TF-macro response AUROC and AP-lift, and their test against 0.5 / 1 (n_tf >= 2: exact sign-flip of the
    per-TF (AUROC - .5) / (AP-lift - 1); n_tf == 1: conditional gene bootstrap).  Holm across substrates for the declared cell RESID_REF x AUROC (two-sided)."""
    d = red[red.status == "OK"]
    out = []
    for (s, vt), g in d.groupby(["system", "vtype"], sort=False):
        n = len(g)
        idx = np.random.default_rng(zlib.crc32(s.encode())).integers(0, n, (N_TFBOOT, n)) if n >= 2 else None
        r = dict(system=s, vtype=vt, n_tf=n, tfs=",".join(sorted(g.tf)), mean_R2_ref=float(g.R2_heldout.mean()), cond_max=float(g.cond_max.max()), n_refmask_min=int(g.n_refmask.min()))
        for k, v0 in (("AUROC", 0.5), ("APLIFT", 1.0), ("P50", np.nan)):
            vals = g[k].to_numpy(float)
            r[f"mean_{k}"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else np.nan
            if k == "P50":
                continue
            st = _tf_stats(vals - v0, idx, float(g[f"p_{k}_null"].iloc[0]) if n == 1 else None, float(g[f"p1_{k}_null"].iloc[0]) if n == 1 else None)
            lo, hi = (float(g[f"{k}_lo"].iloc[0]), float(g[f"{k}_hi"].iloc[0])) if n == 1 else (st["tf_lo"] + v0, st["tf_hi"] + v0)
            r.update({f"{k}_lo": lo, f"{k}_hi": hi, f"p_{k}": st["p_two"], f"p1_{k}": st["p_greater"], f"p_{k}_tfboot": st["p_tfboot"], f"p_{k}_ttest": st["p_ttest"], f"p_method_{k}": st["p_method"],
                      f"min_attainable_p_{k}": st["min_p"]})
        r["n_tf_above_null_AUROC"] = int((g["AUROC"] > 0.5).sum())
        out.append(r)
    S = pd.DataFrame(out)
    S["p_holm_AUROC_across_substrates"] = np.nan
    f = np.flatnonzero(S.vtype == "RESID_REF")
    S.loc[S.index[f], "p_holm_AUROC_across_substrates"] = A.holm(S.iloc[f]["p_AUROC"].to_numpy(float))
    return S


def panel_dependence(parts, gated=None):
    """TFs of one substrate share a perturbation panel: pairwise responder Jaccard on the common M1 genes, the mean pairwise phi correlation, and whether the composite / REF target vectors are
    identical across the substrate's TFs (max |diff| on the common genes; composites are refit per endpoint universe, so they can differ slightly).  `parts` = [(system, npz path)]: the
    parts of one system are pooled; `gated` = {system: set of gated TFs} restricts the pairs to the TFs that enter the substrate statistics."""
    rows = []
    zs = {}
    for sysn, f in parts:
        z = np.load(f, allow_pickle=True)
        for k in z.files:
            zs.setdefault(sysn, {})[k] = z[k]
    for sysn, z0 in zs.items():
        z = type("Z", (), {"files": list(z0), "__getitem__": lambda self, k, z0=z0: z0[k]})()
        tfs = sorted({k.split("::")[0] for k in z.files if gated is None or k.split("::")[0] in gated.get(sysn, set())})
        for i, t1 in enumerate(tfs):
            for t2 in tfs[i + 1:]:
                g1, g2 = [str(x) for x in z[f"{t1}::genes"]], [str(x) for x in z[f"{t2}::genes"]]
                c = sorted(set(g1) & set(g2))
                i1, i2 = {g: j for j, g in enumerate(g1)}, {g: j for j, g in enumerate(g2)}
                a1 = np.array([z[f"{t1}::resp"][i1[g]] for g in c], bool)
                a2 = np.array([z[f"{t2}::resp"][i2[g]] for g in c], bool)
                un = (a1 | a2).sum()
                phi = float(np.corrcoef(a1, a2)[0, 1]) if a1.any() and a2.any() and (~a1).any() and (~a2).any() else np.nan
                r = dict(system=sysn, tf1=t1, tf2=t2, n_common=len(c), n_resp1=int(a1.sum()), n_resp2=int(a2.sum()), n_both=int((a1 & a2).sum()), jaccard=float((a1 & a2).sum() / un) if un else np.nan,
                         jaccard_expected_independent=float((a1.mean() * a2.mean()) / (a1.mean() + a2.mean() - a1.mean() * a2.mean())) if len(c) else np.nan, phi=phi)
                dif = []
                for k in z.files:
                    if k.startswith(f"{t1}::vec::") and f"{t2}::{k.split('::', 1)[1]}" in z.files:
                        v1, v2 = np.asarray(z[k], float), np.asarray(z[f"{t2}::{k.split('::', 1)[1]}"], float)
                        v1c = np.array([v1[i1[g]] for g in c])
                        v2c = np.array([v2[i2[g]] for g in c])
                        dif.append((k.split("::vec::")[1], float(np.nanmax(np.abs(v1c - v2c))) if len(c) else np.nan))
                r["max_absdiff_target_vectors"] = max((x for _, x in dif), default=np.nan)
                r["target_vectors_identical"] = bool(all(x < 1e-9 for _, x in dif)) if dif else np.nan
                rows.append(r)
    return pd.DataFrame(rows)


def cmd_merge(a):
    import gb_layer_lattice as L
    mfs = sorted(glob.glob(f"{a.out}/part_*.meta.json"))
    if not mfs:
        raise SystemExit("REFUSING: no part_*.meta.json")
    metas = [json.load(open(p)) for p in mfs]
    for k in ("B", "script_sha256", "attribution_sha256"):
        if len({str(m[k]) for m in metas}) != 1:
            raise SystemExit(f"REFUSING_MERGE_MISMATCHED_{k.upper()}: {sorted({str(m[k]) for m in metas})}")
    pref = [p[:-len(".meta.json")] for p in mfs if os.path.exists(p[:-len(".meta.json")] + ".vec.tsv")]
    parts = [p + ".vec.tsv" for p in pref]
    rd = lambda suf: pd.concat([pd.read_csv(p + suf, sep="\t", dtype={"tf": str}) for p in pref], ignore_index=True)
    vec, con, dec, red = rd(".vec.tsv"), rd(".con.tsv"), rd(".dec.tsv"), rd(".red.tsv")
    for nm, df, keys in (("vec", vec, ["system", "tf", "vtype", "vector", "target"]), ("con", con, ["system", "tf", "comparator", "target"]), ("dec", dec, ["system", "tf", "method"]),
                         ("red", red, ["system", "tf", "vtype"])):
        if df.duplicated(keys).any():
            raise SystemExit(f"REFUSING_MERGE_DUPLICATE_KEYS in {nm}")
    vec.to_csv(f"{a.out}/per_tf_vectors.tsv", sep="\t", index=False)
    con.to_csv(f"{a.out}/per_tf_contrasts.tsv", sep="\t", index=False)
    dec.to_csv(f"{a.out}/decomposition_stats.tsv", sep="\t", index=False)
    red.to_csv(f"{a.out}/per_tf_reduction.tsv", sep="\t", index=False)
    SV, SC, SR = summarize_vectors(vec), summarize_contrasts(con), summarize_reduction(red)
    SV.to_csv(f"{a.out}/substrate_vectors.tsv", sep="\t", index=False)
    SC.to_csv(f"{a.out}/substrate_contrasts.tsv", sep="\t", index=False)
    SR.to_csv(f"{a.out}/substrate_reduction.tsv", sep="\t", index=False)
    sysof = {p: json.load(open(p + ".meta.json"))["system"] for p in pref}
    gated = con[con.status == "OK"].groupby("system").tf.apply(set).to_dict()
    PD = panel_dependence([(sysof[p], p + ".panel.npz") for p in pref], gated)
    PD.to_csv(f"{a.out}/panel_dependence.tsv", sep="\t", index=False)
    notrun = {m["system"]: {tf: x["status"] for tf, x in m["tfs"].items() if "status" in x} for m in metas}
    L.atomic_json(f"{a.out}/receipt.json", dict(script_sha256=L.sha_file(__file__), run_script_sha256=sorted({m["script_sha256"] for m in metas}), attribution_sha256=L.sha_file(A.__file__),
                                                parts={os.path.basename(p): L.sha_file(p) for p in parts}, B=sorted({m["B"] for m in metas}), n_vec=len(vec), n_con=len(con), n_red=len(red), n_substrate_vec=len(SV), n_substrate_con=len(SC), not_run=notrun,
                                                composites_no_call={m["system"]: {tf: x.get("composites_no_call") for tf, x in m["tfs"].items()} for m in metas},
                                                module_shas=metas[0]["module_shas"], tf_bootstrap_draws=N_TFBOOT,
                                                inference_note="Substrate p (n_tf >= 2) = EXACT sign-flip permutation of the per-TF paired differences (2^n flips, statistic = mean; two-sided primary, one-sided in p1_*); floor = 2/2^n "
                                                               "(n=4: .125, n=7: .0156), min_attainable_p / p_floor_exceeds_alpha reported.  TF-bootstrap CI / p (*_tfboot) and paired t (*_ttest) are DESCRIPTIVE "
                                                               "sensitivity only (the TF bootstrap is anti-conservative for few TFs).  n_tf == 1: conditional gene-bootstrap p.  Holm over the 11 declared contrasts per substrate.  "
                                                               "DEPENDENCE: the TFs of a substrate share one perturbation panel (BM_CHIPATLAS: GSE274113; K562: one screen / state file) and (near-)identical target-side vectors "
                                                               "(REF features exactly; composites refit per endpoint universe, max |diff| in panel_dependence.tsv; the GB prior up to the leave-focal-out term); see panel_dependence.tsv "
                                                               "(responder Jaccard vs independence). "
                                                               "The flip test is therefore CONDITIONAL on the panel and the effective number of independent units is below n_tf.",
                                                note="FULL/PRIOR/FIT/RESID evaluated on the attribution_v3 mask M1 (parity asserted per TF); residual/fit from fixed lattice gene folds; gene-bootstrap CIs are CONDITIONAL "
                                                     "on the fitted (a, b)."))
    print(f"[decomp] merged {len(parts)} parts: {len(vec)} vector rows, {len(con)} contrasts, {len(red)} reduction rows -> {a.out}")


def cmd_tfs(a):
    allt = []
    for s in a.systems.split(","):
        allt += v3_tfs(s.strip())
    sel = allt[a.slice::a.nslice]
    if not sel:
        raise SystemExit("REFUSING: empty TF slice (an empty --tfs would run ALL TFs)")
    print(",".join(sel))


# ===================================================================================================== selftest (synthetic)
def selftest():
    rg = np.random.default_rng(0)
    G, B = 1200, 300
    gi = np.arange(G)
    counts = CT.make_counts(G, B, 1)
    folds = np.array([int(hashlib.sha1(f"g{i}".encode()).hexdigest(), 16) % 5 for i in range(G)])
    mask = np.ones(G, bool)
    # 1 zrow + fold_decompose == brute force per-fold OLS; the held-out gene is never in its own fit (planted outlier stays a residual)
    zp = rg.normal(size=G)
    zf = 0.7 * zp + 0.3 * rg.normal(size=G)
    zf[3] += 8.0
    f3 = folds[3]
    fit, res, st = fold_decompose(zrow(zf, mask), zrow(zp, mask), folds, mask)
    Zf, Zp = zrow(zf, mask), zrow(zp, mask)
    for f in range(5):
        tr, te = folds != f, folds == f
        b, a_ = np.polyfit(Zp[tr], Zf[tr], 1)
        assert np.allclose(fit[te], a_ + b * Zp[te], atol=1e-10) and np.allclose(res[te], Zf[te] - fit[te], atol=1e-10)
    assert res[3] > 5 * Zf.std() * 0.5 and abs(np.nanmean(zrow(zf, mask))) < 1e-12 and abs(np.nanstd(zrow(zf, mask)) - 1) < 1e-12
    assert st["n_folds"] == 5 and 0 < st["R2_heldout"] < 1 and st["b_mean"] > 0
    # partial mask: NaN outside, folds built on the mask only
    mk = np.arange(G) % 3 != 0
    zk = zrow(zf, mk)
    f2, r2, s2 = fold_decompose(zk, zrow(zp, mk), folds, mk)
    assert np.isnan(f2[~mk]).all() and np.isfinite(f2[mk]).all() and np.isfinite(r2[mk]).all()
    # 2 PURE TARGET-LEVEL world (RESID is NOT a pure regulator-specific term when the prior is a noisy proxy of the target axis; the test uses a near-exact proxy): labels driven by t; row = lam t + noise, prior = t + noise -> RESID carries no label signal, PRIOR and FULL do
    def world(seed, a_spec):
        r_ = np.random.default_rng(seed)
        t = r_.normal(size=G)
        y = r_.random(G) < 1 / (1 + np.exp(-(1.5 * t - 0.8)))
        row = 1.0 * t + 0.8 * r_.normal(size=G) + a_spec * y
        pri = t + 0.15 * r_.normal(size=G)                 # near-exact proxy: a NOISY prior leaves target-level signal in RESID (errors-in-variables), see the docstring caveat
        fo, ro, so = fold_decompose(zrow(row, mask), zrow(pri, mask), folds, mask)
        e = {n: eval_vec(v, y, mask, gi, counts) for n, v in (("full", row), ("prior", pri), ("fit", fo), ("resid", ro))}
        return e, so
    e0, s0 = world(11, 0.0)
    assert abs(e0["resid"]["AUROC"] - 0.5) < 0.07 and e0["prior"]["AUROC"] > 0.7 and e0["full"]["AUROC"] > 0.7 and e0["fit"]["AUROC"] > 0.7, {k: v["AUROC"] for k, v in e0.items()}
    assert e0["resid"]["AUROC_lo"] < 0.5 + 1e-9 or e0["resid"]["AUROC_hi"] > 0.5 - 1e-9
    # 3 PLANTED regulator-specific increment (row = t + a y): RESID clearly above .5 and above the pure-target world
    e1, _ = world(11, 1.5)
    assert e1["resid"]["AUROC"] > 0.7 and e1["resid"]["AUROC"] > e0["resid"]["AUROC"] + 0.15, {k: v["AUROC"] for k, v in e1.items()}
    assert e1["resid"]["AUROC_lo"] > 0.5
    # 4 ANTI-PREDICTIVE prior (b < 0): FIT carries the sign (tracks the row), PRIOR keeps its raw orientation (AUROC < .5), no truth-based sign flip anywhere
    r_ = np.random.default_rng(5)
    t = r_.normal(size=G)
    y = r_.random(G) < 1 / (1 + np.exp(-(1.5 * t - 0.8)))
    pri = -t + 0.5 * r_.normal(size=G)
    row = t + 0.5 * r_.normal(size=G)
    fo, ro, so = fold_decompose(zrow(row, mask), zrow(pri, mask), folds, mask)
    assert so["b_mean"] < -0.5 and eval_vec(pri, y, mask, gi, counts)["AUROC"] < 0.35 and eval_vec(fo, y, mask, gi, counts)["AUROC"] > 0.65
    # 5 eval_vec on a partial-finite vector: mask shrinks, frac_mask < 1, NaN outside is ignored; too few classes -> None
    v = rg.normal(size=G)
    v[:100] = np.nan
    ev = eval_vec(v, y, mask, gi, counts)
    assert ev["n"] == G - 100 and abs(ev["frac_mask"] - (G - 100) / G) < 1e-12
    assert eval_vec(v, np.zeros(G, bool), mask, gi, counts) is None
    # 6 P50 top-50 yield tie-aware: a constant score yields prevalence (random tie-breaking), a perfect score yields min(1, n_pos/50)
    yy = np.zeros(G, bool)
    yy[:200] = True
    e_const = eval_vec(np.zeros(G), yy, mask, gi, counts)
    e_perf = eval_vec(yy.astype(float) + 1e-3 * np.arange(G) / G, yy, mask, gi, counts)
    assert abs(e_const["P50"] - 200 / G) < 1e-9 and e_perf["P50"] == 1.0 and abs(e_const["AUROC"] - 0.5) < 1e-12
    # 7 contrast: identical vectors -> d = 0 exactly and p = 1; a strictly better vector -> d > 0 with small p; comparator with NaNs uses the joint mask
    c0 = contrast(pri, pri, y, mask, gi, counts)
    assert c0["d_AUROC"] == 0 and c0["p_AUROC"] == 1.0
    good, bad = t + 0.3 * r_.normal(size=G), t + 3.0 * r_.normal(size=G)
    c1 = contrast(good, bad, y, mask, gi, counts)
    assert c1["d_AUROC"] > 0.05 and c1["p_AUROC"] < 0.02 and c1["d_AUROC_lo"] > 0
    bad2 = bad.copy()
    bad2[:50] = np.nan
    c2 = contrast(good, bad2, y, mask, gi, counts)
    assert c2["n"] == G - 50 and c2["frac_mask"] < 1
    # 8 TF bootstrap: paired index matrix; macro mean exact; all-positive differences -> tiny p; symmetric zero-mean differences -> p large; sign test exact
    idx = np.random.default_rng(0).integers(0, 6, (N_TFBOOT, 6))
    m_, lo, hi = tf_boot(np.arange(6.0), idx)
    assert m_ == 2.5 and lo < 2.5 < hi
    assert tf_boot_p([0.1, 0.12, 0.09, 0.11, 0.1, 0.13], idx) < 0.01 and tf_boot_p([0.1, -0.1, 0.2, -0.2, 0.05, -0.05], idx) > 0.5
    assert abs(sign_test_p([1, 1, 1, 1]) - 0.125) < 1e-12 and boot_p([0.0] * 10) == 1.0
    # 9 substrate summaries end-to-end on a tiny synthetic table: macro, Holm family only over declared+OK comparators, NO_CALL comparator leaves the family
    rows, crow = [], []
    for tf in ("t1", "t2", "t3"):
        for vt, kind, nm, au in (("PRIOR", "method", GB, .7), ("RESID", "method", GB, .5)):
            rows.append(dict(system="S", tf=tf, vtype=vt, kind=kind, vector=nm, target="resp", status="OK", **{k: (au if k == "AUROC" else 1.0) for k in MET},
                             **{f"{k}_lo": 0.0 for k in MET}, **{f"{k}_hi": 2.0 for k in MET}))
        for cn, kind, decl in (("Velorama", "method", True), ("COMP_Z", "composite", True), ("COMP_PC1", "composite", False)):
            crow.append(dict(system="S", tf=tf, comparator=cn, kind=kind, declared=decl, target="resp", status="OK", a_AUROC=.7, c_AUROC=.6,
                             **{f"d_{k}": 0.1 for k in MET}, **{f"d_{k}_lo": 0.0 for k in MET}, **{f"d_{k}_hi": 0.2 for k in MET}, **{f"p_{k}": 0.01 for k in MET}, **{f"p1_{k}": 0.005 for k in MET}))
        if tf != "t3":
            crow.append(dict(system="S", tf=tf, comparator="COMP_RANK", kind="composite", declared=True, target="resp", status="OK", a_AUROC=.7, c_AUROC=.6,
                             **{f"d_{k}": 0.1 for k in MET}, **{f"d_{k}_lo": 0.0 for k in MET}, **{f"d_{k}_hi": 0.2 for k in MET}, **{f"p_{k}": 0.01 for k in MET}, **{f"p1_{k}": 0.005 for k in MET}))
    SV = summarize_vectors(pd.DataFrame(rows))
    assert len(SV) == 2 and abs(SV[SV.vtype == "PRIOR"].mean_AUROC.iloc[0] - .7) < 1e-12 and (SV.n_tf == 3).all()
    SC = summarize_contrasts(pd.DataFrame(crow))
    assert set(SC[SC.status == "OK"].comparator) == {"Velorama", "COMP_Z", "COMP_PC1"} and (SC[SC.comparator == "COMP_RANK"].status == "MISSING_ON_SOME_GATED_TFS").all()
    okd = SC[(SC.status == "OK") & SC.declared]
    assert len(okd) == 2 and (okd.holm_family_size == 2).all() and np.isfinite(okd.p_holm_AUROC).all() and SC[SC.comparator == "COMP_PC1"].p_holm_AUROC.isna().all()
    # 10 exact sign-flip: brute force on n = 3; all-positive -> two-sided 2/8, one-sided 1/8; symmetric -> p = 1; n = 7 floor 2/128; signs of ties; n < 2 undefined
    sf = sign_flip_p([0.1, 0.2, 0.3])
    assert sf["p_two"] == 0.25 and sf["p_greater"] == 0.125 and sf["n_flips"] == 8 and sf["min_p"] == 0.25
    assert sign_flip_p([0.1, -0.1, 0.2, -0.2])["p_two"] == 1.0 and abs(sign_flip_p(np.ones(7))["p_two"] - 2 / 128) < 1e-15 and np.isnan(sign_flip_p([0.3])["p_two"])
    dv = np.array([0.5, -0.1, 0.3, 0.2])
    brute = np.mean([abs(np.mean(np.array(sg_) * dv)) >= abs(dv.mean()) - 1e-12 for sg_ in __import__("itertools").product([-1, 1], repeat=4)])
    assert abs(sign_flip_p(dv)["p_two"] - brute) < 1e-15
    # 11 fold_ols == brute-force per-fold OLS on standardised training columns; held-out; no labels
    Xo = rg.normal(size=(G, 4))
    yo = Xo @ np.array([1.0, -0.5, 0.0, 0.3]) + 0.5 * rg.normal(size=G)
    fo_, ro_, so_ = fold_ols(yo, Xo, folds, mask)
    for f in range(5):
        tr, te = folds != f, folds == f
        mu_, sd_ = Xo[tr].mean(0), Xo[tr].std(0)
        bb = np.linalg.lstsq(np.c_[np.ones(tr.sum()), (Xo[tr] - mu_) / sd_], yo[tr], rcond=None)[0]
        assert np.allclose(fo_[te], np.c_[np.ones(te.sum()), (Xo[te] - mu_) / sd_] @ bb, atol=1e-9)
    assert 0.6 < so_["R2_heldout"] < 0.9 and so_["n_regressors"] == 4
    # 12 REDUCTION world: prior = linear function of the features + noise, labels driven by the features only -> residual prior carries NO label signal while the prior does;
    #     a prior with an extra label-related component keeps a residual AUROC well above .5
    def red_world(extra, seed):
        r_ = np.random.default_rng(seed)
        Xf = r_.normal(size=(G, 9))
        t = Xf @ r_.normal(size=9)
        yl = r_.random(G) < 1 / (1 + np.exp(-(1.2 * t / t.std() - 0.8)))
        pr = t / t.std() + 0.3 * r_.normal(size=G) + extra * yl
        fi, re, _ = fold_ols(zrow(pr, mask), Xf, folds, mask)
        return eval_vec(pr, yl, mask, gi, counts, null={"AUROC": .5, "APLIFT": 1.0}), eval_vec(re, yl, mask, gi, counts, null={"AUROC": .5, "APLIFT": 1.0})
    e_pr0, e_re0 = red_world(0.0, 21)
    e_pr1, e_re1 = red_world(1.2, 21)
    assert e_pr0["AUROC"] > 0.7 and abs(e_re0["AUROC"] - 0.5) < 0.06 and e_re0["p_AUROC_null"] > 0.05 and e_re1["AUROC"] > 0.68 and e_re1["p_AUROC_null"] < 0.01 and e_re1["p1_AUROC_null"] < 0.005, (e_pr0, e_re0, e_re1)
    # 13 summarize_reduction / _tf_stats: n = 3 -> exact sign-flip; n = 1 -> gene-bootstrap p; Holm across substrates on RESID_REF only
    rr = []
    for sysn, ntf in (("S3", 3), ("S1", 1)):
        for k_ in range(ntf):
            for vt, au in (("RESID_REF", 0.56), ("FIT_REF", 0.6), ("PRIOR_ON_REFMASK", 0.62)):
                rr.append(dict(system=sysn, tf=f"t{k_}", vtype=vt, status="OK", AUROC=au, AUPRC=.2, APLIFT=1.3, P50=.2, AUROC_lo=.5, AUROC_hi=.6, APLIFT_lo=1.0, APLIFT_hi=1.6, R2_heldout=.3, cond_max=5.0,
                               n_refmask=1000, p_AUROC_null=.01, p1_AUROC_null=.005, p_APLIFT_null=.02, p1_APLIFT_null=.01))
    SR = summarize_reduction(pd.DataFrame(rr))
    x3 = SR[(SR.system == "S3") & (SR.vtype == "RESID_REF")].iloc[0]
    x1 = SR[(SR.system == "S1") & (SR.vtype == "RESID_REF")].iloc[0]
    assert x3.p_method_AUROC == "signflip_exact" and x3.p_AUROC == 0.25 and x3.p1_AUROC == 0.125 and x1.p_method_AUROC == "gene_bootstrap_conditional" and x1.p_AUROC == 0.01
    assert SR[SR.vtype == "RESID_REF"].p_holm_AUROC_across_substrates.notna().all() and SR[SR.vtype != "RESID_REF"].p_holm_AUROC_across_substrates.isna().all()
    # 14 contrasts at n_tf = 3 use the exact flip (p .25 for equal positive diffs), keep the bootstrap p as descriptive, and flag a floor above alpha
    assert (okd.p_method == "signflip_exact").all() and (okd.p_AUROC == 0.25).all() and (okd.p_AUROC_tfboot < 0.01).all() and (okd.min_attainable_p == 0.25).all() and okd.p_floor_exceeds_alpha.all()
    # 15 panel_dependence on a synthetic npz: identical composites -> identical flag; overlapping responders -> Jaccard above the independent expectation
    import tempfile
    gg = np.array([f"g{i}" for i in range(500)])
    base = rg.random(500) < 0.3
    r1 = base.copy()
    r2 = base.copy()
    r2[:60] = ~r2[:60]
    cz = rg.normal(size=500)
    with tempfile.TemporaryDirectory() as td:
        fz = os.path.join(td, "part_S_all.panel.npz")
        np.savez(fz, **{"a::genes": gg, "a::resp": r1, "a::vec::COMP_Z": cz, "b::genes": gg, "b::resp": r2, "b::vec::COMP_Z": cz})
        pdp = panel_dependence([("S", fz)], {"S": {"a", "b"}})
    assert len(pdp) == 1 and bool(pdp.target_vectors_identical.iloc[0]) and pdp.jaccard.iloc[0] > 3 * pdp.jaccard_expected_independent.iloc[0] and pdp.n_common.iloc[0] == 500
    print("[decomp] selftest OK: zrow, fold OLS == brute force + held-out outlier, pure target-level (resid AUROC "
          f"{e0['resid']['AUROC']:.3f} vs prior {e0['prior']['AUROC']:.3f}), planted specific (resid {e1['resid']['AUROC']:.3f}), anti-predictive prior sign, partial masks, P50 tie-aware, contrasts, TF bootstrap, Holm family")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", nargs="?", choices=["run", "merge", "tfs"])
    ap.add_argument("--systems", default=",".join(s for s in A.SYSTEMS if s not in ("BONEMARROW", "BM_GATA1_ACUTE")))
    ap.add_argument("--tfs", default="")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--nslice", type=int, default=1)
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
