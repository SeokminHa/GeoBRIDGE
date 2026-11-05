#!/usr/bin/env python
"""A4 (nascent-contribution meta-analysis) and A5 (GB S comparative utility, substrate-balanced), PRE-DECLARED 2026-10-03
(internal design note (not shipped), sha256 06a5ebda...).  EVALUATION ONLY: existing frozen scores / priors / truth; nothing is re-fit, no score, depth, universe or k is changed.
Post-exposure syntheses: they cannot relabel the frozen heart NULL or create prospective confirmation.

INFRASTRUCTURE (imported, never edited)  gb_ptable.py sha 5db6f77d  (lattice_unit: bank / masks / MR / cognate rows / P; Draws: synchronized BlockAxis gene + frozen control-only G/50 module draws; split_eval; pct_ci; boot_p)
                                         gb_prospective.ablation_priors  (leave-focal-out P_pairM / P_cellU / P_cellM on the frozen T7 from the stored catbank (Z, S, C))
ENDPOINTS (response; one endpoint = (system, TF); substrate in brackets)
   PANC/RFX6 [PANC] | GATA1/GATA1, KLF1/KLF1, KLF1_B/KLF1 [ERYTHROID] | BM_CHIPATLAS/8 TFs + BM_GATA1_ACUTE/GATA1 [BM] | K562_EMP_ALLNULL/18 TFs [K562] | CARDIACV/TBX5, NKX2-5 [HEART]
   (RETINA is INCOMPLETE (no RegVelo) and absent.)  BM per-TF truth: CALIBRATED (system BM_CHIPATLAS_CALTOT = the calbm_reeval PRIMARY arm CAL_TOTAL, labels from gb_calbm_reeval.register_arms, imported unedited) LEADS in agg_cal;
   the HISTORICAL pooled-cell lists (system BM_CHIPATLAS) are aggregated alongside in agg_hist.  BM_GATA1_ACUTE has no calibrated analogue (separate experiment) and is historical in both.
A4  paired contrast d_e = metric(P_pairU) - metric(P_pairM) (frozen catbank leave-focal-out priors, T7, MR universe, identical genes), per-draw on the SAME multiplicities; sensitivity P_cellU - P_cellM.
A5  paired contrast GB S - comparator cognate row (9 comparators), identical universe / draws; mean rank of each method (rank 1 = best) per endpoint and per draw.
AGGREGATION  panels within TF (mean), TFs within substrate (mean), then EQUAL substrate weights.  Draw b is shared by every endpoint of a substrate (synchronized).
INFERENCE   hierarchical bootstrap: resample substrates (with replacement), then one synchronized draw index per picked substrate; PRIMARY inner scheme = frozen G/50 module blocks ('m50'),
            sensitivity = gene-identity ('g').  95% percentile CI, two-sided bootstrap p (+1 rule).  LOSO = every single-substrate omission, re-bootstrapped.  Exact sign-flip test on substrate values
            with its attainable floor 2/2^K published.
READING (predeclared; applied to the m50 PRIMARY aggregate)  NOT_SUPPORTED: CI includes 0 | CONSISTENT: CI > 0 AND survives every single-substrate omission | RESTRICTED: CI > 0 but not every omission.
            "survives" has TWO operationalisations, BOTH reported: strict (every LOSO CI > 0) and lenient (every LOSO point > 0).  The strict one is the headline.
ELIGIBILITY (the predeclaration says EVERY existing response endpoint / every eligible endpoint; the gate is not specified, so three nested sets are all reported, none dropped silently)
   PRIMARY  base response gate on MR (>= 20 responders, >= 20 non-responders, >= 20 independent gene units)  |  TABLEA  endpoints that passed gb_ptable's frozen gate (K562 ChIP-support rule) + KLF1_B + BM acute + heart
   ALL      every endpoint with both classes present (descriptive-only endpoints included)

  python scratch/gb_meta_nascent_sutil.py selftest
  python scratch/gb_meta_nascent_sutil.py run --out DIR [--systems A,B] [--tfs X,Y] [--B 2000] [--slice i --nslice n]
  python scratch/gb_meta_nascent_sutil.py breadth --out DIR [--B 2000] [--n_jobs 8]      # BM breadth: within-BM sensitivity endpoint (A4 only)
  python scratch/gb_meta_nascent_sutil.py control --out DIR [--B 200] [--R 20000] --rep0 0 --nrep 10      # NEGATIVE CONTROL: label permutation through the identical pipeline (then control_summary)
  python scratch/gb_meta_nascent_sutil.py agg --out DIR [--R 100000]
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "4")

import argparse
import glob
import hashlib
import itertools
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
import gb_ptable as PT  # noqa: E402  (sha 5db6f77d, imported UNEDITED; asserted in `run`)

PRED_SHA256 = "06a5ebda1a7033094ea95038a641cf87aca623fa26180a754685e0ba4aeb27bc"
PRED_NAME = "PREDECLARE_NASCENT_META_AND_SUTILITY_2026_10_03.md"
PT_SHA8 = "5db6f77d"
OUT_DEFAULT = f"{R}/results/real_uniform/meta_nascent_sutil"
PROSP = f"{R}/results/real_uniform/prospective"
CONSTR = f"{R}/results/real_uniform/construction_ablation_v2/paired_unit.tsv"
ROSTER = [("PANC", "PANC"), ("GATA1", "ERYTHROID"), ("KLF1", "ERYTHROID"), ("KLF1_B", "ERYTHROID"), ("BM_CHIPATLAS", "BM"), ("BM_GATA1_ACUTE", "BM"),
          ("K562_EMP_ALLNULL", "K562"), ("CARDIACV", "HEART"), ("BM_CHIPATLAS_CALTOT", "BM")]
SUB_OF = dict(ROSTER)
CAL_SYS, HIST_BM_SYS = "BM_CHIPATLAS_CALTOT", "BM_CHIPATLAS"       # calibrated BM per-TF truth (gb_calbm_reeval PRIMARY arm CAL_TOTAL) vs the historical pooled-cell lists, same truth-free vectors
TRUTH_OF = {CAL_SYS: "cal"}
CALBM_FULL = f"{R}/results/real_uniform/calbm_reeval/full"
SUB_ORDER = ["PANC", "ERYTHROID", "BM", "K562", "HEART"]
HEART_TFS = ["TBX5", "NKX2-5"]
PRI = ["P_pairU", "P_pairM", "P_cellU", "P_cellM"]
MET = ("AUROC", "AUPRC", "APLIFT", "P50")
SCHEMES = ("g", "m50")
PRIMARY_SCHEME = "m50"
METHODS = ["GeoBRIDGE", "|Pearson|", "GENIE3", "GRNBoost2", "RegVelo |J|", "Velorama", "RegDiffusion", "DeepSEM", "PIDC", "FastTENET"]
GB = "GeoBRIDGE"
VARIANTS = ("PRIMARY", "TABLEA", "ALL")
TOL = 1e-9


def _sha(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


SCRIPT_SHA = _sha(__file__)


def atomic_json(p, obj):
    tmp = f"{p}.tmp{os.getpid()}"
    json.dump(obj, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, p)


def atomic_tsv(df, p):
    tmp = f"{p}.tmp{os.getpid()}"
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, p)


# ===================================================================================================== pure numerics (unit-tested by `selftest`)
def signflip_exact(v):
    """exact two-sided sign-flip test of mean(v) over the 2^K sign patterns (K = number of substrate values).  Returns (p, attainable floor 2/2^K)."""
    v = np.asarray(v, float)
    K = len(v)
    if K == 0:
        return np.nan, np.nan
    sg = np.array(list(itertools.product((-1.0, 1.0), repeat=K)))
    m = np.abs((sg * v).mean(1))
    return float((m >= abs(v.mean()) - 1e-12).mean()), 2.0 / 2 ** K


def tree_mean(eps, fn, schemes=SCHEMES):
    """panels within TF (mean) -> TFs within substrate (mean).  eps: endpoint dicts; fn(ep) -> (point, {scheme: (B,) draws}).  Returns {substrate: dict(pt, dr, n_tf, n_ep, tfs, fp)}.
    Draw b is shared by every endpoint of a substrate; a substrate whose endpoints carry different draw fingerprints is refused."""
    by = {}
    for e in eps:
        by.setdefault(e["substrate"], {}).setdefault(e["tf"], []).append(e)
    out = {}
    for s, d in by.items():
        fps = {e["draws_fp"] for t in d.values() for e in t}
        if len(fps) != 1:
            raise SystemExit(f"REFUSING: substrate {s} mixes resample draws {sorted(fps)}")
        pts, drs = [], {sch: [] for sch in schemes}
        for tf, es in d.items():
            vals = [fn(e) for e in es]
            pts.append(float(np.mean([v[0] for v in vals])))
            for sch in schemes:
                drs[sch].append(np.mean([np.asarray(v[1][sch], float) for v in vals], 0))
        out[s] = dict(pt=float(np.mean(pts)), dr={sch: np.mean(drs[sch], 0) for sch in schemes}, n_tf=len(d), n_ep=sum(len(t) for t in d.values()), tfs=sorted(d), fp=fps.pop())
    return out


def hier_boot(TT, nR, seed):
    """TT (K, B): substrate x draw table.  Resample substrates with replacement, then ONE draw index per picked substrate (independent); return the (nR,) aggregate = equal-weight mean over the K picks."""
    K, B = TT.shape
    rng = np.random.default_rng(seed)
    S = rng.integers(0, K, (nR, K))
    J = rng.integers(0, B, (nR, K))
    return TT[S, J].mean(1)


def boot_seed(base, key):
    return int((base + zlib.crc32(key.encode())) % (2 ** 32))


def pct(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5))) if len(x) else (np.nan, np.nan)


def boot_p(d):
    d = np.asarray(d, float)
    d = d[np.isfinite(d)]
    return float(min(1.0, 2 * min(1 + (d <= 0).sum(), 1 + (d >= 0).sum()) / (len(d) + 1))) if len(d) else np.nan


def summarize(sub, scheme, nR, base_seed, key):
    """aggregate of the substrate dict `sub` under `scheme`: point, hierarchical CI / p, exact sign-flip p + floor, and the full LOSO table.  `key` seeds the bootstrap (shared across contrasts of one variant)."""
    names = [s for s in SUB_ORDER if s in sub]
    K = len(names)
    out = dict(K=K, names=names)
    if K == 0:
        return out
    vals = np.array([sub[s]["pt"] for s in names])
    TT = np.stack([sub[s]["dr"][scheme] for s in names])
    agg = hier_boot(TT, nR, boot_seed(base_seed, f"{key}|{scheme}|{','.join(names)}"))
    lo, hi = pct(agg)
    psf, floor = signflip_exact(vals)
    out.update(pt=float(vals.mean()), lo=lo, hi=hi, p_boot=boot_p(agg), p_signflip=psf, floor_signflip=floor, n_nonfinite_draws=int((~np.isfinite(TT)).sum()))
    loso = []
    for i, s in enumerate(names):
        keep = [j for j in range(K) if j != i]
        if not keep:
            continue
        ag = hier_boot(TT[keep], nR, boot_seed(base_seed, f"{key}|{scheme}|{','.join(names[j] for j in keep)}"))
        l, h = pct(ag)
        pf, fl = signflip_exact(vals[keep])
        loso.append(dict(omit=s, K=len(keep), pt=float(vals[keep].mean()), lo=l, hi=h, p_boot=boot_p(ag), p_signflip=pf, floor_signflip=fl))
    out["loso"] = loso
    return out


def reading(summ):
    """predeclared A4 reading (applied to the PRIMARY scheme).  Returns (strict label, lenient label, concentration note)."""
    if not summ.get("K"):
        return "NO_DATA", "NO_DATA", ""
    lo, hi, pt = summ["lo"], summ["hi"], summ["pt"]
    if not np.isfinite(lo):
        return "NO_DATA", "NO_DATA", ""
    if lo <= 0 <= hi:
        lab = "NOT_SUPPORTED"
        return lab, lab, ""
    if hi < 0:
        return "NEGATIVE_CI_BELOW_0", "NEGATIVE_CI_BELOW_0", ""
    L = summ.get("loso", [])
    strict = all(x["lo"] > 0 for x in L) if L else False
    lenient = all(x["pt"] > 0 for x in L) if L else False
    worst = min(L, key=lambda x: x["pt"]) if L else None
    note = f"weakest omission: {worst['omit']} (LOSO point {worst['pt']:+.4f}, CI [{worst['lo']:+.4f}, {worst['hi']:+.4f}])" if worst else ""
    return ("CONSISTENT" if strict else "RESTRICTED"), ("CONSISTENT" if lenient else "RESTRICTED"), note


# ===================================================================================================== one endpoint: build vectors, evaluate, parity
def _auroc(v, y, rows):
    import gb_audit_decomp as AD
    return float(AD.auroc_rows(np.asarray(v, float)[None, rows], np.asarray(y, bool)[rows])[0])


def heart_parity(tf, pts, drs, B):
    """CARDIACV vs the frozen prospective run (M1 == MR here): GB prior / ten cognate rows (AUROC points) and the stored paired-difference DRAWS of ablation::P_pairM / P_cellU / P_cellM (gene scheme; prefix-stable)."""
    vec = pd.read_csv(f"{PROSP}/part_CARDIACV_all.vec.tsv", sep="\t")
    vec = vec[vec.tf == tf].set_index("vector")
    worst, n = 0.0, 0
    chk = [("GB_prior", "P_pairU")] + [(f"cognate::{m}", f"cog::{m}") for m in METHODS] + [(f"ablation::{x}", x) for x in PRI[1:]]
    for pv, mine in chk:
        for k in ("AUROC", "APLIFT", "P50"):
            d = abs(float(vec.loc[pv, k]) - pts[(mine, k)])
            worst, n = max(worst, d), n + 1
            if d > TOL:
                raise SystemExit(f"REFUSING_PARITY heart {tf}/{pv}/{k}: now {pts[(mine, k)]!r} vs prospective {vec.loc[pv, k]!r} (|d|={d:.3e})")
    z = np.load(f"{PROSP}/part_CARDIACV_all.draws.npz")
    wd, nd = 0.0, 0
    for x in PRI[1:]:
        for k in ("AUROC", "P50"):
            key = f"{tf}::M1::ablation::{x}::{k}"
            mine = drs[("P_pairU", k, "g")] - drs[(x, k, "g")]
            ref = np.asarray(z[key], float)[:B]
            if not np.array_equal(np.isfinite(mine), np.isfinite(ref)):
                raise SystemExit(f"REFUSING_PARITY heart draws {key}: finite pattern differs")
            d = float(np.nanmax(np.abs(mine - ref)))
            wd, nd = max(wd, d), nd + 1
            if d > TOL:
                raise SystemExit(f"REFUSING_PARITY heart draws {key}: max |now - stored| = {d:.3e} over the first {B} draws")
    return dict(status="PASS", worst_point=worst, n_point_cells=n, worst_draw=wd, n_draw_cells=nd, source="prospective part_CARDIACV_all (vec.tsv + draws.npz)")


def build_endpoint(system, tf, dr, B):
    """truth-free vectors + the (historical) truth label of ONE endpoint, evaluated on the synchronized draws.  Returns (meta, pts, drs, vecs)."""
    import gb_attribution as A
    import gb_audit_decomp as AD
    import gb_catbank as CB
    import gb_lattice_extra as LE
    import gb_prospective as GPR
    import gb_layer_lattice as L
    u = PT.lattice_unit(system, tf, dr.gene)
    bank, genes, V, use = u["bank"], u["genes"], u["V"], u["use"]
    if sorted(use) != sorted(METHODS):
        raise SystemExit(f"REFUSING: {system}/{tf} methods {use} != the ten Tier-1 methods")
    r = bank.row_of(tf)
    ai, has, ai0, tc, prior, M1 = AD.masks_and_priors(bank, genes, V, use, r, u["M0"])
    if not np.array_equal(M1, u["M1"]):
        raise SystemExit(f"REFUSING: {system}/{tf} recomputed M1 differs from lattice_unit M1")
    abl = GPR.ablation_priors(u["axis"], bank, r, tc, ai0)
    with np.load(f"{CB.OUT}/{u['axis']}_priors.npz", allow_pickle=True) as z:
        Z, S, C = (np.asarray(z[f"T7|P_pairU|{c}"]) for c in "ZSC")
    tcz = np.where(tc >= 0, tc, 0)
    pU_chk = np.where(tc >= 0, CB.loo_prior(Z, S, C, r)[tcz], np.nan)
    f = np.isfinite(u["P"]) & u["MR"]
    dU = float(np.abs(pU_chk[f] - u["P"][f]).max()) if f.any() else np.nan
    if not dU <= TOL:
        raise SystemExit(f"REFUSING_PARITY {system}/{tf}: P_pairU leave-focal-out from stored (Z,S,C) differs from the lattice prior P by {dU}")
    MR, y, gi = np.asarray(u["MR"], bool), np.asarray(u["y"], bool), np.asarray(u["gi"])
    vecs = {"P_pairU": np.asarray(u["P"], float), **{k: np.asarray(abl[k], float) for k in PRI[1:]}}
    cogs = {f"cog::{m}": np.asarray(V[m], float) for m in METHODS}
    rows4 = MR & np.all([np.isfinite(vecs[k]) for k in PRI], 0)
    rows5 = MR & np.all([np.isfinite(cogs[k]) for k in cogs], 0)
    yy4, yy5 = y[rows4], y[rows5]
    base_gate = bool(LE._ORIG["gate_of"](dict(u["ep"], system="BASE"), y, rows5, rows5))
    gate_tableA = bool(u["gate"]["MR"])
    both = bool(yy4.any() and (~yy4).any() and yy5.any() and (~yy5).any())
    meta = dict(system=system, substrate=SUB_OF[system], tf=tf, truth=TRUTH_OF.get(system, "hist"), n_U=int(u["U"].sum()), n_M1=int(u["M1"].sum()), n_MR=int(MR.sum()), n_rows_A4=int(rows4.sum()), n_rows_A5=int(rows5.sum()),
                A4_rows_equal_MR=bool(np.array_equal(rows4, MR)), A5_rows_equal_MR=bool(np.array_equal(rows5, MR)), n_pos=int(yy5.sum()), n_neg=int((~yy5).sum()), prevalence=float(yy5.mean()) if len(yy5) else np.nan,
                gate_base=base_gate, gate_tableA=gate_tableA, both_classes=both, draws_fp=dr.fp, B=B, schemes=list(dr.schemes), parity_prior_GB=u["parity_prior_GB"], parity_PpairU_stored=dU,
                n_modules=dr.n_modules, ref_lattice_auroc=u["ref_lattice_auroc"], bank_sha=getattr(bank, "sha", None), methods=use)
    pts, drs = {}, {}
    if not both:
        meta["status"] = "TOO_FEW"
        return meta, pts, drs, dict(genes=genes, gi=gi, y=y, MR=MR, rows4=rows4, rows5=rows5, **vecs, **cogs)
    meta["status"] = "OK"
    for name, v in {**{k: vecs[k] for k in PRI}, **cogs}.items():
        rows = rows4 if name in PRI else rows5
        pt, bs = PT.split_eval(v, y, rows, gi, dr.C, B)
        for k in MET:
            pts[(name, k)] = pt[k]
            for sch, b in zip(dr.schemes, bs):
                drs[(name, k, sch)] = np.asarray(b[k], float)
    # ---------------------------------------------------------------- parity (fail-closed)
    if system != "CARDIACV":
        mrows = [dict(predictor=PT.P_NAME, mask="MR", **{k: pts[("P_pairU", k)] for k in MET})]
        meta["parity"] = PT.parity_lattice(u, mrows) if meta["A4_rows_equal_MR"] else dict(status="SKIPPED_ROWS_NE_MR")
        if meta["A5_rows_equal_MR"]:
            sM1 = _auroc(cogs[f"cog::{GB}"], y, u["M1"])
            sMR = pts[(f"cog::{GB}", "AUROC")]
            if abs(sM1 - sMR) > TOL and np.array_equal(MR, u["M1"]):
                raise SystemExit(f"REFUSING_PARITY {system}/{tf}: GB S AUROC on MR {sMR!r} != on M1 {sM1!r}")
        ref = None
        if os.path.exists(CONSTR):
            ct = pd.read_csv(CONSTR, sep="\t", dtype={"tf": str})
            q = ct[(ct.system == system) & (ct.tf == tf) & (ct.a == "P_pairU") & (ct.b == "P_pairM") & (ct.metric == "AUROC")]
            if len(q):
                ref = dict(diff_ref=float(q["diff"].iloc[0]), diff_now=pts[("P_pairU", "AUROC")] - pts[("P_pairM", "AUROC")])
                ref["absdiff"] = abs(ref["diff_ref"] - ref["diff_now"])
        meta["parity_construction_ablation_v2"] = ref
    else:
        meta["parity"] = heart_parity(tf, pts, drs, B)
    return meta, pts, drs, dict(genes=genes, gi=gi, y=y, MR=MR, rows4=rows4, rows5=rows5, **vecs, **cogs)


def save_endpoint(out, meta, pts, drs, vecs):
    stem = f"{out}/parts/ep_{meta['system']}__{meta['tf']}__{meta['truth']}"
    arr = {f"pt|{n}|{k}": np.float64(v) for (n, k), v in pts.items()}
    arr.update({f"dr|{n}|{k}|{s}": v for (n, k, s), v in drs.items()})
    tmp = f"{stem}.tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, meta=np.array(json.dumps(meta, default=str)), **arr)
    os.replace(tmp, f"{stem}.npz")
    tmp = f"{stem}.vec.tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **{k: np.asarray(v) for k, v in vecs.items()})
    os.replace(tmp, f"{stem}.vec.npz")


def cmd_run(a):
    import gb_audit_decomp as AD
    import gb_layer_lattice as L
    import gb_lattice_extra as LE
    pdoc = f"{a.out}/{PRED_NAME}"
    if __import__("os").environ.get("GB_SKIP_PREFLIGHT") == "1":      # opt-in (public release): skip only the brief/sha pins
        print("preflight skipped (public release)")
    else:
        if not PT.SCRIPT_SHA0.startswith(PT_SHA8):
            raise SystemExit(f"REFUSING: gb_ptable.py sha {PT.SCRIPT_SHA0[:8]} != {PT_SHA8}")
        if not os.path.exists(pdoc) or _sha(pdoc) != PRED_SHA256:
            raise SystemExit(f"REFUSING: predeclaration copy {pdoc} missing or sha != {PRED_SHA256[:8]}")
    os.makedirs(f"{a.out}/parts", exist_ok=True)
    systems = [s.strip() for s in a.systems.split(",") if s.strip()] or [s for s, _ in ROSTER]
    cal_info = None
    if CAL_SYS in systems:                      # the calibrated BM truth is the OTHER agent's builder (gb_calbm_reeval, imported unedited); register_arms() also calls LE.register()
        import gb_calbm_reeval as CAL
        CAL.register_arms()
        cal_info = dict(script=CAL.__file__, script_sha256=_sha(CAL.__file__), arm="CAL_TOTAL", system=CAL_SYS)
    else:
        LE.register()
    for system in systems:
        if system not in SUB_OF:
            raise SystemExit(f"REFUSING: {system} not in the A4/A5 roster {list(SUB_OF)}")
        t0 = time.time()
        dr, _mrc = PT.lattice_draws(system, a.B, sizes=(PT.PRIMARY_SIZE,))
        if tuple(dr.schemes) != ("g", "m50"):
            raise SystemExit(f"REFUSING: unexpected draw schemes {dr.schemes}")
        allowed = list(L.SYSTEMS[system]["tfs"]) if system == "CARDIACV" else AD.tf_list_audit(HIST_BM_SYS if system == CAL_SYS else system, L, LE)
        if system == "CARDIACV" and allowed != HEART_TFS:
            raise SystemExit(f"REFUSING: CARDIACV roster {allowed} != {HEART_TFS}")
        tfs = [t.strip() for t in a.tfs.split(",") if t.strip()] or allowed
        bad = [t for t in tfs if t not in allowed]
        if bad:
            raise SystemExit(f"REFUSING: --tfs {bad} not in the roster {allowed} of {system}")
        if a.nslice > 1:
            tfs = [t for i, t in enumerate(tfs) if i % a.nslice == a.slice]
        log = {}
        for tf in tfs:
            t1 = time.time()
            try:
                meta, pts, drs, vecs = build_endpoint(system, tf, dr, a.B)
            except (L.Blocked,) as e:
                log[tf] = dict(status=f"NOT_RUN: {e}")
                print(f"[meta] {system}/{tf}: NOT_RUN {e}", flush=True)
                atomic_json(f"{a.out}/parts/notrun_{system}__{tf}.json", dict(system=system, tf=tf, status=f"NOT_RUN: {e}"))
                continue
            save_endpoint(a.out, meta, pts, drs, vecs)
            um = pts.get(("P_pairU", "AUROC"), np.nan) - pts.get(("P_pairM", "AUROC"), np.nan)
            par = meta.get("parity", {})
            cv = (meta.get("parity_construction_ablation_v2") or {}).get("absdiff", np.nan)
            print(f"[meta] {system}/{tf}: n_MR={meta['n_MR']} pos={meta['n_pos']} gate_base={meta['gate_base']} gate_tableA={meta['gate_tableA']} dU-M AUROC={um:+.4f} "
                  f"S AUROC={pts.get((f'cog::{GB}', 'AUROC'), np.nan):.4f} parity={par.get('status')} constr_v2_absdiff={cv:.2e} {time.time() - t1:.0f}s", flush=True)
            log[tf] = dict(status=meta["status"], parity=par)
        atomic_json(f"{a.out}/parts/run_{system}_s{a.slice}of{a.nslice}.json",
                    dict(system=system, B=a.B, draws_fp=dr.fp, seeds=dict(gene=dr.gene.seed, m50=dr.mods[PT.PRIMARY_SIZE].seed), n_modules=dr.n_modules, tfs=log, seconds=round(time.time() - t0, 1),
                         script_sha256=SCRIPT_SHA, gb_ptable_sha256=PT.SCRIPT_SHA0, predeclaration_sha256=PRED_SHA256, calbm=cal_info, host=platform.node(), created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))


# ===================================================================================================== BM breadth (within-BM sensitivity only)
BRD_SYS = "BM_BREADTH"
BRD_MKEY = {"AUROC": "auc_any", "APLIFT": "aplift_any", "P50": "yield_any@50_stratified"}


def cmd_breadth(a):
    """P_pairU - P_pairM on the BM held-out perturbation-BREADTH endpoint ('any' label, ladder_universe mask = the ladder's own complete-case genes), on the ladder's own two-level resampling
    (perturbation cluster x [gene identity = the ladder's shared draws | frozen G/50 control-module block]), exactly the gb_ptable.run_breadth objects (imported).  Parity gate: the four priors' ladder-universe
    points equal results/real_uniform/ladder/final/breadth_BM.tsv.  Written as ONE pseudo-endpoint (system BM_BREADTH) used only by the within-BM sensitivity variant PRIMARY+BRD."""
    import gb_layer_lattice as L
    pdoc = f"{a.out}/{PRED_NAME}"
    if __import__("os").environ.get("GB_SKIP_PREFLIGHT") == "1":      # opt-in (public release): skip only the brief/sha pins
        print("preflight skipped (public release)")
    else:
        if not PT.SCRIPT_SHA0.startswith(PT_SHA8):
            raise SystemExit(f"REFUSING: gb_ptable.py sha {PT.SCRIPT_SHA0[:8]} != {PT_SHA8}")
        if not os.path.exists(pdoc) or _sha(pdoc) != PRED_SHA256:
            raise SystemExit(f"REFUSING: predeclaration copy {pdoc} missing or sha != {PRED_SHA256[:8]}")
    os.makedirs(f"{a.out}/parts", exist_ok=True)
    t0, B = time.time(), a.B
    U = PT.breadth_unit_bm(B, a.n_jobs)
    LD = U["LD"]
    n_all = len(U["genes"])
    units, mrc = PT.load_modules("BONEMARROW", U["t7"], PT.PRIMARY_SIZE)
    if mrc["n_names_missing_in_map"]:
        raise SystemExit("REFUSING breadth: breadth genes absent from the frozen G/50 module map")
    wp = f"{LD.LAD}/final/draws_{PT.BM_SEED_TAG}_{n_all}.npy"
    rc = json.load(open(PT.BREADTH_RCPT))
    rec = [x for x in rc["analyses"] if x["estimand"] == "TOTAL" and x["arm"] == "BM19" and x["variant"] == "full"][0]
    if rec["n_mapped_to_bank"] != n_all or _sha(wp) != rec["draws_sha256"]:
        raise SystemExit("REFUSING_PARITY breadth: mapped genes / draws sha differ from the ladder receipt")
    Wg = np.asarray(np.load(wp, mmap_mode="r")[:B])
    bi_blk = L.BLOCKS.index("BONEMARROW")
    bm50 = L.BlockAxis(U["t7"], B, int(np.random.SeedSequence(PT.mod_seed([L.BOOT_SEED, bi_blk, 20261002, 1], PT.PRIMARY_SIZE)).generate_state(1)[0]), units, "module")
    nP = len(U["perts"])
    PW = np.random.default_rng(np.random.SeedSequence([LD.SEED, zlib.crc32(PT.BM_SEED_TAG.encode()), 1])).multinomial(nP, np.full(nP, 1.0 / nP), size=6000)[:B]
    keep = np.flatnonzero(U["keep_lad"])
    pri = {m: np.asarray(LD.prior_excl(U["bankL"], m, [])[U["bi"]], float)[keep] for m in PRI}
    if not all(np.isfinite(v).all() for v in pri.values()):
        raise SystemExit("REFUSING breadth: a prior is non-finite on the ladder universe")
    D = dict(models=list(pri), fitted={}, raw=pri, keep=keep, gf=U["folds"][keep], pf=U["pf"], truth=tuple(x[:, keep] for x in (U["truth"]["T"], U["truth"]["R"], U["truth"]["E"], U["truth"]["CN"])),
             perts=list(U["perts"]), tie=None)
    pt, dg = PT._breadth_draw_all(LD, D, Wg[:, keep], PW, a.n_jobs)
    _pt2, dm = PT._breadth_draw_all(LD, D, bm50.counts[:, keep], PW, a.n_jobs)
    bt = pd.read_csv(PT.BREADTH_TSV, sep="\t", low_memory=False)
    bt = bt[(bt.estimand == "TOTAL") & (bt.arm == "BM19") & (bt.variant == "full") & (bt.kind == "model")]
    worst, nchk = 0.0, 0
    for m in PRI:
        for k, mkey in BRD_MKEY.items():
            q = bt[(bt.name == f"raw:{m}") & (bt.metric == {"yield_any@50_stratified": "yield_any@50"}.get(mkey, mkey))]
            if len(q):
                d = abs(float(q.point.iloc[0]) - float(pt[m][mkey]))
                worst, nchk = max(worst, d), nchk + 1
                if d > TOL:
                    raise SystemExit(f"REFUSING_PARITY breadth {m}/{k}: now {pt[m][mkey]!r} vs ladder {q.point.iloc[0]!r}")
    pts, drs = {}, {}
    for m in PRI:
        for k, mkey in BRD_MKEY.items():
            pts[(m, k)] = float(pt[m][mkey])
            drs[(m, k, "g")] = np.array([d[m][mkey] for d in dg], float)
            drs[(m, k, "m50")] = np.array([d[m][mkey] for d in dm], float)
        pts[(m, "AUPRC")] = np.nan
    yb = U["truth"]
    fp = f"ladder:{_sha(wp)[:12]}|{bm50.fingerprint()[:12]}"
    meta = dict(system=BRD_SYS, substrate="BM", tf="BREADTH", truth="hist", status="OK", both_classes=True, gate_base=True, gate_tableA=True, n_MR=int(len(keep)), n_pos=-1, n_neg=-1, prevalence=np.nan,
                A4_rows_equal_MR=True, A5_rows_equal_MR=True, draws_fp=fp, B=B, schemes=list(SCHEMES), parity=dict(status="PASS", worst=worst, n_cells=nchk, source=PT.BREADTH_TSV),
                note="BM held-out perturbation breadth ('any'), ladder_universe mask; two-level resampling; no cognate vectors (A5 not defined)", n_modules=int(bm50.unit.max() + 1), seconds=round(time.time() - t0, 1))
    arr = {f"pt|{n}|{k}": np.float64(v) for (n, k), v in pts.items()}
    arr.update({f"dr|{n}|{k}|{s_}": v for (n, k, s_), v in drs.items()})
    stem = f"{a.out}/parts/ep_{BRD_SYS}__BREADTH__hist"
    np.savez_compressed(f"{stem}.tmp{os.getpid()}.npz", meta=np.array(json.dumps(meta, default=str)), **arr)
    os.replace(f"{stem}.tmp{os.getpid()}.npz", f"{stem}.npz")
    print(f"[meta] BM_BREADTH: rows={len(keep)} U-M AUROC={pts[('P_pairU', 'AUROC')] - pts[('P_pairM', 'AUROC')]:+.4f} AP-lift={pts[('P_pairU', 'APLIFT')] - pts[('P_pairM', 'APLIFT')]:+.3f} "
          f"parity worst={worst:.2e} over {nchk} cells; {time.time() - t0:.0f}s", flush=True)


# ===================================================================================================== negative control: label permutation through the IDENTICAL pipeline
def cmd_control(a):
    """NEGATIVE CONTROL (results gate 5): every endpoint's labels are permuted among its MR genes (prevalence and gate status preserved), the identical vectors / synchronized draws / aggregation / reading are
    re-run.  Under this no-association null the A4 macro and every S - comparator macro must centre on 0 and the 95% CI must exclude 0 in <= ~5% of replicates, and no replicate may read CONSISTENT.
    B is small (a calibration check, not a result).  Inputs are the saved parts/*.vec.npz."""
    import gb_lattice_extra as LE
    LE.register()
    eps = load_parts(a.out)
    base_seed = int(PRED_SHA256[:8], 16)
    dr_cache = {}
    vecs = {}
    for e in eps:
        if e["status"] != "OK" or e["system"] == BRD_SYS:
            continue
        z = np.load(f"{a.out}/parts/ep_{e['system']}__{e['tf']}__{e['truth']}.vec.npz", allow_pickle=True)
        vecs[e["key"]] = {k: z[k] for k in z.files}
        dsys = HIST_BM_SYS if e["system"] == CAL_SYS else e["system"]
        if dsys not in dr_cache:
            dr_cache[dsys] = PT.lattice_draws(dsys, a.B, sizes=(PT.PRIMARY_SIZE,))[0]
        if dr_cache[dsys].fp != e["draws_fp"]:
            raise SystemExit(f"REFUSING: control draws fingerprint {dr_cache[dsys].fp} != the endpoint's {e['draws_fp']} ({e['key']})")
    names = PRI + [f"cog::{m}" for m in METHODS]
    rows = []
    for rep in range(a.rep0, a.rep0 + a.nrep):
        EPr = []
        for e in eps:
            if e["key"] not in vecs:
                continue
            v = vecs[e["key"]]
            dsys = HIST_BM_SYS if e["system"] == CAL_SYS else e["system"]
            dr = dr_cache[dsys]
            rng = np.random.default_rng(boot_seed(base_seed, f"control|{rep}|{e['key']}"))
            y = np.asarray(v["y"], bool).copy()
            idx = np.flatnonzero(v["rows5"])
            y[idx] = y[idx][rng.permutation(len(idx))]
            pts, drs = {}, {}
            for nm in names:
                vec = v[nm] if nm in PRI else v[nm]
                pt, bs = PT.split_eval(np.asarray(vec, float), y, v["rows5"], v["gi"], dr.C, a.B)
                for k in ("AUROC",):
                    pts[(nm, k)] = pt[k]
                    for sch, b in zip(dr.schemes, bs):
                        drs[(nm, k, sch)] = np.asarray(b[k], float)
            EPr.append(dict(e, pt=pts, dr=drs))
        for mode in ("cal", "hist"):
            drop = HIST_BM_SYS if mode == "cal" else CAL_SYS
            sel = [x for x in EPr if x["system"] != drop]
            for label, fn in [("A4 P_pairU - P_pairM", contr_diff("P_pairU", "P_pairM", "AUROC"))] + [(f"A5 S - {m}", contr_diff(f"cog::{GB}", f"cog::{m}", "AUROC")) for m in METHODS[1:]]:
                sub = tree_mean([x for x in sel if eligible(x, "PRIMARY")], fn)
                sm = summarize(sub, PRIMARY_SCHEME, a.R, base_seed, f"control|{rep}|{mode}")
                st, ln, _ = reading(sm)
                rows.append(dict(rep=rep, mode=mode, contrast=label, K=sm["K"], pt=sm["pt"], lo=sm["lo"], hi=sm["hi"], reading_strict=st, reading_lenient=ln))
        print(f"[control] replicate {rep} done", flush=True)
    os.makedirs(f"{a.out}/control", exist_ok=True)
    atomic_tsv(pd.DataFrame(rows), f"{a.out}/control/control_rep{a.rep0}_{a.rep0 + a.nrep - 1}.tsv")


def control_summary(out):
    fs = sorted(glob.glob(f"{out}/control/control_rep*.tsv"))
    if not fs:
        return None
    d = pd.concat([pd.read_csv(f, sep="\t") for f in fs], ignore_index=True)
    g = d.groupby(["mode", "contrast"])
    t = g.agg(n_rep=("rep", "nunique"), mean_pt=("pt", "mean"), sd_pt=("pt", "std"), frac_ci_gt0=("lo", lambda x: float((x > 0).mean())),
              frac_ci_lt0=("hi", lambda x: float((x < 0).mean())), frac_consistent_strict=("reading_strict", lambda x: float((x == "CONSISTENT").mean())),
              frac_consistent_lenient=("reading_lenient", lambda x: float((x == "CONSISTENT").mean()))).reset_index()
    atomic_tsv(t, f"{out}/control/control_summary.tsv")
    return t


# ===================================================================================================== aggregation
def load_parts(out):
    EP = []
    for f in sorted(glob.glob(f"{out}/parts/ep_*.npz")):
        if f.endswith(".vec.npz"):
            continue
        z = np.load(f, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        e = dict(meta, key=os.path.basename(f)[:-4], pt={}, dr={})
        if meta["status"] == "OK":
            for k in z.files:
                if k.startswith("pt|"):
                    n, m = k[3:].rsplit("|", 1)                       # method names contain '|' ("|Pearson|", "RegVelo |J|"): split from the right
                    e["pt"][(n, m)] = float(z[k])
                elif k.startswith("dr|"):
                    n, m, s = k[3:].rsplit("|", 2)
                    e["dr"][(n, m, s)] = np.asarray(z[k], float)
        EP.append(e)
    return EP


def eligible(e, variant):
    if e["status"] != "OK" or not e["both_classes"]:
        return False
    if e["system"] == BRD_SYS:                                       # breadth is a within-BM sensitivity ONLY: never a standard-variant endpoint, never a replicate
        return False
    if variant == "PRIMARY":
        return bool(e["gate_base"])
    if variant == "TABLEA":
        return bool(e["gate_tableA"]) or e["system"] in ("KLF1_B", "BM_GATA1_ACUTE", "CARDIACV")
    return True


def contr_diff(a, b, metric):
    def fn(e):
        return e["pt"][(a, metric)] - e["pt"][(b, metric)], {s: e["dr"][(a, metric, s)] - e["dr"][(b, metric, s)] for s in SCHEMES}
    return fn


def contr_rank(method, metric):
    from scipy.stats import rankdata

    def fn(e):
        names = [f"cog::{m}" for m in METHODS]
        i = METHODS.index(method)
        pt = rankdata(-np.array([e["pt"][(n, metric)] for n in names]))[i]
        out = {s: rankdata(-np.stack([e["dr"][(n, metric, s)] for n in names]), axis=0)[i] for s in SCHEMES}
        return float(pt), out
    return fn


def contr_value(name, metric):
    def fn(e):
        return e["pt"][(name, metric)], {s: e["dr"][(name, metric, s)] for s in SCHEMES}
    return fn


def run_contrast(EP, variant, label, fn, nR, base_seed, rows, loso_rows, sub_rows, bm_extra=None, vlabel=None, **tags):
    """bm_extra: the BM breadth pseudo-endpoint; when given, the BM substrate value is the equal-weight mean of (its per-TF aggregate, the breadth value) -- the within-BM sensitivity, labelled `vlabel`.
    The breadth draws are the ladder's own two-level draws (a different resample structure): index-aligned, treated as independent of the per-TF draws (flagged; sensitivity only)."""
    inc = [e for e in EP if eligible(e, variant)]
    sub = tree_mean(inc, fn)
    if bm_extra is not None and "BM" in sub:
        bp, bd = fn(bm_extra)
        x = sub["BM"]
        sub["BM"] = dict(x, pt=0.5 * (x["pt"] + bp), dr={sch: 0.5 * (x["dr"][sch] + np.asarray(bd[sch], float)) for sch in SCHEMES}, tfs=x["tfs"] + ["BREADTH(avg)"], n_ep=x["n_ep"] + 1)
    variant = vlabel or variant
    for s in SUB_ORDER:
        if s in sub:
            x = sub[s]
            lo, hi = pct(x["dr"][PRIMARY_SCHEME])
            glo, ghi = pct(x["dr"]["g"])
            sub_rows.append(dict(tags, variant=variant, contrast=label, substrate=s, n_tf=x["n_tf"], n_ep=x["n_ep"], tfs=",".join(x["tfs"]), pt=x["pt"], lo_m50=lo, hi_m50=hi, lo_g=glo, hi_g=ghi))
    for sch in SCHEMES:
        sm = summarize(sub, sch, nR, base_seed, f"{variant}")
        if not sm.get("K"):
            continue
        st, ln, note = reading(sm) if sch == PRIMARY_SCHEME else ("", "", "")
        rows.append(dict(tags, variant=variant, contrast=label, scheme=sch, K=sm["K"], substrates=",".join(sm["names"]), pt=sm["pt"], lo=sm["lo"], hi=sm["hi"], p_boot=sm["p_boot"],
                         p_signflip=sm["p_signflip"], floor_signflip=sm["floor_signflip"], loso_min_pt=min([x["pt"] for x in sm["loso"]], default=np.nan), loso_max_pt=max([x["pt"] for x in sm["loso"]], default=np.nan),
                         loso_min_lo=min([x["lo"] for x in sm["loso"]], default=np.nan), n_loso_ci_pos=sum(x["lo"] > 0 for x in sm["loso"]), n_loso_pt_pos=sum(x["pt"] > 0 for x in sm["loso"]),
                         reading_strict=st, reading_lenient=ln, note=note, n_nonfinite_draws=sm["n_nonfinite_draws"]))
        for x in sm["loso"]:
            loso_rows.append(dict(tags, variant=variant, contrast=label, scheme=sch, **x))


def cal_parity(out, EPall):
    """CALIBRATED BM endpoints: (i) truth-free vectors, gene axis, universe MR bit-identical to the historical BM_CHIPATLAS endpoint of the same TF (only the truth differs); (ii) n_pos and the GB S / GB P AUROC
    equal the other agent's calbm_reeval/full part tables (arm CAL_TOTAL, mask MR).  Fail-closed.  Returns the per-TF report."""
    rep = {}
    for e in EPall:
        if e["system"] != CAL_SYS or e["status"] != "OK":
            continue
        tf = e["tf"]
        r = {}
        hv, cv = f"{out}/parts/ep_{HIST_BM_SYS}__{tf}__hist.vec.npz", f"{out}/parts/ep_{CAL_SYS}__{tf}__cal.vec.npz"
        if os.path.exists(hv):
            zh, zc = np.load(hv, allow_pickle=True), np.load(cv, allow_pickle=True)
            bad = [k for k in zh.files if k not in ("y", "rows4", "rows5") and not np.array_equal(zh[k], zc[k], equal_nan=zh[k].dtype.kind == "f")]
            if bad or not (np.array_equal(zh["rows4"], zc["rows4"]) and np.array_equal(zh["rows5"], zc["rows5"])):
                raise SystemExit(f"REFUSING_PARITY cal {tf}: truth-free vectors / universe differ from the historical endpoint: {bad}")
            r["vectors_identical_to_hist"] = True
        else:
            r["vectors_identical_to_hist"] = "hist endpoint absent"
        mt = f"{CALBM_FULL}/part_{tf}.metrics.tsv"
        if os.path.exists(mt):
            t = pd.read_csv(mt, sep="\t")
            t = t[(t.arm == "CAL_TOTAL") & (t["mask"] == "MR") & (t.method == GB)]
            for kind, mine in (("S", e["pt"][(f"cog::{GB}", "AUROC")]), ("P", e["pt"][("P_pairU", "AUROC")])):
                q = t[t.kind == kind]
                if len(q):
                    d = abs(float(q.AUROC.iloc[0]) - mine)
                    r[f"AUROC_{kind}_absdiff_vs_calbm_reeval"] = d
                    if d > TOL:
                        raise SystemExit(f"REFUSING_PARITY cal {tf}: GB {kind} AUROC {mine!r} != calbm_reeval {float(q.AUROC.iloc[0])!r}")
                    if int(q.n_pos.iloc[0]) != e["n_pos"]:
                        raise SystemExit(f"REFUSING_PARITY cal {tf}: n_pos {e['n_pos']} != calbm_reeval {int(q.n_pos.iloc[0])}")
            r["checked_vs_calbm_reeval"] = bool(len(t))
        else:
            r["checked_vs_calbm_reeval"] = False
        rep[tf] = r
    return rep


def analyze(EP, outm, a, mode, notrun_systems):
    os.makedirs(outm, exist_ok=True)
    base_seed = int(PRED_SHA256[:8], 16)
    # ------------------------------------------------------------ endpoint inventory (losses and unavailable rows shown)
    inv = []
    for e in EP:
        r = {k: e.get(k) for k in ("system", "substrate", "tf", "truth", "status", "n_MR", "n_pos", "n_neg", "prevalence", "gate_base", "gate_tableA", "both_classes", "A4_rows_equal_MR", "A5_rows_equal_MR",
                                    "draws_fp")}
        if e["status"] == "OK":
            for k in MET:
                r[f"UM_{k}"] = e["pt"][("P_pairU", k)] - e["pt"][("P_pairM", k)]
                r[f"cUcM_{k}"] = e["pt"][("P_cellU", k)] - e["pt"][("P_cellM", k)]
            if e["system"] != BRD_SYS:
                for m in METHODS:
                    r[f"AUROC::{m}"] = e["pt"][(f"cog::{m}", "AUROC")]
            r["parity"] = (e.get("parity") or {}).get("status")
        inv.append(r)
    inv = pd.DataFrame(inv)
    for j in glob.glob(f"{a.out}/parts/notrun_*.json"):
        d = json.load(open(j))
        if d["system"] in notrun_systems:
            inv = pd.concat([inv, pd.DataFrame([dict(system=d["system"], tf=d["tf"], status=d["status"], substrate=SUB_OF.get(d["system"]))])], ignore_index=True)
    inv["substrate"] = inv["substrate"].astype(str)
    atomic_tsv(inv, f"{outm}/endpoints.tsv")
    rows, loso_rows, sub_rows = [], [], []
    for variant in VARIANTS:
        for metric in ("AUROC", "APLIFT", "P50"):
            run_contrast(EP, variant, "A4 P_pairU - P_pairM", contr_diff("P_pairU", "P_pairM", metric), a.R, base_seed, rows, loso_rows, sub_rows, analysis="A4", metric=metric)
            run_contrast(EP, variant, "A4s P_cellU - P_cellM", contr_diff("P_cellU", "P_cellM", metric), a.R, base_seed, rows, loso_rows, sub_rows, analysis="A4s", metric=metric)
            for m in METHODS[1:]:
                run_contrast(EP, variant, f"A5 S - {m}", contr_diff(f"cog::{GB}", f"cog::{m}", metric), a.R, base_seed, rows, loso_rows, sub_rows, analysis="A5", metric=metric)
            for m in METHODS:
                run_contrast(EP, variant, f"A5 meanrank {m}", contr_rank(m, metric), a.R, base_seed, rows, loso_rows, sub_rows, analysis="A5rank", metric=metric)
            run_contrast(EP, variant, f"A5 value S", contr_value(f"cog::{GB}", metric), a.R, base_seed, rows, loso_rows, sub_rows, analysis="A5val", metric=metric)
    brd = next((e for e in EP if e["system"] == BRD_SYS and e["status"] == "OK"), None)
    if brd is not None:
        for metric in ("AUROC", "APLIFT", "P50"):
            run_contrast(EP, "PRIMARY", "A4 P_pairU - P_pairM", contr_diff("P_pairU", "P_pairM", metric), a.R, base_seed, rows, loso_rows, sub_rows, bm_extra=brd, vlabel="PRIMARY+BRD", analysis="A4", metric=metric)
            run_contrast(EP, "PRIMARY", "A4s P_cellU - P_cellM", contr_diff("P_cellU", "P_cellM", metric), a.R, base_seed, rows, loso_rows, sub_rows, bm_extra=brd, vlabel="PRIMARY+BRD", analysis="A4s", metric=metric)
    agg = pd.DataFrame(rows)
    atomic_tsv(agg, f"{outm}/aggregate.tsv")
    atomic_tsv(pd.DataFrame(loso_rows), f"{outm}/loso.tsv")
    atomic_tsv(pd.DataFrame(sub_rows), f"{outm}/substrate_values.tsv")
    write_report(outm, agg, pd.DataFrame(loso_rows), pd.DataFrame(sub_rows), inv, mode)


def cmd_agg(a):
    EPall = load_parts(a.out)
    if not EPall:
        raise SystemExit(f"REFUSING: no endpoint parts in {a.out}/parts")
    have_cal = any(e["system"] == CAL_SYS and e["status"] == "OK" for e in EPall)
    calrep = cal_parity(a.out, EPall) if have_cal else {}
    modes = (["cal"] if have_cal else []) + ["hist"]                       # calibrated BM truth LEADS when present; the historical-truth aggregate is always produced alongside
    for mode in modes:
        drop = HIST_BM_SYS if mode == "cal" else CAL_SYS
        EP = [e for e in EPall if e["system"] != drop]
        analyze(EP, f"{a.out}/agg_{mode}", a, mode, {s for s, _ in ROSTER if s != drop})
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"{a.out}/parts/run_*.json"))]
    atomic_json(f"{a.out}/receipt.json", dict(
        analysis="A4 nascent-contribution meta-analysis + A5 S comparative utility (substrate-balanced)", predeclaration=PRED_NAME, predeclaration_sha256=PRED_SHA256, script=__file__, script_sha256=SCRIPT_SHA,
        gb_ptable_sha256=PT.SCRIPT_SHA0, python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, host=platform.node(), created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        n_endpoints=int(len(EPall)), n_endpoints_OK=int(sum(e["status"] == "OK" for e in EPall)), outer_bootstrap_R=a.R, bootstrap_seed_base=int(PRED_SHA256[:8], 16), schemes=SCHEMES, primary_scheme=PRIMARY_SCHEME,
        substrates=SUB_ORDER, aggregates=modes, bm_truth=("CALIBRATED per-TF (calbm_reeval CAL_TOTAL) LEADS in agg_cal; HISTORICAL in agg_hist" if have_cal else "HISTORICAL only (calibrated BM labels not available)"),
        calibrated_parity=calrep, variants=dict(PRIMARY="base response gate on MR (>=20 resp, >=20 non-resp, >=20 units)", TABLEA="gb_ptable gate + KLF1_B + BM acute + heart", ALL="every endpoint with both classes"),
        reading_rule=reading.__doc__, run_files=runs, run_script_shas=sorted({r.get("script_sha256", "") for r in runs}),
        file_shas={n: _sha(f"{R}/scratch/{n}.py") for n in tuple(PT.DEP_MODULES) + ("gb_ptable", "gb_prospective", "gb_calbm_reeval", "gb_canon_table", "gb_catbank") if os.path.exists(f"{R}/scratch/{n}.py")},
        interpretive_choices=["eligibility gate not specified by the predeclaration: PRIMARY = base response gate (>=20/>=20/>=20 units) on MR, TABLEA and ALL reported alongside",
                              "LOSO 'survives' operationalised two ways (strict: every LOSO CI>0; lenient: every LOSO point>0); the strict one is the headline",
                              "BM calibrated truth = calbm_reeval PRIMARY arm CAL_TOTAL (TOTAL estimand, negatives = all tested non-responders); BM_GATA1_ACUTE stays historical",
                              "BM breadth = within-BM sensitivity variant PRIMARY+BRD (BM value = equal-weight mean of per-TF aggregate and breadth; breadth draws are the ladder's two-level draws, index-aligned, independence assumed)",
                              "CARDIACV module map G/50 = the frozen control-only ptable map (built before this analysis); truth read after the prospective run was complete (section 400)"], parts={e["key"]: dict(status=e["status"], parity=e.get("parity"), draws_fp=e.get("draws_fp")) for e in EPall}))


# ===================================================================================================== compact report
def f3(x):
    return "   nan" if x is None or not np.isfinite(x) else f"{x:+.3f}"


def write_report(out, agg, loso, subv, inv, mode="hist"):
    L = []
    P = L.append
    P(f"A4/A5 compact report | predeclaration sha {PRED_SHA256[:8]} | script {SCRIPT_SHA[:8]} | BM per-TF truth = {'CALIBRATED (calbm_reeval CAL_TOTAL)' if mode == 'cal' else 'HISTORICAL'}")
    P("")
    P("ENDPOINT INVENTORY (per substrate): n endpoints / pass base gate / pass TableA gate / OK")
    for s in SUB_ORDER:
        g = inv[(inv.substrate == s) & (inv.system != BRD_SYS)]                       # the breadth pseudo-endpoint is a sensitivity, listed separately below
        if len(g):
            P(f"  {s:10s} n={len(g):2d}  OK={int((g.status == 'OK').sum()):2d}  base_gate={int(g.gate_base.fillna(False).astype(bool).sum()):2d}  tableA_gate={int(g.gate_tableA.fillna(False).astype(bool).sum()):2d}  "
              f"NOT_RUN/TOO_FEW={int((g.status != 'OK').sum())}")
    P("")
    brd_rows = inv[inv.system == BRD_SYS] if "system" in inv else inv.iloc[0:0]
    if len(brd_rows):
        r = brd_rows.iloc[0]
        P(f"BM BREADTH (within-BM sensitivity endpoint, ladder universe {int(r.n_MR)} genes): U-M AUROC {f3(r.UM_AUROC)}  AP-lift {f3(r.UM_APLIFT)}  P50 {f3(r.UM_P50)}  | cell U-M AUROC {f3(r.cUcM_AUROC)}")
        P("")
    for variant in VARIANTS + (("PRIMARY+BRD",) if (agg.variant == "PRIMARY+BRD").any() else ()):
        P("=" * 150)
        P(f"VARIANT {variant}" + ("  (BM substrate = equal-weight mean of its per-TF aggregate and the BM breadth endpoint; A4 only)" if variant == "PRIMARY+BRD" else ""))
        for an, title in (("A4", "A4  P_pairU - P_pairM"), ("A4s", "A4s P_cellU - P_cellM (construction-robustness sensitivity)")):
            for metric in ("AUROC", "APLIFT", "P50"):
                lab = {"A4": "A4 P_pairU - P_pairM", "A4s": "A4s P_cellU - P_cellM"}[an]
                sv = subv[(subv.variant == variant) & (subv.contrast == lab) & (subv.metric == metric)]
                ag = agg[(agg.variant == variant) & (agg.contrast == lab) & (agg.metric == metric)]
                if not len(ag):
                    continue
                P(f"-- {title} | {metric}")
                P("   substrate  n_tf n_ep   point   [m50 CI]            [gene CI]")
                for r in sv.itertuples():
                    P(f"   {r.substrate:9s} {r.n_tf:4d} {r.n_ep:4d}  {f3(r.pt)}  [{f3(r.lo_m50)},{f3(r.hi_m50)}]  [{f3(r.lo_g)},{f3(r.hi_g)}]")
                for sch in SCHEMES:
                    r = ag[ag.scheme == sch].iloc[0]
                    P(f"   MACRO[{sch:3s}] K={r.K}  {f3(r.pt)} [{f3(r.lo)},{f3(r.hi)}]  p_boot={r.p_boot:.4f}  p_signflip={r.p_signflip:.4f} (floor {r.floor_signflip:.4f})  LOSO pt [{f3(r.loso_min_pt)},{f3(r.loso_max_pt)}] "
                      f"min LOSO lo {f3(r.loso_min_lo)}  LOSO CI>0 {r.n_loso_ci_pos}/{r.K}  pt>0 {r.n_loso_pt_pos}/{r.K}")
                r = ag[ag.scheme == PRIMARY_SCHEME].iloc[0]
                P(f"   READING (m50): strict={r.reading_strict} lenient={r.reading_lenient}  {r.note}")
                lo_ = loso[(loso.variant == variant) & (loso.contrast == lab) & (loso.metric == metric) & (loso.scheme == PRIMARY_SCHEME)]
                P("   LOSO (m50): " + "; ".join(f"-{x.omit}: {f3(x.pt)} [{f3(x.lo)},{f3(x.hi)}]" for x in lo_.itertuples()))
        P("")
        if variant == "PRIMARY+BRD":
            continue
        for metric in ("AUROC", "APLIFT", "P50"):
            P(f"-- A5 | {metric} | substrate-balanced MEAN RANK (1 = best of 10) and paired GB S - comparator")
            sv = subv[(subv.variant == variant) & (subv.metric == metric)]
            for m in METHODS:
                ag = agg[(agg.variant == variant) & (agg.contrast == f"A5 meanrank {m}") & (agg.metric == metric) & (agg.scheme == PRIMARY_SCHEME)]
                if not len(ag):
                    continue
                r = ag.iloc[0]
                per = " ".join(f"{s[:4]}:{x.pt:4.1f}" for s in SUB_ORDER for x in sv[(sv.contrast == f"A5 meanrank {m}") & (sv.substrate == s)].itertuples())
                P(f"   rank {m:13s} {r.pt:5.2f} [{r.lo:5.2f},{r.hi:5.2f}]   per-substrate {per}")
            for m in METHODS[1:]:
                lab = f"A5 S - {m}"
                ag = agg[(agg.variant == variant) & (agg.contrast == lab) & (agg.metric == metric)]
                if not len(ag):
                    continue
                r = ag[ag.scheme == PRIMARY_SCHEME].iloc[0]
                rg = ag[ag.scheme == "g"].iloc[0]
                per = " ".join(f"{s[:4]}:{f3(x.pt)}" for s in SUB_ORDER for x in sv[(sv.contrast == lab) & (sv.substrate == s)].itertuples())
                P(f"   S - {m:13s} {f3(r.pt)} [{f3(r.lo)},{f3(r.hi)}] g[{f3(rg.lo)},{f3(rg.hi)}] p_sf={r.p_signflip:.3f} LOSO[{f3(r.loso_min_pt)},{f3(r.loso_max_pt)}] ciLOSO>0 {r.n_loso_ci_pos}/{r.K} "
                  f"{r.reading_strict}/{r.reading_lenient} | {per}")
            P("")
    txt = "\n".join(L)
    open(f"{out}/compact_report.txt", "w").write(txt + "\n")
    print(txt)


# ===================================================================================================== selftest (synthetic numerics; data parity is enforced inside `run`)
def selftest(a=None):
    from scipy.stats import rankdata
    # (1) sign-flip exact: K=3 all positive -> p = 2/8, floor 2/8; K=5 all positive -> floor .0625
    p, fl = signflip_exact([1.0, 2.0, 3.0])
    assert abs(p - 0.25) < 1e-12 and abs(fl - 0.25) < 1e-12, (p, fl)
    p, fl = signflip_exact([1, 1, 1, 1, 1])
    assert abs(fl - 2 / 32) < 1e-12 and abs(p - 2 / 32) < 1e-12
    p, fl = signflip_exact([1, -1, 1, -1])
    assert p == 1.0
    # (2) tree_mean: panels within TF, TFs within substrate (hand-computed), draws synchronized
    B = 7
    rng = np.random.default_rng(0)
    mk = lambda sub, tf, v: dict(substrate=sub, tf=tf, draws_fp="fp", v=v, vd=np.full(B, v))
    eps = [mk("S1", "A", 1.0), mk("S1", "A", 3.0), mk("S1", "B", 10.0), mk("S2", "C", -2.0)]
    t = tree_mean(eps, lambda e: (e["v"], {s: e["vd"] for s in SCHEMES}))
    assert abs(t["S1"]["pt"] - (2.0 + 10.0) / 2) < 1e-12 and t["S1"]["n_tf"] == 2 and t["S1"]["n_ep"] == 3 and abs(t["S2"]["pt"] + 2.0) < 1e-12
    assert np.allclose(t["S1"]["dr"]["m50"], 6.0)
    try:
        tree_mean([mk("S1", "A", 1.0), dict(mk("S1", "B", 2.0), draws_fp="other")], lambda e: (e["v"], {s: e["vd"] for s in SCHEMES}))
        raise AssertionError("mixed draw fingerprints were not refused")
    except SystemExit:
        pass
    # (3) hierarchical bootstrap: constant table -> degenerate CI at the constant; mean unbiased on a toy table; LOSO count
    TT = np.tile(np.array([[0.1], [0.3], [0.5]]), (1, 50))
    ag = hier_boot(TT, 20000, 1)
    assert abs(ag.mean() - 0.3) < 0.005
    sub = {s: dict(pt=v, dr={sch: np.full(50, v) for sch in SCHEMES}) for s, v in zip(["PANC", "BM", "K562"], [0.1, 0.3, -0.5])}
    sm = summarize(sub, "m50", 4000, 7, "t")
    assert sm["K"] == 3 and len(sm["loso"]) == 3 and abs(sm["pt"] - (-0.1 / 3)) < 1e-12 and sm["floor_signflip"] == 0.25
    assert [x["omit"] for x in sm["loso"]] == ["PANC", "BM", "K562"] and abs(sm["loso"][2]["pt"] - 0.2) < 1e-12
    # (4) reading rules
    mk_s = lambda pt, lo, hi, loso: dict(K=5, pt=pt, lo=lo, hi=hi, loso=loso)
    L_ok = [dict(omit=f"s{i}", pt=0.02, lo=0.001, hi=0.04) for i in range(5)]
    assert reading(mk_s(0.03, 0.01, 0.05, L_ok))[:2] == ("CONSISTENT", "CONSISTENT")
    L_bad = L_ok[:4] + [dict(omit="K562", pt=0.02, lo=-0.01, hi=0.05)]
    assert reading(mk_s(0.03, 0.01, 0.05, L_bad))[:2] == ("RESTRICTED", "CONSISTENT")
    L_neg = L_ok[:4] + [dict(omit="PANC", pt=-0.01, lo=-0.03, hi=0.01)]
    assert reading(mk_s(0.03, 0.01, 0.05, L_neg))[:2] == ("RESTRICTED", "RESTRICTED")
    assert reading(mk_s(0.02, -0.01, 0.05, L_ok))[0] == "NOT_SUPPORTED" and reading(mk_s(-0.04, -0.06, -0.01, L_ok))[0] == "NEGATIVE_CI_BELOW_0"
    # (5) rank contrast: ties get average ranks; best AUROC = rank 1; per-draw rank follows the draw
    e = dict(pt={(f"cog::{m}", "AUROC"): v for m, v in zip(METHODS, [0.6, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.7, 0.55])}, dr={})
    for m, v in zip(METHODS, [0.6, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0, 0.7, 0.55]):
        for s in SCHEMES:
            e["dr"][(f"cog::{m}", "AUROC", s)] = np.full(4, v)
    pt, dd = contr_rank(GB, "AUROC")(e)
    assert pt == 2.5 and np.all(dd["m50"] == 2.5), (pt, dd["m50"])
    assert contr_rank("PIDC", "AUROC")(e)[0] == 1.0 and contr_rank("DeepSEM", "AUROC")(e)[0] == 10.0
    # (6) paired contrast: difference of draws == draws of the difference (eval_rows is linear in the same multiplicities); AUROC of a perfect score is 1 on every draw
    import gb_canon_table as CT
    n, G = 300, 300
    y = rng.random(n) < 0.3
    v1 = y + 0.3 * rng.standard_normal(n)
    v2 = rng.standard_normal(n)
    counts = CT.make_counts(G, 40, 5)
    gi = np.arange(n)
    rows = np.ones(n, bool)
    p1, b1 = CT.eval_rows(v1, y, rows, gi, counts)
    p2, b2 = CT.eval_rows(v2, y, rows, gi, counts)
    pperf, bperf = CT.eval_rows(y.astype(float), y, rows, gi, counts)
    assert abs(pperf["AUROC"] - 1) < 1e-12 and np.allclose(bperf["AUROC"][np.isfinite(bperf["AUROC"])], 1.0)
    ref_auc = float((rankdata(v1)[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))
    assert abs(p1["AUROC"] - ref_auc) < 1e-12 and p1["AUROC"] > p2["AUROC"]
    print("[selftest] OK: sign-flip floors, tree mean + draw-fingerprint refusal, hierarchical bootstrap + LOSO, reading rules, rank ties, eval_rows AUROC parity")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("selftest")
    r = sp.add_parser("run")
    r.add_argument("--out", default=OUT_DEFAULT)
    r.add_argument("--systems", default="")
    r.add_argument("--tfs", default="")
    r.add_argument("--B", type=int, default=2000)
    r.add_argument("--slice", type=int, default=0)
    r.add_argument("--nslice", type=int, default=1)
    b = sp.add_parser("breadth")
    b.add_argument("--out", default=OUT_DEFAULT)
    b.add_argument("--B", type=int, default=2000)
    b.add_argument("--n_jobs", type=int, default=8)
    c = sp.add_parser("control")
    c.add_argument("--out", default=OUT_DEFAULT)
    c.add_argument("--B", type=int, default=200)
    c.add_argument("--R", type=int, default=20000)
    c.add_argument("--rep0", type=int, default=0)
    c.add_argument("--nrep", type=int, default=10)
    cs = sp.add_parser("control_summary")
    cs.add_argument("--out", default=OUT_DEFAULT)
    g = sp.add_parser("agg")
    g.add_argument("--out", default=OUT_DEFAULT)
    g.add_argument("--R", type=int, default=100000)
    a = ap.parse_args()
    if a.cmd == "selftest":
        selftest()
    elif a.cmd == "run":
        cmd_run(a)
    elif a.cmd == "breadth":
        cmd_breadth(a)
    elif a.cmd == "control":
        cmd_control(a)
    elif a.cmd == "control_summary":
        print(control_summary(a.out).to_string())
    else:
        cmd_agg(a)


if __name__ == "__main__":
    main()
