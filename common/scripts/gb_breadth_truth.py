#!/usr/bin/env python
# Response-breadth truth (evaluation only) for BM (GSE274113) and K562 (Replogle essential)
import argparse, glob, gzip, hashlib, io, json, math, os, re, shutil, sys, time, warnings
from types import SimpleNamespace
import numpy as np, pandas as pd
from scipy import sparse, stats

R = __import__("os").environ["GEOBRIDGE_ROOT"]
OUT = f"{R}/results/breadth_truth"
SCRATCH = f"{R}/scratch"
K562_SRC = f"{R}/data/scperturb/ReplogleWeissman2022_K562_essential.h5ad"
LAMBERT = f"{R}/GB_real_v2/lambert2018_human_tfs.txt"
BM_DIR = f"{R}/data/GSE274113"
BM_META_DIR = f"{R}/data/gse274113_meta"
BM_BRIEF = f"{BM_META_DIR}/gse274113_brief.txt"
BM_ALLOWED_REPS = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 13, 14, 16}
K562_NT = "non-targeting"
# K562 endpoint TFs from gb_cop_alignment.py.
K562_F2_36 = ("ATF4 ATF5 BPTF CDC5L CEBPZ CXXC1 DNMT1 DR1 DRAP1 E4F1 GABPA GATA1 HINFP HSF1 JAZF1 KIN MAX MYBL2 "
              "MYC NRF1 SETDB1 SNAPC2 SON SP2 TBP TERF2 TFDP1 THAP1 THAP11 YY1 ZBTB17 ZNF236 ZNF317 ZNF335 "
              "ZNF574 ZNF687").split()

# frozen thresholds
PS_COUNTS = 0.5         # pseudocount = 0.5 counts of the smaller pooled library (converted to cpm)
MIN_DET = 0.05          # NT detection floor (gene level)
MIN_CELLS = 50          # perturbed and NT cells in the comparison
MIN_HALF_CELLS = 25     # K562: perturbed and NT cells per half
MIN_UNITS = 3           # BM valid units
WS_MIN_MASS = 0.5       # BM WITHIN_STATE: a unit needs >= this share of the NT reference composition present
ALPHA, LFC_MIN, EQ_STRICT = 0.05, 0.5, 0.25
BROAD_B, BROAD_NR = 0.25, 2
CONF_FRAC = 0.8         # Minimum fraction of testable perturbations classified as confident non-response.
COVER_FRAC, COVER_FLOOR = 0.5, 5
NEIGHBOR_BP = 10_000
NULL_SEED = 20260928
NULL_EMP_M = 200          # K562 empirical-null draws per perturbation, first stage (M0; --m_null)
NULL_EMP_MAX = 10_000     # adaptive refinement cap per gene (M1; --m_max)
REFINE_TRIGGER = 1        # genes with <= this many exceedances after M0 get refined
REFINE_STOP = 20          # ... and stop once they reach this many exceedances (sequential stopping)
REFINE_BATCH = 1000
N_BINS = 10               # expression deciles for the pooled (secondary) null
NULL_M = 5               # independent pseudo draws per real perturbation
NULL_PRIMARY_Q = 0.95    # primary: #comparisons with >=1 responder <= this quantile of Binomial(K, .05)
NULL_TAIL_P, NULL_TAIL_MAX_RATIO = 1e-3, 2.0
STATUS = {0: "UNTESTED", 1: "RESPONDER", 2: "CONFIDENT_NONRESPONDER", 3: "AMBIGUOUS"}
LABEL_RULES = dict(
    any_response=">=1 RESPONDER", confident_nonresponse="0 RESPONDER AND >=80% of testable perts CONF (ambiguous <=20%) AND n_testable>=cover_min",
    broad="b_g>=.25 AND >=2 RESPONDER", single="exactly 1 RESPONDER AND >=(cover_min-1) of the other testable perts CONF",
    cover_min="max(5, ceil(.5 * perturbations in arm))")
NOTES = [
    "NT-detection floor: genes with <5% NT detection are UNTESTED for every perturbation; they can be neither "
    "responders nor negatives, so b_g/burden are defined on the >=5%-expressed universe and are conditioned on "
    "baseline expression (induction from an off state is invisible; a gene near the floor has low power).",
    "Halves are technical replicates only (BM: 10x channels of one donor; K562: gem groups = lanes of one culture; "
    "one dual-guide construct per gene, no guide replication). Split consistency is not biological replication.",
    "K562 scales: p comes from the empirical null of the log1p(CP10K) gem-group-matched mean difference; the CI/equivalence is on "
    "the COUNT-scale log2FC with se_cal = max(delta-method se, SD of the same quantity over the first-stage null draws), normal "
    "quantile. Welch/Satterthwaite is kept only as *_LEGACY.",
    "K562 refinement cap M1=10,000 (p floor 1/10,001; a perturbation needs >=~17 genes at the floor for any BH call); the pooled-binned null (TOTAL_ALT_EMP_POOLED_BINNED) is the deeper-tail sensitivity.",
    "Null gate: each pseudo-perturbation is a uniform random NT draw within each unit (no cell-type matching), "
    "size min(n_r,u, floor(NT_u/2)), tested against its own complement NT in that unit; M=5 independent draws per real "
    "perturbation. Real perturbations always use ALL NT cells, so real results do not depend on --null-gate.",
]
BLOCKED_BY_SPEC = [
    "min_unit_cells (10; 5 per cell-type stratum), MIN_UNITS=3, >=2 valid units per BM day, WS_MIN_MASS=.5 and the "
    "K562 requirement of >=25 NT cells per half are unspecified in the design; declared here.",
    "BM efficacy: genotyping-file format unknown -> every BM perturbation EFFICACY_UNVERIFIED (candidates listed only).",
    "K562: needs RAW COUNTS (layers/counts or integer X); otherwise refuses (NO_RAW_COUNTS). Run `inspect` first.",
    "Null-gate rule (N2): PRIMARY = #pseudo comparisons with >=1 responder <= 95% quantile of Binomial(K,.05), K=5*n_perts, "
    "per estimand per system; SECONDARY = pooled tail ratio (p<1e-3, observed/expected over T pairs) <= 2; failing either "
    "labels the estimand UNCALIBRATED.",
    "Promoter-neighbour sensitivity needs var chr/start/end (strand optional; TSS approximated by gene start if absent); "
    "q-values are NOT recomputed after the exclusion.",
]


# small utilities
# NaN-safe BH (same logic as gb_gse274113_truth.py:44-52): NaN p is treated as 1, never dropped
def bh(p):
    p = np.asarray(p, float); n = p.size
    if n == 0: return p.copy()
    p = np.where(np.isfinite(p), p, 1.0)
    o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""): h.update(b)
    return h.hexdigest()


# --smalln: first three in sorted order (deterministic, never response-driven)
def first_three(names):
    return sorted(names)[:3]


def write_json(path, obj):
    with open(path, "w") as fh: json.dump(obj, fh, indent=1, default=str)


def input_record(paths, skip_hash):
    rec = {}
    for p in paths:
        if not os.path.exists(p): rec[p] = "MISSING"; continue
        rec[p] = dict(bytes=os.stat(p).st_size, sha256=("SKIPPED" if skip_hash else sha256_file(p)))
    return rec


def thresholds():
    return dict(PS_COUNTS=PS_COUNTS, MIN_DET=MIN_DET, MIN_CELLS=MIN_CELLS, MIN_HALF_CELLS=MIN_HALF_CELLS, MIN_UNITS=MIN_UNITS,
                WS_MIN_MASS=WS_MIN_MASS, ALPHA=ALPHA, LFC_MIN=LFC_MIN, EQ_STRICT=EQ_STRICT, BROAD_B=BROAD_B,
                BROAD_NR=BROAD_NR, CONF_FRAC=CONF_FRAC, COVER_FRAC=COVER_FRAC, COVER_FLOOR=COVER_FLOOR,
                NEIGHBOR_BP=NEIGHBOR_BP, NULL_M=NULL_M, NULL_EMP_M=NULL_EMP_M, NULL_EMP_MAX=NULL_EMP_MAX, REFINE_TRIGGER=REFINE_TRIGGER, REFINE_STOP=REFINE_STOP, N_BINS=N_BINS, NULL_PRIMARY_Q=NULL_PRIMARY_Q, NULL_TAIL_P=NULL_TAIL_P, NULL_TAIL_MAX_RATIO=NULL_TAIL_MAX_RATIO, NULL_SEED=NULL_SEED, STATUS=STATUS)


# rp = receipt path (final receipt, or a shard's part receipt)
def refuse_if_complete(rp, force):
    if os.path.exists(rp) and not force:
        try: ph = json.load(open(rp)).get("phase")
        except Exception: ph = "UNREADABLE"
        if ph == "COMPLETE": raise SystemExit(f"REFUSING: {rp} is COMPLETE; pass --force to overwrite")


# pseudo-perturbation draws (NT-vs-NT null gate)
# One pseudo-perturbation
def draw_pseudo(rng, pools, sizes):
    return {k: rng.choice(np.asarray(pools[k]), size=min(int(sizes.get(k, 0)), len(pools[k]) // 2), replace=False)
            for k in sorted(pools)}


# Stable per-perturbation seed (sha1 of the name, not its sorted position)
def pert_seed(name):
    return int(hashlib.sha1(str(name).encode()).hexdigest()[:8], 16)


def null_rng(*ids):
    return np.random.default_rng([NULL_SEED, *ids])


# BM statistics (pure numpy; exercised by --selftest)
def _ps_cpm(Lp, Ln):
    return PS_COUNTS / np.maximum(np.minimum(Lp, Ln), 1e-12) * 1e6


# Per-unit log2 delta
def deltas_total(Sp, Tp, Np, Sn, Tn, Nn, minu):
    valid = (Np >= minu) & (Nn >= minu) & (Tp > 0) & (Tn > 0)
    tp, tn = np.where(valid, Tp, 1.0), np.where(valid, Tn, 1.0)
    ps = _ps_cpm(tp, tn)[:, None]
    d = np.log2((Sp / tp[:, None] * 1e6 + ps) / (Sn / tn[:, None] * 1e6 + ps))
    return d, valid, np.asarray(Np, float), np.asarray(Nn, float)


def ref_composition(nN, block):
    D = int(block.max()) + 1; pi = np.zeros((D, nN.shape[1]))
    for d in range(D):
        x = nN[block == d].sum(0).astype(float); pi[d] = x / max(x.sum(), 1e-12)
    return pi


# Direct standardisation inside each unit
def deltas_std(Sp, Tp, Np, Sn, Tn, Nn, pi, block, minu):
    U, C, G = Sp.shape; d = np.zeros((U, G)); valid = np.zeros(U, bool); n1 = np.zeros(U); n0 = np.zeros(U)
    for u in range(U):
        pres = (Np[u] >= minu) & (Nn[u] >= minu) & (Tp[u] > 0) & (Tn[u] > 0)
        mass = pi[block[u]][pres].sum()
        if not pres.any() or mass < WS_MIN_MASS: continue
        w = pi[block[u]] * pres / mass; tp, tn = np.where(pres, Tp[u], 1.0), np.where(pres, Tn[u], 1.0)
        cp = (w[:, None] * Sp[u] / tp[:, None]).sum(0) * 1e6; cn = (w[:, None] * Sn[u] / tn[:, None]).sum(0) * 1e6
        ps = _ps_cpm(Tp[u][pres].sum(), Tn[u][pres].sum())
        d[u] = np.log2((cp + ps) / (cn + ps)); valid[u] = True; n1[u] = Np[u][pres].sum(); n0[u] = Nn[u][pres].sum()
    return d, valid, n1, n0


# Paired contrast of per-unit deltas
def paired(d, valid, n1u, n0u, block, half):
    n1, n0 = float(n1u[valid].sum()), float(n0u[valid].sum())
    if valid.sum() < MIN_UNITS or n1 < MIN_CELLS or n0 < MIN_CELLS: return None
    if not all((valid & (half == h)).any() for h in (0, 1)): return None
    days = np.unique(block[valid]); nb = np.array([(valid & (block == b)).sum() for b in days])
    if nb.min() < 2: return None
    def wmean(mask): return np.mean([d[mask & (block == b)].mean(0) for b in np.unique(block[mask])], 0)
    m = wmean(valid); Dn = len(days)
    A = np.stack([d[valid & (block == b)].var(0, ddof=1) / n / Dn ** 2 for b, n in zip(days, nb)])
    var = A.sum(0); se = np.sqrt(var)                      # SE of the equal-day-weighted log2FC (CI only)
    nv = int(valid.sum()); mu = d[valid].mean(0); se_u = d[valid].std(0, ddof=1) / np.sqrt(nv)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = mu / se_u                                      # N3: p-value from the unweighted paired t, df = n_valid - 1
        df = var ** 2 / (A ** 2 / (nb - 1)[:, None]).sum(0)   # Satterthwaite df for the CI of the weighted log2FC
    est = np.isfinite(t) & (se_u > 0) & (se > 0) & np.isfinite(df)
    p = np.where(est, 2 * stats.t.sf(np.abs(np.where(est, t, 0.0)), nv - 1), np.nan)
    z = np.where(est, np.sign(t) * stats.norm.isf(np.clip(np.nan_to_num(p, nan=1.0) / 2, 1e-300, 0.5)), 0.0)
    return dict(lfc=m, lfcA=wmean(valid & (half == 0)), lfcB=wmean(valid & (half == 1)), dm=mu, se=se, z=z, p=p, est=est,
                df=np.where(est, df, 1.0), n1=int(n1), n0=int(n0), nu=nv)


# K562 statistics (cell-level test from sufficient stats)
# Count-based log2FC: pooled perturbed cpm vs NT cpm reweighted to the perturbation's library mix
def _lfc_cell(P, N, mask):
    Lp = P["L"][mask]; L0 = N["L"][mask]; v = Lp / Lp.sum()
    cp = P["C"][mask].sum(0) / Lp.sum() * 1e6
    cn = (v[:, None] * N["C"][mask] / L0[:, None]).sum(0) * 1e6
    ps = _ps_cpm(Lp.sum(), L0.sum())
    return np.log2((cp + ps) / (cn + ps))


def _resid_ss(S1, S2, n):
    return np.maximum(S2 - S1 ** 2 / n[:, None], 0.0)


# P,N dicts for one perturbation / the NT reference: n,L (U,), C,Y,Y2,X,X2 (U,G) per-gem-group sums
def contrast_cell(P, N, half):
    m = (P["n"] > 0) & (N["n"] >= 2) & (P["L"] > 0) & (N["L"] > 0)
    if not m.any(): return None
    npu, n0u = P["n"][m].astype(float), N["n"][m].astype(float); n1, n0 = npu.sum(), n0u.sum()
    if n1 < MIN_CELLS or n0 < MIN_CELLS: return None
    for h in (0, 1):
        mh = m & (half == h)
        if P["n"][mh].sum() < MIN_HALF_CELLS or N["n"][mh].sum() < MIN_HALF_CELLS: return None
    dfp = n1 - m.sum()
    if dfp < 1: return None
    w = npu / n1
    def stat(key, key2):
        Pk1, Pk2, Nk1, Nk2 = (np.asarray(x[m], np.float64) for x in (P[key], P[key2], N[key], N[key2]))   # X/X2 are float32 on disk
        yp = Pk1 / npu[:, None]; y0 = Nk1 / n0u[:, None]
        s2p = _resid_ss(Pk1, Pk2, npu).sum(0) / dfp
        s20 = _resid_ss(Nk1, Nk2, n0u) / (n0u - 1)[:, None]
        b = (w ** 2)[:, None] * s20 / n0u[:, None]
        return (w[:, None] * yp).sum(0), (w[:, None] * y0).sum(0), s2p / n1, b
    yp, y0, VarP, b = stat("Y", "Y2")
    dm = yp - y0; var = VarP + b.sum(0); se_t = np.sqrt(var)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = dm / se_t
        df = var ** 2 / (VarP ** 2 / dfp + (b ** 2 / (n0u - 1)[:, None]).sum(0))
    xp, xn, VarXp, bx = stat("X", "X2")
    ps = _ps_cpm(P["L"][m].sum(), N["L"][m].sum())
    with np.errstate(divide="ignore", invalid="ignore"):
        se = np.sqrt(VarXp / (xp + ps) ** 2 + bx.sum(0) / (xn + ps) ** 2) / np.log(2)
    est = np.isfinite(t) & (se_t > 0) & np.isfinite(df) & np.isfinite(se) & (se > 0)
    p = np.where(est, 2 * stats.t.sf(np.abs(np.where(est, t, 0.0)), np.where(est, np.maximum(df, 1.0), 1.0)), np.nan)
    z = np.where(est, np.sign(t) * stats.norm.isf(np.clip(np.nan_to_num(p, nan=1.0) / 2, 1e-300, 0.5)), 0.0)
    return dict(lfc=_lfc_cell(P, N, m), lfcA=_lfc_cell(P, N, m & (half == 0)), lfcB=_lfc_cell(P, N, m & (half == 1)),
                dm=dm, se=se, z=z, p=p, est=est, df=np.where(est, np.maximum(df, 1.0), 1.0),
                n1=int(n1), n0=int(n0), nu=int(m.sum()), m=m, n_u=np.where(m, P["n"], 0).astype(int))


# K562 empirical per-gene null (replaces the anti-conservative Welch p-value; Welch kept as *_LEGACY)
# M independent uniform n-subsets of range(Nu), shape (M, n)
def _uniform_subsets(rng, M, Nu, n):
    if n * 16 > Nu: return np.argpartition(rng.random((M, Nu)), n - 1, axis=1)[:, :n]
    idx = rng.integers(0, Nu, (M, n))
    while True:
        srt = np.sort(idx, axis=1); bad = (srt[:, 1:] == srt[:, :-1]).any(1) if n > 1 else np.zeros(M, bool)
        if not bad.any(): return idx
        idx[bad] = rng.integers(0, Nu, (int(bad.sum()), n))


# M pseudo-perturbation draws with the perturbation's gem-group mix: n*_u = min(nsz[u], floor(|pool_u|/2)) uniform NT cells per gem group
def draw_subsets(rng, pools, nsz, M):
    us, ns, idx, capped, lost, tot = [], [], [], 0, 0, 0
    for u in range(len(pools)):
        Nu = len(pools[u]); want = int(nsz[u]); n = min(want, Nu // 2); tot += want
        if want > n: capped += 1; lost += want - n
        if n >= 1: us.append(u); ns.append(n); idx.append(_uniform_subsets(rng, M, Nu, n))
    if not us: return None
    return dict(us=us, ns=np.array(ns, float), idx=idx, cap_units=capped, cap_frac=lost / max(tot, 1))


# Statistics of drawn pseudo-perturbations against the rest of the reference NT in the same gem group
def subset_stats(sub, pools, ctx, refs, cols, want_lfc):
    us, ns, idx = sub["us"], sub["ns"], sub["idx"]; M, K = idx[0].shape[0], len(us)
    rows = np.concatenate([np.repeat(np.arange(M) * K + j, idx[j].shape[1]) for j in range(K)])
    cell = np.concatenate([np.asarray(pools[u])[idx[j]].ravel() for j, u in enumerate(us)])
    Ind = sparse.csr_matrix((np.ones(rows.size, np.float32), (rows, cell)), shape=(M * K, ctx["YNT"].shape[0]))
    sl_ = slice(None) if cols is None else cols
    SY = np.asarray(Ind @ ctx["YNT"][:, sl_], np.float64).reshape(M, K, -1)
    nref = np.array([len(pools[u]) for u in us], float)
    diff = SY / ns[None, :, None] - (np.asarray(refs["Y"])[us][:, sl_][None] - SY) / (nref - ns)[None, :, None]
    delta = ((ns / ns.sum())[None, :, None] * diff).sum(1)
    if not want_lfc: return delta, None
    SC = np.asarray(Ind @ ctx["NTCg"][:, sl_], np.float64).reshape(M, K, -1)
    SL = np.asarray(Ind @ ctx["Lc"].astype(np.float64)[:, None], np.float64).reshape(M, K)
    Lref = np.asarray(refs["L"], float)[us][None]; Cref = np.asarray(refs["C"], float)[us][:, sl_][None]
    SLt = SL.sum(1); cp = SC.sum(1) / SLt[:, None] * 1e6
    cn = ((SL / SLt[:, None])[:, :, None] * (Cref - SC) / (Lref - SL)[:, :, None]).sum(1) * 1e6
    ps = _ps_cpm(SLt, (Lref - SL).sum(1))[:, None]
    return delta, np.log2((cp + ps) / (cn + ps))


# secondary null: null studentised by each gene's own null SD, pooled over genes and draws within expression-decile bins
def pooled_binned_p(obs, d0, sd0, bins):
    p = np.full(len(obs), np.nan)
    for b in np.unique(bins):
        g = np.flatnonzero((bins == b) & (sd0 > 1e-12))
        if g.size == 0: continue
        zn = np.sort(np.abs(d0[:, g] / sd0[g]).ravel()); zo = np.abs(obs[g] / sd0[g])
        p[g] = (1.0 + zn.size - np.searchsorted(zn, zo - 1e-12, side="left")) / (1.0 + zn.size)
    return p


# In-memory NT reference used by every null: NTCg (n_nt, G') counts, Lc (n_nt,) total library, nt_u gem group
def nt_context(NTCg, Lc, nt_u, U):
    YNT = np.log1p(NTCg * (1e4 / np.maximum(Lc, 1))[:, None].astype(np.float32)).astype(np.float32)
    mu = YNT.mean(0); bins = np.digitize(mu, np.quantile(mu, np.linspace(0, 1, N_BINS + 1)[1:-1]))
    return dict(YNT=YNT, NTCg=NTCg, Lc=np.asarray(Lc, float), bins=bins, pools=[np.flatnonzero(nt_u == u) for u in range(U)])


# Per-gene empirical null with adaptive refinement (primary) + expression-binned pooled null (secondary)
def empirical_test(res, ctx, pools, refs, rng, M0=NULL_EMP_M, M1=NULL_EMP_MAX, B=REFINE_BATCH, stop=REFINE_STOP):
    G = len(res["dm"]); obs = res["dm"]; aobs = np.abs(obs) - 1e-12
    sub = draw_subsets(rng, pools, res["n_u"], M0)
    d0, l0 = subset_stats(sub, pools, ctx, refs, None, True)
    sd0 = d0.std(0, ddof=1); zero = ~(sd0 > 1e-12); est = res["est"] & ~zero
    ex = (np.abs(d0) >= aobs[None]).sum(0); Mg = np.full(G, M0); done = M0
    active = np.flatnonzero(est & (ex <= REFINE_TRIGGER))
    while active.size and done < M1:
        b = min(B, M1 - done); sb = draw_subsets(rng, pools, res["n_u"], b)
        db, _ = subset_stats(sb, pools, ctx, refs, active, False)
        ex[active] += (np.abs(db) >= aobs[active][None]).sum(0); Mg[active] += b; done += b
        active = active[ex[active] < stop]
    se_cal = np.maximum(res["se"], l0.std(0, ddof=1))
    def mk(p):
        e = est & np.isfinite(p)
        return dict(res, p=np.where(e, p, np.nan), est=e, se=se_cal, df=np.full(G, 1e9), se_welch=res["se"],
                    z=np.where(e, np.sign(obs) * stats.norm.isf(np.clip(np.nan_to_num(p, nan=1.0) / 2, 1e-300, 0.5)), 0.0),
                    p_legacy=res["p"], z_legacy=res["z"], n_draws=Mg)
    info = dict(Mg=Mg, n_refined=int((Mg > M0).sum()), n_zero_sd=int((zero & res["est"]).sum()), cap_units=sub["cap_units"],
                cap_frac=sub["cap_frac"], exceed=ex)
    return mk((1.0 + ex) / (1.0 + Mg)), mk(pooled_binned_p(obs, d0, sd0, ctx["bins"])), dict(res, est=est), info


# Level of the gate's secondary tail rule
def tail_level(mode, M0, M1):
    if mode == "gene" and M1 < 2000: return max(2, int(NULL_TAIL_P * (M0 + 1))) / (M0 + 1)
    return NULL_TAIL_P


def alloc_stats(ng, G):
    return dict(n=np.zeros(ng), L=np.zeros(ng), C=np.zeros((ng, G), np.float32), Y=np.zeros((ng, G)), Y2=np.zeros((ng, G)),
                X=np.zeros((ng, G), np.float32), X2=np.zeros((ng, G), np.float32))


# Sufficient stats (U, G') of a set of cells (counts rows, gem-group index u) restricted to genes gsel
def stats_sub(cnt, u, U, gsel):
    A = alloc_stats(U, cnt.shape[1]); add_cell_stats(A, cnt, np.asarray(u, np.int64))
    out = {"n": A["n"], "L": A["L"]}
    for k in ("C", "Y", "Y2", "X", "X2"): out[k] = A[k][:, gsel].astype(np.float32 if k in ("X", "X2") else np.float64)
    return out


# classification / aggregation
# res = contrast dict (or None)
# Arrays over the G' testable-gene axis
def classify(res, det, self_mask):
    G = len(det); z0 = np.zeros(G); nan = np.full(G, np.nan)
    if res is None:
        return dict(status=np.zeros(G, np.int8), T=np.zeros(G, bool), e=z0, e_legacy=z0, q=nan, cons=np.zeros(G, bool),
                    strict=np.zeros(G, bool), se=nan, lo=nan, hi=nan, res=None)
    T = res["est"] & det & ~self_mask
    q = np.full(G, np.nan); q[T] = bh(res["p"][T])
    with np.errstate(invalid="ignore"):
        s = np.sign(res["lfc"])
        cons = T & (s != 0) & (np.sign(res["lfcA"]) == s) & (np.sign(res["lfcB"]) == s) & (np.sign(res["dm"]) == s)
        Rr = T & (q < ALPHA) & (np.abs(res["lfc"]) >= LFC_MIN) & cons
        tc = stats.t.ppf(0.975, np.maximum(res["df"], 1))
        lo, hi = res["lfc"] - tc * res["se"], res["lfc"] + tc * res["se"]     # CI of the reported log2FC
        cn = T & ~Rr & (lo >= -LFC_MIN) & (hi <= LFC_MIN)
        strict = T & ~Rr & (lo >= -EQ_STRICT) & (hi <= EQ_STRICT)
    status = np.zeros(G, np.int8); status[T] = 3; status[cn] = 2; status[Rr] = 1
    e = np.where(T, np.nan_to_num(res["lfcA"]) * np.nan_to_num(res["lfcB"]), 0.0)
    e_leg = np.where(cons, np.abs(np.nan_to_num(res["lfc"])), 0.0)
    return dict(status=status, T=T, e=e, e_legacy=e_leg, q=q, cons=cons, strict=strict, se=res["se"], lo=lo, hi=hi, res=res)


# All args (P, G') for the perturbations of one arm
def per_gene_table(status, E, EL, SE, STRICT, cover_min):
    Tm = status > 0
    nT, nR = Tm.sum(0), (status == 1).sum(0)
    nC, nA, nU = (status == 2).sum(0), (status == 3).sum(0), (status == 0).sum(0)
    den = np.maximum(nT, 1)
    with np.errstate(invalid="ignore", divide="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        b = np.where(nT > 0, nR / den, np.nan)
        me = np.where(nT > 0, (E * Tm).sum(0) / den, np.nan)
        burden = np.where(nT > 0, np.sqrt(np.maximum(me, 0.0)), np.nan)
        leg = np.where(nT > 0, (EL * Tm).sum(0) / den, np.nan)
        m2 = (E ** 2 * Tm).sum(0) / den; var = np.maximum(m2 - np.nan_to_num(me) ** 2, 0) * nT / np.maximum(nT - 1, 1)
        me_se = np.where(nT > 1, np.sqrt(var / den), np.nan)
        med_se = np.nanmedian(np.where(Tm, SE, np.nan), axis=0)
        any_lab = np.where(nR >= 1, 1.0, np.where((nR == 0) & (nC >= CONF_FRAC * nT) & (nT >= cover_min), 0.0, np.nan))
        broad = np.where((b >= BROAD_B) & (nR >= BROAD_NR), 1.0,
                         np.where((nR == 1) & (nC >= cover_min - 1) & (nT >= cover_min), 0.0, np.nan))
    return pd.DataFrame(dict(n_testable=nT, n_responder=nR, n_conf_nonresp=nC, n_conf_strict=(STRICT & Tm).sum(0),
                             n_ambiguous=nA, n_untested=nU, b_g=b, burden=burden, mean_e_crossfit=me,
                             mean_e_crossfit_se=me_se, burden_legacy_mean_abs=leg, median_se=med_se,
                             label_any_vs_confnon=any_lab, label_broad_vs_single=broad))


# output assembly
PAIR_COLS = ["estimand", "pert", "gene", "T", "status", "log2fc", "log2fc_A", "log2fc_B", "split_consistent", "strict_conf",
             "e_crossfit", "se", "z", "p", "q", "ci_lo", "ci_hi", "n_pert_cells", "n_nt_cells", "n_units", "promoter_neighbor"]
VARIANT_COLS = ("p", "q", "status")     # written as p_LEGACY / p_ALT ... for every extra estimand variant


# Write per_pair.tsv.gz with mtime=0 and one write per perturbation block.
class PairWriter:
    def __init__(self, path):
        self.raw = open(path, "wb"); self.gz = gzip.GzipFile(filename="", mode="wb", fileobj=self.raw, mtime=0); self.first = True

    def add_block(self, text, header_line):
        self.gz.write(((header_line if self.first else "") + text).encode("utf-8")); self.first = False

    def add(self, estimand, pert, genes_sel, o, variants=None, extra=None):
        res = o["res"]
        if res is None or not o["T"].any(): return
        df = pd.DataFrame({
            "estimand": estimand, "pert": pert, "gene": genes_sel, "T": o["T"].astype(np.int8),
            "status": [STATUS[i] for i in o["status"]], "log2fc": res["lfc"], "log2fc_A": res["lfcA"], "log2fc_B": res["lfcB"],
            "split_consistent": o["cons"].astype(np.int8), "strict_conf": o["strict"].astype(np.int8), "e_crossfit": o["e"],
            "se": res["se"], "z": res["z"], "p": res["p"], "q": o["q"], "ci_lo": o["lo"], "ci_hi": o["hi"],
            "n_pert_cells": res["n1"], "n_nt_cells": res["n0"], "n_units": res["nu"],
            "promoter_neighbor": o.get("nbr", np.zeros(len(genes_sel), bool)).astype(np.int8)})
        cols = list(PAIR_COLS)
        for name, v in (variants or {}).items():           # e.g. LEGACY (Welch), ALT (binned pooled null)
            df[f"p_{name}"], df[f"q_{name}"] = v["res"]["p"], v["q"]
            df[f"status_{name}"] = [STATUS[i] for i in v["status"]]; cols += [f"p_{name}", f"q_{name}", f"status_{name}"]
        for name, v in (extra or {}).items(): df[name] = v; cols.append(name)
        self.add_block(df[cols].to_csv(None, sep="\t", index=False, header=False, float_format="%.5g"), "\t".join(cols) + "\n")

    def close(self): self.gz.close(); self.raw.close()


def _stack(rd, pl, key, G):
    return np.vstack([rd[p][key] if rd[p][key] is not None else np.full(G, np.nan) for p in pl])


# results[estimand][pert] = classify dict
def build_outputs(out_dir, genes, gene_ids, det_all, gsel, perts, results, arms, self_lfc, extra_cols):
    Gall, Gp = len(genes), len(gsel); tabs = []
    for est, rd in results.items():
        variants = [(est, False)] + ([(est + "_EXCL_PROMOTER_NEIGHBOR", True)] if any("nbr" in o for o in rd.values()) else [])
        for arm, plist in arms.items():
            pl = [p for p in plist if p in rd]
            if not pl: continue
            cover = max(COVER_FLOOR, math.ceil(COVER_FRAC * len(pl)))
            for vname, excl in variants:
                st = np.vstack([rd[p]["status"] for p in pl]).copy()
                if excl: st[np.vstack([rd[p].get("nbr", np.zeros(Gp, bool)) for p in pl])] = 0
                t = per_gene_table(st, _stack(rd, pl, "e", Gp), _stack(rd, pl, "e_legacy", Gp), _stack(rd, pl, "se", Gp),
                                   np.vstack([rd[p]["strict"] for p in pl]), cover)
                t = t.set_index(gsel).reindex(np.arange(Gall))
                for c in ("n_testable", "n_responder", "n_conf_nonresp", "n_conf_strict", "n_ambiguous"): t[c] = t[c].fillna(0).astype(int)
                t["n_untested"] = len(pl) - t["n_testable"]
                t["gene"] = genes; t["gene_id"] = gene_ids; t["ctrl_detect"] = det_all; t["estimand"] = vname; t["arm"] = arm
                t["n_perts_in_arm"] = len(pl); t["cover_min"] = cover
                tabs.append(t.reset_index(drop=True))
    pg = pd.concat(tabs, ignore_index=True)
    lead = ["estimand", "arm", "gene", "gene_id", "ctrl_detect", "n_perts_in_arm", "cover_min"]
    pg = pg[lead + [c for c in pg.columns if c not in lead]]
    pg.to_csv(f"{out_dir}/per_gene.tsv", sep="\t", index=False, float_format="%.6g")
    rows = []
    for est, rd in results.items():
        for p in perts:
            if p not in rd: continue
            o = rd[p]; r = o["res"]
            rows.append(dict(estimand=est, pert=p, arms=",".join(a for a, pl in arms.items() if p in pl),
                             n_valid_units=(r["nu"] if r else 0), n_testable_genes=int(o["T"].sum()),
                             n_responder=int((o["status"] == 1).sum()), n_conf_nonresp=int((o["status"] == 2).sum()),
                             n_ambiguous=int((o["status"] == 3).sum()), self_log2fc=self_lfc.get((est, p), np.nan),
                             **extra_cols(est, p)))
    pd.DataFrame(rows).to_csv(f"{out_dir}/perturbations.tsv", sep="\t", index=False, float_format="%.5g")
    return pg


NULL_AMENDMENT = ("The first null gate was defective by construction. Its pseudo-perturbations copied "
                  "the real perturbation's cell-type depletion (via the pool//(2K) cap), so the 9 TOTAL pseudo-responders were "
                  "basophil/myeloid composition markers dropping in day-14 units. Replaced by M=5 uniform within-unit NT draws per "
                  "perturbation (no cell-type matching), primary = binomial-quantile rule, secondary = tail-ratio rule.")


# N2. n_resp/n_T/n_tail: per pseudo comparison (K = len)
def gate_verdict(n_resp, n_T, n_tail, tail_p=None):
    K = len(n_resp); c = int(sum(1 for r in n_resp if r >= 1)); thr = int(stats.binom.ppf(NULL_PRIMARY_Q, K, 0.05))
    tail_p = NULL_TAIL_P if tail_p is None else tail_p
    exp_tail = tail_p * float(sum(n_T)); ratio = (float(sum(n_tail)) / exp_tail) if exp_tail > 0 else float("nan")
    prim = bool(c <= thr); sec = bool(np.isfinite(ratio) and ratio <= NULL_TAIL_MAX_RATIO)
    return dict(K=K, comparisons_with_responder=c, primary_threshold=thr, PRIMARY_PASS=prim, tail_level=tail_p, tail_observed=int(sum(n_tail)),
                tail_expected=exp_tail, tail_ratio=ratio, SECONDARY_PASS=sec, label="CALIBRATED" if prim and sec else "UNCALIBRATED")


# null_results[estimand][pseudo] = classify dict
def gate_rows(null_results, matched, size_ratio, tail_levels=None, bins=None):
    rows, per_bin = [], {}
    for est, rd in null_results.items():
        lvl = (tail_levels or {}).get(est, NULL_TAIL_P)
        for ps, o in rd.items():
            r = o["res"]; T = o["T"]
            tail = int(((r["p"] <= lvl + 1e-12) & T).sum()) if r is not None else 0
            excl = int((T & ((o["lo"] > 0) | (o["hi"] < 0))).sum()) if r is not None else 0
            if r is not None and bins is not None:
                pb = per_bin.setdefault(est, np.zeros((2, N_BINS)))
                for b in range(N_BINS):
                    m = T & (bins == b); pb[0, b] += m.sum(); pb[1, b] += (m & (r["p"] <= lvl + 1e-12)).sum()
            rows.append(dict(estimand=est, pseudo=ps, matched_to=matched[ps], size_match_ratio=size_ratio.get(ps, np.nan),
                             n_testable=int(T.sum()), n_responder=int((o["status"] == 1).sum()), n_tail_at_level=tail,
                             n_ci_excl0=excl, n_conf_nonresp=int((o["status"] == 2).sum()), n_conf_strict=int(o["strict"].sum()),
                             n_ambiguous=int((o["status"] == 3).sum()),
                             min_q=float(np.nanmin(o["q"])) if np.isfinite(o["q"]).any() else np.nan))
    return rows, per_bin


# Stage 2 (serial, or merge): null_gate.tsv, verdict per estimand, expected[(estimand, real_pert)] = mean responders / testable
def gate_finalize(out_dir, rows, per_bin, tail_levels=None):
    df = pd.DataFrame(rows); df.to_csv(f"{out_dir}/null_gate.tsv", sep="\t", index=False, float_format="%.5g")
    verdict, expected = {}, {}
    for est, sub in df.groupby("estimand"):
        lvl = (tail_levels or {}).get(est)
        v = verdict[est] = gate_verdict(sub.n_responder.tolist(), sub.n_testable.tolist(), sub.n_tail_at_level.tolist(), lvl)
        nT = max(int(sub.n_testable.sum()), 1)
        v.update(ci_excl0_frac=float(sub.n_ci_excl0.sum() / nT), ci_coverage0=float(1 - sub.n_ci_excl0.sum() / nT),
                 conf_frac=float(sub.n_conf_nonresp.sum() / nT), conf_strict_frac=float(sub.n_conf_strict.sum() / nT))
        if est in per_bin:
            v["tail_ratio_by_bin"] = [float(per_bin[est][1, b] / (v["tail_level"] * per_bin[est][0, b])) if per_bin[est][0, b] > 0 else None
                                      for b in range(N_BINS)]
        print(f"NULL GATE {est}: {v['label']} (comparisons with >=1 responder {v['comparisons_with_responder']}/{v['K']} "
              f"[thr {v['primary_threshold']}]; tail ratio {v['tail_ratio']:.2f}; CI excludes 0 in {100*v['ci_excl0_frac']:.1f}% "
              f"(nominal 5%); CONF {100*v['conf_frac']:.1f}%)", flush=True)
        for pr, g in sub.groupby("matched_to"):
            expected[(est, pr)] = dict(resp=float(g.n_responder.mean()), testable=float(g.n_testable.mean()))
    return verdict, expected




def write_null_gate(out_dir, null_results, matched, size_ratio, tail_levels=None, bins=None):
    rows, per_bin = gate_rows(null_results, matched, size_ratio, tail_levels, bins)
    return gate_finalize(out_dir, rows, per_bin, tail_levels)


def null_columns(expected, verdict):
    def f(est, p):
        e = expected.get((est, p))
        return dict(null_expected_responders=(e["resp"] if e else np.nan), null_expected_testable=(e["testable"] if e else np.nan),
                    null_calibration=verdict.get(est, {}).get("label", "NOT_RUN"))
    return f


def finish_receipt(out_dir, rec, files):
    rec["output_sha256"] = {os.path.basename(f): sha256_file(f) for f in files if os.path.exists(f)}
    write_json(f"{out_dir}/receipt.json", rec)


# shared sparse pseudobulk accumulation (BM counts)
# SUM (ngroups,G) += per-group gene sums of blk rows; gid -1 = skip. tot = all-gene library totals
def add_group_sums(SUM, tot, cnt, blk, gid):
    rows = np.flatnonzero(gid >= 0)
    if rows.size == 0: return
    sub = blk if rows.size == blk.shape[0] else blk[rows]; g = gid[rows]
    uniq, inv = np.unique(g, return_inverse=True)
    Ind = sparse.csr_matrix((np.ones(rows.size, np.float32), (inv, np.arange(rows.size))), shape=(uniq.size, rows.size))
    part = Ind @ sub
    SUM[uniq] += part.toarray() if sparse.issparse(part) else part
    cp = np.asarray(sub.sum(1)).ravel()
    tot += np.bincount(g, weights=cp, minlength=tot.size); cnt += np.bincount(g, minlength=cnt.size)


# K562: A = dict of accumulators (n,L,C,Y,Y2,X,X2 flattened over groups). blk dense counts (cells,G)
def add_cell_stats(A, blk, gid):
    rows = np.flatnonzero(gid >= 0)
    if rows.size == 0: return
    sub = np.asarray(blk[rows], np.float32); g = gid[rows]
    L = sub.sum(1).astype(np.float64); Ls = np.maximum(L, 1.0)
    Y = np.log1p(sub * (1e4 / Ls)[:, None].astype(np.float32)).astype(np.float64)
    X = sub * (1e6 / Ls)[:, None].astype(np.float32)                         # N4: cpm moments in float32
    uniq, inv = np.unique(g, return_inverse=True)
    Ind = sparse.csr_matrix((np.ones(rows.size), (inv, np.arange(rows.size))), shape=(uniq.size, rows.size))
    I32 = Ind.astype(np.float32)
    A["C"][uniq] += (I32 @ sub); A["Y"][uniq] += Ind @ Y; A["Y2"][uniq] += Ind @ (Y * Y)
    A["X"][uniq] += I32 @ X; A["X2"][uniq] += I32 @ (X * X)
    A["L"] += np.bincount(g, weights=L, minlength=A["L"].size); A["n"] += np.bincount(g, minlength=A["n"].size)


# BM
# Si (U,C,G'),Ti,Ni (U,C) for the perturbation; nS.. same for NT
def bm_eval(Si, Ti, Ni, nS, nT, nN, pi, block, half, sidx, det, sm, minu, minu_s):
    d, v, n1, n0 = deltas_total(Si.sum(1), Ti.sum(1), Ni.sum(1), nS.sum(1), nT.sum(1), nN.sum(1), minu)
    out = {"TOTAL": classify(paired(d, v, n1, n0, block, half), det, sm)}
    d, v, n1, n0 = deltas_std(Si[:, sidx], Ti[:, sidx], Ni[:, sidx], nS[:, sidx], nT[:, sidx], nN[:, sidx], pi, block, minu_s)
    out["WITHIN_STATE"] = classify(paired(d, v, n1, n0, block, half), det, sm)
    return out


def bm_gene_ids(files):
    import h5py
    with h5py.File(files[0], "r") as f:
        ft = f["matrix"]["features"]; typ = np.array(ft["feature_type"]).astype(str)
        return np.array(ft["id"]).astype(str)[typ == "Gene Expression"]


def second_donor_inventory(others):
    inv = dict(other_files_in_GSE274113_dir=others, brief=BM_BRIEF, brief_lines_mentioning_donor=[])
    if os.path.isfile(BM_BRIEF):      # text brief only; no dataset is opened
        inv["brief_lines_mentioning_donor"] = [l.strip()[:300] for l in open(BM_BRIEF, errors="replace")
                                               if re.search(r"donor|second|GSE\d+", l, re.I)][:40]
    inv["NOTE"] = "Second-donor data are not loaded by this analysis."
    return inv


def cmd_bm(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, SCRATCH)
    import gb_gse274113_truth as TR          # reuse load_reps (:54-76), META (:40), CTRL (:41); lazy: needs h5py
    t0 = time.time()
    out_dir = a.out or f"{OUT}/BM{'_smalln' if a.smalln else ''}"; os.makedirs(out_dir, exist_ok=True)
    refuse_if_complete(f"{out_dir}/receipt.json", a.force)
    md = pd.read_csv(TR.META, index_col=0)
    md = md[md.index.notna() & md.target.notna()]
    rp = pd.Series(md.index.astype(str)).str.extract(r"^(rep\d+)_")[0].values
    if pd.isna(rp).any(): raise SystemExit("REFUSING: metadata barcodes without a repN_ prefix")
    md["repnum"] = pd.Series(rp).str[3:].astype(int).values
    reps = sorted(md.repnum.unique()); U = len(reps)
    assert set(reps) == BM_ALLOWED_REPS, f"replicates {reps} != allowed {sorted(BM_ALLOWED_REPS)}"
    tgt_all = sorted(t for t in md.target.astype(str).unique() if t != TR.CTRL)
    assert len(tgt_all) == 19, f"expected 19 targets, found {len(tgt_all)}: {tgt_all}"
    perts = first_three(tgt_all) if a.smalln else tgt_all
    null_on = a.null_gate or a.smalln; null_perts = list(perts) if null_on else []
    rep2day = md.groupby("repnum").Timepoint.agg(lambda s: str(s.mode().iloc[0])).to_dict()
    days = sorted(set(rep2day.values()), key=lambda s: (len(s), s)); day_id = {d: i for i, d in enumerate(days)}
    block = np.array([day_id[rep2day[r]] for r in reps]); half = np.array([0 if r % 2 == 1 else 1 for r in reps])
    files = sorted(glob.glob(f"{BM_DIR}/GSE274113_rep*_filtered_feature_bc_matrix.h5"))
    others = sorted((os.path.basename(p), os.path.getsize(p)) for p in glob.glob(f"{BM_DIR}/*")
                    if not re.search(r"_rep\d+_filtered_feature_bc_matrix\.h5$", p) and os.path.isfile(p))
    geno = sorted(p for d in (BM_DIR, BM_META_DIR) if os.path.isdir(d) for p in glob.glob(f"{d}/*")
                  if re.search(r"genotyp|editing|efficien", os.path.basename(p), re.I))
    rec = dict(phase="PREDECLARED", created=time.strftime("%Y-%m-%dT%H:%M:%S"), system="BM GSE274113 (Cas9 RNP KO)",
               script=os.path.abspath(__file__), script_sha256=sha256_file(os.path.abspath(__file__)), smalln=bool(a.smalln),
               perturbations=perts, n_targets=len(tgt_all), control=TR.CTRL, units=[f"rep{r}" for r in reps],
               unit_to_day={f"rep{r}": rep2day[r] for r in reps},
               day_weighting="equal weight per day (log2FC = mean over days of within-day mean of unit deltas)",
               split_definition=dict(rule="numeric replicate index parity", A_odd=[f"rep{r}" for r in reps if r % 2 == 1],
                                     B_even=[f"rep{r}" for r in reps if r % 2 == 0],
                                     note="each half re-estimated from its own units only; NT from the same units"),
               estimands=["TOTAL", "WITHIN_STATE (direct standardisation to fixed per-day NT reference composition)"],
               labels=LABEL_RULES, thresholds=thresholds(), min_unit_cells=a.min_unit_cells,
               min_unit_cells_strat=a.min_unit_cells_strat, null_gate=dict(on=null_on, matched_to=null_perts, M=NULL_M, seed=NULL_SEED),
               efficacy="EFFICACY_UNVERIFIED for all perturbations; no efficacy or DE-count selection. Genotyping "
                        "candidate files (listed, NOT parsed): " + json.dumps(geno),
               second_donor_inventory_names_only=second_donor_inventory(others), notes=NOTES,
               inputs=input_record([TR.META] + files, a.skip_hash), BLOCKED_BY_SPEC=BLOCKED_BY_SPEC)
    rec["null_gate_amendment"] = NULL_AMENDMENT
    write_json(f"{out_dir}/receipt.json", rec)
    print(f"predeclared: {len(perts)} perts, {U} units, days {days}, A(odd)={rec['split_definition']['A_odd']}", flush=True)

    X, cells, genes = TR.load_reps(); gene_ids = bm_gene_ids(files)
    keep = pd.Index(cells).isin(md.index); X, cells = X[keep], cells[keep]
    md = md.reindex(cells)
    tg = md.target.astype(str).values
    ct = md.new_CellType.astype(str).where(md.new_CellType.notna(), "NA").values
    types = sorted(set(ct)); tid = {c: i for i, c in enumerate(types)}; C = len(types)
    pseudo = {p: [f"NTNULL{m}::{p}" for m in range(NULL_M)] for p in null_perts}; pnames = [n for p in null_perts for n in pseudo[p]]
    lev = [TR.CTRL] + perts + pnames; lid = {t: i for i, t in enumerate(lev)}; Tn = len(lev)
    u_ = md.repnum.map({r: i for i, r in enumerate(reps)}).values.astype(int)
    c_ = np.array([tid[c] for c in ct]); ti = np.array([lid.get(t, -1) for t in tg])
    gid = np.where(ti >= 0, (u_ * Tn + ti) * C + c_, -1)
    ng = U * Tn * C
    SUM = np.zeros((ng, len(genes)), np.float32); tot = np.zeros(ng); cnt = np.zeros(ng)
    add_group_sums(SUM, tot, cnt, X, gid)                  # real perturbations + all NT cells
    size_ratio = {}
    if null_perts:       # N1: M independent uniform within-unit NT draws per real perturbation, no cell-type matching
        nt = np.flatnonzero(tg == TR.CTRL); pools = {u: nt[u_[nt] == u] for u in range(U)}
        for k, p in enumerate(null_perts):
            sz = {u: int(((u_ == u) & (tg == p)).sum()) for u in range(U)}
            for m, name in enumerate(pseudo[p]):
                dr = draw_pseudo(null_rng(k, m), pools, sz); idx = np.concatenate([dr[u] for u in sorted(dr)]).astype(np.int64)
                add_group_sums(SUM, tot, cnt, X[idx], (u_[idx] * Tn + lid[name]) * C + c_[idx])
                size_ratio[name] = len(idx) / max(sum(sz.values()), 1)
    ntm = tg == TR.CTRL
    det_all = np.asarray((X[ntm] > 0).sum(0)).ravel() / max(int(ntm.sum()), 1)
    gsel = np.flatnonzero(det_all >= MIN_DET)
    print(f"NT detection >= {MIN_DET}: {gsel.size}/{len(genes)} genes; {int(ntm.sum())} NT cells {time.time()-t0:.0f}s", flush=True)
    S = SUM[:, gsel].astype(np.float64).reshape(U, Tn, C, gsel.size); del SUM, X
    T = tot.reshape(U, Tn, C); N = cnt.reshape(U, Tn, C)
    strata = [c for c in types if c != "NA"]; sidx = [tid[c] for c in strata]
    pi = ref_composition(N[:, 0][:, sidx], block)
    gnames = np.asarray(genes); gu = np.char.upper(gnames[gsel].astype(str)); det1 = np.ones(gsel.size, bool)
    writer = PairWriter(f"{out_dir}/per_pair.tsv.gz")
    results = {"TOTAL": {}, "WITHIN_STATE": {}}; self_lfc = {}; ncells = {}
    for p in perts:
        i = lid[p]; sm = gu == p.upper()
        outs = bm_eval(S[:, i], T[:, i], N[:, i], S[:, 0], T[:, 0], N[:, 0], pi, block, half, sidx, det1, sm,
                       a.min_unit_cells, a.min_unit_cells_strat)
        for est, o in outs.items():
            self_lfc[(est, p)] = float(o["res"]["lfc"][sm][0]) if (o["res"] is not None and sm.any()) else np.nan
            results[est][p] = o; writer.add(est, p, gnames[gsel], o)
        ncells[p] = int(N[:, i].sum())
        print("  %s: TOTAL T=%d R=%d | WITHIN T=%d R=%d" % (p, outs["TOTAL"]["T"].sum(), (outs["TOTAL"]["status"] == 1).sum(),
              outs["WITHIN_STATE"]["T"].sum(), (outs["WITHIN_STATE"]["status"] == 1).sum()), flush=True)
    writer.close()
    gate, expected = {}, {}
    if null_perts:       # each pseudo is tested against its own complement NT (NT minus its cells) in every unit
        nres = {"TOTAL": {}, "WITHIN_STATE": {}}
        for name in pnames:
            j = lid[name]
            outs = bm_eval(S[:, j], T[:, j], N[:, j], S[:, 0] - S[:, j], T[:, 0] - T[:, j], N[:, 0] - N[:, j], pi, block, half,
                           sidx, det1, np.zeros(gsel.size, bool), a.min_unit_cells, a.min_unit_cells_strat)
            for est, o in outs.items(): nres[est][name] = o
        gate, expected = write_null_gate(out_dir, nres, {n: p for p in null_perts for n in pseudo[p]}, size_ratio)
        rec["null_gate_result"] = gate
    ncol = null_columns(expected, gate)
    build_outputs(out_dir, gnames, gene_ids, det_all, gsel, perts, results, {"BM19" if not a.smalln else "BM_SMALLN": perts}, self_lfc,
                  lambda est, p: dict(n_cells=ncells[p], n_units_present=int((N[:, lid[p]].sum(1) > 0).sum()),
                                      efficacy_status="EFFICACY_UNVERIFIED", **ncol(est, p)))
    pd.DataFrame(N.sum(0)[:len(perts) + 1], index=lev[:len(perts) + 1], columns=types).to_csv(f"{out_dir}/cells_by_target_celltype.tsv", sep="\t")
    rec.update(phase="COMPLETE", n_cells=int(N[:, 1:1 + len(perts)].sum() + N[:, 0].sum()), n_genes_total=len(genes),
               n_genes_testable_floor=int(gsel.size), cell_types=types, wall_s=time.time() - t0)
    finish_receipt(out_dir, rec, [f"{out_dir}/{n}" for n in ("per_pair.tsv.gz", "per_gene.tsv", "perturbations.tsv", "null_gate.tsv")])
    print(f"wrote {out_dir} in {time.time()-t0:.0f}s", flush=True)


# K562
def read_col(grp, name):
    import h5py
    o = grp[name]
    if isinstance(o, h5py.Group):
        return np.asarray(o["categories"][:]).astype(str)[np.asarray(o["codes"][:])]
    return np.asarray(o[:]).astype(str)


# Yield (lo, hi, block) row chunks from a dense dataset or a CSR group (anndata layout)
def iter_chunks(f, path, ncols, chunk):
    import h5py
    node = f[path]
    if isinstance(node, h5py.Dataset):
        for lo in range(0, node.shape[0], chunk):
            hi = min(lo + chunk, node.shape[0]); yield lo, hi, np.asarray(node[lo:hi], dtype=np.float32)
        return
    enc = node.attrs.get("encoding-type", node.attrs.get("h5sparse_format", ""))
    if isinstance(enc, bytes): enc = enc.decode()
    if "csr" not in str(enc): raise SystemExit(f"BLOCKED_BY_SPEC: {path} sparse encoding {enc!r} is not CSR")
    ip = node["indptr"][:]; n = len(ip) - 1
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n); s, e = ip[lo], ip[hi]
        yield lo, hi, sparse.csr_matrix((node["data"][s:e].astype(np.float32), node["indices"][s:e], ip[lo:hi + 1] - s),
                                        shape=(hi - lo, ncols))


def is_integer_counts(blk):
    v = blk.data if sparse.issparse(blk) else blk[blk != 0]
    return bool(v.size and v.min() >= 0 and np.all(v == np.round(v)))


def pick_counts(f, ncols):
    cands = (["layers/counts"] if "layers" in f and "counts" in f["layers"] else []) + ["X"]
    for c in cands:
        if is_integer_counts(next(iter_chunks(f, c, ncols, 2000))[2]): return c
    raise SystemExit("BLOCKED_BY_SPEC[NO_RAW_COUNTS]: neither layers/counts nor X is integer-valued. "
                     "Run `inspect --which k562` to check count availability.")


def k562_obs(f):
    ob = f["obs"]; keys = list(ob.keys())
    unit_col = next((c for c in ("gem_group", "batch") if c in keys), None)
    if unit_col is None or "gene" not in keys:
        raise SystemExit(f"BLOCKED_BY_SPEC: need obs['gene'] and obs['gem_group'|'batch']; obs keys = {keys}")
    return read_col(ob, "gene"), read_col(ob, unit_col), unit_col, keys


# symbols, Ensembl-like ids (or None), coords dict (chr,start,end[,strand]) or None
def k562_var(f):
    v = f["var"]; keys = list(v.keys())
    sym = read_col(v, "gene_name")
    idc = next((c for c in ("ensembl_id", "ensemble_id", "gene_id", "gene_ids", "ensembl", "_index") if c in keys), None)
    ids = read_col(v, idc) if idc else np.array([""] * len(sym))
    lc = {k.lower(): k for k in keys}
    ch = next((lc[k] for k in ("chr", "chromosome", "chrom", "seqname") if k in lc), None)
    st, en = lc.get("start"), lc.get("end")
    coords = None
    if ch and st and en:
        coords = dict(chr=read_col(v, ch), start=pd.to_numeric(pd.Series(read_col(v, st)), errors="coerce").values,
                      end=pd.to_numeric(pd.Series(read_col(v, en)), errors="coerce").values,
                      strand=(read_col(v, lc["strand"]) if "strand" in lc else None))
    return sym, ids, coords, keys


# genes (over G') whose body lies within NEIGHBOR_BP of the target TSS (start if + / unknown, end if -), excluding itself
def neighbor_mask(coords, gsel, symbols_up, pert):
    j = np.flatnonzero(symbols_up == pert.upper())
    if coords is None or j.size == 0: return np.zeros(len(gsel), bool)
    j = j[0]; minus = coords["strand"] is not None and str(coords["strand"][j]) in ("-", "-1", "minus")
    tss = coords["end"][j] if minus else coords["start"][j]
    with np.errstate(invalid="ignore"):
        m = (coords["chr"] == coords["chr"][j]) & (coords["end"] >= tss - NEIGHBOR_BP) & (coords["start"] <= tss + NEIGHBOR_BP)
    m[j] = False
    return np.asarray(m)[gsel]


def cmd_inspect(a):
    if a.which == "bm":
        md = pd.read_csv(f"{BM_DIR}/GSE274113_annotated_metadata.csv.gz", index_col=0, low_memory=False)
        print("columns:", list(md.columns)); print(md.head(3).T)
        md = md[md.target.notna()]
        print(md.target.value_counts().to_string()); print(pd.crosstab(md.Timepoint, md.new_CellType))
        print(sorted(os.path.basename(p) for p in glob.glob(f"{BM_DIR}/*") if "_rep" not in p))
        print("candidate genotyping files:", [p for d in (BM_DIR, BM_META_DIR) if os.path.isdir(d) for p in glob.glob(f"{d}/*")
                                             if re.search(r"genotyp|editing|efficien", p, re.I)])
        return
    import h5py
    f = h5py.File(K562_SRC, "r")
    print("root keys:", list(f.keys()), "| layers:", list(f["layers"].keys()) if "layers" in f else None)
    print("obs keys:", list(f["obs"].keys()), "| var keys:", list(f["var"].keys()))
    print("X:", type(f["X"]).__name__, dict(f["X"].attrs) if hasattr(f["X"], "attrs") else "")
    lab, unit, ucol, _ = k562_obs(f)
    sym, ids, coords, _ = k562_var(f)
    print(f"unit column = {ucol}: {len(set(unit))} units; cells per unit:", pd.Series(unit).value_counts().sort_index().to_dict())
    vc = pd.Series(lab).value_counts()
    print(f"{len(vc)} labels; NT cells = {int(vc.get(K562_NT, 0))}; cells/pert quantiles:", vc.drop(K562_NT, errors="ignore").quantile([0, .1, .5, .9, 1]).to_dict())
    print("F2-36 missing from screen:", [t for t in K562_F2_36 if t not in set(lab) and t not in {x.upper() for x in lab}])
    print("n genes:", len(sym), "| ids example:", ids[:3], "| coordinates available:", coords is not None)
    try: print("count source =", pick_counts(f, len(sym)))
    except SystemExit as e: print(e)


E_MAIN, E_LEG, E_ALT = "TOTAL", "TOTAL_LEGACY_SATTERTHWAITE", "TOTAL_ALT_EMP_POOLED_BINNED"
EST_ORDER = [E_MAIN, E_LEG, E_ALT]


def parse_shard(txt):
    if not txt: return None
    m = re.fullmatch(r"(\d+)/(\d+)", txt)
    if not m or not (0 <= int(m.group(1)) < int(m.group(2))): raise SystemExit(f"--shard must be i/n with 0<=i<n, got {txt!r}")
    return int(m.group(1)), int(m.group(2))


# One pass: per-(gem group x perturbation) sufficient stats (A) + the NT cells' count rows (NTC, kept in memory)
def k562_stream(chunks, lab, u_, lid, U, G, log=None):
    P = len(lid); is_nt = lab == K562_NT; n_nt = int(is_nt.sum())
    gid = np.array([(u * P + lid[l]) if l in lid else -1 for u, l in zip(u_, lab)])
    A = alloc_stats(U * P, G); NTC = np.zeros((n_nt, G), np.float32); nfill = 0
    for lo, hi, blk in chunks:
        if sparse.issparse(blk): blk = blk.toarray()
        if not is_integer_counts(blk): raise SystemExit(f"BLOCKED_BY_SPEC[NON_INTEGER_CHUNK]: rows {lo}-{hi} are not integer counts")
        add_cell_stats(A, blk, gid[lo:hi])
        m = np.flatnonzero(is_nt[lo:hi])
        if m.size: NTC[nfill:nfill + m.size] = blk[m]; nfill += m.size       # NT cells kept in memory: null draws need no re-streaming
        if log: log(hi)
    assert nfill == n_nt
    return A, NTC


# Everything derived from the stream that the per-perturbation loop needs (identical in every shard)
def k562_prepare(A, NTC, lab, u_, U, G, lid, genes, gene_ids, coords, half, arms, null_perts):
    P = len(lid); n_nt = NTC.shape[0]; run = sorted(arms["ALL_PERT"])
    det_all = (NTC > 0).sum(0) / max(n_nt, 1); gsel = np.flatnonzero(det_all >= MIN_DET)
    D = {"n": A["n"].reshape(U, P), "L": A["L"].reshape(U, P)}
    for k in ("C", "Y", "Y2", "X", "X2"):
        D[k] = A.pop(k)[:, gsel].astype(np.float32 if k in ("X", "X2") else np.float64).reshape(U, P, gsel.size)   # frees one full array per step
    nt_u = u_[lab == K562_NT]
    ctx = nt_context(np.ascontiguousarray(NTC[:, gsel]), NTC.sum(1, dtype=np.float64), nt_u, U)
    gu_all = np.char.upper(np.asarray(genes).astype(str))
    return SimpleNamespace(D=D, ntf={k: v[:, 0] for k, v in D.items()}, NTC=NTC, nt_u=nt_u, u_=u_, lab=lab, U=U, G=G, lid=lid, n_nt=n_nt,
                           ctx=ctx, pools_full=ctx["pools"], det_all=det_all, gsel=gsel, genes=np.asarray(genes), gene_ids=np.asarray(gene_ids),
                           coords=coords, half=half, arms=arms, run=run, null_perts=list(null_perts), gu_all=gu_all, gu=gu_all[gsel],
                           det1=np.ones(gsel.size, bool), zero=np.zeros(gsel.size, bool),
                           pseudo={p: [f"NTNULL{m}::{p}" for m in range(NULL_M)] for p in null_perts})


def common_hash(inp_or_c):
    return hashlib.sha256(np.asarray(inp_or_c.gsel, np.int64).tobytes() + np.asarray(inp_or_c.det_all, np.float64).tobytes()
                          + "|".join(map(str, inp_or_c.genes)).encode()).hexdigest()


# Real-perturbation loop + held-out null gate
def k562_run(a, out_dir, rec, inp, shard, t0, tm):
    sl = lambda d, i: {k: v[:, i] for k, v in d.items()}
    D, ntf, ctx, U, lid, half = inp.D, inp.ntf, inp.ctx, inp.U, inp.lid, inp.half
    i_, n_ = shard if shard else (0, 1); tag = f"shard_{i_}_of_{n_}"; parts = f"{out_dir}/parts"
    if shard: os.makedirs(parts, exist_ok=True)
    tails = {E_MAIN: tail_level("gene", a.m_null, a.m_max), E_ALT: NULL_TAIL_P, E_LEG: NULL_TAIL_P}
    mine = [p for k, p in enumerate(inp.run) if k % n_ == i_]
    writer = PairWriter(f"{parts}/{tag}.per_pair.tsv.gz" if shard else f"{out_dir}/per_pair.tsv.gz")
    results = {e: {} for e in EST_ORDER}; self_lfc = {}; pinfo = {}; cellmeta = {}; t1 = time.time()
    print(f"empirical null: M0={a.m_null}, refinement to M1={a.m_max} (trigger <={REFINE_TRIGGER}, stop at {REFINE_STOP} exceedances); "
          f"{'shard %d/%d: ' % (i_, n_) if shard else ''}{len(mine)} perturbations", flush=True)
    for k, p in enumerate(mine):
        i = lid[p]; sm = inp.gu == p.upper(); res = contrast_cell(sl(D, i), ntf, half)
        if res is None: rmain = ralt = rleg = None; info = {}
        else: rmain, ralt, rleg, info = empirical_test(res, ctx, inp.pools_full, ntf, null_rng(101, pert_seed(p)), a.m_null, a.m_max, a.refine_batch)
        om, oa, ol = classify(rmain, inp.det1, sm), classify(ralt, inp.det1, sm), classify(rleg, inp.det1, sm)
        if inp.coords is not None:
            nb = neighbor_mask(inp.coords, inp.gsel, inp.gu_all, p)
            for o in (om, oa, ol): o["nbr"] = nb
        self_lfc[p] = float(res["lfc"][sm][0]) if (res is not None and sm.any()) else np.nan
        results[E_MAIN][p], results[E_ALT][p], results[E_LEG][p] = om, oa, ol
        cellmeta[p] = dict(n_cells=int(D["n"][:, i].sum()), n_units_present=int((D["n"][:, i] > 0).sum()))
        if info:
            Mt = info["Mg"][om["T"]]
            pinfo[p] = dict(cap_units_binding=info["cap_units"], cap_cell_frac_lost=info["cap_frac"], n_zero_sd_genes=info["n_zero_sd"],
                            n_refined_genes=info["n_refined"], median_M_gene_T=float(np.median(Mt)) if Mt.size else np.nan)
            writer.add(E_MAIN, p, inp.genes[inp.gsel], om, variants={"LEGACY": ol, "ALT": oa}, extra={"n_null_draws": info["Mg"]})
        if k % 100 == 0: print(f"  pert {k}/{len(mine)} {time.time()-t0:.0f}s", flush=True)
    writer.close(); tm["null_real_s"] = time.time() - t1
    rows, per_bin, pkey, mine_null = [], {}, {}, []
    if inp.null_perts:   # independent held-OUT pseudo-perturbations: own draws, own reference (NT minus the pseudo's cells), own null
        t2 = time.time(); nres = {e: {} for e in EST_ORDER}; nfull = inp.ntf; size_ratio = {}; sp = sorted(inp.null_perts)
        mine_null = [p for kk, p in enumerate(sp) if kk % n_ == i_]
        for kk, p in enumerate(inp.null_perts):
            if p not in mine_null: continue
            sz = {u: int(((inp.u_ == u) & (inp.lab == p)).sum()) for u in range(U)}
            for m, name in enumerate(inp.pseudo[p]):
                pkey[name] = [kk, m]
                dr = draw_pseudo(null_rng(102, pert_seed(p), m), {u: inp.pools_full[u] for u in range(U)}, sz)      # seeds: sha1(name)
                idx = np.concatenate([dr[u] for u in sorted(dr)]).astype(np.int64); size_ratio[name] = len(idx) / max(sum(sz.values()), 1)
                Ps = stats_sub(inp.NTC[idx], inp.nt_u[idx], U, inp.gsel); comp = {q: nfull[q] - Ps[q] for q in Ps}
                res = contrast_cell(Ps, comp, half)
                if res is None: rm = ra = rl = None
                else:
                    keep = np.ones(inp.n_nt, bool); keep[idx] = False
                    rm, ra, rl, _ = empirical_test(res, ctx, [np.flatnonzero(keep & (inp.nt_u == u)) for u in range(U)], comp,
                                                   null_rng(103, pert_seed(p), m), a.m_null, a.m_max, a.refine_batch)
                for e, r_ in ((E_MAIN, rm), (E_ALT, ra), (E_LEG, rl)): nres[e][name] = classify(r_, inp.det1, inp.zero)
        rows, per_bin = gate_rows(nres, {n: p for p in inp.null_perts for n in inp.pseudo[p]}, size_ratio, tails, ctx["bins"])
        tm["gate_s"] = time.time() - t2
    facts = dict(n_cells=int(D["n"].sum()), n_nt_cells=int(inp.n_nt), n_genes_total=int(inp.G), n_genes_testable_floor=int(inp.gsel.size))
    if shard:                                   # write this shard's parts
        arrs = dict(perts=np.array(mine, dtype=object), genes=inp.genes.astype(str), gene_ids=inp.gene_ids.astype(str),
                    det_all=inp.det_all, gsel=inp.gsel)
        Gp = inp.gsel.size
        for j, e in enumerate(EST_ORDER):
            for key, dt in (("status", np.int8), ("e", float), ("e_legacy", float), ("se", float), ("strict", bool)):
                arrs[f"e{j}_{key}"] = np.array([results[e][p][key] for p in mine], dt).reshape(len(mine), Gp)
            arrs[f"e{j}_nu"] = np.array([results[e][p]["res"]["nu"] if results[e][p]["res"] is not None else -1 for p in mine], int)
        if inp.coords is not None: arrs["nbr"] = np.array([results[E_MAIN][p]["nbr"] for p in mine], bool).reshape(len(mine), Gp)
        np.savez(f"{parts}/{tag}.npz", **arrs)
        write_json(f"{parts}/{tag}.meta.json", dict(perts={p: dict(self_lfc=self_lfc[p], pinfo=pinfo.get(p), **cellmeta[p]) for p in mine},
                                                    gate=dict(rows=rows, per_bin={e: v.tolist() for e, v in per_bin.items()}, pkey=pkey)))
        rec.update(phase="COMPLETE", timing_s=tm, wall_s=time.time() - t0, facts=facts, common_sha256=common_hash(inp),
                   shard=dict(i=i_, n=n_, perts_processed=mine, null_perts_processed=mine_null))
        write_json(f"{parts}/{tag}.receipt.json", rec)
        print(f"wrote shard {i_}/{n_} parts in {time.time()-t0:.0f}s  timing {tm}", flush=True); return
    k562_finalize(out_dir, rec, inp, results, self_lfc, pinfo, cellmeta, rows, per_bin, tails, facts, tm, t0)


# All aggregation
def k562_finalize(out_dir, rec, common, results, self_lfc, pinfo, cellmeta, rows, per_bin, tails, facts, tm, t0):
    gate, expected = {}, {}
    if rows:
        gate, expected = gate_finalize(out_dir, rows, per_bin, tails); rec["null_gate_result"] = gate
    ncol = null_columns(expected, gate)
    build_outputs(out_dir, common.genes, common.gene_ids, common.det_all, common.gsel, common.run, results, common.arms,
                  {(E_MAIN, p): v for p, v in self_lfc.items()},
                  lambda est, p: dict(n_cells=cellmeta[p]["n_cells"], n_units_present=cellmeta[p]["n_units_present"],
                                      efficacy_status="NOT_FILTERED", **(pinfo.get(p) or {}), **ncol(est, p)))
    rec.update(phase="COMPLETE", timing_s=tm, wall_s=time.time() - t0, **facts)
    finish_receipt(out_dir, rec, [f"{out_dir}/{n}" for n in ("per_pair.tsv.gz", "per_gene.tsv", "perturbations.tsv", "null_gate.tsv")])
    print(f"wrote {out_dir} in {time.time()-t0:.0f}s  timing {tm}", flush=True)


# Assemble the final K562 files from n shards
def k562_merge(out_dir, n):
    parts = f"{out_dir}/parts"; t0 = time.time()
    if os.path.exists(f"{out_dir}/receipt.json"): raise SystemExit(f"REFUSING: {out_dir}/receipt.json already exists (merge writes the final files)")
    recs, metas, npzs = [], [], []
    for i in range(n):
        tag = f"{parts}/shard_{i}_of_{n}"
        miss = [x for x in (".receipt.json", ".meta.json", ".npz", ".per_pair.tsv.gz") if not os.path.exists(tag + x)]
        if miss: raise SystemExit(f"REFUSING: shard {i}/{n} is missing {miss}")
        r = json.load(open(tag + ".receipt.json"))
        if r.get("phase") != "COMPLETE" or r.get("shard", {}).get("i") != i or r["shard"].get("n") != n:
            raise SystemExit(f"REFUSING: shard {i}/{n} receipt is not a COMPLETE i={i} n={n} receipt")
        recs.append(r); metas.append(json.load(open(tag + ".meta.json"))); npzs.append(np.load(tag + ".npz", allow_pickle=True))
    for key in ("script_sha256", "args", "common_sha256", "arms", "units", "thresholds", "count_source", "null_gate"):
        if any(json.dumps(r.get(key), sort_keys=True) != json.dumps(recs[0].get(key), sort_keys=True) for r in recs):
            raise SystemExit(f"REFUSING: shards disagree on {key!r}")
    run = sorted(recs[0]["arms"]["ALL_PERT"]["perts"]); got = [p for r in recs for p in r["shard"]["perts_processed"]]
    if sorted(got) != run: raise SystemExit(f"REFUSING: shards do not cover the perturbation list exactly once ({len(got)} processed vs {len(run)})")
    nl = sorted(recs[0]["null_gate"]["matched_to"]); gotn = [p for r in recs for p in r["shard"]["null_perts_processed"]]
    if sorted(gotn) != nl: raise SystemExit("REFUSING: shards do not cover the gate perturbations exactly once")
    for z in npzs[1:]:
        for key in ("genes", "gene_ids", "det_all", "gsel"):
            if not np.array_equal(z[key], npzs[0][key]): raise SystemExit(f"REFUSING: shards disagree on {key}")
    z0 = npzs[0]; common = SimpleNamespace(genes=z0["genes"].astype(str), gene_ids=z0["gene_ids"].astype(str), det_all=z0["det_all"],
                                            gsel=z0["gsel"], run=run, arms=recs[0]["arms"])
    common.arms = {k: v["perts"] for k, v in recs[0]["arms"].items()}
    if common_hash(common) != recs[0]["common_sha256"]: raise SystemExit("REFUSING: common arrays do not match the receipt hash")
    # per_pair: re-emit each perturbation's block in global sorted order (identical bytes to the
    # serial file)
    blocks, header = {}, None
    for i in range(n):
        with gzip.open(f"{parts}/shard_{i}_of_{n}.per_pair.tsv.gz", "rt", newline="") as fh: lines = fh.readlines()
        if not lines: continue
        header = header or lines[0]
        for ln in lines[1:]: blocks.setdefault(ln.split("\t", 2)[1], []).append(ln)
    w = PairWriter(f"{out_dir}/per_pair.tsv.gz")
    for p in run:
        if p in blocks: w.add_block("".join(blocks[p]), header)
    w.close()
    results = {e: {} for e in EST_ORDER}; self_lfc, pinfo, cellmeta = {}, {}, {}; Gp = common.gsel.size
    for z, m in zip(npzs, metas):
        for k, p in enumerate(z["perts"].astype(str)):
            for j, e in enumerate(EST_ORDER):
                nu = int(z[f"e{j}_nu"][k]); st = z[f"e{j}_status"][k]
                o = dict(status=st, T=st > 0, e=z[f"e{j}_e"][k], e_legacy=z[f"e{j}_e_legacy"][k], se=z[f"e{j}_se"][k],
                         strict=z[f"e{j}_strict"][k], res=(dict(nu=nu) if nu >= 0 else None))
                if "nbr" in z.files: o["nbr"] = z["nbr"][k]
                results[e][p] = o
            pm = m["perts"][p]; self_lfc[p] = pm["self_lfc"]; pinfo[p] = pm["pinfo"]
            cellmeta[p] = dict(n_cells=pm["n_cells"], n_units_present=pm["n_units_present"])
    rows, pkey, per_bin = [], {}, {}
    for m in metas:
        rows += m["gate"]["rows"]; pkey.update(m["gate"]["pkey"])
        for e, v in m["gate"]["per_bin"].items(): per_bin[e] = per_bin.get(e, 0) + np.array(v)
    rows.sort(key=lambda r: (EST_ORDER.index(r["estimand"]), pkey[r["pseudo"]]))
    tm = {k: max(r["timing_s"].get(k, 0) for r in recs) for k in ("stream_s", "null_real_s", "gate_s")}
    rec = {k: v for k, v in recs[0].items() if k not in ("shard", "timing_s", "wall_s", "created")}
    rec["created"] = time.strftime("%Y-%m-%dT%H:%M:%S"); rec["sharding"] = dict(n=n, shard_receipts=[f"parts/shard_{i}_of_{n}.receipt.json" for i in range(n)],
                                                                                 shard_wall_s=[r["wall_s"] for r in recs])
    tails = {E_MAIN: tail_level("gene", rec["args"]["m_null"], rec["args"]["m_max"]), E_ALT: NULL_TAIL_P, E_LEG: NULL_TAIL_P}
    facts = recs[0]["facts"]; rec.pop("facts", None)
    k562_finalize(out_dir, rec, common, results, self_lfc, pinfo, cellmeta, rows, per_bin, tails, facts, tm, t0)


def cmd_k562(a):
    import h5py
    t0 = time.time(); shard = parse_shard(a.shard)
    out_dir = a.out or f"{OUT}/K562_{'EMP_SHARD' if shard else 'EMP'}{'_smalln' if a.smalln else ''}"
    ab = os.path.abspath(out_dir).rstrip("/")
    if ab == f"{OUT}/K562": raise SystemExit(f"REFUSING: never overwrite {OUT}/K562 (the Welch-era run)")
    if shard and ab.startswith(f"{OUT}/K562_EMP") and not ab.startswith(f"{OUT}/K562_EMP_SHARD"):
        raise SystemExit("REFUSING: sharded runs must not write into the serial K562_EMP directory")
    os.makedirs(out_dir, exist_ok=True)
    rp = f"{out_dir}/parts/shard_{shard[0]}_of_{shard[1]}.receipt.json" if shard else f"{out_dir}/receipt.json"
    if shard: os.makedirs(f"{out_dir}/parts", exist_ok=True)
    refuse_if_complete(rp, a.force)
    f = h5py.File(K562_SRC, "r")
    lab, unit, ucol, _ = k562_obs(f)
    genes, gene_ids, coords, _ = k562_var(f); G = len(genes)
    if K562_NT not in set(lab): raise SystemExit(f"REFUSING: no {K562_NT!r} label")
    lam = {l.strip().upper() for l in open(LAMBERT) if l.strip()}
    allp = sorted(set(lab) - {K562_NT}); f2 = [t for t in K562_F2_36 if t in set(allp)]
    if a.smalln: f2 = first_three(f2); allp = f2
    arms = {"F2_36": f2, "LAMBERT_TF": [p for p in allp if p.upper() in lam], "ALL_PERT": allp}
    null_on = a.null_gate or a.smalln; null_perts = list(f2) if null_on else []
    ulev = sorted(set(unit), key=lambda s: (0, int(s)) if re.fullmatch(r"\d+", s) else (1, s))
    num = all(re.fullmatch(r"\d+", s) for s in ulev); rank = {s: i + 1 for i, s in enumerate(ulev)}
    half = np.array([0 if (int(s) if num else rank[s]) % 2 == 1 else 1 for s in ulev]); U = len(ulev)
    rec = dict(phase="PREDECLARED", created=time.strftime("%Y-%m-%dT%H:%M:%S"), system="K562 Replogle essential (CRISPRi)",
               script=os.path.abspath(__file__), script_sha256=sha256_file(os.path.abspath(__file__)), smalln=bool(a.smalln),
               args=dict(smalln=bool(a.smalln), null_gate=bool(null_on), m_null=a.m_null, m_max=a.m_max, refine_batch=a.refine_batch),
               unit_column=ucol, units=ulev,
               split_definition=dict(rule=("integer parity of " if num else "parity of 1-based sorted rank of ") + ucol,
                                     A_odd=[s for s, h in zip(ulev, half) if h == 0], B_even=[s for s, h in zip(ulev, half) if h == 1],
                                     note="each half re-estimated from its own gem groups; NT from the same gem groups. "
                                          "Halves are technical lanes of ONE culture; one dual-guide construct per gene (no guide replication)"),
               arms={k: dict(n=len(v), perts=v) for k, v in arms.items()},
               arm_notes=dict(F2_36="frozen 36 F2 endpoint TFs (gb_cop_alignment.py:252) intersected with screen; missing: "
                                     + json.dumps([t for t in K562_F2_36 if t not in set(lab)]),
                              LAMBERT_TF="all screen perturbations whose label is in Lambert 2018 catalogue",
                              ALL_PERT="all perturbations; SEPARATE estimand (essential genes)"),
               estimands=["TOTAL"], labels=LABEL_RULES, thresholds=thresholds(), efficacy="not filtered; self_log2fc informational only",
               promoter_neighbor_sensitivity=("AVAILABLE (var chr/start/end; strand=%s)" % (coords is not None and coords["strand"] is not None)
                                              if coords is not None else "UNAVAILABLE: var has no chr/start/end"),
               null_gate=dict(on=null_on, matched_to=null_perts, M=NULL_M, seed=NULL_SEED), notes=NOTES,
               k562_test=dict(primary=f"EMPIRICAL per-gene null with ADAPTIVE REFINEMENT: M0={a.m_null} seeded draws for every gene; genes with "
                                      f"<= {REFINE_TRIGGER} exceedances get more draws (batches of {REFINE_BATCH}, only their columns) until {REFINE_STOP} "
                                      f"exceedances or M1={a.m_max}; p=(1+#{{|null|>=|obs|}})/(1+M_gene) (M_gene per pair in per_pair.n_null_draws); BH within "
                                      "perturbation. Draws = NT pseudo-perturbations with the perturbation's exact gem-group mix (n*_u=min(n_u,floor(N_u/2)); cap "
                                      "binding reported per perturbation), compared with the rest of NT in the same gem groups; statistic = gem-group-matched mean "
                                      "difference on log1p(CP10K). Per-perturbation seed = sha1(name)[:8]. CI/equivalence on the count-scale log2FC with "
                                      "se_cal=max(delta-method se, first-stage null SD of that log2FC), normal quantile.",
                              secondary=f"TOTAL_ALT_EMP_POOLED_BINNED: null studentised by the gene's null SD and pooled WITHIN {N_BINS} NT-expression-decile "
                                        "bins (first-stage draws only). Secondary; per-bin gate tail ratio reported.",
                              zero_sd="genes whose first-stage null SD is 0 are UNTESTED in every mode (incl. legacy); counted per perturbation",
                              legacy="Welch/Satterthwaite kept as estimand TOTAL_LEGACY_SATTERTHWAITE and p/q/status_LEGACY columns; "
                                     "it FAILED the null gate (150/180 pseudo comparisons with a responder, tail ratio 3.24).",
                              gate="held-out pseudo-perturbations: own draws (seed from sha1 of the matched perturbation name), own reference "
                                   "(NT minus the pseudo's cells), own null; never used to build any real null",
                              bucket_sharing="not used (exact per-perturbation draws)",
                              sharding="--shard i/n processes perturbations at sorted-position % n == i (statistics identical to serial; k562-merge)"),
               inputs=input_record([K562_SRC, LAMBERT], a.skip_hash or bool(shard and shard[0] != 0)), BLOCKED_BY_SPEC=BLOCKED_BY_SPEC)   # shards: only shard 0 hashes the inputs
    rec["null_gate_amendment"] = NULL_AMENDMENT
    write_json(rp, rec)
    print(f"predeclared: arms { {k: len(v) for k, v in arms.items()} }, {U} units ({ucol}), A(odd) = {rec['split_definition']['A_odd']}"
          + (f"; SHARD {shard[0]}/{shard[1]}" if shard else ""), flush=True)

    src = pick_counts(f, G); rec["count_source"] = src; tm = {}
    lev = [K562_NT] + sorted(allp); lid = {t: i for i, t in enumerate(lev)}; P = len(lev)
    uid = {s: i for i, s in enumerate(ulev)}; u_ = np.array([uid[s] for s in unit]); n_nt = int((lab == K562_NT).sum())
    est_bytes = U * P * G * (4 + 8 + 8 + 4 + 4) + U * P * G * 8 + n_nt * G * 8
    if est_bytes > 40e9: raise SystemExit(f"REFUSING: est. {est_bytes/1e9:.0f} GB (U={U} x P={P} x G={G}); reduce arm")
    A, NTC = k562_stream(iter_chunks(f, src, G, a.chunk), lab, u_, lid, U, G, log=lambda hi: print(f"  {hi}/{len(lab)} cells {time.time()-t0:.0f}s", flush=True))
    f.close(); tm["stream_s"] = time.time() - t0
    inp = k562_prepare(A, NTC, lab, u_, U, G, lid, genes, gene_ids, coords, half, arms, null_perts)
    print(f"NT detection >= {MIN_DET}: {inp.gsel.size}/{G} genes; {n_nt} NT cells", flush=True)
    k562_run(a, out_dir, rec, inp, shard, t0, tm)


# selftest
# Synthetic paired BM-like design: U units, 4 day-blocks, alternating halves
def _syn_bm(rng, U, G, eff, cells=200):
    block = np.repeat(np.arange(4), U // 4); half = np.arange(U) % 2
    base = rng.uniform(1.0, 4.0, G)
    Sn = rng.poisson(base * cells, size=(U, G)).astype(float); mult = np.ones((U, G))
    for g, e in eff.items():
        if e == "split": mult[half == 0, g] = 2.0; mult[half == 1, g] = 0.5
        else: mult[:, g] = e
    Sp = rng.poisson(base * cells * mult).astype(float); Nc = np.full(U, cells)
    return (Sp, Sp.sum(1), Nc, Sn, Sn.sum(1), Nc.copy()), block, half


# Synthetic K562-like cells: n_gg gem groups; pert with npert[gg] cells; returns accumulated per-gg dict P, NT
def _syn_cells(rng, G, npert, effect, batch_mult=None, nt_per_gg=300, n_gg=6):
    base = rng.uniform(0.5, 3.0, G); rows, lab, gg = [], [], []
    for u in range(n_gg):
        bm = np.ones(G) if batch_mult is None or u % 2 == 0 else batch_mult      # odd index (u=1,3,5) = 'even' gem group numbers
        for name, n, eff in (("NT", nt_per_gg, np.ones(G)), ("P", npert[u], effect)):
            lib = rng.gamma(20, 1 / 20, n)[:, None]
            rows.append(rng.poisson(base * bm * eff * lib * 4, size=(n, G))); lab += [name] * n; gg += [u] * n
    cnt = np.vstack(rows).astype(np.float32); lab = np.array(lab); gg = np.array(gg)
    gid = np.where(lab == "NT", gg * 2 + 0, gg * 2 + 1)
    A = dict(n=np.zeros(n_gg * 2), L=np.zeros(n_gg * 2), C=np.zeros((n_gg * 2, G), np.float32), Y=np.zeros((n_gg * 2, G)),
             Y2=np.zeros((n_gg * 2, G)), X=np.zeros((n_gg * 2, G), np.float32), X2=np.zeros((n_gg * 2, G), np.float32))
    add_cell_stats(A, cnt, gid)
    D = {"n": A["n"].reshape(n_gg, 2), "L": A["L"].reshape(n_gg, 2)}
    for k in ("C", "Y", "Y2", "X", "X2"): D[k] = A[k].astype(np.float32 if k in ("X", "X2") else np.float64).reshape(n_gg, 2, G)
    return D, cnt, lab, gg


def selftest():
    rng = np.random.default_rng(0)
    # 0. BH NaN-safe
    q = bh([np.nan, 0.01, 0.5]); assert np.all(np.isfinite(q)) and q[0] == 1.0 and q[1] < 0.05
    # 1. BM log2FC math: ps = 0.5 counts of the smaller library (T fixed 1e6 -> ps = 0.5 cpm)
    U, G = 7, 3; block = np.array([0, 0, 0, 0, 0, 1, 1]); half = np.array([0, 1, 0, 1, 0, 1, 0]); Nn = np.full(U, 100.0)
    Tt = np.full(U, 1e6); Sn = np.full((U, G), 100.0); Sp = Sn.copy(); Sp[:, 0] = 200.0; Sp[:5, 1] = 200.0   # gene1: 2x day0 only
    d, v, n1, n0 = deltas_total(Sp, Tt, Nn, Sn, Tt, Nn, 10)
    r = paired(d, v, n1, n0, block, half)
    assert r is not None and abs(r["lfc"][0] - np.log2(200.5 / 100.5)) < 1e-12, r["lfc"]
    assert abs(r["lfc"][1] - 0.5 * np.log2(200.5 / 100.5)) < 1e-12          # equal-day (5:2 units would give 5/7)
    assert r["se"][0] < 1e-9                                                 # identical units -> zero variance
    # 2. classify: split consistency, sign(test), CI of reported lfc inside +-.5 => conf, strict
    # +-.25, no q>.5 branch
    G = 8; ones = np.ones(G)
    res = dict(lfc=np.array([1., 1., 1., 1., .3, 1., .1, .3]), lfcA=np.array([1., -1., 1., 0., .3, 1., .1, .3]),
               lfcB=np.array([1., 1., -1., 1., .3, 1., .1, .3]), dm=np.array([1., 1., 1., 1., .3, 1., .1, .3]),
               se=np.array([.05, .05, .05, .05, .05, .05, .05, .05]), z=8 * ones, p=np.array([1e-12] * 6 + [.9, .9]),
               est=np.ones(G, bool), df=10 * ones, n1=100, n0=100, nu=6)
    res["se"][7] = .3                                                          # gene7: wide CI -> AMBIGUOUS despite p=.9
    det = np.ones(G, bool); selfm = np.zeros(G, bool); selfm[5] = True
    o = classify(res, det, selfm)
    assert list(o["status"]) == [1, 3, 3, 3, 2, 0, 2, 3], o["status"]
    assert o["strict"][6] and not o["strict"][7]                               # gene6 CI [.0,.2] inside +-.25
    res2 = dict(res); res2["dm"] = -res["dm"]; assert classify(res2, det, selfm)["status"][0] == 3   # sign(test) disagrees
    det2 = det.copy(); det2[0] = False; assert classify(res, det2, selfm)["status"][0] == 0
    assert o["e"][0] == 1.0 and o["e"][1] == -1.0 and o["e_legacy"][1] == 0.0   # cross-fit product vs legacy
    # 3. BM full synthetic pipeline through paired()
    (Sp, Tp, Np, Sn, Tn, Nn), blk, hf = _syn_bm(rng, 12, 200, {0: 2.0, 1: 0.4, 2: "split", 3: 2.0, 4: 2.0})
    dd, vv, a1, a0 = deltas_total(Sp, Tp, Np, Sn, Tn, Nn, 10)
    det = np.ones(200, bool); det[4] = False; selfm = np.zeros(200, bool); selfm[3] = True
    st = classify(paired(dd, vv, a1, a0, blk, hf), det, selfm)["status"]
    assert st[0] == 1 and st[1] == 1, st[:5]
    assert st[2] != 1 and st[3] == 0 and st[4] == 0
    assert (st[5:] == 1).sum() <= 1 and (st[5:] == 2).sum() > 100
    # 4. WITHIN_STATE direct standardisation removes a pure composition shift that TOTAL reports as
    # an effect
    Uu, C, Gg = 8, 2, 4; blk = np.repeat([0, 1], 4); hf = np.arange(Uu) % 2
    mu = np.array([[3.0, 3.0, 3.0, 3.0], [30.0, 30.0, 30.0, 30.0]])           # two cell types, 10x different expression
    def mk(frac):
        n = np.stack([np.round(300 * frac), np.round(300 * (1 - frac))], 1).astype(float)   # (U,C) cells per stratum
        return rng.poisson(mu[None] * n[:, :, None]).astype(float), n
    Sp_s, Np_s = mk(np.full(Uu, .8)); Sn_s, Nn_s = mk(np.full(Uu, .5))         # pert: 80% type-0; NT: 50/50; no within-state effect
    Tp_s, Tn_s = Np_s * 1000.0, Nn_s * 1000.0                             # library 1000 UMI/cell, independent of the 4 genes
    pi = ref_composition(Nn_s, blk)
    dS, vS, a1, a0 = deltas_std(Sp_s, Tp_s, Np_s, Sn_s, Tn_s, Nn_s, pi, blk, 5); rS = paired(dS, vS, a1, a0, blk, hf)
    dT, vT, b1, b0 = deltas_total(Sp_s.sum(1), Tp_s.sum(1), Np_s.sum(1), Sn_s.sum(1), Tn_s.sum(1), Nn_s.sum(1), 10); rT = paired(dT, vT, b1, b0, blk, hf)
    assert np.all(np.abs(rS["lfc"]) < 0.15) and np.all(np.abs(rT["lfc"]) > 0.5), (rS["lfc"], rT["lfc"])
    # 5. K562 cell-level estimator on synthetic cells: reweighting to gem-group mix, halves,
    # self/detection, nulls
    G = 120; eff = np.ones(G); eff[0] = 2.0; eff[1] = 0.4
    bmult = np.ones(G); bmult[5] = 4.0                                          # batch effect on gene5 in odd-index gem groups
    npert = [30, 80, 30, 80, 30, 80]                                            # perturbed cells concentrated in the batch-shifted groups
    Dk, cnt, lab, gg = _syn_cells(rng, G, npert, eff, batch_mult=bmult)
    hlf = np.array([0 if (u + 1) % 2 == 1 else 1 for u in range(6)])            # gem groups numbered 1..6: odd -> A
    Pk = {k: v[:, 1] for k, v in Dk.items()}; Nk = {k: v[:, 0] for k, v in Dk.items()}
    rc = contrast_cell(Pk, Nk, hlf); assert rc is not None
    selfm = np.zeros(G, bool); selfm[3] = True; detk = np.ones(G, bool); detk[4] = False
    oc = classify(rc, detk, selfm)
    assert oc["status"][0] == 1 and oc["status"][1] == 1 and oc["status"][3] == 0 and oc["status"][4] == 0, oc["status"][:6]
    naive = np.log2((cnt[lab == "P"][:, 5].sum() / cnt[lab == "P"].sum() + 1e-9) / (cnt[lab == "NT"][:, 5].sum() / cnt[lab == "NT"].sum() + 1e-9))
    assert abs(rc["lfc"][5]) < 0.2 < abs(naive), (rc["lfc"][5], naive)          # matched NT removes the gem-group confound
    assert (oc["status"][6:] == 1).sum() <= 1
    Ph = {k: (v.copy()) for k, v in Pk.items()}; Ph["n"] = Ph["n"].copy(); Ph["n"][hlf == 1] = 3     # < 25 perturbed cells in half B
    assert contrast_cell(Ph, Nk, hlf) is None                                   # halves-size gate makes the pair UNTESTED
    # 6. per-gene denominators, cross-fit burden, M5 labels
    status = np.array([[1, 0, 2], [1, 0, 2], [0, 0, 2], [3, 0, 0]], np.int8); E = np.array([[1., 0, 0], [.6, 0, 0], [0, 0, 0], [0, 0, 0]])
    z = np.zeros_like(E); t = per_gene_table(status, E, z, z, status == 2, cover_min=3)
    assert t.n_testable.tolist() == [3, 0, 3] and t.n_untested.tolist() == [1, 4, 1]
    assert abs(t.b_g[0] - 2 / 3) < 1e-12 and np.isnan(t.b_g[1]) and t.b_g[2] == 0.0
    assert abs(t.burden[0] - np.sqrt((1 + .6) / 3)) < 1e-12 and np.isnan(t.burden[1])
    Eneg = -np.ones((3, 1)); assert per_gene_table(np.full((3, 1), 3, np.int8), Eneg, Eneg, np.ones((3, 1)), np.zeros((3, 1), bool), 1).burden[0] == 0.0   # sqrt(max(.,0))
    def lab_(nr, nc, na, cover=5):
        s = np.array([1] * nr + [2] * nc + [3] * na, np.int8)[:, None]; q = np.zeros(s.shape)
        r = per_gene_table(s, q, q, q, s == 2, cover); return r.label_any_vs_confnon[0], r.label_broad_vs_single[0]
    assert lab_(0, 8, 2)[0] == 0.0 and np.isnan(lab_(0, 7, 3)[0]) and lab_(1, 1, 0)[0] == 1.0     # >=80% conf; ambiguous <=20%
    assert lab_(1, 4, 5)[1] == 0.0 and np.isnan(lab_(1, 3, 6)[1]) and lab_(3, 6, 1)[1] == 1.0     # single needs >=cover-1 conf others
    # 7. N1 pseudo draws: uniform within unit, size = min(matched, floor(pool/2)), reproducible,
    # independent across draws
    pools = {0: np.arange(100), 1: np.arange(100, 130)}; sizes = {0: 10, 1: 50}
    d1 = draw_pseudo(null_rng(0, 0), pools, sizes); d1b = draw_pseudo(null_rng(0, 0), pools, sizes); d2 = draw_pseudo(null_rng(0, 1), pools, sizes)
    assert len(d1[0]) == 10 and len(d1[1]) == 15                                  # capped at floor(30/2)
    assert all(np.array_equal(d1[k], d1b[k]) for k in pools) and not np.array_equal(d1[0], d2[0])
    assert all(set(d1[k].tolist()) <= set(pools[k].tolist()) and len(set(d1[k].tolist())) == len(d1[k]) for k in pools)
    # complement additivity: NT stats minus pseudo stats == stats of the remaining NT cells
    cellsY = rng.poisson(3.0, size=(40, 5)).astype(float); pick = np.sort(rng.choice(40, 12, replace=False)); rest = np.setdiff1d(np.arange(40), pick)
    assert np.allclose(cellsY.sum(0) - cellsY[pick].sum(0), cellsY[rest].sum(0))
    # N2 gate verdicts: K=15 -> primary threshold = 95% quantile of Binomial(15,.05) = 2
    assert int(stats.binom.ppf(0.95, 15, 0.05)) == 2
    v1 = gate_verdict([1, 1] + [0] * 13, [10000] * 15, [8] * 15)                     # 2 comps with a responder; tail 120 vs 150 -> .8
    assert v1["PRIMARY_PASS"] and v1["SECONDARY_PASS"] and v1["label"] == "CALIBRATED" and abs(v1["tail_ratio"] - 0.8) < 1e-12
    v2 = gate_verdict([0] * 15, [10000] * 15, [100] * 15)                            # no responders but tail 1500 vs 150 -> ratio 10
    assert v2["PRIMARY_PASS"] and not v2["SECONDARY_PASS"] and v2["label"] == "UNCALIBRATED"
    v3 = gate_verdict([1, 1, 1] + [0] * 12, [10000] * 15, [8] * 15)                  # 3 comps with a responder > 2
    assert not v3["PRIMARY_PASS"] and v3["SECONDARY_PASS"] and v3["label"] == "UNCALIBRATED"
    # N3: BM p from the unweighted paired t (df = n_valid-1)
    Uu = 7; blk3 = np.array([0, 0, 0, 0, 0, 1, 1]); hf3 = np.array([0, 1, 0, 1, 0, 1, 0])
    dd = np.zeros((Uu, 2)); dd[:5, 0] = -0.3; dd[5:, 0] = 0.5; dd[:, 0] += rng.normal(0, 0.02, Uu); dd[:, 1] = rng.normal(0.4, 0.05, Uu)
    r3 = paired(dd, np.ones(Uu, bool), np.full(Uu, 100.), np.full(Uu, 100.), blk3, hf3)
    assert r3["lfc"][0] > 0 > r3["dm"][0]                                           # equal-day mean > 0 but unweighted mean < 0
    tt = stats.ttest_1samp(dd[:, 1], 0.0); assert abs(r3["p"][1] - tt.pvalue) < 1e-12 and abs(r3["dm"][1] - dd[:, 1].mean()) < 1e-12
    o3 = classify(r3, np.ones(2, bool), np.zeros(2, bool)); assert not o3["cons"][0]   # sign disagreement blocks RESPONDER
    # 8. NT-vs-NT null through the K562 estimator: no responders
    Dn, _, _, _ = _syn_cells(rng, G, [60] * 6, np.ones(G))
    on = classify(contrast_cell({k: v[:, 1] for k, v in Dn.items()}, {k: v[:, 0] for k, v in Dn.items()}, hlf), np.ones(G, bool), np.zeros(G, bool))
    assert (on["status"] == 1).sum() == 0 and (on["status"] == 2).sum() > 80
    assert first_three(["c", "a", "d", "b"]) == ["a", "b", "c"]
    # 9. empirical null
    rt = np.random.default_rng(3); Ut, Gt = 2, 4; ntu_t = np.repeat([0, 1], 12); Ct = rt.poisson(2.0, (24, Gt)).astype(np.float32)
    Lt = Ct.sum(1).astype(np.float64) + 40.0                                       # library includes genes outside the tested set
    cx = nt_context(Ct, Lt, ntu_t, Ut); pl = cx["pools"]
    refs_t = dict(Y=np.stack([cx["YNT"][pl[u]].sum(0) for u in range(Ut)]).astype(float), C=np.stack([Ct[pl[u]].sum(0) for u in range(Ut)]).astype(float),
                  L=np.array([Lt[pl[u]].sum() for u in range(Ut)]))
    sb = draw_subsets(np.random.default_rng(5), pl, np.array([3, 2]), 6); dd_, ll_ = subset_stats(sb, pl, cx, refs_t, None, True)
    assert all(len(set(r.tolist())) == r.shape[0] for ix in sb["idx"] for r in ix)             # subsets have no repeats
    for m in range(6):
        Sy, Sc, Sl, n_, Ny, Nc_, Nl = [], [], [], [], [], [], []
        for j_, u in enumerate(sb["us"]):
            c = pl[u][sb["idx"][j_][m]]; n_.append(len(c)); Sy.append(cx["YNT"][c].mean(0))
            Ny.append((refs_t["Y"][u] - cx["YNT"][c].sum(0)) / (len(pl[u]) - len(c)))
            Sc.append(Ct[c].sum(0)); Sl.append(Lt[c].sum()); Nc_.append(refs_t["C"][u] - Ct[c].sum(0)); Nl.append(refs_t["L"][u] - Lt[c].sum())
        n_ = np.array(n_, float); w = n_ / n_.sum()
        assert np.allclose(dd_[m], sum(w[j_] * (Sy[j_] - Ny[j_]) for j_ in range(len(n_))), atol=1e-9)
        v = np.array(Sl) / sum(Sl); cp = sum(Sc) / sum(Sl) * 1e6; cn = sum(v[j_] * Nc_[j_] / Nl[j_] for j_ in range(len(n_))) * 1e6
        psb = 0.5e6 / min(sum(Sl), sum(Nl)); assert np.allclose(ll_[m], np.log2((cp + psb) / (cn + psb)), atol=1e-9)
    rs_ = np.random.default_rng(0)
    for n_, N_ in ((3, 100), (40, 100)):                                                       # both subset samplers: distinct, in range
        ss = _uniform_subsets(rs_, 500, N_, n_); assert ss.shape == (500, n_) and ss.min() >= 0 and ss.max() < N_ and all(len(set(r)) == n_ for r in ss.tolist())
    # (b) calibration on cluster-structured NT cells (cells dependent within gem group through
    # shared states)
    Uq, Gq, per, ncl = 6, 300, 250, 4; hq = np.array([0, 1, 0, 1, 0, 1]); one, zer = np.ones(Gq, bool), np.zeros(Gq, bool)
    def build_nt(heavy):
        lib_sh, g_sh, mu0 = (0.7, 0.6, 0.15) if heavy else (20.0, 50.0, 2.0)
        base = np.exp(rng.normal(np.log(mu0), 0.9, Gq)); cm = np.exp(rng.normal(0, 0.9, (ncl, Gq))); rows, uu = [], []
        for g in range(Uq):
            cl = rng.integers(0, ncl, per); lib = rng.gamma(lib_sh, 1 / lib_sh, per)[:, None]
            rows.append(rng.poisson(rng.gamma(g_sh, 1 / g_sh, (per, Gq)) * base * cm[cl] * lib * (1 + 0.3 * g))); uu += [g] * per
        N_ = np.vstack(rows).astype(np.float32); N_[:, 5] = 0; return N_, np.array(uu)          # gene 5 never expressed
    def mk_ptest(NTC, uq):
        cq = nt_context(NTC, NTC.sum(1, dtype=np.float64), uq, Uq); poolq = cq["pools"]; full = stats_sub(NTC, uq, Uq, np.arange(Gq))
        def ptest(seed, per_u, genes_=None, mult=1, M1=1000, B=400):
            rk = np.random.default_rng([7, seed]); idx = np.concatenate([rk.choice(poolq[g], per_u, replace=False) for g in range(Uq)])
            cells = NTC[idx].copy()
            if genes_ is not None: cells[:, genes_] *= mult
            Ps = stats_sub(cells, uq[idx], Uq, np.arange(Gq)); Po = stats_sub(NTC[idx], uq[idx], Uq, np.arange(Gq)); comp = {q_: full[q_] - Po[q_] for q_ in Po}
            rq = contrast_cell(Ps, comp, hq); keep = np.ones(len(uq), bool); keep[idx] = False
            return rq, empirical_test(rq, cq, [np.flatnonzero(keep & (uq == g)) for g in range(Uq)], comp, np.random.default_rng([8, seed]), 200, M1, B)
        return cq, ptest
    NTC, uq = build_nt(True); cq, ptest = mk_ptest(NTC, uq)
    pw, pg, pb, bn, ex_m = [], [], [], [], []
    for k in range(25):
        rq, (rm, ra, rl, inf) = ptest(k, 10); om = classify(rm, one, zer)
        assert not rm["est"][5] and not ra["est"][5] and not rl["est"][5]                         # dead gene UNTESTED in every mode
        ok_ = rq["est"] & rm["est"]; pw.append(rq["p"][ok_]); pg.append(rm["p"][ok_]); pb.append(ra["p"][ok_]); bn.append(cq["bins"][ok_])
        ex_m.append((om["lo"][om["T"]] > 0) | (om["hi"][om["T"]] < 0))
    pw, pg, pb, bn, ex_m = map(np.concatenate, (pw, pg, pb, bn, ex_m))
    assert np.mean(pw < .01) > 0.025, np.mean(pw < .01)                                        # Welch/Satterthwaite anti-conservative here
    assert 0.035 < np.mean(pg < .05) < 0.065 and np.mean(pg < .01) < 0.02 and np.mean(pg < .001) < 0.004, (np.mean(pg < .05), np.mean(pg < .01), np.mean(pg < .001))
    assert 0.035 < np.mean(pb < .05) < 0.065 and np.mean(pb < .01) < 0.02, (np.mean(pb < .05), np.mean(pb < .01))
    for b in range(N_BINS): assert 0.015 < np.mean(pb[bn == b] < .05) < 0.10, (b, np.mean(pb[bn == b] < .05))   # calibrated inside each decile
    assert 0.02 < ex_m.mean() < 0.09, ex_m.mean()                                              # se_cal CI: ~5% exclude the true 0
    # CONFIDENT_NONRESPONDER coverage under the null on a well-measured (low-noise) dataset with the
    # same within-gem-group cluster structure
    N2, u2 = build_nt(False); _, ptest2 = mk_ptest(N2, u2); cf, ex2 = [], []
    for k in range(6):
        rq, (rm, ra, rl, inf) = ptest2(300 + k, 50, M1=200); om = classify(rm, one, zer); cn_ = om["status"] == 2
        cf.append((om["lo"][cn_] <= 0) & (om["hi"][cn_] >= 0)); ex2.append((om["lo"][om["T"]] > 0) | (om["hi"][om["T"]] < 0))
    cf, ex2 = np.concatenate(cf), np.concatenate(ex2)
    assert len(cf) > 200 and cf.mean() > 0.93 and 0.02 < ex2.mean() < 0.09, (len(cf), cf.mean(), ex2.mean())
    # (c) adaptive refinement: planted strong effect (3 most expressed genes x15, 40 cells/unit)
    # reaches p<1e-3 beyond the M0 floor
    hi_ = np.argsort(-NTC.mean(0))[:3]
    rq, (rm, ra, rl, inf) = ptest(99, 40, hi_, 15, M1=3000, B=500)
    assert (rm["p"][hi_] < 1e-3).sum() >= 2 and (inf["Mg"][hi_][rm["p"][hi_] < 1e-3] > 200).all() and inf["n_refined"] >= 2, (rm["p"][hi_], inf["Mg"][hi_])
    assert (ra["p"][hi_] < 1e-3).sum() >= 2
    assert tail_level("gene", 200, 200) == 2 / 201 and tail_level("gene", 200, 10000) == NULL_TAIL_P and tail_level("pooled", 200, 200) == NULL_TAIL_P
    assert pert_seed("GATA1") == pert_seed("GATA1") != pert_seed("GATA2") and null_rng(1, 2).integers(1 << 30) == null_rng(1, 2).integers(1 << 30)
    # 10. sharding: serial vs 3-shard + merge must give identical files
    import copy, tempfile
    U4, G4, npert = 4, 80, 7; names = [f"G{j}" for j in range(npert)]; genes4 = np.array([f"G{j}" for j in range(G4)]); ids4 = np.array([f"ENSG{j:05d}" for j in range(G4)])
    n_nt4, per4 = 600, 90; lab4 = np.array([K562_NT] * n_nt4 + [p_ for p_ in names for _ in range(per4)]); u4 = np.tile(np.arange(U4), len(lab4))[:len(lab4)]
    u4 = np.random.default_rng(5).permutation(u4); base4 = rng.uniform(0.5, 3.0, G4); C4 = np.zeros((len(lab4), G4), np.float32)
    for r_ in range(len(lab4)):
        eff = np.ones(G4); j_ = names.index(lab4[r_]) if lab4[r_] != K562_NT else -1
        if j_ >= 0: eff[5 * j_ + 8:5 * j_ + 12] = 4.0
        C4[r_] = rng.poisson(base4 * eff * rng.gamma(20, 1 / 20) * 3)
    lev4 = [K562_NT] + sorted(names); lid4 = {t: i for i, t in enumerate(lev4)}
    chunks4 = lambda: ((lo, min(lo + 200, len(lab4)), C4[lo:lo + 200]) for lo in range(0, len(lab4), 200))
    A4, NT4 = k562_stream(chunks4(), lab4, u4, lid4, U4, G4)
    arms4 = {"F2_36": names[:4], "LAMBERT_TF": names, "ALL_PERT": names}; nullp = ["G3", "G1", "G2", "G0"]           # not sorted (as K562_F2_36 order)
    coords4 = dict(chr=np.array(["1"] * G4), start=np.arange(G4) * 700.0, end=np.arange(G4) * 700.0 + 500, strand=None)     # promoter-neighbour path
    inp4 = k562_prepare(A4, NT4, lab4, u4, U4, G4, lid4, genes4, ids4, coords4, np.array([0, 1, 0, 1]), arms4, nullp)
    a4 = argparse.Namespace(m_null=30, m_max=90, refine_batch=30)
    rec4 = dict(script_sha256="selftest", args=dict(smalln=False, null_gate=True, m_null=30, m_max=90, refine_batch=30), units=["1", "2", "3", "4"],
                thresholds=thresholds(), count_source="X", arms={k: dict(n=len(v), perts=v) for k, v in arms4.items()}, phase="PREDECLARED",
                null_gate=dict(on=True, matched_to=nullp, M=NULL_M, seed=NULL_SEED))
    tmp = tempfile.mkdtemp(); ser, shd = f"{tmp}/serial", f"{tmp}/shard"; os.makedirs(ser); os.makedirs(shd)
    k562_run(a4, ser, copy.deepcopy(rec4), inp4, None, time.time(), {})
    for i_ in range(3): k562_run(a4, shd, copy.deepcopy(rec4), inp4, (i_, 3), time.time(), {})
    k562_merge(shd, 3)
    for fn in ("per_pair.tsv.gz", "per_gene.tsv", "perturbations.tsv", "null_gate.tsv"):
        assert open(f"{ser}/{fn}", "rb").read() == open(f"{shd}/{fn}", "rb").read(), f"serial != merged for {fn}"
    rs_, rm_ = json.load(open(f"{ser}/receipt.json")), json.load(open(f"{shd}/receipt.json"))
    for key in ("null_gate_result", "output_sha256", "n_cells", "n_nt_cells", "n_genes_testable_floor", "arms", "args"): assert rs_[key] == rm_[key], key
    assert rm_["sharding"]["n"] == 3
    with gzip.open(f"{ser}/per_pair.tsv.gz", "rt") as fh: n_pairs = sum(1 for _ in fh)
    assert n_pairs > 100 and len(pd.read_csv(f"{ser}/null_gate.tsv", sep="\t")) == 3 * NULL_M * len(nullp)              # non-trivial, full gate
    assert set(pd.read_csv(f"{ser}/per_gene.tsv", sep="\t").estimand) >= {E_MAIN, E_LEG, E_ALT, E_MAIN + "_EXCL_PROMOTER_NEIGHBOR"}
    assert "p_ALT" in pd.read_csv(f"{ser}/per_pair.tsv.gz", sep="\t", nrows=2).columns and "n_null_draws" in pd.read_csv(f"{ser}/per_pair.tsv.gz", sep="\t", nrows=2).columns
    def refuses(mut):
        d = f"{tmp}/bad"; shutil.rmtree(d, ignore_errors=True); shutil.copytree(shd, d)
        for fn in ("per_pair.tsv.gz", "per_gene.tsv", "perturbations.tsv", "null_gate.tsv", "receipt.json"): os.remove(f"{d}/{fn}")
        mut(d)
        try: k562_merge(d, 3)
        except SystemExit: return True
        return False
    def edit(d, i_, fn):
        f_ = f"{d}/parts/shard_{i_}_of_3.receipt.json"; r_ = json.load(open(f_)); fn(r_); write_json(f_, r_)
    assert refuses(lambda d: os.remove(f"{d}/parts/shard_1_of_3.npz"))                                                     # missing shard part
    assert refuses(lambda d: edit(d, 2, lambda r_: r_.update(script_sha256="other")))                                     # different script
    assert refuses(lambda d: edit(d, 1, lambda r_: r_["args"].update(m_max=91)))                                          # different args
    assert refuses(lambda d: edit(d, 0, lambda r_: r_["shard"]["perts_processed"].append("G6")))                          # overlapping coverage
    assert refuses(lambda d: edit(d, 0, lambda r_: r_["shard"]["perts_processed"].pop()))                                 # gap in coverage
    shutil.rmtree(tmp, ignore_errors=True)
    print("SELFTEST PASS")


def main():
    if "--selftest" in sys.argv: return selftest()
    ap = argparse.ArgumentParser(); sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("bm", "k562"):
        s = sp.add_parser(name); s.add_argument("--smalln", action="store_true"); s.add_argument("--out", default="")
        s.add_argument("--null-gate", dest="null_gate", action="store_true"); s.add_argument("--force", action="store_true")
        s.add_argument("--skip_hash", action="store_true")
        if name == "bm":
            s.add_argument("--min_unit_cells", type=int, default=10); s.add_argument("--min_unit_cells_strat", type=int, default=5)
        else:
            s.add_argument("--chunk", type=int, default=5000); s.add_argument("--m_null", type=int, default=NULL_EMP_M)
            s.add_argument("--m_max", type=int, default=NULL_EMP_MAX); s.add_argument("--refine_batch", type=int, default=REFINE_BATCH)
            s.add_argument("--shard", default="", help="i/n: process perturbations at sorted-position %% n == i; parts go to <out>/parts")
    s = sp.add_parser("inspect"); s.add_argument("--which", choices=["bm", "k562"], required=True)
    s = sp.add_parser("k562-merge"); s.add_argument("--out", required=True); s.add_argument("--n", type=int, required=True)
    a = ap.parse_args()
    if a.cmd == "k562-merge": return k562_merge(a.out, a.n)
    {"bm": cmd_bm, "k562": cmd_k562, "inspect": cmd_inspect}[a.cmd](a)


if __name__ == "__main__":
    main()
