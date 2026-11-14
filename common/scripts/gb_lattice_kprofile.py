#!/usr/bin/env python
"""Extended Data: score and evaluation profiles over conditioning depth."""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
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
import gb_layer_lattice as L  # noqa: E402
import gb_primary as GP  # noqa: E402

LABEL = "DESCRIPTIVE_DEPTH_PROFILE"
SPEC_LOG = f"{L.R}/internal_review_log_not_shipped_a.log"
OUT_DIR = f"{L.R}/results/real_uniform/lattice_kprofile"
KALL = tuple(range(31))                                   # catbank raw |S| at k = 0..30 (kmax = 30 is required for a production shard)
KP = tuple(GP.KS)                                         # 0..8: the PRIMARY window = the common-mask window
GB_VARIANTS = {"pairU": ("PU", False), "pairM": ("PM", False), "cellU": ("CU", True), "cellM": ("CM", True)}   # name: (catbank suffix, cell space)
ROOT_SERIES = [f"{v}@{h}" for v in ("pairU", "pairM") for h in GP.ROOTS]      # per-root descriptive series (M4)
INFER_SERIES = list(GB_VARIANTS) + ["pcorr"]              # M2 family = 7 contrasts x these 5
PLAN = {"PANC": [("RFX6", "primary")], "BONEMARROW": [("RUNX1", "primary"), ("GATA1", "descriptive"), ("TAL1", "descriptive")],
        "GATA1": [("GATA1", "primary")], "KLF1": [("KLF1", "primary")]}
CONTRASTS = ["resp", "occ_prim", "AB", "CD", "AC", "BD", "AD"]
CMAP = {"resp": "resp", "occ_prim": "occ_prim", "AB": "AB_prim", "CD": "CD_prim", "AC": "AC_prim", "BD": "BD_prim", "AD": "AD_prim"}   # -> gb_layer_lattice.targets keys
CLABEL = {"resp": "responder vs non", "occ_prim": "bound vs unbound", "AB": "A>B", "CD": "C>D", "AC": "A>C", "BD": "B>D", "AD": "A>D",
          "dial_F": "F_k=mean(A>B,C>D)-.5", "dial_O": "O_k=mean(occ,B>D)-.5"}
METRICS = ("AUROC", "AUPRC")
PT_KEYS = ("AUROC", "AUPRC", "APLIFT", "PREV", "P20", "P50")   # point values stored (P@K NaN unless n >= 2K, the CT rule); draws kept for AUROC/AUPRC
B_FULL, B_SMALL = 1000, 200
PARITY_TOL = 1e-5                                         # = gb_catbank --tol-primary default
DIAG_TOL = 1e-6                                           # = gb_catbank --tol-raw default: recomputed absS vs shard raw
COV_MIN, MASK_LOSS_MAX = 0.95, 0.10                       # fail-closed thresholds
N_MATCH = 20
EQ_BAND, PRES_LO, PRES_AUC = 0.05, -0.02, 0.50            # M2 margins
DK = tuple(range(9)) + (30,)                              # depths at which diagnostics are summarised
SUMMARY_STATS = ("D_c", "delta_k8_minus_k0", "spearman_k0_8", "primary_minus_k0")
TIER = {"D_c": "PRIMARY", "delta_k8_minus_k0": "SECONDARY", "spearman_k0_8": "SUPPLEMENTARY", "primary_minus_k0": "SUPPLEMENTARY"}
PARTS = ("profile", "summary", "inference", "outcomes", "null", "diagnostics", "exclusions")


# score construction (no truth argument anywhere)
def _scatter(v, uni):
    out = np.full(len(uni), np.nan)
    out[uni] = v
    return out


# Raw {variant: {root: (G, K+1) |S|}}, uni (G,) bool U_obs -> {(series, k): (G,) z, (variant, 'PRIMARY'): (G,) z}, NaN outside U_obs
def depth_scores(raw, uni, ks=KALL, per_root=True):
    out = {}
    for v, per in raw.items():
        Ru = {h: np.asarray(per[h])[uni] for h in GP.ROOTS}
        for k in ks:
            out[v, k] = _scatter(GP.zcons(Ru, ks=(k,)), uni)
            if per_root and v in ("pairU", "pairM"):
                for h in GP.ROOTS:
                    out[f"{v}@{h}", k] = _scatter(GP.zcol(Ru[h][:, k]), uni)
        out[v, "PRIMARY"] = _scatter(GP.zcons(Ru), uni)
    return out


# Cell-level partial correlations after removing the first k columns of Qc.
def pcorr_profile(x, Y, Qc, chunk=2000):
    x = np.asarray(x, float)
    xc = x - x.mean()
    K = Qc.shape[1]
    a = Qc.T @ xc
    xx = float(xc @ xc)
    cdx = np.concatenate([[xx], xx - np.cumsum(a ** 2)])[:, None]
    out = np.full((Y.shape[1], K + 1), np.nan)
    for g0 in range(0, Y.shape[1], chunk):
        Yc = np.asarray(Y[:, g0:g0 + chunk], np.float64)
        Yc = Yc - Yc.mean(0)
        Bm = Qc.T @ Yc
        yy, xy = (Yc ** 2).sum(0), xc @ Yc
        num = np.vstack([xy[None], xy[None] - np.cumsum(a[:, None] * Bm, 0)])
        cdy = np.vstack([yy[None], yy[None] - np.cumsum(Bm ** 2, 0)])
        ok = (cdx > 1e-10 * xx) & (cdy > 1e-10 * yy[None])
        with np.errstate(invalid="ignore", divide="ignore"):
            r = np.abs(num) / np.sqrt(np.where(ok, cdx * cdy, np.nan))
        out[g0:g0 + chunk] = np.where(ok, r, np.nan).T
    return out


def score_digest(scores):
    h = hashlib.sha256()
    for k in sorted(scores, key=str):
        h.update(str(k).encode())
        h.update(np.ascontiguousarray(np.nan_to_num(np.asarray(scores[k], float), nan=-999.0)).tobytes())
    return h.hexdigest()


# m2: PRIMARY of every GB variant rebuilt from the raw |S| must equal the shard's stored P|* (full axis; same finiteness; <= PARITY_TOL)
def check_primary_parity(scores, P):
    worst = {}
    for v in GB_VARIANTS:
        a, b = scores[v, "PRIMARY"], np.asarray(P[v], float)
        fa, fb = np.isfinite(a), np.isfinite(b)
        d = float(np.abs(a[fa & fb] - b[fa & fb]).max()) if (fa & fb).any() else np.nan
        if not (np.array_equal(fa, fb) and d <= PARITY_TOL):
            raise L.Blocked(f"SHARD_PRIMARY_PARITY_FAILED {v}: finiteness equal={np.array_equal(fa, fb)}, max|diff|={d}")
        worst[v] = d
    return worst


# shard loading (fail closed)
# Validated catbank: bank sha self-consistent
def open_bank(axis_sys, catbank=None):
    catbank = catbank or L.CATBANK
    bp = f"{catbank}/{axis_sys}_bank.json"
    if not os.path.exists(bp):
        raise L.Blocked(f"NO_BANK {bp}")
    bank = json.load(open(bp))
    if bank["sha256_members"] != L.members_sha(bank["members"]):
        raise L.Blocked(f"BANK_INCONSISTENT {axis_sys}")
    rx = re.compile(rf"{re.escape(axis_sys)}_scores_\d+-\d+\.npz")                       # == gb_catbank.shard_pattern; never the _smalln shard
    files = sorted(f for f in glob.glob(f"{catbank}/{axis_sys}_scores_*-*.npz") if rx.fullmatch(os.path.basename(f)))
    if not files:
        raise L.Blocked(f"NO_SCORE_SHARDS {axis_sys}")
    seen, where = np.zeros(len(bank["members"]), int), {}
    for f in files:
        rp = Path(f).with_suffix(".receipt.json")
        if not rp.exists():
            raise L.Blocked(f"NO_RECEIPT {rp}")
        rec = json.load(open(rp))
        if rec["binding"]["config"]["bank_sha256"] != bank["sha256_members"] or rec.get("production_universe") is not True or rec.get("parity_verified") is not True:
            raise L.Blocked(f"SCORES_SHARD_NOT_BOUND_TO_BANK_OR_UNVERIFIED {f}")
        with np.load(f, allow_pickle=True) as z:
            seen[np.asarray(z["bank_idx"], int)] += 1
            for i, s in enumerate(z["sources"]):
                where[str(s).upper()] = (f, i)
    if not (seen == 1).all():
        raise L.Blocked(f"SHARD_COVERAGE {int((seen != 1).sum())} of {len(seen)} bank rows not covered exactly once in {axis_sys}")
    return bank, where


# Raw |S| (k 0..30), U_obs universe and stored PRIMARY P|* of each requested source
def load_raw_many(axis_sys, focals, catbank=None):
    bank, where = open_bank(axis_sys, catbank)
    miss = [f for f in focals if f.upper() not in where]
    if miss:
        raise L.Blocked(f"FOCAL_NOT_IN_BANK {miss[:3]} ({axis_sys})")
    out, byfile = {}, {}
    for s in focals:
        f, i = where[s.upper()]
        byfile.setdefault(f, []).append((s.upper(), i))
    genes = None
    for f, items in byfile.items():
        L.receipt_of(f)
        with np.load(f, allow_pickle=True) as z:
            g = z["genes"].astype(str)
            if genes is not None and not np.array_equal(genes, g):
                raise L.Blocked(f"AXIS_DIFFERS_BETWEEN_SHARDS {f}")
            genes = g
            for s, i in items:
                out[s] = dict(raw={}, P={}, uni=np.asarray(z["universe"][i], bool), genes=g, shard=None, focal=s)
            for v, (suf, cell) in GB_VARIANTS.items():
                for s, i in items:
                    out[s]["P"][v] = np.asarray(z[f"P|{suf}"][i], float)
                    out[s]["raw"][v] = {}
                for key, hs in (((f"raw|cell|{suf}", GP.ROOTS),) if cell else tuple((f"raw|{h}|{suf}", (h,)) for h in GP.ROOTS)):
                    A = z[key]                                                       # one key at a time: (ns, G, 31)
                    if A.shape[1:] != (len(g), len(KALL)):
                        raise L.Blocked(f"SHARD_SHAPE {f}: {key} not (G, 31)")
                    for s, i in items:
                        for h in hs:                                                 # cell variants: the same values at both roots
                            out[s]["raw"][v][h] = np.asarray(A[i], np.float32).copy()
                    del A
        for s, i in items:
            out[s]["shard"] = L.frec(f)
    for s in out:
        out[s]["bank_sha256"] = bank["sha256_members"]
    return bank, out


_AX = {}


# cells x axis Ms/Mu, the centred-PC basis and the pair/fixture inputs the catbank scores used (catbank.py:320-346)
def axis_ctx(axis_sys):
    if axis_sys not in _AX:
        import gb_real_uniform as RU
        D = RU.canonical_inputs(axis_sys)
        A, cols, S = D["A"], D["axis_cols"], D["S"]
        P30 = np.asarray(A.obsm["X_pca"], float)[:, :30]
        Qc, _ = np.linalg.qr(P30 - P30.mean(0))
        Ms, Mu = RU.dense(A.layers["Ms"][:, cols], np.float32), RU.dense(A.layers["Mu"][:, cols], np.float32)
        _AX[axis_sys] = dict(axis=np.asarray(D["axis"]).astype(str), Ms=Ms, Mu=Mu, Qc=Qc, P30=P30, X=np.asarray(S["X"], np.float64),
                             fi={g: k for k, g in enumerate(D["fixture_genes"])}, pairs={h: tuple(np.asarray(x, int) for x in S["pairs"][h]) for h in GP.ROOTS},
                             lms=np.log1p(Ms.astype(np.float64).mean(0)), lmu=np.log1p(Mu.astype(np.float64).mean(0)))
    return _AX[axis_sys]


# M3 diagnostics
def _default_project():
    import gb_target_extension as TE
    return TE.project_scores


# dx (P, 1) float32 source differences (pairs) / centred source (cells)
def diag_pass(project, dx, getter, Q, G, kmax=30, chunk=500):
    Qk = np.ascontiguousarray(Q[:, :kmax], np.float64)
    x64 = np.asarray(dx, np.float64)[:, 0]
    Cx = Qk.T @ x64
    xx = float(x64 @ x64)
    rxf = np.concatenate([[1.0], 1.0 - np.cumsum(Cx ** 2) / xx])
    ch = {c: {n: np.full((kmax + 1, G), np.nan) for n in ("num", "den", "z2", "absS", "ryf")} | {"c": np.full((kmax, G), np.nan), "rxf": rxf} for c in "UM"}
    for g0 in range(0, G, chunk):
        sl = slice(g0, min(g0 + chunk, G))
        yU, yM = getter(sl)
        n = yU.shape[1]
        y = np.concatenate([yU, yM], axis=1).astype(np.float32)
        Cy = Qk.T @ y.astype(np.float64)
        yy = (y.astype(np.float64) ** 2).sum(0)
        with np.errstate(invalid="ignore", divide="ignore"):
            ryf = np.vstack([np.ones(y.shape[1]), 1.0 - np.cumsum(Cy ** 2, 0) / yy])
        c = Cx[:, None] * Cy
        for k, num, den, z2, absS in project(dx, y, Q, kmax):
            for name, a in (("num", num), ("den", den), ("z2", z2), ("absS", absS)):
                ch["U"][name][k, sl], ch["M"][name][k, sl] = np.asarray(a)[0, :n], np.asarray(a)[0, n:]
        for name, a in (("ryf", ryf), ("c", c)):
            ch["U"][name][:, sl], ch["M"][name][:, sl] = a[:, :n], a[:, n:]
    return ch


# {level: {'U': .., 'M': ..}} for level in the two roots and 'cell'; the recomputation follows gb_catbank.cmd_scores line by line
def diagnostics(ctx, src_name, src_ax, project=None, kmax=30, chunk=500):
    project = project or _default_project()
    Ms, Mu = ctx["Ms"], ctx["Mu"]
    G = Ms.shape[1]
    xs = np.asarray(ctx["X"][:, ctx["fi"][src_name]], np.float64)
    out = {}
    for h in GP.ROOTS:
        I, J = ctx["pairs"][h]
        Qf, _ = np.linalg.qr(ctx["P30"][J] - ctx["P30"][I])
        dx = (xs[J] - xs[I]).astype(np.float32)[:, None]
        out[h] = diag_pass(project, dx, lambda sl: (Mu[J, sl] - Mu[I, sl], Ms[J, sl] - Ms[I, sl]), Qf, G, kmax, chunk)
    mu_c, ms_c = Mu.astype(np.float64).mean(0), Ms.astype(np.float64).mean(0)
    dxc = (xs - xs.mean()).astype(np.float32)[:, None]
    out["cell"] = diag_pass(project, dxc, lambda sl: ((Mu[:, sl] - mu_c[sl]).astype(np.float32), (Ms[:, sl] - ms_c[sl]).astype(np.float32)), ctx["Qc"], G, kmax, chunk)
    for lv in out.values():                                                                # the self target never scores (catbank.py:358-361)
        for c in lv.values():
            for n in ("num", "den", "z2", "absS", "c", "ryf"):
                c[n][:, src_ax] = np.nan
    return out


# fail closed: the recomputed absS of every (root|cell) x (U|M) must equal the catbank shard raw |S| at k = 0..30 (same finiteness, <= tol)
def check_diag_parity(diag, raw, tol=DIAG_TOL):
    worst = {}
    for lv in (*GP.ROOTS, "cell"):
        for ch, v in ((("U", "pairU"), ("M", "pairM")) if lv != "cell" else (("U", "cellU"), ("M", "cellM"))):
            a = diag[lv][ch]["absS"].T
            b = np.asarray(raw[v][lv if lv != "cell" else GP.ROOTS[0]], float)
            fa, fb = np.isfinite(a), np.isfinite(b)
            d = float(np.abs(a[fa & fb] - b[fa & fb]).max()) if (fa & fb).any() else np.nan
            if not (np.array_equal(fa, fb) and d <= tol):
                raise L.Blocked(f"DIAGNOSTIC_PARITY_FAILED {lv}|{ch}: finiteness equal={np.array_equal(fa, fb)} ({int((fa != fb).sum())} cells), max|diff|={d} > {tol}")
            worst[f"{lv}|{ch}"] = d
    return worst


# point AUROC of `score` for y on rows & finite(score) (gb_canon_table tie-grouped engine)
# NaN if a class is empty
def auroc_pt(score, y, rows):
    idx = np.flatnonzero(rows & np.isfinite(score))
    yy = np.asarray(y, bool)[idx]
    if not (yy.any() and (~yy).any()):
        return np.nan
    o, starts = CT.prep_scores(np.asarray(score, float)[idx])
    return float(CT.weighted_metrics(np.ones((1, len(idx))), yy[o], starts, n_rows=len(idx))["AUROC"][0])


def class_masks(ep, common):
    r, o = ep["resp"], ep["occ"]["prim"]
    return {"A": common & o & r, "B": common & o & ~r, "C": common & ~o & r, "D": common & ~o & ~r}


# {(level|channel, quantity): (K+1, G) array on the axis}; k = 0 rows of c_over_D0 are NaN (defined for removed axes l >= 1)
def diag_quantities(diag, disagree):
    Q = {}
    for lv, chs in diag.items():
        for ch, d in chs.items():
            key = f"{lv}|{ch}"
            with np.errstate(invalid="ignore", divide="ignore"):
                D0 = np.where(d["den"][0] > 1e-12, d["den"][0], np.nan)
                Q[key, "absS"] = d["absS"]
                Q[key, "N_over_D0"] = d["num"] / D0
                Q[key, "D_over_D0"] = d["den"] / D0
                Q[key, "neff"] = np.where(d["z2"] > 0, d["den"] ** 2 / np.where(d["z2"] > 0, d["z2"], np.nan), np.nan)
                Q[key, "c_over_D0"] = np.vstack([np.full((1, d["c"].shape[1]), np.nan), d["c"] / D0])
                Q[key, "target_resid_frac"] = d["ryf"]
    Q.update(disagree)
    return Q


# class A/B/C/D medians/means and per-contrast AUROC of every diagnostic quantity (diagnostics_SYSTEM.tsv); everything on the common rows
def diag_summary(diag, disagree, sel, ep, common, meta):
    Q = diag_quantities(diag, disagree)
    onep = lambda A: np.where(sel[None, :] >= 0, A[:, np.maximum(sel, 0)], np.nan)      # noqa: E731
    cm = class_masks(ep, common)
    T = targets_of(ep, common)
    rows = []
    for (key, qn), A in Q.items():
        A = onep(A)
        for k in DK:
            if k >= A.shape[0] or (qn == "c_over_D0" and k == 0):
                continue
            v = A[k]
            for cl, m in cm.items():
                x = v[m & np.isfinite(v)]
                rows.append(dict(meta, level="class", group=cl, key=key, quantity=qn, k=k, n=len(x), median=float(np.median(x)) if len(x) else np.nan,
                                 mean=float(x.mean()) if len(x) else np.nan, auroc=np.nan, label=LABEL))
            for c, (y, rws) in T.items():
                rows.append(dict(meta, level="contrast", group=c, key=key, quantity=qn, k=k, n=int((rws & np.isfinite(v)).sum()), median=np.nan, mean=np.nan,
                                 auroc=auroc_pt(v, y, rws), label=LABEL))
    for lv, chs in diag.items():
        for k in DK:
            rows.append(dict(meta, level="source", group="SOURCE", key=f"{lv}", quantity="source_resid_frac", k=k, n=1, median=float(chs["U"]["rxf"][k]),
                             mean=float(chs["U"]["rxf"][k]), auroc=np.nan, label=LABEL))
    return pd.DataFrame(rows)


# evaluation (shared bootstrap, paired draws)
def targets_of(ep, mask):
    t = L.targets(ep, mask)
    return {c: t[k] for c, k in CMAP.items()}


# V {(series, depth): (G,) score on ep genes}; store[(series, depth, contrast)] = dict(status, n, n_pos, n_neg, n_lost, gate, pt, bs)
def evaluate(ep, V, common, B, supported):
    counts = ep["blk"].counts[:B]
    T = targets_of(ep, common)
    store = {}
    for key, v in V.items():
        for c, (y, rows) in T.items():
            if not supported[key]:
                store[key + (c,)] = dict(status="UNSUPPORTED_DEPTH")
                continue
            rws = rows & np.isfinite(v)                                                  # == rows on the common window; deep depths / |Pearson| carry their own coverage
            yy = np.asarray(y, bool)[rws]
            e = dict(n=int(rws.sum()), n_pos=int(yy.sum()), n_neg=int((~yy).sum()), n_lost=int(rows.sum() - rws.sum()))
            if not (yy.any() and (~yy).any()):
                store[key + (c,)] = dict(e, status="TOO_FEW")
                continue
            pt, bs = CT.eval_rows(v, y, rws, ep["gi"], counts)
            gate = L.gate_of(ep, y, rws, common)
            store[key + (c,)] = dict(e, gate=gate, status="OK" if gate else "DESCRIPTIVE_ONLY", pt={m: pt[m] for m in PT_KEYS},
                                     bs={m: np.asarray(bs[m], float) for m in METRICS})
    add_dial(store)
    return store


def _ok(e):
    return e is not None and e.get("bs") is not None and np.isfinite(e["pt"]["AUROC"])


# SUPPLEMENTARY layer dial: F_k = mean(A>B, C>D AUROC) - .5, O_k = mean(occ, B>D AUROC) - .5; draws are the mean of the paired component draws
def add_dial(store):
    for series, depth in sorted({(v, d) for v, d, _ in store}, key=str):
        for name, comps in (("dial_F", ("AB", "CD")), ("dial_O", ("occ_prim", "BD"))):
            es = [store.get((series, depth, c)) for c in comps]
            if not all(_ok(e) for e in es):
                continue
            store[series, depth, name] = dict(status="OK" if all(e["gate"] for e in es) else "DESCRIPTIVE_ONLY", gate=all(e["gate"] for e in es),
                                              n=min(e["n"] for e in es), n_pos=min(e["n_pos"] for e in es), n_neg=min(e["n_neg"] for e in es), n_lost=max(e["n_lost"] for e in es),
                                              pt={"AUROC": float(np.mean([e["pt"]["AUROC"] for e in es]) - .5), "AUPRC": np.nan},
                                              bs={"AUROC": np.mean([e["bs"]["AUROC"] for e in es], 0) - .5, "AUPRC": None})


def profile_frame(store, meta):
    rows = []
    for (series, depth, contrast), e in store.items():
        r = dict(meta, variant=series, depth=str(depth), k=depth if isinstance(depth, int) else np.nan, window="common_0_8" if (depth in KP or depth == "PRIMARY") else
                 ("deep_9_30" if isinstance(depth, int) else "NA"), contrast=contrast, contrast_label=CLABEL[contrast], label=LABEL, status=e["status"],
                 n=e.get("n"), n_pos=e.get("n_pos"), n_neg=e.get("n_neg"), n_lost_vs_common=e.get("n_lost"), gate=e.get("gate"))
        if e.get("bs") is not None:
            for m in METRICS:
                if e["bs"][m] is not None:
                    lo, hi, nb = L.pct(e["bs"][m])
                    r.update({m: e["pt"][m], f"{m}_lo": lo, f"{m}_hi": hi, f"nboot_{m}": nb})
            r.update(prevalence=e["pt"].get("PREV"), AP_lift=e["pt"].get("APLIFT"), P20=e["pt"].get("P20"), P50=e["pt"].get("P50"))
        rows.append(r)
    return pd.DataFrame(rows)


def _get(store, series, depth, contrast, metric):
    e = store.get((series, depth, contrast))
    if e is None or e.get("bs") is None or e["bs"].get(metric) is None or not np.isfinite(e["pt"][metric]):
        return None
    return e["pt"][metric], e["bs"][metric]


# M1: D_c = mean_{k=1..8} AUROC(s_k) - AUROC(s_0); get(k) -> (pt, draws) or None
def dc_parts(get):
    es = [get(k) for k in KP]
    if any(e is None for e in es):
        return None
    return float(np.mean([e[0] for e in es[1:]]) - es[0][0]), np.mean([e[1] for e in es[1:]], 0) - es[0][1]


def d8_parts(get):
    a, b = get(8), get(0)
    return None if a is None or b is None else (float(a[0] - b[0]), a[1] - b[1])


def _res(x, status_na="NA_UNDEFINED_DEPTH"):
    if x is None:
        return dict(est=np.nan, lo=np.nan, hi=np.nan, n_draws=0, status=status_na)
    lo, hi, nb = L.pct(x[1])                                                             # paired: the same draws index the same resample
    return dict(est=x[0], lo=lo, hi=hi, n_draws=nb, status="OK")


def _diff(a, b):
    return _res(None if a is None or b is None else (float(a[0] - b[0]), a[1] - b[1]))


# Spearman(x, Y[:, j]) for every column j (average ranks)
# NaN for a constant column or a column with a NaN
def spearman_cols(Y, x):
    from scipy.stats import rankdata
    Y = np.asarray(Y, float)
    out = np.full(Y.shape[1], np.nan)
    ok = np.isfinite(Y).all(0)
    if not ok.any():
        return out
    rx = rankdata(x)
    rx = rx - rx.mean()
    R = rankdata(Y[:, ok], axis=0)
    R = R - R.mean(0)
    den = np.sqrt((R ** 2).sum(0) * (rx ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        out[ok] = np.where(den > 0, (rx @ R) / den, np.nan)
    return out


def _trend(es, ks):
    if any(e is None for e in es):
        return _res(None)
    x = np.asarray(ks, float)
    est = spearman_cols(np.array([[e[0]] for e in es]), x)[0]
    lo, hi, nb = L.pct(spearman_cols(np.stack([e[1] for e in es]), x))
    return dict(est=float(est), lo=lo, hi=hi, n_draws=nb, status="OK" if np.isfinite(est) else "NA_CONSTANT")


# summary_SYSTEM.tsv: pointwise (paired-draw) statistics for every series x contrast; tiers per summary tier map
def summarize(store, meta, ks=KP):
    rows = []
    for series, contrast in sorted({(v, c) for v, _, c in store}):
        dial = contrast.startswith("dial_")
        for metric in METRICS:
            E = lambda d: _get(store, series, d, contrast, metric)                       # noqa: E731
            if all(E(k) is None for k in ks):
                continue                                                                 # nothing defined (|Pearson| k-invariant row; dial AUPRC)
            base = dict(meta, variant=series, contrast=contrast, contrast_label=CLABEL[contrast], metric=metric, label=LABEL)
            def add(stat, res):
                rows.append(dict(base, stat=stat, tier=("SUPPLEMENTARY" if dial else TIER[stat]) + ("" if metric == "AUROC" else "_AP"), **res))
            if metric == "AUROC" and not dial:
                add("D_c", _res(dc_parts(E)))
            add("delta_k8_minus_k0", _res(d8_parts(E)))
            add("spearman_k0_8", _trend([E(k) for k in ks], ks))
            if series in GB_VARIANTS:
                add("primary_minus_k0", _diff(E("PRIMARY"), E(0)))
    return pd.DataFrame(rows)


# (F_k, O_k) trajectory per (system, tf, variant, depth) from the profile rows (SUPPLEMENTARY)
def dial_wide(P):
    d = P[P.contrast.isin(["dial_F", "dial_O"]) & P.AUROC.notna()] if "AUROC" in P.columns else pd.DataFrame()
    if d.empty:
        return pd.DataFrame()
    key = [c for c in ("system", "tf", "tier", "variant", "depth") if c in d.columns]
    w = d.pivot_table(index=key, columns="contrast", values=["AUROC", "AUROC_lo", "AUROC_hi"], aggfunc="first", dropna=False)
    w.columns = [f"{'F' if c == 'dial_F' else 'O'}{'' if m == 'AUROC' else m[5:]}" for m, c in w.columns]
    w = w.reset_index()
    return w.assign(k=pd.to_numeric(w.depth, errors="coerce"), label=LABEL, tier_of_stat="SUPPLEMENTARY")


# M2: simultaneous inference, classes, outcomes
# items [(pt, draws)] -> (q, lo, hi, frac_complete)
# L.maxT over the family; a series with < 95% complete draws gets NaN bands (as gb_layer_lattice.within_branches)
def _bands(items):
    m = len(items)
    lo, hi, comp = np.full(m, np.nan), np.full(m, np.nan), np.array([float(np.isfinite(b).mean()) for _, b in items])
    good = np.flatnonzero(comp >= 0.95)
    q = np.nan
    if len(good):
        q, se, l, h = L.maxT(np.array([items[i][0] for i in good]), np.stack([items[i][1] for i in good]))
        lo[good], hi[good] = l, h
    return q, lo, hi, comp


# EQUIVALENT: band inside +-.05
def classify(contrast, est, lo, hi, prim_lo, gate=True):
    if not np.isfinite([est, lo, hi]).all():
        return dict(is_equivalent=False, is_preserved=False, is_improved=False, cls="NA")
    if not gate:                                                                         # gate failed (DESCRIPTIVE_ONLY): never a class label
        return dict(is_equivalent=False, is_preserved=False, is_improved=False, cls="DESCRIPTIVE_ONLY")
    eq = bool(lo >= -EQ_BAND and hi <= EQ_BAND)
    pres = bool(lo > PRES_LO and np.isfinite(prim_lo) and prim_lo > PRES_AUC)
    imp = bool(contrast in ("AB", "CD") and lo > 0)
    return dict(is_equivalent=eq, is_preserved=pres, is_improved=imp, cls="IMPROVED" if imp else "PRESERVED" if pres else "EQUIVALENT" if eq else "UNRESOLVED")


# inference_SYSTEM.tsv rows + per-(series, contrast) D_c class table
def infer(store, meta):
    rows, cls = [], {}
    prim = {}
    items = [((s, c), _get(store, s, "PRIMARY", c, "AUROC")) for s in GB_VARIANTS for c in CONTRASTS]
    items = [(k, v) for k, v in items if v is not None]
    if items:
        q, lo, hi, comp = _bands([v for _, v in items])
        for (k, v), l_, h_, cp in zip(items, lo, hi, comp):
            prim[k] = l_
            rows.append(dict(meta, variant=k[0], contrast=k[1], contrast_label=CLABEL[k[1]], stat="PRIMARY_AUROC", est=v[0], lo_point=L.pct(v[1])[0], hi_point=L.pct(v[1])[1],
                             lo_simul=l_, hi_simul=h_, q_maxT=q, family_size=len(items), frac_complete_draws=cp, inference="simultaneous_maxT_per_TF_family", label=LABEL))
    for stat, fn in (("D_c", dc_parts), ("delta_k8_minus_k0", d8_parts)):
        its = []
        for s in INFER_SERIES:
            for c in CONTRASTS:
                r = fn(lambda d: _get(store, s, d, c, "AUROC"))
                if r is not None:
                    its.append(((s, c), r))
        if not its:
            continue
        q, lo, hi, comp = _bands([r for _, r in its])
        for ((s, c), r), l_, h_, cp in zip(its, lo, hi, comp):
            row = dict(meta, variant=s, contrast=c, contrast_label=CLABEL[c], stat=stat, tier=TIER[stat], est=r[0], lo_point=L.pct(r[1])[0], hi_point=L.pct(r[1])[1], lo_simul=l_, hi_simul=h_,
                       q_maxT=q, family_size=len(its), frac_complete_draws=cp, inference="simultaneous_maxT_per_TF_family", label=LABEL,
                       gate=all(bool(store[(s, k, c)].get("gate")) for k in KP))
            if stat == "D_c":
                pl = prim.get((s, c), np.nan)
                row.update(primary_auroc_lo_simul=pl, **classify(c, r[0], l_, h_, pl, row["gate"]))
                cls[(s, c)] = dict(est=r[0], lo=l_, hi=h_, gate=row["gate"], eq_raw=bool(np.isfinite([l_, h_]).all() and l_ >= -EQ_BAND and h_ <= EQ_BAND),
                                   **{k: row[k] for k in ("is_equivalent", "is_preserved", "is_improved", "cls")})
            rows.append(row)
    return pd.DataFrame(rows), cls


# occupancy-down = simultaneous hi < 0 for both occ_prim and B>D
def outcomes(cls, meta):
    rows = []
    for s in GB_VARIANTS:
        g = lambda c: cls.get((s, c))                                                    # noqa: E731
        need = [g(c) for c in ("occ_prim", "BD", "AB", "CD")]
        if any(x is None for x in need):
            rows.append(dict(meta, variant=s, outcome="NOT_EVALUABLE", label=LABEL))
            continue
        oc, bd, ab, cd = need
        if not all(x["gate"] for x in need):
            rows.append(dict(meta, variant=s, outcome="NO_LABEL_DESCRIPTIVE_ONLY", classes="; ".join(f"{c}:{g(c)['cls'] if g(c) else 'NA'}" for c in CONTRASTS), label=LABEL))
            continue
        down = lambda x: bool(x["hi"] < 0)                                               # noqa: E731
        occ_down = down(oc) and down(bd)
        occ_eq_neg = bool(all(x["eq_raw"] and x["est"] < 0 for x in (oc, bd)))           # EQUIVALENT-down: reported, never an outcome driver
        sel = occ_down and ab["is_preserved"] and cd["is_preserved"]
        sel_lit = bool(oc["lo"] < 0 and bd["lo"] < 0 and ab["is_preserved"] and cd["is_preserved"])   # sensitivity only (literal "lo < 0")
        enr = bool(ab["is_improved"] or cd["is_improved"])
        loss = bool(down(ab) and down(cd))
        rem = bool(loss and occ_down)
        names = [n for n, f in (("selective_retention", sel), ("predictive_enrichment", enr), ("signal_removal", rem), ("response_loss", loss and not rem)) if f]
        rows.append(dict(meta, variant=s, selective_retention=sel, predictive_enrichment=enr, response_loss=loss, signal_removal=rem, occupancy_down=occ_down,
                         occupancy_equivalent_neg=occ_eq_neg, SENSITIVITY_selective_retention_literal_lo_lt0=sel_lit, outcome="+".join(names) or "none",
                         classes="; ".join(f"{c}:{g(c)['cls'] if g(c) else 'NA'}" for c in CONTRASTS), label=LABEL))
    return pd.DataFrame(rows)


# M6 matched alternative-source null
# n bank members nearest to `focal` on (spliced detection, log1p mean Ms, log1p mean Mu) (standardised Euclidean over the candidates)
def nearest_sources(members, det, lms, lmu, focal, n=N_MATCH):
    cand = [m for m in members if m.upper() != focal.upper()]
    F = np.array([[det[m], lms[m], lmu[m]] for m in cand], float)
    f = np.array([det[focal], lms[focal], lmu[focal]], float)
    mu, sd = F.mean(0), F.std(0)
    sd[sd == 0] = 1.0
    d = np.sqrt((((F - mu) / sd - (f - mu) / sd) ** 2).sum(1))
    o = np.argsort(d, kind="mergesort")[:n]
    return [cand[i] for i in o]


# point D_c of one series: vecs = 9 score vectors (k = 0..8); rows & finite for all of them
# NaN unless every depth's AUROC is defined
def dc_point(vecs, y, rows):
    r = rows.copy()
    for v in vecs:
        r &= np.isfinite(v)
    a = np.array([auroc_pt(v, y, r) for v in vecs])
    return float(a[1:].mean() - a[0]) if np.isfinite(a).all() else np.nan


def nul_na_status(t):
    return set(t.status.dropna()) if "status" in t.columns else set()


# Report focal percentiles and upper/lower pool-rank p-values.
def null_table(focal_vecs, alt_vecs, T, gates, meta):
    out = []
    for s, vv in focal_vecs.items():
        for c, (y, rows) in T.items():
            f = dc_point(vv, y, rows)
            pool = np.array([dc_point(a[s], y, rows) for a in alt_vecs.values() if s in a], float)
            pool = pool[np.isfinite(pool)]
            r = dict(meta, variant=s, contrast=c, contrast_label=CLABEL[c], stat="D_c", focal=f, pool_n=len(pool), gate=gates.get(c), label=LABEL,
                     note="pointwise, unadjusted across series x contrast", p_label="pool rank (descriptive; min p = 1/(1+pool), 1/21 for a pool of 20)")
            if np.isfinite(f) and len(pool) >= 10:
                r.update(pool_median=float(np.median(pool)), pool_lo=float(np.percentile(pool, 2.5)), pool_hi=float(np.percentile(pool, 97.5)),
                         focal_percentile=float((np.sum(pool < f) + 0.5 * np.sum(pool == f)) / len(pool)), p_ge=float((1 + np.sum(pool >= f)) / (1 + len(pool))),
                         p_le=float((1 + np.sum(pool <= f)) / (1 + len(pool))))
            else:
                r.update(status="NA_POOL_LT10_OR_UNDEFINED")
            out.append(r)
    return pd.DataFrame(out)


# one endpoint
# M5: the k<=8 window (incl PRIMARY) must be fully supported or fully collapsed; deep depths and |Pearson| only need > 0
def _coverage_gate(V, U):
    sup = {}
    for k, v in V.items():
        cov = float(np.isfinite(v)[U].mean()) if U.any() else 0.0
        win = k[1] in KP or k[1] == "PRIMARY"
        if win and 0.0 < cov < COV_MIN and k[0] != "pearson":
            raise L.Blocked(f"SUPPORT_BELOW_MIN {k}: finite on {cov:.3f} of U < {COV_MIN} (fail closed)")
        sup[k] = cov > 0.0
    return sup


# st: per-system state
def run_tf(system, tf, tier, st, B, diag_on=True):
    cfg = L.SYSTEMS[system]
    S = cfg["axis"]
    bank, ent = load_raw_many(S, [tf])
    raw = ent[tf.upper()]
    if st.get("blk") is None:
        seed = int(np.random.SeedSequence([L.BOOT_SEED, L.BLOCKS.index(cfg["block"])]).generate_state(1)[0])
        units, scheme = L.load_modules(S, raw["genes"])
        st.update(blk=L.BlockAxis(raw["genes"], B, seed, units, scheme), seed=seed, axis=raw["genes"], scheme=scheme)
    if not np.array_equal(raw["genes"], st["axis"]):
        raise L.Blocked(f"AXIS_DIFFERS_BETWEEN_SHARDS {system}/{tf}")
    ctx = axis_ctx(S)
    if not np.array_equal(ctx["axis"], raw["genes"]):
        raise L.Blocked(f"SHARD_AXIS_DIFFERS_FROM_FIXTURE_AXIS {system}")
    si = np.flatnonzero(np.char.upper(ctx["axis"]) == tf.upper())
    if len(si) != 1:
        raise L.Blocked(f"SOURCE_NOT_UNIQUE_ON_AXIS {tf}")
    si = int(si[0])
    scores = depth_scores(raw["raw"], raw["uni"])
    par_primary = check_primary_parity(scores, raw["P"])
    x = np.asarray(ctx["Ms"][:, si], float)
    for nm, Y in (("pcorr", ctx["Mu"]), ("pcorrMM", ctx["Ms"])):
        pc = pcorr_profile(x, Y, ctx["Qc"])
        for k in KALL:
            scores[nm, k] = np.where(raw["uni"], pc[:, k], np.nan)
    diag, par_diag, flat = None, None, {}
    if diag_on:
        diag = diagnostics(ctx, ctx["axis"][si], si)
        par_diag = check_diag_parity(diag, raw["raw"])
        flat = {("diag", lv, ch, n): a for lv, cc in diag.items() for ch, d in cc.items() for n, a in d.items()}
    members = bank["members"]
    ax = {g: i for i, g in enumerate(ctx["axis"])}
    det, lms, lmu = bank["detection"], {m: ctx["lms"][ax[m]] for m in members}, {m: ctx["lmu"][ax[m]] for m in members}
    alts = nearest_sources(members, det, lms, lmu, next(m for m in members if m.upper() == tf.upper()))
    _, alt_ent = load_raw_many(S, alts)
    alt_scores = {a: depth_scores(alt_ent[a.upper()]["raw"], alt_ent[a.upper()]["uni"], ks=KP, per_root=False) for a in alts}
    pre = score_digest({**scores, **flat, **{("alt", a) + k: v for a, sc in alt_scores.items() for k, v in sc.items()}})   # truth is opened below this line only
    ep = L.build_endpoint(system, tf, st["blk"], False)
    if score_digest({**scores, **flat, **{("alt", a) + k: v for a, sc in alt_scores.items() for k, v in sc.items()}}) != pre:
        raise SystemExit("REFUSING: score vectors changed between construction and truth join")
    pos = {g: i for i, g in enumerate(raw["genes"])}
    sel = np.asarray([pos.get(g, -1) for g in ep["genes"]])
    onep = lambda v: np.where(sel >= 0, np.asarray(v)[np.maximum(sel, 0)], np.nan)      # noqa: E731
    V = {key: onep(v) for key, v in scores.items()}
    a, b = V["pairU", "PRIMARY"], np.asarray(ep["V"][L.GB], float)
    fa, fb = np.isfinite(a), np.isfinite(b)
    d = float(np.abs(a[fa & fb] - b[fa & fb]).max()) if (fa & fb).any() else np.nan
    if not (np.array_equal(fa, fb) and d <= PARITY_TOL):
        raise L.Blocked(f"PRIMARY_PARITY_FAILED {system}/{tf}: shard pairU PRIMARY vs production GB vector: finiteness equal={np.array_equal(fa, fb)}, max|diff|={d}")
    pearson_parity = None
    if "|Pearson|" in ep["V"]:
        V["pearson", "NA"] = np.asarray(ep["V"]["|Pearson|"], float)
        p0, pv = V["pcorrMM", 0], V["pearson", "NA"]
        f_ = np.isfinite(p0) & np.isfinite(pv)
        pearson_parity = float(np.abs(p0[f_] - pv[f_]).max()) if f_.any() else np.nan     # Record without enforcing.
    U = ep["U"]
    supported = _coverage_gate(V, U)
    common = U.copy()
    for k, v in V.items():
        if supported[k] and k[0] != "pearson" and (k[1] in KP or k[1] == "PRIMARY"):
            common &= np.isfinite(v)
    if not common.any() or common.sum() < (1 - MASK_LOSS_MAX) * U.sum():
        raise L.Blocked(f"COMMON_MASK_LOSS {system}/{tf}: {int(common.sum())} of {int(U.sum())} U rows survive the k=0..8 common mask (fail closed)")
    store = evaluate(ep, V, common, B, supported)
    if score_digest({**scores, **flat, **{("alt", a) + k: v for a, sc in alt_scores.items() for k, v in sc.items()}}) != pre:
        raise SystemExit("REFUSING: score vectors changed during evaluation")
    meta = dict(system=system, tf=tf, tier=tier)
    inf, cls = infer(store, meta)
    T = targets_of(ep, common)
    gates = {c: bool(store.get(("pairU", 0, c), {}).get("gate")) for c in CONTRASTS}
    focal_vecs = {s: [V[s, k] for k in KP] for s in GB_VARIANTS if all((s, k) in V for k in KP)}
    for s, vv in focal_vecs.items():                                                     # the null's D_c function must reproduce the bootstrap store's point D_c
        for c in CONTRASTS:
            r = dc_parts(lambda k: _get(store, s, k, c, "AUROC"))
            f = dc_point(vv, *T[c])
            if r is not None and np.isfinite(f) and abs(r[0] - f) > 1e-9:
                raise SystemExit(f"REFUSING: null-table D_c {f} != store D_c {r[0]} for {s}/{c}")
    alt_vecs = {a: {s: [onep(sc[s, k]) for k in KP] for s in GB_VARIANTS} for a, sc in alt_scores.items()}
    nul = null_table(focal_vecs, alt_vecs, T, gates, meta)
    dg = diag_summary(diag, {**{(f"pair|{v}", "root_disagree_z"): np.vstack([0.5 * np.abs(scores[f"{v}@{GP.ROOTS[0]}", k] - scores[f"{v}@{GP.ROOTS[1]}", k]) for k in KALL]) for v in ("pairU", "pairM")}},
                      sel, ep, common, meta) if diag_on else pd.DataFrame()
    cm = class_masks(ep, common)
    r_, o_ = ep["resp"], ep["occ"]["prim"]
    full = {"A": U & o_ & r_, "B": U & o_ & ~r_, "C": U & ~o_ & r_, "D": U & ~o_ & ~r_}                 # per-class exclusion table (U = testable & mappable, unknown never a class)
    excl = pd.DataFrame([{**meta, "class": c, "n_U": int(full[c].sum()), "n_common": int(cm[c].sum()), "n_excluded_by_common_mask": int(full[c].sum() - cm[c].sum()),
                          "n_unknown_resp_axis": ep["meta"]["n_resp_unknown"], "n_unknown_occ_axis": ep["meta"]["n_occ_unknown"]} for c in "ABCD"])
    lost = {f"{k[0]}|{k[1]}": int(U.sum() - (U & np.isfinite(v)).sum()) for k, v in V.items() if supported[k] and U.sum() > (U & np.isfinite(v)).sum()}
    rc = dict(status="OK", system=system, tf=tf, tier=tier, n_U=int(U.sum()), n_common=int(common.sum()), rows_lost_by_series=lost,
              unsupported_scores=[f"{k[0]}|{k[1]}" for k, s in supported.items() if not s], primary_parity_max_abs_diff=d, shard_primary_parity=par_primary,
              diagnostic_parity_max_abs_diff=par_diag, pearson_k0_parity_max_abs_diff=pearson_parity, scores_sha256=pre, alternatives=alts,
              n_resp=int((ep["resp"] & common).sum()), n_occ=int((ep["occ"]["prim"] & common).sum()), shard=raw["shard"], bank_sha256=raw["bank_sha256"],
              gb_sha256=ep["meta"]["gb_sha256"], truth_sha256=ep["meta"]["truth_sha256"], mapping=ep["meta"].get("manifest", {}).get("mapping"),
              fixture_sha256=bank.get("inputs", {}).get("fixture_sha256"), pearson_available="|Pearson|" in ep["V"], diagnostics=diag_on)
    return dict(profile=profile_frame(store, meta), summary=summarize(store, meta), inference=inf, outcomes=outcomes(cls, meta), null=nul, diagnostics=dg, exclusions=excl), rc


def receipt_common(B, st):
    mods = {n: sys.modules[n] for n in ("gb_primary", "gb_canon_table", "gb_layer_lattice", "gb_target_extension", "gb_catbank", "gb_real_uniform") if n in sys.modules}
    return dict(created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), label=LABEL, script_sha256=L.sha_file(__file__), B=B, boot_seed=st.get("seed"),
                block_scheme=st.get("scheme"), counts_fingerprint=None if st.get("blk") is None else st["blk"].fingerprint(),
                bootstrap="one shared multinomial resample matrix per substrate block; pointwise 2.5/97.5 percentile bands; paired differences use the same draws; "
                "simultaneous bands = gb_layer_lattice.maxT per-TF family", spec_log=dict(path=SPEC_LOG, sha256=L.sha_file(SPEC_LOG) if os.path.exists(SPEC_LOG) else None),
                k_rule="CLOSED (production PRIMARY = mean k 0..8, root-min); this table selects nothing", summary_statistics=list(SUMMARY_STATS), tiers=TIER,
                truth_read_after_scores=True, common_row_mask="U & finite(every supported series/depth over k=0..8 and PRIMARY); k=9..30 on common & finite(own score)",
                thresholds=dict(cov_min=COV_MIN, mask_loss_max=MASK_LOSS_MAX, parity_tol=PARITY_TOL, diag_tol=DIAG_TOL, eq_band=EQ_BAND, pres_lo=PRES_LO, pres_auc=PRES_AUC),
                module_shas={n: L.sha_file(m.__file__) for n, m in mods.items()}, python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__,
                note="Descriptive depth profiles; adjusted utility is evaluated separately.")


def run_system(system, out, B, tfs, diag_on=True):
    st, parts, rcs = {}, {p: [] for p in PARTS}, []
    plan = dict(PLAN[system])
    for tf in L.SYSTEMS[system]["tfs"] or []:
        if tf not in plan and (not tfs or tf in tfs):
            parts["profile"].append(pd.DataFrame([dict(system=system, tf=tf, tier="not_in_plan", variant="*", depth="*", contrast="*", label=LABEL, status="NOT_IN_PLAN")]))
            rcs.append(dict(status="NOT_IN_PLAN", system=system, tf=tf))
    for tf, tier in plan.items():
        if tfs and tf not in tfs:
            continue
        t0 = time.time()
        try:
            res, rc = run_tf(system, tf, tier, st, B, diag_on)
            for p in PARTS:
                parts[p].append(res[p])
            rcs.append(rc)
            print(f"[kprofile] {system}/{tf}: n_common={rc['n_common']}/{rc['n_U']} done {time.time()-t0:.0f}s", flush=True)
        except L.Blocked as e:
            print(f"[kprofile] {system}/{tf}: NOT_RUN {e}", flush=True)
            parts["profile"].append(pd.DataFrame([dict(system=system, tf=tf, tier=tier, variant="*", depth="*", contrast="*", label=LABEL, status=f"NOT_RUN: {e}")]))
            rcs.append(dict(status=f"NOT_RUN: {e}", system=system, tf=tf, tier=tier))
    if not rcs:
        raise SystemExit(f"REFUSING: no TF of {system} selected by --tfs")
    for p in PARTS:
        L.atomic_tsv(pd.concat(parts[p], ignore_index=True) if parts[p] else pd.DataFrame(), f"{out}/{p}_{system}.tsv")
    L.atomic_json(f"{out}/receipt_{system}.json", dict(receipt_common(B, st), system=system, block=L.SYSTEMS[system]["block"], endpoints=rcs))


def merge(out):
    rs = sorted(glob.glob(f"{out}/receipt_*.json"))
    if not rs or not glob.glob(f"{out}/profile_*.tsv"):
        raise SystemExit("REFUSING_MERGE: no per-system parts")
    rec = [json.load(open(f)) for f in rs]
    for k in ("script_sha256", "B", "spec_log"):
        if len({json.dumps(r[k], sort_keys=True) for r in rec}) > 1:
            raise SystemExit(f"REFUSING_MERGE_MIXED_{k.upper()}: {sorted({json.dumps(r[k], sort_keys=True) for r in rec})}")
    mods = {}
    for r in rec:
        for n, h in r["module_shas"].items():
            mods.setdefault(n, set()).add(h)
    if any(len(v) > 1 for v in mods.values()):
        raise SystemExit(f"REFUSING_MERGE_MIXED_MODULE_SHAS: {[n for n, v in mods.items() if len(v) > 1]}")
    byblock = {}
    for r in rec:
        byblock.setdefault(r.get("block"), set()).add((r["counts_fingerprint"], r["boot_seed"], r.get("block_scheme")))
    if any(len(v) > 1 for v in byblock.values()):
        raise SystemExit(f"REFUSING_MERGE_DESYNCHRONISED_DRAWS: {byblock}")             # systems of one block (GATA1, KLF1) must share the same resample matrix

    def rd(f):
        try:
            return pd.read_csv(f, sep="\t")
        except pd.errors.EmptyDataError:                                                 # a system whose every TF was NOT_RUN has empty tables
            return pd.DataFrame()
    for p in PARTS:
        L.atomic_tsv(pd.concat([rd(f) for f in sorted(glob.glob(f"{out}/{p}_*.tsv"))] or [pd.DataFrame()], ignore_index=True), f"{out}/{p}.tsv")
    L.atomic_tsv(dial_wide(rd(f"{out}/profile.tsv")), f"{out}/dial.tsv")
    L.atomic_json(f"{out}/receipt.json", dict(rec[0], system=[r["system"] for r in rec], endpoints=[e for r in rec for e in r["endpoints"]], parts=[os.path.basename(f) for f in rs],
                                              counts_fingerprints={r["system"]: r["counts_fingerprint"] for r in rec}, merged=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    print(f"[kprofile] merged {len(rs)} systems -> {out}")


# selftest (synthetic only)
# exact float64 reference of gb_target_extension.project_scores (same yields); used only where numexpr / the real projector is unavailable
def _project_np(dx, y, Q, kmax):
    Qk = np.asarray(Q[:, :kmax], float)
    rx, ry = np.asarray(dx, float), np.asarray(y, float)
    for k in range(kmax + 1):
        if k:
            q = Qk[:, k - 1:k]
            rx, ry = rx - q @ (q.T @ rx), ry - q @ (q.T @ ry)
        num, den, z2 = rx.T @ ry, np.abs(rx).T @ np.abs(ry), (rx ** 2).T @ (ry ** 2)
        with np.errstate(invalid="ignore", divide="ignore"):
            absS = np.where(den > 1e-12, np.abs(num / np.where(den > 1e-12, den, 1.0)), np.nan)
        yield k, num, den, z2, absS


# score family: 'enrich' occupancy drains and response rises with depth (k 0 -> 8); 'remove' response content fades to noise; 'null' pure noise
def _planted(ep, rng, kind):
    o, r = ep["occ"]["prim"].astype(float), ep["resp"].astype(float)
    n1, n2 = rng.normal(size=len(o)), rng.normal(size=len(o))
    zc = lambda x: (x - x.mean()) / x.std()                                              # noqa: E731
    V = {}
    for k in KALL:
        w = min(k, 8) / 8
        V[k] = {"enrich": (1 - w) * (2.0 * o + n1) + w * (2.0 * r + n2), "remove": (1 - w) * (2.0 * r + n1) + w * n2, "null": rng.normal(size=len(o))}[kind]
    V["PRIMARY"] = np.mean([zc(V[k]) for k in KP], 0)
    return V


def selftest():
    import inspect
    import tempfile
    from scipy.stats import spearmanr
    GP._selftest()
    rng = np.random.default_rng(3)
    # 1. depth_scores: single depth == gb_primary.zcons(ks=(k,))
    G, K = 400, 31
    raw = {h: rng.standard_t(3, size=(G, K)) for h in GP.ROOTS}
    uni = rng.random(G) < 0.8
    raw["dc1_min"][:, 3] = 5.0
    ds = depth_scores({"pairU": raw, "cellU": {h: raw["dc1_min"] for h in GP.ROOTS}}, uni)
    for k in (0, 3, 8, 17, 30):
        ref = GP.zcons({h: raw[h][uni] for h in GP.ROOTS}, ks=(k,))
        assert np.allclose(ds["pairU", k][uni], ref, equal_nan=True) and np.isnan(ds["pairU", k][~uni]).all()
        for h in GP.ROOTS:
            assert np.allclose(ds[f"pairU@{h}", k][uni], GP.zcol(raw[h][uni][:, k]), equal_nan=True)
        assert np.array_equal(np.minimum(ds[f"pairU@{GP.ROOTS[0]}", k], ds[f"pairU@{GP.ROOTS[1]}", k]), ds["pairU", k], equal_nan=True)
    assert np.isnan(ds["pairU", 3]).all() and np.isnan(ds["cellU", 3]).all() and "cellU@dc1_min" not in {k[0] for k in ds}
    per = []
    for h in GP.ROOTS:
        Z = np.stack([GP.zcons({x: raw[h][uni] for x in GP.ROOTS}, ks=(k,)) for k in KP])
        m = np.nanmean(Z, 0)
        m[np.isfinite(Z).sum(0) < 6] = np.nan
        per.append(m)
    assert np.allclose(ds["pairU", "PRIMARY"][uni], np.minimum(*per), equal_nan=True)
    assert np.allclose(ds["pairU", "PRIMARY"][uni], GP.zcons({h: raw[h][uni] for h in GP.ROOTS}), equal_nan=True)
    # 2. pcorr_profile == direct residualisation on intercept + PC1..k
    n, Gp, Kp = 220, 7, 6
    P = rng.normal(size=(n, Kp)) @ rng.normal(size=(Kp, Kp))
    x, Y = rng.normal(size=n) + P[:, 0], rng.normal(size=(n, Gp)) + P[:, :2] @ rng.normal(size=(2, Gp))
    Qc, _ = np.linalg.qr(P - P.mean(0))
    pc = pcorr_profile(x, Y, Qc)
    for k in range(Kp + 1):
        Q, _ = np.linalg.qr(np.column_stack([np.ones(n), P[:, :k]]))
        res = lambda M: M - Q @ (Q.T @ M)                                                # noqa: E731
        rx, ry = res(x[:, None]), res(Y)
        assert np.allclose(pc[:, k], np.abs((rx[:, 0] @ ry) / (np.linalg.norm(rx) * np.linalg.norm(ry, axis=0))), atol=1e-9)
    assert np.allclose(pc[:, 0], np.abs([np.corrcoef(x, Y[:, j])[0, 1] for j in range(Gp)]), atol=1e-9)         # k = 0 is |Pearson|
    # 3. spearman_cols == scipy
    Yr = rng.normal(size=(9, 50))
    Yr[:, 4] = np.nan
    sp = spearman_cols(Yr, np.arange(9))
    assert np.isnan(sp[4]) and all(abs(sp[j] - spearmanr(np.arange(9), Yr[:, j])[0]) < 1e-12 for j in range(50) if j != 4)
    # 4. planted series through evaluate -> summarize / infer / outcomes
    blk = L._syn_block(G=1500, B=200, seed=5)
    ep = L._syn_ep(blk, "T", rng=np.random.default_rng(1), resp_rate=.3, occ_rate=.35)
    assert ep["blk"].counts is L._syn_ep(blk, "T2", rng=np.random.default_rng(2))["blk"].counts
    V = {}
    for name, kind in (("pairU", "enrich"), ("cellU", "remove"), ("cellM", "null"), ("pairM", "null")):
        for d, v in _planted(ep, np.random.default_rng({"pairU": 11, "cellU": 12, "cellM": 13, "pairM": 14}[name]), kind).items():
            V[name, d] = v
    for k in (0, 8):
        V["dup", k] = V["pairU", k].copy()
    V["pairM", 3] = np.full(1500, np.nan)                                                # a collapsed depth (D_c NA for this series)
    V["pairU", 20] = np.where(np.arange(1500) < 30, np.nan, V["pairU", 20])            # a deep depth with partial coverage: own coverage, not in the common mask
    supported = _coverage_gate(V, ep["U"])
    assert supported["pairU", 5] and not supported["pairM", 3] and supported["pairU", 20]
    common = ep["U"].copy()
    for k, v in V.items():
        if supported[k] and (k[1] in KP or k[1] == "PRIMARY"):
            common &= np.isfinite(v)
    assert (common == ep["U"]).all()                                                     # deep-depth partial coverage does not touch the common mask
    st = evaluate(ep, V, common, 200, supported)
    assert st["pairU", 20, "resp"]["n_lost"] > 0 and st["pairU", 5, "resp"]["n_lost"] == 0                 # deep depth on its own coverage; window depth on the common rows
    assert st["pairM", 3, "resp"]["status"] == "UNSUPPORTED_DEPTH" and "pt" not in st["pairM", 3, "resp"]
    assert set(st["pairU", 4, "resp"]["pt"]) == set(PT_KEYS) and 0 < st["pairU", 4, "resp"]["pt"]["PREV"] < 1 and np.isfinite(st["pairU", 4, "resp"]["pt"]["P20"])
    o0, o8 = st["pairU", 0, "occ_prim"]["pt"]["AUROC"], st["pairU", 8, "occ_prim"]["pt"]["AUROC"]
    assert o0 > 0.85 and o8 < 0.75, (o0, o8)
    for c in CONTRASTS:                                                                  # one shared bootstrap: identical scores -> identical draws
        assert np.array_equal(st["dup", 8, c]["bs"]["AUROC"], st["pairU", 8, c]["bs"]["AUROC"], equal_nan=True)
    T = targets_of(ep, common)
    assert T["AD"][1].sum() == (common & (ep["resp"] == ep["occ"]["prim"])).sum() and st["pairU", 8, "AD"]["n"] == T["AD"][1].sum()
    # M1: D_c = mean_{1..8} - k0 with paired draws
    auc = lambda s, k, c: st[s, k, c]["pt"]["AUROC"]                                     # noqa: E731
    dcm = np.mean([auc("pairU", k, "occ_prim") for k in range(1, 9)]) - auc("pairU", 0, "occ_prim")
    Sm = summarize(st, dict(system="GATA1", tf="T", tier="primary"))
    g = lambda v, c, m, s: Sm[(Sm.variant == v) & (Sm.contrast == c) & (Sm.metric == m) & (Sm.stat == s)].iloc[0]   # noqa: E731
    d = g("pairU", "occ_prim", "AUROC", "D_c")
    assert abs(d.est - dcm) < 1e-12 and d.hi < 0 and d.tier == "PRIMARY"
    dd = np.mean([st["pairU", k, "occ_prim"]["bs"]["AUROC"] for k in range(1, 9)], 0) - st["pairU", 0, "occ_prim"]["bs"]["AUROC"]
    assert np.isclose(d.hi, np.percentile(dd, 97.5)) and np.isclose(d.lo, np.percentile(dd, 2.5))
    d8 = g("pairU", "occ_prim", "AUROC", "delta_k8_minus_k0")
    assert abs(d8.est - (o8 - o0)) < 1e-12 and d8.tier == "SECONDARY" and g("pairU", "occ_prim", "AUROC", "spearman_k0_8").tier == "SUPPLEMENTARY"
    assert g("pairU", "occ_prim", "AUROC", "spearman_k0_8").est < -0.9
    assert Sm[(Sm.variant == "pairM") & (Sm.stat.isin(["D_c", "spearman_k0_8"]))].est.isna().all()      # missing depth 3 -> NA, never zero
    assert not ((Sm.stat == "D_c") & (Sm.metric != "AUROC")).any() and set(Sm.stat) <= set(SUMMARY_STATS) and (Sm.label == LABEL).all()
    # M2: simultaneous families, classes, outcomes
    inf, cls = infer(st, dict(system="GATA1", tf="T", tier="primary"))
    fam = inf[(inf.stat == "D_c")]
    assert fam.family_size.iloc[0] == len(fam) == 21 and (fam.q_maxT >= 1.96).all()      # 3 fully defined series (pairU, cellU, cellM) x 7; pairM has D_c NA; no pcorr here
    ptw = inf[(inf.stat == "D_c") & (inf.variant == "pairU") & (inf.contrast == "occ_prim")].iloc[0]
    assert ptw.hi_simul - ptw.lo_simul > ptw.hi_point - ptw.lo_point and inf[inf.stat == "delta_k8_minus_k0"].family_size.iloc[0] == 28
    assert cls["pairU", "AB"]["cls"] == "IMPROVED" and cls["pairU", "CD"]["is_improved"] and not cls["pairU", "AD"]["is_improved"]      # A>D never IMPROVED
    assert cls["pairU", "occ_prim"]["hi"] < 0 and cls["cellU", "AB"]["hi"] < 0 and cls["cellU", "CD"]["hi"] < 0
    oc = outcomes(cls, dict(system="GATA1", tf="T", tier="primary")).set_index("variant")
    assert oc.loc["pairU", "predictive_enrichment"] and oc.loc["pairU", "selective_retention"] and not oc.loc["pairU", "signal_removal"], oc.loc["pairU"]
    assert oc.loc["cellU", "response_loss"] and not oc.loc["cellU", "predictive_enrichment"] and oc.loc["pairM", "outcome"] == "NOT_EVALUABLE"
    assert ("response_loss" in oc.loc["cellU", "outcome"]) or ("signal_removal" in oc.loc["cellU", "outcome"])
    def crafted(**over):                                                                  # hand-built classes: all four contrasts, occupancy / response D_c bands
        base = {"occ_prim": (-.2, -.3, -.1), "BD": (-.2, -.3, -.1), "AB": (-.2, -.3, -.1), "CD": (-.2, -.3, -.1)}
        base.update(over)
        cl = {}
        for c, (e_, l_, h_) in base.items():
            cl["pairU", c] = dict(est=e_, lo=l_, hi=h_, gate=True, eq_raw=bool(l_ >= -EQ_BAND and h_ <= EQ_BAND), **classify(c, e_, l_, h_, .6, True))
        return cl
    o1 = outcomes(crafted(), dict(system="X", tf="T", tier="primary")).set_index("variant").loc["pairU"]
    assert o1.signal_removal and o1.response_loss and o1.outcome == "signal_removal" and not o1.selective_retention                                     # both deteriorate
    o2 = outcomes(crafted(AB=(0., -.01, .01), CD=(0., -.01, .01)), dict(system="X", tf="T", tier="primary")).set_index("variant").loc["pairU"]
    assert o2.selective_retention and o2.outcome == "selective_retention" and not o2.response_loss                                                     # occupancy down, response PRESERVED
    o3 = outcomes(crafted(occ_prim=(-.02, -.04, .01), BD=(-.02, -.04, .01), AB=(0., -.01, .01), CD=(0., -.01, .01)), dict(system="X", tf="T", tier="primary")).set_index("variant").loc["pairU"]
    assert not o3.selective_retention and o3.SENSITIVITY_selective_retention_literal_lo_lt0 and "selective_retention" not in o3.outcome and not o3.occupancy_down     # hi >= 0: lo<0 never drives outcome
    o4 = outcomes(crafted(occ_prim=(-.02, -.04, -.005)), dict(system="X", tf="T", tier="primary")).set_index("variant").loc["pairU"]
    assert not o4.occupancy_equivalent_neg                                                                                                              # BD not equivalent -> not both
    o5 = outcomes(crafted(occ_prim=(-.02, -.03, -.01), BD=(-.02, -.03, .03), AB=(0., -.01, .01), CD=(0., -.01, .01)), dict(system="X", tf="T", tier="primary")).set_index("variant").loc["pairU"]
    assert o5.occupancy_equivalent_neg and not o5.occupancy_down and not o5.selective_retention                                                         # EQUIVALENT-down: own column only
    cg = crafted()
    cg["pairU", "AB"] = dict(cg["pairU", "AB"], gate=False, **classify("AB", -.2, -.3, -.1, .6, False))
    o6 = outcomes(cg, dict(system="X", tf="T", tier="primary")).set_index("variant").loc["pairU"]
    assert o6.outcome == "NO_LABEL_DESCRIPTIVE_ONLY" and cg["pairU", "AB"]["cls"] == "DESCRIPTIVE_ONLY"                                                 # gate failure: no label
    e = classify("AB", .1, .05, .2, .6)
    assert e["cls"] == "IMPROVED" and classify("AD", .1, .05, .2, .6)["cls"] == "PRESERVED" and classify("CD", 0, -.03, .03, .6)["cls"] == "EQUIVALENT"
    assert classify("resp", 0, -.03, .03, .4)["cls"] == "EQUIVALENT" and classify("resp", 0, -.1, .3, .6)["cls"] == "UNRESOLVED" and classify("resp", np.nan, 0, 0, .6)["cls"] == "NA"
    assert not classify("AB", 0, -.019, .01, .5)["is_preserved"]                        # PRIMARY-level lo must be strictly > .50
    Pf = profile_frame(st, dict(system="GATA1", tf="T", tier="primary"))
    assert (Pf.label == LABEL).all() and {"AUROC_lo", "prevalence", "P50", "n_lost_vs_common", "window"} <= set(Pf.columns) and set(Pf.window) >= {"common_0_8", "deep_9_30"}
    dw = dial_wide(Pf)
    assert {"F", "O", "F_lo", "O_hi"} <= set(dw.columns) and len(dw[dw.variant == "pairU"]) == 32 and (dw.tier_of_stat == "SUPPLEMENTARY").all()
    # 5. no depth selection anywhere
    bad = re.compile("arg" + "max|arg" + "min|idx" + "max|idx" + "min|best" + "_k|nlarg" + "est")
    assert not bad.search(open(__file__).read())
    assert set(inspect.signature(depth_scores).parameters) == {"raw", "uni", "ks", "per_root"} and set(inspect.signature(pcorr_profile).parameters) == {"x", "Y", "Qc", "chunk"}
    assert "truth" not in "".join(inspect.signature(diagnostics).parameters) and "truth" not in "".join(inspect.signature(null_table).parameters)
    # 6. shard loader (P|*, several sources), fail-closed cases
    def mk(d, verified=True, tamper=False, cover=True, n_src=3):
        Gs = 30
        srcs = [f"S{i}" for i in range(n_src)]
        members = srcs + ([] if cover else ["Z"])
        bank = dict(members=members, sha256_members=L.members_sha(members), detection={m: .5 for m in members})
        json.dump(bank, open(f"{d}/SYS_bank.json", "w"))
        arrs = dict(genes=np.array([f"g{i}" for i in range(Gs)]), sources=np.array(srcs), bank_idx=np.arange(n_src), universe=rng.random((n_src, Gs)) < .9)
        for h in GP.ROOTS:
            for suf in ("PU", "PM"):
                arrs[f"raw|{h}|{suf}"] = rng.random((n_src, Gs, 31)).astype(np.float32)
        for suf in ("CU", "CM"):
            arrs[f"raw|cell|{suf}"] = rng.random((n_src, Gs, 31)).astype(np.float32)
        for suf in ("PU", "PM", "CU", "CM"):
            arrs[f"P|{suf}"] = rng.normal(size=(n_src, Gs))
        f = f"{d}/SYS_scores_0-{n_src}.npz"
        np.savez(f, **arrs)
        json.dump(dict(output_sha256="0" * 64 if tamper else L._sha_file(f), binding=dict(config=dict(bank_sha256=bank["sha256_members"])), production_universe=True,
                       parity_verified=verified), open(f"{d}/SYS_scores_0-{n_src}.receipt.json", "w"))
        return arrs

    def blocked(**kw):
        with tempfile.TemporaryDirectory() as d2:
            mk(d2, **kw)
            try:
                load_raw_many("SYS", ["S1"], d2)
            except L.Blocked as e:
                return str(e)
            raise AssertionError(f"accepted {kw}")
    with tempfile.TemporaryDirectory() as d:
        arrs = mk(d)
        bk, r = load_raw_many("SYS", ["s1", "S2"], d)
        assert set(r) == {"S1", "S2"} and np.array_equal(r["S1"]["uni"], arrs["universe"][1]) and np.array_equal(r["S2"]["raw"]["pairM"][GP.ROOTS[1]], arrs[f"raw|{GP.ROOTS[1]}|PM"][2])
        assert np.array_equal(r["S1"]["P"]["cellM"], arrs["P|CM"][1]) and np.array_equal(r["S1"]["raw"]["cellU"][GP.ROOTS[0]], r["S1"]["raw"]["cellU"][GP.ROOTS[1]])
        try:                                                                              # m2: a stored P|* that disagrees with the raw-derived PRIMARY is refused
            sc = depth_scores(r["S1"]["raw"], r["S1"]["uni"], ks=KP, per_root=False)
            check_primary_parity(sc, r["S1"]["P"])
            raise AssertionError("mismatching stored PRIMARY accepted")
        except L.Blocked as e:
            assert "SHARD_PRIMARY_PARITY_FAILED" in str(e)
        okP = {v: sc[v, "PRIMARY"].copy() for v in GB_VARIANTS}
        assert max(check_primary_parity(sc, okP).values()) == 0.0
        for nm, kw in (("FOCAL_NOT_IN_BANK", dict(f=["NOPE"])), ("NO_BANK", dict(f=["S1"], d=d + "/nonexistent"))):
            try:
                load_raw_many("SYS", kw["f"], kw.get("d", d))
                raise AssertionError(nm)
            except L.Blocked as e:
                assert nm in str(e)
    assert "STALE_ARTIFACT" in blocked(tamper=True) and "UNVERIFIED" in blocked(verified=False) and "SHARD_COVERAGE" in blocked(cover=False)
    with tempfile.TemporaryDirectory() as d:
        json.dump(dict(members=["A"], sha256_members=L.members_sha(["A"])), open(f"{d}/SYS_bank.json", "w"))
        try:
            load_raw_many("SYS", ["A"], d)
            raise AssertionError("no shards accepted")
        except L.Blocked as e:
            assert "NO_SCORE_SHARDS" in str(e)
    # 7. M3 diagnostics: identities against an exact reference projector, fail-closed parity vs the
    # shard raw
    P_, Gd = 240, 13
    Qd, _ = np.linalg.qr(rng.normal(size=(P_, 30)))
    dxs = rng.normal(size=(P_, 1)).astype(np.float32)
    yU, yM = rng.normal(size=(P_, Gd)).astype(np.float32), (rng.normal(size=(P_, Gd)) + .5 * dxs).astype(np.float32)
    dg = diag_pass(_project_np, dxs, lambda sl: (yU[:, sl], yM[:, sl]), Qd, Gd, 30, chunk=5)
    for ch, yy in (("U", yU), ("M", yM)):
        d_ = dg[ch]
        assert np.allclose(d_["num"][0], (dxs.astype(float).T @ yy.astype(float))[0]) and np.allclose(d_["den"][0], (np.abs(dxs).astype(float).T @ np.abs(yy).astype(float))[0])
        assert np.allclose(d_["num"] - d_["num"][0][None], -np.vstack([np.zeros((1, Gd)), np.cumsum(d_["c"], 0)]), atol=1e-8)                                # N_k = N_0 - sum_{l<=k} c_l
        Cx, Cy = Qd.T @ dxs.astype(float)[:, 0], Qd.T @ yy.astype(float)
        assert np.allclose(d_["c"], Cx[:, None] * Cy)
        rx30 = dxs.astype(float)[:, 0] - Qd[:, :30] @ Cx
        assert abs(d_["rxf"][30] - (rx30 @ rx30) / (dxs.astype(float)[:, 0] @ dxs.astype(float)[:, 0])) < 1e-9 and d_["rxf"][0] == 1.0
        assert np.allclose(d_["ryf"][5], ((yy.astype(float) - Qd[:, :5] @ Cy[:5]) ** 2).sum(0) / (yy.astype(float) ** 2).sum(0), atol=1e-9)
        assert np.all(d_["z2"] > 0) and np.allclose(d_["absS"][0], np.abs(d_["num"][0]) / d_["den"][0])
    fake = {"pairU": {h: dg["U"]["absS"].T.astype(np.float32) for h in GP.ROOTS}, "pairM": {h: dg["M"]["absS"].T.astype(np.float32) for h in GP.ROOTS},
            "cellU": {h: dg["U"]["absS"].T.astype(np.float32) for h in GP.ROOTS}, "cellM": {h: dg["M"]["absS"].T.astype(np.float32) for h in GP.ROOTS}}
    D3 = {GP.ROOTS[0]: dg, GP.ROOTS[1]: dg, "cell": dg}
    assert max(check_diag_parity(D3, fake).values()) < 1e-6
    fake["pairU"][GP.ROOTS[1]][4, 7] += 1e-4
    try:
        check_diag_parity(D3, fake)
        raise AssertionError("diagnostic parity violation accepted")
    except L.Blocked as e:
        assert "DIAGNOSTIC_PARITY_FAILED" in str(e)
    try:                                                                                  # the real projector (numexpr) must agree with the reference where available
        import numexpr  # noqa: F401
        dg2 = diag_pass(_default_project(), dxs, lambda sl: (yU[:, sl], yM[:, sl]), Qd, Gd, 30, chunk=5)
        assert np.abs(dg2["U"]["absS"] - dg["U"]["absS"]).max() < 1e-5
    except ImportError:
        pass
    # diag_summary: class medians / contrast AUROC of a planted quantity on ep genes
    selr = np.arange(1500)
    diagS = {lv: {ch: {n: np.tile(np.linspace(1, 2, 1500)[None, :], ((31 if n not in ("c",) else 30), 1)) for n in ("num", "den", "z2", "absS", "ryf")} | {"c": np.ones((30, 1500)), "rxf": np.linspace(1, .5, 31)} for ch in "UM"}
             for lv in (GP.ROOTS[0], GP.ROOTS[1], "cell")}
    ds_ = diag_summary(diagS, {}, selr, ep, common, dict(system="GATA1", tf="T", tier="primary"))
    a_ = ds_[(ds_.level == "class") & (ds_.quantity == "absS") & (ds_.k == 0) & (ds_.key == f"{GP.ROOTS[0]}|U")].set_index("group")
    mm = class_masks(ep, common)
    assert set(a_.index) == set("ABCD") and abs(a_.loc["A", "median"] - np.median(np.linspace(1, 2, 1500)[mm["A"]])) < 1e-12 and a_.loc["B", "n"] == mm["B"].sum()
    assert ds_[(ds_.level == "source")].shape[0] == 3 * len(DK) and ((ds_.level == "contrast") & ds_.auroc.notna()).any()
    # 8. M6 matched null: nearest sources are the nearest, focal excluded
    mem = [f"M{i}" for i in range(60)]
    det = {m: i / 60 for i, m in enumerate(mem)}
    lms, lmu = {m: 0.0 for m in mem}, {m: 0.0 for m in mem}                              # distance driven by detection only
    nn = nearest_sources(mem, det, lms, lmu, "M30", 20)
    assert len(nn) == 20 and "M30" not in nn and set(nn) >= {"M29", "M31"} and nn == nearest_sources(mem, det, lms, lmu, "M30", 20)
    nn2 = nearest_sources(mem + ["m30"], {**det, "m30": .5}, {**lms, "m30": 0.0}, {**lmu, "m30": 0.0}, "M30", 20)
    assert "M30" not in nn2 and "m30" not in nn2                                         # case-collision of the focal excluded
    Tn = {c: t for c, t in T.items() if c in ("resp", "AB")}
    fv = {"pairU": [V["pairU", k] for k in KP]}
    alt = {f"a{i}": {"pairU": [np.random.default_rng(100 + i).normal(size=1500) for _ in KP]} for i in range(20)}
    nt = null_table(fv, alt, Tn, {"resp": True, "AB": True}, dict(system="GATA1", tf="T", tier="primary"))
    r_ = nt[nt.contrast == "AB"].iloc[0]
    assert "pool rank" in r_.p_label and r_.pool_n == 20 and r_.p_ge == 1 / 21 and r_.focal_percentile == 1.0 and r_.p_le == 1.0 and abs(r_.focal - dc_point(fv["pairU"], *Tn["AB"])) < 1e-15
    assert abs(dc_point(fv["pairU"], *T["resp"]) - dc_parts(lambda k: _get(st, "pairU", k, "resp", "AUROC"))[0]) < 1e-9        # null D_c function == bootstrap-store point D_c
    assert nul_na_status(null_table(fv, {"a": alt["a0"]}, Tn, {}, {})) == {"NA_POOL_LT10_OR_UNDEFINED"}                          # pool < 10 -> NA, no p
    # 9. merge refuses mixed provenance / desynchronised draws
    def mkpart(d, s, block, sha="x", fp="f", seed=1, mod="m1"):
        for p in PARTS:
            L.atomic_tsv(pd.DataFrame([dict(system=s, tf="T", variant="pairU", depth="0", contrast="resp", AUROC=.5)]) if p == "profile" else pd.DataFrame(), f"{d}/{p}_{s}.tsv")
        L.atomic_json(f"{d}/receipt_{s}.json", dict(script_sha256=sha, B=200, spec_log=dict(sha256="a"), module_shas={"gb_primary": mod}, counts_fingerprint=fp, boot_seed=seed,
                                                     block_scheme="gene", block=block, system=s, endpoints=[dict(system=s)]))
    for kw, tag in ((dict(sha="y"), "MIXED_SCRIPT_SHA256"), (dict(mod="m2"), "MIXED_MODULE_SHAS"), (dict(fp="g"), "DESYNCHRONISED_DRAWS")):
        with tempfile.TemporaryDirectory() as d:
            mkpart(d, "GATA1", "ERY")
            mkpart(d, "KLF1", "ERY", **kw)
            try:
                merge(d)
                raise AssertionError(f"{tag} accepted")
            except SystemExit as e:
                assert tag in str(e), str(e)
    with tempfile.TemporaryDirectory() as d:
        mkpart(d, "GATA1", "ERY")
        mkpart(d, "PANC", "PANC", fp="other", seed=2)
        merge(d)
        assert len(pd.read_csv(f"{d}/profile.tsv", sep="\t")) == 2 and json.load(open(f"{d}/receipt.json"))["counts_fingerprints"] == {"GATA1": "f", "PANC": "other"}
    # 10. end-to-end run_tf / run_system / merge on a synthetic world
    def world(G=150, nc=160, Pn=300, nsrc=30, seed=0):
        r = np.random.default_rng(seed)
        axis = np.array([f"g{i}" for i in range(G)])
        Ms = r.gamma(2, 1, size=(nc, G)).astype(np.float32)
        Mu = (0.5 * Ms + r.gamma(2, 1, size=(nc, G))).astype(np.float32)
        P30 = r.normal(size=(nc, 30))
        Qc_, _ = np.linalg.qr(P30 - P30.mean(0))
        pairs = {}
        for h in GP.ROOTS:
            I = r.integers(0, nc, Pn)
            pairs[h] = (I, (I + 1 + r.integers(0, nc - 1, Pn)) % nc)
        return dict(axis=axis, Ms=Ms, Mu=Mu, Qc=Qc_, P30=P30, X=Ms.astype(np.float64), fi={g: i for i, g in enumerate(axis)}, pairs=pairs,
                    lms=np.log1p(Ms.astype(float).mean(0)), lmu=np.log1p(Mu.astype(float).mean(0))), nsrc
    ctx, nsrc = world()
    G_ = len(ctx["axis"])
    raws = {v: {h: np.full((nsrc, G_, 31), np.nan, np.float32) for h in GP.ROOTS} for v in GB_VARIANTS}
    unis, Ps = np.zeros((nsrc, G_), bool), {v: np.full((nsrc, G_), np.nan) for v in GB_VARIANTS}
    for j in range(nsrc):
        dj = diagnostics(ctx, ctx["axis"][j], j, project=_project_np)
        for h in GP.ROOTS:
            raws["pairU"][h][j], raws["pairM"][h][j] = dj[h]["U"]["absS"].T, dj[h]["M"]["absS"].T
            raws["cellU"][h][j], raws["cellM"][h][j] = dj["cell"]["U"]["absS"].T, dj["cell"]["M"]["absS"].T
        unis[j] = np.all([np.isfinite(raws[v][h][j]).all(1) for v in ("pairU", "pairM") for h in GP.ROOTS], 0)
        pj = depth_scores({v: {h: raws[v][h][j] for h in GP.ROOTS} for v in GB_VARIANTS}, unis[j], ks=(), per_root=False)
        for v in GB_VARIANTS:
            Ps[v][j] = pj[v, "PRIMARY"]
    old = dict(cat=L.CATBANK, bld=L.build_endpoint, dp=_default_project, sys=L.SYSTEMS.get("SYSX"), plan=PLAN.get("SYSX"), ax=_AX.get("SYS"))
    try:
        with tempfile.TemporaryDirectory() as cb, tempfile.TemporaryDirectory() as out:
            members = [str(g) for g in ctx["axis"][:nsrc]]
            bank = dict(members=members, sha256_members=L.members_sha(members), detection={m: float(i) / nsrc for i, m in enumerate(members)}, inputs=dict(fixture_sha256="fx"))
            json.dump(bank, open(f"{cb}/SYS_bank.json", "w"))
            arrs = dict(genes=ctx["axis"], sources=np.array(members), bank_idx=np.arange(nsrc), universe=unis)
            for v, (suf, cell) in GB_VARIANTS.items():
                arrs[f"P|{suf}"] = Ps[v]
                for h in GP.ROOTS:
                    arrs[f"raw|cell|{suf}" if cell else f"raw|{h}|{suf}"] = raws[v][h]
            f = f"{cb}/SYS_scores_0-{nsrc}.npz"
            np.savez(f, **arrs)
            json.dump(dict(output_sha256=L._sha_file(f), binding=dict(config=dict(bank_sha256=bank["sha256_members"])), production_universe=True, parity_verified=True),
                      open(f"{cb}/SYS_scores_0-{nsrc}.receipt.json", "w"))
            L.CATBANK = cb
            _AX["SYS"] = ctx
            globals()["_default_project"] = lambda: _project_np
            L.SYSTEMS["SYSX"] = dict(block="ERYTHROID", axis="SYS", tfs=["g3", "g4", "g5"])
            PLAN["SYSX"] = [("g3", "primary"), ("NOPE", "descriptive")]

            def fake_build(system, tf, blk_, smalln):
                e = L._syn_ep(blk_, tf, rng=np.random.default_rng(21), resp_rate=.3, occ_rate=.35)
                j = members.index(tf)
                e["V"][L.GB] = Ps["pairU"][j].copy()                                          # the production GB vector == the shard pairU PRIMARY (parity)
                e["V"]["|Pearson|"] = np.nan_to_num(np.abs(np.random.default_rng(5).normal(size=len(e["genes"]))))
                return e
            L.build_endpoint = fake_build
            st_ = {}
            res, rc = run_tf("SYSX", "g3", "primary", st_, 30, True)
            assert rc["status"] == "OK" and rc["n_common"] >= 0.9 * rc["n_U"] and rc["diagnostic_parity_max_abs_diff"]["cell|U"] < 1e-6 and len(rc["alternatives"]) == 20
            assert "g3" not in rc["alternatives"] and rc["primary_parity_max_abs_diff"] == 0.0 and set(rc["shard_primary_parity"]) == set(GB_VARIANTS)
            assert set(res) == set(PARTS) and len(res["exclusions"]) == 4 and set(res["exclusions"]["class"]) == set("ABCD")
            assert (res["exclusions"].n_common <= res["exclusions"].n_U).all()
            pf = res["profile"]
            assert {"pairU", "pairU@dc1_min", "pairM@dc1_max", "pcorr", "pcorrMM", "cellU", "cellM", "pearson"} <= set(pf.variant) and set(pf.window) >= {"common_0_8", "deep_9_30"}
            assert res["null"].pool_n.max() == 20 and (res["null"].label == LABEL).all() and len(res["diagnostics"]) > 500
            assert {"class", "contrast", "source"} == set(res["diagnostics"].level) and {"root_disagree_z", "neff", "c_over_D0", "target_resid_frac"} <= set(res["diagnostics"].quantity)
            assert set(res["outcomes"].variant) == set(GB_VARIANTS) and set(res["summary"].stat) <= set(SUMMARY_STATS)
            d0 = res["diagnostics"]
            dsrc = d0[d0.level == "source"]
            assert (dsrc[dsrc.k == 0]["median"] == 1.0).all() and (dsrc[dsrc.k == 30]["median"] < 1.0).all()                                    # source residual energy shrinks with depth
            run_system("SYSX", out, 30, [], True)                                                                                           # g3 OK, NOPE NOT_RUN, g4/g5 NOT_IN_PLAN
            merge(out)
            prof = pd.read_csv(f"{out}/profile.tsv", sep="\t")
            stt = prof[prof.variant == "*"].set_index("tf").status
            assert stt["g4"] == "NOT_IN_PLAN" and stt["g5"] == "NOT_IN_PLAN" and stt["NOPE"].startswith("NOT_RUN: FOCAL_NOT_IN_BANK")
            recm = json.load(open(f"{out}/receipt.json"))
            assert "NO UTILITY CLAIM" in recm["note"] and recm["spec_log"]["path"] == SPEC_LOG and any(e.get("status") == "OK" for e in recm["endpoints"])
            for p_ in PARTS:
                assert os.path.exists(f"{out}/{p_}.tsv")
            # fail closed: a shard whose stored raw disagrees with the recomputed diagnostics is
            # refused, not evaluated
            z = dict(np.load(f))
            z["raw|dc1_min|PU"] = z["raw|dc1_min|PU"].copy()
            z["raw|dc1_min|PU"][3, 5, 4] += 1e-3
            np.savez(f, **z)
            rj = json.load(open(f.replace(".npz", ".receipt.json")))
            json.dump(dict(rj, output_sha256=L._sha_file(f)), open(f.replace(".npz", ".receipt.json"), "w"))
            L.sha_file.__globals__["_SHA"].clear()
            try:
                run_tf("SYSX", "g3", "primary", {}, 30, True)
                raise AssertionError("diagnostic mismatch accepted")
            except L.Blocked as e:
                assert "PARITY_FAILED" in str(e), str(e)
    finally:
        L.CATBANK, L.build_endpoint = old["cat"], old["bld"]
        globals()["_default_project"] = old["dp"]
        for d_, key, v_ in ((L.SYSTEMS, "SYSX", old["sys"]), (PLAN, "SYSX", old["plan"]), (_AX, "SYS", old["ax"])):
            d_.pop(key, None) if v_ is None else d_.__setitem__(key, v_)
    print("gb_lattice_kprofile selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--systems", default="PANC,BONEMARROW,GATA1,KLF1")
    ap.add_argument("--tfs", default="", help="optional TF filter within the plan (e.g. RUNX1)")
    ap.add_argument("--smalln", action="store_true", help="PANC (RFX6) only, B=200 -> lattice_kprofile_smalln/")
    ap.add_argument("--B", type=int, default=None)
    ap.add_argument("--stage", choices=["all", "shard", "merge"], default="all")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-diagnostics", action="store_true", help="skip M3 (the project_scores recomputation is ~10 min per root per TF); the receipt records diagnostics=false")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    systems = ["PANC"] if a.smalln else [s.strip().upper() for s in a.systems.split(",") if s.strip()]
    bad = [s for s in systems if s not in PLAN]
    if bad:
        raise SystemExit(f"REFUSING: systems {bad} not in the declared plan {list(PLAN)}")
    B = B_SMALL if a.smalln else (a.B or B_FULL)
    out = a.out or (OUT_DIR + "_smalln" if a.smalln else OUT_DIR)
    os.makedirs(out, exist_ok=True)
    if a.stage in ("all", "shard"):
        for s in systems:
            run_system(s, out, B, [t.strip().upper() for t in a.tfs.split(",") if t.strip()], not a.no_diagnostics)
    if a.stage in ("all", "merge"):
        merge(out)


if __name__ == "__main__":
    main()
