#!/usr/bin/env python
# Permutation and split-sample diagnostics.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

for _i, _a in enumerate(sys.argv):                                    # thread cap before numpy import (node is shared)
    if _a == "--threads" and _i + 1 < len(sys.argv):
        for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "NUMEXPR_MAX_THREADS"):
            os.environ[_v] = sys.argv[_i + 1]

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_primary as GP  # noqa: E402
import gb_real_uniform as RU  # noqa: E402
import gb_catbank as CB  # noqa: E402

ROOTS = GP.ROOTS
SYSTEM_LIST = ["BONEMARROW", "CARDIACV", "DENTATEGYRUS", "GASTRULATION", "GATA1", "K562", "PANC", "RETINA", "ZEBRAFISH"]   # alphabetical index
SEED0, NSUB, KMAX = 20261001, 100, 30
OUT = RU.OUT / "defense"
FOCAL = {"GATA1": ["GATA1", "KLF1"], "PANC": ["RFX6"], "BONEMARROW": ["GATA1", "GATA2", "RUNX1", "FLI1", "TAL1"]}   # gb_layer_lattice.SYSTEMS (K562: see focal_tfs)
STRATA_COL = {"GASTRULATION": "sample", "DENTATEGYRUS": "age(days)"}   # obs column: D5 permutes within levels; D6 donor-held-out halves (partition of levels)
CANON = Path(RU.R) / "results" / "canonical_2026_09_18"


# utilities
def me_sha():
    return RU.sha(Path(__file__).resolve())


def sidx(system):
    return SYSTEM_LIST.index(system.upper())


def rng_for(system, diag, rep):
    return np.random.default_rng(np.random.SeedSequence([SEED0, sidx(system), diag, rep]))


# Spearman on the entries finite in both (NaN if < minn)
def spear(a, b, minn=3):
    from scipy.stats import spearmanr
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < minn or np.ptp(a[m]) == 0 or np.ptp(b[m]) == 0:
        return float("nan")
    return float(spearmanr(a[m], b[m])[0])


def q3(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {"n": 0, "median": None, "q25": None, "q75": None, "p5": None}
    return {"n": int(len(v)), "median": float(np.median(v)), "q25": float(np.percentile(v, 25)), "q75": float(np.percentile(v, 75)),
            "p5": float(np.percentile(v, 5))}


def auroc(score, y):
    from scipy.stats import rankdata
    m = np.isfinite(score)
    s, y = score[m], np.asarray(y, bool)[m]
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def focal_tfs(system):
    if system in FOCAL:
        return FOCAL[system]
    if system == "K562":
        try:
            import gb_layer_lattice as LL
            return list(LL.k562_tfs())
        except Exception:
            return []
    return []


# 100 uniformly sampled NONFOCAL members (from the SORTED nonfocal ids) + all focal TFs present in the bank, returned sorted
def pick_subset(system, members):
    foc = {f.upper() for f in focal_tfs(system)}
    present = sorted(m for m in members if m.upper() in foc)
    nonfocal = sorted(m for m in members if m.upper() not in foc)
    rng = rng_for(system, 0, 0)
    chosen = sorted(np.asarray(nonfocal)[np.sort(rng.choice(len(nonfocal), min(NSUB, len(nonfocal)), replace=False))].tolist())
    return sorted(chosen + present), present


def strata_labels(system, A_obs_names):
    col = STRATA_COL.get(system)
    if col is None:
        return None, None
    import anndata as ad
    raw = ad.read_h5ad(RU.R / "data" / {"GASTRULATION": "Gastrulation/gastrulation_e75.h5ad", "DENTATEGYRUS": "DentateGyrus/dentategyrus_raw.h5ad"}[system], backed="r")
    if col not in raw.obs.columns:
        return None, None
    lab = raw.obs[col].reindex(A_obs_names).astype(str).to_numpy()
    return lab, col


def stratified_perm(n, labels, rng):
    perm = np.arange(n)
    if labels is None:
        return rng.permutation(n)
    for lv in np.unique(labels):
        idx = np.flatnonzero(labels == lv)
        perm[idx] = rng.permutation(idx)
    return perm


# scoring (production algebra)
# Production pair-space algebra (gb_catbank.cmd_scores:340-360) on arbitrary (Ms, Mu, P30, pairs)
def raw_rows(Ms, Mu, P30, pairs, src_col, col_axis, G, kmax=KMAX, perm=None, block=128, x_from=None):  # cost is dominated by the TARGET side (independent of #sources): one block for the whole subset
    ns, K = len(src_col), kmax + 1
    out = {(h, c): np.full((ns, G, K), np.nan, np.float32) for h in ROOTS for c in ("U", "X")}
    ok = np.asarray(src_col) >= 0
    Mu_t = Mu if perm is None else Mu[perm]
    for root in ROOTS:
        I, J = (np.asarray(x, int) for x in pairs[root])
        Qf, _ = np.linalg.qr(P30[J] - P30[I])
        for b0 in range(0, ns, block):
            idx = np.flatnonzero(ok[b0:b0 + block]) + b0
            if not len(idx):
                continue
            xb = np.asarray(Ms[:, np.asarray(src_col)[idx]], np.float64)
            dx = (xb[J] - xb[I]).astype(np.float32)
            if x_from is None:
                rU, rM, _, _ = CB.score_block(dx, lambda sl: (Mu_t[J, sl] - Mu_t[I, sl], Ms[J, sl] - Ms[I, sl]), Ms.shape[1], Qf, kmax)
                out[(root, "U")][np.ix_(idx, col_axis)] = rU
                out[(root, "X")][np.ix_(idx, col_axis)] = rM
            else:                                                     # U channel only (same per-target algebra, half the cost); X channel copied from x_from
                from gb_target_extension import project_scores
                rU = np.full((len(idx), Ms.shape[1], K), np.nan, np.float32)
                for g0 in range(0, Ms.shape[1], 500):
                    sl = slice(g0, min(g0 + 500, Ms.shape[1]))
                    for k, _n, _d, _z, absS in project_scores(dx, (Mu_t[J, sl] - Mu_t[I, sl]).astype(np.float32), Qf, kmax=kmax):
                        rU[:, sl, k] = absS
                out[(root, "U")][np.ix_(idx, col_axis)] = rU
    if x_from is not None:
        for h in ROOTS:
            out[(h, "X")] = np.array(x_from[(h, "X")], copy=True)
    return out


def blank_self(raw, src_axis):
    for s, a in enumerate(src_axis):
        if a >= 0:
            for v in raw.values():
                v[s, a, :] = np.nan


# production PRIMARY rows through CB.obs_universe + CB.primary_diag (-> GP.zcons)
def primary_rows(raw, srcs, au):
    ns, G = raw[(ROOTS[0], "U")].shape[:2]
    P, U = np.full((ns, G), np.nan), np.zeros((ns, G), bool)
    for s in range(ns):
        rows = {k: v[s] for k, v in raw.items()}
        if not np.isfinite(rows[(ROOTS[0], "U")]).any():
            continue                                                  # absent source: unsupported, stays missing
        U[s] = uni = CB.obs_universe(srcs[s], rows, au)
        if uni.sum() >= 3:
            P[s] = CB.primary_diag({h: rows[(h, "U")] for h in ROOTS}, uni)["P"]
    return P, U


# subset-P = full-bank prior (mean of row-z over strict common finite T) of the given PRIMARY rows, through gb_catbank.build_set/loo_prior
# Rows that are all-NaN or degenerate on T are dropped ITERATIVELY and LISTED (build_set refuses them; rows are otherwise never dropped)
def subset_prior(Prows, bank_cols, names):
    keep = np.arange(len(Prows))
    dropped = []
    while True:
        X = np.array(Prows[keep], float)
        X[np.arange(len(keep)), np.asarray(bank_cols)[keep]] = np.nan
        bad = ~np.isfinite(X).any(1)
        if not bad.any():
            T = np.isfinite(X).all(0)
            if T.sum() < 3:
                return None
            mu, _sd = CB.row_consts(X, T)
            bad = ~np.isfinite(mu)
        if not bad.any():
            break
        dropped += [names[i] for i in keep[bad]]
        keep = keep[~bad]
        if len(keep) < 3:
            return None
    R = CB.build_set({"PU": np.asarray(Prows[keep], float)}, np.asarray(bank_cols)[keep], names=[names[i] for i in keep])
    r = R["methods"]["PU"]
    P = np.full(Prows.shape[1], np.nan)
    P[R["T"]] = CB.loo_prior(r["Z"], r["S"], r["C"], None)
    Z = r["Z"]
    return {"P": P, "T": R["T"], "n_rows": int(len(keep)), "dropped": dropped, "nT": int(R["T"].sum()),
            "varfrac": float(len(keep) * np.nansum(P[R["T"]] ** 2) / np.sum(Z ** 2)), "keep": keep}


# per-arm D5 metrics. raw: raw dict
def arm_summary(raw, P, U, srcs, au, bank_cols, det, real=None):
    sp = subset_prior(P, bank_cols, srcs)
    out = {"support": {"universe_frac_median": float(np.median(U.mean(1))), "P_finite_frac_median": float(np.median(np.isfinite(P).mean(1))),
                       "n_sources_with_P": int((np.isfinite(P).any(1)).sum())}}
    for k in (0, 4, 8):                                              # raw |S| spread inside each member's universe (pair-U, root-mean of the two roots)
        sds, meds = [], []
        for s in range(len(srcs)):
            if U[s].any():
                v = np.concatenate([raw[(h, "U")][s][U[s], k] for h in ROOTS]).astype(float)
                sds.append(float(np.std(v)))
                meds.append(float(np.median(v)))
        out[f"raw_k{k}"] = {"sd_across_targets": q3(sds), "median_abs_S": q3(meds)}
    if sp is None:
        out["subset_P"] = None
        return out, None
    out["subset_P"] = {"nT": sp["nT"], "n_rows": sp["n_rows"], "dropped": sp["dropped"], "varfrac": sp["varfrac"],
                       "spearman_P_vs_detection": spear(sp["P"], det)}
    return out, sp


# real arm (shards)
# raw |S| for `srcs` from the production catbank score shards (verified-parity shards only), same layout as raw_rows
def real_raw(system, D, B, srcs):
    members = B["members"]
    pos = {m: i for i, m in enumerate(members)}
    want = {pos[s]: si for si, s in enumerate(srcs)}
    G = len(D["axis"])
    out = {(h, c): np.full((len(srcs), G, KMAX + 1), np.nan, np.float32) for h in ROOTS for c in ("U", "X")}
    files = sorted(f for f in CB.OUT.iterdir() if CB.shard_pattern(system).fullmatch(f.name))
    used = []
    for f in files:
        with np.load(f, allow_pickle=True) as z:
            idx = np.asarray(z["bank_idx"], int)
            js = [j for j, bi in enumerate(idx) if int(bi) in want]
            if js:                                                    # each npz key is read once per shard (every access decompresses the whole array)
                dst = [want[int(idx[j])] for j in js]
                for h in ROOTS:
                    out[(h, "U")][dst] = np.asarray(z[f"raw|{h}|PU"])[js]
                    out[(h, "X")][dst] = np.asarray(z[f"raw|{h}|PM"])[js]
            used.append(f)
    return out, used


def common_setup(system):
    D = RU.canonical_inputs(system)
    B = CB.load_bank(system, D)
    srcs, present = pick_subset(system, B["members"])
    axis = np.asarray(D["axis"]).astype(str)
    au = np.char.upper(axis)
    ai = {g: i for i, g in enumerate(axis)}
    src_axis = np.asarray([ai[s] for s in srcs])
    det = np.asarray((D["A"].layers["spliced"][:, D["axis_cols"]] > 0).mean(0)).ravel()
    return D, B, srcs, present, axis, au, src_axis, det


def full_fixture_arrays(D):
    cols, A = D["axis_cols"], D["A"]
    return (RU.dense(A.layers["Ms"][:, cols], np.float32), RU.dense(A.layers["Mu"][:, cols], np.float32),
            np.asarray(A.obsm["X_pca"], float)[:, :CB.NPC], {r: tuple(np.asarray(x, int) for x in D["S"]["pairs"][r]) for r in ROOTS})


def receipt(path, system, extra):
    path = Path(path)
    RU.atomic_json(path.with_suffix(".receipt.json"), {"output": str(path), "output_sha256": RU.sha(path), "script": str(Path(__file__).resolve()),
                                                       "script_sha256": me_sha(), "system": system, "truth_read": extra.pop("truth_read", False),
                                                       "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **extra})


# D5
def cmd_d5(a):
    system = a.system.upper()
    t0 = time.time()
    D, B, srcs, present, axis, au, src_axis, det = common_setup(system)
    od = OUT / "D5"
    od.mkdir(parents=True, exist_ok=True)
    tag = f"{system}{'_smalln' if a.smalln else ''}"
    Ms, Mu, P30, pairs = full_fixture_arrays(D)
    labels, col = strata_labels(system, np.asarray(D["A"].obs_names).astype(str))
    G = len(axis)
    col_axis = np.arange(G)
    print(f"[D5] {system}: subset {len(srcs)} (focal present {present}); strata={'global' if labels is None else col}; {time.time()-t0:.0f}s", flush=True)
    rr, used = real_raw(system, D, B, srcs)
    blank_self(rr, src_axis)
    Pr, Ur = primary_rows(rr, srcs, au)
    sum_real, sp_real = arm_summary(rr, Pr, Ur, srcs, au, src_axis, det)
    # fixed disjoint halves of the SUBSET (same in every arm)
    hp = rng_for(system, 5, 999).permutation(len(srcs))
    halves = (np.sort(hp[:len(srcs) // 2]), np.sort(hp[len(srcs) // 2:]))
    half_P = lambda Pm: [subset_prior(Pm[h], src_axis[h], [srcs[i] for i in h]) for h in halves]
    hr = half_P(Pr)
    rec = {"real": {**sum_real, "disjoint_half_P_spearman": spear(hr[0]["P"], hr[1]["P"]) if all(hr) else None}, "perms": []}
    sel = list(range(a.nperm))
    for rep in sel:
        pf = od / f"{tag}_perm{rep:02d}.npz"
        t1 = time.time()
        perm = stratified_perm(Mu.shape[0], labels, rng_for(system, 5, rep))
        if labels is not None:
            assert (labels[perm] == labels).all()
        raw = raw_rows(Ms, Mu, P30, pairs, src_axis, col_axis, G, perm=perm, x_from=rr)   # X channel is Mu-independent: reused from the real arm
        blank_self(raw, src_axis)
        Pn, Un = primary_rows(raw, srcs, au)
        s_n, sp_n = arm_summary(raw, Pn, Un, srcs, au, src_axis, det)
        hn = half_P(Pn)
        row_rho = [spear(Pn[i], Pr[i]) for i in range(len(srcs))]
        d = {**s_n, "rep": rep, "perm_sha": hashlib.sha256(perm.astype(np.int64).tobytes()).hexdigest()[:16],
             "spearman_nullP_vs_realP": spear(sp_n["P"], sp_real["P"]) if sp_n and sp_real else None,
             "disjoint_half_P_spearman": spear(hn[0]["P"], hn[1]["P"]) if all(hn) else None,
             "per_source_row_spearman_null_vs_real": q3(row_rho), "seconds": time.time() - t1}
        rec["perms"].append(d)
        RU.atomic_npz(pf, P_null=(sp_n["P"] if sp_n else np.full(G, np.nan)), P_real=(sp_real["P"] if sp_real else np.full(G, np.nan)),
                      Prows_null=Pn.astype(np.float32), perm=perm)
        receipt(pf, system, {"rep": rep, "seed": [SEED0, sidx(system), 5, rep], "subset": srcs, "strata": col,
                             "input_sha256": {"fixture": RU.sha(D["fixture"]), "pairs": RU.sha(D["pairs_file"]), "bank": RU.sha(CB.OUT / f"{system}_bank.json"),
                                              **{f.name: RU.sha(f) for f in used}}, "imports_sha256": CB.imports_sha()})
        print(f"[D5] {system} perm {rep}: nullP~realP rho={d['spearman_nullP_vs_realP']} varfrac real={sum_real['subset_P']['varfrac'] if sum_real['subset_P'] else None} "
              f"null={s_n['subset_P']['varfrac'] if s_n['subset_P'] else None} {d['seconds']:.0f}s", flush=True)
    rec["agg"] = {"nullP_vs_realP": q3([p["spearman_nullP_vs_realP"] for p in rec["perms"] if p["spearman_nullP_vs_realP"] is not None]),
                  "disjoint_half": q3([p["disjoint_half_P_spearman"] for p in rec["perms"] if p["disjoint_half_P_spearman"] is not None]),
                  "varfrac_null": q3([p["subset_P"]["varfrac"] for p in rec["perms"] if p["subset_P"]]),
                  "P_vs_detection_null": q3([p["subset_P"]["spearman_P_vs_detection"] for p in rec["perms"] if p["subset_P"]])}
    rec["label"] = ("persistence = target-level structure survives destroyed within-stratum alignment; disappearance = alignment dependence. "
                    "NEITHER is an 'artifact fraction' (the shuffle also destroys target-state geometry and smoothness).")
    rec["meta"] = {"subset_n": len(srcs), "subset_sha256": CB.members_sha(srcs), "focal_present": present, "strata_col": col, "seconds_total": time.time() - t0}
    p = od / f"{tag}_summary.json"
    RU.atomic_json(p, rec)
    receipt(p, system, {"seeds": [SEED0, sidx(system), 5, "rep"], "nperm": a.nperm})
    print(f"[D5] {system}: summary {p}  total {time.time()-t0:.0f}s", flush=True)


# D6
# REFIT the production pipeline on a cell subset: write a half-cell copy of the RAW file, point gb_canonical.RAW at it, call the production GC.build
def refit(system, cell_ids, workdir):
    import anndata as ad
    import gb_canonical as GC
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    src, species = GC.RAW[system]
    raw = ad.read_h5ad(src)
    if cell_ids is not None:
        raw = raw[np.asarray(cell_ids)].copy()
    hp = workdir / f"{system}_rawhalf.h5ad"
    raw.write_h5ad(hp)
    del raw
    old = GC.RAW[system]
    GC.RAW[system] = (str(hp), species)
    try:
        cfg = json.load(open(GC.CONFIG))
        GC.build(str(workdir), system, cfg)
        GC.build_pairs(str(workdir), system, cfg)
    finally:
        GC.RAW[system] = old
    A = ad.read_h5ad(workdir / f"{system}_canonical.h5ad")
    z = np.load(workdir / f"{system}_canonical_pairs.npz", allow_pickle=True)
    pairs = {r: (np.asarray(z[f"{r}_I"], int), np.asarray(z[f"{r}_J"], int)) for r in ROOTS}
    return A, pairs


# production scoring algebra on a refit fixture, frozen gene axis / frozen sources
def score_fixture(A, pairs, srcs, axis, au, src_axis, kmax=KMAX):
    names = np.asarray(A.var_names).astype(str)
    ai = {g: i for i, g in enumerate(axis)}
    keep = [j for j, g in enumerate(names) if g in ai]
    col_axis = np.asarray([ai[names[j]] for j in keep])
    Ms = RU.dense(A.layers["Ms"][:, keep], np.float32)
    Mu = RU.dense(A.layers["Mu"][:, keep], np.float32)
    cpos = {g: i for i, g in enumerate(names[keep])}
    src_col = np.asarray([cpos.get(s, -1) for s in srcs])
    P30 = np.asarray(A.obsm["X_pca"], float)[:, :CB.NPC]
    raw = raw_rows(Ms, Mu, P30, pairs, src_col, col_axis, len(axis), kmax)
    blank_self(raw, src_axis)
    Pm, U = primary_rows(raw, srcs, au)
    return raw, Pm, U, {"n_fixture_genes": len(names), "n_axis_found": len(keep), "n_sources_missing": int((src_col < 0).sum()),
                        "n_cells": int(A.n_obs), "n_pairs": {r: int(len(pairs[r][0])) for r in ROOTS}}


# EVALUATION only (truth read here, after scoring): AUROC of each focal source's PRIMARY row for the frozen response endpoint (resp among resp_ok)
def focal_auroc(system, srcs, axis, Pm):
    res = {}
    try:
        import gb_layer_lattice as LL
        for tf in focal_tfs(system):
            tf = next((x for x in srcs if x.upper() == tf.upper()), None)    # bank symbols keep species case (Rfx6)
            if tf is None:
                continue
            fn = {"GATA1": lambda g: LL.truth_gata1(g), "PANC": lambda g: LL.truth_panc(g),
                  "BONEMARROW": lambda g: LL.truth_bm(tf, g), "K562": lambda g: LL.truth_k562(tf, g)}[system]
            tr = (LL.truth_klf1(axis) if tf == "KLF1" else fn(axis))[0]
            ok = tr["resp_ok"]
            res[tf] = auroc(np.where(ok, Pm[srcs.index(tf)], np.nan), tr["resp"])
    except Exception as e:                                          # never fail the stability diagnostic on the evaluator
        res["_skipped"] = f"{type(e).__name__}: {str(e)[:200]}"
    return res


# greedy size-balanced bipartition of the levels of the donor/batch column -> (idxA, idxB, col) or None
def donor_split(system, obs_names):
    lab, col = strata_labels(system, obs_names)
    if lab is None:
        return None
    lv, ct = np.unique(lab, return_counts=True)
    if len(lv) < 2:
        return None
    A_, B_ = [], []
    sa = sb = 0
    for i in np.argsort(-ct):
        if sa <= sb:
            A_.append(lv[i]); sa += ct[i]
        else:
            B_.append(lv[i]); sb += ct[i]
    return np.flatnonzero(np.isin(lab, A_)), np.flatnonzero(np.isin(lab, B_)), f"{col}: {sorted(A_)} | {sorted(B_)}"


def cmd_d6(a):
    system = a.system.upper()
    t0 = time.time()
    D, B, srcs, present, axis, au, src_axis, det = common_setup(system)
    od = OUT / "D6"
    od.mkdir(parents=True, exist_ok=True)
    tag = f"{system}{'_smalln' if a.smalln else ''}"
    obs = np.asarray(D["A"].obs_names).astype(str)
    n = len(obs)
    rr, used = real_raw(system, D, B, srcs)
    blank_self(rr, src_axis)
    Pr, Ur = primary_rows(rr, srcs, au)
    sp_real = subset_prior(Pr, src_axis, srcs)
    splits = []
    for rep in range(a.nsplit):
        p = rng_for(system, 6, rep).permutation(n)
        splits.append((f"random{rep}", np.sort(p[:n // 2]), np.sort(p[n // 2:])))
    ds = donor_split(system, obs)
    if ds is not None:
        splits.append((f"donor[{ds[2]}]", ds[0], ds[1]))
    else:
        print(f"[D6] {system}: no donor/batch column with >=2 levels -> donor-held-out splits INFEASIBLE", flush=True)
    recs = []
    floor = None
    if not a.nofloor:                                                 # NOISE FLOOR: the same refit pipeline on all cells vs the production rows (DPT/eigensolver nondeterminism, selftest 2)
        ff = od / f"{tag}_refit_floor.json"
        if ff.exists() and not a.force:
            floor = json.load(open(ff))
        else:
            t1 = time.time()
            A, pairs = refit(system, None, Path(a.workdir) / f"{system}_floor")
            raw, Pm, U, info = score_fixture(A, pairs, srcs, axis, au, src_axis)
            sp = subset_prior(Pm, src_axis, srcs)
            floor = {"row_S_spearman_fullrefit_vs_production": q3([spear(Pm[i], Pr[i]) for i in range(len(srcs))]),
                     "subset_P_spearman_fullrefit_vs_production": spear(sp["P"], sp_real["P"]) if sp and sp_real else None,
                     "pair_set_jaccard": {r: (lambda x, y: len(x & y) / len(x | y))(set(zip(*map(lambda v: v.tolist(), pairs[r]))),
                                                                                    set(zip(*map(lambda v: v.tolist(), (np.asarray(D["S"]["pairs"][r][0], int), np.asarray(D["S"]["pairs"][r][1], int)))))) for r in ROOTS},
                     "info": info, "seconds": time.time() - t1}
            RU.atomic_json(ff, floor)
            receipt(ff, system, {"note": "full-cell refit vs production rows = refit noise floor", "subset": srcs})
            print(f"[D6] {system} refit floor: {floor}", flush=True)
    for name, ia, ib, in splits:
        sf = od / f"{tag}_{name.split('[')[0]}.json"
        if sf.exists() and not a.force:
            recs.append(json.load(open(sf)))
            print(f"[D6] reuse {sf}")
            continue
        t1 = time.time()
        arms = {}
        for hn_, ix in (("A", ia), ("B", ib)):
            th = time.time()
            wd = Path(a.workdir) / f"{system}_{name.split('[')[0]}_{hn_}"
            A, pairs = refit(system, obs[ix], wd)
            raw, Pm, U, info = score_fixture(A, pairs, srcs, axis, au, src_axis)
            sp = subset_prior(Pm, src_axis, srcs)
            arms[hn_] = {"P_rows": Pm, "sp": sp, "info": info, "universe_frac": float(np.median(U.mean(1)))}
            arms[hn_]["focal"] = focal_auroc(system, srcs, axis, Pm)
            print(f"[D6] {system} {name} half {hn_}: n={len(ix)} refit+score {time.time()-th:.0f}s info={info}", flush=True)
        rho_ab = [spear(arms["A"]["P_rows"][i], arms["B"]["P_rows"][i]) for i in range(len(srcs))]
        rho_ar = [spear(arms[h]["P_rows"][i], Pr[i]) for h in "AB" for i in range(len(srcs))]
        pa, pb = arms["A"]["sp"], arms["B"]["sp"]
        rec = {"split": name, "n_A": int(len(ia)), "n_B": int(len(ib)),
               "row_S_spearman_A_vs_B": q3(rho_ab), "row_S_spearman_half_vs_full": q3(rho_ar),
               "stability_flag_AvsB": bool(np.isfinite(rho_ab).any() and np.nanmedian(rho_ab) >= .90 and np.nanpercentile(rho_ab, 5) >= .80),
               "subset_P_spearman_A_vs_B": spear(pa["P"], pb["P"]) if pa and pb else None,
               "subset_P_spearman_A_vs_full": spear(pa["P"], sp_real["P"]) if pa and sp_real else None,
               "subset_P_spearman_B_vs_full": spear(pb["P"], sp_real["P"]) if pb and sp_real else None,
               "coverage": {h: {**arms[h]["info"], "universe_frac_median": arms[h]["universe_frac"],
                                "P_finite_frac_median": float(np.median(np.isfinite(arms[h]["P_rows"]).mean(1))),
                                "n_sources_with_P": int(np.isfinite(arms[h]["P_rows"]).any(1).sum()),
                                "subset_P_nT": arms[h]["sp"]["nT"] if arms[h]["sp"] else None,
                                "subset_P_dropped_rows": arms[h]["sp"]["dropped"] if arms[h]["sp"] else None} for h in "AB"},
               "coverage_full_real": {"universe_frac_median": float(np.median(Ur.mean(1))), "P_finite_frac_median": float(np.median(np.isfinite(Pr).mean(1))),
                                      "subset_P_nT": sp_real["nT"] if sp_real else None},
               "focal_AUROC_response": {h: arms[h]["focal"] for h in "AB"},
               "criteria": "median rho>=.90 AND 5th pct>=.80 (D1 rank criteria) = operational stability flag; random halves = subsampling stability, NOT donor generalisation",
               "seconds": time.time() - t1}
        RU.atomic_json(sf, rec)
        np.savez_compressed(sf.with_suffix(".npz"), PA=arms["A"]["P_rows"].astype(np.float32), PB=arms["B"]["P_rows"].astype(np.float32), ia=ia, ib=ib)
        receipt(sf, system, {"split": name, "seed": [SEED0, sidx(system), 6, "rep"], "subset": srcs, "truth_read": True,
                             "truth_note": "truth read only by the post-scoring focal AUROC readout", "imports_sha256": CB.imports_sha(),
                             "input_sha256": {"fixture": RU.sha(D["fixture"]), "pairs": RU.sha(D["pairs_file"]), "bank": RU.sha(CB.OUT / f"{system}_bank.json"),
                                              "raw": RU.sha(Path(__import__("gb_canonical").RAW[system][0])), **{f.name: RU.sha(f) for f in used}},
                             "config": RU.sha(Path(__import__("gb_canonical").CONFIG))})
        recs.append(rec)
        print(f"[D6] {system} {name}: row-S A~B {rec['row_S_spearman_A_vs_B']} P A~B {rec['subset_P_spearman_A_vs_B']} flag={rec['stability_flag_AvsB']} {rec['seconds']:.0f}s", flush=True)
    summ = od / f"{tag}_summary.json"
    RU.atomic_json(summ, {"system": system, "subset_n": len(srcs), "focal_present": present, "refit_floor": floor, "splits": recs, "seconds_total": time.time() - t0,
                          "label": "random halves = subsampling stability, not donor generalisation; donor-held-out only where a donor/batch column exists"})
    print(f"[D6] {system}: {summ} total {time.time()-t0:.0f}s", flush=True)


# selftest
def cmd_selftest(a):
    system = a.system.upper()
    t0 = time.time()
    D, B, srcs, present, axis, au, src_axis, det = common_setup(system)
    assert srcs == sorted(srcs) and len(set(srcs)) == len(srcs)
    s2, _ = pick_subset(system, B["members"])
    assert s2 == srcs, "subset not reproducible"
    foc = {f.upper() for f in focal_tfs(system)}
    assert not any(s.upper() in foc for s in srcs if s not in present)
    print(f"[selftest] subset n={len(srcs)} focal_present={present} sha={CB.members_sha(srcs)[:12]}")
    # permutation invariants
    lab = np.array(list("aabbbcc"))
    p = stratified_perm(7, lab, np.random.default_rng(0))
    assert sorted(p) == list(range(7)) and (lab[p] == lab).all()
    p = stratified_perm(50, None, rng_for(system, 5, 0))
    assert sorted(p) == list(range(50)) and not (p == np.arange(50)).all()
    # (1) unpermuted arm == catbank raw
    Ms, Mu, P30, pairs = full_fixture_arrays(D)
    G = len(axis)
    nh = min(a.n, len(srcs))
    head = list(range(nh))
    rr, _ = real_raw(system, D, B, [srcs[i] for i in head])
    t1 = time.time()
    raw = raw_rows(Ms, Mu, P30, pairs, src_axis[head], np.arange(G), G)
    print(f"[selftest] raw_rows({nh} sources, 2 roots, G={G}, n_pairs={[len(pairs[r][0]) for r in ROOTS]}) {time.time()-t1:.0f}s", flush=True)
    worst = 0.0
    for k in rr:
        x, y = raw[k], rr[k]
        for s in head:                                               # shards blank the self target; raw_rows does not -> compare after blanking
            x[s, src_axis[s], :] = np.nan
        fin = np.isfinite(x) & np.isfinite(y)
        assert np.array_equal(np.isfinite(x), np.isfinite(y)), f"finiteness differs {k}"
        worst = max(worst, float(np.abs(x[fin] - y[fin]).max()))
    assert worst <= 1e-5, f"unpermuted arm differs from catbank raw |S|: {worst:.2e}"
    print(f"[selftest] (1) unpermuted arm reproduces catbank raw |S| (U and X, both roots, k0..30, {nh} sources): max|diff|={worst:.2e}  PASS  {time.time()-t0:.0f}s")
    # U-only path (used by D5) == full path == catbank, and a permuted arm differs on U while X is
    # carried over unchanged
    t1 = time.time()
    raw_u = raw_rows(Ms, Mu, P30, pairs, src_axis[head], np.arange(G), G, x_from=rr)
    wu = 0.0
    for k in ((h, "U") for h in ROOTS):
        for s_ in head:
            raw_u[k][s_, src_axis[s_], :] = np.nan
        fin = np.isfinite(raw_u[k]) & np.isfinite(rr[k])
        assert np.array_equal(np.isfinite(raw_u[k]), np.isfinite(rr[k]))
        wu = max(wu, float(np.abs(raw_u[k][fin] - rr[k][fin]).max()))
    assert wu <= 1e-5, f"U-only path differs from catbank: {wu:.2e}"
    print(f"[selftest] (1b) U-only path (D5) reproduces catbank raw U: max|diff|={wu:.2e} PASS {time.time()-t1:.0f}s", flush=True)
    perm = rng_for(system, 5, 0).permutation(Ms.shape[0])
    raw_p = raw_rows(Ms, Mu, P30, pairs, src_axis[head[:2]], np.arange(G), G, perm=perm, x_from={k: v[:2] for k, v in rr.items()})
    assert np.nanmax(np.abs(raw_p[(ROOTS[0], "U")][0, :, 0] - raw[(ROOTS[0], "U")][0, :, 0])) > 1e-3
    print("[selftest] permutation changes the U channel  PASS", flush=True)
    # (2) refit on the full cell set
    if a.skip_refit:
        return
    A, prs = refit(system, None, Path(a.workdir) / f"{system}_selftest_full")
    names = np.asarray(A.var_names).astype(str)
    fg = np.asarray(D["A"].var_names).astype(str)
    rep = {"n_genes_refit": len(names), "n_genes_prod": len(fg), "genes_equal": bool(np.array_equal(names, fg)),
           "obs_equal": bool(np.array_equal(np.asarray(A.obs_names).astype(str), np.asarray(D["A"].obs_names).astype(str)))}
    com = [g for g in fg if g in set(names)]
    ia_, ib_ = [list(names).index(g) for g in com], [list(fg).index(g) for g in com]
    for lay in ("Ms", "Mu", "velocity"):
        x, y = RU.dense(A.layers[lay][:, ia_], np.float64), RU.dense(D["A"].layers[lay][:, ib_], np.float64)
        rep[f"max_abs_diff_{lay}"] = float(np.nanmax(np.abs(x - y))) if x.shape == y.shape else "shape"
    rep["pca_max_abs_diff"] = float(np.abs(np.asarray(A.obsm["X_pca"])[:, :30] - np.asarray(D["A"].obsm["X_pca"])[:, :30]).max())
    rep["pairs_equal"] = {r: bool(np.array_equal(prs[r][0], pairs[r][0]) and np.array_equal(prs[r][1], pairs[r][1])) for r in ROOTS}
    rep["n_pairs"] = {r: [int(len(prs[r][0])), int(len(pairs[r][0]))] for r in ROOTS}
    print("[selftest] (2) full-set refit vs production:", json.dumps(rep))
    exact = rep["genes_equal"] and rep["obs_equal"] and all(rep["pairs_equal"].values())
    raw2, Pm2, U2, info = score_fixture(A, prs, srcs[:nh], axis, au, src_axis[:nh])
    w2 = 0.0
    for k in rr:
        fin = np.isfinite(raw2[k]) & np.isfinite(rr[k])
        w2 = max(w2, float(np.abs(raw2[k][fin] - rr[k][fin]).max()) if fin.any() else 0.0)
    fe = float((np.isfinite(raw2[(ROOTS[0], "U")]) != np.isfinite(rr[(ROOTS[0], "U")])).mean())
    print(f"[selftest] (2b) raw |S| from full-set refit vs catbank: max|diff|={w2:.2e}, finiteness mismatch frac={fe:.4f}; exact_inputs={exact}")
    if exact and w2 <= 1e-4:
        print("[selftest] (2) refit pipeline on FULL cell set REPRODUCES production pairs and raw scores  PASS")
    else:
        print("[selftest] (2) refit does NOT reproduce production bit-for-bit; see the diffs above (document the cause)  NOT-EXACT")
    print(f"[selftest] done {time.time()-t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["d5", "d6", "selftest"])
    ap.add_argument("system")
    ap.add_argument("--nperm", type=int, default=20)
    ap.add_argument("--nsplit", type=int, default=5)
    ap.add_argument("--n", type=int, default=8, help="selftest: number of subset sources to recompute")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--smalln", action="store_true", help="tag outputs _smalln (pilot)")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-refit", action="store_true")
    ap.add_argument("--nofloor", action="store_true")
    ap.add_argument("--workdir", default="/tmp/gb_defense_work")
    a = ap.parse_args()
    {"d5": cmd_d5, "d6": cmd_d6, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    main()
