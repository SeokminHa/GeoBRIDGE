#!/usr/bin/env python
# Regulator-matching null and incremental-prediction checks.
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_cop_alignment as CA
import gb_primary as GP

ROOTS = GP.ROOTS
LEGACY_BANK_PRODUCERS = {"3c1ccd8218bff8c147fd733162e557034086d2d79c5ee1fd5c7f4d36fe01ef8d"}
REPS = {"RAW-z": (0,), "PRIMARY": tuple(range(9)), "K8-diag": (8,)}
PAIRED = (("PRIMARY", "RAW-z"), ("PRIMARY", "RANDOM-8"), ("SAME-only", "CROSS-only"))
POOL_TARGET, MIN_STRATUM, NBIN = 50, 20, 10
_init = CA.Endpoint.__init__


def _init_keep(self, name, family, system, source, raw_h, *a, **k):
    _init(self, name, family, system, source, raw_h, *a, **k)
    self.RAWH = {h: np.asarray(raw_h[h], float) for h in raw_h}


CA.Endpoint.__init__ = _init_keep


# metrics / statistics (pure; selftested)
def fixed_bins(strat, nbin=10):
    from scipy.stats import rankdata
    return np.minimum((rankdata(strat, method="ordinal") - 1) * nbin // len(strat), nbin - 1)


def metric(v, y, q):
    if q is None:
        return CA.auroc(v, y)
    num = w = 0.0
    for b in np.unique(q):
        m = q == b
        p = int((m & y).sum())
        if p and (m & ~y).sum():
            num += p * CA.auroc(v[m], y[m])
            w += p
    return num / w if w else np.nan


CALIPER = 2.5


# k-means strata (K = round(n/50), reduced until every cluster >= MIN_STRATUM), then the caliper
def strata(F, seed=0, pool_target=None):
    from sklearn.cluster import KMeans
    K = max(1, int(round(len(F) / (pool_target or POOL_TARGET))))
    while True:
        lab = KMeans(K, n_init=20, random_state=seed).fit_predict(F) if K > 1 else np.zeros(len(F), int)
        if K == 1 or np.bincount(lab).min() >= MIN_STRATUM:
            break
        K -= 1
    cen = np.vstack([F[lab == c].mean(0) for c in range(lab.max() + 1)])
    d = np.linalg.norm(F - cen[lab], axis=1)
    out = lab.copy()
    out[d > CALIPER * np.median(d)] = -1
    for c in np.unique(out[out >= 0]):
        if (out == c).sum() < MIN_STRATUM:
            out[out == c] = -1
    return out, d


# Candidate matcher: deterministic greedy blocks of `size` nearest unassigned sources, seeded from the unassigned source farthest from the bank centroid
def nn_blocks(F, size=20):
    n = len(F)
    lab = np.full(n, -1)
    free = np.ones(n, bool)
    c0 = F.mean(0)
    b = 0
    while free.sum() >= size:
        idx = np.flatnonzero(free)
        seed = idx[np.argmax(np.linalg.norm(F[idx] - c0, axis=1))]
        near = idx[np.argsort(np.linalg.norm(F[idx] - F[seed], axis=1), kind="stable")[:size]]
        lab[near] = b
        free[near] = False
        b += 1
    if free.any():
        cen = np.vstack([F[lab == c].mean(0) for c in range(b)])
        for i in np.flatnonzero(free):
            lab[i] = int(np.argmin(np.linalg.norm(cen - F[i], axis=1)))
    cen = np.vstack([F[lab == c].mean(0) for c in range(b)])
    return lab, np.linalg.norm(F - cen[lab], axis=1)


# One set of within-stratum permutations per family ({stratum: (n_perm, size) positions})
def perm_maps(sizes, n_perm, rng):
    return {c: np.argsort(rng.random((n_perm, n)), axis=1) for c, n in sizes.items()}


# M[r]: metric array over the members of r's stratum (member order fixed)
def family_stats(M, pos, strat_of, maps):
    focal = list(M)
    T = float(np.median([M[r][pos[r]] for r in focal]))
    null = np.median(np.vstack([M[r][maps[strat_of[r]][:, pos[r]]] for r in focal]), axis=0)
    return T, null


def pval(null, obs):
    return float((1 + (null >= obs).sum()) / (1 + len(null)))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj, run = np.empty_like(p), 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


# ΔAUROC of 5-fold logistic (X1 vs X0); bootstrap refits both models, folds grouped by original gene id
def kt4_refit(X0, X1, y, B, rng):
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import StandardScaler

    def oof_delta(idx):
        yy, grp, pr = y[idx], idx, {}
        for lab, X in (("0", X0[idx]), ("1", X1[idx])):
            p = np.zeros(len(idx))
            for tr, te in StratifiedGroupKFold(5, shuffle=True, random_state=0).split(X, yy, grp):
                sc = StandardScaler().fit(X[tr])
                p[te] = LogisticRegression(max_iter=1000).fit(sc.transform(X[tr]), yy[tr]).predict_proba(sc.transform(X[te]))[:, 1]
            pr[lab] = p
        return CA.auroc(pr["1"], yy) - CA.auroc(pr["0"], yy), CA.auroc(pr["0"], yy), CA.auroc(pr["1"], yy)

    obs = oof_delta(np.arange(len(y)))
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    bs = [oof_delta(np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))]))[0] for _ in range(B)]
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return dict(delta=obs[0], auroc_base=obs[1], auroc_base_plus=obs[2], lo=lo, hi=hi)


# Sources in blobs (co-regulatory modules)
def _simulate(rng, sizes, overlap, beta, G=400, reg=30, mod=60, focal_frac=.10, gamma=1.0, sep=4.0):
    n = sum(sizes)
    blob = np.repeat(np.arange(len(sizes)), sizes)
    F = rng.normal(scale=sep, size=(len(sizes), 2 + NBIN))[blob] + rng.normal(size=(n, 2 + NBIN))
    t = rng.normal(size=G)
    hub = np.exp(t) / np.exp(t).sum()                               # hub-biased gene choice (target prior)
    modg = [rng.choice(G, mod, replace=False, p=hub) for _ in sizes]
    n_ov = int(round(overlap * reg))
    S = t + rng.normal(size=(n, G))
    for s_ in range(n):
        S[s_, modg[blob[s_]]] += gamma
    focal, regul = [], {}
    for b, sz in enumerate(sizes):
        members = np.flatnonzero(blob == b)
        for r in members[: max(1, int(round(focal_frac * sz)))]:
            rest = np.setdiff1d(np.arange(G), modg[b])
            uniq = rng.choice(rest, reg - n_ov, replace=False, p=hub[rest] / hub[rest].sum())
            S[r, uniq] += beta
            y = np.zeros(G, bool)
            y[rng.choice(modg[b], n_ov, replace=False)] = True
            y[uniq] = True
            focal.append(r)
            regul[r] = y
    S[rng.random(S.shape) < .005] = np.nan                          # asymmetric missingness
    return F, S, focal, regul


def _one_system(args_):
    seed, sizes, overlap, beta, n_perm = args_[:5]
    pool_target = args_[5] if len(args_) > 5 else None
    matcher = args_[6] if len(args_) > 6 else "kmeans"
    rng = np.random.default_rng(seed)
    F, S, focal, regul = _simulate(rng, sizes, overlap, beta)
    absent = focal[-1]
    Fz = (F - F.mean(0)) / F.std(0)
    lab, _ = strata(Fz, pool_target=pool_target) if matcher == "kmeans" else nn_blocks(Fz)
    lab[absent] = -2
    M, pos, sof, status, incr = {}, {}, {}, {}, []
    for r in focal:
        if lab[r] == -2:
            status[r] = "NO_CALL_ABSENT"
            continue
        if lab[r] < 0:
            status[r] = "NO_CALL_UNMATCHED"
            continue
        mem = list(np.flatnonzero(lab == lab[r]))
        rows = np.isfinite(S[mem]).all(0)
        assert all(np.isfinite(S[m][rows]).all() for m in mem)     # identical rows for focal and every pseudo
        y = regul[r][rows]
        M[r] = np.array([metric(S[m][rows], y, None) for m in mem])
        pos[r], sof[r], status[r] = mem.index(r), int(lab[r]), "OK"
        incr.append(M[r][pos[r]] - np.delete(M[r], pos[r]).mean())
    assert len(status) == len(focal) and status[absent] == "NO_CALL_ABSENT"
    sizes_ = {c: int((lab == c).sum()) for c in set(sof.values())}
    maps = perm_maps(sizes_, n_perm, rng)
    T, null = family_stats(M, pos, sof, maps)
    return pval(null, T), float(np.median(incr)), sum(v == "NO_CALL_UNMATCHED" for v in status.values())


REGIMES = {"S20": [20] * 24, "S50": [50] * 10, "S150": [150] * 3, "S235": [235] * 2, "mixed": [20, 50, 150, 235]}
OVERLAPS = (0.0, .25, .50, .75)


# PASS iff type-I <= .05 at beta=0 in every regime and power >= .80 at a +.03 matched-AUROC increment in every regime
def falsifier(n_null=1000, n_power=200, n_perm=500, target=.03, procs=40, pool_target=None, regimes=None,
              matcher="kmeans"):
    from multiprocessing import Pool
    rows = []
    with Pool(procs) as pool:
        for rg, sizes in (REGIMES if regimes is None else {k: REGIMES[k] for k in regimes}).items():
            for ov in OVERLAPS:
                lo, hi = 0.0, 3.0                                   # calibrate beta (separate seeds 10^6+)
                for _ in range(12):
                    mid = (lo + hi) / 2
                    inc = np.median([r[1] for r in pool.map(_one_system, [(10**6 + i, sizes, ov, mid, 50, pool_target, matcher)
                                                                          for i in range(40)])])
                    lo, hi = (mid, hi) if inc < target else (lo, mid)
                beta = (lo + hi) / 2
                p0 = [r[0] for r in pool.map(_one_system, [(i, sizes, ov, 0.0, n_perm, pool_target, matcher)
                                                           for i in range(n_null)])]
                p1 = pool.map(_one_system, [(5 * 10**6 + i, sizes, ov, beta, n_perm, pool_target, matcher)
                                            for i in range(n_power)])
                rows.append(dict(regime=rg, overlap=ov, beta=beta, typeI=float(np.mean(np.asarray(p0) <= .05)),
                                 power=float(np.mean([r[0] <= .05 for r in p1])),
                                 realised_increment=float(np.median([r[1] for r in p1])),
                                 unmatched_median=float(np.median([r[2] for r in p1]))))
                print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    rng = np.random.default_rng(3)
    y = rng.random(400) < .3
    X0 = rng.normal(size=(400, 3))
    k4 = kt4_refit(X0, np.column_stack([X0, rng.normal(size=400)]), y, 50, rng)
    ok = bool((T.typeI <= .05).all() and (T.power >= .80).all() and k4["lo"] <= 0 <= k4["hi"])
    print(T.round(3).to_string(index=False), "\nKT4 noise:", {k: round(v, 3) for k, v in k4.items()},
          "\nFALSIFIER", "PASS" if ok else "FAIL", flush=True)
    return T, ok


# real data
def load_bank(system, eg, rotation=None, subset=None):
    import gb_real_uniform as RU
    bpath = RU.system_dir(system) / ("catalogue_U_k0_8.npz" if rotation is None and subset is None else
                                     f"catalogue_U_k0_8_rot{rotation}.npz" if subset is None else
                                     f"catalogue_U_k0_8_sub{subset}.npz")
    accepted = ({RU.sha(RU.SCRATCH / "gb_pairsubset_scores.py")} if subset else
                {RU.sha(RU.SCRATCH / "gb_catalogue_scores.py")} | (LEGACY_BANK_PRODUCERS if rotation is None else set()))
    RU.check_artifact(bpath, accepted_producer_sha256=accepted)
    Zb = np.load(bpath, allow_pickle=True)
    if list(np.asarray(Zb["genes"]).astype(str)) != list(eg):
        raise SystemExit(f"REFUSING_BANK_AXIS: {system}")
    return [s.upper() for s in np.asarray(Zb["sources"]).astype(str)], {h: np.asarray(Zb[f"{h}|U"]) for h in ROOTS}


def zcons_bank(Ub, ks):
    n, G = Ub[ROOTS[0]].shape[:2]
    out = np.full((n, G), np.nan)
    for j in range(n):
        r = {h: Ub[h][j].astype(float) for h in ROOTS}
        m = np.isfinite(r[ROOTS[0]]).all(1) & np.isfinite(r[ROOTS[1]]).all(1)
        out[j, m] = GP.zcons({h: r[h][m] for h in ROOTS}, ks)
    return out


# {rep: (n_bank, G)} for RAW-z, PRIMARY, K8-diag; one shared bank order
def bank_reps(system, eg, with_random=False, with_subsets=False):
    import gb_real_uniform as RU
    bank, Ub = load_bank(system, eg)
    reps = {k: zcons_bank(Ub, ks) for k, ks in REPS.items()}
    for sub, lab in (("same", "SAME-only"), ("cross", "CROSS-only")):
        if with_subsets:
            bank_s, Us = load_bank(system, eg, subset=sub)
            if bank_s != bank:
                raise SystemExit(f"REFUSING_SUBSET_BANK_ORDER: {system}")
            reps[lab] = zcons_bank(Us, tuple(range(9)))
    if with_random:
        bank_r, Ur = load_bank(system, eg, rotation=0)
        if bank_r != bank:
            raise SystemExit(f"REFUSING_RANDOM8_BANK_ORDER: {system}")
        reps["RANDOM-8"] = zcons_bank(Ur, tuple(range(9)))
    return bank, reps


def system_context(system, with_random=False, with_subsets=False, pool_target=None, matcher="kmeans"):
    import anndata as ad
    import gb_real_uniform as RU
    import gb_substrate as GS

    t0 = time.time()
    rows, eg = RU.gb_raw_rows(system)
    eg = np.asarray(eg).astype(str)
    G = len(eg)
    gi = {g.upper(): i for i, g in enumerate(eg)}
    bank, reps = bank_reps(system, eg, with_random, with_subsets)
    BP = reps["PRIMARY"]
    bi = {s: j for j, s in enumerate(bank)}
    PZ = {}
    for s in rows:
        m, _ = CA.observational_universe(s, rows[s], eg)
        v = np.full(G, np.nan)
        v[m] = GP.zcons({h: rows[s][(h, "U")][m] for h in ROOTS})
        PZ[s] = v
    Zr = np.vstack([PZ[s] for s in sorted(PZ)])
    fx, pairs_file, _ = GS.paths(system, canonical=True)
    A = ad.read_h5ad(fx)
    fm = {g: i for i, g in enumerate(np.asarray(A.var_names).astype(str))}
    cols = np.asarray([fm[g] for g in eg])
    dn = lambda X: X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    Ms = dn(A.layers["Ms"])[:, cols].astype(np.float64)
    Mu = dn(A.layers["Mu"])[:, cols].astype(np.float64)
    det_x = (dn(A.X)[:, cols] > 0).mean(0)
    det_s = (dn(A.layers["spliced"])[:, cols] > 0).mean(0)
    mms = Ms.mean(0)
    pt = np.asarray(np.load(pairs_file, allow_pickle=True)["dc1_min_pt"], float)
    okc = np.flatnonzero(np.isfinite(pt))
    binid = np.empty(len(pt), int)
    binid[okc[np.argsort(pt[okc], kind="stable")]] = np.arange(len(okc)) * NBIN // len(okc)
    bj = np.asarray([gi[s] for s in bank])
    prof = np.vstack([Ms[okc][binid[okc] == b][:, bj].mean(0) for b in range(NBIN)]).T      # (bank, NBIN)
    prof = (prof - prof.mean(1, keepdims=True)) / np.where(prof.std(1, keepdims=True) > 0, prof.std(1, keepdims=True), 1)
    F = np.column_stack([np.log1p(mms[bj]), det_s[bj], prof])
    F = (F - F.mean(0)) / np.where(F.std(0) > 0, F.std(0), 1.0)
    lab, sdist = strata(F, pool_target=pool_target) if matcher == "kmeans" else nn_blocks(F)
    P = np.asarray(A.obsm["X_pca"], float)
    if P.shape[1] < 8 or not np.isfinite(P[:, :8]).all():
        raise SystemExit(f"REFUSING_PCS: {system}")
    focal = sorted(rows)  # PC-partial is only needed for endpoint sources (all roster)
    fidx = np.asarray([gi[s] for s in focal])
    C = np.empty((9, len(focal), G), np.float32)
    for k in range(9):
        Bm = np.column_stack([np.ones(len(P)), P[:, :k]])
        res = lambda Y: Y - Bm @ np.linalg.lstsq(Bm, Y, rcond=None)[0]
        Ur, Sr = res(Mu), res(Ms[:, fidx])
        Ur /= np.where(np.linalg.norm(Ur, axis=0) > 0, np.linalg.norm(Ur, axis=0), np.nan)
        Sr /= np.where(np.linalg.norm(Sr, axis=0) > 0, np.linalg.norm(Sr, axis=0), np.nan)
        C[k] = np.abs(Sr.T @ Ur)
    PC8, PCZ = {}, {}
    for i, s in enumerate(focal):
        c = C[:, i, :].T.astype(float)
        c[gi[s]] = np.nan
        PC8[s] = c[:, 8]
        v = np.full(G, np.nan)
        m = np.isfinite(c).all(1)
        v[m] = GP.zcons({h: c[m] for h in ROOTS})
        PCZ[s] = v
    kt5 = {}
    k5 = RU.system_dir(system) / "kt5_shuffledpt_U_k0_8.npz"
    if k5.exists():
        RU.check_artifact(k5, accepted_producer_sha256={RU.sha(RU.SCRATCH / "gb_kt5.py")})
        Z5 = np.load(k5, allow_pickle=True)
        if list(np.asarray(Z5["genes"]).astype(str)) != list(eg):
            raise SystemExit(f"REFUSING_KT5_AXIS: {system}")
        U5 = {h: np.asarray(Z5[f"{h}|U"]) for h in ROOTS}
        for j, s in enumerate(np.asarray(Z5["sources"]).astype(str)):
            r = {h: U5[h][j].astype(float) for h in ROOTS}
            m = np.isfinite(r[ROOTS[0]]).all(1) & np.isfinite(r[ROOTS[1]]).all(1)
            v = np.full(G, np.nan)
            v[m] = GP.zcons({h: r[h][m] for h in ROOTS})
            kt5[s.upper()] = v
    sz = np.bincount(lab[lab >= 0])
    sz = sz[sz > 0]
    print(f"[ctx] {system}: bank {len(bank)} in {len(sz)} strata (sizes {sz.min()}-{sz.max()}), unmatched "
          f"{int((lab < 0).sum())}, roster {len(rows)}, {time.time()-t0:.0f}s", flush=True)
    return dict(eg=eg, gi={g: i for i, g in enumerate(eg)}, G=G, bank=bank, bi=bi, BP=BP, lab=lab, sdist=sdist, PZ=PZ,
                reps=reps, kt5=kt5,
                rep_sum={k: np.nansum(v, 0) for k, v in reps.items()}, rep_cnt={k: np.isfinite(v).sum(0) for k, v in reps.items()},
                rsum=np.nansum(Zr, 0), rcnt=np.isfinite(Zr).sum(0), bsum=np.nansum(BP, 0), bcnt=np.isfinite(BP).sum(0),
                det=det_x, mms=mms, PC8=PC8, PCZ=PCZ)


def loo_mean(v, tot, cnt, n):
    s = tot - np.nan_to_num(v)
    c = cnt - np.isfinite(v)
    out = np.full(len(s), np.nan)
    ok = c >= max(3, 0.5 * (n - 1))
    out[ok] = s[ok] / c[ok]
    return out


def run(args):
    import gb_real_uniform as RU

    rng = np.random.default_rng(20260926)
    out = RU.OUT / args.out
    out.mkdir(parents=True, exist_ok=True)
    EP, KT2E, BOOTF1, BOOTF2, KT4, FAM, PAIR, STRATA = [], [], [], [], [], [], [], []
    for system in [s.upper() for s in args.systems]:
        ctx = system_context(system, with_random=args.before_after and not args.t7, with_subsets=args.t7,
                             pool_target=args.pool_target, matcher=args.matcher)
        reps_used = (("PRIMARY", "SAME-only", "CROSS-only") if args.t7 else
                     ("RAW-z", "PRIMARY", "K8-diag", "RANDOM-8") if args.before_after else ("PRIMARY",))
        pear = RU.load_matrix(RU.system_dir(system) / "pearson.npz", "score")
        RU.check_artifact(RU.system_dir(system) / "pearson.npz")
        famM, famP, famS, declared = {}, {}, {}, {}
        store = []
        for name, family, _s, src, eps, n, npos in RU.trace_endpoints(systems=[system]):
            src = src.upper()
            E, EX = eps["U"], eps["X"]
            ix = np.asarray([ctx["gi"][g] for g in E.obs_genes])
            V = {"GeoBRIDGE-PRIMARY": GP.zcons(E.RAWH), "PairStatic-PRIMARY": GP.zcons(EX.RAWH),
                 "GeoBRIDGE-RAW": np.minimum(*[E.P[h][:, 0] for h in ROOTS]),
                 "Pearson": RU.static_vector(pear, src, E.obs_genes)}
            if not np.allclose(V["GeoBRIDGE-PRIMARY"], ctx["PZ"][src][ix], equal_nan=True):
                raise SystemExit(f"REFUSING_PRIMARY_PARITY: {name}")
            infocus = src in ctx["bi"]
            V["KT1-catalogue"] = (loo_mean(ctx["BP"][ctx["bi"][src]], ctx["bsum"], ctx["bcnt"], len(ctx["bank"]))[ix]
                                  if infocus else np.full(len(ix), np.nan))
            V["KT1-roster(stress)"] = loo_mean(ctx["PZ"][src], ctx["rsum"], ctx["rcnt"], len(ctx["PZ"]))[ix]
            V["KT3-PCpartial-8PC"], V["KT3-PCpartial-Z0:8"] = ctx["PC8"][src][ix], ctx["PCZ"][src][ix]
            V["detection"], V["meanMs"] = ctx["det"][ix], ctx["mms"][ix]
            if src in ctx["kt5"]:
                V["KT5-shuffledPT"] = ctx["kt5"][src][ix]
            common = E.eval_mask.copy()
            for m, v in V.items():
                if m != "KT5-shuffledPT" and np.isfinite(v[E.eval_mask]).any():
                    common &= np.isfinite(v)
            rc = common[E.eval_mask]
            y = np.asarray(E.y, bool)[rc]
            q = fixed_bins(np.asarray(E.strat)[rc]) if E.strat is not None else None
            res = {m: (metric(v[common], y, q) if np.isfinite(v[common]).all() else np.nan) for m, v in V.items()}
            for m, val in res.items():
                EP.append(dict(endpoint=name, family=family, system=system, source=src, method=m, metric=val,
                               n_rows=int(common.sum()), n_pos=int(y.sum())))
            keep = dict(name=name, family=family, y=y, q=q, gidx=ix[common], V={m: v[common] for m, v in V.items()
                                                                                if np.isfinite(v[common]).all()})
            fk = ("F1|" + system) if family == "F1" else ("F2|K562" if system == "K562" else
                                                           "F2|BM" if system == "BONEMARROW" else None)
            status = ("NO_CALL_ABSENT" if not infocus else
                      "NO_CALL_UNMATCHED" if ctx["lab"][ctx["bi"][src]] < 0 else "OK")
            if fk:
                declared.setdefault(fk, []).append((src, status))
            if status != "OK":
                KT2E.append(dict(endpoint=name, family=family, system=system, source=src, rep="PRIMARY", status=status,
                                 p_endpoint=1.0, stratum=-1, stratum_size=0))
            else:  # KT2 (+ T3/T7 representations): focal + matched stratum on identical rows
                c = int(ctx["lab"][ctx["bi"][src]])
                mem = [ctx["bank"][j] for j in np.flatnonzero(ctx["lab"] == c)]
                R = {k: ctx["reps"][k] for k in reps_used}
                vecs = {k: dict(prior=loo_mean(R[k][ctx["bi"][src]], ctx["rep_sum"][k], ctx["rep_cnt"][k],
                                               len(ctx["bank"]))[ix],
                                mem=np.vstack([R[k][ctx["bi"][s2]][ix] for s2 in mem])) for k in R}
                rows3 = common.copy()
                for d in vecs.values():
                    rows3 &= np.isfinite(d["prior"]) & np.isfinite(d["mem"]).all(0)
                r3 = rows3[E.eval_mask]
                y3 = np.asarray(E.y, bool)[r3]
                q3 = fixed_bins(np.asarray(E.strat)[r3]) if E.strat is not None else None
                pos = mem.index(src)
                for k, d in vecs.items():
                    Mr = np.array([metric(d["mem"][i][rows3], y3, q3) for i in range(len(mem))])
                    others = np.delete(Mr, pos)
                    KT2E.append(dict(endpoint=name, family=family, system=system, source=src, rep=k, status="OK",
                                     stratum=c, stratum_size=len(mem), n_rows=int(rows3.sum()), focal=Mr[pos],
                                     prior=metric(d["prior"][rows3], y3, q3),
                                     kt1_delta=Mr[pos] - metric(d["prior"][rows3], y3, q3),
                                     pool_median=float(np.median(others)),
                                     focal_percentile=float(np.mean(others < Mr[pos])),
                                     p_endpoint=float((1 + (others >= Mr[pos]).sum()) / len(mem))))
                    if fk:
                        famM.setdefault(fk, {}).setdefault(k, {})[src] = Mr
                        famP.setdefault(fk, {})[src] = pos
                        famS.setdefault(fk, {})[src] = c
            store.append(keep)
            if name in ("F2|GATA1|chronicKO", "F2|PANC|RFX6", "F3|GATA1|V1_fourclass") and args.kt4:
                k = keep["V"]
                X0 = np.column_stack([k["Pearson"], k["detection"], np.log1p(k["meanMs"])])
                KT4.append(dict(endpoint=name, n=len(y), n_pos=int(y.sum()),
                                **kt4_refit(X0, np.column_stack([X0, k["GeoBRIDGE-PRIMARY"]]), y, args.boot, rng)))
        for fk, byrep in famM.items():  # one permutation-map set per family, reused for every representation
            sizes = {c: int((ctx["lab"] == c).sum()) for c in set(famS[fk].values())}
            maps = perm_maps(sizes, args.perm, rng)
            stats = {k: family_stats(M, famP[fk], famS[fk], maps) for k, M in byrep.items()}
            n_decl = len(declared.get(fk, []))
            for k, (T, null) in stats.items():
                FAM.append(dict(family=fk, rep=k, n_declared=n_decl, n_evaluable=len(byrep[k]), T_median_focal=T,
                                null_median=float(np.median(null)), T_minus_null=T - float(np.median(null)),
                                null_q95=float(np.quantile(null, .95)), p_shared_perm=pval(null, T), n_perm=args.perm))
            for a, b in PAIRED:
                if a in stats and b in stats:
                    (Ta, na), (Tb, nb) = stats[a], stats[b]
                    PAIR.append(dict(family=fk, contrast=f"{a} - {b}", delta_obs=Ta - Tb,
                                     delta_null_median=float(np.median(na - nb)), p_paired=pval(na - nb, Ta - Tb),
                                     n_perm=args.perm))
        STRATA.extend(dict(system=system, source=b_, stratum=int(l_), dist=float(d_)) for b_, l_, d_ in
                      zip(ctx["bank"], ctx["lab"], ctx["sdist"]))
        BOOTF1.extend(boot_f1([k for k in store if k["family"] == "F1"], system, ctx["G"], rng, args.boot))
        BOOTF2.extend(r for k in store if k["family"] != "F1" for r in boot_ep(k, system, rng, args.boot))
        print(f"[kt] {system} done", flush=True)
    K2 = pd.DataFrame(KT2E)
    if len(K2):
        K2["holm_family"] = [("K562" if sy == "K562" else "BM" if sy == "BONEMARROW" else "F2other") if f == "F2" else f
                             for f, sy in zip(K2.family, K2.system)]
        K2["p_holm"] = np.nan
        for (fam, rp), g in K2[K2.family != "F1"].groupby(["holm_family", "rep"]):
            K2.loc[g.index, "p_holm"] = holm(g.p_endpoint.to_numpy())
            zero = (g.stratum_size > 0) & (1.0 / g.stratum_size.clip(lower=1) > .05 / len(g))
            K2.loc[g.index[zero.to_numpy()], "status"] = "NO_CALL_ZERO_POWER"
    for nm, df in (("ENDPOINTS", pd.DataFrame(EP)), ("KT2_ENDPOINT", K2), ("KT2_FAMILY", pd.DataFrame(FAM)),
                   ("KT2_PAIRED", pd.DataFrame(PAIR)), ("STRATA", pd.DataFrame(STRATA)),
                   ("F1_BOOT", pd.DataFrame(BOOTF1)), ("F2F3_BOOT", pd.DataFrame(BOOTF2)), ("KT4", pd.DataFrame(KT4))):
        df.to_csv(out / f"{nm}.tsv", sep="\t", index=False)
    inputs = {}
    for system in [s_.upper() for s_ in args.systems]:
        d_ = RU.system_dir(system)
        for f_ in sorted(d_.glob("catalogue_U_k0_8*.npz")) + [d_ / "kt5_shuffledpt_U_k0_8.npz", d_ / "pearson.npz"]:
            if f_.exists():
                inputs[f"{system}/{f_.name}"] = RU.sha(f_)
    RU.atomic_json(out / "receipt.json", {
        "script_sha256": RU.sha(Path(__file__).resolve()),
        "dependencies": {f_: RU.sha(RU.SCRATCH / f_) for f_ in ("gb_primary.py", "gb_cop_alignment.py", "gb_real_uniform.py",
                                                               "gb_catalogue_scores.py", "gb_kt5.py", "gb_pairsubset_scores.py",
                                                               "gb_target_extension.py", "gb_gata1_fourclass_canon.py")},
        "contracts": {"fig1_eligibility_sha256": RU.FIG1_ELIGIBILITY_SHA256, "figure_contract_sha256": RU.FIGURE_CONTRACT_SHA256},
        "inputs": inputs, "systems": args.systems, "boot": args.boot, "perm": args.perm,
        "modes": {"before_after": bool(args.before_after), "t7": bool(args.t7), "kt4": bool(args.kt4)},
        "caliper": CALIPER, "paired": PAIRED, "pool_target": args.pool_target or POOL_TARGET, "matcher": args.matcher,
        "outputs": {p_.name: RU.sha(p_) for p_ in sorted(out.glob("*.tsv"))}})
    print(f"[done] {out}", flush=True)


COMPARE = ("KT1-catalogue", "KT1-roster(stress)", "KT3-PCpartial-8PC", "KT3-PCpartial-Z0:8", "KT5-shuffledPT",
           "PairStatic-PRIMARY", "GeoBRIDGE-RAW", "Pearson")


def boot_f1(store, system, G, rng, B):
    if not store:
        return []
    obs = {c: np.array([metric(k["V"]["GeoBRIDGE-PRIMARY"], k["y"], k["q"]) - metric(k["V"][c], k["y"], k["q"])
                        if c in k["V"] else np.nan for k in store]) for c in COMPARE}
    bs = {c: [] for c in COMPARE}
    for _ in range(B):
        cnt = rng.multinomial(G, np.full(G, 1.0 / G))
        d = {c: [] for c in COMPARE}
        for k in store:
            rep = np.repeat(np.arange(len(k["gidx"])), cnt[k["gidx"]])
            y, q = k["y"][rep], k["q"][rep]
            base = metric(k["V"]["GeoBRIDGE-PRIMARY"][rep], y, q)
            for c in COMPARE:
                d[c].append(base - metric(k["V"][c][rep], y, q) if c in k["V"] else np.nan)
        for c in COMPARE:
            bs[c].append(np.nanmedian(d[c]) if np.isfinite(d[c]).any() else np.nan)
    rows = []
    for c in COMPARE:
        o = obs[c]
        if not np.isfinite(o).any():
            continue
        lo, hi = np.nanpercentile(bs[c], [2.5, 97.5])
        rows.append(dict(system=system, comparator=c, n_tf=int(np.isfinite(o).sum()), median_delta=float(np.nanmedian(o)),
                         lo=lo, hi=hi, frac_tf_positive=float((o[np.isfinite(o)] > 0).mean())))
    return rows


def boot_ep(k, system, rng, B):
    y = k["y"]
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    rows = []
    for c in COMPARE:
        if c not in k["V"]:
            continue
        a, b = k["V"]["GeoBRIDGE-PRIMARY"], k["V"][c]
        obs = CA.auroc(a, y) - CA.auroc(b, y)
        bs = []
        for _ in range(B):
            idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
            bs.append(CA.auroc(a[idx], y[idx]) - CA.auroc(b[idx], y[idx]))
        bs = np.asarray(bs)
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        rows.append(dict(endpoint=k["name"], system=system, comparator=c, delta=obs, lo=lo, hi=hi,
                         p=min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))))
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    fz = sub.add_parser("falsifier")
    fz.add_argument("--out", default="killtests_falsifier")
    fz.add_argument("--pool-target", type=int, default=None)
    fz.add_argument("--regimes", nargs="*", default=None)
    fz.add_argument("--matcher", default="kmeans", choices=("kmeans", "nnblocks"))
    r = sub.add_parser("run")
    r.add_argument("--systems", nargs="+", required=True)
    r.add_argument("--boot", type=int, default=1000)
    r.add_argument("--perm", type=int, default=10000)
    r.add_argument("--kt4", action="store_true")
    r.add_argument("--before-after", action="store_true", help="RAW-z / PRIMARY / K8-diag / RANDOM-8")
    r.add_argument("--t7", action="store_true", help="PRIMARY vs SAME-only vs CROSS-only pair-subset banks")
    r.add_argument("--pool-target", type=int, default=None, help="stratum size target (default POOL_TARGET=50)")
    r.add_argument("--matcher", default="kmeans", choices=("kmeans", "nnblocks"), help="nnblocks = candidate matcher")
    r.add_argument("--out", default="killtests_v3")
    a = ap.parse_args()
    if a.cmd == "falsifier":
        import gb_real_uniform as RU
        T, ok = falsifier(pool_target=a.pool_target, regimes=a.regimes, matcher=a.matcher)
        (RU.OUT / a.out).mkdir(parents=True, exist_ok=True)
        T.to_csv(RU.OUT / a.out / "FALSIFIER.tsv", sep="\t", index=False)
        RU.atomic_json(RU.OUT / a.out / "receipt.json", {"script_sha256": RU.sha(Path(__file__).resolve()), "pass": ok})
    elif a.cmd == "selftest":
        falsifier(n_null=40, n_power=20, n_perm=200, procs=8)
    else:
        run(a)
