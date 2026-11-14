#!/usr/bin/env python
# Count-scale source calibration.
import hashlib, json, math, os, sys, time
import numpy as np, scipy, scipy.sparse as sp
from scipy.optimize import least_squares, minimize
_LSQ = least_squares                                                             # indirection so selftests can inject solver endpoints

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
if __name__ == "__main__" and sys.argv[1:2] != ["--selftest"]:
    _here = os.path.dirname(os.path.abspath(__file__))
    def _h16(p_):
        with open(p_, "rb") as fh_: return hashlib.sha256(fh_.read()).hexdigest()[:16]
    _exp = dict(kv.split("=") for kv in os.environ.get("GB_EXPECT_HASHES", "").split(",") if "=" in kv)
    _got = dict(calib3=_h16(os.path.abspath(__file__)), select=_h16(os.path.join(_here, "gb_m1_select.py")), source=_h16(os.path.join(_here, "gb_m1_source.py")))
    if os.environ.get("GB_SKIP_PREFLIGHT") == "1": print("preflight skipped (public release)")
    elif _exp != _got: sys.exit(f"RUNTIME_HASH_MISMATCH expected {_exp} got {_got}")
import gb_m1_source as M
import gb_m1_select as SEL

NS, NANCH, NPC, NRIV = "r111v1", 50, 10, 8
DET_MIN, NDET_MIN, ESS_MIN, LB_SUP = .10, 20, 30, .10
COND_GRID = (.10, .20, .30, .40)
LADDER = {"LADDER_0": 0.0, "LADDER_05": .05, "LADDER_10": .10, "LADDER_20": .20, "LADDER_40": .40,
          "F2_STRONG_P20": .20, "F2_WEAK_P20": .20, "F2_MIXED_P20": .20}          # F2 *_P20 = shared program + focal innovation power
FIXTURES = ("F1_RATE_Q2", "F1_NONLIN", "F2_STRONG", "F2_WEAK", "F2_MIXED", "F2_STRONG_P20", "F2_WEAK_P20", "F2_MIXED_P20",
            "LADDER_0", "LADDER_05", "LADDER_10", "LADDER_20", "LADDER_40")
SIGMA_GRID = (.125, .25, .5, .625, .75, .875, 1.0, 1.25, 1.5, 2.0, 3.0, 4.0)
F2_DEPTH = {"F2_STRONG": [4.0] * 8, "F2_WEAK": [.25] * 8, "F2_MIXED": [4.0] * 4 + [.25] * 4}
F2_DEPTH.update({k + "_P20": v for k, v in list(F2_DEPTH.items())})
F2_LOAD = [.9, -.8, .7, -.6, .9, -.8, .7, -.6]
C_PROGRAM = .8


# typed raw-byte sha-256 (SEL.raw_hash, algorithm sha256-typed-frames-v1), first 16 hex
def h16(x):
    if isinstance(x, (bytes, bytearray, np.ndarray, str)): return SEL.raw_hash(x)["hex16"]
    raise TypeError(f"h16: unsupported type {type(x)}")


def hkey(s):
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def balanced_split(ids):
    order = np.argsort(np.array([hashlib.sha256(f"split||{c}".encode()).hexdigest() for c in ids]), kind="stable")
    train = np.zeros(len(ids), bool); train[order[: len(ids) // 2]] = True
    return train


def roster(ids, pool, n, tag):
    keys = np.array([hashlib.sha256(f"{NS}|{tag}||{ids[i]}".encode()).hexdigest() for i in pool])
    sel = np.sort(pool[np.argsort(keys, kind="stable")[:n]])
    return sel, h16("\n".join(sorted(ids[i] for i in sel)))


def knn(W, pool, a, m):
    d = ((W[pool] - W[a]) ** 2).sum(1); d[pool == a] = np.inf
    return pool[np.argsort(d, kind="stable")[:m]]


def neighbourhoods(W, pool, anchors, m):
    N = [knn(W, pool, a, m) for a in anchors]
    return N, [M.basis(M.q2_design(W[n_] - W[a])) for n_, a in zip(N, anchors)]


def f_nonlin(W): return np.sin(np.pi * W[:, 0]) + .5 * W[:, 1] ** 2 + .3 * W[:, 0] * W[:, 1]


def program_z(phi):
    z = np.sin(phi) + .5 * np.cos(2 * phi); return (z - z.mean()) / z.std()


def innov_oracle(mu0, e, sigma, neigh, Qs):
    ri, rt = [], []
    for Nn, Q in zip(neigh, Qs):
        h = (Q ** 2).sum(1); x = mu0[Nn]
        S = ((1 - h) * x ** 2).sum() * (np.exp(sigma ** 2) - 1); N = ((1 - h) * x / e[Nn]).sum()
        r = x - Q @ (Q.T @ x); T = (r ** 2).sum()
        ri.append(S / (S + N)); rt.append((S + T) / (S + T + N))
    return np.array(ri), np.array(rt)


def moments(lam, e):
    return float((e * lam).mean()), float((1 - np.exp(-e * lam)).mean())


# Multistart bounded least squares on the exact tolerance-normalised residual
def match(loglam_fn, e, tmean, tdet, starts, two_param=True):
    def res(p):
        mm, dd = moments(np.exp(loglam_fn(p)), e)
        return [(mm / tmean - 1) / .10, (dd - tdet) / .02]
    bounds = ([0.0, -np.inf], [np.inf, np.inf]) if two_param else ([-np.inf], [np.inf])
    n_params = 2 if two_param else 1
    records = []
    for s0 in starts:
        s_ok = SEL.valid_start(s0, n_params)
        if s_ok is None:
            records.append(SEL.invalid_start_record(s0)); continue
        s0 = s_ok
        try:
            r = _LSQ(res, s0, method="trf", bounds=bounds, xtol=1e-12, ftol=1e-12, gtol=1e-12, max_nfev=500)
        except Exception as err:
            records.append(SEL.exception_record(s0, err)); continue
        records.append(SEL.evaluate_endpoint(r, s0, loglam_fn, e, tmean, tdet, n_params))
    return SEL.select_equal_best(records, tmean, tdet, two_param)


# identity-link Poisson on training cells subject to X beta >= eps at every cell
def fit_q2_rate_constrained(W, tr, y, e, eps=1e-6, outer=30):
    Xall = M.q2_design(W); U, s, Vt = np.linalg.svd(Xall, full_matrices=False); keep = s > 1e-10 * s.max()
    B_ = U[:, keep] * s[keep]; Xt = B_[tr]; yt, et = y[tr], e[tr]
    g = np.linalg.lstsq(B_, np.full(len(W), max(yt.sum() / et.sum(), 1e-3)), rcond=None)[0]
    slack = lambda g_: B_ @ g_ - eps
    assert slack(g).min() > 0
    def obj(g_, mu):
        m_ = Xt @ g_; sl = slack(g_)
        if (m_ <= 0).any() or (sl <= 0).any(): return np.inf
        return (et * m_).sum() - (yt * np.log(et * m_)).sum() - mu * np.log(sl).sum()
    mu = 1.0; ls_fail = 0; last_mu = mu; kkt = np.inf
    for _ in range(outer):
        conv = False
        for _ in range(100):
            m_ = Xt @ g; sl = slack(g)
            grad = Xt.T @ (et - yt / m_) - mu * (B_.T @ (1 / sl))
            H = (Xt * (yt / m_ ** 2)[:, None]).T @ Xt + mu * (B_ * (1 / sl ** 2)[:, None]).T @ B_
            step = np.linalg.solve(H + 1e-12 * np.eye(len(g)), grad)
            dec = float(grad @ step)                                          # Newton decrement^2
            if dec < 1e-10: conv = True; break
            f0 = obj(g, mu); t = 1.0
            while t > 1e-12 and not (obj(g - t * step, mu) <= f0 - 1e-4 * t * dec): t *= .5
            if t <= 1e-12: ls_fail += 1; break
            g = g - t * step
        last_mu = mu; kkt = dec
        if not conv: break
        if mu < 1e-9: break
        mu *= .2
    q = B_ @ g
    nll = float((et * (Xt @ g)).sum() - (yt * np.log(et * (Xt @ g))).sum())
    ok = bool(np.all(np.isfinite(q)) and q.min() >= eps and conv and ls_fail == 0 and last_mu < 1e-8)
    return q, dict(success=ok, converged_final=bool(conv), linesearch_failures=ls_fail, newton_decrement_final=kkt,
                   barrier_optimized=last_mu, min_rate=float(q.min()), nll=nll, active_frac=float((q < 10 * eps).mean()))


def top1(y):
    k = math.ceil(0.01 * len(y)); t = float(y.sum())
    return float(np.sort(y)[::-1][:k].sum() / t) if t > 0 else float("nan")


# Each draw redraws the program (phi -> z over the full world, standardised as generated), the innovation and Poisson counts
def concentration_check(y_real, tr, e, mu_fn, sigma, has_program, seed, n, draws=4096):
    c_real = top1(y_real[tr])
    if not np.isfinite(c_real): return dict(status="CALIBRATION_UNMATCHED", reason="zero_total_reference")
    g = np.random.default_rng(seed); vals = np.empty(draws); zero = 0
    for d in range(draws):
        z = program_z(g.uniform(0, 2 * np.pi, n)) if has_program else None
        lam = mu_fn(z)[tr] * np.exp(sigma * g.normal(size=len(tr)) - sigma ** 2 / 2)
        v = top1(g.poisson(e[tr] * lam).astype(float))
        if not np.isfinite(v): zero += 1; v = np.nan
        vals[d] = v
    if zero > 0:
        return dict(status="CALIBRATION_UNRESOLVED", reason="zero_total_draws", zero_total_draws=zero, draws=draws, c_real=c_real)
    mc = float(vals.mean()); se = float(vals.std(ddof=1) / math.sqrt(draws))
    lo, hi = mc - 1.96 * se, mc + 1.96 * se; band = (2 / 3 * c_real, 1.5 * c_real)
    st = "MATCHED" if (lo >= band[0] and hi <= band[1]) else ("CALIBRATION_UNMATCHED" if (hi < band[0] or lo > band[1]) else "CALIBRATION_UNRESOLVED")
    return dict(status=st, c_real=c_real, expected_c=mc, mc_ci=[lo, hi], band=list(band), draws=draws, zero_total_draws=0)


def statuses(rho, lb, det, ndet, ess, logsb, thr=LB_SUP):
    reasons = []
    if not np.isfinite(rho): reasons.append("rho_undefined")
    if det < DET_MIN: reasons.append("detection")
    if ndet < NDET_MIN: reasons.append("detected_cells")
    if ess < ESS_MIN: reasons.append("contribution_ess")
    if lb is None or not np.isfinite(lb): reasons.append("bound_undefined")
    if reasons: return "UNRESOLVED", reasons
    if lb > thr:
        if not np.isfinite(logsb): return "UNRESOLVED", ["count_log_undefined"]
        if logsb <= LB_SUP: return "UNRESOLVED", ["count_log_point_disagreement"]
        return "MEASUREMENT_SUPPORTED", []
    return ("WEAK", []) if lb > 0 else ("UNRESOLVED", ["lower_bound_le_0"])


# total and fail-closed: NaN unless lc is finite and -1 < lc <= 1
def log_agreement(lc):
    try:
        lc = float(lc)
    except (TypeError, ValueError):
        return float("nan")
    if not (np.isfinite(lc) and -1.0 < lc <= 1.0): return float("nan")
    return 2 * lc / (1 + lc)


def anchor_diag(Q, N, x1, x2, Y, e):
    r1 = x1[N] - Q @ (Q.T @ x1[N]); r2 = x2[N] - Q @ (Q.T @ x2[N]); r = (r1 + r2) / 2
    c = r ** 2; den = float(c.sum())
    ess = den ** 2 / max(float((c ** 2).sum()), 1e-300) if den > 0 else 0.0
    y1 = np.rint(x1[N] * e[N] / 2); y2 = Y[N] - y1
    l1 = np.log1p(np.maximum(y1, 0) / (e[N] / 2)); l2 = np.log1p(np.maximum(y2, 0) / (e[N] / 2))
    lr1 = l1 - Q @ (Q.T @ l1); lr2 = l2 - Q @ (Q.T @ l2)
    dd = float(np.sqrt((lr1 ** 2).sum() * (lr2 ** 2).sum()))
    lc = float((lr1 * lr2).sum() / dd) if dd > 0 else float("nan")
    return dict(den=den, ess=ess, maxc=float(c.max() / den) if den > 0 else float("nan"), det=float((Y[N] > 0).mean()),
                ndet=int((Y[N] > 0).sum()), logsb=log_agreement(lc), qrank=int(Q.shape[1]),
                resid_df=int(len(N) - Q.shape[1] - 1))


# One bootstrap family over the included anchors
def family_table(neigh, Qs, x1, x2, Y, e, ids, anchors, pool, B, seed, oracle=None, grid=None, excluded=None):
    excluded = excluded or {}
    inc = [k for k in range(len(neigh)) if k not in excluded]
    rows = [None] * len(neigh)
    for k, why in excluded.items():
        rows[k] = dict(anchor=ids[anchors[k]], status="INCOMPLETE_PANEL" if why == "INCOMPLETE_PANEL" else "UNRESOLVED", reasons=[why])
    if not inc:
        return dict(family="EMPTY", crit=None, union_failed_frac=None, counts=None, threshold_curve=None, source_pass=None, anchors=rows)
    rho, lb, sd, crit_, info = M.boot_rho_simultaneous([neigh[k] for k in inc], [Qs[k] for k in inc], x1[:, None], x2[:, None],
                                                       len(e), B, seed, pool=pool)
    fam = info["status"]
    decided = fam in ("OK", "NO_MEMBERS")
    counts = dict(MEASUREMENT_SUPPORTED=0, WEAK=0, UNRESOLVED=0) if decided else None
    curve = {str(t): 0 for t in (grid or ())} if decided else None
    for j, k in enumerate(inc):
        dg = anchor_diag(Qs[k], neigh[k], x1, x2, Y, e)
        member = bool(info["members"][j, 0])
        lbk = float(lb[j, 0]) if (fam == "OK" and member and np.isfinite(lb[j, 0])) else None
        if not decided:
            s, why = None, ["FAMILY_BOOTSTRAP_FAILURE"]
        elif not member:
            s, why = "UNRESOLVED", ["non_member_undefined_estimate_or_sd"]
        else:
            s, why = statuses(float(rho[j, 0]), lbk if lbk is not None else float("nan"), dg["det"], dg["ndet"], dg["ess"], dg["logsb"])
        if decided:
            counts[s] += 1
            for t in (grid or ()):
                if member and lbk is not None:
                    s_t, _ = statuses(float(rho[j, 0]), lbk, dg["det"], dg["ndet"], dg["ess"], dg["logsb"], thr=t)
                    curve[str(t)] += int(s_t == "MEASUREMENT_SUPPORTED")
        rows[k] = dict(anchor=ids[anchors[k]], rho=float(rho[j, 0]), lb=lbk, sd=float(sd[j, 0]), member=member,
                       failed_draw_frac=float(info["cell_failed_frac"][j, 0]), oracle_innovation=None if oracle is None else oracle[k],
                       status=s, reasons=why, **dg)
    return dict(family=fam, crit=float(crit_) if fam == "OK" else None, union_failed_frac=info.get("union_failed_frac"),
                n_members=info.get("n_members"), counts=counts, threshold_curve=curve,
                source_pass=(counts["MEASUREMENT_SUPPORTED"] >= 10) if decided else None, anchors=rows)


FRAME_DIR = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/results/m1_frames_2026_09_14")


def load_frame(ds, freeze=False):
    import anndata as ad
    import gb_real_feasibility as F
    from sklearn.decomposition import PCA
    cfg = F.CFG[ds]
    raw = ad.read_h5ad(cfg["raw"]); sub = ad.read_h5ad(cfg["sub"], backed="r")
    if cfg["wt"]: raw = raw[raw.obs["genotype"] == cfg["wt"]].copy()
    assert list(raw.obs_names) == list(sub.obs_names)
    ids = np.array(raw.obs_names); names = list(raw.var_names); nameset = set(names)
    S = sp.csc_matrix(raw.layers["spliced"])
    assert np.allclose(S.data, np.round(S.data))
    cat = F.mouse_catalogue()
    feat = [g for g in sub.var_names if g in nameset and g not in cat]
    Xf = S[:, [names.index(g) for g in feat]].toarray().astype(float)
    e = Xf.sum(1); assert (e > 0).all(); e = e / np.median(e)
    train = balanced_split(ids)
    inputs = dict(raw=F.fsha(cfg["raw"]), sub=F.fsha(cfg["sub"]), features=h16("\n".join(feat)), exposure=h16(e),
                  split=h16("\n".join(sorted(ids[train]))), cells=h16("\n".join(ids)))
    stem = f"{FRAME_DIR}/{ds}_frame"
    if freeze:                                                              # one-time deterministic freeze (full lapack SVD)
        L = np.log1p(Xf / e[:, None]); mu, sd = L[train].mean(0), L[train].std(0) + 1e-8
        Wpca = PCA(NPC, svd_solver="full").fit((L[train] - mu) / sd).transform((L - mu) / sd)
        W = (Wpca - Wpca[train].mean(0)) / Wpca[train].std(0)
        os.makedirs(FRAME_DIR, exist_ok=True)
        with open(stem + ".npz", "xb") as fh: np.savez(fh, Wpca=Wpca, W=W)
        meta = dict(dataset=ds, inputs=inputs, npz_sha256=SEL.file_sha256_16(stem + ".npz"), arrays=dict(Wpca=SEL.raw_hash(Wpca)["hex16"], W=SEL.raw_hash(W)["hex16"]),
                    producer=SEL.file_sha256_16(__file__), pca="sklearn PCA svd_solver=full on training-standardised log1p(X/e)")
        with open(stem + ".json", "x") as fh: json.dump(meta, fh, indent=1)
    meta = json.load(open(stem + ".json"))
    if meta["inputs"] != inputs: raise ValueError(f"frozen frame inputs mismatch for {ds}")
    if SEL.file_sha256_16(stem + ".npz") != meta["npz_sha256"]: raise ValueError("frozen frame file hash mismatch")
    z_ = np.load(stem + ".npz"); Wpca, W = z_["Wpca"], z_["W"]
    if SEL.raw_hash(Wpca)["hex16"] != meta["arrays"]["Wpca"] or SEL.raw_hash(W)["hex16"] != meta["arrays"]["W"]: raise ValueError("frozen frame array hash mismatch")
    focal = np.asarray(S[:, names.index(cfg["focal"])].todense()).ravel().astype(float)
    tf_names = sorted(g for g in cat if g in nameset and g != cfg["focal"])
    TF = np.asarray(S[:, [names.index(g) for g in tf_names]].todense()).astype(float)
    import sklearn, anndata
    return dict(ds=ds, ids=ids, e=e, W=W, Wpca=Wpca, train=train, focal=focal, tf_names=tf_names, TF=TF,
                hashes=dict(inputs, W=h16(W), frozen_frame=meta["npz_sha256"], tf_catalogue=h16("\n".join(tf_names))),
                versions=dict(numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, anndata=anndata.__version__))


# Return the rate function, sigma, latent variables, match record and planted parameters.
def calibrate_world(fr, fixture, world, Ntr, Qtr, tr, rng_seed, _hooks=None):
    W, e, n = fr["W"], fr["e"], len(fr["ids"])
    g = np.random.default_rng(rng_seed)
    xi = g.normal(size=n); z = program_z(g.uniform(0, 2 * np.pi, n))
    tmean, tdet = float(fr["focal"][tr].mean()), float((fr["focal"][tr] > 0).mean())
    if fixture == "F1_RATE_Q2":
        q, fit = fit_q2_rate_constrained(W, tr, fr["focal"], e)
        if not fit["success"]:
            return None, 0.0, z, xi, dict(status="CALIBRATION_UNRESOLVED", reason="constrained_q2_fit_failed", fit=fit), dict(q2_fit=fit)
        p, mt = match(lambda p: np.log(np.exp(p[0]) * q[tr]), e[tr], tmean, tdet, [np.array([s]) for s in np.linspace(-3, 3, 7)], two_param=False)
        mu_fn = (lambda z_, p=p: np.exp(p[0]) * q) if p is not None else None
        return mu_fn, 0.0, z, xi, mt, dict(q2_fit=fit, scale=None if p is None else float(p[0]))
    f = f_nonlin(W); c = C_PROGRAM if fixture.startswith("F2") else 0.0
    if fixture not in FIXTURES: raise ValueError(f"unknown fixture {fixture}")
    starts = [np.array([a0, b0]) for a0 in (.2, .8, 1.5, 2.5) for b0 in (-4, -2, 0)]
    warm = []
    def solve(sig):
        fn = lambda p: p[1] + p[0] * f[tr] + c * z[tr] + sig * xi[tr] - sig ** 2 / 2
        pp, mm = match(fn, e[tr], tmean, tdet, warm[-1:] + starts)
        if pp is None or mm["status"] != "MATCHED":
            pp, mm = match(fn, e[tr], tmean, tdet, warm[::-1] + starts)
            mm = dict(mm, retried_with_all_matched=len(warm))
        if pp is not None and mm["status"] == "MATCHED": warm.append(pp.copy())
        return pp, mm
    target = LADDER.get(fixture)
    sigma = 0.0
    p, mt = solve(0.0)
    if target is not None and target > 0:
        hooks = _hooks or {}
        def tenth(sig):                                                         # _hooks: selftest-only injection of solver / oracle
            pp, mm = (hooks["solve"](sig) if "solve" in hooks else solve(sig))
            if pp is None or mm["status"] != "MATCHED": return None, pp, mm
            if "tenth" in hooks: return float(hooks["tenth"](sig)), pp, mm
            mu0 = np.exp(pp[1] + pp[0] * f + c * z)
            ri, _ = innov_oracle(mu0, e, sig, Ntr, Qtr)
            return float(np.sort(ri)[::-1][9]), pp, mm
        lo, hi, bracket, bisection = 0.0, None, [], []
        for cand in SIGMA_GRID:
            tv, _, mm = tenth(cand)
            if tv is None:                                                     # retry with all multistarts + matched bracket endpoints
                extra = [np.array(b_["params"]) for b_ in bracket if b_["params"] is not None]
                pp, mm2 = (hooks["solve"](cand) if "solve" in hooks else
                           match(lambda p: p[1] + p[0] * f[tr] + c * z[tr] + cand * xi[tr] - cand ** 2 / 2, e[tr], tmean, tdet, extra + starts))
                mm = dict(mm2, retry=True)
                if pp is not None and mm2["status"] == "MATCHED":
                    warm.append(pp.copy()); tv, _, mm = tenth(cand)
            bracket.append(dict(sigma=cand, tenth=tv, params=None if tv is None else warm[-1].tolist(), match_status=mm.get("status")))
            if tv is None:
                return None, cand, z, xi, dict(status="CALIBRATION_UNRESOLVED", reason="matching_failed_during_bracket", sigma=cand,
                                               bracket=bracket, bisection=bisection, last_match=mm), dict(target=target)
            if tv >= target: hi = cand; break
            lo = cand
        if hi is None:
            return None, lo, z, xi, dict(status="CALIBRATION_UNRESOLVED", reason="target_unreachable_within_grid", sigma=lo, bracket=bracket,
                                         bisection=bisection, last_match=mm), dict(target=target)
        for _ in range(40):
            mid = (lo + hi) / 2; tv, _, mm = tenth(mid)
            bisection.append(dict(sigma=mid, tenth=tv, match_status=mm.get("status")))
            if tv is None:
                return None, mid, z, xi, dict(status="CALIBRATION_UNRESOLVED", reason="matching_failed_during_bisection", sigma=mid,
                                              bracket=bracket, bisection=bisection, last_match=mm), dict(target=target)
            (lo, hi) = (mid, hi) if tv < target else (lo, mid)
        sigma = (lo + hi) / 2
        tv, p, mt = tenth(sigma)
        bisection.append(dict(sigma=sigma, tenth=tv, match_status=mt.get("status"), final_verification=True))
        if tv is None or abs(tv - target) > .005:
            return None, sigma, z, xi, dict(status="CALIBRATION_UNRESOLVED", reason="ladder_verification_failed", achieved=tv, sigma=sigma,
                                            bracket=bracket, bisection=bisection, last_match=mt), dict(target=target)
        mt = dict(mt, ladder_training_10th=tv, ladder_target=target, bracket=bracket, bisection=bisection)
    if p is None or mt["status"] != "MATCHED":
        return None, sigma, z, xi, mt, dict(sigma=sigma)
    a, b = float(p[0]), float(p[1])
    mu_fn = lambda z_, a=a, b=b: np.exp(b + a * f + c * (z if z_ is None else z_))
    return mu_fn, sigma, z, xi, mt, dict(a=a, b=b, c=c, sigma=sigma)


def select_panel(fr, cand_X, cand_Y, cand_names, focal_rate, Wtr_nbhd, anchor, Wall):
    Q = M.basis(M.q2_design(Wall[Wtr_nbhd] - Wall[anchor]))
    rf = focal_rate[Wtr_nbhd] - Q @ (Q.T @ focal_rate[Wtr_nbhd])
    Xc = cand_X[Wtr_nbhd]; rc = Xc - Q @ (Q.T @ Xc)
    det = (cand_Y[Wtr_nbhd] > 0).mean(0); nd = (cand_Y[Wtr_nbhd] > 0).sum(0)
    den = np.sqrt((rc ** 2).sum(0) * (rf ** 2).sum())
    cc = np.where(den > 0, np.abs((rc * rf[:, None]).sum(0)) / np.where(den > 0, den, 1), -1.0)
    elig = np.flatnonzero((det >= DET_MIN) & (nd >= NDET_MIN) & (den > 0))
    order = sorted(elig.tolist(), key=lambda j: (-cc[j], cand_names[j]))
    return order[:NRIV], len(elig)


METRICS = ("whitened", "variance")


# analysis-geometry coordinates
def metric_coords(fr, metric):
    if metric == "whitened": return fr["W"]
    if metric == "variance":
        Zc = fr["Wpca"] - fr["Wpca"][fr["train"]].mean(0); return Zc / np.sqrt((Zc[fr["train"]] ** 2).sum(1).mean())
    raise ValueError(f"unknown metric {metric}")


def run(fr, fixture, world, B, outpath=None, metric="whitened"):
    if fixture not in FIXTURES: raise ValueError(f"unknown fixture {fixture}")
    if metric not in METRICS: raise ValueError(f"unknown metric {metric}")
    t0 = time.time()
    ids, e, Wg, train = fr["ids"], fr["e"], fr["W"], fr["train"]; n = len(ids)
    W = metric_coords(fr, metric)
    tr, an = np.flatnonzero(train), np.flatnonzero(~train)
    m_tr, m_an = math.ceil(4 * math.sqrt(len(tr))), math.ceil(4 * math.sqrt(len(an)))
    atr, hash_tr = roster(ids, tr, NANCH, "train"); aan, hash_an = roster(ids, an, NANCH, "analysis")
    Ntr_g, Qtr_g = neighbourhoods(Wg, tr, atr, m_tr)                            # generator calibration geometry (metric-independent)
    Nan, Qan = neighbourhoods(W, an, aan, m_an)                                 # analysis geometry (metric arm)
    seeds = dict(generator=[hkey(NS), hkey(fixture), int(world)], counts=hkey(f"{NS}|counts|{fixture}|{world}"),
                 split=hkey(NS) + int(world), boot=hkey(f"{NS}|boot|{fixture}|{world}"), bootC=hkey(f"{NS}|bootC|{fixture}|{world}"),
                 concentration=hkey(f"{NS}|fid|{fixture}|{world}"))
    mu_fn, sigma, z, xi, mt, planted = calibrate_world(fr, fixture, world, Ntr_g, Qtr_g, tr, seeds["generator"])
    res = dict(dataset=fr["ds"], fixture=fixture, world=int(world), metric=metric, namespace=NS, B=int(B), n=n, n_train=len(tr), n_analysis=len(an),
               m_train=m_tr, m_analysis=m_an, roster_train=hash_tr, roster_analysis=hash_an,
               anchor_ids_train=[ids[i] for i in atr], anchor_ids_analysis=[ids[i] for i in aan], match=mt, planted=planted,
               seeds=seeds, hashes=fr["hashes"], versions=fr["versions"],
               config=dict(NANCH=NANCH, NPC=NPC, NRIV=NRIV, DET_MIN=DET_MIN, NDET_MIN=NDET_MIN, ESS_MIN=ESS_MIN, LB_SUP=LB_SUP, COND_GRID=COND_GRID),
               code=dict(calib3=SEL.file_sha256_16(__file__), m1=SEL.file_sha256_16(M.__file__), select=SEL.file_sha256_16(SEL.__file__),
                         file_hash_algorithm=SEL.FILE_HASH_ALG, generator_world_hash_algorithm=SEL.HASH_ALG))
    if mu_fn is None or mt.get("status") != "MATCHED":
        res.update(decision_status="NO_DECISION_" + mt.get("status", "CALIBRATION_UNRESOLVED"), focal=None, conditioned=None)
        return finish(res, outpath, t0)
    has_prog = fixture.startswith("F2")
    mu0 = mu_fn(None)
    lam = mu0 * np.exp(sigma * xi - sigma ** 2 / 2)
    cols = [lam]; names = ["focal"]
    if has_prog:
        a, b = planted["a"], planted["b"]; f = f_nonlin(Wg)
        for k in range(NRIV):
            cols.append(F2_DEPTH[fixture][k] * np.exp(b + F2_LOAD[k] * C_PROGRAM * z + .5 * a * f)); names.append(f"proxy{k}")
    Lam = np.column_stack(cols)
    Y = np.random.default_rng(seeds["counts"]).poisson(e[:, None] * Lam)
    res["generator_world_hash"] = SEL.raw_hash(np.ascontiguousarray(Lam), np.ascontiguousarray(Y))["hex16"]
    fidc = concentration_check(fr["focal"], tr, e, mu_fn, sigma, has_prog, seeds["concentration"], n)
    res["concentration"] = fidc
    if fidc["status"] != "MATCHED":
        res.update(decision_status="NO_DECISION_" + fidc["status"], focal=None, conditioned=None)
        return finish(res, outpath, t0)
    ri_an, rt_an = innov_oracle(mu0, e, sigma, Nan, Qan)
    res["oracle"] = dict(innovation=ri_an.tolist(), total_reproducibility=rt_an.tolist(),
                         training_between_state_rate_var=float(np.var(mu0[tr])))
    Y1, Y2 = M.split_counts(Y[:, [0]], [f"{NS}|{fixture}|{world}|focal"], seeds["split"])
    x1, x2 = 2 * Y1[:, 0] / e, 2 * Y2[:, 0] / e
    res["focal"] = family_table(Nan, Qan, x1, x2, Y[:, 0], e, ids, aan, an, B, seeds["boot"], oracle=ri_an)
    if has_prog:
        cand_Y = np.column_stack([Y[:, 1:], fr["TF"]]); cand_names = names[1:] + fr["tf_names"]; cand_X = cand_Y / e[:, None]
        focal_rate = Y[:, 0] / e
        panels, QC, excluded = [], [], {}
        for k, a in enumerate(aan):
            Nt = knn(W, tr, a, m_tr)                                                    # anchor's nearest training neighbourhood
            pnl, n_elig = select_panel(fr, cand_X, cand_Y, cand_names, focal_rate, Nt, a, W)
            panels.append(dict(anchor=ids[a], rivals=[cand_names[j] for j in pnl], n_eligible=int(n_elig)))
            if len(pnl) < NRIV:
                excluded[k] = "INCOMPLETE_PANEL"; QC.append(None); continue
            QC.append(M.basis(np.column_stack([Qan[k], cand_X[Nan[k]][:, pnl]])))
        ri_c = [None if Qc is None else float(innov_oracle(mu0, e, sigma, [Nan[k]], [Qc])[0][0]) for k, Qc in enumerate(QC)]
        QCf = [Qc if Qc is not None else Qan[k] for k, Qc in enumerate(QC)]
        res["rival_panels"] = panels
        res["conditioned"] = family_table(Nan, QCf, x1, x2, Y[:, 0], e, ids, aan, an, B, seeds["bootC"], oracle=ri_c,
                                          grid=COND_GRID, excluded=excluded)
        res["conditioned"]["any_false_attribution_by_threshold"] = (
            {t: v > 0 for t, v in res["conditioned"]["threshold_curve"].items()} if res["conditioned"]["threshold_curve"] is not None else None)
    fam_states = [res["focal"]["family"]] + ([res["conditioned"]["family"]] if has_prog else [])
    ok_states = ("OK", "NO_MEMBERS")
    if all(sf in ok_states for sf in fam_states):
        res["decision_status"] = "DECIDED"
    elif has_prog and res["conditioned"]["family"] == "EMPTY" and res["focal"]["family"] in ok_states:
        res["decision_status"] = "NO_DECISION_CONDITIONED_EMPTY"
    else:
        res["decision_status"] = "FAMILY_BOOTSTRAP_FAILURE:" + ",".join(fam_states)
    return finish(res, outpath, t0)


def finish(res, outpath, t0):
    res["runtime_s"] = time.time() - t0
    def clean(o):
        if isinstance(o, dict): return {str(k): clean(v) for k, v in o.items()}
        if isinstance(o, (list, tuple)): return [clean(v) for v in o]
        if isinstance(o, (np.floating, float)): return float(o) if np.isfinite(o) else None
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, np.ndarray): return clean(o.tolist())
        return o
    res = clean(res)
    if outpath:
        with open(outpath, "x") as fh: json.dump(res, fh, indent=1, allow_nan=False)
    return res


def _toy_frame(n=2400, seed=0, ntf=40):
    g = np.random.default_rng(seed)
    ids = np.array([f"c{i:05d}" for i in range(n)]); W = g.normal(size=(n, 3)) * .7
    e = g.lognormal(0, .3, n); e = e / np.median(e)
    focal = g.poisson(e * np.exp(-1.2 + .5 * f_nonlin(W))).astype(float)
    TF = g.poisson(e[:, None] * np.exp(g.normal(-1, .5, ntf))[None, :]).astype(float)
    return dict(ds="TOY", ids=ids, e=e, W=W, Wpca=W.copy(), train=balanced_split(ids), focal=focal, tf_names=[f"tf{j:03d}" for j in range(ntf)], TF=TF,
                hashes={}, versions={})


def selftest():
    global anchor_diag
    g = np.random.default_rng(0)
    ids = np.array([f"cell{i}" for i in range(2000)])
    tr = balanced_split(ids); perm = g.permutation(2000)
    assert tr.sum() == 1000 and (balanced_split(ids[perm]) == tr[perm]).all()
    s1, hA = roster(ids, np.flatnonzero(tr), 50, "x"); s2, hB = roster(ids[perm], np.flatnonzero(tr[perm]), 50, "x")
    assert hA == hB and set(ids[s1]) == set(ids[perm][s2])
    # D2: constrained positive Q2 rate is exactly in the Q2 span (annihilated by a global Q2
    # projector) and positive everywhere
    fr = _toy_frame()
    q, fit = fit_q2_rate_constrained(fr["W"], np.flatnonzero(fr["train"]), fr["focal"], fr["e"])
    Qg = M.basis(M.q2_design(fr["W"]))
    assert fit["success"] and q.min() >= 1e-6 * (1 - 1e-6) and np.abs(q - Qg @ (Qg.T @ q)).max() < 1e-7 * max(1, q.max())
    # D6: bootstrap weights live on the analysis pool only (neighbourhood outside the pool must be
    # rejected)
    x = np.abs(g.normal(size=(2000, 1))); Wt = g.normal(size=(2000, 2)); pool = np.flatnonzero(~tr)
    Nn = [knn(Wt, pool, pool[0], 150)]; Qn = [M.basis(M.q2_design(Wt[Nn[0]] - Wt[pool[0]]))]
    M.boot_rho_simultaneous(Nn, Qn, x, x + .1, 2000, 20, 1, pool=pool)
    try:
        M.boot_rho_simultaneous([np.flatnonzero(tr)[:150]], Qn, x, x, 2000, 5, 1, pool=pool); raise AssertionError("pool leak")
    except AssertionError as err:
        assert "pool" in str(err) or "outside" in str(err)
    # D5: union failure fraction -> family failure when a degenerate (all-zero) source makes every
    # draw undefined
    z0 = np.zeros((2000, 1)); rho, lb, sd, crit, info = M.boot_rho_simultaneous(Nn, Qn, z0, z0, 2000, 20, 1, pool=pool)
    assert not np.isfinite(crit) and not info["members"].any()                                        # no member -> no decision
    sparse = np.zeros((2000, 1)); sparse[Nn[0][:2], 0] = [3.0, 5.0]                                  # defined estimate, fragile draws
    two = np.column_stack([sparse[:, 0], x[:, 0]])
    rho, lb, sd, crit, info = M.boot_rho_simultaneous(Nn, Qn, two, two + .05, 2000, 200, 3, pool=pool)
    assert np.isfinite(rho).all() and info["union_failed_frac"] > .01 and not np.isfinite(crit)          # union failure -> family failure
    degen = np.column_stack([z0[:, 0], x[:, 0]])
    rho, lb, sd, crit, info = M.boot_rho_simultaneous(Nn, Qn, degen, degen + .05, 2000, 50, 4, pool=pool)
    assert np.isfinite(crit) and np.isnan(lb[0, 0]) and np.isfinite(lb[0, 1])                            # degenerate member only
    # D7: concentration check redraws the program: expected share differs from a frozen-z
    # computation for a program-dominated rate
    n = 3000; zfix = program_z(g.uniform(0, 2 * np.pi, n)); ee = np.ones(n); trn = np.arange(1500)
    mu_fn = lambda z_: np.exp(-1 + 2.5 * (zfix if z_ is None else z_))
    yreal = g.poisson(mu_fn(None)).astype(float)
    cr = concentration_check(yreal, trn, ee, mu_fn, 0.0, True, 7, n, draws=64)
    assert cr["draws"] == 64 and cr["status"] in ("MATCHED", "CALIBRATION_UNMATCHED", "CALIBRATION_UNRESOLVED")
    # D1: ladder world verifies matching + target at the final sigma
    frt = _toy_frame(seed=1); trt = np.flatnonzero(frt["train"])
    at, _ = roster(frt["ids"], trt, NANCH, "train"); Ntt, Qtt = neighbourhoods(frt["W"], trt, at, math.ceil(4 * math.sqrt(len(trt))))
    mu, sg, _, xi_, mtr, pl = calibrate_world(frt, "LADDER_20", 1, Ntt, Qtt, trt, [1, 2, 3])
    assert mu is not None, ("toy ladder must be reachable", mtr)
    if mu is not None:
        lam_tr = mu(None)[trt] * np.exp(sg * xi_[trt] - sg ** 2 / 2); mm, dd = moments(lam_tr, frt["e"][trt])
        assert abs(mm / frt["focal"][trt].mean() - 1) <= .10 and abs(dd - (frt["focal"][trt] > 0).mean()) <= .02
        assert abs(mtr["ladder_training_10th"] - .20) <= .005
    else:
        assert mtr["status"] in ("CALIBRATION_UNRESOLVED", "CALIBRATION_UNMATCHED")
    # D3/D4: per-anchor panels (training neighbourhood), incomplete panel excluded, strata
    # definitions
    assert F2_DEPTH["F2_MIXED"] == [4.0] * 4 + [.25] * 4 and len(F2_DEPTH["F2_STRONG"]) == NRIV
    r = run(frt, "F2_MIXED", 1, 49)
    if r["decision_status"] == "DECIDED":
        pn = r["rival_panels"]; assert len(pn) == NANCH and len({tuple(p["rivals"]) for p in pn}) >= 1
        assert set(r["conditioned"]["threshold_curve"]) == {str(t) for t in COND_GRID}
        for row in r["conditioned"]["anchors"]:
            if row["status"] == "INCOMPLETE_PANEL": assert row["reasons"] == ["INCOMPLETE_PANEL"]
    json.dumps(r, allow_nan=False)
    # statuses: log veto, ess gate, label
    assert statuses(.9, .8, .5, 50, 5.0, .9)[0] == "UNRESOLVED"
    assert statuses(.9, .8, .5, 50, 80.0, .02) == ("UNRESOLVED", ["count_log_point_disagreement"])
    assert statuses(.9, .8, .5, 50, 80.0, .9) == ("MEASUREMENT_SUPPORTED", [])
    for bad_log in (float("nan"), float("inf"), -float("inf")):
        assert statuses(.9, .8, .5, 50, 80.0, bad_log) == ("UNRESOLVED", ["count_log_undefined"])
        assert statuses(.9, .05, .5, 50, 80.0, bad_log) == ("WEAK", [])
    for v_, ok_ in ((np.nextafter(-1.0, -np.inf), False), (-1.0, False), (float("nan"), False), (float("inf"), False), (-float("inf"), False),
                    (None, False), (.5, True), (1.0, True), (np.nextafter(1.0, 2.0), False), (0.0, True), (np.nextafter(-1.0, 0.0), True)):
        assert bool(np.isfinite(log_agreement(v_))) == ok_, (v_, log_agreement(v_))
    assert log_agreement(1.0) == 1.0 and log_agreement(0.0) == 0.0
    gA = np.random.default_rng(5); WA = gA.uniform(-1, 1, size=(600, 2)); poolA = np.arange(300, 600); aA = poolA[:3]
    NA_ = [knn(WA, poolA, a, 120) for a in aA]; QA_ = [M.basis(M.q2_design(WA[n_] - WA[a])) for n_, a in zip(NA_, aA)]
    YA = np.ones(600); yA1 = gA.binomial(1, .5, 600).astype(float); xA1, xA2 = 2 * yA1, 2 * (1 - yA1)
    dA = anchor_diag(QA_[0], NA_[0], xA1, xA2, YA, np.ones(600))
    assert not np.isfinite(dA["logsb"])
    tA = family_table(NA_, QA_, xA1, xA2, YA, np.ones(600), np.array([f"A{i}" for i in range(600)]), aA, poolA, 30, 13, grid=COND_GRID)
    assert all(r_["status"] in ("UNRESOLVED", None) for r_ in tA["anchors"]) and (tA["counts"] is None or tA["counts"]["MEASUREMENT_SUPPORTED"] == 0)
    assert tA["threshold_curve"] is None or all(v_ == 0 for v_ in tA["threshold_curve"].values())
    json.dumps(finish(dict(tA), None, time.time()), allow_nan=False)
    # through family_table: strong local innovation (supported rows exist), then logsb forced
    # undefined -> no supported row
    gL = np.random.default_rng(7); WL = gL.uniform(-1, 1, size=(900, 2)); poolL = np.arange(300, 900); aL = poolL[::120][:5]
    NL = [knn(WL, poolL, a, 200) for a in aL]; QL = [M.basis(M.q2_design(WL[n_] - WL[a])) for n_, a in zip(NL, aL)]
    lamL = 6 * np.exp(1.5 * np.sin(9 * WL[:, 0]) * np.cos(7 * WL[:, 1])); YL = gL.poisson(lamL).astype(float)
    y1L = gL.binomial(YL.astype(int), .5).astype(float); xL1, xL2 = 2 * y1L, 2 * (YL - y1L)
    idsL = np.array([f"L{i}" for i in range(900)])
    t_ok = family_table(NL, QL, xL1, xL2, YL, np.ones(900), idsL, aL, poolL, 49, 11)
    n_sup = sum(r["status"] == "MEASUREMENT_SUPPORTED" for r in t_ok["anchors"]); assert t_ok["family"] == "OK" and n_sup >= 1, (t_ok["family"], n_sup)
    real_diag = anchor_diag
    try:
        for bad_log in (float("nan"), float("inf"), -float("inf")):
            anchor_diag = lambda *a, _b=bad_log, **k: dict(real_diag(*a, **k), logsb=_b)
            t_bad = family_table(NL, QL, xL1, xL2, YL, np.ones(900), idsL, aL, poolL, 49, 11)
            assert t_bad["counts"]["MEASUREMENT_SUPPORTED"] == 0 and all(t_bad["threshold_curve"][k_] == 0 for k_ in t_bad["threshold_curve"])
            for r_ok, r_bad in zip(t_ok["anchors"], t_bad["anchors"]):
                if r_ok["status"] == "MEASUREMENT_SUPPORTED": assert r_bad["reasons"] == ["count_log_undefined"]
    finally:
        anchor_diag = real_diag
    # amplitude admissible region: a >= 0, no upper bound
    Wm = g.normal(size=(3000, 2)); em = g.lognormal(0, .3, 3000); fm = f_nonlin(Wm)
    p, mt = match(lambda q_: q_[1] + q_[0] * fm, em, .25, .19, [np.array([a0, b0]) for a0 in (.2, .8, 1.5) for b0 in (-3, -1)])
    assert p is None or p[0] >= 0
    tgt_det = (1 - np.exp(-1.0)) + .01
    idsN = np.array([f"k{i}" for i in range(600)]); poolN = np.arange(300, 600); aN = poolN[:3]
    WN = g.normal(size=(600, 2)); NN = [knn(WN, poolN, a, 120) for a in aN]; QN = [M.basis(M.q2_design(WN[n_] - WN[a])) for n_, a in zip(NN, aN)]
    zero = np.zeros(600); YN = np.zeros(600)
    t_all = family_table(NN, QN, zero, zero, YN, np.ones(600), idsN, aN, poolN, 30, 1)           # all-degenerate family
    assert t_all["family"] == "NO_MEMBERS" and t_all["counts"]["UNRESOLVED"] == 3 and all(r["status"] == "UNRESOLVED" for r in t_all["anchors"])
    xmix = zero.copy(); xmix[NN[1]] = g.poisson(3, len(NN[1])); xmix[NN[2]] = g.poisson(3, len(NN[2]))
    xm1 = np.floor(xmix / 2); xm2 = xmix - xm1
    only0 = np.setdiff1d(NN[0], np.r_[NN[1], NN[2]]); xm1[only0] = 0; xm2[only0] = 0
    t_mix = family_table(NN, QN, xm1, xm2, xmix, np.ones(600), idsN, aN, poolN, 30, 2)            # mixed degenerate
    assert t_mix["family"] in ("OK", "FAMILY_BOOTSTRAP_FAILURE")
    t_emp = family_table(NN, QN, xm1, xm2, xmix, np.ones(600), idsN, aN, poolN, 30, 3, excluded={0: "INCOMPLETE_PANEL", 1: "INCOMPLETE_PANEL", 2: "INCOMPLETE_PANEL"})
    assert t_emp["family"] == "EMPTY" and t_emp["source_pass"] is None and all(r["status"] == "INCOMPLETE_PANEL" for r in t_emp["anchors"])
    # forced optimizer stall: 1 outer iteration cannot reach the final barrier -> success False
    _, fit_stall = fit_q2_rate_constrained(fr["W"], np.flatnonzero(fr["train"]), fr["focal"], fr["e"], outer=1)
    assert not fit_stall["success"]
    # invalid fixture
    try:
        run(frt, "F1_TYPO", 1, 5); raise AssertionError("invalid fixture accepted")
    except ValueError: pass
    em1 = np.ones(4000); fm1 = np.r_[np.zeros(2000), np.ones(2000)]
    pu, mu_ = match(lambda q_: q_[1] + 0 * q_[0] * fm1, em1, 1.0, tgt_det, [np.array([.5, 0.0])])
    assert pu is None and "rank_deficient_jacobian" in mu_["attempts"][0]["rejection_reasons"]                  # unidentified amplitude
    from types import SimpleNamespace as NS_
    global _LSQ
    def fake_lsq(endpoints):
        it = iter(endpoints)
        return lambda *a, **k: next(it)
    good = NS_(x=np.array([.7, -1.0]), jac=np.array([[1.0, .2], [.1, 1.0]]), success=True, status=1, optimality=1e-10, active_mask=np.array([0, 0]))
    singular = NS_(x=np.array([.7, -1.0]), jac=np.zeros((2, 2)), success=True, status=1, optimality=0.0, active_mask=np.array([0, 0]))
    illcond = NS_(x=np.array([.7, -1.0]), jac=np.array([[1.0, 1.0], [1.0, 1.0 + 1e-12]]), success=True, status=1, optimality=0.0, active_mask=np.array([0, 0]))
    nanopt = NS_(x=np.array([.7, -1.0]), jac=np.eye(2), success=True, status=1, optimality=float("nan"), active_mask=np.array([0, 0]))
    fn_ = lambda q_: q_[1] + q_[0] * fm1; tm_, td_ = moments(np.exp(-1.0 + .7 * fm1), em1)
    try:
        for bad_ep in (singular, illcond, nanopt):
            _LSQ = fake_lsq([bad_ep]); pz, mz = match(fn_, em1, tm_, td_, [np.array([0.0, 0.0])])
            assert pz is None and mz["status"] == "CALIBRATION_UNRESOLVED" and mz["attempts"][0]["admissible"] is False and mz["attempts"][0]["solver_valid"] is False, mz["attempts"][0]
        import itertools
        for perm in itertools.permutations([good, singular, illcond, nanopt]):
            _LSQ = fake_lsq(list(perm)); pz, mz = match(fn_, em1, tm_, td_, [np.array([0.0, 0.0])] * 4)
            assert mz["status"] == "MATCHED" and np.allclose(pz, [.7, -1.0])
        eqA = NS_(x=np.array([.7, -1.0]), jac=np.eye(2), success=True, status=1, optimality=0.0, active_mask=np.array([0, 0]))
        eqB = NS_(x=np.array([.7 + 1e-9, -1.0]), jac=np.eye(2), success=True, status=1, optimality=0.0, active_mask=np.array([0, 0]))
        sel, hashes = [], []
        for order_ in ([eqA, eqB], [eqB, eqA], [eqB, eqA, eqB], [eqA, eqA, eqB]):
            _LSQ = fake_lsq(order_); pz, mz = match(fn_, em1, tm_, td_, [np.array([0.0, 0.0])] * len(order_)); sel.append(pz.tolist())
            lam_ = np.exp(fn_(pz)); hashes.append(SEL.raw_hash(lam_, np.random.default_rng(1).poisson(em1 * lam_))["hex"])
        assert all(x_ == sel[0] for x_ in sel) and len(set(hashes)) == 1, (sel, hashes)                   # parameters + generator hash invariant
        n4 = len(em1)
        bad_ll = [lambda q_: -1.0, lambda q_: np.full((n4, 1), -1.0), lambda q_: np.full((1, n4), -1.0), lambda q_: np.zeros(0),
                  lambda q_: np.full(n4 - 1, -1.0), lambda q_: np.full(n4, np.inf), lambda q_: (_ for _ in ()).throw(ValueError("x"))]
        tmb, tdb = moments(np.exp(np.full(n4, -1.0)), em1)
        for bl in bad_ll:
            _LSQ = fake_lsq([good]); pz, mz = match(bl, em1, tmb, tdb, [np.array([0.0, 0.0])])
            assert pz is None and mz["status"] == "CALIBRATION_UNRESOLVED" and set(mz["attempts"][0]) == set(SEL.SCHEMA)
        for order_ in (["bad", "good"], ["good", "bad"], ["bad", "good", "bad"]):
            # malformed shape injected via the endpoint parameters (wrong-length x) interleaved with
            # a valid endpoint
            eps_ = [NS_(x=np.array([.7, -1.0, 0.0]), jac=np.eye(2), success=True, status=1, optimality=0.0, active_mask=np.array([0, 0])) if o == "bad" else good for o in order_]
            _LSQ = fake_lsq(eps_); pz, mz = match(fn_, em1, tm_, td_, [np.array([0.0, 0.0])] * len(eps_))
            assert mz["status"] == "MATCHED" and np.allclose(pz, [.7, -1.0])
        for field in ("success", "status"):
            for tp, ep_x, fn_m, st in ((True, np.array([.7, -1.0]), fn_, [np.array([0.0, 0.0])]),
                                        (False, np.array([-1.0]), (lambda q_: np.full(len(em1), q_[0])), [np.array([0.0])])):
                class FlipEP:
                    def __init__(self): self._n = 0; self._b = NS_(x=ep_x, jac=np.eye(2)[:, :len(ep_x)] if len(ep_x) == 1 else np.eye(2), success=True, status=1, optimality=0.0, active_mask=np.zeros(len(ep_x)))
                    def __getattr__(self, name, _f=field):
                        if name == _f:
                            self._n += 1
                            return (False if _f == "success" else 0) if self._n == 1 else (True if _f == "success" else 1)
                        return getattr(self._b, name)
                tmf, tdf = moments(np.exp(fn_m(ep_x)), em1)
                _LSQ = fake_lsq([FlipEP()]); pz, mz = match(fn_m, em1, tmf, tdf, st, two_param=tp)
                assert pz is None and mz["attempts"][0]["admissible"] is False, (field, tp)
        _LSQ = fake_lsq([good]); pz, mz = match(fn_, em1, tm_, td_, [["bad"], [np.nan, 0.0], np.array([0.0, 0.0])])
        assert [a_["rejection_reasons"] for a_ in mz["attempts"][:2]] == [["invalid_start"], ["invalid_start"]] and mz["status"] == "MATCHED"
        import json as _json
        for tp, ep_x, fn_m, st0 in ((True, np.array([.7, -1.0]), fn_, np.array([0.0, 0.0])),
                                     (False, np.array([-1.0]), (lambda q_: np.full(len(em1), q_[0])), np.array([0.0]))):
            k_ = len(ep_x); jac0 = np.eye(2)[:, :k_]
            base_ = dict(x=ep_x, jac=jac0, success=True, status=1, optimality=0.0, active_mask=np.zeros(k_, int))
            valid_ = NS_(**base_)
            muts = dict(success="False", status="1", optimality="0", active_mask=np.array(["0"] * k_))
            muts2 = [("success", 1), ("status", 1.0), ("status", True), ("optimality", 0j), ("optimality", object()),
                     ("active_mask", np.zeros(k_) + 0j), ("active_mask", np.zeros(k_, bool)), ("active_mask", np.zeros(k_, object)),
                     ("x", ep_x.astype(str)), ("x", ep_x + 0j), ("x", ep_x.astype(object)), ("jac", jac0 + 0j), ("jac", jac0.astype(str))]
            cases = [NS_(**dict(base_, **muts))] + [NS_(**dict(base_, **{f_: v_})) for f_, v_ in muts2]
            tmf, tdf = moments(np.exp(fn_m(ep_x)), em1)
            for bad_ in cases:
                _LSQ = fake_lsq([bad_]); pz, mz = match(fn_m, em1, tmf, tdf, [st0], two_param=tp)
                assert pz is None and mz["attempts"][0]["admissible"] is False and set(mz["attempts"][0]) == set(SEL.SCHEMA), (tp, mz["attempts"][0]["rejection_reasons"])
                _json.dumps(mz["attempts"], allow_nan=False)
                for order_ in ([bad_, valid_], [valid_, bad_], [bad_, valid_, bad_]):
                    _LSQ = fake_lsq(order_); pz, mz = match(fn_m, em1, tmf, tdf, [st0] * len(order_), two_param=tp)
                    assert mz["status"] == "MATCHED" and np.allclose(pz, ep_x), (tp, mz["status"])
            class MutEP_:
                def __init__(self, kind):
                    self.kind = kind; self._x = ep_x.copy(); self._j = jac0.copy()
                    if kind == "x": self._x[:] = np.nan
                    if kind == "jac": self._j[:] = 0.0
                    self.success, self.status, self.optimality = True, 1, 0.0
                @property
                def x(self): return self._x
                @property
                def jac(self):
                    if self.kind == "x": self._x[:] = ep_x
                    return self._j
                @property
                def active_mask(self):
                    if self.kind == "jac": self._j[:] = jac0
                    return np.zeros(k_, int)
            class RevEP_:
                def __init__(self, kind):
                    self.kind = kind; self._x = ep_x.copy(); self._j = jac0.copy(); self.success, self.status, self.optimality = True, 1, 0.0
                @property
                def x(self): return self._x
                @property
                def jac(self):
                    if self.kind == "x": self._x[:] = np.nan
                    return self._j
                @property
                def active_mask(self):
                    if self.kind == "jac": self._j[:] = 0.0
                    return np.zeros(k_, int)
            for kind in ("x", "jac"):
                _LSQ = fake_lsq([RevEP_(kind)]); pz, mz = match(fn_m, em1, tmf, tdf, [st0], two_param=tp)
                assert mz["status"] == "MATCHED" and np.allclose(pz, ep_x) and mz["attempts"][0]["admissible"] is True, (tp, kind)
                _json.dumps(mz["attempts"], allow_nan=False)
            for kind in ("x", "jac"):
                _LSQ = fake_lsq([MutEP_(kind)]); pz, mz = match(fn_m, em1, tmf, tdf, [st0], two_param=tp)
                assert pz is None and mz["attempts"][0]["admissible"] is False, (tp, kind); _json.dumps(mz["attempts"], allow_nan=False)
                for order_ in ([MutEP_(kind), valid_], [valid_, MutEP_(kind)]):
                    _LSQ = fake_lsq(order_); pz, mz = match(fn_m, em1, tmf, tdf, [st0] * 2, two_param=tp)
                    assert mz["status"] == "MATCHED" and np.allclose(pz, ep_x) and mz["attempts"][[o is valid_ for o in order_].index(False)]["admissible"] is False
            class BadRepr_:
                def __array__(self, *a, **k): raise ValueError("no array")
                def __repr__(self): raise ValueError("no repr")
            bad_starts = [np.array(["0"] * k_), np.zeros(k_, bool), np.zeros(k_) + 0j, BadRepr_()]
            _LSQ = fake_lsq([valid_]); pz, mz = match(fn_m, em1, tmf, tdf, bad_starts + [st0], two_param=tp)
            assert [a_["rejection_reasons"] for a_ in mz["attempts"][:4]] == [["invalid_start"]] * 4 and mz["status"] == "MATCHED"
            _json.dumps(mz["attempts"], allow_nan=False)
        # schema completeness: solver exception and nonfinite Jacobian attempts carry every field
        def boom(*a, **k): raise RuntimeError("injected")
        _LSQ = boom; pz, mz = match(fn_, em1, tm_, td_, [np.array([0.0, 0.0])])
        assert pz is None and mz["attempts"][0]["rejection_reasons"] == ["solver_exception"] and set(mz["attempts"][0]) == set(SEL.SCHEMA)
        nanjac = NS_(x=np.array([.7, -1.0]), jac=np.array([[np.nan, 0.0], [0.0, 1.0]]), success=True, status=1, optimality=0.0, active_mask=np.array([0, 0]))
        _LSQ = fake_lsq([nanjac]); pz, mz = match(fn_, em1, tm_, td_, [np.array([0.0, 0.0])])
        at = mz["attempts"][0]
        assert pz is None and "nonfinite_jacobian" in at["rejection_reasons"] and at["normalised_residual"] is not None and set(at) == set(SEL.SCHEMA) and at["admissible"] is False
    finally:
        _LSQ = least_squares
    # (b) unique approximate optimum: match() must return the minimiser of the exact normalised
    # residual (independent 1-D check)
    pb, mb = match(lambda q_: np.full(4000, q_[0]), em1, 1.0, tgt_det, [np.array([0.0]), np.array([-.3])], two_param=False)   # identified 1-D
    grid_b = np.linspace(-.3, .3, 60001)
    dgrid = [np.hypot((np.exp(bb) - 1) / .10, ((1 - np.exp(-np.exp(bb))) - tgt_det) / .02) for bb in grid_b]
    assert mb["status"] == "MATCHED" and mb["normalised_residual"] <= min(dgrid) + 1e-4 and mb["normalised_residual"] > 1e-3
    # (c) equal-best distinct optima -> ambiguous
    pc, mc = match(lambda q_: q_[1] + (q_[0] - 1.0) ** 2 * fm1, em1, *moments(np.exp(-1.0 + .25 * fm1), em1),
                   [np.array([.5, -1.0]), np.array([1.5, -1.0])])
    assert mc["status"] == "CALIBRATION_AMBIGUOUS" and len(mc["equal_best"]) == 2
    # (d) solver exhaustion / nonconvergence -> UNRESOLVED (non-finite objective)
    pd_, md_ = match(lambda q_: np.full(4000, np.nan) + q_[0], em1, 1.0, .5, [np.array([.5, 0.0])])
    assert pd_ is None and md_["status"] == "CALIBRATION_UNRESOLVED"
    # (e) bracket failure, bisection failure, genuinely unreachable grid (injected solver/oracle),
    # complete histories
    okm = (np.array([.5, -1.0]), dict(status="MATCHED"))
    failm = (None, dict(status="CALIBRATION_UNMATCHED"))
    r1 = calibrate_world(frt, "LADDER_20", 1, Ntt, Qtt, trt, [7], _hooks=dict(solve=lambda sg: failm if sg >= .5 else okm, tenth=lambda sg: sg * .1))
    assert r1[4]["reason"] == "matching_failed_during_bracket" and r1[4]["bracket"][-1]["sigma"] == .5 and r1[4]["bracket"][-1]["tenth"] is None
    r2 = calibrate_world(frt, "LADDER_20", 1, Ntt, Qtt, trt, [7], _hooks=dict(solve=lambda sg: failm if .68 < sg < .70 else okm, tenth=lambda sg: sg * .3))
    assert r2[4]["reason"] == "matching_failed_during_bisection" and len(r2[4]["bisection"]) >= 1 and "last_match" in r2[4]
    r3 = calibrate_world(frt, "LADDER_20", 1, Ntt, Qtt, trt, [7], _hooks=dict(solve=lambda sg: okm, tenth=lambda sg: .01))
    assert r3[4]["reason"] == "target_unreachable_within_grid" and len(r3[4]["bracket"]) == len(SIGMA_GRID)
    for rr_ in (r1, r2, r3):
        assert {"reason", "sigma", "bracket", "bisection", "last_match"} <= set(rr_[4])                   # uniform failure schema
    r4 = calibrate_world(frt, "LADDER_20", 1, Ntt, Qtt, trt, [7], _hooks=dict(solve=lambda sg: okm, tenth=lambda sg: min(sg * .3, .6)))
    assert r4[0] is not None and isinstance(r4[4]["bisection"], list) and len(r4[4]["bisection"]) >= 2 and r4[4]["bisection"][-1]["final_verification"]
    # (f) metric names validated
    try: run(frt, "F1_NONLIN", 3, 5, metric="cosine"); raise AssertionError("bad metric accepted")
    except ValueError: pass
    frt["Wpca"] = frt["W"] * np.array([3.0, 1.0, .2])
    rw = run(frt, "F1_NONLIN", 3, 9, metric="whitened"); rv = run(frt, "F1_NONLIN", 3, 9, metric="variance")
    assert rw["generator_world_hash"] == rv["generator_world_hash"] and rw["planted"] == rv["planted"]
    assert rw["anchor_ids_analysis"] == rv["anchor_ids_analysis"]
    # F2 power fixture exists and calibrates the ladder with the shared program present
    assert "F2_MIXED_P20" in FIXTURES and LADDER["F2_MIXED_P20"] == .20 and F2_DEPTH["F2_MIXED_P20"] == F2_DEPTH["F2_MIXED"]
    SEL.selftest()
    print("calib3 selftest PASS (no-member/mixed/empty families, optimizer stall, invalid fixture, bracket class, F2 power; split/roster invariance, constrained positive Q2 span, analysis-pool bootstrap, union family failure, "
          "program-redraw concentration, final-sigma ladder verification, per-anchor panels + curve, status labels, a>=0)")


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        selftest(); sys.exit()
    if sys.argv[1] == "--freeze-frame":
        fr_ = load_frame(sys.argv[2], freeze=True); print("frozen", sys.argv[2], fr_["hashes"]["frozen_frame"]); sys.exit()
    exp_ = dict(kv.split("=") for kv in os.environ.get("GB_EXPECT_HASHES", "").split(",") if "=" in kv)
    got_ = dict(calib3=SEL.file_sha256_16(__file__), select=SEL.file_sha256_16(SEL.__file__), source=SEL.file_sha256_16(M.__file__))
    if os.environ.get("GB_SKIP_PREFLIGHT") == "1": print("preflight skipped (public release)")
    elif set(exp_) != set(got_) or any(exp_[k] != got_[k] for k in got_):
        sys.exit(f"RUNTIME_HASH_MISMATCH expected {exp_} got {got_}")
    outdir, ds, fixture, world = sys.argv[1:5]; B = int(sys.argv[5]) if len(sys.argv) > 5 else 999
    metric = sys.argv[6] if len(sys.argv) > 6 else "whitened"
    os.makedirs(outdir, exist_ok=True)
    r = run(load_frame(ds), fixture, int(world), B, f"{outdir}/{ds}_{fixture}_w{world}_B{B}_{metric}.json", metric=metric)
    fo = r.get("focal") or {}; co = r.get("conditioned") or {}
    print(ds, fixture, world, r["decision_status"], (r["match"] or {}).get("status"), "focal", fo.get("counts"), "cond", co.get("threshold_curve"),
          "conc", (r.get("concentration") or {}).get("status"), f"{r['runtime_s']:.0f}s")
