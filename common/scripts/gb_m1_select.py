#!/usr/bin/env python
"""Small calibration-selection kernel (internal review r123 rewrite recommendation; PI 2026-09-14 §2). Pure functions only:
endpoint evaluation, equal-best selection, raw-byte hashing. Generators, thresholds, fixtures, concentration, family membership and
bootstrap mathematics live elsewhere and are unchanged.

evaluate_endpoint(r, start, loglam_fn, e, tmean, tdet, n_params)
    uniform record: start, params, success, status, optimality, active_mask, mean, det, jac_rank, jac_cond, normalised_residual,
    solver_valid (converged + finite params/Jacobian/optimality/active mask/moments + full column rank + cond <= 1e8 + finite residual),
    in_bands (|mean/mean*-1| <= .10 and |det-det*| <= .02), admissible (= solver_valid AND in_bands), rejection_reasons, error
select_equal_best(records, target_mean, target_det, two_param)
    admissible = records with admissible is True; none solver_valid -> CALIBRATION_UNRESOLVED; solver_valid but none in bands ->
    CALIBRATION_UNMATCHED; equal-best = admissible with d <= d_min + 1e-4, ordered by (d, UNROUNDED parameter tuple); material
    clusters (symmetric atol .05*(1+max|x|,|y|)); representative = first in canonical order; >1 cluster -> CALIBRATION_AMBIGUOUS
raw_hash(*objs)
    SHA-256 over typed frames: bytes -> b"bytes"+len+payload; ndarray -> b"ndarray"+dtype.str+shape+C-order bytes; str -> utf-8 frame.
    Returns dict(algorithm="sha256-typed-frames-v1", hex16=..., hex=...)
file_sha256_16(path)  plain SHA-256 of raw file bytes (first 16 hex), identical to `sha256sum | cut -c1-16`
"""
import hashlib
import struct
import numpy as np

SCHEMA = ("start", "params", "success", "status", "optimality", "active_mask", "mean", "det", "jac_rank", "jac_cond",
          "normalised_residual", "solver_valid", "in_bands", "admissible", "rejection_reasons", "error")
COND_MAX, EQ_TOL, BAND_MEAN, BAND_DET = 1e8, 1e-4, .10, .02
HASH_ALG = "sha256-typed-frames-v1"
FILE_HASH_ALG = "sha256"
_RANK, _COND = np.linalg.matrix_rank, np.linalg.cond                             # indirection so selftests can force LinAlgError


def _real_array(v, shape):
    """internal review r129 A1/B1: strict real numeric array (int/uint/float dtype; no bool, complex, string or object) of exact shape, else None"""
    try:
        a = np.array(v, copy=True)
    except Exception:
        return None
    if a.dtype.kind not in "iuf" or a.shape != shape: return None
    return a.astype(float)


def _is_bool(v): return isinstance(v, (bool, np.bool_))
def _is_int(v): return isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_))
def _is_real(v): return isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_))


def safe_start(start):
    """JSON-safe representation of a start that never raises (internal review r128 B1, r129 B1)"""
    try:
        a = np.array(start, copy=True)
        if a.dtype.kind in "iuf":
            return [float(v) if np.isfinite(v) else None for v in a.astype(float).ravel()]
    except Exception:
        pass
    try:
        return "unparsed:" + str(repr(start))[:200]
    except Exception:
        return "unrepresentable_start"


def valid_start(start, n_params):
    a = _real_array(start, (n_params,))
    return a if (a is not None and np.all(np.isfinite(a))) else None


def blank(start):
    rec = {k: None for k in SCHEMA}
    rec.update(start=safe_start(start), solver_valid=False, in_bands=False, admissible=False, rejection_reasons=[])
    return rec


def invalid_start_record(start):
    rec = blank(start); rec.update(rejection_reasons=["invalid_start"]); return rec


def moments(lam, e):
    return float((e * lam).mean()), float((1 - np.exp(-e * lam)).mean())


def evaluate_endpoint(r, start, loglam_fn, e, tmean, tdet, n_params, n_resid=2):
    """never raises: any evaluation / linear-algebra failure becomes a complete non-admissible record"""
    rec = blank(start)
    try:
        return _evaluate(rec, r, loglam_fn, e, tmean, tdet, n_params, n_resid)
    except Exception as err:                                                     # internal review r124 A2: uniform record instead of abort
        rec.update(solver_valid=False, in_bands=False, admissible=False, error=type(err).__name__,
                   rejection_reasons=sorted(set((rec.get("rejection_reasons") or []) + ["evaluation_exception"])))
        return rec


def _evaluate(rec, r, loglam_fn, e, tmean, tdet, n_params, n_resid):
    # internal review r128 A1: read every endpoint attribute exactly once into snapshots; internal review r129 A1: strict types, no lossy coercion.
    # Serialised fields are built only from validated values; any invalid type is a named rejection.
    # internal review r131 A1: each array is copied immediately when its property is read, before any later getter can mutate it
    x = _real_array(r.x, (n_params,))
    jac = _real_array(r.jac, (n_resid, n_params))
    am = _real_array(r.active_mask, (n_params,))
    raw_success = r.success; raw_status = r.status; raw_opt = r.optimality
    success_snap = bool(raw_success) if _is_bool(raw_success) else None
    status_snap = int(raw_status) if _is_int(raw_status) else None
    opt_snap = float(raw_opt) if _is_real(raw_opt) else None
    rec.update(params=None if x is None else [float(v) if np.isfinite(v) else None for v in x], success=success_snap, status=status_snap,
               optimality=opt_snap if (opt_snap is not None and np.isfinite(opt_snap)) else None,
               active_mask=None if am is None else [float(v) if np.isfinite(v) else None for v in am])
    reasons = []
    if success_snap is None: reasons.append("invalid_type_success")
    if status_snap is None: reasons.append("invalid_type_status")
    if not (success_snap is True and status_snap is not None and status_snap > 0): reasons.append("not_converged")
    shape_x = x is not None
    if not shape_x: reasons.append("params_wrong_shape_or_type")
    finite_x = shape_x and bool(np.all(np.isfinite(x)))
    if shape_x and not finite_x: reasons.append("nonfinite_params")
    shape_j = jac is not None
    if not shape_j: reasons.append("jacobian_wrong_shape_or_type")
    finite_j = shape_j and bool(np.all(np.isfinite(jac)))
    if shape_j and not finite_j: reasons.append("nonfinite_jacobian")
    if am is None: reasons.append("active_mask_wrong_shape_or_type")
    elif not np.all(np.isfinite(am)): reasons.append("nonfinite_active_mask")
    if opt_snap is None or not np.isfinite(opt_snap): reasons.append("nonfinite_or_invalid_optimality")
    mm = dd = float("nan")
    ev = np.asarray(e, float)
    if ev.ndim != 1 or ev.size == 0: reasons.append("exposure_wrong_shape")
    if finite_x and "exposure_wrong_shape" not in reasons:
        try:
            ll = np.asarray(loglam_fn(x), float)
        except Exception as err:
            ll = None; reasons.append("loglam_exception"); rec["error"] = type(err).__name__
        if ll is not None:
            if ll.shape != ev.shape: reasons.append("loglam_wrong_shape")                  # internal review r126 A1: one rate per cell, no broadcasting
            elif not np.all(np.isfinite(ll)): reasons.append("nonfinite_loglam")
            else:
                with np.errstate(all="ignore"): mm, dd = moments(np.exp(ll), ev)
    if np.isfinite(mm) and np.isfinite(dd):
        rec.update(mean=mm, det=dd)
        rn = float(np.hypot((mm / tmean - 1) / BAND_MEAN, (dd - tdet) / BAND_DET))
        if np.isfinite(rn):
            rec["normalised_residual"] = rn
            rec["in_bands"] = bool(abs(mm / tmean - 1) <= BAND_MEAN and abs(dd - tdet) <= BAND_DET)
        else:
            reasons.append("nonfinite_residual")
    elif not any(r_ in reasons for r_ in ("loglam_wrong_shape", "nonfinite_loglam", "loglam_exception", "exposure_wrong_shape", "nonfinite_params", "params_wrong_shape_or_type")):
        reasons.append("nonfinite_moments")
    else:
        reasons.append("moments_not_computed")
    if finite_j:
        try:
            rank = int(_RANK(jac)); rec["jac_rank"] = rank
            if rank < n_params:
                rec["jac_cond"] = "inf"; reasons.append("rank_deficient_jacobian")
            else:
                cond = float(_COND(jac)); rec["jac_cond"] = cond if np.isfinite(cond) else "inf"
                if not (np.isfinite(cond) and cond <= COND_MAX): reasons.append("ill_conditioned_jacobian")
        except np.linalg.LinAlgError:
            reasons.append("linalg_failure")
    rec["solver_valid"] = not reasons
    rec["admissible"] = bool(rec["solver_valid"] and rec["in_bands"])
    if rec["solver_valid"] and not rec["in_bands"]: reasons.append("outside_matching_bands")
    rec["rejection_reasons"] = reasons
    return rec


def exception_record(start, err):
    rec = blank(start); rec.update(error=type(err).__name__, rejection_reasons=["solver_exception"]); return rec


def _same(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    return bool(np.all(np.abs(x - y) <= .05 * (1 + np.maximum(np.abs(x), np.abs(y)))))


def select_equal_best(records, target_mean, target_det, two_param=True):
    base = dict(target_mean=target_mean, target_det=target_det, attempts=records)
    def finite_res(r):
        v = r.get("normalised_residual")
        return isinstance(v, (int, float)) and not isinstance(v, bool) and bool(np.isfinite(v))
    valid = [r for r in records if r.get("solver_valid") is True and finite_res(r)]          # internal review r124 A1: finiteness enforced here too
    if not valid:
        return None, dict(base, status="CALIBRATION_UNRESOLVED", reason="no_solver_valid_endpoint")
    adm = [r for r in valid if r.get("admissible") is True and r.get("in_bands") is True]
    if not adm:
        return None, dict(base, status="CALIBRATION_UNMATCHED")
    dmin = min(r["normalised_residual"] for r in adm)
    eq = sorted([r for r in adm if r["normalised_residual"] <= dmin + EQ_TOL],
                key=lambda r: (r["normalised_residual"], tuple(float(v) for v in r["params"])))
    clusters = []
    for r in eq:
        if not any(_same(r["params"], c["params"]) for c in clusters): clusters.append(r)
    others = [r for r in adm if not any(_same(r["params"], c["params"]) for c in clusters)]
    rep = clusters[0]
    x = np.asarray(rep["params"], float)
    return x, dict(base, status="MATCHED" if len(clusters) == 1 else "CALIBRATION_AMBIGUOUS", matched_mean=rep["mean"], matched_det=rep["det"],
                   normalised_residual=rep["normalised_residual"], boundary_solution=bool(two_param and x[0] <= 1e-9),
                   equal_best=clusters, other_tolerance_band_solutions=others)


def raw_hash(*objs):
    h = hashlib.sha256()
    for o in objs:
        if isinstance(o, (bytes, bytearray)):
            h.update(b"bytes"); h.update(struct.pack("<Q", len(o))); h.update(bytes(o))
        elif isinstance(o, np.ndarray):
            a = np.ascontiguousarray(o); hdr = (a.dtype.str + "|" + ",".join(map(str, a.shape))).encode()
            h.update(b"ndarray"); h.update(struct.pack("<Q", len(hdr))); h.update(hdr); h.update(a.tobytes())
        elif isinstance(o, str):
            b = o.encode("utf-8"); h.update(b"str"); h.update(struct.pack("<Q", len(b))); h.update(b)
        else:
            raise TypeError(f"raw_hash: unsupported type {type(o)}")
    hx = h.hexdigest()
    return dict(algorithm=HASH_ALG, hex16=hx[:16], hex=hx)


def file_sha256_16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    return h.hexdigest()[:16]


def selftest():
    import itertools, os, tempfile
    from types import SimpleNamespace as NS
    e = np.ones(4000); f = np.r_[np.zeros(2000), np.ones(2000)]
    fn = lambda q: q[1] + q[0] * f
    tm, td = moments(np.exp(-1.0 + .7 * f), e)
    def ep(x, jac=np.eye(2), success=True, status=1, opt=0.0, am=(0, 0)):
        return NS(x=np.asarray(x, float), jac=np.asarray(jac, float), success=success, status=status, optimality=opt, active_mask=np.asarray(am))
    good = evaluate_endpoint(ep([.7, -1.0]), [0, 0], fn, e, tm, td, 2)
    assert good["solver_valid"] and good["in_bands"] and good["admissible"] and set(good) == set(SCHEMA)
    bad = dict(
        singular=evaluate_endpoint(ep([.7, -1.0], jac=np.zeros((2, 2))), [0, 0], fn, e, tm, td, 2),
        illcond=evaluate_endpoint(ep([.7, -1.0], jac=[[1, 1], [1, 1 + 1e-12]]), [0, 0], fn, e, tm, td, 2),
        nanopt=evaluate_endpoint(ep([.7, -1.0], opt=float("nan")), [0, 0], fn, e, tm, td, 2),
        nanjac=evaluate_endpoint(ep([.7, -1.0], jac=[[np.nan, 0], [0, 1]]), [0, 0], fn, e, tm, td, 2),
        notconv=evaluate_endpoint(ep([.7, -1.0], success=False, status=0), [0, 0], fn, e, tm, td, 2),
        outband=evaluate_endpoint(ep([.0, -3.0]), [0, 0], fn, e, tm, td, 2),
        exc=exception_record([0, 0], RuntimeError("x")))
    for k, r in bad.items():
        assert set(r) == set(SCHEMA) and r["admissible"] is False, (k, r)
    assert bad["outband"]["solver_valid"] is True and bad["outband"]["in_bands"] is False and "outside_matching_bands" in bad["outband"]["rejection_reasons"]
    assert bad["nanjac"]["normalised_residual"] is not None and bad["exc"]["rejection_reasons"] == ["solver_exception"]
    # valid endpoint wins under every permutation of invalid companions
    for perm in itertools.permutations([good] + list(bad.values())):
        x, m = select_equal_best(list(perm), tm, td); assert m["status"] == "MATCHED" and np.allclose(x, [.7, -1.0])
    # missing admission / only out-of-band / none valid
    miss = dict(good); miss.pop("admissible")
    assert select_equal_best([miss], tm, td)[1]["status"] == "CALIBRATION_UNMATCHED"
    assert select_equal_best([bad["outband"]], tm, td)[1]["status"] == "CALIBRATION_UNMATCHED"
    assert select_equal_best([bad["singular"], bad["exc"]], tm, td)[1]["status"] == "CALIBRATION_UNRESOLVED"
    # exact-residual ties separated by < 5e-13: identical selection and world hash under reversal and duplication
    a = dict(good, params=[.7, -1.0], normalised_residual=.01); b = dict(good, params=[.7 + 3e-13, -1.0], normalised_residual=.01)
    sels = []
    for order in ([a, b], [b, a], [b, a, b], [a, a, b], [b, b, a]):
        x, m = select_equal_best(order, tm, td); lam = np.exp(fn(x))
        sels.append((tuple(x.tolist()), raw_hash(lam, np.random.default_rng(1).poisson(e * lam))["hex"]))
    assert len(set(sels)) == 1, sels
    # materially distinct equal-best -> AMBIGUOUS; inferior in-band near-miss -> MATCHED with other solution recorded
    c = dict(good, params=[.0, -.6], normalised_residual=.01 + 5e-5)
    assert select_equal_best([a, c], tm, td)[1]["status"] == "CALIBRATION_AMBIGUOUS"
    near = dict(good, params=[.0, -.6], normalised_residual=.7)
    x, m = select_equal_best([near, a], tm, td); assert m["status"] == "MATCHED" and len(m["other_tolerance_band_solutions"]) == 1
    # internal review r124 injected malformed / nonfinite controls: never admitted, never crash, complete schema
    for v in (float("nan"), float("inf"), -float("inf"), None, "0.1", True):
        rr = dict(good, normalised_residual=v)
        xz, mz = select_equal_best([rr], tm, td); assert xz is None and mz["status"] == "CALIBRATION_UNRESOLVED", (v, mz["status"])
        xz, mz = select_equal_best([rr, good], tm, td); assert mz["status"] == "MATCHED" and np.allclose(xz, [.7, -1.0])
    malformed = [ep([.7, -1.0], jac=np.ones((2, 3))), ep([.7, -1.0], jac=np.ones(2)), ep([.7, -1.0], jac=np.zeros((0, 2))),
                 ep([.7, -1.0], am=(0, 0, 0)), ep([.7, -1.0, 0.0]), ep([.7, -1.0], jac=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])]
    class Boom:                                                                   # forced failure while reading attributes
        x = np.array([.7, -1.0]); jac = np.eye(2); success = True; status = 1; active_mask = np.array([0, 0])
        @property
        def optimality(self): raise np.linalg.LinAlgError("forced")
    malformed.append(Boom())
    recs = [evaluate_endpoint(m_, [0, 0], fn, e, tm, td, 2) for m_ in malformed]
    for rr in recs:
        assert set(rr) == set(SCHEMA) and rr["admissible"] is False and rr["solver_valid"] is False, rr["rejection_reasons"]
    for perm in itertools.permutations(recs[:4] + [good]):
        xz, mz = select_equal_best(list(perm), tm, td); assert mz["status"] == "MATCHED" and np.allclose(xz, [.7, -1.0])
    assert select_equal_best(recs, tm, td)[1]["status"] == "CALIBRATION_UNRESOLVED"
    # internal review r126: malformed loglam outputs, loglam exceptions, forced rank / cond LinAlgError, condition boundary
    n_ = len(e)
    bad_fns = dict(scalar=lambda q: -1.0, col=lambda q: np.full((n_, 1), -1.0), row=lambda q: np.full((1, n_), -1.0),
                   empty=lambda q: np.zeros(0), short=lambda q: np.full(n_ - 1, -1.0), nonfinite=lambda q: np.full(n_, np.nan),
                   raises=lambda q: (_ for _ in ()).throw(ValueError("boom")))
    tm0, td0 = moments(np.exp(np.full(n_, -1.0)), e)
    for k, bf in bad_fns.items():
        rr = evaluate_endpoint(ep([.7, -1.0]), [0, 0], bf, e, tm0, td0, 2)
        assert set(rr) == set(SCHEMA) and rr["solver_valid"] is False and rr["admissible"] is False, (k, rr["rejection_reasons"])
    global _RANK, _COND
    def raise_linalg(*a, **k): raise np.linalg.LinAlgError("forced")
    try:
        _RANK = raise_linalg; rr = evaluate_endpoint(ep([.7, -1.0]), [0, 0], fn, e, tm, td, 2)
        assert "linalg_failure" in rr["rejection_reasons"] and rr["admissible"] is False and set(rr) == set(SCHEMA)
        _RANK = np.linalg.matrix_rank; _COND = raise_linalg; rr = evaluate_endpoint(ep([.7, -1.0]), [0, 0], fn, e, tm, td, 2)
        assert "linalg_failure" in rr["rejection_reasons"] and rr["admissible"] is False and set(rr) == set(SCHEMA)
    finally:
        _RANK, _COND = np.linalg.matrix_rank, np.linalg.cond
    lo_c = evaluate_endpoint(ep([.7, -1.0], jac=np.diag([1.0, 1 / .99e8])), [0, 0], fn, e, tm, td, 2)
    hi_c = evaluate_endpoint(ep([.7, -1.0], jac=np.diag([1.0, 1 / 1.01e8])), [0, 0], fn, e, tm, td, 2)
    assert lo_c["admissible"] is True and hi_c["admissible"] is False and "ill_conditioned_jacobian" in hi_c["rejection_reasons"]
    for attr in ("x", "jac", "success", "status", "active_mask", "optimality"):
        class Raiser:
            def __getattr__(self, name, _a=attr):
                if name == _a: raise RuntimeError("attr " + name)
                return getattr(ep([.7, -1.0]), name)
        rr = evaluate_endpoint(Raiser(), [0, 0], fn, e, tm, td, 2)
        assert set(rr) == set(SCHEMA) and rr["admissible"] is False, attr
    # internal review r128: flip-on-read convergence fields cannot be admitted; malformed starts never crash
    for field in ("success", "status"):
        class Flip:
            def __init__(self): self._n = 0; self._b = ep([.7, -1.0])
            def __getattr__(self, name):
                if name == field:
                    self._n += 1
                    return (False if field == "success" else 0) if self._n == 1 else (True if field == "success" else 1)
                return getattr(self._b, name)
        rr = evaluate_endpoint(Flip(), [0, 0], fn, e, tm, td, 2)
        assert rr["admissible"] is False and rr["solver_valid"] is False and "not_converged" in rr["rejection_reasons"], (field, rr)
    for bad_start in (["bad"], [np.nan, 0.0], [0.0], [[0.0, 0.0]], object()):
        assert valid_start(bad_start, 2) is None
        rr = invalid_start_record(bad_start); assert set(rr) == set(SCHEMA) and rr["admissible"] is False
        import json as _json; _json.dumps(rr, allow_nan=False)
    # internal review r129: type-invalid endpoint fields are never admitted; records are JSON-safe; starts are strictly numeric; safe_start is total
    import json as _json
    type_bad = dict(success_str=dict(success="False"), success_int=dict(success=1), status_str=dict(status="1"), status_float=dict(status=1.0),
                    status_bool=dict(status=True), opt_str=dict(opt="0"), opt_complex=dict(opt=0j), opt_obj=dict(opt=object()),
                    am_str=dict(am=["0", "0"]), am_complex=dict(am=np.array([1j, 0j])), am_bool=dict(am=np.array([False, False])),
                    am_obj=dict(am=np.array([object(), object()], dtype=object)))
    def ep_raw(x=(.7, -1.0), jac=np.eye(2), success=True, status=1, opt=0.0, am=(0, 0)):
        return NS(x=x, jac=jac, success=success, status=status, optimality=opt, active_mask=am)
    arr_bad = dict(x_str=dict(x=["0.7", "-1.0"]), x_complex=dict(x=np.array([.7 + 0j, -1.0 + 0j])), x_bool=dict(x=np.array([True, False])),
                   x_obj=dict(x=np.array([.7, -1.0], dtype=object)), jac_str=dict(jac=[["1", "0"], ["0", "1"]]),
                   jac_complex=dict(jac=np.eye(2) + 0j), jac_obj=dict(jac=np.eye(2).astype(object)))
    for k, kw in list(type_bad.items()) + list(arr_bad.items()):
        rr = evaluate_endpoint(ep_raw(**kw), [0, 0], fn, e, tm, td, 2)
        assert set(rr) == set(SCHEMA) and rr["admissible"] is False and rr["solver_valid"] is False, (k, rr["rejection_reasons"])
        _json.dumps(rr, allow_nan=False)
        xz, mz = select_equal_best([rr], tm, td); assert xz is None, k
        xz, mz = select_equal_best([rr, good], tm, td); assert mz["status"] == "MATCHED" and np.allclose(xz, [.7, -1.0]), k
    assert evaluate_endpoint(ep_raw(x=np.array([.7, -1.0]), am=np.array([0, 0])), [0, 0], fn, e, tm, td, 2)["admissible"] is True   # plain SciPy-typed endpoint
    # internal review r131: a later getter mutating an earlier-returned array cannot change the snapshot (first read authoritative)
    def mutating(first_bad):
        class MutEP:
            def __init__(self):
                self._x = np.array([np.nan, -1.0]) if first_bad == "x" else np.array([.7, -1.0])
                self._j = np.zeros((2, 2)) if first_bad == "jac" else np.eye(2)
                self.success, self.status, self.optimality = True, 1, 0.0
            @property
            def x(self): return self._x
            @property
            def jac(self):
                if first_bad == "x": self._x[:] = [.7, -1.0]                  # repair x after it was read
                if first_bad == "valid_x_then_break": self._x[:] = [np.nan, np.nan]
                return self._j
            @property
            def active_mask(self):
                if first_bad == "jac": self._j[:] = np.eye(2)                  # repair jac after it was read
                if first_bad == "valid_jac_then_break": self._j[:] = 0.0
                return np.array([0, 0])
        return MutEP()
    for kind, expect in (("x", False), ("jac", False), ("valid_x_then_break", True), ("valid_jac_then_break", True)):
        rr = evaluate_endpoint(mutating(kind), [0, 0], fn, e, tm, td, 2)
        assert rr["admissible"] is expect and set(rr) == set(SCHEMA), (kind, rr["rejection_reasons"]); _json.dumps(rr, allow_nan=False)
        for order in ([mutating(kind), ep([.7, -1.0])], [ep([.7, -1.0]), mutating(kind)]):
            recs_m = [evaluate_endpoint(o, [0, 0], fn, e, tm, td, 2) for o in order]
            xz, mz = select_equal_best(recs_m, tm, td); assert mz["status"] == "MATCHED" and np.allclose(xz, [.7, -1.0]), kind
    class BadRepr:
        def __array__(self, *a, **k): raise ValueError("no array")
        def __repr__(self): raise ValueError("no repr")
    for bad_start in (["0.1", "0.2"], [True, False], np.array([1j, 0j]), BadRepr()):
        assert valid_start(bad_start, 2) is None, bad_start if not isinstance(bad_start, BadRepr) else "BadRepr"
        rr = invalid_start_record(bad_start); assert set(rr) == set(SCHEMA) and rr["admissible"] is False; _json.dumps(rr, allow_nan=False)
    assert valid_start([0, 1], 2) is not None and valid_start(np.array([.5, -1.0]), 2) is not None
    # hashing: typed frames distinguish dtype/shape; file hash equals raw sha256
    z = np.arange(6, dtype=np.float64)
    assert raw_hash(z)["hex"] != raw_hash(z.reshape(2, 3))["hex"] != raw_hash(z.astype(np.float32))["hex"]
    assert raw_hash(b"ab", b"c")["hex"] != raw_hash(b"a", b"bc")["hex"]
    with tempfile.NamedTemporaryFile(delete=False) as tf: tf.write(b"hello"); pth = tf.name
    assert file_sha256_16(pth) == hashlib.sha256(b"hello").hexdigest()[:16]; os.remove(pth)
    print("select kernel selftest PASS (uniform schema; solver_valid/in_bands/admissible separation; permutation invariance; "
          "sub-1e-12 exact ties with identical world hash; ambiguity; near-miss; typed raw hashing; file sha)")


if __name__ == "__main__":
    selftest()
