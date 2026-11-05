#!/usr/bin/env python
"""CERTIFIED MATCHED-REGULATOR PANEL SELECTION (internal review r156 ruling, r158 corrections).

A FAIL here terminates the GATA1 branch, so this module may only return a scientific NO-CALL when it
holds a proof. Everything else is SEARCH_UNRESOLVED, which is an implementation problem.

d_i = F_i^z - F_target^z over the frozen feature axes, standardised WITHIN the eligible pool.

STEP 0  COMMON SUPPORT. If the target lies outside the pool's range on ANY axis, no matched control
        can exist. internal review r158: this must FORCE refusal, not merely be recorded -- the previous version
        reported the unsupported axis and then carried on to certify a panel.
STEP 1  LP NEGATIVE CERTIFICATE (one-sided; can only prove IMPOSSIBILITY)
            sum_i w_i = 1,  0 <= w_i <= 1/100,  |sum_i w_i d_ij| <= 0.10
        Any integer subset of size q >= 100 induces w_i = 1/q <= 1/100, so infeasibility here
        excludes every admissible panel of every allowed size. HiGHS does NOT return usable duals for
        an infeasible LP, so the Farkas ray is CONSTRUCTED by solving Farkas' lemma as its own LP and
        then verified arithmetically. The instantiated matrices and the ray are serialized.
STEP 2  EXACT MILP.  q = sum z >= 100,  -0.10 q <= sum_i z_i d_ij <= 0.10 q.  One formulation over
        every cardinality. Randomized rounding is not permitted.
STEP 3  PLACEBO LOCALITY. Mean balance is necessary but NOT sufficient: a bimodal panel straddling
        the target satisfies every mean constraint while resembling nothing. With r_i = ||d_i||_inf,
        R* = min over balance-feasible panels of max_{i in S} r_i, obtained by ONE EXACT MILP
        minimising R subject to R >= r_i z_i (internal review r158 replaced the binary search). R* is computed
        for the target and for EVERY eligible regulator by the identical leave-one-out construction.
        internal review r158: no subsampling, and ANY unresolved regulator makes the whole gate
        SEARCH_UNRESOLVED -- excluding arms that timed out would select on solver success and bias
        the p95.

No scientific slack is added: N_MIN = 100 and max|SMD| <= 0.10 stand. The only tolerance is
numerical (tau <= 1e-8, values within 10*tau are NUMERICALLY_UNRESOLVED).
"""
import hashlib
import numpy as np

N_MIN = 100
SMD_GATE = 0.10
TAU = 1e-8
SOLVER_OPTIONS = {"presolve": True, "mip_rel_gap": 0.0}
# internal review r159 P0: solver options were RECORDED in the artifact but never passed to linprog, so the
# serialized options did not describe the run that actually happened.
LP_OPTIONS = {"presolve": True}


def load_verified_module(path, name):
    """Load a module from the EXACT bytes that are hashed, and return (module, sha256).

    internal review r176/r177: hashing a file and then importing it by name is not provenance -- an unqualified
    import can resolve elsewhere under `-m` or a shadowed sys.path, and even spec_from_file_location
    may execute a timestamp-valid stale .pyc rather than the source that was hashed. Reading the
    bytes once, hashing THOSE bytes, and exec'ing the code compiled from THEM makes the executed
    module and the recorded hash the same object by construction. No bytecode cache is consulted.
    """
    import hashlib as _h, types as _t
    with open(path, "rb") as fh:
        src = fh.read()
    sha = _h.sha256(src).hexdigest()
    mod = _t.ModuleType(name)
    mod.__file__ = path
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod, sha


def _balance_constraints(Ds, n_min, gate, extra_cols=0):
    """q = sum z >= n_min and -gate*q <= sum_i z_i d_ij <= gate*q, as rows over [z, extras]."""
    from scipy.optimize import LinearConstraint
    n, m = Ds.shape
    w = n + extra_cols
    def row(v):
        r = np.zeros(w); r[:n] = v; return r
    cons = [LinearConstraint(row(np.ones(n))[None, :], n_min, np.inf)]
    for j in range(m):
        cons.append(LinearConstraint(row(Ds[:, j] - gate)[None, :], -np.inf, 0.0))
        cons.append(LinearConstraint(row(Ds[:, j] + gate)[None, :], 0.0, np.inf))
    return cons


def verify_milp_solution(z, D, n_min=N_MIN, gate=SMD_GATE, R=None, tol=TAU):
    """Independently recompute every MILP constraint (internal review r156/r164 P1-4).

    Accepting solver status 0 and returning R* without checking is exactly the failure mode that
    would let a numerically infeasible placebo arm alter the p95 and hence gate C."""
    z = np.asarray(z, float)
    # internal review r165 P1-4: 1e-6 slack let a 5e-7 violation pass. The binding tolerance is TAU.
    if np.max(np.abs(z - np.round(z))) > 10 * tol:
        return False, "integrality violated"
    zb = np.round(z).astype(int)
    if not np.all((zb == 0) | (zb == 1)):
        return False, "binary domain violated"
    q = int(zb.sum())
    if q < n_min:
        return False, f"cardinality {q} < n_min {n_min}"
    # The comparison tolerance must be SCALE-RELATIVE. HiGHS's feasibility tolerance is absolute
    # (~1e-7) but is applied to constraints whose coefficients carry the data's magnitude, so the
    # effective slack grows with that magnitude. An absolute 10*tau verifier therefore rejected a
    # legitimate optimum on a constraint with coefficient 47 ("max r_i = 47 > R 47", a sub-1e-6
    # discrepancy) and turned one arm into SEARCH_UNRESOLVED -- which, under the no-subsampling rule,
    # would have blocked the ENTIRE placebo sweep and hence the whole gate on real data.
    lhs = np.abs(D[zb == 1].sum(0))
    # internal review r165 required a 5e-7 violation to REJECT; HiGHS legitimately returns ~1e-6 slack on a
    # constraint whose coefficient carries the data magnitude. Those two cannot both be met by one
    # tolerance. They are reconciled by observing that R is a DERIVED quantity, not a measurement:
    # min_radius_exact now recomputes R* = max_{i in S} r_i exactly from the returned panel, so the
    # radius constraint holds BY CONSTRUCTION and needs no tolerance at all. Balance is a genuine
    # constraint and keeps the ABSOLUTE 10*tau bound, which rejects 5e-7 as required -- a measured
    # probe shows real solutions satisfy it with a margin of 0.38, not marginally.
    if np.any(lhs > gate * q + 10 * tol):
        return False, (f"balance violated: max |sum z d| = {lhs.max():.9g} > "
                       f"{gate * q:.9g} + {10 * tol:.3g}")
    if R is not None:
        r = np.max(np.abs(D), axis=1)
        if np.any(r[zb == 1] > R + 10 * tol):
            return False, (f"radius violated: max r_i in panel = {r[zb == 1].max():.9g} > "
                           f"{R:.9g} + {10 * tol:.3g}")
    return True, "all constraints recomputed and satisfied"


def _milp_feasible(D, n_min=N_MIN, gate=SMD_GATE, cand=None, time_limit=60.0):
    from scipy.optimize import milp, Bounds
    idx = np.arange(D.shape[0]) if cand is None else np.asarray(cand, int)
    if len(idx) < n_min:
        return {"status": "PROVEN_INFEASIBLE", "feasible": False, "panel": None,
                "reason": f"only {len(idx)} candidates, fewer than n_min={n_min}"}
    Ds = D[idx]
    n = Ds.shape[0]
    res = milp(c=np.zeros(n), constraints=_balance_constraints(Ds, n_min, gate),
               integrality=np.ones(n), bounds=Bounds(0, 1),
               options={**SOLVER_OPTIONS, "time_limit": time_limit})
    if res.status == 0 and res.x is not None:
        xv = np.asarray(res.x)
        # A returned vector whose length does not match the problem cannot be interpreted; indexing
        # it would silently mis-map candidates. Refuse rather than guess.
        if xv.shape[0] != n:
            return {"status": "SEARCH_UNRESOLVED", "feasible": None, "panel": None,
                    "message": f"solution vector length {xv.shape[0]} != {n} decision variables"}
        # internal review audit #1, finding 4: this trusted solver status without recomputing the constraints, so
        # a bogus status-0 incumbent was accepted as FEASIBLE. Verify here too.
        ok, why = verify_milp_solution(xv, Ds, n_min, gate)
        if not ok:
            return {"status": "SEARCH_UNRESOLVED", "feasible": None, "panel": None,
                    "verification_failure": why}
        return {"status": "FEASIBLE", "feasible": True,
                "panel": [int(i) for i in idx[xv > 0.5]]}
    if res.status == 2:
        return {"status": "PROVEN_INFEASIBLE", "feasible": False, "panel": None}
    return {"status": "SEARCH_UNRESOLVED", "feasible": None, "panel": None,
            "solver_status": int(res.status), "message": str(res.message)}


def min_radius_exact(D, n_min=N_MIN, gate=SMD_GATE, time_limit=120.0):
    """R* by ONE exact MILP: minimise R subject to R >= r_i z_i plus balance/cardinality.

    internal review r158 replaced the binary search with this, so R* is an optimum with a proof rather than the
    endpoint of a search whose monotonicity had to be argued."""
    from scipy.optimize import milp, Bounds, LinearConstraint
    n = D.shape[0]
    if n < n_min:
        return {"status": "PROVEN_INFEASIBLE", "R_star": np.inf, "panel": None}
    r = np.max(np.abs(D), axis=1)
    cons = _balance_constraints(D, n_min, gate, extra_cols=1)
    for i in range(n):                       # R - r_i z_i >= 0
        row = np.zeros(n + 1); row[i] = -r[i]; row[n] = 1.0
        cons.append(LinearConstraint(row[None, :], 0.0, np.inf))
    c = np.zeros(n + 1); c[n] = 1.0
    lb = np.zeros(n + 1); ub = np.ones(n + 1); ub[n] = float(r.max()) if r.size else 1.0
    integ = np.ones(n + 1); integ[n] = 0
    res = milp(c=c, constraints=cons, integrality=integ, bounds=Bounds(lb, ub),
               options={**SOLVER_OPTIONS, "time_limit": time_limit})
    if res.status == 0 and res.x is not None:
        if np.asarray(res.x).shape[0] != n + 1:
            return {"status": "SEARCH_UNRESOLVED", "R_star": None, "panel": None,
                    "message": (f"solution vector length {np.asarray(res.x).shape[0]} != {n + 1} "
                                f"decision variables")}
        z = np.asarray(res.x)[:n]
        # DERIVE R* exactly from the returned panel instead of trusting the solver's continuous R.
        # The MILP's R variable exists only to drive the objective; the achieved radius is a function
        # of the selected set, so computing it directly makes the radius constraint exact.
        zb = np.round(z).astype(int)
        R = float(np.max(r[zb == 1])) if int(zb.sum()) else 0.0
        R_solver = float(np.asarray(res.x)[n])
        # internal review r167 P1: deriving R made the radius comparison vacuous (it checked R against itself).
        # The check that actually carries information is OPTIMALITY: the derived radius of the
        # returned panel must equal the solver's proven optimal objective. If the panel's true radius
        # EXCEEDS the proven optimum, the returned panel is not optimal and R* would be overstated,
        # making the locality gate too permissive. Note this compares two computed values of the SAME
        # quantity, where solver slop is expected -- it is an optimality-consistency check, NOT the
        # constraint-feasibility check that internal review strict 5e-7 rule governs, so a scale-relative
        # bound is appropriate here and the absolute bound still governs balance.
        obj = float(res.fun) if res.fun is not None else R_solver
        # internal review r169: a tolerance-based optimality check is NOT decision-safe. R* feeds both the
        # selected panel and the exact `obs <= p95` locality comparison, which carries no uncertainty
        # band, so a non-optimal panel inside ANY accepted slack could still flip gate C. Certify the
        # DISCRETE property instead, with no dependence on the solver's continuous feasibility slack:
        # U is the true minimum achievable radius iff NO balance-feasible panel exists using only
        # candidates strictly below U. That is an auxiliary MILP whose PROVEN_INFEASIBLE is a proof.
        r_all = np.max(np.abs(D), axis=1)
        # internal review r170 P0: `< R - TAU` excluded candidates whose radius lies in [U-TAU, U). Those are
        # STRICTLY SMALLER than U and must be available to the auxiliary problem, or it can report
        # infeasibility for the wrong reason and rubber-stamp a non-optimal panel. U is itself a max
        # over the same r_all array, so `r_all < R` is EXACT in floating point -- a margin is not
        # merely unnecessary here, it breaks the certificate. Candidates with r_i == U stay excluded,
        # since they cannot witness a smaller optimum.
        below = np.flatnonzero(r_all < R)
        aux = _milp_feasible(D, n_min, gate, cand=below, time_limit=time_limit)
        if aux["status"] == "FEASIBLE":
            return {"status": "SEARCH_UNRESOLVED", "R_star": None, "panel": None,
                    "verification_failure": (f"returned panel is not optimal: a balance-feasible "
                                             f"panel exists using only candidates with radius "
                                             f"< {R:.12g}"),
                    "solver_objective": obj}
        if aux["status"] != "PROVEN_INFEASIBLE":
            return {"status": "SEARCH_UNRESOLVED", "R_star": None, "panel": None,
                    "verification_failure": ("could not PROVE that no smaller-radius panel exists; "
                                             "optimality is unresolved, not refuted"),
                    "aux_status": aux["status"]}
        # radius is exact by construction on this path, so it is NOT re-checked here
        ok, why = verify_milp_solution(z, D, n_min, gate)
        if not ok:
            # internal review r164 P1-4: a solver "success" that fails independent recomputation is numerical
            # ambiguity, which must be unresolved -- never an accepted R* feeding the p95.
            return {"status": "SEARCH_UNRESOLVED", "R_star": None, "panel": None,
                    "verification_failure": why}
        return {"status": "OK", "R_star": R, "R_from_solver": R_solver, "verified": why,
                "R_derivation": "R* = max_i in S ||d_i||_inf, recomputed from the panel",
                "optimality_certificate": ("auxiliary MILP over candidates with r_i < R* is "
                                           "PROVEN_INFEASIBLE, so R* is the true discrete minimum"),
                "panel": [int(i) for i in np.flatnonzero(z > 0.5)]}
    if res.status == 2:
        # no balance-feasible panel exists at all; R* is unattainable
        return {"status": "PROVEN_INFEASIBLE", "R_star": np.inf, "panel": None}
    return {"status": "SEARCH_UNRESOLVED", "R_star": None, "panel": None,
            "solver_status": int(res.status), "message": str(res.message)}


# ------------------------------------------------------------------------------ Farkas certificate
def _verify_farkas(A_ub, b_ub, A_eq, b_eq, lb, ub, y, lam, tol=1e-9):
    """Check the proof arithmetically; do not trust a solver status.

    y >= 0 gives y'A_ub w <= y'b_ub; lam gives lam'A_eq w = lam'b_eq. With s = A_ub'y + A_eq'lam,
    s'w <= y'b_ub + lam'b_eq, while over the box s'w >= sum_i (s_i>=0 ? s_i*lb_i : s_i*ub_i).
    A strict gap between those two bounds IS the infeasibility proof."""
    y = np.asarray(y, float).ravel(); lam = np.asarray(lam, float).ravel()
    if y.shape[0] != A_ub.shape[0] or lam.shape[0] != A_eq.shape[0]:
        return False, None, "multiplier shapes do not match the constraint system"
    if np.any(y < -tol):
        return False, None, "y has negative entries; not a valid Farkas multiplier"
    y = np.maximum(y, 0.0)
    s = A_ub.T @ y + A_eq.T @ lam
    gap = float(np.sum(np.where(s >= 0, s * lb, s * ub)) - (y @ b_ub + lam @ b_eq))
    return bool(gap > tol), gap, ("box minimum exceeds the constraint bound, which is a contradiction"
                                  if gap > tol else "no contradiction; the proof does not hold")


def _farkas_ray(A_ub, b_ub, A_eq, b_eq, ub):
    """Construct the ray by solving Farkas' lemma as its own LP (no dependence on dual conventions)."""
    from scipy.optimize import linprog
    m2, n = A_ub.shape
    me = A_eq.shape[0]
    nv = m2 + me + n
    rows = np.zeros((n, nv))
    rows[:, :m2] = -A_ub.T
    rows[:, m2:m2 + me] = -A_eq.T
    rows[np.arange(n), m2 + me + np.arange(n)] = 1.0
    c = np.zeros(nv)
    c[:m2] = b_ub; c[m2:m2 + me] = b_eq; c[m2 + me:] = -float(np.max(ub))
    bounds = [(0.0, 1.0)] * m2 + [(-1.0, 1.0)] * me + [(-1.0, 0.0)] * n
    res = linprog(c=c, A_ub=rows, b_ub=np.zeros(n), bounds=bounds, method="highs",
                  options=LP_OPTIONS)
    if res.status != 0 or res.x is None:
        return None, None, f"farkas LP did not solve (status {res.status})"
    return res.x[:m2], res.x[m2:m2 + me], f"farkas LP objective {float(res.fun):.6g}"


def lp_negative_certificate(D, n_min=N_MIN, gate=SMD_GATE, sidecar_path=None):
    import scipy
    from scipy.optimize import linprog
    n, m = D.shape
    A_ub = np.vstack([D.T, -D.T]); b_ub = np.full(2 * m, gate)
    A_eq = np.ones((1, n)); b_eq = np.array([1.0])
    lb = np.zeros(n); ub = np.full(n, 1.0 / n_min)
    res = linprog(c=np.zeros(n), A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                  bounds=list(zip(lb, ub)), method="highs", options=LP_OPTIONS)
    out = {"n_candidates": int(n), "n_axes": int(m), "bound_per_weight": 1.0 / n_min,
           "solver": "scipy.optimize.linprog/highs", "scipy_version": scipy.__version__,
           "solver_options": {"method": "highs", "lp_options": LP_OPTIONS,
                              "milp_options": SOLVER_OPTIONS},
           "status": int(res.status), "message": str(res.message),
           "verification_tolerance": 1e-9}
    if res.status != 2:
        out["verdict"] = ("RELAXATION_FEASIBLE" if res.status == 0 else "INFEASIBILITY_NOT_CERTIFIED")
        if res.status == 0:
            out["NOTE"] = "does NOT establish an integer panel exists; the MILP decides that"
        return out

    cands = []
    yf, lamf, note = _farkas_ray(A_ub, b_ub, A_eq, b_eq, ub)
    if yf is not None:
        cands.append(("constructed_farkas_lp", yf, lamf))
    try:
        y_raw = np.asarray(res.ineqlin.marginals, float).ravel()
        lam_raw = np.asarray(res.eqlin.marginals, float).ravel()
        for sgn in (1.0, -1.0):
            cands.append((f"solver_marginals_sign{int(sgn)}", sgn * y_raw, sgn * lam_raw))
    except Exception:
        pass
    best = None
    for src, yv, lv in cands:
        ok, gap, why = _verify_farkas(A_ub, b_ub, A_eq, b_eq, lb, ub, yv, lv)
        if ok and (best is None or gap > best[1]):
            best = (src, gap, why, yv, lv)
    if best is None:
        out.update({"verdict": "INFEASIBILITY_NOT_CERTIFIED",
                    "dual_certificate": {"verified": False, "farkas_lp_note": note,
                                         "n_candidate_rays": len(cands),
                                         "reason": ("solver reported infeasible but no ray passed "
                                                    "arithmetic verification; NOT a NO-CALL")}})
        return out
    src, gap, why, yv, lv = best
    cert = {"source": src, "contradiction_gap": float(gap), "verified": True, "explanation": why,
            "farkas_lp_note": note}
    if sidecar_path:
        # internal review r158: serialize the INSTANTIATED system, not merely a hash of it, so the proof can be
        # recomputed independently of this code.
        np.savez_compressed(sidecar_path, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                            lb=lb, ub=ub, y=np.asarray(yv), lam=np.asarray(lv))
        cert["sidecar"] = sidecar_path
        cert["sidecar_sha256"] = hashlib.sha256(open(sidecar_path, "rb").read()).hexdigest()
    else:
        cert["y"] = np.asarray(yv).tolist(); cert["lam"] = np.asarray(lv).tolist()
    out.update({"verdict": "PROVEN_INFEASIBLE", "dual_certificate": cert,
                "NESTING_ARGUMENT": ("any subset of size q >= n_min induces w_i = 1/q <= 1/n_min, so "
                                     "LP infeasibility at that box excludes every admissible panel")})
    return out


# ------------------------------------------------------------------------- deterministic selection
def deterministic_panel(D, radius, names, n_min=N_MIN, gate=SMD_GATE, time_limit=120.0):
    """min cardinality -> min summed squared distance -> TRUE lexicographic fixing by gene name.

    internal review r158: an epsilon rank term is not a lexicographic tie-break, because equal rank sums and
    near-ties both defeat it. Genes are therefore fixed sequentially in name order, each by its own
    feasibility MILP, which yields a unique panel by construction."""
    from scipy.optimize import milp, Bounds, LinearConstraint
    r = np.max(np.abs(D), axis=1)
    # internal review r173: admitting r_i <= R* + TAU lets a candidate ABOVE the certified optimum enter, and
    # its counterexample D = (-1, 0.5, 0.5, 1+tau/2), n_min=2, gate=0.1 shows the consequence: the
    # true optimum R*=1 needs the three-member mean-zero panel, but the admitted fourth candidate
    # pairs with -1 at SMALLER cardinality, so stage 1 returns a panel whose radius EXCEEDS the
    # certified R*. The admissible set must be exact.
    cand = np.flatnonzero(r <= radius)
    Ds = D[cand]
    n = Ds.shape[0]
    base = _balance_constraints(Ds, n_min, gate)
    r1 = milp(c=np.ones(n), constraints=base, integrality=np.ones(n), bounds=Bounds(0, 1),
              options={**SOLVER_OPTIONS, "time_limit": time_limit})
    if r1.status != 0 or r1.x is None:
        return {"status": "SEARCH_UNRESOLVED", "stage": "min_cardinality"}
    q = int(round(float(np.sum(r1.x))))
    sq = (Ds ** 2).sum(1)
    fixq = base + [LinearConstraint(np.ones((1, n)), q, q)]
    r2 = milp(c=sq, constraints=fixq, integrality=np.ones(n), bounds=Bounds(0, 1),
              options={**SOLVER_OPTIONS, "time_limit": time_limit})
    if r2.status != 0 or r2.x is None:
        return {"status": "SEARCH_UNRESOLVED", "stage": "min_sq_distance"}
    target_sq = float(r2.fun)
    fixed_lo, fixed_hi = np.zeros(n), np.ones(n)
    # internal review r164 P1-1 (demonstrated by probe): a 1e-7 near-optimum slack admits STRICTLY WORSE panels,
    # so lexicographic fixing could return a lexically earlier but worse set while the artifact
    # reported the true optimum. The slack is now the project numerical tolerance, and the SELECTED
    # panel's ACTUAL objective is recomputed and reported below.
    cons3 = fixq + [LinearConstraint(sq[None, :], -np.inf, target_sq + TAU)]
    order = np.argsort(np.asarray(names, dtype=object)[cand].astype(str), kind="stable")
    for i in order:
        trial_lo = fixed_lo.copy(); trial_lo[i] = 1.0
        # internal review r165 P1-1: forced inclusion must MINIMISE the objective, not merely be feasible under
        # an objective constraint. A bad returned incumbent does not prove no valid forced-inclusion
        # solution exists, and HiGHS's own feasibility tolerance (~1e-7) is looser than TAU so the
        # constraint cannot decide gaps this fine. Solve for the OPTIMUM under forced inclusion and
        # compare it to the global optimum: include if it attains it, exclude only if its optimal
        # LOWER BOUND is strictly above the numerical band, otherwise refuse to guess.
        rr = milp(c=sq, constraints=fixq, integrality=np.ones(n),
                  bounds=Bounds(trial_lo, fixed_hi),
                  options={**SOLVER_OPTIONS, "time_limit": time_limit})
        if rr.status == 0 and rr.x is not None:
            forced_opt = float(rr.fun)
            # forced_opt is a PROVEN optimum (mip_rel_gap 0), so exceeding the global optimum by
            # more than tau is itself the proof that this gene cannot belong to an optimal panel.
            # "Unresolved" is reserved for the solver failing to prove optimality, not for a small
            # but real gap -- a 10*tau band would swallow internal review own 5e-8 probe and refuse to decide
            # a case that is decidable.
            if forced_opt <= target_sq + TAU:
                fixed_lo = trial_lo
            else:
                fixed_hi = fixed_hi.copy(); fixed_hi[i] = 0.0     # proven strictly worse
        elif rr.status == 2:
            fixed_hi = fixed_hi.copy(); fixed_hi[i] = 0.0
        else:
            # internal review r159 P1: treating a TIMEOUT as proof the gene must be excluded silently produces
            # a feasible-but-not-lexicographic panel. Only a PROOF of infeasibility may exclude.
            return {"status": "SEARCH_UNRESOLVED", "stage": "lexicographic_fixing",
                    "detail": f"solver status {int(rr.status)} on gene index {int(i)}; a "
                              f"non-proof cannot be read as exclusion"}
    sel_local = np.flatnonzero(fixed_lo > 0.5)
    sel = cand[sel_local]
    if len(sel) != q:
        return {"status": "SEARCH_UNRESOLVED", "stage": "lexicographic_fixing",
                "detail": f"fixed {len(sel)} of {q}"}
    achieved = float(sq[sel_local].sum())
    if achieved > target_sq + TAU:
        # the selected panel must actually attain the optimum, not merely lie near it
        return {"status": "SEARCH_UNRESOLVED", "stage": "objective_verification",
                "detail": f"selected objective {achieved:.12g} exceeds optimum {target_sq:.12g}"}
    ok, why = verify_milp_solution(np.isin(np.arange(n), sel_local).astype(float), Ds, n_min, gate)
    if not ok:
        return {"status": "SEARCH_UNRESOLVED", "stage": "constraint_verification", "detail": why}
    achieved_radius = float(np.max(np.max(np.abs(Ds[sel_local]), axis=1))) if len(sel_local) else 0.0
    if achieved_radius > radius:
        return {"status": "SEARCH_UNRESOLVED", "stage": "radius_verification",
                "detail": (f"selected panel radius {achieved_radius:.12g} exceeds the certified "
                           f"optimum {radius:.12g}")}
    return {"status": "OK", "panel": [int(i) for i in np.sort(sel)], "cardinality": q,
            "sum_sq_distance_optimum": target_sq,
            "sum_sq_distance_selected": achieved,
            "achieved_radius": achieved_radius,
            "verified": why,
            "RULE": "min cardinality, then min summed squared distance, then lexicographic fixing"}


def certified_panel(Fz, pool, target_idx, n_min=N_MIN, gate=SMD_GATE, placebo=True,
                    time_limit=120.0, names=None, sidecar_path=None):
    cand = np.array([i for i in pool if i != target_idx], int)
    D = Fz[cand] - Fz[target_idx][None, :]
    out = {"n_eligible": int(len(cand)), "n_min": n_min, "smd_gate": gate}

    unsupported = common_support_ok(Fz, cand, target_idx)   # one rule, shared with placebo arms
    out["axes_without_common_support"] = unsupported
    if unsupported:
        # internal review r158 P0: recording this and continuing could still certify a panel. Support failure
        # must terminate here -- no matched control can exist on an axis with no overlap.
        out["VERDICT"] = "MATCHING_INFEASIBLE_CERTIFIED"
        out["reason"] = ("common-support failure: the target lies outside the eligible pool's range "
                         "on these axes, so no matched regulator control can exist")
        return out

    out["lp_certificate"] = lp_negative_certificate(D, n_min, gate, sidecar_path)
    v = out["lp_certificate"]["verdict"]
    if v == "PROVEN_INFEASIBLE":
        out["VERDICT"] = "MATCHING_INFEASIBLE_CERTIFIED"; return out
    if v == "INFEASIBILITY_NOT_CERTIFIED":
        out["VERDICT"] = "INFEASIBILITY_NOT_CERTIFIED"; return out

    mi = _milp_feasible(D, n_min, gate, time_limit=time_limit)
    out["milp"] = {k: q for k, q in mi.items() if k != "panel"}
    if mi["status"] == "PROVEN_INFEASIBLE":
        out["VERDICT"] = "MATCHING_INFEASIBLE_CERTIFIED"; return out
    if mi["feasible"] is not True:
        out["VERDICT"] = "SEARCH_UNRESOLVED"; return out

    mm = min_radius_exact(D, n_min, gate, time_limit=time_limit)
    out["min_radius"] = {k: q for k, q in mm.items() if k != "panel"}
    if mm["status"] != "OK":
        out["VERDICT"] = ("MATCHING_INFEASIBLE_CERTIFIED" if mm["status"] == "PROVEN_INFEASIBLE"
                          else "SEARCH_UNRESOLVED")
        return out

    nm = np.asarray(names, dtype=object)[cand] if names is not None else np.arange(len(cand))
    det = deterministic_panel(D, mm["R_star"], nm, n_min, gate, time_limit)
    out["deterministic_selection"] = {k: q for k, q in det.items() if k != "panel"}
    if det["status"] != "OK":
        out["VERDICT"] = "SEARCH_UNRESOLVED"; return out

    sel_local = np.array(det["panel"], int)
    sel = cand[sel_local]
    smd = np.abs(D[sel_local].mean(0))
    band = [int(j) for j in range(len(smd)) if abs(smd[j] - gate) <= 10 * TAU]
    out.update({"selected_indices": [int(i) for i in np.sort(sel)],
                "panel_size": int(len(sel)),
                "achieved_max_abs_SMD": float(smd.max()),
                "achieved_margin": float(gate - smd.max()),
                "coordinatewise_rms_distance": [float(q) for q in
                                                np.sqrt((D[sel_local] ** 2).mean(0))],
                "R_star": mm["R_star"], "numerically_unresolved_axes": band})
    if band:
        out["VERDICT"] = "NUMERICALLY_UNRESOLVED"; return out
    if smd.max() > gate:
        out["VERDICT"] = "SEARCH_UNRESOLVED"; return out
    out["VERDICT"] = "PANEL_CERTIFIED_PENDING_PLACEBO" if placebo else "PANEL_CERTIFIED"
    return out


def _limit_threads():
    """internal review r161 P2: HiGHS chooses its thread count automatically, and scipy exposes no option to
    override it, so n_jobs workers each spawning threads would oversubscribe the CPUs and make the
    sweep slower rather than faster. Constrain the underlying thread pools in each worker instead.
    Measured cost is 1.5-4 s per arm single-threaded, so this loses nothing."""
    import os
    for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[v] = "1"


def common_support_ok(Fz, cand, target_idx):
    """Is the target inside the candidate range on EVERY axis? Shared by the regulator and placebo
    paths so the two use one rule."""
    lo, hi = Fz[cand].min(0), Fz[cand].max(0)
    t = Fz[target_idx]
    return [int(j) for j in range(Fz.shape[1]) if not (lo[j] <= t[j] <= hi[j])]


def _placebo_one(args):
    """One leave-one-out R*. Module level so it is picklable by multiprocessing."""
    Fz, pool, g, n_min, gate, time_limits = args
    cand = np.array([i for i in pool if i != g], int)
    # internal review audit #1, finding 1: the regulator path REFUSES on common-support failure while placebo
    # arms skipped that test entirely. internal review counterexample -- 100 candidates at 1.0, target at 1.05,
    # gate 0.10 -- has the regulator declared support-infeasible while a placebo accepts a finite
    # R* = 0.05. Unsupported placebos would then look WELL MATCHED, depressing the p95 threshold and
    # capable of converting a valid outcome into a FALSE GATA1 FAILURE. The construction must be
    # identical, so an unsupported placebo is unmatchable: R* = +inf.
    if common_support_ok(Fz, cand, int(g)):
        return int(g), float("inf")
    D = Fz[cand] - Fz[g][None, :]
    for tl in time_limits:
        mm = min_radius_exact(D, n_min, gate, time_limit=tl)
        if mm["status"] == "OK":
            return int(g), float(mm["R_star"])
        if mm["status"] == "PROVEN_INFEASIBLE":
            return int(g), float("inf")
    return int(g), None


def placebo_locality(Fz, pool, target_idx, n_min=N_MIN, gate=SMD_GATE,
                     time_limits=(60.0, 300.0, 900.0), n_jobs=1, wall_clock_budget=None):
    """R* for the target AND EVERY eligible regulator, by the identical leave-one-out construction.

    internal review r158: no subsampling, and ANY unresolved regulator makes the whole gate SEARCH_UNRESOLVED --
    dropping arms that timed out would select on solver success and bias the p95. A proven-infeasible
    placebo resolves as R* = +inf; only a timeout or numerical ambiguity is unresolved. Unresolved
    arms are retried under a pre-declared escalating schedule before being given up on."""
    # internal review r160 P1: serial worst case is len(pool) * sum(time_limits) -- about 119 h for 340 arms,
    # which would never finish. internal review r158 sanctioned parallelising across regulators; a declared
    # global budget is also recorded so an over-budget sweep degrades to SEARCH_UNRESOLVED rather
    # than running indefinitely. Neither relaxes any gate: both only ever produce "unresolved".
    import time as _time
    t0 = _time.time()
    tasks = [(Fz, pool, int(g), n_min, gate, time_limits) for g in pool]
    rs, unresolved = {}, []
    if n_jobs and n_jobs > 1:
        import multiprocessing as mp
        done = set()
        with mp.Pool(processes=int(n_jobs), initializer=_limit_threads) as p:
            it = p.imap_unordered(_placebo_one, tasks, chunksize=1)
            for g, val in it:
                done.add(int(g))
                (rs.__setitem__(g, val) if val is not None else unresolved.append(g))
                # internal review r161 P1: the budget was checked only in the serial branch, so a parallel
                # sweep could run past it. Enforce it INSIDE the collection loop and abandon the
                # remaining arms as unresolved rather than waiting for the whole sweep.
                if wall_clock_budget and (_time.time() - t0) > wall_clock_budget:
                    p.terminate()
                    break
        for t in tasks:
            if int(t[2]) not in done and int(t[2]) not in unresolved:
                unresolved.append(int(t[2]))
    else:
        for t in tasks:
            if wall_clock_budget and (_time.time() - t0) > wall_clock_budget:
                unresolved.append(int(t[2])); continue
            g, val = _placebo_one(t)
            (rs.__setitem__(g, val) if val is not None else unresolved.append(g))
    elapsed = float(_time.time() - t0)
    # internal review r161 P1: the `and unresolved` conjunct meant a fully resolved but over-budget sweep could
    # still reach PASS. An over-budget sweep is unresolved regardless of how many arms finished.
    if wall_clock_budget and elapsed > wall_clock_budget:
        return {"status": "SEARCH_UNRESOLVED", "n_unresolved": len(unresolved),
                "n_resolved": len(rs), "elapsed_seconds": elapsed,
                "wall_clock_budget": wall_clock_budget, "PASS": None,
                "RULE": "declared global budget exceeded; this is a computational, not scientific, outcome"}
    if unresolved:
        return {"status": "SEARCH_UNRESOLVED", "n_unresolved": len(unresolved),
                "n_resolved": len(rs), "unresolved_examples": unresolved[:10],
                "RULE": ("every eligible regulator must resolve; excluding timed-out arms would "
                         "select on solver success and bias the percentile"),
                "PASS": None}
    if int(target_idx) not in rs:
        return {"status": "TARGET_UNRESOLVED", "PASS": None}
    vals = np.array([q for k, q in rs.items() if k != int(target_idx)], float)
    finite = vals[np.isfinite(vals)]
    if vals.size < 20:
        return {"status": "INSUFFICIENT_PLACEBOS", "n_placebo": int(vals.size),
                "n_unresolved": 0, "PASS": None}
    # internal review r159 P0 / internal review: the previous guard set p95 = +inf whenever ANY placebo was infinite,
    # which made `obs <= p95` vacuously true -- a silent auto-pass on the very gate meant to catch a
    # non-local target. numpy already orders +inf last, so the empirical percentile is correct as-is
    # and only becomes infinite when MORE THAN 5% of regulators are genuinely unmatchable, which is
    # itself a reportable finding rather than an accident.
    p95 = float(np.percentile(vals, 95))
    obs = rs[int(target_idx)]
    if not np.isfinite(p95):
        return {"status": "PLACEBO_DISTRIBUTION_DEGENERATE", "target_R_star": obs,
                "placebo_p95": p95, "n_infinite_placebos": int((~np.isfinite(vals)).sum()),
                "n_placebo": int(vals.size), "n_unresolved": 0, "PASS": None,
                "REASON": ("more than 5% of eligible regulators cannot be matched at all, so the "
                           "95th percentile is infinite and cannot calibrate the target; passing "
                           "against an infinite threshold would be vacuous")}
    return {"status": "OK", "target_R_star": obs, "placebo_p95": p95,
            "placebo_median": float(np.median(vals)),
            "n_placebo": int(vals.size), "n_infinite_placebos": int((~np.isfinite(vals)).sum()),
            "n_unresolved": 0,
            "target_percentile": float((finite < obs).mean() * 100) if finite.size else None,
            "PASS": bool(obs <= p95),
            "RULE": ("mean balance is necessary but not sufficient; this rejects a target that can "
                     "only be mean-matched by a bimodal set resembling nothing")}
