#!/usr/bin/env python
"""FROZEN truth-free UNSUPERVISED COMPOSITE baselines over the nine REF_SPEC columns (spec 2026-09-29).  No label / perturbation / truth enters here.
  COMP_Z     mean_j Z_gj                                   Z = signed columns standardised over U* (population mean/SD, ddof=0; constant column -> 0)
  COMP_RANK  mean_j (midrank_U*(s_j f_j) - .5)/|U*|        (a constant column contributes .5)
  COMP_PC1   Z v, v = leading eigenvector of Z'Z/|U*|      orient sum(v)>0 (exact-zero sum -> first nonzero loading positive); top-2 eigenvalues tied within 1e-10 rel -> NO_CALL
  COMP_FA    one-factor Gaussian ML FA posterior mean      sklearn FactorAnalysis(1, tol=1e-8, max_iter=10000, svd_method='lapack', random_state=0), same sign rule on the loading;
                                                           n_iter_ == max_iter -> NO_CALL, OR min unique variance Psi < 0.01 on the standardised scale over the NON-CONSTANT columns (Heywood case) -> NO_CALL
NO_CALL composites are ABSENT from the returned dict (never replaced, never zero).  U* = rows with all nine signed features finite (caller supplies the frozen universe).
  python scratch/gb_composites.py        # synthetic self-test
"""
import warnings

import numpy as np
from scipy.stats import rankdata

# canonical listed order (= gb_ladder.REF_SPEC; gb_ladder asserts equality, gb_layer_lattice asserts the same (column, sign) set)
SPEC = [("raw_det", -1), ("sm_det", -1), ("abundance", -1), ("pair_support", -1), ("mu_det", -1), ("dynamics", 1), ("mu_dynamics", 1),
        ("abs_spearman_mu_dpt", 1), ("mu_logvar", 1)]
COMP_NAMES = ["COMP_Z", "COMP_RANK", "COMP_PC1", "COMP_FA"]
TIE_RTOL, FA_MAX_ITER, FA_MIN_PSI = 1e-10, 10000, 0.01


def orient(v):
    """+1/-1 so that sum(v) > 0; exact-zero sum -> sign of the first nonzero entry (listed order) made positive; all-zero -> +1"""
    s = v.sum()
    if s == 0:
        nz = np.flatnonzero(v != 0)
        s = v[nz[0]] if len(nz) else 1.0
    return 1.0 if s > 0 else -1.0


def composites(X, fa_max_iter=FA_MAX_ITER):
    """X (n, 9) SIGNED features (sign_j * f_j, columns in SPEC order), every entry finite.  Returns ({name: (n,) score or None=NO_CALL}, diag)."""
    X = np.asarray(X, float)
    if not np.isfinite(X).all():
        raise ValueError("composites(): non-finite entry (U* must contain only rows with all nine features finite)")
    n = len(X)
    const = np.ptp(X, axis=0) == 0
    Z = np.where(const, 0.0, (X - X.mean(0)) / np.where(const, 1.0, X.std(0)))          # ddof=0
    out = dict.fromkeys(COMP_NAMES)
    out["COMP_Z"] = Z.mean(1)
    out["COMP_RANK"] = ((np.column_stack([rankdata(X[:, j]) for j in range(X.shape[1])]) - 0.5) / n).mean(1)   # average midrank, 1-based
    w, V = np.linalg.eigh(Z.T @ Z / n)
    l1, l2 = w[-1], w[-2]
    v1 = V[:, -1] * orient(V[:, -1])
    diag = dict(n=n, constant_columns=[int(j) for j in np.flatnonzero(const)], pc1_top2=[float(l1), float(l2)], pc1_loadings=[float(x) for x in v1], fa_n_iter=None)
    if l1 > 0 and l1 - l2 > TIE_RTOL * l1:
        out["COMP_PC1"] = Z @ v1
    from sklearn.decomposition import FactorAnalysis
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")                              # non-convergence is recorded via n_iter_ (-> NO_CALL), not via a warning
        fa = FactorAnalysis(n_components=1, tol=1e-8, max_iter=fa_max_iter, svd_method="lapack", random_state=0).fit(Z)
    ll = list(getattr(fa, "loglike_", []))
    sg = orient(fa.components_[0])
    with np.errstate(all="ignore"):                                  # a constant column gives -inf log-likelihoods
        diag.update(fa_n_iter=int(fa.n_iter_), fa_loadings=[float(x) for x in fa.components_[0] * sg], fa_psi=[float(x) for x in fa.noise_variance_],
                    fa_min_psi=(float(fa.noise_variance_[~const].min()) if (~const).any() else float("nan")), fa_last_loglik_change=(float(np.float64(ll[-1]) - np.float64(ll[-2])) if len(ll) > 1 else None))
    why = "+".join(r for r, bad in (("nonconvergence", fa.n_iter_ >= fa_max_iter), ("heywood_min_psi<%g" % FA_MIN_PSI, bool((~const).any() and fa.noise_variance_[~const].min() < FA_MIN_PSI))) if bad) or None
    diag["fa_no_call_reason"] = why
    if why is None:
        out["COMP_FA"] = fa.transform(Z)[:, 0] * sg
    diag["no_call"] = [k for k, v in out.items() if v is None]
    return out, diag


def comp_from_features(F):
    """F: mapping column -> (N,) array (dict or DataFrame) over ONE gene axis.  U* = rows with all nine signed features finite.  Returns ({name: (N,) vector,
    NaN outside U*; NO_CALL names absent}, diag with n_universe / n_Ustar / n_excluded)."""
    X = np.column_stack([sg * np.asarray(F[c], float) for c, sg in SPEC])
    fin = np.isfinite(X).all(1)
    diag = dict(n_universe=int(len(X)), n_Ustar=int(fin.sum()), n_excluded=int((~fin).sum()))
    if fin.sum() < 2:
        return {}, dict(diag, no_call=list(COMP_NAMES), note="fewer than 2 rows in U*")
    res, d = composites(X[fin])
    vec = {}
    for k, v in res.items():
        if v is not None:
            vec[k] = np.full(len(X), np.nan)
            vec[k][fin] = v
    return vec, dict(diag, **d)


def _selftest():
    rng = np.random.default_rng(0)
    # orientation rule
    assert orient(np.array([-1.0, -2.0, 0.5])) == -1.0 and orient(np.array([1.0, 2.0])) == 1.0
    assert orient(np.array([0.0, 1.0, -1.0])) == 1.0 and orient(np.array([0.0, -1.0, 1.0])) == -1.0 and orient(np.zeros(3)) == 1.0
    # latent-factor world: signed columns all load positively on f -> every composite is positively oriented to f
    n = 400
    f = rng.normal(size=n)
    F = {c: sg * (f + 0.5 * rng.normal(size=n)) for c, sg in SPEC}                     # raw column = sign * (f + noise), so sign*column = f + noise > 0 loading
    vec, dg = comp_from_features(F)
    assert set(vec) == set(COMP_NAMES) and dg["n_excluded"] == 0 and dg["no_call"] == [], dg
    for k, v in vec.items():
        assert np.corrcoef(v, f)[0, 1] > 0.8, (k, np.corrcoef(v, f)[0, 1])
    # sign flips on the signed matrix flip PC1 (orientation by sum(v) > 0 is on the LOADING, not a free sign)
    X = np.column_stack([sg * F[c] for c, sg in SPEC])
    a, _ = composites(X)
    b, _ = composites(-X)
    assert np.allclose(b["COMP_PC1"], -a["COMP_PC1"]) and np.allclose(b["COMP_Z"], -a["COMP_Z"]) and np.allclose(b["COMP_FA"], -a["COMP_FA"], atol=1e-6)
    # rank formula + constant column (midrank -> .5) + ties, checked against a hand formula
    X2 = np.round(rng.normal(size=(7, 9)), 0)
    X2[:, 3] = 4.0
    o2, d2 = composites(X2)
    R = np.array([[(X2[:, j] < x).sum() + ((X2[:, j] == x).sum() + 1) / 2 for x in X2[:, j]] for j in range(9)]).T          # midrank by definition
    assert np.allclose(o2["COMP_RANK"], ((R - 0.5) / 7).mean(1)) and d2["constant_columns"] == [3]
    assert np.allclose(((R[:, 3] - 0.5) / 7), 0.5)
    Zm = np.where(np.arange(9) == 3, 0.0, (X2 - X2.mean(0)) / np.where(np.arange(9) == 3, 1.0, X2.std(0)))
    assert np.allclose(o2["COMP_Z"], Zm.mean(1)) and np.allclose(Zm[:, 3], 0.0)
    # exact eigenvalue tie -> PC1 NO_CALL (full 2^9 factorial +-1 design: Z'Z/n = I) and the call is absent, not replaced
    H = np.array([[1 if (i >> j) & 1 else -1 for j in range(9)] for i in range(512)], float)
    oh, dh = composites(H)
    assert oh["COMP_PC1"] is None and "COMP_PC1" in dh["no_call"]
    # all-constant input -> PC1 NO_CALL
    assert composites(np.ones((20, 9)))[0]["COMP_PC1"] is None
    # FA non-convergence (max_iter=1 -> n_iter_ == max_iter) -> NO_CALL
    assert composites(X, fa_max_iter=1)[0]["COMP_FA"] is None and composites(X, fa_max_iter=1)[0]["COMP_Z"] is not None
    # Heywood: a column that IS the factor drives its unique variance to ~0 -> COMP_FA NO_CALL (reason recorded); a well-conditioned world keeps min Psi >= .01
    Xh = X.copy()
    Xh[:, 1] = f
    oh2, dh2 = composites(Xh)
    assert oh2["COMP_FA"] is None and dh2["fa_min_psi"] < 0.01 and "heywood" in dh2["fa_no_call_reason"], dh2
    assert oh2["COMP_Z"] is not None and a is not None
    _, dg0 = composites(X)
    assert dg0["fa_min_psi"] >= 0.01 and dg0["fa_no_call_reason"] is None and len(dg0["fa_loadings"]) == 9 and len(dg0["fa_psi"]) == 9 and len(dg0["pc1_loadings"]) == 9
    # a constant column has Psi ~ 0 by construction: the Heywood check must skip it (non-convergence may still trigger NO_CALL, but never the Heywood reason)
    Xc = X.copy()
    Xc[:, 3] = 2.0
    ocn, dcn = composites(Xc)
    assert dcn["constant_columns"] == [3] and "heywood" not in (dcn["fa_no_call_reason"] or ""), dcn
    assert dcn["fa_min_psi"] >= 0.01, "min Psi is taken over the non-constant columns"
    # U* = all nine finite; excluded rows are NaN in the output and do not influence the fit; labels/extra keys are irrelevant; row-permutation equivariant
    Fn = {c: F[c].copy() for c, _ in SPEC}
    Fn["raw_det"][:5] = np.nan
    Fn["label"] = rng.random(n) < 0.3
    vn, dn = comp_from_features(Fn)
    assert dn["n_excluded"] == 5 and all(np.isnan(v[:5]).all() and np.isfinite(v[5:]).all() for v in vn.values())
    Fl = {c: Fn[c] for c, _ in SPEC}
    Fl["label"] = ~Fn["label"]
    vl, _ = comp_from_features(Fl)
    assert all(np.array_equal(vn[k], vl[k], equal_nan=True) for k in vn), "composites must not depend on labels"
    p = rng.permutation(n)
    vp, _ = comp_from_features({c: Fn[c][p] for c, _ in SPEC})
    assert all(np.allclose(vp[k], vn[k][p], equal_nan=True, atol=1e-6) for k in vn)
    print("gb_composites SELFTEST PASSED", {k: round(float(np.corrcoef(v, f)[0, 1]), 3) for k, v in vec.items()})


if __name__ == "__main__":
    _selftest()
