#!/usr/bin/env python
"""INFORMATION-MATCHED COMPARATOR (internal directive §183 item 22; internal review §182): uses EXACTLY GeoBRIDGE's observational information —
the same forward pairs, the same R1/R3 conditioning (nested QR prefix Q_k of dP, both source and target projected, as
gb_target_extension.project_scores), the same source pair displacement ΔX_r^(k) and target nascent displacement ΔU_g^(k) — but
replaces the concordance statistic by a CROSS-FITTED PREDICTOR of ΔU_g^(k) from ΔX_r^(k). If this beats concordance, the value
is in the measurement, not in the statistic.
PREDECLARED PROTOCOL (fixed; never tuned on any biological endpoint):
  folds (internal review 2026-09-24 repair): cells assigned to NFOLD=5 folds by PCG64(SEED=20260924) permutation. For held-out fold f:
         TRAIN on every pair whose two cells are both OUTSIDE f (expected coverage 16/25 of pairs), TEST on every pair whose two
         cells are both INSIDE f (1/25); pairs with one cell in f are unused for that fold. Zero train/test cell overlap. The same
         held-out pair union (both-in-same-fold pairs, 5/25) is used for the MATCHED GeoBRIDGE statistic (|S| on those pairs).
  conditioning: for every fold and k, fit source and target nuisance coefficients against the corresponding dP/Q prefix using
         TRAIN pairs only, then apply the frozen coefficients to that fold's TRAIN and TEST pairs.
  features of source r (per pair): LINEAR  B = [1, x];  SPLINE (fixed-basis additive model) B = [1, x, x^2, x^3,
         (x - t_j)_+^3 for knots t_j = training-fold quantiles .25/.5/.75 of x]; x standardised on the training fold.
  model: multi-output ridge, closed form W = (B'B + LAM·I')^{-1} B'Y for ALL targets at once; LAM = 1.0 × n_train (intercept
         unpenalised). Y = the conditioned target displacements (training fold).
  score: held-out R^2 pooled over folds, 1 - Σ(y - ŷ)^2 / Σ(y - ȳ_train)^2, per (source, target); NaN if < MIN_PAIRS held-out pairs.
         Unsigned evidence (like E = |S|); may be negative (no predictive information).
output: score matrices (n_sources, n_targets) per k, same layout as the other comparators.
usage: import gb_infomatched as IM; for k, lin, spl in IM.scores(dx, y, Qf, I, J, ks): ...   |   python gb_infomatched.py (selftest)"""
import numpy as np

NFOLD, SEED, LAM_PER_N, MIN_PAIRS = 5, 20260924, 1.0, 50


def cell_folds(n_cells, nfold=NFOLD, seed=SEED):
    cf = np.empty(n_cells, int); cf[np.random.Generator(np.random.PCG64(seed)).permutation(n_cells)] = np.arange(n_cells) % nfold
    return cf


def fold_masks(I, J, n_cells, nfold=NFOLD, seed=SEED):
    """list over folds of (train_mask, test_mask): train = both cells outside f; test = both cells inside f"""
    cf = cell_folds(n_cells, nfold, seed); fI, fJ = cf[I], cf[J]
    masks = [((fI != f) & (fJ != f), (fI == f) & (fJ == f)) for f in range(nfold)]
    cov_tr, cov_te = np.mean([tr.mean() for tr, _ in masks]), np.mean([te.mean() for _, te in masks])
    tol_tr, tol_te = max(.03, 5 / np.sqrt(max(len(I), 1))), max(.01, 3 / np.sqrt(max(len(I), 1)))
    assert abs(cov_tr - ((nfold - 1) / nfold) ** 2) <= tol_tr, \
        f"train-pair coverage {cov_tr:.4f} is not approximately {(nfold-1)**2/nfold**2:.4f}; expected both endpoints outside test fold"
    assert abs(cov_te - 1 / nfold ** 2) <= tol_te, \
        f"test-pair coverage {cov_te:.4f} is not approximately {1/nfold**2:.4f}; expected both endpoints inside test fold"
    return masks, cf


def fold_of_pairs(I, J, n_cells, nfold=NFOLD, seed=SEED):
    """held-out fold id per pair (-1 = the pair is never a TEST pair: its cells lie in different folds)"""
    cf = cell_folds(n_cells, nfold, seed); f = cf[I]; return np.where(f == cf[J], f, -1)


def basis(x, ref, spline):
    m, s = ref.mean(), ref.std() + 1e-12; z = (x - m) / s
    cols = [np.ones_like(z), z]
    if spline:
        zr = (ref - m) / s
        cols += [z ** 2, z ** 3] + [np.clip(z - t, 0, None) ** 3 for t in np.quantile(zr, [.25, .5, .75])]
    return np.column_stack(cols)


def fit_nuisance(Z, Q, train):
    """Fit Z ~ Q on training pairs only. No intercept: the fold-local analogue of (I-QQ')Z."""
    Z, Q = np.asarray(Z, np.float64), np.asarray(Q, np.float64)
    if Q.shape[1] == 0:
        return np.empty((0,) + Z.shape[1:], np.float64)
    return np.linalg.lstsq(Q[train], Z[train], rcond=None)[0]


def apply_nuisance(Z, Q, coef, rows):
    Z, Q = np.asarray(Z, np.float64), np.asarray(Q, np.float64)
    return Z[rows] - Q[rows] @ coef if Q.shape[1] else Z[rows].copy()


def ridge_predict(x_train, y_train, x_test, spline):
    Bt, Be = basis(x_train, x_train, spline), basis(x_test, x_train, spline)
    P = np.eye(Bt.shape[1]) * LAM_PER_N * len(x_train); P[0, 0] = 0.0          # unpenalised intercept
    W = np.linalg.solve(Bt.T @ Bt + P, Bt.T @ y_train)
    return Be @ W


def scores(dx, y, Qf, I, J, ks, n_cells=None, matched=False):
    """dx (P, ns), y (P, G), Qf (P, >=max ks) orthonormal (production nested QR). Yields (k, R2_linear (ns,G), R2_spline (ns,G))
    or, with matched=True, (k, R2_linear, R2_spline, absS_matched (ns,G)) where absS_matched = GeoBRIDGE |S| computed on EXACTLY the
    held-out pair union used by the predictors. At every fold and k, nuisance coefficients for source and target are fitted only on
    that fold's training pairs, then applied to its training and test pairs."""
    I, J = np.asarray(I), np.asarray(J); masks, _ = fold_masks(I, J, n_cells or int(max(I.max(), J.max())) + 1)
    for k in ks:
        Q = np.asarray(Qf[:, :k], np.float64); dx_, y_ = np.asarray(dx, np.float64), np.asarray(y, np.float64)
        shape = (dx_.shape[1], y_.shape[1]); sse_lin = np.zeros(shape); sse_spl = np.zeros(shape); sst = np.zeros(shape)
        num = np.zeros(shape); den = np.zeros(shape); n_test = 0
        for tr, te in masks:
            if te.sum() == 0 or tr.sum() < 10:
                continue
            bx, by = fit_nuisance(dx_, Q, tr), fit_nuisance(y_, Q, tr)
            xtr, xte = apply_nuisance(dx_, Q, bx, tr), apply_nuisance(dx_, Q, bx, te)
            ytr, yte = apply_nuisance(y_, Q, by, tr), apply_nuisance(y_, Q, by, te)
            base = ((yte - ytr.mean(0)) ** 2).sum(0)
            for r in range(dx_.shape[1]):
                sse_lin[r] += ((yte - ridge_predict(xtr[:, r], ytr, xte[:, r], False)) ** 2).sum(0)
                sse_spl[r] += ((yte - ridge_predict(xtr[:, r], ytr, xte[:, r], True)) ** 2).sum(0)
                sst[r] += base
            num += xte.T @ yte; den += np.abs(xte).T @ np.abs(yte); n_test += te.sum()
        with np.errstate(invalid="ignore", divide="ignore"):
            lin, spl = 1 - sse_lin / sst, 1 - sse_spl / sst
        ok = (n_test >= MIN_PAIRS) & (sst > 0); lin = np.where(ok, lin, np.nan); spl = np.where(ok, spl, np.nan)
        if matched:
            with np.errstate(invalid="ignore", divide="ignore"):
                absS = np.where((n_test >= MIN_PAIRS) & (den > 1e-12), np.abs(num) / den, np.nan)
            yield k, lin, spl, absS
        else:
            yield k, lin, spl


def selftest():
    rng = np.random.default_rng(0); nc, P, ns, G = 400, 6000, 3, 30
    I = rng.integers(0, nc, P); J = rng.integers(0, nc, P)
    masks, cf_ = fold_masks(I, J, nc)
    for tr, te in masks:                                                      # zero train/test cell overlap, coverage ~16/25 and 1/25
        assert not (set(I[tr]) | set(J[tr])) & (set(I[te]) | set(J[te]))
    cov_tr = np.mean([tr.mean() for tr, _ in masks]); cov_te = np.mean([te.mean() for _, te in masks])
    assert abs(cov_tr - 16 / 25) < .03 and abs(cov_te - 1 / 25) < .01, (cov_tr, cov_te)
    dx = rng.normal(size=(P, ns)); y = rng.normal(size=(P, G))
    y[:, 0] += 1.5 * dx[:, 0]; y[:, 1] -= 1.0 * dx[:, 1]                     # planted linear edges (0->0, 1->1)
    y[:, 2] += 1.5 * (dx[:, 2] ** 2 - 1)                                      # planted nonlinear edge (2->2)
    Qf = np.linalg.qr(rng.normal(size=(P, 5)))[0]
    (_, lin0, spl0), = scores(dx, y, Qf, I, J, [0], nc)
    assert lin0[0, 0] > .5 and lin0[1, 1] > .3, (lin0[0, 0], lin0[1, 1])       # linear recovers planted linear edges
    assert spl0[2, 2] > .5 and lin0[2, 2] < .05, (spl0[2, 2], lin0[2, 2])      # only the spline recovers the quadratic edge
    null = np.ones((ns, G), bool); null[0, 0] = null[1, 1] = null[2, 2] = False
    assert np.nanmax(lin0[null]) < .02 and np.nanmax(spl0[null]) < .02          # non-edges ~0 held-out R^2
    ys = y[rng.permutation(P)]                                                # shuffled U -> null
    (_, lins, spls), = scores(dx, ys, Qf, I, J, [0], nc)
    assert np.nanmax(lins) < .02 and np.nanmax(spls) < .02, (np.nanmax(lins), np.nanmax(spls))
    (_, link, _), = scores(dx, y, Qf, I, J, [3], nc)                          # conditioning path runs; signal survives random Q
    assert link[0, 0] > .5
    (_, _, _, sm), = scores(dx, y, Qf, I, J, [0], nc, matched=True)          # matched |S| on the identical held-out pair union
    ho = np.logical_or.reduce([te for _, te in masks]); ref = np.abs(dx[ho].T @ y[ho]) / (np.abs(dx[ho]).T @ np.abs(y[ho]))
    assert np.abs(sm - ref).max() < 1e-12
    # Exact fold leakage test: changing only fold-0 held-out y cannot alter fold-0 nuisance coefficients or predictions.
    tr, te = masks[0]; Q = Qf[:, :3]; bx0, by0 = fit_nuisance(dx, Q, tr), fit_nuisance(y, Q, tr)
    xtr0, xte0 = apply_nuisance(dx, Q, bx0, tr), apply_nuisance(dx, Q, bx0, te)
    ytr0 = apply_nuisance(y, Q, by0, tr); pred0 = ridge_predict(xtr0[:, 0], ytr0, xte0[:, 0], True)
    y_changed = y.copy(); y_changed[te] += rng.normal(1000, 100, size=y_changed[te].shape)
    bx1, by1 = fit_nuisance(dx, Q, tr), fit_nuisance(y_changed, Q, tr)
    ytr1 = apply_nuisance(y_changed, Q, by1, tr); pred1 = ridge_predict(xtr0[:, 0], ytr1, xte0[:, 0], True)
    assert np.array_equal(bx0, bx1) and np.array_equal(by0, by1) and np.array_equal(pred0, pred1)
    print(f"coverage train {cov_tr:.3f} (16/25={16/25:.3f}) test {cov_te:.4f} (1/25=.04)")
    print("gb_infomatched selftest PASS: planted lin R2 %.2f/%.2f, spline-only quad %.2f (lin %.3f), shuffled max %.3f"
          % (lin0[0, 0], lin0[1, 1], spl0[2, 2], lin0[2, 2], max(np.nanmax(lins), np.nanmax(spls))))


if __name__ == "__main__":
    selftest()
