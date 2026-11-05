#!/usr/bin/env python
"""M1 calibrated count-scale source innovation (PI_REALDATA_OPERATION VI-X; internal review r109 REQUIRED ACTION 2,3,6).
Core kernels only; the feasibility runner and the calibration harness import these.

hashed_anchors(cell_ids, n, seed)       lowest SHA256(seed||cell_id), serialised roster hash
split_counts(Y, gene_keys, seed)        per-gene seeded binomial thinning of integer counts -> independent halves Y1, Y2
rho_full(r1, r2)                        full-depth reliability r1'r2 / (r'r), r=(r1+r2)/2, r_h = M(2 Y_h / e)  (internal review r109 ruling 4)
                                        not clipped; equals Var_bio / (Var_bio + full-depth Poisson noise) in expectation
boot_rho_simultaneous(...)              B common cell-level multinomial resamples shared by every anchor and source; the
                                        residualising projection (Q2 basis, optionally + rivals) is REFIT (weighted) inside every
                                        resample; max-statistic studentised one-sided simultaneous lower bounds over anchors x sources
"""
import hashlib
import numpy as np


def hashed_anchors(cell_ids, n_anchor, seed):
    keys = [hashlib.sha256(f"{seed}||{c}".encode()).hexdigest() for c in cell_ids]
    order = np.argsort(np.array(keys), kind="stable")[:n_anchor]
    idx = np.sort(order)
    roster = hashlib.sha256("\n".join(sorted(cell_ids[i] for i in idx)).encode()).hexdigest()[:16]   # order invariant
    return idx, roster


def split_counts(Y, gene_keys, seed):
    Y = np.asarray(Y)
    if not (np.all(np.isfinite(Y)) and np.all(Y >= 0) and np.array_equal(Y, np.round(Y))):
        raise ValueError("split_counts requires finite non-negative integer counts (internal review r86 D8)")   # fail closed before casting
    Y = Y.astype(np.int64)
    Y1 = np.column_stack([np.random.default_rng([seed, int(hashlib.sha256(g.encode()).hexdigest()[:8], 16)]).binomial(Y[:, j], 0.5)
                          for j, g in enumerate(gene_keys)])
    return Y1, Y - Y1


def basis(Z, tol=1e-10):
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    return U[:, s > tol * s.max()]


def q2_design(D):
    k = D.shape[1]
    return np.column_stack([np.ones(len(D))] + [D[:, j] for j in range(k)] + [D[:, i] * D[:, j] for i in range(k) for j in range(i, k)])


def rho_full(r1, r2, tol=1e-12):
    r = (r1 + r2) / 2; den = (r * r).sum(0)
    return np.where(den > tol, (r1 * r2).sum(0) / np.where(den > tol, den, 1.0), np.nan)   # degenerate -> NaN = UNRESOLVED


def wresid(Q, w, Y):
    """weighted least-squares residual of columns of Y on orthonormal basis Q with cell weights w (batched over resamples)
    Q: m x p ; w: B x m ; Y: m x S  -> B x m x S (residuals weighted by sqrt w for the inner product)"""
    G = np.einsum("bm,mp,mq->bpq", w, Q, Q)
    cond = np.linalg.cond(G)
    bad = ~np.isfinite(cond) | (cond > 1e8)                                                     # weighted rank loss diagnosed, no ridge rescue
    G = np.where(bad[:, None, None], np.eye(Q.shape[1])[None], G)
    rhs = np.einsum("bm,mp,ms->bps", w, Q, Y)
    coef = np.linalg.solve(G, rhs)
    R = Y[None] - np.einsum("mp,bps->bms", Q, coef)
    R[bad] = np.nan
    return R


def rho_weighted(w, R1, R2):
    R = (R1 + R2) / 2
    den = np.einsum("bm,bms,bms->bs", w, R, R)
    return np.where(den > 1e-12, np.einsum("bm,bms,bms->bs", w, R1, R2) / np.where(den > 1e-12, den, 1.0), np.nan)


def boot_rho_simultaneous(neigh, Qs, X1, X2, n_cells, B, seed, alpha=0.05, chunk=500, pool=None, fail_max=0.01):
    """neigh: anchor neighbourhood index arrays (global indices); Qs: orthonormal nuisance bases (m x p); X1, X2: n x S normalised
    half-rates 2Y_h/e. pool: indices of the ANALYSIS cell pool; multinomial weights of total size len(pool) are drawn over the pool
    only and shared by every anchor/source (internal review r111 defect 6). A draw FAILS for the family if any anchor/source bootstrap value is
    undefined; if the union failed-draw fraction exceeds fail_max the family returns crit=NaN (FAMILY_BOOTSTRAP_FAILURE).
    Returns rho_hat, lower bound, bootstrap sd, crit, info(dict: per-cell failed fraction, union failed fraction)."""
    A, S = len(neigh), X1.shape[1]
    pool = np.arange(n_cells) if pool is None else np.asarray(pool)
    pos = np.full(n_cells, -1); pos[pool] = np.arange(len(pool))
    assert all((pos[N] >= 0).all() for N in neigh), "neighbourhood cell outside the bootstrap pool"
    rho = np.zeros((A, S))
    for a, (N, Q) in enumerate(zip(neigh, Qs)):
        r1 = X1[N] - Q @ (Q.T @ X1[N]); r2 = X2[N] - Q @ (Q.T @ X2[N]); rho[a] = rho_full(r1, r2)
    rng = np.random.default_rng(seed)
    boots = np.zeros((B, A, S), dtype=np.float64)
    for b0 in range(0, B, chunk):
        nb = min(chunk, B - b0)
        Wp = rng.multinomial(len(pool), np.full(len(pool), 1.0 / len(pool)), size=nb).astype(float)   # analysis-pool weights
        for a, (N, Q) in enumerate(zip(neigh, Qs)):
            w = Wp[:, pos[N]]
            boots[b0:b0 + nb, a] = rho_weighted(w, wresid(Q, w, X1[N]), wresid(Q, w, X2[N]))
    # internal review r112 ruling 2: family members = cells with a finite point estimate AND a finite positive bootstrap SD. Non-members are
    # individually UNRESOLVED and enter neither the max statistic nor the union-failure set. A draw fails if ANY member is undefined.
    sd = np.nanstd(boots, 0)
    member = np.isfinite(rho) & np.isfinite(sd) & (sd > 0)
    bm = boots[:, member]
    bad_draw = np.isnan(bm).any(1) if bm.size else np.zeros(B, bool)
    info = dict(cell_failed_frac=np.isnan(boots).mean(0), union_failed_frac=float(bad_draw.mean()) if member.any() else None,
                members=member, n_members=int(member.sum()), status="OK")
    if not member.any():
        info["status"] = "NO_MEMBERS"
        return rho, np.full_like(rho, np.nan), sd, float("nan"), info
    if info["union_failed_frac"] > fail_max:
        info["status"] = "FAMILY_BOOTSTRAP_FAILURE"
        return rho, np.full_like(rho, np.nan), sd, float("nan"), info
    dev = np.where(member[None], (boots - rho[None]) / np.where(member, sd, 1.0)[None], -np.inf)
    dev = np.where(np.isnan(dev), np.inf, dev)                                                  # failed draw of a member = worst case
    tmax = dev.reshape(B, -1).max(1)                                                            # max-z over the declared family
    crit = float(np.quantile(tmax, 1 - alpha))
    return rho, np.where(member, rho - crit * sd, np.nan), sd, crit, info


def selftest():
    rng = np.random.default_rng(0)
    ids = [f"c{i}" for i in range(500)]
    perm = rng.permutation(500); idsp = [ids[i] for i in perm]
    a1, h1 = hashed_anchors(ids, 10, 7); a2, h2 = hashed_anchors(idsp, 10, 7)
    assert h1 == h2 and {ids[i] for i in a1} == {idsp[i] for i in a2} and len(set(a1)) == 10          # genuine permutation
    # count model log lambda = m(W) + u. LOCAL Q2 neighbourhood (m=300 of 4000): ~0 at zero innovation, positive with innovation.
    # KNOWN LIMITATION (documented, 2026-09-13): a GLOBAL / wide quadratic count-scale residualiser leaves exp-curvature misfit
    # shared by both halves -> false reliability at zero innovation (grad 2.0: m=1000 .21, m=4000 .99). Falsifier 1 must be
    # calibrated at real-matched state gradients and neighbourhood sizes.
    n = 4000; Wst = rng.normal(size=(n, 2)); e = rng.lognormal(0, .3, n)
    N = np.argsort(((Wst - Wst[0]) ** 2).sum(1))[:300]; Q = basis(q2_design(Wst[N] - Wst[0]))
    for vu, lo, hi in ((0.0, -.10, .08), (0.5, .20, 1.0)):
        lam = np.exp(1.0 + 0.8 * Wst[:, 0] + rng.normal(size=n) * np.sqrt(vu))
        Y1, Y2 = split_counts(rng.poisson(lam * e)[:, None], ["g"], 3)
        X1, X2 = 2 * Y1 / e[:, None], 2 * Y2 / e[:, None]
        rh = rho_full(X1[N] - Q @ (Q.T @ X1[N]), X2[N] - Q @ (Q.T @ X2[N]))[0]
        assert lo < rh < hi, (vu, rh)
    lam = np.exp(1.0 + 2.0 * Wst[:, 0]); Y1, Y2 = split_counts(rng.poisson(lam * e)[:, None], ["g"], 5)
    Qg = basis(q2_design(Wst)); X1, X2 = 2 * Y1 / e[:, None], 2 * Y2 / e[:, None]
    assert rho_full(X1 - Qg @ (Qg.T @ X1), X2 - Qg @ (Qg.T @ X2))[0] > .5                    # documents the global misfit failure
    # batched weighted residual equals explicit weighted LS
    Qs = basis(q2_design(Wst[:200])); w = rng.multinomial(200, np.full(200, .005), size=3).astype(float); Yt = rng.normal(size=(200, 2))
    R = wresid(Qs, w, Yt)
    for b in range(3):
        sw = np.sqrt(w[b]); c = np.linalg.lstsq(Qs * sw[:, None], Yt * sw[:, None], rcond=None)[0]
        assert np.allclose(R[b], Yt - Qs @ c, atol=1e-6)
    # simultaneous bound: zero-innovation anchors -> lower bound below 0.05 for all
    lam = np.exp(1.0 + 0.8 * Wst[:, 0]); Y = rng.poisson(lam * e)[:, None]; Y1, Y2 = split_counts(Y, ["g"], 4)
    X1, X2 = 2 * Y1 / e[:, None], 2 * Y2 / e[:, None]
    neigh = [np.argsort(((Wst - Wst[i]) ** 2).sum(1))[:300] for i in range(5)]
    Qn = [basis(q2_design(Wst[N] - Wst[N[0]])) for N in neigh]
    rho, lb, sd, crit, _ = boot_rho_simultaneous(neigh, Qn, X1, X2, n, 199, 1)
    assert (lb < .05).all() and crit > 1.5, (lb, crit)
    for bad in (np.array([[1.5]]), np.array([[-1]])):
        try: split_counts(bad, ["g"], 0); raise AssertionError("fail-open")
        except ValueError: pass
    assert np.isnan(rho_full(np.zeros((5, 1)), np.zeros((5, 1))))[0]
    print("M1 selftest PASS (hashed anchors, rho_full zero/positive, weighted refit == explicit, simultaneous LB at zero innovation)")


if __name__ == "__main__":
    selftest()
