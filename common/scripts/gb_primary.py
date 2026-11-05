"""THE one PRIMARY implementation (freeze condition B1, live log §234-§236).  Every evaluator imports this.

PRIMARY(source) = min over the two roots of mean_{k=0..8} z(|S_U(k)|), z within the source over its truth-free
observational universe U_obs.  §234 collapse rule: a depth column with < 3 distinct finite values or sd <= 1e-12(1+|mean|)
is NaN; a gene needs >= ceil(2/3 * #depths) finite depths; a NaN at either root gives NaN (UNSUPPORTED).
No final percentile transform (§236).
"""
import numpy as np

ROOTS = ("dc1_min", "dc1_max")
KS = tuple(range(0, 9))


def zcol(x):
    x = np.asarray(x, float)
    ok = np.isfinite(x)
    out = np.full(len(x), np.nan)
    if ok.sum() < 3 or len(np.unique(x[ok])) < 3:
        return out
    m, sd = x[ok].mean(), x[ok].std()
    if sd > 1e-12 * (1 + abs(m)):
        out[ok] = (x[ok] - m) / sd
    return out


def zcons(per_root, ks=KS):
    """per_root: {root: (G, n_k) raw |S| on U_obs}; returns (G,) PRIMARY (NaN = UNSUPPORTED)."""
    per = []
    for h in ROOTS:
        Zk = np.stack([zcol(per_root[h][:, k]) for k in ks])
        n_ok = np.isfinite(Zk).sum(0)
        with np.errstate(invalid="ignore"):
            m = np.nanmean(Zk, 0)
        m[n_ok < np.ceil(2 * len(ks) / 3)] = np.nan
        per.append(m)
    return np.minimum(*per)  # NaN propagates: either root unsupported -> unsupported


def zcons_matrix(A, ks=KS):
    """Development layout: A = {root: [(nsrc, ntarg) array for k = 0..K]} -> (nsrc, ntarg) PRIMARY (row = source)."""
    nk = max(ks) + 1
    nsrc = np.asarray(A[ROOTS[0]][0]).shape[0]
    return np.vstack([zcons({h: np.column_stack([np.asarray(A[h][k], float)[i] for k in range(nk)]) for h in ROOTS}, ks)
                      for i in range(nsrc)])


GOLDEN_SHA256 = "27893d7fea1eecee7aff343fe331bc9d7dd3e529301dd07c8e8755851c300374"  # frozen 2026-09-26 (_golden); any algebra change must fail loudly


def _golden():
    """Frozen fixture: heavy tails, per-gene missing depths, a collapsed depth, a <3-distinct depth, one-root failure."""
    import hashlib
    rng = np.random.default_rng(12345)
    outs = []
    R = {h: rng.standard_t(3, size=(300, 31)) for h in ROOTS}
    for g in range(60):                                   # 1..4 missing depths inside 0..8 for 60 genes
        R[ROOTS[g % 2]][g, rng.choice(9, 1 + g % 4, replace=False)] = np.nan
    outs.append(zcons(R))
    R2 = {h: R[h].copy() for h in ROOTS}
    R2["dc1_min"][:, 3] = 5.0                             # collapsed depth (sd 0) -> 8/9 depths
    R2["dc1_max"][:, 5] = rng.choice([1.0, 2.0], 300)     # < 3 distinct -> collapsed
    outs.append(zcons(R2))
    R3 = {h: R[h].copy() for h in ROOTS}
    R3["dc1_max"][:, :4] = 1.0                            # 4 collapsed depths at one root -> every gene unsupported
    outs.append(zcons(R3))
    blob = np.round(np.concatenate(outs), 10)
    return hashlib.sha256(np.nan_to_num(blob, nan=-999.0).tobytes()).hexdigest(), blob


def _selftest():
    rng = np.random.default_rng(0)
    R = {h: rng.normal(size=(50, 31)) for h in ROOTS}
    p = zcons(R)
    assert p.shape == (50,) and np.isfinite(p).all()
    R["dc1_min"][:, :4] = 1.0  # 4 of 9 depths constant -> collapse -> 5/9 < 6 finite -> all NaN
    assert np.isnan(zcons(R)).all()
    R = {h: rng.normal(size=(50, 31)) for h in ROOTS}
    R["dc1_max"][7, :] = np.nan  # one gene unsupported at one root
    q = zcons(R)
    assert np.isnan(q[7]) and np.isfinite(np.delete(q, 7)).all()
    # z is shift/scale invariant per depth -> PRIMARY invariant to a positive affine map of |S| at every depth
    R2 = {h: 3.0 * R[h] + 2.0 for h in ROOTS}
    assert np.allclose(zcons(R2), q, equal_nan=True)
    h, blob = _golden()
    assert np.isnan(blob[600:]).all() and np.isfinite(blob[:300]).sum() > 200
    if GOLDEN_SHA256 is None:
        print("GOLDEN (freeze this):", h)
    else:
        assert h == GOLDEN_SHA256, f"gb_primary golden fixture changed: {h}"
    print("gb_primary selftest OK")


if __name__ == "__main__":
    _selftest()
