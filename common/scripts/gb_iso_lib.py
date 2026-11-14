# Near-isochronic pair library: pair construction, rewiring, pair-sum kernels, graph/smoothing diagnostics
import hashlib, os, sys
import numpy as np

R0 = __import__("os").environ["GEOBRIDGE_ROOT"]
SG = R0 + '/results/signed/'
ROOTS = ('dc1_min', 'dc1_max')
EPS = 1e-12
WINDOW, CAP, SEEDS, REWIRE_SEED = 0.02, 40, (42, 43, 44), 42


# hashing
def sha_file(p, blk=1 << 24):
    h = hashlib.sha256()
    with open(p, 'rb') as fh:
        for b in iter(lambda: fh.read(blk), b''):
            h.update(b)
    return h.hexdigest()


def sha_np(*arrs):
    h = hashlib.sha256()
    for a in arrs:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


# pair construction
# Near-isochronic unordered pairs
def iso_pairs(pt, strata, window=WINDOW, cap=CAP, seed=42, root_idx=0):
    pt = np.asarray(pt, np.float64); n = len(pt); strata = np.asarray(strata)
    assert strata.shape == (n,) and np.isfinite(pt).all()
    lo = np.zeros(n, np.int64); hi = np.zeros(n, np.int64); pos = np.zeros(n, np.int64); order_of = {}
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s)
        o = idx[np.lexsort((idx, pt[idx]))]
        p = pt[o]
        lo[o] = np.searchsorted(p, p - window, 'left'); hi[o] = np.searchsorted(p, p + window, 'right'); pos[o] = np.arange(len(o))
        order_of[s] = o
    rng = np.random.default_rng([int(seed), int(root_idx)])
    A, B = [], []
    for i in range(n):
        m = int(hi[i] - lo[i] - 1)
        if m <= 0:
            continue
        o = order_of[strata[i]]
        offs = np.arange(m) if m <= cap else rng.choice(m, cap, replace=False)
        p = lo[i] + offs
        p = p + (p >= pos[i])                       # skip the cell's own sorted position
        A.append(np.full(len(p), i, np.int64)); B.append(o[p])
    a = np.concatenate(A); b = np.concatenate(B)
    key = np.unique(np.minimum(a, b) * n + np.maximum(a, b))
    return (key // n).astype(np.int64), (key % n).astype(np.int64)


# Shuffle endpoint stubs within strata and repair invalid edges by degree-preserving swaps
def rewire_pairs(I, J, strata, seed=REWIRE_SEED, max_try=5000):
    I = np.asarray(I, np.int64); J = np.asarray(J, np.int64); strata = np.asarray(strata); n = len(strata)
    assert (strata[I] == strata[J]).all(), 'input pairs cross strata'
    rng = np.random.default_rng([int(seed), 7])
    su, sv, ss = [], [], []
    for s in np.unique(strata[I]):
        m = strata[I] == s
        stubs = np.concatenate([I[m], J[m]]); rng.shuffle(stubs); h = len(stubs) // 2
        su.append(stubs[:h]); sv.append(stubs[h:2 * h]); ss.append(np.full(h, s, dtype=strata.dtype))
    u = np.concatenate(su).copy(); v = np.concatenate(sv).copy(); sE = np.concatenate(ss)
    key = lambda a, b: min(a, b) * n + max(a, b)
    k0 = np.minimum(u, v) * n + np.maximum(u, v)
    selfl = u == v
    _, first = np.unique(k0, return_index=True); dup = np.ones(len(k0), bool); dup[first] = False
    bad = selfl | dup
    present = set(k0[~bad].tolist()); good_by = {s: np.flatnonzero((~bad) & (sE == s)).tolist() for s in np.unique(sE)}
    n_bad0 = int(bad.sum()); unresolved = []
    for e in np.flatnonzero(bad):
        s = sE[e]; ok = False; gl = good_by[s]
        for _ in range(max_try):
            if not gl:
                break
            f = gl[rng.integers(len(gl))]; x, y = int(u[f]), int(v[f])
            if rng.random() < 0.5:
                x, y = y, x
            a, b = int(u[e]), int(v[e])
            if a == y or x == b:
                continue
            k1, k2 = key(a, y), key(x, b)
            if k1 == k2 or k1 in present or k2 in present:
                continue
            present.discard(key(int(u[f]), int(v[f]))); present.add(k1); present.add(k2)
            u[e], v[e] = a, y; u[f], v[f] = x, b; gl.append(int(e)); ok = True
            break
        if not ok:
            unresolved.append(int(e))
    keep = np.ones(len(u), bool); keep[unresolved] = False
    kk = np.unique(np.minimum(u[keep], v[keep]) * n + np.maximum(u[keep], v[keep]))
    I2, J2 = (kk // n).astype(np.int64), (kk % n).astype(np.int64)
    info = dict(n_in=int(len(I)), n_out=int(len(I2)), n_bad_initial=n_bad0, n_unresolved_dropped=len(unresolved))
    return I2, J2, info


# keep pairs whose two cells share a stratum (forward-pair reference restricted to the iso strata)
def restrict_pairs(I, J, strata):
    m = np.asarray(strata)[I] == np.asarray(strata)[J]
    return I[m], J[m]


def degree_stats(I, J, n):
    deg = np.bincount(np.concatenate([I, J]), minlength=n); d = deg[deg > 0].astype(np.float64)
    if len(d) == 0:
        return dict(n_cells=0, max=np.nan, median=np.nan, max_over_median=np.nan, gini=np.nan)
    ds = np.sort(d); k = len(ds); gini = float((2 * np.arange(1, k + 1) - k - 1).dot(ds) / (k * ds.sum()))
    return dict(n_cells=int(k), max=float(ds.max()), median=float(np.median(ds)), max_over_median=float(ds.max() / np.median(ds)), gini=gini)


# graph / smoothing diagnostics
def sym_bool(C):
    import scipy.sparse as sp
    B = (sp.csr_matrix(C) > 0).astype(np.uint8).tocsr(); B = B.maximum(B.T).tocsr(); B.setdiag(0); B.eliminate_zeros()
    return B


def with_self(B):
    import scipy.sparse as sp
    return (B + sp.identity(B.shape[0], dtype=np.uint8, format='csr')).tocsr()


# Direct adjacency in the (symmetrised) kNN graph and Jaccard overlap of the two cells' smoothing neighbourhoods
def adj_jaccard(Bsym, Aself, I, J, chunk=250000):
    P = len(I); adj = np.empty(P, bool); jac = np.empty(P, np.float32); sz = np.asarray(Aself.sum(1)).ravel().astype(np.float64)
    for a in range(0, P, chunk):
        i, j = I[a:a + chunk], J[a:a + chunk]
        adj[a:a + chunk] = np.asarray(Bsym[i, j]).ravel() > 0
        inter = np.asarray(Aself[i].multiply(Aself[j]).sum(1)).ravel().astype(np.float64)
        jac[a:a + chunk] = inter / (sz[i] + sz[j] - inter)
    return adj, jac


# kernels
# X (n, ns) float64 source matrix, Y (n, G) float32/64 target matrix, pairs (I,J) with dx = X[J]-X[I], dy = Y[J]-Y[I] (float64)
def pair_sums(X, Y, I, J, centred=False, pb=50000, gc=500):
    I = np.asarray(I, np.int64); J = np.asarray(J, np.int64); n, ns = X.shape; G = Y.shape[1]; P = len(I)
    dcnt = (np.bincount(J, minlength=n) - np.bincount(I, minlength=n)).astype(np.float64)
    sx = dcnt @ X; sy = np.empty(G)
    for g0 in range(0, G, gc):
        sy[g0:g0 + gc] = dcnt @ np.asarray(Y[:, g0:g0 + gc], np.float64)
    mx = sx / P if centred else np.zeros(ns); my = sy / P if centred else np.zeros(G)
    N = np.zeros((ns, G)); D = np.zeros((ns, G)); Sxx = np.zeros(ns); Syy = np.zeros(G)
    for g0 in range(0, G, gc):
        sl = slice(g0, min(G, g0 + gc)); Yc = np.ascontiguousarray(Y[:, sl])
        for a in range(0, P, pb):
            Ib, Jb = I[a:a + pb], J[a:a + pb]
            DX = X[Jb] - X[Ib] - mx
            DY = Yc[Jb].astype(np.float64) - Yc[Ib].astype(np.float64) - my[sl]
            N[:, sl] += DX.T @ DY
            D[:, sl] += np.abs(DX).T @ np.abs(DY)
            Syy[sl] += (DY * DY).sum(0)
            if g0 == 0:
                Sxx += (DX * DX).sum(0)
    return dict(N=N, D=D, Sxx=Sxx, Syy=Syy, sx=sx, sy=sy, P=P)


def ratio_C(N, D):
    ok = D > EPS
    return np.where(ok, N / np.where(ok, D, 1.0), np.nan)


def ratio_PP(N, Sxx, Syy):
    den = np.sqrt(Sxx)[:, None] * np.sqrt(Syy)[None, :]
    ok = den > EPS
    return np.where(ok, N / np.where(ok, den, 1.0), np.nan)


# cell-level Pearson of each source column of Xs (n,ns) against each column of Y (n,G) (gb_grand_stage1 `pear`)
def cell_pearson(Xs, Y, gc=1000):
    A = Xs - Xs.mean(0); na = np.linalg.norm(A, axis=0); G = Y.shape[1]; out = np.empty((Xs.shape[1], G))
    for g0 in range(0, G, gc):
        B = np.asarray(Y[:, g0:g0 + gc], np.float64); B = B - B.mean(0)
        out[:, g0:g0 + gc] = (A.T @ B) / (na[:, None] * np.linalg.norm(B, axis=0)[None, :] + 1e-300)
    return out


# Return N and D after projecting pair differences off Q[:, :k].
def deepk_sums(X, Y, I, J, Q, k, gc_pairs=3e7):
    I = np.asarray(I, np.int64); J = np.asarray(J, np.int64); P = len(I); G = Y.shape[1]
    DX = X[J] - X[I]; Qk = Q[:, :k]; DXk = DX - Qk @ (Qk.T @ DX); ADXk = np.abs(DXk)
    N = np.zeros((X.shape[1], G)); D = np.zeros_like(N); CH = max(10, int(gc_pairs / P))
    for g0 in range(0, G, CH):
        sl = slice(g0, min(G, g0 + CH)); Yc = np.ascontiguousarray(Y[:, sl])
        dy = Yc[J].astype(np.float64) - Yc[I].astype(np.float64)
        ryk = dy - Qk @ (Qk.T @ dy)
        N[:, sl] = DXk.T @ ryk; D[:, sl] = ADXk.T @ np.abs(ryk)
    return dict(N=N, D=D)


# selftest (synthetic; algebra + construction checks)
def selftest():
    rng = np.random.default_rng(0)
    # 1. iso_pairs vs brute force (no sampling when cap is huge)
    n = 400; pt = rng.random(n); pt[:20] = pt[20:40]                     # ties
    st = rng.integers(0, 3, n); w = 0.05
    I, J = iso_pairs(pt, st, window=w, cap=10 ** 6, seed=1, root_idx=0)
    bf = set()
    for i in range(n):
        for j in range(n):
            if i != j and st[i] == st[j] and pt[j] >= pt[i] - w and pt[j] <= pt[i] + w:
                bf.add((min(i, j), max(i, j)))
    got = set(zip(I.tolist(), J.tolist()))
    assert got == bf, ('iso brute-force mismatch', len(got), len(bf))
    assert (I < J).all() and np.array_equal(np.lexsort((J, I)), np.arange(len(I)))
    # 2. capped: every returned pair is valid, per-cell outgoing <= cap, determinism, seed
    # dependence
    I1, J1 = iso_pairs(pt, st, window=w, cap=5, seed=1, root_idx=0); I2, J2 = iso_pairs(pt, st, window=w, cap=5, seed=1, root_idx=0); I3, J3 = iso_pairs(pt, st, window=w, cap=5, seed=2, root_idx=0)
    assert np.array_equal(I1, I2) and np.array_equal(J1, J2) and not (len(I1) == len(I3) and np.array_equal(J1, J3))
    assert set(zip(I1.tolist(), J1.tolist())) <= bf and len(I1) <= 5 * n
    assert len(set(zip(I1.tolist(), J1.tolist()))) == len(I1)
    # 3. rewire: degrees preserved exactly (when nothing is dropped), no self pairs, no duplicates,
    # within strata, time constraint gone
    Ir, Jr, info = rewire_pairs(I1, J1, st, seed=42)
    assert info['n_unresolved_dropped'] == 0, info
    d0 = np.bincount(np.concatenate([I1, J1]), minlength=n); d1 = np.bincount(np.concatenate([Ir, Jr]), minlength=n)
    assert np.array_equal(d0, d1) and (Ir != Jr).all() and (st[Ir] == st[Jr]).all() and len(set(zip(Ir.tolist(), Jr.tolist()))) == len(Ir)
    assert np.abs(pt[Ir] - pt[Jr]).mean() > 2 * np.abs(pt[I1] - pt[J1]).mean(), 'rewiring did not remove the time constraint'
    # 4. kernel vs naive algebra, uncentred + centred, chunk-size invariance
    ns, G, nn = 3, 17, 200; X = rng.normal(size=(nn, ns)); Y = rng.normal(size=(nn, G)).astype(np.float32)
    Ip = rng.integers(0, nn, 5000); Jp = rng.integers(0, nn, 5000)
    for cen in (False, True):
        r = pair_sums(X, Y, Ip, Jp, centred=cen, pb=777, gc=5)
        dx = X[Jp] - X[Ip]; dy = Y[Jp].astype(np.float64) - Y[Ip].astype(np.float64)
        if cen: dx = dx - dx.mean(0); dy = dy - dy.mean(0)
        assert np.allclose(r['N'], dx.T @ dy, rtol=1e-10, atol=1e-9) and np.allclose(r['D'], np.abs(dx).T @ np.abs(dy), rtol=1e-10, atol=1e-9)
        assert np.allclose(r['Sxx'], (dx ** 2).sum(0)) and np.allclose(r['Syy'], (dy ** 2).sum(0))
    # 5. orientation invariance of iso concordance / Pearson (swap I<->J on a random subset leaves C
    # and PP unchanged), centring is a no-op after symmetrisation
    flip = rng.random(len(I1)) < 0.5; If = np.where(flip, J1, I1); Jf = np.where(flip, I1, J1)
    Xs = rng.normal(size=(n, 2)); Ys = rng.normal(size=(n, 9)).astype(np.float32)
    a = pair_sums(Xs, Ys, I1, J1); b = pair_sums(Xs, Ys, If, Jf)
    assert np.allclose(ratio_C(a['N'], a['D']), ratio_C(b['N'], b['D']), atol=1e-12) and np.allclose(ratio_PP(a['N'], a['Sxx'], a['Syy']), ratio_PP(b['N'], b['Sxx'], b['Syy']), atol=1e-12)
    both = pair_sums(Xs, Ys, np.concatenate([I1, J1]), np.concatenate([J1, I1]), centred=True)       # both orientations => mean-zero => centring no-op
    assert np.allclose(both['sx'], 0, atol=1e-9) and np.allclose(ratio_C(both['N'], both['D']), ratio_C(a['N'], a['D']), atol=1e-12)
    assert np.allclose(ratio_PP(both['N'], both['Sxx'], both['Syy']), ratio_PP(a['N'], a['Sxx'], a['Syy']), atol=1e-12)
    # 6. deepk_sums against explicit projector
    Pn = 1500; Ik = rng.integers(0, nn, Pn); Jk = rng.integers(0, nn, Pn); Q = np.linalg.qr(rng.normal(size=(Pn, 30)))[0]
    r = deepk_sums(X, Y, Ik, Jk, Q, 8); dx = X[Jk] - X[Ik]; dy = Y[Jk].astype(np.float64) - Y[Ik].astype(np.float64); H = Q[:, :8] @ Q[:, :8].T
    dxk, dyk = dx - H @ dx, dy - H @ dy
    assert np.allclose(r['N'], dxk.T @ dyk, atol=1e-9) and np.allclose(r['D'], np.abs(dxk).T @ np.abs(dyk), atol=1e-9)
    # 7. adjacency / Jaccard on a toy graph
    import scipy.sparse as sp
    C = sp.csr_matrix(np.array([[0, 1, 1, 0], [1, 0, 0, 0], [1, 0, 0, 1], [0, 0, 1, 0]], float)); B = sym_bool(C); A = with_self(B)
    adj, jac = adj_jaccard(B, A, np.array([0, 0, 1]), np.array([1, 3, 3]))
    assert adj.tolist() == [True, False, False] and np.allclose(jac, [2 / 3, 1 / 4, 0.0]), jac
    # 8. degree stats
    ds = degree_stats(np.array([0, 0, 1]), np.array([1, 2, 2]), 4); assert ds['n_cells'] == 3 and abs(ds['gini']) < 1e-12 and ds['max'] == 2
    print('gb_iso_lib selftest OK')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'selftest':
        selftest()
