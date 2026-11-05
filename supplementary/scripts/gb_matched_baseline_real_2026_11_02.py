"""Real-data matched-baseline ablation on the Fig 3b/c KO endpoints (external reviewer 2026-11-02). Reuses scratch/gb_fig3d_depth_multi_2026_10_29.py
verbatim (rebuild prefix: rows, truth, synchronized module draws; parity cum0-8 == production D, cum0-0 == K0), replacing only the scored views:
same canonical fixture (results/canonical_2026_09_18/<SYS>_canonical.h5ad: Ms, Mu, X_pca[:, :30]), same canonical forward pairs per root, same QR basis
of PC differences and nested projection, same production universe and frozen gb_primary.zcons. Statistics per depth (focal regulator vs every target):
CONC (production |S|, parity vs results/signed/<SYS>.npz asserted), COS = |sum x~y~|/(||x~|| ||y~||), PEAR = |corr(x~,y~)| over pairs; at k=0 and k=0..8.
Contrasts on the same draws: CONC08 - COS08, CONC08 - PEAR08, COS08 - COS0, PEAR08 - PEAR0. DESCRIPTIVE ablation, production unchanged.
usage (server, repo root): script OUT --endpoints A[,B] [--B 9999]"""
import sys, numpy as np, anndata as ad
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); CAN = R + 'results/canonical_2026_09_18/'
src = open(R + 'scratch/gb_fig3d_depth_multi_2026_10_29.py').read().replace('\r', '')
old_subs = src[src.index('SUBS = {'):src.index("PAIRS = [")]; src = src.replace(old_subs, "SUBS = {'cum0-0': [0], 'cum0-8': list(range(9))}\n")
old_pairs = src[src.index('PAIRS = ['):src.index('\n', src.index('PAIRS = ['))]
src = src.replace(old_pairs, "PAIRS = [('cum0-8', 'COS_k08'), ('cum0-8', 'PEAR_k08'), ('COS_k08', 'COS_k0'), ('PEAR_k08', 'PEAR_k0'), ('COS_k0', 'cum0-0'), ('PEAR_k0', 'cum0-0')]")
dense = lambda M: np.asarray(M.todense() if hasattr(M, 'todense') else M)
def matched(fx, tf, genes, uni, A):
    F = ad.read_h5ad(f'{CAN}{fx}_canonical.h5ad'); vn = np.char.upper(np.asarray(F.var_names).astype(str)); pos = {v: i for i, v in enumerate(vn)}
    gi = np.array([pos[x.upper()] for x in genes]); si = pos[tf.upper()]
    Ms = dense(F.layers['Ms']).astype(np.float64); Mu = dense(F.layers['Mu']).astype(np.float64); P30 = np.asarray(F.obsm['X_pca'], float)[:, :30]
    pz = np.load(f'{CAN}{fx}_canonical_pairs.npz', allow_pickle=True); out = {}
    for r in ('dc1_min', 'dc1_max'):
        I, J = np.asarray(pz[f'{r}_I'], int), np.asarray(pz[f'{r}_J'], int); Q, _ = np.linalg.qr(P30[J] - P30[I]); x0 = Ms[J, si] - Ms[I, si]
        S = {s: np.full((len(genes), 9), np.nan) for s in ('CONC', 'COS', 'PEAR')}
        for g0 in range(0, len(gi), 500):
            sl = slice(g0, g0 + 500); Mc = Mu[:, gi[sl]]; y0 = Mc[J] - Mc[I]
            for k in range(9):
                x = x0 - Q[:, :k] @ (Q[:, :k].T @ x0) if k else x0; y = y0 - Q[:, :k] @ (Q[:, :k].T @ y0) if k else y0
                num = x @ y
                with np.errstate(invalid='ignore', divide='ignore'):
                    S['CONC'][sl, k] = np.abs(num) / (np.abs(x) @ np.abs(y)); S['COS'][sl, k] = np.abs(num) / (np.linalg.norm(x) * np.linalg.norm(y, axis=0))
                    xc = x - x.mean(); yc = y - y.mean(0); S['PEAR'][sl, k] = np.abs(xc @ yc) / (np.linalg.norm(xc) * np.linalg.norm(yc, axis=0))
        st = A[r][:, :9]; m = uni[:, None] & np.isfinite(st); d_ = np.nanmax(np.abs(S['CONC'][m] - st[m]))
        print(fx, tf, r, 'CONC parity vs stored |S| (k=0..8, universe):', d_, flush=True); assert d_ < 1e-4, (fx, tf, r, d_)
        out[r] = S
    return out
old_v = 'V = {nm: sub(ks) for nm, ks in SUBS.items()}'; assert old_v in src
src = src.replace(old_v, old_v + """
    _M = matched(fx, tf, genes, uni, A)
    for _s in ('COS', 'PEAR'):
        for _w, _ks in (('k0', [0]), ('k08', list(range(9)))):
            _f = np.full(len(genes), np.nan); _f[uni] = GP.zcons({r: _M[r][_s][uni] for r in GP.ROOTS}, _ks); V[f'{_s}_{_w}'] = mapto(genes, _f, g)""")
exec(compile(src, 'depth_multi', 'exec'))
