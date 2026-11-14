# Fig3 pseudotime comparators, stage 1 (compute)
import sys, os, json, time
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); SG = R + 'results/signed/'
sys.path[:0] = [R + 'scratch', R + 'pairspace']
import numpy as np
import gb_substrate as GS, gb_isofull_lib as IL, gb_primary as GP, anndata as ad
SRCS = {'PANC': ['RFX6'], 'GATA1': ['GATA1', 'KLF1'], 'BONEMARROW': ['GATA1', 'RUNX1'], 'CARDIACV': ['TBX5', 'NKX2-5']}
ROOTS = GP.ROOTS; NT = 16; TOL = 1e-6


def nat_spline(x, knots):   # copy of gb_projmech_PM.nat_spline (importing PM parses argv); columns: 1, x, K-2 natural terms
    K = len(knots); x = np.asarray(x, float); cols = [np.ones_like(x), x]; dK = lambda k: (np.maximum(x - knots[k], 0) ** 3 - np.maximum(x - knots[-1], 0) ** 3) / (knots[-1] - knots[k])
    for k in range(K - 2): cols.append(dK(k) - dK(K - 2))
    return np.stack(cols, 1)


# pair-space projection by an arbitrary orthonormal Q (P,m): S = sum(rx*ry)/sum|rx||ry| ; returns (ns, len(gu)); no self handling (caller NaNs it)
def own_score(Q, Xs, Mu, I, J, gu, CH=128):
    dx = Xs[J] - Xs[I]; rx = dx - Q @ (Q.T @ dx); out = np.empty((dx.shape[1], len(gu)))
    for b in range(0, len(gu), CH):
        Mc = Mu[:, gu[b:b + CH]]; y = Mc[J].astype(np.float64) - Mc[I]; ry = y - Q @ (Q.T @ y)
        out[:, b:b + CH] = IL.ratio(rx.T @ ry, np.abs(rx).T @ np.abs(ry))
    return out


def main(system, OUT, nseeds, NG=None, mode='all', s0=0, s1=None):
    s1 = nseeds if s1 is None else s1; CORE = mode in ('all', 'core'); RNDM = mode in ('all', 'rnd'); seeds = list(range(s0, s1)) if RNDM else []
    os.makedirs(OUT, exist_ok=True); log = open(f'{OUT}/log_{system}.txt', 'a')
    def P(*a): s = ' '.join(str(x) for x in a); print(s, flush=True); log.write(s + '\n'); log.flush()
    t00 = time.time(); srcs = SRCS[system]; ns = len(srcs)
    S = GS.load(system, canonical=True); fixture, prs, _ = GS.paths(system, canonical=True); F = ad.read_h5ad(fixture)
    sc = np.load(f'{SG}{system}.npz', allow_pickle=True); axis = np.asarray(sc['genes']).astype(str); Gax = len(axis); AU = np.char.upper(axis)
    fi = {g.upper(): i for i, g in enumerate(np.asarray(F.var_names).astype(str))}; assert all(g in fi for g in AU), 'sc axis genes missing in fixture'
    fc = np.array([fi[g] for g in AU]); sj = [[s_.upper() for s_ in sc['sources'].astype(str)].index(s) for s in srcs]
    unis = []
    for j in sj:
        u = np.asarray(sc['uni'][j]); u = np.flatnonzero(u.astype(bool)) if (u.dtype == bool or (u.dtype == object and isinstance(u.flat[0], (bool, np.bool_)))) else u.astype(int); m = np.zeros(Gax, bool); m[u] = True; unis.append(m)
    gu = np.flatnonzero(np.any(unis, 0))
    if NG: gu = gu[:NG]
    Mfull = F.layers['Mu']; Mu = np.asarray(Mfull[:, fc].toarray() if hasattr(Mfull, 'toarray') else Mfull[:, fc], np.float32)
    ai = {g: i for i, g in enumerate(AU)}; si = np.array([ai[s] for s in srcs]); Xs = np.asarray(S['X'], np.float64)[:, fc[si]]
    P30 = np.asarray(F.obsm['X_pca'], float)[:, :30]; zp = np.load(prs, allow_pickle=True); n = Mu.shape[0]
    Y = np.ascontiguousarray(Mu[:, gu]); selfidx = [int(np.flatnonzero(gu == i)[0]) if i in set(gu) else -1 for i in si]
    Uj = [m[gu] for m in unis]                                                        # (len gu) per-source universe mask
    P(f'{system}: n={n} G_axis={Gax} union_uni={len(gu)} srcs={srcs} nseeds={nseeds}')
    SIGN = {a: {} for a in ['prod', 'T', 'SPL', 'INT', 'P1T']}; SIGN.update({f'RND{s}': {} for s in seeds}); PARITY = []; NP = {}
    for ri, r in enumerate(ROOTS):
        t0 = time.time(); I, J = (np.asarray(a, int) for a in S['pairs'][r]); pt = np.asarray(zp[f'{r}_pt'], float); NP[r] = len(I)
        def eng(Pc, km): res = IL.engine(Xs, Y, Pc, I, J, km=km, self_idx=selfidx, nthreads=NT); return IL.ratio(res['N'], res['D'])   # (km+1, ns, Gu)
        if CORE:
          SIGN['prod'][r] = eng(P30, 8)
          # GATE 1: production parity with the cached signed S
          for jj, s in enumerate(srcs):
              c = np.asarray(sc[f'{r}|U'][sj[jj]], float)[gu][:, :9].T; o = SIGN['prod'][r][:, jj, :]; m = Uj[jj] & (gu != si[jj]); cc, oo = c[:, m], o[:, m]
              both = np.isfinite(cc) & np.isfinite(oo); d = float(np.abs(cc[both] - oo[both]).max()); mm = int((np.isfinite(cc) != np.isfinite(oo)).sum())
              PARITY.append(dict(root=r, src=s, check='S_k0..8 vs cached', max_abs=d, finite_mask_mismatch=mm)); P(f'GATE1 {system} {r} {s}: max|S-cached|={d:.3e} mask mismatches={mm}')
              if d > TOL or mm: P('GATE1 FAIL -> STOP'); sys.exit(2)
          # T (Q = t/|t|): single depth k=1 of a 1-column P
          t = pt[J] - pt[I]; SIGN['T'][r] = eng(pt[:, None], 1)
          # GATE 2: direct numpy vs engine on 5 genes (source 0, first universe genes)
          jj = 0; gg = np.flatnonzero(Uj[0] & (gu != si[0]))[:5]; dx = Xs[J, 0] - Xs[I, 0]; rx = dx - t * (t @ dx) / (t @ t); dd = []
          for g in gg:
              y = Y[J, g].astype(np.float64) - Y[I, g]; ry = y - t * (t @ y) / (t @ t); direct = (rx @ ry) / np.abs(rx).dot(np.abs(ry)); dd.append(abs(direct - SIGN['T'][r][1, jj, g]))
          PARITY.append(dict(root=r, src=srcs[0], check='Q=t engine vs direct numpy (5 genes)', max_abs=float(max(dd)), finite_mask_mismatch=0)); P(f'GATE2 {system} {r}: max|engine-direct|={max(dd):.3e}')
          if max(dd) > TOL: P('GATE2 FAIL -> STOP'); sys.exit(2)
          # SPL: 8 directions (9 knots at evenly spaced quantiles .05..95, constant dropped), engine QRs the pair differences in column order
          kn = np.quantile(pt, np.linspace(.05, .95, 9)); Bs = nat_spline(pt, kn)[:, 1:]; assert Bs.shape[1] == 8; SIGN['SPL'][r] = eng(Bs, 8)
          # diagnostics: pair-space intercept (ones/sqrt P) and [1,t]
          Pn = len(I); q1 = np.ones((Pn, 1)) / np.sqrt(Pn); SIGN['INT'][r] = own_score(q1, Xs, Mu, I, J, gu)[None]
          q2, _ = np.linalg.qr(np.column_stack([np.ones(Pn), t])); SIGN['P1T'][r] = own_score(q2, Xs, Mu, I, J, gu)[None]
          for jj in range(ns):
              if selfidx[jj] >= 0: SIGN['INT'][r][0, jj, selfidx[jj]] = np.nan; SIGN['P1T'][r][0, jj, selfidx[jj]] = np.nan
        # RND
        for s in seeds:
            rg = np.random.default_rng([20261021, s]); SIGN[f'RND{s}'][r] = eng(rg.standard_normal((n, 8)), 8); P(f'{system} {r} RND seed {s} {time.time() - t0:.0f}s')
        P(f'{system} {r} done {time.time() - t0:.0f}s  n_pairs={NP[r]}')
    # aggregation per source -> Z vectors on the sc axis
    def Zvec(arm, jj, cols):
        per = {h: np.abs(SIGN[arm][h][cols, jj, :][:, Uj[jj]]).T for h in ROOTS}; z = GP.zcons(per, tuple(range(len(cols)))); o = np.full(Gax, np.nan); o[gu[Uj[jj]]] = z; return o
    out = {'genes': axis}; SUP = []
    for jj, s in enumerate(srcs):
        if CORE:
            prod = Zvec('prod', jj, list(range(9))); cache = np.full(Gax, np.nan)
            zc = {h: np.stack([GP.zcol(np.abs(np.asarray(sc[f'{h}|U'][sj[jj]], float)[unis[jj], k])) for k in range(9)], 1) for h in ROOTS}; cache[unis[jj]] = GP.zcons(zc, GP.KS)
            a, b = prod[unis[jj]], cache[unis[jj]]; ok = np.isfinite(a) & np.isfinite(b)
            dz = float(np.abs(a[ok] - b[ok]).max()) if NG is None else float('nan'); mm = int((np.isfinite(a) != np.isfinite(b)).sum()) if NG is None else -1
            PARITY.append(dict(root='both', src=s, check='production Z (9-depth, root-min) vs cached-S Z', max_abs=dz, finite_mask_mismatch=mm)); P(f'GATE1b {system} {s}: max|Zprod - Zcached|={dz:.3e} mismatches={mm}')
            if NG is None and (dz > TOL or mm): P('GATE1b FAIL -> STOP'); sys.exit(2)
            arms = {'D': prod, 'A': Zvec('prod', jj, [0]), 'Q1': Zvec('prod', jj, [1]), 'T': Zvec('T', jj, [1]), 'SPL': Zvec('SPL', jj, list(range(9))), 'INT': Zvec('INT', jj, [0]), 'P1T': Zvec('P1T', jj, [0])}
            for k, v in arms.items(): out[f'{s}|{k}'] = v
            out[f'{s}|uni'] = unis[jj]
            for k, v in arms.items(): SUP.append(dict(system=system, src=s, arm=k, n_uni=int(unis[jj].sum()), n_finite=int(np.isfinite(v[unis[jj]]).sum())))
        if RNDM:
            rnd = np.stack([Zvec(f'RND{q}', jj, list(range(9))) for q in seeds]); out[f'{s}|RND'] = rnd; out[f'{s}|seeds'] = np.array(seeds)
            for q, v in zip(seeds, rnd): SUP.append(dict(system=system, src=s, arm=f'RND{q}', n_uni=int(unis[jj].sum()), n_finite=int(np.isfinite(v[unis[jj]]).sum())))
    tag = 'core' if mode == 'core' else (f'rnd{s0}-{s1}' if mode == 'rnd' else 'all')
    np.savez_compressed(f'{OUT}/arms_{system}_{tag}.npz', **out)
    import hashlib; shaf = lambda f: hashlib.sha256(open(f, 'rb').read()).hexdigest()   # manifest: code + input sha256 (receipt is written only after both gates passed; gates sys.exit(2) before)
    json.dump(dict(npz_sha256=shaf(f'{OUT}/arms_{system}_{tag}.npz'), parity=PARITY, support=SUP, n_pairs=NP, nseeds=nseeds, seeds=seeds, mode=mode, ngenes_limit=NG, runtime=time.time() - t00, code_sha256=shaf(__file__), status='PASS',
                   inputs={'fixture': [fixture, shaf(fixture)], 'pairs': [prs, shaf(prs)], 'signed_cache': [f'{SG}{system}.npz', shaf(f'{SG}{system}.npz')]}), open(f'{OUT}/receipt_{system}_{tag}.json', 'w'), indent=1)
    P(f'{system} DONE {time.time() - t00:.0f}s')


if __name__ == '__main__':
    a = sys.argv; main(a[1].upper(), a[2].rstrip('/') + '/', int(a[a.index('--nseeds') + 1]) if '--nseeds' in a else 20, int(a[a.index('--ngenes') + 1]) if '--ngenes' in a else None,
         a[a.index('--mode') + 1] if '--mode' in a else 'all', int(a[a.index('--s0') + 1]) if '--s0' in a else 0, int(a[a.index('--s1') + 1]) if '--s1' in a else None)
