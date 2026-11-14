# cached score components and axis helpers (diagnostic only)
import sys, os, json, time, hashlib, gzip, glob
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch'), (__import__("os").environ["GEOBRIDGE_ROOT"] + '/pairspace'), __import__("os").environ["GEOBRIDGE_ROOT"]]
import numpy as np, pandas as pd

REPO = __import__("os").environ["GEOBRIDGE_ROOT"]
CANON = f'{REPO}/results/canonical_2026_09_18'
DEC = f'{REPO}/results/signed/decomp_2026_10_14'
WORK = f'{REPO}/results/signed/splitrefit_2026_10_14/work'
ZR = f'{REPO}/results/signed/zrepro_2026_10_14'
WS = f'{REPO}/results/signed/wrongsource_2026_10_14'
EXAMPLES = f'{REPO}/results/signed/wrongsign_diag_2026_10_14/examples.tsv'
ROOTS = ('dc1_min', 'dc1_max'); DEPTHS = (0, 2, 8, 20, 30); EPS = 1e-12
SPLITTAG = {'GATA1': 'random0', 'PANC': 'random0', 'CARDIACV': 'batch-embryo'}
SRC_SYSTEM = {'GATA1': 'GATA1', 'KLF1': 'GATA1', 'TBX5': 'CARDIACV', 'NKX2-5': 'CARDIACV', 'RFX6': 'PANC'}


def sha_file(f, blk=1 << 24):
    h = hashlib.sha256()
    with open(f, 'rb') as fh:
        while True:
            b = fh.read(blk)
            if not b: break
            h.update(b)
    return h.hexdigest()


def axis_of(system):
    return np.load(f'{WS}/wscores_{system}.npz', allow_pickle=True)['genes'].astype(str)


def paths(system, split):
    if split == 'full':
        return f'{CANON}/{system}_canonical.h5ad', f'{CANON}/{system}_canonical_pairs.npz'
    d = f'{WORK}/{system}_{SPLITTAG[system]}_{split}'
    return f'{d}/{system}_canonical.h5ad', f'{d}/{system}_canonical_pairs.npz'


# full-data cached components (N,D,S per k=0..30, c, uni) for both roots; same files as gb_signed_decomp.py
def load_cache(system, source):
    out = {}
    for r in ROOTS:
        z = np.load(f'{DEC}/components_{system}_{r}_{source}.npz', allow_pickle=True)
        out[r] = dict(N=z['N'], D=z['D'], S=z['S'].astype(float), uni=z['uni'].astype(bool), genes=z['genes'].astype(str))
    assert (out[ROOTS[0]]['genes'] == axis_of(system)).all() and (out[ROOTS[1]]['genes'] == axis_of(system)).all(), 'cache axis != wscores axis'
    return out


# roster
def make_roster(system, source, OUT):
    import anndata as ad, gb_primary as GP
    os.makedirs(OUT, exist_ok=True)
    C = load_cache(system, source); axis = C[ROOTS[0]]['genes']; G = len(axis)
    ra = {r: np.abs(np.where(C[r]['uni'][:, None], C[r]['S'][:, :9], np.nan)) for r in ROOTS}
    Z = GP.zcons(ra)                                           # production-Z over the source's observational universe (NaN = unsupported)
    fx, _ = paths(system, 'full'); A = ad.read_h5ad(fx, backed='r'); names = np.char.upper(np.asarray(A.var_names).astype(str))
    hv = {}
    for j, g in enumerate(names): hv.setdefault(g, j)
    colm = np.array([hv.get(g.upper(), -1) for g in axis])
    sp = A.layers['spliced'][:]; det = np.asarray((sp > 0).mean(0)).ravel(); del sp
    det_axis = np.where(colm >= 0, det[np.where(colm >= 0, colm, 0)], np.nan)
    cand = np.flatnonzero(np.isfinite(Z) & np.isfinite(det_axis) & (np.char.upper(axis) != source.upper()))
    def rank_bin(v, nb):
        o = sorted(cand, key=lambda i: (v[i], axis[i])); b = {}
        for r_, i in enumerate(o): b[i] = min(nb - 1, int(nb * r_ / len(o)))
        return b
    zq, dq = rank_bin(Z, 4), rank_bin(det_axis, 3)
    h = lambda g: hashlib.sha256(f'{system}|{source}|{g}'.encode()).hexdigest()
    roster = []
    for a in range(4):
        for b in range(3):
            cell = sorted([i for i in cand if zq[i] == a and dq[i] == b], key=lambda i: (h(axis[i]), axis[i]))[:8]
            roster += [dict(gene=str(axis[i]), idx=int(i), zq=a, dq=b, example=False, group='', prodZ=float(Z[i]), det=float(det_axis[i])) for i in cell]
    ex = pd.read_csv(EXAMPLES, sep='\t'); ex = ex[ex.endpoint.str.upper() == source.upper()]
    have = {r_['gene'].upper() for r_ in roster}; missing = []; au = {g.upper(): i for i, g in enumerate(axis)}
    for _, e in ex.iterrows():
        g = str(e.gene).upper()
        if g not in au: missing.append(dict(gene=str(e.gene), reason='not_on_axis')); continue
        i = au[g]
        if g in have:
            for r_ in roster:
                if r_['gene'].upper() == g: r_['example'] = True; r_['group'] = str(e.group)
            continue
        roster.append(dict(gene=str(axis[i]), idx=int(i), zq=-1, dq=-1, example=True, group=str(e.group), prodZ=float(Z[i]), det=float(det_axis[i]) if np.isfinite(det_axis[i]) else np.nan))
        have.add(g)
    json.dump(dict(system=system, source=source, n_cand=len(cand), n_axis=G, roster=roster, missing_examples=missing,
                   stratum_rule='4 production-Z quartiles (rank bins over finite-Z universe genes) x 3 raw-spliced-detection terciles (marginal rank bins), 8 smallest SHA256(system|source|gene) per cell, tie gene name'),
              open(f'{OUT}/roster_{source}.json', 'w'), indent=1)
    print(source, 'roster', len(roster), 'cand', len(cand), 'examples missing', missing, flush=True)


# geometry
def basis(pt):
    t = (pt - pt.min()) / (pt.max() - pt.min())
    return np.stack([np.ones_like(t), t, t ** 2, t ** 3, np.maximum(t - .5, 0) ** 3], 1)


def hspan(pt, I, J):
    Hm = basis(pt)[J] - basis(pt)[I]; U, s, _ = np.linalg.svd(Hm, full_matrices=False); rk = int((s > 1e-10 * s[0]).sum())
    return U[:, :rk], rk, s


def energy_ret(UH, v):      # v (P,g) or (P,): fraction of ||v||^2 retained by (I - UH UH^T)
    v = v.reshape(len(v), -1); tot = (v ** 2).sum(0); r = v - UH @ (UH.T @ v)
    return (r ** 2).sum(0) / np.where(tot > 0, tot, np.nan)


# run
def run(system, source, split, OUT, NMAX=0):
    import anndata as ad, gb_real_uniform as RU, gb_target_extension as TE
    t0 = time.time(); D_ = f'{OUT}/{source}_{split}'; os.makedirs(D_, exist_ok=True)
    R = json.load(open(f'{OUT}/roster_{source}.json')); ros = R['roster']
    if NMAX: ros = ros[:NMAX]
    axis = axis_of(system); G = len(axis); C = load_cache(system, source)
    h5, pz = paths(system, split); A = ad.read_h5ad(h5); Z = np.load(pz, allow_pickle=True); obs = np.asarray(A.obs_names).astype(str)
    names = np.char.upper(np.asarray(A.var_names).astype(str)); hv = {}
    for j, g in enumerate(names): hv.setdefault(g, j)
    colm = np.array([hv.get(g.upper(), -1) for g in axis]); have = np.flatnonzero(colm >= 0); fc = colm[have]
    sc_ = hv.get(source.upper(), -1); assert sc_ >= 0, 'source absent in fixture'
    Mu = RU.dense(A.layers['Mu'][:, fc], np.float32); xs = RU.dense(A.layers['Ms'][:, [sc_]], np.float64)[:, 0]
    P30 = np.asarray(A.obsm['X_pca'], float)[:, :TE.NPC]
    pos = {int(g): j for j, g in enumerate(have)}                       # axis index -> Mu column
    checks = []; rows = []; geom = []; allg = {}
    chk = lambda name, val, thr, extra='': checks.append(dict(source=source, split=split, check=name, value=val, threshold=thr, status='PASS' if (val <= thr if thr is not None else True) else 'FAIL', note=extra))
    chk('cells_match_npz', float(not all((Z[f'{r}_cells'].astype(str) == obs).all() for r in ROOTS)), 0.0, 'pairs-npz cell ids == fixture obs_names')
    ids = obs; open(f'{D_}/cell_ids.txt', 'w').write('\n'.join(ids))
    for ri, r in enumerate(ROOTS):
        I, J = np.asarray(Z[f'{r}_I'], int), np.asarray(Z[f'{r}_J'], int); P = len(I); n = len(obs); pt = np.asarray(Z[f'{r}_pt'], float)
        if not np.isfinite(pt).all(): raise SystemExit('non-finite pt')
        Q = np.linalg.qr(P30[J] - P30[I])[0]                              # (P,30) float64
        dx = xs[J] - xs[I]; ncell = len(np.unique(np.concatenate([I, J])))
        UH, rkH, sv = hspan(pt, I, J); ov = float(((Q[:, :8].T @ UH) ** 2).sum() / rkH)
        Cx = Q.T @ dx
        ge = [g for g in ros if g['idx'] in pos and g['idx'] != -1]
        miss = [g['gene'] for g in ros if g['idx'] not in pos]
        idx = np.array([pos[g['idx']] for g in ge], int)
        if len(idx):
            dy = (Mu[J][:, idx].astype(np.float64) - Mu[I][:, idx].astype(np.float64)); Cy = Q.T @ dy
            N0 = dx @ dy; D0 = np.abs(dx) @ np.abs(dy); y0e = (dy ** 2).sum(0); x0e = (dx ** 2).sum()
            ident_max = 0.0; cs = np.cumsum(Cx[:, None] * Cy, 0)
            for k in DEPTHS:
                rx = dx - Q[:, :k] @ Cx[:k] if k else dx.copy(); ry = dy - Q[:, :k] @ Cy[:k] if k else dy.copy()
                N = rx @ ry; Dk = np.abs(rx) @ np.abs(ry); Ck = N0 - N
                ident = N0 - (cs[k - 1] if k else 0.0); ident_max = max(ident_max, float(np.nanmax(np.abs(N - ident) / np.where(D0 > 0, D0, np.nan))))
                mx, my = rx.mean(), ry.mean(0); drift = P * mx * my; cov = (rx - mx) @ (ry - my)
                vx = ((rx - mx) ** 2).sum(); vy = ((ry - my) ** 2).sum(0); pear = cov / np.sqrt(vx * vy)
                pxy2 = (rx[:, None] * ry) ** 2; conc = Dk ** 2 / pxy2.sum(0)
                Sk = N / np.where(Dk > EPS, Dk, np.nan)
                Dc = np.abs(rx - mx) @ np.abs(ry - my); Scv = cov / np.where(Dc > EPS, Dc, np.nan)
                Ex = float((rx ** 2).sum() / x0e); Ey = (ry ** 2).sum(0) / y0e
                sret = float(energy_ret(UH, rx)[0]); tret = energy_ret(UH, ry)
                if k == 0: sret0, tret0 = sret, tret
                for a, g in enumerate(ge):
                    w = np.bincount(I, np.abs(rx * ry[:, a]) / 2, n) + np.bincount(J, np.abs(rx * ry[:, a]) / 2, n); tw = w.sum(); w = w / tw if tw > 0 else w
                    cached = C[r]['S'][g['idx'], k] if split == 'full' else np.nan
                    rows.append(dict(system=system, source=source, split=split, root=r, k=k, gene=g['gene'], example=g['example'], group=g['group'], zq=g['zq'], dq=g['dq'], prodZ=g['prodZ'], det=g['det'],
                                     pairs=P, cells=ncell, N=N[a], D=Dk[a], S=Sk[a], C_k=Ck[a], N_over_D0=N[a] / D0[a], C_over_D0=Ck[a] / D0[a], D_over_D0=Dk[a] / D0[a], Ex=Ex, Ey=Ey[a],
                                     mean_x=mx, mean_y=my[a], pearson_c=pear[a], Dc=Dc[a], Sc=Scv[a], drift=drift[a], cov=cov[a], drift_over_D0=drift[a] / D0[a], cov_over_D0=cov[a] / D0[a],
                                     conc=conc[a], cellmass_eff=(1 / (w ** 2).sum()) if tw > 0 else np.nan, cellmass_max=w.max() if tw > 0 else np.nan,
                                     src_ret=sret, tgt_ret=tret[a], overlapQ8=ov, S_cache=cached, S_cache_diff=(abs(Sk[a] - cached) if split == 'full' and np.isfinite(Sk[a]) and np.isfinite(cached) else np.nan),
                                     cache_mask_ok=(bool(np.isfinite(Sk[a]) == np.isfinite(cached)) if split == 'full' else np.nan)))
            chk(f'identity_float64_{r}', ident_max, 1e-9, f'max|N_k-(N_0-sum c_j)|/D_0 over roster x depths {DEPTHS}')
            for k, nm in ((0, 'k0'), (8, 'k8')):
                pass
        geom.append(dict(system=system, source=source, split=split, root=r, pairs=P, cells=ncell, rank_H=rkH, H_sv=json.dumps([float(s) for s in sv]), overlapQ8=ov, n_roster_present=len(ge), roster_missing=json.dumps(miss)))
        # all-gene N, D (halves only; full uses the stored cache) for reproducibility
        if split != 'full' and not os.environ.get('COLS_ONLY'):
            Nn = np.full((G, len(DEPTHS)), np.nan); Dd = np.full_like(Nn, np.nan); CH = max(20, int(4e7 / P)); dxf = dx.astype(np.float32)[:, None]
            for g0 in range(0, len(have), CH):
                sl = slice(g0, min(g0 + CH, len(have))); dyc = (Mu[J, sl] - Mu[I, sl]).astype(np.float32)
                for k, num, den, _z, _a in TE.project_scores(dxf, dyc, Q, 30):
                    if k in DEPTHS: Nn[have[sl], DEPTHS.index(k)] = num[0]; Dd[have[sl], DEPTHS.index(k)] = den[0]
            sa = np.flatnonzero(np.char.upper(axis) == source.upper())
            Nn[sa] = np.nan; Dd[sa] = np.nan; allg[r] = (Nn, Dd, P)
            Sx = f'{ZR}/Sx_{system}_{SPLITTAG[system]}_{split}.npz'
            if os.path.exists(Sx):
                z = np.load(Sx, allow_pickle=True); j = list(z['sources'].astype(str)).index(source); SU = z['SU'][j, ri].astype(float)    # (G,31)
                Sm = np.where(Dd > EPS, Nn / np.where(Dd > EPS, Dd, 1.0), np.nan); a_ = Sm; b_ = SU[:, list(DEPTHS)]
                both = np.isfinite(a_) & np.isfinite(b_); mm = int((np.isfinite(a_) != np.isfinite(b_)).sum())
                chk(f'scorecache_vs_Sx_maxabs_{r}', float(np.abs(a_[both] - b_[both]).max()) if both.any() else np.nan, 1e-6, f'n_cmp={int(both.sum())} mask_mismatch={mm}')
                chk(f'scorecache_vs_Sx_maskmismatch_{r}', float(mm), 0.0, 'identical finite masks')
        print(source, split, r, f'P={P} rankH={rkH} ov={ov:.3f}', f'{time.time()-t0:.0f}s', flush=True)
    df = pd.DataFrame(rows)
    if split == 'full' and len(df):
        chk('scorecache_full_maxabs', float(df.S_cache_diff.max()), 1e-6, f'roster x depths; mask mismatches={int((df.cache_mask_ok == False).sum())}')
        chk('scorecache_full_maskmismatch', float((df.cache_mask_ok == False).sum()), 0.0, 'identical finite masks')
    df.to_csv(f'{D_}/per_gene.tsv', sep='\t', index=False)
    if os.environ.get('COLS_ONLY'): print('cols-only done', source, split); return
    pd.DataFrame(geom).to_csv(f'{D_}/geometry.tsv', sep='\t', index=False); pd.DataFrame(checks).to_csv(f'{D_}/checks.tsv', sep='\t', index=False)
    if allg: np.savez_compressed(f'{D_}/allgene.npz', genes=axis, depths=np.array(DEPTHS), **{f'{r}_N': allg[r][0] for r in ROOTS}, **{f'{r}_D': allg[r][1] for r in ROOTS}, pairs=np.array([allg[r][2] for r in ROOTS]))
    json.dump(dict(system=system, source=source, split=split, fixture=h5, pairs_npz=pz, fixture_sha256=sha_file(h5), pairs_sha256=sha_file(pz), n_cells=len(obs), source_channel='Ms (layers[Ms], float64)',
                   target_channel='Mu', depths=list(DEPTHS), nmax=NMAX, secs=time.time() - t0, cell_id_file='cell_ids.txt', cell_ids_sha256=hashlib.sha256('\n'.join(ids).encode()).hexdigest()), open(f'{D_}/manifest_part.json', 'w'), indent=1)
    print('done', source, split, f'{time.time()-t0:.0f}s', flush=True)


# synthetic
def synth(OUT):
    from geobridge_model.pairspace import build_forward_pairs
    import gb_gata1_preflight as PF
    os.makedirs(OUT, exist_ok=True); rows = []
    for seed in (0, 1):
        for case, beta in (('beta-1', -1.0), ('beta0', 0.0), ('beta+1', 1.0), ('x=2t (no excitation)', None)):
            rng = np.random.default_rng(1000 + seed); n = 1000; t = rng.uniform(0, 1, n)
            x = 2 * t + (.3 * rng.standard_normal(n) if beta is not None else 0); y = 4 * t + ((beta * x + .3 * rng.standard_normal(n)) if beta is not None else 0)
            nuis = np.column_stack([t, rng.standard_normal((n, 29))]); o = np.argsort(t, kind='stable')
            pr = build_forward_pairs(x[o][:, None], y[o][:, None], t[o], min_forward_dt=PF.MIN_FORWARD_DT, max_forward_dt=PF.MAX_FORWARD_DT, max_pairs_per_cell=PF.MAX_PAIRS_PER_CELL, pair_seed=seed)
            I, J = o[np.asarray(pr.pair_i, int)], o[np.asarray(pr.pair_j, int)]; P = len(I)
            Q = np.linalg.qr(nuis[J] - nuis[I])[0]; dx, dy = x[J] - x[I], y[J] - y[I]; Cx, Cy = Q.T @ dx, Q.T @ dy; N0 = dx @ dy; D0 = np.abs(dx) @ np.abs(dy)
            UH, rkH, _ = hspan(t, I, J)
            for k in (0, 1, 2, 8, 20):
                rx = dx - Q[:, :k] @ Cx[:k]; ry = dy - Q[:, :k] @ Cy[:k]; N = rx @ ry; Dk = np.abs(rx) @ np.abs(ry); mx, my = rx.mean(), ry.mean(); cov = (rx - mx) @ (ry - my)
                pear = cov / np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum()) if ((rx - mx) ** 2).sum() > 1e-20 else np.nan
                rows.append(dict(seed=seed, case=case, beta=beta, k=k, pairs=P, N=N, D=Dk, S=N / Dk if Dk > 1e-12 else np.nan, N_over_D0=N / D0, D_over_D0=Dk / D0, Ex=(rx ** 2).sum() / (dx ** 2).sum(),
                                 Ey=(ry ** 2).sum() / (dy ** 2).sum(), drift_over_D0=P * mx * my / D0, cov_over_D0=cov / D0, pearson_c=pear, src_ret=float(energy_ret(UH, rx)[0]), tgt_ret=float(energy_ret(UH, ry)[0]),
                                 identity_err=abs(N - (N0 - np.sum(Cx[:k] * Cy[:k]))) / D0, rankH=rkH))
    pd.DataFrame(rows).to_csv(f'{OUT}/synthetic_control.tsv', sep='\t', index=False); print('synth done', len(rows), flush=True)


# merge
def merge(OUT):
    from scipy.stats import spearmanr
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    parts = sorted(glob.glob(f'{OUT}/*_*/per_gene.tsv')); dirs = [os.path.dirname(p) for p in parts]
    cat = lambda nm: pd.concat([pd.read_csv(f'{d}/{nm}', sep='\t') for d in dirs if os.path.exists(f'{d}/{nm}')], ignore_index=True)
    pg = cat('per_gene.tsv'); ge = cat('geometry.tsv'); ck = cat('checks.tsv')
    pg.to_csv(f'{OUT}/per_gene.tsv', sep='\t', index=False); ck.to_csv(f'{OUT}/checks.tsv', sep='\t', index=False)
    man = dict(spec='internal_design_note_not_shipped_GEOM.md', status='PRELIMINARY - no truth, no p-values', parts=[json.load(open(f'{d}/manifest_part.json')) for d in dirs],
               rosters={os.path.basename(p)[7:-5]: json.load(open(p)) for p in glob.glob(f'{OUT}/roster_*.json')}, script_sha256=sha_file(os.path.abspath(__file__)))
    json.dump(man, open(f'{OUT}/manifest.json', 'w'), indent=1)
    # State-geometry diagnostics and support checks.
    pan = pg[(pg.k == 0) & (~pg.example)].copy(); rows = []
    k8 = pg[(pg.k == 8) & (~pg.example)][['source', 'split', 'root', 'gene', 'src_ret', 'tgt_ret']].rename(columns={'src_ret': 'src_ret8', 'tgt_ret': 'tgt_ret8'})
    pan = pan.merge(k8, on=['source', 'split', 'root', 'gene'], how='left')
    for (s, sp, r), g in pan.groupby(['source', 'split', 'root']):
        gm = ge[(ge.source == s) & (ge.split == sp) & (ge.root == r)].iloc[0]
        f0 = ((g.src_ret >= .20) & (g.tgt_ret >= .10)).mean(); f8 = ((g.src_ret8 >= .20) & (g.tgt_ret8 >= .10)).mean()
        rows.append(dict(system=gm.system, source=s, split=sp, root=r, pairs=gm.pairs, cells=gm.cells, rank_H=gm.rank_H, overlapQ8=gm.overlapQ8, n_panel=len(g),
                         src_ret_k0_med=g.src_ret.median(), tgt_ret_k0_med=g.tgt_ret.median(), src_ret_k8_med=g.src_ret8.median(), tgt_ret_k8_med=g.tgt_ret8.median(),
                         frac_src_ge_p20_k0=(g.src_ret >= .20).mean(), frac_tgt_ge_p10_k0=(g.tgt_ret >= .10).mean(), frac_both_k0=f0, frac_both_k8=f8,
                         pass_k0=bool(gm.overlapQ8 <= .8 and f0 >= .75), pass_k8=bool(gm.overlapQ8 <= .8 and f8 >= .75)))
    sg = pd.DataFrame(rows); sg.to_csv(f'{OUT}/state_geometry.tsv', sep='\t', index=False)
    # reproducibility
    ss = []; sc_data = {}
    for s in sorted(pg.source.unique()):
        system = pg[pg.source == s].system.iloc[0]; fa = f'{OUT}/{s}_A/allgene.npz'; fb = f'{OUT}/{s}_B/allgene.npz'
        if not (os.path.exists(fa) and os.path.exists(fb)): continue
        a, b = np.load(fa, allow_pickle=True), np.load(fb, allow_pickle=True); C = load_cache(system, s)
        for ri, r in enumerate(ROOTS):
            uni = C[r]['uni'] & (np.char.upper(C[r]['genes']) != s.upper()); Pa, Pb = a['pairs'][ri], b['pairs'][ri]
            for di, k in enumerate(DEPTHS):
                Nf, Df, Sf = C[r]['N'][:, k], C[r]['D'][:, k], C[r]['S'][:, k]; Na, Da, Nb, Db = a[f'{r}_N'][:, di], a[f'{r}_D'][:, di], b[f'{r}_N'][:, di], b[f'{r}_D'][:, di]
                D0a, D0b = a[f'{r}_D'][:, 0], b[f'{r}_D'][:, 0]; sup = lambda N_, D_: np.where(D_ > EPS, N_ / np.where(D_ > EPS, D_, 1.), np.nan)
                Sa, Sb = sup(Na, Da), sup(Nb, Db)
                met = {'N/P': (Na / Pa, Nb / Pb), 'D/P': (Da / Pa, Db / Pb), 'N/D0': (Na / D0a, Nb / D0b), 'D/D0': (Da / D0a, Db / D0b), 'S': (Sa, Sb)}
                row = dict(system=system, source=s, root=r, k=k, n_universe=int(uni.sum()), cov_A=float((uni & np.isfinite(Sa)).sum() / uni.sum()), cov_B=float((uni & np.isfinite(Sb)).sum() / uni.sum()),
                           cov_AB=float((uni & np.isfinite(Sa) & np.isfinite(Sb)).sum() / uni.sum()))
                for nm, (u, v) in met.items():
                    m = uni & np.isfinite(u) & np.isfinite(v); row[f'spearman_{nm}'] = float(spearmanr(u[m], v[m])[0]) if m.sum() > 10 else np.nan
                m = uni & np.isfinite(Sa) & np.isfinite(Sb) & (Sa != 0) & (Sb != 0); row['sign_agree_AB'] = float((np.sign(Sa[m]) == np.sign(Sb[m])).mean()) if m.any() else np.nan
                for lab, sgn in (('pos', 1), ('neg', -1)):
                    mf = uni & np.isfinite(Sf) & (np.sign(Sf) == sgn); row[f'n_full_{lab}'] = int(mf.sum())
                    for hn, Sh in (('A', Sa), ('B', Sb)):
                        mm = mf & np.isfinite(Sh); row[f'sign_agree_full_{lab}_{hn}'] = float((np.sign(Sh[mm]) == sgn).mean()) if mm.any() else np.nan
                ss.append(row)
                if k in (0, 8) and ri == 0: sc_data[(s, k)] = (Sa[uni & np.isfinite(Sa) & np.isfinite(Sb)], Sb[uni & np.isfinite(Sa) & np.isfinite(Sb)])
    pd.DataFrame(ss).to_csv(f'{OUT}/split_summary.tsv', sep='\t', index=False)
    with PdfPages(f'{OUT}/geom_phase1_diagnostic.pdf') as pdf:
        fig, axs = plt.subplots(2, max(1, len(set(s for s, _ in sc_data))), figsize=(4 * max(1, len(set(s for s, _ in sc_data))), 8), squeeze=False)
        for ci, s in enumerate(sorted(set(s for s, _ in sc_data))):
            for ri_, k in enumerate((0, 8)):
                u, v = sc_data[(s, k)]; axs[ri_, ci].scatter(u, v, s=2, alpha=.3); axs[ri_, ci].set_title(f'{s} dc1_min k={k}: S half A vs B'); axs[ri_, ci].set_xlabel('S A'); axs[ri_, ci].set_ylabel('S B')
        fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
        for s, g in pg[(pg.split == 'full') & (~pg.example)].groupby('source'):
            fig, axs = plt.subplots(1, 3, figsize=(13, 3.6))
            for r, gg in g.groupby('root'):
                med = gg.groupby('k')[['N_over_D0', 'D_over_D0', 'drift_over_D0', 'cov_over_D0']].median()
                axs[0].plot(med.index, med.N_over_D0, label=r); axs[1].plot(med.index, med.D_over_D0, label=r); axs[2].plot(med.index, med.drift_over_D0, label=f'{r} drift'); axs[2].plot(med.index, med.cov_over_D0, '--', label=f'{r} cov')
            for a_, t_ in zip(axs, ('median N_k/D_0', 'median D_k/D_0', 'drift (solid) & centered cov (dashed) /D_0')): a_.set_title(f'{s}: {t_}', fontsize=8); a_.legend(fontsize=6); a_.set_xlabel('k')
            fig.tight_layout(); pdf.savefig(fig); plt.close(fig)
    print('merged', len(pg), 'rows')


if __name__ == '__main__':
    m = sys.argv[1]
    if m == 'roster': make_roster(sys.argv[2].upper(), sys.argv[3].upper(), sys.argv[4])
    elif m == 'run': run(sys.argv[2].upper(), sys.argv[3].upper(), sys.argv[4], sys.argv[5], int(sys.argv[6]) if len(sys.argv) > 6 else 0)
    elif m == 'synth': synth(sys.argv[2])
    elif m == 'merge': merge(sys.argv[2])
