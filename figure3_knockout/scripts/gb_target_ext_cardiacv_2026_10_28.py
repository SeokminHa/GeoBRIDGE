# Target-axis extension for CARDIACV
import sys, json, os, time, hashlib
import numpy as np
R = __import__("os").environ["GEOBRIDGE_ROOT"]
sys.path[:0] = [f'{R}/scratch', f'{R}/pairspace']
import gb_canonical as GC, gb_target_extension as TE
import anndata as ad
sha = TE.sha
FIX = f'{R}/results/canonical_2026_09_18/CARDIACV_canonical.h5ad'
raw = dict(ad.read_h5ad(FIX, backed='r').uns['gb_contract'])['raw_source']
GC.RAW['CARDIACV'] = (raw, 'mouse')
mode = sys.argv[1]
if mode in ('chunk', 'merge'):
    import gb_real_uniform as RU
    from gb_substrate import load
    sources = sys.argv[2].split(','); CH = f'{TE.OUT}/CARDIACV_ext_chunks'; os.makedirs(CH, exist_ok=True); t0 = time.time()
    rec = json.load(open(f'{TE.OUT}/CARDIACV_ext.receipt.json')); assert sha(rec['out']) == rec['sha256'] and rec['script_sha256'] == sha(TE.__file__)
    E = np.load(rec['out']); eg = E['genes'].astype(str); gu = np.char.upper(eg); u_, c_ = np.unique(gu, return_counts=True); coll = set(u_[c_ > 1])
    keep = np.array([g not in coll for g in gu]); eg, gu = eg[keep], gu[keep]; G = len(eg)
    if mode == 'chunk':
        g0, g1 = int(sys.argv[3]), min(int(sys.argv[4]), G)
        S = load('CARDIACV', canonical=True); assert sha(S['fixture']) == rec['v1_fixture_sha256']
        cols = np.flatnonzero(keep)[g0:g1]; Ms, Mu = E['Ms'][:, cols], E['Mu'][:, cols]
        vg = np.asarray(S['genes']).astype(str); vp = {g.upper(): i for i, g in enumerate(vg)}; src = [vp[t.upper()] for t in sources]
        Xsrc = np.asarray(S['X'], np.float64)[:, src]; P30 = np.asarray(ad.read_h5ad(S['fixture'], backed='r').obsm['X_pca'], float)[:, :TE.NPC]
        out = {}
        for root, (I, J) in S['pairs'].items():
            dP = P30[J] - P30[I]; Qf, _ = np.linalg.qr(dP); dx = (Xsrc[J] - Xsrc[I]).astype(np.float32)
            for c, y in (('U', (Mu[J] - Mu[I]).astype(np.float32)), ('M', (Ms[J] - Ms[I]).astype(np.float32))):
                for k, num, den, z2, absS in TE.project_scores(dx, y, Qf): out[f'{root}|{c}|k{k}|absS'] = absS.astype(np.float32)
            print(f'  chunk {g0}-{g1} {root} {time.time()-t0:.0f}s', flush=True)
        np.savez_compressed(f'{CH}/chunk_{g0:05d}_{g1:05d}.npz', g0=g0, g1=g1, genes=eg[g0:g1], sources=np.asarray(sources), **out); print('chunk done', g0, g1)
    else:
        import glob as _g
        fs = sorted(_g.glob(f'{CH}/chunk_*.npz')); Z = [np.load(f, allow_pickle=True) for f in fs]
        cover = np.zeros(G, int)
        for z in Z: cover[int(z['g0']):int(z['g1'])] += 1
        if not (cover == 1).all(): raise SystemExit(f'REFUSING: chunk coverage not exactly once ({(cover == 0).sum()} missing, {(cover > 1).sum()} duplicated)')
        S = load('CARDIACV', canonical=True); vg = np.asarray(S['genes']).astype(str); vp = {g.upper(): i for i, g in enumerate(vg)}; src = [vp[t.upper()] for t in sources]
        out = dict(genes=eg, sources=np.asarray(sources), ks=np.arange(TE.KMAX + 1), is_v1=np.isin(eg, vg))
        for root in S['pairs']:
            for c in ('U', 'M'):
                for k in range(TE.KMAX + 1):
                    key = f'{root}|{c}|k{k}|absS'; arr = np.full((len(sources), G), np.nan, np.float32)
                    for z in Z: arr[:, int(z['g0']):int(z['g1'])] = z[key]
                    out[key] = arr
        for s_ in range(len(sources)):
            own = eg == vg[src[s_]]
            for key in [k_ for k_ in out if k_.endswith('|absS')]: out[key][s_, own] = np.nan
        rows, rg = RU.gb_raw_rows('CARDIACV', srcs=sources); rg = np.char.upper(np.asarray(rg).astype(str)); ep_ = {g: i for i, g in enumerate(gu)}
        zi = np.array([ep_.get(g, -1) for g in rg]); assert (zi >= 0).all(), 'cache genes absent from extension axis'
        worst = {'U': 0.0, 'M': 0.0}; roster = {}
        for s_, t in enumerate(sources):
            if not isinstance(rows.get(t), dict): raise SystemExit(f'REFUSING: production cache rows for {t} missing or not a dict')
            want = {(r, ch) for r in S['pairs'] for ch in ('U', 'X')}
            if set(rows[t].keys()) != want: raise SystemExit(f'REFUSING: cache roster for {t} {sorted(rows[t].keys())} != {sorted(want)}')
            selfm = rg == t.upper(); n = 0
            for (root, ch), arr in rows[t].items():
                a_ = np.asarray(arr, float).copy()
                if a_.shape[1] != TE.KMAX + 1: raise SystemExit(f'REFUSING: {t} {root} {ch} has {a_.shape[1]} depths, need 31')
                c2 = 'M' if ch == 'X' else ch; b_ = np.stack([out[f'{root}|{c2}|k{k}|absS'][s_, zi] for k in range(31)], 1).astype(float)
                a_[selfm] = np.nan; b_[selfm] = np.nan
                if not np.array_equal(np.isfinite(a_), np.isfinite(b_)): raise SystemExit(f'REFUSING parity finiteness {t} {root} {ch}')
                f = np.isfinite(a_); worst[c2] = max(worst[c2], float(np.abs(a_[f] - b_[f]).max())); n += 1
            roster[t] = n
            if n != 4: raise SystemExit(f'REFUSING: roster {t} = {n} arrays, need 4')
        print('STRICT PARITY vs production caches: roster', roster, 'worst', worst, flush=True)
        if max(worst.values()) > TE.TOL_SCORE: raise SystemExit(f'REFUSING: parity failed {worst}')
        o = f'{TE.OUT}/CARDIACV_ext_scores.npz'; np.savez_compressed(o, **out)
        json.dump(dict(system='CARDIACV', out=o, sha256=sha(o), sources=sources, n_targets=G, n_new=int((~out['is_v1']).sum()), n_chunks=len(fs),
                       parity_vs_production_caches=worst, parity_roster=roster, stored_statistics='Only absS for U and M is stored. Numerator, denominator, z2 and V are unavailable, '
                       'so PR support cannot be calculated.', wrapper_sha256=sha(os.path.abspath(__file__)),
                       audited_core_sha256=sha(TE.__file__), truth_read=False, wall_s=round(time.time() - t0, 1)), open(o.replace('.npz', '.receipt.json'), 'w'), indent=1)
        print('saved', o)
    sys.exit(0)
if mode == 'build':
    TE.build('CARDIACV')
    rp = f'{TE.OUT}/CARDIACV_ext.receipt.json'; rec = json.load(open(rp)); rec['wrapper'] = os.path.abspath(__file__); rec['wrapper_sha256'] = sha(os.path.abspath(__file__))
    json.dump(rec, open(rp, 'w'), indent=1); print('built', rec['n_v1'], '->', rec['n_ext'], rec['parity_rel_maxabs'])
else:
    import gb_real_uniform as RU
    from gb_substrate import load
    sources = sys.argv[2].split(','); t0 = time.time()
    rec = json.load(open(f'{TE.OUT}/CARDIACV_ext.receipt.json')); assert sha(rec['out']) == rec['sha256'] and rec['script_sha256'] == sha(TE.__file__)
    S = load('CARDIACV', canonical=True); assert sha(S['fixture']) == rec['v1_fixture_sha256']
    E = np.load(rec['out']); eg = E['genes'].astype(str); gu = np.char.upper(eg); u_, c_ = np.unique(gu, return_counts=True); coll = set(u_[c_ > 1])
    keep = np.array([g not in coll for g in gu]); eg, gu = eg[keep], gu[keep]; Ms, Mu = E['Ms'][:, keep], E['Mu'][:, keep]
    vg = np.asarray(S['genes']).astype(str); vp = {g.upper(): i for i, g in enumerate(vg)}; src = [vp[t.upper()] for t in sources]
    Xsrc = np.asarray(S['X'], np.float64)[:, src]; P30 = np.asarray(ad.read_h5ad(S['fixture'], backed='r').obsm['X_pca'], float)[:, :TE.NPC]
    G, ns = len(eg), len(sources); out = dict(genes=eg, sources=np.asarray(sources), ks=np.arange(TE.KMAX + 1), is_v1=np.isin(eg, vg))
    for root in S['pairs']:
        for c in ('U', 'M'):
            for k in range(TE.KMAX + 1): out[f'{root}|{c}|k{k}|absS'] = np.full((ns, G), np.nan, np.float32)
    for root, (I, J) in S['pairs'].items():
        dP = P30[J] - P30[I]; Qf, _ = np.linalg.qr(dP); dx = (Xsrc[J] - Xsrc[I]).astype(np.float32)
        for g0 in range(0, G, TE.CHUNK):
            sl = slice(g0, min(g0 + TE.CHUNK, G))
            for c, y in (('U', (Mu[J, sl] - Mu[I, sl]).astype(np.float32)), ('M', (Ms[J, sl] - Ms[I, sl]).astype(np.float32))):
                for k, num, den, z2, absS in TE.project_scores(dx, y, Qf): out[f'{root}|{c}|k{k}|absS'][:, sl] = absS
        print(f'  {root}: {len(I)} pairs {time.time()-t0:.0f}s', flush=True)
    for s_ in range(ns):
        own = eg == vg[src[s_]]
        for key in [k_ for k_ in out if k_.endswith('|absS')]: out[key][s_, own] = np.nan
    # PARITY vs current production caches (U, X=M channel) on V1 genes
    rows, rg = RU.gb_raw_rows('CARDIACV', srcs=sources); rg = np.char.upper(np.asarray(rg).astype(str)); ep_ = {g: i for i, g in enumerate(gu)}
    zi = np.array([ep_.get(g, -1) for g in rg]); assert (zi >= 0).all(), 'cache genes absent from extension axis'
    worst = {'U': 0.0, 'M': 0.0}; n = 0
    for s_, t in enumerate(sources):
        selfm = rg == t.upper()
        for (root, ch), arr in rows[t].items() if isinstance(rows[t], dict) else []:
            c2 = 'M' if ch == 'X' else ch
            a_ = np.asarray(arr, float).copy(); b_ = np.stack([out[f'{root}|{c2}|k{k}|absS'][s_, zi] for k in range(a_.shape[1])], 1).astype(float)
            a_[selfm] = np.nan; b_[selfm] = np.nan
            if not np.array_equal(np.isfinite(a_), np.isfinite(b_)): raise SystemExit(f'REFUSING parity finiteness {t} {root} {ch}')
            f = np.isfinite(a_); worst[c2] = max(worst[c2], float(np.abs(a_[f] - b_[f]).max())); n += 1
    print('PARITY vs production caches:', n, 'arrays, worst', worst, flush=True)
    if n == 0 or max(worst.values()) > TE.TOL_SCORE: raise SystemExit(f'REFUSING: parity failed {worst} n={n}')
    o = f'{TE.OUT}/CARDIACV_ext_scores.npz'; np.savez_compressed(o, **out)
    json.dump(dict(system='CARDIACV', out=o, sha256=sha(o), sources=sources, n_targets=G, n_new=int((~out['is_v1']).sum()), parity_vs_production_caches=worst,
                   parity_arrays=n, wrapper_sha256=sha(os.path.abspath(__file__)), audited_core_sha256=sha(TE.__file__), truth_read=False, wall_s=round(time.time() - t0, 1)),
              open(o.replace('.npz', '.receipt.json'), 'w'), indent=1)
    print('saved', o)
