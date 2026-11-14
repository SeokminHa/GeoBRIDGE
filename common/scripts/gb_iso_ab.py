# Compare forward and isochronic pairs on raw and DPT-residual layers.
import sys, os, json, time, hashlib, socket
import numpy as np, pandas as pd, anndata as ad
import gb_iso_lib as L
from gb_iso_pilot import load_freeze, dense, get_uni, strata_codes, log, OUT, TDIR, PHB, ROOTS, SG

ARMS = ('fwd_raw', 'fwd_dpt', 'iso_raw', 'iso_dpt')


def basis(t):
    return np.stack([np.ones_like(t), t, t ** 2, t ** 3, np.maximum(t - .5, 0) ** 3], 1)


def run(system, focal, smoke=False):
    FZ, fsha = load_freeze(); cfg = FZ['systems'][system]; sd = FZ['sources'][f'{system}/{focal}']; units = sd['units']; sources = sd['sources']
    od = OUT + f'ab/{system}_{focal}' + ('_smoke' if smoke else '') + '/'; os.makedirs(od, exist_ok=True)
    log('AB', system, focal, 'sources', sources, 'freeze', fsha[:12], socket.gethostname())
    uni, axis = get_uni(system, focal); ai = {g: i for i, g in enumerate(axis)}
    # full-data root DPT (Phase B: min-max over full-data cells) for the residual basis
    fz = np.load(cfg['pairs_npz'], allow_pickle=True); c2t = {}
    for r in ROOTS:
        fc = fz[f'{r}_cells'].astype(str); fpt = np.asarray(fz[f'{r}_pt'], float); lo, hi = fpt.min(), fpt.max(); c2t[r] = dict(zip(fc, (fpt - lo) / (hi - lo)))
    # targets = T-union (no cls) present in both halves
    tg = list(dict.fromkeys(g for u in units for g in pd.read_csv(TDIR + f'T_{u}.tsv', sep='\t', usecols=['gene']).gene.astype(str).str.upper()))
    if smoke: tg = tg[:200]
    H = {}
    for h in 'AB':
        fx = cfg['ab_fixtures'][h]; assert L.sha_file(fx['pairs_npz']) == fx['pairs_sha256']
        A = ad.read_h5ad(fx['h5ad']); Zp = np.load(fx['pairs_npz'], allow_pickle=True); obs = np.asarray(A.obs_names).astype(str)
        for r in ROOTS: assert (Zp[f'{r}_cells'].astype(str) == obs).all()
        names = np.char.upper(np.asarray(A.var_names).astype(str)); hv = {}
        for j, g in enumerate(names): hv.setdefault(g, j)
        H[h] = dict(A=A, Zp=Zp, obs=obs, hv=hv, P30=np.asarray(A.obsm['X_pca'], float)[:, :30], strata=strata_codes(A.obs, cfg['strata_columns_used']),
                    t={r: np.array([c2t[r][c] for c in obs]) for r in ROOTS}, pt={r: np.asarray(Zp[f'{r}_pt'], float) for r in ROOTS})
    srcs = [s for s in sources if s.upper() in H['A']['hv'] and s.upper() in H['B']['hv']]; dropped_src = [s for s in sources if s not in srcs]; assert srcs[0].upper() == focal.upper()
    tgt = [g for g in tg if g in H['A']['hv'] and g in H['B']['hv'] and g in ai]; dropped_tgt = len(tg) - len(tgt); tgt = np.array(tgt); Gt = len(tgt); ns = len(srcs)
    log('sources kept', ns, 'dropped', dropped_src, 'targets', Gt, 'dropped', dropped_tgt)
    for h in 'AB':
        d = H[h]; d['Xs'] = np.column_stack([dense(d['A'].layers['Ms'][:, [d['hv'][s.upper()]]], np.float32)[:, 0].astype(np.float64) for s in srcs])
        d['Mu'] = dense(d['A'].layers['Mu'][:, [d['hv'][g] for g in tgt]], np.float32); d['A'] = None
    RES = {}; meta = dict(freeze_sha256=fsha, system=system, focal=focal, sources=srcs, dropped_sources=dropped_src, n_targets=Gt, dropped_targets=dropped_tgt, feas={}, parity={})
    for te, tr in (('B', 'A'), ('A', 'B')):
        Hh, T = H[te], H[tr]
        for ri, r in enumerate(ROOTS):
            XT, XH = basis(T['t'][r]), basis(Hh['t'][r]); pinv = np.linalg.pinv(XT)
            FX = XH @ (pinv @ T['Xs']); RX = Hh['Xs'] - FX
            YT = T['Mu'].astype(np.float64); fit = XH @ (pinv @ YT); res = Hh['Mu'].astype(np.float64) - fit; del YT
            Ifw = np.asarray(Hh['Zp'][f'{r}_I'], np.int64); Jfw = np.asarray(Hh['Zp'][f'{r}_J'], np.int64)
            def put(arm, seed, d): RES.update({f'{arm}|{te}|{r}|{seed}|{k}': v for k, v in d.items()})
            put('fwd_raw', 0, L.pair_sums(Hh['Xs'], Hh['Mu'], Ifw, Jfw)); put('fwd_dpt', 0, L.pair_sums(RX, res, Ifw, Jfw))
            Q = np.linalg.qr(Hh['P30'][Jfw] - Hh['P30'][Ifw])[0]; k8 = L.deepk_sums(Hh['Xs'], Hh['Mu'], Ifw, Jfw, Q, 8); put('fwd_k8', 0, dict(N=k8['N'], D=k8['D'])); del Q
            for sdd in L.SEEDS:
                Ii, Ji = L.iso_pairs(Hh['pt'][r], Hh['strata'], L.WINDOW, L.CAP, sdd, ri)
                cells = int(len(np.unique(np.concatenate([Ii, Ji])))); meta['feas'][f'{te}|{r}|{sdd}'] = dict(n_pairs=int(len(Ii)), participating_cells=cells, ok_ge_100=bool(cells >= 100), pairs_sha256=L.sha_np(Ii, Ji))
                put('iso_raw', sdd, L.pair_sums(Hh['Xs'], Hh['Mu'], Ii, Ji)); put('iso_dpt', sdd, L.pair_sums(RX, res, Ii, Ji))
            meta['feas'][f'{te}|{r}|forward'] = dict(n_pairs=int(len(Ifw)), participating_cells=int(len(np.unique(np.concatenate([Ifw, Jfw])))))
            log('AB fold test', te, 'train', tr, r, 'done')
    # parity vs Phase-B parts (focal, forward arms): |N-N_phb|/D0_phb <= 1e-6
    for r in ROOTS:
        pth = PHB + f'{system}_{focal}_{r}.npz'
        if not os.path.exists(pth): meta['parity'][r] = 'no Phase-B part'; continue
        z = np.load(pth, allow_pickle=True); pax = np.char.upper(z['genes'].astype(str)); pidx = np.array([{g: i for i, g in enumerate(pax)}[g] for g in tgt])
        out = {}
        for te in 'AB':
            d0 = z[f'D0_{te}'][0][pidx]
            for nm, (a, b) in {'fwd_raw': ('N0', 'D0'), 'fwd_dpt': ('Nr', 'Dr'), 'fwd_k8': ('N8', 'D8')}.items():
                nn = z[f'{a}_{te}'][0][pidx]; dd = z[f'{b}_{te}'][0][pidx]; mN = RES[f'{nm}|{te}|{r}|0|N'][0]; mD = RES[f'{nm}|{te}|{r}|0|D'][0]
                fin = np.isfinite(nn) & (d0 > 0)
                out[f'{nm}|{te}'] = dict(n=int(fin.sum()), maxrel_N_over_D0=(float((np.abs(mN[fin] - nn[fin]) / d0[fin]).max()) if fin.any() else None), maxrel_D_over_D0=(float((np.abs(mD[fin] - dd[fin]) / d0[fin]).max()) if fin.any() else None))
        meta['parity'][r] = out
    np.savez_compressed(od + 'ab_sums.npz', sources=np.array(srcs), targets=tgt, **RES)
    json.dump(meta, open(od + 'ab_meta.json', 'w'), indent=1, default=str)
    man = dict(freeze_sha256=fsha, script_sha256=L.sha_file(os.path.abspath(__file__)), lib_sha256=L.sha_file(os.path.abspath(L.__file__)), host=socket.gethostname(), files={f: L.sha_file(od + f) for f in sorted(os.listdir(od)) if f != 'manifest.json'})
    json.dump(man, open(od + 'manifest.json', 'w'), indent=1)
    log('AB DONE'); print(json.dumps(meta['parity'], indent=1)); print(json.dumps(meta['feas'], indent=1))
