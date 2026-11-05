"""E14 real-KO sign-estimator grid (EVALUATION ONLY; exploratory). (same candidate list as E13).
All candidates come from ONE script on stored scores (no new scoring except I-cen, which re-runs gb_iso_lib.pair_sums centred on the pilot iso pairs).
Sources: features_<unit>.tsv (iso_pilot: d1,d4,iso_C,cellMsMu,deepk8), iso_full scores (fwd/iso42-44 N,D for k=0..8 per root), pilot iso pairs + fixture (I-cen).
Truth (cls) joined at evaluation as gb_iso_eval.load_unit; BA = zero-threshold, ambiguity(0)/NaN = miss, over KO responders (DN=A_act,C_act; UP=A_rep,C_rep).
usage: script OUT [--nocen]"""
import sys, os, json, numpy as np, pandas as pd
sys.path[:0] = [os.environ['GEOBRIDGE_ROOT'] + '/scratch', os.environ['GEOBRIDGE_ROOT'] + '/pairspace', os.environ['GEOBRIDGE_ROOT']]
import gb_iso_lib as L
R0 = os.environ['GEOBRIDGE_ROOT'] + '/'; SG = R0 + 'results/signed/'; PIL = SG + 'iso_pilot_2026_10_15/'; FULL = SG + 'iso_full_2026_10_15/scores/'
TDIR = SG + 'grand_v3d_dumpT_2026_10_15/parts/'
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True); NOCEN = '--nocen' in sys.argv
ROOTS = L.ROOTS; SEEDS = L.SEEDS; EPS = L.EPS
UP, DN, NUL = ['A_rep', 'C_rep'], ['A_act', 'C_act'], ['B', 'D']; CLS_ALL = DN + NUL + UP
UNITS = {'TBX5': ('CARDIACV', 'TBX5', 'CARDIACV_TBX5'), 'NKX2-5': ('CARDIACV', 'NKX2-5', 'CARDIACV_NKX2-5'), 'GATA1': ('GATA1', 'GATA1', 'GATA1_GATA1'),
         'KLF1': ('GATA1', 'KLF1', 'KLF1_KLF1'), 'KLF1_B': ('GATA1', 'KLF1', 'KLF1_B_KLF1'), 'RFX6': ('PANC', 'RFX6', 'PANC_RFX6')}
TAG = {'TBX5': 'CARDIACV_TBX5', 'NKX2-5': 'CARDIACV_NKX2-5', 'GATA1': 'GATA1_GATA1', 'KLF1': 'GATA1_KLF1', 'KLF1_B': 'GATA1_KLF1', 'RFX6': 'PANC_RFX6'}   # pilot score dir
ORDER = ['F-k0', 'F-mean0..8', 'F-mean0..8(2/3)', 'F-k8', 'F-vote', 'F-pool', 'F-cen(d4)', 'I-k0', 'I-cen', 'I-mean0..8', 'Pearson Ms-Mu', 'AGREE', 'AGREE(d1&iso)']
PAR = {}                                                                     # parity diagnostics


def log(*a): print(*a, flush=True)
def sgn(x): x = np.asarray(x, float); return np.sign(np.where(np.isfinite(x), x, 0.0))
def rootmean(a, b): return np.where(np.isfinite(a) & np.isfinite(b), (a + b) / 2, np.nan)


def metrics(s, dn, up):
    rd = float((s[dn] > 0).mean()); ru = float((s[up] < 0).mean())
    return dict(BA=0.5 * (rd + ru), KO_up_recall=ru, KO_down_recall=rd, coverage=float((s[dn | up] != 0).mean()), n_down=int(dn.sum()), n_up=int(up.sum()))


def load_full(system, st, root, tag_foc):                                    # stored iso_full sums; all-source file
    fn = f'{FULL}{system}/{st}__{root}__km8.npz'; z = np.load(fn, allow_pickle=True)
    src = [str(s).upper() for s in z['sources']]; j = src.index(tag_foc.upper())
    N = z['N'][:, j, :].astype(float); D = z['D'][:, j, :].astype(float)       # (9, G)
    out = dict(N=N, D=D, tg=np.array([str(t).upper() for t in z['targets']]))
    if 'Nc' in z.files: out.update(Nc=z['Nc'][j].astype(float), Dc=z['Dc'][j].astype(float))
    return out


def S(N, D): ok = D > EPS; return np.where(ok, N / np.where(ok, D, 1.0), np.nan)


def icen_all(system, foc, genes_by_unit):                                    # I-cen: centred k0 concordance on pilot iso pairs, mean over 3 seeds, root mean
    import anndata as ad, scipy.sparse as sp
    fz = json.load(open(PIL + 'iso_freeze.json')); cfg = fz['systems'][system]
    F = ad.read_h5ad(cfg['fixture']); vn = np.char.upper(np.asarray(F.var_names).astype(str)); fi = {}
    for j, g in enumerate(vn): fi.setdefault(g, j)
    dense = lambda M: np.asarray(M.toarray() if hasattr(M, 'toarray') else M, np.float32)
    tg = sorted(set(g for gs in genes_by_unit for g in gs)); tg = [g for g in tg if g in fi]; col = np.array([fi[g] for g in tg])
    Xs = dense(F.layers['Ms'][:, [fi[foc.upper()]]]).astype(np.float64); Yt = dense(F.layers['Mu'][:, col])
    res = {}
    for r in ROOTS:
        cs = []
        for sd in SEEDS:
            z = np.load(PIL + f'pairs/{system}/{r}_iso{sd}.npz'); S_ = L.pair_sums(Xs, Yt, z['I'].astype(np.int64), z['J'].astype(np.int64), centred=True)
            cs.append(L.ratio_C(S_['N'], S_['D'])[0])
        cs = np.stack(cs); res[r] = np.where(np.isfinite(cs).all(0), cs.mean(0), np.nan)
    v = rootmean(res[ROOTS[0]], res[ROOTS[1]]); return dict(zip(tg, v))


rows = []; PER = {}
ICEN = {}
for u, (system, foc, unit) in UNITS.items():
    f = pd.read_csv(PIL + f'scores/{TAG[u]}/features_{unit}.tsv', sep='\t')
    t = pd.read_csv(TDIR + f'T_{unit}.tsv', sep='\t', usecols=['gene', 'cls']); t['gene'] = t.gene.astype(str).str.upper()
    assert (t.gene.values == f.gene.values.astype(str)).all(); f['cls'] = t.cls.values
    f = f[f.cls.isin(CLS_ALL)].reset_index(drop=True); c = f.cls.to_numpy(str); dn, up = np.isin(c, DN), np.isin(c, UP)
    genes = f.gene.astype(str).str.upper().to_numpy(); inuni = f.in_uni.astype(bool).to_numpy()
    # ---- stored sums (forward + iso42..44), per root
    FW = {r: load_full(system, 'fwd', r, foc) for r in ROOTS}; ISO = {(r, sd): load_full(system, f'iso{sd}', r, foc) for r in ROOTS for sd in SEEDS}
    tg = FW[ROOTS[0]]['tg']; gi = pd.Series(np.arange(len(tg)), index=tg); assert gi.index.is_unique; ix = gi.reindex(genes).to_numpy()
    has = np.isfinite(ix); ixi = np.where(has, ix, 0).astype(int)
    pick = lambda A: np.where(has, np.asarray(A)[..., ixi], np.nan)           # (..., G) -> (..., n)
    mask = inuni & has
    FS = np.stack([pick(S(FW[r]['N'], FW[r]['D'])) for r in ROOTS])             # (2,9,n)
    IS = np.stack([[pick(S(ISO[(r, sd)]['N'], ISO[(r, sd)]['D'])) for sd in SEEDS] for r in ROOTS])   # (2,3,9,n)
    fN = pick(np.stack([FW[r]['N'] for r in ROOTS])); fD = pick(np.stack([FW[r]['D'] for r in ROOTS]))  # (2,9,n)
    fc = np.stack([pick(S(FW[r]['Nc'], FW[r]['Dc'])) for r in ROOTS]) if 'Nc' in FW[ROOTS[0]] else None
    fin = lambda x, axes: np.where(np.isfinite(x).all(axes), x.mean(axes), np.nan)
    # ---- parity vs pilot-stored columns (continuous values; sign agreement)
    def par(name, a, b):
        m = np.isfinite(a) & np.isfinite(b) & mask; PAR[f'{u}|{name}'] = dict(n=int(m.sum()), sign_agree=float((np.sign(a[m]) == np.sign(b[m])).mean()) if m.any() else None,
                                                                              max_abs=float(np.abs(a[m] - b[m]).max()) if m.any() else None)
    cand = {}
    cand['F-k0'] = f.d1.to_numpy(float)                                          # stored stage-1 cache (the pilot's d1)
    par('F-k0 fresh(iso_full) vs d1', rootmean(FS[0, 0], FS[1, 0]), cand['F-k0'])
    cand['F-mean0..8'] = np.where(mask, fin(FS, (0, 1)), np.nan)
    m2 = np.isfinite(FS).sum((0, 1)) >= int(np.ceil(2 * 18 / 3)); cand['F-mean0..8(2/3)'] = np.where(mask & m2, np.nanmean(FS, (0, 1)), np.nan)
    cand['F-k8'] = f.deepk8_cached_full.to_numpy(float); par('F-k8 fresh vs cached', rootmean(FS[0, 8], FS[1, 8]), cand['F-k8'])
    cand['F-vote'] = np.where(mask & np.isfinite(FS).all((0, 1)), np.sign(FS).sum((0, 1)), np.nan)
    with np.errstate(invalid='ignore', divide='ignore'):
        sD = fD.sum((0, 1)); cand['F-pool'] = np.where(mask & (sD > EPS) & np.isfinite(fN).all((0, 1)) & np.isfinite(fD).all((0, 1)), fN.sum((0, 1)) / sD, np.nan)
    cand['F-cen(d4)'] = f.d4.to_numpy(float)
    if fc is not None: par('F-cen fresh vs d4', rootmean(fc[0], fc[1]), cand['F-cen(d4)'])
    cand['I-k0'] = f.iso_C.to_numpy(float)
    ik0 = np.where(np.isfinite(IS[:, :, 0]).all(1), IS[:, :, 0].mean(1), np.nan); par('I-k0 fresh(iso_full pairs) vs iso_C', rootmean(ik0[0], ik0[1]), cand['I-k0'])
    cand['I-mean0..8'] = np.where(mask, fin(IS, (0, 1, 2)), np.nan)
    cand['Pearson Ms-Mu'] = f.cellMsMu.to_numpy(float)
    if not NOCEN:
        key = system + '|' + foc
        if key not in ICEN:
            gbu = [pd.read_csv(TDIR + f'T_{x[2]}.tsv', sep='\t', usecols=['gene']).gene.astype(str).str.upper().tolist() for x in UNITS.values() if x[0] == system and x[1] == foc]
            ICEN[key] = icen_all(system, foc, gbu)
        cand['I-cen'] = np.where(mask, pd.Series(genes).map(ICEN[key]).to_numpy(float), np.nan)
    else: cand['I-cen'] = np.full(len(f), np.nan)
    sg = {k: sgn(v) for k, v in cand.items()}
    sg['AGREE'] = np.where((sg['F-mean0..8'] == sg['I-k0']) & (sg['I-k0'] != 0), sg['I-k0'], 0.0)
    sg['AGREE(d1&iso)'] = np.where((sg['F-k0'] == sg['I-k0']) & (sg['I-k0'] != 0), sg['I-k0'], 0.0)
    for k in ORDER:
        m = metrics(sg[k], dn, up); m.update(unit=u, estimator=k); rows.append(m)
    # for the d4/ISO/cell Ms-Mu parity: also pilot rows rewire etc not needed
    log(u, {k: round(r['BA'], 3) for r in rows if r['unit'] == u for k in [r['estimator']]})
df = pd.DataFrame(rows)
# equal-TF-weight means: 5 TF units (KLF1_B not a vote, §444.122) and all 6 rows
eq = []
for k in ORDER:
    d = df[df.estimator == k].set_index('unit'); five = d.loc[['TBX5', 'NKX2-5', 'GATA1', 'KLF1', 'RFX6']]; six = d
    eq.append(dict(estimator=k, **{f'mean5_{c}': float(five[c].mean()) for c in ('BA', 'KO_up_recall', 'KO_down_recall', 'coverage')}, **{f'mean6_{c}': float(six[c].mean()) for c in ('BA', 'KO_up_recall', 'KO_down_recall', 'coverage')}))
eq = pd.DataFrame(eq)
df['set'] = 'exploratory'; eq['set'] = 'exploratory'
df.to_csv(OUT + '/E14_sign_real_per_unit.tsv', sep='\t', index=False); eq.to_csv(OUT + '/E14_sign_real_equal_TF_mean.tsv', sep='\t', index=False)
json.dump(PAR, open(OUT + '/parity_fresh_vs_stored.json', 'w'), indent=1)
log(eq.round(3).to_string())
