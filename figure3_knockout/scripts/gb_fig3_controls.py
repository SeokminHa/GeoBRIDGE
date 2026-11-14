# Figure 3 control-arm evaluation.
import sys, os, json, hashlib, glob
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
a = sys.argv; OUT = a[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
B = a[a.index('--B') + 1] if '--B' in a else '1000'; NS = int(a[a.index('--nseeds') + 1]) if '--nseeds' in a else 20
ALLEP = ['PANC_RFX6', 'BM_GATA1_acute', 'HEART_TBX5', 'HEART_NKX2-5']
EPS = a[a.index('--endpoints') + 1].split(',') if '--endpoints' in a else ALLEP
CUT = 'for s in SYS4:\n    cs = '
def load(path, argv):
    src = open(path).read().replace('\r', ''); src = src[:src.index(CUT)]
    ns = {'__file__': path, '__name__': 'prefix'}; sys.argv = argv; exec(compile(src, path, 'exec'), ns); return ns
nonbm = [e for e in EPS if e != 'BM_GATA1_acute']
nsA = load(R + 'scratch/gb_fig3_endpoints.py', ['x', OUT + '_scrA', '--B', B, '--endpoints', ','.join(nonbm or ['PANC_RFX6'])])
nsB = load(R + 'results/signed/bm_acute_design_audit_2026_10_26/noF2/fig3_ugb_v2_noF2_velalltargets.py', ['x', OUT + '_scrB', '--B', B, '--rule', 'noF2', '--endpoints', 'BM_GATA1_acute']) if 'BM_GATA1_acute' in EPS else None
import numpy as np, pandas as pd
sys.path[:0] = [R + 'scratch', R + 'pairspace']
import gb_primary as GP
CT, PT = nsA['CT'], nsA['PT']
EXT = ['|Pearson|', 'GENIE3', 'GRNBoost2', 'Velorama', 'RegDiffusion', 'DeepSEM', 'PIDC', 'FastTENET']; METH = ['GeoBRIDGE'] + EXT
SYSF = {'PANC_RFX6': ('PANC', 'RFX6'), 'BM_GATA1_acute': ('BONEMARROW', 'GATA1'), 'HEART_TBX5': ('CARDIACV', 'TBX5'), 'HEART_NKX2-5': ('CARDIACV', 'NKX2-5')}
ARM = R + 'results/figures/manuscript_2026_10_19/fig3_pt_comparator_v2/arms/'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
REC = dict(inputs={}, parity=[], B=int(B), nseeds=NS, endpoints=EPS)
ARMIN = {}
def chk(f, fx):
    rc = json.load(open(f.replace('arms_', 'receipt_').replace('.npz', '.json'))); assert rc['status'] == 'PASS' and rc['npz_sha256'] == sha(f), ('arm receipt', f)
    for p_ in rc.get('parity', []): assert p_['max_abs'] <= 1e-6 and p_['finite_mask_mismatch'] == 0, ('arm parity', f, p_)
    b = {k: v[1] for k, v in rc['inputs'].items()}; ARMIN.setdefault(fx, b); assert ARMIN[fx] == b, ('comparator shard inputs differ', f, b, ARMIN[fx])
    REC['inputs'][f] = sha(f)
def mapto(names, vals, g):
    names = np.char.upper(np.asarray(names).astype(str)); s = pd.Series(np.arange(len(names)), index=names); assert not s.index.duplicated().any(), 'dup genes in cache'
    ii = pd.Series(g).map(s).to_numpy(); has = np.isfinite(ii); out = np.full(len(g), np.nan); out[has] = np.asarray(vals, float)[ii[has].astype(int)]; return out
def catbank(fx, tf, g):
    hits = []
    for f in sorted(glob.glob(f'{R}results/real_uniform/catbank/{fx}_scores_*-*.npz')):
        z = np.load(f, allow_pickle=True); srcs = [s.upper() for s in z['sources'].astype(str)]
        n_ = srcs.count(tf.upper()); assert n_ <= 1, ('duplicate focal rows in catbank shard', f, tf)
        if n_: hits.append((f, z, srcs.index(tf.upper())))
    assert len(hits) == 1, ('focal must be in exactly one catbank shard', fx, tf, [h[0] for h in hits])
    f, z, i = hits[0]; rc = json.load(open(f.replace('.npz', '.receipt.json'))); bank = json.load(open(f'{R}results/real_uniform/catbank/{fx}_bank.json'))
    assert rc.get('production_universe') is True and rc.get('parity_verified') is True, ('catbank receipt flags', f)
    assert rc['output_sha256'] == sha(f), ('catbank artifact != receipt output_sha256', f)
    assert rc['binding']['config']['bank_sha256'] == bank['sha256_members'], ('catbank shard bank sha', f)
    for k in ('fixture_sha256', 'pairs_sha256', 'axis_sha256'): assert rc['binding']['inputs'].get(k) == bank['inputs'].get(k), ('catbank shard inputs != bank', k, f)
    cb_in = rc['binding']['inputs']; REC.setdefault('catbank_bindings', {})[f'{fx}:{tf}'] = {k: cb_in.get(k) for k in ('fixture_sha256', 'pairs_sha256', 'axis_sha256')}
    REC['inputs'][f] = sha(f); return mapto(z['genes'], z['P|PU'][i], g), mapto(z['genes'], z['P|PM'][i], g), cb_in
def signed_depths(fx, tf, g):
    z = np.load(f'{R}results/signed/{fx}.npz', allow_pickle=True); REC['inputs'][f'{R}results/signed/{fx}.npz'] = sha(f'{R}results/signed/{fx}.npz')
    srcs = [s.upper() for s in z['sources'].astype(str)]; i = srcs.index(tf.upper())
    genes = z['genes'].astype(str); A = {r: np.abs(np.asarray(z[f'{r}|U'][i], float)) for r in GP.ROOTS}            # (genes, 31)
    u_ = z['uni'][i] if z['uni'].ndim == 1 and z['uni'].dtype == object else np.asarray(z['uni'])[i]; u_ = np.asarray(u_)        # per-source observational universe
    if len(u_) == len(genes) and set(np.unique(u_.astype(object)).tolist()) <= {True, False}: uni = u_.astype(bool)   # boolean mask (may be stored as object)
    else: uni = np.zeros(len(genes), bool); uni[u_.astype(int)] = True
    out = {}
    for k in ['P'] + list(range(9)):
        ks = range(9) if k == 'P' else [k]; zc = GP.zcons({r: A[r][uni] for r in GP.ROOTS}, ks)
        full = np.full(len(genes), np.nan); full[uni] = zc; out[k] = mapto(genes, full, g)
    return out
D = {}; AV = {}
for lab in EPS:
    X = (nsB if lab == 'BM_GATA1_acute' else nsA)['D'][lab]; V = dict(X['V']); g = np.asarray(X['g']).astype(str); fx, tf = SYSF[lab]
    U = np.asarray(X['U'], bool) & np.all([np.isfinite(V[m]) for m in METH], 0)
    EPT = {x[0]: x for x in nsA['EP9']}; zv = np.load(f"{nsA['VEC']}ep_{EPT[lab][5]}.vec.npz", allow_pickle=True); U &= np.isfinite(zv['cog::Velorama'].astype(float))   # UOLD
    core = f'{ARM}arms_{fx}_core.npz'; chk(core, fx); zc = np.load(core, allow_pickle=True)
    arms = {k: mapto(zc['genes'], zc[f'{tf}|{k}'], g) for k in ('D', 'A', 'SPL')}; RV = {}
    for f_ in sorted(glob.glob(f'{ARM}arms_{fx}_rnd*.npz')):
        chk(f_, fx); zr = np.load(f_, allow_pickle=True); assert (zr['genes'] == zc['genes']).all()
        for q, v in zip(zr[f'{tf}|seeds'], zr[f'{tf}|RND']):
            if int(q) < NS: assert int(q) not in RV; RV[int(q)] = mapto(zr['genes'], v, g)
    assert sorted(RV) == list(range(NS)), ('RND seeds', lab, sorted(RV))
    PU, PM, cb_in = catbank(fx, tf, g); SD = signed_depths(fx, tf, g); GB = np.asarray(V['GeoBRIDGE'], float)
    # same substrate: comparator cache and catbank built on identical fixture and pairs files
    assert ARMIN[fx]['fixture'] == cb_in.get('fixture_sha256') and ARMIN[fx]['pairs'] == cb_in.get('pairs_sha256'), ('comparator vs catbank fixture/pairs differ', fx, ARMIN[fx], cb_in)
    assert ARMIN[fx]['signed_cache'] == sha(f'{R}results/signed/{fx}.npz'), ('signed cache != comparator-cache binding', fx)
    for nm, v, tol in (('armD', arms['D'], 1e-6), ('catbank_PU', PU, 1e-6), ('signed_zcons08', SD['P'], 1e-6)):
        d_ = np.abs(v[U] - GB[U]); REC['parity'].append(dict(endpoint=lab, check=f'{nm} vs GeoBRIDGE on UOLD', max_abs=float(np.nanmax(d_)), n_nonfinite=int((~np.isfinite(v[U])).sum())))
        assert np.isfinite(v[U]).all() and np.nanmax(d_) <= tol, REC['parity'][-1]
    d0 = np.abs(arms['A'][U] - SD[0][U]); REC['parity'].append(dict(endpoint=lab, check='K0 arm vs signed zcons([0])', max_abs=float(np.nanmax(d0)), n_nonfinite=int((~np.isfinite(arms['A'][U])).sum()),
                                                                   finite_mask_mismatch=int((np.isfinite(arms['A'][U]) != np.isfinite(SD[0][U])).sum())))
    assert REC['parity'][-1]['finite_mask_mismatch'] == 0 and np.nanmax(d0) <= 1e-6, REC['parity'][-1]
    A_ = dict(D=GB, K0=arms['A'], SPL=arms['SPL'], PS=PM, **{f'RND{q}': RV[q] for q in range(NS)})
    Uc = U & np.logical_and.reduce([np.isfinite(v) for v in A_.values()])
    AV[lab] = dict(main=A_, depth={f'K{k}': SD[k] for k in range(1, 9)})
    D[lab] = dict(X=X, U=U, Uc=Uc, y0=np.asarray(X['y'], bool), e=np.asarray(X['e'], float))
    REC.setdefault('support', []).append(dict(endpoint=lab, n_UOLD=int(U.sum()), n_common=int(Uc.sum()), lost={k: int((U & ~np.isfinite(v)).sum()) for k, v in A_.items() if not k.startswith('RND')}, lost_any_RND=int((U & ~np.logical_and.reduce([np.isfinite(A_[f'RND{q}']) for q in range(NS)])).sum())))
    print(lab, 'parity OK; UOLD', int(U.sum()), 'common', int(Uc.sum()), flush=True)
if 'HEART_TBX5' in D and 'HEART_NKX2-5' in D:
    assert np.array_equal(D['HEART_TBX5']['X']['C'], D['HEART_NKX2-5']['X']['C']) and np.array_equal(D['HEART_TBX5']['X']['gi'].shape, D['HEART_TBX5']['X']['gi'].shape), 'cardiac draws differ'
REC['draw_fingerprints'] = {lab: hashlib.sha256(np.ascontiguousarray(D[lab]['X']['C']).tobytes()).hexdigest()[:16] for lab in EPS}
REC['deps'] = {p: sha(R + p) for p in ('scratch/gb_fig3_endpoints.py', 'results/signed/bm_acute_design_audit_2026_10_26/noF2/fig3_ugb_v2_noF2_velalltargets.py',
                                       'scratch/gb_canon_table.py', 'scratch/gb_ptable.py', 'scratch/gb_primary.py')}
from sklearn.metrics import roc_auc_score as _AUC, average_precision_score as _APS
_l = EPS[0]; _X = D[_l]['X']; _y0 = D[_l]['y0']; _rr = D[_l]['Uc'] & ((_y0 & (D[_l]['e'] > 0)) | ~_y0); _yl = _y0 & (D[_l]['e'] > 0); _v = AV[_l]['main']['D']
_pt, _bs = CT.eval_rows(_v, _yl, _rr, _X['gi'], _X['C']); _idx = np.flatnonzero(_rr)
for _b in (0, 1, 2):
    _w = _X['C'][_b][np.asarray(_X['gi'])[_idx]].astype(int); _rep = np.repeat(_idx, _w)
    if _yl[_rep].any() and (~_yl[_rep]).any():
        _a, _p = _AUC(_yl[_rep], _v[_rep]), _APS(_yl[_rep], _v[_rep]); REC.setdefault('replay', []).append(dict(endpoint=_l, draw=_b, auroc_eval=float(_bs['AUROC'][_b]), auroc_dup=float(_a), ap_eval=float(_bs['AUPRC'][_b]), ap_dup=float(_p)))
        assert abs(_a - _bs['AUROC'][_b]) < 1e-9 and abs(_p - _bs['AUPRC'][_b]) < 1e-9, REC['replay'][-1]
MT = ('AUROC', 'APLIFT', 'AUPRC'); DRAW = {}; COV = []
def labels(lab, d):
    y0, e = D[lab]['y0'], D[lab]['e']; yl = y0.copy() if d == 'ALL' else (y0 & np.isfinite(e) & ((e > 0) if d == 'KOup' else (e < 0))); return yl, (yl | ~y0) if d != 'ALL' else np.ones_like(y0)
for lab in EPS:
    X = D[lab]['X']
    for d in ('ALL', 'KOup', 'KOdown'):
        yl, keep = labels(lab, d); rr = D[lab]['Uc'] & keep; npos, nneg = int((yl & rr).sum()), int((~yl & rr).sum())
        COV.append(dict(endpoint=lab, direction=d, n_rows=int(rr.sum()), n_pos=npos, n_neg=nneg, prevalence=npos / max(rr.sum(), 1)))
        if npos < 5 or nneg < 5: continue
        for arm, v in AV[lab]['main'].items():
            pt, bs = CT.eval_rows(v, yl, rr, X['gi'], X['C'])
            for k in MT: DRAW[(lab, d, k, arm)] = np.r_[pt[k], np.asarray(bs[k], float)]
        for arm, v in AV[lab]['depth'].items():
            rd = rr & np.isfinite(v)
            if (yl & rd).sum() >= 5 and (~yl & rd).sum() >= 5:
                pt, bs = CT.eval_rows(v, yl, rd, X['gi'], X['C'])
                for k in MT: DRAW[(lab, d, k, arm)] = np.r_[pt[k], np.asarray(bs[k], float)]
        for k in MT: DRAW[(lab, d, k, 'RND')] = np.median(np.vstack([DRAW[(lab, d, k, f'RND{q}')] for q in range(NS)]), 0)   # seed median per draw (NaN propagates)
    print(lab, 'evaluated', flush=True)
def holm(p):
    p = np.asarray(p, float); m = len(p); q = np.where(np.isfinite(p), p, 1.0); o = np.argsort(q, kind='stable'); out = np.full(m, np.nan); run = 0.0
    for i, k in enumerate(o): run = max(run, (m - i) * q[k]); out[k] = min(1.0, run)
    out[~np.isfinite(p)] = np.nan; return out
def stat(v):
    nb = len(v) - 1; f = v[1:][np.isfinite(v[1:])]; ok = len(f) >= .95 * nb
    return dict(value=v[0], lo=np.percentile(f, 2.5) if ok else np.nan, hi=np.percentile(f, 97.5) if ok else np.nan, n_valid=len(f), n_draws=nb, p_boot=PT.boot_p(f) if ok else np.nan)
SYS = {'PANC': ['PANC_RFX6'], 'BM': ['BM_GATA1_acute'], 'HEART': ['HEART_TBX5', 'HEART_NKX2-5']}
# fn(lab) -> draw vector or None; units = endpoint | system | EQSYS | LOSO_<sys>; fixed weights, never renormalised around a missing endpoint
def level(fn, units):
    if units in EPS: return fn(units)
    def sysv(s):
        vs = [fn(l) for l in SYS[s]]
        return None if any(v is None for v in vs) else np.mean(vs, 0)
    if units in SYS: return sysv(units) if all(l in EPS for l in SYS[units]) else None
    use = [s for s in SYS if not units.startswith('LOSO_') or s != units[5:]]
    vs = [sysv(s) if all(l in EPS for l in SYS[s]) else None for s in use]
    return None if any(v is None for v in vs) else np.mean(vs, 0)
LEVELS = EPS + ['PANC', 'BM', 'HEART', 'EQSYS', 'LOSO_PANC', 'LOSO_BM', 'LOSO_HEART']
if set(EPS) == set(ALLEP):   # missing-endpoint propagation self-test: a single missing endpoint must void every level containing it (no reweighting)
    _one = lambda drop: (lambda l: None if l == drop else np.ones(3))
    assert level(_one('HEART_TBX5'), 'EQSYS') is None and level(_one('HEART_TBX5'), 'HEART') is None and level(_one('HEART_TBX5'), 'LOSO_HEART') is not None
    assert level(_one('BM_GATA1_acute'), 'EQSYS') is None and level(_one('BM_GATA1_acute'), 'LOSO_BM') is not None and level(lambda l: np.ones(3), 'EQSYS') is not None
ROWS = []
for d in ('ALL', 'KOup', 'KOdown'):
    for k in MT:
        for lv in LEVELS:
            for arm in ['D', 'K0', 'SPL', 'RND', 'PS'] + [f'K{j}' for j in range(1, 9)]:
                v = level(lambda l: DRAW.get((l, d, k, arm)), lv)
                if v is not None: ROWS.append(dict(kind='level', direction=d, metric=k, level=lv, arm=arm, **{**stat(v), 'p_boot': np.nan}))
            for arm in ('K0', 'SPL', 'RND', 'PS'):
                v = level(lambda l: (DRAW[(l, d, k, 'D')] - DRAW[(l, d, k, arm)]) if (l, d, k, arm) in DRAW and (l, d, k, 'D') in DRAW else None, lv)
                if v is not None: ROWS.append(dict(kind='D_minus', direction=d, metric=k, level=lv, arm=arm, **stat(v)))
T = pd.DataFrame(ROWS); T['family'] = ''; T['p_holm'] = np.nan
for (d, k), fam in ((('KOup', 'AUROC'), 'PRIMARY'), (('KOdown', 'AUROC'), 'SEC_KOdown_AUROC'), (('KOup', 'APLIFT'), 'SEC_KOup_APLIFT'), (('KOdown', 'APLIFT'), 'SEC_KOdown_APLIFT')):
    ix = T[(T.kind == 'D_minus') & (T.direction == d) & (T.metric == k) & (T.level == 'EQSYS')].set_index('arm').reindex(['K0', 'SPL', 'RND', 'PS'])
    ph = holm(ix.p_boot.values)
    for arm, p_ in zip(ix.index, ph):
        m_ = (T.kind == 'D_minus') & (T.direction == d) & (T.metric == k) & (T.level == 'EQSYS') & (T.arm == arm); T.loc[m_, 'family'] = fam + ' (Holm-4)'; T.loc[m_, 'p_holm'] = p_
IR = []
for k in ('AUROC', 'AUPRC'):
    for lv in LEVELS:
        f_ = lambda l: ((DRAW[(l, 'KOup', k, 'D')] - DRAW[(l, 'KOup', k, 'K0')]) - (DRAW[(l, 'KOdown', k, 'D')] - DRAW[(l, 'KOdown', k, 'K0')])) if all((l, dd, k, aa) in DRAW for dd in ('KOup', 'KOdown') for aa in ('D', 'K0')) else None
        v = level(f_, lv)
        if v is not None: IR.append(dict(kind='interaction', metric=k, level=lv, **stat(v)))
IR = pd.DataFrame(IR)
if len(IR):
    ix = IR[IR.level == 'EQSYS']; IR.loc[ix.index, 'p_holm2_exploratory'] = holm(ix.p_boot.values)
T.to_csv(OUT + 'fig3d_rebuild_table.tsv', sep='\t', index=False); IR.to_csv(OUT + 'fig3d_rebuild_interaction.tsv', sep='\t', index=False); pd.DataFrame(COV).to_csv(OUT + 'fig3d_rebuild_coverage.tsv', sep='\t', index=False)
np.savez_compressed(OUT + 'fig3d_rebuild_draws.npz', **{'|'.join(map(str, kk)): vv for kk, vv in DRAW.items()})
REC['script_sha'] = sha(os.path.abspath(__file__)) if os.path.exists(os.path.abspath(__file__)) else None
json.dump(REC, open(OUT + 'receipt.json', 'w'), indent=1, default=str)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 400)
print(pd.DataFrame(COV).to_string())
print(T[(T.kind == 'D_minus') & (T.metric.isin(['AUROC', 'APLIFT']))].pivot_table(index=['direction', 'metric', 'arm'], columns='level', values='value').round(3).to_string())
print(T[(T.kind == 'D_minus') & (T.level == 'EQSYS') & (T.direction != 'ALL') & (T.metric != 'AUPRC')][['direction', 'metric', 'arm', 'value', 'lo', 'hi', 'p_boot', 'family', 'p_holm', 'n_valid']].round(4).to_string())
print(IR.round(4).to_string() if len(IR) else 'no interaction rows')
