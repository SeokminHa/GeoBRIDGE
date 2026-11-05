"""EVALUATION-ONLY story analyses E1 (degradation, native/per-source-z/macro), E2 (Delta_gain GB-k0), E3 (class rank mechanism), E7 (k-curve summary).
Inputs: stored GB raw |S| (raw/geobridge_raw.npz), comparator score npz, TABLE.tsv, T5_kcurve/kcurve_datasets.tsv. No rescoring beyond frozen zcons (B.zc) over depths.
usage (server, repo root): script OUT [--grns 501,..] [--topos bifurcation,tree]"""
import sys, os, json, itertools, numpy as np, pandas as pd
from scipy import stats
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch')]
import gb_tf32rep_fast_2026_10_22 as D
import gb_b4_score as B
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as APs
a = sys.argv; OUT = a[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
GRNS = [int(x) for x in a[a.index('--grns') + 1].split(',')] if '--grns' in a else [501, 502, 503, 504, 505, 506]
TOPOS = a[a.index('--topos') + 1].split(',') if '--topos' in a else ['bifurcation', 'tree']
CONDS = ['C0', 'C1', 'C2', 'C3']; SIMS = [1, 2]; FULL = len(GRNS) == 6 and len(TOPOS) == 2
RT = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/results/sergio_tf32rep_2026_10_22'); ROOTS = B.GP.ROOTS
TAB = pd.read_csv(f'{RT}/TABLE.tsv', sep='\t').set_index(['method', 'grn', 'topo', 'cond', 'sim'])
KC = pd.read_csv((__import__("os").environ["GEOBRIDGE_ROOT"] + '/results/signed/r2fix_2026_11_01/T5_kcurve/kcurve_datasets.tsv'), sep='\t').query("kind=='cumulative'").set_index(['topo', 'grn', 'cond', 'sim', 'K'])
EXT = {'deepsem': 'DeepSEM', 'regdiffusion': 'RegDiffusion', 'fasttenet': 'FastTENET', 'genie3': 'GENIE3', 'grnboost2': 'GRNBoost2', 'pidc': 'PIDC', 'pearson': 'Pearson'}
TBL = {'GB': 'GB', 'VEL': 'VEL', **{k: k for k in EXT}}                                            # internal name -> TABLE method
NAME = {'GB': 'GB', 'GBk0': 'GB_k0', 'VEL': 'Velorama', **EXT}
def perm_p(x):                                                                                      # exact two-sided sign-flip p of the mean
    x = np.asarray(x, float); n = len(x); o = abs(x.mean()); m = [abs((x * np.array(s)).mean()) for s in itertools.product([1, -1], repeat=n)]
    return float(np.mean(np.array(m) >= o - 1e-12))
def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2: return dict(mean=x.mean() if n else np.nan, lo=np.nan, hi=np.nan, n_pos=int((x > 0).sum()), n=n)
    h = stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n); return dict(mean=x.mean(), lo=x.mean() - h, hi=x.mean() + h, n_pos=int((x > 0).sum()), n=n)
def zrow(sc):
    out = np.zeros_like(sc, float)
    for i in range(sc.shape[0]):
        r = sc[i]; f = np.isfinite(r); sd = r[f].std() if f.any() else 0; out[i] = np.where(f, (r - r[f].mean()) / sd if sd > 0 else 0., np.nan)
    return out
def dvi(sc, Dm, neg, ok):
    m = ok & ((Dm == 1) | np.isin(Dm, neg)); return AUC(Dm[m] == 1, sc[m])
def metrics(sc, Dm):
    Y = Dm == 1; r = {}
    for view, S in (('native', sc), ('z', zrow(sc))):
        ok = np.isfinite(S); r[f'AP_{view}'] = APs(Y[ok], S[ok]); r[f'DvI23_{view}'] = dvi(S, Dm, (2, 3), ok); r[f'DvI2_{view}'] = dvi(S, Dm, (2,), ok)
    for nm, neg in (('DvI23', (2, 3)), ('DvI2', (2,))):
        v = []
        for i in range(sc.shape[0]):
            ok = np.isfinite(sc[i]); m = ok & ((Dm[i] == 1) | np.isin(Dm[i], neg))
            if (Dm[i][m] == 1).any() and (Dm[i][m] != 1).any(): v.append(AUC(Dm[i][m] == 1, sc[i][m]))
        r[f'{nm}_macro'] = np.mean(v) if v else np.nan
    return r
CLS = [('d1', lambda d: d == 1), ('d2', lambda d: d == 2), ('d3', lambda d: d == 3), ('d4plus', lambda d: (d >= 4) & (d < 99)), ('unreach', lambda d: d >= 99)]
def classmeans(M, Dm): return {nm: (float(M[f(Dm)].mean()) if f(Dm).any() else np.nan) for nm, f in CLS}
rows, e3, par, missing, nds = [], [], [], [], 0
for t in TOPOS:
    for g in GRNS:
        gd = D.gdir(g, t); pre = json.load(open(f'{gd}/roster_pre.json')); G = D.V_().generate_grn(g); DM = D.dist_matrix(sorted((int(x), int(y)) for x, y in G['gt']))
        for c in CONDS:
            for s in SIMS:
                nds += 1; tag = D.tagof(t, g, s); dd = f'{gd}/{c}/{tag}'; Z = np.load(f'{dd}/raw/geobridge_raw.npz')
                src = np.asarray(Z['source_gene_index'], int); cols = np.asarray(Z['cols_PI'], int); assert list(src) == pre['cands']
                assert np.array_equal(cols, np.array([[x for x in range(100) if x != q] for q in pre['cands']]))
                Dm = np.array([[DM[s_, int(x)] for x in cols[i]] for i, s_ in enumerate(src)]); assert (Dm == 1).sum() == 240
                A = {r: np.asarray(Z[f'TRUE|U|{r}|absS'], float) for r in ROOTS}; per = [{r: A[r][:, i, cols[i]].T for r in ROOTS} for i in range(len(src))]
                S = {'GB': B.zc(per, range(0, 9)), 'GBk0': B.zc(per, [0])}
                vf = f'{gd}/velorama_rt0/{c}/{tag}.npz'
                if os.path.exists(vf): z = np.load(vf); assert np.array_equal(z['src'], src) and np.array_equal(z['cols'], cols); S['VEL'] = np.asarray(z['score'], float)
                else: missing.append(('VEL', t, g, c, s))
                for m in EXT:
                    p = D.score_path(m, ('rep', g, t, c, s))
                    if os.path.exists(p):
                        z = np.load(p); assert np.array_equal(z['src'], src) and np.array_equal(z['cols'], cols) and str(z['cols_sha']) == pre['cols_sha']; S[m] = np.asarray(z['score'], float)
                    else: missing.append((m, t, g, c, s))
                missing.append(('RegVelo', t, g, c, s))                                              # no RegVelo scores exist in tf32rep
                for m, sc in S.items():
                    assert sc.shape == (32, 99) and np.isfinite(sc).all(), (m, t, g, c, s)
                    r = metrics(sc, Dm); rows.append(dict(method=NAME[m], topo=t, grn=g, cond=c, sim=s, **r))
                    if m in TBL:                                                                         # parity vs TABLE.tsv
                        tr = TAB.loc[(TBL[m], g, t, c, s)]; d = max(abs(r['AP_native'] - tr.AP), abs(r['DvI2_native'] - tr.DvI2)); par.append((m, d))
                        assert d < (1e-9 if m == 'GB' else 1e-6), ('TABLE parity', m, t, g, c, s, d)
                    pct = np.apply_along_axis(lambda x: stats.rankdata(x) / len(x), 1, sc); e3.append(dict(method=NAME[m], topo=t, grn=g, cond=c, sim=s, quantity='rank_pct', **classmeans(pct, Dm)))
                    if m in ('GB', 'GBk0'): e3.append(dict(method=NAME[m], topo=t, grn=g, cond=c, sim=s, quantity='score_value', **classmeans(sc, Dm)))
                for K, nm in ((8, 'GB'), (0, 'GBk0')):                                                  # parity vs stored k-curve
                    kr = KC.loc[(t, g, c, s, K)]; rr = [x for x in rows if x['method'] == NAME[nm]][-1]
                    d = max(abs(rr['AP_native'] - kr.AP), abs(rr['DvI23_native'] - kr.DvI23_AUROC), abs(rr['DvI2_native'] - kr.DvI2_AUROC)); par.append(('kcurve_' + nm, d)); assert d < 1e-9, ('kcurve parity', nm, t, g, c, s, d)
    print('done', t, flush=True)
assert nds == len(GRNS) * len(TOPOS) * 8; 
if FULL: assert nds == 96
miss = pd.DataFrame(missing, columns=['method', 'topo', 'grn', 'cond', 'sim']); miss.to_csv(OUT + 'missing_methods.tsv', sep='\t', index=False)
print('MISSING (method: n datasets):', miss.groupby('method').size().to_dict())
pp = pd.DataFrame(par, columns=['what', 'd']).groupby('what').d.agg(['count', 'max']); print('PARITY max|diff| per method (count):\n', pp.to_string()); pp.to_csv(OUT + 'parity.tsv', sep='\t')
R = pd.DataFrame(rows); R.to_csv(OUT + 'dataset_metrics.tsv', sep='\t', index=False)
VIEWS = [c for c in R.columns if c not in ('method', 'topo', 'grn', 'cond', 'sim')]
# GRN level: sims -> GRN (per topo), plus pooled over topologies ('all')
Gt = R.groupby(['method', 'topo', 'grn', 'cond'])[VIEWS].mean().reset_index()
Ga = Gt.groupby(['method', 'grn', 'cond'])[VIEWS].mean().reset_index().assign(topo='all')
GG = pd.concat([Gt, Ga]); GG.to_csv(OUT + 'grn_metrics.tsv', sep='\t', index=False)
def deltas(GG, views, base='C0', pairs=None):
    out = []
    for (m, tp), d in GG.groupby(['method', 'topo']):
        P = d.pivot(index='grn', columns='cond', values=views) if len(views) > 1 else None
        for v in views:
            p = d.pivot(index='grn', columns='cond', values=v)
            for cj in ['C1', 'C2', 'C3']: out.append(dict(method=m, topo=tp, metric=v, contrast=f'{cj}-C0', **ci(p[cj] - p['C0'])))
            for cj in CONDS: out.append(dict(method=m, topo=tp, metric=v, contrast=f'level_{cj}', **ci(p[cj])))
    return pd.DataFrame(out)
E1 = deltas(GG, VIEWS); E1.to_csv(OUT + 'E1_degradation.tsv', sep='\t', index=False)
# ---- E2
e2 = []
for tp, d in GG[GG.method.isin(['GB', 'GB_k0'])].groupby('topo'):
    for v in ['AP_native', 'DvI23_native', 'DvI2_native', 'AP_z', 'DvI23_z']:
        p = d.pivot(index='grn', columns=['method', 'cond'], values=v); df = {c: p[('GB', c)] - p[('GB_k0', c)] for c in CONDS}
        for nm, x in [(f'gain_C{j}', df[f'C{j}'] - df['C0']) for j in (1, 2, 3)] + [('interaction_C3-C1-C2+C0', df['C3'] - df['C1'] - df['C2'] + df['C0'])] + [(f'GBminusk0_level_{c}', df[c]) for c in CONDS]:
            e2.append(dict(topo=tp, metric=v, contrast=nm, p_signflip=perm_p(x.values), **ci(x)))
E2 = pd.DataFrame(e2); E2.to_csv(OUT + 'E2_delta_gain.tsv', sep='\t', index=False)
# ---- E3
E3 = pd.DataFrame(e3); CL = [n for n, _ in CLS]
E3g = E3.groupby(['quantity', 'method', 'topo', 'grn', 'cond'])[CL].mean().reset_index()
E3g = pd.concat([E3g, E3g.groupby(['quantity', 'method', 'grn', 'cond'])[CL].mean().reset_index().assign(topo='all')])
lv = E3g.groupby(['quantity', 'method', 'topo', 'cond'])[CL].mean().reset_index(); lv.to_csv(OUT + 'E3_class_means.tsv', sep='\t', index=False)
e3d = []
for (q, m, tp), d in E3g.groupby(['quantity', 'method', 'topo']):
    for cl in CL:
        p = d.pivot(index='grn', columns='cond', values=cl)
        for cj in ['C1', 'C2', 'C3']: e3d.append(dict(quantity=q, method=m, topo=tp, cls=cl, contrast=f'{cj}-C0', **ci(p[cj] - p['C0'])))
E3d = pd.DataFrame(e3d); E3d.to_csv(OUT + 'E3_class_delta.tsv', sep='\t', index=False)
# ---- E7 (from stored k-curve)
e7, e7g = [], []
for m in ['DvI23_AUROC', 'AP']:
    for tp in TOPOS + ['all']:
        k = KC.reset_index(); k = k[k.grn.isin(GRNS) & k.topo.isin(TOPOS)]
        if tp != 'all': k = k[k.topo == tp]
        k = k.groupby(['cond', 'grn', 'K'])[m].mean().reset_index()
        g8 = {}
        for c in CONDS:
            P = k[k.cond == c].pivot(index='grn', columns='K', values=m); gain = P.sub(P[0], axis=0); mg = gain.mean(0); mx = mg.max()
            k95 = int(mg.index[np.argmax((mg >= .95 * mx).values)]) if mx > 0 else np.nan
            pk = [int(r.index[np.argmax((r >= .95 * r.max()).values)]) if r.max() > 0 else np.nan for _, r in gain.iterrows()]
            g8[c] = gain[8]; e7.append(dict(metric=m, topo=tp, cond=c, K95_of_mean_curve=k95, max_gain=mx, K_at_max=int(mg.idxmax()), gain_K8=mg[8], frac_of_max_at_K8=mg[8] / mx if mx > 0 else np.nan,
                                              per_grn_K95=str(pk), per_grn_K95_median=np.nanmedian(pk), **{f'g8_{kk}': vv for kk, vv in ci(gain[8]).items()}))
        for cj in ['C1', 'C2', 'C3']: e7g.append(dict(metric=m, topo=tp, contrast=f'gain8_{cj}-C0', p_signflip=perm_p((g8[cj] - g8['C0']).values), **ci(g8[cj] - g8['C0'])))
E7 = pd.DataFrame(e7); E7.to_csv(OUT + 'E7_kcurve_summary.tsv', sep='\t', index=False); E7g = pd.DataFrame(e7g); E7g.to_csv(OUT + 'E7_gain8_contrasts.tsv', sep='\t', index=False)
# ---- print
pd.set_option('display.width', 250); f = lambda d: d.round(4)
print('\nE1 C3-C0, topo=all'); x = E1[(E1.topo == 'all') & (E1.contrast == 'C3-C0') & E1.metric.isin(['AP_native', 'AP_z', 'DvI23_native', 'DvI23_z', 'DvI23_macro', 'DvI2_native', 'DvI2_z'])]
print(f(x.pivot(index='method', columns='metric', values='mean')).to_string()); print(f(x[x.metric.isin(['DvI23_native', 'DvI23_z'])][['method', 'metric', 'mean', 'lo', 'hi', 'n_pos']]).to_string())
print('\nE1 levels DvI23_native, topo=all'); print(f(E1[(E1.topo == 'all') & E1.contrast.str.startswith('level') & (E1.metric == 'DvI23_native')].pivot(index='method', columns='contrast', values='mean')).to_string())
print('\nE2 topo all'); print(f(E2[E2.topo == 'all'][['metric', 'contrast', 'mean', 'lo', 'hi', 'n_pos', 'p_signflip']]).to_string())
print('\nE3 delta C3-C0 rank_pct topo all'); print(f(E3d[(E3d.quantity == 'rank_pct') & (E3d.topo == 'all') & (E3d.contrast == 'C3-C0')].pivot(index='method', columns='cls', values='mean')).to_string())
print('\nE3 GB zcons delta C3-C0'); print(f(E3d[(E3d.quantity == 'score_value') & (E3d.topo == 'all') & (E3d.contrast == 'C3-C0')].pivot(index='method', columns='cls', values='mean')).to_string())
print('\nE7'); print(f(E7[E7.topo == 'all'][['metric', 'cond', 'K95_of_mean_curve', 'per_grn_K95_median', 'max_gain', 'gain_K8', 'frac_of_max_at_K8', 'g8_lo', 'g8_hi']]).to_string()); print(f(E7g[E7g.topo == 'all']).to_string())
json.dump(dict(n_datasets=nds, grns=GRNS, topos=TOPOS), open(OUT + 'receipt.json', 'w'))
