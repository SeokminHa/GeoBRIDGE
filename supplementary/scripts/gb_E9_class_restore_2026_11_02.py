"""EVALUATION-ONLY E9: does conditioning restore d1 ranks or suppress non-direct ones? Delta=meanrankpct(GB)-meanrankpct(GB_k0) per class.
usage (server, repo root): script OUT [--grns ..] [--topos ..] [--conds ..]"""
import sys, os, json, itertools, numpy as np, pandas as pd
from scipy import stats
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch')]
import gb_tf32rep_fast_2026_10_22 as D
import gb_b4_score as B
from sklearn.metrics import average_precision_score as APs
a = sys.argv; OUT = a[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
arg = lambda k, d: a[a.index(k) + 1].split(',') if k in a else d
GRNS = [int(x) for x in arg('--grns', [501, 502, 503, 504, 505, 506])]; TOPOS = arg('--topos', ['bifurcation', 'tree']); CONDS = arg('--conds', ['C0', 'C1', 'C2', 'C3'])
FULL = len(GRNS) == 6 and len(TOPOS) == 2 and len(CONDS) == 4; SIMS = [1, 2]; ROOTS = B.GP.ROOTS
R0 = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/results')
TAB = pd.read_csv(f'{R0}/sergio_tf32rep_2026_10_22/TABLE.tsv', sep='\t').set_index(['method', 'grn', 'topo', 'cond', 'sim'])
def perm_p(x):
    x = np.asarray(x, float); o = abs(x.mean()); m = [abs((x * np.array(s)).mean()) for s in itertools.product([1, -1], repeat=len(x))]; return float(np.mean(np.array(m) >= o - 1e-12))
def ci(x):
    x = np.asarray(x, float); x = x[np.isfinite(x)]; n = len(x)
    if n < 2: return dict(mean=x.mean() if n else np.nan, lo=np.nan, hi=np.nan, n_pos=int((x > 0).sum()), n=n)
    h = stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n); return dict(mean=x.mean(), lo=x.mean() - h, hi=x.mean() + h, n_pos=int((x > 0).sum()), n=n)
CLS = [('d1', lambda d: d == 1), ('d2', lambda d: d == 2), ('d3', lambda d: d == 3), ('d4plus', lambda d: (d >= 4) & (d < 99)), ('unreach', lambda d: d >= 99)]
pct = lambda sc: np.apply_along_axis(lambda x: stats.rankdata(x) / len(x), 1, sc)
rows, par = [], []
for t in TOPOS:
    for g in GRNS:
        gd = D.gdir(g, t); pre = json.load(open(f'{gd}/roster_pre.json')); G = D.V_().generate_grn(g); mrs = set(G['mrs'])
        DM = D.dist_matrix(sorted((int(x), int(y)) for x, y in G['gt']))
        for c in CONDS:
            for s in SIMS:
                tag = D.tagof(t, g, s); Z = np.load(f'{gd}/{c}/{tag}/raw/geobridge_raw.npz')
                src = np.asarray(Z['source_gene_index'], int); cols = np.asarray(Z['cols_PI'], int); assert list(src) == pre['cands']
                Dm = np.array([[DM[s_, int(x)] for x in cols[i]] for i, s_ in enumerate(src)]); assert (Dm == 1).sum() == 240
                A = {r: np.asarray(Z[f'TRUE|U|{r}|absS'], float) for r in ROOTS}; per = [{r: A[r][:, i, cols[i]].T for r in ROOTS} for i in range(len(src))]
                S = {'GB': B.zc(per, range(0, 9)), 'GB_k0': B.zc(per, [0])}
                tr = TAB.loc[('GB', g, t, c, s)]; Y = Dm == 1
                P = {m: pct(v) for m, v in S.items()}; diff = P['GB'] - P['GB_k0']
                ismr = np.array([x in mrs for x in src])[:, None] & np.ones_like(Dm, bool)
                # parity AP (native pooled over all entries, as metrics() does with finite scores)
                ap = APs(Y.ravel(), S['GB'].ravel()); par.append(abs(ap - tr.AP)); assert par[-1] < 1e-9, ('AP parity', t, g, c, s, par[-1])
                r = dict(topo=t, grn=g, cond=c, sim=s)
                for nm, f in CLS:
                    m = f(Dm); r[f'GB_{nm}'] = P['GB'][m].mean() if m.any() else np.nan; r[f'GBk0_{nm}'] = P['GB_k0'][m].mean() if m.any() else np.nan
                    r[f'D_{nm}'] = diff[m].mean() if m.any() else np.nan
                for nm, mm in (('d1_MR', ismr), ('d1_INT', ~ismr)):
                    m = (Dm == 1) & mm; r[f'D_{nm}'] = diff[m].mean() if m.any() else np.nan; r[f'n_{nm}'] = int(m.sum())
                rows.append(r)
    print('done', t, flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT + 'dataset_class_pct.tsv', sep='\t', index=False)
V = [c for c in R.columns if c.startswith(('D_', 'GB_', 'GBk0_'))]
Gt = R.groupby(['topo', 'grn', 'cond'])[V].mean().reset_index()
GG = pd.concat([Gt, Gt.groupby(['grn', 'cond'])[V].mean().reset_index().assign(topo='all')])
# parity of class means vs T12 E3_class_means (topo all, mean over GRNs)
if FULL:
    E3 = pd.read_csv(f'{R0}/signed/r2fix_2026_11_01/T12_story_E1E2E3E7/E3_class_means.tsv', sep='\t').query("quantity=='rank_pct' and topo=='all'").set_index(['method', 'cond'])
    pd_ = 0
    for c in CONDS:
        for m, pre_ in (('GB', 'GB_'), ('GB_k0', 'GBk0_')):
            for nm, _ in CLS:
                pd_ = max(pd_, abs(GG[(GG.topo == 'all') & (GG.cond == c)][pre_ + nm].mean() - E3.loc[(m, c), nm]))
    print('class-mean parity vs T12 max|diff|', pd_); assert pd_ < 1e-9
else: pd_ = None
print('AP parity max', max(par), 'n', len(par))
out = []; NMS = [n for n, _ in CLS] + ['d1_MR', 'd1_INT']
for tp, d in GG.groupby('topo'):
    for nm in NMS:
        p = d.pivot(index='grn', columns='cond', values='D_' + nm)
        for c in CONDS: out.append(dict(topo=tp, cls=nm, contrast=f'Delta_{c}', p_signflip=perm_p(p[c].values), **ci(p[c])))
        if 'C0' in CONDS and 'C3' in CONDS: x = p['C3'] - p['C0']; out.append(dict(topo=tp, cls=nm, contrast='Delta_C3-Delta_C0', p_signflip=perm_p(x.values), **ci(x)))
E = pd.DataFrame(out); E.to_csv(OUT + 'E9_summary.tsv', sep='\t', index=False)
pd.set_option('display.width', 250)
for tp in ['all', 'bifurcation', 'tree']:
    x = E[(E.topo == tp)]
    if len(x): print('\ntopo', tp); print(x.pivot(index='cls', columns='contrast', values='mean').round(4).to_string()); print(x[['cls', 'contrast', 'mean', 'lo', 'hi', 'n_pos', 'p_signflip']].round(4).to_string(index=False)) if tp == 'all' else None
json.dump(dict(grns=GRNS, topos=TOPOS, conds=CONDS, n_datasets=len(R), ap_parity_max=max(par), class_parity_max=pd_), open(OUT + 'receipt.json', 'w'))
