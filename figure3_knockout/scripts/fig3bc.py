# Fig 3 b/c as individual panels
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from statsmodels.stats.multitest import multipletests
a = sys.argv; T = pd.read_csv(a[1], sep='\t'); O = a[2].rstrip('/') + '/'; os.makedirs(O, exist_ok=True)
W = float(a[a.index('--w') + 1]) if '--w' in a else 2.3; H = float(a[a.index('--h') + 1]) if '--h' in a else 1.9
if '--keep-pearson' not in a: T = T[T.method != '|Pearson|'].copy()
T['p_holm'] = np.nan
for _, g in T[T.method != 'GeoBRIDGE'].groupby(['endpoint', 'direction', 'metric']):
    ok = g.p_boot.notna(); T.loc[g.index[ok], 'p_holm'] = multipletests(g.p_boot[ok], method='holm')[1]
T.drop(columns=['p_holm8']).to_csv(O + 'fig3_bc_holm%d.tsv' % (T[T.method != 'GeoBRIDGE'].method.nunique()), sep='\t', index=False)
K0 = None
if '--k0' in a:   # unconditioned ablation (k=0) on the same rows and draws; arm D must reproduce the GeoBRIDGE row
    R_ = pd.read_csv(a[a.index('--k0') + 1], sep='	'); R_ = R_[R_.kind == 'level'].rename(columns={'level': 'endpoint'})
    chk = R_[R_.arm == 'D'].merge(T[T.method == 'GeoBRIDGE'], on=['endpoint', 'direction', 'metric']); assert len(chk) and np.allclose(chk.value_x, chk.value_y, atol=1e-9), 'arm D != GeoBRIDGE'
    K0 = R_[R_.arm == 'K0'].assign(method='GeoBRIDGE (k=0)')[['endpoint', 'direction', 'metric', 'method', 'value', 'lo', 'hi']]; print('k0 parity rows', len(chk))
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Nimbus Sans', 'Arial', 'DejaVu Sans'], 'font.size': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6})
RED, GREY, BLUE = '#c0392b', '#b5b9be', '#2166ac'; PINK = '#e8a39a'
def stars(p): return '***' if p < .005 else '**' if p < .01 else '*' if p < .05 else ''
for dr in ('ALL', 'KOup', 'KOdown'):
    for ep in ['EQSYS', 'PANC_RFX6', 'BM_GATA1_acute', 'HEART_TBX5', 'HEART_NKX2-5']:
        for met, ref, tag in (('APLIFT', 1.0, 'b'), ('AUROC', 0.5, 'c')):
            d = T[(T.endpoint == ep) & (T.direction == dr) & (T.metric == met)].copy()
            if d.empty: continue
            gb = d[d.method == 'GeoBRIDGE']; rest = d[d.method != 'GeoBRIDGE'].sort_values('value', ascending=False)
            k0 = K0[(K0.endpoint == ep) & (K0.direction == dr) & (K0.metric == met)] if K0 is not None else gb.iloc[:0]; d = pd.concat([gb, k0, rest])
            fig, ax = plt.subplots(figsize=(W, H)); y = np.arange(len(d))[::-1]
            lo = min(d.lo.min(), ref); hi = max(d.hi.max(), ref); pad = .06 * (hi - lo); xr = hi + pad
            for yi, (_, r) in zip(y, d.iterrows()):
                ax.barh(yi, r.value - ref, left=ref, color=RED if r.method == 'GeoBRIDGE' else PINK if r.method == 'GeoBRIDGE (k=0)' else GREY, height=.66, lw=0)
                ax.plot([r.lo, r.hi], [yi, yi], color='#333333', lw=.7); ax.plot([r.lo, r.lo], [yi - .15, yi + .15], color='#333333', lw=.7); ax.plot([r.hi, r.hi], [yi - .15, yi + .15], color='#333333', lw=.7)
                if r.method not in ('GeoBRIDGE', 'GeoBRIDGE (k=0)') and np.isfinite(r.p_holm):
                    s = stars(r.p_holm)
                    if s: ax.text(xr, yi, s, ha='left', va='center', fontsize=7, color='k' if r.GB_minus > 0 else BLUE, clip_on=False)
            ax.set_xlim(lo - pad if met == 'AUROC' else max(0, lo - pad), xr); ax.axvline(ref, color='#666666', lw=.6, ls=':')
            ax.set_yticks(y); ax.set_yticklabels(d.method, fontsize=6.5); [t.set_color(RED if t.get_text() == 'GeoBRIDGE' else '#c98476') for t in ax.get_yticklabels() if t.get_text().startswith('GeoBRIDGE')]
            ax.set_xlabel('AUPRC lift' if met == 'APLIFT' else 'AUROC', fontsize=7); ax.tick_params(axis='x', labelsize=6.5)
            fig.savefig(O + (f'Fig3{tag}_{ep}_{met}.png' if dr == 'ALL' else f'SI_dir/FigS3_{dr}_{ep}_{met}.png'), dpi=300, bbox_inches='tight') if dr == 'ALL' or os.makedirs(O + 'SI_dir', exist_ok=True) is None else None; plt.close(fig)
print('saved', O)
