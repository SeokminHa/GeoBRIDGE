"""Fig 2f: direct (d=1) vs ALL indirect (d in 2..5) pooled AUROC, bifurcation (unreachable pairs excluded). This is the Fig. 2f of the paper.
Reads the per-pair tables of gb_dvi_audit3_2026_10_29.py (full shards, not shipped); same aggregation/test/Holm-7/plot style as gb_fig2_nopearson_2026_10_30.py.
First asserts that DvI23 recomputed from the pairs equals OUT/REF_Fig2_values_holm7.tsv (copy of results/Fig2_values_holm7.tsv; <=1e-6).
usage: script OUT                (recompute; needs $GEOBRIDGE_ROOT with the per-pair tables)
       script OUT VALUES_TSV     (re-plot only from the shipped results/fig2f_dviall/Fig2f_DvIall_values.tsv)"""
import sys, os, glob
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
from sklearn.metrics import roc_auc_score as AUC
R = os.environ.get('GEOBRIDGE_ROOT', ''); OUT = sys.argv[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Nimbus Sans', 'Arial', 'DejaVu Sans'], 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
M = [('GB', 'GeoBRIDGE (Ours)', '#2166ac'), ('VEL', 'Velorama', '#2ca02c'), ('fasttenet', 'FastTENET', '#17becf'), ('genie3', 'GENIE3', '#ff7f0e'), ('grnboost2', 'GRNBoost2', '#8c564b'),
     ('pidc', 'PIDC', '#9467bd'), ('regdiffusion', 'RegDiffusion', '#e377c2'), ('deepsem', 'DeepSEM', '#bcbd22')]
COND = [('C0', 'No shared\neffect'), ('C1', 'Temporal\neffect (A)'), ('C2', 'Branch\neffect (B)'), ('C3', 'Dual effect\n(A+B)')]
NEG = {'DvI2': (2,), 'DvI23': (2, 3), 'DvIall': (2, 3, 4, 5), 'DvI45': (4, 5)}
def stars(p): return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'
if len(sys.argv) > 2:   # re-plot only: script OUT VALUES_TSV (the shipped results/fig2f_dviall/Fig2f_DvIall_values.tsv); no recomputation, no parity checks
    V = pd.read_csv(sys.argv[2], sep="\t")
else:
    fs = sorted(glob.glob(f'{R}/results/signed/dvi_audit3_2026_10_29/full/shard*/pairs_*_bifurcation_*.tsv.gz')); assert len(fs) == 48, len(fs)
    rows = []
    for f in fs:
        T = pd.read_csv(f, sep='\t'); fin = T.d[(T.d > 1) & (T.d < 99)]; assert fin.max() <= 5 and (T.d >= 1).all(), (f, sorted(T.d.unique()))
        r = dict(grn=int(T.grn[0]), cond=T.cond[0], sim=int(T.sim[0]), n1=int((T.d == 1).sum()), **{f'n{k}': int((T.d == k).sum()) for k in (2, 3, 4, 5)})
        for nm, ks in NEG.items():
            msk = (T.d == 1) | T.d.isin(ks)
            for m, *_ in M: r[f'{nm}|{m}'] = AUC(T.d[msk] == 1, T[f's|{m}'][msk])
        rows.append(r)
    D = pd.DataFrame(rows); D.to_csv(OUT + 'dviall_per_dataset.tsv', sep='\t', index=False)
    G = D.groupby(['grn', 'cond']).mean(numeric_only=True).reset_index(); assert len(G) == 24
    OUTR = []
    for nm in NEG:
        for c, _ in COND:
            S = G[G.cond == c]; ps = []
            for m, *_ in M:
                x = S[f'{nm}|{m}'].values; OUTR.append(dict(metric=nm, cond=c, method=m, mean=x.mean(), ci95_half=stats.t.ppf(.975, 5) * x.std(ddof=1) / np.sqrt(6)))
                if m != 'GB': ps.append(stats.ttest_rel(S[f'{nm}|GB'], x).pvalue)
            for (m, *_), p, ph in zip(M[1:], ps, multipletests(ps, method='holm')[1]): OUTR.append(dict(metric=nm, cond=c, method=m, p_paired_t=p, p_holm7=ph, mark=stars(ph)))
    V = pd.DataFrame(OUTR); V.to_csv(OUT + 'Fig2f_DvIall_values.tsv', sep='\t', index=False)
    # ---- PARITY (step 1): DvI23 vs shipped Fig2_values_holm7.tsv
    old = pd.read_csv(OUT + 'REF_Fig2_values_holm7.tsv', sep='\t'); old = old[(old.topo == 'bifurcation') & (old.metric == 'DvI23')]
    mine = V[V.metric == 'DvI23']
    a = old[old['mean'].notna()].merge(mine[mine['mean'].notna()], on=['cond', 'method']); b = old[old.p_holm7.notna()].merge(mine[mine.p_holm7.notna()], on=['cond', 'method'])
    dm = max((a.mean_x - a.mean_y).abs().max(), (a.ci95_half_x - a.ci95_half_y).abs().max()); dp = max((b.p_paired_t_x - b.p_paired_t_y).abs().max(), (b.p_holm7_x - b.p_holm7_y).abs().max())
    print('PARITY rows', len(a), len(b), 'max|dmean,dci|', dm, 'max|dp|', dp); assert len(a) == 32 and len(b) == 28 and dm <= 1e-6 and dp <= 1e-6, 'PARITY FAIL'
    TB = pd.read_csv(f'{R}/results/sergio_tf32rep_2026_10_22/TABLE.tsv', sep='\t'); TB = TB[TB.topo == 'bifurcation']
    L = D.melt(id_vars=['grn', 'cond', 'sim'], value_vars=[f'DvI2|{m}' for m, *_ in M]); L['method'] = L.variable.str[5:]
    q = L.merge(TB[['grn', 'cond', 'sim', 'method', 'DvI2']], on=['grn', 'cond', 'sim', 'method']); e2 = (q.value - q.DvI2).abs().max()
    print('DvI2 per-dataset PARITY vs TABLE.tsv rows', len(q), 'max|d|', e2); assert len(q) == 384 and e2 <= 1e-6, 'DvI2 PARITY FAIL'
    print('distance counts per network (sim-1 C0):'); print(D[(D.cond == 'C0') & (D.sim == 1)][['grn', 'n1', 'n2', 'n3', 'n4', 'n5']].to_string(index=False))
for NM in ('DvIall', 'DvI2'):
    # ---- panel f (DvIall), identical style
    w = .105; XO = np.arange(len(M)) * w; base = np.arange(4) * (XO[-1] + w + .28); ymax = 1.0
    fig, ax = plt.subplots(figsize=(6.2, 4.4)); tops, cph = [], []
    for ci, (c, cl) in enumerate(COND):
        for mi, (m, lab, col) in enumerate(M):
            r = V[(V.metric == NM) & (V.cond == c) & (V.method == m) & V['mean'].notna()].iloc[0]; tops.append(r['mean'] + r.ci95_half)
            ax.bar(base[ci] + XO[mi], r['mean'], w * .9, color=col); ax.errorbar(base[ci] + XO[mi], r['mean'], yerr=r.ci95_half, color='#222', lw=.7, capsize=1.5)
        cph.append([V[(V.metric == NM) & (V.cond == c) & (V.method == m) & V.p_holm7.notna()].p_holm7.iloc[0] for m, *_ in M[1:]])
    top = max(tops) + .035 * ymax
    for ci, ph in enumerate(cph):
        x0 = base[ci]; ax.plot([x0 + XO[0], x0 + XO[-1]], [top, top], color='k', lw=.7)
        for mi, p in enumerate(ph, start=1):
            xm = x0 + XO[mi]; ax.plot([xm, xm], [top, top - .012 * ymax], color='k', lw=.7); s_ = stars(p)
            if s_ != 'ns': ax.scatter([xm] * len(s_), top + .022 * ymax + np.arange(len(s_)) * .021 * ymax, marker=(6, 2, 0), s=14, color='k', lw=.8, clip_on=False, zorder=5)
    ax.axhline(.5, ls='--', color='k', lw=.9); ax.set_ylim(0, max(ymax, top + .13 * ymax)); ax.set_xticks(base + XO[-1] / 2); ax.set_xticklabels([cl for _, cl in COND], fontsize=10.5)
    ax.set_ylabel('Direct vs indirect AUROC', fontsize=15)
    for ext in ('png', 'jpg'): fig.savefig(OUT + f'Fig2f_{NM}_bifurcation.{ext}', dpi=300, bbox_inches='tight')
    plt.close(fig)
pd.set_option('display.width', 250)
Mn = V[V['mean'].notna()].assign(s=lambda d: d['mean'].map('{:.3f}'.format) + ' [' + (d['mean'] - d.ci95_half).map('{:.3f}'.format) + ',' + (d['mean'] + d.ci95_half).map('{:.3f}'.format) + ']')
print(Mn.pivot_table(index=['cond', 'method'], columns='metric', values='s', aggfunc='first').to_string())
P = V[V.p_holm7.notna()].pivot_table(index=['cond', 'method'], columns='metric', values=['p_holm7', 'mark'], aggfunc='first'); print(P.to_string())
