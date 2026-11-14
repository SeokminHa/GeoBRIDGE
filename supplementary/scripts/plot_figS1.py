import sys, os, argparse, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
ap = argparse.ArgumentParser(); ap.add_argument('--data', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')); ap.add_argument('--out', required=True)
args = ap.parse_args(); DATA = args.data.rstrip('/') + '/'; O = args.out.rstrip('/') + '/'; os.makedirs(O, exist_ok=True); BIF = 'bifurcation'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
 'legend.fontsize': 6.5, 'axes.linewidth': 0.7, 'lines.linewidth': 1.1, 'axes.edgecolor': '#888888', 'xtick.color': '#555555', 'ytick.color': '#555555',
 'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.5, 'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
M = [('GB', 'GeoBRIDGE (Ours)', '#2166ac'), ('VEL', 'Velorama', '#2ca02c'), ('fasttenet', 'FastTENET', '#17becf'), ('genie3', 'GENIE3', '#ff7f0e'), ('grnboost2', 'GRNBoost2', '#8c564b'),
     ('pidc', 'PIDC', '#9467bd'), ('regdiffusion', 'RegDiffusion', '#e377c2'), ('deepsem', 'DeepSEM', '#bcbd22')]   # = main Fig 2
CC = ['C0', 'C1', 'C2', 'C3']; CL = ['None', 'Temporal (A)', 'Branch (B)', 'A+B']; CP = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']; CS = ['none', 'A', 'B', 'AB']
CELL = (3.1, 2.4)
def t95(x): x = np.asarray(x, float); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return x.mean(), h
def stars(p): return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'
def save(f, n): f.savefig(O + n, dpi=300, bbox_inches='tight', facecolor='white'); plt.close(f); print('wrote', n)
def rgb():   # RGB, no alpha
    from PIL import Image
    for n in os.listdir(O):
        if n.endswith('.png'): Image.open(O + n).convert('RGB').save(O + n, dpi=(300, 300))

# S1a
D = pd.read_csv(DATA + 'figS1_dvi_auprc.tsv', sep='\t'); assert len(D) == 48
G = D.groupby(['grn', 'cond']).mean(numeric_only=True).reset_index(); assert len(G) == 24
rows = []; f, ax = plt.subplots(figsize=(CELL[0] * 1.5, CELL[1])); w = .105; XO = np.arange(8) * w; base = np.arange(4) * (XO[-1] + w + .28); tops, cph = [], []
for ci, c in enumerate(CC):
    S = G[G.cond == c].set_index('grn'); ps = []
    for mi, (m, l, col) in enumerate(M):
        mu, h = t95(S[f'AP|{m}']); tops.append(mu + h); rows.append(dict(panel='S1a', metric='DvIall_AUPRC', cond=c, method=m, mean=mu, ci95_half=h, n_grn=6))
        ax.bar(base[ci] + XO[mi], mu, w * .9, color=col); ax.errorbar(base[ci] + XO[mi], mu, yerr=h, color='#222', lw=.6, capsize=1.2)
        if m != 'GB': ps.append(stats.ttest_rel(S['AP|GB'], S[f'AP|{m}']).pvalue)
    ph = multipletests(ps, method='holm')[1]; cph.append(ph)
    rows += [dict(panel='S1a', metric='DvIall_AUPRC', cond=c, method=m, p_paired_t=p, p_holm7=q, mark=stars(q)) for (m, *_), p, q in zip(M[1:], ps, ph)]
    pv, h = t95(S.prev_DvIall); rows.append(dict(panel='S1a', metric='prevalence_DvIall', cond=c, method='baseline', mean=pv, ci95_half=h, n_grn=6))
    ax.plot([base[ci] - w / 2, base[ci] + XO[-1] + w / 2], [pv, pv], ls='--', color='k', lw=.8)
top = max(tops) + .035
for ci, ph in enumerate(cph):
    x0 = base[ci]; ax.plot([x0 + XO[0], x0 + XO[-1]], [top, top], color='k', lw=.6)
    for mi, p in enumerate(ph, start=1):
        xm = x0 + XO[mi]; ax.plot([xm, xm], [top, top - .012], color='k', lw=.6); s_ = stars(p)
        if s_ != 'ns': ax.scatter([xm] * len(s_), top + .022 + np.arange(len(s_)) * .028, marker=(6, 2, 0), s=6, color='k', lw=.6, clip_on=False, zorder=5)
ax.set_ylim(0, max(1, top + .13)); ax.set_xticks(base + XO[-1] / 2); ax.set_xticklabels(['No shared\neffect', 'Temporal\neffect (A)', 'Branch\neffect (B)', 'Dual effect\n(A+B)'])
ax.set_ylabel('Direct vs indirect AUPRC'); ax.grid(axis='x', visible=False); save(f, 'FigS1a.png')
f = plt.figure(figsize=CELL); f.legend([plt.Rectangle((0, 0), 1, 1, color=c) for *_, c in M] + [plt.Line2D([], [], ls='--', color='k')], [l for _, l, _ in M] + ['Prevalence (random)'], loc='center', ncol=1, fontsize=7); save(f, 'FigS1_legend.png')

# S1b-e : GeoBRIDGE vs max depth K
kd = pd.read_csv(DATA + 'figS1_kcurve.tsv', sep='\t'); assert len(kd) == 6 * 4 * 2 * 31   # bifurcation, cumulative depth window 0..K
Gk = kd.groupby(['cond', 'grn', 'K']).mean(numeric_only=True).reset_index()
for p, m, yl in [('b', 'AP', 'AUPRC'), ('c', 'AUROC', 'AUROC'), ('d', 'DvIall_AUROC', 'Direct vs indirect AUROC'), ('e', 'DvIall_AP', 'Direct vs indirect AUPRC')]:
    f, ax = plt.subplots(figsize=CELL)
    for i, c in enumerate(CC):
        d = Gk[Gk.cond == c].groupby('K')[m].apply(lambda x: pd.Series(t95(x), index=['mean', 'h'])).unstack()
        ax.fill_between(d.index, d['mean'] - d.h, d['mean'] + d.h, color=CP[i], alpha=.15, lw=0); ax.plot(d.index, d['mean'], color=CP[i], label=CL[i])
        rows += [dict(panel='S1' + p, metric=m, cond=c, method='GB', K=K, mean=r['mean'], ci95_half=r.h, n_grn=6) for K, r in d.iterrows()]
    ax.axvline(8, color='k', ls=':', lw=1); ax.set_xlabel('Maximum depth K'); ax.set_ylabel(yl)
    if p == 'b': ax.set_ylim(bottom=0.03); ax.legend(loc='lower right', ncol=2, fontsize=5.5, frameon=False, columnspacing=0.8, handlelength=1.2, borderaxespad=0.2)
    save(f, f'FigS1{p}.png')
pd.DataFrame(rows).to_csv(O + 'FigS1_values.tsv', sep='\t', index=False)

rgb()
