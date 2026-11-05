"""Fig. S2: AUPRC by target channel (unspliced, spliced, velocity) and conditioning operator, depth window 0..8 (no conditioning: k = 0).
Bifurcation benchmark, 6 GRNs; paired t-test vs GeoBRIDGE, Holm over the other 11 arms; dashed line = label-permutation AUPRC.
Reads data/figS2_channel_operator.tsv, data/figS2_permutation_null.tsv. usage: python plot_figS2.py --out OUTDIR [--data DATADIR]"""
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

# ---------------- S2 : channel x operator, AUPRC only
GL = pd.read_csv(DATA + 'figS2_channel_operator.tsv', sep='\t'); GL = GL[(GL.kind == 'window') & (GL.topo == BIF)]
perm = float(pd.read_csv(DATA + 'figS2_permutation_null.tsv', sep='\t')['mean'].iloc[0])   # gb_si_figS2_permutation_null.py
OPS = [('R1', 'PC differences\n(GeoBRIDGE)'), ('R2h', 'Pair-difference\nSVD'), ('R2s', 'Standardized\npair-difference SVD'), ('K0', 'No conditioning\n(k = 0)')]
CH = [('U', 'Unspliced $M_u$ (GeoBRIDGE)', '#cc3311'), ('X', 'Spliced $M_s$', '#0072b2'), ('V', 'RNA velocity', '#e69f00')]
def arm(o, c, k): d = GL[(GL.arm == ('R1' if o == 'K0' else o) + '/' + c) & (GL.K == k)].set_index('grn')['AP']; assert len(d) == 6; return d
f, ax = plt.subplots(figsize=(3.8, 2.9)); ref = arm('R1', 'U', 8); rows, ps, keys = [], [], []
for i, (o, ol) in enumerate(OPS):
    for j, (c, cl, col) in enumerate(CH):
        d = arm(o, c, 0 if o == 'K0' else 8); x = i + (j - 1) * .22; mu, h = t95(d); prod = (o, c) == ('R1', 'U')
        ax.scatter(x + np.linspace(-.05, .05, 6), d.values, s=5, color=col, alpha=.45, lw=0)
        ax.errorbar(x, mu, yerr=h, fmt='o', color=col, ms=4.5, mfc=col, mec='k' if prod else col, mew=1.2 if prod else 0, capsize=2, lw=1)
        if not prod: ps.append(stats.ttest_rel(ref.values, d.values).pvalue); keys.append(len(rows))
        rows.append(dict(metric='AP', operator=o, channel=c, mean=mu, ci95_half=h, n_higher_than_production=int((d.values > ref.values).sum())))
for kk, q in zip(keys, multipletests(ps, method='holm')[1]): rows[kk]['p_holm_vs_production'] = q
rows.append(dict(metric='AP', operator='label_permutation', channel='all', mean=perm))
ax.axhline(perm, color='grey', lw=.7, ls='--'); ax.text(len(OPS) - .55, perm + .005, 'label permutation', fontsize=5.5, color='grey', ha='right', va='bottom')
ax.set_xticks(range(4)); ax.set_xticklabels([l for _, l in OPS], fontsize=6); ax.set_ylabel('AUPRC'); ax.set_xlim(-.5, 3.5)
ax.legend(handles=[plt.Line2D([], [], marker='o', ls='', color=c, ms=4, label=l) for _, l, c in CH] + [plt.Line2D([], [], marker='o', ls='', mfc='#cc3311', mec='k', mew=1.2, ms=4.5, label='GeoBRIDGE (outlined)')],
          loc='upper center', bbox_to_anchor=(.5, -.25), ncol=2, fontsize=6)
save(f, 'FigS2.png'); pd.DataFrame(rows).to_csv(O + 'FigS2_values.tsv', sep='\t', index=False)

rgb()
