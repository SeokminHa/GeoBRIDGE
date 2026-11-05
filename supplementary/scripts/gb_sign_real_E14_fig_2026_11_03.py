# Plot-only: reads E14 tables. SI style of gb_si_figs_2026_11_03.py. usage: script DIR
import sys, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = sys.argv[1].rstrip('/') + '/'
G1, G2, G3 = '#555555', '#888888', '#bbbbbb'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5, 'legend.fontsize': 6.5,
 'axes.linewidth': 0.7, 'axes.edgecolor': G2, 'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.5, 'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False,
 'legend.frameon': False, 'savefig.dpi': 300})
u = pd.read_csv(D + 'E14_sign_real_per_unit.tsv', sep='\t'); e = pd.read_csv(D + 'E14_sign_real_equal_TF_mean.tsv', sep='\t')
order = [k for k in e.estimator if k != 'F-mean0..8(2/3)']; units = ['TBX5', 'NKX2-5', 'GATA1', 'KLF1', 'KLF1_B', 'RFX6']
PAL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
f, (a, b) = plt.subplots(1, 2, figsize=(7.2, 3.6), gridspec_kw={'width_ratios': [1.5, 1], 'wspace': 0.08}, sharey=True)
y = np.arange(len(order))
for i, un in enumerate(units):
    d = u[u.unit == un].set_index('estimator').loc[order]
    a.scatter(d.BA, y + (i - 2.5) * 0.1, s=9, color=PAL[i], label=un, zorder=3, marker='o' if un != 'KLF1_B' else 'D', facecolors=PAL[i] if un != 'KLF1_B' else 'white')
a.axvline(0.5, color='black', lw=0.8); a.set_yticks(y); a.set_yticklabels(order); a.invert_yaxis(); a.set_xlabel('Direction balanced accuracy (KO responders; miss = wrong)')
a.legend(loc='lower right', title='KO unit', ncol=2); a.text(-0.02, 1.04, 'a', transform=a.transAxes, fontweight='bold', fontsize=9, ha='right')
e = e.set_index('estimator').loc[order]
b.barh(y - 0.2, e.mean5_KO_up_recall, 0.2, color='#eb6834', label='KO-up recall'); b.barh(y, e.mean5_BA, 0.2, color='#555555', label='BA'); b.barh(y + 0.2, e.mean5_KO_down_recall, 0.2, color='#2a78d6', label='KO-down recall')
for yy, c in zip(y, e.mean5_coverage): b.text(0.99, yy, f'cov {c:.2f}', va='center', ha='right', fontsize=5.5, color=G1)
b.axvline(0.5, color='black', lw=0.8); b.set_xlim(0.3, 1.0); b.set_xlabel('Equal-TF-weight mean (5 TFs, KLF1_B excluded)'); b.legend(loc='lower left', fontsize=5.8)
b.text(-0.02, 1.04, 'b', transform=b.transAxes, fontweight='bold', fontsize=9, ha='right')
f.suptitle('Real KO sign estimators (selection set; exploratory)', fontsize=7.5, y=1.0)
f.savefig(D + 'E14_sign_real.png', dpi=300, bbox_inches='tight'); print('wrote')
