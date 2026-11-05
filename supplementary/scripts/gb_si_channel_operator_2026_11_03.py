"""SI: operator x target channel, fixed window k=0..8 (kind=window,K=8), GRN-level (6 GRNs, all topologies pooled). usage: script AGG_DIR OUT"""
import sys, os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
A, OUT = sys.argv[1].rstrip('/') + '/', sys.argv[2].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
G = pd.read_csv(A + 'GRN_LEVEL_ALL.tsv', sep='\t'); G = G[(G.kind == 'window') & (G.K == 8)]
assert G.groupby('arm').grn.nunique().eq(6).all() and G.arm.nunique() == 9
P = pd.read_csv(A + 'LABEL_PERMUTATION_CONTROL.tsv', sep='\t', header=None, skiprows=4, index_col=0)
P.columns = ['AP_mean', 'AP_std', 'DvI2_mean', 'DvI2_std']; perm = P[['AP_mean', 'DvI2_mean']].mean()
MET = {'AP': 'AUPRC', 'DvI2': 'Direct vs indirect AUROC (DvI2)'}
ops = [('R1', 'R1 cell-PC\n(production)'), ('R2h', 'R2h pair-SVD'), ('R2s', 'R2s standardized\npair-SVD')]
ch = [('U', 'U (Mu, production)', '#cc3311'), ('X', 'X (Ms)', '#0072b2'), ('V', 'V (velocity)', '#e69f00')]
def ci(x):
    x = np.asarray(x, float); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return x.mean(), x.mean() - h, x.mean() + h
rows = []
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Nimbus Sans', 'Arial', 'DejaVu Sans'], 'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6})
fig, axs = plt.subplots(1, 2, figsize=(5.4, 2.4))
for ax, m in zip(axs, MET):
    for i, (o, _) in enumerate(ops):
        for j, (c, _, col) in enumerate(ch):
            arm = f'{o}/{c}'; d = G[G.arm == arm].sort_values('grn')[m].to_numpy(); mu, lo, hi = ci(d); x = i + (j - 1) * .26; prod = arm == 'R1/U'
            ax.scatter(x + np.linspace(-.05, .05, 6), d, s=6, color=col, alpha=.5, lw=0)
            ax.errorbar(x + .1, mu, yerr=[[mu - lo], [hi - mu]], fmt='o', color=col, ms=4.5 if prod else 3.5, capsize=2, lw=1.4 if prod else .8, mec='k' if prod else col, mew=.9)
            rows.append(dict(arm=arm, operator=o, channel=c, metric=m, mean=mu, lo95=lo, hi95=hi, n_grn=6, production=prod, label_perm_mean=perm[m + '_mean']))
    ax.axhline(perm[m + '_mean'], color='grey', lw=.7, ls='--'); ax.set_xticks(range(3)); ax.set_xticklabels([l for _, l in ops], fontsize=6)
    ax.set_ylabel(MET[m]); ax.set_xlim(-.5, 2.5)
axs[0].text(2.45, perm['AP_mean'] + .01, 'label-permutation', fontsize=5.5, color='grey', ha='right', va='bottom')
h = [plt.Line2D([], [], marker='o', ls='', color=c, ms=4, label=l) for _, l, c in ch] + [plt.Line2D([], [], marker='o', ls='', mfc='w', mec='k', ms=4.5, label='R1/U = production (outlined)')]
fig.legend(handles=h, frameon=False, loc='lower center', ncol=4, fontsize=6); fig.tight_layout(rect=(0, .08, 1, 1))
fig.savefig(OUT + 'S_channel_operator.png', dpi=300); pd.DataFrame(rows).to_csv(OUT + 'S_channel_operator.tsv', sep='\t', index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))
