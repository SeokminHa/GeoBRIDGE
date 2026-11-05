"""SI figure: sign-estimator comparison, synthetic truth (E13) vs real KO selection set (E14). Plotting only."""
import pandas as pd, numpy as np, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False})
D = 'results/signed/r2fix_2026_11_01/'
S = pd.read_csv(D + 'E13_sign_synth/SUMMARY.tsv', sep='\t'); S = S[S.slice == 'overall'].set_index('cand')
R = pd.read_csv(D + 'E14_sign_real/E14_sign_real_equal_TF_mean.tsv', sep='\t').set_index('estimator').rename(index={'F-cen(d4)': 'F-cen', 'I-k0(adopted)': 'I-k0', 'Pearson Ms-Mu': 'Pearson'})
NAMES = {'I-k0': 'Near-isochronic, k=0 (adopted)', 'I-mean0..8': 'Near-isochronic, mean k=0..8', 'F-k0': 'Forward, k=0', 'F-cen': 'Forward, k=0, centred',
         'F-mean0..8': 'Forward, mean k=0..8', 'F-k8': 'Forward, k=8', 'F-vote': 'Forward, vote over k=0..8', 'F-pool': 'Forward, pooled k=0..8',
         'Pearson': 'Cell Pearson (spliced–unspliced)', 'AGREE': 'Forward–isochronic agreement'}
order = [c for c in NAMES if c in S.index and c in R.index]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
y = np.arange(len(order))[::-1]
for ax, (ttl, vals, lo, hi) in zip(axs, [('Synthetic benchmark (true edge sign)', S.loc[order, 'BA'], S.loc[order, 'BA_lo'], S.loc[order, 'BA_hi']),
                                         ('Real knockout endpoints (selection set; exploratory)', R.loc[order, 'mean5_BA'], None, None)]):
    col = ['k' if c == 'I-k0' else '#8a8a8a' for c in order]
    if lo is not None: ax.hlines(y, lo.values, hi.values, color=col, lw=1)
    ax.scatter(vals.values, y, c=col, s=22, zorder=3)
    ax.axvline(.5, color='#bbb', lw=.7, ls='--'); ax.set_title(ttl, fontsize=7); ax.set_xlabel('Sign balanced accuracy')
    i0 = order.index('I-k0'); ax.axhspan(y[i0] - .45, y[i0] + .45, color='#eeeeee', zorder=0)
axs[0].set_yticks(y); axs[0].set_yticklabels([NAMES[c] for c in order])
axs[0].text(.01, -.22, 'mean and 95% t-CI over 6 networks', transform=axs[0].transAxes, fontsize=6, color='#555')
axs[1].text(.01, -.22, 'equal-weight mean over 5 TFs', transform=axs[1].transAxes, fontsize=6, color='#555')
fig.tight_layout(); fig.savefig(D + 'SI_FIGS_2026_11_03/S_sign_estimators.png', dpi=300); print('ok')
