"""Fig. S7: sign estimators on the synthetic benchmark (true edge sign, bifurcation) and on the knockout endpoints.
Plot only, from supplementary/data/figS7_sign_synthetic.tsv and figS7_sign_knockout.tsv.
usage: python supplementary/scripts/plot_figS7.py --out DIR [--data DIR]"""
import argparse, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument('--out', required=True)
ap.add_argument('--data', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Arial', 'DejaVu Sans'], 'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False})
S = pd.read_csv(os.path.join(a.data, 'figS7_sign_synthetic.tsv'), sep='\t'); S = S[S.slice == 'topo=bifurcation'].set_index('cand')
R = pd.read_csv(os.path.join(a.data, 'figS7_sign_knockout.tsv'), sep='\t').set_index('estimator').rename(index={'F-cen(d4)': 'F-cen', 'Pearson Ms-Mu': 'Pearson'})
NAMES = {'I-k0': 'Near-isochronic, k=0 (used)', 'I-mean0..8': 'Near-isochronic, mean k=0..8', 'F-k0': 'Forward, k=0', 'F-cen': 'Forward, k=0, centred',
         'F-mean0..8': 'Forward, mean k=0..8', 'F-k8': 'Forward, k=8', 'F-vote': 'Forward, vote over k=0..8', 'F-pool': 'Forward, pooled k=0..8',
         'Pearson': 'Cell Pearson (spliced–unspliced)', 'AGREE': 'Forward–isochronic agreement'}
order = [c for c in NAMES if c in S.index and c in R.index]
fig, axs = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
y = np.arange(len(order))[::-1]
for ax, (ttl, vals, lo, hi) in zip(axs, [('Synthetic benchmark, bifurcation (true edge sign)', S.loc[order, 'BA'], S.loc[order, 'BA_lo'], S.loc[order, 'BA_hi']),
                                         ('Real knockout endpoints (exploratory)', R.loc[order, 'mean5_BA'], None, None)]):
    col = ['k' if c == 'I-k0' else '#8a8a8a' for c in order]
    if lo is not None: ax.hlines(y, lo.values, hi.values, color=col, lw=1)
    ax.scatter(vals.values, y, c=col, s=22, zorder=3)
    ax.axvline(.5, color='#bbb', lw=.7, ls='--'); ax.set_title(ttl, fontsize=7); ax.set_xlabel('Sign balanced accuracy')
    i0 = order.index('I-k0'); ax.axhspan(y[i0] - .45, y[i0] + .45, color='#eeeeee', zorder=0)
axs[0].set_yticks(y); axs[0].set_yticklabels([NAMES[c] for c in order])
axs[0].text(.01, -.22, 'mean and 95% t-CI over 6 networks', transform=axs[0].transAxes, fontsize=6, color='#555')
axs[1].text(.01, -.22, 'equal-weight mean over 5 TFs', transform=axs[1].transAxes, fontsize=6, color='#555')
fig.tight_layout(); p = os.path.join(a.out, 'FigS7_sign_estimators.png'); fig.savefig(p, dpi=300)
Image.open(p).convert('RGB').save(p)
pd.DataFrame({'estimator': order, 'synthetic_BA': S.loc[order, 'BA'].values, 'synthetic_BA_lo': S.loc[order, 'BA_lo'].values,
              'synthetic_BA_hi': S.loc[order, 'BA_hi'].values, 'knockout_BA_mean5': R.loc[order, 'mean5_BA'].values}).to_csv(os.path.join(a.out, 'FigS7_values.tsv'), sep='\t', index=False)
print('wrote', p)
