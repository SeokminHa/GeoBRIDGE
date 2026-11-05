"""R2-fix T1(A) figure: Fig 1 d-f (bifurcation) with extra 'GeoBRIDGE (k=0)' pink bar after GeoBRIDGE. Copy of gb_fig2_nopearson_2026_10_30.py (bar/CI/star code unchanged:
paired t GB vs each of 7 comparators, Holm-7; k=0 bar is NOT in the test family). Panel f here = DvI2 (2-hop), the metric of the anchors; k0 per-dataset values from K0_TABLE_bifurcation.tsv.
usage (server): script K0_DIR OUT"""
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
R = __import__("os").environ["GEOBRIDGE_ROOT"]; RT = f'{R}/results/sergio_tf32rep_2026_10_22'; OUT = sys.argv[2].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Nimbus Sans', 'Arial', 'DejaVu Sans'], 'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
T = pd.read_csv(f'{RT}/TABLE.tsv', sep='\t'); T = T[T.topo == 'bifurcation']
K0 = pd.read_csv(sys.argv[1].rstrip('/') + '/K0_TABLE_bifurcation.tsv', sep='\t'); T = pd.concat([T, K0], ignore_index=True)
COND = [('C0', 'No shared\neffect'), ('C1', 'Temporal\neffect (A)'), ('C2', 'Branch\neffect (B)'), ('C3', 'Dual effect\n(A+B)')]
def stars(p): return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'
M = [('GB', 'GeoBRIDGE (Ours)', '#c0392b'), ('GBk0', 'GeoBRIDGE (k=0)', '#e8a39a'), ('VEL', 'Velorama', '#2ca02c'), ('fasttenet', 'FastTENET', '#17becf'), ('genie3', 'GENIE3', '#ff7f0e'),
     ('grnboost2', 'GRNBoost2', '#8c564b'), ('pidc', 'PIDC', '#9467bd'), ('regdiffusion', 'RegDiffusion', '#e377c2'), ('deepsem', 'DeepSEM', '#bcbd22')]
MET = [('AP', 'AUPRC', .5, 240 / 3168, 'd'), ('AUROC', 'AUROC', 1.0, .5, 'e'), ('DvI2', 'Direct vs indirect AUROC', 1.0, .5, 'f')]
G = T.groupby(['grn', 'cond', 'method'])[[k for k, *_ in MET]].mean().reset_index()
w = .095; XO = np.arange(len(M)) * w; base = np.arange(4) * (XO[-1] + w + .28)
for k, ylab, ymax, rnd, lt in MET:
    fig, ax = plt.subplots(figsize=(6.6, 4.4)); tops, cond_ph = [], []
    for ci, (c, cl) in enumerate(COND):
        S = G[G.cond == c].pivot(index='grn', columns='method', values=k); ps = []; tm = []
        for mi, (m, lab, col) in enumerate(M):
            x = S[m].dropna().values; assert len(x) == 6; mu = x.mean(); h = stats.t.ppf(.975, 5) * x.std(ddof=1) / np.sqrt(6); tops.append(mu + h)
            ax.bar(base[ci] + XO[mi], mu, w * .9, color=col); ax.errorbar(base[ci] + XO[mi], mu, yerr=h, color='#222', lw=.7, capsize=1.5)
            if m not in ('GB', 'GBk0'): ps.append(stats.ttest_rel(S['GB'], S[m]).pvalue); tm.append(mi)
        cond_ph.append((tm, multipletests(ps, method='holm')[1]))
    top = max(tops) + .035 * ymax
    for ci, (tm, ph) in enumerate(cond_ph):
        x0 = base[ci]; ax.plot([x0 + XO[tm[0]], x0 + XO[-1]], [top, top], color='k', lw=.7)
        for mi, p in zip(tm, ph):
            xm = x0 + XO[mi]; ax.plot([xm, xm], [top, top - .012 * ymax], color='k', lw=.7); s_ = stars(p)
            if s_ != 'ns': ax.scatter([xm] * len(s_), top + .022 * ymax + np.arange(len(s_)) * .021 * ymax, marker=(6, 2, 0), s=14, color='k', lw=.8, clip_on=False, zorder=5)
    ax.axhline(rnd, ls='--', color='k', lw=.9); ax.set_ylim(0, max(ymax, top + .13 * ymax)); ax.set_xticks(base + XO[-1] / 2); ax.set_xticklabels([cl for _, cl in COND], fontsize=10.5)
    ax.set_ylabel(ylab, fontsize=15); fig.savefig(OUT + f'Fig1{lt}_{k}_bifurcation_withK0.png', dpi=300, bbox_inches='tight'); plt.close(fig)
fig = plt.figure(figsize=(12, .6)); hd = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in M] + [plt.Line2D([], [], ls='--', color='k')]
fig.legend(hd, [l for _, l, _ in M] + ['Random'], loc='center', ncol=10, frameon=False, fontsize=9); fig.savefig(OUT + 'Fig1_legend_withK0.png', dpi=300, bbox_inches='tight'); print('saved', OUT)
