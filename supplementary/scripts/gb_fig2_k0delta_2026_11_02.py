"""Fig 2 conditioning-gain panels (PI 2026-11-01 proposal; internal review consensus, internal design note (not shipped)).
Input: kcurve_datasets.tsv from scratch/gb_fig2_kcurve_2026_11_01.py (per dataset x kind x K; parity to Fig 2 table exact). Plotting + paired stats only.
MAIN panel e: paired differences production (cumulative K=8) minus unconditioned (K=0), sims averaged within GRN, 6 GRN points per condition,
  mean and 95% t CI; metrics AUPRC and DvI23 AUROC (manuscript Fig 2f definition); bifurcation (main topology).
SI: Delta(K) = metric(cumulative 0..K) - metric(K=0), paired within GRN, pointwise 95% t CI over GRNs, K=0..30, 4 conditions, both topologies;
  production K=8 marked (not the observed peak). usage: script KCURVE_DIR OUT"""
import sys, os, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
IN, OUT = sys.argv[1].rstrip('/') + '/', sys.argv[2].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
R = pd.read_csv(IN + 'kcurve_datasets.tsv', sep='\t'); R = R[R.kind == 'cumulative']
MET = {'AP': 'AUPRC', 'DvI23_AUROC': 'Direct vs indirect AUROC'}
G = R.groupby(['topo', 'cond', 'grn', 'K'])[list(MET)].mean().reset_index()                    # sims -> GRN
base = G[G.K == 0].set_index(['topo', 'cond', 'grn'])[list(MET)]
D = G.set_index(['topo', 'cond', 'grn']); D = D[list(MET)].sub(base).assign(K=D.K).reset_index()   # paired within GRN
assert np.allclose(D[D.K == 0][list(MET)].to_numpy(), 0)
def ci(x):
    x = np.asarray(x, float); n = len(x); h = stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n); return x.mean(), x.mean() - h, x.mean() + h, n
rows = []
for (t, c, K), d in D.groupby(['topo', 'cond', 'K']):
    for m in MET:
        mu, lo, hi, n = ci(d[m]); r = dict(topo=t, cond=c, K=K, metric=m, mean=mu, lo=lo, hi=hi, n_grn=n, n_pos=int((d[m] > 0).sum()))
        if K > 0: r['p_t'] = stats.ttest_1samp(d[m], 0).pvalue
        rows.append(r)
S = pd.DataFrame(rows); S.to_csv(OUT + 'delta_vs_k0.tsv', sep='\t', index=False)
CC = {'C0': ('#333333', 'None'), 'C1': ('#e69f00', 'A'), 'C2': ('#0072b2', 'B'), 'C3': ('#cc3311', 'A+B')}
LONG = {'None': 'No shared effect', 'A': 'Temporal (A)', 'B': 'Branch (B)', 'A+B': 'Dual (A+B)'}; SH = {'AP': 'AUPRC', 'DvI23_AUROC': 'DvI AUROC'}
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Nimbus Sans', 'Arial', 'DejaVu Sans'], 'font.size': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6})
# ---- MAIN panel e (bifurcation, K=8 vs K=0)
fig, axs = plt.subplots(1, 2, figsize=(3.6, 1.9))
for ax, m in zip(axs, MET):
    for i, (c, (col, lab)) in enumerate(CC.items()):
        d = D[(D.topo == 'bifurcation') & (D.cond == c) & (D.K == 8)][m].to_numpy(); mu, lo, hi, n = ci(d)
        ax.scatter(i + np.linspace(-.12, .12, len(d)), d, s=9, color=col, alpha=.55, lw=0)
        ax.errorbar(i + .3, mu, yerr=[[mu - lo], [hi - mu]], fmt='o', color=col, ms=4, capsize=2, lw=.9)
    ax.axhline(0, color='grey', lw=.6, ls='--'); ax.set_xticks(range(4)); ax.set_xticklabels([v[1] for v in CC.values()], fontsize=6.5); ax.set_xlabel('Shared effect')
    ax.set_ylabel(f'Δ {SH[m]} (GeoBRIDGE − k = 0)'); ax.set_xlim(-.5, 3.6)
fig.tight_layout(); fig.savefig(OUT + 'Fig2e_k0_paired_delta_bifurcation.png', dpi=300); plt.close(fig)
# ---- SI Delta(K) curves
fig, axs = plt.subplots(2, 2, figsize=(5.2, 3.8), sharex=True)
for i, t in enumerate(['bifurcation', 'tree']):
    for j, m in enumerate(MET):
        ax = axs[i, j]
        for c, (col, lab) in CC.items():
            d = S[(S.topo == t) & (S.cond == c) & (S.metric == m)].sort_values('K')
            ax.plot(d.K, d['mean'], color=col, lw=1, label=LONG[lab]); ax.fill_between(d.K, d.lo, d.hi, color=col, alpha=.12, lw=0)
        ax.axhline(0, color='grey', lw=.6, ls='--'); ax.axvline(8, color='k', lw=.6, ls=':')
        ax.set_xlim(0, 30); ax.set_ylabel(f'Δ {SH[m]} vs K = 0'); ax.set_title(t.capitalize(), fontsize=7)
        if i == 1: ax.set_xlabel('Maximum depth K (aggregated over 0..K)')
h, l = axs[0, 0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, loc='lower center', ncol=4, fontsize=6.5)
fig.tight_layout(rect=(0, .06, 1, 1)); fig.savefig(OUT + 'FigS_k0_delta_curves.png', dpi=300); plt.close(fig)
print(S[(S.K == 8)].round(4).to_string(index=False))
