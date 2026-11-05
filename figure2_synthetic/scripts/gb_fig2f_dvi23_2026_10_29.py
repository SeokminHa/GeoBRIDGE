"""Fig 2f v3 (PI 2026-10-29): direct-vs-indirect AUROC with indirect = strict 2-hop AND 3-hop (D1 vs D2 u D3), pooled over the 32 regulators
exactly as the Fig 2 metric (gb_tf32rep_fast_2026_10_22.metrics: AUC over finite pairs with Dm==1 vs Dm in negatives). PLOTTING + metric swap only.
Per-dataset AUROCs come from the cached final-audit table (results/signed/dvi_audit3_2026_10_29/full/agg/D1_vs_D2D3_datasets.tsv), whose pair tables
were asserted to reproduce TABLE.tsv DvI2/DvI3 to 1e-12; here the D2-only column is re-asserted against TABLE.tsv DvI2 before plotting.
Style, aggregation (sims -> GRN mean; mean over 6 GRNs; 95% t CI), paired t over GRNs + Holm over 8 comparators per condition, and star rendering are
copied unchanged from manuscript/figures_2026_10_28/fig2_draft/gb_fig2_schematic_2026_10_28.py (panel f).
usage (server, repo root): script OUT"""
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
R = __import__("os").environ["GEOBRIDGE_ROOT"]; RT = f'{R}/results/sergio_tf32rep_2026_10_22'
OUT = sys.argv[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Nimbus Sans', 'Arial', 'DejaVu Sans'], 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
A = pd.read_csv(f'{R}/results/signed/dvi_audit3_2026_10_29/full/agg/D1_vs_D2D3_datasets.tsv', sep='\t')
T = pd.read_csv(f'{RT}/TABLE.tsv', sep='\t')
K = ['grn', 'topo', 'cond', 'sim', 'method']
chk = A[A.neg == 'D2'].merge(T[K + ['DvI2']], on=K); assert len(chk) == 864 and np.allclose(chk.pooled, chk.DvI2, atol=1e-12), 'DvI2 parity vs TABLE.tsv'
chk3 = A[A.neg == 'D3'].merge(T[K + ['DvI3']], on=K); assert len(chk3) == 864 and np.allclose(chk3.pooled, chk3.DvI3, atol=1e-12), 'DvI3 parity vs TABLE.tsv'
D = A[A.neg == 'D2+D3'].rename(columns={'pooled': 'DvI23'})[K + ['DvI23']].merge(T[K + ['DvI2']], on=K); assert len(D) == 864
COND = [('C0', 'No shared\neffect'), ('C1', 'Temporal\neffect (A)'), ('C2', 'Branch\neffect (B)'), ('C3', 'Dual effect\n(A+B)')]
def stars(p): return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'
M = [('GB', 'GeoBRIDGE (Ours)', '#2166ac'), ('VEL', 'Velorama', '#2ca02c'), ('fasttenet', 'FastTENET', '#17becf'), ('genie3', 'GENIE3', '#ff7f0e'), ('grnboost2', 'GRNBoost2', '#8c564b'),
     ('pidc', 'PIDC', '#9467bd'), ('regdiffusion', 'RegDiffusion', '#e377c2'), ('deepsem', 'DeepSEM', '#bcbd22'), ('pearson', 'Pearson |r|', '#7f7f7f')]
ROWS = []
for TOPO in ('bifurcation', 'tree'):
    G = D[D.topo == TOPO].groupby(['grn', 'cond', 'method'])[['DvI23', 'DvI2']].mean().reset_index()
    for k, ylab, ymax, rnd, lt in (('DvI23', 'Direct vs indirect AUROC', 1.0, .5, 'f'), ('DvI2', 'Direct vs indirect AUROC', 1.0, .5, 'f')):
        w = .095; XO = np.arange(len(M)) * w; base = np.arange(4) * (XO[-1] + w + .28)
        fig, ax = plt.subplots(figsize=(6.2, 4.4)); tops, cond_ph = [], []
        for ci, (c, cl) in enumerate(COND):
            S = G[G.cond == c].pivot(index='grn', columns='method', values=k); ps = []
            for mi, (m, lab, col) in enumerate(M):
                x = S[m].dropna().values; mu = x.mean(); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); tops.append(mu + h)
                ax.bar(base[ci] + XO[mi], mu, w * .9, color=col, label=lab if ci == 0 else None); ax.errorbar(base[ci] + XO[mi], mu, yerr=h, color='#222', lw=.7, capsize=1.5)
                ROWS.append(dict(topo=TOPO, metric=k, cond=c, method=m, mean=mu, ci95_half=h, n_grn=len(x)))
                if m != 'GB': ps.append(stats.ttest_rel(S['GB'], S[m]).pvalue)
            ph = multipletests(ps, method='holm')[1]; cond_ph.append(ph)
            for (m_, *_), p_, ph_ in zip(M[1:], ps, ph): ROWS.append(dict(topo=TOPO, metric=k, cond=c, method=m_, p_paired_t=p_, p_holm8=ph_, mark=stars(ph_)))
        top = max(tops) + .035 * ymax
        for ci, ph in enumerate(cond_ph):
            x0 = base[ci]; ax.plot([x0 + XO[0], x0 + XO[-1]], [top, top], color='k', lw=.7)
            for mi, p in enumerate(ph, start=1):
                xm = x0 + XO[mi]; ax.plot([xm, xm], [top, top - .012 * ymax], color='k', lw=.7); s_ = stars(p)
                if s_ != 'ns':
                    ax.scatter([xm] * len(s_), top + .022 * ymax + np.arange(len(s_)) * .021 * ymax, marker=(6, 2, 0), s=14, color='k', lw=.8, clip_on=False, zorder=5)
        ax.axhline(rnd, ls='--', color='k', lw=.9); ax.set_ylim(0, max(ymax, top + .13 * ymax)); ax.set_xticks(base + XO[-1] / 2); ax.set_xticklabels([cl for _, cl in COND], fontsize=10.5)
        ax.set_ylabel(ylab, fontsize=15)
        nm = f'Fig2{lt}_{k}_{TOPO}' + ('_REBUILT_CHECK' if k == 'DvI2' else '')
        fig.savefig(OUT + nm + '.png', dpi=300, bbox_inches='tight'); fig.savefig(OUT + nm + '.pdf', bbox_inches='tight'); plt.close(fig)
V = pd.DataFrame(ROWS); V.to_csv(OUT + 'Fig2f_DvI23_values.tsv', sep='\t', index=False)
# old Fig 2f table (bifurcation DvI2) must be reproduced exactly by the rebuilt DvI2 branch
old = pd.read_csv(f'{R}/manuscript/figures_2026_10_28/fig2_draft/Fig2_bottom_values.tsv', sep='\t') if os.path.exists(f'{R}/manuscript/figures_2026_10_28/fig2_draft/Fig2_bottom_values.tsv') else None
pd.set_option('display.width', 220)
P = V[V.p_paired_t.notna()].pivot_table(index=['topo', 'cond', 'method'], columns='metric', values=['p_holm8'], aggfunc='first')
P.columns = [f'{a}_{b}' for a, b in P.columns]; P['mark_DvI2'] = P.p_holm8_DvI2.map(stars); P['mark_DvI23'] = P.p_holm8_DvI23.map(stars)
P['changed'] = P.mark_DvI2 != P.mark_DvI23; print(P.round(4).to_string()); print('marks changed:', int(P.changed.sum()), 'of', len(P))
Mn = V[V['mean'].notna()].pivot_table(index=['topo', 'cond', 'method'], columns='metric', values='mean'); print(Mn.round(3).to_string())
print('saved', OUT)
