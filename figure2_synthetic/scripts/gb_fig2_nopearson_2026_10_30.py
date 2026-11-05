"""Fig 2 d/e/f without |Pearson| (PI 2026-10-30: drop Pearson; Holm over the remaining 7 comparators). PLOTTING + Holm family change only.
Numbers, aggregation and test are unchanged from manuscript/figures_2026_10_28/fig2_draft/gb_fig2_schematic_2026_10_28.py (d AUPRC, e AUROC) and
gb_fig2f_dvi23_2026_10_29.py (f, direct vs 2-and-3-hop AUROC from results/signed/dvi_audit3_2026_10_29/full/agg/D1_vs_D2D3_datasets.tsv):
sims -> GRN mean; mean over 6 GRNs; 95% t CI; two-sided paired t over the 6 GRNs (GB vs each method), Holm over the 7 comparators per condition and metric.
usage (server, repo root): script OUT"""
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
R = __import__("os").environ["GEOBRIDGE_ROOT"]; RT = f'{R}/results/sergio_tf32rep_2026_10_22'; OUT = sys.argv[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Nimbus Sans', 'Arial', 'DejaVu Sans'], 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False})
T = pd.read_csv(f'{RT}/TABLE.tsv', sep='\t'); K = ['grn', 'topo', 'cond', 'sim', 'method']
A = pd.read_csv(f'{R}/results/signed/dvi_audit3_2026_10_29/full/agg/D1_vs_D2D3_datasets.tsv', sep='\t')
chk = A[A.neg == 'D2'].merge(T[K + ['DvI2']], on=K); assert len(chk) == 864 and np.allclose(chk.pooled, chk.DvI2, atol=1e-12), 'DvI2 parity vs TABLE.tsv'
T = T.merge(A[A.neg == 'D2+D3'].rename(columns={'pooled': 'DvI23'})[K + ['DvI23']], on=K, how='left'); assert T.DvI23.notna().all()
COND = [('C0', 'No shared\neffect'), ('C1', 'Temporal\neffect (A)'), ('C2', 'Branch\neffect (B)'), ('C3', 'Dual effect\n(A+B)')]
def stars(p): return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'
M = [('GB', 'GeoBRIDGE (Ours)', '#2166ac'), ('VEL', 'Velorama', '#2ca02c'), ('fasttenet', 'FastTENET', '#17becf'), ('genie3', 'GENIE3', '#ff7f0e'), ('grnboost2', 'GRNBoost2', '#8c564b'),
     ('pidc', 'PIDC', '#9467bd'), ('regdiffusion', 'RegDiffusion', '#e377c2'), ('deepsem', 'DeepSEM', '#bcbd22')]
MET = [('AP', 'AUPRC', .5, 240 / 3168, 'd'), ('AUROC', 'AUROC', 1.0, .5, 'e'), ('DvI23', 'Direct vs indirect AUROC', 1.0, .5, 'f')]
ROWS = []
for TOPO in ('bifurcation', 'tree'):
    G = T[T.topo == TOPO].groupby(['grn', 'cond', 'method'])[[k for k, *_ in MET]].mean().reset_index()
    w = .105; XO = np.arange(len(M)) * w; base = np.arange(4) * (XO[-1] + w + .28)
    for k, ylab, ymax, rnd, lt in MET:
        fig, ax = plt.subplots(figsize=(6.2, 4.4)); tops, cond_ph = [], []
        for ci, (c, cl) in enumerate(COND):
            S = G[G.cond == c].pivot(index='grn', columns='method', values=k); ps = []
            for mi, (m, lab, col) in enumerate(M):
                x = S[m].dropna().values; mu = x.mean(); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); tops.append(mu + h)
                ax.bar(base[ci] + XO[mi], mu, w * .9, color=col); ax.errorbar(base[ci] + XO[mi], mu, yerr=h, color='#222', lw=.7, capsize=1.5)
                ROWS.append(dict(topo=TOPO, metric=k, cond=c, method=m, mean=mu, ci95_half=h, n_grn=len(x)))
                if m != 'GB': ps.append(stats.ttest_rel(S['GB'], S[m]).pvalue)
            ph = multipletests(ps, method='holm')[1]; cond_ph.append(ph)
            for (m_, *_), p_, ph_ in zip(M[1:], ps, ph): ROWS.append(dict(topo=TOPO, metric=k, cond=c, method=m_, p_paired_t=p_, p_holm7=ph_, mark=stars(ph_)))
        top = max(tops) + .035 * ymax
        for ci, ph in enumerate(cond_ph):
            x0 = base[ci]; ax.plot([x0 + XO[0], x0 + XO[-1]], [top, top], color='k', lw=.7)
            for mi, p in enumerate(ph, start=1):
                xm = x0 + XO[mi]; ax.plot([xm, xm], [top, top - .012 * ymax], color='k', lw=.7); s_ = stars(p)
                if s_ != 'ns':
                    ax.scatter([xm] * len(s_), top + .022 * ymax + np.arange(len(s_)) * .021 * ymax, marker=(6, 2, 0), s=14, color='k', lw=.8, clip_on=False, zorder=5)
        ax.axhline(rnd, ls='--', color='k', lw=.9); ax.set_ylim(0, max(ymax, top + .13 * ymax)); ax.set_xticks(base + XO[-1] / 2); ax.set_xticklabels([cl for _, cl in COND], fontsize=10.5)
        ax.set_ylabel(ylab, fontsize=15); fig.savefig(OUT + f'Fig2{lt}_{k}_{TOPO}.png', dpi=300, bbox_inches='tight'); plt.close(fig)
fig = plt.figure(figsize=(12, .6)); hd = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in M] + [plt.Line2D([], [], ls='--', color='k')]
fig.legend(hd, [l for _, l, _ in M] + ['Random'], loc='center', ncol=9, frameon=False, fontsize=9); fig.savefig(OUT + 'Fig2_bottom_legend.png', dpi=300, bbox_inches='tight')
V = pd.DataFrame(ROWS); V.to_csv(OUT + 'Fig2_values_holm7.tsv', sep='\t', index=False)
# check: means identical to the frozen 8-method table; report every star change vs Holm-8
old = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else f'{R}/manuscript/figures_2026_10_28/fig2_draft/Fig2_bottom_values.tsv', sep='\t')
mo = old[old['mean'].notna()].merge(V[(V.topo == 'bifurcation') & V['mean'].notna()], on=['metric', 'cond', 'method']); print('mean max |diff| vs frozen:', float((mo.mean_x - mo.mean_y).abs().max()), len(mo))
po = old[old.p_holm8.notna()].merge(V[(V.topo == 'bifurcation') & V.p_holm7.notna()], on=['metric', 'cond', 'method'])
po = po[po.mark_x != po.mark_y]; print('bifurcation d/e star changes (Holm-8 -> Holm-7):'); print(po[['metric', 'cond', 'method', 'p_holm8', 'mark_x', 'p_holm7', 'mark_y']].to_string() if len(po) else 'none')
print(V[(V.topo == 'bifurcation') & V.p_holm7.notna()].pivot_table(index=['cond', 'method'], columns='metric', values='mark', aggfunc='first').to_string())
print('saved', OUT)
