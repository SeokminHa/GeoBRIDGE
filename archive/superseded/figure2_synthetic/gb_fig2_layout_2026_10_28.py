"""Fig 2 layout draft (PI 2026-10-28): 2 x 3.  Top: (a) bifurcation topology / pseudotime design, (b) time-shared hidden input A(tau),
(c) branch-shared hidden input B(tau, branch).  Bottom: AUPRC, AUROC, direct-vs-indirect AUROC on the bifurcation benchmark (TABLE.tsv, the same
numbers and statistics as gb_fig2_bars_v2_2026_10_27.py: GRN-level mean over sims, mean over 6 GRNs, 95% t interval). PLOTTING ONLY.
Design facts are read from the generator (scratch/gb_pilot_AB_tf32rep_2026_10_22.py: TOPOLOGIES, DEPTH, BRANCH, CELLS_PER_TYPE, ab_profiles, ab_hn) and the
driver trajectories are the SIMULATED latent hidden-driver concentrations of one dataset (raw/counts_truth.npz conc_unspliced_latent columns 100 (A), 101 (B)).
usage (server, repo root): script OUT [--grn 501] [--cond C3] [--sim 1]"""
import sys, os
R = __import__("os").environ["GEOBRIDGE_ROOT"]
a = sys.argv; OUT = a[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
GRN = int(a[a.index('--grn') + 1]) if '--grn' in a else 501; CND = a[a.index('--cond') + 1] if '--cond' in a else 'C3'; SIM = int(a[a.index('--sim') + 1]) if '--sim' in a else 1
sys.path[:0] = [f'{R}/scratch']
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
import gb_pilot_AB_tf32rep_2026_10_22 as V
TOPO = 'bifurcation'; NB, BMAT = V.TOPOLOGIES[TOPO]; DEP = V.DEPTH[TOPO]; BR = V.BRANCH[TOPO]; NC = V.CELLS_PER_TYPE
rateA, rateB = V.ab_profiles(TOPO); hA, hB = V.ab_hn(TOPO)
RT = f'{R}/results/sergio_tf32rep_2026_10_22'
d = f'{RT}/g{GRN}/{TOPO}/{CND}/{TOPO}_AB_q2_lam1_obsUMI_DEPTH_depth10_g{GRN}_s{SIM}'
Z = np.load(f'{d}/raw/counts_truth.npz', allow_pickle=True)
tau, bn = np.asarray(Z['sampled_tau'], float), np.asarray(Z['sampled_bin'], int); C = np.asarray(Z['conc_unspliced_latent'], float); drv = np.asarray(Z['driver_ids'], int)
assert list(drv) == [100, 101] and C.shape[1] == 102 and len(tau) == NB * NC
print('design: bins', NB, 'cells/bin', NC, 'depth', DEP, 'branch', BR, 'rateA', rateA.round(3), 'rateB', rateB.round(3), 'hA', hA, 'hB', hB, flush=True)
print('tau range per bin', {b: (float(tau[bn == b].min()), float(tau[bn == b].max()), float((tau[bn == b] == DEP[b] + 1).mean())) for b in range(NB)}, flush=True)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Nimbus Sans', 'Arial', 'DejaVu Sans'], 'font.size': 10,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
BC = {0: '#9e9e9e', 1: '#5f5f5f', 2: '#d6604d', 3: '#4393c3'}; BL = {0: 'bin 0 (root)', 1: 'bin 1 (trunk)', 2: 'bin 2 (branch +)', 3: 'bin 3 (branch −)'}
fig = plt.figure(figsize=(17, 10)); gs = fig.add_gridspec(2, 3, hspace=.42, wspace=.28, height_ratios=[1, 1.05])
# ---------------- (a) topology
ax = fig.add_subplot(gs[0, 0]); rng = np.random.default_rng(0)
yb = {0: 0.0, 1: 0.0, 2: .6, 3: -.6}
for b in range(NB):
    m = bn == b; x = tau[m]; y = yb[b] + (0 if b < 2 else (x - 2) * 0) + rng.normal(0, .06, m.sum())
    if b >= 2: y = np.sign(yb[b]) * .6 * (x - 2) + rng.normal(0, .06, m.sum())
    ax.scatter(x, y, s=4, color=BC[b], alpha=.5, lw=0)
ax.plot([0, 2], [0, 0], color='k', lw=1.4); ax.plot([2, 3], [0, .6], color='k', lw=1.4); ax.plot([2, 3], [0, -.6], color='k', lw=1.4)
ax.plot([2], [0], 'o', ms=8, mfc='white', mec='k', zorder=5); ax.annotate('branch point\nτ = 2', (2, 0), (1.62, -.62), fontsize=8.5, arrowprops=dict(arrowstyle='-', lw=.6))
for b, (xt, yt) in {0: (.5, .2), 1: (1.5, .2), 2: (2.55, .72), 3: (2.55, -.78)}.items():
    ax.text(xt, yt, f'{BL[b]}\n{NC} cells', ha='center', fontsize=8.5, color=BC[b] if b >= 2 else 'k')
ax.set_xlim(-.1, 3.25); ax.set_ylim(-1.0, 1.0); ax.set_yticks([]); ax.spines['left'].set_visible(False)
ax.set_xlabel('simulation pseudotime τ = depth(bin) + min(step / 300, 1)'); ax.set_title('a  Bifurcation topology (SERGIO dynamics, 1,200 cells)', loc='left', fontsize=11)
# ---------------- (b) A(tau)
def driver_panel(ax, col, rate, hn, title, ylab):
    for b in range(NB):
        m = bn == b; ax.scatter(tau[m], C[m, col], s=4, color=BC[b], alpha=.45, lw=0)
    ax2 = ax.twinx(); ax2.spines['top'].set_visible(False)
    for b in range(NB):
        ax2.plot([DEP[b], DEP[b] + 1], [rate[b], rate[b]], color=BC[b], lw=2.4, solid_capstyle='butt')
    ax2.set_ylabel('per-bin basal production rate (input to SERGIO)', fontsize=8.5); ax2.set_ylim(0, 4.3)
    ax.set_xlabel('pseudotime τ'); ax.set_ylabel(ylab, fontsize=9); ax.set_title(title, loc='left', fontsize=11); ax.axvline(2, ls=':', color='k', lw=.8)
driver_panel(fig.add_subplot(gs[0, 1]), 100, rateA, hA, 'b  Time-shared hidden input A(τ)', 'simulated latent driver A (unspliced conc.)')
driver_panel(fig.add_subplot(gs[0, 2]), 101, rateB, hB, 'c  Branch-shared hidden input B(τ, branch)', 'simulated latent driver B (unspliced conc.)')
# ---------------- bottom: benchmark bars (bifurcation)
T = pd.read_csv(f'{RT}/TABLE.tsv', sep='\t'); T = T[T.topo == TOPO]
COND = [('C0', 'No shared\ninput'), ('C1', 'Temporal\ninput (A)'), ('C2', 'Branch\ninput (B)'), ('C3', 'Dual\ninput (A+B)')]
M = [('GB', 'GeoBRIDGE', '#2166ac'), ('VEL', 'Velorama', '#2ca02c'), ('fasttenet', 'FastTENET', '#17becf'), ('genie3', 'GENIE3', '#ff7f0e'), ('grnboost2', 'GRNBoost2', '#8c564b'),
     ('pidc', 'PIDC', '#9467bd'), ('regdiffusion', 'RegDiffusion', '#e377c2'), ('deepsem', 'DeepSEM', '#bcbd22'), ('pearson', 'Pearson |r|', '#7f7f7f')]
MET = [('AP', 'AUPRC', .5, 240 / 3168, 'd'), ('AUROC', 'AUROC', 1.0, .5, 'e'), ('DvI2', 'Direct vs indirect AUROC', 1.0, .5, 'f')]
G = T.groupby(['grn', 'cond', 'method'])[[k for k, *_ in MET]].mean().reset_index()
w = .095; XO = np.arange(len(M)) * w; base = np.arange(4) * (XO[-1] + w + .28)
for j, (k, ylab, ymax, rnd, lt) in enumerate(MET):
    ax = fig.add_subplot(gs[1, j])
    for ci, (c, cl) in enumerate(COND):
        S = G[G.cond == c].pivot(index='grn', columns='method', values=k)
        for mi, (m, lab, col) in enumerate(M):
            x = S[m].dropna().values; mu = x.mean(); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
            ax.bar(base[ci] + XO[mi], mu, w * .9, color=col, label=lab if ci == 0 else None); ax.errorbar(base[ci] + XO[mi], mu, yerr=h, color='#222', lw=.7, capsize=1.5)
    ax.axhline(rnd, ls='--', color='k', lw=.9); ax.set_ylim(0, ymax); ax.set_xticks(base + XO[-1] / 2); ax.set_xticklabels([cl for _, cl in COND], fontsize=8.5)
    ax.set_ylabel(ylab); ax.set_title(f'{lt}  {ylab} (bifurcation, 6 GRNs)', loc='left', fontsize=11)
hd, lb = fig.axes[-1].get_legend_handles_labels()
fig.legend(hd + [plt.Line2D([], [], ls='--', color='k')], lb + ['Random'], loc='lower center', ncol=10, frameon=False, fontsize=9, bbox_to_anchor=(.5, .005))
fig.savefig(OUT + 'Fig2_layout_draft_2026_10_28.png', dpi=220, bbox_inches='tight'); fig.savefig(OUT + 'Fig2_layout_draft_2026_10_28.pdf', bbox_inches='tight')
print('saved', OUT, 'dataset', d, flush=True)
