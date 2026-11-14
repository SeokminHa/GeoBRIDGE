# Fig 2 a-c: schematic and benchmark panels saved as independent files
import sys, os, glob
R = __import__("os").environ["GEOBRIDGE_ROOT"]
OUT = sys.argv[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch
from scipy import stats
from scipy.ndimage import gaussian_filter1d
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Nimbus Sans', 'Arial', 'DejaVu Sans'], 'font.size': 11,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
RT = f'{R}/results/sergio_tf32rep_2026_10_22'; TOPO = 'bifurcation'
ORANGE, RED, BLUE, GREY = '#f6b93b', '#d6604d', '#4393c3', '#9e9e9e'
# empirical driver curves (C3, all GRNs x sims)
edges = np.linspace(0, 3, 31); mid = (edges[:-1] + edges[1:]) / 2
acc = {k: [] for k in ('A', 'Bp', 'Bm', 'Bt')}
for f in sorted(glob.glob(f'{RT}/g*/{TOPO}/C3/*/raw/counts_truth.npz')):
    Z = np.load(f, allow_pickle=True); tau = Z['sampled_tau'].astype(float); bn = Z['sampled_bin'].astype(int); C = Z['conc_unspliced_latent'].astype(float)
    def bmean(x, m):
        return np.array([x[m & (tau >= lo) & (tau < hi + (1e-9 if hi == 3 else 0))].mean() if (m & (tau >= lo) & (tau < hi + (1e-9 if hi == 3 else 0))).any() else np.nan for lo, hi in zip(edges[:-1], edges[1:])])
    acc['A'].append(bmean(C[:, 100], np.ones(len(tau), bool))); acc['Bt'].append(bmean(C[:, 101], bn <= 1))
    acc['Bp'].append(bmean(C[:, 101], bn == 2)); acc['Bm'].append(bmean(C[:, 101], bn == 3))
n_ds = len(acc['A']); print('datasets pooled', n_ds, flush=True)
cur = {k: np.nanmean(np.vstack(v), 0) for k, v in acc.items()}
A = gaussian_filter1d(cur['A'], 1.2); A = (A - np.nanmin(A)) / (np.nanmax(A) - np.nanmin(A))
# TOP row schematic
fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.2), gridspec_kw=dict(wspace=.25))
P = {0: (0, 0), 1: (1.3, 0), 2: (2.6, .55), 3: (2.6, -.55)}
def topo(ax, col=None, lab=True):
    for i, j in ((0, 1), (1, 2), (1, 3)):
        (x0, y0), (x1, y1) = P[i], P[j]; v = np.array([x1 - x0, y1 - y0]); u = v / np.linalg.norm(v)
        c = 'k' if col is None else col.get((i, j), 'k'); lw = 1.6 if col is None else 4.5
        ax.add_patch(FancyArrowPatch((x0 + .3 * u[0], y0 + .3 * u[1]), (x1 - .3 * u[0], y1 - .3 * u[1]), arrowstyle='-|>', mutation_scale=14, lw=lw, color=c))
    for b, (x, y) in P.items(): ax.add_patch(Circle((x, y), .22, color=ORANGE, zorder=3))
    ax.set_xlim(-.4, 3.1); ax.set_ylim(-1.05, 1.05); ax.set_aspect('equal'); ax.axis('off')
ax = axs[0]; topo(ax); ax.set_title('a  Bifurcating trajectory', loc='left')
for b, t in {0: 'root', 1: 'trunk', 2: 'branch 1', 3: 'branch 2'}.items():
    ax.text(P[b][0], P[b][1] - .42 if b != 2 else P[b][1] + .32, t, ha='center', fontsize=9)
ax.text(1.3, .98, '4 states × 300 cells; branch point after trunk', ha='center', fontsize=8.5, color='#555')
ax = axs[1]; ax.plot(mid, A, color=ORANGE, lw=3); ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel('pseudotime (t)'); ax.set_ylabel('A(t)')
ax.set_title('b  Time-shared effect A(t)', loc='left'); ax.text(1.5, 1.08, 'same in both branches', ha='center', fontsize=8.5, color='#555'); ax.set_ylim(-.05, 1.2)
ax = axs[2]; T0 = np.nanmean(cur['Bt']); br = mid > 2
Bp = gaussian_filter1d(cur['Bp'][br] - T0, 1.0); Bm = gaussian_filter1d(cur['Bm'][br] - T0, 1.0); sc = max(abs(Bp[-1]), abs(Bm[-1]))
XS = lambda t: np.where(np.asarray(t) <= 2, np.asarray(t) / 2.0, 1.0 + (np.asarray(t) - 2.0) * 1.5)
ax.plot([0, 1], [0, 0], color=GREY, lw=3, ls=(0, (2, 1.2))); ax.axhline(0, color='k', lw=.6)
ax.plot(XS(np.r_[2, mid[br]]), np.r_[0, Bp / sc], color=BLUE, lw=3); ax.plot(XS(np.r_[2, mid[br]]), np.r_[0, Bm / sc], color=RED, lw=3)
ax.axvline(1, ls="--", color="k", lw=.8); ax.text(1.03, 1.15, 'branch point', ha='left', fontsize=8.5)
ax.set_xlim(0, 2.5); ax.set_ylim(-1.3, 1.3); ax.set_xticks([]); ax.set_yticks([0]); ax.set_xlabel('pseudotime (t)'); ax.set_ylabel(r'$H_B(t,b)$ − trunk level')
ax.set_title('c  Branch-shared effect B', loc='left')
fig.savefig(OUT + 'Fig2_top_schematic_draft.png', dpi=250, bbox_inches='tight'); fig.savefig(OUT + 'Fig2_top_schematic_draft.pdf', bbox_inches='tight'); plt.close(fig)
PW, PH = 13.5 * (1 - .25 * 2 / 3) / 3, 1.6
def clean_axes(ax, xlim, ylim, y0):
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis('off')
    kw = dict(arrowprops=dict(arrowstyle='-|>,head_length=.55,head_width=.22', lw=1.1, color='k', shrinkA=0, shrinkB=0))
    ax.annotate('', xy=(xlim[1], y0), xytext=(xlim[0], y0), **kw)            # x axis (through y0)
    ax.annotate('', xy=(xlim[0], ylim[1]), xytext=(xlim[0], ylim[0]), **kw)  # y axis (left)
fb, ax = plt.subplots(figsize=(PW, PH)); ax.plot(mid, A, color=ORANGE, lw=2.4, solid_capstyle='butt')
clean_axes(ax, (mid[0], 3.15), (-.08, 1.12), -.08)
fb.savefig(OUT + 'Fig2b_clean.png', dpi=300, bbox_inches='tight', transparent=True); fb.savefig(OUT + 'Fig2b_clean.pdf', bbox_inches='tight', transparent=True); plt.close(fb)
fc, ax = plt.subplots(figsize=(PW, PH)); ax.plot([0, 1], [0, 0], color=GREY, lw=2.4, solid_capstyle='butt')
ax.plot(XS(np.r_[2, mid[br]]), np.r_[0, Bp / sc], color=BLUE, lw=2.4, solid_capstyle='butt'); ax.plot(XS(np.r_[2, mid[br]]), np.r_[0, Bm / sc], color=RED, lw=2.4, solid_capstyle='butt')
clean_axes(ax, (0.0, 2.62), (-1.25, 1.25), -1.25)
fc.savefig(OUT + 'Fig2c_clean.png', dpi=300, bbox_inches='tight', transparent=True); fc.savefig(OUT + 'Fig2c_clean.pdf', bbox_inches='tight', transparent=True); plt.close(fc)
fig, axs = plt.subplots(1, 2, figsize=(9, 3)); axs[0].plot(mid, cur['A'], color=ORANGE, lw=2); axs[0].set_title('A: simulated driver (mean, C3, %d datasets)' % n_ds, fontsize=9)
for k, c, l in (('Bt', GREY, 'trunk'), ('Bp', RED, 'branch 1 (bin 2)'), ('Bm', BLUE, 'branch 2 (bin 3)')): axs[1].plot(mid, cur[k], color=c, lw=2, label=l)
axs[1].legend(frameon=False, fontsize=8); axs[1].set_title('B: simulated driver by branch', fontsize=9)
for a_ in axs: a_.set_xlabel('simulation pseudotime τ'); a_.axvline(2, ls=':', color='#888')
fig.tight_layout(); fig.savefig(OUT + 'Fig2_driver_curves_empirical.png', dpi=200); plt.close(fig)
# bottom panels, independent files
T = pd.read_csv(f'{RT}/TABLE.tsv', sep='\t'); T = T[T.topo == TOPO]
COND = [('C0', 'No shared\neffect'), ('C1', 'Temporal\neffect (A)'), ('C2', 'Branch\neffect (B)'), ('C3', 'Dual effect\n(A+B)')]
from statsmodels.stats.multitest import multipletests
def stars(p): return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'
M = [('GB', 'GeoBRIDGE (Ours)', '#2166ac'), ('VEL', 'Velorama', '#2ca02c'), ('fasttenet', 'FastTENET', '#17becf'), ('genie3', 'GENIE3', '#ff7f0e'), ('grnboost2', 'GRNBoost2', '#8c564b'),
     ('pidc', 'PIDC', '#9467bd'), ('regdiffusion', 'RegDiffusion', '#e377c2'), ('deepsem', 'DeepSEM', '#bcbd22'), ('pearson', 'Pearson |r|', '#7f7f7f')]
MET = [('AP', 'AUPRC', .5, 240 / 3168, 'd'), ('AUROC', 'AUROC', 1.0, .5, 'e'), ('DvI2', 'Direct vs indirect AUROC', 1.0, .5, 'f')]
G = T.groupby(['grn', 'cond', 'method'])[[k for k, *_ in MET]].mean().reset_index()
w = .095; XO = np.arange(len(M)) * w; base = np.arange(4) * (XO[-1] + w + .28); ROWS = []
for k, ylab, ymax, rnd, lt in MET:
    fig, ax = plt.subplots(figsize=(6.2, 4.4)); tops, cond_ph = [], []
    for ci, (c, cl) in enumerate(COND):
        S = G[G.cond == c].pivot(index='grn', columns='method', values=k); ps = []
        for mi, (m, lab, col) in enumerate(M):
            x = S[m].dropna().values; mu = x.mean(); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); tops.append(mu + h)
            ax.bar(base[ci] + XO[mi], mu, w * .9, color=col, label=lab if ci == 0 else None); ax.errorbar(base[ci] + XO[mi], mu, yerr=h, color='#222', lw=.7, capsize=1.5)
            ROWS.append(dict(metric=k, cond=c, method=m, mean=mu, ci95_half=h, n_grn=len(x)))
            if m != 'GB': ps.append(stats.ttest_rel(S['GB'], S[m]).pvalue)   # Paired comparisons over the six GRNs.
        ph = multipletests(ps, method='holm')[1]; cond_ph.append(ph)
        for (m_, *_), p_, ph_ in zip(M[1:], ps, ph): ROWS.append(dict(metric=k, cond=c, method=m_, p_paired_t=p_, p_holm8=ph_, mark=stars(ph_)))
    top = max(tops) + .035 * ymax
    for ci, ph in enumerate(cond_ph):   # one bracket height per panel; mark on each comparator tick (GB vs that method)
        x0 = base[ci]; ax.plot([x0 + XO[0], x0 + XO[-1]], [top, top], color='k', lw=.7)
        for mi, p in enumerate(ph, start=1):
            xm = x0 + XO[mi]; ax.plot([xm, xm], [top, top - .012 * ymax], color='k', lw=.7); s_ = stars(p)
            if s_ != 'ns':
                ax.scatter([xm] * len(s_), top + .022 * ymax + np.arange(len(s_)) * .021 * ymax, marker=(6, 2, 0), s=14, color='k', lw=.8, clip_on=False, zorder=5)
    ax.axhline(rnd, ls='--', color='k', lw=.9); ax.set_ylim(0, max(ymax, top + .13 * ymax)); ax.set_xticks(base + XO[-1] / 2); ax.set_xticklabels([cl for _, cl in COND], fontsize=10.5)
    ax.set_ylabel(ylab, fontsize=15)
    fig.savefig(OUT + f'Fig2{lt}_{k}_bifurcation.png', dpi=300, bbox_inches='tight'); fig.savefig(OUT + f'Fig2{lt}_{k}_bifurcation.pdf', bbox_inches='tight'); plt.close(fig)
fig = plt.figure(figsize=(12, .6)); hd = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in M] + [plt.Line2D([], [], ls='--', color='k')]
fig.legend(hd, [l for _, l, _ in M] + ['Random'], loc='center', ncol=10, frameon=False, fontsize=9); fig.savefig(OUT + 'Fig2_bottom_legend.png', dpi=300, bbox_inches='tight'); fig.savefig(OUT + 'Fig2_bottom_legend.pdf', bbox_inches='tight')
pd.DataFrame(ROWS).to_csv(OUT + 'Fig2_bottom_values.tsv', sep='\t', index=False); print('saved', OUT, flush=True)
