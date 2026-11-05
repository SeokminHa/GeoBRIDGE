"""Fig 3d as individual panels with minimal text (PI 2026-10-30 polish; internal review FIG34_POLISH Q3: keep all systems in f). Plotting only; identical
inputs and quantities as scratch/gb_fig3d_v8_plot_2026_10_29.py (v8 A -> d, B -> e, C top -> f_top, C bottom -> f_bottom). All explanatory text moved to the legend.
usage (server, repo root): script OUTDIR"""
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); O = sys.argv[1].rstrip('/') + '/'; os.makedirs(O, exist_ok=True); FB = R + 'results/signed/fig3d_rebuild_2026_10_29/'
DS = pd.read_csv(FB + 'devstructure/panc_devstructure.tsv', sep='\t'); GP_ = pd.read_csv(FB + 'geom30/geom30_pooled.tsv', sep='\t')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Nimbus Sans', 'Arial', 'DejaVu Sans'], 'font.size': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6, 'axes.labelsize': 8.5})
GREEN, PURP, GREY = '#1b9e77', '#7570b3', '#7f7f7f'; FC = {'Alpha': '#e7298a', 'Beta': '#7570b3', 'Delta': '#e6ab02', 'Epsilon': '#66a61e'}
def save(fig, n): fig.savefig(O + n, dpi=300, bbox_inches='tight'); plt.close(fig)
os.makedirs(O + 'SI', exist_ok=True)
# d
fig, ax = plt.subplots(figsize=(1.5, 2.0))
for j, (sel, col) in enumerate(((DS.block == 'T_temporal', GREEN), (DS.quantity == 'incremental CV R2 of fate one-hot beyond spline', PURP))):
    q = DS[sel]; m = q.value.mean(); ax.bar(j, m, .6, color=col, alpha=.85)
    ax.plot([j - .28, j + .28], [q.null_mean.mean()] * 2, color='k', lw=1); ax.text(j, m + .03, f'{m:.2f}', ha='center', fontsize=6.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(['Pseudotime', 'Endocrine\nidentity'], fontsize=6.5); ax.set_ylim(0, .85)
ax.set_ylabel('Cross-validated R²'); save(fig, 'Fig3d_devstructure.png')
# e
ks = np.arange(1, 31)
def curve(sig):
    g = GP_[GP_.signal == sig].set_index('k').captured; return np.r_[0, g.loc[ks].to_numpy()], g.loc[30]
# e split (PI + internal review 2026-10-30, internal design note (not shipped)): e1 pseudotime, e2 identities; both k=0..10, shared y 0..1, same Pi(k) reference
for nm_out, sigs in (('Fig3e1_capture_pseudotime.png', [('tau', 'Pseudotime', GREEN, 1.3)]),
                     ('Fig3e2_capture_identity.png', [(f'fate_{f}|spline', f, FC[f], 1.1) for f in ('Alpha', 'Beta', 'Delta', 'Epsilon')])):
    fig, ax = plt.subplots(figsize=(3.0, 1.9)); ax.axvspan(0, 8, color='k', alpha=.06, lw=0)
    for sg, lb, cl, lw in sigs:
        y, _ = curve(sg); ax.plot(np.r_[0, ks][:11], y[:11], '-', color=cl, lw=lw, label=lb)
    y, _ = curve('Pi'); ax.plot(np.r_[0, ks][:11], y[:11], ':', color=GREY, lw=1.1, label='Pair-difference\nenergy (reference)')
    ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.set_xlabel('Conditioning depth k'); ax.set_ylabel('Fraction captured')
    ax.legend(frameon=False, fontsize=6.8, loc='upper left' if len(sigs) > 1 else 'lower right', ncol=2 if len(sigs) > 1 else 1, handlelength=1.2, columnspacing=.8, borderaxespad=.2); save(fig, nm_out)
# full 0..30 single panel -> SI
fig, ax = plt.subplots(figsize=(3.1, 2.0)); ax.axvspan(-.3, 8, color='k', alpha=.06, lw=0)
y, c30 = curve('tau'); ax.plot(np.r_[0, ks], y, '-', color=GREEN, lw=1.3, label='Pseudotime')
for f in ('Alpha', 'Beta', 'Delta', 'Epsilon'):
    y, c30 = curve(f'fate_{f}|spline'); ax.plot(np.r_[0, ks], y, '-', color=FC[f], lw=.9, label=f)
y, _ = curve('Pi'); ax.plot(np.r_[0, ks], y, ':', color=GREY, lw=1.1, label='Overall energy')
ax.set_xlim(-.3, 30.3); ax.set_ylim(-.02, 1.02); ax.set_xlabel('Conditioning depth k'); ax.set_ylabel('Fraction captured')
ax.legend(frameon=False, fontsize=5.2, loc='lower right', bbox_to_anchor=(1.0, .0), ncol=2, handlelength=1.0, columnspacing=.6, handletextpad=.4, borderaxespad=.2); save(fig, 'SI/FigS3e_capture_0to30.png')
# f top
MB = FB + 'depth_multi_B9999/'
DM0 = pd.concat([pd.read_csv(MB + f'{s}/depth_multi.tsv', sep='\t') for s in ('panc', 'heart', 'bm')]); DM = DM0[(DM0.metric == 'AUROC') & (DM0.direction == 'ALL')]
EPC = [('PANC_RFX6', 'Pancreas RFX6', '#7570b3'), ('HEART_TBX5', 'Heart TBX5', '#d95f02'), ('HEART_NKX2-5', 'Heart NKX2-5', '#e7298a')]
os.makedirs(O + 'SI', exist_ok=True)
for eps, KM, nm_out, leg in ((EPC[:1], 16, 'Fig3f_PANC_depth_curve.png', False), (EPC, 30, 'SI/FigS3f_depth_curves_all_0to30.png', True)):   # PI 2026-10-30: Main = PANC 0..16; all systems 0..30 -> SI
    K = np.arange(KM + 1); fig, ax = plt.subplots(figsize=(3.0, 1.9)); ax.axvspan(0, 8, color='k', alpha=.06, lw=0)
    for ep, nm, c in eps:
        q = DM[DM.endpoint == ep].set_index('subset'); v = np.array([q.loc[f'cum0-{k}', 'value'] for k in K])
        ax.plot(K, v, '-', color=c, lw=1.2, label=nm); ax.plot(8, v[8], 'D', color=c, ms=4.2)
    ax.axhline(.5, color='k', lw=.5, ls=':'); ax.set_xlim(0, KM); ax.set_ylim(.50, .60); ax.set_xticks(range(0, KM + 1, 4)); ax.set_xlabel('Conditioning depth k'); ax.set_ylabel('AUROC')
    if leg: ax.legend(frameon=False, fontsize=5.8, loc='upper right')
    save(fig, nm_out)
# f bottom (forest: 3 systems x ALL/KO-up/KO-down, production - k0; Holm over 3 systems x 4 control contrasts within direction)
CT_ = pd.read_csv(FB + 'full/contrasts_all_directions.tsv', sep='\t'); CT_ = CT_[(CT_.contrast == 'D-K0') & (CT_.metric == 'AUROC')]
rows = [(s, d) for s in ('PANC', 'HEART', 'BM') for d in ('ALL', 'KOup', 'KOdown')]; COL = {'PANC': '#7570b3', 'HEART': '#d95f02', 'BM': '#999999'}
SN = {'PANC': 'Pancreas', 'HEART': 'Heart', 'BM': 'Bone marrow'}; DN = {'ALL': 'all', 'KOup': 'KO-up', 'KOdown': 'KO-down'}
fig, ax = plt.subplots(figsize=(2.6, 2.0))
for i, (s, d) in enumerate(rows):
    r = CT_[(CT_.level == s) & (CT_.direction == d)].iloc[0]
    ax.errorbar(r.value, i, xerr=[[r.value - r.lo], [r.hi - r.value]], fmt='o', color=COL[s], ms=3.2, capsize=1.6, lw=1, mfc=COL[s] if d == 'ALL' else 'white')
    ax.text(1.02, i, f'{r.p_holm12_system:.2g}' if np.isfinite(r.p_holm12_system) else '', transform=ax.get_yaxis_transform(), va='center', fontsize=5.8)
ax.text(1.02, -1.1, 'P$_{Holm}$', transform=ax.get_yaxis_transform(), va='center', fontsize=5.8)
ax.set_yticks(range(len(rows))); ax.set_yticklabels([f'{SN[s]} · {DN[d]}' for s, d in rows], fontsize=6); ax.invert_yaxis()
ax.axvline(0, color='k', lw=.6, ls=':'); ax.set_xlabel('ΔAUROC (GeoBRIDGE − k=0)'); save(fig, 'SI/FigS3f_forest_prod_minus_k0.png')
print('saved', O)
