"""Fig 3d v8 (internal review scope review round 2, 2026-10-29): v7 with PANC worked-example labels on A/B, BM curve removed from C (forest only), responder counts in the forest, equal-TF heart mean wording, Π/pseudotime selectivity sentence.
A  Developmental structure in the PC embedding (cell space, devstructure/panc_devstructure.tsv): bar = mean of the two pseudotime roots, dots = each root.
B  Root-pooled RAW capture (geom30/geom30_pooled.tsv) C(k) = sum_r ||Q_k,r'Y_r||^2 / sum_r ||Y_r||^2 of pseudotime, each terminal endocrine identity
   (one-vs-rest, after the pseudotime spline) and the 30-PC pair-difference energy Pi(k). Ticks at right = C(30). Grey bar = production depths 0..8.
C  AUROC (RFX6 KO responders vs non-responders) of GeoBRIDGE aggregated over depths 0..K, K=0..30 (depth_deep/panc_depth_deep.tsv; K=8 = frozen production),
   single-depth points, spline reference; lower strip = paired difference vs k=0 on the synchronized draws.
usage (server, repo root): script OUTDIR"""
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); O = sys.argv[1].rstrip('/') + '/'; os.makedirs(O, exist_ok=True); FB = R + 'results/signed/fig3d_rebuild_2026_10_29/'
DS = pd.read_csv(FB + 'devstructure/panc_devstructure.tsv', sep='\t')
GP_ = pd.read_csv(FB + 'geom30/geom30_pooled.tsv', sep='\t')
DD = pd.read_csv(FB + 'depth_deep/panc_depth_deep.tsv', sep='\t'); DA = DD[DD.metric == 'AUROC'].set_index('subset'); DP = DD[DD.metric == 'AUPRC'].set_index('subset')
P = pd.read_csv(FB + 'full/depth_profiles_all.tsv', sep='\t'); P = P[(P.level == 'PANC_RFX6') & (P.direction == 'ALL') & (P.metric == 'AUROC')].set_index('arm')
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Nimbus Sans', 'Arial', 'DejaVu Sans'], 'font.size': 8.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'pdf.fonttype': 42})
GREEN, PURP, GREY = '#1b9e77', '#7570b3', '#7f7f7f'; FC = {'Alpha': '#e7298a', 'Beta': '#7570b3', 'Delta': '#e6ab02', 'Epsilon': '#66a61e'}
fig = plt.figure(figsize=(17, 6.6)); GSp = fig.add_gridspec(3, 3, width_ratios=[.7, 1.2, 1.3], height_ratios=[2.1, 1, 1], wspace=.36, hspace=.14)
fig = plt.figure(figsize=(17, 6.4)); GSp = fig.add_gridspec(2, 3, width_ratios=[.7, 1.2, 1.3], height_ratios=[1.45, 1], wspace=.36, hspace=.38)
axA, axB, axC = fig.add_subplot(GSp[:, 0]), fig.add_subplot(GSp[:, 1]), fig.add_subplot(GSp[0, 2]); axf = fig.add_subplot(GSp[1, 2])
for j, (sel, col) in enumerate(((DS.block == 'T_temporal', GREEN), (DS.quantity == 'incremental CV R2 of fate one-hot beyond spline', PURP))):
    q = DS[sel]; m = q.value.mean(); axA.bar(j, m, .55, color=col, alpha=.85); axA.plot([j - .12, j + .12], q.value, 'o', color='k', ms=3.5)
    axA.plot([j - .25, j + .25], [q.null_mean.mean()] * 2, color='k', lw=1.2); axA.text(j, m + .03, f'{m:.2f}', ha='center', fontsize=8)
axA.set_xticks([0, 1]); axA.set_xticklabels(['pseudotime\n(all cells)', 'terminal endocrine identity\nbeyond pseudotime\n(terminal cells)'], fontsize=7.6)
axA.set_ylim(-.03, .95); axA.axhline(0, color='k', lw=.5); axA.set_ylabel('CV R² of 30-PC variance'); axA.set_title('A  Developmental structure\n     (pancreas, worked example)', loc='left', fontweight='bold')
axA.text(.03, .98, 'bar = mean of two pseudotime roots, dots = roots\nblack line = permutation null; 5-fold cell-level CV\nlabels and pseudotime from the same transcriptome', transform=axA.transAxes, va='top', fontsize=6.4)
# ---- B
ks = np.arange(1, 31)
def curve(sig):
    g = GP_[GP_.signal == sig].set_index('k').captured; return np.r_[0, g.loc[ks].to_numpy()], g.loc[30]   # RAW capture (internal review r3)
axB.axvspan(-.3, 8, color='k', alpha=.05, lw=0); axB.text(4, .985, 'production depths 0–8', ha='center', fontsize=6.5, color=GREY)
y, c30 = curve('tau'); axB.plot(np.r_[0, ks], y, '-', color=GREEN, lw=2.2, label='pseudotime'); axB.plot([30.5, 31.5], [c30] * 2, '-', color=GREEN, lw=2.5)
for f in ('Beta', 'Alpha', 'Epsilon', 'Delta'):
    y, c30 = curve(f'fate_{f}|spline'); axB.plot(np.r_[0, ks], y, '-', color=FC[f], lw=1.6, label=f'{f} identity beyond pseudotime'); axB.plot([30.5, 31.5], [c30] * 2, '-', color=FC[f], lw=2.5)
y, _ = curve('Pi'); axB.plot(np.r_[0, ks], y, ':', color=GREY, lw=1.6, label='30-PC pair-difference energy Π(k) (reference)')
axB.set_xlim(-.3, 31.8); axB.set_ylim(-.02, 1.02); axB.set_xlabel('Conditioning depth k'); axB.set_ylabel('Fraction of each signal captured\nby the first k conditioning directions')
axB.set_title('B  What the conditioning directions capture\n     (pancreas, worked example)', loc='left', fontweight='bold'); axB.legend(frameon=False, fontsize=6.2, loc='upper right', bbox_to_anchor=(1, .715))
axB.text(0, -.1, 'C(k) = ||Q$_k^\\top$Y||²/||Y||² over production cell pairs, pooled over both pseudotime roots (energy-weighted).\n'
         'Ticks at right = C(30), the share of each signal inside the 30-PC pair space. Identity signals: one-vs-rest differences\n'
         'between annotated terminal endocrine cells after removing the pseudotime spline. Truth-free. Per-root and\n'
         'representable-share-normalised curves in SI (identity capture is weaker under root dc1_max). Within depths 0–8 pseudotime is\n'
         'captured in all three systems (0.74 pancreas, 0.95 heart, 0.76 bone marrow); only in pancreas is it\nconcentrated ahead of Π (k=1: 0.67 vs 0.14).',
         transform=axB.transAxes, fontsize=6.0, va='top')
# ---- C (internal review scope review 2026-10-29: all-system C in Main; PANC-only A/B as worked example)
MB = FB + 'depth_multi_B9999/'
DM0 = pd.concat([pd.read_csv(MB + f'{s}/depth_multi.tsv', sep='\t') for s in ('panc', 'heart', 'bm')]); DM = DM0[(DM0.metric == 'AUROC') & (DM0.direction == 'ALL')]
NP = DM0[(DM0.subset == 'cum0-0') & (DM0.metric == 'AUROC')].groupby(['endpoint', 'direction']).n_pos.first()
NSYS = {('PANC', d): int(NP[('PANC_RFX6', d)]) for d in ('ALL', 'KOup', 'KOdown')}; NSYS.update({('BM', d): int(NP[('BM_GATA1_acute', d)]) for d in ('ALL', 'KOup', 'KOdown')})
NSYS.update({('HEART', d): f"{int(NP[('HEART_TBX5', d)])}+{int(NP[('HEART_NKX2-5', d)])}" for d in ('ALL', 'KOup', 'KOdown')})
EPC = [('PANC_RFX6', 'pancreas RFX6', '#7570b3', '-'), ('HEART_TBX5', 'heart TBX5', '#d95f02', '-'), ('HEART_NKX2-5', 'heart NKX2-5', '#e7298a', '-')]   # BM curve -> SI (25 responders; internal review r2)
K = np.arange(31); axC.axvspan(-.3, 8, color='k', alpha=.05, lw=0)
for ep, nm, c, ls in EPC:
    q = DM[DM.endpoint == ep].set_index('subset'); v = np.array([q.loc[f'cum0-{k}', 'value'] for k in K])
    axC.plot(K, v, ls, color=c, lw=1.8 if ls == '-' else 1.2, label=nm); axC.plot(0, v[0], 'o', color=c, ms=4, mfc='white'); axC.plot(8, v[8], 'D', color=c, ms=6)
axC.axhline(.5, color='k', lw=.6, ls=':'); axC.set_xlim(-.3, 30.3); axC.set_ylim(.45, .72); axC.set_xlabel('Largest depth included in aggregate, K (depths 0..K)')
axC.set_ylabel('AUROC, all KO responders\nvs non-responders'); axC.legend(frameon=False, fontsize=6.6, loc='upper right', ncol=2)
axC.set_title('C  KO-response ranking vs conditioning depth (all systems)', loc='left', fontweight='bold'); axC.text(4, .705, 'production 0–8', ha='center', fontsize=6.3, color=GREY)
axC.text(0, -.36, '' if True else 'open circle = no conditioning (k=0); diamond = frozen production (depths 0..8, selected on synthetic data and frozen 2026-09-26); other K descriptive.',
         transform=axC.transAxes, fontsize=6.1)
CT_ = pd.read_csv(FB + 'full/contrasts_all_directions.tsv', sep='\t'); CT_ = CT_[(CT_.contrast == 'D-K0') & (CT_.metric == 'AUROC')]
rows = [(s, d) for s in ('PANC', 'HEART', 'BM') for d in ('ALL', 'KOup', 'KOdown')]
for i, (s, d) in enumerate(rows):
    r = CT_[(CT_.level == s) & (CT_.direction == d)].iloc[0]; col = {'PANC': '#7570b3', 'HEART': '#d95f02', 'BM': '#999999'}[s]
    axf.errorbar(r.value, i, xerr=[[r.value - r.lo], [r.hi - r.value]], fmt='o', color=col, ms=4.5, capsize=2, lw=1.2)
    axf.text(1.01, i, f'Holm p={r.p_holm12_system:.2g}' if np.isfinite(r.p_holm12_system) else '', transform=axf.get_yaxis_transform(), va='center', fontsize=6.2)
e = CT_[(CT_.level == 'EQSYS')].set_index('direction')
axf.set_yticks(range(len(rows))); axf.set_yticklabels([f"{ {'PANC': 'pancreas', 'HEART': 'heart', 'BM': 'bone marrow'}[s]} · {d}" + f' (n={NSYS[(s, d)]})' for s, d in rows], fontsize=6.6)
axf.invert_yaxis(); axf.axvline(0, color='k', lw=.7, ls=':'); axf.set_xlabel('Production − no conditioning (k=0): ΔAUROC, 95% module-bootstrap CI')
axf.text(0, -.42, f"Equal-system mean ΔAUROC: ALL {e.loc['ALL','value']:+.3f} (Holm-4 p={e.loc['ALL','p_holm4_EQSYS']:.2g}), KO-up {e.loc['KOup','value']:+.3f} ({e.loc['KOup','p_holm4_EQSYS']:.2g}), "
         f"KO-down {e.loc['KOdown','value']:+.3f} ({e.loc['KOdown','p_holm4_EQSYS']:.2g}).\nPer-system Holm over 3 systems × 4 control contrasts within each direction. Heart = equal-weight mean of TBX5 and NKX2-5 (n = TBX5+NKX2-5 responders). Conditioning helps KO-up heart genes and hurts KO-down heart genes (direction-dependent trade-off).\nCurves: open circle = k=0; diamond = frozen production (depths 0..8, selected on synthetic data, frozen 2026-09-26); other K descriptive.",
         transform=axf.transAxes, fontsize=6.1, va='top')
for f in ('png', 'pdf'): fig.savefig(O + f'Fig3d_v8.{f}', dpi=300, bbox_inches='tight')
print('saved', O)
