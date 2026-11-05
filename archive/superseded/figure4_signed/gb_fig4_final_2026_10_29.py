"""Fig 4 final candidate (PI 2026-10-29: 'cell-program polarity + residual binding effect'). Plotting from cached, audited outputs only.
a  L1-L5 schematic (as the PI-approved rendering scratch/gb_fig4_render_v6.py).
b  heart TBX5, NKX2-5 + pancreas RFX6: class means with GENE-level 95% CI from the frozen kochip_audit_2026_10_15/p1_levels.tsv (descriptive; same as the
   PI-approved figure). No gene-level p-values (directive: not reused). Inference printed per TF: module-level ordered-trend slope (null-imposed, Holm-3;
   fig4b_trend.tsv, B=9,999) and the slope after conditioning on the truth-free cell-type specificity axis (celltype_axis_trend.tsv, B=2,000).
c  cell-program polarity: signed score T vs truth-free cell-type specificity (heart: endothelial-vs-cardiomyocyte, markers fixed in
   gb_fig4_endo_sensitivity_2026_10_29.py; pancreas: endocrine-vs-ductal by fixture clusters), genes coloured by L-class; Spearman in non-responders.
usage (server, repo root): script OUTDIR"""
import sys, os
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); sys.path[:0] = [R + 'scratch', R + 'pairspace']
import numpy as np, pandas as pd, matplotlib, anndata as ad, scipy.sparse as sp, gb_substrate as GS
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.stats import spearmanr
O = sys.argv[1].rstrip('/') + '/'; os.makedirs(O, exist_ok=True)
LV = pd.read_csv(R + 'results/signed/kochip_audit_2026_10_15/p1_levels.tsv', sep='\t')
TR = pd.read_csv(R + 'manuscript/figures_2026_10_28/fig4_dvi/style_2026_10_29/fig4b_trend.tsv', sep='\t').set_index('TF')
CA = pd.read_csv(R + 'results/signed/fig4_celltype_axis_2026_10_29/B9999c/celltype_axis_trend.tsv', sep='\t')
HS = pd.read_csv(R + 'results/signed/fig4_endo_2026_10_29/smallN_B200/heart_gene_specificity.tsv', sep='\t'); SPEC = {'HEART': dict(zip(HS.gene.str.upper(), HS.s_EC_vs_CM))}
fx, _, _ = GS.paths('PANC', canonical=True); F = ad.read_h5ad(fx); X = F.X.tocsr() if sp.issparse(F.X) else sp.csr_matrix(F.X)
Xn = sp.diags(1e4 / np.maximum(np.asarray(X.sum(1)).ravel(), 1)) @ X; lab = F.obs['clusters'].astype(str).to_numpy()
en, du = np.isin(lab, ['Alpha', 'Beta', 'Delta', 'Epsilon']), lab == 'Ductal'
SPEC['PANC'] = dict(zip(np.char.upper(np.asarray(F.var_names).astype(str)), np.log2((np.asarray(Xn[en].mean(0)).ravel() + .01) / (np.asarray(Xn[du].mean(0)).ravel() + .01))))
RUN = R + 'results/signed/fig4_bind_prod_k0_2026_10_29/v2_full_B2000/'
UNITS = [('CARDIACV_TBX5', 'TBX5', 'Heart · TBX5', 'HEART'), ('CARDIACV_NKX2-5', 'NKX2-5', 'Heart · NKX2-5', 'HEART'), ('PANC_RFX6', 'RFX6', 'Pancreas · RFX6', 'PANC')]
LC = ['#c0392b', '#e8a39a', '#9aa0a6', '#9bb8d9', '#2c5d9e']; LTX = ['w', 'k', 'k', 'k', 'w']; GREY = '#8a8f94'
plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans', 'pdf.fonttype': 42})
fig = plt.figure(figsize=(7.4, 9.0))
gA = fig.add_gridspec(1, 1, left=.075, right=.985, top=.985, bottom=.865); gB = fig.add_gridspec(1, 3, left=.09, right=.985, top=.77, bottom=.50, wspace=.32)
gC = fig.add_gridspec(1, 3, left=.09, right=.985, top=.36, bottom=.09, wspace=.32)
# a
ax = fig.add_subplot(gA[0]); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis('off'); fig.text(.012, .965, 'a', fontsize=11, fontweight='bold')
cw, ch, gx, gy0 = 19.5, 7.2, 12, 7.0; xs = [gx, gx + cw + 1, gx + 2 * (cw + 1)]
def cell(x, y, h, i, t):
    ax.add_patch(FancyBboxPatch((x, y), cw, h, boxstyle='round,pad=0.15,rounding_size=0.8', fc=LC[i], ec='#444', lw=.6))
    ax.text(x + cw / 2, y + h / 2 + 1.0, f'L{i+1}', ha='center', va='center', fontsize=8, fontweight='bold', color=LTX[i]); ax.text(x + cw / 2, y + h / 2 - 1.6, t, ha='center', va='center', fontsize=5.6, color=LTX[i])
cell(xs[0], gy0 + ch + 1, ch, 0, 'KO-down, bound'); cell(xs[0], gy0, ch, 1, 'KO-down, not bound'); cell(xs[1], gy0, 2 * ch + 1, 2, 'non-responder (bound or not)')
cell(xs[2], gy0, ch, 3, 'KO-up, not bound'); cell(xs[2], gy0 + ch + 1, ch, 4, 'KO-up, bound')
for k_, t in enumerate(['KO-down', 'non-responder', 'KO-up']): ax.text(xs[k_] + cw / 2, gy0 + 2 * ch + 3.4, t, ha='center', va='center', fontsize=6.5, fontweight='bold')
ax.text(gx - 1.5, gy0 + ch * 1.5 + 1, 'ChIP-bound', ha='right', va='center', fontsize=6.3); ax.text(gx - 1.5, gy0 + ch * .5, 'not bound', ha='right', va='center', fontsize=6.3)
ax.text(xs[2] + cw + 2, gy0 + ch + .5, 'expected T:\nL1 > L2 > L3 > L4 > L5\n\nChIP: evaluation only;\nbinding \u2260 direct\nregulation.\nT = sign(iso)\u00b7F(Z)', ha='left', va='center', fontsize=5.6, color='#333')
# b
fig.text(.012, .785, 'b', fontsize=11, fontweight='bold')
for k, (u, tf, disp, sysn) in enumerate(UNITS):
    a = fig.add_subplot(gB[k]); d = LV[LV.unit == disp].reset_index(drop=True); assert len(d) == 5, disp
    for i in range(5): a.errorbar(i + 1, d['mean'][i], yerr=[[d['mean'][i] - d.mean_lo[i]], [d.mean_hi[i] - d['mean'][i]]], fmt='o', color=LC[i], capsize=2, ms=4, lw=1.1)
    a.axhline(0, color=GREY, lw=.5, ls='--'); a.set_xlim(.4, 5.6); a.set_ylim(-1, 1); a.set_xticks(range(1, 6)); a.set_xticklabels([f'L{i+1}\n{int(d.n[i])}' for i in range(5)], fontsize=5.6)
    t = TR.loc[tf]; c = CA[(CA.TF == tf) & (CA.adjustment == 'within s-deciles')].iloc[0]
    a.set_title(tf, fontsize=8, fontweight='bold', pad=40)
    a.text(.5, 1.13, f'ordered trend {t.slope:.2f} [{t.lo:.2f}, {t.hi:.2f}]\nHolm-3 p={t.p_holm3:.2g} (module-level)', transform=a.transAxes, ha='center', va='center', fontsize=5.3)
    a.text(.5, 1.03, f'within cell-type strata: {c.slope:.2f} [{c.lo:.2f}, {c.hi:.2f}], nominal p={c.p_null:.2g}', transform=a.transAxes, ha='center', va='center', fontsize=5.1, color='#555')
    if k == 0: a.set_ylabel('mean T (signed percentile), 95% CI', fontsize=6.5)
for lab_, ks in (('heart', (0, 1)), ('pancreas', (2,))):
    pass
# c
fig.text(.012, .375, 'c', fontsize=11, fontweight='bold')
for k, (u, tf, disp, sysn) in enumerate(UNITS):
    a = fig.add_subplot(gC[k]); D = pd.read_csv(RUN + f'genes_{u}.tsv', sep='\t'); D['s'] = D.gene.str.upper().map(SPEC[sysn]); D = D[np.isfinite(D.s)]
    # binned display (PI 2026-10-29: fewer points): deciles of the cell-type axis over all genes; mean T per decile for KO-down (L1+L2), non-responders (L3), KO-up (L4+L5)
    edges = np.quantile(D.s, np.linspace(0, 1, 11)); D['bin'] = np.clip(np.searchsorted(edges, D.s, 'right') - 1, 0, 9); xc = D.groupby('bin').s.median()
    for nm, ls_, col, lw in (('KO-down (L1+L2)', ['L1', 'L2'], LC[0], 1.8), ('non-responder (L3)', ['L3'], GREY, 1.2), ('KO-up (L4+L5)', ['L4', 'L5'], LC[4], 1.8)):
        g = D[D.level.isin(ls_)].groupby('bin').agg(m=('T_prod', 'mean'), n=('T_prod', 'size')); g = g[g.n >= 3]
        a.plot(xc.loc[g.index], g.m, '-o', color=col, lw=lw, ms=3, label=nm)
    rho = spearmanr(D.T_prod[D.level == 'L3'], D.s[D.level == 'L3'])[0]
    a.axhline(0, color=GREY, lw=.5, ls='--'); a.set_ylim(-1.0, 1.0)
    a.set_xlabel('log2 endothelial / cardiomyocyte expression\n(← cardiomyocyte program   non-myocyte →)' if sysn == 'HEART' else 'log2 endocrine / ductal expression\n(← ductal   endocrine →)', fontsize=5.8)
    a.set_title(f'{tf}: ρ(T, axis) = {rho:.2f} in non-responders', fontsize=6.3)
    if k == 0: a.set_ylabel('mean signed score T per axis decile', fontsize=6.5); a.legend(frameon=False, fontsize=5.4, loc='lower left')
fig.text(.09, .035, 'b: points = class means with gene-bootstrap 95% CI (descriptive; genes are not independent). Inference = ordered-trend slope of T on class score (L1..L5 = +2..−2),\n'
         'null-imposed bootstrap over 50 control modules, Holm over 3 TFs; second line = same slope within deciles of the truth-free cell-type axis in c (centered module\n'
         'bootstrap, 9,999 draws, nominal, not multiplicity-adjusted). c: observational program-axis association — mean T per decile of a truth-free cell-type axis\n'
         '(bins with ≥3 genes). In heart, scores of responders and non-responders follow the cardiomyocyte vs non-myocyte axis; the pancreas score does not.\n'
         'The axis cannot distinguish cell-intrinsic regulation from composition.', fontsize=5.4, va='top')
for ext in ('png', 'pdf'): fig.savefig(O + f'Fig4_final_candidate.{ext}', dpi=300, bbox_inches='tight')
# FINAL Main (internal review 2026-10-30: no Main 4c; program-axis panel -> SI): a/b only, legend without c
for ax_ in fig.axes[4:]: ax_.remove()
for t_ in list(fig.texts):
    if t_.get_text() in ('c',) or t_.get_text().startswith('b: points'): t_.remove()
fig.text(.09, .46, 'b: points = class means with gene-bootstrap 95% CI (descriptive; genes are not independent). Inference = ordered-trend slope of T on class score\n'
         '(L1..L5 = +2..−2), null-imposed bootstrap over 50 control modules, Holm over 3 TFs. Second line = the same slope within deciles of a truth-free cell-type axis\n'
         '(heart: endothelial vs cardiomyocyte; pancreas: endocrine vs ductal; centered module bootstrap, nominal; Supplementary Fig. X).', fontsize=5.4, va='top')
for ext in ('png', 'pdf'): fig.savefig(O + f'Fig4_final_ab.{ext}', dpi=300, bbox_inches='tight')
print('saved', O)
