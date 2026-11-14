# Figure 4 class schematic and score summaries.
import sys, os
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); a_ = sys.argv; O = a_[1].rstrip('/') + '/'; os.makedirs(O, exist_ok=True); W = float(a_[a_.index('--w') + 1]) if '--w' in a_ else 6.2
LV = pd.read_csv(R + 'results/signed/kochip_audit_2026_10_15/p1_levels.tsv', sep='\t')
TR = pd.read_csv(R + 'manuscript/figures_2026_10_28/fig4_dvi/style_2026_10_29/fig4b_trend.tsv', sep='\t').set_index('TF')
CP = pd.read_csv(R + 'results/signed/kochip_audit_2026_10_15/fig19_contrast_pvalues.tsv', sep='	')
UNITS = [('TBX5', 'Heart · TBX5'), ('NKX2-5', 'Heart · NKX2-5'), ('RFX6', 'Pancreas · RFX6')]
LC = ['#c0392b', '#e8a39a', '#9aa0a6', '#9bb8d9', '#2c5d9e']; LTX = ['w', 'k', 'k', 'k', 'w']; GREY = '#8a8f94'
TICK = ['D+', 'D−', 'N', 'U−', 'U+']
plt.rcParams.update({'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans', 'axes.linewidth': .6})
def fmtp(p): return f'{p:.2g}'
fig = plt.figure(figsize=(W, W * 30 / 100 * .62)); ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis('off')
cw, ch, gx, gy0 = 27, 7.2, 13, 4.0; xs = [gx, gx + cw + 1.5, gx + 2 * (cw + 1.5)]
def cell(x, y, h, i, t):
    ax.add_patch(FancyBboxPatch((x, y), cw, h, boxstyle='round,pad=0.15,rounding_size=0.8', fc=LC[i], ec='#444', lw=.6))
    ax.text(x + cw / 2, y + h / 2, t, ha='center', va='center', fontsize=10, fontweight='bold', color=LTX[i])
cell(xs[0], gy0 + ch + 1, ch, 0, 'D+'); cell(xs[0], gy0, ch, 1, 'D−'); cell(xs[1], gy0, 2 * ch + 1, 2, 'N')
cell(xs[2], gy0, ch, 3, 'U−'); cell(xs[2], gy0 + ch + 1, ch, 4, 'U+')
for k_, t in enumerate(['Downregulated after KO', 'Non-responsive', 'Upregulated after KO']): ax.text(xs[k_] + cw / 2, gy0 + 2 * ch + 3.6, t, ha='center', va='center', fontsize=7, fontweight='bold')
ax.text(gx - 1.5, gy0 + ch * 1.5 + 1, 'ChIP-bound', ha='right', va='center', fontsize=6.5); ax.text(gx - 1.5, gy0 + ch * .5, 'no detected\nbinding', ha='right', va='center', fontsize=6.5)
ax.annotate('', xy=(xs[2] + cw, gy0 - 2.2), xytext=(xs[0], gy0 - 2.2), arrowprops=dict(arrowstyle='->', color='#333', lw=.8))
ax.text((xs[0] + xs[2] + cw) / 2, gy0 - 3.6, 'expected signed score T: decreasing', ha='center', va='top', fontsize=6.3, color='#333')
fig.savefig(O + 'Fig4a.png', dpi=300, bbox_inches='tight'); plt.close(fig)
# b
fig, axs = plt.subplots(1, 3, figsize=(W, 2.0), gridspec_kw=dict(wspace=.28))
for k, (tf, disp) in enumerate(UNITS):
    a = axs[k]; d = LV[LV.unit == disp].reset_index(drop=True); assert len(d) == 5, disp
    for i in range(5): a.errorbar(i + 1, d['mean'][i], yerr=[[d['mean'][i] - d.mean_lo[i]], [d.mean_hi[i] - d['mean'][i]]], fmt='o', color=LC[i], capsize=1.8, ms=3.4, lw=1)
    a.axhline(0, color=GREY, lw=.5, ls='--'); a.set_xlim(.4, 5.6); a.set_ylim(-1, 1.1); a.set_yticks([-1, -.5, 0, .5, 1])
    a.set_xticks(range(1, 6)); a.set_xticklabels([f'{TICK[i]}\n{int(d.n[i])}' for i in range(5)], fontsize=6.4)
    C_ = CP[(CP.unit == disp) & (CP.score == 'iso')].set_index('contrast')
    for j, (lab, y) in enumerate((('L1>L2', .97), ('L2>L3', .84), ('L3>L4', .71), ('L4>L5', .58))):
        h = C_.loc[lab, 'p_holm6']; st = '***' if h < 1e-3 else '**' if h < 1e-2 else '*' if h < .05 else 'n.s.'; col = 'k' if h < .05 else '#8a8f94'
        a.plot([j + 1, j + 2], [y, y], color=col, lw=.8)
        a.text(j + 1.5, y + .015, f"{st} {('%.1e' % h).replace('e-0', 'e-') if h < 1e-3 else ('%.3f' % h if h < .1 else '%.2f' % h)}", ha='center', va='bottom', fontsize=4.8, color=col)
    a.text(.98, .02, f'Trend P$_{{Holm}}$ = {TR.loc[tf].p_holm3:.2g}', transform=a.transAxes, ha='right', va='bottom', fontsize=5.4)
    if k == 0: a.set_ylabel('Mean signed score T', fontsize=6.8)
fig.savefig(O + 'Fig4b.png', dpi=300, bbox_inches='tight'); plt.close(fig)
print('saved', O, TR[['slope', 'lo', 'hi', 'p_holm3']].to_string())
