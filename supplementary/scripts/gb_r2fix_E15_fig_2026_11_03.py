# Plot/summary only: merges E15 tables (run on servers) -> summary TSV + PNGs. usage: script DIR (containing class_means_*.tsv etc.)
import sys, glob, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
D = sys.argv[1].rstrip('/') + '/'; cat = lambda n: pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob.glob(D + n + '_*.tsv'))], ignore_index=True)
CM, TR, CO, AG = cat('class_means'), cat('trend'), cat('contrasts'), cat('sign_agree'); TFS = ['TBX5', 'NKX2-5', 'RFX6']
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.
    for r, i in enumerate(o): run = max(run, min(1., (m - r) * p[i])); adj[i] = run
    return adj
TR['p_holm3'] = np.nan
for c in TR.cand.unique(): k = TR.cand == c; TR.loc[k, 'p_holm3'] = holm(TR.p[k])
dn = CO[CO.contrast == 'dT_down'].set_index(['TF', 'cand']); up = CO[CO.contrast == 'dT_up'].set_index(['TF', 'cand'])
S = TR.set_index(['TF', 'cand']).join(AG.set_index(['TF', 'cand']))
for nm, d in (('dT_down', dn), ('dT_up', up)):
    for c in ('diff', 'lo', 'hi', 'p'): S[f'{nm}_{c if c != "diff" else "val"}'] = d[c]
S = S.reset_index().rename(columns={'slope': 'trend_slope', 'lo': 'trend_lo', 'hi': 'trend_hi', 'p': 'trend_p'}); S.to_csv(D + 'E15_summary.tsv', sep='\t', index=False)
CM.to_csv(D + 'E15_class_means_all.tsv', sep='\t', index=False)
COL = {'L1': '#a50f15', 'L2': '#fb9a99', 'L3': '#9a9a9a', 'L4': '#a6c8e8', 'L5': '#08519c'}; LV = list(COL); NM = {'L1': 'D+', 'L2': 'D-', 'L3': 'N', 'L4': 'U-', 'L5': 'U+'}
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 6.5, 'axes.linewidth': .7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': .5, 'axes.axisbelow': True})
ALL = list(TR.cand.unique()); MAIN = [c for c in ['I-k0(adopted)'] + [x for x in ALL if x.startswith('I-mean')] + [x for x in ALL if x.startswith('F-mean0..8') and '2/3' not in x] + ['F-k0', 'F-cen(d4)', 'Pearson Ms-Mu'] if c in ALL]
def grid(cands, fn):
    f, ax = plt.subplots(len(cands), 3, figsize=(7.2, 1.25 * len(cands) + .3), sharey=True, squeeze=False)
    for i, c in enumerate(cands):
        for j, tf in enumerate(TFS):
            a = ax[i, j]; d = CM[(CM.TF == tf) & (CM.cand == c)].set_index('level').loc[LV]
            for x, l in enumerate(LV): a.errorbar(x, d['mean'][l], yerr=[[d['mean'][l] - d.lo[l]], [d.hi[l] - d['mean'][l]]], fmt='o', color=COL[l], ms=3.5, lw=1, capsize=0)
            a.set_ylim(-.7, 1.15); a.axhline(0, color='#888', lw=.6); a.set_xticks(range(5)); a.set_xticklabels([NM[l] for l in LV]); a.set_xlim(-.6, 4.6)
            r = S[(S.TF == tf) & (S.cand == c)].iloc[0]
            a.text(.02, .97, f'trend {r.trend_slope:.3f}, Holm p={r.p_holm3:.3g}\ndT_dn {r.dT_down_val:+.2f} (p={r.dT_down_p:.2g}); dT_up {r.dT_up_val:+.2f} (p={r.dT_up_p:.2g})', transform=a.transAxes, va='top', fontsize=5, color='#222')
            if i == 0: a.set_title(tf, fontweight='bold')
            if j == 0: a.set_ylabel(c, fontsize=6)
    f.supylabel('mean T = sign x F', fontsize=7); f.tight_layout(); f.savefig(D + fn, dpi=300, bbox_inches='tight'); plt.close(f)
grid(MAIN, 'E15_fig4_by_sign.png'); grid([c for c in ALL if c not in MAIN], 'E15_fig4_by_sign_SI.png')
order = ALL; f, a = plt.subplots(figsize=(3.6, 0.2 * len(order) + .8)); mk = dict(zip(TFS, 'osD')); cc = dict(zip(TFS, ['#2a78d6', '#eb6834', '#1baf7a']))
for tf in TFS:
    d = S[S.TF == tf].set_index('cand').loc[order]; y = np.arange(len(order)) + (TFS.index(tf) - 1) * .2
    a.errorbar(d.trend_slope, y, xerr=[d.trend_slope - d.trend_lo, d.trend_hi - d.trend_slope], fmt=mk[tf], color=cc[tf], ms=3, lw=.8, label=tf, mfc=[cc[tf] if p < .05 else 'white' for p in d.p_holm3] if False else cc[tf])
a.axvline(0, color='#888', lw=.6); a.set_yticks(range(len(order))); a.set_yticklabels(order); a.invert_yaxis(); a.set_xlabel('ordered-trend slope (T vs L1..L5 score)'); a.legend(frameon=False, fontsize=6)
f.savefig(D + 'E15_trend_dotplot.png', dpi=300, bbox_inches='tight')
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 200)
print(S[['TF', 'cand', 'trend_slope', 'trend_p', 'p_holm3', 'dT_down_val', 'dT_down_p', 'dT_up_val', 'dT_up_p', 'agree_down', 'agree_up', 'BA']].round(3).to_string(index=False))
