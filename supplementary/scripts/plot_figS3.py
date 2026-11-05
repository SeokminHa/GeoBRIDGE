"""Fig. S3: cell-pair window (lower x upper bound of the pseudotime difference, pair cap 40) and number of pairs per cell (cap 10/20/40/80, window 0.02-0.25).
GeoBRIDGE AUPRC, bifurcation benchmark, mean of 6 GRNs (95% t interval for the cap panels).
Reads data/figS3_pairwindow.tsv. usage: python plot_figS3.py --out OUTDIR [--data DATADIR]"""
import sys, os, argparse, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.stats.multitest import multipletests
ap = argparse.ArgumentParser(); ap.add_argument('--data', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')); ap.add_argument('--out', required=True)
args = ap.parse_args(); DATA = args.data.rstrip('/') + '/'; O = args.out.rstrip('/') + '/'; os.makedirs(O, exist_ok=True); BIF = 'bifurcation'
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['DejaVu Sans'], 'font.size': 7, 'axes.labelsize': 7, 'xtick.labelsize': 6.5, 'ytick.labelsize': 6.5,
 'legend.fontsize': 6.5, 'axes.linewidth': 0.7, 'lines.linewidth': 1.1, 'axes.edgecolor': '#888888', 'xtick.color': '#555555', 'ytick.color': '#555555',
 'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.5, 'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False})
M = [('GB', 'GeoBRIDGE (Ours)', '#2166ac'), ('VEL', 'Velorama', '#2ca02c'), ('fasttenet', 'FastTENET', '#17becf'), ('genie3', 'GENIE3', '#ff7f0e'), ('grnboost2', 'GRNBoost2', '#8c564b'),
     ('pidc', 'PIDC', '#9467bd'), ('regdiffusion', 'RegDiffusion', '#e377c2'), ('deepsem', 'DeepSEM', '#bcbd22')]   # = main Fig 2
CC = ['C0', 'C1', 'C2', 'C3']; CL = ['None', 'Temporal (A)', 'Branch (B)', 'A+B']; CP = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']; CS = ['none', 'A', 'B', 'AB']
CELL = (3.1, 2.4)
def t95(x): x = np.asarray(x, float); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)); return x.mean(), h
def stars(p): return '***' if p < .001 else '**' if p < .01 else '*' if p < .05 else 'ns'
def save(f, n): f.savefig(O + n, dpi=300, bbox_inches='tight', facecolor='white'); plt.close(f); print('wrote', n)
def rgb():   # RGB, no alpha
    from PIL import Image
    for n in os.listdir(O):
        if n.endswith('.png'): Image.open(O + n).convert('RGB').save(O + n, dpi=(300, 300))

# ---------------- S3 : pair window grid (cap 40) and pair cap (production window), GB k=0..8
pw = pd.read_csv(DATA + 'figS3_pairwindow.tsv', sep='\t'); pw = pw[(pw.topo == BIF) & (pw.arm == 'GB_k08')].groupby(['cond', 'grn', 'lo', 'hi', 'cap']).AP.mean().reset_index()
LO = [0.0, 0.01, 0.02, 0.05]; HI = [0.1, 0.15, 0.25, 0.4, 0.6]; rows = []; SUB = (1.9, 1.7)
gw = pw[pw.cap == 40]; Zs = {c: gw[gw.cond == c].groupby(['lo', 'hi']).AP.mean().unstack().loc[LO, HI].values for c in CC}
assert all(len(gw[gw.cond == c]) == 6 * 20 for c in CC)
vmin, vmax = min(np.nanmin(z) for z in Zs.values()), max(np.nanmax(z) for z in Zs.values()); rg = vmax - vmin
def heat(a, c, i):
    Z = Zs[c]; a.imshow(Z, cmap='Greys', aspect='auto', vmin=vmin - .2 * rg, vmax=vmax + 2.5 * rg)
    for y in range(4):
        for x in range(5): a.text(x, y, '%.2f' % Z[y, x], ha='center', va='center', fontsize=5.5)
    a.add_patch(plt.Rectangle((HI.index(.25) - .5, LO.index(.02) - .5), 1, 1, fill=False, ec='#eda100', lw=1.8))
    a.set_xticks(range(5)); a.set_xticklabels(HI, fontsize=5.5); a.set_yticks(range(4)); a.set_yticklabels(LO); a.grid(False)
    for sp in a.spines.values(): sp.set_visible(False)
    a.set_title(CL[i], fontsize=7); a.set_xlabel('Upper bound of Δpseudotime'); a.set_ylabel('Lower bound of Δpseudotime')
fw, axs = plt.subplots(1, 4, figsize=(SUB[0] * 4, SUB[1]), gridspec_kw={'wspace': .45})
for i, c in enumerate(CC):
    f, a = plt.subplots(figsize=SUB); heat(a, c, i); save(f, f'FigS3a_{CS[i]}.png'); heat(axs[i], c, i)
    rows += [dict(panel='S3a', cond=c, cap=40, lo=lo, hi=hi, metric='AP', mean=Zs[c][LO.index(lo), HI.index(hi)], n_grn=6) for lo in LO for hi in HI]
save(fw, 'FigS3a.png')
pm = pw[(pw.lo == .02) & (pw.hi == .25)]; caps = [10, 20, 40, 80]
for i, c in enumerate(CC):
    f, a = plt.subplots(figsize=(SUB[0] * 2, SUB[1])); d = pm[pm.cond == c]; mh = [t95(d[d.cap == k].AP) for k in caps]
    a.errorbar(range(4), [x[0] for x in mh], yerr=[x[1] for x in mh], color=CP[i], marker='o', ms=3, lw=1, capsize=2)
    a.set_xticks(range(4)); a.set_xticklabels(caps); a.set_xlabel('Pair cap per cell'); a.set_ylabel('AUPRC'); a.set_title(CL[i], fontsize=7)
    rows += [dict(panel='S3' + 'bcde'[i], cond=c, cap=k, lo=.02, hi=.25, metric='AP', mean=x[0], ci95_half=x[1], n_grn=6) for k, x in zip(caps, mh)]
    save(f, f'FigS3{"bcde"[i]}.png')
pd.DataFrame(rows).to_csv(O + 'FigS3_values.tsv', sep='\t', index=False)

rgb()
