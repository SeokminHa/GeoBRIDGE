# Aggregate pair-window sensitivity by GRN and topology.
import sys, glob, numpy as np, pandas as pd
from scipy import stats
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
E = sys.argv[1].rstrip('/') + '/'; R = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob.glob(E + 'full_*/datasets.tsv'))])
R.to_csv(E + 'datasets_all.tsv', sep='\t', index=False); M = ['AP', 'DvI23_AUROC']; PROD = (.02, .25, 40)
assert len(R) == 96 * 23 * 2, len(R)
G = R.groupby(['cond', 'grn', 'lo', 'hi', 'cap', 'arm'])[M + ['n_pairs_min', 'n_pairs_max']].mean().reset_index()
G['n_pairs'] = (G.n_pairs_min + G.n_pairs_max) / 2
# conditioning gain as extra arm
w = G.pivot_table(index=['cond', 'grn', 'lo', 'hi', 'cap'], columns='arm', values=M); gain = (w.xs('GB_k08', axis=1, level=1) - w.xs('GB_k0', axis=1, level=1)).reset_index(); gain['arm'] = 'GAIN_k08-k0'
G = pd.concat([G, gain], ignore_index=True)
pk = G[(G.lo == PROD[0]) & (G.hi == PROD[1]) & (G.cap == PROD[2])].set_index(['cond', 'grn', 'arm'])[M]
out = []
for (cond, lo, hi, cap, arm), d in G.groupby(['cond', 'lo', 'hi', 'cap', 'arm']):
    d = d.set_index('grn'); b = pk.xs((cond,), level=0).reset_index().query('arm==@arm').set_index('grn')
    r = dict(cond=cond, lo=lo, hi=hi, cap=cap, arm=arm, n_pairs=d.n_pairs.mean() if arm != 'GAIN_k08-k0' else np.nan)
    for m in M:
        x = d[m]; df = (x - b[m].reindex(x.index)).to_numpy(); n = len(df); h = stats.t.ppf(.975, n - 1) * df.std(ddof=1) / np.sqrt(n)
        r.update({m: x.mean(), f'{m}_dprod': df.mean(), f'{m}_lo': df.mean() - h, f'{m}_hi': df.mean() + h, f'{m}_npos': int((df > 0).sum())})
    out.append(r)
S = pd.DataFrame(out); S.to_csv(E + 'summary.tsv', sep='\t', index=False)
fig, ax = plt.subplots(2, 4, figsize=(17, 7)); LOS = [0, .01, .02, .05]; HIS = [.10, .15, .25, .40, .60]
for j, (cond, m) in enumerate([(c, m) for c in ('C0', 'C3') for m in M]):
    for i, arm in enumerate(('GB_k08', 'GB_k0')):
        T = S[(S.cond == cond) & (S.arm == arm) & (S.cap == 40)].pivot(index='lo', columns='hi', values=m).reindex(index=LOS, columns=HIS)
        a = ax[i, j]; im = a.imshow(T.values, cmap='viridis', aspect='auto'); plt.colorbar(im, ax=a)
        for y in range(4):
            for x in range(5): a.text(x, y, f'{T.values[y, x]:.3f}', ha='center', va='center', color='w', fontsize=8)
        a.set_xticks(range(5)); a.set_xticklabels(HIS); a.set_yticks(range(4)); a.set_yticklabels(LOS); a.set_xlabel('upper'); a.set_ylabel('lower')
        a.add_patch(plt.Rectangle((HIS.index(.25) - .5, LOS.index(.02) - .5), 1, 1, fill=False, ec='red', lw=2.5)); a.set_title(f'{arm} {cond} {m}', fontsize=9)
plt.tight_layout(); plt.savefig(E + 'heatmap_lower_upper.png', dpi=130)
pd.set_option('display.width', 250)
print(S[(S.arm == 'GB_k08') & (S.cap == 40)].pivot_table(index=['lo', 'hi'], columns='cond', values=['AP', 'AP_dprod', 'DvI23_AUROC', 'DvI23_AUROC_dprod']).round(4).to_string())
print(S[(S.lo == .02) & (S.hi == .25)].pivot_table(index='cap', columns=['cond', 'arm'], values='AP').round(4).to_string())
