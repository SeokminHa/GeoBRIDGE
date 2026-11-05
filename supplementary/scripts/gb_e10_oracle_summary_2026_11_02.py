"""E10 summary: merge part_* of E10_oracle/full; GRN-level (sims, topologies averaged). usage: script"""
import glob, json, numpy as np, pandas as pd
from scipy import stats
D = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/results/signed/r2fix_2026_11_01/E10_oracle/full/')
R = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob.glob(D + 'part_*/datasets.tsv'))]); I = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob.glob(D + 'part_*/O_info.tsv'))])
rc = [json.load(open(f)) for f in sorted(glob.glob(D + 'part_*/receipt.json'))]
print('datasets', len(R), 'GRNs', sorted(R.grn.unique()), 'max_parity', max(r['max_parity'] for r in rc), '(CONC vs stored <1e-4, TABLE AP <1e-6, prefix3 <1e-9 all asserted)')
M = ['AP', 'DvI23_AUROC']; ARMS = ['GB_k0', 'GB_k08', 'ORACLE', 'ORACLE_Q08', 'FULL2', 'FULL2_Q08', 'FULL5', 'FULL5_Q08']
G = R.groupby(['cond', 'grn', 'arm'])[M].mean().reset_index()
def t95(d):
    d = np.asarray(d, float); n = len(d); m = d.mean(); h = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n); return m, m - h, m + h
out = []
for m in M:
    W = G.pivot_table(index=['cond', 'grn'], columns='arm', values=m)[ARMS]
    for c in ['C0', 'C1', 'C2', 'C3']:
        w = W.loc[c]
        for a in ARMS:
            row = dict(metric=m, cond=c, arm=a, mean=w[a].mean())
            for ref in ('GB_k0', 'GB_k08'):
                if a != ref: d = (w[a] - w[ref]).values; mu, lo, hi = t95(d); row.update({f'd_vs_{ref}': mu, f'lo_{ref}': lo, f'hi_{ref}': hi, f'npos_{ref}': int((d > 0).sum())})
            out.append(row)
    gap = (W.loc['C3'] - W.loc['C0'])
    for a in ARMS: mu, lo, hi = t95(gap[a].values); out.append(dict(metric=m, cond='C3-C0', arm=a, mean=mu, lo=lo, hi=hi))
S = pd.DataFrame(out); S.to_csv(D + 'summary_full.tsv', sep='\t', index=False)
pd.set_option('display.width', 300); pd.set_option('display.max_columns', 30)
for m in M:
    print('\n==', m, 'GRN-mean per condition'); print(S[(S.metric == m) & (S.cond != 'C3-C0')].pivot(index='cond', columns='arm', values='mean')[ARMS].round(4).to_string())
    print('-- paired arm - GB_k0 / GB_k08: mean [lo,hi] npos/6'); 
    for c in ['C1', 'C2', 'C3']:
        for a in ARMS[2:]:
            r = S[(S.metric == m) & (S.cond == c) & (S.arm == a)].iloc[0]
            print(f"{c} {a:11s} vsk0 {r.d_vs_GB_k0:+.4f} [{r.lo_GB_k0:+.4f},{r.hi_GB_k0:+.4f}] {int(r.npos_GB_k0)}/6 | vsk08 {r.d_vs_GB_k08:+.4f} [{r.lo_GB_k08:+.4f},{r.hi_GB_k08:+.4f}] {int(r.npos_GB_k08)}/6")
    print('-- C3 minus C0 gap per arm (mean [lo,hi])'); print(S[(S.metric == m) & (S.cond == 'C3-C0')].set_index('arm').loc[ARMS][['mean', 'lo', 'hi']].round(4).to_string())
I['label'] = I.amp.where(I.amp.str.startswith('FULL'), 'LINEAR'); print('\n== hidden_energy_in_O (mean over datasets/roots)'); print(I.groupby(['label', 'cond'])[['dimO', 'hidden_energy_in_O', 'pair_energy_dX_in_O']].mean().round(3).to_string())
