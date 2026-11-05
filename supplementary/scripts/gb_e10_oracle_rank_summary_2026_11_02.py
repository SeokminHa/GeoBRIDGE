"""E10 rank-audit summary: asserts complete keys (96 datasets x 4 arms, no dups), then GRN-level summaries."""
import glob, json, numpy as np, pandas as pd
from scipy import stats
D = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/results/signed/r2fix_2026_11_01/E10_oracle/rank_audit/')
R = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob.glob(D + 'part_*/datasets.tsv'))]); I = pd.concat([pd.read_csv(f, sep='\t') for f in sorted(glob.glob(D + 'part_*/O_info.tsv'))])
rc = [json.load(open(f)) for f in sorted(glob.glob(D + 'part_*/receipt.json'))]
M = ['AP', 'DvI23_AUROC']; ARMS = ['GB_k0', 'GB_k08', 'ORACLE_RANK', 'ORACLE_RANK_Q08']
K = ['topo', 'grn', 'cond', 'sim', 'arm']
assert not R.duplicated(K).any(), 'duplicate keys'
exp = pd.MultiIndex.from_product([['bifurcation', 'tree'], range(501, 507), ['C0', 'C1', 'C2', 'C3'], [1, 2], ARMS], names=K)
assert len(R) == 96 * len(ARMS) and exp.difference(pd.MultiIndex.from_frame(R[K])).empty, 'incomplete keys'
assert not I.duplicated(['grn', 'topo', 'cond', 'sim', 'root']).any() and len(I) == 96 * 2
print('KEYS OK: 96 datasets x', len(ARMS), 'arms =', len(R), 'rows, no duplicates; max_parity', max(r['max_parity'] for r in rc), '(CONC<1e-4, TABLE AP<1e-6, prefix3<1e-9 asserted)')
G = R.groupby(['cond', 'grn', 'arm'])[M].mean().reset_index()
def t95(d):
    d = np.asarray(d, float); n = len(d); m = d.mean(); h = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n); return m, m - h, m + h
out = []; pd.set_option('display.width', 250)
for m in M:
    W = G.pivot_table(index=['cond', 'grn'], columns='arm', values=m)[ARMS]
    print('\n==', m, 'GRN-mean'); print(W.groupby('cond').mean().round(4).to_string())
    for c in ['C1', 'C2', 'C3']:
        for a in ARMS[2:]:
            w = W.loc[c]; s = f'{c} {a:16s}'
            for ref in ('GB_k0', 'GB_k08'):
                d = (w[a] - w[ref]).values; mu, lo, hi = t95(d); s += f' | vs {ref} {mu:+.4f} [{lo:+.4f},{hi:+.4f}] {int((d > 0).sum())}/6'; out.append(dict(metric=m, cond=c, arm=a, ref=ref, diff=mu, lo=lo, hi=hi, npos=int((d > 0).sum())))
            print(s)
    gap = W.loc['C3'] - W.loc['C0']; print('C3-C0 gap'); 
    for a in ARMS: mu, lo, hi = t95(gap[a].values); print(f'  {a:16s} {mu:+.4f} [{lo:+.4f},{hi:+.4f}]'); out.append(dict(metric=m, cond='C3-C0', arm=a, diff=mu, lo=lo, hi=hi))
pd.DataFrame(out).to_csv(D + 'summary_rank.tsv', sep='\t', index=False)
print('\n== r* distribution per condition (per root)'); print(I.groupby('cond').rstar.describe()[['mean', 'min', '25%', '50%', '75%', 'max']].round(2).to_string())
print(I.groupby(['cond', 'rstar']).size().unstack(fill_value=0).to_string())
print('\n== energies'); print(I.groupby('cond')[['hidden_energy_in_O', 'residual_hidden_energy', 'pair_energy_dX_in_O']].mean().round(4).to_string())
