"""R2-fix T1(A): GeoBRIDGE unconditioned (k=0) vs production (window K=8) on SERGIO bifurcation. CACHED-ONLY evaluation.
Source: ablation_2026_10_27/full/abl_g*_bifurcation_C*_s*.tsv rows op=R1,ch=U,kind=window,K in {0,8}. Production must equal TABLE.tsv GB (asserted).
usage (server): python script OUT"""
import sys, os, glob, json, hashlib, itertools
import numpy as np, pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/results/sergio_tf32rep_2026_10_22'); OUT = sys.argv[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
fs = sorted(glob.glob(f'{R}/ablation_2026_10_27/full/abl_g*_bifurcation_C*_s*.tsv')); assert len(fs) == 48, len(fs)
A = pd.concat([pd.read_csv(f, sep='\t') for f in fs]); A = A[(A.op == 'R1') & (A.ch == 'U') & (A.kind == 'window') & A.K.isin([0, 8])]
MET = ['AP', 'AUROC', 'DvI2']; key = ['grn', 'cond', 'sim']
P = A[A.K == 8].set_index(key)[MET]; Z = A[A.K == 0].set_index(key)[MET]; assert len(P) == len(Z) == 48
T = pd.read_csv(f'{R}/TABLE.tsv', sep='\t'); Tg = T[(T.topo == 'bifurcation') & (T.method == 'GB')].set_index(key)[MET]; assert len(Tg) == 48
dmax = float((P.sort_index() - Tg.sort_index()).abs().max().max()); print('production(abl K=8) vs TABLE GB max|diff| =', dmax); assert dmax < 1e-9
# k0 table in TABLE-like long format (for the figure script)
K0 = Z.reset_index(); K0['method'] = 'GBk0'; K0['topo'] = 'bifurcation'; K0.to_csv(OUT + 'K0_TABLE_bifurcation.tsv', sep='\t', index=False)
Pg = P.groupby(['grn', 'cond']).mean(); Zg = Z.groupby(['grn', 'cond']).mean()      # sims -> GRN
anch = {'AP': {'C0': (.362, .248), 'C1': (.303, None), 'C2': (.288, None), 'C3': (.211, .119)}, 'DvI2': {'C0': (.734, .657), 'C3': (.666, .554)}}
for m, d in anch.items():
    for c, (p_, z_) in d.items():
        assert abs(Pg.loc[(slice(None), c), m].mean() - p_) < 6e-4, (m, c, 'prod'); 
        if z_: assert abs(Zg.loc[(slice(None), c), m].mean() - z_) < 6e-4, (m, c, 'k0')
print('parity anchors OK')
# comparator means (same condition) from TABLE
Tc = T[T.topo == 'bifurcation'].groupby(['grn', 'cond', 'method'])[MET].mean().reset_index()
cm = Tc.groupby(['cond', 'method'])[MET].mean().unstack('method')
rows = []
for m in MET:
    pr = []
    for c in ['C0', 'C1', 'C2', 'C3']:
        p = Pg.xs(c, level='cond')[m]; z = Zg.xs(c, level='cond')[m]; d = (p - z).values; n = len(d)
        h = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
        signs = np.array(list(itertools.product([-1, 1], repeat=n))); pf = float((np.abs((signs * d).mean(1)) >= abs(d.mean()) - 1e-15).mean())
        rows.append(dict(metric=m, cond=c, mean_k0=z.mean(), mean_prod=p.mean(), diff=d.mean(), ci_lo=d.mean() - h, ci_hi=d.mean() + h, p_t=stats.ttest_1samp(d, 0).pvalue, p_signflip=pf, n_improved=int((d > 0).sum()), n_grn=n))
    r = pd.DataFrame(rows[-4:]); r['p_t_holm4'] = multipletests(r.p_t, method='holm')[1]; r['p_flip_holm4'] = multipletests(r.p_signflip, method='holm')[1]; rows[-4:] = r.to_dict('records')
R_ = pd.DataFrame(rows); R_.to_csv(OUT + 'T1A_k0_vs_prod.tsv', sep='\t', index=False)
pd.set_option('display.width', 250); print(R_.round(4).to_string())
print('\ncomparator means (bifurcation, mean over 6 GRNs), k0 shown alongside:')
for m in MET:
    t = cm[m].copy(); t['GBk0'] = Zg[m].groupby('cond').mean(); print(m); print(t.round(3).to_string())
json.dump(dict(inputs={os.path.basename(f): hashlib.sha256(open(f, 'rb').read()).hexdigest() for f in fs}, table_sha=hashlib.sha256(open(f'{R}/TABLE.tsv', 'rb').read()).hexdigest(), parity_max=dmax), open(OUT + 'receipt_T1A.json', 'w'), indent=1)
