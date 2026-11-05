"""R2-fix T1(B): within-regulator (macro) direct-vs-indirect AUROC, SERGIO bifurcation. CACHED-ONLY evaluation of per-pair score tables
results/signed/dvi_audit3_2026_10_29/full/shard*/pairs_g*_bifurcation_C*_s*.tsv.gz (identical 32x99 support; d = shortest directed path; scores = frozen Fig 1 scores,
parity-asserted against TABLE.tsv). Per regulator: AUROC(d==1 vs d in NEG); keep regulators with >=1 direct and >=1 indirect target; mean over regulators; sims -> GRN mean.
NEG = {2} ("D2", = pooled DvI2 definition of TABLE.tsv) and {2,3} ("D23", = Fig 1f definition of gb_fig2_nopearson_2026_10_30.py). Both reported.
usage (server): python script OUT [grns comma list, default all]"""
import sys, os, glob, json, hashlib
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata
from sklearn.metrics import roc_auc_score as AUC
from statsmodels.stats.multitest import multipletests
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/results'); RT = f'{R}/sergio_tf32rep_2026_10_22'; OUT = sys.argv[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
GRNS = [int(x) for x in sys.argv[2].split(',')] if len(sys.argv) > 2 else [501, 502, 503, 504, 505, 506]
M = {'GB': 's|GB', 'GBk0': 's|GB_K0', 'VEL': 's|VEL', 'fasttenet': 's|fasttenet', 'genie3': 's|genie3', 'grnboost2': 's|grnboost2', 'pidc': 's|pidc', 'regdiffusion': 's|regdiffusion', 'deepsem': 's|deepsem'}
COMP = ['VEL', 'fasttenet', 'genie3', 'grnboost2', 'pidc', 'regdiffusion', 'deepsem']
T = pd.read_csv(f'{RT}/TABLE.tsv', sep='\t').set_index(['grn', 'topo', 'cond', 'sim', 'method'])
ABL = {}
def abl(g, c, s):
    f = f'{RT}/ablation_2026_10_27/full/abl_g{g}_bifurcation_{c}_s{s}.tsv'; a = pd.read_csv(f, sep='\t'); return a[(a.op == 'R1') & (a.ch == 'U') & (a.kind == 'window') & (a.K == 0)].iloc[0]
fl = {}
for f in glob.glob(f'{R}/signed/dvi_audit3_2026_10_29/full/shard*/pairs_g*_bifurcation_C*_s*.tsv.gz'): fl[os.path.basename(f)] = f
rows, par, shas = [], [], {}
for g in GRNS:
    for c in ['C0', 'C1', 'C2', 'C3']:
        for s in [1, 2]:
            f = fl[f'pairs_g{g}_bifurcation_{c}_s{s}.tsv.gz']; P = pd.read_csv(f, sep='\t'); shas[os.path.basename(f)] = hashlib.sha256(open(f, 'rb').read()).hexdigest()
            assert len(P) == 3168 and P.src.nunique() == 32
            for m, col in M.items():
                for nm, neg in (('D2', [2]), ('D23', [2, 3])):
                    sel = P[(P.d == 1) | P.d.isin(neg)]; pooled = AUC(sel.d == 1, sel[col])
                    if nm == 'D2' and m not in ('GBk0',):
                        ref = T.loc[(g, 'bifurcation', c, s, m), 'DvI2']; par.append((g, c, s, m, pooled, ref)); assert abs(pooled - ref) < 1e-9, ('parity', g, c, s, m, pooled, ref)
                    if nm == 'D2' and m == 'GBk0': par.append((g, c, s, 'GBk0_vs_ablK0', pooled, float(abl(g, c, s).DvI2)))
                    a = []
                    for q, d in sel.groupby('src'):
                        y = (d.d == 1).values
                        if y.sum() >= 1 and (~y).sum() >= 1: a.append(AUC(y, d[col].values))
                    rows.append(dict(grn=g, cond=c, sim=s, method=m, neg=nm, macro=np.mean(a), n_reg=len(a), pooled=pooled))
D = pd.DataFrame(rows); D.to_csv(OUT + 'T1B_macro_dvi_datasets.tsv', sep='\t', index=False)
pp = pd.DataFrame(par, columns=['grn', 'cond', 'sim', 'method', 'recomputed', 'ref']); pp['absdiff'] = (pp.recomputed - pp.ref).abs()
print('PARITY (pooled DvI2 recomputed vs TABLE.tsv): n =', int((pp.method != 'GBk0_vs_ablK0').sum()), 'max |diff| =', pp[pp.method != 'GBk0_vs_ablK0'].absdiff.max(), ' GB/VEL datasets:', int(pp.method.isin(['GB', 'VEL']).sum()))
kk = pp[pp.method == 'GBk0_vs_ablK0']; print('GB_K0 pair-file pooled DvI2 vs ablation K=0 DvI2: max |diff| =', kk.absdiff.max(), '(informational)'); print(kk.round(4).to_string())
G = D.groupby(['grn', 'cond', 'method', 'neg'])['macro'].mean().reset_index(); pd.set_option('display.width', 250); out = []
for nm in ('D2', 'D23'):
    print(f'\n=== macro DvI ({nm}) ; per-reg n_reg mean:', D[D.neg == nm].n_reg.mean().round(1))
    for c in ['C0', 'C1', 'C2', 'C3']:
        S = G[(G.cond == c) & (G.neg == nm)].pivot(index='grn', columns='method', values='macro'); n = len(S); ps = []; rr = []
        for m in COMP + ['GBk0']:
            d = (S['GB'] - S[m]).values; h = stats.t.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            p = stats.ttest_rel(S['GB'], S[m]).pvalue if n > 1 else np.nan; rr.append(dict(neg=nm, cond=c, comparator=m, mean_GB=S.GB.mean(), mean_comp=S[m].mean(), diff=d.mean(), ci_lo=d.mean() - h, ci_hi=d.mean() + h, p_t=p, n_gb_wins=int((d > 0).sum()), n_grn=n))
        r = pd.DataFrame(rr)
        if n > 1: r.loc[r.comparator.isin(COMP), 'p_holm7'] = multipletests(r[r.comparator.isin(COMP)].p_t, method='holm')[1]
        out.append(r); print(c); print(r.drop(columns=['neg', 'cond']).round(4).to_string(index=False))
pd.concat(out).to_csv(OUT + 'T1B_macro_dvi_GB_minus_method.tsv', sep='\t', index=False)
json.dump(dict(inputs=shas, table_sha=hashlib.sha256(open(f'{RT}/TABLE.tsv', 'rb').read()).hexdigest(), grns=GRNS), open(OUT + 'receipt_T1B.json', 'w'), indent=1)
