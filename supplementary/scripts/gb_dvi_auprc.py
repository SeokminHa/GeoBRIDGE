# Figure S1: AUPRC for direct versus indirect pairs (d=1 vs 2<=d<=5, unreachable excluded).
import sys, os, glob, json, numpy as np, pandas as pd
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch')]
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as APs
R = __import__("os").environ["GEOBRIDGE_ROOT"]; RF = f'{R}/results/signed/r2fix_2026_11_01/'
a = sys.argv; MODE = a[1]; OUT = a[2].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True); TOL = 1e-6; T0 = 'bifurcation'
arg = lambda k, d: a[a.index(k) + 1].split(',') if k in a else d
MASK = {'DvI23': lambda D: (D == 2) | (D == 3), 'DvIall': lambda D: (D >= 2) & (D < 99)}
def dvi(sc, D, nm, f):
    m = np.isfinite(sc) & ((D == 1) | MASK[nm](D)); return f(D[m] == 1, sc[m])
M = ['GB', 'VEL', 'fasttenet', 'genie3', 'grnboost2', 'pidc', 'regdiffusion', 'deepsem']

if MODE == 'pairs':
    fs = sorted(glob.glob(f'{R}/results/signed/dvi_audit3_2026_10_29/full/shard*/pairs_*_bifurcation_*.tsv.gz')); assert len(fs) == 48, len(fs)
    fs = fs[:int(arg('--n', ['48'])[0])]
    ST = pd.read_csv(f'{R}/results/fig2_dviall_2026_11_05/dviall_per_dataset.tsv', sep='\t').set_index(['grn', 'cond', 'sim'])
    KC = pd.read_csv(RF + 'T5_kcurve/kcurve_datasets.tsv', sep='\t').query("kind=='cumulative' and K==8").set_index(['topo', 'grn', 'cond', 'sim'])
    rows, par = [], []
    for f in fs:
        T = pd.read_csv(f, sep='\t'); D = T.d.to_numpy(); k = (int(T.grn[0]), T.cond[0], int(T.sim[0]))
        assert D[(D > 1) & (D < 99)].max() <= 5
        r = dict(grn=k[0], cond=k[1], sim=k[2], prev_DvIall=float((D == 1).sum() / ((D == 1) | MASK['DvIall'](D)).sum()))
        for m in M:
            s = T[f's|{m}'].to_numpy(float); r[f'AP|{m}'] = dvi(s, D, 'DvIall', APs); r[f'AUROC|{m}'] = dvi(s, D, 'DvIall', AUC)
            d = abs(r[f'AUROC|{m}'] - ST.loc[k][f'DvIall|{m}']); par.append(('pairs_DvIall_AUROC_vs_stored', d)); assert d <= TOL, ('auroc parity', k, m, d)
        d = abs(dvi(T['s|GB'].to_numpy(float), D, 'DvI23', APs) - KC.loc[(T0,) + k].DvI23_AP); par.append(('pairs_GB_DvI23_AP_vs_T5_K8', d)); assert d <= TOL, ('AP parity', k, d)
        rows.append(r); print('ok', k, flush=True)
    pd.DataFrame(rows).to_csv(OUT + 'dviall_ap_per_dataset.tsv', sep='\t', index=False)
    P = pd.DataFrame(par, columns=['what', 'd']).groupby('what').d.agg(['count', 'max']); print(P.to_string()); P.to_csv(OUT + 'parity_pairs.tsv', sep='\t')

elif MODE == 'kc':
    import gb_tf32rep as Dd, gb_b4_score as B
    GRNS = [int(x) for x in arg('--grns', ['501', '502', '503', '504', '505', '506'])]; CONDS = arg('--conds', ['C0', 'C1', 'C2', 'C3']); SIMS = [int(x) for x in arg('--sims', ['1', '2'])]
    KC = pd.read_csv(RF + 'T5_kcurve/kcurve_datasets.tsv', sep='\t').query("kind=='cumulative'").set_index(['topo', 'grn', 'cond', 'sim', 'K'])
    KA = pd.read_csv(RF + 'SI_dviall_2026_11_05/full/kcurve_dviall_datasets.tsv', sep='\t').set_index(['grn', 'cond', 'sim', 'K'])
    rows, par = [], []
    for g in GRNS:
        gd = Dd.gdir(g, T0); pre = json.load(open(f'{gd}/roster_pre.json')); DM = Dd.dist_matrix(sorted((int(x), int(y)) for x, y in Dd.V_().generate_grn(g)['gt']))
        for c in CONDS:
            for s in SIMS:
                Z = np.load(f'{gd}/{c}/{Dd.tagof(T0, g, s)}/raw/geobridge_raw.npz'); src = np.asarray(Z['source_gene_index'], int); cols = np.asarray(Z['cols_PI'], int); assert list(src) == pre['cands']
                Dm = np.array([[DM[q, int(x)] for x in cols[i]] for i, q in enumerate(src)]); assert (Dm == 1).sum() == 240
                A = {r: np.asarray(Z[f'TRUE|U|{r}|absS'], float) for r in B.GP.ROOTS}; per = [{r: A[r][:, i, cols[i]].T for r in B.GP.ROOTS} for i in range(len(src))]
                for K in range(31):
                    sc = B.zc(per, range(0, K + 1)); r = dict(grn=g, cond=c, sim=s, K=K, DvIall_AP=dvi(sc, Dm, 'DvIall', APs), prev_DvIall=float((Dm == 1).sum() / ((Dm == 1) | MASK['DvIall'](Dm)).sum()))
                    for nm, v in (('T5_DvI23_AP', abs(dvi(sc, Dm, 'DvI23', APs) - KC.loc[(T0, g, c, s, K)].DvI23_AP)), ('SI_DvIall_AUROC', abs(dvi(sc, Dm, 'DvIall', AUC) - KA.loc[(g, c, s, K)].DvIall_AUROC))):
                        par.append((nm, v)); assert v <= TOL, (nm, g, c, s, K, v)
                    rows.append(r)
        print('done', g, flush=True)
    tag = '' if len(GRNS) == 6 else '_small'
    pd.DataFrame(rows).to_csv(OUT + f'kcurve_dviall_ap_datasets{tag}.tsv', sep='\t', index=False)
    P = pd.DataFrame(par, columns=['what', 'd']).groupby('what').d.agg(['count', 'max']); print(P.to_string()); P.to_csv(OUT + f'parity_kc{tag}.tsv', sep='\t')
