"""Fig 2 (SERGIO TF32REP) k-curves: GeoBRIDGE performance vs conditioning depth, K = 0..30.
EVALUATION ONLY on stored PRIMARY raw |S| (raw/geobridge_raw.npz 'TRUE|U|<root>|absS', operator R1, channel U, depths 0..30); no rescoring.
Two depth views: cumulative = frozen aggregator gb_primary.zcons over k = 0..K (production = K 8); single = zcons over [K] only.
Metrics per dataset: AP, AUROC (direct edges vs all other candidate pairs), DvI AUROC / DvI AP for direct (d=1) vs indirect,
indirect = {2,3} (DvI23, manuscript Fig 2f definition) and {2} (DvI2, TABLE.tsv definition). Sims averaged within GRN; mean and 95% t CI over 6 GRNs.
Parity: cumulative K=8 must equal TABLE.tsv GB (AP, AUROC, DvI2) and ablation abl_*.tsv R1/U window rows (AP, AUROC, DvI2, DvI3) for every K.
usage (from $GEOBRIDGE_ROOT): script OUT [--grns 501,...] [--topos bifurcation,tree]"""
import sys, os, json, hashlib, numpy as np, pandas as pd
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch')]
import gb_tf32rep_fast_2026_10_22 as D
import gb_b4_score as B
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as APs
a = sys.argv; OUT = a[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
GRNS = [int(x) for x in a[a.index('--grns') + 1].split(',')] if '--grns' in a else [501, 502, 503, 504, 505, 506]
TOPOS = a[a.index('--topos') + 1].split(',') if '--topos' in a else ['bifurcation', 'tree']
RT = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/results/sergio_tf32rep_2026_10_22'); KMAX = 30; ROOTS = B.GP.ROOTS
TAB = pd.read_csv(f'{RT}/TABLE.tsv', sep='\t').query("method=='GB'").set_index(['grn', 'topo', 'cond', 'sim'])
rows, par, shas = [], [], {}
for t in TOPOS:
    for g in GRNS:
        gd = D.gdir(g, t); pre = json.load(open(f'{gd}/roster_pre.json'))
        G = D.V_().generate_grn(g); gt = sorted((int(x), int(y)) for x, y in G['gt']); DM = D.dist_matrix(gt)
        for c in ['C0', 'C1', 'C2', 'C3']:
            for s in [1, 2]:
                tag = D.tagof(t, g, s); f = f'{gd}/{c}/{tag}/raw/geobridge_raw.npz'; Z = np.load(f); shas[f] = hashlib.sha256(open(f, 'rb').read()).hexdigest()
                src = np.asarray(Z['source_gene_index'], int); cols = np.asarray(Z['cols_PI'], int); assert list(src) == pre['cands']
                Dm = np.array([[DM[s_, int(x)] for x in cols[i]] for i, s_ in enumerate(src)]); Y = Dm == 1; assert Y.sum() == 240
                A = {r: np.asarray(Z[f'TRUE|U|{r}|absS'], float) for r in ROOTS}; assert A[ROOTS[0]].shape[0] == KMAX + 1   # layout (K, src, gene)
                per = [{r: A[r][:, i, cols[i]].T for r in ROOTS} for i in range(len(src))]
                ab = pd.read_csv(f'{RT}/ablation_2026_10_27/full/abl_g{g}_{t}_{c}_s{s}.tsv', sep='\t'); ab = ab[(ab.op == 'R1') & (ab.ch == 'U')].set_index(['kind', 'K'])
                for kind in ('cumulative', 'single'):
                    for K in range(KMAX + 1):
                        sc = B.zc(per, range(0, K + 1) if kind == 'cumulative' else [K]); ok = np.isfinite(sc)
                        r = dict(topo=t, grn=g, cond=c, sim=s, kind=kind, K=K, coverage=ok.mean(), AP=APs(Y[ok], sc[ok]), AUROC=AUC(Y[ok], sc[ok]))
                        for nm, neg in (('DvI23', (2, 3)), ('DvI2', (2,))):
                            m = ok & ((Dm == 1) | np.isin(Dm, neg)); r[f'{nm}_AUROC'] = AUC(Dm[m] == 1, sc[m]); r[f'{nm}_AP'] = APs(Dm[m] == 1, sc[m]); r[f'{nm}_prev'] = (Dm[m] == 1).mean()
                        m3 = ok & ((Dm == 1) | (Dm == 3)); r['DvI3_AUROC'] = AUC(Dm[m3] == 1, sc[m3])
                        rows.append(r)
                        ref = ab.loc[('window' if kind == 'cumulative' else 'single', K)]
                        d = max(abs(r['AP'] - ref.AP), abs(r['AUROC'] - ref.AUROC), abs(r['DvI2_AUROC'] - ref.DvI2), abs(r['DvI3_AUROC'] - ref.DvI3))
                        par.append(d); assert d < 1e-9, ('ablation parity', t, g, c, s, kind, K, d)
                        if kind == 'cumulative' and K == 8:
                            tr = TAB.loc[(g, t, c, s)]; assert max(abs(r['AP'] - tr.AP), abs(r['AUROC'] - tr.AUROC), abs(r['DvI2_AUROC'] - tr.DvI2)) < 1e-9, ('TABLE parity', g, t, c, s)
        print('done', t, g, flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT + 'kcurve_datasets.tsv', sep='\t', index=False)
print('PARITY vs ablation (AP, AUROC, DvI2, DvI3; all K, both kinds): n =', len(par), 'max |diff| =', max(par))
MET = ['AP', 'AUROC', 'DvI23_AUROC', 'DvI23_AP', 'DvI2_AUROC', 'DvI2_AP', 'DvI3_AUROC', 'DvI23_prev', 'DvI2_prev']
Gm = R.groupby(['topo', 'kind', 'cond', 'K', 'grn'])[MET].mean().reset_index()                 # sims -> GRN
from scipy import stats
def ci(x): n = len(x); h = stats.t.ppf(.975, n - 1) * np.std(x, ddof=1) / np.sqrt(n); return pd.Series(dict(mean=np.mean(x), lo=np.mean(x) - h, hi=np.mean(x) + h, n_grn=n))
S = Gm.melt(id_vars=['topo', 'kind', 'cond', 'K', 'grn'], value_vars=MET, var_name='metric').groupby(['topo', 'kind', 'cond', 'K', 'metric']).value.apply(ci).unstack().reset_index()
S.to_csv(OUT + 'kcurve_summary.tsv', sep='\t', index=False)
json.dump(dict(inputs=shas, n_rows=len(R), grns=GRNS, topos=TOPOS), open(OUT + 'receipt.json', 'w'), indent=1)
# ---- figure: one row of panels (AUPRC, AUROC, DvI AUROC, DvI AUPRC), lines = conditions, cumulative depth; production K=8 marked
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Nimbus Sans', 'Arial', 'DejaVu Sans'], 'font.size': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6})
CC = {'C0': ('#333333', 'No shared effect'), 'C1': ('#e69f00', 'Temporal effect (A)'), 'C2': ('#0072b2', 'Branch effect (B)'), 'C3': ('#cc3311', 'Dual effect (A+B)')}
PAN = [('AP', 'AUPRC', 'prevAP'), ('AUROC', 'AUROC', .5), ('DvI23_AUROC', 'Direct vs indirect AUROC', .5), ('DvI23_AP', 'Direct vs indirect AUPRC', 'prevDvI')]
for t in TOPOS:
    for kind in ('cumulative', 'single'):
        fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.15))
        for ax, (m, yl, ref) in zip(axs, PAN):
            for c, (col, lab) in CC.items():
                d = S[(S.topo == t) & (S.kind == kind) & (S.cond == c) & (S.metric == m)].sort_values('K')
                ax.plot(d.K, d['mean'], color=col, lw=1, label=lab); ax.fill_between(d.K, d.lo, d.hi, color=col, alpha=.12, lw=0)
            if ref == 'prevAP': ax.axhline(240 / 3168, ls='--', color='grey', lw=.6)
            elif ref == 'prevDvI':
                pv = S[(S.topo == t) & (S.kind == kind) & (S.metric == 'DvI23_prev')]['mean'].mean(); ax.axhline(pv, ls='--', color='grey', lw=.6)
            else: ax.axhline(ref, ls='--', color='grey', lw=.6)
            if kind == 'cumulative': ax.axvspan(0, 8, color='#eeeeee', zorder=0); ax.axvline(8, color='k', lw=.5, ls=':')
            ax.set_xlim(0, KMAX); ax.set_xlabel('Maximum conditioning depth K' if kind == 'cumulative' else 'Conditioning depth k'); ax.set_ylabel(yl)
        h, l = axs[0].get_legend_handles_labels(); fig.legend(h, l, frameon=False, fontsize=6.5, loc='lower center', ncol=4)
        fig.tight_layout(rect=(0, .1, 1, 1)); fig.savefig(OUT + f'Fig2_kcurve_{t}_{kind}.png', dpi=300); plt.close(fig)
print(S[(S.kind == 'cumulative') & (S.K.isin([0, 8, 30]))].pivot_table(index=['topo', 'cond', 'K'], columns='metric', values='mean').round(3).to_string())
