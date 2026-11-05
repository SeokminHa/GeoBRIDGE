"""Matched-baseline ablation on the Fig 2 SERGIO TF32REP benchmark (external reviewer 2026-11-02: does the gain come from the concordance statistic
or from the shared pipeline?). Everything except the per-depth association statistic is held fixed: same stored forward pairs per root
(raw/geometry_raw.npz), same production 30-PC embedding, same QR basis Q of the PC differences and nested projection, same target universe
(cols_PI), same frozen aggregator gb_primary.zcons (z within source, mean k=0..8, >=6/9 finite, min over roots).
Per depth k, with x~, y~ = projected pair differences (Ms source, Mu target):
  CONC = |sum x~y~| / sum|x~||y~|   (production; parity vs stored TRUE|U|<root>|absS asserted)
  COS  = |sum x~y~| / (||x~|| ||y~||)                (uncentred pair-space cosine = residual association)
  PEAR = |corr(x~, y~)| over pairs                     (centred pair-space Pearson)
Plus cell-space PCPART = |partial r(Ms_src, Mu_tgt | first k PCs)| over cells, k=0..8, z mean (gb_b4_score.pcpart_matched, no roots).
Each statistic also at k=0 only. Metrics per dataset: AP (direct edge vs all), DvI23 AUROC, DvI2 AUROC. Sims averaged within GRN; paired GRN-level
differences CONC08 - arm, mean and 95% t CI over 6 GRNs, per topo x cond and pooled. EVALUATION ONLY, no truth enters any score.
usage (server): script OUT [--grns 501,..] [--topos ..] [--conds C0,..] [--sims 1,2]"""
import sys, os, json, hashlib, numpy as np, pandas as pd
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch')]
import gb_tf32rep_fast_2026_10_22 as D, gb_primary as GP, gb_b4_score as B
import anndata as ad
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as APs
from scipy import stats
a = sys.argv; OUT = a[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
arg = lambda k, d: a[a.index(k) + 1].split(',') if k in a else d
GRNS = [int(x) for x in arg('--grns', ['501', '502', '503', '504', '505', '506'])]; TOPOS = arg('--topos', ['bifurcation', 'tree'])
CONDS = arg('--conds', ['C0', 'C1', 'C2', 'C3']); SIMS = [int(x) for x in arg('--sims', ['1', '2'])]
KS = range(0, 9); RT = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/results/sergio_tf32rep_2026_10_22')
TAB = pd.read_csv(f'{RT}/TABLE.tsv', sep='\t').query("method=='GB'").set_index(['grn', 'topo', 'cond', 'sim'])
dense = lambda M: np.asarray(M.todense() if hasattr(M, 'todense') else M, float)
def stats_root(Ms, Mu, P30, I, J, src):
    Q, _ = np.linalg.qr(P30[J] - P30[I]); dX = Ms[J][:, src] - Ms[I][:, src]; dY = Mu[J] - Mu[I]; out = {s: [] for s in ('CONC', 'COS', 'PEAR')}
    for k in KS:
        x = dX - Q[:, :k] @ (Q[:, :k].T @ dX) if k else dX; y = dY - Q[:, :k] @ (Q[:, :k].T @ dY) if k else dY
        num = x.T @ y; out['CONC'].append(np.abs(num) / (np.abs(x).T @ np.abs(y)))
        out['COS'].append(np.abs(num) / np.outer(np.linalg.norm(x, axis=0), np.linalg.norm(y, axis=0)))
        xc = x - x.mean(0); yc = y - y.mean(0); out['PEAR'].append(np.abs(xc.T @ yc) / np.outer(np.linalg.norm(xc, axis=0), np.linalg.norm(yc, axis=0)))
    return {s: np.stack(v) for s, v in out.items()}   # (9, nsrc, ngenes)
rows, par = [], []
for t in TOPOS:
    for g in GRNS:
        G = D.V_().generate_grn(g); gt = sorted((int(x), int(y)) for x, y in G['gt']); DM = D.dist_matrix(gt)
        for c in CONDS:
            for s in SIMS:
                dd = D.dsdir(g, t, c, s); Z = np.load(f'{dd}/raw/geobridge_raw.npz'); Gm = np.load(f'{dd}/raw/geometry_raw.npz')
                src = np.asarray(Z['source_gene_index'], int); cols = np.asarray(Z['cols_PI'], int)
                A = ad.read_h5ad(D.h5of(g, t, c, s) if hasattr(D, 'h5of') else [f'{dd}/{f}' for f in os.listdir(dd) if f.endswith('.h5ad')][0])
                Ms, Mu = dense(A.layers['Ms']), dense(A.layers['Mu']); P30 = np.asarray(Gm['production_pca30'], float)
                Sr = {r: stats_root(Ms, Mu, P30, np.asarray(Gm[f'pairs_{r}_I'], int), np.asarray(Gm[f'pairs_{r}_J'], int), src) for r in GP.ROOTS}
                for r in GP.ROOTS:   # parity: CONC == stored production |S| (float32 producer) on the scored targets
                    st = np.asarray(Z[f'TRUE|U|{r}|absS'], float)[:9]; m = np.isfinite(st)
                    for i in range(len(src)): m[:, i, src[i]] = False
                    d_ = np.nanmax(np.abs(Sr[r]['CONC'][m] - st[m])); par.append(d_); assert d_ < 1e-4, ('parity', dd, r, d_)
                Dm = np.array([[DM[s_, int(x)] for x in cols[i]] for i, s_ in enumerate(src)]); Y = Dm == 1; assert Y.sum() == 240
                def agg(stat, ks):
                    return np.vstack([GP.zcons({r: Sr[r][stat][:, i, cols[i]].T for r in GP.ROOTS}, list(ks)) for i in range(len(src))])
                V = {f'{st}_{w}': agg(st, ks) for st in ('CONC', 'COS', 'PEAR') for w, ks in (('k0', [0]), ('k08', KS))}
                pcp = B.pcpart_matched(Ms, Mu, P30, src, cols); V['PCPART_cell_k08'] = pcp
                prod = V['CONC_k08']; tr = TAB.loc[(g, t, c, s)]; ok = np.isfinite(prod)
                assert abs(APs(Y[ok], prod[ok]) - tr.AP) < 1e-6, ('TABLE parity', dd, APs(Y[ok], prod[ok]), tr.AP)
                for nm, sc in V.items():
                    ok = np.isfinite(sc); r_ = dict(topo=t, grn=g, cond=c, sim=s, arm=nm, coverage=ok.mean(), AP=APs(Y[ok], sc[ok]))
                    for lab, neg in (('DvI23', (2, 3)), ('DvI2', (2,))):
                        mm = ok & ((Dm == 1) | np.isin(Dm, neg)); r_[f'{lab}_AUROC'] = AUC(Dm[mm] == 1, sc[mm])
                    rows.append(r_)
        print('done', t, g, 'max parity', max(par), flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT + 'datasets.tsv', sep='\t', index=False)
M = ['AP', 'DvI23_AUROC', 'DvI2_AUROC']; Gm_ = R.groupby(['topo', 'cond', 'grn', 'arm'])[M].mean().reset_index()
base = Gm_[Gm_.arm == 'CONC_k08'].set_index(['topo', 'cond', 'grn'])[M]; out = []
for scope, sel in [('pooled', lambda d: d.groupby('grn')[M].mean())] + [((t, c), None) for t in TOPOS for c in CONDS]:
    for arm in sorted(Gm_.arm.unique()):
        d = Gm_[Gm_.arm == arm]
        if scope == 'pooled': x = d.groupby('grn')[M].mean(); b = Gm_[Gm_.arm == 'CONC_k08'].groupby('grn')[M].mean()
        else: x = d[(d.topo == scope[0]) & (d.cond == scope[1])].set_index('grn')[M]; b = base.loc[scope].reindex(x.index)
        for m in M:
            diff = (b[m] - x[m]).to_numpy(); n = len(diff); h = stats.t.ppf(.975, n - 1) * diff.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
            out.append(dict(scope=str(scope), arm=arm, metric=m, arm_mean=x[m].mean(), prod_minus_arm=diff.mean(), lo=diff.mean() - h, hi=diff.mean() + h,
                            n_grn_prod_better=int((diff > 0).sum()), n_grn=n, p_t=stats.ttest_1samp(diff, 0).pvalue if n > 1 and arm != 'CONC_k08' else np.nan))
S = pd.DataFrame(out); S.to_csv(OUT + 'summary.tsv', sep='\t', index=False)
json.dump(dict(max_parity=max(par), n=len(R), grns=GRNS, topos=TOPOS, conds=CONDS, sims=SIMS), open(OUT + 'receipt.json', 'w'), indent=1)
pd.set_option('display.width', 250); print(S[S.scope == 'pooled'].round(4).to_string(index=False))
print(S.pivot_table(index=['scope', 'arm'], columns='metric', values='arm_mean').round(3).to_string())
