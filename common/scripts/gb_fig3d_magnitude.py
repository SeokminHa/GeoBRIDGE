# Fig3d magnitude association
import sys, os, gzip, numpy as np, pandas as pd
from pathlib import Path
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch'), (__import__("os").environ["GEOBRIDGE_ROOT"] + '/pairspace')]
RU = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); ROOT = RU; RU += 'results/'
VEC = RU + 'real_uniform/meta_nascent_sutil/full/parts/'
BS = RU + 'real_uniform/lowrank_audit/full_20261007/biology_scatter.tsv'
METH = ['GeoBRIDGE', '|Pearson|', 'GENIE3', 'GRNBoost2', 'RegVelo |J|', 'Velorama', 'RegDiffusion', 'DeepSEM', 'PIDC', 'FastTENET']
K562 = 'CDC5L CEBPZ E4F1 GATA1 HINFP MAX MYBL2 NRF1 SETDB1 TFDP1 THAP1 ZBTB17'.split()
EPS = ([('PANC', 'PANC', 'RFX6', 'hist'), ('GATA1', 'ERYTHROID', 'GATA1', 'hist'), ('KLF1', 'ERYTHROID', 'KLF1', 'hist'), ('KLF1_B', 'ERYTHROID', 'KLF1', 'hist'),
        ('BM_CHIPATLAS_CALTOT', 'BM', 'GATA1', 'cal'), ('BM_CHIPATLAS_CALTOT', 'BM', 'RUNX1', 'cal'), ('BM_GATA1_ACUTE', 'BM_ACUTE', 'GATA1', 'hist'),
        ('CARDIACV', 'HEART', 'TBX5', 'hist'), ('CARDIACV', 'HEART', 'NKX2-5', 'hist')] + [('K562_EMP_ALLNULL', 'K562', t, 'hist') for t in K562])
BIOSYS = {'PANC': 'PANC', 'GATA1': 'GATA1', 'KLF1': 'GATA1', 'KLF1_B': 'GATA1', 'BM_CHIPATLAS_CALTOT': 'BONEMARROW', 'BM_GATA1_ACUTE': 'BONEMARROW', 'CARDIACV': 'CARDIACV', 'K562_EMP_ALLNULL': 'K562'}


def up(s): return pd.Series(np.asarray(s).astype(str)).str.upper().to_numpy()


def uniq(sym, val):
    s = pd.Series(np.asarray(val, float), index=up(sym)); return s[~s.index.duplicated(keep=False)]


_pp = {}
def per_pair(d, est, tf):
    if d not in _pp:
        _pp[d] = pd.read_csv(f'{RU}breadth_truth/{d}/per_pair.tsv.gz', sep='\t', usecols=['estimand', 'pert', 'gene', 'T', 'log2fc'])
    x = _pp[d]; x = x[(x.estimand == est) & (x.pert == tf) & (x['T'] == 1)]
    return uniq(x.gene, x.log2fc)


def effect(system, tf):
    if system == 'PANC':
        d = pd.read_csv(ROOT + 'real_1_Pancreas/external_validation/aldous_rfx6/results/aldous_PPs_WT_vs_KO_DE.tsv', sep='\t'); return uniq(d.gene_name.fillna(''), d.log2FoldChange)
    if system == 'KLF1_B':
        d = pd.read_csv(RU + 'lattice_extra/truth/klf1_GSE33979_deseq2.tsv', sep='\t'); return uniq(d.gene, d.log2FC)
    if system == 'KLF1':
        D = pd.read_csv(ROOT + 'results/klf1_heldout/truth/GSE94351_gene_exp.diff.gz', sep='\t')
        D['s1'], D['s2'] = D.sample_1.astype(str).str.upper(), D.sample_2.astype(str).str.upper()
        D = D[((D.s1 == 'Q2') & (D.s2 == 'Q3')) | ((D.s1 == 'Q3') & (D.s2 == 'Q2'))]
        D = D[(D.status == 'OK') & ~D.gene.astype(str).str.contains(',') & (D.gene.astype(str) != '-')]
        l2 = D['log2(fold_change)'].astype(float); return uniq(D.gene, np.where(D.s1 == 'Q3', l2, -l2))
    if system == 'CARDIACV':
        d = pd.read_csv(RU + f"cardiac_truth_v1/de_{'Tbx5' if tf == 'TBX5' else 'Nkx2-5'}KO_vs_WT.tsv", sep='\t'); return uniq(d.gene, d.log2FC)
    if system == 'BM_GATA1_ACUTE':
        d = pd.read_csv(ROOT + 'data/gse274110_proseq/GSE274110_Sankaran003_gga_deseq_24h_GATA1vs24h_AAVS1_results_table.txt.gz', sep='\t')
        d = d[d.Symbol.astype(str) != 'NA']; return uniq(d.Symbol, d.log2FoldChange_24h_GATA1vs24h_AAVS1)
    if system == 'BM_CHIPATLAS_CALTOT': return per_pair('BM', 'TOTAL', tf)
    if system == 'K562_EMP_ALLNULL': return per_pair('K562_EMP_SHARD', 'TOTAL', tf)
    if system == 'GATA1':   # gb_gata1_fourclass_canon.py:74 (mean log-ratio from stored matrices, KO vs WT sign as there)
        import anndata as ad; from gb_substrate import d_
        G = ROOT + 'results/gata1_clean_2026_08_13'
        W_ = ad.read_h5ad(f'{G}/gata1_control_deterministic.h5ad'); K_ = ad.read_h5ad(f'{G}/gata1_ko_projected_deterministic.h5ad')
        T_ = -np.log2((np.expm1(d_(K_.X)).mean(0) + 1e-6) / (np.expm1(d_(W_.X)).mean(0) + 1e-6)); return uniq(W_.var_names, T_)
    raise KeyError(system)


def rk(x):
    from scipy.stats import rankdata
    return rankdata(x)


# weighted partial Pearson of a,b (already ranks) given design X (n,k incl. const); w (n,) nonneg weights
def pcorr(a, b, X, w):
    sw = np.sqrt(w)[:, None]; Xw = X * sw
    def res(v):
        beta = np.linalg.lstsq(Xw, v * sw[:, 0], rcond=None)[0]; return (v - X @ beta) * sw[:, 0]
    ra, rb = res(a), res(b); den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum()); return float((ra * rb).sum() / den) if den > 0 else np.nan


def stat(a, b, cov, w, adj):
    if (w > 0).sum() < 10: return np.nan
    X = np.ones((len(a), 1)) if not adj else np.column_stack([np.ones(len(a)), cov])
    return pcorr(a, b, X, w)


def agg(vals, key):  # mean ignoring nan
    return np.nanmean(np.stack(vals), 0)


if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); B = int(sys.argv[2]) if len(sys.argv) > 2 else 500
    only = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    import gb_ptable as PT, gb_lattice_extra as LE
    LE.register()
    import gb_calbm_propagate as CP; CP.preflight()   # registers BM_CHIPATLAS_CALTOT (as lowrank_audit units())
    bs = pd.read_csv(BS, sep='\t', usecols=['system', 'method', 'mode', 'gene', 'value', 'raw_det', 'abundance', 'B1_mu'])
    bsg = bs[(bs.method == 'GeoBRIDGE') & (bs['mode'] == 'P')]
    drc = {}; rows = []; draws = {}; dec = []
    for sysn, sub, tf, tr in EPS:
        if only and sysn not in only: continue
        z = np.load(f'{VEC}ep_{sysn}__{tf}__{tr}.vec.npz', allow_pickle=True)
        g = up(z['genes']); M = z['MR'].astype(bool); gi = z['gi']; y = z['y'].astype(bool)
        eff = effect(sysn, tf); b = bsg[bsg.system == BIOSYS[sysn]]; b = b.assign(G=up(b.gene)).drop_duplicates('G', keep=False).set_index('G')
        dup = pd.Series(g).duplicated(keep=False).to_numpy()
        pred = {m: z[f'cog::{m}'].astype(float) for m in METH}
        pred['P'] = pd.Series(g).map(b['value']).to_numpy(float); pred['B1'] = pd.Series(g).map(b['B1_mu']).to_numpy(float)
        afc = np.abs(pd.Series(g).map(eff).to_numpy(float)); det = pd.Series(g).map(b['raw_det']).to_numpy(float); ab = pd.Series(g).map(b['abundance']).to_numpy(float)
        ok = M & ~dup & np.isfinite(afc) & np.isfinite(det) & np.isfinite(ab) & np.all([np.isfinite(v) for v in pred.values()], 0)
        n = int(ok.sum()); print(sysn, tf, 'MR', int(M.sum()), 'universe', n, 'resp', int(y[ok].sum()), flush=True)
        if n < 30: rows.append(dict(system=sub, endpoint=sysn, tf=tf, n=n, status='NO_CALL_TOO_FEW')); continue
        if sysn not in drc: drc[sysn] = PT.lattice_draws(sysn, B, sizes=(50,))[0]
        W = drc[sysn].C[B:2 * B][:, gi[ok]].astype(float)  # module-50 bootstrap weights (B,n)
        cov = np.column_stack([rk(det[ok]), rk(ab[ok])]); rf = rk(afc[ok]); yy = y[ok]
        for m, v in pred.items():
            rv = rk(v[ok])
            for var, msk, adj in (('total_raw', None, 0), ('total_adj', None, 1), ('resp_raw', yy, 0), ('resp_adj', yy, 1)):
                mk = np.ones(n) if msk is None else msk.astype(float)
                pt = stat(rv, rf, cov, mk, adj); dr = np.array([stat(rv, rf, cov, W[i] * mk, adj) for i in range(B)])
                draws[(sub, sysn, tf, m, var)] = np.r_[pt, dr]
        # decile display (S): endpoint-standardised |FC| mean by S decile
        zf = (afc[ok] - afc[ok].mean()) / afc[ok].std(); dcl = np.minimum((rk(pred['GeoBRIDGE'][ok]) - 1) * 10 // n, 9).astype(int)
        dec.append(pd.DataFrame(dict(system=sub, endpoint=sysn, tf=tf, decile=range(10), mean_z_absFC=[zf[dcl == k].mean() for k in range(10)])))
        rows.append(dict(system=sub, endpoint=sysn, tf=tf, n=n, n_resp=int(yy.sum()), status='OK'))
    pd.DataFrame(rows).to_csv(out / 'endpoints.tsv', sep='\t', index=False)
    if dec: pd.concat(dec).to_csv(out / 'deciles_S.tsv', sep='\t', index=False)
    import pickle; pickle.dump(draws, open(out / 'draws.pkl', 'wb'))
    # aggregation endpoint -> TF -> system -> equal-system (acute excluded from BM aggregate, reported alone)
    keys = pd.DataFrame(list(draws), columns=['sub', 'ep', 'tf', 'm', 'var']); res = []
    for m in keys.m.unique():
        for var in keys['var'].unique():
            sel = keys[(keys.m == m) & (keys['var'] == var)]
            sysv = {}
            for sub, d in sel.groupby('sub'):
                tfv = [agg([draws[(sub, e, t, m, var)] for e in dd.ep], 0) for t, dd in d.groupby('tf')]
                sysv[sub] = agg(tfv, 0)
                pt, lo, hi = sysv[sub][0], *np.nanpercentile(sysv[sub][1:], [2.5, 97.5])
                res.append(dict(level='acute_alone' if sub == 'BM_ACUTE' else 'system', unit=sub, method=m, variant=var, est=pt, lo=lo, hi=hi, n_tf=d.tf.nunique()))
            S5 = [v for k, v in sysv.items() if k != 'BM_ACUTE']
            if S5:
                eq = agg(S5, 0); se = np.array([np.nanstd(v[1:]) for v in S5]); th = np.array([v[0] for v in S5])
                w = 1 / se ** 2; mu = (w * th).sum() / w.sum(); Q = (w * (th - mu) ** 2).sum(); I2 = max(0, (Q - (len(th) - 1)) / Q) if Q > 0 else 0
                res.append(dict(level='equal_system', unit=f'{len(S5)} systems', method=m, variant=var, est=eq[0], lo=np.nanpercentile(eq[1:], 2.5), hi=np.nanpercentile(eq[1:], 97.5),
                                n_tf=len(S5), range_lo=th.min(), range_hi=th.max(), I2=I2))
    R = pd.DataFrame(res); R.to_csv(out / 'fig3d_results.tsv', sep='\t', index=False)
    eqs = R[R.level == 'equal_system']
    for var in ('total_raw', 'total_adj', 'resp_raw', 'resp_adj'):
        e = eqs[eqs['variant'] == var].set_index('method').est; print(var, 'rank of GB among 10 comparators:', int(e[METH].rank(ascending=False)['GeoBRIDGE']), e.round(3).to_dict())
