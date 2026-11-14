# SI synthetic DvI23 -> DvI_all recompute (DvI = direct d=1 vs all indirect 2<=d<=5, unreachable (99) excluded)
import sys, os, json, numpy as np, pandas as pd
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch')]
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as APs
R = __import__("os").environ["GEOBRIDGE_ROOT"]; RF = f'{R}/results/signed/r2fix_2026_11_01/'; RT = f'{R}/results/sergio_tf32rep_2026_10_22'
a = sys.argv; MODE = a[1]; OUT = a[2].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
arg = lambda k, d: a[a.index(k) + 1].split(',') if k in a else d
GRNS = [int(x) for x in arg('--grns', ['501', '502', '503', '504', '505', '506'])]; CONDS = arg('--conds', ['C0', 'C1', 'C2', 'C3']); SIMS = [int(x) for x in arg('--sims', ['1', '2'])]
T0 = 'bifurcation'; TOL = 1e-6
MASK = {'DvI23': lambda Dm: (Dm == 2) | (Dm == 3), 'DvIall': lambda Dm: (Dm >= 2) & (Dm < 99)}
def dvi(sc, Dm, nm):
    ok = np.isfinite(sc); m = ok & ((Dm == 1) | MASK[nm](Dm)); return AUC(Dm[m] == 1, sc[m])
def zrow(sc):   # Within-row standardization.
    out = np.zeros_like(sc, float)
    for i in range(sc.shape[0]):
        r = sc[i]; f = np.isfinite(r); sd = r[f].std() if f.any() else 0; out[i] = np.where(f, (r - r[f].mean()) / sd if sd > 0 else 0., np.nan)
    return out

if MODE == 'kc_e1':
    import gb_tf32rep as D, gb_b4_score as B
    ROOTS = B.GP.ROOTS
    KC = pd.read_csv(RF + 'T5_kcurve/kcurve_datasets.tsv', sep='\t').query("kind=='cumulative'").set_index(['topo', 'grn', 'cond', 'sim', 'K'])
    E1 = pd.read_csv(RF + 'T12_story_E1E2E3E7/dataset_metrics.tsv', sep='\t').set_index(['method', 'topo', 'grn', 'cond', 'sim'])
    EXT = {'deepsem': 'DeepSEM', 'regdiffusion': 'RegDiffusion', 'fasttenet': 'FastTENET', 'genie3': 'GENIE3', 'grnboost2': 'GRNBoost2', 'pidc': 'PIDC', 'pearson': 'Pearson'}
    NAME = {'GB': 'GB', 'GBk0': 'GB_k0', 'VEL': 'Velorama', **EXT}
    kr, er, par = [], [], []
    for g in GRNS:
        gd = D.gdir(g, T0); pre = json.load(open(f'{gd}/roster_pre.json')); DM = D.dist_matrix(sorted((int(x), int(y)) for x, y in D.V_().generate_grn(g)['gt']))
        for c in CONDS:
            for s in SIMS:
                tag = D.tagof(T0, g, s); Z = np.load(f'{gd}/{c}/{tag}/raw/geobridge_raw.npz')
                src = np.asarray(Z['source_gene_index'], int); cols = np.asarray(Z['cols_PI'], int); assert list(src) == pre['cands']
                Dm = np.array([[DM[q, int(x)] for x in cols[i]] for i, q in enumerate(src)]); assert (Dm == 1).sum() == 240
                fin = Dm[(Dm > 1) & (Dm < 99)]; assert fin.max() <= 5, fin.max()
                A = {r: np.asarray(Z[f'TRUE|U|{r}|absS'], float) for r in ROOTS}; per = [{r: A[r][:, i, cols[i]].T for r in ROOTS} for i in range(len(src))]
                for K in range(31):
                    sc = B.zc(per, range(0, K + 1)); r = dict(topo=T0, grn=g, cond=c, sim=s, kind='cumulative', K=K, DvI23_AUROC=dvi(sc, Dm, 'DvI23'), DvIall_AUROC=dvi(sc, Dm, 'DvIall'))
                    d = abs(r['DvI23_AUROC'] - KC.loc[(T0, g, c, s, K)].DvI23_AUROC); par.append(('kcurve', d)); assert d <= TOL, ('kcurve parity', g, c, s, K, d)
                    kr.append(r)
                S = {'GB': B.zc(per, range(0, 9)), 'GBk0': B.zc(per, [0])}
                vf = f'{gd}/velorama_rt0/{c}/{tag}.npz'; z = np.load(vf); assert np.array_equal(z['src'], src) and np.array_equal(z['cols'], cols); S['VEL'] = np.asarray(z['score'], float)
                for m in EXT:
                    z = np.load(D.score_path(m, ('rep', g, T0, c, s))); assert np.array_equal(z['src'], src) and np.array_equal(z['cols'], cols); S[m] = np.asarray(z['score'], float)
                for m, sc in S.items():
                    assert sc.shape == (32, 99) and np.isfinite(sc).all(), (m, g, c, s)
                    r = dict(method=NAME[m], topo=T0, grn=g, cond=c, sim=s)
                    for view, X in (('native', sc), ('z', zrow(sc))):
                        for nm in MASK: r[f'{nm}_{view}'] = dvi(X, Dm, nm)
                        st = E1.loc[(NAME[m], T0, g, c, s)][f'DvI23_{view}']; d = abs(r[f'DvI23_{view}'] - st); par.append((f'E1_{view}', d)); assert d <= TOL, ('E1 parity', m, view, g, c, s, d)
                    er.append(r)
        print('done', g, flush=True)
    pd.DataFrame(kr).to_csv(OUT + 'kcurve_dviall_datasets.tsv', sep='\t', index=False); pd.DataFrame(er).to_csv(OUT + 'E1_dviall_datasets.tsv', sep='\t', index=False)
    P = pd.DataFrame(par, columns=['what', 'd']).groupby('what').d.agg(['count', 'max']); print('PARITY (DvI23 recomputed vs stored per dataset):\n', P.to_string()); P.to_csv(OUT + 'parity_kc_e1.tsv', sep='\t')

elif MODE == 'dose':
    import gb_dose_full_2026_11_04 as DF, gb_b4_score as B
    T = DF.T; V = T.V_(); ST = pd.read_csv(f'{DF.DOSE}/DOSE_DATASETS.tsv', sep='\t').set_index(['grn', 'topo', 'cond', 'sim', 'method']); rows, par = [], []
    for g in GRNS:
        Dd = T.dist_matrix(sorted((int(x), int(y)) for x, y in V.generate_grn(g)['gt']))
        for c in DF.AMP:
            for s in SIMS:
                dd = T.dsdir(g, T0, c, s); Z = np.load(f'{dd}/raw/geobridge_raw.npz'); src, cols = np.asarray(Z['source_gene_index'], int), np.asarray(Z['cols_PI'], int)
                A = {r: np.asarray(Z[f'TRUE|U|{r}|absS'], float) for r in B.GP.ROOTS}; P = [{r: A[r][:, i, cols[i]].T for r in B.GP.ROOTS} for i in range(len(src))]
                S = {'GB': B.zc(P, range(0, 9)), 'GB_k0': B.zc(P, range(0, 1))}
                vf = DF.velo_path(g, T0, c, s)
                if os.path.exists(vf): z = np.load(vf); assert np.array_equal(z['src'], src) and np.array_equal(z['cols'], cols); S['VEL'] = np.asarray(z['score'], float)
                for m in DF.EXT:
                    p = DF.score_file(m, g, T0, c, s)
                    if os.path.exists(p): z = np.load(p); assert np.array_equal(z['src'], src) and np.array_equal(z['cols'], cols); S[m] = np.asarray(z['score'], float)
                Dm = np.array([[Dd[q, int(x)] for x in cols[i]] for i, q in enumerate(src)]); fin = Dm[(Dm > 1) & (Dm < 99)]; assert fin.max() <= 5
                for m, sc in S.items():
                    if not (sc.shape == (32, 99) and np.isfinite(sc).all()): continue        # same skip rule as eval_all
                    r = dict(grn=g, topo=T0, cond=c, amp=DF.AMP[c], sim=s, method=m, DvI23=dvi(sc, Dm, 'DvI23'), DvIall=dvi(sc, Dm, 'DvIall'))
                    d = abs(r['DvI23'] - ST.loc[(g, T0, c, s, m)].DvI23); par.append(d); assert d <= TOL, ('dose parity', g, c, s, m, d)
                    rows.append(r)
        print('done', g, flush=True)
    R_ = pd.DataFrame(rows); n_st = len(ST.reset_index().query("topo==@T0 and grn in @GRNS and sim in @SIMS"))
    print('DOSE PARITY rows', len(par), 'stored rows', n_st, 'max|d|', max(par)); assert len(par) == n_st, 'row count mismatch vs stored DOSE_DATASETS'
    R_.to_csv(OUT + 'dose_dviall_datasets.tsv', sep='\t', index=False)

elif MODE == 'plot':
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    def t95(x): x = np.asarray(x, float); n = len(x); h = stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n); return x.mean(), x.mean() - h, x.mean() + h
    pd.set_option('display.width', 250); PAR = []
    # k-curve summary + delta vs K0 (same aggregation as gb_depth_profile / gb_fig2_k0delta)
    kd = pd.read_csv(OUT + 'kcurve_dviall_datasets.tsv', sep='\t'); Gk = kd.groupby(['cond', 'grn', 'K'])[['DvI23_AUROC', 'DvIall_AUROC']].mean().reset_index()
    kc, dk = [], []
    for m in ('DvI23_AUROC', 'DvIall_AUROC'):
        for (c, K), d in Gk.groupby(['cond', 'K']): mu, lo, hi = t95(d[m]); kc.append(dict(topo=T0, cond=c, K=K, metric=m, mean=mu, lo=lo, hi=hi))
        b = Gk[Gk.K == 0].set_index(['cond', 'grn'])[m]
        for (c, K), d in Gk.groupby(['cond', 'K']):
            x = d.set_index(['cond', 'grn'])[m] - b.loc[d.set_index(['cond', 'grn']).index]; mu, lo, hi = t95(x) if K > 0 else (0., 0., 0.); dk.append(dict(topo=T0, cond=c, K=K, metric=m, mean=mu, lo=lo, hi=hi))
    kc, dk = pd.DataFrame(kc), pd.DataFrame(dk)
    skc = pd.read_csv(RF + 'T5_kcurve/kcurve_summary.tsv', sep='\t').query("kind=='cumulative' and topo==@T0 and metric=='DvI23_AUROC'")
    q = skc.merge(kc[kc.metric == 'DvI23_AUROC'], on=['cond', 'K']); PAR.append(('kcurve_summary', len(q), max((q.mean_x - q.mean_y).abs().max(), (q.lo_x - q.lo_y).abs().max())))
    sdk = pd.read_csv(RF + 'T6_k0delta/delta_vs_k0.tsv', sep='\t').query("topo==@T0 and metric=='DvI23_AUROC' and K>0")
    q = sdk.merge(dk[dk.metric == 'DvI23_AUROC'], on=['cond', 'K']); PAR.append(('delta_vs_k0', len(q), max((q.mean_x - q.mean_y).abs().max(), (q.lo_x - q.lo_y).abs().max())))
    kc.to_csv(OUT + 'kcurve_dviall_summary.tsv', sep='\t', index=False); dk.to_csv(OUT + 'delta_vs_k0_dviall.tsv', sep='\t', index=False)
    # LOGO (same rule as gb_si_figs_bif): train mean curve idxmax, held-out at selected / K8 / K0
    st = pd.read_csv(RF + 'T5_kcurve/kcurve_datasets.tsv', sep='\t').query("kind=='cumulative' and topo==@T0").groupby(['cond', 'grn', 'K'])[['AP', 'AUROC', 'DvI23_AUROC']].mean().reset_index()
    GL = st.drop(columns='DvI23_AUROC').merge(Gk, on=['cond', 'grn', 'K']); assert len(GL) == len(st); lg = []
    for m in ('AP', 'DvI23_AUROC', 'DvIall_AUROC', 'AUROC'):
        for g in sorted(GL.grn.unique()):
            tr = GL[GL.grn != g].groupby('K')[m].mean(); ks = int(tr.idxmax()); te = GL[GL.grn == g].groupby('K')[m].mean(); lg.append(dict(metric=m, heldout=g, K_selected=ks, heldout_selected=te[ks], heldout_K8=te[8], heldout_K0=te[0]))
    lg = pd.DataFrame(lg); lg.to_csv(OUT + 'logo_dviall.tsv', sep='\t', index=False); print('LOGO\n', lg.round(4).to_string()); print(lg.groupby('metric')[['heldout_selected', 'heldout_K8', 'heldout_K0']].mean().round(4))
    # Bifurcation contrasts against C0.
    e1 = pd.read_csv(OUT + 'E1_dviall_datasets.tsv', sep='\t'); VW = ['DvI23_native', 'DvI23_z', 'DvIall_native', 'DvIall_z']
    G1 = e1.groupby(['method', 'grn', 'cond'])[VW].mean().reset_index(); E = []
    for (m), d in G1.groupby('method'):
        for v in VW:
            p = d.pivot(index='grn', columns='cond', values=v)
            for cj in ['C1', 'C2', 'C3']: mu, lo, hi = t95(p[cj] - p['C0']); E.append(dict(method=m, topo=T0, metric=v, contrast=f'{cj}-C0', mean=mu, lo=lo, hi=hi, n_pos=int(((p[cj] - p['C0']) > 0).sum())))
            for cj in ['C0', 'C1', 'C2', 'C3']: mu, lo, hi = t95(p[cj]); E.append(dict(method=m, topo=T0, metric=v, contrast=f'level_{cj}', mean=mu, lo=lo, hi=hi))
    E = pd.DataFrame(E); E.to_csv(OUT + 'E1_dviall_degradation.tsv', sep='\t', index=False)
    s1 = pd.read_csv(RF + 'T12_story_E1E2E3E7/E1_degradation.tsv', sep='\t').query("topo==@T0 and metric in ['DvI23_native','DvI23_z']")
    q = s1.merge(E, on=['method', 'metric', 'contrast']); PAR.append(('E1_degradation', len(q), max((q.mean_x - q.mean_y).abs().max(), (q.lo_x - q.lo_y).abs().max())))
    print('E1 levels\n', E[E.contrast.str.startswith('level')].pivot_table(index='method', columns=['metric', 'contrast'], values='mean').round(3).to_string())
    print('E1 C3-C0\n', E[E.contrast == 'C3-C0'].pivot_table(index='method', columns='metric', values='mean').round(4).to_string())
    # dose summary (same as gb_dose_full agg, bifurcation)
    dd = pd.read_csv(OUT + 'dose_dviall_datasets.tsv', sep='\t'); Gd = dd.groupby(['amp', 'method', 'grn'])[['DvI23', 'DvIall']].mean().reset_index(); SD = []
    for (am, m), d in Gd.groupby(['amp', 'method']):
        r = dict(topo=T0, amp=am, method=m)
        for k in ('DvI23', 'DvIall'): mu, lo, hi = t95(d[k]); r[k] = mu; r[k + '_ci95_half'] = hi - mu
        SD.append(r)
    SD = pd.DataFrame(SD); SS = pd.read_csv(f'{RT}/dose_2026_11_04/SUMMARY.tsv', sep='\t').query("topo==@T0")
    q = SS.merge(SD, on=['amp', 'method']); PAR.append(('dose_summary', len(q), max((q.DvI23_x - q.DvI23_y).abs().max(), (q.DvI23_ci95_half_x - q.DvI23_ci95_half_y).abs().max())))
    PD = []
    for am, d in Gd.groupby('amp'):
        w = d.pivot(index='grn', columns='method', values='DvIall'); cm = [m for m in w.columns if m not in ('GB', 'GB_k0')]
        ps = [stats.ttest_rel(w['GB'], w[m]).pvalue for m in cm]; ph = multipletests(ps, method='holm')[1]
        PD += [dict(amp=am, comparator=m, diff=(w['GB'] - w[m]).mean(), p=p, p_holm=h) for m, p, h in zip(cm, ps, ph)]
    print('DOSE DvIall paired (GB vs comparators incl. pearson, Holm over the comparators present)\n', pd.DataFrame(PD).round(4).to_string())
    print('DOSE DvIall means\n', SD.pivot(index='method', columns='amp', values='DvIall').round(3).to_string())
    # stored-plot-compatible SUMMARY (column DvI23 holds DvI_all, so the unchanged replot script
    # draws it)
    os.makedirs(OUT + 'dose_dir', exist_ok=True)
    SS.drop(columns=['DvI23', 'DvI23_ci95_half']).merge(SD[['amp', 'method', 'DvIall', 'DvIall_ci95_half']].rename(columns={'DvIall': 'DvI23', 'DvIall_ci95_half': 'DvI23_ci95_half'}), on=['amp', 'method']).to_csv(OUT + 'dose_dir/SUMMARY.tsv', sep='\t', index=False)
    P = pd.DataFrame(PAR, columns=['what', 'n', 'max_abs_diff']); print('SUMMARY PARITY (DvI23 recomputed vs stored):\n', P.to_string()); P.to_csv(OUT + 'parity_summaries.tsv', sep='\t', index=False); assert (P.max_abs_diff <= TOL).all() and (P.n > 0).all()
