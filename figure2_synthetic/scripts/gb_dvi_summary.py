# Aggregate the direct-vs-indirect audit (reads pairs_*.tsv.gz and receipt_*.json written by gb_dvi_pairs.py)
import sys, os, glob, json, itertools, hashlib
import numpy as np, pandas as pd
from scipy import stats
from scipy.stats import rankdata
from multiprocessing import Pool
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch'))
import gb_tf32rep as TF
D = sys.argv[1].rstrip('/') + '/'; PART = '--partial' in sys.argv; OD = D + ('agg_partial/' if PART else 'agg/'); os.makedirs(OD, exist_ok=True)
sh_ = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
F = sorted(glob.glob(D + '**/pairs_*.tsv.gz', recursive=True)); P = pd.concat([pd.read_csv(f, sep='\t') for f in F], ignore_index=True)
KEY = ['grn', 'topo', 'cond', 'sim']; DSET = set(map(tuple, P[KEY].drop_duplicates().values.tolist())); NDS = len(DSET)
assert not P.duplicated(KEY + ['src', 'tgt']).any() and (P.groupby(KEY).size() == 3168).all(), 'pair grid'
GR = sorted(P.grn.unique()) if PART else list(TF.GRNS); assert PART or sorted(P.grn.unique()) == sorted(TF.GRNS)
EXPG = {(g, t, c, s) for g in GR for t in TF.TOPOS for c in TF.COND for s in TF.SIMS}; assert DSET == EXPG, ('dataset grid', len(DSET), len(EXPG))
RC = {os.path.basename(f): json.load(open(f)) for f in sorted(glob.glob(D + '**/receipt_*.json', recursive=True))}
assert len({r['script_sha'] for r in RC.values()}) == 1 and len({json.dumps(r['deps'], sort_keys=True) for r in RC.values()}) == 1, 'mixed scorer versions'
cov = [d_['dataset'] for r in RC.values() for d_ in r['datasets']]; assert len(cov) == len(set(cov)), 'dataset scored twice'
assert {tuple(int(p) if p.isdigit() else p for p in d_.split(':')) for d_ in cov} == DSET, 'receipt coverage != pair tables'
METH = ['GB', 'GB_K0', 'GBX', 'GBX_K0', 'GBraw', 'GBraw_K0', 'GBXraw', 'GBXraw_K0', 'GB_NUMONLY', 'GB_DENONLY',
        'VEL', 'fasttenet', 'genie3', 'grnboost2', 'pidc', 'regdiffusion', 'deepsem', 'pearson']
P['NP'] = P.d == 99; P['NPx'] = (P.d == 99) & (P.drev == 99); P['D1'] = P.d == 1; P['D2'] = P.d == 2; P['D3'] = P.d == 3; P['REV'] = P.rev == 1
# strength strata
def tert(v):
    q1, q2 = np.quantile(v, [1 / 3, 2 / 3]); return np.where(v <= q1, 'weak', np.where(v > q2, 'strong', 'mid'))
GT = ['grn', 'topo']; PK = GT + ['src', 'tgt']
C0m = P[P.cond == 'C0'].groupby(PK)[['gain_lin', 'gain1', 'gain_net_absmean', 'rassoc', 'mshare']].mean().add_suffix('_C0').reset_index(); P = P.merge(C0m, on=PK, how='left')
P['clean'] = P.D2 & (P.n_med == 1) & (P.n_paths3 == 0)
for nm, col, base in (('lin', 'gain_lin_C0', 'D2'), ('net', 'gain_net_absmean_C0', 'D2'), ('rassoc', 'rassoc_C0', 'D2'), ('mshare', 'mshare_C0', 'D2'), ('clean', 'gain_lin_C0', 'clean')):
    P[f'st_{nm}'] = ''
    for k, x in P[P[base] & (P.cond == 'C0')].drop_duplicates(PK).groupby(GT):
        nn = int(x[col].isna().sum()); assert nn <= 0.05 * len(x), (nm, k, nn)
        if nn: print('strata', nm, k, 'NaN strength pairs (unassigned):', nn)
        q1, q2 = np.nanquantile(x[col].values, [1 / 3, 2 / 3]); m_ = P[base] & (P.grn == k[0]) & (P.topo == k[1]) & P[col].notna()
        P.loc[m_, f'st_{nm}'] = np.where(P.loc[m_, col] <= q1, 'weak', np.where(P.loc[m_, col] > q2, 'strong', 'mid'))
for nm, col in (('lin_ds', 'gain_lin'), ('K', 'gain_K'), ('pear', 's|pearson')):
    P[f'st_{nm}'] = ''
    for k, ix in P[P.D2].groupby(KEY).groups.items(): P.loc[ix, f'st_{nm}'] = tert(P.loc[ix, col].values)
d1m = P[P.D1 & (P.cond == 'C0')].drop_duplicates(PK).groupby(GT).gain1_C0.median().rename('d1med_C0'); P = P.merge(d1m, left_on=GT, right_index=True)
for b, nm in ((0.8, 'abs08'), (1.2, 'abs12')): P[f'st_{nm}'] = np.where(P.D2, np.where(P.gain1_C0 >= b * P.d1med_C0, 'strong', 'below'), '')
P['multi'] = np.where(P.D2, np.where(P.n_med >= 2, 'multi', 'single'), ''); P['hub'] = ''
for k, ix in P[P.D2].groupby(KEY).groups.items(): v = P.loc[ix, 'med_outdeg_max']; P.loc[ix, 'hub'] = np.where(v > v.median(), 'hub', 'nonhub')
# coupling-gain quartile per (GRN, topo) over D1 (direct gain) and D2 (cancellation-aware net gain),
# C0-defined
P['cgain'] = np.where(P.D1, P.gain_lin_C0, np.where(P.D2, P.gain_net_absmean_C0, np.nan)); P['gbin'] = -1
for k, x in P[(P.D1 | P.D2) & (P.cond == 'C0')].drop_duplicates(PK).groupby(GT):
    assert x.cgain.notna().all(), ('cgain NaN', k); qs = np.quantile(np.log(x.cgain.values), [.25, .5, .75]); m_ = (P.D1 | P.D2) & (P.grn == k[0]) & (P.topo == k[1]); P.loc[m_, 'gbin'] = np.searchsorted(qs, np.log(P.loc[m_, 'cgain'].values))
P['g_src_bin'] = P.src * 10 + P.gbin; P['g_src_bin_lay'] = P.g_src_bin * 100 + P.lay_t
def auc(pos, neg):
    if len(pos) == 0 or len(neg) == 0: return np.nan
    r = rankdata(np.r_[pos, neg]); return (r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
SCOLS = [f's|{m}' for m in METH]
CONTR = {'D1vNP': ('D1', 'NP'), 'D2vNP': ('D2', 'NP'), 'D3vNP': ('D3', 'NP'), 'D1vD2': ('D1', 'D2'), 'D1vD3': ('D1', 'D3'), 'D2vNPx': ('D2', 'NPx'), 'D1vNPx': ('D1', 'NPx'), 'D1vREV': ('D1', 'REV'), 'D2vD3': ('D2', 'D3')}
STRATA = ('lin', 'net', 'rassoc', 'mshare', 'clean', 'lin_ds', 'K', 'pear')
def one_dataset(k):
    df = P[(P.grn == k[0]) & (P.topo == k[1]) & (P.cond == k[2]) & (P.sim == k[3])].reset_index(drop=True); base = dict(zip(KEY, k)); rows = []
    V = {c: df[c].values for c in SCOLS}; srcv, tv = df.src.values, df.tgt.values
    def add(name, pm, nm_, family, req=(), tgt=False, grp=None):
        pm = np.asarray(pm, bool); nm_ = np.asarray(nm_, bool); gv = srcv if grp is None else df[grp].values
        G = [(np.flatnonzero((gv == r) & pm), np.flatnonzero((gv == r) & nm_)) for r in np.unique(gv) if pm[gv == r].any() and nm_[gv == r].any() and all(np.asarray(q, bool)[gv == r].any() for q in req)]
        T_ = [(np.flatnonzero((tv == x) & pm), np.flatnonzero((tv == x) & nm_)) for x in np.unique(tv) if pm[tv == x].any() and nm_[tv == x].any()] if tgt else []
        sup = dict(n_groups=len(G), n_src_eligible=int(len({int(srcv[a[0]]) for a, _ in G})), n_pos_used=int(sum(len(a) for a, _ in G)), n_neg_used=int(sum(len(b) for _, b in G)))
        if grp is not None:
            lg = np.log(df.cgain.values); sup.update(gain_gap_matched=float(np.mean([lg[a].mean() - lg[b].mean() for a, b in G])) if G else np.nan, gain_gap_unmatched=float(lg[pm].mean() - lg[nm_].mean()))
        for c in SCOLS:
            v = V[c]; w = [auc(v[a], v[b]) for a, b in G]; wt = [auc(v[a], v[b]) for a, b in T_]
            rows.append(dict(base, family=family, contrast=name, method=c[2:], pooled=auc(v[pm], v[nm_]), within=float(np.mean(w)) if w else np.nan, n_reg=len(G),
                             within_tgt=float(np.mean(wt)) if wt else np.nan, n_tgt=len(T_), n_pos=int(pm.sum()), n_neg=int(nm_.sum()), **sup))
    for nm, (a, b) in CONTR.items(): add(nm, df[a], df[b], '1A', tgt=True)
    for st in STRATA:
        pool = df.clean if st == 'clean' else df.D2; S_, W_ = pool & (df[f'st_{st}'] == 'strong'), pool & (df[f'st_{st}'] == 'weak')
        for lev in ('strong', 'mid', 'weak'):
            m2 = pool & (df[f'st_{st}'] == lev); add(f'D1vD2{lev}', df.D1, m2, f'1B_{st}', tgt=(st == 'lin')); add(f'D2{lev}vNP', m2, df.NP, f'1B_{st}', tgt=(st == 'lin'))
        add('D1vD2strong_m', df.D1, S_, f'1B_{st}', req=(W_,)); add('D1vD2weak_m', df.D1, W_, f'1B_{st}', req=(S_,))
    for st in ('abs08', 'abs12'):
        S_, W_ = df.D2 & (df[f'st_{st}'] == 'strong'), df.D2 & (df[f'st_{st}'] == 'below')
        for lev, m2 in (('strong', S_), ('below', W_)): add(f'D1vD2{lev}', df.D1, m2, f'1B_{st}'); add(f'D2{lev}vNP', m2, df.NP, f'1B_{st}')
        add('D1vD2strong_m', df.D1, S_, f'1B_{st}', req=(W_,)); add('D1vD2below_m', df.D1, W_, f'1B_{st}', req=(S_,))
    for grp, levs in (('multi', ('single', 'multi')), ('hub', ('nonhub', 'hub'))):
        for lev in levs:
            m2 = df.D2 & (df[grp] == lev); add(f'D2{lev}vNP', m2, df.NP, f'1C_{grp}'); add(f'D1vD2{lev}', df.D1, m2, f'1C_{grp}')
    for comp in ('net+', 'net-', 'mixed'):
        m2 = df.D2 & ((df.sign_mixed == 1) if comp == 'mixed' else ((df.sign_mixed == 0) & (df.sign_net == (1 if comp == 'net+' else -1))))
        add(f'D2{comp}vNP', m2, df.NP, '1C_sign')
    add('D1vD2_gainmatched', df.D1, df.D2, '1D_gain', grp='g_src_bin'); add('D1vD2_gainlaymatched', df.D1, df.D2, '1D_gain', grp='g_src_bin_lay')
    return rows
with Pool(min(24, NDS)) as pool_: R = pd.DataFrame([r for rows in pool_.map(one_dataset, sorted(DSET)) for r in rows])
R.to_csv(OD + 'auc_datasets.tsv.gz', sep='\t', index=False)
# GRN-level machinery (strict: missing cells propagate)
def smean(s): return s.mean() if s.notna().all() and len(s) else np.nan
def grn_vec(df, val, cond=None):
    x = df if cond is None else df[df.cond == cond]
    return x.groupby(['grn', 'topo', 'cond', 'sim'])[val].agg(smean).groupby(['grn', 'topo', 'cond']).agg(smean).groupby(['grn', 'topo']).agg(smean).groupby('grn').agg(smean)
def summ(x):
    x = np.asarray(x, float); nmiss = int((~np.isfinite(x)).sum()); x = x[np.isfinite(x)]; n = len(x); out = dict(n_grn=n, n_grn_missing=nmiss, mean=float(x.mean()) if n else np.nan, lo=np.nan, hi=np.nan, sd=np.nan)
    if n >= 2 and not PART:
        h = stats.t.ppf(.975, n - 1) * x.std(ddof=1) / np.sqrt(n); out.update(lo=float(x.mean() - h), hi=float(x.mean() + h), sd=float(x.std(ddof=1)))
    return out
def signflip2(d):
    d = np.asarray(d, float)
    if PART or len(d) != 6 or not np.isfinite(d).all(): return np.nan
    obs = abs(d.mean()); return float(np.mean([abs((d * np.array(s)).mean()) >= obs - 1e-12 for s in itertools.product([1, -1], repeat=6)]))
CONDS = (None, 'C0', 'C1', 'C2', 'C3')
S1 = []
for (fam, con, meth), x in R.groupby(['family', 'contrast', 'method']):
    for ev in ('pooled', 'within', 'within_tgt'):
        if x[ev].isna().all(): continue
        for cond in CONDS:
            xc = x if cond is None else x[x.cond == cond]; v = grn_vec(x, ev, cond)
            S1.append(dict(family=fam, contrast=con, method=meth, eval=ev, cond=cond or 'ALL', **summ(v.values), grn_values=';'.join(f'{a:.4f}' for a in v.values), n_cells_missing=int(xc[ev].isna().sum()),
                           n_reg_mean=float(xc.n_reg.mean()), n_src_eligible_mean=float(xc.n_src_eligible.mean()), n_pos_used_mean=float(xc.n_pos_used.mean()), n_neg_used_mean=float(xc.n_neg_used.mean()),
                           gain_gap_matched=float(xc.gain_gap_matched.mean()) if 'gain_gap_matched' in xc and xc.gain_gap_matched.notna().any() else np.nan,
                           gain_gap_unmatched=float(xc.gain_gap_unmatched.mean()) if 'gain_gap_unmatched' in xc and xc.gain_gap_unmatched.notna().any() else np.nan, n_tgt_mean=float(xc.n_tgt.mean()), n_pos_mean=float(xc.n_pos.mean()), n_neg_mean=float(xc.n_neg.mean()), n_pos_min=int(xc.n_pos.min()), n_neg_min=int(xc.n_neg.min())))
S1 = pd.DataFrame(S1); S1.to_csv(OD + 'auc_grn_summary.tsv', sep='\t', index=False)
DF = []
for (fam, con), x in R.groupby(['family', 'contrast']):
    for ev in ('pooled', 'within', 'within_tgt'):
        if x[ev].isna().all(): continue
        for cond in CONDS:
            gb = grn_vec(x[x.method == 'GB'], ev, cond)
            for m in METH[1:]:
                d = (grn_vec(x[x.method == m], ev, cond) - gb)
                DF.append(dict(family=fam, contrast=con, method=m, eval=ev, cond=cond or 'ALL', **{f'diff_{a}': b for a, b in summ(d.values).items()}, n_pos_grn=int((d > 0).sum()), p_signflip_2s_unadj=signflip2(d.values)))
pd.DataFrame(DF).to_csv(OD + 'auc_diff_vs_GB.tsv', sep='\t', index=False)
WP = R[R.family == '1A'].assign(w_minus_p=lambda d: d.within - d.pooled)
pd.DataFrame([dict(contrast=c, method=m, **summ(grn_vec(x, 'w_minus_p').values)) for (c, m), x in WP.groupby(['contrast', 'method'])]).to_csv(OD + 'within_minus_pooled.tsv', sep='\t', index=False)
# 1B/1C counts and strength-measure validity
CN = []
for k, x in P[P.D2].groupby(KEY):
    d1 = P[P.D1 & (P.grn == k[0]) & (P.topo == k[1]) & (P.cond == k[2]) & (P.sim == k[3])]
    CN.append(dict(zip(KEY, k), n_d2=len(x), n_d1=len(d1), **{f'n_strong_{st}': int((x[f'st_{st}'] == 'strong').sum()) for st in STRATA + ('abs08', 'abs12')},
                   **{f'nreg_strong_{st}': int(x[x[f'st_{st}'] == 'strong'].src.nunique()) for st in ('lin', 'net', 'abs08', 'abs12')}, n_clean=int(x.clean.sum()),
                   gain_d2_median=float(x.gain_lin.median()), gain_d1_median=float(d1.gain_lin.median()), ratio_med_gross=float(x.gain_lin.median() / d1.gain_lin.median()),
                   ratio_med_net=float(x.gain_net_absmean.median() / d1.gain_lin.median()), frac_net_lt_half_gross=float((x.gain_net_absmean < .5 * x.gain_lin).mean()),
                   rassoc_d2_median=float(x.rassoc.median()), rassoc_d1_median=float(d1.rassoc.median()), mshare_d2_median=float(x.mshare.median()), mshare_d1_median=float(d1.mshare.median()),
                   n_multi=int((x.n_med >= 2).sum()), n_mixed=int(x.sign_mixed.sum()), n_med_max=int(x.n_med.max()), n_with_path3=int((x.n_paths3 > 0).sum()),
                   **{f'n_comp_{q}': int(x.comp.fillna('').str.contains(q).sum()) for q in ('AA', 'AR', 'RA', 'RR')},
                   tgt_det_d2_median=float(x.tgt_det.median()), med_det_min_median=float(x.med_det_min.median())))
CN = pd.DataFrame(CN); CN.to_csv(OD + 'B_counts_datasets.tsv', sep='\t', index=False)
CNg = pd.DataFrame([dict(quantity=q, cond=c or 'ALL', **summ(grn_vec(CN, q, c).values)) for q in CN.columns if q not in KEY for c in CONDS]); CNg.to_csv(OD + 'B_counts_grn.tsv', sep='\t', index=False)
VAL = []
for k, x in P.groupby(KEY):
    for cls in ('D1', 'D2'):
        y = x[x[cls]]
        for m in METH: VAL.append(dict(zip(KEY, k), cls=cls, method=m, **{f'rho_{q}': stats.spearmanr(y[q], y[f's|{m}']).correlation for q in ('gain_lin', 'gain_K', 'rassoc', 'mshare') + (('gain_net_absmean',) if cls == 'D2' else ())}))
VAL = pd.DataFrame(VAL); VAL.to_csv(OD + 'B_gain_validity_datasets.tsv', sep='\t', index=False)
pd.DataFrame([dict(cls=c, method=m, measure=q[4:], **summ(grn_vec(x, q).values)) for (c, m), x in VAL.groupby(['cls', 'method']) for q in VAL.columns if q.startswith('rho_') and x[q].notna().any()]
             ).to_csv(OD + 'B_gain_validity_grn.tsv', sep='\t', index=False)
# 2A attribution (D1 vs D2): production vs K0 vs numerator-only vs denominator-only; symmetric phi and the explicit interaction
AT = []
for ev in ('pooled', 'within'):
    x = R[(R.family == '1A') & (R.contrast == 'D1vD2')].pivot_table(index=KEY, columns='method', values=ev)
    x['phiN'] = .5 * ((x.GB_NUMONLY - x.GB_K0) + (x.GB - x.GB_DENONLY)); x['phiD'] = .5 * ((x.GB_DENONLY - x.GB_K0) + (x.GB - x.GB_NUMONLY))
    x['interaction'] = x.GB - x.GB_NUMONLY - x.GB_DENONLY + x.GB_K0; x['full'] = x.GB - x.GB_K0; x['numonly'] = x.GB_NUMONLY - x.GB_K0; x['denonly'] = x.GB_DENONLY - x.GB_K0
    x['native_full'] = x.GBraw - x.GBraw_K0; x = x.reset_index()
    for q in ('full', 'numonly', 'denonly', 'phiN', 'phiD', 'interaction', 'native_full'):
        for cond in CONDS:
            v = grn_vec(x, q, cond); AT.append(dict(eval=ev, quantity=q, cond=cond or 'ALL', **summ(v.values), n_pos_grn=int((v > 0).sum()), p_signflip_2s_unadj=signflip2(v.values)))
AT = pd.DataFrame(AT); AT.to_csv(OD + 'A2_attribution.tsv', sep='\t', index=False)
# 2B residual diagnostics by class (regulator-first), U and X; residual sign faithfulness (D1 vs true sign, D2 vs net path sign)
DG = ['sgncons', 'rootagree8', 'dlognum8', 'dlogden8', 'logden0', 'zsd_depth', 'zdiff_root']
for ch in ('U', 'X'): P[f'{ch}_dlogS8'] = P[f'{ch}_dlognum8'] - P[f'{ch}_dlogden8']; P[f'{ch}_faith0'] = (P[f'{ch}_sgn0'] == P.sign_net).astype(float); P[f'{ch}_faith8'] = (P[f'{ch}_sgn8'] == P.sign_net).astype(float)
DG += ['dlogS8']; CLS = ['D1', 'D2', 'D3', 'REV', 'NPx']; RB = []
for k, x in P.groupby(KEY):
    for ch in ('U', 'X'):
        for cls in CLS:
            y = x[x[cls]].groupby('src')[[f'{ch}_{q}' for q in DG]].mean().mean(); RB.append(dict(zip(KEY, k), ch=ch, cls=cls, **{q: float(y[f'{ch}_{q}']) for q in DG}))
        for cls in ('D1', 'D2'):
            y = x[x[cls] & x.sign_net.notna() & (x.sign_net != 0)]; RB.append(dict(zip(KEY, k), ch=ch, cls=f'FAITH_{cls}', faith0=float(y[f'{ch}_faith0'].mean()), faith8=float(y[f'{ch}_faith8'].mean())))
        for nm, (a, b) in (('D1vD2', ('D1', 'D2')), ('D1vNPx', ('D1', 'NPx')), ('D2vNPx', ('D2', 'NPx'))):
            srcv = x.src.values; row = dict(zip(KEY, k), ch=ch, cls=f'AUC_{nm}')
            for q in DG:
                w = [auc(x[f'{ch}_{q}'].values[(srcv == r) & x[a].values], x[f'{ch}_{q}'].values[(srcv == r) & x[b].values]) for r in np.unique(srcv) if (x[a].values[srcv == r]).any() and (x[b].values[srcv == r]).any()]
                row[q] = float(np.nanmean(w)) if w else np.nan
            RB.append(row)
RB = pd.DataFrame(RB); RB.to_csv(OD + 'B2_residual_datasets.tsv', sep='\t', index=False)
RS = []
for (ch, cls), x in RB.groupby(['ch', 'cls']):
    for q in DG + ['faith0', 'faith8']:
        if q in x and x[q].notna().any():
            for cond in CONDS:
                for tp in (None, 'bifurcation', 'tree'):
                    xx = x if tp is None else x[x.topo == tp]; RS.append(dict(ch=ch, cls=cls, diag=q, cond=cond or 'ALL', topo=tp or 'ALL', **summ(grn_vec(xx, q, cond).values)))
pd.DataFrame(RS).to_csv(OD + 'B2_residual_grn.tsv', sep='\t', index=False)
# 2C paired confounding summary (GB production vs K0, within-regulator D1vD2), direct percentile, Hill operating-point gain (truth-side)
W = R[(R.family == '1A') & (R.contrast == 'D1vD2')].pivot_table(index=KEY, columns='method', values='within').reset_index()
for m in ('GB', 'GB_K0'): P[f'pct|{m}'] = P.groupby(KEY + ['src'])[f's|{m}'].rank(pct=True)
PC = P[P.D1].groupby(KEY + ['src'])[['pct|GB', 'pct|GB_K0', 'gain_lin']].mean().groupby(KEY).mean().reset_index(); W = W.merge(PC, on=KEY)
C2 = []
for grn, x in W.groupby('grn'):
    for tp, xt in x.groupby('topo'):
        cm = xt.groupby('cond')[['GB', 'GB_K0', 'pct|GB', 'pct|GB_K0', 'gain_lin']].mean()
        for c in ('C1', 'C2', 'C3'):
            C2.append(dict(grn=grn, topo=tp, cond=c, L=cm.GB_K0['C0'] - cm.GB_K0[c], R=cm.GB[c] - cm.GB_K0[c], R_C0=cm.GB['C0'] - cm.GB_K0['C0'], dR=(cm.GB[c] - cm.GB_K0[c]) - (cm.GB['C0'] - cm.GB_K0['C0']),
                           unrecovered=cm.GB['C0'] - cm.GB[c], pctD1_K0_C0=cm['pct|GB_K0']['C0'], pctD1_K0_c=cm['pct|GB_K0'][c], pctD1_P_C0=cm['pct|GB']['C0'], pctD1_P_c=cm['pct|GB'][c],
                           hill_gain_ratio=cm.gain_lin[c] / cm.gain_lin['C0']))
C2 = pd.DataFrame(C2); C2.to_csv(OD + 'C2_paired_confounding_grn_topo.tsv', sep='\t', index=False)
pd.DataFrame([dict(quantity=q, cond=c, **summ(C2[C2.cond == c].groupby(['grn', 'topo'])[q].mean().groupby('grn').mean().values)) for q in C2.columns[3:] for c in ('C1', 'C2', 'C3')]
             + [dict(quantity=q, cond='C1-C3', **summ(C2.groupby(['grn', 'topo'])[q].mean().groupby('grn').mean().values)) for q in C2.columns[3:]]).to_csv(OD + 'C2_paired_confounding_grn.tsv', sep='\t', index=False)
# 3A reverse-class audit
RA = []
for k, x in P.groupby(KEY):
    rv = x[x.REV]; RA.append(dict(zip(KEY, k), n_rev=len(rv), rev_also_d_lt99=int((rv.d < 99).sum()), rev_tgt_is_reg=int(rv.tgt_is_reg.sum()), n_revpath_only=int(((x.drev >= 2) & (x.drev < 99)).sum()),
                                  revpath_and_d_lt99=int(((x.drev < 99) & (x.d < 99)).sum()), n_NPx=int(x.NPx.sum()), n_NP=int(x.NP.sum()), n_D1_regreg=int((x.D1 & (x.tgt_is_reg == 1)).sum())))
pd.DataFrame(RA).to_csv(OD + 'A3_reverse_audit.tsv', sep='\t', index=False)
OR = []
for k, x in P.groupby(KEY):
    x = x[x.tgt_is_reg == 1].copy()
    for m in METH: x[f'p|{m}'] = x.groupby('src')[f's|{m}'].rank(pct=True)
    y = x.merge(x, left_on=['src', 'tgt'], right_on=['tgt', 'src'], suffixes=('', '_r')); edge = y[y.d == 1]; rest = y[(y.d != 1) & (y.rev == 0)]
    for m in METH:
        for typ, pre_ in (('score', 's|'), ('pct', 'p|')):
            tol = 1e-9 * np.nanmax(np.abs(y[f'{pre_}{m}'].values)); De = (edge[f'{pre_}{m}'] - edge[f'{pre_}{m}_r']).values; De = np.where(np.abs(De) <= tol, 0, De)
            Dr = (rest[f'{pre_}{m}'] - rest[f'{pre_}{m}_r']).values; Dr = np.where(np.abs(Dr) <= tol, 0, Dr)
            OR.append(dict(zip(KEY, k), method=m, scale=typ, n_edges=len(edge), n_rest=len(rest), orient_acc=float(((De > 0) + .5 * (De == 0)).mean()), n_ties=int((De == 0).sum()),
                           D_edge_vs_rest_auc=auc(De, Dr), F_edge_vs_rest_auc=auc(edge[f'{pre_}{m}'].values, rest[f'{pre_}{m}'].values)))
OR = pd.DataFrame(OR); OR.to_csv(OD + 'C3_orientation_datasets.tsv', sep='\t', index=False)
xc = OR[(OR.scale == 'score') & OR.method.isin(['GBXraw', 'GBXraw_K0', 'pearson'])]
assert (np.abs(xc.orient_acc - .5) < 1e-12).all() and (np.abs(xc.D_edge_vs_rest_auc - .5) < 1e-12).all(), xc[np.abs(xc.orient_acc - .5) >= 1e-12]
OS = []
for (m, typ), x in OR.groupby(['method', 'scale']):
    for q in ('orient_acc', 'D_edge_vs_rest_auc', 'F_edge_vs_rest_auc'):
        for cond in CONDS:
            v = grn_vec(x, q, cond); OS.append(dict(method=m, scale=typ, quantity=q, cond=cond or 'ALL', **summ(v.values), n_grn_gt_half=int((v > .5).sum()), p_vs_half_2s_unadj=signflip2(v.values - .5)))
    v = grn_vec(x.assign(dmf=x.D_edge_vs_rest_auc - x.F_edge_vs_rest_auc), 'dmf'); OS.append(dict(method=m, scale=typ, quantity='D_minus_F', cond='ALL', **summ(v.values), n_grn_gt_half=int((v > 0).sum()), p_vs_half_2s_unadj=signflip2(v.values)))
OS = pd.DataFrame(OS); OS.to_csv(OD + 'C3_orientation_grn.tsv', sep='\t', index=False)
# conditioning x channel difference-in-differences on identical reciprocal edges (orientation) and
# on the D1vREV / D1vNPx ranking endpoints
DD = []
O2 = OR.pivot_table(index=KEY + ['scale'], columns='method', values='orient_acc').reset_index()
for typ, x in O2.groupby('scale'):
    for nmU, nmX in (('GB', 'GBX'), ('GBraw', 'GBXraw')):
        x = x.assign(gU=x[nmU] - x[f'{nmU}_K0'], gX=x[nmX] - x[f'{nmX}_K0']); x = x.assign(did=x.gU - x.gX, UmX=x[nmU] - x[nmX], UmX_K0=x[f'{nmU}_K0'] - x[f'{nmX}_K0'])
        for q in ('gU', 'gX', 'did', 'UmX', 'UmX_K0'):
            for cond in CONDS:
                v = grn_vec(x, q, cond); DD.append(dict(endpoint='orient_acc', scale=typ, pair=f'{nmU}/{nmX}', quantity=q, cond=cond or 'ALL', **summ(v.values), n_pos_grn=int((v > 0).sum()), p_signflip_2s_unadj=signflip2(v.values)))
for con in ('D1vREV', 'D1vNPx', 'D1vD2'):
    x = R[(R.family == '1A') & (R.contrast == con)].pivot_table(index=KEY, columns='method', values='within').reset_index()
    for nmU, nmX in (('GB', 'GBX'), ('GBraw', 'GBXraw')):
        x = x.assign(gU=x[nmU] - x[f'{nmU}_K0'], gX=x[nmX] - x[f'{nmX}_K0']); x = x.assign(did=x.gU - x.gX)
        for q in ('gU', 'gX', 'did'):
            for cond in CONDS:
                v = grn_vec(x, q, cond); DD.append(dict(endpoint=f'{con}_within', scale='score', pair=f'{nmU}/{nmX}', quantity=q, cond=cond or 'ALL', **summ(v.values), n_pos_grn=int((v > 0).sum()), p_signflip_2s_unadj=signflip2(v.values)))
DD = pd.DataFrame(DD); DD.to_csv(OD + 'C3_orientation_did.tsv', sep='\t', index=False)
json.dump(dict(aggregator_sha=sh_(os.path.abspath(__file__)), partial=PART, n_datasets=NDS, grns=[int(g) for g in GR], scorer_script_sha=next(iter(RC.values()))['script_sha'],
               inputs={os.path.basename(f): sh_(f) for f in F}, scorer_receipts=RC, outputs={os.path.basename(f): sh_(f) for f in sorted(glob.glob(OD + '*.tsv'))}), open(OD + 'agg_receipt.json', 'w'), indent=1)
# console summary
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500); pd.set_option('display.max_columns', 60)
s = S1[(S1.family == '1A') & (S1.cond == 'ALL')].pivot_table(index='method', columns=['contrast', 'eval'], values='mean')
print(s[[c for c in s.columns if c[0] in ('D1vNP', 'D2vNP', 'D3vNP', 'D1vD2', 'D1vD3', 'D1vREV')]].round(3).to_string())
print(CNg[CNg.cond == 'ALL'][['quantity', 'mean', 'lo', 'hi']].round(3).to_string())
s = S1[(S1.family.str.startswith('1B') | S1.family.str.startswith('1D')) & (S1.cond == 'ALL') & (S1['eval'] == 'within')]
print(s[s.method.isin(['GB', 'GB_K0', 'VEL', 'genie3', 'grnboost2', 'pearson', 'pidc'])].pivot_table(index=['family', 'contrast'], columns='method', values='mean').round(3).to_string())
print(AT[AT.cond == 'ALL'].round(4).to_string())
print(OS[(OS.cond == 'ALL') & (OS.scale == 'score')].pivot_table(index='method', columns='quantity', values='mean').round(3).to_string())
print(DD[DD.cond == 'ALL'][['endpoint', 'scale', 'pair', 'quantity', 'mean', 'lo', 'hi', 'n_pos_grn', 'p_signflip_2s_unadj']].round(4).to_string())
print('done', NDS, 'datasets ->', OD)
