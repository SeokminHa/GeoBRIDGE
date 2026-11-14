# Direction and coverage checks for gb_fig3_endpoints.py.
import sys, os, json
import numpy as np, pandas as pd
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
BASE = R + 'scratch/gb_fig3_endpoint_reference.py'
STORED = R + 'results/figures/manuscript_2026_10_19/'
a = sys.argv; OUT = a[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
B = int(a[a.index('--B') + 1]) if '--B' in a else 1000
MODE = a[a.index('--mode') + 1]
sys.path[:0] = [R + 'scratch', R + 'pairspace']
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 600)

if MODE == 'compare':
    o = pd.read_csv(f'{OUT}B1000_nofix/paired_diffs.tsv', sep='\t'); n = pd.read_csv(f'{OUT}B1000_fix/paired_diffs.tsv', sep='\t')
    k = ['set', 'scope', 'scheme', 'direction', 'metric', 'external']
    m = o.merge(n, on=k, suffixes=('_before', '_after'))
    m = m[(m.set == 'SET9') & (m.scope == 'directional') & (m.scheme.isin(['raw', 'primary'])) & m.direction.isin(['KOup', 'KOdown'])]
    m['d_point'] = m.point_after - m.point_before
    m = m[k + ['GB_before', 'GB_after', 'ext_before', 'ext_after', 'point_before', 'lo_before', 'hi_before', 'p_before', 'p_holm8_before', 'point_after', 'lo_after', 'hi_after', 'p_after', 'p_holm8_after', 'd_point']]
    m.to_csv(OUT + 'compare_fix_paired.tsv', sep='\t', index=False)
    eo = pd.read_csv(f'{OUT}B1000_nofix/equal_system.tsv', sep='\t'); en = pd.read_csv(f'{OUT}B1000_fix/equal_system.tsv', sep='\t')
    k2 = ['set', 'scope', 'scheme', 'direction', 'metric', 'method']
    e = eo.merge(en, on=k2, suffixes=('_before', '_after')); e = e[(e.scope == 'directional') & (e.set == 'SET9')]
    e.to_csv(OUT + 'compare_fix_equal_system.tsv', sep='\t', index=False)
    print(m[m.external.isin(['Velorama', 'RegDiffusion'])].round(4).to_string(index=False))
    sys.exit(0)

import gb_fig3d_magnitude as GM
EFF0 = GM.effect
SRC = open(BASE).read()


# PANC: same 1:1 HOM orthology + same ambiguous-symbol drop as the responder labels
def effect_fixed(system, tf):
    if system != 'PANC': return EFF0(system, tf)
    de = pd.read_csv(R + 'real_1_Pancreas/external_validation/aldous_rfx6/results/aldous_PPs_WT_vs_KO_DE.tsv', sep='\t')
    hom = pd.read_csv(R + 'data/orthology/HOM_MouseHumanSequence.rpt', sep='\t', low_memory=False)
    key = hom.columns[0]; org = hom['Common Organism Name']
    mm = hom[org.str.startswith('mouse')].groupby(key)['Symbol'].apply(list); hs = hom[org.str.startswith('human')].groupby(key)['Symbol'].apply(list)
    h2m = {hs[k][0].upper(): mm[k][0] for k in set(mm.index) & set(hs.index) if len(mm[k]) == 1 and len(hs[k]) == 1}
    dsym = de.gene_name.astype(str).str.upper(); dup = set(dsym[dsym.duplicated(keep=False)]); out = {}
    for hg, l in zip(dsym, de.log2FoldChange):
        m = h2m.get(hg)
        if m is None or hg in dup: continue
        m = m.upper()
        if m in out and out[m] != l: raise SystemExit('REFUSING: conflicting DE effects map to ' + m)
        out[m] = l
    return pd.Series(out, dtype=float)


def run_base(out, fix, fullcov):
    src = SRC
    if fullcov:
        pats = [("MS = list(METH) if M10 else [m for m in METH if m != 'RegVelo |J|']", "MS = list(METH) if M10 else [m for m in METH if m not in ('RegVelo |J|', 'Velorama')]"),
                ("for ex in ('Velorama', 'RegDiffusion'):", "for ex in ('RegDiffusion',):"),
                ("assert cov['cov_raw_det_u'] == 1.0, 'STOP: primary covariate coverage < 100%'", "print('FULLCOV: cov_raw_det_u =', cov['cov_raw_det_u'])"),
                ("S = {'raw': None, 'primary': lab_(lambda m: qb(d[m], 5), U)}", "S = {'raw': None, 'primary': lab_(lambda m: qb(d[m], 5), U & np.isfinite(d))}")]
        for o, n in pats:
            assert src.count(o) >= 1, 'PATCH MISS: ' + o
            src = src.replace(o, n)
    sys.argv = ['base', out, '--B', str(B)]
    ns = {'__name__': '__base__'}
    GM.effect = effect_fixed if fix else EFF0
    try: exec(compile(src, BASE, 'exec'), ns)
    finally: GM.effect = EFF0
    return ns


# max |diff| of every numeric column, exact equality of others, row-aligned
def cmp_dir(out, ref):
    rep = []
    for f in sorted(os.listdir(ref)):
        if not f.endswith('.tsv') or f.startswith('parity'): continue
        if not os.path.exists(out + f): rep.append(dict(file=f, status='MISSING', maxabs=np.inf)); continue
        x = pd.read_csv(ref + f, sep='\t'); y = pd.read_csv(out + f, sep='\t')
        if x.shape != y.shape or list(x.columns) != list(y.columns): rep.append(dict(file=f, status='SHAPE ' + str(x.shape) + ' vs ' + str(y.shape), maxabs=np.inf)); continue
        mx = 0.0; bad = []
        for c in x.columns:
            if pd.api.types.is_numeric_dtype(x[c]):
                dd = (x[c].astype(float) - y[c].astype(float)).abs(); both_nan = x[c].isna() & y[c].isna()
                dd = dd.where(~both_nan, 0.0)
                if dd.isna().any(): bad.append(c); mx = np.inf
                else: mx = max(mx, float(dd.max()) if len(dd) else 0.0)
            elif not x[c].fillna('').astype(str).equals(y[c].fillna('').astype(str)): bad.append(c); mx = np.inf
        rep.append(dict(file=f, status='OK' if not bad else 'DIFF ' + ','.join(bad), maxabs=mx))
    return pd.DataFrame(rep)


# analysis helpers (operate on the exec'd base namespace)
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for i, k in enumerate(o): run = max(run, (len(p) - i) * p[k]); adj[k] = min(1, run)
    return adj


# equal-system levels + GB-external paired diffs (Holm over externals) + production-k0 depth gain, on endpoint subset eps and n_pos gate
def sens_block(ns, name, eps, gate=5, schemes=('raw', 'primary'), dirs=('ALL', 'KOup', 'KOdown'), skip=()):
    agg, RES, RAP, META, MS, EXT = ns['agg'], ns['RES'], ns['RAP'], ns['META'], ns['MS'], ns['EXT']
    ci, row = ns['ci'], ns['row']
    npos = {(r.direction, r.endpoint): r.n_pos for r in META.itertuples()}
    def okdep(d, lab): return npos[(d, lab)] >= gate and (d, lab) not in skip
    LV, PD, DGs, INC = [], [], [], []
    for d in dirs:
        for sch in (schemes if d != 'ALL' else ('raw',)):
            for mt in (('AUROC', 'APLIFT') if d == 'ALL' else ('AUROC',)):
                if d == 'ALL': get = lambda m: (lambda lab: RAP[(lab, m)][mt] if (lab, m) in RAP and okdep('ALL', lab) else None)
                else: get = lambda m: (lambda lab: RES.get((sch, d, lab, m)) if okdep(d, lab) else None)
                ms = [m for m in MS + ['GB_k0', 'GB_prod@k0univ'] if agg(get(m), eps)[0] is not None]
                if 'GeoBRIDGE' not in ms: continue
                for m in ms:
                    r, ps = agg(get(m), eps); lo, hi = ci(r); LV.append(dict(sens=name, direction=d, scheme=sch, metric=mt, method=m, value=r[0], lo=lo, hi=hi, n_systems=len(ps), systems=','.join(ps)))
                inc = [l for l in eps if get('GeoBRIDGE')(l) is not None]; INC.append(dict(sens=name, direction=d, scheme=sch, metric=mt, endpoints=','.join(inc), n_endpoints=len(inc), gate=gate))
                gb = agg(get('GeoBRIDGE'), eps)[0]; res = []
                for m in EXT:
                    e_ = agg(get(m), eps)[0]
                    if e_ is None: continue
                    res.append(row(gb - e_, sens=name, direction=d, scheme=sch, metric=mt, external=m, GB=gb[0], ext=e_[0]))
                for r_, h in zip(res, holm([r['p'] for r in res])): r_['p_holm'] = h; r_['holm_family'] = len(res)
                PD += res
                if 'GB_k0' in ms and 'GB_prod@k0univ' in ms:
                    DGs.append(row(agg(get('GB_prod@k0univ'), eps)[0] - agg(get('GB_k0'), eps)[0], sens=name, direction=d, scheme=sch, metric=mt, contrast='DEPTHGAIN(prod-k0)'))
    return LV, PD, DGs, INC


def write_sens(out, fname, parts):
    for k, idx in (('levels', 0), ('paired', 1), ('depthgain', 2), ('included', 3)):
        pd.DataFrame([r for p in parts for r in p[idx]]).to_csv(f'{out}{fname}_{k}.tsv', sep='\t', index=False)


def receipts(ns, out):
    EPD = ns['EPD']; rows, summ = [], []
    for lab, D in EPD.items():
        y, U, e, gi, C = D['y'], D['U'], D['e'], D['gi'], D['C']; Cg = C[:, gi].astype(float)
        for nm, yl in (('ALL', y.copy()), ('KOup', y & np.isfinite(e) & (e > 0)), ('KOdown', y & np.isfinite(e) & (e < 0))):
            P0, N0 = int((yl & U).sum()), int((~y & U).sum())
            for sch, bins in D['S'].items():
                if sch == 'raw':
                    summ.append(dict(endpoint=lab, direction=nm, scheme='raw', n_pos_before=P0, n_neg_before=N0, n_pos_retained=P0, n_neg_retained=N0, n_valid_bins='NA(no matching)', n_contributing_bins='NA', n_pos_unmatched_dropped=0, n_neg_unmatched_dropped=0))
                    continue
                if nm == 'ALL': continue
                wb = {}
                for b in np.unique(bins[U & (bins >= 0)]):
                    rb = U & (bins == b); Pm, Nm = rb & yl, rb & ~y; p_, n_ = int(Pm.sum()), int(Nm.sum()); ok = p_ >= 3 and n_ >= 3
                    wb[int(b)] = (p_, n_, ok, float(p_ * n_) if ok else 0.0, Cg[:, Pm].sum(1) * Cg[:, Nm].sum(1) if ok else np.zeros(C.shape[0]))
                tp = sum(v[3] for v in wb.values()); tb = np.sum([v[4] for v in wb.values()], 0) if wb else np.zeros(C.shape[0])
                for b, (p_, n_, ok, wpt, wbt) in wb.items():
                    with np.errstate(all='ignore'): wbm = float(np.nanmean(np.where(tb > 0, wbt / np.where(tb > 0, tb, 1), np.nan))) if ok else 0.0
                    rows.append(dict(endpoint=lab, direction=nm, scheme=sch, bin=b, n_pos=p_, n_neg=n_, valid=ok, w_point=wpt / tp if tp > 0 else 0.0, w_boot_mean=wbm))
                Pr = sum(v[0] for v in wb.values() if v[2]); Nr = sum(v[1] for v in wb.values() if v[2]); nvb = sum(v[2] for v in wb.values())
                summ.append(dict(endpoint=lab, direction=nm, scheme=sch, n_pos_before=P0, n_neg_before=N0, n_pos_retained=Pr, n_neg_retained=Nr, n_valid_bins=nvb, n_contributing_bins=sum(1 for v in wb.values() if v[3] > 0),
                                 n_pos_unmatched_dropped=P0 - Pr, n_neg_unmatched_dropped=N0 - Nr, n_bins_total=len(wb), matched_undefined=bool(nvb == 0)))
    pd.DataFrame(rows).to_csv(out + 'receipts_bins.tsv', sep='\t', index=False); S_ = pd.DataFrame(summ); S_.to_csv(out + 'receipts_summary.tsv', sep='\t', index=False)
    return S_


# GB vs leave-focal-out target propensity P_pairU (SI internal reference, not a competitor), as gb_ms_fig3_kodir_ext_2026_10_19.py: rows = U & finite(P)
def pcontrol(ns, out, eps_labs, tag):
    agg, EPD, strat, ci, row, VEC, up = ns['agg'], ns['EPD'], ns['strat'], ns['ci'], ns['row'], ns['VEC'], ns['up']
    EP9 = {x[0]: x for x in ns['EP9']}; PR, GR, NP = {}, {}, {}
    for lab in eps_labs:
        D = EPD[lab]; z = np.load(f"{VEC}ep_{EP9[lab][5]}.vec.npz", allow_pickle=True); assert (up(z['genes']) == D['g']).all()
        P = z['P_pairU'].astype(float); fin = np.isfinite(P); y, U, e, gi, C = D['y'], D['U'], D['e'], D['gi'], D['C']; Cg = C[:, gi].astype(float); gbv = D['V']['GeoBRIDGE']
        for nm, yl in (('ALL', y.copy()), ('KOup', y & np.isfinite(e) & (e > 0)), ('KOdown', y & np.isfinite(e) & (e < 0))):
            rr = U & fin & (yl | ~y); n = int((yl & U & fin).sum()); NP[(nm, lab)] = (n, int((yl & U).sum()), int((U & ~fin).sum()))
            if n < 5: continue
            for sch, bins in D['S'].items():
                if sch not in ('raw', 'primary') or (nm == 'ALL' and sch != 'raw'): continue
                g_ = strat(gbv, yl, rr, bins, gi, C, Cg, [])
                if not np.isfinite(g_[0]): continue          # matched undefined for GB (as base undef)
                PR[(sch, nm, lab)] = strat(P, yl, rr, bins, gi, C, Cg, []); GR[(sch, nm, lab)] = g_
    LV, PD, PE = [], [], []
    for nm in ('ALL', 'KOup', 'KOdown'):
        for sch in (('raw',) if nm == 'ALL' else ('raw', 'primary')):
            labs = [l for l in eps_labs if (sch, nm, l) in PR]
            if not labs: continue
            p_, ps = agg(lambda l: PR.get((sch, nm, l)), labs); g_, gs = agg(lambda l: GR.get((sch, nm, l)), labs)
            lo, hi = ci(p_); LV.append(dict(tag=tag, direction=nm, scheme=sch, ref='P', value=p_[0], lo=lo, hi=hi, n_systems=len(ps), endpoints=','.join(labs), GB_same_rows=g_[0]))
            PD.append(row(g_ - p_, tag=tag, direction=nm, scheme=sch, contrast='GB-P', scope='equal_system', GB=g_[0], P=p_[0], n_systems=len(ps), endpoints=','.join(labs)))
            for s in ps: PD.append(row(gs[s] - ps[s], tag=tag, direction=nm, scheme=sch, contrast='GB-P', scope=s, GB=gs[s][0], P=ps[s][0], n_systems=1, endpoints=''))
            for l in labs: PE.append(row(GR[(sch, nm, l)] - PR[(sch, nm, l)], tag=tag, direction=nm, scheme=sch, contrast='GB-P', scope=l, GB=GR[(sch, nm, l)][0], P=PR[(sch, nm, l)][0], n_systems=1, endpoints=''))
    pd.DataFrame(LV).to_csv(f'{out}pcontrol_{tag}_levels.tsv', sep='\t', index=False); pd.DataFrame(PD + PE).to_csv(f'{out}pcontrol_{tag}_paired.tsv', sep='\t', index=False)
    pd.DataFrame([dict(direction=k[0], endpoint=k[1], n_pos_U_finiteP=v[0], n_pos_U=v[1], n_U_nonfiniteP=v[2]) for k, v in NP.items()]).to_csv(f'{out}pcontrol_{tag}_coverage.tsv', sep='\t', index=False)
    return pd.DataFrame(LV), pd.DataFrame(PD)


# (d) production-only all-responder sensitivity for BM_NFE2_cal / BM_TAL1_cal
def nfe2_tal1(ns, out):
    L, PT, CT, up, VEC, MS, EXT, drc, B_ = ns['L'], ns['PT'], ns['CT'], ns['up'], ns['VEC'], ns['MS'], ns['EXT'], ns['drc'], ns['B']
    agg, ci, row, RAP = ns['agg'], ns['ci'], ns['row'], ns['RAP']; labs = []; info = []
    for lab, sysn, esys, tf, dsys, stem in ns['EPV']:
        z = np.load(f'{VEC}ep_{stem}.vec.npz', allow_pickle=True); g = up(z['genes']); gi = z['gi']; y = z['y'].astype(bool)
        V = {m: z[f'cog::{m}'].astype(float) for m in MS}; allfin = np.all([np.isfinite(V[m]) for m in MS], 0)
        dup = pd.Series(g).duplicated(keep=False).to_numpy(); selfm = (g == tf.upper()); cap = {}; orig = L.assemble
        def _a(s_, t_, g_, V_, truth_, *aa, **kk): cap['t'] = truth_; return orig(s_, t_, g_, V_, truth_, *aa, **kk)
        L.assemble = _a
        try: PT.lattice_unit(dsys, tf, drc[dsys].gene)
        finally: L.assemble = orig
        rok = np.asarray(cap['t']['resp_ok'], bool); U = allfin & rok & ~dup & ~selfm; C = drc[dsys].C[B_:2 * B_]
        info.append(dict(endpoint=lab, n_y_total=int(y.sum()), n_U=int(U.sum()), n_pos_U=int((y & U).sum()), n_pos_outside_respok=int(y.sum() - y[rok].sum())))
        ns['SYSOF'][lab] = sysn; ns['TFOF'][lab] = tf; labs.append(lab)
        for m in MS:
            pt, bs = CT.eval_rows(V[m], y, U, gi, C); RAP[(lab, m)] = {mt: np.r_[pt[mt], np.asarray(bs[mt], float)] for mt in ('AUROC', 'APLIFT')}
    pd.DataFrame(info).to_csv(out + 'sens_d_nfe2_tal1_info.tsv', sep='\t', index=False)
    eps9 = [x[0] for x in ns['EP9']]; LV, PD = [], []
    for scope, eps in (('SET9+NFE2+TAL1', eps9 + labs), ('NFE2_only', ['BM_NFE2_cal']), ('TAL1_only', ['BM_TAL1_cal']), ('BM_cal_pair', ['BM_NFE2_cal', 'BM_TAL1_cal'])):
        for mt in ('AUROC', 'APLIFT'):
            get = lambda m: (lambda lab: RAP[(lab, m)][mt] if (lab, m) in RAP else None)
            for m in MS:
                r, ps = agg(get(m), eps); lo, hi = ci(r); LV.append(dict(scope=scope, metric=mt, method=m, value=r[0], lo=lo, hi=hi, n_systems=len(ps)))
            gb = agg(get('GeoBRIDGE'), eps)[0]; res = [row(gb - agg(get(m), eps)[0], scope=scope, metric=mt, external=m, GB=gb[0], ext=agg(get(m), eps)[0][0]) for m in EXT]
            for r_, h in zip(res, holm([r['p'] for r in res])): r_['p_holm'] = h
            PD += res
    pd.DataFrame(LV).to_csv(out + 'sens_d_nfe2_tal1_levels.tsv', sep='\t', index=False); pd.DataFrame(PD).to_csv(out + 'sens_d_nfe2_tal1_paired.tsv', sep='\t', index=False)
    return pd.DataFrame(info), pd.DataFrame(PD)


# modes
if MODE == 'nofix':
    ns = run_base(OUT, fix=False, fullcov=False)
    ref = f'{STORED}fig3_set9/B{B}/'
    rep = cmp_dir(OUT, ref); rep.to_csv(OUT + 'parity_vs_stored_set9.tsv', sep='\t', index=False); print(rep.to_string(index=False))
    assert rep.status.eq('OK').all() and rep.maxabs.max() < 1e-12, 'PARITY vs stored SET9 FAILED'
    if B == 1000:   # P-control machinery parity vs stored kodir_ext (SET6 subset, raw; unfixed effect => same labels as kodir_ext)
        LV, PD = pcontrol(ns, OUT, [x[0] for x in ns['EP6']], 'SET6_parity')
        Kb = STORED + 'fig3_kodir_ext/'; bl = pd.read_csv(Kb + 'B_refs_equal_system.tsv', sep='\t'); bp = pd.read_csv(Kb + 'B_paired_GB_minus_ref.tsv', sep='\t'); mx = 0.0
        for d in ('KOup', 'KOdown'):
            l = LV[(LV.direction == d) & (LV.scheme == 'raw')].iloc[0]; s = bl[(bl.direction == d) & (bl.ref == 'P')].iloc[0]
            p = PD[(PD.direction == d) & (PD.scheme == 'raw') & (PD.scope == 'equal_system')].iloc[0]; t = bp[(bp.direction == d) & (bp.ref == 'P')].iloc[0]
            dd = [abs(l.value - s.AUROC), abs(l.lo - s.lo), abs(l.hi - s.hi), abs(l.GB_same_rows - s.GB_same_rows), abs(p['point'] - t['point']), abs(p.lo - t.lo), abs(p.hi - t.hi), abs(p.p - t.p)]
            print('PCTRL parity', d, max(dd)); mx = max(mx, max(dd))
        json.dump(dict(pcontrol_parity_maxabs=mx), open(OUT + 'pcontrol_parity.json', 'w')); assert mx < 1e-12, 'P-control parity FAILED'
    print('NOFIX PARITY OK')
elif MODE == 'fix':
    ns = run_base(OUT, fix=True, fullcov=False)
    EPD, META = ns['EPD'], ns['META']; D = EPD['PANC_RFX6']; g, y, U, e_new = D['g'], D['y'], D['U'], D['e']
    old = EFF0('PANC', 'RFX6'); e_old = pd.Series(g).map(old).to_numpy(float)
    def cnt(e, m): return dict(n_pos=int((m & y).sum()), KOup=int((m & y & np.isfinite(e) & (e > 0)).sum()), KOdown=int((m & y & np.isfinite(e) & (e < 0)).sum()), unsigned=int((m & y & ~np.isfinite(e)).sum()))
    flips = int((y & U & np.isfinite(e_old) & np.isfinite(e_new) & (np.sign(e_old) != np.sign(e_new))).sum())
    allg = np.ones(len(g), bool)
    rc = pd.DataFrame([dict(universe='U (SET9 universe)', version='before (human-symbol join)', **cnt(e_old, U)), dict(universe='U (SET9 universe)', version='after (1:1 HOM orthology join)', **cnt(e_new, U)),
                       dict(universe='all axis genes', version='before', **cnt(e_old, allg)), dict(universe='all axis genes', version='after', **cnt(e_new, allg))])
    rc['n_sign_flips_among_signed_both'] = flips; rc.to_csv(OUT + 'panc_fix_receipt.tsv', sep='\t', index=False); print(rc.to_string(index=False))
    lost = y & ~np.isfinite(e_old) & np.isfinite(e_new)
    pd.DataFrame(dict(gene_upper=g[lost], in_U=U[lost], log2FC_after=e_new[lost])).to_csv(OUT + 'panc_fix_recovered_genes.tsv', sep='\t', index=False)
    assert not (y & U & ~np.isfinite(e_new)).any(), 'responders without sign remain after fix'
    eps9 = [x[0] for x in ns['EP9']]
    REPL = ['PANC_RFX6', 'ERY_KLF1', 'ERY_KLF1_B', 'HEART_TBX5', 'HEART_NKX2-5']          # 3v3 bulk deg tier
    REPL2 = REPL + ['BM_GATA1_acute']                                                       # Include 3v3 bulk DE and 2v2 comparisons with uncertified donor pairing; exclude BM_cal and ERY_GATA1.
    parts = [sens_block(ns, 'primary_SET9_fixed', eps9, 5), sens_block(ns, 'a_replicated_3v3_bulkDEG', REPL, 5), sens_block(ns, 'a2_replicated_plus_2v2_acute', REPL2, 5),
             sens_block(ns, 'b_gate20', eps9, 20), sens_block(ns, 'e_common_KOdown_noGATA1acute', eps9, 5, dirs=('KOdown',), skip={('KOdown', 'BM_GATA1_acute')}),
             sens_block(ns, 'e_ref_KOdown_with_acute', eps9, 5, dirs=('KOdown',))]
    write_sens(OUT, 'sens_abe', parts)
    S_ = receipts(ns, OUT); print(S_.to_string(index=False))
    pcontrol(ns, OUT, eps9, 'SET9')
    nfe2_tal1(ns, OUT)
    print('FIX DONE')
elif MODE == 'fullcov':
    ns = run_base(OUT, fix=True, fullcov=True)
    receipts(ns, OUT); print('FULLCOV DONE')
