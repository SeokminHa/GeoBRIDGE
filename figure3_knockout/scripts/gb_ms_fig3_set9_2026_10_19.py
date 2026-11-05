"""Fig3 SET6-9method vs SET9 evaluation (EVALUATION ONLY: no rescoring, no label changes; only re-reads stored k=0 z columns, as gb_ms_fig3_support_matched_2026_10_19.py).
internally written, NOT independently reviewed.  README: OUT/README.md.
Methods: GB + |Pearson|,GENIE3,GRNBoost2,PIDC,DeepSEM,RegDiffusion,FastTENET,Velorama(rt0 = canonical 'Velorama' column of the vec parts).  RegVelo REMOVED from the finite mask.
Universe U = allfin(9 methods) & resp_ok (recomputed via PT.lattice_unit, as gb_fig3_kotested) & ~dup & ~self.  --mask10 = PARITY MODE: SET6 only, original 10-method mask, compared to the stored tables.
Draws: PT.lattice_draws(dsys,B,sizes=(50,)), C=drc.C[B:2B] (gb_fig3_kotested.py); asserted identical across draw-systems of one system (same block+axis => same seeds).
Aggregation: endpoint -> TF (mean over assays of the same TF) -> system -> equal-system mean over systems present.  p = PT.boot_p, Holm over 8 externals within set x metric x direction x scheme.
usage (server): script OUT [--B 1000] [--mask10]"""
import sys, os, json
import numpy as np, pandas as pd
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
sys.path[:0] = [R + 'scratch', R + 'pairspace']
a = sys.argv; OUT = a[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
B = int(a[a.index('--B') + 1]) if '--B' in a else 1000
M10 = '--mask10' in a
import gb_layer_lattice as L
from gb_fig3d_magnitude import up, METH, VEC, effect, BIOSYS
import gb_ptable as PT, gb_lattice_extra as LE, gb_canon_table as CT, gb_primary as GP
LE.register(); import gb_calbm_propagate as CP; CP.preflight()
FLIP = {'GATA1': -1.0}; GATE = 5; SYS4 = ['PANC', 'ERYTHROID', 'BM', 'HEART']
MS = list(METH) if M10 else [m for m in METH if m != 'RegVelo |J|']; EXT = MS[1:]; NEXT = len(EXT)
# (label, system, effect/draw-system, tf, draw-system, vec stem)
EP6 = [('PANC_RFX6', 'PANC', 'PANC', 'RFX6', 'PANC', 'PANC__RFX6__hist'), ('ERY_GATA1', 'ERYTHROID', 'GATA1', 'GATA1', 'GATA1', 'GATA1__GATA1__hist'),
       ('BM_GATA1_acute', 'BM', 'BM_GATA1_ACUTE', 'GATA1', 'BM_GATA1_ACUTE', 'BM_GATA1_ACUTE__GATA1__hist'), ('BM_RUNX1_cal', 'BM', 'BM_CHIPATLAS_CALTOT', 'RUNX1', 'BM_CHIPATLAS_CALTOT', 'BM_CHIPATLAS_CALTOT__RUNX1__cal'),
       ('HEART_TBX5', 'HEART', 'CARDIACV', 'TBX5', 'CARDIACV', 'CARDIACV__TBX5__hist'), ('HEART_NKX2-5', 'HEART', 'CARDIACV', 'NKX2-5', 'CARDIACV', 'CARDIACV__NKX2-5__hist')]
EP9 = EP6 + [('ERY_KLF1', 'ERYTHROID', 'KLF1', 'KLF1', 'KLF1', 'KLF1__KLF1__hist'), ('ERY_KLF1_B', 'ERYTHROID', 'KLF1_B', 'KLF1', 'KLF1_B', 'KLF1_B__KLF1__hist'),
             ('BM_GATA1_cal', 'BM', 'BM_CHIPATLAS_CALTOT', 'GATA1', 'BM_CHIPATLAS_CALTOT', 'BM_CHIPATLAS_CALTOT__GATA1__cal')]
EPV = [('BM_NFE2_cal', 'BM', 'BM_CHIPATLAS_CALTOT', 'NFE2', 'BM_CHIPATLAS_CALTOT', 'BM_CHIPATLAS_CALTOT__NFE2__cal'), ('BM_TAL1_cal', 'BM', 'BM_CHIPATLAS_CALTOT', 'TAL1', 'BM_CHIPATLAS_CALTOT', 'BM_CHIPATLAS_CALTOT__TAL1__cal')]   # VARIANT only: meet n_pos>=5 but fail the repo's >=20-responder gate
EPS = EP6 if M10 else EP9   # EPV (NFE2/TAL1 cal) NOT run: no stored signed-score cache source for them (no k0), and they fail the repo's >=20-responder gate
SETS = {'SET6': [x[0] for x in EP6]} if M10 else {'SET6': [x[0] for x in EP6], 'SET9': [x[0] for x in EP9]}
TFOF = {x[0]: x[3] for x in EPS}; SYSOF = {x[0]: x[1] for x in EPS}
FEAT = {'PANC': 'PANC', 'ERYTHROID': 'GATA1', 'BM': 'BONEMARROW', 'HEART': 'CARDIACV'}
MK = np.load(R + 'results/signed/figdec_2026_10_17/T1_a1_audit/p1c_masks.npz'); SG = R + 'results/signed/'
import anndata as ad
feat = {s: pd.read_csv(f'{R}results/real_uniform/catbank/{f}_features.tsv', sep='\t', usecols=['gene', 'raw_det', 'abundance', 'raw_det_u']).assign(G=lambda d: d.gene.str.upper()).drop_duplicates('G', keep=False).set_index('G') for s, f in FEAT.items()}
load = {}
for s, f in FEAT.items():
    A_ = ad.read_h5ad(f'{R}results/canonical_2026_09_18/{f}_canonical.h5ad', backed='r'); P = np.asarray(A_.varm['PCs'])[:, :8]
    nm = pd.Series(np.linalg.norm(P, axis=1), index=A_.var_names.astype(str).str.upper()); load[s] = nm[~nm.index.duplicated(keep=False)]
def qb(x, k): return pd.qcut(pd.Series(x).rank(method='first'), k, labels=False).to_numpy()
drc = {}; EPD = {}; COVER = []; MASKCH = []
for lab, sysn, esys, tf, dsys, stem in EPS:
    z = np.load(f'{VEC}ep_{stem}.vec.npz', allow_pickle=True)
    g = up(z['genes']); gi = z['gi']; y = z['y'].astype(bool)
    V = {m: z[f'cog::{m}'].astype(float) for m in MS}
    fin10 = np.all([np.isfinite(z[f'cog::{m}'].astype(float)) for m in METH], 0); allfin = np.all([np.isfinite(V[m]) for m in MS], 0)
    dup = pd.Series(g).duplicated(keep=False).to_numpy(); selfm = (g == tf.upper())
    cap = {}; orig = L.assemble
    def _a(s_, t_, g_, V_, truth_, *aa, **kk): cap['t'] = truth_; return orig(s_, t_, g_, V_, truth_, *aa, **kk)
    if dsys not in drc: drc[dsys] = PT.lattice_draws(dsys, B, sizes=(50,))[0]
    L.assemble = _a
    try: PT.lattice_unit(dsys, tf, drc[dsys].gene)
    finally: L.assemble = orig
    rok = np.asarray(cap['t']['resp_ok'], bool); assert len(rok) == len(y) and y.sum() == y[rok].sum(), 'positive outside resp_ok ' + lab
    if lab in [x[0] for x in EP6]:
        assert (MK[lab + '_respok'] == rok).all(), 'T1 resp_ok parity ' + lab
        if M10: assert (MK[lab + '_allfin'] == allfin).all(), 'T1 allfin parity ' + lab
    U = allfin & rok & ~dup & ~selfm; U10 = fin10 & rok & ~dup & ~selfm
    assert not (U10 & ~U).any()
    MASKCH.append(dict(endpoint=lab, n_U_10method=int(U10.sum()), n_U_9method=int(U.sum()), n_genes_added=int((U & ~U10).sum()), n_pos_10=int((y & U10).sum()), n_pos_9=int((y & U).sum()), n_pos_added=int((y & U & ~U10).sum())))
    e = pd.Series(g).map(pd.Series(FLIP.get(esys, 1.0) * effect(esys, tf))).to_numpy(float)
    C = drc[dsys].C[B:2 * B]; assert C.shape[0] == B
    sc = np.load(f'{SG}{BIOSYS[esys]}.npz', allow_pickle=True); axis = np.char.upper(sc['genes'].astype(str)); ai = {x: i for i, x in enumerate(axis)}
    j = [s_.upper() for s_ in sc['sources'].astype(str)].index(tf.upper()); uni = np.asarray(sc['uni'][j])
    uni = np.flatnonzero(uni.astype(bool)) if (uni.dtype == bool or (uni.dtype == object and isinstance(uni.flat[0], (bool, np.bool_)))) else uni.astype(int)
    idx = np.array([ai.get(x, -1) for x in g]); has = idx >= 0; Zk = {}
    for h in GP.ROOTS:
        Ah = sc[f'{h}|U'][j]; zz = np.full((9, len(axis)), np.nan)
        for k in range(9): zz[k, uni] = GP.zcol(np.abs(Ah[uni, k].astype(float)))
        Zk[h] = np.stack([np.where(has, zz[k][np.where(has, idx, 0)], np.nan) for k in range(9)])
    k0 = np.minimum(*[Zk[h][0] for h in GP.ROOTS])
    with np.errstate(all='ignore'):
        def mean08(Zx): m = np.nanmean(Zx, 0); m[np.isfinite(Zx).sum(0) < 6] = np.nan; return m
        p08 = np.minimum(*[mean08(Zk[h]) for h in GP.ROOTS])
    pd_ = np.nanmax(np.abs(p08[U] - V['GeoBRIDGE'][U])) if np.isfinite(p08[U]).all() else np.inf
    f = feat[sysn]; ld = load[sysn]
    d = pd.Series(g).map(f.raw_det_u).to_numpy(float); ds = pd.Series(g).map(f.raw_det).to_numpy(float); ab = pd.Series(g).map(f.abundance).to_numpy(float); w8 = pd.Series(g).map(ld).to_numpy(float)
    cov = dict(endpoint=lab, nU=int(U.sum()), cov_raw_det_u=np.isfinite(d[U]).mean(), cov_raw_det=np.isfinite(ds[U]).mean(), cov_abundance=np.isfinite(ab[U]).mean(), cov_loading8=np.isfinite(w8[U]).mean(),
               cov_k0_finite=np.isfinite(k0)[U].mean(), prod08_recompute_maxabsdiff=float(pd_))
    COVER.append(cov); print(cov, MASKCH[-1], flush=True)
    assert cov['cov_raw_det_u'] == 1.0, 'STOP: primary covariate coverage < 100%'
    assert cov['prod08_recompute_maxabsdiff'] < 1e-6, 'STOP: stored k-depth reconstruction does not reproduce production GB ' + lab
    V['GB_k0'] = k0
    def lab_(fn, m):
        o = np.full(len(g), -1); o[m] = fn(m); return o
    S = {'raw': None, 'primary': lab_(lambda m: qb(d[m], 5), U)}
    mu_ = U & np.isfinite(ds) & np.isfinite(ab); S['S1'] = lab_(lambda m: qb(ds[m], 5) * 3 + qb(ab[m], 3), mu_)
    if cov['cov_loading8'] > 0: mm = U & np.isfinite(w8); S['S2'] = lab_(lambda m: qb(w8[m], 5), mm)
    EPD[lab] = dict(sysn=sysn, g=g, gi=gi, y=y, V=V, U=U, e=e, C=C, S=S, d=d, k0f=np.isfinite(k0), dsys=dsys)
COVER = pd.DataFrame(COVER); MASKCH = pd.DataFrame(MASKCH)
# draws synchronised within a system (same block+axis => same seeds): assert
for s in SYS4:
    cs = [(l, D['C']) for l, D in EPD.items() if D['sysn'] == s]
    for l, c in cs[1:]: assert np.array_equal(cs[0][1], c), f'draws differ within system {s}: {cs[0][0]} vs {l}'
DRAWSYNC = {s: sorted({D['dsys'] for D in EPD.values() if D['sysn'] == s}) for s in SYS4}

def strat(sc, yl, rr, bins, gi, C, Cg, log):
    if bins is None: pt, bs = CT.eval_rows(sc, yl, rr, gi, C); return np.r_[pt['AUROC'], np.asarray(bs['AUROC'], float)]
    num = np.zeros(1 + B); den = np.zeros(1 + B)
    for b in np.unique(bins[rr & (bins >= 0)]):
        rb = rr & (bins == b); P_, N_ = rb & yl, rb & ~yl
        if P_.sum() < 3 or N_.sum() < 3: log.append((b, int(P_.sum()), int(N_.sum()))); continue
        pt, bs = CT.eval_rows(sc, yl, rb, gi, C); aa = np.r_[pt['AUROC'], np.asarray(bs['AUROC'], float)]
        w = np.r_[P_.sum() * N_.sum(), Cg[:, P_].sum(1) * Cg[:, N_].sum(1)]; ok = np.isfinite(aa) & (w > 0); num[ok] += w[ok] * aa[ok]; den[ok] += w[ok]
    with np.errstate(all='ignore'): return np.where(den > 0, num / den, np.nan)

RES = {}; RAP = {}; META = []; DROP = []; DESC = []   # RES[(scheme, dir, ep, method)] AUROC ; RAP[(ep, method)] = {'AUROC','APLIFT'} all-responder raw
for lab, D_ in EPD.items():
    gi, y, U, e, C = D_['gi'], D_['y'], D_['U'], D_['e'], D_['C']; Cg = C[:, gi].astype(float); d = D_['d']
    # all-responder AUROC + AP-lift (raw), incl. GB_k0 and production on the k0-finite universe
    if int((y & U).sum()) >= GATE:
        for m, sc in list(D_['V'].items()) + [('GB_prod@k0univ', D_['V']['GeoBRIDGE'])]:
            rm = U & D_['k0f'] if m in ('GB_k0', 'GB_prod@k0univ') else U
            pt, bs = CT.eval_rows(sc, y, rm, gi, C); RAP[(lab, m)] = {mt: np.r_[pt[mt], np.asarray(bs[mt], float)] for mt in ('AUROC', 'APLIFT')}
    for nm, yl in (('ALL', y.copy()), ('KOup', y & np.isfinite(e) & (e > 0)), ('KOdown', y & np.isfinite(e) & (e < 0))):
        npos = int((yl & U).sum()); META.append(dict(direction=nm, endpoint=lab, system=D_['sysn'], n_pos=npos, included=npos >= GATE))
        rr = U & (yl | ~y); q = D_['S']['primary']
        if nm != 'ALL':
            for b in range(5):
                m = U & (q == b); DESC.append(dict(direction=nm, endpoint=lab, quintile=b, n_pos=int((m & yl).sum()), n_neg=int((m & ~y).sum())))
        if npos < GATE: continue
        for sch, bins in D_['S'].items():
            for m, sc in D_['V'].items():
                rm = rr & D_['k0f'] if m == 'GB_k0' else rr
                lg = []; RES[(sch, nm, lab, m)] = strat(sc, yl, rm, bins, gi, C, Cg, lg)
                if m == 'GeoBRIDGE': DROP += [dict(scheme=sch, direction=nm, endpoint=lab, bin=int(b), n_pos=p_, n_neg=n_) for b, p_, n_ in lg]
            if sch != 'S2': RES[(sch, nm, lab, 'GB_prod@k0univ')] = strat(D_['V']['GeoBRIDGE'], yl, rr & D_['k0f'], bins, gi, C, Cg, [])
UNDEF = [k for k in RES if k[3] == 'GeoBRIDGE' and not np.isfinite(RES[k][0])]
for k in list(RES):
    if (k[0], k[1], k[2], 'GeoBRIDGE') in UNDEF: del RES[k]
META = pd.DataFrame(META); META['strat_undefined'] = [';'.join(k[0] for k in UNDEF if k[1] == r.direction and k[2] == r.endpoint) for r in META.itertuples()]
DESC = pd.DataFrame(DESC)

def ci(x): return PT.pct_ci(np.asarray(x)[1:])
def holm(p):
    p = np.asarray(p); o = np.argsort(p); adj = np.empty(len(p)); run = 0
    for i, k in enumerate(o): run = max(run, (len(p) - i) * p[k]); adj[k] = min(1, run)
    return adj
def agg(getter, eps):
    """endpoint -> TF (mean over assays of same TF in system) -> system -> equal-system mean over systems present. getter(ep)->(1+B,) or None"""
    ps = {}
    for s in SYS4:
        by = {}
        for lab in eps:
            if SYSOF[lab] != s: continue
            v = getter(lab)
            if v is not None: by.setdefault(TFOF[lab], []).append(v)
        if by: ps[s] = np.mean([np.mean(v, 0) for v in by.values()], 0)
    return (np.mean(list(ps.values()), 0) if ps else None), ps
G_ = lambda sch, d, m: (lambda lab: RES.get((sch, d, lab, m)))
G_ap = lambda mt, m: (lambda lab: RAP[(lab, m)][mt] if (lab, m) in RAP else None)
def row(r, **kw): lo, hi = ci(r); return dict(**kw, point=r[0], lo=lo, hi=hi, p=PT.boot_p(r[1:]))
SCHEMES = ['raw', 'primary', 'S1', 'S2']; DIRS = ['ALL', 'KOup', 'KOdown']
EQ, CON, PER, DG, LOSO, PSYS = [], [], [], [], [], []
for sn, eps in SETS.items():
    # ---- item 1 (all-responder, raw): AUROC + AP-lift, equal-system, GB - each external
    for mt in ('AUROC', 'APLIFT'):
        E = {m: agg(G_ap(mt, m), eps) for m in MS}
        for m in MS + ['GB_k0', 'GB_prod@k0univ']:
            r, ps = agg(G_ap(mt, m), eps)
            if r is None: continue
            lo, hi = ci(r); EQ.append(dict(set=sn, scope='all-responder', scheme='raw', direction='ALL', metric=mt, method=m, value=r[0], lo=lo, hi=hi, n_systems=len(ps)))
            for s_, v in ps.items(): lo, hi = ci(v); PSYS.append(dict(set=sn, direction='ALL', scheme='raw', metric=mt, system=s_, method=m, value=v[0], lo=lo, hi=hi))
        gb = E['GeoBRIDGE'][0]; res = []
        for m in EXT:
            df = gb - E[m][0]; res.append(row(df, set=sn, scope='all-responder', scheme='raw', direction='ALL', metric=mt, external=m, GB=gb[0], ext=E[m][0][0]))
        for r_, h in zip(res, holm([r['p'] for r in res])): r_['p_holm8'] = h
        CON += res
        for dn, a_, b_ in (('DEPTHGAIN(prod-k0)', 'GB_prod@k0univ', 'GB_k0'),):
            df = agg(G_ap(mt, a_), eps)[0] - agg(G_ap(mt, b_), eps)[0]; DG.append(row(df, set=sn, scope='all-responder', scheme='raw', direction='ALL', metric=mt, endpoint='EQUAL_SYSTEM', contrast=dn))
        # LOSO GB - Velorama, GB - RegDiffusion
        for s_ in SYS4:
            ee = [l for l in eps if SYSOF[l] != s_]
            for ex in ('Velorama', 'RegDiffusion'):
                df = agg(G_ap(mt, 'GeoBRIDGE'), ee)[0] - agg(G_ap(mt, ex), ee)[0]
                LOSO.append(row(df, set=sn, direction='ALL', scheme='raw', metric=mt, dropped_system=s_, external=ex, n_systems=len(set(SYSOF[l] for l in ee))))
    # ---- item 2 (directional), all schemes
    for sch in SCHEMES:
        for d in DIRS:
            if sch == 'S2' and d == 'ALL': continue
            ms = [m for m in MS + ['GB_k0'] + (['GB_prod@k0univ'] if sch != 'S2' else []) if agg(G_(sch, d, m), eps)[0] is not None]
            if 'GeoBRIDGE' not in ms: continue
            for m in ms:
                r, ps = agg(G_(sch, d, m), eps); lo, hi = ci(r)
                if d != 'ALL' or sch != 'raw': EQ.append(dict(set=sn, scope='directional' if d != 'ALL' else 'all-responder', scheme=sch, direction=d, metric='AUROC', method=m, value=r[0], lo=lo, hi=hi, n_systems=len(ps)))
                if d != 'ALL' or sch != 'raw':
                    for s_, v in ps.items(): lo, hi = ci(v); PSYS.append(dict(set=sn, direction=d, scheme=sch, metric='AUROC', system=s_, method=m, value=v[0], lo=lo, hi=hi))
            if d == 'ALL' and sch == 'raw': continue   # already in item 1 (AP-lift loop); AUROC identical
            gb = agg(G_(sch, d, 'GeoBRIDGE'), eps)[0]; res = []
            for m in EXT:
                df = gb - agg(G_(sch, d, m), eps)[0]; res.append(row(df, set=sn, scope='directional' if d != 'ALL' else 'all-responder', scheme=sch, direction=d, metric='AUROC', external=m, GB=gb[0], ext=agg(G_(sch, d, m), eps)[0][0]))
            for r_, h in zip(res, holm([r['p'] for r in res])): r_['p_holm8'] = h
            CON += res
            if sch != 'S2':
                df = agg(G_(sch, d, 'GB_prod@k0univ'), eps)[0] - agg(G_(sch, d, 'GB_k0'), eps)[0]; DG.append(row(df, set=sn, scope='directional' if d != 'ALL' else 'all-responder', scheme=sch, direction=d, metric='AUROC', endpoint='EQUAL_SYSTEM', contrast='DEPTHGAIN(prod-k0)'))
                for s_ in SYS4:
                    p_, k_ = agg(G_(sch, d, 'GB_prod@k0univ'), [l for l in eps if SYSOF[l] == s_])[0], agg(G_(sch, d, 'GB_k0'), [l for l in eps if SYSOF[l] == s_])[0]
                    if p_ is not None: DG.append(row(p_ - k_, set=sn, scope='per-system', scheme=sch, direction=d, metric='AUROC', endpoint=s_, contrast='DEPTHGAIN(prod-k0)'))
            if sch in ('raw', 'primary'):
                for s_ in SYS4:
                    ee = [l for l in eps if SYSOF[l] != s_]
                    for ex in ('Velorama', 'RegDiffusion'):
                        a1, b1 = agg(G_(sch, d, 'GeoBRIDGE'), ee)[0], agg(G_(sch, d, ex), ee)[0]
                        if a1 is not None and b1 is not None: LOSO.append(row(a1 - b1, set=sn, direction=d, scheme=sch, metric='AUROC', dropped_system=s_, external=ex, n_systems=len(set(SYSOF[l] for l in ee if (sch, d, l, 'GeoBRIDGE') in RES))))
# per-endpoint (point + CI), all-responder AUROC/APLIFT and directional; per-endpoint depth gain
for (lab, m), dd in RAP.items():
    for mt, r in dd.items(): lo, hi = ci(r); PER.append(dict(scheme='raw', direction='ALL', metric=mt, endpoint=lab, system=SYSOF[lab], tf=TFOF[lab], method=m, value=r[0], lo=lo, hi=hi, n_pos=int(META[(META.direction == 'ALL') & (META.endpoint == lab)].n_pos.iloc[0])))
for (sch, d, lab, m), r in RES.items():
    if d == 'ALL' and sch == 'raw': continue
    lo, hi = ci(r); PER.append(dict(scheme=sch, direction=d, metric='AUROC', endpoint=lab, system=SYSOF[lab], tf=TFOF[lab], method=m, value=r[0], lo=lo, hi=hi, n_pos=int(META[(META.direction == d) & (META.endpoint == lab)].n_pos.iloc[0]), n_nonfinite_draws=int((~np.isfinite(r[1:])).sum())))
for (sch, d, lab, m), r in list(RES.items()):
    if m == 'GB_k0' and (sch, d, lab, 'GB_prod@k0univ') in RES: DG.append(row(RES[(sch, d, lab, 'GB_prod@k0univ')] - r, set='per-endpoint', scope='per-endpoint', scheme=sch, direction=d, metric='AUROC', endpoint=lab, contrast='DEPTHGAIN(prod-k0)'))
for lab in EPD:
    for mt in ('AUROC', 'APLIFT'):
        if (lab, 'GB_k0') in RAP: DG.append(row(RAP[(lab, 'GB_prod@k0univ')][mt] - RAP[(lab, 'GB_k0')][mt], set='per-endpoint', scope='per-endpoint', scheme='raw', direction='ALL', metric=mt, endpoint=lab, contrast='DEPTHGAIN(prod-k0)'))
tabs = dict(equal_system=pd.DataFrame(EQ), paired_diffs=pd.DataFrame(CON), per_endpoint=pd.DataFrame(PER), depth_gain=pd.DataFrame(DG), loso=pd.DataFrame(LOSO), per_system=pd.DataFrame(PSYS),
            classes=META, mask_change=MASKCH, coverage=COVER, descriptive=DESC, dropped_bins=pd.DataFrame(DROP, columns=['scheme', 'direction', 'endpoint', 'bin', 'n_pos', 'n_neg']))
for k, v in tabs.items(): v.to_csv(OUT + k + '.tsv', sep='\t', index=False)
CN = tabs['paired_diffs']; CN[CN.external.isin(['Velorama', 'RegDiffusion'])].to_csv(OUT + 'named_contrasts_Velorama_RegDiffusion.tsv', sep='\t', index=False)
json.dump(dict(B=B, mask10_parity_mode=M10, methods=MS, sets=SETS, draw_systems_per_system=DRAWSYNC, nonfinite_draws=int(pd.DataFrame(PER).get('n_nonfinite_draws', pd.Series([0])).fillna(0).sum())), open(OUT + 'run.json', 'w'), indent=1)
# ---- parity vs stored tables (SET6 rows; only in --mask10 mode, i.e. where the mask is unchanged)
PAR = []
if M10:
    G0 = R + 'results/figures/manuscript_2026_10_19/'
    def vn(m): return 'Velorama' if str(m).startswith('Velorama') else m
    pe = pd.read_csv(G0 + 'fig3_rt0_compute/per_endpoint_rt0.tsv', sep='\t'); pe = pe[pe.method != 'Velorama'].assign(method=lambda x: x.method.map(vn))   # stored 'Velorama' = pre-swap rt1; 'Velorama_rt0' -> Velorama
    P_ = pd.DataFrame(PER); P_ = P_[(P_.direction == 'ALL') & (P_.scheme == 'raw')]
    for _, r in pe.iterrows():
        for mt in ('AUROC', 'APLIFT'):
            k = P_[(P_.endpoint == r.endpoint) & (P_.method == r.method) & (P_.metric == mt)]
            PAR.append(dict(check=f'all_resp_point_{mt}', key=f'{r.endpoint}|{r.method}', absdiff=abs(k.value.iloc[0] - r[mt]) if len(k) else np.inf))
    sp = pd.read_csv(G0 + 'fig3_support_matched_rt0/per_endpoint.tsv', sep='\t'); sp['method'] = sp.method.map(vn)
    sp = sp[sp.method != 'REF_negdet']
    for _, r in sp.iterrows():
        k = P_x = pd.DataFrame(PER); k = k[(k.scheme == r.scheme) & (k.direction == r.direction) & (k.endpoint == r.endpoint) & (k.method == r.method)]
        PAR.append(dict(check='directional_point', key=f'{r.scheme}|{r.direction}|{r.endpoint}|{r.method}', absdiff=abs(k.value.iloc[0] - r.AUROC) if len(k) else np.inf))
    if B == 1000:
        es = pd.read_csv(G0 + 'fig3_rt0_compute/equal_system_rt0.tsv', sep='\t'); print('stored equal_system_rt0 methods:', es.method.unique())
        Cn = tabs['paired_diffs']
        for _, r in es.iterrows():
            k = Cn[(Cn.scope == 'all-responder') & (Cn.metric == r.metric) & (Cn.external == vn(r.method))]
            if len(k): k = k.iloc[0]; PAR.append(dict(check='all_resp_diff_B1000', key=f'{r.metric}|{r.method}', absdiff=max(abs(k['point'] - r['diff']), abs(k.lo - r.lo), abs(k.hi - r.hi), abs(k.p - r.p))))
        pdd = pd.read_csv(G0 + 'fig3_support_matched_rt0/paired_diffs.tsv', sep='\t')
        for _, r in pdd.iterrows():
            k = Cn[(Cn.scope == 'directional') & (Cn.scheme == r.scheme) & (Cn.direction == r.direction) & (Cn.external == vn(r.external))]
            if len(k): k = k.iloc[0]; PAR.append(dict(check='directional_diff_B1000', key=f'{r.scheme}|{r.direction}|{r.external}', absdiff=max(abs(k['point'] - r['diff']), abs(k.lo - r.lo), abs(k.hi - r.hi), abs(k.p - r.p))))
P = pd.DataFrame(PAR); P.to_csv(OUT + 'parity.tsv', sep='\t', index=False)
if len(P): print('PARITY', P.groupby('check').absdiff.agg(['max', 'count']).to_dict('index')); assert P.absdiff.max() < 1e-9, 'PARITY FAIL'
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 600)
print(MASKCH.to_string(index=False)); print(META.to_string(index=False)); print(CN[(CN.scheme == 'raw')].round(4).to_string(index=False))
print('DONE')
