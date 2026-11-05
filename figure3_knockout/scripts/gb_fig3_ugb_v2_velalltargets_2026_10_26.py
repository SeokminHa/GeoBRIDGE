"""EXPLORATORY v2. internally written; v1 internal review (CODE FIX). Implements SPEC.md (sha8 8bbe6a88ecf080ac) + SPEC_ADDENDUM.md (module resampling G/50, Holm, validity rule, quality subset, BM-acute design sensitivity).
EVALUATION ONLY: no score recomputed. Mirrors scratch/gb_ms_fig3_set9_2026_10_19.py (:41-61, :96-116, :138) with PANC effect_fixed.
usage (server): script OUT [--B 1000] [--endpoints A,B]"""
import sys, os, ast, json, hashlib
import numpy as np, pandas as pd
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
sys.path[:0] = [R + 'scratch', R + 'pairspace']
a = sys.argv; OUT = a[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
B = int(a[a.index('--B') + 1]) if '--B' in a else 1000
ONLY = a[a.index('--endpoints') + 1].split(',') if '--endpoints' in a else None
import gb_layer_lattice as L
from gb_fig3d_magnitude import up, METH, VEC, effect as EFF0
import gb_ptable as PT, gb_lattice_extra as LE, gb_canon_table as CT
LE.register(); import gb_calbm_propagate as CP; CP.preflight()
ns = {}
for node in ast.parse(open(R + 'scratch/gb_ms_fig3_set9_2026_10_19.py').read()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('EP6', 'EP9') for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<roster>', 'exec'), ns)
EP9 = ns['EP9']
en = dict(R=R, pd=pd, EFF0=EFF0)
fn = next(n for n in ast.parse(open(R + 'scratch/gb_ms_fig3_set9_sens_2026_10_20.py').read()).body if isinstance(n, ast.FunctionDef) and n.name == 'effect_fixed')
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<effect>', 'exec'), en); effect = en['effect_fixed']
FLIP = {'GATA1': -1.0}; GATE = 5; DESC_N = 20; SYS4 = ['PANC', 'ERYTHROID', 'BM', 'HEART']; VALID_FRAC = 0.95
MS = [m for m in METH if m != 'RegVelo |J|']; EXT = MS[1:]; assert MS[0] == 'GeoBRIDGE' and len(EXT) == 8, MS
MS2 = [m for m in MS if m != 'Velorama']
EPS = [x for x in EP9 if ONLY is None or x[0] in ONLY]; assert EPS
TFOF = {x[0]: x[3] for x in EP9}; SYSOF = {x[0]: x[1] for x in EP9}
QUALITY = ['PANC_RFX6', 'ERY_KLF1_B', 'HEART_TBX5', 'HEART_NKX2-5']
DIRS = ['ALL', 'KOup', 'KOdown']; MT = ('AUROC', 'APLIFT')
CAN = R + 'results/figures/manuscript_2026_10_19/fig3_set9_sens/'
TTF = R + 'results/signed/bm_acute_design_audit_2026_10_26/responder_truth_table_by_design.tsv'
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()[:16]


def ci(r):
    return PT.pct_ci(np.asarray(r)[1:])


def nvalid(r):
    return int(np.isfinite(np.asarray(r)[1:]).sum())


def holm(p):
    p = np.asarray(p, float); out = np.full(len(p), np.nan); ok = np.flatnonzero(np.isfinite(p)); o = ok[np.argsort(p[ok])]; run = 0.0
    for i, k in enumerate(o): run = max(run, (len(o) - i) * p[k]); out[k] = min(1.0, run)
    return out


def classify(r):
    """returns sig (+,-,0,U), lo, hi, nvalid, p. U = <95% valid draws (undefined interval; never counted as includes-0)"""
    nv = nvalid(r); lo, hi = ci(r)
    if nv < VALID_FRAC * (len(r) - 1) or not np.isfinite(lo): return 'U', lo, hi, nv, np.nan
    return ('+' if lo > 0 else '-' if hi < 0 else '0'), lo, hi, nv, PT.boot_p(np.asarray(r)[1:])


HASHES = {}; IDENT = []
drc = {}; D = {}; DRAWS = {}
for lab, sysn, esys, tf, dsys, stem in EPS:
    vp = f'{VEC}ep_{stem}.vec.npz'; z = np.load(vp, allow_pickle=True); HASHES['vec:' + lab] = sha(vp)
    g_raw = z['genes'].astype(str); g = up(z['genes']); gi = z['gi']; y = z['y'].astype(bool)
    V = {m: z[f'cog::{m}'].astype(float) for m in MS}
    assert len(g) == len(y) == len(gi) == len(V['GeoBRIDGE']) and all(len(v) == len(g) for v in V.values()), 'length ' + lab
    dup = pd.Series(g).duplicated(keep=False).to_numpy(); selfm = (g == tf.upper())
    if dsys not in drc: drc[dsys] = PT.lattice_draws(dsys, B, sizes=(50,))[0]
    cap = {}; orig = L.assemble
    def _a(s_, t_, g_, V_, truth_, *aa, **kk): cap['t'] = truth_; return orig(s_, t_, g_, V_, truth_, *aa, **kk)
    L.assemble = _a
    try: ru = PT.lattice_unit(dsys, tf, drc[dsys].gene)       # rebuilt endpoint (return value KEPT for identity checks)
    finally: L.assemble = orig
    # ---- identity asserts (internal review #1)
    rg = np.asarray(ru['genes']).astype(str)
    assert np.array_equal(rg, g_raw), 'rebuilt genes != cached genes (order/content) ' + lab
    assert np.array_equal(np.asarray(ru['gi']), np.asarray(gi)), 'rebuilt gi != cached gi ' + lab
    axis = np.asarray(L.axis_of(L.SYSTEMS[dsys]['axis'])).astype(str)
    assert np.array_equal(axis[np.asarray(gi)], g_raw), 'draw axis[gi] != genes ' + lab
    for m in MS + ['RegVelo |J|']:
        assert m in ru['V'], 'rebuilt V missing ' + m
        rv = np.asarray(ru['V'][m], float); cv = z['cog::' + m].astype(float)
        assert np.array_equal(np.isfinite(rv), np.isfinite(cv)), f'finite pattern rebuilt != cached {m} {lab}'
        both = np.isfinite(rv) & np.isfinite(cv); mxd = float(np.abs(rv[both] - cv[both]).max()) if both.any() else 0.0
        IDENT.append(dict(endpoint=lab, method=m, n_finite=int(both.sum()), max_abs_diff=mxd, n_unequal=int((rv[both] != cv[both]).sum())))
        if m != 'Velorama': assert mxd == 0.0, f'rebuilt score != cached {m} {lab}'
        elif mxd != 0.0: print('IDENTITY WARNING: cached Velorama column != rebuilt lattice Velorama', lab, mxd, flush=True)   # cached column is what the canonical evaluator uses
    rok = np.asarray(cap['t']['resp_ok'], bool); assert len(rok) == len(y) and y.sum() == y[rok].sum(), 'pos outside resp_ok ' + lab
    if 'resp' in cap['t']: assert np.array_equal(np.asarray(cap['t']['resp'], bool), y), 'label mismatch vs truth fn ' + lab
    for k_, v_ in (ru.get('files') or {}).items():
        try:
            if isinstance(v_, str) and os.path.exists(v_): HASHES[f'truthfile:{lab}:{k_}'] = sha(v_)
            else: HASHES[f'truthfile:{lab}:{k_}'] = repr(v_)[:200]
        except Exception as ex_: HASHES[f'truthfile:{lab}:{k_}'] = 'ERR ' + str(ex_)[:80]
    e = pd.Series(g).map(pd.Series(FLIP.get(esys, 1.0) * effect(esys, tf))).to_numpy(float)
    C = drc[dsys].C[B:2 * B]; assert C.shape[0] == B
    DRAWS[dsys] = C
    # ALLTARGETS VARIANT (2026-10-26): Velorama column replaced AFTER the identity checks by the alltargets_rt0 arm; all else identical to fig3_ugb_v2.py
    _SYSD = {'PANC': 'PANC', 'ERYTHROID': 'GATA1', 'BM': 'BONEMARROW', 'HEART': 'CARDIACV'}
    _zz = np.load(R + f'results/real_uniform/{_SYSD[sysn]}/velorama_v2_alltargets_rt0.npz', allow_pickle=True); _rg = [x.upper() for x in _zz['regulators'].astype(str)]
    _gg = {q: i for i, q in enumerate(_zz['genes'].astype(str))}; _row = np.asarray(_zz['score_all'], float)[_rg.index(tf.upper())]
    V['Velorama'] = np.array([_row[_gg[q]] if q in _gg else np.nan for q in g_raw]); print(lab, 'VELORAMA := alltargets_rt0, finite', int(np.isfinite(V['Velorama']).sum()), flush=True)
    U_GB = rok & ~dup & ~selfm & np.isfinite(V['GeoBRIDGE'])
    U_S1 = U_GB & np.all([np.isfinite(V[m]) for m in MS], 0)
    U_S2 = U_GB & np.all([np.isfinite(V[m]) for m in MS2], 0)
    assert not (U_S1 & ~U_S2).any()
    D[lab] = dict(g=g, gi=gi, y=y, V=V, e=e, C=C, U=U_GB, U1=U_S1, U2=U_S2, dsys=dsys, sysn=sysn)
    print(lab, 'IDENTITY OK; nU_GB', int(U_GB.sum()), 'nU_S1', int(U_S1.sum()), 'nU_S2', int(U_S2.sum()), flush=True)
for s in SYS4:
    cs = [D[l]['C'] for l in D if D[l]['sysn'] == s]
    for c in cs[1:]: assert np.array_equal(cs[0], c), 'draws differ within system ' + s
pd.DataFrame(IDENT).to_csv(OUT + 'identity_check.tsv', sep='\t', index=False)
np.savez_compressed(OUT + 'draws.npz', **{k: v.astype(np.int16) for k, v in DRAWS.items()})
for f_ in ['scratch/gb_ptable.py', 'scratch/gb_canon_table.py', 'scratch/gb_layer_lattice.py', 'scratch/gb_ms_fig3_set9_2026_10_19.py', 'scratch/gb_ms_fig3_set9_sens_2026_10_20.py', 'results/signed/fig3_ugb_2026_10_26/SPEC.md', 'results/signed/fig3_ugb_2026_10_26/fig3_ugb_v2.py', 'results/signed/bm_acute_design_audit_2026_10_26/responder_truth_table_by_design.tsv']:
    HASHES['code:' + f_] = sha(R + f_)
HASHES['draws.npz'] = sha(OUT + 'draws.npz')


def dir_labels(d, y, e):
    return y.copy() if d == 'ALL' else (y & np.isfinite(e) & ((e > 0) if d == 'KOup' else (e < 0)))


def ev(lab, m, d, mask, y=None, e=None):
    X = D[lab]; y = X['y'] if y is None else y; e = X['e'] if e is None else e
    yl = dir_labels(d, y, e); rr = mask if d == 'ALL' else mask & (yl | ~y)
    pt, bs = CT.eval_rows(X['V'][m], yl, rr, X['gi'], X['C'])
    return {mt: np.r_[pt[mt], np.asarray(bs[mt], float)] for mt in MT}, int((yl & rr).sum()), int(rr.sum())


def npos_of(lab, d, mask, y=None, e=None):
    X = D[lab]; y = X['y'] if y is None else y; e = X['e'] if e is None else e
    yl = dir_labels(d, y, e); rr = mask if d == 'ALL' else mask & (yl | ~y); return int((yl & rr).sum())


# ---------------- STRICT parity (internal review #3): complete key set, n_pos, point AND both CI endpoints
def parity(tag, mask_key, ms, ref):
    can = pd.read_csv(CAN + ref + '/per_endpoint.tsv', sep='\t'); can = can[(can.scheme == 'raw') & can.endpoint.isin(list(D)) & can.method.isin(ms)]
    exp_n = len(D) * 0
    rows = []; cache = {}; strict_ci = (B == 1000)
    for r in can.itertuples():
        mask = D[r.endpoint][mask_key]; k = (r.endpoint, r.method, r.direction)
        if k not in cache: cache[k] = None if npos_of(r.endpoint, r.direction, mask) < GATE else ev(r.endpoint, r.method, r.direction, mask)
        if cache[k] is None:
            rows.append(dict(endpoint=r.endpoint, method=r.method, direction=r.direction, metric=r.metric, canon=r.value, here=np.nan, absdiff=np.inf, lo_diff=np.inf, hi_diff=np.inf, n_pos_canon=r.n_pos, n_pos_here=-1)); continue
        v, npos, _ = cache[k]; lo, hi = ci(v[r.metric])
        rows.append(dict(endpoint=r.endpoint, method=r.method, direction=r.direction, metric=r.metric, canon=r.value, here=v[r.metric][0], absdiff=abs(v[r.metric][0] - r.value), lo_diff=abs(lo - r.lo), hi_diff=abs(hi - r.hi), n_pos_canon=r.n_pos, n_pos_here=npos))
    P = pd.DataFrame(rows); P.to_csv(OUT + f'parity_{tag}.tsv', sep='\t', index=False)
    # completeness: every (endpoint, method, direction, metric) expected from canonical must be present; canonical must cover every endpoint x method for ALL
    exp_keys = {(e_, m_, 'ALL', mt) for e_ in D for m_ in ms for mt in MT}
    got = {(r.endpoint, r.method, r.direction, r.metric) for r in can.itertuples()}
    miss = exp_keys - got
    npm = int((P.n_pos_canon != P.n_pos_here).sum()); mx = P.absdiff.max()
    ok = (not miss) and npm == 0 and mx < 1e-12 and (P.lo_diff.max() < 1e-12 and P.hi_diff.max() < 1e-12 if strict_ci else True) and len(P) > 0
    txt = f'{tag}: ref={ref} rows={len(P)} missing_expected_keys={len(miss)} n_pos_mismatch={npm} max|point|={mx!r} max|lo|={P.lo_diff.max()!r} max|hi|={P.hi_diff.max()!r} strict_CI_checked={strict_ci} PASS={ok}\n'
    open(OUT + f'parity_{tag}_receipt.txt', 'w').write(txt); print(txt, flush=True)
    return ok


ok1 = parity('S1', 'U1', MS, 'B1000_fix'); ok2 = parity('S2', 'U2', MS2, 'B1000_fullcov')
if not ok2: print('PARITY FAIL (blocking)'); sys.exit(2)   # ALLTARGETS VARIANT: S1 ref (B1000_fix) is the native-Velorama intersection, so S1 cannot match by construction; S2 (no Velorama) stays blocking
if not ok1: print('S1 parity not applicable (Velorama arm replaced); recorded in run.json', flush=True)

# ---------------- primary U_GB pairwise
PP = []; DIFF = {}; GBA = []; SUBGATE = []
for lab, X in D.items():
    U = X['U']
    for d in DIRS:
        npu = npos_of(lab, d, U)
        if npu < GATE: SUBGATE.append(dict(endpoint=lab, direction=d, n_pos=npu, nU_GB=int(U.sum()), note='GB-only sub-gate (n+<5): not evaluated'))
        else:
            v, npos, nrow = ev(lab, 'GeoBRIDGE', d, U)
            for mt in MT:
                lo, hi = ci(v[mt]); GBA.append(dict(endpoint=lab, direction=d, metric=mt, nU_GB=int(U.sum()), n_rows=nrow, n_pos=npos, prevalence=npos / nrow, value=v[mt][0], lo=lo, hi=hi, n_valid_draws=nvalid(v[mt]), flag_descriptive=npos < DESC_N))
        for m in EXT:
            P = U & np.isfinite(X['V'][m]); npos = npos_of(lab, d, P)
            if npos < GATE:
                PP.append(dict(endpoint=lab, comparator=m, direction=d, metric='SUBGATE', n_P=int(P.sum()), n_pos=npos, coverage_m_on_UGB=P.sum() / U.sum(), note=f'n_pos<{GATE}: not evaluated')); continue
            vg, npg, nrow = ev(lab, 'GeoBRIDGE', d, P); vm, npm_, _ = ev(lab, m, d, P); assert npg == npm_ == npos
            for mt in MT:
                df = vg[mt] - vm[mt]; DIFF[(lab, m, d, mt)] = df; sg, lo, hi, nv, p = classify(df)
                PP.append(dict(endpoint=lab, comparator=m, direction=d, metric=mt, n_P=int(P.sum()), n_rows=nrow, n_pos=npos, coverage_m_on_UGB=P.sum() / U.sum(), GB=vg[mt][0], m=vm[mt][0], diff=df[0], lo=lo, hi=hi,
                               p_boot=p, n_valid_draws=nv, sig_nominal=sg, flag_descriptive=npos < DESC_N, note=''))
    print('done', lab, flush=True)
PP = pd.DataFrame(PP); GBA = pd.DataFrame(GBA)
ev_ = PP[PP.metric != 'SUBGATE'].copy()
ev_['p_holm8'] = np.nan
for _, idx in ev_.groupby(['endpoint', 'direction', 'metric']).groups.items(): ev_.loc[idx, 'p_holm8'] = holm(ev_.loc[idx, 'p_boot'].values)
ev_['sig_holm'] = np.where(ev_.sig_nominal == 'U', 'U', np.where((ev_.p_holm8 < 0.05) & (ev_['diff'] > 0), '+', np.where((ev_.p_holm8 < 0.05) & (ev_['diff'] < 0), '-', '0')))
PP = pd.concat([ev_, PP[PP.metric == 'SUBGATE']], ignore_index=True)
PP.to_csv(OUT + 'per_pair.tsv', sep='\t', index=False, na_rep='NA'); GBA.to_csv(OUT + 'ugb_gb_only.tsv', sep='\t', index=False); pd.DataFrame(SUBGATE).to_csv(OUT + 'subgate_gb_only.tsv', sep='\t', index=False)
pd.DataFrame([dict(endpoint=l, nU_GB=int(X['U'].sum()), nU_current_S1=int(X['U1'].sum()), nU_S2=int(X['U2'].sum()), n_pos_GB=int((X['y'] & X['U']).sum()), n_pos_current=int((X['y'] & X['U1']).sum())) for l, X in D.items()]).to_csv(OUT + 'universe_sizes.tsv', sep='\t', index=False)


def agg(getter, eps):
    ps = {}
    for s in SYS4:
        by = {}
        for lab in eps:
            if SYSOF[lab] != s: continue
            v = getter(lab)
            if v is not None: by.setdefault(TFOF[lab], []).append(v)
        if by: ps[s] = np.mean([np.mean(v, 0) for v in by.values()], 0)
    return (np.mean(list(ps.values()), 0) if ps else None), ps


S = []
for sname, eps in (('all9', list(D)), ('quality4', [q for q in QUALITY if q in D])):
    for d in DIRS:
        for mt in MT:
            rows = []
            for m in EXT:
                q = PP[(PP.comparator == m) & (PP.direction == d) & (PP.metric == mt) & PP.endpoint.isin(eps)]
                r, ps = agg(lambda l: DIFF.get((l, m, d, mt)), eps)
                row = dict(set=sname, comparator=m, direction=d, metric=mt, n_endpoints=len(q), n_nom_gt0=int((q.sig_nominal == '+').sum()), n_nom_lt0=int((q.sig_nominal == '-').sum()), n_nom_incl0=int((q.sig_nominal == '0').sum()),
                           n_undefined=int((q.sig_nominal == 'U').sum()), n_holm_gt0=int((q.sig_holm == '+').sum()), n_holm_lt0=int((q.sig_holm == '-').sum()),
                           n_nom_gt0_npos20=int(((q.sig_nominal == '+') & ~q.flag_descriptive.astype(bool)).sum()), n_nom_lt0_npos20=int(((q.sig_nominal == '-') & ~q.flag_descriptive.astype(bool)).sum()))
                if r is not None:
                    sg, lo, hi, nv, p = classify(r); row.update(agg_diff=r[0], agg_lo=lo, agg_hi=hi, agg_p=p, agg_sig_nominal=sg, agg_n_valid_draws=nv, agg_n_systems=len(ps), agg_sys=';'.join(ps))
                rows.append(row)
            ph = holm([r_.get('agg_p', np.nan) for r_ in rows])
            for r_, h in zip(rows, ph):
                r_['agg_p_holm8'] = h; r_['agg_sig_holm'] = 'U' if r_.get('agg_sig_nominal') == 'U' else ('NA' if 'agg_diff' not in r_ else ('+' if h < 0.05 and r_['agg_diff'] > 0 else '-' if h < 0.05 and r_['agg_diff'] < 0 else '0'))
            S += rows
pd.DataFrame(S).to_csv(OUT + 'summary.tsv', sep='\t', index=False, na_rep='NA')

# ---------------- BM_GATA1_acute truth-design sensitivity on U_GB (pairwise) (internal review #3 of next-2-weeks list)
if 'BM_GATA1_acute' in D:
    X = D['BM_GATA1_acute']; g = X['g']; U = X['U']
    TT = pd.read_csv(TTF, sep='\t'); TT['G'] = TT.gene.astype(str).str.upper(); assert not TT.G.duplicated().any(); TT = TT.set_index('G')
    DES = {'current': None, 'DEPOSITED_table': 'DEPOSITED', 'C': 'C_all8_group', 'B': 'B_24h_donor_cond', 'D': 'D_all8_donor_group'}
    def labels(col):
        if col is None: return X['y'].copy(), X['e'].copy()
        s = pd.Series(g).map(TT[col]).fillna('ns').to_numpy()     # genes absent from table = non-responder; universe already restricted to the deposited tested set (resp_ok)
        return np.isin(s, ['UP', 'DOWN']) & U, np.where(s == 'UP', 1.0, np.where(s == 'DOWN', -1.0, np.nan))
    yd, ed = labels('DEPOSITED'); assert (yd[U] == X['y'][U]).all() and (np.sign(ed[U & yd]) == np.sign(X['e'][U & X['y']])).all(), 'DEPOSITED column != canonical labels on U_GB'
    BR = []
    for dn, col in DES.items():
        yy, ee = labels(col)
        for d in DIRS:
            for m in EXT:
                P = U & np.isfinite(X['V'][m]); npos = npos_of('BM_GATA1_acute', d, P, yy, ee)
                if npos < GATE: BR.append(dict(design=dn, comparator=m, direction=d, metric='SUBGATE', n_pos=npos)); continue
                vg, _, nrow = ev('BM_GATA1_acute', 'GeoBRIDGE', d, P, yy, ee); vm, _, _ = ev('BM_GATA1_acute', m, d, P, yy, ee)
                for mt in MT:
                    df = vg[mt] - vm[mt]; sg, lo, hi, nv, p = classify(df)
                    BR.append(dict(design=dn, comparator=m, direction=d, metric=mt, n_P=int(P.sum()), n_pos=npos, GB=vg[mt][0], m=vm[mt][0], diff=df[0], lo=lo, hi=hi, p_boot=p, n_valid_draws=nv, sig_nominal=sg))
    BR = pd.DataFrame(BR); BR['p_holm8'] = np.nan
    for _, idx in BR[BR.metric != 'SUBGATE'].groupby(['design', 'direction', 'metric']).groups.items(): BR.loc[idx, 'p_holm8'] = holm(BR.loc[idx, 'p_boot'].values)
    BR.to_csv(OUT + 'bm_acute_design_pairwise.tsv', sep='\t', index=False, na_rep='NA')
json.dump(dict(B=B, endpoints=[x[0] for x in EPS], methods=MS, quality_subset=QUALITY, bootstrap='module resampling G/50 (C[B:2B]), per SPEC_ADDENDUM', spec_sha8='8bbe6a88ecf080ac', S1_parity_ok=bool(ok1), S2_parity_ok=bool(ok2), valid_frac=VALID_FRAC, hashes=HASHES), open(OUT + 'run.json', 'w'), indent=1)
print('DONE')
