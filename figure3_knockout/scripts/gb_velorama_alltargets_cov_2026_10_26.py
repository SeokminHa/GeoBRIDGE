"""Coverage ledger only (no AUROC/AP): U_GB coverage of Velorama native vs alltargets_rt0 for the 5 main endpoints, by TF-status x KO direction.
Derived from scratch/gb_fig3_main5_mask_2026_10_26.py loader. usage: script MODE(v2|noF2) OUT.tsv"""
import sys, ast, numpy as np, pandas as pd
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
sys.path[:0] = [R + 'scratch', R + 'pairspace']
MODE, OUT = sys.argv[1], sys.argv[2]
SYSD = {'PANC': 'PANC', 'ERYTHROID': 'GATA1', 'BM': 'BONEMARROW', 'HEART': 'CARDIACV'}
def vmap(path, tf, genes):
    zz = np.load(path, allow_pickle=True); s_ = [x.upper() for x in zz['regulators'].astype(str)]
    gg = {g: i for i, g in enumerate(zz['genes'].astype(str))}; row = np.asarray(zz['score_all'], float)[s_.index(tf.upper())]
    return np.array([row[gg[g]] if g in gg else np.nan for g in genes])
NF = R + 'results/signed/bm_acute_design_audit_2026_10_26/noF2'
if MODE == 'noF2': sys.path.insert(0, NF)
import gb_layer_lattice as L
from gb_fig3d_magnitude import up, METH, VEC, effect as EFF0
import gb_ptable as PT, gb_lattice_extra as LE
LE.register(); import gb_calbm_propagate as CP; CP.preflight()
if MODE == 'noF2':
    import bm_acute_noF2_adapter as AD; AD.install_noF2()
ns = {}
for node in ast.parse(open(R + 'scratch/gb_ms_fig3_set9_2026_10_19.py').read()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('EP6', 'EP9') for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<roster>', 'exec'), ns)
EP9 = ns['EP9']
en = dict(R=R, pd=pd, EFF0=EFF0)
fn = next(n for n in ast.parse(open(R + 'scratch/gb_ms_fig3_set9_sens_2026_10_20.py').read()).body if isinstance(n, ast.FunctionDef) and n.name == 'effect_fixed')
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<effect>', 'exec'), en); effect = en['effect_fixed']
FLIP = {'GATA1': -1.0}
ONLY = ['BM_GATA1_acute'] if MODE == 'noF2' else ['PANC_RFX6', 'HEART_TBX5', 'HEART_NKX2-5', 'ERY_KLF1_B']
import os
if os.environ.get("ONLY"): ONLY = os.environ["ONLY"].split(",")
COV = []
import gb_substrate as GS
for lab, sysn, esys, tf, dsys, stem in [x for x in EP9 if x[0] in ONLY]:
    z = np.load(f'{VEC}ep_{stem}.vec.npz', allow_pickle=True)
    g_raw = z['genes'].astype(str); g = up(z['genes']); y = z['y'].astype(bool); GB = z['cog::GeoBRIDGE'].astype(float)
    nat = vmap(R + f'results/real_uniform/{SYSD[sysn]}/velorama_v2_native_rt0.npz', tf, g_raw)
    alt = vmap(R + f'results/real_uniform/{SYSD[sysn]}/velorama_v2_alltargets_rt0.npz', tf, g_raw)
    dup = pd.Series(g).duplicated(keep=False).to_numpy(); selfm = (g == tf.upper())
    cap = {}; orig = L.assemble
    def _a(s_, t_, g_, V_, truth_, *aa, **kk): cap['t'] = truth_; return orig(s_, t_, g_, V_, truth_, *aa, **kk)
    L.assemble = _a
    try: ru = PT.lattice_unit(dsys, tf, PT.lattice_draws(dsys, 2, sizes=(50,))[0].gene)
    finally: L.assemble = orig
    assert np.array_equal(np.asarray(ru['genes']).astype(str), g_raw)
    rok = np.asarray(cap['t']['resp_ok'], bool)
    if MODE == 'noF2':
        y = np.asarray(cap['t']['resp'], bool).copy(); e = pd.Series(g).map(AD.effect_norm()).to_numpy(float)
    else:
        e = pd.Series(g).map(pd.Series(FLIP.get(esys, 1.0) * effect(esys, tf))).to_numpy(float)
    U = rok & ~dup & ~selfm & np.isfinite(GB)
    cat = {l.strip().upper() for l in open(GS.paths(SYSD[sysn], canonical=True)[2]) if l.strip()}
    tfm = np.isin(g, list(cat))
    print(lab, 'vec Velorama == native npz mapping:', bool(np.array_equal(nat, z['cog::Velorama'].astype(float), equal_nan=True)),
          '| alltargets finite', int(np.isfinite(alt).sum()), 'of', len(alt), flush=True)
    for cls, cm in [('KOup', y & (e > 0)), ('KOdown', y & (e < 0)), ('noresp', ~y), ('all', np.ones_like(y))]:
        for ts, tm in [('TF', tfm), ('nonTF', ~tfm), ('both', np.ones_like(tfm))]:
            sel = U & cm & tm
            COV.append(dict(endpoint=lab, cls=cls, tf=ts, n_UGB=int(sel.sum()), n_native=int((sel & np.isfinite(nat)).sum()), n_alltargets=int((sel & np.isfinite(alt)).sum()),
                            frac_alltargets=(float((sel & np.isfinite(alt)).sum()) / sel.sum()) if sel.sum() else np.nan))
pd.DataFrame(COV).to_csv(OUT, sep='\t', index=False)
