"""EVAL ONLY (no score recompute): PANC_RFX6 point AUROC/AUPRC on U_GB ∩ finite(m), Velorama = alltargets_rt0 (and native for reference), plus common set. Loader copied from scratch/gb_velorama_alltargets_cov_2026_10_26.py."""
import sys, ast, numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
sys.path[:0] = [R + 'scratch', R + 'pairspace']
def vmap(path, tf, genes):
    zz = np.load(path, allow_pickle=True); s_ = [x.upper() for x in zz['regulators'].astype(str)]
    gg = {g: i for i, g in enumerate(zz['genes'].astype(str))}; row = np.asarray(zz['score_all'], float)[s_.index(tf.upper())]
    return np.array([row[gg[g]] if g in gg else np.nan for g in genes])
import gb_layer_lattice as L
from gb_fig3d_magnitude import up, METH, VEC, effect as EFF0
import gb_ptable as PT, gb_lattice_extra as LE
LE.register(); import gb_calbm_propagate as CP; CP.preflight()
ns = {}
for node in ast.parse(open(R + 'scratch/gb_ms_fig3_set9_2026_10_19.py').read()).body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('EP6', 'EP9') for t in node.targets):
        exec(compile(ast.Module(body=[node], type_ignores=[]), '<roster>', 'exec'), ns)
en = dict(R=R, pd=pd, EFF0=EFF0)
fn = next(n for n in ast.parse(open(R + 'scratch/gb_ms_fig3_set9_sens_2026_10_20.py').read()).body if isinstance(n, ast.FunctionDef) and n.name == 'effect_fixed')
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<effect>', 'exec'), en); effect = en['effect_fixed']
MS = [m for m in METH if m != 'RegVelo |J|']
lab, sysn, esys, tf, dsys, stem = next(x for x in ns['EP9'] if x[0] == 'PANC_RFX6')
z = np.load(f'{VEC}ep_{stem}.vec.npz', allow_pickle=True)
g_raw = z['genes'].astype(str); g = up(z['genes']); y = z['y'].astype(bool)
V = {m: z[f'cog::{m}'].astype(float) for m in MS}
nat = vmap(R + 'results/real_uniform/PANC/velorama_v2_native_rt0.npz', tf, g_raw)
alt = vmap(R + 'results/real_uniform/PANC/velorama_v2_alltargets_rt0.npz', tf, g_raw)
assert np.array_equal(nat, V['Velorama'], equal_nan=True)
V['Velorama'] = alt; V['Velorama(native)'] = nat
cap = {}; orig = L.assemble
def _a(s_, t_, g_, V_, truth_, *aa, **kk): cap['t'] = truth_; return orig(s_, t_, g_, V_, truth_, *aa, **kk)
L.assemble = _a
try: ru = PT.lattice_unit(dsys, tf, PT.lattice_draws(dsys, 2, sizes=(50,))[0].gene)
finally: L.assemble = orig
assert np.array_equal(np.asarray(ru['genes']).astype(str), g_raw)
rok = np.asarray(cap['t']['resp_ok'], bool)
e = pd.Series(g).map(pd.Series(effect(esys, tf))).to_numpy(float)
dup = pd.Series(g).duplicated(keep=False).to_numpy()
U = rok & ~dup & (g != tf.upper()) & np.isfinite(V['GeoBRIDGE'])
COMMON = U & np.all([np.isfinite(V[m]) for m in MS], 0)
rows = []
for setn in ('pairwise', 'common'):
    for m in MS + ['Velorama(native)']:
        mask = (U & np.isfinite(V[m])) if setn == 'pairwise' else (COMMON & np.isfinite(V[m]))
        for d, pos in [('KO-responsive', y), ('KO-up', y & (e > 0)), ('KO-down', y & (e < 0))]:
            sel = mask & (pos | ~y); yy = pos[sel]
            for mm in ('GeoBRIDGE', m):
                pass
            s = V[m][sel]; sg = V['GeoBRIDGE'][sel]
            rows.append(dict(set=setn, method=m, dir=d, n_pos=int(yy.sum()), n_neg=int((~yy).sum()),
                AUROC=roc_auc_score(yy, s), AUPRC=average_precision_score(yy, s), prev=yy.mean(),
                GB_AUROC=roc_auc_score(yy, sg), GB_AUPRC=average_precision_score(yy, sg)))
print('nU_GB', int(U.sum()), 'common', int(COMMON.sum()), 'alt finite in U', int((U & np.isfinite(alt)).sum()))
pd.DataFrame(rows).to_csv(sys.argv[1], sep='\t', index=False, float_format='%.4f')
