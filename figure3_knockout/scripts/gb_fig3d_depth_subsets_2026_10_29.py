"""Fig 3d depth-aggregation check (internal review 2026-10-29 next-check 1). Is the production gain (zcons over k=0..8) a property of AGGREGATING depths or of the deeper depths?
Reuses the frozen rebuild prefix (rows, truth, draws unchanged; PANC_RFX6 only). Recomputes zcons from results/signed/PANC.npz exactly as signed_depths()
(gb_fig3d_rebuild_2026_10_29.py) but over depth SUBSETS: cumulative {0..j}, tail {j..8}, halves {0-4},{5-8}, even/odd, drop-one. Parity: subset {0..8} == arm D.
Metric: ALL-responder AUROC and AUPRC on the common rows with the synchronized module draws (CT.eval_rows); paired difference vs K0 per subset.
Descriptive (no new family claim).  usage (server, repo root): script OUT"""
import sys, os
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
OUT = sys.argv[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True); OUT0 = OUT
src = open(R + 'scratch/gb_fig3d_rebuild_2026_10_29.py').read().replace('\r', '')
src = src[:src.index("MT = ('AUROC', 'APLIFT', 'AUPRC'); DRAW = {}")]
sys.argv = ['x', OUT + '_rb', '--B', '9999', '--endpoints', 'PANC_RFX6']
exec(compile(src, 'rebuild_prefix', 'exec'))
OUT = OUT0
lab = 'PANC_RFX6'; X = D[lab]['X']; g = np.asarray(X['g']).astype(str); fx, tf = SYSF[lab]
z = np.load(f'{R}results/signed/{fx}.npz', allow_pickle=True); srcs = [s.upper() for s in z['sources'].astype(str)]; i = srcs.index(tf.upper()); genes = z['genes'].astype(str)
A = {r: np.abs(np.asarray(z[f'{r}|U'][i], float)) for r in GP.ROOTS}
u_ = z['uni'][i] if z['uni'].ndim == 1 and z['uni'].dtype == object else np.asarray(z['uni'])[i]; u_ = np.asarray(u_)
if len(u_) == len(genes) and set(np.unique(u_.astype(object)).tolist()) <= {True, False}: uni = u_.astype(bool)
else: uni = np.zeros(len(genes), bool); uni[u_.astype(int)] = True
def sub(ks):
    full = np.full(len(genes), np.nan); full[uni] = GP.zcons({r: A[r][uni] for r in GP.ROOTS}, list(ks)); return mapto(genes, full, g)
SUBS = {f'cum0-{j}': range(j + 1) for j in range(9)}
SUBS.update({f'tail{j}-8': range(j, 9) for j in range(1, 9)}); SUBS.update({'half0-4': range(5), 'half5-8': range(5, 9), 'even': range(0, 9, 2), 'odd': range(1, 9, 2)})
SUBS.update({f'drop{j}': [k for k in range(9) if k != j] for j in range(9)})
DEEP = os.environ.get('GB_DEEP') == '1'   # PI 2026-10-29: extend the cumulative and single-depth views to k=30 (signed cache holds k=0..30); output panc_depth_deep.tsv
if DEEP: SUBS = {**{f'cum0-{j}': range(j + 1) for j in range(31)}, **{f'single{j}': [j] for j in range(31)}}
V = {nm: sub(ks) for nm, ks in SUBS.items()}

par = np.nanmax(np.abs(V['cum0-8'] - AV[lab]['main']['D'])[D[lab]['Uc']]); par0 = np.nanmax(np.abs(V['cum0-0'] - AV[lab]['main']['K0'])[D[lab]['Uc']])
print('parity cum0-8 vs D', par, '| cum0-0 vs K0', par0); assert par < 1e-6 and par0 < 1e-6
y0 = D[lab]['y0']; rr = D[lab]['Uc']; yl = y0.copy(); ROWS = []
NF = {nm: int((~np.isfinite(v[rr])).sum()) for nm, v in V.items()}
if DEEP: print('rows lost to non-finite support per subset (deep):', {k: n for k, n in NF.items() if n})
else: assert not any(NF.values()), 'subset support differs from common rows'
if DEEP: rr = rr & np.logical_and.reduce([np.isfinite(v) for v in V.values()])   # one common support across every deep subset (reported)
pk, bk = CT.eval_rows(V['cum0-0'], yl, rr, X['gi'], X['C'])
for nm, v in V.items():
    p, b = CT.eval_rows(v, yl, rr, X['gi'], X['C'])
    for m in ('AUROC', 'AUPRC'):
        dd = np.asarray(b[m]) - np.asarray(bk[m]); ok = np.isfinite(dd)
        ROWS.append(dict(subset=nm, n_depths=len(list(SUBS[nm])), metric=m, value=p[m], lo=np.nanpercentile(b[m], 2.5), hi=np.nanpercentile(b[m], 97.5),
                         minus_K0=p[m] - pk[m], d_lo=np.percentile(dd[ok], 2.5), d_hi=np.percentile(dd[ok], 97.5), p_boot=PT.boot_p(dd[ok])))
T = pd.DataFrame(ROWS); T['n_rows'] = int(rr.sum()); T['n_pos'] = int((yl & rr).sum()); T.to_csv(OUT + ('panc_depth_deep.tsv' if DEEP else 'panc_depth_subsets.tsv'), sep='\t', index=False)
pd.set_option('display.width', 220); pd.set_option('display.max_rows', 200); print(T[T.metric == 'AUROC'].round(3).to_string(index=False)); print('n_rows', int(rr.sum()), 'n_pos', int((yl & rr).sum()))
