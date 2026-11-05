"""Fig 3d multi-system depth profile (internal review 2026-10-29 scope review: PANC-only C is not acceptable; port C to HEART TBX5, HEART NKX2-5, BM GATA1).
Generalises scratch/gb_fig3d_depth_subsets_2026_10_29.py (PANC) to any rebuild endpoint and to the three response directions (ALL / KOup / KOdown, the rebuild's
labels(): KO-direction responders vs tested non-responders). Reuses the frozen rebuild prefix (rows, truth, synchronized module draws unchanged); recomputes
PRIMARY zcons from results/signed/<SYS>.npz exactly as signed_depths() over depth subsets:
  cum0-K (K=0..30), single k (k=0..30), equal-cardinality early4={0,1,2,3} vs late4={5,6,7,8} and early4b={1,2,3,4} vs late4.
Parity asserted: cum0-8 == arm D, cum0-0 == arm K0 on the common rows. One common finite support per endpoint across all subsets (losses reported).
Per subset x direction x metric: value, pointwise 95% CI, paired difference vs cum0-0 (K0) with CI and PT.boot_p. Extra paired contrasts on the same draws:
  late4 - early4, late4 - early4b, cum0-30 - cum0-8. Descriptive; no new family claim; PRIMARY unchanged.
usage (server, repo root): script OUT --endpoints A[,B] [--B 9999]"""
import sys, os
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
a = sys.argv; OUT = a[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True); OUT0 = OUT
EPS_ = a[a.index('--endpoints') + 1]; B_ = a[a.index('--B') + 1] if '--B' in a else '9999'
src = open(R + 'scratch/gb_fig3d_rebuild_2026_10_29.py').read().replace('\r', '')
src = src[:src.index("MT = ('AUROC', 'APLIFT', 'AUPRC'); DRAW = {}")]
sys.argv = ['x', OUT + '_rb', '--B', B_, '--endpoints', EPS_]
exec(compile(src, 'rebuild_prefix', 'exec'))
OUT = OUT0
SUBS = {**{f'cum0-{j}': range(j + 1) for j in range(31)}, **{f'single{j}': [j] for j in range(31)},
        'early4': range(0, 4), 'early4b': range(1, 5), 'late4': range(5, 9)}
PAIRS = [('late4', 'early4'), ('late4', 'early4b'), ('cum0-30', 'cum0-8'), ('cum0-8', 'late4'), ('cum0-8', 'early4b')]   # last two: internal review small-N review
def gate(dd):   # inherited validity rule (internal review small-N review): >=95% finite paired draws, else NaN interval/p; valid count exported
    ok = np.isfinite(dd); f = dd[ok]   # same draw set as the original rows (no draw-0 assumption)
    return (np.percentile(f, 2.5), np.percentile(f, 97.5), PT.boot_p(f), int(ok.sum())) if ok.mean() >= .95 else (np.nan, np.nan, np.nan, int(ok.sum()))
ROWS, PROWS, SUPP = [], [], []
for lab in EPS:
    X = D[lab]['X']; g = np.asarray(X['g']).astype(str); fx, tf = SYSF[lab]
    z = np.load(f'{R}results/signed/{fx}.npz', allow_pickle=True); srcs = [s.upper() for s in z['sources'].astype(str)]; i = srcs.index(tf.upper()); genes = z['genes'].astype(str)
    A = {r: np.abs(np.asarray(z[f'{r}|U'][i], float)) for r in GP.ROOTS}
    u_ = z['uni'][i] if z['uni'].ndim == 1 and z['uni'].dtype == object else np.asarray(z['uni'])[i]; u_ = np.asarray(u_)
    if len(u_) == len(genes) and set(np.unique(u_.astype(object)).tolist()) <= {True, False}: uni = u_.astype(bool)
    else: uni = np.zeros(len(genes), bool); uni[u_.astype(int)] = True
    def sub(ks):
        full = np.full(len(genes), np.nan); full[uni] = GP.zcons({r: A[r][uni] for r in GP.ROOTS}, list(ks)); return mapto(genes, full, g)
    V = {nm: sub(ks) for nm, ks in SUBS.items()}
    Uc = D[lab]['Uc']
    par = np.nanmax(np.abs(V['cum0-8'] - AV[lab]['main']['D'])[Uc]); par0 = np.nanmax(np.abs(V['cum0-0'] - AV[lab]['main']['K0'])[Uc])
    print(lab, 'parity cum0-8 vs D', par, '| cum0-0 vs K0', par0, flush=True); assert par < 1e-6 and par0 < 1e-6, lab
    fin = np.logical_and.reduce([np.isfinite(v) for v in V.values()]); lost = {nm: int((Uc & ~np.isfinite(v)).sum()) for nm, v in V.items()}
    SUPP.append(dict(endpoint=lab, n_common=int(Uc.sum()), n_deep_common=int((Uc & fin).sum()), lost_by_subset={k: v for k, v in lost.items() if v}))
    y0, e = D[lab]['y0'], D[lab]['e']
    for d in ('ALL', 'KOup', 'KOdown'):
        yl = y0.copy() if d == 'ALL' else (y0 & np.isfinite(e) & ((e > 0) if d == 'KOup' else (e < 0)))
        rr = Uc & fin & ((yl | ~y0) if d != 'ALL' else True); npos, nneg = int((yl & rr).sum()), int((~yl & rr).sum())
        if npos < 5 or nneg < 5: print(lab, d, 'skipped: n_pos', npos); continue
        EV = {nm: CT.eval_rows(v, yl, rr, X['gi'], X['C']) for nm, v in V.items()}; pk, bk = EV['cum0-0']
        for nm, (p, b) in EV.items():
            for m in ('AUROC', 'AUPRC'):
                bm_ = np.asarray(b[m]); dlo, dhi, pb, nv = gate(bm_ - np.asarray(bk[m])); fm = np.isfinite(bm_)
                ROWS.append(dict(endpoint=lab, direction=d, subset=nm, metric=m, value=p[m], lo=np.percentile(bm_[fm], 2.5) if fm.mean() >= .95 else np.nan,
                                 hi=np.percentile(bm_[fm], 97.5) if fm.mean() >= .95 else np.nan, minus_K0=p[m] - pk[m], d_lo=dlo, d_hi=dhi, p_boot=pb, n_valid=nv,
                                 n_rows=int(rr.sum()), n_pos=npos))
        for x1, x0 in PAIRS:
            for m in ('AUROC', 'AUPRC'):
                lo_, hi_, pb, nv = gate(np.asarray(EV[x1][1][m]) - np.asarray(EV[x0][1][m]))
                PROWS.append(dict(endpoint=lab, direction=d, contrast=f'{x1} - {x0}', metric=m, value=EV[x1][0][m] - EV[x0][0][m], lo=lo_, hi=hi_, p_boot=pb,
                                  n_valid=nv, n_rows=int(rr.sum()), n_pos=npos))
        print(lab, d, 'done n_pos', npos, flush=True)
T = pd.DataFrame(ROWS); T.to_csv(OUT + 'depth_multi.tsv', sep='\t', index=False); Pq = pd.DataFrame(PROWS); Pq.to_csv(OUT + 'depth_multi_pairs.tsv', sep='\t', index=False)
pd.DataFrame(SUPP).to_csv(OUT + 'depth_multi_support.tsv', sep='\t', index=False)
pd.set_option('display.width', 240); pd.set_option('display.max_rows', 400)
q = T[(T.metric == 'AUROC') & T.subset.isin(['cum0-0', 'cum0-4', 'cum0-8', 'cum0-12', 'cum0-20', 'cum0-30', 'early4', 'late4'])]
print(q.pivot_table(index=['endpoint', 'direction'], columns='subset', values='value').round(3).to_string()); print(Pq.round(3).to_string(index=False)); print(pd.DataFrame(SUPP).to_string(index=False))
