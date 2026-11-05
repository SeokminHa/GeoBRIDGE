"""Per-depth support check for the Fig 3d depth profile (internal review 2026-10-29: depth arms allow depth-specific finite-row filtering).
Reuses the frozen rebuild prefix (rows, truth, arms unchanged); exports, per endpoint x direction, n_rows / n_pos on the common rows and on
common & finite(K_k) for k=1..8. Identical support <=> every depth row equals the common row.  usage (server, repo root): script OUT"""
import sys, os
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
OUT = sys.argv[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
src = open(R + 'scratch/gb_fig3d_rebuild_2026_10_29.py').read().replace('\r', '')
src = src[:src.index("MT = ('AUROC', 'APLIFT', 'AUPRC'); DRAW = {}")]
sys.argv = ['x', OUT + '_rb', '--B', '10']
exec(compile(src, 'rebuild_prefix', 'exec'))
ROWS = []
for lab in EPS:
    y0, e = D[lab]['y0'], D[lab]['e']
    for d in ('ALL', 'KOup', 'KOdown'):
        yl = y0.copy() if d == 'ALL' else (y0 & np.isfinite(e) & ((e > 0) if d == 'KOup' else (e < 0))); rr = D[lab]['Uc'] & ((yl | ~y0) if d != 'ALL' else True)
        ROWS.append(dict(endpoint=lab, direction=d, arm='common', n_rows=int(rr.sum()), n_pos=int((yl & rr).sum())))
        for k, v in AV[lab]['depth'].items():
            rd = rr & np.isfinite(v); ROWS.append(dict(endpoint=lab, direction=d, arm=k, n_rows=int(rd.sum()), n_pos=int((yl & rd).sum())))
T = pd.DataFrame(ROWS); T.to_csv(OUT + 'depth_support.tsv', sep='\t', index=False)
c = T[T.arm == 'common'].set_index(['endpoint', 'direction']); T['identical'] = [(r.n_rows, r.n_pos) == tuple(c.loc[(r.endpoint, r.direction), ['n_rows', 'n_pos']]) for r in T.itertuples()]
print(T.pivot_table(index=['endpoint', 'direction'], columns='arm', values='n_rows').to_string()); print('ALL IDENTICAL:', bool(T.identical.all()))
