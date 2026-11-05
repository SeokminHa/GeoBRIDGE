"""Diagnose why FastTENET (and GB) heart AUROC rose between the 10/19-23 SET9 Fig 3 and the 10/26+ U_GB evaluator. EVALUATION ONLY.
Both pipelines read the SAME cached ep_<stem>.vec.npz (labels y, cog::<method> columns). Differences checked here, one at a time, on the
new evaluator's data dict (gb_fig3_ugb_v2_velalltargets prefix):
  universes: U_GB (new primary) ; U_OLD = U_GB & all(MS finite) with the OLD cached Velorama column (cog::Velorama) = old SET9 universe ;
             DROPPED = U_GB & ~U_OLD ; U_NEW1 = U_GB & all(MS finite) with the alltargets Velorama (new common rows)
  per universe: n, n_pos, prevalence, point AUROC (Mann-Whitney, average ties) for every method; plus within-DROPPED AUROC.
  score identity: FastTENET column in evaluator D vs the raw vec file (max |diff|).
usage (server, repo root): script OUT"""
import sys, os
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
OUT = sys.argv[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
CUT = 'for s in SYS4:\n    cs = '
src = open(R + 'scratch/gb_fig3_ugb_v2_velalltargets_2026_10_26.py').read().replace('\r', ''); src = src[:src.index(CUT)]
ns = {'__file__': 'x', '__name__': 'prefix'}; sys.argv = ['x', OUT + '_scr', '--B', '20', '--endpoints', 'PANC_RFX6,HEART_TBX5,HEART_NKX2-5,ERY_KLF1_B']
exec(compile(src, 'prefix', 'exec'), ns)
import numpy as np, pandas as pd
from scipy.stats import rankdata
VEC = ns['VEC']; EP = {x[0]: x for x in ns['EP9']}; MS = ns['MS']
def auroc(s, y):
    ok = np.isfinite(s); s, y = s[ok], y[ok]; P, N = y.sum(), (~y).sum()
    if P == 0 or N == 0: return np.nan
    r = rankdata(s); return (r[y].sum() - P * (P + 1) / 2) / (P * N)
ROWS = []
for lab, X in ns['D'].items():
    z = np.load(f'{VEC}ep_{EP[lab][5]}.vec.npz', allow_pickle=True)
    oldvel = z['cog::Velorama'].astype(float); y = np.asarray(X['y'], bool); V = X['V']; U = np.asarray(X['U'], bool)
    ft = z['cog::FastTENET'].astype(float); b = np.isfinite(ft) & np.isfinite(V['FastTENET'])
    print(lab, 'FastTENET D vs vec max|diff|', float(np.abs(ft[b] - V['FastTENET'][b]).max()), 'finite-pattern equal', bool(np.array_equal(np.isfinite(ft), np.isfinite(V['FastTENET']))), flush=True)
    others = [m for m in MS if m != 'Velorama']
    U_OLD = U & np.all([np.isfinite(V[m]) for m in others], 0) & np.isfinite(oldvel)
    U_NEW1 = U & np.all([np.isfinite(V[m]) for m in MS], 0)
    UNI = dict(U_GB=U, U_OLD=U_OLD, DROPPED=U & ~U_OLD, U_NEW1=U_NEW1)
    for un, msk in UNI.items():
        for m in MS + ['Velorama_old']:
            s = oldvel if m == 'Velorama_old' else np.asarray(V[m], float)
            mm = msk & np.isfinite(s)
            ROWS.append(dict(endpoint=lab, universe=un, method=m, n=int(mm.sum()), n_pos=int((y & mm).sum()), prev=(y & mm).sum() / max(mm.sum(), 1), AUROC=auroc(s[mm], y[mm])))
    # what kind of genes did the old Velorama mask drop? detection proxy = GB finite rank of |score| is not truth-free-relevant; report y-prevalence + FastTENET score level
    d = UNI['DROPPED']
    print(lab, 'U_GB', int(U.sum()), 'U_OLD', int(U_OLD.sum()), 'DROPPED', int(d.sum()), 'prev U_OLD %.3f DROPPED %.3f' % (y[U_OLD].mean(), y[d].mean() if d.any() else np.nan),
          'median FastTENET U_OLD %.4g DROPPED %.4g' % (np.nanmedian(V['FastTENET'][U_OLD]), np.nanmedian(V['FastTENET'][d]) if d.any() else np.nan), flush=True)
T = pd.DataFrame(ROWS); T.to_csv(OUT + 'heart_jump_universe.tsv', sep='\t', index=False)
pd.set_option('display.width', 250)
P = T.pivot_table(index=['endpoint', 'method'], columns='universe', values='AUROC').round(3)
print(P[['U_OLD', 'U_GB', 'U_NEW1', 'DROPPED']].to_string())
print(T.pivot_table(index=['endpoint', 'universe'], values=['n', 'n_pos'], aggfunc='max').to_string())
