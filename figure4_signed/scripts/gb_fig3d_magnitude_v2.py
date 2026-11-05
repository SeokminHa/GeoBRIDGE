"""Fig3d v2 (evaluation only; frozen S/P/B1/comparators). usage: gb_fig3d_magnitude_v2.py OUT [B] [only_systems]
Repairs v1: ranks (score, |FC|, covariates) AND the partial-rank residualisation are recomputed INSIDE every G/50-module bootstrap draw
(weighted midranks under the draw's gene weights; responders-only = weights*responder mask). Paired contrasts on identical draws:
S - each of 9 comparators (Holm over 9), S-B1, P-B1; equal-system mean + system-LOSO."""
import sys, pickle, numpy as np, pandas as pd
from pathlib import Path
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch'), (__import__("os").environ["GEOBRIDGE_ROOT"] + '/pairspace')]
import gb_fig3d_magnitude as F
from gb_fig3d_magnitude import up, EPS, METH, BIOSYS, VEC, effect, pcorr


def inv_of(v): u, i = np.unique(v, return_inverse=True); return i, len(u)


def wrank(iv, w):
    i, ng = iv; gs = np.bincount(i, w, ng); return ((np.cumsum(gs) - gs) + (gs + 1) / 2)[i]


def stat2(iv, ifc, icv, w, adj):
    if (w > 0).sum() < 10: return np.nan
    a, b = wrank(iv, w), wrank(ifc, w)
    X = np.ones((len(a), 1)) if not adj else np.column_stack([np.ones(len(a))] + [wrank(c, w) for c in icv])
    return pcorr(a, b, X, w)


def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for k, j in enumerate(o): run = max(run, (m - k) * p[j]); adj[j] = min(1, run)
    return adj


def agg(vals): return np.nanmean(np.stack(vals), 0)


def pval(d):  # two-sided bootstrap p on draws d[1:]
    d = d[1:]; d = d[np.isfinite(d)]; return (1 + 2 * min((d <= 0).sum(), (d >= 0).sum())) / (len(d) + 1)


if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); B = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    only = sys.argv[3].split(',') if len(sys.argv) > 3 else None
    import gb_ptable as PT, gb_lattice_extra as LE
    LE.register(); import gb_calbm_propagate as CP; CP.preflight()
    bs = pd.read_csv(F.BS, sep='\t', usecols=['system', 'method', 'mode', 'gene', 'value', 'raw_det', 'abundance', 'B1_mu'])
    bsg = bs[(bs.method == 'GeoBRIDGE') & (bs['mode'] == 'P')]
    drc = {}; draws = {}; rows = []
    for sysn, sub, tf, tr in EPS:
        if only and sysn not in only: continue
        z = np.load(f'{VEC}ep_{sysn}__{tf}__{tr}.vec.npz', allow_pickle=True)
        g = up(z['genes']); M = z['MR'].astype(bool); gi = z['gi']; y = z['y'].astype(bool)
        eff = effect(sysn, tf); b = bsg[bsg.system == BIOSYS[sysn]]; b = b.assign(G=up(b.gene)).drop_duplicates('G', keep=False).set_index('G')
        dup = pd.Series(g).duplicated(keep=False).to_numpy()
        pred = {m: z[f'cog::{m}'].astype(float) for m in METH}
        pred['P'] = pd.Series(g).map(b['value']).to_numpy(float); pred['B1'] = pd.Series(g).map(b['B1_mu']).to_numpy(float)
        afc = np.abs(pd.Series(g).map(eff).to_numpy(float)); det = pd.Series(g).map(b['raw_det']).to_numpy(float); ab = pd.Series(g).map(b['abundance']).to_numpy(float)
        ok = M & ~dup & np.isfinite(afc) & np.isfinite(det) & np.isfinite(ab) & np.all([np.isfinite(v) for v in pred.values()], 0)
        n = int(ok.sum()); print(sysn, tf, 'universe', n, 'resp', int(y[ok].sum()), flush=True)
        if n < 30: rows.append(dict(system=sub, endpoint=sysn, tf=tf, n=n, status='NO_CALL_TOO_FEW')); continue
        if sysn not in drc: drc[sysn] = PT.lattice_draws(sysn, B, sizes=(50,))[0]
        W = drc[sysn].C[B:2 * B][:, gi[ok]].astype(float); yy = y[ok]
        icv = [inv_of(det[ok]), inv_of(ab[ok])]; ifc = inv_of(afc[ok])
        for m, v in pred.items():
            iv = inv_of(v[ok])
            for var, msk, adj in (('total_raw', None, 0), ('total_adj', None, 1), ('resp_raw', yy, 0), ('resp_adj', yy, 1)):
                mk = np.ones(n) if msk is None else msk.astype(float)
                draws[(sub, sysn, tf, m, var)] = np.r_[stat2(iv, ifc, icv, mk, adj), [stat2(iv, ifc, icv, W[i] * mk, adj) for i in range(B)]]
        rows.append(dict(system=sub, endpoint=sysn, tf=tf, n=n, n_resp=int(yy.sum()), status='OK'))
    pd.DataFrame(rows).to_csv(out / 'endpoints.tsv', sep='\t', index=False); pickle.dump(draws, open(out / 'draws.pkl', 'wb'))
    keys = pd.DataFrame(list(draws), columns=['sub', 'ep', 'tf', 'm', 'var'])
    SY = {}   # (m,var) -> {sub: draws}  (endpoint -> TF -> system)
    for m in keys.m.unique():
        for var in keys['var'].unique():
            sel = keys[(keys.m == m) & (keys['var'] == var)]
            SY[(m, var)] = {sub: agg([agg([draws[(sub, e, t, m, var)] for e in dd.ep]) for t, dd in d.groupby('tf')]) for sub, d in sel.groupby('sub')}
    def eq(m, var, drop=None): return agg([v for k, v in SY[(m, var)].items() if k != 'BM_ACUTE' and k != drop])
    ci = lambda d: (d[0], *np.nanpercentile(d[1:], [2.5, 97.5]))
    res = []
    for var in ('total_raw', 'total_adj', 'resp_raw', 'resp_adj'):
        for m in METH + ['P', 'B1']:
            for sub, d in SY[(m, var)].items(): res.append(dict(level='acute_alone' if sub == 'BM_ACUTE' else 'system', unit=sub, method=m, variant=var, **dict(zip(('est', 'lo', 'hi'), ci(d)))))
            res.append(dict(level='equal_system', unit='all', method=m, variant=var, **dict(zip(('est', 'lo', 'hi'), ci(eq(m, var))))))
        # paired contrasts (identical draws)
        comps = [m for m in METH if m != 'GeoBRIDGE']
        ds = [eq('GeoBRIDGE', var) - eq(m, var) for m in comps]; hp = holm([pval(d) for d in ds])
        for m, d, h in zip(comps, ds, hp): res.append(dict(level='contrast', unit='equal_system', method=f'S-{m}', variant=var, **dict(zip(('est', 'lo', 'hi'), ci(d))), p=pval(d), p_holm9=h))
        for nm, a_, b_ in (('S-B1', 'GeoBRIDGE', 'B1'), ('P-B1', 'P', 'B1')):
            d = eq(a_, var) - eq(b_, var); res.append(dict(level='contrast', unit='equal_system', method=nm, variant=var, **dict(zip(('est', 'lo', 'hi'), ci(d))), p=pval(d)))
        for drop in [k for k in SY[('GeoBRIDGE', var)] if k != 'BM_ACUTE']:   # system-LOSO
            for m in ('GeoBRIDGE', 'P', 'B1'):
                res.append(dict(level='LOSO', unit=f'drop_{drop}', method=m, variant=var, **dict(zip(('est', 'lo', 'hi'), ci(eq(m, var, drop))))))
            d = eq('GeoBRIDGE', var, drop) - eq('B1', var, drop); res.append(dict(level='LOSO', unit=f'drop_{drop}', method='S-B1', variant=var, **dict(zip(('est', 'lo', 'hi'), ci(d)))))
            best = max((m for m in comps), key=lambda m: eq(m, var, drop)[0]); d = eq('GeoBRIDGE', var, drop) - eq(best, var, drop)
            res.append(dict(level='LOSO', unit=f'drop_{drop}', method=f'S-bestcomp({best})', variant=var, **dict(zip(('est', 'lo', 'hi'), ci(d)))))
    R = pd.DataFrame(res); R.to_csv(out / 'fig3d_v2_results.tsv', sep='\t', index=False)
    pd.set_option('display.width', 250); print(R[(R.level.isin(['equal_system', 'contrast'])) & (R.method.isin(['GeoBRIDGE', 'P', 'B1']) | (R.level == 'contrast'))].round(3).to_string())
