"""E11 pair-window sensitivity on Fig 2 SERGIO TF32REP (see README). Rebuilds forward pairs from STORED per-root pseudotime with the exact
canonical_pairs post-DPT logic (stable-id canonical order, lexsort by pt, pairspace.build_forward_pairs, same seed); production setting must
reproduce stored pairs exactly. Scoring = gb_matched_baseline_sergio CONC path (zcons depths 0..8, min over roots). Evaluation only.
usage: script OUT --grns 501,.. [--topos ..] [--conds ..] [--sims ..] [--workers N] [--configs prod,..]"""
import sys, os, json, time, numpy as np, pandas as pd, importlib.util
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch')]
import gb_tf32rep_fast_2026_10_22 as D, gb_primary as GP, gb_canonical as GC
import anndata as ad
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as APs
from multiprocessing import Pool
a = sys.argv; OUT = a[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
arg = lambda k, d: a[a.index(k) + 1].split(',') if k in a else d
GRNS = [int(x) for x in arg('--grns', ['501'])]; TOPOS = arg('--topos', ['bifurcation', 'tree']); CONDS = arg('--conds', ['C0', 'C1', 'C2', 'C3'])
SIMS = [int(x) for x in arg('--sims', ['1', '2'])]; NW = int(arg('--workers', ['1'])[0])
LOS = [0, .01, .02, .05]; HIS = [.10, .15, .25, .40, .60]
CFG = [(lo, hi, 40) for lo in LOS for hi in HIS] + [(.02, .25, c) for c in (10, 20, 80)]
if '--configs' in a: CFG = [tuple(float(v) if i < 2 else int(v) for i, v in enumerate(s.split(':'))) for s in arg('--configs', [])]
PROD = (.02, .25, 40); KS = range(9); RT = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/results/sergio_tf32rep_2026_10_22')
PC = json.load(open(GC.CONFIG))['pair_construction']; assert (PC['min_forward_dt'], PC['max_forward_dt'], PC['max_pairs_per_cell']) == PROD, PC
spec = importlib.util.spec_from_file_location('ps', (__import__("os").environ["GEOBRIDGE_ROOT"] + '/geobridge_model/pairspace.py')); PS = importlib.util.module_from_spec(spec); spec.loader.exec_module(PS)
TAB = pd.read_csv(f'{RT}/TABLE.tsv', sep='\t').query("method=='GB'").set_index(['grn', 'topo', 'cond', 'sim'])
dense = lambda M: np.asarray(M.todense() if hasattr(M, 'todense') else M, float)
def mkpairs(ptv, ids, lo, hi, cap):
    can = np.argsort(np.asarray(ids).astype(str), kind='stable'); ptc = ptv[can]; o2 = np.lexsort((np.arange(len(ptc)), ptc)); n = len(ptc)
    pr = PS.build_forward_pairs(np.zeros((n, 1)), np.zeros((n, 1)), ptc[o2], min_forward_dt=lo, max_forward_dt=hi, max_pairs_per_cell=cap, pair_seed=PC['pair_seed'])
    return can[o2[np.asarray(pr.pair_i, int)]], can[o2[np.asarray(pr.pair_j, int)]]
def conc(Ms, Mu, P30, I, J, src):
    Q, _ = np.linalg.qr(P30[J] - P30[I]); dX = Ms[J][:, src] - Ms[I][:, src]; dY = Mu[J] - Mu[I]; out = []
    for k in KS:
        x = dX - Q[:, :k] @ (Q[:, :k].T @ dX) if k else dX; y = dY - Q[:, :k] @ (Q[:, :k].T @ dY) if k else dY
        out.append(np.abs(x.T @ y) / (np.abs(x).T @ np.abs(y)))
    return np.stack(out)
def one(task):
    g, t, c, s = task; t0 = time.time()
    DM = D.dist_matrix(sorted((int(x), int(y)) for x, y in D.V_().generate_grn(g)['gt']))
    dd = D.dsdir(g, t, c, s); Z = np.load(f'{dd}/raw/geobridge_raw.npz'); Gm = np.load(f'{dd}/raw/geometry_raw.npz')
    src = np.asarray(Z['source_gene_index'], int); cols = np.asarray(Z['cols_PI'], int)
    A = ad.read_h5ad(f'{dd}/{os.path.basename(dd)}.h5ad'); ids = np.asarray(A.obs_names).astype(str)
    Ms, Mu = dense(A.layers['Ms']), dense(A.layers['Mu']); P30 = np.asarray(Gm['production_pca30'], float)
    Dm = np.array([[DM[s_, int(x)] for x in cols[i]] for i, s_ in enumerate(src)]); Y = Dm == 1; assert Y.sum() == 240
    rows = []; par = 0.
    for (lo, hi, cap) in CFG:
        Sr, npairs = {}, {}
        for r in GP.ROOTS:
            I, J = mkpairs(np.asarray(Z[f'pairs|{r}|pt'], float), ids, lo, hi, cap)
            if (lo, hi, cap) == PROD:
                assert np.array_equal(I, np.asarray(Gm[f'pairs_{r}_I'], int)) and np.array_equal(J, np.asarray(Gm[f'pairs_{r}_J'], int)), ('PAIR PARITY', dd, r)
                assert np.array_equal(I, np.asarray(Z[f'pairs|{r}|I'], int)) and np.array_equal(J, np.asarray(Z[f'pairs|{r}|J'], int))
            Sr[r] = conc(Ms, Mu, P30, I, J, src); npairs[r] = len(I)
            if (lo, hi, cap) == PROD:
                st = np.asarray(Z[f'TRUE|U|{r}|absS'], float)[:9]; m = np.isfinite(st)
                for i in range(len(src)): m[:, i, src[i]] = False
                d_ = np.nanmax(np.abs(Sr[r][m] - st[m])); par = max(par, d_); assert d_ < 1e-4, ('CONC parity', dd, r, d_)
        for lab, ks in (('k08', list(KS)), ('k0', [0])):
            sc = np.vstack([GP.zcons({r: Sr[r][:, i, cols[i]].T for r in GP.ROOTS}, ks) for i in range(len(src))]); ok = np.isfinite(sc)
            if (lo, hi, cap) == PROD and lab == 'k08':
                ap = APs(Y[ok], sc[ok]); assert abs(ap - TAB.loc[(g, t, c, s)].AP) < 1e-6, ('TABLE parity', dd, ap); par = max(par, 0.)
            rr = dict(grn=g, topo=t, cond=c, sim=s, lo=lo, hi=hi, cap=cap, arm=f'GB_{lab}', coverage=ok.mean(), AP=APs(Y[ok], sc[ok]),
                      n_pairs_min=npairs['dc1_min'], n_pairs_max=npairs['dc1_max'], conc_parity=par if (lo, hi, cap) == PROD else np.nan)
            mm = ok & ((Dm == 1) | np.isin(Dm, (2, 3))); rr['DvI23_AUROC'] = AUC(Dm[mm] == 1, sc[mm]); rows.append(rr)
    print('done', task, f'{time.time()-t0:.0f}s', flush=True); return rows
if __name__ == '__main__':
    T = [(g, t, c, s) for g in GRNS for t in TOPOS for c in CONDS for s in SIMS]; t0 = time.time()
    with Pool(NW) as p: R = [r for rs in p.imap_unordered(one, T) for r in rs]
    pd.DataFrame(R).to_csv(OUT + 'datasets.tsv', sep='\t', index=False)
    json.dump(dict(wall_s=time.time() - t0, n_tasks=len(T), grns=GRNS, cfg=CFG), open(OUT + 'receipt.json', 'w'))
    print('WALL', time.time() - t0)
