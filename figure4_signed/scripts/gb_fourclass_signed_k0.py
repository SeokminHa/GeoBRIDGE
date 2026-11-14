# four-class split by KO direction at k=0
import sys, os, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch'), (__import__("os").environ["GEOBRIDGE_ROOT"] + '/pairspace')]
import gb_fig3d_magnitude as F, gb_primary as GP
from gb_fig3d_magnitude import up, effect
SG = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/results/signed/'); FLIP = {'GATA1': -1.0}
RU = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/results/real_uniform/"); FIG = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/results/figures/meeting_2026_10_13/main_3sys/")
D1 = RU + "fourclass_dist/raw/scores_%s.npz"; D2 = RU + "fourclass_cluster_merged/cardiacv/raw/scores_%s.npz"
# unit: (file, effect system, tf, signed-cache system, group)
U = {'PANC_RFX6': (D1, 'PANC', 'RFX6', 'PANC', 'PANC'), 'GATA1_GATA1': (D1, 'GATA1', 'GATA1', 'GATA1', 'ERY'), 'KLF1_KLF1': (D1, 'KLF1', 'KLF1', 'GATA1', 'ERY'),
     'KLF1_B_KLF1': (D1, 'KLF1_B', 'KLF1', 'GATA1', 'ERY'), 'BM_GATA1_ACUTE_GATA1': (D1, 'BM_GATA1_ACUTE', 'GATA1', 'BONEMARROW', 'BM_ACUTE'),
     'BONEMARROW_RUNX1': (D1, 'BM_CHIPATLAS_CALTOT', 'RUNX1', 'BONEMARROW', 'BM'), 'BM_CHIPATLAS_KLF1': (D1, 'BM_CHIPATLAS_CALTOT', 'KLF1', 'BONEMARROW', 'BM'),
     'BM_CHIPATLAS_NFE2': (D1, 'BM_CHIPATLAS_CALTOT', 'NFE2', 'BONEMARROW', 'BM'), 'CARDIACV_TBX5': (D2, 'CARDIACV', 'TBX5', 'CARDIACV', 'HEART'),
     'CARDIACV_NKX2-5': (D2, 'CARDIACV', 'NKX2-5', 'CARDIACV', 'HEART')}
CL = ['A_act', 'A_rep', 'C_act', 'C_rep', 'B', 'D']
B = 1000
def pct(s):
    ok = np.isfinite(s); r = np.full(len(s), np.nan); r[ok] = rankdata(s[ok]) / ok.sum(); return r
def auc(x, y):   # P(x>y)+.5 ties
    if len(x) < 5 or len(y) < 5: return np.nan
    r = rankdata(np.concatenate([x, y])); return (r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y))
def contr(m):   # m: dict class->mean pct ; returns resp, occ (act/rep variant given by keys 'A','C')
    return (m['A'] + m['C'] - m['B'] - m['D']) / 2, (m['A'] + m['B'] - m['C'] - m['D']) / 2
# r: percentile ranks (oriented for panel); cl class labels; panel 'act'/'rep'; idx: dict class->index array (bootstrap) or None
def stats(r, cl, panel, idx=None):
    g = lambda c: (np.flatnonzero(cl == c) if idx is None else idx[c])
    P = {'A': r[g('A_' + panel)], 'C': r[g('C_' + panel)], 'B': r[g('B')], 'D': r[g('D')]}
    P = {k: v[np.isfinite(v)] for k, v in P.items()}
    m = {k: (v.mean() if len(v) else np.nan) for k, v in P.items()}; rs, oc = contr(m)
    return dict(resp=rs, occ=oc, **{f'pct_{k}': m[k] for k in 'ABCD'}, **{'A>B': auc(P['A'], P['B']), 'C>D': auc(P['C'], P['D']), 'A>C': auc(P['A'], P['C']), 'B>D': auc(P['B'], P['D'])})
def rawS(sc, tf, ai, gg):
    j = [s.upper() for s in sc['sources'].astype(str)].index(tf.upper()); uni = np.asarray(sc['uni'][j])
    uni = np.flatnonzero(uni.astype(bool)) if (uni.dtype == bool or (uni.dtype == object and isinstance(uni.flat[0], (bool, np.bool_)))) else uni.astype(int)
    inuni = np.zeros(len(ai), bool); inuni[uni] = True; out = {}
    for h in GP.ROOTS:
        A = np.where(inuni[:, None], sc[f'{h}|U'][j], np.nan); Z = A[:, :9]
        km = np.where(np.isfinite(Z).sum(1) >= 6, np.nanmean(np.where(np.isfinite(Z), Z, np.nan), 1), np.nan); out[h] = (A[:, 0], km)
    (a0, ak), (b0, bk) = out['dc1_min'], out['dc1_max']; axis_ix = {x: i for i, x in enumerate(ai)}
    mp = lambda full: np.array([full[axis_ix[x]] if x in axis_ix else np.nan for x in gg])
    return mp((a0 + b0) / 2), mp((ak + bk) / 2)
def build(unit):
    f, es, tf, bsy, grp = U[unit]; z = np.load(f % unit, allow_pickle=True); gg = up(z['genes']); cls = z['cls'].astype(str)
    sc = np.load(f'{SG}{bsy}.npz', allow_pickle=True)
    if tf.upper() not in [x.upper() for x in sc['sources'].astype(str)]: sc = np.load(f'{SG}{bsy}_bank.npz', allow_pickle=True); print(unit, 'USING BANK cache (k=0..8; uni = finite over both roots x all depths 0..8)')
    axis = np.char.upper(sc['genes'].astype(str))
    s0, sk = rawS(sc, tf, axis, gg); e = pd.Series(gg).map(FLIP.get(es, 1.0) * effect(es, tf)).to_numpy(float)
    dupg = pd.Series(gg).duplicated(keep=False).to_numpy()
    c6 = cls.copy().astype(object)
    for c in 'AC':
        m = cls == c
        c6[m & (e < 0) & ~dupg] = c + '_act'; c6[m & (e > 0) & ~dupg] = c + '_rep'; c6[m & ~((e < 0) | (e > 0)) | (m & dupg)] = 'X'
    return dict(unit=unit, grp=grp, gg=gg, cls=cls, c6=c6.astype(str), uns=np.asarray(z['GeoBRIDGE'], float), s0=s0, sk=sk, e=e)
def run(unit, rng, bs):
    d = build(unit); c6 = d['c6']; B1 = None
    if bs is not None:
        gm = d['gg']; b = bs.assign(G=up(bs.gene)).drop_duplicates('G', keep=False).set_index('G').B1_mu; B1 = pd.Series(gm).map(b).to_numpy(float)
    scores = {'unsigned': d['uns'], 'signed_k0': d['s0'], 'signed_kmean': d['sk']}
    if B1 is not None: scores['B1'] = B1
    rows_sz = [dict(unit=unit, **{c: int((c6 == c).sum()) for c in CL}, X_excluded=int((c6 == 'X').sum()), A_orig=int((d['cls'] == 'A').sum()), C_orig=int((d['cls'] == 'C').sum()),
                    n_genes=len(c6), fin_unsigned=int(np.isfinite(d['uns']).sum()), fin_k0=int(np.isfinite(d['s0']).sum()), fin_kmean=int(np.isfinite(d['sk']).sum()))]
    idx_b = [{c: rng.choice(np.flatnonzero(c6 == c), (c6 == c).sum()) if (c6 == c).any() else np.array([], int) for c in CL} for _ in range(B)]   # same draws for every score
    res = {}; pw = {}
    for nm, s in scores.items():
        for panel in ('act', 'rep'):
            o = s if nm in ('unsigned', 'B1') else (s if panel == 'act' else -s); r = pct(o)
            res[(nm, panel)] = (stats(r, c6, panel), [stats(r, c6, panel, ix) for ix in idx_b])
        if nm in ('signed_k0', 'signed_kmean'):   # direction within ChIP-bound / unbound responders: AUROC(+S: act > rep)
            for a, b_, lab in (('A_act', 'A_rep', 'A_act>A_rep'), ('C_act', 'C_rep', 'C_act>C_rep')):
                f0 = auc(s[(c6 == a) & np.isfinite(s)], s[(c6 == b_) & np.isfinite(s)])
                bb = [auc(s[ix[a]][np.isfinite(s[ix[a]])], s[ix[b_]][np.isfinite(s[ix[b_]])]) for ix in idx_b]
                pw[(nm, lab)] = (f0, bb)
    # sanity: unsigned original-class fact reproduces fig5f_b1
    ro = pct(d['uns']); mo = {c: np.nanmean(ro[d['cls'] == c]) for c in 'ABCD'}; repro = contr(mo)
    return d, rows_sz, res, pw, repro, scores
def ci(v): v = np.asarray(v, float); return np.nanpercentile(v, [2.5, 97.5]) if np.isfinite(v).sum() > 20 else (np.nan, np.nan)
if __name__ == '__main__':
    out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True); units = sys.argv[2].split(',') if len(sys.argv) > 2 else list(U)
    bs = pd.read_csv(RU + "lowrank_audit/full_20261007/biology_scatter.tsv", sep="\t", usecols=["system", "method", "mode", "gene", "B1_mu"]); bs = bs[(bs.method == "GeoBRIDGE") & (bs["mode"] == "P")]
    BSYS = {'PANC_RFX6': 'PANC', 'BONEMARROW_RUNX1': 'BONEMARROW', 'BM_CHIPATLAS_KLF1': 'BONEMARROW', 'BM_CHIPATLAS_NFE2': 'BONEMARROW', 'CARDIACV_TBX5': 'CARDIACV', 'CARDIACV_NKX2-5': 'CARDIACV'}
    SZ, CT, PW, UD, EX = [], [], [], {}, {}
    for u in units:
        rng = np.random.default_rng(0)
        bsu = bs[bs.system == BSYS[u]] if u in BSYS else None
        d, sz, res, pw, repro, scores = run(u, rng, bsu); SZ += sz; EX[u] = d
        print(u, sz[0], 'unsigned orig resp/occ', np.round(repro, 4), flush=True)
        for (nm, panel), (f0, bb) in res.items():
            UD[(u, nm, panel)] = (f0, bb)
            row = dict(unit=u, grp=d['grp'], level='unit', score=nm, panel=panel)
            for k in f0:
                lo, hi = ci([x[k] for x in bb]); row.update({k: f0[k], k + '_lo': lo, k + '_hi': hi})
            CT.append(row)
        for (nm, lab), (f0, bb) in pw.items():
            lo, hi = ci(bb); PW.append(dict(unit=u, grp=d['grp'], level='unit', score=nm, contrast=lab, auroc=f0, lo=lo, hi=hi))
        for (nm, panel), (f0, bb) in res.items():
            for k in ('A>B', 'C>D', 'A>C', 'B>D'): PW.append(dict(unit=u, grp=d['grp'], level='unit', score=nm + ('' if nm in ('unsigned', 'B1') else ''), contrast=f'{panel}:{k}', auroc=f0[k], lo=ci([x[k] for x in bb])[0], hi=ci([x[k] for x in bb])[1]))
    # group / 3-system aggregation
    GR = {'PANC': ['PANC_RFX6'], 'BM': ['BONEMARROW_RUNX1', 'BM_CHIPATLAS_KLF1', 'BM_CHIPATLAS_NFE2'], 'HEART': ['CARDIACV_TBX5', 'CARDIACV_NKX2-5'], 'ERY': ['GATA1_GATA1', 'KLF1_KLF1', 'KLF1_B_KLF1']}
    KEYS = ['resp', 'occ', 'A>B', 'C>D', 'A>C', 'B>D']
    def agg(items):   # items: list of (f0, bb) ; mean over items (nan-aware)
        f0 = {k: np.nanmean([i[0][k] for i in items]) for k in KEYS}; bb = [{k: np.nanmean([i[1][b][k] for i in items]) for k in KEYS} for b in range(B)]; return f0, bb
    GD = {}
    names = sorted({(n, p) for (_, n, p) in UD})
    for (nm, panel) in names:
        for g, us in GR.items():
            us = [x for x in us if (x, nm, panel) in UD]
            if us: GD[(g, nm, panel)] = agg([UD[(x, nm, panel)] for x in us])
        gs = [g for g in ('PANC', 'BM', 'HEART') if (g, nm, panel) in GD]
        if len(gs) == 3: GD[('3SYS', nm, panel)] = agg([GD[(g, nm, panel)] for g in gs])
    for (g, nm, panel), (f0, bb) in GD.items():
        row = dict(unit=g, grp=g, level='3system' if g == '3SYS' else 'system', score=nm, panel=panel)
        for k in KEYS: lo, hi = ci([x[k] for x in bb]); row.update({k: f0[k], k + '_lo': lo, k + '_hi': hi})
        CT.append(row)
        for k in ('A>B', 'C>D', 'A>C', 'B>D'): PW.append(dict(unit=g, grp=g, level=row['level'], score=nm, contrast=f'{panel}:{k}', auroc=f0[k], lo=row[k + '_lo'], hi=row[k + '_hi']))
    pd.DataFrame(SZ).to_csv(out / 'class_sizes.tsv', sep='\t', index=False); pd.DataFrame(CT).to_csv(out / 'contrasts.tsv', sep='\t', index=False); pd.DataFrame(PW).to_csv(out / 'pairwise.tsv', sep='\t', index=False)
    import pickle; pickle.dump({u: {k: v for k, v in e.items() if k in ('gg', 'cls', 'c6', 'uns', 's0', 'sk')} for u, e in EX.items()}, open(out / 'unit_arrays.pkl', 'wb'))
    pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
    print(pd.DataFrame(SZ).to_string())
    C = pd.DataFrame(CT); print(C[['unit', 'level', 'score', 'panel', 'resp', 'resp_lo', 'resp_hi', 'occ', 'occ_lo', 'occ_hi']].round(3).to_string())
    P = pd.DataFrame(PW); print(P[P.contrast.str.contains('A_')].round(3).to_string())
