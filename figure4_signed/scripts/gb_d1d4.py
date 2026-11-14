# k=0 full-data S_0 and centred Sc_0 for every observational-universe target, both roots
import sys, os, json, numpy as np, pandas as pd
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch'), (__import__("os").environ["GEOBRIDGE_ROOT"] + '/pairspace')]
import gb_geom_phase1 as G1
ROOTS = G1.ROOTS; EPS = 1e-12
EPS_ = {'KLF1': ('KLF1_KLF1', 'GATA1'), 'GATA1': ('GATA1_GATA1', 'GATA1'), 'TBX5': ('CARDIACV_TBX5', 'CARDIACV'), 'NKX2-5': ('CARDIACV_NKX2-5', 'CARDIACV'), 'RFX6': ('PANC_RFX6', 'PANC')}

def score(system, source, OUT):
    import anndata as ad, gb_real_uniform as RU
    axis = G1.axis_of(system); C = G1.load_cache(system, source); A = ad.read_h5ad(G1.paths(system, 'full')[0]); Z = np.load(G1.paths(system, 'full')[1], allow_pickle=True)
    hv = {}
    for j, g in enumerate(np.char.upper(np.asarray(A.var_names).astype(str))): hv.setdefault(g, j)
    colm = np.array([hv.get(g.upper(), -1) for g in axis]); have = np.flatnonzero(colm >= 0)
    Mu = RU.dense(A.layers['Mu'][:, colm[have]], np.float32); xs = RU.dense(A.layers['Ms'][:, [hv[source.upper()]]], np.float64)[:, 0]
    out = pd.DataFrame(dict(gene=axis)); mx_err = 0.0
    for r in ROOTS:
        I, J = np.asarray(Z[f'{r}_I'], int), np.asarray(Z[f'{r}_J'], int); P = len(I); dx = xs[J] - xs[I]; xc = dx - dx.mean()
        S0 = np.full(len(axis), np.nan); Sc = np.full(len(axis), np.nan); CH = max(20, int(4e7 / P))
        for g0 in range(0, len(have), CH):
            sl = slice(g0, min(g0 + CH, len(have))); dy = Mu[J, sl].astype(np.float64) - Mu[I, sl].astype(np.float64)
            N = dx @ dy; D = np.abs(dx) @ np.abs(dy); yc = dy - dy.mean(0); Nc = xc @ yc; Dc = np.abs(xc) @ np.abs(yc)
            S0[have[sl]] = np.where(D > EPS, N / np.where(D > EPS, D, 1), np.nan); Sc[have[sl]] = np.where(Dc > EPS, Nc / np.where(Dc > EPS, Dc, 1), np.nan)
        u = C[r]['uni'] & (np.char.upper(axis) != source.upper()); S0[~u] = np.nan; Sc[~u] = np.nan
        cs = C[r]['S'][:, 0]; m = np.isfinite(S0) & np.isfinite(cs); mx_err = max(mx_err, float(np.abs(S0[m] - cs[m]).max()))
        out[f'S0_{r}'] = S0; out[f'Sc0_{r}'] = Sc; out[f'uni_{r}'] = u
        print(source, r, 'done', flush=True)
    both = lambda a: np.isfinite(out[f'{a}_dc1_min']) & np.isfinite(out[f'{a}_dc1_max'])
    out['S0_mean'] = np.where(both('S0'), (out.S0_dc1_min + out.S0_dc1_max) / 2, np.nan); out['Sc0_mean'] = np.where(both('Sc0'), (out.Sc0_dc1_min + out.Sc0_dc1_max) / 2, np.nan)
    out['d1'] = np.sign(out.S0_mean); out['d4'] = np.sign(out.Sc0_mean)
    out.to_csv(f'{OUT}/d1_d4_scores_{source}.tsv', sep='\t', index=False); open(f'{OUT}/d1_d4_parity_{source}.txt', 'w').write(f'max|S0 - cache S_0| over universe = {mx_err:.3e}\n'); print('saved; parity', mx_err)

def ba(ok1, cls_up):  # recall per class from correctness vector
    return np.nanmean([ok1[~cls_up].mean(), ok1[cls_up].mean()]) if (~cls_up).any() and cls_up.any() else np.nan

def evaluate(OUT):
    import gb_fourclass_signed_k0 as GF
    RESP = ('A_act', 'A_rep', 'C_act', 'C_rep'); rows = []; rng = np.random.default_rng(0)
    for src, (unit, system) in EPS_.items():
        T = pd.read_csv(f'{OUT}/d1_d4_scores_{src}.tsv', sep='\t', keep_default_na=False, na_values=['']); au = np.char.upper(np.array(T.gene.astype(str).tolist())); aix = {}
        for i, g in enumerate(au): aix.setdefault(g, i)
        d = GF.build(unit); gg, c6 = d['gg'], d['c6']; tfi = aix[src.upper()]
        ridx, cls = [], []
        for g, c in zip(gg, c6):
            a = aix.get(g, -1)
            if c in RESP and a >= 0 and a != tfi: ridx.append(a); cls.append(c)
        ridx = np.array(ridx); cls = np.array(cls); up = np.array([c.endswith('_rep') for c in cls]); n_all = len(ridx)
        d1, d4, s0 = T.d1.to_numpy()[ridx], T.d4.to_numpy()[ridx], T.S0_mean.to_numpy()[ridx]
        fin = np.isfinite(d1) & np.isfinite(d4)
        def sub(msk, tag):
            m = fin & msk; u = up[m]; e = np.where(u, -1.0, 1.0); a, b = d1[m], d4[m]; ok1, ok4 = (a == e), (b == e)
            r = dict(endpoint=src, subset=tag, n_resp_total=n_all, n_analyzed=int(m.sum()), n_KOdown=int((~u).sum()), n_KOup=int(u.sum()),
                     BA_d1=ba(ok1, u), BA_d4=ba(ok4, u), recall_down_d1=ok1[~u].mean() if (~u).any() else np.nan, recall_up_d1=ok1[u].mean() if u.any() else np.nan,
                     recall_down_d4=ok4[~u].mean() if (~u).any() else np.nan, recall_up_d4=ok4[u].mean() if u.any() else np.nan,
                     wrong_to_right=int((~ok1 & ok4).sum()), right_to_wrong=int((ok1 & ~ok4).sum()), n_sign_changed=int((a != b).sum()))
            for nm, mk in (('down', ~u), ('up', u)):
                r[f'fixed_{nm}'] = int((~ok1 & ok4)[mk].sum()); r[f'broken_{nm}'] = int((ok1 & ~ok4)[mk].sum())
            r['net_classbal'] = np.nanmean([(r['fixed_down'] - r['broken_down']) / max((~u).sum(), 1), (r['fixed_up'] - r['broken_up']) / max(u.sum(), 1)]) if (~u).any() and u.any() else np.nan
            if tag == 'all' or True:
                idx = np.flatnonzero(m); dn, upi = idx[~u], idx[u]; e_all = np.where(up, -1.0, 1.0); bs = []
                for _ in range(500):
                    bi = np.concatenate([rng.choice(dn, len(dn)), rng.choice(upi, len(upi))]) if len(dn) and len(upi) else np.array([], int)
                    if len(bi) == 0: bs.append(np.nan); continue
                    uu = up[bi]; ee = e_all[bi]; bs.append(ba(d4[bi] == ee, uu) - ba(d1[bi] == ee, uu))
                r['dBA_boot_lo'], r['dBA_boot_hi'] = np.nanpercentile(bs, [2.5, 97.5]) if np.isfinite(bs).any() else (np.nan, np.nan)
            r['dBA'] = r['BA_d4'] - r['BA_d1']; rows.append(r)
        sub(np.ones(n_all, bool), 'all'); sub(np.isfinite(s0) & (np.abs(np.nan_to_num(s0)) >= .5), '|S0|>=.5')
        print(src, 'responders', n_all, 'finite', int(fin.sum()), 'unsupported', n_all - int(fin.sum()), flush=True)
    pd.DataFrame(rows).to_csv(f'{OUT}/d1_d4_eval.tsv', sep='\t', index=False)
    pd.set_option('display.width', 300, 'display.max_columns', 50, 'display.float_format', '{:.4g}'.format); print(pd.DataFrame(rows).to_string(index=False))

if __name__ == '__main__':
    if sys.argv[1] == 'score': score(sys.argv[2].upper(), sys.argv[3].upper(), sys.argv[4])
    else: evaluate(sys.argv[2])
