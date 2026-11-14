# Geometry profiles over k=0..30, pooled across the two pseudotime roots.
import sys, os
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); sys.path[:0] = [R + 'scratch', R + 'pairspace']
import numpy as np, pandas as pd, anndata as ad, gb_substrate as GS
from gb_pt_controls import nat_spline
OUT = sys.argv[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
SYSN = os.environ.get('GB_SYS', 'PANC')
FATES = {'PANC': ['Alpha', 'Beta', 'Delta', 'Epsilon'], 'BONEMARROW': ['Ery_2', 'Mono_2', 'Mega', 'DCs', 'CLP'], 'CARDIACV': []}[SYSN]
qr = lambda A: np.linalg.qr(A)[0]
fx, prs, _ = GS.paths(SYSN, canonical=True); S = GS.load(SYSN, canonical=True); F = ad.read_h5ad(fx)
P30 = np.asarray(F.obsm['X_pca'], float)[:, :30]; zp = np.load(prs, allow_pickle=True); lab = F.obs['clusters'].astype(str).to_numpy() if 'clusters' in F.obs else np.full(F.n_obs, 'NA'); isf = np.isin(lab, FATES)
ROWS, START = [], []
for r in ('dc1_min', 'dc1_max'):
    I, J = (np.asarray(x, int) for x in S['pairs'][r]); pt = np.asarray(zp[f'{r}_pt'], float); dP = P30[J] - P30[I]; Q = qr(dP)
    assert Q.shape[1] == 30
    kn = np.quantile(pt, np.linspace(.05, .95, 9)); Bs = nat_spline(pt, kn)[:, 1:]; Qs = qr(Bs[J] - Bs[I]); sub = isf[I] & isf[J]
    res = lambda Y: Y - Qs @ (Qs.T @ Y)
    sig = {'tau': (pt[J] - pt[I])[:, None], 'Pi': dP}
    if FATES: L = np.stack([(lab == f).astype(float) for f in FATES], 1); sig['fate_all|spline'] = res((L[J] - L[I]) * sub[:, None])
    for f in FATES: sig[f'fate_{f}|spline'] = res((((lab == f)[J].astype(float) - (lab == f)[I]) * sub)[:, None])
    for nm, Y in sig.items():
        den = float((Y ** 2).sum()); C = (Q.T @ Y) ** 2; cum = np.cumsum(C.sum(1))
        for k in range(1, 31): ROWS.append(dict(root=r, signal=nm, k=k, num=float(cum[k - 1]), den=den))
    o = np.argsort(pt)[: max(1, int(.02 * len(pt)))]; START.append(dict(root=r, **pd.Series(lab[o]).value_counts().to_dict()))
    print(r, 'done', flush=True)
D = pd.DataFrame(ROWS); D['captured'] = D.num / D.den; D.to_csv(OUT + 'geom30_per_root.tsv', sep='\t', index=False)
Pd = D.groupby(['signal', 'k'])[['num', 'den']].sum().reset_index(); Pd['captured'] = Pd.num / Pd.den; Pd.to_csv(OUT + 'geom30_pooled.tsv', sep='\t', index=False)
pd.DataFrame(START).fillna(0).to_csv(OUT + 'root_start_composition.tsv', sep='\t', index=False)
pd.set_option('display.width', 250); print(Pd.pivot(index='signal', columns='k', values='captured')[[1, 2, 3, 4, 6, 8, 12, 16, 20, 30]].round(3).to_string()); print(pd.DataFrame(START).fillna(0).to_string(index=False))
