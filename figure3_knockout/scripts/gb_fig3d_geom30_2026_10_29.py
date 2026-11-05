"""Fig 3d geometry, extended to k=30 and pooled over the two pseudotime roots (PI 2026-10-29). TRUTH-FREE, cached inputs only; same Q_k, pairs, spline basis
and fate signals as scratch/gb_fig3d_branch_geom_v2_2026_10_29.py (no re-mask after the spline residual).
For each root r and signal Y_r: num_r(k) = ||Q_k,r' Y_r||^2, den_r = ||Y_r||^2. Per-root capture = num/den; ROOT-POOLED capture = (num_min + num_max) / (den_min + den_max),
a fixed a-priori aggregation (PRIMARY uses both roots; internal review 2026-10-29), never a root selection.
Signals: tau = pseudotime difference; Pi = 30-PC pair differences (generic reference); fate_all|spline = 4-fate one-hot differences among terminal-terminal pairs,
spline-residualised; fate_<F>|spline = one-vs-rest indicator of fate F among terminal-terminal pairs, spline-residualised. k = 1..30 (k=30 = full 30-PC span).
Also: cluster composition of the first 2% of cells along each root's pseudotime (where each root starts).
usage (server, repo root): script OUT"""
import sys, os
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); sys.path[:0] = [R + 'scratch', R + 'pairspace']
import numpy as np, pandas as pd, anndata as ad, gb_substrate as GS
from gb_pt_comparator_compute_2026_10_21 import nat_spline
OUT = sys.argv[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
SYSN = os.environ.get('GB_SYS', 'PANC')   # 2026-10-29 scope review: BM identities pre-specified (internal review); CARDIACV: pseudotime + Pi only
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
