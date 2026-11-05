"""Fig 3d panel A (internal directive 2026-10-29 Part I.2): quantify developmental structure in PANC observational data INDEPENDENTLY of the GeoBRIDGE operator.
TRUTH-FREE (no KO/ChIP read). Cell space = the 30 fixture PCs used by GeoBRIDGE (F.obsm['X_pca'][:, :30]); pseudotime = the per-root pseudotime stored with the
production pairs (zp['<root>_pt']); spline basis = nat_spline with the production knots (5..95% quantiles, 9 knots) -> 8 columns, identical to the SPL arm.
 T  Temporal structure: fraction of total 30-PC variance explained by the pseudotime spline, 5-fold cross-validated (out-of-fold R^2 = 1 - SSE_oof / SST, SST about the
    training-fold mean), per root. Also per PC. Null: same CV after permuting pseudotime across cells (20 perms).
 F  Fate structure beyond pseudotime: terminal endocrine cells only (Alpha/Beta/Delta/Epsilon), restricted to COMMON pseudotime support = 10 quantile bins of pt among
    terminal cells keeping bins with >=2 fates each having >=15 cells. Incremental out-of-fold R^2 of [spline + fate one-hot] over [spline], as a fraction of total
    30-PC variance of those cells; per fate (one-vs-rest indicator added to spline). Null: fate labels permuted WITHIN pt bins (20 perms) -> preserves pt-fate coupling.
    Fate classifier: multinomial logistic on spline-residualised PCs (residualised within fold), balanced accuracy, chance = 1/#fates; same within-bin label-permutation null.
 Caveat recorded in output: fate labels and pseudotime are derived from the same transcriptome, so these quantify reproducible STRUCTURE, not independent biology.
usage (server, repo root): script OUT"""
import sys, os, json, hashlib
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); sys.path[:0] = [R + 'scratch', R + 'pairspace']
import numpy as np, pandas as pd, anndata as ad, gb_substrate as GS
from gb_pt_comparator_compute_2026_10_21 import nat_spline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
OUT = sys.argv[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True); NPERM = 20
SYSN = os.environ.get('GB_SYS', 'PANC')   # 2026-10-29 scope review: BM identities pre-specified before any result (internal review); CARDIACV has no cell-type annotation -> temporal only
FATES = {'PANC': ['Alpha', 'Beta', 'Delta', 'Epsilon'], 'BONEMARROW': ['Ery_2', 'Mono_2', 'Mega', 'DCs', 'CLP'], 'CARDIACV': []}[SYSN]; PFX = {'PANC': 'panc'}.get(SYSN, SYSN.lower())
fx, prs, _ = GS.paths(SYSN, canonical=True); F = ad.read_h5ad(fx); P30 = np.asarray(F.obsm['X_pca'], float)[:, :30]; zp = np.load(prs, allow_pickle=True)
lab = F.obs['clusters'].astype(str).to_numpy() if 'clusters' in F.obs else np.full(F.n_obs, 'NA'); n = len(lab)

def folds(m, seed):
    return np.random.default_rng([20261029, 31, seed]).permutation(m) % 5

def oof_r2(Y, Xd, fo):
    """out-of-fold R^2 of Y on design Xd (intercept added), pooled over columns; SST about training-fold means."""
    sse = sst = 0.0; per_sse = np.zeros(Y.shape[1]); per_sst = np.zeros(Y.shape[1])
    for f in range(5):
        tr, te = fo != f, fo == f; A = np.c_[np.ones(tr.sum()), Xd[tr]]; b = np.linalg.lstsq(A, Y[tr], rcond=None)[0]
        e = Y[te] - np.c_[np.ones(te.sum()), Xd[te]] @ b; t = Y[te] - Y[tr].mean(0); per_sse += (e ** 2).sum(0); per_sst += (t ** 2).sum(0)
    return 1 - per_sse.sum() / per_sst.sum(), 1 - per_sse / per_sst

ROWS, PERPC = [], []
for r in ('dc1_min', 'dc1_max'):
    pt = np.asarray(zp[f'{r}_pt'], float); kn = np.quantile(pt, np.linspace(.05, .95, 9)); Bs = nat_spline(pt, kn)[:, 1:]
    assert Bs.shape == (n, 8) and np.isfinite(Bs).all()
    fo = folds(n, 0); r2, per = oof_r2(P30, Bs, fo)
    null = [oof_r2(P30, nat_spline(p_, kn)[:, 1:], fo)[0] for p_ in (np.random.default_rng([20261029, 32, s]).permutation(pt) for s in range(NPERM))]
    ROWS.append(dict(root=r, block='T_temporal', quantity='CV R2 of 30-PC variance on pseudotime spline (all cells)', n_cells=n, value=r2, null_mean=np.mean(null), null_max=np.max(null)))
    PERPC += [dict(root=r, pc=i + 1, var_share=P30[:, i].var() / P30.var(0).sum(), cv_r2=per[i]) for i in range(30)]
    if not FATES: print(r, 'done (temporal only)', flush=True); continue
    # ---- F: terminal fate beyond pseudotime, common support
    isf = np.isin(lab, FATES); ptf = pt[isf]; edges = np.quantile(ptf, np.linspace(0, 1, 11)); b = np.clip(np.searchsorted(edges, pt, 'right') - 1, 0, 9)
    keep = np.zeros(n, bool); supp = []
    for k in range(10):
        m = isf & (b == k); cnt = {f: int((lab[m] == f).sum()) for f in FATES}; ok = [f for f in FATES if cnt[f] >= 15]
        if len(ok) >= 2: keep |= m & np.isin(lab, ok)
        supp.append(dict(root=r, bin=k, pt_lo=edges[k], pt_hi=edges[k + 1], **cnt, kept=len(ok) >= 2))
    pd.DataFrame(supp).to_csv(OUT + f'fate_support_bins_{r}.tsv', sep='\t', index=False)
    idx = np.where(keep)[0]; Y = P30[idx]; S = Bs[idx]; L = lab[idx]; bb = b[idx]; fs = [f for f in FATES if (L == f).any()]
    Hot = np.stack([(L == f).astype(float) for f in fs[1:]], 1); fo = folds(len(idx), 1)
    def incr(Lab):
        H = np.stack([(Lab == f).astype(float) for f in fs[1:]], 1); return oof_r2(Y, np.c_[S, H], fo)[0] - oof_r2(Y, S, fo)[0]
    def within_bin_perm(seed):
        g = np.random.default_rng([20261029, 33, seed]); Lp = L.copy()
        for k in np.unique(bb): w = np.where(bb == k)[0]; Lp[w] = L[w][g.permutation(len(w))]
        return Lp
    perms = [within_bin_perm(s) for s in range(NPERM)]
    base = oof_r2(Y, S, fo)[0]; inc = incr(L); ninc = [incr(Lp) for Lp in perms]
    ROWS.append(dict(root=r, block='F_fate', quantity='CV R2 of 30-PC variance on spline (terminal cells, common support)', n_cells=len(idx), value=base, null_mean=np.nan, null_max=np.nan))
    ROWS.append(dict(root=r, block='F_fate', quantity='incremental CV R2 of fate one-hot beyond spline', n_cells=len(idx), value=inc, null_mean=np.mean(ninc), null_max=np.max(ninc)))
    for f in fs:
        def inc1(Lab): return oof_r2(Y, np.c_[S, (Lab == f).astype(float)], fo)[0] - base
        nf = [inc1(Lp) for Lp in perms]
        ROWS.append(dict(root=r, block='F_fate', quantity=f'incremental CV R2 of 1{{{f}}} beyond spline', n_cells=int((L == f).sum()), value=inc1(L), null_mean=np.mean(nf), null_max=np.max(nf)))
    def bacc(Lab):
        pred = np.empty(len(idx), object)
        for f_ in range(5):
            tr, te = fo != f_, fo == f_; A = np.c_[np.ones(tr.sum()), S[tr]]; bcoef = np.linalg.lstsq(A, Y[tr], rcond=None)[0]
            Rtr = Y[tr] - A @ bcoef; Rte = Y[te] - np.c_[np.ones(te.sum()), S[te]] @ bcoef
            pred[te] = LogisticRegression(max_iter=2000, C=1.0).fit(Rtr, Lab[tr]).predict(Rte)
        return balanced_accuracy_score(Lab, pred)
    ba = bacc(L); nba = [bacc(Lp) for Lp in perms[:10]]
    ROWS.append(dict(root=r, block='F_fate', quantity=f'fate balanced accuracy from spline-residualised PCs (chance {1/len(fs):.2f})', n_cells=len(idx), value=ba, null_mean=np.mean(nba), null_max=np.max(nba)))
    print(r, 'done', flush=True)
D = pd.DataFrame(ROWS); D.to_csv(OUT + f'{PFX}_devstructure.tsv', sep='\t', index=False); pd.DataFrame(PERPC).to_csv(OUT + f'{PFX}_temporal_perPC.tsv', sep='\t', index=False)
json.dump(dict(code_sha256=hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest(), fixture=fx, pairs=prs, nperm=NPERM,
               caveat='pseudotime and fate labels are derived from the same transcriptome; measures quantify reproducible structure, not independent biology'), open(OUT + 'receipt.json', 'w'), indent=1)
pd.set_option('display.width', 250); pd.set_option('display.max_colwidth', 90); print(D.round(4).to_string(index=False))
