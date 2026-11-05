"""E10 ORACLE shared-subspace arm (EVALUATION/DIAGNOSTIC ONLY: O is built from the generator's hidden-driver profiles = simulator truth, never a method input).
Reuses gb_matched_baseline_sergio_2026_11_02 (stored pairs, production_pca30, QR basis Q, CONC=|sum xy|/sum|x||y|, gb_primary.zcons over cols_PI, min over roots)
and the exact hidden-production reconstruction of gb_t13_story_E4E5_2026_11_02.
O (per root) = orthonormal basis of span{dhA, dhB} over that root's pairs, dh_X = h_X[bin[J]] - h_X[bin[I]] (V.ab_hn(topo), sampled_bin from geometry_raw),
only for drivers with amp != 0 (rec['ab']['amp']); C0 -> no O.
Arms: ORACLE = linear injected-profile oracle (lower bound) (project off O, depth 1, zcons [0]); ORACLE_Q08 (project off span(O u Q_k), k=0..8, zcons over 9 depths); GB_k0; GB_k08 (production).
usage: script OUTSUB [--grns ..] [--topos ..] [--conds ..] [--sims ..]"""
import sys, os, json, numpy as np, pandas as pd
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch')]
import gb_tf32rep_fast_2026_10_22 as T, gb_primary as GP
import anndata
from sklearn.metrics import roc_auc_score as AUC, average_precision_score as APs
from scipy import stats
a = sys.argv; OUT = f"{T.R}/results/signed/r2fix_2026_11_01/E10_oracle/rank_audit/{a[1]}/"; os.makedirs(OUT, exist_ok=True)
arg = lambda k, d: a[a.index(k) + 1].split(',') if k in a else d
GRNS = [int(x) for x in arg('--grns', ['501', '502'])]; TOPOS = arg('--topos', ['bifurcation', 'tree'])
CONDS = arg('--conds', ['C0', 'C1', 'C2', 'C3']); SIMS = [int(x) for x in arg('--sims', ['1'])]
V = T.V_(); TAB = pd.read_csv(f'{T.MAIN_ROOT}/TABLE.tsv', sep='\t').query("method=='GB'").set_index(['grn', 'topo', 'cond', 'sim'])
dense = lambda M: np.asarray(M.todense() if hasattr(M, 'todense') else M, float)
def orth(M, tol=1e-10):
    U, s, _ = np.linalg.svd(M, full_matrices=False); return U[:, s > tol * s[0]]
def conc(x, y): return np.abs(x.T @ y) / (np.abs(x).T @ np.abs(y))
def proj(Z, B): return Z - B @ (B.T @ Z) if B is not None and B.shape[1] else Z
def hidden(g, t, c, s, G, rec, Gm):
    """exact E4E5 reconstruction -> Ph (hidden production), plus injection pieces; asserts injection decomposition"""
    amp = rec["ab"]["amp"]; Ul = np.asarray(Gm["conc_unspliced_latent"], float); Ed = Gm["edges"]; cm = np.asarray(Gm["component_edge_mask"], bool); bb = np.asarray(Gm["sampled_bin"], int)
    hA, hB = V.ab_hn(t); cA, cB = V.ab_loadings(g); n = len(Ul); iA, iB = np.zeros((n, 100)), np.zeros((n, 100))
    for m in G["mrs"]: iA[:, m] = amp[0] * cA[m] * hA[bb]; iB[:, m] = amp[1] * cB[m] * hB[bb]
    V.PILOT["mr_inj"] = iA + iB; Ph = V.production_parts(Ul, Ed, component_mask=cm)[1]; V.PILOT["mr_inj"] = None
    Ph0 = V.production_parts(Ul, Ed, component_mask=cm)[1]; assert np.abs(Ph - Ph0 - (iA + iB)).max() < 1e-9
    return Ph, bb, hA, hB, amp, iA, iB, cA, cB
rows, par, info = [], [], []
for g in GRNS:
    G = V.generate_grn(g); DM = T.dist_matrix(sorted((int(x), int(y)) for x, y in G['gt']))
    for t in TOPOS:
        for c in CONDS:
            for s in SIMS:
                dd = T.dsdir(g, t, c, s); rec = json.load(open(f'{dd}/record.json')); assert rec['status'] == 'OK'
                Z = np.load(f'{dd}/raw/geobridge_raw.npz'); Gm = np.load(f'{dd}/raw/geometry_raw.npz')
                src = np.asarray(Z['source_gene_index'], int); cols = np.asarray(Z['cols_PI'], int)
                A = anndata.read_h5ad(T.h5path(g, t, c, s)); Ms, Mu = dense(A.layers['Ms']), dense(A.layers['Mu']); P30 = np.asarray(Gm['production_pca30'], float)
                Ph, bb, hA, hB, amp, iA, iB, cA, cB = hidden(g, t, c, s, G, rec, Gm)
                Sr = {k: {} for k in ('GB0', 'GB8', 'RK', 'RKQ')}
                for r in GP.ROOTS:
                    I, J = np.asarray(Gm[f'pairs_{r}_I'], int), np.asarray(Gm[f'pairs_{r}_J'], int)
                    Q, _ = np.linalg.qr(P30[J] - P30[I]); dX = Ms[J][:, src] - Ms[I][:, src]; dY = Mu[J] - Mu[I]; dH = Ph[J] - Ph[I]
                    # parity: driver_pair_energy_prefix3 (hidden reconstruction)
                    p3 = np.linalg.norm(Q[:, :3].T @ dH, 'fro') ** 2 / np.maximum(np.linalg.norm(dH, 'fro') ** 2, V.EPS)
                    assert abs(p3 - rec[f'geometry_{r}']['driver_pair_energy_prefix3']) < 1e-9, ('prefix3 parity', dd, r)
                    def C(B): return conc(proj(dX, B), proj(dY, B))                   # (nsrc, 100)
                    def withQ(O_, k): return orth(np.column_stack([O_, Q[:, :k]])) if O_ is not None and k else (O_ if O_ is not None else (Q[:, :k] if k else None))
                    # ORACLE_RANK: left singular vectors of dH up to cumulative energy >= 99.9% (numerical rank r*, per root)
                    if np.linalg.norm(dH) ** 2 > V.EPS:
                        Uh, sh, _ = np.linalg.svd(dH, full_matrices=False); cum = np.cumsum(sh ** 2) / np.sum(sh ** 2); rs = int(np.searchsorted(cum, 0.999) + 1); OK = Uh[:, :rs]
                    else: rs, OK = 0, None
                    info.append(dict(grn=g, topo=t, cond=c, sim=s, root=r, rstar=rs, dimO=0 if OK is None else OK.shape[1],
                                     hidden_energy_in_O=float(np.linalg.norm(OK.T @ dH) ** 2 / max(np.linalg.norm(dH) ** 2, V.EPS)) if OK is not None else 0.0,
                                     residual_hidden_energy=float(np.linalg.norm(proj(dH, OK)) ** 2 / max(np.linalg.norm(dH) ** 2, V.EPS)) if OK is not None else np.nan,
                                     pair_energy_dX_in_O=float(np.linalg.norm(OK.T @ dX) ** 2 / np.linalg.norm(dX) ** 2) if OK is not None else 0.0))
                    Sr['GB0'][r] = [C(None)]; Sr['GB8'][r] = [C(Q[:, :k] if k else None) for k in range(9)]
                    Sr['RK'][r] = [C(OK)]; Sr['RKQ'][r] = [C(withQ(OK, k)) for k in range(9)]
                    st = np.asarray(Z[f'TRUE|U|{r}|absS'], float)[:9]; m = np.isfinite(st)
                    for i in range(len(src)): m[:, i, src[i]] = False
                    d_ = np.nanmax(np.abs(np.stack([x for x in Sr['GB8'][r]])[m] - st[m])); par.append(d_); assert d_ < 1e-4, ('CONC parity', dd, r, d_)
                Dm = np.array([[DM[s_, int(x)] for x in cols[i]] for i, s_ in enumerate(src)]); Y = Dm == 1; assert Y.sum() == 240
                def agg(arm, nk): return np.vstack([GP.zcons({r: np.stack([Sr[arm][r][k][i, cols[i]] for k in range(nk)], 1) for r in GP.ROOTS}, list(range(nk))) for i in range(len(src))])
                Vs = {'GB_k0': agg('GB0', 1), 'GB_k08': agg('GB8', 9), 'ORACLE_RANK': agg('RK', 1), 'ORACLE_RANK_Q08': agg('RKQ', 9)}
                ok = np.isfinite(Vs['GB_k08']); tr = TAB.loc[(g, t, c, s)]; assert abs(APs(Y[ok], Vs['GB_k08'][ok]) - tr.AP) < 1e-6, ('TABLE parity', dd)
                for nm, sc in Vs.items():
                    ok = np.isfinite(sc); r_ = dict(topo=t, grn=g, cond=c, sim=s, arm=nm, coverage=ok.mean(), AP=APs(Y[ok], sc[ok]))
                    for lab, neg in (('DvI23', (2, 3)), ('DvI2', (2,))):
                        mm = ok & ((Dm == 1) | np.isin(Dm, neg)); r_[f'{lab}_AUROC'] = AUC(Dm[mm] == 1, sc[mm])
                    rows.append(r_)
                print('done', g, t, c, s, 'amp', amp, 'max parity', max(par), flush=True)
R = pd.DataFrame(rows); R.to_csv(OUT + 'datasets.tsv', sep='\t', index=False); pd.DataFrame(info).to_csv(OUT + 'O_info.tsv', sep='\t', index=False)
json.dump(dict(max_parity=float(max(par)), n=len(R), grns=GRNS, topos=TOPOS, conds=CONDS, sims=SIMS), open(OUT + 'receipt.json', 'w'), indent=1)
print('WROTE', OUT)
