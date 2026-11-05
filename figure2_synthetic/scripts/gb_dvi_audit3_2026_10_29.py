"""Final mechanistic DvI audit (internal directive 2026-10-29, 'Final Mechanistic Audit'). CACHED-ONLY EVALUATION: no new simulation, no refit, production unchanged.
Per TF32REP dataset (32 regulators x 99 non-self targets, Fig 2 universe) writes ONE long per-pair table with
  * truth annotations (truth-side, never a method input): shortest directed path d(s->t) and d(t->s), reverse edge t->s, strict 2-hop mediators,
    signed path composition, mediator out-degree, linearised operating-point pathway gain (generator Hill derivatives on the latent unspliced
    state; steady state dU_t/dU_s ~ prod(sens_e / beta), beta = SERGIO decay 0.8), structural |K| path product, mediator/target expression;
  * scores of every cached Fig 2 method (GB production = frozen zcons k=0..8 root-min; GB K0; GB spliced-target analog; Velorama; FastTENET;
    GENIE3; GRNBoost2; PIDC; RegDiffusion; DeepSEM; Pearson |r|) on the identical 32x99 support;
  * GB residual diagnostics from the round-2 cached numerator/denominator arrays (U and X target channels).
Integrity asserts: GB production (round-2 arrays) == frozen gb_primary recomputed from geobridge_raw; per-depth |num_k/den_k| == stored absS (k<=8);
recomputed zcons == stored production; pooled Fig 2 DvI2/DvI3 of every method == TABLE.tsv; method src/cols == GB roster.
usage (server, repo root): script OUT [--only g:topo:cond:sim,...]"""
import sys, os, json, glob, hashlib
import numpy as np, pandas as pd
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch'), (__import__("os").environ["GEOBRIDGE_ROOT"] + '/pairspace')]
import gb_tf32rep_fast_2026_10_22 as TF, gb_b4_score as B, gb_pilot_AB_tf32rep_2026_10_22 as GEN
from sklearn.metrics import roc_auc_score as AUC
import anndata
OUT = sys.argv[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
ONLY = sys.argv[sys.argv.index('--only') + 1].split(',') if '--only' in sys.argv else None
R2 = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/results/signed/dvi_mech2_2026_10_29/full')
AF = {os.path.basename(f): f for f in glob.glob(f'{R2}/**/arrays/*.npz', recursive=True)}
TAB = pd.read_csv(f'{TF.MAIN_ROOT}/TABLE.tsv', sep='\t').set_index(['grn', 'topo', 'cond', 'sim', 'method'])
ROOTS = B.GP.ROOTS; BETA = float(GEN.SERGIO_PARAMS['decays']); assert BETA == 0.8
EXT = ('fasttenet', 'genie3', 'grnboost2', 'pidc', 'regdiffusion', 'deepsem', 'pearson')
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
dn = lambda M: M.toarray() if hasattr(M, 'toarray') else np.asarray(M)
def zc_from(A):   # A: root -> (KE, 32, 100) |S|; frozen zcons over k = 0..8 (columns 0..8 of KE are k = 0..8), min over roots
    return B.zc([{r: A[r][:9, i, cols[i]].T for r in ROOTS} for i in range(32)], range(9))
DONE, REC = [], []
for g in TF.GRNS:
    G = GEN.generate_grn(g); gt = sorted((int(a), int(b)) for a, b in G['gt']); E = set(gt); DM = TF.dist_matrix(gt); Kd = {(int(a), int(b)): float(v) for (a, b), v in G['K'].items()}
    ch_ = {}; [ch_.setdefault(a, []).append(b) for a, b in gt]; par_ = {}; [par_.setdefault(b, []).append(a) for a, b in gt]
    outdeg = {v: len(ch_.get(v, [])) for v in range(100)}; lay = G['layer']
    Adj = np.zeros((100, 100)); [Adj.__setitem__((a, b), 1) for a, b in gt]; A3 = Adj @ Adj @ Adj   # number of directed length-3 walks (DAG: paths)
    A2m = np.zeros((100, 100))                                                                     # for direct edges: parallel 2-paths
    for a, b in gt: A2m[a, b] = sum(1 for m_ in ch_.get(a, []) if (m_, b) in E)
    for t in TF.TOPOS:
        pre = json.load(open(f'{TF.gdir(g, t)}/roster_pre.json'))
        for c in TF.COND:
            for s in TF.SIMS:
                key = f'{g}:{t}:{c}:{s}'
                if ONLY and key not in ONLY: continue
                dd = TF.dsdir(g, t, c, s); tag = os.path.basename(dd); Zr = np.load(f'{dd}/raw/geobridge_raw.npz')
                A2 = np.load(AF[f'g{g}_{t}_{c}_s{s}.npz']); src = np.asarray(A2['src'], int); cols = np.asarray(A2['cols'], int)
                assert list(src) == pre['cands'] and np.array_equal(cols, np.array([[x for x in range(100) if x != q] for q in src])), 'roster/cols'
                assert np.array_equal(src, np.asarray(Zr['source_gene_index'], int)) and np.array_equal(cols, np.asarray(Zr['cols_PI'], int)), 'raw roster'
                # ---- scores (identical 32x99 support); GB production parity with the frozen Fig 2 primary
                SC = {'GB': np.asarray(A2['z|U|PROD'], float), 'GB_K0': np.asarray(A2['z|U|k0'], float), 'GBX': np.asarray(A2['z|X|PROD'], float), 'GBX_K0': np.asarray(A2['z|X|k0'], float)}
                gp = TF.gb_primary(dd); assert np.allclose(gp, SC['GB'], atol=1e-6, rtol=0), ('GB primary parity', np.nanmax(np.abs(gp - SC['GB'])))   # round-2 float32 recompute ~1e-7
                SC['GB'] = gp                                                                                                              # exact frozen Fig 2 primary
                vf = f'{TF.gdir(g, t)}/velorama_rt0/{c}/{tag}.npz'; z = np.load(vf); assert np.array_equal(z['src'], src) and np.array_equal(z['cols'], cols); SC['VEL'] = np.asarray(z['score'], float)
                for m in EXT:
                    z = np.load(TF.score_path(m, ('rep', g, t, c, s))); assert np.array_equal(z['src'], src) and np.array_equal(z['cols'], cols) and str(z['cols_sha']) == pre['cols_sha']; SC[m] = np.asarray(z['score'], float)
                for m, S in SC.items(): assert S.shape == (32, 99) and np.isfinite(S).all(), ('nonfinite', m)
                Dm = np.array([[DM[s_, int(x)] for x in cols[i]] for i, s_ in enumerate(src)]); assert np.array_equal(Dm, np.asarray(A2['Dm'])), 'Dm vs round-2'
                for m in ['GB', 'VEL'] + list(EXT):   # Fig 2 pooled parity
                    for k in (2, 3):
                        msk = (Dm == 1) | (Dm == k); v = AUC(Dm[msk] == 1, SC[m][msk]); ref = TAB.loc[(g, t, c, s, m), f'DvI{k}']; assert abs(v - ref) < 1e-12, ('TABLE parity', m, k, v, ref)
                # ---- GB residual diagnostics from cached num/den (round 2), with identity checks
                KE = list(A2['KE']); assert KE[:9] == list(range(9)) and KE[-1] == 30
                NUM = {ch: {r: np.asarray(A2[f'num|{ch}|{r}'], float) for r in ROOTS} for ch in ('U', 'X')}; DEN = {ch: {r: np.asarray(A2[f'den|{ch}|{r}'], float) for r in ROOTS} for ch in ('U', 'X')}
                idmax = 0.0
                for r in ROOTS:
                    st = np.asarray(Zr[f'TRUE|U|{r}|absS'], float)[:9]
                    with np.errstate(invalid='ignore', divide='ignore'): mine = np.abs(NUM['U'][r][:9] / DEN['U'][r][:9])
                    for i, q in enumerate(src): mine[:, i, q] = np.nan
                    assert np.array_equal(np.isnan(mine), np.isnan(st)); idmax = max(idmax, float(np.nanmax(np.abs(mine - st))))
                assert idmax < 1e-5, ('absS identity', idmax)
                with np.errstate(invalid='ignore', divide='ignore'):
                    AF_ = {r: np.abs(NUM['U'][r] / DEN['U'][r]) for r in ROOTS}; AN_ = {r: np.abs(NUM['U'][r] / DEN['U'][r][[0]]) for r in ROOTS}; AD_ = {r: np.abs(NUM['U'][r][[0]] / DEN['U'][r]) for r in ROOTS}
                for r in ROOTS:
                    for A_ in (AF_, AN_, AD_):
                        for i, q in enumerate(src): A_[r][:, i, q] = np.nan
                zF = zc_from(AF_); assert np.allclose(zF, SC['GB'], atol=1e-5, rtol=0), ('zcons recompute', np.nanmax(np.abs(zF - SC['GB'])))
                SC['GB_NUMONLY'] = zc_from(AN_); SC['GB_DENONLY'] = zc_from(AD_)
                sel = lambda M: np.take_along_axis(M, cols, 1)
                # NATIVE (pre-normalisation) GB scores (internal review #1): root-min of the depth-mean |S_k| (k = 0..8) and of |S_0|, no within-source z.
                # For the spliced channel X these are exactly symmetric in (r, t) up to floating point -> the genuine symmetric orientation control.
                for ch, nm in (('U', 'GBraw'), ('X', 'GBXraw')):
                    with np.errstate(invalid='ignore', divide='ignore'): ab = {r: np.abs(NUM[ch][r][:9] / DEN[ch][r][:9]) for r in ROOTS}
                    SC[nm] = np.minimum(*[sel(ab[r].mean(0)) for r in ROOTS]); SC[nm + '_K0'] = np.minimum(*[sel(ab[r][0]) for r in ROOTS])
                Mx = np.full((32, 100), np.nan); Mx[np.arange(32)[:, None], cols] = SC['GBXraw']; Mx = Mx[:, src]
                xasym = float(np.nanmax(np.abs(Mx - Mx.T) / np.nanmax(np.abs(Mx)))); assert xasym < 1e-5, ('X native not symmetric', xasym)
                diag = {}
                for ch in ('U', 'X'):
                    nk = {r: np.stack([sel(NUM[ch][r][k]) for k in range(9)]) for r in ROOTS}; dk = {r: np.stack([sel(DEN[ch][r][k]) for k in range(9)]) for r in ROOTS}   # (9, 32, 99)
                    sg = np.concatenate([np.sign(nk[r][1:]) for r in ROOTS])           # residual signs k = 1..8, both roots
                    diag[f'{ch}_sgncons'] = np.abs(sg.mean(0))                           # 1 = same residual sign at every depth and root
                    diag[f'{ch}_rootagree8'] = (np.sign(nk[ROOTS[0]][8]) == np.sign(nk[ROOTS[1]][8])).astype(float)
                    diag[f'{ch}_S0'] = np.mean([np.abs(nk[r][0] / dk[r][0]) for r in ROOTS], 0); diag[f'{ch}_S8'] = np.mean([np.abs(nk[r][8] / dk[r][8]) for r in ROOTS], 0)
                    diag[f'{ch}_dlognum8'] = np.mean([np.log(np.abs(nk[r][8])) - np.log(np.abs(nk[r][0])) for r in ROOTS], 0)
                    diag[f'{ch}_dlogden8'] = np.mean([np.log(dk[r][8]) - np.log(dk[r][0]) for r in ROOTS], 0)
                    diag[f'{ch}_logden0'] = np.mean([np.log(dk[r][0]) for r in ROOTS], 0)
                    diag[f'{ch}_sgn0'] = np.sign(sum(nk[r][0] for r in ROOTS)); diag[f'{ch}_sgn8'] = np.sign(sum(nk[r][8] for r in ROOTS))   # root-summed numerator sign
                    # depth/root stability of the per-row z (z within source over the 99 targets, per k and root)
                    zz = {r: np.stack([(lambda a: (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True))(np.abs(nk[r][k] / dk[r][k])) for k in range(9)]) for r in ROOTS}
                    diag[f'{ch}_zsd_depth'] = np.mean([zz[r].std(0) for r in ROOTS], 0); diag[f'{ch}_zdiff_root'] = np.abs(zz[ROOTS[0]].mean(0) - zz[ROOTS[1]].mean(0))
                # ---- truth-side pathway annotations (generator only; never enter any score)
                Gz = np.load(f'{dd}/raw/geometry_raw.npz'); Ul = np.asarray(Gz['conc_unspliced_latent'], float); ed = Gz['edges']; erow = {(int(x['target']), int(x['regulator'])): x for x in ed}
                sens = {}
                for (a_, b_) in gt:
                    x_ = erow[(b_, a_)]; K_ = float(x_['K']); assert abs(K_ - Kd[(a_, b_)]) < 1e-9, 'edge K'
                    u = Ul[:, a_]; h, cp = float(x_['half_response']), float(x_['coop'])
                    with np.errstate(divide='ignore', invalid='ignore'): v = abs(K_) * cp * np.where(u > 0, u, 0) ** (cp - 1) * h ** cp / (h ** cp + u ** cp) ** 2
                    sens[(a_, b_)] = np.nan_to_num(v)
                # realised truth-side strength (internal review): latent unspliced association and the share of the target's GRN-production variance carried by the path
                cm = np.asarray(Gz['component_edge_mask'], bool); Pg = GEN.production_parts(Ul, ed, component_mask=cm)[0]; vPg = Pg.var(0)
                prodv = lambda m_, x: abs(float(erow[(x, m_)]['K'])) * GEN.hill_sergio(Ul[:, m_], erow[(x, m_)]['half_response'], erow[(x, m_)]['coop'], float(erow[(x, m_)]['K']) < 0)
                Uz = (Ul - Ul.mean(0)) / np.where(Ul.std(0) > 0, Ul.std(0), np.nan); RA_ = np.abs(Uz.T @ Uz / len(Ul))
                a = anndata.read_h5ad(f'{dd}/{tag}.h5ad'); Ms = dn(a.layers['Ms']).astype(np.float64); Xc = dn(a.X); det = (Xc > 0).mean(0); msm = Ms.mean(0)
                rows = []
                for i, s_ in enumerate(src):
                    for j, x in enumerate(cols[i]):
                        x = int(x); d = int(Dm[i, j]); meds = [m_ for m_ in ch_.get(s_, []) if (m_, x) in E] if d == 2 else []
                        r_ = dict(src=int(s_), tgt=x, d=d, drev=int(DM[x, s_]), rev=int((x, s_) in E), tgt_is_reg=int(x in set(src)), lay_s=lay[s_], lay_t=lay[x],
                                  tgt_det=float(det[x]), tgt_mean=float(msm[x]), rassoc=float(RA_[s_, x]))
                        if d == 1: r_.update(gain_lin=float(sens[(s_, x)].mean() / BETA), gain1=float(sens[(s_, x)].mean()), gain_K=abs(Kd[(s_, x)]), sign_net=float(np.sign(Kd[(s_, x)])),
                                             mshare=float(prodv(s_, x).var() / vPg[x]) if vPg[x] > 0 else np.nan)
                        if d == 1: r_.update(d1alt=int(A3[s_, x] > 0 or (A2m[s_, x] > 0)))   # direct edge that also has a longer parallel path
                        if d == 2:
                            pg = [float((sens[(s_, m_)] * sens[(m_, x)]).mean() / BETA ** 2) for m_ in meds]; sg_ = [np.sign(Kd[(s_, m_)]) * np.sign(Kd[(m_, x)]) for m_ in meds]
                            gnet = sum(q * sens[(s_, m_)] * sens[(m_, x)] for q, m_ in zip(sg_, meds)) / BETA ** 2   # per-cell signed net two-hop gain (internal review #3)
                            r_.update(gain_net_absmean=float(abs(gnet.mean())), gain_net_meanabs=float(np.abs(gnet).mean()), n_paths3=int(A3[s_, x]))
                            r_.update(gain1=float(sum((sens[(s_, m_)] * sens[(m_, x)]).mean() for m_ in meds)), mshare=float(sum(prodv(m_, x) for m_ in meds).var() / vPg[x]) if vPg[x] > 0 else np.nan)
                            r_.update(n_med=len(meds), gain_lin=float(sum(pg)), gain_lin_signed=float(sum(p * q for p, q in zip(pg, sg_))), gain_K=float(sum(abs(Kd[(s_, m_)] * Kd[(m_, x)]) for m_ in meds)),
                                      sign_net=float(np.sign(sum(p * q for p, q in zip(pg, sg_)))), sign_mixed=int(len(set(sg_)) > 1), comp=';'.join(sorted({('A' if Kd[(s_, m_)] > 0 else 'R') + ('A' if Kd[(m_, x)] > 0 else 'R') for m_ in meds})),
                                      med_outdeg_max=max(outdeg[m_] for m_ in meds), med_det_min=float(min(det[m_] for m_ in meds)), med_mean_max=float(max(msm[m_] for m_ in meds)))
                        for m, S in SC.items(): r_[f's|{m}'] = float(S[i, j])
                        for k_, M_ in diag.items(): r_[k_] = float(M_[i, j])
                        rows.append(r_)
                T = pd.DataFrame(rows); T.insert(0, 'sim', s); T.insert(0, 'cond', c); T.insert(0, 'topo', t); T.insert(0, 'grn', g)
                assert len(T) == 3168 and int((T.d == 1).sum()) == 240 and int(((T.rev == 1) & (T.d < 99)).sum()) == 0 and int(((T.d == 2) & (T.drev < 99)).sum()) == 0, 'class sanity'
                T.to_csv(f'{OUT}/pairs_g{g}_{t}_{c}_s{s}.tsv.gz', sep='\t', index=False)
                REC.append(dict(dataset=key, absS_identity_max=idmax, x_native_asym=xasym,gb_r2_vs_frozen_max=float(np.nanmax(np.abs(np.asarray(A2['z|U|PROD'], float) - gp))), n_d2=int((T.d == 2).sum()), n_rev=int(T.rev.sum()),
                                h5ad_sha=sha(f'{dd}/{tag}.h5ad'), arrays_sha=sha(AF[f'g{g}_{t}_{c}_s{s}.npz']), raw_sha=sha(f'{dd}/raw/geobridge_raw.npz'), geometry_raw_sha=sha(f'{dd}/raw/geometry_raw.npz'),
                                record_sha=sha(f'{dd}/record.json'), velorama_sha=sha(vf), score_shas={m: sha(TF.score_path(m, ('rep', g, t, c, s))) for m in EXT}))
                DONE.append(key); print(key, 'ok', len(T), flush=True)
EXP = [f'{g}:{t}:{c}:{s}' for g in TF.GRNS for t in TF.TOPOS for c in TF.COND for s in TF.SIMS]
assert sorted(DONE) == sorted(ONLY if ONLY else EXP), 'grid'
json.dump(dict(script_sha=sha(os.path.abspath(__file__)), deps={m.__name__: sha(m.__file__) for m in (TF, B, B.GP, GEN)}, only=ONLY, datasets=REC), open(f'{OUT}/receipt_{hashlib.md5(",".join(sorted(DONE)).encode()).hexdigest()[:8]}.json', 'w'), indent=1)
