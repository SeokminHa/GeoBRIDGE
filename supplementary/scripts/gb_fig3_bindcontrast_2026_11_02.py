"""Fig 3 (KO x binding) within-response binding contrasts, external-reviewer requests (2026-11-02). EVALUATION ONLY on frozen per-gene tables
results/signed/fig4_bind_prod_k0_2026_10_29/v2_full_B2000/genes_<unit>.tsv (levels L1 = KO-down & bound, L2 = KO-down & no detected binding,
L4 = KO-up & no detected binding, L5 = KO-up & bound, L3 = non-responder). Contrasts: KO-down L1 vs L2, KO-up L5 vs L4.
Per unit x direction and pooled (equal-weight mean over the 6 contrasts; module bootstrap independent per unit, B draws, draw 0 = observed):
 (1) unsigned AUROC (bound vs unbound) for GeoBRIDGE F, k0 F_k0, |extmag| of 9 comparators, and the label-side covariates |log2FC|, abundance;
 (2) sign agreement with KO direction (s0 = sign iso_C; KO-down expects +, KO-up expects -): bound minus unbound;
 (3) signed contrast dT for T_prod, T_k0 and sign-only s0 (does magnitude add to sign?);
 (4) |log2FC| x abundance tertile-stratified unsigned AUROC (weighted by n_pos*n_neg per stratum), point estimate.
wstats verbatim from scratch/gb_fig4_bind_prod_k0_2026_10_29.py:22-31 (parity: AUROC_unsignedF_prod and Tcontrast_prod vs fig4_bind_prod_k0.tsv).
usage: script OUT [--B 2000]"""
import sys, os, json, hashlib, numpy as np, pandas as pd
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); RUN = R + 'results/signed/fig4_bind_prod_k0_2026_10_29/v2_full_B2000/'
a = sys.argv; OUT = a[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True); B = int(a[a.index('--B') + 1]) if '--B' in a else 2000
UNITS = [('CARDIACV_TBX5', 'TBX5'), ('CARDIACV_NKX2-5', 'NKX2-5'), ('PANC_RFX6', 'RFX6')]
SIGNC = ['iso_C', 'iso_PP', 'fwd_C0', 'fwdPP', 'cellMsMu', 'rew_C']   # iso_C = adopted
DIRS = [('KO-down', 'L1', 'L2', 1.), ('KO-up', 'L5', 'L4', -1.)]
COMP = ['|Pearson|', 'GENIE3', 'GRNBoost2', 'RegVelo |J|', 'Velorama', 'RegDiffusion', 'DeepSEM', 'PIDC', 'FastTENET']
def wstats(score, pos, neg, W):
    idx = np.flatnonzero(pos | neg); sc = score[idx]; o = np.argsort(-sc, kind='stable'); sc = sc[o]; idx = idx[o]
    st = np.r_[0, np.flatnonzero(np.diff(sc) != 0) + 1]
    wp = W[:, idx] * pos[idx]; wn = W[:, idx] * neg[idx]
    bp = np.add.reduceat(wp, st, axis=1); bn = np.add.reduceat(wn, st, axis=1)
    P = bp.sum(1); N = bn.sum(1); tp = np.cumsum(bp, 1); fp = np.cumsum(bn, 1)
    with np.errstate(invalid='ignore', divide='ignore'):
        prec = np.where(tp + fp > 0, tp / np.where(tp + fp > 0, tp + fp, 1.0), 0.0); ap = (bp * prec).sum(1) / P; auc = (bp * ((N[:, None] - fp) + 0.5 * bn)).sum(1) / (P * N); prev = P / (P + N)
    return dict(auroc=auc, ap=ap, lift=ap / prev)
def wmean(x, W):
    with np.errstate(invalid='ignore', divide='ignore'): return (W * x).sum(1) / W.sum(1)
REF = pd.read_csv(RUN + 'fig4_bind_prod_k0.tsv', sep='\t'); DRAWS = {}; STRAT = []
for u, tf in UNITS:
    D = pd.read_csv(RUN + f'genes_{u}.tsv', sep='\t'); n = len(D); assert D.module.notna().all()
    mod = D.module.to_numpy()
    if tf == 'NKX2-5':   # cardiac synchronisation (internal review R1): map NKX2-5 modules onto TBX5 module ids by maximal gene overlap, then reuse the TBX5 draws
        hm = HEART_MOD.reindex(D.gene).to_numpy(); ok_ = pd.notna(hm)
        mp = pd.crosstab(mod[ok_], hm[ok_].astype(int)).idxmax(axis=1); inv = np.searchsorted(HEART_UM, pd.Series(mod).map(mp).to_numpy())
        print('NKX2-5 module map: genes shared with TBX5', int(ok_.sum()), '/', n, '; NKX modules mapped', len(mp), '/', len(np.unique(mod)), flush=True)
        um_n = np.unique(mod); extra = [m_ for m_ in um_n if m_ not in mp.index]   # modules with no gene shared with TBX5: independent draws (Poisson-size multiplicities)
        cnt = HEART_CNT[:, np.clip(inv, 0, len(HEART_UM) - 1)].copy(); rng = np.random.default_rng([20261102, 2])
        for m_ in extra: cnt[:, mod == m_] = rng.poisson(1., (B, 1))
        print('NKX2-5 unmapped modules (independent draws):', extra, 'genes', int(np.isin(mod, extra).sum()), flush=True)
        W = np.vstack([np.ones(n), cnt])
    else:
        um, inv = np.unique(mod, return_inverse=True); rng = np.random.default_rng([20261102, {'TBX5': 1, 'RFX6': 3}[tf]])   # fixed per-system seed
        cnt = np.vstack([np.bincount(rng.integers(0, len(um), len(um)), minlength=len(um)) for _ in range(B)]).astype(float); W = np.vstack([np.ones(n), cnt[:, inv]])
        if tf == 'TBX5': HEART_MOD, HEART_UM, HEART_CNT = D.set_index('gene').module, um, cnt
    sc = {'GeoBRIDGE': D.F.to_numpy(float), 'GeoBRIDGE k=0': D.F_k0.to_numpy(float), **{c: np.abs(D[f'extmag|{c}|d4'].to_numpy(float)) for c in COMP},
          'label: |log2FC|': np.abs(D.log2fc.to_numpy(float)), 'label: abundance': D.abundance.to_numpy(float)}
    s0 = np.sign(D.iso_C.to_numpy(float))
    for dn, lb, lu, sg in DIRS:
        pos = (D.level == lb).to_numpy(); neg = (D.level == lu).to_numpy(); k = (tf, dn)
        assert all(np.isfinite(v[pos | neg]).all() for v in sc.values()), k   # NaN log2fc only on non-responders (L3)
        for nm, v in sc.items(): DRAWS[k + ('AUROC_unsigned', nm)] = wstats(v, pos, neg, W)['auroc']
        agree = (s0 == sg).astype(float); DRAWS[k + ('sign_agree_bound', '')] = wmean(agree * pos, W * pos); DRAWS[k + ('sign_agree_unbound', '')] = wmean(agree * neg, W * neg)
        DRAWS[k + ('sign_agree_diff', '')] = DRAWS[k + ('sign_agree_bound', '')] - DRAWS[k + ('sign_agree_unbound', '')]
        for sc_nm in SIGNC:   # (5) candidate sign definitions compared during development (Supp S1.5): agreement bound vs unbound
            sv = np.sign(D[sc_nm].to_numpy(float)); ag = (sv == sg).astype(float)
            DRAWS[k + ('signcand_agree_diff', sc_nm)] = wmean(ag * pos, W * pos) - wmean(ag * neg, W * neg)
            DRAWS[k + ('signcand_agree_responders', sc_nm)] = wmean(ag * (pos | neg), W * (pos | neg))
        for nm, t in (('T_prod', D.T_prod), ('T_k0', D.T_k0), ('sign_only_s0', s0)):
            o = sg * np.asarray(t, float); DRAWS[k + ('dT', nm)] = wmean(o * pos, W * pos) - wmean(o * neg, W * neg)
        # parity vs frozen table
        ref = REF[(REF.unit.str.contains(tf)) & REF.direction.str.startswith(dn)].set_index('metric').value
        assert abs(DRAWS[k + ('AUROC_unsigned', 'GeoBRIDGE')][0] - ref['AUROC_unsignedF_prod']) < 1e-9 and abs(DRAWS[k + ('dT', 'T_prod')][0] - ref['Tcontrast_prod']) < 1e-9, k
        # (4) |log2FC| x abundance tertile stratification
        cand = pos | neg; lfc = np.abs(D.log2fc.to_numpy(float)); ab = D.abundance.to_numpy(float)
        tert = lambda x: np.digitize(x, np.nanquantile(x[cand], [1 / 3, 2 / 3])); st = tert(lfc) * 3 + tert(ab)
        for nm in ('GeoBRIDGE', 'GeoBRIDGE k=0', 'Velorama', '|Pearson|', 'GENIE3'):
            num = den = 0.; ninf = 0
            for s in np.unique(st[cand]):
                p_, q_ = pos & (st == s), neg & (st == s)
                if p_.sum() and q_.sum(): w = p_.sum() * q_.sum(); num += w * wstats(sc[nm], p_, q_, np.ones((1, n)))['auroc'][0]; den += w; ninf += 1
            STRAT.append(dict(TF=tf, direction=dn, score=nm, AUROC_strat_lfc_x_abund=num / den if den else np.nan, n_informative_strata=ninf, n_bound=int(pos.sum()), n_unbound=int(neg.sum())))
    print(tf, 'done', flush=True)
def summ(v):
    v = np.asarray(v, float); b = v[1:][np.isfinite(v[1:])]; return v[0], np.percentile(b, 2.5), np.percentile(b, 97.5), len(b)
rows = []
for key, v in DRAWS.items():
    tf, dn, met, nm = key; p, lo, hi, nv = summ(v); rows.append(dict(TF=tf, direction=dn, metric=met, score=nm, value=p, lo=lo, hi=hi, n_valid=nv))
# pooled over the 6 contrasts (equal weight) and paired differences; per-unit draws are independent so draw-wise averaging is valid
cells = [(tf, dn) for _, tf in UNITS for dn, *_ in DIRS]
def pooled(met, nm): return np.mean(np.vstack([DRAWS[c + (met, nm)] for c in cells]), 0)   # fixed equal weights; a draw with any undefined contrast is invalid (NaN), internal review R1
def boot_p(d):   # two-sided percentile-bootstrap p for H0: 0
    b = d[1:][np.isfinite(d[1:])]; return min(1., 2 * min((b <= 0).mean(), (b >= 0).mean()) + 1 / (len(b) + 1))
PR = []
names = ['GeoBRIDGE', 'GeoBRIDGE k=0'] + COMP + ['label: |log2FC|', 'label: abundance']
for nm in names:
    d = pooled('AUROC_unsigned', nm); p, lo, hi, nv = summ(d); PR.append(dict(quantity='pooled unsigned AUROC (bound vs unbound)', score=nm, value=p, lo=lo, hi=hi, p_vs_null=boot_p(d - .5), n_valid=nv))
for nm in names[1:]:
    d = pooled('AUROC_unsigned', 'GeoBRIDGE') - pooled('AUROC_unsigned', nm); p, lo, hi, nv = summ(d); PR.append(dict(quantity='pooled unsigned AUROC: GeoBRIDGE minus', score=nm, value=p, lo=lo, hi=hi, p_vs_null=boot_p(d), n_valid=nv))
for nm in ('T_prod', 'T_k0', 'sign_only_s0'):
    d = pooled('dT', nm); p, lo, hi, nv = summ(d); PR.append(dict(quantity='pooled signed contrast dT (bound - unbound, expected direction)', score=nm, value=p, lo=lo, hi=hi, p_vs_null=boot_p(d), n_valid=nv))
d = pooled('dT', 'T_prod') - pooled('dT', 'T_k0'); p, lo, hi, nv = summ(d); PR.append(dict(quantity='pooled dT: T_prod minus T_k0', score='', value=p, lo=lo, hi=hi, p_vs_null=boot_p(d), n_valid=nv))
d = pooled('sign_agree_diff', ''); p, lo, hi, nv = summ(d); PR.append(dict(quantity='pooled sign agreement: bound minus unbound', score='s0', value=p, lo=lo, hi=hi, p_vs_null=boot_p(d), n_valid=nv))
for nm in ('sign_agree_bound', 'sign_agree_unbound'):
    d = pooled(nm, ''); p, lo, hi, nv = summ(d); PR.append(dict(quantity='pooled ' + nm, score='s0', value=p, lo=lo, hi=hi, p_vs_null=np.nan))
for sc_nm in SIGNC:
    for met in ('signcand_agree_diff', 'signcand_agree_responders'):
        d = pooled(met, sc_nm); p, lo, hi, nv = summ(d); PR.append(dict(quantity='pooled ' + met, score=sc_nm, value=p, lo=lo, hi=hi, p_vs_null=boot_p(d - (0 if 'diff' in met else .5))))
T = pd.DataFrame(rows); P = pd.DataFrame(PR); S = pd.DataFrame(STRAT)
T.to_csv(OUT + 'per_contrast.tsv', sep='\t', index=False); P.to_csv(OUT + 'pooled.tsv', sep='\t', index=False); S.to_csv(OUT + 'stratified_lfc_abund.tsv', sep='\t', index=False)
json.dump(dict(B=B, inputs={f: hashlib.sha256(open(RUN + f, 'rb').read()).hexdigest() for f in os.listdir(RUN) if f.endswith('.tsv')}), open(OUT + 'receipt.json', 'w'), indent=1)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 500)
print(T[T.metric == 'AUROC_unsigned'].pivot_table(index='score', columns=['TF', 'direction'], values='value').round(3).to_string())
print(T[T.metric != 'AUROC_unsigned'].round(3).to_string(index=False)); print(P.round(4).to_string(index=False)); print(S.round(3).to_string(index=False))
