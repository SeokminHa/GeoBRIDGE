"""Fig 4 (internal directive 2026-10-29 'Strengthen Figure 4's direct-versus-indirect story'): binding-supported vs unsupported KO responders WITHIN each response
direction, production GeoBRIDGE vs depth-0 magnitude. EVALUATION ONLY on cached, frozen per-gene tables (original Fig 4 universe; no rescoring).
internal review review (internal design note (not shipped)):
  - rows / production score come from the canonical Fig 4 loader scratch/gb_kochip_audit_lib.load_unit (T_iso = sign(iso_C)*|Z|d4|, features, labels, RP);
    universe = occ_ok (ChIP-testable) rows, as Fig 4. F_prod = |Z|d4| on the ORIGINAL dump reference (not re-percentiled after filtering).
  - depth-0 magnitude arm: m_C2 = forward production-style k=0 z root-min (scratch/gb_estcmp_main.py EST 'C2'); F_k0 = percentile of m_C2 over the SAME full dump
    reference; T_k0 = sign(iso_C)*F_k0 ('depth-0 magnitude, production sign': isolates magnitude conditioning; not a sign ablation, not the benchmarked C2).
  - AP: tie-aware, zero-weight-safe weighted statistics copied verbatim from scratch/gb_estcmp_main.wstats; any p-value is reported only when >= 95% draws are valid.
  - adds unsigned F discrimination (bound vs unbound on F) for both arms; detection (spliced_det) tercile stratification; within-module AUROC with
    leave-one-module-out range; module concentration.
Orientation: KO-down -> +T favours bound; KO-up -> -T favours bound. Module bootstrap over frozen control modules (aux module column), B draws.
All results descriptive/exploratory. usage (server, repo root): script OUT [--B 2000]"""
import sys, os, json, hashlib
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); sys.path[:0] = [R + 'scratch', R + 'pairspace']
import numpy as np, pandas as pd
from scipy.stats import rankdata
import gb_kochip_audit_lib as KL
a = sys.argv; OUT = a[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True); B = int(a[a.index('--B') + 1]) if '--B' in a else 2000
UNITS = ['CARDIACV_TBX5', 'CARDIACV_NKX2-5', 'PANC_RFX6']
LV = {'A_act': 'L1', 'C_act': 'L2', 'B': 'L3', 'D': 'L3', 'C_rep': 'L4', 'A_rep': 'L5'}
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
def wstats(score, pos, neg, W):   # verbatim scratch/gb_estcmp_main.py wstats (ties = one threshold block; weights = module multiplicities)
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
def auc1(s, pos, neg): return float(wstats(s, pos, neg, np.ones((1, len(s))))['auroc'][0])
def strat_auc(s, pos, neg, strata):
    num = den = 0.0; k_ = 0
    for v in np.unique(strata[pos | neg]):
        pk, nk = pos & (strata == v), neg & (strata == v)
        if pk.sum() and nk.sum(): w8 = pk.sum() * nk.sum(); num += w8 * auc1(s, pk, nk); den += w8; k_ += 1
    return (num / den if den else np.nan), k_
def summ(v):
    f = v[1:][np.isfinite(v[1:])]; ok = len(f) >= .95 * (len(v) - 1)
    return (np.percentile(f, 2.5), np.percentile(f, 97.5), 2 * min((f <= 0).mean(), (f >= 0).mean()), len(f)) if ok else (np.nan, np.nan, np.nan, len(f))
INPUTS = {}; ROWS, COUNTS, SENS, CONC = [], [], [], []
for unit in UNITS:
    name = KL.UNITS[unit][4]; df = KL.load_unit(unit)
    pg = R + f'results/signed/figdec_2026_10_19/ESTCMP/genes_{unit}.tsv'; pa = R + f'results/signed/kochip_audit_2026_10_15/aux_{unit}.tsv'
    for p in (pg, pa): INPUTS[p] = sha(p)
    G = pd.read_csv(pg, sep='\t'); A = pd.read_csv(pa, sep='\t')
    for t in (G, A): t['gene'] = t.gene.astype(str); assert t.gene.is_unique
    df = df.merge(G[['gene', 'm_C2', 'm_A', 's_A']], on='gene', how='left', validate='1:1').merge(A[['gene', 'module']], on='gene', how='left', validate='1:1')
    bothf = np.isfinite(df.m_A) & np.isfinite(df.F); okS = np.isfinite(df.s_A) & np.isfinite(df.iso_C) & (df.iso_C != 0)
    assert np.allclose(df.m_A[bothf], df.F[bothf]) and (np.sign(df.iso_C[okS]) == df.s_A[okS]).all(), (unit, 'ESTCMP/Fig4 mismatch')
    ref = np.isfinite(df.m_C2.to_numpy(float)); df['F_k0'] = np.nan; df.loc[ref, 'F_k0'] = (rankdata(df.m_C2[ref]) - .5) / ref.sum()   # full-dump reference
    df['T_prod'] = df.T_iso; df['T_k0'] = np.sign(df.iso_C) * df.F_k0; df['dF'] = df.F - df.F_k0
    D = df[df.occ_ok.fillna(False).astype(bool) & np.isfinite(df.T_prod) & np.isfinite(df.T_k0) & df.c6.isin(LV)].reset_index(drop=True)
    assert D.module.notna().all(); D['level'] = D.c6.map(LV); n = len(D)
    COUNTS.append(dict(unit=name, n=n, **D.level.value_counts().reindex(['L1', 'L2', 'L3', 'L4', 'L5']).fillna(0).astype(int).to_dict()))
    D.to_csv(OUT + f'genes_{unit}.tsv', sep='\t', index=False)
    mods = D.module.to_numpy(); um, inv = np.unique(mods, return_inverse=True); rng = np.random.default_rng([20261029, len(um)])
    W = np.vstack([np.ones(n)] + [np.bincount(rng.integers(0, len(um), len(um)), minlength=len(um))[inv].astype(float) for _ in range(B)])
    det = D.spliced_det.to_numpy(float); lfc = np.abs(D.log2fc.to_numpy(float))
    for dname, b_, u_, sgn in (('KO-down (L1 vs L2)', 'L1', 'L2', 1.), ('KO-up (L5 vs L4)', 'L5', 'L4', -1.)):
        pos = (D.level == b_).to_numpy(); neg = (D.level == u_).to_numpy(); cand = pos | neg; stat = {}
        for arm in ('prod', 'k0'):
            o = sgn * D[f'T_{arm}'].to_numpy(); f_ = D['F' if arm == 'prod' else 'F_k0'].to_numpy()
            ws = wstats(o, pos, neg, W); stat[f'AUROC_signed_{arm}'] = ws['auroc']; stat[f'AP_signed_{arm}'] = ws['ap']; stat[f'APlift_signed_{arm}'] = ws['lift']
            stat[f'AUROC_unsignedF_{arm}'] = wstats(f_, pos, neg, W)['auroc']
            stat[f'Tcontrast_{arm}'] = wmean(o * pos, W * pos) - wmean(o * neg, W * neg)
        for m_ in ('AUROC_signed', 'AP_signed', 'AUROC_unsignedF', 'Tcontrast'): stat[f'{m_}_prod_minus_k0'] = stat[f'{m_}_prod'] - stat[f'{m_}_k0']
        dF = D.dF.to_numpy(); stat['rankshift_bound'] = wmean(dF * pos, W * pos); stat['rankshift_unbound'] = wmean(dF * neg, W * neg)
        stat['rankshift_bound_minus_unbound'] = stat['rankshift_bound'] - stat['rankshift_unbound']
        for k_, v in stat.items():
            lo, hi, p, nv = summ(v); test = ('minus' in k_ or 'contrast' in k_ or 'shift' in k_)
            ROWS.append(dict(unit=name, direction=dname, metric=k_, value=v[0], lo=lo, hi=hi, p_boot=p if test else np.nan, n_valid_draws=nv, n_bound=int(pos.sum()), n_unbound=int(neg.sum()),
                             prevalence=pos.sum() / cand.sum()))
        # sensitivities (point estimates): |log2FC| terciles, detection terciles, within module (+ leave-one-module-out range)
        tert = lambda x: np.digitize(x, np.nanquantile(x[cand], [1 / 3, 2 / 3]))
        sl, sd = tert(lfc), tert(det)
        for arm in ('prod', 'k0'):
            o = sgn * D[f'T_{arm}'].to_numpy(); wm, km = strat_auc(o, pos, neg, mods)
            lomo = []
            for m2 in np.unique(mods[cand]):
                keep = mods != m2
                if (pos & keep).sum() and (neg & keep).sum(): lomo.append(strat_auc(o, pos & keep, neg & keep, mods)[0])
            ok_l = all(((sl == k) & pos).sum() >= 3 and ((sl == k) & neg).sum() >= 3 for k in range(3)); ok_d = all(((sd == k) & pos).sum() >= 3 and ((sd == k) & neg).sum() >= 3 for k in range(3))
            SENS.append(dict(unit=name, direction=dname, arm=arm, AUROC_signed=auc1(o, pos, neg), AUROC_lfc_terciles=strat_auc(o, pos, neg, sl)[0] if ok_l else np.nan,
                             lfc_call='OK' if ok_l else 'NO-CALL', AUROC_detection_terciles=strat_auc(o, pos, neg, sd)[0] if ok_d else np.nan, det_call='OK' if ok_d else 'NO-CALL',
                             AUROC_within_module=wm, n_informative_modules=km, within_module_LOMO_min=np.nanmin(lomo) if lomo else np.nan, within_module_LOMO_max=np.nanmax(lomo) if lomo else np.nan))
        cb, cu = pd.Series(mods[pos]).value_counts(), pd.Series(mods[neg]).value_counts()
        CONC.append(dict(unit=name, direction=dname, n_bound=int(pos.sum()), modules_bound=int(cb.size), top3_share_bound=float(cb.head(3).sum() / max(cb.sum(), 1)),
                         n_unbound=int(neg.sum()), modules_unbound=int(cu.size), top3_share_unbound=float(cu.head(3).sum() / max(cu.sum(), 1)),
                         det_median_bound=float(np.nanmedian(det[pos])), det_median_unbound=float(np.nanmedian(det[neg])), abslfc_median_bound=float(np.nanmedian(lfc[pos])), abslfc_median_unbound=float(np.nanmedian(lfc[neg]))))
T = pd.DataFrame(ROWS); T.to_csv(OUT + 'fig4_bind_prod_k0.tsv', sep='\t', index=False); pd.DataFrame(COUNTS).to_csv(OUT + 'fig4_level_counts.tsv', sep='\t', index=False)
pd.DataFrame(SENS).to_csv(OUT + 'fig4_sensitivity.tsv', sep='\t', index=False); pd.DataFrame(CONC).to_csv(OUT + 'fig4_module_concentration.tsv', sep='\t', index=False)
json.dump(dict(B=B, inputs=INPUTS, kl_sha256=sha(R + 'scratch/gb_kochip_audit_lib.py'), code_sha256=sha(os.path.abspath(__file__))), open(OUT + 'receipt.json', 'w'), indent=1)
pd.set_option('display.width', 250); pd.set_option('display.max_rows', 300)
print(pd.DataFrame(COUNTS).to_string(index=False))
print(T[T.metric.str.contains('AUROC|rankshift_bound_minus|AP_signed')].round(3).to_string(index=False))
print(pd.DataFrame(SENS).round(3).to_string(index=False)); print(pd.DataFrame(CONC).round(3).to_string(index=False))
