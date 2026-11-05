"""E15 (2026-11-03) EVALUATION ONLY: Fig-4 KO x ChIP five-class analysis under each sign candidate. T_c = sign_c * F (sign 0/NaN -> 0).
Module bootstrap / ordered-trend / null-imposed test as gb_r2fix_t3_fig3_2026_11_01.py item 2 (= gb_fig4_style_2026_10_29.py), vectorised over draws.
usage: script SIGNDIR OUT [--B 2000] [--units TBX5,NKX2-5,RFX6] [--cands a,b,..]"""
import sys, os, json
import numpy as np, pandas as pd
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); RUN = R + 'results/signed/fig4_bind_prod_k0_2026_10_29/v2_full_B2000/'; TRD = R + 'manuscript/figures_2026_10_28/fig4_dvi/style_2026_10_29/fig4b_trend.tsv'
a = sys.argv; SD = a[1].rstrip('/') + '/'; OUT = a[2].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
arg = lambda k, d: a[a.index(k) + 1] if k in a else d
B = int(arg('--B', 2000)); SEL = arg('--units', 'TBX5,NKX2-5,RFX6').split(','); CF = arg('--cands', None)
UNITS = {'TBX5': 'CARDIACV_TBX5', 'NKX2-5': 'CARDIACV_NKX2-5', 'RFX6': 'PANC_RFX6'}
LV = ['L1', 'L2', 'L3', 'L4', 'L5']; SC = dict(L1=2., L2=1., L3=0., L4=-1., L5=-2.); TR0 = pd.read_csv(TRD, sep='\t').set_index('TF')
CM, TRR, CON, AG = [], [], [], []
def wmean(W, m, t): d = W[:, m].sum(1); return (W[:, m] * t[m]).sum(1) / np.where(d > 0, d, np.nan)
for tf in SEL:
    u = UNITS[tf]; D = pd.read_csv(RUN + f'genes_{u}.tsv', sep='\t'); Sg = pd.read_csv(SD + f'signs_{u}.tsv', sep='\t')
    assert Sg.gene.is_unique and D.gene.astype(str).str.upper().isin(Sg.gene).all(), 'gene join'
    Sg = Sg.set_index('gene').loc[D.gene.astype(str).str.upper()]; assert (Sg.cls.values == D.cls.values).all()
    lv = D.level.to_numpy(); mods = D.module.to_numpy(); n = len(D); F = D.F.to_numpy(float); sc = pd.Series(lv).map(SC).to_numpy()
    um, inv = np.unique(mods, return_inverse=True); rng = np.random.default_rng([20261029, 41, len(um)])
    W = np.vstack([np.bincount(rng.integers(0, len(um), len(um)), minlength=len(um))[inv].astype(float) for _ in range(B)]); W = W[W.sum(1) > 0]
    def wslope(W_, y): sw = W_.sum(1); xm = (W_ @ sc) / sw; ym = (W_ @ y) / sw; return (W_ * (sc[None] - xm[:, None]) * (y[None] - ym[:, None])).sum(1) / (W_ * (sc[None] - xm[:, None]) ** 2).sum(1)
    for c in [x[5:] for x in Sg.columns if x.startswith('sign|')]:
        if CF and c not in CF.split(','): continue
        s = np.nan_to_num(Sg['sign|' + c].to_numpy(float), nan=0.0); t = s * F
        if c == 'I-k0(adopted)': assert np.abs(t - D.T_prod.to_numpy(float)).max() < 1e-9, 'PARITY T_prod'
        for l in LV:
            m = lv == l; wm = wmean(W, m, t); wm = wm[np.isfinite(wm)]
            CM.append(dict(TF=tf, cand=c, level=l, n=int(m.sum()), mean=t[m].mean(), lo=np.percentile(wm, 2.5), hi=np.percentile(wm, 97.5)))
        w1 = np.ones((1, n)); s0 = wslope(w1, t)[0]; sb = wslope(W, t); res = t - pd.Series(t).groupby(lv).transform('mean').to_numpy(); sn = wslope(W, res)
        if c == 'I-k0(adopted)': assert abs(s0 - TR0.loc[tf, 'slope']) < 1e-3, ('PARITY slope', s0); print('PARITY OK', tf, round(s0, 4), flush=True)
        TRR.append(dict(TF=tf, cand=c, slope=s0, lo=np.percentile(sb, 2.5), hi=np.percentile(sb, 97.5), p=(1 + (np.abs(sn) >= abs(s0)).sum()) / (1 + len(sn))))
        for nm, la, lb in (('dT_down', 'L1', 'L2'), ('dT_up', 'L4', 'L5')):
            ma, mb = lv == la, lv == lb; obs = t[ma].mean() - t[mb].mean(); bs = wmean(W, ma, t) - wmean(W, mb, t); ok = np.isfinite(bs)
            r = t - np.where(ma, t[ma].mean(), np.where(mb, t[mb].mean(), 0.)); nd = (wmean(W, ma, r) - wmean(W, mb, r))[ok]
            CON.append(dict(TF=tf, cand=c, contrast=nm, diff=obs, lo=np.percentile(bs[ok], 2.5), hi=np.percentile(bs[ok], 97.5), p=(1 + (np.abs(nd) >= abs(obs)).sum()) / (1 + ok.sum())))
        ad, au = (s[(lv == 'L1') | (lv == 'L2')] == 1).mean(), (s[(lv == 'L4') | (lv == 'L5')] == -1).mean()
        AG.append(dict(TF=tf, cand=c, agree_down=ad, agree_up=au, BA=(ad + au) / 2, coverage=(s[np.isin(lv, ['L1', 'L2', 'L4', 'L5'])] != 0).mean()))
    print(tf, 'done', flush=True)
CM, TRR, CON, AG = map(pd.DataFrame, (CM, TRR, CON, AG)); tag = '_'.join(SEL)
for nm, d in (('class_means', CM), ('trend', TRR), ('contrasts', CON), ('sign_agree', AG)): d.to_csv(OUT + f'{nm}_{tag}.tsv', sep='\t', index=False)
print(TRR.round(4).to_string(index=False)); print(CON.round(4).to_string(index=False)); print(AG.round(3).to_string(index=False))
