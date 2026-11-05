"""R2-fix T3_fig3 (2026-11-01). EVALUATION/AGGREGATION ONLY on frozen outputs. usage (server): script OUT [--B 200] [--seed-trend]
 item1: summary of v2_full_B2000 (no recompute); item2: sign-vs-magnitude trend (s0, T_k0, T_prod) with fig4b_trend's wslope/null-imposed module bootstrap
 (verbatim from gb_fig4_style_2026_10_29.py); item3: Fig3 (occ_ok universe) vs Fig2 (UOLD) gene reconciliation."""
import sys, os, json, hashlib
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); sys.path[:0] = [R + 'scratch', R + 'pairspace']
import numpy as np, pandas as pd
import gb_kochip_audit_lib as KL
a = sys.argv; OUT = a[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True); B = int(a[a.index('--B') + 1]) if '--B' in a else 200
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
RUN = R + 'results/signed/fig4_bind_prod_k0_2026_10_29/v2_full_B2000/'; P1 = R + 'results/signed/kochip_audit_2026_10_15/p1_levels.tsv'
TRD = R + 'manuscript/figures_2026_10_28/fig4_dvi/style_2026_10_29/fig4b_trend.tsv'; UOLDF = R + 'results/signed/fig3_genes_uold_2026_10_29/uold_all_rows_allmethods.tsv.gz'
UNITS = [('CARDIACV_TBX5', 'TBX5', 'HEART_TBX5'), ('CARDIACV_NKX2-5', 'NKX2-5', 'HEART_NKX2-5'), ('PANC_RFX6', 'RFX6', 'PANC_RFX6')]
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.
    for r, i in enumerate(o): run = max(run, min(1., (m - r) * p[i])); adj[i] = run
    return adj
pd.set_option('display.width', 250); pd.set_option('display.max_columns', 50); pd.set_option('display.max_rows', 300)
# ---------------- item 1
S = pd.read_csv(RUN + 'fig4_bind_prod_k0.tsv', sep='\t'); Bfull = 2000
def g(m): return S[S.metric == m].set_index(['unit', 'direction'])
rows = g('Tcontrast_prod')[['n_bound', 'n_unbound']].copy()
for tag, m in (('dT_prod', 'Tcontrast_prod'), ('dT_k0', 'Tcontrast_k0'), ('dT_prod_minus_k0', 'Tcontrast_prod_minus_k0')):
    x = g(m); rows[tag] = x.value; rows[tag + '_lo'] = x.lo; rows[tag + '_hi'] = x.hi; rows[tag + '_p'] = x.p_boot; rows[tag + '_nvalid'] = x.n_valid_draws
for tag, m in (('AUROC_signed_prod', 'AUROC_signed_prod'), ('AUROC_signed_k0', 'AUROC_signed_k0'), ('AUROC_unsF_prod', 'AUROC_unsignedF_prod'), ('AUROC_unsF_k0', 'AUROC_unsignedF_k0')):
    x = g(m); rows[tag] = x.value; rows[tag + '_ci'] = x.lo.round(3).astype(str) + '..' + x.hi.round(3).astype(str); rows[tag + '_nvalid'] = x.n_valid_draws
# Holm over 6 tests; p floored at 1/(n_valid+1) (percentile-bootstrap p=0 only means < resolution)
for tag in ('dT_prod', 'dT_prod_minus_k0'):
    fl = np.maximum(rows[tag + '_p'], 1. / (rows[tag + '_nvalid'] + 1)); rows[tag + '_p_floor'] = fl; rows[tag + '_holm6'] = holm(fl.to_numpy())
# degenerate-CI flags over EVERY row of the existing table
fl = []
for _, r in S.iterrows():
    f = []
    if r.n_valid_draws < Bfull: f.append(f'nvalid={r.n_valid_draws}<{Bfull}')
    if np.isfinite(r.lo) and not (r.lo <= r.value <= r.hi): f.append('point_outside_CI')
    if np.isfinite(r.lo) and r.lo == r.hi: f.append('CI_width0')
    if r.metric.startswith(('AP_', 'AUROC_')) and ('minus' not in r.metric) and (r.hi >= 1. or r.lo <= 0.): f.append('CI_at_bound')
    if r.metric.startswith('AP_signed_') and 'minus' not in r.metric and np.isfinite(r.lo) and r.lo == r.value: f.append('lo==point(AP_floor)')
    if np.isfinite(r.p_boot) and r.p_boot == 0: f.append('p=0(resolution)')
    fl.append(';'.join(f))
S['flag'] = fl; S[S['flag'] != ''][['unit', 'direction', 'metric', 'value', 'lo', 'hi', 'p_boot', 'n_valid_draws', 'flag']].to_csv(OUT + 'item1_flags.tsv', sep='\t', index=False)
rows.reset_index().to_csv(OUT + 'item1_summary.tsv', sep='\t', index=False)
print('=== ITEM 1 (B_full=2000) ==='); show = ['n_bound', 'n_unbound'] + [c for c in rows.columns if c.startswith('dT_')]
print(rows[show].round(4).to_string()); print(rows[[c for c in rows.columns if c.startswith('AUROC')]].round(3).to_string())
print('FLAGGED rows (not all metrics shown above):'); print(S[S['flag'] != ''][['unit', 'direction', 'metric', 'value', 'lo', 'hi', 'n_valid_draws', 'flag']].round(3).to_string(index=False))
# ---------------- item 2 (wslope / null-imposed test verbatim from gb_fig4_style_2026_10_29.py:35-44)
P = pd.read_csv(P1, sep='\t'); TR0 = pd.read_csv(TRD, sep='\t').set_index('TF')
SC = {'L1': 2., 'L2': 1., 'L3': 0., 'L4': -1., 'L5': -2.}; LV = ['L1', 'L2', 'L3', 'L4', 'L5']; CM, TRR, FR = [], [], []
for u, tf, ep in UNITS:
    D = pd.read_csv(RUN + f'genes_{u}.tsv', sep='\t'); lv = D.level.to_numpy(); mods = D.module.to_numpy(); n = len(D)
    D['s0'] = np.sign(D.iso_C); print(tf, 'n', n, 'iso_C==0:', int((D.iso_C == 0).sum()), 'nonfinite iso_C:', int((~np.isfinite(D.iso_C)).sum()))
    um, inv = np.unique(mods, return_inverse=True); rng = np.random.default_rng([20261029, 41, len(um)])
    W = np.vstack([np.bincount(rng.integers(0, len(um), len(um)), minlength=len(um))[inv].astype(float) for _ in range(B)])
    sc = pd.Series(lv).map(SC).to_numpy(); pe = []
    for sname in ('s0', 'T_k0', 'T_prod'):
        t = D[sname].to_numpy(float)
        for l in LV:
            m = lv == l; wm = (W[:, m] * t[m]).sum(1) / np.where(W[:, m].sum(1) > 0, W[:, m].sum(1), np.nan); wm = wm[np.isfinite(wm)]
            CM.append(dict(TF=tf, score=sname, level=l, n=int(m.sum()), mean=t[m].mean(), lo=np.percentile(wm, 2.5), hi=np.percentile(wm, 97.5)))
        def wslope(w, y=t): xm = (w * sc).sum() / w.sum(); ym = (w * y).sum() / w.sum(); return (w * (sc - xm) * (y - ym)).sum() / (w * (sc - xm) ** 2).sum()
        s_0 = wslope(np.ones(n)); sb = np.array([wslope(w) for w in W if w.sum() > 0])
        rres = t - pd.Series(t).groupby(lv).transform('mean').to_numpy(); s_null = np.array([wslope(w, rres) for w in W if w.sum() > 0])
        lomo = [wslope((mods != m_).astype(float)) for m_ in um]
        TRR.append(dict(TF=tf, score=sname, slope=s_0, lo=np.percentile(sb, 2.5), hi=np.percentile(sb, 97.5), p=(1 + (np.abs(s_null) >= abs(s_0)).sum()) / (1 + len(s_null)), lomo_min=min(lomo), lomo_max=max(lomo)))
        if sname == 'T_prod': assert abs(s_0 - TR0.loc[tf, 'slope']) < 1e-3, (tf, s_0, TR0.loc[tf, 'slope']); print('PARITY slope', tf, round(s_0, 6), TR0.loc[tf, 'slope'])
    for l in LV:   # sign-match with KO direction: L1,L2 KO-down -> +; L4,L5 KO-up -> -; L3 reported as fraction +
        m = lv == l; s = D.s0.to_numpy()[m]; tgt = {'L1': 1, 'L2': 1, 'L4': -1, 'L5': -1}.get(l)
        FR.append(dict(TF=tf, level=l, n=int(m.sum()), frac_sign_matches_KO=(s == tgt).mean() if tgt else np.nan, frac_plus=(s > 0).mean(), frac_minus=(s < 0).mean(), frac_zero=(s == 0).mean()))
CM = pd.DataFrame(CM); TRR = pd.DataFrame(TRR); FR = pd.DataFrame(FR)
for s in TRR.score.unique(): TRR.loc[TRR.score == s, 'p_holm3'] = holm(TRR.p[TRR.score == s].to_numpy())
# parity on class means vs p1_levels (T_prod)
for u, tf, ep in UNITS:
    pp = P[P.unit == KL.UNITS[u][4]].copy(); pp['level'] = pp.level.str[:2].values
    for _, r in pp.iterrows():
        c = CM[(CM.TF == tf) & (CM.score == 'T_prod') & (CM.level == r.level)].iloc[0]; assert abs(c['mean'] - r['mean']) < 1e-6 and c.n == r.n, (tf, r.level, c['mean'], r['mean'])
print('PARITY class means T_prod vs p1_levels: OK (1e-6)')
CM.to_csv(OUT + 'item2_class_means.tsv', sep='\t', index=False); TRR.to_csv(OUT + 'item2_trend.tsv', sep='\t', index=False); FR.to_csv(OUT + 'item2_sign_fraction.tsv', sep='\t', index=False)
print('=== ITEM 2 (B=%d) ===' % B); print(CM.pivot_table(index=['TF', 'score'], columns='level', values='mean').round(3).to_string()); print(TRR.round(4).to_string(index=False)); print(FR.round(3).to_string(index=False))
# ---------------- item 3
UO = pd.read_csv(UOLDF, sep='\t', usecols=['endpoint', 'gene']); REC = []; DET = []
for u, tf, ep in UNITS:
    df = KL.load_unit(u); G = pd.read_csv(R + f'results/signed/figdec_2026_10_19/ESTCMP/genes_{u}.tsv', sep='\t'); G['gene'] = G.gene.astype(str)
    df['gene'] = df.gene.astype(str); df = df.merge(G[['gene', 'm_C2']], on='gene', how='left', validate='1:1')
    ok = {'occ_ok': df.occ_ok.fillna(False).astype(bool), 'T_prod_finite': np.isfinite(df.T_iso), 'k0_finite(m_C2)': np.isfinite(df.m_C2), 'c6_in_LV': df.c6.isin(list(KL.CLS2LVL))}
    LVm = {'A_act', 'C_act', 'B', 'D', 'C_rep', 'A_rep'}; ok['c6_in_LV'] = df.c6.isin(LVm)
    f3 = ok['occ_ok'] & ok['T_prod_finite'] & ok['k0_finite(m_C2)'] & ok['c6_in_LV']   # script lines: gb_fig4_bind_prod_k0_2026_10_29.py D = df[occ_ok & finite T_prod & finite T_k0 & c6 in LV]
    f3 &= np.isfinite(np.sign(df.iso_C) * 1.0) & np.isfinite(df.iso_C)   # T_k0 finite also needs iso_C
    uo = set(UO[UO.endpoint == ep].gene.astype(str).str.upper()); df['U'] = df.gene.str.upper()
    inU = df.U.isin(uo); print(tf, 'Fig2 UOLD n', len(uo), 'Fig3 n', int(f3.sum()), 'loader n', len(df), 'UOLD in loader', int(inU.sum()), 'Fig3&UOLD', int((f3 & inU).sum()))
    REC.append(dict(TF=tf, n_loader=len(df), n_fig2_UOLD=len(uo), n_fig2_in_loader=int(inU.sum()), n_fig2_not_in_loader=len(uo) - int(inU.sum()), n_fig3=int(f3.sum()), n_both=int((f3 & inU).sum()),
                    n_fig2_only=int((inU & ~f3).sum()), n_fig3_only=int((f3 & ~inU).sum())))
    for gp, sel in (('fig2_only', inU & ~f3), ('fig3_only', f3 & ~inU)):
        s = df[sel]
        for gname in s.gene:
            r = s[s.gene == gname].iloc[0]; fails = [k for k, v in ok.items() if not v[r.name]]
            DET.append(dict(TF=tf, group=gp, gene=gname, c6=r.c6, occ_ok=r.occ_ok, iso_C=r.iso_C, fails=';'.join(fails), first_fail=fails[0] if fails else ''))
    miss = sorted(uo - set(df.U)); DET += [dict(TF=tf, group='fig2_only', gene=x, c6='', fails='not_in_loader_universe', first_fail='not_in_loader_universe') for x in miss]
REC = pd.DataFrame(REC); DET = pd.DataFrame(DET); REC.to_csv(OUT + 'item3_universe_counts.tsv', sep='\t', index=False); DET.to_csv(OUT + 'item3_differing_genes.tsv', sep='\t', index=False)
print('=== ITEM 3 ==='); print(REC.to_string(index=False)); print(DET.groupby(['TF', 'group', 'first_fail']).size().to_string()); print(DET.groupby(['TF', 'group', 'fails']).size().to_string())
print(DET[DET.group == 'fig2_only'].groupby(['TF', 'c6']).size().to_string())
json.dump(dict(B=B, inputs={p: sha(p) for p in (P1, TRD, UOLDF, RUN + 'fig4_bind_prod_k0.tsv')}, code_sha256=sha(os.path.abspath(__file__))), open(OUT + 'receipt.json', 'w'), indent=1)
