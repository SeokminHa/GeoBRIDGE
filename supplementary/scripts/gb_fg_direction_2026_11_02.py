"""T14 truth-free dev-subspace fraction f_g and KO-direction (EVALUATION ONLY). Wrapper of gb_fig3d_depth_multi_2026_10_29.py (same pattern as gb_matched_baseline_real_2026_11_02.py):
f_g = 1 - ||(I-Q8 Q8^T) y_g||^2/||y_g||^2, y_g = Mu_g[J]-Mu_g[I] (canonical pairs per root), Q = QR(P30[J]-P30[I]), Q8 = Q[:, :8]; mean over the two roots. Regulator-independent, truth-free.
Outputs: fg_auroc.tsv (f_g as score, vs nonresponders / KOdown vs KOup), fg_halves.tsv (D=cum0-8 and K0=cum0-0 within low/high-f halves, median split on common rows), fg_arm_reslice.tsv (c).
usage (server, repo root): script OUT --endpoints A[,B] [--B 9999]"""
import sys, numpy as np, anndata as ad
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); CAN = R + 'results/canonical_2026_09_18/'
src = open(R + 'scratch/gb_fig3d_depth_multi_2026_10_29.py').read().replace('\r', '')
old_subs = src[src.index('SUBS = {'):src.index("PAIRS = [")]; src = src.replace(old_subs, "SUBS = {'cum0-0': [0], 'cum0-8': list(range(9))}\n")
old_pairs = src[src.index('PAIRS = ['):src.index('\n', src.index('PAIRS = ['))]
src = src.replace(old_pairs, "PAIRS = [('cum0-8', 'cum0-0')]")
dense = lambda M: np.asarray(M.todense() if hasattr(M, 'todense') else M)
FCACHE = {}
def fgene(fx, genes):
    if fx in FCACHE: return FCACHE[fx]
    F = ad.read_h5ad(f'{CAN}{fx}_canonical.h5ad'); vn = np.char.upper(np.asarray(F.var_names).astype(str)); pos = {v: i for i, v in enumerate(vn)}
    gi = np.array([pos.get(x.upper(), -1) for x in genes]); ok = gi >= 0; P30 = np.asarray(F.obsm['X_pca'], float)[:, :30]; Mu = F.layers['Mu']
    pz = np.load(f'{CAN}{fx}_canonical_pairs.npz', allow_pickle=True); fr = []
    for r in ('dc1_min', 'dc1_max'):
        I, J = np.asarray(pz[f'{r}_I'], int), np.asarray(pz[f'{r}_J'], int); Q, _ = np.linalg.qr(P30[J] - P30[I]); Q8 = Q[:, :8]; f = np.full(len(genes), np.nan); ii = np.flatnonzero(ok)
        for g0 in range(0, len(ii), 500):
            sl = ii[g0:g0 + 500]; Mc = dense(Mu[:, gi[sl]]).astype(np.float64); y = Mc[J] - Mc[I]; n2 = (y * y).sum(0); res = y - Q8 @ (Q8.T @ y)
            with np.errstate(invalid='ignore', divide='ignore'): f[sl] = 1 - (res * res).sum(0) / n2
        fr.append(f)
    FCACHE[fx] = (fr[0] + fr[1]) / 2; print(fx, 'f_g finite', int(np.isfinite(FCACHE[fx]).sum()), '/', len(genes), 'range', np.nanmin(FCACHE[fx]), np.nanmax(FCACHE[fx]), flush=True); return FCACHE[fx]
FR, HR = [], []
old_v = 'V = {nm: sub(ks) for nm, ks in SUBS.items()}'; assert old_v in src
_anchor = "    y0, e = D[lab]['y0'], D[lab]['e']"; assert src.count(_anchor) == 1
src = src.replace(_anchor, """
    _f = mapto(genes, fgene(fx, genes), g); _fin0 = Uc & fin & np.isfinite(_f)
    print(lab, 'f_g rows', int(_fin0.sum()), 'of common', int((Uc & fin).sum()), flush=True)
    _y0, _e = D[lab]['y0'], D[lab]['e']; _med = np.median(_f[_fin0]); _fz = np.where(_fin0, _f, np.nan)
    _dirs = {'ALL': (_y0.copy(), np.ones_like(_y0)), 'KOup': (_y0 & np.isfinite(_e) & (_e > 0), None), 'KOdown': (_y0 & np.isfinite(_e) & (_e < 0), None)}
    _dirs['KOup'] = (_dirs['KOup'][0], _dirs['KOup'][0] | ~_y0); _dirs['KOdown'] = (_dirs['KOdown'][0], _dirs['KOdown'][0] | ~_y0)
    _cmp = {'KOdown_vs_KOup': (_y0 & np.isfinite(_e) & (_e < 0), _y0 & np.isfinite(_e))}
    for _nm, (_yl, _rw) in {**{k: v for k, v in _dirs.items() if k != 'ALL'}, **_cmp}.items():
        _rr = _fin0 & _rw; _np, _nn = int((_yl & _rr).sum()), int((~_yl & _rr).sum())
        if _np < 5 or _nn < 5: print(lab, _nm, 'skipped', _np, _nn); continue
        _p, _b = CT.eval_rows(_fz, _yl, _rr, X['gi'], X['C']); _bm = np.asarray(_b['AUROC']); _fm = np.isfinite(_bm); _ok = _fm.mean() >= .95
        FR.append(dict(endpoint=lab, contrast=_nm, auroc_f=_p['AUROC'], lo=np.percentile(_bm[_fm], 2.5) if _ok else np.nan, hi=np.percentile(_bm[_fm], 97.5) if _ok else np.nan,
                       p_boot=PT.boot_p(_bm[_fm] - .5) if _ok else np.nan, n_pos=_np, n_neg=_nn, n_rows=int(_rr.sum())))
    for _h, _hm in (('lowf', _fin0 & (_f <= _med)), ('highf', _fin0 & (_f > _med))):
        for _d, (_yl, _rw) in _dirs.items():
            _rr = _hm & _rw; _np, _nn = int((_yl & _rr).sum()), int((~_yl & _rr).sum())
            if _np < 5 or _nn < 5: print(lab, _h, _d, 'skipped', _np, _nn); continue
            _E = {nm: CT.eval_rows(V[nm], _yl, _rr, X['gi'], X['C']) for nm in ('cum0-8', 'cum0-0')}
            for _m in ('AUROC', 'AUPRC'):
                _bD, _bK = np.asarray(_E['cum0-8'][1][_m]), np.asarray(_E['cum0-0'][1][_m]); _row = dict(endpoint=lab, half=_h, direction=_d, metric=_m, f_median=_med, n_pos=_np, n_neg=_nn, n_rows=int(_rr.sum()))
                for _a, _bb, _nmx in ((_E['cum0-8'], _bD, 'D'), (_E['cum0-0'], _bK, 'K0')):
                    _fm = np.isfinite(_bb); _ok = _fm.mean() >= .95; _row[_nmx] = _a[0][_m]; _row[_nmx + '_lo'] = np.percentile(_bb[_fm], 2.5) if _ok else np.nan; _row[_nmx + '_hi'] = np.percentile(_bb[_fm], 97.5) if _ok else np.nan
                _lo, _hi, _pb, _nv = gate(_bD - _bK); _row.update(D_minus_K0=_E['cum0-8'][0][_m] - _E['cum0-0'][0][_m], dlo=_lo, dhi=_hi, p_boot=_pb, n_valid=_nv); HR.append(_row)
    print(lab, 'f analyses done', flush=True)
""" + _anchor)
src += """
pd.DataFrame(FR).to_csv(OUT + 'fg_auroc.tsv', sep='\\t', index=False); pd.DataFrame(HR).to_csv(OUT + 'fg_halves.tsv', sep='\\t', index=False)
print(pd.DataFrame(FR).round(3).to_string(index=False)); print(pd.DataFrame(HR)[lambda d: d.metric == 'AUROC'].round(3).to_string(index=False))
if 'PANC_RFX6' in EPS:
    TB = pd.read_csv(R + 'results/signed/fig3d_rebuild_2026_10_29/full/fig3d_rebuild_table.tsv', sep='\\t')
    TB = TB[(TB.kind == 'D_minus') & TB.direction.isin(['KOup', 'KOdown']) & (TB.metric == 'AUROC') & TB.arm.isin(['K0', 'SPL', 'RND', 'PS']) & TB.level.isin(['PANC_RFX6', 'BM_GATA1_acute', 'HEART_TBX5', 'HEART_NKX2-5'])]
    TB[['level', 'arm', 'direction', 'value', 'lo', 'hi', 'p_boot']].sort_values(['level', 'arm', 'direction']).to_csv(OUT + 'fg_arm_reslice.tsv', sep='\\t', index=False); print(TB[['level', 'arm', 'direction', 'value', 'lo', 'hi']].round(3).to_string(index=False))
"""
exec(compile(src, 'depth_multi', 'exec'))
