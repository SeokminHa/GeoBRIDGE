"""Phase-1 AUDIT (EVALUATION ONLY, no production change): decomposition of the signed R1 concordance S_k = N_k/D_k into per-component contributions.
N_k = x_k^T y_k, D_k = sum_p |x_k,p||y_k,p| with x_k,y_k the pair-difference vectors after removing the first k columns of Q (QR of ordered cell-PC displacements);
c_j = (q_j^T x)(q_j^T y) so N_k = N_0 - sum_{j<=k} c_j (Q orthonormal). Same algebra/arrays as gb_signed_cache.py (parity-gated to results/signed/<SYS>.npz).
usage:
  gb_signed_decomp.py compute SYSTEM SRC,SRC OUTDIR [ROOT,ROOT] [NGENES]    (NGENES = small-N smoke: first NGENES axis genes only)
  gb_signed_decomp.py analyze OUTDIR FIGPATH                               (truth/classes only AFTER compute; classes/orientation from gb_fourclass_signed_k0.build)"""
import sys, os, time, numpy as np
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch'), (__import__("os").environ["GEOBRIDGE_ROOT"] + '/pairspace')]
R0 = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
U7 = {'KLF1_KLF1': ('GATA1', 'KLF1'), 'GATA1_GATA1': ('GATA1', 'GATA1'), 'CARDIACV_TBX5': ('CARDIACV', 'TBX5'), 'CARDIACV_NKX2-5': ('CARDIACV', 'NKX2-5'),
      'BM_GATA1_ACUTE_GATA1': ('BONEMARROW', 'GATA1'), 'PANC_RFX6': ('PANC', 'RFX6'), 'BONEMARROW_RUNX1': ('BONEMARROW', 'RUNX1')}
KM = 30


def compute(system, srcs, OUT, roots, NG):
    import gb_real_uniform as RU, gb_cop_alignment as CA
    from gb_target_extension import project_scores, CHUNK, NPC, EPS
    import gb_substrate as GS, anndata as ad
    t0 = time.time(); os.makedirs(OUT, exist_ok=True); plog = open(f'{OUT}/parity.txt', 'a')
    def P(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); plog.write(s + '\n'); plog.flush()
    rows, eg = RU.gb_raw_rows(system, srcs=srcs); axis = list(eg.astype(str)); G = len(axis); Gs = NG or G
    S = GS.load(system, canonical=True); fixture, _, _ = GS.paths(system, canonical=True); F = ad.read_h5ad(fixture)
    fi = {g: i for i, g in enumerate(np.asarray(F.var_names).astype(str))}; fc = np.array([fi[g] for g in axis])
    Mu = RU.dense(F.layers["Mu"][:, fc], np.float32); au = {g.upper(): i for i, g in enumerate(axis)}
    si = np.array([au[s] for s in srcs]); Xsrc = np.asarray(S["X"], np.float64)[:, fc[si]]
    P30 = np.asarray(F.obsm["X_pca"], float)[:, :NPC]; ns = len(srcs)
    z = np.load(f'{R0}results/signed/{system}.npz', allow_pickle=True); cs = [s.upper() for s in z['sources'].astype(str)]
    uni = {}
    for j, s in enumerate(srcs):
        u, _ = CA.observational_universe(s, rows[s], eg); u = np.asarray(u); m = np.zeros(G, bool)
        if u.dtype == bool: m = u.copy()
        else: m[u.astype(int)] = True
        uni[s] = m
    P(f'== {system} {srcs} roots={roots} genes={Gs}/{G}')
    for r in roots:
        I, J = (np.asarray(a, int) for a in S["pairs"][r]); Qf, _ = np.linalg.qr(P30[J] - P30[I]); dx = (Xsrc[J] - Xsrc[I]).astype(np.float32)
        Cx = Qf[:, :KM].T @ dx.astype(np.float64)                                   # (KM, ns)
        N = np.full((ns, G, KM + 1), np.nan); D = np.full_like(N, np.nan); Sg = np.full((ns, G, KM + 1), np.nan, np.float32); Cc = np.full((ns, G, KM), np.nan)
        exact = 0.0; exact_rel0 = 0.0
        for g0 in range(0, Gs, CHUNK):
            sl = slice(g0, min(g0 + CHUNK, Gs)); dy = (Mu[J, sl] - Mu[I, sl]).astype(np.float32)
            Cy = Qf[:, :KM].T @ dy.astype(np.float64); Cc[:, sl, :] = np.einsum('js,jg->sgj', Cx, Cy)
            for k, num, den, _z, _a in project_scores(dx, dy, Qf):
                sup = den > EPS; N[:, sl, k] = num; D[:, sl, k] = den
                Sg[:, sl, k] = np.where(sup, num / np.where(sup, den, 1.0), np.nan)
            if g0 == 0:   # float64-exact algebra check of N_k = N_0 - sum_{j<=k} c_j (no float32 residual rounding), first chunk
                x64 = dx.astype(np.float64); y64 = dy.astype(np.float64); N0 = x64.T @ y64
                for k in range(0, 9):
                    Qk = Qf[:, :k]; rx = x64 - Qk @ (Qk.T @ x64); ry = y64 - Qk @ (Qk.T @ y64); Nex = rx.T @ ry
                    ident = N0 - np.cumsum(np.concatenate([np.zeros((1,) + Cc[:, sl, 0].shape), np.moveaxis(Cc[:, sl, :], 2, 0)]), 0)[k]
                    D0c = np.abs(x64).T @ np.abs(y64); exact = max(exact, float(np.nanmax(np.abs(Nex - ident) / np.maximum(D0c, 1e-300)))); exact_rel0 = max(exact_rel0, float(np.nanmax(np.abs(Nex - ident) / np.maximum(np.abs(Nex), 1e-12 * D0c + 1e-300))))
        for j, s in enumerate(srcs):
            for a in (N, D, Sg, Cc): a[j, si[j]] = np.nan
        gm = np.zeros(G, bool); gm[:Gs] = True
        # gate (i): stored N_k (float32-residual project_scores) vs identity N_0 - cumsum c ; scale = D_0
        cum = np.concatenate([np.zeros((ns, G, 1)), np.cumsum(Cc, 2)], 2); ident = N[:, :, :1] - cum
        with np.errstate(invalid='ignore'):
            e_D0 = np.nanmax(np.abs(N - ident) / np.maximum(D[:, :, :1], 1e-300)); e_N0 = np.nanmax(np.abs(N - ident) / np.maximum(np.abs(N), 1e-300 + 1e-6 * D[:, :, :1]))
        P(f'GATE(i) {system} {r}: float64-exact identity (chunk0, k<=8) max|Nex-ident|/D0 = {exact:.3e} ; /|Nex| = {exact_rel0:.3e}  [{"PASS" if exact <= 1e-9 else "FAIL"} @1e-9]')
        P(f'GATE(i) {system} {r}: stored N_k (project_scores, float32 residuals, k<=30) vs identity: max rel to D_0 = {e_D0:.3e} ; rel to |N_k| (floor 1e-6 D0) = {e_N0:.3e}')
        for j, s in enumerate(srcs):
            a = Sg[j][:Gs]; b = z[f'{r}|U'][cs.index(s)][:Gs].astype(np.float32); m = np.isfinite(a) & np.isfinite(b); nm = int((np.isfinite(a) != np.isfinite(b)).sum())
            d = float(np.abs(a[m].astype(np.float64) - b[m]).max()) if m.any() else np.nan
            P(f'GATE(ii) {system} {r} {s}: S_k(k<=30) vs signed cache max|diff| = {d:.3e} nan-mismatch = {nm} n_cmp={int(m.sum())} [{"PASS" if d <= 1e-9 and nm == 0 else "FAIL"} @1e-9]')
            np.savez_compressed(f'{OUT}/components_{system}_{r}_{s}.npz', genes=np.array(axis), N=N[j], D=D[j], S=Sg[j], c=Cc[j].T, uni=uni[s], ngenes_done=Gs, source=s, system=system, root=r)
        P(f'{system} {r} done {time.time()-t0:.0f}s')


def main_analyze(OUT, FIGPATH):
    import pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from scipy.stats import rankdata, spearmanr
    import gb_fourclass_signed_k0 as GF
    from gb_fig3d_magnitude import up
    CL = GF.CL; RESP = ['A_act', 'A_rep', 'C_act', 'C_rep']; KS = np.arange(9); ROOTS = ('dc1_min', 'dc1_max')
    plog = open(f'{OUT}/parity.txt', 'a')
    def P(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); plog.write(s + '\n'); plog.flush()
    DAT = {}
    for unit, (sysn, tf) in U7.items():
        d = GF.build(unit); gg = d['gg']; per = []; axs = []
        for r in ROOTS:
            zz = np.load(f'{OUT}/components_{sysn}_{r}_{tf}.npz', allow_pickle=True); ix = {g: i for i, g in enumerate(up(zz['genes']))}
            m = np.array([ix.get(x, -1) for x in gg]); ok = (m >= 0); mm = np.where(ok, m, 0); un = zz['uni'].astype(bool)[mm] & ok
            f = lambda a: np.where(un[:, None], a[mm], np.nan)
            per.append(dict(N=f(zz['N']), D=f(zz['D']), S=f(zz['S'].astype(float)), c=f(zz['c'].T).copy()))
            Sax = np.where(zz['uni'].astype(bool)[:, None], zz['S'][:, :9].astype(float), np.nan)
            axs.append(dict(S=Sax, mm=mm, ok=ok))
        fin = np.ones(len(gg), bool)
        for p in per: fin &= np.isfinite(p['S'][:, :9]).all(1) & np.isfinite(p['N'][:, :9]).all(1) & np.isfinite(p['D'][:, :9]).all(1) & (np.isfinite(p['c'][:, :8]).all(1))
        for p in per:
            for k in p: p[k] = np.where(fin[:, None], p[k], np.nan)
        # consistency vs fourclass arrays
        s0 = (per[0]['S'][:, 0] + per[1]['S'][:, 0]) / 2; c = np.isfinite(s0) & np.isfinite(d['s0']); dd = float(np.abs(s0[c] - d['s0'][c]).max()) if c.any() else np.nan
        P(f'CONSIST {unit}: genes={len(gg)} analysis_set(finite k0-8 both roots)={int(fin.sum())} ; root-mean S_0 vs fourclass s0 max|diff|={dd:.2e} (n={int(c.sum())}); fourclass-finite genes lost={int((np.isfinite(d["s0"]) & ~fin).sum())}')
        import gb_primary as GP
        ra = {r: np.abs(a['S']) for r, a in zip(ROOTS, axs)}; prod_ax = GP.zcons(ra)       # PRODUCTION object: within-source z of |S_k| per depth, mean over k (>=6/9), min over roots
        Sm = (axs[0]['S'] + axs[1]['S']) / 2; Mx_ = {}
        for nm_, w_ in (('uniform_0to8', np.ones(9) / 9), ('shallow_halving', (0.5 ** (np.arange(9) + 1.0)) / (0.5 ** (np.arange(9) + 1.0)).sum())):
            Mx_['M_' + nm_] = np.abs(Sm @ w_); Mx_['zM_' + nm_] = GP.zcol(Mx_['M_' + nm_])   # |root-mean scalar|, and its within-source z over the universe
        mp_ = lambda a_: np.where(axs[0]['ok'], a_[axs[0]['mm']], np.nan)
        DAT[unit] = dict(d=d, per=per, fin=fin, prod=mp_(prod_ax), Mw={k_: mp_(v_) for k_, v_ in Mx_.items()})
    # ---------- 1. depth curves ----------
    DC, SF, CS, MG = [], [], [], []; GENE_ROWS = []; CURVE = {}
    for unit, X in DAT.items():
        d, per = X['d'], X['per']; c6 = d['c6']; Nr = [p['N'][:, :9] for p in per]; Dr = [p['D'][:, :9] for p in per]; Sr = [p['S'][:, :9] for p in per]; cr = [p['c'][:, :8] for p in per]
        Cr = [np.concatenate([np.zeros((len(c6), 1)), np.cumsum(c, 1)], 1) for c in cr]          # C_k, k=0..8
        avg = lambda L: (L[0] + L[1]) / 2
        Mx = dict(N=avg(Nr), D=avg(Dr), S=avg(Sr), absN_over_D=avg([np.abs(n) / dd_ for n, dd_ in zip(Nr, Dr)]), C=avg(Cr),
                  N_over_D0=avg([n / dd_[:, :1] for n, dd_ in zip(Nr, Dr)]), C_over_D0=avg([cc / dd_[:, :1] for cc, dd_ in zip(Cr, Dr)]))
        for cl in CL:
            sel = (c6 == cl) & X['fin']
            for mt, A in Mx.items():
                v = A[sel]
                for k in KS: DC.append(dict(unit=unit, cls=cl, metric=mt, k=int(k), n=int(sel.sum()), mean=np.nanmean(v[:, k]) if len(v) else np.nan, median=np.nanmedian(v[:, k]) if len(v) else np.nan))
        CURVE[unit] = {(cl, mt): np.array([np.nanmean(Mx[mt][(c6 == cl) & X['fin']][:, k]) if ((c6 == cl) & X['fin']).any() else np.nan for k in KS]) for cl in CL for mt in ('S', 'N_over_D0', 'C_over_D0')}
        # ---------- 2. sign fate ----------
        Nb, Db, Sb, Cb = Mx['N'], Mx['D'], Mx['S'], Mx['C']
        for cl in RESP + ['ALL_RESP']:
            selc = np.isin(c6, RESP) if cl == 'ALL_RESP' else (c6 == cl); selc = selc & X['fin']
            ex = np.where(np.char.endswith(c6.astype(str), '_act'), 1.0, -1.0)
            corr = selc & (ex * Sb[:, 0] > 0); n0 = Nb[:, 0]; n8 = Nb[:, 8]
            a_ = corr & (n8 * n0 < 0) & (np.abs(n8) > .1 * np.abs(n0)); b_ = corr & (np.abs(n8) <= .1 * np.abs(n0)); c_ = corr & ~a_ & ~b_
            if cl != 'ALL_RESP':
                GENE_ROWS.extend(dict(unit=unit, gene=d['gg'][i], cls=cl, S0=Sb[i, 0], k0_sign_correct=bool(corr[i]), N0=n0[i], N8=n8[i], D8_over_D0=Db[i, 8] / Db[i, 0], absC8_gt_absN0=bool(abs(Cb[i, 8]) > abs(n0[i])),
                                       outcome=('a_reversal' if a_[i] else 'b_collapse' if b_[i] else 'c_preserved' if c_[i] else ('k0_wrong' if selc[i] else ''))) for i in np.flatnonzero(selc))
            ratio = (Db[:, 8] / Db[:, 0])[corr]; q = np.nanpercentile(ratio, [25, 50, 75]) if corr.any() else [np.nan] * 3; nc = max(int(corr.sum()), 1)
            SF.append(dict(unit=unit, cls=cl, n_resp=int(selc.sum()), n_k0_correct=int(corr.sum()), frac_k0_correct=corr.sum() / max(selc.sum(), 1),
                           frac_a_signreversal=a_.sum() / nc, frac_b_collapse=b_.sum() / nc, frac_c_preserved=c_.sum() / nc, n_N0_S0_signdisagree=int((selc & (np.sign(n0) != np.sign(Sb[:, 0]))).sum()),
                           D8_over_D0_q25=q[0], D8_over_D0_med=q[1], D8_over_D0_q75=q[2], D8_over_D0_mean=np.nanmean(ratio) if corr.any() else np.nan,
                           frac_absC8_gt_absN0_correct=(np.abs(Cb[:, 8]) > np.abs(n0))[corr].mean() if corr.any() else np.nan,
                           frac_absC8_gt_absN0_allresp=(np.abs(Cb[:, 8]) > np.abs(n0))[selc].mean() if selc.any() else np.nan,
                           frac_S8_correct_among_k0correct=(ex * Sb[:, 8] > 0)[corr].mean() if corr.any() else np.nan))
        # ---------- 3. component sign ----------
        cb = avg(cr); D0r = [dd_[:, :1] for dd_ in Dr]; absc = avg([np.abs(c) / dd_ for c, dd_ in zip(cr, D0r)])
        ex = np.where(np.char.endswith(c6.astype(str), '_act'), 1.0, -1.0)
        for cl in CL:
            sel = (c6 == cl) & X['fin']
            if not sel.any(): continue
            tot = np.nansum([np.nanmean(absc[sel][:, j]) for j in range(8)])
            for j in range(8):
                sgn = ex[sel] if cl in RESP else np.ones(sel.sum())
                CS.append(dict(unit=unit, cls=cl, j=j + 1, n=int(sel.sum()), frac_consistent_sign=float(np.mean(sgn * cb[sel][:, j] > 0)), sign_def='KO-consistent (+act,-rep)' if cl in RESP else 'c_j>0 (no direction)',
                               mean_abs_c_over_D0=float(np.nanmean(absc[sel][:, j])), mean_oriented_c_over_D0=float(np.nanmean(sgn * (avg([c / dd_ for c, dd_ in zip(cr, D0r)]))[sel][:, j])), share_of_sum_j1to8=float(np.nanmean(absc[sel][:, j]) / tot)))
    # ---------- 4. signed-average magnitude ----------
    B = 300
    def auroc(s, y):
        r = rankdata(s); n1 = y.sum(); n0 = len(y) - n1; return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)
    def aplift(s, y):
        o = np.argsort(-s, kind='stable'); yy = y[o]; tp = np.cumsum(yy); prec = tp / np.arange(1, len(yy) + 1); return (prec[yy].sum() / yy.sum()) / yy.mean()
    SC = {}
    PN = 'production_zcons'
    for unit, X in DAT.items():
        d = X['d']; c6 = d['c6']; sc_ = {PN: X['prod'], **X['Mw']}
        ev = X['fin'] & np.isin(c6, RESP + ['B', 'D']) & np.isfinite(X['prod'])
        for v_ in sc_.values(): ev &= np.isfinite(v_)
        y = np.isin(c6, RESP); u = np.asarray(d['uns'], float); cm = ev & np.isfinite(u)
        P(f'CONSIST {unit}: Spearman(production_zcons recomputed, stored unsigned GeoBRIDGE) = {spearmanr(X["prod"][cm], u[cm])[0]:.4f} max|diff|={np.abs(X["prod"][cm]-u[cm]).max():.2e} (n={int(cm.sum())})')
        SC[unit] = dict(ev=ev, y=y, s=sc_)
    def zs(s, ev): v = s[ev]; return (s - v.mean()) / v.std()
    def stack(units, nm, zz):
        S_, Y_, U_ = [], [], []
        for u in units:
            e = SC[u]['ev']; s = SC[u]['s'][nm]; s = zs(s, e) if zz else s; S_.append(s[e]); Y_.append(SC[u]['y'][e]); U_.append(np.full(e.sum(), u))
        return np.concatenate(S_), np.concatenate(Y_), np.concatenate(U_)
    rng = np.random.default_rng(0)
    for scope in list(DAT) + ['POOLED_7']:
        units = [scope] if scope != 'POOLED_7' else list(DAT)
        for zz in (False, True):
            st = {nm: stack(units, nm, zz) for nm in SC[units[0]]['s']}; y = st[PN][1]; un = st[PN][2]
            draws = []
            for _ in range(B):   # gene bootstrap (resample genes with replacement within each unit; NOT a module bootstrap); same draws for all scores
                ix = np.concatenate([rng.choice(np.flatnonzero(un == u_), (un == u_).sum()) for u_ in np.unique(un)]); draws.append(ix)
            base = {nm: (auroc(v[0], v[1]), aplift(v[0], v[1])) for nm, v in st.items()}
            bs = {nm: np.array([(auroc(v[0][ix], v[1][ix]), aplift(v[0][ix], v[1][ix])) for ix in draws]) for nm, v in st.items()}
            for nm in st:
                pa, pl = base[PN]; row = dict(scope=scope, norm='within-source-z' if zz else 'plain', score=nm, n_resp=int(y.sum()), n_non=int((~y).sum()),
                                                                           auroc=base[nm][0], auroc_lo=np.percentile(bs[nm][:, 0], 2.5), auroc_hi=np.percentile(bs[nm][:, 0], 97.5),
                                                                           ap_lift=base[nm][1], ap_lift_lo=np.percentile(bs[nm][:, 1], 2.5), ap_lift_hi=np.percentile(bs[nm][:, 1], 97.5))
                da = bs[nm][:, 0] - bs[PN][:, 0]; dl = bs[nm][:, 1] - bs[PN][:, 1]
                row.update(d_auroc_vs_prod=base[nm][0] - pa, d_auroc_lo=np.percentile(da, 2.5), d_auroc_hi=np.percentile(da, 97.5), d_ap_vs_prod=base[nm][1] - pl, d_ap_lo=np.percentile(dl, 2.5), d_ap_hi=np.percentile(dl, 97.5)); MG.append(row)
    for nm, L in (('depth_curves', DC), ('sign_fate', SF), ('component_sign', CS), ('signed_avg_magnitude', MG)): pd.DataFrame(L).to_csv(f'{OUT}/{nm}.tsv', sep='\t', index=False)
    pd.DataFrame(GENE_ROWS).to_csv(f'{OUT}/sign_fate_genes.tsv', sep='\t', index=False)
    NOTE = ('NOTES: (1) q_j are the columns of QR(ordered cell-PC displacements P30[J]-P30[I]) -- QR-orthogonalised ordered PC-displacement columns, NOT the cell-PC axes themselves. '
            '(2) production reference = within-source z of |S_k| per depth (gb_primary.zcol), mean over k=0..8 with >=6/9 rule, min over roots (gb_primary.zcons). '
            '(3) M_w = |mean over roots of sum_k w_k S_k|; zM = within-source z (zcol over the universe) of that. norm=within-source-z rows = additional z over the evaluated set (pooled rows only differ from plain there). '
            '(4) AUROC decline with depth alone does not locate sign flips: sign_fate.tsv/sign_fate_genes.tsv record per responder whether the k=0 sign is correct at the zero threshold and what the numerator does by k=8. '
            '(5) gene bootstrap (resample genes with replacement within unit), NOT a module bootstrap; B=300.')
    open(f'{OUT}/NOTES.txt', 'w').write(NOTE + '\n'); P(NOTE)
    # ---------- figure ----------
    col = dict(zip(CL, ['#b2182b', '#ef8a62', '#2166ac', '#67a9cf', '#777777', '#bbbbbb'])); fig, ax = plt.subplots(3, len(DAT), figsize=(4 * len(DAT), 9), sharex=True)
    for i, unit in enumerate(DAT):
        for r_, (mt, lab) in enumerate((('S', 'mean S_k'), ('N_over_D0', 'mean N_k / D_0'), ('C_over_D0', 'mean C_k / D_0'))):
            for cl in CL: ax[r_, i].plot(KS, CURVE[unit][(cl, mt)], '-o', ms=3, color=col[cl], label=cl)
            ax[r_, i].axhline(0, color='k', lw=.5)
            if i == 0: ax[r_, 0].set_ylabel(lab)
            if r_ == 0: ax[0, i].set_title(unit, fontsize=9)
            if r_ == 2: ax[2, i].set_xlabel('k')
    ax[0, 0].legend(fontsize=7); os.makedirs(os.path.dirname(FIGPATH), exist_ok=True); fig.tight_layout(); fig.savefig(FIGPATH, dpi=130); P('figure', FIGPATH)


if __name__ == '__main__':
    if sys.argv[1] == 'compute':
        import gb_real_uniform as RU
        roots = sys.argv[5].split(',') if len(sys.argv) > 5 else list(RU.ROOTS); NG = int(sys.argv[6]) if len(sys.argv) > 6 else None
        compute(sys.argv[2].upper(), [s.upper() for s in sys.argv[3].split(',')], sys.argv[4], roots, NG)
    else:
        main_analyze(sys.argv[2], sys.argv[3])
