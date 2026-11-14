# Near-isochronic polarity scoring from observational inputs.
import sys, os, json, time, hashlib, socket
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch'), (__import__("os").environ["GEOBRIDGE_ROOT"] + '/pairspace'), __import__("os").environ["GEOBRIDGE_ROOT"]]
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp
import gb_iso_lib as L

R0 = L.R0; SG = L.SG; OUT = SG + 'iso_pilot_2026_10_15/'; ROOTS = L.ROOTS
FREEZE = OUT + 'iso_freeze.json'
TDIR = SG + 'grand_v3d_dumpT_2026_10_15/parts/'
WORK = SG + 'splitrefit_2026_10_14/work'
PHB = SG + 'actrep_B_2026_10_15/parts/'
t00 = time.time()


def log(*a):
    print(f'[{time.time() - t00:7.0f}s]', *a, flush=True)


def dense(M, dtype=np.float32):
    return np.asarray(M.toarray() if hasattr(M, 'toarray') else M, dtype=dtype)


def load_freeze():
    txt = open(FREEZE).read(); h = hashlib.sha256(txt.encode()).hexdigest()
    assert h == open(FREEZE + '.sha256').read().split()[0], 'freeze sha mismatch'
    return json.loads(txt), h


def rootmean(a, b):
    return np.where(np.isfinite(a) & np.isfinite(b), (a + b) / 2, np.nan)


def get_uni(bsy, focal):
    sc = np.load(f'{SG}{bsy}.npz', allow_pickle=True)
    srcs = [s.upper() for s in sc['sources'].astype(str)]
    if focal.upper() not in srcs:
        sc = np.load(f'{SG}{bsy}_bank.npz', allow_pickle=True); srcs = [s.upper() for s in sc['sources'].astype(str)]
    j = srcs.index(focal.upper()); uni = np.asarray(sc['uni'][j]); axis = np.char.upper(sc['genes'].astype(str))
    if not (uni.dtype == bool or (uni.dtype == object and isinstance(uni.flat[0], (bool, np.bool_)))):
        m = np.zeros(len(axis), bool); m[uni.astype(int)] = True; uni = m
    return uni.astype(bool), axis


def strata_codes(obs, cols):
    if not cols:
        return np.zeros(len(obs), np.int64)
    key = obs[cols].astype(str).agg('|'.join, axis=1)
    return pd.factorize(key, sort=True)[0].astype(np.int64)


def pair_diag(I, J, n, pt, strata, Bsym, Aself, Cdpt):
    d = L.degree_stats(I, J, n)
    d.update(n_pairs=int(len(I)), mean_abs_dpt=float(np.abs(pt[J] - pt[I]).mean()), frac_cross_strata=float((strata[I] != strata[J]).mean()))
    adj, jac = L.adj_jaccard(Bsym, Aself, I, J)
    d.update(adj_smoothing_graph=float(adj.mean()), jaccard_mean=float(jac.mean()), jaccard_median=float(np.median(jac)), frac_jaccard_gt0=float((jac > 0).mean()), frac_jaccard_ge_0p5=float((jac >= 0.5).mean()))
    d['adj_dpt_graph'] = float((np.asarray(Cdpt[I, J]).ravel() > 0).mean() + 0.0) if Cdpt is not None else None
    return d


# SCORE STAGE
def score(system, focal, smoke=False):
    FZ, fsha = load_freeze(); cfg = FZ['systems'][system]; sd = FZ['sources'][f'{system}/{focal}']; units = sd['units']; sources = sd['sources']
    tag = f'{system}_{focal}'; od = OUT + f'scores/{tag}' + ('_smoke' if smoke else '') + '/'; pdir = OUT + f'pairs/{system}/'
    os.makedirs(od, exist_ok=True); os.makedirs(pdir, exist_ok=True)
    log('SCORE', tag, 'units', units, 'sources', sources, 'freeze sha', fsha[:12], 'host', socket.gethostname())
    # canonical loader (fail-closed validation) + pairs
    import gb_substrate as GS
    S = GS.load(system, canonical=True); fixture, pairs_npz, _ = GS.paths(system, canonical=True)
    assert fixture == cfg['fixture'] and pairs_npz == cfg['pairs_npz']
    zp = np.load(pairs_npz, allow_pickle=True); assert str(zp['fixture_sha256']) == cfg['fixture_sha256'], 'fixture differs from freeze'
    assert L.sha_file(pairs_npz) == cfg['pairs_sha256'], 'pairs npz differs from freeze'
    fwd = {r: (np.asarray(S['pairs'][r][0], np.int64), np.asarray(S['pairs'][r][1], np.int64)) for r in ROOTS}
    for r in ROOTS:
        assert np.array_equal(fwd[r][0], zp[f'{r}_I'].astype(np.int64)) and np.array_equal(fwd[r][1], zp[f'{r}_J'].astype(np.int64))
    del S
    F = ad.read_h5ad(fixture); n = F.n_obs; obs_names = np.asarray(F.obs_names).astype(str)
    for r in ROOTS:
        assert (zp[f'{r}_cells'].astype(str) == obs_names).all()
    pt = {r: np.asarray(zp[f'{r}_pt'], np.float64) for r in ROOTS}
    strata = strata_codes(F.obs, cfg['strata_columns_used']); log('n cells', n, 'strata', np.bincount(strata).tolist())
    # axis / sources / targets
    uni, axis = get_uni(system, focal)
    vn = np.char.upper(np.asarray(F.var_names).astype(str)); cnt = pd.Series(vn).value_counts(); fi = {}
    for j, g in enumerate(vn): fi.setdefault(g, j)
    amb = [g for g in axis if cnt.get(g, 0) != 1]; assert not amb, ('axis genes absent/ambiguous in fixture', amb[:5])
    ax_i = {g: i for i, g in enumerate(axis)}
    T = {}
    for u in units:
        t = pd.read_csv(TDIR + f'T_{u}.tsv', sep='\t', usecols=lambda c: c != 'cls'); t['gene'] = t.gene.astype(str).str.upper(); T[u] = t
        assert len(t) == FZ['units'][u]['n_rows'] and 'cls' not in t.columns
    tg = list(dict.fromkeys(g for u in units for g in T[u].gene))
    if smoke:
        tg = tg[:200] + [g for g in sources if g.upper() not in tg[:200]]; T = {u: t[t.gene.isin(set(tg))].reset_index(drop=True) for u, t in T.items()}
    tgt = np.array(tg); tgt_ax = np.array([ax_i[g] for g in tgt]); tgt_col = np.array([fi[g] for g in tgt]); Gt = len(tgt)
    src_ax = np.array([ax_i[s.upper()] for s in sources]); src_col = np.array([fi[s.upper()] for s in sources]); ns = len(sources)
    Xs = dense(F.layers['Ms'][:, src_col], np.float32).astype(np.float64)
    Yt = dense(F.layers['Mu'][:, tgt_col], np.float32)
    Mst = dense(F.layers['Ms'][:, tgt_col], np.float32)
    abund = Mst.mean(0, dtype=np.float64); spdet = (dense(F.layers['spliced'][:, tgt_col]) > 0).mean(0); undet = (dense(F.layers['unspliced'][:, tgt_col]) > 0).mean(0)
    src_spdet = (dense(F.layers['spliced'][:, src_col]) > 0).mean(0); src_undet = (dense(F.layers['unspliced'][:, src_col]) > 0).mean(0); del Mst
    mask_t = uni[tgt_ax]; self_t = np.stack([tgt == s.upper() for s in sources])          # (ns, Gt)
    log('sources', ns, 'targets', Gt, 'in-uni targets', int(mask_t.sum()), 'load done')
    # smoothing graph (moments kNN) and DPT graph
    Cs = F.obsp['connectivities'].tocsr(); Bsym = L.sym_bool(Cs); Aself = L.with_self(Bsym)
    Cdpt = {}
    for r in ROOTS:
        Cdpt[r] = sp.csr_matrix((zp[f'{r}_conn_data'], zp[f'{r}_conn_indices'], zp[f'{r}_conn_indptr']), shape=tuple(zp[f'{r}_conn_shape']))
    # pair sets
    PS = {}; pm = dict(); pman_path = pdir + 'pairs_manifest.json'; pman = json.load(open(pman_path)) if os.path.exists(pman_path) else {}
    for ri, r in enumerate(ROOTS):
        PS[(r, 'fwd')] = fwd[r]
        fI, fJ = L.restrict_pairs(*fwd[r], strata); PS[(r, 'fwdS')] = (fI, fJ)
        for sd_ in L.SEEDS:
            PS[(r, f'iso{sd_}')] = L.iso_pairs(pt[r], strata, L.WINDOW, L.CAP, sd_, ri)
        rI, rJ, rinfo = L.rewire_pairs(*PS[(r, f'iso{L.REWIRE_SEED}')], strata, L.REWIRE_SEED)
        PS[(r, 'rew42')] = (rI, rJ); pm[f'{r}|rew42'] = rinfo
        log(r, {k[1]: len(v[0]) for k, v in PS.items() if k[0] == r}, 'rewire', rinfo)
    for (r, kind), (I, J) in PS.items():
        h = L.sha_np(I.astype(np.int64), J.astype(np.int64)); f = pdir + f'{r}_{kind}.npz'
        if kind != 'fwd':
            if os.path.exists(f) and f'{r}|{kind}' in pman:
                assert pman[f'{r}|{kind}']['sha256'] == h, f'NON-DETERMINISTIC pair set {r} {kind}'
            else:
                np.savez_compressed(f, I=I.astype(np.int32), J=J.astype(np.int32))
        pman[f'{r}|{kind}'] = dict(n_pairs=int(len(I)), sha256=h, file=(f if kind != 'fwd' else 'canonical pairs npz (not copied)'))
    json.dump(pman, open(pman_path, 'w'), indent=1)
    # diagnostics per pair set
    diag = {}
    for (r, kind), (I, J) in PS.items():
        diag[f'{r}|{kind}'] = pair_diag(I, J, n, pt[r], strata, Bsym, Aself, Cdpt[r]); diag[f'{r}|{kind}'].update(pm.get(f'{r}|{kind}', {}))
    log('pair diagnostics done')
    # pair sums
    RAW = {}
    kinds_unc = ['fwd', 'fwdS'] + [f'iso{s}' for s in L.SEEDS] + ['rew42']; kinds_cen = ['fwd', 'fwdS']
    for r in ROOTS:
        for kind in kinds_unc:
            if kind == 'fwdS' and len(PS[(r, 'fwdS')][0]) == len(PS[(r, 'fwd')][0]): RAW[(r, 'fwdS', 'unc')] = RAW[(r, 'fwd', 'unc')]; continue
            RAW[(r, kind, 'unc')] = L.pair_sums(Xs, Yt, *PS[(r, kind)], centred=False)
        for kind in kinds_cen:
            if kind == 'fwdS' and len(PS[(r, 'fwdS')][0]) == len(PS[(r, 'fwd')][0]): RAW[(r, 'fwdS', 'cen')] = RAW[(r, 'fwd', 'cen')]; continue
            RAW[(r, kind, 'cen')] = L.pair_sums(Xs, Yt, *PS[(r, kind)], centred=True)
        log('pair sums done', r)
    cellP = L.cell_pearson(Xs, Yt)
    # rows (ns, Gt)
    R = {}
    C = lambda r, k, m: L.ratio_C(RAW[(r, k, m)]['N'], RAW[(r, k, m)]['D'])
    PPf = lambda r, k, m: L.ratio_PP(RAW[(r, k, m)]['N'], RAW[(r, k, m)]['Sxx'], RAW[(r, k, m)]['Syy'])
    R['fwd_C0'] = rootmean(C(ROOTS[0], 'fwd', 'unc'), C(ROOTS[1], 'fwd', 'unc'))
    R['fwd_d4f'] = rootmean(C(ROOTS[0], 'fwd', 'cen'), C(ROOTS[1], 'fwd', 'cen'))
    R['fwdPP'] = rootmean(PPf(ROOTS[0], 'fwd', 'cen'), PPf(ROOTS[1], 'fwd', 'cen'))
    R['fwdS_C0'] = rootmean(C(ROOTS[0], 'fwdS', 'unc'), C(ROOTS[1], 'fwdS', 'unc'))
    R['fwdS_d4'] = rootmean(C(ROOTS[0], 'fwdS', 'cen'), C(ROOTS[1], 'fwdS', 'cen'))
    R['fwdS_PP'] = rootmean(PPf(ROOTS[0], 'fwdS', 'cen'), PPf(ROOTS[1], 'fwdS', 'cen'))
    iso_c = {}; iso_pp = {}
    for r in ROOTS:
        cs = np.stack([C(r, f'iso{s}', 'unc') for s in L.SEEDS]); ps = np.stack([PPf(r, f'iso{s}', 'unc') for s in L.SEEDS])
        iso_c[r] = np.where(np.isfinite(cs).all(0), cs.mean(0), np.nan); iso_pp[r] = np.where(np.isfinite(ps).all(0), ps.mean(0), np.nan)
        iso_c[r + '_seeds'] = cs
    R['iso_C'] = rootmean(iso_c[ROOTS[0]], iso_c[ROOTS[1]]); R['iso_PP'] = rootmean(iso_pp[ROOTS[0]], iso_pp[ROOTS[1]])
    R['iso_C_rootmin'] = iso_c[ROOTS[0]]; R['iso_C_rootmax'] = iso_c[ROOTS[1]]
    R['rew_C'] = rootmean(C(ROOTS[0], 'rew42', 'unc'), C(ROOTS[1], 'rew42', 'unc')); R['rew_PP'] = rootmean(PPf(ROOTS[0], 'rew42', 'unc'), PPf(ROOTS[1], 'rew42', 'unc'))
    allsg = np.concatenate([np.sign(iso_c[r + '_seeds']) for r in ROOTS], 0)                        # (6, ns, Gt)
    R['iso_rep'] = np.where(np.isfinite(allsg).all(0), np.abs(allsg.mean(0)), np.nan)
    pe = lambda r, k: RAW[(r, k, 'unc')]['Syy'] / RAW[(r, k, 'unc')]['P']
    R['iso_Et'] = np.broadcast_to(np.mean([np.mean([pe(r, f'iso{s}') for s in L.SEEDS], 0) for r in ROOTS], 0)[None, :], (ns, Gt)).copy()
    R['fwd_Et'] = np.broadcast_to(np.mean([pe(r, 'fwd') for r in ROOTS], 0)[None, :], (ns, Gt)).copy()
    R['cellMsMu_fresh'] = cellP
    for k in list(R):
        if k in ('iso_Et', 'fwd_Et'): continue
        R[k] = np.where(mask_t[None, :] & ~self_t, R[k], np.nan)
    log('rows built')
    # cached references for the focal (stage-1 caches), mapped to targets
    cached = {}
    z1 = {r: np.load(cfg['stage1_caches'][r]['path'], allow_pickle=True) for r in ROOTS}
    for r in ROOTS:
        assert L.sha_file(cfg['stage1_caches'][r]['path']) == cfg['stage1_caches'][r]['sha256']
    s1src = [s.upper() for s in z1[ROOTS[0]]['sources'].astype(str)]; j1 = s1src.index(focal.upper())
    assert (np.char.upper(z1[ROOTS[0]]['genes'].astype(str)) == axis).all()
    ud = lambda a: np.where(uni, a.astype(np.float64), np.nan)
    Uc = {r: z1[r]['U'][j1] for r in ROOTS}; ax_self = ax_i[focal.upper()]
    def axis_row(name, f):
        a = f(); a[ax_self] = np.nan; return a
    cached['d1'] = axis_row('d1', lambda: rootmean(ud(Uc[ROOTS[0]][:, 0]), ud(Uc[ROOTS[1]][:, 0])))
    cached['d4'] = axis_row('d4', lambda: rootmean(ud(z1[ROOTS[0]]['Sc0'][j1]), ud(z1[ROOTS[1]]['Sc0'][j1])))
    cached['deepk8_cached_full'] = axis_row('k8', lambda: rootmean(ud(Uc[ROOTS[0]][:, 8]), ud(Uc[ROOTS[1]][:, 8])))
    cached['cellMsMu'] = axis_row('pm', lambda: ud(z1[ROOTS[0]]['pear_MsMu'][j1]))
    cached = {k: v[tgt_ax] for k, v in cached.items()}
    # Phase B D_both (A/B-fixture cached), focal only
    phb = None; phb_path = [PHB + f'{system}_{focal}_{r}.npz' for r in ROOTS]
    if all(os.path.exists(p) for p in phb_path):
        ph = {}
        for r, p in zip(ROOTS, phb_path):
            z = np.load(p, allow_pickle=True); assert z['sources'][0].upper() == focal.upper()
            with np.errstate(invalid='ignore', divide='ignore'):
                num = z['Nr_A'][0] + z['Nr_B'][0]; den = z['Dr_A'][0] + z['Dr_B'][0]
            ph[r] = (np.where(den > L.EPS, num / np.where(den > L.EPS, den, 1), np.nan), np.char.upper(z['genes'].astype(str)))
        assert (ph[ROOTS[0]][1] == axis).all()
        a = rootmean(ph[ROOTS[0]][0], ph[ROOTS[1]][0]); a[ax_self] = np.nan; phb = a[tgt_ax]
    # parity QC vs the T score columns (no labels) and fresh-vs-cached
    qc = dict(); tF = {}
    for u in units:
        t = T[u]; zd4 = t['Z|d4'].to_numpy(float); zd1 = t['Z|d1'].to_numpy(float)
        Fz = np.where(np.isfinite(zd4) & (zd4 != 0), np.abs(zd4), np.where(np.isfinite(zd1) & (zd1 != 0), np.abs(zd1), np.nan))
        both = np.isfinite(zd4) & np.isfinite(zd1) & (zd4 != 0) & (zd1 != 0); fdiff = float(np.abs(np.abs(zd4[both]) - np.abs(zd1[both])).max()) if both.any() else None
        tix = pd.Series(np.arange(Gt), index=tgt).loc[t.gene].to_numpy()
        for nm, cs_, zz in (('d4', cached['d4'][tix], zd4), ('d1', cached['d1'][tix], zd1)):
            fin = np.isfinite(cs_) & np.isfinite(zz) & (cs_ != 0) & (zz != 0)
            qc[f'{u}|cached_{nm}_vs_T'] = dict(n_both=int(fin.sum()), sign_mismatch=int((np.sign(cs_[fin]) != np.sign(zz[fin])).sum()), finite_pattern_mismatch=int((np.isfinite(cs_) != np.isfinite(zz)).sum()),
                                                  n_T_finite=int(np.isfinite(zz).sum()), n_cached_finite=int(np.isfinite(cs_).sum()))
        for nm, fr, cs_ in (('d4f_vs_cached_d4', R['fwd_d4f'][0][tix], cached['d4'][tix]), ('C0_vs_cached_d1', R['fwd_C0'][0][tix], cached['d1'][tix]), ('cellMsMu_fresh_vs_cached', R['cellMsMu_fresh'][0][tix], cached['cellMsMu'][tix])):
            fin = np.isfinite(fr) & np.isfinite(cs_)
            qc[f'{u}|{nm}'] = dict(n_both=int(fin.sum()), max_abs_diff=float(np.abs(fr[fin] - cs_[fin]).max()) if fin.any() else None, sign_mismatch=int((np.sign(fr[fin]) != np.sign(cs_[fin])).sum()),
                                  finite_pattern_mismatch=int((np.isfinite(fr) != np.isfinite(cs_)).sum()))
        qc[f'{u}|F_from_d4_vs_d1'] = dict(max_abs_diff=fdiff, n_F_finite=int(np.isfinite(Fz).sum()), n_rows=int(len(t)))
        tF[u] = Fz
    # features per unit (truth-free) -- one TSV per unit
    ok_cols = {}
    for u in units:
        t = T[u]; tix = pd.Series(np.arange(Gt), index=tgt).loc[t.gene].to_numpy()
        df = pd.DataFrame({'gene': t.gene.to_numpy(), 'F': tF[u], 'T_Zd1': t['Z|d1'].to_numpy(float), 'T_Zd4': t['Z|d4'].to_numpy(float), 'abundance': abund[tix], 'spliced_det': spdet[tix], 'unspliced_det': undet[tix], 'in_uni': mask_t[tix]})
        for k in ('d1', 'd4', 'deepk8_cached_full', 'cellMsMu'): df[k] = cached[k][tix]
        df['PhaseB_D_both'] = phb[tix] if phb is not None else np.nan
        for k in ('fwd_C0', 'fwd_d4f', 'fwdPP', 'fwdS_C0', 'fwdS_d4', 'fwdS_PP', 'iso_C', 'iso_PP', 'iso_C_rootmin', 'iso_C_rootmax', 'rew_C', 'rew_PP', 'iso_rep', 'iso_Et', 'fwd_Et'): df[k] = R[k][0][tix]
        df.to_csv(od + f'features_{u}.tsv', sep='\t', index=False); ok_cols[u] = list(df.columns)
    # feasibility (per unit): coverage, root disagreement, gates
    feas = dict(freeze_sha256=fsha, system=system, focal=focal, pair_diag=diag, units={})
    en = {}
    for r in ROOTS:
        for kind in ['fwd', 'fwdS', 'rew42'] + [f'iso{s}' for s in L.SEEDS]:
            a = RAW[(r, kind, 'unc')]; ref = RAW[(r, 'fwd', 'unc')]; Pk, Pf = a['P'], ref['P']
            es = (a['Sxx'][0] / Pk) / (ref['Sxx'][0] / Pf); et = (a['Syy'] / Pk) / (ref['Syy'] / Pf); fin = np.isfinite(et) & (et > 0) & (ref['Syy'] > 0)
            en[f'{r}|{kind}'] = dict(retained_pair_energy_ratio_source_focal=float(es), retained_pair_energy_ratio_targets_median=float(np.median(et[fin])), total_energy_ratio_source_focal=float(a['Sxx'][0] / ref['Sxx'][0]),
                                    total_energy_ratio_targets_median=float(np.median((a['Syy'] / ref['Syy'])[fin])), mean_sq_pair_diff_source_focal=float(a['Sxx'][0] / Pk))
    feas['energy'] = en
    feas['source_detection'] = {s: dict(spliced_det=float(a), unspliced_det=float(b)) for s, a, b in zip(sources, src_spdet, src_undet)}
    feas['target_detection_median'] = dict(spliced=float(np.median(spdet)), unspliced=float(np.median(undet)))
    for u in units:
        df = pd.read_csv(od + f'features_{u}.tsv', sep='\t'); fz = df.F.notna().to_numpy(); ic = df.iso_C.to_numpy(float)
        a0, a1 = df.iso_C_rootmin.to_numpy(float), df.iso_C_rootmax.to_numpy(float); bothr = np.isfinite(a0) & np.isfinite(a1) & (a0 != 0) & (a1 != 0)
        mp = min(diag[f'{r}|iso{s}']['n_cells'] for r in ROOTS for s in L.SEEDS); cov = float(np.isfinite(ic[fz]).mean()) if fz.any() else np.nan
        feas['units'][u] = dict(n_T_rows=int(len(df)), n_F_finite=int(fz.sum()), coverage_iso_C_over_F_finite=cov, coverage_iso_C_over_all_T_rows=float(np.isfinite(ic).mean()),
                                root_sign_disagreement_iso_C=float((np.sign(a0[bothr]) != np.sign(a1[bothr])).mean()) if bothr.any() else None,
                                mean_iso_rep=float(np.nanmean(df.iso_rep)), frac_iso_rep_eq_1=float((df.iso_rep.dropna() == 1).mean()),
                                min_participating_cells=int(mp), gate_cells_ge_100=bool(mp >= 100), gate_coverage_ge_0p80=bool(cov >= 0.80), FEASIBLE=bool(mp >= 100 and cov >= 0.80))
        for nm in ('d1', 'd4', 'fwd_d4f', 'fwdPP', 'cellMsMu', 'fwdS_C0', 'rew_C'):
            x = df[nm].to_numpy(float); y = df['iso_C'].to_numpy(float); m = np.isfinite(x) & np.isfinite(y) & (x != 0) & (y != 0)
            feas['units'][u][f'sign_agreement_{nm}_vs_iso_C'] = float((np.sign(x[m]) == np.sign(y[m])).mean()) if m.any() else None
    feas['qc_parity'] = qc
    # save arrays
    np.savez_compressed(od + 'rows.npz', sources=np.array(sources), targets=tgt, mask_t=mask_t, self_t=self_t, **{f'row|{k}': v for k, v in R.items()},
                        **{f'cached|{k}': v for k, v in cached.items()}, phaseB=(phb if phb is not None else np.full(Gt, np.nan)))
    np.savez_compressed(od + 'raw_sums.npz', sources=np.array(sources), targets=tgt, **{f'{k[0]}|{k[1]}|{k[2]}|{q}': (RAW[k][q] if q != 'P' else np.array(RAW[k][q])) for k in RAW for q in ('N', 'D', 'Sxx', 'Syy', 'sx', 'sy', 'P')})
    json.dump(feas, open(od + 'feasibility.json', 'w'), indent=1, default=str)
    man = dict(freeze_sha256=fsha, script_sha256=L.sha_file(os.path.abspath(__file__)), lib_sha256=L.sha_file(os.path.abspath(L.__file__)), host=socket.gethostname(), runtime_s=time.time() - t00, smoke=smoke,
               files={f: L.sha_file(od + f) for f in sorted(os.listdir(od)) if f != 'manifest.json'}, pairs_manifest_sha256=L.sha_file(pman_path))
    json.dump(man, open(od + 'manifest.json', 'w'), indent=1)
    log('SCORE DONE', tag, f'{time.time() - t00:.0f}s')
    # console summary (no truth)
    for u in units:
        print(u, json.dumps(feas['units'][u]))
    for k, v in qc.items():
        print('QC', k, v)
    print('PAIR DIAG'); print(pd.DataFrame(diag).T[['n_pairs', 'n_cells', 'max_over_median', 'gini', 'mean_abs_dpt', 'frac_cross_strata', 'adj_smoothing_graph', 'adj_dpt_graph', 'jaccard_mean', 'frac_jaccard_gt0']].round(4).to_string())
    print('ENERGY'); print(pd.DataFrame(en).T.round(4).to_string())


if __name__ == '__main__':
    m = sys.argv[1]
    if m == 'score':
        score(sys.argv[2].upper(), sys.argv[3].upper(), smoke='--smoke' in sys.argv)
    elif m == 'ab':
        import gb_iso_ab as AB
        AB.run(sys.argv[2].upper(), sys.argv[3].upper(), smoke='--smoke' in sys.argv)
