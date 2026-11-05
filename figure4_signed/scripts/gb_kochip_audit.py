"""KOCHIP_AUDIT_2026_10_15 stage driver (EVALUATION ONLY of saved scores; PRELIMINARY / exploratory).
usage: gb_kochip_audit.py STAGE [UNIT,UNIT,..]    STAGE in p0 aux p1 p1b p2 p3 p4 p5 p6 p7 p8 p9(=tables over all units)
Outputs: results/signed/kochip_audit_2026_10_15/<stage>_*.tsv.  No score is recomputed; only truth-free gene modules / trajectory rho from CONTROL Ms (stage aux)."""
import sys, os, json, time
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch'), (__import__("os").environ["GEOBRIDGE_ROOT"] + '/pairspace')]
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu, wilcoxon, spearmanr, rankdata
import gb_kochip_audit_lib as K
from gb_kochip_audit_lib import *

os.makedirs(OUT, exist_ok=True)
STAGE = sys.argv[1]; UL = sys.argv[2].split(',') if len(sys.argv) > 2 and sys.argv[2] != 'all' else list(UNITS)
B = int(os.environ.get('KB', 2000)); t00 = time.time()
def log(*a): print(f'[{time.time() - t00:6.0f}s]', *a, flush=True)
def save(df, fn): df.to_csv(OUT + fn, sep='\t', index=False); log('wrote', fn, df.shape)
def rng_for(*tag): return np.random.default_rng([SEED] + [sum(ord(c) * (i + 1) for i, c in enumerate(str(t))) for t in tag])
def ordc(s, lvl, keep=None):
    g = lvl.copy(); s = np.asarray(s, float)
    if keep is not None: g = np.where(keep, g, -1)
    m = g >= 0; return ordinal(s[m], g[m])


# =============================================================== P0
def p0():
    ref = pd.read_csv(FIGDIR + 'fig5e_isosign_vs_d4_contrasts.tsv', sep='\t'); n0 = len(ref); ref = ref.drop_duplicates(); log('ref rows', n0, '->', len(ref), 'after exact-dup removal')
    assert not ref.duplicated(['unit', 'contrast']).any(), 'ref has conflicting duplicates'
    rows = []; fail = 0
    for u in UNITS:
        df = load_unit(u); nm = UNITS[u][4]
        for tag, col in (('d4', 'Z|d4'), ('iso', 'T_iso')):
            g = levels(df, col); ps = [mwu_one(g[i], g[j], a) for i, j, a in PAIRS]; adj = holm(ps)
            for (i, j, a), p, h in zip(PAIRS, ps, adj):
                lab = f"L{i + 1}{'>' if a == 'greater' else '<'}L{j + 1}"; r = ref[(ref.unit == nm) & (ref.contrast == lab)]
                assert len(r) == 1, (nm, lab)
                r = r.iloc[0]; ni, nj, mi, mj, pr, ph = (r[f'{k}_{tag}'] for k in ('n_i', 'n_j', 'mean_i', 'mean_j', 'p_raw', 'p_holm6'))
                ok = (ni == len(g[i])) and (nj == len(g[j])) and np.isclose(mi, g[i].mean(), rtol=0, atol=1e-9) and np.isclose(mj, g[j].mean(), rtol=0, atol=1e-9) \
                    and np.isclose(pr, p, rtol=1e-9, atol=1e-15, equal_nan=True) and np.isclose(ph, h, rtol=1e-9, atol=1e-15, equal_nan=True)
                fail += (not ok); rows.append(dict(unit=nm, score=tag, contrast=lab, n_i=len(g[i]), n_j=len(g[j]), mean_i=g[i].mean(), mean_j=g[j].mean(), p_raw=p, p_holm6=h,
                                                    ref_n_i=ni, ref_n_j=nj, ref_mean_i=mi, ref_mean_j=mj, ref_p_raw=pr, ref_p_holm6=ph, match=bool(ok)))
    res = pd.DataFrame(rows); save(res, 'p0_reproduction.tsv')
    # PI-stated level counts / means (iso score)
    exp = {'Heart · TBX5': ([10, 61, 978, 165, 16], [.69, .38, .02, -.21, -.22]), 'Heart · NKX2-5': ([63, 163, 785, 211, 54], [.30, .11, -.01, -.16, -.20])}
    stated = []
    for nm, (ns, ms) in exp.items():
        u = [k for k, v in UNITS.items() if v[4] == nm][0]; g = levels(load_unit(u), 'T_iso'); got_n = [len(x) for x in g]; got_m = [round(x.mean(), 2) for x in g]
        okn = got_n == ns; okm = np.allclose(got_m, ms, atol=0.011); fail += (not okn) + (not okm); stated.append(dict(unit=nm, n_stated=ns, n_got=got_n, mean_stated=ms, mean_got=got_m, n_ok=okn, mean_ok=okm))
    save(pd.DataFrame(stated), 'p0_pi_stated_values.tsv'); print(pd.DataFrame(stated).to_string())
    log('P0 mismatches:', fail, 'of', len(res) + 2 * len(exp))
    json.dump(dict(n_compared=int(len(res)), n_mismatch=int(fail), pass_=bool(fail == 0)), open(OUT + 'p0_status.json', 'w'))
    if fail: print(res[~res.match].to_string()); sys.exit('P0 FAILED')
    print('P0 PASS')


# =============================================================== AUX (control Ms: modules, trajectory rho, source diagnostics)
def aux():
    import anndata as ad, scipy.sparse as sp
    from scipy.cluster.hierarchy import linkage, fcluster
    from scipy.spatial.distance import squareform
    import gb_iso_lib as L
    FZ = json.load(open(IP + 'iso_freeze.json')); done = {}
    for u in UL:
        sysn, foc, sd, ab, name, grp = UNITS[u]; df = load_unit(u); cfg = FZ['systems'][sysn]
        if sysn not in done:
            A = ad.read_h5ad(cfg['fixture']); vn = np.char.upper(np.asarray(A.var_names).astype(str)); Ms = A.layers['Ms']; Ms = Ms.toarray() if sp.issparse(Ms) else np.asarray(Ms)
            fz = np.load(cfg['pairs_npz'], allow_pickle=True); obs = np.asarray(A.obs_names).astype(str); pt = {}
            for r in L.ROOTS:
                c2t = dict(zip(fz[f'{r}_cells'].astype(str), np.asarray(fz[f'{r}_pt'], float))); pt[r] = np.array([c2t[c] for c in obs])
            done[sysn] = (vn, Ms.astype(np.float32), pt); del A; log('loaded fixture', sysn, Ms.shape)
        vn, Ms, pt = done[sysn]; pos = {}
        for j, g in enumerate(vn): pos.setdefault(g, j)
        have = df.gene_up.map(lambda g: g in pos).to_numpy(); idx = np.array([pos[g] for g in df.gene_up[have]]); X = Ms[:, idx].astype(np.float64)
        log(u, 'genes in fixture', int(have.sum()), 'of', len(df))
        # truth-free modules: average-linkage on 1-corr of control Ms over the evaluated genes, fixed k
        k_mod = 50; Xc = X - X.mean(0); sdv = np.sqrt((Xc ** 2).sum(0)); ok = sdv > 0; C = np.zeros((X.shape[1],) * 2); C[np.ix_(ok, ok)] = (Xc[:, ok].T @ Xc[:, ok]) / np.outer(sdv[ok], sdv[ok]); np.fill_diagonal(C, 1.0)
        Dm = np.clip(1 - C, 0, 2); Dm = (Dm + Dm.T) / 2; np.fill_diagonal(Dm, 0)
        lab = fcluster(linkage(squareform(Dm, checks=False), 'average'), k_mod, 'maxclust'); mod = np.full(len(df), -1, int); mod[np.where(have)[0]] = lab
        miss = np.where(~have)[0]; mod[miss] = lab.max() + 1 + np.arange(len(miss))      # genes absent from the fixture: singleton modules
        # trajectory association: spearman of Ms with the full-data root pt (dc1_min / dc1_max), abs, mean of the two roots
        rho = {}
        for r in L.ROOTS:
            rk = rankdata(X, axis=0); tr = rankdata(pt[r]); rk -= rk.mean(0); tr -= tr.mean(); rho[r] = (rk.T @ tr) / (np.sqrt((rk ** 2).sum(0)) * np.sqrt((tr ** 2).sum()) + 1e-300)
        out = pd.DataFrame(dict(gene=df.gene.values, module=mod, in_fixture=have, rho_dc1min=np.nan, rho_dc1max=np.nan, mean_Ms=np.nan)); out.loc[have, 'rho_dc1min'] = rho[L.ROOTS[0]]; out.loc[have, 'rho_dc1max'] = rho[L.ROOTS[1]]; out.loc[have, 'mean_Ms'] = X.mean(0)
        out['abs_rho_pt'] = np.nanmean(np.abs(out[['rho_dc1min', 'rho_dc1max']].to_numpy()), 1); save(out, f'aux_{u}.tsv')
        # source diagnostics (focal + 10 alternatives) for P7
        z = np.load(IP + f'scores/{sd}/rows.npz', allow_pickle=True); srcs = [str(s).upper() for s in z['sources']]; rows = []
        for s in srcs:
            if s not in pos: rows.append(dict(unit=u, source=s, in_fixture=False)); continue
            x = Ms[:, pos[s]].astype(np.float64); d = dict(unit=u, source=s, in_fixture=True, is_focal=(s == foc.upper()), mean_Ms=x.mean(), det=float((x > 0).mean()))
            for r in L.ROOTS: d[f'rho_{r}'] = float(spearmanr(x, pt[r])[0])
            rows.append(d)
        save(pd.DataFrame(rows), f'srcdiag_{u}.tsv')



# =============================================================== helpers
LVN = [x[2] for x in LV]
def amask(df): return (df.lvl.to_numpy() >= 0) & np.isfinite(df.T_iso.to_numpy())
def ord_boot(groups, K, Bn, rng):
    """stratified-by-group bootstrap of ordinal c and all pair AUCs; groups=list of arrays (group 0 expected highest). returns (Bn, 1+npairs), pair order (a,b) a<b"""
    prs = [(a, b) for a in range(K) for b in range(a + 1, K)]; out = np.full((Bn, 1 + len(prs)), np.nan)
    if sum(len(g) > 0 for g in groups) < 2: return out, prs
    for i in range(Bn):
        gs = [g[rng.integers(0, len(g), len(g))] if len(g) else g for g in groups]; s = np.concatenate(gs); gi = np.concatenate([np.full(len(g), k) for k, g in enumerate(gs)])
        c, U, n = ordinal(s, gi, K); out[i, 0] = c
        for j, p in enumerate(prs): out[i, 1 + j] = U.get(p, np.nan)
    return out, prs
def variant_groups(df, which):
    s = df.T_iso.to_numpy(); l = df.lvl.to_numpy(); ok = np.isfinite(s)
    if which == 'five_level': return [s[ok & (l == i)] for i in range(5)], 5
    if which == 'responders4': return [s[ok & (l == i)] for i in (0, 1, 3, 4)], 4
    if which == 'three_level': return [s[ok & np.isin(l, (0, 1))], s[ok & (l == 2)], s[ok & np.isin(l, (3, 4))]], 3
def flat(groups): return np.concatenate(groups), np.concatenate([np.full(len(g), k) for k, g in enumerate(groups)])


# =============================================================== P1
def p1():
    rows, brows, lrows = [], [], []
    for u in UL:
        df = load_unit(u); name = UNITS[u][4]
        for which in ('five_level', 'responders4', 'three_level'):
            gr, Kk = variant_groups(df, which); s, g = flat(gr); c, Ud, n = ordinal(s, g, Kk); z, pz, JT, mu, var = jt_test(s, g, Kk)
            bt, prs = ord_boot(gr, Kk, B, rng_for(u, which)); lo, hi = ci(bt[:, 0])
            rows.append(dict(unit=name, unit_id=u, variant=which, K=Kk, n=list(map(int, n)), c=c, c_lo=lo, c_hi=hi, JT=JT, JT_mean0=mu, JT_z=z, JT_p_onesided=pz, B=B))
            if which == 'five_level':
                den = sum(n[a] * n[b] for (a, b) in Ud)
                for j, (a, b) in enumerate(prs):
                    if (a, b) not in Ud: continue
                    w = n[a] * n[b] / den; l_, h_ = ci(bt[:, 1 + j]); brows.append(dict(unit=name, pair=f'L{a + 1}>L{b + 1}', adjacent=(b == a + 1), n_a=int(n[a]), n_b=int(n[b]), P_correct=Ud[(a, b)], lo=l_, hi=h_, weight=w, contribution=w * Ud[(a, b)], share_of_c=w * Ud[(a, b)] / c))
                for i in range(5):
                    x = gr[i]
                    if not len(x): lrows.append(dict(unit=name, level=LVN[i].replace('\n', ' '), n=0)); continue
                    r = rng_for(u, 'lev', i); ix = r.integers(0, len(x), (B, len(x))); mm = x[ix].mean(1); md = np.median(x[ix], 1)
                    lrows.append(dict(unit=name, level=f'L{i + 1} {LVN[i]}', n=len(x), mean=x.mean(), mean_lo=ci(mm)[0], mean_hi=ci(mm)[1], median=np.median(x), median_lo=ci(md)[0], median_hi=ci(md)[1],
                                      P_correct_vs_next=(Ud.get((i, i + 1), np.nan))))
    save(pd.DataFrame(rows), 'p1_ordering.tsv'); save(pd.DataFrame(brows), 'p1_boundaries.tsv'); save(pd.DataFrame(lrows), 'p1_levels.tsv')


# =============================================================== P1b
NREP = int(os.environ.get('KNREP', 300)); NCOV = int(os.environ.get('KNCOV', 150)); BM = int(os.environ.get('KBM', 300))
def mod_boot_c(s, g, mods_idx, Bn, rng, K=5):
    out = np.full(Bn, np.nan); M = len(mods_idx)
    for b in range(Bn):
        ix = np.concatenate([mods_idx[j] for j in rng.integers(0, M, M)]); out[b] = ordinal(s[ix], g[ix], K)[0]
    return out
def iid_boot_c(s, g, Bn, rng, K=5):
    out = np.full(Bn, np.nan); n = len(s)
    for b in range(Bn): ix = rng.integers(0, n, n); out[b] = ordinal(s[ix], g[ix], K)[0]
    return out
def p1b():
    rows, crow = [], []
    for u in UL:
        df = load_unit(u); name = UNITS[u][4]; a = aux_df(u); df = df.merge(a[['gene', 'module']], on='gene', how='left', validate='1:1'); m = amask(df)
        s = df.T_iso.to_numpy()[m]; g = df.lvl.to_numpy()[m]; mod = df.module.to_numpy()[m]; um = np.unique(mod); idxs = [np.where(mod == x)[0] for x in um]
        c0, U0, n0 = ordinal(s, g); sz = np.array([len(i) for i in idxs]); dn = np.isin(g, (0, 1)); up_ = np.isin(g, (3, 4))
        md = {x: set(mod[dn]) for x in [0]}[0]; mu_ = set(mod[up_])
        bm = mod_boot_c(s, g, idxs, B, rng_for(u, 'mb')); bi = iid_boot_c(s, g, B, rng_for(u, 'ib'))
        # within-module and global permutations
        rngp = rng_for(u, 'perm'); cw = np.full(NREP, np.nan); cg = np.full(NREP, np.nan); cov = {k: [] for k in ('wm_mod', 'wm_iid', 'gl_mod', 'gl_iid')}; wid = {k: [] for k in cov}
        gw_store = []
        for r in range(NREP):
            gp = g.copy()
            for ix in idxs:
                if len(ix) > 1: gp[ix] = g[ix][rngp.permutation(len(ix))]
            gg_ = g[rngp.permutation(len(g))]; cw[r] = ordinal(s, gp)[0]; cg[r] = ordinal(s, gg_)[0]; gw_store.append((gp, gg_))
        mw, mg = np.nanmean(cw), np.nanmean(cg)
        for r in range(NCOV):
            gp, gg_ = gw_store[r]
            for tag, gx, tru in (('wm', gp, mw), ('gl', gg_, 0.5)):
                lo, hi = ci(mod_boot_c(s, gx, idxs, BM, rngp)); cov[tag + '_mod'].append((lo <= 0.5 <= hi, lo <= tru <= hi)); wid[tag + '_mod'].append(hi - lo)
                lo, hi = ci(iid_boot_c(s, gx, BM, rngp)); cov[tag + '_iid'].append((lo <= 0.5 <= hi, lo <= tru <= hi)); wid[tag + '_iid'].append(hi - lo)
        for k, v in cov.items():
            v = np.array(v); crow.append(dict(unit=name, null=('within-module label permutation' if k.startswith('wm') else 'global label permutation'), ci_method=('module block bootstrap' if k.endswith('mod') else 'iid gene bootstrap'),
                                              cover_0p5=v[:, 0].mean(), cover_nullmean=v[:, 1].mean(), null_mean=(mw if k.startswith('wm') else mg), mean_width=np.mean(wid[k]), n_null_reps=len(v), B_boot=BM))
        pw = (1 + np.sum(cw >= c0)) / (1 + NREP); pg = (1 + np.sum(cg >= c0)) / (1 + NREP)
        # boundary AUC (L1>L2, L4>L5) under module bootstrap
        def bnd(a_, b_):
            out = np.full(B, np.nan); r_ = rng_for(u, 'mbb', a_)
            for b in range(B):
                ix = np.concatenate([idxs[j] for j in r_.integers(0, len(idxs), len(idxs))]); out[b] = auc(s[ix][g[ix] == a_], s[ix][g[ix] == b_])
            return out
        b12 = bnd(0, 1); b45 = bnd(3, 4)
        rows.append(dict(unit=name, unit_id=u, n_analysed=int(m.sum()), n_modules=len(um), k_modules=50, median_module_size=float(np.median(sz)), max_module_size=int(sz.max()), n_singleton_modules=int((sz == 1).sum()),
                         n_mod_with_KOdown=len(md), n_mod_with_KOup=len(mu_), n_mod_with_both=len(md & mu_), n_KOdown=int(dn.sum()), n_KOup=int(up_.sum()),
                         frac_KOdown_in_top_module=float(max(np.sum(dn & (mod == x)) for x in um) / max(dn.sum(), 1)), frac_KOup_in_top_module=float(max(np.sum(up_ & (mod == x)) for x in um) / max(up_.sum(), 1)),
                         c=c0, c_mod_lo=ci(bm)[0], c_mod_hi=ci(bm)[1], c_iid_lo=ci(bi)[0], c_iid_hi=ci(bi)[1], width_ratio_mod_over_iid=(ci(bm)[1] - ci(bm)[0]) / (ci(bi)[1] - ci(bi)[0]),
                         AUC_L1gtL2=U0.get((0, 1), np.nan), L1L2_mod_lo=ci(b12)[0], L1L2_mod_hi=ci(b12)[1], AUC_L4gtL5=U0.get((3, 4), np.nan), L4L5_mod_lo=ci(b45)[0], L4L5_mod_hi=ci(b45)[1],
                         null_within_mean=mw, null_within_sd=np.nanstd(cw), null_within_q95=np.nanquantile(cw, .95), p_within_module_perm=pw, null_global_mean=mg, null_global_sd=np.nanstd(cg), p_global_perm=pg, NREP=NREP))
        log(u, 'p1b done')
    save(pd.DataFrame(rows), 'p1b_module.tsv'); save(pd.DataFrame(crow), 'p1b_calibration.tsv')


# =============================================================== P2
def p2():
    grp, wrows = [], []
    for u in UL:
        df = load_unit(u); name = UNITS[u][4]; m = amask(df); df = df[m].reset_index(drop=True)
        for i in range(5):
            d = df[df.lvl == i]
            if not len(d): continue
            grp.append(dict(unit=name, level=f'L{i + 1} {LVN[i]}', n=len(d), frac_pos=(d.iso_C > 0).mean(), frac_neg=(d.iso_C < 0).mean(), F_mean=d.F.mean(), F_median=d.F.median(), F_q25=d.F.quantile(.25), F_q75=d.F.quantile(.75),
                            isoC_mean=d.iso_C.mean(), isoC_median=d.iso_C.median(), isoC_q25=d.iso_C.quantile(.25), isoC_q75=d.iso_C.quantile(.75), T_mean=d.T_iso.mean(), T_median=d.T_iso.median(), mean_pos_part_of_T=(d.T_iso.clip(lower=0)).mean()))
        def W(tag, a_i, b_i, col, sub=None, transform=None):
            A_ = df[(df.lvl == a_i)]; B_ = df[(df.lvl == b_i)]
            if sub is not None: A_ = A_[sub(A_)]; B_ = B_[sub(B_)]
            xa = A_[col].to_numpy(float) if transform is None else transform(A_[col].to_numpy(float)); xb = B_[col].to_numpy(float) if transform is None else transform(B_[col].to_numpy(float))
            r = rng_for(u, tag, col); bt = boot_strat(lambda l: auc(l[0], l[1]), [xa, xb], B, r); lo, hi = ci(bt)
            return dict(unit=name, contrast=tag, score=col + ('' if transform is None else '(sign)') + ('' if sub is None else '|subset'), n_hi=len(xa), n_lo=len(xb), AUROC=auc(xa, xb), lo=lo, hi=hi)
        sg = lambda x: np.sign(x)
        # KO-down: L1 vs L2 (expected L1 higher T); F-only oriented P(F_L1>F_L2)
        wrows += [W('L1>L2 (KO-down)', 0, 1, 'iso_C', transform=sg), W('L1>L2 (KO-down)', 0, 1, 'F'), W('L1>L2 (KO-down)', 0, 1, 'T_iso'), W('L1>L2 (KO-down)', 0, 1, 'iso_C'),
                  W('L1>L2 (KO-down)', 0, 1, 'F', sub=lambda d: d.iso_C > 0)]
        # KO-up: L4 vs L5: T orientation P(T_L4>T_L5); F-only oriented P(F_L5>F_L4) (ChIP+ => larger magnitude)
        wrows += [W('L4>L5 (KO-up)', 3, 4, 'iso_C', transform=sg), W('L5>L4 magnitude (KO-up)', 4, 3, 'F'), W('L4>L5 (KO-up)', 3, 4, 'T_iso'), W('L4>L5 (KO-up)', 3, 4, 'iso_C'),
                  W('L5>L4 magnitude (KO-up)', 4, 3, 'F', sub=lambda d: d.iso_C < 0)]
    save(pd.DataFrame(grp), 'p2_groups.tsv'); save(pd.DataFrame(wrows), 'p2_within.tsv')


# =============================================================== P3
def zs(x): x = np.asarray(x, float); sd = np.nanstd(x); return (x - np.nanmean(x)) / (sd if sd > 0 else 1)
COVS = [('log10_abundance', lambda d: np.log10(d.abundance + 1e-6), ''), ('spliced_det', lambda d: d.spliced_det, ''), ('unspliced_det', lambda d: d.unspliced_det, ''), ('abs_log2FC', lambda d: d.log2fc.abs(), ''),
        ('neglog10_padj', lambda d: -np.log10(d.padj.clip(lower=1e-300)), ''), ('F_Zfwd', lambda d: d.F, ''), ('abs_rho_trajectory', lambda d: d.abs_rho_pt, ''),
        ('log10_peak_TSS_dist', lambda d: np.log10(d.dist + 1), 'ChIP+/- is DEFINED by distance: separation by construction'), ('BETA_RP', lambda d: d.RP, 'peak-defined: partly by construction'),
        ('n_peaks_100kb', lambda d: d.n_peaks_100kb, 'peak-defined: partly by construction'), ('iso_C_abs', lambda d: d.iso_C.abs(), 'score-derived')]
CAL2 = 0.75   # supplementary 3-covariate caliper (declared before outcomes were examined)
CAL = 0.5     # matching caliper in pooled-SD units (Euclidean over the 2 matching covariates), FIXED before outcomes were examined
KEEP_CFG = ['baseline', 'ko_mag_0.5', 'ko_mag_1.0', 'ko_sig_0.05', 'ko_sig_0.01', 'occ_1kb', 'occ_0.5kb']
def keep_mask(lab, cfg):   # copy of gb_chip5_sens.keep_mask (existing definitions only)
    c6, l2, pa, ds, ok = lab.c6.to_numpy(str), lab.log2fc.to_numpy(float), lab.padj.to_numpy(float), lab.dist.to_numpy(float), lab.occ_ok.to_numpy(bool)
    resp = np.isin(c6, ['A_act', 'A_rep', 'C_act', 'C_rep']); non = np.isin(c6, ['B', 'D']); a = np.abs(l2); base = np.isin(c6, ['A_act', 'C_act', 'B', 'D', 'C_rep', 'A_rep'])
    with np.errstate(invalid='ignore'):
        if cfg == 'baseline': return base
        if cfg.startswith('ko_mag'): t = float(cfg.split('_')[-1]); return (resp & (a >= t)) | (non & (a <= 0.25))
        if cfg.startswith('ko_sig'): t = float(cfg.split('_')[-1]); return (resp & (pa < t)) | (non & (pa >= 0.10) & (a <= 0.25))
        if cfg.startswith('occ'):
            km = 500. if cfg.endswith('0.5kb') else 1000.; pos = np.isin(c6, ['A_act', 'A_rep', 'B']); neg = np.isin(c6, ['C_act', 'C_rep', 'D'])
            return base & ((pos & (ds <= km)) | (neg & ok & ~(ds <= 10000)))
def p3():
    bal, mat, strat, win = [], [], [], []
    for u in UL:
        df = load_unit(u); name = UNITS[u][4]; df = df.merge(aux_df(u)[['gene', 'abs_rho_pt']], on='gene', how='left', validate='1:1'); m = amask(df); d0 = df[m].reset_index(drop=True)
        for cname, hi_l, lo_l, tr_l, ct_l, sgn in (('L1 vs L2 (KO-down)', 0, 1, 0, 1, +1), ('L5 vs L4 (KO-up)', 4, 3, 4, 3, -1)):
            A_ = d0[d0.lvl == hi_l]; B_ = d0[d0.lvl == lo_l]
            for cn, f, note in COVS:
                xa, xb = f(A_).to_numpy(float), f(B_).to_numpy(float)
                try: p = mannwhitneyu(xa[np.isfinite(xa)], xb[np.isfinite(xb)]).pvalue
                except Exception: p = np.nan
                bal.append(dict(unit=name, comparison=cname, covariate=cn, n_a=int(np.isfinite(xa).sum()), n_b=int(np.isfinite(xb).sum()), median_a=np.nanmedian(xa) if np.isfinite(xa).any() else np.nan, median_b=np.nanmedian(xb) if np.isfinite(xb).any() else np.nan,
                                SMD=cohen_d(xa, xb), AUC_a_gt_b=auc(xa, xb), mwu_p_two=p, note=note))
            bal.append(dict(unit=name, comparison=cname, covariate='local_gene_density', note='UNAVAILABLE: no genomic-coordinate table for the evaluated genes in saved artifacts (not recomputed)'))
            # matched: treated = the smaller group of the pair (L1 for down, L5 for up); controls = L2 / L4; effect oriented so that + = expected direction (L1>L2: T_L1-T_L2; L4>L5: T_L4-T_L5)
            tr = d0[d0.lvl == tr_l].reset_index(drop=True); ct = d0[d0.lvl == ct_l].reset_index(drop=True)
            mf = {'abs_log2FC': lambda d: d.log2fc.abs().to_numpy(float), 'log10_abundance': lambda d: np.log10(d.abundance + 1e-6).to_numpy(float), 'abs_rho_trajectory': lambda d: d.abs_rho_pt.to_numpy(float)}
            for mlabel, mvars, cal in (('PRIMARY |logFC|+abundance', ['abs_log2FC', 'log10_abundance'], CAL), ('SENSITIVITY +trajectory |rho|', ['abs_log2FC', 'log10_abundance', 'abs_rho_trajectory'], CAL2)):
                Xt = np.c_[tuple(mf[v](tr) for v in mvars)]; Xc = np.c_[tuple(mf[v](ct) for v in mvars)]; allx = np.r_[Xt, Xc]; sd = np.nanstd(allx, 0); Xt_, Xc_ = Xt / sd, Xc / sd
                ok_t = np.isfinite(Xt_).all(1); nn = np.full(len(tr), -1); dist = np.full(len(tr), np.inf)
                for i in range(len(tr)):
                    if not ok_t[i]: continue
                    dd = np.sqrt(np.nansum((Xc_ - Xt_[i]) ** 2, 1)); dd[~np.isfinite(Xc_).all(1)] = np.inf; j = int(np.argmin(dd)); nn[i] = j; dist[i] = dd[j]
                sel = (nn >= 0) & (dist <= cal); n_t = len(tr)
                if sel.sum() >= 2:
                    tt = tr.T_iso.to_numpy()[sel]; cc = ct.T_iso.to_numpy()[nn[sel]]; eff = sgn * (tt - cc)      # + = expected direction (T_L1-T_L2 ; T_L4-T_L5)
                    r = rng_for(u, cname, 'mt', mlabel); bt = np.array([eff[r.integers(0, len(eff), len(eff))].mean() for _ in range(B)])
                    try: pw = wilcoxon(eff).pvalue
                    except Exception: pw = np.nan
                    uncond = sgn * (tr.T_iso.to_numpy()[sel].mean() - ct.T_iso.mean()); post = {}
                    for cn, f, note in COVS[:4] + COVS[5:7]: post[f'SMD_after_{cn}'] = cohen_d(f(tr).to_numpy(float)[sel], f(ct).to_numpy(float)[nn[sel]])
                    mat.append(dict(unit=name, comparison=cname, matching=mlabel, caliper_SD=cal, n_treated=n_t, n_matched=int(sel.sum()), n_overlap_fail=int(n_t - sel.sum()), n_unique_controls=int(len(set(nn[sel]))), mean_effect_matched=eff.mean(), eff_lo=ci(bt)[0], eff_hi=ci(bt)[1],
                                    frac_pairs_expected_direction=float((eff > 0).mean()), wilcoxon_p_two=pw, unmatched_effect_same_treated=uncond, **post))
                else:
                    mat.append(dict(unit=name, comparison=cname, matching=mlabel, caliper_SD=cal, n_treated=n_t, n_matched=int(sel.sum()), n_overlap_fail=int(n_t - sel.sum()), note='matched set <2 (overlap failure): not estimable'))
            # tercile strata (pooled terciles over the two groups) on |logFC| and log abundance
            for cn, f in (('abs_log2FC', lambda d: d.log2fc.abs()), ('log10_abundance', lambda d: np.log10(d.abundance + 1e-6))):
                pool = pd.concat([A_, B_]); v = f(pool).to_numpy(float); q = np.nanquantile(v, [1 / 3, 2 / 3]); Ae, Be = (A_, B_) if sgn > 0 else (B_, A_)    # Ae = group EXPECTED to have higher T (L1 / L4)
                sa = np.digitize(f(Ae).to_numpy(float), q); sb = np.digitize(f(Be).to_numpy(float), q)
                ta, tb = Ae.T_iso.to_numpy(), Be.T_iso.to_numpy(); num = den = 0.0; parts = []
                for k in range(3):
                    xa, xb = ta[sa == k], tb[sb == k]; w = len(xa) * len(xb)
                    if w == 0: parts.append(None); continue
                    a_ = auc(xa, xb); num += w * a_; den += w; parts.append((len(xa), len(xb), a_))
                def wauc(l):
                    nu = de = 0.0
                    for k in range(3):
                        xa, xb = l[k], l[3 + k]
                        if len(xa) and len(xb): nu += len(xa) * len(xb) * auc(xa, xb); de += len(xa) * len(xb)
                    return nu / de if de else np.nan
                cells = [ta[sa == k] for k in range(3)] + [tb[sb == k] for k in range(3)]; r = rng_for(u, cname, cn, 'st'); bt = np.array([wauc([c_[r.integers(0, len(c_), len(c_))] if len(c_) else c_ for c_ in cells]) for _ in range(B)])
                row = dict(unit=name, comparison=cname, stratify_on=cn, weighted_AUC_expectedhigher_gt_lower_Tiso=(num / den if den else np.nan), lo=ci(bt)[0], hi=ci(bt)[1])
                for k in range(3): row[f'tercile{k + 1}'] = ('n/a' if parts[k] is None else f'{parts[k][0]}v{parts[k][1]}:AUC={parts[k][2]:.2f}')
                strat.append(row)
        # ChIP-assignment sensitivity (existing configs)
        for cfg in KEEP_CFG:
            km = keep_mask(df, cfg) & np.isfinite(df.T_iso.to_numpy()); d = df[km]; r = rng_for(u, cfg, 'win'); g = [d.T_iso[d.lvl == i].to_numpy() for i in range(5)]
            s_, g_ = flat(g); c_, Ud, n = ordinal(s_, g_); row = dict(unit=name, config=cfg, n=list(map(int, n)), c=c_)
            for tag, i, j in (('L1>L2', 0, 1), ('L4>L5', 3, 4)):
                bt = boot_strat(lambda l: auc(l[0], l[1]), [g[i], g[j]], B, r); row[f'AUC_{tag}'] = auc(g[i], g[j]); row[f'AUC_{tag}_lo'], row[f'AUC_{tag}_hi'] = ci(bt)
            win.append(row)
    save(pd.DataFrame(bal), 'p3_balance.tsv'); save(pd.DataFrame(mat), 'p3_matched.tsv'); save(pd.DataFrame(strat), 'p3_strata.tsv'); save(pd.DataFrame(win), 'p3_chipwindow.tsv')


# =============================================================== P4 / P5
def cls_stats(s, down, up, nul, rng, tag):
    """KO-down vs null, null vs KO-up, KO-down vs KO-up AUROC, one-vs-rest AP, sign recalls & BA; bootstrap CIs stratified by class."""
    d_, n_, u_ = s[down], s[nul], s[up]; ok = lambda x: x[np.isfinite(x)]; d_, n_, u_ = ok(d_), ok(n_), ok(u_)
    def f(l):
        dd, nn, uu = l; rd, ru, ba = (np.mean(dd > 0), np.mean(uu < 0), 0.5 * (np.mean(dd > 0) + np.mean(uu < 0)))
        y = np.r_[np.ones(len(dd)), np.zeros(len(nn) + len(uu))]; yu = np.r_[np.zeros(len(dd) + len(nn)), np.ones(len(uu))]; sc = np.r_[dd, nn, uu]
        return np.array([auc(dd, nn), auc(nn, uu), auc(dd, uu), ap_score(sc, y), ap_score(-sc, yu), rd, ru, ba])
    pt = f([d_, n_, u_]); names = ['AUROC_down_vs_null', 'AUROC_null_vs_up', 'AUROC_down_vs_up', 'AP_down_vs_rest', 'AP_up_vs_rest', 'recall_down', 'recall_up', 'BA']
    out = {'n_down': len(d_), 'n_null': len(n_), 'n_up': len(u_), 'prev_down': len(d_) / max(len(d_) + len(n_) + len(u_), 1), 'prev_up': len(u_) / max(len(d_) + len(n_) + len(u_), 1)}
    if min(len(d_), len(n_), len(u_)) == 0: return {**out, 'note': 'a class is empty'}
    bt = np.array([f([x[rng.integers(0, len(x), len(x))] for x in (d_, n_, u_)]) for _ in range(B)])
    for k, nm in enumerate(names): out[nm] = pt[k]; out[nm + '_lo'], out[nm + '_hi'] = ci(bt[:, k])
    return out
def p4():
    rows = []
    for u in UL:
        df = load_unit(u); name = UNITS[u][4]; lv = df.lvl.to_numpy()
        for scope, dn, nl, upm in (('A_all_responders', np.isin(lv, (0, 1)), lv == 2, np.isin(lv, (3, 4))), ('B_ChIPneg_only(L2,L3,L4)', lv == 1, lv == 2, lv == 3)):
            for col in ('T_iso', 'iso_C'):
                r = cls_stats(df[col].to_numpy(float), dn, upm, nl, rng_for(u, scope, col), col); rows.append(dict(unit=name, scope=scope, score=col, **r))
    save(pd.DataFrame(rows), 'p4_direction.tsv')
def p5():
    rows = []
    for u in UL:
        df = load_unit(u); name = UNITS[u][4]; df = df[amask(df)]; a = df[df.lvl == 3]; b = df[df.lvl == 4]; n4, n5 = len(a), len(b)
        if min(n4, n5) < 2: rows.append(dict(unit=name, n_L4=n4, n_L5=n5, note='undefined contrast (n<2)')); continue
        r = rng_for(u, 'p5'); bt = boot_strat(lambda l: auc(l[0], l[1]), [a.T_iso.to_numpy(), b.T_iso.to_numpy()], B, r); bd = boot_strat(lambda l: l[0].mean() - l[1].mean(), [a.T_iso.to_numpy(), b.T_iso.to_numpy()], B, r)
        se0 = np.sqrt((n4 + n5 + 1) / (12.0 * n4 * n5)); mde = 0.5 + (1.6449 + 0.8416) * se0
        rows.append(dict(unit=name, n_L4=n4, n_L5=n5, AUC_L4_gt_L5=auc(a.T_iso, b.T_iso), lo=ci(bt)[0], hi=ci(bt)[1], mean_diff=a.T_iso.mean() - b.T_iso.mean(), md_lo=ci(bd)[0], md_hi=ci(bd)[1], cohen_d=cohen_d(a.T_iso, b.T_iso),
                         p_onesided_mwu=mwu_one(a.T_iso, b.T_iso), frac_neg_L4=(a.iso_C < 0).mean(), frac_neg_L5=(b.iso_C < 0).mean(), F_med_L4=a.F.median(), F_med_L5=b.F.median(), isoC_med_L4=a.iso_C.median(), isoC_med_L5=b.iso_C.median(),
                         RP_med_L4=a.RP.median(), RP_med_L5=b.RP.median(), dist_med_L4=a.dist.median(), dist_med_L5=b.dist.median(), MDE_AUC_80pct_power_onesided_a05=mde, MDE_note='0.5+(z.95+z.80)*sqrt((n4+n5+1)/(12 n4 n5)); null-variance approximation'))
    save(pd.DataFrame(rows), 'p5_L4L5.tsv')


# =============================================================== P6
S1 = SG + 'grand_signed_bench_2026_10_15/stage1/'
def signed_stats(s, df, hi_pairs=True):
    s = np.asarray(s, float); lv = df.lvl.to_numpy(); m = (lv >= 0) & np.isfinite(s); ss, gg = s[m], lv[m]; r = dict(n=int(m.sum()))
    if len(np.unique(gg)) < 2: return r
    c, U, n = ordinal(ss, gg); z, p, *_ = jt_test(ss, gg); r.update(c=c, JT_z=z, JT_p=p, AUC_L1gtL2=U.get((0, 1), np.nan), AUC_L4gtL5=U.get((3, 4), np.nan), AUC_L2gtL3=U.get((1, 2), np.nan), AUC_L3gtL4=U.get((2, 3), np.nan))
    dn = np.isin(lv, (0, 1)) & np.isfinite(s); up = np.isin(lv, (3, 4)) & np.isfinite(s)
    r['AUROC_down_vs_up'] = auc(s[dn], s[up]); rd, ru, ba = ba_recalls(s, dn, up); r.update(recall_down=rd, recall_up=ru, BA=ba)
    return r
def unsigned_stats(mag, df):
    mag = np.asarray(mag, float); lv = df.lvl.to_numpy(); ok = np.isfinite(mag) & (lv >= 0); resp = np.isin(lv, (0, 1, 3, 4)) & ok; non = (lv == 2) & ok
    return dict(n=int(ok.sum()), AUROC_resp_vs_nonresp=auc(mag[resp], mag[non]), AUROC_ChIPpos_gt_neg_KOdown=auc(mag[(lv == 0) & ok], mag[(lv == 1) & ok]), AUROC_ChIPpos_gt_neg_KOup=auc(mag[(lv == 4) & ok], mag[(lv == 3) & ok]),
                n_L1=int(((lv == 0) & ok).sum()), n_L2=int(((lv == 1) & ok).sum()), n_L4=int(((lv == 3) & ok).sum()), n_L5=int(((lv == 4) & ok).sum()))
def p6():
    srows, urows = [], []
    for u in UL:
        sysn, foc, sd, ab, name, grp = UNITS[u]; df = load_unit(u); sg = np.sign(df.iso_C.to_numpy()); F = df.F.to_numpy()
        sc = {'GB T_iso (iso polarity x F)  [reference]': df.T_iso, 'GB forward d4 composite (Z|d4)': df['Z|d4'], 'GB forward d1 composite (Z|d1)': df['Z|d1'], 'GB iso_C raw continuous': df.iso_C,
              'iso pair-Pearson (iso_PP; shares iso sign per root)': df.iso_PP, 'cell Ms-Mu Pearson (feature cellMsMu)': df.cellMsMu}
        z = None
        for r in ('dc1_min',):
            fn = S1 + f'{sysn}__{r}__R1.npz'
            if os.path.exists(fn):
                z = np.load(fn, allow_pickle=True); srcs = [str(x).upper() for x in z['sources']]
                if foc.upper() in srcs:
                    j = srcs.index(foc.upper()); gmap = pd.Series(np.arange(len(z['genes'])), index=np.char.upper(z['genes'].astype(str))); gmap = gmap[~gmap.index.duplicated()]; gi = gmap.reindex(df.gene_up).to_numpy()
                    for key, nm in (('pear_MsMs', 'cell-level signed Pearson Ms-Ms (stage1)'), ('pear_MsMu', 'cell-level signed Pearson Ms-Mu (stage1)')):
                        v = np.where(np.isfinite(gi), z[key][j][np.nan_to_num(gi).astype(int)], np.nan); v[(df.gene_up == foc.upper()).to_numpy()] = np.nan; sc[nm] = pd.Series(v, index=df.index)
        for nm, v in sc.items(): srows.append(dict(unit=name, kind='SIGNED (own sign)', score=nm, **signed_stats(v.to_numpy(float), df)))
        srows.append(dict(unit=name, kind='SIGNED', score='RegVelo signed J', note='UNAVAILABLE: only unsigned |J| magnitude saved per gene (extmag|RegVelo |J||d4); aggregate signed metrics exist in grand_v3_extsign parts but no per-gene vector'))
        # shared-polarity magnitude benchmark: iso sign x method magnitude  (EVERY row shares the iso sign; only magnitude differs)
        mcols = {('|Pearson|' if c == 'extmag||Pearson||d4' else c.split('|')[1]): c for c in df.columns if c.startswith('extmag|') and c.endswith('|d4')}
        mcols['GB F=|Z|d4|'] = 'Z|d4'
        for nm, c in mcols.items():
            mag = np.abs(df[c].to_numpy(float)); srows.append(dict(unit=name, kind='SHARED-POLARITY magnitude benchmark (iso sign x |method|; NOT the method\'s own sign)', score=nm, **signed_stats(sg * mag, df)))
            urows.append(dict(unit=name, kind='unsigned magnitude', score=nm, **unsigned_stats(mag, df)))
        for nm, v in (('target abundance (mean Ms)', df.abundance), ('spliced detection', df.spliced_det), ('unspliced detection', df.unspliced_det)):
            urows.append(dict(unit=name, kind='target-only covariate', score=nm, **unsigned_stats(v.to_numpy(float), df)))
    save(pd.DataFrame(srows), 'p6_baselines_signed.tsv'); save(pd.DataFrame(urows), 'p6_baselines_unsigned.tsv')


# =============================================================== P7
def p7():
    rows = []
    for u in UL:
        sysn, foc, sd, ab, name, grp = UNITS[u]; df = load_unit(u); z = np.load(IP + f'scores/{sd}/rows.npz', allow_pickle=True); srcs = [str(x).upper() for x in z['sources']]; tg = np.char.upper(z['targets'].astype(str))
        pos = pd.Series(np.arange(len(tg)), index=tg); gi = pos.reindex(df.gene_up).to_numpy(); have = np.isfinite(gi); gx = np.where(have, gi, 0).astype(int); R_ = z['row|iso_C']; selft = z['self_t']
        dg = pd.read_csv(OUT + f'srcdiag_{u}.tsv', sep='\t'); dg = dg.set_index('source'); fi = srcs.index(foc.upper())
        chk = np.nanmax(np.abs(np.where(have & np.isfinite(df.iso_C.to_numpy()), R_[fi][gx] - df.iso_C.to_numpy(), 0))); assert chk < 1e-9, ('focal row != features iso_C', chk)
        per = []
        for i, s_ in enumerate(srcs):
            x = np.where(have, R_[i][gx], np.nan); selfg = np.where(have, selft[i][gx], False).astype(bool); x = np.where(selfg, np.nan, x); Tn = np.sign(x) * df.F.to_numpy()
            # self gene of the source (by name) also removed
            Tn = np.where((df.gene_up == s_).to_numpy(), np.nan, Tn); d = df.copy(); d['T_iso'] = Tn; st = signed_stats(Tn, d)
            fz_ = float(np.nanmean(np.sign(x[np.isfinite(x)]) > 0)) if np.isfinite(x).any() else np.nan
            per.append(dict(unit=name, source=s_, is_focal=(s_ == foc.upper()), frac_pos_sign=fz_, **st, d_abund=(dg.loc[s_, 'mean_Ms'] - dg.loc[foc.upper(), 'mean_Ms']) if s_ in dg.index and dg.loc[s_, 'in_fixture'] else np.nan,
                            d_abs_rho_pt=(abs(dg.loc[s_, 'rho_dc1_min']) - abs(dg.loc[foc.upper(), 'rho_dc1_min'])) if s_ in dg.index and dg.loc[s_, 'in_fixture'] else np.nan))
        P = pd.DataFrame(per)
        for col in ('c', 'AUC_L1gtL2', 'AUC_L2gtL3', 'AUC_L3gtL4', 'AUC_L4gtL5', 'AUROC_down_vs_up', 'BA'):
            fv = P.loc[P.is_focal, col].iloc[0]; alts = P.loc[~P.is_focal, col]
            P.loc[P.is_focal, f'focal_rank_{col}'] = float((alts > fv).sum() + 1); P.loc[P.is_focal, f'alt_median_{col}'] = alts.median(); P.loc[P.is_focal, f'alt_min_{col}'] = alts.min(); P.loc[P.is_focal, f'alt_max_{col}'] = alts.max()
        P['n_alt_finite'] = int(P.loc[~P.is_focal, 'c'].notna().sum()); P['note'] = 'alts = random roster draw (seed 0) per iso_freeze, NOT trajectory-matched; T_alt = sign(alt iso_C) x FOCAL F (polarity-only; alt-specific Z not saved); KO/ChIP truth = FOCAL TF'
        rows.append(P)
    save(pd.concat(rows), 'p7_specificity.tsv')


# =============================================================== P8
def p8():
    import gb_iso_lib as L
    rows = []
    for u in UL:
        sysn, foc, sd, ab, name, grp = UNITS[u]; df = load_unit(u); fn = IP + f'ab/{ab}/ab_sums.npz'
        if not os.path.exists(fn): rows.append(dict(unit=name, note='no A/B arm artifacts')); continue
        z = np.load(fn, allow_pickle=True); srcs = [str(x).upper() for x in z['sources']]; tg = np.char.upper(z['targets'].astype(str)); fi = srcs.index(foc.upper())
        pos = pd.Series(np.arange(len(tg)), index=tg); gi = pos.reindex(df.gene_up).to_numpy(); have = np.isfinite(gi); gx = np.where(have, gi, 0).astype(int)
        rm = lambda a, b: np.where(np.isfinite(a) & np.isfinite(b), (a + b) / 2, np.nan)
        def arm_row(arm, scope):
            seeds = L.SEEDS if arm.startswith('iso') else (0,); pr = []
            for r in L.ROOTS:
                vals = []
                for sd_ in seeds:
                    if scope == 'pooled': N = z[f'{arm}|A|{r}|{sd_}|N'] + z[f'{arm}|B|{r}|{sd_}|N']; D = z[f'{arm}|A|{r}|{sd_}|D'] + z[f'{arm}|B|{r}|{sd_}|D']
                    else: N = z[f'{arm}|{scope}|{r}|{sd_}|N']; D = z[f'{arm}|{scope}|{r}|{sd_}|D']
                    vals.append(L.ratio_C(N, D))
                v = np.stack(vals); pr.append(np.where(np.isfinite(v).all(0), v.mean(0), np.nan))
            return rm(pr[0], pr[1])
        F = df.F.to_numpy(); full = df.iso_C.to_numpy(); variants = {}
        for arm in ('iso_raw', 'fwd_raw'):
            for scope in ('A', 'B', 'pooled'):
                x = arm_row(arm, scope)[fi]; variants[f'{arm}|{scope}'] = np.where(have, x[gx], np.nan)
        variants['fullfixture|iso_C'] = full; variants['fullfixture|iso_C_rootmin'] = df.iso_C_rootmin.to_numpy(); variants['fullfixture|iso_C_rootmax'] = df.iso_C_rootmax.to_numpy()
        selfm = (df.gene_up == foc.upper()).to_numpy()
        common = np.all([np.isfinite(variants[k]) for k in ('iso_raw|A', 'iso_raw|B', 'fullfixture|iso_C_rootmin', 'fullfixture|iso_C_rootmax')], 0) & ~selfm
        lv = df.lvl.to_numpy()
        for k, x in variants.items():
            for supp, msk in (('own support', ~selfm), ('common support (iso A,B,rootmin,rootmax finite)', common)):
                d = df.copy(); T = np.where(msk, np.sign(x) * F, np.nan); d['T_iso'] = T; st = signed_stats(T, d)
                lvl_cov = {f'cov_L{i + 1}': float(np.isfinite(T[lv == i]).mean()) if (lv == i).any() else np.nan for i in range(5)}
                rows.append(dict(unit=name, variant=k, support=supp, n_genes_support=int(np.isfinite(T).sum()), **st, **lvl_cov))
        for a_, b_ in (('iso_raw|A', 'iso_raw|B'), ('iso_raw|A', 'fullfixture|iso_C'), ('iso_raw|B', 'fullfixture|iso_C'), ('fullfixture|iso_C_rootmin', 'fullfixture|iso_C_rootmax'), ('fwd_raw|A', 'fwd_raw|B')):
            m = np.isfinite(variants[a_]) & np.isfinite(variants[b_]) & ~selfm
            rows.append(dict(unit=name, variant=f'SIGN AGREEMENT {a_} vs {b_}', support='pairwise finite', n_genes_support=int(m.sum()), sign_agree=float((np.sign(variants[a_][m]) == np.sign(variants[b_][m])).mean()),
                             sign_agree_responders=float((np.sign(variants[a_][m & (lv >= 0) & (lv != 2)]) == np.sign(variants[b_][m & (lv >= 0) & (lv != 2)])).mean())))
        rows.append(dict(unit=name, variant='LEAVE-ONE-EMBRYO-OUT', note='PENDING: the running iso_full jackknife (results/signed/iso_full_2026_10_15); not run here'))
    save(pd.DataFrame(rows), 'p8_refit.tsv')


STAGES = {'p0': p0, 'aux': aux, 'p1': p1, 'p1b': p1b, 'p2': p2, 'p3': p3, 'p4': p4, 'p5': p5, 'p6': p6, 'p7': p7, 'p8': p8}


def aux_df(u): return pd.read_csv(OUT + f'aux_{u}.tsv', sep='\t')


if __name__ == '__main__':
    STAGES[STAGE]()
