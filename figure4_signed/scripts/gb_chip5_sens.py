# ChIP label-quality sensitivity of the signed scores
import sys, os, json, numpy as np, pandas as pd
from pathlib import Path
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch'), (__import__("os").environ["GEOBRIDGE_ROOT"] + '/pairspace')]
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
DUMP = R + 'results/signed/grand_v3d_dumpT_2026_10_15/parts/T_%s.tsv'
UNITS = ['CARDIACV_TBX5', 'CARDIACV_NKX2-5', 'GATA1_GATA1', 'KLF1_KLF1', 'KLF1_B_KLF1', 'PANC_RFX6', 'BONEMARROW_RUNX1', 'BM_GATA1_ACUTE_GATA1']
GB = ['Z|d1', 'Z|d4']
COMP = {n: f'extmag|{n}|d4' for n in ['GENIE3', 'GRNBoost2', 'PIDC', 'DeepSEM', 'RegDiffusion']}
COMP['|Pearson|(d4)'] = 'extmag||Pearson||d4'; COMP['absPear|d_Pear'] = 'absPear|d_Pear'
SCORES = GB + list(COMP.values())
CONFIGS = ['baseline', 'ko_mag_0.5', 'ko_mag_1.0', 'ko_sig_0.05', 'ko_sig_0.01', 'occ_1kb', 'occ_0.5kb']
LEV = {'L1': ['A_act'], 'L2': ['C_act'], 'L3': ['B', 'D'], 'L4': ['C_rep'], 'L5': ['A_rep']}

# labels (score-blind)
def labels(unit):
    import gb_fig3d_magnitude as F, gb_layer_lattice as L, gb_lattice_extra as LE
    from gb_fig3d_magnitude import up, uniq, effect
    import gb_fourclass_signed_k0 as K
    d = K.build(unit); f, es, tf, bsy, grp = K.U[unit]
    z = np.load(f % unit, allow_pickle=True); genes = np.asarray(z['genes']).astype(str); n = len(genes)
    assert (up(genes) == d['gg']).all()
    lf = pd.Series(up(genes)).map(effect(es, tf)).to_numpy(float); pa = np.full(n, np.nan); pnote = 'UNAVAILABLE (no padj in cited loader)'
    def padj_from(sym, val): return pd.Series(up(genes)).map(uniq(sym, val)).to_numpy(float)
    if unit == 'PANC_RFX6':
        tr, _ = L.truth_panc(genes); x = pd.read_csv(R + 'real_1_Pancreas/external_validation/aldous_rfx6/results/aldous_PPs_WT_vs_KO_DE.tsv', sep='\t')
        pa = padj_from(x.gene_name.fillna(''), x.padj); pnote = 'aldous DESeq2 padj'
    elif unit == 'GATA1_GATA1':
        tr, _ = L.truth_gata1(genes)
    elif unit == 'KLF1_KLF1':
        tr, _ = L.truth_klf1(genes)
        D = pd.read_csv(R + 'results/klf1_heldout/truth/GSE94351_gene_exp.diff.gz', sep='\t'); D['s1'], D['s2'] = D.sample_1.astype(str).str.upper(), D.sample_2.astype(str).str.upper()
        D = D[((D.s1 == 'Q2') & (D.s2 == 'Q3')) | ((D.s1 == 'Q3') & (D.s2 == 'Q2'))]; D = D[(D.status == 'OK') & ~D.gene.astype(str).str.contains(',') & (D.gene.astype(str) != '-')]
        pa = padj_from(D.gene, D.q_value); pnote = 'cuffdiff q_value (FDR; baseline "significant" = q<.05)'
    elif unit == 'KLF1_B_KLF1':
        tr, _ = LE.truth_klf1_b('KLF1', genes); x = pd.read_csv(LE.KLF1_DE, sep='\t'); pa = padj_from(x.gene, x.padj); pnote = 'GSE33979 DESeq2 padj'
    elif unit.startswith('CARDIACV'):
        tr, _ = LE.truth_cardiac(tf, genes); x = pd.read_csv(R + f"results/cardiac_truth_v1/de_{'Tbx5' if tf == 'TBX5' else 'Nkx2-5'}KO_vs_WT.tsv", sep='\t'); pa = padj_from(x.gene, x.padj); pnote = 'pydeseq2 padj'
    elif unit == 'BONEMARROW_RUNX1':
        tr, _ = L.truth_bm('RUNX1', genes)
    elif unit == 'BM_GATA1_ACUTE_GATA1':
        tr, _ = LE.truth_bm_gata1_acute('GATA1', genes)
        x = pd.read_csv(R + 'data/gse274110_proseq/GSE274110_Sankaran003_gga_deseq_24h_GATA1vs24h_AAVS1_results_table.txt.gz', sep='\t'); x = x[x.Symbol.astype(str) != 'NA']
        pa = padj_from(x.Symbol, x['padj_24h_GATA1vs24h_AAVS1']); pnote = 'GSE274110 DESeq2 padj'
    occ, resp = np.asarray(tr['occ']['prim'], bool), np.asarray(tr['resp'], bool)
    rc = np.where(occ & resp, 'A', np.where(occ, 'B', np.where(resp, 'C', 'D')))
    cls = d['cls']; m = np.isin(cls, list('ABCD'))
    par = dict(unit=unit, n_genes=n, n_cls_ABCD=int(m.sum()), cls_reconstruction_mismatch=int((rc[m] != cls[m]).sum()), c6_vs_dump_mismatch=None)
    dist = np.asarray(tr['occ_dist'], float) if tr.get('occ_dist') is not None else np.full(n, np.nan)
    lab = pd.DataFrame(dict(gene=d['gg'], cls=cls, c6=d['c6'], log2fc=lf, padj=pa, dist=dist, occ_ok=np.asarray(tr['occ_ok'], bool), recon_cls=rc))
    return lab, par, pnote

# filters
def keep_mask(lab, cfg):
    c6, l2, pa, ds, ok = lab.c6.to_numpy(str), lab.log2fc.to_numpy(float), lab.padj.to_numpy(float), lab.dist.to_numpy(float), lab.occ_ok.to_numpy(bool)
    resp = np.isin(c6, ['A_act', 'A_rep', 'C_act', 'C_rep']); non = np.isin(c6, ['B', 'D']); a = np.abs(l2)
    with np.errstate(invalid='ignore'):
        if cfg == 'baseline': return np.isin(c6, list(sum(LEV.values(), [])))
        if cfg.startswith('ko_mag'):
            t = float(cfg.split('_')[-1]); return (resp & (a >= t)) | (non & (a <= 0.25))
        if cfg.startswith('ko_sig'):
            t = float(cfg.split('_')[-1]); return (resp & (pa < t)) | (non & (pa >= 0.10) & (a <= 0.25))
        if cfg.startswith('occ'):
            km = 500. if cfg.endswith('0.5kb') else 1000.; pos = np.isin(c6, ['A_act', 'A_rep', 'B']); neg = np.isin(c6, ['C_act', 'C_rep', 'D'])
            return np.isin(c6, list(sum(LEV.values(), []))) & ((pos & (ds <= km)) | (neg & ok & ~(ds <= 10000)))
def available(lab, cfg):
    if cfg == 'baseline': return True, ''
    if cfg.startswith('ko_mag'): return (lab.log2fc.notna().any(), 'no log2FC')
    if cfg.startswith('ko_sig'): return (lab.padj.notna().any() and lab.log2fc.notna().any(), 'no padj')
    return (lab.dist.notna().any() and lab.occ_ok.any(), 'no peak-TSS distance')

# stats
def auc(x, y):   # P(x>y)+.5P(=)
    if len(x) < 1 or len(y) < 1: return np.nan
    ys = np.sort(y); return (np.searchsorted(ys, x, 'left') + 0.5 * (np.searchsorted(ys, x, 'right') - np.searchsorted(ys, x, 'left'))).sum() / (len(x) * len(y))
def mwu_p(x, y):
    from scipy.stats import mannwhitneyu
    if len(x) < 2 or len(y) < 2: return np.nan
    return float(mannwhitneyu(x, y, alternative='greater', method='asymptotic').pvalue)
PAIRS = {'thetaA': ('L1', 'L2'), 'thetaR': ('L4', 'L5'), 'adj_L2_L3': ('L2', 'L3'), 'adj_L3_L4': ('L3', 'L4')}
def thetas(T, idx):   # idx: level->index array
    return {k: auc(T[idx[a]], T[idx[b]]) - 0.5 for k, (a, b) in PAIRS.items()}

def run_unit(unit, out, B):
    parts = Path(out) / 'parts'; parts.mkdir(parents=True, exist_ok=True)
    lab, par, pnote = labels(unit)
    dump = pd.read_csv(DUMP % unit, sep='\t'); assert len(dump) == len(lab) and (dump.gene.to_numpy(str) == lab.gene.to_numpy(str)).all(), 'dump/gene order mismatch'
    par['c6_vs_dump_mismatch'] = int((dump.cls.to_numpy(str) != lab.c6.to_numpy(str)).sum()); par['dup_rows'] = int(lab.gene.duplicated().sum()); par['padj_source'] = pnote
    print(unit, par, flush=True)
    if par['c6_vs_dump_mismatch'] or par['cls_reconstruction_mismatch']: print('PARITY_WARNING', flush=True)
    lab.to_csv(parts / f'{unit}_labels.tsv', sep='\t', index=False)
    S = {s: dump[s].to_numpy(float) for s in SCORES}; common = np.all([np.isfinite(v) for v in S.values()], 0) & ~lab.gene.duplicated(keep=False).to_numpy()
    par['n_common'] = int(common.sum()); c6 = lab.c6.to_numpy(str)
    CL, EF, PR, CLM, AV = [], [], [], [], []; rng_seed = 20261015
    for cfg in CONFIGS:
        ok, why = available(lab, cfg); AV.append(dict(unit=unit, config=cfg, available=ok, reason='' if ok else why, padj_source=pnote if cfg.startswith('ko_sig') else ''))
        if not ok: continue
        keep = keep_mask(lab, cfg) & common
        idx = {L_: np.flatnonzero(keep & np.isin(c6, cs)) for L_, cs in LEV.items()}
        CL.append(dict(unit=unit, config=cfg, n_common=int(common.sum()), **{f'n_{k}': len(v) for k, v in idx.items()}, **{f'n_{c}': int((keep & (c6 == c)).sum()) for c in ['A_act', 'C_act', 'B', 'D', 'C_rep', 'A_rep']}))
        rng = np.random.default_rng([rng_seed, UNITS.index(unit), CONFIGS.index(cfg)])
        bidx = [{k: (rng.choice(v, len(v)) if len(v) else v) for k, v in idx.items()} for _ in range(B)]
        pt, bs = {}, {}
        for s in SCORES:
            T = S[s]; pt[s] = thetas(T, idx); bs[s] = [thetas(T, b) for b in bidx]
            row = dict(unit=unit, config=cfg, score=s, **{f'n_{k}': len(v) for k, v in idx.items()})
            for k in PAIRS:
                v = np.array([b[k] for b in bs[s]]); row[k] = pt[s][k]; row[k + '_lo'], row[k + '_hi'] = (np.nanpercentile(v, [2.5, 97.5]) if np.isfinite(v).sum() > 20 else (np.nan, np.nan))
            row['p_thetaA_unadj'] = mwu_p(S[s][idx['L1']], S[s][idx['L2']]); row['p_thetaR_unadj'] = mwu_p(S[s][idx['L4']], S[s][idx['L5']])
            row['p_adjL2L3_unadj'] = mwu_p(S[s][idx['L2']], S[s][idx['L3']]); row['p_adjL3L4_unadj'] = mwu_p(S[s][idx['L3']], S[s][idx['L4']])
            EF.append(row)
        for g in GB:
            for c, cs in COMP.items():
                for k in ('thetaA', 'thetaR'):
                    v = np.array([bs[g][i][k] - bs[cs][i][k] for i in range(B)]); lo, hi = np.nanpercentile(v, [2.5, 97.5]) if np.isfinite(v).sum() > 20 else (np.nan, np.nan)
                    PR.append(dict(unit=unit, config=cfg, gb=g, comparator=c, theta=k, gb_theta=pt[g][k], comp_theta=pt[cs][k], delta=pt[g][k] - pt[cs][k], lo=lo, hi=hi))
    for cfg in CONFIGS:   # claims: unit x direction x annotation x config unadjusted one-sided MWU p
        pass
    ef = pd.DataFrame(EF); cl = []
    for g in GB:
        for k, pc in (('thetaA', 'p_thetaA_unadj'), ('thetaR', 'p_thetaR_unadj')):
            for cfg in CONFIGS:
                r = ef[(ef.config == cfg) & (ef.score == g)]
                cl.append(dict(unit=unit, annotation=g, theta=k, config=cfg, p=float(r[pc].iloc[0]) if len(r) else np.nan, available=bool(len(r))))
    pd.DataFrame(CL).to_csv(parts / f'{unit}_classes.tsv', sep='\t', index=False); ef.to_csv(parts / f'{unit}_effects.tsv', sep='\t', index=False)
    pd.DataFrame(PR).to_csv(parts / f'{unit}_paired.tsv', sep='\t', index=False); pd.DataFrame(cl).to_csv(parts / f'{unit}_claims.tsv', sep='\t', index=False)
    pd.DataFrame(AV).to_csv(parts / f'{unit}_availability.tsv', sep='\t', index=False); json.dump(par, open(parts / f'{unit}_parity.json', 'w'))
    print(unit, 'DONE', flush=True)

def holm(p):
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n); q[o] = np.minimum(1, np.maximum.accumulate(p[o] * (n - np.arange(n)))); return q

def agg(out):
    P = Path(out) / 'parts'; cat = lambda suf: pd.concat([pd.read_csv(f, sep='\t') for u in UNITS if (f := P / f'{u}_{suf}.tsv').exists()], ignore_index=True)
    cl, ef, pr, av = cat('classes'), cat('effects'), cat('paired'), cat('availability'); c = cat('claims')
    cl.to_csv(Path(out) / 'classes.tsv', sep='\t', index=False); ef.to_csv(Path(out) / 'effects.tsv', sep='\t', index=False); pr.to_csv(Path(out) / 'paired.tsv', sep='\t', index=False); av.to_csv(Path(out) / 'availability.tsv', sep='\t', index=False)
    rows = []
    for u in UNITS:
        for g in GB:
            for k in ('thetaA', 'thetaR'):
                x = c[(c.unit == u) & (c.annotation == g) & (c.theta == k)] if len(c) else c
                bp = x[x.config == 'baseline'].p; base = float(bp.iloc[0]) if len(bp) and np.isfinite(bp.iloc[0]) else np.nan
                full = len(x) == len(CONFIGS) and x.available.all() and x.p.notna().all()
                rows.append(dict(unit=u, annotation=g, theta=k, baseline_p_unadj=base, robust_maxp_unadj=float(x.p.max()) if full else np.nan, robust_full=bool(full), n_configs_available=int(x.available.sum()) if len(x) else 0))
    F = pd.DataFrame(rows); F['baseline_holm32'] = holm(F.baseline_p_unadj.fillna(1).to_numpy()); F['robust_holm32'] = holm(F.robust_maxp_unadj.fillna(1).to_numpy())
    F.to_csv(Path(out) / 'family.tsv', sep='\t', index=False)
    par = [json.load(open(f)) for u in UNITS if (f := P / f'{u}_parity.json').exists()]
    with open(Path(out) / 'run.md', 'w') as fh:
        fh.write('# ChIP label-quality sensitivity (preliminary); p-values nominal/unadjusted unless column says holm32\n\n## parity\n' + pd.DataFrame(par).to_markdown(index=False) + '\n\n## availability\n' + av.to_markdown(index=False) + '\n\n## classes\n' + cl.to_markdown(index=False) + '\n\n## family (32 claims)\n' + F.round(4).to_markdown(index=False) + '\n')
        fh.write('\n## effects (GB annotations and comparators)\n' + ef.round(4).to_markdown(index=False) + '\n\n## paired GB-comparator\n' + pr.round(4).to_markdown(index=False) + '\n')
    print('agg done')

if __name__ == '__main__':
    if sys.argv[1] == 'unit': run_unit(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 2000)
    else: agg(sys.argv[2])
