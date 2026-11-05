"""BETA regulatory potential vs frozen scores (spec internal design note (not shipped)). EVALUATION ONLY; scores are FROZEN dumped T; peaks/TSS = the existing truth builders'. PRELIMINARY.
usage: gb_beta_rp.py unit UNIT OUTDIR [B]   -> OUTDIR/parts/<UNIT>_{rp,spearman,paired,parity}.*
       gb_beta_rp.py agg OUTDIR             -> rp_<unit>.tsv copies, spearman.tsv, paired.tsv, parity.log, run.md
RP_g = sum_{peaks within 100kb of TSS} exp(-(0.5+4*d)), d=|pos-TSS|/100kb. Several TSS per gene -> max over TSS (BETA is single-TSS; declared choice)."""
import sys, os, gzip, json, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata, spearmanr
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
sys.path[:0] = [R + 'scratch', R + 'pairspace']
DUMP = R + 'results/signed/grand_v3d_dumpT_2026_10_15/parts/T_%s.tsv'
LAB = R + 'results/signed/chip5_sens_2026_10_15/parts/%s_labels.tsv'
UNITS = ['CARDIACV_TBX5', 'CARDIACV_NKX2-5', 'GATA1_GATA1', 'KLF1_KLF1', 'KLF1_B_KLF1', 'PANC_RFX6']
GB = 'Z|d4'
COMP = {n: f'extmag|{n}|d4' for n in ['GENIE3', 'GRNBoost2', 'PIDC', 'DeepSEM', 'RegDiffusion']}
COMP['|Pearson|'] = 'extmag||Pearson||d4'
W = 100_000

def up(s): return pd.Series(np.asarray(s).astype(str)).str.upper().to_numpy()

# ---------------------------------------------------------------- per-unit sources: returns sites{UPPER:[(chrom,pos)]}, peaks{chrom:sorted int array}, dist_parity{UPPER:float}, parity_rule
def src_gata1():
    import gb_gata1_truth_v2 as TV
    pk = pd.read_csv(TV.PK, sep='\t', header=None); pk = pk[~TV.malformed_summit(pk[1], pk[2], pk[9])]
    peaks = pd.DataFrame(dict(chrom=pk[0].astype(str), summit=pk[1].astype(np.int64) + pk[9].astype(np.int64), signal=pk[6].astype(float)))
    tss = TV.load_tss(); F = TV.gene_features(tss, peaks)
    return tss, peaks.groupby('chrom').summit.apply(lambda s: np.sort(s.to_numpy())).to_dict(), dict(zip(F.gene, F.nearest_summit_tss_dist)), 'ENCFF676TDJ summit; GENCODE vM25 all-transcript TSS; plain nearest (TV.gene_features)'

def src_klf1():
    import gb_gata1_truth_v2 as TV, gb_layer_lattice as L
    rows = []
    with gzip.open(L.KLF1_CHIP_BED, 'rt') as fh:   # verbatim gb_klf1_eval.main_chip
        for ln in fh:
            if ln.startswith(('#', 'track', 'browser')): continue
            f = ln.rstrip('\n').split('\t'); st, en = int(f[1]), int(f[2])
            summit = st + int(f[9]) if len(f) >= 10 and f[9] not in ('.', '-1') else (st + en) // 2
            sig = float(f[6]) if len(f) >= 7 else float(f[4]) if len(f) >= 5 else 1.0
            rows.append(dict(chrom=f[0], summit=summit, signal=sig))
    pk = pd.DataFrame(rows); tss = TV.load_tss(); F = TV.gene_features(tss, pk).set_index('gene')
    F.index = F.index.str.upper(); F = F[~F.index.duplicated(keep=False)]
    return tss, pk.groupby('chrom').summit.apply(lambda s: np.sort(s.to_numpy())).to_dict(), F.nearest_summit_tss_dist.to_dict(), 'GSE210779 bed (midpoint, no col10); GENCODE vM25 all-transcript TSS; plain nearest'

def src_klf1_b():
    import gb_layer_lattice as L, gb_lattice_extra as LE
    tss = LE.canonical_tss_mm9(pd.read_csv(LE.REFGENE_MM9, sep='\t', header=None)); pk = LE.read_peaks(LE.EKLF_BED, score_col=3, sep=r'\s+')
    F = LE.eklf_gene_features(tss, pk)
    return {r.sym: [(r.chrom, int(r.tss0))] for r in tss.itertuples()}, pk.groupby('chrom').mid.apply(lambda s: np.sort(s.to_numpy())).to_dict(), F.nearest_summit_tss_dist.to_dict(), 'GSE20478 EKLF mm9 midpoint; canonical refGene mm9 TSS; parity = unique-nearest assignment within 10kb (eklf_gene_features)'

def src_rfx6():
    rc = json.load(open(R + 'results/endpoints_v2/rfx6_occupancy.receipt.json'))
    pk = pd.read_csv(rc['peaks'], sep='\t', comment='#'); pk = pk[pk['FDR(%)'] <= 5].copy(); pk['summit_pos'] = pk.start - 1 + pk.summit   # macs_summit_1b
    rg = pd.read_csv(rc['refgene'], sep='\t', header=None); rg = rg[~rg[2].astype(str).str.contains('_')].copy()
    rg['len'] = rg[5] - rg[4]; rg['tss'] = np.where(rg[3] == '+', rg[4] + 1, rg[5])                                                      # refgene_tss_1b
    rg = rg.sort_values([12, 'len', 4, 1], ascending=[True, False, True, True]); canon = rg.groupby(12).head(1)
    dist = {}; tb = {c: g for c, g in canon.groupby(2)}   # assign_summits verbatim: co-located TSS grouped, exact tie between different symbols -> unassigned
    for c, g in pk.groupby('chr'):
        T_ = tb.get(c)
        if T_ is None: continue
        grp = T_.groupby('tss')[12].apply(lambda x: sorted(set(x))); tpos = grp.index.to_numpy(); tsyms = grp.to_numpy()
        for spos in g.summit_pos.to_numpy():
            i = np.searchsorted(tpos, spos); cand = [j for j in (i - 1, i) if 0 <= j < len(tpos)]
            d = np.array([abs(tpos[j] - spos) for j in cand]); best = d.min()
            syms = {x for j, dj in zip(cand, d) if dj == best for x in tsyms[j]}
            if len(syms) > 1: continue
            s = syms.pop(); dist[s] = min(dist.get(s, np.inf), float(best))
    sites = {}
    for r in canon.itertuples(index=False): sites.setdefault(str(r[12]).upper(), []).append((r[2], int(r.tss)))
    peaks = pk.groupby('chr').summit_pos.apply(lambda s: np.sort(s.to_numpy().astype(np.int64))).to_dict()
    dist = {k.upper(): v for k, v in dist.items()}
    return sites, peaks, {s: dist.get(s, np.inf) for s in sites}, 'GSE62844 HA-Rfx6 FDR<=5 1-based summit; canonical refGene mm9 longest-span TSS (1-based); parity = unique-nearest assignment (no window)'

def src_cardiac(tf):
    import cardiac_truth_build as C
    tss, _ = C.tss_table(); fn = C.FOOT['Tbx5' if tf == 'TBX5' else 'Nkx2-5']
    lines = list(gzip.open(f'{C.FEAS}/chipexo_footprints/{fn}', 'rt'))
    b, _ = C.bound_genes(tss, lines)
    mids = {}
    for ln in lines:
        c, s, e = ln.split()[:3]; mids.setdefault(c, []).append((int(s) + int(e)) // 2)
    sites = {}
    for c, recs in tss.items():
        for p, sym in recs: sites.setdefault(sym.upper(), []).append((c, int(p)))
    dist = {}
    for g, (_, d) in b.items(): dist[g.upper()] = min(dist.get(g.upper(), np.inf), d)
    return sites, {c: np.sort(np.asarray(v, np.int64)) for c, v in mids.items()}, dist, 'ChIP-exo WT-CM footprint midpoint; refGene mm9 all-record TSS; plain nearest over all midpoints (cardiac_truth_build.bound_genes)'

def source(unit):
    if unit == 'GATA1_GATA1': return src_gata1()
    if unit == 'KLF1_KLF1': return src_klf1()
    if unit == 'KLF1_B_KLF1': return src_klf1_b()
    if unit == 'PANC_RFX6': return src_rfx6()
    return src_cardiac(unit.split('_')[1])

# ---------------------------------------------------------------- RP
def rp_gene(sites, peaks):
    """max over TSS of sum exp(-(0.5+4*|p-TSS|/100kb)) over peaks within 100kb; also n peaks (at the maximising TSS) and nearest-peak distance over ALL peaks on the chromosome"""
    best, nb, dmin = 0.0, 0, np.inf
    for c, t in sites:
        a = peaks.get(c)
        if a is None or not len(a): continue
        lo, hi = np.searchsorted(a, t - W, 'left'), np.searchsorted(a, t + W, 'right')
        if hi > lo:
            v = np.exp(-(0.5 + 4 * np.abs(a[lo:hi] - t) / W)).sum()
            if v > best: best, nb = float(v), int(hi - lo)
        j = np.searchsorted(a, t); dmin = min(dmin, *(abs(int(a[k]) - t) for k in (j - 1, j) if 0 <= k < len(a)), np.inf)
    return best, nb, dmin

def selftest():
    a = {'c': np.array([1000, 51_000, 200_000])}   # 200000 is >100kb from both TSS
    v, n, d = rp_gene([('c', 1000)], a); assert n == 2 and abs(v - (np.exp(-0.5) + np.exp(-2.5))) < 1e-12 and d == 0
    v, n, d = rp_gene([('c', 51_000)], a); assert n == 2 and abs(v - (np.exp(-0.5) + np.exp(-(0.5 + 2)))) < 1e-12
    assert rp_gene([('x', 5)], a)[0] == 0.0
    print('selftest PASS')

# ---------------------------------------------------------------- stats
def rho(x, y): return float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])

def run_unit(unit, out, B):
    selftest(); parts = Path(out) / 'parts'; parts.mkdir(parents=True, exist_ok=True)
    sites, peaks, dpar, rule = source(unit)
    s2, d2 = {}, {}
    for k, v in sites.items(): s2.setdefault(str(k).upper(), []).extend(v)
    for k, v in dpar.items(): d2[str(k).upper()] = min(d2.get(str(k).upper(), np.inf), v)   # exact-case collisions: union of TSS / min distance
    sites, dpar = s2, d2
    lab = pd.read_csv(LAB % unit, sep='\t'); dump = pd.read_csv(DUMP % unit, sep='\t')
    assert len(lab) == len(dump) and (lab.gene.astype(str).to_numpy() == dump.gene.astype(str).to_numpy()).all(), 'labels/dump order mismatch'
    G = up(lab.gene); rows = []
    for g in G:
        s = sites.get(g)
        if s is None: rows.append((np.nan, 0, np.nan, np.nan)); continue
        v, n, dm = rp_gene(s, peaks); dp = dpar.get(g, np.nan); rows.append((v, n, dm, dp))
    rows = np.array(rows, float)
    df = pd.DataFrame(dict(gene=lab.gene, c6=lab.c6, dist_label=lab.dist, dist_recomp_builder=rows[:, 3], dist_recomp_plain=rows[:, 2], RP=rows[:, 0], n_peaks_100kb=rows[:, 1]))
    df.to_csv(parts / f'rp_{unit}.tsv', sep='\t', index=False)
    # ---- parity
    P = dict(unit=unit, rule=rule, n_genes=len(df), n_mapped_tss=int(np.isfinite(df.RP).sum()))
    both = df.dist_label.notna() & df.dist_recomp_builder.notna() & np.isfinite(df.dist_recomp_builder)
    mm = both & (df.dist_label != df.dist_recomp_builder)
    P.update(n_dist_compared=int(both.sum()), n_dist_mismatch=int(mm.sum()),
             n_label_nan_recomp_finite=int((df.dist_label.isna() & np.isfinite(df.dist_recomp_builder)).sum()), n_label_finite_recomp_nonfinite=int((df.dist_label.notna() & ~np.isfinite(df.dist_recomp_builder)).sum()),
             n_label_le1kb=int((df.dist_label <= 1000).sum()), n_label_le1kb_RP_not_positive=int(((df.dist_label <= 1000) & ~(df.RP > 0)).sum()),
             n_recomp_le1kb=int((df.dist_recomp_builder <= 1000).sum()), n_recomp_le1kb_RP_not_positive=int(((df.dist_recomp_builder <= 1000) & ~(df.RP > 0)).sum()),
             n_plain_vs_label_mismatch=int((df.dist_label.notna() & np.isfinite(df.dist_recomp_plain) & (df.dist_label != df.dist_recomp_plain)).sum()))
    P['mismatch_examples'] = df[mm].head(8)[['gene', 'dist_label', 'dist_recomp_builder', 'dist_recomp_plain']].to_dict('records')
    json.dump(P, open(parts / f'parity_{unit}.json', 'w'), default=str, indent=1); print(json.dumps(P, default=str), flush=True)
    # ---- analyses
    names = [GB] + list(COMP.values()); disp = ['GB(Z|d4)'] + list(COMP.keys())
    S = {n: dump[n].to_numpy(float) for n in names}; cls = lab.c6.to_numpy(str); RP = df.RP.to_numpy(float)
    ok0 = np.isfinite(RP) & ~lab.gene.duplicated(keep=False).to_numpy() & np.all([np.isfinite(v) for v in S.values()], 0)
    SP, PR = [], []
    for ui, (dirn, cs, sgn) in enumerate([('KO-down', ['A_act', 'C_act'], 1.0), ('KO-up', ['A_rep', 'C_rep'], -1.0)]):
        idx = np.flatnonzero(ok0 & np.isin(cls, cs)); n = len(idx)
        if n < 10:
            SP.append(dict(unit=unit, direction=dirn, n=n, status='UNAVAILABLE n<10')); continue
        x = {d: sgn * S[nm][idx] for d, nm in zip(disp, names)}; y = RP[idx]
        r = {}
        for d in disp:
            rr, p = spearmanr(x[d], y); r[d] = rr
            SP.append(dict(unit=unit, direction=dirn, method=d, n=n, n_RP_pos=int((y > 0).sum()), rho=rr, p=p, rho_rawscore=spearmanr(S[names[disp.index(d)]][idx], y)[0], status='ok'))
        order = pd.Series(r).rank(ascending=False, method='min')
        for q in SP[-len(disp):]: q['gb_rank_among_methods'] = int(order['GB(Z|d4)']); q['rank_of_method'] = int(order[q['method']])
        rng = np.random.default_rng([20261015, UNITS.index(unit), ui]); rx = {d: rankdata(x[d]) for d in disp}; ry = rankdata(y)
        bd = {d: np.empty(B) for d in disp[1:]}
        for b in range(B):
            i = rng.integers(0, n, n); ryb = rankdata(y[i]); rb = {d: np.corrcoef(rankdata(x[d][i]), ryb)[0, 1] for d in disp}
            for d in disp[1:]: bd[d][b] = rb[disp[0]] - rb[d]
        for d in disp[1:]:
            v = bd[d]; lo, hi = np.nanpercentile(v, [2.5, 97.5]) if np.isfinite(v).sum() > 20 else (np.nan, np.nan)
            PR.append(dict(unit=unit, direction=dirn, comparator=d, n=n, rho_GB=r[disp[0]], rho_comp=r[d], delta=r[disp[0]] - r[d], lo=lo, hi=hi, B=B))
    pd.DataFrame(SP).to_csv(parts / f'spearman_{unit}.tsv', sep='\t', index=False); pd.DataFrame(PR).to_csv(parts / f'paired_{unit}.tsv', sep='\t', index=False); print(unit, 'DONE', flush=True)

def agg(out):
    P = Path(out) / 'parts'; cat = lambda pre: pd.concat([pd.read_csv(f, sep='\t') for u in UNITS if (f := P / f'{pre}_{u}.tsv').exists()], ignore_index=True)
    sp, pr, rp = cat('spearman'), cat('paired'), cat('rp'); sp.to_csv(Path(out) / 'spearman.tsv', sep='\t', index=False); pr.to_csv(Path(out) / 'paired.tsv', sep='\t', index=False)
    for u in UNITS:
        if (P / f'rp_{u}.tsv').exists(): (Path(out) / f'rp_{u}.tsv').write_bytes((P / f'rp_{u}.tsv').read_bytes())
    par = [json.load(open(P / f'parity_{u}.json')) for u in UNITS if (P / f'parity_{u}.json').exists()]
    missing = [u for u in UNITS if not (P / f'parity_{u}.json').exists()]
    with open(Path(out) / 'parity.log', 'w') as fh:
        for p in par: fh.write(json.dumps(p, default=str) + '\n')
        fh.write('UNAVAILABLE_UNITS ' + json.dumps(missing) + '\n')
    flat = pd.DataFrame([{k: v for k, v in p.items() if k != 'mismatch_examples'} for p in par])
    with open(Path(out) / 'run.md', 'w') as fh:
        fh.write('# BETA RP vs frozen scores - PRELIMINARY (spec BETA_RP_SPEC_2026_10_15). RP=sum exp(-(0.5+4d)), 100kb, max over TSS per gene; KO-up uses -score for every method (literal spec); rho_rawscore = unflipped.\n\n## parity\n' + flat.to_markdown(index=False) + f'\n\nunavailable units: {missing}\n\n## spearman\n' + sp.round(4).to_markdown(index=False) + '\n\n## paired GB - comparator (bootstrap 95% CI)\n' + pr.round(4).to_markdown(index=False) + '\n')
    print('agg done')

if __name__ == '__main__':
    if sys.argv[1] == 'unit': run_unit(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 2000)
    else: agg(sys.argv[2])
