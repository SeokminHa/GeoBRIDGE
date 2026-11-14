# BM_GATA1_acute truth adapter without the NT-detection filter.
# unique non-missing symbols kept, upper-cased before duplicate detection; no BH recomputation
import sys, numpy as np, pandas as pd
R = __import__('os').environ['GEOBRIDGE_ROOT'] + '/'
sys.path[:0] = [R + 'scratch', R + 'pairspace']
import gb_layer_lattice as L, gb_lattice_extra as LE


def sets_norm(path=None):
    d = pd.read_csv(path or LE.PROSEQ_GATA1, sep='\t'); q = d[LE.PADJ24].to_numpy(float)
    su = d.Symbol.astype('string').str.upper()
    dupU = (su.duplicated(keep=False) | su.isna()).to_numpy(bool)
    ok = np.isfinite(q) & ~dupU
    return set(su[ok]), set(su[ok & (q < 0.05)]), dict(n_rows=len(d), n_dup_or_missing_symbol_upper=int(dupU.sum()), n_finite_padj=int(np.isfinite(q).sum()))


def truth_noF2(tf, genes, variant='MK'):
    C = LE.bm_ctx()
    tested, resp, st = sets_norm()                      # NO `tested &= C["nt"]`
    pk, pkf = L.peak_mids('GATA1')
    if pk is None: raise L.Blocked('NO_MK_GATA1_PEAKS')
    occ = LE.occ_from_peaks(pk, C['tss'])
    occ['mf'] = dict(mapping=LE.WIN_BM, occupancy=f'ChIP-Atlas hg38 GSE24674 MK {pkf.split("/")[-1]} (the BONEMARROW primary rule)', files=[L.frec(pkf)])
    return LE.bm_truth(genes, resp & tested, tested, occ, dict(response='GSE274110 24 h GATA1 vs AAVS1 PRO-seq DESeq2 padj<.05; tested = finite padj, unique (UPPERCASE-normalised) non-NA symbol; NO NTC detect filter (BM_ACUTE_TRUTH_RULE)',
                                                               proseq_stats=st, files=[L.frec(LE.PROSEQ_GATA1), L.frec(L.BM_GTF)]))


def install_noF2():
    LE.EXTRA['BM_GATA1_ACUTE']['truth'] = lambda tf, g: truth_noF2(tf, g)


def effect_norm():
    d = pd.read_csv(R + 'data/gse274110_proseq/GSE274110_Sankaran003_gga_deseq_24h_GATA1vs24h_AAVS1_results_table.txt.gz', sep='\t')
    su = d.Symbol.astype('string').str.upper(); lf = d['log2FoldChange_24h_GATA1vs24h_AAVS1'].astype(float)
    ok = (~(su.duplicated(keep=False) | su.isna())).to_numpy(bool)
    return pd.Series(lf.to_numpy()[ok], index=su[ok].astype(str).to_numpy())
