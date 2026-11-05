"""Genome-wide KO truth tables for the four main endpoints, exactly as in scratch/gb_truth_attrition_2026_10_28.py (same files, same rules; frozen):
PANC_RFX6  Aldous human PP RFX6-KO DESeq2 (our re-analysis) padj<.05, no LFC; human->mouse MGI 1:1 orthologs; ambiguous symbols dropped.
HEART_*    results/cardiac_truth_v1/truth_gene_table.tsv <TF>_tested / _padj / _log2FC / _resp (padj<.05 & |LFC|>=.5).
BM_GATA1_acute  GSE274110 deposited 24 h GATA1 vs AAVS1 DESeq2 padj<.05, no LFC, unique uppercase symbol (noF2 primary).
Returns DataFrame indexed by UPPERCASE target symbol with columns padj, lfc, testable, resp, cls (KOup/KOdown/nonresp/untestable)."""
import numpy as np, pandas as pd
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
up = lambda s: pd.Series(s).astype('string').str.upper()
TF = {'PANC_RFX6': 'RFX6', 'HEART_TBX5': 'TBX5', 'HEART_NKX2-5': 'NKX2-5', 'BM_GATA1_acute': 'GATA1', 'ERY_KLF1': 'KLF1', 'ERY_KLF1_B': 'KLF1'}
# 2026-10-28 (PI: expand the two ERY KLF1 endpoints, SI): the Fig 3 ERY rules, unchanged —
#  ERY_KLF1   GSE94351 cuffdiff Q2 vs Q3, status OK, single-symbol rows (gb_klf1_eval.ko_truth); resp = significant == yes; effect = log2 KO/het
#             oriented as gb_fig3d_magnitude.effect('KLF1'); tested = every retained row (pseudo-padj: 0 if resp else 1, so testable = True)
#  ERY_KLF1_B GSE33979 DESeq2 (results/lattice_extra/truth/klf1_GSE33979_deseq2.tsv): resp = padj<.05; tested = finite padj; effect = log2FC

def truth_table(lab):
    tf = TF[lab]
    if lab == 'PANC_RFX6':
        d = pd.read_csv(R + 'real_1_Pancreas/external_validation/aldous_rfx6/results/aldous_PPs_WT_vs_KO_DE.tsv', sep='\t')
        hs = up(d.gene_name); dup = (hs.duplicated(keep=False) | hs.isna()).to_numpy()
        hom = pd.read_csv(R + 'data/orthology/HOM_MouseHumanSequence.rpt', sep='\t', low_memory=False); key = hom.columns[0]; org = hom['Common Organism Name']
        mm = hom[org.str.startswith('mouse')].groupby(key)['Symbol'].apply(list); hh = hom[org.str.startswith('human')].groupby(key)['Symbol'].apply(list)
        h2m = {hh[k][0].upper(): mm[k][0].upper() for k in set(mm.index) & set(hh.index) if len(mm[k]) == 1 and len(hh[k]) == 1}
        T = pd.DataFrame(dict(gene=[h2m.get(str(s)) if not dd else None for s, dd in zip(hs, dup)], padj=d.padj.astype(float), lfc=d.log2FoldChange.astype(float)))
        T['resp'] = T.padj < .05
    elif lab.startswith('HEART'):
        t = 'Tbx5' if lab == 'HEART_TBX5' else 'Nkx2-5'; d = pd.read_csv(R + 'results/cardiac_truth_v1/truth_gene_table.tsv', sep='\t')
        g = up(d.gene); dup = (g.duplicated(keep=False) | g.isna()).to_numpy()
        T = pd.DataFrame(dict(gene=g.where(~dup, None).to_numpy(), padj=d[f'{t}_padj'].astype(float).where(d[f'{t}_tested'].astype(bool)).to_numpy(), lfc=d[f'{t}_log2FC'].astype(float).to_numpy()))
        T['resp'] = d[f'{t}_resp'].astype(bool).to_numpy()
    elif lab == 'ERY_KLF1':
        import sys; sys.path.insert(0, R + 'scratch'); import gb_klf1_eval as KE
        D = KE.ko_truth(R + 'results/klf1_heldout/truth/GSE94351_gene_exp.diff.gz')
        T = pd.DataFrame(dict(gene=D.index.astype(str), padj=np.where(D.resp, 0.0, 1.0), lfc=D.ko_effect.astype(float).to_numpy()))
        T['resp'] = D.resp.to_numpy(bool)
    elif lab == 'ERY_KLF1_B':
        d = pd.read_csv(R + 'results/lattice_extra/truth/klf1_GSE33979_deseq2.tsv', sep='\t')
        g = up(d.gene); dup = (g.duplicated(keep=False) | g.isna()).to_numpy()
        T = pd.DataFrame(dict(gene=g.where(~dup, None).to_numpy(), padj=d.padj.astype(float), lfc=d.log2FC.astype(float)))
        T['resp'] = T.padj < .05
    else:
        d = pd.read_csv(R + 'data/gse274110_proseq/GSE274110_Sankaran003_gga_deseq_24h_GATA1vs24h_AAVS1_results_table.txt.gz', sep='\t')
        g = up(d.Symbol); dup = (g.duplicated(keep=False) | g.isna()).to_numpy()
        T = pd.DataFrame(dict(gene=g.where(~dup, None).to_numpy(), padj=d['padj_24h_GATA1vs24h_AAVS1'].astype(float), lfc=d['log2FoldChange_24h_GATA1vs24h_AAVS1'].astype(float)))
        T['resp'] = T.padj < .05
    T = T[T.gene.notna() & (T.gene != tf)]
    T = T[~T.gene.duplicated(keep=False)].set_index('gene')
    T['testable'] = np.isfinite(T.padj); T['resp'] = T.resp & T.testable
    T['cls'] = np.where(~T.testable, 'untestable', np.where(T.resp & (T.lfc > 0), 'KOup', np.where(T.resp & (T.lfc < 0), 'KOdown', 'nonresp')))
    return T
