#!/usr/bin/env python
"""ENDPOINT CONSTRUCTORS v2 (truth side only; no GeoBRIDGE score is read) — internal review D1/D2 (2026-09-23, results doc §102).
usage: gb_endpoints_v2.py rfx6_occupancy   --out <tsv>
       gb_endpoints_v2.py gata1_proseq     --out <tsv>

rfx6_occupancy (D1). Occupancy = GSE62844 mouse Min6b1 HA-Rfx6 ChIP-seq, MACS 1.4.2 peaks (genome mm9, inferred in §101), peaks
  with FDR(%) <= 5 only. ONE coordinate convention, 1-based (internal review r8): MACS 1.4 start is 1-based and summit is a 1-based offset,
  so the absolute summit = start - 1 + summit; UCSC refGene txStart is 0-based, so a plus-strand TSS = txStart + 1 and a
  minus-strand TSS = txEnd (0-based exclusive end = 1-based last base). Gene TSS from UCSC refGene mm9;
  ONE canonical transcript per gene symbol = the longest genomic span (PROVISIONAL, auditable; ties broken by lowest start, then
  transcript ID — never by peak score); non-primary contigs dropped. Each summit is assigned to its single nearest canonical TSS
  (unassigned whenever the nearest coordinate carries >1 symbol or equidistant coordinates carry >1 symbol); a gene is OCCUPIED at window W if it receives >= 1 assigned summit within W: 1 kb (PI primary),
  10 kb (internal review D1 primary), 50 kb (sensitivity). Truth-free WINDOW-SPECIFIC occupancy propensity logits (logit_ps_occ1/10/50,
  same covariates) exported for matching. Genes absent from refGene mm9 = UNKNOWN_MAPPING. Symbols are
  joined to the PANC evaluator axis by exact symbol (mm9 → evaluator annotation by symbol; not a coordinate liftover).
  KO classes come ONLY from the frozen RFX6 DE endpoint (truth_panc: padj<.05; KO-down = activation-associated, KO-up =
  repression-associated; tested non-DE = TESTED_NULL). Output per gene: ko_class, occupancy flags, distance to nearest retained
  summit, and prespecified propensity covariates (log canonical transcript length, number of refGene TSSs; WT abundance and
  detection are added by the evaluator from the substrate). Promoter GC and mappability are NOT available here (declared gap).
  DIRECT_LIKE = responder & OCCUPIED_10KB; KO_ONLY = responder & not occupied & mapped. "KO_ONLY" means no supporting peak under
  this assay and rule — not indirect regulation.

gata1_proseq (D2). GSE274110 GATA1 KO vs AAVS1 PRO-seq, human primary erythroblasts, two donors, deposited DESeq2 24 h table.
  Donor effects at 24 h from the deposited normalised counts: d_i = log2((KO_i + 1)/(AAVS1_i + 1)), i = 1, 2 (sample pairing by the
  deposited column suffix _1/_2 — donor identity of the suffix is ASSUMED, flagged `donor_pairing_assumed`). Eligibility:
  normalised 24 h AAVS1 count >= 10 in both donors and a finite deposited log2FC and padj. RESPONDER: padj < .05 AND sign(d_1) =
  sign(d_2) = sign(deposited log2FC); TESTED_NULL: eligible and padj >= .05; UNRESOLVED: eligible, padj < .05, donor-discordant;
  UNKNOWN: ineligible / missing symbol / duplicated symbol. PRIMARY `label` is PAIRING-FREE (see gata1_proseq); the donor-paired
  rule is kept as `label_paired_assumed`. Direction: KO-down (log2FC<0) = activation-associated. 6 h donor effects
  exported as SECONDARY columns only (no 6 h calls). GEO lists only the four 24 h samples, so the 6 h columns' provenance is NOT
  certified: the receipt carries sixh_provenance_certified=false and consumers must treat the 6 h rung as NO_CALL_PROVENANCE. Symbols are human; the BM substrate is human (joined by exact symbol).
"""
import argparse, hashlib, json, os, sys
import numpy as np, pandas as pd

R = __import__("os").environ["GEOBRIDGE_ROOT"]


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def macs_summit_1b(pk):
    """MACS 1.4: 1-based start + 1-based summit offset -> absolute 1-based summit"""
    return pk.start - 1 + pk.summit


def refgene_tss_1b(rg):
    """UCSC refGene (col 3 strand, 4 txStart 0-based, 5 txEnd exclusive) -> 1-based TSS"""
    return np.where(rg[3] == "+", rg[4] + 1, rg[5])


def assign_summits(canon, pk):
    """canon: one row per gene with columns 12 (symbol), 2 (chrom), 'tss' (1-based); pk: chr, summit_pos (1-based),
    fold_enrichment. Returns ({symbol: [(distance, FE), ...]}, n_ambiguous)."""
    tss_by_chr = {c: g for c, g in canon.groupby(2)}
    assigned = {}                                  # gene -> list of (distance, FE)
    n_amb = 0
    for c, g in pk.groupby("chr"):
        T_ = tss_by_chr.get(c)
        if T_ is None:
            continue
        # internal review r8: GROUP canonical TSSs by coordinate, so co-located genes are one candidate carrying several symbols
        grp = T_.groupby("tss")[12].apply(lambda x: sorted(set(x))); tpos = grp.index.to_numpy(); tsyms = grp.to_numpy()
        for spos, fe in zip(g.summit_pos.to_numpy(), g.fold_enrichment.to_numpy()):
            i = np.searchsorted(tpos, spos); cand = [j for j in (i - 1, i) if 0 <= j < len(tpos)]
            d = np.array([abs(tpos[j] - spos) for j in cand]); best = d.min()
            syms = {x for j, dj in zip(cand, d) if dj == best for x in tsyms[j]}
            if len(syms) > 1:
                n_amb += 1; continue
            assigned.setdefault(syms.pop(), []).append((float(best), float(fe)))
    return assigned, n_amb


def rfx6_occupancy(a):
    sys.path.insert(0, f"{R}/scratch")
    from gb_substrate import load
    import gb_polarity_compare as PC
    pk_p = f"{R}/data/rfx6_binding/GSE62844/GSM1534425_PSR-06_HA-Rfx6.peaks.txt.gz"
    rg_p = f"{R}/data/rfx6_binding/refGene_mm9.txt.gz"
    pk = pd.read_csv(pk_p, sep="\t", comment="#")
    n_all = len(pk); pk = pk[pk["FDR(%)"] <= 5].copy(); pk["summit_pos"] = macs_summit_1b(pk)
    rg = pd.read_csv(rg_p, sep="\t", header=None)
    rg = rg[~rg[2].astype(str).str.contains("_")]
    rg["len"] = rg[5] - rg[4]; rg["tss"] = refgene_tss_1b(rg)
    rg = rg.sort_values([12, "len", 4, 1], ascending=[True, False, True, True])
    canon = rg.groupby(12).head(1).set_index(12)
    ntss = rg.groupby(12)["tss"].nunique()
    # PEAK -> NEAREST CANONICAL TSS (internal review r-C defect 2): every retained summit is assigned to exactly ONE gene, the canonical TSS
    # nearest to it on the same chromosome; exact-distance ties are ambiguous and left unassigned. A gene is occupied within window W
    # if it receives >= 1 summit at distance <= W. Graded evidence = max fold-enrichment among its assigned summits (<= 10 kb).
    assigned, n_amb = assign_summits(canon.reset_index(), pk)
    occ = {sym: assigned.get(sym, []) for sym in canon.index}
    S = load("PANC", canonical=True)
    Zn_genes = np.asarray(np.load(f"{R}/results/canon_PANC/per_gene.npz", allow_pickle=True)["genes"]).astype(str)
    genes = np.asarray(S["genes"]).astype(str); keep = np.isin(genes, Zn_genes)
    if list(genes[keep]) != list(Zn_genes) or len(set(np.char.upper(Zn_genes))) != len(Zn_genes):
        raise SystemExit("REFUSING: substrate/evaluator gene axis order or case-uniqueness mismatch")
    AX, _, _, _, _ = PC.truth_panc_classes(genes, keep)
    rows = []
    for i, g in enumerate(Zn_genes):
        ko = ("ACT_RESPONDER" if AX["activation"]["pos"][i] else "REP_RESPONDER" if AX["repression"]["pos"][i]
              else "TESTED_NULL" if AX["repression"]["contrasts"]["pos_vs_null"][0][i] else "UNKNOWN")
        if g in canon.index:
            pk_g = occ[g]; d = min([x[0] for x in pk_g], default=np.inf)
            fe10 = max([x[1] for x in pk_g if x[0] <= 10000], default=0.0)
            rows.append(dict(gene=g, ko_class=ko, mapping="MAPPED", dist_nearest_assigned_summit=d,
                             occupied_1kb=bool(d <= 1000), occupied_10kb=bool(d <= 10000), occupied_50kb=bool(d <= 50000),
                             maxFE_10kb=fe10, log_tx_len=float(np.log10(canon.loc[g, "len"])), n_tss=int(ntss[g])))
        else:
            rows.append(dict(gene=g, ko_class=ko, mapping="UNKNOWN_MAPPING"))
    T = pd.DataFrame(rows)
    # truth-free occupancy propensity for matching (internal review D1): logistic P(occupied_10kb | log length, n_tss, WT log abundance,
    # detection) over mapped genes. Promoter GC / mappability unavailable (declared gap).
    import anndata as ad
    from sklearn.linear_model import LogisticRegression
    Ms = np.asarray(S["X"], float)[:, keep]
    A_ = ad.read_h5ad(S["fixture"], backed="r"); fm = {g_: i for i, g_ in enumerate(np.asarray(A_.var_names).astype(str))}
    T["log_abund"] = np.log1p(Ms.mean(0)); T["detect"] = np.asarray((A_.layers["spliced"][:, [fm[g_] for g_ in Zn_genes]] > 0).mean(0)).ravel()
    mp = T.mapping == "MAPPED"; cov = ["log_tx_len", "n_tss", "log_abund", "detect"]
    Xc = T.loc[mp, cov].to_numpy(float); Xc = (Xc - Xc.mean(0)) / Xc.std(0)
    for w in (1, 10, 50):          # internal review r8: one propensity per occupancy window, identical covariates
        lr = LogisticRegression(C=1.0, max_iter=1000).fit(Xc, T.loc[mp, f"occupied_{w}kb"].astype(int))
        T.loc[mp, f"logit_ps_occ{w}"] = lr.decision_function(Xc)
    T["occupancy_class"] = np.where(T.mapping != "MAPPED", "UNKNOWN",
                                    np.where(T.ko_class.isin(["ACT_RESPONDER", "REP_RESPONDER"]),
                                             np.where(T.occupied_10kb == True, "DIRECT_LIKE", "KO_ONLY"),
                                             np.where(T.ko_class == "TESTED_NULL",
                                                      np.where(T.occupied_10kb == True, "NULL_OCCUPIED", "NULL_UNOCCUPIED"),
                                                      "UNKNOWN")))
    T.to_csv(a.out, sep="\t", index=False)
    import gb_dev_evaluate as DE_
    de_p = f"{R}/real_1_Pancreas/external_validation/aldous_rfx6/results/aldous_PPs_WT_vs_KO_DE.tsv"
    hom_p = f"{R}/data/orthology/HOM_MouseHumanSequence.rpt"
    rec = dict(out=a.out, sha256=sha(a.out), script_sha256=sha(os.path.abspath(__file__)), peaks=pk_p, peaks_sha256=sha(pk_p),
               refgene=rg_p, refgene_sha256=sha(rg_p), de=de_p, de_sha256=sha(de_p), orthology=hom_p, orthology_sha256=sha(hom_p),
               fixture=S["fixture"], fixture_sha256=sha(S["fixture"]), per_gene_sha256=sha(f"{R}/results/canon_PANC/per_gene.npz"),
               truth_helper_sha256=sha(DE_.__file__), class_helper_sha256=sha(PC.__file__),
               propensity_covariates=cov, coordinate_convention="1-based (MACS start-1+summit; refGene txStart+1 / txEnd)", n_peaks_all=int(n_all), n_peaks_fdr5=int(len(pk)),
               counts=T.groupby(["ko_class", "occupancy_class"]).size().to_dict().__repr__(), labels_read_scores=False,
               n_summits_ambiguous_tie=int(n_amb), assignment="each summit -> single nearest canonical (longest-span) TSS; ties unassigned")
    json.dump(rec, open(a.out.replace(".tsv", ".receipt.json"), "w"), indent=1)
    print(T.groupby(["ko_class", "occupancy_class"]).size())


def gata1_proseq(a):
    p = f"{R}/data/gse274110_proseq/GSE274110_Sankaran003_gga_deseq_24h_GATA1vs24h_AAVS1_results_table.txt.gz"
    D = pd.read_csv(p, sep="\t")
    lf, pa = "log2FoldChange_24h_GATA1vs24h_AAVS1", "padj_24h_GATA1vs24h_AAVS1"
    d24 = [np.log2((D[f"NormCounts_24h_GATA1_{i}"] + 1) / (D[f"NormCounts_24h_AAVS1_{i}"] + 1)) for i in (1, 2)]
    d6 = [np.log2((D[f"NormCounts_6h_GATA1_{i}"] + 1) / (D[f"NormCounts_6h_AAVS1_{i}"] + 1)) for i in (1, 2)]
    sym = D.Symbol.astype(str)
    dup = sym.duplicated(keep=False) | D.Symbol.isna()
    elig = (D["NormCounts_24h_AAVS1_1"] >= 10) & (D["NormCounts_24h_AAVS1_2"] >= 10) & np.isfinite(D[lf]) & np.isfinite(D[pa]) & ~dup
    agree = (np.sign(d24[0]) == np.sign(d24[1])) & (np.sign(d24[0]) == np.sign(D[lf]))
    lab = np.where(~elig, "UNKNOWN", np.where(D[pa] < 0.05, np.where(agree, "RESPONDER", "UNRESOLVED"), "TESTED_NULL"))
    # PAIRING-FREE label (primary from 2026-09-23): GEO's D1/D2 "samples" are the F/R STRAND bigwigs of one merged library per
    # condition (verified: F carries only plus-strand genes, R only minus-strand), so no deposited evidence certifies which
    # _1/_2 columns share a donor. A responder must agree with the deposited LFC sign in ALL FOUR KO_i vs AAVS1_j cross-ratios,
    # which is invariant to the (unknown) pairing and a subset of the paired rule.
    xr = [np.log2((D[f"NormCounts_24h_GATA1_{i}"] + 1) / (D[f"NormCounts_24h_AAVS1_{j}"] + 1)) for i in (1, 2) for j in (1, 2)]
    agree_x = np.logical_and.reduce([np.sign(x) == np.sign(D[lf]) for x in xr])
    lab_pf = np.where(~elig, "UNKNOWN", np.where(D[pa] < 0.05, np.where(agree_x, "RESPONDER", "UNRESOLVED"), "TESTED_NULL"))
    T = pd.DataFrame(dict(gene=sym, ensembl=D.ID, log2FC_24h=D[lf], padj_24h=D[pa], donor1_24h=d24[0], donor2_24h=d24[1],
                          donor1_6h=d6[0], donor2_6h=d6[1], label=lab_pf, label_paired_assumed=lab,
                          direction=np.where(D[lf] < 0, "KO_DOWN_activation_assoc", "KO_UP_repression_assoc"),
                          donor_pairing_assumed=True))
    T.to_csv(a.out, sep="\t", index=False)
    rec = dict(out=a.out, sha256=sha(a.out), script_sha256=sha(os.path.abspath(__file__)), source=p, source_sha256=sha(p),
               counts=pd.crosstab(T.label, T.direction).to_dict().__repr__(), labels_read_scores=False,
               donor_pairing_assumed=True, sixh_provenance_certified=False, primary_label="label = PAIRING-FREE cross-concordance",
               geo_samples_are_strand_tracks=True,
               geo_note="GSM8443537-40 titled AAVS1_D1/D2, GATA1_KO_D1/D2 are *_H24_F/_R bigwigs (strand-specific: F plus-strand genes, R minus-strand); donor identity and _1/_2 pairing of the DESeq2 table are NOT certified by the deposit",
               n_label_changes_vs_paired=int((lab_pf != lab).sum()),
               sixh_note="GEO GSE274110 lists only the 24 h D1/D2 samples; 6 h columns are uncertified -> NO_CALL_PROVENANCE")
    json.dump(rec, open(a.out.replace(".tsv", ".receipt.json"), "w"), indent=1)
    print(pd.crosstab(T.label, T.direction))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["rfx6_occupancy", "gata1_proseq"]); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    {"rfx6_occupancy": rfx6_occupancy, "gata1_proseq": gata1_proseq}[a.cmd](a)
