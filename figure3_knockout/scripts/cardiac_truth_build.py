#!/usr/bin/env python
# Build cardiac knockout-response and ChIP-occupancy labels.
import gzip, bisect, collections, hashlib, json, os, sys, subprocess
import numpy as np, pandas as pd

R = __import__("os").environ["GEOBRIDGE_ROOT"]
RAW = f"{R}/data/raw/mouse_cardiac"
LZ = f"{RAW}/lz_rnaseq"
FEAS = f"{R}/data/raw/mouse_cardiac_feasibility"
OUT = f"{R}/results/cardiac_truth_v1"
T2G = f"{R}/refs/kb_nac_GRCm38/t2g.txt"
REFGENE = f"{FEAS}/annotation/refGene_mm9.txt.gz"
FROZEN = f"{R}/scratch/CARDIAC_TRUTH_V1_FROZEN.md"
SAMPLES = {"WT": ["SRR2177623", "SRR2177624", "SRR2177625"], "Tbx5KO": ["SRR2177629", "SRR2177630", "SRR2177631"], "Nkx2-5KO": ["SRR2177635", "SRR2177636", "SRR2177637"]}
CONTRASTS = {"Tbx5": "Tbx5KO", "Nkx2-5": "Nkx2-5KO"}           # TF -> KO genotype (both vs WT)
FOOT = {"Tbx5": "GSM2054345_Tbx5_WT_CM_avgfootprints.txt.gz", "Nkx2-5": "GSM2054346_Nkx_WT_CM_avgfootprints.txt.gz", "Gata4": "GSM2054347_Gata4_WT_CM_avgfootprints.txt.gz"}
PADJ, LFC, WIN, WIN10 = 0.05, 0.5, 2000, 10000     # Sensitivity occupancy window.


def sha(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def load_t2g():
    t = pd.read_csv(T2G, sep="\t", header=None, usecols=[0, 1, 2], names=["tx", "gene_id", "gene"], dtype=str)
    return t


# sum salmon NumReads over transcripts of a gene_id; returns (counts DataFrame samples x gene_id, gene_id -> symbol Series)
def gene_counts():
    t2g = load_t2g().set_index("tx")
    cols = {}
    for g, runs in SAMPLES.items():
        for r in runs:
            q = pd.read_csv(f"{LZ}/quant/{r}/quant.sf", sep="\t", index_col=0)
            miss = q.index.difference(t2g.index)
            if len(miss):
                raise SystemExit(f"[ABORT] {len(miss)} salmon transcripts absent from t2g, e.g. {miss[:3]}")
            gid = t2g.gene_id.reindex(q.index)
            cols[r] = q.NumReads.groupby(gid).sum()
    C = pd.DataFrame(cols).fillna(0.0)
    sym = load_t2g().drop_duplicates("gene_id").set_index("gene_id").gene.reindex(C.index)
    return np.rint(C).astype(int).T, sym


# counts: samples x genes ints
def run_deseq(counts, meta, design, contrast):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    dds = DeseqDataSet(counts=counts.loc[:, counts.sum(0) >= 10], metadata=meta, design=design, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=contrast, quiet=True)
    st.summary()
    return st.results_df.reindex(counts.columns)


def do_de():
    os.makedirs(OUT, exist_ok=True)
    C, sym = gene_counts()
    C.T.assign(gene=sym).to_csv(f"{OUT}/gene_counts_lz_cm.tsv", sep="\t")
    rec = {}
    for tf, ko in CONTRASTS.items():
        runs = SAMPLES["WT"] + SAMPLES[ko]
        meta = pd.DataFrame(dict(genotype=["WT"] * 3 + ["KO"] * 3), index=runs)
        res = run_deseq(C.loc[runs], meta, "~genotype", ["genotype", "KO", "WT"])
        d = res.rename_axis("gene_id").reset_index().rename(columns={"log2FoldChange": "log2FC"})
        d.insert(1, "gene", sym.reindex(d.gene_id).to_numpy())
        d.to_csv(f"{OUT}/de_{tf}KO_vs_WT.tsv", sep="\t", index=False)
        rec[tf] = dict(n_genes_input=int(C.shape[1]), n_prefiltered_in=int((C.loc[runs].sum(0) >= 10).sum()), n_tested=int(d.padj.notna().sum()),
                       n_padj_lt_05=int((d.padj < PADJ).sum()), n_resp=int(((d.padj < PADJ) & (d.log2FC.abs() >= LFC)).sum()),
                       library_sizes={r: int(C.loc[r].sum()) for r in runs})
        print(tf, rec[tf], flush=True)
    json.dump(rec, open(f"{OUT}/de_summary.json", "w"), indent=1)


# occupancy
# chrom -> sorted [(TSS, symbol)], all refGene mm9 transcript records on non-alt contigs (lz_feas.py rule); + set of symbols with a TSS record
def tss_table():
    tss, syms = collections.defaultdict(list), set()
    for line in gzip.open(REFGENE, "rt"):
        f = line.split("\t")
        chrom, strand, s, e, sym = f[2], f[3], int(f[4]), int(f[5]), f[12]
        if "_" in chrom:
            continue
        tss[chrom].append((s if strand == "+" else e, sym))
        syms.add(sym)
    for c in tss:
        tss[c].sort()
    return tss, syms


# Per symbol
def bound_genes(tss, bed_lines, win=WIN):
    mids = collections.defaultdict(list)
    n = 0
    for line in bed_lines:
        c, s, e = line.split()[:3]
        n += 1
        mids[c].append((int(s) + int(e)) // 2)
    mids = {c: np.sort(np.asarray(v, np.int64)) for c, v in mids.items()}
    out = {}
    for c, recs in tss.items():
        if c not in mids:
            continue
        m = mids[c]
        t = np.asarray([p for p, _ in recs], np.int64)
        lo, hi = np.searchsorted(m, t - win, "left"), np.searchsorted(m, t + win, "right")
        j = np.searchsorted(m, t)
        d = np.minimum(np.abs(t - m[np.clip(j, 0, len(m) - 1)]), np.abs(t - m[np.clip(j - 1, 0, len(m) - 1)]))
        for (_, sym), k, dd in zip(recs, hi - lo, d):
            k0, d0 = out.get(sym, (0, np.inf))
            out[sym] = (k0 + int(k), min(d0, float(dd)))
    return out, n


def do_bound():
    os.makedirs(OUT, exist_ok=True)
    tss, syms = tss_table()
    ens = set(load_t2g().gene.dropna().unique())
    rec = dict(n_refgene_symbols=len(syms))
    ex = sorted(s for s in syms if s not in ens)
    low = {e.lower() for e in ens}
    rec["n_refgene_symbols_exact_in_ensembl"] = len(syms) - len(ex)
    rec["n_refgene_symbols_unmapped_exact"] = len(ex)
    rec["n_unmapped_rescuable_by_case_insensitive_NOTE_ONLY"] = int(sum(s.lower() in low for s in ex))
    open(f"{OUT}/unmapped_refgene_symbols.txt", "w").write("\n".join(ex) + "\n")
    pd.DataFrame(dict(gene=sorted(syms & ens))).to_csv(f"{OUT}/refgene_mapped_symbols.tsv", sep="\t", index=False)
    for tf, fn in FOOT.items():
        b, n = bound_genes(tss, gzip.open(f"{FEAS}/chipexo_footprints/{fn}", "rt"))
        d = pd.DataFrame([(g, k, md) for g, (k, md) in sorted(b.items())], columns=["gene", "n_footprints_2kb", "min_abs_dist"])
        d["in_ensembl_symbols"] = d.gene.isin(ens)
        d["bound_2kb"], d["bound_10kb"] = d.min_abs_dist <= WIN, d.min_abs_dist <= WIN10
        d.to_csv(f"{OUT}/bound_{tf}_CM.tsv", sep="\t", index=False)
        rec[tf] = dict(n_footprints=n, n_bound_2kb_refgene=int(d.bound_2kb.sum()), n_bound_2kb_mapped_to_ensembl=int((d.bound_2kb & d.in_ensembl_symbols).sum()), n_bound_10kb_mapped_to_ensembl=int((d.bound_10kb & d.in_ensembl_symbols).sum()), file_sha256=sha(f"{FEAS}/chipexo_footprints/{fn}"))
        print(tf, rec[tf], flush=True)
    json.dump(rec, open(f"{OUT}/bound_summary.json", "w"), indent=1)


# join + counts
def classes(resp_ok, resp, occ_ok, occ):
    U = resp_ok & occ_ok
    return dict(n_universe=int(U.sum()), n_bound=int((occ & U).sum()), n_resp=int((resp & U).sum()), A=int((occ & resp & U).sum()), B=int((occ & ~resp & U).sum()),
                C=int((~occ & resp & U).sum()), D=int((~occ & ~resp & U).sum()))


def do_table():
    t2g = load_t2g().drop_duplicates("gene_id")
    dup_sym = set(t2g.gene[t2g.gene.duplicated(keep=False) | t2g.gene.isna()])
    mapped = set(pd.read_csv(f"{OUT}/refgene_mapped_symbols.tsv", sep="\t").gene)
    T = pd.DataFrame(dict(gene=sorted(set(t2g.gene.dropna()) - dup_sym)))
    T["refgene_mapped"] = T.gene.isin(mapped)
    for tf in CONTRASTS:
        d = pd.read_csv(f"{OUT}/de_{tf}KO_vs_WT.tsv", sep="\t")
        d = d[d.gene.notna() & ~d.gene.isin(dup_sym)].set_index("gene")
        ok = d.padj.notna()
        T[f"{tf}_tested"] = T.gene.map(ok).fillna(False).astype(bool)
        T[f"{tf}_log2FC"] = T.gene.map(d.log2FC)
        T[f"{tf}_padj"] = T.gene.map(d.padj)
        T[f"{tf}_resp"] = T[f"{tf}_tested"] & (T[f"{tf}_padj"] < PADJ) & (T[f"{tf}_log2FC"].abs() >= LFC)
        T[f"{tf}_resp_up"] = T[f"{tf}_resp"] & (T[f"{tf}_log2FC"] > 0)
        T[f"{tf}_resp_down"] = T[f"{tf}_resp"] & (T[f"{tf}_log2FC"] < 0)
    for tf in FOOT:
        b = pd.read_csv(f"{OUT}/bound_{tf}_CM.tsv", sep="\t").set_index("gene")
        T[f"bound_{tf}"] = T.gene.map(b.bound_2kb).fillna(False).astype(bool)
        T[f"bound10_{tf}"] = T.gene.map(b.bound_10kb).fillna(False).astype(bool)
        T[f"dist_{tf}"] = T.gene.map(b.min_abs_dist)
        T[f"nfoot2kb_{tf}"] = T.gene.map(b.n_footprints_2kb)
    ma = f"{OUT}/microarray_de_E875.tsv"
    if os.path.exists(ma):
        m = pd.read_csv(ma, sep="\t").set_index("gene")
        for tf in CONTRASTS:
            T[f"{tf}_micro_tested"] = T.gene.isin(m.index[m[f"{tf}_padj"].notna()])
            T[f"{tf}_micro_resp"] = T.gene.map(m[f"{tf}_resp"]).fillna(False).astype(bool)
    T.to_csv(f"{OUT}/truth_gene_table.tsv", sep="\t", index=False)
    cnt = {}
    for tf in CONTRASTS:
        cnt[tf] = classes(T[f"{tf}_tested"].to_numpy(), T[f"{tf}_resp"].to_numpy(), T.refgene_mapped.to_numpy(), T[f"bound_{tf}"].to_numpy())
        cnt[tf]["gate_bound_ge_100"] = cnt[tf]["n_bound"] >= 100
        cnt[tf]["gate_AB_ge_20"] = cnt[tf]["A"] >= 20 and cnt[tf]["B"] >= 20
        cnt[tf]["gate_CD_ge_20"] = cnt[tf]["C"] >= 20 and cnt[tf]["D"] >= 20
        cnt[tf]["n_resp_up"] = int(T[f"{tf}_resp_up"].sum()); cnt[tf]["n_resp_down"] = int(T[f"{tf}_resp_down"].sum())
        if f"{tf}_micro_tested" in T:
            cnt[tf]["microarray_sensitivity"] = classes(T[f"{tf}_micro_tested"].to_numpy(), T[f"{tf}_micro_resp"].to_numpy(), T.refgene_mapped.to_numpy(), T[f"bound_{tf}"].to_numpy())
    for tf in CONTRASTS:
        cnt[tf]["sens_10kb"] = classes(T[f"{tf}_tested"].to_numpy(), T[f"{tf}_resp"].to_numpy(), T.refgene_mapped.to_numpy(), T[f"bound10_{tf}"].to_numpy())
    cnt["Gata4"] = dict(n_bound_mapped=int((T.bound_Gata4 & T.refgene_mapped).sum()), role="DESCRIPTIVE occupancy only (no KO)")
    files = {}
    for p in sorted(os.listdir(OUT)):
        if p.endswith((".tsv", ".json", ".txt")) and p != "receipt.json":
            files[p] = sha(f"{OUT}/{p}")
    inputs = {f"quant/{r}": sha(f"{LZ}/quant/{r}/quant.sf") for runs in SAMPLES.values() for r in runs}
    inputs.update({"t2g": sha(T2G), "refGene_mm9": sha(REFGENE)})
    rec = dict(frozen_declaration=FROZEN, frozen_declaration_sha256=sha(FROZEN), builder_sha256=sha(__file__), class_counts=cnt, n_genes_table=len(T), n_refgene_mapped=int(T.refgene_mapped.sum()),
               n_dup_or_missing_symbols_excluded=len(dup_sym), thresholds=dict(padj=PADJ, abs_log2FC=LFC, tss_window=WIN), output_files_sha256=files, input_sha256=inputs,
               software=dict(pydeseq2=__import__("pydeseq2").__version__, numpy=np.__version__, pandas=pd.__version__), created=pd.Timestamp.utcnow().isoformat())
    json.dump(rec, open(f"{OUT}/receipt.json", "w"), indent=1, default=str)
    print(json.dumps(cnt, indent=1))


def selftest():
    tss = {"chr1": sorted([(1000, "A"), (5000, "B"), (100000, "C")])}
    b, n = bound_genes(tss, ["chr1\t1500\t1502", "chr1\t3000\t3002", "chr1\t7000\t7001", "chr1\t7002\t7003", "chr1\t102999\t103001", "chr2\t1\t3"])
    # mids 1501, 3001, 7000, 7002, 103000
    assert b == {"A": (1, 501.0), "B": (2, 1999.0), "C": (0, 3000.0)} and n == 6, b
    c = classes(np.array([1, 1, 1, 1, 0, 1], bool), np.array([1, 0, 1, 0, 1, 0], bool), np.array([1, 1, 1, 1, 1, 0], bool), np.array([1, 1, 0, 0, 1, 1], bool))
    assert (c["A"], c["B"], c["C"], c["D"], c["n_universe"]) == (1, 1, 1, 1, 4), c        # row 5: untested -> excluded; row 6: occupancy unknown -> excluded
    print("cardiac_truth_build selftest OK")


if __name__ == "__main__":
    {"selftest": selftest, "de": do_de, "bound": do_bound, "table": do_table}[sys.argv[1]]()
