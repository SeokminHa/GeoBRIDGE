#!/usr/bin/env python
# build GATA1 occupancy and response labels for evaluation
import gzip, hashlib, json, os, re, sys, time
import numpy as np, pandas as pd

R = __import__("os").environ["GEOBRIDGE_ROOT"]
PK = f"{R}/data/gata1_chip_mouse/ENCFF676TDJ.bed.gz"; PK_MD5 = "68bd64a501d693522380c8da4001a377"
GTF = f"{R}/data/gencode/gencode.vM25.annotation.gtf.gz"
OUT = f"{R}/results/truth_v2"; PROM, W10, W50 = 1000, 10_000, 50_000
RULE = ("D-V2: BOUND_PROM = ENCFF676TDJ summit within ±1 kb of any GENCODE vM25 protein_coding/lncRNA TSS (all transcripts); "
        "BOUND_10KB ±10 kb sensitivity; KO axis = gb_gata1_fourclass_canon (|T|>0.5); A=bound∧resp(d), B=bound∧non-resp, "
        "C=unbound∧resp(d), D=unbound∧non-resp")


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# TSS: {gene: [(chrom, pos0), ...]}; peaks: DataFrame chrom, summit, signal
def gene_features(tss, peaks):
    by = {c: (g.summit.to_numpy(np.int64)[np.argsort(g.summit.to_numpy(), kind="stable")],
              g.signal.to_numpy(float)[np.argsort(g.summit.to_numpy(), kind="stable")],
              g.index.to_numpy()[np.argsort(g.summit.to_numpy(), kind="stable")]) for c, g in peaks.groupby("chrom")}
    rows = []
    for gene, sites in tss.items():
        dmin, s10, n50 = np.inf, -np.inf, set()
        for c, p in sites:
            if c not in by:
                continue
            s, sig, pid = by[c]
            i = np.searchsorted(s, p)
            for j in (i - 1, i):
                if 0 <= j < len(s):
                    dmin = min(dmin, abs(int(s[j]) - p))
            lo, hi = np.searchsorted(s, p - W10, "left"), np.searchsorted(s, p + W10, "right")
            if hi > lo:
                s10 = max(s10, float(sig[lo:hi].max()))
            lo, hi = np.searchsorted(s, p - W50, "left"), np.searchsorted(s, p + W50, "right")
            n50.update(pid[lo:hi].tolist())
        rows.append(dict(gene=gene, nearest_summit_tss_dist=dmin if np.isfinite(dmin) else np.nan,
                         max_signal_10kb=s10 if np.isfinite(s10) else 0.0, n_peaks_50kb=len(n50),
                         BOUND_PROM=bool(dmin <= PROM), BOUND_10KB=bool(dmin <= W10)))
    return pd.DataFrame(rows)


# A valid narrowPeak summit satisfies start <= start + off < end.
def malformed_summit(start, end, off):
    start, end, off = (np.asarray(x, np.int64) for x in (start, end, off))
    return (off < 0) | (off >= end - start)


def selftest():
    pk = pd.DataFrame(dict(chrom=["chr1", "chr1", "chr2"], summit=[1500, 20_000, 100], signal=[5.0, 9.0, 1.0]))
    t = {"a": [("chr1", 1000)], "b": [("chr1", 12_000), ("chr3", 5)], "c": [("chr2", 100_000)], "d": [("chrX", 1)]}
    f = gene_features(t, pk).set_index("gene")
    assert f.loc["a", "BOUND_PROM"] and f.loc["a", "nearest_summit_tss_dist"] == 500 and f.loc["a", "max_signal_10kb"] == 5.0
    assert f.loc["a", "n_peaks_50kb"] == 2                                        # 1500 and 20000 both within 50 kb
    assert not f.loc["b", "BOUND_PROM"] and f.loc["b", "BOUND_10KB"] and f.loc["b", "nearest_summit_tss_dist"] == 8000
    assert f.loc["b", "max_signal_10kb"] == 9.0                                   # 20000 within 10 kb of 12000; 1500 is not
    assert not f.loc["c", "BOUND_10KB"] and f.loc["c", "n_peaks_50kb"] == 0
    assert np.isnan(f.loc["d", "nearest_summit_tss_dist"]) and not f.loc["d", "BOUND_PROM"]
    pk2 = pd.DataFrame(dict(chrom=["c"] * 4, summit=[11_000, 31_001, 60_000, 90_001], signal=[1.0] * 4))
    f2 = gene_features({"p": [("c", 10_000)], "q": [("c", 30_000)], "r": [("c", 50_000)], "s": [("c", 80_000)]}, pk2).set_index("gene")
    assert f2.loc["p", "BOUND_PROM"] and not f2.loc["q", "BOUND_PROM"] and f2.loc["q", "BOUND_10KB"]
    assert f2.loc["r", "BOUND_10KB"] and not f2.loc["s", "BOUND_10KB"]
    pk3 = pd.DataFrame(dict(chrom=["c"] * 4, summit=[9_000, 28_999, 40_000, 69_999], signal=[1.0] * 4))
    f3 = gene_features({"p": [("c", 10_000)], "q": [("c", 30_000)], "r": [("c", 50_000)], "s": [("c", 80_000)]}, pk3).set_index("gene")
    assert f3.loc["p", "BOUND_PROM"] and f3.loc["p", "nearest_summit_tss_dist"] == 1000
    assert not f3.loc["q", "BOUND_PROM"] and f3.loc["q", "BOUND_10KB"] and f3.loc["q", "nearest_summit_tss_dist"] == 1001
    assert f3.loc["r", "BOUND_10KB"] and f3.loc["r", "nearest_summit_tss_dist"] == 10_000
    assert not f3.loc["s", "BOUND_10KB"] and f3.loc["s", "nearest_summit_tss_dist"] == 10_001
    # strand parsing: + TSS = start-1, - TSS = end-1 (GTF 1-based closed)
    a = '1\tH\ttranscript\t101\t500\t.\t+\t.\tgene_type "protein_coding"; gene_name "Gp";'
    b = '1\tH\ttranscript\t101\t500\t.\t-\t.\tgene_type "lncRNA"; gene_name "Gm";'
    c = '1\tH\ttranscript\t101\t500\t.\t+\t.\tgene_type "miRNA"; gene_name "Mir";'
    assert parse_tss(a) == ("Gp", "1", 100) and parse_tss(b) == ("Gm", "1", 499) and parse_tss(c) is None
    assert parse_tss(a.replace("transcript", "exon")) is None
    assert malformed_summit([100, 100, 100, 100], [200, 200, 200, 200], [0, 99, 100, -1]).tolist() == [False, False, True, True]
    print("gb_gata1_truth_v2 selftest PASS")


_PAT = re.compile(r'(gene_type|gene_name) "([^"]+)"')


# one GTF line -> (gene_name, chrom, 0-based TSS) for protein_coding/lncRNA transcripts, else None (1-based GTF: + start-1, - end-1)
def parse_tss(ln):
    if ln.startswith("#"):
        return None
    f = ln.split("\t", 9)
    if f[2] != "transcript":
        return None
    kv = dict(_PAT.findall(f[8]))
    if kv.get("gene_type") not in ("protein_coding", "lncRNA"):
        return None
    return kv["gene_name"], f[0], (int(f[3]) - 1 if f[6] == "+" else int(f[4]) - 1)


def load_tss():
    tss = {}
    with gzip.open(GTF, "rt") as fh:
        for ln in fh:
            r = parse_tss(ln)
            if r:
                tss.setdefault(r[0], set()).add((r[1], r[2]))
    return {g: sorted(s) for g, s in tss.items()}


# Responsive / non-responsive masks straight from FC.four_classes (no KO arithmetic here): every gene is declared bound
def fc_ko_masks(FC, genes):
    orig = FC.pd.read_csv

    def allbound(path, *a, **k):
        if str(path).endswith("chipseq_labels.tsv"):
            return pd.DataFrame(dict(gene=list(genes), ChIP_bound=1.0))
        return orig(path, *a, **k)
    FC.pd.read_csv = allbound
    try:
        cls, _ = FC.four_classes(list(genes))
    finally:
        FC.pd.read_csv = orig
    resp = {pol: np.asarray(cls[pol]["A"], bool) for pol in cls}
    nonresp = np.asarray(cls["any_change"]["B"], bool)
    assert not (resp["any_change"] & nonresp).any() and not (cls["any_change"]["C"].any() or cls["any_change"]["D"].any())
    return resp, nonresp


def main():
    t0 = time.time()
    sys.path.insert(0, f"{R}/scratch")
    import hashlib as _h
    if _h.md5(open(PK, "rb").read()).hexdigest() != PK_MD5:
        raise SystemExit("REFUSING: peak file md5 != ENCODE record")
    from gb_substrate import load, d_
    import gb_gata1_fourclass_canon as FC
    import anndata as ad
    import gb_canonical as _gc
    S_ = load("GATA1", canonical=True); genes = [str(g) for g in np.asarray(S_["genes"]).astype(str)]
    pk = pd.read_csv(PK, sep="\t", header=None)
    bad = malformed_summit(pk[1], pk[2], pk[9]); n_bad_summit = int(bad.sum())
    if n_bad_summit > 5:
        raise SystemExit(f"REFUSING: {n_bad_summit} malformed summits (policy tolerates ≤5 excluded records)")
    pk = pk[~bad]
    peaks = pd.DataFrame(dict(chrom=pk[0].astype(str), summit=pk[1].astype(np.int64) + pk[9].astype(np.int64), signal=pk[6].astype(float)))
    tss = load_tss()
    F = gene_features({g: tss[g] for g in genes if g in tss}, peaks).set_index("gene").reindex(genes)
    matched = np.array([g in tss for g in genes])
    resp, nonresp = fc_ko_masks(FC, genes)

    def classes(bound, unbound):
        return {pol: dict(A=bound & resp[pol], B=bound & nonresp, C=unbound & resp[pol], D=unbound & nonresp) for pol in resp}
    dfc = pd.read_csv(f"{FC.G}/chipseq_labels.tsv", sep="\t"); chip = dict(zip(dfc.gene.astype(str), dfc.ChIP_bound.astype(float)))
    cb = np.array([chip.get(g, np.nan) for g in genes])
    ref, _ = FC.four_classes(genes); mine = classes(cb == 1.0, cb == 0.0)   # consistency check of the mask extraction
    assert all((ref[p][c] == mine[p][c]).all() for p in ref for c in "ABCD"), "FC mask extraction disagrees with four_classes"
    has_T = resp["any_change"] | nonresp
    T = pd.DataFrame(dict(gene=genes, gencode_matched=matched, KO_tested=has_T,
                          KO_resp=np.select([resp["activation"], resp["repression"], nonresp], ["act", "rep", "none"], "")))
    for col in ("nearest_summit_tss_dist", "max_signal_10kb", "n_peaks_50kb"):
        T[col] = F[col].to_numpy()
    counts = {}
    for lab in ("BOUND_PROM", "BOUND_10KB"):
        b = np.where(matched, F[lab].fillna(False).to_numpy(bool), False); u = matched & ~b
        T[lab] = np.where(matched, b.astype(float), np.nan)
        cl = classes(b, u)
        for pol in cl:
            T[f"{pol}_{lab}"] = np.select([cl[pol][c] for c in "ABCD"], list("ABCD"), ".")
        counts[lab] = {pol: {c: int(cl[pol][c].sum()) for c in "ABCD"} for pol in cl}
    b = T.BOUND_PROM.to_numpy(); ok = np.isfinite(b) & np.isfinite(cb)
    tab = pd.crosstab(pd.Series(cb[ok], name="ChEA3_V1"), pd.Series(b[ok], name="BOUND_PROM_V2"))
    po = np.trace(tab.to_numpy()) / tab.to_numpy().sum()
    pe = (tab.sum(1).to_numpy() * tab.sum(0).to_numpy()).sum() / tab.to_numpy().sum() ** 2
    os.makedirs(OUT, exist_ok=True); out = f"{OUT}/GATA1_PHYSICAL_FUNCTIONAL_V2.tsv"
    T.to_csv(out, sep="\t", index=False)
    rec = dict(truth="GATA1_PHYSICAL_FUNCTIONAL_V2", rule=RULE, encode_experiment="ENCSR000DIL", peak_file="ENCFF676TDJ",
               peak_analysis="ENCAN674VPA ENCODE4 v1.6.1 mm10", peak_md5=PK_MD5, peak_sha256=sha(PK), n_peaks=len(peaks),
               gtf="gencode.vM25.annotation.gtf.gz", gtf_sha256=sha(GTF),
               fixture=S_["fixture"], fixture_sha256=sha(S_["fixture"]), pairs_file=S_["pairs_file"],
               pairs_sha256=sha(S_["pairs_file"]), n_malformed_summits_excluded=n_bad_summit,
               code_sha256={os.path.basename(q): sha(q) for q in (f"{R}/scratch/gb_substrate.py", f"{R}/scratch/gb_canonical.py")},
               canonical_config=_gc.CONFIG, canonical_config_sha256=sha(_gc.CONFIG),
               runtime=dict(python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, anndata=ad.__version__),
               ko_mask_digest=hashlib.sha256(("\n".join(genes) + "|" + "".join(
                   "".join("1" if x else "0" for x in m) for m in (resp["activation"], resp["repression"], resp["any_change"], nonresp))
               ).encode()).hexdigest(),
               gene_axis_sha256=hashlib.sha256("\n".join(genes).encode()).hexdigest(),
               fc_constructor=FC.__file__, fc_constructor_sha256=sha(FC.__file__),
               fc_inputs={os.path.basename(q): sha(q) for q in (f"{FC.G}/chipseq_labels.tsv", f"{FC.G}/gata1_control_deterministic.h5ad",
                                                                 f"{FC.G}/gata1_ko_projected_deterministic.h5ad")}, n_axis_genes=len(genes), n_unmatched=int((~matched).sum()),
               n_has_KO=int(has_T.sum()), counts=counts, chea3_vs_v2_2x2={f"{int(i)}_{int(j)}": int(tab.loc[i, j]) for i in tab.index for j in tab.columns},
               kappa=float((po - pe) / (1 - pe)), ko_axis_from_fourclass_canon=True, out=out, sha256=sha(out),
               script_sha256=sha(os.path.abspath(__file__)), geobridge_scores_read=False,
               built_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), wall_s=round(time.time() - t0, 1))
    json.dump(rec, open(out.replace(".tsv", ".receipt.json"), "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("n_peaks", "n_axis_genes", "n_unmatched", "n_has_KO", "counts", "chea3_vs_v2_2x2", "kappa", "sha256")}, indent=1))


if __name__ == "__main__":
    selftest()
    if "--selftest" not in sys.argv:
        main()
