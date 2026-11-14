#!/usr/bin/env python
"""Register additional response and occupancy datasets for gb_layer_lattice."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gb_layer_lattice as L  # noqa: E402

R = L.R
X_DIR = f"{R}/results/lattice_extra"
X_OUT = f"{R}/results/real_uniform/layer_lattice_extra"
D = f"{R}/data"
REFGENE_MM9 = f"{D}/rfx6_binding/refGene_mm9.txt.gz"
EKLF_BED = f"{D}/klf1_GSE33979_GSE20478/GSE20478_EKLF_Peaks_5-fold_withScores_mm9.bed.gz"
KLF1_BAMS = {"WT": ["GSM839870_s_6", "GSM839871_s_7", "GSM839872_s_1_8"], "KO": ["GSM839873_s_2", "GSM839874_s_3", "GSM839875_s_5"]}   # geo titles: WT library 1-3 / KO library 1-3
KLF1_DE = f"{X_DIR}/truth/klf1_GSE33979_deseq2.tsv"
PROSEQ_GATA1 = f"{D}/gse274110_proseq/GSE274110_Sankaran003_gga_deseq_24h_GATA1vs24h_AAVS1_results_table.txt.gz"
PADJ24 = "padj_24h_GATA1vs24h_AAVS1"
BLOODCHIP = f"{D}/bloodchip"
CA_DIR = f"{D}/chipatlas_extra"
CA_SEL = f"{CA_DIR}/selection.json"
BCL_NPZ, BCL_MAP = f"{R}/results/proseq_counts_BCL11A.npz", f"{R}/proseq/gse194345/sample_map.tsv"
BCL_DE = {"EARLY": f"{X_DIR}/truth/bcl11a_deseq2_3h.tsv", "LATE": f"{X_DIR}/truth/bcl11a_deseq2_6_24h.tsv"}
BCL_PEAKS_HG38 = f"{X_DIR}/occupancy/bcl11a_HUDEP2_60min_ge2of3_hg38.bed"
BCL_DIR = f"{D}/bcl11a_gse104676"
BCL_REPS = ["GSM2805366_CR8_BCL11A_HDP_60min_Rep1", "GSM2805367_CR8_BCL11A_HDP_60min_Rep2", "GSM2805368_CR8_BCL11A_HDP_60min_Rep3"]
LIFTOVER, CHAIN = f"{D}/refs/liftOver", f"{D}/refs/hg19ToHg38.over.chain.gz"
BC_POPS, BC_ALIAS = ("HSC", "CMP", "GMP", "MEP"), {"SPI1": "PU.1"}
T1 = ("CD34+", "Hematopoietic Stem Cells", "Erythroblasts", "Erythroid Cells", "Erythroid Precursor Cells", "Erythroid progenitors", "CD36+", "PBDE", "PBDEFetal")
T2, T3 = ("HUDEP-2", "HUDEP-1"), ("K-562", "MEG-01", "CHRF-288-11")
# R1 exclusion: only perturbation experiments and non-ChIP
# background (Input/IgG) are dropped
EXCL_PERTURB = re.compile(r"(?i)crybmim|shbrg1|\bko\b|knockout|crispr|dtag|ifn|\bil4\b|\btpp\b|\bbeti\b|(?<!non )wnt-treated|\bkd\b|knock-?down|mybmim|acy-?957|\bre (cb|n822k)\b")
# Additional perturbation-ChIP exclusions missed by the title filter.
SRX_EXCLUDE = {"SRX17966409": "Let-7 knockdown (GATA1)", "SRX17966411": "Let-7 knockdown (GATA1)", "SRX687557": "ACY-957 HDAC1/2 inhibitor treatment (GATA2)",
               "SRX8536093": "RUNX1-ETO (RE CB / RE N822K) transduced cells (GATA2)", "SRX8536094": "RUNX1-ETO (RE CB / RE N822K) transduced cells (GATA2)",
               "SRX8536088": "RUNX1-ETO (RE N822K / RE CB) transduced cells (RUNX1)", "SRX8536095": "RUNX1-ETO (RE N822K / RE CB) transduced cells (RUNX1)",
               "SRX2524364": "MYBMIM inhibitor (MYB)", "SRX2524365": "MYBMIM inhibitor (MYB)", "SRX2524366": "MYBMIM inhibitor (MYB)"}
JUDGMENT_KEEP = re.compile(r"(?i)\bcase\s?\d")          # patient erythroid samples ("Case1/Case2 Ery GATA1"): kept as unperturbed primary erythroid ChIP -- a judgment call recorded in selection.json
EXCL_BACKGROUND = re.compile(r"(?i)\binput\b|\bigg\b")
EXCL_TITLE_V1 = re.compile(r"(?i)crybmim|shbrg1|shluc|shrna|sirna|\bko\b|knockout|dtag|crispr|treat|ifn|\bigg\b|input|control")       # the first (over-broad) rule, kept only for the before/after record
CA_URL = "https://chip-atlas.dbcls.jp/data/hg38/eachData/bed05/{}.05.bed"
PANEL_TFS = ["ATF4", "BCL11A", "FOSL1", "GATA1", "GATA2", "GFI1B", "IRF9", "KLF1", "LDB1", "LMO2", "MYB", "NFE2", "RUNX1", "SNAI2", "SOX12", "SPI1", "TAL1"]   # every GSE274113 stratified-panel TF
CA_TFS = PANEL_TFS
# Reference counts under EXCL_TITLE_V1 and depth >= min(2, n).
PRE_MAJORITY_COUNTS = dict(rule="depth>=min(2,n), EXCL_TITLE_V1", BM_CHIPATLAS={
    "GATA1": dict(prim=[307, 1413, 45, 252], sens=[340, 1599, 12, 66]), "KLF1": dict(prim=[13, 405, 51, 1548], sens=[14, 465, 50, 1488]),
    "NFE2": dict(prim=[95, 1424, 56, 442], sens=[116, 1579, 35, 287]), "SPI1": dict(prim=[8, 457, 21, 1531], sens=[14, 817, 15, 1171]),
    "MYB": dict(prim=[1, 38, 101, 1877], sens=[4, 56, 98, 1859]), "GFI1B": dict(prim=[6, 27, 355, 1674], sens=[20, 71, 341, 1630]),
    "LMO2": dict(prim=[0, 8, 81, 1973], sens=[0, 9, 81, 1972]), "LDB1": dict(prim=[0, 175, 7, 1880], sens=[0, 494, 7, 1561])},
    BM_GATA1_ACUTE_T1={"GATA1": dict(prim=[24, 1599, 6, 251], sens=[30, 1787, 0, 63])})
WIN_BM = ("peak overlap TSS+-2kb", "nearest peak midpoint <=10kb of TSS")


# generic peak utilities (pure)
def read_peaks(path, score_col=4, sep="\t"):
    d = pd.read_csv(path, sep=sep, header=None, comment="#", compression="infer")
    o = pd.DataFrame(dict(chrom=d[0].astype(str), s=d[1].astype(np.int64), e=d[2].astype(np.int64)))
    o["mid"] = (o.s + o.e) // 2
    o["score"] = pd.to_numeric(d[score_col], errors="coerce") if d.shape[1] > score_col else np.nan
    return o


# union of overlapping/touching intervals per chromosome -> chrom, s, e
def merge_union(pk):
    out = []
    for c, g in pk.groupby("chrom"):
        g = g.sort_values("s")
        s, e = g.s.to_numpy(np.int64), g.e.to_numpy(np.int64)
        cm = np.maximum.accumulate(e)
        idx = np.flatnonzero(np.r_[True, s[1:] > cm[:-1]])
        out.append(pd.DataFrame(dict(chrom=c, s=s[idx], e=np.r_[cm[idx[1:] - 1], cm[-1]])))
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=["chrom", "s", "e"])


# regions covered by >= k of the peak sets (each set is unioned first, so depth counts sets, not peaks)
def consensus_depth(sets, k):
    allp = pd.concat([merge_union(x) for x in sets], ignore_index=True)
    out = []
    for c, g in allp.groupby("chrom"):
        pos = np.r_[g.s.to_numpy(np.int64), g.e.to_numpy(np.int64)]
        dl = np.r_[np.ones(len(g), int), -np.ones(len(g), int)]
        o = np.lexsort((dl, pos))                                              # at equal position ends (-1) sort before starts (+1): touching intervals do not stack
        pos, dep = pos[o], np.cumsum(dl[o])
        for i in np.flatnonzero((dep[:-1] >= k) & (pos[1:] > pos[:-1])):
            if out and out[-1][0] == c and out[-1][2] == pos[i]:
                out[-1][2] = int(pos[i + 1])
            else:
                out.append([c, int(pos[i]), int(pos[i + 1])])
    r = pd.DataFrame(out, columns=["chrom", "s", "e"])
    r["mid"], r["score"] = (r.s + r.e) // 2, np.nan
    return r


# genes (upper sym -> (chrom, pos)) with a peak [s,e] such that e >= t-win and s <= t+win == gb_bm_sweep.bound_set rule (same inclusive comparisons)
def overlap_within(tss, pk, win):
    by = {}
    for c, g in pk.groupby("chrom"):
        g = g.sort_values("s")
        by[c] = (g.s.to_numpy(np.int64), np.maximum.accumulate(g.e.to_numpy(np.int64)))
    out = set()
    for sym, (c, t) in tss.items():
        if c in by:
            s, cm = by[c]
            j = np.searchsorted(s, int(t) + win, "right")
            if j > 0 and cm[j - 1] >= int(t) - win:
                out.add(sym)
    return out


# UCSC liftOver on a peak frame (chrom, s, e); returns (lifted frame, n_unmapped)
def lift_hg19_hg38(pk, tmp_prefix):
    if not (os.path.exists(LIFTOVER) and os.path.exists(CHAIN)):
        raise L.Blocked(f"LIFTOVER_MISSING {LIFTOVER} or {CHAIN}")
    src, dst, unm = f"{tmp_prefix}.in.bed", f"{tmp_prefix}.out.bed", f"{tmp_prefix}.unmapped.bed"
    pk.assign(n=np.arange(len(pk)))[["chrom", "s", "e", "n"]].to_csv(src, sep="\t", header=False, index=False)
    subprocess.run([LIFTOVER, src, CHAIN, dst, unm], check=True, capture_output=True)
    o = pd.read_csv(dst, sep="\t", header=None, names=["chrom", "s", "e", "n"]) if os.path.getsize(dst) else pd.DataFrame(columns=["chrom", "s", "e", "n"])
    o["mid"], o["score"] = (o.s + o.e) // 2, np.nan
    return o[["chrom", "s", "e", "mid", "score"]], int(len(pk) - len(o))


# ChIP-Atlas selection rule R1
def excl_reason(title, v1=False, srx=""):
    if v1:
        return "v1" if EXCL_TITLE_V1.search(title) else ""
    if srx in SRX_EXCLUDE:
        return f"perturbation (manual SRX exclusion: {SRX_EXCLUDE[srx]})"
    return "perturbation" if EXCL_PERTURB.search(title) else "input_or_IgG_background" if EXCL_BACKGROUND.search(title) else ""


# E: experiment frame (srx, genome, cls, antigen, ctclass, celltype, title)
def select_chipatlas(E, tf, v1=False):
    x = E[(E.genome == "hg38") & (E.cls == "TFs and others") & (E.antigen == tf)]
    why = pd.Series([excl_reason(r.title, v1, r.srx) for r in x.itertuples()], index=x.index, dtype=object)
    ex, x = x[why != ""], x[why == ""]
    exd = {r.srx: excl_reason(r.title, v1, r.srx) for r in ex.itertuples()}
    for name, tier in (("T1", T1), ("T2", T2), ("T3", T3)):
        y = x[x.celltype.isin(tier)]
        if len(y):
            return dict(tier=name, celltypes=sorted(set(y.celltype)), srx=sorted(y.srx), excluded=exd, n_candidates=int(len(x)))
    y = x[x.ctclass == "Blood"]
    if len(y):
        vc = y.celltype.value_counts()
        best = sorted(vc[vc == vc.max()].index)[0]
        return dict(tier="T4", celltypes=[best], srx=sorted(y[y.celltype == best].srx), excluded=exd, n_candidates=int(len(x)))
    return None


def fetch_chipatlas(args):
    os.makedirs(f"{CA_DIR}/bed05", exist_ok=True)
    lst = f"{CA_DIR}/experimentList.tab"
    if not os.path.exists(lst):
        urllib.request.urlretrieve("https://chip-atlas.dbcls.jp/data/metadata/experimentList.tab", lst)
    rows = []
    with open(lst) as fh:
        for ln in fh:
            f = ln.split("\t", 9)
            if len(f) >= 9 and f[1] == "hg38" and f[2] == "TFs and others" and f[3] in CA_TFS:
                rows.append(f[:9])
    E = pd.DataFrame(rows, columns=["srx", "genome", "cls", "antigen", "ctclass", "celltype", "ctdesc", "stats", "title"])
    sel, before, fails, why_none = {}, {}, {}, {}
    for tf in [t for t in CA_TFS if not args.tfs or t in args.tfs.split(",")]:
        b = select_chipatlas(E, tf, v1=True)
        before[tf] = b and dict(tier=b["tier"], celltypes=b["celltypes"], srx=b["srx"], n_excluded=len(b["excluded"]))
        s = select_chipatlas(E, tf)
        if s is None:
            sel[tf] = None
            why_none[tf] = f"EXCLUDED: rule R1 found no usable hg38 'TFs and others' experiment in any tier (n_hg38_experiments={int(((E.antigen == tf) & (E.genome == 'hg38')).sum())})"
            continue
        for srx in s["srx"]:
            p = f"{CA_DIR}/bed05/{tf}_{srx}.hg38.bed"
            if not os.path.exists(p):
                try:
                    urllib.request.urlretrieve(CA_URL.format(srx), p)
                except Exception as e:                                   # a missing bed05 is recorded here and makes chipatlas_occ fail closed
                    fails[srx] = str(e)
        s["files"] = {x: dict(path=f"{CA_DIR}/bed05/{tf}_{x}.hg38.bed", sha256=L.sha_file(f"{CA_DIR}/bed05/{tf}_{x}.hg38.bed")) for x in s["srx"] if os.path.exists(f"{CA_DIR}/bed05/{tf}_{x}.hg38.bed")}
        s["titles"] = E.set_index("srx").title.reindex(s["srx"]).to_dict()
        s["excluded_titles"] = E.set_index("srx").title.reindex(list(s["excluded"])).to_dict()
        s["judgment_calls_kept"] = {k: v for k, v in s["titles"].items() if JUDGMENT_KEEP.search(str(v))}
        sel[tf] = s
    L.atomic_json(CA_SEL, dict(rule="R1 (perturbation-only exclusion; strict-majority consensus)", experimentList_sha256=L.sha_file(lst), script_sha256=L.sha_file(__file__), selection=sel, srx_manual_exclusions=SRX_EXCLUDE,
                               judgment_call_note="'Case1/Case2 Ery GATA1' patient erythroid samples are KEPT (unperturbed primary erythroid ChIP); see judgment_calls_kept per TF", excluded_tfs=why_none, selection_before_exclusion_rule_change=before, pre_majority_rule_counts=PRE_MAJORITY_COUNTS, download_failures=fails))
    print(json.dumps({k: (v and dict(tier=v["tier"], n=len(v["srx"]), celltypes=v["celltypes"], n_excl=len(v["excluded"]))) for k, v in sel.items()}, indent=1), "failures:", fails, "\nno-data TFs:", why_none)
    print("BEFORE(v1 exclusion):", json.dumps({k: (v and (v["tier"], len(v["srx"]))) for k, v in before.items()}))


def occ_from_peaks(pk, tss):
    dist, score = L.bm_peak_desc(tss, pk)
    return dict(prim=overlap_within(tss, pk, 2000), sens=L.near_within(tss, pk, 10_000), dist=dist, score=score, mappable=set(tss), dup=set(), mf=dict(mapping=WIN_BM))


# The prep receipt next to `path` (<stem>.receipt.json) must exist and its out_sha256 must equal the file's sha256
def verified_receipt(path):
    rp = path.rsplit(".", 1)[0] + ".receipt.json"
    if not os.path.exists(rp):
        raise L.Blocked(f"NO_RECEIPT {rp} (re-run the prep command)")
    rc = json.load(open(rp))
    if rc.get("out_sha256") != L.sha_file(path):
        raise L.Blocked(f"STALE_OR_ALTERED {path}: sha256 != receipt out_sha256 (re-run the prep command)")
    return dict(rc, prep_script_sha_matches_current=(rc.get("script_sha256") == L.sha_file(__file__)))


def chipatlas_occ(tf, tss):
    if not os.path.exists(CA_SEL):
        raise L.Blocked(f"NO_CHIPATLAS_SELECTION {CA_SEL} (run fetch-chipatlas)")
    J = json.load(open(CA_SEL))
    if tf in J.get("excluded_tfs", {}):
        raise L.Blocked(f"NO_CHIPATLAS_EXPERIMENT_FOR_{tf}: {J['excluded_tfs'][tf]}")
    S = J["selection"].get(tf)
    if not S:
        raise L.Blocked(f"NO_CHIPATLAS_SELECTION_FOR_{tf} (run fetch-chipatlas with this TF)")
    if len(S["files"]) != len(S["srx"]) or set(S["files"]) != set(S["srx"]):
        raise L.Blocked(f"CHIPATLAS_FILES_INCOMPLETE_FOR_{tf}: {len(S['files'])} files for {len(S['srx'])} selected experiments (download failures: {J.get('download_failures')})")
    for x, v in S["files"].items():
        if not os.path.exists(v["path"]) or L.sha_file(v["path"]) != v["sha256"]:
            raise L.Blocked(f"CHIPATLAS_BED_ALTERED_OR_MISSING {v['path']} (sha256 != selection.json)")
    sets = [read_peaks(v["path"]) for v in S["files"].values()]
    kdep = len(sets) // 2 + 1                                                   # strict majority of the pooled experiments (n=1:1, 2:2, 3:2, 4:3, 45:23)
    pk = consensus_depth(sets, kdep) if len(sets) > 1 else sets[0]
    occ = occ_from_peaks(pk, tss)
    occ["mf"] = dict(mapping=WIN_BM, occupancy=f"ChIP-Atlas hg38 bed05 rule R1: tier {S['tier']} {S['celltypes']}, n_experiments={len(sets)}, depth>={kdep}, consensus_peaks={len(pk)}",
                     experiments=S["srx"], titles=S["titles"], excluded_perturbation_or_background=S["excluded"], selection_json_sha256=L.sha_file(CA_SEL),
                     files=[L.frec(v["path"]) for v in S["files"].values()])
    return occ


# BM substrate (GENCODE v32 TSS, GSE274113 KO panel)
_BM = {}


def bm_ctx():
    if "st" not in _BM:
        ST = pd.read_csv(f"{L.BM_TRUTH}/gse274113_stratified_panel.tsv", sep="\t")
        nt = pd.read_csv(f"{L.BM_TRUTH}/gene_stats_nt.tsv", sep="\t")
        tss, dup = L.bm_tss()
        _BM.update(st=ST, nt=set(nt.gene[nt.nt_detect_frac >= 0.05].str.upper()), tss=tss, dup=dup)
    return _BM


# resp/tested: sets of upper symbols (tested must already include the NTC detect >= 5% intersection); occ: occ_from_peaks-style dict
def bm_truth(genes, resp, tested, occ, mf):
    C = bm_ctx()
    tr = L.truth_bm_parts(genes, resp, tested, occ["mappable"], occ["prim"], occ["sens"], occ["dup"] or C["dup"], None, occ.get("dist"), occ.get("score"))
    if not np.isfinite(tr["occ_score"]).any():                              # no quantitative occupancy (BloodChIP / consensus peaks): descriptors report NOT_RUN instead of fitting a NaN column
        tr["occ_score"] = None
    om = dict(occ.get("mf", {}))
    base = dict(assembly="GRCh38", annotation="GENCODE v32 gene records (L.bm_tss), symbols shared by >1 gene_id UNKNOWN", n_resp_tested=len(tested), n_resp=len(resp & tested),
                efficacy_gate="NONE APPLIED", unknown_mapping=int((~tr["occ_ok"]).sum()), n_axis_genes=len(genes))
    files = list(om.pop("files", [])) + list(mf.get("files", []))
    return tr, {**base, **om, **{k: v for k, v in mf.items() if k != "files"}, "files": files}


def resp_ko_panel(tf):
    C = bm_ctx()
    st = C["st"]
    if tf not in set(st.tf):
        raise L.Blocked(f"NO_KO_PANEL_FOR_{tf}")
    x = st[st.tf == tf]
    return set(x.target.str.upper()), set(C["nt"]), dict(response=f"GSE274113 stratified panel rows for {tf} (all rows are responders; max padj {x.padj.max():.3g}); testable = NTC detect >= 5%",
                                                      files=[L.frec(f"{L.BM_TRUTH}/gse274113_stratified_panel.tsv"), L.frec(f"{L.BM_TRUTH}/gene_stats_nt.tsv")])


# GSE274110 24 h GATA1 vs AAVS1 PRO-seq DESeq2 table: tested = finite padj & non-NA unique symbol
def proseq_gata1_sets(path=PROSEQ_GATA1):
    d = pd.read_csv(path, sep="\t")
    sym = d.Symbol.astype("string")
    dup = (sym.duplicated(keep=False) | sym.isna()).to_numpy(bool)
    ok = np.isfinite(d[PADJ24].to_numpy(float)) & ~dup
    su = sym.str.upper()
    return set(su[ok]), set(su[ok & (d[PADJ24].to_numpy(float) < 0.05)]), dict(n_rows=len(d), n_dup_or_missing_symbol=int(dup.sum()), n_finite_padj=int(np.isfinite(d[PADJ24]).sum()))


def truth_bm_gata1_acute(tf, genes, variant="MK"):
    C = bm_ctx()
    tested, resp, st = proseq_gata1_sets()
    tested &= C["nt"]
    if variant == "MK":
        pk, pkf = L.peak_mids("GATA1")
        if pk is None:
            raise L.Blocked("NO_MK_GATA1_PEAKS")
        occ = occ_from_peaks(pk, C["tss"])
        occ["mf"] = dict(mapping=WIN_BM, occupancy=f"ChIP-Atlas hg38 GSE24674 MK {os.path.basename(pkf)} (the BONEMARROW primary rule)", files=[L.frec(pkf)])
    else:
        occ = chipatlas_occ("GATA1", C["tss"])
    return bm_truth(genes, resp & tested, tested, occ, dict(response="GSE274110 24 h GATA1 vs AAVS1 PRO-seq DESeq2 padj<.05; tested = finite padj, unique non-NA symbol, NTC detect >= 5%",
                                                              proseq_stats=st, files=[L.frec(PROSEQ_GATA1), L.frec(L.BM_GTF)]))


# Return bound-gene sets from region-to-gene mappings.
def bloodchip_bound(reg, gmap, cols):
    rid = pd.Index(reg.index)
    arr = {c: reg[c].to_numpy(bool) for c in cols}
    per = {c: set() for c in cols}
    for g, s in zip(gmap.Gene, gmap.Regions):
        ix = rid.get_indexer([r.strip() for r in str(s).split(",")])
        ix = ix[ix >= 0]
        for c in cols:
            if ix.size and arr[c][ix].any():
                per[c].add(str(g).upper())
    return set().union(*per.values()), per


def bloodchip_occ(tf):
    name = BC_ALIAS.get(tf, tf)
    p_reg, p_map = f"{BLOODCHIP}/regions_by_sample.tsv", f"{BLOODCHIP}/gene-map-to-region.tsv"
    head = pd.read_csv(p_reg, sep="\t", nrows=0).columns
    cols = [f"{p}_{name}" for p in BC_POPS if f"{p}_{name}" in head]
    if f"HSC_{name}" not in cols:
        raise L.Blocked(f"BLOODCHIP_HAS_NO_HSC_COLUMN_FOR_{tf}")
    reg = pd.read_csv(p_reg, sep="\t", usecols=["Region"] + cols).set_index("Region").astype(bool)
    gmap = pd.read_csv(p_map, sep="\t")
    union, per = bloodchip_bound(reg, gmap, cols)
    return dict(prim=per[f"HSC_{name}"], sens=union, dist=None, score=None, mappable=set(gmap.Gene.astype(str).str.upper()), dup=set(),
                mf=dict(mapping=("DESCRIPTIVE_DOMAIN_OCCUPANCY: BloodChIP any author-mapped (GREAT) region TRUE in HSC-MPP", "DESCRIPTIVE_DOMAIN_OCCUPANCY: union of HSC/CMP/GMP/MEP"),
                        occupancy_kind="DESCRIPTIVE_DOMAIN_OCCUPANCY (regulatory-domain, not TSS-proximal)", inherited_rule="HSC-MPP primary / 4-population union sensitivity INHERITED from ERG_TRUTH_PIPELINE_V2_FROZEN_2026_09_06, not chosen for these TFs",
                        assembly="GRCh38 (BloodChIP Xtra; no coordinates on disk)",
                        occupancy=f"BloodChIP Xtra columns {cols}; n_regions_TRUE={ {c: int(reg[c].sum()) for c in cols} }", files=[L.frec(p_reg), L.frec(p_map)]))


def truth_bm_bloodchip(tf, genes):
    resp, tested, mf = resp_ko_panel(tf)
    return bm_truth(genes, resp & tested, tested, bloodchip_occ(tf), mf)


def truth_bm_chipatlas(tf, genes):
    resp, tested, mf = resp_ko_panel(tf)
    return bm_truth(genes, resp & tested, tested, chipatlas_occ(tf, bm_ctx()["tss"]), mf)


def truth_bm_bcl11a(tf, genes, window="EARLY"):
    p = BCL_DE[window]
    if not os.path.exists(p):
        raise L.Blocked(f"NO_BCL11A_DE {p} (run prep-bcl11a-de)")
    if not os.path.exists(BCL_PEAKS_HG38):
        raise L.Blocked(f"NO_BCL11A_PEAKS {BCL_PEAKS_HG38} (run prep-bcl11a-peaks)")
    rc, rp = verified_receipt(p), verified_receipt(BCL_PEAKS_HG38)
    de = pd.read_csv(p, sep="\t")
    C = bm_ctx()
    ok = de.padj.notna()
    tested, resp = set(de.gene[ok].str.upper()) & C["nt"], set(de.gene[ok & (de.padj < 0.05)].str.upper())
    occ = occ_from_peaks(read_peaks(BCL_PEAKS_HG38, score_col=4), C["tss"])
    occ["mf"] = dict(mapping=WIN_BM, occupancy="GSE104676 HUDEP-2 BCL11A CUT&RUN 60 min, peaks in >=2 of 3 replicates (hg19 -> hg38 liftOver)",
                     files=[L.frec(BCL_PEAKS_HG38), L.frec(BCL_PEAKS_HG38.replace(".bed", ".receipt.json"))])
    return bm_truth(genes, resp & tested, tested, occ, dict(response=f"GSE194345 BCL11A dTAG PRO-seq HUDEP-2 {window}: pydeseq2 padj<.05, tested = non-NA padj & NTC detect >= 5% (rule R3)",
                                                            files=[L.frec(p), L.frec(p.replace(".tsv", ".receipt.json"))],
                                                            prep_script_sha_matches_current=bool(rc["prep_script_sha_matches_current"] and rp["prep_script_sha_matches_current"])))


# KLF1_B (mouse, mm9 refGene)
# rg: refGene frame (header=None ints)
def canonical_tss_mm9(rg):
    rg = rg[~rg[2].astype(str).str.contains("_")].copy()
    rg["len"] = rg[5] - rg[4]
    rg = rg.sort_values([12, "len", 4, 1], ascending=[True, False, True, True]).groupby(12).head(1)
    rg["sym"] = rg[12].astype(str).str.upper()
    rg = rg[~rg.sym.duplicated(keep=False)]
    rg["tss0"] = np.where(rg[3] == "+", rg[4], rg[5] - 1)
    return rg[["sym", 2, "tss0"]].rename(columns={2: "chrom"}).reset_index(drop=True)


# Summit = peak midpoint -> unique nearest canonical TSS (nearest_unique: exact ties unassigned) within 10 kb
def eklf_gene_features(tss, pk):
    dist = {s: np.inf for s in tss.sym}
    sig = {s: -np.inf for s in tss.sym}
    for c, g in pk.groupby("chrom"):
        t = tss[tss.chrom == c].sort_values("tss0")
        if not len(t):
            continue
        arr, sym = t.tss0.to_numpy(np.int64), t.sym.to_numpy()
        pos, sc = g.mid.to_numpy(np.int64), g.score.to_numpy(float)
        a = L.nearest_unique(pos, arr, 10_000)
        m = a >= 0
        for sy, d, sg in zip(sym[a[m]], np.abs(pos[m] - arr[a[m]]), sc[m]):
            dist[sy] = min(dist[sy], float(d))
            if np.isfinite(sg):
                sig[sy] = max(sig[sy], float(sg))
    F = pd.DataFrame(dict(nearest_summit_tss_dist=pd.Series(dist).replace(np.inf, np.nan), max_signal_10kb=pd.Series(sig).replace(-np.inf, 0.0)))
    F["BOUND_PROM"] = F.nearest_summit_tss_dist <= 1000
    F["BOUND_10KB"] = F.nearest_summit_tss_dist <= 10_000
    return F


def truth_klf1_b(tf, genes):
    if not os.path.exists(KLF1_DE):
        raise L.Blocked(f"NO_KLF1_DE {KLF1_DE} (run prep-klf1)")
    rc = verified_receipt(KLF1_DE)
    de = pd.read_csv(KLF1_DE, sep="\t")
    de = de[de.padj.notna()].copy()
    de["sym"] = de.gene.astype(str).str.upper()
    de = de[~de.sym.duplicated(keep=False)]
    de["resp"] = de.padj < 0.05
    T = de.set_index("sym")
    tss = canonical_tss_mm9(pd.read_csv(REFGENE_MM9, sep="\t", header=None))
    pk = read_peaks(EKLF_BED, score_col=3, sep=r"\s+")
    tr = L.truth_klf1_frames(genes, T, eklf_gene_features(tss, pk))
    mf = dict(mapping=("+-1kb unique-nearest canonical TSS (mm9 refGene)", "+-10kb"), assembly="mm9 (symbol join to the mm10 GATA1 axis; NO coordinate liftover, as RFX6)",
              response=f"GSE33979 Klf1-/- vs +/+ E14.5 fetal liver 3v3, pydeseq2 ~genotype, padj<.05 (rule R4); tested = non-NA padj & unique upper-case symbol; n_tested={len(T)} n_resp={int(T.resp.sum())}",
              occupancy=f"GSE20478 EKLF peaks ({len(pk)}), summit = midpoint, score = BED col 4", prep_script_sha_matches_current=rc["prep_script_sha_matches_current"], n_canonical_tss=len(tss), n_ambiguous_upper_unknown=tr["n_ambiguous_upper"],
              files=[L.frec(KLF1_DE), L.frec(KLF1_DE.replace(".tsv", ".receipt.json")), L.frec(EKLF_BED), L.frec(REFGENE_MM9)], unknown_mapping=int((~tr["occ_ok"]).sum()))
    return tr, mf


# truth-side data preparation (run once on a server)
def refgene_to_gtf(rg, out):
    n = 0
    with open(out, "w") as fh:
        for r in rg[~rg[2].astype(str).str.contains("_")].itertuples(index=False):
            for a, b in zip([int(x) for x in str(r[9]).strip(",").split(",")], [int(x) for x in str(r[10]).strip(",").split(",")]):
                fh.write(f'{r[2]}\trefGene\texon\t{a + 1}\t{b}\t.\t{r[3]}\t.\tgene_id "{r[12]}"; transcript_id "{r[1]}";\n')
                n += 1
    return n


# counts: samples x genes int DataFrame; returns results_df (padj NaN = independent filtering / outlier / prefiltered) reindexed to all genes
def run_deseq(counts, meta, design, contrast):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    dds = DeseqDataSet(counts=counts.loc[:, counts.sum(0) >= 10], metadata=meta, design=design, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=contrast, quiet=True)
    st.summary()
    return st.results_df.reindex(counts.columns)


def _de_table(res):
    return res.rename_axis("gene").reset_index().rename(columns={"log2FoldChange": "log2FC"})[["gene", "baseMean", "log2FC", "padj"]]


def prep_klf1(args):
    tdir = f"{X_DIR}/truth"
    os.makedirs(tdir, exist_ok=True)
    gtf = f"{tdir}/refGene_mm9_exons.gtf"
    n = refgene_to_gtf(pd.read_csv(REFGENE_MM9, sep="\t", header=None), gtf)
    hts = str(Path(sys.executable).parent / "htseq-count")
    samples = [(g, s) for s, gs in KLF1_BAMS.items() for g in gs]
    bam = lambda g: f"{D}/klf1_GSE33979_GSE20478/bam/{g}.bam"
    done = lambda g: os.path.exists(f"{tdir}/{g}.htseq.txt") and os.path.getsize(f"{tdir}/{g}.htseq.txt") > 0      # htseq-count writes its table at the very end: non-empty == complete (resume)
    procs = {g: subprocess.Popen([hts, "-f", "bam", "-r", "pos", "-s", "no", "-m", "union", "-t", "exon", "-i", "gene_id", "-a", "10", "--nonunique", "none", bam(g), gtf],
                                 stdout=open(f"{tdir}/{g}.htseq.txt", "w"), stderr=open(f"{tdir}/{g}.htseq.err", "w")) for g, _ in samples if not done(g)}
    for g, p in procs.items():
        if p.wait() != 0:
            raise SystemExit(f"htseq-count failed for {g}: see {tdir}/{g}.htseq.err")
    cnt = pd.concat({g: pd.read_csv(f"{tdir}/{g}.htseq.txt", sep="\t", header=None, index_col=0)[1] for g, _ in samples}, axis=1)
    cnt = cnt[~cnt.index.str.startswith("__")]
    cnt.to_csv(f"{tdir}/klf1_GSE33979_counts.tsv", sep="\t")
    meta = pd.DataFrame(dict(genotype=[s for _, s in samples]), index=[g for g, _ in samples])
    out = _de_table(run_deseq(cnt.T.astype(int), meta, "~genotype", ["genotype", "KO", "WT"]))
    out.to_csv(KLF1_DE, sep="\t", index=False)
    rc = dict(rule="R4: pydeseq2 ~genotype KO vs WT, prefilter sum counts>=10, htseq-count union -s no -a 10 --nonunique none over the exon union of UCSC refGene mm9 (gene_id = name2)",
              samples=dict(KLF1_BAMS), n_exons=n, n_genes_counted=int(len(cnt)), n_tested=int(out.padj.notna().sum()), n_padj_lt_05=int((out.padj < .05).sum()), out_sha256=L.sha_file(KLF1_DE),
              gtf_sha256=L.sha_file(gtf), counts_sha256=L.sha_file(f"{tdir}/klf1_GSE33979_counts.tsv"), script_sha256=L.sha_file(__file__), bams={g: L.sha_file(bam(g)) for g, _ in samples})
    L.atomic_json(KLF1_DE.replace(".tsv", ".receipt.json"), rc)
    print({k: rc[k] for k in ("n_genes_counted", "n_tested", "n_padj_lt_05")})


def prep_bcl11a_de(args):
    z = np.load(BCL_NPZ, allow_pickle=True)
    sm = pd.read_csv(BCL_MAP, sep="\t").set_index("run").loc[[str(r) for r in z["runs"]]]
    sym = pd.Series(z["gene_symbol"].astype(str))
    dup = sym.duplicated(keep=False).to_numpy()
    raw = pd.DataFrame(np.rint(z["raw_body500"]).astype(int), index=sm.index, columns=sym.to_numpy()).loc[:, ~dup]
    os.makedirs(f"{X_DIR}/truth", exist_ok=True)
    for name, times, design in (("EARLY", [3], "~clone + treatment"), ("LATE", [6, 12, 24], "~clone + time + treatment")):
        m = sm[sm.time_h.isin(times) & sm.treatment.isin(["DMSO", "dTAG"])].copy()
        m["time"] = m.time_h.astype(str)
        out = _de_table(run_deseq(raw.loc[m.index], m[["clone", "treatment", "time"]], design, ["treatment", "dTAG", "DMSO"]))
        p = BCL_DE[name]
        out.to_csv(p, sep="\t", index=False)
        L.atomic_json(p.replace(".tsv", ".receipt.json"), dict(rule="R3", design=design, times_h=times, samples=list(m.index), n_dup_symbols_dropped=int(dup.sum()), counts="raw_body500 rounded",
                                                               counts_sha256=L.sha_file(BCL_NPZ), sample_map_sha256=L.sha_file(BCL_MAP), n_tested=int(out.padj.notna().sum()),
                                                               n_padj_lt_05=int((out.padj < .05).sum()), out_sha256=L.sha_file(p), script_sha256=L.sha_file(__file__)))
        print(name, "n_tested", int(out.padj.notna().sum()), "n_padj<.05", int((out.padj < .05).sum()))


def prep_bcl11a_peaks(args):
    os.makedirs(f"{X_DIR}/occupancy", exist_ok=True)
    sets = [read_peaks(f"{BCL_DIR}/{g}_peaks.narrowPeak.gz", score_col=6) for g in BCL_REPS]
    cons = consensus_depth(sets, 2)
    lifted, n_un = lift_hg19_hg38(cons, f"{X_DIR}/occupancy/bcl11a_tmp")
    lifted = lifted[lifted.chrom.isin(L.PRIMARY_CHR)]
    lifted.assign(name=".")[["chrom", "s", "e", "name"]].to_csv(BCL_PEAKS_HG38, sep="\t", header=False, index=False)
    L.atomic_json(BCL_PEAKS_HG38.replace(".bed", ".receipt.json"), dict(rule="R3", n_rep_peaks=[len(s) for s in sets], n_consensus_hg19=len(cons), n_unmapped_liftover=n_un, n_hg38_primary=len(lifted),
                                                                       replicates=BCL_REPS, chain_sha256=L.sha_file(CHAIN), liftover_sha256=L.sha_file(LIFTOVER), out_sha256=L.sha_file(BCL_PEAKS_HG38),
                                                                       inputs={g: L.sha_file(f"{BCL_DIR}/{g}_peaks.narrowPeak.gz") for g in BCL_REPS}, script_sha256=L.sha_file(__file__)))
    print("consensus hg19", len(cons), "unmapped", n_un, "hg38", len(lifted))


# K562 v4 (calibrated empirical-null response) systems
K562_EMP_STATE = os.environ.get("GB_K562_EMP_STATE", f"{R}/results/k562_truth_v4_emp/k562_state.npz")                     # env override = smoke/dry runs only
K562_EMP_ALLNULL_STATE = os.environ.get("GB_K562_EMP_ALLNULL_STATE", f"{R}/results/k562_truth_v4_emp_allnull/k562_state.npz")
K562_ALIAS = ("K562_EMP", "K562_EMP_ALLNULL")
_K562_CACHES = {}                                                              # one truth_k562 cache per system: never shared, so the two v4 states cannot cross-contaminate


# L.truth_k562 (ChIP mapping, MANE TSS, blacklist, A/B/C/D construction all reused) against another state file
def truth_k562_v4(tf, genes, key, state_path, what):
    old = L.K562_STATE
    L.K562_STATE = state_path
    try:
        tr, mf = L.truth_k562(tf, genes, _K562_CACHES.setdefault(key, {}))
    finally:
        L.K562_STATE = old
    mf["response"] = f"{state_path}: {what}"
    mf["k562_state_file"], mf["k562_state_system"] = state_path, key
    mf["deviation_response_test"] = "response = calibrated empirical per-gene null (results/breadth_truth/K562_EMP_SHARD) replaces the v3 per-cell Welch (failed the NT-vs-NT null gate)"
    return tr, mf


# CARDIAC (mouse cardiogenesis; Luna-Zurita 2016 truth x de Soysa 2019 WT scRNA)
CARDIAC_DIR = f"{R}/results/cardiac_truth_v1"
CARDIAC_TABLE = f"{CARDIAC_DIR}/truth_gene_table.tsv"
CARDIAC_KEY = {"TBX5": "Tbx5", "NKX2-5": "Nkx2-5", "GATA4": "Gata4"}                  # lattice source name (upper) -> truth-table column stem
WIN_CARDIAC = ("+-2kb any refGene mm9 TSS (WT-CM ChIP-exo footprint midpoint)", "+-10kb")


# fail closed: the truth table must be the one the receipt hashed, and the frozen declaration + addendum on disk must be the ones the receipt hashed
def cardiac_verify():
    rp = f"{CARDIAC_DIR}/receipt.json"
    if not (os.path.exists(rp) and os.path.exists(CARDIAC_TABLE)):
        raise L.Blocked(f"NO_CARDIAC_TRUTH {CARDIAC_DIR} (run scratch/cardiac_truth_build.py de|bound|table)")
    rc = json.load(open(rp))
    if rc["output_files_sha256"].get("truth_gene_table.tsv") != L.sha_file(CARDIAC_TABLE):
        raise L.Blocked(f"STALE_OR_ALTERED {CARDIAC_TABLE}: sha256 != receipt")
    fz = f"{CARDIAC_DIR}/CARDIAC_TRUTH_V1_FROZEN.md"
    if not os.path.exists(fz) or rc["frozen_declaration_sha256"] != L.sha_file(fz):
        raise L.Blocked(f"FROZEN_DECLARATION_MISSING_OR_ALTERED {fz}")
    return rc


# truth table joined to the axis by exact symbol (no upper-casing; a symbol carried by >1 gene_id, or absent from the table, is unknown)
def cardiac_frame(genes):
    T = pd.read_csv(CARDIAC_TABLE, sep="\t").set_index("gene")
    return T.reindex(np.asarray(genes).astype(str))


def truth_cardiac(tf, genes, occupancy_only=False):
    k = CARDIAC_KEY.get(tf.upper())
    if k is None or (k == "Gata4") != occupancy_only:
        raise L.Blocked(f"NO_CARDIAC_{'OCCUPANCY_ONLY' if occupancy_only else 'KO'}_TRUTH_FOR_{tf}" + ("" if k != "Gata4" or occupancy_only else ": Gata4 has ChIP-exo but NO KO -> system CARDIAC_GATA4_OCC (descriptive)"))
    rc = cardiac_verify()
    Tg = cardiac_frame(genes)
    ook = Tg.refgene_mapped.eq(True).to_numpy()
    prim, sens = Tg[f"bound_{k}"].eq(True).to_numpy(), Tg[f"bound10_{k}"].eq(True).to_numpy()
    if occupancy_only:
        # no response layer exists
        rok, resp = ook.copy(), np.zeros(len(ook), bool)
    else:
        rok, resp = Tg[f"{k}_tested"].eq(True).to_numpy(), Tg[f"{k}_resp"].eq(True).to_numpy()
    tr = L.mk_truth(rok, resp, ook, prim, sens, occ_dist=L.numcol(Tg, f"dist_{k}"), occ_score=None)
    cc = rc["class_counts"].get(k, {})
    mf = dict(mapping=WIN_CARDIAC, assembly="mm9 refGene symbols joined to the mm10/Ensembl-102 scRNA axis by EXACT symbol (no liftover, no case-insensitive rescue); refGene symbols without an exact Ensembl match are unmapped",
              response=("NONE (Gata4 has no KO): resp_ok=occ_ok placeholder, resp all False -> response layer TOO_FEW by construction" if occupancy_only else
                        f"Luna-Zurita 2016 CM-stage RNA-seq {k}KO vs WT 3v3 (salmon -> pydeseq2 ~genotype, padj<.05 & |log2FC|>=.5; tested = non-NA padj); in-vitro (mESC-derived CM) -> in-vivo (embryo scRNA) transfer"),
              occupancy=f"Luna-Zurita 2016 WT-CM ChIP-exo average footprints ({k}); midpoint within +-2 kb of any refGene TSS (sensitivity +-10 kb); occ_dist = nearest footprint midpoint to the gene's TSS",
              frozen_declaration_sha256=rc["frozen_declaration_sha256"], truth_builder_sha256=rc["builder_sha256"], class_counts_all_genes_of_table=cc,
              unknown_mapping=int((~tr["occ_ok"]).sum()), n_axis_genes=len(genes), n_resp_tested=int(tr["resp_ok"].sum()),
              files=[L.frec(CARDIAC_TABLE), L.frec(f"{CARDIAC_DIR}/receipt.json"), L.frec(f"{CARDIAC_DIR}/CARDIAC_TRUTH_V1_FROZEN.md")])
    return tr, mf

RETINA_DIR = f"{R}/results/retina_truth_v1"
RETINA_TABLE = f"{RETINA_DIR}/truth_gene_table.tsv"
RETINA_SENS_TABLE = f"{RETINA_DIR}/truth_sens_NRL_P10_3v2.tsv"                                # SENSITIVITY: NRL_P10 3 WT vs 2 KO (SRR2936861 dropped); scratch/retina_truth_nrl3v2.py
RETINA_KEY = {"NRL": ("NRL_P10", "Nrl"), "CRX": ("CRX_P10", "Crx")}                          # lattice source name (upper) -> (response contrast stem, occupancy stem)
WIN_RETINA = ("+-2kb any TSS (NRL: Ens102 transcript TSS x P10 IDR CUT&RUN peaks; CRX: refGene mm9 TSS x MACS3 WT ChIP peaks)", "+-10kb")


# fail closed (as cardiac_verify): the truth table must be the one the receipt hashed and the frozen declaration on disk the one the receipt hashed
def retina_verify(sens=False):
    rp = f"{RETINA_DIR}/receipt.json"
    if not (os.path.exists(rp) and os.path.exists(RETINA_TABLE)):
        raise L.Blocked(f"NO_RETINA_TRUTH {RETINA_DIR} (run scratch/retina_truth_build.py de|bound|table)")
    rc = json.load(open(rp))
    if rc["output_files_sha256"].get("truth_gene_table.tsv") != L.sha_file(RETINA_TABLE):
        raise L.Blocked(f"STALE_OR_ALTERED {RETINA_TABLE}: sha256 != receipt")
    fz = f"{RETINA_DIR}/RETINA_TRUTH_V1_FROZEN.md"
    if not os.path.exists(fz) or rc["frozen_declaration_sha256"] != L.sha_file(fz):
        raise L.Blocked(f"FROZEN_DECLARATION_MISSING_OR_ALTERED {fz}")
    if sens:
        sp = f"{RETINA_DIR}/sens_NRL_P10_3v2_receipt.json"
        if not (os.path.exists(sp) and os.path.exists(RETINA_SENS_TABLE)):
            raise L.Blocked(f"NO_RETINA_SENS_TRUTH {RETINA_SENS_TABLE} (run scratch/retina_truth_nrl3v2.py)")
        sr = json.load(open(sp))
        if sr["output_files_sha256"].get("truth_sens_NRL_P10_3v2.tsv") != L.sha_file(RETINA_SENS_TABLE) or sr["primary_table_sha256"] != L.sha_file(RETINA_TABLE):
            raise L.Blocked(f"STALE_OR_ALTERED {RETINA_SENS_TABLE}: sha256 != sens receipt")
    return rc


# Join labels by exact symbol; sens=True uses the NRL 3v2 response table.
def truth_retina(tf, genes, sens=False):
    kk = RETINA_KEY.get(tf.upper())
    if kk is None or (sens and tf.upper() != "NRL"):
        raise L.Blocked(f"NO_RETINA_{'SENS_' if sens else ''}TRUTH_FOR_{tf}")
    name, k = kk
    rc = retina_verify(sens)
    g = np.asarray(genes).astype(str)
    Tg = pd.read_csv(RETINA_TABLE, sep="\t").set_index("gene").reindex(g)
    ook = Tg[f"occ_ok_{k}"].eq(True).to_numpy()
    prim, sn = Tg[f"bound_{k}"].eq(True).to_numpy(), Tg[f"bound10_{k}"].eq(True).to_numpy()
    if sens:
        Sg = pd.read_csv(RETINA_SENS_TABLE, sep="\t").set_index("gene").reindex(g)
        rok, resp = Sg["NRL_P10_3v2_tested"].eq(True).to_numpy(), Sg["NRL_P10_3v2_resp"].eq(True).to_numpy()
        rname = "NRL_P10_3v2"
    else:
        rok, resp = Tg[f"{name}_tested"].eq(True).to_numpy(), Tg[f"{name}_resp"].eq(True).to_numpy()
        rname = name
    tr = L.mk_truth(rok, resp, ook, prim, sn, occ_dist=L.numcol(Tg, f"dist_{k}"), occ_score=None)
    mf = dict(mapping=WIN_RETINA, assembly="NRL: Ensembl-102 GRCm38 transcript TSS (mm10 IDR peaks); CRX: mm9 refGene TSS joined to the mm10/Ensembl-102 axis by EXACT symbol (no liftover, no case-insensitive rescue)",
              response=(f"{rname}: " + ("Nrl-/- vs WT GSE74660 P10 sorted photoreceptors, 3 WT vs 2 KO (SRR2936861 dropped; sensitivity)" if sens else
                        ("Nrl-/- vs WT GSE74660 P10 sorted photoreceptors 3v3" if k == "Nrl" else "CrxR90W/W (loss-of-function allele, not a null) vs WT GSE223439 P10 3v3")) +
                        "; salmon -> pydeseq2 ~genotype, padj<.05 & |log2FC|>=.5; tested = non-NA padj; frozen truth results/retina_truth_v1"),
              occupancy=("Nrl P10 CUT&RUN GSE197420 IDR peaks (authors' release); midpoint within +-2 kb of any Ens102 transcript TSS (sens +-10 kb)" if k == "Nrl" else
                         "Crx WT ChIP-seq GSE20012 (8 wk; age mismatch stated) MACS3 q<.01 peaks; midpoint within +-2 kb of any refGene mm9 TSS (sens +-10 kb)"),
              frozen_declaration_sha256=rc["frozen_declaration_sha256"], truth_builder_sha256=rc["builder_sha256"], class_counts_all_genes_of_table=rc["class_counts"].get(name, {}),
              unknown_mapping=int((~tr["occ_ok"]).sum()), n_axis_genes=len(g), n_resp_tested=int(tr["resp_ok"].sum()),
              files=[L.frec(RETINA_TABLE), L.frec(f"{RETINA_DIR}/receipt.json"), L.frec(f"{RETINA_DIR}/RETINA_TRUTH_V1_FROZEN.md")] + ([L.frec(RETINA_SENS_TABLE), L.frec(f"{RETINA_DIR}/sens_NRL_P10_3v2_receipt.json")] if sens else []))
    return tr, mf



# registry + extension point
EXTRA = {
    "KLF1_B": dict(block="ERYTHROID", axis="GATA1", tfs=["KLF1"], base="KLF1", truth=truth_klf1_b, win=("+-1kb unique-nearest canonical TSS (mm9)", "+-10kb"), role="PRIMARY"),
    "BM_GATA1_ACUTE": dict(block="BONEMARROW", axis="BONEMARROW", tfs=["GATA1"], base="BONEMARROW", truth=lambda tf, g: truth_bm_gata1_acute(tf, g, "MK"), win=WIN_BM, role="PRIMARY"),
    "BM_GATA1_ACUTE_T1": dict(block="BONEMARROW", axis="BONEMARROW", tfs=["GATA1"], base="BONEMARROW", truth=lambda tf, g: truth_bm_gata1_acute(tf, g, "T1"), win=WIN_BM,
                                role="SENSITIVITY (alternative occupancy for BM_GATA1_ACUTE, same response; never pooled with it)"),
    "BM_BLOODCHIP": dict(block="BONEMARROW", axis="BONEMARROW", tfs=["GATA2", "LMO2", "RUNX1", "SPI1", "TAL1"], base="BONEMARROW", truth=truth_bm_bloodchip,
                         win=("DESCRIPTIVE_DOMAIN_OCCUPANCY: BloodChIP any author-mapped (GREAT) region TRUE in HSC-MPP", "DESCRIPTIVE_DOMAIN_OCCUPANCY: union of HSC/CMP/GMP/MEP"), role="DESCRIPTIVE_DOMAIN_OCCUPANCY"),
    "BM_CHIPATLAS": dict(block="BONEMARROW", axis="BONEMARROW", tfs=list(CA_TFS), base="BONEMARROW", truth=truth_bm_chipatlas, win=WIN_BM, role="PRIMARY"),
    "BM_BCL11A_EARLY": dict(block="BONEMARROW", axis="BONEMARROW", tfs=["BCL11A"], base="BONEMARROW", truth=lambda tf, g: truth_bm_bcl11a(tf, g, "EARLY"), win=WIN_BM, role="PRIMARY"),
    "BM_BCL11A_LATE": dict(block="BONEMARROW", axis="BONEMARROW", tfs=["BCL11A"], base="BONEMARROW", truth=lambda tf, g: truth_bm_bcl11a(tf, g, "LATE"), win=WIN_BM, role="PRIMARY"),
    "K562_EMP": dict(block="K562", axis="K562", tfs=L.k562_tfs(), base="K562", win=L.WIN["K562"],
                     role="SENSITIVITY (AMBIGUOUS excluded as LOW_INFORMATION: confident-nonresponder negatives only; same ChIP/universe; never pooled with K562_EMP_ALLNULL)",
                     truth=lambda tf, g: truth_k562_v4(tf, g, "K562_EMP", K562_EMP_STATE, "RESPONDER->UP/DOWN vs CONFIDENT_NONRESPONDER->TESTED_NULL; AMBIGUOUS excluded (LOW_INFORMATION); UNTESTED->UNKNOWN")),
    "K562_EMP_ALLNULL": dict(block="K562", axis="K562", tfs=L.k562_tfs(), base="K562", win=L.WIN["K562"],
                             role="PRIMARY (calibrated empirical-null response; AMBIGUOUS counted as TESTED_NULL = v3 'not significant' semantics)",
                             truth=lambda tf, g: truth_k562_v4(tf, g, "K562_EMP_ALLNULL", K562_EMP_ALLNULL_STATE, "RESPONDER->UP/DOWN vs CONFIDENT_NONRESPONDER + AMBIGUOUS -> TESTED_NULL; UNTESTED->UNKNOWN")),
    "CARDIAC": dict(block="CARDIAC", axis="CARDIAC", tfs=["TBX5", "NKX2-5"], base="CARDIAC", truth=lambda tf, g: truth_cardiac(tf, g), win=WIN_CARDIAC,
                    role="PRIMARY (Tbx5, Nkx2-5; mouse cardiogenesis, in-vitro CM truth -> in-vivo embryo scRNA transfer; K562-style support gate: >=100 bound, >=20 A, >=20 B)"),
    "CARDIAC_GATA4_OCC": dict(block="CARDIAC", axis="CARDIAC", tfs=["GATA4"], base="CARDIAC", truth=lambda tf, g: truth_cardiac(tf, g, occupancy_only=True), win=WIN_CARDIAC,
                              role="DESCRIPTIVE_OCCUPANCY_ONLY (Gata4: ChIP-exo but no KO; response layer undefined -> only score-vs-bound occupancy is evaluable; never pooled with CARDIAC)"),
    "CARDIACV": dict(block="CARDIACV", axis="CARDIACV", tfs=["TBX5", "NKX2-5"], base="CARDIACV", truth=lambda tf, g: truth_cardiac(tf, g), win=WIN_CARDIAC,
                     role="PRIMARY (velocyto build of the de Soysa WT BAMs; Tbx5, Nkx2-5; frozen CARDIAC truth unchanged; same support gate as CARDIAC; focal TFs must pass the 5% spliced-detection floor or CARDIACV is NON-EVALUABLE)"),
    "CARDIACV_GATA4_OCC": dict(block="CARDIACV", axis="CARDIACV", tfs=["GATA4"], base="CARDIACV", truth=lambda tf, g: truth_cardiac(tf, g, occupancy_only=True), win=WIN_CARDIAC,
                               role="DESCRIPTIVE_OCCUPANCY_ONLY (Gata4, velocyto build; never pooled with CARDIACV)"),
    "RETINA": dict(block="RETINA", axis="RETINA", tfs=["NRL", "CRX"], base="RETINA", truth=lambda tf, g: truth_retina(tf, g), win=WIN_RETINA,
                   role="PRIMARY (Clark 2019 postnatal retina scRNA velocyto build x P10 Nrl-/- (GSE74660 3v3) and CrxR90W/W loss-of-function (GSE223439 3v3); NRL IDR CUT&RUN / Crx ChIP occupancy; K562-style support gate: >=100 bound, >=20 A, >=20 B)"),
    "RETINA_NRL_3V2": dict(block="RETINA", axis="RETINA", tfs=["NRL"], base="RETINA", truth=lambda tf, g: truth_retina(tf, g, sens=True), win=WIN_RETINA,
                           role="SENSITIVITY (NRL_P10 re-tested 3 WT vs 2 KO, SRR2936861 dropped; same occupancy/universe; never pooled with RETINA)"),
}
CARDIAC_GATED = ("CARDIAC", "CARDIACV", "RETINA", "RETINA_NRL_3V2")
_ORIG = {}


# same order as L.build_endpoint (scores -> catbank -> truth); scores/catbank come from the base system, only the truth builder differs
def build_endpoint_extra(system, tf, blk, smalln):
    X = EXTRA.get(system)
    if X is None:
        return _ORIG["build_endpoint"](system, tf, blk, smalln)
    cfg = L.SYSTEMS[system]
    if X["base"] == "KLF1":                                                   # a TF without a GB row raises L.Blocked inside scores_prod (reported NOT_RUN); every other error propagates
        genes, V, miss = L.scores_klf1()
        cfiles = {}
    else:
        genes, V, miss, _, cfiles = L.scores_prod(X["base"], tf)
    pre = dict(genes=np.asarray(genes).astype(str), V=V, files={}, meta={}, F1=None)
    gb_sha = L.sha_arr(np.nan_to_num(V[L.GB], nan=-999.0))
    L.catbank_attach(pre, cfg["axis"], tf, smalln)
    tr, mf = X["truth"](tf, genes)                                            # truth is opened below this line only
    mf["lattice_extra_module_sha256"], mf["system_role"] = L.sha_file(__file__), X["role"]
    if miss:
        mf["missing_methods"] = miss
    if X["base"] != "KLF1":
        mf["comparator_files"] = cfiles
    ep = L.assemble(system, tf, pre["genes"], pre["V"], tr, blk, dict(manifest=mf, **pre["meta"]), gb_sha)
    ep["F1"], ep["adj_ok"], ep["files"] = pre["F1"], pre["adj_ok"], pre["files"]
    ep["meta"]["truth_loaded_after"] = ["scores", "catbank"]
    return ep


def register():
    if "build_endpoint" in _ORIG:
        return
    _ORIG["build_endpoint"], _ORIG["module_shas"] = L.build_endpoint, L.module_shas
    for k, X in EXTRA.items():
        L.SYSTEMS[k] = dict(block=X["block"], axis=X["axis"], tfs=X["tfs"])
        L.WIN[k] = X["win"]
    L.build_endpoint = build_endpoint_extra
    _ORIG["gate_of"] = L.gate_of
    L.gate_of = lambda ep, y, rows, mask: _ORIG["gate_of"](dict(ep, system="K562") if ep["system"] in K562_ALIAS + CARDIAC_GATED else ep, y, rows, mask)
    if "CARDIAC" not in L.BLOCKS:
        L.BLOCKS.append("CARDIAC")                                                # own bootstrap block (its own gene axis); appended last so the seeds of the existing blocks are unchanged
    if "CARDIACV" not in L.BLOCKS:
        L.BLOCKS.append("CARDIACV")                                               # appended after CARDIAC so no existing block's bootstrap seed changes
    if "RETINA" not in L.BLOCKS:
        L.BLOCKS.append("RETINA")                                                 # own bootstrap block; appended after CARDIACV so no existing block seed changes
    L.module_shas = lambda: dict(_ORIG["module_shas"](), gb_lattice_extra=L.sha_file(__file__))


def unregister():
    L.build_endpoint, L.module_shas, L.gate_of = _ORIG.pop("build_endpoint"), _ORIG.pop("module_shas"), _ORIG.pop("gate_of")
    if "CARDIAC" in L.BLOCKS:
        L.BLOCKS.remove("CARDIAC")
    if "CARDIACV" in L.BLOCKS:
        L.BLOCKS.remove("CARDIACV")
    if "RETINA" in L.BLOCKS:
        L.BLOCKS.remove("RETINA")
    for k in EXTRA:
        L.SYSTEMS.pop(k)
        L.WIN.pop(k)


# inventory (A/B/C/D counts only -- no score-vs-truth statistic)
_ROWS = {}


# the lattice universe = GB observational universe of the TF's row (as scores_prod)
# TF without a GB row -> axis genes, flagged
def universe(base, tf):
    import gb_cop_alignment as CA
    import gb_real_uniform as RU
    if base == "KLF1":
        return np.asarray(L.scores_klf1()[0]).astype(str), "KLF1 frozen predictions.npz gene axis", True
    if base not in _ROWS:
        rows, eg = RU.gb_raw_rows(base)
        if eg is None and base in ("CARDIAC", "CARDIACV", "RETINA"):                                     # no GB score cache yet (truth inventory before scoring): the fixture gene axis, flagged as not scoreable
            import anndata as ad
            fx = f"{R}/results/canonical_2026_09_18/{base}_canonical.h5ad"
            if not os.path.exists(fx):
                raise L.Blocked(f"NO_CARDIAC_FIXTURE {fx} (run scratch/gb_canonical_cardiac.py build ... --system {base})")
            eg = np.asarray(ad.read_h5ad(fx, backed="r").var_names).astype(str)
        _ROWS[base] = (rows, np.asarray(eg).astype(str))
    rows, eg = _ROWS[base]
    if tf.upper() in rows:
        uni, _ = CA.observational_universe(tf, rows[tf.upper()], eg)
        return eg[uni], "GB observational universe", True
    return eg, "AXIS (no GB row for this TF: NOT scoreable by the lattice)", False


def abcd(tr):
    U = tr["resp_ok"] & tr["occ_ok"]
    d = L.class_counts(dict(resp=tr["resp"], occ=tr["occ"]), U)
    d["n_resp_unknown"], d["n_occ_unknown"] = int((~tr["resp_ok"]).sum()), int((~tr["occ_ok"]).sum())
    for w in ("prim", "sens"):
        d[f"gate_AB_{w}"] = bool(d[f"n_A_{w}"] >= L.MIN_CLASS and d[f"n_B_{w}"] >= L.MIN_CLASS)
        d[f"gate_CD_{w}"] = bool(d[f"n_C_{w}"] >= L.MIN_CLASS and d[f"n_D_{w}"] >= L.MIN_CLASS)
    return d


def inventory(args):
    systems = [s.strip() for s in args.systems.split(",") if s.strip()]
    os.makedirs(f"{X_DIR}/mapping", exist_ok=True)
    rows = []
    for s in systems:
        if s == "K562":
            X = dict(base="K562", tfs=L.k562_tfs(), truth=L.truth_k562, win=L.WIN["K562"], role="Existing lattice-analysis system")
        elif s in EXTRA:
            X = EXTRA[s]
        else:
            raise SystemExit(f"unknown system {s}; known: K562, {', '.join(EXTRA)}")
        mfs, excl = {}, []
        for tf in ([t for t in args.tfs.split(",") if t] or X["tfs"]):
            try:
                genes, uni, gbrow = universe(X["base"], tf)
                tr, mf = X["truth"](tf, genes)
                c = abcd(tr)
                rows.append(dict(system=s, role=X["role"], tf=tf, universe=uni, gb_row=gbrow, **c))
                if not gbrow:
                    excl.append(dict(system=s, tf=tf, reason="NO_GB_ROW on this substrate: truth counts exist but the lattice cannot score this TF (NOT_RUN GB_ROW_MISSING)"))
                mfs[tf] = dict(mf, universe=uni, gb_row=gbrow, counts=c)
                print(f"[inventory] {s}/{tf} gb_row={gbrow} rows={c['n_rows']} prim A/B/C/D={c['n_A_prim']}/{c['n_B_prim']}/{c['n_C_prim']}/{c['n_D_prim']} "
                      f"sens={c['n_A_sens']}/{c['n_B_sens']}/{c['n_C_sens']}/{c['n_D_sens']} gateAB={c['gate_AB_prim']} gateCD={c['gate_CD_prim']}", flush=True)
            except L.Blocked as e:
                rows.append(dict(system=s, role=X["role"], tf=tf, status=f"NOT_READY: {e}"))
                excl.append(dict(system=s, tf=tf, reason=f"NOT_READY: {e}"))
                print(f"[inventory] {s}/{tf} NOT_READY {e}", flush=True)
        L.atomic_json(f"{X_DIR}/mapping/{s}_inventory_manifest.json", dict(system=s, module_sha256=L.sha_file(__file__), lattice_sha256=L.sha_file(L.__file__), windows=X["win"], system_role=X["role"],
                                                                           note="inventory: truth-side counts only; NO score-vs-truth statistic computed", exclusions=excl, endpoints=mfs))
        L.atomic_tsv(pd.DataFrame(excl, columns=["system", "tf", "reason"]), f"{X_DIR}/mapping/{s}_exclusions.tsv")
        print(f"[inventory] {s} exclusions ({len(excl)}): " + "; ".join(f"{e['tf']}: {e['reason'][:90]}" for e in excl), flush=True)
    tag = hashlib.sha1(",".join(systems).encode()).hexdigest()[:10]                                     # hash, not a truncated system list
    L.atomic_tsv(pd.DataFrame(rows), f"{X_DIR}/mapping/inventory_{tag}.tsv")
    L.atomic_json(f"{X_DIR}/mapping/inventory_{tag}.systems.json", dict(systems=systems))


# selftest (synthetic, no data)
def selftest():
    import tempfile
    rng = np.random.default_rng(0)
    P = lambda *iv, c="chr1": pd.DataFrame(dict(chrom=c, s=[a for a, _ in iv], e=[b for _, b in iv]))
    # Check consensus intervals at depths 1 and 2.
    assert consensus_depth([P((0, 10), (20, 30)), P((5, 25)), P((8, 12))], 2)[["s", "e"]].values.tolist() == [[5, 12], [20, 25]]
    assert consensus_depth([P((0, 10), (10, 20)), P((100, 110))], 2).empty and consensus_depth([P((0, 10)), P((10, 20))], 2).empty
    assert consensus_depth([P((0, 10), (20, 30)), P((5, 25))], 1)[["s", "e"]].values.tolist() == [[0, 30]]
    # overlap_within == brute force of gb_bm_sweep.bound_set (inclusive comparisons)
    pk = pd.DataFrame(dict(chrom="chr1", s=rng.integers(0, 100_000, 300), w=rng.integers(1, 400, 300)))
    pk["e"] = pk.s + pk.w
    tss = {f"G{i}": ("chr1" if i % 7 else "chr2", int(rng.integers(0, 100_000))) for i in range(200)}
    ref = {g for g, (c, t) in tss.items() if c == "chr1" and any(e >= t - 2000 and s <= t + 2000 for s, e in zip(pk.s, pk.e))}
    assert overlap_within(tss, pk, 2000) == ref and len(ref) > 5
    # tier selection (R1): T1 pooled across cell types, hg19 excluded, perturbation + Input/IgG
    # excluded but scramble/vehicle/untreated controls kept, K-562
    rows = [("S1", "hg38", "TF", "Blood", "CD34+", "x"), ("S2", "hg38", "TF", "Blood", "Erythroid Cells", "y"), ("S3", "hg38", "TF", "Blood", "CD34+", "z HIC2 KO rep1"),
            ("S4", "hg38", "TF", "Blood", "K-562", "k"), ("S5", "hg19", "TF", "Blood", "CD34+", "old"), ("S6", "hg38", "T4", "Blood", "Jurkat", "j"),
            ("S7", "hg38", "T4", "Blood", "THP-1", "t1"), ("S8", "hg38", "T4", "Blood", "THP-1", "t2"), ("S9", "hg38", "T4", "Other", "HeLa", "h"),
            ("S10", "hg38", "T2", "Blood", "HUDEP-2", "h2"), ("S11", "hg38", "T2", "Blood", "K-562", "k2"), ("S12", "hg38", "TF", "Blood", "CD34+", "Scramble control TF ChIP-seq rep1"),
            ("S13", "hg38", "TF", "Blood", "CD34+", "Input (TF) - baseline"), ("S14", "hg38", "TF", "Blood", "CD34+", "TF ChIP Untreated")]
    E = pd.DataFrame([(a, b, "TFs and others", c, d, e, f) for a, b, c, d, e, f in rows], columns=["srx", "genome", "cls", "antigen", "ctclass", "celltype", "title"])
    s = select_chipatlas(E, "TF")
    assert s["tier"] == "T1" and s["srx"] == ["S1", "S12", "S14", "S2"] and s["excluded"] == {"S3": "perturbation", "S13": "input_or_IgG_background"}, s
    assert select_chipatlas(E, "TF", v1=True)["srx"] == ["S1", "S2"]                                          # the v1 rule dropped the scramble/untreated controls (before/after record)
    for t, bad in (("CD36 shbrg1 TAL1", 1), ("CD36 shLuc TAL1", 0), ("K562 DMSO Vehicle Control PU.1 ChIP-seq Rep 1", 0), ("ChIP-seq K562 Hub KO GATA1 rep1", 1), ("MOLM13 dI8 4hr dTAG SPI1", 1),
                   ("Gfi1b ChIP non Wnt-Treated", 0), ("Gfi1b ChIP Wnt-Treated", 1), ("MUTZ3 - MYB ChIP-seq  - Untreated", 0), ("RUNX1 ChIP-seq in NT siRNA-transfected SEM cells", 0),
                   ("MV411 1hr CRYBMIM MYB rep1", 1), ("MV411 1hr PBS MYB rep1", 0), ("PU.1 IFN-g rep1", 1), ("Stanford ChipSeq K562 GATA1 IgG-mus", 1), ("MYB ChIP TG3 peptide control replicate 1", 0),
                   ("ChIP-seq K562 enCRISPRi-KL sgHS2 GATA1 rep1", 1),
                   ("GATA1 ChIP Let-7 KD rep1", 1), ("GATA1 ChIP shRNA knock-down", 1), ("MYB ChIP MYBMIM 1h", 1), ("GATA2 ChIP ACY-957 24h", 1), ("GATA2 RE CB ChIP", 1), ("RUNX1 RE N822K", 1),
                   ("Case1 Ery GATA1 ChIP", 0), ("Ery GATA1 control", 0)):
        assert bool(excl_reason(t)) == bool(bad), t
    E2 = pd.concat([E, pd.DataFrame([("S20", "hg38", "TFs and others", "TF", "Blood", "CD34+", "manual-list experiment with an innocuous title")], columns=E.columns)], ignore_index=True)
    SRX_EXCLUDE["S20"] = "selftest"
    try:
        s2 = select_chipatlas(E2, "TF")
        assert "S20" not in s2["srx"] and s2["excluded"]["S20"].startswith("perturbation (manual SRX exclusion")
    finally:
        SRX_EXCLUDE.pop("S20")
    assert not set(SRX_EXCLUDE) & set(select_chipatlas(E2, "TF")["srx"]) and JUDGMENT_KEEP.search("Case1 Ery GATA1")
    assert select_chipatlas(E, "T4")["tier"] == "T4" and select_chipatlas(E, "T4")["celltypes"] == ["THP-1"]
    assert select_chipatlas(E, "T2")["tier"] == "T2" and select_chipatlas(E, "NOPE") is None
    # fail-closed loaders: receipt integrity (verified_receipt) and ChIP-Atlas selection
    # completeness / sha (chipatlas_occ)
    global CA_SEL
    with tempfile.TemporaryDirectory() as d:
        f = f"{d}/x.tsv"
        open(f, "w").write("a\n")
        try:
            verified_receipt(f)
            raise AssertionError("missing receipt accepted")
        except L.Blocked as e:
            assert "NO_RECEIPT" in str(e)
        json.dump(dict(out_sha256="0" * 64), open(f"{d}/x.receipt.json", "w"))
        try:
            verified_receipt(f)
            raise AssertionError("stale receipt accepted")
        except L.Blocked as e:
            assert "STALE_OR_ALTERED" in str(e)
        json.dump(dict(out_sha256=L.sha_file(f), script_sha256="old"), open(f"{d}/x.receipt.json", "w"))
        assert verified_receipt(f)["prep_script_sha_matches_current"] is False
        bed = f"{d}/A_S1.hg38.bed"
        open(bed, "w").write("chr1\t100\t200\tn\t5\t.\t1\t1\t1\t50\n")
        old_sel, CA_SEL = CA_SEL, f"{d}/selection.json"
        try:
            sel = dict(selection=dict(A=dict(tier="T1", celltypes=["CD34+"], srx=["S1", "S2"], excluded={}, titles={}, files={"S1": dict(path=bed, sha256=L.sha_file(bed))})),
                       excluded_tfs=dict(B="EXCLUDED: none"), download_failures={})
            json.dump(sel, open(CA_SEL, "w"))
            for tf_, tag in (("A", "CHIPATLAS_FILES_INCOMPLETE"), ("B", "NO_CHIPATLAS_EXPERIMENT_FOR_B")):                 # 1 file for 2 selected experiments -> fail closed
                try:
                    chipatlas_occ(tf_, {})
                    raise AssertionError("incomplete selection accepted")
                except L.Blocked as e:
                    assert tag in str(e), str(e)
            sel["selection"]["A"]["srx"] = ["S1"]
            sel["selection"]["A"]["files"]["S1"]["sha256"] = "1" * 64
            json.dump(sel, open(CA_SEL, "w"))
            try:
                chipatlas_occ("A", {})
                raise AssertionError("altered bed accepted")
            except L.Blocked as e:
                assert "ALTERED" in str(e)
        finally:
            CA_SEL = old_sel
    # BloodChIP rule R2: any TRUE region of the gene's mapped set
    reg = pd.DataFrame({"HSC_TF": [True, False, False, False], "CMP_TF": [False, False, True, False]}, index=[f"Region_{i}" for i in range(4)])
    gm = pd.DataFrame(dict(Gene=["a", "b", "c", "d"], Regions=["Region_0, Region_1", "Region_1", "Region_2, Region_3", "Region_99"]))
    un, per = bloodchip_bound(reg, gm, ["HSC_TF", "CMP_TF"])
    assert per["HSC_TF"] == {"A"} and per["CMP_TF"] == {"C"} and un == {"A", "C"}
    # KLF1_B: canonical TSS (longest span), minus-strand TSS = txEnd-1, unique-nearest summit
    # assignment, 1 kb / 10 kb windows, co-located TSSs (ties) unassigned
    rg = pd.DataFrame([[0, "t1", "chr1", "+", 1000, 5000, 0, 0, 1, "", "", 0, "Ga"], [0, "t2", "chr1", "+", 1000, 3000, 0, 0, 1, "", "", 0, "Ga"],
                       [0, "t3", "chr1", "-", 20000, 30000, 0, 0, 1, "", "", 0, "Gb"], [0, "t4", "chr1_random", "+", 5, 9, 0, 0, 1, "", "", 0, "Gc"],
                       [0, "t5", "chr1", "+", 50000, 60000, 0, 0, 1, "", "", 0, "Gd"], [0, "t6", "chr1", "+", 50000, 61000, 0, 0, 1, "", "", 0, "Ge"]])
    ct = canonical_tss_mm9(rg)
    tt = ct.set_index("sym")
    assert set(tt.index) == {"GA", "GB", "GD", "GE"} and tt.tss0["GA"] == 1000 and tt.tss0["GB"] == 29_999
    pkk = pd.DataFrame(dict(chrom="chr1", s=[1400, 35000, 50000], e=[1600, 35100, 50100], score=[9.0, 5.0, 7.0]))
    pkk["mid"] = (pkk.s + pkk.e) // 2
    F = eklf_gene_features(ct, pkk)
    assert F.BOUND_PROM["GA"] and F.nearest_summit_tss_dist["GA"] == 500 and F.max_signal_10kb["GA"] == 9.0
    assert not F.BOUND_PROM["GB"] and F.BOUND_10KB["GB"] and F.nearest_summit_tss_dist["GB"] == 5051     # summit 35050 vs TSS 29999
    assert not F.BOUND_10KB["GD"] and not F.BOUND_10KB["GE"]                                              # summit 50050 sits on a coordinate shared by two symbols -> unassigned
    # PRO-seq rule: tested = finite padj & unique non-NA symbol, responder padj<.05
    with tempfile.TemporaryDirectory() as d:
        p = f"{d}/t.tsv"
        pd.DataFrame(dict(Symbol=["A", "B", "B", None, "C", "D"], **{PADJ24: [.01, .01, .01, .01, np.nan, .5]})).to_csv(p, sep="\t", index=False)
        t, r, _ = proseq_gata1_sets(p)
        assert r == {"A"} and t == {"A", "D"}, (r, t)
        rg2 = pd.DataFrame([[0, "t1", "chr1", "+", 100, 300, 0, 0, 2, "100,200,", "150,300,", 0, "Ga"]])    # refGene -> gtf exons (0-based start -> 1-based)
        assert refgene_to_gtf(rg2, f"{d}/o.gtf") == 2 and open(f"{d}/o.gtf").read().splitlines()[1].split("\t")[3:5] == ["201", "300"]
    # pydeseq2 wrapper: a strongly DE gene is called
    try:
        import pydeseq2  # noqa: F401
        cnt = pd.DataFrame(rng.poisson(200, (6, 60)), index=[f"s{i}" for i in range(6)], columns=[f"g{i}" for i in range(60)])
        cnt["g0"], cnt["g1"] = [50, 55, 45, 400, 420, 380], 0
        rs = run_deseq(cnt, pd.DataFrame(dict(genotype=["WT"] * 3 + ["KO"] * 3), index=cnt.index), "~genotype", ["genotype", "KO", "WT"])
        assert rs.padj["g0"] < 0.05 and np.isnan(rs.padj["g1"]) and _de_table(rs).columns.tolist() == ["gene", "baseMean", "log2FC", "padj"]
    except ImportError:
        print("selftest: pydeseq2 absent -> DE wrapper not tested")
    # inventory counts on a synthetic truth (row 9 has occupancy unknown)
    tr = L.mk_truth(np.ones(10, bool), np.arange(10) < 5, np.r_[np.ones(9, bool), False], np.arange(10) % 2 == 0, np.ones(10, bool))
    c = abcd(tr)
    assert (c["n_A_prim"], c["n_B_prim"], c["n_C_prim"], c["n_D_prim"]) == (3, 2, 2, 2) and c["n_occ_unknown"] == 1 and not c["gate_AB_prim"], c
    # bm_truth: all-NaN occupancy score -> None (descriptor NOT_RUN, never a NaN column)
    _BM.update(st=None, nt=set(), tss={}, dup=set())
    tr2, mf2 = bm_truth(np.array(["G1", "G2", "G3"]), {"G1"}, {"G1", "G2", "G3"}, dict(prim={"G1"}, sens={"G1", "G2"}, dist=None, score=None, mappable={"G1", "G2", "G3"}, dup=set(),
                                                                                       mf=dict(files=["a"])), dict(files=["b"]))
    _BM.clear()
    assert tr2["occ_score"] is None and mf2["files"] == ["a", "b"] and tr2["resp"].tolist() == [True, False, False] and tr2["occ"]["sens"].tolist() == [True, True, False]
    # CARDIAC truth builder: exact-symbol join (unknown/duplicate symbols unknown), receipt +
    # frozen-declaration fail-closed checks, Gata4 routing
    global CARDIAC_DIR, CARDIAC_TABLE
    with tempfile.TemporaryDirectory() as d:
        Tt = pd.DataFrame(dict(gene=["Tbx5", "Nppa", "Gm1", "Ttn"], refgene_mapped=[True, True, False, True], Tbx5_tested=[True, True, True, False], Tbx5_resp=[True, False, True, False],
                               bound_Tbx5=[True, True, False, False], bound10_Tbx5=[True, True, False, True], dist_Tbx5=[100.0, 1500.0, np.nan, 8000.0],
                               bound_Gata4=[True, False, False, True], bound10_Gata4=[True, False, False, True], dist_Gata4=[10.0, 5e4, np.nan, 20.0]))
        Tt.to_csv(f"{d}/truth_gene_table.tsv", sep="\t", index=False)
        open(f"{d}/CARDIAC_TRUTH_V1_FROZEN.md", "w").write("frozen")
        rc = dict(output_files_sha256={"truth_gene_table.tsv": L.sha_file(f"{d}/truth_gene_table.tsv")}, frozen_declaration_sha256=L.sha_file(f"{d}/CARDIAC_TRUTH_V1_FROZEN.md"), builder_sha256="b", class_counts={})
        json.dump(rc, open(f"{d}/receipt.json", "w"))
        old_d, old_t, CARDIAC_DIR, CARDIAC_TABLE = CARDIAC_DIR, CARDIAC_TABLE, d, f"{d}/truth_gene_table.tsv"
        try:
            axis = np.array(["Tbx5", "Ttn", "Unknown1", "Nppa", "Gm1"])
            tr, mf = truth_cardiac("TBX5", axis)
            assert tr["resp_ok"].tolist() == [True, False, False, True, True] and tr["resp"].tolist() == [True, False, False, False, True]
            assert tr["occ_ok"].tolist() == [True, True, False, True, False] and tr["occ"]["prim"].tolist() == [True, False, False, True, False] and tr["occ"]["sens"].tolist() == [True, True, False, True, False]
            assert np.isnan(tr["occ_dist"][2]) and tr["occ_dist"][0] == 100.0 and tr["occ_score"] is None and mf["unknown_mapping"] == 2
            for bad_tf, bad_only in (("GATA4", False), ("TBX5", True), ("HAND2", False)):
                try:
                    truth_cardiac(bad_tf, axis, occupancy_only=bad_only)
                    raise AssertionError(f"routing accepted {bad_tf}/{bad_only}")
                except L.Blocked:
                    pass
            g4, mf4 = truth_cardiac("GATA4", axis, occupancy_only=True)
            assert g4["resp"].sum() == 0 and g4["resp_ok"].tolist() == g4["occ_ok"].tolist() and g4["occ"]["prim"].tolist() == [True, True, False, False, False]      # placeholder: zero responders by construction
            open(f"{d}/truth_gene_table.tsv", "a").write("x\n")                                                                                                   # altered table -> refused
            try:
                truth_cardiac("TBX5", axis)
                raise AssertionError("altered truth table accepted")
            except L.Blocked as e:
                assert "STALE_OR_ALTERED" in str(e)
            Tt.to_csv(f"{d}/truth_gene_table.tsv", sep="\t", index=False)
            open(f"{d}/CARDIAC_TRUTH_V1_FROZEN.md", "w").write("edited")                                                                                             # edited declaration -> refused
            try:
                truth_cardiac("TBX5", axis)
                raise AssertionError("edited frozen declaration accepted")
            except L.Blocked as e:
                assert "FROZEN_DECLARATION" in str(e)
        finally:
            CARDIAC_DIR, CARDIAC_TABLE = old_d, old_t
    # registration extension point: rebinds L.build_endpoint, adds systems on existing blocks/axes,
    # idempotent, fully reversible
    orig = L.build_endpoint
    register()
    try:
        assert L.build_endpoint is build_endpoint_extra and all(k in L.SYSTEMS and k in L.WIN for k in EXTRA)
        assert all(L.SYSTEMS[k]["block"] in L.BLOCKS and L.SYSTEMS[k]["axis"] in L.SYSTEMS and EXTRA[k]["role"] for k in EXTRA)
        assert L.BLOCKS[-2:] == ["CARDIAC", "CARDIACV"] and L.BLOCKS.count("CARDIAC") == 1 and L.BLOCKS.count("CARDIACV") == 1 and L.BLOCKS[:4] == ["ERYTHROID", "PANC", "BONEMARROW", "K562"]     # appended last: existing block seeds unchanged
        assert EXTRA["CARDIACV"]["role"].startswith("PRIMARY") and EXTRA["CARDIACV_GATA4_OCC"]["role"].startswith("DESCRIPTIVE") and EXTRA["CARDIACV"]["tfs"] == ["TBX5", "NKX2-5"] and EXTRA["CARDIACV"]["base"] == "CARDIACV"
        assert EXTRA["CARDIAC"]["role"].startswith("PRIMARY") and EXTRA["CARDIAC_GATA4_OCC"]["role"].startswith("DESCRIPTIVE") and EXTRA["CARDIAC"]["tfs"] == ["TBX5", "NKX2-5"]
        assert EXTRA["BM_BLOODCHIP"]["role"] == "DESCRIPTIVE_DOMAIN_OCCUPANCY" and EXTRA["BM_GATA1_ACUTE_T1"]["role"].startswith("SENSITIVITY") and set(EXTRA["BM_CHIPATLAS"]["tfs"]) == set(PANEL_TFS)
        register()
        assert _ORIG["build_endpoint"] is orig
        import types
        assert all(L.SYSTEMS[k]["block"] == "K562" and L.SYSTEMS[k]["axis"] == "K562" and EXTRA[k]["base"] == "K562" for k in K562_ALIAS)
        assert EXTRA["K562_EMP_ALLNULL"]["role"].startswith("PRIMARY") and EXTRA["K562_EMP"]["role"].startswith("SENSITIVITY") and EXTRA["K562_EMP"]["tfs"] == L.k562_tfs() and L.WIN["K562_EMP"] == L.WIN["K562"]
        n = 60
        ep = dict(system="X", blk=types.SimpleNamespace(unit=np.arange(n)), gi=np.arange(n), occ=dict(prim=np.ones(n, bool)), resp=np.arange(n) % 2 == 0)
        y, rows, mask = np.arange(n) < 30, np.ones(n, bool), np.ones(n, bool)
        assert L.gate_of(ep, y, rows, mask) is True and L.gate_of(dict(ep, system="K562_EMP"), y, rows, mask) is False and L.gate_of(dict(ep, system="K562_EMP_ALLNULL"), y, rows, mask) is False   # only 60 evaluable ChIP+ < 100
        assert L.gate_of(dict(ep, system="CARDIACV"), y, rows, mask) is False and L.gate_of(dict(ep, system="CARDIACV_GATA4_OCC"), y, rows, mask) is True
        assert L.gate_of(dict(ep, system="CARDIAC"), y, rows, mask) is False and L.gate_of(dict(ep, system="CARDIAC_GATA4_OCC"), y, rows, mask) is True          # CARDIAC inherits the >=100-bound rule; the descriptive system does not
        # truth_k562_v4: state swapped only during the call (restored even on error), one cache per
        # system, missing state never falls back to v3
        real, seen = L.truth_k562, []
        L.truth_k562 = lambda tf, genes, cache: (seen.append((L.K562_STATE, id(cache))) or ({}, {}))
        keep = L.K562_STATE
        try:
            truth_k562_v4("T", [], "K562_EMP", "/x/emp.npz", "w")
            truth_k562_v4("T", [], "K562_EMP_ALLNULL", "/x/allnull.npz", "w")
            truth_k562_v4("T", [], "K562_EMP", "/x/emp.npz", "w")
            assert seen[0][0] == "/x/emp.npz" and seen[1][0] == "/x/allnull.npz" and seen[0][1] == seen[2][1] != seen[1][1] and L.K562_STATE == keep
            L.truth_k562 = lambda tf, genes, cache: 1 / 0
            try:
                truth_k562_v4("T", [], "K562_EMP", "/x/emp.npz", "w")
                raise AssertionError("must propagate")
            except ZeroDivisionError:
                assert L.K562_STATE == keep
        finally:
            L.truth_k562 = real
    finally:
        unregister()
    assert L.build_endpoint is orig and "KLF1_B" not in L.SYSTEMS and "K562_EMP" not in L.SYSTEMS and "CARDIAC" not in L.SYSTEMS and "CARDIAC" not in L.BLOCKS and L.gate_of.__name__ == "gate_of"
    print("gb_lattice_extra selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["selftest", "inventory", "prep-klf1", "prep-bcl11a-de", "prep-bcl11a-peaks", "fetch-chipatlas", "lattice", "tfs"])
    ap.add_argument("--systems", default="K562," + ",".join(EXTRA))
    ap.add_argument("--tfs", default="")
    a, rest = ap.parse_known_args()
    if a.cmd == "tfs":                                                            # `tfs --systems K562_EMP --slice i --nslice n`: comma list of the i-th round-robin slice of a system's TF list (multi-host --tfs sharding)
        sl = argparse.ArgumentParser()
        sl.add_argument("--slice", type=int, default=0)
        sl.add_argument("--nslice", type=int, default=1)
        b = sl.parse_args(rest)
        if not (b.nslice >= 1 and 0 <= b.slice < b.nslice):
            raise SystemExit(f"REFUSING: --slice {b.slice} outside [0, {b.nslice})")
        sel = EXTRA[a.systems.split(",")[0]]["tfs"][b.slice::b.nslice]
        if not sel:
            raise SystemExit("REFUSING: empty TF slice (an empty --tfs would run ALL TFs)")
        print(",".join(sel))
        return
    if a.cmd == "lattice":
        register()
        rest = list(rest)
        base = X_OUT
        if "--out" in rest:                                                       # --out is the base dir; every system gets its own subdir (no pooling of duplicate endpoints into macro/Holm families)
            i = rest.index("--out")
            base = rest[i + 1]
            del rest[i:i + 2]
        syslist = [x.strip() for x in a.systems.split(",") if x.strip()]
        if a.tfs and len(syslist) != 1:
            raise SystemExit(f"REFUSING: --tfs is only accepted with exactly ONE system in --systems (got {len(syslist)}): the TF list would be applied to every system and hide NOT_RUN endpoints")
        for sysname in syslist:
            out = f"{base}/{sysname}"
            os.makedirs(out, exist_ok=True)
            role = EXTRA[sysname]["role"] if sysname in EXTRA else "Existing lattice-analysis system"
            L.atomic_json(f"{out}/SYSTEM_ROLE.json", dict(system=sysname, role=role, module_sha256=L.sha_file(__file__),
                                                          note="own out dir: never merge/pool with another system's endpoints into macro or Holm families"))
            sys.argv = ["gb_layer_lattice", "--systems", sysname] + (["--tfs", a.tfs] if a.tfs else []) + rest + ["--out", out]
            L.main()
    else:
        {"selftest": lambda _: selftest(), "inventory": inventory, "fetch-chipatlas": fetch_chipatlas, "prep-klf1": prep_klf1, "prep-bcl11a-de": prep_bcl11a_de,
         "prep-bcl11a-peaks": prep_bcl11a_peaks}[a.cmd](a)


if __name__ == "__main__":
    main()
