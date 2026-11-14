#!/usr/bin/env python
"""Evaluate functional response and physical occupancy across systems."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
for _p in (str(HERE), (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import gb_canon_table as CT  # noqa: E402
import gb_composites as GC  # noqa: E402

# loader constants (fail closed if absent)
R = __import__("os").environ["GEOBRIDGE_ROOT"]
SPEC_LOG = f"{R}/internal_review_log_not_shipped_b.log"
OUT_DIR = f"{R}/results/real_uniform/layer_lattice"
CATBANK = f"{R}/results/real_uniform/catbank"                      # SYSTEM_priors.npz, SYSTEM_features.tsv, SYSTEM_scores_i-j.npz  (gb_catbank outputs)
MODULE_DIR = f"{OUT_DIR}/modules"                                  # optional control-defined gene modules SYSTEM_modules.tsv (gene, module); absent -> gene-level blocks
RFX6_TSV = f"{R}/results/endpoints_v2/rfx6_occupancy.tsv"          # gb_signed_endpoints.py rfx6_occupancy (not data/rfx6_binding/rfx6_mm9_gene_binding.tsv)
BM_TRUTH = f"{R}/results/gse274113_truth_2026_09_19"               # historical stratified panel + gene_stats_nt.tsv (gb_cop_alignment.py:339-350)
BM_CHIP_DIR = f"{R}/data/mk_chip_gse24674/chipatlas"
BM_GTF = f"{R}/data/refs/gencode.v32.annotation.gtf.gz"           # gencode v32 hg38; the sealed scPerturb/RPE1 file is never opened (equivalence to the BM CellRanger reference is not verified here)
K562_STATE = f"{R}/results/k562_truth_v3_2026_09_21/k562_state.npz"
K562_CHIP_DIR = f"{R}/data/encode/k562_v2"
K562_GTF = f"{R}/data/refs/gencode.v44.annotation.gtf.gz"
K562_BLACKLIST = f"{R}/data/refs/hg38-blacklist.v2.bed.gz"        # K562 requires the ENCODE hg38 blacklist.
KLF1_TRUTH_KO = f"{R}/results/klf1_heldout/truth/GSE94351_gene_exp.diff.gz"
KLF1_CHIP_BED = f"{R}/results/klf1_heldout/truth/GSE210779_w_peaks_125_250_sig_fold.bed.gz"

BOOT_SEED, B_FINAL, B_PILOT = 20260930, 9999, 200
MIN_CLASS, MIN_UNITS = 20, 20
K562_MIN_BOUND = 100
DELTA_AP, DELTA_P50 = 0.01, 0.05
NFOLD = 5                                                           # gb_decomp.py:29 K
MATCH_CALIPER, MATCH_RETAIN, MATCH_SMD = 0.5, 0.80, 0.10
ALPHA = 0.05
GB = "GeoBRIDGE"
EXT = ["|Pearson|", "GENIE3", "GRNBoost2", "RegVelo |J|", "Velorama"]  # gb_canon_table.COMPS
SIX = [GB] + EXT                                                    # all six methods
M_PAIR, M_CELL = "pair-static M", "matched cell concordance"
SOTA = ["RegDiffusion", "DeepSEM", "PIDC", "FastTENET"]
SOTA_SPECS = {"RegDiffusion": ("regdiffusion_v1.npz", "score_all", "tfs"), "DeepSEM": ("deepsem_v1.npz", "score_all", "tfs"),
              "PIDC": ("pidc_v1.npz", "score_all", "tfs"), "FastTENET": ("fasttenet_v1.npz", "score_all", "tfs")}
SOTA_SCOPE = "ALL6+SOTA"
GEN_PRIORS = {"prior |Pearson|": "pearson", "prior GENIE3": "genie3", "prior GRNBoost2": "grnboost2"}   # generic priors
GB_PRIOR, GB_PRIOR_SRC = "prior GB", "P_pairU"
# Single-feature reference rankers with fixed, label-independent signs.
REF_SIGNS = {"REF -raw_det": ("raw_det", -1), "REF -sm_det": ("sm_det", -1), "REF -abundance": ("abundance", -1), "REF +dynamics": ("dynamics", +1),
             "REF -pair_support": ("pair_support", -1), "REF -mu_det": ("mu_det", -1), "REF +mu_dynamics": ("mu_dynamics", +1),
             "REF +abs_spearman_mu_dpt": ("abs_spearman_mu_dpt", +1), "REF +mu_logvar": ("mu_logvar", +1)}
REFS = list(REF_SIGNS)
COMPS = list(GC.COMP_NAMES)                                                                                  # 4 frozen composites of the 9 REF columns: members of the REF family; a NO_CALL composite is absent from V (reported NOT_RUN, never replaced)
HINDSIGHT = "best-in-hindsight REF"                                                                          # descriptive upper bound over the 9 REF AUROCs; never in inference
PAIR_METHODS = EXT + [M_PAIR, M_CELL] + list(GEN_PRIORS) + [GB_PRIOR] + REFS + COMPS + SOTA
F0_COLS = ["raw_det", "sm_det", "abundance", "pair_support", "dynamics"]
F1_COLS = F0_COLS + ["raw_det_u", "mu_det", "mu_abundance", "mu_logvar", "mu_pair_support", "mu_dynamics", "abs_spearman_ms_dpt", "abs_spearman_mu_dpt"]   # gb_catbank f1cols
F1X_COLS = F1_COLS + ["hvg_disp_norm", "cv2_spliced_norm", "us_ratio_norm"]                                   # optional auxiliary tier (gb_catbank f1xcols), --features F1x
FEATURE_TIER = ["F1"]                                                                                         # mutable holder set from the cli
MODELS = ["M0", "M1", "M2", "M3"]                                   # F1 | +generic priors | +P_GB | +full GB
INCR = [("M1", "M0"), ("M2", "M1"), ("M3", "M2"), ("M3", "M1")]
KEEP = ("AUROC", "AUPRC", "APLIFT", "PREV", "P20", "P50", "R20", "R50")
BSK = ("AUROC", "AUPRC", "P50")                                     # metrics whose draws are stored (raw arm)
BSK_ADJ = ("AUROC", "AUPRC", "APLIFT", "P20", "P50")               # ladder draws: AP lift and fixed-budget yield (P@K only where n >= 2K) are reported too
WITHIN_T = ["AB", "CD", "AC", "BD", "AD"]                           # within-method four-class contrasts (A>D added)
SUP_POINT, SUP_LB, EQ_LO, EQ_HI = 0.55, 0.50, 0.45, 0.55
DIST_BINS = [("0-1kb", -1.0, 1000.0), ("1-10kb", 1000.0, 10000.0), (">10kb", 10000.0, np.inf)]   # evaluation-only descriptor bins (distance TSS - nearest peak)
SYSTEMS = {"GATA1": dict(block="ERYTHROID", axis="GATA1", tfs=["GATA1"]), "KLF1": dict(block="ERYTHROID", axis="GATA1", tfs=["KLF1"]),
           "PANC": dict(block="PANC", axis="PANC", tfs=["RFX6"]),
           "BONEMARROW": dict(block="BONEMARROW", axis="BONEMARROW", tfs=["GATA1", "GATA2", "RUNX1", "FLI1", "TAL1"]),
           "K562": dict(block="K562", axis="K562", tfs=None)}
BLOCKS = ["ERYTHROID", "PANC", "BONEMARROW", "K562"]
WIN = {"GATA1": ("+-1kb any transcript TSS", "+-10kb"), "KLF1": ("+-1kb any transcript TSS", "+-10kb"), "PANC": ("+-1kb unique-nearest canonical TSS", "+-10kb"),
       "BONEMARROW": ("peak overlap TSS+-2kb", "nearest peak midpoint <=10kb of TSS (NEW sensitivity)"),
       "K562": ("unique-nearest selected TSS summit +-2kb", "distal: unassigned summits -> nearest TSS <=100kb")}


# a required frozen asset is absent / unusable: the endpoint is reported NOT_RUN with this reason, never dropped silently or zero-filled
class Blocked(Exception):
    ...


_SHA = {}


def sha_file(p, chunk=1 << 24):
    k = (str(p), os.path.getsize(p), os.path.getmtime(p))
    if k not in _SHA:
        _SHA[k] = _sha_file(p, chunk)
    return _SHA[k]


def _sha_file(p, chunk=1 << 24):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def frec(p, **kw):
    p = str(p)
    if not os.path.exists(p):
        raise Blocked(f"MISSING_FILE {p}")
    return dict(path=p, sha256=sha_file(p), bytes=os.path.getsize(p), **kw)


def sha_arr(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def atomic_json(p, obj):
    tmp = f"{p}.tmp{os.getpid()}"
    json.dump(obj, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, p)


def atomic_tsv(df, p):
    tmp = f"{p}.tmp{os.getpid()}"
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, p)


# bootstrap blocks (synchronised draws)
# Gene axis, resampling units and shared bootstrap draws for one substrate.
class BlockAxis:

    def __init__(self, names, B, seed, units=None, scheme="gene"):
        self.names = np.asarray(names).astype(str)
        self.idx = {g: i for i, g in enumerate(self.names)}
        if len(self.idx) != len(self.names):
            raise Blocked("AXIS_NOT_UNIQUE_EXACT")
        G = len(self.names)
        self.unit_name = self.names if units is None else np.asarray(units).astype(str)
        _, self.unit = np.unique(self.unit_name, return_inverse=True)
        self.scheme, self.B, self.seed = scheme, B, seed
        if units is None:
            self.counts = CT.make_counts(G, B, seed)                      # exact gb_canon_table draw scheme
        else:
            J = self.unit.max() + 1
            rng = np.random.default_rng(seed)
            m = rng.multinomial(J, np.full(J, 1.0 / J), size=B).astype(np.int16)
            self.counts = m[:, self.unit]                                   # a module is resampled as one unit: its genes move together

    def gi(self, genes):
        try:
            return np.fromiter((self.idx[g] for g in genes), int, len(genes))
        except KeyError as e:
            raise Blocked(f"GENE_NOT_ON_BLOCK_AXIS {e}")

    # fixed folds by hash of the unit (module or gene) name, identical across TFs/systems (gb_decomp.fold_of recipe)
    def fold_of(self, gi):
        return np.array([int(hashlib.sha1(self.unit_name[i].upper().encode()).hexdigest(), 16) % NFOLD for i in gi])

    def fingerprint(self):
        return sha_arr(self.counts[:5])


def load_modules(axis_sys, names):
    p = f"{MODULE_DIR}/{axis_sys}_modules.tsv"
    if not os.path.exists(p):
        return None, "gene(no control-defined module map supplied)"
    d = pd.read_csv(p, sep="\t", dtype=str).set_index("gene")["module"]
    miss = [g for g in names if g not in d.index]
    if miss:
        raise Blocked(f"MODULE_MAP_INCOMPLETE {p}: {len(miss)} axis genes missing, e.g. {miss[:3]}")
    return d.reindex(names).to_numpy(str), f"module({p})"


# truth constructors (pure parts are unit-tested)
def upper(a):
    return np.char.upper(np.asarray(a).astype(str))


# axis genes whose upper-cased symbol is shared with another axis gene are ambiguous on an upper-case join -> UNKNOWN
def amb_upper(genes):
    return pd.Series(upper(genes)).duplicated(keep=False).to_numpy()


# numeric column of a truth frame as float (NaN where absent / non-finite): evaluation-only occupancy descriptors (distance, peak score)
def numcol(T, name, tf=lambda x: x):
    if name not in T:
        return np.full(len(T), np.nan)
    v = pd.to_numeric(T[name], errors="coerce").to_numpy(float)
    return np.where(np.isfinite(v), tf(v), np.nan)


def mk_truth(resp_ok, resp, occ_ok, occ_prim, occ_sens, **extra):
    resp_ok, occ_ok = np.asarray(resp_ok, bool), np.asarray(occ_ok, bool)
    return dict(resp_ok=resp_ok, resp=np.asarray(resp, bool) & resp_ok, occ_ok=occ_ok,
                occ=dict(prim=np.asarray(occ_prim, bool) & occ_ok, sens=np.asarray(occ_sens, bool) & occ_ok), **extra)


def truth_gata1_frame(V, genes):
    T = V.set_index("gene").reindex(genes)                                           # exact-name join (case collisions never merged)
    ok = T.BOUND_PROM.notna().to_numpy() & T.BOUND_10KB.notna().to_numpy()
    rok = T.KO_tested.eq(True).to_numpy()
    return mk_truth(rok, T.KO_resp.isin(["act", "rep"]).to_numpy(), ok, T.BOUND_PROM.eq(1).to_numpy(), T.BOUND_10KB.eq(1).to_numpy(),
                    occ_dist=numcol(T, "nearest_summit_tss_dist"), occ_score=numcol(T, "max_signal_10kb", np.log1p))


# T = gb_klf1_eval.ko_truth (index upper symbol, .resp)
# Fc = chip_gene_features (index upper symbol, BOUND_PROM/BOUND_10KB)
def truth_klf1_frames(genes, T, Fc):
    k, amb = upper(genes), amb_upper(genes)
    rok, ook = np.isin(k, T.index) & ~amb, np.isin(k, Fc.index) & ~amb
    g = lambda s: s.reindex(k).eq(True).to_numpy()
    Fk = Fc.reindex(k)
    return mk_truth(rok, g(T.resp), ook, g(Fc["BOUND_PROM"]), g(Fc["BOUND_10KB"]), n_ambiguous_upper=int(amb.sum()),
                    occ_dist=numcol(Fk, "nearest_summit_tss_dist"), occ_score=numcol(Fk, "max_signal_10kb", np.log1p))


def truth_panc_frame(T, genes):
    T = T.drop_duplicates("gene", keep=False).set_index("gene").reindex(genes)       # duplicated symbols dropped as a group (harmonized_tri.panc_polarity)
    k = T.ko_class.fillna("UNKNOWN")
    rok = k.isin(["TESTED_NULL", "ACT_RESPONDER", "REP_RESPONDER"]).to_numpy()
    return mk_truth(rok, k.isin(["ACT_RESPONDER", "REP_RESPONDER"]).to_numpy(), T["mapping"].eq("MAPPED").to_numpy(),
                    T.occupied_1kb.eq(True).to_numpy(), T.occupied_10kb.eq(True).to_numpy(),
                    occ_dist=numcol(T, "dist_nearest_assigned_summit"), occ_score=numcol(T, "maxFE_10kb", np.log1p))


# genes: axis (human, upper-case join)
def truth_bm_parts(genes, panel, testable, tss_syms, bound_prim, bound_sens, dup_syms, ctl=None, dist=None, score=None):
    k, amb = upper(genes), amb_upper(genes)
    isin = lambda s: np.isin(k, list(s))
    ook = isin(tss_syms) & ~isin(dup_syms) & ~amb
    dv = lambda d: np.full(len(k), np.nan) if d is None else np.array([d.get(g, np.nan) for g in k], float)
    return mk_truth(isin(testable) & ~amb, isin(panel), ook, isin(bound_prim), isin(bound_sens),
                    occ_ctl=None if ctl is None else (isin(ctl) & ook), occ_dist=dv(dist), occ_score=dv(score))


# state_row: 0 UNKNOWN / 1 TESTED_NULL / 2 UP / 3 down on state_genes (upper); prim/sens/unmappable: sets of upper symbols
def truth_k562_parts(genes, state_row, state_genes, prim, sens, unmappable, dist=None, score=None):
    k, amb = upper(genes), amb_upper(genes)
    st = pd.Series(np.asarray(state_row), index=upper(state_genes)).groupby(level=0).first().reindex(k).fillna(0).to_numpy()
    ook = ~np.isin(k, list(unmappable)) & ~amb
    dv = lambda d: np.full(len(k), np.nan) if d is None else np.array([d.get(g, np.nan) for g in k], float)
    return mk_truth(np.isin(st, [1, 2, 3]) & ~amb, np.isin(st, [2, 3]), ook, np.isin(k, list(prim)), np.isin(k, list(sens)), occ_dist=dv(dist), occ_score=dv(score))


PRIMARY_CHR = {f"chr{i}" for i in list(range(1, 23)) + ["X", "Y"]}


def _attr(a, key):
    m = re.search(key + r' "([^"]+)"', a)
    return m.group(1) if m else None


# Gencode v44: one TSS per gene_id = MANE Select transcript, else longest transcript, transcript-ID tie-break
def k562_tss_from_gtf(path):
    import gzip
    rec = []
    with gzip.open(path, "rt") as fh:
        for ln in fh:
            if ln[0] == "#":
                continue
            f = ln.split("\t", 8)
            if f[2] != "transcript" or f[0] not in PRIMARY_CHR:
                continue
            a = f[8]
            rec.append((_attr(a, "gene_id"), _attr(a, "gene_name"), _attr(a, "transcript_id"), f[0], int(f[3]), int(f[4]), f[6], 'tag "MANE_Select"' in a))
    return k562_select_tss(pd.DataFrame(rec, columns=["gene_id", "sym", "tid", "chrom", "start", "end", "strand", "mane"]))


def k562_select_tss(df):
    df = df.dropna(subset=["gene_id", "sym"]).copy()
    df["len"] = df.end - df.start + 1
    df = df.sort_values(["gene_id", "mane", "len", "tid"], ascending=[True, False, False, True]).groupby("gene_id").head(1)
    df["tss0"] = np.where(df.strand == "+", df.start, df.end) - 1
    df["sym"] = df.sym.str.upper()
    amb = set(df.sym[df.sym.duplicated(keep=False)])
    return df[~df.sym.isin(amb)][["sym", "chrom", "tss0"]].reset_index(drop=True), amb


# index into sorted `arr` of the nearest coordinate within `win` of each pos; -1 if none within win or if the nearest distance is shared by >1 entry
def nearest_unique(pos, arr, win):
    pos, arr = np.asarray(pos, np.int64), np.asarray(arr, np.int64)
    if len(arr) == 0 or len(pos) == 0:
        return np.full(len(pos), -1)
    i = np.searchsorted(arr, pos)
    lft, rgt = np.clip(i - 1, 0, len(arr) - 1), np.clip(i, 0, len(arr) - 1)
    dl, dr = np.abs(pos - arr[lft]), np.abs(arr[rgt] - pos)
    d = np.minimum(dl, dr)
    cnt = lambda x: np.searchsorted(arr, x, "right") - np.searchsorted(arr, x, "left")
    n_at = cnt(pos - d) + np.where(d > 0, cnt(pos + d), 0)
    return np.where((d <= win) & (n_at == 1), np.where(dl <= dr, lft, rgt), -1)


# peaks: DataFrame chrom,start,end,off (narrowPeak cols 1,2,3,10)
def k562_assign(peaks, tss, blacklist=None, win=2000, distal=100_000):
    ok = (peaks.off >= 0) & (peaks.off < peaks.end - peaks.start)
    st = dict(n_peaks=len(peaks), n_invalid_summit=int((~ok).sum()))
    p = peaks[ok].copy()
    if blacklist is not None:
        keep = np.ones(len(p), bool)
        for c, g in p.groupby("chrom"):
            b = blacklist[blacklist.chrom == c].sort_values("start")
            if not len(b):
                continue
            bs, be = b.start.to_numpy(np.int64), b.end.to_numpy(np.int64)
            cm = np.maximum.accumulate(be)
            j = np.searchsorted(bs, g.end.to_numpy(np.int64), "left")
            hit = (j > 0) & (cm[np.maximum(j - 1, 0)] > g.start.to_numpy(np.int64))
            keep[np.flatnonzero((p.chrom == c).to_numpy())[hit]] = False
        st["n_blacklisted"] = int((~keep).sum())
        p = p[keep]
    prim, sens, dist, score = set(), set(), {}, {}
    for c, g in p.groupby("chrom"):
        t = tss[tss.chrom == c].sort_values("tss0")
        arr, sym = t.tss0.to_numpy(np.int64), t.sym.to_numpy()
        pos = (g.start + g.off).to_numpy(np.int64)
        sig = g.sig.to_numpy(float) if "sig" in g else np.full(len(g), np.nan)
        a1 = nearest_unique(pos, arr, win)
        prim |= set(sym[a1[a1 >= 0]])
        for sy, d, sg in zip(sym[a1[a1 >= 0]], np.abs(pos[a1 >= 0] - arr[a1[a1 >= 0]]), sig[a1 >= 0]):
            dist[sy] = min(dist.get(sy, np.inf), float(d))
            if np.isfinite(sg):
                score[sy] = max(score.get(sy, -np.inf), float(np.log1p(sg)))
        rest = a1 < 0
        a2 = nearest_unique(pos[rest], arr, distal)
        sens |= set(sym[a2[a2 >= 0]])
        for sy, d in zip(sym[a2[a2 >= 0]], np.abs(pos[rest][a2 >= 0] - arr[a2[a2 >= 0]])):
            dist[sy] = min(dist.get(sy, np.inf), float(d))
    st["n_prim_genes"], st["n_sens_only_genes"] = len(prim), len(sens - prim)
    return prim, prim | sens, st, dist, score


def read_bed_gz(p, cols):
    return pd.read_csv(p, sep="\t", header=None, usecols=list(cols.keys()), names=None, compression="infer").rename(columns=cols)


def truth_gata1(genes):
    from gb_truth_v2_load import load_verified_gata1_v2, MANIFEST, EXPECTED_MANIFEST_SHA256
    V = load_verified_gata1_v2()                                                     # pinned manifest, all artifact hashes verified inside
    M = json.load(open(MANIFEST))["GATA1_V2"]
    rc = json.load(open(M["v2_receipt"]["path"]))
    T = truth_gata1_frame(V, genes)
    mf = dict(mapping=WIN["GATA1"], assembly="mm10", annotation="GENCODE vM25 protein_coding|lncRNA, TSS of every transcript",
              peaks="ENCODE ENCFF676TDJ (IDR, ENCSR000DIL), summit = start + col10", response="finite |log2 KO/WT| > .5 (gb_gata1_fourclass_canon)",
              manifest_sha256=EXPECTED_MANIFEST_SHA256, artifacts={k: v["sha256"] for k, v in M.items()},
              receipt={k: rc.get(k) for k in ("peak_sha256", "gtf_sha256", "n_peaks", "n_unmatched", "n_malformed_summits_excluded", "n_has_KO")},
              unknown_mapping=int((~T["occ_ok"]).sum()))
    return T, mf


def truth_klf1(genes):
    import gb_klf1_eval as KE
    T = KE.ko_truth(KLF1_TRUTH_KO)
    fc_p = f"{KE.OUT}/chip_gene_features.tsv"
    Fc = pd.read_csv(fc_p, sep="\t", index_col=0)
    tr = truth_klf1_frames(genes, T, Fc)
    mf = dict(mapping=WIN["KLF1"], assembly="mm10", annotation="GENCODE vM25 (gb_gata1_truth_v2.gene_features, summit = start+col10, midpoint if absent)",
              response="cuffdiff Q2 vs Q3 status OK, significant == yes", files=[frec(KLF1_TRUTH_KO), frec(KLF1_CHIP_BED), frec(fc_p), frec(f"{KE.OUT}/predictions.npz"),
                                                                                 frec(f"{KE.OUT}/ko_receipt.json")],
              n_ambiguous_upper_unknown=tr["n_ambiguous_upper"], unknown_mapping=int((~tr["occ_ok"]).sum()))
    return tr, mf


def truth_panc(genes):
    T = pd.read_csv(RFX6_TSV, sep="\t")
    rc_p = RFX6_TSV.replace(".tsv", ".receipt.json")
    rc = json.load(open(rc_p))
    if rc["sha256"] != sha_file(RFX6_TSV):
        raise Blocked(f"RFX6_TSV_SHA_MISMATCH: {RFX6_TSV} sha256 != its receipt {rc_p}")
    tr = truth_panc_frame(T, genes)
    mf = dict(mapping=WIN["PANC"], assembly="mm9", annotation="UCSC refGene mm9, one canonical transcript = longest span, primary contigs",
              peaks="GSE62844 HA-Rfx6 (Min6b1, transfected) MACS FDR<=5%, summit 1-based; each summit -> unique nearest canonical TSS, ties unassigned",
              response="frozen RFX6 DE padj<.05 (PANC evaluator); TESTED_NULL = tested non-DE",
              files=[frec(RFX6_TSV), frec(rc_p), frec(rc["peaks"]), frec(rc["refgene"]), frec(rc["de"])],
              n_summits_ambiguous_tie=rc.get("n_summits_ambiguous_tie"), unknown_mapping=int((~tr["occ_ok"]).sum()))
    return tr, mf


# Gencode gene-level TSS: gene start (+) / end (-) of gene_type protein_coding|lncRNA
def bm_tss_from_lines(lines):
    rec = []
    for ln in lines:
        if ln[0] == "#":
            continue
        f = ln.split("\t", 8)
        if f[2] != "gene":
            continue
        gt, gn, gid = _attr(f[8], "gene_type"), _attr(f[8], "gene_name"), _attr(f[8], "gene_id")
        if gid and gid.endswith("_PAR_Y"):                                            # pseudo-autosomal chrY copies duplicate the chrX gene: skipped before the ambiguity check
            continue
        if gt in ("protein_coding", "lncRNA") and gn and gid:
            rec.append((gn.upper(), gid, f[0], int(f[3]) if f[6] == "+" else int(f[4])))
    d = pd.DataFrame(rec, columns=["sym", "gid", "chrom", "pos"])
    amb = set(d.sym[d.groupby("sym").gid.transform("nunique") > 1])
    d = d[~d.sym.isin(amb)].drop_duplicates("sym")
    return {r.sym: (r.chrom, r.pos) for r in d.itertuples()}, amb


def bm_tss(_c={}):
    import gzip
    if "v" not in _c:
        with gzip.open(BM_GTF, "rt") as fh:
            _c["v"] = bm_tss_from_lines(fh)
    return _c["v"]


def peak_mids(tf):
    import glob
    hits = sorted(glob.glob(f"{BM_CHIP_DIR}/{tf}_*.bed") + glob.glob(f"{BM_CHIP_DIR}/{tf}_*.bed.gz"))
    if not hits:
        return None, None
    d = pd.read_csv(hits[0], sep="\t", header=None, comment="#")
    o = pd.DataFrame(dict(chrom=d[0].astype(str), s=d[1].astype(np.int64), e=d[2].astype(np.int64)))
    o["mid"] = (o.s + o.e) // 2
    o["score"] = pd.to_numeric(d[4], errors="coerce") if d.shape[1] > 4 else np.nan          # bed score column when present (evaluation-only descriptor)
    return o, hits[0]


# {sym: distance TSS -> nearest peak midpoint (same chromosome)}, {sym: log1p(max peak score with midpoint <= 10 kb; 0 if none)}
def bm_peak_desc(tss, pk):
    by = {}
    for c, g in pk.groupby("chrom"):
        g = g.sort_values("mid")
        by[c] = (g.mid.to_numpy(np.int64), g.score.to_numpy(float))
    dist, score = {}, {}
    for s_, (c, t) in tss.items():
        if c not in by:
            continue
        a, sc = by[c]
        i = np.searchsorted(a, int(t))
        dist[s_] = float(min(abs(int(a[j]) - int(t)) for j in (max(i - 1, 0), min(i, len(a) - 1))))
        lo, hi = np.searchsorted(a, int(t) - 10_000, "left"), np.searchsorted(a, int(t) + 10_000, "right")
        v = sc[lo:hi]
        score[s_] = float(np.log1p(np.nanmax(v))) if hi > lo and np.isfinite(v).any() else 0.0
    return dist, score


# genes (upper) whose TSS has a peak midpoint within `win` (same chromosome)
def near_within(tss, pk, win):
    by = {c: np.sort(g.mid.to_numpy(np.int64)) for c, g in pk.groupby("chrom")}
    out = set()
    for s, (c, t) in tss.items():
        a = by.get(c)
        if a is None or not len(a):
            continue
        i = np.searchsorted(a, int(t))
        if min(abs(int(a[j]) - int(t)) for j in (max(i - 1, 0), min(i, len(a) - 1))) <= win:
            out.add(s)
    return out


def truth_bm(tf, genes):
    import gb_bm_sweep as BS
    ST = pd.read_csv(f"{BM_TRUTH}/gse274113_stratified_panel.tsv", sep="\t")
    nt = pd.read_csv(f"{BM_TRUTH}/gene_stats_nt.tsv", sep="\t")
    if tf not in set(ST.tf):
        raise Blocked(f"NO_KO_PANEL_FOR_{tf} (GSE274113 stratified panel has no rows: response layer undefined)")
    panel = set(ST.target[ST.tf == tf].str.upper())
    testable = set(nt.gene[nt.nt_detect_frac >= 0.05].str.upper())
    tss, dup = bm_tss()                                                              # gencode v32 gene-level TSS; not gb_bm_sweep.tss_map (scPerturb var, sealed file)
    bp, npk = BS.bound_set(tf, tss, 2000)                                            # frozen primary rule (gb_bm_sweep.py:69)
    if bp is None:
        raise Blocked(f"NO_CHIP_PEAK_FILE_FOR_{tf}")
    pk, pkf = peak_mids(tf)
    ctl_pk, ctlf = peak_mids("rIgG")
    ctl = BS.bound_set("rIgG", tss, 2000)[0] if ctl_pk is not None else None
    bdist, bscore = bm_peak_desc(tss, pk)
    tr = truth_bm_parts(genes, panel, testable, set(tss), bp, near_within(tss, pk, 10_000), dup, ctl, bdist, bscore)
    mf = dict(mapping=WIN["BONEMARROW"], assembly="GRCh38", annotation="GENCODE v32 hg38 GTF gene records, gene_type protein_coding|lncRNA, TSS = gene start (+) / end (-), symbol upper-cased, symbols shared by >1 gene_id -> UNKNOWN",
              tss_source_differs_from="gb_bm_sweep.tss_map (scPerturb RPE1 var table) -- not used",
              historical_tss_map="historical map not reproduced by design (n genes with TSS moved > 1 kb: not computed)",
              n_dup_symbols_unknown=len(dup), n_tss=len(tss), n_peaks_frozen_rule=npk,
              n_endpoint_genes_without_v32_symbol=int((~np.isin(upper(genes), list(tss))).sum() - int(np.isin(upper(genes), list(dup)).sum())),
              note_v32="axis genes without a GENCODE v32 protein_coding|lncRNA symbol are UNKNOWN (never negative); _PAR_Y gene copies are skipped before the ambiguity check",
              peaks=f"ChIP-Atlas hg38 GSE24674 {os.path.basename(pkf)}", response="historical stratified panel padj<.05 (results/gse274113_truth_2026_09_19); testable = NTC detect >= 5%",
              files=[frec(f"{BM_TRUTH}/gse274113_stratified_panel.tsv"), frec(f"{BM_TRUTH}/gene_stats_nt.tsv"), frec(pkf), frec(BM_GTF)] + ([frec(ctlf)] if ctlf else []),
              sensitivity_note="nearest peak-midpoint <= 10 kb is new",
              efficacy_gate="NONE APPLIED: the mRNA-efficacy gate is withdrawn and no genotyping-qualification table is available", unknown_mapping=int((~tr["occ_ok"]).sum()))
    return tr, mf


def k562_tfs():
    import glob
    z = np.load(K562_STATE, allow_pickle=True)
    stf = {str(x).upper() for x in z["tfs"]}
    out = []
    for p in sorted(glob.glob(f"{K562_CHIP_DIR}/*.bed.gz")):
        tf = os.path.basename(p).split("_ENC")[0].upper()
        if tf in stf:
            out.append(tf)
    return out


def truth_k562(tf, genes, _cache={}):
    import glob
    if not os.path.exists(K562_BLACKLIST):
        raise Blocked(f"K562 NOT_RUN: ENCODE hg38 blacklist filtering is required and {K562_BLACKLIST} is absent (fetch ENCFF356LFX to that path; it is never skipped silently)")
    if "tss" not in _cache:
        _cache["tss"], _cache["amb"] = k562_tss_from_gtf(K562_GTF)
        _cache["bl"] = pd.read_csv(K562_BLACKLIST, sep="\t", header=None, usecols=[0, 1, 2], names=["chrom", "start", "end"], comment="#")
        _cache["state"] = np.load(K562_STATE, allow_pickle=True)
    z = _cache["state"]
    tfs = [str(x).upper() for x in z["tfs"]]
    hit = sorted(glob.glob(f"{K562_CHIP_DIR}/{tf}_ENC*.bed.gz"))
    if tf not in tfs or not hit:
        raise Blocked(f"K562_TF_NOT_ELIGIBLE {tf}")
    pk = pd.read_csv(hit[0], sep="\t", header=None, usecols=[0, 1, 2, 6, 9], names=["chrom", "start", "end", "sig", "off"])
    prim, sens, st, kdist, kscore = k562_assign(pk, _cache["tss"], _cache["bl"])
    mappable = set(_cache["tss"].sym)
    unmap = set(upper(genes)) - mappable
    tr = truth_k562_parts(genes, z["state"][tfs.index(tf)], z["genes"].astype(str), prim, sens, unmap, kdist, kscore)
    mf = dict(mapping=WIN["K562"], assembly="GRCh38", annotation="GENCODE v44 primary chr; ONE TSS/gene: MANE Select else longest transcript, tid tie-break; ambiguous symbols excluded",
              response="k562_state.npz: TESTED_UP/DOWN (within-TF BH q<.05) vs TESTED_NULL; UNKNOWN excluded", assign_stats=st, n_ambiguous_symbols=len(_cache["amb"]),
              files=[frec(hit[0]), frec(K562_GTF), frec(K562_BLACKLIST), frec(K562_STATE)], unknown_mapping=int((~tr["occ_ok"]).sum()),
              deviation="historical gb_k562_boundfunc.py used first gene-record TSS, no MANE, no blacklist, no summit validity check")
    return tr, mf


# scores / priors / features
# optional SOTA comparator rows (score_all over catalogue TFs) mapped onto `genes`; an absent/incomplete artifact is recorded in `miss`
def sota_rows(d, src, genes, V, miss, files=None):
    for m, (f, k, sk) in SOTA_SPECS.items():
        try:
            if files is not None:
                files[m] = frec(d / f)
            z = np.load(d / f, allow_pickle=True)
            s = [x.upper() for x in z[sk].astype(str)]
            gi = {g: i for i, g in enumerate(z["genes"].astype(str))}
            row = np.asarray(z[k], float)[s.index(src.upper())]
            V[m] = np.array([row[gi[g]] if g in gi else np.nan for g in genes])
        except (FileNotFoundError, KeyError, ValueError, Blocked) as e:
            miss[m] = f"{type(e).__name__}: {e}"


# GB primary (gb_primary.zcons on U_obs, computed before any truth join -- never re-z'd on a truth-restricted set) + catalogue comparators
def scores_prod(system, src):
    import gb_cop_alignment as CA
    import gb_harmonized_tri as HT
    import gb_primary as GP
    import gb_real_uniform as RU
    rows, eg = RU.gb_raw_rows(system, srcs=[src])
    if src.upper() not in rows:
        raise Blocked(f"GB_ROW_MISSING {system}/{src}")
    eg = np.asarray(eg).astype(str)
    Rw = rows[src.upper()]
    uni, obs_sha = CA.observational_universe(src, Rw, eg)
    genes = eg[uni]
    V, miss, files = {GB: GP.zcons({h: Rw[(h, "U")][uni] for h in GP.ROOTS})}, {}, {}
    d = RU.system_dir(system)
    for m, (f, k, sk) in HT.SPECS.items():
        try:
            files[m] = frec(d / f)
            z = np.load(d / f, allow_pickle=True)
            s = [x.upper() for x in z[sk].astype(str)]
            gi = {g: i for i, g in enumerate(z["genes"].astype(str))}
            row = np.asarray(z[k], float)[s.index(src.upper())]
            v = np.array([row[gi[g]] if g in gi else np.nan for g in genes])
        except (FileNotFoundError, KeyError, ValueError, Blocked) as e:
            miss[m] = f"{type(e).__name__}: {e}"
            continue
        V["RegVelo |J|" if m == "RegVelo" else m] = np.abs(v) if m == "RegVelo" else v
    sota_rows(d, src, genes, V, miss, files)
    files["gb_caches"] = [frec(q) for q in RU.cache_paths(system) if os.path.exists(q)]                     # production caches, sha256 (also validated by RU.validate_completion_cache)
    files["u_obs_sha256"] = obs_sha
    return genes, V, miss, eg, files


def scores_klf1():
    import gb_klf1_eval as KE
    g, m, S, _ = KE.load_pred()                                                      # sha-checked frozen predictions
    V = {n: S[i] for i, n in enumerate(m) if n in SIX}
    miss = {n: "absent from predictions.npz" for n in SIX if n not in V}
    import gb_real_uniform as RU
    sota_rows(RU.system_dir("GATA1"), "KLF1", g.astype(str), V, miss)                # KLF1 = a catalogue TF on the GATA1 fixture: its row of the GATA1 SOTA artifacts
    return g.astype(str), V, miss


def axis_of(system):
    import gb_real_uniform as RU
    _, eg = RU.gb_raw_rows(system, srcs=[SYSTEMS[system]["tfs"][0] if SYSTEMS[system]["tfs"] else "GATA1"])
    return np.asarray(eg).astype(str)


# the 9 reference rankers from a features frame: sign * feature (NaN where the feature is undefined); a missing column raises KeyError
def ref_vectors(F, genes=None):
    return {n: np.where(np.isfinite(v := sg * F[c].to_numpy(float)), v, np.nan) for n, (c, sg) in REF_SIGNS.items()}


# The 4 composites (gb_composites), fit once on U* = axis genes of this endpoint's system with all nine REF columns finite
def comp_vectors(F):
    assert {(c, sg) for c, sg in REF_SIGNS.values()} == set(GC.SPEC), "REF_SIGNS != gb_composites.SPEC"
    return GC.comp_from_features(F)


def members_sha(members):
    return hashlib.sha256("\n".join(members).encode()).hexdigest()


# the artifact's receipt (write_receipt / write_part layout): must exist and its output_sha256 must equal the file's sha256 (as gb_catbank.load_gb)
def receipt_of(path):
    rp = Path(path).with_suffix(".receipt.json")
    if not rp.exists():
        raise Blocked(f"NO_RECEIPT {rp}")
    rec = json.load(open(rp))
    if rec.get("output_sha256") != sha_file(path):
        raise Blocked(f"STALE_ARTIFACT {path}: sha256 != receipt output_sha256")
    return rec


# M / matched-cell rows (SYSTEM_scores shards), leave-focal-out priors (gb_catbank.loo_prior), F1 features
def catbank_attach(ep, axis_sys, focal, smalln=False):
    import glob
    reasons = []
    S = axis_sys                                                                      # always the full catbank artifacts (the lattice --smalln is about B and systems)
    genes = ep["genes"]
    tier = FEATURE_TIER[0]
    cols = F1X_COLS if tier == "F1x" else F1_COLS
    ep["meta"]["feature_tier"] = tier
    try:
        bank = json.load(open(f"{CATBANK}/{S}_bank.json"))
        if bank["sha256_members"] != members_sha(bank["members"]):
            raise Blocked(f"BANK_INCONSISTENT {S}")
        ep["files"]["bank"] = frec(f"{CATBANK}/{S}_bank.json")
    except (FileNotFoundError, Blocked) as e:
        bank = None
        reasons.append(f"BANK: {type(e).__name__}: {e}")
    try:
        if bank is None:
            raise Blocked("no validated bank")
        import gb_catbank as CB
        pf = f"{CATBANK}/{S}_priors.npz"
        rec = receipt_of(pf)
        if rec["config"]["bank_sha256"] != bank["sha256_members"]:
            raise Blocked(f"PRIORS_BANK_SHA_MISMATCH {pf}")
        pz = np.load(pf, allow_pickle=True)
        if [str(x) for x in pz["members"]] != bank["members"]:
            raise Blocked(f"PRIORS_MEMBERS_DIFFER_FROM_BANK {pf}")
        Tg = pz["T7|genes"].astype(str)
        members = [str(x).upper() for x in pz["members"]]
        r = members.index(focal.upper()) if focal.upper() in members else None
        pos = {g: i for i, g in enumerate(Tg)}
        for name, src in list(GEN_PRIORS.items()) + [(GB_PRIOR, GB_PRIOR_SRC)]:
            Z, s_, c_ = pz[f"T7|{src}|Z"], pz[f"T7|{src}|S"], pz[f"T7|{src}|C"]
            P = CB.loo_prior(Z, s_, c_, r)                                             # leave-focal-out; r None -> full bank (focal not in bank)
            ep["V"][name] = np.array([P[pos[g]] if g in pos else np.nan for g in genes])
        ep["files"]["priors"] = frec(pf)
        ep["meta"]["prior_focal_in_bank"] = r is not None
    except (FileNotFoundError, Blocked, ImportError) as e:
        reasons.append(f"PRIORS: {type(e).__name__}: {e}")
    try:
        if bank is None:
            raise Blocked("no validated bank")
        ff = f"{CATBANK}/{S}_features.tsv"
        rec = receipt_of(ff)
        for tname, tcols in (("F1", F1_COLS), ("F1x", F1X_COLS)):
            if tname == "F1" or tier == "F1x":
                if rec["config"][tname] != tcols:
                    raise SystemExit(f"REFUSING_FEATURE_LIST_MISMATCH: {tname} in {ff} receipt {rec['config'][tname]} != script {tcols}")
        if rec["input_sha256"].get("fixture") != bank["inputs"]["fixture_sha256"]:
            raise Blocked(f"FEATURES_FIXTURE_DIFFERS_FROM_BANK {ff}")
        F = pd.read_csv(ff, sep="\t", index_col="gene").reindex(genes)
        ep["F1"] = F[cols].to_numpy(float)                                            # KeyError here is a structural error and propagates
        ep["V"].update(ref_vectors(F))                                                # truth-free reference rankers (receipt-validated features file)
        cv, cd = comp_vectors(F)
        ep["V"].update(cv)
        ep["meta"]["composites"] = cd                                                 # U* size, excluded-gene count, NO_CALL list, PC1/FA diagnostics
        ep["files"]["features"] = frec(ff)
    except (FileNotFoundError, Blocked) as e:
        reasons.append(f"FEATURES: {type(e).__name__}: {e}")
    try:
        if bank is None:
            raise Blocked("no validated bank")
        got = False
        for f in sorted(glob.glob(f"{CATBANK}/{S}_scores_*-*.npz")):
            rec = json.load(open(Path(f).with_suffix(".receipt.json")))
            if rec["binding"]["config"]["bank_sha256"] != bank["sha256_members"] or rec.get("production_universe") is not True or rec.get("parity_verified") is not True:
                raise Blocked(f"SCORES_SHARD_NOT_BOUND_TO_BANK_OR_UNVERIFIED {f}")
            z = np.load(f, allow_pickle=True)
            srcs = [str(x).upper() for x in z["sources"]]
            if focal.upper() in srcs:
                receipt_of(f)                                                          # full sha check of the shard actually used (cached)
                i, ax = srcs.index(focal.upper()), {g: j for j, g in enumerate(z["genes"].astype(str))}
                for name, key in ((M_PAIR, "P|PM"), (M_CELL, "P|CU")):
                    v = np.asarray(z[key][i], float)
                    ep["V"][name] = np.array([v[ax[g]] if g in ax else np.nan for g in genes])
                ep["files"]["catbank_scores"] = frec(f)
                got = True
        if not got:
            reasons.append("CATBANK_SCORES: focal source not in any shard")
    except (FileNotFoundError, Blocked) as e:
        reasons.append(f"CATBANK_SCORES: {type(e).__name__}: {e}")
    ep["meta"]["catbank_reasons"] = reasons
    ep["adj_ok"] = ep["F1"] is not None and all(k in ep["V"] for k in list(GEN_PRIORS) + [GB_PRIOR])
    return ep


# U = testable & mappable
def assemble(system, tf, genes, V, truth, blk, meta, gb_sha_pre=None):
    U = truth["resp_ok"] & truth["occ_ok"]
    gsha = sha_arr(np.nan_to_num(V[GB], nan=-999.0))
    if gb_sha_pre is not None and gsha != gb_sha_pre:
        raise SystemExit("REFUSING: GeoBRIDGE vector changed between score construction and truth join")
    ep = dict(system=system, tf=tf, block=SYSTEMS[system]["block"], eid=f"{system}|{tf}", genes=np.asarray(genes).astype(str), V=dict(V), resp=truth["resp"],
              occ=truth["occ"], occ_ctl=truth.get("occ_ctl"), occ_dist=truth.get("occ_dist"), occ_score=truth.get("occ_score"), U=U, blk=blk, gi=blk.gi(genes), F1=None, adj_ok=False, files={}, meta=dict(meta))
    assert (truth["occ"]["prim"] <= truth["occ_ok"]).all() and (truth["resp"] <= truth["resp_ok"]).all()
    ep["fold"] = blk.fold_of(ep["gi"])
    u8 = lambda a: sha_arr(np.asarray(a, np.uint8))
    ep["meta"].update(n_genes=len(genes), n_resp_unknown=int((~truth["resp_ok"]).sum()), n_occ_unknown=int((~truth["occ_ok"]).sum()), n_U=int(U.sum()), gb_sha256=gsha,
                      truth_sha256=dict(resp=u8(truth["resp"]), resp_ok=u8(truth["resp_ok"]), occ_ok=u8(truth["occ_ok"]), occ_prim=u8(truth["occ"]["prim"]),
                                        occ_sens=u8(truth["occ"]["sens"]), genes=sha_arr(np.char.encode(np.asarray(genes).astype(str)))))
    return ep


# evaluation helpers
# contrast store: meta rows + draw vectors (float32); pointwise/simultaneous inference happens in merge
class CS:

    def __init__(self):
        self.meta, self.pt, self.bs = [], [], []

    def add(self, meta, pt, bs):
        if np.isfinite(pt):
            self.meta.append(meta)
            self.pt.append(float(pt))
            self.bs.append(np.asarray(bs, np.float32))

    def save(self, path):
        by = {}
        for i, b in enumerate(self.bs):
            by.setdefault(len(b), []).append(i)
        arrs, meta = {}, pd.DataFrame(self.meta)
        if len(meta):
            meta["pt"], meta["len"], meta["row"] = self.pt, [len(b) for b in self.bs], -1
            for L, idx in by.items():
                arrs[f"L{L}"] = np.stack([self.bs[i] for i in idx])
                meta.loc[idx, "row"] = np.arange(len(idx))
        tmp = f"{path}.tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, **arrs)
        os.replace(tmp, path)
        return meta


def pct(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return (float(np.percentile(x, 2.5)), float(np.percentile(x, 97.5)), len(x)) if len(x) else (np.nan, np.nan, 0)


# cached (point, draws) of one score on one (target, row set); rows are always a subset of the score's finite set
class Ev:

    def __init__(self, ep, B):
        self.ep, self.B, self.c = ep, B, {}

    def get(self, meth, y, rows, tag):
        k = (meth, tag, np.packbits(rows).tobytes())
        if k not in self.c:
            yy = np.asarray(y, bool)[rows]
            self.c[k] = CT.eval_rows(self.ep["V"][meth], y, rows, self.ep["gi"], self.ep["blk"].counts[:self.B]) if yy.any() and (~yy).any() else None
        return self.c[k]


def targets(ep, mask):
    r, o = ep["resp"], ep["occ"]
    T = {"resp": (r, mask)}
    for w in ("prim", "sens"):
        T[f"occ_{w}"] = (o[w], mask)
        T[f"AB_{w}"], T[f"CD_{w}"] = (r, mask & o[w]), (r, mask & ~o[w])
        T[f"AC_{w}"], T[f"BD_{w}"] = (o[w], mask & r), (o[w], mask & ~r)
        T[f"AD_{w}"] = (o[w], mask & ((o[w] & r) | (~o[w] & ~r)))                    # A (occupied responder) vs D (unoccupied non-responder)
    if ep.get("occ_ctl") is not None:
        T["occ_rIgG"] = (ep["occ_ctl"], mask)
    return T


# >=20 per contrasted class and >=20 independent resampling units
def gate_of(ep, y, rows, mask):
    y = np.asarray(y, bool)
    n1, n0 = int((y & rows).sum()), int((~y & rows).sum())
    ok = n1 >= MIN_CLASS and n0 >= MIN_CLASS and len(np.unique(ep["blk"].unit[ep["gi"][rows]])) >= MIN_UNITS
    if ep["system"] == "K562":
        o, r = ep["occ"]["prim"] & mask, ep["resp"]
        ok &= int(o.sum()) >= K562_MIN_BOUND and int((o & r).sum()) >= MIN_CLASS and int((o & ~r).sum()) >= MIN_CLASS
    return bool(ok)


def erow(ep, scope, paired, meth, tgt, y, rows, r, gate):
    yy = np.asarray(y, bool)[rows]
    base = dict(system=ep["system"], block=ep["block"], tf=ep["tf"], scope=scope, paired_with=paired, method=meth, target=tgt, n=int(rows.sum()),
                n_pos=int(yy.sum()), n_neg=int((~yy).sum()), gate=bool(gate))
    if r is None:
        return dict(**base, status="TOO_FEW")
    pt, bs = r
    row = dict(**base, status="OK" if gate else "DESCRIPTIVE_ONLY", prevalence=pt["PREV"])
    for k in ("AUROC", "AUPRC", "APLIFT"):
        lo, hi, nb = pct(bs[k]) if k in bs else (np.nan, np.nan, 0)
        row.update({k if k != "APLIFT" else "AP_lift": pt[k], f"{k}_lo": lo, f"{k}_hi": hi, f"nboot_{k}": nb})
    row.update({k: pt[k] for k in ("P20", "P50", "R20", "R50")})
    return row


def class_counts(ep, mask):
    r, o = ep["resp"], ep["occ"]
    d = {"n_rows": int(mask.sum())}
    for w in ("prim", "sens"):
        d.update({f"n_{c}_{w}": int(m.sum()) for c, m in zip("ABCD", (mask & o[w] & r, mask & o[w] & ~r, mask & ~o[w] & r, mask & ~o[w] & ~r))})
    return d


# RAW arm: pairwise (GB & comparator finite) and all-six masks
def lattice(ep, B, cs, rows_out, cov_out):
    V, U = ep["V"], ep["U"]
    gbf = np.isfinite(V[GB])
    ev = Ev(ep, B)
    cov_out.append(dict(system=ep["system"], tf=ep["tf"], scope="U", comparator="", **class_counts(ep, U),
                        **{f"cov|{m}": (float(np.isfinite(V[m][U]).mean()) if m in V and U.any() else np.nan) for m in [GB] + PAIR_METHODS}))
    masks = [("PAIRWISE", m, U & gbf & np.isfinite(V[m])) for m in PAIR_METHODS if m in V and (U & gbf & np.isfinite(V[m])).any()]
    six = all(m in V for m in SIX)
    if six:
        a6 = U & np.all([np.isfinite(V[m]) for m in SIX], 0)
        masks.append(("ALL6", None, a6))
        if all(m in V for m in SOTA):                                                # new common mask: ALL6 and every SOTA method finite (ALL6 itself unchanged)
            masks.append((SOTA_SCOPE, None, a6 & np.all([np.isfinite(V[m]) for m in SOTA], 0)))
        if all(m in V for m in REFS):                                                # one common mask for every REF: all six external methods and all 9 REFs finite
            a6r = a6 & np.all([np.isfinite(V[m]) for m in REFS], 0)
            cov_out.append(dict(system=ep["system"], tf=ep["tf"], scope="ALL6\u2229REF", comparator="(all 9 REF)", n_all6=int(a6.sum()), n_common=int(a6r.sum()),
                                n_dropped_vs_ALL6=int(a6.sum() - a6r.sum()), **class_counts(ep, a6r)))
            comps = [m for m in COMPS if m in V]
            if a6r.any():
                if not all(np.isfinite(V[m][a6r]).all() for m in comps):
                    raise Blocked("COMP composite non-finite on the ALL6\u2229REF mask (U* must contain it)")
                masks += [("ALL6\u2229REF", m, a6r) for m in REFS + comps]
    for m in PAIR_METHODS:
        if m not in V:
            rows_out.append(dict(system=ep["system"], block=ep["block"], tf=ep["tf"], scope="BOTH", method=m, status="NOT_RUN_MISSING_METHOD"))
    if not six:
        rows_out.append(dict(system=ep["system"], block=ep["block"], tf=ep["tf"], scope="ALL6", status="NOT_RUN_MISSING_METHOD",
                             method=",".join(m for m in SIX if m not in V)))
    if not (six and all(m in V for m in SOTA)):
        rows_out.append(dict(system=ep["system"], block=ep["block"], tf=ep["tf"], scope=SOTA_SCOPE, status="NOT_RUN_MISSING_METHOD",
                             method=",".join(m for m in SIX + SOTA if m not in V)))
    emitted = set()
    mself = U & gbf                                                                  # SELF: GeoBRIDGE's own ranking, all four-class contrasts (branch inference is built on these)
    if mself.any():
        cov_out.append(dict(system=ep["system"], tf=ep["tf"], scope="SELF", comparator="", **class_counts(ep, mself)))
        for t, (y, rows) in targets(ep, mself).items():
            g_, r_ = gate_of(ep, y, rows, mself), ev.get(GB, y, rows, t)
            rows_out.append(erow(ep, "SELF", "", GB, t, y, rows, r_, g_))
            if r_ and t.split("_")[0] in WITHIN_T and t.endswith(("_prim", "_sens")):
                cs.add(dict(family="WITHIN", scope="SELF", cmp=GB, stat=f"{t}|AUROC", system=ep["system"], block=ep["block"], tf=ep["tf"], gate=g_),
                       r_[0]["AUROC"], r_[1]["AUROC"])
    if all(m in V for m in REFS):                                                    # descriptive upper bound (point estimate only, no draws, never in inference)
        mh = U & gbf & np.all([np.isfinite(V[m]) for m in REFS], 0)
        hs_m = REFS + [m for m in COMPS if m in V]                                   # best observed comparator of the REF family (9 REF + present composites)
        pt1 = lambda meth, y, rows: CT.eval_rows(V[meth], y, rows, ep["gi"], ep["blk"].counts[:1])[0]
        for t, (y, rows) in targets(ep, mh).items():
            yy = np.asarray(y, bool)[rows]
            if t == "occ_rIgG" or not (yy.any() and (~yy).any()):
                continue
            au = {m: a_ for m in hs_m if np.isfinite(a_ := pt1(m, y, rows)["AUROC"])}     # non-finite AUROCs never enter the max
            if not au:
                continue
            best = max(au, key=au.get)
            gate_h = gate_of(ep, y, rows, mh)
            rows_out.append(dict(system=ep["system"], block=ep["block"], tf=ep["tf"], scope="HINDSIGHT", paired_with="", method=HINDSIGHT, target=t, n=int(rows.sum()),
                                 n_pos=int(yy.sum()), n_neg=int((~yy).sum()), gate=gate_h, status="DESCRIPTIVE_ONLY", AUROC=au[best], hindsight_ref=best,
                                 GB_AUROC_same_rows=pt1(GB, y, rows)["AUROC"], label="HINDSIGHT_UPPER_BOUND", descriptive=True,
                                 note="max over the 9 pre-declared REF AUROCs + the composites present on this target; selected with truth in hindsight; descriptive only, never in inference"))
    for scope, m, mask in masks:
        cov_out.append(dict(system=ep["system"], tf=ep["tf"], scope=scope, comparator=m or "", **class_counts(ep, mask)))
        T = targets(ep, mask)
        gates = {t: gate_of(ep, y, rows, mask) for t, (y, rows) in T.items()}
        meths = [GB] + ([m] if m else EXT + (SOTA if scope == SOTA_SCOPE else []))
        S = {}
        for meth in meths:
            for t, (y, rows) in T.items():
                S[meth, t] = ev.get(meth, y, rows, t)
                key = (scope, m or "", meth, t)
                if key not in emitted:
                    emitted.add(key)
                    rows_out.append(erow(ep, scope, m or "", meth, t, y, rows, S[meth, t], gates[t]))
        for meth in meths:                                                       # Delta_m = AUC_resp - AUC_occ (standalone, with CI)
            for w in ("prim", "sens"):
                a, b = S[meth, "resp"], S[meth, f"occ_{w}"]
                if a and b and gates["resp"] and gates[f"occ_{w}"]:
                    d = a[1]["AUROC"] - b[1]["AUROC"]
                    lo, hi, nb = pct(d)
                    rows_out.append(dict(system=ep["system"], block=ep["block"], tf=ep["tf"], scope=scope, paired_with=m or "", method=meth, target=f"delta_{w}",
                                         status="OK", AUROC=a[0]["AUROC"] - b[0]["AUROC"], AUROC_lo=lo, AUROC_hi=hi, nboot_AUROC=nb, gate=True))
        for meth in meths[1:]:
            cm = dict(family="RAW", scope=scope, cmp=meth, system=ep["system"], block=ep["block"], tf=ep["tf"])
            for t, (y, rows) in T.items():
                g, c = S[GB, t], S[meth, t]
                if not (g and c) or t == "occ_rIgG":
                    continue
                for k in BSK:
                    cs.add(dict(cm, stat=f"{t}|{k}", gate=gates[t]), g[0][k] - c[0][k], g[1][k] - c[1][k])
            for w in ("prim", "sens"):                                           # interaction I_m = Delta_GB - Delta_m
                gr, go, cr, co = S[GB, "resp"], S[GB, f"occ_{w}"], S[meth, "resp"], S[meth, f"occ_{w}"]
                if gr and go and cr and co:
                    f = lambda i: (gr[i]["AUROC"] - go[i]["AUROC"]) - (cr[i]["AUROC"] - co[i]["AUROC"])
                    cs.add(dict(cm, stat=f"I_{w}", gate=gates["resp"] and gates[f"occ_{w}"]), f(0), f(1))


# ADJ arm: nuisance ladder refit in every draw
def fit_pred(Xtr, ytr, wtr, Xte, ctx=None):
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import SplineTransformer, StandardScaler
    keep = np.ptp(Xtr, axis=0) > 1e-12
    if not keep.any():
        return np.full(len(Xte), np.average(ytr, weights=wtr))
    Xtr, Xte = Xtr[:, keep], Xte[:, keep]
    sc = StandardScaler().fit(Xtr, sample_weight=wtr)                              # training-only preprocessing, weighted = resampled
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)
    sp = SplineTransformer(n_knots=4, degree=3).fit(Ztr)
    lr = LogisticRegression(C=1.0, max_iter=2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr.fit(sp.transform(Ztr), ytr, sample_weight=wtr)
    return lr.predict_proba(sp.transform(Zte))[:, 1]


def oof(X, y, w, fold):
    pred = np.full(len(y), np.nan)
    for k in range(NFOLD):
        tr, te = (fold != k) & (w > 0), (fold == k) & (w > 0)
        if not te.any():
            continue
        if (w[tr] * y[tr]).sum() < 2 or (w[tr] * ~y[tr]).sum() < 2:
            return None
        pred[te] = fit_pred(X[tr], y[tr], w[tr], X[te])
    return pred


def pred_metrics(pred, y, w, n_rows):
    i = (w > 0) & np.isfinite(pred)
    if i.sum() == 0 or (w[i] * y[i]).sum() == 0 or (w[i] * ~y[i]).sum() == 0:
        return None
    p, yy, ww = pred[i], y[i], w[i].astype(float)
    o = np.argsort(-p, kind="mergesort")
    ps = p[o]
    out = CT.weighted_metrics(ww[o][None, :], yy[o], np.flatnonzero(np.r_[True, ps[1:] != ps[:-1]]), n_rows=n_rows)
    return {k: float(out[k][0]) for k in KEEP if k in out}


_LAD = {}


def ladder_draw(b, ctx=None):
    ctx = ctx or _LAD
    out = {}
    for t, T in ctx["T"].items():
        w = np.ones(len(T["y"])) if b < 0 else ctx["counts"][b, T["gi"]].astype(float)
        for m, X in T["X"].items():
            p = oof(X, T["y"], w, T["fold"])
            out[t, m] = None if p is None else pred_metrics(p, T["y"], w, len(T["y"]))
    return out


def _ladder_chunk(bs_):
    return [ladder_draw(b) for b in bs_]


# M0 = F1 -> M1 = F1 + generic priors -> M2 = M1 + P_GB -> M3 = M2 + full GB. nested by construction and asserted
def nested_models(F, Pg, pgb, gb):
    X = {"M0": F, "M1": np.column_stack([F, Pg]), "M2": np.column_stack([F, Pg, pgb]), "M3": np.column_stack([F, Pg, pgb, gb])}
    for a, b in zip(MODELS[:-1], MODELS[1:]):
        assert X[b].shape[1] > X[a].shape[1] and np.array_equal(X[b][:, :X[a].shape[1]], X[a], equal_nan=True), f"ladder not nested: {a} -> {b}"
    return X


# per-target row subsets / labels / gates for the ladder: resp = all rows
def ladder_targets(ep, mask, X, tset):
    U = ep["U"]
    r, o = ep["resp"][mask], ep["occ"]["prim"][mask]
    sub = dict(resp=np.ones(mask.sum(), bool), CD=~o, occ=np.ones(mask.sum(), bool), AB=o)
    yy = dict(resp=r, CD=r, occ=o, AB=r)
    pos_ix = np.flatnonzero(mask)
    T, YR, gates = {}, {}, {}
    for t in tset:
        s_ = sub[t]
        T[t] = dict(y=yy[t][s_], X={m: x[s_] for m, x in X.items()}, fold=ep["fold"][mask][s_], gi=ep["gi"][mask][s_])
        y_full, rows_t = np.zeros(len(U), bool), np.zeros(len(U), bool)
        rows_t[pos_ix[s_]] = True
        y_full[pos_ix[yy[t] & s_]] = True
        YR[t], gates[t] = (y_full, rows_t), gate_of(ep, y_full, rows_t, mask)
    return T, YR, gates


def run_ladder(T, counts, B, jobs):
    _LAD.clear()
    _LAD.update(dict(T=T, counts=counts))
    pts = ladder_draw(-1)
    ids = list(range(B))
    if jobs > 1:
        from concurrent.futures import ProcessPoolExecutor
        import multiprocessing as mp
        ch = [ids[i::jobs * 4] for i in range(jobs * 4)]
        with ProcessPoolExecutor(jobs, mp_context=mp.get_context("fork")) as ex:           # fork: _LAD (matrices + counts) is inherited copy-on-write
            res = dict(zip([i for c in ch for i in c], [d for part in ex.map(_ladder_chunk, ch) for d in part]))
    else:
        res = {b: ladder_draw(b) for b in ids}
    return pts, res, ids


# standalone model rows (AUROC, AUPRC, AP lift, P@20/P@50 where n >= 2K, pointwise CI) + increment contrasts a - b for every stored metric
def emit_ladder(ep, scope, fam, tag, tset, YR, gates, pts, res, ids, models, incr, cs, rows_out):
    bsd_all = {}
    for t in tset:
        y_full, rows_t = YR[t]
        bsd = {(m, k): np.array([(res[b][t, m] or {}).get(k, np.nan) for b in ids]) for m in models for k in BSK_ADJ}
        bsd_all[t] = bsd
        for m in models:
            if pts[t, m] is not None:
                rows_out.append(dict(erow(ep, scope, "", m, f"{tag}{t}", y_full, rows_t, (pts[t, m], {k: bsd[m, k] for k in BSK_ADJ}), gates[t]), descriptive=(fam == "DESC")))
            else:                                                                    # never a silent skip: the point fit failed (too few positives in a training fold / one class) 
                rows_out.append(dict(system=ep["system"], block=ep["block"], tf=ep["tf"], scope=scope, paired_with="", method=m, target=f"{tag}{t}", gate=gates[t],
                                     status="FAILED_TOO_FEW", n=int(rows_t.sum()), n_pos=int(y_full[rows_t].sum()), descriptive=(fam == "DESC")))
        for a, b in incr:
            if pts[t, a] is None or pts[t, b] is None:
                continue
            for k in BSK_ADJ:
                cs.add(dict(family=fam, scope=scope, cmp=f"{a}-{b}", stat=f"{t}|{k}", system=ep["system"], block=ep["block"], tf=ep["tf"], gate=gates[t]),
                       pts[t, a][k] - pts[t, b][k], bsd[a, k] - bsd[b, k])
    return bsd_all


# ADJ arm, separately per target (A>B within occupied genes, C>D within unoccupied genes; resp optional)
def adjusted(ep, B, cs, rows_out, cov_out, jobs=1, tset=("AB", "CD")):
    if not ep["adj_ok"]:
        rows_out.append(dict(system=ep["system"], block=ep["block"], tf=ep["tf"], scope="ADJ", status="NOT_RUN_CATBANK", method="; ".join(ep["meta"].get("catbank_reasons", []))))
        return
    V, U = ep["V"], ep["U"]
    fin = np.isfinite(ep["F1"]).all(1) & np.isfinite(V[GB]) & np.all([np.isfinite(V[k]) for k in list(GEN_PRIORS) + [GB_PRIOR]], 0)
    mask = U & fin                                                                   # one common row set for every model / target
    cov_out.append(dict(system=ep["system"], tf=ep["tf"], scope="ADJ", comparator="", **class_counts(ep, mask)))
    X = nested_models(ep["F1"][mask], np.column_stack([V[k][mask] for k in GEN_PRIORS]), V[GB_PRIOR][mask], V[GB][mask])
    T, YR, gates = ladder_targets(ep, mask, X, tset)
    pts, res, ids = run_ladder(T, ep["blk"].counts, B, jobs)
    emit_ladder(ep, "ADJ", "ADJ", "adj_", tset, YR, gates, pts, res, ids, MODELS, INCR, cs, rows_out)
    ev = Ev(ep, B)
    for t in tset:
        y_full, rows_t = YR[t]
        for meth in [GB, GB_PRIOR] + list(GEN_PRIORS):                             # raw standalone scores on the same rows (fair reference)
            rows_out.append(erow(ep, "ADJ", "", meth, f"adj_{t}", y_full, rows_t, ev.get(meth, y_full, rows_t, f"adj_{t}"), gates[t]))


# evaluation-only descriptors
def descriptors(ep, B, cs, rows_out, jobs=1, ladder=True):
    V, U = ep["V"], ep["U"]
    mask = U & np.isfinite(V[GB])
    dist, score = ep.get("occ_dist"), ep.get("occ_score")
    if dist is not None and np.isfinite(dist[mask]).any():
        ev = Ev(ep, B)
        for name, lo, hi in DIST_BINS:
            sel = mask & np.isfinite(dist) & (dist > lo) & (dist <= hi)
            rows_out.append(dict(erow(ep, "DESC", "", GB, f"dist_{name}", ep["resp"], sel, ev.get(GB, ep["resp"], sel, f"dist_{name}"), gate_of(ep, ep["resp"], sel, mask)), descriptive=True))
    else:
        rows_out.append(dict(system=ep["system"], block=ep["block"], tf=ep["tf"], scope="DESC", target="dist_bins", status="NOT_RUN_NO_DISTANCE", descriptive=True))
    if not ladder:
        return
    if ep["F1"] is None or score is None:
        rows_out.append(dict(system=ep["system"], block=ep["block"], tf=ep["tf"], scope="DESC", target="occ_score_model", status="NOT_RUN_NO_F1_OR_SCORE", descriptive=True))
        return
    fin = mask & np.isfinite(ep["F1"]).all(1) & np.isfinite(score)
    Fq = np.column_stack([ep["F1"][fin], score[fin]])
    X = {"Q0": Fq, "Q1": np.column_stack([Fq, V[GB][fin]])}
    assert np.array_equal(X["Q1"][:, :Fq.shape[1]], X["Q0"])                        # nested
    T, YR, gates = ladder_targets(ep, fin, X, ("AB",))
    pts, res, ids = run_ladder(T, ep["blk"].counts, B, jobs)
    emit_ladder(ep, "DESC", "DESC", "desc_", ("AB",), YR, gates, pts, res, ids, ["Q0", "Q1"], [("Q1", "Q0")], cs, rows_out)


# matching sensitivity
# 1:1 nearest control (Euclidean on F1 standardized by the combined-sample SD of positives and controls), replacement allowed
def match_1to1(F, pos, ctrl):
    from scipy.spatial import cKDTree
    Fp, Fc = F[pos], F[ctrl]
    allv = np.vstack([Fp, Fc])
    mu, sd = allv.mean(0), allv.std(0)
    sd[sd == 0] = 1.0
    Zp, Zc = (Fp - mu) / sd, (Fc - mu) / sd
    den = np.sqrt((Fp.var(0) + Fc.var(0)) / 2)
    den[den == 0] = 1.0
    before = np.abs(Fp.mean(0) - Fc.mean(0)) / den
    nb = cKDTree(Zc).query_ball_point(Zp, r=MATCH_CALIPER, p=np.inf)
    pi, ci = [], []
    for i, cand in enumerate(nb):
        if cand:
            cand = np.asarray(cand)
            pi.append(i)
            ci.append(int(cand[np.argmin(((Zc[cand] - Zp[i]) ** 2).sum(1))]))
    pi, ci = np.asarray(pi, int), np.asarray(ci, int)
    ret = len(pi) / max(len(pos), 1)
    if len(pi) == 0:
        return dict(status="NO_CALL_NO_MATCHES", retention=0.0, max_smd=np.nan, smd_before=before, smd_after=np.full(F.shape[1], np.nan), pi=pi, ci=ci)
    smd = np.abs(Fp[pi].mean(0) - Fc[ci].mean(0)) / den
    st = "OK" if ret >= MATCH_RETAIN and smd.max() <= MATCH_SMD else ("NO_CALL_RETENTION" if ret < MATCH_RETAIN else "NO_CALL_SMD")
    return dict(status=st, retention=float(ret), max_smd=float(smd.max()), smd_before=before, smd_after=smd, pi=pi, ci=ci)


# Weighted metrics of `score` on the matched set: each retained positive carries its draw multiplicity
def matched_eval(score, W_pos, pi_idx, ci_idx, B_chunk=400):
    import scipy.sparse as sp
    ucs, inv = np.unique(ci_idx, return_inverse=True)
    n_p = len(pi_idx)
    inc = sp.csr_matrix((np.ones(n_p), (np.arange(n_p), inv)), shape=(n_p, len(ucs)))
    sc = np.r_[score[pi_idx], score[ucs]]
    y = np.r_[np.ones(n_p, bool), np.zeros(len(ucs), bool)]
    o, starts = CT.prep_scores(sc)
    n_rows = n_p + len(ucs)
    pt = CT.weighted_metrics(np.r_[np.ones(n_p), np.asarray(inc.T @ np.ones(n_p))].astype(float)[None, o], y[o], starts, n_rows=n_rows)
    outs = []
    for a in range(0, W_pos.shape[0], B_chunk):
        Wp = W_pos[a:a + B_chunk].astype(float)
        Wc = np.asarray((inc.T @ Wp.T).T)
        outs.append(CT.weighted_metrics(np.hstack([Wp, Wc])[:, o], y[o], starts, n_rows=n_rows))
    return {k: float(v[0]) for k, v in pt.items()}, {k: np.concatenate([q[k] for q in outs]) for k in outs[0]}


MATCH_NOTE = {"resp": "changed estimand: responders vs their F1-nearest non-responders (retained responders only)",
              "CD": "changed estimand: unoccupied responders (C) vs their F1-nearest unoccupied non-responders (D), retained C only",
              "AB": "changed estimand: WITHIN BOUND GENES, each retained A gene vs its F1-nearest B gene (1:1, replacement); retained A only"}


# Sensitivity: 1:1 A-to-B (bound genes), C-to-D and responder-to-nonresponder matching on the frozen F1
def matching(ep, B, cs, rows_out, mrows):
    six = all(m in ep["V"] for m in SIX)
    if not (ep["adj_ok"] and six):
        mrows.append(dict(system=ep["system"], tf=ep["tf"], target="*", status="NOT_RUN_" + ("CATBANK" if not ep["adj_ok"] else "SIX_INCOMPLETE")))
        return
    V, o, r = ep["V"], ep["occ"]["prim"], ep["resp"]
    mask = ep["U"] & np.isfinite(ep["F1"]).all(1) & np.all([np.isfinite(V[m]) for m in SIX + list(GEN_PRIORS) + [GB_PRIOR]], 0)
    cols = F1X_COLS if FEATURE_TIER[0] == "F1x" else F1_COLS
    for t, (pm, cm) in dict(AB=(r & o, ~r & o), CD=(r & ~o, ~r & ~o), resp=(r, ~r)).items():
        pos, ctrl = np.flatnonzero(mask & pm), np.flatnonzero(mask & cm)
        base = dict(system=ep["system"], tf=ep["tf"], target=t, estimand=MATCH_NOTE[t], n_pos=len(pos), n_ctrl=len(ctrl))
        if len(pos) < MIN_CLASS or len(ctrl) < MIN_CLASS:
            mrows.append(dict(base, status="NO_CALL_SUPPORT", claim="NO_CALL: adjustment not claimed"))
            continue
        m = match_1to1(ep["F1"], pos, ctrl)
        mrows.append(dict(base, status=m["status"], retention=m["retention"], max_smd=m["max_smd"],
                          claim="matched-set result (conditional on the matched set)" if m["status"] == "OK" else "NO_CALL: adjustment not claimed"))
        for c_, b_, a_ in zip(cols, m["smd_before"], m["smd_after"]):
            mrows.append(dict(system=ep["system"], tf=ep["tf"], target=t, kind="balance", covariate=c_, smd_before=float(b_), smd_after=float(a_)))
        if m["status"] != "OK":
            continue
        pi, ci = pos[m["pi"]], ctrl[m["ci"]]
        Wp = ep["blk"].counts[:B][:, ep["gi"][pi]]
        R_ = {}
        for meth in SIX + list(GEN_PRIORS) + [GB_PRIOR]:
            R_[meth] = matched_eval(V[meth], Wp, pi, ci)
            pt, bs = R_[meth]
            rows_out.append(dict(system=ep["system"], block=ep["block"], tf=ep["tf"], scope="MATCH", method=meth, target=f"match_{t}", status="OK", n=len(pi) + len(np.unique(ci)),
                                 n_pos=len(pi), gate=True, AUROC=pt["AUROC"], AUROC_lo=pct(bs["AUROC"])[0], AUROC_hi=pct(bs["AUROC"])[1], AUPRC=pt["AUPRC"],
                                 AUPRC_lo=pct(bs["AUPRC"])[0], AUPRC_hi=pct(bs["AUPRC"])[1], retention=m["retention"], max_smd=m["max_smd"],
                                 interval_note="conditional on the matched set (pairs fixed on the original sample)", estimand=MATCH_NOTE[t]))
        for meth in EXT + list(GEN_PRIORS) + [GB_PRIOR]:
            for k in ("AUROC", "AUPRC"):
                cs.add(dict(family="MATCH", scope="MATCH", cmp=meth, stat=f"{t}|{k}", system=ep["system"], block=ep["block"], tf=ep["tf"], gate=True),
                       R_[GB][0][k] - R_[meth][0][k], R_[GB][1][k] - R_[meth][1][k])


# inference (merge stage)
def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty(len(p))
    run = 0.0
    for r, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - r) * p[i]))
        adj[i] = run
    return adj


def boot_p(pt, bs):
    b = np.asarray(bs, float)
    b = b[np.isfinite(b)]
    if len(b) < 100:
        return np.nan
    return float(min(1.0, 2 * min((np.sum(b <= 0) + 1) / (len(b) + 1), (np.sum(b >= 0) + 1) / (len(b) + 1))))


# simultaneous max-|t| bands: t_c(b) = (theta*_c(b) - theta_c)/se_c, se from the bootstrap SD; q = 1-ALPHA quantile of max_c |t_c(b)| over draws
def maxT(pts, BS):
    pts, BS = np.asarray(pts, float), np.asarray(BS, float)
    se = np.nanstd(BS, axis=1, ddof=1)
    good = np.isfinite(se) & (se > 0)
    t = np.abs(BS - pts[:, None]) / np.where(good, se, np.nan)[:, None]
    t = np.where(np.isfinite(t) & good[:, None], t, -np.inf)
    mx = t.max(0)
    mx = mx[np.isfinite(mx)]
    q = float(np.quantile(mx, 1 - ALPHA)) if len(mx) else np.nan
    return q, se, np.where(good, pts - q * se, np.nan), np.where(good, pts + q * se, np.nan)


# ent: [(block, tf, pt, bs)] -> {level: (pt, bs, n_endpoints)}
def agg_levels(ent):
    by = {}
    for b, tf, pt, bs in ent:
        by.setdefault(b, []).append((pt, bs))
    out, bp, bb = {}, [], []
    for b, v in sorted(by.items()):
        pt, bs = np.mean([x[0] for x in v]), np.mean(np.stack([x[1] for x in v]), 0)
        out[b] = (pt, bs, len(v))
        bp.append(pt)
        bb.append(bs)
    if by:
        out["ALL"] = (float(np.mean(bp)), np.mean(np.stack(bb), 0), sum(len(v) for v in by.values()))
    return out


SIM_RAW_STATS = ["resp|AUROC", "resp|AUPRC", "occ_prim|AUROC", "I_prim", "AB_prim|AUROC", "AB_prim|AUPRC", "CD_prim|AUROC", "CD_prim|AUPRC",
                 "AC_prim|AUROC", "BD_prim|AUROC", "AD_prim|AUROC"]
PER_TF_STATS = {"RAW": ["resp|AUROC", "I_prim", "AB_prim|AUROC", "CD_prim|AUROC", "AC_prim|AUROC", "BD_prim|AUROC", "AD_prim|AUROC", "AB_prim|AUPRC", "CD_prim|AUPRC"],
                "ADJ": ["resp|AUPRC", "AB|AUPRC", "CD|AUPRC"]}
PER_TF_CMP = {"RAW": EXT, "ADJ": ["M2-M1", "M3-M2", "M3-M1"]}


# meta: shard contrast index (+pt,len,row); arrs: {L: 2D}
def macro(meta, arrs):
    g = meta[meta.gate.astype(bool) & meta.family.isin(["RAW", "ADJ", "MATCH", "DESC"])].copy()          # within (branch) contrasts are handled by within_branches
    rows, sims, ptf = [], {}, []
    key = ["family", "scope", "cmp", "stat"]
    for k, d in g.groupby(key, sort=True):
        ent = [(r.block, r.tf, r.pt, arrs[r.shard][f"L{r.len}"][r.row]) for r in d.itertuples()]
        fam, scope, cmp_, stat = k
        for lvl, (pt, bs, n) in agg_levels(ent).items():
            nc = float(np.isfinite(bs).mean())
            sim = (fam == "RAW" and stat in SIM_RAW_STATS and lvl == "ALL") or (fam == "ADJ" and stat.split("|")[0] in ("resp", "CD", "AB"))
            fam_key = (f"RAW|{scope}" + ("|REF" if cmp_ in REFS + COMPS else "|SOTA" if cmp_ in SOTA and scope == "PAIRWISE" else "")) if fam == "RAW" else "ADJ" if fam == "ADJ" else None      # REFs form their own max-|t| family
            lo, hi, nb = pct(bs)
            row = dict(family=fam, scope=scope, comparator=cmp_, stat=stat, level=lvl, n_endpoints=n, est=pt, se=float(np.nanstd(bs, ddof=1)) if nc > 0 else np.nan,
                       lo_point=lo, hi_point=hi, frac_complete_draws=nc, inference="simultaneous_maxT" if sim and fam_key else "pointwise")
            rows.append(row)
            if sim and fam_key and nc >= 0.95:
                sims.setdefault(fam_key, []).append((len(rows) - 1, pt, bs))
        if fam in PER_TF_STATS and stat in PER_TF_STATS[fam] and cmp_ in PER_TF_CMP[fam] and (fam == "ADJ" or scope == "PAIRWISE"):
            for e in ent:
                ptf.append(dict(family=fam, scope=scope, comparator=cmp_, stat=stat, block=e[0], tf=e[1], est=e[2], p=boot_p(e[2], e[3])))
    C = pd.DataFrame(rows)
    for c in ("lo_simul", "hi_simul", "q_maxT"):
        C[c] = np.nan
    for fk, v in sims.items():
        idx = [x[0] for x in v]
        q, se, lo, hi = maxT([x[1] for x in v], np.stack([x[2] for x in v]))
        C.loc[idx, ["lo_simul", "hi_simul", "q_maxT"]] = np.c_[lo, hi, np.full(len(idx), q)]
        C.loc[idx, "sim_family"] = fk
    P = pd.DataFrame(ptf)
    if len(P):
        P["claim"] = P.family + "|" + P.stat                                            # Holm families = claim types (family|stat) across comparators and TFs; a global-Holm column is also given
        P["holm_p"] = np.nan
        for _, ix in P.groupby("claim").groups.items():
            P.loc[ix, "holm_p"] = holm(P.loc[ix, "p"].fillna(1.0).to_numpy())
        P["holm_p_global"] = holm(P.p.fillna(1.0).to_numpy())
        P["holm_reject"] = P.holm_p < ALPHA
    return C, P


BRANCH_SENT = {
    "I": "Ranking signature: occupancy-supported responders rank above occupancy-supported non-responders, above unoccupied responders and above unoccupied non-responders (A>B, A>C, A>D); "
         "this is enrichment for occupancy-supported responders and does not show that the assigned binding caused the response.",
    "II": "Occupancy-neutral score discrimination within the measured annotation: A>B and C>D with A~=C and B~=D under the +-.05 AUROC equivalence bound "
          "(not 'functional response independent of occupancy'; distal or undetected binding is not excluded).",
    "III": "Responders without assigned proximal occupancy rank above their non-responsive counterparts and above occupancy-supported responders (C>D and C>A); "
           "no direct-versus-indirect claim.",
    "UNRESOLVED": "No frozen branch condition is met: not demonstrated."}
REL_TXT = {">": "superior (AUROC >= .55 and simultaneous lower bound > .50)", "\u2248": "equivalent (interval inside [.45,.55])", "<": "inferior (simultaneous upper bound < .50)", "unresolved": "not demonstrated", "NA": "not evaluable"}


# .55 AND simultaneous lower bound > .50; '≈' iff the 95% interval lies inside [.45,.55]; '<' iff the simultaneous upper bound is < .50
def relation(est, lo, hi):
    if not np.isfinite([est, lo, hi]).all():
        return "NA"
    if est >= SUP_POINT and lo > SUP_LB:
        return ">"
    if lo >= EQ_LO and hi <= EQ_HI:
        return "\u2248"
    if hi < SUP_LB:
        return "<"                                                                    # simultaneous upper bound below .50 (equivalence keeps priority; branch logic unchanged)
    return "unresolved"


# every branch whose conditions hold (not forced exclusive) else UNRESOLVED
def held_branches(rel, hi_ac):
    I = rel["AB"] == ">" and rel["AC"] == ">" and rel["AD"] == ">"
    II = rel["AB"] == ">" and rel["CD"] == ">" and rel["AC"] == "\u2248" and rel["BD"] == "\u2248"
    III = rel["CD"] == ">" and np.isfinite(hi_ac) and hi_ac < 0.50
    return [n for n, ok in (("I", I), ("II", II), ("III", III)) if ok] or ["UNRESOLVED"]


# Per TF, per SYSTEM and per BLOCK (GATA1+KLF1 = one), per window: GB's within-method AUROC for A>B, C>D, A>C, B>D
def within_branches(meta, arrs):
    g = meta[(meta.family == "WITHIN") & meta.gate.astype(bool)]
    out = []
    for win in ("prim", "sens"):
        d = g[g.stat.isin([f"{t}_{win}|AUROC" for t in WITHIN_T])]
        ent = {}
        for r in d.itertuples():
            ent.setdefault((r.system, r.tf, r.block), {})[r.stat.split("_")[0]] = (r.pt, arrs[r.shard][f"L{r.len}"][r.row])
        units = {}
        for (sy, tf, bl), v in ent.items():
            units.setdefault(("TF", f"{sy}|{tf}"), []).append((sy, v))
            units.setdefault(("SYSTEM", sy), []).append((sy, v))
            units.setdefault(("BLOCK", bl), []).append((sy, v))
        for (lvl, name), lst in sorted(units.items()):
            if lvl != "TF" and len(lst) < 2:                                          # SYSTEM/BLOCK rows only where they aggregate >1 TF (else identical to the TF row)
                continue
            if lvl == "BLOCK" and len({sy for sy, _ in lst}) < 2:
                continue
            vs = [v for _, v in lst]
            ts = [t for t in WITHIN_T if all(t in v for v in vs)]
            row = dict(level=lvl, unit=name, window=win, primary_window=(win == "prim"), n_tf=len(vs), margins=f"> : AUROC>={SUP_POINT} & simult LB>{SUP_LB}; ~= : interval in [{EQ_LO},{EQ_HI}]")
            rel, hi_ac = {t: "NA" for t in WITHIN_T}, np.nan
            if ts:
                pts = np.array([np.mean([v[t][0] for v in vs]) for t in ts])
                BS = np.stack([np.mean(np.stack([v[t][1] for v in vs]), 0) for t in ts])
                comp = float(np.isfinite(BS).all(0).mean())
                q, se, lo, hi = maxT(pts, BS)
                row.update(q_maxT=q, frac_complete_draws=comp)
                for i, t in enumerate(ts):
                    plo, phi, _ = pct(BS[i])
                    ok = comp >= 0.95
                    row.update({f"{t}_auroc": pts[i], f"{t}_lo_simul": lo[i], f"{t}_hi_simul": hi[i], f"{t}_lo_point": plo, f"{t}_hi_point": phi})
                    rel[t] = relation(pts[i], lo[i], hi[i]) if ok else "NA"
                    row[f"{t}_equiv_pointwise"] = bool(plo >= EQ_LO and phi <= EQ_HI)
                    if t == "AC":
                        hi_ac = hi[i] if ok else np.nan
            row.update({f"{t}_rel": rel[t] for t in WITHIN_T})
            held = held_branches(rel, hi_ac)
            row["branches"] = ",".join(held)
            row["sentence"] = " | ".join(f"[{b}] {BRANCH_SENT[b]}" for b in held)
            row["not_evaluable"] = ",".join(t for t in WITHIN_T if rel[t] == "NA")
            out.append(row)
    return pd.DataFrame(out)


# adjusted-ladder increment statements per target (A>B, C>D, resp): M1-M0 (generic priors), M2-M1 (P_GB), M3-M2 (full score), M3-M1
def increments(C):
    rows = []
    A = C[C.family == "ADJ"] if len(C) else C
    for tgt in sorted({s.split("|")[0] for s in A.stat}) if len(A) else []:
        for a, b in INCR:
            x = A[(A.comparator == f"{a}-{b}") & (A.level == "ALL")]
            row = dict(target=tgt, increment=f"{a}-{b}")
            for met in BSK_ADJ:
                r = x[x.stat == f"{tgt}|{met}"]
                if len(r):
                    row.update({f"{met}_est": r.est.iloc[0], f"{met}_lo_simul": r.lo_simul.iloc[0], f"{met}_hi_simul": r.hi_simul.iloc[0]})
            lo, hi = row.get("AUPRC_lo_simul", np.nan), row.get("AUPRC_hi_simul", np.nan)
            row["statement"] = ("not evaluable" if not np.isfinite(lo) else "increment demonstrated (simultaneous LB of dAUPRC > 0)" if lo > 0 else
                                f"an increment of at least {DELTA_AP} AUPRC is excluded (simultaneous UB < margin)" if hi < DELTA_AP else "not demonstrated")
            if (a, b) == ("M3", "M1"):
                npass = nub = nev = 0
                for bl in BLOCKS:
                    ap = A[(A.comparator == "M3-M1") & (A.level == bl) & (A.stat == f"{tgt}|AUPRC")]
                    p5 = A[(A.comparator == "M3-M1") & (A.level == bl) & (A.stat == f"{tgt}|P50")]
                    if len(ap) and len(p5) and np.isfinite([ap.lo_simul.iloc[0], p5.lo_simul.iloc[0], ap.hi_simul.iloc[0], p5.hi_simul.iloc[0]]).all():
                        nev += 1
                        npass += int(ap.lo_simul.iloc[0] > DELTA_AP and p5.lo_simul.iloc[0] > DELTA_P50)
                        nub += int(ap.hi_simul.iloc[0] < DELTA_AP and p5.hi_simul.iloc[0] < DELTA_P50)
                row.update(utility_status=("NO_CALL_LT2_BLOCKS" if nev < 2 else "UTILITY_SUPPORTED" if npass >= 2 else "UTILITY_MARGIN_EXCLUDED" if nub >= 2 else "not demonstrated"),
                           utility_blocks_pass=npass, utility_blocks_ub_below=nub, utility_blocks_evaluable=nev,
                           utility_margins=f"dAUPRC>{DELTA_AP} and dP@50>{DELTA_P50} vs joint baseline M1, simultaneous bounds, distinct blocks (GATA1+KLF1 = one)")
            rows.append(row)
    return pd.DataFrame(rows)


# method-comparison statements (GB - method) for A>B and C>D, kept separate from the branch statements: macro (simultaneous, cross-block) and per-TF
def method_statements(C, P):
    rows = []
    if len(C):
        for r in C[(C.family == "RAW") & C.stat.isin(["AB_prim|AUROC", "AB_prim|AUPRC", "CD_prim|AUROC", "CD_prim|AUPRC"]) & (C.level == "ALL") & C.comparator.isin(EXT)].itertuples():
            st = "GB exceeds the method (simultaneous LB > 0)" if r.lo_simul > 0 else "the method exceeds GB (simultaneous UB < 0)" if r.hi_simul < 0 else "not demonstrated"
            rows.append(dict(level="macro", scope=r.scope, target=r.stat.split("_")[0], metric=r.stat.split("|")[1], comparator=r.comparator, est=r.est, lo=r.lo_simul, hi=r.hi_simul,
                             inference="simultaneous_maxT", statement=st))
    if len(P):
        for r in P[P.stat.str.startswith(("AB_prim", "CD_prim"))].itertuples():
            st = ("GB exceeds the method (Holm)" if r.est > 0 else "the method exceeds GB (Holm)") if r.holm_reject else "not demonstrated"
            rows.append(dict(level=f"TF {r.block}|{r.tf}", scope=r.scope, target=r.stat.split("_")[0], metric=r.stat.split("|")[1], comparator=r.comparator, est=r.est, p=r.p, holm_p=r.holm_p,
                             inference="per-TF Holm (claim family)", statement=st))
    return pd.DataFrame(rows)


# driver
# order (truth_read_after_scores): 1 comparator + GB primary scores, 2 catbank priors/features/M/cell, 3 only then the truth tables
def build_endpoint(system, tf, blk, smalln):
    cfg = SYSTEMS[system]
    if system == "KLF1":
        genes, V, miss = scores_klf1()
        cfiles = {}
    else:
        genes, V, miss, _, cfiles = scores_prod(system, tf)
    pre = dict(genes=np.asarray(genes).astype(str), V=V, files={}, meta={}, F1=None)
    gb_sha = sha_arr(np.nan_to_num(V[GB], nan=-999.0))
    catbank_attach(pre, cfg["axis"], tf, smalln)
    if system == "KLF1":                                                              # truth is opened below this line only
        tr, mf = truth_klf1(genes)
    elif system in ("GATA1", "PANC"):
        tr, mf = {"GATA1": truth_gata1, "PANC": truth_panc}[system](genes)
    else:
        tr, mf = truth_bm(tf, genes) if system == "BONEMARROW" else truth_k562(tf, genes)
    if miss:
        mf["missing_methods"] = miss
    if system != "KLF1":
        mf["comparator_files"] = cfiles
    ep = assemble(system, tf, pre["genes"], pre["V"], tr, blk, dict(manifest=mf, **pre["meta"]), gb_sha)
    ep["F1"], ep["adj_ok"], ep["files"] = pre["F1"], pre["adj_ok"], pre["files"]
    ep["meta"]["truth_loaded_after"] = ["scores", "catbank"]
    return ep


def module_shas():
    out = {}
    for n in ("gb_canon_table", "gb_primary", "gb_catbank", "gb_harmonized_tri", "gb_real_uniform", "gb_cop_alignment", "gb_truth_v2_load", "gb_klf1_eval", "gb_bm_sweep", "gb_composites"):
        m = sys.modules.get(n)
        if m is not None and getattr(m, "__file__", None):
            out[n] = sha_file(m.__file__)
    return out


def run_shard(system, out, B, B_adj, jobs, tfs, smalln, adj=True, tset=("AB", "CD")):
    os.makedirs(f"{out}/mapping", exist_ok=True)
    cfg = SYSTEMS[system]
    if B_adj > B:
        raise SystemExit("REFUSING: --B-adj must be <= --B (ADJ draws are the first B_adj rows of the shared resample matrix)")
    if tfs:
        allowed = cfg["tfs"] if cfg["tfs"] is not None else k562_tfs()
        bad = [t for t in tfs if t not in allowed]
        if bad:
            raise SystemExit(f"REFUSING: --tfs {bad} not in the declared TF list of {system}: {allowed}")
    axis = axis_of(cfg["axis"])
    units, scheme = load_modules(cfg["axis"], axis)
    seed = int(np.random.SeedSequence([BOOT_SEED, BLOCKS.index(cfg["block"])]).generate_state(1)[0])
    blk = BlockAxis(axis, B, seed, units, scheme)
    tf_list = tfs or cfg["tfs"] or k562_tfs()
    cs, rows, cov, mrows, mfs = CS(), [], [], [], {}
    for tf in tf_list:
        t0 = time.time()
        if system == "K562" and not os.path.exists(K562_BLACKLIST):                    # checked before any scoring
            msg = f"K562 NOT_RUN: ENCODE hg38 blacklist {K562_BLACKLIST} is absent (fetch ENCFF356LFX there); blacklist filtering is never skipped"
            rows.append(dict(system=system, block=cfg["block"], tf=tf, scope="ALL", method="", status=msg))
            print(f"[lattice] {system}/{tf}: {msg}", flush=True)
            continue
        try:
            ep = build_endpoint(system, tf, blk, smalln)
        except Blocked as e:
            rows.append(dict(system=system, block=cfg["block"], tf=tf, scope="ALL", method="", status=f"NOT_RUN: {e}"))
            print(f"[lattice] {system}/{tf}: NOT_RUN {e}", flush=True)
            continue
        mfs[tf] = dict(ep["meta"], files={k: v for k, v in ep["files"].items()})
        lattice(ep, B, cs, rows, cov)
        if adj:
            adjusted(ep, B_adj, cs, rows, cov, jobs, tset)
            matching(ep, B, cs, rows, mrows)
        descriptors(ep, B_adj, cs, rows, jobs, ladder=adj)
        print(f"[lattice] {system}/{tf}: U={ep['meta']['n_U']} done {time.time()-t0:.0f}s", flush=True)
    tag = "all" if not tfs else "_".join(tfs)
    meta = cs.save(f"{out}/shard_{system}_{tag}.npz")
    meta["shard"], meta["counts_fp"] = f"shard_{system}_{tag}.npz", blk.fingerprint()
    meta["script_sha256"], meta["B"], meta["B_adj"] = sha_file(__file__), B, B_adj
    atomic_tsv(meta, f"{out}/shard_{system}_{tag}.index.tsv")
    atomic_tsv(pd.DataFrame(rows), f"{out}/part_endpoints_{system}_{tag}.tsv")
    atomic_tsv(pd.DataFrame(cov), f"{out}/part_coverage_{system}_{tag}.tsv")
    atomic_tsv(pd.DataFrame(mrows), f"{out}/part_matching_{system}_{tag}.tsv")
    atomic_json(f"{out}/mapping/{system}_{tag}_mapping_manifest.json", dict(system=system, block=cfg["block"], block_scheme=scheme, windows=WIN[system], B=B, B_adj=B_adj,
                                                                               seed=seed, counts_fingerprint=blk.fingerprint(), script_sha256=sha_file(__file__), module_shas=module_shas(),
                                                                               feature_tier=FEATURE_TIER[0], adj_targets=list(tset) if adj else [], matching_note="matched-set intervals are CONDITIONAL on the matched set (pairs fixed on the original sample)",
                                                                               endpoints=mfs))


# composites that are NO_CALL (absent) in any endpoint: not reported at the macro / cross-block level (per-endpoint rows stay in endpoints.tsv)
def nocall_comps(ep_rows):
    d = pd.DataFrame(ep_rows)
    if not len(d) or "method" not in d:
        return set()
    return set(d[(d.status == "NOT_RUN_MISSING_METHOD") & d.method.isin(COMPS)].method)


def merge(out):
    import glob
    idx = sorted(glob.glob(f"{out}/shard_*.index.tsv"))
    if not idx:
        raise SystemExit("REFUSING_MERGE: no shards")
    metas, arrs, fps = [], {}, {}
    for f in idx:
        m = pd.read_csv(f, sep="\t")
        name = m.shard.iloc[0] if len(m) else os.path.basename(f).replace(".index.tsv", ".npz")
        if len(m):
            arrs[name] = dict(np.load(f"{out}/{name}"))
            metas.append(m)
            fps.setdefault(m.block.iloc[0], set()).add(m.counts_fp.iloc[0])
    mans = [json.load(open(f)) for f in sorted(glob.glob(f"{out}/mapping/*_mapping_manifest.json"))]
    for k in ("script_sha256", "B", "B_adj", "feature_tier", "seed"):
        vals = {(m["block"], m[k]) if k == "seed" else m[k] for m in mans}
        if (k == "seed" and len({b for b, _ in vals}) != len(vals)) or (k != "seed" and len(vals) > 1):
            raise SystemExit(f"REFUSING_MERGE_MIXED_{k.upper()}: {sorted(map(str, vals))}")
    ms = {}
    for m in mans:
        for n, h in m.get("module_shas", {}).items():
            ms.setdefault(n, set()).add(h)
    if any(len(v) > 1 for v in ms.values()):
        raise SystemExit(f"REFUSING_MERGE_MIXED_MODULE_SHAS: {[n for n, v in ms.items() if len(v) > 1]}")
    if not metas:
        raise SystemExit("REFUSING_MERGE: no contrast rows in any shard")
    allm = pd.concat(metas, ignore_index=True)
    for k in ("script_sha256", "B", "B_adj"):
        if allm[k].nunique() > 1:
            raise SystemExit(f"REFUSING_MERGE_MIXED_{k.upper()} in shard indices")
    dupk = allm.duplicated(["system", "tf", "family", "scope", "cmp", "stat"], keep=False)
    if dupk.any():
        raise SystemExit(f"REFUSING_MERGE_REPEATED_KEYS across shards: {allm[dupk][['system', 'tf', 'family', 'scope', 'cmp', 'stat']].drop_duplicates().head(5).to_dict('records')}")
    bad = {b: s for b, s in fps.items() if len(s) != 1}
    if bad:
        raise SystemExit(f"REFUSING_MERGE_DESYNCHRONISED_DRAWS: {bad}")                # shards of one block must share the same resample matrix
    pe = []
    for f in sorted(glob.glob(f"{out}/part_endpoints_*.tsv")):
        try:
            pe.append(pd.read_csv(f, sep="\t"))
        except pd.errors.EmptyDataError:
            pass
    nc = nocall_comps(pd.concat(pe, ignore_index=True) if pe else [])
    C, P = macro(allm[~allm.cmp.isin(nc)] if nc else allm, arrs)          # a composite NO_CALL in any endpoint is not a macro / cross-block comparator
    Br, Inc, Ms_ = within_branches(allm, arrs), increments(C), method_statements(C, P)
    def cat(pat):
        parts = []
        for f in sorted(glob.glob(f"{out}/{pat}")):
            try:
                parts.append(pd.read_csv(f, sep="\t"))
            except pd.errors.EmptyDataError:                                          # a shard with no rows of this kind (e.g. no matching) writes an empty table
                pass
        return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()

    atomic_tsv(cat("part_endpoints_*.tsv"), f"{out}/endpoints.tsv")
    atomic_tsv(cat("part_coverage_*.tsv"), f"{out}/coverage.tsv")
    atomic_tsv(cat("part_matching_*.tsv"), f"{out}/matching.tsv")
    atomic_tsv(C, f"{out}/contrasts.tsv")
    atomic_tsv(P, f"{out}/per_tf_holm.tsv")
    atomic_tsv(Br, f"{out}/branches.tsv")
    atomic_tsv(Inc, f"{out}/increments.tsv")
    atomic_tsv(Ms_, f"{out}/method_comparisons.tsv")
    inputs = {}
    for k, p in (("spec_log", SPEC_LOG), ("this_script", __file__)):
        inputs[k] = sha_file(p) if os.path.exists(p) else None
    inputs.update({f"module:{n}": sorted(v)[0] for n, v in ms.items()})               # dependency module shas as recorded at shard time
    mans = {os.path.basename(f): json.load(open(f)) for f in sorted(glob.glob(f"{out}/mapping/*_mapping_manifest.json"))}
    atomic_json(f"{out}/receipt.json", dict(created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), inputs_sha256=inputs, shards=idx, counts_fingerprints={b: sorted(s) for b, s in fps.items()},
                                            mapping_manifests=mans, boot_seed=BOOT_SEED, margins=dict(dAP=DELTA_AP, dP50=DELTA_P50),
                                            branch_margins=dict(superiority=f"AUROC point >= {SUP_POINT} AND simultaneous lower bound > {SUP_LB}", equivalence=f"interval inside [{EQ_LO}, {EQ_HI}] (simultaneous interval; pointwise flag also given)",
                                                                branch_I="A>B and A>C and A>D", branch_II="A>B and C>D and A~=C and B~=D", branch_III="C>D and A>C upper simultaneous bound < .50",
                                                                rule="all branches whose conditions hold are reported (not exclusive), else UNRESOLVED; 'not demonstrated' never 'no increment' unless an equivalence bound excludes the utility margin"),
                                            reference_rankers=dict(signs={n: dict(feature=c, sign=sg) for n, (c, sg) in REF_SIGNS.items()},
                                                                   rule="unsupervised single-feature rankers, no fitting, sign fixed a priori (never chosen from truth); source catbank SYSTEM_features.tsv (receipt-validated); "
                                                                        "signs match gb_final_benchmark REF rows (-detection, -pair_support, +dynamics) and their nascent analogues; "
                                                                        "evaluated like the external methods (paired GB - REF contrasts, same bootstrap) in their OWN simultaneous families on PAIRWISE masks and on ONE common mask ALL6\u2229REF (all six external methods and all 9 REFs finite)",
                                                                   composites=dict(names=COMPS, spec="gb_composites: COMP_Z mean of column z-scores; COMP_RANK mean midrank fraction; COMP_PC1 leading PC of Z'Z/n (sum>0, top-2 tie -> NO_CALL); COMP_FA one-factor ML FA (NO_CALL if n_iter_==max_iter OR min unique variance Psi < 0.01 over the non-constant columns, i.e. Heywood); fit on U* = axis genes with all nine features finite (per-endpoint meta.composites); members of the REF families",
                                                                   fit_universe="frozen observational axis of the system, truth-free; per-endpoint fit universe + diagnostics (n_universe, n_Ustar, n_excluded, no_call, PC1 loadings/eigenvalues, FA loadings/Psi/loglik change/n_iter, FA no-call reason "
                                                                                                       "[nonconvergence | heywood min Psi<.01]) are in mapping_manifests[*].endpoints[tf].composites",
                                                                   macro_excluded_no_call=sorted(nc),
                                                                   macro_rule="a composite that is NO_CALL in ANY endpoint is dropped from the macro / cross-block families (per-endpoint rows stay in endpoints.tsv as NOT_RUN_MISSING_METHOD)"),
                                                                   hindsight=f"'{HINDSIGHT}' rows = max over the 9 REF + composite AUROCs per target, label HINDSIGHT_UPPER_BOUND, descriptive only, never in inference"),
                                            targets=dict(raw=["resp", "occ", "AB", "CD", "AC", "BD", "AD"], windows=["prim", "sens"], adjusted=sorted({t for m in mans.values() for t in m.get("adj_targets", [])}),
                                                         descriptors=["distance bins (0-1 kb, 1-10 kb, >10 kb) response discrimination", "F1+occ_score vs +GB within bound genes"],
                                                         descriptors_note="evaluation-only, declared descriptive; no bin or covariate can become primary", matching=["AB", "CD", "resp"]),
                                            multiplicity_families=dict(simultaneous_maxT=["RAW|ALL6 and RAW|PAIRWISE (cross-block macro; external methods, M, cell, priors): resp/occ/I/AB/CD/AC/BD/AD stats x comparators",
                                                                                       "RAW|PAIRWISE|REF and RAW|ALL6\u2229REF|REF: the 9 truth-free REF comparators form their OWN families and do not widen the GB-vs-external bands", "RAW|PAIRWISE|SOTA: the 4 extra RNA-only comparators (RegDiffusion, DeepSEM, PIDC, FastTENET) pairwise, their OWN family; RAW|ALL6+SOTA: ONE family over the 5 frozen externals + the 4 SOTA methods on the common mask (RAW|ALL6 / RAW|PAIRWISE unchanged)", "ADJ: targets AB, CD (resp optional) x increments x metrics, at ALL and per block",
                                                                                       "WITHIN (branches): the 5 within-method contrasts per (TF | SYSTEM | BLOCK) unit and window"],
                                                                       holm_per_TF="one Holm family per claim type (family|stat) across comparators and TFs; a global-Holm column is also written", pointwise="everything else (sensitivity window, MATCH, DESC, per-block RAW)"), truth_read_after_scores=True, truth_order="scores -> catbank -> truth (build_endpoint)", matching_note="matched-set intervals are conditional on the matched set", B=[m["B"] for m in mans.values()][:1], B_adj=[m["B_adj"] for m in mans.values()][:1],
                                            python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, min_class=MIN_CLASS, min_units=MIN_UNITS,
                                            note="PRELIMINARY"))
    print(Br[[c for c in ('level', 'unit', 'window', 'AB_rel', 'CD_rel', 'AC_rel', 'BD_rel', 'AD_rel', 'branches') if c in Br]].to_string(index=False) if len(Br) else 'no branch rows')
    print(f"[lattice] merged {len(idx)} shards -> {out}")


# selftest (synthetic only)
def _syn_block(G=600, B=120, seed=0, units=None):
    return BlockAxis([f"g{i}" for i in range(G)], B, seed, units)


def _syn_ep(blk, tf, sysname="GATA1", rng=None, signal=0.9, adj=True, resp_rate=.25, occ_rate=.3):
    rng = rng or np.random.default_rng(1)
    G = len(blk.names)
    genes = blk.names
    F = rng.normal(size=(G, len(F1_COLS)))
    occ = rng.random(G) < occ_rate
    resp = rng.random(G) < np.where(occ, resp_rate * 1.2, resp_rate)
    u = rng.normal(size=G)
    gbs = signal * resp + rng.normal(size=G)                                 # GB carries response signal, none for occupancy
    V = {GB: gbs, "|Pearson|": 0.7 * occ + rng.normal(size=G), "GENIE3": .5 * resp + .7 * occ + rng.normal(size=G), "GRNBoost2": rng.normal(size=G) + .3 * resp,
         "RegVelo |J|": rng.normal(size=G), "Velorama": rng.normal(size=G), M_PAIR: rng.normal(size=G), M_CELL: rng.normal(size=G)}
    for k in GEN_PRIORS:
        V[k] = 0.3 * F[:, 0] + rng.normal(size=G) * .5
    V[GB_PRIOR] = 0.3 * F[:, 1] + rng.normal(size=G) * .5
    for i_, r_ in enumerate(REFS):                                                    # independent stream: does not perturb the tuned synthetic draws above
        V[r_] = np.random.default_rng(1234 + i_).normal(size=G) + (0.4 * occ if i_ == 0 else 0.0)
    for i_, m_ in enumerate(SOTA):                                                    # independent streams (optional SOTA comparators); do not perturb the tuned draws
        V[m_] = np.random.default_rng(4321 + i_).normal(size=G) + 0.2 * resp
    unk = rng.random(G) < 0.1
    tr = mk_truth(~unk, resp, ~(rng.random(G) < 0.05), occ, occ)
    ep = assemble(sysname, tf, genes, V, tr, blk, {})
    ep["F1"], ep["adj_ok"] = F, adj
    return ep


def selftest():
    from sklearn.metrics import roc_auc_score as _auc
    import glob
    rng = np.random.default_rng(7)
    # 1. block draws: same seed -> identical
    b1, b2 = _syn_block(seed=3), _syn_block(seed=3)
    assert np.array_equal(b1.counts, b2.counts) and b1.fingerprint() == b2.fingerprint()
    mods = np.repeat([f"m{i}" for i in range(60)], 10)
    bm = _syn_block(units=mods, seed=3)
    assert (bm.counts[:, 0] == bm.counts[:, 9]).all() and (bm.counts[:, 0] != bm.counts[:, 10]).any()
    # 2. UNKNOWN never negative: untested / unmappable / ambiguous genes are excluded from U and
    # from every class
    Vt = pd.DataFrame(dict(gene=["a", "b", "c", "d"], gencode_matched=[1, 1, 0, 1], KO_tested=[True, False, True, True], KO_resp=["act", "none", "none", "none"],
                           BOUND_PROM=[1.0, 0.0, np.nan, 0.0], BOUND_10KB=[1.0, 1.0, np.nan, 0.0],
                           nearest_summit_tss_dist=[500.0, 3000.0, np.nan, 20000.0], max_signal_10kb=[5.0, 0.0, np.nan, 0.0]))
    t = truth_gata1_frame(Vt, np.array(["a", "b", "c", "d", "zz"]))
    U = t["resp_ok"] & t["occ_ok"]
    assert U.tolist() == [True, False, False, True, False] and t["resp"].tolist() == [True, False, False, False, False]      # c unmatched, b untested, zz absent
    assert not (t["occ"]["prim"] & ~t["occ_ok"]).any()
    assert np.allclose(t["occ_dist"][:4], [500, 3000, np.nan, 20000], equal_nan=True) and np.isnan(t["occ_dist"][4]) and abs(t["occ_score"][0] - np.log1p(5)) < 1e-12
    Tk = pd.DataFrame(dict(resp=[True, False]), index=["X", "Y"])
    Fk = pd.DataFrame(dict(BOUND_PROM=[True, False], BOUND_10KB=[True, True]), index=["X", "Y"])
    tk = truth_klf1_frames(np.array(["X", "x", "Y", "Q"]), Tk, Fk)                                          # X/x collide on an upper-case join -> UNKNOWN
    assert (tk["resp_ok"] == [False, False, True, False]).all() and (tk["occ_ok"] == [False, False, True, False]).all()
    tp = truth_panc_frame(pd.DataFrame(dict(gene=["a", "b", "c"], ko_class=["REP_RESPONDER", "TESTED_NULL", "TESTED_NULL"], mapping=["MAPPED", "MAPPED", "UNKNOWN_MAPPING"],
                                            occupied_1kb=[True, False, False], occupied_10kb=[True, True, False])), ["a", "b", "c", "d"])
    assert (tp["resp_ok"] & tp["occ_ok"]).tolist() == [True, True, False, False]
    # K562 mapping: tie unassigned, blacklist removes, invalid summit removed, MANE beats longer
    # transcript, ambiguous symbol excluded
    tx = pd.DataFrame(dict(gene_id=["g1", "g1", "g2", "g3", "g4", "g5"], sym=["A", "A", "B", "C", "D", "D"], tid=["t1", "t2", "t3", "t4", "t5", "t6"],
                           chrom=["chr1"] * 6, start=[10_000, 9_000, 20_000, 30_000, 50_000, 60_000], end=[10_500, 12_000, 20_500, 30_500, 50_500, 60_500],
                           strand=["+", "+", "+", "+", "+", "+"], mane=[True, False, False, False, False, False]))
    tss, amb = k562_select_tss(tx)
    assert amb == {"D"} and tss.set_index("sym").tss0["A"] == 9_999 and set(tss.sym) == {"A", "B", "C"}                # MANE t1 (start 10000) not longest t2
    pk = pd.DataFrame(dict(chrom=["chr1"] * 5, start=[9_000, 20_500, 29_500, 40_000, 15_000], end=[11_000, 21_500, 30_500, 41_000, 16_000], off=[1_000, 0, 900, -1, 500]))
    bl = pd.DataFrame(dict(chrom=["chr1"], start=[29_000], end=[31_000]))
    pk["sig"] = [10.0, 30.0, 5.0, 1.0, 2.0]
    pr, se, st, kd, ks = k562_assign(pk, tss, bl)
    assert kd["A"] == 1.0 and kd["B"] == 501.0 and abs(ks["B"] - np.log1p(30.0)) < 1e-12 and "C" not in kd
    assert pr == {"A", "B"} and st["n_invalid_summit"] == 1 and st["n_blacklisted"] == 1                             # C's peak blacklisted; off=-1 invalid
    assert se >= pr and nearest_unique([100], [50, 150], 100)[0] == -1 and nearest_unique([100], [50, 151], 100)[0] == 0   # exact tie -> unassigned
    bd_, bs_ = bm_peak_desc({"AA": ("chr1", 1000), "BB": ("chr1", 50_000), "CC": ("chr9", 5)},
                            pd.DataFrame(dict(chrom=["chr1", "chr1"], mid=[1200, 20_000], score=[7.0, 3.0])))
    assert bd_ == {"AA": 200.0, "BB": 30_000.0} and abs(bs_["AA"] - np.log1p(7.0)) < 1e-12 and bs_["BB"] == 0.0
    gl = ['chr1\tH\tgene\t100\t500\t.\t+\t.\tgene_id "G1"; gene_type "protein_coding"; gene_name "Aa";',
          'chr1\tH\tgene\t100\t500\t.\t-\t.\tgene_id "G2"; gene_type "lncRNA"; gene_name "Bb";',
          'chr1\tH\tgene\t900\t950\t.\t+\t.\tgene_id "G3"; gene_type "protein_coding"; gene_name "Dup";',
          'chr2\tH\tgene\t900\t950\t.\t+\t.\tgene_id "G4"; gene_type "protein_coding"; gene_name "DUP";',
          'chr1\tH\tgene\t10\t20\t.\t+\t.\tgene_id "G5"; gene_type "miRNA"; gene_name "Mir";']
    bt, bamb = bm_tss_from_lines(gl)
    assert bt == {"AA": ("chr1", 100), "BB": ("chr1", 500)} and bamb == {"DUP"}
    # 3. primary normalisation before truth restriction: restricted re-z differs, assemble() keeps
    # the vector untouched
    import gb_primary as GP
    raw = {h: rng.standard_t(3, size=(300, 31)) for h in GP.ROOTS}
    full = GP.zcons(raw)
    keep = rng.random(300) < 0.4
    assert not np.allclose(full[keep], GP.zcons({h: raw[h][keep] for h in GP.ROOTS}), equal_nan=True)
    ep0 = _syn_ep(_syn_block(G=300, seed=1), "T", rng=rng)
    try:                                                                                                          # assemble() refuses a GB vector whose pre-truth digest differs
        assemble("GATA1", "T", ep0["genes"], ep0["V"], dict(resp_ok=ep0["U"], resp=ep0["resp"], occ_ok=ep0["U"], occ=ep0["occ"]), ep0["blk"], {}, gb_sha_pre="0" * 64)
        raise AssertionError("changed GB digest accepted")
    except SystemExit:
        pass
    assert F1_COLS == F0_COLS + ["raw_det_u", "mu_det", "mu_abundance", "mu_logvar", "mu_pair_support", "mu_dynamics", "abs_spearman_ms_dpt", "abs_spearman_mu_dpt"] and len(F1_COLS) == 13
    # 4. synchronised paired bootstrap: a comparator that is GB has contrast draws identically 0
    # (raw arm) for every stat
    blk = _syn_block(G=600, B=120, seed=11)
    epA, epB = _syn_ep(blk, "A", rng=np.random.default_rng(2)), _syn_ep(blk, "B", rng=np.random.default_rng(3), signal=1.2)
    assert epA["blk"].counts is epB["blk"].counts
    epA["V"]["GENIE3"] = epA["V"][GB].copy()
    epA["V"]["REF -raw_det"] = epA["V"][GB].copy()
    epA["V"]["COMP_Z"] = epA["V"][GB].copy()                                                                     # a composite is a REF-family member: same masks, same shared draws
    cs, rows, cov = CS(), [], []
    lattice(epA, 120, cs, rows, cov)
    zero = [(m, b) for m, b in zip(cs.meta, cs.bs) if m["scope"] == "PAIRWISE" and m["cmp"] == "GENIE3"]
    assert zero and all(np.allclose(b, 0, atol=1e-12) for _, b in zero), "same score must give identical draws across layers (draws are shared)"
    assert any(m["stat"] == "I_prim" for m, _ in zero)
    # 4b. truth-free REF rankers: evaluated in both scopes with shared draws (a REF identical to GB
    # gives identically-zero contrasts)
    for sc_ in ("PAIRWISE", "ALL6\u2229REF"):
        zr = [(m, b) for m, b in zip(cs.meta, cs.bs) if m["scope"] == sc_ and m["cmp"] == "REF -raw_det"]
        assert zr and all(np.allclose(b, 0, atol=1e-12) for _, b in zr) and {m["stat"] for m, _ in zr} >= {"resp|AUROC", "AD_prim|AUROC", "I_prim"}
        assert any(m["cmp"] == "REF +dynamics" and m["scope"] == sc_ for m in cs.meta)
        zc = [(m, b) for m, b in zip(cs.meta, cs.bs) if m["scope"] == sc_ and m["cmp"] == "COMP_Z"]
        assert zc and all(np.allclose(b, 0, atol=1e-12) for _, b in zc), "COMP_Z identical to GB must give identically-zero contrasts in both REF scopes"
    assert any(r.get("method") == "COMP_PC1" and r.get("status") == "NOT_RUN_MISSING_METHOD" for r in rows), "an absent (NO_CALL) composite is reported missing, never replaced"
    assert nocall_comps(rows) >= {"COMP_PC1", "COMP_RANK", "COMP_FA"} and "COMP_Z" not in nocall_comps(rows) and nocall_comps([]) == set(), "NO_CALL in any endpoint -> excluded from macro"
    assert not any(m["cmp"] == HINDSIGHT or m["scope"] == "HINDSIGHT" for m in cs.meta)
    mh_ = epA["U"] & np.isfinite(epA["V"][GB]) & np.all([np.isfinite(epA["V"][r_]) for r_ in REFS], 0)
    au_ = {r_: _auc(epA["resp"][mh_], epA["V"][r_][mh_]) for r_ in REFS + ["COMP_Z"]}
    hs_ = next(r_ for r_ in rows if r_.get("scope") == "HINDSIGHT" and r_.get("target") == "resp")
    assert hs_["label"] == "HINDSIGHT_UPPER_BOUND" and hs_["descriptive"] and abs(hs_["AUROC"] - max(au_.values())) < 1e-12 and hs_["hindsight_ref"] == max(au_, key=au_.get)
    Fr = pd.DataFrame({c_: np.arange(4.0) for c_, _ in REF_SIGNS.values()})                                       # Detection features use a negative sign, so lower values rank higher.
    rv = ref_vectors(Fr)
    assert (np.diff(rv["REF -raw_det"]) < 0).all() and (np.diff(rv["REF +dynamics"]) > 0).all() and len(rv) == 9 and set(rv) == set(REFS)
    nz = [b for m, b in zip(cs.meta, cs.bs) if m["scope"] == "PAIRWISE" and m["cmp"] == "|Pearson|" and m["stat"] == "resp|AUROC"][0]
    assert np.nanstd(nz) > 0
    assert not any(r.get("method") == "GENIE3" and r.get("scope") == "ALL6" and r.get("status") == "NOT_RUN_MISSING_METHOD" for r in rows)
    # 5. interaction through lattice(): GB carries response signal only, |Pearson| carries occupancy
    # signal -> I_prim(GB vs |Pearson|) == (AUC_GB,resp - AUC_GB,occ)
    from sklearn.metrics import roc_auc_score as _auc
    mk = epA["U"] & np.isfinite(epA["V"][GB]) & np.isfinite(epA["V"]["|Pearson|"])
    fA = lambda m, y: _auc(y[mk], epA["V"][m][mk])
    I_manual = (fA(GB, epA["resp"]) - fA(GB, epA["occ"]["prim"])) - (fA("|Pearson|", epA["resp"]) - fA("|Pearson|", epA["occ"]["prim"]))
    I_lat = [m_ for m_ in zip(cs.meta, cs.pt) if m_[0]["scope"] == "PAIRWISE" and m_[0]["cmp"] == "|Pearson|" and m_[0]["stat"] == "I_prim"][0][1]
    assert I_lat > 0 and abs(I_lat - I_manual) < 1e-12, (I_lat, I_manual)
    dm_ = [r_ for r_ in rows if r_.get("scope") == "PAIRWISE" and r_.get("paired_with") == "|Pearson|" and r_.get("target") == "delta_prim"]
    assert {r_["method"] for r_ in dm_} == {GB, "|Pearson|"} and next(r_ for r_ in dm_ if r_["method"] == GB)["AUROC"] > next(r_ for r_ in dm_ if r_["method"] == "|Pearson|")["AUROC"]
    # 6. matching NO_CALL: separated covariates -> NO_CALL
    Fm = rng.normal(size=(600, 3))
    pos, ctrl = np.arange(0, 100), np.arange(100, 600)
    far = Fm.copy()
    far[pos] += 3.0
    assert match_1to1(far, pos, ctrl)["status"].startswith("NO_CALL")
    ok = match_1to1(Fm, pos, ctrl)
    assert ok["status"] in ("OK", "NO_CALL_SMD") and ok["retention"] > .8
    tight = Fm.copy()
    tight[pos] = tight[ctrl[:100]]                                                                                # perfectly matchable
    assert match_1to1(tight, pos, ctrl)["status"] == "OK"
    # 7. ADJ ladder refit per draw: injected GB signal beyond F1/priors -> M3-M2 > 0
    blk2 = _syn_block(G=450, B=6, seed=5)
    e1 = _syn_ep(blk2, "S", rng=np.random.default_rng(9), signal=2.0)
    cs2, r2, c2 = CS(), [], []
    adjusted(e1, 6, cs2, r2, c2, jobs=1, tset=("resp",))
    inc = {m["cmp"]: p for m, p in zip(cs2.meta, cs2.pt) if m["stat"] == "resp|AUPRC"}
    assert inc["M3-M2"] > 0.02 and abs(inc["M2-M1"]) < 0.05, inc
    e0 = _syn_ep(blk2, "N", rng=np.random.default_rng(9), signal=0.0)
    e0["V"][GB] = np.random.default_rng(4).normal(size=450)
    cs3, r3, c3 = CS(), [], []
    adjusted(e0, 6, cs3, r3, c3, jobs=1, tset=("resp",))
    inc0 = {m["cmp"]: p for m, p in zip(cs3.meta, cs3.pt) if m["stat"] == "resp|AUPRC"}
    assert abs(inc0["M3-M2"]) < 0.05, inc0
    tiny = dict(T=dict(resp=dict(y=np.r_[np.ones(2), np.zeros(41)].astype(bool), X={m_: np.random.default_rng(1).normal(size=(43, 3)) for m_ in MODELS}, fold=np.arange(43) % NFOLD,
                                gi=np.arange(43))), counts=np.ones((2, 43), np.int16))
    pt_tiny = ladder_draw(-1, tiny)                                                                                   # <2 positives in a training fold -> failed fit -> status row, never silent
    assert all(v_ is None for v_ in pt_tiny.values())
    r_st = []
    emit_ladder(e1, "ADJ", "ADJ", "adj_", ("resp",), {"resp": (np.zeros(len(e1["U"]), bool), np.zeros(len(e1["U"]), bool))}, {"resp": False}, {("resp", m_): None for m_ in MODELS},
                {0: {("resp", m_): None for m_ in MODELS}}, [0], MODELS, INCR, CS(), r_st)
    assert len(r_st) == 4 and all(r_["status"] == "FAILED_TOO_FEW" for r_ in r_st)
    d0 = ladder_draw(-1, dict(T=_LAD["T"], counts=blk2.counts))
    d1 = ladder_draw(0, dict(T=_LAD["T"], counts=blk2.counts))
    assert d0["resp", "M3"]["AUROC"] != d1["resp", "M3"]["AUROC"]
    # 8. maxT: simultaneous band is at least as wide as pointwise
    BS = rng.normal(size=(6, 2000)) * 0.1
    q, se, lo, hi = maxT(np.zeros(6), BS)
    assert q > 1.96 and (hi - lo > 2 * 1.96 * se).all()
    p = np.array([.001, .04, .02])
    assert (holm(p) >= p).all() and holm(p)[0] == .003
    assert relation(.60, .56, .64) == ">" and relation(.54, .52, .56) == "unresolved" and relation(.56, .49, .62) == "unresolved"
    assert relation(.50, .46, .54) == "\u2248" and relation(.50, .44, .54) == "unresolved" and relation(np.nan, 0, 1) == "NA"
    assert relation(.45, .40, .49) == "<" and relation(.47, .46, .49) == "\u2248" and relation(.52, .45, .56) == "unresolved"                       # '<' only if not equivalent; equivalence keeps priority
    Rl = lambda **k: dict(dict(AB="unresolved", CD="unresolved", AC="unresolved", BD="unresolved", AD="unresolved"), **k)
    assert held_branches(Rl(AB=">", AC=">", AD=">"), .9) == ["I"]
    assert held_branches(Rl(AB=">", CD=">", AC="\u2248", BD="\u2248"), .55) == ["II"]
    assert held_branches(Rl(CD=">"), .49) == ["III"] and held_branches(Rl(CD=">"), .51) == ["UNRESOLVED"]
    assert held_branches(Rl(AB=">", CD=">", AC="\u2248", BD="\u2248"), .49) == ["II", "III"]                              # overlapping conditions -> both reported
    assert held_branches(Rl(AB=">"), .40) == ["UNRESOLVED"]

    def Wm(vals, sd=.01, Bn=400, sy="PANC", tf="RFX6", bl="PANC", win="prim", off=0):
        rr, ar = [], []
        for t, v in vals.items():
            rr.append(dict(family="WITHIN", system=sy, tf=tf, block=bl, stat=f"{t}_{win}|AUROC", pt=v, len=Bn, row=off + len(ar), shard="s.npz", gate=True))
            ar.append(v + sd * np.random.default_rng(len(ar) + off + 1).normal(size=Bn))
        return rr, np.array(ar, np.float32)
    def branches_of(**kw):
        rr, ar = Wm(kw.pop("vals"), **kw)
        return within_branches(pd.DataFrame(rr), {"s.npz": {"L400": ar}}).query("level=='TF' and window=='prim'").iloc[0]
    assert branches_of(vals=dict(AB=.70, CD=.68, AC=.50, BD=.50, AD=.60)).branches == "II"
    assert branches_of(vals=dict(AB=.70, CD=.50, AC=.65, BD=.50, AD=.70)).branches == "I"
    assert branches_of(vals=dict(AB=.50, CD=.68, AC=.40, BD=.50, AD=.50)).branches == "III"
    bu = branches_of(vals=dict(AB=.60, CD=.60, AC=.50, BD=.50, AD=.60), sd=.10)
    assert bu.branches == "UNRESOLVED" and bu.AC_rel == "unresolved"                                                       # wide interval is not equivalence
    r1, a1 = Wm(dict(AB=.7, CD=.68, AC=.5, BD=.5, AD=.6), tf="T1", sy="GATA1", bl="ERYTHROID")
    r2, a2 = Wm(dict(AB=.7, CD=.68, AC=.5, BD=.5, AD=.6), tf="T2", sy="KLF1", bl="ERYTHROID", off=5)
    wb = within_branches(pd.DataFrame(r1 + r2), {"s.npz": {"L400": np.vstack([a1, a2])}})
    assert set(wb.level) == {"TF", "BLOCK"} and wb.query("level=='BLOCK' and window=='prim'").n_tf.iloc[0] == 2          # GATA1+KLF1 aggregate as one block
    # increments wording: 'not demonstrated' unless an equivalence bound excludes the margin
    def cn(cmp_, tgt, met, lvl, est, lo, hi):
        return dict(family="ADJ", scope="ADJ", comparator=cmp_, stat=f"{tgt}|{met}", level=lvl, est=est, lo_simul=lo, hi_simul=hi)
    Ci = pd.DataFrame([cn("M2-M1", "AB", "AUPRC", "ALL", .02, .005, .04), cn("M3-M2", "AB", "AUPRC", "ALL", 0.0, -.02, .02), cn("M3-M2", "CD", "AUPRC", "ALL", 0.0, -.004, .004)] +
                      [cn("M3-M1", "AB", met, b, .05, lo, .2) for b in ("PANC", "BONEMARROW") for met, lo in (("AUPRC", .02), ("P50", .06))])
    inc = increments(Ci)
    get = lambda t, i: inc[(inc.target == t) & (inc.increment == i)].iloc[0]
    assert get("AB", "M2-M1").statement.startswith("increment demonstrated") and get("AB", "M3-M2").statement == "not demonstrated"
    assert "excluded" in get("CD", "M3-M2").statement and get("AB", "M3-M1").utility_status == "UTILITY_SUPPORTED"
    assert not inc.statement.str.contains("no increment", case=False).any()
    # AD target: A (occupied responder) vs D (unoccupied non-responder) equals the manual auc
    mkd = epA["U"] & np.isfinite(epA["V"][GB]) & ((epA["occ"]["prim"] & epA["resp"]) | (~epA["occ"]["prim"] & ~epA["resp"]))
    ad_row = next(r_ for r_ in rows if r_.get("scope") == "SELF" and r_.get("target") == "AD_prim")
    assert abs(ad_row["AUROC"] - _auc(epA["occ"]["prim"][mkd], epA["V"][GB][mkd])) < 1e-12
    assert {m_["stat"] for m_ in cs.meta if m_["family"] == "WITHIN"} == {f"{t}_{w}|AUROC" for t in WITHIN_T for w in ("prim", "sens")}
    # nested ladder: prefix property asserted
    Xn = nested_models(rng.normal(size=(50, 4)), rng.normal(size=(50, 3)), rng.normal(size=50), rng.normal(size=50))
    assert [Xn[m].shape[1] for m in MODELS] == [4, 7, 8, 9]
    # matching within bound genes (A to B), balance rows, NO_CALL claim text
    epM = _syn_ep(_syn_block(G=700, B=20, seed=21), "M", rng=np.random.default_rng(31))
    epM["F1"] = np.zeros_like(epM["F1"])
    epM["F1"][:, :2] = np.random.default_rng(5).normal(size=(700, 2))                                              # 2-D covariates: matchable
    cs4, r4, m4 = CS(), [], []
    matching(epM, 20, cs4, r4, m4)
    ab = [r_ for r_ in m4 if r_.get("target") == "AB" and "status" in r_][0]
    assert ab["status"] == "OK" and "WITHIN BOUND GENES" in ab["estimand"] and any(r_.get("kind") == "balance" and r_["target"] == "AB" for r_ in m4)
    assert any(r_.get("target") == "match_AB" for r_ in r4) and any(m_["cmp"] == "|Pearson|" and m_["stat"] == "AB|AUROC" for m_ in cs4.meta if m_["family"] == "MATCH")
    epM["F1"] = np.random.default_rng(6).normal(size=(700, len(F1_COLS)))                                          # 13-D covariates: retention gate fails -> NO_CALL, no adjustment claim
    m5 = []
    matching(epM, 20, CS(), [], m5)
    assert [r_ for r_ in m5 if r_.get("target") == "AB" and "status" in r_][0]["claim"] == "NO_CALL: adjustment not claimed"
    # evaluation-only descriptors: fixed distance bins + quantitative occupancy covariate model
    # (declared descriptive)
    epD = _syn_ep(_syn_block(G=600, B=8, seed=41), "D", rng=np.random.default_rng(42))
    dd = np.random.default_rng(43)
    epD["occ_dist"] = dd.choice([300.0, 5000.0, 30000.0, np.nan], 600)
    epD["occ_score"] = dd.random(600) * 3
    csD, rD = CS(), []
    descriptors(epD, 4, csD, rD, 1)
    assert {r_["target"] for r_ in rD if r_.get("method") == GB} >= {"dist_0-1kb", "dist_1-10kb", "dist_>10kb"} and all(r_.get("descriptive") for r_ in rD)
    assert {r_["method"] for r_ in rD if r_.get("scope") == "DESC" and r_.get("target") == "desc_AB"} == {"Q0", "Q1"} and {m_["family"] for m_ in csD.meta} == {"DESC"}
    # 10. end-to-end: two shards of one block -> merge -> macro (GATA1+KLF1 one block),
    # synchronisation guard
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        os.makedirs(f"{d}/mapping")
        for name, ep in (("GATA1", epA), ("KLF1", epB)):
            c_, r_, cv_, mr_ = CS(), [], [], []
            lattice(ep, 120, c_, r_, cv_)
            adjusted(ep, 4, c_, r_, cv_, 1, ("AB", "CD"))
            matching(ep, 120, c_, r_, mr_)
            m_ = c_.save(f"{d}/shard_{name}_all.npz")
            m_["shard"], m_["counts_fp"] = f"shard_{name}_all.npz", ep["blk"].fingerprint()
            m_["script_sha256"], m_["B"], m_["B_adj"] = "s" * 8, 120, 6
            atomic_tsv(m_, f"{d}/shard_{name}_all.index.tsv")
            atomic_json(f"{d}/mapping/{name}_all_mapping_manifest.json", dict(block="ERYTHROID", script_sha256="s" * 8, B=120, B_adj=6, feature_tier="F1", seed=1, module_shas={"gb_primary": "m1"}, adj_targets=["AB", "CD"]))
            for pre, rr in (("endpoints", r_), ("coverage", cv_), ("matching", mr_)):
                atomic_tsv(pd.DataFrame(rr if rr else [dict(x=0)]), f"{d}/part_{pre}_{name}_all.tsv")
        merge(d)
        Cm = pd.read_csv(f"{d}/contrasts.tsv", sep="\t")
        assert Cm[(Cm.family == "RAW") & (Cm.level == "ALL")].n_endpoints.max() == 2
        assert set(Cm[Cm.level != "ALL"].level) <= {"ERYTHROID"} and (Cm.level == "ALL").any()                     # GATA1+KLF1 -> one block "ERYTHROID"
        assert Cm.loc[Cm.inference == "simultaneous_maxT", "hi_simul"].notna().any()
        assert (Cm[(Cm.family == "RAW") & (Cm.level == "ERYTHROID")].inference == "pointwise").all()               # per-block RAW rows are not in the max-|t| family
        assert Cm[(Cm.family == "ADJ") & (Cm.level == "ERYTHROID")].inference.eq("simultaneous_maxT").any()
        assert Cm[(Cm.family == "ADJ") & Cm.stat.str.startswith("AB|") & (Cm.inference == "simultaneous_maxT")].shape[0] > 0                     # adjusted A>B is in the simultaneous family
        Pm = pd.read_csv(f"{d}/per_tf_holm.tsv", sep="\t")
        assert {"AB|AUPRC", "CD|AUPRC"} <= set(Pm[Pm.family == "ADJ"].stat) and {"AC_prim|AUROC", "BD_prim|AUROC", "AD_prim|AUROC"} <= set(Pm[Pm.family == "RAW"].stat)
        assert "holm_p_global" in Pm and pd.read_csv(f"{d}/branches.tsv", sep="\t").branches.notna().all() and len(pd.read_csv(f"{d}/increments.tsv", sep="\t")) > 0
        assert not Cm[(Cm.comparator == "REF +dynamics") & (Cm.scope == "ALL6\u2229REF") & (Cm.level == "ALL") & (Cm.inference == "simultaneous_maxT") & Cm.lo_simul.notna()].empty
        assert (pd.read_csv(f"{d}/endpoints.tsv", sep="\t").query("scope=='HINDSIGHT'").label == "HINDSIGHT_UPPER_BOUND").all()
        assert set(Cm[Cm.comparator.isin(REFS) & (Cm.inference == "simultaneous_maxT")].sim_family) == {"RAW|PAIRWISE|REF", "RAW|ALL6\u2229REF|REF"}
        assert pd.read_csv(f"{d}/coverage.tsv", sep="\t").query("scope=='ALL6\u2229REF' and comparator=='(all 9 REF)'").n_dropped_vs_ALL6.ge(0).all()
        # REFs do not widen the GB-vs-external bands: the RAW|ALL6 / RAW|PAIRWISE (external)
        # families have the same members and the same q with and without the REF rows
        mm = pd.concat([pd.read_csv(f_, sep="\t") for f_ in sorted(glob.glob(f"{d}/shard_*.index.tsv"))], ignore_index=True)
        aa = {os.path.basename(f_): dict(np.load(f_)) for f_ in glob.glob(f"{d}/shard_*.npz")}
        C_full, _ = macro(mm, aa)
        C_noref, _ = macro(mm[~mm.cmp.isin(REFS)], aa)
        for fk in ("RAW|ALL6", "RAW|PAIRWISE"):
            a_, b_ = C_full[C_full.sim_family == fk], C_noref[C_noref.sim_family == fk]
            assert len(a_) == len(b_) > 0 and np.allclose(a_.q_maxT.unique(), b_.q_maxT.unique()) and np.allclose(np.sort(a_.hi_simul), np.sort(b_.hi_simul))
        assert C_full[C_full.sim_family.fillna("").str.endswith("|REF")].q_maxT.nunique() >= 1
        # SOTA comparators: their own pairwise family + the new ALL6+SOTA family
        C_nosota, _ = macro(mm[~mm.cmp.isin(SOTA)], aa)
        for fk in ("RAW|ALL6", "RAW|PAIRWISE"):
            a_, b_ = C_full[C_full.sim_family == fk], C_nosota[C_nosota.sim_family == fk]
            assert len(a_) == len(b_) > 0 and np.allclose(a_.q_maxT.unique(), b_.q_maxT.unique()) and np.allclose(np.sort(a_.hi_simul), np.sort(b_.hi_simul))
        assert set(C_full[C_full.comparator.isin(SOTA) & (C_full.inference == "simultaneous_maxT")].sim_family) == {"RAW|PAIRWISE|SOTA", f"RAW|{SOTA_SCOPE}"}
        assert not C_full[(C_full.scope == SOTA_SCOPE) & (C_full.comparator == "PIDC")].empty
        rc_ = json.load(open(f"{d}/receipt.json"))
        assert rc_["reference_rankers"]["signs"]["REF -raw_det"] == dict(feature="raw_det", sign=-1) and rc_["reference_rankers"]["signs"]["REF +mu_logvar"]["sign"] == 1
        assert rc_["branch_margins"]["branch_I"] and "AB" in rc_["targets"]["raw"] and rc_["multiplicity_families"]["holm_per_TF"]
        assert rc_["inputs_sha256"]["module:gb_primary"] == "m1"
        for k_, v_, tag_ in (("B", 999, "MIXED_B"), ("script_sha256", "zzz", "MIXED_SCRIPT_SHA256")):                # mixed shard provenance is refused
            mp = f"{d}/mapping/KLF1_all_mapping_manifest.json"
            mj = json.load(open(mp))
            atomic_json(mp, dict(mj, **{k_: v_}))
            try:
                merge(d)
                raise AssertionError(f"mixed {k_} accepted")
            except SystemExit as e:
                assert tag_ in str(e), str(e)
            atomic_json(mp, mj)
        import shutil
        for ext in ("index.tsv", "npz"):                                                                            # a repeated (system, tf, family, scope, cmp, stat) key is refused
            shutil.copy(f"{d}/shard_GATA1_all.{ext}", f"{d}/shard_GATA1_dup.{ext}")
        di = pd.read_csv(f"{d}/shard_GATA1_dup.index.tsv", sep="\t")
        di["shard"] = "shard_GATA1_dup.npz"
        atomic_tsv(di, f"{d}/shard_GATA1_dup.index.tsv")
        shutil.copy(f"{d}/mapping/GATA1_all_mapping_manifest.json", f"{d}/mapping/GATA1_dup_mapping_manifest.json")
        try:
            merge(d)
            raise AssertionError("repeated keys accepted")
        except SystemExit as e:
            assert "REPEATED_KEYS" in str(e)
        for ext in ("index.tsv", "npz"):
            os.remove(f"{d}/shard_GATA1_dup.{ext}")
        os.remove(f"{d}/mapping/GATA1_dup_mapping_manifest.json")
        bad = pd.read_csv(f"{d}/shard_KLF1_all.index.tsv", sep="\t")
        bad["counts_fp"] = "deadbeef"
        atomic_tsv(bad, f"{d}/shard_KLF1_all.index.tsv")
        try:
            merge(d)
            raise AssertionError("desynchronised draws must be refused")
        except SystemExit as e:
            assert "DESYNCHRONISED" in str(e)
    print("gb_layer_lattice selftest OK")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--systems", default="GATA1,KLF1,PANC,BONEMARROW,K562")
    ap.add_argument("--smalln", action="store_true", help="GATA1+KLF1 only, B=200; the raw arm always runs, ADJ only if catbank artifacts exist")
    ap.add_argument("--B", type=int, default=None)
    ap.add_argument("--B-adj", type=int, default=None, help="refit draws for the ADJ arm (default = --B)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--tfs", default="")
    ap.add_argument("--stage", choices=["all", "shard", "merge"], default="all")
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-adjusted", action="store_true")
    ap.add_argument("--features", choices=["F1", "F1x"], default="F1", help="ADJ feature tier: F1 or extended F1x")
    ap.add_argument("--adj-targets", default="AB,CD", help="ladder targets, run separately: AB (A vs B within bound genes), CD (C vs D), resp, occ (each multiplies the refit cost)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        return
    FEATURE_TIER[0] = a.features
    systems = ["GATA1", "KLF1"] if a.smalln else [s.strip().upper() for s in a.systems.split(",") if s.strip()]
    B = B_PILOT if a.smalln else (a.B or B_FINAL)
    out = a.out or (OUT_DIR + "_smalln" if a.smalln else OUT_DIR)
    os.makedirs(out, exist_ok=True)
    if a.stage in ("all", "shard"):
        for s in systems:
            run_shard(s, out, B, a.B_adj or B, a.jobs, [t.strip().upper() for t in a.tfs.split(",") if t.strip()], a.smalln, not a.no_adjusted, tuple(a.adj_targets.split(",")))
    if a.stage in ("all", "merge"):
        merge(out)


if __name__ == "__main__":
    main()
