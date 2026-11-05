#!/usr/bin/env python
"""LAYERED EVALUATION of the frozen GeoBRIDGE score on every scoreable real substrate.

usage: gb_layers.py <out_dir> --system SYS [--kmax 30] [--panel_source auto|perturb|chip]
       ALWAYS launched through gb_launch.py, which snapshots and hashes the executed source.

PI master directive v3: a failed inferential LAYER must not condemn a DATASET. Report which question
each system can validly answer, and label every answer honestly.

    L1 raw target prioritization   AP/nAP, P@10/20/50/100, lift@K, recall@K, cumulative gain
    L2 incremental dynamic value   vs a STATIC source-target association and a TARGET-ONLY baseline
    L3 actual-TF attribution       cognate r vs ALL truth-independently eligible real TFs, full matrix
    L4 matched non-TF calibration  NOT HERE -- needs gb_m1v2_roster, runs separately
    L5 reference-network recovery  ChIP/BEELINE panels, only with a PRE-DECLARED context grade
    L6 perturbation validation     supplies the truth panels used by L1-L3

ROSTER INDEPENDENCE, confirmed by internal review 2026-09-21: num = Xs.T @ Y, den = |Xs|.T @ |Y|, E = |num/den|.
No matched non-TF roster enters. L1/L2/L3/L5 are therefore computable without M1-v2. That independence
does NOT certify labels, masks, competitor population or multiplicity, which are handled below.

--------------------------------------------------------------------------------------------------
DEFECTS FIXED HERE, each one found by internal review or internal review on the previous revision:

 1. COMPETITOR CAP REMOVED. An alphabetical prefix changed the estimand to "rank among an alphabetical
    prefix" and made the answer depend on catalogue naming. Every ELIGIBLE TF is now scored.
    Eligibility is frozen BEFORE scoring from truth-independent observational criteria only, and each
    rejected TF is recorded as INELIGIBLE_SOURCE with its reason rather than silently dropped
    (directive v3 section 4 -- "all competitors or refuse" is too strong, because an unexpressed TF
    has no observational source signal at all and would make attribution impossible in every real
    dataset). NO_CALL_INCOMPLETE_COMPETITION fires only when an ELIGIBLE TF fails during scoring.
    n_candidate / n_eligible / n_scored are always reported.
 2. COMPARISON-SPECIFIC MASKS (PI v4 section 2A). A single intersection over every source, both
    channels and all depths let an unrelated competitor failure at an exploratory depth delete a
    target from the focal k=1 comparison. Each row now carries the scope its own question needs, with
    a hash: `RECOVERY` (panel universe INTERSECT focal support at that channel/depth), `UV` (focal
    support in U INTERSECT V at that depth -- the ONLY scope in which the channels may be compared),
    `NATIVE` (per-channel sensitivity), and for attribution a per-(channel, depth) universe shared by
    the eligible competitors.
 3. FULL ATTRIBUTION MATRIX SERIALIZED. Only summaries were written before, so competitor scores could
    not be reconstructed. Every (panel, source, root, channel, depth) cell is now emitted.
 4. UNSUPPORTED IS MISSING, NOT ZERO. den was clipped by a floor, so an unsupported edge scored 0 and
    entered rankings as a confident low value. Unsupported cells are NaN everywhere, including in the
    per-gene dump.
 5. top1/top3 CANNOT BE EMITTED ON AN INCOMPLETE COMPETITION. Previously an eligible competitor that
    failed was dropped from the rank denominator, so top1=True could be reported while competitors
    were missing -- which directly changes which TFs look cognate-attributed.
 6. ATTRIBUTION AT EVERY DEPTH, not only at dump depths.
 7. STATIC and TARGET-ONLY comparators implemented (they were promised in the docstring and absent).
 8. K562 TRUTH-FACING IS REFUSED. Its producer stores significant pairs only, so "not listed" conflates
    tested-null with untested. A descriptive caveat does not cure a wrong negative universe, and the
    caveat was not even serialized. K562 emits truth-free score tensors only until the four-state
    table exists.
 9. BM follows PI ruling A exactly, plus the predeclared STRICT-detection sensitivity, plus an explicit
    status row for EVERY preregistered TF.
10. PROVENANCE carries the gene-universe hash, catalogue hash, truth-input hashes, per-root source
    roster hash and the common-mask hash.
11. P@K reports the tie size at the cutoff, because low-support |S| produces exact maxima and a
    cutoff tie silently decided by gene order is not reproducible science.

U-vs-V SEMANTICS. Delta = T_U - T_V on the UV universe is reported DESCRIPTIVELY with its paired
label-permutation reference. That permutation tests a CONDITIONAL LABEL-ASSOCIATION null on fixed
score vectors; it is NOT a test of equal predictive performance and NOT a biological-replicate test.
Per directive v3 section 6 the channel decision rests on cross-regulator and cross-system consistency
of per-TF Delta_r, not on comparing two separate p-values. U remains primary.

ATTRIBUTION SEMANTICS. cognate rank and rank percentile are ATTRIBUTION METRICS, not biological
p-values (directive v3 section 20).
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
for _p in ((__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"), (__import__("os").environ["GEOBRIDGE_ROOT"] + "/pairspace")):
    if _p not in sys.path:
        sys.path.append(_p)
import anndata as ad
from gb_substrate import load
from gb_uvnull import uv_perm_null

R = __import__("os").environ["GEOBRIDGE_ROOT"]
EPS = 1e-12
NS = (10, 20, 50, 100)
N_PERM = 500
# =====================================================================================
# NO DEPTH IS PRIVILEGED IN THIS RUN.
# PI, repeatedly and most recently 2026-09-21: run the FULL k sweep FIRST, then decide the depth rule
# from that sweep. Do not carry k_star or k=10 into the computation as special objects.
# An earlier revision of this file violated that: it used {0, k_star, k_hist} to decide where the
# expensive competitor competition ran and where the matrix was serialized, and it REFUSED TO START
# without a k_star. Both are removed. k_star and k=10 are written into provenance only so older
# numbers can be cross-referenced; they gate nothing and select nothing.
# Every depth 0..kmax is scored, and RECOVERY plus the U-V contrast are emitted at every depth for
# every panel. MAX-OVER-k IS NEVER A CONFIRMATORY STATISTIC unless the null gets the same freedom.
# =====================================================================================
K_HIST = 10                      # historical continuity ONLY
TFCAT = {"GATA1": f"{R}/data/LaraAstiaso_raw/ext_ref/external_regulators.txt",
         "PANC": f"{R}/data/LaraAstiaso_raw/ext_ref/external_regulators.txt",
         "DENTATEGYRUS": f"{R}/data/LaraAstiaso_raw/ext_ref/external_regulators.txt",
         "GASTRULATION": f"{R}/data/LaraAstiaso_raw/ext_ref/external_regulators.txt",
         "K562": f"{R}/GB_real_v2/lambert2018_human_tfs.txt",
         "BONEMARROW": f"{R}/GB_real_v2/lambert2018_human_tfs.txt"}

# PRE-DECLARED reference networks. Frozen here BEFORE any scoring; a network is never chosen after
# performance is seen. Grade is the CONTEXT match of the network to the substrate, not its quality.
# Populated from the reviewer's pre-declaration; an entry absent from this table cannot be scored.
BEELINE = f"{R}/data/beeline_networks/Networks"
CHIP_DECL = {
    # (system, network): (role, context_grade, rank)   rank 1 = PRIMARY, 2 = prespecified secondary
    # Frozen from the reviewer's pre-declaration 2026-09-21, BEFORE any scoring. Cross-species
    # pairings are REJECT (species-incompatible; no implicit orthology conversion).
    ("DENTATEGYRUS", "mouse/Non-Specific-ChIP-seq-network.csv"): ("USE", "LOW", 1),
    ("GASTRULATION", "mouse/mESC-ChIP-seq-network.csv"):         ("USE", "MEDIUM", 1),
    ("GASTRULATION", "mouse/Non-Specific-ChIP-seq-network.csv"): ("USE", "LOW", 2),
    ("PANC", "mouse/Non-Specific-ChIP-seq-network.csv"):         ("USE", "LOW", 1),
    ("GATA1", "mouse/mHSC-ChIP-seq-network.csv"):                ("USE", "MEDIUM", 1),
    ("GATA1", "mouse/Non-Specific-ChIP-seq-network.csv"):        ("USE", "LOW", 2),
    ("BONEMARROW", "human/Non-specific-ChIP-seq-network.csv"):   ("USE", "LOW", 1),
}
# Frozen reference semantics, also from the pre-declaration:
#  * BEELINE "Non-Specific" is NOT pure occupancy -- it mixes DoRothEA, RegNetwork and TRRUST. The
#    licensed phrase is REFERENCE-NETWORK RECOVERY, never "binding truth" and never "regulation".
#  * mESC-lofgof is a PERTURBATION-RESPONSE reference and STRING is FUNCTIONAL ASSOCIATION, not
#    directed transcriptional truth; neither is declared USE for any substrate here.
#  * Released edge lists are used as published: no edge-score threshold is tuned. Directed pairs are
#    deduplicated and the identifier mapping is recorded.
#  * A reference panel needs >= 10 positives AND >= 10 background genes AFTER masking. That filters
#    evaluable REFERENCE PANELS; it never filters observational source eligibility.
#  * Non-edges are UNANNOTATED BACKGROUND, not demonstrated non-binding. Coverage and prevalence are
#    always reported, and overlapping networks are never averaged as independent replications.
CHIP_MIN_POS = 10
CHIP_MIN_NEG = 10


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def sha_mask(m):
    return hashlib.sha256(np.asarray(m, bool).tobytes()).hexdigest()


# ------------------------------------------------------------------------------------ metrics
def ap_prev(y, s):
    """Tie-grouped AP and prevalence. NaN scores are already excluded by the caller's mask."""
    n = len(y); prev = y.mean()
    if prev <= 0 or prev >= 1:
        return np.nan, prev
    o = np.argsort(-s, kind="mergesort"); y, s = y[o], s[o]
    idx = np.flatnonzero(np.r_[True, s[1:] != s[:-1]]); ends = np.r_[idx[1:], n]
    tp = 0.0; ap = 0.0
    for a, b in zip(idx, ends):
        h = y[a:b].sum()
        if h:
            ap += h * ((tp + h) / b)
        tp += h
    return ap / y.sum(), prev


def nap(y, s):
    a, p = ap_prev(y, s)
    return a / p if a == a and p > 0 else np.nan


def nap_rows(X, y):
    """Tie-grouped nAP for EVERY ROW of X at once. Mathematically identical to nap() applied per row.

    WHY. The attribution competition needs one nAP per eligible source, at every depth, for every
    panel. Scalar looping costs n_panels x n_sources x n_depths x 2 calls: 547 x 2366 x 31 x 2 = 8e7
    for GATA1, which is roughly eleven hours of pure Python. Vectorising turns it into a handful of
    array ops.

    The tie-collapsed AP is  sum_g h_g * T_g / B_g  over groups g of equal score, where h_g is the
    number of positives in the group, T_g the cumulative positives through the end of the group and
    B_g the 1-based end position. Groups are found per row; `h_g` comes from the cumulative count at
    the previous group end, located with a running maximum over end positions.
    """
    return nap_apply(nap_prepare(X), y)


def nap_prepare(X):
    """The ORDER-DEPENDENT half of nap_rows, which is everything expensive and which depends ONLY on
    the score matrix -- never on the labels.

    Measured 2026-09-21 on a K562 small-N run: across 120 evaluable attribution cells there were
    exactly TWO distinct comparison masks, so the identical argsort of a 1265 x 2700 matrix was
    being recomputed sixty times per (channel, depth) with only the label vector changing. At the
    measured throughput the six-system benchmark came to 161 host-hours, essentially all of it in
    that repeated sort. Splitting the sort from the label application lets the caller cache it per
    (channel, depth, mask, roster) and reuse it for every panel sharing that mask. The arithmetic is
    unchanged -- `nap_rows` is now literally `nap_apply(nap_prepare(X), y)`."""
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[1]
    o = np.argsort(-X, axis=1, kind="stable")
    Ssort = np.take_along_axis(X, o, axis=1)
    newgrp = np.ones(X.shape, bool); newgrp[:, 1:] = Ssort[:, 1:] != Ssort[:, :-1]
    end = np.zeros(X.shape, bool); end[:, :-1] = newgrp[:, 1:]; end[:, -1] = True
    end_shift = np.zeros(X.shape, bool); end_shift[:, 1:] = end[:, :-1]
    pos_i = np.arange(n)[None, :]
    # end_shift is True at position p+1 when p was a group END, so the value recorded there must be
    # p -- the index OF THE PREVIOUS END -- not p+1. Recording pos_i instead of pos_i-1 made T_prev
    # read the cumulative count one element too late and silently deflated every AP.
    last_prev = np.maximum.accumulate(np.where(end_shift, pos_i - 1, -1), axis=1)
    return dict(o=o, end=end, last_prev=last_prev, n=n, shape=X.shape,
                finite=np.isfinite(X).all(1),
                idx=np.arange(1, n + 1, dtype=np.float64)[None, :])


def nap_apply(prep, y):
    """The LABEL-DEPENDENT half: one gather, one cumsum, one reduction. O(rows x n) with no sort."""
    n = prep["n"]; n_pos = int(y.sum())
    if n_pos == 0 or n_pos == n:
        return np.full(prep["shape"][0], np.nan)
    cum = np.cumsum(y[prep["o"]].astype(np.float64), axis=1)
    end, last_prev, idx = prep["end"], prep["last_prev"], prep["idx"]
    T_prev = np.where(last_prev >= 0,
                      np.take_along_axis(cum, np.maximum(last_prev, 0), axis=1), 0.0)
    h = np.where(end, cum - T_prev, 0.0)
    ap = (h * np.where(end, cum / idx, 0.0)).sum(1) / n_pos
    out = ap / (n_pos / n)
    # a row that is entirely NaN (an unsupported source) must stay NaN, not become a number
    out[~prep["finite"]] = np.nan
    return out


def cl_auc(y, s):
    n1 = int(y.sum()); n0 = len(y) - n1
    if n1 < 5 or n0 < 5:
        return np.nan
    r = pd.Series(s).rank(method="average").to_numpy()
    return float((r[y.astype(bool)].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def topk(y, s, prev):
    o = np.argsort(-s, kind="mergesort"); yy = y[o]; ss = s[o]
    out = {}; tot = max(int(y.sum()), 1)
    for N in NS:
        if len(yy) < N:
            continue
        out[f"P@{N}"] = float(yy[:N].mean())
        out[f"lift@{N}"] = out[f"P@{N}"] / prev if prev > 0 else np.nan
        out[f"recall@{N}"] = float(yy[:N].sum()) / tot
        # FRACTIONAL EXPECTED P@K. Reporting the tie size was not enough: a cutoff tie broken by gene
        # order leaves P@K dependent on the input ordering. Here every gene strictly above the cutoff
        # score is counted, and the tied block contributes its hit rate times the fraction of the
        # block that fits -- the expectation under uniform random tie-breaking, which is
        # order-independent.
        cut = ss[N - 1]
        above = ss > cut
        tied = ss == cut
        n_above = int(above.sum()); n_tied = int(tied.sum())
        slots = N - n_above
        hit_tied = float(yy[tied].sum())
        out[f"P@{N}_exp"] = float(yy[above].sum() + hit_tied * slots / n_tied) / N
        out[f"lift@{N}_exp"] = out[f"P@{N}_exp"] / prev if prev > 0 else np.nan
        out[f"tie@{N}"] = n_tied
        # `n_above < N` is TAUTOLOGICAL: the Kth score always has fewer than K strictly above it
        # (internal review 2026-09-21 finding 1). The real question is whether the tied block STRADDLES K.
        out[f"tie_crosses@{N}"] = bool(n_above < N < n_above + n_tied)
    for q in (0.01, 0.05, 0.10):
        m = max(int(round(q * len(yy))), 1)
        out[f"gain@{int(q*100)}pct"] = float(yy[:m].sum()) / tot
        # FRACTIONAL EXPECTED gain, for the same reason as P@K_exp: the raw form breaks a cutoff tie
        # by gene order and it feeds the U-V channel contrast (internal review finding 1).
        cq = ss[m - 1]; aq = ss > cq; tq = ss == cq
        out[f"gain@{int(q*100)}pct_exp"] = float(yy[aq].sum() +
                                                 yy[tq].sum() * (m - int(aq.sum())) / int(tq.sum())) / tot
    return out


def selftest():
    y = np.r_[np.ones(20), np.zeros(80)].astype(int)
    good = np.r_[np.linspace(1, .5, 20), np.linspace(.4, 0, 80)]
    assert abs(nap(y, good) - 5.0) < 1e-9
    assert abs(cl_auc(y, good) - 1.0) < 1e-12
    assert abs(nap(y, np.ones(100)) - 1.0) < 1e-9, "all ties must be chance, not above"
    assert abs(cl_auc(y, np.ones(100)) - 0.5) < 1e-12
    t = topk(y, good, .2)
    assert t["P@10"] == 1.0 and abs(t["lift@10"] - 5) < 1e-12 and t["tie@10"] == 1
    assert topk(y, np.ones(100), .2)["tie@20"] == 100, "a fully tied cutoff must be reported as such"
    # ap must be invariant to input order
    rng = np.random.default_rng(0); p = rng.permutation(100)
    assert abs(nap(y[p], good[p]) - nap(y, good)) < 1e-12
    # VECTORISED nAP must equal the scalar one row by row, including with heavy ties and a fully
    # unsupported (all-NaN) row, which must stay NaN rather than becoming a number.
    M = rng.normal(size=(40, 100))
    M[:, ::7] = 0.25                      # deliberate tie blocks
    M[3] = np.nan
    v = nap_rows(M, y)
    for i in (0, 1, 2, 5, 17, 39):
        assert abs(v[i] - nap(y, M[i])) < 1e-9, (i, v[i], nap(y, M[i]))
    assert np.isnan(v[3]), "an all-NaN source row must stay NaN"
    assert abs(nap_rows(np.ones((2, 100)), y)[0] - 1.0) < 1e-9, "all ties must be chance"
    print("selftest OK (scalar and vectorised nAP agree, ties and NaN rows included)", flush=True)


# Every truth / reference-network file actually READ is recorded here with its hash and written
# into provenance. The snapshot gate binds the CODE; without this the SCIENTIFIC INPUT was free to
# change underneath a fixed SHA, so two runs with identical provenance could disagree (internal review
# finding 4, 2026-09-21).
TRUTH_ARTIFACTS = {}


def bind(label, path):
    """Record a scientific input's hash and return the path. Every truth / network file that
    reaches a panel must pass through here, or the snapshot SHA certifies only the CODE while the
    DATA underneath it stays free to change (internal review blocker 1, 2026-09-21)."""
    if os.path.exists(path):
        TRUTH_ARTIFACTS[label] = dict(path=path, sha256=sha_file(path))
    else:
        TRUTH_ARTIFACTS[label] = dict(path=path, sha256="ABSENT")
    return path


# ------------------------------------------------------------------------------------ truth
def panels(system, genes, truth_sha):
    gu = np.char.upper(np.asarray(genes).astype(str))
    gp = {g: i for i, g in enumerate(gu)}
    out = {}; status = []
    if system == "GATA1":
        import gb_gata1_fourclass_canon as _g4
        from gb_gata1_fourclass_canon import four_classes
        for _lbl, _fn in (("gata1_chipseq_labels", "chipseq_labels.tsv"),
                          ("gata1_control", "gata1_control_deterministic.h5ad"),
                          ("gata1_ko_projected", "gata1_ko_projected_deterministic.h5ad")):
            bind(_lbl, f"{_g4.G}/{_fn}")
        cls, counts = four_classes(list(genes))
        for ax in ("any_change", "activation", "repression"):
            C = cls[ax]
            for nm, pos, uni in (("A_vs_B", C["A"], C["A"] | C["B"]),
                                 ("A_vs_rest", C["A"], C["A"] | C["B"] | C["C"] | C["D"])):
                key = f"GATA1|{ax}|{nm}"
                if pos.sum() >= 5 and uni.sum() >= 20:
                    out[key] = (pos, uni, dict(axis=ax, contrast=nm, modality="ChIPxKO", tier="A"))
                    status.append(dict(panel=key, tf="GATA1", status="EVALUABLE",
                                       n_pos=int(pos.sum()), n_uni=int(uni.sum())))
                else:
                    status.append(dict(panel=key, tf="GATA1", status="NO_CALL_LOW_POSITIVES",
                                       n_pos=int(pos.sum()), n_uni=int(uni.sum())))
    elif system == "PANC":
        import gb_dev_evaluate as E
        # The two files gb_dev_evaluate.truth_panc reads (verified at gb_dev_evaluate.py:74-75).
        # Bound here rather than left opaque: an unhashed orthology table silently changes which
        # mouse gene each human DE row lands on.
        bind("panc_de", f"{E.R}/real_1_Pancreas/external_validation/aldous_rfx6/results/"
                        f"aldous_PPs_WT_vs_KO_DE.tsv")
        bind("panc_orthology", f"{E.R}/data/orthology/HOM_MouseHumanSequence.rpt")
        y, _ = E.truth_panc(list(genes)); y = np.asarray(y)
        testable = y != -1
        key = "RFX6|any_change|resp_vs_null"
        out[key] = ((y == 1) & testable, testable,
                    dict(axis="any_change", contrast="resp_vs_null", modality="perturbation", tier="A"))
        status.append(dict(panel=key, tf="RFX6", status="EVALUABLE",
                           n_pos=int(((y == 1) & testable).sum()), n_uni=int(testable.sum())))
    elif system == "BONEMARROW":
        P = pd.read_csv(bind("bm_stratified_panel",
                             f"{R}/results/gse274113_truth_2026_09_19/gse274113_stratified_panel.tsv"),
                        sep="\t")
        P = P[P.panel.astype(str) == "stratified"]
        nt = pd.read_csv(bind("bm_gene_stats_nt",
                              f"{R}/results/gse274113_truth_2026_09_19/gene_stats_nt.tsv"), sep="\t")
        sym = nt["gene"].astype(str).str.upper()
        # Colliding symbols are DROPPED ENTIRELY, never resolved by keeping one arbitrarily: a
        # dict(zip(...)) would silently retain whichever row came last. A dropped symbol simply has
        # no NT detection value, so it cannot enter the testable set -- the conservative direction.
        dupmask = sym.duplicated(keep=False)
        n_dup = int(dupmask.sum())
        if n_dup:
            print(f"  NT table: dropped {n_dup} rows in "
                  f"{int(sym[dupmask].nunique())} uppercase-colliding symbol group(s)", flush=True)
        good = nt.loc[~dupmask]
        det = dict(zip(good["gene"].astype(str).str.upper(), good["nt_detect_frac"].astype(float)))
        dv = np.array([det.get(g, np.nan) for g in gu])
        base = np.isfinite(dv) & (dv >= 0.05)
        # EVERY preregistered TF gets an explicit status row, including the ones absent from the
        # significant-only export -- they must not simply disappear.
        PREREG = ["GATA1", "GATA2", "RUNX1", "TAL1", "KLF1", "SPI1", "MYB", "BCL11A", "NFE2",
                  "IRF1", "IRF9", "SOX12", "PRDM16", "FOSL1", "LMO2", "GFI1B", "LDB1", "ATF4", "SNAI2"]
        have = {t.upper() for t in P.tf.astype(str).str.upper().unique()}
        for tf in PREREG:
            key_p = f"{tf}|any_change|resp_vs_nttestable"
            if tf not in gp:
                status.append(dict(panel=key_p, tf=tf, status="NO_CALL_ABSENT_FROM_UNIVERSE")); continue
            if tf not in have:
                status.append(dict(panel=key_p, tf=tf, status="NO_CALL_ABSENT_FROM_PANEL")); continue
            sub = P[P.tf.str.upper() == tf]
            pos = np.zeros(len(genes), bool)
            pos[[gp[t] for t in sub.target.astype(str).str.upper() if t in gp]] = True
            pos[gp[tf]] = False
            # PI RULING A (primary): positives are NOT filtered by NTC detection; negatives must be
            # NTC-testable; universe = positives UNION testable.
            uni = (base | pos); uni[gp[tf]] = False
            # PREDECLARED STRICT sensitivity: NTC>=5% applied symmetrically to BOTH classes.
            pos_s = pos & base; uni_s = base.copy(); uni_s[gp[tf]] = False
            for tag, pp, uu in (("primary", pos & uni, uni), ("strict", pos_s & uni_s, uni_s)):
                key = f"{tf}|any_change|resp_vs_nttestable_{tag}"
                if pp.sum() >= 10 and uu.sum() >= 20:
                    out[key] = (pp, uu, dict(axis="any_change", contrast=f"resp_vs_nttestable_{tag}",
                                             modality="perturbation_crossstudy", tier="A",
                                             arm=tag, n_nt_testable=int(base.sum()),
                                             caveat="CROSS-STUDY: observational substrate and truth "
                                                    "come from different studies; a negative cannot "
                                                    "separate method failure from cross-study mismatch"))
                    status.append(dict(panel=key, tf=tf, status="EVALUABLE", arm=tag,
                                       n_pos=int(pp.sum()), n_uni=int(uu.sum())))
                else:
                    status.append(dict(panel=key, tf=tf, status="NO_CALL_LOW_POSITIVES", arm=tag,
                                       n_pos=int(pp.sum()), n_uni=int(uu.sum())))
    elif system == "K562":
        # FIVE-STATE TRUTH. Positives = the prespecified tested significant states; negatives =
        # TESTED_NULL only in the primary arm; UNKNOWN is EXCLUDED, never counted as a negative.
        # The v2 artifact carries only FOUR codes, so under it the primary and sensitivity negative
        # sets are IDENTICAL and the sensitivity arm silently becomes a duplicate of the primary
        # rather than a wider one (internal review finding 1, 2026-09-21). v3 is therefore REQUIRED, not
        # preferred, and the code_map is validated rather than assumed.
        sp = f"{R}/results/k562_truth_v3_2026_09_21/k562_state.npz"
        if not os.path.exists(sp):
            raise SystemExit(
                "REFUSING[K562_TRUTH_INCOMPLETE]: k562_state.npz is absent. The legacy producer "
                "stored SIGNIFICANT PAIRS ONLY, so absence conflated tested-null with never-tested "
                "and a non-significant gene could not be used as a negative. Rebuild with the "
                "repaired gb_k562_truth.py, then rerun.")
        Zst = np.load(sp, allow_pickle=True)
        if "code_map" not in Zst:
            raise SystemExit("REFUSING[K562_TRUTH_NO_CODE_MAP]: cannot verify which integer means "
                             "what, so the negative sets cannot be trusted")
        cmap = json.loads(str(Zst["code_map"])) if isinstance(Zst["code_map"].item(), str) \
            else dict(Zst["code_map"].item())
        REQUIRED_CODES = {"UNKNOWN": 0, "TESTED_NULL": 1, "TESTED_UP": 2, "TESTED_DOWN": 3,
                          "LOW_INFORMATION_TESTED_NULL": 4}
        if cmap != REQUIRED_CODES:
            raise SystemExit(f"REFUSING[K562_TRUTH_CODE_MAP_MISMATCH]: {cmap} != {REQUIRED_CODES}. "
                             "The four-code v2 artifact makes the sensitivity arm a duplicate of "
                             "the primary arm instead of a wider one.")
        bind("k562_state", sp); TRUTH_ARTIFACTS["k562_state"]["code_map"] = cmap
        sgen = np.char.upper(Zst["genes"].astype(str)); stfs = Zst["tfs"].astype(str)
        st = Zst["state"]
        if len(set(sgen.tolist())) != sgen.size:
            raise SystemExit("REFUSING[K562_STATE_SYMBOL_COLLISION]")
        col = {g: i for i, g in enumerate(sgen)}
        have = np.array([g in col for g in gu])
        cidx = np.array([col.get(g, 0) for g in gu])
        for i, tf in enumerate(stfs):
            T = tf.upper()
            if T not in gp:
                status.append(dict(panel=f"{T}|any_change|tested", tf=T,
                                   status="NO_CALL_ABSENT_FROM_UNIVERSE")); continue
            row = st[i][cidx]
            pos0 = have & ((row == 2) | (row == 3))     # tested-up OR tested-down
            # PRIMARY negatives are WELL-OBSERVED tested-nulls (state 1). The SENSITIVITY arm adds
            # LOW_INFORMATION_TESTED_NULL (state 4) -- formally testable but below the control
            # detection floor. UNKNOWN (state 0) is never a negative in either arm.
            for arm, negstates in (("primary", (1,)), ("sensitivity", (1, 4))):
                pos = pos0.copy()
                neg = have & np.isin(row, negstates)
                uni = pos | neg
                pos[gp[T]] = False; uni[gp[T]] = False
                key = f"{T}|any_change|tested_sig_vs_tested_null_{arm}"
                if pos.sum() >= 10 and (uni & ~pos).sum() >= 10:
                    out[key] = (pos, uni, dict(axis="any_change",
                                               contrast=f"tested_sig_vs_tested_null_{arm}",
                                               modality="perturbation_CRISPRi", tier="A", arm=arm,
                                               n_unknown_excluded=int(have.sum() - uni.sum()),
                                               caveat="chronic CRISPRi RESPONSE: downstream and "
                                                      "indirect effects included, NOT direct "
                                                      "binding. tested-null means 'not significant "
                                                      "at the frozen alpha', NOT 'biologically "
                                                      "unaffected'."))
                    status.append(dict(panel=key, tf=T, status="EVALUABLE", arm=arm,
                                       n_pos=int(pos.sum()), n_uni=int(uni.sum()),
                                       n_unknown_excluded=int(have.sum() - uni.sum())))
                else:
                    status.append(dict(panel=key, tf=T, status="NO_CALL_LOW_POSITIVES", arm=arm,
                                       n_pos=int(pos.sum()), n_uni=int(uni.sum())))
    else:
        raise SystemExit(f"REFUSING[UNKNOWN_SYSTEM]: {system}")
    return out, pd.DataFrame(status)


def chip_panels(system, genes):
    """REFERENCE-NETWORK panels from the PRE-DECLARED networks only. A (system, network) pair absent
    from CHIP_DECL cannot be scored, which is what stops a network being chosen after performance."""
    decl = {k[1]: v for k, v in CHIP_DECL.items() if k[0] == system}
    if not decl:
        return {}, pd.DataFrame([dict(panel="-", tf="-", status="NO_DECLARED_REFERENCE_NETWORK")])
    gu = np.char.upper(np.asarray(genes).astype(str))
    inuni = set(gu.tolist())
    out = {}; status = []
    for net, (role, grade, rank) in sorted(decl.items(), key=lambda x: x[1][2]):
        path = f"{BEELINE}/{net}"
        if not os.path.exists(path):
            status.append(dict(panel=net, tf="-", status="NO_CALL_NETWORK_ABSENT")); continue
        bind(f"network:{net}", path)
        d = pd.read_csv(path, header=None, dtype=str, usecols=[0, 1]).dropna()
        d.columns = ["tf", "tg"]
        d = d[d.tf.str.upper() != "GENE1"]                      # drop a header row if present
        d = d.assign(tf=d.tf.str.upper(), tg=d.tg.str.upper()).drop_duplicates()
        d = d[(d.tf != d.tg) & d.tf.isin(inuni) & d.tg.isin(inuni)]
        gp = {g: i for i, g in enumerate(gu)}
        for tf, sub in d.groupby("tf"):
            pos = np.zeros(len(genes), bool)
            pos[[gp[t] for t in sub.tg]] = True
            pos[gp[tf]] = False
            uni = np.ones(len(genes), bool); uni[gp[tf]] = False
            key = f"{tf}|reference|{os.path.basename(net)}"
            if pos.sum() >= CHIP_MIN_POS and (uni & ~pos).sum() >= CHIP_MIN_NEG:
                out[key] = (pos, uni, dict(axis="reference", contrast=os.path.basename(net),
                                           modality="reference_network", tier="C",
                                           network=net, context_grade=grade, network_rank=rank,
                                           caveat="REFERENCE-NETWORK RECOVERY. Non-edges are "
                                                  "UNANNOTATED BACKGROUND, not demonstrated "
                                                  "non-binding. This is NOT functional regulation."))
                status.append(dict(panel=key, tf=tf, status="EVALUABLE", network=net,
                                   context_grade=grade, n_pos=int(pos.sum()), n_uni=int(uni.sum())))
            else:
                status.append(dict(panel=key, tf=tf, status="NO_CALL_LOW_POSITIVES", network=net,
                                   context_grade=grade, n_pos=int(pos.sum())))
    return out, pd.DataFrame(status)


# ------------------------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("--system", required=True)
    ap.add_argument("--kmax", type=int, default=30); ap.add_argument("--npc", type=int, default=30)
    a = ap.parse_args(); t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    selftest()

    S = load(a.system, canonical=True)
    genes0 = np.asarray(S["genes"]).astype(str); gu0 = np.char.upper(genes0)
    # UPPERCASE COLLISIONS. GATA1's canonical universe contains both `Setd1a` and `SETD1A`, which
    # collapse to one key in any uppercase map and would silently merge two distinct genes. Refusing
    # the whole system for one pair is the wrong trade (directive v3: mark only the invalid part
    # unavailable). ALL members of a colliding group are dropped -- never one arbitrarily kept -- and
    # the exclusion is recorded.
    _u, _c = np.unique(gu0, return_counts=True)
    collided = set(_u[_c > 1].tolist())
    keep_g = np.array([g not in collided for g in gu0])
    dropped_syms = sorted({g for g in genes0[~keep_g]})
    if collided:
        print(f"  DROPPED {int((~keep_g).sum())} genes in {len(collided)} uppercase-colliding "
              f"group(s): {dropped_syms}", flush=True)
    genes = genes0[keep_g]; gu = gu0[keep_g]
    if len(set(gu.tolist())) != gu.size:
        raise SystemExit("REFUSING[SYMBOL_COLLISION]: collisions persist after dropping")
    S["X"] = S["X"][:, keep_g]; S["V"] = S["V"][:, keep_g]
    Ms = S["X"]; Vl = S["V"]
    # The pair artifact DEFINES which cell pairs every difference is taken over, and its external
    # reconstruction receipt is the evidence the artifact was not edited. Binding only the fixture
    # left both outside the post-run rehash, so the pairing could change under a fixed code SHA
    # (internal review finding 4, second pass).
    bind("substrate_fixture", S["fixture"])
    bind("pair_artifact", S["pairs_file"])
    bind("pair_reconstruction_receipt", os.path.splitext(S["pairs_file"])[0] + ".receipt.json")
    bind("preprocess_contract", f"{R}/GEOBRIDGE_PREPROCESS_V1.json")
    A_ = ad.read_h5ad(S["fixture"])
    fv = np.asarray(A_.var_names).astype(str); pmap = {g: i for i, g in enumerate(fv)}
    colidx = np.array([pmap[g] for g in genes])   # fixture columns for the KEPT genes only
    Mu = np.asarray(A_.layers["Mu"], float)[:, colidx]
    P30 = np.asarray(A_.obsm["X_pca"], float)[:, :a.npc]
    kmax = min(a.kmax, a.npc)
    # NO DEPTH IS PRIVILEGED. PI, repeatedly and again on 2026-09-21: run the FULL k sweep FIRST and
    # decide the depth rule AFTERWARDS from it. An earlier revision used {0, k_star, k_hist} to decide
    # WHERE the expensive competitor competition ran and WHERE the matrix was serialized, and refused
    # to start without a k_star -- baking a not-yet-decided rule into the computation. Removed.
    kd_path = f"{R}/results/sv2_{a.system}/k_decision_v2.json"
    kd = json.load(open(kd_path)) if os.path.exists(kd_path) else {}
    KS = list(range(0, kmax + 1))
    DEPTH_OBJECTS = dict(all_depths=KS, reference_only_k_star=kd.get("k_star"),
                         reference_only_k_hist=K_HIST, reference_only_ladder=kd.get("ladder"),
                         note="recorded for cross-referencing older numbers ONLY; gates nothing, "
                              "selects nothing, restricts nothing")
    print(f"  FULL SWEEP over every depth 0..{kmax}; NO depth is privileged", flush=True)

    # DENTATEGYRUS and GASTRULATION have no perturbation truth, so their panels come from the
    # PRE-DECLARED reference networks. This dispatch was missing: chip_panels() was written but never
    # called, so both systems hit the UNKNOWN_SYSTEM branch.
    if a.system in ("DENTATEGYRUS", "GASTRULATION"):
        PAN, PANSTAT = chip_panels(a.system, genes)
        if not PAN:
            raise SystemExit(f"REFUSING[NO_EVALUABLE_REFERENCE_PANEL]: {a.system} produced zero "
                             f"panels from its declared networks")
    else:
        PAN, PANSTAT = panels(a.system, genes, None)
        # a declared reference network is an ADDITIONAL orthogonal layer where one exists
        CP, CPSTAT = chip_panels(a.system, genes)
        PAN.update(CP)
        PANSTAT = pd.concat([PANSTAT, CPSTAT], ignore_index=True)
    cat_path = TFCAT[a.system]
    bind("tf_catalogue", cat_path)
    cat = set(l.strip().upper() for l in open(cat_path) if l.strip())
    is_tf = np.array([g in cat for g in gu])
    gp = {g: i for i, g in enumerate(gu)}
    n_cand = int(is_tf.sum())
    print(f"{a.system}: {Ms.shape[0]} x {len(genes)} | panels {len(PAN)} | candidate TFs {n_cand} | "
          f"depths 0..{kmax}", flush=True)

    # STATIC comparator: |Pearson| of the source against every target, on the substrate. Dynamics are
    # not involved, so this is the first thing a reviewer reaches for.
    Z = Ms - Ms.mean(0); Z = Z / np.maximum(np.linalg.norm(Z, axis=0), 1e-300)
    # TARGET-ONLY baseline: regulator-independent target magnitude. Any score that merely tracks how
    # much a target moves must not be mistaken for regulation.
    TGT = {"target_Mu_magnitude": np.abs(Mu).mean(0), "target_V_magnitude": np.abs(Vl).mean(0),
           "target_expression": Ms.mean(0), "target_detection": (Ms > 0).mean(0)}

    rows, attrib, deltas, inelig, pergene = [], [], [], [], {}
    CKPT_EVERY = 20

    def checkpoint(root, done, total, phase):
        """Write everything finished so far. Partial files are named *_partial so nothing can
        mistake them for the final frozen artifact, and progress.json says exactly which panels are
        complete -- a partial file with no manifest of what is in it is unusable."""
        def _atomic(df, name):
            """Write then rename. A plain to_csv can be interrupted mid-row, leaving a truncated
            TSV beside a progress.json that claims it is complete (internal review checkpoint ruling)."""
            tmp = f"{a.out_dir}/.{name}.tmp"
            df.to_csv(tmp, sep="\t", index=False)
            os.replace(tmp, f"{a.out_dir}/{name}")
        try:
            # DATA FIRST, MANIFEST LAST: progress.json must never describe rows that are not yet on
            # disk, so it is written only after every partial table has been atomically replaced.
            if rows:
                _atomic(pd.DataFrame(rows), "scores_partial.tsv")
            if attrib:
                _atomic(pd.DataFrame(attrib), "attribution_partial.tsv")
            if deltas:
                _atomic(pd.DataFrame(deltas), "delta_uv_partial.tsv")
            json.dump(dict(system=a.system, phase=phase, root=root,
                           panels_complete=done, panels_total=total,
                           roots_total=len(S["pairs"]),
                           depths_per_panel=f"0..{kmax} (COMPLETE for every panel listed as done)",
                           score_rows=len(rows), attribution_rows=len(attrib),
                           elapsed_s=round(time.time() - t0),
                           note="PARTIAL. Each completed panel carries its full depth sweep in both "
                                "channels. Panels not yet listed are absent, not null.",
                           updated_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())),
                      open(f"{a.out_dir}/.progress.tmp", "w"), indent=1)
            os.replace(f"{a.out_dir}/.progress.tmp", f"{a.out_dir}/progress.json")
        except Exception as e:                      # a checkpoint must never kill the run
            print(f"  [checkpoint failed, continuing] {type(e).__name__}: {e}", flush=True)
    MATRIX_TOPN = 50; MATRIX_FLUSH = 200_000
    # UNIFORM depth grid for the expensive reference-network competition: every REF_STRIDE-th rung
    # plus the last. ARITHMETIC, not a depth rule -- 547 panels x 31 depths x 2 channels x 2 roots at
    # ~1.2 s per competition is ~22 h for one system. Perturbation/ChIPxKO panels keep EVERY depth,
    # and RECOVERY plus the U-V contrast keep EVERY depth for EVERY panel. No rung here is
    # privileged, chosen or interpreted.
    REF_STRIDE = 3
    REF_DEPTHS = sorted(set(range(0, kmax + 1, REF_STRIDE)) | {kmax})
    mrows = []
    import gzip as _gzip
    _mfh = _gzip.open(f"{a.out_dir}/attribution_matrix.tsv.gz", "wt")
    _mfh.write("system\troot\tpanel\ttf\tchannel\tk\tscope\tsource_tf\tis_cognate\t"
               "nAP\tstatic_nAP\tsupported\n")

    def _flush_matrix():
        for t in mrows:
            _mfh.write("\t".join(str(x) for x in t) + "\n")
        mrows.clear()
    for root, (I, J) in S["pairs"].items():
        dP = P30[J] - P30[I]
        dX = (Ms[J] - Ms[I]).astype(np.float32)
        dU = (Mu[J] - Mu[I]).astype(np.float32)
        dV = (Vl[J] - Vl[I]).astype(np.float32)
        print(f"  {root}: {len(I)} pairs  {time.time()-t0:.0f}s", flush=True)

        # ---- DEPTH-LOCAL SOURCE ELIGIBILITY (PI v5 section 3) ----------------------------------
        # Requiring support at EVERY depth 0..kmax to compete at ONE depth is a real bias: a
        # competitor that degenerates only at k=28 vanishes from the k=1 competition where it might
        # have beaten the cognate, which inflates the cognate's standing. Eligibility is therefore
        # evaluated AT THE DEPTH OF THE COMPARISON. The denominator now varies with depth, so absolute
        # ranks are NOT comparable across depths and every row carries n_eligible for its own depth.
        elig_k = {}
        # QR factors are kept and reused by the scoring pass below: the same projector must not be
        # factorised twice per (root, k) (PI 2026-09-21 item 2; internal review). The PROJECTION of dX is
        # still applied in both passes because holding 31 projected copies would cost ~27 GB; that
        # duplication is outside the panel loop and measured at <1% of runtime.
        QK = {}
        for k in KS:
            if k == 0:
                qx = dX
            else:
                QK[k], _ = np.linalg.qr(dP[:, :k]); Q = QK[k]; qx = dX - (Q @ (Q.T @ dX)).astype(np.float32)
            elig_k[k] = is_tf & (np.abs(qx).sum(0) > EPS)
            del qx
        # the union over depths is scored once; each comparison then restricts to its own depth
        any_elig = np.zeros(len(genes), bool)
        for k in KS:
            any_elig |= elig_k[k]
        src = np.flatnonzero(any_elig)
        src = src[np.argsort(gu[src], kind="stable")]
        src_pos = {int(i): j for j, i in enumerate(src)}
        for i in np.flatnonzero(is_tf & ~any_elig):
            inelig.append(dict(system=a.system, root=root, tf=genes[i], status="INELIGIBLE_SOURCE",
                               reason="degenerate_source_support_at_every_depth"))
        src_mask = np.zeros(len(genes), bool); src_mask[src] = True
        # the static source x target association, computed ONCE per root for the scored roster
        ST_full = np.abs(Z[:, src].T @ Z).astype(np.float32)
        src_sha = hashlib.sha256("\n".join(gu[src].tolist()).encode()).hexdigest()
        n_el = {k: int(elig_k[k][src].sum()) for k in KS}
        print(f"    candidate {n_cand} -> scored {src.size} (roster {src_sha[:12]}); "
              f"eligible per depth {min(n_el.values())}-{max(n_el.values())} "
              f"(k=0 {n_el[0]}, k={kmax} {n_el[kmax]})", flush=True)
        # An ineligible FOCAL is a per-panel status, never a run abort: one unscoreable regulator
        # must not destroy every other panel in the system (internal review 2026-09-21).
        skip = set()
        for pname in PAN:
            tf = pname.split("|")[0]
            if tf in gp and not any_elig[gp[tf]]:
                skip.add(pname)
                rows.append(dict(system=a.system, root=root, panel=pname, tf=tf,
                                 status="NO_CALL_FOCAL_INELIGIBLE_SOURCE",
                                 reason="degenerate_source_support_at_every_depth"))

        # ---- score every eligible source at every depth, both channels ----
        SC = {}
        for k in KS:
            if k == 0:
                Qx, YU, YV = dX, dU, dV
            else:
                Q = QK[k]                              # reused, not refactorised
                pj = lambda Y: (Y - Q @ (Q.T @ Y)).astype(np.float32)
                Qx, YU, YV = pj(dX), pj(dU), pj(dV)
            Xs = Qx[:, src]; aXs = np.abs(Xs)
            for chan, Y in (("U", YU), ("V", YV)):
                num = Xs.T @ Y; den = aXs.T @ np.abs(Y)
                sup = den > EPS
                with np.errstate(invalid="ignore", divide="ignore"):
                    E = np.where(sup, np.abs(num / np.where(sup, den, 1.0)), np.nan).astype(np.float32)
                SC[(k, chan)] = (E, sup, num.astype(np.float32), den.astype(np.float32))
            del Qx, YU, YV, Xs, aXs

        # ---- COMPARISON-SPECIFIC MASKS (PI v4 section 2A; internal review finding 2) --------------------
        # The previous single intersection over EVERY source x BOTH channels x ALL 31 depths let an
        # unrelated competitor failure at an exploratory depth delete a target from the focal k=1
        # comparison. Each comparison now carries the mask its own question needs, and each mask is
        # hashed:
        #   RECOVERY(chan,k)  panel universe INTERSECT focal support at THAT channel and depth
        #   UV(k)             focal support in U INTERSECT focal support in V at that depth
        #   DEPTHC(chan,k1,k2) focal support at both compared depths in that channel
        #   ATTRIB(chan,k)    support shared by the ELIGIBLE COMPETITORS at that channel and depth
        # NATIVE is retained as the per-channel sensitivity scope.
        # DEPTH-LOCAL, not union. `sup.all(0)` over the union roster let a source that is
        # INELIGIBLE at depth k still delete targets at depth k, so the competition at k was run on
        # a universe shaped by competitors not in it (internal review finding 2, 2026-09-21). The mask, the
        # source exclusion, the denominator and the roster hash must all describe EXACTLY the set
        # that competes.
        # SORT CACHE, keyed by exactly what the sort depends on: the score matrix identity
        # (channel, depth, or "static") and the set of columns and rows selected. Panels sharing a
        # mask reuse the sort; the labels are applied per panel. Bounded because there are only a
        # handful of distinct masks per (channel, depth).
        SORTC = {}
        _st_cache = {}                             # per-root focal static rows (float64)

        def prep_for(key, build):
            if key not in SORTC:
                if len(SORTC) > 96:                 # keep a lid on memory, never on correctness
                    SORTC.clear()
                SORTC[key] = nap_prepare(build())
            return SORTC[key]

        attrib_mask, attrib_roster_sha, src_mask_k = {}, {}, {}
        for k in KS:
            ekk = elig_k[k][src]
            mk = np.zeros(len(genes), bool); mk[src[ekk]] = True
            src_mask_k[k] = mk
            attrib_roster_sha[k] = hashlib.sha256(
                "\n".join(gu[src[ekk]].tolist()).encode()).hexdigest()
        for (k, chan), (_, sup, _, _) in SC.items():
            ekk = elig_k[k][src]
            attrib_mask[(k, chan)] = (sup[ekk].all(0) if ekk.any()
                                      else np.zeros(len(genes), bool))
        common_uv_mask = {k: attrib_mask[(k, "U")] & attrib_mask[(k, "V")] & ~src_mask_k[k]
                          for k in KS}
        print(f"    attribution universes per (channel,depth): "
              f"{int(min(m.sum() for m in attrib_mask.values()))}-"
              f"{int(max(m.sum() for m in attrib_mask.values()))} of {len(genes)} genes", flush=True)

        _npan = 0
        for pname, (pos, uni, meta) in PAN.items():
            tf = pname.split("|")[0]
            if tf not in gp or pname in skip:
                continue
            _npan += 1
            ti = gp[tf]; si = int(np.flatnonzero(src == ti)[0])
            for k in KS:
                for chan in ("U", "V"):
                    E, sup, num, den = SC[(k, chan)]
                    supU = SC[(k, "U")][1][si]; supV = SC[(k, "V")][1][si]
                    for uname, base in (("NATIVE", uni & sup[si]),
                                        ("RECOVERY", uni & sup[si]),
                                        ("UV", uni & supU & supV)):
                        m = base.copy(); m[ti] = False
                        p_ = pos & m
                        if m.sum() < 20 or p_.sum() < 5 or (m & ~p_).sum() < 5:
                            rows.append(dict(system=a.system, root=root, panel=pname, tf=tf,
                                             channel=chan, k=k, universe=uname,
                                             status="NO_CALL_LOW_SUPPORT",
                                             n_pos=int(p_.sum()), n_uni=int(m.sum())))
                            continue
                        y = p_[m].astype(int); f = E[si][m]
                        apv, prev = ap_prev(y, f)
                        r = dict(system=a.system, root=root, panel=pname, tf=tf, channel=chan, k=k,
                                 universe=uname, status="EVALUABLE", modality=meta.get("modality"),
                                 tier=meta.get("tier"), arm=meta.get("arm"),
                                 AP=apv, nAP=apv / prev if apv == apv else np.nan,
                                 cl_auc=cl_auc(y, f), prevalence=prev,
                                 n_pos=int(y.sum()), n_uni=int(m.sum()),
                                 mask_sha=sha_mask(m), roster_sha=src_sha)
                        r.update(topk(y, f, prev))
                        # L2 COMPARATORS ON EVERY ROW (PI v4 section 20; they were attached to a
                        # single k=0/U row before, which made an incremental claim uncheckable at
                        # every other depth). Computed on the IDENTICAL mask as the dynamic score.
                        # focal static row: identical float64 expression, computed once per (root,
                        # focal) instead of once per (depth, channel, universe). NOT the float32
                        # ST_full row -- that would not be bit-equal (internal review 2026-09-21).
                        if ti not in _st_cache:
                            _st_cache[ti] = np.abs(Z[:, ti] @ Z)
                        st = _st_cache[ti]
                        r["STATIC_nAP"] = nap(y, st[m])
                        r["STATIC_P@20_exp"] = topk(y, st[m], prev).get("P@20_exp")
                        for bn, bv in TGT.items():
                            r[f"TGTONLY_{bn}_nAP"] = nap(y, bv[m])
                        rows.append(r)

                    # ---- L3 attribution ----------------------------------------------------
                    # DECLARED COMPUTATIONAL SCOPE, fixed before any result is seen and recorded in
                    # provenance. A full competition costs one nap_rows over ~1900 sources per
                    # (panel, channel, depth); with 419 panels x 31 depths x 2 channels x 2 roots
                    # that is ~18 h PER ROOT for PANC alone, which is not a scientific choice but an
                    # arithmetic wall. The split is by PANEL CLASS, never by outcome:
                    #   PERTURBATION / ChIPxKO panels (few per system) -> EVERY depth
                    #   REFERENCE-NETWORK panels (hundreds per system) -> REF_DEPTHS, the uniform
                    #                                                     stride grid (NOT k_star /
                    #                                                     k_hist, which are deleted)
                    # RECOVERY and the U-V contrast are computed at EVERY depth for EVERY panel, so
                    # the full descriptive depth profile the directive asks for is unaffected.
                    _is_ref = (meta.get("modality") == "reference_network")
                    _attr_here = (not _is_ref) or (k in REF_DEPTHS)
                    if not _attr_here:
                        # Emit an explicit placeholder rather than nothing. A silently absent row is
                        # indistinguishable downstream from a row that was attempted and failed, and
                        # an available-case analysis over depth would then treat the stride grid as
                        # if it were the whole sweep (internal review finding 5).
                        attrib.append(dict(system=a.system, root=root, panel=pname, tf=tf,
                                           channel=chan, k=k, depth_role="sparse_reference_grid",
                                           depth_coverage_complete=False,
                                           competition_status="NOT_COMPUTED_OFF_REFERENCE_GRID",
                                           reason="reference-network attribution is computed only "
                                                  "on REF_DEPTHS for arithmetic reasons; this depth "
                                                  "MUST NOT enter any full-depth attribution "
                                                  "summary"))
                    elif True:
                        # STATIC ATTRIBUTION is computed once per panel on the identical mask and the
                        # identical source set, so "does the cognate win?" is asked of the dynamic score
                        # AND of plain co-expression. A dynamic cognate rank that a static score matches
                        # is not evidence of dynamic source specificity (reviewer-2 objection 1).
                        # exclude the DEPTH-LOCAL competitor set, not the union roster
                        m_a = uni & attrib_mask[(k, chan)] & ~src_mask_k[k]
                        ya = (pos & m_a)[m_a].astype(int)
                        ek = elig_k[k][src]
                        # ORDER MATTERS. The support test used to run first, so a focal that was
                        # INELIGIBLE at this depth -- or a depth with NO eligible competitors at all
                        # -- was reported as ordinary NO_CALL_ATTRIB_LOW_SUPPORT and the real reason
                        # disappeared (internal review blocker 3, 2026-09-21). Structural impossibility is
                        # diagnosed BEFORE support, and the two causes are named separately.
                        if not ek.any():
                            attrib.append(dict(system=a.system, root=root, panel=pname, tf=tf,
                                               channel=chan, k=k,
                                               competition_status="NO_CALL_NO_ELIGIBLE_SOURCES_AT_DEPTH",
                                               n_eligible_at_depth=0, n_candidate=n_cand,
                                               n_attrib_uni=int(m_a.sum())))
                            continue
                        if not ek[si]:
                            attrib.append(dict(system=a.system, root=root, panel=pname, tf=tf,
                                               channel=chan, k=k,
                                               competition_status="NO_CALL_FOCAL_INELIGIBLE_AT_DEPTH",
                                               n_eligible_at_depth=int(ek.sum()),
                                               n_candidate=n_cand, n_attrib_uni=int(m_a.sum()),
                                               reason="the focal regulator has no support at this "
                                                      "depth, so it is not in its own competition"))
                            continue
                        if not (m_a.sum() >= 20 and ya.sum() >= 5 and (len(ya) - ya.sum()) >= 5):
                            attrib.append(dict(system=a.system, root=root, panel=pname, tf=tf,
                                               channel=chan, k=k,
                                               competition_status="NO_CALL_ATTRIB_LOW_SUPPORT",
                                               n_attrib_uni=int(m_a.sum()), n_attrib_pos=int(ya.sum()),
                                               n_candidate=n_cand, n_eligible=int(src.size)))
                        if m_a.sum() >= 20 and ya.sum() >= 5 and (len(ya) - ya.sum()) >= 5:
                            # DEPTH-LOCAL competitor set: only sources eligible AT THIS DEPTH compete.
                            _ekI = np.flatnonzero(ek)
                            _mkey = sha_mask(m_a)[:24]
                            vals = np.full(src.size, np.nan)
                            vals[ek] = nap_apply(
                                prep_for(("dyn", chan, k, _mkey),
                                         lambda: E[_ekI][:, m_a]), ya)
                            nfail = int((~np.isfinite(vals[ek])).sum())
                            cog = vals[si]
                            sel = ek.copy(); sel[si] = False
                            oth = vals[sel]
                            # STATIC ON THE IDENTICAL MASK, ROSTER AND LABELS (PI v5 section 8). Caching
                            # one static competition per (panel, root) and reusing it across channel- and
                            # depth-specific masks meant the reported static rank could describe a
                            # DIFFERENT gene set than the dynamic rank it was compared against.
                            # STATIC on the IDENTICAL mask. internal review finding was that the static RESULT was
                            # cached across masks; the fix is to recompute the METRIC per mask, NOT the
                            # matmul. |Z[:,src].T @ Z| depends only on the substrate -- not on channel,
                            # depth or mask -- so it is computed ONCE per root. Recomputing it per cell
                            # meant 18,436 matmuls of 2.7e10 flops for PANC alone, which is days of work
                            # for a quantity that never changes.
                            sv = np.full(src.size, np.nan)
                            # the static matrix depends on NEITHER channel NOR depth, so one sort
                            # per (mask, roster) serves every panel and both channels
                            sv[ek] = nap_apply(
                                prep_for(("static", k, _mkey),
                                         lambda: ST_full[_ekI][:, m_a]), ya)
                            # COMMON U-AND-V TASK. Serializing a common mask was not enough: the
                            # REPORTED U and V ranks were still computed on separate channel masks,
                            # so "U beats V at attribution" compared two different questions. The
                            # rank on the shared mask is computed here for both channels, which is
                            # the only form in which the two are comparable (internal review finding 3, second
                            # pass). Where the shared mask cannot support a competition the fields
                            # are NaN and the comparison is simply not licensed.
                            m_c = common_uv_mask[k] & uni
                            yc = (pos & m_c)[m_c].astype(int)
                            c_rank = c_cog = c_med = np.nan; c_n = int(m_c.sum())
                            if m_c.sum() >= 20 and yc.sum() >= 5 and (len(yc) - yc.sum()) >= 5:
                                cv = np.full(src.size, np.nan)
                                cv[ek] = nap_apply(
                                    prep_for(("common", chan, k, sha_mask(m_c)[:24]),
                                             lambda: E[_ekI][:, m_c]), yc)
                                if np.isfinite(cv[si]):
                                    _sel = ek.copy(); _sel[si] = False
                                    c_cog = float(cv[si])
                                    c_rank = int(1 + np.nansum(cv[_sel] >= cv[si]))
                                    c_med = float(np.nanmedian(cv[_sel]))
                            s_cog = sv[si]; s_oth = sv[sel]
                            n_elig_k = int(ek.sum())
                            # The focal must be IN the competition for the competition to be
                            # evaluable. Deciding this from competitor failures alone let a
                            # depth-ineligible focal be labelled EVALUABLE with rank = -1, and
                            # `rank <= 3` then made it top3 (internal review finding 2 / ruling A).
                            # ek[si] is guaranteed above; what remains is a focal that is eligible
                            # but whose metric is non-finite. Static and dynamic are reported
                            # separately so a non-finite STATIC metric cannot suppress a legitimate
                            # dynamic competition (internal review question 2).
                            comp_status = ("NO_CALL_FOCAL_METRIC_NONFINITE" if not np.isfinite(cog)
                                           else "EVALUABLE" if nfail == 0 else
                                           "NO_CALL_INCOMPLETE_COMPETITION")
                            rank = int(1 + np.nansum(oth >= cog)) if np.isfinite(cog) else -1
                            s_rank = int(1 + np.nansum(s_oth >= s_cog)) if np.isfinite(s_cog) else -1
                            # PERCENTILES USE THE DEPTH-SPECIFIC DENOMINATOR (v5 section 3): absolute
                            # ranks are not comparable across depths because the competitor set changes.
                            pct = (rank / n_elig_k) if rank > 0 and n_elig_k else np.nan
                            s_pct = (s_rank / n_elig_k) if s_rank > 0 and n_elig_k else np.nan
                            attrib.append(dict(
                                system=a.system, root=root, panel=pname, tf=tf, channel=chan, k=k,
                                # A reference-network panel is competed only on REF_DEPTHS, so its
                                # depth coverage is NOT a full sweep and an available-case summary
                                # across depths would silently privilege that grid (internal review finding 5).
                                depth_role=("sparse_reference_grid" if _is_ref else "full_sweep"),
                                depth_coverage_complete=bool(not _is_ref),
                                competition_status=comp_status,
                                n_candidate=n_cand, n_scored_roster=int(src.size),
                                n_eligible_at_depth=n_elig_k, n_failed=nfail,
                                cognate_nAP=float(cog), cognate_rank=rank, rank_percentile=pct,
                                rank_tie_convention="WORST rank within a tie: rank = 1 + #{competitors "
                                                    "with nAP >= cognate}, so ties count against the "
                                                    "cognate (internal review 2026-09-21, disclose explicitly)",
                                competitor_median=float(np.nanmedian(oth)),
                                competitor_best=float(np.nanmax(oth)),
                                cognate_minus_median=float(cog - np.nanmedian(oth)),
                                cognate_minus_best=float(cog - np.nanmax(oth)),
                                static_cognate_nAP=float(s_cog), static_cognate_rank=s_rank,
                                static_rank_percentile=s_pct,
                                static_competitor_median=float(np.nanmedian(s_oth)),
                                # v5 section 21 axes X and Y, both on ONE mask / roster / label vector.
                                # NAMING BUG FIXED (internal review 2026-09-21): `cog` and `s_cog` come from
                                # nap_rows and are nAP, so their difference is a nAP difference, not an
                                # AP difference. On this mask AP = nAP * prevalence, so the raw AP
                                # increment is prevalence * (cog - s_cog). Reporting the nAP difference
                                # under the name Delta_AP implicitly UPWEIGHTS LOW-PREVALENCE PANELS when
                                # pooled across panels, which is exactly what the cross-system X axis
                                # does. Both are emitted, correctly named.
                                prevalence_attrib=float(ya.mean()),
                                Delta_nAP=float(cog - s_cog),
                                Delta_AP=float(ya.mean() * (cog - s_cog)),
                                Delta_rank_percentile=float(s_pct - pct) if (pct == pct and s_pct == s_pct) else np.nan,
                                top1=(bool(rank == 1) if comp_status == "EVALUABLE" else None),
                                top3=(bool(1 <= rank <= 3) if comp_status == "EVALUABLE" else None),
                                n_attrib_uni=int(m_a.sum()), n_attrib_pos=int(ya.sum()),
                                mask_sha=sha_mask(m_a),
                                roster_sha=attrib_roster_sha[k],      # the DEPTH-LOCAL competitors
                                union_roster_sha=src_sha,
                                # A U-versus-V attribution comparison is only well posed on a task
                                # both channels actually share; these two columns are identical
                                # across the U and V rows of a cell, so a reviewer can verify that
                                # the two ranks answered the SAME question (internal review finding 3).
                                common_uv_mask_sha=sha_mask(m_c), n_common_uv_uni=c_n,
                                common_task_cognate_nAP=c_cog, common_task_cognate_rank=c_rank,
                                common_task_competitor_median=c_med,
                                common_task_n_pos=int(yc.sum()) if c_n else 0,
                                common_task_note="the ONLY fields in which the U and V attribution "
                                                 "ranks may be compared; the per-channel rank uses "
                                                 "a channel-specific mask and answers a different "
                                                 "question"))
                            # COMPETITOR MATRIX, STREAMED (internal review 2026-09-21 finding 5). Accumulating one
                            # Python dict per source x panel x depth x channel x root is ~34M objects for
                            # PANC and ~160M for GATA1; the PANC smoke reached 11.8 GB RSS in 16 minutes
                            # before this was fixed. Rows are now streamed to a gzipped TSV and never
                            # retained. The FULL competitor set is written at REF_DEPTHS and the
                            # TOP-N at the other depths. NOTE, precisely: this applies only to
                            # depths that REACH this code. Reference-network panels return above for
                            # k not in REF_DEPTHS, so they get NO competitor record there at all --
                            # the earlier comment claimed every depth had one, and also still named
                            # the deleted k_star / k_hist depths (internal review ruling B).
                            cand = np.flatnonzero(ek)
                            if k not in REF_DEPTHS:
                                ordv = cand[np.argsort(-np.nan_to_num(vals[cand], nan=-np.inf))][:MATRIX_TOPN]
                                cand = np.unique(np.concatenate([ordv, [si]]))
                            for j in cand:
                                mrows.append((a.system, root, pname, tf, chan, k,
                                              "FULL" if k in REF_DEPTHS else f"TOP{MATRIX_TOPN}",
                                              genes[src[j]], int(src[j] == ti),
                                              float(vals[j]), float(sv[j]),
                                              int(np.isfinite(vals[j]))))
                            if len(mrows) >= MATRIX_FLUSH:
                                _flush_matrix()

                # ---- paired U vs V on the UV universe, DESCRIPTIVE ----
                EU, sU, _, _ = SC[(k, "U")]; EV, sV, _, _ = SC[(k, "V")]
                m = uni & SC[(k, "U")][1][si] & SC[(k, "V")][1][si]; m[ti] = False
                p_ = pos & m
                if m.sum() >= 20 and p_.sum() >= 5 and (m & ~p_).sum() >= 5:
                    y = p_[m].astype(int); u = EU[si][m]; v = EV[si][m]
                    assert np.isfinite(u).all() and np.isfinite(v).all(), \
                        "the UV scope must be fully supported in both channels"
                    # FRACTIONAL EXPECTED P@K, not the order-dependent raw P@20: PANC shows a tie
                    # block crossing K=20 in 44 of 44 rows, so raw P@20 would let gene order decide
                    # the channel contrast (PI v4 section 18).
                    # VECTORISED NULL (cProfile 2026-09-21: the scalar loop here was 91% of total
                    # runtime -- 3 metrics x N_PERM x 2 channels of Python ap_prev/topk per (panel,
                    # depth)). gb_uvnull sorts each channel ONCE and evaluates all N_PERM permuted
                    # label vectors as one matrix; the rng is consumed in the identical order, so
                    # every null draw is bit-identical to the scalar path (test_uvnull_equiv.py:
                    # 54 cells x 500 draws, worst |diff| 6.7e-16).
                    _null = uv_perm_null(y, u, v, N_PERM, 20260921 + k)
                    for mn in ("nAP", "P@20_exp", "gain@5pct_exp"):
                        tu, tv, dn = _null[mn]
                        d = tu - tv
                        lo = float(np.nanmean(dn <= d)); hi = float(np.nanmean(dn >= d))
                        deltas.append(dict(system=a.system, root=root, panel=pname, tf=tf, k=k,
                                           metric=mn, T_U=tu, T_V=tv, delta_UV=d,
                                           null_mean=float(np.nanmean(dn)), null_sd=float(np.nanstd(dn)),
                                           emp_lower_tail=(1 + float(np.nansum(dn <= d))) / (N_PERM + 1),
                                           emp_upper_tail=(1 + float(np.nansum(dn >= d))) / (N_PERM + 1),
                                           n_perm=N_PERM, mask_sha=sha_mask(m),
                                           semantics="DESCRIPTIVE paired difference on the UV "
                                                     "scope. The permutation reference is a "
                                                     "CONDITIONAL LABEL-ASSOCIATION null on fixed "
                                                     "score vectors -- NOT a test of equal predictive "
                                                     "performance, NOT a biological-replicate test."))
                if True:                       # per-gene arrays at EVERY depth
                    for chan in ("U", "V"):
                        E, sup, num, den = SC[(k, chan)]
                        tag = f"{root}|{chan}|k{k}|{pname}"
                        pergene[f"{tag}|absS"] = E[si]
                        pergene[f"{tag}|num"] = np.where(sup[si], num[si], np.nan).astype(np.float32)
                        pergene[f"{tag}|den"] = np.where(sup[si], den[si], np.nan).astype(np.float32)
                        pergene[f"{tag}|pos"] = pos
                        pergene[f"{tag}|panel_universe"] = uni
                        # THE EXACT MASKS THE STATISTICS USED. The computed RECOVERY and UV masks
                        # clear the FOCAL TARGET (`m[ti] = False`); the serialized ones did not, so
                        # offline reconstruction could return a different AP/nAP/top-k and a
                        # different mask hash whenever the focal is itself a panel member -- which
                        # is reachable for GATA1, whose focal appears in class A (internal review finding 3).
                        _mr = uni & sup[si]; _mr[ti] = False
                        _mu = uni & SC[(k, "U")][1][si] & SC[(k, "V")][1][si]; _mu[ti] = False
                        _ma = uni & attrib_mask[(k, chan)] & ~src_mask_k[k]
                        pergene[f"{tag}|mask_RECOVERY"] = _mr
                        pergene[f"{tag}|mask_UV"] = _mu
                        pergene[f"{tag}|attrib_mask"] = _ma
                        # the COMMON U-and-V attribution task. U and V attribution ranks are
                        # otherwise computed on separate channel-specific masks, so a direct
                        # "U beats V at attribution" claim would compare two different tasks.
                        _mc = (uni & attrib_mask[(k, "U")] & attrib_mask[(k, "V")]
                               & ~src_mask_k[k])
                        pergene[f"{tag}|attrib_mask_common_UV"] = _mc
            # AFTER the panel is finished, never before. Checkpointing at the TOP of the iteration
            # reported N complete when only N-1 were, so the partial file contradicted its own
            # manifest (internal review checkpoint ruling).
            if _npan % CKPT_EVERY == 0:
                _flush_matrix()
                checkpoint(root, _npan, len(PAN), "scoring")
                print(f"    [{root}] {_npan}/{len(PAN)} panels COMPLETE over all depths "
                      f"({time.time()-t0:.0f}s) -- partial outputs refreshed", flush=True)
        SORTC.clear()
        _flush_matrix(); checkpoint(root, _npan, _npan, "root_complete")
        print(f"    [{root}] ROOT COMPLETE ({time.time()-t0:.0f}s)", flush=True)
        del dP, dX, dU, dV, SC, ST_full

    _flush_matrix(); _mfh.close()
    D = pd.DataFrame(rows); D.to_csv(f"{a.out_dir}/scores.tsv", sep="\t", index=False)
    pd.DataFrame(attrib).to_csv(f"{a.out_dir}/attribution.tsv", sep="\t", index=False)
    pd.DataFrame(deltas).to_csv(f"{a.out_dir}/delta_uv.tsv", sep="\t", index=False)
    if dropped_syms:
        inelig.append(dict(system=a.system, root="-", tf=";".join(dropped_syms),
                           status="DROPPED_UPPERCASE_COLLISION",
                           reason="all members of an uppercase-colliding symbol group are removed "
                                  "from the universe; none is arbitrarily kept"))
    pd.concat([PANSTAT, pd.DataFrame(inelig)], ignore_index=True).to_csv(
        f"{a.out_dir}/status_roster.tsv", sep="\t", index=False)
    np.savez_compressed(f"{a.out_dir}/per_gene.npz", genes=np.asarray(genes, dtype=object), **pergene)
    json.dump(dict(system=a.system, operator="R1/R3", estimand="E = |S| (unsigned)",
                   layers_run=["L1_raw", "L2_static_and_targetonly", "L3_actualTF_attribution"],
                   layers_absent={"L4_matched_nonTF": "separate job (gb_m1v2_roster)",
                                  "L5_reference_network": "requires a PRE-DECLARED network"},
                   depths=KS,
                   depth_objects=DEPTH_OBJECTS,
                   deployment_contrast="NONE DECLARED. The full sweep is produced FIRST and the depth "
                                       "rule is decided afterwards from it (PI 2026-09-21). No depth "
                                       "is privileged and no max-over-k is confirmatory.",
                   reference_panel_depth_grid=REF_DEPTHS,
                   universes={"RECOVERY": "panel universe INTERSECT focal support at that channel "
                                          "and depth",
                              "UV": "focal support in U INTERSECT V at that depth -- the ONLY scope "
                                    "in which the two channels may be compared",
                              "NATIVE": "the channel's own scoreable genes -- per-channel sensitivity",
                              "ATTRIB": "per (channel, depth) universe shared by the eligible "
                                        "competitors, with every candidate source excluded"},
                   participation_ratio="edge_contribution_PR is NOT an effective sample size and is "
                                       "not computed here; the primary regulator-independent "
                                       "observability controls are target expression, detection and "
                                       "target U/V magnitude, emitted on every row",
                   attribution_semantics="cognate rank and rank percentile are ATTRIBUTION METRICS, "
                                         "not biological p-values",
                   n_candidate_tfs=n_cand, npc=a.npc, n_perm=N_PERM,
                   n_genes_dropped_collision=len(dropped_syms), dropped_symbols=dropped_syms,
                   gene_universe_sha256=hashlib.sha256("\n".join(gu.tolist()).encode()).hexdigest(),
                   tf_catalogue=cat_path, tf_catalogue_sha256=sha_file(cat_path),
                   truth_artifacts=TRUTH_ARTIFACTS,
                   truth_artifacts_sha256=hashlib.sha256(
                       json.dumps(TRUTH_ARTIFACTS, sort_keys=True).encode()).hexdigest(),
                   snapshot_sha=os.environ.get("GB_SNAPSHOT_SHA"),
                   benchmark_run_id=os.environ.get("GB_BENCHMARK_RUN_ID"),
                   canonical_sha=os.environ.get("GB_CANONICAL_SHA"),
                   selection_performed=False, wall_s=time.time() - t0),
              open(f"{a.out_dir}/layers_provenance.json", "w"), indent=1)

    pd.set_option("display.width", 250)
    Ev = D[(D.status == "EVALUABLE") & (D.universe == "RECOVERY")]
    print(f"\n=== {a.system}: {len(Ev)} evaluable RECOVERY-scope cells ===")
    for pname in sorted(Ev.panel.unique())[:4]:
        s = Ev[(Ev.panel == pname) & (Ev.root == sorted(Ev.root.unique())[0])]
        if not len(s):
            continue
        print(f"\n--- {pname}  n_pos={int(s.n_pos.iloc[0])}  prevalence={s.prevalence.iloc[0]:.4f} ---")
        print(s[s.k.isin([0, 1, 2, 5, K_HIST, 15, 20, kmax])]
              .pivot_table(index="k", columns="channel", values=["nAP", "P@20", "lift@20"])
              .round(3).to_string())
    print(f"\nwrote {a.out_dir} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
