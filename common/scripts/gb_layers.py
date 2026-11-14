#!/usr/bin/env python
# Layered evaluation of the GeoBRIDGE score on every scoreable real substrate
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
K_HIST = 10                      # historical continuity only
TFCAT = {"GATA1": f"{R}/data/LaraAstiaso_raw/ext_ref/external_regulators.txt",
         "PANC": f"{R}/data/LaraAstiaso_raw/ext_ref/external_regulators.txt",
         "DENTATEGYRUS": f"{R}/data/LaraAstiaso_raw/ext_ref/external_regulators.txt",
         "GASTRULATION": f"{R}/data/LaraAstiaso_raw/ext_ref/external_regulators.txt",
         "K562": f"{R}/GB_real_v2/lambert2018_human_tfs.txt",
         "BONEMARROW": f"{R}/GB_real_v2/lambert2018_human_tfs.txt"}

BEELINE = f"{R}/data/beeline_networks/Networks"
CHIP_DECL = {
    ("DENTATEGYRUS", "mouse/Non-Specific-ChIP-seq-network.csv"): ("USE", "LOW", 1),
    ("GASTRULATION", "mouse/mESC-ChIP-seq-network.csv"):         ("USE", "MEDIUM", 1),
    ("GASTRULATION", "mouse/Non-Specific-ChIP-seq-network.csv"): ("USE", "LOW", 2),
    ("PANC", "mouse/Non-Specific-ChIP-seq-network.csv"):         ("USE", "LOW", 1),
    ("GATA1", "mouse/mHSC-ChIP-seq-network.csv"):                ("USE", "MEDIUM", 1),
    ("GATA1", "mouse/Non-Specific-ChIP-seq-network.csv"):        ("USE", "LOW", 2),
    ("BONEMARROW", "human/Non-specific-ChIP-seq-network.csv"):   ("USE", "LOW", 1),
}
# BEELINE "Non-Specific" combines DoRothEA, RegNetwork and TRRUST; it is not an occupancy-only reference.
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


# metrics
# Tie-grouped AP and prevalence
# NaN scores are already excluded by the caller's mask
def ap_prev(y, s):
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


# Tie-grouped nAP for every row of X at once
def nap_rows(X, y):
    return nap_apply(nap_prepare(X), y)


# The order-dependent half of nap_rows, which is everything expensive and which depends only on the score matrix -- never on the labels
def nap_prepare(X):
    X = np.asarray(X, dtype=np.float64)
    n = X.shape[1]
    o = np.argsort(-X, axis=1, kind="stable")
    Ssort = np.take_along_axis(X, o, axis=1)
    newgrp = np.ones(X.shape, bool); newgrp[:, 1:] = Ssort[:, 1:] != Ssort[:, :-1]
    end = np.zeros(X.shape, bool); end[:, :-1] = newgrp[:, 1:]; end[:, -1] = True
    end_shift = np.zeros(X.shape, bool); end_shift[:, 1:] = end[:, :-1]
    pos_i = np.arange(n)[None, :]
    # end_shift is True at position p+1 when p was a group end, so the value recorded there must be
    # p -- the index of the previous end -- not p+1
    last_prev = np.maximum.accumulate(np.where(end_shift, pos_i - 1, -1), axis=1)
    return dict(o=o, end=end, last_prev=last_prev, n=n, shape=X.shape,
                finite=np.isfinite(X).all(1),
                idx=np.arange(1, n + 1, dtype=np.float64)[None, :])


# The label-dependent half: one gather, one cumsum, one reduction
# O(rows x n) with no sort
def nap_apply(prep, y):
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
        # fractional expected P@K
        cut = ss[N - 1]
        above = ss > cut
        tied = ss == cut
        n_above = int(above.sum()); n_tied = int(tied.sum())
        slots = N - n_above
        hit_tied = float(yy[tied].sum())
        out[f"P@{N}_exp"] = float(yy[above].sum() + hit_tied * slots / n_tied) / N
        out[f"lift@{N}_exp"] = out[f"P@{N}_exp"] / prev if prev > 0 else np.nan
        out[f"tie@{N}"] = n_tied
        out[f"tie_crosses@{N}"] = bool(n_above < N < n_above + n_tied)
    for q in (0.01, 0.05, 0.10):
        m = max(int(round(q * len(yy))), 1)
        out[f"gain@{int(q*100)}pct"] = float(yy[:m].sum()) / tot
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
    # vectorised nAP must equal the scalar one row by row, including with heavy ties and a fully
    # unsupported (all-NaN) row
    M = rng.normal(size=(40, 100))
    M[:, ::7] = 0.25                      # deliberate tie blocks
    M[3] = np.nan
    v = nap_rows(M, y)
    for i in (0, 1, 2, 5, 17, 39):
        assert abs(v[i] - nap(y, M[i])) < 1e-9, (i, v[i], nap(y, M[i]))
    assert np.isnan(v[3]), "an all-NaN source row must stay NaN"
    assert abs(nap_rows(np.ones((2, 100)), y)[0] - 1.0) < 1e-9, "all ties must be chance"
    print("selftest OK (scalar and vectorised nAP agree, ties and NaN rows included)", flush=True)


TRUTH_ARTIFACTS = {}


# Record a scientific input's hash and return the path
def bind(label, path):
    if os.path.exists(path):
        TRUTH_ARTIFACTS[label] = dict(path=path, sha256=sha_file(path))
    else:
        TRUTH_ARTIFACTS[label] = dict(path=path, sha256="ABSENT")
    return path


# truth
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
        # The two files gb_dev_evaluate.truth_panc reads (verified at gb_dev_evaluate.py:74-75)
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
        # Colliding symbols are dropped entirely, never resolved by keeping one arbitrarily: a
        # dict(zip(...)) would silently retain whichever row came last
        dupmask = sym.duplicated(keep=False)
        n_dup = int(dupmask.sum())
        if n_dup:
            print(f"  NT table: dropped {n_dup} rows in "
                  f"{int(sym[dupmask].nunique())} uppercase-colliding symbol group(s)", flush=True)
        good = nt.loc[~dupmask]
        det = dict(zip(good["gene"].astype(str).str.upper(), good["nt_detect_frac"].astype(float)))
        dv = np.array([det.get(g, np.nan) for g in gu])
        base = np.isfinite(dv) & (dv >= 0.05)
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
            uni = (base | pos); uni[gp[tf]] = False
            # predeclared strict sensitivity: NTC>=5% applied symmetrically to both classes.
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
            pos0 = have & ((row == 2) | (row == 3))     # tested-up or tested-down
            # primary negatives are well-observed tested-nulls (state 1)
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


# reference-network panels from the specified networks only
def chip_panels(system, genes):
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


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("--system", required=True)
    ap.add_argument("--kmax", type=int, default=30); ap.add_argument("--npc", type=int, default=30)
    a = ap.parse_args(); t0 = time.time()
    os.makedirs(a.out_dir, exist_ok=True)
    selftest()

    S = load(a.system, canonical=True)
    genes0 = np.asarray(S["genes"]).astype(str); gu0 = np.char.upper(genes0)
    # uppercase collisions
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
    bind("substrate_fixture", S["fixture"])
    bind("pair_artifact", S["pairs_file"])
    bind("pair_reconstruction_receipt", os.path.splitext(S["pairs_file"])[0] + ".receipt.json")
    bind("preprocess_contract", f"{R}/GEOBRIDGE_PREPROCESS_V1.json")
    A_ = ad.read_h5ad(S["fixture"])
    fv = np.asarray(A_.var_names).astype(str); pmap = {g: i for i, g in enumerate(fv)}
    colidx = np.array([pmap[g] for g in genes])   # fixture columns for the kept genes only
    Mu = np.asarray(A_.layers["Mu"], float)[:, colidx]
    P30 = np.asarray(A_.obsm["X_pca"], float)[:, :a.npc]
    kmax = min(a.kmax, a.npc)
    kd_path = f"{R}/results/sv2_{a.system}/k_decision_v2.json"
    kd = json.load(open(kd_path)) if os.path.exists(kd_path) else {}
    KS = list(range(0, kmax + 1))
    DEPTH_OBJECTS = dict(all_depths=KS, reference_only_k_star=kd.get("k_star"),
                         reference_only_k_hist=K_HIST, reference_only_ladder=kd.get("ladder"),
                         note="recorded for cross-referencing older numbers ONLY; gates nothing, "
                              "selects nothing, restricts nothing")
    print(f"  FULL SWEEP over every depth 0..{kmax}; NO depth is privileged", flush=True)

    # DENTATEGYRUS and GASTRULATION have no perturbation truth, so their panels come from the
    # specified reference networks
    if a.system in ("DENTATEGYRUS", "GASTRULATION"):
        PAN, PANSTAT = chip_panels(a.system, genes)
        if not PAN:
            raise SystemExit(f"REFUSING[NO_EVALUABLE_REFERENCE_PANEL]: {a.system} produced zero "
                             f"panels from its declared networks")
    else:
        PAN, PANSTAT = panels(a.system, genes, None)
        # a declared reference network is an additional orthogonal layer where one exists
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

    Z = Ms - Ms.mean(0); Z = Z / np.maximum(np.linalg.norm(Z, axis=0), 1e-300)
    # target-only baseline: regulator-independent target magnitude
    TGT = {"target_Mu_magnitude": np.abs(Mu).mean(0), "target_V_magnitude": np.abs(Vl).mean(0),
           "target_expression": Ms.mean(0), "target_detection": (Ms > 0).mean(0)}

    rows, attrib, deltas, inelig, pergene = [], [], [], [], {}
    CKPT_EVERY = 20

    # Write everything finished so far
    def checkpoint(root, done, total, phase):
        # Write then rename
        def _atomic(df, name):
            tmp = f"{a.out_dir}/.{name}.tmp"
            df.to_csv(tmp, sep="\t", index=False)
            os.replace(tmp, f"{a.out_dir}/{name}")
        try:
            # data first, manifest last: progress.json must never describe rows that are not yet on
            # disk
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
    # uniform depth grid for the expensive reference-network competition: every REF_STRIDE-th rung
    # plus the last
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

        elig_k = {}
        QK = {}
        for k in KS:
            if k == 0:
                qx = dX
            else:
                QK[k], _ = np.linalg.qr(dP[:, :k]); Q = QK[k]; qx = dX - (Q @ (Q.T @ dX)).astype(np.float32)
            elig_k[k] = is_tf & (np.abs(qx).sum(0) > EPS)
            del qx
        # the union over depths is scored once
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
        # the static source x target association, computed once per root for the scored roster
        ST_full = np.abs(Z[:, src].T @ Z).astype(np.float32)
        src_sha = hashlib.sha256("\n".join(gu[src].tolist()).encode()).hexdigest()
        n_el = {k: int(elig_k[k][src].sum()) for k in KS}
        print(f"    candidate {n_cand} -> scored {src.size} (roster {src_sha[:12]}); "
              f"eligible per depth {min(n_el.values())}-{max(n_el.values())} "
              f"(k=0 {n_el[0]}, k={kmax} {n_el[kmax]})", flush=True)
        skip = set()
        for pname in PAN:
            tf = pname.split("|")[0]
            if tf in gp and not any_elig[gp[tf]]:
                skip.add(pname)
                rows.append(dict(system=a.system, root=root, panel=pname, tf=tf,
                                 status="NO_CALL_FOCAL_INELIGIBLE_SOURCE",
                                 reason="degenerate_source_support_at_every_depth"))

        # score every eligible source at every depth, both channels
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

        SORTC = {}
        _st_cache = {}                             # per-root focal static rows (float64)

        def prep_for(key, build):
            if key not in SORTC:
                if len(SORTC) > 96:                 # Process source rows in blocks to limit memory.
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
                        if ti not in _st_cache:
                            _st_cache[ti] = np.abs(Z[:, ti] @ Z)
                        st = _st_cache[ti]
                        r["STATIC_nAP"] = nap(y, st[m])
                        r["STATIC_P@20_exp"] = topk(y, st[m], prev).get("P@20_exp")
                        for bn, bv in TGT.items():
                            r[f"TGTONLY_{bn}_nAP"] = nap(y, bv[m])
                        rows.append(r)

                    # L3 attribution
                    # Reference-network attribution uses REF_DEPTHS; perturbation and ChIP/KO
                    # attribution use every depth. Recovery and U-V contrasts use every depth for
                    # all panels.
                    _is_ref = (meta.get("modality") == "reference_network")
                    _attr_here = (not _is_ref) or (k in REF_DEPTHS)
                    if not _attr_here:
                        attrib.append(dict(system=a.system, root=root, panel=pname, tf=tf,
                                           channel=chan, k=k, depth_role="sparse_reference_grid",
                                           depth_coverage_complete=False,
                                           competition_status="NOT_COMPUTED_OFF_REFERENCE_GRID",
                                           reason="reference-network attribution is computed only "
                                                  "on REF_DEPTHS for arithmetic reasons; this depth "
                                                  "MUST NOT enter any full-depth attribution "
                                                  "summary"))
                    elif True:
                        m_a = uni & attrib_mask[(k, chan)] & ~src_mask_k[k]
                        ya = (pos & m_a)[m_a].astype(int)
                        ek = elig_k[k][src]
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
                            # depth-local competitor set: only sources eligible at this depth
                            # compete.
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
                            sv = np.full(src.size, np.nan)
                            # the static matrix depends on neither channel nor depth, so one sort
                            # per (mask, roster) serves every panel and both channels
                            sv[ek] = nap_apply(
                                prep_for(("static", k, _mkey),
                                         lambda: ST_full[_ekI][:, m_a]), ya)
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
                            comp_status = ("NO_CALL_FOCAL_METRIC_NONFINITE" if not np.isfinite(cog)
                                           else "EVALUABLE" if nfail == 0 else
                                           "NO_CALL_INCOMPLETE_COMPETITION")
                            rank = int(1 + np.nansum(oth >= cog)) if np.isfinite(cog) else -1
                            s_rank = int(1 + np.nansum(s_oth >= s_cog)) if np.isfinite(s_cog) else -1
                            # Attribution percentiles use depth-specific competitor sets:
                            # absolute ranks are not comparable across depths because the competitor
                            # set changes.
                            pct = (rank / n_elig_k) if rank > 0 and n_elig_k else np.nan
                            s_pct = (s_rank / n_elig_k) if s_rank > 0 and n_elig_k else np.nan
                            attrib.append(dict(
                                system=a.system, root=root, panel=pname, tf=tf, channel=chan, k=k,
                                depth_role=("sparse_reference_grid" if _is_ref else "full_sweep"),
                                depth_coverage_complete=bool(not _is_ref),
                                competition_status=comp_status,
                                n_candidate=n_cand, n_scored_roster=int(src.size),
                                n_eligible_at_depth=n_elig_k, n_failed=nfail,
                                cognate_nAP=float(cog), cognate_rank=rank, rank_percentile=pct,
                                rank_tie_convention="WORST rank within a tie: rank = 1 + #{competitors "
                                                    "with nAP >= cognate}, so ties count against the "
                                                    "cognate (disclosed explicitly)",
                                competitor_median=float(np.nanmedian(oth)),
                                competitor_best=float(np.nanmax(oth)),
                                cognate_minus_median=float(cog - np.nanmedian(oth)),
                                cognate_minus_best=float(cog - np.nanmax(oth)),
                                static_cognate_nAP=float(s_cog), static_cognate_rank=s_rank,
                                static_rank_percentile=s_pct,
                                static_competitor_median=float(np.nanmedian(s_oth)),
                                prevalence_attrib=float(ya.mean()),
                                Delta_nAP=float(cog - s_cog),
                                Delta_AP=float(ya.mean() * (cog - s_cog)),
                                Delta_rank_percentile=float(s_pct - pct) if (pct == pct and s_pct == s_pct) else np.nan,
                                top1=(bool(rank == 1) if comp_status == "EVALUABLE" else None),
                                top3=(bool(1 <= rank <= 3) if comp_status == "EVALUABLE" else None),
                                n_attrib_uni=int(m_a.sum()), n_attrib_pos=int(ya.sum()),
                                mask_sha=sha_mask(m_a),
                                roster_sha=attrib_roster_sha[k],      # the depth-local competitors
                                union_roster_sha=src_sha,
                                common_uv_mask_sha=sha_mask(m_c), n_common_uv_uni=c_n,
                                common_task_cognate_nAP=c_cog, common_task_cognate_rank=c_rank,
                                common_task_competitor_median=c_med,
                                common_task_n_pos=int(yc.sum()) if c_n else 0,
                                common_task_note="the ONLY fields in which the U and V attribution "
                                                 "ranks may be compared; the per-channel rank uses "
                                                 "a channel-specific mask and answers a different "
                                                 "question"))
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

                # paired U vs V on the UV universe, DESCRIPTIVE
                EU, sU, _, _ = SC[(k, "U")]; EV, sV, _, _ = SC[(k, "V")]
                m = uni & SC[(k, "U")][1][si] & SC[(k, "V")][1][si]; m[ti] = False
                p_ = pos & m
                if m.sum() >= 20 and p_.sum() >= 5 and (m & ~p_).sum() >= 5:
                    y = p_[m].astype(int); u = EU[si][m]; v = EV[si][m]
                    assert np.isfinite(u).all() and np.isfinite(v).all(), \
                        "the UV scope must be fully supported in both channels"
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
                if True:                       # per-gene arrays at every depth
                    for chan in ("U", "V"):
                        E, sup, num, den = SC[(k, chan)]
                        tag = f"{root}|{chan}|k{k}|{pname}"
                        pergene[f"{tag}|absS"] = E[si]
                        pergene[f"{tag}|num"] = np.where(sup[si], num[si], np.nan).astype(np.float32)
                        pergene[f"{tag}|den"] = np.where(sup[si], den[si], np.nan).astype(np.float32)
                        pergene[f"{tag}|pos"] = pos
                        pergene[f"{tag}|panel_universe"] = uni
                        _mr = uni & sup[si]; _mr[ti] = False
                        _mu = uni & SC[(k, "U")][1][si] & SC[(k, "V")][1][si]; _mu[ti] = False
                        _ma = uni & attrib_mask[(k, chan)] & ~src_mask_k[k]
                        pergene[f"{tag}|mask_RECOVERY"] = _mr
                        pergene[f"{tag}|mask_UV"] = _mu
                        pergene[f"{tag}|attrib_mask"] = _ma
                        # the common U-and-V attribution task
                        _mc = (uni & attrib_mask[(k, "U")] & attrib_mask[(k, "V")]
                               & ~src_mask_k[k])
                        pergene[f"{tag}|attrib_mask_common_UV"] = _mc
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
                                  "L5_reference_network": "requires a reference network specified before evaluation"},
                   depths=KS,
                   depth_objects=DEPTH_OBJECTS,
                   deployment_contrast="Historical per-depth diagnostic. Production uses U-channel Z0:8; "
                                       "the full profile is descriptive and does not select depth.",
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
    print(f"\n{a.system}: {len(Ev)} evaluable RECOVERY-scope cells")
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
