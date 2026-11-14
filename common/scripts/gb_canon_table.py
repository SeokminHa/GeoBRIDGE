#!/usr/bin/env python
# Canonical performance tables.
import argparse
import contextlib
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

BOOT_SEED = 20260928
KS = (20, 50, 100)
MET = ["AUROC", "AUPRC", "APLIFT", "P20", "P50", "P100"]      # contrasted metrics
COMPS = ["|Pearson|", "GENIE3", "GRNBoost2", "RegVelo |J|", "Velorama"]
REFS = ["REF -detection", "REF -pair_support", "REF +dynamics"]
SOTA = ["RegDiffusion", "DeepSEM", "PIDC", "FastTENET"]
SOTA_FILES = {"RegDiffusion": "regdiffusion_v1.npz", "DeepSEM": "deepsem_v1.npz", "PIDC": "pidc_v1.npz", "FastTENET": "fasttenet_v1.npz"}
SOTA_SCOPE = "COMMON_AVAILABLE_SOTA"                          # common mask over GB + the frozen comparators + every available SOTA method (COMMON_AVAILABLE itself is unchanged)
OUT_METHODS = COMPS + REFS                                    # RegVelo +J/-J stay in V (they define common) but are not reported
MIN_CLASS = 5                                                 # gb_final_benchmark.py:108 (>=5 pos and >=5 neg or TOO_FEW)
TOL = 1e-9
PIN_ENDPOINTS_SHA_PREFIX = "7a09e9c7"
PIN_FOREST_SHA_PREFIX = "96d55256"
MIN_N_FACTOR = 2                                              # P@K / R@K defined only for n >= 2K
COMP_FILES = ("pearson.npz", "genie3_v2.npz", "grnboost2_v2.npz", "regvelo_v2_full.npz", "velorama_v2_native.npz")
MIN_COMPLETE = 0.95                                           # label NA when fewer than 95% of draws are complete cases
FCLASS = {"F1": "regulon", "F2": "functional", "F3": "functional_occupancy"}
DEFAULT_SYSTEMS = ["PANC", "GATA1", "BONEMARROW", "DENTATEGYRUS", "GASTRULATION", "K562", "ZEBRAFISH"]


# Label by class, never average KO with occupancy
def endpoint_class(name, family):
    if family == "F1":
        return "regulon"
    if family == "F3":
        return "occupancy_conditioned"
    return "chronicKO" if name.endswith("|chronicKO") else "functional"


# metrics (weighted, tie-grouped)
def prep_scores(s):
    o = np.argsort(-s, kind="mergesort")
    ss = s[o]
    return o, np.flatnonzero(np.r_[True, ss[1:] != ss[:-1]])


# W (b, n) non-negative multiplicities in score-descending order
def weighted_metrics(W, ys, starts, ks=KS, n_rows=None):
    W = np.asarray(W, float)
    n_rows = W.shape[1] if n_rows is None else n_rows
    c = np.add.reduceat(W, starts, axis=1)
    h = np.add.reduceat(W * ys.astype(float), starts, axis=1)
    C, T = np.cumsum(c, 1), np.cumsum(h, 1)
    P, Nt = T[:, -1], C[:, -1]
    Neg = Nt - P
    ok = (P > 0) & (Neg > 0)
    nan = np.full(len(P), np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        ap = (h * np.divide(T, C, out=np.zeros_like(T), where=C > 0)).sum(1) / P
        ng = c - h
        auc = (h * (Neg[:, None] - np.cumsum(ng, 1) + 0.5 * ng)).sum(1) / (P * Neg)
        prev = P / Nt
        out = {"AUROC": np.where(ok, auc, nan), "AUPRC": np.where(ok, ap, nan), "APLIFT": np.where(ok, ap / prev, nan),
               "PREV": np.where(ok, prev, nan)}
        r = np.arange(len(P))
        for K in ks:
            j = (C >= K).argmax(1)
            hit = T[r, j] - h[r, j] + h[r, j] * (K - (C[r, j] - c[r, j])) / c[r, j]   # expected hits in top K, random tie-breaking
            okk = ok & (Nt >= K) & (n_rows >= MIN_N_FACTOR * K)
            out[f"P{K}"] = np.where(okk, hit / K, nan)
            out[f"R{K}"] = np.where(okk, hit / P, nan)
    return out


# (point dict of floats, boot dict of (B,) arrays) on the universe `rows`; gi = system-gene index of every position of `score`
def eval_rows(score, y, rows, gi, counts, B_chunk_elems=4_000_000):
    idx = np.flatnonzero(rows)
    sc = np.asarray(score, float)[idx]
    if not np.isfinite(sc).all():
        raise RuntimeError("eval_rows: non-finite score inside the universe")
    o, starts = prep_scores(sc)
    ys, gs = np.asarray(y, bool)[idx][o], np.asarray(gi)[idx][o]
    pt = {k: float(v[0]) for k, v in weighted_metrics(np.ones((1, len(idx))), ys, starts, n_rows=len(idx)).items()}
    step = max(1, B_chunk_elems // max(len(idx), 1))
    parts = [weighted_metrics(counts[a:a + step][:, gs], ys, starts, n_rows=len(idx)) for a in range(0, counts.shape[0], step)]
    return pt, {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


# one bootstrap draw set over the system gene axis: row b = multinomial(G, uniform(G)) = multiplicities of a size-G resample
def make_counts(G, B, seed):
    rng = np.random.default_rng(seed)
    out = np.empty((B, G), np.int16)
    step = max(1, int(2e7 // G))
    for a in range(0, B, step):
        out[a:a + step] = rng.multinomial(G, np.full(G, 1.0 / G), size=min(step, B - a))
    return out


# one endpoint
# rec: system, family, fclass, endpoint, source, obs_genes, ev (bool), y (bool, full length), V {method: (len,) scores}
def process_endpoint(rec, counts, gidx):
    V, ev, y = rec["V"], rec["ev"], rec["y"]
    gi = np.fromiter((gidx[g] for g in rec["obs_genes"]), int, len(rec["obs_genes"]))
    fin = {m: np.isfinite(v) for m, v in V.items()}
    six = ev.copy()
    for m in V:
        if m not in REFS and m not in SOTA:                        # the frozen COMMON_AVAILABLE mask never sees the SOTA methods
            six &= fin[m]
    base = dict(system=rec["system"], family=rec["fclass"], endpoint_class=rec["endpoint_class"], endpoint=rec["endpoint"], source=rec["source"])
    lbase = dict(system=rec["system"], endpoint=rec["endpoint"])
    cover = {m: (float(fin[m][ev].mean()) if m in V else np.nan) for m in ["GeoBRIDGE"] + OUT_METHODS + SOTA}
    meths = [m for m in OUT_METHODS if m in V]
    sota = [m for m in SOTA if m in V]
    rows_out, legacy, items, cache = [], [], [], {}

    def get(m, rows):
        k = (m, np.packbits(rows).tobytes())
        if k not in cache:
            cache[k] = eval_rows(V[m], y, rows, gi, counts)
        return cache[k]

    def erow(scope, m, paired, rows, pt=None):
        yy = y[rows]
        r = dict(**base, scope=scope, method=m, paired_with=paired, n_eval=int(ev.sum()), n=int(rows.sum()), n_pos=int(yy.sum()), coverage=cover[m])
        if pt is None:
            return dict(**r, status="TOO_FEW")
        return dict(**r, status="OK", prevalence=pt["PREV"], AUROC=pt["AUROC"], AUPRC=pt["AUPRC"], AP_lift=pt["APLIFT"],
                    **{f"P@{K}": pt[f"P{K}"] for K in KS}, **{f"R@{K}": pt[f"R{K}"] for K in KS})

    for m in COMPS:
        if m not in V:
            rows_out.append(dict(**base, scope="BOTH", method=m, paired_with="", status="MISSING"))
            legacy.append(dict(**lbase, scope="BOTH", comparator=m, status="MISSING"))
    scopes = [("COMMON_AVAILABLE", m, six) for m in meths] + [("PAIRWISE", m, ev & fin["GeoBRIDGE"] & fin[m]) for m in meths]
    if sota:                                                       # new scopes only when at least one SOTA artifact exists; rows never enter the legacy parity gate
        for m in SOTA:
            if m not in V:
                rows_out.append(dict(**base, scope="BOTH", method=m, paired_with="", status="MISSING"))
        six_s = six.copy()
        for m in sota:
            six_s &= fin[m]
        scopes += [(SOTA_SCOPE, m, six_s) for m in meths + sota] + [("PAIRWISE", m, ev & fin["GeoBRIDGE"] & fin[m]) for m in sota]
    common_gb = set()                                              # GB's COMMON_AVAILABLE[_SOTA] row is one per endpoint and scope, not one per method
    for scope, m, rows in scopes:
        yy = y[rows]
        emit_gb = scope == "PAIRWISE" or scope not in common_gb
        if scope != "PAIRWISE":
            common_gb.add(scope)
        paired = m if scope == "PAIRWISE" else ""
        new = scope == SOTA_SCOPE or m in SOTA                     # not in final_benchmark: excluded from the legacy rows
        if yy.sum() < MIN_CLASS or (~yy).sum() < MIN_CLASS:
            if emit_gb:
                rows_out.append(erow(scope, "GeoBRIDGE", paired, rows))
            rows_out.append(erow(scope, m, "GeoBRIDGE" if scope == "PAIRWISE" else "", rows))
            if not new:
                legacy.append(dict(**lbase, scope=scope, comparator=m, status="TOO_FEW", n=int(rows.sum()), n_pos=int(yy.sum())))
            continue
        pg, bg = get("GeoBRIDGE", rows)
        pm, bm = get(m, rows)
        if emit_gb:
            rows_out.append(erow(scope, "GeoBRIDGE", paired, rows, pg))
        rows_out.append(erow(scope, m, "GeoBRIDGE" if scope == "PAIRWISE" else "", rows, pm))
        if not new:
            legacy.append(dict(**lbase, scope=scope, comparator=m, status="OK", n=int(rows.sum()), n_pos=int(yy.sum()), AUROC_GB=pg["AUROC"],
                               AUROC_comp=pm["AUROC"], APlift_GB=pg["APLIFT"], APlift_comp=pm["APLIFT"]))
        dpt = {k: pg[k] - pm[k] for k in MET}
        dbs = {k: (bg[k] - bm[k] if np.isfinite(dpt[k]) else np.full(len(bg[k]), np.nan)) for k in MET}   # no draws where the point is undefined
        items.append((scope, m, dpt, dbs, {k: pg[k] for k in MET}, {k: pm[k] for k in MET}))
    return rows_out, legacy, items


# aggregation
def nmean(a):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    return float(a.mean()) if len(a) else np.nan


# (fclass, scope, method) -> TF-level running sums
class Agg:

    def __init__(self, B):
        self.B, self.g, self.unit, self.key, self.done = B, {}, {}, None, set()

    def add(self, fclass, source, items):
        key = (fclass, source)
        if key != self.key:
            self.flush()
            if key in self.done:
                raise RuntimeError(f"AGG_UNIT_NOT_CONTIGUOUS: {key} re-appeared after flush (endpoint order assumption violated)")
            self.done.add(key)
            self.key = key
        for it in items:
            self.unit.setdefault((it[0], it[1]), []).append(it)

    def flush(self):
        if self.key is None:
            return
        for (scope, m), its in self.unit.items():
            a = self.g.setdefault((self.key[0], scope, m), dict(
                n_ep=0, n_tf=0, pt={k: dict(d=[], gb=[], m=[]) for k in MET}, S={k: np.zeros(self.B) for k in MET}, N={k: np.zeros(self.B) for k in MET},
                ntf={k: 0 for k in MET}, rew={k: np.zeros(self.B, bool) for k in MET}, maxund={k: 0.0 for k in MET}))
            a["n_ep"] += len(its)
            a["n_tf"] += 1
            for k in MET:
                use = [i for i in its if np.isfinite(i[2][k])]
                if not use:
                    continue
                for slot, j in (("d", 2), ("gb", 4), ("m", 5)):
                    a["pt"][k][slot].append(float(np.mean([i[j][k] for i in use])))
                X = np.stack([i[3][k] for i in use])
                fin = np.isfinite(X)
                c = fin.sum(0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    tf = np.where(c > 0, np.nansum(X, 0) / c, np.nan)
                a["S"][k] += np.nan_to_num(tf)
                a["N"][k] += np.isfinite(tf)
                a["rew"][k] |= c < len(use)
                a["maxund"][k] = max(a["maxund"][k], float(1.0 - fin.mean(1).min()))
                a["ntf"][k] += 1
        self.unit, self.key = {}, None

    def rows(self, system):
        self.flush()
        out = []
        for (fclass, scope, m), a in sorted(self.g.items()):
            r = dict(system=system, family=fclass, scope=scope, method=m, n_tf=a["n_tf"], n_endpoints=a["n_ep"])
            for k in MET:
                use = a["N"][k] > 0
                sb = a["S"][k][use] / a["N"][k][use]
                r[f"GB_{k}"], r[f"method_{k}"], r[f"d_{k}"] = nmean(a["pt"][k]["gb"]), nmean(a["pt"][k]["m"]), nmean(a["pt"][k]["d"])
                r[f"lo_{k}"], r[f"hi_{k}"] = (np.percentile(sb, 2.5), np.percentile(sb, 97.5)) if len(sb) else (np.nan, np.nan)
                r[f"nboot_{k}"] = int(use.sum())
                r[f"ntf_{k}"] = a["ntf"][k]
                r[f"frac_draws_reweighted_{k}"] = float(a["rew"][k].mean())
                r[f"max_endpoint_undefined_rate_{k}"] = a["maxund"][k]
            enough = lambda k2: r[f"nboot_{k2}"] >= MIN_COMPLETE * self.B
            r["label"] = label(r["lo_AUROC"], r["hi_AUROC"], r["lo_AUPRC"], r["hi_AUPRC"]) if enough("AUROC") and enough("AUPRC") else "NA"
            r["label_lift"] = label(r["lo_AUROC"], r["hi_AUROC"], r["lo_APLIFT"], r["hi_APLIFT"]) if enough("AUROC") and enough("APLIFT") else "NA"
            out.append(r)
        return out


# Win = CI excludes 0 in GB's favour
def label(lo_a, hi_a, lo_p, hi_p):
    if not all(np.isfinite([lo_a, hi_a, lo_p, hi_p])):
        return "NA"
    wa, la, wp, lp = lo_a > 0, hi_a < 0, lo_p > 0, hi_p < 0
    if wa and wp: return "GB_WINS_BOTH"
    if wa and lp: return "GB_WINS_AUROC_LOSES_AUPRC"
    if wp and la: return "GB_WINS_AUPRC_LOSES_AUROC"
    if la and lp: return "METHOD_WINS_BOTH"
    if wa: return "GB_WINS_AUROC_ONLY"
    if wp: return "GB_WINS_AUPRC_ONLY"
    if la: return "METHOD_WINS_AUROC_ONLY"
    if lp: return "METHOD_WINS_AUPRC_ONLY"
    return "NULL"


def cross_system(S):
    out = []
    for (fam, scope, m), g in S.groupby(["family", "scope", "method"]):
        for met in ("AUROC", "AUPRC", "APLIFT"):
            d, lo, hi = g[f"d_{met}"], g[f"lo_{met}"], g[f"hi_{met}"]
            lab = "label_lift" if met == "APLIFT" else "label"
            out.append(dict(family=fam, scope=scope, method=m, metric=met, n_systems=int(d.notna().sum()), median_delta=d.median(),
                            min_delta=d.min(), max_delta=d.max(), n_delta_pos=int((d > 0).sum()), n_ci_pos=int((lo > 0).sum()),
                            n_ci_neg=int((hi < 0).sum()), systems_ci_pos=",".join(g.system[lo > 0]), systems_ci_neg=",".join(g.system[hi < 0]),
                            label_counts=";".join(f"{k}:{v}" for k, v in g[lab].value_counts().items()),
                            labels_by_system=";".join(f"{s}:{l}" for s, l in zip(g.system, g[lab]))))
    return pd.DataFrame(out)


# parity gate (fail-closed)
# Compare one recomputed row against final_benchmark/endpoints.tsv; raise on any mismatch
def check_legacy(row, old_idx, tol=TOL):
    key = (row["endpoint"], row["scope"], row["comparator"])
    o = old_idx.get(key)
    if o is None:
        raise RuntimeError(f"PARITY_GATE_FAILED: recomputed row {key} absent from endpoints.tsv")
    if o["status"] != row["status"]:
        raise RuntimeError(f"PARITY_GATE_FAILED: {key} status {row['status']} != {o['status']}")
    if row["status"] != "MISSING":
        for f in ("n", "n_pos"):
            if int(o[f]) != int(row[f]):
                raise RuntimeError(f"PARITY_GATE_FAILED: {key} {f} {row[f]} != {o[f]}")
    if row["status"] == "OK":
        for f in ("AUROC_GB", "AUROC_comp", "APlift_GB", "APlift_comp"):
            if f not in o:
                continue                                              # older table without AP columns
            if bool(np.isfinite(o[f])) != bool(np.isfinite(row[f])):
                raise RuntimeError(f"PARITY_GATE_FAILED: {key} {f} finiteness new={row[f]!r} old={o[f]!r}")
            if np.isfinite(row[f]) and abs(float(o[f]) - row[f]) > tol:
                raise RuntimeError(f"PARITY_GATE_FAILED: {key} {f} new={row[f]!r} old={o[f]!r} |diff|={abs(float(o[f]) - row[f]):.3e} > {tol}")
    return key


# endpoints.tsv rows restricted to the run's systems and to the methods this script reports
def old_index(path, systems):
    o = pd.read_csv(path, sep="\t")
    o = o[o.system.isin(systems) & o.comparator.isin(OUT_METHODS)]
    return {(r["endpoint"], r["scope"], r["comparator"]): r for r in o.to_dict("records")}


# groups whose composition is identical to gb_final_benchmark's: regulon everywhere; functional only where no GATA1 F3 was folded in
def unchanged_group(system, family):
    return family == "regulon" or (family == "functional" and system in ("PANC", "BONEMARROW", "K562"))


def forest_index(path, systems):
    f = pd.read_csv(path, sep="\t")
    f = f[f.system.isin(systems) & f.comparator.isin(OUT_METHODS)]
    return {(r["system"], r["fam"], r["scope"], r["comparator"]): r for r in f.to_dict("records") if unchanged_group(r["system"], r["fam"])}


# Second fail-closed gate: system-level point values vs final_benchmark/system_forest.tsv for groups of unchanged composition
def check_forest(sys_rows, fidx, system, tol=TOL):
    seen = set()
    for r in sys_rows:
        if r["scope"] == SOTA_SCOPE or r["method"] in SOTA:        # SOTA groups do not exist in final_benchmark (no parity reference)
            continue
        if not unchanged_group(r["system"], r["family"]):
            continue
        key = (r["system"], r["family"], r["scope"], r["method"])
        o = fidx.get(key)
        if o is None:
            raise RuntimeError(f"PARITY_GATE_FAILED(forest): recomputed group {key} absent from system_forest.tsv")
        seen.add(key)
        for a, b in (("n_tf", "n_tf"), ("n_endpoints", "n_endpoints")):
            if int(o[a]) != int(r[b]):
                raise RuntimeError(f"PARITY_GATE_FAILED(forest): {key} {a} new={r[b]} old={o[a]}")
        for a, b in (("AUROC_GB", "GB_AUROC"), ("AUROC_comp", "method_AUROC"), ("delta", "d_AUROC")):
            if bool(np.isfinite(o[a])) != bool(np.isfinite(r[b])) or (np.isfinite(r[b]) and abs(float(o[a]) - r[b]) > tol):
                raise RuntimeError(f"PARITY_GATE_FAILED(forest): {key} {a} new={r[b]!r} old={o[a]!r}")
    miss = [k for k in fidx if k[0] == system and k not in seen]
    if miss:
        raise RuntimeError(f"PARITY_GATE_FAILED(forest): {len(miss)} system_forest groups not recomputed for {system}, e.g. {miss[:3]}")
    return len(seen)


# K562 truth adapter
GB_ROOT = __import__("os").environ["GEOBRIDGE_ROOT"]
K562_TRUTH = {                                               # kind -> (state dir, required receipt variant, adapter output dir name, human label)
    "v4": ("k562_truth_v4_emp_allnull", "PRIMARY_AMBIGUOUS_AS_NULL", "canon_K562_v4", "v4_calibrated_emp_allnull"),
    "v4_emp": ("k562_truth_v4_emp", "SENSITIVITY_AMBIGUOUS_EXCLUDED", "canon_K562_v4_emp", "v4_calibrated_emp_ambiguous_excluded"),
}
V3_STATE = "k562_truth_v3_2026_09_21/k562_state.npz"
K562_LEGACY_PG = "results/canon_K562/per_gene.npz"
K562_CODES = {"UNKNOWN": 0, "TESTED_NULL": 1, "TESTED_UP": 2, "TESTED_DOWN": 3, "LOW_INFORMATION_TESTED_NULL": 4}
K562_ARMS = (("primary", (1,)), ("sensitivity", (1, 4)))    # gb_layers.py:453-454
K562_STEM = "dc1_min|U|k10|{tf}|any_change|tested_sig_vs_tested_null_{arm}"     # the only keys gb_cop_alignment.py:356-361 reads (|pos, |panel_universe)


# Fail closed unless the truth directory is a calibrated, canonical, hash-consistent artifact of the requested variant
def check_k562_truth_receipt(kind, root=GB_ROOT):
    d, variant, _, _ = K562_TRUTH[kind]
    d = Path(root) / "results" / d
    for f in ("receipt.json", "k562_state.npz", "READY_FLAG"):
        if not (d / f).exists():
            raise SystemExit(f"K562_TRUTH_NOT_CALIBRATED: {d / f} missing")
    r = json.loads((d / "receipt.json").read_text())
    bad = []
    if r.get("label") != "CANONICAL":
        bad.append(f"label={r.get('label')!r}")
    if r.get("variant") != variant:
        bad.append(f"variant={r.get('variant')!r} != {variant!r}")
    if not r.get("per_pair_sha256") or r.get("per_pair_sha256") != r.get("per_pair_sha256_in_receipt"):
        bad.append("per_pair_sha256 != per_pair_sha256_in_receipt")
    sh = sha256(d / "k562_state.npz")
    if r.get("out_state_sha256") != sh:
        bad.append(f"out_state_sha256 {r.get('out_state_sha256')} != actual {sh}")
    if r.get("n_gene_unmapped") != 0:
        bad.append(f"n_gene_unmapped={r.get('n_gene_unmapped')}")
    if bad:
        raise SystemExit("K562_TRUTH_NOT_CALIBRATED: " + "; ".join(bad))
    return dict(kind=kind, dir=str(d), variant=variant, receipt_sha256=sha256(d / "receipt.json"), state_path=str(d / "k562_state.npz"), state_sha256=sh, per_pair_sha256=r["per_pair_sha256"],
                source_receipt_sha256=r.get("source_receipt_sha256"), state_totals=r.get("state_totals"), tfs_missing_from_v4=r.get("tfs_missing_from_v4"))


# TFs the legacy per_gene.npz carries = the TFs scored by the canon K562 run (gb_layers roster); the adapter never emits a panel for a TF outside it
def legacy_tfs(Zl):
    return {k.split("|")[3].upper() for k in Zl.files if k.startswith("dc1_min|U|k10|")}


# Mirror of gb_layers.py:426-475 (the K562 panel builder that produced canon_K562/per_gene.npz) on the gene axis `genes`.
def k562_panels(state_path, genes, tf_whitelist=None):
    Z = np.load(state_path, allow_pickle=True)
    cmap = json.loads(str(Z["code_map"])) if isinstance(Z["code_map"].item(), str) else dict(Z["code_map"].item())
    if cmap != K562_CODES:
        raise SystemExit(f"K562_TRUTH_CODE_MAP_MISMATCH: {cmap} != {K562_CODES}")
    sgen, stfs, st = np.char.upper(Z["genes"].astype(str)), Z["tfs"].astype(str), Z["state"]
    if len(set(sgen.tolist())) != sgen.size:
        raise SystemExit("K562_STATE_SYMBOL_COLLISION")
    gu = np.char.upper(np.asarray(genes).astype(str))
    gp = {g: i for i, g in enumerate(gu)}
    col = {g: i for i, g in enumerate(sgen)}
    have = np.array([g in col for g in gu])
    cidx = np.array([col.get(g, 0) for g in gu])
    out, skipped = {}, []
    for i, tf in enumerate(stfs):
        T = tf.upper()
        if T not in gp:
            skipped.append((T, "ABSENT_FROM_UNIVERSE"))
            continue
        if tf_whitelist is not None and T not in tf_whitelist:
            skipped.append((T, "NOT_IN_CANON_K562_SCORED_ROSTER"))
            continue
        row = st[i][cidx]
        pos0 = have & ((row == 2) | (row == 3))
        for arm, negstates in K562_ARMS:
            pos = pos0.copy()
            uni = pos | (have & np.isin(row, negstates))
            pos[gp[T]] = False
            uni[gp[T]] = False
            if pos.sum() >= 10 and (uni & ~pos).sum() >= 10:
                stem = K562_STEM.format(tf=T, arm=arm)
                out[f"{stem}|pos"], out[f"{stem}|panel_universe"] = pos, uni
            else:
                skipped.append((f"{T}|{arm}", "LOW_POSITIVES_OR_NEGATIVES"))
    return out, skipped


# -> path of canon_K562_v4[_emp]/per_gene.npz (written once, atomically; an existing file must be byte-content-identical)
def build_k562_v4_pergene(kind, root=GB_ROOT):
    rec = check_k562_truth_receipt(kind, root)
    out_dir = Path(root) / "results" / K562_TRUTH[kind][2]
    legacy = Path(root) / K562_LEGACY_PG
    Zl = np.load(legacy, allow_pickle=True)
    genes = Zl["genes"]
    panels, skipped = k562_panels(rec["state_path"], genes, legacy_tfs(Zl))
    if not panels:
        raise SystemExit("K562_ADAPTER_EMPTY: no TF passed the >=10 pos / >=10 neg rule")
    pg = out_dir / "per_gene.npz"
    if pg.exists():
        old = np.load(pg, allow_pickle=True)
        if set(old.files) - {"genes"} != set(panels) or not all(np.array_equal(old[k], v) for k, v in panels.items()) or list(old["genes"]) != list(genes):
            raise SystemExit(f"K562_ADAPTER_OUTPUT_DIFFERS: {pg} exists with different content; refusing to overwrite (use a new dir)")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        tmp = out_dir / "per_gene.tmp.npz"
        np.savez_compressed(tmp, genes=np.asarray(genes, dtype=object), **panels)
        tmp.replace(pg)
        (out_dir / "receipt.json").write_text(json.dumps(dict(
            purpose="K562 functional label adapter for gb_canon_table (mirror of gb_layers.py k562 panel builder; keys dc1_min|U|k10|TF|any_change|tested_sig_vs_tested_null_{primary,sensitivity}|{pos,panel_universe} only)",
            truth=rec, gene_axis_source=str(legacy), gene_axis_sha256=sha256(legacy), n_genes=int(len(genes)), n_keys=len(panels), skipped=skipped,
            builder_script=str(Path(__file__).resolve()), builder_script_sha256=sha256(__file__), output_sha256=sha256(pg)), indent=1, default=str))
    return pg, rec, skipped


# Regression proof of the adapter: v3 state -> adapter == legacy per_gene.npz pos/panel_universe, key-for-key, both arms
def k562_mirror_check(root=GB_ROOT):
    legacy = Path(root) / K562_LEGACY_PG
    Zl = np.load(legacy, allow_pickle=True)
    panels, _ = k562_panels(Path(root) / "results" / V3_STATE, Zl["genes"], legacy_tfs(Zl))
    lk = {k for k in Zl.files if k.startswith("dc1_min|U|k10|") and (k.endswith("|pos") or k.endswith("|panel_universe"))}
    only_legacy, only_new = sorted(lk - set(panels)), sorted(set(panels) - lk)
    diff = [k for k in lk & set(panels) if not np.array_equal(np.asarray(Zl[k], bool), panels[k])]
    print(f"[k562-mirror] keys legacy={len(lk)} adapter(v3)={len(panels)} only_legacy={len(only_legacy)} only_adapter={len(only_new)} differing={len(diff)}")
    if only_legacy or only_new or diff:
        raise SystemExit(f"K562_MIRROR_FAILED: only_legacy={only_legacy[:3]} only_adapter={only_new[:3]} differing={diff[:3]}")
    print("[k562-mirror] OK: adapter(v3) == legacy per_gene labels")
    return len(lk)


# gb_cop_alignment hard-codes canon_K562/per_gene.npz
@contextlib.contextmanager
def redirect_k562_pergene(CA, new_path, root=GB_ROOT):
    target = os.path.realpath(str(Path(root) / K562_LEGACY_PG))
    hits = []
    real = CA.np

    class Proxy:
        def __getattr__(self, k):
            return getattr(real, k)

        def load(self, f, *a, **kw):
            if isinstance(f, (str, os.PathLike)) and os.path.realpath(str(f)) == target:
                hits.append(str(f))
                f = str(new_path)
            return real.load(f, *a, **kw)
    CA.np = Proxy()
    try:
        yield hits
    finally:
        CA.np = real
    if len(hits) != 1:
        raise RuntimeError(f"K562 per_gene redirect fired {len(hits)} times (expected exactly 1)")


# Endpoint-level fail-closed check of the K562 labels against the calibrated state (primary arm): every evaluated gene is a tested gene (state 1/2/3)
def k562_truth_gate(rec, state_path, cache={}):
    if state_path not in cache:
        Z = np.load(state_path, allow_pickle=True)
        cache[state_path] = (dict(zip(np.char.upper(Z["genes"].astype(str)), range(len(Z["genes"])))), dict(zip(np.char.upper(Z["tfs"].astype(str)), range(len(Z["tfs"])))), Z["state"])
    gcol, trow, st = cache[state_path]
    tf = rec["source"].upper()
    if tf not in trow:
        raise RuntimeError(f"K562 truth gate: {tf} absent from the calibrated state")
    g = np.char.upper(rec["obs_genes"].astype(str))
    ev, y = np.asarray(rec["ev"], bool), np.asarray(rec["y"], bool)
    ix = np.array([gcol.get(x, -1) for x in g[ev]])
    if (ix < 0).any():
        raise RuntimeError(f"K562 truth gate: {rec['endpoint']} evaluates genes absent from the state")
    sv = st[trow[tf]][ix]
    if not np.isin(sv, (1, 2, 3)).all() or not np.array_equal(y[ev], np.isin(sv, (2, 3))) or y[~ev].any():
        raise RuntimeError(f"K562 truth gate FAILED for {rec['endpoint']}: labels differ from the calibrated state")
    return int(ev.sum())


# real-data plumbing
def heavy():
    here = str(Path(__file__).resolve().parent)
    for p in (here, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import gb_final_benchmark as FB          # imports gb_interim_fig1, gb_killtests (RAWH), gb_primary, gb_real_uniform
    import gb_interim_fig1 as IF
    import gb_primary as GP
    import gb_real_uniform as RU
    for m in COMPS + REFS:
        assert m in FB.METHODS + FB.REFS, f"method name {m!r} not in gb_final_benchmark"
    assert list(FB.REFS) == REFS, "REF names drifted from gb_final_benchmark"
    return FB, IF, GP, RU


def loaders(system, IF, RU):
    d = RU.system_dir(system)
    return {"|Pearson|": IF.loader(d / "pearson.npz", "score"), "GENIE3": IF.loader(d / "genie3_v2.npz", "score"),
            "GRNBoost2": IF.loader(d / "grnboost2_v2.npz", "score"), "RegVelo": IF.loader(d / "regvelo_v2_full.npz", "score_signed"),
            "Velorama": IF.loader(d / "velorama_v2_native.npz", "score"),
            **{m: IF.loader(d / f, "score") for m, f in SOTA_FILES.items()}}          # None where the artifact is absent (optional)


def _full_y(E):
    y = np.zeros(len(E.eval_mask), bool)
    y[E.eval_mask] = np.asarray(E.y, bool)
    return y


# -> (sorted union of endpoint gene names, generator of endpoint records)
def load_system(system, H, k562=None):
    FB, IF, GP, RU = H
    L = loaders(system, IF, RU)
    OC = FB.obs_cov(system)
    print(f"[canon] {system}: available {[m for m, v in L.items() if v is not None]}", flush=True)
    if system == "K562" and k562 is not None:                    # calibrated truth: redirect the label file for the duration of the trace only
        import gb_cop_alignment as CA
        with redirect_k562_pergene(CA, k562["per_gene"]):
            trace = RU.trace_endpoints(systems=[system])
    else:
        trace = RU.trace_endpoints(systems=[system])
    union = sorted(set().union(*[set(map(str, t[4]["U"].obs_genes)) for t in trace]))

    def gen():
        for name, family, _s, src, eps, n, npos in trace:
            src = src.upper()
            E = eps["U"]
            if system == "K562" and k562 is not None:
                k562["gate_checked"] = k562.get("gate_checked", 0) + 1
                k562_truth_gate(dict(source=src, obs_genes=np.asarray(E.obs_genes), ev=E.eval_mask, y=_full_y(E), endpoint=name), k562["state_path"])
            V = {"GeoBRIDGE": GP.zcons(E.RAWH)}
            for m, l in L.items():
                v = IF.row(l, src, E.obs_genes)
                if v is None:
                    continue
                if m == "RegVelo":
                    V["RegVelo |J|"], V["RegVelo +J"], V["RegVelo -J"] = np.abs(v), v, -v
                else:
                    V[m] = v
            for r in REFS:
                V[r] = np.array([OC[r].get(g, np.nan) for g in E.obs_genes], float)
            ev = E.eval_mask.copy()
            y = np.zeros(len(ev), bool)
            y[ev] = np.asarray(E.y, bool)
            yield dict(system=system, family=family, fclass=FCLASS[family], endpoint_class=endpoint_class(name, family), endpoint=name,
                       source=src, obs_genes=np.asarray(E.obs_genes).astype(str), ev=ev, y=y, V=V)
    return union, gen()


# Core loop, independent of the heavy loaders (selftest drives it with synthetic records)
def run_system(system, recs, union, B, sys_idx, old_idx=None, forest_idx=None, log_every=10):
    gidx = {g: i for i, g in enumerate(union)}
    counts = make_counts(len(union), B, [BOOT_SEED, sys_idx])
    agg, ep_rows, seen, t0, n = Agg(B), [], set(), time.time(), 0
    for rec in recs:
        rows, legacy, items = process_endpoint(rec, counts, gidx)
        if old_idx is not None:
            seen |= {check_legacy(l, old_idx) for l in legacy}
        ep_rows += rows
        agg.add(rec["fclass"], rec["source"], items)
        n += 1
        if n % log_every == 0:
            print(f"[canon] {system}: {n} endpoints, {time.time() - t0:.0f}s", flush=True)
    if old_idx is not None:
        miss = [k for k in old_idx if k not in seen and old_idx[k]["system"] == system]
        if miss:
            raise RuntimeError(f"PARITY_GATE_FAILED: {len(miss)} rows of endpoints.tsv not recomputed for {system}, e.g. {miss[:3]}")
    sr = agg.rows(system)
    nforest = check_forest(sr, forest_idx, system) if forest_idx is not None else 0
    return ep_rows, sr, len(seen), nforest


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def pin_check(path, prefix):
    d = sha256(path)
    if __import__("os").environ.get("GB_SKIP_PREFLIGHT") == "1":      # opt-in (public release)
        print("preflight skipped (public release)")
        return d
    if not d.startswith(prefix):
        raise SystemExit(f"PARITY_REFERENCE_SHA_MISMATCH: {path} sha256 {d[:12]}.. does not start with pinned {prefix}")
    return d


# Every input the run consumes must exist (raise otherwise) and is hashed at start
def collect_inputs(systems, RU, ref, forest):
    req = [ref, forest, RU.FIG1_ELIGIBILITY, RU.FIGURE_CONTRACT]
    for s in systems:
        gb = [Path(p) for p in RU.cache_paths(s) if Path(p).exists()]
        if not gb:
            raise SystemExit(f"MISSING_INPUT: no GB score cache for {s}")
        req += gb + [RU.system_dir(s) / f for f in COMP_FILES]
    miss = [str(p) for p in req if not Path(p).exists()]
    if miss:
        raise SystemExit(f"MISSING_INPUT: {miss}")
    req += [p for s in systems for f in SOTA_FILES.values() for p in (RU.system_dir(s) / f,) if p.exists()]      # optional SOTA artifacts: hashed when present, never required
    return {str(p): sha256(p) for p in req}


# sha256 of every loaded module whose file lives under one of `dirs` (transitive imports of scratch/ included)
def module_hashes(dirs):
    out = {}
    for m in list(sys.modules.values()):
        f = getattr(m, "__file__", None)
        if f and Path(f).exists() and any(Path(f).resolve().is_relative_to(Path(d).resolve()) for d in dirs):
            out[str(Path(f).resolve())] = sha256(f)
    return out


# tables -> tmp names; receipt (with the tmp files' sha256) -> tmp; only then rename, receipt last
def write_atomic(out, tables, receipt):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    tmps, rt = {}, out / "receipt.json.tmp"
    try:
        for nm, df in tables.items():
            tmps[nm] = out / f"{nm}.tsv.tmp"
            df.to_csv(tmps[nm], sep="\t", index=False)
        receipt = dict(receipt, output_sha256={f"{nm}.tsv": sha256(p) for nm, p in tmps.items()})
        rt.write_text(json.dumps(receipt, indent=1, default=str))
    except BaseException:
        for p in list(tmps.values()) + [rt]:
            p.unlink(missing_ok=True)
        raise
    for nm, p in tmps.items():
        p.replace(out / f"{nm}.tsv")
    rt.replace(out / "receipt.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", default=",".join(DEFAULT_SYSTEMS))
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--smalln", action="store_true", help="PANC only, B=50")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument("--parity-ref", default=None)
    ap.add_argument("--k562-truth", choices=["v4", "v4_emp", "legacy"], default="v4", help="K562 functional truth: v4 = calibrated empirical null, ambiguous->null (DEFAULT); v4_emp = ambiguous excluded (sensitivity); legacy = uncalibrated v3")
    ap.add_argument("--k562-mirror-check", action="store_true", help="prove the adapter reproduces canon_K562/per_gene.npz from the v3 state, then exit")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if a.k562_mirror_check:
        k562_mirror_check()
        return
    systems = ["PANC"] if a.smalln else [s.strip().upper() for s in a.systems.split(",") if s.strip()]
    B = 50 if a.smalln else a.B
    H = heavy()
    FB, IF, GP, RU = H
    for s in systems:
        if s not in FB.SEED:
            raise SystemExit(f"unknown system {s}")
    full = B == 2000 and sorted(systems) == sorted(DEFAULT_SYSTEMS)
    out = Path(a.out) if a.out else RU.OUT / ("canon_table" if full else "canon_table_smalln" if a.smalln else "canon_table_partial")
    ref = Path(a.parity_ref) if a.parity_ref else RU.OUT / "final_benchmark" / "endpoints.tsv"
    forest = ref.parent / "system_forest.tsv"
    inputs = collect_inputs(systems, RU, ref, forest)                    # raises on any missing input
    pin_check(ref, PIN_ENDPOINTS_SHA_PREFIX)
    pin_check(forest, PIN_FOREST_SHA_PREFIX)
    old, fidx = old_index(ref, systems), forest_index(forest, systems)
    k562 = None
    if "K562" in systems and a.k562_truth != "legacy":           # calibrated truth: adapter labels; the legacy (v3) K562 parity gates are not applicable to a deliberately changed truth
        pg, trec, skipped = build_k562_v4_pergene(a.k562_truth)
        k562 = dict(per_gene=str(pg), state_path=trec["state_path"], truth=trec, per_gene_sha256=sha256(pg), skipped=skipped)
        inputs.update({str(pg): k562["per_gene_sha256"], trec["state_path"]: trec["state_sha256"], str(Path(trec["dir"]) / "receipt.json"): trec["receipt_sha256"], str(Path(GB_ROOT) / K562_LEGACY_PG): sha256(Path(GB_ROOT) / K562_LEGACY_PG)})
        print(f"[canon] K562 truth = {a.k562_truth} ({trec['variant']}); per_gene {pg}; legacy K562 parity gates replaced by the truth-consistency gate", flush=True)
        old = {k: v for k, v in old.items() if v["system"] != "K562"}
        fidx = {k: v for k, v in fidx.items() if k[0] != "K562"}
    print(f"[canon] systems={systems} B={B} out={out} parity_ref={ref} ({len(old)} endpoint rows, {len(fidx)} forest groups)", flush=True)
    EP, SY, npar, nfor, t0 = [], [], {}, {}, time.time()
    for s in systems:
        kk = k562 if s == "K562" else None
        union, gen = load_system(s, H, kk)
        er, sr, npar[s], nfor[s] = run_system(s, gen, union, B, FB.SEED[s], None if kk else old, None if kk else fidx)      # calibrated K562: legacy parity replaced by the truth gate
        EP += er
        SY += sr
        print(f"[canon] {s}: parity OK on {npar[s]} endpoint rows + {nfor[s]} forest groups; {len(er)} endpoint rows; {len(sr)} system rows", flush=True)
    E, S = pd.DataFrame(EP), pd.DataFrame(SY)          # nothing is written before every gate has passed
    tv = "K562:legacy_v3_uncalibrated" if a.k562_truth == "legacy" else f"K562:{K562_TRUTH[a.k562_truth][3]}"
    for df in (E, S):
        df["truth_version"] = np.where(df["system"] == "K562", tv, "")      # the only added column; non-K562 rows are otherwise byte-identical to canon_table_sota
    X = cross_system(S)
    here = Path(__file__).resolve().parent
    receipt = dict(script=str(Path(__file__).resolve()), script_sha256=sha256(__file__), argv=sys.argv, B=B, seed=BOOT_SEED,
                   seed_per_system={s: [BOOT_SEED, FB.SEED[s]] for s in systems}, systems=systems, parity_endpoint_rows_checked=npar,
                   parity_forest_groups_checked=nfor, parity_tol=TOL, pinned_reference_sha256_prefix=dict(endpoints=PIN_ENDPOINTS_SHA_PREFIX, system_forest=PIN_FOREST_SHA_PREFIX),
                   input_sha256=inputs, scratch_module_sha256=module_hashes({here, RU.SCRATCH}),
                   predeclared_rules=dict(PK_RK_defined_only_if_n_ge=f"{MIN_N_FACTOR}K (point and all draws NaN otherwise)", K=list(KS),
                                          draw_reweighting="endpoint undefined in a draw (no positives/negatives in the resample) is dropped from that draw only; TF mean reweighted over its remaining endpoints; TF undefined in a draw dropped from that draw's system mean; unusable draw only if no TF defined",
                                          reweighting_bound_per_endpoint=f"exp(-{MIN_CLASS})={np.exp(-MIN_CLASS):.5f}", point_NaN_implies_draws_NaN=True,
                                          label_NA_if_nboot_below=MIN_COMPLETE,
                                          min_class_per_endpoint=MIN_CLASS, ci="2.5/97.5 percentile, pointwise, conditional on the fixed TF set (genes resampled only)"),
                   k562_truth=(dict(kind=a.k562_truth, label=K562_TRUTH[a.k562_truth][3], receipt=k562["truth"], adapter_per_gene=k562["per_gene"], adapter_per_gene_sha256=k562["per_gene_sha256"],
                                   adapter_skipped=k562["skipped"], truth_gate_endpoints_checked=k562.get("gate_checked", 0),
                                   legacy_parity="NOT APPLICABLE (truth changed by design); replaced by the endpoint-level truth-consistency gate (y == state in {UP,DOWN}; evaluated genes all tested)")
                               if k562 else dict(kind=a.k562_truth, label="legacy_v3_uncalibrated" if a.k562_truth == "legacy" else "K562 not in this run", legacy_parity="applied" if a.k562_truth == "legacy" else "n/a")),
                   numpy=np.__version__, pandas=pd.__version__, elapsed_s=time.time() - t0)
    write_atomic(out, {"endpoints": E, "system_contrasts": S, "cross_system": X}, receipt)
    pd.set_option("display.width", 250)
    print(S[["system", "family", "scope", "method", "n_tf", "GB_AUROC", "d_AUROC", "lo_AUROC", "hi_AUROC", "d_APLIFT", "lo_APLIFT", "hi_APLIFT", "label"]]
          [S.scope == "PAIRWISE"].round(3).to_string(index=False))
    print(f"[canon] wrote {out}")


# synthetic selftest
def _expanded(counts_row, s, y):
    ix = np.repeat(np.arange(len(s)), counts_row)
    return s[ix], y[ix]


# independent expected P@K / R@K under uniform random tie-breaking, per item
def _pk_bruteforce(s, y, K):
    before = np.array([(s > v).sum() for v in s])
    gs = np.array([(s == v).sum() for v in s])
    p = np.clip((K - before) / gs, 0, 1)
    hits = float((y * p).sum())
    return hits / K, hits / y.sum()


def selftest():
    from sklearn.metrics import average_precision_score as APS, roc_auc_score
    rng = np.random.default_rng(1)
    # 1. weighted metrics == metrics on the expanded resample (ties included), P@K / R@K ==
    # independent brute force
    n = 300
    y = rng.random(n) < 0.2
    s = np.round(rng.normal(size=n) + 1.0 * y, 1)                     # heavy ties
    cnt = rng.multinomial(n, np.full(n, 1 / n), size=6)
    o, st = prep_scores(s)
    Wm = weighted_metrics(cnt[:, o], y[o], st)
    for b in range(6):
        se, ye = _expanded(cnt[b], s, y)
        assert abs(Wm["AUROC"][b] - roc_auc_score(ye, se)) < 1e-12
        assert abs(Wm["AUPRC"][b] - APS(ye, se)) < 1e-12
        assert abs(Wm["APLIFT"][b] - APS(ye, se) / ye.mean()) < 1e-12 and abs(Wm["PREV"][b] - ye.mean()) < 1e-15
        for K in KS:
            pk, rk = _pk_bruteforce(se, ye, K)
            assert abs(Wm[f"P{K}"][b] - pk) < 1e-12 and abs(Wm[f"R{K}"][b] - rk) < 1e-12, (b, K)
    p1 = weighted_metrics(np.ones((1, n)), y[o], st)                   # point == sklearn
    assert abs(p1["AUROC"][0] - roc_auc_score(y, s)) < 1e-12 and abs(p1["AUPRC"][0] - APS(y, s)) < 1e-12
    print("selftest 1 OK: weighted == expanded resample (AUROC, AP, lift, P@K, R@K; ties)")

    # synthetic system: 3 TFs, gene axis 400, per-TF obs_genes differ (source gene dropped), GB
    # informative, comps random
    G, B = 400, 200
    names = np.array([f"g{i}" for i in range(G)])

    def mk(src, k, fclass="F1"):
        keep = np.arange(G) != k
        yv = np.zeros(G, bool); yv[rng.choice(np.flatnonzero(keep), 60, replace=False)] = True
        yv &= keep
        sig = yv * 1.5 + rng.normal(size=G)
        V = {"GeoBRIDGE": sig[keep], "|Pearson|": rng.normal(size=G)[keep], "GENIE3": np.round(rng.normal(size=G), 1)[keep],
             "REF -detection": rng.normal(size=G)[keep]}
        V["GENIE3"][::7] = np.nan
        return dict(system="SYN", family=fclass, fclass=FCLASS[fclass], endpoint_class=endpoint_class(f"{fclass}|SYN|{src}", fclass),
                    endpoint=f"{fclass}|SYN|{src}", source=src, obs_genes=names[keep], ev=np.ones(keep.sum(), bool), y=yv[keep], V=V)
    recs = [mk("A", 3), mk("B", 9), mk("C", 27), mk("D", 45)]
    union = sorted(set().union(*[set(r["obs_genes"]) for r in recs]))
    gidx = {g: i for i, g in enumerate(union)}
    counts = make_counts(len(union), B, [BOOT_SEED, 3])
    assert counts.shape == (B, len(union)) and (counts.sum(1) == len(union)).all()
    assert np.array_equal(counts, make_counts(len(union), B, [BOOT_SEED, 3])) and not np.array_equal(counts, make_counts(len(union), B, [BOOT_SEED, 4]))

    # 2. shared draws: a gene present in two endpoints carries the same resample multiplicity in
    # both
    g1, g2 = recs[0]["obs_genes"], recs[1]["obs_genes"]
    common = sorted(set(g1) & set(g2))
    i1 = np.array([np.flatnonzero(g1 == g)[0] for g in common]); i2 = np.array([np.flatnonzero(g2 == g)[0] for g in common])
    gi1 = np.array([gidx[g] for g in g1]); gi2 = np.array([gidx[g] for g in g2])
    assert np.array_equal(counts[:, gi1[i1]], counts[:, gi2[i2]]) and len(common) > 300
    print("selftest 2 OK: one shared resample matrix across endpoints")

    # 3. delta AP-lift uses per-resample prevalence: recompute a draw by brute force on the expanded
    # resample
    r0 = recs[0]
    rows, legacy, items = process_endpoint(r0, counts, gidx)
    it = [i for i in items if i[0] == "PAIRWISE" and i[1] == "GENIE3"][0]
    rowmask = np.isfinite(r0["V"]["GENIE3"])
    gi0 = np.array([gidx[g] for g in r0["obs_genes"]])[rowmask]
    diff_from_global = 0
    for b in (0, 7, 33):
        ix = np.repeat(np.arange(rowmask.sum()), counts[b][gi0])
        yy = r0["y"][rowmask][ix]
        lg = APS(yy, r0["V"]["GeoBRIDGE"][rowmask][ix]) / yy.mean()
        lm = APS(yy, r0["V"]["GENIE3"][rowmask][ix]) / yy.mean()
        assert abs(it[3]["APLIFT"][b] - (lg - lm)) < 1e-10
        wrong = (APS(yy, r0["V"]["GeoBRIDGE"][rowmask][ix]) - APS(yy, r0["V"]["GENIE3"][rowmask][ix])) / r0["y"][rowmask].mean()
        diff_from_global += abs(wrong - (lg - lm)) > 1e-6
    assert diff_from_global >= 2, "test would not detect division by the original prevalence"
    print("selftest 3 OK: dAP-lift = per-resample-prevalence lift difference (differs from original-prevalence division)")

    # 4. parity gate raises on a perturbed value, passes on the true one, flags missing/extra rows
    legacy_all = [l for r in recs for l in process_endpoint(r, counts, gidx)[1]]
    old = {(l["endpoint"], l["scope"], l["comparator"]): dict(l, system="SYN") for l in legacy_all}
    for l in legacy_all:
        check_legacy(l, old)
    run_system("SYN", recs, union, 20, 3, old)                          # complete + exact reference passes
    bad = dict(next(l for l in legacy if l["status"] == "OK"))
    bad["AUROC_GB"] += 1e-6
    try:
        check_legacy(bad, old)
        raise AssertionError("parity gate did not raise")
    except RuntimeError as e:
        assert "PARITY_GATE_FAILED" in str(e)
    ghost = dict(old[next(iter(old))]); ghost["endpoint"] = "ghost"
    try:
        run_system("SYN", recs, union, 20, 3, {**old, ("ghost", "PAIRWISE", "GENIE3"): dict(ghost, comparator="GENIE3", scope="PAIRWISE")})
        raise AssertionError("missing-row gate did not raise")
    except RuntimeError as e:
        assert "PARITY_GATE_FAILED" in str(e)
    print("selftest 4 OK: parity gate raises on perturbed value and on unrecomputed/absent rows")

    # 5. end to end: GB informative vs random comps -> GB_WINS_BOTH vs |Pearson|; REF random ->
    # NULL-ish
    er, sr, _, _ = run_system("SYN", recs, union, B, 3)
    S = pd.DataFrame(sr)
    r = S[(S.scope == "PAIRWISE") & (S.method == "|Pearson|")].iloc[0]
    assert r.n_tf == 4 and r.n_endpoints == 4 and r.ntf_P20 == 4 and r.nboot_AUROC == B and r.lo_AUROC > 0 and r.lo_AUPRC > 0 and r.label == "GB_WINS_BOTH", r
    assert abs(r.d_AUROC - (r.GB_AUROC - r.method_AUROC)) < 1e-12
    E = pd.DataFrame(er)
    assert set(E.scope) == {"PAIRWISE", "COMMON_AVAILABLE", "BOTH"} and (E.status == "MISSING").sum() == 4 * 3     # RegVelo, Velorama, GRNBoost2 x 3 endpoints
    assert (E[(E.status == "OK") & (E.scope == "PAIRWISE")].groupby("endpoint").size() == 2 * 3).all()      # GB + method row per pair, 3 methods
    assert (E[(E.status == "OK") & (E.scope == "COMMON_AVAILABLE") & (E.method == "GeoBRIDGE")].groupby("endpoint").size() == 1).all()
    assert len(cross_system(S)) == 3 * S.groupby(["family", "scope", "method"]).ngroups
    assert label(.1, .2, .1, .2) == "GB_WINS_BOTH" and label(.1, .2, -.2, -.1) == "GB_WINS_AUROC_LOSES_AUPRC" and label(-.2, -.1, .1, .2) == "GB_WINS_AUPRC_LOSES_AUROC"
    assert label(-.2, -.1, -.2, -.1) == "METHOD_WINS_BOTH" and label(-.1, .1, -.1, .1) == "NULL" and label(.1, .2, -.1, .1) == "GB_WINS_AUROC_ONLY"
    # (2a/2b) P@K rule: n < 2K -> point and draws NaN
    small = mk("S", 60)
    keep = np.arange(150)
    small = dict(small, endpoint="F1|SYN|S", obs_genes=small["obs_genes"][keep], ev=small["ev"][keep], y=small["y"][keep],
                 V={m: v[keep] for m, v in small["V"].items()})
    small["y"][:30] = True                                             # enough positives
    _, _, it_s = process_endpoint(small, counts, gidx)
    a_ = [i for i in it_s if i[0] == "PAIRWISE" and i[1] == "|Pearson|"][0]
    assert np.isfinite(a_[2]["P50"]) and np.isfinite(a_[3]["P50"]).sum() > 0.9 * B      # 150 >= 2*50
    assert np.isnan(a_[2]["P100"]) and np.isnan(a_[3]["P100"]).all() and np.isnan(a_[4]["P100"]) and np.isnan(a_[5]["P100"])   # 150 < 200
    ws = weighted_metrics(np.ones((3, 150)), np.arange(150) % 3 == 0, np.arange(150), n_rows=150)
    assert np.isnan(ws["P100"]).all() and np.isnan(ws["R100"]).all() and np.isfinite(ws["P50"]).all()
    # (2c) reweighting rule: T2's first endpoint (draw value .5) is undefined in 10/40 draws -> on
    # those draws T2 = its one remaining endpoint (.3)
    def item(dpt, bs):
        z = {k: dpt for k in MET}
        return ("PAIRWISE", "X", z, {k: bs.copy() for k in MET}, {k: 0.7 for k in MET}, {k: 0.7 - dpt for k in MET})
    ag = Agg(40)
    b2 = np.full(40, .5)
    b2[:10] = np.nan
    ag.add("regulon", "T1", [item(.1, np.full(40, .1))])
    ag.add("regulon", "T2", [item(.5, b2), item(.3, np.full(40, .3))])
    ag.add("regulon", "T3", [item(.2, np.full(40, .2))])
    rr = ag.rows("SYN")[0]
    assert rr["n_tf"] == 3 and rr["n_endpoints"] == 4 and rr["nboot_AUROC"] == 40 and rr["label"] != "NA" and abs(rr["d_AUROC"] - (.1 + .4 + .2) / 3) < 1e-12
    assert abs(rr["lo_AUROC"] - (.1 + .3 + .2) / 3) < 1e-9 and abs(rr["hi_AUROC"] - (.1 + .4 + .2) / 3) < 1e-9, rr   # affected draws .2, others .2333
    assert abs(rr["frac_draws_reweighted_AUROC"] - .25) < 1e-12 and abs(rr["max_endpoint_undefined_rate_AUROC"] - .25) < 1e-12
    # a TF whose only endpoint is undefined in 10/40 draws: those draws have no defined TF ->
    # unusable (30 < .95*40 -> label NA)
    ag = Agg(40)
    u = np.full(40, .2); u[:10] = np.nan
    ag.add("regulon", "T1", [item(.2, u)])
    r1 = ag.rows("SYN")[0]
    assert r1["nboot_AUROC"] == 30 and r1["label"] == "NA" and r1["frac_draws_reweighted_AUROC"] == .25
    # (4) non-finite score inside a universe is an error (explicit raise, survives python -O)
    try:
        eval_rows(np.array([1.0, np.nan, 2.0, 3.0]), np.array([1, 0, 1, 0], bool), np.ones(4, bool), np.arange(4), counts[:, :4])
        raise AssertionError("non-finite score accepted")
    except RuntimeError as e:
        assert "non-finite" in str(e)
    # (3) finiteness mismatch in the endpoint gate raises both ways
    okl = next(l for l in legacy_all if l["status"] == "OK")
    for new_v, old_v in ((np.nan, 0.5), (0.5, np.nan)):
        o2 = {(okl["endpoint"], okl["scope"], okl["comparator"]): dict(okl, system="SYN", AUROC_GB=old_v)}
        try:
            check_legacy(dict(okl, AUROC_GB=new_v), o2)
            raise AssertionError("finiteness mismatch accepted")
        except RuntimeError as e:
            assert "finiteness" in str(e)
    # (5) forest gate: exact reference passes, perturbed / missing / extra groups raise
    fr = {(r["system"], r["family"], r["scope"], r["method"]): dict(r, fam=r["family"], comparator=r["method"], AUROC_GB=r["GB_AUROC"],
                                                                     AUROC_comp=r["method_AUROC"], delta=r["d_AUROC"]) for r in sr}
    assert check_forest(sr, fr, "SYN") == len(sr)
    k0 = next(iter(fr))
    for mut in (lambda d: d.update(delta=d["delta"] + 1e-6), lambda d: d.update(n_tf=d["n_tf"] + 1)):
        f2 = {k: dict(v) for k, v in fr.items()}
        mut(f2[k0])
        try:
            check_forest(sr, f2, "SYN")
            raise AssertionError("forest gate accepted a perturbed group")
        except RuntimeError as e:
            assert "PARITY_GATE_FAILED(forest)" in str(e)
    for bad_f in ({k: v for k, v in fr.items() if k != k0}, {**fr, ("SYN", "regulon", "PAIRWISE", "ghost"): dict(fr[k0])}):
        try:
            check_forest(sr, bad_f, "SYN")
            raise AssertionError("forest gate accepted missing/extra group")
        except RuntimeError as e:
            assert "PARITY_GATE_FAILED(forest)" in str(e)
    assert unchanged_group("GATA1", "regulon") and not unchanged_group("GATA1", "functional") and unchanged_group("K562", "functional") and not unchanged_group("ZEBRAFISH", "functional")
    # (6) atomic write: no tmp files left, receipt records output shas
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        write_atomic(td, {"endpoints": E, "system_contrasts": S}, dict(B=B))
        names = sorted(q.name for q in Path(td).iterdir())
        assert names == ["endpoints.tsv", "receipt.json", "system_contrasts.tsv"], names
        rc = json.loads((Path(td) / "receipt.json").read_text())
        assert rc["output_sha256"]["endpoints.tsv"] == sha256(Path(td) / "endpoints.tsv")
        try:
            pin_check(Path(td) / "endpoints.tsv", "deadbeef")
            raise AssertionError("pin accepted")
        except SystemExit as e:
            assert "SHA_MISMATCH" in str(e)
        class Boom:
            def to_csv(self, *a, **k):
                Path(a[0]).write_text("x")
                raise ValueError("boom")
        with tempfile.TemporaryDirectory() as td2:
            try:
                write_atomic(td2, {"endpoints": E, "system_contrasts": Boom()}, dict(B=B))
                raise AssertionError("write_atomic swallowed the error")
            except ValueError:
                pass
            assert list(Path(td2).iterdir()) == [], list(Path(td2).iterdir())               # tmp files cleaned up, nothing renamed
        assert module_hashes({Path(__file__).resolve().parent}).get(str(Path(__file__).resolve())) == sha256(__file__)
    # non-contiguous TF units must raise
    try:
        run_system("SYN", [recs[0], recs[1], recs[0]], union, 20, 3)
        raise AssertionError("non-contiguous unit accepted")
    except RuntimeError as e:
        assert "AGG_UNIT_NOT_CONTIGUOUS" in str(e)
    assert endpoint_class("F2|GATA1|chronicKO", "F2") == "chronicKO" and endpoint_class("F3|GATA1|V2_BOUND_PROM", "F3") == "occupancy_conditioned"
    # 6. SOTA extension: an optional method adds new scopes/rows only
    r0 = recs[0]
    rs = dict(r0, V={**r0["V"], "PIDC": np.where(rng.random(len(r0["obs_genes"])) < 0.15, np.nan, rng.normal(size=len(r0["obs_genes"])) + 0.2 * r0["y"])})
    e0, l0, i0 = process_endpoint(r0, counts, gidx)
    e1, l1, i1 = process_endpoint(rs, counts, gidx)
    assert l0 == l1, "SOTA method changed the legacy (parity) rows"
    key = lambda r: (r["scope"], r["method"], r["paired_with"])
    old_rows = {key(r): r for r in e0 if r["scope"] in ("COMMON_AVAILABLE", "PAIRWISE") and r["status"] == "OK"}
    for r in e1:
        if key(r) in old_rows and r["status"] == "OK":
            o_ = old_rows[key(r)]
            assert all(r[k] == o_[k] or (r[k] != r[k] and o_[k] != o_[k]) for k in ("n", "n_pos", "AUROC", "AUPRC")), key(r)
    sc1 = {(r["scope"], r["method"]) for r in e1 if r["status"] == "OK"}
    assert (SOTA_SCOPE, "PIDC") in sc1 and (SOTA_SCOPE, "GeoBRIDGE") in sc1 and ("PAIRWISE", "PIDC") in sc1 and (SOTA_SCOPE, "|Pearson|") in sc1
    assert sum(r["status"] == "MISSING" and r["method"] in SOTA for r in e1) == 3 and not any(r["method"] in SOTA for r in e0)
    n_six, n_six_s = (next(r["n"] for r in e1 if r["scope"] == sc and r["method"] == "GeoBRIDGE") for sc in ("COMMON_AVAILABLE", SOTA_SCOPE))
    assert n_six_s < n_six, (n_six, n_six_s)                                   # the SOTA mask additionally requires PIDC finite
    assert sum(r["scope"] == SOTA_SCOPE and r["method"] == "GeoBRIDGE" for r in e1) == 1
    ss = [dict(r, scope=r["scope"]) for r in run_system("SYN", [rs, dict(recs[1], V={**recs[1]["V"], "PIDC": rng.normal(size=len(recs[1]["obs_genes"]))})], union, 20, 3)[1]]
    assert any(r["method"] == "PIDC" for r in ss)
    print("selftest 5b OK: SOTA methods add scopes without touching COMMON_AVAILABLE / legacy rows / forest gate")
    print("selftest 5 OK: end-to-end contrast/label/cross-system/coverage plumbing")
    selftest_k562()
    print("gb_canon_table selftest OK")


def selftest_k562():
    import tempfile
    import types
    rng = np.random.default_rng(7)
    G = [f"G{i}" for i in range(60)]
    genes_state = np.array(G + ["EXTRA1"], dtype=object)               # state carries one gene outside the universe
    universe = np.array([g.lower() if i % 5 == 0 else g for i, g in enumerate(G)] + ["NOSTATE"], dtype=object)   # lower-case symbols + a gene absent from the state
    tfs = np.array(["G3", "G7", "G9", "NOTINUNI"], dtype=object)
    S = np.zeros((4, len(genes_state)), np.int8)
    S[0, 10:30] = 2; S[0, 30:36] = 3; S[0, 36:58] = 1; S[0, 58:60] = 4; S[0, 3] = 2          # G3: focal gene labelled UP -> must be removed; 4 = LOW_INFO
    S[1, 10:14] = 2; S[1, 14:60] = 1                                                            # G7: only 4 positives -> skipped
    S[2, 10:25] = 2; S[2, 25:33] = 1; S[2, 33:60] = 4                                            # G9: 8 primary negatives (<10) but 35 with LOW_INFO -> primary skipped, sensitivity kept
    S[3, 10:25] = 2; S[3, 25:60] = 1
    code = json.dumps(K562_CODES)

    def write_state(path, S_=S, code_=code):
        np.savez(path, genes=genes_state, tfs=tfs, state=S_, qval=np.zeros(S_.shape, np.float32), log2fc=np.zeros(S_.shape, np.float32), code_map=np.array(code_))
    with tempfile.TemporaryDirectory() as d:
        sp = Path(d) / "st.npz"
        write_state(sp)
        out, skipped = k562_panels(sp, universe)
        stem = lambda tf, arm: K562_STEM.format(tf=tf, arm=arm)
        pos, uni = out[stem("G3", "primary") + "|pos"], out[stem("G3", "primary") + "|panel_universe"]
        gu = np.char.upper(universe.astype(str))
        exp_pos = np.zeros(len(gu), bool); exp_pos[[i for i in range(10, 36)]] = True                # genes 10..35 up/down; focal G3 (idx 3) forced out
        assert np.array_equal(pos, exp_pos) and not uni[3] and not pos[3] and not uni[len(gu) - 1] and not uni[58] and not uni[59]   # LOW_INFO (58,59) not in primary; NOSTATE gene absent
        assert np.array_equal(uni, exp_pos | np.isin(np.arange(len(gu)), np.arange(36, 58)))
        sens = out[stem("G3", "sensitivity") + "|panel_universe"]
        assert sens[58] and sens[59] and sens.sum() == uni.sum() + 2                                 # sensitivity arm adds state 4
        assert stem("G7", "primary") + "|pos" not in out and ("G7|primary", "LOW_POSITIVES_OR_NEGATIVES") in skipped and ("NOTINUNI", "ABSENT_FROM_UNIVERSE") in skipped
        assert stem("G9", "primary") + "|pos" not in out and stem("G9", "sensitivity") + "|pos" in out
        try:
            write_state(Path(d) / "bad.npz", code_=json.dumps({**K562_CODES, "LOW_INFORMATION_TESTED_NULL": 5}))
            k562_panels(Path(d) / "bad.npz", universe); raise AssertionError("code-map mismatch not refused")
        except SystemExit:
            pass
        # receipt gate (fail closed) + adapter output (written once, idempotent, refuses a differing
        # overwrite)
        root = Path(d) / "root"
        td = root / "results" / K562_TRUTH["v4"][0]
        td.mkdir(parents=True)
        (root / "results" / "canon_K562").mkdir(parents=True)
        np.savez(root / K562_LEGACY_PG, genes=universe, **{f"dc1_min|U|k10|{t}|any_change|x|pos": np.zeros(2, bool) for t in ("G3", "G7", "G9")})      # scored roster = G3, G7, G9
        write_state(td / "k562_state.npz")
        (td / "READY_FLAG").write_text("ok")
        rec = dict(label="CANONICAL", variant="PRIMARY_AMBIGUOUS_AS_NULL", per_pair_sha256="ab", per_pair_sha256_in_receipt="ab", out_state_sha256=sha256(td / "k562_state.npz"), n_gene_unmapped=0)
        (td / "receipt.json").write_text(json.dumps(rec))
        assert check_k562_truth_receipt("v4", root)["variant"] == "PRIMARY_AMBIGUOUS_AS_NULL"
        for bad in (dict(label="EXPLORATORY"), dict(variant="SENSITIVITY_AMBIGUOUS_EXCLUDED"), dict(per_pair_sha256="zz"), dict(out_state_sha256="0" * 64), dict(n_gene_unmapped=3)):
            (td / "receipt.json").write_text(json.dumps({**rec, **bad}))
            try:
                check_k562_truth_receipt("v4", root); raise AssertionError(f"accepted a bad receipt {bad}")
            except SystemExit:
                pass
        (td / "receipt.json").write_text(json.dumps(rec))
        (td / "READY_FLAG").unlink()
        try:
            check_k562_truth_receipt("v4", root); raise AssertionError("accepted a missing READY_FLAG")
        except SystemExit:
            pass
        (td / "READY_FLAG").write_text("ok")
        pg, _, _ = build_k562_v4_pergene("v4", root)
        assert pg.parent.name == "canon_K562_v4" and (pg.parent / "receipt.json").exists()
        pg2, _, _ = build_k562_v4_pergene("v4", root)                                                # idempotent
        S2 = S.copy(); S2[0, 10:20] = 1
        write_state(td / "k562_state.npz", S2)
        (td / "receipt.json").write_text(json.dumps({**rec, "out_state_sha256": sha256(td / "k562_state.npz")}))
        try:
            build_k562_v4_pergene("v4", root); raise AssertionError("overwrote a different adapter output")
        except SystemExit:
            pass
        # per_gene redirect: exactly one hit, other paths untouched, zero hits abort
        CA = types.SimpleNamespace(np=np)
        np.savez(pg.parent / "other.npz", x=np.arange(3))
        with redirect_k562_pergene(CA, pg, root) as hits:
            assert set(CA.np.load(root / K562_LEGACY_PG, allow_pickle=True).files) == set(np.load(pg, allow_pickle=True).files) and len(hits) == 1     # legacy path served from the adapter file
            assert list(CA.np.load(pg.parent / "other.npz")["x"]) == [0, 1, 2] and CA.np.zeros(2).shape == (2,)
        assert CA.np is np
        try:
            with redirect_k562_pergene(CA, pg, root):
                pass
            raise AssertionError("zero-hit redirect not refused")
        except RuntimeError:
            assert CA.np is np
        # truth gate: consistent endpoint passes, a flipped label / an untested evaluated gene fails
        write_state(sp)
        ar = np.arange(len(gu))
        ev = (ar >= 10) & (ar < 58)                                                                   # G3 endpoint: focal gene 3 is outside, LOW_INFO genes 58/59 outside
        y = (ar >= 10) & (ar < 36)
        rec_ = dict(source="G3", obs_genes=np.asarray(gu), ev=ev, y=y, endpoint="e")
        gcache = {}
        assert k562_truth_gate(rec_, str(sp), gcache) == int(rec_["ev"].sum())
        flipped = dict(rec_, y=rec_["y"] ^ np.isin(np.arange(len(gu)), [12]))
        try:
            k562_truth_gate(flipped, str(sp), gcache); raise AssertionError("flipped label accepted")
        except RuntimeError:
            pass
        lowinfo = dict(rec_, ev=rec_["ev"] | np.isin(np.arange(len(gu)), [58]))
        try:
            k562_truth_gate(lowinfo, str(sp), gcache); raise AssertionError("LOW_INFO gene accepted into the primary universe")
        except RuntimeError:
            pass
    print("selftest 6 OK: K562 v4 adapter (mirror rules, focal removal, arms, thresholds), fail-closed receipt gate, idempotent output, redirect, truth gate")


if __name__ == "__main__":
    main()
