#!/usr/bin/env python
"""REAL FIGURE 1 eligibility — COVERAGE ONLY (internal design note (not shipped)). No score is read.
One orthology rule for every system: HUMAN CollecTRI (data/collectri/collectri_9606.tsv) mapped to the substrate species by
STRICT 1:1 orthologs only — mouse: MGI HOM_MouseHumanSequence.rpt (DB Class Key groups with exactly one mouse and one human symbol);
zebrafish: ZFIN zfin_human_orthos.txt (zebrafish symbol <-> human symbol pairs that are 1:1 in both directions); human: identity.
Edges whose source or target lacks a 1:1 ortholog are dropped (logged). TF eligibility: catalogue TF on the axis with spliced
detection >= .05, and >= 10 mapped positives AND >= 10 unannotated comparators inside the axis universe (axis minus the source).
FINAL-UNIVERSE RULE (internal review 2026-09-24 finding: eligibility must use the contract's final common finite universe, not the evaluator
axis): per system × TF the universe is the evaluator-axis genes, minus the source, that are FINITE FOR EVERY PLOTTED METHOD — the
frozen GeoBRIDGE rows (results/regulon_panel/<SYS>/rankings.npz: EU, TU, EX, PEARSON) and every comparator array present under
results/canon_<SYS>/sota/{grnboost2,genie3,velorama}.npz (SMOKE files excluded). Finiteness is truth-free. A TF whose GeoBRIDGE row
has not been scored is PENDING_SCORES (not eligible yet); methods absent for a system are listed and eligibility is re-run when
they exist. Positive/comparator counts are taken on that final universe.
usage: gb_fig1_eligibility.py            -> results/fig_contracts/fig1_eligibility.json"""
import collections, hashlib, json, os, sys, time
import numpy as np
R = __import__("os").environ["GEOBRIDGE_ROOT"]; sys.path.insert(0, f"{R}/scratch")
SYSTEMS = {"GATA1": "mouse", "PANC": "mouse", "BONEMARROW": "human", "DENTATEGYRUS": "mouse", "GASTRULATION": "mouse",
           "ZEBRAFISH": "zebrafish"}
DET, MIN_POS, MIN_NEG = 0.05, 10, 10
COLLECTRI = f"{R}/data/collectri/collectri_9606.tsv"
MGI = f"{R}/data/orthology/HOM_MouseHumanSequence.rpt"
ZFIN = f"{R}/data/zebrafish_raw/zfin_human_orthos.txt"


def sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def one_to_one(pairs):
    """pairs: iterable of (species_symbol, human_symbol) -> {human_upper: species_upper} keeping only 1:1 relations"""
    pairs = {(a.upper(), b.upper()) for a, b in pairs if a and b}
    ca, cb = collections.Counter(a for a, _ in pairs), collections.Counter(b for _, b in pairs)
    return {b: a for a, b in pairs if ca[a] == 1 and cb[b] == 1}


def ortho(species):
    if species == "human":
        return None
    if species == "mouse":
        g = collections.defaultdict(lambda: {"m": set(), "h": set()})
        for ln in list(open(MGI))[1:]:
            f = ln.rstrip("\n").split("\t")
            if f[2] == "10090":
                g[f[0]]["m"].add(f[3])
            elif f[2] == "9606":
                g[f[0]]["h"].add(f[3])
        return one_to_one((next(iter(v["m"])), next(iter(v["h"]))) for v in g.values() if len(v["m"]) == 1 and len(v["h"]) == 1)
    return one_to_one((f[1], f[3]) for f in (ln.rstrip("\n").split("\t") for ln in open(ZFIN)) if len(f) > 3)


def regulons(species):
    """human CollecTRI mapped by 1:1 orthologs: {source_upper: set(target_upper)}"""
    m = ortho(species); out = collections.defaultdict(set); drop = 0
    for ln in list(open(COLLECTRI))[1:]:
        f = ln.rstrip("\n").split("\t"); s, t = f[2].upper(), f[3].upper()
        if m is not None:
            if s not in m or t not in m:
                drop += 1; continue
            s, t = m[s], m[t]
        out[s].add(t)
    return out, drop


def axis(system):
    from gb_substrate import load
    import anndata as ad
    S = load(system, canonical=True)
    p = f"{R}/results/canon_{system}/per_gene.npz"
    eg = np.asarray(np.load(p, allow_pickle=True)["genes"] if os.path.exists(p) else S["genes"]).astype(str)
    gu = np.char.upper(eg)
    if len(set(gu)) != len(gu):
        raise SystemExit(f"REFUSING: {system} axis not case-unique")
    gi = {g: i for i, g in enumerate(np.asarray(S["genes"]).astype(str))}
    A = ad.read_h5ad(S["fixture"], backed="r"); fm = {g: i for i, g in enumerate(np.asarray(A.var_names).astype(str))}
    det = np.asarray((A.layers["spliced"][:, [fm[g] for g in eg]] > 0).mean(0)).ravel()
    return gu, det, np.asarray(S["is_tf"])[[gi[g] for g in eg]], p if os.path.exists(p) else "substrate genes"


def selftest():
    m = one_to_one([("a", "A"), ("b", "B"), ("b2", "B"), ("c", "C"), ("c", "C2")])
    assert m == {"A": "A"}, m                       # only the strict 1:1 pair survives
    print("gb_fig1_eligibility selftest PASS")


def method_arrays(system, axis_gu):
    """{method: {TF_upper: finite-mask over axis (bool)}} for every scored method of the system"""
    out, missing = {}, []
    rp = f"{R}/results/regulon_panel/{system}/rankings.npz"
    if os.path.exists(rp):
        Z = np.load(rp, allow_pickle=True); zg = np.char.upper(np.asarray(Z["genes"]).astype(str)); pos = {g: i for i, g in enumerate(zg)}
        idx = np.array([pos.get(g, -1) for g in axis_gu])
        for m in ("EU", "TU", "EX", "PEARSON"):
            out[m] = {str(t).upper(): np.where(idx >= 0, np.isfinite(np.asarray(Z[m][i], float)[np.maximum(idx, 0)]), False)
                      for i, t in enumerate(np.asarray(Z["sources"]).astype(str))}
    else:
        missing.append("GEOBRIDGE(regulon_panel)")
    for m in ("grnboost2", "genie3", "velorama"):
        p = f"{R}/results/canon_{system}/sota/{m}.npz"
        if not os.path.exists(p):
            missing.append(m); continue
        Z = np.load(p, allow_pickle=True); zg = np.char.upper(np.asarray(Z["genes"]).astype(str)); pos = {g: i for i, g in enumerate(zg)}
        idx = np.array([pos.get(g, -1) for g in axis_gu]); sc = np.asarray(Z["score"], float)
        out[m] = {str(t).upper(): np.where(idx >= 0, np.isfinite(sc[i][np.maximum(idx, 0)]), False) for i, t in enumerate(np.asarray(Z["tfs"]).astype(str))}
    return out, missing


def main():
    out = dict(contract="REAL_FIGURE_CONTRACTS_v1 Fig1", collectri=COLLECTRI, collectri_sha256=sha(COLLECTRI), mgi_sha256=sha(MGI),
               zfin_sha256=sha(ZFIN), det_floor=DET, min_pos=MIN_POS, min_neg=MIN_NEG, score_read=False, systems={})
    cache = {}
    for sysn, sp in SYSTEMS.items():
        if sp not in cache:
            cache[sp] = regulons(sp)
        reg, drop = cache[sp]
        gu, det, is_tf, ax = axis(sysn); MA, missing = method_arrays(sysn, gu)
        rows, pending = {}, []
        for g, t, d in zip(gu, is_tf, det):
            if not t or d < DET:
                continue
            if "EU" not in MA or g not in MA["EU"]:
                pending.append(g); continue
            fin = np.ones(len(gu), bool); fin &= gu != g
            for m, rowsm in MA.items():
                if g in rowsm:
                    fin &= rowsm[g]
            uni = set(gu[fin]); pos = len(reg.get(g, set()) & uni); neg = len(uni) - pos
            if pos >= MIN_POS and neg >= MIN_NEG:
                rows[g] = dict(detection=round(float(d), 4), n_pos=pos, n_neg=neg, universe_n=len(uni),
                               methods_in_universe=sorted(m for m in MA if g in MA[m]))
        out["systems"][sysn] = dict(species=sp, axis=ax, axis_n=len(gu), methods_found=sorted(MA), methods_missing=missing,
                                    n_pending_scores=len(pending), pending_scores=pending,
                                    n_catalogue_detected=int(((is_tf) & (det >= DET)).sum()),
                                    n_edges_dropped_no_1to1=drop, n_eligible_tf=len(rows), eligible=rows,
                                    system_eligible=len(rows) > 0)
        print(sysn, sp, "axis", len(gu), "detected TFs", out["systems"][sysn]["n_catalogue_detected"], "eligible", len(rows), flush=True)
    os.makedirs(f"{R}/results/fig_contracts", exist_ok=True); p = f"{R}/results/fig_contracts/fig1_eligibility.json"
    out.update(script_sha256=sha(os.path.abspath(__file__)), frozen_at=time.strftime("%FT%TZ", time.gmtime()))
    json.dump(out, open(p, "w"), indent=1); print("wrote", p, sha(p))


if __name__ == "__main__":
    selftest()
    if "--selftest" not in sys.argv:
        main()
