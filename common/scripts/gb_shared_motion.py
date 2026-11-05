#!/usr/bin/env python
"""§248 T1/T2/T4 — shared-motion descriptors (truth-free; spec internal design note (not shipped)).

T1 pair geometry: per root, fraction of pairs same-branch / trunk->branch / cross-branch under the PRE-DECLARED branch
   maps (§248 §5), |Δpt|, and A_p(8) = ||first-8 PC coordinates of d_p||^2 / ||d_p||^2 by pair type within |Δpt|
   deciles (d_p = P30[J] − P30[I]; A_p is a per-pair leverage proxy, the production removal is a pair-space regression).
T2 sharedness per representation (RAW-z, PRIMARY, K8-diag, RANDOM-8): median Spearman between the target rankings of
   500 fixed random bank-TF pairs (TF-pair bootstrap CI) and per-gene hubness (mean within-TF percentile across the bank).
T4 Spearman(hubness drop RAW-z -> rep, sum of squared PC1..8 loadings) for rep in PRIMARY, K8-diag, RANDOM-8.

  python scratch/gb_shared_motion.py DENTATEGYRUS [...]  -> results/real_uniform/shared_motion/{SYSTEM}.json + T2_PAIRS.tsv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_killtests as KT
import gb_real_uniform as RU

OUT = RU.OUT / "shared_motion"
TRUNK = "__trunk__"
BRANCH_MAPS = {  # §248 §5, fixed before computation; value = branch name (TRUNK for trunk sets)
    "BONEMARROW": ("clusters", {**{c: TRUNK for c in ("HSC_1", "HSC_2", "Precursors")},
                                **{c: "ery_mega" for c in ("Ery_1", "Ery_2", "Mega")},
                                **{c: "mono" for c in ("Mono_1", "Mono_2")}, "DCs": "DC", "CLP": "lymphoid"}),
    "DENTATEGYRUS": ("clusters", {**{c: "neuronal" for c in ("nIPC", "Neuroblast", "Granule immature", "Granule mature")},
                                  **{c: "astroglial" for c in ("Radial Glia-like", "Astrocytes")},
                                  **{c: "oligo" for c in ("OPC", "OL")},
                                  **{c: c for c in ("Microglia", "Endothelial", "Mossy", "GABA", "Cajal Retzius", "Cck-Tox")}}),
    "PANC": ("clusters", {**{c: TRUNK for c in ("Ductal", "Ngn3 low EP", "Ngn3 high EP", "Pre-endocrine")},
                          "Alpha": "Alpha", "Beta": "Beta", "Delta": "Delta", "Epsilon": "Epsilon"}),
    "ZEBRAFISH": ("cell_type", {**{c: TRUNK for c in ("NPB_nohox", "NPB_hox3")},
                                **{c: "dNC" for c in ("dNC_nohox", "dNC_hoxa2b", "dNC_hox34")},
                                **{c: "mNC" for c in ("mNC_vagal", "mNC_nohox", "mNC_head_mesenchymal", "mNC_arch1",
                                                      "mNC_arch2", "mNC_hox34")},
                                "Pigment": "Pigment", "NC_trunk": "NC_trunk"}),
    "GASTRULATION": ("celltype", {**{c: TRUNK for c in ("Epiblast", "Primitive Streak", "Anterior Primitive Streak",
                                                        "Caudal epiblast")},
                                  **{c: "ectoderm" for c in ("Rostral neurectoderm", "Caudal neurectoderm", "Surface ectoderm")},
                                  **{c: "mesoderm" for c in ("Nascent mesoderm", "Mixed mesoderm", "Intermediate mesoderm",
                                                             "Paraxial mesoderm", "Somitic mesoderm", "Pharyngeal mesoderm",
                                                             "Caudal Mesoderm", "Mesenchyme", "ExE mesoderm", "Allantois")},
                                  **{c: "endoderm" for c in ("Def. endoderm", "Gut", "Visceral endoderm", "Notochord")},
                                  **{c: "blood_endo" for c in ("Haematoendothelial progenitors", "Blood progenitors 1",
                                                               "Blood progenitors 2", "Erythroid1", "Endothelium")},
                                  "PGC": "PGC"}),
}


def pair_types(bi, bj):
    same = bi == bj
    trunk = (bi == TRUNK) ^ (bj == TRUNK)
    return np.where(same, "same", np.where(trunk, "trunk_branch", "cross"))


def t1(system, A, S, pt_all):
    if system not in BRANCH_MAPS:
        return None
    key, mp = BRANCH_MAPS[system]
    lab = np.asarray(A.obs[key]).astype(str)
    miss = sorted(set(lab) - set(mp))
    if miss:
        raise SystemExit(f"REFUSING_UNMAPPED_CLUSTERS: {system}: {miss}")
    br = np.asarray([mp[c] for c in lab])
    P = np.asarray(A.obsm["X_pca"], float)[:, :30]
    out = {}
    for root in RU.ROOTS:
        I, J = (np.asarray(a, int) for a in S["pairs"][root])
        d = P[J] - P[I]
        Ap = (d[:, :8] ** 2).sum(1) / (d ** 2).sum(1)                   # PC-energy share (NOT leverage)
        Q8 = np.linalg.qr(d)[0][:, :8]
        lev = (Q8 ** 2).sum(1) * len(d) / 8.0                            # exact pair-space leverage h_p, mean 1
        typ = pair_types(br[I], br[J])
        dpt = np.abs(pt_all[root][J] - pt_all[root][I])
        dec = np.minimum((rankdata(dpt, method="ordinal") - 1) * 10 // len(dpt), 9)
        per = []
        for b in range(10):
            m = dec == b
            row = {t: float(np.median(Ap[m & (typ == t)])) if (m & (typ == t)).sum() >= 30 else None
                   for t in ("same", "trunk_branch", "cross")}
            per.append(row)
        diffs = [r["cross"] - r["same"] for r in per if r["cross"] is not None and r["same"] is not None]
        out[root] = {"frac": {t: float((typ == t).mean()) for t in ("same", "trunk_branch", "cross")},
                     "median_dpt": {t: float(np.median(dpt[typ == t])) if (typ == t).any() else None
                                    for t in ("same", "trunk_branch", "cross")},
                     "median_pc_dist": {t: float(np.median(np.linalg.norm(d[typ == t], axis=1))) if (typ == t).any() else None
                                        for t in ("same", "trunk_branch", "cross")},
                     "pc_energy_share_top8_median": {t: float(np.median(Ap[typ == t])) if (typ == t).any() else None
                                                     for t in ("same", "trunk_branch", "cross")},
                     "leverage_Q8_mean_relative": {t: float(np.mean(lev[typ == t])) if (typ == t).any() else None
                                                   for t in ("same", "trunk_branch", "cross")},
                     "leverage_Q8_share": {t: float(lev[typ == t].sum() / lev.sum()) for t in ("same", "trunk_branch", "cross")},
                     "Ap8_by_dpt_decile": per,
                     "cross_minus_same_Ap8_deciles": diffs,
                     "prediction_cross_gt_same": (float(np.mean(np.asarray(diffs) > 0)) if diffs else None)}
    return out


def finite_pct(v):
    """Within-source percentile over FINITE entries only (internal review §255: the NaN->-inf ranking was a bug)."""
    out = np.full(len(v), np.nan)
    ok = np.isfinite(v)
    out[ok] = rankdata(v[ok]) / ok.sum()
    return out


def t2_t4(system, bank, reps, A, eg, rng, n_src=150, B=200):
    """T2: all pairs among n_src fixed random bank sources; CI by a SOURCE (node) bootstrap (internal review §255)."""
    pct = {k: np.vstack([finite_pct(v) for v in R]) for k, R in reps.items()}
    src = np.sort(rng.choice(len(bank), min(n_src, len(bank)), replace=False))
    rows, summ = [], {}
    for k, R in reps.items():
        n = len(src)
        rho = np.full((n, n), np.nan)
        for i in range(n):
            for j in range(i + 1, n):
                a, b = R[src[i]], R[src[j]]
                ok = np.isfinite(a) & np.isfinite(b)
                rho[i, j] = rho[j, i] = spearmanr(a[ok], b[ok])[0]
        iu = np.triu_indices(n, 1)
        bs = []
        for _ in range(B):
            pick = rng.integers(0, n, n)
            sub = rho[np.ix_(pick, pick)]
            same = pick[:, None] == pick[None, :]
            bs.append(np.nanmedian(sub[~same]))
        summ[k] = {"median_rho": float(np.nanmedian(rho[iu])), "lo": float(np.percentile(bs, 2.5)),
                   "hi": float(np.percentile(bs, 97.5)), "n_sources": int(n)}
        rows += [dict(system=system, rep=k, i=int(src[a]), j=int(src[b]), rho=float(rho[a, b])) for a, b in zip(*iu)]
    hub = {k: np.nanmean(P, 0) for k, P in pct.items()}
    fm = {g: i for i, g in enumerate(np.asarray(A.var_names).astype(str))}
    L = np.asarray(A.varm["PCs"], float)[[fm[g] for g in eg]]
    load8 = (L[:, :8] ** 2).sum(1)
    t4 = {}
    for k in reps:
        if k == "RAW-z":
            continue
        drop = hub["RAW-z"] - hub[k]
        ok = np.isfinite(drop)
        t4[k] = float(spearmanr(drop[ok], load8[ok])[0])
    hubsum = {k: {"hub_sd": float(np.nanstd(h)), "hub_q90_minus_q10": float(np.nanquantile(h, .9) - np.nanquantile(h, .1))}
              for k, h in hub.items()}
    return summ, rows, t4, hubsum


def main(systems):
    import anndata as ad
    import gb_substrate as GS

    OUT.mkdir(parents=True, exist_ok=True)
    allrows = []
    for system in systems:
        rng = np.random.default_rng(0)
        D = RU.canonical_inputs(system)
        eg = np.asarray(D["axis"]).astype(str)
        bank, reps = KT.bank_reps(system, eg, with_random=True)
        A = ad.read_h5ad(D["fixture"])
        S = GS.load(system, canonical=True)
        Z = np.load(GS.paths(system, canonical=True)[1], allow_pickle=True)
        pt = {r: np.asarray(Z[f"{r}_pt"], float) for r in RU.ROOTS}
        rep = {"system": system, "n_bank": len(bank), "T1": t1(system, A, S, pt),
               "provenance": {"producer_sha256": RU.sha(Path(__file__).resolve()),
                              "gb_killtests_sha256": RU.sha(RU.SCRATCH / "gb_killtests.py"),
                              "inputs": {p.name: RU.sha(p) for p in sorted(RU.system_dir(system).glob("catalogue_U_k0_8*.npz"))},
                              "fixture_sha256": RU.sha(D["fixture"]), "pairs_sha256": RU.sha(D["pairs_file"])}}
        rep["T2"], rows, rep["T4_spearman_hubdrop_vs_load8"], rep["hubness"] = t2_t4(system, bank, reps, A, eg, rng)
        allrows += rows
        RU.atomic_json(OUT / f"{system}.json", rep)
        print(json.dumps({"system": system, "T2": rep["T2"], "T4": rep["T4_spearman_hubdrop_vs_load8"],
                          "T1": {r: {"frac": v["frac"], "pc_energy": v["pc_energy_share_top8_median"],
                                     "leverage": v["leverage_Q8_mean_relative"], "cross>same": v["prediction_cross_gt_same"]}
                                 for r, v in (rep["T1"] or {}).items()}}, default=str), flush=True)
    pd.DataFrame(allrows).to_csv(OUT / "T2_PAIRS.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main([s.upper() for s in sys.argv[1:]])
