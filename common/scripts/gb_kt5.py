#!/usr/bin/env python
"""KT5 (§244/§248 T6): does pseudotime-informed pair selection matter?  Truth-free scoring; evaluation in gb_killtests.

Per root: take the stored canonical pseudotime ({root}_pt) and component labels ({root}_component) from the canonical
pairs npz; permute pseudotime among cells WITHIN each component (seed 0); rebuild pairs with the production builder and
settings (GEOBRIDGE_PREPROCESS_V1 pair_construction: window (.02, .25], cap 40, pair_seed 42, stable-cell-id order then
pseudotime order, exactly as gb_gata1_preflight.canonical_pairs).  Fail-closed parity first: rebuilding from the
UNshuffled stored pseudotime must reproduce the stored production pairs exactly.  Then score the roster sources with
the production algebra (U channel, k = 0..8) on the shuffled pairs.
  python scratch/gb_kt5.py SYSTEM  -> results/real_uniform/SYSTEM/kt5_shuffledpt_U_k0_8.npz (+ receipt)
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_real_uniform as RU

KMAX, SEED = 8, 0


def rebuild(pt, ids, pc, build_forward_pairs):
    can = np.argsort(np.asarray(ids).astype(str), kind="stable")
    ptc = pt[can]
    o2 = np.lexsort((np.arange(len(ptc)), ptc))
    dummy = np.zeros((len(pt), 1), np.float32)
    pair = build_forward_pairs(dummy, dummy, ptc[o2], min_forward_dt=pc["min_forward_dt"],
                               max_forward_dt=pc["max_forward_dt"], max_pairs_per_cell=pc["max_pairs_per_cell"],
                               pair_seed=pc["pair_seed"])
    return can[o2[np.asarray(pair.pair_i, int)]], can[o2[np.asarray(pair.pair_j, int)]]


def main(system):
    sys.path.insert(0, str(RU.PAIRSPACE))
    import gb_gata1_preflight as PF
    import gb_substrate as GS
    from gb_target_extension import CHUNK, NPC, project_scores

    t0 = time.time()
    system = system.upper()
    D = RU.canonical_inputs(system)
    A, S = D["A"], D["S"]
    pc = json.load(open(RU.R / "GEOBRIDGE_PREPROCESS_V1.json"))["pair_construction"]
    bfp = PF._pairspace_verified()[0].build_forward_pairs
    Z = np.load(GS.paths(system, canonical=True)[1], allow_pickle=True)
    ids = np.asarray(A.obs_names).astype(str)
    axis, cols = list(D["axis"]), D["axis_cols"]
    ai = {g.upper(): i for i, g in enumerate(axis)}
    fi = {g: i for i, g in enumerate(D["fixture_genes"])}
    srcs = list(D["sources"])
    Mu = RU.dense(A.layers["Mu"][:, cols], np.float32)
    Xall = np.asarray(S["X"], np.float64)
    Xs = Xall[:, [fi[g] for g in srcs]]
    P30 = np.asarray(A.obsm["X_pca"], float)[:, :NPC]
    rng = np.random.default_rng(SEED)
    out, meta = {}, {}
    for root in RU.ROOTS:
        pt = np.asarray(Z[f"{root}_pt"], float)
        comp = np.asarray(Z[f"{root}_component"]).ravel()
        I0, J0 = (np.asarray(a, int) for a in S["pairs"][root])
        Ir, Jr = rebuild(pt, ids, pc, bfp)
        o0, orr = np.lexsort((J0, I0)), np.lexsort((Jr, Ir))  # stored pairs are saved in a different ORDER; |S| is
        # invariant to pair order (residual sums and the QR basis permute together), so compare (I,J)-sorted arrays exactly
        if not (np.array_equal(Ir[orr], I0[o0]) and np.array_equal(Jr[orr], J0[o0])):
            raise SystemExit(f"REFUSING_KT5_PARITY: {system}/{root}: rebuilt pairs != production pairs (sorted arrays)")
        pts = pt.copy()
        for c in np.unique(comp):
            m = np.flatnonzero(comp == c)
            pts[m] = pt[rng.permutation(m)]
        I, J = rebuild(pts, ids, pc, bfp)
        Qf, _ = np.linalg.qr(P30[J] - P30[I])
        dx = (Xs[J] - Xs[I]).astype(np.float32)
        arr = np.full((len(srcs), len(axis), KMAX + 1), np.nan, np.float32)
        for g0 in range(0, len(axis), CHUNK):
            sl = slice(g0, min(g0 + CHUNK, len(axis)))
            dy = (Mu[J, sl] - Mu[I, sl]).astype(np.float32)
            for k, _n, _d, _z, absS in project_scores(dx, dy, Qf, kmax=KMAX):
                arr[:, sl, k] = absS
        for j, s in enumerate(srcs):
            arr[j, ai[s.upper()], :] = np.nan
        out[root] = arr
        meta[root] = {"n_pairs_production": int(len(I0)), "n_pairs_shuffled": int(len(I)),
                      "parity_rebuild_exact_sorted_arrays": True,
                      "median_abs_dpt_shuffled_pairs_original_pt": float(np.median(np.abs(pt[J] - pt[I]))),
                      "median_abs_dpt_production_pairs": float(np.median(pt[J0] - pt[I0]))}
        print(f"[kt5] {system} {root}: {meta[root]} {time.time()-t0:.0f}s", flush=True)
    path = RU.system_dir(system) / "kt5_shuffledpt_U_k0_8.npz"
    binding = {"method": "GB-KT5-shuffled-pseudotime-U-k0..8", "inputs": RU.input_manifest(D),
               "config": {"shuffle": "pseudotime permuted within component", "interpretation": "DPT-assignment destruction control (changes locality, branch composition and residualisation together; one pre-declared seed)", "seed": SEED, "pair_construction": pc,
                          "kmax": KMAX, "target_extension_sha256": RU.sha(RU.SCRATCH / "gb_target_extension.py")},
               "producer_sha256": RU.sha(Path(__file__).resolve()), "truth_read": False}
    binding["binding_sha256"] = RU.digest_object(binding)
    RU.write_part(path, binding, {"genes": np.asarray(axis), "sources": np.asarray(srcs),
                                  **{f"{r}|U": a for r, a in out.items()}}, {"meta": meta, "wall_s": time.time() - t0})
    print(f"[kt5] {system} -> {path}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
