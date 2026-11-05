#!/usr/bin/env python
"""§248-T7 diagnostic: catalogue-bank |S| from PRODUCTION residuals (full production Qf over ALL pairs, k 0..8) with the
sums restricted to one pre-declared pair type (same-branch or cross-branch; gb_shared_motion.BRANCH_MAPS).  Truth-free.
Not an estimator and not a pair sampler.
  python scratch/gb_pairsubset_scores.py SYSTEM same|cross -> results/real_uniform/SYSTEM/catalogue_U_k0_8_sub{type}.npz
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_real_uniform as RU
from gb_shared_motion import BRANCH_MAPS, pair_types

KMAX, BLOCK, CHUNK, EPS = 8, 256, 1000, 1e-12


def main(system, ptype):
    from gb_target_extension import NPC
    t0 = time.time()
    system = system.upper()
    D = RU.canonical_inputs(system)
    A, S = D["A"], D["S"]
    Zb = np.load(RU.system_dir(system) / "catalogue_U_k0_8.npz", allow_pickle=True)
    bank = list(np.asarray(Zb["sources"]).astype(str))
    axis, cols = list(D["axis"]), D["axis_cols"]
    if list(np.asarray(Zb["genes"]).astype(str)) != axis:
        raise SystemExit("REFUSING_BANK_AXIS")
    key, mp = BRANCH_MAPS[system]
    br = np.asarray([mp[c] for c in np.asarray(A.obs[key]).astype(str)])
    ai = {g: i for i, g in enumerate(axis)}
    fi = {g: i for i, g in enumerate(D["fixture_genes"])}
    Mu = RU.dense(A.layers["Mu"][:, cols], np.float32)
    Xall = np.asarray(S["X"], np.float64)
    P30 = np.asarray(A.obsm["X_pca"], float)[:, :NPC]
    G, nb = len(axis), len(bank)
    out, meta = {}, {}
    for root in RU.ROOTS:
        I, J = (np.asarray(a, int) for a in S["pairs"][root])
        sub = np.ones(len(I), bool) if ptype == "all" else pair_types(br[I], br[J]) == ptype
        meta[root] = {"n_pairs": int(len(I)), "n_subset": int(sub.sum())}
        Q = np.linalg.qr(P30[J] - P30[I])[0][:, :KMAX]
        arr = np.full((nb, G, KMAX + 1), np.nan, np.float32)
        for b0 in range(0, nb, BLOCK):
            blk = bank[b0:b0 + BLOCK]
            dx = (Xall[J][:, [fi[g] for g in blk]] - Xall[I][:, [fi[g] for g in blk]]).astype(np.float32).astype(np.float64)
            Cx = Q.T @ dx
            for g0 in range(0, G, CHUNK):
                sl = slice(g0, min(g0 + CHUNK, G))
                dy = (Mu[J, sl] - Mu[I, sl]).astype(np.float32).astype(np.float64)
                Cy = Q.T @ dy
                rx, ry = dx.copy(), dy.copy()
                for k in range(KMAX + 1):
                    if k:
                        rx -= Q[:, k - 1:k] * Cx[k - 1:k]
                        ry -= Q[:, k - 1:k] * Cy[k - 1:k]
                    x, y = rx[sub].astype(np.float32).astype(np.float64), ry[sub].astype(np.float32).astype(np.float64)
                    num, den = x.T @ y, np.abs(x).T @ np.abs(y)
                    with np.errstate(invalid="ignore", divide="ignore"):
                        arr[b0:b0 + len(blk), sl, k] = np.where(den > EPS, np.abs(num / np.where(den > EPS, den, 1)), np.nan)
            print(f"[subset] {system} {ptype} {root}: {min(b0 + BLOCK, nb)}/{nb} {time.time()-t0:.0f}s", flush=True)
        for j, g in enumerate(bank):
            arr[j, ai[g], :] = np.nan
        out[root] = arr
    if ptype == "all":  # parity self-check: must reproduce the production bank
        for r in RU.ROOTS:
            ref = np.asarray(Zb[f"{r}|U"], float)
            d = np.nanmax(np.abs(ref - out[r]))
            if not np.array_equal(np.isfinite(ref), np.isfinite(out[r])) or d > 1e-5:
                raise SystemExit(f"REFUSING_SUBSET_PARITY: {system}/{r} max|diff| {d}")
        print(f"[subset] {system} all-pairs parity with the production bank OK", flush=True)
        return
    path = RU.system_dir(system) / f"catalogue_U_k0_8_sub{ptype}.npz"
    binding = {"method": f"GB-catalogue-bank-U-k0..8-pairsubset-{ptype}", "inputs": RU.input_manifest(D),
               "config": {"pair_type": ptype, "branch_map": {k: v for k, v in mp.items()}, "residuals": "full production Qf",
                          "kmax": KMAX}, "producer_sha256": RU.sha(Path(__file__).resolve()), "truth_read": False}
    binding["binding_sha256"] = RU.digest_object(binding)
    RU.write_part(path, binding, {"genes": np.asarray(axis), "sources": np.asarray(bank),
                                  **{f"{r}|U": a for r, a in out.items()}}, {"meta": meta, "wall_s": time.time() - t0})
    print(f"[subset] {system} {ptype} -> {path} {meta}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
