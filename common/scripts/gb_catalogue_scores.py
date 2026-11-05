#!/usr/bin/env python
"""Truth-free PRIMARY inputs for ALL observationally eligible catalogue TFs (the KT1/KT2 source bank; internal review 2026-09-26, §247).

Bank = catalogue TFs (substrate is_tf) on the evaluator axis with spliced detection >= .05: the Fig-1 observational
criterion WITHOUT the truth-coverage criterion (gb_fig1_eligibility.py:20,64-76,118).  Same algebra and inputs as the
production caches (gb_real_uniform.py gbcache: source rows from substrate X, targets Mu on the axis, Qf = QR of the pair
P30 displacement, gb_target_extension.project_scores), U channel only, k = 0..8 (the PRIMARY depths).
  python scratch/gb_catalogue_scores.py SYSTEM  -> results/real_uniform/SYSTEM/catalogue_U_k0_8.npz (+ .receipt.json)
Fail-closed parity: every bank source that is also in the production caches must match RU.gb_raw_rows to 1e-5 at k=0..8.
"""
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_real_uniform as RU

DET, KMAX, BLOCK, TOL = 0.05, 8, 256, 1e-5


def main(system: str, rotation: int | None = None) -> None:
    """rotation=None: production Qf.  rotation=SEED: RANDOM-8 control (§248) -- Qf = QR(dP @ R), R a fixed random
    orthogonal rotation of the 30-PC space, so the removed prefix spans displacement along random PC-space directions."""
    from gb_target_extension import CHUNK, NPC, project_scores

    t0 = time.time()
    system = system.upper()
    D = RU.canonical_inputs(system)
    A, S = D["A"], D["S"]
    axis, cols = list(D["axis"]), D["axis_cols"]
    is_tf = np.asarray(S["is_tf"], bool)[cols]
    det = np.asarray((A.layers["spliced"][:, cols] > 0).mean(0)).ravel()
    bank = [g for g, t, d in zip(axis, is_tf, det) if t and d >= DET]
    ai = {g: i for i, g in enumerate(axis)}
    fi = {g: i for i, g in enumerate(D["fixture_genes"])}
    Mu = RU.dense(A.layers["Mu"][:, cols], np.float32)
    Xall = np.asarray(S["X"], np.float64)
    P30 = np.asarray(A.obsm["X_pca"], float)[:, :NPC]
    G, nb = len(axis), len(bank)
    out = {r: np.full((nb, G, KMAX + 1), np.nan, np.float32) for r in RU.ROOTS}
    for root in RU.ROOTS:
        I, J = (np.asarray(a, int) for a in S["pairs"][root])
        dP = P30[J] - P30[I]
        if rotation is not None:
            Rot = np.linalg.qr(np.random.default_rng(rotation).normal(size=(NPC, NPC)))[0]
            dP = dP @ Rot
        Qf, _ = np.linalg.qr(dP)
        for b0 in range(0, nb, BLOCK):
            blk = bank[b0:b0 + BLOCK]
            Xs = Xall[:, [fi[g] for g in blk]]
            dx = (Xs[J] - Xs[I]).astype(np.float32)
            for g0 in range(0, G, CHUNK):
                sl = slice(g0, min(g0 + CHUNK, G))
                dy = (Mu[J, sl] - Mu[I, sl]).astype(np.float32)
                for k, _n, _d, _z, absS in project_scores(dx, dy, Qf, kmax=KMAX):
                    out[root][b0:b0 + len(blk), sl, k] = absS
            print(f"[catalogue] {system} {root}: {min(b0 + BLOCK, nb)}/{nb} sources {time.time()-t0:.0f}s", flush=True)
    for j, g in enumerate(bank):
        for a in out.values():
            a[j, ai[g], :] = np.nan
    rows, _ = RU.gb_raw_rows(system) if rotation is None else ({}, None)  # parity only for the production Qf
    checked, worst = 0, 0.0
    for j, g in enumerate(bank):
        if g.upper() in rows:
            for root in RU.ROOTS:
                ref = np.asarray(rows[g.upper()][(root, "U")], float)[:, : KMAX + 1]
                d = np.nanmax(np.abs(ref - out[root][j])) if np.isfinite(ref).any() else 0.0
                if not np.array_equal(np.isfinite(ref), np.isfinite(out[root][j])) or d > TOL:
                    raise SystemExit(f"REFUSING_CATALOGUE_PARITY: {system}/{g}/{root} max|diff| {d}")
                worst = max(worst, float(d))
            checked += 1
    path = RU.system_dir(system) / ("catalogue_U_k0_8.npz" if rotation is None else f"catalogue_U_k0_8_rot{rotation}.npz")
    binding = {"method": "GB-catalogue-bank-U-k0..8", "inputs": RU.input_manifest(D),
               "config": {"bank": "substrate is_tf ∩ evaluator axis ∩ spliced detection >= .05", "det": DET,
                          "kmax": KMAX, "channel": "U", "rotation_seed": rotation,
                          "target_extension_sha256": RU.sha(RU.SCRATCH / "gb_target_extension.py")},
               "producer_sha256": RU.sha(Path(__file__).resolve()), "truth_read": False}
    binding["binding_sha256"] = RU.digest_object(binding)
    RU.write_part(path, binding, {"genes": np.asarray(axis), "sources": np.asarray(bank),
                                  **{f"{r}|U": a for r, a in out.items()}},
                  {"n_bank": nb, "parity_sources_checked": checked, "parity_max_abs_diff": worst,
                   "wall_s": time.time() - t0})
    print(f"[catalogue] {system}: bank {nb}, parity checked {checked} sources (max|diff| {worst:.2e}) -> {path}",
          flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[3]) if len(sys.argv) > 3 and sys.argv[2] == "--rotation" else None)
