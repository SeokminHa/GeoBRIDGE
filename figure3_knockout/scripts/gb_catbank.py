#!/usr/bin/env python
"""Catalogue bank: sources chosen by catalogue membership and observational eligibility only"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_primary as GP  # noqa: E402
import gb_real_uniform as RU  # noqa: E402

DET, KMAX, SMALLN, NPC = 0.05, 30, 5, 30
OUT = RU.OUT / "catbank"
ROOTS = GP.ROOTS
VARS = ("PU", "PM", "CU", "CM")
VNAME = {"PU": "P_pairU", "PM": "P_pairM", "CU": "P_cellU", "CM": "P_cellM"}
GB_METHODS = tuple(VNAME.values())
SEVEN = GB_METHODS + ("pearson", "genie3", "grnboost2")
EXTRA = ("regvelo", "velorama")                            # completed but not part of the strict common set T7 (partial target coverage)
EXT_SPECS = {  # method: (file, matrix key, row-names key, transform, required)
    "genie3": ("genie3_v2.npz", "score_all", "tfs", None, True),
    "grnboost2": ("grnboost2_v2.npz", "score_all", "tfs", None, True),
    "regvelo": ("regvelo_v2_full.npz", "score_all_signed", "regulators", "abs", False),
    "velorama": ("velorama_v2_native.npz", "score_all", "regulators", None, False),
    "regdiffusion": ("regdiffusion_v1.npz", "score_all", "tfs", None, False),
    "deepsem": ("deepsem_v1.npz", "score_all", "tfs", None, False),
    "pidc": ("pidc_v1.npz", "score_all", "tfs", None, False),
    "fasttenet": ("fasttenet_v1.npz", "score_all", "tfs", None, False),
}
SOTA_EXT = ("regdiffusion", "deepsem", "pidc", "fasttenet")             # gb_sota_ext.methods (same names); consumed by gb_canon_table / gb_layer_lattice


# utilities
def me_sha() -> str:
    return RU.sha(Path(__file__).resolve())


def stem(system: str, smalln: bool) -> str:
    return system.upper() + ("_smalln" if smalln else "")


def members_sha(members) -> str:
    import hashlib
    return hashlib.sha256("\n".join(members).encode()).hexdigest()


# receipt for a non-`write_part` output: script sha, input file sha256s, parameters; truth_read is always False
def write_receipt(path, config, inputs=None, extra=None):
    path = Path(path)
    ins = {k: (RU.sha(v) if v is not None and Path(v).exists() else None) for k, v in (inputs or {}).items()}
    RU.atomic_json(path.with_suffix(".receipt.json"), {
        "output": str(path), "output_sha256": RU.sha(path), "producer": str(Path(__file__).resolve()), "producer_sha256": me_sha(),
        "config": config, "input_sha256": ins, "truth_read": False,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **(extra or {})})


# sha-256 of the imported code that defines the algebra (a change in any of them invalidates every binding)
def imports_sha():
    import gb_target_extension as TE
    return {m.__name__: RU.sha(Path(m.__file__).resolve()) for m in (GP, RU, TE)}


def binding(D, cmd, config, extra_inputs=None):
    obj = {"method": f"catbank-{cmd}", "inputs": RU.input_manifest(D), "config": config, "extra_inputs": extra_inputs or {},
           "producer_sha256": me_sha(), "imports_sha256": imports_sha(), "truth_read": False}
    obj["binding_sha256"] = RU.digest_object(obj)
    return obj


# Copy of gb_cop_alignment.observational_universe (gb_cop_alignment.py:188-195) minus the hash
def obs_universe(src, rows, axis_upper):
    uni = np.asarray(axis_upper) != src.upper()
    for h in ROOTS:
        for c in ("U", "X"):
            uni &= np.isfinite(rows[(h, c)]).all(1)
    return uni


def parse_slice(s, n):
    i, j = s.split(":")
    i, j = (int(i) if i else 0), (int(j) if j else n)
    if not 0 <= i < j <= n:
        raise SystemExit(f"REFUSING_BAD_SOURCE_SLICE {s!r} for a bank of {n}")
    return i, j


# bank
# Catalogue TFs (substrate is_tf) on the fixture/evaluator axis with raw spliced detection >= DET (fraction of control cells with spliced count > 0)
def compute_bank(D):
    from collections import Counter
    axis, cols = list(D["axis"]), D["axis_cols"]
    is_tf = np.asarray(D["S"]["is_tf"], bool)[cols]
    det = np.asarray((D["A"].layers["spliced"][:, cols] > 0).mean(0)).ravel()
    cand = [g for g, t, d in zip(axis, is_tf, det) if t and d >= DET]
    cc = Counter(str(g).upper() for g in D["fixture_genes"])               # same fixture-wide rule as gb_tree_v2 / gb_regvelo_v2 / gb_velorama_v2
    members = [g for g in cand if cc[g.upper()] == 1]                       # drop_case_collisions: the whole colliding group is dropped
    dsel = dict(zip(axis, det))
    collided = {g: float(dsel[g]) for g in cand if cc[g.upper()] != 1}
    if len({g.upper() for g in members}) != len(members):
        raise SystemExit("REFUSING_BANK_CASE_COLLISION")
    mset = set(members)
    return {"members": members, "excluded_case_collision": collided, "detection": {g: float(d) for g, d in zip(axis, det) if g in mset},
            "below_floor": {g: float(d) for g, t, d in zip(axis, is_tf, det) if t and d < DET}, "n_catalogue_on_axis": int(is_tf.sum())}


def load_bank(system, D):
    import json
    p = OUT / f"{system}_bank.json"
    B = json.load(open(p))
    cur = compute_bank(D)
    if B["members"] != cur["members"] or B["sha256_members"] != members_sha(cur["members"]):
        raise SystemExit(f"REFUSING_BANK_STALE: {p} differs from the bank recomputed from the current inputs")
    return B


def cmd_bank(a):
    import gb_substrate as GS
    system = a.system.upper()
    D = RU.canonical_inputs(system)
    b = compute_bank(D)
    cat = GS.paths(system, canonical=True)[2]
    rec = {"system": system, "det_floor": DET, "rule": "substrate is_tf (species TF catalogue) ∩ evaluator axis ∩ spliced detection >= det_floor, minus fixture-wide "
                    "case-colliding symbols (the comparators' drop_case_collisions rule; observational, listed in excluded_case_collision)",
           **b, "n_members": len(b["members"]), "sha256_members": members_sha(b["members"]), "catalogue": str(cat), "catalogue_sha256": RU.sha(cat),
           "inputs": RU.input_manifest(D), "producer_sha256": me_sha(), "truth_read": False,
           "note": "no perturbation roster, regulon, ChIP or KO object is read; D['sources'] is not used"}
    OUT.mkdir(parents=True, exist_ok=True)
    RU.atomic_json(OUT / f"{system}_bank.json", rec)
    print(f"[catbank] bank {system}: {len(b['members'])} members of {b['n_catalogue_on_axis']} catalogue TFs on the axis; sha {rec['sha256_members'][:12]}")


# scoring primitives
# dx (n, ns) float32 source differences (pairs) or centred source vectors (cells)
def score_block(dx, get_y, G, Qf, kmax=KMAX, chunk=500):
    from gb_target_extension import EPS, project_scores
    ns = dx.shape[1]
    rU = np.full((ns, G, kmax + 1), np.nan, np.float32)
    rM = np.full_like(rU, np.nan)
    nU = np.full((ns, G), np.nan, np.float32)
    nM = np.full_like(nU, np.nan)
    for g0 in range(0, G, chunk):
        sl = slice(g0, min(g0 + chunk, G))
        yU, yM = get_y(sl)
        n = yU.shape[1]
        y = np.concatenate([yU, yM], axis=1).astype(np.float32)
        for k, _num, den, z2, absS in project_scores(dx, y, Qf, kmax=kmax):
            rU[:, sl, k], rM[:, sl, k] = absS[:, :n], absS[:, n:]
            if k == 0:
                ok = (den > EPS) & (z2 > 0)
                with np.errstate(invalid="ignore", divide="ignore"):
                    ne = np.where(ok, den ** 2 / np.where(z2 > 0, z2, 1.0), np.nan)
                nU[:, sl], nM[:, sl] = ne[:, :n], ne[:, n:]
    return rU, rM, nU, nM


# Raw {root: (G, K+1) |S|}, uni (G,) -> dict of (G,) vectors, NaN outside uni
def primary_diag(raw, uni):
    G = len(uni)
    put = lambda v: _scatter(v, uni, G)
    H = {h: GP.zcons({x: raw[h][uni] for x in ROOTS}) for h in ROOTS}
    P = GP.zcons({h: raw[h][uni] for h in ROOTS})
    if not np.array_equal(np.minimum(H[ROOTS[0]], H[ROOTS[1]]), P, equal_nan=True):
        raise SystemExit("REFUSING_ROOT_MIN_IDENTITY: zcons(two roots) != min of the per-root H")
    pm = 0.5 * (H[ROOTS[0]] + H[ROOTS[1]])
    dg = 0.5 * np.abs(H[ROOTS[0]] - H[ROOTS[1]])
    if not np.allclose(pm - dg, P, atol=1e-12, equal_nan=True):
        raise SystemExit("REFUSING_ROOT_DISAGREEMENT_IDENTITY: ½(H1+H2) - ½|H1-H2| != min(H1,H2)")
    return {"H|" + ROOTS[0]: put(H[ROOTS[0]]), "H|" + ROOTS[1]: put(H[ROOTS[1]]), "P": put(P), "Pmean": put(pm), "disag": put(dg)}


def _scatter(v, uni, G):
    out = np.full(G, np.nan)
    out[uni] = v
    return out


# parity
def _cmp(a, b, what, tol):
    a, b = np.asarray(a, float), np.asarray(b, float)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        raise SystemExit(f"REFUSING_PARITY_FINITENESS: {what}: {int((np.isfinite(a) != np.isfinite(b)).sum())} cells differ")
    f = np.isfinite(a)
    d = float(np.abs(a[f] - b[f]).max()) if f.any() else 0.0
    if d > tol:
        raise SystemExit(f"REFUSING_PARITY: {what}: max|diff| {d:.3e} > {tol:.1e}")
    return d


# fail-closed reproduction checks
def parity(system, D, srcs, R, a):
    au = np.char.upper(np.asarray(D["axis"]).astype(str))
    rep = {"catalogue": {"n": 0, "worst_raw": 0.0}, "production": {"n": 0, "worst_raw_U": 0.0, "worst_raw_M": 0.0, "worst_primary": 0.0,
                                                                     "universe_equal": True}, "cell": {"n": 0, "worst": 0.0}}
    cp = RU.system_dir(system) / "catalogue_U_k0_8.npz"
    if cp.exists():
        with np.load(cp, allow_pickle=True) as z:
            if [str(g) for g in z["genes"]] != list(D["axis"]):
                raise SystemExit("REFUSING_CATALOGUE_AXIS_MISMATCH")
            pos = {str(s): i for i, s in enumerate(z["sources"])}
            mats = {h: np.asarray(z[f"{h}|U"]) for h in ROOTS if any(s in pos for s in srcs)}
            for si, s in enumerate(srcs):
                if s in pos:
                    for h in ROOTS:
                        rep["catalogue"]["worst_raw"] = max(rep["catalogue"]["worst_raw"], _cmp(
                            R["raw|" + h + "|PU"][si, :, :9], mats[h][pos[s]][:, :9], f"catalogue {system}/{s}/{h}", a.tol_raw))
                    rep["catalogue"]["n"] += 1
    roster = {s.upper() for s in D["sources"]}
    hit = [s for s in srcs if s.upper() in roster]
    if hit:
        import gb_cop_alignment as CA
        rows, eg = RU.gb_raw_rows(system, srcs=[s.upper() for s in hit])
        if [str(g) for g in eg] != list(D["axis"]):
            raise SystemExit("REFUSING_PRODUCTION_AXIS_MISMATCH")
        kk = min(a.kmax, 30) + 1
        for s in hit:
            si, r = srcs.index(s), rows[s.upper()]
            for h in ROOTS:
                rep["production"]["worst_raw_U"] = max(rep["production"]["worst_raw_U"], _cmp(
                    R["raw|" + h + "|PU"][si][:, :kk], np.asarray(r[(h, "U")])[:, :kk], f"production U {system}/{s}/{h}", a.tol_raw))
                rep["production"]["worst_raw_M"] = max(rep["production"]["worst_raw_M"], _cmp(
                    R["raw|" + h + "|PM"][si][:, :kk], np.asarray(r[(h, "X")])[:, :kk], f"production X(M) {system}/{s}/{h}", a.tol_raw))
            if a.kmax == 30:
                uni_p, _ = CA.observational_universe(s.upper(), r, eg)
                if not (np.array_equal(uni_p, R["universe"][si]) and np.array_equal(uni_p, obs_universe(s, r, au))):
                    raise SystemExit(f"REFUSING_UNIVERSE_PARITY: {system}/{s}")
                pp = GP.zcons({h: np.asarray(r[(h, "U")])[uni_p] for h in ROOTS})
                rep["production"]["worst_primary"] = max(rep["production"]["worst_primary"], _cmp(
                    R["P|PU"][si][uni_p], pp, f"PRIMARY {system}/{s}", a.tol_primary))
            rep["production"]["n"] += 1
    if a.cell_parity_n > 0:
        import gb_adds as GA
        ctx = GA.context(system)
        if [str(g) for g in ctx["eg"]] != list(D["axis"]):
            raise SystemExit("REFUSING_GB_ADDS_AXIS_MISMATCH")
        for si, s in enumerate(srcs[:a.cell_parity_n]):
            ref, _r, _p = GA.cell_scores(ctx, s)
            keep = np.ones(len(au), bool)
            keep[np.flatnonzero(au == s.upper())] = False            # gb_adds does not blank the self target
            for ch, key in (("U", "CU"), ("M", "CM")):
                rep["cell"]["worst"] = max(rep["cell"]["worst"], _cmp(
                    R["raw|cell|" + key][si][keep, :9], ref[ch][keep, :9], f"cell twin {system}/{s}/{ch}", a.tol_cell))
            rep["cell"]["n"] += 1
    if rep["catalogue"]["n"] + rep["production"]["n"] == 0 and not a.allow_no_parity:
        raise SystemExit("REFUSING_NO_PARITY_REFERENCE: no shard source is in the catalogue bank file or the production caches "
                         "(pass --allow-no-parity to record an UNVERIFIED shard)")
    return rep


# scores
def cmd_scores(a):
    system = a.system.upper()
    t0 = time.time()
    D = RU.canonical_inputs(system)
    B = load_bank(system, D)
    members = B["members"]
    nb = len(members)
    i, j = (0, min(SMALLN, nb)) if a.smalln else parse_slice(a.sources, nb) if a.sources else (0, nb)
    srcs = members[i:j]
    outp = OUT / (f"{stem(system, True)}_scores.npz" if a.smalln else f"{system}_scores_{i}-{j}.npz")
    bd = binding(D, "scores", {"bank_sha256": B["sha256_members"], "i": i, "j": j, "kmax": a.kmax, "ks_primary": list(GP.KS),
                               "variants": {k: VNAME[k] for k in VARS}, "golden_sha256": GP.GOLDEN_SHA256, "det": DET,
                               "tol_raw": a.tol_raw, "tol_primary": a.tol_primary, "tol_cell": a.tol_cell,
                               "cell_parity_n": a.cell_parity_n, "allow_no_parity": a.allow_no_parity})
    if RU.valid_part(outp, bd) and not a.force:
        print(f"[catbank] reuse {outp}")
        return
    if not 8 <= a.kmax <= NPC:
        raise SystemExit("REFUSING_KMAX: need 8 <= kmax <= 30")
    axis = np.asarray(D["axis"]).astype(str)
    au = np.char.upper(axis)
    if len(set(au)) != len(au):
        raise SystemExit("REFUSING_AXIS_CASE_COLLISION")
    ai = {g: k for k, g in enumerate(axis)}
    fi = {g: k for k, g in enumerate(D["fixture_genes"])}
    cols, A, S = D["axis_cols"], D["A"], D["S"]
    G, ns, K = len(axis), len(srcs), a.kmax + 1
    Mu = RU.dense(A.layers["Mu"][:, cols], np.float32)
    Ms = RU.dense(A.layers["Ms"][:, cols], np.float32)
    src_ax = np.asarray([ai[g] for g in srcs])
    Xsrc = np.asarray(S["X"], np.float64)[:, [fi[g] for g in srcs]]
    if not np.allclose(Xsrc, Ms[:, src_ax], rtol=1e-5, atol=1e-6):
        raise SystemExit("REFUSING_SOURCE_MS_MISMATCH: V1 source Ms != fixture Ms on the evaluator axis")
    P30 = np.asarray(A.obsm["X_pca"], float)[:, :NPC]
    R = {}
    for ch in ("PU", "PM"):
        for h in ROOTS:
            R[f"raw|{h}|{ch}"] = np.full((ns, G, K), np.nan, np.float32)
            R[f"neff|{h}|{ch}"] = np.full((ns, G), np.nan, np.float32)
    for root in ROOTS:                                              # pair space (production algebra)
        I, J = (np.asarray(x, int) for x in S["pairs"][root])
        Qf, _ = np.linalg.qr(P30[J] - P30[I])
        for b0 in range(0, ns, a.block):
            b1 = min(b0 + a.block, ns)
            xb = Xsrc[:, b0:b1]
            dx = (xb[J] - xb[I]).astype(np.float32)
            rU, rM, nU, nM = score_block(dx, lambda sl: (Mu[J, sl] - Mu[I, sl], Ms[J, sl] - Ms[I, sl]), G, Qf, a.kmax)
            R[f"raw|{root}|PU"][b0:b1], R[f"raw|{root}|PM"][b0:b1] = rU, rM
            R[f"neff|{root}|PU"][b0:b1], R[f"neff|{root}|PM"][b0:b1] = nU, nM
            print(f"[catbank] {system} pair {root}: {b1}/{ns} sources {time.time()-t0:.0f}s", flush=True)
    mu_c, ms_c = Mu.astype(np.float64).mean(0), Ms.astype(np.float64).mean(0)   # cell space (root independent)
    Qc, _ = np.linalg.qr(P30 - P30.mean(0))
    R["raw|cell|CU"], R["raw|cell|CM"] = (np.full((ns, G, K), np.nan, np.float32) for _ in range(2))
    R["neff|cell|CU"], R["neff|cell|CM"] = (np.full((ns, G), np.nan, np.float32) for _ in range(2))
    for b0 in range(0, ns, a.block):
        b1 = min(b0 + a.block, ns)
        xb = Xsrc[:, b0:b1]
        xc = (xb - xb.mean(0)).astype(np.float32)
        rU, rM, nU, nM = score_block(xc, lambda sl: ((Mu[:, sl] - mu_c[sl]).astype(np.float32), (Ms[:, sl] - ms_c[sl]).astype(np.float32)),
                                     G, Qc, a.kmax)
        R["raw|cell|CU"][b0:b1], R["raw|cell|CM"][b0:b1] = rU, rM
        R["neff|cell|CU"][b0:b1], R["neff|cell|CM"][b0:b1] = nU, nM
    print(f"[catbank] {system} cell twin done {time.time()-t0:.0f}s", flush=True)
    for s in range(ns):                                             # self target never scores
        for key, v in R.items():
            if key.startswith(("raw|", "neff|")):
                v[s, src_ax[s]] = np.nan
    R["universe"] = np.zeros((ns, G), bool)
    for v in VARS:
        for key in ("P", "Pmean", "disag") + tuple("H|" + h for h in ROOTS):
            R[f"{key}|{v}"] = np.full((ns, G), np.nan)
    for s in range(ns):
        rows = {(h, c): R[f"raw|{h}|{ch}"][s] for h in ROOTS for c, ch in (("U", "PU"), ("X", "PM"))}
        R["universe"][s] = uni = obs_universe(srcs[s], rows, au)
        for v in VARS:
            raw = {h: (R[f"raw|cell|{v}"][s] if v[0] == "C" else R[f"raw|{h}|{v}"][s]) for h in ROOTS}   # cell: same array at both roots
            for key, vec in primary_diag(raw, uni).items():
                R[f"{key}|{v}"][s] = vec
    par = parity(system, D, srcs, R, a)
    verified = par["catalogue"]["n"] + par["production"]["n"] > 0        # False only when --allow-no-parity let an unreferenced shard through
    print(f"[catbank] parity OK (verified={verified}) {par}", flush=True)
    summ = {}
    for s in range(ns):
        u = R["universe"][s]
        summ[srcs[s]] = {"n_universe": int(u.sum()), **{VNAME[v]: int(np.isfinite(R[f"P|{v}"][s]).sum()) for v in VARS},
                         "mean_disag_pairU": float(np.nanmean(R["disag|PU"][s])) if np.isfinite(R["disag|PU"][s]).any() else None}
    arrays = {"genes": axis, "sources": np.asarray(srcs), "bank_idx": np.arange(i, j), "src_axis_idx": src_ax,
              "ks_primary": np.asarray(GP.KS), **R}
    RU.write_part(outp, bd, arrays, {"bank_sha256": B["sha256_members"], "i": i, "j": j, "kmax": a.kmax, "parity": par,
                                     "cell_variants_root_independent": True, "summary": summ, "wall_s": time.time() - t0,
                                     "production_universe": a.kmax == 30, "parity_verified": bool(verified),
                                     "neff_w_depth": "k=0 only",
                                     "note": "cell variants: raw stored once (root independent); H/P/Pmean/disag rows duplicated across roots"})
    print(f"[catbank] {system}: wrote {outp} ({ns} sources) {time.time()-t0:.0f}s", flush=True)
    for s in srcs[:5]:
        print(f"  {s}: {summ[s]}", flush=True)


# external
class MissingRows(SystemExit):
    pass


# Rows of a catalogue-wide matrix for `members` (case-insensitive row match) and columns for `axis` (exact)
def pick_rows(M, row_names, col_genes, members, axis, method):
    rn = [str(x).upper() for x in row_names]
    if len(set(rn)) != len(rn):
        raise SystemExit(f"REFUSING_EXT_ROW_CASE_COLLISION: {method}")
    ri = {n: i for i, n in enumerate(rn)}
    cp = {str(g): i for i, g in enumerate(col_genes)}
    miss_r = [g for g in members if g.upper() not in ri]
    miss_c = [g for g in axis if g not in cp]
    if miss_r or miss_c:
        e = MissingRows(f"REFUSING_EXTERNAL_MISSING: {method}: {len(miss_r)} bank rows and {len(miss_c)} axis columns missing")
        e.rows, e.cols = miss_r, miss_c
        raise e
    return np.asarray(M, np.float64)[np.ix_([ri[g.upper()] for g in members], [cp[str(g)] for g in axis])]


# None if the comparator artifact is bound to this fixture and unmodified (its own receipt: fixture_sha256 and output_sha256), else the reason
def ext_artifact_problem(p, fixture_sha):
    import json
    rp = Path(p).with_suffix(".receipt.json")
    if not rp.exists():
        return f"no receipt {rp.name}"
    r = json.load(open(rp))
    if r.get("output_sha256") != RU.sha(p):
        return "artifact sha256 != receipt output_sha256"
    fx = ((r.get("binding") or {}).get("inputs") or {}).get("fixture_sha256")
    if fx != fixture_sha:
        return f"receipt fixture_sha256 {str(fx)[:12]} != current fixture {fixture_sha[:12]}"
    return None


def cmd_external(a):
    import json
    system = a.system.upper()
    D = RU.canonical_inputs(system)
    B = load_bank(system, D)
    members = B["members"][:SMALLN] if a.smalln else B["members"]
    axis = [str(g) for g in D["axis"]]
    d = RU.system_dir(system)
    mats, fail, notdone, cov, optional = {}, {}, [], {}, {}
    fxsha = RU.sha(D["fixture"])
    D2 = dict(D)                                                       # |Pearson| for the bank (pearson.npz only holds the cached roster)
    D2["sources"], D2["source_axis_idx"] = members, np.asarray([axis.index(g) for g in members], int)
    mats["pearson"] = RU.pearson_scores(D2).astype(np.float64)
    pp = d / "pearson.npz"
    if pp.exists():                                                    # parity vs the existing roster rows (same function, must be identical)
        with np.load(pp, allow_pickle=True) as z:
            ps = {str(s).upper(): i for i, s in enumerate(z["sources"])}
            old = np.asarray(z["score"])
            chk = [(k, ps[g.upper()]) for k, g in enumerate(members) if g.upper() in ps]
            worst = max((_cmp(mats["pearson"][k], old[r], f"pearson {members[k]}", 1e-6) for k, r in chk), default=0.0)
        cov["pearson_parity"] = {"n": len(chk), "worst": worst}
    for m, (fn, key, nk, tf, req) in EXT_SPECS.items():
        p = d / fn
        prob = "file missing" if not p.exists() else ext_artifact_problem(p, fxsha)
        if prob:
            if req:
                fail[m] = {"artifact_problem": f"{prob}: {p}"}
            else:
                notdone.append(m)
                optional[m] = {"artifact_problem": f"{prob}: {p}"}
            continue
        with np.load(p, allow_pickle=True) as z:
            try:
                x = pick_rows(z[key], z[nk].astype(str), z["genes"].astype(str), members, axis, m)
            except MissingRows as e:
                if req:
                    fail[m] = {"missing_rows": e.rows, "missing_axis_cols": e.cols}
                else:                                                   # optional comparator incomplete: recorded (not used), never silently dropped
                    notdone.append(m)
                    optional[m] = {"missing_rows": e.rows, "missing_axis_cols": e.cols}
                continue
        mats[m] = np.abs(x) if tf == "abs" else x
    for m, x in mats.items():
        fr = np.isfinite(x).mean(1)
        cov[m] = {"row_finite_frac_min": float(fr.min()), "row_finite_frac_median": float(np.median(fr)),
                  "all_nan_rows": [members[k] for k in np.flatnonzero(fr == 0)], "axis_target_coverage": float(np.isfinite(x).any(0).mean())}
    if fail:
        RU.atomic_json(OUT / f"{stem(system, a.smalln)}_external_FAILURES.json", {"failures": fail, "not_completed_optional": optional})
        raise SystemExit(f"REFUSING_EXTERNAL_MISSING_ROWS: {json.dumps({m: {k: (v if isinstance(v, str) else len(v)) for k, v in f.items()} for m, f in fail.items()})} "
                         f"(listed in {stem(system, a.smalln)}_external_FAILURES.json; rows are never silently dropped)")
    outp = OUT / f"{stem(system, a.smalln)}_external.npz"
    OUT.mkdir(parents=True, exist_ok=True)
    RU.atomic_npz(outp, members=np.asarray(members), axis=np.asarray(axis), methods=np.asarray(list(mats)), **{f"M|{m}": x for m, x in mats.items()})
    write_receipt(outp, {"members_sha256": members_sha(members), "bank_sha256": B["sha256_members"], "smalln": a.smalln,
                         "specs": {m: list(v[:4]) for m, v in EXT_SPECS.items()}},
                  {"fixture": D["fixture"], **{m: d / v[0] for m, v in EXT_SPECS.items()}, "pearson_existing": pp},
                  {"coverage": cov, "not_completed_optional": optional, "n_members": len(members), "n_axis": len(axis),
                   "fixture_sha256": fxsha, "bank_sha256": B["sha256_members"]})
    print(f"[catbank] external {system}: {sorted(mats)} for {len(members)} bank rows; not completed: {notdone}")


# priors
# per-row mean / population sd over the strict common target set T (NaN if < 3 finite or sd == 0)
def row_consts(x, T):
    mu, sd = np.full(len(x), np.nan), np.full(len(x), np.nan)
    for r in range(len(x)):
        v = x[r, T]
        if np.isfinite(v).sum() > 2 and np.nanstd(v) > 0:
            mu[r], sd[r] = np.nanmean(v), np.nanstd(v)
    return mu, sd


# sum and count of finite row-z entries per target (float64)
def prior_sums(Z):
    f = np.isfinite(Z)
    return np.where(f, Z, 0.0).sum(0), f.sum(0)


# P^(-r)(g) = mean over the other rows' finite Z(., g) = (S - Z[r]) / (C - finite(Z[r]))
def loo_prior(Z, S, C, r, min_contrib=2):
    if r is None:
        s, c = S, C
    else:
        f = np.isfinite(Z[r])
        s, c = S - np.where(f, Z[r], 0.0), C - f
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(c >= min_contrib, s / np.maximum(c, 1), np.nan)


# Mats {method: (nb, G)} with rows in bank order
def build_set(mats, bank_cols, min_contrib=2, diag=None, names=None):
    diag = diag or {}
    ms = list(mats)
    nb, G = mats[ms[0]].shape
    nm = list(names) if names is not None else [str(i) for i in range(nb)]
    selfm = np.zeros((nb, G), bool)
    selfm[np.arange(nb), bank_cols] = True
    forced, X = {}, {}
    for m in ms + list(diag):
        x = np.array(mats[m] if m in mats else diag[m], float)
        forced[m] = int(np.isfinite(x[selfm]).sum())
        x[selfm] = np.nan
        X[m] = x
    empty = {m: [nm[r] for r in np.flatnonzero(~np.isfinite(X[m]).any(1))] for m in X}
    empty = {m: v for m, v in empty.items() if v}
    if empty:
        raise SystemExit(f"REFUSING_ALL_NAN_ROWS (checked before T; rows are never dropped): {empty}")
    fin = {m: np.isfinite(X[m]) for m in ms}
    T = np.all([f.all(0) for f in fin.values()], 0)
    Trel = np.all([(f | selfm).all(0) for f in fin.values()], 0)
    lost = {m: int((np.all([fin[o].all(0) for o in ms if o != m], 0) & ~fin[m].all(0)).sum()) if len(ms) > 1 else 0 for m in ms}
    if T.sum() < 3:
        raise SystemExit(f"REFUSING_T_TOO_SMALL: strict common finite target set has {int(T.sum())} genes; targets lost by that method alone: {lost}")
    out = {"T": T, "Trel": Trel, "forced_self_nan": forced, "methods": {}, "n_contrib_min": min_contrib, "lost_only_by": lost, "diag": list(diag)}
    bad, notfin = {}, {}
    for m in X:
        mu, sd = row_consts(X[m], T)
        if m in diag and not np.isfinite(X[m][:, T]).all():
            notfin[m] = int((~np.isfinite(X[m][:, T])).sum())
        if (~np.isfinite(mu)).any():
            bad[m] = [nm[r] for r in np.flatnonzero(~np.isfinite(mu))]
        out["methods"][m] = {"mu": mu, "sd": sd}
    if notfin:
        raise SystemExit(f"REFUSING_DIAG_NOT_FINITE_ON_T (diagnostics cannot change T): {notfin}")
    if bad:
        raise SystemExit(f"REFUSING_BAD_ROWS (constant/degenerate on T; rows are never dropped): {bad}")
    for m, r in out["methods"].items():
        Z = (X[m][:, T] - r["mu"][:, None]) / r["sd"][:, None]
        Zr = (X[m][:, Trel] - r["mu"][:, None]) / r["sd"][:, None]
        r["Z"], r["Zr"] = Z, Zr
        r["S"], r["C"] = prior_sums(Z)
        r["Sr"], r["Cr"] = prior_sums(Zr)
    return out


def shard_pattern(system):
    import re
    return re.compile(rf"{re.escape(system)}_scores_\d+-\d+\.npz")          # == ^SYSTEM_scores_\d+-\d+\.npz$ (fullmatch)


# gather the score shards in bank order; refuses stale/mixed shards and any bank row not covered exactly once
def load_gb(system, smalln, members, B):
    import json
    nb = len(members)
    pat = shard_pattern(system)
    files = [OUT / f"{stem(system, True)}_scores.npz"] if smalln else sorted(f for f in OUT.iterdir() if pat.fullmatch(f.name))
    if not files:
        raise SystemExit(f"REFUSING_NO_SCORE_SHARDS for {system}")
    mats, seen, prod, kmax, used = None, np.zeros(nb, int), set(), set(), []
    for f in files:
        rec = json.load(open(f.with_suffix(".receipt.json")))
        if rec["output_sha256"] != RU.sha(f) or rec["binding"]["config"]["bank_sha256"] != B["sha256_members"]:
            raise SystemExit(f"REFUSING_STALE_SHARD: {f}")
        if any(rec["binding"]["inputs"].get(k) != B["inputs"].get(k) for k in ("fixture_sha256", "pairs_sha256", "axis_sha256")):
            raise SystemExit(f"REFUSING_SHARD_INPUTS_DIFFER_FROM_BANK: {f}")
        if rec.get("production_universe") is not True or rec.get("parity_verified") is not True:
            raise SystemExit(f"REFUSING_UNVERIFIED_OR_NONPRODUCTION_SHARD: {f} (production_universe={rec.get('production_universe')}, "
                             f"parity_verified={rec.get('parity_verified')})")
        prod.add((rec["binding"]["producer_sha256"], json.dumps(rec["binding"].get("imports_sha256"), sort_keys=True)))
        kmax.add(rec["kmax"])
        with np.load(f, allow_pickle=True) as z:
            idx = np.asarray(z["bank_idx"], int)
            if idx.max() >= nb or [str(s) for s in z["sources"]] != [members[i] for i in idx]:
                raise SystemExit(f"REFUSING_SHARD_SOURCES_DIFFER_FROM_BANK: {f}")
            if mats is None:
                genes = [str(g) for g in z["genes"]]
                mats = {v: np.full((nb, len(genes)), np.nan) for v in VARS}
                dg = {f"{t}_{VNAME[v][2:]}": np.full((nb, len(genes)), np.nan) for v in ("PU", "PM") for t in ("H1", "H2", "Pmean")}
            for v in VARS:
                mats[v][idx] = z[f"P|{v}"]
            for v in ("PU", "PM"):                                     # root diagnostics: per-root H (pre-min) and the root mean
                dg[f"H1_{VNAME[v][2:]}"][idx] = z[f"H|{ROOTS[0]}|{v}"]
                dg[f"H2_{VNAME[v][2:]}"][idx] = z[f"H|{ROOTS[1]}|{v}"]
                dg[f"Pmean_{VNAME[v][2:]}"][idx] = z[f"Pmean|{v}"]
            seen[idx] += 1
        used.append(str(f))
    if not (seen == 1).all():
        raise SystemExit(f"REFUSING_SHARD_COVERAGE: {int((seen != 1).sum())} of {nb} bank rows not covered exactly once")
    if len(prod) != 1 or len(kmax) != 1:
        raise SystemExit(f"REFUSING_MIXED_SHARDS: producer shas {prod}, kmax {kmax}")
    return genes, {VNAME[v]: mats[v] for v in VARS}, dg, used, {"producer_and_imports": sorted(prod), "kmax": sorted(kmax)}


def cmd_priors(a):
    import json
    system = a.system.upper()
    B = json.load(open(OUT / f"{system}_bank.json"))
    if B["sha256_members"] != members_sha(B["members"]):                # bank currency: self-consistent, and (below) every shard/external artifact is bound to it
        raise SystemExit("REFUSING_BANK_INCONSISTENT")
    members = B["members"][:SMALLN] if a.smalln else B["members"]
    genes, mats, dg, shards, meta = load_gb(system, a.smalln, members, B)
    ep = OUT / f"{stem(system, a.smalln)}_external.npz"
    er = json.load(open(ep.with_suffix(".receipt.json")))
    if er["output_sha256"] != RU.sha(ep) or er.get("bank_sha256") != B["sha256_members"] or er.get("fixture_sha256") != B["inputs"]["fixture_sha256"]:
        raise SystemExit("REFUSING_EXTERNAL_RECEIPT_STALE: output sha / bank sha / fixture sha differ from the current bank")
    with np.load(ep, allow_pickle=True) as z:
        if [str(x) for x in z["members"]] != members or [str(x) for x in z["axis"]] != genes:
            raise SystemExit("REFUSING_EXTERNAL_BANK_OR_AXIS_MISMATCH")
        ext = {str(m): np.asarray(z[f"M|{m}"], float) for m in z["methods"]}
    gi = {g: i for i, g in enumerate(genes)}
    bank_cols = np.asarray([gi[g] for g in members])
    allm = {**mats, **{m: ext[m] for m in ("pearson", "genie3", "grnboost2") if m in ext}}
    miss = [m for m in SEVEN if m not in allm]
    if miss:
        raise SystemExit(f"REFUSING_METHODS_MISSING: {miss}")
    seven = {m: allm[m] for m in SEVEN}
    sets = {"T7": (seven, dg)}                                            # root diagnostics ride on the frozen T7 only
    for x in EXTRA:
        if x in ext:
            sets[f"T7+{x}"] = ({**seven, x: ext[x]}, None)              # per-extra-method set: strict over the seven and x
    arrays, info = {"members": np.asarray(members), "axis": np.asarray(genes)}, {}
    for name, (mm, dd) in sets.items():
        try:
            R = build_set(mm, bank_cols, a.min_contrib, diag=dd, names=members)
        except SystemExit as e:
            if name == "T7":
                raise
            info[name] = {"status": f"FAILED: {e}"}                     # a partial-coverage extra may be empty / degenerate; T7 may not
            continue
        T, Tr = R["T"], R["Trel"]
        keepx = (Tr & ~T)[Tr]                                           # columns of Zr that are not in T (regulator genes)
        arrays[f"{name}|genes"] = np.asarray(genes)[T]
        arrays[f"{name}|relaxed_genes"] = np.asarray(genes)[Tr]
        arrays[f"{name}|relaxed_is_extra"] = keepx
        for m, r in R["methods"].items():
            for k in ("mu", "sd", "Z", "S", "C", "Sr", "Cr"):
                arrays[f"{name}|{m}|{k}"] = r[k]
            arrays[f"{name}|{m}|Zx"] = r["Zr"][:, keepx]                # relaxed Z on regulator-gene targets only; on T it equals Z
            arrays[f"{name}|{m}|Pfull"] = loo_prior(r["Z"], r["S"], r["C"], None, a.min_contrib)
            arrays[f"{name}|{m}|Pfull_relaxed"] = loo_prior(None, r["Sr"], r["Cr"], None, a.min_contrib)
        info[name] = {"status": "OK", "n_T": int(T.sum()), "n_T_relaxed": int(Tr.sum()), "n_axis": len(genes), "n_bank": len(members),
                      "methods": list(mm), "diagnostic_methods": R["diag"], "forced_self_nan": R["forced_self_nan"],
                      "lost_only_by_method": R["lost_only_by"],
                      "min_contributors_full": {m: int(r["C"].min()) for m, r in R["methods"].items()},
                      "min_contributors_relaxed": {m: int(r["Cr"].min()) for m, r in R["methods"].items()},
                      "n_targets_lt_min_contrib_after_focal_out": {m: int((r["C"] - 1 < a.min_contrib).sum()) for m, r in R["methods"].items()}}
        print(f"[catbank] priors {system} {name}: T {int(T.sum())} / relaxed {int(Tr.sum())} of {len(genes)} genes, bank {len(members)}", flush=True)
    outp = OUT / f"{stem(system, a.smalln)}_priors.npz"
    RU.atomic_npz(outp, **arrays)
    write_receipt(outp, {"min_contrib": a.min_contrib, "smalln": a.smalln, "bank_sha256": B["sha256_members"], "seven": SEVEN, "extra": EXTRA,
                         "row_z": "population-sd z over T (gb_neutral.zrow semantics; NOT production depth-z)",
                         "loo": "loo_prior(Z,S,C,r): exact from stored float64 (Z,S,C); no precomputed leave-focal-out matrices",
                         "diagnostic_priors": "H1/H2/Pmean of P_pairU, P_pairM row-z'd on the frozen T7 (cannot change T)"},
                  {"external": ep, "bank": OUT / f"{system}_bank.json", **{f"shard{i}": s for i, s in enumerate(shards)}}, {"sets": info, "shards": meta})


# Tier-1 SOTA (RNA-only) target priors on the frozen T7 target set T (bool over the axis), constructed exactly as build_set builds the generic priors
def sota_prior_arrays(X, bank_cols, T, min_contrib=2):
    X = np.array(X, float)
    X[np.arange(len(X)), bank_cols] = np.nan
    mu, sd = row_consts(X, T)
    with np.errstate(invalid="ignore", divide="ignore"):
        Z = (X[:, T] - mu[:, None]) / sd[:, None]
    S, C = prior_sums(Z)
    return {"mu": mu, "sd": sd, "Z": Z, "S": S, "C": C, "Pfull": loo_prior(Z, S, C, None, min_contrib), "bad_rows": np.flatnonzero(~np.isfinite(mu))}


# {system}_sota_priors.npz: target priors of the optional SOTA comparators
def cmd_sota(a):
    import json
    system = a.system.upper()
    B = json.load(open(OUT / f"{system}_bank.json"))
    if B["sha256_members"] != members_sha(B["members"]):
        raise SystemExit("REFUSING_BANK_INCONSISTENT")
    members = B["members"][:SMALLN] if a.smalln else B["members"]
    pf = OUT / f"{stem(system, a.smalln)}_priors.npz"
    pr = json.load(open(pf.with_suffix(".receipt.json")))
    if pr["output_sha256"] != RU.sha(pf) or pr["config"]["bank_sha256"] != B["sha256_members"]:
        raise SystemExit("REFUSING_PRIORS_RECEIPT_STALE")
    with np.load(pf, allow_pickle=True) as z:
        axis, Tg = [str(x) for x in z["axis"]], [str(x) for x in z["T7|genes"]]
        if [str(x) for x in z["members"]] != members:
            raise SystemExit("REFUSING_PRIORS_MEMBERS_DIFFER_FROM_BANK")
    T = np.isin(axis, Tg)
    if [g for g, t in zip(axis, T) if t] != Tg:
        raise SystemExit("REFUSING_T7_NOT_A_SUBSEQUENCE_OF_THE_AXIS")
    gi = {g: i for i, g in enumerate(axis)}
    bank_cols = np.asarray([gi[g] for g in members])
    d = RU.system_dir(system)
    out, optional = {}, {}
    for m in SOTA_EXT:
        fn, key, nk, _, _ = EXT_SPECS[m]
        p = d / fn
        prob = "file missing" if not p.exists() else ext_artifact_problem(p, B["inputs"]["fixture_sha256"])
        if prob:
            optional[m] = {"artifact_problem": f"{prob}: {p}"}
            continue
        with np.load(p, allow_pickle=True) as z:
            try:
                X = pick_rows(z[key], z[nk].astype(str), z["genes"].astype(str), members, axis, m)
            except MissingRows as e:
                optional[m] = {"missing_rows": e.rows, "missing_axis_cols": e.cols}
                continue
        out[m] = sota_prior_arrays(X, bank_cols, T, a.min_contrib)
    arrays = {"members": np.asarray(members), "genes": np.asarray(Tg), "methods": np.asarray(list(out))}
    diag = {}
    for m, r in out.items():
        for k in ("mu", "sd", "Z", "S", "C", "Pfull"):
            arrays[f"{m}|{k}"] = r[k]
        diag[m] = {"bad_rows": [members[i] for i in r["bad_rows"]], "n_targets_nan_full_prior": int((~np.isfinite(r["Pfull"])).sum()), "n_T": int(T.sum()),
                   "min_contributors": int(r["C"].min()), "frac_finite_Z": float(np.isfinite(r["Z"]).mean())}
    outp = OUT / f"{stem(system, a.smalln)}_sota_priors.npz"
    RU.atomic_npz(outp, **arrays)
    write_receipt(outp, {"min_contrib": a.min_contrib, "smalln": a.smalln, "bank_sha256": B["sha256_members"], "methods": list(SOTA_EXT),
                         "row_z": "population-sd z over the FROZEN T7 (same constants as build_set; NaN-tolerant, T never changed)",
                         "loo": "loo_prior(Z,S,C,r) from the stored float64 (Z,S,C), as the generic priors"},
                  {"priors": pf, "bank": OUT / f"{system}_bank.json", **{m: d / EXT_SPECS[m][0] for m in out}},
                  {"diagnostics": diag, "not_completed_optional": optional, "n_members": len(members), "n_axis": len(axis), "bank_sha256": B["sha256_members"],
                   "priors_sha256": RU.sha(pf)})
    print(f"[catbank] sota priors {system}: {sorted(out)} not completed: {sorted(optional)}; {json.dumps(diag)}")


# features
# |Spearman(M[:, g], pt)| per column (average ranks)
# NaN for a constant column
def abs_spearman_cols(M, pt, block=512):
    from scipy.stats import rankdata
    r = rankdata(pt)
    r = r - r.mean()
    out = np.full(M.shape[1], np.nan)
    for g0 in range(0, M.shape[1], block):
        Rk = rankdata(np.asarray(M[:, g0:g0 + block], np.float64), axis=0)
        Rk = Rk - Rk.mean(0)
        den = np.sqrt((Rk ** 2).sum(0) * (r ** 2).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            out[g0:g0 + block] = np.abs(r @ Rk / den)
    out[~np.isfinite(out)] = np.nan
    return out


def cmd_features(a):
    import anndata as ad
    import pandas as pd
    import scanpy as sc
    import gb_decomp as GD
    system = a.system.upper()
    D = RU.canonical_inputs(system)
    A, S = D["A"], D["S"]
    F = GD.feats(system)                                              # F0 = N + D exactly as gb_decomp (raw_det, sm_det, abundance, pair_support, dynamics)
    genes = list(D["fixture_genes"])
    if list(F.index) != genes:
        raise SystemExit("REFUSING_F0_INDEX")
    Mu = RU.dense(A.layers["Mu"], np.float64)
    Ms = RU.dense(A.layers["Ms"], np.float64)
    Un = RU.dense(A.layers["unspliced"], np.float64)
    Sp = RU.dense(A.layers["spliced"], np.float64)
    F0cols = list(F.columns)
    F["raw_det_u"] = (Un > 0).mean(0)
    F["mu_det"] = (Mu > 0).mean(0)
    F["mu_abundance"] = np.log1p(Mu.mean(0))
    F["mu_logvar"] = np.log1p(Mu.var(0))
    dyn, sup = [], []
    for root in RU.ROOTS:                                              # same definitions as gb_decomp.feats, on Mu, in gene blocks
        I, J = (np.asarray(x, int) for x in S["pairs"][root])
        dd, ss = np.empty(Mu.shape[1]), np.empty(Mu.shape[1])
        for g0 in range(0, Mu.shape[1], 1000):
            dm = Mu[J, g0:g0 + 1000] - Mu[I, g0:g0 + 1000]
            with np.errstate(invalid="ignore", divide="ignore"):
                dd[g0:g0 + 1000] = np.std(dm, 0) / np.std(Mu[:, g0:g0 + 1000], 0)
            ss[g0:g0 + 1000] = np.mean(dm != 0, 0)
        dyn.append(dd)
        sup.append(ss)
    F["mu_pair_support"] = np.mean(sup, 0)
    F["mu_dynamics"] = np.mean(dyn, 0)
    z = np.load(D["pairs_file"], allow_pickle=True)
    sp_ms, sp_mu = [], []
    for root in RU.ROOTS:
        pt = np.asarray(z[f"{root}_pt"], float)                        # the frozen pair-artifact ordering variable (DPT_window_cap)
        if pt.shape != (Ms.shape[0],) or not np.isfinite(pt).all():
            raise SystemExit(f"REFUSING_PT_INVALID {root}")
        sp_ms.append(abs_spearman_cols(Ms, pt))
        sp_mu.append(abs_spearman_cols(Mu, pt))
    F["abs_spearman_ms_dpt"] = np.mean(sp_ms, 0)
    F["abs_spearman_mu_dpt"] = np.mean(sp_mu, 0)
    tmp = ad.AnnData(X=RU.dense(A.X, np.float32))                     # control log-normalised counts; scanpy seurat flavour (expm1 inside)
    hv = sc.pp.highly_variable_genes(tmp, flavor="seurat", n_bins=20, inplace=False)
    F["hvg_mean"], F["hvg_disp"], F["hvg_disp_norm"] = (hv[c].to_numpy(float) for c in ("means", "dispersions", "dispersions_norm"))
    mS = Sp.mean(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        F["cv2_spliced_norm"] = np.where(mS > 0, Sp.var(0) / mS ** 2, np.nan)      # on the fixture's normalised spliced layer (no raw counts here)
        F["us_ratio_norm"] = Un.sum(0) / (Un.sum(0) + Sp.sum(0))                    # on the normalised unspliced / spliced layers
    F = F.replace([np.inf, -np.inf], np.nan)                          # undefined features remain missing
    f1cols = F0cols + ["raw_det_u", "mu_det", "mu_abundance", "mu_logvar", "mu_pair_support", "mu_dynamics", "abs_spearman_ms_dpt",
                       "abs_spearman_mu_dpt"]                          
    f1xcols = f1cols + ["hvg_disp_norm", "cv2_spliced_norm", "us_ratio_norm"]  # tier F1x (auxiliary; own completeness flag)
    F["F0_complete"] = np.isfinite(F[F0cols].to_numpy()).all(1)
    F["F1_complete"] = np.isfinite(F[f1cols].to_numpy()).all(1)
    F["F1x_complete"] = np.isfinite(F[f1xcols].to_numpy()).all(1)
    OUT.mkdir(parents=True, exist_ok=True)
    outp = OUT / f"{stem(system, a.smalln)}_features.tsv"
    F.to_csv(outp, sep="\t", index_label="gene")
    write_receipt(outp, {"F0": F0cols, "F1": f1cols, "F1x": f1xcols, "smalln": a.smalln,
                         "F1x_layers": "cv2/us_ratio computed on normalised spliced/unspliced layers (raw integer counts are not in the fixture)",
                         "hvg": {"flavor": "seurat", "n_bins": 20, "scanpy": sc.__version__},
                         "pt": "pair-artifact {root}_pt", "spearman": "average ranks, root mean of |rho|"},
                  {"fixture": D["fixture"], "pairs": D["pairs_file"], "gb_decomp": Path(GD.__file__)},
                  {"n_genes": len(F), "n_F0_complete": int(F.F0_complete.sum()), "n_F1_complete": int(F.F1_complete.sum()),
                   "n_F1x_complete": int(F.F1x_complete.sum()),
                   "undefined_by_feature": {c: int((~np.isfinite(F[c])).sum()) for c in f1xcols}})
    print(f"[catbank] features {system}: {len(F)} genes, F0 complete {int(F.F0_complete.sum())}, F1 complete {int(F.F1_complete.sum())}, F1x complete {int(F.F1x_complete.sum())}")


# selftest
def selftest():
    import types
    rng = np.random.default_rng(1)
    try:
        import numexpr  # noqa: F401
    except ImportError:                                               # numpy stand-in that reproduces numexpr's float64 promotion (selftest only)
        def _ev(expr, out=None, casting=None, **kw):
            loc = {k: (v.astype(np.float64) if isinstance(v, np.ndarray) and v.dtype == np.float32 else v) for k, v in sys._getframe(1).f_locals.items()}
            val = eval(expr, {"abs": np.abs}, loc)
            np.copyto(out, val, casting="unsafe")
            return out
        sys.modules["numexpr"] = types.SimpleNamespace(evaluate=_ev)
    # 0. SOTA prior construction == build_set's for a fully finite matrix
    nb0, G0 = 6, 40
    X0 = rng.normal(size=(nb0, G0))
    bc0 = np.arange(nb0)
    R0 = build_set({"m": X0.copy()}, bc0, 2, names=[f"r{i}" for i in range(nb0)])
    A0 = sota_prior_arrays(X0, bc0, R0["T"])
    assert np.allclose(A0["Z"], R0["methods"]["m"]["Z"]) and np.allclose(A0["S"], R0["methods"]["m"]["S"]) and np.array_equal(A0["C"], R0["methods"]["m"]["C"])
    assert np.allclose(A0["Pfull"], loo_prior(R0["methods"]["m"]["Z"], R0["methods"]["m"]["S"], R0["methods"]["m"]["C"], None), equal_nan=True) and len(A0["bad_rows"]) == 0
    X1 = X0.copy()
    X1[:, 10] = np.nan                                                 # a target the method cannot score -> NaN prior for that target only
    X1[2, :] = 1.0                                                     # degenerate row -> all-NaN Z row, listed
    A1 = sota_prior_arrays(X1, bc0, R0["T"])
    j10 = np.flatnonzero(R0["T"]).tolist().index(10)
    assert A1["bad_rows"].tolist() == [2] and np.isnan(A1["Z"][2]).all() and A1["Z"].shape == A0["Z"].shape
    assert np.isnan(A1["Pfull"][j10]) and np.isfinite(np.delete(A1["Pfull"], j10)).all()
    print("ok 0: SOTA priors mirror build_set; NaN targets tolerated; degenerate rows listed")
    # 1. leave-focal-out == mean over the other rows (with NaNs, and the < 2 contributor rule)
    nb, T = 9, 25
    Z = rng.normal(size=(nb, T))
    Z[2, 3] = np.nan
    Z[:, 7] = np.nan
    Z[:6, 8] = np.nan                                                   # 3 contributors left; removing one focal leaves 2; removing none of NaNs stays >= 2
    S, C = prior_sums(Z)
    for r in range(nb):
        direct = np.array([np.nanmean(np.delete(Z, r, 0)[:, g]) if np.isfinite(np.delete(Z, r, 0)[:, g]).sum() >= 2 else np.nan
                           for g in range(T)])
        assert np.allclose(loo_prior(Z, S, C, r), direct, atol=1e-12, equal_nan=True), r
    assert np.isnan(loo_prior(Z, S, C, None)[7]) and np.isfinite(loo_prior(Z, S, C, None)[8])
    Z2 = Z.copy()
    Z2[:7, 8] = np.nan                                                  # 2 contributors: full prior defined, focal-out of one of them undefined
    S2, C2 = prior_sums(Z2)
    assert np.isfinite(loo_prior(Z2, S2, C2, None)[8]) and np.isnan(loo_prior(Z2, S2, C2, 7)[8]) and np.isfinite(loo_prior(Z2, S2, C2, 0)[8])
    # 2. regulator-independent (identical rows) matrix -> prior constant across leave-focal-out, Z -
    # P = 0
    v = rng.normal(size=40)
    mats = {"a": np.tile(v, (6, 1)), "b": np.tile(rng.normal(size=40), (6, 1))}
    bc = np.arange(6)
    R = build_set(mats, bc)
    Tm = R["T"]
    assert Tm.sum() == 40 - 6 and not Tm[:6].any()                      # regulator genes drop out of the strict T
    za = R["methods"]["a"]
    Pfull = loo_prior(za["Z"], za["S"], za["C"], None)
    assert np.allclose(za["Z"], Pfull[None, :], atol=1e-12)
    for r in range(6):
        assert np.allclose(loo_prior(za["Z"], za["S"], za["C"], r), Pfull, atol=1e-12)
    try:
        build_set({"a": np.ones((6, 40)), "b": mats["b"]}, bc)
        raise AssertionError("constant rows must be refused")
    except SystemExit as e:
        assert "BAD_ROWS" in str(e)
    # 3. relaxed sensitivity uses the T-derived constants and keeps regulator genes as targets (self
    # excluded)
    X = rng.normal(size=(6, 40))
    R = build_set({"a": X, "b": X ** 2}, bc)
    r_ = R["methods"]["a"]
    Xs = X.copy()
    Xs[np.arange(6), bc] = np.nan
    assert np.array_equal(R["Trel"], np.ones(40, bool)) and R["T"].sum() == 34   # every regulator gene has exactly one NaN (self)
    Tcols = R["T"]
    mu = np.array([Xs[r][Tcols].mean() for r in range(6)])
    sd = np.array([Xs[r][Tcols].std() for r in range(6)])
    assert np.allclose(r_["mu"], mu) and np.allclose(r_["sd"], sd)
    assert np.allclose(r_["Zr"], (Xs - mu[:, None]) / sd[:, None], equal_nan=True)                 # relaxed Z = T constants applied to all targets
    assert np.allclose(r_["Zr"][:, Tcols[R["Trel"]]], r_["Z"])                                      # identical on T
    assert np.isnan(r_["Zr"][np.arange(6), bc]).all() and (r_["Cr"][:6] == 5).all()                 # regulator target: 5 contributors, self excluded
    Xc = X.copy()
    Xc[:, :6] += 100.0                                                                              # change values only outside T -> constants unchanged
    R2 = build_set({"a": Xc, "b": X ** 2}, bc)
    assert np.allclose(R2["methods"]["a"]["mu"], mu) and np.allclose(R2["methods"]["a"]["Z"], r_["Z"])
    # 4. root-disagreement identity and root-min == production zcons, on the production code path
    G = 80
    raw = {h: rng.uniform(size=(G, 31)) for h in ROOTS}
    uni = rng.uniform(size=G) > 0.2
    d = primary_diag(raw, uni)
    H1, H2 = d["H|" + ROOTS[0]], d["H|" + ROOTS[1]]
    assert np.allclose(np.minimum(H1, H2), 0.5 * (H1 + H2) - 0.5 * np.abs(H1 - H2), equal_nan=True, atol=1e-12)
    assert np.array_equal(d["P"][uni], GP.zcons({h: raw[h][uni] for h in ROOTS}), equal_nan=True) and np.isnan(d["P"][~uni]).all()
    assert np.allclose(d["Pmean"] - d["disag"], d["P"], equal_nan=True, atol=1e-12) and (d["disag"][uni & np.isfinite(d["P"])] > 0).all()
    dc = primary_diag({h: raw[ROOTS[0]] for h in ROOTS}, uni)                                       # cell variant: identical at both roots
    assert np.allclose(dc["disag"][uni], 0.0, equal_nan=True) and np.array_equal(dc["P"], dc["H|" + ROOTS[0]], equal_nan=True)
    # 5. a missing external row / axis column is a failure, never dropped
    M = rng.normal(size=(4, 6))
    names, cols = ["A", "B", "C", "D"], list("uvwxyz")
    assert np.allclose(pick_rows(M, names, cols, ["c", "A"], cols[:3], "m"), M[[2, 0]][:, :3])
    for bad_members, bad_cols in ((["A", "ZZZ"], cols), (["A"], cols + ["q"])):
        try:
            pick_rows(M, names, cols, bad_members, bad_cols, "m")
            raise AssertionError("missing row/col must raise")
        except MissingRows as e:
            assert e.rows == ["ZZZ"] or e.cols == ["q"]
    # 6. score_block: U and M halves are not mixed
    n, ns, G, kmax = 220, 3, 11, 6
    dx = rng.normal(size=(n, ns))
    yU, yM = rng.normal(size=(n, G)) + 0.3 * dx[:, [0]], rng.normal(size=(n, G)) ** 3
    Qf, _ = np.linalg.qr(rng.normal(size=(n, 30)))
    rU, rM, nU, nM = score_block(dx.astype(np.float32), lambda sl: (yU[:, sl].astype(np.float32), yM[:, sl].astype(np.float32)), G, Qf, kmax, chunk=4)
    for k in range(kmax + 1):
        Qk = Qf[:, :k]
        proj = lambda v: v - Qk @ (Qk.T @ v)
        for Y, R_ in ((yU, rU), (yM, rM)):
            pr = proj(dx)[:, :, None] * proj(Y)[:, None, :]
            ref = np.abs(pr.sum(0)) / np.abs(pr).sum(0)
            assert np.allclose(R_[:, :, k], ref, atol=2e-6), (k, np.abs(R_[:, :, k] - ref).max())
            if k == 0:
                w = np.abs(pr)
                assert np.allclose(nU if Y is yU else nM, w.sum(0) ** 2 / (w ** 2).sum(0), rtol=1e-4)
    Pc = rng.normal(size=(n, 30))                                                                   # cell twin vs the gb_adds.cell_scores float64 formula
    Xs_, Mu_, Ms_ = rng.normal(size=(n, 2)) + Pc[:, :2], rng.normal(size=(n, G)) + Pc[:, [0]], rng.normal(size=(n, G))
    Qc, _ = np.linalg.qr(Pc - Pc.mean(0))
    xc = (Xs_ - Xs_.mean(0)).astype(np.float32)
    cU, cM, _, _ = score_block(xc, lambda sl: ((Mu_[:, sl] - Mu_[:, sl].mean(0)).astype(np.float32), (Ms_[:, sl] - Ms_[:, sl].mean(0)).astype(np.float32)),
                               G, Qc, kmax)
    for k in range(kmax + 1):                                                                       # gb_adds.py:81-86, verbatim algebra
        Q, _ = np.linalg.qr(np.column_stack([np.ones(n), Pc[:, :k]]))
        res = lambda Y: Y - Q @ (Q.T @ Y)
        for s in range(2):
            x = res(Xs_[:, [s]])
            for Y, C_ in ((Mu_, cU), (Ms_, cM)):
                prod = x * res(Y)
                ref = np.abs(prod.sum(0)) / np.abs(prod).sum(0)
                assert np.allclose(C_[s, :, k], ref, atol=2e-6), ("cell", k, s)
    # 7. universe = production observational_universe, bank rule, slice parser, |Spearman|
    rows = {(h, c): rng.uniform(size=(9, 31)) for h in ROOTS for c in ("U", "X")}
    rows[(ROOTS[0], "X")][4, 17] = np.nan
    u = obs_universe("g2", rows, np.char.upper(np.array([f"g{i}" for i in range(9)])))
    assert (u == np.array([1, 1, 0, 1, 0, 1, 1, 1, 1], bool)).all()
    assert parse_slice("2:5", 10) == (2, 5) and parse_slice(":3", 10) == (0, 3) and parse_slice("7:", 10) == (7, 10)
    from scipy.stats import spearmanr
    Mm, pt = rng.normal(size=(60, 4)), rng.normal(size=60)
    Mm[:, 3] = 1.0
    sp = abs_spearman_cols(Mm, pt, block=3)
    assert np.allclose(sp[:3], [abs(spearmanr(Mm[:, g], pt)[0]) for g in range(3)]) and np.isnan(sp[3])
    # Check missing rows, fixed T7 diagnostics, symbol collisions, shard names and receipts.
    nmn = [f"TF{i}" for i in range(6)]
    Xn = rng.normal(size=(6, 40))
    Xn[3, :] = np.nan
    try:
        build_set({"a": Xn, "b": X}, bc, names=nmn)
        raise AssertionError("all-NaN row must be refused")
    except SystemExit as e:
        assert "ALL_NAN_ROWS" in str(e) and "TF3" in str(e) and "'a'" in str(e) and "'b'" not in str(e)
    try:
        build_set({"a": np.where(np.arange(6)[:, None] == 2, 1.0, X), "b": X}, bc, names=nmn)
        raise AssertionError
    except SystemExit as e:
        assert "BAD_ROWS" in str(e) and "TF2" in str(e)
    Dg = X ** 3
    Dg[:, 20] = np.nan                                                                             # diagnostic NaN on a T column -> refused, T untouched
    try:
        build_set({"a": X, "b": X ** 2}, bc, diag={"d": Dg}, names=nmn)
        raise AssertionError
    except SystemExit as e:
        assert "DIAG_NOT_FINITE_ON_T" in str(e)
    Dg = X ** 3
    R0 = build_set({"a": X, "b": X ** 2}, bc)
    R1 = build_set({"a": X, "b": X ** 2}, bc, diag={"d": Dg}, names=nmn)
    assert np.array_equal(R0["T"], R1["T"]) and R1["diag"] == ["d"]
    Xd = Dg.copy()
    Xd[np.arange(6), bc] = np.nan
    zd = np.array([(Xd[r][R1["T"]] - Xd[r][R1["T"]].mean()) / Xd[r][R1["T"]].std() for r in range(6)])
    assert np.allclose(R1["methods"]["d"]["Z"], zd)
    Dc = types.SimpleNamespace(layers={"spliced": (rng.uniform(size=(50, 5)) < np.array([.5, .5, .5, .01, .5])).astype(float)})
    Dbank = {"axis": ["Aa", "Bb", "Cc", "Dd", "Ee"], "axis_cols": np.arange(5), "S": {"is_tf": np.array([1, 1, 1, 1, 0], bool)}, "A": Dc,
             "fixture_genes": ["Aa", "Bb", "Cc", "Dd", "Ee", "AA", "z"]}
    Dbank["A"].layers["spliced"][:, 3] = 0.0
    Dbank["A"].layers["spliced"][:20, 0] = 1.0
    Dbank["A"].layers["spliced"][:, 1] = 1.0
    Dbank["A"].layers["spliced"][:, 2] = 1.0
    bk = compute_bank(Dbank)
    assert bk["members"] == ["Bb", "Cc"] and list(bk["excluded_case_collision"]) == ["Aa"] and "Dd" in bk["below_floor"], bk   # Aa/AA collide; Dd < .05; Ee not TF
    pt = shard_pattern("PANC")
    assert pt.fullmatch("PANC_scores_0-160.npz") and not pt.fullmatch("PANC_scores_0-160.receipt.json") and not pt.fullmatch("PANC_smalln_scores.npz")
    assert not pt.fullmatch("XPANC_scores_0-160.npz") and not pt.fullmatch("PANC_scores_0-160.npz.bak") and not pt.fullmatch("PANC_scores_a-b.npz")
    import json
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        f = Path(td) / "x.npz"
        f.write_bytes(b"abc")
        assert "no receipt" in ext_artifact_problem(f, "F")
        rp = f.with_suffix(".receipt.json")
        rp.write_text(json.dumps({"output_sha256": RU.sha(f), "binding": {"inputs": {"fixture_sha256": "F"}}}))
        assert ext_artifact_problem(f, "F") is None and "fixture_sha256" in ext_artifact_problem(f, "G")
        f.write_bytes(b"abd")
        assert "sha256" in ext_artifact_problem(f, "F")
    print("gb_catbank selftest OK")


# CLI
def main():
    if "--selftest" in sys.argv:
        selftest()
        return
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for c in ("bank", "scores", "external", "priors", "features", "sota"):
        p = sub.add_parser(c)
        p.add_argument("system")
        p.add_argument("--smalln", action="store_true", help="first 5 bank members; files tagged _smalln")
        if c == "scores":
            p.add_argument("--sources", default="", help="bank slice i:j (python semantics)")
            p.add_argument("--kmax", type=int, default=KMAX)
            p.add_argument("--block", type=int, default=256, help="sources per project_scores call (memory only; result-invariant)")
            p.add_argument("--tol-raw", type=float, default=1e-6, help="raw |S| parity vs catalogue bank / production caches / (fail-closed)")
            p.add_argument("--tol-primary", type=float, default=1e-5, help="PRIMARY-level parity (z amplifies raw diffs by 1/sd)")
            p.add_argument("--tol-cell", type=float, default=1e-5, help="cell twin vs gb_adds.cell_scores (float32 residual cast)")
            p.add_argument("--cell-parity-n", type=int, default=2)
            p.add_argument("--allow-no-parity", action="store_true")
            p.add_argument("--force", action="store_true")
        if c in ("priors", "sota"):
            p.add_argument("--min-contrib", type=int, default=2)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"bank": cmd_bank, "scores": cmd_scores, "external": cmd_external, "priors": cmd_priors, "features": cmd_features, "sota": cmd_sota}[a.cmd](a)


if __name__ == "__main__":
    main()
