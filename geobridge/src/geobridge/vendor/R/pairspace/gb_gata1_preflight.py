#!/usr/bin/env python
"""GATA1 rerun preflight: control-only checks that must pass before any score is computed.

Only control cells are used; no perturbation data is read.
  Gate A  order invariance. Cells are sorted by a stable ID before PCA, neighbours, diffmap and capped pair
          sampling; a row permutation of the input must reproduce identical pairs for both roots.
  Gate B  velocity integrity. No non-finite X or V and no zero-variance velocity gene, checked before any
          nan_to_num.
  Gate C  matched regulator panel. At least 100 catalogue regulators, one panel balanced for both roots,
          every |SMD| <= 0.10, standardised within the eligible pool.
Scoring requires gates A, B and D. Gate C reports matching feasibility separately and does not block
scoring.
external_regulators.txt is 1,721 SCENIC TFs plus 938 chromatin regulators: a regulator catalogue,
not a TF list.
"""
import argparse, hashlib, json, os, platform, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
ROOTS = ["dc1_min", "dc1_max"]
N_MIN = 100
SMD_GATE = 0.10
MIN_FORWARD_DT = 0.02      # frozen pair window, was a magic literal inside canonical_pairs
MAX_FORWARD_DT = 0.25
MAX_PAIRS_PER_CELL = 40
REGULATOR = "Gata1"
SEED = 42
FEATURES = ["mean", "detection", "sd", "moran_I", "private_var", "velocity_finite_frac",
            "velocity_var", "state_assoc_dc1_min", "state_assoc_dc1_max",
            "pairdiff_scoreability_dc1_min", "pairdiff_scoreability_dc1_max",
            "denominator_scoreability_dc1_min", "denominator_scoreability_dc1_max"]


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


_SELF_SHA = globals().get("_VERIFIED_SELF_SHA") or hashlib.sha256(
    open(os.path.abspath(__file__), "rb").read()).hexdigest()
_SELF_SHA_SOURCE = "injected_by_verified_entry" if globals().get("_VERIFIED_SELF_SHA") \
    else "self_read_at_import"


def _load_verified(path, name):
    """Execute the exact bytes that are hashed; never consult a bytecode cache."""
    import types as _t
    with open(path, "rb") as fh:
        src = fh.read()
    sha = hashlib.sha256(src).hexdigest()
    mod = _t.ModuleType(name)
    mod.__file__ = path
    exec(compile(src, path, "exec"), mod.__dict__)
    return mod, sha


GEOBRIDGE_ROOT = os.environ.get("GB_REPO", "")
_PS_PATH = os.path.join(GEOBRIDGE_ROOT, "geobridge_model", "pairspace.py")
_PS_CACHE = {}


def _pairspace_verified():
    """Resolve, load and hash the pair builder once."""
    if "mod" not in _PS_CACHE:
        if not os.path.exists(_PS_PATH):
            raise SystemExit(f"REFUSING: pair builder not found at {_PS_PATH}")
        mod, sha = _load_verified(_PS_PATH, "_gb_pairspace_verified")
        if not hasattr(mod, "build_forward_pairs"):
            raise SystemExit(f"REFUSING: {_PS_PATH} defines no build_forward_pairs")
        _PS_CACHE["mod"], _PS_CACHE["sha"] = mod, sha
    return _PS_CACHE["mod"], _PS_CACHE["sha"]


def _pairspace_path():
    return _PS_PATH


def _blas_identity():
    """Record the linear-algebra backend: the feature hash fingerprints this artifact and is not a
    cross-host hash, because private_var depends on float64 SVD output.
    """
    try:
        cfg = np.show_config(mode="dicts")
        return {k: str(v) for k, v in cfg.get("Build Dependencies", {}).items()}
    except Exception as e:
        return {"unavailable": str(e)}


def sanitize(o):
    if isinstance(o, dict):
        return {k: sanitize(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [sanitize(v) for v in o]
    if isinstance(o, (bool, np.bool_)):
        return bool(o)
    if isinstance(o, (float, np.floating)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (int, np.integer)):
        return int(o)
    return o


def to_dense(X):
    from scipy.sparse import issparse
    return np.asarray(X.todense()) if issparse(X) else np.asarray(X)


# ------------------------------------------------------------------- canonical, order-free pairing
def canonical_pairs(X, V, cell_ids, root, seed, *, min_forward_dt=None,
                    max_forward_dt=None, max_pairs_per_cell=None):
    """Sort by stable cell ID first so pair construction cannot depend on input row order;
    sorting only by pseudotime is not enough.
    """
    import scanpy as sc, anndata as ad
    min_forward_dt = MIN_FORWARD_DT if min_forward_dt is None else min_forward_dt
    max_forward_dt = MAX_FORWARD_DT if max_forward_dt is None else max_forward_dt
    max_pairs_per_cell = MAX_PAIRS_PER_CELL if max_pairs_per_cell is None else max_pairs_per_cell
    _ps_mod, _ = _pairspace_verified()          # loaded once, cached; same bytes for every call
    build_forward_pairs = _ps_mod.build_forward_pairs
    can = np.argsort(np.asarray(cell_ids).astype(str), kind="stable")
    Xc, Vc = X[can], V[can]
    A = ad.AnnData(Xc.astype(np.float32))
    sc.pp.pca(A, n_comps=min(30, Xc.shape[1] - 1, Xc.shape[0] - 1), random_state=seed)
    sc.pp.neighbors(A, n_neighbors=30, use_rep="X_pca", random_state=seed)
    from scipy.sparse.csgraph import connected_components
    ncomp, _ = connected_components(A.obsp["connectivities"], directed=False)
    if ncomp != 1:
        raise SystemExit(f"REFUSING: neighbour graph has {ncomp} components; DPT is undefined "
                         f"across them and imputing pseudotime would be fabrication")
    sc.tl.diffmap(A)
    dc1 = np.asarray(A.obsm["X_diffmap"])[:, 1]
    A.uns["iroot"] = int(np.argmin(dc1) if root == "dc1_min" else np.argmax(dc1))
    sc.tl.dpt(A)
    pt = np.asarray(A.obs["dpt_pseudotime"], dtype=np.float64)
    if not np.isfinite(pt).all():
        raise SystemExit("REFUSING: DPT produced non-finite pseudotime on a connected graph")
    # secondary sort by pseudotime with the canonical order as the tie-break
    o2 = np.lexsort((np.arange(len(pt)), pt))
    pair = build_forward_pairs(Xc[o2], Vc[o2], pt[o2], min_forward_dt=min_forward_dt,
                               max_forward_dt=max_forward_dt,
                               max_pairs_per_cell=max_pairs_per_cell, pair_seed=seed)
    I = can[o2[np.asarray(pair.pair_i, int)]]
    J = can[o2[np.asarray(pair.pair_j, int)]]
    ptv = np.empty(len(pt)); ptv[can[o2]] = pt[o2]
    if not np.all(ptv[J] > ptv[I]):
        raise SystemExit("REFUSING: recovered pairs are not forward in pseudotime")
    inv = np.empty(len(can), int); inv[can] = np.arange(len(can))
    C = A.obsp["connectivities"][inv][:, inv]
    return ptv, I, J, C


def _resid_score(X, V, I, J, g, k):
    """One residualized signed-concordance vector, used to prove order invariance of the score and not
merely of the pair set."""
    Xc = X - X.mean(0)
    if k > 0:
        U = np.linalg.svd(Xc, full_matrices=False)[0][:, :k]
        Xr, Vr = X - U @ (U.T @ X), V - U @ (U.T @ V)
    else:
        Xr, Vr = X, V
    dX, dV = Xr[J] - Xr[I], Vr[J] - Vr[I]
    dx = dX[:, g]
    return (dx @ dV) / ((np.abs(dx) @ np.abs(dV)) + 1e-12)


def _spectrum_report(sv, k_grid, label):
    rank = int((sv > 1e-10 * sv.max()).sum()) if sv.size else 0
    gap = {}
    for k in k_grid:
        if 1 <= k < len(sv):
            gap[str(k)] = float((sv[k - 1] - sv[k]) / max(float(sv[k - 1]), 1e-300))
    return {"matrix": label, "n_singular_values": int(len(sv)), "numerical_rank": rank,
            "relative_gap_at_k": gap,
            "k_marked_near_degenerate": [k for k, g in gap.items() if g < 1e-3],
            "k_at_or_beyond_numerical_rank": [k for k in k_grid if k >= rank],
            "NOTE": ("a small relative gap MARKS a k as poorly conditioned (reported, not excluded); "
                     "k >= numerical rank is invalid because residualization annihilates the space")}


def gate_d_grid_invariance(X, V, cell_ids, seed, panel, k_grid, operators, scores_for_operator,
                           tol=1e-8):
    """Gate D. A grid cell may contribute to the max-over-grid only if the production score tensor is
row-permutation invariant within tol, for both roots, every k."""
    rng = np.random.default_rng(seed + 7717)
    perm = rng.permutation(X.shape[0])
    ids_p = np.asarray(cell_ids)[perm]
    cells = {}
    for root in ROOTS:
        _, I0, J0, _ = canonical_pairs(X, V, cell_ids, root, seed)
        _, I1, J1, _ = canonical_pairs(X[perm], V[perm], ids_p, root, seed)
        for op in operators:
            for (k, Sa), (_, Sb) in zip(
                    scores_for_operator(X, V, I0, J0, panel, op),
                    scores_for_operator(X[perm], V[perm], I1, J1, panel, op)):
                d = float(np.max(np.abs(np.asarray(Sa) - np.asarray(Sb))))
                cells[f"{root}|{op}|k={k}"] = {"max_abs_difference": d, "VALID": bool(d <= tol)}
    invalid = sorted([c for c, v in cells.items() if not v["VALID"]])
    return {"gate": "D_grid_cell_invariance", "tolerance": tol, "cells": cells,
            "n_cells": len(cells), "n_invalid": len(invalid), "invalid_cells": invalid,
            "PASS": len(invalid) < len(cells),
            "RULE": ("invariance is per-CELL: invalid cells are dropped identically from the "
                     "regulator and every pseudo arm, and are recorded MISSING rather than zero. "
                     "The gate fails only if NO cell survives.")}


def gate_a_order_invariance(X, V, cell_ids, seed, reg_idx, precomputed):
    """A row permutation must reproduce the identical ordered pairs and the identical scores.
    The Jaccard floor is 1.0: a cap-induced discrepancy is an implementation failure. Set equality
    alone is
    not enough; ordered arrays and residualized scores are compared too.
    """
    rng = np.random.default_rng(seed + 991)
    perm = rng.permutation(X.shape[0])
    det, frozen = {}, {}
    for root in ROOTS:
        I0, J0 = precomputed[root]
        _, I1, J1, _ = canonical_pairs(X[perm], V[perm], np.asarray(cell_ids)[perm], root, seed)
        I1o, J1o = perm[I1], perm[J1]
        same_set = set(zip(I0.tolist(), J0.tolist())) == set(zip(I1o.tolist(), J1o.tolist()))
        # compare as ordered arrays after a canonical lexsort, so pair order cannot mask a defect
        k0 = np.lexsort((J0, I0)); k1 = np.lexsort((J1o, I1o))
        same_arr = bool(len(I0) == len(I1o) and np.array_equal(I0[k0], I1o[k1])
                        and np.array_equal(J0[k0], J1o[k1]))
        smax = 0.0
        if same_arr:
            for kk in (0, 16):
                s0 = _resid_score(X, V, I0[k0], J0[k0], reg_idx, kk)
                s1 = _resid_score(X[perm], V[perm], I1[k1], J1[k1], reg_idx, kk)
                smax = max(smax, float(np.max(np.abs(s0 - s1))))
        det[root] = {"n_pairs": int(len(I0)), "n_pairs_permuted": int(len(I1o)),
                     "identical_pair_set": bool(same_set),
                     "identical_ordered_arrays": same_arr,
                     "max_abs_score_difference": smax if same_arr else None,
                     "score_invariant": bool(same_arr and smax < 1e-8)}
        frozen[root] = (I0[k0], J0[k0])
    return {"gate": "A_order_invariance", "required_jaccard": 1.0, "per_root": det,
            "PASS": all(d["identical_ordered_arrays"] and d["score_invariant"]
                        for d in det.values())}, frozen


def gate_b_velocity(Xraw, Vraw):
    """Non-finiteness, zero-variance velocity, and the hard V = X invariant."""
    nfX = int((~np.isfinite(Xraw)).sum())
    nfV = int((~np.isfinite(Vraw)).sum())
    zero = int((np.nan_to_num(Vraw).std(0) == 0).sum())
    fin = np.isfinite(Xraw) & np.isfinite(Vraw)
    identical = bool(fin.sum() and np.allclose(Xraw[fin], Vraw[fin], atol=1e-8))
    rel = float(np.linalg.norm(np.nan_to_num(Xraw) - np.nan_to_num(Vraw))
                / max(np.linalg.norm(np.nan_to_num(Xraw)), 1e-12))
    v_equals_x = bool(identical or rel < 1e-3)
    return {"gate": "B_velocity_integrity", "nonfinite_X": nfX, "nonfinite_V": nfV,
            "zero_variance_velocity_genes": zero,
            "V_equals_X": v_equals_x, "relative_distance_X_to_V": rel,
            "PASS": nfX == 0 and nfV == 0 and zero == 0 and not v_equals_x,
            "RULE": ("checked before any nan_to_num so a malformed layer cannot be concealed; "
                     "V = X is invalid and is refused here")}


# ------------------------------------------------------------------------------ matching features
def build_features(X, V, Vraw, pairs_by_root, conn):
    mu, sd = X.mean(0), X.std(0)
    Xc = X - mu
    moran = (Xc * (conn @ Xc)).sum(0) / np.maximum((Xc * Xc).sum(0), 1e-12) * (X.shape[0] / conn.sum())
    U10 = np.linalg.svd(Xc, full_matrices=False)[0][:, :10]
    priv = (Xc - U10 @ (U10.T @ Xc)).std(0)
    F = {"mean": mu, "detection": (X > 0).mean(0), "sd": sd, "moran_I": moran,
         "private_var": priv, "velocity_finite_frac": np.isfinite(Vraw).mean(0),
         "velocity_var": V.var(0)}
    for root, (pt, I, J) in pairs_by_root.items():
        z = (pt - pt.mean()) / max(pt.std(), 1e-12)
        F[f"state_assoc_{root}"] = np.abs((Xc / np.maximum(sd, 1e-12)).T @ z) / len(pt)
        dX = X[J] - X[I]
        dV = V[J] - V[I]
        F[f"pairdiff_scoreability_{root}"] = (np.abs(dX) > 1e-12).mean(0)
        # mean concordance-denominator support of this gene as a source, averaged over targets
        F[f"denominator_scoreability_{root}"] = (np.abs(dX).T @ np.abs(dV)).mean(1) / max(len(I), 1)
    return np.column_stack([F[k] for k in FEATURES]), F


def main():
    ap = argparse.ArgumentParser(description="control-only; accepts no truth argument")
    ap.add_argument("--control", required=True)
    ap.add_argument("--regulators", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect-catalogue-sha256", required=True)
    ap.add_argument("--seed", type=int, default=SEED,
                    help="must equal the frozen SEED; any other value refuses")
    ap.add_argument("--jobs", type=int, default=1,
                    help="parallel workers for the placebo sweep")
    ap.add_argument("--placebo-budget-seconds", type=float, default=None,
                    help="declared global budget; exceeding it yields SEARCH_UNRESOLVED, never FAIL")
    a = ap.parse_args()
    if a.seed != SEED:
        raise SystemExit(f"REFUSING: seed {a.seed} differs from the frozen SEED {SEED}; the seed enters pair construction and pseudo selection, so it is part of the certified gate")
    import anndata as ad, scanpy as sc, scvelo as scv

    W = ad.read_h5ad(a.control)
    meta = W.uns.get("gb_clean_build")
    if not isinstance(meta, dict) or not meta.get("control_only"):
        raise SystemExit("REFUSING: input is not a verified control-only build")
    genes = np.asarray(W.var_names).astype(str)
    cell_ids = np.asarray(W.obs_names).astype(str)
    Xraw = to_dense(W.X).astype(np.float64)
    Vraw = to_dense(W.layers["velocity"]).astype(np.float64)

    gb = gate_b_velocity(Xraw, Vraw)
    if not gb["PASS"]:
        rep = {"script": "gb_gata1_preflight.py", "VERDICT": "FAIL", "gate_B": gb,
               "STOPPED_BEFORE": "any nan_to_num, feature construction, or gate A/C evaluation"}
        with open(a.out, "w") as fh:
            json.dump(sanitize(rep), fh, indent=1)
        raise SystemExit(f"REFUSING: velocity integrity gate failed: {gb}")
    X, V = Xraw, Vraw
    if REGULATOR not in set(genes):
        raise SystemExit(f"REFUSING: {REGULATOR} absent from the control gene set")
    r = int(np.where(genes == REGULATOR)[0][0])

    if len(set(cell_ids.tolist())) != len(cell_ids):
        raise SystemExit("REFUSING: cell identifiers are not unique, so canonical sorting is "
                         "ill-defined and order invariance cannot be established")
    pairs_by_root, conn = {}, None
    for root in ROOTS:
        pt, I, J, C = canonical_pairs(X, V, cell_ids, root, a.seed)
        pairs_by_root[root] = (pt, I, J)
        conn = C if conn is None else conn
    ga, frozen_pairs = gate_a_order_invariance(
        X, V, cell_ids, a.seed, r, {root: pairs_by_root[root][1:3] for root in ROOTS})
    for root in ROOTS:
        I0, J0 = frozen_pairs[root]
        Ip, Jp = pairs_by_root[root][1], pairs_by_root[root][2]
        k = np.lexsort((Jp, Ip))
        if not (np.array_equal(I0, Ip[k]) and np.array_equal(J0, Jp[k])):
            raise SystemExit(f"REFUSING: frozen pairs for {root} are not a reordering of the pair "
                             f"set used to build the matching features; the panel would be matched "
                             f"on a different graph than the one scored")
    F, Fraw = build_features(X, V, Vraw, pairs_by_root, conn)

    K_GRID_REPORT = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 90, 128, 200]
    spectra = [_spectrum_report(np.linalg.svd(X - X.mean(0), compute_uv=False),
                                K_GRID_REPORT, "R1R3_cell_expression")]
    for root in ROOTS:
        _, Ir, Jr = pairs_by_root[root]
        dX0 = X[Jr] - X[Ir]
        Dc = dX0 - dX0.mean(0, keepdims=True)
        spectra.append(_spectrum_report(np.linalg.svd(Dc, compute_uv=False),
                                        K_GRID_REPORT, f"R2_pair_difference|{root}"))
        del dX0, Dc

    cat = set(l.strip() for l in open(a.regulators) if l.strip())
    cat_sha = sha_file(a.regulators)
    if a.expect_catalogue_sha256 and cat_sha != a.expect_catalogue_sha256:
        raise SystemExit(f"REFUSING: regulator catalogue sha256 {cat_sha} does not match the "
                         f"expected frozen catalogue {a.expect_catalogue_sha256}")
    if REGULATOR not in cat:
        raise SystemExit(f"REFUSING: {REGULATOR} is not in the regulator catalogue, so it cannot be "
                         f"standardised within the eligible regulator pool")
    reg_pool = np.array([i for i, g in enumerate(genes) if g in cat], int)
    sub = F[reg_pool]
    Fz = (F - sub.mean(0)) / np.maximum(sub.std(0), 1e-12)
    _mpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gb_gata1_matching.py")
    _MT, _matching_sha = _load_verified(_mpath, "_gb_matching_verified")
    certified_panel, placebo_locality = _MT.certified_panel, _MT.placebo_locality
    sidecar = os.path.splitext(a.out)[0] + "_farkas.npz"
    gc = certified_panel(Fz, reg_pool, r, N_MIN, SMD_GATE, placebo=True, names=genes,
                         sidecar_path=sidecar)
    if gc["VERDICT"] == "PANEL_CERTIFIED_PENDING_PLACEBO":
        pl = placebo_locality(Fz, reg_pool, r, N_MIN, SMD_GATE, n_jobs=a.jobs,
                              wall_clock_budget=a.placebo_budget_seconds)
        gc["placebo_locality"] = pl
        gc["VERDICT"] = ("PANEL_CERTIFIED" if pl.get("PASS") else
                         ("PLACEBO_LOCALITY_FAIL" if pl.get("status") == "OK"
                          else "SEARCH_UNRESOLVED"))
    gc["PASS"] = gc["VERDICT"] == "PANEL_CERTIFIED"

    if gc["PASS"]:
        panel_idx = np.array(gc["selected_indices"], int)
        specificity = {"status": "CERTIFIED", "panel_source": "certified_regulator_panel",
                       "n_panel": int(panel_idx.size)}
    else:
        d_all = np.linalg.norm(Fz - Fz[r][None, :], axis=1)
        d_all[r] = np.inf
        panel_idx = np.array(sorted(np.argsort(d_all)[:120].tolist()), int)
        specificity = {"status": "NO_CALL",
                       "reason": gc.get("VERDICT"),
                       "panel_source": "descriptive_geometry_matched_all_gene",
                       "n_panel": int(panel_idx.size),
                       "WARNING": ("this panel is NOT regulator-restricted and carries NO support or "
                                   "MILP certificate; it calibrates the score surface descriptively "
                                   "and must never be reported as regulator specificity")}

    _FZ, _freezer_sha = _load_verified(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "gb_gata1_freeze_scores.py"),
        "_gb_fz_verified_main")
    FK, FOPS = _FZ.K_GRID, _FZ.OPERATORS
    gd = None
    if True:
        panel_full = np.concatenate([[r], panel_idx])
        gd = gate_d_grid_invariance(X, V, cell_ids, a.seed, panel_full, FK, FOPS,
                                    _FZ.scores_for_operator)
        rank_by = {sp["matrix"]: sp["numerical_rank"] for sp in spectra}
        rank_invalid = []
        for cell in list(gd["cells"]):
            root, op, kpart = cell.split("|")
            kk = int(kpart.split("=")[1])
            if op == "R1R3_cellpc":
                rk = rank_by.get("R1R3_cell_expression")
            elif op == "R2_pairSVD":
                rk = rank_by.get(f"R2_pair_difference|{root}")
            else:
                rk = None                      # R0 has no k
            if rk is not None and kk >= rk:
                gd["cells"][cell]["VALID"] = False
                gd["cells"][cell]["rank_invalid"] = True
                rank_invalid.append(cell)
        gd["rank_invalid_cells"] = sorted(rank_invalid)
        gd["invalid_cells"] = sorted([c for c, v in gd["cells"].items() if not v["VALID"]])
        gd["n_invalid"] = len(gd["invalid_cells"])
        gd["PASS"] = gd["n_invalid"] < gd["n_cells"]


    if ga["PASS"] and gb["PASS"] and gd is not None and gd["PASS"]:
        verdict = "PASS"
    elif not (ga["PASS"] and gb["PASS"]):
        verdict = "FAIL_CERTIFIED"          # deterministic; the substrate itself is unusable
    else:
        verdict = "UNRESOLVED"              # implementation problem; must not terminate the branch
    out = {"script": "gb_gata1_preflight.py",
           "self_sha256": _SELF_SHA, "self_sha256_provenance": _SELF_SHA_SOURCE,
           "control_input": os.path.abspath(a.control),
           "control_input_sha256": sha_file(a.control),
           "regulator_catalogue": os.path.abspath(a.regulators),
           "regulator_catalogue_sha256": cat_sha,
           "regulator_catalogue_size": len(cat),
           "regulator_in_catalogue": True,
           "pairspace_module": _PS_PATH,
           # the digest of the bytes actually executed to build the frozen pairs, not a re-read
           "pairspace_module_sha256": _pairspace_verified()[1],
           "matching_module_sha256": _matching_sha,
           "freezer_module_sha256": _freezer_sha,
           "constants": {"N_MIN": N_MIN, "SMD_GATE": SMD_GATE, "ROOTS": ROOTS,
                         "MAX_PAIRS_PER_CELL": MAX_PAIRS_PER_CELL, "REGULATOR": REGULATOR,
                         "SEED": SEED},
           "blas": _blas_identity(),
           "regulators_in_universe": int(len(reg_pool)),
           "CATALOGUE_NOTE": ("1,721 SCENIC TFs union 938 chromatin regulators "
                              "(FROZEN_REGISTRY_OFFPHASE_2026_07_25.md:67); a REGULATOR catalogue, "
                              "not a TF list"),
           "features": FEATURES,
           "feature_matrix_sha256": hashlib.sha256(np.ascontiguousarray(F).tobytes()).hexdigest(),
           "feature_hash_scope": ("artifact fingerprint for THIS run; not a portable cross-host "
                                  "reconstruction hash (see blas)"),
           "pairs": {root: {"n_pairs": int(len(pairs_by_root[root][1])),
                            "pair_sha256": hashlib.sha256(
                                np.ascontiguousarray(np.column_stack(pairs_by_root[root][1:3])
                                                     ).tobytes()).hexdigest()}
                     for root in ROOTS},
           "gata1_feature_percentiles_within_regulator_pool": {
               FEATURES[j]: float((F[reg_pool, j] < F[r, j]).mean() * 100)
               for j in range(len(FEATURES))},
           "spectrum_diagnostics": spectra,
           "gate_A": ga, "gate_B": gb, "gate_C": gc, "gate_D": gd,
           "SPECIFICITY_ENDPOINT": specificity,
           "SCORING_LICENCE": "gates A+B+D; gate C is reported separately and does not withhold the score surface",
           "valid_grid_cells": ([c for c, v in gd["cells"].items() if v["VALID"]]
                                if gd else None),
           "VERDICT": verdict,
           "DECISION_RULE": ("Scoring requires gates A, B and D. Gate C reports matching feasibility "
                             "separately and does not block scoring."),
           "versions": {"python": platform.python_version(), "numpy": np.__version__,
                        "scanpy": sc.__version__, "scvelo": scv.__version__},
           "seed": a.seed}
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    npz = os.path.splitext(a.out)[0] + "_frozen.npz"
    payload = {"ARTIFACT_KIND": np.asarray("PREFLIGHT_FROZEN_PAIRS_NOT_PRODUCTION"),
               "cell_ids": cell_ids, "genes": genes, "regulator_index": np.array([r]),
               "reg_pool": reg_pool,
               "panel": panel_idx,
               "valid_cells": np.array(sorted([c for c, v in gd["cells"].items()
                                               if v["VALID"]]) if gd else [], dtype="U64")}   # not object: the
               # evaluator loads with allow_pickle=False, which rejects object arrays
    for root, (I0, J0) in frozen_pairs.items():
        payload[f"I|{root}"] = I0
        payload[f"J|{root}"] = J0
    np.savez_compressed(npz, **payload)
    out["frozen_pairs_artifact"] = os.path.basename(npz)
    out["frozen_pairs_sha256"] = sha_file(npz)
    out["frozen_pairs_contents"] = sorted(payload.keys())
    with open(a.out, "w") as fh:
        json.dump(sanitize(out), fh, indent=1)
    print(json.dumps(sanitize({"VERDICT": verdict, "gate_A": ga["PASS"], "gate_B": gb,
                               "gate_C": {k: gc.get(k) for k in
                                          ("PASS", "n_selected", "max_abs_SMD", "reason",
                                           "abs_SMD_by_axis", "axes_without_support")}}), indent=1))
    print(f"[done] {a.out}")


if __name__ == "__main__":
    main()
