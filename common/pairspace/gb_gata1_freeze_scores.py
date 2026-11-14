#!/usr/bin/env python
# Compute control-only scores from the certified preflight artifact.
import argparse, hashlib, json, os, platform, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")
EPS = 1e-12                       # matches gb_causal_null_killer.py:181
K_GRID = [0, 1, 2, 4, 8, 16, 24, 32, 48, 64, 90, 128, 200]
OPERATORS = ["R0_none", "R1R3_cellpc", "R2_pairSVD"]
ROOTS = ["dc1_min", "dc1_max"]


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


_SELF_SHA = globals().get("_VERIFIED_SELF_SHA") or hashlib.sha256(
    open(os.path.abspath(__file__), "rb").read()).hexdigest()
_SELF_SHA_SOURCE = "injected_by_verified_entry" if globals().get("_VERIFIED_SELF_SHA") \
    else "self_read_at_import"


def to_dense(X):
    from scipy.sparse import issparse
    return np.asarray(X.todense()) if issparse(X) else np.asarray(X)


# S_g = sum(dx dV) / (sum|dx dV| + eps). |dx*dV| = |dx|*|dV| exactly, so both are GEMMs and the whole panel is two matrix products
def concordance(dx, dV, adV):
    return (dx @ dV) / ((np.abs(dx) @ adV) + EPS)


def scores_for_operator(X, V, I, J, panel, op):
    if op == "R0_none":
        dX, dV = X[J] - X[I], V[J] - V[I]
        adV = np.abs(dV)
        S = np.vstack([concordance(dX[:, g], dV, adV) for g in panel])
        for k in K_GRID:
            yield k, S                     # R0 has no k; the repeated row makes that explicit
        return
    if op == "R1R3_cellpc":
        # basis from the centered matrix, projection applied to the uncentered X and V --
        # reproducing gb_causal_null_killer.py:172-178 exactly.
        U = np.linalg.svd(X - X.mean(0), full_matrices=False)[0][:, :max(K_GRID)]
        for k in K_GRID:
            Xr, Vr = (X, V) if k <= 0 else (X - U[:, :k] @ (U[:, :k].T @ X),
                                            V - U[:, :k] @ (U[:, :k].T @ V))
            dX, dV = Xr[J] - Xr[I], Vr[J] - Vr[I]
            adV = np.abs(dV)
            yield k, np.vstack([concordance(dX[:, g], dV, adV) for g in panel])
            del dX, dV, adV
        return
    if op == "R2_pairSVD":
        dX0, dV0 = X[J] - X[I], V[J] - V[I]
        Dc = dX0 - dX0.mean(0, keepdims=True)
        Up, Sp, Vtp = np.linalg.svd(Dc, full_matrices=False)
        for k in K_GRID:
            if k <= 0:
                dX, dV = Dc, dV0
            else:
                Uk = Up[:, :k]
                dX = Dc - (Uk * Sp[:k][None, :]) @ Vtp[:k, :]
                dV = dV0 - Uk @ (Uk.T @ dV0)
            adV = np.abs(dV)
            yield k, np.vstack([concordance(dX[:, g], dV, adV) for g in panel])
            del dX, dV, adV
        return
    raise SystemExit(f"REFUSING: unknown operator {op}")


def main():
    ap = argparse.ArgumentParser(description="control-only; accepts NO truth argument by design")
    ap.add_argument("--control", required=True)
    ap.add_argument("--preflight", required=True, help="preflight JSON; must carry VERDICT PASS")
    ap.add_argument("--out-prefix", required=True)
    a = ap.parse_args()
    import anndata as ad, scanpy as sc, scvelo as scv

    pf = json.load(open(a.preflight))
    if pf.get("VERDICT") != "PASS":
        raise SystemExit(f"REFUSING: preflight VERDICT is {pf.get('VERDICT')}, not PASS. Scoring is "
                         f"licensed only by a passed preflight.")
    npz_path = os.path.join(os.path.dirname(os.path.abspath(a.preflight)),
                            pf["frozen_pairs_artifact"])
    if sha_file(npz_path) != pf["frozen_pairs_sha256"]:
        raise SystemExit("REFUSING: frozen pair artifact sha256 does not match the preflight record")
    if sha_file(a.control) != pf["control_input_sha256"]:
        raise SystemExit("REFUSING: control object differs from the one the preflight certified")
    ps = pf.get("pairspace_module")
    if not ps or sha_file(ps) != pf.get("pairspace_module_sha256"):
        raise SystemExit("REFUSING: the resolved pair-builder module differs from the certified one")
    here = os.path.dirname(os.path.abspath(__file__))
    for mod, key in (("gb_gata1_preflight.py", "self_sha256"),
                     ("gb_gata1_matching.py", "matching_module_sha256"),
                     (os.path.abspath(__file__), "freezer_module_sha256")):
        want = pf.get(key)
        have = (_SELF_SHA if os.path.isabs(mod)
                else sha_file(os.path.join(here, mod)))
        if not want or have != want:
            raise SystemExit(f"REFUSING: {mod} sha256 {have} does not match the certified "
                             f"{want}; the preflight must be re-run against the current code")
    import types as _t
    _pf_path = os.path.join(here, "gb_gata1_preflight.py")
    with open(_pf_path, "rb") as _fh:
        _src = _fh.read()
    if hashlib.sha256(_src).hexdigest() != pf.get("self_sha256"):
        raise SystemExit("REFUSING: preflight source bytes do not match the certified hash")
    _PF = _t.ModuleType("_gb_pf_verified")
    _PF.__file__ = _pf_path
    _PF.__dict__["_VERIFIED_SELF_SHA"] = hashlib.sha256(_src).hexdigest()
    exec(compile(_src, _pf_path, "exec"), _PF.__dict__)
    EXPECTED_FEATURES = _PF.FEATURES
    if list(pf.get("features") or []) != list(EXPECTED_FEATURES):
        raise SystemExit(f"REFUSING: feature schema mismatch.\n  certified {pf.get('features')}\n"
                         f"  current   {list(EXPECTED_FEATURES)}")
    EXP_N_MIN, EXP_GATE = _PF.N_MIN, _PF.SMD_GATE
    EXP_MPC, EXP_REG = _PF.MAX_PAIRS_PER_CELL, _PF.REGULATOR
    EXP_ROOTS, EXP_SEED = _PF.ROOTS, _PF.SEED
    consts = pf.get("constants") or {}
    expected = {"N_MIN": EXP_N_MIN, "SMD_GATE": EXP_GATE, "MAX_PAIRS_PER_CELL": EXP_MPC,
                "REGULATOR": EXP_REG, "ROOTS": list(EXP_ROOTS), "SEED": EXP_SEED}
    for key, want in expected.items():
        have = consts.get(key)
        if isinstance(want, list):
            same = list(have or []) == want
        elif isinstance(want, (int, float)):
            same = (isinstance(have, (int, float)) and np.isfinite(have)
                    and float(have) == float(want))
        else:
            same = have == want
        if not same:
            raise SystemExit(f"REFUSING: certified constant {key} is {have!r}, expected exactly "
                             f"{want!r}; the gate that licensed this run is not the current gate")
    if list(EXP_ROOTS) != list(ROOTS):
        raise SystemExit(f"REFUSING: freezer root family {ROOTS} differs from preflight {EXP_ROOTS}")

    Z = np.load(npz_path, allow_pickle=False)
    W = ad.read_h5ad(a.control)
    genes = np.asarray(W.var_names).astype(str)
    if list(genes) != list(Z["genes"].astype(str)):
        raise SystemExit("REFUSING: control gene order differs from the certified artifact")
    if list(np.asarray(W.obs_names).astype(str)) != list(Z["cell_ids"].astype(str)):
        raise SystemExit("REFUSING: control cell order differs from the certified artifact")

    reg = int(Z["regulator_index"][0])
    pseudo = np.asarray(Z["panel"], int)
    if pseudo.size < 100:
        raise SystemExit(f"REFUSING: panel has {pseudo.size} members, fewer than 100")
    panel = np.concatenate([[reg], pseudo])
    # one identical evaluation universe for every arm
    excluded = np.zeros(len(genes), bool)
    excluded[panel] = True

    X = to_dense(W.X).astype(np.float64)
    V = to_dense(W.layers["velocity"]).astype(np.float64)
    if not (np.isfinite(X).all() and np.isfinite(V).all()):
        raise SystemExit("REFUSING: non-finite X or V; the preflight should have caught this")

    if "valid_cells" not in Z.files:
        raise SystemExit("REFUSING: the certified artifact carries no valid_cells record; gate D "
                         "proof is required before any grid cell may contribute")
    valid = set(map(str, np.asarray(Z["valid_cells"]).tolist()))
    if not valid:
        raise SystemExit("REFUSING: the certified artifact records ZERO row-permutation-invariant "
                         "grid cells; there is nothing that may contribute to the max-over-grid")
    S_all, dropped = {}, []
    for root in ROOTS:
        I = np.asarray(Z[f"I|{root}"], int)
        J = np.asarray(Z[f"J|{root}"], int)
        for op in OPERATORS:
            for k, S in scores_for_operator(X, V, I, J, panel, op):
                key = f"{root}|{op}|k={k}"
                if key not in valid:
                    dropped.append(key)
                    continue
                S_all[key] = S.astype(np.float32)
                print(f"[freeze] {root} {op:12s} k={k:4d} npairs={len(I)}", flush=True)
    if dropped:
        print(f"[freeze] {len(dropped)} cells dropped as MISSING (gate D): {dropped}", flush=True)

    npz = a.out_prefix + "_scores.npz"
    np.savez_compressed(npz, panel=panel, excluded=excluded, genes=genes, **S_all)
    sha = sha_file(npz)
    meta = {"stage": "1_control_only_freezer", "script": "gb_gata1_freeze_scores.py",
            "self_sha256": _SELF_SHA, "self_sha256_provenance": _SELF_SHA_SOURCE,
            "artifact": os.path.basename(npz), "artifact_sha256": sha,
            "control_input": os.path.abspath(a.control),
            "control_input_sha256": sha_file(a.control),
            "preflight": os.path.abspath(a.preflight),
            "preflight_sha256": sha_file(a.preflight),
            "preflight_verdict": pf["VERDICT"],
            "frozen_pairs_sha256": pf["frozen_pairs_sha256"],
            "pairspace_module_sha256": pf.get("pairspace_module_sha256"),
            "regulator_catalogue_sha256": pf.get("regulator_catalogue_sha256"),
            "operators": OPERATORS, "k_grid": K_GRID, "roots": ROOTS, "eps": EPS,
            "regulator_index": reg, "n_pseudo": int(pseudo.size),
            "pseudo_genes": [str(genes[i]) for i in pseudo],
            # carried forward so stage 2 can never read a descriptive panel as a specificity null
            "SPECIFICITY_ENDPOINT": pf.get("SPECIFICITY_ENDPOINT"),
            "excluded_from_evaluation_universe": int(excluded.sum()),
            "n_pairs": {root: int(len(np.asarray(Z[f'I|{root}'], int))) for root in ROOTS},
            "grid_cells_scored": sorted(S_all.keys()),
            "grid_cells_dropped_missing": sorted(dropped),
            "MISSING_RULE": ("cells failing gate D row-permutation invariance are absent "
                             "from the artifact entirely, so they are MISSING for the "
                             "regulator and every pseudo arm alike, never zero"),
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scanpy": sc.__version__, "scvelo": scv.__version__},
            "NO_SELECTOR_HERE": ("this stage contains no pseudo-regulator selection and no pair "
                                 "construction; both are read from the certified preflight artifact "
                                 "so exactly one gated path exists"),
            "NO_TRUTH_HERE": "this executable has no KO, ChIP or DepMap argument"}
    with open(a.out_prefix + "_meta.json", "w") as fh:
        json.dump(meta, fh, indent=1)
    print(f"[frozen] {npz}\n[sha256] {sha}", flush=True)


if __name__ == "__main__":
    main()
