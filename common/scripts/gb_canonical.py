#!/usr/bin/env python
"""CANONICAL PREPROCESSING BUILDER + FAIL-CLOSED VALIDATOR. ONE CODEPATH FOR EVERY DATASET.

usage:
  gb_canonical.py build    <out_dir> --system GATA1|PANC|BONEMARROW [--config ...]
  gb_canonical.py validate <h5ad>    [--config ...]
  gb_canonical.py selftest

Implements `GEOBRIDGE_PREPROCESS_V1.json`. Every decision lives in that config; nothing here is
dataset-specific except the raw input path and the species TF catalogue, which are biological-design
metadata rather than performance choices.

WHY THIS EXISTS. The three development fixtures store a LOG-NORMALISED `.X` (max 6.6-7.6) while their
`velocity` layer is a rate in `Ms` units (max 457-1433). Every GeoBRIDGE number computed on them took
dX from a log matrix and dV from a native-rate matrix. scVelo 0.3.4 defines velocity as `Mu - gamma*Ms`
(tools/velocity.py:79,167) over `Ms = connectivities . spliced` with no log (preprocessing/moments.py:89-95),
and its own velocity-vs-state tooling uses `xkey="Ms"` (tools/velocity_graph.py:49). So the coherent
representation is dMs against dvelocity, and the legacy pairing is a hybrid the contract prohibits.

FAIL CLOSED. `validate()` raises on any violation. It is called at the end of `build()` and is meant to
be called by every downstream script before it computes anything. It checks NUMERICAL semantics, not
just metadata, because metadata can lie: a layer named "Ms" is not accepted as equivalent to another
dataset's "Ms" until its construction is verified.

NO TRUTH IS READ ANYWHERE IN THIS FILE.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np, pandas as pd
import anndata as ad
from scipy.sparse import issparse

R = __import__("os").environ["GEOBRIDGE_ROOT"]
sys.path.insert(0, f"{R}/scratch"); sys.path.insert(0, R)
CONFIG = f"{R}/GEOBRIDGE_PREPROCESS_V1.json"

# raw sources verified PRIMARY-COMPATIBLE (integer spliced+unspliced) by gb_preprocess_audit.py
RAW = {
    "GATA1":      (f"{R}/results/canonical_velocity_builder_fixture_2026_08_15/anchor_rebuild_input.h5ad", "mouse"),
    "PANC":       (f"{R}/data/Pancreas/endocrinogenesis_day15.h5ad", "mouse"),
    "BONEMARROW": (f"{R}/data/BoneMarrow/human_cd34_bone_marrow.h5ad", "human"),
    # additional PRIMARY-COMPATIBLE datasets from results/preprocess_audit_2026_09_18 (integer spliced+unspliced)
    "GASTRULATION": (f"{R}/data/Gastrulation/gastrulation_e75.h5ad", "mouse"),
    "DENTATEGYRUS": (f"{R}/data/DentateGyrus/dentategyrus_raw.h5ad", "mouse"),
    # the 18-LIBRARY control arm (1752 cells), not the 736-cell single-library file. gb_k562_prep.py
    # used the 18-library arm deliberately; building from the smaller one discards 58% of the cells.
    "K562":         (f"{R}/results/k562_control_only_18.h5ad", "human"),
    "RPE1":         (f"{R}/results/rpe1_ntc_assembled_2026_08_23/rpe1_ntc_spliced_unspliced.h5ad", "human"),
    # 2026-09-23 lane L3b: integer counts recovered EXACTLY from the RegVelo object (data/zebrafish_raw/RECEIPT.json)
    "ZEBRAFISH":    (f"{R}/data/zebrafish_raw/zebrafish_nc_raw_recovered.h5ad", "zebrafish"),
}
# species absent from the frozen config's tf_retention (the config is hash-bound by every artifact, so it is not edited)
EXTRA_TF_CATALOGUE = {"zebrafish": "data/zebrafish_raw/zebrafish_tf_catalogue.txt"}
d_ = lambda M: np.asarray(M.todense()) if issparse(M) else np.asarray(M)


def _hash(M, n=4000):
    A = d_(M[:n]).astype(np.float64)
    return hashlib.sha256(np.ascontiguousarray(np.nan_to_num(A)).tobytes()).hexdigest()[:16]


class ContractViolation(RuntimeError):
    pass


def validate(a, cfg, where="validate"):
    """FAIL-CLOSED contract validator. Raises ContractViolation on the first failure."""
    fails = []
    req = ["Ms", "Mu", "velocity", "spliced", "unspliced"]
    for l in req:
        if l not in a.layers: fails.append(f"missing required layer '{l}'")
    if fails: raise ContractViolation(f"[{where}] " + "; ".join(fails))

    Ms, V = d_(a.layers["Ms"]), d_(a.layers["velocity"])
    Mu = d_(a.layers["Mu"])          # 2026-09-19: Mu was never bound here, so it was never validated
    X = d_(a.X)
    sp, un = d_(a.layers["spliced"]), d_(a.layers["unspliced"])

    # 1. shapes and ordering
    for l in req:
        if a.layers[l].shape != a.shape: fails.append(f"layer '{l}' shape {a.layers[l].shape} != {a.shape}")
    if len(set(a.var_names)) != a.n_vars: fails.append("var_names are not unique - gene ordering cannot be trusted")

    # 2. finiteness (velocity legitimately carries NaN for unfitted genes; nothing else may)
    # 2026-09-19 (internal review): the "Mu" entry previously passed `Ms`, so Mu finiteness was NEVER validated.
    for nm, M in (("Ms", Ms), ("Mu", Mu), ("spliced", sp), ("unspliced", un), ("X", X)):
        if not np.isfinite(M).all(): fails.append(f"'{nm}' contains non-finite values")

    # 3. Ms MUST NOT be log-transformed - this is the defect the contract exists to prevent
    if Ms.max() < 30:
        fails.append(f"Ms.max()={Ms.max():.3f} < 30: Ms looks LOG-TRANSFORMED or rescaled; "
                     f"velocity is defined in linear Ms units so the representation would be incoherent")
    # 4. .X is expected to be log1p'd (state space only); flag the inconsistency either way
    x_log = X.max() < 30
    if ("log1p" in a.uns) != x_log:
        fails.append(f"uns['log1p'] present={'log1p' in a.uns} but X.max()={X.max():.3f} "
                     f"({'looks logged' if x_log else 'does NOT look logged'}) - metadata contradicts the numbers")
    # 5. spliced/unspliced must be on the normalised LINEAR scale (moments are built from them)
    for nm, M in (("spliced", sp), ("unspliced", un)):
        if M.max() < 30:
            fails.append(f"'{nm}'.max()={M.max():.3f} < 30: looks log-transformed; "
                         f"moments() would then build a logged Ms")
    # 6. Ms must actually track spliced (it is a kNN smoothing of it), and be strictly denser
    if (sp == 0).mean() <= (Ms == 0).mean():
        fails.append(f"Ms zero-fraction {(Ms==0).mean():.3f} is not below spliced {(sp==0).mean():.3f}; "
                     f"Ms does not look like a smoothing of spliced")
    # 7. velocity must be a rate in Ms units: signed, and comparable in magnitude to Ms
    if (V < 0).sum() == 0: fails.append("velocity has no negative entries - not a signed rate")
    nzV, nzM = np.abs(V[np.isfinite(V) & (V != 0)]), Ms[Ms != 0]
    if nzV.size and nzM.size:
        ratio = float(np.median(nzV) / np.median(nzM))
        if not (1e-4 < ratio < 1e2):
            fails.append(f"median|velocity|/median Ms = {ratio:.2e} is outside [1e-4, 1e2]; "
                         f"velocity does not look like a rate in Ms units")
    # 8. provenance receipt must exist and must name the canonical layers
    rec = a.uns.get("gb_contract")
    if rec is None:
        fails.append("uns['gb_contract'] missing - provenance receipt is required, no silent fallback")
    else:
        rec = dict(rec)
        if rec.get("source_matrix_for_dX") != cfg["geobridge_representation"]["source_matrix_for_dX"]:
            fails.append(f"receipt source_matrix_for_dX={rec.get('source_matrix_for_dX')!r} != contract")
        if rec.get("target_matrix_for_dV") != cfg["geobridge_representation"]["target_matrix_for_dV"]:
            fails.append(f"receipt target_matrix_for_dV={rec.get('target_matrix_for_dV')!r} != contract")
    if fails:
        raise ContractViolation(f"[{where}] CONTRACT VIOLATION:\n  - " + "\n  - ".join(fails))
    return True


def build(out_dir, system, cfg):
    import scanpy as sc, scvelo as scv
    t0 = time.time(); os.makedirs(out_dir, exist_ok=True)
    src, species = RAW[system]
    a = ad.read_h5ad(src)
    print(f"{system}: raw {a.shape} from {os.path.basename(src)}", flush=True)
    for l in ("spliced", "unspliced"):
        M = d_(a.layers[l][:3000])
        if not np.allclose(M[M != 0], np.round(M[M != 0]), atol=1e-6):
            raise ContractViolation(f"[build] '{l}' is not integer counts; {system} is not PRIMARY-COMPATIBLE. "
                                    f"Classify it SENSITIVITY-ONLY / GEOMETRY-ONLY and EXCLUDE it - do not substitute a layer.")

    cat_path = cfg["tf_retention"].get(species) or EXTRA_TF_CATALOGUE[species]
    cat = {l.strip().upper() for l in open(f"{R}/{cat_path}" if not cat_path.startswith("/") else cat_path) if l.strip()}
    fn = cfg["filter_and_normalize"]
    # scVelo 0.3.4's filter_and_normalize does ONLY filter_genes + normalize_per_cell -- it performs
    # NEITHER HVG selection NOR log1p (preprocessing/utils.py:389-405; **kwargs are forwarded to
    # normalize_per_cell, which is why passing n_top_genes/log raises). scv.pp.filter_genes_dispersion
    # and scv.pp.log1p do not exist in this version. Each step is therefore called EXPLICITLY, so the
    # contract cannot depend on an undocumented convenience wrapper.
    retain = [g for g in a.var_names if str(g).upper() in cat]
    scv.pp.filter_genes(a, min_shared_counts=fn["min_shared_counts"], retain_genes=retain)
    scv.pp.normalize_per_cell(a)                    # normalises X, spliced and unspliced (utils.py:271)
    # HVG on a LOG view, without logging the stored layers: spliced/unspliced must stay linear because
    # moments() builds Ms/Mu from them (moments.py:89-95).
    tmp = ad.AnnData(a.X.copy()); tmp.var_names = a.var_names
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=fn["n_top_genes"], flavor="seurat")
    hv = np.asarray(tmp.var["highly_variable"]).astype(bool); del tmp
    keep = hv | np.isin(np.char.upper(np.asarray(a.var_names).astype(str)), list(cat))
    a = a[:, keep].copy()
    if fn["log"]:
        sc.pp.log1p(a)                              # .X ONLY; layer= not passed, so layers untouched
    n_tf = int(np.isin(np.char.upper(np.asarray(a.var_names).astype(str)), list(cat)).sum())
    print(f"  after filter+HVG+TF retention: {a.shape}, {n_tf} catalogue TFs  {time.time()-t0:.0f}s", flush=True)

    p, nb, ve = cfg["pca"], cfg["neighbors"], cfg["velocity"]
    sc.pp.pca(a, n_comps=p["n_comps"], zero_center=p["zero_center"], random_state=p["random_state"])
    sc.pp.neighbors(a, n_neighbors=nb["n_neighbors"], n_pcs=nb["n_pcs"], random_state=nb["random_state"])
    scv.pp.moments(a, n_pcs=None, n_neighbors=None)
    scv.tl.velocity(a, mode=ve["mode"], fit_offset=ve["fit_offset"], perc=ve["perc"])
    print(f"  moments + velocity done {time.time()-t0:.0f}s", flush=True)

    a.uns["gb_contract"] = {
        "contract": cfg["contract"], "system": system, "species": species,
        "raw_source": src, "raw_sha_head": _hash(ad.read_h5ad(src, backed="r").layers["spliced"]),
        **cfg["geobridge_representation"],
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "Ms_sha": _hash(a.layers["Ms"]), "velocity_sha": _hash(a.layers["velocity"]),
        "n_catalogue_TFs": n_tf}
    validate(a, cfg, where=f"build:{system}")
    out = f"{out_dir}/{system}_canonical.h5ad"
    a.write_h5ad(out)
    json.dump(dict(a.uns["gb_contract"], shape=list(a.shape), out=out),
              open(f"{out_dir}/{system}_receipt.json", "w"), indent=1, default=str)
    print(f"  VALIDATED and wrote {out}  ({a.shape}) in {time.time()-t0:.0f}s", flush=True)
    return out


def build_pairs(out_dir, system, cfg, config_path=CONFIG):
    """Canonical pair construction. STATE SPACE is the log .X (standard scanpy/scVelo practice for
    PCA/neighbours/diffusion); the GeoBRIDGE SOURCE MATRIX is Ms. Those are different roles and the
    contract assigns them separately -- the pair INDICES come from the state space, D_X = dMs is formed
    on those indices."""
    sys.path.insert(0, f"{R}/pairspace")
    from gb_gata1_preflight import canonical_pairs
    t0 = time.time()
    h5 = f"{out_dir}/{system}_canonical.h5ad"
    a = ad.read_h5ad(h5)
    validate(a, cfg, where=f"pairs:{system}")
    X = d_(a.X).astype(np.float64)                      # log state space, for trajectory only
    V = np.nan_to_num(d_(a.layers["velocity"]).astype(np.float64), nan=0.0)
    ids = np.asarray(a.obs_names).astype(str)
    if len(set(ids)) != len(ids):
        raise SystemExit(f"REFUSING {system}: the fixture has duplicate cell ids, so pairs cannot "
                         f"be bound to cells unambiguously")
    pc = cfg["pair_construction"]
    out = {"genes": np.asarray(a.var_names).astype(str),
           "builder_config": json.dumps(pc), "preprocess_contract": cfg.get("contract", "")}
    # 2026-09-20 PROVENANCE REPAIR v2, after internal review rejected v1 and the PI froze the production pair
    # operator (an internal lock amendment (not shipped)). Three defects are fixed here:
    # the frozen window/cap are PASSED rather than hardcoded in the builder; the fourth return value
    # is the CONNECTIVITY MATRIX, so real component LABELS are derived instead of storing a matrix
    # under the name "component"; and every contract violation now FAILS instead of printing.
    from scipy.sparse.csgraph import connected_components
    import scipy.sparse as _sp
    out["pair_operator"] = "DPT_window_cap"          # production per PI ruling 2026-09-20
    # internal review 2026-09-20 finding 6: hashing only this wrapper does not prove which bytes SELECTED the
    # pairs -- that work is delegated to geobridge_model/pairspace.py, byte-loaded and digested by
    # _pairspace_verified(). Record the digest of THOSE bytes, the wrapper, and the frozen config.
    import gb_gata1_preflight as _pf
    out["pairspace_builder_sha256"] = _pf._pairspace_verified()[1]
    out["pairspace_builder_path"] = _pf._pairspace_path()
    out["wrapper_sha256"] = hashlib.sha256(
        open(f"{R}/pairspace/gb_gata1_preflight.py", "rb").read()).hexdigest()
    out["config_sha256"] = hashlib.sha256(open(config_path, "rb").read()).hexdigest()
    out["config_path"] = config_path
    out["fixture_sha256"] = hashlib.sha256(open(h5, "rb").read()).hexdigest()
    for root in pc["roots"]:
        _pt, I, J, _C = canonical_pairs(X, V, ids, root, seed=pc["pair_seed"],
                                        min_forward_dt=pc["min_forward_dt"],
                                        max_forward_dt=pc["max_forward_dt"],
                                        max_pairs_per_cell=pc["max_pairs_per_cell"])
        _pt = np.asarray(_pt, dtype=np.float64)
        I = I.astype(np.int64); J = J.astype(np.int64)
        n = X.shape[0]
        if len(ids) != n:
            raise SystemExit(f"REFUSING {system}/{root}: {len(ids)} cell ids for {n} rows")
        if len(I) == 0 or len(I) != len(J):
            raise SystemExit(f"REFUSING {system}/{root}: {len(I)} I vs {len(J)} J pairs")
        if I.min() < 0 or J.min() < 0 or I.max() >= n or J.max() >= n:
            raise SystemExit(f"REFUSING {system}/{root}: pair index outside [0,{n})")
        dt = _pt[J] - _pt[I]
        if not np.all(dt > 0):
            raise SystemExit(f"REFUSING {system}/{root}: {int((dt <= 0).sum())} pairs are not "
                             f"forward in the ordering variable")
        lo, hi = float(pc["min_forward_dt"]), float(pc["max_forward_dt"])
        # EXACT predicate, matching monotonic_forward_windows in geobridge_model/pairspace.py:
        # searchsorted(pt+min, side="right") includes j only when dt > min (STRICT), and
        # searchsorted(pt+max, side="right") includes j when dt <= max (INCLUSIVE). A tolerance-
        # widened interval accepted dt == min, which the builder never emits (internal review 2026-09-20).
        if dt.min() <= lo or dt.max() > hi:
            raise SystemExit(f"REFUSING {system}/{root}: realised dt [{dt.min():.9f},{dt.max():.9f}] "
                             f"escapes the frozen window (min,max] = ({lo},{hi}]")
        # CANONICAL PAIR ORDER. Within a capped source the builder samples partners with rng.choice
        # and does not sort them, so a row permutation of the identical pair set would digest
        # differently. Order the stored arrays by (I, J) and carry dt with them, so the digest is a
        # property of the SET and consumers receive the same canonical order.
        _o = np.lexsort((J, I))
        I, J, dt = I[_o], J[_o], dt[_o]
        cap = int(pc["max_pairs_per_cell"]); worst = int(np.bincount(I, minlength=n).max())
        if worst > cap:
            raise SystemExit(f"REFUSING {system}/{root}: a source cell has {worst} pairs, cap is {cap}")
        # internal review 2026-09-20: the same mathematical matrix has many valid CSR encodings (unsorted
        # indices, duplicate entries, explicit zeros), so components, serialization and the digest
        # must all be taken on ONE canonical form or the digest is not a property of the graph.
        Cc = _sp.csr_matrix(_C)
        Cc.sum_duplicates(); Cc.eliminate_zeros(); Cc.sort_indices()
        ncomp, lab = connected_components(Cc, directed=False)
        out[f"{root}_I"], out[f"{root}_J"] = I, J
        out[f"{root}_pt"] = _pt
        out[f"{root}_dt"] = dt.astype(np.float64)
        out[f"{root}_component"] = np.asarray(lab, dtype=np.int32)          # LABELS, not the matrix
        out[f"{root}_conn_data"] = Cc.data.astype(np.float32)               # connectivity, encoded
        out[f"{root}_conn_indices"] = Cc.indices.astype(np.int32)
        out[f"{root}_conn_indptr"] = Cc.indptr.astype(np.int64)
        out[f"{root}_conn_shape"] = np.asarray(Cc.shape, dtype=np.int64)
        # internal review 2026-09-20: component labels cannot detect a connectivity edit that preserves the
        # partition, so bind a canonical digest of the CSR triple itself.
        _h = hashlib.sha256()
        for _a in (Cc.data.astype(np.float32), Cc.indices.astype(np.int32),
                   Cc.indptr.astype(np.int64), np.asarray(Cc.shape, dtype=np.int64)):
            _h.update(np.ascontiguousarray(_a).tobytes())
        out[f"{root}_conn_sha256"] = _h.hexdigest()
        out[f"{root}_conn_canonical"] = np.asarray("sum_duplicates+eliminate_zeros+sort_indices")
        out[f"{root}_cells"] = ids
        # internal review 2026-09-20: swapping one eligible pair for a DIFFERENT eligible pair leaves every
        # structural invariant intact, so the pair set must be digest-bound to be verifiable.
        _ph = hashlib.sha256()
        for _a in (I.astype(np.int64), J.astype(np.int64)):
            _ph.update(np.ascontiguousarray(_a).tobytes())
        out[f"{root}_pairs_sha256"] = _ph.hexdigest()
        print(f"  {system}/{root}: {len(I)} pairs, dt [{dt.min():.5f},{dt.max():.5f}] within "
              f"[{lo},{hi}], max pairs/cell {worst} <= {cap}, {ncomp} component(s)  "
              f"{time.time()-t0:.0f}s", flush=True)
    np.savez_compressed(f"{out_dir}/{system}_canonical_pairs.npz", **out)
    print(f"  wrote {out_dir}/{system}_canonical_pairs.npz", flush=True)


def reconstruct(out_dir, system, cfg, config_path=CONFIG):
    """Regenerate the pairs from the bound fixture/config/code and compare EXACTLY to the artifact.

    internal review 2026-09-20: a digest stored INSIDE the artifact can be recomputed by whoever edits the
    artifact, so it proves nothing on its own. This runs the real builder again and writes an
    EXTERNAL receipt keyed to the artifact's current sha256. Any edit to the NPZ -- including one
    that helpfully updates the internal digests -- changes that sha256 and the loader refuses.
    It is deliberately NOT run on every load: it re-runs PCA/neighbours/diffmap/DPT."""
    sys.path.insert(0, f"{R}/pairspace")
    import gb_gata1_preflight as _pf
    from gb_gata1_preflight import canonical_pairs
    import scipy.sparse as _sp
    from scipy.sparse.csgraph import connected_components

    h5 = f"{out_dir}/{system}_canonical.h5ad"
    npz = f"{out_dir}/{system}_canonical_pairs.npz"
    a = ad.read_h5ad(h5); validate(a, cfg, where=f"reconstruct:{system}")
    X = d_(a.X).astype(np.float64)
    V = np.nan_to_num(d_(a.layers["velocity"]).astype(np.float64), nan=0.0)
    ids = np.asarray(a.obs_names).astype(str)
    Z = np.load(npz, allow_pickle=True)
    pc = cfg["pair_construction"]
    rep = dict(system=system, artifact=npz,
               artifact_sha256=hashlib.sha256(open(npz, "rb").read()).hexdigest(),
               fixture_sha256=hashlib.sha256(open(h5, "rb").read()).hexdigest(),
               config_sha256=hashlib.sha256(open(config_path, "rb").read()).hexdigest(),
               config_path=config_path,
               wrapper_sha256=hashlib.sha256(
                   open(f"{R}/pairspace/gb_gata1_preflight.py", "rb").read()).hexdigest(),
               pairspace_builder_sha256=_pf._pairspace_verified()[1],
               reconstructor_sha256=hashlib.sha256(open(__file__, "rb").read()).hexdigest(),
               roots={})
    allok = True
    for root in pc["roots"]:
        pt2, I2, J2, C2 = canonical_pairs(X, V, ids, root, seed=pc["pair_seed"],
                                          min_forward_dt=pc["min_forward_dt"],
                                          max_forward_dt=pc["max_forward_dt"],
                                          max_pairs_per_cell=pc["max_pairs_per_cell"])
        o = np.lexsort((J2, I2)); I2, J2 = I2[o].astype(np.int64), J2[o].astype(np.int64)
        Cc = _sp.csr_matrix(C2); Cc.sum_duplicates(); Cc.eliminate_zeros(); Cc.sort_indices()
        n2, lab2 = connected_components(Cc, directed=False)
        r = dict(
            pt=bool(np.array_equal(np.asarray(Z[f"{root}_pt"], np.float64),
                                   np.asarray(pt2, np.float64))),
            I=bool(np.array_equal(np.asarray(Z[f"{root}_I"], np.int64), I2)),
            J=bool(np.array_equal(np.asarray(Z[f"{root}_J"], np.int64), J2)),
            conn=bool(np.array_equal(np.asarray(Z[f"{root}_conn_indices"]), Cc.indices)
                      and np.array_equal(np.asarray(Z[f"{root}_conn_indptr"]), Cc.indptr)
                      and np.allclose(np.asarray(Z[f"{root}_conn_data"], np.float64),
                                      Cc.data.astype(np.float64), rtol=0, atol=0)),
            components=bool(len(np.unique(np.stack([
                np.unique(np.asarray(Z[f"{root}_component"]).ravel(), return_inverse=True)[1],
                np.unique(lab2, return_inverse=True)[1]], 1), axis=0)) == n2))
        r["matches"] = all(r.values()); allok = allok and r["matches"]
        rep["roots"][root] = r
        print(f"  {system}/{root}: reconstruction {'MATCHES' if r['matches'] else 'DIFFERS'} {r}",
              flush=True)
    rep["reconstruction_matches"] = bool(allok)
    out = f"{out_dir}/{system}_canonical_pairs.receipt.json"
    json.dump(rep, open(out, "w"), indent=1)
    print(f"  wrote {out}  reconstruction_matches={allok}")
    if not allok:
        raise SystemExit(f"REFUSING {system}: the artifact is not the deterministic builder output")


def selftest():
    """The validator must REJECT each way the legacy representation is broken, and ACCEPT a clean one."""
    cfg = json.load(open(CONFIG))
    rng = np.random.default_rng(0); n, g = 200, 50
    sp = rng.poisson(8, (n, g)).astype(float) * 20
    un = rng.poisson(4, (n, g)).astype(float) * 20
    Ms = sp + rng.normal(0, 1, (n, g)); Ms = np.abs(Ms)
    Ms[rng.random((n, g)) < 0.02] = 0.0                  # denser than spliced
    sp[rng.random((n, g)) < 0.4] = 0.0
    good = ad.AnnData(np.log1p(sp))
    good.uns["log1p"] = {}
    for k, v in (("spliced", sp), ("unspliced", un), ("Ms", Ms), ("Mu", un + 1.0),
                 ("velocity", rng.normal(0, 3, (n, g)))):
        good.layers[k] = v
    good.uns["gb_contract"] = {"source_matrix_for_dX": "Ms", "target_matrix_for_dV": "velocity"}
    assert validate(good, cfg, "selftest-good")
    print("  clean case ACCEPTED")

    def must_reject(mut, label):
        b = good.copy(); mut(b)
        try:
            validate(b, cfg, "selftest"); raise AssertionError(f"validator ACCEPTED a broken case: {label}")
        except ContractViolation as e:
            print(f"  REJECTED {label}: {str(e).splitlines()[-1].strip()[:90]}")
    must_reject(lambda b: b.layers.__setitem__("Ms", np.log1p(b.layers["Ms"])), "log-transformed Ms")
    must_reject(lambda b: b.layers.__setitem__("spliced", np.log1p(b.layers["spliced"])), "log-transformed spliced")
    must_reject(lambda b: b.layers.__setitem__("velocity", np.abs(b.layers["velocity"])), "unsigned velocity")
    must_reject(lambda b: b.layers.__setitem__("velocity", b.layers["velocity"] * 1e-8), "velocity not in Ms units")
    must_reject(lambda b: b.uns.pop("gb_contract"), "missing provenance receipt")
    must_reject(lambda b: b.uns.__setitem__("gb_contract", {"source_matrix_for_dX": "X", "target_matrix_for_dV": "velocity"}),
                "receipt naming .X as the dX source")
    must_reject(lambda b: b.layers.pop("Ms"), "missing Ms layer")
    print("selftest PASS")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["build", "pairs", "reconstruct", "validate", "selftest"])
    ap.add_argument("target", nargs="?")
    ap.add_argument("--system"); ap.add_argument("--config", default=CONFIG)
    a = ap.parse_args()
    if a.action == "selftest": selftest(); return
    cfg = json.load(open(a.config))
    if a.action == "build":
        build(a.target, a.system, cfg)
    elif a.action == "pairs":
        build_pairs(a.target, a.system, cfg, config_path=a.config)
    elif a.action == "reconstruct":
        reconstruct(a.target, a.system, cfg, config_path=a.config)
    else:
        validate(ad.read_h5ad(a.target), cfg, where=os.path.basename(a.target))
        print(f"CONTRACT OK: {a.target}")


if __name__ == "__main__":
    main()
