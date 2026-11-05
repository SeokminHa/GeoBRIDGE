"""One loader for the three frozen substrates, so the k_geom / entanglement / dense-k scripts cannot
disagree about which file or which pair set they read. Nothing here touches truth."""
import os, sys, numpy as np, anndata as ad
from scipy.sparse import issparse

R = __import__("os").environ["GEOBRIDGE_ROOT"]
for _p in (f"{R}/scratch", f"{R}/pairspace", R):
    if _p not in sys.path: sys.path.insert(0, _p)

ROOTS = ("dc1_min", "dc1_max")
d_ = lambda M: np.asarray(M.todense()) if issparse(M) else np.asarray(M)


CANON = f"{R}/results/canonical_2026_09_18"


def paths(system, canonical=False):
    s = system.upper()
    if canonical:
        cat = f"{R}/GB_real_v2/lambert2018_human_tfs.txt" if s in ("BONEMARROW", "K562", "RPE1") \
            else f"{R}/data/zebrafish_raw/zebrafish_tf_catalogue.txt" if s == "ZEBRAFISH" \
            else f"{R}/data/LaraAstiaso_raw/ext_ref/external_regulators.txt"
        return f"{CANON}/{s}_canonical.h5ad", f"{CANON}/{s}_canonical_pairs.npz", cat
    if s == "BONEMARROW":
        B = f"{R}/results/bonemarrow_substrate_2026_09_18"
        return (f"{B}/bonemarrow_velocity.h5ad", f"{B}/bonemarrow_frozen_pairs.npz",
                f"{R}/GB_real_v2/lambert2018_human_tfs.txt")
    if s == "GATA1":
        import gb_hist_specificity_family as H
        return H.FIXTURE, H.PAIRS, f"{R}/data/LaraAstiaso_raw/ext_ref/external_regulators.txt"
    return (f"{R}/results/pr_arm2_2026_08_16/panc/panc_pr_stochastic.h5ad",
            f"{R}/results/panc_pr_arm2_surfaces_2026_08_16/panc_pr_frozen_pairs.npz",
            f"{R}/data/LaraAstiaso_raw/ext_ref/external_regulators.txt")


def load(system, max_pairs=0, seed=0, layer="X", canonical=False):
    """Returns dict(genes, X, V, pairs={root: (I, J)}, obs, is_tf). max_pairs>0 subsamples pairs
    (SMALL-N smoke only) with a fixed seed.

    `layer` picks the SOURCE MATRIX. This matters: the fixtures store a LOG-NORMALISED unsmoothed
    A.X while `velocity` is in Ms units, and gb_layer_scale.py showed the log transform -- not the
    kNN smoothing -- is what decides whether removing the top-k helps. Any rank estimated from D_X
    must be estimated on the SAME matrix the operator will act on, or the two are incomparable.
      X       A.X as stored (log-normalised)   -- current operator input
      Ms      A.layers["Ms"] (smoothed, raw scale) -- historical operator input, matches velocity units
      Ms_log  log1p(Ms): smoothing WITHOUT the scale change"""
    fix, prs, cat = paths(system, canonical=canonical)
    A = ad.read_h5ad(fix); f = np.load(prs)
    if canonical:
        # FAIL CLOSED, with STABLE REFUSAL CODES. Every refusal below is `REFUSING[CODE] ...`; the
        # mutation battery matches the CODE, not prose, so rewording a message can never silently
        # turn a mutation test into a pass (internal review 2026-09-20 finding 8).
        import json as _json, hashlib as _hl, gb_canonical as _gc
        import scipy.sparse as _sp
        from scipy.sparse.csgraph import connected_components as _cc
        import gb_gata1_preflight as _pf

        def _no(code, msg):
            raise SystemExit(f"REFUSING[{code}] load:{system}: {msg}")

        _cfg = _json.load(open(_gc.CONFIG))
        _gc.validate(A, _cfg, where=f"load:{system}")
        _pc = _cfg["pair_construction"]
        _ids = np.asarray(A.obs_names).astype(str)
        _n_cells = A.n_obs
        if len(set(_ids)) != len(_ids):
            _no("CELL_IDS_NOT_UNIQUE", "the fixture has duplicate cell ids, so a pair artifact "
                                       "cannot be matched to it unambiguously")
        if "builder_config" not in f.files:
            _no("ARTIFACT_PREDATES_REPAIR", "pair artifact has no builder_config; regenerate it "
                                            "with gb_canonical.py")
        _bc = _json.loads(str(f["builder_config"]))
        if _bc != _pc:                      # EXACT object equality, not a four-key spot check
            _diff = sorted(set(_bc) ^ set(_pc)) or \
                sorted(k for k in _pc if _bc.get(k) != _pc.get(k))
            _no("BUILDER_CONFIG_MISMATCH", f"artifact builder_config differs from the frozen "
                                           f"pair_construction on {_diff}")
        if str(f["preprocess_contract"]) != str(_cfg.get("contract", "")):
            _no("PREPROCESS_CONTRACT_MISMATCH",
                f"artifact contract {str(f['preprocess_contract'])!r} != {_cfg.get('contract', '')!r}")
        if "pair_operator" not in f.files:
            _no("NO_PAIR_OPERATOR", "artifact records no pair_operator")
        if str(f["pair_operator"]) != "DPT_window_cap":
            _no("WRONG_PAIR_OPERATOR",
                f"pair_operator {str(f['pair_operator'])!r} is not the production operator frozen "
                f"by an internal lock amendment (not shipped)")
        if str(f["pairspace_builder_path"]) != _pf._pairspace_path():
            _no("BUILDER_PATH_MISMATCH",
                f"artifact builder path {str(f['pairspace_builder_path'])!r} != "
                f"{_pf._pairspace_path()!r}")
        # internal review 2026-09-20: an internal digest can be regenerated by whoever edits the artifact. The
        # binding evidence is an EXTERNAL receipt written by a real reconstruction and keyed to the
        # artifact's sha256, so editing the NPZ (even self-consistently) invalidates it.
        _rp = os.path.splitext(prs)[0] + ".receipt.json"
        if not os.path.exists(_rp):
            _no("NO_RECONSTRUCTION_RECEIPT",
                f"no reconstruction receipt at {_rp}; run `gb_canonical.py reconstruct`")
        _rec = _json.load(open(_rp))
        if _rec.get("artifact_sha256") != _hl.sha256(open(prs, "rb").read()).hexdigest():
            _no("RECEIPT_ARTIFACT_MISMATCH",
                "the reconstruction receipt was written for a different artifact than the one on "
                "disk; the pair file has been modified since it was verified")
        if not _rec.get("reconstruction_matches"):
            _no("RECONSTRUCTION_FAILED",
                f"the receipt records that the artifact is NOT the deterministic builder output: "
                f"{_rec.get('roots')}")
        if "config_path" not in f.files:
            _no("MISSING_PROVENANCE_FIELD", "artifact is missing config_path")
        if str(f["config_path"]) != _gc.CONFIG:
            _no("CONFIG_PATH_MISMATCH",
                f"artifact was built from {str(f['config_path'])!r}, not the frozen contract "
                f"{_gc.CONFIG!r}; a custom config can change non-pair preprocessing")
        _sha = lambda q: _hl.sha256(open(q, "rb").read()).hexdigest()
        for _field, _actual, _code in (
                ("config_sha256", _sha(_gc.CONFIG), "CONFIG_HASH_MISMATCH"),
                ("wrapper_sha256", _sha(f"{_gc.R}/pairspace/gb_gata1_preflight.py"),
                 "WRAPPER_HASH_MISMATCH"),
                ("pairspace_builder_sha256", _pf._pairspace_verified()[1], "BUILDER_HASH_MISMATCH"),
                ("fixture_sha256", _sha(fix), "FIXTURE_HASH_MISMATCH")):
            if _field not in f.files:
                _no("MISSING_PROVENANCE_FIELD", f"artifact is missing {_field}")
            if str(f[_field]) != str(_actual):
                _no(_code, f"{_field} is {str(f[_field])[:16]} but the file on disk hashes to "
                           f"{str(_actual)[:16]}; the artifact does not describe this code")
        for _r in _pc["roots"]:
            for _need in (f"{_r}_I", f"{_r}_J", f"{_r}_pt", f"{_r}_dt", f"{_r}_cells",
                          f"{_r}_component", f"{_r}_conn_data", f"{_r}_conn_indices",
                          f"{_r}_conn_indptr", f"{_r}_conn_shape"):
                if _need not in f.files:
                    _no("MISSING_PAIR_FIELD", f"artifact is missing {_need}")
            if not np.array_equal(np.asarray(f[f"{_r}_cells"]).astype(str), _ids):
                _no("CELLS_MISMATCH", f"{_r}: artifact cells do not match the fixture rows; the "
                                      f"pairs index a different substrate")
            _pt = np.asarray(f[f"{_r}_pt"], dtype=np.float64)
            _dt = np.asarray(f[f"{_r}_dt"], dtype=np.float64)
            _I = np.asarray(f[f"{_r}_I"], dtype=np.int64)
            _J = np.asarray(f[f"{_r}_J"], dtype=np.int64)
            if _pt.size != _n_cells or not np.isfinite(_pt).all():
                _no("ORDERING_VARIABLE_INVALID",
                    f"{_r}: ordering variable has size {_pt.size} for {_n_cells} cells or is "
                    f"non-finite")
            # bounds and raggedness BEFORE any indexing, so a bad index is a refusal not an IndexError
            if len(_I) == 0 or len(_I) != len(_J) or len(_dt) != len(_I):
                _no("RAGGED_PAIRS", f"{_r}: {len(_I)} I, {len(_J)} J, {len(_dt)} dt")
            if _I.min() < 0 or _J.min() < 0 or _I.max() >= _n_cells or _J.max() >= _n_cells:
                _no("PAIR_INDEX_OUT_OF_RANGE", f"{_r}: a pair index is outside [0,{_n_cells})")
            if len(np.unique(np.stack([_I, _J], 1), axis=0)) != len(_I):
                _no("DUPLICATE_PAIRS", f"{_r}: the builder emits each eligible pair once, but the "
                                       f"artifact repeats one")
            if not np.allclose(_dt, _pt[_J] - _pt[_I], rtol=0, atol=1e-12):
                _no("DT_INCONSISTENT", f"{_r}: stored dt disagrees with the stored ordering variable")
            if not np.all(_dt > 0):
                _no("PAIRS_NOT_FORWARD", f"{_r}: {int((_dt <= 0).sum())} pairs are not forward")
            # EXACT predicate: the builder emits dt > min (strict) and dt <= max (inclusive).
            if _dt.min() <= _pc["min_forward_dt"] or _dt.max() > _pc["max_forward_dt"]:
                _no("DT_OUTSIDE_WINDOW",
                    f"{_r}: realised dt [{_dt.min():.9f},{_dt.max():.9f}] escapes the frozen "
                    f"window (min,max] = ({_pc['min_forward_dt']},{_pc['max_forward_dt']}]")
            if np.any(np.lexsort((_J, _I)) != np.arange(_I.size)):
                _no("PAIRS_NOT_CANONICAL",
                    f"{_r}: stored pairs are not in canonical (I, J) order, so the pair digest "
                    f"would depend on row order rather than on the pair SET")
            _worst = int(np.bincount(_I, minlength=_n_cells).max())
            if _worst > int(_pc["max_pairs_per_cell"]):
                _no("CAP_EXCEEDED", f"{_r}: a source cell has {_worst} pairs, cap is "
                                    f"{_pc['max_pairs_per_cell']}")
            # LAST line of defence, AFTER every structural check, so a structural violation is
            # reported as itself and the digest catches only a set that is valid but DIFFERENT.
            if f"{_r}_pairs_sha256" not in f.files:
                _no("MISSING_PAIR_FIELD", f"artifact is missing {_r}_pairs_sha256")
            _pH = _hl.sha256()
            for _a in (_I.astype(np.int64), _J.astype(np.int64)):
                _pH.update(np.ascontiguousarray(_a).tobytes())
            if _pH.hexdigest() != str(f[f"{_r}_pairs_sha256"]):
                _no("PAIRS_DIGEST_MISMATCH",
                    f"{_r}: the pair set does not match its recorded digest; substituting one "
                    f"eligible pair for another would otherwise leave every invariant intact")
            # CSR is RECONSTRUCTED, not merely shape-checked, and the stored component labels must
            # BE the connected components of that reconstruction -- a same-shape corruption or a
            # same-length wrong labelling would otherwise pass (internal review 2026-09-20 finding 7).
            _shape = tuple(np.asarray(f[f"{_r}_conn_shape"], dtype=np.int64).tolist())
            if _shape != (_n_cells, _n_cells):
                _no("CSR_SHAPE_WRONG", f"{_r}: connectivity shape {_shape} != ({_n_cells},{_n_cells})")
            _cd = np.asarray(f[f"{_r}_conn_data"])
            _ci = np.asarray(f[f"{_r}_conn_indices"])
            _cp = np.asarray(f[f"{_r}_conn_indptr"])
            if _cp.size != _n_cells + 1 or _cd.size != _ci.size or int(_cp[-1]) != _cd.size \
                    or _cp[0] != 0 or np.any(np.diff(_cp) < 0):
                _no("CSR_MALFORMED", f"{_r}: CSR arrays do not form a valid matrix "
                                     f"(indptr {_cp.size}, data {_cd.size}, indices {_ci.size})")
            if _ci.size and (_ci.min() < 0 or _ci.max() >= _n_cells):
                _no("CSR_INDEX_OUT_OF_RANGE", f"{_r}: a CSR column index is outside [0,{_n_cells})")
            if not np.isfinite(_cd).all():
                _no("CSR_NONFINITE", f"{_r}: connectivity contains non-finite weights")
            try:
                _C = _sp.csr_matrix((_cd, _ci, _cp), shape=_shape)
                _C.check_format(full_check=True)
            except Exception as _e:
                _no("CSR_MALFORMED", f"{_r}: CSR does not reconstruct ({type(_e).__name__}: {_e})")
            # The STORED encoding must itself be canonical, or an equivalent-but-different encoding
            # of the same graph would fail the digest and a real edit could hide behind that noise.
            _Cn = _C.copy(); _Cn.sum_duplicates(); _Cn.eliminate_zeros(); _Cn.sort_indices()
            if _Cn.data.size != _cd.size or not np.array_equal(_Cn.indices, _ci) \
                    or not np.array_equal(_Cn.indptr, _cp):
                _no("CSR_NOT_CANONICAL",
                    f"{_r}: stored CSR is not in canonical form "
                    f"(sum_duplicates + eliminate_zeros + sort_indices)")
            if f"{_r}_conn_sha256" not in f.files:
                _no("MISSING_PAIR_FIELD", f"artifact is missing {_r}_conn_sha256")
            _ch = _hl.sha256()
            for _a in (_cd.astype(np.float32), _ci.astype(np.int32), _cp.astype(np.int64),
                       np.asarray(_shape, dtype=np.int64)):
                _ch.update(np.ascontiguousarray(_a).tobytes())
            if _ch.hexdigest() != str(f[f"{_r}_conn_sha256"]):
                _no("CONN_DIGEST_MISMATCH",
                    f"{_r}: the connectivity does not match its recorded digest; an edit that "
                    f"preserves the component partition would otherwise pass")
            _lab = np.asarray(f[f"{_r}_component"])
            if _lab.ndim != 1 or _lab.size != _n_cells:
                _no("COMPONENT_SHAPE_WRONG",
                    f"{_r}: component labels have shape {_lab.shape}, expected {_n_cells}")
            _n_true, _lab_true = _cc(_C, directed=False)
            # Compare PARTITIONS, not label values: connected_components may number groups
            # differently. Canonicalise both sides with np.unique(return_inverse=True) rather than
            # casting to int64 -- internal review 2026-09-20 gave a counterexample where distinct FLOAT labels
            # (0.0 and 0.4) both cast to 0 and a genuinely wrong partition passed the count test.
            _sc = np.unique(np.asarray(_lab).ravel(), return_inverse=True)[1]
            _st = np.unique(np.asarray(_lab_true).ravel(), return_inverse=True)[1]
            if len(np.unique(_sc)) != _n_true or \
                    len(np.unique(np.stack([_sc, _st], 1), axis=0)) != _n_true:
                _no("COMPONENT_CONTENT_WRONG",
                    f"{_r}: stored component labels are not the connected components of the stored "
                    f"connectivity ({len(np.unique(_lab))} stored groups vs {_n_true} actual)")
        _fg = np.asarray(A.var_names).astype(str); _ag = np.asarray(f["genes"]).astype(str)
        if len(set(_ag)) != len(_ag):
            _no("ARTIFACT_GENES_NOT_UNIQUE", "the pair artifact repeats a gene name")
        if not np.array_equal(_fg, _ag):
            # Reordering the fixture to the artifact silently repairs a mismatch that should be
            # fatal: if the two disagree, one of them is stale.
            _no("GENE_ORDER_MISMATCH",
                f"fixture gene order differs from the pair artifact ({len(_fg)} vs {len(_ag)} names)")
        if layer == "X":
            layer = "Ms"          # the canonical SOURCE MATRIX; .X is the state space, not the source
    A = A[:, [str(g) for g in f["genes"]]].copy()
    genes = np.asarray(A.var_names).astype(str)
    catalogue = {l.strip().upper() for l in open(cat) if l.strip()}
    if layer == "X":
        Xm = d_(A.X).astype(np.float64)
    elif layer == "Ms":
        Xm = d_(A.layers["Ms"]).astype(np.float64)
    elif layer == "Ms_log":
        Xm = np.log1p(d_(A.layers["Ms"]).astype(np.float64))
    elif layer == "Ms_std":
        # Ms on native scale; the caller standardizes the PAIR matrix per gene with pair_scale().
        # Kept as a distinct layer name so provenance records which METRIC estimated the subspace.
        Xm = d_(A.layers["Ms"]).astype(np.float64)
    else:
        raise ValueError(f"unknown layer {layer!r}; use X, Ms, Ms_log or Ms_std")
    out = dict(genes=genes, layer=layer, X=Xm,
               V=np.nan_to_num(d_(A.layers["velocity"]).astype(np.float64), nan=0.0),
               obs=A.obs.copy(), pairs={}, is_tf=np.isin(np.char.upper(genes), list(catalogue)),
               fixture=fix, pairs_file=prs)
    for root in ROOTS:
        I, J = f[f"{root}_I"].astype(np.int64), f[f"{root}_J"].astype(np.int64)
        if max_pairs and len(I) > max_pairs:
            sel = np.random.default_rng(seed).choice(len(I), max_pairs, replace=False)
            I, J = I[sel], J[sel]
        out["pairs"][root] = (I, J)
    return out


def cell_folds(n_cells, folds, seed=0):
    cf = np.empty(n_cells, int)
    cf[np.random.default_rng(seed).permutation(n_cells)] = np.arange(n_cells) % folds
    return cf


def gene_folds(n_genes, folds, seed=1):
    gf = np.empty(n_genes, int)
    gf[np.random.default_rng(seed).permutation(n_genes)] = np.arange(n_genes) % folds
    return gf


def overlap(Ua, Ub, k):
    """Normalised projector overlap (1/k)||Ua_k^T Ub_k||_F^2 in [0,1]; 1 iff the spans coincide."""
    M = Ua[:, :k].T @ Ub[:, :k]
    return float((M ** 2).sum() / k)


def pair_scale(M, floor_quantile=0.05):
    """Robust per-gene scale of a PAIR-DIFFERENCE matrix, truth-free: 1.4826*MAD about zero (pair
    differences are centred by construction). Genes whose MAD is 0 (never change) get the floor, so
    they cannot blow up. Used ONLY to define the metric in which the shared subspace is estimated."""
    mad = 1.4826 * np.median(np.abs(M - np.median(M, 0, keepdims=True)), 0)
    pos = mad[mad > 0]
    floor = np.quantile(pos, floor_quantile) if pos.size else 1.0
    return np.maximum(mad, floor)
