#!/usr/bin/env python
"""Uniform real-data comparator tables on the frozen endpoint contracts"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd


R = Path(__import__("os").environ["GEOBRIDGE_ROOT"])
SCRATCH = R / "scratch"
PAIRSPACE = R / "pairspace"
for _p in (str(SCRATCH), str(PAIRSPACE), str(R)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = R / "results" / "real_uniform"
FIG1_ELIGIBILITY = R / "results" / "fig_contracts" / "fig1_eligibility.axis_v1.json"
FIG1_ELIGIBILITY_SHA256 = "5e0ac89c6fe5ca92c08b4f910f83329ab016520e779f67776a41438dd8559a86"
FIGURE_CONTRACT = R / "design_notes_not_shipped" / "REAL_FIGURE_CONTRACTS_v1_2026_09_24.md"
FIGURE_CONTRACT_SHA256 = "88ca9bf1988b5a5e779ab00542b3678c29905db59e2eb30e6d7b12867a0f0db9"
SYSTEMS = ("GATA1", "PANC", "BONEMARROW", "DENTATEGYRUS", "ZEBRAFISH", "K562", "GASTRULATION")
ROOTS = ("dc1_min", "dc1_max")
GB_KS = tuple(range(31))
GB_PARITY_TOL = 1e-5
PRE_UNIVERSE_REPAIR_SHA256 = "3e47e995921c627eb5145779e151f4db83e47fa3e451181da8ace6b5cae29033"
GBCACHE_EXPECTED_MISSING = {
    "GATA1": 66,
    "PANC": 40,
    "DENTATEGYRUS": 23,
    "GASTRULATION": 43,
}
KS_COND = tuple(range(1, 6))
KS_INFO = tuple(range(6))
NO_CALL_SYSTEMS = {"GASTRULATION": "PREDECLARED_NO_CALL_VELOCITY"}
VELORAMA_LAMS = np.logspace(-2, 1, num=19).tolist()
VELORAMA_CONFIG = {
    "x_norm": "zscore_clip5",
    "dynamics": "rna_velocity",
    "proba": 1,
    "n_neighbors": 30,
    "velo_mode": "stochastic",
    "use_time": 0,
    "n_comps": 50,
    "lag": 5,
    "hidden": [32],
    "penalty": "H",
    "lr": 0.01,
    "lam_ridge": 0.0,
    "max_iter": 1000,
    "lookback": 5,
    "check_every": 10,
    "seed": 0,
    "reg_target": 1,
    "lams": [round(x, 10) for x in VELORAMA_LAMS],
}
REGVELO_CONFIG = {
    "seed": 0,
    "max_epochs": 500,
    "batch_size": 1024,
    "skeleton": "zeros",
    "lam": 0.0,
    "soft_constraint": True,
}
METHOD_ORDER = (
    "GeoBRIDGE-RAW",
    "GeoBRIDGE-COND",
    "PairStatic-RAW",
    "PairStatic-COND",
    "Pearson",
    "GENIE3",
    "GRNBoost2",
    "Velorama",
    "RegVelo",
    "InfoLinear-COND",
    "InfoSpline-COND",
)


def sha(path: os.PathLike | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json(x) -> str:
    return json.dumps(x, sort_keys=True, separators=(",", ":"), allow_nan=False)


def digest_object(x) -> str:
    return hashlib.sha256(canonical_json(x).encode()).hexdigest()


def script_sha() -> str:
    return sha(Path(__file__).resolve())


def atomic_json(path: os.PathLike | str, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as fh:
            json.dump(obj, fh, indent=1, sort_keys=True, default=json_default)
            fh.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def atomic_npz(path: os.PathLike | str, **arrays) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    try:
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def json_default(x):
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, Path):
        return str(x)
    raise TypeError(type(x).__name__)


def dense(M, dtype=np.float32):
    return np.asarray(M.toarray() if hasattr(M, "toarray") else M, dtype=dtype)


def unique_exact(names, what: str) -> list[str]:
    out = [str(x) for x in names]
    if len(set(out)) != len(out):
        raise SystemExit(f"REFUSING_DUPLICATE_{what.upper()}: exact names are not unique")
    return out


def upper_index(names: list[str], what: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for i, name in enumerate(names):
        key = name.upper()
        if key in out:
            raise SystemExit(f"REFUSING_CASE_COLLISION_{what.upper()}: {name!r}")
        out[key] = i
    return out


def system_dir(system: str) -> Path:
    return OUT / system.upper()


def production_cache_paths(system: str) -> list[Path]:
    system = system.upper()
    return [
        R / "results" / "cop_alignment" / "scores" / f"{system}.npz",
        R / "results" / "regulon_panel" / system / "rankings.npz",
    ]


def completion_cache_path(system: str) -> Path:
    return OUT / "gb_cache" / f"{system.upper()}.npz"


# Frozen first-hit order for every GeoBRIDGE cache consumer
def cache_paths(system: str) -> list[Path]:
    return production_cache_paths(system) + [completion_cache_path(system)]


# Resolve the two frozen cache schemas without inferring from a pathname
def gb_score_key(z, root: str, channel: str, path: Path) -> str:
    for key in (f"{root}|{channel}", f"{root}|{channel}|absS"):
        if key in z.files:
            return key
    raise SystemExit(f"REFUSING_GB_CACHE_SCHEMA: {path}: no {root}/{channel} absS array")


# Fail closed on an unbound, stale, or truth-facing completion cache
def validate_completion_cache(path: Path, system: str) -> None:
    if os.path.realpath(path) != os.path.realpath(completion_cache_path(system)):
        return
    receipt = path.with_suffix(".receipt.json")
    if not receipt.exists():
        raise SystemExit(f"REFUSING_GBCACHE_NO_RECEIPT: {path}")
    rec = json.load(open(receipt))
    binding = rec.get("binding", {})
    if rec.get("output") != str(path) or rec.get("output_sha256") != sha(path):
        raise SystemExit(f"REFUSING_GBCACHE_OUTPUT_BINDING: {path}")
    if rec.get("truth_read") is not False or binding.get("truth_read") is not False:
        raise SystemExit(f"REFUSING_GBCACHE_TRUTH_READ: {path}")
    claimed = binding.get("binding_sha256")
    payload = dict(binding)
    payload.pop("binding_sha256", None)
    if not claimed or claimed != digest_object(payload):
        raise SystemExit(f"REFUSING_GBCACHE_BINDING_DIGEST: {path}")
    if binding.get("command") != "gbcache" or binding.get("system") != system.upper():
        raise SystemExit(f"REFUSING_GBCACHE_IDENTITY: {path}")
    # The universe repair changes only evaluation code below the cached-score producer
    if binding.get("producer_sha256") not in {script_sha(), PRE_UNIVERSE_REPAIR_SHA256}:
        raise SystemExit(f"REFUSING_GBCACHE_STALE_PRODUCER: {path}")
    if binding.get("project_scores_sha256") != sha(SCRATCH / "gb_target_extension.py"):
        raise SystemExit(f"REFUSING_GBCACHE_STALE_PROJECTOR: {path}")
    for pkey, hkey in (
            ("fixture", "fixture_sha256"),
            ("pairs", "pairs_sha256"),
            ("fig1_eligibility", "fig1_eligibility_sha256"),
            ("figure_contract", "figure_contract_sha256")):
        p = Path(binding.get(pkey, ""))
        if not p.is_file() or binding.get(hkey) != sha(p):
            raise SystemExit(f"REFUSING_GBCACHE_STALE_{pkey.upper()}: {path}")
    for item in binding.get("production_caches", []):
        p = Path(item.get("path", ""))
        if not p.is_file() or item.get("sha256") != sha(p):
            raise SystemExit(f"REFUSING_GBCACHE_STALE_PRODUCTION_CACHE: {path}")


# Load GB rows in frozen first-hit order and require one identical axis
def gb_raw_rows(system: str, srcs=None, paths: list[Path] | None = None):
    paths = cache_paths(system) if paths is None else [Path(p) for p in paths]
    present = [p for p in paths if p.exists()]
    if srcs is None:
        requested = []
        for path in present:
            validate_completion_cache(path, system)
            with np.load(path, allow_pickle=True) as z:
                for source in unique_exact(np.asarray(z["sources"]).astype(str), f"{path.name}_sources"):
                    key = source.upper()
                    if key not in requested:
                        requested.append(key)
    else:
        requested = [str(s).upper() for s in srcs]
        if len(set(requested)) != len(requested):
            raise SystemExit(f"REFUSING_DUPLICATE_GB_REQUEST: {system}")
    out: dict[str, dict[tuple[str, str], np.ndarray]] = {}
    axis: list[str] | None = None
    for path in present:
        validate_completion_cache(path, system)
        with np.load(path, allow_pickle=True) as z:
            genes = unique_exact(np.asarray(z["genes"]).astype(str), f"{path.name}_genes")
            sources = unique_exact(np.asarray(z["sources"]).astype(str), f"{path.name}_sources")
            source_index = upper_index(sources, f"{path.name}_sources")
            if axis is None:
                axis = genes
            elif genes != axis:
                raise SystemExit(f"REFUSING_GB_CACHE_AXIS_MISMATCH: {system}: {path}")
            hits = [(source, source_index[source]) for source in requested
                    if source not in out and source in source_index]
            for root in ROOTS:
                for channel in ("U", "X"):
                    a = np.asarray(z[gb_score_key(z, root, channel, path)])
                    if a.shape != (len(sources), len(genes), len(GB_KS)):
                        raise SystemExit(
                            f"REFUSING_GB_CACHE_SHAPE: {path}: {root}/{channel} {a.shape}"
                        )
                    for source, j in hits:
                        out.setdefault(source, {})[(root, channel)] = np.asarray(a[j], float)
    return out, None if axis is None else np.asarray(axis)


# Return an axis and first-seen source union without materialising scores
def cache_inventory(system: str, paths: list[Path]) -> tuple[list[str], list[str]]:
    axes: list[list[str]] = []
    source_keys: list[str] = []
    for path in paths:
        validate_completion_cache(path, system)
        with np.load(path, allow_pickle=True) as z:
            axes.append(unique_exact(np.asarray(z["genes"]).astype(str), f"{system}_axis"))
            sources = unique_exact(np.asarray(z["sources"]).astype(str), f"{system}_sources")
            upper_index(sources, f"{system}_sources")
            for s in sources:
                key = str(s).upper()
                if key not in source_keys:
                    source_keys.append(key)
    if not axes:
        raise SystemExit(f"REFUSING_NO_GB_CACHE: no cached GeoBRIDGE scores for {system}")
    if any(axis != axes[0] for axis in axes[1:]):
        raise SystemExit(f"REFUSING_GB_CACHE_AXIS_MISMATCH: {system}")
    ui = upper_index(axes[0], f"{system}_axis")
    missing = [s for s in source_keys if s not in ui]
    if missing:
        raise SystemExit(f"REFUSING_SOURCE_OFF_AXIS: {system}: {missing[:10]}")
    return axes[0], [axes[0][ui[s]] for s in source_keys]


# Truth-free union of the source rosters already scored by GeoBRIDGE
def cached_roster(system: str) -> tuple[list[str], list[str], list[Path]]:
    paths = [p for p in cache_paths(system) if p.exists()]
    if not paths:
        raise SystemExit(f"REFUSING_NO_GB_CACHE: no cached GeoBRIDGE scores for {system}")
    axis, sources = cache_inventory(system, paths)
    return axis, sources, paths


# Coverage-only Fig-1 roster from the hash-pinned pre-score object
def frozen_fig1_roster(system: str) -> list[str]:
    if __import__("os").environ.get("GB_SKIP_PREFLIGHT") == "1":      # opt-in (public release): skip only the sha pins of the eligibility object / figure contract
        print("preflight skipped (public release)")
    elif sha(FIG1_ELIGIBILITY) != FIG1_ELIGIBILITY_SHA256:
        raise SystemExit("BLOCKED_BY_SPEC_FIG1_ELIGIBILITY_HASH_MISMATCH")
    elif sha(FIGURE_CONTRACT) != FIGURE_CONTRACT_SHA256:
        raise SystemExit("BLOCKED_BY_SPEC_REAL_FIGURE_CONTRACT_HASH_MISMATCH")
    data = json.load(open(FIG1_ELIGIBILITY))
    return [str(x).upper() for x in data.get("systems", {}).get(system.upper(), {}).get("eligible", [])]


# Do not spend comparator compute on a roster lacking frozen GB rows
def assert_frozen_roster_covered(system: str, sources: list[str]) -> None:
    expected = frozen_fig1_roster(system)
    if not expected:  # K562 has no Fig-1 endpoint.
        return
    have = {s.upper() for s in sources}
    missing = [s for s in expected if s not in have]
    if missing:
        raise SystemExit(
            f"BLOCKED_BY_SPEC_F1_GB_CACHE_INCOMPLETE: {system} frozen={len(expected)} "
            f"cached={len(expected)-len(missing)} missing={len(missing)} "
            f"examples={missing[:10]}; run the independently reviewed gbcache command first"
        )


# Require identical support and max |delta absS| <= tol over all 124 cells
def assert_absS_parity(
        scored: dict[tuple[str, str], np.ndarray],
        scored_sources: list[str],
        reference: dict[str, dict[tuple[str, str], np.ndarray]],
        parity_sources: list[str],
        tol: float = GB_PARITY_TOL) -> tuple[float, dict[str, float]]:
    si = upper_index(scored_sources, "parity_scored_sources")
    worst_by_source: dict[str, float] = {}
    for source in parity_sources:
        key = source.upper()
        if key not in si or key not in reference:
            raise SystemExit(f"REFUSING_GBCACHE_PARITY_SOURCE_MISSING: {source}")
        j, worst = si[key], 0.0
        for root in ROOTS:
            for channel in ("U", "X"):
                a = np.asarray(scored[(root, channel)][j], float)
                b = np.asarray(reference[key][(root, channel)], float)
                if a.shape != b.shape:
                    raise SystemExit(
                        f"REFUSING_GBCACHE_PARITY_SHAPE: {source}/{root}/{channel}: {a.shape} != {b.shape}"
                    )
                fa, fb = np.isfinite(a), np.isfinite(b)
                if not np.array_equal(fa, fb):
                    raise SystemExit(
                        f"REFUSING_GBCACHE_PARITY_FINITE: {source}/{root}/{channel}: "
                        f"{int(np.count_nonzero(fa != fb))} cells differ"
                    )
                if fa.any():
                    worst = max(worst, float(np.max(np.abs(a[fa] - b[fa]))))
        worst_by_source[key] = worst
        if worst > tol:
            raise SystemExit(
                f"REFUSING_GBCACHE_PARITY_VALUE: {source}: max_abs_delta={worst:.9g} > {tol:.9g}"
            )
    return max(worst_by_source.values(), default=0.0), worst_by_source


# Frozen production algebra, byte-for-byte structured like CA.score
def score_gb_sources(system: str, axis: list[str], sources: list[str]):
    import anndata as ad
    import gb_substrate as GS
    from gb_target_extension import project_scores, CHUNK, NPC

    t0 = time.time()
    S = GS.load(system, canonical=True)
    fixture, pairs_file, _ = GS.paths(system, canonical=True)
    F = ad.read_h5ad(fixture)
    fixture_genes = unique_exact(np.asarray(F.var_names).astype(str), f"{system}_fixture_genes")
    substrate_genes = unique_exact(np.asarray(S["genes"]).astype(str), f"{system}_substrate_genes")
    if substrate_genes != fixture_genes:
        raise SystemExit(f"REFUSING_GBCACHE_SUBSTRATE_FIXTURE_AXIS: {system}")
    fi = {g: i for i, g in enumerate(fixture_genes)}
    missing_targets = [g for g in axis if g not in fi]
    if missing_targets:
        raise SystemExit(f"REFUSING_GBCACHE_EXACT_AXIS_JOIN: {system}: {missing_targets[:10]}")
    fc = np.asarray([fi[g] for g in axis], int)
    Mu = dense(F.layers["Mu"][:, fc], np.float32)
    Ms = dense(F.layers["Ms"][:, fc], np.float32)
    axis_upper = upper_index(axis, f"{system}_gbcache_axis")
    source_keys = [s.upper() for s in sources]
    missing_sources = [s for s in source_keys if s not in axis_upper]
    if missing_sources:
        raise SystemExit(f"REFUSING_GBCACHE_SOURCE_OFF_AXIS: {system}: {missing_sources[:10]}")
    si = np.asarray([axis_upper[s] for s in source_keys], int)
    Xsrc = np.asarray(S["X"], np.float64)[:, fc[si]]
    if not np.allclose(Xsrc, Ms[:, si], rtol=1e-5, atol=1e-6):
        raise SystemExit(f"REFUSING_GBCACHE_SOURCE_MS_PARITY: {system}")
    P30 = np.asarray(F.obsm["X_pca"], float)[:, :NPC]
    if P30.shape[1] != NPC:
        raise SystemExit(f"REFUSING_GBCACHE_P30: {system}: {P30.shape}")
    G = len(axis)
    raw = {
        (root, channel): np.full((len(sources), G, len(GB_KS)), np.nan, np.float32)
        for root in ROOTS for channel in ("U", "X")
    }
    pair_counts = {}
    for root in ROOTS:
        I, J = S["pairs"][root]
        I, J = np.asarray(I, int), np.asarray(J, int)
        pair_counts[root] = len(I)
        Qf, _ = np.linalg.qr(P30[J] - P30[I])
        dx = (Xsrc[J] - Xsrc[I]).astype(np.float32)
        for g0 in range(0, G, CHUNK):
            sl = slice(g0, min(g0 + CHUNK, G))
            for channel, M in (("U", Mu), ("X", Ms)):
                dy = (M[J, sl] - M[I, sl]).astype(np.float32)
                for k, _num, _den, _z2, absS in project_scores(dx, dy, Qf):
                    raw[(root, channel)][:, sl, k] = absS
        print(
            f"[gbcache] {system} {root}: {len(I)} pairs, {len(sources)} sources "
            f"{time.time()-t0:.0f}s",
            flush=True,
        )
    for j, source in enumerate(source_keys):
        own = axis_upper[source]
        for a in raw.values():
            a[j, own, :] = np.nan
    return raw, {
        "fixture": Path(fixture),
        "pairs_file": Path(pairs_file),
        "pair_counts": pair_counts,
        "wall_s": time.time() - t0,
    }


# Score missing roster members without modifying existing caches.
def run_gbcache(args) -> None:
    system = args.system.upper()
    expected_n = GBCACHE_EXPECTED_MISSING[system]
    roster = frozen_fig1_roster(system)
    base_paths = [p for p in production_cache_paths(system) if p.exists()]
    if not base_paths:
        raise SystemExit(f"REFUSING_GBCACHE_NO_PRODUCTION_CACHE: {system}")
    axis, cached_sources = cache_inventory(system, base_paths)
    axis = unique_exact(axis, f"{system}_gbcache_axis")
    axis_upper = upper_index(axis, f"{system}_gbcache_axis")
    covered = {s.upper() for s in cached_sources}
    missing = [s for s in roster if s not in covered]
    if len(missing) != expected_n:
        raise SystemExit(
            f"REFUSING_GBCACHE_FROZEN_MISSING_COUNT: {system}: {len(missing)} != {expected_n}"
        )
    off_axis = [s for s in roster if s not in axis_upper]
    if off_axis:
        raise SystemExit(f"REFUSING_GBCACHE_ROSTER_OFF_AXIS: {system}: {off_axis[:10]}")
    parity_sources = [s for s in roster if s in covered][:3]
    if len(parity_sources) != 3:
        raise SystemExit(f"REFUSING_GBCACHE_TOO_FEW_PARITY_SOURCES: {system}")
    base_rows, parity_axis = gb_raw_rows(system, parity_sources, paths=base_paths)
    if parity_axis is None or list(parity_axis.astype(str)) != axis:
        raise SystemExit(f"REFUSING_GBCACHE_PARITY_AXIS: {system}")
    scored_sources = missing + parity_sources

    import gb_substrate as GS
    from gb_target_extension import project_scores  # noqa: F401; hash-bound dependency

    fixture, pairs_file, _ = GS.paths(system, canonical=True)
    binding = {
        "command": "gbcache",
        "system": system,
        "algebra": "gb_cop_alignment.score / gb_target_extension.project_scores",
        "channels": {"U": "dMu", "X": "dMs"},
        "source": "V1 Ms",
        "roots": list(ROOTS),
        "ks": list(GB_KS),
        "projector": "Q=qr(P30[J]-P30[I]); nested prefixes",
        "self_edge": "NaN",
        "fixture": str(fixture),
        "fixture_sha256": sha(fixture),
        "pairs": str(pairs_file),
        "pairs_sha256": sha(pairs_file),
        "production_caches": [{"path": str(p), "sha256": sha(p)} for p in base_paths],
        "fig1_eligibility": str(FIG1_ELIGIBILITY),
        "fig1_eligibility_sha256": sha(FIG1_ELIGIBILITY),
        "figure_contract": str(FIGURE_CONTRACT),
        "figure_contract_sha256": sha(FIGURE_CONTRACT),
        "axis_sha256": hashlib.sha256("\n".join(axis).encode()).hexdigest(),
        "frozen_roster_sha256": hashlib.sha256("\n".join(roster).encode()).hexdigest(),
        "missing_sources": missing,
        "parity_sources": parity_sources,
        "parity_tolerance": GB_PARITY_TOL,
        "project_scores_sha256": sha(SCRATCH / "gb_target_extension.py"),
        "producer_sha256": script_sha(),
        "truth_read": False,
    }
    binding["binding_sha256"] = digest_object(binding)
    out = completion_cache_path(system)
    receipt = out.with_suffix(".receipt.json")
    if out.exists() or receipt.exists():
        if valid_part(out, binding):
            print(f"[gbcache] exact cache already exists and validates: {out}", flush=True)
            return
        raise SystemExit(f"REFUSING_GBCACHE_EXISTING_INVALID: {out}")

    raw, meta = score_gb_sources(system, axis, scored_sources)
    parity_worst, parity_by_source = assert_absS_parity(
        raw, scored_sources, base_rows, parity_sources, GB_PARITY_TOL
    )
    source_index = upper_index(scored_sources, "gbcache_scored_sources")
    keep = np.asarray([source_index[s] for s in missing], int)
    arrays = {
        "genes": np.asarray(axis),
        "sources": np.asarray(missing),
        **{f"{root}|{channel}": raw[(root, channel)][keep]
           for root in ROOTS for channel in ("U", "X")},
    }
    for j, source in enumerate(missing):
        own = axis_upper[source]
        for root in ROOTS:
            for channel in ("U", "X"):
                if not np.isnan(arrays[f"{root}|{channel}"][j, own]).all():
                    raise SystemExit(f"REFUSING_GBCACHE_SELF_EDGE_NOT_NAN: {system}/{source}")
    rec = write_part(out, binding, arrays, {
        "n_missing_sources": len(missing),
        "n_parity_sources": len(parity_sources),
        "parity_max_abs_delta": parity_worst,
        "parity_max_abs_delta_by_source": parity_by_source,
        "pair_counts": meta["pair_counts"],
        "wall_s": meta["wall_s"],
        "truth_read": False,
    })
    print(json.dumps(rec, indent=1), flush=True)


# Load and validate the canonical fixture/pairs; return aligned metadata
def canonical_inputs(system: str):
    import anndata as ad
    import gb_substrate as GS

    system = system.upper()
    S = GS.load(system, canonical=True)
    fixture, pairs_file, _ = GS.paths(system, canonical=True)
    A = ad.read_h5ad(fixture)
    axis, sources, score_paths = cached_roster(system)
    fgenes = unique_exact(np.asarray(A.var_names).astype(str), f"{system}_fixture_genes")
    fi = {g: i for i, g in enumerate(fgenes)}
    missing = [g for g in axis if g not in fi]
    if missing:
        raise SystemExit(f"REFUSING_EXACT_AXIS_JOIN: {system}: {missing[:10]}")
    if unique_exact(np.asarray(S["genes"]).astype(str), f"{system}_substrate_genes") != fgenes:
        raise SystemExit(f"REFUSING_SUBSTRATE_FIXTURE_AXIS_MISMATCH: {system}")
    return {
        "system": system,
        "S": S,
        "A": A,
        "fixture": Path(fixture),
        "pairs_file": Path(pairs_file),
        "axis": axis,
        "axis_cols": np.asarray([fi[g] for g in axis], int),
        "fixture_genes": fgenes,
        "sources": sources,
        "source_axis_idx": np.asarray([axis.index(s) for s in sources], int),
        "source_fixture_idx": np.asarray([fi[s] for s in sources], int),
        "score_paths": score_paths,
    }


def input_manifest(D) -> dict:
    return {
        "system": D["system"],
        "fixture": str(D["fixture"]),
        "fixture_sha256": sha(D["fixture"]),
        "pairs": str(D["pairs_file"]),
        "pairs_sha256": sha(D["pairs_file"]),
        "gb_score_caches": [{"path": str(p), "sha256": sha(p)} for p in D["score_paths"]],
        "axis_sha256": hashlib.sha256("\n".join(D["axis"]).encode()).hexdigest(),
        "sources_sha256": hashlib.sha256("\n".join(D["sources"]).encode()).hexdigest(),
        "n_cells": int(D["A"].n_obs),
        "n_fixture_genes": int(D["A"].n_vars),
        "n_axis_genes": len(D["axis"]),
        "n_sources": len(D["sources"]),
    }


def part_binding(D, method: str, config: dict) -> dict:
    obj = {
        "method": method,
        "inputs": input_manifest(D),
        "config": config,
        "producer_sha256": script_sha(),
        "truth_read": False,
    }
    obj["binding_sha256"] = digest_object(obj)
    return obj


def valid_part(path: Path, binding: dict) -> bool:
    receipt = path.with_suffix(".receipt.json")
    if not path.exists() or not receipt.exists():
        return False
    try:
        r = json.load(open(receipt))
        return r.get("binding") == binding and r.get("output_sha256") == sha(path)
    except Exception:
        return False


def write_part(path: Path, binding: dict, arrays: dict, extra: dict | None = None) -> dict:
    atomic_npz(path, **arrays)
    rec = {
        "binding": binding,
        "output": str(path),
        "output_sha256": sha(path),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "truth_read": False,
        **(extra or {}),
    }
    atomic_json(path.with_suffix(".receipt.json"), rec)
    return rec


def pearson_scores(D) -> np.ndarray:
    M = dense(D["A"].layers["Ms"][:, D["axis_cols"]], np.float64)
    X = M[:, D["source_axis_idx"]]
    X -= X.mean(0)
    nx = np.sqrt(np.sum(X * X, axis=0))
    out = np.full((len(D["sources"]), len(D["axis"])), np.nan, np.float32)
    block = 512
    for g0 in range(0, M.shape[1], block):
        Y = M[:, g0:g0 + block].copy()
        Y -= Y.mean(0)
        ny = np.sqrt(np.sum(Y * Y, axis=0))
        den = nx[:, None] * ny[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            C = np.abs(X.T @ Y / den)
        C[~np.isfinite(C)] = np.nan
        out[:, g0:g0 + block] = C.astype(np.float32)
    for i, j in enumerate(D["source_axis_idx"]):
        out[i, j] = np.nan
    return out


def arboreto_scores(D, learner: str, client) -> np.ndarray:
    import gb_sota_run as SR

    X = dense(D["A"].X[:, D["axis_cols"]], np.float64)
    links = SR.arboreto_fit(X, D["axis"], D["sources"], D["axis"], learner, 0, client)
    score = np.zeros((len(D["sources"]), len(D["axis"])), np.float32)
    si = {s: i for i, s in enumerate(D["sources"])}
    gi = {g: i for i, g in enumerate(D["axis"])}
    for tf, target, weight in links[["TF", "target", "importance"]].itertuples(index=False):
        if tf not in si or target not in gi:
            raise SystemExit(f"REFUSING_ARBORETO_UNKNOWN_NAME: {tf!r}, {target!r}")
        score[si[tf], gi[target]] = float(weight)
    for i, j in enumerate(D["source_axis_idx"]):
        score[i, j] = np.nan
    return score


# Fold-honest information-matched matrices, chunked only over targets
def info_scores(D, target_chunk: int = 128) -> tuple[dict[str, np.ndarray], dict]:
    import gb_infomatched as IM

    A, S = D["A"], D["S"]
    Ms = dense(A.layers["Ms"][:, D["axis_cols"]], np.float32)
    Mu = dense(A.layers["Mu"][:, D["axis_cols"]], np.float32)
    P30 = np.asarray(A.obsm["X_pca"], np.float64)[:, :30]
    nsrc, ng = len(D["sources"]), len(D["axis"])
    out = {
        f"{root}|{kind}|k{k}": np.full((nsrc, ng), np.nan, np.float32)
        for root in ROOTS for kind in ("linear", "spline") for k in KS_INFO
    }
    coverage = {}
    for root in ROOTS:
        I, J = S["pairs"][root]
        I, J = np.asarray(I, int), np.asarray(J, int)
        Qf = np.linalg.qr(P30[J] - P30[I])[0]
        dx = (Ms[J][:, D["source_axis_idx"]] - Ms[I][:, D["source_axis_idx"]]).astype(np.float32)
        masks, _ = IM.fold_masks(I, J, len(Ms))
        coverage[root] = {
            "mean_train_pair_fraction": float(np.mean([tr.mean() for tr, _ in masks])),
            "mean_test_pair_fraction": float(np.mean([te.mean() for _, te in masks])),
            "test_union_fraction": float(np.logical_or.reduce([te for _, te in masks]).mean()),
            "n_pairs": len(I),
        }
        for g0 in range(0, ng, target_chunk):
            sl = slice(g0, min(g0 + target_chunk, ng))
            dy = (Mu[J, sl] - Mu[I, sl]).astype(np.float32)
            for k, lin, spl in IM.scores(dx, dy, Qf, I, J, KS_INFO, len(Ms)):
                out[f"{root}|linear|k{k}"][:, sl] = lin.astype(np.float32)
                out[f"{root}|spline|k{k}"][:, sl] = spl.astype(np.float32)
            print(f"[info] {D['system']} {root} targets {sl.stop}/{ng}", flush=True)
    for key in out:
        for i, j in enumerate(D["source_axis_idx"]):
            out[key][i, j] = np.nan
    return out, coverage


def run_cpu(args) -> None:
    from distributed import Client, LocalCluster

    system = args.system.upper()
    if system not in SYSTEMS:
        raise SystemExit(f"unknown system {system}")
    D = canonical_inputs(system)
    assert_frozen_roster_covered(system, D["sources"])
    d = system_dir(system)
    d.mkdir(parents=True, exist_ok=True)
    common = {"genes": np.asarray(D["axis"]), "sources": np.asarray(D["sources"])}
    receipts = {}

    p = d / "pearson.npz"
    b = part_binding(D, "Pearson", {"matrix": "Ms", "statistic": "abs Pearson"})
    if not valid_part(p, b):
        t0 = time.time()
        receipts["Pearson"] = write_part(p, b, {**common, "score": pearson_scores(D)},
                                          {"wall_s": time.time() - t0})
    else:
        receipts["Pearson"] = json.load(open(p.with_suffix(".receipt.json")))
        print(f"[cpu] reuse {p}", flush=True)

    tree_work = []
    for learner, label in (("genie3", "GENIE3"), ("grnboost2", "GRNBoost2")):
        p = d / f"{learner}.npz"
        b = part_binding(D, label, {"matrix": "canonical fixture log .X", "seed": 0,
                                    "arboreto_wrapper": sha(SCRATCH / "gb_sota_run.py")})
        if valid_part(p, b):
            receipts[label] = json.load(open(p.with_suffix(".receipt.json")))
            print(f"[cpu] reuse {p}", flush=True)
        else:
            tree_work.append((learner, label, p, b))
    if tree_work:
        cluster = LocalCluster(n_workers=args.workers, threads_per_worker=1, memory_limit="0")
        client = Client(cluster)
        try:
            for learner, label, p, b in tree_work:
                t0 = time.time()
                score = arboreto_scores(D, learner, client)
                receipts[label] = write_part(p, b, {**common, "score": score},
                                              {"wall_s": time.time() - t0})
        finally:
            client.close()
            cluster.close()

    p = d / "infomatched.npz"
    b = part_binding(D, "information-matched", {
        "implementation_sha256": sha(SCRATCH / "gb_infomatched.py"),
        "folds": 5,
        "seed": 20260924,
        "ks": list(KS_INFO),
        "target_chunk": args.target_chunk,
        "models": ["linear", "fixed cubic spline"],
    })
    if not valid_part(p, b):
        t0 = time.time()
        scores, coverage = info_scores(D, args.target_chunk)
        receipts["information-matched"] = write_part(
            p, b, {**common, **scores}, {"fold_pair_coverage": coverage, "wall_s": time.time() - t0})
    else:
        receipts["information-matched"] = json.load(open(p.with_suffix(".receipt.json")))
        print(f"[cpu] reuse {p}", flush=True)

    summary = {
        "command": "cpu",
        "system": system,
        "inputs": input_manifest(D),
        "truth_read": False,
        "producer_sha256": script_sha(),
        "parts": {k: {"path": v["output"], "sha256": v["output_sha256"]} for k, v in receipts.items()},
    }
    atomic_json(d / "cpu.receipt.json", summary)
    print(json.dumps(summary, indent=1), flush=True)


def set_gpu(gpu: int) -> None:
    if gpu < 0:
        raise SystemExit("REFUSING_GPU_ID: --gpu must be nonnegative")
    if "torch" in sys.modules:
        raise SystemExit("REFUSING_GPU_ALREADY_INITIALISED: CUDA_VISIBLE_DEVICES must be set before importing torch")
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu)


def model_manifest(path: Path) -> list[dict]:
    return [{"path": str(p), "sha256": sha(p)} for p in sorted(path.rglob("*")) if p.is_file()]


def velorama_binding(D) -> dict:
    upstream = sorted((R / "external" / "velorama_official" / "velorama" / "velorama").glob("*.py"))
    return part_binding(D, "Velorama", {
        "config": VELORAMA_CONFIG,
        "upstream": {p.name: sha(p) for p in upstream},
        "harvester_sha256": sha(PAIRSPACE / "gb_velorama_harvest.py"),
        "target_roster": "full fixture; exact evaluator-axis projection after official aggregation",
        "source_roster": "cached GeoBRIDGE roster",
    })


def zscore_torch(M):
    import torch

    M = np.asarray(M, np.float64)
    sd = M.std(0)
    sd[sd == 0] = 1.0
    return torch.FloatTensor(np.clip((M - M.mean(0)) / sd, -5, 5))


def run_velorama(D, gpu: int) -> None:
    import scanpy as sc
    import torch
    from velorama.train import train_model
    from velorama.utils import calculate_diffusion_lags, construct_dag

    if not torch.cuda.is_available():
        raise SystemExit("REFUSING_NO_CUDA: Velorama must run on GPU")
    d = system_dir(D["system"])
    out = d / "velorama_cache"
    result = d / "velorama.npz"
    binding = velorama_binding(D)
    if valid_part(result, binding):
        print(f"[gpu] reuse {result}", flush=True)
        return
    key_path = out / "CACHE_KEY.json"
    reuse = False
    if key_path.exists():
        try:
            reuse = json.load(open(key_path)) == binding
        except Exception:
            reuse = False
    if out.exists() and not reuse:
        stale = out.with_name(out.name + time.strftime(".stale_%Y%m%dT%H%M%S") + f"_{time.time_ns()}")
        shutil.move(out, stale)
        print(f"[velorama] binding changed; moved old cache to {stale}", flush=True)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(key_path, binding)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    A = sc.read_h5ad(D["fixture"])
    fgenes = np.asarray(A.var_names).astype(str)
    fi = {g: i for i, g in enumerate(fgenes)}
    src_cols = np.asarray([fi[s] for s in D["sources"]], int)
    Xall = dense(A.X, np.float64)
    X = zscore_torch(Xall[:, src_cols])
    Y = zscore_torch(Xall)
    sc.pp.scale(A)
    dag = torch.FloatTensor(construct_dag(
        A,
        dynamics="rna_velocity",
        proba=1,
        n_neighbors=30,
        velo_mode="stochastic",
        use_time=0,
        n_comps=50,
    ))
    AX = calculate_diffusion_lags(dag, X, 5)
    AY = calculate_diffusion_lags(dag, Y, 5)
    name = f"{D['system']}_real_uniform"
    config_name = f"{name}.seed0.h32.H.lag5.rna_velocity"
    model_dir = out / config_name
    model_dir.mkdir(parents=True, exist_ok=True)
    reg_file, target_file = out / "reg_genes.txt", out / "target_genes.txt"
    reg_file.write_text("\n".join(D["sources"]) + "\n")
    target_file.write_text("\n".join(fgenes) + "\n")
    for lam in VELORAMA_LAMS:
        fn = model_dir / f"{name}.seed0.lam{np.round(lam, 4)}.h32.H.lag5.rna_velocity.pt"
        if fn.exists():
            continue
        t0 = time.time()
        train_model({
            "name": name,
            "AX": AX,
            "AY": AY,
            "Y": Y,
            "seed": 0,
            "lr": 0.01,
            "lam": lam,
            "lam_ridge": 0.0,
            "penalty": "H",
            "lag": 5,
            "hidden": [32],
            "max_iter": 1000,
            "device": "cuda",
            "lookback": 5,
            "check_every": 10,
            "verbose": False,
            "dynamics": "rna_velocity",
            "results_dir": str(out),
            "dir_name": config_name,
            "reg_target": 1,
        })
        print(f"[velorama] {D['system']} lambda={lam:.5g} {time.time()-t0:.0f}s", flush=True)
    expected = [model_dir / f"{name}.seed0.lam{np.round(l, 4)}.h32.H.lag5.rna_velocity.pt" for l in VELORAMA_LAMS]
    missing = [str(p) for p in expected if not p.exists()]
    if missing:
        raise SystemExit(f"REFUSING_INCOMPLETE_VELORAMA_GRID: {missing[:3]}")

    stage, harvest = out / "stage", out / "harvest"
    subprocess.run([
        sys.executable,
        str(PAIRSPACE / "gb_velorama_harvest.py"),
        "--velorama-root", str(R / "external" / "velorama_official"),
        "--search-root", str(out),
        "--name", name,
        "--dynamics", "rna_velocity",
        "--reg-genes", str(reg_file),
        "--target-genes", str(target_file),
        "--stage-dir", str(stage),
        "--out-dir", str(harvest),
        "--input-h5ad", str(D["fixture"]),
    ], check=True)
    tsv = harvest / f"{name}.rna_velocity.velorama.interactions.tsv"
    hj = harvest / f"{name}.rna_velocity.harvest.json"
    H = json.load(open(hj))
    if H.get("VERDICT") != "HARVESTED" or H.get("interactions_tsv_sha256") != sha(tsv):
        raise SystemExit("REFUSING_UNBOUND_VELORAMA_HARVEST")
    if H.get("input_h5ad_sha256") != sha(D["fixture"]) or H.get("n_lambda_found") != 19 or H.get("missing_lambdas"):
        raise SystemExit("REFUSING_VELORAMA_HARVEST_CONFIG")
    T = pd.read_csv(tsv, sep="\t", index_col=0)
    if list(T.index.astype(str)) != list(fgenes) or list(T.columns.astype(str)) != D["sources"]:
        raise SystemExit("REFUSING_VELORAMA_EXACT_NAME_OR_ORDER_MISMATCH")
    score = T.loc[D["axis"], D["sources"]].to_numpy(np.float32).T
    for i, j in enumerate(D["source_axis_idx"]):
        score[i, j] = np.nan
    write_part(result, binding, {
        "genes": np.asarray(D["axis"]),
        "sources": np.asarray(D["sources"]),
        "score": score,
    }, {
        "gpu_physical_id": gpu,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_name": torch.cuda.get_device_name(0),
        "harvest_receipt": str(hj),
        "harvest_receipt_sha256": sha(hj),
        "models": [{"path": str(p), "sha256": sha(p)} for p in expected],
        "score_semantics": "official aggregated Granger interaction strength; unsigned evaluation",
    })


def run_regvelo(D, gpu: int) -> None:
    import anndata as ad
    import regvelo as rgv
    import scvi
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("REFUSING_NO_CUDA: RegVelo must run on GPU")
    d = system_dir(D["system"])
    result = d / "regvelo.npz"
    binding = part_binding(D, "RegVelo", {
        "config": REGVELO_CONFIG,
        "regvelo_version": getattr(rgv, "__version__", "unknown"),
        "regulators": "cached GeoBRIDGE source roster",
        "extraction": "GRN_Jacobian(mean Ms)[:, source] outgoing columns",
    })
    if valid_part(result, binding):
        print(f"[gpu] reuse {result}", flush=True)
        return
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    scvi.settings.seed = 0
    A = ad.read_h5ad(D["fixture"])
    genes = np.asarray(A.var_names).astype(str)
    fi = {g: i for i, g in enumerate(genes)}
    src_cols = np.asarray([fi[s] for s in D["sources"]], int)
    for layer in ("Ms", "Mu"):
        if layer not in A.layers:
            raise SystemExit(f"REFUSING_REGVELO_MISSING_LAYER: {layer}")
    rgv.REGVELOVI.setup_anndata(A, spliced_layer="Ms", unspliced_layer="Mu")
    W = torch.zeros((len(genes), len(genes)), dtype=torch.float32)
    model = rgv.REGVELOVI(A, W=W, regulators=D["sources"], soft_constraint=True, lam=0.0)
    t0 = time.time()
    model.train(max_epochs=500, batch_size=1024)
    module = model.module
    device = str(next(module.parameters()).device)
    if not device.startswith("cuda"):
        raise SystemExit(f"REFUSING_REGVELO_NOT_ON_CUDA: {device}")
    fn = getattr(module.v_encoder, "GRN_Jacobian", None) or getattr(module.v_encoder, "GRN_Jacobian2", None)
    if fn is None:
        raise SystemExit("REFUSING_REGVELO_NO_JACOBIAN: no weight fallback is allowed")
    with torch.no_grad():
        mean_ms = torch.tensor(dense(A.layers["Ms"], np.float32), device=device).mean(0)
        J = fn(mean_ms).detach().cpu().numpy()
    if J.shape != (len(genes), len(genes)):
        raise SystemExit(f"REFUSING_REGVELO_JACOBIAN_SHAPE: {J.shape}")
    score_signed = J[:, src_cols].T[:, D["axis_cols"]].astype(np.float32)
    for i, j in enumerate(D["source_axis_idx"]):
        score_signed[i, j] = np.nan
    fin = np.isfinite(score_signed)
    nz = np.abs(score_signed[fin & (score_signed != 0)])
    rounded = np.round(score_signed[fin], 12)
    _, counts = np.unique(rounded, return_counts=True)
    diagnostics = {
        "n": int(score_signed.size),
        "nonfinite_frac": float((~fin).mean()),
        "exact_zero_frac": float((score_signed == 0).mean()),
        "near_zero_frac": float((np.abs(score_signed) < 1e-10).mean()),
        "n_distinct": int(len(counts)),
        "largest_tie_frac": float(counts.max() / score_signed.size) if len(counts) else 1.0,
        "median_abs_nonzero": float(np.median(nz)) if len(nz) else 0.0,
    }
    diagnostics["OUTPUT_DEGENERATE"] = bool(
        diagnostics["n_distinct"] < 0.25 * score_signed.size
        and diagnostics["median_abs_nonzero"] < 1e-12
    )
    model_dir = d / "regvelo_model"
    if model_dir.exists():
        stale = model_dir.with_name(model_dir.name + time.strftime(".stale_%Y%m%dT%H%M%S") + f"_{time.time_ns()}")
        shutil.move(model_dir, stale)
    model.save(model_dir, overwrite=True, save_anndata=False)
    manifest = model_manifest(model_dir)
    if not manifest:
        raise SystemExit("REFUSING_REGVELO_MODEL_NOT_SAVED")
    write_part(result, binding, {
        "genes": np.asarray(D["axis"]),
        "sources": np.asarray(D["sources"]),
        "score_signed": score_signed,
    }, {
        "gpu_physical_id": gpu,
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "cuda_name": torch.cuda.get_device_name(0),
        "train_device": device,
        "wall_s": time.time() - t0,
        "model_manifest": manifest,
        "output_diagnostics": diagnostics,
        "score_semantics": "signed outgoing RegVelo Jacobian column; abs(score) for unsigned evaluation",
    })


def run_gpu(args) -> None:
    system = args.system.upper()
    if system not in SYSTEMS:
        raise SystemExit(f"unknown system {system}")
    set_gpu(args.gpu)
    D = canonical_inputs(system)
    assert_frozen_roster_covered(system, D["sources"])
    system_dir(system).mkdir(parents=True, exist_ok=True)
    if args.method == "velorama":
        run_velorama(D, args.gpu)
    else:
        run_regvelo(D, args.gpu)


# Average-tie percentiles on finite values; missing remains missing
def pct(v: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    v = np.asarray(v, float)
    out = np.full(v.shape, np.nan, float)
    ok = np.isfinite(v)
    if ok.any():
        out[ok] = rankdata(v[ok], method="average") / ok.sum()
    return out


# Rank on one method-specific observational universe, then aggregate
def consensus(root_scores: dict[str, np.ndarray], ks: tuple[int, ...]) -> np.ndarray:
    n = len(next(iter(root_scores.values())))
    universe = np.ones(n, bool)
    for root in ROOTS:
        universe &= np.isfinite(root_scores[root][:, ks]).all(1)
    roots = []
    for root in ROOTS:
        A = np.column_stack([pct(root_scores[root][universe, k]) for k in ks])
        z = np.full(n, np.nan)
        z[universe] = A.mean(1)
        roots.append(z)
    out = np.full(n, np.nan)
    out[universe] = np.minimum(roots[0][universe], roots[1][universe])
    return out


# Call the endpoint generator with the hash-pinned eligibility object
def trace_endpoints(legacy_universe=False, systems=None):
    import gb_cop_alignment as CA

    if __import__("os").environ.get("GB_SKIP_PREFLIGHT") == "1":      # opt-in (public release): skip only the sha pins of the eligibility object / figure contract
        print("preflight skipped (public release)")
    elif sha(FIG1_ELIGIBILITY) != FIG1_ELIGIBILITY_SHA256:
        raise SystemExit("BLOCKED_BY_SPEC_FIG1_ELIGIBILITY_HASH_MISMATCH")
    elif sha(FIGURE_CONTRACT) != FIGURE_CONTRACT_SHA256:
        raise SystemExit("BLOCKED_BY_SPEC_REAL_FIGURE_CONTRACT_HASH_MISMATCH")
    original_json = CA.json
    original_raw_rows = CA.raw_rows

    # Redirect only CA's overwritten Fig-1 path to the frozen axis-v1 bytes
    class FrozenJson:
        @staticmethod
        def load(fh, *args, **kwargs):
            if os.path.realpath(getattr(fh, "name", "")) == os.path.realpath(
                    R / "results" / "fig_contracts" / "fig1_eligibility.json"):
                with open(FIG1_ELIGIBILITY) as frozen:
                    return json.load(frozen, *args, **kwargs)
            return json.load(fh, *args, **kwargs)

    CA.json = FrozenJson
    CA.raw_rows = gb_raw_rows
    try:
        return list(CA.endpoints(legacy_universe=legacy_universe, systems=systems))
    finally:
        CA.json = original_json
        CA.raw_rows = original_raw_rows


def load_matrix(path: Path, key: str = "score") -> dict:
    with np.load(path, allow_pickle=True) as z:
        genes = unique_exact(np.asarray(z["genes"]).astype(str), f"{path.name}_genes")
        sources = unique_exact(np.asarray(z["sources"]).astype(str), f"{path.name}_sources")
        return {"genes": genes, "sources": sources, "score": np.asarray(z[key], float)}


def source_row(M: dict, source: str) -> np.ndarray | None:
    ui = upper_index(M["sources"], "artifact_sources")
    j = ui.get(source.upper())
    return None if j is None else np.asarray(M["score"][j], float)


def exact_project(row: np.ndarray, artifact_genes: list[str], endpoint_genes: np.ndarray) -> np.ndarray:
    gi = {g: i for i, g in enumerate(artifact_genes)}
    missing = [g for g in endpoint_genes if g not in gi]
    if missing:
        raise SystemExit(f"REFUSING_EXACT_TARGET_JOIN: {missing[:10]}")
    return np.asarray(row[[gi[g] for g in endpoint_genes]], float)


# Frozen full-universe expression bins, then method-specific missingness
def fixed_strat_metric(CA, score, y, strat, nbin=10) -> tuple[float, float]:
    from scipy.stats import rankdata

    score, y, strat = np.asarray(score, float), np.asarray(y, bool), np.asarray(strat, float)
    q = np.minimum((rankdata(strat, method="ordinal") - 1) * nbin // len(strat), nbin - 1)
    finite = np.isfinite(score)
    num = weight = 0.0
    for b in range(nbin):
        m = (q == b) & finite
        p = int((m & y).sum())
        n = int((m & ~y).sum())
        if p and n:
            num += p * CA.auroc(score[m], y[m])
            weight += p
    npos = int(y.sum())
    return (num / weight if weight else np.nan, weight / npos if npos else np.nan)


def eval_vector(CA, E, score: np.ndarray, metric_mask=None, panel_member=True) -> dict:
    score = np.asarray(score, float)
    finite = np.isfinite(score)
    if score.shape != E.eval_mask.shape:
        raise ValueError(f"score axis {score.shape} != endpoint observational axis {E.eval_mask.shape}")
    y = np.asarray(E.y, bool)
    own_eval_finite = finite[E.eval_mask]
    selected = (E.eval_mask & finite) if metric_mask is None else np.asarray(metric_mask, bool)
    if selected.shape != E.eval_mask.shape or np.any(selected & ~E.eval_mask):
        raise ValueError("metric mask must be a subset of the endpoint truth-evaluation mask")
    y_selected = y[selected[E.eval_mask]]
    if not panel_member or not selected.any():
        metric = comparable = np.nan
    elif metric_mask is None and E.strat is not None:
        # Exact legacy behavior: bins are formed on the old endpoint universe, with method-specific
        # non-finite rows removed only inside each bin.
        metric, comparable = fixed_strat_metric(CA, score[E.eval_mask], y, E.strat)
    elif E.strat is None:
        metric = CA.auroc(score[selected], y_selected)
        comparable = np.nan
    else:
        metric, comparable = fixed_strat_metric(
            CA, score[selected], y_selected, np.asarray(E.strat)[selected[E.eval_mask]])
    return {
        "metric": float(metric) if np.isfinite(metric) else np.nan,
        "coverage": float(own_eval_finite.mean()),
        "positive_coverage": float(own_eval_finite[y].mean()) if y.any() else np.nan,
        "negative_coverage": float(own_eval_finite[~y].mean()) if (~y).any() else np.nan,
        "positive_comparable_fraction": comparable,
        "n_scored": int(own_eval_finite.sum()),
        "n_positive_scored": int((own_eval_finite & y).sum()),
        "n_metric": int(selected.sum()) if panel_member else 0,
        "n_positive_metric": int(y_selected.sum()) if panel_member else 0,
        "panel_member": bool(panel_member),
    }


# Evaluate active methods on one common finite truth-labelled row set
def evaluate_panel(CA, E, vectors: dict[str, np.ndarray], legacy_universe=False):
    finite = {m: np.isfinite(np.asarray(v, float)) for m, v in vectors.items()}
    active = {m: bool(mask[E.eval_mask].any()) for m, mask in finite.items()}
    common = E.eval_mask.copy()
    if not legacy_universe:
        for method, mask in finite.items():
            if active[method]:
                common &= mask
    out = {}
    for method, score in vectors.items():
        # Legacy parity reproduces the former method-specific finite subset.
        mask = None if legacy_universe else common
        out[method] = eval_vector(CA, E, score, metric_mask=mask, panel_member=active[method])
    common_hash = hashlib.sha256(np.flatnonzero(common).tobytes()).hexdigest()
    return out, common, common_hash


def static_vector(M: dict | None, source: str, genes: np.ndarray, absolute=False) -> np.ndarray:
    if M is None:
        return np.full(len(genes), np.nan)
    row = source_row(M, source)
    if row is None:
        return np.full(len(genes), np.nan)
    v = exact_project(row, M["genes"], genes)
    return np.abs(v) if absolute else v


def info_vector(path: Path | None, source: str, genes: np.ndarray, kind: str) -> np.ndarray:
    if path is None:
        return np.full(len(genes), np.nan)
    with np.load(path, allow_pickle=True) as z:
        ag = unique_exact(np.asarray(z["genes"]).astype(str), "info_genes")
        sources = unique_exact(np.asarray(z["sources"]).astype(str), "info_sources")
        ui = upper_index(sources, "info_sources")
        if source.upper() not in ui:
            return np.full(len(genes), np.nan)
        r = ui[source.upper()]
        root_scores = {}
        for root in ROOTS:
            root_scores[root] = np.column_stack([
                np.asarray(z[f"{root}|{kind}|k{k}"][r], float) for k in KS_INFO
            ])
    # Rank on this method's own full non-self finite observational universe, then perform the
    # exact-name projection to GeoBRIDGE U_obs.
    self_mask = np.char.upper(np.asarray(ag).astype(str)) == source.upper()
    for root in ROOTS:
        root_scores[root][self_mask] = np.nan
    full = consensus(root_scores, KS_COND)
    return exact_project(full, ag, genes)


# Read GB vectors ranked on U_obs by the first-hit endpoint loader
def gb_vectors(CA, system: str, source: str, eps: dict) -> dict[str, np.ndarray]:
    if set(eps) != {"U", "X"}:
        raise SystemExit(f"REFUSING_GB_ENDPOINT_CHANNELS: {system}/{source}: {sorted(eps)}")
    if not np.array_equal(eps["U"].obs_genes, eps["X"].obs_genes):
        raise SystemExit(f"REFUSING_GB_ENDPOINT_AXIS_MISMATCH: {system}/{source}")
    out = {}
    for channel, raw_name, cond_name in (
        ("U", "GeoBRIDGE-RAW", "GeoBRIDGE-COND"),
        ("X", "PairStatic-RAW", "PairStatic-COND"),
    ):
        E = eps[channel]
        out[raw_name] = np.minimum(*[E.P[root][:, [0]].mean(1) for root in ROOTS])
        out[cond_name] = np.minimum(*[E.P[root][:, list(KS_COND)].mean(1) for root in ROOTS])
    return out


def context_of(CA, endpoint: str) -> str:
    return CA.UNIT(endpoint)


def required_artifacts() -> dict[str, list[Path]]:
    return {s: [
        system_dir(s) / "pearson.npz",
        system_dir(s) / "genie3.npz",
        system_dir(s) / "grnboost2.npz",
        system_dir(s) / "infomatched.npz",
        system_dir(s) / "velorama.npz",
        system_dir(s) / "regvelo.npz",
    ] for s in SYSTEMS}


def check_artifact(path: Path, accepted_producer_sha256=None) -> None:
    receipt = path.with_suffix(".receipt.json")
    if not receipt.exists():
        raise SystemExit(f"REFUSING_NO_RECEIPT: {path}")
    r = json.load(open(receipt))
    if r.get("output_sha256") != sha(path):
        raise SystemExit(f"REFUSING_OUTPUT_HASH_MISMATCH: {path}")
    b = r.get("binding", {})
    if r.get("truth_read") is not False or b.get("truth_read") is not False:
        raise SystemExit(f"REFUSING_NON_TRUTHFREE_COMPARATOR: {path}")
    # A self-consistent old receipt is still stale if any canonical input, GB roster cache,
    # dependency, or these producer bytes changed after scoring.
    claimed = b.get("binding_sha256")
    chk = dict(b)
    chk.pop("binding_sha256", None)
    if not claimed or claimed != digest_object(chk):
        raise SystemExit(f"REFUSING_INVALID_BINDING_DIGEST: {path}")
    accepted = ({script_sha(), PRE_UNIVERSE_REPAIR_SHA256} if accepted_producer_sha256 is None
                else set(accepted_producer_sha256))
    if b.get("producer_sha256") not in accepted:
        raise SystemExit(f"REFUSING_STALE_PRODUCER: {path}")
    inp = b.get("inputs", {})
    for label, pkey, hkey in (
        ("fixture", "fixture", "fixture_sha256"),
        ("pairs", "pairs", "pairs_sha256"),
    ):
        p = Path(inp.get(pkey, ""))
        if not p.is_file() or inp.get(hkey) != sha(p):
            raise SystemExit(f"REFUSING_STALE_{label.upper()}: {path}")
    for item in inp.get("gb_score_caches", []):
        p = Path(item.get("path", ""))
        if not p.is_file() or item.get("sha256") != sha(p):
            raise SystemExit(f"REFUSING_STALE_GB_SCORE_CACHE: {path}")
    cfg = b.get("config", {})
    for key, dep in (
        ("arboreto_wrapper", SCRATCH / "gb_sota_run.py"),
        ("implementation_sha256", SCRATCH / "gb_infomatched.py"),
        ("harvester_sha256", PAIRSPACE / "gb_velorama_harvest.py"),
    ):
        if key in cfg and cfg[key] != sha(dep):
            raise SystemExit(f"REFUSING_STALE_DEPENDENCY_{key.upper()}: {path}")
    if "upstream" in cfg:
        upstream_dir = R / "external" / "velorama_official" / "velorama" / "velorama"
        now = {p.name: sha(p) for p in sorted(upstream_dir.glob("*.py"))}
        if cfg["upstream"] != now:
            raise SystemExit(f"REFUSING_STALE_VELORAMA_UPSTREAM: {path}")
    # Output scores are sufficient for evaluation, but the receipt's assertion that they came from
    # persisted trained models must remain auditable.
    for item in r.get("models", []) + r.get("model_manifest", []):
        p = Path(item.get("path", ""))
        if not p.is_file() or item.get("sha256") != sha(p):
            raise SystemExit(f"REFUSING_MODEL_MANIFEST_MISMATCH: {path}")


def evaluation_inputs(partial: bool, legacy_universe=False) -> tuple[dict, dict[str, list[Path]]]:
    req = required_artifacts()
    missing = {s: [p for p in ps if not p.exists()] for s, ps in req.items()}
    missing = {s: ps for s, ps in missing.items() if ps}
    if missing and not partial:
        msg = {s: [str(p) for p in ps] for s, ps in missing.items()}
        raise SystemExit("REFUSING_INCOMPLETE_COMPARATOR_SET: " + json.dumps(msg, sort_keys=True))
    present = {s: [p for p in ps if p.exists()] for s, ps in req.items()}
    for ps in present.values():
        for p in ps:
            check_artifact(p)
    contracts = [
        FIGURE_CONTRACT,
        FIG1_ELIGIBILITY,
        R / "GEOBRIDGE_CANONICAL_RESULTS_2026_09_18-19.md",
    ]
    manifest = {
        "partial": partial,
        "legacy_universe": bool(legacy_universe),
        "artifacts": {s: {p.name: sha(p) for p in ps} for s, ps in present.items()},
        "gb_score_caches": {
            s: {str(p): sha(p) for p in cache_paths(s) if p.exists()} for s in SYSTEMS
        },
        "contracts": {str(p): sha(p) for p in contracts},
        "producer_sha256": script_sha(),
        "missing": {s: [str(p) for p in ps] for s, ps in missing.items()},
    }
    manifest["binding_sha256"] = digest_object(manifest)
    return manifest, present


def load_system_methods(system: str, present: list[Path]) -> dict:
    by = {p.name: p for p in present}
    out = {}
    for filename, label, key in (
        ("pearson.npz", "Pearson", "score"),
        ("genie3.npz", "GENIE3", "score"),
        ("grnboost2.npz", "GRNBoost2", "score"),
        ("velorama.npz", "Velorama", "score"),
        ("regvelo.npz", "RegVelo", "score_signed"),
    ):
        if filename not in by:
            out[label] = None
            continue
        M = load_matrix(by[filename], key)
        if label == "RegVelo":
            M["score"] = np.abs(M["score"])
            rec = json.load(open(by[filename].with_suffix(".receipt.json")))
            if rec.get("output_diagnostics", {}).get("OUTPUT_DEGENERATE"):
                M["score"][:] = np.nan
                M["status"] = "NOT_SCORED_DEGENERATE"
        out[label] = M
    out["info_path"] = by.get("infomatched.npz")
    return out


def build_endpoint_table(partial: bool, present: dict[str, list[Path]], legacy_universe=False,
                         systems=None) -> pd.DataFrame:
    import gb_cop_alignment as CA

    endpoints = trace_endpoints(legacy_universe=legacy_universe, systems=systems)
    if not partial and systems is None:
        seen = {system for _, _, system, _, _, _, _ in endpoints}
        expected = {"GATA1", "PANC", "BONEMARROW", "DENTATEGYRUS", "ZEBRAFISH", "K562", "GASTRULATION"}
        if seen != expected:
            raise SystemExit(f"REFUSING_ENDPOINT_SYSTEM_COVERAGE: seen={sorted(seen)} expected={sorted(expected)}")
        f1 = pd.DataFrame([{"system": system, "source": source} for _, family, system, source, _, _, _ in endpoints
                           if family == "F1"])
        for system in sorted(set(expected) - {"K562"}):
            want = set(frozen_fig1_roster(system))
            got = set(f1.loc[f1.system == system, "source"].str.upper())
            if want != got:
                missing, extra = sorted(want - got), sorted(got - want)
                raise SystemExit(
                    f"BLOCKED_BY_SPEC_F1_ENDPOINT_COVERAGE: {system} frozen={len(want)} yielded={len(got)} "
                    f"missing={missing[:10]} extra={extra[:10]}; endpoints() skips uncached GB sources "
                    "and the completion cache is still insufficient"
                )
    selected_systems = SYSTEMS if systems is None else tuple(s.upper() for s in systems)
    loaded = {s: load_system_methods(s, present.get(s, [])) for s in selected_systems}
    rows = []
    for endpoint, family, system, source, eps, n, npos in endpoints:
        E = eps["U"]
        if not hasattr(E, "genes"):
            raise AssertionError("endpoint trace did not attach genes")
        vectors = gb_vectors(CA, system, source, eps)
        M = loaded[system]
        for method in ("Pearson", "GENIE3", "GRNBoost2", "Velorama", "RegVelo"):
            vectors[method] = static_vector(M[method], source, E.obs_genes)
        vectors["InfoLinear-COND"] = info_vector(M["info_path"], source, E.obs_genes, "linear")
        vectors["InfoSpline-COND"] = info_vector(M["info_path"], source, E.obs_genes, "spline")
        vals, common, common_hash = evaluate_panel(CA, E, vectors, legacy_universe=legacy_universe)
        for method in METHOD_ORDER:
            val = vals[method]
            status = "OK" if val["panel_member"] and np.isfinite(val["metric"]) else "NOT_SCORED"
            if method == "RegVelo" and M["RegVelo"] is not None:
                status = M["RegVelo"].get("status", status)
            rows.append({
                "family": family,
                "endpoint": endpoint,
                "context": context_of(CA, endpoint),
                "system": system,
                "source": source,
                "method": method,
                "metric_name": "within_expression_stratum_AUROC" if family == "F1" else "AUROC",
                "n_universe": n,
                "n_positive": npos,
                "n_u_obs": E.n_obs,
                "u_obs_sha256": E.obs_sha256,
                "score_universe_sha256": E.score_universe_sha256,
                "legacy_universe": bool(legacy_universe),
                "n_panel_common": int(common.sum()),
                "n_positive_panel_common": int(E.y[common[E.eval_mask]].sum()),
                "panel_common_sha256": common_hash,
                "no_call": system in NO_CALL_SYSTEMS,
                "no_call_reason": NO_CALL_SYSTEMS.get(system, ""),
                "status": status,
                **val,
            })
    return pd.DataFrame(rows)


def context_table(E: pd.DataFrame, family: str) -> pd.DataFrame:
    rows = []
    sub = E[E.family == family]
    for (context, method), g in sub.groupby(["context", "method"], sort=False):
        ok = np.isfinite(g.metric)
        rows.append({
            "family": family,
            "context": context,
            "system": ",".join(sorted(g.system.unique())),
            "method": method,
            "metric_name": g.metric_name.iloc[0],
            "tf_median_metric": float(g.loc[ok, "metric"].median()) if ok.any() else np.nan,
            "tf_q25_metric": float(g.loc[ok, "metric"].quantile(.25)) if ok.any() else np.nan,
            "tf_q75_metric": float(g.loc[ok, "metric"].quantile(.75)) if ok.any() else np.nan,
            "median_coverage": float(g.coverage.median()),
            "minimum_coverage": float(g.coverage.min()),
            "median_positive_coverage": float(g.positive_coverage.median()),
            "median_positive_comparable_fraction": float(g.positive_comparable_fraction.median())
            if g.positive_comparable_fraction.notna().any() else np.nan,
            "n_tf_total": int(len(g)),
            "n_tf_scored": int(ok.sum()),
            "no_call": bool(g.no_call.any()),
            "no_call_reason": ";".join(sorted(set(x for x in g.no_call_reason if x))),
        })
    T = pd.DataFrame(rows)
    T["method"] = pd.Categorical(T.method, METHOD_ORDER, ordered=True)
    return T.sort_values(["context", "method"]).assign(method=lambda x: x.method.astype(str))


def plot_family(T: pd.DataFrame, family: str, path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    contexts = list(dict.fromkeys(T.context))
    methods = [m for m in METHOD_ORDER if m in set(T.method)]
    A = T.pivot(index="context", columns="method", values="tf_median_metric").reindex(index=contexts, columns=methods)
    C = T.pivot(index="context", columns="method", values="median_coverage").reindex(index=contexts, columns=methods)
    width = max(11, .72 * len(methods) + 4)
    height = max(3.5, .55 * len(contexts) + 2.4)
    fig, ax = plt.subplots(figsize=(width, height))
    im = ax.imshow(A.to_numpy(float), vmin=.35, vmax=.75, cmap="RdBu_r", aspect="auto")
    for i, context in enumerate(contexts):
        for j, method in enumerate(methods):
            v, c = A.loc[context, method], C.loc[context, method]
            txt = "NA" if not np.isfinite(v) else f"{v:.3f}\n({c:.0%})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7,
                    color="white" if np.isfinite(v) and (v < .41 or v > .69) else "black")
    ax.set_xticks(range(len(methods)), methods, rotation=45, ha="right")
    labels = [c + ("  [NO CALL]" if T[T.context == c].no_call.any() else "") for c in contexts]
    ax.set_yticks(range(len(contexts)), labels)
    ax.set_title(f"{family}: TF-median frozen metric (parentheses = median coverage)")
    fig.colorbar(im, ax=ax, label="AUROC")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def existing_eval_ok(receipt: Path, manifest: dict) -> bool:
    if not receipt.exists():
        return False
    old = json.load(open(receipt))
    if old.get("input_manifest") != manifest:
        raise SystemExit("REFUSING_ONE_EVALUATION_ALREADY_EXISTS_WITH_DIFFERENT_INPUTS")
    for p, h in old.get("outputs", {}).items():
        if not os.path.exists(p) or sha(p) != h:
            raise SystemExit("REFUSING_EXISTING_EVALUATION_OUTPUT_CHANGED")
    print(f"[eval] exact evaluation already exists and validates: {receipt}", flush=True)
    return True


def run_eval(args) -> None:
    manifest, present = evaluation_inputs(args.partial, args.legacy_universe)
    if args.legacy_universe:
        out_dir = OUT / ("partial_legacy" if args.partial else "legacy")
    else:
        out_dir = OUT / "partial" if args.partial else OUT
    out_dir.mkdir(parents=True, exist_ok=True)
    receipt = out_dir / "EVAL_RECEIPT.json"
    if existing_eval_ok(receipt, manifest):
        return
    E = build_endpoint_table(args.partial, present, legacy_universe=args.legacy_universe)
    outputs = {}
    ep = out_dir / "ENDPOINTS.tsv"
    E.to_csv(ep, sep="\t", index=False, na_rep="NA")
    outputs[str(ep)] = sha(ep)
    for family in ("F1", "F2", "F3"):
        T = context_table(E, family)
        table = out_dir / f"TABLE_{family}.tsv"
        fig = out_dir / f"FIG_{family}.png"
        T.to_csv(table, sep="\t", index=False, na_rep="NA")
        plot_family(T, family, fig)
        outputs[str(table)], outputs[str(fig)] = sha(table), sha(fig)
    rec = {
        "command": "eval",
        "partial": args.partial,
        "legacy_universe": bool(args.legacy_universe),
        "one_evaluation": True,
        "input_manifest": manifest,
        "outputs": outputs,
        "n_endpoint_method_rows": int(len(E)),
        "n_endpoints": int(E.endpoint.nunique()),
        "gastrulation_no_call_preserved": bool(
            E.loc[E.system == "GASTRULATION", "no_call"].all()
            and (E.loc[E.system == "GASTRULATION", "no_call_reason"] == NO_CALL_SYSTEMS["GASTRULATION"]).all()
        ),
        "metric_contract": {"F1": "within-expression-decile AUROC", "F2": "AUROC", "F3": "AUROC"},
        "missing_score_policy": ("legacy method-specific finite subsets; never zero-filled" if args.legacy_universe else
                                 "method coverage reported separately; active panel evaluated on identical common finite rows; never zero-filled"),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    atomic_json(receipt, rec)
    print(json.dumps(rec, indent=1), flush=True)


# One-system old/new evaluation comparison; never writes figure outputs
def run_smoke(args) -> None:
    system = args.system.upper()
    paths = [p for p in required_artifacts()[system] if p.exists()]
    for path in paths:
        check_artifact(path)
    present = {system: paths}
    old = build_endpoint_table(True, present, legacy_universe=True, systems=[system])
    new = build_endpoint_table(True, present, legacy_universe=False, systems=[system])
    keys = ["family", "endpoint", "system", "source", "method"]
    keep = keys + ["metric", "coverage", "n_universe", "n_u_obs", "u_obs_sha256",
                   "score_universe_sha256", "n_panel_common", "panel_common_sha256", "status"]
    out = old[keep].merge(new[keep], on=keys, suffixes=("_legacy", "_new"), validate="one_to_one")
    out["metric_delta_new_minus_legacy"] = out.metric_new - out.metric_legacy
    print(f"SMOKE system={system} comparator_artifacts={len(paths)} "
          f"producer_compat={PRE_UNIVERSE_REPAIR_SHA256}", flush=True)
    out.to_csv(sys.stdout, sep="\t", index=False, na_rep="NA")


def selftest() -> None:
    import gb_cop_alignment as CA

    # Tie-aware percentile and missingness.
    p = pct(np.array([1.0, 1.0, 3.0, np.nan]))
    assert np.allclose(p[:3], [0.5, 0.5, 1.0]) and np.isnan(p[3])

    # Frozen order of operations: rank within k/universe, mean k, min roots.
    a = np.array([[1., 3., 2.], [2., 2., 3.], [3., 1., 1.]])
    b = np.array([[3., 3., 3.], [2., 2., 2.], [1., 1., 1.]])
    got = consensus({"dc1_min": a, "dc1_max": b}, (1, 2))
    ref_a = np.column_stack([pct(a[:, 1]), pct(a[:, 2])]).mean(1)
    ref_b = np.column_stack([pct(b[:, 1]), pct(b[:, 2])]).mean(1)
    assert np.allclose(got, np.minimum(ref_a, ref_b))

    # Missing at one required root/k reduces coverage and is never zero-filled.
    bm = b.copy(); bm[1, 2] = np.nan
    z = consensus({"dc1_min": a, "dc1_max": bm}, (1, 2))
    assert np.isnan(z[1]) and np.isfinite(z[[0, 2]]).all()

    # AUROC and fixed full-universe strata: a perfect score stays perfect.
    y = np.tile(np.array([True, False]), 20)
    strat = np.repeat(np.arange(10), 4)
    score = np.where(y, 2.0, 1.0)
    m, cov = fixed_strat_metric(CA, score, y, strat)
    assert m == 1.0 and cov == 1.0
    score[0] = np.nan
    m2, cov2 = fixed_strat_metric(CA, score, y, strat)
    assert m2 == 1.0 and cov2 < 1.0

    # Active comparators share one identical finite evaluation set, while each retains its own
    # coverage and a fully missing method remains NOT_SCORED.
    raw = {h: np.tile(np.arange(6, dtype=float)[:, None], (1, len(GB_KS))) for h in ROOTS}
    Et = CA.Endpoint("toy", "F2", "S", "SRC", raw, np.array([1, 0, 1, 0], bool), CA.auroc,
                     eval_mask=np.array([1, 1, 1, 1, 0, 0], bool), obs_genes=np.array(list("abcdef")))
    va = np.arange(6, dtype=float)
    vb = va.copy(); vb[1] = np.nan
    vc = np.full(6, np.nan)
    vals, common, _ = evaluate_panel(CA, Et, {"a": va, "b": vb, "c": vc})
    assert np.array_equal(common, [True, False, True, True, False, False])
    assert vals["a"]["n_metric"] == vals["b"]["n_metric"] == 3
    assert vals["a"]["coverage"] == 1.0 and vals["b"]["coverage"] == .75
    assert not vals["c"]["panel_member"] and np.isnan(vals["c"]["metric"])

    # Information-matched percentiles are formed on that method's own full non-self universe before
    # projection to GeoBRIDGE U_obs.
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "info.npz"
        arr = np.array([[9.0, 3.0, 2.0, 1.0]])
        arrays = {f"{root}|{kind}|k{k}": arr for root in ROOTS
                  for kind in ("linear", "spline") for k in KS_INFO}
        atomic_npz(path, genes=np.array(["SRC", "A", "B", "C"]),
                   sources=np.array(["SRC"]), **arrays)
        info = info_vector(path, "SRC", np.array(["A", "B"]), "linear")
        assert np.allclose(info, [1.0, 2 / 3])

    # Exact-name projection refuses case-only or missing aliases.
    assert np.array_equal(exact_project(np.array([4., 7.]), ["Gata1", "Tal1"], np.array(["Tal1"])), [7.])
    try:
        exact_project(np.array([4., 7.]), ["Gata1", "Tal1"], np.array(["TAL1"]))
        raise AssertionError("case-insensitive target join was accepted")
    except SystemExit as e:
        assert "EXACT_TARGET_JOIN" in str(e)

    # Hash-bound part reuse and tamper refusal.
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "x.npz"
        binding = {"binding_sha256": "toy", "truth_read": False}
        write_part(path, binding, {"genes": np.array(["a"]), "sources": np.array(["a"]), "score": np.array([[1.]])})
        assert valid_part(path, binding)
        with open(path, "ab") as fh:
            fh.write(b"tamper")
        assert not valid_part(path, binding)

    # Frozen GB lookup order: cop -> regulon panel -> completion cache
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        genes = np.array(["A", "B", "C"])
        cop, panel, completion = td / "cop.npz", td / "panel.npz", td / "completion.npz"

        def cube(n, value):
            return np.full((n, len(genes), len(GB_KS)), value, np.float32)

        atomic_npz(cop, genes=genes, sources=np.array(["A"]),
                   **{f"{r}|{c}": cube(1, 1.0) for r in ROOTS for c in ("U", "X")})
        atomic_npz(panel, genes=genes, sources=np.array(["A", "B"]),
                   **{f"{r}|{c}|absS": np.concatenate([cube(1, 2.0), cube(1, 3.0)])
                      for r in ROOTS for c in ("U", "X")})
        atomic_npz(completion, genes=genes, sources=np.array(["B", "C"]),
                   **{f"{r}|{c}": np.concatenate([cube(1, 4.0), cube(1, 5.0)])
                      for r in ROOTS for c in ("U", "X")})
        rows, got_axis = gb_raw_rows("TOY", ["A", "B", "C"], [cop, panel, completion])
        assert list(got_axis) == list(genes)
        assert np.all(rows["A"][("dc1_min", "U")] == 1.0)
        assert np.all(rows["B"][("dc1_min", "U")] == 3.0)
        assert np.all(rows["C"][("dc1_min", "U")] == 5.0)

    # Parity covers every root/channel/target/k cell, accepts <= tolerance, and actually refuses a
    # value beyond the bound.
    scored = {(r, c): np.full((1, 2, len(GB_KS)), 0.25, np.float32)
              for r in ROOTS for c in ("U", "X")}
    reference = {"A": {(r, c): np.full((2, len(GB_KS)), 0.25 + GB_PARITY_TOL / 2)
                       for r in ROOTS for c in ("U", "X")}}
    worst, _ = assert_absS_parity(scored, ["A"], reference, ["A"])
    assert worst <= GB_PARITY_TOL
    bad = {key: value.copy() for key, value in scored.items()}
    bad[("dc1_max", "X")][0, 1, 30] += 2 * GB_PARITY_TOL
    try:
        assert_absS_parity(bad, ["A"], reference, ["A"])
        raise AssertionError("out-of-tolerance GB parity was accepted")
    except SystemExit as e:
        assert "GBCACHE_PARITY_VALUE" in str(e)
    print("gb_real_uniform selftest PASS")


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("selftest")
    b = sub.add_parser("gbcache")
    b.add_argument("system", choices=tuple(GBCACHE_EXPECTED_MISSING))
    c = sub.add_parser("cpu")
    c.add_argument("system", choices=SYSTEMS)
    c.add_argument("--workers", type=int, default=32)
    c.add_argument("--target-chunk", type=int, default=128)
    g = sub.add_parser("gpu")
    g.add_argument("system", choices=SYSTEMS)
    g.add_argument("--gpu", type=int, required=True)
    g.add_argument("--method", choices=("velorama", "regvelo"), required=True)
    e = sub.add_parser("eval")
    e.add_argument("--partial", action="store_true")
    e.add_argument("--legacy-universe", action="store_true")
    s = sub.add_parser("smoke")
    s.add_argument("system", choices=SYSTEMS)
    return ap


if __name__ == "__main__":
    args = parser().parse_args()
    if args.cmd == "selftest":
        selftest()
    elif args.cmd == "gbcache":
        run_gbcache(args)
    elif args.cmd == "cpu":
        run_cpu(args)
    elif args.cmd == "gpu":
        run_gpu(args)
    elif args.cmd == "smoke":
        run_smoke(args)
    else:
        run_eval(args)
