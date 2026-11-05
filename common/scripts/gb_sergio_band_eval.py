#!/usr/bin/env python
"""Evaluate the preregistered SERGIO World-P conditioned-band experiment.

This is a read-only evaluator of ``gb_sergio_drift_v5.run_one`` artifacts.  It
does not simulate, refit, or select a depth from real data.  The score and gate
contract is pinned to::

  results/sergio_band/PREREG_BAND_v2_2026_09_25.md
  sha256 4bdd24bb1a96a81e5f8e66684b2a027a651e77a1709329a65595bccdfd6a6c82

Usage (run on a compute server, from the repository root)::

  python scratch/gb_sergio_band_eval.py selftest
  python scratch/gb_sergio_band_eval.py evaluate \
      --input results/sergio_drift/core_P --output results/sergio_band

Confirmation data are sealed from the development selector.  They can only be
read with an already-frozen development band::

  python scratch/gb_sergio_band_eval.py evaluate --confirm --frozen-band 5 \
      --input results/sergio_drift/core_P_confirmation \
      --output results/sergio_band/confirmation_B5

Oracle k and K90/K95 are descriptive diagnostics only.  They are never emitted
as comparator rows and never participate in band selection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO = Path(__file__).resolve().parents[1]
SCRATCH = REPO / "scratch"
if str(SCRATCH) not in sys.path:
    sys.path.insert(0, str(SCRATCH))

PREREG = REPO / "results" / "sergio_band" / "PREREG_BAND_v2_2026_09_25.md"
PREREG_SHA256 = "4bdd24bb1a96a81e5f8e66684b2a027a651e77a1709329a65595bccdfd6a6c82"
ROOTS = ("dc1_min", "dc1_max")
CHANNELS = ("U", "X")
OBSERVATIONS = ("UMI_DEPTH", "CONC")
TOPOLOGIES = ("linear", "bifurcation", "trifurcation", "tree")
TOPOLOGY_COMPLEXITY = {x: i + 1 for i, x in enumerate(TOPOLOGIES)}
CANDIDATE_M = (3, 4, 5, 6, 8)
K_GRID = tuple(range(31))
SELECTION_G = (0.15, 0.30, 0.45, 0.60)
REQUIRED_G = (0.0, 0.15, 0.30, 0.45, 0.60, 0.75)
DEV_GRNS = tuple(range(10))
CONF_GRNS = tuple(range(100, 110))
DEV_SEEDS = (0, 1)  # gb_sergio_drift_v5 run default; seed 0 is the existing development launch
CONF_SEEDS = (1, 2)
PRIMARY_CHANNEL = "U"
FROZEN_UMI_DEPTH = 5.718

SWEEP_COLUMNS = [
    "dataset", "dataset_path", "mode", "obs", "depth", "topology", "G_target",
    "G_hidden", "grn", "seed", "channel", "k", "status", "n_primary_total",
    "n_primary_scored", "coverage_primary", "n_direct_primary", "prevalence",
    "pooled_AP", "AP_over_prevalence", "AUROC", "direct_vs_indirect_AUROC",
    "indirect_at_50", "macro_AP", "decoy_universe_AP", "n_decoy_total",
    "n_decoy_scored", "coverage_decoy", "decoy_prevalence",
]
BAND_COLUMNS = [
    "dataset", "dataset_path", "mode", "obs", "depth", "topology", "G_target",
    "G_hidden", "grn", "seed", "channel", "band", "m", "status",
] + SWEEP_COLUMNS[13:]
COMPARATOR_COLUMNS = [
    "dataset", "dataset_path", "mode", "obs", "depth", "topology", "G_target",
    "G_hidden", "grn", "seed", "method", "source_key", "artifact", "status",
] + SWEEP_COLUMNS[13:]

METRIC_COLUMNS = [
    "n_primary_total", "n_primary_scored", "coverage_primary", "n_direct_primary",
    "prevalence", "pooled_AP", "AP_over_prevalence", "AUROC",
    "direct_vs_indirect_AUROC", "indirect_at_50", "macro_AP",
    "decoy_universe_AP", "n_decoy_total", "n_decoy_scored", "coverage_decoy",
    "decoy_prevalence",
]

COMPARATOR_KEYS = (
    ("Pearson", "Pearson_abs"),
    ("GENIE3", "GENIE3"),
    ("GRNBoost2", "GRNBOOST2"),
    ("InfoMatched", "INFO_MATCHED_U"),
    ("InfoLinear", "INFO_LINEAR"),
    ("InfoSpline", "INFO_SPLINE"),
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def jsonable(x: Any) -> Any:
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return None if not np.isfinite(x) else float(x)
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    return x


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp.{os.getpid()}")
    tmp.write_text(text)
    os.replace(tmp, path)


def atomic_json(path: Path, obj: Any) -> None:
    atomic_text(path, json.dumps(jsonable(obj), indent=2, sort_keys=True) + "\n")


def atomic_tsv(path: Path, frame: pd.DataFrame, columns: list[str]) -> None:
    out = frame.reindex(columns=columns)
    atomic_text(path, out.to_csv(sep="\t", index=False, na_rep="NA"))


def assert_prereg() -> None:
    if __import__("os").environ.get("GB_SKIP_PREFLIGHT") == "1":      # opt-in (public release)
        print("preflight skipped (public release)")
        return
    if not PREREG.exists():
        raise SystemExit(f"REFUSING_MISSING_PREREG: {PREREG}")
    got = sha256_file(PREREG)
    if got != PREREG_SHA256:
        raise SystemExit(f"REFUSING_PREREG_SHA: expected={PREREG_SHA256} got={got}")


def rank_percentile(matrix: np.ndarray) -> np.ndarray:
    """Average-tie percentile within each fixed row universe.

    Missing entries remain missing.  Finite ranks are divided by the original
    universe size, not by the number of observed entries, so missingness cannot
    silently redefine the observational universe.
    """
    from scipy.stats import rankdata

    a = np.asarray(matrix, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"rank_percentile expects 2-D input, got {a.shape}")
    out = np.full(a.shape, np.nan, dtype=float)
    n_universe = a.shape[1]
    for i, row in enumerate(a):
        finite = np.isfinite(row)
        if finite.any():
            out[i, finite] = rankdata(row[finite], method="average") / n_universe
    return out


def root_consensus(root_scores: dict[str, np.ndarray]) -> np.ndarray:
    if set(root_scores) != set(ROOTS):
        raise ValueError(f"root keys must be {ROOTS}, got {sorted(root_scores)}")
    a, b = (np.asarray(root_scores[r], float) for r in ROOTS)
    if a.shape != b.shape:
        raise ValueError(f"root shape mismatch: {a.shape} vs {b.shape}")
    return np.minimum(a, b)  # np.minimum deliberately propagates missingness


def conditioned_band(raw: Any, cols: np.ndarray, channel: str, m: int) -> np.ndarray:
    """C_m = min_root mean_{k=1..m} pct_Ur(|S_root,k|)."""
    if m not in CANDIDATE_M:
        raise ValueError(f"unregistered band m={m}")
    per_root: dict[str, np.ndarray] = {}
    for root in ROOTS:
        ranked = []
        key = f"TRUE|{channel}|{root}|absS"
        if key not in raw.files:
            raise KeyError(key)
        arr = np.asarray(raw[key], float)
        for k in range(1, m + 1):
            ranked.append(rank_percentile(np.take_along_axis(arr[k], cols, axis=1)))
        per_root[root] = np.mean(np.stack(ranked), axis=0)
    return root_consensus(per_root)


def single_depth(raw: Any, cols: np.ndarray, channel: str, k: int) -> np.ndarray:
    per_root = {}
    for root in ROOTS:
        key = f"TRUE|{channel}|{root}|absS"
        if key not in raw.files:
            raise KeyError(key)
        arr = np.asarray(raw[key], float)
        per_root[root] = rank_percentile(np.take_along_axis(arr[k], cols, axis=1))
    return root_consensus(per_root)


def strict_two_hop(gt: Iterable[tuple[int, int]], n_genes: int) -> set[tuple[int, int]]:
    """Exactly length-two paths, excluding every direct edge."""
    direct = {(int(s), int(t)) for s, t in gt}
    out_by_source: dict[int, set[int]] = {}
    for s, t in direct:
        out_by_source.setdefault(s, set()).add(t)
    indirect = set()
    for source, mids in out_by_source.items():
        for middle in mids:
            for target in out_by_source.get(middle, ()):
                if (source, target) not in direct:
                    indirect.add((source, target))
    return indirect


def truth_arrays(
    sources: np.ndarray, cols: np.ndarray, direct: set[tuple[int, int]],
    indirect: set[tuple[int, int]], true_regulators: set[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.zeros(cols.shape, dtype=bool)
    labels = np.full(cols.shape, "none", dtype="<U8")
    for i, source in enumerate(sources):
        for j, target in enumerate(cols[i]):
            edge = (int(source), int(target))
            if edge in direct:
                y[i, j] = True
                labels[i, j] = "direct"
            elif edge in indirect:
                labels[i, j] = "indirect"
    true_rows = np.asarray([int(s) in true_regulators for s in sources], dtype=bool)
    return y, labels, true_rows


def average_precision(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    y = np.asarray(y, bool)
    score = np.asarray(score, float)
    if not (0 < int(y.sum()) < len(y)):
        return float("nan")
    return float(average_precision_score(y, score))


def auc(y: np.ndarray, score: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.asarray(y, bool)
    score = np.asarray(score, float)
    if not (0 < int(y.sum()) < len(y)):
        return float("nan")
    return float(roc_auc_score(y, score))


def tie_averaged_top_fraction(score: np.ndarray, indicator: np.ndarray, k: int) -> float:
    """Expected indicator fraction under uniform ordering within the boundary tie."""
    score = np.asarray(score, float)
    indicator = np.asarray(indicator, bool)
    finite = np.isfinite(score)
    score, indicator = score[finite], indicator[finite]
    if score.size == 0:
        return float("nan")
    k_eff = min(int(k), score.size)
    cutoff = np.partition(score, score.size - k_eff)[score.size - k_eff]
    above, tied = score > cutoff, score == cutoff
    remaining = k_eff - int(above.sum())
    expected = float(indicator[above].sum())
    if remaining and tied.sum():
        expected += remaining * float(indicator[tied].mean())
    return expected / k_eff


def metrics(
    score: np.ndarray, y: np.ndarray, labels: np.ndarray, true_rows: np.ndarray,
) -> dict[str, float | int]:
    score = np.asarray(score, float)
    if score.shape != y.shape or score.shape != labels.shape:
        raise ValueError(f"metric shape mismatch score={score.shape} truth={y.shape} labels={labels.shape}")

    sp = score[true_rows].ravel()
    yp = y[true_rows].ravel()
    lp = labels[true_rows].ravel()
    fp = np.isfinite(sp)
    n_total = int(sp.size)
    n_scored = int(fp.sum())
    prevalence = float(yp[fp].mean()) if n_scored else float("nan")
    ap = average_precision(yp[fp], sp[fp]) if n_scored else float("nan")
    di = fp & ((lp == "direct") | (lp == "indirect"))
    dvi = auc(lp[di] == "direct", sp[di]) if di.any() else float("nan")

    macro = []
    true_indices = np.flatnonzero(true_rows)
    for i in true_indices:
        finite = np.isfinite(score[i])
        value = average_precision(y[i, finite], score[i, finite]) if finite.any() else float("nan")
        if np.isfinite(value):
            macro.append(value)

    sd = score.ravel()
    yd = y.ravel()
    fd = np.isfinite(sd)
    decoy_prev = float(yd[fd].mean()) if fd.any() else float("nan")
    decoy_ap = average_precision(yd[fd], sd[fd]) if fd.any() else float("nan")

    return {
        "n_primary_total": n_total,
        "n_primary_scored": n_scored,
        "coverage_primary": n_scored / n_total if n_total else float("nan"),
        "n_direct_primary": int(yp[fp].sum()),
        "prevalence": prevalence,
        "pooled_AP": ap,
        "AP_over_prevalence": ap / prevalence if np.isfinite(ap) and prevalence > 0 else float("nan"),
        "AUROC": auc(yp[fp], sp[fp]) if n_scored else float("nan"),
        "direct_vs_indirect_AUROC": dvi,
        "indirect_at_50": tie_averaged_top_fraction(sp, lp == "indirect", 50),
        "macro_AP": float(np.mean(macro)) if macro else float("nan"),
        "decoy_universe_AP": decoy_ap,
        "n_decoy_total": int(sd.size),
        "n_decoy_scored": int(fd.sum()),
        "coverage_decoy": float(fd.mean()) if fd.size else float("nan"),
        "decoy_prevalence": decoy_prev,
    }


def empty_metrics() -> dict[str, float]:
    return {k: float("nan") for k in METRIC_COLUMNS}


def canonical_g(value: Any) -> float:
    x = float(value)
    matches = [g for g in REQUIRED_G if abs(x - g) <= 1e-9]
    if len(matches) != 1:
        raise ValueError(f"unregistered G_target={value!r}")
    return matches[0]


def record_value(rec: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in rec:
            return rec[name]
    raise KeyError(f"record missing all aliases {names}")


def base_record(dataset_dir: Path, rec: dict[str, Any], mode: str) -> dict[str, Any]:
    return {
        "dataset": str(rec.get("tag", dataset_dir.name)),
        "dataset_path": str(dataset_dir.resolve()),
        "mode": mode,
        "obs": str(rec["obs"]),
        "depth": rec.get("depth"),
        "topology": str(record_value(rec, "topo", "topology")),
        "G_target": canonical_g(record_value(rec, "G_target", "G")),
        "G_hidden": float(rec.get("G_E", rec.get("G_hidden", np.nan))),
        "grn": int(record_value(rec, "grn_seed", "grn")),
        "seed": int(record_value(rec, "sim_seed", "seed")),
    }


def discover(input_root: Path, confirm: bool) -> list[tuple[Path, dict[str, Any], dict[str, Any]]]:
    records = []
    seen = {}
    for path in sorted(input_root.glob("*/*/record.json")):
        rec = json.loads(path.read_text())
        if rec.get("world") != "P":
            raise SystemExit(f"REFUSING_NON_WORLD_P: {path} world={rec.get('world')}")
        q = int(record_value(rec, "q_req", "q"))
        if q != 3:
            raise SystemExit(f"REFUSING_NON_Q3: {path} q={q}")
        base = base_record(path.parent, rec, "CONFIRMATION" if confirm else "DEVELOPMENT")
        if base["obs"] not in OBSERVATIONS:
            raise SystemExit(f"REFUSING_OBSERVATION: {path} obs={base['obs']}")
        if path.parent.parent.name != base["obs"]:
            raise SystemExit(f"REFUSING_OBSERVATION_DIRECTORY_MISMATCH: {path} record={base['obs']} directory={path.parent.parent.name}")
        if base["obs"] == "UMI_DEPTH" and (base["depth"] is None or
                                             not np.isclose(float(base["depth"]), FROZEN_UMI_DEPTH, rtol=0, atol=1e-12)):
            raise SystemExit(f"REFUSING_UMI_DEPTH: {path} expected={FROZEN_UMI_DEPTH} observed={base['depth']}")
        if base["topology"] not in TOPOLOGIES:
            raise SystemExit(f"REFUSING_TOPOLOGY: {path} topology={base['topology']}")
        role = "CONFIRMATION" if base["grn"] in CONF_GRNS else "DEVELOPMENT" if base["grn"] in DEV_GRNS else "OTHER"
        # Development and confirmation outputs may share one root.  Each mode
        # reads only its authorized roster; no confirmation score file is
        # opened by development mode and no development score is pooled into
        # confirmation.
        if confirm and role == "DEVELOPMENT":
            continue
        if not confirm and role == "CONFIRMATION":
            continue
        if role == "OTHER":
            raise SystemExit(f"REFUSING_UNREGISTERED_GRN: {path} grn={base['grn']}")
        key = tuple(base[x] for x in ("obs", "topology", "G_target", "grn", "seed"))
        if key in seen:
            raise SystemExit(f"REFUSING_DUPLICATE_DATASET: key={key} paths={seen[key]} and {path}")
        seen[key] = str(path)
        records.append((path.parent, rec, base))
    return records


def load_generator():
    import gb_sergio_drift_v5 as producer

    return producer


def validate_axes(raw: Any, producer: Any, grn: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    generated = producer.generate_grn(grn)
    sources = np.asarray(raw["source_gene_index"], int)
    cols = np.asarray(raw["cols_PI"], int)
    expected_sources = np.asarray(generated["cands"], int)
    expected_cols = np.asarray([[t for t in range(producer.N_GENES) if t != s] for s in expected_sources], int)
    if not np.array_equal(sources, expected_sources):
        raise ValueError("source_gene_index does not equal generate_grn(grn)['cands']")
    if not np.array_equal(cols, expected_cols):
        raise ValueError("cols_PI is not the exact all-non-self observational target universe")
    for channel in CHANNELS:
        for root in ROOTS:
            key = f"TRUE|{channel}|{root}|absS"
            if key not in raw.files:
                raise ValueError(f"missing raw key {key}")
            if np.asarray(raw[key]).shape != (31, len(sources), producer.N_GENES):
                raise ValueError(f"wrong shape for {key}: {np.asarray(raw[key]).shape}")
    return sources, cols, generated


def gpu_status(dataset_dir: Path, method: str) -> str | None:
    path = dataset_dir / "metrics_gpu.tsv"
    if not path.exists():
        return None
    table = pd.read_csv(path, sep="\t")
    if "method" not in table or "status" not in table:
        raise ValueError(f"malformed GPU status table: {path}")
    hit = table[table["method"].astype(str).str.upper() == method.upper()]
    if "univ" in hit:
        hit = hit[hit["univ"].astype(str) == "PI"]
    statuses = set(hit["status"].astype(str))
    if "OK" in statuses:
        return "OK"
    return "NOT_SCORED_GPU_STATUS_" + ("+".join(sorted(statuses)) if statuses else "MISSING")


def roster_matrix(matrix: np.ndarray, cols: np.ndarray, n_genes: int) -> np.ndarray:
    matrix = np.asarray(matrix, float)
    if matrix.shape == cols.shape:
        return matrix
    if matrix.shape == (cols.shape[0], n_genes):
        return np.take_along_axis(matrix, cols, axis=1)
    raise ValueError(f"score matrix has shape {matrix.shape}; expected {cols.shape} or {(cols.shape[0], n_genes)}")


def find_velorama(dataset_dir: Path) -> list[Path]:
    canonical = dataset_dir / "raw" / "velorama.npz"
    if canonical.exists():
        return [canonical]
    paths = sorted((dataset_dir / "raw").glob("velorama*.npz"))
    if paths:
        return paths
    return sorted((dataset_dir / "velorama_v2").glob("*.npz"))


def load_external_score(dataset_dir: Path, method: str, cols: np.ndarray, n_genes: int) -> tuple[np.ndarray | None, str, str]:
    if method == "Velorama":
        paths = find_velorama(dataset_dir)
        status = gpu_status(dataset_dir, "VELORAMA")
    else:
        path = dataset_dir / "raw" / "regvelo.npz"
        paths = [path] if path.exists() else []
        status = gpu_status(dataset_dir, "REGVELO")
        diag = dataset_dir / "raw" / "regvelo.json"
        if diag.exists() and json.loads(diag.read_text()).get("OUTPUT_DEGENERATE"):
            return None, "NOT_SCORED_DEGENERATE", str(path)
    if status is not None and status != "OK":
        return None, status, ";".join(map(str, paths))
    if not paths:
        return None, "NOT_SCORED_MISSING", ""
    matrices = []
    for path in paths:
        with np.load(path, allow_pickle=False) as z:
            if "score" not in z.files:
                raise ValueError(f"persisted {method} artifact has no score key: {path}")
            matrices.append(roster_matrix(z["score"], cols, n_genes))
    ref = matrices[0]
    if any(not np.array_equal(ref, other, equal_nan=True) for other in matrices[1:]):
        raise ValueError(f"conflicting persisted {method} score artifacts: {paths}")
    if not np.isfinite(ref).any():
        return None, "NOT_SCORED_ALL_MISSING", ";".join(map(str, paths))
    return ref, "OK", ";".join(map(str, paths))


def not_scored_rows(base: dict[str, Any], reason: str, frozen_band: int | None) -> tuple[list[dict], list[dict], list[dict]]:
    sweep = []
    bands = []
    comps = []
    for channel in CHANNELS:
        for k in K_GRID:
            sweep.append({**base, "channel": channel, "k": k, "status": reason, **empty_metrics()})
        allowed = (frozen_band,) if frozen_band is not None else CANDIDATE_M
        bands.append({**base, "channel": channel, "band": "RAW", "m": 0, "status": reason, **empty_metrics()})
        for m in allowed:
            bands.append({**base, "channel": channel, "band": f"C_{m}", "m": m, "status": reason, **empty_metrics()})
    for method, source_key in COMPARATOR_KEYS + (("Velorama", "score"), ("RegVelo", "score")):
        comps.append({**base, "method": method, "source_key": source_key, "artifact": "", "status": reason, **empty_metrics()})
    return sweep, bands, comps


def evaluate_dataset(
    dataset_dir: Path, rec: dict[str, Any], base: dict[str, Any], producer: Any,
    frozen_band: int | None,
) -> tuple[list[dict], list[dict], list[dict]]:
    rec_status = str(rec.get("status", "MISSING_STATUS"))
    raw_path = dataset_dir / "raw" / "geobridge_raw.npz"
    if rec_status != "OK" or not raw_path.exists():
        reason = "NOT_SCORED_RECORD_" + rec_status.replace(" ", "_") if rec_status != "OK" else "NOT_SCORED_MISSING_RAW"
        return not_scored_rows(base, reason, frozen_band)
    try:
        raw = np.load(raw_path, allow_pickle=False)
        sources, cols, generated = validate_axes(raw, producer, base["grn"])
        direct = {(int(s), int(t)) for s, t in generated["gt"]}
        indirect = strict_two_hop(direct, producer.N_GENES)
        y, labels, true_rows = truth_arrays(sources, cols, direct, indirect, set(generated["regs"]))
        if int(true_rows.sum()) != 20 or y[true_rows].shape != (20, 99) or int(y[true_rows].sum()) != 193:
            raise ValueError(f"primary truth invariant failed: rows={true_rows.sum()} shape={y[true_rows].shape} edges={y[true_rows].sum()}")

        sweep, bands, comps = [], [], []
        allowed_bands = (frozen_band,) if frozen_band is not None else CANDIDATE_M
        for channel in CHANNELS:
            for k in K_GRID:
                score = single_depth(raw, cols, channel, k)
                sweep.append({**base, "channel": channel, "k": k, "status": "OK", **metrics(score, y, labels, true_rows)})
            raw_score = single_depth(raw, cols, channel, 0)
            bands.append({**base, "channel": channel, "band": "RAW", "m": 0, "status": "OK", **metrics(raw_score, y, labels, true_rows)})
            for m in allowed_bands:
                score = conditioned_band(raw, cols, channel, int(m))
                bands.append({**base, "channel": channel, "band": f"C_{m}", "m": int(m), "status": "OK", **metrics(score, y, labels, true_rows)})

        comparator_path = dataset_dir / "raw" / "comparators.npz"
        comparator = np.load(comparator_path, allow_pickle=False) if comparator_path.exists() else None
        for method, source_key in COMPARATOR_KEYS:
            if comparator is None or source_key not in comparator.files:
                comps.append({**base, "method": method, "source_key": source_key, "artifact": str(comparator_path),
                              "status": "NOT_SCORED_MISSING", **empty_metrics()})
                continue
            score = roster_matrix(comparator[source_key], cols, producer.N_GENES)
            status = "OK" if np.isfinite(score).any() else "NOT_SCORED_ALL_MISSING"
            vals = metrics(score, y, labels, true_rows) if status == "OK" else empty_metrics()
            comps.append({**base, "method": method, "source_key": source_key, "artifact": str(comparator_path),
                          "status": status, **vals})
        for method in ("Velorama", "RegVelo"):
            score, status, artifact = load_external_score(dataset_dir, method, cols, producer.N_GENES)
            vals = metrics(score, y, labels, true_rows) if status == "OK" and score is not None else empty_metrics()
            comps.append({**base, "method": method, "source_key": "score", "artifact": artifact,
                          "status": status, **vals})
        raw.close()
        if comparator is not None:
            comparator.close()
        return sweep, bands, comps
    except Exception as exc:
        raise RuntimeError(f"REFUSING_INVALID_DATASET_ARTIFACT: {dataset_dir}: {type(exc).__name__}: {exc}") from exc


def mean_technical(frame: pd.DataFrame, value_cols: list[str], extra: list[str] | None = None) -> pd.DataFrame:
    extra = extra or []
    keys = ["mode", "obs", "topology", "G_target", "grn"] + extra
    valid = frame[frame["status"] == "OK"].copy()
    if valid.empty:
        return pd.DataFrame(columns=keys + value_cols + ["G_hidden", "n_technical_seeds"])
    grouped = valid.groupby(keys, dropna=False, sort=True)
    out = grouped[value_cols + ["G_hidden"]].mean().reset_index()
    counts = grouped["seed"].nunique().rename("n_technical_seeds").reset_index()
    return out.merge(counts, on=keys, validate="one_to_one")


def observed_seed_sets(frame: pd.DataFrame, topology: str) -> dict[tuple[str, float, int], list[int]]:
    d = frame[(frame["topology"] == topology) & (frame["status"] == "OK")]
    out = {}
    for key, g in d.groupby(["obs", "G_target", "grn"]):
        out[(str(key[0]), float(key[1]), int(key[2]))] = sorted(set(map(int, g["seed"])))
    return out


def grid_completeness(sweep: pd.DataFrame, topology: str, confirm: bool) -> dict[str, Any]:
    base = sweep[(sweep["channel"] == PRIMARY_CHANNEL) & (sweep["k"] == 0)]
    seeds_by_cell = observed_seed_sets(base, topology)
    grns = CONF_GRNS if confirm else DEV_GRNS
    required = [(obs, g, grn) for obs in OBSERVATIONS for g in REQUIRED_G for grn in grns]
    missing = [dict(obs=o, G_target=g, grn=r) for o, g, r in required if (o, g, r) not in seeds_by_cell]
    if confirm:
        expected_seeds = list(CONF_SEEDS)
    else:
        expected_seeds = list(DEV_SEEDS)
    inconsistent = []
    for o, g, r in required:
        got = seeds_by_cell.get((o, g, r))
        if got is not None and got != expected_seeds:
            inconsistent.append(dict(obs=o, G_target=g, grn=r, expected=expected_seeds, observed=got))
    extra = [dict(obs=o, G_target=g, grn=r, seeds=s) for (o, g, r), s in seeds_by_cell.items()
             if (o, g, r) not in set(required)]
    complete = not missing and not inconsistent and not extra and bool(expected_seeds)
    return {
        "status": "COMPLETE" if complete else "PENDING",
        "complete": complete,
        "required_cells": len(required),
        "observed_cells": sum((o, g, r) in seeds_by_cell for o, g, r in required),
        "expected_seeds": expected_seeds,
        "missing_cells": missing,
        "inconsistent_seed_cells": inconsistent,
        "extra_cells": extra,
    }


def regret(oracle: float, band_ap: float, prevalence: float) -> tuple[float, float]:
    absolute = max(float(oracle) - float(band_ap), 0.0)
    relative = absolute / max(float(oracle) - float(prevalence), 0.02)
    return relative, absolute


def condition_curves(sweep: pd.DataFrame, channel: str = PRIMARY_CHANNEL) -> pd.DataFrame:
    return mean_technical(
        sweep[sweep["channel"] == channel],
        ["pooled_AP", "prevalence", "direct_vs_indirect_AUROC"],
        extra=["channel", "k"],
    )


def condition_bands(bands: pd.DataFrame, channel: str = PRIMARY_CHANNEL) -> pd.DataFrame:
    return mean_technical(
        bands[bands["channel"] == channel],
        ["pooled_AP", "prevalence", "direct_vs_indirect_AUROC"],
        extra=["channel", "band", "m"],
    )


def gate_for_cell(curves: pd.DataFrame, banded: pd.DataFrame, topology: str, obs: str, m: int) -> dict[str, Any]:
    c = curves[(curves["topology"] == topology) & (curves["obs"] == obs)]
    b = banded[(banded["topology"] == topology) & (banded["obs"] == obs) & (banded["m"] == m)]
    key = ["mode", "obs", "topology", "G_target", "grn", "channel"]
    wide = c.pivot_table(index=key, columns="k", values=["pooled_AP", "prevalence", "direct_vs_indirect_AUROC"], aggfunc="first")
    rows = []
    for r in b.itertuples(index=False):
        ix = tuple(getattr(r, x) for x in key)
        if ix not in wide.index:
            continue
        aps = np.asarray([wide.loc[ix, ("pooled_AP", k)] for k in K_GRID], float)
        prevs = np.asarray([wide.loc[ix, ("prevalence", k)] for k in K_GRID], float)
        d0 = float(wide.loc[ix, ("direct_vs_indirect_AUROC", 0)])
        if not np.isfinite(aps).all() or not np.isfinite(prevs).all():
            continue
        pi = float(prevs[0])
        oracle = max(pi, float(aps.max()))
        rel, absolute = regret(oracle, float(r.pooled_AP), pi)
        rows.append({
            "G_target": float(r.G_target), "grn": int(r.grn), "oracle_AP": oracle,
            "band_AP": float(r.pooled_AP), "k0_AP": float(aps[0]), "prevalence": pi,
            "regret": rel, "absolute_regret": absolute,
            "directness_delta": float(r.direct_vs_indirect_AUROC) - d0,
        })
    d = pd.DataFrame(rows)
    if d.empty:
        return {"status": "PENDING", "reason": "no complete condition metrics"}
    pos = d[d["G_target"].isin(SELECTION_G)]
    nodrift = d[d["G_target"] == 0.0]
    expected_n = len(DEV_GRNS) * (len(SELECTION_G) + 1)
    if len(pos) != len(DEV_GRNS) * len(SELECTION_G) or len(nodrift) != len(DEV_GRNS):
        return {"status": "PENDING", "reason": f"condition rows={len(d)} expected selection rows={expected_n}"}

    regret_cell_means = pos.groupby("G_target")["regret"].mean()
    direct_cell_means = pos.groupby("G_target")["directness_delta"].mean()
    g1_median = float(np.quantile(pos["regret"], 0.5, method="linear"))
    g1_p90 = float(np.quantile(pos["regret"], 0.9, method="linear"))
    g1 = bool(g1_median <= 0.15 and g1_p90 <= 0.30 and (regret_cell_means <= 0.30).all())
    g2_delta = float((nodrift["band_AP"] - nodrift["k0_AP"]).mean())
    g2 = bool(g2_delta >= -0.02)
    g3 = bool((direct_cell_means >= -0.02).all())
    return {
        "status": "PASS" if g1 and g2 and g3 else "FAIL",
        "G1": {
            "pass": g1, "median_regret": g1_median, "p90_regret": g1_p90,
            "mean_regret_by_drift": {f"{k:.2f}": float(v) for k, v in regret_cell_means.items()},
            "thresholds": {"median_max": 0.15, "p90_max": 0.30, "cell_mean_max": 0.30},
        },
        "G2": {"pass": g2, "mean_AP_band_minus_k0": g2_delta, "lower_threshold": -0.02},
        "G3": {
            "pass": g3,
            "mean_direct_vs_indirect_AUROC_delta_by_drift": {f"{k:.2f}": float(v) for k, v in direct_cell_means.items()},
            "lower_threshold": -0.02,
        },
        "regret_summary": {
            "mean_absolute_regret": float(pos["absolute_regret"].mean()),
            "max_absolute_regret": float(pos["absolute_regret"].max()),
            "n_conditions": int(len(pos)),
        },
        "condition_rows": rows,
    }


def select_admissible(gates: dict[str, Any], topologies: Iterable[str] = TOPOLOGIES) -> tuple[str, list[int], int | str]:
    topologies = tuple(topologies)
    if any(gates["completeness"][t]["status"] != "COMPLETE" for t in topologies):
        return "PENDING", [], "PENDING"
    admissible = []
    for m in CANDIDATE_M:
        if all(gates["by_topology"][t][str(m)][obs]["status"] == "PASS" for t in topologies for obs in OBSERVATIONS):
            admissible.append(m)
    return ("SELECTED", admissible, min(admissible)) if admissible else ("NO_BAND", [], "NO_BAND")


def development_gates(sweep: pd.DataFrame, bands: pd.DataFrame) -> dict[str, Any]:
    curves, banded = condition_curves(sweep), condition_bands(bands)
    out: dict[str, Any] = {
        "mode": "DEVELOPMENT", "primary_channel": PRIMARY_CHANNEL,
        "development_grns": list(DEV_GRNS), "required_sim_seeds": list(DEV_SEEDS),
        "candidate_m": list(CANDIDATE_M),
        "selection_drift": list(SELECTION_G), "stress_drift_excluded_from_selection": 0.75,
        "completeness": {}, "by_topology": {},
    }
    for topology in TOPOLOGIES:
        out["completeness"][topology] = grid_completeness(sweep, topology, confirm=False)
        out["by_topology"][topology] = {}
        for m in CANDIDATE_M:
            out["by_topology"][topology][str(m)] = {}
            for obs in OBSERVATIONS:
                if out["completeness"][topology]["complete"]:
                    out["by_topology"][topology][str(m)][obs] = gate_for_cell(curves, banded, topology, obs, m)
                else:
                    out["by_topology"][topology][str(m)][obs] = {"status": "PENDING", "reason": "topology grid incomplete"}
    status, admissible, chosen = select_admissible(out)
    out.update(selection_status=status, admissible_set=admissible, chosen_band=chosen)
    return out


def constraint_margins(gate: dict[str, Any], prefix: str) -> dict[str, float]:
    if gate.get("status") == "PENDING":
        return {}
    margins = {
        f"{prefix}|G1_median": 0.15 - float(gate["G1"]["median_regret"]),
        f"{prefix}|G1_p90": 0.30 - float(gate["G1"]["p90_regret"]),
        f"{prefix}|G2": float(gate["G2"]["mean_AP_band_minus_k0"]) + 0.02,
    }
    margins.update({f"{prefix}|G1_mean_G{g}": 0.30 - float(v) for g, v in gate["G1"]["mean_regret_by_drift"].items()})
    margins.update({f"{prefix}|G3_G{g}": float(v) + 0.02 for g, v in gate["G3"]["mean_direct_vs_indirect_AUROC_delta_by_drift"].items()})
    return margins


def resample_grns(frame: pd.DataFrame, sampled: np.ndarray) -> pd.DataFrame:
    chunks = []
    for draw, grn in enumerate(sampled):
        x = frame[frame["grn"] == int(grn)].copy()
        x["grn"] = draw  # duplicated clusters remain distinct bootstrap units
        chunks.append(x)
    return pd.concat(chunks, ignore_index=True)


def confirmation_gates(
    sweep: pd.DataFrame, bands: pd.DataFrame, frozen_band: int, reps: int, seed: int,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "mode": "CONFIRMATION", "primary_channel": PRIMARY_CHANNEL,
        "confirmation_grns": list(CONF_GRNS), "required_sim_seeds": list(CONF_SEEDS),
        "frozen_band": frozen_band, "bootstrap_reps": reps, "bootstrap_seed": seed,
        "bootstrap_unit": "whole GRN with all topology, observation, drift and technical-seed conditions attached",
        "simultaneous_bound": "one-sided 95% max centered bootstrap error over every registered gate constraint",
        "completeness": {}, "by_topology": {},
    }
    for topology in TOPOLOGIES:
        out["completeness"][topology] = grid_completeness(sweep, topology, confirm=True)
    if not all(v["complete"] for v in out["completeness"].values()):
        out.update(confirmation_status="PENDING", reason="full confirmation grid incomplete")
        return out

    curves, banded = condition_curves(sweep), condition_bands(bands)
    point_margins = {}
    for topology in TOPOLOGIES:
        out["by_topology"][topology] = {str(frozen_band): {}}
        for obs in OBSERVATIONS:
            g = gate_for_cell(curves, banded, topology, obs, frozen_band)
            out["by_topology"][topology][str(frozen_band)][obs] = g
            point_margins.update(constraint_margins(g, f"{topology}|{obs}"))
    if not point_margins or not all(np.isfinite(list(point_margins.values()))):
        out.update(confirmation_status="UNCONFIRMED", reason="non-finite point constraint")
        return out

    rng = np.random.Generator(np.random.PCG64(seed))
    max_errors = []
    for _ in range(reps):
        sampled = rng.choice(CONF_GRNS, size=len(CONF_GRNS), replace=True)
        cs = resample_grns(sweep, sampled)
        bs = resample_grns(bands, sampled)
        cc, bb = condition_curves(cs), condition_bands(bs)
        boot = {}
        for topology in TOPOLOGIES:
            for obs in OBSERVATIONS:
                g = gate_for_cell(cc, bb, topology, obs, frozen_band)
                boot.update(constraint_margins(g, f"{topology}|{obs}"))
        if set(boot) != set(point_margins) or not all(np.isfinite(list(boot.values()))):
            raise RuntimeError("confirmation cluster bootstrap produced incomplete/non-finite gate margins")
        max_errors.append(max(point_margins[k] - boot[k] for k in point_margins))
    critical = max(0.0, float(np.quantile(max_errors, 0.95, method="higher")))
    bounds = {k: {"point_margin": float(v), "simultaneous_lower_95": float(v - critical)} for k, v in point_margins.items()}
    confirmed = all(v["simultaneous_lower_95"] >= 0 for v in bounds.values())
    out.update(
        point_constraint_margins=point_margins,
        simultaneous_critical_value=critical,
        simultaneous_bounds=bounds,
        confirmation_status="CONFIRMED" if confirmed else "UNCONFIRMED",
    )
    return out


def spearman(x: Iterable[float], y: Iterable[float]) -> dict[str, Any]:
    from scipy.stats import spearmanr

    x, y = np.asarray(list(x), float), np.asarray(list(y), float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3 or np.unique(x[ok]).size < 2 or np.unique(y[ok]).size < 2:
        return {"rho": None, "pvalue_descriptive": None, "n": int(ok.sum()), "status": "UNDEFINED"}
    result = spearmanr(x[ok], y[ok])
    return {"rho": float(result.statistic), "pvalue_descriptive": float(result.pvalue), "n": int(ok.sum()), "status": "OK"}


def useful_depth(aps: np.ndarray, prevalence: float, fraction: float) -> tuple[list[int], float]:
    aps = np.asarray(aps, float)
    if not np.isfinite(aps).all() or not np.isfinite(prevalence):
        return [], float("nan")
    excess = float(aps.max() - prevalence)
    if excess <= 0:
        return [], float("nan")
    threshold = prevalence + fraction * excess
    ks = np.flatnonzero(aps >= threshold - 1e-15).astype(int).tolist()
    return ks, float(np.median(ks)) if ks else float("nan")


def mechanism(sweep: pd.DataFrame) -> dict[str, Any]:
    all_conditions = []
    h1, h2_drift = {}, {}
    for channel in CHANNELS:
        curves = condition_curves(sweep, channel)
        if curves.empty:
            continue
        index = ["mode", "obs", "topology", "G_target", "grn", "channel"]
        for key, group in curves.groupby(index, sort=True):
            group = group.set_index("k").reindex(K_GRID)
            aps = group["pooled_AP"].to_numpy(float)
            if not np.isfinite(aps).all():
                continue
            prevalence = float(group["prevalence"].iloc[0])
            k90, c90 = useful_depth(aps, prevalence, 0.90)
            k95, c95 = useful_depth(aps, prevalence, 0.95)
            oracle_k = int(np.flatnonzero(aps == np.nanmax(aps))[0])
            row = {
                "mode": key[0], "obs": key[1], "topology": key[2], "G_target": float(key[3]),
                "grn": int(key[4]), "channel": key[5], "G_hidden": float(group["G_hidden"].mean()),
                "prevalence": prevalence, "benefit_max_minus_k0": float(max(aps.max() - aps[0], 0.0)),
                "oracle_k": oracle_k, "K90": k90, "K90_centre": c90,
                "K95": k95, "K95_centre": c95,
            }
            all_conditions.append(row)
        d = pd.DataFrame([x for x in all_conditions if x["channel"] == channel])
        if d.empty:
            continue
        for (obs, topology), g in d.groupby(["obs", "topology"]):
            name = f"{channel}|{obs}|{topology}"
            h1[name] = spearman(g["G_hidden"], g["benefit_max_minus_k0"])
            h2_drift[name] = {
                "oracle_k_vs_G_hidden": spearman(g["G_hidden"], g["oracle_k"]),
                "K90_centre_vs_G_hidden": spearman(g["G_hidden"], g["K90_centre"]),
                "K95_centre_vs_G_hidden": spearman(g["G_hidden"], g["K95_centre"]),
            }
    d = pd.DataFrame(all_conditions)
    h2_complexity = {}
    if not d.empty:
        d["topology_complexity"] = d["topology"].map(TOPOLOGY_COMPLEXITY)
        for (channel, obs), g in d.groupby(["channel", "obs"]):
            name = f"{channel}|{obs}"
            h2_complexity[name] = {
                "oracle_k_vs_topology_complexity": spearman(g["topology_complexity"], g["oracle_k"]),
                "K90_centre_vs_topology_complexity": spearman(g["topology_complexity"], g["K90_centre"]),
                "K95_centre_vs_topology_complexity": spearman(g["topology_complexity"], g["K95_centre"]),
            }
    return {
        "diagnostic_only": True,
        "K90_K95_definition": "k values attaining q of excess AP above prevalence; centre is their median",
        "H1_benefit_vs_G_hidden_per_topology": h1,
        "H2_depth_vs_G_hidden_per_topology": h2_drift,
        "H2_depth_vs_topology_complexity": h2_complexity,
        "condition_diagnostics": all_conditions,
    }


def aggregate_plot_rows(bands: pd.DataFrame, comps: pd.DataFrame, topology: str, channel: str = PRIMARY_CHANNEL) -> pd.DataFrame:
    b = bands[(bands["topology"] == topology) & (bands["channel"] == channel) & (bands["status"] == "OK")].copy()
    b["method"] = b["band"]
    c = comps[(comps["topology"] == topology) & (comps["status"] == "OK")].copy()
    columns = ["obs", "G_target", "G_hidden", "grn", "seed", "method", "pooled_AP", "AP_over_prevalence", "AUROC",
               "direct_vs_indirect_AUROC", "indirect_at_50", "macro_AP", "decoy_universe_AP"]
    d = pd.concat([b[columns], c[columns]], ignore_index=True)
    if d.empty:
        return d
    tech = d.groupby(["obs", "G_target", "grn", "method"], sort=True).agg(
        G_hidden=("G_hidden", "mean"),
        pooled_AP=("pooled_AP", "mean"), AP_over_prevalence=("AP_over_prevalence", "mean"),
        AUROC=("AUROC", "mean"), direct_vs_indirect_AUROC=("direct_vs_indirect_AUROC", "mean"),
        indirect_at_50=("indirect_at_50", "mean"), macro_AP=("macro_AP", "mean"),
        decoy_universe_AP=("decoy_universe_AP", "mean"),
    ).reset_index()
    return tech.groupby(["obs", "G_target", "method"], sort=True).mean(numeric_only=True).reset_index()


def plot_outputs(output: Path, sweep: pd.DataFrame, bands: pd.DataFrame, comps: pd.DataFrame, gates: dict[str, Any]) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    made = []
    chosen = gates.get("chosen_band") if gates.get("mode") == "DEVELOPMENT" else gates.get("frozen_band")
    for topology in TOPOLOGIES:
        d = aggregate_plot_rows(bands, comps, topology)
        if d.empty:
            continue
        # Overview: required primary/guard metrics versus achieved G_hidden.
        metrics_to_plot = [
            ("pooled_AP", "Pooled AP"), ("AP_over_prevalence", "AP / prevalence"),
            ("AUROC", "AUROC"), ("macro_AP", "Macro AP"),
            ("decoy_universe_AP", "Decoy-universe AP"), ("indirect_at_50", "Strict indirect@50"),
        ]
        fig, axes = plt.subplots(len(OBSERVATIONS), len(metrics_to_plot), figsize=(24, 8.5), squeeze=False)
        for oi, obs in enumerate(OBSERVATIONS):
            sub = d[d["obs"] == obs]
            for ji, (metric_name, label) in enumerate(metrics_to_plot):
                ax = axes[oi, ji]
                for method, g in sub.groupby("method"):
                    g = g.sort_values("G_hidden")
                    highlight = method == f"C_{chosen}" if isinstance(chosen, int) else False
                    ax.plot(g["G_hidden"], g[metric_name], marker="o", linewidth=2.5 if highlight else 1,
                            alpha=1 if highlight or method == "RAW" else 0.65, label=method)
                ax.set(xlabel="achieved G_hidden", ylabel=label, title=f"{obs}: {label}")
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="center right", fontsize=7)
        fig.suptitle(f"{topology}: World P metrics (GeoBRIDGE {PRIMARY_CHANNEL})")
        fig.tight_layout(rect=(0, 0, 0.88, 0.96))
        path = output / f"METRICS_{topology}.png"
        fig.savefig(path, dpi=150); plt.close(fig); made.append(str(path))

        # AP advantage versus the best scored comparator in each dataset condition.
        fig, axes = plt.subplots(1, len(OBSERVATIONS), figsize=(12, 4.5), squeeze=False)
        comp_names = {x[0] for x in COMPARATOR_KEYS} | {"Velorama", "RegVelo"}
        for oi, obs in enumerate(OBSERVATIONS):
            ax = axes[0, oi]
            sub = d[d["obs"] == obs]
            cp = sub[sub["method"].isin(comp_names)]
            best = cp.groupby("G_target")["pooled_AP"].max()
            for method, g in sub[sub["method"].str.startswith("C_")].groupby("method"):
                g = g.sort_values("G_target").copy()
                g["advantage"] = [row.pooled_AP - best.get(row.G_target, np.nan) for row in g.itertuples()]
                highlight = method == f"C_{chosen}" if isinstance(chosen, int) else False
                ax.plot(g["G_hidden"], g["advantage"], marker="o", linewidth=2.5 if highlight else 1, label=method)
            ax.axhline(0, color="black", linewidth=.8)
            ax.set(xlabel="achieved G_hidden", ylabel="AP advantage", title=f"{obs}: GB band - best comparator")
            ax.legend(fontsize=7)
        fig.suptitle(f"{topology}: comparator advantage")
        fig.tight_layout()
        path = output / f"ADVANTAGE_{topology}.png"
        fig.savefig(path, dpi=150); plt.close(fig); made.append(str(path))

        # Oracle and K90 are diagnostic only.
        mech = mechanism(sweep[(sweep["topology"] == topology) & (sweep["channel"] == PRIMARY_CHANNEL)])
        md = pd.DataFrame(mech["condition_diagnostics"])
        if not md.empty:
            fig, axes = plt.subplots(1, len(OBSERVATIONS), figsize=(12, 4.5), squeeze=False)
            for oi, obs in enumerate(OBSERVATIONS):
                ax = axes[0, oi]
                sub = md[md["obs"] == obs]
                for field, label in (("oracle_k", "oracle k"), ("K90_centre", "K90 centre")):
                    g = sub.groupby("G_target").agg(G_hidden=("G_hidden", "mean"), value=(field, "mean")).reset_index()
                    ax.plot(g["G_hidden"], g["value"], marker="o", label=label)
                ax.set(xlabel="achieved G_hidden", ylabel="depth", title=f"{obs} (DIAGNOSTIC ONLY)")
                ax.legend(fontsize=8)
            fig.suptitle(f"{topology}: oracle/K90 depth diagnostics")
            fig.tight_layout()
            path = output / f"DEPTH_DIAGNOSTIC_{topology}.png"
            fig.savefig(path, dpi=150); plt.close(fig); made.append(str(path))

        fig, axes = plt.subplots(1, len(OBSERVATIONS), figsize=(12, 4.5), squeeze=False)
        for oi, obs in enumerate(OBSERVATIONS):
            ax = axes[0, oi]
            sub = d[d["obs"] == obs]
            for method, g in sub.groupby("method"):
                g = g.sort_values("G_hidden")
                highlight = method == f"C_{chosen}" if isinstance(chosen, int) else False
                ax.plot(g["G_hidden"], g["direct_vs_indirect_AUROC"], marker="o",
                        linewidth=2.5 if highlight else 1, alpha=1 if highlight or method == "RAW" else .65, label=method)
            ax.axhline(.5, color="black", linewidth=.8, linestyle=":")
            ax.set(xlabel="achieved G_hidden", ylabel="strict D-vs-I AUROC", title=obs)
        fig.suptitle(f"{topology}: directness versus drift")
        fig.tight_layout()
        path = output / f"DIRECTNESS_{topology}.png"
        fig.savefig(path, dpi=150); plt.close(fig); made.append(str(path))
    return made


def selftest() -> dict[str, str]:
    # C_m: average ties within the fixed 4-target universe, mean k=1..3, then min roots.
    class ToyRaw:
        def __init__(self, data):
            self.data = data
            self.files = list(data)
        def __getitem__(self, key):
            return self.data[key]

    cols = np.array([[0, 1, 2, 3]])
    amin = np.zeros((31, 1, 4)); amax = np.zeros((31, 1, 4))
    amin[1, 0] = [1, 2, 2, 4]; amin[2, 0] = [4, 3, 2, 1]; amin[3, 0] = [1, 1, 3, 4]
    amax[1:4, 0] = [4, 3, 2, 1]
    raw = ToyRaw({"TRUE|U|dc1_min|absS": amin, "TRUE|U|dc1_max|absS": amax})
    got = conditioned_band(raw, cols, "U", 3)
    expected_min = np.mean(np.array([[.25, .625, .625, 1], [1, .75, .5, .25], [.375, .375, .75, 1]]), axis=0)
    expected_max = np.array([1, .75, .5, .25])
    assert np.allclose(got[0], np.minimum(expected_min, expected_max))
    amin[2, 0, 1] = np.nan
    got_missing = conditioned_band(raw, cols, "U", 3)
    assert np.isnan(got_missing[0, 1]) and np.isfinite(got_missing[0, [0, 2, 3]]).all()

    # Regret floor and positive part.
    assert np.allclose(regret(.11, .10, .10), (.5, .01), rtol=0, atol=1e-14)
    assert np.allclose(regret(.09, .10, .10), (0.0, 0.0), rtol=0, atol=1e-14)

    # Strict two-hop: direct shortcuts are excluded and length-three-only reachability is none.
    direct = {(0, 1), (1, 2), (0, 2), (2, 3), (4, 5), (5, 6)}
    indirect = strict_two_hop(direct, 7)
    assert (0, 2) not in indirect and (0, 3) in indirect and (1, 3) in indirect
    assert (4, 6) in indirect and (4, 7) not in indirect

    # The actual G1/G2/G3 calculator passes a within-tolerance fixture and fails
    # when regret, no-drift AP, and directness all cross their registered bounds.
    curve_rows, band_rows = [], []
    for grn in DEV_GRNS:
        for g in (0.0,) + SELECTION_G:
            for k in K_GRID:
                curve_rows.append(dict(mode="DEVELOPMENT", obs="UMI_DEPTH", topology="linear", G_target=g,
                                       grn=grn, channel="U", k=k, pooled_AP=.30 if k == 5 else .20,
                                       prevalence=.10, direct_vs_indirect_AUROC=.70, G_hidden=g))
            band_rows.append(dict(mode="DEVELOPMENT", obs="UMI_DEPTH", topology="linear", G_target=g,
                                  grn=grn, channel="U", band="C_3", m=3,
                                  pooled_AP=.19 if g == 0 else .28, prevalence=.10,
                                  direct_vs_indirect_AUROC=.69, G_hidden=g))
    toy_curves, toy_bands = pd.DataFrame(curve_rows), pd.DataFrame(band_rows)
    passed = gate_for_cell(toy_curves, toy_bands, "linear", "UMI_DEPTH", 3)
    assert passed["status"] == "PASS" and passed["G1"]["pass"] and passed["G2"]["pass"] and passed["G3"]["pass"]
    toy_bands.loc[toy_bands.G_target > 0, "pooled_AP"] = .20
    toy_bands.loc[toy_bands.G_target == 0, "pooled_AP"] = .17
    toy_bands.loc[:, "direct_vs_indirect_AUROC"] = .60
    failed = gate_for_cell(toy_curves, toy_bands, "linear", "UMI_DEPTH", 3)
    assert failed["status"] == "FAIL" and not failed["G1"]["pass"] and not failed["G2"]["pass"] and not failed["G3"]["pass"]

    # Selection requires both observation arms.  Passing UMI_DEPTH alone is not admissible.
    def fixture(status="PASS"):
        return {obs: {"status": status} for obs in OBSERVATIONS}
    gates = {
        "completeness": {t: {"status": "COMPLETE"} for t in TOPOLOGIES},
        "by_topology": {t: {str(m): fixture("FAIL") for m in CANDIDATE_M} for t in TOPOLOGIES},
    }
    for t in TOPOLOGIES:
        gates["by_topology"][t]["4"] = fixture("PASS")
        gates["by_topology"][t]["5"] = fixture("PASS")
    gates["by_topology"]["tree"]["4"]["CONC"] = {"status": "FAIL"}
    status, admissible, chosen = select_admissible(gates)
    assert status == "SELECTED" and admissible == [5] and chosen == 5
    for t in TOPOLOGIES:
        for m in CANDIDATE_M:
            gates["by_topology"][t][str(m)] = fixture("FAIL")
    status, admissible, chosen = select_admissible(gates)
    assert status == "NO_BAND" and admissible == [] and chosen == "NO_BAND"
    gates["completeness"]["linear"] = {"status": "PENDING"}
    assert select_admissible(gates)[0] == "PENDING"

    return {
        "C_m_formula_and_missingness": "PASS",
        "regret_floor_and_positive_part": "PASS",
        "strict_two_hop_labels": "PASS",
        "G1_G2_G3_threshold_logic": "PASS",
        "both_observations_NO_BAND_and_PENDING": "PASS",
    }


def evaluate(args: argparse.Namespace) -> None:
    assert_prereg()
    tests = selftest()
    if args.confirm and args.frozen_band is None:
        raise SystemExit("REFUSING_CONFIRMATION_WITHOUT_--frozen-band")
    if not args.confirm and args.frozen_band is not None:
        raise SystemExit("REFUSING_--frozen-band_OUTSIDE_--confirm")
    if args.frozen_band is not None and args.frozen_band not in CANDIDATE_M:
        raise SystemExit(f"REFUSING_UNREGISTERED_FROZEN_BAND: {args.frozen_band}")
    if args.bootstrap_reps < 1:
        raise SystemExit("REFUSING_NONPOSITIVE_--bootstrap-reps")
    input_root, output = Path(args.input).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    producer = load_generator()
    records = discover(input_root, args.confirm)
    sweep_rows, band_rows, comparator_rows = [], [], []
    for dataset_dir, rec, base in records:
        sw, ba, co = evaluate_dataset(dataset_dir, rec, base, producer, args.frozen_band)
        sweep_rows.extend(sw); band_rows.extend(ba); comparator_rows.extend(co)
    sweep = pd.DataFrame(sweep_rows, columns=SWEEP_COLUMNS)
    bands = pd.DataFrame(band_rows, columns=BAND_COLUMNS)
    comparators = pd.DataFrame(comparator_rows, columns=COMPARATOR_COLUMNS)
    atomic_tsv(output / "SWEEP.tsv", sweep, SWEEP_COLUMNS)
    atomic_tsv(output / "BANDS.tsv", bands, BAND_COLUMNS)
    atomic_tsv(output / "COMPARATORS.tsv", comparators, COMPARATOR_COLUMNS)
    if args.confirm:
        gates = confirmation_gates(sweep, bands, args.frozen_band, args.bootstrap_reps, args.bootstrap_seed)
    else:
        gates = development_gates(sweep, bands)
    gates["provenance"] = {
        "prereg_path": str(PREREG), "prereg_sha256": PREREG_SHA256,
        "evaluator_path": str(Path(__file__).resolve()), "evaluator_sha256": sha256_file(Path(__file__).resolve()),
        "producer_path": str(Path(producer.__file__).resolve()), "producer_sha256": sha256_file(Path(producer.__file__).resolve()),
        "input_root": str(input_root), "n_discovered_datasets": len(records), "selftests": tests,
    }
    atomic_json(output / "GATES.json", gates)
    mech = mechanism(sweep)
    mech["provenance"] = gates["provenance"]
    atomic_json(output / "MECHANISM.json", mech)
    plots = plot_outputs(output, sweep, bands, comparators, gates)
    print(json.dumps({
        "status": "OK", "mode": "CONFIRMATION" if args.confirm else "DEVELOPMENT",
        "datasets": len(records), "selection_status": gates.get("selection_status", gates.get("confirmation_status")),
        "chosen_or_frozen_band": gates.get("chosen_band", gates.get("frozen_band")),
        "plots": len(plots), "selftests": tests,
    }, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("selftest", help="run deterministic toy-array tests only")
    ev = sub.add_parser("evaluate", help="evaluate available World-P run_one artifacts")
    ev.add_argument("--input", default=str(REPO / "results" / "sergio_drift" / "core_P"))
    ev.add_argument("--output", default=str(REPO / "results" / "sergio_band"))
    ev.add_argument("--confirm", action="store_true", help="open confirmation GRNs 100..109 using a frozen development band")
    ev.add_argument("--frozen-band", type=int, choices=CANDIDATE_M)
    ev.add_argument("--bootstrap-reps", type=int, default=10000)
    ev.add_argument("--bootstrap-seed", type=int, default=20260925)
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    if ARGS.command == "selftest":
        assert_prereg()
        print(json.dumps({"prereg_sha256": PREREG_SHA256, "selftests": selftest()}, sort_keys=True))
    else:
        evaluate(ARGS)
