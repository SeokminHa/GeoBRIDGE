from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass

import numpy as np

try:
    from threadpoolctl import threadpool_limits as _threadpool_limits
except ImportError:  # pragma: no cover

    def _threadpool_limits(limits=1, user_api=None):
        return nullcontext()

@dataclass
class PairData:
    delta_x: np.ndarray
    delta_v: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray


def _maybe_pair_normalize_rows(dx, dv, pair_normalize):
    """Scale each pair row so ‖ΔX‖=1 (ΔV scaled the same). ``pair_normalize`` None = no-op."""
    if pair_normalize is None:
        return dx, dv
    pn = str(pair_normalize).strip().lower()
    if pn in ("", "none"):
        return dx, dv
    if pn == "l2":
        nrm = np.linalg.norm(dx, axis=1, keepdims=True) + 1e-12
        return dx / nrm, dv / nrm
    raise ValueError(f"pair_normalize must be None, 'none', or 'l2'; got {pair_normalize!r}")


def monotonic_forward_windows(pseudotime, min_forward_dt, max_forward_dt):
    """For non-decreasing pseudotime, return (start, end) such that valid j for source i are start[i] <= j
< end[i] (half-open), matching iter_forward_pair_blocks."""
    pt = np.asarray(pseudotime, dtype=np.float64).reshape(-1)
    n = int(pt.shape[0])
    thresh_lo = pt + float(min_forward_dt)
    start = np.searchsorted(pt, thresh_lo, side="right")
    start = np.maximum(start, np.arange(n, dtype=np.int64) + 1)
    if max_forward_dt is None:
        end = np.full(n, n, dtype=np.int64)
    else:
        thresh_hi = pt + float(max_forward_dt)
        end = np.searchsorted(pt, thresh_hi, side="right")
        end = np.maximum(end, start)
    return start, end


def _accumulate_monotone_range(
    i_begin,
    i_end,
    X,
    V,
    start,
    end,
    max_pairs_per_cell,
    pair_seed,
    pair_normalize=None,
):
    """Accumulate sufficient statistics for i in [i_begin, i_end) (half-open)."""
    n = int(X.shape[0])
    G = int(X.shape[1])
    sum_dx = np.zeros(G, dtype=np.float64)
    sum_dv = np.zeros(G, dtype=np.float64)
    sum_dx_dx = np.zeros((G, G), dtype=np.float64)
    sum_dx_dv = np.zeros((G, G), dtype=np.float64)
    sum_dv_dv = np.zeros((G, G), dtype=np.float64)
    n_pairs = 0
    rng = np.random.default_rng(int(pair_seed)) if pair_seed is not None else np.random
    cap = None if max_pairs_per_cell is None else int(max_pairs_per_cell)
    i_end = min(i_end, n - 1)
    for i in range(int(i_begin), i_end):
        s = int(start[i])
        e = int(end[i])
        if s >= e:
            continue
        js = np.arange(s, e, dtype=np.int64)
        if cap is not None and js.size > cap:
            js = rng.choice(js, size=cap, replace=False)
        dx = X[js] - X[i]
        dv = V[js] - V[i]
        dx, dv = _maybe_pair_normalize_rows(dx, dv, pair_normalize)
        m = int(dx.shape[0])
        if m <= 0:
            continue
        n_pairs += m
        sum_dx += dx.sum(axis=0)
        sum_dv += dv.sum(axis=0)
        sum_dx_dx += dx.T @ dx
        sum_dx_dv += dx.T @ dv
        sum_dv_dv += dv.T @ dv
    return sum_dx, sum_dv, sum_dx_dx, sum_dx_dv, sum_dv_dv, n_pairs


def accumulate_streaming_pairspace_stats(
    X,
    V,
    pseudotime,
    min_forward_dt=1e-4,
    max_forward_dt=None,
    max_pairs_per_cell=None,
    pair_seed=None,
    num_threads=None,
    pair_normalize=None,
):
    """
    Sum ΔX/ΔV moments over forward pairs (same statistics as materialized pair matrices).

    When pseudotime is non-decreasing, max_pairs_per_cell is None, and n_cells is large enough,
    splits source cells across a thread pool. Requires threadpoolctl (recommended) so each
    worker uses single-threaded BLAS; otherwise nested parallelism can slow this down.

    num_threads: None = auto worker count; 1 = single-threaded; >1 = cap workers.
    Env GEOBRIDGE_STREAM_THREADS overrides when num_threads is None (same semantics).

    pair_normalize: None / \"none\" = raw pair deltas; \"l2\" = row-wise ‖ΔX‖ normalization
    (ΔV scaled by the same factor per pair).
    """
    X = np.ascontiguousarray(X, dtype=np.float64)
    V = np.ascontiguousarray(V, dtype=np.float64)
    pt = np.asarray(pseudotime, dtype=np.float64).reshape(-1)
    n, G = X.shape
    if V.shape != (n, G):
        raise ValueError("V must have same shape as X.")
    if pt.shape[0] != n:
        raise ValueError("pseudotime must have length n_cells (X.shape[0]).")
    if n <= 1:
        raise ValueError("Need at least two cells.")
    if max_pairs_per_cell is not None and int(max_pairs_per_cell) <= 0:
        raise ValueError("max_pairs_per_cell must be positive when set.")

    monotonic = bool(np.all(pt[1:] >= pt[:-1]))
    cap = None if max_pairs_per_cell is None else int(max_pairs_per_cell)

    if num_threads is None:
        raw = os.environ.get("GEOBRIDGE_STREAM_THREADS", "").strip()
        if raw:
            try:
                num_threads = int(raw)
            except ValueError:
                pass

    if monotonic:
        start, end = monotonic_forward_windows(pt, min_forward_dt, max_forward_dt)
        use_parallel = cap is None and (n - 1) >= 128
        if num_threads is not None and int(num_threads) <= 1:
            use_parallel = False
        if use_parallel:
            cpu = int(os.cpu_count() or 4)
            nw = max(1, min(cpu, 16, (n - 1) // 64 + 1))
            if num_threads is not None and int(num_threads) > 1:
                nw = max(1, min(int(num_threads), n - 1))
            if nw <= 1:
                return _accumulate_monotone_range(
                    0,
                    n - 1,
                    X,
                    V,
                    start,
                    end,
                    max_pairs_per_cell,
                    pair_seed,
                    pair_normalize,
                )
            edges = np.linspace(0, n - 1, nw + 1, dtype=np.int64)
            chunks = [(int(edges[w]), int(edges[w + 1])) for w in range(nw)]

            def _run_chunk(ab):
                a, b = ab
                return _accumulate_monotone_range(
                    a, b, X, V, start, end, max_pairs_per_cell, pair_seed, pair_normalize
                )

            # Avoid oversubscription: each chunk uses blas gemm
            with _threadpool_limits(limits=1):
                with ThreadPoolExecutor(max_workers=nw) as ex:
                    parts = list(ex.map(_run_chunk, chunks))
            sum_dx = np.zeros(G, dtype=np.float64)
            sum_dv = np.zeros(G, dtype=np.float64)
            sum_dx_dx = np.zeros((G, G), dtype=np.float64)
            sum_dx_dv = np.zeros((G, G), dtype=np.float64)
            sum_dv_dv = np.zeros((G, G), dtype=np.float64)
            n_pairs = 0
            for p in parts:
                sum_dx += p[0]
                sum_dv += p[1]
                sum_dx_dx += p[2]
                sum_dx_dv += p[3]
                sum_dv_dv += p[4]
                n_pairs += p[5]
            return sum_dx, sum_dv, sum_dx_dx, sum_dx_dv, sum_dv_dv, n_pairs
        return _accumulate_monotone_range(
            0, n - 1, X, V, start, end, max_pairs_per_cell, pair_seed, pair_normalize
        )

    # Non-monotonic: same logic as iter_forward_pair_blocks fallback
    rng = np.random.default_rng(int(pair_seed)) if pair_seed is not None else np.random
    sum_dx = np.zeros(G, dtype=np.float64)
    sum_dv = np.zeros(G, dtype=np.float64)
    sum_dx_dx = np.zeros((G, G), dtype=np.float64)
    sum_dx_dv = np.zeros((G, G), dtype=np.float64)
    sum_dv_dv = np.zeros((G, G), dtype=np.float64)
    n_pairs = 0
    for i in range(n - 1):
        dt = pt[i + 1 :] - pt[i]
        valid = np.where(dt > min_forward_dt)[0]
        if max_forward_dt is not None:
            valid = valid[dt[valid] <= float(max_forward_dt)]
        if valid.size == 0:
            continue
        js = (i + 1 + valid).astype(np.int64)
        if cap is not None and js.size > cap:
            js = rng.choice(js, size=cap, replace=False)
        dx = X[js] - X[i]
        dv = V[js] - V[i]
        dx, dv = _maybe_pair_normalize_rows(dx, dv, pair_normalize)
        m = int(dx.shape[0])
        if m <= 0:
            continue
        n_pairs += m
        sum_dx += dx.sum(axis=0)
        sum_dv += dv.sum(axis=0)
        sum_dx_dx += dx.T @ dx
        sum_dx_dv += dx.T @ dv
        sum_dv_dv += dv.T @ dv
    return sum_dx, sum_dv, sum_dx_dx, sum_dx_dv, sum_dv_dv, n_pairs


def iter_forward_pair_blocks(
    pseudotime,
    min_forward_dt=1e-4,
    max_forward_dt=None,
    max_pairs_per_cell=None,
    pair_seed=None,
):
    """Yield blocks of admissible forward pairs as (i, js) where js is an int array of j indices."""
    pt = np.asarray(pseudotime).reshape(-1)
    n = int(pt.shape[0])
    if n <= 1:
        return

    if max_pairs_per_cell is not None and int(max_pairs_per_cell) <= 0:
        return
    rng = np.random.default_rng(int(pair_seed)) if pair_seed is not None else np.random

    if np.all(pt[1:] >= pt[:-1]):
        start, end = monotonic_forward_windows(pt, min_forward_dt, max_forward_dt)
        cap = None if max_pairs_per_cell is None else int(max_pairs_per_cell)
        for i in range(n - 1):
            s = int(start[i]); e = int(end[i])
            if s >= e:
                continue
            js = np.arange(s, e, dtype=int)
            if cap is not None and js.size > cap:
                js = rng.choice(js, size=cap, replace=False)
            yield i, js
        return

    # Fallback: correct for arbitrary order (still yields blocks).
    cap = None if max_pairs_per_cell is None else int(max_pairs_per_cell)
    for i in range(n - 1):
        dt = pt[i + 1 :] - pt[i]
        valid = np.where(dt > min_forward_dt)[0]
        if max_forward_dt is not None:
            valid = valid[dt[valid] <= float(max_forward_dt)]
        if valid.size == 0:
            continue
        js = (i + 1 + valid).astype(int)
        if cap is not None and js.size > cap:
            js = rng.choice(js, size=cap, replace=False)
        yield i, js

def build_forward_pairs(
    X,
    V,
    pseudotime,
    min_forward_dt=1e-4,
    max_forward_dt=None,
    max_pairs_per_cell=None,
    pair_seed=None,
):
    """Build all admissible forward (i,j) pairs with i < j and pseudotime[j]-pseudotime[i] >
min_forward_dt."""
    n = int(X.shape[0])
    pt = np.asarray(pseudotime).reshape(-1)
    if pt.shape[0] != n:
        raise ValueError("pseudotime must have length n_cells (X.shape[0]).")

    ii, jj = [], []
    for i, js in iter_forward_pair_blocks(
        pt,
        min_forward_dt=min_forward_dt,
        max_forward_dt=max_forward_dt,
        max_pairs_per_cell=max_pairs_per_cell,
        pair_seed=pair_seed,
    ):
        ii.append(np.full(js.shape, i, dtype=int))
        jj.append(js.astype(int))
    if not ii:
        raise ValueError("No admissible forward pairs were generated.")
    pair_i = np.concatenate(ii)
    pair_j = np.concatenate(jj)
    return PairData(
        delta_x=X[pair_j] - X[pair_i],
        delta_v=V[pair_j] - V[pair_i],
        pair_i=pair_i,
        pair_j=pair_j,
    )
