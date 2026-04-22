"""Direct pseudotime-drift projection test for GeoBRIDGE.

Hypothesis (Day 5): the sign inversion on lineage activators in BM is caused
by a pseudotime-drift direction dominating the pair-space cross-covariance
M = ΔXc^T ΔV. Projecting out the low-dimensional gene-space subspace
spanned by Δpt-aligned drift should restore the correct sign.

This is the clean mechanistic test of Day 4's V_nuis result, whose top-gene
loadings (MPO/LYZ/ELANE/HBB/CD79B/VPREB) showed V_nuis was in fact a
pseudotime-correlated lineage commitment axis, not a technical-nuisance
basis. Using Δpt directly removes the "is it really nuisance?" ambiguity.

Pipeline:
  1. Sample N_p forward pairs (i<j in pt-sorted order).
  2. Regress centered ΔX on polynomial features of Δpt up to --orders:
       ΔX ≈ [Δpt, Δpt², ..., Δpt^d] · B         (d = order)
  3. V_drift = QR(B.T) → orthonormal (G, d) gene-space basis.
  4. Reuse the streaming GxG pair-space stats (C, M, Y).
  5. Score = (I − V_drift V_drift^T) M  with production normalization.

Tags:
  gb_drift_none            baseline (same as raw M; emitted once)
  gb_drift_out_o{d}dim{d}  main test, one per order
"""
from __future__ import annotations
import argparse
import sys
import time
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

_here = Path(__file__).resolve()
for p in _here.parents:
    if (p / "geobridge_model" / "__init__.py").exists():
        sys.path.insert(0, str(p))
        break

from geobridge_model.config import GeoBridgeConfig
from geobridge_model.model import GeoBridgeModel
from geobridge_val.io import save_ranking


def _get_pseudotime(adata):
    for key in ("velocity_pseudotime", "dpt_pseudotime",
                "palantir_pseudotime", "pseudotime"):
        if key in adata.obs.columns:
            pt = np.asarray(adata.obs[key]).astype(np.float64)
            if np.all(np.isfinite(pt)):
                return pt
    import scvelo as scv
    scv.tl.velocity_pseudotime(adata)
    return np.asarray(adata.obs["velocity_pseudotime"]).astype(np.float64)


def _drift_basis(X, pt, n_pairs=100_000, dt_max=50, order=1, seed=0):
    """Pair-space drift basis via ΔX ≈ poly(Δpt) @ B OLS.

    Returns V_drift ∈ R^{G × d} with d = order (orthonormal columns).
    Caller is responsible for pt-ordering X and pt (j > i in index => Δpt > 0).
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    i_idx = rng.integers(0, n - 1, size=n_pairs)
    max_off = np.clip(n - 1 - i_idx, 1, dt_max)
    off = (rng.random(n_pairs) * max_off).astype(np.int64) + 1
    j_idx = np.clip(i_idx + off, 0, n - 1)

    dpt = (pt[j_idx] - pt[i_idx]).astype(np.float64)
    dX = X[j_idx] - X[i_idx]
    dX -= dX.mean(axis=0, keepdims=True)

    if order == 1:
        DN = dpt[:, None].copy()
    else:
        cols = [dpt]
        for p in range(2, order + 1):
            cols.append(dpt ** p)
        DN = np.column_stack(cols)
    DN -= DN.mean(axis=0, keepdims=True)

    B, *_ = np.linalg.lstsq(DN, dX, rcond=None)   # B: (order, G)
    V_drift, _ = np.linalg.qr(B.T)                # (G, order)
    return V_drift


def _scores_with_basis(pre, V, mode="out", eps=1e-12):
    """Score matrix using explicit gene-space basis V (G, d)."""
    C = pre["C"]; M = pre["M"]; Y = pre["Y"]; G = int(pre["G"])
    d = 0 if V is None else V.shape[1]
    if d == 0 or mode == "none":
        num = M
        x_norm2 = np.clip(np.diag(C), 0.0, None)
    else:
        P = V @ V.T
        if mode == "out":
            Q = np.eye(G) - P
            num = Q @ M
            x_norm2 = np.clip(np.diag(Q @ C @ Q), 0.0, None)
        elif mode == "onto":
            num = P @ M
            x_norm2 = np.clip(np.diag(P @ C @ P), 0.0, None)
        else:
            raise ValueError(f"bad mode {mode!r}; expected out|onto|none")
    y_norm2 = np.clip(np.diag(Y), 0.0, None)
    denom = np.maximum(np.sqrt(x_norm2)[:, None] * np.sqrt(y_norm2)[None, :], eps)
    return num / denom


def _scores_to_long(scores, genes, tf_set):
    tf_idx = [i for i, g in enumerate(genes) if g in tf_set]
    rows = []
    for i in tf_idx:
        tf = genes[i]
        for j, tgt in enumerate(genes):
            if j == i:
                continue
            rows.append((tf, tgt, float(scores[i, j])))
    return pd.DataFrame(rows, columns=["regulator", "target", "score"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/rankings_drift")
    ap.add_argument("--layer-X", default="Ms")
    ap.add_argument("--layer-V", default="velocity")
    ap.add_argument("--orders", default="1,2,3",
                    help="comma-separated polynomial orders for drift basis "
                         "(order=1 => 1-d Δpt; order=2 => [Δpt, Δpt²]; etc.)")
    ap.add_argument("--modes", default="none,out",
                    help="scoring modes; 'none' emits raw-M baseline once")
    ap.add_argument("--n-pairs-for-basis", type=int, default=100_000)
    ap.add_argument("--max-pairs-per-cell", type=int, default=None,
                    help="override config.max_pairs_per_cell for streaming "
                         "precompute (lower = faster, e.g. 200 ~ 5 min on BM)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    adata = sc.read(cfg["h5ad"])
    tfs = [l.strip() for l in open(cfg["tf_list"]) if l.strip()]
    tf_set = set(tfs)
    genes = adata.var_names.tolist()

    X = np.asarray(adata.layers.get(args.layer_X, adata.X)).astype(np.float64)
    V_lay = np.asarray(adata.layers[args.layer_V]).astype(np.float64)
    pt = _get_pseudotime(adata)
    order_idx = np.argsort(pt, kind="mergesort")
    X = np.nan_to_num(X[order_idx], nan=0.0, posinf=0.0, neginf=0.0)
    V_lay = np.nan_to_num(V_lay[order_idx], nan=0.0, posinf=0.0, neginf=0.0)
    pt = pt[order_idx]

    base = cfg.get("geobridge", {})
    gcfg = GeoBridgeConfig(
        max_k=base.get("max_k", 40),
        tail_fraction=base.get("tail_fraction", 0.2),
        tail_min_start=base.get("tail_min_start", 20),
        slope_window=base.get("slope_window", 3),
        stable_m=base.get("stable_m", 3),
        tau=base.get("tau", 2.0),
        project_y=False,
        normalize_score=base.get("normalize_score", True),
    )
    mpc = args.max_pairs_per_cell if args.max_pairs_per_cell is not None \
          else gcfg.max_pairs_per_cell

    print(f"[drift] precomputing streaming stats "
          f"(n_cells={X.shape[0]}, G={X.shape[1]}, max_pairs_per_cell={mpc})...")
    t0 = time.time()
    pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
        X, V_lay, pt,
        min_forward_dt=gcfg.min_forward_dt,
        max_forward_dt=gcfg.max_forward_dt,
        max_pairs_per_cell=mpc,
        pair_seed=gcfg.pair_seed,
        stream_num_threads=gcfg.stream_num_threads,
    )
    print(f"[drift] streaming stats: n_pairs={pre['n_pairs']:,} G={pre['G']} "
          f"({time.time()-t0:.1f}s)")

    out_dir = Path(args.out) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    orders = [int(o.strip()) for o in args.orders.split(",") if o.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]

    none_emitted = False
    for od in orders:
        V_drift = _drift_basis(X, pt, n_pairs=args.n_pairs_for_basis, order=od)
        d = V_drift.shape[1]
        np.savez(out_dir / f"vdrift_order{od}_{cfg['name']}.npz",
                 V_drift=V_drift, order=np.array([od]))
        print(f"[drift] order={od}  V_drift.shape={V_drift.shape}")

        for mode in modes:
            if mode == "none":
                if none_emitted:
                    continue
                tag = "gb_drift_none"
                none_emitted = True
            else:
                tag = f"gb_drift_{mode}_o{od}d{d}"
            out_pq = out_dir / f"{tag}_{cfg['name']}.parquet"
            if out_pq.exists() and not args.force:
                print(f"[skip] {out_pq}")
                continue
            t1 = time.time()
            S = _scores_with_basis(pre, V_drift, mode=mode, eps=gcfg.eps)
            df = _scores_to_long(np.asarray(S), genes, tf_set)
            save_ranking(df, out_pq, method=tag, dataset=cfg["name"])
            print(f"[ok]  {tag}  {time.time()-t1:4.1f}s  n_edges={len(df):,}")

    print(f"[drift] done; artifacts in {out_dir}")


if __name__ == "__main__":
    main()
