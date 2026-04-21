"""Sweep k for GeoBRIDGE using the streaming path.

Precomputes pair-space streaming stats ONCE (expensive), then iterates cheaply
over k values, writing one ranking parquet per k. Also records the auto-selected
k from the spectral criterion for reference.
"""
from __future__ import annotations
import argparse, time, yaml, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc

# Ensure parent repo with geobridge_model/ is on path
_here = Path(__file__).resolve()
for p in _here.parents:
    if (p / "geobridge_model" / "__init__.py").exists():
        sys.path.insert(0, str(p))
        break

from geobridge_model.config import GeoBridgeConfig
from geobridge_model.model import GeoBridgeModel
from geobridge_model.spectral_k import select_k_by_tail_exhaustion

from geobridge_val.io import save_ranking


def _get_pseudotime(adata):
    for key in ("velocity_pseudotime", "dpt_pseudotime", "palantir_pseudotime", "pseudotime"):
        if key in adata.obs.columns:
            pt = np.asarray(adata.obs[key]).astype(np.float64)
            if np.all(np.isfinite(pt)):
                return pt
    import scvelo as scv
    scv.tl.velocity_pseudotime(adata)
    return np.asarray(adata.obs["velocity_pseudotime"]).astype(np.float64)


def scores_to_long(scores: np.ndarray, genes: list[str], tf_set: set[str]) -> pd.DataFrame:
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
    ap.add_argument("--k-grid", default="0,1,2,5,10,20,40,auto",
                    help="comma-separated ints, plus optional 'auto' for spectral choice")
    ap.add_argument("--out", default="results/sweeps")
    ap.add_argument("--layer-X", default="Ms")
    ap.add_argument("--layer-V", default="velocity")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    adata = sc.read(cfg["h5ad"])
    tfs = [l.strip() for l in open(cfg["tf_list"]) if l.strip()]
    tf_set = set(tfs)
    genes = adata.var_names.tolist()

    X = np.asarray(adata.layers.get(args.layer_X, adata.X)).astype(np.float64)
    V = np.asarray(adata.layers[args.layer_V]).astype(np.float64)
    pt = _get_pseudotime(adata)
    order = np.argsort(pt, kind="mergesort")
    X = np.nan_to_num(X[order], nan=0.0, posinf=0.0, neginf=0.0)
    V = np.nan_to_num(V[order], nan=0.0, posinf=0.0, neginf=0.0)
    pt = pt[order]

    base = cfg.get("geobridge", {})
    gcfg = GeoBridgeConfig(
        max_k=base.get("max_k", 40),
        tail_fraction=base.get("tail_fraction", 0.2),
        tail_min_start=base.get("tail_min_start", 20),
        slope_window=base.get("slope_window", 3),
        stable_m=base.get("stable_m", 3),
        tau=base.get("tau", 2.0),
        project_y=base.get("project_y", True),
        normalize_score=base.get("normalize_score", True),
    )

    print(f"[sweep] precomputing streaming stats (n_cells={X.shape[0]}, G={X.shape[1]})...")
    t0 = time.time()
    pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
        X, V, pt,
        min_forward_dt=gcfg.min_forward_dt,
        max_forward_dt=gcfg.max_forward_dt,
        max_pairs_per_cell=gcfg.max_pairs_per_cell,
        pair_seed=gcfg.pair_seed,
        stream_num_threads=gcfg.stream_num_threads,
    )
    print(f"[sweep] streaming stats done in {time.time()-t0:.1f}s  "
          f"(n_pairs={pre['n_pairs']:,}, G={pre['G']})")

    spectral = select_k_by_tail_exhaustion(
        pre["S"], gcfg.max_k, gcfg.tail_fraction, gcfg.tail_min_start,
        gcfg.slope_window, gcfg.stable_m, gcfg.tau,
    )
    k_auto = int(spectral.k_selected)
    print(f"[sweep] auto-selected k = {k_auto}")
    S = pre["S"]
    print(f"[sweep] top singular values: {np.round(S[:10], 3).tolist()}")
    print(f"[sweep] S decay: S[0]={S[0]:.3g}, S[10]={S[10]:.3g}, S[40]={S[min(40, len(S)-1)]:.3g}")

    ks = []
    for x in args.k_grid.split(","):
        x = x.strip()
        if x == "auto":
            ks.append(k_auto)
        else:
            ks.append(int(x))
    ks = sorted(set(ks))

    out_dir = Path(args.out) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    for k in ks:
        tag = f"geobridge_k{k:02d}"
        out_pq = out_dir / f"{tag}_{cfg['name']}.parquet"
        if out_pq.exists() and not args.force:
            print(f"[skip] {out_pq}")
            continue
        t0 = time.time()
        scores = GeoBridgeModel.scores_from_streaming_stats(
            pre, k=k, project_y=gcfg.project_y,
            normalize_score=gcfg.normalize_score, eps=gcfg.eps,
        )
        df = scores_to_long(np.asarray(scores), genes, tf_set)
        save_ranking(df, out_pq, method=tag, dataset=cfg["name"])
        print(f"[ok]  {tag}  k={k:2d}  {time.time()-t0:5.1f}s  n_edges={len(df):,}")

    (out_dir / "auto_k.txt").write_text(f"{k_auto}\n")
    print(f"[sweep] done; parquets in {out_dir}")


if __name__ == "__main__":
    main()
