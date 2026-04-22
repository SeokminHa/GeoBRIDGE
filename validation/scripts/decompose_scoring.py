"""Subspace decomposition of GeoBRIDGE scoring.

Precomputes pair-space stats ONCE from the dataset, then emits three ranking
parquets at a fixed k, corresponding to three projector choices on the
numerator of the GeoBRIDGE score:

  none (k=0)      : M                         (unprojected baseline)
  onto            : Pk @ M = V_k V_k^T  M     (lineage / dominant-geometry component)
  out (current)   : (I - Pk) @ M              (orthogonal / residual component)

Decision rule (see real_day2.md "Day 3 update"):
  if `onto` recovers lineage TFs and `out` keeps stress/IEG wins,
  the decomposition hypothesis is confirmed and the method should be
  redefined as a two-score operator.

Downstream V1/V2 enrichment scripts consume these parquets unchanged.
"""
from __future__ import annotations
import argparse, time, yaml, sys
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


def scores_to_long(scores: np.ndarray, genes: list[str],
                   tf_set: set[str]) -> pd.DataFrame:
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
    ap.add_argument("--k", type=int, required=True,
                    help="single k at which to decompose (e.g. 17 for BM 'auto')")
    ap.add_argument("--modes", default="none,onto,out",
                    help="comma-separated subset of {none,onto,out}")
    ap.add_argument("--out", default="results/rankings_decomp")
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

    print(f"[decomp] precomputing streaming stats "
          f"(n_cells={X.shape[0]}, G={X.shape[1]})...")
    t0 = time.time()
    pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
        X, V, pt,
        min_forward_dt=gcfg.min_forward_dt,
        max_forward_dt=gcfg.max_forward_dt,
        max_pairs_per_cell=gcfg.max_pairs_per_cell,
        pair_seed=gcfg.pair_seed,
        stream_num_threads=gcfg.stream_num_threads,
    )
    print(f"[decomp] streaming stats done in {time.time()-t0:.1f}s  "
          f"(n_pairs={pre['n_pairs']:,}, G={pre['G']})")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        if m not in ("none", "onto", "out"):
            raise SystemExit(f"bad mode {m!r}; expected none|onto|out")

    out_dir = Path(args.out) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    K = int(args.k)
    for mode in modes:
        # 'none' is k-independent; tag without k; 'onto'/'out' carry k
        tag = f"gb_{mode}" if mode == "none" else f"gb_{mode}_k{K:02d}"
        out_pq = out_dir / f"{tag}_{cfg['name']}.parquet"
        if out_pq.exists() and not args.force:
            print(f"[skip] {out_pq}")
            continue
        t0 = time.time()
        scores = GeoBridgeModel.scores_from_streaming_stats(
            pre, k=K, mode=mode,
            # y-side residualization only applies to 'out' by current design
            project_y=(mode == "out") and gcfg.project_y,
            normalize_score=True,
            eps=gcfg.eps,
        )
        df = scores_to_long(np.asarray(scores), genes, tf_set)
        save_ranking(df, out_pq, method=tag, dataset=cfg["name"])
        print(f"[ok]  {tag}  mode={mode} k={K:2d}  "
              f"{time.time()-t0:5.1f}s  n_edges={len(df):,}")

    print(f"[decomp] done; parquets in {out_dir}")


if __name__ == "__main__":
    main()
