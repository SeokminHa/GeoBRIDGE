"""Early-pseudotime restriction test for GeoBRIDGE.

Hypothesis (Day 4): sign-flip on lineage TFs (GATA2/KLF1/NFE2) is driven by
late/committed cells — TF has already peaked and plateaued while the target
gene is still overshooting, producing ΔX_TF·ΔV_target < 0 in raw M.
Restricting pairs to the earliest pseudotime subset should eliminate that
plateau regime and (if the hypothesis holds) recover lineage TFs.

Implementation:
  1. Load adata, X = Ms, V = velocity, pseudotime.
  2. Keep only cells with pt ≤ --pt-quantile (default 0.25 = first quartile).
  3. Recompute streaming pair-space stats on the filtered subset.
  4. Pick k via select_k_by_tail_exhaustion (same spectral rule production
     uses in fit()), unless --k overrides.
  5. Score with the production operator (mode="out", project_y per config).
  6. Emit parquet: gb_earlyPT{Q}_<name>.parquet.

Decision rule is enforced by the downstream V2 ChIP evaluation: lineage TFs
≥ +0.5 NES on 2/3 → success; weaker → restriction did not help → combined
with nuisance failure, GeoBRIDGE's projection-space scope is closed.
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
from geobridge_model.spectral_k import select_k_by_tail_exhaustion

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
    ap.add_argument("--pt-quantile", type=float, default=0.25,
                    help="keep cells with pseudotime <= this quantile")
    ap.add_argument("--ks", default="auto",
                    help="comma-separated list of k values; special value "
                         "'auto' = spectral auto-select on subset. "
                         "Example: '0,2,auto' produces three parquets in "
                         "one run, sharing the streaming stats.")
    ap.add_argument("--out", default="results/rankings_earlyPT")
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

    q = float(args.pt_quantile)
    if not 0.0 < q <= 1.0:
        raise SystemExit(f"--pt-quantile must be in (0, 1], got {q}")
    cutoff = float(np.quantile(pt, q))
    mask = pt <= cutoff
    n_keep = int(mask.sum())
    print(f"[earlyPT] pt_quantile={q:.2f} cutoff={cutoff:.4f}  "
          f"n_cells kept={n_keep}/{X.shape[0]}")
    if n_keep < 200:
        raise SystemExit(f"too few cells after restriction ({n_keep}); "
                         f"increase --pt-quantile")

    X = X[mask]
    V = V[mask]
    pt_sub = pt[mask]
    order = np.argsort(pt_sub, kind="mergesort")
    X = np.nan_to_num(X[order], nan=0.0, posinf=0.0, neginf=0.0)
    V = np.nan_to_num(V[order], nan=0.0, posinf=0.0, neginf=0.0)
    pt_sub = pt_sub[order]

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

    t0 = time.time()
    pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
        X, V, pt_sub,
        min_forward_dt=gcfg.min_forward_dt,
        max_forward_dt=gcfg.max_forward_dt,
        max_pairs_per_cell=gcfg.max_pairs_per_cell,
        pair_seed=gcfg.pair_seed,
        stream_num_threads=gcfg.stream_num_threads,
    )
    print(f"[earlyPT] streaming stats: n_pairs={pre['n_pairs']:,} G={pre['G']} "
          f"({time.time()-t0:.1f}s)")
    if pre["n_pairs"] < 500:
        raise SystemExit(f"too few pairs ({pre['n_pairs']}); restriction too tight")

    ks_list = []
    for tok in args.ks.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok == "auto":
            spectral = select_k_by_tail_exhaustion(
                pre["S"], gcfg.max_k, gcfg.tail_fraction, gcfg.tail_min_start,
                gcfg.slope_window, gcfg.stable_m, gcfg.tau,
            )
            k_auto = int(spectral.k_selected)
            print(f"[earlyPT] auto k={k_auto} "
                  f"(tail_baseline={spectral.tail_baseline:.3f}, "
                  f"threshold={spectral.threshold:.3f})")
            ks_list.append(("auto", k_auto))
        else:
            ks_list.append((tok, int(tok)))

    out_dir = Path(args.out) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    qtag = f"q{int(round(q*100)):02d}"

    for label, K in ks_list:
        # tag: use label for auto (resolves to number), else raw int
        tag = f"gb_earlyPT_{qtag}_k{K:02d}"
        out_pq = out_dir / f"{tag}_{cfg['name']}.parquet"
        if out_pq.exists() and not args.force:
            print(f"[skip] {out_pq}")
            continue
        t1 = time.time()
        scores = GeoBridgeModel.scores_from_streaming_stats(
            pre, k=K, mode="out",
            project_y=gcfg.project_y, normalize_score=True, eps=gcfg.eps,
        )
        df = _scores_to_long(np.asarray(scores), genes, tf_set)
        save_ranking(df, out_pq, method=tag, dataset=cfg["name"])
        print(f"[ok]  {tag}  ({label})  {time.time()-t1:4.1f}s  "
              f"n_edges={len(df):,}")

    print(f"[earlyPT] done; parquets in {out_dir}")


if __name__ == "__main__":
    main()
