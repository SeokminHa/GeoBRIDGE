"""Run one or more GRN methods on a dataset. Outputs unified ranking parquet per method."""
from __future__ import annotations
import argparse, time, yaml, traceback
from pathlib import Path
import scanpy as sc

from geobridge_val.io import save_ranking
from geobridge_val.methods import REGISTRY


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--methods", required=True,
                    help="comma-separated, any of: " + ",".join(REGISTRY))
    ap.add_argument("--out", default="results/rankings")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    adata = sc.read(cfg["h5ad"])
    tfs = [l.strip() for l in open(cfg["tf_list"]) if l.strip()]
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)

    wanted = [m.strip() for m in args.methods.split(",")]
    for m in wanted:
        if m not in REGISTRY:
            raise KeyError(f"unknown method {m}; available: {list(REGISTRY)}")

    for m in wanted:
        out = out_dir / f"{m}_{cfg['name']}.parquet"
        if out.exists() and not args.force:
            print(f"[skip] {out}  (use --force to rerun)")
            continue
        t0 = time.time()
        try:
            df = REGISTRY[m](adata, tfs=tfs, **cfg.get(m, {}))
            save_ranking(df, out, method=m, dataset=cfg["name"])
            print(f"[ok]  {m:12s} {time.time()-t0:7.1f}s  n_edges={len(df):,}  -> {out}")
        except Exception as e:
            print(f"[ERR] {m:12s} {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
