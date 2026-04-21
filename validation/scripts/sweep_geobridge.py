"""GeoBRIDGE hyperparameter sweep. Writes one ranking per (k, lambda, mode)."""
from __future__ import annotations
import argparse, itertools, time, yaml, traceback
from pathlib import Path
import scanpy as sc

from geobridge_val.io import save_ranking
from geobridge_val.methods import geobridge


def parse_list(s: str, cast=str):
    return [cast(x) for x in s.split(",")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k-grid", default="0,5,10,20,auto")
    ap.add_argument("--lambda-grid", default="0.01,0.1,1,10")
    ap.add_argument("--modes", default="xonly,dual")
    ap.add_argument("--out", default="results/sweeps")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    adata = sc.read(cfg["h5ad"])
    tfs = [l.strip() for l in open(cfg["tf_list"]) if l.strip()]

    ks = [x if x == "auto" else int(x) for x in parse_list(args.k_grid)]
    lams = parse_list(args.lambda_grid, float)
    modes = parse_list(args.modes)

    out_dir = Path(args.out) / cfg["name"]; out_dir.mkdir(parents=True, exist_ok=True)

    base = cfg.get("geobridge", {}).copy()
    for k, lam, mode in itertools.product(ks, lams, modes):
        tag = f"geobridge_k{k}_lam{lam}_{mode}"
        out = out_dir / f"{tag}_{cfg['name']}.parquet"
        if out.exists():
            print(f"[skip] {out}"); continue
        params = {**base, "k": k, "lambda_": lam, "mode": mode}
        t0 = time.time()
        try:
            df = geobridge.run(adata, tfs=tfs, **params)
            save_ranking(df, out, method=tag, dataset=cfg["name"])
            print(f"[ok]  {tag:40s} {time.time()-t0:6.1f}s")
        except Exception as e:
            print(f"[ERR] {tag}  {type(e).__name__}: {e}")
            traceback.print_exc()


if __name__ == "__main__":
    main()
