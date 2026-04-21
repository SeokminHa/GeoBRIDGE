"""V1: DoRothEA tier-A / CollecTRI enrichment per TF, per method."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from geobridge_val.io import load_rankings
from geobridge_val.gt.dorothea import load_dorothea, gt_sets
from geobridge_val.eval.enrichment import enrich_all_methods


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings", required=True, help="glob, e.g. 'results/rankings/*_bm.parquet'")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-perm", type=int, default=1000)
    ap.add_argument("--min-size", type=int, default=5)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rankings = load_rankings(args.rankings)
    gt = load_dorothea(args.gt, tiers=("A",))
    sets = gt_sets(gt, min_size=args.min_size)

    print(f"[v1] rankings: {len(rankings):,} rows, {rankings['method'].nunique()} methods")
    print(f"[v1] GT: {len(sets)} TFs with ≥{args.min_size} tier-A targets")

    res = enrich_all_methods(rankings, sets, n_perm=args.n_perm, min_size=args.min_size)
    res.to_csv(out / "per_tf.csv", index=False)

    agg = (res.dropna(subset=["nes"])
              .groupby("method")
              .agg(mean_nes=("nes", "mean"),
                   median_nes=("nes", "median"),
                   frac_signif=("fdr", lambda s: (s < 0.05).mean()),
                   n_tf=("tf", "nunique"))
              .reset_index())
    agg.to_csv(out / "summary.csv", index=False)
    print(agg.to_string(index=False))
    print(f"[v1] wrote {out}")


if __name__ == "__main__":
    main()
