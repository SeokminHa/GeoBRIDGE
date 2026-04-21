"""V2: AUPRC / EPR vs ChIP-derived edges, with bootstrap CIs."""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

from geobridge_val.io import load_rankings
from geobridge_val.gt.chip import load_chip_edges
from geobridge_val.eval.auprc import eval_all_methods
from geobridge_val.eval.nulls import shuffle_label_auprc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings", required=True)
    ap.add_argument("--gt", required=True, help="TSV with columns tf,target")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--null-perm", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    rankings = load_rankings(args.rankings)
    gt = load_chip_edges(args.gt)
    gt_set = set(zip(gt["tf"].astype(str), gt["target"].astype(str)))
    print(f"[v2] rankings: {len(rankings):,}  GT edges: {len(gt_set):,}")

    per_tf = eval_all_methods(rankings, gt_set, n_boot=args.bootstrap)
    per_tf.to_csv(out / "per_tf.csv", index=False)

    null = shuffle_label_auprc(rankings, gt_set, n_perm=args.null_perm)
    null.to_csv(out / "null.csv", index=False)

    summary = (per_tf.groupby("method")
                     .agg(mean_auprc=("auprc", "mean"),
                          median_auprc=("auprc", "median"),
                          mean_epr=("epr", "mean"),
                          n_tf=("tf", "nunique"))
                     .reset_index()
                     .merge(null, on="method", how="left"))
    summary.to_csv(out / "summary.csv", index=False)
    print(summary.to_string(index=False))
    print(f"[v2] wrote {out}")


if __name__ == "__main__":
    main()
