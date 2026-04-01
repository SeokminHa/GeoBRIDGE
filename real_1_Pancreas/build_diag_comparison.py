#!/usr/bin/env python3
"""
Merge branch_geobridge_summary + topn_home_vs_away_summary from multiple pancreas_fig3 runs
into one diagnostic comparison CSV (and print to stdout).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Output directories (each with tables/branch_geobridge_summary.csv).",
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("outputs/pancreas_diag_comparison.csv"),
        help="Combined CSV (branch metrics + topn rows; union of columns).",
    )
    args = ap.parse_args()

    branch_frames: list[pd.DataFrame] = []
    topn_frames: list[pd.DataFrame] = []

    for run_dir in args.runs:
        rd = Path(run_dir).resolve()
        manifest: dict = {}
        mp = rd / "run_manifest.json"
        if mp.exists():
            manifest = json.loads(mp.read_text(encoding="utf-8"))
        run_name = str(manifest.get("run_name") or rd.name)

        bpath = rd / "tables" / "branch_geobridge_summary.csv"
        if not bpath.exists():
            print(f"[warn] missing {bpath}")
            continue
        df = pd.read_csv(bpath)
        if "run_name" not in df.columns:
            df.insert(0, "run_name", run_name)
        else:
            df["run_name"] = df["run_name"].astype(str).replace("", pd.NA).fillna(run_name)
        branch_frames.append(df)

        tpath = rd / "tables" / "topn_home_vs_away_summary.csv"
        if tpath.exists():
            td = pd.read_csv(tpath)
            td.insert(0, "run_name", run_name)
            topn_frames.append(td)

    if not branch_frames:
        raise SystemExit("No branch_geobridge_summary.csv files loaded; check --runs paths.")

    br = pd.concat(branch_frames, ignore_index=True)
    want_branch = [
        "run_name",
        "branch",
        "label",
        "pt_source",
        "n_cells",
        "n_pairs",
        "min_dt",
        "max_dt",
        "k_auto_raw",
        "k_used",
        "variance_retained_trace_ratio",
        "num_sum_over_abs",
        "dt_mean",
        "dt_median",
        "dt_p90",
        "dt_p95",
    ]
    for c in want_branch:
        if c not in br.columns:
            br[c] = float("nan")
    br_sel = br[want_branch].copy()
    br_sel.insert(0, "section", "branch_metrics")

    parts: list[pd.DataFrame] = [br_sel]
    if topn_frames:
        tn = pd.concat(topn_frames, ignore_index=True)
        want_topn = ["run_name", "TF", "TOP_N", "median_home", "median_away", "delta_median", "p"]
        for c in want_topn:
            if c not in tn.columns:
                tn[c] = float("nan")
        tn_sel = tn[want_topn].copy()
        tn_sel.insert(0, "section", "topn_home_vs_away")
        parts.append(tn_sel)

    combined = pd.concat(parts, ignore_index=True, sort=False)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.out_csv, index=False)

    print("=== branch_metrics ===")
    print(br_sel.drop(columns=["section"]).to_string(index=False))
    print()
    if topn_frames:
        print("=== topn_home_vs_away ===")
        print(parts[1].drop(columns=["section"]).to_string(index=False))
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
