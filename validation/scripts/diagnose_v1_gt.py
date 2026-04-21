"""Audit V1 evaluation trustworthiness before interpreting enrichment metrics.

For each ranking parquet:
  - count TFs in ranking, in GT, and in the overlap that gseapy would evaluate
  - distribution of (GT targets ∩ ranked genes) sizes per TF
  - gene-symbol case/species mismatch probe
  - compare datasets side-by-side (BM human vs pancreas mouse)

If BM and pancreas overlap distributions differ substantially, cross-dataset
V1 comparisons are confounded by GT quality.
"""
from __future__ import annotations
import argparse
from glob import glob
from pathlib import Path
import pandas as pd
import numpy as np


def load_gt(path: Path) -> dict[str, set[str]]:
    df = pd.read_csv(path, sep="\t")
    if "tf" not in df.columns and "regulator" in df.columns:
        df = df.rename(columns={"regulator": "tf"})
    if not {"tf", "target"}.issubset(df.columns):
        raise ValueError(f"GT needs tf,target cols; got {df.columns.tolist()}")
    return df.groupby("tf")["target"].apply(lambda s: set(s.astype(str))).to_dict()


def diagnose_ranking(df_rank: pd.DataFrame, gt: dict[str, set[str]], min_overlap: int):
    all_ranked = set(df_rank["target"].astype(str).unique())
    per_tf = []
    for tf, grp in df_rank.groupby("regulator"):
        gt_t = gt.get(tf, set())
        if not gt_t:
            per_tf.append({"tf": tf, "n_gt": 0, "n_overlap": 0, "in_gt": False})
            continue
        overlap = gt_t & set(grp["target"].astype(str))
        per_tf.append({"tf": tf, "n_gt": len(gt_t), "n_overlap": len(overlap), "in_gt": True})
    tf_df = pd.DataFrame(per_tf)

    in_gt = tf_df[tf_df["in_gt"]]
    return tf_df, {
        "n_tf_in_ranking": int(df_rank["regulator"].nunique()),
        "n_tf_in_gt_and_ranking": int(len(in_gt)),
        "n_evaluable_ge_min": int((in_gt["n_overlap"] >= min_overlap).sum()),
        "n_skipped_lt_min": int((in_gt["n_overlap"] < min_overlap).sum()),
        "mean_overlap": float(in_gt["n_overlap"].mean()) if len(in_gt) else float("nan"),
        "median_overlap": float(in_gt["n_overlap"].median()) if len(in_gt) else float("nan"),
        "n_ranked_genes": len(all_ranked),
    }, all_ranked


def symbol_audit(ranked: set[str], gt: dict[str, set[str]]):
    gt_genes = set()
    for s in gt.values():
        gt_genes |= s
    r_upper = {g.upper() for g in ranked}
    g_upper = {g.upper() for g in gt_genes}
    exact = ranked & gt_genes
    ci_match = r_upper & g_upper
    case_only = len(ci_match) - len(exact)
    return {
        "ranked_size": len(ranked),
        "gt_gene_universe": len(gt_genes),
        "exact_match": len(exact),
        "case_insensitive_match": len(ci_match),
        "case_mismatch_only": case_only,
        "frac_ranked_in_gt_exact": len(exact) / max(len(ranked), 1),
        "frac_ranked_in_gt_ci": len(ci_match) / max(len(ranked), 1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings", required=True, help="glob for parquets")
    ap.add_argument("--gt", required=True)
    ap.add_argument("--dataset-name", default=None)
    ap.add_argument("--min-overlap", type=int, default=5,
                    help="gseapy default min_size")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    gt = load_gt(Path(args.gt))
    print(f"[diag] GT: {len(gt)} TFs; {sum(len(v) for v in gt.values())} total edges")

    paths = sorted(glob(args.rankings))
    if not paths:
        raise SystemExit(f"no parquets match {args.rankings}")

    summary_rows = []
    per_tf_rows = []
    audit_rows = []
    for p in paths:
        stem = Path(p).stem
        parts = stem.rsplit("_", 1)
        method = parts[0] if len(parts) > 1 else stem
        dataset = args.dataset_name or (parts[1] if len(parts) > 1 else "?")

        df_rank = pd.read_parquet(p)
        tf_df, summary, ranked_set = diagnose_ranking(df_rank, gt, args.min_overlap)
        audit = symbol_audit(ranked_set, gt)

        tf_df["dataset"] = dataset; tf_df["method"] = method
        per_tf_rows.append(tf_df)

        row = {"dataset": dataset, "method": method, **summary}
        summary_rows.append(row)
        audit_rows.append({"dataset": dataset, "method": method, **audit})

        print(f"[diag] {dataset:10s} {method:30s} "
              f"eval/skip={summary['n_evaluable_ge_min']}/{summary['n_skipped_lt_min']}  "
              f"mean_overlap={summary['mean_overlap']:.1f}  "
              f"exact_match={audit['exact_match']}/{audit['ranked_size']}  "
              f"case_only={audit['case_mismatch_only']}")

    summary_df = pd.DataFrame(summary_rows)
    audit_df = pd.DataFrame(audit_rows)
    per_tf_df = pd.concat(per_tf_rows, ignore_index=True)
    summary_df.to_csv(out / "summary.csv", index=False)
    audit_df.to_csv(out / "symbol_audit.csv", index=False)
    per_tf_df.to_csv(out / "per_tf_overlap.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(9, 5))
        for (ds, mt), grp in per_tf_df[per_tf_df["in_gt"]].groupby(["dataset", "method"]):
            ax.hist(grp["n_overlap"], bins=np.arange(0, grp["n_overlap"].max() + 2),
                    alpha=0.4, label=f"{ds}/{mt}", density=True)
        ax.axvline(args.min_overlap, color="red", ls="--",
                   label=f"min_size={args.min_overlap}")
        ax.set_xlabel("overlap (GT targets ∩ ranked genes) per TF")
        ax.set_ylabel("density")
        ax.legend(fontsize=6)
        fig.tight_layout()
        fig.savefig(out / "overlap_histogram.png", dpi=150)
    except Exception as e:
        print(f"[diag] plot skipped: {e}")

    print(f"\n[diag] wrote {out}/ {{summary,symbol_audit,per_tf_overlap}}.csv + histogram")


if __name__ == "__main__":
    main()
