"""
Master kill-criteria checker. Aggregates results from Top-N, stdAUPRC,
per-TF AUROC density into per-dataset "evidence state" with explicit
consistency check across metrics.

Evidence states:
  strong         -- ≥ 2 of 3 metrics pass effect-size threshold
  weak           -- exactly 1 of 3 passes
  contradictory  -- one metric passes in +direction AND another passes
                    in -direction (SVD < Random with effect)
  inconclusive   -- all 3 have |Δ| < effect threshold

Global decisions:
  PUBLISHABLE    -- SERGIO A AND C strong; pancreas strong; bonemarrow ≥ weak
  INTERMEDIATE   -- pancreas ≥ weak; bonemarrow possibly weak
  PIVOT          -- pancreas inconclusive/contradictory, OR SERGIO A/C fails

Usage (after running Tasks 2, 3, 5):
  python -u -m benchmark.post_analysis.decision_check
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

POST = Path("results/post_analysis")


# Effect-size thresholds per metric per dataset-category
#  SERGIO primary (A, C): strict
#  SERGIO rescue (B, D):  narrative-oriented, separate bar
#  pancreas:              strict (primary real-data)
#  bonemarrow:            relaxed (secondary real-data)

def classify_metric(ds, metric, delta, ratio_or_abs=None, p=None):
    """
    Returns one of: "pass", "flat", "negative" for this dataset/metric.
    Effect thresholds differ per (ds_class, metric).
    """
    # Treat bonemarrow separately (relaxed)
    ds_class = (
        "sergio_primary" if ds in ("A", "C")
        else "sergio_rescue" if ds in ("B", "D")
        else "pancreas" if ds == "pancreas"
        else "bonemarrow" if ds == "bonemarrow"
        else "other"
    )

    if metric == "topN":  # Δ precision@20
        thr = {"sergio_primary": 0.05, "sergio_rescue": 0.05,
               "pancreas": 0.005, "bonemarrow": 0.002}[ds_class]
        neg_thr = thr  # symmetric
    elif metric == "std_auprc":
        thr = {"sergio_primary": 0.5, "sergio_rescue": 0.5,
               "pancreas": 0.3, "bonemarrow": 0.3}[ds_class]
        neg_thr = thr
    elif metric == "auroc":
        thr = {"sergio_primary": 0.05, "sergio_rescue": 0.05,
               "pancreas": 0.02, "bonemarrow": 0.02}[ds_class]
        neg_thr = thr
    else:
        raise ValueError(metric)

    if delta >= thr:
        return "pass"
    if delta <= -neg_thr:
        return "negative"
    return "flat"


def evidence_state(labels):
    """Given list of 'pass' | 'flat' | 'negative', classify."""
    n_pass = sum(l == "pass" for l in labels)
    n_neg = sum(l == "negative" for l in labels)
    if n_pass >= 1 and n_neg >= 1:
        return "contradictory"
    if n_pass >= 2:
        return "strong"
    if n_pass == 1:
        return "weak"
    return "inconclusive"


def load_inputs():
    topN = POST / "topN_stats.csv"
    stdap = POST / "stdauprc_stats_k5.csv"
    tfauc = POST / "tf_auroc_dist_paired.csv"
    if not topN.exists() or not stdap.exists() or not tfauc.exists():
        print(f"[ERROR] Run post_topN.py, post_stdauprc_vs_k.py, "
              f"post_tf_auroc_density.py first.")
        sys.exit(1)
    tn = pd.read_csv(topN)
    sa = pd.read_csv(stdap)
    au = pd.read_csv(tfauc)
    return tn, sa, au


def main():
    tn, sa, au = load_inputs()
    # Keep Top-N at N=20 for the decision
    tn20 = tn[tn["N"] == 20].set_index("dataset")
    sa = sa.set_index("dataset")
    au = au.set_index("dataset")

    datasets = sorted(set(tn20.index) | set(sa.index) | set(au.index))
    rows = []
    for ds in datasets:
        d_topn = float(tn20.loc[ds, "delta"]) if ds in tn20.index else np.nan
        d_stda = float(sa.loc[ds, "delta"]) if ds in sa.index else np.nan
        d_auc = float(au.loc[ds, "delta_median"]) if ds in au.index else np.nan

        labels = [
            classify_metric(ds, "topN", d_topn) if not np.isnan(d_topn) else "flat",
            classify_metric(ds, "std_auprc", d_stda) if not np.isnan(d_stda) else "flat",
            classify_metric(ds, "auroc", d_auc) if not np.isnan(d_auc) else "flat",
        ]
        state = evidence_state(labels)
        rows.append({
            "dataset": ds,
            "delta_topN20": d_topn,
            "delta_std_auprc": d_stda,
            "delta_median_auroc": d_auc,
            "topN_label": labels[0],
            "std_auprc_label": labels[1],
            "auroc_label": labels[2],
            "evidence": state,
        })

    result = pd.DataFrame(rows).sort_values("dataset")
    result.to_csv(POST / "decision_summary.csv", index=False)
    print("=== PER-DATASET EVIDENCE STATES ===")
    print(result.to_string(index=False))

    # Global decision
    def state(ds):
        r = result[result["dataset"] == ds]
        return r["evidence"].iloc[0] if len(r) else "missing"

    print("\n=== GLOBAL DECISION ===")
    sergio_primary_ok = (state("A") == "strong") and (state("C") == "strong")
    pancreas_state = state("pancreas")
    bonemarrow_state = state("bonemarrow")

    print(f"  SERGIO primary (A AND C both strong): {sergio_primary_ok}")
    print(f"  SERGIO rescue (B, D): {state('B')}, {state('D')}")
    print(f"  pancreas: {pancreas_state}")
    print(f"  bonemarrow: {bonemarrow_state}")

    if not sergio_primary_ok:
        verdict = "PIVOT — SERGIO primary failed"
    elif pancreas_state == "strong":
        if bonemarrow_state in ("strong", "weak"):
            verdict = "PUBLISHABLE"
        elif bonemarrow_state == "inconclusive":
            verdict = "INTERMEDIATE — pancreas strong, bonemarrow null; constrain generality claim"
        elif bonemarrow_state == "contradictory":
            verdict = "INTERMEDIATE — bonemarrow contradictory; run Task 6 (gene-level)"
        else:
            verdict = "INTERMEDIATE — bonemarrow weak"
    elif pancreas_state == "weak":
        verdict = "INTERMEDIATE — pancreas weak; run Task 6 before publication"
    elif pancreas_state == "contradictory":
        verdict = "INVESTIGATE — pancreas contradictory; run Task 6 to resolve before pivot"
    else:  # inconclusive
        verdict = "PIVOT — pancreas inconclusive → synthetic-primary paper"

    print(f"\n  >>> {verdict} <<<\n")

    (POST / "decision_verdict.md").write_text(
        f"# Decision verdict\n\n"
        f"- SERGIO primary (A and C strong): **{sergio_primary_ok}**\n"
        f"- pancreas evidence: **{pancreas_state}**\n"
        f"- bonemarrow evidence: **{bonemarrow_state}**\n"
        f"- SERGIO rescue (B, D): {state('B')}, {state('D')}\n\n"
        f"## Verdict: {verdict}\n\n"
        f"See `decision_summary.csv` for per-dataset metric labels.\n"
    )


if __name__ == "__main__":
    main()
