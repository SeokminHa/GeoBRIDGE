"""
EXP 2. Statistical testing.

Tests:
  (1) GeoBRIDGE per-TF AUROC > 0.5        — one-sample Wilcoxon
  (2) Velorama per-TF AUROC > 0.5          — one-sample Wilcoxon
  (3) GeoBRIDGE > Velorama per-TF AUROC    — paired Wilcoxon signed-rank

Uses the existing real_comparison_full.csv (output of phase_compare).

Usage:
  python -u -m benchmark.validation.exp2_statistical_tests
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from benchmark.validation.config import CSV_DIR, EXP2_DIR, DATASETS, K_FOCUS, ensure_dirs


def one_sample_vs_baseline(values, baseline=0.5):
    """Wilcoxon signed-rank test of median(values) vs baseline."""
    values = np.asarray(values)
    values = values[~np.isnan(values)]
    if len(values) < 3:
        return None
    # stats.wilcoxon tests median(values - baseline) == 0
    try:
        stat, p = stats.wilcoxon(values - baseline, alternative="greater")
    except ValueError:
        return None
    return {
        "n": len(values),
        "mean": values.mean(),
        "median": np.median(values),
        "std": values.std(),
        "wilcoxon_stat": stat,
        "wilcoxon_p_greater": p,
    }


def paired_test(values_a, values_b):
    """Paired Wilcoxon: is A > B ?"""
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)
    mask = ~(np.isnan(values_a) | np.isnan(values_b))
    a, b = values_a[mask], values_b[mask]
    if len(a) < 3:
        return None
    try:
        stat, p = stats.wilcoxon(a, b, alternative="greater")
    except ValueError:
        stat, p = np.nan, np.nan
    return {
        "n_paired": len(a),
        "mean_diff": (a - b).mean(),
        "median_diff": float(np.median(a - b)),
        "n_A_gt_B": int((a > b).sum()),
        "n_ties": int((a == b).sum()),
        "wilcoxon_stat": stat,
        "wilcoxon_p_A_greater_B": p,
    }


def run():
    full_path = CSV_DIR / "real_comparison_full.csv"
    if not full_path.exists():
        print(f"[ERROR] Not found: {full_path}")
        print("  Run `phase compare` first.")
        return
    df = pd.read_csv(full_path)

    rows = []
    for ds in DATASETS:
        gb_sub = df[(df["dataset"] == ds) & (df["method"] == "GeoBRIDGE") &
                    (df["k"] == K_FOCUS)].set_index("tf")
        vr_sub = df[(df["dataset"] == ds) & (df["method"] == "Velorama")].set_index("tf")

        print(f"\n=== {ds}  (GB k={K_FOCUS} n={len(gb_sub)}, Velo n={len(vr_sub)}) ===")

        for metric in ["auroc", "auprc", "epr", "gt_vs_bg_ratio"]:
            gb_vals = gb_sub[metric]
            vr_vals = vr_sub[metric]

            baseline = 0.5 if metric == "auroc" else (
                1.0 if metric in ("epr", "gt_vs_bg_ratio") else 0.0)

            # One-sample vs baseline
            if metric in ("auroc", "epr", "gt_vs_bg_ratio"):
                gb_one = one_sample_vs_baseline(gb_vals.values, baseline=baseline)
                vr_one = one_sample_vs_baseline(vr_vals.values, baseline=baseline)
                if gb_one:
                    rows.append({"dataset": ds, "test": f"GB_{metric}_vs_{baseline}",
                                 **gb_one})
                if vr_one:
                    rows.append({"dataset": ds, "test": f"Velo_{metric}_vs_{baseline}",
                                 **vr_one})

            # Paired: GB vs Velo on matched TFs
            common = gb_sub.index.intersection(vr_sub.index)
            if len(common) >= 3:
                pair = paired_test(gb_sub.loc[common, metric].values,
                                   vr_sub.loc[common, metric].values)
                if pair:
                    rows.append({"dataset": ds, "test": f"paired_{metric}_GB_vs_Velo",
                                 **pair})

    results_df = pd.DataFrame(rows)
    results_df.to_csv(EXP2_DIR / "statistical_tests.csv", index=False)

    # Markdown report
    lines = ["# EXP 2 Statistical tests\n"]
    for ds in DATASETS:
        lines.append(f"\n## {ds}\n")
        sub = results_df[results_df["dataset"] == ds]

        lines.append("### One-sample (is method > random baseline?)\n")
        lines.append("| test | n | mean | median | p(>baseline) |")
        lines.append("|---|---|---|---|---|")
        for _, r in sub[sub["test"].str.contains("_vs_")].iterrows():
            p = r.get("wilcoxon_p_greater", np.nan)
            lines.append(f"| {r['test']} | {r.get('n', '')} | "
                         f"{r.get('mean', np.nan):.4f} | "
                         f"{r.get('median', np.nan):.4f} | "
                         f"{p:.3g} |")

        lines.append("\n### Paired (GB vs Velo)\n")
        lines.append("| test | n_pairs | mean_diff | median_diff | n_GB>V | p(GB>V) |")
        lines.append("|---|---|---|---|---|---|")
        for _, r in sub[sub["test"].str.contains("paired")].iterrows():
            p = r.get("wilcoxon_p_A_greater_B", np.nan)
            lines.append(
                f"| {r['test']} | {r.get('n_paired', '')} | "
                f"{r.get('mean_diff', np.nan):.4f} | "
                f"{r.get('median_diff', np.nan):.4f} | "
                f"{r.get('n_A_gt_B', '')} | {p:.3g} |"
            )

    report = "\n".join(lines)
    (EXP2_DIR / "REPORT.md").write_text(report)
    print(f"\n[SAVE] {EXP2_DIR}/statistical_tests.csv")
    print(f"[SAVE] {EXP2_DIR}/REPORT.md")
    print(report)


if __name__ == "__main__":
    ensure_dirs()
    run()
