"""
T1 — SVD vs Random Projection ablation (CRITICAL).

Tests whether pair-space SVD outperforms matched-rank random Gaussian
projection when plugged into the identical scoring formula.

Design:
  For each dataset, sample n_pairs pair-space features (F_X, F_V).
  For each k ∈ K_VALUES, compute three methods:
    - IDENTITY (no projection):     scores from raw F_V
    - SVD:                           P = top-k right singular vectors of F_V
    - RANDOM × N_RANDOM_SEEDS:       P = orthonormalized Gaussian (G × k)
  Apply identical scoring formula; evaluate per-TF AUROC / AUPRC.

Success criterion (pre-specified):
  SVD AUROC > (mean Random AUROC + 1.96·SE) for ≥4 of N datasets at k=5,
  AND paired Wilcoxon(SVD, Random across TFs) p < 0.05 in ≥3 datasets.

Usage:
  python -u -m benchmark.ablation_projection --datasets A B C D bonemarrow pancreas
"""
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from benchmark.utils import (
    load_dataset, sample_pair_features,
    projection_random, score_with_projection, evaluate_scores_per_tf,
    SERGIO_DATASETS,
)

OUTDIR = Path("results/ablation_projection")

K_VALUES_DEFAULT = [1, 3, 5, 10, 20, 50]
N_PAIRS_DEFAULT = 100_000
N_RANDOM_SEEDS_DEFAULT = 10


def run_dataset(name, n_pairs, k_values, n_random_seeds):
    print(f"\n=== Dataset: {name} ===")
    X, V, pt, gene_names, gt_dict = load_dataset(name)
    G = len(gene_names)
    print(f"  G={G} cells={X.shape[0]} TFs(GT)={len(gt_dict)}")

    # Pair-space features
    t0 = time.perf_counter()
    F_X, F_V, _ = sample_pair_features(X, V, pt, n_pairs=n_pairs, seed=42)
    print(f"  pairs: {F_X.shape}  ({time.perf_counter() - t0:.1f}s)")

    # Precompute full SVD once for all k (truncation is cheap)
    t0 = time.perf_counter()
    _, _, Vt_svd = np.linalg.svd(F_V, full_matrices=False)
    print(f"  SVD done ({time.perf_counter() - t0:.1f}s)")

    all_rows = []

    # Identity (k=None) baseline — once per dataset
    t0 = time.perf_counter()
    scores_id = score_with_projection(F_X, F_V, P=None, normalize=True)
    df_id = evaluate_scores_per_tf(scores_id, gt_dict, gene_names)
    for _, r in df_id.iterrows():
        all_rows.append({"dataset": name, "method": "Identity",
                         "k": 0, "seed": -1, **r.to_dict()})
    print(f"  Identity  AUROC={df_id['auroc'].mean():.4f} "
          f"AUPRC={df_id['auprc'].mean():.4f}  "
          f"({time.perf_counter() - t0:.1f}s)")

    # Rank-k sweep
    for k in k_values:
        if k > G:
            continue

        # SVD
        P_svd = Vt_svd[:k].T
        scores_svd = score_with_projection(F_X, F_V, P_svd, normalize=True)
        df_svd = evaluate_scores_per_tf(scores_svd, gt_dict, gene_names)
        for _, r in df_svd.iterrows():
            all_rows.append({"dataset": name, "method": "SVD",
                             "k": k, "seed": -1, **r.to_dict()})

        # Random × seeds
        rand_auroc_means = []
        for seed in range(n_random_seeds):
            P_rand = projection_random(G, k, seed=seed)
            scores_rand = score_with_projection(F_X, F_V, P_rand, normalize=True)
            df_rand = evaluate_scores_per_tf(scores_rand, gt_dict, gene_names)
            rand_auroc_means.append(df_rand["auroc"].mean())
            for _, r in df_rand.iterrows():
                all_rows.append({"dataset": name, "method": "Random",
                                 "k": k, "seed": seed, **r.to_dict()})

        print(f"  k={k:3d}: SVD AUROC={df_svd['auroc'].mean():.4f} "
              f"AUPRC={df_svd['auprc'].mean():.4f}  "
              f"| Random AUROC={np.mean(rand_auroc_means):.4f} "
              f"± {np.std(rand_auroc_means):.4f}")

    return pd.DataFrame(all_rows)


def summarize(all_df, outdir):
    """Aggregate summary + paired Wilcoxon tests."""
    lines = ["# T1 — SVD vs Random Projection Ablation\n"]
    lines.append(f"- n_random_seeds per k: "
                 f"{all_df[all_df['method'] == 'Random']['seed'].nunique()}")
    lines.append("")

    summary_rows = []

    for ds in sorted(all_df["dataset"].unique()):
        sub = all_df[all_df["dataset"] == ds]
        lines.append(f"\n## {ds}\n")

        # Identity baseline
        id_sub = sub[sub["method"] == "Identity"]
        if len(id_sub):
            id_mean = id_sub["auroc"].mean()
            lines.append(f"- Identity (no projection) AUROC: {id_mean:.4f} "
                         f"AUPRC: {id_sub['auprc'].mean():.4f}\n")

        lines.append("| k | SVD AUROC | Random AUROC ± SE | "
                     "paired-Wilcoxon p | Δ median | SVD>Rand(k-normal) |")
        lines.append("|---|---|---|---|---|---|")

        for k in sorted(sub[sub["k"] > 0]["k"].unique()):
            svd_tfs = sub[(sub["k"] == k) & (sub["method"] == "SVD")]
            rnd_tfs = sub[(sub["k"] == k) & (sub["method"] == "Random")]
            if len(svd_tfs) == 0 or len(rnd_tfs) == 0:
                continue

            svd_per_tf = svd_tfs.groupby("tf")["auroc"].first()
            rnd_per_tf = rnd_tfs.groupby("tf")["auroc"].mean()

            common_tfs = svd_per_tf.index.intersection(rnd_per_tf.index)
            if len(common_tfs) < 5:
                continue

            # Paired Wilcoxon: per-TF, SVD vs mean-over-seeds Random
            try:
                _, p_pair = stats.wilcoxon(
                    svd_per_tf.loc[common_tfs].values,
                    rnd_per_tf.loc[common_tfs].values,
                    alternative="greater",
                )
            except ValueError:
                p_pair = np.nan

            # Across-seed SE (of mean Random AUROC)
            rand_seed_means = rnd_tfs.groupby("seed")["auroc"].mean()
            rand_mean = rand_seed_means.mean()
            rand_se = rand_seed_means.std() / np.sqrt(max(len(rand_seed_means), 1))

            # Pre-specified criterion: SVD > mean + 1.96·SE?
            crit = svd_per_tf.mean() > (rand_mean + 1.96 * rand_se)

            median_diff = (svd_per_tf.loc[common_tfs].values
                           - rnd_per_tf.loc[common_tfs].values).mean()

            lines.append(
                f"| {k} | {svd_per_tf.mean():.4f} | "
                f"{rand_mean:.4f} ± {rand_se:.4f} | "
                f"{p_pair:.3g} | {median_diff:+.4f} | "
                f"{'✓' if crit else '✗'} |"
            )

            summary_rows.append({
                "dataset": ds, "k": k,
                "svd_auroc": svd_per_tf.mean(),
                "random_auroc_mean": rand_mean,
                "random_auroc_se": rand_se,
                "paired_wilcoxon_p": p_pair,
                "auroc_diff_mean": median_diff,
                "criterion_pass": crit,
                "n_tfs": len(common_tfs),
            })

    # Decision block
    lines.append("\n## Pre-specified criterion evaluation (at k=5)\n")
    k5 = pd.DataFrame([r for r in summary_rows if r["k"] == 5])
    if len(k5):
        n_pass_crit = int(k5["criterion_pass"].sum())
        n_pass_wilc = int((k5["paired_wilcoxon_p"] < 0.05).sum())
        n_total = len(k5)
        lines.append(f"- SVD > Random+1.96·SE: **{n_pass_crit}/{n_total}** datasets "
                     f"(need ≥4)")
        lines.append(f"- Paired Wilcoxon p < 0.05: **{n_pass_wilc}/{n_total}** datasets "
                     f"(need ≥3)")

        verdict = ("**PASS**" if (n_pass_crit >= 4 and n_pass_wilc >= 3) else
                   "**FAIL — reframe paper**" if (n_pass_crit == 0) else
                   "**MIXED — interpret carefully**")
        lines.append(f"- Verdict: {verdict}\n")

    (outdir / "REPORT.md").write_text("\n".join(lines))
    pd.DataFrame(summary_rows).to_csv(outdir / "summary.csv", index=False)
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=SERGIO_DATASETS + ["bonemarrow", "pancreas"])
    ap.add_argument("--n-pairs", type=int, default=N_PAIRS_DEFAULT)
    ap.add_argument("--n-random-seeds", type=int, default=N_RANDOM_SEEDS_DEFAULT)
    ap.add_argument("--k-values", type=int, nargs="+", default=K_VALUES_DEFAULT)
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for ds in args.datasets:
        try:
            df = run_dataset(ds, args.n_pairs, args.k_values, args.n_random_seeds)
            df.to_csv(OUTDIR / f"{ds}_ablation.csv", index=False)
            frames.append(df)
        except Exception as e:
            print(f"  [ERROR] {ds}: {e}")
            import traceback; traceback.print_exc()

    if frames:
        full = pd.concat(frames, ignore_index=True)
        full.to_csv(OUTDIR / "all_ablation.csv", index=False)
        summarize(full, OUTDIR)


if __name__ == "__main__":
    main()
