"""
T2 — GeoBRIDGE-only stability under cell subsampling.

Design:
  For each dataset, run GeoBRIDGE scoring on N_BOOTSTRAPS independent
  80%-cell subsamples. Extract top-50 predicted targets per TF from each
  run, compute pairwise Jaccard across subsamples, average per TF.
  Compare against a NULL baseline: same procedure but with velocity
  permuted across cells (breaks V↔X coupling).

IMPORTANT:
  - No arbitrary Jaccard thresholds. All claims are RELATIVE to null.
  - Uses the same simplified GeoBRIDGE scoring as utils.score_with_projection
    (cross-covariance + rank-k SVD projection). This gives a self-consistent
    comparison against null; absolute numbers may differ from the production
    GeoBridgeModel but relative stability conclusions transfer.

Usage:
  python -u -m benchmark.gb_stability --datasets bonemarrow pancreas D
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
    score_with_projection, top_k_targets_per_tf, pairwise_jaccard_per_tf,
)

OUTDIR = Path("results/gb_stability")


def gb_scores(X, V, pt, k=5, n_pairs=100_000, pair_seed=42):
    """Run simplified GeoBRIDGE scoring and return (G, G) abs scores."""
    F_X, F_V, _ = sample_pair_features(X, V, pt, n_pairs=n_pairs, seed=pair_seed)
    _, _, Vt = np.linalg.svd(F_V, full_matrices=False)
    P = Vt[: min(k, F_V.shape[1])].T
    return score_with_projection(F_X, F_V, P=P, normalize=True)


def run_bootstrap(name, n_bootstraps, subsample_frac, k, top_k,
                  include_null=True):
    """
    Returns (top_sets_gb, top_sets_null, gt_dict, gene_names)
    """
    print(f"\n=== {name} (n={n_bootstraps}, frac={subsample_frac}, "
          f"k={k}, top_K={top_k}) ===")

    X, V, pt, gene_names, gt_dict = load_dataset(name)
    n_cells = X.shape[0]
    n_keep = int(n_cells * subsample_frac)
    print(f"  cells: {n_cells} → subsample {n_keep}")
    print(f"  TFs in GT: {len(gt_dict)}")

    top_gb, top_null = [], []

    for i in range(n_bootstraps):
        rng = np.random.default_rng(42 + i)
        idx = np.sort(rng.choice(n_cells, size=n_keep, replace=False))
        Xs, Vs, pts = X[idx], V[idx], pt[idx]

        t0 = time.perf_counter()
        scores = gb_scores(Xs, Vs, pts, k=k, pair_seed=100 + i)
        top_gb.append(top_k_targets_per_tf(scores, gt_dict, gene_names, top_k))
        print(f"  [GB]   bootstrap {i+1}/{n_bootstraps}: "
              f"{time.perf_counter() - t0:.1f}s")

        if include_null:
            # NULL: permute velocity rows to break V↔X coupling
            perm = rng.permutation(n_keep)
            Vs_null = Vs[perm]
            t0 = time.perf_counter()
            scores_n = gb_scores(Xs, Vs_null, pts, k=k, pair_seed=100 + i)
            top_null.append(top_k_targets_per_tf(scores_n, gt_dict, gene_names,
                                                 top_k))
            print(f"  [NULL] bootstrap {i+1}/{n_bootstraps}: "
                  f"{time.perf_counter() - t0:.1f}s")

    return top_gb, top_null, gt_dict, gene_names


def summarize(df, outdir):
    lines = ["# T2 — GeoBRIDGE Stability\n"]
    lines.append("All claims are RELATIVE to NULL (velocity-permuted baseline).\n")

    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        lines.append(f"\n## {ds}\n")

        gb_vals = sub[sub["method"] == "GB"]["mean_jaccard"].dropna().values
        null_vals = sub[sub["method"] == "Null"]["mean_jaccard"].dropna().values

        if len(gb_vals) == 0:
            lines.append("  (no GB data)")
            continue

        lines.append(f"- GB   n={len(gb_vals):3d}  "
                     f"mean={gb_vals.mean():.4f}  median={np.median(gb_vals):.4f}  "
                     f"sd={gb_vals.std():.4f}")
        if len(null_vals):
            lines.append(f"- Null n={len(null_vals):3d}  "
                         f"mean={null_vals.mean():.4f}  "
                         f"median={np.median(null_vals):.4f}  "
                         f"sd={null_vals.std():.4f}")

            # Paired per-TF
            gb_per_tf = sub[sub["method"] == "GB"].set_index("tf")["mean_jaccard"]
            nl_per_tf = sub[sub["method"] == "Null"].set_index("tf")["mean_jaccard"]
            common = gb_per_tf.index.intersection(nl_per_tf.index)
            g = gb_per_tf.loc[common].values
            n = nl_per_tf.loc[common].values
            valid = ~(np.isnan(g) | np.isnan(n))
            g, n = g[valid], n[valid]
            if len(g) >= 5:
                try:
                    _, p = stats.wilcoxon(g, n, alternative="greater")
                except ValueError:
                    p = np.nan
                lines.append(f"- paired Wilcoxon (GB > Null) p = **{p:.3g}** "
                             f"({len(g)} TFs, Δmean = {(g-n).mean():+.4f})")

    (outdir / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["bonemarrow", "pancreas", "D"])
    ap.add_argument("--n-bootstraps", type=int, default=10)
    ap.add_argument("--subsample-frac", type=float, default=0.8)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--top-k", type=int, default=50)
    ap.add_argument("--skip-null", action="store_true")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    rows = []

    for ds in args.datasets:
        try:
            gb_list, null_list, _, _ = run_bootstrap(
                ds, args.n_bootstraps, args.subsample_frac,
                args.k, args.top_k, include_null=not args.skip_null)
            jac_gb = pairwise_jaccard_per_tf(gb_list)
            jac_null = pairwise_jaccard_per_tf(null_list) if null_list else {}
            for tf, j in jac_gb.items():
                rows.append({"dataset": ds, "tf": tf,
                             "method": "GB", "mean_jaccard": j})
            for tf, j in jac_null.items():
                rows.append({"dataset": ds, "tf": tf,
                             "method": "Null", "mean_jaccard": j})
        except Exception as e:
            print(f"[ERROR] {ds}: {e}")
            import traceback; traceback.print_exc()

    df = pd.DataFrame(rows)
    df.to_csv(OUTDIR / "stability_per_tf.csv", index=False)
    summarize(df, OUTDIR)


if __name__ == "__main__":
    main()
