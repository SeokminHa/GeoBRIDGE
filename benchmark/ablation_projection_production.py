"""
Production-faithful T1: SVD vs Random projection inside GeoBridgeModel.

Replaces the surrogate ablation in benchmark/ablation_projection.py.

Methodology:
  1. Use production `precompute_pairspace_stats_streaming` (unchanged).
  2. For each k ∈ K_VALUES:
       - SVD arm:    Vk = pre["Vt"][:k].T  (production default)
       - Random arm: Vk_rand = orthonormal(G, k, seed) × N_RANDOM_SEEDS
       - Both arms call `scores_with_custom_vk(pre, Vk, normalize_score=True)`
  3. project_y is FIXED to False for both arms to isolate the x-projection
     effect (random basis has no associated singular values for y-sharpening).
  4. A third reference arm reports SVD with project_y=True (production default),
     to quantify how much the y-projection adds.

Pre-specified success criterion (mirrors the surrogate but now on production):
  SVD(project_y=False) > Random(project_y=False) at k=5:
    - paired Wilcoxon p < 0.05 in ≥ 2/3 datasets run
    - mean Δ AUROC > 0.02 in those datasets

Usage:
  # Stage 1: SERGIO C sanity check (~3 min)
  python -u -m benchmark.ablation_projection_production --datasets C

  # Stage 2: SERGIO A/B/D (~10 min)
  python -u -m benchmark.ablation_projection_production --datasets A B D

  # Stage 3: real data (~1-2 hours)
  python -u -m benchmark.ablation_projection_production \
    --datasets bonemarrow pancreas

  # Everything at once
  python -u -m benchmark.ablation_projection_production \
    --datasets A B C D bonemarrow pancreas \
    --n-random-seeds 10

  # Skip identity-baseline assertion (for debugging when signal is known bad)
  python -u -m benchmark.ablation_projection_production --datasets B \
    --allow-antisignal-identity
"""
import sys
import argparse
import time
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from benchmark.utils_projection import (
    scores_with_custom_vk, random_orthonormal_vk, svd_vk,
    verify_match, load_for_production, precompute_pre,
    evaluate_scores_per_tf,
)
from geobridge_model.model import GeoBridgeModel

OUTDIR = Path("results/ablation_projection_production")

K_VALUES_DEFAULT = [1, 3, 5, 10, 20, 50]
N_RANDOM_SEEDS_DEFAULT = 10


def run_dataset(name, k_values, n_random_seeds, verify=True,
                allow_antisignal_identity=False):
    print(f"\n=== {name} ===")
    X, V, pt, gene_names, gt_dict = load_for_production(name)
    G = len(gene_names)
    print(f"  G={G}  cells={X.shape[0]}  TFs(GT)={len(gt_dict)}")

    # Production streaming stats
    t0 = time.perf_counter()
    pre = precompute_pre(X, V, pt, pair_seed=42)
    print(f"  precompute_pre done in {time.perf_counter() - t0:.1f}s")

    # One-time verification: custom-basis scoring must match production
    if verify:
        d = verify_match(pre, k=5, tol=1e-8)
        print(f"  [verify] custom Vk @ k=5 matches production, max|diff|={d:.2e}")

    rows = []

    # k = 0 (identity) -- reference (only one variant)
    scores0 = scores_with_custom_vk(pre, Vk=None, normalize_score=True)
    df0 = evaluate_scores_per_tf(scores0, gt_dict, gene_names)
    for _, r in df0.iterrows():
        rows.append({"dataset": name, "method": "Identity",
                     "project_y": False, "k": 0, "seed": -1, **r.to_dict()})
    id_auroc = df0["auroc"].mean()
    id_auprc = df0["auprc"].mean()
    id_sauprc = df0["sauprc"].mean()
    print(f"  [k=0 Identity] n_TFs={len(df0)}  "
          f"AUROC={id_auroc:.4f}  AUPRC={id_auprc:.4f}  "
          f"sAUPRC={id_sauprc:.2f}x baseline")

    # CRITICAL GATE: identity AUROC must be > 0.5 for the ablation to be
    # interpretable. If identity is anti-signal, SVD and Random are both
    # amplifying/preserving an inverted signal and results are meaningless.
    if id_auroc <= 0.5 and not allow_antisignal_identity:
        print(f"  [!!!] Identity AUROC = {id_auroc:.4f} ≤ 0.5 — base scoring is")
        print(f"        anti-signal or null on this dataset. Projection ablation")
        print(f"        is NOT interpretable. Check velocity / pseudotime / gene")
        print(f"        orientation first.  Use --allow-antisignal-identity to")
        print(f"        proceed anyway (flagged in the output).")
        return pd.DataFrame(rows), {"identity_auroc_gate_pass": False}

    # k sweep
    for k in k_values:
        if k > G:
            continue

        # SVD, project_y=False (fair arm)
        Vk_svd = svd_vk(pre, k)
        scores_svd = scores_with_custom_vk(pre, Vk_svd, normalize_score=True)
        df_svd = evaluate_scores_per_tf(scores_svd, gt_dict, gene_names)
        for _, r in df_svd.iterrows():
            rows.append({"dataset": name, "method": "SVD",
                         "project_y": False, "k": k, "seed": -1, **r.to_dict()})

        # SVD, project_y=True (production default; reference only)
        scores_svd_y = GeoBridgeModel.scores_from_streaming_stats(
            pre, k=k, project_y=True, normalize_score=True,
        )
        df_svd_y = evaluate_scores_per_tf(scores_svd_y, gt_dict, gene_names)
        for _, r in df_svd_y.iterrows():
            rows.append({"dataset": name, "method": "SVD_projectY",
                         "project_y": True, "k": k, "seed": -1, **r.to_dict()})

        # Random, project_y=False × seeds
        rand_aurocs = []
        for seed in range(n_random_seeds):
            Vk_rand = random_orthonormal_vk(G, k, seed=seed)
            scores_rand = scores_with_custom_vk(pre, Vk_rand, normalize_score=True)
            df_rand = evaluate_scores_per_tf(scores_rand, gt_dict, gene_names)
            rand_aurocs.append(df_rand["auroc"].mean())
            for _, r in df_rand.iterrows():
                rows.append({"dataset": name, "method": "Random",
                             "project_y": False, "k": k, "seed": seed, **r.to_dict()})

        # Build seed-mean random AUPRC/sAUPRC for the summary line
        rand_auprc_means = []
        rand_sauprc_means = []
        for seed in range(n_random_seeds):
            rseed_rows = [r for r in rows if r.get("seed") == seed
                          and r.get("method") == "Random" and r.get("k") == k]
            if rseed_rows:
                rand_auprc_means.append(np.mean([r["auprc"] for r in rseed_rows]))
                rand_sauprc_means.append(np.mean([r["sauprc"] for r in rseed_rows]))

        print(
            f"  k={k:3d}: "
            f"SVD(pyF) AUROC={df_svd['auroc'].mean():.4f} "
            f"AUPRC={df_svd['auprc'].mean():.4f} "
            f"sAUPRC={df_svd['sauprc'].mean():.2f}x  |  "
            f"SVD(pyT) AUROC={df_svd_y['auroc'].mean():.4f} "
            f"AUPRC={df_svd_y['auprc'].mean():.4f} "
            f"sAUPRC={df_svd_y['sauprc'].mean():.2f}x  |  "
            f"Random(pyF) AUROC={np.mean(rand_aurocs):.4f}±{np.std(rand_aurocs):.4f} "
            f"AUPRC={np.mean(rand_auprc_means):.4f} "
            f"sAUPRC={np.mean(rand_sauprc_means):.2f}x"
        )

    return pd.DataFrame(rows), {"identity_auroc_gate_pass": True,
                                 "identity_auroc": id_auroc,
                                 "identity_auprc": id_auprc}


def plot_auroc_vs_k(all_df, outdir):
    """AUROC vs k per dataset, with CI for random, for the paper."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [plot] matplotlib not available; skipping plots")
        return
    datasets = sorted(all_df["dataset"].unique())
    n_ds = len(datasets)
    n_cols = min(3, n_ds)
    n_rows = int(np.ceil(n_ds / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows),
                             squeeze=False)
    axes = axes.flatten()

    for ax, ds in zip(axes, datasets):
        sub = all_df[all_df["dataset"] == ds]

        # SVD pyF
        svd_tfs = sub[sub["method"] == "SVD"]
        svd_per_k = svd_tfs.groupby("k")["auroc"].mean().sort_index()
        if len(svd_per_k):
            ax.plot(svd_per_k.index, svd_per_k.values,
                    "o-", color="#1f77b4", label="SVD (project_y=F)")

        # SVD pyT
        svd_y_tfs = sub[sub["method"] == "SVD_projectY"]
        svd_y_per_k = svd_y_tfs.groupby("k")["auroc"].mean().sort_index()
        if len(svd_y_per_k):
            ax.plot(svd_y_per_k.index, svd_y_per_k.values,
                    "s--", color="#2ca02c", alpha=0.7,
                    label="SVD (project_y=T, production)")

        # Random with CI
        rnd_tfs = sub[sub["method"] == "Random"]
        if len(rnd_tfs):
            # For each k, compute mean over seeds first, then std of seed-means
            seed_means = rnd_tfs.groupby(["k", "seed"])["auroc"].mean().reset_index()
            agg = seed_means.groupby("k")["auroc"].agg(["mean", "std", "count"]).reset_index()
            agg["se"] = agg["std"] / np.sqrt(agg["count"])
            ax.errorbar(agg["k"], agg["mean"], yerr=1.96 * agg["se"],
                        fmt="o-", color="#d62728", label="Random (project_y=F, 95% CI)")

        # Identity
        id_tfs = sub[sub["method"] == "Identity"]
        if len(id_tfs):
            id_auroc = id_tfs["auroc"].mean()
            ax.axhline(id_auroc, ls=":", color="gray",
                       label=f"Identity k=0 ({id_auroc:.3f})")

        ax.axhline(0.5, ls="--", color="black", lw=0.5, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("k (rank of projected subspace)")
        ax.set_ylabel("per-TF AUROC (mean)")
        ax.set_title(ds)
        ax.legend(fontsize=8, loc="best")
        ax.grid(alpha=0.3)

    for ax in axes[len(datasets):]:
        ax.axis("off")

    fig.suptitle("T1 Production Ablation — AUROC vs k per dataset")
    fig.tight_layout()
    plot_path = outdir / "auroc_vs_k.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] {plot_path}")


def summarize(all_df, outdir):
    lines = ["# T1 Production — SVD vs Random (inside GeoBridgeModel)\n",
             "Scoring = `scores_with_custom_vk(pre, Vk, normalize_score=True)`\n",
             "project_y fixed to False for the fair SVD-vs-Random arms.\n",
             "SVD_projectY arm = production default (project_y=True) — reference.\n"]

    summary_rows = []

    for ds in sorted(all_df["dataset"].unique()):
        sub = all_df[all_df["dataset"] == ds]
        lines.append(f"\n## {ds}\n")

        id_sub = sub[sub["method"] == "Identity"]
        if len(id_sub):
            lines.append(f"- Identity (k=0) AUROC={id_sub['auroc'].mean():.4f}  "
                         f"AUPRC={id_sub['auprc'].mean():.4f}\n")

        lines.append("| k | SVD(pyF) | SVD(pyT) | Random(pyF) ± SE | "
                     "paired-Wilcoxon(SVD>Rand) p | Δ AUROC |")
        lines.append("|---|---|---|---|---|---|")

        for k in sorted(sub[sub["k"] > 0]["k"].unique()):
            svd_tfs = sub[(sub["k"] == k) & (sub["method"] == "SVD")]
            svdy_tfs = sub[(sub["k"] == k) & (sub["method"] == "SVD_projectY")]
            rnd_tfs = sub[(sub["k"] == k) & (sub["method"] == "Random")]

            if len(svd_tfs) == 0 or len(rnd_tfs) == 0:
                continue

            svd_per_tf = svd_tfs.groupby("tf")["auroc"].first()
            rnd_per_tf = rnd_tfs.groupby("tf")["auroc"].mean()
            common = svd_per_tf.index.intersection(rnd_per_tf.index)

            if len(common) >= 5:
                try:
                    _, p = stats.wilcoxon(
                        svd_per_tf.loc[common].values,
                        rnd_per_tf.loc[common].values,
                        alternative="greater",
                    )
                except ValueError:
                    p = np.nan
            else:
                p = np.nan

            rand_seed_means = rnd_tfs.groupby("seed")["auroc"].mean()
            rmean = rand_seed_means.mean()
            rse = rand_seed_means.std() / np.sqrt(max(len(rand_seed_means), 1))

            svd_mean = svd_per_tf.mean()
            svdy_mean = svdy_tfs["auroc"].mean() if len(svdy_tfs) else float("nan")
            delta = svd_mean - rmean

            lines.append(
                f"| {k} | {svd_mean:.4f} | {svdy_mean:.4f} | "
                f"{rmean:.4f} ± {rse:.4f} | {p:.3g} | {delta:+.4f} |"
            )

            summary_rows.append({
                "dataset": ds, "k": k,
                "svd_auroc": svd_mean,
                "svd_projectY_auroc": svdy_mean,
                "random_auroc_mean": rmean,
                "random_auroc_se": rse,
                "paired_wilcoxon_p": p,
                "delta_svd_minus_rand": delta,
                "n_tfs": len(common),
            })

    # Pre-specified criterion
    lines.append("\n## Pre-specified criterion at k=5\n")
    k5 = pd.DataFrame([r for r in summary_rows if r["k"] == 5])
    if len(k5):
        n_total = len(k5)
        n_sig = int((k5["paired_wilcoxon_p"] < 0.05).sum())
        n_eff = int((k5["delta_svd_minus_rand"] > 0.02).sum())
        threshold = int(np.ceil(2 * n_total / 3))
        lines.append(f"- Paired Wilcoxon p<0.05: **{n_sig}/{n_total}** (need ≥{threshold})")
        lines.append(f"- Δ AUROC > 0.02: **{n_eff}/{n_total}** (need ≥{threshold})")

        verdict = (
            "**PASS**" if (n_sig >= threshold and n_eff >= threshold)
            else "**FAIL — reframe pair-space novelty**" if (n_sig == 0)
            else "**MIXED — interpret carefully**"
        )
        lines.append(f"- Verdict: {verdict}\n")

    pd.DataFrame(summary_rows).to_csv(outdir / "summary.csv", index=False)
    (outdir / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--k-values", type=int, nargs="+", default=K_VALUES_DEFAULT)
    ap.add_argument("--n-random-seeds", type=int, default=N_RANDOM_SEEDS_DEFAULT)
    ap.add_argument("--skip-verify", action="store_true",
                    help="Skip custom-basis/production match check")
    ap.add_argument("--allow-antisignal-identity", action="store_true",
                    help="Proceed even if k=0 identity AUROC < 0.5 (ablation "
                         "results will be flagged as uninterpretable)")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    frames = []
    gates = {}
    for ds in args.datasets:
        try:
            df, gate = run_dataset(ds, args.k_values, args.n_random_seeds,
                                    verify=not args.skip_verify,
                                    allow_antisignal_identity=args.allow_antisignal_identity)
            gates[ds] = gate
            if len(df) > 0:
                df.to_csv(OUTDIR / f"{ds}_ablation.csv", index=False)
                frames.append(df)
        except Exception as e:
            print(f"[ERROR] {ds}: {e}")
            import traceback; traceback.print_exc()

    # Save gate results
    pd.DataFrame([{"dataset": ds, **g} for ds, g in gates.items()]).to_csv(
        OUTDIR / "identity_baseline_gates.csv", index=False,
    )

    if frames:
        full = pd.concat(frames, ignore_index=True)
        full.to_csv(OUTDIR / "all_ablation.csv", index=False)
        summarize(full, OUTDIR)
        plot_auroc_vs_k(full, OUTDIR)


if __name__ == "__main__":
    main()
