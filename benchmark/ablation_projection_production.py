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


def _build_signed_gt_for_dataset(name, gene_names, unsigned_gt):
    """
    Return a signed gt_dict {tf: {target: sign}} for this dataset.
    Falls back to {tf: {target: 1}} when sign info is unavailable.
    """
    from benchmark.utils import SERGIO_BASE, SERGIO_DATASETS, DATASET_ORGANISM
    from benchmark.bm_real_data_comparison import (
        _load_collectri_regulons, _build_signed_ground_truth,
        _load_sergio_signed_gt,
    )

    if name in SERGIO_DATASETS:
        gt_csv = SERGIO_BASE / "gt_GRN.csv"
        if gt_csv.exists():
            try:
                return _load_sergio_signed_gt(gene_names, gt_csv)
            except Exception as e:
                print(f"  [WARN] signed SERGIO GT fallback to unsigned: {e}")
        # fallback: wrap unsigned into all-+1
        return {tf: {t: 1 for t in tgts} for tf, tgts in unsigned_gt.items()}

    # Real (bonemarrow / pancreas)
    organism = DATASET_ORGANISM.get(name, "human")
    uppercase = (organism == "mouse")
    try:
        net = _load_collectri_regulons(organism=organism)
        return _build_signed_ground_truth(
            net, gene_names, uppercase_match=uppercase,
        )
    except Exception as e:
        print(f"  [WARN] signed CollecTRI GT fallback to unsigned: {e}")
        return {tf: {t: 1 for t in tgts} for tf, tgts in unsigned_gt.items()}


def run_dataset(name, k_values, n_random_seeds, verify=True,
                allow_antisignal_identity=False, use_signed_gt=True):
    print(f"\n=== {name} ===")
    X, V, pt, gene_names, gt_unsigned = load_for_production(name)
    if use_signed_gt:
        gt_dict = _build_signed_gt_for_dataset(name, gene_names, gt_unsigned)
    else:
        gt_dict = gt_unsigned
    G = len(gene_names)
    print(f"  G={G}  cells={X.shape[0]}  TFs(GT)={len(gt_dict)}  "
          f"signed={use_signed_gt}")

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
    id_std_auprc = df0["std_auprc"].mean()
    print(f"  [k=0 Identity] n_TFs={len(df0)}  "
          f"AUROC={id_auroc:.4f}  AUPRC={id_auprc:.4f}  "
          f"stdAUPRC={id_std_auprc:.2f}x baseline")

    # FLAG (not filter): identity AUROC is saved and gate-pass is flagged
    # on every row. Downstream analysis decides whether to include.
    gate_pass = bool(id_auroc > 0.5)
    if not gate_pass:
        print(f"  [WARN] Identity AUROC = {id_auroc:.4f} ≤ 0.5 — base scoring is")
        print(f"         anti-signal or null. Ablation will run but results on")
        print(f"         this dataset are NOT interpretable as SVD/Random evidence.")
        print(f"         Retained for transparency; gate_pass=False in CSV.")
        if not allow_antisignal_identity:
            # Legacy path: still return early if explicit flag not set
            # (preserves old behavior for scripts that rely on it)
            pass  # we now always continue
    # Tag the k=0 identity rows with gate status
    for r in rows:
        r["identity_auroc"] = id_auroc
        r["gate_pass"] = gate_pass

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
                         "project_y": False, "k": k, "seed": -1,
                         "identity_auroc": id_auroc, "gate_pass": gate_pass,
                         **r.to_dict()})

        # SVD, project_y=True (production default; reference only)
        scores_svd_y = GeoBridgeModel.scores_from_streaming_stats(
            pre, k=k, project_y=True, normalize_score=True,
        )
        df_svd_y = evaluate_scores_per_tf(scores_svd_y, gt_dict, gene_names)
        for _, r in df_svd_y.iterrows():
            rows.append({"dataset": name, "method": "SVD_projectY",
                         "project_y": True, "k": k, "seed": -1,
                         "identity_auroc": id_auroc, "gate_pass": gate_pass,
                         **r.to_dict()})

        # Random, project_y=False × seeds
        rand_aurocs = []
        for seed in range(n_random_seeds):
            Vk_rand = random_orthonormal_vk(G, k, seed=seed)
            scores_rand = scores_with_custom_vk(pre, Vk_rand, normalize_score=True)
            df_rand = evaluate_scores_per_tf(scores_rand, gt_dict, gene_names)
            rand_aurocs.append(df_rand["auroc"].mean())
            for _, r in df_rand.iterrows():
                rows.append({"dataset": name, "method": "Random",
                             "project_y": False, "k": k, "seed": seed,
                             "identity_auroc": id_auroc, "gate_pass": gate_pass,
                             **r.to_dict()})

        # Build seed-mean random AUPRC/stdAUPRC for the summary line
        rand_auprc_means = []
        rand_std_auprc_means = []
        for seed in range(n_random_seeds):
            rseed_rows = [r for r in rows if r.get("seed") == seed
                          and r.get("method") == "Random" and r.get("k") == k]
            if rseed_rows:
                rand_auprc_means.append(np.mean([r["auprc"] for r in rseed_rows]))
                rand_std_auprc_means.append(np.mean([r["std_auprc"] for r in rseed_rows]))

        def _fmt(df, col, fmt=".4f"):
            if col in df.columns and not df[col].isna().all():
                return format(df[col].mean(), fmt)
            return "-"

        print(
            f"  k={k:3d}:\n"
            f"    SVD(pyF)    AUROC={df_svd['auroc'].mean():.4f}  "
            f"AUPRC={df_svd['auprc'].mean():.4f}  "
            f"stdAUPRC={df_svd['std_auprc'].mean():.2f}x  |  "
            f"signed_pos_AUPRC={_fmt(df_svd, 'auprc_signed_pos')}  "
            f"signed_neg_AUPRC={_fmt(df_svd, 'auprc_signed_neg')}  "
            f"dir_acc={_fmt(df_svd, 'dir_acc', '.3f')}\n"
            f"    SVD(pyT)    AUROC={df_svd_y['auroc'].mean():.4f}  "
            f"AUPRC={df_svd_y['auprc'].mean():.4f}  "
            f"stdAUPRC={df_svd_y['std_auprc'].mean():.2f}x  |  "
            f"signed_pos_AUPRC={_fmt(df_svd_y, 'auprc_signed_pos')}  "
            f"signed_neg_AUPRC={_fmt(df_svd_y, 'auprc_signed_neg')}  "
            f"dir_acc={_fmt(df_svd_y, 'dir_acc', '.3f')}\n"
            f"    Random(pyF) AUROC={np.mean(rand_aurocs):.4f}±{np.std(rand_aurocs):.4f}  "
            f"AUPRC={np.mean(rand_auprc_means):.4f}  "
            f"stdAUPRC={np.mean(rand_std_auprc_means):.2f}x"
        )

    return pd.DataFrame(rows), {"gate_pass": gate_pass,
                                 "identity_auroc": id_auroc,
                                 "identity_auprc": id_auprc,
                                 "identity_std_auprc": id_std_auprc}


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


def _tf_bootstrap_delta_ci(svd_per_tf, rand_tfs_df, n_boot=1000, seed=0):
    """
    Bootstrap a 95% CI for mean Δ AUROC (SVD - Random) by resampling TFs
    with replacement, drawing ONE random-seed observation per sampled TF.

    This is NOT a full joint (TF × seed) bootstrap; it captures TF-level
    variability plus a single seed draw per TF. CI is narrower than a full
    joint bootstrap would give. Use for directional evidence, not precise
    coverage claims.

    Returns (mean_delta, lo, hi).
    """
    rng = np.random.default_rng(seed)
    tfs = svd_per_tf.index.tolist()
    if len(tfs) < 3:
        return float("nan"), float("nan"), float("nan")

    # Build per-(tf, seed) random AUROC table
    rnd_pivot = rand_tfs_df.pivot_table(
        index="tf", columns="seed", values="auroc", aggfunc="first"
    )
    # Restrict to common TFs
    common = [t for t in tfs if t in rnd_pivot.index]
    if len(common) < 3:
        return float("nan"), float("nan"), float("nan")
    svd_vec = svd_per_tf.loc[common].values

    deltas = []
    for _ in range(n_boot):
        # Resample TFs with replacement
        sampled_tfs_idx = rng.integers(0, len(common), size=len(common))
        svd_b = svd_vec[sampled_tfs_idx]
        rnd_b = rnd_pivot.loc[[common[i] for i in sampled_tfs_idx]].values
        # For each sampled TF, pick a random seed
        seed_idx = rng.integers(0, rnd_b.shape[1], size=rnd_b.shape[0])
        rnd_sampled = rnd_b[np.arange(rnd_b.shape[0]), seed_idx]
        deltas.append((svd_b - rnd_sampled).mean())
    arr = np.array(deltas)
    return float(arr.mean()), float(np.quantile(arr, 0.025)), float(np.quantile(arr, 0.975))


def _compute_three_wilcoxon(svd_per_tf, rand_tfs_df):
    """
    (a) seed-mean: Wilcoxon(SVD[tf], mean_seeds(R[tf, s]))
    (b) seed-max:  Wilcoxon(SVD[tf], max_seeds(R[tf, s]))
    (c) per-TF win-rate: for each TF, fraction of seeds SVD > R; Wilcoxon vs 0.5
    """
    out = {"p_seedmean": np.nan, "p_seedmax": np.nan, "p_winrate": np.nan,
           "mean_winrate": np.nan}

    rnd_pivot = rand_tfs_df.pivot_table(
        index="tf", columns="seed", values="auroc", aggfunc="first"
    )
    common = [t for t in svd_per_tf.index if t in rnd_pivot.index]
    if len(common) < 3:
        return out

    svd_vec = svd_per_tf.loc[common].values
    r_mat = rnd_pivot.loc[common].values  # shape (n_tf, n_seeds)

    try:
        _, p_a = stats.wilcoxon(svd_vec, r_mat.mean(axis=1),
                                alternative="greater")
    except ValueError:
        p_a = np.nan
    try:
        _, p_b = stats.wilcoxon(svd_vec, r_mat.max(axis=1),
                                alternative="greater")
    except ValueError:
        p_b = np.nan

    # Win-rate per TF
    winrates = ((svd_vec[:, None] > r_mat).sum(axis=1) / r_mat.shape[1])
    try:
        _, p_c = stats.wilcoxon(winrates - 0.5, alternative="greater")
    except ValueError:
        p_c = np.nan

    out.update({
        "p_seedmean": p_a, "p_seedmax": p_b, "p_winrate": p_c,
        "mean_winrate": float(winrates.mean()),
    })
    return out


def summarize(all_df, outdir):
    lines = [
        "# T1 Production — SVD vs Random (inside GeoBridgeModel)\n",
        "Scoring: `scores_with_custom_vk(pre, Vk, normalize_score=True)`\n",
        "Fair comparison uses project_y=False for both SVD and Random arms.\n",
        "SVD_projectY arm = production default (project_y=True), reference only.\n",
        "",
        "## Column definitions",
        "- **stdAUPRC**: AUPRC / baseline_AUPRC (standardized; NOT sign-aware).",
        "- **signed_pos_AUPRC**: AUPRC using signed scores, positive edges as "
        "positive class, negative edges EXCLUDED, non-edges as negative class.",
        "- **signed_neg_AUPRC**: AUPRC using negated signed scores, negative "
        "edges as positive class, positive edges EXCLUDED, non-edges as "
        "negative class.",
        "- **dir_acc**: among sign-labeled GT edges, fraction where "
        "sign(score) == sign(GT).",
        "",
        "## Statistical tests reported",
        "- **p (seed-mean)**: paired Wilcoxon SVD[tf] vs mean_seeds(Random[tf, s]) — primary, anti-conservative.",
        "- **p (seed-max)**: paired Wilcoxon SVD[tf] vs max_seeds(Random[tf, s]) — stress test.",
        "- **p (win-rate)**: Wilcoxon of per-TF win-rates (fraction of seeds SVD > Random) against 0.5.",
        "- **Δ AUROC 95% CI**: TF-resample bootstrap with one random seed draw "
        "per sampled TF (narrower than full joint (TF×seed) bootstrap; "
        "directional evidence only).",
        "",
    ]

    summary_rows = []
    gate_pass_datasets = set()
    gate_fail_datasets = set()
    for ds in sorted(all_df["dataset"].unique()):
        gp = all_df[all_df["dataset"] == ds]["gate_pass"].iloc[0]
        (gate_pass_datasets if gp else gate_fail_datasets).add(ds)

    # Process datasets; split output by gate-pass vs gate-fail
    for section_name, ds_set in [("Gate-PASS datasets", gate_pass_datasets),
                                  ("Gate-FAIL datasets (diagnostic only)",
                                   gate_fail_datasets)]:
        if not ds_set:
            continue
        lines.append(f"\n# {section_name}\n")

        for ds in sorted(ds_set):
            sub = all_df[all_df["dataset"] == ds]
            id_sub = sub[sub["method"] == "Identity"]
            id_auroc = id_sub["auroc"].mean() if len(id_sub) else float("nan")
            id_std_auprc = id_sub["std_auprc"].mean() if (len(id_sub) and "std_auprc" in id_sub.columns) else float("nan")
            lines.append(f"\n## {ds}  (identity AUROC = {id_auroc:.4f}, "
                         f"stdAUPRC = {id_std_auprc:.2f}×)\n")
            lines.append(
                "| k | SVD(pyF) AUROC | SVD(pyT) AUROC | "
                "Random mean AUROC | Δ (95% CI) | p(seed-mean) | p(seed-max) | p(win-rate) | "
                "SVD signed_pos_AUPRC | SVD signed_neg_AUPRC | SVD dir_acc | "
                "SVD stdAUPRC | Rand stdAUPRC |"
            )
            lines.append("|" + "---|" * 13)

            for k in sorted(sub[sub["k"] > 0]["k"].unique()):
                svd_tfs = sub[(sub["k"] == k) & (sub["method"] == "SVD")]
                svdy_tfs = sub[(sub["k"] == k) & (sub["method"] == "SVD_projectY")]
                rnd_tfs = sub[(sub["k"] == k) & (sub["method"] == "Random")]
                if len(svd_tfs) == 0 or len(rnd_tfs) == 0:
                    continue

                svd_per_tf = svd_tfs.groupby("tf")["auroc"].first()
                tests = _compute_three_wilcoxon(svd_per_tf, rnd_tfs)
                dmean, dlo, dhi = _tf_bootstrap_delta_ci(svd_per_tf, rnd_tfs)

                rand_mean_auroc = rnd_tfs.groupby("seed")["auroc"].mean().mean()
                svd_mean = svd_per_tf.mean()
                svdy_mean = svdy_tfs["auroc"].mean() if len(svdy_tfs) else float("nan")

                svd_std_auprc = svd_tfs["std_auprc"].mean() if "std_auprc" in svd_tfs.columns else float("nan")
                rand_std_auprc = rnd_tfs["std_auprc"].mean() if "std_auprc" in rnd_tfs.columns else float("nan")

                def _mean_or_nan(df, col):
                    if col in df.columns and not df[col].isna().all():
                        return df[col].mean()
                    return float("nan")

                svd_ap_pos = _mean_or_nan(svd_tfs, "auprc_signed_pos")
                svd_ap_neg = _mean_or_nan(svd_tfs, "auprc_signed_neg")
                svd_dir    = _mean_or_nan(svd_tfs, "dir_acc")

                def _fmtf(v, fmt=".4f"):
                    return format(v, fmt) if not np.isnan(v) else "-"

                lines.append(
                    f"| {k} | {svd_mean:.4f} | {svdy_mean:.4f} | "
                    f"{rand_mean_auroc:.4f} | {dmean:+.4f} [{dlo:+.4f}, {dhi:+.4f}] | "
                    f"{tests['p_seedmean']:.3g} | {tests['p_seedmax']:.3g} | "
                    f"{tests['p_winrate']:.3g} | "
                    f"{_fmtf(svd_ap_pos)} | {_fmtf(svd_ap_neg)} | {_fmtf(svd_dir, '.3f')} | "
                    f"{svd_std_auprc:.2f}× | {rand_std_auprc:.2f}× |"
                )

                summary_rows.append({
                    "dataset": ds, "k": k, "gate_pass": ds in gate_pass_datasets,
                    "identity_auroc": id_auroc,
                    "svd_auroc": svd_mean, "svd_projectY_auroc": svdy_mean,
                    "random_auroc_mean": rand_mean_auroc,
                    "svd_std_auprc": svd_std_auprc, "random_std_auprc": rand_std_auprc,
                    "svd_auprc_signed_pos": svd_ap_pos,
                    "svd_auprc_signed_neg": svd_ap_neg,
                    "svd_dir_acc": svd_dir,
                    "delta_auroc": dmean, "delta_ci95_lo": dlo, "delta_ci95_hi": dhi,
                    "p_seedmean": tests["p_seedmean"],
                    "p_seedmax": tests["p_seedmax"],
                    "p_winrate": tests["p_winrate"],
                    "mean_winrate": tests["mean_winrate"],
                    "n_tfs": int(len(svd_per_tf)),
                })

    # Pre-specified criterion, applied to GATE-PASS datasets only
    lines.append("\n## Pre-specified criterion at k=5 (gate-pass only)\n")
    k5 = pd.DataFrame([r for r in summary_rows
                       if r["k"] == 5 and r["gate_pass"]])
    lines.append(f"Gate pass: **{len(gate_pass_datasets)}/{len(gate_pass_datasets) + len(gate_fail_datasets)}** datasets "
                 f"(pass: {sorted(gate_pass_datasets)}, fail: {sorted(gate_fail_datasets)})")
    if len(k5):
        n_total = len(k5)
        n_sig_mean = int((k5["p_seedmean"] < 0.05).sum())
        n_sig_max = int((k5["p_seedmax"] < 0.05).sum())
        n_eff = int((k5["delta_auroc"] > 0.02).sum())
        n_large_eff = int((k5["delta_auroc"] > 0.05).sum())
        threshold = max(1, int(np.ceil(2 * n_total / 3)))
        lines.append(f"- p(seed-mean) < 0.05: **{n_sig_mean}/{n_total}**  "
                     f"(need ≥{threshold} for majority)")
        lines.append(f"- p(seed-max)  < 0.05: **{n_sig_max}/{n_total}**  (stress test)")
        lines.append(f"- Δ AUROC > 0.02: **{n_eff}/{n_total}**")
        lines.append(f"- Δ AUROC > 0.05: **{n_large_eff}/{n_total}**")

        if n_sig_mean >= threshold and n_eff >= threshold:
            verdict = "**PASS**"
        elif n_sig_mean == 0 and n_eff == 0:
            verdict = "**FAIL — reframe pair-space novelty**"
        else:
            verdict = "**MIXED — interpret carefully**"
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
    ap.add_argument("--no-signed-gt", action="store_true",
                    help="Disable signed GT (use unsigned only). Signed GT is "
                         "default and uses CollecTRI sign_decision (real) or "
                         "SERGIO sign column if available (else all-positive).")
    args = ap.parse_args()

    OUTDIR.mkdir(parents=True, exist_ok=True)
    frames = []
    gates = {}
    for ds in args.datasets:
        try:
            df, gate = run_dataset(ds, args.k_values, args.n_random_seeds,
                                    verify=not args.skip_verify,
                                    allow_antisignal_identity=args.allow_antisignal_identity,
                                    use_signed_gt=not args.no_signed_gt)
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
