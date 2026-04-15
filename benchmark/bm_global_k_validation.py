"""
Final validation: does a moderate global k consistently improve
discriminative signal over k=0 across ground truth databases?

Candidates:
  - k=0   (baseline: no projection)
  - k=3   (small fixed)
  - k=5   (moderate fixed, was bootstrap global AUPRC k_opt)
  - k=tail (tail-exhaustion auto-k from the spectrum)
  - k=10  (larger fixed)

Discriminative metrics per TF per k per database:
  - Cohen's d (target vs non-target score separation)
  - AUROC (target vs non-target classification)
  - AUPRC (for reference, but NOT the optimization target)

Decision criterion:
  If a moderate global k (tail-exhaustion or small fixed) shows
  consistent discriminative improvement over k=0 across databases,
  the k story is sufficiently complete.

Outputs:
  results/bm_global_k_validation/csv/metrics_per_db_k_tf.csv
  results/bm_global_k_validation/csv/summary_by_k.csv
  results/bm_global_k_validation/figures/...
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from geobridge_model import GeoBridgeModel
from geobridge_model.spectral_k import select_k_by_tail_exhaustion

KEY_TFS = ["ERG", "FLI1", "GATA2", "SPI1", "KLF1", "MYB", "MEIS1"]

# Fixed candidates + tail-exhaustion (computed from spectrum)
FIXED_K_CANDIDATES = [0, 1, 2, 3, 5, 7, 10, 15]


# =====================================================================
# Data loading
# =====================================================================

def load_and_preprocess(data_dir):
    """Load bone marrow, preprocess, compute velocity."""
    import warnings
    warnings.filterwarnings("ignore")
    import scanpy as sc
    import scvelo as scv

    cache = Path(data_dir) / "human_cd34_bone_marrow.h5ad" if data_dir else None
    if cache and cache.exists():
        adata = sc.read_h5ad(cache)
    else:
        adata = scv.datasets.bonemarrow()
        if cache:
            adata.write_h5ad(cache)

    scv.pp.filter_genes(adata, min_shared_counts=20)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

    print("[BM] Computing stochastic velocity...")
    t0 = time.perf_counter()
    scv.tl.velocity(adata, mode="stochastic")
    print(f"[BM] Velocity done in {time.perf_counter()-t0:.0f}s")

    X = np.asarray(adata.layers["Ms"], dtype=float)
    V = np.asarray(adata.layers["velocity"], dtype=float)
    V = np.nan_to_num(V, nan=0.0)
    genes = list(adata.var_names)

    if "palantir_pseudotime" in adata.obs.columns:
        pt = np.asarray(adata.obs["palantir_pseudotime"], dtype=float)
    else:
        sc.tl.diffmap(adata)
        if "CD34" in genes:
            adata.uns["iroot"] = np.argmax(X[:, genes.index("CD34")])
        else:
            adata.uns["iroot"] = 0
        sc.tl.dpt(adata)
        pt = np.asarray(adata.obs["dpt_pseudotime"], dtype=float)
    pt = np.nan_to_num(pt, nan=float(np.nanmedian(pt)))

    print(f"[BM] {X.shape[0]} cells x {X.shape[1]} genes")
    return X, V, pt, genes


def load_ground_truths(genes, min_targets=5):
    """Load CollecTRI and DoRothEA."""
    import decoupler as dc

    gene_set = set(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    def _build(net, name):
        tf_targets = {}
        for tf in net["source"].unique():
            if tf not in gene_set:
                continue
            targets = net[net["source"] == tf]["target"].values
            matched = [gene_to_idx[t] for t in targets if t in gene_set and t != tf]
            if len(matched) >= min_targets:
                tf_targets[tf] = matched
        print(f"  [{name}] {len(net)} edges -> {len(tf_targets)} evaluable TFs")
        return tf_targets

    databases = {}

    # CollecTRI
    try:
        if hasattr(dc, "op") and hasattr(dc.op, "collectri"):
            net = dc.op.collectri(organism="human")
        else:
            net = dc.get_collectri(organism="human", split_complexes=False)
        databases["CollecTRI"] = _build(net, "CollecTRI")
    except Exception as e:
        print(f"  [CollecTRI] FAILED: {e}")

    # DoRothEA (full A+B+C)
    try:
        if hasattr(dc, "op") and hasattr(dc.op, "dorothea"):
            net = dc.op.dorothea(organism="human")
        else:
            net = dc.get_dorothea(organism="human")
        databases["DoRothEA_ABC"] = _build(net, "DoRothEA_ABC")
        # High confidence A+B
        net_ab = net[net["confidence"].isin(["A", "B"])]
        databases["DoRothEA_AB"] = _build(net_ab, "DoRothEA_AB")
    except Exception as e:
        print(f"  [DoRothEA] FAILED: {e}")

    return databases, gene_to_idx


# =====================================================================
# Metrics
# =====================================================================

def compute_tf_metrics(S_score, tf_idx, target_idxs, G):
    """Compute AUPRC, AUROC, Cohen's d for one TF at one k."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    target_set = set(target_idxs)
    target_scores = []
    nontarget_scores = []
    all_scores = []
    all_labels = []

    for j in range(G):
        if j == tf_idx:
            continue
        s = abs(S_score[tf_idx, j])
        lab = 1 if j in target_set else 0
        all_scores.append(s)
        all_labels.append(lab)
        if lab == 1:
            target_scores.append(s)
        else:
            nontarget_scores.append(s)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    target_scores = np.array(target_scores)
    nontarget_scores = np.array(nontarget_scores)

    if all_labels.sum() == 0 or all_labels.sum() == len(all_labels):
        return None

    auprc = average_precision_score(all_labels, all_scores)
    try:
        auroc = roc_auc_score(all_labels, all_scores)
    except ValueError:
        auroc = np.nan

    # Cohen's d
    pooled_std = np.sqrt(
        (target_scores.var() * len(target_scores)
         + nontarget_scores.var() * len(nontarget_scores))
        / (len(target_scores) + len(nontarget_scores))
    )
    cohens_d = ((target_scores.mean() - nontarget_scores.mean()) / pooled_std
                if pooled_std > 1e-15 else 0.0)

    return {
        "auprc": float(auprc),
        "auroc": float(auroc),
        "cohens_d": float(cohens_d),
        "mean_target_score": float(target_scores.mean()),
        "mean_nontarget_score": float(nontarget_scores.mean()),
    }


# =====================================================================
# Main
# =====================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Validate global k via discriminative metrics")
    ap.add_argument("--data-dir", default="E:/data_cache")
    ap.add_argument("--outdir", default="results/bm_global_k_validation")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    csvdir = outdir / "csv"
    figdir = outdir / "figures"
    csvdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # ------------------------------------------------------------------
    # Load
    # ------------------------------------------------------------------
    X, V, pt, genes = load_and_preprocess(args.data_dir)
    G = len(genes)
    print("\n[GT] Loading ground truth databases...")
    databases, gene_to_idx = load_ground_truths(genes)

    # ------------------------------------------------------------------
    # Streaming stats
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("COMPUTING PAIRSPACE STATS")
    print("=" * 90)

    sort_order = np.argsort(pt)
    t_pre = time.perf_counter()
    pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
        X[sort_order], V[sort_order], pt[sort_order], pair_seed=42,
    )
    print(f"[PRE] {time.perf_counter()-t_pre:.1f}s, {pre['n_pairs']} pairs")

    # ------------------------------------------------------------------
    # Tail-exhaustion k
    # ------------------------------------------------------------------
    S = pre["S"]
    spec = select_k_by_tail_exhaustion(S)
    k_tail = spec.k_selected
    print(f"[TAIL] Tail-exhaustion k = {k_tail}")
    print(f"[TAIL] tail_baseline = {spec.tail_baseline:.4f}, "
          f"threshold = {spec.threshold:.4f}")

    # Build full candidate list
    k_candidates = sorted(set(FIXED_K_CANDIDATES) | {k_tail})
    k_labels = {}
    for k in k_candidates:
        labels = []
        if k == 0:
            labels.append("baseline")
        if k == k_tail:
            labels.append("tail-exhaust")
        k_labels[k] = f"k={k}" + (f" ({','.join(labels)})" if labels else "")

    print(f"[K] Candidates: {[k_labels[k] for k in k_candidates]}")

    # ------------------------------------------------------------------
    # Compute scores and evaluate
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("EVALUATING ALL CANDIDATES")
    print("=" * 90)

    all_rows = []
    for k in k_candidates:
        if k >= len(S):
            continue
        S_k = GeoBridgeModel.scores_from_streaming_stats(
            pre, k=k, project_y=True, normalize_score=True, eps=1e-12,
        )

        for db_name, tf_targets in databases.items():
            for tf, target_idxs in tf_targets.items():
                tf_idx = gene_to_idx[tf]
                metrics = compute_tf_metrics(S_k, tf_idx, target_idxs, G)
                if metrics is None:
                    continue
                metrics["k"] = k
                metrics["k_label"] = k_labels[k]
                metrics["database"] = db_name
                metrics["tf"] = tf
                metrics["n_targets"] = len(target_idxs)
                metrics["is_key_tf"] = tf in KEY_TFS
                metrics["is_tail_k"] = (k == k_tail)
                all_rows.append(metrics)

        print(f"  k={k:2d} done")

    df = pd.DataFrame(all_rows)
    df.to_csv(csvdir / "metrics_per_db_k_tf.csv", index=False)
    print(f"\n[SAVE] metrics_per_db_k_tf.csv ({len(df)} rows)")

    # ------------------------------------------------------------------
    # Core analysis: improvement over k=0
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("IMPROVEMENT OVER k=0 BY DATABASE AND CANDIDATE k")
    print("=" * 90)

    # For each (database, k), compute fraction of TFs improved over k=0
    summary_rows = []
    for db_name in sorted(databases.keys()):
        db_data = df[df["database"] == db_name]
        baseline = db_data[db_data["k"] == 0][["tf", "auroc", "cohens_d", "auprc"]].rename(
            columns={"auroc": "auroc_0", "cohens_d": "d_0", "auprc": "auprc_0"})

        for k in k_candidates:
            if k == 0:
                continue
            kdata = db_data[db_data["k"] == k][["tf", "auroc", "cohens_d", "auprc"]].rename(
                columns={"auroc": "auroc_k", "cohens_d": "d_k", "auprc": "auprc_k"})
            merged = baseline.merge(kdata, on="tf")
            if len(merged) == 0:
                continue

            n_tfs = len(merged)
            # AUROC improvement
            auroc_improved = (merged["auroc_k"] > merged["auroc_0"]).sum()
            auroc_mean_delta = (merged["auroc_k"] - merged["auroc_0"]).mean()
            auroc_median_delta = (merged["auroc_k"] - merged["auroc_0"]).median()
            # Cohen's d improvement
            d_improved = (merged["d_k"] > merged["d_0"]).sum()
            d_mean_delta = (merged["d_k"] - merged["d_0"]).mean()
            d_median_delta = (merged["d_k"] - merged["d_0"]).median()
            # AUPRC improvement (for reference)
            auprc_improved = (merged["auprc_k"] > merged["auprc_0"]).sum()
            auprc_mean_delta = (merged["auprc_k"] - merged["auprc_0"]).mean()

            # Sign test p-value for AUROC improvement
            auroc_deltas = (merged["auroc_k"] - merged["auroc_0"]).values
            n_pos = (auroc_deltas > 0).sum()
            n_neg = (auroc_deltas < 0).sum()
            n_test = n_pos + n_neg
            sign_p = sp_stats.binom_test(n_pos, n_test, 0.5) if n_test > 0 else 1.0

            # Wilcoxon signed-rank for d improvement
            d_deltas = (merged["d_k"] - merged["d_0"]).values
            nonzero = d_deltas[d_deltas != 0]
            if len(nonzero) >= 5:
                wstat, wilcox_p = sp_stats.wilcoxon(nonzero)
            else:
                wilcox_p = 1.0

            summary_rows.append({
                "database": db_name,
                "k": k,
                "k_label": k_labels[k],
                "n_tfs": n_tfs,
                "auroc_frac_improved": auroc_improved / n_tfs,
                "auroc_mean_delta": auroc_mean_delta,
                "auroc_median_delta": auroc_median_delta,
                "auroc_sign_p": sign_p,
                "d_frac_improved": d_improved / n_tfs,
                "d_mean_delta": d_mean_delta,
                "d_median_delta": d_median_delta,
                "d_wilcoxon_p": wilcox_p,
                "auprc_frac_improved": auprc_improved / n_tfs,
                "auprc_mean_delta": auprc_mean_delta,
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(csvdir / "summary_by_k.csv", index=False)
    print(f"[SAVE] summary_by_k.csv")

    # ------------------------------------------------------------------
    # Report: main table
    # ------------------------------------------------------------------
    for db_name in sorted(databases.keys()):
        db_sum = summary_df[summary_df["database"] == db_name]
        print(f"\n  [{db_name}] (n TFs varies by k overlap)")
        print(f"  {'k':>4s}  {'label':<20s}  {'n':>3s}  "
              f"{'AUROC%+':>7s}  {'AUROC_d':>8s}  {'sign_p':>7s}  "
              f"{'d%+':>5s}  {'d_delta':>8s}  {'wilcox':>7s}  "
              f"{'AUPRC%+':>7s}")
        print("  " + "-" * 95)
        for _, r in db_sum.iterrows():
            k = int(r["k"])
            label = r["k_label"]
            sig_auroc = "*" if r["auroc_sign_p"] < 0.05 else ""
            sig_d = "*" if r["d_wilcoxon_p"] < 0.05 else ""
            print(f"  {k:4d}  {label:<20s}  {r['n_tfs']:3.0f}  "
                  f"{r['auroc_frac_improved']:6.0%}{sig_auroc}  "
                  f"{r['auroc_mean_delta']:+8.4f}  "
                  f"{r['auroc_sign_p']:7.4f}  "
                  f"{r['d_frac_improved']:4.0%}{sig_d}  "
                  f"{r['d_mean_delta']:+8.4f}  "
                  f"{r['d_wilcoxon_p']:7.4f}  "
                  f"{r['auprc_frac_improved']:6.0%}")

    # ------------------------------------------------------------------
    # Report: key TFs detail
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("KEY TFs: METRICS AT k=0, k=tail, k=5")
    print("=" * 90)

    for db_name in sorted(databases.keys()):
        db_data = df[df["database"] == db_name]
        key_data = db_data[db_data["is_key_tf"]]
        if len(key_data) == 0:
            continue

        print(f"\n  [{db_name}]")
        print(f"  {'TF':<8s}  {'n':>3s}  "
              f"{'AUROC@0':>8s}  {'AUROC@t':>8s}  {'AUROC@5':>8s}  "
              f"{'d@0':>7s}  {'d@tail':>7s}  {'d@5':>7s}")
        print("  " + "-" * 70)

        for tf in KEY_TFS:
            tf_data = key_data[key_data["tf"] == tf]
            if len(tf_data) == 0:
                continue
            n = int(tf_data["n_targets"].iloc[0])
            row_str = f"  {tf:<8s}  {n:3d}"

            for metric in ["auroc", "cohens_d"]:
                for k_val in [0, k_tail, 5]:
                    krow = tf_data[tf_data["k"] == k_val]
                    if len(krow) > 0:
                        v = krow[metric].values[0]
                        row_str += f"  {v:8.4f}"
                    else:
                        row_str += f"  {'N/A':>8s}"
            print(row_str)

    # ------------------------------------------------------------------
    # Report: consistency across databases
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("CROSS-DATABASE CONSISTENCY")
    print("=" * 90)

    # For each k, check if the direction of improvement is consistent
    db_names = sorted(databases.keys())
    for k in [k for k in k_candidates if k > 0]:
        label = k_labels[k]
        improvements = {}
        for db_name in db_names:
            row = summary_df[(summary_df["database"] == db_name) &
                             (summary_df["k"] == k)]
            if len(row) > 0:
                improvements[db_name] = {
                    "auroc_pct": row["auroc_frac_improved"].values[0],
                    "d_pct": row["d_frac_improved"].values[0],
                }

        if len(improvements) >= 2:
            auroc_pcts = [v["auroc_pct"] for v in improvements.values()]
            d_pcts = [v["d_pct"] for v in improvements.values()]
            consistent_auroc = all(p > 0.5 for p in auroc_pcts)
            consistent_d = all(p > 0.5 for p in d_pcts)
            print(f"  {label:<25s}  "
                  f"AUROC improved: {'/'.join(f'{p:.0%}' for p in auroc_pcts)} "
                  f"{'CONSISTENT' if consistent_auroc else 'INCONSISTENT'}  "
                  f"d improved: {'/'.join(f'{p:.0%}' for p in d_pcts)} "
                  f"{'CONSISTENT' if consistent_d else 'INCONSISTENT'}")

    # ------------------------------------------------------------------
    # Final verdict
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("FINAL VERDICT")
    print("=" * 90)

    # Check tail-exhaustion k
    tail_rows = summary_df[summary_df["k"] == k_tail]
    if len(tail_rows) > 0:
        all_auroc_majority = all(
            tail_rows[tail_rows["database"] == db]["auroc_frac_improved"].values[0] > 0.5
            for db in tail_rows["database"].unique()
        )
        all_d_majority = all(
            tail_rows[tail_rows["database"] == db]["d_frac_improved"].values[0] > 0.5
            for db in tail_rows["database"].unique()
        )
        mean_auroc_delta = tail_rows["auroc_mean_delta"].mean()
        mean_d_delta = tail_rows["d_mean_delta"].mean()

        print(f"\n  Tail-exhaustion k = {k_tail}:")
        print(f"    AUROC: majority improved in all databases? "
              f"{'YES' if all_auroc_majority else 'NO'}")
        print(f"    Cohen's d: majority improved in all databases? "
              f"{'YES' if all_d_majority else 'NO'}")
        print(f"    Mean AUROC delta (across databases): {mean_auroc_delta:+.4f}")
        print(f"    Mean Cohen's d delta (across databases): {mean_d_delta:+.4f}")

    # Find best consistent k
    print(f"\n  Best consistent k (majority AUROC improvement in all databases):")
    for k in k_candidates:
        if k == 0:
            continue
        k_rows = summary_df[summary_df["k"] == k]
        if len(k_rows) == 0:
            continue
        all_majority = all(
            k_rows[k_rows["database"] == db]["auroc_frac_improved"].values[0] > 0.5
            for db in k_rows["database"].unique()
            if len(k_rows[k_rows["database"] == db]) > 0
        )
        if all_majority:
            mean_delta = k_rows["auroc_mean_delta"].mean()
            print(f"    {k_labels[k]}: mean AUROC delta = {mean_delta:+.4f}")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n[PLOT] Generating figures...")
    try:
        _make_plots(df, summary_df, k_candidates, k_labels, k_tail,
                    databases, figdir)
    except Exception as e:
        print(f"[PLOT ERROR] {e}")
        import traceback
        traceback.print_exc()

    wall = time.perf_counter() - t0
    print(f"\n[DONE] {wall:.1f}s total")


# =====================================================================
# Plots
# =====================================================================

def _make_plots(df, summary_df, k_candidates, k_labels, k_tail,
                databases, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    db_names = sorted(databases.keys())
    colors_db = {"CollecTRI": "C0", "DoRothEA_ABC": "C1", "DoRothEA_AB": "C2"}

    # ----- Figure 1: fraction improved vs k, by database -----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for ax, metric, title in zip(
        axes,
        ["auroc_frac_improved", "d_frac_improved", "auprc_frac_improved"],
        ["AUROC", "Cohen's d", "AUPRC (reference)"],
    ):
        for db in db_names:
            sub = summary_df[summary_df["database"] == db].sort_values("k")
            ax.plot(sub["k"], sub[metric], "o-",
                    color=colors_db.get(db, "C5"), label=db, markersize=5)

        ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="50% line")
        ax.axvline(k_tail, color="red", ls=":", alpha=0.5,
                   label=f"tail-exhaust (k={k_tail})")
        ax.set_xlabel("k")
        ax.set_ylabel(f"fraction of TFs improved over k=0")
        ax.set_title(f"{title}: % TFs improved")
        ax.legend(fontsize=7)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Fraction of TFs with Improved Discrimination vs k=0", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "frac_improved_by_k.png", dpi=150)
    plt.close(fig)
    print(f"[FIG] frac_improved_by_k.png")

    # ----- Figure 2: mean delta vs k, by database -----
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, title in zip(
        axes,
        ["auroc_mean_delta", "d_mean_delta"],
        ["Mean AUROC delta vs k=0", "Mean Cohen's d delta vs k=0"],
    ):
        for db in db_names:
            sub = summary_df[summary_df["database"] == db].sort_values("k")
            ax.plot(sub["k"], sub[metric], "o-",
                    color=colors_db.get(db, "C5"), label=db, markersize=5)

        ax.axhline(0, color="gray", ls="--", alpha=0.5)
        ax.axvline(k_tail, color="red", ls=":", alpha=0.5,
                   label=f"tail-exhaust (k={k_tail})")
        ax.set_xlabel("k")
        ax.set_ylabel("mean delta")
        ax.set_title(title)
        ax.legend(fontsize=7)

    fig.suptitle("Mean Discriminative Improvement vs k=0", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "mean_delta_by_k.png", dpi=150)
    plt.close(fig)
    print(f"[FIG] mean_delta_by_k.png")

    # ----- Figure 3: per-TF AUROC delta at tail-k, by database -----
    fig, axes = plt.subplots(1, len(db_names), figsize=(6 * len(db_names), 5))
    if len(db_names) == 1:
        axes = [axes]

    for ax, db in zip(axes, db_names):
        db_data = df[df["database"] == db]
        base = db_data[db_data["k"] == 0][["tf", "auroc", "is_key_tf"]].rename(
            columns={"auroc": "auroc_0"})
        tail = db_data[db_data["k"] == k_tail][["tf", "auroc"]].rename(
            columns={"auroc": "auroc_t"})
        merged = base.merge(tail, on="tf")
        merged["delta"] = merged["auroc_t"] - merged["auroc_0"]
        merged = merged.sort_values("delta")

        colors = ["C3" if kf else "C0" for kf in merged["is_key_tf"]]
        ax.barh(range(len(merged)), merged["delta"], color=colors, alpha=0.7)
        ax.axvline(0, color="black", lw=0.5)

        # Label key TFs
        for i, (_, r) in enumerate(merged.iterrows()):
            if r["is_key_tf"]:
                ax.text(r["delta"], i, f" {r['tf']}", va="center", fontsize=7)

        ax.set_xlabel(f"AUROC delta (k={k_tail} vs k=0)")
        ax.set_title(f"{db} ({len(merged)} TFs)")
        ax.set_yticks([])

        n_improved = (merged["delta"] > 0).sum()
        ax.text(0.02, 0.98, f"{n_improved}/{len(merged)} improved",
                transform=ax.transAxes, va="top", fontsize=9,
                bbox=dict(boxstyle="round", fc="wheat", alpha=0.8))

    fig.suptitle(f"Per-TF AUROC Change at Tail-Exhaustion k={k_tail}", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "per_tf_auroc_delta.png", dpi=150)
    plt.close(fig)
    print(f"[FIG] per_tf_auroc_delta.png")

    # ----- Figure 4: Key TFs AUROC curves across k, faceted by database -----
    fig, axes = plt.subplots(len(db_names), len(KEY_TFS),
                             figsize=(3 * len(KEY_TFS), 3.5 * len(db_names)))
    if len(db_names) == 1:
        axes = axes[np.newaxis, :]

    for di, db in enumerate(db_names):
        for ti, tf in enumerate(KEY_TFS):
            ax = axes[di, ti]
            tf_data = df[(df["database"] == db) & (df["tf"] == tf)].sort_values("k")
            if len(tf_data) == 0:
                ax.set_visible(False)
                continue

            ax.plot(tf_data["k"], tf_data["auroc"], "o-", color="C0",
                    markersize=3, label="AUROC")
            ax2 = ax.twinx()
            ax2.plot(tf_data["k"], tf_data["cohens_d"], "s--", color="C3",
                     markersize=2, label="d")

            ax.axvline(k_tail, color="red", ls=":", alpha=0.4)
            n = int(tf_data["n_targets"].iloc[0])

            if di == 0:
                ax.set_title(f"{tf} (n={n})", fontsize=9)
            if ti == 0:
                ax.set_ylabel(f"{db}\nAUROC", fontsize=8)
            ax.set_xlabel("k", fontsize=7)
            ax.tick_params(labelsize=6)
            ax2.tick_params(labelsize=6)

    fig.suptitle("Key TFs: AUROC and Cohen's d vs k (by database)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figdir / "key_tfs_metrics_by_db.png", dpi=150)
    plt.close(fig)
    print(f"[FIG] key_tfs_metrics_by_db.png")


if __name__ == "__main__":
    main()
