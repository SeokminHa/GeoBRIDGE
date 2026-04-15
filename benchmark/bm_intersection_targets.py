"""
Intersection-target control: isolate projection effect from label-set
differences.

For each TF shared across CollecTRI and DoRothEA_ABC:
  - Identify intersection targets (in both databases)
  - Identify exclusive targets (in one database only)
  - Evaluate AUROC / Cohen's d at each k using:
    (a) intersection targets only
    (b) CollecTRI-exclusive targets
    (c) DoRothEA-exclusive targets

This isolates whether direction reversal (projection helps under one
database but hurts under another) is driven by:
  1. Non-overlapping targets (different classification tasks)
  2. Something about projection itself (label-independent effect)

If projection effect is consistent on intersection targets but reverses
on exclusive targets -> inconsistency is a target-definition problem
If it reverses even on intersection targets -> projection is
fundamentally label-dependent (very unlikely but worth checking)

Output:
  results/bm_intersection_targets/csv/intersection_metrics.csv
  results/bm_intersection_targets/csv/target_overlap.csv
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

KEY_TFS = ["ERG", "FLI1", "GATA2", "SPI1", "KLF1", "MYB", "MEIS1"]
K_GRID = [0, 1, 2, 3, 5, 7, 10, 15, 19, 25]


# =====================================================================
# Data loading (identical to other scripts)
# =====================================================================

def load_and_preprocess(data_dir):
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


def load_both_databases(genes):
    """Load CollecTRI and DoRothEA_ABC, return raw target index sets."""
    import decoupler as dc

    gene_set = set(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}

    def _get_targets(net):
        """Return {tf: set of target indices}."""
        tf_targets = {}
        for tf in net["source"].unique():
            if tf not in gene_set:
                continue
            targets = net[net["source"] == tf]["target"].values
            matched = set(gene_to_idx[t] for t in targets
                          if t in gene_set and t != tf)
            if matched:
                tf_targets[tf] = matched
        return tf_targets

    # CollecTRI
    if hasattr(dc, "op") and hasattr(dc.op, "collectri"):
        ct_net = dc.op.collectri(organism="human")
    else:
        ct_net = dc.get_collectri(organism="human", split_complexes=False)
    ct_targets = _get_targets(ct_net)
    print(f"[CollecTRI] {len(ct_targets)} TFs with any targets in HVG")

    # DoRothEA ABC
    if hasattr(dc, "op") and hasattr(dc.op, "dorothea"):
        do_net = dc.op.dorothea(organism="human")
    else:
        do_net = dc.get_dorothea(organism="human")
    do_targets = _get_targets(do_net)
    print(f"[DoRothEA_ABC] {len(do_targets)} TFs with any targets in HVG")

    return ct_targets, do_targets, gene_to_idx


# =====================================================================
# Metrics
# =====================================================================

def compute_metrics(S_score, tf_idx, target_idxs, G):
    """AUROC and Cohen's d for a given target set."""
    from sklearn.metrics import roc_auc_score

    if len(target_idxs) == 0:
        return None

    target_set = set(target_idxs)
    target_scores = []
    nontarget_scores = []

    for j in range(G):
        if j == tf_idx:
            continue
        s = abs(S_score[tf_idx, j])
        if j in target_set:
            target_scores.append(s)
        else:
            nontarget_scores.append(s)

    target_scores = np.array(target_scores)
    nontarget_scores = np.array(nontarget_scores)

    if len(target_scores) < 2 or len(nontarget_scores) < 2:
        return None

    pooled_std = np.sqrt(
        (target_scores.var() * len(target_scores)
         + nontarget_scores.var() * len(nontarget_scores))
        / (len(target_scores) + len(nontarget_scores))
    )
    cohens_d = ((target_scores.mean() - nontarget_scores.mean()) / pooled_std
                if pooled_std > 1e-15 else 0.0)

    labels = np.concatenate([np.ones(len(target_scores)),
                              np.zeros(len(nontarget_scores))])
    scores = np.concatenate([target_scores, nontarget_scores])
    try:
        auroc = roc_auc_score(labels, scores)
    except ValueError:
        auroc = np.nan

    return {"auroc": float(auroc), "cohens_d": float(cohens_d)}


# =====================================================================
# Main
# =====================================================================

def main():
    import argparse
    ap = argparse.ArgumentParser(
        description="Intersection-target control for projection effect")
    ap.add_argument("--data-dir", default="E:/data_cache")
    ap.add_argument("--outdir", default="results/bm_intersection_targets")
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
    idx_to_gene = {i: g for i, g in enumerate(genes)}
    gene_to_idx_map = {g: i for i, g in enumerate(genes)}

    ct_targets, do_targets, gene_to_idx = load_both_databases(genes)

    # ------------------------------------------------------------------
    # Find shared TFs and compute target overlaps
    # ------------------------------------------------------------------
    shared_tfs = sorted(set(ct_targets.keys()) & set(do_targets.keys()))
    print(f"\n[SHARED] {len(shared_tfs)} TFs in both databases")

    overlap_rows = []
    analysis_tfs = []  # TFs with enough intersection targets
    MIN_INTERSECTION = 3

    for tf in shared_tfs:
        ct_set = ct_targets[tf]
        do_set = do_targets[tf]
        inter = ct_set & do_set
        ct_only = ct_set - do_set
        do_only = do_set - ct_set
        union = ct_set | do_set
        jaccard = len(inter) / len(union) if len(union) > 0 else 0

        overlap_rows.append({
            "tf": tf,
            "is_key_tf": tf in KEY_TFS,
            "n_collectri": len(ct_set),
            "n_dorothea": len(do_set),
            "n_intersection": len(inter),
            "n_ct_only": len(ct_only),
            "n_do_only": len(do_only),
            "n_union": len(union),
            "jaccard": jaccard,
        })

        if len(inter) >= MIN_INTERSECTION:
            analysis_tfs.append(tf)

    overlap_df = pd.DataFrame(overlap_rows).sort_values("jaccard", ascending=False)
    overlap_df.to_csv(csvdir / "target_overlap.csv", index=False)
    print(f"[SAVE] target_overlap.csv")

    print(f"\n  Target overlap summary:")
    print(f"  {'TF':<10s}  {'CT':>4s}  {'DO':>4s}  {'inter':>5s}  "
          f"{'CT_only':>7s}  {'DO_only':>7s}  {'Jaccard':>7s}  {'key':>3s}")
    print("  " + "-" * 60)
    for _, r in overlap_df.iterrows():
        key = "*" if r["is_key_tf"] else ""
        print(f"  {r['tf']:<10s}  {r['n_collectri']:4.0f}  {r['n_dorothea']:4.0f}  "
              f"{r['n_intersection']:5.0f}  {r['n_ct_only']:7.0f}  "
              f"{r['n_do_only']:7.0f}  {r['jaccard']:7.3f}  {key:>3s}")

    print(f"\n  TFs with >= {MIN_INTERSECTION} intersection targets: "
          f"{len(analysis_tfs)}")
    print(f"  Mean Jaccard: {overlap_df['jaccard'].mean():.3f}")
    print(f"  Median Jaccard: {overlap_df['jaccard'].median():.3f}")

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
    # Evaluate at each k with different target subsets
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("EVALUATING INTERSECTION / EXCLUSIVE TARGET SETS")
    print("=" * 90)

    metric_rows = []

    for k in K_GRID:
        if k >= len(pre["S"]):
            continue

        S_k = GeoBridgeModel.scores_from_streaming_stats(
            pre, k=k, project_y=True, normalize_score=True, eps=1e-12,
        )

        for tf in analysis_tfs:
            tf_idx = gene_to_idx[tf]
            ct_set = ct_targets[tf]
            do_set = do_targets[tf]
            inter = ct_set & do_set
            ct_only = ct_set - do_set
            do_only = do_set - ct_set

            # Evaluate each target subset
            for subset_name, subset in [
                ("intersection", inter),
                ("ct_only", ct_only),
                ("do_only", do_only),
                ("collectri_full", ct_set),
                ("dorothea_full", do_set),
            ]:
                if len(subset) < 2:
                    continue
                m = compute_metrics(S_k, tf_idx, subset, G)
                if m is None:
                    continue
                m["k"] = k
                m["tf"] = tf
                m["is_key_tf"] = tf in KEY_TFS
                m["target_subset"] = subset_name
                m["n_targets"] = len(subset)
                metric_rows.append(m)

        print(f"  k={k:2d} done")

    mdf = pd.DataFrame(metric_rows)
    mdf.to_csv(csvdir / "intersection_metrics.csv", index=False)
    print(f"\n[SAVE] intersection_metrics.csv ({len(mdf)} rows)")

    # ------------------------------------------------------------------
    # Report: projection effect by target subset
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("PROJECTION EFFECT BY TARGET SUBSET")
    print("=" * 90)

    for subset_name in ["intersection", "ct_only", "do_only",
                        "collectri_full", "dorothea_full"]:
        sub = mdf[mdf["target_subset"] == subset_name]
        if len(sub) == 0:
            continue

        baseline = sub[sub["k"] == 0][["tf", "auroc", "cohens_d"]].rename(
            columns={"auroc": "a0", "cohens_d": "d0"})

        print(f"\n  --- {subset_name} ---")
        print(f"  {'k':>4s}  {'n_tf':>4s}  {'AUROC%+':>7s}  "
              f"{'AUROC_d':>9s}  {'d%+':>5s}  {'d_delta':>9s}")
        print("  " + "-" * 50)

        for k in K_GRID:
            if k == 0:
                continue
            kdata = sub[sub["k"] == k][["tf", "auroc", "cohens_d"]].rename(
                columns={"auroc": "ak", "cohens_d": "dk"})
            merged = baseline.merge(kdata, on="tf")
            if len(merged) == 0:
                continue

            n = len(merged)
            auroc_imp = (merged["ak"] > merged["a0"]).sum()
            auroc_delta = (merged["ak"] - merged["a0"]).mean()
            d_imp = (merged["dk"] > merged["d0"]).sum()
            d_delta = (merged["dk"] - merged["d0"]).mean()

            print(f"  {k:4d}  {n:4d}  {auroc_imp/n:6.0%}   "
                  f"{auroc_delta:+9.4f}  {d_imp/n:4.0%}   {d_delta:+9.4f}")

    # ------------------------------------------------------------------
    # Report: per-TF direction comparison (intersection targets)
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("PER-TF DETAIL: INTERSECTION vs EXCLUSIVE TARGETS")
    print("=" * 90)

    for tf in [t for t in KEY_TFS if t in analysis_tfs] + \
              [t for t in analysis_tfs if t not in KEY_TFS]:
        ct_n = len(ct_targets.get(tf, set()))
        do_n = len(do_targets.get(tf, set()))
        inter_n = len(ct_targets.get(tf, set()) & do_targets.get(tf, set()))

        tf_data = mdf[mdf["tf"] == tf]
        if len(tf_data) == 0:
            continue

        print(f"\n  {tf} (CT={ct_n}, DO={do_n}, intersection={inter_n}):")
        print(f"    {'k':>3s}  {'inter_AUROC':>11s}  {'ct_full':>9s}  "
              f"{'do_full':>9s}  {'ct_only':>9s}  {'do_only':>9s}")
        print(f"    " + "-" * 55)

        for k in K_GRID:
            row_str = f"    {k:3d}"
            for subset in ["intersection", "collectri_full", "dorothea_full",
                           "ct_only", "do_only"]:
                sub_row = tf_data[(tf_data["k"] == k) &
                                  (tf_data["target_subset"] == subset)]
                if len(sub_row) > 0:
                    row_str += f"  {sub_row['auroc'].values[0]:9.4f}"
                else:
                    row_str += f"  {'N/A':>9s}"
            print(row_str)

    # ------------------------------------------------------------------
    # Key diagnostic: does direction reverse on intersection targets?
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("CRITICAL TEST: DIRECTION REVERSAL ON INTERSECTION TARGETS?")
    print("=" * 90)

    # Compare AUROC trajectory on intersection vs full databases
    print(f"\n  For each TF, does projection direction (help/hurt) match")
    print(f"  when evaluated on intersection vs database-specific targets?")

    for k in [3, 5, 10, 19]:
        print(f"\n  k={k}:")
        agree_count = 0
        disagree_count = 0
        total = 0

        for tf in analysis_tfs:
            inter_data = mdf[(mdf["tf"] == tf) & (mdf["target_subset"] == "intersection")]
            ct_data = mdf[(mdf["tf"] == tf) & (mdf["target_subset"] == "collectri_full")]
            do_data = mdf[(mdf["tf"] == tf) & (mdf["target_subset"] == "dorothea_full")]

            inter_0 = inter_data[inter_data["k"] == 0]["auroc"].values
            inter_k = inter_data[inter_data["k"] == k]["auroc"].values
            ct_0 = ct_data[ct_data["k"] == 0]["auroc"].values
            ct_k = ct_data[ct_data["k"] == k]["auroc"].values
            do_0 = do_data[do_data["k"] == 0]["auroc"].values
            do_k = do_data[do_data["k"] == k]["auroc"].values

            if len(inter_0) == 0 or len(inter_k) == 0:
                continue
            if len(ct_0) == 0 or len(ct_k) == 0:
                continue
            if len(do_0) == 0 or len(do_k) == 0:
                continue

            dir_inter = np.sign(inter_k[0] - inter_0[0])
            dir_ct = np.sign(ct_k[0] - ct_0[0])
            dir_do = np.sign(do_k[0] - do_0[0])

            # Does the direction on full databases match?
            ct_do_agree = dir_ct == dir_do
            # Does intersection agree with both?
            inter_ct_agree = dir_inter == dir_ct
            inter_do_agree = dir_inter == dir_do

            total += 1
            if ct_do_agree:
                agree_count += 1
            else:
                disagree_count += 1

            is_key = "*" if tf in KEY_TFS else " "
            print(f"    {is_key}{tf:<10s}  inter={dir_inter:+.0f}  "
                  f"CT={dir_ct:+.0f}  DO={dir_do:+.0f}  "
                  f"CT-DO {'AGREE' if ct_do_agree else 'REVERSE'}")

        if total > 0:
            print(f"    --> CT vs DO agreement: {agree_count}/{total} "
                  f"({agree_count/total:.0%})")

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    print("\n[PLOT] Generating figures...")
    try:
        _make_plots(mdf, overlap_df, analysis_tfs, figdir)
    except Exception as e:
        print(f"[PLOT ERROR] {e}")
        import traceback
        traceback.print_exc()

    wall = time.perf_counter() - t0
    print(f"\n[DONE] {wall:.1f}s total")


def _make_plots(mdf, overlap_df, analysis_tfs, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ----- Figure 1: AUROC vs k by target subset (aggregated) -----
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    subsets = ["intersection", "collectri_full", "dorothea_full"]
    titles = ["Intersection targets", "CollecTRI full", "DoRothEA full"]
    colors = ["C2", "C0", "C1"]

    for ax, subset, title, color in zip(axes, subsets, titles, colors):
        sub = mdf[mdf["target_subset"] == subset]
        if len(sub) == 0:
            continue
        mean_auroc = sub.groupby("k")["auroc"].mean()
        std_auroc = sub.groupby("k")["auroc"].std()
        ax.plot(mean_auroc.index, mean_auroc.values, "o-",
                color=color, markersize=5)
        ax.fill_between(mean_auroc.index,
                        mean_auroc.values - std_auroc.values,
                        mean_auroc.values + std_auroc.values,
                        alpha=0.2, color=color)
        ax.axhline(0.5, color="gray", ls="--", alpha=0.3)
        ax.set_xlabel("k")
        ax.set_ylabel("Mean AUROC")
        ax.set_title(f"{title}\n(n={sub[sub['k']==0]['tf'].nunique()} TFs)")

    fig.suptitle("Mean AUROC vs k by Target Subset", fontsize=12)
    fig.tight_layout()
    fig.savefig(figdir / "auroc_by_target_subset.png", dpi=150)
    plt.close(fig)
    print(f"[FIG] auroc_by_target_subset.png")

    # ----- Figure 2: Key TFs AUROC comparison across subsets -----
    key_present = [tf for tf in KEY_TFS if tf in analysis_tfs]
    if key_present:
        n_key = len(key_present)
        fig, axes = plt.subplots(1, n_key, figsize=(4 * n_key, 4))
        if n_key == 1:
            axes = [axes]

        for ax, tf in zip(axes, key_present):
            for subset, color, label in [
                ("intersection", "C2", "intersection"),
                ("collectri_full", "C0", "CollecTRI"),
                ("dorothea_full", "C1", "DoRothEA"),
            ]:
                tf_sub = mdf[(mdf["tf"] == tf) &
                             (mdf["target_subset"] == subset)].sort_values("k")
                if len(tf_sub) > 0:
                    n = int(tf_sub["n_targets"].iloc[0])
                    ax.plot(tf_sub["k"], tf_sub["auroc"], "o-",
                            color=color, markersize=4,
                            label=f"{label} (n={n})")

            ax.axhline(0.5, color="gray", ls="--", alpha=0.3)
            ax.set_xlabel("k")
            ax.set_ylabel("AUROC")
            ax.set_title(tf, fontsize=10)
            ax.legend(fontsize=7)

        fig.suptitle("Key TFs: AUROC by Target Subset", fontsize=11)
        fig.tight_layout()
        fig.savefig(figdir / "key_tfs_auroc_by_subset.png", dpi=150)
        plt.close(fig)
        print(f"[FIG] key_tfs_auroc_by_subset.png")

    # ----- Figure 3: Jaccard overlap distribution -----
    fig, ax = plt.subplots(figsize=(8, 4))
    jaccards = overlap_df["jaccard"].values
    ax.hist(jaccards, bins=20, alpha=0.7, edgecolor="black")
    ax.axvline(np.median(jaccards), color="red", ls="--",
               label=f"median={np.median(jaccards):.3f}")
    ax.set_xlabel("Jaccard index (target overlap)")
    ax.set_ylabel("count (TFs)")
    ax.set_title("Target Set Overlap: CollecTRI vs DoRothEA_ABC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(figdir / "target_overlap_distribution.png", dpi=150)
    plt.close(fig)
    print(f"[FIG] target_overlap_distribution.png")


if __name__ == "__main__":
    main()
