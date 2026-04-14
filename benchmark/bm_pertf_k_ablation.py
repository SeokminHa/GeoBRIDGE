"""
Bone marrow per-TF-per-k analysis for key hematopoietic TFs.

For each k in a grid, compute per-TF AUPRC for the 7 key hematopoietic
TFs individually, plus the full 63-TF mean.

Output:
  results/bm_pertf_k/csv/pertf_k_ablation.csv   — full per-TF × per-k data
  results/bm_pertf_k/csv/key_tf_summary.csv      — key TFs only, wide format
  results/bm_pertf_k/figures/key_tf_curves.png    — per-TF AUPRC vs k
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from geobridge_model import GeoBridgeModel

# ===================================================================

K_GRID = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25]

KEY_TFS = ["ERG", "FLI1", "GATA2", "SPI1", "KLF1", "MYB", "MEIS1"]


def load_and_preprocess(data_dir):
    """Load bone marrow, preprocess, compute velocity, return X, V, pt, genes."""
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

    # Pseudotime
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


def load_ground_truth(genes):
    """Load CollecTRI edges, return tf_targets dict."""
    import decoupler as dc

    if hasattr(dc, "op") and hasattr(dc.op, "collectri"):
        net = dc.op.collectri(organism="human")
    elif hasattr(dc, "get_collectri"):
        net = dc.get_collectri(organism="human", split_complexes=False)
    else:
        raise RuntimeError("Cannot load CollecTRI")

    gene_set = set(genes)
    gene_to_idx = {g: i for i, g in enumerate(genes)}
    tf_targets = {}
    for tf in net["source"].unique():
        if tf not in gene_set:
            continue
        targets = net[net["source"] == tf]["target"].values
        matched = [gene_to_idx[t] for t in targets if t in gene_set and t != tf]
        if len(matched) >= 5:
            tf_targets[tf] = matched

    print(f"[BM] {len(tf_targets)} evaluable TFs, {len(net)} CollecTRI edges")
    return tf_targets, gene_to_idx


def compute_per_tf_auprc(S, tf, tf_idx, target_idxs, G):
    """Compute AUPRC for one TF row."""
    from sklearn.metrics import average_precision_score

    target_set = set(target_idxs)
    scores = []
    labels = []
    for j in range(G):
        if j == tf_idx:
            continue
        scores.append(abs(S[tf_idx, j]))
        labels.append(1 if j in target_set else 0)
    scores = np.array(scores)
    labels = np.array(labels)
    if labels.sum() == 0 or labels.sum() == len(labels):
        return np.nan
    return average_precision_score(labels, scores)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="E:/data_cache")
    ap.add_argument("--outdir", default="results/bm_pertf_k")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    csvdir = outdir / "csv"
    figdir = outdir / "figures"
    csvdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()

    # Load data
    X, V, pt, genes = load_and_preprocess(args.data_dir)
    tf_targets, gene_to_idx = load_ground_truth(genes)
    G = len(genes)

    # Precompute
    print("[BM] Precomputing pairspace stats...")
    pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
        X, V, pt, pair_seed=42,
    )
    print(f"[BM] {pre['n_pairs']} pairs, {len(pre['S'])} singular values")

    # Also compute correlation baseline
    Xn = X - X.mean(axis=0, keepdims=True)
    Xn = Xn / (np.linalg.norm(Xn, axis=0, keepdims=True) + 1e-12)
    Vn = V - V.mean(axis=0, keepdims=True)
    Vn = Vn / (np.linalg.norm(Vn, axis=0, keepdims=True) + 1e-12)
    S_corr = Xn.T @ Vn

    # Per-TF-per-k evaluation
    rows = []

    # Focus TFs: key hematopoietic + all evaluable for mean
    focus_tfs = [tf for tf in KEY_TFS if tf in tf_targets]
    all_tfs = sorted(tf_targets.keys())

    print(f"[BM] Key TFs available: {focus_tfs}")
    print(f"[BM] Evaluating {len(K_GRID)} k values...")

    for ki, k in enumerate(K_GRID):
        if k >= len(pre["S"]):
            continue

        S_k = GeoBridgeModel.scores_from_streaming_stats(
            pre, k=k, project_y=True, normalize_score=True, eps=1e-12,
        )

        # Evaluate all TFs at this k
        for tf in all_tfs:
            tf_idx = gene_to_idx[tf]
            target_idxs = tf_targets[tf]
            auprc = compute_per_tf_auprc(S_k, tf, tf_idx, target_idxs, G)
            random_bl = len(target_idxs) / (G - 1)

            rows.append({
                "tf": tf,
                "k": k,
                "method": "geobridge_dual",
                "auprc": auprc,
                "random_baseline": random_bl,
                "lift": auprc / random_bl if random_bl > 0 else np.nan,
                "n_targets": len(target_idxs),
                "is_key_tf": tf in KEY_TFS,
            })

        # Also evaluate correlation at k=0 only
        if k == 0:
            for tf in all_tfs:
                tf_idx = gene_to_idx[tf]
                target_idxs = tf_targets[tf]
                auprc = compute_per_tf_auprc(S_corr, tf, tf_idx, target_idxs, G)
                random_bl = len(target_idxs) / (G - 1)
                rows.append({
                    "tf": tf,
                    "k": -1,  # sentinel for correlation
                    "method": "correlation",
                    "auprc": auprc,
                    "random_baseline": random_bl,
                    "lift": auprc / random_bl if random_bl > 0 else np.nan,
                    "n_targets": len(target_idxs),
                    "is_key_tf": tf in KEY_TFS,
                })

        if (ki + 1) % 5 == 0 or ki == 0:
            # Quick summary
            key_auprc = [r["auprc"] for r in rows
                         if r["k"] == k and r["tf"] in focus_tfs
                         and r["method"] == "geobridge_dual"]
            all_auprc = [r["auprc"] for r in rows
                         if r["k"] == k and r["method"] == "geobridge_dual"]
            print(f"  k={k:2d}  mean_all={np.nanmean(all_auprc):.6f}  "
                  f"mean_key={np.nanmean(key_auprc):.6f}")

    df = pd.DataFrame(rows)
    df.to_csv(csvdir / "pertf_k_ablation.csv", index=False)
    print(f"\n[SAVE] {csvdir / 'pertf_k_ablation.csv'} ({len(df)} rows)")

    # Key TF summary: wide format (TF x k)
    key_df = df[(df["is_key_tf"]) & (df["method"] == "geobridge_dual")]
    if len(key_df):
        pivot = key_df.pivot_table(
            index="tf", columns="k", values="auprc", aggfunc="first"
        )
        pivot.to_csv(csvdir / "key_tf_summary.csv")
        print(f"[SAVE] {csvdir / 'key_tf_summary.csv'}")

    # Print key TF table
    print("\n" + "=" * 80)
    print("KEY HEMATOPOIETIC TF: AUPRC by k")
    print("=" * 80)

    for tf in focus_tfs:
        sub = df[(df["tf"] == tf) & (df["method"] == "geobridge_dual")].sort_values("k")
        corr_row = df[(df["tf"] == tf) & (df["method"] == "correlation")]
        corr_auprc = corr_row["auprc"].values[0] if len(corr_row) else np.nan
        n_tgt = sub["n_targets"].iloc[0]

        best_k = sub.loc[sub["auprc"].idxmax(), "k"]
        best_auprc = sub["auprc"].max()
        k0_auprc = sub.loc[sub["k"] == 0, "auprc"].values[0] if 0 in sub["k"].values else np.nan

        print(f"\n  {tf} (n_targets={n_tgt}, corr={corr_auprc:.4f})")
        print(f"  {'k':>3s}  {'AUPRC':>8s}  {'lift':>6s}  {'vs_k0':>8s}")
        for _, r in sub.iterrows():
            vs_k0 = r["auprc"] - k0_auprc if not np.isnan(k0_auprc) else np.nan
            marker = " <-- best" if r["k"] == best_k else ""
            print(f"  {int(r['k']):3d}  {r['auprc']:8.4f}  {r['lift']:6.2f}  "
                  f"{vs_k0:+8.4f}{marker}")

    # Aggregate: mean over key TFs at each k
    print("\n" + "=" * 80)
    print("MEAN OVER KEY TFs")
    print("=" * 80)
    key_means = key_df.groupby("k")["auprc"].mean().sort_index()
    all_means = df[(df["method"] == "geobridge_dual")].groupby("k")["auprc"].mean().sort_index()
    best_k_key = key_means.idxmax()
    best_k_all = all_means.idxmax()

    print(f"\n  {'k':>3s}  {'key_TF_mean':>12s}  {'all_63_mean':>12s}")
    for k in sorted(set(key_means.index) | set(all_means.index)):
        km = key_means.get(k, np.nan)
        am = all_means.get(k, np.nan)
        markers = []
        if k == best_k_key:
            markers.append("best_key")
        if k == best_k_all:
            markers.append("best_all")
        m_str = f"  <-- {', '.join(markers)}" if markers else ""
        print(f"  {int(k):3d}  {km:12.6f}  {am:12.6f}{m_str}")

    # Rule simulation
    print("\n" + "=" * 80)
    print("RULE SIMULATION ON BONE MARROW")
    print("=" * 80)

    from geobridge_model.spectral_k import select_k_by_tail_exhaustion

    S_sv = pre["S"]
    te_k = int(select_k_by_tail_exhaustion(S_sv, max_k=40, tau=2.0, stable_m=3).k_selected)
    ct_k = int(select_k_by_tail_exhaustion(S_sv, max_k=40, tau=3.0, stable_m=5).k_selected)

    rules = {
        "tail_exhaustion": te_k,
        "conservative_tail": ct_k,
        "hybrid_cap_10": min(ct_k, 10),
        "hybrid_cap_8": min(ct_k, 8),
        "hybrid_cap_5": min(ct_k, 5),
    }

    print(f"\n  {'Rule':<22s}  {'k':>3s}  {'key_TF_mean':>12s}  {'all_63_mean':>12s}  {'%_oracle_key':>12s}")
    oracle_key = float(key_means.max())
    oracle_all = float(all_means.max())

    for rule_name, rule_k in rules.items():
        # Find closest k in grid
        closest_k = min(K_GRID, key=lambda x: abs(x - rule_k))
        km = key_means.get(closest_k, np.nan)
        am = all_means.get(closest_k, np.nan)
        pct = km / oracle_key * 100 if oracle_key > 0 else np.nan
        print(f"  {rule_name:<22s}  {rule_k:3d}  {km:12.6f}  {am:12.6f}  {pct:11.1f}%")

    # Per key TF at each rule's k
    print(f"\n  Per key TF AUPRC at each rule's k:")
    print(f"  {'Rule':<22s}", end="")
    for tf in focus_tfs:
        print(f"  {tf:>8s}", end="")
    print()

    for rule_name, rule_k in rules.items():
        closest_k = min(K_GRID, key=lambda x: abs(x - rule_k))
        print(f"  {rule_name:<22s}", end="")
        for tf in focus_tfs:
            sub = df[(df["tf"] == tf) & (df["k"] == closest_k) & (df["method"] == "geobridge_dual")]
            val = sub["auprc"].values[0] if len(sub) else np.nan
            print(f"  {val:8.4f}", end="")
        print()

    # k=0 baseline for comparison
    print(f"  {'k=0 (baseline)':<22s}", end="")
    for tf in focus_tfs:
        sub = df[(df["tf"] == tf) & (df["k"] == 0) & (df["method"] == "geobridge_dual")]
        val = sub["auprc"].values[0] if len(sub) else np.nan
        print(f"  {val:8.4f}", end="")
    print()

    print(f"  {'correlation':<22s}", end="")
    for tf in focus_tfs:
        sub = df[(df["tf"] == tf) & (df["method"] == "correlation")]
        val = sub["auprc"].values[0] if len(sub) else np.nan
        print(f"  {val:8.4f}", end="")
    print()

    # Plots
    print("\n[PLOT] Generating figures...")
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 4, figsize=(16, 8))
        axes = axes.flatten()

        for ti, tf in enumerate(focus_tfs):
            ax = axes[ti]
            sub = df[(df["tf"] == tf) & (df["method"] == "geobridge_dual")].sort_values("k")
            corr_row = df[(df["tf"] == tf) & (df["method"] == "correlation")]

            ax.plot(sub["k"], sub["auprc"], "o-", color="C0", markersize=4, label="GeoBRIDGE")
            ax.axhline(sub.loc[sub["k"] == 0, "auprc"].values[0],
                       color="gray", ls="--", alpha=0.5, label="k=0")
            if len(corr_row):
                ax.axhline(corr_row["auprc"].values[0],
                           color="C1", ls=":", alpha=0.7, label="correlation")
            ax.axhline(sub["random_baseline"].values[0],
                       color="red", ls=":", alpha=0.3, label="random")

            # Mark rule k values
            for rule_name, rule_k in rules.items():
                if "hybrid" in rule_name or rule_name == "conservative_tail":
                    closest_k = min(K_GRID, key=lambda x: abs(x - rule_k))
                    r_sub = sub[sub["k"] == closest_k]
                    if len(r_sub):
                        marker = "v" if "hybrid" in rule_name else "s"
                        ax.plot(closest_k, r_sub["auprc"].values[0], marker,
                                color="C2" if "5" in rule_name else "C3" if "8" in rule_name else "C4",
                                markersize=8, zorder=5)

            ax.set_title(f"{tf} (n={sub['n_targets'].iloc[0]})", fontsize=10)
            ax.set_xlabel("k")
            ax.set_ylabel("AUPRC")
            if ti == 0:
                ax.legend(fontsize=7, loc="best")

        # Last panel: mean over key TFs
        ax = axes[7]
        ax.plot(key_means.index, key_means.values, "o-", color="C0", markersize=4, label="key TF mean")
        ax.plot(all_means.index, all_means.values, "s--", color="C5", markersize=3, label="all 63 TF mean")
        ax.axhline(key_means.iloc[0], color="gray", ls="--", alpha=0.5)
        ax.set_title("Mean AUPRC", fontsize=10)
        ax.set_xlabel("k")
        ax.set_ylabel("AUPRC")
        ax.legend(fontsize=7)

        fig.suptitle("Bone Marrow: Per-TF AUPRC vs k (Key Hematopoietic TFs)", fontsize=12)
        fig.tight_layout()
        fig.savefig(figdir / "key_tf_curves.png", dpi=150)
        plt.close(fig)
        print(f"[FIG] key_tf_curves.png")

    except Exception as e:
        print(f"[PLOT ERROR] {e}")
        import traceback
        traceback.print_exc()

    wall = time.perf_counter() - t0
    print(f"\n[DONE] {wall:.1f}s total")


if __name__ == "__main__":
    main()
