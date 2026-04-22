"""Test whether GeoBRIDGE's strong-negative Δ TFs are branch/lineage-restricted.

Hypothesis: global TF-target enrichment penalizes method that captures
branch-conditioned dynamics, because cells outside the branch dilute the signal.
If strong-loss TFs (correlation wins, geobridge loses) are enriched for
branch-specific regulators, a positive reframe (within-branch eval) is viable.

Specificity metric: tau (Yanai 2005):
  tau = sum(1 - x_i / x_max) / (n - 1)   where x_i = mean expr in cluster i
  tau = 0: uniform across clusters; tau = 1: expressed in one cluster only.

Reports Spearman(tau, Δ): expect ρ < 0 (higher specificity → more negative Δ)
if branch-specificity hypothesis holds.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr, mannwhitneyu


def tissue_specificity_tau(expr_by_cluster: np.ndarray) -> float:
    """Yanai tau from 1D array of per-cluster mean expression."""
    x = np.asarray(expr_by_cluster, dtype=float)
    x = np.clip(x, 0, None)
    xmax = x.max()
    if xmax <= 0 or len(x) < 2:
        return float("nan")
    return float(np.sum(1 - x / xmax) / (len(x) - 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta-csv", required=True,
                    help="e.g. results/per_tf_delta/bm/delta_*_vs_correlation.csv")
    ap.add_argument("--h5ad", required=True,
                    help="ready h5ad with clusters/celltype in .obs")
    ap.add_argument("--cluster-key", default=None,
                    help="obs column for cluster/celltype; auto-detects common keys")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.delta_csv)
    if "delta" not in d.columns or "tf" not in d.columns:
        raise SystemExit(f"delta csv missing tf/delta; cols={d.columns.tolist()}")

    adata = sc.read(args.h5ad)
    key = args.cluster_key
    if key is None:
        for cand in ["clusters", "celltype", "cell_type", "leiden", "louvain"]:
            if cand in adata.obs.columns:
                key = cand; break
    if key is None:
        raise SystemExit(f"no cluster key in obs; cols={adata.obs.columns.tolist()}")
    print(f"[spec] using cluster key '{key}' with "
          f"{adata.obs[key].nunique()} groups, {adata.n_obs} cells")

    # per-cluster mean expression (log1p preferred; use .X as is)
    X = adata.X
    var_idx = {g: i for i, g in enumerate(adata.var_names)}
    clusters = adata.obs[key].astype("category")
    cluster_codes = clusters.cat.codes.to_numpy()
    n_clusters = len(clusters.cat.categories)

    rows = []
    missing = 0
    for tf in d["tf"]:
        # gene symbol case handling: try exact, then case-insensitive
        gi = var_idx.get(tf)
        if gi is None:
            # fallback: case-insensitive match
            m = [v for v in adata.var_names if v.lower() == tf.lower()]
            gi = var_idx[m[0]] if m else None
        if gi is None:
            rows.append({"tf": tf, "tau": float("nan"),
                         "max_cluster": None, "max_frac_of_mean": float("nan")})
            missing += 1
            continue
        col = X[:, gi]
        col = col.toarray().ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()

        per_c = np.zeros(n_clusters)
        for c in range(n_clusters):
            mask = cluster_codes == c
            if mask.any():
                per_c[c] = col[mask].mean()
        tau = tissue_specificity_tau(per_c)
        cmax = int(np.argmax(per_c))
        max_ratio = (per_c.max() / per_c.mean()) if per_c.mean() > 0 else float("nan")
        rows.append({"tf": tf, "tau": tau,
                     "max_cluster": str(clusters.cat.categories[cmax]),
                     "max_frac_of_mean": max_ratio})
    if missing:
        print(f"[spec] {missing}/{len(d)} TFs not in adata var_names")

    spec = pd.DataFrame(rows)
    merged = d.merge(spec, on="tf", how="left")
    merged.to_csv(out / "tf_specificity_delta.csv", index=False)

    # Spearman(tau, delta) — primary test
    valid = merged.dropna(subset=["tau", "delta"])
    rho, pval = spearmanr(valid["tau"], valid["delta"])
    print(f"\n[spec] n={len(valid)}  Spearman(tau, delta) = {rho:+.3f}  p={pval:.3g}")
    print(f"      (negative ρ supports branch-specificity hypothesis)")

    # Group comparison: top 10 losses vs top 10 wins
    losses = merged.nsmallest(10, "delta")
    wins = merged.nlargest(10, "delta")
    tau_losses = losses["tau"].dropna().values
    tau_wins = wins["tau"].dropna().values
    if len(tau_losses) >= 3 and len(tau_wins) >= 3:
        U, p = mannwhitneyu(tau_losses, tau_wins, alternative="greater")
        print(f"\n[spec] top-10 losses vs wins, Mann-Whitney (losses tau > wins tau):"
              f"  U={U:.0f}  p={p:.3g}")
        print(f"       mean tau: losses={tau_losses.mean():.3f}  wins={tau_wins.mean():.3f}")
        print(f"       median tau: losses={np.median(tau_losses):.3f}  "
              f"wins={np.median(tau_wins):.3f}")

    # Optional plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(valid["tau"], valid["delta"], s=18, alpha=0.7)
        for _, r in pd.concat([valid.nsmallest(8, "delta"),
                                valid.nlargest(8, "delta")]).iterrows():
            ax.annotate(r["tf"], (r["tau"], r["delta"]),
                        fontsize=6, alpha=0.8)
        ax.axhline(0, color="gray", lw=0.5)
        ax.set_xlabel(f"tissue-specificity τ ({key})")
        ax.set_ylabel("Δ NES (geobridge − correlation)")
        ax.set_title(f"Branch-specificity vs Δ  |  ρ={rho:+.3f}  p={pval:.2g}")
        fig.tight_layout()
        fig.savefig(out / "tau_vs_delta.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        print(f"[spec] plot skipped: {e}")

    print(f"\n[spec] wrote {out}/tf_specificity_delta.csv + plot")


if __name__ == "__main__":
    main()
