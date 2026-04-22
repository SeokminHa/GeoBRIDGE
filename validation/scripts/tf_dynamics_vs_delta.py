"""Test whether TF's own velocity dynamics explain Δ (geobridge − correlation).

Per-TF features (from adata.layers):
  - mean_abs_velocity : |velocity|.mean() over cells
  - velocity_var      : variance of velocity across cells
  - mean_expression   : mean .X
  - mean_unspliced    : mean of layers['unspliced'] (if present)
  - unspliced_frac    : unspliced / (spliced + unspliced)
  - velocity_confidence: if adata has scv.tl.velocity_confidence output

Report Spearman(feature, Δ) for each. If mean_abs_velocity ρ > 0 strongly,
supports "geobridge prefers fast-dynamic TFs" mechanistic explanation.
Must be read together with branch-specificity output to avoid confounding
(branch-specific TFs also tend to have localized dynamics).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr


def col_values(adata, layer: str | None, gi: int) -> np.ndarray:
    mat = adata.layers[layer] if layer else adata.X
    col = mat[:, gi]
    return col.toarray().ravel() if hasattr(col, "toarray") else np.asarray(col).ravel()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--delta-csv", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(args.delta_csv)
    adata = sc.read(args.h5ad)
    layers = list(adata.layers.keys())
    print(f"[dyn] adata layers: {layers}")
    has_vel = "velocity" in layers
    has_uns = "unspliced" in layers
    has_spl = "spliced" in layers
    if not has_vel:
        print("[dyn] WARN: no 'velocity' layer — mean_abs_velocity skipped")

    var_idx = {g: i for i, g in enumerate(adata.var_names)}
    rows = []
    missing = 0
    for tf in d["tf"]:
        gi = var_idx.get(tf)
        if gi is None:
            m = [v for v in adata.var_names if v.lower() == tf.lower()]
            gi = var_idx[m[0]] if m else None
        if gi is None:
            rows.append({"tf": tf})
            missing += 1; continue

        feat = {"tf": tf}
        x = col_values(adata, None, gi)
        feat["mean_expression"] = float(np.nanmean(x))

        if has_vel:
            v = col_values(adata, "velocity", gi)
            v_finite = v[np.isfinite(v)]
            if v_finite.size:
                feat["mean_abs_velocity"] = float(np.mean(np.abs(v_finite)))
                feat["velocity_var"] = float(np.var(v_finite))
                # |v| / |expr| — scale-normalized dynamics
                m_expr = float(np.mean(np.abs(x)))
                feat["relative_vel"] = (feat["mean_abs_velocity"] / m_expr
                                        if m_expr > 0 else float("nan"))
        if has_uns and has_spl:
            u = col_values(adata, "unspliced", gi).sum()
            s = col_values(adata, "spliced", gi).sum()
            feat["unspliced_frac"] = float(u / (u + s)) if (u + s) > 0 else float("nan")
            feat["total_counts"] = float(u + s)
        rows.append(feat)
    if missing:
        print(f"[dyn] {missing}/{len(d)} TFs missing from var_names")

    dyn = pd.DataFrame(rows)
    merged = d.merge(dyn, on="tf", how="left")
    merged.to_csv(out / "tf_dynamics_delta.csv", index=False)

    feature_cols = [c for c in dyn.columns if c != "tf"]
    print(f"\n[dyn] Spearman(feature, Δ), n_valid shown, high |ρ| = strong link")
    print("-" * 60)
    rho_rows = []
    for c in feature_cols:
        v = merged.dropna(subset=[c, "delta"])
        if len(v) < 5: continue
        rho, p = spearmanr(v[c], v["delta"])
        rho_rows.append({"feature": c, "n": len(v), "rho": rho, "p": p})
        print(f"  {c:22s}  n={len(v):3d}  ρ={rho:+.3f}  p={p:.3g}")
    pd.DataFrame(rho_rows).to_csv(out / "dynamics_correlation_summary.csv", index=False)

    # Scatter plots for top features
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        top_feats = sorted(rho_rows, key=lambda r: -abs(r["rho"]))[:4]
        if top_feats:
            n = len(top_feats)
            fig, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
            for ax, r in zip(axes[0], top_feats):
                c = r["feature"]
                v = merged.dropna(subset=[c, "delta"])
                ax.scatter(v[c], v["delta"], s=18, alpha=0.7)
                for _, rr in pd.concat([v.nsmallest(5, "delta"),
                                         v.nlargest(5, "delta")]).iterrows():
                    ax.annotate(rr["tf"], (rr[c], rr["delta"]),
                                fontsize=6, alpha=0.8)
                ax.axhline(0, color="gray", lw=0.5)
                ax.set_xlabel(c)
                ax.set_ylabel("Δ NES")
                ax.set_title(f"ρ={r['rho']:+.3f}")
            fig.tight_layout()
            fig.savefig(out / "dynamics_vs_delta.png", dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"[dyn] plot skipped: {e}")

    print(f"\n[dyn] wrote {out}/tf_dynamics_delta.csv + summary + plot")


if __name__ == "__main__":
    main()
