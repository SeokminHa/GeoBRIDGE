"""Per-TF delta-vs-correlation analysis — primary decision signal.

Reads per_tf.csv produced by run_v1_enrichment.py (columns: tf, method, nes,
optionally pval/fdr). For each non-baseline method, computes
  Δ_nes = method_nes - correlation_nes  per TF

Reports:
  - per-method summary (mean/median Δ, counts of strong wins/losses, robust
    wins with FDR support)
  - per-method ranked CSV (tf, nes_baseline, nes_method, Δ, [fdr], lineage tag)
  - ranked bar plot

Annotation: optional lineage-TF highlight via --lineage-set {bm|pancreas}.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np


LINEAGE = {
    "bm": {  # human hematopoiesis master regulators
        "GATA1", "GATA2", "SPI1", "CEBPA", "CEBPB", "KLF1", "GFI1", "GFI1B",
        "RUNX1", "TAL1", "LMO2", "MEF2C", "IKZF1", "IRF8", "MYB", "FLI1",
        "NFE2", "LDB1", "HHEX", "ERG", "MAFB", "EBF1", "TCF3", "PAX5",
    },
    "pancreas": {  # mouse pancreas endocrine
        "Pdx1", "Neurog3", "Neurod1", "Nkx2-2", "Nkx6-1", "Pax4", "Pax6",
        "Mafa", "Mafb", "Isl1", "Arx", "Foxa2", "Hnf1a", "Hnf1b", "Hnf4a",
        "Rfx6", "Sox9", "Ptf1a", "Mnx1", "Insm1",
    },
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tf", required=True,
                    help="per_tf.csv from run_v1_enrichment.py")
    ap.add_argument("--baseline", default="correlation")
    ap.add_argument("--methods", default=None,
                    help="comma-separated methods; default = all non-baseline")
    ap.add_argument("--lineage-set", default=None, choices=list(LINEAGE.keys()))
    ap.add_argument("--delta-threshold", type=float, default=0.3)
    ap.add_argument("--fdr-cutoff", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.per_tf)
    if not {"tf", "method", "nes"}.issubset(df.columns):
        raise SystemExit(f"missing required cols; got {df.columns.tolist()}")

    pv_nes = df.pivot_table(index="tf", columns="method", values="nes", aggfunc="first")
    if args.baseline not in pv_nes.columns:
        raise SystemExit(f"baseline '{args.baseline}' not found; "
                         f"available: {pv_nes.columns.tolist()}")

    has_fdr = "fdr" in df.columns
    pv_fdr = (df.pivot_table(index="tf", columns="method", values="fdr", aggfunc="first")
              if has_fdr else None)

    lineage = LINEAGE.get(args.lineage_set, set())
    methods = (args.methods.split(",") if args.methods
               else [m for m in pv_nes.columns if m != args.baseline])

    summary_rows = []
    for mt in methods:
        if mt not in pv_nes.columns:
            print(f"[warn] {mt} not in data; skipping")
            continue
        both = pv_nes[[args.baseline, mt]].dropna()
        if len(both) == 0:
            print(f"[warn] no TFs evaluated by both {args.baseline} and {mt}")
            continue
        d = both[mt] - both[args.baseline]

        s = {
            "method": mt,
            "n_tf_compared": int(d.size),
            "mean_delta": float(d.mean()),
            "median_delta": float(d.median()),
            "frac_positive_delta": float((d > 0).mean()),
            "n_strong_wins": int((d > args.delta_threshold).sum()),
            "n_strong_losses": int((d < -args.delta_threshold).sum()),
        }
        if has_fdr and mt in pv_fdr.columns:
            robust = ((d > args.delta_threshold) &
                      (pv_fdr.loc[d.index, mt] < args.fdr_cutoff))
            s["n_robust_wins"] = int(robust.sum())
            s["robust_lineage_wins"] = int(
                robust[robust].index.intersection(lineage).size
            ) if lineage else None
        summary_rows.append(s)

        per = pd.DataFrame({
            "tf": d.index,
            f"nes_{args.baseline}": both[args.baseline].values,
            f"nes_{mt}": both[mt].values,
            "delta": d.values,
        })
        if has_fdr and mt in pv_fdr.columns:
            per[f"fdr_{mt}"] = pv_fdr.loc[d.index, mt].values
        if lineage:
            per["is_lineage_tf"] = per["tf"].isin(lineage)
        per = per.sort_values("delta", ascending=False)
        per.to_csv(out / f"delta_{mt}_vs_{args.baseline}.csv", index=False)

        print(f"\n[delta] {mt}  n={s['n_tf_compared']}  "
              f"mean_Δ={s['mean_delta']:+.3f}  "
              f"wins={s['n_strong_wins']}/losses={s['n_strong_losses']}"
              f"{'  robust=' + str(s.get('n_robust_wins')) if has_fdr else ''}")
        print("  top-10 wins:")
        for _, r in per.head(10).iterrows():
            tag = "  *lineage*" if lineage and r["tf"] in lineage else ""
            print(f"    {r['tf']:25s} Δ={r['delta']:+.3f}"
                  f"  {args.baseline}={r[f'nes_{args.baseline}']:+.3f}"
                  f"  {mt}={r[f'nes_{mt}']:+.3f}{tag}")
        print("  top-5 losses:")
        for _, r in per.tail(5).iterrows():
            tag = "  *lineage*" if lineage and r["tf"] in lineage else ""
            print(f"    {r['tf']:25s} Δ={r['delta']:+.3f}{tag}")

    pd.DataFrame(summary_rows).to_csv(out / "delta_summary.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        for mt in methods:
            if mt not in pv_nes.columns: continue
            both = pv_nes[[args.baseline, mt]].dropna()
            d = (both[mt] - both[args.baseline]).sort_values()
            if len(d) == 0: continue
            colors = ["tab:red" if v < -args.delta_threshold
                      else "tab:green" if v > args.delta_threshold
                      else "gray" for v in d.values]
            fig, ax = plt.subplots(figsize=(6, max(4, len(d) * 0.14)))
            ax.barh(range(len(d)), d.values, color=colors)
            ax.set_yticks(range(len(d)))
            labels = [f"{t} *" if t in lineage else t for t in d.index]
            ax.set_yticklabels(labels, fontsize=6)
            ax.axvline(0, color="black", lw=0.5)
            ax.axvline(args.delta_threshold, color="green", lw=0.5, ls="--")
            ax.axvline(-args.delta_threshold, color="red", lw=0.5, ls="--")
            ax.set_xlabel(f"Δ NES ({mt} − {args.baseline})")
            fig.tight_layout()
            fig.savefig(out / f"delta_{mt}_bar.png", dpi=150)
            plt.close(fig)
    except Exception as e:
        print(f"[delta] plot skipped: {e}")

    print(f"\n[delta] wrote {out}/")


if __name__ == "__main__":
    main()
