"""Bootstrap-CI summary of GeoBRIDGE k-sweep vs baseline.

Reads per_tf.csv (from run_v1_enrichment.py) with columns tf, method, nes.
For each method computes mean_nes with 95% TF-bootstrap CI, plus per-method
Δ_mean (method - baseline) and its CI. Flags CI-excludes-0.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def bootstrap_ci(x: np.ndarray, n_boot: int = 2000, seed: int = 0, ci: float = 95.0):
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    means = x[idx].mean(axis=1)
    lo, hi = (100 - ci) / 2, 100 - (100 - ci) / 2
    return float(x.mean()), float(np.percentile(means, lo)), float(np.percentile(means, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-tf", required=True)
    ap.add_argument("--baseline", default="correlation")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.per_tf)
    if not {"tf", "method", "nes"}.issubset(df.columns):
        raise SystemExit(f"required cols missing; got {df.columns.tolist()}")

    rows = []
    for m, grp in df.groupby("method"):
        nes = grp["nes"].dropna().values
        mean, lo, hi = bootstrap_ci(nes, n_boot=args.n_boot)
        rows.append({
            "method": m, "n_tf": len(nes),
            "mean_nes": mean, "ci_lo": lo, "ci_hi": hi,
            "median_nes": float(np.median(nes)) if len(nes) else float("nan"),
        })
    s = pd.DataFrame(rows)

    pv = df.pivot_table(index="tf", columns="method", values="nes", aggfunc="first")
    if args.baseline in pv.columns:
        for m in pv.columns:
            if m == args.baseline: continue
            d = (pv[m] - pv[args.baseline]).dropna().values
            if len(d) == 0: continue
            mean, lo, hi = bootstrap_ci(d, n_boot=args.n_boot)
            mask = s["method"] == m
            s.loc[mask, "delta_mean"] = mean
            s.loc[mask, "delta_ci_lo"] = lo
            s.loc[mask, "delta_ci_hi"] = hi
            s.loc[mask, "delta_excludes_0"] = bool(lo > 0 or hi < 0)
            s.loc[mask, "n_paired"] = len(d)

    s = s.sort_values("mean_nes", ascending=False)
    s.to_csv(out / "sweep_summary_boot.csv", index=False)

    def fmt(v):
        return "nan" if pd.isna(v) else f"{v:+.3f}"
    print(f"{'method':28s} {'n':>4} {'mean [95% CI]':>22} "
          f"{'Δ vs ' + args.baseline + ' [95% CI]':>28} {'sig':>4}")
    print("-" * 90)
    for _, r in s.iterrows():
        sig = "***" if r.get("delta_excludes_0") else ""
        delta = (f"{fmt(r.get('delta_mean'))} [{fmt(r.get('delta_ci_lo'))},"
                 f"{fmt(r.get('delta_ci_hi'))}]") if "delta_mean" in r else ""
        ci = f"{fmt(r['mean_nes'])} [{fmt(r['ci_lo'])},{fmt(r['ci_hi'])}]"
        print(f"{r['method']:28s} {int(r['n_tf']):>4d} {ci:>22s} {delta:>28s} {sig:>4s}")

    print(f"\n[sweep] wrote {out / 'sweep_summary_boot.csv'}")


if __name__ == "__main__":
    main()
