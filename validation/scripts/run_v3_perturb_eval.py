"""V3: per-TF enrichment of GeoBRIDGE-predicted targets in CRISPRi DE gene lists.

Uses Mann-Whitney U on the rank distribution (GeoBRIDGE rank of targets vs
non-targets) restricted to genes expressed in the perturb arm.

Reports per-TF lift vs correlation baseline, stratified by WT-trajectory
variance of the perturbed TF (guards against the 'TF already varies in WT'
confound).
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import mannwhitneyu

from geobridge_val.io import load_rankings


def de_sets(pert: "sc.AnnData", sgrna_key: str, control: set[str],
            method: str = "wilcoxon", fdr: float = 0.05,
            min_cells: int = 30) -> dict[str, set[str]]:
    pert.obs["__grp"] = pert.obs[sgrna_key].astype(str)
    grp_counts = pert.obs["__grp"].value_counts()
    tfs = [g for g, n in grp_counts.items() if g not in control and n >= min_cells]

    sc.tl.rank_genes_groups(pert, groupby="__grp", reference="rest" if not control
                             else list(control)[0], method=method,
                             use_raw=False, n_genes=pert.n_vars)
    out = {}
    for tf in tfs:
        names = pd.DataFrame({
            "gene": pert.uns["rank_genes_groups"]["names"][tf],
            "padj": pert.uns["rank_genes_groups"]["pvals_adj"][tf],
        })
        de = set(names.query("padj < @fdr")["gene"].astype(str))
        if len(de) >= 10:
            out[tf] = de
    return out


def wt_tf_variance(wt: "sc.AnnData", tfs: list[str], layer: str = "Ms") -> pd.Series:
    X = wt.layers[layer] if layer in wt.layers else wt.X
    arr = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    var = pd.Series(arr.var(axis=0), index=wt.var_names)
    return var.reindex(tfs).fillna(0)


def per_tf_enrichment(rankings: pd.DataFrame, de: dict[str, set[str]]) -> pd.DataFrame:
    out = []
    for tf, tgts in de.items():
        sub = rankings.query("regulator == @tf")
        if sub.empty: continue
        s_in  = sub.loc[sub["target"].isin(tgts), "score"].values
        s_out = sub.loc[~sub["target"].isin(tgts), "score"].values
        if len(s_in) < 5 or len(s_out) < 20: continue
        u, p = mannwhitneyu(s_in, s_out, alternative="greater")
        lift = (np.mean(s_in) / max(np.mean(s_out), 1e-8)) if np.mean(s_out) else np.nan
        out.append({"tf": tf, "n_de": len(tgts), "mwu_u": u, "pval": p,
                    "median_rank_in":  float(np.median(sub.loc[sub["target"].isin(tgts), "rank"])),
                    "median_rank_out": float(np.median(sub.loc[~sub["target"].isin(tgts), "rank"])),
                    "lift_mean_score": float(lift)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rankings", required=True)
    ap.add_argument("--pert", required=True, help="perturb h5ad")
    ap.add_argument("--wt", required=True, help="WT h5ad (for variance stratification)")
    ap.add_argument("--tf-list", required=True)
    ap.add_argument("--sgrna-key", default="perturbation")
    ap.add_argument("--control", default="non-targeting,NT,safe-harbor")
    ap.add_argument("--fdr", type=float, default=0.05)
    ap.add_argument("--min-cells", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    pert = sc.read(args.pert)
    wt   = sc.read(args.wt)
    rankings = load_rankings(args.rankings)

    tfs = set(l.strip() for l in open(args.tf_list))
    rankings = rankings[rankings["regulator"].isin(tfs)]

    de = de_sets(pert, args.sgrna_key,
                 control=set(args.control.split(",")),
                 fdr=args.fdr, min_cells=args.min_cells)
    pd.Series({k: len(v) for k, v in de.items()}).to_csv(out / "de_set_sizes.csv")

    var = wt_tf_variance(wt, list(de))

    frames = []
    for m in rankings["method"].unique():
        df = per_tf_enrichment(rankings.query("method == @m"), de)
        df["method"] = m
        df["wt_var"] = df["tf"].map(var)
        frames.append(df)
    per_tf = pd.concat(frames, ignore_index=True)
    per_tf.to_csv(out / "per_tf.csv", index=False)

    # stratify by WT-variance quartile
    per_tf["var_q"] = pd.qcut(per_tf["wt_var"], q=4, labels=False, duplicates="drop")
    strat = (per_tf.groupby(["method", "var_q"])
                   .agg(mean_lift=("lift_mean_score", "mean"),
                        frac_signif=("pval", lambda s: (s < 0.05).mean()),
                        n_tf=("tf", "nunique"))
                   .reset_index())
    strat.to_csv(out / "by_variance.csv", index=False)
    print(strat.to_string(index=False))
    print(f"[v3] wrote {out}")


if __name__ == "__main__":
    main()
