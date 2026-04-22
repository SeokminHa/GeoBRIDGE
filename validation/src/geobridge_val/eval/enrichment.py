"""V1: GSEA preranked on TF regulons against a scored gene list per TF."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd


def gsea_per_tf(rankings: pd.DataFrame, gt_sets: dict[str, set[str]],
                n_perm: int = 1000, min_size: int = 5,
                max_size: int = 500) -> pd.DataFrame:
    """rankings: long format (regulator,target,score,...). gt_sets: {tf -> {targets}}."""
    from gseapy import prerank                  # lazy

    out = []
    tfs_present = rankings["regulator"].unique()
    for tf in tfs_present:
        if tf not in gt_sets or len(gt_sets[tf]) < min_size:
            continue
        sub = rankings.query("regulator == @tf")
        rnk = (sub.groupby("target")["score"].mean()
                  .sort_values(ascending=False))
        if len(rnk) < 50:
            continue
        # Sparse-output methods (e.g. SVD-based) frequently tie scores at 0,
        # causing gseapy's Rust backend to panic on near-zero variance.
        # Apply rank-preserving infinitesimal jitter to break ties while keeping
        # the descending order that the upstream sort already established.
        n = len(rnk)
        jitter = np.linspace(0, -1e-9, n)
        rnk = pd.Series(rnk.values + jitter, index=rnk.index)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res = prerank(rnk=rnk, gene_sets={tf: list(gt_sets[tf])},
                              permutation_num=n_perm, min_size=min_size,
                              max_size=max_size,
                              no_plot=True, verbose=False).res2d
        except BaseException as e:
            # pyo3 PanicException is BaseException, not Exception
            out.append({"tf": tf, "nes": None, "pval": None,
                        "error": f"{type(e).__name__}: {str(e)[:200]}",
                        "n_targets": len(gt_sets[tf])})
            continue
        out.append({
            "tf": tf,
            "nes": float(res["NES"].iloc[0]),
            "pval": float(res["NOM p-val"].iloc[0]),
            "fdr":  float(res["FDR q-val"].iloc[0]),
            "n_targets": len(gt_sets[tf]),
        })
    return pd.DataFrame(out)


def enrich_all_methods(rankings_long: pd.DataFrame, gt_sets: dict[str, set[str]],
                       **kw) -> pd.DataFrame:
    frames = []
    for m in rankings_long["method"].unique():
        df = gsea_per_tf(rankings_long.query("method == @m"), gt_sets, **kw)
        df["method"] = m
        frames.append(df)
    return pd.concat(frames, ignore_index=True)
