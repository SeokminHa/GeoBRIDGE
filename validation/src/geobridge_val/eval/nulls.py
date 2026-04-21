"""Null controls: shuffle-label, degree-preserved, gene-matched."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def shuffle_label_auprc(rankings: pd.DataFrame, gt_edges: set[tuple[str, str]],
                        n_perm: int = 100, seed: int = 0) -> pd.DataFrame:
    """Shuffle regulator labels across rows, preserving the score distribution."""
    rng = np.random.default_rng(seed)
    out = []
    for m, sub in rankings.groupby("method"):
        aps = []
        for _ in range(n_perm):
            perm = sub.copy()
            perm["regulator"] = rng.permutation(perm["regulator"].values)
            y_true = perm[["regulator", "target"]].apply(tuple, axis=1).map(
                lambda e: e in gt_edges).values
            if y_true.sum() < 5: continue
            aps.append(average_precision_score(y_true, perm["score"].values))
        out.append({"method": m, "null_auprc_mean": float(np.mean(aps)),
                    "null_auprc_lo": float(np.percentile(aps, 2.5)),
                    "null_auprc_hi": float(np.percentile(aps, 97.5))})
    return pd.DataFrame(out)


def gene_matched_null(rankings: pd.DataFrame, expr_mean: pd.Series,
                      n_bins: int = 10, seed: int = 0) -> pd.DataFrame:
    """Replace predicted targets per TF with expression-matched random targets.

    expr_mean: pd.Series indexed by gene name → mean expression.
    """
    rng = np.random.default_rng(seed)
    df = rankings.copy()
    df["bin"] = pd.qcut(df["target"].map(expr_mean).fillna(0), n_bins,
                        labels=False, duplicates="drop")
    permuted = df.groupby("bin", group_keys=False).apply(
        lambda g: g.assign(target=rng.permutation(g["target"].values))
    )
    return permuted.drop(columns="bin")
