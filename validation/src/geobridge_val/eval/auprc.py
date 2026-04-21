"""V2: AUPRC / EPR against ChIP-derived GT edges, with bootstrap CIs."""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def auprc_per_tf(rankings: pd.DataFrame, gt_edges: set[tuple[str, str]],
                 min_pos: int = 5) -> pd.DataFrame:
    out = []
    for tf, sub in rankings.groupby("regulator"):
        y_true = sub[["regulator", "target"]].apply(tuple, axis=1).map(
            lambda e: e in gt_edges).values
        if y_true.sum() < min_pos:
            continue
        y_score = sub["score"].values
        ap = float(average_precision_score(y_true, y_score))
        prev = float(y_true.mean())
        out.append({"tf": tf, "auprc": ap, "epr": ap / prev if prev > 0 else None,
                    "n_pos": int(y_true.sum()), "n_total": int(len(sub)),
                    "prevalence": prev})
    return pd.DataFrame(out)


def bootstrap_auprc(rankings: pd.DataFrame, gt_edges: set[tuple[str, str]],
                    n_boot: int = 1000, seed: int = 0, min_pos: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    recs = []
    for tf, sub in rankings.groupby("regulator"):
        y_true = sub[["regulator", "target"]].apply(tuple, axis=1).map(
            lambda e: e in gt_edges).values
        y_score = sub["score"].values
        if y_true.sum() < min_pos:
            continue
        n = len(sub)
        aps = np.empty(n_boot)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            if y_true[idx].sum() < 2:
                aps[b] = np.nan; continue
            aps[b] = average_precision_score(y_true[idx], y_score[idx])
        aps = aps[~np.isnan(aps)]
        recs.append({
            "tf": tf,
            "auprc_mean": float(np.mean(aps)),
            "auprc_lo":   float(np.percentile(aps, 2.5)),
            "auprc_hi":   float(np.percentile(aps, 97.5)),
            "n_pos": int(y_true.sum()),
        })
    return pd.DataFrame(recs)


def eval_all_methods(rankings_long: pd.DataFrame,
                     gt_edges: set[tuple[str, str]],
                     n_boot: int = 1000) -> pd.DataFrame:
    frames = []
    for m in rankings_long["method"].unique():
        sub = rankings_long.query("method == @m")
        point = auprc_per_tf(sub, gt_edges)
        boot  = bootstrap_auprc(sub, gt_edges, n_boot=n_boot, seed=hash(m) % (2**31))
        merged = point.merge(boot, on="tf", how="left")
        merged["method"] = m
        frames.append(merged)
    return pd.concat(frames, ignore_index=True)
