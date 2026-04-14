"""
k-selection rule comparison.

Tests candidate conservative k-rules against tail exhaustion and oracle
on both synthetic (.npz) and real (bone marrow) data.

Candidate rules:
  1. tail_exhaustion  — current default
  2. hard_cap         — min(tail_exhaustion, cap) for cap in {3,5,8}
  3. variance_fraction — smallest k retaining >= threshold of residual variance
  4. spectral_gap     — largest gap in log-spectrum after index 1
  5. elbow            — max-distance-from-chord in log-spectrum
  6. conservative_tail — tail exhaustion with tau=3.0, stable_m=5 (more conservative)

Evaluation:
  - Synthetic: per-TF AUPRC against ground-truth W
  - Bone marrow: per-TF AUPRC against CollecTRI edges
  - Oracle: best k from grid search (upper bound)

Output:
  results/k_selection/csv/k_rule_comparison.csv
  results/k_selection/csv/k_rule_summary.csv
  results/k_selection/figures/*.png
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from geobridge_model import GeoBridgeModel
from geobridge_model.spectral_k import (
    select_k_by_tail_exhaustion,
    _spectral_elbow_k,
    SpectralResult,
)
from benchmark.metrics import evaluate_all_metrics, target_precision_at_k


# ===================================================================
# k-selection rules
# ===================================================================

def rule_tail_exhaustion(S, **kw):
    """Current default: tau=2.0, stable_m=3."""
    spec = select_k_by_tail_exhaustion(S, max_k=40, tau=2.0, stable_m=3)
    return int(spec.k_selected)


def rule_conservative_tail(S, **kw):
    """More conservative tail exhaustion: tau=3.0, stable_m=5."""
    spec = select_k_by_tail_exhaustion(S, max_k=40, tau=3.0, stable_m=5)
    return int(spec.k_selected)


def rule_hard_cap(S, cap=5, **kw):
    """min(tail_exhaustion, cap)."""
    spec = select_k_by_tail_exhaustion(S, max_k=40, tau=2.0, stable_m=3)
    return min(int(spec.k_selected), cap)


def rule_hard_cap_3(S, **kw):
    return rule_hard_cap(S, cap=3)


def rule_hard_cap_5(S, **kw):
    return rule_hard_cap(S, cap=5)


def rule_hard_cap_8(S, **kw):
    return rule_hard_cap(S, cap=8)


def rule_hybrid_cap(S, cap=10, **kw):
    """min(conservative_tail, cap)."""
    k_ct = rule_conservative_tail(S)
    return min(k_ct, cap)


def rule_hybrid_cap_10(S, **kw):
    return rule_hybrid_cap(S, cap=10)


def rule_hybrid_cap_8(S, **kw):
    return rule_hybrid_cap(S, cap=8)


def rule_hybrid_cap_5(S, **kw):
    return rule_hybrid_cap(S, cap=5)


def rule_variance_fraction(S, threshold=0.10, **kw):
    """
    Smallest k such that the removed variance (sum of top-k squared SVs)
    is <= threshold fraction of total variance.
    i.e., we remove at most `threshold` of the total variance.
    """
    s = np.asarray(S, dtype=float)
    s = s[s > 0]
    total_var = np.sum(s ** 2)
    if total_var < 1e-12:
        return 1
    cumvar = np.cumsum(s ** 2) / total_var
    # We want the fraction REMOVED to be <= threshold
    # cumvar[k-1] = fraction of variance in top-k components
    # We want cumvar[k-1] <= threshold
    candidates = np.where(cumvar <= threshold)[0]
    if len(candidates) == 0:
        return 1  # even first component exceeds threshold → remove just 1
    return int(candidates[-1]) + 1  # +1 because k is 1-indexed


def rule_variance_fraction_05(S, **kw):
    return rule_variance_fraction(S, threshold=0.05)


def rule_variance_fraction_10(S, **kw):
    return rule_variance_fraction(S, threshold=0.10)


def rule_variance_fraction_20(S, **kw):
    return rule_variance_fraction(S, threshold=0.20)


def rule_spectral_gap(S, **kw):
    """
    Position of largest gap (drop) in log-spectrum, searching indices 1..cap.
    The largest gap often separates "signal" SVs from "noise" SVs.
    """
    s = np.asarray(S, dtype=float)
    s = s[s > 0]
    cap = min(30, len(s) - 1)
    if cap < 2:
        return 1
    log_s = np.log(s[:cap + 1])
    gaps = -np.diff(log_s)  # positive = drop
    # Skip index 0 (gap between SV1 and SV2 is often huge but uninformative)
    best = int(np.argmax(gaps[1:]) + 1) + 1  # +1 for skip, +1 for k indexing
    return max(1, min(best, cap))


def rule_elbow(S, **kw):
    """Max-distance-from-chord elbow in log-spectrum."""
    return _spectral_elbow_k(S, cap=25)


def rule_first_plateau(S, window=3, factor=1.5, **kw):
    """
    First k where smoothed log-spectrum slope drops below
    factor * median(tail slopes). Less aggressive than tail_exhaustion
    because it doesn't require stable_m consecutive.
    """
    s = np.asarray(S, dtype=float)
    s = s[s > 0]
    cap = min(40, len(s) - 1)
    if cap < 3:
        return 1
    log_s = np.log(s[:cap + 1])
    slope = np.abs(np.diff(log_s))
    # Simple moving average
    smoothed = np.convolve(slope, np.ones(window) / window, mode='valid')
    if len(smoothed) < 3:
        return 1
    # Tail baseline: median of last 30%
    tail_start = max(1, int(len(smoothed) * 0.7))
    tail_baseline = np.median(smoothed[tail_start:])
    threshold = factor * tail_baseline
    for i, v in enumerate(smoothed):
        if v <= threshold:
            return max(1, i + 1)
    return cap


# ===================================================================
# All rules registry
# ===================================================================

RULES = {
    "tail_exhaustion": rule_tail_exhaustion,
    "conservative_tail": rule_conservative_tail,
    "hard_cap_3": rule_hard_cap_3,
    "hard_cap_5": rule_hard_cap_5,
    "hard_cap_8": rule_hard_cap_8,
    "hybrid_cap_10": rule_hybrid_cap_10,
    "hybrid_cap_8": rule_hybrid_cap_8,
    "hybrid_cap_5": rule_hybrid_cap_5,
    "var_frac_05": rule_variance_fraction_05,
    "var_frac_10": rule_variance_fraction_10,
    "var_frac_20": rule_variance_fraction_20,
    "spectral_gap": rule_spectral_gap,
    "elbow": rule_elbow,
    "first_plateau": rule_first_plateau,
}


# ===================================================================
# Oracle: best k from grid search
# ===================================================================

def find_oracle_k(pre, W_or_evaluator, k_grid, project_y=True):
    """
    Find oracle k that maximizes the evaluation metric.

    W_or_evaluator: either a (G,G) ground-truth matrix (synthetic)
                    or a callable(S) -> float (real data)
    """
    best_k = 0
    best_metric = -np.inf
    results = {}

    for k in k_grid:
        if k >= len(pre["S"]):
            continue
        S = GeoBridgeModel.scores_from_streaming_stats(
            pre, k=k, project_y=project_y, normalize_score=True, eps=1e-12,
        )
        if callable(W_or_evaluator):
            metric = W_or_evaluator(S)
        else:
            m = evaluate_all_metrics(W_or_evaluator, S, topk=50)
            metric = m["AUPRC"]
        results[k] = metric
        if metric > best_metric:
            best_metric = metric
            best_k = k

    return best_k, best_metric, results


# ===================================================================
# Synthetic evaluation
# ===================================================================

def evaluate_synthetic(data_dir, outdir, max_reps=3):
    """Run k-rule comparison on synthetic .npz datasets."""
    data_dir = Path(data_dir)
    npz_paths = sorted(data_dir.glob("*.npz"))
    print(f"[SYNTH] Found {len(npz_paths)} datasets")

    # Focus on regimes where k matters
    target_regimes = {
        "baseline", "R2_high_overlap", "R7_active_tf_overlap",
        "R8_strong_multirank_confound", "R9_asymmetric_confound",
    }

    k_grid = list(range(0, 31))
    rows = []
    seen = {}

    for pi, path in enumerate(npz_paths):
        stem = path.stem
        parts = stem.split("__")
        if len(parts) == 3:
            topo, drift, rep_s = parts
            regime = "baseline"
        elif len(parts) == 4:
            topo, drift, rep_s, regime = parts
        else:
            continue

        if regime not in target_regimes:
            continue

        key = (topo, drift, regime)
        seen.setdefault(key, 0)
        if max_reps and seen[key] >= max_reps:
            continue
        seen[key] += 1

        rep = int(rep_s.replace("rep", ""))

        data = np.load(path, allow_pickle=True)
        X = np.asarray(data["X"], dtype=float)
        V = np.asarray(data["V"], dtype=float)
        pt = np.asarray(data["pseudotime"], dtype=float)
        W = np.asarray(data["W"], dtype=float)
        data.close()

        # Precompute
        pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
            X, V, pt, pair_seed=42,
        )

        # Oracle
        oracle_k, oracle_auprc, all_k_results = find_oracle_k(pre, W, k_grid)

        # k=0 baseline
        auprc_k0 = all_k_results.get(0, np.nan)

        # Each rule
        for rule_name, rule_fn in RULES.items():
            try:
                k_sel = rule_fn(pre["S"])
            except Exception as e:
                print(f"  [WARN] {rule_name} failed on {stem}: {e}")
                k_sel = 1

            k_sel = max(0, min(k_sel, len(pre["S"]) - 1))

            # Score at selected k
            auprc_at_k = all_k_results.get(k_sel, np.nan)
            if np.isnan(auprc_at_k):
                S_k = GeoBridgeModel.scores_from_streaming_stats(
                    pre, k=k_sel, project_y=True, normalize_score=True, eps=1e-12,
                )
                m = evaluate_all_metrics(W, S_k, topk=50)
                auprc_at_k = m["AUPRC"]

            # Also get target precision
            S_k = GeoBridgeModel.scores_from_streaming_stats(
                pre, k=k_sel, project_y=True, normalize_score=True, eps=1e-12,
            )
            tp = target_precision_at_k(W, S_k, ks=(20,))

            rows.append({
                "dataset": stem,
                "topology": topo,
                "drift": drift,
                "rep": rep,
                "regime": regime,
                "rule": rule_name,
                "k_selected": k_sel,
                "oracle_k": oracle_k,
                "auprc": auprc_at_k,
                "oracle_auprc": oracle_auprc,
                "auprc_k0": auprc_k0,
                "pct_of_oracle": auprc_at_k / oracle_auprc * 100 if oracle_auprc > 0 else np.nan,
                "target_precision_at_20": tp.get("target_precision_at_20", np.nan),
                "source": "synthetic",
            })

        if (pi + 1) % 10 == 0 or pi == 0:
            print(f"  [{pi+1}/{len(npz_paths)}] {stem}  oracle_k={oracle_k}")

    return pd.DataFrame(rows)


# ===================================================================
# Bone marrow evaluation
# ===================================================================

def evaluate_bonemarrow(outdir, data_dir=None):
    """Run k-rule comparison on bone marrow with CollecTRI ground truth."""
    import warnings
    warnings.filterwarnings("ignore")

    print("\n[BM] Loading bone marrow data...")

    # Load cached data or download
    import scanpy as sc
    import scvelo as scv

    cache = Path(data_dir) / "human_cd34_bone_marrow.h5ad" if data_dir else None
    if cache and cache.exists():
        adata = sc.read_h5ad(cache)
    else:
        adata = scv.datasets.bonemarrow()
        if cache:
            adata.write_h5ad(cache)

    # Preprocess
    scv.pp.filter_genes(adata, min_shared_counts=20)
    sc.pp.normalize_total(adata)
    sc.pp.log1p(adata)
    sc.pp.highly_variable_genes(adata, n_top_genes=2000, subset=True)
    sc.pp.pca(adata, n_comps=30)
    sc.pp.neighbors(adata, n_pcs=30, n_neighbors=30)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(adata, mode="stochastic")

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

    # Load ground truth
    import decoupler as dc
    if hasattr(dc, "op") and hasattr(dc.op, "collectri"):
        net = dc.op.collectri(organism="human")
    elif hasattr(dc, "get_collectri"):
        net = dc.get_collectri(organism="human", split_complexes=False)
    else:
        raise RuntimeError("Cannot load CollecTRI")
    print(f"[BM] CollecTRI: {len(net)} edges")

    # Match to gene universe
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
    print(f"[BM] {len(tf_targets)} evaluable TFs")

    # Build evaluator: mean per-TF AUPRC
    from sklearn.metrics import average_precision_score

    def bm_evaluator(S):
        """Mean per-TF AUPRC against CollecTRI."""
        G = len(genes)
        aps = []
        for tf, target_idxs in tf_targets.items():
            tf_idx = gene_to_idx[tf]
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
                continue
            aps.append(average_precision_score(labels, scores))
        return np.mean(aps) if aps else 0.0

    # Precompute
    print("[BM] Precomputing pairspace stats...")
    pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
        X, V, pt, pair_seed=42,
    )
    print(f"[BM] {pre['n_pairs']} pairs")

    # Oracle
    k_grid = list(range(0, 31))
    oracle_k, oracle_metric, all_k_results = find_oracle_k(
        pre, bm_evaluator, k_grid
    )
    auprc_k0 = all_k_results.get(0, np.nan)
    print(f"[BM] Oracle k={oracle_k}, metric={oracle_metric:.6f}, k=0={auprc_k0:.6f}")

    # Each rule
    rows = []
    for rule_name, rule_fn in RULES.items():
        try:
            k_sel = rule_fn(pre["S"])
        except Exception as e:
            print(f"  [WARN] {rule_name} failed: {e}")
            k_sel = 1

        k_sel = max(0, min(k_sel, len(pre["S"]) - 1))
        metric_at_k = all_k_results.get(k_sel, np.nan)
        if np.isnan(metric_at_k):
            S_k = GeoBridgeModel.scores_from_streaming_stats(
                pre, k=k_sel, project_y=True, normalize_score=True, eps=1e-12,
            )
            metric_at_k = bm_evaluator(S_k)

        rows.append({
            "dataset": "bonemarrow",
            "topology": "real",
            "drift": "N/A",
            "rep": 0,
            "regime": "real",
            "rule": rule_name,
            "k_selected": k_sel,
            "oracle_k": oracle_k,
            "auprc": metric_at_k,
            "oracle_auprc": oracle_metric,
            "auprc_k0": auprc_k0,
            "pct_of_oracle": metric_at_k / oracle_metric * 100 if oracle_metric > 0 else np.nan,
            "target_precision_at_20": np.nan,
            "source": "bonemarrow",
        })
        print(f"  {rule_name}: k={k_sel}, metric={metric_at_k:.6f}, "
              f"pct_oracle={metric_at_k/oracle_metric*100:.1f}%")

    return pd.DataFrame(rows)


# ===================================================================
# Plotting
# ===================================================================

def make_plots(df, figdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = Path(figdir)
    figdir.mkdir(parents=True, exist_ok=True)

    # ---- Plot 1: pct_of_oracle by rule, split by source ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, source in zip(axes, ["synthetic", "bonemarrow"]):
        sub = df[df["source"] == source]
        if len(sub) == 0:
            ax.set_visible(False)
            continue

        rule_means = sub.groupby("rule")["pct_of_oracle"].mean().sort_values(ascending=False)
        rule_stds = sub.groupby("rule")["pct_of_oracle"].std().fillna(0)

        colors = []
        for r in rule_means.index:
            if r == "tail_exhaustion":
                colors.append("#C44E52")
            elif "cap" in r:
                colors.append("#55A868")
            elif "var_frac" in r:
                colors.append("#4C72B0")
            elif r in ("spectral_gap", "elbow", "first_plateau"):
                colors.append("#CCB974")
            else:
                colors.append("#8172B2")

        bars = ax.barh(range(len(rule_means)), rule_means.values, color=colors,
                       xerr=rule_stds.reindex(rule_means.index).values, capsize=3)
        ax.set_yticks(range(len(rule_means)))
        ax.set_yticklabels(rule_means.index, fontsize=9)
        ax.set_xlabel("% of Oracle AUPRC")
        ax.set_title(f"{source}: Rule Performance (% of Oracle)")
        ax.axvline(100, color="red", ls=":", alpha=0.5)
        ax.invert_yaxis()

        # Annotate values
        for i, (val, name) in enumerate(zip(rule_means.values, rule_means.index)):
            ax.text(val + 1, i, f"{val:.1f}%", va="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(figdir / "rule_pct_oracle.png", dpi=150)
    plt.close(fig)
    print(f"[FIG] rule_pct_oracle.png")

    # ---- Plot 2: k_selected distribution by rule (synthetic only) ----
    synth = df[df["source"] == "synthetic"]
    if len(synth):
        rules_ordered = synth.groupby("rule")["pct_of_oracle"].mean().sort_values(ascending=False).index
        fig, ax = plt.subplots(figsize=(10, 6))

        rule_data = []
        rule_labels = []
        for rule in rules_ordered:
            vals = synth[synth["rule"] == rule]["k_selected"].values
            rule_data.append(vals)
            rule_labels.append(rule)

        # Add oracle
        oracle_vals = synth.groupby("dataset")["oracle_k"].first().values
        rule_data.append(oracle_vals)
        rule_labels.append("oracle")

        bp = ax.boxplot(rule_data, labels=rule_labels, vert=True, patch_artist=True)
        for patch in bp["boxes"][:-1]:
            patch.set_facecolor("#4C72B0")
            patch.set_alpha(0.6)
        bp["boxes"][-1].set_facecolor("#55A868")
        bp["boxes"][-1].set_alpha(0.6)

        ax.set_ylabel("k selected")
        ax.set_title("k Selected by Each Rule (synthetic)")
        ax.set_xticklabels(rule_labels, rotation=45, ha="right", fontsize=9)
        fig.tight_layout()
        fig.savefig(figdir / "k_selected_distribution.png", dpi=150)
        plt.close(fig)
        print(f"[FIG] k_selected_distribution.png")

    # ---- Plot 3: regime breakdown for top rules ----
    if len(synth):
        top_rules = synth.groupby("rule")["pct_of_oracle"].mean().nlargest(5).index.tolist()
        top_rules = [r for r in top_rules if r != "tail_exhaustion"] + ["tail_exhaustion"]

        regimes = sorted(synth["regime"].unique())
        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(regimes))
        width = 0.8 / len(top_rules)

        for ri, rule in enumerate(top_rules):
            means = []
            for regime in regimes:
                sub = synth[(synth["rule"] == rule) & (synth["regime"] == regime)]
                means.append(sub["pct_of_oracle"].mean() if len(sub) else 0)
            offset = (ri - len(top_rules) / 2 + 0.5) * width
            color = "#C44E52" if rule == "tail_exhaustion" else f"C{ri}"
            ax.bar(x + offset, means, width, label=rule, color=color, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(regimes, rotation=30, ha="right", fontsize=9)
        ax.set_ylabel("% of Oracle AUPRC")
        ax.set_title("Top Rules vs Tail Exhaustion by Regime")
        ax.legend(fontsize=8, loc="lower right")
        ax.axhline(100, color="red", ls=":", alpha=0.3)
        fig.tight_layout()
        fig.savefig(figdir / "regime_breakdown.png", dpi=150)
        plt.close(fig)
        print(f"[FIG] regime_breakdown.png")


# ===================================================================
# Main
# ===================================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synth-data-dir", default=None,
                    help="Directory with synthetic .npz files")
    ap.add_argument("--bm-data-dir", default=None,
                    help="Directory for bone marrow h5ad cache")
    ap.add_argument("--outdir", default="results/k_selection")
    ap.add_argument("--max-reps", type=int, default=3)
    ap.add_argument("--skip-synth", action="store_true")
    ap.add_argument("--skip-bm", action="store_true")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    csvdir = outdir / "csv"
    figdir = outdir / "figures"
    csvdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    dfs = []

    # Synthetic
    if not args.skip_synth and args.synth_data_dir:
        print("=" * 70)
        print("SYNTHETIC k-SELECTION COMPARISON")
        print("=" * 70)
        df_synth = evaluate_synthetic(args.synth_data_dir, outdir, max_reps=args.max_reps)
        dfs.append(df_synth)
        print(f"\n[SYNTH] {len(df_synth)} rows")

    # Bone marrow
    if not args.skip_bm and args.bm_data_dir:
        print("\n" + "=" * 70)
        print("BONE MARROW k-SELECTION COMPARISON")
        print("=" * 70)
        df_bm = evaluate_bonemarrow(outdir, data_dir=args.bm_data_dir)
        dfs.append(df_bm)
        print(f"\n[BM] {len(df_bm)} rows")

    if not dfs:
        print("[ERROR] No data sources specified. Use --synth-data-dir and/or --bm-data-dir")
        return

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(csvdir / "k_rule_comparison.csv", index=False)
    print(f"\n[SAVE] {csvdir / 'k_rule_comparison.csv'}")

    # Summary: mean pct_of_oracle by rule and source
    summary = df.groupby(["source", "rule"]).agg({
        "k_selected": ["mean", "std", "median"],
        "pct_of_oracle": ["mean", "std"],
        "auprc": ["mean", "std"],
    }).round(4)
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
    summary = summary.reset_index().sort_values(["source", "pct_of_oracle_mean"],
                                                 ascending=[True, False])
    summary.to_csv(csvdir / "k_rule_summary.csv", index=False)
    print(f"[SAVE] {csvdir / 'k_rule_summary.csv'}")

    # Plots
    print("\n[PLOT] Generating figures...")
    try:
        make_plots(df, figdir)
    except Exception as e:
        print(f"[PLOT ERROR] {e}")
        import traceback
        traceback.print_exc()

    # Print final summary
    print("\n" + "=" * 70)
    print("k-SELECTION RULE COMPARISON SUMMARY")
    print("=" * 70)

    for source in df["source"].unique():
        sub = df[df["source"] == source]
        print(f"\n  === {source.upper()} ===")
        rule_perf = sub.groupby("rule").agg({
            "k_selected": "mean",
            "pct_of_oracle": "mean",
            "auprc": "mean",
        }).sort_values("pct_of_oracle", ascending=False)

        oracle_k_mean = sub.groupby("rule")["oracle_k"].mean().iloc[0]
        auprc_k0 = sub["auprc_k0"].mean()

        print(f"  Oracle k (mean): {oracle_k_mean:.1f}")
        print(f"  AUPRC at k=0: {auprc_k0:.4f}")
        print(f"\n  {'Rule':<22} {'k_mean':>7} {'AUPRC':>8} {'%Oracle':>8}")
        print(f"  {'-'*22} {'-'*7} {'-'*8} {'-'*8}")
        for rule, row in rule_perf.iterrows():
            marker = " <--" if row["pct_of_oracle"] == rule_perf["pct_of_oracle"].max() else ""
            print(f"  {rule:<22} {row['k_selected']:>7.1f} {row['auprc']:>8.4f} "
                  f"{row['pct_of_oracle']:>7.1f}%{marker}")

    wall = time.perf_counter() - t0
    print(f"\n[DONE] {wall:.1f}s total. Results in {outdir}")


if __name__ == "__main__":
    main()
