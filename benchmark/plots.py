"""
Plot helpers for benchmarks.

Regime × method heatmaps and related figures summarize the **realism-stress** synthetic benchmark
(``benchmark.run_benchmark`` on regime-tagged ``.npz`` from ``benchmark.synthetic``): controlled
failure modes and deconfounding stress, not a claim of faithful biological realism.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import numpy.ma as ma
from matplotlib import colormaps
from matplotlib.patches import Patch

# Fig2 bar chart: method colors (match paper-style legend)
FIG2_METHOD_COLORS = {
    "GeoBRIDGE (Dual)": "#8B1538",
    "BRIDGE": "#1E3A5F",
    "Naive-Path": "#E8A838",
    "Naive-Fork": "#2D8F47",
    "Vel-Corr": "#C4704F",
    "Ridge": "#6B4C9A",
    "Granger": "#4A3728",
    "Scribe": "#D87DAC",
}

FIG2_TOPOLOGY_LABELS = {
    "LinearChain": "Linear",
    "Bifurcation": "Bifurcation",
    "Trifurcation": "Trifurcation",
    "ManyBifurcations": "Many Bifurc.",
}

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def save_fig2_bar_plots(df, outdir, topology_order=None, method_order=None, filename_stem="Fig2_bar_metrics"):
    """
    Three panel bar chart: mean ± std (across reps) for AUPRC, S-AUPRC, EDP@50.
    X = trajectory topologies; grouped bars = methods.
    """
    outdir = ensure_dir(outdir)
    metrics = ["AUPRC", "S-AUPRC", "EDP@50"]
    if topology_order is None:
        topo_seen = list(df["Topology"].unique())
        preferred = [
            "LinearChain",
            "Bifurcation",
            "Trifurcation",
            "ManyBifurcations",
        ]
        topologies = [t for t in preferred if t in topo_seen] + [t for t in topo_seen if t not in preferred]
    else:
        topologies = list(topology_order)

    if method_order is None:
        preferred_m = list(FIG2_METHOD_COLORS.keys())
        meth_seen = list(df["Method"].unique())
        methods = [m for m in preferred_m if m in meth_seen] + [m for m in meth_seen if m not in preferred_m]
    else:
        methods = list(method_order)

    x_labels = [FIG2_TOPOLOGY_LABELS.get(t, t) for t in topologies]
    n_t, n_m = len(topologies), len(methods)
    if n_t == 0 or n_m == 0:
        return

    width = min(0.8 / max(n_m, 1), 0.12)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharey=False)
    for ax, metric in zip(axes, metrics):
        x = np.arange(n_t, dtype=float)
        for mi, method in enumerate(methods):
            means = []
            stds = []
            for topo in topologies:
                sub = df[(df["Topology"] == topo) & (df["Method"] == method)]
                if len(sub) == 0:
                    means.append(np.nan)
                    stds.append(0.0)
                else:
                    v = sub[metric].astype(float)
                    means.append(float(v.mean()))
                    stds.append(float(v.std(ddof=1)) if len(v) > 1 else 0.0)
            color = FIG2_METHOD_COLORS.get(method, "#666666")
            pos = x + (mi - (n_m - 1) / 2) * width
            ax.bar(
                pos,
                means,
                width=width * 0.92,
                yerr=stds,
                label=method if ax is axes[0] else "_nolegend_",
                color=color,
                ecolor="#333333",
                capsize=2,
                error_kw={"linewidth": 0.8, "alpha": 0.85},
            )
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=0, fontsize=9)
        ax.set_ylabel(metric, fontsize=10)
        ax.set_title(metric, fontsize=11)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    pairs = [(h, lab) for h, lab in zip(handles, labels) if not str(lab).startswith("_")]
    if pairs:
        handles, labels = zip(*pairs)
        handles, labels = list(handles), list(labels)
    else:
        handles = [Patch(facecolor=FIG2_METHOD_COLORS.get(m, "#666666"), label=m) for m in methods]
        labels = list(methods)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.06),
        ncol=4,
        frameon=False,
        fontsize=7,
    )
    fig.suptitle("Fig2 benchmark (mean ± std over reps)", fontsize=12, y=1.01)
    fig.tight_layout(rect=[0, 0.12, 1, 0.96])
    out_png = outdir / f"{filename_stem}.png"
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_main_benchmark_plots(df, outdir):
    """Bar summaries for the realism-stress regime benchmark (mean metric by topology × model)."""
    outdir = ensure_dir(outdir)
    for metric in ["S-AUPRC", "AUPRC", "EDP@50", "AUROC"]:
        plt.figure(figsize=(10,5))
        grouped = df.groupby(["topology","model"], as_index=False)[metric].mean()
        topologies = list(grouped["topology"].unique())
        models = list(grouped["model"].unique())
        width = 0.11
        x = np.arange(len(topologies))
        for i, model in enumerate(models):
            sub = grouped[grouped["model"] == model].set_index("topology").reindex(topologies)
            plt.bar(x + i*width, sub[metric].values, width=width, label=model)
        plt.xticks(x + width*(len(models)-1)/2, topologies, rotation=20)
        plt.ylabel(metric)
        plt.title(f"Realism-stress benchmark: mean {metric}")
        plt.legend(frameon=False, fontsize=8)
        plt.tight_layout()
        plt.savefig(outdir / f"main_benchmark_{metric}.png", dpi=160)
        plt.close()

def save_k_sweep_plots(df, outdir):
    outdir = ensure_dir(outdir)
    for topology, sub in df.groupby("topology"):
        for metric in ["S-AUPRC", "AUPRC", "EDP@50", "AUROC"]:
            plt.figure(figsize=(8,5))
            for mode, ss in sub.groupby("project_y_mode"):
                g = ss.groupby("k", as_index=False)[metric].mean().sort_values("k")
                plt.plot(g["k"], g[metric], marker="o", label=mode)
            plt.xlabel("k")
            plt.ylabel(metric)
            plt.title(f"{metric} vs k: {topology}")
            plt.legend(frameon=False)
            plt.tight_layout()
            plt.savefig(outdir / f"k_sweep_{topology}_{metric}.png", dpi=160)
            plt.close()

def save_pair_scale_plots(df, outdir):
    outdir = ensure_dir(outdir)
    for topology, sub in df.groupby("topology"):
        for metric in ["S-AUPRC", "AUPRC", "EDP@50", "AUROC"]:
            g = sub.groupby("cell_frac", as_index=False)[metric].mean()
            plt.figure(figsize=(7,5))
            plt.plot(g["cell_frac"], g[metric], marker="o")
            plt.xlabel("cell fraction")
            plt.ylabel(metric)
            plt.title(f"{metric} vs cell fraction: {topology}")
            plt.tight_layout()
            plt.savefig(outdir / f"pair_scale_{topology}_{metric}.png", dpi=160)
            plt.close()

def save_cell_scale_plots(df, outdir):
    outdir = ensure_dir(outdir)
    for topology, sub in df.groupby("topology"):
        for metric in ["S-AUPRC", "AUPRC", "EDP@50", "AUROC"]:
            g = sub.groupby("n_cells", as_index=False)[metric].mean().sort_values("n_cells")
            plt.figure(figsize=(7, 5))
            plt.plot(g["n_cells"], g[metric], marker="o")
            plt.xlabel("n_cells")
            plt.ylabel(metric)
            plt.title(f"{metric} vs n_cells: {topology}")
            plt.tight_layout()
            plt.savefig(outdir / f"cell_scale_{topology}_{metric}.png", dpi=160)
            plt.close()

def save_pair_budget_sweep_plots(df, outdir):
    outdir = ensure_dir(outdir)
    # Expect long-form with columns: method, pairs_per_cell, metric_name, value
    metric_list = ["S-AUPRC", "AUPRC", "EDP@50", "AUROC"]
    for metric in metric_list:
        sub = df[df["metric_name"] == metric]
        if not len(sub):
            continue
        plt.figure(figsize=(8, 5))
        for method, ss in sub.groupby("method"):
            g = ss.groupby("pairs_per_cell", as_index=False)["value"].mean().sort_values("pairs_per_cell")
            plt.plot(g["pairs_per_cell"], g["value"], marker="o", label=method)
        plt.xlabel("pairs per cell")
        plt.ylabel(metric)
        plt.title(f"{metric} vs pair budget")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(outdir / f"pair_budget_sweep_{metric}.png", dpi=160)
        plt.close()

    # Runtime plot (mean seconds per dataset/method)
    if "runtime" in df.columns:
        plt.figure(figsize=(8, 5))
        for method, ss in df[df["metric_name"] == "runtime"].groupby("method"):
            g = ss.groupby("pairs_per_cell", as_index=False)["value"].mean().sort_values("pairs_per_cell")
            plt.plot(g["pairs_per_cell"], g["value"], marker="o", label=method)
        plt.xlabel("pairs per cell")
        plt.ylabel("runtime (s)")
        plt.title("Runtime vs pair budget")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(outdir / "pair_budget_sweep_runtime.png", dpi=160)
        plt.close()

def save_spectrum_plots(df, outdir):
    outdir = ensure_dir(outdir)
    for (topology, drift_type), sub in df.groupby(["topology","drift_type"]):
        rep0 = sub[sub["rep"] == sub["rep"].min()].sort_values("component")
        plt.figure(figsize=(7,5))
        plt.plot(rep0["component"], rep0["singular_value"], marker="o")
        if len(rep0):
            k_sel = int(rep0["k_selected"].iloc[0])
            plt.axvline(k_sel, color="red", linestyle="--", label=f"k={k_sel}")
        plt.xlabel("component")
        plt.ylabel("singular value")
        plt.title(f"Spectrum: {topology} / {drift_type}")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(outdir / f"spectrum_{topology}_{drift_type}.png", dpi=160)
        plt.close()


def _regime_heatmap_title_suffix(metric: str) -> str:
    higher = ("AUPRC", "target_precision_at_20", "signal_preservation_corr")
    lower = ("pseudo_hub_fdr_at_20", "residual_confound_r2")
    if metric in higher:
        return " (higher better)"
    if metric in lower:
        return " (lower better)"
    return ""


def save_regime_realism_heatmaps(df, outdir):
    """
    Realism-stress benchmark: mean metric over reps, scenario_regime × model.

    Missing combinations become NaN in the pivot; those cells use the colormap's "bad" color
    (light gray) instead of blending into viridis.
    """
    outdir = ensure_dir(outdir)
    if "scenario_regime" not in df.columns:
        return
    metrics = [
        "AUPRC",
        "pseudo_hub_fdr_at_20",
        "target_precision_at_20",
        "signal_preservation_corr",
        "residual_confound_r2",
    ]
    cmap = colormaps["viridis"].copy()
    cmap.set_bad(color="#d9d9d9")
    for metric in metrics:
        if metric not in df.columns:
            continue
        sub = df.groupby(["scenario_regime", "model"], as_index=False)[metric].mean()
        pivot = sub.pivot(index="scenario_regime", columns="model", values=metric)
        if pivot.size == 0:
            continue
        vals = pivot.values.astype(float)
        masked = ma.masked_invalid(vals)
        valid = vals[np.isfinite(vals)]
        if valid.size == 0:
            vmin, vmax = 0.0, 1.0
        elif np.allclose(valid.min(), valid.max()):
            c = float(valid.min())
            vmin, vmax = c - 0.05, c + 0.05
        else:
            vmin, vmax = float(valid.min()), float(valid.max())
        fig, ax = plt.subplots(figsize=(max(8, 1.2 * pivot.shape[1]), max(3.5, 0.45 * pivot.shape[0])))
        im = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels(list(pivot.columns), rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels(list(pivot.index), fontsize=8)
        suf = _regime_heatmap_title_suffix(metric)
        ax.set_title(f"Mean {metric}: scenario_regime × model{suf}")
        fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
        fig.tight_layout()
        safe = metric.replace("@", "_at_").replace(" ", "_")
        fig.savefig(outdir / f"heatmap_regime_method__{safe}.png", dpi=170, bbox_inches="tight")
        plt.close(fig)


def save_realism_summary_plots(df, outdir):
    """
    Realism-stress benchmark: mean AUPRC vs scenario_regime, one curve per model
    (aggregated over topologies / drifts / reps).
    """
    outdir = ensure_dir(outdir)
    if "scenario_regime" not in df.columns or "AUPRC" not in df.columns:
        return
    g = df.groupby(["scenario_regime", "model"], as_index=False)["AUPRC"].mean()
    regimes = sorted(g["scenario_regime"].unique())
    if len(regimes) <= 1:
        return
    x = np.arange(len(regimes))
    models = list(g["model"].unique())
    fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(regimes)), 4.5))
    for m in models:
        ys = [
            float(g[(g["scenario_regime"] == r) & (g["model"] == m)]["AUPRC"].iloc[0])
            if len(g[(g["scenario_regime"] == r) & (g["model"] == m)])
            else float("nan")
            for r in regimes
        ]
        ax.plot(x, ys, marker="o", label=m, lw=1.4)
    ax.set_xticks(x)
    ax.set_xticklabels(regimes, rotation=25, ha="right", fontsize=8)
    ax.set_ylabel("Mean AUPRC")
    ax.set_title("AUPRC vs realism regime (mean over topology × drift × rep)")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(outdir / "realism_stratified_auprc_by_regime.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
