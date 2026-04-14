"""
Synthetic **realism-stress** benchmark driver.

Evaluates regime-perturbed ``.npz`` datasets from ``benchmark.synthetic`` (pseudo-hubs, noisy
pseudotime, combined real-like stressors, etc.). The goal is controlled *failure-mode* and
deconfounding stress—not a claim that simulations are biologically faithful.

This path is intentionally distinct from the Fig2 **structural stress** scripts
(``benchmark.fig2_benchmark_paper``, ``benchmark.fig2_rich_simulation``), which engineer
misspecification of pooled global linear **X→V** maps for method comparison.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
from geobridge_model import GeoBridgeConfig, GeoBridgeModel
from geobridge_model.spectral_k import select_k_by_tail_exhaustion
from .baselines import method_pairwise_raw, method_corr, method_ridge, method_cell_pca_preresidualize, method_pair_pca_direct
from . import benchmark_diagnostics as bd
from .metrics import evaluate_all_metrics, evaluate_deconfounding_metrics
from .pair_cli import add_pair_construction_args, pair_fields_from_args
from .plots import save_main_benchmark_plots, save_regime_realism_heatmaps, save_realism_summary_plots
from .scenario_meta import load_scenario_meta, meta_to_flat_row

MODEL_NAMES = ["geobridge_dual","geobridge_xonly","pairwise_raw","corr","ridge","cell_pca_preresidualize","pair_pca_direct"]

def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Run the regime-based synthetic realism-stress benchmark on .npz data "
            "(tables/ + figures/ under --outdir)."
        )
    )
    p.add_argument("--data-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument(
        "--plots-outdir",
        default=None,
        help="Figures directory (default: OUTDIR/figures). Tables always go under OUTDIR/tables/.",
    )
    p.add_argument("--models", nargs="+", default=MODEL_NAMES)
    p.add_argument("--max-k", type=int, default=40)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--stable-m", type=int, default=3)
    p.add_argument("--tail-fraction", type=float, default=0.2)
    p.add_argument("--tail-min-start", type=int, default=20)
    p.add_argument("--slope-window", type=int, default=3)
    p.add_argument("--baseline-k", type=int, default=3)
    add_pair_construction_args(p)
    p.add_argument(
        "--no-pairspace-cache",
        action="store_true",
        help=(
            "Disable on-disk cache for GeoBridge streaming pairspace precompute. "
            "Default: cache enabled under DATA_DIR/.pairspace_cache (speeds re-runs and extra models)."
        ),
    )
    p.add_argument(
        "--pairspace-cache-dir",
        default=None,
        help="Directory for pairspace precompute cache (default: DATA_DIR/.pairspace_cache).",
    )
    p.add_argument(
        "--no-incremental-checkpoints",
        action="store_true",
        help=(
            "Disable incremental writes: by default, tables/benchmark_results.csv is saved after each model, "
            "and tables/scenario_diagnostics.csv + figures/* refresh after each dataset (safe if a long run stops early)."
        ),
    )
    return p.parse_args()

def parse_dataset_stem(stem: str):
    """
    topology__drift__repN (legacy) or topology__drift__repN__regime (realism benchmark).
    """
    parts = stem.split("__")
    if len(parts) == 3:
        topology, drift_type, rep_s = parts
        scenario_regime = "baseline"
    elif len(parts) == 4:
        topology, drift_type, rep_s, scenario_regime = parts
    else:
        raise ValueError(
            f"stem must have 3 parts (topology__drift__repK) or 4 (topology__drift__repK__regime); got {len(parts)}"
        )
    rep = int(rep_s.replace("rep", ""))
    return topology, drift_type, rep, scenario_regime

def run_model(name, X, V, pseudotime, args, pre_stats=None):
    if name in {"geobridge_dual","geobridge_xonly"}:
        # Reuse the same pairspace decomposition for dual/xonly.
        cfg = GeoBridgeConfig(
            max_k=args.max_k, tau=args.tau, stable_m=args.stable_m,
            tail_fraction=args.tail_fraction, tail_min_start=args.tail_min_start,
            slope_window=args.slope_window, project_y=(name=="geobridge_dual"), normalize_score=True,
            **pair_fields_from_args(args),
        )
        if pre_stats is None:
            pre_stats = GeoBridgeModel.precompute_pairspace_stats_streaming(
                X,
                V,
                pseudotime,
                min_forward_dt=cfg.min_forward_dt,
                max_forward_dt=cfg.max_forward_dt,
                max_pairs_per_cell=cfg.max_pairs_per_cell,
                pair_seed=cfg.pair_seed,
                stream_num_threads=cfg.stream_num_threads,
            )
        spec = select_k_by_tail_exhaustion(
            pre_stats["S"],
            cfg.max_k,
            cfg.tail_fraction,
            cfg.tail_min_start,
            cfg.slope_window,
            cfg.stable_m,
            cfg.tau,
        )
        k_sel = int(spec.k_selected)
        scores = GeoBridgeModel.scores_from_streaming_stats(
            pre_stats,
            k=k_sel,
            project_y=cfg.project_y,
            normalize_score=cfg.normalize_score,
            eps=cfg.eps,
        )
        return scores, k_sel
    if name == "pairwise_raw": return method_pairwise_raw(X, V, pseudotime)
    if name == "corr": return method_corr(X, V, pseudotime)
    if name == "ridge": return method_ridge(X, V, pseudotime)
    if name == "cell_pca_preresidualize": return method_cell_pca_preresidualize(X, V, pseudotime, k=args.baseline_k)
    if name == "pair_pca_direct": return method_pair_pca_direct(X, V, pseudotime, k=args.baseline_k)
    raise ValueError(name)


def _pairspace_cache_key(path: Path, cfg: GeoBridgeConfig) -> str:
    st = path.stat()
    payload = {
        "path": str(path.resolve()),
        "mtime_ns": st.st_mtime_ns,
        "size": st.st_size,
        "min_forward_dt": cfg.min_forward_dt,
        "max_forward_dt": cfg.max_forward_dt,
        "max_pairs_per_cell": cfg.max_pairs_per_cell,
        "pair_seed": cfg.pair_seed,
        "stream_num_threads": cfg.stream_num_threads,
    }
    h = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    return h[:40]


def _pairspace_cache_file(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"pairspace_{key}.npz"


def _load_pairspace_cache(cache_path: Path) -> dict | None:
    if not cache_path.is_file():
        return None
    try:
        z = np.load(cache_path, allow_pickle=False)
        pre = {
            "n_pairs": int(z["n_pairs"].item()),
            "G": int(z["G"].item()),
            "mu_dx": np.asarray(z["mu_dx"]),
            "C": np.asarray(z["C"]),
            "M": np.asarray(z["M"]),
            "Y": np.asarray(z["Y"]),
            "S": np.asarray(z["S"]),
            "Vt": np.asarray(z["Vt"]),
        }
        z.close()
        return pre
    except (OSError, ValueError, KeyError):
        return None


def _save_pairspace_cache(cache_path: Path, pre: dict) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        n_pairs=np.int64(pre["n_pairs"]),
        G=np.int64(pre["G"]),
        mu_dx=pre["mu_dx"],
        C=pre["C"],
        M=pre["M"],
        Y=pre["Y"],
        S=pre["S"],
        Vt=pre["Vt"],
    )


def _atomic_df_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp, index=False)
    tmp.replace(path)


def _checkpoint_result_tables(rows: list, diagnostic_rows: list, tabdir: Path) -> None:
    if rows:
        _atomic_df_to_csv(pd.DataFrame(rows), tabdir / "benchmark_results.csv")
    if diagnostic_rows:
        _atomic_df_to_csv(pd.DataFrame(diagnostic_rows), tabdir / "scenario_diagnostics.csv")


def _checkpoint_benchmark_results_only(rows: list, tabdir: Path) -> None:
    if rows:
        _atomic_df_to_csv(pd.DataFrame(rows), tabdir / "benchmark_results.csv")


def _checkpoint_plots(rows: list, plots_outdir: str) -> None:
    if not rows or not plots_outdir:
        return
    df = pd.DataFrame(rows)
    save_main_benchmark_plots(df, plots_outdir)
    save_regime_realism_heatmaps(df, plots_outdir)
    save_realism_summary_plots(df, plots_outdir)


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    if not data_dir.is_dir():
        raise SystemExit(f"Data directory does not exist or is not a directory: {data_dir.resolve()}")

    npz_paths = sorted(data_dir.glob("*.npz"))
    if not npz_paths:
        raise SystemExit(
            f"No .npz files found under {data_dir.resolve()}.\n"
            "The benchmark expects simulated datasets as *.npz with keys X, V, pseudotime, W, "
            "and filenames like: topology__drift_type__rep0.npz or topology__drift_type__rep0__R6_combined_real_like.npz."
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tabdir = outdir / "tables"
    figdir = outdir / "figures"
    tabdir.mkdir(parents=True, exist_ok=True)
    figdir.mkdir(parents=True, exist_ok=True)
    plots_outdir = args.plots_outdir or str(figdir)
    if args.no_pairspace_cache:
        pairspace_cache_dir: Path | None = None
    else:
        pairspace_cache_dir = Path(args.pairspace_cache_dir) if args.pairspace_cache_dir else (data_dir / ".pairspace_cache")
    rows = []
    diagnostic_rows: list[dict] = []
    t_bench0 = time.perf_counter()
    print(
        f"[BENCH] Starting benchmark evaluation over {len(npz_paths)} datasets "
        f"({len(args.models)} models each) -> {outdir.resolve()} "
        f"(tables -> {tabdir.resolve()}, figures -> {Path(plots_outdir).resolve()})"
    )
    print(
        "[BENCH] Realism-stress benchmark: each .npz is loaded once; all models share the same X, V, W, pseudotime. "
        "Data come from `python -m benchmark.synthetic` (not regenerated here)."
    )
    if pairspace_cache_dir is not None and any(m in {"geobridge_dual", "geobridge_xonly"} for m in args.models):
        print(f"[BENCH] Pairspace precompute cache ON -> {pairspace_cache_dir.resolve()}")
    elif pairspace_cache_dir is None and any(m in {"geobridge_dual", "geobridge_xonly"} for m in args.models):
        print("[BENCH] Pairspace precompute cache OFF (--no-pairspace-cache)")
    incremental = not args.no_incremental_checkpoints
    if incremental:
        print(
            "[BENCH] Incremental checkpoints ON (tables/benchmark_results.csv after each model; "
            "tables/scenario_diagnostics.csv + figures/ after each dataset)."
        )
    n_ds = len(npz_paths)
    n_m = len(args.models)
    for di, path in enumerate(npz_paths, start=1):
        print(f"[DATASET] ({di}/{n_ds}) Loading {path.stem}")
        data = np.load(path, allow_pickle=True)
        X = np.asarray(data["X"], dtype=float, order="C")
        V = np.asarray(data["V"], dtype=float, order="C")
        pseudotime = np.asarray(data["pseudotime"], dtype=float, order="C")
        W = np.asarray(data["W"], dtype=float, order="C")
        data.close()
        try:
            topology, drift_type, rep, scenario_regime = parse_dataset_stem(path.stem)
        except Exception as e:
            raise SystemExit(
                f"Invalid dataset filename stem {path.stem!r}: expected topology__drift__repN "
                f"or topology__drift__repN__regime. Underlying error: {e}"
            ) from e
        meta = load_scenario_meta(path)
        if str(meta.get("scenario_regime", "baseline")) == "baseline" and scenario_regime != "baseline":
            meta["scenario_regime"] = scenario_regime
        meta_row = meta_to_flat_row(meta)
        diag = bd.compute_dataset_diagnostics(
            np.asarray(W, float),
            np.asarray(X, float),
            np.asarray(V, float),
            np.asarray(pseudotime, float),
            topology=topology,
            meta=meta,
        )
        diag["dataset"] = path.stem
        diag["topology"] = topology
        diag["drift_type"] = drift_type
        diag["scenario_regime"] = meta.get("scenario_regime", scenario_regime)
        diag["ridge_manybifurcations_hypothesis"] = bd.explain_ridge_many_bifurcations_row(diag, topology)
        diagnostic_rows.append(diag)

        pre_stats = None
        t_pre = float("nan")
        if any(m in {"geobridge_dual","geobridge_xonly"} for m in args.models):
            # Compute once per dataset and reuse across both geobridge variants.
            default_cfg = GeoBridgeConfig(**pair_fields_from_args(args))
            cache_path = None
            if pairspace_cache_dir is not None:
                ck = _pairspace_cache_key(path, default_cfg)
                cache_path = _pairspace_cache_file(pairspace_cache_dir, ck)
            t_pre0 = time.perf_counter()
            if cache_path is not None:
                pre_stats = _load_pairspace_cache(cache_path)
                if pre_stats is not None:
                    t_pre = time.perf_counter() - t_pre0
                    print(f"[BENCH] Pairspace precompute for {path.stem} from cache ({t_pre:.2f}s)")
            if pre_stats is None:
                t_pre0 = time.perf_counter()
                pre_stats = GeoBridgeModel.precompute_pairspace_stats_streaming(
                    X,
                    V,
                    pseudotime,
                    min_forward_dt=default_cfg.min_forward_dt,
                    max_forward_dt=default_cfg.max_forward_dt,
                    max_pairs_per_cell=default_cfg.max_pairs_per_cell,
                    pair_seed=default_cfg.pair_seed,
                    stream_num_threads=default_cfg.stream_num_threads,
                )
                t_pre = time.perf_counter() - t_pre0
                print(f"[BENCH] Pairspace precompute for {path.stem} done ({t_pre:.2f}s)")
                if cache_path is not None:
                    try:
                        _save_pairspace_cache(cache_path, pre_stats)
                        print(f"[BENCH] Saved pairspace cache {cache_path.name}")
                    except OSError as e:
                        print(f"[BENCH] Warning: could not write pairspace cache ({e})")
        for name in args.models:
            print(f"[MODEL] Running {name}")
            t_m0 = time.perf_counter()
            scores, k_sel = run_model(name, X, V, pseudotime, args, pre_stats=pre_stats)
            t_model = time.perf_counter() - t_m0
            print(f"[MODEL] Finished {name} ({t_model:.2f}s)")
            metrics = evaluate_all_metrics(W, scores, topk=50)
            deco = evaluate_deconfounding_metrics(
                np.asarray(W, float),
                np.asarray(scores, float),
                np.asarray(X, float),
                np.asarray(V, float),
                np.asarray(pseudotime, float),
                pseudo_hub_indices=meta.get("pseudo_hub_indices") or [],
            )
            rows.append(
                {
                    **meta_row,
                    "dataset": path.stem,
                    "topology": topology,
                    "drift_type": drift_type,
                    "rep": rep,
                    "scenario_regime": str(meta.get("scenario_regime", scenario_regime)),
                    "model": name,
                    "k_selected": k_sel,
                    "seconds_pairspace_precompute": t_pre,
                    "seconds_model": t_model,
                    **metrics,
                    **deco,
                }
            )
            if incremental:
                _checkpoint_benchmark_results_only(rows, tabdir)
                print(
                    f"[SAVE] checkpoint tables/benchmark_results.csv ({len(rows)} rows) after model={name} dataset={path.stem}"
                )
        if incremental:
            _checkpoint_result_tables(rows, diagnostic_rows, tabdir)
            try:
                _checkpoint_plots(rows, plots_outdir)
                print(f"[SAVE] checkpoint tables/scenario_diagnostics.csv + figures/ (after dataset {path.stem})")
            except Exception as e:
                print(f"[SAVE] checkpoint plots skipped: {e}")
    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise SystemExit("Benchmark produced no result rows (every dataset run failed or was skipped).")

    _checkpoint_result_tables(rows, diagnostic_rows, tabdir)
    print(f"[SAVE] Wrote {tabdir / 'benchmark_results.csv'}")
    print(f"[SAVE] Wrote {tabdir / 'scenario_diagnostics.csv'}")

    grp_cols = ["topology", "drift_type", "scenario_regime", "model"]
    core_metrics = ["S-AUPRC", "AUPRC", "EDP@50", "AUROC"]
    ext_metrics = [
        c
        for c in df.columns
        if c.startswith("hub_trap_rate_at_")
        or c.startswith("pseudo_hub_fdr_at_")
        or c.startswith("target_precision_at_")
        or c
        in (
            "median_target_pct_rank",
            "signal_preservation_corr",
            "spurious_mass_ratio",
            "residual_confound_r2",
        )
    ]
    agg_cols = [
        c
        for c in core_metrics + ext_metrics
        if c in df.columns and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not agg_cols:
        agg_cols = [c for c in core_metrics if c in df.columns and pd.api.types.is_numeric_dtype(df[c])]
    summary = df.groupby(grp_cols, as_index=False)[agg_cols].agg(["mean", "std"])
    summary.columns = [
        "_".join([str(x) for x in c if str(x) != ""]).strip("_")
        for c in summary.columns.to_flat_index()
    ]
    summary.to_csv(tabdir / "benchmark_summary.csv", index=False)
    print(f"[SAVE] Wrote {tabdir / 'benchmark_summary.csv'}")

    _write_benchmark_realism_summary_md(tabdir / "benchmark_realism_summary.md", df, diagnostic_rows)
    print(f"[SAVE] Wrote {tabdir / 'benchmark_realism_summary.md'}")
    _write_metric_interpretation_md(tabdir / "metric_interpretation.md")
    print(f"[SAVE] Wrote {tabdir / 'metric_interpretation.md'}")
    _write_benchmark_roles_md(tabdir / "benchmark_roles.md")
    print(f"[SAVE] Wrote {tabdir / 'benchmark_roles.md'}")
    diag_df = pd.DataFrame(diagnostic_rows) if diagnostic_rows else pd.DataFrame()
    wrote_topo_diag = _write_topology_model_diagnostics_summary(df, diag_df, tabdir)
    if wrote_topo_diag:
        print(f"[SAVE] Wrote {tabdir / 'topology_model_diagnostics_summary.csv'}")

    if plots_outdir:
        try:
            _checkpoint_plots(rows, plots_outdir)
            print(f"[SAVE] Wrote figures under {plots_outdir}")
        except Exception as e:
            print(f"[SAVE] Final plots failed: {e}")
    bench_wall_s = time.perf_counter() - t_bench0
    table_note = (
        f"{tabdir.resolve() / 'benchmark_results.csv'}, "
        f"{tabdir.resolve() / 'benchmark_summary.csv'}, "
        f"{tabdir.resolve() / 'scenario_diagnostics.csv'}"
    )
    if wrote_topo_diag:
        table_note += f", {tabdir.resolve() / 'topology_model_diagnostics_summary.csv'}"
    print(
        f"[BENCH] Complete. Total wall {bench_wall_s:.2f}s. Tables: {table_note} | "
        f"Figures: {Path(plots_outdir).resolve()}"
    )


def _write_metric_interpretation_md(path: Path) -> None:
    text = """# Metric interpretation (synthetic realism-stress benchmark)

Plain-language directionality for key columns in `benchmark_results.csv` and related tables.
See also `benchmark_roles.md` for how this benchmark differs from Fig2 hard mode.

| Metric | Better when | Notes |
|--------|-------------|-------|
| **AUPRC** | Higher | Area under precision–recall for ranking true regulatory edges. |
| **S-AUPRC** | Higher | Signed / directional variant of AUPRC used in this codebase. |
| **AUROC** | Higher | ROC area for the same edge-ranking task. |
| **EDP@50** | Higher | Enrichment of true edges in the top 50 predictions (definition as implemented in `metrics.py`). |
| **pseudo_hub_fdr_at_20** | Lower | Fraction of false positives among pseudo-hub decoys in the top 20 (stress metric). |
| **target_precision_at_20** | Higher | Precision on declared true targets within the top 20. |
| **signal_preservation_corr** | Higher | Correlation measuring how well true signal is preserved in scores (regime diagnostics). |
| **residual_confound_r2** | Lower | **Cheap proxy only:** linear R²-style summary of residual structure vs confound directions after scoring. It is **not** a direct measurement of true biological confounding; use it comparatively across regimes/methods, not as ground truth. |

"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_benchmark_roles_md(path: Path) -> None:
    text = """# Benchmark roles: realism-stress vs structural stress

Two synthetic benchmark **stories** live in this repository. They answer different questions.

## 1. Main regime benchmark (realism-stress)

- **Scripts:** `python -m benchmark.synthetic` then `python -m benchmark.run_benchmark`
- **What it is:** Regime-tagged `.npz` datasets (baseline, pseudo-hubs, noisy pseudotime, combined real-like, etc.) evaluated under shared metrics and baselines.
- **Intent:** Stress **deconfounding** and **failure modes** (PCA-easy regimes, pseudo-hubs, overlap) in a controlled grid—**not** a claim that the simulator is a faithful organoid or tissue model.
- **Outputs:** `tables/benchmark_results.csv`, `figures/main_benchmark_*.png`, regime heatmaps, `tables/benchmark_realism_summary.md`, etc.

## 2. Fig2 hard mode (structural misspecification stress)

- **Scripts:** `benchmark/fig2_rich_simulation.py`, `benchmark/fig2_benchmark_paper.py`
- **What it is:** Rich dynamics engineered so that **pooled global linear X→V** (e.g. Ridge-style summaries) is **structurally wrong** while local forward pairs remain informative for Jacobian-style methods.
- **Intent:** Method comparison under **deliberate misspecification** of a global linear coupling story—**not** packaged as biological realism.
- **Outputs:** Fig2 CSVs/figures under the Fig2 run directory (see module docstrings).

Use the regime benchmark to discuss **robustness under declared stressors**; use Fig2 hard mode to discuss **behavior when the global linear map is intentionally broken**.

"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_topology_model_diagnostics_summary(
    df: pd.DataFrame, diag_df: pd.DataFrame, tabdir: Path
) -> bool:
    metric_cols = [
        "AUPRC",
        "target_precision_at_20",
        "pseudo_hub_fdr_at_20",
        "residual_confound_r2",
        "signal_preservation_corr",
    ]
    use = [c for c in metric_cols if c in df.columns]
    if not use:
        return False
    merged = df.copy()
    if (
        len(diag_df)
        and "dataset" in diag_df.columns
        and "dataset" in merged.columns
        and "mean_ridge_r2_v_given_x" in diag_df.columns
    ):
        merged = merged.merge(
            diag_df[["dataset", "mean_ridge_r2_v_given_x"]].drop_duplicates(subset=["dataset"]),
            on="dataset",
            how="left",
        )
        use = use + ["mean_ridge_r2_v_given_x"]
    out = merged.groupby(["topology", "model"], as_index=False)[use].mean()
    _atomic_df_to_csv(out, tabdir / "topology_model_diagnostics_summary.csv")
    return True


def _write_benchmark_realism_summary_md(path: Path, df: pd.DataFrame, diagnostic_rows: list) -> None:
    lines = [
        "# Realism-stress synthetic benchmark — summary",
        "",
        "This run is the **regime-based realism-stress** benchmark (controlled failure modes / deconfounding), ",
        "not the Fig2 **structural X→V misspecification** track. See `tables/benchmark_roles.md`.",
        "",
        "Auto-generated from `tables/benchmark_results.csv` and `tables/scenario_diagnostics.csv`.",
        "",
        "## 1. Which regimes are still too easy / PCA-friendly?",
        "",
        "- Compare **mean AUPRC** across `scenario_regime` for `pair_pca_direct` vs `geobridge_dual`.",
        "- Regimes where **residual_confound_r2** is low and **signal_preservation_corr** stays high are often PCA-easy.",
        "",
        "## 2. Which realism-stress regimes mimic real-data failure (Corr competitive, weak top-K)?",
        "",
        "- Look for **high `pseudo_hub_fdr_at_20`** and **high `hub_trap_rate_at_20`** with **low `target_precision_at_20`**. ",
        "- **R6_combined_real_like** and **R3_pseudo_hubs** are intended stressors.",
        "",
        "## 3. Why does ridge dominate in ManyBifurcations?",
        "",
        "- Open `tables/scenario_diagnostics.csv`: **`mean_ridge_r2_v_given_x`** and **`ridge_manybifurcations_hypothesis`**. ",
        "- High **V ~ X** linear R² suggests global linear coupling that ridge exploits; sparse pseudo-hubs / low overlap in metadata support that story.",
        "",
        "## 4. Suggested paper figures",
        "",
        "- **Regime × method heatmaps** (`figures/heatmap_regime_method__*.png`).",
        "- **Classic topology bars** (`figures/main_benchmark_*.png`) on **baseline** subset only if mixing regimes.",
        "",
    ]
    if len(df) and "scenario_regime" in df.columns:
        sub = df[df["topology"] == "ManyBifurcations"]
        if len(sub):
            best = sub.groupby("model")["AUPRC"].mean().sort_values(ascending=False)
            lines.append("### ManyBifurcations — mean AUPRC by model (all regimes in this run)")
            lines.append("")
            lines.append(str(best.head(8)))
            lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

if __name__ == "__main__":
    main()
