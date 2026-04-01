import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from geobridge_model import GeoBridgeConfig, GeoBridgeModel
from geobridge_model.spectral_k import select_k_by_tail_exhaustion
from .baselines import method_pairwise_raw, method_corr, method_ridge, method_cell_pca_preresidualize, method_pair_pca_direct
from .metrics import evaluate_all_metrics
from .pair_cli import add_pair_construction_args, pair_fields_from_args
from .plots import save_main_benchmark_plots

MODEL_NAMES = ["geobridge_dual","geobridge_xonly","pairwise_raw","corr","ridge","cell_pca_preresidualize","pair_pca_direct"]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--plots-outdir", default=None)
    p.add_argument("--models", nargs="+", default=MODEL_NAMES)
    p.add_argument("--max-k", type=int, default=40)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--stable-m", type=int, default=3)
    p.add_argument("--tail-fraction", type=float, default=0.2)
    p.add_argument("--tail-min-start", type=int, default=20)
    p.add_argument("--slope-window", type=int, default=3)
    p.add_argument("--baseline-k", type=int, default=3)
    add_pair_construction_args(p)
    return p.parse_args()

def parse_stem(stem):
    topology, drift_type, rep = stem.split("__")
    return topology, drift_type, int(rep.replace("rep", ""))

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

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    plots_outdir = args.plots_outdir or str(outdir)
    rows = []
    for path in sorted(Path(args.data_dir).glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        X, V, pseudotime, W = data["X"], data["V"], data["pseudotime"], data["W"]
        topology, drift_type, rep = parse_stem(path.stem)
        pre_stats = None
        if any(m in {"geobridge_dual","geobridge_xonly"} for m in args.models):
            # Compute once per dataset and reuse across both geobridge variants.
            default_cfg = GeoBridgeConfig(**pair_fields_from_args(args))
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
        for name in args.models:
            scores, k_sel = run_model(name, X, V, pseudotime, args, pre_stats=pre_stats)
            metrics = evaluate_all_metrics(W, scores, topk=50)
            rows.append({"dataset": path.stem, "topology": topology, "drift_type": drift_type, "rep": rep, "model": name, "k_selected": k_sel, **metrics})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "benchmark_results.csv", index=False)
    summary = df.groupby(["topology","drift_type","model"], as_index=False)[["S-AUPRC","AUPRC","EDP@50","AUROC"]].agg(["mean","std"])
    summary.columns = ["_".join(c).strip("_") for c in summary.columns.to_flat_index()]
    summary.to_csv(outdir / "benchmark_summary.csv", index=False)
    if plots_outdir:
        save_main_benchmark_plots(df, plots_outdir)
    print(f"Saved benchmark results to {outdir}")

if __name__ == "__main__":
    main()
