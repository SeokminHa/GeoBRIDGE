import argparse
import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd

from geobridge_model import GeoBridgeConfig, GeoBridgeModel
from geobridge_model.spectral_k import select_k_by_tail_exhaustion

from .metrics import evaluate_all_metrics
from .plots import save_pair_budget_sweep_plots


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--plots-outdir", default=None)
    p.add_argument("--pairs-per-cell", nargs="+", type=int, default=[5, 10, 20, 50, 100])
    p.add_argument("--max-forward-dt", type=float, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--pair-seed",
        type=int,
        default=None,
        help="Base seed mixed into per-dataset pair subsampling; default: --seed.",
    )
    p.add_argument("--models", nargs="+", default=["geobridge_dual", "geobridge_xonly"])
    p.add_argument("--max-k", type=int, default=40)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--stable-m", type=int, default=3)
    p.add_argument("--tail-fraction", type=float, default=0.2)
    p.add_argument("--tail-min-start", type=int, default=20)
    p.add_argument("--slope-window", type=int, default=3)
    return p.parse_args()


def _stable_int(*parts):
    s = "||".join(map(str, parts)).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:16], 16) % (2**32)


def parse_stem(stem):
    topology, drift_type, rep = stem.split("__")
    return topology, drift_type, int(rep.replace("rep", ""))


def run_geobridge_from_stats(pre_stats, cfg):
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


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plots_outdir = args.plots_outdir or str(outdir)

    rows = []
    for path in sorted(Path(args.data_dir).glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        X, V, pseudotime, W = data["X"], data["V"], data["pseudotime"], data["W"]
        topology, drift_type, rep = parse_stem(path.stem)

        seed0 = int(args.pair_seed) if args.pair_seed is not None else int(args.seed)
        for ppc in args.pairs_per_cell:
            base_cfg = GeoBridgeConfig(
                max_k=args.max_k,
                tau=args.tau,
                stable_m=args.stable_m,
                tail_fraction=args.tail_fraction,
                tail_min_start=args.tail_min_start,
                slope_window=args.slope_window,
                normalize_score=True,
                max_forward_dt=args.max_forward_dt,
                max_pairs_per_cell=int(ppc),
                pair_seed=int(_stable_int(seed0, path.stem, int(ppc))),
            )
            pre_stats = GeoBridgeModel.precompute_pairspace_stats_streaming(
                X,
                V,
                pseudotime,
                min_forward_dt=base_cfg.min_forward_dt,
                max_forward_dt=base_cfg.max_forward_dt,
                max_pairs_per_cell=base_cfg.max_pairs_per_cell,
                pair_seed=base_cfg.pair_seed,
            )

            for model_name in args.models:
                cfg = GeoBridgeConfig(**{**base_cfg.__dict__})
                cfg.project_y = (model_name == "geobridge_dual")

                t0 = time.perf_counter()
                scores, k_sel = run_geobridge_from_stats(pre_stats, cfg)
                metrics = evaluate_all_metrics(W, scores, topk=50)
                runtime_s = time.perf_counter() - t0

                for metric_name in ["AUPRC", "S-AUPRC", "AUROC", "EDP@50"]:
                    rows.append(
                        {
                            "method": model_name,
                            "pairs_per_cell": int(ppc),
                            "metric_name": metric_name,
                            "value": float(metrics[metric_name]),
                            "runtime": float(runtime_s),
                            "dataset": path.stem,
                            "topology": topology,
                            "drift_type": drift_type,
                            "rep": rep,
                            "k_selected": int(k_sel),
                            "n_pairs": int(pre_stats["n_pairs"]),
                        }
                    )

                rows.append(
                    {
                        "method": model_name,
                        "pairs_per_cell": int(ppc),
                        "metric_name": "runtime",
                        "value": float(runtime_s),
                        "runtime": float(runtime_s),
                        "dataset": path.stem,
                        "topology": topology,
                        "drift_type": drift_type,
                        "rep": rep,
                        "k_selected": int(k_sel),
                        "n_pairs": int(pre_stats["n_pairs"]),
                    }
                )

    df = pd.DataFrame(rows)
    # Spec wants: method, pairs_per_cell, metric_name, value, runtime (keep extra columns too)
    df.to_csv(outdir / "pair_budget_sweep_metrics.csv", index=False)
    if plots_outdir:
        save_pair_budget_sweep_plots(df, plots_outdir)
    print(f"Saved pair-budget sweep to {outdir}")


if __name__ == "__main__":
    main()

