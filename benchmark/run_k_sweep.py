import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from geobridge_model import GeoBridgeConfig, GeoBridgeModel
from geobridge_model.spectral_k import select_k_by_tail_exhaustion
from .metrics import evaluate_all_metrics
from .pair_cli import add_pair_construction_args, pair_fields_from_args
from .plots import save_k_sweep_plots

# k=0..30 step 1, then tail grid. (Do not rely on argparse default=list(...) + nargs="+": it can
# fail to apply the full default on some invocations; we resolve in main() instead.)
_DEFAULT_K_VALUES = tuple(sorted({*range(0, 31), 35, 40, 45, 50, 70}))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--plots-outdir", default=None)
    p.add_argument(
        "--k-values",
        nargs="*",
        type=int,
        default=None,
        help="If omitted or empty, uses built-in dense grid (0..30 step 1, then 35,40,45,50,70).",
    )
    p.add_argument("--project-y-modes", nargs="+", default=["xonly","dual"])
    add_pair_construction_args(p)
    return p.parse_args()


def resolve_k_values(args):
    """Always return a sorted unique list of k values."""
    raw = args.k_values
    if raw is None or len(raw) == 0:
        return list(_DEFAULT_K_VALUES)
    out = sorted(set(int(k) for k in raw))
    return out if out else list(_DEFAULT_K_VALUES)

def parse_stem(stem):
    topology, drift_type, rep = stem.split("__")
    return topology, drift_type, int(rep.replace("rep", ""))

def main():
    args = parse_args()
    k_list = resolve_k_values(args)
    print(
        f"k-sweep: using {len(k_list)} k values "
        f"(min={k_list[0]}, max={k_list[-1]}; has 6–9: "
        f"{all(x in k_list for x in (6, 7, 8, 9))})"
    )
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    plots_outdir = args.plots_outdir or str(outdir)
    rows = []
    for path in sorted(Path(args.data_dir).glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        X, V, pseudotime, W = data["X"], data["V"], data["pseudotime"], data["W"]
        topology, drift_type, rep = parse_stem(path.stem)
        k_max = int(max(k_list)) if k_list else 0
        # Streaming stats + eigh(C): same math as SVD(ΔXc), no materialized pair matrix.
        default_cfg = GeoBridgeConfig(**pair_fields_from_args(args))
        pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
            X,
            V,
            pseudotime,
            min_forward_dt=default_cfg.min_forward_dt,
            max_forward_dt=default_cfg.max_forward_dt,
            max_pairs_per_cell=default_cfg.max_pairs_per_cell,
            pair_seed=default_cfg.pair_seed,
            stream_num_threads=default_cfg.stream_num_threads,
        )
        spectral = select_k_by_tail_exhaustion(
            pre["S"],
            max(k_max, 1),
            default_cfg.tail_fraction,
            default_cfg.tail_min_start,
            default_cfg.slope_window,
            default_cfg.stable_m,
            default_cfg.tau,
        )
        auto_k = int(spectral.k_selected)
        # Rank of pair-space covariance C = ΔXc^T ΔXc; scores stabilize for k beyond this (see doc below).
        C = pre["C"]
        rank_c = int(np.linalg.matrix_rank(C, hermitian=True))
        s = pre["S"]
        n_s_pos = int(np.sum(s > 1e-12 * (float(s[0]) if s.size else 1.0)))
        for mode in args.project_y_modes:
            for k in k_list:
                scores = GeoBridgeModel.scores_from_streaming_stats(
                    pre,
                    k=k,
                    project_y=(mode == "dual"),
                    normalize_score=True,
                    eps=default_cfg.eps,
                )
                rows.append({
                    "dataset": path.stem,
                    "topology": topology,
                    "drift_type": drift_type,
                    "rep": rep,
                    "project_y_mode": mode,
                    "k": int(k),
                    "auto_k": auto_k,
                    "pairspace_rank_C": rank_c,
                    "n_singular_vals_gt_rtol": n_s_pos,
                    **evaluate_all_metrics(W, scores, topk=50),
                })
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "k_sweep_metrics.csv", index=False)
    if plots_outdir:
        save_k_sweep_plots(df, plots_outdir)
    print(f"Saved k-sweep to {outdir}")

if __name__ == "__main__":
    main()
