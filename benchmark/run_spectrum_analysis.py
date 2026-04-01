import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from geobridge_model import GeoBridgeConfig, GeoBridgeModel
from geobridge_model.spectral_k import select_k_by_tail_exhaustion
from .pair_cli import add_pair_construction_args, pair_fields_from_args
from .plots import save_spectrum_plots

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--plots-outdir", default=None)
    p.add_argument("--max-k", type=int, default=80)
    p.add_argument("--tail-fraction", type=float, default=0.2)
    p.add_argument("--tail-min-start", type=int, default=20)
    p.add_argument("--slope-window", type=int, default=3)
    p.add_argument("--stable-m", type=int, default=3)
    p.add_argument("--tau", type=float, default=2.0)
    add_pair_construction_args(p)
    return p.parse_args()

def parse_stem(stem):
    topology, drift_type, rep = stem.split("__")
    return topology, drift_type, int(rep.replace("rep", ""))

def main():
    args = parse_args()
    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    plots_outdir = args.plots_outdir or str(outdir)
    rows = []
    for path in sorted(Path(args.data_dir).glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        topology, drift_type, rep = parse_stem(path.stem)
        default_cfg = GeoBridgeConfig(**pair_fields_from_args(args))
        pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
            data["X"],
            data["V"],
            data["pseudotime"],
            min_forward_dt=default_cfg.min_forward_dt,
            max_forward_dt=default_cfg.max_forward_dt,
            max_pairs_per_cell=default_cfg.max_pairs_per_cell,
            pair_seed=default_cfg.pair_seed,
            stream_num_threads=default_cfg.stream_num_threads,
        )
        S = pre["S"]
        spec = select_k_by_tail_exhaustion(S, args.max_k, args.tail_fraction, args.tail_min_start, args.slope_window, args.stable_m, args.tau)
        for comp, sval in enumerate(S, start=1):
            rows.append({"dataset": path.stem, "topology": topology, "drift_type": drift_type, "rep": rep, "component": comp, "singular_value": sval, "k_selected": spec.k_selected, "tail_baseline": spec.tail_baseline, "threshold": spec.threshold})
    df = pd.DataFrame(rows)
    df.to_csv(outdir / "pairspace_spectrum_metrics.csv", index=False)
    if plots_outdir:
        save_spectrum_plots(df, plots_outdir)
    print(f"Saved spectrum metrics to {outdir}")

if __name__ == "__main__":
    main()
