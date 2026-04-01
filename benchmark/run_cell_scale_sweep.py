import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from geobridge_model import GeoBridgeConfig, GeoBridgeModel
from .metrics import evaluate_all_metrics
from .plots import save_cell_scale_plots


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--plots-outdir", default=None)
    p.add_argument("--n-cells", nargs="+", type=int, default=[100, 200, 400, 800, 1200])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--pair-seed",
        type=int,
        default=None,
        help="Seed for pair subsampling; default: same as --seed.",
    )
    p.add_argument("--max-forward-dt", type=float, default=None)
    p.add_argument("--max-pairs-per-cell", type=int, default=50)
    return p.parse_args()


def parse_stem(stem):
    topology, drift_type, rep = stem.split("__")
    return topology, drift_type, int(rep.replace("rep", ""))


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plots_outdir = args.plots_outdir or str(outdir)

    rows = []
    for path in sorted(Path(args.data_dir).glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        X, V, pseudotime, W = data["X"], data["V"], data["pseudotime"], data["W"]
        topology, drift_type, rep = parse_stem(path.stem)
        n = len(pseudotime)
        # Unique effective sizes after clamping to [20, n] — avoids re-running identical fits
        # when e.g. n=400 and requested 800,1200 all clamp to 400.
        effective_ms = sorted(
            {int(min(max(20, int(m)), n)) for m in args.n_cells}
        )

        for m in effective_ms:
            idx = np.sort(rng.choice(n, size=m, replace=False))
            pseed = args.pair_seed if args.pair_seed is not None else args.seed
            cfg = GeoBridgeConfig(
                project_y=True,
                max_forward_dt=args.max_forward_dt,
                max_pairs_per_cell=args.max_pairs_per_cell,
                pair_seed=pseed,
            )
            fit = GeoBridgeModel(cfg).fit(X[idx], V[idx], pseudotime[idx])
            rows.append(
                {
                    "dataset": path.stem,
                    "topology": topology,
                    "drift_type": drift_type,
                    "rep": rep,
                    "n_cells": m,
                    "k_selected": fit["k_selected"],
                    "max_pairs_per_cell": args.max_pairs_per_cell,
                    "max_forward_dt": args.max_forward_dt,
                    **evaluate_all_metrics(W, fit["scores"], topk=50),
                }
            )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / "cell_scale_sweep_metrics.csv", index=False)
    if plots_outdir:
        save_cell_scale_plots(df, plots_outdir)
    print(f"Saved cell-scale sweep to {outdir}")


if __name__ == "__main__":
    main()

