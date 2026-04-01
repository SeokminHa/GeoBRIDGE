import argparse, subprocess, sys
from pathlib import Path

from benchmark.pair_cli import pair_construction_argv, pair_window_and_seed_argv

sys.path.append(str(Path(__file__).parent))
PY = sys.executable

def run(cmd):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", type=int, default=30)
    p.add_argument("--drift-reps", type=int, default=10)
    p.add_argument("--n-cells", type=int, default=1200)
    p.add_argument("--n-genes", type=int, default=100)
    p.add_argument("--run-main", action="store_true")
    p.add_argument("--run-drift", action="store_true")
    p.add_argument("--run-k-sweep", action="store_true")
    p.add_argument("--run-cell-scale", action="store_true")
    p.add_argument("--run-pair-budget", action="store_true")
    p.add_argument("--run-spectrum", action="store_true")
    p.add_argument("--indirect-strength", type=float, default=0.35)
    p.add_argument("--indirect-hops", type=int, default=2)
    p.add_argument(
        "--max-pairs-per-cell",
        type=int,
        default=None,
        help="Forward to benchmark/k-sweep/spectrum/cell-scale (omit for full pair set; use ~50 for n_cells=1200).",
    )
    p.add_argument("--max-forward-dt", type=float, default=None)
    p.add_argument("--pair-seed", type=int, default=None)
    return p.parse_args()

def main():
    args = parse_args()
    run_any = any(
        [
            args.run_main,
            args.run_drift,
            args.run_k_sweep,
            args.run_cell_scale,
            args.run_pair_budget,
            args.run_spectrum,
        ]
    )
    # Default behavior: run only the main benchmark unless explicit flags are provided.
    if not run_any:
        args.run_main = True

    if args.run_main:
        run([PY, "-m", "benchmark.synthetic", "--outdir", "generated/main_benchmark",
             "--topologies", "LinearChain", "Bifurcation", "Trifurcation", "ManyBifurcations",
             "--drift-types", "branch_specific",
             "--reps", str(args.reps), "--n-genes", str(args.n_genes), "--n-cells", str(args.n_cells),
             "--indirect-strength", str(args.indirect_strength), "--indirect-hops", str(args.indirect_hops)])
        run(
            [PY, "-m", "benchmark.run_benchmark", "--data-dir", "generated/main_benchmark",
             "--outdir", "results/main_benchmark"]
            + pair_construction_argv(args)
        )
    if args.run_drift:
        run([PY, "-m", "benchmark.synthetic", "--outdir", "generated/drift_sweep",
             "--topologies", "LinearChain", "Bifurcation", "Trifurcation", "ManyBifurcations",
             "--drift-types", "none", "linear", "nonlinear", "branch_specific",
             "--reps", str(args.drift_reps), "--n-genes", str(args.n_genes), "--n-cells", str(args.n_cells),
             "--indirect-strength", str(args.indirect_strength), "--indirect-hops", str(args.indirect_hops)])
        run(
            [PY, "-m", "benchmark.run_benchmark", "--data-dir", "generated/drift_sweep",
             "--outdir", "results/drift_sweep"]
            + pair_construction_argv(args)
        )
    if args.run_k_sweep:
        run(
            [PY, "-m", "benchmark.run_k_sweep", "--data-dir", "generated/main_benchmark",
             "--outdir", "results/k_sweep"]
            + pair_construction_argv(args)
        )
    if args.run_cell_scale:
        run(
            [PY, "-m", "benchmark.run_cell_scale_sweep", "--data-dir", "generated/main_benchmark",
             "--outdir", "results/cell_scale_sweep"]
            + pair_construction_argv(args)
        )
    if args.run_pair_budget:
        run(
            [PY, "-m", "benchmark.run_pair_budget_sweep", "--data-dir", "generated/main_benchmark",
             "--outdir", "results/pair_budget_sweep"]
            + pair_window_and_seed_argv(args)
        )
    if args.run_spectrum:
        run(
            [PY, "-m", "benchmark.run_spectrum_analysis", "--data-dir", "generated/main_benchmark",
             "--outdir", "results/spectrum_analysis"]
            + pair_construction_argv(args)
        )
    print("\nALL DONE.")

if __name__ == "__main__":
    main()
