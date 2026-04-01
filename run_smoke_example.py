"""
End-to-end smoke test: all four topologies, modest scale (default 100 genes, 400 cells).

Usage (from project root):
    python run_smoke_example.py
    python run_smoke_example.py --n-cells 350 --n-genes 100 --reps 2
"""
import argparse
import subprocess
import sys
from pathlib import Path

PY = sys.executable
ROOT = Path(__file__).resolve().parent


def run(cmd):
    print("\n[RUN]", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=str(ROOT))


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--n-cells",
        type=int,
        default=400,
        help="More cells → more pairs; 400 is a reasonable balance with streaming stats.",
    )
    p.add_argument(
        "--n-genes",
        type=int,
        default=100,
        help="Must be >80 for a richer pair-space rank sweep; matches main benchmark default.",
    )
    p.add_argument("--reps", type=int, default=2)
    args = p.parse_args()

    data = "generated/smoke"
    topologies = ["LinearChain", "Bifurcation", "Trifurcation", "ManyBifurcations"]

    run(
        [
            PY,
            "-m",
            "benchmark.synthetic",
            "--outdir",
            data,
            "--topologies",
            *topologies,
            "--drift-types",
            "branch_specific",
            "--reps",
            str(args.reps),
            "--n-cells",
            str(args.n_cells),
            "--n-genes",
            str(args.n_genes),
        ]
    )

    run(
        [
            PY,
            "-m",
            "benchmark.run_benchmark",
            "--data-dir",
            data,
            "--outdir",
            "results/smoke_main",
        ]
    )
    run(
        [
            PY,
            "-m",
            "benchmark.run_k_sweep",
            "--data-dir",
            data,
            "--outdir",
            "results/smoke_k",
        ]
    )
    run(
        [
            PY,
            "-m",
            "benchmark.run_spectrum_analysis",
            "--data-dir",
            data,
            "--outdir",
            "results/smoke_spec",
        ]
    )
    run(
        [
            PY,
            "-m",
            "benchmark.run_pair_budget_sweep",
            "--data-dir",
            data,
            "--outdir",
            "results/smoke_pair_budget",
            "--pairs-per-cell",
            "5",
            "10",
            "20",
        ]
    )
    nc = int(args.n_cells)
    cell_scale_ns = sorted(
        {
            max(80, nc // 4),
            max(80, nc // 2),
            max(80, (3 * nc) // 4),
            nc,
        }
    )
    run(
        [
            PY,
            "-m",
            "benchmark.run_cell_scale_sweep",
            "--data-dir",
            data,
            "--outdir",
            "results/smoke_cell_scale",
            "--n-cells",
            *[str(x) for x in cell_scale_ns],
            "--max-pairs-per-cell",
            "40",
        ]
    )
    print("\nSmoke example done. Data:", data, "| results under results/smoke_*")


if __name__ == "__main__":
    main()
