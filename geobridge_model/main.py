import argparse
import numpy as np
from .config import GeoBridgeConfig
from .model import GeoBridgeModel

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--max-k", type=int, default=40)
    p.add_argument("--tail-fraction", type=float, default=0.2)
    p.add_argument("--tail-min-start", type=int, default=20)
    p.add_argument("--slope-window", type=int, default=3)
    p.add_argument("--stable-m", type=int, default=3)
    p.add_argument("--tau", type=float, default=2.0)
    p.add_argument("--no-project-y", action="store_true")
    p.add_argument("--unnormalized", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    data = np.load(args.input, allow_pickle=True)
    cfg = GeoBridgeConfig(
        max_k=args.max_k,
        tail_fraction=args.tail_fraction,
        tail_min_start=args.tail_min_start,
        slope_window=args.slope_window,
        stable_m=args.stable_m,
        tau=args.tau,
        project_y=not args.no_project_y,
        normalize_score=not args.unnormalized,
    )
    model = GeoBridgeModel(cfg)
    result = model.fit(data["X"], data["V"], data["pseudotime"])
    model.save(args.outdir)
    print(f"Saved to {args.outdir}")
    print(f"Selected k = {result['k_selected']}")

if __name__ == "__main__":
    main()
