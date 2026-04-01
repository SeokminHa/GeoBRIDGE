"""Shared CLI helpers for pair construction (large-n runs)."""


def add_pair_construction_args(parser):
    parser.add_argument(
        "--max-pairs-per-cell",
        type=int,
        default=None,
        help="Cap random forward pairs per source cell (default: no cap). Strongly recommended for n_cells ~ 1200.",
    )
    parser.add_argument(
        "--max-forward-dt",
        type=float,
        default=None,
        help="Only pairs with pseudotime gap in (min_forward_dt, max_forward_dt]; default: no upper cap.",
    )
    parser.add_argument(
        "--pair-seed",
        type=int,
        default=None,
        help="Seed for per-cell pair subsampling (reproducible).",
    )
    parser.add_argument(
        "--stream-num-threads",
        type=int,
        default=None,
        dest="stream_num_threads",
        help="GeoBRIDGE streaming pair accumulation: None=auto parallel when safe; 1=single-threaded. "
        "Overrides env GEOBRIDGE_STREAM_THREADS.",
    )


def pair_construction_argv(args):
    """Build argv fragment for subprocess calls."""
    out = []
    if getattr(args, "max_pairs_per_cell", None) is not None:
        out.extend(["--max-pairs-per-cell", str(args.max_pairs_per_cell)])
    if getattr(args, "max_forward_dt", None) is not None:
        out.extend(["--max-forward-dt", str(args.max_forward_dt)])
    if getattr(args, "pair_seed", None) is not None:
        out.extend(["--pair-seed", str(args.pair_seed)])
    return out


def pair_window_and_seed_argv(args):
    """Like pair_construction_argv but omit --max-pairs-per-cell (e.g. pair-budget sweep sets its own)."""
    out = []
    if getattr(args, "max_forward_dt", None) is not None:
        out.extend(["--max-forward-dt", str(args.max_forward_dt)])
    if getattr(args, "pair_seed", None) is not None:
        out.extend(["--pair-seed", str(args.pair_seed)])
    return out


def pair_fields_from_args(args):
    return dict(
        max_forward_dt=getattr(args, "max_forward_dt", None),
        max_pairs_per_cell=getattr(args, "max_pairs_per_cell", None),
        pair_seed=getattr(args, "pair_seed", None),
        stream_num_threads=getattr(args, "stream_num_threads", None),
    )
