from dataclasses import dataclass

@dataclass
class GeoBridgeConfig:
    min_forward_dt: float = 1e-4
    max_forward_dt: float | None = None
    max_pairs_per_cell: int | None = None
    pair_seed: int | None = None
    # Streaming pair accumulation: None / unset = auto parallel when safe; 1 = single-threaded
    stream_num_threads: int | None = None
    max_k: int = 40
    tail_fraction: float = 0.2
    tail_min_start: int = 20
    slope_window: int = 3
    stable_m: int = 3
    tau: float = 2.0
    project_y: bool = True
    normalize_score: bool = True
    eps: float = 1e-12
