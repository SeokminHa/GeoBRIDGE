from dataclasses import dataclass
import numpy as np

@dataclass
class SpectralResult:
    k_selected: int
    singular_values: np.ndarray
    log_spectrum: np.ndarray
    slope: np.ndarray
    slope_smoothed: np.ndarray
    tail_baseline: float
    threshold: float
    tail_start: int

def moving_average(x, w):
    if w <= 1:
        return x.copy()
    out = np.zeros_like(x, dtype=float)
    for i in range(len(x)):
        lo = max(0, i - w + 1)
        out[i] = np.mean(x[lo:i+1])
    return out

def select_k_by_tail_exhaustion(
    singular_values,
    max_k=40,
    tail_fraction=0.2,
    tail_min_start=20,
    slope_window=3,
    stable_m=3,
    tau=2.0,
):
    s = np.asarray(singular_values, dtype=float)
    s = s[s > 0]
    if s.size < 3:
        k = max(1, int(s.size))
        z = np.zeros(max(0, s.size - 1), dtype=float)
        return SpectralResult(k, s, np.log(s + 1e-12), z, z, 0.0, 0.0, 0)
    k_cap = min(max_k, s.size - 1)
    log_s = np.log(s[:k_cap + 1])
    slope = np.abs(np.diff(log_s))
    slope_smoothed = moving_average(slope, slope_window)
    tail_start = max(min(len(slope_smoothed)-1, tail_min_start), int(len(slope_smoothed) * (1 - tail_fraction)))
    tail = slope_smoothed[tail_start:]
    tail_baseline = float(np.median(tail)) if tail.size else float(np.median(slope_smoothed))
    threshold = float(tau * tail_baseline)
    k_selected = k_cap
    for i in range(len(slope_smoothed) - stable_m + 1):
        if np.all(slope_smoothed[i:i+stable_m] <= threshold):
            k_selected = i + 1
            break
    k_selected = max(1, min(k_selected, k_cap))
    return SpectralResult(k_selected, s, log_s, slope, slope_smoothed, tail_baseline, threshold, tail_start)
