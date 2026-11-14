from dataclasses import dataclass, field
import numpy as np

# Diagnostics from band-based k selection
@dataclass
class BandResult:
    chosen_k: int
    k_low: int
    k_high: int
    stability_curve: list = field(default_factory=list)
    mean_score_at_k: float = 0.0

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


# Elbow in log-spectrum via max distance from chord
def _spectral_elbow_k(S, cap=25):
    s = S[:min(cap + 1, len(S))]
    if len(s) < 4:
        return max(1, len(s) - 1)
    log_s = np.log(s + 1e-12)
    n = len(log_s)
    x = np.arange(n, dtype=float) / max(n - 1, 1)
    y = (log_s - log_s[-1]) / max(log_s[0] - log_s[-1], 1e-12)
    a = y[-1] - y[0]
    b = -(x[-1] - x[0])
    c = x[-1] * y[0] - x[0] * y[-1]
    denom = np.sqrt(a**2 + b**2) + 1e-12
    dist = np.abs(a * x + b * y + c) / denom
    elbow = int(np.argmax(dist[2:]) + 2) + 1
    return elbow


# Band-based k selection for pair-space projection
def select_k_band_hybrid(
    singular_values,
    DXc,
    DV,
    U=None,
    Vt=None,
    floor=8,
    cap=20,
    stability_threshold=0.90,
    stability_run=3,
):
    from geobridge_model.scoring import dual_project_pairspace, score_matrix

    S = np.asarray(singular_values, dtype=float)
    if U is None or Vt is None:
        U, S, Vt = np.linalg.svd(DXc, full_matrices=False)

    G = DXc.shape[1]
    max_k = min(cap + 5, len(S) - 1)

    # Spectral elbow lower bound
    elbow_k = _spectral_elbow_k(S, cap=max_k)

    # Stability plateau upper bound
    topn = min(100, G * (G - 1))
    prev_order = None
    overlaps = []
    stable_count = 0
    plateau_k = max_k

    for k in range(1, max_k + 1):
        DXr, DVr = dual_project_pairspace(DXc, DV, U, S, Vt, k)
        sc = score_matrix(DXr, DVr, normalize=True)
        flat = np.abs(sc[~np.eye(G, dtype=bool)])
        order = np.argsort(-flat)[:topn]

        if prev_order is not None:
            overlap = len(set(prev_order.tolist()) & set(order.tolist())) / topn
            overlaps.append(overlap)
            if overlap >= stability_threshold:
                stable_count += 1
            else:
                stable_count = 0
            if stable_count >= stability_run and plateau_k == max_k:
                plateau_k = k - stability_run + 1
        else:
            overlaps.append(float('nan'))
        prev_order = order

    # Tail-exhaustion upper bound
    spec = select_k_by_tail_exhaustion(S)
    tail_k = spec.k_selected

    # Form the band
    k_low = max(floor, elbow_k)
    k_high = min(cap, plateau_k, tail_k + 3)

    if k_low > k_high:
        k_low = max(floor, min(elbow_k, plateau_k, tail_k + 3))
        k_high = min(cap, max(elbow_k, plateau_k))

    # Pick the two-thirds point within the band
    band_width = k_high - k_low
    chosen_k = k_low + int(round(band_width * 0.67))
    chosen_k = max(k_low, min(chosen_k, k_high, len(S) - 1))

    # Compute mean |score| at chosen k for diagnostics
    DXr, DVr = dual_project_pairspace(DXc, DV, U, S, Vt, chosen_k)
    sc = score_matrix(DXr, DVr, normalize=True)
    ms = float(np.mean(np.abs(sc[~np.eye(G, dtype=bool)])))

    return BandResult(
        chosen_k=chosen_k,
        k_low=k_low,
        k_high=k_high,
        stability_curve=overlaps,
        mean_score_at_k=ms,
    )
