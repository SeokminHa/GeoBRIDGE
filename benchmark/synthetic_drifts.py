"""
Extended drift modes for benchmark/synthetic.py without modifying the original simulator.

Built-in modes (none, linear, nonlinear, branch_specific) are delegated to
``benchmark.synthetic.simulate_from_base`` unchanged.

New modes live here only; see ``EXTENDED_DRIFT_TYPES``.
"""
from __future__ import annotations

import numpy as np

from . import synthetic as syn
from .synthetic import SIM_STATE_CLIP, SIM_VELOCITY_CLIP

EXTENDED_DRIFT_TYPES = ("nonlinear_branch",)
ALL_DRIFT_TYPES = tuple(syn.DRIFT_TYPES) + EXTENDED_DRIFT_TYPES


def drift_nonlinear_branch(
    drift_amp: float,
    alpha: float,
    global_vec: np.ndarray,
    branch_name: str,
    branch_vecs: dict,
) -> np.ndarray:
    """
    Nonlinear-in-α drift with a branch-specific direction (harder for global linear X→V baselines).

    α ∈ [0, 1] along each segment. Combines a smooth nonlinear time profile with a direction
    tied to ``global_vec`` and ``branch_vecs[branch_name]``.
    """
    phase = float(alpha) ** 2 + 0.5 * np.sin(np.pi * alpha)
    direction = 0.55 * global_vec + 0.85 * branch_vecs[branch_name]
    direction = direction / (np.linalg.norm(direction) + 1e-12)
    return drift_amp * phase * direction


def _simulate_with_custom_drift(base, drift_vec_fn):
    """
    Same trajectory construction as synthetic.simulate_from_base, but drift at each step comes from
    drift_vec_fn(alpha, branch_name) -> (G,) array.
    """
    W = base["W"]
    segments = base["segments"]
    branch_offsets = base["branch_offsets"]
    branch_vecs = base["branch_vecs"]
    dt = base["dt"]
    decay = base["decay"]
    indirect_strength = base["indirect_strength"]
    indirect_ops = base["indirect_ops"]

    X_parts, V_parts, D_parts, T_parts, P_parts = [], [], [], [], []
    seg_data = {}
    for name, parent, duration in segments:
        steps = int(round(duration / dt)) + 1
        alpha = np.linspace(0.0, 1.0, steps)
        x = np.zeros((steps, W.shape[0]), dtype=float)
        x[0] = base["init_by_seg"][name].copy() if parent is None else seg_data[parent]["x"][-1] + branch_offsets[name]
        v = np.zeros_like(x)
        d = np.zeros_like(x)
        state_noise = base["state_noise_by_seg"][name]
        vel_noise = base["vel_noise_by_seg"][name]
        branch_bias = branch_offsets[name]
        for t in range(steps - 1):
            drift = drift_vec_fn(alpha[t], name)
            d[t] = drift
            direct_reg = x[t] @ W
            indirect_reg = np.zeros_like(direct_reg)
            for op in indirect_ops:
                indirect_reg += x[t] @ op
            reg_for_state = direct_reg - decay * x[t]
            x[t + 1] = x[t] + dt * (reg_for_state + drift) + branch_bias + state_noise[t]
            np.clip(x[t + 1], -SIM_STATE_CLIP, SIM_STATE_CLIP, out=x[t + 1])
            v[t] = direct_reg + indirect_strength * indirect_reg + drift - decay * x[t] + vel_noise[t]
            np.clip(v[t], -SIM_VELOCITY_CLIP, SIM_VELOCITY_CLIP, out=v[t])
        drift_last = drift_vec_fn(alpha[-1], name)
        d[-1] = drift_last
        direct_last = x[-1] @ W
        indirect_last = np.zeros_like(direct_last)
        for op in indirect_ops:
            indirect_last += x[-1] @ op
        v[-1] = direct_last + indirect_strength * indirect_last + drift_last - decay * x[-1] + vel_noise[-1]
        np.clip(x[-1], -SIM_STATE_CLIP, SIM_STATE_CLIP, out=x[-1])
        np.clip(v[-1], -SIM_VELOCITY_CLIP, SIM_VELOCITY_CLIP, out=v[-1])

        seg_data[name] = {"x": x}
        X_parts.append(x)
        V_parts.append(v)
        D_parts.append(d)
        T_parts.append(alpha)
        P_parts.append(np.array([name] * steps, dtype=object))

    X_full, V_full, D_full = np.vstack(X_parts), np.vstack(V_parts), np.vstack(D_parts)
    T_full, P_full = np.concatenate(T_parts), np.concatenate(P_parts)
    idx = base["sample_indices"]
    X, V, D, time, path = X_full[idx], V_full[idx], D_full[idx], T_full[idx], P_full[idx]
    order = np.argsort(time)
    return {
        "X": X[order],
        "V": V[order],
        "D": D[order],
        "time": time[order],
        "path": path[order],
        "pseudotime": time[order],
        "W": W,
    }


def simulate_from_base_extended(base, drift_type: str, drift_amp: float):
    """
    Run simulation: built-in ``drift_type`` values delegate to ``synthetic.simulate_from_base``;
    extended names use implementations in this module.
    """
    if drift_type in syn.DRIFT_TYPES:
        return syn.simulate_from_base(base, drift_type, drift_amp)
    if drift_type == "nonlinear_branch":
        g = base["global_vec"]
        bv = base["branch_vecs"]

        def drift_vec_fn(alpha, branch_name):
            return drift_nonlinear_branch(drift_amp, alpha, g, branch_name, bv)

        return _simulate_with_custom_drift(base, drift_vec_fn)
    raise ValueError(
        f"Unknown drift_type {drift_type!r}. "
        f"Built-in: {syn.DRIFT_TYPES}; extended: {EXTENDED_DRIFT_TYPES}"
    )
