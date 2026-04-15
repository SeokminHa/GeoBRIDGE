"""
Modular Fig2-only simulation extensions (latent drift, X/V mismatch, state-dependent edges,
gene-scale heterogeneity, library-size factors). Each feature is optional and CLI-ablatable.

**Design intent (hard preset):** this stack supports a **structural stress test** for methods that
assume a **pooled global linear X→V** map—not a claim of biological realism. Pooled linear **X→V**
is intentionally **misspecified** when (i) effective **W** varies with pseudotime/branch, (ii) **V**
sees a different drift path than **X**, and (iii) a shared latent **U** couples to **X** and **V**
through **different** loadings with **larger** amplitude in **V**. Local forward pairs with **small**
Δ(pseudotime) then see an approximately **constant** regime mask and similar **U**, so pairwise
**ΔX, ΔV** can still align with local direct structure.

Evaluation uses the full structural **W** (union of direct edges). Dynamics use **masked** ``Wm`` for
the direct term only; see ``edge_regime`` / ``branch_id`` for metadata (-1 = globally active).
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib

import numpy as np

from . import synthetic as syn
from .synthetic import SIM_STATE_CLIP, SIM_VELOCITY_CLIP, _drift_and_dev
from .synthetic_drifts import drift_nonlinear_branch


def _stable_float(seed: int, key: str) -> float:
    h = hashlib.sha256(f"{seed}||{key}".encode()).digest()
    return int.from_bytes(h[:8], "little") / (2**64)


@dataclass
class Fig2RealismKnobs:
    latent_drift: bool = False
    latent_x_scale: float = 0.0
    latent_v_scale: float = 0.0
    latent_phase_shift: float = 0.0
    latent_drift_rank: int = 2
    xv_drift_mismatch: bool = False
    xv_mismatch_strength: float = 0.0
    xv_phase_shift: float = 0.0
    state_dependent_edges: bool = False
    state_edge_frac: float = 0.3
    branch_specific_frac: float = 0.4
    early_late_frac: float = 0.3
    early_threshold: float = 0.35
    late_threshold: float = 0.65
    gene_scale_heterogeneity: bool = False
    gene_scale_sigma: float = 0.25
    gene_scale_apply_to_v: bool = True
    lib_size_noise: bool = False
    lib_size_sigma: float = 0.15
    lib_size_apply_to_v: bool = False
    # Nonlinear regulation: applied to direct_reg = f(x[t]) @ W instead of x[t] @ W
    # "linear" (default), "saturation", "threshold", "mild_nonlinear"
    regulation_nonlinearity: str = "linear"
    regulation_K: float = 5.0       # saturation half-max (Michaelis-Menten K)
    regulation_threshold: float = 0.3  # threshold center for switch-like
    regulation_steepness: float = 2.0  # Hill coefficient for threshold

    def any_enabled(self) -> bool:
        return bool(
            self.latent_drift
            or self.xv_drift_mismatch
            or self.state_dependent_edges
            or self.gene_scale_heterogeneity
            or self.lib_size_noise
            or self.regulation_nonlinearity != "linear"
        )


def _apply_regulation_nonlinearity(x: np.ndarray, knobs: Fig2RealismKnobs) -> np.ndarray:
    """Apply nonlinear transform to state vector before regulation matrix multiply.

    Given state x, returns f(x) such that direct_reg = f(x) @ W.

    Modes:
      - "linear": f(x) = x  (identity, default)
      - "saturation": f(x) = x * K / (K + |x|)  (Michaelis-Menten-like)
        Linear near zero (f'(0)=1), saturates for |x|>>K.
        At |x|=K, effect is x/2.  Preserves sign.
      - "threshold": f(x) = x * sigmoid(steepness * (|x| - threshold))
        Switch-like: weak below threshold, full above. Preserves sign.
      - "mild_nonlinear": f(x) = sign(x) * |x|^0.85
        Sublinear power law. Compresses large values mildly.
    """
    mode = knobs.regulation_nonlinearity
    if mode == "linear":
        return x
    ax = np.abs(x)
    if mode == "saturation":
        K = knobs.regulation_K
        scale = K / (K + ax)  # 1 at x=0, approaches 0 for |x|>>K => f(x)~K*sign(x)
        return x * scale
    elif mode == "threshold":
        thr = knobs.regulation_threshold
        steep = knobs.regulation_steepness
        gate = 1.0 / (1.0 + np.exp(-steep * (ax - thr)))
        return x * gate
    elif mode == "mild_nonlinear":
        return np.sign(x) * np.power(ax + 1e-12, 0.85)
    else:
        raise ValueError(f"Unknown regulation nonlinearity: {mode}")


def _drift_vector(
    drift_type: str,
    drift_amp: float,
    alpha: float,
    branch_name: str,
    base: dict,
) -> np.ndarray:
    if drift_type in syn.DRIFT_TYPES:
        return _drift_and_dev(
            drift_type,
            drift_amp,
            alpha,
            base["global_vec"],
            base["nonlinear_vec"],
            branch_name,
            base["branch_vecs"],
        )
    if drift_type == "nonlinear_branch":
        return drift_nonlinear_branch(
            drift_amp,
            alpha,
            base["global_vec"],
            branch_name,
            base["branch_vecs"],
        )
    raise ValueError(drift_type)


def _drift_vector_v(
    drift_type: str,
    drift_amp: float,
    alpha: float,
    branch_name: str,
    base: dict,
    knobs: Fig2RealismKnobs,
) -> np.ndarray:
    dx = _drift_vector(drift_type, drift_amp, alpha, branch_name, base)
    if not knobs.xv_drift_mismatch or knobs.xv_mismatch_strength <= 0:
        return dx
    # Original implementation phase-shifted alpha, but for linear drift types
    # (branch_specific: drift = amp * alpha * constant_direction), this only
    # changes magnitude, not direction. Phase-shifted drift is perfectly
    # collinear with original drift, so blending has zero directional effect.
    #
    # Fix: add a stable orthogonal mismatch direction that creates genuine
    # directional divergence between X and V drift.  The mismatch direction
    # is seeded from the branch name + xv_phase_shift for reproducibility.
    m = float(np.clip(knobs.xv_mismatch_strength, 0.0, 1.0))
    G = len(dx)
    # Generate stable mismatch direction from branch name
    seed_str = f"xv_mismatch|{branch_name}|{knobs.xv_phase_shift}"
    seed_int = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed_int)
    raw_dir = rng.standard_normal(G)
    # Orthogonalize against dx to get a pure directional mismatch
    dx_norm = np.linalg.norm(dx)
    if dx_norm > 1e-12:
        proj = np.dot(raw_dir, dx) / (dx_norm ** 2) * dx
        orth = raw_dir - proj
    else:
        orth = raw_dir
    orth_norm = np.linalg.norm(orth)
    if orth_norm > 1e-12:
        orth = orth / orth_norm * dx_norm  # scale to same magnitude as dx
    # Blend: at m=0 pure dx, at m=0.5 equal mix of dx and orthogonal
    return (1.0 - m) * dx + m * orth


def _latent_U(alpha: float, branch_name: str, rank: int, phase0: float, trial_seed: int) -> np.ndarray:
    """Low-frequency sinusoidal latent; phase depends on branch (stable across runs for fixed seed)."""
    u = np.zeros(rank, dtype=float)
    br = int(hashlib.sha256(branch_name.encode()).hexdigest()[:8], 16) % 10_007
    ph = phase0 + 2 * np.pi * (br / 10_007.0)
    for k in range(rank):
        u[k] = np.sin(2.0 * np.pi * (k + 1) * alpha + ph + 0.15 * (k + 1) * _stable_float(trial_seed, f"uk{k}"))
    return u


def _assign_edge_regimes(
    W: np.ndarray,
    segment_names: list[str],
    rng: np.random.Generator,
    knobs: Fig2RealismKnobs,
) -> tuple[np.ndarray, np.ndarray]:
    """
    edge_regime[i,j]: -1 = global (always on), 0 = branch-tied, 1 = early-only, 2 = late-only.
    branch_id[i,j]: segment name index for type 0, else -1.
    """
    G = W.shape[0]
    edge_regime = np.full((G, G), -1, dtype=np.int8)
    branch_id = np.full((G, G), -1, dtype=np.int16)
    nz = list(zip(*np.nonzero(W != 0)))
    rng.shuffle(nz)
    n_sd = int(round(len(nz) * knobs.state_edge_frac))
    n_sd = max(0, min(n_sd, len(nz)))
    name_to_idx = {n: i for i, n in enumerate(segment_names)}
    wb = max(0.0, float(knobs.branch_specific_frac))
    we = max(0.0, float(knobs.early_late_frac))
    w = np.array([wb, 0.5 * we, 0.5 * we], dtype=float)
    if w.sum() < 1e-12:
        w = np.array([1.0, 0.0, 0.0])
    else:
        w /= w.sum()
    for k in range(n_sd):
        i, j = nz[k]
        cat = int(rng.choice(3, p=w))
        if cat == 0:
            edge_regime[i, j] = 0
            branch_id[i, j] = name_to_idx[rng.choice(segment_names)]
        elif cat == 1:
            edge_regime[i, j] = 1
        else:
            edge_regime[i, j] = 2
    return edge_regime, branch_id


def _regime_mask(
    edge_regime: np.ndarray,
    branch_id: np.ndarray,
    alpha: float,
    seg_name: str,
    knobs: Fig2RealismKnobs,
    name_to_idx: dict[str, int],
) -> np.ndarray:
    """Elementwise mask in [0,1] for W (same shape as W)."""
    G = edge_regime.shape[0]
    M = np.ones((G, G), dtype=float)
    si = name_to_idx.get(seg_name, -1)
    for i in range(G):
        for j in range(G):
            er = edge_regime[i, j]
            if er < 0:
                continue
            if er == 0:
                M[i, j] = 1.0 if branch_id[i, j] == si else 0.0
            elif er == 1:
                M[i, j] = 1.0 if alpha < knobs.early_threshold else 0.0
            else:
                M[i, j] = 1.0 if alpha > knobs.late_threshold else 0.0
    return M


def simulate_fig2_rich_trajectory(
    base: dict,
    drift_type: str,
    drift_amp: float,
    knobs: Fig2RealismKnobs,
    trial_seed: int,
) -> dict:
    """
    Same return keys as ``synthetic.simulate_from_base`` plus optional ``edge_regime``, ``branch_id``.
    """
    W = base["W"].copy()
    G = W.shape[0]
    dt = base["dt"]
    decay = base["decay"]
    indirect_strength = base["indirect_strength"]
    indirect_ops = base["indirect_ops"]
    segments = base["segments"]
    rng = np.random.default_rng(trial_seed)

    gene_scale = np.ones(G, dtype=float)
    if knobs.gene_scale_heterogeneity and knobs.gene_scale_sigma > 0:
        gene_scale = np.exp(knobs.gene_scale_sigma * rng.standard_normal(G))
        gene_scale /= np.mean(gene_scale) + 1e-12
        gene_scale = np.clip(gene_scale, 0.2, 5.0)

    segment_names = [s[0] for s in segments]
    name_to_idx = {n: i for i, n in enumerate(segment_names)}
    edge_regime = np.full((G, G), -1, dtype=np.int8)
    branch_id = np.full((G, G), -1, dtype=np.int16)
    if knobs.state_dependent_edges:
        edge_regime, branch_id = _assign_edge_regimes(W, segment_names, rng, knobs)

    rank = max(1, int(knobs.latent_drift_rank))
    Bx = rng.standard_normal((G, rank)) / np.sqrt(rank)
    Bv = rng.standard_normal((G, rank)) / np.sqrt(rank)
    phase0 = float(knobs.latent_phase_shift) + 2 * np.pi * _stable_float(trial_seed, "latent_phase")

    X_parts, V_parts, D_parts, T_parts, P_parts = [], [], [], [], []
    seg_data = {}

    for name, parent, duration in segments:
        steps = int(round(duration / dt)) + 1
        alpha = np.linspace(0.0, 1.0, steps)
        x = np.zeros((steps, G), dtype=float)
        x[0] = base["init_by_seg"][name].copy() if parent is None else seg_data[parent]["x"][-1] + base["branch_offsets"][name]
        if knobs.gene_scale_heterogeneity:
            x[0] *= gene_scale
        v = np.zeros_like(x)
        d = np.zeros_like(x)
        state_noise = base["state_noise_by_seg"][name]
        vel_noise = base["vel_noise_by_seg"][name]
        branch_bias = base["branch_offsets"][name]

        for t in range(steps - 1):
            al = float(alpha[t])
            drift_x = _drift_vector(drift_type, drift_amp, al, name, base)
            drift_v = _drift_vector_v(drift_type, drift_amp, al, name, base, knobs)
            d[t] = drift_v

            if knobs.state_dependent_edges:
                M = _regime_mask(edge_regime, branch_id, al, name, knobs, name_to_idx)
                Wm = W * M
            else:
                Wm = W

            x_eff = _apply_regulation_nonlinearity(x[t], knobs)
            direct_reg = x_eff @ Wm
            indirect_reg = np.zeros_like(direct_reg)
            for op in indirect_ops:
                indirect_reg += x_eff @ op
            reg_for_state = direct_reg - decay * x[t]

            x[t + 1] = x[t] + dt * (reg_for_state + drift_x) + branch_bias + state_noise[t]
            v[t] = direct_reg + indirect_strength * indirect_reg + drift_v - decay * x[t] + vel_noise[t]

            if knobs.latent_drift:
                u = _latent_U(al, name, rank, phase0, trial_seed)
                if knobs.latent_x_scale > 0:
                    x[t + 1] += knobs.latent_x_scale * (Bx @ u)
                if knobs.latent_v_scale > 0:
                    v[t] += knobs.latent_v_scale * (Bv @ u)

            np.clip(x[t + 1], -SIM_STATE_CLIP, SIM_STATE_CLIP, out=x[t + 1])
            np.clip(v[t], -SIM_VELOCITY_CLIP, SIM_VELOCITY_CLIP, out=v[t])

            if knobs.gene_scale_heterogeneity:
                x[t + 1] *= gene_scale
                if knobs.gene_scale_apply_to_v:
                    v[t] *= gene_scale

        al = float(alpha[-1])
        drift_x_last = _drift_vector(drift_type, drift_amp, al, name, base)
        drift_v_last = _drift_vector_v(drift_type, drift_amp, al, name, base, knobs)
        d[-1] = drift_v_last
        if knobs.state_dependent_edges:
            M = _regime_mask(edge_regime, branch_id, al, name, knobs, name_to_idx)
            Wm = W * M
        else:
            Wm = W
        x_eff_last = _apply_regulation_nonlinearity(x[-1], knobs)
        direct_last = x_eff_last @ Wm
        indirect_last = np.zeros_like(direct_last)
        for op in indirect_ops:
            indirect_last += x_eff_last @ op
        v[-1] = direct_last + indirect_strength * indirect_last + drift_v_last - decay * x[-1] + vel_noise[-1]
        if knobs.latent_drift:
            u = _latent_U(al, name, rank, phase0, trial_seed)
            if knobs.latent_x_scale > 0:
                x[-1] += knobs.latent_x_scale * (Bx @ u)
            if knobs.latent_v_scale > 0:
                v[-1] += knobs.latent_v_scale * (Bv @ u)
        np.clip(x[-1], -SIM_STATE_CLIP, SIM_STATE_CLIP, out=x[-1])
        np.clip(v[-1], -SIM_VELOCITY_CLIP, SIM_VELOCITY_CLIP, out=v[-1])
        if knobs.gene_scale_heterogeneity:
            x[-1] *= gene_scale
            if knobs.gene_scale_apply_to_v:
                v[-1] *= gene_scale

        seg_data[name] = {"x": x}
        X_parts.append(x)
        V_parts.append(v)
        D_parts.append(d)
        T_parts.append(alpha)
        P_parts.append(np.array([name] * steps, dtype=object))

    X_full = np.vstack(X_parts)
    V_full = np.vstack(V_parts)
    D_full = np.vstack(D_parts)
    T_full = np.concatenate(T_parts)
    P_full = np.concatenate(P_parts)
    idx = base["sample_indices"]
    X, V, D, time, path = X_full[idx], V_full[idx], D_full[idx], T_full[idx], P_full[idx]
    order = np.argsort(time)
    X, V = X[order].copy(), V[order].copy()

    if knobs.lib_size_noise and knobs.lib_size_sigma > 0:
        lib = np.exp(knobs.lib_size_sigma * rng.standard_normal(X.shape[0]))
        lib /= np.mean(lib) + 1e-12
        lib = np.clip(lib, 0.25, 4.0)
        X = X * lib[:, None]
        # Optional V scaling: velocity is not guaranteed to track sequencing depth like counts; keep off unless ablating.
        if knobs.lib_size_apply_to_v:
            V = V * lib[:, None]

    out = {
        "X": X,
        "V": V,
        "D": D[order],
        "time": time[order],
        "path": path[order],
        "pseudotime": time[order],
        "W": W,
        "edge_regime": edge_regime,
        "branch_id": branch_id,
    }
    return out


def knobs_from_namespace(ns) -> Fig2RealismKnobs:
    """Build knobs from argparse Namespace (attribute names match CLI dest)."""
    return Fig2RealismKnobs(
        latent_drift=ns.latent_drift,
        latent_x_scale=ns.latent_x_scale,
        latent_v_scale=ns.latent_v_scale,
        latent_phase_shift=ns.latent_phase_shift,
        latent_drift_rank=ns.latent_drift_rank,
        xv_drift_mismatch=ns.xv_drift_mismatch,
        xv_mismatch_strength=ns.xv_mismatch_strength,
        xv_phase_shift=ns.xv_phase_shift,
        state_dependent_edges=ns.state_dependent_edges,
        state_edge_frac=ns.state_edge_frac,
        branch_specific_frac=ns.branch_specific_frac,
        early_late_frac=ns.early_late_frac,
        early_threshold=ns.early_threshold,
        late_threshold=ns.late_threshold,
        gene_scale_heterogeneity=ns.gene_scale_heterogeneity,
        gene_scale_sigma=ns.gene_scale_sigma,
        gene_scale_apply_to_v=ns.gene_scale_apply_to_v,
        lib_size_noise=ns.lib_size_noise,
        lib_size_sigma=ns.lib_size_sigma,
        lib_size_apply_to_v=ns.lib_size_apply_to_v,
    )
