"""
Fig2-style metrics on the same synthetic topologies as ``benchmark/run_benchmark.py``, but with a
**different purpose** than the regime-based realism-stress benchmark.

**Hard Mode (default, ``--hard-mode``):** a **structural stress test** aimed at breaking **pooled
global linear X→V** summaries (e.g. Ridge): **state-dependent** direct edges (branch / early–late
masks), **X–V drift mismatch**, and **asymmetric latent** programs (**V** ≫ **X** loadings). This is
**not** framed as a faithful biological realism benchmark—it engineers misspecification so local
pair-based structure can still be informative. Physics use **moderate** velocity noise and
**standard-ish** indirect strength—not extreme SNR or indirect-as-crutch. **Narrow** pseudotime pair
gaps (~0.04–0.18) keep local pairs in comparable regimes so **GeoBRIDGE** and pairwise scores can
track local Jacobian structure. Gene-scale and library factors stay **off** by default (CLI-ablatable).

**Baseline (``--no-hard-mode``):** milder settings aligned with the main synthetic .npz recipe
(100 genes, 5% edges, drift amp 10, indirect 0.35, ``branch_specific`` drift); realism toggles
default off; pair gap defaults to the older wide window (0.1–0.8).

Run from project root:
  python -m benchmark.fig2_benchmark_paper --reps 30
  python -m benchmark.fig2_benchmark_paper --no-hard-mode --reps 30   # mild baseline
  python -m benchmark.fig2_benchmark_paper --reps 10 --seed 0 --outdir results/fig2

Outputs (under --outdir, default current directory):
  Fig2_final_benchmark_results_edp50_rep{reps}.csv
  Fig2_final_benchmark_curve_data_rep{reps}.csv
  Fig2_wilcoxon_rep{reps}.csv
  Fig2_bar_metrics_rep{reps}.png  (AUPRC, S-AUPRC, EDP@50 bar chart; use --no-bar-plot to skip)
  simdata/*.npz under --simdata-dir
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, Ridge
from sklearn.metrics import average_precision_score, roc_auc_score

from geobridge_model.config import GeoBridgeConfig
from geobridge_model.model import GeoBridgeModel
from geobridge_model.spectral_k import select_k_by_tail_exhaustion

from .plots import save_fig2_bar_plots
from .synthetic import TOPOLOGIES as SYNTHETIC_TOPOLOGIES, generate_base, _module_ids
from .synthetic_drifts import ALL_DRIFT_TYPES, simulate_from_base_extended
from .fig2_rich_simulation import Fig2RealismKnobs, knobs_from_namespace, simulate_fig2_rich_trajectory

# Presets: Hard Mode (default CLI) vs mild baseline (--no-hard-mode).
HARD_MODE_PRESET = {
    "n_genes": 100,
    "n_cells": 2000,
    "edge_prob": 0.05,
    "drift_amp": 11.0,
    "indirect_strength": 0.18,
    "indirect_hops": 2,
    "drift_type": "nonlinear_branch",
    "pair_dt_lo": 0.04,
    "pair_dt_hi": 0.18,
    "velocity_noise_std": 0.18,
    "weight": 8.0,
    "state_noise_scale": 0.06,
    "init_scale": 0.15,
    "n_pairs": 22000,
}
BASELINE_PRESET = {
    "n_genes": 100,
    "n_cells": 1200,
    "edge_prob": 0.05,
    "drift_amp": 10.0,
    "indirect_strength": 0.35,
    "indirect_hops": 2,
    "drift_type": "branch_specific",
    "pair_dt_lo": 0.1,
    "pair_dt_hi": 0.8,
    "velocity_noise_std": 0.6,
    "weight": 4.0,
    "state_noise_scale": 0.10,
    "init_scale": 0.2,
    "n_pairs": 20000,
}

# Applied when user does not pass the corresponding CLI flag (ablatable per factor).
BASELINE_REALISM_PRESET = {
    "latent_drift": False,
    "latent_x_scale": 0.0,
    "latent_v_scale": 0.0,
    "latent_phase_shift": 0.0,
    "latent_drift_rank": 2,
    "xv_drift_mismatch": False,
    "xv_mismatch_strength": 0.0,
    "xv_phase_shift": 0.0,
    "state_dependent_edges": False,
    "state_edge_frac": 0.0,
    "branch_specific_frac": 0.45,
    "early_late_frac": 0.55,
    "early_threshold": 0.35,
    "late_threshold": 0.65,
    "gene_scale_heterogeneity": False,
    "gene_scale_sigma": 0.28,
    "gene_scale_apply_to_v": True,
    "lib_size_noise": False,
    "lib_size_sigma": 0.12,
    "lib_size_apply_to_v": False,
}

# Structural bundle (default hard): hurts pooled linear maps; local pairs still informative when
# ``pair_dt`` is narrow. Gene-scale / library off unless enabled on CLI.
HARD_REALISM_PRESET = {
    "latent_drift": True,
    "latent_x_scale": 0.05,
    "latent_v_scale": 0.22,
    "latent_phase_shift": 0.12,
    "latent_drift_rank": 2,
    "xv_drift_mismatch": True,
    "xv_mismatch_strength": 0.10,
    "xv_phase_shift": 0.02,
    "state_dependent_edges": True,
    "state_edge_frac": 0.03,
    "branch_specific_frac": 0.50,
    "early_late_frac": 0.50,
    "early_threshold": 0.38,
    "late_threshold": 0.62,
    "gene_scale_heterogeneity": False,
    "gene_scale_sigma": 0.25,
    "gene_scale_apply_to_v": False,
    "lib_size_noise": False,
    "lib_size_sigma": 0.12,
    "lib_size_apply_to_v": False,
}

# Module-level knobs (overwritten in main() from CLI; initialized to Hard Mode)
N_GENES = HARD_MODE_PRESET["n_genes"]
N_MODULES = 5
N_CELLS = HARD_MODE_PRESET["n_cells"]
REPS = 30
TOPOLOGIES = list(SYNTHETIC_TOPOLOGIES)
EDGE_PROB = HARD_MODE_PRESET["edge_prob"]
WEIGHT = HARD_MODE_PRESET["weight"]
DECAY = 2.0
V_NOISE_STD = HARD_MODE_PRESET["velocity_noise_std"]
DRIFT_AMP = HARD_MODE_PRESET["drift_amp"]
INDIRECT_STRENGTH = HARD_MODE_PRESET["indirect_strength"]
INDIRECT_HOPS = HARD_MODE_PRESET["indirect_hops"]
N_PAIRS = HARD_MODE_PRESET.get("n_pairs", 20000)
LAM = 1e-2
DT = 0.05
DRIFT_TYPE = HARD_MODE_PRESET["drift_type"]

# GeoBRIDGE / pairwise baselines: forward pseudotime gap window (overwritten in main() from CLI).
FIG2_PAIR_DT_LO = HARD_MODE_PRESET["pair_dt_lo"]
FIG2_PAIR_DT_HI = HARD_MODE_PRESET["pair_dt_hi"]


def module_groups_from_assignment(n_genes: int, n_modules: int):
    """Lists of gene indices per module (matches synthetic.make_grn assignment)."""
    mid = _module_ids(n_genes, n_modules)
    return [np.flatnonzero(mid == m) for m in range(n_modules)]


def compute_indirect_mask(W):
    A = (W != 0).astype(int)
    reach = np.zeros_like(A)
    Ak = A.copy()
    for _ in range(2, 6):
        Ak = Ak @ A
        reach |= (Ak != 0)
    indirect = (reach != 0) & (A == 0)
    np.fill_diagonal(indirect, False)
    return indirect


def simulate_synthetic_trial(
    topology: str,
    trial_seed: int,
    n_genes: int,
    n_modules: int,
    edge_prob: float,
    weight: float,
    decay: float,
    dt: float,
    n_cells: int,
    velocity_noise_std: float,
    indirect_strength: float,
    indirect_hops: int,
    drift_type: str,
    drift_amp: float,
    realism_knobs: Fig2RealismKnobs,
    state_noise_scale: float,
    init_scale: float,
):
    """
    One dataset: ``generate_base`` then either ``simulate_fig2_rich_trajectory`` (if any realism
    knob is on) or ``simulate_from_base_extended``. Ground-truth ``W`` is the full structural
    adjacency; regime metadata may be stored in ``edge_regime`` / ``branch_id`` when present.

    Returns X, V, t, branch_ids B, W, modules, indirect_mask, edge_regime, branch_id (last two
    may be None when using the legacy simulator only).
    """
    base = generate_base(
        topology,
        trial_seed,
        n_genes=n_genes,
        n_modules=n_modules,
        edge_prob=edge_prob,
        weight=weight,
        decay=decay,
        dt=dt,
        n_cells=n_cells,
        velocity_noise_std=velocity_noise_std,
        indirect_strength=indirect_strength,
        indirect_hops=indirect_hops,
        state_noise_scale=state_noise_scale,
        init_scale=init_scale,
    )
    if realism_knobs.any_enabled():
        sim = simulate_fig2_rich_trajectory(base, drift_type, drift_amp, realism_knobs, trial_seed)
    else:
        sim = simulate_from_base_extended(base, drift_type, drift_amp)
    X = sim["X"]
    V = sim["V"]
    if not np.isfinite(X).all() or not np.isfinite(V).all():
        warnings.warn(
            "Synthetic X/V contain non-finite values; try lowering --drift-amp or --indirect-strength.",
            stacklevel=2,
        )
    t = np.asarray(sim["pseudotime"], dtype=np.float64)
    path = sim["path"]
    path_str = np.asarray(path, dtype=str)
    _, branch_ids = np.unique(path_str, return_inverse=True)
    W = sim["W"]
    modules = module_groups_from_assignment(n_genes, n_modules)
    indirect_mask = compute_indirect_mask(W)
    edge_regime = sim.get("edge_regime")
    branch_id = sim.get("branch_id")
    return (
        X,
        V,
        t,
        branch_ids.astype(np.int32),
        W,
        modules,
        indirect_mask,
        edge_regime,
        branch_id,
    )


def sample_pairs(t):
    i = np.random.randint(0, len(t), N_PAIRS * 2)
    j = np.random.randint(0, len(t), N_PAIRS * 2)
    lo, hi = FIG2_PAIR_DT_LO, FIG2_PAIR_DT_HI
    mask = (t[j] - t[i] > lo) & (t[j] - t[i] < hi)
    return i[mask][:N_PAIRS], j[mask][:N_PAIRS]


def geobridge_dual_projection_score(X, V, t, B=None):
    """
    Repo GeoBRIDGE (dual): streaming pairspace + select_k_by_tail_exhaustion + scores_from_streaming_stats,
    same as benchmark/run_benchmark.py geobridge_dual. Pair window matches Fig2 sample_pairs time gap.
    """
    cfg = GeoBridgeConfig(
        min_forward_dt=FIG2_PAIR_DT_LO,
        max_forward_dt=FIG2_PAIR_DT_HI,
        max_pairs_per_cell=None,
        pair_seed=None,
        max_k=40,
        tail_fraction=0.25,
        tail_min_start=15,
        slope_window=3,
        stable_m=3,
        tau=2.5,
        project_y=True,
        normalize_score=True,
    )
    try:
        pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
            X,
            V,
            t,
            min_forward_dt=cfg.min_forward_dt,
            max_forward_dt=cfg.max_forward_dt,
            max_pairs_per_cell=cfg.max_pairs_per_cell,
            pair_seed=cfg.pair_seed,
            stream_num_threads=cfg.stream_num_threads,
        )
        spec = select_k_by_tail_exhaustion(
            pre["S"],
            cfg.max_k,
            cfg.tail_fraction,
            cfg.tail_min_start,
            cfg.slope_window,
            cfg.stable_m,
            cfg.tau,
        )
        S = GeoBridgeModel.scores_from_streaming_stats(
            pre,
            k=int(spec.k_selected),
            project_y=cfg.project_y,
            normalize_score=cfg.normalize_score,
            eps=cfg.eps,
        )
    except Exception as e:
        warnings.warn(f"geobridge_dual_projection_score failed: {e}", stacklevel=2)
        S = np.zeros((X.shape[1], X.shape[1]))
    np.fill_diagonal(S, 0.0)
    return S


def bridge_score(X, V, t, B=None):
    i, j = sample_pairs(t)
    dX = X[j] - X[i]
    dV = V[j] - V[i]
    Z = PCA(n_components=N_MODULES).fit_transform(X)
    dZ = Z[j] - Z[i]
    reg = LAM * np.eye(N_MODULES)
    try:
        BX = np.linalg.solve(dZ.T @ dZ + reg, dZ.T @ dX)
        BV = np.linalg.solve(dZ.T @ dZ + reg, dZ.T @ dV)
        dXp = dX - dZ @ BX
        dVp = dV - dZ @ BV
        S = (dXp.T @ dVp) / (np.abs(dXp).T @ np.abs(dVp) + 1e-9)
        np.fill_diagonal(S, 0.0)
    except Exception:
        S = np.zeros((N_GENES, N_GENES))
    return S


def naive_path_score(X, V, t, B=None):
    i, j = sample_pairs(t)
    dX = X[j] - X[i]
    dV = V[j] - V[i]
    S = (dX.T @ dV) / (np.abs(dX).T @ np.abs(dV) + 1e-9)
    np.fill_diagonal(S, 0.0)
    return S


def naive_fork_score(X, V, t, B):
    i = np.random.randint(0, len(t), N_PAIRS * 5)
    j = np.random.randint(0, len(t), N_PAIRS * 5)
    mask = (B[i] != B[j]) & (np.abs(t[i] - t[j]) < 0.05)
    i, j = i[mask][:N_PAIRS], j[mask][:N_PAIRS]
    if len(i) == 0:
        return np.zeros((N_GENES, N_GENES))
    dX = X[j] - X[i]
    dV = V[j] - V[i]
    S = (dX.T @ dV) / (np.abs(dX).T @ np.abs(dV) + 1e-9)
    np.fill_diagonal(S, 0.0)
    return S


def vel_corr_score(X, V, t=None, B=None):
    Xc = X - X.mean(0)
    Vc = V - V.mean(0)
    S = (Xc.T @ Vc) / (
        np.linalg.norm(Xc, axis=0)[:, None] * np.linalg.norm(Vc, axis=0)[None, :] + 1e-9
    )
    np.fill_diagonal(S, 0.0)
    return S


def ridge_score(X, V, t=None, B=None):
    S = np.zeros((N_GENES, N_GENES))
    for j in range(N_GENES):
        reg = Ridge(alpha=1.0).fit(X, V[:, j])
        S[:, j] = reg.coef_
    np.fill_diagonal(S, 0.0)
    return S


# Granger proxy: sklearn fits one Lasso per target gene — cap rows + iterations for Fig2 runtime.
GRANGER_MAX_TRANSITIONS = 900
GRANGER_LASSO_MAX_ITER = 280
GRANGER_LASSO_TOL = 5e-4


def granger_lasso_score(X, V, t, B=None):
    order = np.argsort(t)
    Xs = X[order]
    Xt, Xt1 = Xs[:-1], Xs[1:]
    n = Xt.shape[0]
    cap = min(n, GRANGER_MAX_TRANSITIONS)
    if n > cap:
        idx = np.unique(np.linspace(0, n - 1, num=cap, dtype=np.int64))
        Xt, Xt1 = Xt[idx], Xt1[idx]
    clf = Lasso(
        alpha=0.005,
        max_iter=GRANGER_LASSO_MAX_ITER,
        tol=GRANGER_LASSO_TOL,
    )
    clf.fit(Xt, Xt1)
    S = clf.coef_.T
    np.fill_diagonal(S, 0.0)
    return S


def scribe_proxy_score(X, V, t=None, B=None):
    Vres = np.zeros_like(V)
    for j in range(N_GENES):
        reg = Ridge(alpha=1.0).fit(X[:, [j]], V[:, j])
        Vres[:, j] = V[:, j] - reg.predict(X[:, [j]])
    Xc = X - X.mean(0)
    Vc = Vres - Vres.mean(0)
    S = (Xc.T @ Vc) / (
        np.linalg.norm(Xc, axis=0)[:, None] * np.linalg.norm(Vc, axis=0)[None, :] + 1e-9
    )
    np.fill_diagonal(S, 0.0)
    return S


def compute_metrics(W, S, modules, indirect_mask):
    mask_diag = ~np.eye(W.shape[0], dtype=bool)
    y_true = (W[mask_diag] != 0).astype(int)
    w_flat = W[mask_diag]
    s_raw = S[mask_diag]
    s_abs = np.abs(s_raw)

    auprc = average_precision_score(y_true, s_abs)
    auroc = roc_auc_score(y_true, s_abs)

    sign_match = np.sign(s_raw) == np.sign(w_flat)
    y_signed = (y_true == 1) & sign_match
    s_auprc = average_precision_score(y_signed, s_abs)

    ind_flat = indirect_mask[mask_diag]
    universe_mask = (y_true == 1) | (ind_flat == 1)
    s_univ = s_abs[universe_mask]
    y_univ = y_true[universe_mask]

    edp_50 = 0.0
    topk_curve_data = (np.array([0]), np.array([0]))

    if len(y_univ) > 0:
        sort_idx = np.argsort(-s_univ)
        y_sorted = y_univ[sort_idx]
        cumsum_direct = np.cumsum(y_sorted)
        k_axis = np.arange(1, len(y_sorted) + 1)
        ratio_curve = cumsum_direct / k_axis
        topk_curve_data = (k_axis, ratio_curve)

        limit = min(50, len(y_sorted))
        prec_sum = 0.0
        cum_correct = 0
        for k in range(limit):
            if y_sorted[k] == 1:
                cum_correct += 1
            prec_sum += cum_correct / (k + 1)
        if limit > 0:
            edp_50 = prec_sum / limit

    K = int(y_true.sum())
    top_k_indices = np.argsort(-s_abs)[:K]
    flat_indices_all = np.flatnonzero(mask_diag)
    flat_indices_selected = flat_indices_all[top_k_indices]
    r, c = np.unravel_index(flat_indices_selected, W.shape)

    mod_map = np.zeros(N_GENES, dtype=int)
    for m_id, genes in enumerate(modules):
        mod_map[genes] = m_id
    cross_fp = np.mean(mod_map[r] != mod_map[c])

    return auprc, s_auprc, auroc, edp_50, cross_fp, topk_curve_data


def modules_to_module_id(modules, G):
    module_id = -np.ones(G, dtype=np.int32)
    for m, genes in enumerate(modules):
        module_id[np.array(genes, dtype=int)] = m
    return module_id


METHODS = {
    "GeoBRIDGE (Dual)": geobridge_dual_projection_score,
    "BRIDGE": bridge_score,
    "Naive-Path": naive_path_score,
    "Naive-Fork": naive_fork_score,
    "Vel-Corr": vel_corr_score,
    "Ridge": ridge_score,
    "Granger": granger_lasso_score,
    "Scribe": scribe_proxy_score,
}


def _cli_provided(long_flag: str) -> bool:
    """True if ``long_flag`` or ``long_flag=...`` appears in argv (user override)."""
    eq = long_flag + "="
    for a in sys.argv[1:]:
        if a == long_flag or a.startswith(eq):
            return True
    return False


def _realism_cli_touched(attr: str) -> bool:
    """True if user set this dest (``--key``, ``--key=``, or BooleanOptionalAction ``--no-key``)."""
    needle = attr.replace("_", "-")
    for a in sys.argv[1:]:
        if not a.startswith("--"):
            continue
        body = a[2:].split("=", 1)[0]
        if needle == body:
            return True
        if body.startswith("no-") and needle == body[3:]:
            return True
    return False


def apply_fig2_realism_presets(args: argparse.Namespace) -> None:
    """Set modular realism defaults from hard vs baseline mode unless the user set that flag."""
    preset = HARD_REALISM_PRESET if args.hard_mode else BASELINE_REALISM_PRESET
    for k in preset:
        if not _realism_cli_touched(k):
            setattr(args, k, preset[k])


def apply_simulation_mode(args: argparse.Namespace) -> None:
    """
    If ``--no-hard-mode``, set simulation knobs to ``BASELINE_PRESET`` for any option
    not explicitly passed on the CLI (so ``--no-hard-mode --n-genes 200`` keeps 200 genes).
    """
    if args.hard_mode:
        return
    b = BASELINE_PRESET
    if not _cli_provided("--n-genes"):
        args.n_genes = b["n_genes"]
    if not _cli_provided("--edge-prob"):
        args.edge_prob = b["edge_prob"]
    if not _cli_provided("--drift-amp"):
        args.drift_amp = b["drift_amp"]
    if not _cli_provided("--indirect-strength"):
        args.indirect_strength = b["indirect_strength"]
    if not _cli_provided("--indirect-hops"):
        args.indirect_hops = b["indirect_hops"]
    if not _cli_provided("--drift-type"):
        args.drift_type = b["drift_type"]
    if not _cli_provided("--pair-dt-lo"):
        args.pair_dt_lo = b["pair_dt_lo"]
    if not _cli_provided("--pair-dt-hi"):
        args.pair_dt_hi = b["pair_dt_hi"]
    if not _cli_provided("--n-cells"):
        args.n_cells = b["n_cells"]
    if not _cli_provided("--velocity-noise-std"):
        args.velocity_noise_std = b["velocity_noise_std"]
    if not _cli_provided("--weight"):
        args.weight = b["weight"]


def parse_args():
    p = argparse.ArgumentParser(
        description="Fig2-style metrics on benchmark/synthetic.py data (main-benchmark topologies)."
    )
    p.add_argument(
        "--hard-mode",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Hard preset (default): sparse GRN + structural realism (state edges, X–V mismatch, "
        "latent V≫X) + moderate noise; narrow pair gaps for local methods. "
        "Use --no-hard-mode for mild baseline without structural stressors.",
    )
    p.add_argument("--reps", type=int, default=30, help="Repetitions per topology")
    p.add_argument("--seed", type=int, default=42, help="RNG seed")
    p.add_argument(
        "--n-genes",
        type=int,
        default=HARD_MODE_PRESET["n_genes"],
        help="Number of genes (Hard default: 100)",
    )
    p.add_argument("--n-modules", type=int, default=5)
    p.add_argument(
        "--n-cells",
        type=int,
        default=HARD_MODE_PRESET["n_cells"],
        help="Cells per trial (Hard default: 2000)",
    )
    p.add_argument(
        "--n-pairs",
        type=int,
        default=HARD_MODE_PRESET.get("n_pairs", 20000),
        help="Random pairs for pairwise baselines / BRIDGE (Hard default: 22000).",
    )
    p.add_argument(
        "--edge-prob",
        type=float,
        default=HARD_MODE_PRESET["edge_prob"],
        help="Within-module edge probability for GRN (Hard default: 0.05)",
    )
    p.add_argument(
        "--drift-type",
        default=HARD_MODE_PRESET["drift_type"],
        choices=list(ALL_DRIFT_TYPES),
        help="Drift mode (Hard default: nonlinear_branch).",
    )
    p.add_argument(
        "--drift-amp",
        type=float,
        default=HARD_MODE_PRESET["drift_amp"],
        help="Drift scale (Hard default: 11)",
    )
    p.add_argument(
        "--indirect-strength",
        type=float,
        default=HARD_MODE_PRESET["indirect_strength"],
        help="Indirect pathway weight in velocity (Hard default: ~0.18).",
    )
    p.add_argument(
        "--indirect-hops",
        type=int,
        default=HARD_MODE_PRESET["indirect_hops"],
        help="Multi-hop indirect operators (>=2)",
    )
    p.add_argument(
        "--velocity-noise-std",
        type=float,
        default=HARD_MODE_PRESET["velocity_noise_std"],
        help="Gaussian noise std on V in the simulator (Hard default: 0.18; baseline 0.6).",
    )
    p.add_argument(
        "--weight",
        type=float,
        default=HARD_MODE_PRESET["weight"],
        help="GRN edge weight scale in generate_base (Hard default: 8.0; baseline 4.0).",
    )
    p.add_argument(
        "--state-noise-scale",
        type=float,
        default=HARD_MODE_PRESET["state_noise_scale"],
        help="Gaussian noise std on X updates in the simulator (Hard default: 0.06; baseline 0.10).",
    )
    p.add_argument(
        "--init-scale",
        type=float,
        default=HARD_MODE_PRESET["init_scale"],
        help="Std of initial X per segment (Hard default: 0.15; baseline 0.2).",
    )
    p.add_argument(
        "--outdir",
        type=Path,
        default=Path("."),
        help="Directory for CSV outputs (created if missing)",
    )
    p.add_argument(
        "--simdata-dir",
        type=Path,
        default=Path("simdata"),
        help="Directory for per-rep .npz saves",
    )
    p.add_argument(
        "--topologies",
        nargs="+",
        default=list(SYNTHETIC_TOPOLOGIES),
        help="Same names as benchmark/synthetic.py and run_all.py",
    )
    p.add_argument(
        "--no-bar-plot",
        action="store_true",
        help="Skip saving Fig2_bar_metrics_rep{N}.png at the end",
    )
    p.add_argument(
        "--pair-dt-lo",
        type=float,
        default=HARD_MODE_PRESET["pair_dt_lo"],
        help="Min forward pseudotime gap (Hard default: 0.04).",
    )
    p.add_argument(
        "--pair-dt-hi",
        type=float,
        default=HARD_MODE_PRESET["pair_dt_hi"],
        help="Max forward pseudotime gap (Hard default: 0.18). Baseline mode defaults to 0.1–0.8 if unset.",
    )
    p.add_argument(
        "--latent-drift",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Latent U (sinusoidal); **on** in hard preset (asymmetric V≫X). Use --no-latent-drift to ablate.",
    )
    p.add_argument("--latent-x-scale", type=float, default=0.0, help="Scale for Bx @ U added to X.")
    p.add_argument("--latent-v-scale", type=float, default=0.0, help="Scale for Bv @ U added to V.")
    p.add_argument(
        "--latent-phase-shift",
        type=float,
        default=0.0,
        help="Extra phase offset for the latent program (U uses independent X/V loadings).",
    )
    p.add_argument("--latent-drift-rank", type=int, default=2, help="Latent dimension (rank of U).")
    p.add_argument(
        "--xv-drift-mismatch",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="V drift path blended vs X; **on** in hard preset. Use --no-xv-drift-mismatch to ablate.",
    )
    p.add_argument(
        "--xv-mismatch-strength",
        type=float,
        default=0.0,
        help="Mixing weight between X drift path and V drift path (0–1).",
    )
    p.add_argument(
        "--xv-phase-shift",
        type=float,
        default=0.0,
        help="Pseudotime shift (in [0,1] units) for the V-only drift path relative to X.",
    )
    p.add_argument(
        "--state-dependent-edges",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Regime-specific true edges (eval W unchanged); **on** in hard preset. Use --no-state-dependent-edges to ablate.",
    )
    p.add_argument(
        "--state-edge-frac",
        type=float,
        default=0.3,
        help="Fraction of nonzero W edges made regime-specific (among those assigned).",
    )
    p.add_argument(
        "--branch-specific-frac",
        type=float,
        default=0.45,
        help="Among regime-specific edges, relative weight for branch-tied vs early/late (normalized internally).",
    )
    p.add_argument(
        "--early-late-frac",
        type=float,
        default=0.55,
        help="Combined weight for early-only + late-only vs branch-specific (split 50/50 early/late).",
    )
    p.add_argument("--early-threshold", type=float, default=0.35, help="Early-only edges: alpha < this.")
    p.add_argument("--late-threshold", type=float, default=0.65, help="Late-only edges: alpha > this.")
    p.add_argument(
        "--gene-scale-heterogeneity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Per-gene scales; **off** in hard preset (structural stressors only).",
    )
    p.add_argument("--gene-scale-sigma", type=float, default=0.25, help="Log-normal sigma for gene scales.")
    p.add_argument(
        "--gene-scale-apply-to-v",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also multiply dynamic V by the same gene_scale vector (default true in hard preset).",
    )
    p.add_argument(
        "--lib-size-noise",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Per-cell lognormal library factor (off by default; ablation only).",
    )
    p.add_argument("--lib-size-sigma", type=float, default=0.15, help="Lognormal sigma for library factors.")
    p.add_argument(
        "--lib-size-apply-to-v",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Also scale V by library factors. Often questionable for RNA velocity vs counts depth — keep off unless ablating.",
    )
    return p.parse_args()


def main() -> None:
    global N_GENES, N_MODULES, N_CELLS, REPS, TOPOLOGIES, N_PAIRS
    global EDGE_PROB, WEIGHT, DECAY, DT, V_NOISE_STD, DRIFT_AMP
    global INDIRECT_STRENGTH, INDIRECT_HOPS, DRIFT_TYPE
    global FIG2_PAIR_DT_LO, FIG2_PAIR_DT_HI

    args = parse_args()
    apply_simulation_mode(args)
    apply_fig2_realism_presets(args)

    REPS = max(1, args.reps)
    N_GENES = args.n_genes
    N_MODULES = args.n_modules
    N_CELLS = args.n_cells
    N_PAIRS = args.n_pairs
    TOPOLOGIES = list(args.topologies)
    DRIFT_TYPE = args.drift_type
    DRIFT_AMP = args.drift_amp
    INDIRECT_STRENGTH = args.indirect_strength
    INDIRECT_HOPS = max(2, args.indirect_hops)
    EDGE_PROB = args.edge_prob
    WEIGHT = float(args.weight)
    V_NOISE_STD = float(args.velocity_noise_std)
    FIG2_PAIR_DT_LO = float(args.pair_dt_lo)
    FIG2_PAIR_DT_HI = float(args.pair_dt_hi)
    if FIG2_PAIR_DT_HI <= FIG2_PAIR_DT_LO:
        raise ValueError("--pair-dt-hi must be greater than --pair-dt-lo")

    realism_knobs = knobs_from_namespace(args)

    args.outdir.mkdir(parents=True, exist_ok=True)
    args.simdata_dir.mkdir(parents=True, exist_ok=True)

    warnings.filterwarnings("ignore")
    np.random.seed(args.seed)

    mode = "Hard mode" if args.hard_mode else "Baseline mode"
    print(
        f"Benchmark [{mode}]: {REPS} reps × {len(TOPOLOGIES)} topologies | "
        f"N_GENES={N_GENES} edge_prob={EDGE_PROB} N_CELLS={N_CELLS} | "
        f"drift={DRIFT_TYPE} (amp={DRIFT_AMP}) indirect={INDIRECT_STRENGTH} | "
        f"V_noise={V_NOISE_STD} state_noise={args.state_noise_scale:.3f} weight={WEIGHT} | "
        f"pair_dt=({FIG2_PAIR_DT_LO},{FIG2_PAIR_DT_HI}) | rich_sim={realism_knobs.any_enabled()}"
    )
    print(f"Topologies: {', '.join(TOPOLOGIES)}")
    print(f"Outputs → {args.outdir.resolve()}")

    results = {k: [] for k in METHODS}
    curve_data_list = []

    for ti, topo in enumerate(TOPOLOGIES):
        print(f"\nTopology: {topo}")
        for r in range(REPS):
            trial_seed = args.seed + 1_000_003 * r + ti * 97
            X, V, t, B, W, modules, indirect_mask, edge_regime, branch_id_mat = simulate_synthetic_trial(
                topo,
                trial_seed,
                n_genes=N_GENES,
                n_modules=N_MODULES,
                edge_prob=EDGE_PROB,
                weight=WEIGHT,
                decay=DECAY,
                dt=DT,
                n_cells=N_CELLS,
                velocity_noise_std=V_NOISE_STD,
                indirect_strength=INDIRECT_STRENGTH,
                indirect_hops=INDIRECT_HOPS,
                drift_type=DRIFT_TYPE,
                drift_amp=DRIFT_AMP,
                realism_knobs=realism_knobs,
                state_noise_scale=float(args.state_noise_scale),
                init_scale=float(args.init_scale),
            )
            G = W.shape[0]
            module_id = modules_to_module_id(modules, G)

            save_kw = dict(
                X=X.astype(np.float32),
                V=V.astype(np.float32),
                t=t.astype(np.float32),
                B=B.astype(np.int32),
                W=W.astype(np.float32),
                module_id=module_id.astype(np.int32),
                indirect_mask=indirect_mask.astype(np.int8),
                topology=np.array(topo),
                drift_type=np.array(DRIFT_TYPE),
                trial_seed=np.int32(trial_seed),
            )
            if edge_regime is not None:
                save_kw["edge_regime"] = np.asarray(edge_regime, dtype=np.int8)
            if branch_id_mat is not None:
                save_kw["branch_id"] = np.asarray(branch_id_mat, dtype=np.int16)
            np.savez(args.simdata_dir / f"sim_{topo}_rep{r}.npz", **save_kw)

            print(f"  [Rep {r + 1}/{REPS}]")
            for name, fn in METHODS.items():
                t0 = time.perf_counter()
                S = fn(X, V, t, B)
                elapsed = time.perf_counter() - t0
                auprc, s_auprc, auroc, edp_50, cross_fp, topk_curve = compute_metrics(
                    W, S, modules, indirect_mask
                )

                results[name].append(
                    {
                        "Topology": topo,
                        "Rep": r,
                        "AUPRC": auprc,
                        "S-AUPRC": s_auprc,
                        "AUROC": auroc,
                        "EDP@50": edp_50,
                        "Cross@K": cross_fp,
                        "Time": elapsed,
                        "TopK_Curve": topk_curve,
                    }
                )

                k_vals, prec_vals = topk_curve
                for kk, pr in zip(k_vals[:300], prec_vals[:300]):
                    curve_data_list.append(
                        {
                            "Method": name,
                            "Topology": topo,
                            "Rep": r,
                            "K": kk,
                            "Precision": pr,
                        }
                    )

                print(
                    f"    - {name:<18} | Time={elapsed:5.2f}s | "
                    f"AUPRC={auprc:.3f} | S-AUPRC={s_auprc:.3f} | "
                    f"AUROC={auroc:.3f} | EDP@50={edp_50:.3f}"
                )

    df_list = []
    for m, vals in results.items():
        for v in vals:
            df_list.append(
                {
                    "Method": m,
                    "Topology": v["Topology"],
                    "Rep": v["Rep"],
                    "AUPRC": v["AUPRC"],
                    "S-AUPRC": v["S-AUPRC"],
                    "AUROC": v["AUROC"],
                    "EDP@50": v["EDP@50"],
                    "Cross@K": v["Cross@K"],
                    "Time": v["Time"],
                }
            )

    df = pd.DataFrame(df_list)
    bench_csv = args.outdir / f"Fig2_final_benchmark_results_edp50_rep{REPS}.csv"
    curve_csv = args.outdir / f"Fig2_final_benchmark_curve_data_rep{REPS}.csv"
    df.to_csv(bench_csv, index=False)
    pd.DataFrame(curve_data_list).to_csv(curve_csv, index=False)
    print(f"\nWrote {bench_csv}")
    print(f"Wrote {curve_csv}")

    test_metrics = ["AUPRC", "S-AUPRC", "EDP@50"]
    comparison_method = "GeoBRIDGE (Dual)"
    stat_results = []

    for topo in TOPOLOGIES:
        topo_df = df[df["Topology"] == topo]
        for metric in test_metrics:
            pivot_df = topo_df.pivot(index="Rep", columns="Method", values=metric)
            if comparison_method not in pivot_df.columns:
                continue
            ref_scores = pivot_df[comparison_method]
            for other_method in METHODS:
                if other_method == comparison_method:
                    continue
                other_scores = pivot_df[other_method]
                try:
                    _, p_val = wilcoxon(ref_scores, other_scores, alternative="two-sided")
                except ValueError:
                    p_val = 1.0
                stars = "ns"
                if p_val < 0.001:
                    stars = "***"
                elif p_val < 0.01:
                    stars = "**"
                elif p_val < 0.05:
                    stars = "*"
                stat_results.append(
                    {
                        "Topology": topo,
                        "Metric": metric,
                        "Comparison": f"{comparison_method} vs {other_method}",
                        "P-Value": p_val,
                        "Significance": stars,
                    }
                )

    wilcox_csv = args.outdir / f"Fig2_wilcoxon_rep{REPS}.csv"
    pd.DataFrame(stat_results).to_csv(wilcox_csv, index=False)
    print(f"Wrote {wilcox_csv}")

    if not args.no_bar_plot:
        try:
            bar_name = f"Fig2_bar_metrics_rep{REPS}"
            save_fig2_bar_plots(
                df,
                args.outdir,
                topology_order=TOPOLOGIES,
                method_order=list(METHODS.keys()),
                filename_stem=bar_name,
            )
            print(f"Wrote {args.outdir / (bar_name + '.png')}")
        except Exception as e:
            warnings.warn(f"Fig2 bar plot failed: {e}", stacklevel=2)


if __name__ == "__main__":
    main()
