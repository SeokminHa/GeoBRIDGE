"""
Pancreas (A→C) real-data analysis for the GeoBRIDGE paper-style Fig3.

This script is a .py port of the notebook:
  Fig3_AtoC_Pancreas_paper.ipynb

Key difference vs the notebook: uses the repo's current GeoBRIDGE implementation
(`geobridge_model`) instead of the notebook's hand-rolled "partial bridge" code.

Outputs:
  - UMAP + velocity stream overlay
  - Branch-specificity plots (Pax4 / Arx) using GeoBRIDGE scores per branch
  - Top-N sensitivity + effect size (home vs away) using GeoBRIDGE |scores|
  - k-grid stability plot for Neurog3 with Enrichr (optional; requires gseapy + internet)
  - Graph plots: global top edges network + per-regulator top-target barplots
  - Optional `--orientation-verify`: full col/row TF vectors + duplicate top-N / specificity / Wilcoxon with __col/__row
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime

import numpy as np
try:
    import pandas as pd
except ModuleNotFoundError as e:
    raise ModuleNotFoundError(
        "Missing dependency 'pandas'. Install the notebook-equivalent stack, e.g.\n"
        "  pip install pandas numpy scipy scikit-learn matplotlib seaborn scanpy scvelo networkx gseapy\n"
        "Then re-run this script."
    ) from e


def _set_matplotlib_backend() -> None:
    # Avoid GUI requirements on headless runs.
    import matplotlib

    matplotlib.use("Agg", force=True)


def to_dense(X):
    import scipy.sparse as sp

    if sp.issparse(X):
        return X.toarray()
    return np.asarray(X)


@dataclass
class BranchPack:
    name: str
    adata: "object"  # anndata.AnnData
    pt_key: str
    pt_source: str  # "latent_time" | "dpt_pseudotime"


def compute_covariates(adata) -> None:
    import scanpy as sc

    # total counts
    if "total_counts" not in adata.obs.columns:
        Xc = to_dense(adata.X)
        adata.obs["total_counts"] = Xc.sum(axis=1)

    # percent mitochondrial
    if "pct_counts_mt" not in adata.obs.columns:
        genes = np.array(adata.var_names, dtype=str)
        mt_mask = (
            np.char.startswith(genes, "mt-")
            | np.char.startswith(genes, "Mt-")
            | np.char.startswith(genes, "MT-")
        )
        if int(mt_mask.sum()) > 0:
            Xc = to_dense(adata.X)
            mt_counts = Xc[:, mt_mask].sum(axis=1)
            adata.obs["pct_counts_mt"] = 100.0 * mt_counts / (
                adata.obs["total_counts"].values + 1e-12
            )
        else:
            adata.obs["pct_counts_mt"] = 0.0

    # cell cycle (minimal marker sets, robust to missing genes)
    S_GENES = ["MCM5", "PCNA", "TYMS", "MCM2", "MCM4", "RRM2", "CDC45", "MCM6", "MCM7"]
    G2M_GENES = ["TOP2A", "MKI67", "CENPF", "CCNB2", "CDK1", "UBE2C", "BIRC5"]
    s_present = [g for g in S_GENES if g in adata.var_names]
    g_present = [g for g in G2M_GENES if g in adata.var_names]
    if len(s_present) >= 3 and len(g_present) >= 3:
        sc.tl.score_genes_cell_cycle(adata, s_genes=s_present, g2m_genes=g_present)
    else:
        adata.obs["S_score"] = 0.0
        adata.obs["G2M_score"] = 0.0
        adata.obs["phase"] = "NA"


def pick_root_idx(ad_sub, early_clusters: list[str]) -> int:
    early_mask = ad_sub.obs["clusters"].isin(early_clusters).values
    idxs = np.where(early_mask)[0]
    if len(idxs) == 0:
        return 0
    Xp = ad_sub.obsm["X_pca"][idxs]
    return int(idxs[np.argmin(np.linalg.norm(Xp, axis=1))])


def make_branches(
    adata,
    *,
    seed: int = 42,
    force_dpt: bool = False,
    force_latent_time: bool = False,
    allow_dpt_fallback: bool = False,
) -> dict[str, BranchPack]:
    import scanpy as sc
    import scvelo as scv

    BRANCHES = {
        "Beta_branch": ["Pre-endocrine", "Ngn3 high EP", "Ngn3 low EP", "Beta"],
        "Alpha_branch": ["Pre-endocrine", "Ngn3 high EP", "Ngn3 low EP", "Alpha"],
    }
    EARLY_CLUSTERS = ["Pre-endocrine", "Ngn3 high EP", "Ngn3 low EP"]

    has_latent = "latent_time" in adata.obs.columns
    if force_latent_time and not has_latent and not allow_dpt_fallback:
        raise RuntimeError(
            "--force-latent-time is active, but adata.obs['latent_time'] is missing. "
            "Refusing to fall back to DPT. (Pass --allow-dpt-fallback to override, "
            "or pass --no-force-latent-time.)"
        )
    out: dict[str, BranchPack] = {}
    for bname, clist in BRANCHES.items():
        mask = adata.obs["clusters"].isin(clist).values
        ad_sub = adata[mask].copy()
        if ad_sub.n_obs < 200:
            continue

        # ensure embeddings exist
        if "X_pca" not in ad_sub.obsm:
            sc.tl.pca(ad_sub, n_comps=50, svd_solver="arpack", random_state=seed)

        use_latent = has_latent and ("latent_time" in ad_sub.obs.columns) and (not force_dpt)
        if force_latent_time:
            use_latent = bool("latent_time" in ad_sub.obs.columns)

        if use_latent:
            ad_sub.obs["pt_use"] = ad_sub.obs["latent_time"].values
            pt_key = "pt_use"
            pt_source = "latent_time"
        else:
            if force_latent_time and not allow_dpt_fallback:
                raise RuntimeError(
                    f"--force-latent-time is active but latent_time is missing on branch {bname}. "
                    "Refusing to fall back to DPT. (Pass --allow-dpt-fallback to override.)"
                )
            sc.tl.diffmap(ad_sub)
            root = pick_root_idx(ad_sub, EARLY_CLUSTERS)
            ad_sub.uns["iroot"] = root
            sc.tl.dpt(ad_sub)
            ad_sub.obs["pt_use"] = ad_sub.obs["dpt_pseudotime"].values
            pt_key = "pt_use"
            pt_source = "dpt_pseudotime"

        # scVelo expects neighbors/moments on subsets for some plots; keep minimal here
        try:
            sc.pp.neighbors(ad_sub, n_neighbors=30, n_pcs=50)
            scv.pp.moments(ad_sub, n_pcs=50, n_neighbors=30)
        except Exception:
            pass

        out[bname] = BranchPack(name=bname, adata=ad_sub, pt_key=pt_key, pt_source=pt_source)
    return out


def geobridge_scores(
    X: np.ndarray,
    V: np.ndarray,
    t: np.ndarray,
    *,
    min_dt: float,
    max_dt: float,
    max_pairs_per_cell: int | None = 40,
    stream_num_threads: int | None = 1,
    pair_mode: str = "stream_per_cell",
    global_max_pairs: int = 30000,
    max_k: int = 40,
    tail_fraction: float = 0.25,
    tail_min_start: int = 15,
    tau: float = 2.5,
    project_y: bool = True,
    normalize_score: bool = True,
    k_override: int | None = None,
    k_clamp: int | None = None,
    debug: bool = False,
):
    from geobridge_model.config import GeoBridgeConfig
    from geobridge_model.model import GeoBridgeModel
    from geobridge_model.spectral_k import select_k_by_tail_exhaustion

    cfg = GeoBridgeConfig(
        min_forward_dt=min_dt,
        max_forward_dt=max_dt,
        max_pairs_per_cell=max_pairs_per_cell,
        pair_seed=42,
        max_k=max_k,
        tail_fraction=tail_fraction,
        tail_min_start=tail_min_start,
        slope_window=3,
        stable_m=3,
        tau=tau,
        project_y=project_y,
        normalize_score=normalize_score,
    )
    if pair_mode == "stream_per_cell":
        pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
            X,
            V,
            t,
            min_forward_dt=cfg.min_forward_dt,
            max_forward_dt=cfg.max_forward_dt,
            max_pairs_per_cell=cfg.max_pairs_per_cell,
            pair_seed=cfg.pair_seed,
            stream_num_threads=stream_num_threads,
        )
    elif pair_mode == "global_sample":
        pre = precompute_pairspace_stats_global_sample(
            X,
            V,
            t,
            min_forward_dt=cfg.min_forward_dt,
            max_forward_dt=cfg.max_forward_dt,
            global_max_pairs=int(global_max_pairs),
            pair_seed=int(cfg.pair_seed or 42),
        )
    else:
        raise ValueError(f"Unknown pair_mode={pair_mode!r}")
    spec = select_k_by_tail_exhaustion(
        pre["S"],
        cfg.max_k,
        cfg.tail_fraction,
        cfg.tail_min_start,
        cfg.slope_window,
        cfg.stable_m,
        cfg.tau,
    )
    k_auto_raw = int(spec.k_selected)
    if k_override is None:
        k_use = k_auto_raw
    else:
        k_use = int(k_override)

    if k_clamp is not None:
        k_use = min(int(k_use), int(k_clamp))

    diag = None
    if debug:
        diag = projection_diagnostics_from_streaming(pre, k_use, project_y=cfg.project_y)
        diag["k_auto_raw"] = int(k_auto_raw)
        diag["k_used"] = int(k_use)

    S = GeoBridgeModel.scores_from_streaming_stats(
        pre,
        k=k_use,
        project_y=cfg.project_y,
        normalize_score=cfg.normalize_score,
        eps=cfg.eps,
    )
    np.fill_diagonal(S, 0.0)
    return S, int(k_use), int(pre["n_pairs"]), int(k_auto_raw), diag


def projection_diagnostics_from_streaming(pre: dict, k: int, *, project_y: bool) -> dict:
    """
    Lightweight diagnostics for how much ΔX energy is removed by projection and how much
    sign cancellation might be present in the numerator.
    """
    C = pre["C"]
    M = pre["M"]
    S = pre["S"]
    Vt = pre["Vt"]
    G = int(pre["G"])
    k = int(max(0, min(int(k), G)))
    trC = float(np.trace(C))
    if k <= 0:
        var_ret = 1.0
        num = M
    else:
        V = Vt.T
        Vk = V[:, :k]
        Q = np.eye(G) - Vk @ Vk.T
        trC_after = float(np.trace(Q @ C @ Q))
        var_ret = (trC_after / trC) if trC > 1e-12 else np.nan
        num = Q @ M

    # "num/den" proxy: global sign cancellation in numerator
    num_sum = float(np.sum(num))
    num_abs = float(np.sum(np.abs(num))) + 1e-12
    num_ratio = num_sum / num_abs

    # Useful spectral info: how large is the chosen component relative to top?
    s0 = float(S[0]) if len(S) > 0 else 0.0
    sk = float(S[k - 1]) if k > 0 and len(S) >= k else 0.0
    return {
        "traceC": trC,
        "variance_retained_trace_ratio": var_ret,
        "num_sum_over_abs": num_ratio,
        "s0": s0,
        "s_k": sk,
        "project_y": bool(project_y),
    }


def select_k_by_projection_rule(
    pre_by_branch: dict[str, dict],
    k_auto_raw_by_branch: dict[str, int],
    *,
    K_cap: int = 15,
    r_min: float = 0.05,
    asym_max: float = 3.0,
    project_y: bool = True,
    log_fn=None,
    debug_plot_path: "Path | None" = None,
) -> tuple[int, list[dict]]:
    """
    Unsupervised k selection based ONLY on projection diagnostics.

    Candidate set: k in {0..min(k_auto_raw_global, K_cap)} where k_auto_raw_global is the
    minimum k_auto_raw across branches (so k <= k_auto_raw for every branch).

    Validity constraints:
      - min_retained_variance >= r_min
      - branch_variance_asymmetry <= asym_max

    Selection:
      - k_selected = largest valid k (spec D: maximize k subject to constraints).
      - Tie-break (spec E): if several valid k share the same maximal k (impossible for integers), or for
        logging only — we record score_k = mean_retained_variance - abs(mean_num_proxy) at k_selected.
      - If none valid, fallback to k=1.
    """
    import numpy as _np

    if log_fn is None:
        log_fn = lambda _s: None

    if not pre_by_branch:
        raise ValueError("pre_by_branch is empty.")
    if set(pre_by_branch.keys()) != set(k_auto_raw_by_branch.keys()):
        raise ValueError("pre_by_branch and k_auto_raw_by_branch must have identical branch keys.")

    k_auto_raw_global = int(min(int(v) for v in k_auto_raw_by_branch.values()))
    k_hi = int(min(int(K_cap), int(max(0, k_auto_raw_global))))
    K_candidates = list(range(0, k_hi + 1))

    rows: list[dict] = []
    log_fn("[DEBUG] Projection-rule k selection (unsupervised):")
    log_fn(f"[DEBUG]   K_cap={int(K_cap)} r_min={float(r_min)} asym_max={float(asym_max)}")
    log_fn(f"[DEBUG]   k_auto_raw_by_branch={ {k: int(v) for k, v in k_auto_raw_by_branch.items()} }")
    log_fn(f"[DEBUG]   k_auto_raw_global(min)={k_auto_raw_global} => K_candidates={K_candidates}")
    log_fn("[DEBUG]   Per-k aggregate diagnostics:")

    for k in K_candidates:
        retained = {}
        num_proxy = {}
        for b, pre in pre_by_branch.items():
            d = projection_diagnostics_from_streaming(pre, int(k), project_y=project_y)
            retained[b] = float(d["variance_retained_trace_ratio"])
            num_proxy[b] = float(d["num_sum_over_abs"])

        vals = _np.asarray(list(retained.values()), dtype=float)
        min_ret = float(_np.min(vals))
        mean_ret = float(_np.mean(vals))
        max_ret = float(_np.max(vals))
        worst_branch = min(retained, key=retained.get)
        asym = float(max_ret / (min_ret + 1e-8))
        mean_num = float(_np.mean(_np.asarray(list(num_proxy.values()), dtype=float)))

        row = {
            "k": int(k),
            "min_retained_variance": min_ret,
            "mean_retained_variance": mean_ret,
            "max_retained_variance": max_ret,
            "worst_branch_retained_variance": str(worst_branch),
            "branch_variance_asymmetry": asym,
            "mean_num_proxy": mean_num,
            "score_k": float(mean_ret) - abs(float(mean_num)),
        }
        rows.append(row)
        log_fn(
            "[DEBUG]     "
            f"k={k:2d} min_ret={min_ret:.6f} mean_ret={mean_ret:.6f} "
            f"asym={asym:.3f} mean_num_proxy={mean_num:.6f}"
        )

    valid = [
        r
        for r in rows
        if (float(r["min_retained_variance"]) >= float(r_min))
        and (float(r["branch_variance_asymmetry"]) <= float(asym_max))
    ]
    if not valid:
        k_selected = 1
        log_fn("[DEBUG]   No k satisfied constraints; fallback k_selected=1")
    else:
        k_largest_valid = int(max(int(r["k"]) for r in valid))
        k_selected = k_largest_valid
        row_sel = next((r for r in rows if int(r["k"]) == int(k_selected)), None)
        if row_sel is not None:
            log_fn(
                f"[DEBUG]   k_largest_valid={k_largest_valid} -> k_selected={k_selected} "
                f"(score_k={float(row_sel['score_k']):.6f})"
            )
        else:
            log_fn(f"[DEBUG]   k_largest_valid={k_largest_valid} -> k_selected={k_selected}")

    if debug_plot_path is not None:
        try:
            import matplotlib.pyplot as plt

            ks = [int(r["k"]) for r in rows]
            y_min = [float(r["min_retained_variance"]) for r in rows]
            y_mean = [float(r["mean_retained_variance"]) for r in rows]
            y_asym = [float(r["branch_variance_asymmetry"]) for r in rows]
            y_num = [float(r["mean_num_proxy"]) for r in rows]

            fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.8))
            ax.plot(ks, y_min, marker="o", label="min retained variance")
            ax.plot(ks, y_mean, marker="o", label="mean retained variance")
            ax.axhline(float(r_min), color="red", linestyle="--", linewidth=1.4, label="r_min")
            ax.set_xlabel("k (PCs removed)")
            ax.set_ylabel("variance retained (trace ratio)")
            ax.set_title("Projection diagnostics vs k")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, loc="best")
            fig.tight_layout()
            fig.savefig(debug_plot_path, dpi=220)
            plt.close(fig)

            # secondary plot: asymmetry + mean_num_proxy
            fig, ax = plt.subplots(1, 1, figsize=(8.8, 4.8))
            ax.plot(ks, y_asym, marker="o", label="branch variance asymmetry")
            ax.axhline(float(asym_max), color="red", linestyle="--", linewidth=1.4, label="asym_max")
            ax2 = ax.twinx()
            ax2.plot(ks, y_num, marker="o", color="#ff7f0e", label="mean num proxy")
            ax.set_xlabel("k (PCs removed)")
            ax.set_ylabel("asymmetry (max/min)")
            ax2.set_ylabel("mean_num_proxy (sum/abs)")
            ax.set_title("Asymmetry + sign-cancellation proxy vs k")
            ax.grid(True, alpha=0.25)
            # merge legends
            lines, labels = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines + lines2, labels + labels2, frameon=False, loc="best")
            fig.tight_layout()
            fig.savefig(str(debug_plot_path).replace(".png", "__asym_num.png"), dpi=220)
            plt.close(fig)
        except Exception as e:
            log_fn(f"[DEBUG] k-selection debug plot failed: {type(e).__name__}: {e}")

    return int(k_selected), rows


def precompute_pairspace_stats_global_sample(
    X,
    V,
    pseudotime,
    *,
    min_forward_dt: float,
    max_forward_dt: float,
    global_max_pairs: int = 30000,
    pair_seed: int = 42,
):
    """
    Compatibility mode: sample up to global_max_pairs uniformly from the set of all valid forward
    pairs (i,j) with min_dt < t[j]-t[i] <= max_dt, then compute exact streaming statistics over
    that sampled pair set (without materializing all pairs).
    """
    from geobridge_model.pairspace import monotonic_forward_windows

    X = np.ascontiguousarray(X, dtype=np.float64)
    V = np.ascontiguousarray(V, dtype=np.float64)
    pt = np.asarray(pseudotime, dtype=np.float64).reshape(-1)
    order = np.argsort(pt)
    X = X[order]
    V = V[order]
    pt = pt[order]
    n, G = X.shape
    if n <= 1:
        raise ValueError("Need at least two cells.")

    start, end = monotonic_forward_windows(pt, float(min_forward_dt), float(max_forward_dt))
    m = (end - start).astype(np.int64)
    m = np.maximum(m, 0)
    total = int(m.sum())
    if total <= 0:
        raise ValueError("No admissible forward pairs were generated.")

    rng = np.random.default_rng(int(pair_seed))
    P = min(int(global_max_pairs), total)
    probs = m / float(total)
    # sample source i proportional to how many valid js it has
    i_samp = rng.choice(n, size=P, replace=True, p=probs)

    # for each chosen i, sample a valid j uniformly in [start[i], end[i])
    by_i: dict[int, list[int]] = {}
    for i in i_samp.tolist():
        s = int(start[i])
        e = int(end[i])
        if s >= e:
            continue
        j = int(rng.integers(s, e))
        by_i.setdefault(int(i), []).append(j)

    sum_dx = np.zeros(G, dtype=np.float64)
    sum_dv = np.zeros(G, dtype=np.float64)
    sum_dx_dx = np.zeros((G, G), dtype=np.float64)
    sum_dx_dv = np.zeros((G, G), dtype=np.float64)
    sum_dv_dv = np.zeros((G, G), dtype=np.float64)
    n_pairs = 0
    for i, js in by_i.items():
        js = np.asarray(js, dtype=np.int64)
        dx = X[js] - X[i]
        dv = V[js] - V[i]
        n_pairs += int(dx.shape[0])
        sum_dx += dx.sum(axis=0)
        sum_dv += dv.sum(axis=0)
        sum_dx_dx += dx.T @ dx
        sum_dx_dv += dx.T @ dv
        sum_dv_dv += dv.T @ dv

    if n_pairs <= 0:
        raise ValueError("No admissible forward pairs were generated (after sampling).")

    mu = sum_dx / float(n_pairs)
    C = sum_dx_dx - float(n_pairs) * np.outer(mu, mu)
    M = sum_dx_dv - np.outer(mu, sum_dv)
    Y = sum_dv_dv
    C = 0.5 * (C + C.T)
    evals, evecs = np.linalg.eigh(C)
    o = np.argsort(evals)[::-1]
    evals = np.maximum(evals[o], 0.0)
    evecs = evecs[:, o]
    S = np.sqrt(evals)
    Vt = evecs.T
    return {
        "n_pairs": int(n_pairs),
        "G": int(G),
        "mu_dx": mu,
        "C": C,
        "M": M,
        "Y": Y,
        "S": S,
        "Vt": Vt,
    }


def save_umap_velocity(adata, out_png: Path) -> None:
    import matplotlib.pyplot as plt
    import scanpy as sc
    import scvelo as scv

    if "X_umap" not in adata.obsm:
        sc.tl.umap(adata, min_dist=0.3, spread=1.0, random_state=42)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    sc.pl.umap(
        adata,
        color="clusters",
        size=18,
        frameon=False,
        legend_loc=None,
        show=False,
        ax=ax,
    )
    scv.pl.velocity_embedding_stream(
        adata,
        basis="umap",
        color="clusters",
        density=1.2,
        linewidth=1,
        arrowsize=1,
        legend_loc=None,
        show=False,
        ax=ax,
    )
    ax.set_title("")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def build_pancreas_tf_score_cache(
    score_mats: dict,
    configs: list,
    analysis_label: str,
    orientation: str,
) -> dict[tuple[str, str], pd.DataFrame]:
    """
    Extract per-TF score vectors for Pax4/Arx-style analyses.

    orientation:
      - \"col\": scores = S[:, j]  (TF indexes a column — upstream-of-TF view if rows=regulators)
      - \"row\": scores = S[j, :]  (TF indexes a row — downstream-of-TF view if rows=regulators)
    """
    o = str(orientation).lower().strip()
    if o not in ("col", "row"):
        raise ValueError(f"orientation must be 'col' or 'row', got {orientation!r}")
    out: dict[tuple[str, str], pd.DataFrame] = {}
    for cfg in configs:
        tf = cfg["name"]
        for b in (cfg["home_branch"], cfg["away_branch"]):
            key = (b, analysis_label)
            if key not in score_mats:
                continue
            genes = score_mats[key]["genes"]
            S = score_mats[key]["S"]
            glist = genes.tolist()
            if tf not in set(glist):
                continue
            j = int(np.where(genes == tf)[0][0])
            if o == "col":
                vec = np.asarray(S[:, j], dtype=float)
            else:
                vec = np.asarray(S[j, :], dtype=float)
            out[(b, tf)] = pd.DataFrame({"Target": genes, "Score": vec})
    return out


def topn_home_vs_away(
    summary_out: Path,
    plot_png: Path,
    score_cache: dict,
    configs,
    top_list,
    *,
    analysis_label: str = "k_auto",
    log_fn=None,
    score_orientation: str | None = None,
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import mannwhitneyu

    def mannwhitney_effects(home_vals, away_vals, alternative="greater"):
        home_vals = np.asarray(home_vals, dtype=float)
        away_vals = np.asarray(away_vals, dtype=float)
        res = mannwhitneyu(home_vals, away_vals, alternative=alternative)
        n1, n2 = len(home_vals), len(away_vals)
        U = res.statistic
        auc = U / (n1 * n2)
        cliffs_delta = 2 * auc - 1
        return {
            "U": U,
            "p": res.pvalue,
            "AUC": auc,
            "cliffs_delta": cliffs_delta,
            "median_home": np.median(home_vals),
            "median_away": np.median(away_vals),
            "delta_median": np.median(home_vals) - np.median(away_vals),
            "n": (n1, n2),
        }

    rows = []
    long_df = []
    for cfg in configs:
        tf = cfg["name"]
        home_b = cfg["home_branch"]
        away_b = cfg["away_branch"]
        df_home_all = score_cache[(home_b, tf)].copy()
        df_away_all = score_cache[(away_b, tf)].copy()
        df_home_all["Abs"] = df_home_all["Score"].abs()
        df_away_all["Abs"] = df_away_all["Score"].abs()
        df_home_sorted = df_home_all.sort_values("Abs", ascending=False).reset_index(drop=True)
        for top_n in top_list:
            top_targets = df_home_sorted.head(top_n)["Target"].tolist()
            home_vals = df_home_all[df_home_all["Target"].isin(top_targets)]["Abs"].values
            away_vals = df_away_all[df_away_all["Target"].isin(top_targets)]["Abs"].values
            stats = mannwhitney_effects(home_vals, away_vals, alternative="greater")
            row_d = {
                "analysis_label": analysis_label,
                "TF": tf,
                "TOP_N": top_n,
                "home_branch": home_b,
                "away_branch": away_b,
                **stats,
            }
            if score_orientation is not None:
                row_d["score_orientation"] = str(score_orientation)
            rows.append(row_d)
            long_df.append(
                pd.DataFrame(
                    {
                        "TF": tf,
                        "TOP_N": top_n,
                        "Branch": [f"{home_b} (Home)"] * len(home_vals) + [f"{away_b} (Away)"] * len(away_vals),
                        "Abs_Score": np.concatenate([home_vals, away_vals]),
                    }
                )
            )

    summary = pd.DataFrame(rows).sort_values(["TF", "TOP_N"])
    plot_df = pd.concat(long_df, ignore_index=True)
    summary_out = Path(summary_out)
    plot_png = Path(plot_png)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    plot_png.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(summary_out, index=False)
    if log_fn is not None:
        log_fn(f"Saved top-N summary CSV: {summary_out.resolve()}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, cfg in zip(axes, configs):
        tf = cfg["name"]
        d = plot_df[plot_df["TF"] == tf].copy()
        home_label = f"{cfg['home_branch']} (Home)"
        away_label = f"{cfg['away_branch']} (Away)"
        d["Branch"] = pd.Categorical(d["Branch"], categories=[home_label, away_label], ordered=True)
        sns.boxplot(
            data=d,
            x="TOP_N",
            y="Abs_Score",
            hue="Branch",
            ax=ax,
            showfliers=False,
            palette={home_label: "#2ca02c", away_label: "#1f77b4"},
        )
        otag = f", orient={score_orientation}" if score_orientation else ""
        ax.set_title(f"{tf}: Home vs Away |Score| (label={analysis_label}{otag})")
        ax.set_xlabel("Top N targets (by |Home|)")
        ax.set_ylabel("|GeoBRIDGE score|")
        ax.legend(frameon=False, title="")
        ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(plot_png, dpi=220)
    if log_fn is not None:
        log_fn(f"Saved top-N boxplots PNG: {plot_png.resolve()}")
    plt.close(fig)
    return summary


def plot_branch_specificity(
    tf_name: str,
    home_branch: str,
    away_branch: str,
    score_cache: dict,
    out_png: Path,
    *,
    log_fn=None,
    analysis_label: str = "k_auto",
    score_orientation: str | None = None,
):
    """
    Paired Wilcoxon on Top-200 targets (by |home score|): test |Score_home| > |Score_away| (alternative='greater').
    Returns a dict suitable for CSV + logging; saves figure to out_png.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import wilcoxon

    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    out_png = Path(out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    df_home = score_cache[(home_branch, tf_name)].copy()
    df_away = score_cache[(away_branch, tf_name)].copy()
    merged = pd.merge(df_home, df_away, on="Target", suffixes=("_home", "_away"))
    merged["Abs_Home"] = merged["Score_home"].abs()
    top_targets = merged.sort_values("Abs_Home", ascending=False).head(200)

    fig = plt.figure(figsize=(14.5, 7.2))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.0, 0.6], wspace=0.05)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    ax1.scatter(
        top_targets["Score_home"],
        top_targets["Score_away"],
        color="#333333",
        alpha=0.7,
        s=55,
        edgecolors="none",
    )
    ax1.plot([-0.8, 0.8], [-0.8, 0.8], color="#d62728", linestyle="--", alpha=0.6, lw=2)
    ax1.set_xlim(-0.8, 0.8)
    ax1.set_ylim(-0.8, 0.8)
    ax1.set_aspect("equal")
    ax1.set_xlabel(f"GeoBRIDGE score in {home_branch.split('_')[0]} (Home)")
    ax1.set_ylabel(f"GeoBRIDGE score in {away_branch.split('_')[0]} (Away)")
    ax1.set_title(f"{tf_name}: Branch specificity (Top 200 by |Home|)")

    plot_data = pd.DataFrame(
        {
            "Branch": [f"{home_branch.split('_')[0]}\n(Home)"] * len(top_targets)
            + [f"{away_branch.split('_')[0]}\n(Away)"] * len(top_targets),
            "Abs_Score": np.concatenate([top_targets["Score_home"].abs(), top_targets["Score_away"].abs()]),
        }
    )
    palette = {
        f"{home_branch.split('_')[0]}\n(Home)": "#2ca02c" if "Beta" in home_branch else "#1f77b4",
        f"{away_branch.split('_')[0]}\n(Away)": "#2ca02c" if "Beta" in away_branch else "#1f77b4",
    }
    sns.boxplot(
        data=plot_data,
        x="Branch",
        y="Abs_Score",
        palette=palette,
        ax=ax2,
        width=0.7,
        linewidth=2.0,
        showfliers=False,
    )
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("")
    ax2.set_xlabel("")

    home_vals = top_targets["Score_home"].abs().to_numpy(dtype=float)
    away_vals = top_targets["Score_away"].abs().to_numpy(dtype=float)
    n = int(len(home_vals))
    p_val = float("nan")
    stat = float("nan")
    wilcoxon_error = ""
    try:
        # SciPy >= 1.11: method= avoids warnings; paired |home| vs |away|, one-sided greater.
        w = wilcoxon(home_vals, away_vals, alternative="greater", method="auto")
        p_val = float(w.pvalue)
        stat = float(w.statistic)
    except TypeError:
        try:
            w = wilcoxon(home_vals, away_vals, alternative="greater")
            p_val = float(w.pvalue)
            stat = float(w.statistic)
        except Exception as e:
            wilcoxon_error = f"{type(e).__name__}: {e}"
    except Exception as e:
        wilcoxon_error = f"{type(e).__name__}: {e}"

    if wilcoxon_error:
        _log(f"[WARN] Wilcoxon failed for TF={tf_name}: {wilcoxon_error}")

    p_title = p_val if np.isfinite(p_val) else float("nan")
    otag = f" orient={score_orientation}" if score_orientation else ""
    fig.suptitle(
        f"{tf_name}: Wilcoxon(|Home|>|Away|) p={p_title:.2e} (n={n}){otag}",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    try:
        fig.savefig(out_png, dpi=220, bbox_inches="tight")
        _log(f"Saved branch specificity figure: {out_png.resolve()}")
    except Exception as e:
        _log(f"[ERROR] Failed to save {out_png}: {type(e).__name__}: {e}")
        raise
    finally:
        plt.close(fig)

    return {
        "analysis_label": str(analysis_label),
        "score_orientation": score_orientation,
        "TF": tf_name,
        "home_branch": home_branch,
        "away_branch": away_branch,
        "test": "paired_wilcoxon_abs_score_greater",
        "alternative": "greater",
        "top_n_targets": n,
        "wilcoxon_statistic": stat,
        "wilcoxon_p": p_val,
        "wilcoxon_error": wilcoxon_error or None,
        "figure_path": str(out_png.resolve()),
    }


def save_top_edges_graph(scores: np.ndarray, genes: np.ndarray, out_png: Path, top_m: int = 300):
    import matplotlib.pyplot as plt
    import networkx as nx

    G = scores.shape[0]
    s = np.abs(scores).copy()
    np.fill_diagonal(s, 0.0)
    flat = s.ravel()
    idx = np.argpartition(-flat, top_m)[:top_m]
    r, c = np.unravel_index(idx, (G, G))
    w = flat[idx]

    g = nx.DiGraph()
    for i in range(G):
        g.add_node(genes[i])
    for i, j, wij in zip(r.tolist(), c.tolist(), w.tolist()):
        g.add_edge(genes[i], genes[j], weight=float(wij))

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111)
    ax.axis("off")
    pos = nx.spring_layout(g, seed=42, k=0.6)
    widths = [0.2 + 2.5 * g[u][v]["weight"] / (max(w) + 1e-12) for u, v in g.edges()]
    nx.draw_networkx_nodes(g, pos, node_size=30, node_color="#444444", alpha=0.85, ax=ax)
    nx.draw_networkx_edges(g, pos, width=widths, alpha=0.25, edge_color="#1f77b4", arrows=False, ax=ax)
    ax.set_title(f"GeoBRIDGE top-{top_m} |edges| (global)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


# Regulators used for optional artifact / orientation diagnostics (extend for other datasets).
REGULATOR_DIAGNOSTIC_TFS: tuple[str, ...] = ("Pax4", "Arx")


def _signed_vector_stats(scores: np.ndarray, *, eps_near: float = 0.01) -> dict:
    s = np.asarray(scores, dtype=float).ravel()
    n = int(s.size)
    if n == 0:
        return {
            "n_genes": 0,
            "frac_positive": float("nan"),
            "frac_negative": float("nan"),
            "frac_near_zero": float("nan"),
            "median_signed": float("nan"),
            "median_abs": float("nan"),
            "q05_signed": float("nan"),
            "q25_signed": float("nan"),
            "q50_signed": float("nan"),
            "q75_signed": float("nan"),
            "q95_signed": float("nan"),
        }
    return {
        "n_genes": n,
        "frac_positive": float(np.mean(s > 0)),
        "frac_negative": float(np.mean(s < 0)),
        "frac_near_zero": float(np.mean(np.abs(s) < eps_near)),
        "median_signed": float(np.median(s)),
        "median_abs": float(np.median(np.abs(s))),
        "q05_signed": float(np.percentile(s, 5)),
        "q25_signed": float(np.percentile(s, 25)),
        "q50_signed": float(np.percentile(s, 50)),
        "q75_signed": float(np.percentile(s, 75)),
        "q95_signed": float(np.percentile(s, 95)),
    }


def collect_pairwise_delta_gene(
    X: np.ndarray,
    t: np.ndarray,
    gene_idx: int,
    *,
    pair_mode: str,
    min_forward_dt: float,
    max_forward_dt: float,
    max_pairs_per_cell: int | None,
    global_max_pairs: int,
    pair_seed: int,
) -> np.ndarray:
    """
    ΔX_gene = X[j,g] - X[i,g] over admissible forward pairs, matching GeoBRIDGE pair construction
    (stream_per_cell cap / RNG or global_sample with replacement sampling of sources).
    """
    from geobridge_model.pairspace import monotonic_forward_windows

    X = np.ascontiguousarray(X, dtype=np.float64)
    pt = np.asarray(t, dtype=np.float64).reshape(-1)
    n = int(X.shape[0])
    g = int(gene_idx)
    deltas: list[float] = []

    if str(pair_mode) == "global_sample":
        order = np.argsort(pt)
        Xs = X[order]
        pts = pt[order]
        start, end = monotonic_forward_windows(pts, float(min_forward_dt), float(max_forward_dt))
        m = (end - start).astype(np.int64)
        m = np.maximum(m, 0)
        total = int(m.sum())
        if total <= 0:
            return np.array([], dtype=np.float64)
        rng = np.random.default_rng(int(pair_seed))
        P = min(int(global_max_pairs), total)
        probs = m / float(total)
        i_samp = rng.choice(n, size=P, replace=True, p=probs)
        by_i: dict[int, list[int]] = {}
        for i in i_samp.tolist():
            s = int(start[i])
            e = int(end[i])
            if s >= e:
                continue
            j = int(rng.integers(s, e))
            by_i.setdefault(int(i), []).append(j)
        for i, js in by_i.items():
            xi = Xs[int(i), g]
            for j in js:
                deltas.append(float(Xs[int(j), g] - xi))
        return np.asarray(deltas, dtype=np.float64)

    if str(pair_mode) == "stream_per_cell":
        rng = np.random.default_rng(int(pair_seed))
        cap = None if max_pairs_per_cell is None else int(max_pairs_per_cell)
        monotonic = bool(np.all(pt[1:] >= pt[:-1]))
        if monotonic:
            start, end = monotonic_forward_windows(pt, float(min_forward_dt), float(max_forward_dt))
            for i in range(n - 1):
                s = int(start[i])
                e = int(end[i])
                if s >= e:
                    continue
                js = np.arange(s, e, dtype=np.int64)
                if cap is not None and js.size > cap:
                    js = rng.choice(js, size=cap, replace=False)
                xi = X[int(i), g]
                for j in js.tolist():
                    deltas.append(float(X[int(j), g] - xi))
        else:
            for i in range(n - 1):
                dt = pt[i + 1 :] - pt[i]
                valid = np.where(dt > float(min_forward_dt))[0]
                if max_forward_dt is not None:
                    valid = valid[dt[valid] <= float(max_forward_dt)]
                if valid.size == 0:
                    continue
                js = (i + 1 + valid).astype(np.int64)
                if cap is not None and js.size > cap:
                    js = rng.choice(js, size=cap, replace=False)
                xi = X[int(i), g]
                for j in js.tolist():
                    deltas.append(float(X[int(j), g] - xi))
        return np.asarray(deltas, dtype=np.float64)

    raise ValueError(f"Unknown pair_mode={pair_mode!r}")


def _delta_x_summary_stats(d: np.ndarray) -> dict:
    x = np.asarray(d, dtype=float).ravel()
    n = int(x.size)
    if n == 0:
        return {
            "n_pairs_examined": 0,
            "mean_delta_x": float("nan"),
            "median_delta_x": float("nan"),
            "std_delta_x": float("nan"),
            "mean_abs_delta_x": float("nan"),
            "median_abs_delta_x": float("nan"),
            "q05_delta_x": float("nan"),
            "q25_delta_x": float("nan"),
            "q50_delta_x": float("nan"),
            "q75_delta_x": float("nan"),
            "q95_delta_x": float("nan"),
            "frac_abs_delta_x_below_1e-4": float("nan"),
            "frac_abs_delta_x_below_1e-3": float("nan"),
            "frac_positive_delta_x": float("nan"),
            "frac_negative_delta_x": float("nan"),
        }
    ax = np.abs(x)
    return {
        "n_pairs_examined": n,
        "mean_delta_x": float(np.mean(x)),
        "median_delta_x": float(np.median(x)),
        "std_delta_x": float(np.std(x)),
        "mean_abs_delta_x": float(np.mean(ax)),
        "median_abs_delta_x": float(np.median(ax)),
        "q05_delta_x": float(np.percentile(x, 5)),
        "q25_delta_x": float(np.percentile(x, 25)),
        "q50_delta_x": float(np.percentile(x, 50)),
        "q75_delta_x": float(np.percentile(x, 75)),
        "q95_delta_x": float(np.percentile(x, 95)),
        "frac_abs_delta_x_below_1e-4": float(np.mean(ax < 1e-4)),
        "frac_abs_delta_x_below_1e-3": float(np.mean(ax < 1e-3)),
        "frac_positive_delta_x": float(np.mean(x > 0)),
        "frac_negative_delta_x": float(np.mean(x < 0)),
    }


def run_regulator_artifact_diagnostics(
    *,
    branches: dict,
    score_mats: dict,
    analysis_label: str,
    configs: list,
    args,
    figdir: Path,
    tabdir: Path,
    log,
    k_rule,
) -> dict:
    """
    A–C + interpretation (E). Does not modify score_mats or default outputs.
    Returns a dict of key metrics for logging.
    """
    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    figdir = Path(figdir)
    tabdir = Path(tabdir)
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)

    # --- A: signed row scores ---
    rows_a: list[dict] = []
    hist_series: dict[str, dict[str, np.ndarray]] = {}  # TF -> {branch_role: scores}

    tf_home_map = {c["name"]: (c["home_branch"], c["away_branch"]) for c in configs}

    for tf in REGULATOR_DIAGNOSTIC_TFS:
        if tf not in tf_home_map:
            continue
        home_b, away_b = tf_home_map[tf]
        hist_series[tf] = {}
        for bname, role in [(home_b, "home"), (away_b, "away")]:
            key = (bname, analysis_label)
            if key not in score_mats:
                continue
            genes = score_mats[key]["genes"]
            S = score_mats[key]["S"]
            if tf not in genes.tolist():
                continue
            j = int(np.where(genes == tf)[0][0])
            vec = np.asarray(S[j, :], dtype=float)
            st = _signed_vector_stats(vec)
            row = {"TF": tf, "branch": bname, "branch_role": role, **st}
            rows_a.append(row)
            hist_series[tf][role] = vec

    pd.DataFrame(rows_a).to_csv(tabdir / "signed_score_distribution_row.csv", index=False)
    log(f"Saved {tabdir / 'signed_score_distribution_row.csv'}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, tf in zip(axes, REGULATOR_DIAGNOSTIC_TFS):
        if tf not in hist_series or not hist_series[tf]:
            ax.set_title(f"{tf}: (no data)")
            continue
        hm = hist_series[tf].get("home")
        aw = hist_series[tf].get("away")
        home_b, away_b = tf_home_map[tf]
        for v, color, lab in [
            (hm, "#2ca02c", f"{home_b.split('_')[0]} (home)"),
            (aw, "#1f77b4", f"{away_b.split('_')[0]} (away)"),
        ]:
            if v is None or v.size == 0:
                continue
            try:
                kde = scipy_stats.gaussian_kde(v)
                xs = np.linspace(float(np.min(v)), float(np.max(v)), 200)
                if float(np.max(v)) == float(np.min(v)):
                    xs = np.linspace(float(np.min(v)) - 0.1, float(np.max(v)) + 0.1, 50)
                ax.plot(xs, kde(xs), color=color, lw=2, label=lab)
            except Exception:
                ax.hist(v, bins=40, density=True, color=color, alpha=0.35, label=lab)
        ax.axvline(0.0, color="k", linestyle="--", lw=1.2)
        ax.set_xlabel("signed GeoBRIDGE score (row / regulator→targets)")
        ax.set_ylabel("density")
        ax.set_title(f"{tf}: signed scores (row)")
        ax.legend(frameon=False, fontsize=8)
        ax.grid(True, alpha=0.25)
    fig.tight_layout()
    p_hist = figdir / "signed_score_distribution_row.png"
    fig.savefig(p_hist, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {p_hist.resolve()}")

    # --- B: Ms / velocity / pt bins (Pax4 + Arx) ---
    bin_rows: list[dict] = []
    summary_expr: list[dict] = []
    pax4_branches = next(
        (c["home_branch"], c["away_branch"]) for c in configs if c.get("name") == "Pax4"
    )

    for tf in REGULATOR_DIAGNOSTIC_TFS:
        for bname, pack in branches.items():
            ad = pack.adata
            genes = np.array(ad.var_names, dtype=str)
            if tf not in genes.tolist():
                continue
            gi = int(np.where(genes == tf)[0][0])
            Ms = to_dense(ad.layers["Ms"]).astype(np.float64)
            Vel = to_dense(ad.layers["velocity"]).astype(np.float64)
            pt = np.asarray(ad.obs["pt_use"].values, dtype=np.float64)
            xg = Ms[:, gi]
            vg = Vel[:, gi]
            flat_p10 = float(np.percentile(Ms.ravel(), 10))
            summary_expr.append(
                {
                    "TF": tf,
                    "branch": bname,
                    "mean_Ms": float(np.mean(xg)),
                    "median_Ms": float(np.median(xg)),
                    "std_Ms": float(np.std(xg)),
                    "frac_Ms_below_1e-6": float(np.mean(xg < 1e-6)),
                    "frac_Ms_below_branch_p10_all_genes": float(np.mean(xg < flat_p10)),
                    "frac_Ms_nonzero": float(np.mean(xg > 0)),
                    "mean_velocity": float(np.mean(vg)),
                    "median_velocity": float(np.median(vg)),
                    "std_velocity": float(np.std(vg)),
                    "frac_velocity_near_zero": float(np.mean(np.abs(vg) < 1e-6)),
                }
            )

            try:
                qbins = pd.qcut(pt, q=20, duplicates="drop")
            except Exception:
                qbins = pd.cut(pt, bins=min(20, max(3, len(np.unique(pt)))))
            centers = []
            for lab in qbins.categories:
                mask = qbins == lab
                if not np.any(mask):
                    continue
                c0, c1 = float(np.min(pt[mask])), float(np.max(pt[mask]))
                bin_rows.append(
                    {
                        "TF": tf,
                        "branch": bname,
                        "bin_left": c0,
                        "bin_right": c1,
                        "bin_center": 0.5 * (c0 + c1),
                        "n_cells": int(np.sum(mask)),
                        "mean_Ms": float(np.mean(xg[mask])),
                        "median_Ms": float(np.median(xg[mask])),
                        "mean_velocity": float(np.mean(vg[mask])),
                    }
                )

    pd.DataFrame(summary_expr).to_csv(tabdir / "regulator_expression_velocity_by_branch.csv", index=False)
    pd.DataFrame(bin_rows).to_csv(tabdir / "pax4_expression_by_branch_and_bin.csv", index=False)
    log(f"Saved {tabdir / 'regulator_expression_velocity_by_branch.csv'} and pax4_expression_by_branch_and_bin.csv")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax2 = ax.twinx()
    for bname in pax4_branches:
        if bname not in branches:
            continue
        sub = pd.DataFrame([r for r in bin_rows if r["TF"] == "Pax4" and r["branch"] == bname])
        if sub.empty:
            continue
        sub = sub.sort_values("bin_center")
        short = bname.split("_")[0]
        ax.plot(sub["bin_center"], sub["mean_Ms"], marker="o", lw=2, label=f"{short} mean Ms")
        ax2.plot(sub["bin_center"], sub["mean_velocity"], marker="s", ls="--", lw=1.8, label=f"{short} mean velocity")
    ax.set_xlabel("pt_use bin center")
    ax.set_ylabel("mean Pax4 Ms")
    ax2.set_ylabel("mean Pax4 velocity")
    ax.set_title("Pax4 expression & velocity vs pseudotime (by branch)")
    ax.grid(True, alpha=0.25)
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, frameon=False, loc="best")
    fig.tight_layout()
    p_pt = figdir / "pax4_expression_velocity_vs_pseudotime.png"
    fig.savefig(p_pt, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {p_pt.resolve()}")

    # --- C: pairwise ΔX Pax4 only ---
    rows_c: list[dict] = []
    pax4_deltas: dict[str, np.ndarray] = {}
    for bname, pack in branches.items():
        ad = pack.adata
        genes = np.array(ad.var_names, dtype=str)
        if "Pax4" not in genes.tolist():
            continue
        gi = int(np.where(genes == "Pax4")[0][0])
        Xb = to_dense(ad.layers["Ms"]).astype(np.float64)
        tb = np.asarray(ad.obs["pt_use"].values, dtype=np.float64)
        d = collect_pairwise_delta_gene(
            Xb,
            tb,
            gi,
            pair_mode=str(args.pair_mode),
            min_forward_dt=float(args.min_dt),
            max_forward_dt=float(args.max_dt),
            max_pairs_per_cell=int(args.max_pairs_per_cell) if args.max_pairs_per_cell > 0 else None,
            global_max_pairs=int(args.global_max_pairs),
            pair_seed=int(args.seed),
        )
        pax4_deltas[bname] = d
        st = _delta_x_summary_stats(d)
        rows_c.append({"branch": bname, **st})
    pd.DataFrame(rows_c).to_csv(tabdir / "pax4_pairwise_delta_x_summary.csv", index=False)
    log(f"Saved {tabdir / 'pax4_pairwise_delta_x_summary.csv'}")

    branch_order = [b for b in pax4_branches if b in pax4_deltas]
    if len(branch_order) < len(pax4_branches):
        branch_order = list(pax4_deltas.keys())
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, bname in enumerate(branch_order[:2]):
        if bname not in pax4_deltas:
            continue
        d = pax4_deltas[bname]
        if d.size == 0:
            continue
        ax[i].hist(d, bins=60, color="#444444", alpha=0.85)
        ax[i].axvline(0.0, color="red", linestyle="--")
        ax[i].set_title(f"Pax4 ΔX signed ({bname.split('_')[0]})")
        ax[i].set_xlabel("ΔX_Pax4")
        ax[i].set_ylabel("count")
    fig.tight_layout()
    p_dx = figdir / "pax4_pairwise_delta_x_hist.png"
    fig.savefig(p_dx, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {p_dx.resolve()}")

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for i, bname in enumerate(branch_order[:2]):
        if bname not in pax4_deltas:
            continue
        ad = np.abs(pax4_deltas[bname])
        if ad.size == 0:
            continue
        ax[i].hist(ad, bins=60, color="#333333", alpha=0.85)
        ax[i].set_title(f"|ΔX_Pax4| ({bname.split('_')[0]})")
        ax[i].set_xlabel("|ΔX|")
        ax[i].set_ylabel("count")
    fig.tight_layout()
    p_abs = figdir / "pax4_pairwise_abs_delta_x_hist.png"
    fig.savefig(p_abs, dpi=220, bbox_inches="tight")
    plt.close(fig)
    log(f"Saved {p_abs.resolve()}")

    # Optional: mean |ΔX| vs source-cell pt bin
    bin_dx_rows: list[dict] = []
    for bname, pack in branches.items():
        if bname not in pax4_deltas or pax4_deltas[bname].size == 0:
            continue
        ad = pack.adata
        genes = np.array(ad.var_names, dtype=str)
        gi = int(np.where(genes == "Pax4")[0][0])
        Xb = to_dense(ad.layers["Ms"]).astype(np.float64)
        tb = np.asarray(ad.obs["pt_use"].values, dtype=np.float64)
        # Rebuild pairs with source index to aggregate |dX| by pt(i)
        from geobridge_model.pairspace import monotonic_forward_windows

        n = Xb.shape[0]
        rng = np.random.default_rng(int(args.seed))
        cap = None if args.max_pairs_per_cell <= 0 else int(args.max_pairs_per_cell)
        pair_pt: list[float] = []
        pair_adx: list[float] = []
        pt_arr = tb
        if str(args.pair_mode) == "global_sample":
            order = np.argsort(pt_arr)
            Xs = Xb[order]
            pts = pt_arr[order]
            inv_map = np.empty_like(order)
            inv_map[order] = np.arange(n)
            start, end = monotonic_forward_windows(pts, float(args.min_dt), float(args.max_dt))
            m = np.maximum((end - start).astype(np.int64), 0)
            total = int(m.sum())
            if total > 0:
                P = min(int(args.global_max_pairs), total)
                probs = m / float(total)
                i_samp = rng.choice(n, size=P, replace=True, p=probs)
                for ii in i_samp:
                    s, e = int(start[ii]), int(end[ii])
                    if s >= e:
                        continue
                    jj = int(rng.integers(s, e))
                    ddx = float(abs(Xs[jj, gi] - Xs[int(ii), gi]))
                    pair_pt.append(float(0.5 * (pts[int(ii)] + pts[jj])))
                    pair_adx.append(ddx)
        else:
            if np.all(pt_arr[1:] >= pt_arr[:-1]):
                st, en = monotonic_forward_windows(pt_arr, float(args.min_dt), float(args.max_dt))
                for ii in range(n - 1):
                    s, e = int(st[ii]), int(en[ii])
                    if s >= e:
                        continue
                    js = np.arange(s, e, dtype=np.int64)
                    if cap is not None and js.size > cap:
                        js = rng.choice(js, size=cap, replace=False)
                    for jj in js:
                        ddx = float(abs(Xb[int(jj), gi] - Xb[int(ii), gi]))
                        pair_pt.append(float(0.5 * (pt_arr[int(ii)] + pt_arr[int(jj)])))
                        pair_adx.append(ddx)
            else:
                for ii in range(n - 1):
                    dt = pt_arr[ii + 1 :] - pt_arr[ii]
                    valid = np.where(dt > float(args.min_dt))[0]
                    valid = valid[dt[valid] <= float(args.max_dt)]
                    if valid.size == 0:
                        continue
                    js = (ii + 1 + valid).astype(np.int64)
                    if cap is not None and js.size > cap:
                        js = rng.choice(js, size=cap, replace=False)
                    for jj in js:
                        ddx = float(abs(Xb[int(jj), gi] - Xb[int(ii), gi]))
                        pair_pt.append(float(0.5 * (pt_arr[int(ii)] + pt_arr[int(jj)])))
                        pair_adx.append(ddx)
        if pair_pt:
            pt_mid = np.asarray(pair_pt)
            adxv = np.asarray(pair_adx)
            try:
                qb = pd.qcut(pt_mid, q=20, duplicates="drop")
            except Exception:
                qb = pd.cut(pt_mid, bins=min(20, max(3, len(np.unique(pt_mid)))))
            for lab in qb.categories:
                msk = qb == lab
                if not np.any(msk):
                    continue
                c0, c1 = float(np.min(pt_mid[msk])), float(np.max(pt_mid[msk]))
                bin_dx_rows.append(
                    {
                        "branch": bname,
                        "bin_left": c0,
                        "bin_right": c1,
                        "bin_center": 0.5 * (c0 + c1),
                        "n_pairs": int(np.sum(msk)),
                        "mean_abs_delta_x": float(np.mean(adxv[msk])),
                    }
                )
    if bin_dx_rows:
        pd.DataFrame(bin_dx_rows).to_csv(tabdir / "pax4_pairwise_abs_delta_x_by_pt_bin.csv", index=False)
        fig, ax = plt.subplots(figsize=(7.5, 4))
        for bname in pax4_branches:
            sub = pd.DataFrame([r for r in bin_dx_rows if r["branch"] == bname])
            if sub.empty:
                continue
            sub = sub.sort_values("bin_center")
            ax.plot(
                sub["bin_center"],
                sub["mean_abs_delta_x"],
                marker="o",
                lw=2,
                label=bname.split("_")[0],
            )
        ax.set_xlabel("pair pseudotime mid (bin center)")
        ax.set_ylabel("mean |ΔX_Pax4|")
        ax.set_title("Pax4 |ΔX| vs pseudotime (pair mid)")
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.25)
        fig.tight_layout()
        fig.savefig(figdir / "pax4_pairwise_abs_delta_x_vs_pt_bin.png", dpi=220, bbox_inches="tight")
        plt.close(fig)
        log(f"Saved {figdir / 'pax4_pairwise_abs_delta_x_vs_pt_bin.png'}")

    # --- E: interpretation ---
    metrics: dict = {"rows_a": rows_a, "rows_c": rows_c, "summary_expr": summary_expr}
    pax4_away_branch = next(c["away_branch"] for c in configs if c.get("name") == "Pax4")
    _log_regulator_interpretation(log, metrics, tf_focus="Pax4", away_branch=pax4_away_branch)
    return metrics


def _log_regulator_interpretation(log, metrics: dict, *, tf_focus: str, away_branch: str) -> None:
    rows_a = metrics.get("rows_a", [])
    rows_c = metrics.get("rows_c", [])
    summ = metrics.get("summary_expr", [])

    def _pick(rows, **kw):
        for r in rows:
            if all(r.get(k) == v for k, v in kw.items()):
                return r
        return None

    away_pax = _pick(rows_a, TF=tf_focus, branch=away_branch)
    away_c = _pick(rows_c, branch=away_branch)
    away_expr = _pick(summ, TF=tf_focus, branch=away_branch)

    log("=== Pax4 artifact diagnostic interpretation (heuristic) ===")
    if away_pax:
        fn = away_pax.get("frac_negative", float("nan"))
        fp = away_pax.get("frac_positive", float("nan"))
        log(
            f"  [{tf_focus} @ {away_branch} row scores] frac_negative={fn:.4f} frac_positive={fp:.4f} "
            f"median_signed={away_pax.get('median_signed')} median_abs={away_pax.get('median_abs')}"
        )
    else:
        log(f"  [{tf_focus} @ {away_branch}] no signed-score row (missing TF or branch).")

    if away_expr:
        log(
            f"  [{tf_focus} @ {away_branch} Ms] mean={away_expr.get('mean_Ms'):.4g} "
            f"frac_below_1e-6={away_expr.get('frac_Ms_below_1e-6'):.4f} "
            f"frac_below_branch_p10={away_expr.get('frac_Ms_below_branch_p10_all_genes'):.4f}"
        )
    if away_c:
        log(
            f"  [{away_branch} Pax4 ΔX pairs] median|ΔX|={away_c.get('median_abs_delta_x'):.4g} "
            f"frac_abs_dx_below_1e-3={away_c.get('frac_abs_delta_x_below_1e-3'):.4f}"
        )

    # crude verdict
    rep = 0
    art = 0
    if away_pax and away_pax.get("frac_negative", 0) > 0.45 and away_pax.get("median_signed", 0) < 0:
        rep += 1
    if away_expr and (away_expr.get("mean_Ms", 1) < 0.05 or away_expr.get("frac_Ms_below_1e-6", 0) > 0.3):
        art += 1
    if away_c and away_c.get("median_abs_delta_x", 1) < 1e-3:
        art += 1

    if rep >= 2 and art == 0:
        verdict = "repression-consistent signal (weak evidence; confirm biologically)"
    elif art >= 2 and rep == 0:
        verdict = "low-expression / small-ΔX artifact concern (weak evidence)"
    elif art >= 1 and rep >= 1:
        verdict = "mixed / inconclusive"
    else:
        verdict = "inconclusive from automated heuristics — inspect CSVs and plots"

    log(f"  Automated verdict (not ground truth): {verdict}")
    log("=== End Pax4 artifact diagnostic interpretation ===")


def run_low_expression_mask_row_diagnostics(
    *,
    branches: dict,
    score_mats: dict,
    analysis_label: str,
    configs: list,
    args,
    figdir: Path,
    tabdir: Path,
    log,
    k_rule,
    mask_threshold: float,
) -> None:
    """D: zero outgoing row for regulators with branch-mean Ms below threshold. Diagnostic only."""
    figdir = Path(figdir)
    tabdir = Path(tabdir)
    mask_threshold = float(mask_threshold)

    score_mats_masked: dict = {}
    for bname, pack in branches.items():
        key = (bname, analysis_label)
        if key not in score_mats:
            continue
        genes = score_mats[key]["genes"]
        S = np.array(score_mats[key]["S"], dtype=float, copy=True)
        Xb = to_dense(pack.adata.layers["Ms"]).astype(np.float64)
        mean_g = Xb.mean(axis=0)
        low = mean_g < mask_threshold
        S[low, :] = 0.0
        score_mats_masked[key] = {
            "S": S,
            "genes": genes,
            "k_selected": score_mats[key].get("k_selected"),
            "n_pairs": score_mats[key].get("n_pairs"),
        }

    sc_row_base = build_pancreas_tf_score_cache(score_mats, configs, analysis_label, "row")
    sc_row_mask = build_pancreas_tf_score_cache(score_mats_masked, configs, analysis_label, "row")

    # Signed distribution: baseline + masked
    rows_b: list[dict] = []
    rows_m: list[dict] = []
    tf_home_map = {c["name"]: (c["home_branch"], c["away_branch"]) for c in configs}
    for tf in REGULATOR_DIAGNOSTIC_TFS:
        if tf not in tf_home_map:
            continue
        home_b, away_b = tf_home_map[tf]
        for bname, role in [(home_b, "home"), (away_b, "away")]:
            key = (bname, analysis_label)
            if key not in score_mats:
                continue
            genes = score_mats[key]["genes"]
            j = int(np.where(genes == tf)[0][0])
            vb = np.asarray(score_mats[key]["S"][j, :], dtype=float)
            vm = np.asarray(score_mats_masked[key]["S"][j, :], dtype=float)
            rb = _signed_vector_stats(vb)
            rm = _signed_vector_stats(vm)
            rows_b.append({"mask": "unmasked", "TF": tf, "branch": bname, "branch_role": role, **rb})
            rows_m.append({"mask": "masked", "TF": tf, "branch": bname, "branch_role": role, **rm})
    pd.DataFrame(rows_b + rows_m).to_csv(tabdir / "signed_score_distribution_row__masked_compare.csv", index=False)

    import matplotlib.pyplot as plt
    from scipy import stats as scipy_stats

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, tf in zip(axes, REGULATOR_DIAGNOSTIC_TFS):
        home_b, away_b = tf_home_map[tf]
        for v, color, lab in [
            (np.asarray(score_mats[(home_b, analysis_label)]["S"][int(np.where(score_mats[(home_b, analysis_label)]["genes"]==tf)[0][0]), :]), "#2ca02c", "home unmasked"),
            (np.asarray(score_mats_masked[(home_b, analysis_label)]["S"][int(np.where(score_mats_masked[(home_b, analysis_label)]["genes"]==tf)[0][0]), :]), "#98df8a", "home masked"),
            (np.asarray(score_mats[(away_b, analysis_label)]["S"][int(np.where(score_mats[(away_b, analysis_label)]["genes"]==tf)[0][0]), :]), "#1f77b4", "away unmasked"),
            (np.asarray(score_mats_masked[(away_b, analysis_label)]["S"][int(np.where(score_mats_masked[(away_b, analysis_label)]["genes"]==tf)[0][0]), :]), "#9ecae1", "away masked"),
        ]:
            if v.size == 0:
                continue
            try:
                kde = scipy_stats.gaussian_kde(v)
                xs = np.linspace(float(np.min(v)), float(np.max(v)), 200)
                if float(np.max(v)) == float(np.min(v)):
                    xs = np.linspace(float(np.min(v)) - 0.1, float(np.max(v)) + 0.1, 50)
                ax.plot(xs, kde(xs), color=color, lw=1.6, label=lab)
            except Exception:
                ax.hist(v, bins=30, density=True, alpha=0.25, label=lab)
        ax.axvline(0.0, color="k", linestyle="--", lw=1)
        ax.set_title(f"{tf} row signed (mask thr={mask_threshold})")
        ax.legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(figdir / "signed_score_distribution_row__masked.png", dpi=220, bbox_inches="tight")
    plt.close(fig)

    topn_home_vs_away(
        tabdir / "topn_home_vs_away_summary__row_baseline.csv",
        figdir / "topn_home_vs_away_boxplots__row_baseline.png",
        sc_row_base,
        configs,
        top_list=[50, 100, 200, 300],
        analysis_label=analysis_label,
        log_fn=log,
        score_orientation="row_baseline",
    )
    topn_home_vs_away(
        tabdir / "topn_home_vs_away_summary__masked.csv",
        figdir / "topn_home_vs_away_boxplots__masked.png",
        sc_row_mask,
        configs,
        top_list=[50, 100, 200, 300],
        analysis_label=analysis_label,
        log_fn=log,
        score_orientation="row_masked",
    )

    wbase: list[dict] = []
    wmask: list[dict] = []
    r = plot_branch_specificity(
        "Pax4",
        "Beta_branch",
        "Alpha_branch",
        sc_row_base,
        figdir / "branch_specificity_Pax4__row_baseline.png",
        log_fn=log,
        analysis_label=analysis_label,
        score_orientation="row_baseline",
    )
    r["k_selected_rule"] = k_rule
    r["pair_mode"] = str(args.pair_mode)
    r["mask_threshold"] = mask_threshold
    wbase.append(r)
    r = plot_branch_specificity(
        "Arx",
        "Alpha_branch",
        "Beta_branch",
        sc_row_base,
        figdir / "branch_specificity_Arx__row_baseline.png",
        log_fn=log,
        analysis_label=analysis_label,
        score_orientation="row_baseline",
    )
    r["k_selected_rule"] = k_rule
    r["pair_mode"] = str(args.pair_mode)
    r["mask_threshold"] = mask_threshold
    wbase.append(r)
    pd.DataFrame(wbase).to_csv(tabdir / "branch_specificity_wilcoxon_top200__row_baseline.csv", index=False)

    r = plot_branch_specificity(
        "Pax4",
        "Beta_branch",
        "Alpha_branch",
        sc_row_mask,
        figdir / "branch_specificity_Pax4__masked.png",
        log_fn=log,
        analysis_label=analysis_label,
        score_orientation="row_masked",
    )
    r["k_selected_rule"] = k_rule
    r["pair_mode"] = str(args.pair_mode)
    r["mask_threshold"] = mask_threshold
    wmask.append(r)
    r = plot_branch_specificity(
        "Arx",
        "Alpha_branch",
        "Beta_branch",
        sc_row_mask,
        figdir / "branch_specificity_Arx__masked.png",
        log_fn=log,
        analysis_label=analysis_label,
        score_orientation="row_masked",
    )
    r["k_selected_rule"] = k_rule
    r["pair_mode"] = str(args.pair_mode)
    r["mask_threshold"] = mask_threshold
    wmask.append(r)
    pd.DataFrame(wmask).to_csv(tabdir / "branch_specificity_wilcoxon_top200__masked.csv", index=False)

    log(
        f"Low-expression mask diagnostic: threshold={mask_threshold} (branch mean Ms); "
        f"see tables/*__masked* and *__row_baseline*."
    )

    # Mask-specific interpretation (Pax4 on its configured "away" branch)
    pax4_away = next(c["away_branch"] for c in configs if c.get("name") == "Pax4")
    log(f"=== Low-expression mask diagnostic (Pax4 away row, branch={pax4_away}) ===")
    key_away = (pax4_away, analysis_label)
    if key_away not in score_mats or "Pax4" not in score_mats[key_away]["genes"].tolist():
        log("  (skip: Pax4 or away branch missing from score_mats)")
    else:
        j_al = int(np.where(score_mats[key_away]["genes"] == "Pax4")[0][0])
        v0 = np.asarray(score_mats[key_away]["S"][j_al, :], dtype=float)
        v1 = np.asarray(score_mats_masked[key_away]["S"][j_al, :], dtype=float)
        log(
            f"  Pax4 row @ {pax4_away}: median|score| unmasked={float(np.median(np.abs(v0))):.4g} "
            f"masked={float(np.median(np.abs(v1))):.4g}"
        )
    log("=== End mask diagnostic ===")


def save_dt_stats_and_hist(
    t: np.ndarray,
    *,
    min_dt: float,
    max_dt: float,
    max_pairs_per_cell: int | None,
    pair_seed: int,
    out_json: Path,
    out_png: Path,
):
    """
    Compute an approximate dt distribution by sampling admissible forward pairs (using the same
    per-cell cap logic as streaming) and save summary + histogram.
    """
    import matplotlib.pyplot as plt

    from geobridge_model.pairspace import iter_forward_pair_blocks

    pt = np.asarray(t, dtype=np.float64).reshape(-1)
    rng = np.random.default_rng(int(pair_seed))
    cap = None if max_pairs_per_cell is None else int(max_pairs_per_cell)
    dts = []
    for i, js in iter_forward_pair_blocks(
        pt, min_forward_dt=float(min_dt), max_forward_dt=float(max_dt), max_pairs_per_cell=cap, pair_seed=int(pair_seed)
    ):
        if js.size == 0:
            continue
        # sample at most 8 per i for stats so it stays fast
        if js.size > 8:
            js = rng.choice(js, size=8, replace=False)
        dts.extend((pt[js] - pt[i]).tolist())
    if len(dts) == 0:
        stats = {"n": 0}
        out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        return stats

    dts = np.asarray(dts, dtype=np.float64)
    qs = np.percentile(dts, [10, 25, 50, 75, 90, 95, 100]).tolist()
    stats = {
        "n": int(dts.size),
        "mean": float(dts.mean()),
        "median": float(np.median(dts)),
        "p10": float(qs[0]),
        "p25": float(qs[1]),
        "p50": float(qs[2]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
        "p95": float(qs[5]),
        "max": float(qs[6]),
        "min_dt": float(min_dt),
        "max_dt": float(max_dt),
        "max_pairs_per_cell": None if max_pairs_per_cell is None else int(max_pairs_per_cell),
    }
    out_json.write_text(json.dumps(stats, indent=2), encoding="utf-8")

    fig = plt.figure(figsize=(6.2, 4.2))
    plt.hist(dts, bins=40, color="#333333", alpha=0.85)
    plt.axvline(float(min_dt), color="red", linestyle="--", linewidth=1.2)
    plt.axvline(float(max_dt), color="red", linestyle="--", linewidth=1.2)
    plt.xlabel("Δt (forward pseudotime)")
    plt.ylabel("count")
    plt.title("Valid forward-pair Δt histogram (sampled)")
    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)
    return stats


def main():
    _set_matplotlib_backend()
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", type=Path, default=Path("real_1_Pancreas"), help="Output directory")
    ap.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="Short label for this run (defaults to outdir folder name; stored in summary CSV/manifest).",
    )
    ap.add_argument("--realdata-preset", choices=["default", "wider_dt"], default="default")
    ap.add_argument("--min-dt", type=float, default=None, help="Min forward Δt for GeoBRIDGE pairs")
    ap.add_argument("--max-dt", type=float, default=None, help="Max forward Δt for GeoBRIDGE pairs")
    ap.add_argument(
        "--max-pairs-per-cell",
        type=int,
        default=40,
        help="Cap forward pairs per cell for GeoBRIDGE (prevents O(n^2) blowups).",
    )
    ap.add_argument(
        "--stream-num-threads",
        type=int,
        default=1,
        help="Threads for streaming pair accumulation (1 is safest on Windows/BLAS).",
    )
    ap.add_argument(
        "--pair-mode",
        choices=["stream_per_cell", "global_sample"],
        default="stream_per_cell",
        help="Pair generation mode for GeoBRIDGE.",
    )
    ap.add_argument("--global-max-pairs", type=int, default=30000, help="For pair-mode=global_sample.")
    ap.add_argument("--max-k", type=int, default=40)
    ap.add_argument("--tau", type=float, default=2.5)
    ap.add_argument("--tail-fraction", type=float, default=0.25)
    ap.add_argument("--tail-min-start", type=int, default=15)
    ap.add_argument("--k-override", type=int, default=None, help="If set, compute an additional fixed-k run.")
    ap.add_argument(
        "--k-clamp",
        type=int,
        default=10,
        help="Clamp k_used = min(k_auto or k_override, k_clamp). Set <=0 to disable clamping.",
    )
    ap.add_argument(
        "--analysis-label",
        type=str,
        default="k_auto",
        metavar="LABEL",
        help=(
            "Which GeoBRIDGE run label to use for score_cache, top-N, and branch-specificity plots "
            '(must match a computed label: default "k_auto", or "k_<int>" if --k-override was set).'
        ),
    )
    ap.add_argument(
        "--debug-dt",
        action="store_true",
        help="Print Δt diagnostics (mean/median/p90/p95/max) per branch before scoring.",
    )
    ap.add_argument(
        "--debug-projection",
        action="store_true",
        help="Print projection diagnostics (variance retained, num sign cancellation) per branch.",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--k-grid-max", type=int, default=5, help="For Neurog3 stability plot, test k=1..K")
    ap.add_argument("--skip-enrichr", action="store_true", help="Skip gseapy Enrichr calls/plot")
    ap.add_argument(
        "--skip-dynamics",
        action="store_true",
        help="Skip scVelo recover_dynamics/latent_time (much faster; uses DPT pseudotime on branches instead).",
    )
    ap.add_argument("--force-dpt", action="store_true")
    ap.add_argument(
        "--force-latent-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Force using scVelo latent_time for pseudotime (default True for real-data reproduction).",
    )
    ap.add_argument(
        "--allow-dpt-fallback",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Allow falling back to DPT pseudotime when latent_time is missing (default False).",
    )
    ap.add_argument(
        "--orientation-check",
        action="store_true",
        help="Save top-20 targets for Pax4/Arx using both column and row orientation.",
    )
    ap.add_argument(
        "--orientation-verify",
        action="store_true",
        help=(
            "Save full col/row TF score vectors, duplicate top-N + branch-specificity + Wilcoxon with __col/__row "
            "suffixes. Default pipeline outputs are unchanged; this adds side-by-side orientation comparison."
        ),
    )
    ap.add_argument(
        "--pax4-artifact-diagnostics",
        action="store_true",
        help=(
            "Optional Pax4/regulator artifact diagnostics: signed row-score distributions, Ms/velocity/pt bins, "
            "pairwise ΔX for Pax4, and a short interpretation log. Writes only under figures/ and tables/."
        ),
    )
    ap.add_argument(
        "--low-expression-mask-diagnostic",
        action="store_true",
        help=(
            "Diagnostic only: copy score matrices with rows zeroed when branch mean Ms < --mask-threshold; "
            "save row-oriented top-N, branch specificity, Wilcoxon, and signed distributions with __masked / "
            "__row_baseline suffixes. Does not change default pipeline outputs."
        ),
    )
    ap.add_argument(
        "--mask-threshold",
        type=float,
        default=0.01,
        help="Branch mean Ms threshold for --low-expression-mask-diagnostic (default 0.01).",
    )
    ap.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse cached processed AnnData (skips expensive scVelo steps).",
    )
    ap.add_argument(
        "--cache-dir",
        type=Path,
        default=None,
        help="Cache directory (default: <outdir>/cache).",
    )
    ap.add_argument(
        "--cache-tag",
        type=str,
        default="default",
        help="Cache filename tag (lets you keep multiple preprocessing variants).",
    )
    args = ap.parse_args()

    presets = {
        "default": (0.02, 0.25),
        "wider_dt": (0.05, 0.35),
    }
    if args.min_dt is None or args.max_dt is None:
        p_lo, p_hi = presets[args.realdata_preset]
        if args.min_dt is None:
            args.min_dt = p_lo
        if args.max_dt is None:
            args.max_dt = p_hi
    args.min_dt = float(args.min_dt)
    args.max_dt = float(args.max_dt)
    if args.max_dt <= args.min_dt:
        raise ValueError("--max-dt must be > --min-dt")

    outdir: Path = args.outdir
    run_name = args.run_name if args.run_name else outdir.name
    figdir = outdir / "figures"
    tabdir = outdir / "tables"
    figdir.mkdir(parents=True, exist_ok=True)
    tabdir.mkdir(parents=True, exist_ok=True)
    cachedir = (outdir / "cache") if args.cache_dir is None else Path(args.cache_dir)
    cachedir.mkdir(parents=True, exist_ok=True)
    cache_h5ad = cachedir / f"pancreas_processed__{args.cache_tag}.h5ad"
    log_path = outdir / "run_log.txt"

    def log(msg: str) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{ts}] {msg}"
        print(line, flush=True)
        try:
            log_path.open("a", encoding="utf-8").write(line + "\n")
        except Exception:
            pass

    import scanpy as sc
    import scvelo as scv
    import warnings

    scv.settings.verbosity = 0
    np.random.seed(args.seed)
    # scVelo emits some numpy deprecation warnings on recent numpy; safe to ignore.
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    # Load pancreas data (scVelo built-in) OR reuse cached processed AnnData.
    if args.use_cache and cache_h5ad.exists():
        log(f"Loading cached AnnData: {cache_h5ad}")
        adata = sc.read_h5ad(cache_h5ad)
        log(f"Loaded cached adata shape={adata.shape}.")
        recover_ok = bool(adata.uns.get("geobridge__recover_dynamics_succeeded", False))
        latent_ok = bool("latent_time" in adata.obs.columns)
        log(
            f"(cache) recover_dynamics_succeeded={recover_ok} | latent_time_exists={latent_ok} | "
            f"final_pt_source=per-branch | force_latent_time={bool(args.force_latent_time)} | "
            f"allow_dpt_fallback={bool(args.allow_dpt_fallback)}"
        )
    else:
        log("Loading scVelo pancreas dataset…")
        adata = scv.datasets.pancreas()
        log(f"Loaded adata shape={adata.shape}.")

        # scVelo pipeline (notebook-equivalent, but version-robust).
        # Some scvelo versions don't accept `n_top_genes` in scv.pp.filter_and_normalize (it gets passed
        # down into normalize_per_cell). So we fall back to scanpy HVG selection when needed.
        try:
            log("Preprocess: scv.pp.filter_and_normalize (with n_top_genes=2000)…")
            scv.pp.filter_and_normalize(adata, min_shared_counts=20, n_top_genes=2000)
        except TypeError:
            log("Preprocess fallback: scv.pp.filter_and_normalize (no n_top_genes) + scanpy HVGs…")
            scv.pp.filter_and_normalize(adata, min_shared_counts=20)
            # `seurat_v3` needs `scikit-misc` (skmisc). Fall back to other flavors if missing.
            hvg_ok = False
            for flavor in ("seurat_v3", "seurat", "cell_ranger"):
                try:
                    sc.pp.highly_variable_genes(adata, n_top_genes=2000, flavor=flavor)
                    hvg_ok = True
                    break
                except ImportError:
                    continue
            if hvg_ok and "highly_variable" in adata.var:
                adata = adata[:, adata.var["highly_variable"].values].copy()
                log(f"HVG subset applied. New shape={adata.shape}.")
            else:
                log("HVG selection skipped (no supported flavor available); continuing without subsetting.")

        log("Compute PCA/neighbors/moments/velocity/velocity_graph…")
        sc.tl.pca(adata, n_comps=50, random_state=args.seed)
        sc.pp.neighbors(adata, n_neighbors=30, n_pcs=50)
        scv.pp.moments(adata, n_pcs=50, n_neighbors=30)
        scv.tl.velocity(adata, mode="stochastic")
        scv.tl.velocity_graph(adata)

        recover_ok = False
        latent_ok = bool("latent_time" in adata.obs.columns)
        if args.skip_dynamics:
            if bool(args.force_latent_time) and not bool(args.allow_dpt_fallback):
                raise RuntimeError(
                    "--skip-dynamics was set but --force-latent-time is active (and --allow-dpt-fallback is false). "
                    "Either drop --skip-dynamics or pass --allow-dpt-fallback (or --no-force-latent-time)."
                )
            log("Skipping recover_dynamics/latent_time due to --skip-dynamics.")
        else:
            log("Running recover_dynamics (can be slow)…")
            try:
                scv.tl.recover_dynamics(adata, n_jobs=1)
                recover_ok = True
            except Exception as e:
                recover_ok = False
                log(f"recover_dynamics failed: {type(e).__name__}: {e}")

            log("Running latent_time…")
            try:
                scv.tl.latent_time(adata)
            except Exception as e:
                log(f"latent_time failed: {type(e).__name__}: {e}")
            latent_ok = bool("latent_time" in adata.obs.columns)

            if bool(args.force_latent_time) and (not recover_ok or not latent_ok) and not bool(args.allow_dpt_fallback):
                raise RuntimeError(
                    "Real-data reproduction requires latent_time. "
                    f"recover_dynamics_succeeded={recover_ok}, latent_time_exists={latent_ok}. "
                    "Refusing to fall back to DPT (allow with --allow-dpt-fallback or disable with --no-force-latent-time)."
                )

        log(
            f"recover_dynamics_succeeded={recover_ok} | latent_time_exists={latent_ok} | "
            f"final_pt_source=per-branch | force_latent_time={bool(args.force_latent_time)} | "
            f"allow_dpt_fallback={bool(args.allow_dpt_fallback)}"
        )

        adata.uns["geobridge__recover_dynamics_succeeded"] = bool(recover_ok)
        adata.uns["geobridge__latent_time_exists"] = bool(latent_ok)
        adata.uns["geobridge__cache_tag"] = str(args.cache_tag)

        if args.use_cache:
            log(f"Saving cached AnnData → {cache_h5ad}")
            adata.write_h5ad(cache_h5ad, compression="gzip")

    log("Computing covariates (total_counts, pct_counts_mt, cell cycle)…")
    compute_covariates(adata)

    # UMAP + velocity stream
    log("Saving UMAP + velocity stream plot…")
    save_umap_velocity(adata, figdir / "umap_velocity_stream.png")

    # Branch packs
    log("Preparing branch subsets + pseudotime…")
    if args.force_dpt and args.force_latent_time:
        raise ValueError("Use only one of --force-dpt or --force-latent-time.")
    branches = make_branches(
        adata,
        seed=args.seed,
        force_dpt=bool(args.force_dpt),
        force_latent_time=bool(args.force_latent_time),
        allow_dpt_fallback=bool(args.allow_dpt_fallback),
    )
    (tabdir / "branches.json").write_text(
        json.dumps(
            {
                k: {"n_cells": int(v.adata.n_obs), "pt_key": v.pt_key, "pt_source": v.pt_source}
                for k, v in branches.items()
            },
            indent=2,
        )
    )
    log(f"Branches: { {k: {'n_cells': int(v.adata.n_obs), 'pt_source': v.pt_source} for k,v in branches.items()} }")

    # Per-branch GeoBRIDGE score matrices
    # Use Ms as X and velocity layer as V (closest to notebook intent).
    if args.k_clamp is not None:
        log("[WARN] --k-clamp is deprecated and ignored (k is now selected by projection diagnostics).")

    from geobridge_model.config import GeoBridgeConfig
    from geobridge_model.model import GeoBridgeModel
    from geobridge_model.spectral_k import select_k_by_tail_exhaustion

    cfg = GeoBridgeConfig(
        min_forward_dt=float(args.min_dt),
        max_forward_dt=float(args.max_dt),
        max_pairs_per_cell=int(args.max_pairs_per_cell) if args.max_pairs_per_cell > 0 else None,
        pair_seed=int(args.seed),
        max_k=int(args.max_k),
        tail_fraction=float(args.tail_fraction),
        tail_min_start=int(args.tail_min_start),
        slope_window=3,
        stable_m=3,
        tau=float(args.tau),
        project_y=True,
        normalize_score=True,
    )

    score_mats: dict[str, dict] = {}
    summaries = []
    pre_by_branch: dict[str, dict] = {}
    k_auto_raw_by_branch: dict[str, int] = {}
    n_pairs_by_branch: dict[str, int] = {}

    # 1) Precompute streaming stats + k_auto_raw per branch (no scoring yet)
    for bname, pack in branches.items():
        ad = pack.adata
        genes = np.array(ad.var_names, dtype=str)
        X = to_dense(ad.layers["Ms"]).astype(np.float64)
        V = to_dense(ad.layers["velocity"]).astype(np.float64)
        t = np.asarray(ad.obs["pt_use"].values, dtype=np.float64)

        # Save dt distribution before pair construction.
        dt_stats_branch = save_dt_stats_and_hist(
            t,
            min_dt=args.min_dt,
            max_dt=args.max_dt,
            max_pairs_per_cell=int(args.max_pairs_per_cell) if args.max_pairs_per_cell > 0 else None,
            pair_seed=int(args.seed),
            out_json=tabdir / f"dt_stats_{bname}.json",
            out_png=figdir / f"dt_hist_{bname}.png",
        )
        if args.debug_dt:
            try:
                dt_stats = json.loads((tabdir / f"dt_stats_{bname}.json").read_text(encoding="utf-8"))
                log("[DEBUG] Δt stats:")
                log(f"  mean:   {dt_stats.get('mean')}")
                log(f"  median: {dt_stats.get('median')}")
                log(f"  p90:    {dt_stats.get('p90')}")
                log(f"  p95:    {dt_stats.get('p95')}")
                log(f"  max:    {dt_stats.get('max')}")
                if float(dt_stats.get("p90", 0.0)) > 0.15:
                    log("  [WARN] p90 > 0.15 (too large)")
                if float(dt_stats.get("p95", 0.0)) > 0.2:
                    log("  [WARN] p95 > 0.20 (very likely too large)")
            except Exception as e:
                log(f"[DEBUG] Δt stats read failed: {e}")

        # Precompute pairspace stats for scoring + projection diagnostics
        if str(args.pair_mode) == "stream_per_cell":
            pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
                X,
                V,
                t,
                min_forward_dt=cfg.min_forward_dt,
                max_forward_dt=cfg.max_forward_dt,
                max_pairs_per_cell=cfg.max_pairs_per_cell,
                pair_seed=cfg.pair_seed,
                stream_num_threads=int(args.stream_num_threads) if args.stream_num_threads > 0 else None,
            )
        elif str(args.pair_mode) == "global_sample":
            pre = precompute_pairspace_stats_global_sample(
                X,
                V,
                t,
                min_forward_dt=cfg.min_forward_dt,
                max_forward_dt=cfg.max_forward_dt,
                global_max_pairs=int(args.global_max_pairs),
                pair_seed=int(cfg.pair_seed or 42),
            )
        else:
            raise ValueError(f"Unknown pair_mode={args.pair_mode!r}")

        spec = select_k_by_tail_exhaustion(
            pre["S"],
            cfg.max_k,
            cfg.tail_fraction,
            cfg.tail_min_start,
            cfg.slope_window,
            cfg.stable_m,
            cfg.tau,
        )
        k_auto_raw = int(spec.k_selected)

        pre_by_branch[bname] = pre
        k_auto_raw_by_branch[bname] = int(k_auto_raw)
        n_pairs_by_branch[bname] = int(pre["n_pairs"])

        log(f"[DEBUG] branch={bname} k_auto_raw={k_auto_raw} n_pairs={int(pre['n_pairs'])}")

    # 2) Choose a single k_selected for this run using ONLY projection diagnostics (no biology)
    debug_plot = None
    if bool(args.debug_projection):
        debug_plot = figdir / "k_selection_projection_diagnostics.png"
    k_selected, k_table = select_k_by_projection_rule(
        pre_by_branch,
        k_auto_raw_by_branch,
        K_cap=15,
        r_min=0.05,
        asym_max=3.0,
        project_y=bool(cfg.project_y),
        log_fn=log if bool(args.debug_projection) else None,
        debug_plot_path=debug_plot,
    )
    log(f"[DEBUG] k selection: k_auto_raw_by_branch={k_auto_raw_by_branch} -> k_selected={k_selected}")

    # 3) Score each branch using k_selected (and optionally a fixed-k override run for comparison)
    for bname, pack in branches.items():
        ad = pack.adata
        genes = np.array(ad.var_names, dtype=str)
        pre = pre_by_branch[bname]
        n_pairs = int(n_pairs_by_branch[bname])
        k_auto_raw = int(k_auto_raw_by_branch[bname])

        # reuse dt stats for the summary row
        try:
            dt_stats_branch = json.loads((tabdir / f"dt_stats_{bname}.json").read_text(encoding="utf-8"))
        except Exception:
            dt_stats_branch = None
        dt_ok = bool(dt_stats_branch and int(dt_stats_branch.get("n", 0)) > 0)

        def _score_from_pre(label: str, k_use: int):
            log(f"GeoBRIDGE: scoring branch={bname} label={label} k_used={k_use} (cells={ad.n_obs}, genes={ad.n_vars})…")
            diag = projection_diagnostics_from_streaming(pre, int(k_use), project_y=bool(cfg.project_y)) if bool(args.debug_projection) else None
            if diag is not None:
                diag["k_auto_raw"] = int(k_auto_raw)
                diag["k_used"] = int(k_use)
                log(f"[DEBUG] k_auto_raw={diag.get('k_auto_raw')} k_used={diag.get('k_used')}")
                log(f"[DEBUG] variance retained: {diag.get('variance_retained_trace_ratio')}")
                log(f"[DEBUG] num/den proxy (sum/abs): {diag.get('num_sum_over_abs')}")

            S = GeoBridgeModel.scores_from_streaming_stats(
                pre,
                k=int(k_use),
                project_y=bool(cfg.project_y),
                normalize_score=bool(cfg.normalize_score),
                eps=cfg.eps,
            )
            np.fill_diagonal(S, 0.0)
            save_top_edges_graph(S, genes, figdir / f"graph_top_edges_{bname}__{label}.png", top_m=250)

            # aggregate selection metrics at the chosen k (repeat per branch row for convenience)
            row_k = next((r for r in k_table if int(r["k"]) == int(k_selected)), None)
            summ = {
                "run_name": str(run_name),
                "branch": bname,
                "label": label,
                "n_cells": int(ad.n_obs),
                "n_pairs": int(n_pairs),
                "k_auto_raw": int(k_auto_raw),
                "k_used": int(k_use),
                "k_selected_rule": int(k_selected),
                "pt_source": pack.pt_source,
                "min_dt": float(args.min_dt),
                "max_dt": float(args.max_dt),
                "max_pairs_per_cell": None if args.max_pairs_per_cell <= 0 else int(args.max_pairs_per_cell),
                "pair_mode": str(args.pair_mode),
                "global_max_pairs": int(args.global_max_pairs),
                "variance_retained_trace_ratio": float(diag["variance_retained_trace_ratio"])
                if diag
                else float("nan"),
                "num_sum_over_abs": float(diag["num_sum_over_abs"]) if diag else float("nan"),
                "rule_min_retained_variance": float(row_k["min_retained_variance"]) if row_k else float("nan"),
                "rule_mean_retained_variance": float(row_k["mean_retained_variance"]) if row_k else float("nan"),
                "rule_branch_variance_asymmetry": float(row_k["branch_variance_asymmetry"]) if row_k else float("nan"),
                "rule_mean_num_proxy": float(row_k["mean_num_proxy"]) if row_k else float("nan"),
                "dt_mean": float(dt_stats_branch["mean"]) if dt_ok else float("nan"),
                "dt_median": float(dt_stats_branch["median"]) if dt_ok else float("nan"),
                "dt_p90": float(dt_stats_branch["p90"]) if dt_ok else float("nan"),
                "dt_p95": float(dt_stats_branch["p95"]) if dt_ok else float("nan"),
            }
            summaries.append(summ)
            log(
                "GeoBRIDGE summary: "
                f"branch={bname} label={label} n_cells={summ['n_cells']} n_pairs={summ['n_pairs']} "
                f"k_auto_raw={summ['k_auto_raw']} k_used={summ['k_used']} k_selected_rule={summ['k_selected_rule']} "
                f"pt_source={summ['pt_source']} dt=({summ['min_dt']},{summ['max_dt']})"
            )
            return S

        # Main run: label is kept as "k_auto" for backward compatibility, but k_used comes from the projection rule.
        S_main = _score_from_pre("k_auto", int(k_selected))
        score_mats[(bname, "k_auto")] = {
            "S": S_main,
            "k_selected": int(k_selected),
            "genes": genes,
            "n_pairs": n_pairs,
        }

        # Optional additional fixed-k run for side-by-side comparison (does not affect k selection)
        if args.k_override is not None:
            k_fix = int(args.k_override)
            label = f"k_{k_fix}"
            S_fix = _score_from_pre(label, k_fix)
            score_mats[(bname, label)] = {
                "S": S_fix,
                "k_selected": int(k_fix),
                "genes": genes,
                "n_pairs": n_pairs,
            }

    analysis_label = str(args.analysis_label).strip()
    available = sorted({lab for (_b, lab) in score_mats.keys()})
    missing = [b for b in branches if (b, analysis_label) not in score_mats]
    if missing:
        raise ValueError(
            f"--analysis-label {analysis_label!r} missing score matrices for branches {missing}. "
            f"Available labels this run: {available}. "
            f'Use e.g. "k_auto" for the auto-k run, or "k_<n>" after --k-override n.'
        )

    # Branch-level summary table (print + log)
    df_sum = pd.DataFrame(summaries)
    log("Branch GeoBRIDGE summary table:")
    log(df_sum.to_string(index=False))
    df_sum.to_csv(tabdir / "branch_geobridge_summary.csv", index=False)

    log(f"Plots / top-N use --analysis-label={analysis_label!r} (score_mats keys per branch).")

    # Cache regulator score vectors per branch (Pax4 / Arx), like the notebook.
    configs = [
        dict(name="Pax4", home_branch="Beta_branch", away_branch="Alpha_branch"),
        dict(name="Arx", home_branch="Alpha_branch", away_branch="Beta_branch"),
    ]
    score_cache: dict[tuple[str, str], pd.DataFrame] = {}
    for cfg in configs:
        tf = cfg["name"]
        for b in [cfg["home_branch"], cfg["away_branch"]]:
            key = (b, analysis_label)
            if key not in score_mats:
                continue
            genes = score_mats[key]["genes"]
            S = score_mats[key]["S"]
            if tf not in set(genes.tolist()):
                continue
            j = int(np.where(genes == tf)[0][0])
            score_cache[(b, tf)] = pd.DataFrame({"Target": genes, "Score": S[:, j]})

            if args.orientation_check:
                # Save both orientations for debugging (column vs row)
                col = pd.DataFrame({"Target": genes, "Score": S[:, j]})
                row = pd.DataFrame({"Target": genes, "Score": S[j, :]})
                col["Abs"] = col["Score"].abs()
                row["Abs"] = row["Score"].abs()
                col.sort_values("Abs", ascending=False).head(20).to_csv(
                    tabdir / f"orientation_{b}_{tf}_col_top20.csv", index=False
                )
                row.sort_values("Abs", ascending=False).head(20).to_csv(
                    tabdir / f"orientation_{b}_{tf}_row_top20.csv", index=False
                )

    # Top-N sensitivity plots (home > away)
    if analysis_label == "k_auto":
        topn_csv = tabdir / "topn_home_vs_away_summary.csv"
        topn_png = figdir / "topn_home_vs_away_boxplots.png"
        spec_pax = figdir / "branch_specificity_Pax4.png"
        spec_arx = figdir / "branch_specificity_Arx.png"
        wilcox_csv = tabdir / "branch_specificity_wilcoxon_top200.csv"
    else:
        topn_csv = tabdir / f"topn_home_vs_away_summary__{analysis_label}.csv"
        topn_png = figdir / f"topn_home_vs_away_boxplots__{analysis_label}.png"
        spec_pax = figdir / f"branch_specificity_Pax4__{analysis_label}.png"
        spec_arx = figdir / f"branch_specificity_Arx__{analysis_label}.png"
        wilcox_csv = tabdir / f"branch_specificity_wilcoxon_top200__{analysis_label}.csv"
    log("Saving Top-N home-vs-away plots/tables…")
    topn_home_vs_away(
        topn_csv,
        topn_png,
        score_cache,
        configs,
        top_list=[50, 100, 200, 300],
        analysis_label=analysis_label,
        log_fn=log,
        score_orientation=None,
    )

    # Branch specificity plots (Fig3B/C analog) + paired Wilcoxon on Top-200 |score|
    k_rule = next((s.get("k_selected_rule") for s in summaries if s.get("label") == "k_auto"), None)
    log("Saving branch-specificity plots (Pax4/Arx)…")
    wilcox_rows = []
    r_pax = plot_branch_specificity(
        "Pax4",
        "Beta_branch",
        "Alpha_branch",
        score_cache,
        spec_pax,
        log_fn=log,
        analysis_label=analysis_label,
        score_orientation=None,
    )
    r_pax["k_selected_rule"] = k_rule
    r_pax["pair_mode"] = str(args.pair_mode)
    r_pax["min_dt"] = float(args.min_dt)
    r_pax["max_dt"] = float(args.max_dt)
    wilcox_rows.append(r_pax)
    log(
        f"Branch specificity Wilcoxon (Top-200 |score|, paired, greater): TF=Pax4 "
        f"p={r_pax.get('wilcoxon_p')} statistic={r_pax.get('wilcoxon_statistic')} n={r_pax.get('top_n_targets')}"
    )

    r_arx = plot_branch_specificity(
        "Arx",
        "Alpha_branch",
        "Beta_branch",
        score_cache,
        spec_arx,
        log_fn=log,
        analysis_label=analysis_label,
        score_orientation=None,
    )
    r_arx["k_selected_rule"] = k_rule
    r_arx["pair_mode"] = str(args.pair_mode)
    r_arx["min_dt"] = float(args.min_dt)
    r_arx["max_dt"] = float(args.max_dt)
    wilcox_rows.append(r_arx)
    log(
        f"Branch specificity Wilcoxon (Top-200 |score|, paired, greater): TF=Arx "
        f"p={r_arx.get('wilcoxon_p')} statistic={r_arx.get('wilcoxon_statistic')} n={r_arx.get('top_n_targets')}"
    )

    pd.DataFrame(wilcox_rows).to_csv(wilcox_csv, index=False)
    log(f"Saved Wilcoxon summary: {wilcox_csv.resolve()}")

    if args.orientation_verify:
        log("Orientation verify: full col/row vectors + duplicate top-N / branch-specificity / Wilcoxon (__col / __row)…")
        for cfg in configs:
            tf = cfg["name"]
            for b in (cfg["home_branch"], cfg["away_branch"]):
                key = (b, analysis_label)
                if key not in score_mats:
                    continue
                genes_arr = score_mats[key]["genes"]
                S = score_mats[key]["S"]
                if tf not in genes_arr.tolist():
                    continue
                j = int(np.where(genes_arr == tf)[0][0])
                gc = np.asarray(S[:, j], dtype=float)
                gr = np.asarray(S[j, :], dtype=float)
                dc = pd.DataFrame({"Target": genes_arr, "Score": gc, "Abs": np.abs(gc)})
                dr = pd.DataFrame({"Target": genes_arr, "Score": gr, "Abs": np.abs(gr)})
                dc.sort_values("Abs", ascending=False).to_csv(
                    tabdir / f"orientation_{b}_{tf}_col_full.csv", index=False
                )
                dr.sort_values("Abs", ascending=False).to_csv(
                    tabdir / f"orientation_{b}_{tf}_row_full.csv", index=False
                )

        topn_stem = "topn_home_vs_away_summary" if analysis_label == "k_auto" else f"topn_home_vs_away_summary__{analysis_label}"
        wilcox_stem = (
            "branch_specificity_wilcoxon_top200"
            if analysis_label == "k_auto"
            else f"branch_specificity_wilcoxon_top200__{analysis_label}"
        )
        spec_stem = lambda tf_name: (
            f"branch_specificity_{tf_name}" if analysis_label == "k_auto" else f"branch_specificity_{tf_name}__{analysis_label}"
        )

        sc_col = build_pancreas_tf_score_cache(score_mats, configs, analysis_label, "col")
        sc_row = build_pancreas_tf_score_cache(score_mats, configs, analysis_label, "row")

        sum_col = topn_home_vs_away(
            tabdir / f"{topn_stem}__col.csv",
            figdir / f"topn_home_vs_away_boxplots__col.png",
            sc_col,
            configs,
            top_list=[50, 100, 200, 300],
            analysis_label=analysis_label,
            log_fn=log,
            score_orientation="col",
        )
        sum_row = topn_home_vs_away(
            tabdir / f"{topn_stem}__row.csv",
            figdir / f"topn_home_vs_away_boxplots__row.png",
            sc_row,
            configs,
            top_list=[50, 100, 200, 300],
            analysis_label=analysis_label,
            log_fn=log,
            score_orientation="row",
        )

        wcol_rows: list[dict] = []
        r_pax_c = plot_branch_specificity(
            "Pax4",
            "Beta_branch",
            "Alpha_branch",
            sc_col,
            figdir / f"{spec_stem('Pax4')}__col.png",
            log_fn=log,
            analysis_label=analysis_label,
            score_orientation="col",
        )
        r_pax_c["k_selected_rule"] = k_rule
        r_pax_c["pair_mode"] = str(args.pair_mode)
        r_pax_c["min_dt"] = float(args.min_dt)
        r_pax_c["max_dt"] = float(args.max_dt)
        wcol_rows.append(r_pax_c)
        r_arx_c = plot_branch_specificity(
            "Arx",
            "Alpha_branch",
            "Beta_branch",
            sc_col,
            figdir / f"{spec_stem('Arx')}__col.png",
            log_fn=log,
            analysis_label=analysis_label,
            score_orientation="col",
        )
        r_arx_c["k_selected_rule"] = k_rule
        r_arx_c["pair_mode"] = str(args.pair_mode)
        r_arx_c["min_dt"] = float(args.min_dt)
        r_arx_c["max_dt"] = float(args.max_dt)
        wcol_rows.append(r_arx_c)
        pd.DataFrame(wcol_rows).to_csv(tabdir / f"{wilcox_stem}__col.csv", index=False)
        log(f"Saved orientation Wilcoxon (col): {(tabdir / f'{wilcox_stem}__col.csv').resolve()}")

        wrow_rows: list[dict] = []
        r_pax_r = plot_branch_specificity(
            "Pax4",
            "Beta_branch",
            "Alpha_branch",
            sc_row,
            figdir / f"{spec_stem('Pax4')}__row.png",
            log_fn=log,
            analysis_label=analysis_label,
            score_orientation="row",
        )
        r_pax_r["k_selected_rule"] = k_rule
        r_pax_r["pair_mode"] = str(args.pair_mode)
        r_pax_r["min_dt"] = float(args.min_dt)
        r_pax_r["max_dt"] = float(args.max_dt)
        wrow_rows.append(r_pax_r)
        r_arx_r = plot_branch_specificity(
            "Arx",
            "Alpha_branch",
            "Beta_branch",
            sc_row,
            figdir / f"{spec_stem('Arx')}__row.png",
            log_fn=log,
            analysis_label=analysis_label,
            score_orientation="row",
        )
        r_arx_r["k_selected_rule"] = k_rule
        r_arx_r["pair_mode"] = str(args.pair_mode)
        r_arx_r["min_dt"] = float(args.min_dt)
        r_arx_r["max_dt"] = float(args.max_dt)
        wrow_rows.append(r_arx_r)
        pd.DataFrame(wrow_rows).to_csv(tabdir / f"{wilcox_stem}__row.csv", index=False)
        log(f"Saved orientation Wilcoxon (row): {(tabdir / f'{wilcox_stem}__row.csv').resolve()}")

        comb_topn = pd.concat([sum_col, sum_row], ignore_index=True)
        comb_topn.to_csv(tabdir / "orientation_verify_topn_mannwhitney_combined.csv", index=False)
        comb_w = pd.concat([pd.DataFrame(wcol_rows), pd.DataFrame(wrow_rows)], ignore_index=True)
        comb_w.to_csv(tabdir / "orientation_verify_wilcoxon_combined.csv", index=False)
        log(f"Saved combined orientation tables: {tabdir / 'orientation_verify_topn_mannwhitney_combined.csv'}")

        log("=== Orientation verify report: Mann–Whitney (Top-N |score|), col vs row ===")
        for orient, sdf in [("col", sum_col), ("row", sum_row)]:
            log(f"--- {orient.upper()} ---")
            sub = sdf[sdf["TOP_N"].isin([50, 100, 200, 300])][
                ["TF", "TOP_N", "median_home", "median_away", "delta_median", "p"]
            ].sort_values(["TF", "TOP_N"])
            log(sub.to_string(index=False))
        log("=== Orientation verify report: Wilcoxon Top-200 |score| (paired, greater) ===")
        log(
            comb_w[["TF", "score_orientation", "wilcoxon_p", "wilcoxon_statistic", "top_n_targets"]]
            .sort_values(["TF", "score_orientation"])
            .to_string(index=False)
        )

    if bool(args.pax4_artifact_diagnostics):
        log("Running optional Pax4/regulator artifact diagnostics (A–C, interpretation)…")
        run_regulator_artifact_diagnostics(
            branches=branches,
            score_mats=score_mats,
            analysis_label=analysis_label,
            configs=configs,
            args=args,
            figdir=figdir,
            tabdir=tabdir,
            log=log,
            k_rule=k_rule,
        )

    if bool(args.low_expression_mask_diagnostic):
        log("Running optional low-expression row-mask diagnostic…")
        run_low_expression_mask_row_diagnostics(
            branches=branches,
            score_mats=score_mats,
            analysis_label=analysis_label,
            configs=configs,
            args=args,
            figdir=figdir,
            tabdir=tabdir,
            log=log,
            k_rule=k_rule,
            mask_threshold=float(args.mask_threshold),
        )
        if bool(args.pax4_artifact_diagnostics):
            log(
                "Note: combined with --pax4-artifact-diagnostics — inspect mask log above for whether "
                "Pax4 away-branch row signal collapses under low-expression row masking."
            )

    # Neurog3 k-grid stability with Enrichr (optional)
    if not args.skip_enrichr and "Neurog3" in set(adata.var_names.astype(str).tolist()):
        try:
            import gseapy as gp
            import matplotlib.pyplot as plt
            import seaborn as sns
            from scipy.stats import chi2

            TARGET_GO_IDS = {
                "Alpha_Repression": ["GO:2000146", "GO:0030334", "GO:0048870", "GO:0016477"],
                "Beta_Repression": ["GO:0051968", "GO:0050804", "GO:0061000", "GO:0043113", "GO:0007268"],
                "Hormone_Activation": ["GO:0050796", "GO:0030072", "GO:0042149", "GO:00090276"],
            }
            GO_SET = "GO_Biological_Process_2021"

            def combine_pvalues_fisher(p_values):
                p_values = np.array(p_values, dtype=float)
                p_values = np.clip(p_values, 1e-300, 1.0)
                if len(p_values) == 0:
                    return 1.0
                chi2_stat = -2 * np.sum(np.log(p_values))
                return float(chi2.sf(chi2_stat, 2 * len(p_values)))

            all_results = []
            for k_val in range(1, args.k_grid_max + 1):
                for bname in ["Alpha_branch", "Beta_branch"]:
                    if bname not in branches:
                        continue
                    ad = branches[bname].adata
                    genes = np.array(ad.var_names, dtype=str)
                    X = to_dense(ad.layers["Ms"]).astype(np.float64)
                    V = to_dense(ad.layers["velocity"]).astype(np.float64)
                    t = np.asarray(ad.obs["pt_use"].values, dtype=np.float64)
                    k_clamp_enr = None if int(args.k_clamp) <= 0 else int(args.k_clamp)
                    S, _, _, _, _ = geobridge_scores(
                        X,
                        V,
                        t,
                        min_dt=args.min_dt,
                        max_dt=args.max_dt,
                        k_override=k_val,
                        k_clamp=k_clamp_enr,
                        debug=False,
                    )
                    j = int(np.where(genes == "Neurog3")[0][0])
                    df = pd.DataFrame({"Target": genes, "Score": S[:, j]})
                    df["Abs"] = df["Score"].abs()
                    top_df = df.sort_values("Abs", ascending=False).head(300)
                    pos_genes = top_df[top_df["Score"] > 0]["Target"].tolist()
                    neg_genes = top_df[top_df["Score"] < 0]["Target"].tolist()

                    try:
                        enr_pos = gp.enrichr(gene_list=pos_genes, gene_sets=GO_SET, organism="Mouse").results
                        enr_neg = gp.enrichr(gene_list=neg_genes, gene_sets=GO_SET, organism="Mouse").results
                    except Exception:
                        enr_pos, enr_neg = pd.DataFrame(), pd.DataFrame()

                    if bname == "Alpha_branch":
                        target_ids = TARGET_GO_IDS["Alpha_Repression"]
                        found_p = []
                        if not enr_neg.empty:
                            found_p = enr_neg[enr_neg["Term"].str.contains("|".join(target_ids), regex=True)][
                                "Adjusted P-value"
                            ].tolist()
                        p_comb = combine_pvalues_fisher(found_p)
                        all_results.append(
                            {"k": k_val, "Category": "Alpha: Motility Repression (-)", "logP": -np.log10(p_comb + 1e-10)}
                        )

                    if bname == "Beta_branch":
                        target_ids = TARGET_GO_IDS["Beta_Repression"]
                        found_p = []
                        if not enr_neg.empty:
                            found_p = enr_neg[enr_neg["Term"].str.contains("|".join(target_ids), regex=True)][
                                "Adjusted P-value"
                            ].tolist()
                        p_comb = combine_pvalues_fisher(found_p)
                        all_results.append(
                            {"k": k_val, "Category": "Beta: Synapse Repression (-)", "logP": -np.log10(p_comb + 1e-10)}
                        )

                    target_ids = TARGET_GO_IDS["Hormone_Activation"]
                    found_p = []
                    if not enr_pos.empty:
                        found_p = enr_pos[enr_pos["Term"].str.contains("|".join(target_ids), regex=True)][
                            "Adjusted P-value"
                        ].tolist()
                    p_comb = combine_pvalues_fisher(found_p)
                    branch_short = bname.split("_")[0]
                    all_results.append(
                        {
                            "k": k_val,
                            "Category": f"Common: Hormone Activation (+) – {branch_short}",
                            "logP": -np.log10(p_comb + 1e-10),
                        }
                    )

            df_plot = pd.DataFrame(all_results)
            df_plot.to_csv(tabdir / "neurog3_kgrid_enrichr.csv", index=False)

            palette = {
                "Alpha: Motility Repression (-)": "#1f77b4",
                "Beta: Synapse Repression (-)": "#2ca02c",
                "Common: Hormone Activation (+) – Alpha": "#ff7f0e",
                "Common: Hormone Activation (+) – Beta": "#ffbb78",
            }
            fig = plt.figure(figsize=(9.8, 5.4))
            sns.lineplot(
                data=df_plot,
                x="k",
                y="logP",
                hue="Category",
                palette=palette,
                markers=True,
                markersize=8,
                linewidth=2.4,
            )
            plt.axhline(-np.log10(0.05), color="red", linestyle="--", linewidth=1.6)
            plt.title("Stability across k-grid: Neurog3 (GeoBRIDGE)")
            plt.ylabel("-log10(Fisher combined P)")
            plt.xlabel("k")
            plt.grid(True, alpha=0.3)
            plt.legend(title=None, fontsize=9, bbox_to_anchor=(1.02, 1), loc="upper left")
            fig.tight_layout()
            fig.savefig(figdir / "neurog3_kgrid_enrichr.png", dpi=220, bbox_inches="tight")
            plt.close(fig)
        except Exception as e:
            (tabdir / "enrichr_error.txt").write_text(str(e))

    # Write a compact run manifest
    (outdir / "run_manifest.json").write_text(
        json.dumps(
            {
                "run_name": str(run_name),
                "analysis_label": str(analysis_label),
                "adata_shape": [int(adata.n_obs), int(adata.n_vars)],
                "recover_dynamics_succeeded": bool(recover_ok),
                "latent_time_exists": bool(latent_ok),
                "pt_forced": {
                    "force_dpt": bool(args.force_dpt),
                    "force_latent_time": bool(args.force_latent_time),
                },
                "pair_settings": {
                    "realdata_preset": str(args.realdata_preset),
                    "min_dt": float(args.min_dt),
                    "max_dt": float(args.max_dt),
                    "pair_mode": str(args.pair_mode),
                    "global_max_pairs": int(args.global_max_pairs),
                    "max_pairs_per_cell": int(args.max_pairs_per_cell),
                    "stream_num_threads": int(args.stream_num_threads),
                },
                "k_selection": {
                    "max_k": int(args.max_k),
                    "tau": float(args.tau),
                    "tail_fraction": float(args.tail_fraction),
                    "tail_min_start": int(args.tail_min_start),
                    "k_clamp_deprecated": None if int(args.k_clamp) <= 0 else int(args.k_clamp),
                    "k_override": None if args.k_override is None else int(args.k_override),
                    "k_selected_rule": int(k_selected),
                    "k_auto_raw_by_branch": {k: int(v) for k, v in k_auto_raw_by_branch.items()},
                    "k_selection_rule": {
                        "K_cap": 15,
                        "r_min": 0.05,
                        "asym_max": 3.0,
                    },
                },
                "k_selection_diagnostics": k_table,
                "orientation_verify": bool(args.orientation_verify),
                "branches": summaries,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log("Done. Outputs written under outdir.")


if __name__ == "__main__":
    main()

