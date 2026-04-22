import json
from dataclasses import asdict
from pathlib import Path
import numpy as np
from .config import GeoBridgeConfig
from .pairspace import accumulate_streaming_pairspace_stats, build_forward_pairs, iter_forward_pair_blocks
from .spectral_k import select_k_by_tail_exhaustion
from .scoring import dual_project_pairspace, score_matrix

class GeoBridgeModel:
    def __init__(self, config=None):
        self.config = config or GeoBridgeConfig()
        self.fit_result = None

    @staticmethod
    def precompute_pairspace_svd(
        X,
        V,
        pseudotime,
        min_forward_dt=1e-4,
        max_forward_dt=None,
        max_pairs_per_cell=None,
        pair_seed=None,
    ):
        """
        Precompute the expensive pieces shared across many k choices:
        forward pairs, mean-centered DX, and SVD(DXc).

        Returns a dict suitable for reuse across many projections.
        """
        pair_data = build_forward_pairs(
            X,
            V,
            pseudotime,
            min_forward_dt=min_forward_dt,
            max_forward_dt=max_forward_dt,
            max_pairs_per_cell=max_pairs_per_cell,
            pair_seed=pair_seed,
        )
        DX, DV = pair_data.delta_x, pair_data.delta_v
        DXc = DX - DX.mean(axis=0, keepdims=True)
        U, S, Vt = np.linalg.svd(DXc, full_matrices=False)
        return {
            "DXc": DXc,
            "DV": DV,
            "U": U,
            "S": S,
            "Vt": Vt,
            "pair_i": pair_data.pair_i,
            "pair_j": pair_data.pair_j,
        }

    @staticmethod
    def precompute_pairspace_stats_streaming(
        X,
        V,
        pseudotime,
        min_forward_dt=1e-4,
        max_forward_dt=None,
        max_pairs_per_cell=None,
        pair_seed=None,
        stream_num_threads=None,
        pair_normalize=None,
    ):
        """
        Exact streaming alternative to materializing ΔX/ΔV and doing SVD on (n_pairs x G).

        Computes small sufficient statistics:
          - mu: mean(ΔX) over pairs
          - C: ΔXc^T ΔXc  (GxG)
          - M: ΔXc^T ΔV   (GxG)
          - Y: ΔV^T ΔV    (GxG)
        plus an eigendecomposition of C, which yields the same singular values as SVD(ΔXc).
        """
        n, G = np.asarray(X).shape
        sum_dx, sum_dv, sum_dx_dx, sum_dx_dv, sum_dv_dv, n_pairs = accumulate_streaming_pairspace_stats(
            X,
            V,
            pseudotime,
            min_forward_dt=min_forward_dt,
            max_forward_dt=max_forward_dt,
            max_pairs_per_cell=max_pairs_per_cell,
            pair_seed=pair_seed,
            num_threads=stream_num_threads,
            pair_normalize=pair_normalize,
        )

        if n_pairs <= 0:
            raise ValueError("No admissible forward pairs were generated.")

        mu = sum_dx / float(n_pairs)
        # Centering ΔX only (matches the current GeoBridgeModel.fit behavior)
        C = sum_dx_dx - float(n_pairs) * np.outer(mu, mu)
        M = sum_dx_dv - np.outer(mu, sum_dv)
        Y = sum_dv_dv

        # Numerical symmetry guard (accumulation may introduce tiny asymmetry)
        C = 0.5 * (C + C.T)

        # Eigendecomposition of C gives right singular vectors of ΔXc and singular values (sqrt eigenvalues).
        evals, evecs = np.linalg.eigh(C)
        order = np.argsort(evals)[::-1]
        evals = evals[order]
        evecs = evecs[:, order]
        evals = np.maximum(evals, 0.0)
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

    @staticmethod
    def scores_from_streaming_stats(pre, k, mode="out", project_y=True,
                                     normalize_score=True, eps=1e-12):
        """
        Compute score matrix for a chosen k using streaming stats (no ΔX/ΔV materialization).

        mode selects the projector applied to the numerator / x-norm:
          - "out"  (default, production): num = (I - Pk) * M, x_norm = diag(Q C Q)
          - "onto"                      : num = Pk * M,       x_norm = diag(Pk C Pk)
          - "none" (or k == 0)          : num = M,            x_norm = diag(C)

          where Pk = V_k V_k^T (top-k right singular vectors of ΔXc), Q = I - Pk.

        y-norms:
          - "out" + project_y=True : diag(Y - M^T V_k Σ_k^{-2} V_k^T M)
          - otherwise              : diag(Y)
        """
        C = pre["C"]
        M = pre["M"]
        Y = pre["Y"]
        S = pre["S"]
        Vt = pre["Vt"]
        G = int(pre["G"])

        k = int(k)
        if k < 0:
            k = 0
        k = min(k, G)

        if mode not in ("out", "onto", "none"):
            raise ValueError(f"mode must be one of out|onto|none, got {mode!r}")

        V = Vt.T  # (G, G)
        if mode == "none" or k == 0:
            num = M
            x_norm2 = np.clip(np.diag(C), 0.0, None)
            y_norm2 = np.clip(np.diag(Y), 0.0, None)
        else:
            Vk = V[:, :k]  # (G, k)
            Pk = Vk @ Vk.T
            if mode == "out":
                Q = np.eye(G) - Pk
                num = Q @ M
                x_norm2 = np.clip(np.diag(Q @ C @ Q), 0.0, None)
                if project_y:
                    # diag(M^T Vk Σ^-2 Vk^T M) computed without forming the full matrix
                    Sk = S[:k]
                    inv_s2 = np.zeros_like(Sk)
                    good = Sk > 1e-12
                    inv_s2[good] = 1.0 / (Sk[good] ** 2)
                    T = Vk.T @ M  # (k, G)
                    diag_R = (T * T).T @ inv_s2  # (G,)
                    y_norm2 = np.clip(np.diag(Y) - diag_R, 0.0, None)
                else:
                    y_norm2 = np.clip(np.diag(Y), 0.0, None)
            else:  # mode == "onto"
                num = Pk @ M
                x_norm2 = np.clip(np.diag(Pk @ C @ Pk), 0.0, None)
                y_norm2 = np.clip(np.diag(Y), 0.0, None)

        if not normalize_score:
            return num
        denom = np.maximum(np.sqrt(x_norm2)[:, None] * np.sqrt(y_norm2)[None, :], eps)
        return num / denom

    @staticmethod
    def scores_from_precomputed(pre, k, project_y=True, normalize_score=True, eps=1e-12):
        """
        Compute scores for a chosen k using precomputed pairspace+SVD.
        """
        DXc, DV, U, S, Vt = pre["DXc"], pre["DV"], pre["U"], pre["S"], pre["Vt"]
        # Guard against k > numerical rank; dual_project_pairspace slices safely but keep k sane.
        k = int(k)
        if k < 0:
            k = 0
        if S.size > 0:
            k = min(k, int(S.size))
        if project_y:
            DX_res, DV_res = dual_project_pairspace(DXc, DV, U, S, Vt, k)
        else:
            DX_res, _ = dual_project_pairspace(DXc, DV, U, S, Vt, k)
            DV_res = DV
        return score_matrix(DX_res, DV_res, normalize_score, eps)

    def fit(self, X, V, pseudotime):
        pre = self.precompute_pairspace_svd(
            X,
            V,
            pseudotime,
            min_forward_dt=self.config.min_forward_dt,
            max_forward_dt=self.config.max_forward_dt,
            max_pairs_per_cell=self.config.max_pairs_per_cell,
            pair_seed=self.config.pair_seed,
        )
        DXc, DV, U, S, Vt = pre["DXc"], pre["DV"], pre["U"], pre["S"], pre["Vt"]
        spectral = select_k_by_tail_exhaustion(
            S,
            self.config.max_k,
            self.config.tail_fraction,
            self.config.tail_min_start,
            self.config.slope_window,
            self.config.stable_m,
            self.config.tau,
        )
        k = spectral.k_selected
        if self.config.project_y:
            DX_res, DV_res = dual_project_pairspace(DXc, DV, U, S, Vt, k)
        else:
            DX_res, _ = dual_project_pairspace(DXc, DV, U, S, Vt, k)
            DV_res = DV.copy()
        scores = score_matrix(DX_res, DV_res, self.config.normalize_score, self.config.eps)
        self.fit_result = {
            "scores": scores,
            "k_selected": int(k),
            "pair_i": pre["pair_i"],
            "pair_j": pre["pair_j"],
            "singular_values": S,
            "spectral": spectral,
        }
        return self.fit_result

    def save(self, outdir):
        if self.fit_result is None:
            raise RuntimeError("Call fit() before save().")
        outdir = Path(outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        np.save(outdir / "scores.npy", self.fit_result["scores"])
        np.save(outdir / "singular_values.npy", self.fit_result["singular_values"])
        spectral = self.fit_result["spectral"]
        np.save(outdir / "log_spectrum.npy", spectral.log_spectrum)
        np.save(outdir / "slope.npy", spectral.slope)
        np.save(outdir / "slope_smoothed.npy", spectral.slope_smoothed)
        meta = {
            "config": asdict(self.config),
            "k_selected": int(self.fit_result["k_selected"]),
            "tail_baseline": float(spectral.tail_baseline),
            "threshold": float(spectral.threshold),
            "tail_start": int(spectral.tail_start),
        }
        (outdir / "metadata.json").write_text(json.dumps(meta, indent=2))
