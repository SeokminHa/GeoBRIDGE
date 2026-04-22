"""Graph signal processing diagnostic over the velocity graph.

Interpretive tool only — this does NOT replace the scoring operator. The goal
is to localize where (in graph-spectral terms) each TF's expression sits.

Pipeline:
  1. Build the symmetric normalized Laplacian of the velocity graph
     (L = I − D^{-1/2} W D^{-1/2}, where W = 0.5 * (A + A.T) and
     A = adata.uns["velocity_graph"] is a cosine-sim transition matrix).
  2. Get the K=50 smallest eigenvectors/values via sparse Lanczos.
  3. Project each TF's (and optionally a reference set's) expression onto
     the eigenvector basis; compute |c_k|^2 per band.
  4. Emit a per-TF CSV of spectral power in low/mid/high bands + cumulative
     power plot.

Reading the output:
  - Lineage TFs sit in the low-frequency band ⇒ smooth along the graph.
    The production operator projects out the low-frequency subspace, so
    their signal becomes small — visual confirmation of the diagnosis.
  - Stress/IEG TFs spread into mid/high bands ⇒ partly survive the
    projection; consistent with NFKB1/HSF1 being the only wins.
"""
from __future__ import annotations
import argparse
import sys
import time
import yaml
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _build_normalized_laplacian(W):
    W = sp.csr_matrix(W).astype(np.float64)
    W = 0.5 * (W + W.T)
    W.setdiag(0.0)
    W.eliminate_zeros()
    d = np.asarray(W.sum(axis=1)).ravel()
    d_inv_sqrt = np.zeros_like(d)
    good = d > 0
    d_inv_sqrt[good] = 1.0 / np.sqrt(d[good])
    D_is = sp.diags(d_inv_sqrt)
    L = sp.eye(W.shape[0]) - D_is @ W @ D_is
    L = 0.5 * (L + L.T)
    return L.tocsr()


def _get_velocity_graph(adata):
    for key in ("velocity_graph", "T_forward", "transitions_forward"):
        if key in adata.uns:
            A = adata.uns[key]
            if sp.issparse(A):
                return A
            return sp.csr_matrix(A)
    raise KeyError(
        "adata.uns has no velocity_graph / T_forward / transitions_forward. "
        "Run scvelo.tl.velocity_graph first."
    )


def _spectral_power(expr, U):
    """expr: (n_cells,), U: (n_cells, K). Return (K,) array of |c_k|^2 / ||x||^2."""
    x = np.asarray(expr, dtype=np.float64)
    x -= x.mean()
    nrm2 = float(x @ x)
    if nrm2 <= 0:
        return np.zeros(U.shape[1])
    c = U.T @ x
    return (c * c) / nrm2


def _band_fractions(p, K):
    """Low = bottom 20%, mid = 20-60%, high = top 40%."""
    lo = slice(0, max(1, K // 5))
    mid = slice(lo.stop, max(lo.stop + 1, 3 * K // 5))
    hi = slice(mid.stop, K)
    return float(p[lo].sum()), float(p[mid].sum()), float(p[hi].sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--K", type=int, default=50,
                    help="number of smallest Laplacian eigenvectors to compute")
    ap.add_argument("--tfs", nargs="*", default=None,
                    help="subset of TFs to score; default = all TFs in cfg['tf_list']")
    ap.add_argument("--out", default="results/gsp_diag")
    ap.add_argument("--layer-X", default="Ms")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    adata = sc.read(cfg["h5ad"])
    tfs_all = [l.strip() for l in open(cfg["tf_list"]) if l.strip()]
    tfs = [t for t in (args.tfs or tfs_all) if t in adata.var_names]
    print(f"[gsp] dataset={cfg['name']} n_cells={adata.n_obs} K={args.K} "
          f"n_tfs_in_scope={len(tfs)}")

    W = _get_velocity_graph(adata)
    L = _build_normalized_laplacian(W)
    print(f"[gsp] Laplacian shape {L.shape}, nnz={L.nnz}")

    t0 = time.time()
    vals, U = spla.eigsh(L, k=args.K, which="SM")
    order = np.argsort(vals)
    vals = vals[order]
    U = U[:, order]
    print(f"[gsp] eigsh done in {time.time()-t0:.1f}s; "
          f"lambda range [{vals[0]:.4f}, {vals[-1]:.4f}]")

    X = adata.layers.get(args.layer_X, adata.X)
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float64)
    name_to_idx = {g: i for i, g in enumerate(adata.var_names)}

    rows = []
    power_mat = np.zeros((len(tfs), args.K))
    for r, tf in enumerate(tfs):
        j = name_to_idx[tf]
        p = _spectral_power(X[:, j], U)
        power_mat[r] = p
        lo, mid, hi = _band_fractions(p, args.K)
        rows.append({"tf": tf, "low_frac": lo, "mid_frac": mid,
                     "high_frac": hi, "peak_idx": int(np.argmax(p))})
    df = pd.DataFrame(rows).sort_values("low_frac", ascending=False)

    out_dir = Path(args.out) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "spectral_summary.csv", index=False)
    np.savez(out_dir / "spectra.npz",
             tfs=np.array(tfs), eigvals=vals, power=power_mat)

    cum = np.cumsum(power_mat, axis=1)
    fig, ax = plt.subplots(figsize=(7, 5))
    for r, tf in enumerate(tfs):
        ax.plot(np.arange(1, args.K + 1), cum[r], alpha=0.6, label=tf)
    ax.set_xlabel("graph-spectrum index k (low → high freq)")
    ax.set_ylabel("cumulative expression power")
    ax.set_title(f"{cfg['name']}: TF spectral localization")
    if len(tfs) <= 12:
        ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_dir / "cumulative_power.png", dpi=150)
    plt.close(fig)
    print(f"[gsp] done; wrote spectral_summary.csv + cumulative_power.png to {out_dir}")


if __name__ == "__main__":
    main()
