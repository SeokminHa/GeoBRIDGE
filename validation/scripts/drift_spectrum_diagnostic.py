"""Diagnostic: drift-basis vs top-k SVD subspace comparison.

Answers two questions after the drift experiment:
  (1) What fraction of pair-space ΔX variance does order=d polynomial basis
      capture? (train + held-out)
  (2) Is the drift basis the same subspace as the production top-k SVD basis
      the paper would otherwise use? (principal angles / cosine similarity)

If drift basis is ~identical to top-k SVD on ΔX, then `onto` on drift is
just the production operator's `onto` with a different label — no new
information. If angles are large, drift basis captures a genuinely distinct
(pseudotime-aligned) low-d subspace.

Output:
  spectrum_<name>.csv  (per-order variance explained + top-SVD angle)
  variance_curve.png
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_here = Path(__file__).resolve()
for p in _here.parents:
    if (p / "geobridge_model" / "__init__.py").exists():
        sys.path.insert(0, str(p))
        break

from geobridge_model.config import GeoBridgeConfig
from geobridge_model.model import GeoBridgeModel

from run_pt_drift_projection import (_sample_forward_pairs, _poly_basis,
                                     _get_pseudotime, _drift_basis)


def _variance_explained(dXc, DN):
    """Fraction of SS(dXc) captured by OLS on DN."""
    B, *_ = np.linalg.lstsq(DN, dXc, rcond=None)
    ss_tot = float((dXc ** 2).sum())
    if ss_tot <= 0:
        return 0.0
    ss_res = float(((dXc - DN @ B) ** 2).sum())
    return 1.0 - ss_res / ss_tot


def _principal_angles(U, V):
    """Sorted cos(principal angles) between columns spans of U (G,a) and V (G,b)."""
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    s = np.linalg.svd(Qu.T @ Qv, compute_uv=False)
    return np.clip(s, 0.0, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/drift_diag")
    ap.add_argument("--layer-X", default="Ms")
    ap.add_argument("--layer-V", default="velocity")
    ap.add_argument("--orders", default="1,2,3,4,5")
    ap.add_argument("--svd-ks", default="2,5,10,17",
                    help="compare V_drift to top-k SVD of ΔXc for these k")
    ap.add_argument("--n-pairs", type=int, default=100_000)
    ap.add_argument("--max-pairs-per-cell", type=int, default=200)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    adata = sc.read(cfg["h5ad"])
    X = np.asarray(adata.layers.get(args.layer_X, adata.X)).astype(np.float64)
    V_lay = np.asarray(adata.layers[args.layer_V]).astype(np.float64)
    pt = _get_pseudotime(adata)
    order_idx = np.argsort(pt, kind="mergesort")
    X = np.nan_to_num(X[order_idx], nan=0.0, posinf=0.0, neginf=0.0)
    V_lay = np.nan_to_num(V_lay[order_idx], nan=0.0, posinf=0.0, neginf=0.0)
    pt = pt[order_idx]

    out_dir = Path(args.out) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diag] sampling {args.n_pairs:,} forward pairs")
    dpt, dX = _sample_forward_pairs(X, pt, n_pairs=args.n_pairs)
    dXc = dX - dX.mean(axis=0, keepdims=True)

    rng = np.random.default_rng(7)
    fold = rng.integers(0, 5, size=dpt.shape[0])
    tr = fold != 0
    te = fold == 0

    orders = [int(o) for o in args.orders.split(",") if o.strip()]
    svd_ks = [int(k) for k in args.svd_ks.split(",") if k.strip()]

    print(f"[diag] computing top-{max(svd_ks)} SVD of ΔXc (train fold) ...")
    t0 = time.time()
    _, _, Vt = np.linalg.svd(dXc[tr], full_matrices=False)
    Vsvd_full = Vt.T  # (G, min(n_tr, G)), right singular vectors
    print(f"[diag]   SVD done ({time.time()-t0:.1f}s)")

    rows = []
    v_drifts = {}
    for o in orders:
        DN_all = _poly_basis(dpt, o)
        ve_train = _variance_explained(dXc[tr], DN_all[tr])
        ve_test = _variance_explained(dXc[te], DN_all[te])
        V_d = _drift_basis(X, pt, n_pairs=args.n_pairs, order=o)
        v_drifts[o] = V_d
        row = {"order": o, "d": V_d.shape[1],
               "ve_train": ve_train, "ve_test": ve_test}
        for k in svd_ks:
            Vk = Vsvd_full[:, :k]
            ang = _principal_angles(V_d, Vk)
            row[f"mean_cos_svd{k}"] = float(ang.mean())
            row[f"max_cos_svd{k}"] = float(ang.max())
        rows.append(row)
        print(f"[diag] order={o} d={V_d.shape[1]}  "
              f"VE_train={ve_train:.4f}  VE_test={ve_test:.4f}  "
              f"max_cos(vs top-{svd_ks[-1]}SVD)={row[f'max_cos_svd{svd_ks[-1]}']:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "spectrum.csv", index=False)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["order"], df["ve_train"], "-o", label="train VE")
    ax.plot(df["order"], df["ve_test"], "-s", label="held-out VE")
    ax.set_xlabel("polynomial order")
    ax.set_ylabel("ΔX variance explained")
    ax.set_title(f"{cfg['name']}: pt-drift variance curve")
    ax.set_ylim(0, max(0.05, df["ve_train"].max() * 1.1))
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / "variance_curve.png", dpi=150)
    plt.close(fig)

    print(f"[diag] wrote spectrum.csv + variance_curve.png to {out_dir}")


if __name__ == "__main__":
    main()
