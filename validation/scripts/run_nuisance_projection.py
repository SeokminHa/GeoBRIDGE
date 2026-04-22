"""Nuisance-only basis projection for GeoBRIDGE.

Motivation: decomposition experiment (decompose_scoring.py) falsified the
"top-k SVD of ΔX removes lineage signal" hypothesis — gb_none, gb_onto,
gb_out all give essentially the same sign-flip pattern on V2 ChIP lineage
TFs (see real_day2.md Day 3 update). Sign flip is in raw M itself.

This script tests whether projecting out a *biologically-motivated*
nuisance basis (cell cycle + library size + mito) recovers lineage TFs,
while data-driven top-k does not.

Prior prediction (see real_day2.md Day 4 critique): likely fails by the
same mechanism — small-rank projection (d≈3–4) perturbs M only slightly.
Run anyway because (a) cheap, (b) closes the "projection-operator space"
falsification, (c) small non-zero chance of success if nuisance direction
happens to align with the sign-flip-causing component.

Decision rule (strict): lineage TFs (GATA2/KLF1/NFE2) must reach NES ≥
+0.5 on V2 ChIP for this to be publishable; anything weaker is effective
failure.
"""
from __future__ import annotations
import argparse
import time
import yaml
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import scanpy as sc
import scipy.sparse as sp

_here = Path(__file__).resolve()
for p in _here.parents:
    if (p / "geobridge_model" / "__init__.py").exists():
        sys.path.insert(0, str(p))
        break

from geobridge_model.config import GeoBridgeConfig
from geobridge_model.model import GeoBridgeModel
from geobridge_val.io import save_ranking


# Tirosh 2016 cell cycle gene lists (human symbols; upper-case match used).
# For mouse datasets we try capitalize() fallback automatically.
S_GENES_HUMAN = [
    "MCM5", "PCNA", "TYMS", "FEN1", "MCM2", "MCM4", "RRM1", "UNG", "GINS2",
    "MCM6", "CDCA7", "DTL", "PRIM1", "UHRF1", "MLF1IP", "HELLS", "RFC2",
    "RPA2", "NASP", "RAD51AP1", "GMNN", "WDR76", "SLBP", "CCNE2", "UBR7",
    "POLD3", "MSH2", "ATAD2", "RAD51", "RRM2", "CDC45", "CDC6", "EXO1",
    "TIPIN", "DSCC1", "BLM", "CASP8AP2", "USP1", "CLSPN", "POLA1", "CHAF1B",
    "BRIP1", "E2F8",
]
G2M_GENES_HUMAN = [
    "HMGB2", "CDK1", "NUSAP1", "UBE2C", "BIRC5", "TPX2", "TOP2A", "NDC80",
    "CKS2", "NUF2", "CKS1B", "MKI67", "TMPO", "CENPF", "TACC3", "FAM64A",
    "SMC4", "CCNB2", "CKAP2L", "CKAP2", "AURKB", "BUB1", "KIF11", "ANP32E",
    "TUBB4B", "GTSE1", "KIF20B", "HJURP", "CDCA3", "HN1", "CDC20", "TTK",
    "CDC25C", "KIF2C", "RANGAP1", "NCAPD2", "DLGAP5", "CDCA2", "CDCA8",
    "ECT2", "KIF23", "HMMR", "AURKA", "PSRC1", "ANLN", "LBR", "CKAP5",
    "CENPE", "CTCF", "NEK2", "G2E3", "GAS2L3", "CBX5", "CENPA",
]


def _score_cc(adata):
    """Score cell cycle (S, G2M) into adata.obs; case-insensitive matching."""
    if "S_score" in adata.obs.columns and "G2M_score" in adata.obs.columns:
        return
    vn = adata.var_names.to_numpy()
    vn_upper = np.char.upper(vn.astype(str))

    def match(lst):
        want = set(lst)
        keep = [vn[i] for i, u in enumerate(vn_upper) if u in want]
        if len(keep) < 5:
            # mouse fallback: try capitalize()
            want_cap = {g.capitalize() for g in lst}
            keep = [g for g in vn if g in want_cap]
        return keep

    s_genes = match(S_GENES_HUMAN)
    g2m_genes = match(G2M_GENES_HUMAN)
    print(f"[nuis] cell cycle overlap: S={len(s_genes)}, G2M={len(g2m_genes)}")
    if len(s_genes) < 5 or len(g2m_genes) < 5:
        raise RuntimeError("Too few cell cycle genes in var_names; check species")
    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)


def _compute_nuisance_scores(adata, mito_prefixes=("MT-", "Mt-")):
    """Return (n_cells, d_nuis) and column names."""
    _score_cc(adata)

    X = adata.X
    if sp.issparse(X):
        X = X.toarray()
    X = np.asarray(X).astype(np.float64)

    cols = [
        np.asarray(adata.obs["S_score"], dtype=np.float64),
        np.asarray(adata.obs["G2M_score"], dtype=np.float64),
        np.log1p(X.sum(axis=1)).astype(np.float64),
        np.log1p((X > 0).sum(axis=1)).astype(np.float64),
    ]
    names = ["S_score", "G2M_score", "log_total_counts", "log_n_genes"]

    vn_upper = np.char.upper(np.asarray(adata.var_names, dtype=str))
    tot = X.sum(axis=1)

    mito_mask = np.zeros(adata.n_vars, dtype=bool)
    for pref in mito_prefixes:
        mito_mask |= np.char.startswith(vn_upper, pref.upper())
    if int(mito_mask.sum()) > 0:
        mito_frac = X[:, mito_mask].sum(axis=1) / np.clip(tot, 1e-12, None)
        cols.append(mito_frac.astype(np.float64))
        names.append("pct_mito")
    else:
        print("[nuis] no mito genes found; skipping pct_mito")

    # ribosomal prefixes RPS / RPL (mouse Rps/Rpl — uppercase covers both)
    ribo_mask = np.char.startswith(vn_upper, "RPS") | \
                np.char.startswith(vn_upper, "RPL")
    if int(ribo_mask.sum()) > 0:
        ribo_frac = X[:, ribo_mask].sum(axis=1) / np.clip(tot, 1e-12, None)
        cols.append(ribo_frac.astype(np.float64))
        names.append("pct_ribo")
    else:
        print("[nuis] no ribosomal genes found; skipping pct_ribo")

    N = np.column_stack(cols)
    print(f"[nuis] nuisance matrix shape {N.shape}; columns: {names}")
    return N, names


def _nuisance_basis_pair_space(X, N, order, n_pairs=100_000,
                                dt_max=50, seed=0):
    """Return V_nuis (G, d), gene directions in ΔX that explain ΔN across pairs.

    X, N: pseudotime-ordered (caller is responsible). Pairs are forward
    within a window of size dt_max, matching the production pairing spirit.
    """
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    i_idx = rng.integers(0, n - 1, size=n_pairs)
    # forward offset in [1, min(dt_max, n-1-i)]
    max_off = np.clip(n - 1 - i_idx, 1, dt_max)
    off = (rng.random(n_pairs) * max_off).astype(np.int64) + 1
    j_idx = np.clip(i_idx + off, 0, n - 1)

    dX = X[j_idx] - X[i_idx]  # (P, G)
    dN = N[j_idx] - N[i_idx]  # (P, d)
    dX -= dX.mean(axis=0, keepdims=True)
    dN -= dN.mean(axis=0, keepdims=True)

    # OLS: dX ≈ dN @ B, B (d, G). solve least squares.
    B, *_ = np.linalg.lstsq(dN, dX, rcond=None)
    # V_nuis: orthonormal gene-space basis spanning B.T columns
    Vn, _ = np.linalg.qr(B.T)
    return Vn


def _scores_with_vnuis(pre, V_nuis, mode="out", eps=1e-12):
    """Score matrix with explicit nuisance basis V_nuis (G, d).

    Mirrors the structure of scores_from_streaming_stats but substitutes
    a fixed V_nuis for the top-k SVD basis. project_y not applied because
    the y-side residualization formula assumes SVD spectrum.
    """
    C = pre["C"]
    M = pre["M"]
    Y = pre["Y"]
    G = int(pre["G"])

    d = 0 if V_nuis is None else V_nuis.shape[1]
    if d == 0 or mode == "none":
        num = M
        x_norm2 = np.clip(np.diag(C), 0.0, None)
    else:
        P = V_nuis @ V_nuis.T
        if mode == "out":
            Q = np.eye(G) - P
            num = Q @ M
            x_norm2 = np.clip(np.diag(Q @ C @ Q), 0.0, None)
        elif mode == "onto":
            num = P @ M
            x_norm2 = np.clip(np.diag(P @ C @ P), 0.0, None)
        else:
            raise ValueError(f"bad mode {mode!r}; expected out|onto|none")
    y_norm2 = np.clip(np.diag(Y), 0.0, None)
    denom = np.maximum(np.sqrt(x_norm2)[:, None] * np.sqrt(y_norm2)[None, :], eps)
    return num / denom


def _scores_to_long(scores, genes, tf_set):
    tf_idx = [i for i, g in enumerate(genes) if g in tf_set]
    rows = []
    for i in tf_idx:
        tf = genes[i]
        for j, tgt in enumerate(genes):
            if j == i:
                continue
            rows.append((tf, tgt, float(scores[i, j])))
    return pd.DataFrame(rows, columns=["regulator", "target", "score"])


def _get_pseudotime(adata):
    for key in ("velocity_pseudotime", "dpt_pseudotime",
                "palantir_pseudotime", "pseudotime"):
        if key in adata.obs.columns:
            pt = np.asarray(adata.obs[key]).astype(np.float64)
            if np.all(np.isfinite(pt)):
                return pt
    import scvelo as scv
    scv.tl.velocity_pseudotime(adata)
    return np.asarray(adata.obs["velocity_pseudotime"]).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", default="results/rankings_nuis")
    ap.add_argument("--layer-X", default="Ms")
    ap.add_argument("--layer-V", default="velocity")
    ap.add_argument("--modes", default="none,onto,out",
                    help="none=baseline (same as gb_none); "
                         "out=project out nuisance (main); "
                         "onto=what the nuisance basis itself captures")
    ap.add_argument("--n-pairs-for-basis", type=int, default=100_000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    adata = sc.read(cfg["h5ad"])
    tfs = [l.strip() for l in open(cfg["tf_list"]) if l.strip()]
    tf_set = set(tfs)
    genes = adata.var_names.tolist()

    # Nuisance scores on unordered adata (obs-aligned)
    N_raw, n_names = _compute_nuisance_scores(adata)

    X = np.asarray(adata.layers.get(args.layer_X, adata.X)).astype(np.float64)
    V = np.asarray(adata.layers[args.layer_V]).astype(np.float64)
    pt = _get_pseudotime(adata)
    order = np.argsort(pt, kind="mergesort")
    X = np.nan_to_num(X[order], nan=0.0, posinf=0.0, neginf=0.0)
    V = np.nan_to_num(V[order], nan=0.0, posinf=0.0, neginf=0.0)
    pt = pt[order]
    N = N_raw[order]

    print(f"[nuis] building V_nuis from ΔX–ΔN regression on "
          f"{args.n_pairs_for_basis:,} pairs...")
    V_nuis = _nuisance_basis_pair_space(X, N, order,
                                         n_pairs=args.n_pairs_for_basis)
    print(f"[nuis] V_nuis shape {V_nuis.shape}; columns: {n_names}")

    base = cfg.get("geobridge", {})
    gcfg = GeoBridgeConfig(
        max_k=base.get("max_k", 40),
        tail_fraction=base.get("tail_fraction", 0.2),
        tail_min_start=base.get("tail_min_start", 20),
        slope_window=base.get("slope_window", 3),
        stable_m=base.get("stable_m", 3),
        tau=base.get("tau", 2.0),
        project_y=False,
        normalize_score=base.get("normalize_score", True),
    )
    print(f"[nuis] precomputing streaming stats "
          f"(n_cells={X.shape[0]}, G={X.shape[1]})...")
    t0 = time.time()
    pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
        X, V, pt,
        min_forward_dt=gcfg.min_forward_dt,
        max_forward_dt=gcfg.max_forward_dt,
        max_pairs_per_cell=gcfg.max_pairs_per_cell,
        pair_seed=gcfg.pair_seed,
        stream_num_threads=gcfg.stream_num_threads,
    )
    print(f"[nuis] streaming stats done in {time.time()-t0:.1f}s  "
          f"(n_pairs={pre['n_pairs']:,}, G={pre['G']})")

    out_dir = Path(args.out) / cfg["name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    d = V_nuis.shape[1]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for mode in modes:
        tag = (f"gb_nuis_none" if mode == "none"
               else f"gb_nuis_{mode}_d{d}")
        out_pq = out_dir / f"{tag}_{cfg['name']}.parquet"
        if out_pq.exists() and not args.force:
            print(f"[skip] {out_pq}")
            continue
        t0 = time.time()
        S = _scores_with_vnuis(pre, V_nuis, mode=mode, eps=gcfg.eps)
        df = _scores_to_long(np.asarray(S), genes, tf_set)
        save_ranking(df, out_pq, method=tag, dataset=cfg["name"])
        print(f"[ok]  {tag}  mode={mode}  {time.time()-t0:5.1f}s  "
              f"n_edges={len(df):,}")

    # Persist V_nuis and column names for reproducibility
    np.savez(out_dir / f"vnuis_{cfg['name']}.npz",
             V_nuis=V_nuis, cols=np.array(n_names))
    print(f"[nuis] done; parquets + V_nuis in {out_dir}")


if __name__ == "__main__":
    main()
