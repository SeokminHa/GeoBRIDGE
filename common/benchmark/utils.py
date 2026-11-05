"""
Shared utilities for Tier-1 GeoBRIDGE experiments (T1/T2/T4).

IMPORTANT design notes:

1. Simplified scoring formula
   `score_with_projection(F_X, F_V, P)` = correlation-normalized
   cross-covariance of pair-space features, after projecting F_V via P P.T.
   This is a simplified reimplementation of GeoBRIDGE-style scoring.
   Absolute numbers may not match production GeoBridgeModel, but both
   SVD and Random-projection variants use the same formula, so the
   ablation comparison is fair.

2. Pair sampling
   Uniform sample of `n_pairs` cell-pairs (i, j) with pt[i] < pt[j].
   For small datasets (all pairs fit), returns all pairs.

3. Projection matrices
   - SVD: P = top-k right singular vectors of F_V
   - Random: P = orthonormalized Gaussian (via QR)
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))


# ---------- Dataset paths ----------

SERGIO_BASE = Path("results/bm_velorama/velorama_data")
REAL_BASE = Path("results/bm_real_comparison/prepared_h5ad")

SERGIO_DATASETS = ["A", "B", "C", "D"]
REAL_DATASETS = ["bonemarrow", "pancreas"]

DATASET_ORGANISM = {
    "bonemarrow": "human",
    "pancreas": "mouse",
}


# ---------- Dataset loaders ----------

def load_sergio(name):
    """SERGIO dataset → (X, V, pt, gene_names, gt_dict)."""
    import scanpy as sc
    # Try both naming conventions
    candidates = [
        SERGIO_BASE / f"dataset_{name}.h5ad",
        SERGIO_BASE / f"test_{name}.h5ad",
    ]
    h5ad = None
    for cand in candidates:
        if cand.exists():
            h5ad = cand
            break
    if h5ad is None:
        raise FileNotFoundError(
            f"No SERGIO h5ad for '{name}'. Tried: "
            + ", ".join(str(c) for c in candidates)
        )
    adata = sc.read_h5ad(str(h5ad))

    # Compute velocity on-the-fly if missing (SERGIO h5ads typically have
    # spliced/unspliced only; velocity is computed downstream by each method)
    if "velocity" not in adata.layers:
        import scvelo as scv
        print(f"    [info] computing velocity for SERGIO {name}...")
        try:
            scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
            scv.tl.velocity(adata, mode="stochastic")
        except Exception as e:
            raise RuntimeError(f"Could not compute velocity for {name}: {e}")

    if "Ms" in adata.layers:
        X = np.asarray(adata.layers["Ms"], dtype=float)
    elif hasattr(adata.X, "toarray"):
        X = adata.X.toarray().astype(float)
    else:
        X = np.asarray(adata.X, dtype=float)

    V = np.nan_to_num(np.asarray(adata.layers["velocity"], dtype=float), nan=0.0)

    # Pseudotime: use existing or derive from DPT
    if "pseudotime" in adata.obs:
        pt = np.asarray(adata.obs["pseudotime"], dtype=float)
    elif "dpt_pseudotime" in adata.obs:
        pt = np.asarray(adata.obs["dpt_pseudotime"], dtype=float)
    else:
        # Derive DPT
        import scanpy as sc_lib
        try:
            if "X_pca" not in adata.obsm:
                sc_lib.pp.pca(adata, n_comps=30)
            if "neighbors" not in adata.uns:
                sc_lib.pp.neighbors(adata, n_pcs=30)
            sc_lib.tl.diffmap(adata)
            adata.uns["iroot"] = 0  # arbitrary root
            sc_lib.tl.dpt(adata)
            pt = np.asarray(adata.obs["dpt_pseudotime"], dtype=float)
        except Exception:
            pt = np.arange(adata.shape[0], dtype=float)
    pt = np.nan_to_num(pt, nan=float(np.nanmedian(pt)))

    gene_names = list(adata.var_names)

    gt_csv = SERGIO_BASE / "gt_GRN.csv"
    if gt_csv.exists():
        edges = pd.read_csv(gt_csv, header=None, names=["reg", "tgt"])
    else:
        # Fallback: try downloading
        import urllib.request
        url = ("https://raw.githubusercontent.com/PayamDiba/SERGIO/master/"
               "data_sets/De-noised_100G_6T_300cPerT_dynamics_7_DS6/gt_GRN.csv")
        gt_csv.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, gt_csv)
        edges = pd.read_csv(gt_csv, header=None, names=["reg", "tgt"])

    gt = {}
    for _, r in edges.iterrows():
        reg_s = str(int(r["reg"]))
        tgt_s = str(int(r["tgt"]))
        gt.setdefault(reg_s, set()).add(tgt_s)

    # Filter: TFs in gene_names with >= 5 valid targets
    filtered = {}
    for tf, tgts in gt.items():
        if tf not in gene_names:
            continue
        valid = {t for t in tgts if t in gene_names and t != tf}
        if len(valid) >= 5:
            filtered[tf] = valid
    return X, V, pt, gene_names, filtered


def load_real(name):
    """Real dataset (bonemarrow / pancreas) → (X, V, pt, gene_names, gt_dict)."""
    import scanpy as sc
    from benchmark.bm_real_data_comparison import (
        _load_collectri_regulons, _build_ground_truth,
    )

    h5ad = REAL_BASE / f"{name}.h5ad"
    adata = sc.read_h5ad(str(h5ad))

    X = np.asarray(
        adata.layers["Ms"] if "Ms" in adata.layers
        else (adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X),
        dtype=float,
    )
    V = np.nan_to_num(np.asarray(adata.layers["velocity"], dtype=float), nan=0.0)
    pt = np.asarray(adata.obs["pseudotime"], dtype=float)
    pt = np.nan_to_num(pt, nan=float(np.nanmedian(pt)))

    gene_names = list(adata.var_names)
    organism = DATASET_ORGANISM.get(name, "human")
    uppercase = (organism == "mouse")
    net = _load_collectri_regulons(organism=organism)
    gt_dict = _build_ground_truth(net, gene_names, uppercase_match=uppercase)
    return X, V, pt, gene_names, gt_dict


def load_dataset(name):
    if name in SERGIO_DATASETS:
        return load_sergio(name)
    if name in REAL_DATASETS:
        return load_real(name)
    raise ValueError(f"Unknown dataset: {name}")


def load_adata(name):
    """Raw AnnData for methods that need an AnnData (Scribe/Velorama)."""
    import scanpy as sc
    if name in SERGIO_DATASETS:
        for cand in [SERGIO_BASE / f"dataset_{name}.h5ad",
                     SERGIO_BASE / f"test_{name}.h5ad"]:
            if cand.exists():
                return sc.read_h5ad(str(cand))
        raise FileNotFoundError(f"No SERGIO h5ad for {name}")
    return sc.read_h5ad(str(REAL_BASE / f"{name}.h5ad"))


# ---------- Pair-space feature construction ----------

def sample_pair_features(X, V, pt, n_pairs, seed=42):
    """
    Sample n_pairs cell-pairs (i, j) with pt[i] < pt[j].

    Returns:
        F_X: (n_actual, G) — X[j] - X[i] stacked
        F_V: (n_actual, G) — V[j] - V[i] stacked
        pair_indices: (n_actual, 2) ordered pairs in pt-sorted coordinates
    """
    n = len(pt)
    order = np.argsort(pt)
    X_s, V_s = X[order], V[order]

    n_max = n * (n - 1) // 2
    rng = np.random.default_rng(seed)

    if n_pairs >= n_max:
        ii, jj = np.triu_indices(n, k=1)
    else:
        # Oversample-and-filter to get ~n_pairs pairs with i < j
        target = int(n_pairs * 2.5)
        i_cand = rng.integers(0, n, size=target)
        j_cand = rng.integers(0, n, size=target)
        mask = i_cand < j_cand
        ii, jj = i_cand[mask], j_cand[mask]
        if len(ii) >= n_pairs:
            ii, jj = ii[:n_pairs], jj[:n_pairs]

    F_X = X_s[jj] - X_s[ii]
    F_V = V_s[jj] - V_s[ii]
    pair_indices = np.column_stack([ii, jj])
    return F_X, F_V, pair_indices


# ---------- Projection matrices ----------

def projection_svd(F_V, k):
    """Top-k right singular vectors of F_V as (G × k) matrix."""
    G = F_V.shape[1]
    k = min(k, G)
    # Economy SVD
    _, _, Vt = np.linalg.svd(F_V, full_matrices=False)
    return Vt[:k].T


def projection_random(G, k, seed):
    """(G × k) matrix with orthonormal columns from Gaussian random matrix."""
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((G, max(k, 1)))
    Q, _ = np.linalg.qr(R)
    return Q[:, :k]


# ---------- Scoring ----------

def score_with_projection(F_X, F_V, P=None, normalize=True, eps=1e-12):
    """
    Compute absolute score matrix S of shape (G_X, G_V).

    If P is None: identity (no projection on F_V).
    Else: F_V is projected via P P.T (rank = P.shape[1]).

    Scoring = correlation-normalized cross-covariance in pair-space.
    """
    n = F_X.shape[0]

    if P is None:
        F_V_proj = F_V
    else:
        F_V_proj = (F_V @ P) @ P.T

    F_X_c = F_X - F_X.mean(axis=0, keepdims=True)
    F_V_c = F_V_proj - F_V_proj.mean(axis=0, keepdims=True)

    S = F_X_c.T @ F_V_c / max(n - 1, 1)

    if normalize:
        std_X = F_X_c.std(axis=0) + eps
        std_V = F_V_c.std(axis=0) + eps
        S = S / np.outer(std_X, std_V)

    return np.abs(S)


# ---------- Evaluation ----------

def evaluate_scores_per_tf(scores, gt_dict, gene_names):
    """
    Per-TF AUROC / AUPRC.

    scores[tf_idx, gene_idx] = prediction that gene is a target of tf.
    """
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    rows = []
    for tf, targets in gt_dict.items():
        # Try case variants for tf lookup
        tf_idx = None
        for cand in (tf, tf.title(), tf.upper(), tf.lower()):
            if cand in name_to_idx:
                tf_idx = name_to_idx[cand]
                break
        if tf_idx is None:
            continue

        s, l = [], []
        for j, g in enumerate(gene_names):
            if j == tf_idx:
                continue
            s.append(float(scores[tf_idx, j]))
            l.append(1 if (g in targets or g.upper() in targets) else 0)
        s = np.asarray(s); l = np.asarray(l)
        if l.sum() == 0 or l.sum() == len(l):
            continue
        rows.append({
            "tf": tf,
            "n_targets": int(l.sum()),
            "auroc": roc_auc_score(l, s),
            "auprc": average_precision_score(l, s),
        })
    return pd.DataFrame(rows)


def top_k_targets_per_tf(scores, gt_dict, gene_names, top_k=50):
    """Dict {tf: set(top_k gene names)}."""
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    out = {}
    for tf in gt_dict.keys():
        tf_idx = None
        for cand in (tf, tf.title(), tf.upper(), tf.lower()):
            if cand in name_to_idx:
                tf_idx = name_to_idx[cand]
                break
        if tf_idx is None:
            continue
        row = scores[tf_idx].copy()
        row[tf_idx] = -np.inf
        top_k_eff = min(top_k, len(row) - 1)
        top_part = np.argpartition(-row, top_k_eff)[:top_k_eff]
        top_sorted = top_part[np.argsort(-row[top_part])]
        out[tf] = set(gene_names[i] for i in top_sorted)
    return out


def pairwise_jaccard_per_tf(top_sets_list):
    """
    List of {tf: set} → {tf: mean pairwise Jaccard across the list}.
    """
    from itertools import combinations
    if not top_sets_list:
        return {}
    common = set(top_sets_list[0].keys())
    for ts in top_sets_list[1:]:
        common &= set(ts.keys())

    out = {}
    for tf in common:
        js = []
        for a, b in combinations(range(len(top_sets_list)), 2):
            sa, sb = top_sets_list[a][tf], top_sets_list[b][tf]
            if not sa and not sb:
                continue
            js.append(len(sa & sb) / max(len(sa | sb), 1))
        out[tf] = float(np.mean(js)) if js else np.nan
    return out
