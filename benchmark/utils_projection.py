"""
Production-faithful helpers for ablation inside GeoBridgeModel.

Key insight from reading geobridge_model/model.py and scoring.py:

  Production GB at rank k uses `Vk = pre["Vt"][:k].T` (G×k top-k right
  singular vectors of ΔXc) and subtracts its subspace from ΔX and (if
  project_y=True) from ΔV. It does NOT "project onto" top-k — it
  "projects out" top-k.

For a production-faithful random-projection ablation, we:
  1. Compute pre via the production streaming pipeline (unchanged).
  2. Replace Vk with a random orthonormal G×k matrix.
  3. Apply the EXACT same scoring expression, but use project_y=False
     (because random basis has no associated singular values to sharpen
     the y-normalization).

We verify that our custom-basis scoring reproduces production exactly
when Vk = pre["Vt"][:k].T and project_y=False (see `verify_match`).

Assumptions:
  - pre exposes keys: "C", "M", "Y", "S", "Vt", "G"
    (confirmed from reading geobridge_model/model.py)
  - pre["Vt"] has shape (G, G) with ΔXc right singular vectors in rows
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from geobridge_model.model import GeoBridgeModel


# -------------------------------------------------------------------------
# Custom-basis scoring
# -------------------------------------------------------------------------

def scores_with_custom_vk(pre, Vk, normalize_score=True, eps=1e-12):
    """
    Production-faithful scoring with a custom (G, k) projection basis.

    This exactly replicates `scores_from_streaming_stats(pre, k, project_y=False,
    normalize_score=normalize_score)` when Vk == pre["Vt"][:k].T.

    Args:
        pre: dict from `precompute_pairspace_stats_streaming`, must contain
             C (G,G), M (G,G), Y (G,G), G (int).
        Vk:  (G, k) matrix with orthonormal columns. If k=0, pass Vk=None.
        normalize_score: bool; divide num by sqrt(x_norm2) * sqrt(y_norm2).
        eps: numerical floor for the denominator.
    Returns:
        scores: (G, G) matrix; scores[a, b] = standardized residual
                cross-covariance of gene a (on ΔX-side) with gene b (on ΔV-side).
    """
    C = pre["C"]
    M = pre["M"]
    Y = pre["Y"]
    G = int(pre["G"])

    if Vk is None or Vk.size == 0:
        # k = 0: no projection
        num = M
        x_norm2 = np.clip(np.diag(C), 0.0, None)
        y_norm2 = np.clip(np.diag(Y), 0.0, None)
    else:
        if Vk.shape[0] != G:
            raise ValueError(f"Vk has shape {Vk.shape}; expected (G={G}, k)")
        Pk = Vk @ Vk.T
        Q = np.eye(G) - Pk
        num = Q @ M
        x_norm2 = np.clip(np.diag(Q @ C @ Q), 0.0, None)
        # project_y=False: use raw diag(Y) (no y-subspace removal)
        y_norm2 = np.clip(np.diag(Y), 0.0, None)

    if not normalize_score:
        return num
    denom = np.maximum(
        np.sqrt(x_norm2)[:, None] * np.sqrt(y_norm2)[None, :], eps,
    )
    return num / denom


def random_orthonormal_vk(G, k, seed):
    """Random (G, k) with orthonormal columns via QR decomposition."""
    if k <= 0:
        return None
    rng = np.random.default_rng(seed)
    R = rng.standard_normal((G, max(k, 1)))
    Q, _ = np.linalg.qr(R)
    return Q[:, :k]


def svd_vk(pre, k):
    """Top-k right singular vectors of ΔXc from production pre (G, k)."""
    if k <= 0:
        return None
    Vt = pre["Vt"]
    G = int(pre["G"])
    k = min(k, G)
    return Vt[:k].T


# -------------------------------------------------------------------------
# Verification: match production scoring exactly
# -------------------------------------------------------------------------

def verify_match(pre, k=5, tol=1e-10):
    """
    Sanity check: our custom-basis scoring with Vk = pre["Vt"][:k].T
    should exactly equal `scores_from_streaming_stats(pre, k, project_y=False)`.

    Raises AssertionError on mismatch. Call once per codepath change.
    """
    Vk = svd_vk(pre, k)
    ours = scores_with_custom_vk(pre, Vk, normalize_score=True)
    prod = GeoBridgeModel.scores_from_streaming_stats(
        pre, k=k, project_y=False, normalize_score=True,
    )
    diff = np.max(np.abs(ours - prod))
    assert diff < tol, f"Custom-basis scoring differs from production: max |diff|={diff:.2e}"
    return diff


# -------------------------------------------------------------------------
# Data loading and evaluation (reuse benchmark/utils where possible)
# -------------------------------------------------------------------------

def load_for_production(name):
    """
    Load a dataset and return everything needed for production GB + eval.

    Returns:
        X, V, pt, gene_names, gt_dict  (as in benchmark/utils.load_dataset)
    """
    from benchmark.utils import load_dataset
    return load_dataset(name)


def precompute_pre(X, V, pt, pair_seed=42):
    """
    Call production `precompute_pairspace_stats_streaming`. Sort by pt.
    """
    order = np.argsort(pt)
    X_s = np.ascontiguousarray(X[order], dtype=float)
    V_s = np.ascontiguousarray(V[order], dtype=float)
    pt_s = np.ascontiguousarray(pt[order], dtype=float)

    pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
        X_s, V_s, pt_s, pair_seed=pair_seed,
    )
    return pre


def evaluate_scores_per_tf(scores, gt_dict, gene_names, top_k_values=(10, 20, 50)):
    """
    Per-TF AUROC / AUPRC / EPR / top-K precision, plus signed-AUPRC metrics
    when gt_dict carries sign information.

    gt_dict accepted shapes:
      - {tf: set(target_names)}               -- unsigned (all +1)
      - {tf: {target_name: sign ∈ {-1,0,1}}}  -- signed

    Unsigned metrics (always computed, based on |score|):
      auroc, auprc, sauprc (= AUPRC / baseline), epr, topk_precision_K.

    Signed metrics (computed only when gt_dict has sign information):
      auroc_signed_pos, auprc_signed_pos
        -- positive edges vs rest-excluding-negatives; score = +signed_score.
      auroc_signed_neg, auprc_signed_neg
        -- negative edges vs rest-excluding-positives; score = -signed_score.
      dir_acc
        -- among GT edges with nonzero sign, fraction where sign(score) == sign(GT).
    """
    from sklearn.metrics import roc_auc_score, average_precision_score
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    rows = []
    for tf, targets in gt_dict.items():
        # Accept both unsigned (set) and signed (dict)
        if isinstance(targets, dict):
            signed_targets = targets
        else:
            signed_targets = {t: 1 for t in targets}

        tf_idx = None
        for cand in (tf, tf.title(), tf.upper(), tf.lower()):
            if cand in name_to_idx:
                tf_idx = name_to_idx[cand]
                break
        if tf_idx is None:
            continue

        # Build three parallel arrays over candidate genes (excluding TF itself)
        signed_s, mag_s, l, sign_l = [], [], [], []
        for j, g in enumerate(gene_names):
            if j == tf_idx:
                continue
            raw = float(scores[tf_idx, j])
            signed_s.append(raw)
            mag_s.append(abs(raw))
            # Unsigned label: 1 if any-sign edge in GT
            is_target = (g in signed_targets) or (g.upper() in signed_targets)
            l.append(1 if is_target else 0)
            # Signed label: +1 / -1 / 0 (non-edge = 0)
            if g in signed_targets:
                sign_l.append(signed_targets[g])
            elif g.upper() in signed_targets:
                sign_l.append(signed_targets[g.upper()])
            else:
                sign_l.append(0)

        signed_s = np.asarray(signed_s)
        mag_s = np.asarray(mag_s)
        l = np.asarray(l)
        sign_l = np.asarray(sign_l)

        if l.sum() == 0 or l.sum() == len(l):
            continue

        baseline = float(l.mean())
        order = np.argsort(-mag_s)
        n_pos = int(l.sum())

        auprc_val = average_precision_score(l, mag_s)
        row = {
            "tf": tf,
            "n_targets": n_pos,
            "n_candidates": int(len(l)),
            "auroc": roc_auc_score(l, mag_s),
            "auprc": auprc_val,
            "baseline_auprc": baseline,
            "sauprc": auprc_val / max(baseline, 1e-12),
        }
        if n_pos > 0 and n_pos <= len(l):
            prec_at_npos = float(l[order[:n_pos]].mean())
            row["epr"] = prec_at_npos / max(baseline, 1e-12)
        for K in top_k_values:
            if K <= len(l):
                row[f"topk_precision_{K}"] = float(l[order[:K]].mean())

        # --- signed metrics (only if GT has sign information) ---
        n_pos_edges = int((sign_l == 1).sum())
        n_neg_edges = int((sign_l == -1).sum())
        has_sign_info = (n_pos_edges + n_neg_edges) > 0

        if has_sign_info:
            # Positive-edge AUPRC: positive edges are positive class, negative
            # edges EXCLUDED from eval (ambiguous), non-edges = negative class.
            pos_mask = sign_l != -1  # keep non-edges + positive edges
            pos_labels = (sign_l[pos_mask] == 1).astype(int)
            pos_scores = signed_s[pos_mask]
            if pos_labels.sum() > 0 and pos_labels.sum() < len(pos_labels):
                row["auroc_signed_pos"] = roc_auc_score(pos_labels, pos_scores)
                row["auprc_signed_pos"] = average_precision_score(pos_labels, pos_scores)
                pos_baseline = float(pos_labels.mean())
                row["sauprc_signed_pos"] = (
                    row["auprc_signed_pos"] / max(pos_baseline, 1e-12)
                )

            # Negative-edge AUPRC: negative edges are positive class (with
            # score = -signed_score so that expected direction is ↑).
            neg_mask = sign_l != 1
            neg_labels = (sign_l[neg_mask] == -1).astype(int)
            neg_scores = -signed_s[neg_mask]
            if neg_labels.sum() > 0 and neg_labels.sum() < len(neg_labels):
                row["auroc_signed_neg"] = roc_auc_score(neg_labels, neg_scores)
                row["auprc_signed_neg"] = average_precision_score(neg_labels, neg_scores)
                neg_baseline = float(neg_labels.mean())
                row["sauprc_signed_neg"] = (
                    row["auprc_signed_neg"] / max(neg_baseline, 1e-12)
                )

            # Direction accuracy: fraction of signed GT edges where predicted
            # sign matches GT sign (ignoring zero-sign GT entries).
            nonzero_edge_mask = (sign_l == 1) | (sign_l == -1)
            if nonzero_edge_mask.any():
                sign_pred = np.sign(signed_s[nonzero_edge_mask])
                sign_gt = sign_l[nonzero_edge_mask]
                # Count pred==0 as wrong (no prediction)
                row["dir_acc"] = float((sign_pred == sign_gt).mean())
                row["n_pos_edges"] = n_pos_edges
                row["n_neg_edges"] = n_neg_edges

        rows.append(row)
    return pd.DataFrame(rows)
