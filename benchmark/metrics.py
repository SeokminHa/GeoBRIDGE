from collections import defaultdict

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

def flatten_scores(M):
    mask = ~np.eye(M.shape[0], dtype=bool)
    return M[mask]

def direct_labels(W):
    mask = ~np.eye(W.shape[0], dtype=bool)
    return (W != 0)[mask].astype(int)

def signed_labels(W):
    mask = ~np.eye(W.shape[0], dtype=bool)
    return np.sign(W[mask]).astype(int)

def signed_average_precision(y_sign, s_flat):
    y_pos = (y_sign == 1).astype(int)
    y_neg = (y_sign == -1).astype(int)
    ap_pos = float(average_precision_score(y_pos, s_flat)) if y_pos.sum() else np.nan
    ap_neg = float(average_precision_score(y_neg, -s_flat)) if y_neg.sum() else np.nan
    vals = [v for v in (ap_pos, ap_neg) if not np.isnan(v)]
    return float(np.mean(vals)) if vals else np.nan, ap_pos, ap_neg

def _adjacency_neighbors(W):
    """List of outgoing neighbor column indices per row (sparse adjacency)."""
    G = W.shape[0]
    return [np.flatnonzero(W[i] != 0) for i in range(G)]


def _indirect_reachable_from_src(W, src, dst_candidates, neighbors, max_hops=4):
    """
    Among dst in dst_candidates with W[src,dst]==0, return those reachable from src
    in <= max_hops along W!=0 edges (same logic as the original per-edge BFS).
    """
    targets = {d for d in dst_candidates if W[src, d] == 0}
    if not targets:
        return set()
    visited = {src}
    frontier = {src}
    found = set()
    for _ in range(max_hops):
        new_frontier = set()
        for u in frontier:
            for v in neighbors[u]:
                if v in targets:
                    found.add(v)
                if v not in visited:
                    visited.add(v)
                    new_frontier.add(v)
        frontier = new_frontier
        if not frontier or found >= targets:
            break
    return found


def edp_at_k(W, flat_scores, topk=50, max_hops=4):
    G = W.shape[0]
    mask = ~np.eye(G, dtype=bool)
    rows, cols = np.where(mask)
    order = np.argsort(-np.abs(flat_scores))[:topk]
    neighbors = _adjacency_neighbors(W)
    # Group (i,j) queries by source i; one BFS per unique i in the top-k list.
    by_src = defaultdict(list)
    for idx in order:
        by_src[rows[idx]].append((idx, cols[idx]))
    indirect_ok = {}
    for src, pairs in by_src.items():
        jset = [j for _, j in pairs]
        reachable = _indirect_reachable_from_src(
            W, src, jset, neighbors, max_hops=max_hops
        )
        for _, j in pairs:
            indirect_ok[(src, j)] = j in reachable

    direct_hits = 0
    true_hits = 0
    for idx in order:
        i = rows[idx]
        j = cols[idx]
        if W[i, j] != 0:
            true_hits += 1
            direct_hits += 1
        elif indirect_ok.get((i, j), False):
            true_hits += 1
    return direct_hits / max(true_hits, 1)

def evaluate_all_metrics(W, scores, topk=50):
    flat = flatten_scores(scores)
    dlab = direct_labels(W)
    slab = signed_labels(W)
    sap, ap_pos, ap_neg = signed_average_precision(slab, flat)
    return {
        "AUPRC": float(average_precision_score(dlab, np.abs(flat))),
        "S-AUPRC": sap,
        "AUROC": float(roc_auc_score(dlab, np.abs(flat))),
        "EDP@50": float(edp_at_k(W, flat, topk=topk)),
        "AP_Pos": ap_pos,
        "AP_Neg": ap_neg,
    }
