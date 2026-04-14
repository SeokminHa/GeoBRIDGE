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


def _out_strength(S: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(S), axis=1)


def _is_true_regulator_row(W: np.ndarray, i: int, eps: float = 1e-12) -> bool:
    G = W.shape[0]
    row = np.abs(W[i, :])
    row[i] = 0.0
    return bool(np.sum(row) > eps)


def hub_trap_rate_at_k(W: np.ndarray, scores: np.ndarray, ks: tuple[int, ...] = (10, 20, 50)) -> dict[str, float]:
    """Fraction of top-K regulator rows (by out-strength) that are not true regulators in W."""
    st = _out_strength(scores)
    order = np.argsort(-st)
    out: dict[str, float] = {}
    G = W.shape[0]
    for k in ks:
        kk = min(k, G)
        top = order[:kk]
        traps = sum(1 for i in top if not _is_true_regulator_row(W, int(i)))
        out[f"hub_trap_rate_at_{k}"] = float(traps / max(kk, 1))
    return out


def pseudo_hub_hits_at_k(
    scores: np.ndarray,
    pseudo_hub_indices: list[int],
    ks: tuple[int, ...] = (10, 20, 50),
) -> dict[str, float]:
    """Among top-K rows by |S| out-strength, fraction that are listed pseudo-hubs (false traps)."""
    if not pseudo_hub_indices:
        return {f"pseudo_hub_fdr_at_{k}": float("nan") for k in ks}
    ph_set = set(int(x) for x in pseudo_hub_indices)
    st = _out_strength(scores)
    order = np.argsort(-st)
    G = scores.shape[0]
    out: dict[str, float] = {}
    for k in ks:
        kk = min(k, G)
        top = set(int(x) for x in order[:kk])
        hits = len(top & ph_set)
        out[f"pseudo_hub_fdr_at_{k}"] = float(hits / max(kk, 1))
    return out


def target_precision_at_k(W: np.ndarray, scores: np.ndarray, ks: tuple[int, ...] = (20, 50, 100)) -> dict[str, float]:
    """
    Macro-averaged precision@K: for each true-regulator row, rank |S_ij| among j!=i,
    take precision in top-K vs true targets in that row; mean over rows with targets.
    """
    G = W.shape[0]
    precs = {k: [] for k in ks}
    for i in range(G):
        true_j = np.where(np.abs(W[i, :]) > 1e-12)[0]
        true_j = [int(j) for j in true_j if j != i]
        if not true_j:
            continue
        row = np.abs(scores[i, :].copy())
        row[i] = -1.0
        order = np.argsort(-row)
        rank_pos = {int(j): r for r, j in enumerate(order)}
        for k in ks:
            kk = min(k, G - 1)
            top = set(int(x) for x in order[:kk])
            hit = len(top & set(true_j))
            precs[k].append(hit / max(kk, 1))
    out = {}
    for k in ks:
        arr = precs[k]
        out[f"target_precision_at_{k}"] = float(np.mean(arr)) if arr else float("nan")
    return out


def median_target_percentile_rank(W: np.ndarray, scores: np.ndarray) -> float:
    """Median percentile rank (0–100, lower=better) of each true edge within its row."""
    G = W.shape[0]
    ranks: list[float] = []
    for i in range(G):
        row = np.abs(scores[i, :].copy())
        row[i] = -np.inf
        order = np.argsort(-row)
        pos = np.empty(G, dtype=int)
        pos[order] = np.arange(G)
        for j in range(G):
            if i == j or abs(W[i, j]) <= 1e-12:
                continue
            # rank pos[j] among off-diagonal: 0 = top
            others = G - 1
            pct = 100.0 * float(pos[j]) / max(others, 1)
            ranks.append(pct)
    return float(np.median(ranks)) if ranks else float("nan")


def signal_preservation_corr(W: np.ndarray, scores: np.ndarray) -> float:
    """Correlation |S| vs |W| on off-diagonal pairs."""
    G = W.shape[0]
    mask = ~np.eye(G, dtype=bool)
    wv = np.abs(W[mask])
    sv = np.abs(scores[mask])
    if np.std(sv) < 1e-12 or np.std(wv) < 1e-12:
        return float("nan")
    return float(np.corrcoef(wv, sv)[0, 1])


def spurious_mass_ratio(W: np.ndarray, scores: np.ndarray) -> float:
    """Fraction of total off-diagonal |S| mass on W==0 pairs (confound / spurious proxy)."""
    mask = ~np.eye(W.shape[0], dtype=bool)
    S = np.abs(scores[mask])
    Z = (np.abs(W[mask]) <= 1e-12).astype(float)
    num = float(np.sum(S * Z))
    den = float(np.sum(S) + 1e-12)
    return num / den


def residual_confound_energy(
    W: np.ndarray,
    scores: np.ndarray,
    X: np.ndarray,
    pseudotime: np.ndarray,
) -> float:
    """
    R² of linear model predicting |S_ij| from low-rank confound features of (i,j,pt):
    uses mean cell expression per gene and pseudotime bin interaction as cheap proxy.
    """
    from sklearn.linear_model import Ridge

    G = W.shape[0]
    n = X.shape[0]
    mask = ~np.eye(G, dtype=bool)
    rows, cols = np.where(mask)
    y = np.abs(scores[rows, cols])
    if y.size < 50:
        return float("nan")
    mu_x = X.mean(axis=0)
    pt_n = (pseudotime - pseudotime.mean()) / (pseudotime.std() + 1e-12)
    # features: mu_x[i], mu_x[j], pt_mean (scalar), product
    fi = mu_x[rows]
    fj = mu_x[cols]
    pm = float(np.mean(pt_n))
    Q = np.column_stack([fi, fj, np.full(len(rows), pm), fi * fj])
    try:
        m = Ridge(alpha=1.0).fit(Q, y)
        return float(m.score(Q, y))
    except Exception:
        return float("nan")


def evaluate_deconfounding_metrics(
    W: np.ndarray,
    scores: np.ndarray,
    X: np.ndarray,
    V: np.ndarray,
    pseudotime: np.ndarray,
    *,
    pseudo_hub_indices: list | None = None,
) -> dict:
    out = {}
    out.update(hub_trap_rate_at_k(W, scores))
    out.update(pseudo_hub_hits_at_k(scores, pseudo_hub_indices or []))
    out.update(target_precision_at_k(W, scores))
    out["median_target_pct_rank"] = median_target_percentile_rank(W, scores)
    out["signal_preservation_corr"] = signal_preservation_corr(W, scores)
    out["spurious_mass_ratio"] = spurious_mass_ratio(W, scores)
    out["residual_confound_r2"] = residual_confound_energy(W, scores, X, pseudotime)
    return out
