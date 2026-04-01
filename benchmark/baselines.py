import numpy as np
from sklearn.decomposition import PCA
from geobridge_model.pairspace import build_forward_pairs
from geobridge_model.scoring import score_matrix, dual_project_pairspace

def normalize_cols(M, eps=1e-12):
    return M / np.maximum(np.linalg.norm(M, axis=0, keepdims=True), eps)

def method_pairwise_raw(X, V, pseudotime):
    pair_data = build_forward_pairs(X, V, pseudotime)
    return score_matrix(pair_data.delta_x, pair_data.delta_v, normalize=True), 0

def method_corr(X, V, pseudotime):
    Xc = normalize_cols(X - X.mean(axis=0, keepdims=True))
    Vc = normalize_cols(V - V.mean(axis=0, keepdims=True))
    return Xc.T @ Vc, 0

def method_ridge(X, V, pseudotime, alpha=1e-2):
    Xc = X - X.mean(axis=0, keepdims=True)
    W = np.linalg.solve(Xc.T @ Xc + alpha * np.eye(X.shape[1]), Xc.T @ V)
    return W, 0

def method_cell_pca_preresidualize(X, V, pseudotime, k=3):
    Xc = X - X.mean(axis=0, keepdims=True)
    basis = PCA(n_components=min(k, X.shape[1], X.shape[0]), svd_solver="full").fit(Xc).components_.T
    Xr = X - X @ basis @ basis.T
    Vr = V - V @ basis @ basis.T
    pair_data = build_forward_pairs(Xr, Vr, pseudotime)
    return score_matrix(pair_data.delta_x, pair_data.delta_v, normalize=True), k

def method_pair_pca_direct(X, V, pseudotime, k=3):
    pair_data = build_forward_pairs(X, V, pseudotime)
    DX, DV = pair_data.delta_x, pair_data.delta_v
    DXc = DX - DX.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(DXc, full_matrices=False)
    kk = min(k, max(0, len(S) - 1))
    DX_res, DV_res = dual_project_pairspace(DXc, DV, U, S, Vt, kk)
    return score_matrix(DX_res, DV_res, normalize=True), kk
