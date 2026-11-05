import numpy as np

def dual_project_pairspace(DX, DV, U, S, Vt, k):
    if k <= 0:
        return DX.copy(), DV.copy()
    U_k = U[:, :k]
    S_k = S[:k]
    Vt_k = Vt[:k, :]
    geom_x = (U_k * S_k[np.newaxis, :]) @ Vt_k
    DX_res = DX - geom_x
    DV_res = DV - U_k @ (U_k.T @ DV)
    return DX_res, DV_res

def score_matrix(DX_res, DV_res, normalize=True, eps=1e-12, method="concordance"):
    """
    Gene-by-gene score matrix S[i,j] = f(DX[:,i], DV[:,j]).

    concordance (default): S = (DX^T DV) / (|DX|^T |DV| + eps)
    pearson:               S = (DX^T DV) / (||DX|| ||DV|| + eps)
    """
    num = DX_res.T @ DV_res
    if not normalize:
        return num
    if method == "concordance":
        den = np.abs(DX_res).T @ np.abs(DV_res)
        return num / np.maximum(den, eps)
    else:
        x_norm = np.linalg.norm(DX_res, axis=0)[:, None]
        y_norm = np.linalg.norm(DV_res, axis=0)[None, :]
        return num / np.maximum(x_norm * y_norm, eps)
