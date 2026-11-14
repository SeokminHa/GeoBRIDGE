# Vectorised U-V paired permutation null
import numpy as np


# Order-dependent structure of one score vector; label-free
def _prep(s):
    n = len(s)
    o = np.argsort(-s, kind="mergesort")               # same stable sort the scalar code uses
    ss = s[o]
    newgrp = np.r_[True, ss[1:] != ss[:-1]]
    end = np.r_[newgrp[1:], True]
    end_shift = np.r_[False, end[:-1]]
    pos = np.arange(n)
    last_prev = np.maximum.accumulate(np.where(end_shift, pos - 1, -1))
    return dict(o=o, ss=ss, end=end, last_prev=last_prev, n=n,
                idx=np.arange(1, n + 1, dtype=np.float64))


# nAP for every row of a (P, n) label matrix against one fixed score ordering
def _nap_many(prep, Y):
    Ys = Y[:, prep["o"]].astype(np.float64)
    npos = Ys.sum(1)
    cum = np.cumsum(Ys, axis=1)
    end, lp, idx, n = prep["end"], prep["last_prev"], prep["idx"], prep["n"]
    T_prev = np.where(lp >= 0, cum[:, np.maximum(lp, 0)], 0.0)
    h = np.where(end, cum - T_prev, 0.0)
    term = h * np.where(end, cum / idx, 0.0)
    ap = np.cumsum(term, axis=1)[:, -1] / np.maximum(npos, 1)
    prev = Ys.mean(1)                      # same expression as the scalar y.mean()
    out = ap / np.where(prev > 0, prev, np.nan)
    out[(npos == 0) | (npos == n)] = np.nan
    return out


# Fractional expected P@N (uniform random tie-breaking) for every row
def _pk_exp_many(prep, Y, N):
    ss = prep["ss"]; n = prep["n"]
    if n < N:
        return np.full(Y.shape[0], np.nan)
    Ys = Y[:, prep["o"]].astype(np.float64)
    cut = ss[N - 1]; above = ss > cut; tied = ss == cut
    n_above = int(above.sum()); n_tied = int(tied.sum()); slots = N - n_above
    return (Ys[:, above].sum(1) + Ys[:, tied].sum(1) * slots / n_tied) / N


# Fractional expected gain@q for every row (tot = row positives, as in topk)
def _gain_exp_many(prep, Y, q):
    ss = prep["ss"]; n = prep["n"]
    Ys = Y[:, prep["o"]].astype(np.float64)
    m = max(int(round(q * n)), 1)
    cq = ss[m - 1]; aq = ss > cq; tq = ss == cq
    tot = np.maximum(Ys.sum(1), 1.0)
    return (Ys[:, aq].sum(1) + Ys[:, tq].sum(1) * (m - int(aq.sum())) / int(tq.sum())) / tot


METRICS = (("nAP", lambda p, Y: _nap_many(p, Y)),
           ("P@20_exp", lambda p, Y: _pk_exp_many(p, Y, 20)),
           ("gain@5pct_exp", lambda p, Y: _gain_exp_many(p, Y, 0.05)))


# Returns {metric: (T_U, T_V, dn)} with dn the (n_perm,) vector of permuted U-V differences
def uv_perm_null(y, u, v, n_perm, seed):
    rg = np.random.default_rng(seed)
    Yp = np.stack([rg.permutation(y) for _ in range(n_perm)])
    y1 = np.asarray(y)[None, :]
    pu, pv = _prep(np.asarray(u, float)), _prep(np.asarray(v, float))
    out = {}
    for name, fn in METRICS:
        tu = float(fn(pu, y1)[0]); tv = float(fn(pv, y1)[0])
        dn = fn(pu, Yp) - fn(pv, Yp)
        out[name] = (tu, tv, dn)
    return out
