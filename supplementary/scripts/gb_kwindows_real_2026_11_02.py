"""Real-data sensitivity of the frozen depth window (PI 2026-11-02): GeoBRIDGE PRIMARY recomputed over alternative depth windows on the Fig 3b/c KO endpoints.
Reuses scratch/gb_fig3d_depth_multi_2026_10_29.py verbatim (rebuild prefix, rows, truth, synchronized module draws, parity cum0-8 == D and cum0-0 == K0), only
replacing the subset list and contrasts. Candidates = rules that were live during development: Z-0:8 (PRIMARY), Z-1:8 (§234 alternative), percentile-mean 1..8
(F8 freeze 2026-09-26, §218), neighbour windows 0..6/0..9/0..10/0..12 (§234 sensitivity list + dev best 0..9), 2..8, 0..4, 0..16, 0..30, K0.
Every contrast is (window - cum0-8) on the same module draws. DESCRIPTIVE ONLY: the frozen window is not changed by this analysis.
usage (server, repo root): script OUT --endpoints A[,B] [--B 9999]"""
import sys
from scipy.stats import rankdata
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
src = open(R + 'scratch/gb_fig3d_depth_multi_2026_10_29.py').read().replace('\r', '')
W = {'cum0-0': range(0, 1), 'cum0-4': range(0, 5), 'cum0-6': range(0, 7), 'cum0-8': range(0, 9), 'cum0-9': range(0, 10), 'cum0-10': range(0, 11),
     'cum0-12': range(0, 13), 'cum0-16': range(0, 17), 'cum0-30': range(0, 31), 'w1-8': range(1, 9), 'w2-8': range(2, 9), 'w1-10': range(1, 11)}
old_subs = src[src.index('SUBS = {'):src.index("PAIRS = [")]; src = src.replace(old_subs, 'SUBS = ' + repr({k: list(v) for k, v in W.items()}) + '\n')
old_pairs = src[src.index('PAIRS = ['):src.index('\n', src.index('PAIRS = ['))]
src = src.replace(old_pairs, 'PAIRS = ' + repr([(k, 'cum0-8') for k in list(W) + ['pct1-8'] if k != 'cum0-8']))
def pcons(per_root, ks):   # F8 freeze rule: within-source percentile per depth, mean over ks (>= 2/3 finite), min over roots
    import numpy as np
    per = []
    for h in ('dc1_min', 'dc1_max'):
        P = []
        for k in ks:
            x = per_root[h][:, k]; ok = np.isfinite(x); p = np.full(len(x), np.nan)
            if ok.sum() >= 3 and len(np.unique(x[ok])) >= 3: p[ok] = rankdata(x[ok]) / ok.sum()
            P.append(p)
        P = np.stack(P); n_ok = np.isfinite(P).sum(0)
        with np.errstate(invalid='ignore'): m = np.nanmean(P, 0)
        m[n_ok < np.ceil(2 * len(ks) / 3)] = np.nan; per.append(m)
    return np.minimum(*per)
old_v = 'V = {nm: sub(ks) for nm, ks in SUBS.items()}'; assert old_v in src
src = src.replace(old_v, old_v + "\n    _f = np.full(len(genes), np.nan); _f[uni] = pcons({r: A[r][uni] for r in GP.ROOTS}, list(range(1, 9))); V['pct1-8'] = mapto(genes, _f, g)")
assert "('cum0-8', 'late4')" not in src and 'early4' not in src.split('SUBS = ')[1].split('\n')[0]
exec(compile(src, 'depth_multi', 'exec'))
