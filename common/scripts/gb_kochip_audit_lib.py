"""KOCHIP_AUDIT_2026_10_15 shared library (EVALUATION ONLY; saved scores + truth labels; KO/ChIP labels never enter a score).
Spec: internal design note (not shipped).  PRELIMINARY / exploratory."""
import numpy as np, pandas as pd, os
from scipy.stats import mannwhitneyu, norm

R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); SG = R + 'results/signed/'
DUMP = SG + 'grand_v3d_dumpT_2026_10_15/parts/'; IP = SG + 'iso_pilot_2026_10_15/'; CHP = SG + 'chip5_sens_2026_10_15/parts/'; BR = SG + 'beta_rp_2026_10_15/'
OUT = SG + 'kochip_audit_2026_10_15/'; FIGDIR = R + 'results/figures/meeting_2026_10_15/'
SEED = 20261015
# unit -> (system, focal, iso-scores dir, iso A/B dir, display name, group)
UNITS = {
    'CARDIACV_TBX5': ('CARDIACV', 'TBX5', 'CARDIACV_TBX5', 'CARDIACV_TBX5', 'Heart · TBX5', 'Heart'),
    'CARDIACV_NKX2-5': ('CARDIACV', 'NKX2-5', 'CARDIACV_NKX2-5', 'CARDIACV_NKX2-5', 'Heart · NKX2-5', 'Heart'),
    'GATA1_GATA1': ('GATA1', 'GATA1', 'GATA1_GATA1', 'GATA1_GATA1', 'Erythroid · GATA1', 'Erythroid'),
    'KLF1_KLF1': ('GATA1', 'KLF1', 'GATA1_KLF1', 'GATA1_KLF1', 'Erythroid · KLF1 (GSE94351)', 'Erythroid'),
    'KLF1_B_KLF1': ('GATA1', 'KLF1', 'GATA1_KLF1', 'GATA1_KLF1', 'Erythroid · KLF1 (GSE33979)', 'Erythroid'),
    'PANC_RFX6': ('PANC', 'RFX6', 'PANC_RFX6', 'PANC_RFX6', 'Pancreas · RFX6', 'Pancreas'),
}
# five ordered levels (expected score order L1 > L2 > L3 > L4 > L5)
LV = [('L1', ['A_act'], 'ChIP+ KO-down'), ('L2', ['C_act'], 'ChIP- KO-down'), ('L3', ['B', 'D'], 'non-responder'), ('L4', ['C_rep'], 'ChIP- KO-up'), ('L5', ['A_rep'], 'ChIP+ KO-up')]
CLS2LVL = {c: i for i, (_, cs, _) in enumerate(LV) for c in cs}
PAIRS = [(0, 1, 'greater'), (3, 4, 'greater'), (0, 2, 'greater'), (2, 3, 'greater'), (1, 2, 'greater'), (4, 2, 'less')]   # == gb_fig5_signed.py PAIRS


def star(p): return '***' if p < 1e-3 else '**' if p < 1e-2 else '*' if p < .05 else None


def holm(ps):   # verbatim gb_fig5_signed.py
    ps = np.asarray(ps, float); o = np.argsort(np.where(np.isfinite(ps), ps, 9)); adj = np.full(len(ps), np.nan); run = 0; n = np.isfinite(ps).sum()
    for k, i in enumerate(o):
        if np.isfinite(ps[i]): run = max(run, min(1, ps[i] * (n - k))); adj[i] = run
    return adj


def load_unit(u):
    sysn, foc, sd, ab, name, grp = UNITS[u]
    T = pd.read_csv(DUMP + f'T_{u}.tsv', sep='\t'); ft = pd.read_csv(IP + f'scores/{sd}/features_{u}.tsv', sep='\t')
    lab = pd.read_csv(CHP + f'{u}_labels.tsv', sep='\t'); rp = pd.read_csv(BR + f'rp_{u}.tsv', sep='\t')
    assert T.gene.is_unique and ft.gene.is_unique and lab.gene.is_unique and rp.gene.is_unique
    assert set(T.gene) == set(ft.gene) == set(lab.gene) == set(rp.gene), u
    keep = ['gene', 'cls'] + [c for c in T.columns if c.startswith(('Z|', 'extmag|', 'absPear'))]
    df = T[keep].merge(ft.drop(columns=['F']), on='gene', how='left', validate='1:1').merge(ft[['gene', 'F']], on='gene', how='left', validate='1:1')
    df = df.merge(lab[['gene', 'c6', 'log2fc', 'padj', 'dist', 'occ_ok']], on='gene', how='left', validate='1:1').merge(rp[['gene', 'RP', 'n_peaks_100kb']], on='gene', how='left', validate='1:1')
    assert len(df) == len(T) and (df.cls == df.c6).all()
    assert np.allclose(df.F, np.abs(df['Z|d4']), equal_nan=True), 'F != |Z|d4|'
    df['T_iso'] = np.sign(df.iso_C) * np.abs(df['Z|d4'])          # EXACTLY gb_fig5_d4_vs_iso.py
    df['lvl'] = df.cls.map(CLS2LVL).fillna(-1).astype(int)
    df['gene_up'] = df.gene.astype(str).str.upper()
    return df


# ---------------------------------------------------------------- statistics
def _auc_sorted(x, ys):   # P(x>y)+.5P(x=y); ys sorted
    if len(x) == 0 or len(ys) == 0: return np.nan
    l = np.searchsorted(ys, x, 'left'); r = np.searchsorted(ys, x, 'right'); return (l + 0.5 * (r - l)).sum() / (len(x) * len(ys))


def auc(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    return _auc_sorted(x, np.sort(y)) if len(x) and len(y) else np.nan


def ordinal(s, g, K=5):
    """c = sum_{i<j} n_i n_j AUC(i>j) / sum n_i n_j over groups 0..K-1 (0 expected highest). Returns c, U dict, n."""
    ok = np.isfinite(s); s = s[ok]; g = g[ok]
    xs = [np.sort(s[g == l]) for l in range(K)]; n = np.array([len(x) for x in xs]); num = den = 0.0; U = {}
    for a in range(K):
        for b in range(a + 1, K):
            if n[a] and n[b]:
                u = _auc_sorted(xs[a], xs[b]); U[(a, b)] = u; num += u * n[a] * n[b]; den += n[a] * n[b]
    return (num / den if den else np.nan), U, n


def jt_test(s, g, K=5):
    """Jonckheere-Terpstra for decreasing order across groups (0 highest), tie-corrected normal approx; one-sided p for the expected order."""
    ok = np.isfinite(s); s = s[ok]; g = g[ok]; c, U, n = ordinal(s, g, K); N = n.sum()
    den = sum(n[a] * n[b] for (a, b) in U); JT = sum(U[k] * n[k[0]] * n[k[1]] for k in U); mean = den / 2.0
    nk = n[n > 0].astype(float); t = np.unique(s, return_counts=True)[1].astype(float); N = float(N)
    v = (N * (N - 1) * (2 * N + 5) - (nk * (nk - 1) * (2 * nk + 5)).sum() - (t * (t - 1) * (2 * t + 5)).sum()) / 72.0
    v += (nk * (nk - 1) * (nk - 2)).sum() * (t * (t - 1) * (t - 2)).sum() / (36 * N * (N - 1) * (N - 2)) + (nk * (nk - 1)).sum() * (t * (t - 1)).sum() / (8 * N * (N - 1))
    z = (JT - mean) / np.sqrt(v); return float(z), float(norm.sf(z)), float(JT), float(mean), float(v)


def boot_strat(fn, arrs, B, rng):
    """stratified (by array) bootstrap; arrs = list of 1-d arrays (groups); fn(list)->float; returns B draws (point estimate NOT included)."""
    arrs = [np.asarray(a, float) for a in arrs]; out = np.full(B, np.nan)
    if any(len(a) == 0 for a in arrs): return out
    for b in range(B): out[b] = fn([a[rng.integers(0, len(a), len(a))] for a in arrs])
    return out


def ci(d, a=0.05):
    d = d[np.isfinite(d)]; return (np.nan, np.nan) if len(d) < 20 else (float(np.quantile(d, a / 2)), float(np.quantile(d, 1 - a / 2)))


def ap_score(score, y):   # tie-invariant AP (sklearn: ties are one threshold block)
    from sklearn.metrics import average_precision_score
    score = np.asarray(score, float); y = np.asarray(y, bool); ok = np.isfinite(score); score = score[ok]; y = y[ok]
    return float(average_precision_score(y, score)) if y.sum() and (~y).sum() else np.nan


def cohen_d(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float); x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    if len(x) < 2 or len(y) < 2: return np.nan
    sp = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2)); return (x.mean() - y.mean()) / sp if sp > 0 else np.nan


def mwu_one(x, y, alt='greater'):
    x = np.asarray(x, float); y = np.asarray(y, float); x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    return mannwhitneyu(x, y, alternative=alt).pvalue if min(len(x), len(y)) >= 3 else np.nan


def levels(df, col):
    v = df[col].to_numpy(float); return [v[(df.lvl.to_numpy() == i) & np.isfinite(v)] for i in range(5)]


def ba_recalls(s, down, up):
    """sign-based recalls: KO-down with s>0, KO-up with s<0 (s==0/NaN counts as miss)."""
    s = np.asarray(s, float); rd = np.mean(s[down] > 0) if down.sum() else np.nan; ru = np.mean(s[up] < 0) if up.sum() else np.nan
    return rd, ru, np.nanmean([rd, ru])
