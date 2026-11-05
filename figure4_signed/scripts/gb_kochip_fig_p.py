"""KO x ChIP audit top row re-render: mean +/- 95% gene-bootstrap CI per level with adjacent-contrast brackets (gb_fig5_d4_vs_iso.py style).
Plot/eval only. Reuses gb_kochip_audit_lib (data, MWU, Holm over the 6 gb_fig5_signed.py contrasts). Output: 19_levels_meanCI_{iso,d4,both}.png
Bootstraps replicate gb_kochip_audit.py (p1 level means: rng_for(u,'lev',i), B=KB; p1b c: rng_for(u,'mb'/'ib')); iso CIs are checked against saved p1_levels.tsv."""
import sys, os
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch')]
import numpy as np, pandas as pd, matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from gb_kochip_audit_lib import *
B = int(os.environ.get('KB', 2000))
def rng_for(*tag): return np.random.default_rng([SEED] + [sum(ord(c) * (i + 1) for i, c in enumerate(str(t))) for t in tag])   # verbatim gb_kochip_audit.py:16
bm = pd.read_csv(OUT + 'p1b_module.tsv', sep='\t').set_index('unit_id'); lv0 = pd.read_csv(OUT + 'p1_levels.tsv', sep='\t')
cols = ['#c0392b', '#e8a39a', '#9aa0a6', '#9bb8d9', '#2c5d9e']
ref = pd.read_csv(FIGDIR + 'fig5e_isosign_vs_d4_contrasts.tsv', sep='\t').drop_duplicates(); assert not ref.duplicated(['unit', 'contrast']).any()
R = {}; nbad = 0; ncmp = 0
for u in UNITS:
    nm = UNITS[u][4]; df = load_unit(u); df = df.merge(pd.read_csv(OUT + f'aux_{u}.tsv', sep='\t')[['gene', 'module']], on='gene', how='left', validate='1:1'); R[u] = {}
    for tag, col in (('d4', 'Z|d4'), ('iso', 'T_iso')):
        g = levels(df, col); ps = [mwu_one(g[i], g[j], a) for i, j, a in PAIRS]; adj = holm(ps); lev = []
        for i, x in enumerate(g):
            if not len(x): lev.append((0, np.nan, np.nan, np.nan)); continue
            ix = rng_for(u, 'lev', i).integers(0, len(x), (B, len(x))); mm = x[ix].mean(1); lo, hi = ci(mm); lev.append((len(x), x.mean(), lo, hi))
        if tag == 'iso':   # reproduces saved audit CIs
            L0 = lv0[lv0.unit == nm].reset_index(drop=True)
            for i, (n, m_, lo, hi) in enumerate(lev):
                if n: assert np.isclose(m_, L0['mean'][i]) and np.isclose(lo, L0.mean_lo[i]) and np.isclose(hi, L0.mean_hi[i]), ('iso CI mismatch', nm, i)
        for (i, j, a), p, h in zip(PAIRS, ps, adj):   # p-value parity vs fig5e_isosign_vs_d4_contrasts.tsv
            lab = f"L{i + 1}{'>' if a == 'greater' else '<'}L{j + 1}"; r = ref[(ref.unit == nm) & (ref.contrast == lab)]; assert len(r) == 1; r = r.iloc[0]; ncmp += 1
            ok = np.isclose(r[f'p_raw_{tag}'], p, rtol=1e-9, atol=1e-15, equal_nan=True) and np.isclose(r[f'p_holm6_{tag}'], h, rtol=1e-9, atol=1e-15, equal_nan=True); nbad += (not ok)
        if tag == 'iso': c_txt = f'c={bm.loc[u, "c"]:.2f}\nmodule CI [{bm.loc[u, "c_mod_lo"]:.2f},{bm.loc[u, "c_mod_hi"]:.2f}]'
        else:
            m = (df.lvl.to_numpy() >= 0) & np.isfinite(df['Z|d4'].to_numpy()); s = df['Z|d4'].to_numpy()[m]; gg = df.lvl.to_numpy()[m]; mod = df.module.to_numpy()[m]
            idxs = [np.where(mod == x)[0] for x in np.unique(mod)]; M = len(idxs); rb = rng_for(u, 'mb'); bt = np.full(B, np.nan)
            for b in range(B): ix = np.concatenate([idxs[j] for j in rb.integers(0, M, M)]); bt[b] = ordinal(s[ix], gg[ix])[0]
            lo, hi = ci(bt); c_txt = f'c={ordinal(s, gg)[0]:.2f}\nmodule CI [{lo:.2f},{hi:.2f}]'
        R[u][tag] = dict(lev=lev, ps=ps, adj=adj, c=c_txt)
print(f'p-value parity vs fig5e_isosign_vs_d4_contrasts.tsv (deduped, {len(ref)} rows): compared {ncmp} (raw+Holm, d4+iso), mismatches {nbad}'); assert nbad == 0
rows = [dict(unit=UNITS[u][4], score=t, contrast=f"L{i + 1}{'>' if a == 'greater' else '<'}L{j + 1}", p_raw=p, p_holm6=h) for u in UNITS for t in ('d4', 'iso') for (i, j, a), p, h in zip(PAIRS, R[u][t]['ps'], R[u][t]['adj'])]
pd.DataFrame(rows).to_csv(OUT + 'fig19_contrast_pvalues.tsv', sep='\t', index=False)
# per-TF shared y-limits (both scores)
YL = {}
for u in UNITS:
    lo = min(l[2] for t in R[u].values() for l in t['lev'] if l[0]); hi = max(l[3] for t in R[u].values() for l in t['lev'] if l[0]); sp = hi - lo
    YL[u] = (lo - 0.30 * sp, hi + 0.08 * sp + 4 * 0.24 * sp + 0.05 * sp)
def panel(ax, u, tag, first):
    d = R[u][tag]; y0, y1 = YL[u]; sp = (y1 - y0) / 1.9   # sp ~ data span
    dsp = max(l[3] for l in d['lev'] if l[0]) - min(l[2] for l in d['lev'] if l[0])
    for i, (n, m_, lo, hi) in enumerate(d['lev']):
        if n: ax.errorbar(i + 1, m_, yerr=[[m_ - lo], [hi - m_]], fmt='o', color=cols[i], capsize=3, ms=6, lw=1.5)
    y = max(l[3] for l in d['lev'] if l[0]) + 0.08 * dsp
    for (i, j, a), p, h in zip(PAIRS, d['ps'], d['adj']):
        if abs(i - j) != 1: continue
        st = star(h) if np.isfinite(h) else None; lab = (st if st else ('n.s.' if np.isfinite(p) else 'n<3')) + (f'\np={p:.2g}' if np.isfinite(p) else '')
        ax.plot([i + 1, j + 1], [y, y], c='k' if st else 'grey', lw=1.1); ax.text((i + j) / 2 + 1, y + 0.01 * dsp, lab, ha='center', va='bottom', fontsize=6.5, color='k' if st else 'grey', linespacing=0.95); y += 0.24 * dsp
    ax.axhline(0, c='grey', lw=.6, ls='--'); ax.set_ylim(y0, y1); ax.set_xlim(0.4, 5.6)
    ax.set_xticks(range(1, 6)); ax.set_xticklabels([f'L{i + 1}\nn={l[0]}' for i, l in enumerate(d['lev'])], fontsize=7); ax.spines[['top', 'right']].set_visible(False)
    ax.text(0.03, 0.03, d['c'], transform=ax.transAxes, fontsize=7, va='bottom')
    if first: ax.set_ylabel({'iso': 'mean T_iso, 95% CI\n(gene bootstrap)', 'd4': 'mean T_d4 (Z|d4), 95% CI\n(gene bootstrap)'}[tag], fontsize=8)
FOOT = ('exploratory; per-TF Holm.  Brackets: one-sided Mann-Whitney in expected direction (raw p shown); stars = Holm over 6 contrasts per TF '
        '(* <.05, ** <.01, *** <.001; grey = n.s.).  c, module CI: five-level ordinal concordance, module block bootstrap (k=50).')
TITLE = {'iso': 'KO x ChIP audit: mean T_iso by level', 'd4': 'KO x ChIP audit: mean T_d4 (Z|d4) by level'}
def draw(tags, fn):
    fig, axs = plt.subplots(len(tags), 6, figsize=(17, 3.9 * len(tags)), squeeze=False)
    for r, t in enumerate(tags):
        for k, u in enumerate(UNITS):
            panel(axs[r, k], u, t, k == 0)
            if r == 0: axs[r, k].set_title(UNITS[u][4], fontsize=8.5)
    fig.suptitle(TITLE[tags[0]] if len(tags) == 1 else 'KO x ChIP audit: mean score by level (top: T_d4 = Z|d4; bottom: T_iso)', fontsize=15)
    fig.text(0.5, 0.0, FOOT, ha='center', va='top', fontsize=7, color='#444'); fig.tight_layout(rect=(0, 0.01, 1, 0.95))
    fig.savefig(FIGDIR + fn, dpi=170, bbox_inches='tight'); plt.close(fig); print('wrote', FIGDIR + fn)
draw(['iso'], '19_levels_meanCI_iso.png'); draw(['d4'], '19_levels_meanCI_d4.png'); draw(['d4', 'iso'], '19_levels_meanCI_both.png')
