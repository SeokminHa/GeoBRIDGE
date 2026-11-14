# Figure 4 class means and trend tests for signed score T (4b: mean T per class)
import sys, os, json, hashlib
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
a = sys.argv; RUN = a[1].rstrip('/') + '/'; O = a[2].rstrip('/') + '/'; os.makedirs(O, exist_ok=True)
B = int(a[a.index('--B') + 1]) if '--B' in a else 9999; MINMOD = int(a[a.index('--minmod') + 1]) if '--minmod' in a else 5
UNITS = [('CARDIACV_TBX5', 'TBX5', 'heart'), ('CARDIACV_NKX2-5', 'NKX2-5', 'heart'), ('PANC_RFX6', 'RFX6', 'pancreas')]
LV = ['L1', 'L2', 'L3', 'L4', 'L5']; COL = {'L1': '#c0392b', 'L2': '#e8a598', 'L3': '#9aa0a6', 'L4': '#9ab7d8', 'L5': '#2c5f9e'}
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0.
    for r, i in enumerate(o): run = max(run, min(1., (m - r) * p[i])); adj[i] = run
    return adj
STAT, TESTS, DATA, TREND = [], [], {}, []
for u, tf, sysn in UNITS:
    D = pd.read_csv(RUN + f'genes_{u}.tsv', sep='\t'); t = D.T_prod.to_numpy(float); lv = D.level.to_numpy(); mods = D.module.to_numpy()
    um, inv = np.unique(mods, return_inverse=True); rng = np.random.default_rng([20261029, 41, len(um)])
    W = np.vstack([np.bincount(rng.integers(0, len(um), len(um)), minlength=len(um))[inv].astype(float) for _ in range(B)])
    for l in LV:
        m = lv == l; wm = (W[:, m] * t[m]).sum(1) / np.where(W[:, m].sum(1) > 0, W[:, m].sum(1), np.nan); wm = wm[np.isfinite(wm)]
        STAT.append(dict(TF=tf, level=l, n=int(m.sum()), n_modules=int(np.unique(mods[m]).size), mean=t[m].mean(), lo=np.percentile(wm, 2.5), hi=np.percentile(wm, 97.5),
                         n_valid=len(wm)))
    for la, lb in zip(LV[:-1], LV[1:]):
        ma, mb = lv == la, lv == lb; obs = t[ma].mean() - t[mb].mean(); r = t - np.where(ma, t[ma].mean(), np.where(mb, t[mb].mean(), 0.))
        wa, wb = W[:, ma], W[:, mb]; ok = (wa.sum(1) > 0) & (wb.sum(1) > 0)
        ds = ((wa * r[ma]).sum(1) / np.maximum(wa.sum(1), 1e-12) - (wb * r[mb]).sum(1) / np.maximum(wb.sum(1), 1e-12))[ok]
        TESTS.append(dict(TF=tf, contrast=f'{la}-{lb}', diff=obs, p=(1 + (np.abs(ds) >= abs(obs)).sum()) / (1 + ok.sum()), mods_a=int(np.unique(mods[ma]).size),
                          mods_b=int(np.unique(mods[mb]).size), valid=int(ok.sum())))
    DATA[tf] = (sysn, D)
    sc = pd.Series(lv).map({'L1': 2., 'L2': 1., 'L3': 0., 'L4': -1., 'L5': -2.}).to_numpy()
    def wslope(w): xm = (w * sc).sum() / w.sum(); ym = (w * t).sum() / w.sum(); return (w * (sc - xm) * (t - ym)).sum() / (w * (sc - xm) ** 2).sum()
    s0 = wslope(np.ones(len(t))); sb = np.array([wslope(w) for w in W if w.sum() > 0])
    cm = pd.Series(t).groupby(lv).transform('mean').to_numpy(); rres = t - cm
    def wslope_r(w): xm = (w * sc).sum() / w.sum(); ym = (w * rres).sum() / w.sum(); return (w * (sc - xm) * (rres - ym)).sum() / (w * (sc - xm) ** 2).sum()
    s_null = np.array([wslope_r(w) for w in W if w.sum() > 0])
    lomo = [wslope((mods != m_).astype(float)) for m_ in um]
    TREND.append(dict(TF=tf, slope=s0, lo=np.percentile(sb, 2.5), hi=np.percentile(sb, 97.5), p=(1 + (np.abs(s_null) >= abs(s0)).sum()) / (1 + len(s_null)),
                      lomo_min=min(lomo), lomo_max=max(lomo), most_influential_module=um[int(np.argmin(lomo))]))
TR = pd.DataFrame(TREND); TR['p_holm3'] = holm(TR.p); TR.to_csv(O + 'fig4b_trend.tsv', sep='\t', index=False)
S = pd.DataFrame(STAT); Tt = pd.DataFrame(TESTS); Tt['estimable'] = (Tt.mods_a >= MINMOD) & (Tt.mods_b >= MINMOD)
Tt['p_holm12'] = np.nan; Tt.loc[Tt.estimable, 'p_holm12'] = holm(Tt.p[Tt.estimable])
S.to_csv(O + 'fig4b_class_stats.tsv', sep='\t', index=False); Tt.to_csv(O + 'fig4b_adjacent_tests.tsv', sep='\t', index=False)
json.dump(dict(B=B, minmod=MINMOD, run=RUN, code_sha256=hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()), open(O + 'receipt.json', 'w'), indent=1)
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Nimbus Sans', 'Arial', 'DejaVu Sans'], 'font.size': 9, 'pdf.fonttype': 42})
fig = plt.figure(figsize=(12.5, 7.6)); GS = fig.add_gridspec(2, 3, height_ratios=[.62, 1.6], hspace=.55, wspace=.28)
# 4a schematic
ax = fig.add_subplot(GS[0, :]); ax.set_xlim(0, 12.5); ax.set_ylim(0, 3); ax.axis('off')
box = lambda x, y, w, h, c, t1, t2, tc='white': (ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.02,rounding_size=0.08', fc=c, ec='#333333', lw=1)),
                                                 ax.text(x + w / 2, y + h * .62, t1, ha='center', va='center', fontsize=12, fontweight='bold', color=tc),
                                                 ax.text(x + w / 2, y + h * .27, t2, ha='center', va='center', fontsize=8.5, color=tc))
for x, t_ in ((2.1, 'KO-down'), (4.85, 'non-responder'), (7.6, 'KO-up')): ax.text(x + 1.25, 2.85, t_, ha='center', fontweight='bold', fontsize=10)
ax.text(1.9, 1.9, 'ChIP-bound', ha='right', va='center', fontsize=10); ax.text(1.9, .85, 'not bound', ha='right', va='center', fontsize=10)
box(2.1, 1.5, 2.6, .8, COL['L1'], 'L1', 'KO-down, bound'); box(2.1, .45, 2.6, .8, COL['L2'], 'L2', 'KO-down, not bound', '#222222')
box(4.85, .45, 2.6, 1.85, COL['L3'], 'L3', 'non-responder (bound or not)', '#222222')
box(7.6, 1.5, 2.6, .8, COL['L5'], 'L5', 'KO-up, bound'); box(7.6, .45, 2.6, .8, COL['L4'], 'L4', 'KO-up, not bound', '#222222')
ax.text(10.45, 2.3, 'expected T:\nL1 > L2 > L3 > L4 > L5\n\nChIP: evaluation only;\nbinding ≠ direct regulation.\nT = sign(iso)·F(Z)', va='top', fontsize=8.5)
ax.text(-.1, 3.0, 'a', fontsize=16, fontweight='bold', va='top')
# 4b
for j, (u, tf, sysn) in enumerate(UNITS):
    ax = fig.add_subplot(GS[1, j]); s = S[S.TF == tf].set_index('level'); tt = Tt[Tt.TF == tf].reset_index(drop=True)
    for i, l in enumerate(LV):
        r = s.loc[l]; ax.errorbar(i, r['mean'], yerr=[[r['mean'] - r.lo], [r.hi - r['mean']]], fmt='o', color=COL[l], ms=8, capsize=0, lw=2)
    ax.axhline(0, color='#888888', ls='--', lw=.8)
    for i, r in tt.iterrows():
        y = .92 - .1 * i; ax.plot([i + .1, i + .9], [y, y], color='k' if (r.estimable and r.p_holm12 < .05) else '#999999', lw=1.3)
        lab = (f"{'***' if r.p_holm12 < .001 else '**' if r.p_holm12 < .01 else '*' if r.p_holm12 < .05 else 'n.s.'} {r.p_holm12:.2g}" if r.estimable
               else f'n.e. ({min(r.mods_a, r.mods_b)} mod)')
        ax.text(i + .5, y + .02, lab, ha='center', fontsize=7.3, color='k' if (r.estimable and r.p_holm12 < .05) else '#888888')
    ax.set_xticks(range(5)); ax.set_xticklabels([f"{l}\n{int(s.loc[l, 'n'])}" for l in LV], fontsize=8); ax.set_ylim(-1, 1); ax.set_xlim(-.5, 4.5)
    tr = TR.set_index('TF').loc[tf]; ax.set_title(tf, fontweight='bold', fontsize=12, pad=18)
    ax.text(.5, 1.015, f'ordered trend slope {tr.slope:.3f} [{tr.lo:.3f}, {tr.hi:.3f}]; null-imposed Holm-3 p={tr.p_holm3:.2g}; leave-one-module-out {tr.lomo_min:.2f}–{tr.lomo_max:.2f}',
            transform=ax.transAxes, ha='center', fontsize=6.6)
    ax.spines[['top', 'right']].set_visible(False)
    if j == 0: ax.set_ylabel('mean T (signed percentile, −1..1), 95% CI'); ax.text(-1.25, 1.18, 'b', fontsize=16, fontweight='bold', transform=ax.transData)
fig.text(.37, .655, 'heart', ha='center', fontsize=11, color='#444444'); fig.text(.795, .655, 'pancreas', ha='center', fontsize=11, color='#444444')
for x0, x1 in ((.13, .615), (.695, .9)): fig.add_artist(plt.Line2D([x0, x1], [.648, .648], color='#666666', lw=1.2))
fig.text(.13, .015, f'CI: 95% bootstrap over the 50 control modules ({B:,} draws; genes are not independent). Brackets: adjacent-class mean difference, two-sided null-imposed module '
         f'bootstrap, Holm-adjusted across the estimable adjacent contrasts;\nn.e. = fewer than {MINMOD} modules in a class (not estimable). Ordered trend: slope of T on class score (L1..L5 = +2..−2), '
         f'null-imposed module bootstrap of class-demeaned residuals, Holm over 3 TFs. Descriptive; ChIP is evaluation truth only and does not establish direct regulation.', fontsize=7)
for x0, x1 in ((.69, .905),): pass
for f in ('png', 'pdf'): fig.savefig(O + f'Fig4_ab_style.{f}', dpi=300, bbox_inches='tight')
pd.set_option('display.width', 200); print(S.round(3).to_string(index=False)); print(Tt.round(4).to_string(index=False)); print(TR.round(4).to_string(index=False))
