"""Fig 3 on ONE common row set per endpoint (internal review 2026-10-28 finding 1: the previous bars mixed full-U_GB GB with comparator-specific rows) plus the
leave-focal-out ensemble reference GENERIC (internal review Part 2: mean own-universe PRIMARY of the ~150 xsys-bank alternatives, >= 80% of them finite;
construction verbatim scratch/gb_xsys_differences_2026_10_28.py:73-80). EVALUATION + plot only.
Rows: U_GB evaluator rows (gb_fig3_ugb_v2_velalltargets prefix; BM via the noF2 adapter) where GeoBRIDGE, all 8 comparators and GENERIC are finite;
coverage vs U_GB reported. Metrics AUROC and AP-lift with the evaluator's synchronized G/50 module bootstrap (CT.eval_rows, B draws); paired GB - X
on the same draws, p = PT.boot_p, Holm over the 8 comparators per endpoint x direction x metric (GENERIC reported separately, not in the family).
Main panel = all responders; SI = KO-up / KO-down. Order: GeoBRIDGE first, GENERIC second (hatched), comparators by descending value.
usage (server, repo root): script OUT [--B 1000]"""
import sys, os
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
a = sys.argv; OUT = a[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True); B = a[a.index('--B') + 1] if '--B' in a else '1000'
CUT = 'for s in SYS4:\n    cs = '
def load(path, argv):
    src = open(path).read().replace('\r', ''); src = src[:src.index(CUT)]
    ns = {'__file__': path, '__name__': 'prefix'}; sys.argv = argv; exec(compile(src, path, 'exec'), ns); return ns
EPS = ['PANC_RFX6', 'BM_GATA1_acute', 'HEART_TBX5', 'HEART_NKX2-5']
nsA = load(R + 'scratch/gb_fig3_ugb_v2_velalltargets_2026_10_26.py', ['x', OUT + '_scrA', '--B', B, '--endpoints', 'PANC_RFX6,HEART_TBX5,HEART_NKX2-5'])
nsB = load(R + 'results/signed/bm_acute_design_audit_2026_10_26/noF2/fig3_ugb_v2_noF2_velalltargets.py', ['x', OUT + '_scrB', '--B', B, '--rule', 'noF2', '--endpoints', 'BM_GATA1_acute'])
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
CT, PT = nsA['CT'], nsA['PT']; EP = {x[0]: x for x in nsA['EP9']}; EXT = nsA['EXT']; MT = ('AUROC', 'APLIFT')
FIX = {'PANC_RFX6': 'PANC', 'HEART_TBX5': 'CARDIACV', 'HEART_NKX2-5': 'CARDIACV', 'BM_GATA1_acute': 'BONEMARROW'}
def holm(p):
    p = np.asarray(p, float); out = np.full(len(p), np.nan); ok = np.flatnonzero(np.isfinite(p)); o = ok[np.argsort(p[ok])]; run = 0.0
    for i, k in enumerate(o): run = max(run, (len(o) - i) * p[k]); out[k] = min(1.0, run)
    return out
def generic(fx, tf, g):
    z = np.load(R + f'results/signed/xsys_2026_10_28/full/primary_bank_{fx}.npz', allow_pickle=True); src = [s.upper() for s in z['sources'].astype(str)]; nf = int(z['nfocal'])
    Z = np.asarray(z['Z'], float); alts = list(range(nf, len(src))); s = np.nansum(Z[alts], 0); n = np.isfinite(Z[alts]).sum(0)
    with np.errstate(invalid='ignore', divide='ignore'): P = np.where(n >= .8 * len(alts), s / np.maximum(n, 1), np.nan)
    gi = {q: i for i, q in enumerate(np.char.upper(z['genes'].astype(str)))}; idx = np.array([gi.get(q, -1) for q in g]); o = np.full(len(g), np.nan); o[idx >= 0] = P[idx[idx >= 0]]
    return o
ROWS, COV = [], []
for lab in EPS:
    X = nsB['D'][lab] if lab == 'BM_GATA1_acute' else nsA['D'][lab]; tf = EP[lab][3].upper(); g = np.asarray(X['g']).astype(str)
    V = dict(X['V']); V['GENERIC'] = generic(FIX[lab], tf, g)
    meths = ['GeoBRIDGE', 'GENERIC'] + list(EXT)
    U = np.asarray(X['U'], bool); Uc = U & np.all([np.isfinite(V[m]) for m in meths], 0)
    y0, e = np.asarray(X['y'], bool), np.asarray(X['e'], float)
    for d in ('ALL', 'KOup', 'KOdown'):
        yl = y0.copy() if d == 'ALL' else (y0 & np.isfinite(e) & ((e > 0) if d == 'KOup' else (e < 0))); rr = Uc if d == 'ALL' else Uc & (yl | ~y0)
        rr0 = U if d == 'ALL' else U & (yl | ~y0); npos = int((yl & rr).sum())
        COV.append(dict(endpoint=lab, direction=d, n_rows_UGB=int(rr0.sum()), n_rows_common=int(rr.sum()), row_coverage=rr.sum() / rr0.sum(), n_pos_UGB=int((yl & rr0).sum()), n_pos_common=npos))
        if npos < 5: continue
        res = {}
        for m in meths:
            pt, bs = CT.eval_rows(V[m], yl, rr, X['gi'], X['C']); res[m] = {k: np.r_[pt[k], np.asarray(bs[k], float)] for k in MT}
        for k in MT:
            ps = []
            for m in meths:
                v = res[m][k]; f = v[1:][np.isfinite(v[1:])]; r = dict(endpoint=lab, direction=d, metric=k, method=m, n_rows=int(rr.sum()), n_pos=npos, value=v[0], lo=np.percentile(f, 2.5), hi=np.percentile(f, 97.5))
                if m != 'GeoBRIDGE':
                    dd = res['GeoBRIDGE'][k] - v; df = dd[1:][np.isfinite(dd[1:])]
                    r.update(GB_minus=dd[0], diff_lo=np.percentile(df, 2.5), diff_hi=np.percentile(df, 97.5), p_boot=PT.boot_p(df) if len(df) >= .95 * (len(dd) - 1) else np.nan)
                ROWS.append(r)
            sub = [i for i, r in enumerate(ROWS) if r['endpoint'] == lab and r['direction'] == d and r['metric'] == k and r['method'] in EXT]
            for i, ph in zip(sub, holm([ROWS[i]['p_boot'] for i in sub])): ROWS[i]['p_holm8'] = ph
    print(lab, 'done; common rows', int(Uc.sum()), 'of U_GB', int(U.sum()), flush=True)
Rr = pd.DataFrame(ROWS); Rr.to_csv(OUT + 'fig3_common.tsv', sep='\t', index=False); C_ = pd.DataFrame(COV); C_.to_csv(OUT + 'fig3_common_coverage.tsv', sep='\t', index=False)
# ---------------- plot
TT = {'PANC_RFX6': 'Pancreas · RFX6', 'BM_GATA1_acute': 'Bone marrow · GATA1', 'HEART_TBX5': 'Heart · TBX5', 'HEART_NKX2-5': 'Heart · NKX2-5'}
RED, GREY, GEN = '#c0392b', '#b8bcc2', '#f2c9a0'
def panel(ax, lab, d, k, title):
    q = Rr[(Rr.endpoint == lab) & (Rr.direction == d) & (Rr.metric == k)]
    if q.empty: ax.axis('off'); ax.set_title(title + '\n(not evaluable)', fontsize=8); return
    gb = q[q.method == 'GeoBRIDGE'].iloc[0]; gen = q[q.method == 'GENERIC'].iloc[0]; ex = q[q.method.isin(EXT)].sort_values('value', ascending=False)
    rows = [gb, gen] + [r for _, r in ex.iterrows()]; names = ['GeoBRIDGE', 'leave-focal-out\nensemble (ref.)'] + ex.method.tolist()
    ref = .5 if k == 'AUROC' else 1.0; y = np.arange(len(rows)); vals = np.array([r.value for r in rows]); los = np.array([r.lo for r in rows]); his = np.array([r.hi for r in rows])
    cols = [RED, GEN] + [GREY] * len(ex)
    ax.barh(y, vals - ref, left=ref, color=cols, height=.68, hatch=[None, '///'] + [None] * len(ex)) if False else None
    for i in range(len(rows)): ax.barh(y[i], vals[i] - ref, left=ref, color=cols[i], height=.68, hatch='///' if i == 1 else None, edgecolor='#a0522d' if i == 1 else None, lw=.0)
    ax.errorbar(vals, y, xerr=[vals - los, his - vals], fmt='none', ecolor='#333', lw=.6, capsize=1.2)
    lo_ = min(los.min(), ref); hi_ = his.max(); sp = hi_ - lo_; ax.set_xlim(lo_ - .05 * sp, hi_ + .3 * sp); xd = hi_ + .2 * sp
    for i, r in enumerate(rows):
        if i == 0: continue
        if r.method == 'GENERIC': fc = 'none'; ph = r.p_boot
        else: ph = r.p_holm8; fc = ('k' if r.GB_minus > 0 else '#2471a3') if (np.isfinite(ph) and ph < .05) else ('#9a9a9a' if np.isfinite(r.p_boot) and r.p_boot < .05 else 'white')
        if r.method == 'GENERIC': ax.text(xd, i, f'p={r.p_boot:.2f}', fontsize=5.8, va='center', ha='center', color='#a0522d')
        else: ax.scatter([xd], [i], s=20, facecolor=fc, edgecolor='k', lw=.6, zorder=3)
    ax.axvline(ref, color='k', lw=.6); ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.8); ax.get_yticklabels()[0].set_color(RED); ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False); ax.set_title(f'{title}  (n+={int(gb.n_pos)}, rows={int(gb.n_rows)})', fontsize=8, color='#b9770e' if gb.n_pos < 20 else 'k')
LEG = [Line2D([], [], marker='o', ls='', mfc='k', mec='k', ms=5, label='GB better, Holm-8 p<.05'), Line2D([], [], marker='o', ls='', mfc='#2471a3', mec='k', ms=5, label='method better, Holm-8 p<.05'),
       Line2D([], [], marker='o', ls='', mfc='#9a9a9a', mec='k', ms=5, label='raw p<.05 only'), Line2D([], [], marker='o', ls='', mfc='white', mec='k', ms=5, label='n.s.')]
for k, xl in (('AUROC', 'AUROC'), ('APLIFT', 'AUPRC lift (AP / prevalence)')):
    fig, axs = plt.subplots(1, 4, figsize=(14, 3.8)); plt.subplots_adjust(left=.08, right=.99, top=.82, bottom=.2, wspace=.62)
    for ax, lab in zip(axs, EPS): panel(ax, lab, 'ALL', k, TT[lab]); ax.set_xlabel(xl, fontsize=7.5)
    fig.legend(handles=LEG, loc='lower center', ncol=4, frameon=False, fontsize=7); fig.suptitle(f'Fig 3 · all KO responders · {xl} · one common row set per endpoint (GB vs ensemble p shown as text)', fontsize=9)
    fig.savefig(OUT + f'Fig3_common_ALL_{k}.png', dpi=220); plt.close(fig)
    fig, axs = plt.subplots(2, 4, figsize=(14, 7.4)); plt.subplots_adjust(left=.08, right=.99, top=.9, bottom=.1, wspace=.62, hspace=.5)
    for i, (d, dn) in enumerate((('KOup', 'KO-up'), ('KOdown', 'KO-down'))):
        for ax, lab in zip(axs[i], EPS): panel(ax, lab, d, k, f'{TT[lab]} · {dn}'); ax.set_xlabel(xl, fontsize=7.5)
    fig.legend(handles=LEG, loc='lower center', ncol=4, frameon=False, fontsize=7); fig.suptitle(f'SI · KO-up / KO-down · {xl} · common rows', fontsize=9)
    fig.savefig(OUT + f'FigS3_common_dir_{k}.png', dpi=220); plt.close(fig)
pd.set_option('display.width', 250); print(C_.to_string(index=False))
print(Rr[(Rr.direction == 'ALL')][['endpoint', 'metric', 'method', 'value', 'lo', 'hi', 'GB_minus', 'p_boot', 'p_holm8']].round(3).to_string(index=False))
