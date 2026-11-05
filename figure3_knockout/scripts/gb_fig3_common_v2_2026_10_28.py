"""Fig 3 common-row benchmark, v2 (PI 2026-10-28: drop the nonfocal-source ensemble bar; add an equal-system mean panel next to the per-endpoint panels).
Same evaluator, rows and statistics as gb_fig3_common_2026_10_28.py except: (1) GENERIC removed from the plot AND from the row-intersection
requirement (rows = U_GB where GeoBRIDGE and all 8 comparators are finite); (2) equal-system mean = mean over PANC, BM and HEART, where
HEART = mean(TBX5, NKX2-5); computed per synchronized bootstrap draw (cardiac draws are synchronized, systems independent), paired GB - X on the
same draws, PT.boot_p, Holm over the 8 comparators. EVALUATION + plot only; no score is recomputed.
usage (server, repo root): script OUT [--B 1000]"""
import sys, os
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
a = sys.argv; OUT = a[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True); B = a[a.index('--B') + 1] if '--B' in a else '1000'
CUT = 'for s in SYS4:\n    cs = '
def load(path, argv):
    src = open(path).read().replace('\r', ''); src = src[:src.index(CUT)]
    ns = {'__file__': path, '__name__': 'prefix'}; sys.argv = argv; exec(compile(src, path, 'exec'), ns); return ns
UOLD = '--uold' in a  # PI 2026-10-28: original-benchmark universe = rows also scored by the ORIGINAL cached Velorama column (cog::Velorama), as in gb_ms_fig3_set9_2026_10_19.py
ERY = '--ery' in a  # optional variant: add ERY_KLF1_B as a fourth system (PI 2026-10-28 "ERY 하나 넣어볼까"); outputs get suffix _ERY
EPS = ['PANC_RFX6', 'BM_GATA1_acute', 'HEART_TBX5', 'HEART_NKX2-5'] + (['ERY_KLF1_B'] if ERY else []); SUF = ('_ERY' if ERY else '') + ('_UOLD' if UOLD else '')
PLOTONLY = '--plotonly' in a
if not PLOTONLY: nsA = load(R + 'scratch/gb_fig3_ugb_v2_velalltargets_2026_10_26.py', ['x', OUT + '_scrA', '--B', B, '--endpoints', 'PANC_RFX6,HEART_TBX5,HEART_NKX2-5' + (',ERY_KLF1_B' if ERY else '')])
if not PLOTONLY: nsB = load(R + 'results/signed/bm_acute_design_audit_2026_10_26/noF2/fig3_ugb_v2_noF2_velalltargets.py', ['x', OUT + '_scrB', '--B', B, '--rule', 'noF2', '--endpoints', 'BM_GATA1_acute'])
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
EXT = ['|Pearson|', 'GENIE3', 'GRNBoost2', 'Velorama', 'RegDiffusion', 'DeepSEM', 'PIDC', 'FastTENET']
if not PLOTONLY: CT, PT = nsA['CT'], nsA['PT']; assert list(nsA['EXT']) == EXT, nsA['EXT']
MT = ('AUROC', 'APLIFT'); METH = ['GeoBRIDGE'] + EXT
def holm(p):
    """Holm over a FIXED family of len(p) slots (internal review 2026-10-28): a missing p keeps its slot (treated as p=1), never shrinks the multiplier."""
    p = np.asarray(p, float); m = len(p); q = np.where(np.isfinite(p), p, 1.0); o = np.argsort(q, kind='stable'); out = np.full(m, np.nan); run = 0.0
    for i, k in enumerate(o): run = max(run, (m - i) * q[k]); out[k] = min(1.0, run)
    out[~np.isfinite(p)] = np.nan; return out
from scipy.stats import norm
def delong(a, b, y):
    """paired DeLong test of AUROC(a) - AUROC(b) on the same rows; gene-level iid (ignores module dependence) -> cross-check only. two-sided p."""
    y = np.asarray(y, bool); X, Y = [], []
    def comps(s):
        sp, sn = s[y], s[~y]; v10 = ((sp[:, None] > sn[None, :]) + .5 * (sp[:, None] == sn[None, :])).astype(float)
        return v10.mean(1), v10.mean(0), v10.mean()
    pa, na, A = comps(np.asarray(a, float)); pb, nb, Bv = comps(np.asarray(b, float))
    S10 = np.cov(np.vstack([pa, pb])); S01 = np.cov(np.vstack([na, nb])); S = S10 / len(pa) + S01 / len(na)
    var = S[0, 0] + S[1, 1] - 2 * S[0, 1]; z = (A - Bv) / np.sqrt(var) if var > 0 else np.nan
    return float(2 * norm.sf(abs(z))) if np.isfinite(z) else np.nan
def summ(lab, d, k, m, v, gbv, npos, nrows):
    # ONE validity status (internal review 2026-10-28): interval AND p are reported only when >= 95% of draws are finite; otherwise NA (never "n.s.")
    nb = len(v) - 1; ok = lambda x: len(x) >= .95 * nb
    f = v[1:][np.isfinite(v[1:])]; okl = ok(f)
    r = dict(endpoint=lab, direction=d, metric=k, method=m, n_rows=nrows, n_pos=npos, value=v[0], lo=np.percentile(f, 2.5) if okl else np.nan,
             hi=np.percentile(f, 97.5) if okl else np.nan, level_valid_draws=len(f), n_draws=nb)
    if m != 'GeoBRIDGE':
        dd = gbv - v; df = dd[1:][np.isfinite(dd[1:])]; okd = ok(df)
        r.update(GB_minus=dd[0], diff_lo=np.percentile(df, 2.5) if okd else np.nan, diff_hi=np.percentile(df, 97.5) if okd else np.nan, n_valid_draws=len(df),
                 p_boot=PT.boot_p(df) if okd else np.nan)
    return r
if PLOTONLY:   # PI 2026-10-28: redraw (significance stars) from the saved table, no recomputation
    Rr = pd.read_csv(OUT + f'fig3_common_v2{SUF}.tsv', sep='\t'); C_ = pd.read_csv(OUT + f'fig3_common_v2{SUF}_coverage.tsv', sep='\t')
else:
    ROWS, COV, DRAW = [], [], {}
    for lab in EPS:
        X = nsB['D'][lab] if lab == 'BM_GATA1_acute' else nsA['D'][lab]; V = dict(X['V'])
        U = np.asarray(X['U'], bool); Uc = U & np.all([np.isfinite(V[m]) for m in METH], 0)
        if UOLD:
            EPT = {x[0]: x for x in nsA['EP9']}; zv = np.load(f"{nsA['VEC']}ep_{EPT[lab][5]}.vec.npz", allow_pickle=True)
            assert np.array_equal(zv['genes'].astype(str), np.asarray(X['g']).astype(str)) or np.array_equal(np.char.upper(zv['genes'].astype(str)), np.asarray(X['g']).astype(str)), 'vec gene order ' + lab
            oldvel = zv['cog::Velorama'].astype(float); U = U & np.isfinite(oldvel); Uc = Uc & np.isfinite(oldvel)
        y0, e = np.asarray(X['y'], bool), np.asarray(X['e'], float)
        for d in ('ALL', 'KOup', 'KOdown'):
            yl = y0.copy() if d == 'ALL' else (y0 & np.isfinite(e) & ((e > 0) if d == 'KOup' else (e < 0))); rr = Uc if d == 'ALL' else Uc & (yl | ~y0)
            rr0 = U if d == 'ALL' else U & (yl | ~y0); npos = int((yl & rr).sum()); nneg = int((~yl & rr).sum())
            COV.append(dict(endpoint=lab, direction=d, n_rows_UGB=int(rr0.sum()), n_rows_common=int(rr.sum()), row_coverage=rr.sum() / rr0.sum(), n_pos_UGB=int((yl & rr0).sum()), n_pos_common=npos, n_neg_common=nneg))
            if npos < 5 or nneg < 5: continue   # gate BOTH classes (internal review)
            for m in METH:
                pt, bs = CT.eval_rows(V[m], yl, rr, X['gi'], X['C'])
                for k in MT: DRAW[(lab, d, k, m)] = np.r_[pt[k], np.asarray(bs[k], float)]
            for k in MT:
                n0 = len(ROWS)
                for m in METH:
                    ROWS.append(summ(lab, d, k, m, DRAW[(lab, d, k, m)], DRAW[(lab, d, k, 'GeoBRIDGE')], npos, int(rr.sum())))
                    if k == 'AUROC' and m != 'GeoBRIDGE': ROWS[-1]['p_delong_iid'] = delong(np.asarray(V['GeoBRIDGE'], float)[rr], np.asarray(V[m], float)[rr], yl[rr])
                sub = [i for i in range(n0, len(ROWS)) if ROWS[i]['method'] in EXT]
                for i, ph in zip(sub, holm([ROWS[i]['p_boot'] for i in sub])): ROWS[i]['p_holm8'] = ph
        print(lab, 'done; common rows', int(Uc.sum()), 'of U_GB', int(U.sum()), flush=True)
    # equal-system mean (PANC, BM, HEART = mean of the two cardiac endpoints), per draw
    for d in ('ALL', 'KOup', 'KOdown'):
        if not all((lab, d, 'AUROC', 'GeoBRIDGE') in DRAW for lab in EPS): continue
        for k in MT:
            M = {m: (DRAW[('PANC_RFX6', d, k, m)] + DRAW[('BM_GATA1_acute', d, k, m)] + .5 * (DRAW[('HEART_TBX5', d, k, m)] + DRAW[('HEART_NKX2-5', d, k, m)])
                     + (DRAW[('ERY_KLF1_B', d, k, m)] if ERY else 0)) / (4 if ERY else 3) for m in METH}
            n0 = len(ROWS)
            for m in METH:
                r = summ('EQSYS', d, k, m, M[m], M['GeoBRIDGE'], np.nan, np.nan)
                for lab in EPS: r['sys_' + lab] = DRAW[(lab, d, k, m)][0]
                ROWS.append(r)
            sub = [i for i in range(n0, len(ROWS)) if ROWS[i]['method'] in EXT]
            for i, ph in zip(sub, holm([ROWS[i]['p_boot'] for i in sub])): ROWS[i]['p_holm8'] = ph
    Rr = pd.DataFrame(ROWS); Rr.to_csv(OUT + f'fig3_common_v2{SUF}.tsv', sep='\t', index=False); C_ = pd.DataFrame(COV); C_.to_csv(OUT + f'fig3_common_v2{SUF}_coverage.tsv', sep='\t', index=False)
    # provenance (internal review): every draw array (index 0 = point estimate) + receipt with input/script hashes
    import json, hashlib
    np.savez_compressed(OUT + f'draws_v2{SUF}.npz', **{'|'.join(kk): vv for kk, vv in DRAW.items()})
    _sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
    json.dump(dict(B=int(B), uold=UOLD, ery=ERY, endpoints=EPS, methods=METH, holm='fixed 8 comparator slots per endpoint x direction x metric (incl. EQSYS)', validity='>=95% finite draws for interval and p',
                   p='two-sided percentile-interval bootstrap p, +1 rule (gb_ptable.boot_p)', eqsys_weights='PANC 1/3, BM 1/3, TBX5 1/6, NKX2-5 1/6' + (' (ERY variant: 1/4 each system)' if ERY else ''),
                   sha_script=_sha(os.path.abspath(sys.argv[0]) if os.path.exists(sys.argv[0]) else R + 'scratch/gb_fig3_common_v2_2026_10_28.py'),
                   sha_evaluator=_sha(R + 'scratch/gb_fig3_ugb_v2_velalltargets_2026_10_26.py'), sha_bm_adapter=_sha(R + 'results/signed/bm_acute_design_audit_2026_10_26/noF2/fig3_ugb_v2_noF2_velalltargets.py'),
                   sha_ptable=_sha(R + 'scratch/gb_ptable.py'), sha_canon_table=_sha(R + 'scratch/gb_canon_table.py')), open(OUT + f'receipt_v2{SUF}.json', 'w'), indent=1)
# ---------------- plot
TT = {'EQSYS': 'Equal-system mean', 'PANC_RFX6': 'Pancreas · RFX6', 'BM_GATA1_acute': 'Bone marrow · GATA1', 'HEART_TBX5': 'Heart · TBX5', 'HEART_NKX2-5': 'Heart · NKX2-5', 'ERY_KLF1_B': 'Erythroid · KLF1'}
RED, GREY = '#c0392b', '#b8bcc2'; MK = {'PANC_RFX6': 's', 'BM_GATA1_acute': '^', 'HEART': 'D'}
if ERY: MK['ERY_KLF1_B'] = 'v'
def panel(ax, lab, d, k, title):
    q = Rr[(Rr.endpoint == lab) & (Rr.direction == d) & (Rr.metric == k)]
    if q.empty: ax.axis('off'); ax.set_title(title + '\n(not evaluable)', fontsize=8); return
    gb = q[q.method == 'GeoBRIDGE'].iloc[0]; ex = q[q.method.isin(EXT)].sort_values('value', ascending=False)
    rows = [gb] + [r for _, r in ex.iterrows()]; names = ['GeoBRIDGE'] + ex.method.tolist()
    ref = .5 if k == 'AUROC' else 1.0; y = np.arange(len(rows)); vals = np.array([r.value for r in rows]); los = np.array([r.lo for r in rows]); his = np.array([r.hi for r in rows])
    for i in range(len(rows)): ax.barh(y[i], vals[i] - ref, left=ref, color=RED if i == 0 else GREY, height=.68, lw=0)
    for i in range(len(rows)):   # draw percentile intervals directly (they may exclude the point estimate; xerr cannot)
        if np.isfinite(los[i]): ax.plot([los[i], his[i]], [y[i], y[i]], color='#333', lw=.6); ax.plot([los[i]] * 2, [y[i] - .12, y[i] + .12], color='#333', lw=.6); ax.plot([his[i]] * 2, [y[i] - .12, y[i] + .12], color='#333', lw=.6)
    xs = [np.nanmin(np.r_[los, vals]), ref, np.nanmax(np.r_[his, vals])]
    if lab == 'EQSYS':
        for i, r in enumerate(rows):
            for s, mk in MK.items():
                v = .5 * (r['sys_HEART_TBX5'] + r['sys_HEART_NKX2-5']) if s == 'HEART' else r['sys_' + s]
                ax.scatter([v], [i], marker=mk, s=11, facecolor='white', edgecolor='#222', lw=.5, zorder=4); xs.append(v)
    lo_ = min(xs); hi_ = max(xs); sp = hi_ - lo_; ax.set_xlim(lo_ - .05 * sp, hi_ + .42 * sp); xd = hi_ + .2 * sp
    for i, r in enumerate(rows):
        if i == 0: continue
        ph = r.p_holm8; gbw = r.GB_minus > 0
        if not np.isfinite(r.p_boot): ax.scatter([xd], [i], s=20, marker='x', color='#777', lw=.8, zorder=3); continue   # undefined (<95% valid draws), not "n.s."
        fc = ('k' if gbw else '#2471a3') if (np.isfinite(ph) and ph < .05) else (('#9a9a9a' if gbw else '#a9cce3') if r.p_boot < .05 else 'white')
        ax.scatter([xd], [i], s=20, facecolor=fc, edgecolor='k', lw=.6, zorder=3)
        st = '***' if ph < .005 else '**' if ph < .01 else '*' if ph < .05 else ''   # PI 2026-10-28: graded stars on the nominal Holm-8 p
        if st: ax.text(xd + .035 * sp, i, st, va='center', ha='left', fontsize=7, color='k' if gbw else '#2471a3')
    ax.axvline(ref, color='k', lw=.6); ax.set_yticks(y); ax.set_yticklabels(names, fontsize=6.8); ax.get_yticklabels()[0].set_color(RED); ax.invert_yaxis()
    ax.spines[['top', 'right']].set_visible(False)
    sub = '' if lab == 'EQSYS' else f'  (n+={int(gb.n_pos)}, rows={int(gb.n_rows)})'
    ax.set_title(title + sub, fontsize=8, fontweight='bold' if lab == 'EQSYS' else 'normal', color='#b9770e' if lab != 'EQSYS' and gb.n_pos < 20 else 'k')
LEG = [Line2D([], [], marker='o', ls='', mfc='k', mec='k', ms=5, label='GB better, nominal Holm-8 p<.05 (* <.05, ** <.01, *** <.005)'), Line2D([], [], marker='o', ls='', mfc='#2471a3', mec='k', ms=5, label='method better, nominal Holm-8 p<.05'),
       Line2D([], [], marker='o', ls='', mfc='#9a9a9a', mec='k', ms=5, label='GB better, raw p<.05 only'), Line2D([], [], marker='o', ls='', mfc='#a9cce3', mec='k', ms=5, label='method better, raw p<.05 only'),
       Line2D([], [], marker='o', ls='', mfc='white', mec='k', ms=5, label='n.s.'), Line2D([], [], marker='x', ls='', color='#777', ms=5, label='undefined (<95% valid draws)')]
LEG2 = [Line2D([], [], marker=MK['PANC_RFX6'], ls='', mfc='white', mec='#222', ms=4, label='PANC'), Line2D([], [], marker=MK['BM_GATA1_acute'], ls='', mfc='white', mec='#222', ms=4, label='BM'),
        Line2D([], [], marker=MK['HEART'], ls='', mfc='white', mec='#222', ms=4, label='HEART (mean of TBX5, NKX2-5)')]
if ERY: LEG2.append(Line2D([], [], marker='v', ls='', mfc='white', mec='#222', ms=4, label='ERY'))
PAN = ['EQSYS'] + EPS; NP = len(PAN)
for k, xl in (('AUROC', 'AUROC'), ('APLIFT', 'AUPRC / prevalence')):
    fig, axs = plt.subplots(1, NP, figsize=(3.4 * NP, 3.8)); plt.subplots_adjust(left=.06, right=.99, top=.84, bottom=.22, wspace=.62)
    for ax, lab in zip(axs, PAN): panel(ax, lab, 'ALL', k, TT[lab]); ax.set_xlabel(xl, fontsize=7.5)
    fig.legend(handles=LEG + LEG2, loc='lower center', ncol=5, frameon=False, fontsize=7); fig.suptitle(f'Fig 3 · all KO responders · {xl} · one common row set per endpoint (95% CI; dots = GB vs method)', fontsize=9)
    fig.savefig(OUT + f'Fig3_common_v2{SUF}_ALL_{k}.png', dpi=220); plt.close(fig)
    fig, axs = plt.subplots(2, NP, figsize=(3.4 * NP, 7.4)); plt.subplots_adjust(left=.06, right=.99, top=.9, bottom=.11, wspace=.62, hspace=.5)
    for i, (d, dn) in enumerate((('KOup', 'KO-up'), ('KOdown', 'KO-down'))):
        for ax, lab in zip(axs[i], PAN): panel(ax, lab, d, k, f'{TT[lab]} · {dn}'); ax.set_xlabel(xl, fontsize=7.5)
    fig.legend(handles=LEG + LEG2, loc='lower center', ncol=5, frameon=False, fontsize=7); fig.suptitle(f'SI · KO-up / KO-down · {xl} · common rows', fontsize=9)
    fig.savefig(OUT + f'FigS3_common_v2{SUF}_dir_{k}.png', dpi=220); plt.close(fig)
pd.set_option('display.width', 250); print(C_.to_string(index=False))
print(Rr[(Rr.direction == 'ALL')][['endpoint', 'metric', 'method', 'value', 'lo', 'hi', 'GB_minus', 'p_boot', 'p_holm8', 'p_delong_iid', 'n_valid_draws']].round(4).to_string(index=False))
