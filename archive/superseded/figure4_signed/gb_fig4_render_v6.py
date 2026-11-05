"""Fig 4 v5: KO x ChIP validation first. Reads EXISTING result tables only; no new statistics. Run from repo root on a server.
Sources: kochip_audit_2026_10_15/{p1_levels,fig19_contrast_pvalues,p1b_module}.tsv; figdec_2026_10_16/{heart5/decomp_summary,crossed/summary_excess,pearsign/{summary_excess,paired_iso_minus_pear},spec53/f1_holm6}.tsv"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

SG = 'results/signed/'; KA = SG + 'kochip_audit_2026_10_15/'; FD = SG + 'figdec_2026_10_16/'
OUT = 'results/figures/manuscript_2026_10_19/'; os.makedirs(OUT, exist_ok=True)
LED = []
def rec(panel, qty, val, src, key):
    LED.append((panel, qty.replace('\n', ' '), float(val), src, key)); return float(val)

BLUE, ORG, GREY, LBLUE = '#1f5fa8', '#e07b00', '#8a8f94', '#9db9dc'
plt.rcParams.update({'font.size': 6.5, 'axes.linewidth': .6, 'axes.labelsize': 6.5, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
                     'xtick.major.width': .5, 'ytick.major.width': .5, 'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
                     'axes.spines.top': False, 'axes.spines.right': False, 'font.family': 'DejaVu Sans'})
def pl(ax, t, dx=-0.16, dy=1.04): ax.text(dx, dy, t, transform=ax.transAxes, fontsize=10, fontweight='bold', va='bottom', ha='left')

UNITS = ['CARDIACV_TBX5', 'CARDIACV_NKX2-5', 'GATA1_GATA1', 'KLF1_KLF1', 'KLF1_B_KLF1', 'PANC_RFX6']
DISP = {'CARDIACV_TBX5': 'Heart · TBX5', 'CARDIACV_NKX2-5': 'Heart · NKX2-5', 'GATA1_GATA1': 'Erythroid · GATA1',
        'KLF1_KLF1': 'Erythroid · KLF1 (GSE94351)', 'KLF1_B_KLF1': 'Erythroid · KLF1 (GSE33979)', 'PANC_RFX6': 'Pancreas · RFX6'}
SHORT = {'CARDIACV_TBX5': 'TBX5', 'CARDIACV_NKX2-5': 'NKX2-5', 'GATA1_GATA1': 'GATA1', 'KLF1_KLF1': 'KLF1', 'KLF1_B_KLF1': 'KLF1_B', 'PANC_RFX6': 'RFX6'}
SYS = {'CARDIACV_TBX5': 'heart', 'CARDIACV_NKX2-5': 'heart', 'GATA1_GATA1': 'erythroid', 'KLF1_KLF1': 'erythroid', 'KLF1_B_KLF1': 'erythroid', 'PANC_RFX6': 'pancreas'}
def fnum(x, d=2): return f'{x:.{d}f}'.replace('-', '−')
def fp(p): return ('%.1e' % p if p < 1e-3 else '%.3f' % p if p < .1 else '%.2f' % p).replace('e-0', 'e-').replace('e-', 'e−')
def star(h): return '***' if h < 1e-3 else '**' if h < 1e-2 else '*' if h < .05 else None

lv = pd.read_csv(KA + 'p1_levels.tsv', sep='\t'); S_LV = 'kochip_audit_2026_10_15/p1_levels.tsv'
cp = pd.read_csv(KA + 'fig19_contrast_pvalues.tsv', sep='\t'); S_CP = 'kochip_audit_2026_10_15/fig19_contrast_pvalues.tsv'
bm = pd.read_csv(KA + 'p1b_module.tsv', sep='\t').set_index('unit_id'); S_BM = 'kochip_audit_2026_10_15/p1b_module.tsv'
dc = pd.read_csv(FD + 'heart5/decomp_summary.tsv', sep='\t'); S_DC = 'figdec_2026_10_16/heart5/decomp_summary.tsv'
cx = pd.read_csv(FD + 'crossed/summary_excess.tsv', sep='\t'); S_CX = 'figdec_2026_10_16/crossed/summary_excess.tsv'
ps = pd.read_csv(FD + 'pearsign/summary_excess.tsv', sep='\t'); S_PS = 'figdec_2026_10_16/pearsign/summary_excess.tsv'
pp = pd.read_csv(FD + 'pearsign/paired_iso_minus_pear.tsv', sep='\t'); S_PP = 'figdec_2026_10_16/pearsign/paired_iso_minus_pear.tsv'
fh = pd.read_csv(FD + 'spec53/f1_holm6.tsv', sep='\t'); S_FH = 'figdec_2026_10_16/spec53/f1_holm6.tsv'

fig = plt.figure(figsize=(7.2, 10.2))
gs = GridSpec(1, 1, figure=fig, left=.075, right=.985, top=.985, bottom=.865)
gsb = GridSpec(1, 1, figure=fig, left=.075, right=.985, top=.80, bottom=.60)
gsc = GridSpec(1, 1, figure=fig, left=.075, right=.985, top=.44, bottom=.105)

# ------------------------------------------------ a: schematic
ax = fig.add_subplot(gs[0]); fig.text(.012, .965, 'a', fontsize=10, fontweight='bold', va='bottom'); ax.set_xlim(0, 100); ax.set_ylim(0, 30); ax.axis('off')
LC = ['#c0392b', '#e8a39a', '#9aa0a6', '#9bb8d9', '#2c5d9e']   # level colours, verbatim gb_kochip_fig_p.py `cols`
LTX = ['w', 'k', 'k', 'k', 'w']
cw, ch, gx, gy0 = 19.5, 7.2, 12, 7.0
xs = [gx, gx + cw + 1, gx + 2 * (cw + 1)]
def cell(x, y, h, i, t):
    ax.add_patch(FancyBboxPatch((x, y), cw, h, boxstyle='round,pad=0.15,rounding_size=0.8', fc=LC[i], ec='#444', lw=.6))
    ax.text(x + cw / 2, y + h / 2 + 1.0, f'L{i+1}', ha='center', va='center', fontsize=8, fontweight='bold', color=LTX[i])
    ax.text(x + cw / 2, y + h / 2 - 1.6, t, ha='center', va='center', fontsize=5.6, color=LTX[i])
cell(xs[0], gy0 + ch + 1, ch, 0, 'KO-down, bound'); cell(xs[0], gy0, ch, 1, 'KO-down, not bound')
cell(xs[1], gy0, 2 * ch + 1, 2, 'non-responder (bound or not)')
cell(xs[2], gy0, ch, 3, 'KO-up, not bound'); cell(xs[2], gy0 + ch + 1, ch, 4, 'KO-up, bound')
for k_, t in enumerate(['KO-down', 'non-responder', 'KO-up']): ax.text(xs[k_] + cw / 2, gy0 + 2 * ch + 3.4, t, ha='center', va='center', fontsize=6.5, fontweight='bold')
ax.text(gx - 1.5, gy0 + ch * 1.5 + 1, 'ChIP-bound', ha='right', va='center', fontsize=6.3); ax.text(gx - 1.5, gy0 + ch * .5, 'not bound', ha='right', va='center', fontsize=6.3)
ax.text(xs[2] + cw + 2, gy0 + ch + .5, 'expected T:\nL1 > L2 > L3 > L4 > L5\n\nChIP: evaluation only;\nbinding \u2260 direct\nregulation.\nT = sign(iso)\u00b7F(Z)', ha='left', va='center', fontsize=5.6, color='#333', linespacing=1.15)

# ------------------------------------------------ b: per-system L1->L5

gb = gsb[0].subgridspec(1, 6, wspace=.42)
PAIRS = [('L1>L2', 0, 1), ('L4>L5', 3, 4), ('L3>L4', 2, 3), ('L2>L3', 1, 2)]
bax = []
for k, u in enumerate(UNITS):
    a = fig.add_subplot(gb[k]); bax.append(a); d = lv[lv.unit == DISP[u].replace('·', '·')]
    assert len(d) == 5, (u, len(d)); d = d.reset_index(drop=True)
    cc = cp[(cp.unit == DISP[u]) & (cp.score == 'iso')]; assert len(cc) == 6
    lo_all = np.nanmin(np.where(d.n > 0, d.mean_lo.fillna(d['mean']), np.nan)); hi_all = np.nanmax(d.mean_hi.fillna(d['mean'])); sp = hi_all - lo_all
    for i in range(5):
        n = int(rec('4b', f'{SHORT[u]} L{i+1} n', d.n[i], S_LV, f'unit={DISP[u]}|level={d.level[i]}|col=n')); m = rec('4b', f'{SHORT[u]} L{i+1} mean T', d['mean'][i], S_LV, f'unit={DISP[u]}|level={d.level[i]}|col=mean')
        lo = d.mean_lo[i]; hi = d.mean_hi[i]
        if np.isfinite(lo):
            rec('4b', f'{SHORT[u]} L{i+1} mean lo', lo, S_LV, f'unit={DISP[u]}|level={d.level[i]}|col=mean_lo'); rec('4b', f'{SHORT[u]} L{i+1} mean hi', hi, S_LV, f'unit={DISP[u]}|level={d.level[i]}|col=mean_hi')
            a.errorbar(i + 1, m, yerr=[[m - lo], [hi - m]], fmt='o', color=LC[i], capsize=2, ms=3.6, lw=1.0, mfc=LC[i])
        else: a.plot(i + 1, m, 'o', color=LC[i], ms=3.6, mfc='white')   # n=1 -> no CI
    HB = {'L1>L2': .90, 'L2>L3': .77, 'L3>L4': .64, 'L4>L5': .51}
    for lab, i, j in PAIRS:
        r = cc[cc.contrast == lab]; assert len(r) == 1; r = r.iloc[0]
        p = rec('4b', f'{SHORT[u]} {lab} raw p', r.p_raw, S_CP, f'unit={DISP[u]}|score=iso|contrast={lab}|col=p_raw'); h = rec('4b', f'{SHORT[u]} {lab} Holm6', r.p_holm6, S_CP, f'unit={DISP[u]}|score=iso|contrast={lab}|col=p_holm6')
        st = star(h) if np.isfinite(h) else None; col = 'k' if st else GREY
        txt = ((st or 'n.s.') + ' ' + fp(p)) if np.isfinite(p) else 'n<3'; y = HB[lab]
        a.plot([i + 1, j + 1], [y, y], color=col, lw=.8); a.text((i + j) / 2 + 1, y + .012, txt, ha='center', va='bottom', fontsize=4.2, color=col)
    a.axhline(0, color=GREY, lw=.5, ls='--'); a.set_xlim(.4, 5.6); a.set_ylim(-1, 1); a.set_yticks([-1, -.5, 0, .5, 1]); assert np.nanmin(d.mean_lo.fillna(d['mean'])) > -1 and np.nanmax(d.mean_hi.fillna(d['mean'])) < 1
    a.set_xticks(range(1, 6)); a.set_xticklabels([f'L{i+1}\n{int(d.n[i])}' for i in range(5)], fontsize=4.4)
    a.set_title(SHORT[u], fontsize=6.8, pad=3, fontweight='bold')
    a.text(.5, 1.0 + .0, '', transform=a.transAxes)
    cgb = bm.loc[u, 'c']; c_lo, c_hi = bm.loc[u, 'c_mod_lo'], bm.loc[u, 'c_mod_hi']
    rec('4b', f'{SHORT[u]} GB c (iso)', cgb, S_BM, f'unit_id={u}|col=c'); rec('4b', f'{SHORT[u]} GB c module lo', c_lo, S_BM, f'unit_id={u}|col=c_mod_lo'); rec('4b', f'{SHORT[u]} GB c module hi', c_hi, S_BM, f'unit_id={u}|col=c_mod_hi')
    dp = dc[(dc.unit == u) & (dc.row == 'T_pear')]; assert len(dp) == 1; dp = dp.iloc[0]
    pc = rec('4b', f'{SHORT[u]} Pearson-sign c', dp.c_total, S_DC, f'unit={u}|row=T_pear|col=c_total'); pl_ = rec('4b', f'{SHORT[u]} Pearson-sign c lo', dp.c_total_lo, S_DC, f'unit={u}|row=T_pear|col=c_total_lo'); ph = rec('4b', f'{SHORT[u]} Pearson-sign c hi', dp.c_total_hi, S_DC, f'unit={u}|row=T_pear|col=c_total_hi')
    xm = (a.get_position().x0 + a.get_position().x1) / 2
    fig.text(xm, .566, f'GB c {fnum(cgb)} [{fnum(c_lo)}, {fnum(c_hi)}]', ha='center', va='top', fontsize=5.0, color=BLUE)
    fig.text(xm, .553, f'Pearson-sign c {fnum(pc)} [{fnum(pl_)}, {fnum(ph)}]', ha='center', va='top', fontsize=4.3, color=ORG)
    if k == 0: a.set_ylabel('mean T (signed percentile, −1..1), 95% CI', fontsize=6)
# system bars
for lab, ks in (('heart', (0, 1)), ('erythroid', (2, 3, 4)), ('pancreas', (5,))):
    x0_ = bax[ks[0]].get_position().x0; x1_ = bax[ks[-1]].get_position().x1; yb = bax[0].get_position().y1 + .028
    fig.add_artist(plt.Line2D([x0_, x1_], [yb, yb], color='#777', lw=.8)); fig.text((x0_ + x1_) / 2, yb + .003, lab, ha='center', va='bottom', fontsize=6.5, color='#444')
fig.text(.012, .835, 'b', fontsize=10, fontweight='bold', va='bottom')
fig.text(.075, .532, 'n per level is printed under the tick labels (KLF1_B L5: n=1, no CI).  Brackets: one-sided Mann\u2013Whitney in the expected direction (number = raw p); stars = Holm over the 6 contrasts per TF\n'
            '(the four adjacent contrasts are drawn; * <.05, ** <.01, *** <.001; grey = n.s.).  GB c = weighted five-level ordinal concordance (normalised Jonckheere\u2013Terpstra) of T, module-bootstrap CI.\n'
            'Pearson-sign c = the same statistic with sign(Pearson Ms\u2013Mu) replacing the iso sign (CI: gene bootstrap, decomp_summary).  Exploratory, post hoc.',
            fontsize=5.0, ha='left', va='top', color='#333', linespacing=1.3)

# ------------------------------------------------ c: specificity
gc = gsc[0].subgridspec(1, 4, width_ratios=[3, 1.5, 3, 1.5], wspace=.2)
yp = {u: 0 for u in UNITS}; ypos = []; y = 0; gaps = {'CARDIACV_NKX2-5': 0, 'KLF1_B_KLF1': 0}
order = UNITS; yv = {}
cur = 0
for i, u in enumerate(order):
    if i > 0 and SYS[u] != SYS[order[i - 1]]: cur += .8
    yv[u] = cur; cur += 1.0
ymax = cur
def get(df, fam, cell, met, u):
    r = df[(df.unit == u) & (df.family == fam) & (df.group == 'both') & (df.cell == cell) & (df.metric == met)]; assert len(r) == 1, (fam, cell, met, u, len(r)); return r.iloc[0]
panels = [('c', 'five-level c', 'direction BA (responder sign accuracy)')]
axc = fig.add_subplot(gc[0]); axb = fig.add_subplot(gc[2]); fig.text(.012, .488, 'c', fontsize=10, fontweight='bold', va='bottom')
DX = {'iso_sign': -.24, 'iso_mag': 0, 'pear_sign': .24}
for ax_, met, ttl in ((axc, 'c', 'five-level c: focal − alternative'), (axb, 'BA', 'direction BA: focal − alternative')):
    for u in order:
        y_ = yv[u]
        items = [('iso_sign', ('crossed', 'isoF', 'Fa'), BLUE, 'o', True), ('pear_sign', ('pearsign', 'pearF', 'Fa'), ORG, 's', True)]
        if met == 'c': items.insert(1, ('iso_mag', ('crossed', 'isoF', 'aF'), BLUE, 'D', False))
        for nm, (src, fam, cell), col, mk, fill in items:
            df, S = (cx, S_CX) if src == 'crossed' else (ps, S_PS); r = get(df, fam, cell, met, u); key = f'unit={u}|family={fam}|group=both|cell={cell}|metric={met}'
            v = rec('4c', f'{SHORT[u]} {nm} {met} excess', r.excess, S, key + '|col=excess'); lo = rec('4c', f'{SHORT[u]} {nm} {met} lo', r.ci_lo, S, key + '|col=ci_lo'); hi = rec('4c', f'{SHORT[u]} {nm} {met} hi', r.ci_hi, S, key + '|col=ci_hi')
            yy = y_ + DX[nm]
            ax_.plot([lo, hi], [yy, yy], color=col, lw=.9, zorder=2); ax_.plot(v, yy, marker=mk, ms=3.4, mfc=col if fill else 'white', mec=col, mew=.9, ls='', zorder=3)
        # reference: paired iso - Pearson (Holm over 6 units, per metric)
        pr = pp[(pp.unit == u) & (pp.metric == met)]; hr = fh[(fh.unit == u) & (fh.metric == met)]; assert len(pr) == 1 and len(hr) == 1
        dd = rec('4c', f'{SHORT[u]} paired iso-Pearson {met}', pr.iloc[0]['diff'], S_PP, f'unit={u}|metric={met}|col=diff'); hh = rec('4c', f'{SHORT[u]} paired iso-Pearson {met} Holm6', hr.iloc[0].p_holm6, S_FH, f'unit={u}|metric={met}|col=p_holm6')
        ax_.text(1.02, y_, f'{"+" if dd >= 0 else ""}{fnum(dd, 3)}  (Holm {fnum(hh, 3) if hh >= .005 else "<.01"})'.replace('+', '+'), fontsize=4.8, ha='left', va='center', transform=matplotlib.transforms.blended_transform_factory(ax_.transAxes, ax_.transData), color='#333')
    ax_.axvline(0, color='k', lw=.5); ax_.set_ylim(ymax - .3, -.7); ax_.set_yticks([yv[u] for u in order]); ax_.set_yticklabels([SHORT[u] for u in order], fontsize=6)
    ax_.set_xlabel('excess = focal − median of 20 matched alternative TFs\n(95% CI, class-stratified gene bootstrap)', fontsize=5.6, labelpad=2); ax_.set_title(ttl, fontsize=6.3, pad=12)
    ax_.text(1.02, -.95, 'Δ iso − Pearson\n(paired, Holm 6)', fontsize=4.8, ha='left', va='bottom', color='#333', transform=matplotlib.transforms.blended_transform_factory(ax_.transAxes, ax_.transData))
    ax_.axhline(yv['CARDIACV_NKX2-5'] + .5 + .0, color='#eee', lw=.4)
fig.text(.5, .985, '', fontsize=1)
axc.legend(handles=[Line2D([], [], marker='o', ms=3.4, ls='', color=BLUE, label='iso sign swapped (focal magnitude)'), Line2D([], [], marker='D', ms=3.4, ls='', mfc='white', color=BLUE, label='magnitude swapped (focal iso sign)'),
                    Line2D([], [], marker='s', ms=3.4, ls='', color=ORG, label='Pearson sign swapped (reference)')], loc='upper left', bbox_to_anchor=(-.05, -.20), ncol=3, frameon=False, fontsize=5.4, handletextpad=.1, columnspacing=.8)
fig.text(.04, .488, 'Is the polarity regulator-specific?', fontsize=8, fontweight='bold', va='bottom')
fig.text(.55, .006, 'Direction BA depends on sign only, so a magnitude swap is not shown there.', fontsize=5.0, va='bottom', color='#333')
fig.savefig(OUT + 'Fig4_v6.jpg', dpi=300, pil_kwargs={'quality': 95})
pd.DataFrame(LED, columns=['panel', 'quantity', 'value', 'source_file', 'row_key']).to_csv(OUT + 'numbers_ledger_Fig4_v6.tsv', sep='\t', index=False)
print('done', len(LED))
