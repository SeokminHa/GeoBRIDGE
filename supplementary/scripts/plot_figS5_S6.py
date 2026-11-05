"""Fig. S5 (AUROC) and Fig. S6 (AUPRC lift): knockout-response prioritization by response direction.
Rows: KO-upregulated and KO-downregulated responders vs non-responders; columns: equal-system mean and the four endpoints.
Bars sorted by value, GeoBRIDGE red, GeoBRIDGE (k=0) pink; 95% module-bootstrap intervals (9,999 draws). Holm over the 7 comparators
from the stored two-sided bootstrap p within each endpoint x direction x metric; stars * <.05, ** <.01, *** <.005, black = GeoBRIDGE better,
blue = method better. Panel code is the one of figure3_knockout/scripts/gb_fig3bc_panels_2026_10_30.py (with --k0); the panels are then
placed side by side with row and column titles.
Reads data/figS5_S6_ko_direction.tsv, data/figS5_S6_k0.tsv. usage: python plot_figS5_S6.py --out OUTDIR [--data DATADIR]"""
import os, argparse
import numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib import font_manager
from statsmodels.stats.multitest import multipletests
from PIL import Image, ImageDraw, ImageFont
ap = argparse.ArgumentParser(); ap.add_argument('--data', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data')); ap.add_argument('--out', required=True)
ap.add_argument('--w', type=float, default=2.3); ap.add_argument('--h', type=float, default=1.9)
args = ap.parse_args(); O = args.out.rstrip('/') + '/'; P = O + 'panels/'; os.makedirs(P, exist_ok=True); W, H = args.w, args.h
T = pd.read_csv(os.path.join(args.data, 'figS5_S6_ko_direction.tsv'), sep='\t')
T = T[T.method != '|Pearson|'].copy()
T['p_holm'] = np.nan
for _, g in T[T.method != 'GeoBRIDGE'].groupby(['endpoint', 'direction', 'metric']):
    ok = g.p_boot.notna(); T.loc[g.index[ok], 'p_holm'] = multipletests(g.p_boot[ok], method='holm')[1]
T.drop(columns=['p_holm8']).to_csv(O + 'figS5_S6_holm7.tsv', sep='\t', index=False)
R_ = pd.read_csv(os.path.join(args.data, 'figS5_S6_k0.tsv'), sep='\t'); R_ = R_[R_.kind == 'level'].rename(columns={'level': 'endpoint'})
chk = R_[R_.arm == 'D'].merge(T[T.method == 'GeoBRIDGE'], on=['endpoint', 'direction', 'metric']); assert len(chk) and np.allclose(chk.value_x, chk.value_y, atol=1e-9), 'arm D != GeoBRIDGE'
K0 = R_[R_.arm == 'K0'].assign(method='GeoBRIDGE (k=0)')[['endpoint', 'direction', 'metric', 'method', 'value', 'lo', 'hi']]; print('k0 parity rows', len(chk))
plt.rcParams.update({'font.family': 'sans-serif', 'font.sans-serif': ['Helvetica', 'Nimbus Sans', 'Arial', 'DejaVu Sans'], 'font.size': 7,
                     'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': .6})
RED, GREY, BLUE = '#c0392b', '#b5b9be', '#2166ac'; PINK = '#e8a39a'
def stars(p): return '***' if p < .005 else '**' if p < .01 else '*' if p < .05 else ''
EPS = ['EQSYS', 'PANC_RFX6', 'BM_GATA1_acute', 'HEART_TBX5', 'HEART_NKX2-5']
for dr in ('KOup', 'KOdown'):
    for ep in EPS:
        for met, ref in (('APLIFT', 1.0), ('AUROC', 0.5)):
            d = T[(T.endpoint == ep) & (T.direction == dr) & (T.metric == met)].copy()
            if d.empty: continue
            gb = d[d.method == 'GeoBRIDGE']; rest = d[d.method != 'GeoBRIDGE'].sort_values('value', ascending=False)
            k0 = K0[(K0.endpoint == ep) & (K0.direction == dr) & (K0.metric == met)]; d = pd.concat([gb, k0, rest])
            fig, ax = plt.subplots(figsize=(W, H)); y = np.arange(len(d))[::-1]
            lo = min(d.lo.min(), ref); hi = max(d.hi.max(), ref); pad = .06 * (hi - lo); xr = hi + pad
            for yi, (_, r) in zip(y, d.iterrows()):
                ax.barh(yi, r.value - ref, left=ref, color=RED if r.method == 'GeoBRIDGE' else PINK if r.method == 'GeoBRIDGE (k=0)' else GREY, height=.66, lw=0)
                ax.plot([r.lo, r.hi], [yi, yi], color='#333333', lw=.7); ax.plot([r.lo, r.lo], [yi - .15, yi + .15], color='#333333', lw=.7); ax.plot([r.hi, r.hi], [yi - .15, yi + .15], color='#333333', lw=.7)
                if r.method not in ('GeoBRIDGE', 'GeoBRIDGE (k=0)') and np.isfinite(r.p_holm):
                    s = stars(r.p_holm)
                    if s: ax.text(xr, yi, s, ha='left', va='center', fontsize=7, color='k' if r.GB_minus > 0 else BLUE, clip_on=False)
            ax.set_xlim(lo - pad if met == 'AUROC' else max(0, lo - pad), xr); ax.axvline(ref, color='#666666', lw=.6, ls=':')
            ax.set_yticks(y); ax.set_yticklabels(d.method, fontsize=6.5); [t.set_color(RED if t.get_text() == 'GeoBRIDGE' else '#c98476') for t in ax.get_yticklabels() if t.get_text().startswith('GeoBRIDGE')]
            ax.set_xlabel('AUPRC lift' if met == 'APLIFT' else 'AUROC', fontsize=7); ax.tick_params(axis='x', labelsize=6.5)
            fig.savefig(P + f'{dr}_{ep}_{met}.png', dpi=300, bbox_inches='tight'); plt.close(fig)

# side-by-side layout: one row per response direction, one column per endpoint (equal panel cells, title strip above each row)
COLT = ['Equal-system mean', 'Pancreas · RFX6', 'Bone marrow · GATA1', 'Heart · TBX5', 'Heart · NKX2-5']
ROWT = [('a', 'KO-upregulated responders', 'KOup'), ('b', 'KO-downregulated responders', 'KOdown')]
FR = font_manager.findfont('DejaVu Sans'); FB = font_manager.findfont(font_manager.FontProperties(family='DejaVu Sans', weight='bold'))
fr, fb = ImageFont.truetype(FR, 34), ImageFont.truetype(FB, 40)
for met, name in (('AUROC', 'FigS5_direction_AUROC.png'), ('APLIFT', 'FigS6_direction_APLIFT.png')):
    ims = {(dr, ep): Image.open(P + f'{dr}_{ep}_{met}.png').convert('RGB') for _, _, dr in ROWT for ep in EPS}
    cw = max(i.size[0] for i in ims.values()); ch = max(i.size[1] for i in ims.values()); strip = 110
    C = Image.new('RGB', (cw * 5, 2 * (ch + strip)), 'white'); D = ImageDraw.Draw(C)
    for ri, (lab, title, dr) in enumerate(ROWT):
        y0 = ri * (ch + strip); D.text((10, y0 + 2), f'{lab}  {title}', font=fb, fill='black')
        for ci, ep in enumerate(EPS):
            D.text((ci * cw + cw // 2, y0 + 75), COLT[ci], font=fr, fill='black', anchor='mm')
            im = ims[(dr, ep)]; C.paste(im, (ci * cw + cw - im.size[0], y0 + strip))
    C.save(O + name, dpi=(300, 300)); print('wrote', name)
