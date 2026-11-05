"""R2-fix T2_realk0 (2026-11-01): conditioning-attribution table + forest + depth curves for manuscript Fig 2.
AGGREGATION + PLOTTING ONLY on the frozen rebuild (results/signed/fig3d_rebuild_2026_10_29/full) and depth_multi_B9999. No score/null changes.
usage (server): script OUT   [no heavy compute; --B/--seed not applicable: p_boot/CI are the cached 9,999-draw module-bootstrap values]
Holm used here (this script): within direction x metric, (a) 12 system tests = {K0,SPL,RND,PS} x {PANC,BM,HEART}  (p_holm12)
                                                              (b) 4 EQSYS tests = {K0,SPL,RND,PS}                  (p_holm4_EQSYS)
Producer's own p_holm column (stored) = EQSYS Holm-4 for KOup/KOdown only, AUROC+APLIFT; asserted equal to (b) where filled."""
import sys, os, json, hashlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/results/signed/fig3d_rebuild_2026_10_29/')
OUT = sys.argv[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
sha = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
T = pd.read_csv(R + 'full/fig3d_rebuild_table.tsv', sep='\t'); C = pd.read_csv(R + 'full/contrasts_all_directions.tsv', sep='\t')
ENDP = ['PANC_RFX6', 'BM_GATA1_acute', 'HEART_TBX5', 'HEART_NKX2-5']; SYSL = ['PANC', 'BM', 'HEART']; LEV = ENDP + SYSL + ['EQSYS']
ARMS = ['K0', 'SPL', 'RND', 'PS']; DIRS = ['ALL', 'KOup', 'KOdown']; METS = ['AUROC', 'APLIFT']

def holm(p):
    p = np.asarray(p, float); m = len(p); o = np.argsort(p, kind='stable'); out = np.full(m, np.nan); run = 0.0
    for i, k in enumerate(o): run = max(run, (m - i) * p[k]); out[k] = min(1.0, run)
    return out

lv = T[T.kind == 'level'].set_index(['direction', 'metric', 'level', 'arm']); dm = T[T.kind == 'D_minus'].set_index(['direction', 'metric', 'level', 'arm'])
rows = []
for d in DIRS:
    for m in METS:
        # Holm families
        sys_idx = [(l, a) for l in SYSL for a in ARMS]; p12 = holm([dm.loc[(d, m, l, a), 'p_boot'] for l, a in sys_idx]); H12 = dict(zip(sys_idx, p12))
        p4 = holm([dm.loc[(d, m, 'EQSYS', a), 'p_boot'] for a in ARMS]); H4 = dict(zip(ARMS, p4))
        for l in LEV:
            for a in ARMS:
                r = dm.loc[(d, m, l, a)]; stored = r['p_holm']
                if l == 'EQSYS' and np.isfinite(stored): assert abs(stored - H4[a]) < 1e-12, ('stored Holm-4 mismatch', d, m, a, stored, H4[a])
                rows.append(dict(direction=d, metric=m, level=l, control_arm=a, D=lv.loc[(d, m, l, 'D'), 'value'], arm_value=lv.loc[(d, m, l, a), 'value'],
                                 D_minus_arm=r['value'], ci_lo=r['lo'], ci_hi=r['hi'], p_boot=r['p_boot'], n_valid=r['n_valid'],
                                 p_holm12_system=H12.get((l, a), np.nan), p_holm4_EQSYS=H4[a] if l == 'EQSYS' else np.nan, producer_p_holm=stored, producer_family=r['family']))
O = pd.DataFrame(rows); O.to_csv(OUT + 'T2_attribution_table.tsv', sep='\t', index=False)
# cross-check vs the producer's contrasts file (ALL; D-K0/D-SPL/D-PS have stored Holm columns)
for _, c in C[C.contrast.isin(['D-K0', 'D-SPL', 'D-PS'])].iterrows():
    a = c.contrast[2:]; x = O[(O.direction == c.direction) & (O.metric == c.metric) & (O.level == c.level) & (O.control_arm == a)]
    if len(x):
        assert abs(x.D_minus_arm.iloc[0] - c.value) < 1e-12 and abs(x.p_boot.iloc[0] - c.p_boot) < 1e-12, ('contrasts file mismatch', c.to_dict())
        for col, mine in (('p_holm4_EQSYS', 'p_holm4_EQSYS'), ('p_holm12_system', 'p_holm12_system')):
            if np.isfinite(c[col]) and np.isfinite(x[mine].iloc[0]) and abs(c[col] - x[mine].iloc[0]) > 1e-9:
                print('NOTE holm differs from contrasts_all_directions.tsv', c.direction, c.metric, c.level, a, col, 'stored', c[col], 'mine', x[mine].iloc[0])

# ---- parity anchors
g = lambda l, a='D', d='ALL', m='AUROC': lv.loc[(d, m, l, a), 'value']; h = lambda l, a, d='ALL', m='AUROC': dm.loc[(d, m, l, a)]
for l, v in (('PANC_RFX6', .5855), ('BM_GATA1_acute', .6477), ('HEART_TBX5', .5795), ('HEART_NKX2-5', .5493)): print('anchor', l, 'rebuild', round(g(l), 4), 'task', v); assert abs(g(l) - v) < (6e-4 if l.startswith('HEART') else 6e-5), (l, g(l), v)   # TBX5: task anchor .5795 vs rebuild .5790 (flagged)
assert abs(g('PANC', 'K0') - .5366) < 6e-5 and abs(g('BM', 'K0') - .6401) < 6e-5
for l, a, v, p in (('PANC', 'K0', .0488, .0294), ('HEART', 'K0', .0522, .0032), ('EQSYS', 'K0', .0363, .0278), ('EQSYS', 'SPL', .0329, .0746), ('EQSYS', 'RND', .0357, .030)):
    r = h(l, a); assert abs(r['value'] - v) < 6e-5 and abs(r['p_boot'] - p) < 6e-4, (l, a, r['value'], r['p_boot'], v, p)
print('PARITY OK')

# ---- Markdown
f3 = lambda x: f'{x:+.4f}'; fp = lambda x: 'NA' if not np.isfinite(x) else f'{x:.4f}'
md = ['# T2_realk0: conditioning attribution (D = production GeoBRIDGE; diff = D - control arm)', '',
      'Source: results/signed/fig3d_rebuild_2026_10_29/full (fig3d_rebuild_table.tsv). CI = 95% percentile of 9,999 synchronized module-bootstrap draws; p_boot two-sided (nominal).',
      'Holm: p_holm12 = Holm over 12 tests {K0,SPL,RND,PS} x {PANC,BM,HEART} within direction x metric (computed here; PANC/BM rows duplicate the single-endpoint tests); '
      'p_holm4 = Holm over the 4 EQSYS arms within direction x metric (computed here). Producer-stored p_holm = EQSYS Holm-4 for KOup/KOdown only (equal to p_holm4 where filled, asserted). '
      'Endpoint rows: descriptive, no Holm.', '']
for d in DIRS:
    for m in METS:
        md += [f'## {d} / {m}', '', '| level | arm | D | arm | D-arm | 95% CI | p_boot | Holm12 | Holm4(EQSYS) |', '|---|---|---|---|---|---|---|---|---|']
        for _, r in O[(O.direction == d) & (O.metric == m)].iterrows():
            md.append(f'| {r.level} | {r.control_arm} | {r.D:.4f} | {r.arm_value:.4f} | {f3(r.D_minus_arm)} | [{r.ci_lo:+.4f}, {r.ci_hi:+.4f}] | {fp(r.p_boot)} | {fp(r.p_holm12_system)} | {fp(r.p_holm4_EQSYS)} |')
        md.append('')
open(OUT + 'T2_attribution_table.md', 'w').write('\n'.join(md))

# ---- Forest
plt.rcParams.update({'font.family': 'sans-serif', 'font.size': 7, 'axes.spines.top': False, 'axes.spines.right': False})
COL = {'K0': '#e8a39a', 'SPL': '#8c8c8c', 'RND': '#bdbdbd', 'PS': '#c0392b'}
def forest(dirs, fn, m='AUROC'):
    fig, axs = plt.subplots(1, len(dirs), figsize=(4.8 * len(dirs), 3.4), squeeze=False)
    for ax, d in zip(axs[0], dirs):
        y = 0; yt = []; yl = []
        for l in SYSL + ['EQSYS']:
            for a in ARMS:
                r = O[(O.direction == d) & (O.metric == m) & (O.level == l) & (O.control_arm == a)].iloc[0]
                ax.errorbar(r.D_minus_arm, y, xerr=[[r.D_minus_arm - r.ci_lo], [r.ci_hi - r.D_minus_arm]], fmt='o', ms=3, color=COL[a], ecolor=COL[a], lw=1)
                ph = r.p_holm4_EQSYS if l == 'EQSYS' else r.p_holm12_system
                ax.text(1.02, y, f'p={r.p_boot:.3f}; Holm={ph:.3f}', transform=ax.get_yaxis_transform(), va='center', fontsize=5)
                yt.append(y); yl.append(f'{l}  D-{a}'); y += 1
            y += .8
        ax.axvline(0, color='k', lw=.6); ax.set_yticks(yt); ax.set_yticklabels(yl); ax.invert_yaxis(); ax.set_xlabel(f'{m} difference (D - control)')
        ax.set_title(f'{d} (Holm12 per system; Holm4 for EQSYS)', fontsize=6.5)
        ax.set_ylabel('')
    fig.subplots_adjust(left=.14 if len(dirs) > 1 else .22, right=.72 if len(dirs) == 1 else .90, wspace=1.0, bottom=.15); fig.savefig(OUT + fn, dpi=300); plt.close(fig)
forest(['ALL'], 'T2_forest_ALL_AUROC.png'); forest(['KOup', 'KOdown'], 'T2_forest_SI_KOup_KOdown_AUROC.png')

# ---- Depth curves (cum0-K, direction ALL, AUROC), depth_multi_B9999
dd = pd.concat([pd.read_csv(R + f'depth_multi_B9999/{s}/depth_multi.tsv', sep='\t') for s in ('panc', 'bm', 'heart')])
dd = dd[(dd.direction == 'ALL') & (dd.metric == 'AUROC') & dd.subset.str.startswith('cum0-')].copy(); dd['K'] = dd.subset.str[5:].astype(int)
dd.sort_values(['endpoint', 'K']).to_csv(OUT + 'T2_depth_cum_ALL_AUROC.tsv', sep='\t', index=False)
chk = {'PANC_RFX6': .5855, 'BM_GATA1_acute': .6477, 'HEART_TBX5': .5795, 'HEART_NKX2-5': .5493}
fig, axs = plt.subplots(1, 4, figsize=(9.5, 2.5), sharey=False)
for ax, e in zip(axs, ENDP):
    x = dd[dd.endpoint == e].sort_values('K'); assert list(x.K) == list(range(31)), (e, len(x))
    assert abs(x[x.K == 8].value.iloc[0] - chk[e]) < 6e-4, (e, x[x.K == 8].value.iloc[0])   # cum0-8 == production D
    ax.fill_between(x.K, x.lo, x.hi, color='#e8a39a', alpha=.25, lw=0); ax.plot(x.K, x.value, color='#c0392b', lw=1.2)
    ax.plot([0], [x.value.iloc[0]], 'o', color='#e8a39a', ms=4); ax.plot([8], [x[x.K == 8].value.iloc[0]], 'o', color='#c0392b', ms=4)
    ax.axhline(.5, color='k', lw=.5, ls=':'); ax.axvline(8, color='#8c8c8c', lw=.5, ls='--'); ax.set_title(e.replace('_', ' '), fontsize=7); ax.set_xlabel('max aggregated depth K (depths 0..K)')
axs[0].set_ylabel('AUROC (ALL)'); fig.tight_layout(); fig.savefig(OUT + 'T2_depth_curves_4endpoints.png', dpi=300); plt.close(fig)
json.dump(dict(inputs={R + 'full/' + f: sha(R + 'full/' + f) for f in ('fig3d_rebuild_table.tsv', 'contrasts_all_directions.tsv', 'depth_profiles_all.tsv', 'receipt.json')} |
          {R + f'depth_multi_B9999/{s}/depth_multi.tsv': sha(R + f'depth_multi_B9999/{s}/depth_multi.tsv') for s in ('panc', 'bm', 'heart')}, script_sha=sha(os.path.abspath(__file__))),
          open(OUT + 'receipt.json', 'w'), indent=1)
print('done', OUT)
