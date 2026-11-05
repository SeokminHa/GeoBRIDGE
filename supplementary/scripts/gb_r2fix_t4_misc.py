"""R2-fix T4_misc (2026-11-01): (1) tie fractions per endpoint x method on Fig 2 UOLD common rows; (2) HOM multi-class ortholog audit for PANC rows.
EVALUATION ONLY. usage (server, repo root): script OUT. Reuses the Fig3 producer prefix (exec), B=2 (no bootstrap output used)."""
import sys, os, json
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); OUT = sys.argv[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
P = R + 'scratch/gb_fig3_common_v2_2026_10_28.py'
src = open(P).read().replace('\r', ''); src = src[:src.index('if PLOTONLY:   #')]
ns = {'__file__': P, '__name__': 'prefix'}; sys.argv = ['x', OUT + '_scr', '--B', '2', '--uold']; exec(compile(src, P, 'exec'), ns)
import numpy as np, pandas as pd
nsA, nsB, METH, EPS = ns['nsA'], ns['nsB'], ns['METH'], ns['EPS']; UOLD = ns['UOLD']
rows, ROWSET = [], {}
for lab in EPS:
    X = nsB['D'][lab] if lab == 'BM_GATA1_acute' else nsA['D'][lab]; V = dict(X['V'])
    U = np.asarray(X['U'], bool); Uc = U & np.all([np.isfinite(V[m]) for m in METH], 0)
    EPT = {x[0]: x for x in nsA['EP9']}; zv = np.load(f"{nsA['VEC']}ep_{EPT[lab][5]}.vec.npz", allow_pickle=True)
    oldvel = zv['cog::Velorama'].astype(float); Uc = Uc & np.isfinite(oldvel)
    ROWSET[lab] = (np.asarray(X['g']).astype(str), Uc, np.asarray(X['y'], bool))
    for m in METH:
        v = np.asarray(V[m], float)[Uc]; n = len(v); u, c = np.unique(v, return_counts=True); j = c.argmax(); tv = u[j]
        # tie block at bottom = largest tied block's value <= min of all other values (ranking ascending: low score = bottom)
        other = v[v != tv]; bottom = bool(other.size == 0 or tv <= other.min()); top = bool(other.size == 0 or tv >= other.max())
        rows.append(dict(endpoint=lab, method=m, n_rows=n, largest_tie_frac=c.max() / n, largest_tie_n=int(c.max()), largest_tie_value=float(tv), n_distinct=len(u),
                         frac_exact_zero=float((v == 0).mean()), tie_block_at_bottom=bottom, tie_block_at_top=top, n_pos_in_tie_block=int(ROWSET[lab][2][Uc][v == tv].sum())))
T = pd.DataFrame(rows); T.to_csv(OUT + 'tie_fractions.tsv', sep='\t', index=False); print(T.to_string(float_format=lambda x: f'{x:.4g}'))
for lab in EPS: print(lab, 'n_rows', int(ROWSET[lab][1].sum()))
# ---- item 2: ortholog audit (replicates gb_dev_evaluate.truth_panc h2m exactly, plus per-human-symbol class tracking)
hom = pd.read_csv(f"{R}data/orthology/HOM_MouseHumanSequence.rpt", sep="\t", low_memory=False)
key = hom.columns[0]; org = hom["Common Organism Name"]
mm = hom[org.str.startswith("mouse")].groupby(key)["Symbol"].apply(list); hs = hom[org.str.startswith("human")].groupby(key)["Symbol"].apply(list)
keys = list(set(mm.index) & set(hs.index))   # same iteration order as adapter (set of same ints)
h2m = {hs[k][0].upper(): mm[k][0] for k in keys if len(mm[k]) == 1 and len(hs[k]) == 1}
h2m_first = {}   # who would win if first-wins
for k in keys:
    if len(mm[k]) == 1 and len(hs[k]) == 1: h2m_first.setdefault(hs[k][0].upper(), mm[k][0])
hcls = {}   # human symbol -> list of (class key, human members, mouse members) over ALL classes with that symbol
for k in hs.index:
    for s in hs[k]: hcls.setdefault(s.upper(), []).append((int(k), hs[k], mm[k] if k in mm.index else []))
multi = {s: v for s, v in hcls.items() if len(v) > 1}
print('human symbols in >1 HOM class:', len(multi), '; of those with >=1 one-to-one class:', sum(any(len(h) == 1 and len(m) == 1 for _, h, m in v) for v in multi.values()))
print('INS classes:', hcls.get('INS')); print('INS ->', h2m.get('INS'), '(first-wins would be', h2m_first.get('INS'), ')')
de = pd.read_csv(f"{R}real_1_Pancreas/external_validation/aldous_rfx6/results/aldous_PPs_WT_vs_KO_DE.tsv", sep="\t")
dehs = set(de.gene_name.astype(str).str.upper())
g, Uc, y = ROWSET['PANC_RFX6']; gr = g[Uc]; yr = y[Uc]; gset = {s.upper(): i for i, s in enumerate(gr)}
out = []
for hsym, mg in h2m.items():
    if mg.upper() in gset and hsym in multi and hsym in dehs:
        i = gset[mg.upper()]; out.append(dict(mouse_row=mg, human_symbol=hsym, label_responder=bool(yr[i]), n_classes=len(multi[hsym]),
              classes='; '.join(f"{k}:H{h}/M{m}" for k, h, m in multi[hsym]), first_wins_mouse=h2m_first[hsym]))
O = pd.DataFrame(out); O.to_csv(OUT + 'ortholog_multiclass_rows.tsv', sep='\t', index=False); print(O.to_string())
# all PANC rows whose label chain (human DE symbol -> h2m -> row) exists; for denominator
lab_rows = {h2m[h] for h in dehs if h in h2m and h2m[h].upper() in gset}
print('PANC eval rows', len(gr), 'rows reachable by a labelled human DE symbol', len(lab_rows), 'positives', int(yr.sum()), 'multiclass-derived rows', len(O), 'of which responders', int(O.label_responder.sum()) if len(O) else 0)
# also: mouse rows that were DROPPED because their human symbol is multi-class and no class is 1:1 (cannot be labelled) -- count human DE symbols in multi with no h2m entry
dropped = [s for s in dehs if s in multi and s not in h2m]; print('human DE symbols in >1 class with NO 1:1 class (unmapped):', len(dropped))
json.dump(dict(n_multi_symbols=len(multi), ins_classes=str(hcls.get('INS')), ins_to=h2m.get('INS'), ins_first_wins=h2m_first.get('INS'), n_dropped_unmapped=len(dropped)), open(OUT + 'receipt.json', 'w'), indent=1)
