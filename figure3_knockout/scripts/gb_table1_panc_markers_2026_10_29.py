"""Table 1 (after Fig 3): PANC endocrine marker analysis on an INDEPENDENTLY defined gene set (internal directive 2026-10-29). EVALUATION/BOOKKEEPING ONLY on frozen
Fig 3 scores and truth; NO ChIP / occupancy / direct-vs-indirect content. internal review review (internal design note (not shipped)).
Marker sets (fixed before reading any rank; neither GB scores nor RFX6-KO results define them):
  PRIMARY     CellMarker 2024 (Enrichr library data/genesets/CellMarker_2024.txt; Hu et al., Nucleic Acids Res 2023): every term matching
              '{Alpha|Beta|Delta|Pancreatic Polypeptide|Endocrine} Cell {Pancreas|Pancreatic Islet} {Human|Mouse}'. All CellMarker terms containing 'Pancrea'
              are exported with matched/excluded status (CellMarker 2024 has no epsilon-cell pancreatic term).
  SENSITIVITY PanglaoDB Augmented 2021 (Enrichr; co-expression-augmented signatures, human symbols): Alpha, Beta, Delta, Epsilon, Gamma (PP) Cells.
  Mapping contract (no cross-species name fallback): mouse-term symbols -> MGI mouse symbol (case-insensitive) = 'mouse_direct'; human symbols -> every mouse
  member of every MGI homology class containing that human symbol: one class with 1 human + 1 mouse = 'MGI_1to1'; anything else = 'MGI_multi' (all mouse
  members kept, flagged; e.g. INS -> Ins1, Ins2); not in MGI = 'unmapped' (dropped). The perturbed regulator RFX6 is removed. NB the frozen KO truth adapter
  (scratch/gb_truth_tables_2026_10_28.py) builds its human->mouse dict from 1:1 classes and therefore maps INS -> Ins1 only; truth is used unchanged.
Universe / truth / scores: frozen Fig 3d rebuild prefix (scratch/gb_fig3d_rebuild_2026_10_29.py, --endpoints PANC_RFX6); asserted Uc == UOLD for PANC.
Percentile = average-rank percentile among all UOLD rows (higher = higher priority). Ties are kept (never excluded): midrank for medians / Q3 half-credit, and
  top-K recall uses fractional allocation at a boundary tie (expected hits under random tie-breaking, as scratch/gb_canon_table.py:103-111). Per-method min/max
  rank percentiles are reported only as LOOSE individual rank bounds for the Q1 median. Tied fraction of UOLD per method is reported.
Metrics (questions kept separate): Q1 response-label-independent prioritisation of eligible markers (median percentile); Q1m detection-matched: Q1 minus the median
  percentile of non-marker UOLD genes re-weighted to the markers' spliced-detection decile distribution (observational annotation, ugb_all_rows.tsv);
  Q2 KO-responsive markers: median percentile, top-5%/10% recall (K = floor(f*N)), enrichment vs K/N and vs non-marker responders (disjoint background);
  Q3 AUROC responders vs non-responders within markers. Median = weighted median, averaging the two central values at an exact half split.
Uncertainty: synchronized module draws (conditional module-reweighting sensitivity with scores/percentiles/thresholds fixed; not donor or label uncertainty);
  paired GB - comparator per draw, 95% percentile interval, PT.boot_p; Holm across the 8 comparators within each (marker set, metric) for Q1, Q2 median,
  Q2 recall10, Q3 (primary families); everything else descriptive. Module diagnostics: occupied modules (responders / non-responders), max module share,
  valid draws, leave-one-module-out range of GB - comparator for Q1 and Q2 median.
Main table: ALL eligible KO-responsive PRIMARY markers (no selection step), with human truth symbol, mouse scored symbol, mapping route, KO effect/padj (human
  progenitor KO), subtype annotation and all nine percentiles.
usage (server, repo root): script OUT [--B 200]"""
import sys, os, json, re, hashlib
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
OUT = OUT0 = sys.argv[1].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True); rest = sys.argv[2:]
src = open(R + 'scratch/gb_fig3d_rebuild_2026_10_29.py').read().replace('\r', '')
src = src[:src.index("MT = ('AUROC', 'APLIFT', 'AUPRC'); DRAW = {}")]
sys.argv = ['x', OUT + '_rb', '--endpoints', 'PANC_RFX6'] + rest
exec(compile(src, 'rebuild_prefix', 'exec'))
OUT = OUT0
from scipy.stats import rankdata
sys.path[:0] = [R + 'scratch']
from gb_truth_tables_2026_10_28 import truth_table
GS_ = R + 'data/genesets/'; HOM = R + 'data/orthology/HOM_MouseHumanSequence.rpt'; ANNF = R + 'results/signed/fig3_gb_unique_2026_10_28/full3b/ugb_all_rows.tsv'; REG = 'RFX6'
METH = ['GeoBRIDGE', '|Pearson|', 'GENIE3', 'GRNBoost2', 'Velorama', 'RegDiffusion', 'DeepSEM', 'PIDC', 'FastTENET']; EXT = METH[1:]
# ---------- 1. marker sets + mapping contract
def enrichr(path):
    out = {}
    for line in open(path):
        p = line.rstrip('\n').split('\t'); out[p[0]] = [x.split(',')[0].strip() for x in p[2:] if x.strip()]
    return out
CM = enrichr(GS_ + 'CellMarker_2024.txt'); PG = enrichr(GS_ + 'PanglaoDB_Augmented_2021.txt')
pat = re.compile(r'^(Alpha|Beta|Delta|Pancreatic Polypeptide|Endocrine) Cell (Pancreas|Pancreatic Islet) (Human|Mouse)$')
pd.DataFrame([dict(term=k, n_genes=len(v), matched=bool(pat.match(k))) for k, v in CM.items() if 'pancrea' in k.lower()]).to_csv(OUT + 'provenance_CellMarker_pancreas_terms.tsv', sep='\t', index=False)
PRIM = {k: v for k, v in CM.items() if pat.match(k)}; SENS = {k: PG[k] for k in ['Alpha Cells', 'Beta Cells', 'Delta Cells', 'Epsilon Cells', 'Gamma (PP) Cells'] if k in PG}
H = pd.read_csv(HOM, sep='\t', dtype=str); key = H.columns[0]; org = H['Common Organism Name']
hm = H[org == 'human']; mm = H[org == 'mouse, laboratory']; cls_m = mm.groupby(key).Symbol.apply(list); cls_h = hm.groupby(key).Symbol.apply(list)
HCLS = hm.groupby(hm.Symbol.str.upper())[key].apply(list); MOUSE = {s.upper(): s for s in mm.Symbol}
def map_symbol(s, species):
    S = s.upper()
    if species == 'mouse':
        return ([S], 'mouse_direct') if S in MOUSE else ([], 'unmapped')
    if S not in HCLS.index: return [], 'unmapped'
    ks = HCLS[S]; ms = sorted({m.upper() for k in ks for m in cls_m.get(k, [])})
    if not ms: return [], 'unmapped'
    one = len(ks) == 1 and len(cls_m.get(ks[0], [])) == 1 and len(cls_h.get(ks[0], [])) == 1
    return ms, 'MGI_1to1' if one else 'MGI_multi'
def build(sets, name):
    rows = []
    for setname, genes in sets.items():
        sp = 'mouse' if setname.endswith('Mouse') else 'human'
        sub = setname.split(' Cell')[0].replace('Gamma (PP)', 'PP').replace('Pancreatic Polypeptide', 'PP')
        for s in genes:
            if s.upper() == REG: continue
            ms, route = map_symbol(s, sp)
            for m in (ms or [None]): rows.append(dict(markerset=name, source_term=setname, subtype=sub, published_symbol=s, term_species=sp, mouse_symbol=m, route=route))
    return pd.DataFrame(rows)
MS = {'PRIMARY_CellMarker2024': build(PRIM, 'PRIMARY_CellMarker2024'), 'SENS_PanglaoDB2021': build(SENS, 'SENS_PanglaoDB2021')}
for k, M in MS.items(): M.to_csv(OUT + f'markerset_{k}.tsv', sep='\t', index=False)
# ---------- 2. per-gene percentiles on UOLD
X = D['PANC_RFX6']['X']; g = np.char.upper(np.asarray(X['g']).astype(str)); V = dict(X['V']); Uc = D['PANC_RFX6']['Uc']; y0 = D['PANC_RFX6']['y0']; e = D['PANC_RFX6']['e']
assert np.array_equal(Uc, D['PANC_RFX6']['U']), 'Table 1 rows must equal UOLD'
TT = truth_table('PANC_RFX6').copy(); TT.index = pd.Index(TT.index).astype(str).str.upper(); TT = TT[~TT.index.duplicated(keep=False)]   # indexed by UPPERCASE mouse symbol
AXIS = set(np.char.upper(np.asarray(np.load(R + 'results/signed/PANC.npz', allow_pickle=True)['genes']).astype(str)))
idx = np.flatnonzero(Uc); n = len(idx); ROW = {s: j for j, s in enumerate(g[idx])}
ANN = pd.read_csv(ANNF, sep='\t'); ANN = ANN[ANN.endpoint == 'PANC_RFX6'].assign(gene=lambda d: d.gene.str.upper()).set_index('gene')
det = ANN.spliced_det_pct.reindex(g[idx]).to_numpy(float); assert np.isfinite(det).mean() > .99, 'detection annotation coverage'
dbin = np.full(n, -1); ok_ = np.isfinite(det); dbin[ok_] = np.minimum((pd.Series(det[ok_]).rank(pct=True).to_numpy() * 10).astype(int), 9)
PCT, LO, HI, TIEC, ABOVE = {}, {}, {}, {}, {}
for m in METH:
    v = np.asarray(V[m], float)[idx]; PCT[m] = (rankdata(v) - .5) / n
    rmin, rmax = rankdata(-v, method='min'), rankdata(-v, method='max')   # positions from the top (1 = best)
    LO[m] = (n - rmax + .5) / n; HI[m] = (n - rmin + .5) / n; TIEC[m] = rmax - rmin + 1; ABOVE[m] = rmin - 1
def topk_prob(m, K):   # inclusion probability in the top K under random tie-breaking (fractional allocation)
    return np.clip((K - ABOVE[m]) / TIEC[m], 0, 1)
TIED = {m: TIEC[m] >= max(2, int(.01 * n)) for m in METH}
# ---------- 3. eligibility, attrition, SI tables
def eligibility(M):
    M = M.dropna(subset=['mouse_symbol']).copy(); M['in_axis'] = M.mouse_symbol.isin(AXIS); M['in_UOLD'] = M.mouse_symbol.isin(ROW)
    M['KO_tested'] = [bool(TT.testable.get(s, False)) if 'testable' in TT else np.isfinite(TT.padj.get(s, np.nan)) for s in M.mouse_symbol]
    M['KO_responsive'] = [bool(y0[idx][ROW[s]]) if s in ROW else bool(TT.resp.get(s, False)) for s in M.mouse_symbol]
    return M
ATT, ELIG = {}, {}
for k, M0 in MS.items():
    M = eligibility(M0); u = M0.drop_duplicates('published_symbol')
    uo = M[M.in_UOLD].drop_duplicates('mouse_symbol')
    ATT[k] = dict(published_unique=len(u), mapped_published=int(M0.dropna(subset=['mouse_symbol']).published_symbol.nunique()), mouse_genes=int(M.mouse_symbol.nunique()),
                  route_counts=M0.drop_duplicates(['published_symbol', 'route']).route.value_counts().to_dict(), in_PANC_axis=int(M[M.in_axis].mouse_symbol.nunique()),
                  scoreable_UOLD=len(uo), KO_tested=int(uo.KO_tested.sum()), KO_responsive=int(uo.KO_responsive.sum()),
                  KO_up=int(sum(e[idx][ROW[s]] > 0 for s in uo[uo.KO_responsive].mouse_symbol)), KO_down=int(sum(e[idx][ROW[s]] < 0 for s in uo[uo.KO_responsive].mouse_symbol)))
    lost = M[M.in_axis & ~M.in_UOLD].drop_duplicates('mouse_symbol')[['mouse_symbol', 'published_symbol', 'subtype', 'route', 'KO_tested', 'KO_responsive']]
    lost.to_csv(OUT + f'SI_axis_not_UOLD_{k}.tsv', sep='\t', index=False)
    agg = M[M.in_UOLD].groupby('mouse_symbol').agg(subtype=('subtype', lambda s: ';'.join(sorted(set(s)))), published_symbol=('published_symbol', lambda s: ';'.join(sorted(set(s)))),
                                                    route=('route', lambda s: ';'.join(sorted(set(s)))), source_terms=('source_term', lambda s: '; '.join(sorted(set(s)))))
    G = agg.reset_index().rename(columns={'mouse_symbol': 'gene'}); j = np.array([ROW[s] for s in G.gene]); G['row'] = j
    G['responder'] = y0[idx][j]; G['direction'] = np.where(~G.responder, 'NON', np.where(e[idx][j] > 0, 'KOup', 'KOdown'))
    G['KO_log2FC_human'] = [TT.lfc.get(s, np.nan) for s in G.gene]; G['KO_padj_human'] = [TT.padj.get(s, np.nan) for s in G.gene]; G['spliced_det_pct'] = det[j]
    for m in METH: G[f'pct_{m}'] = PCT[m][j]; G[f'tied_{m}'] = TIED[m][j]
    G['GB_rank'] = (n - rankdata(np.asarray(V['GeoBRIDGE'], float)[idx], method='max') + 1).astype(int)[j]
    G['n_comp_above_GB'] = (G[[f'pct_{m}' for m in EXT]].values > G.pct_GeoBRIDGE.values[:, None]).sum(1)
    ELIG[k] = G; G.drop(columns='row').sort_values(['responder', 'pct_GeoBRIDGE'], ascending=False).to_csv(OUT + f'SI_table_{k}_allmethods.tsv', sep='\t', index=False)
# ---------- 4. whole-set metrics
gix = np.asarray(X['gi'])[idx]; Cm = np.asarray(X['C']); NB = Cm.shape[0]
_mp, _ = PT._module_paths(X['dsys'], PT.PRIMARY_SIZE); MODRC = PT.load_modules(X['dsys'], [])[1]   # frozen G/50 control-only modules (rule + file sha checked by load_modules)
_mm = pd.read_csv(_mp, sep='	', dtype=str); _mm = dict(zip(_mm.gene.str.upper(), _mm.module))   # map symbols are mouse-case; score axis is upper-case
MODLAB = np.array([_mm.get(s, f'S:{s}') for s in g[idx]]); MODRC['n_marker_axis_missing'] = int(sum(s not in _mm for s in g[idx]))
allresp = y0[idx].astype(bool)
def wmed(x, wt):
    m_ = wt > 0; x, wt = x[m_], wt[m_]
    if len(x) == 0: return np.nan
    o = np.argsort(x, kind='mergesort'); x, c = x[o], np.cumsum(wt[o]); h = c[-1] / 2; i = int(np.searchsorted(c, h - 1e-12 * c[-1]))
    return float((x[i] + x[i + 1]) / 2) if (abs(c[i] - h) <= 1e-12 * c[-1] and i + 1 < len(x)) else float(x[i])
def metrics(w, rows, resp, m):
    p = PCT[m][rows]; ww = w[rows]; out = {'Q1_median_pct': wmed(p, ww), 'Q1_bound_lo': wmed(LO[m][rows], ww), 'Q1_bound_hi': wmed(HI[m][rows], ww)}
    nonm = np.ones(n, bool); nonm[rows] = False; wb = w.copy(); wb[~nonm] = 0
    mk = np.bincount(dbin[rows][dbin[rows] >= 0], weights=ww[dbin[rows] >= 0], minlength=10); bg = np.bincount(dbin[nonm & (dbin >= 0)], weights=wb[nonm & (dbin >= 0)], minlength=10)
    rw = np.zeros(n); okb = (dbin >= 0) & nonm; rw[okb] = wb[okb] * np.where(bg[dbin[okb]] > 0, mk[dbin[okb]] / np.maximum(bg[dbin[okb]], 1e-12), 0)
    out['Q1m_det_matched_bg_median'] = wmed(PCT[m], rw); out['Q1m_minus_matched_bg'] = out['Q1_median_pct'] - out['Q1m_det_matched_bg_median']
    pr, wr = p[resp], ww[resp]; out['Q2_median_pct_KOresp'] = wmed(pr, wr)
    rr = rows[resp]; bgr = np.flatnonzero(allresp & nonm)
    for f in (.05, .10):
        K = int(np.floor(f * n)); P_ = topk_prob(m, K); tag = int(round(f * 100))
        out[f'Q2_recall_top{tag}'] = float((wr * P_[rr]).sum() / wr.sum()) if wr.sum() > 0 else np.nan
        if f == .10:
            out['Q2_expected_K_over_N'] = K / n; out['Q2_bg_recall_top10_nonmarker_resp'] = float((w[bgr] * P_[bgr]).sum() / w[bgr].sum())
            out['Q2_enrich_vs_K_over_N'] = out['Q2_recall_top10'] / (K / n)
            out['Q2_enrich_vs_nonmarker_resp'] = out['Q2_recall_top10'] / out['Q2_bg_recall_top10_nonmarker_resp'] if out['Q2_bg_recall_top10_nonmarker_resp'] > 0 else np.nan
    pn, wn = p[~resp], ww[~resp]
    if wr.sum() > 0 and wn.sum() > 0:
        gt = (pr[:, None] > pn[None, :]) + .5 * (pr[:, None] == pn[None, :]); out['Q3_AUROC_within_markers'] = float((wr[:, None] * wn[None, :] * gt).sum() / (wr.sum() * wn.sum()))
    else: out['Q3_AUROC_within_markers'] = np.nan
    return out
FAM = ['Q1_median_pct', 'Q2_median_pct_KOresp', 'Q2_recall_top10', 'Q3_AUROC_within_markers']
def holm(p):
    p = np.asarray(p, float); m_ = len(p); q = np.where(np.isfinite(p), p, 1.0); o = np.argsort(q, kind='stable'); out = np.full(m_, np.nan); run = 0.0
    for i, k_ in enumerate(o): run = max(run, (m_ - i) * q[k_]); out[k_] = min(1.0, run)
    out[~np.isfinite(p)] = np.nan; return out
RES, DIAG, DRAWS = [], [], {}
for k, G in ELIG.items():
    rows = G.row.to_numpy(); resp = G.responder.to_numpy(bool); DR = {}
    for m in METH:
        pt = metrics(np.ones(n), rows, resp, m); bs = [metrics(Cm[b][gix].astype(float), rows, resp, m) for b in range(NB)]
        for q in pt: DR[(m, q)] = np.r_[pt[q], [d[q] for d in bs]]
    for (m, q), v in DR.items():
        DRAWS[f'{k}|{m}|{q}'] = v; f_ = v[1:][np.isfinite(v[1:])]; ok = len(f_) >= .95 * NB
        row = dict(markerset=k, method=m, metric=q, value=v[0], lo=np.percentile(f_, 2.5) if ok else np.nan, hi=np.percentile(f_, 97.5) if ok else np.nan, n_valid_draws=len(f_),
                   n_markers=len(rows), n_resp=int(resp.sum()), n_markers_in_tied_block=int(TIED[m][rows].sum()), tied_fraction_UOLD=float(TIED[m].mean()))
        if m != 'GeoBRIDGE':
            d_ = DR[('GeoBRIDGE', q)] - v; DRAWS[f'{k}|GB-{m}|{q}'] = d_; fd = d_[1:][np.isfinite(d_[1:])]; okd = len(fd) >= .95 * NB
            row.update(GB_minus=d_[0], GB_minus_lo=np.percentile(fd, 2.5) if okd else np.nan, GB_minus_hi=np.percentile(fd, 97.5) if okd else np.nan, GB_minus_p=PT.boot_p(fd) if okd else np.nan)
        RES.append(row)
    # module diagnostics + leave-one-module-out
    mods_r, mods_n = MODLAB[rows[resp]], MODLAB[rows[~resp]]; cnt = pd.Series(MODLAB[rows]).value_counts()   # actual control-defined module labels (internal review 2026-10-29: gi is a gene index)
    dg = dict(markerset=k, n_markers=len(rows), modules_markers=int(cnt.size), modules_responders=int(len(set(mods_r))), modules_nonresponders=int(len(set(mods_n))),
              max_module_share_markers=float(cnt.iloc[0] / len(rows)), max_module_share_responders=float(pd.Series(mods_r).value_counts().iloc[0] / max(len(mods_r), 1)) if len(mods_r) else np.nan)
    for m in EXT:
        for q in ('Q1_median_pct', 'Q2_median_pct_KOresp'):
            vals = []
            for md in cnt.index:
                keep = MODLAB[rows] != md
                if keep.sum() < 3 or (q.startswith('Q2') and (resp & keep).sum() < 2): continue
                a = metrics(np.ones(n), rows[keep], resp[keep], 'GeoBRIDGE')[q]; b = metrics(np.ones(n), rows[keep], resp[keep], m)[q]; vals.append(a - b)
            dg[f'LOMO_{q}_GB-{m}_min'] = float(np.min(vals)) if vals else np.nan; dg[f'LOMO_{q}_GB-{m}_max'] = float(np.max(vals)) if vals else np.nan
    DIAG.append(dg)
T = pd.DataFrame(RES); T['p_holm8'] = np.nan
for (k, q), sub in T[(T.method != 'GeoBRIDGE') & T.metric.isin(FAM)].groupby(['markerset', 'metric']):
    assert len(sub) == 8, ('Holm family size', k, q, len(sub)); T.loc[sub.index, 'p_holm8'] = holm(sub.GB_minus_p.values)
T.to_csv(OUT + 'table1_wholeset_metrics.tsv', sep='\t', index=False); pd.DataFrame(DIAG).to_csv(OUT + 'table1_module_diagnostics.tsv', sep='\t', index=False)
pd.DataFrame(ATT).T.to_csv(OUT + 'table1_attrition.tsv', sep='\t'); np.savez_compressed(OUT + 'table1_draws.npz', **DRAWS)
# ---------- 5. main table: all eligible KO-responsive PRIMARY markers
G = ELIG['PRIMARY_CellMarker2024']; MT_ = G[G.responder].assign(abslfc=lambda d: d.KO_log2FC_human.abs()).sort_values('abslfc', ascending=False)
cols = ['gene', 'published_symbol', 'route', 'subtype', 'direction', 'KO_log2FC_human', 'KO_padj_human', 'GB_rank'] + [f'pct_{m}' for m in METH] + [f'tied_{m}' for m in ('GRNBoost2', 'RegDiffusion')] + ['n_comp_above_GB']
MT_[cols].to_csv(OUT + 'table1_main_rows.tsv', sep='\t', index=False)
shaf = lambda p: hashlib.sha256(open(p, 'rb').read()).hexdigest()
json.dump(dict(B=int(B), n_UOLD=n, attrition=ATT, tied_fraction_UOLD={m: float(TIED[m].mean()) for m in METH}, rebuild_parity=REC['parity'], rebuild_inputs=REC['inputs'],
               rebuild_deps=REC.get('deps'), draw_fingerprints=REC.get('draw_fingerprints'), rebuild_replay=REC.get('replay'), code_sha256=shaf(os.path.abspath(__file__)),
               rebuild_code_sha256=shaf(R + 'scratch/gb_fig3d_rebuild_2026_10_29.py'), truth_code_sha256=shaf(R + 'scratch/gb_truth_tables_2026_10_28.py'),
               inputs={p: shaf(p) for p in (GS_ + 'CellMarker_2024.txt', GS_ + 'PanglaoDB_Augmented_2021.txt', HOM, ANNF, R + 'results/signed/PANC.npz')},
               draws_sha256=shaf(OUT + 'table1_draws.npz'), module_receipt={k: MODRC.get(k) for k in ('rule_sha256', 'module_file_sha256', 'n_modules')}), open(OUT + 'table1_receipt.json', 'w'), indent=1, default=str)
pd.set_option('display.width', 260); pd.set_option('display.max_rows', 400); pd.set_option('display.max_columns', 40)
print(pd.DataFrame(ATT).T.to_string()); print(pd.DataFrame(DIAG).iloc[:, :7].to_string(index=False))
for ms_ in MS: print(ms_); print(T[T.markerset == ms_].pivot_table(index='method', columns='metric', values='value').round(3).to_string())
print(T[(T.method != 'GeoBRIDGE') & T.metric.isin(FAM)][['markerset', 'metric', 'method', 'GB_minus', 'GB_minus_lo', 'GB_minus_hi', 'GB_minus_p', 'p_holm8']].round(3).to_string(index=False))
print(MT_[['gene', 'published_symbol', 'route', 'subtype', 'direction', 'KO_log2FC_human', 'GB_rank'] + [f'pct_{m}' for m in METH]].round(3).to_string(index=False))
