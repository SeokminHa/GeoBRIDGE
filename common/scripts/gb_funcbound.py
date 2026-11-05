"""K562 FUNCTIONAL-BOUND + MANDATORY MATCHED-SOURCE CONTROL.
Implements the frozen peak-to-gene rule appended to SIMULATOR_2x2_DESIGN_FROZEN_2026_09_03.md.
No window scanning. Rule fixed before any performance value is read."""
import numpy as np, pandas as pd, json, sys, glob, os, gzip
import anndata as ad, h5py
from scipy.sparse import issparse
from sklearn.metrics import roc_auc_score
ntc, pairs_npz, scoredir, truthdir, chipdir, scperturb, out_prefix = sys.argv[1:8]
dense = lambda M: np.asarray(M.todense()) if issparse(M) else np.asarray(M)
A = ad.read_h5ad(ntc); genes = np.asarray(A.var_names).astype(str)
gpos = {g: i for i, g in enumerate(genes)}
Xn = dense(A.X).astype(np.float64)
f = np.load(pairs_npz, allow_pickle=False); I, J = f['dc1_min_I'].astype(np.int64), f['dc1_min_J'].astype(np.int64)

# gene coordinates from scPerturb var (GRCh38)
h = h5py.File(scperturb, 'r')
def vcol(n):
    g = h['var'][n]
    if isinstance(g, h5py.Group) and 'categories' in g:
        c = np.array(g['categories']).astype(str); return c[np.array(g['codes'])]
    return np.array(g)
gn = vcol('gene_name').astype(str); chrom = vcol('chr').astype(str)
start = np.array(vcol('start')).astype(float); end = np.array(vcol('end')).astype(float)
strand = vcol('strand').astype(str)
tss = {}
for k in range(len(gn)):
    if gn[k] in gpos and np.isfinite(start[k]) and np.isfinite(end[k]):
        c = chrom[k] if str(chrom[k]).startswith('chr') else 'chr'+str(chrom[k])
        tss[gn[k]] = (c, float(start[k]) if str(strand[k]) in ('+', '1', 'plus') else float(end[k]))
print('genes with TSS:', len(tss), flush=True)

WIN = 2000
def bound_set(tfname):
    hits = glob.glob(os.path.join(chipdir, f'{tfname}_*.bed.gz'))
    if not hits: return None
    peaks = {}
    with gzip.open(hits[0], 'rt') as fh:
        for line in fh:
            p = line.split('\t')
            if len(p) < 3: continue
            peaks.setdefault(p[0], []).append((int(p[1]), int(p[2])))
    for c in peaks: peaks[c].sort()
    out = set()
    for g, (c, t) in tss.items():
        arr = peaks.get(c)
        if not arr: continue
        lo, hi = t-WIN, t+WIN
        for s, e in arr:
            if e >= lo and s <= hi: out.add(g); break
    return out

def load(npz, key):
    z = np.load(npz, allow_pickle=True); return z[key].astype(np.float64)
ARM = {'M0_U': load(f'{scoredir}/repl_m0_S_U_dc1_min_scores.npz', 'dc1_min|R0_none|k=0'),
       'M0_V': load(f'{scoredir}/repl_m0_S_V_dc1_min_scores.npz', 'dc1_min|R0_none|k=0')}
def zc(M):
    M = M-M.mean(0, keepdims=True); return M/np.maximum(M.std(0, keepdims=True), 1e-12)
Z = zc(Xn); ARM['expr_corr'] = (Z.T@Z)/Xn.shape[0]
Mid = 0.5*(Xn[I]+Xn[J]); Zm = zc(Mid); ARM['local_corr'] = (Zm.T@Zm)/Mid.shape[0]
Y = dense(A.layers['Mu']).astype(np.float64); dX = Xn[J]-Xn[I]; dY = Y[J]-Y[I]
U_, S_, Vt = np.linalg.svd(dY, full_matrices=False)
def sc(d):
    N = dX.T@d; D = np.abs(dX).T@np.abs(d); return N/(D+1e-12)
for k in (16, 48):
    Sk = S_.copy(); Sk[:k] = 0.0; ARM[f'M_LEGACY_k{k}'] = sc((U_*Sk)@Vt)

sup = pd.read_csv((__import__("os").environ["GEOBRIDGE_ROOT"] + '/validation/results/replogle_k562_prep_full_2026_09_03/ntc_per_gene_support.tsv'), sep='\t')
regpool = [g for g in sup.loc[sup['is_lambert_tf'].fillna(False) & sup['clears_support_20'].fillna(False), 'gene'].astype(str) if g in gpos]
rows = []
for tp in sorted(glob.glob(f'{truthdir}/TRUTHv3_*.npz')):
    tf = os.path.basename(tp).replace('TRUTHv3_', '').replace('.npz', '')
    if tf in ('gate', 'summary') or tf not in gpos: continue
    bs = bound_set(tf)
    if bs is None: continue
    t = np.load(tp, allow_pickle=True); tg = t['genes'].astype(str); sig = t['sig']
    resp = {g: bool(s) for g, s in zip(tg, sig)}
    Aset = [g for g in bs if g in gpos and g != tf and resp.get(g, False)]
    Bset = [g for g in bs if g in gpos and g != tf and (g in resp) and not resp[g]]
    rec = {'TF': tf, 'n_bound': len(bs), 'nA': len(Aset), 'nB': len(Bset)}
    if len(Aset) < 5 or len(Bset) < 5:
        rec['status'] = 'FUNCTIONAL_BOUND_NO_CALL'; rows.append(rec); print(tf, rec, flush=True); continue
    rec['status'] = 'OK'
    idx = np.array([gpos[g] for g in Aset+Bset]); lab = np.array([1]*len(Aset)+[0]*len(Bset))
    for arm, S in ARM.items():
        v = np.abs(S[gpos[tf]][idx])
        if not np.isfinite(v).all(): continue
        T_r = roc_auc_score(lab, v); rec[f'{arm}__T_focal'] = float(T_r)
        # MANDATORY SOURCE CONTROL: same A/B sets, alternative regulators
        Tp = []
        for p in regpool:
            if p == tf: continue
            vp = np.abs(S[gpos[p]][idx])
            if np.isfinite(vp).all(): Tp.append(roc_auc_score(lab, vp))
        Tp = np.array(Tp)
        rec[f'{arm}__T_null_median'] = float(np.median(Tp))
        rec[f'{arm}__focal_percentile'] = float((Tp < T_r).mean())
        rec[f'{arm}__emp_p'] = float(((Tp >= T_r).sum()+1)/(len(Tp)+1))
    rows.append(rec); print(tf, 'nA=%d nB=%d M0_U T=%.3f pct=%.3f' % (len(Aset), len(Bset),
          rec.get('M0_U__T_focal', float('nan')), rec.get('M0_U__focal_percentile', float('nan'))), flush=True)
df = pd.DataFrame(rows); df.to_csv(f'{out_prefix}_per_tf.tsv', sep='\t', index=False)
ok = df[df['status'] == 'OK']
summ = {'n_tf_attempted': int(len(df)), 'n_tf_called': int(len(ok))}
for arm in ARM:
    for m in ('T_focal', 'T_null_median', 'focal_percentile', 'emp_p'):
        c = f'{arm}__{m}'
        if c in ok.columns and ok[c].notna().any(): summ[c] = float(ok[c].median())
json.dump(summ, open(f'{out_prefix}_summary.json', 'w'), indent=1); print(json.dumps(summ, indent=1))
