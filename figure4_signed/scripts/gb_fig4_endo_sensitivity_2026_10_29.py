"""Fig 4 final sensitivity (internal directive 2026-10-29 'Finalize Figure 4' §5): is the NKX2-5 KO-up binding-supported signal explained by endothelial vs
cardiomyocyte expression context? TRUTH-FREE specificity from the observational HEART fixture; truth used only for the evaluation classes.
Marker sets FIXED here from literature before any result (no KO/ChIP gene is used to define cells):
  CM = Tnnt2, Myl7, Myh6, Ttn ; EC = Pecam1, Cdh5, Kdr, Emcn, Tie1.
Cell scores: mean over markers of z-scored log1p(CP10k) expression. EC-like cells = top 3% EC score; CM-like = top 3% CM score (disjoint; equal size).
Gene specificity s_g = log2((mean CP10k in EC-like + 0.01) / (mean CP10k in CM-like + 0.01)).
Evaluation (NKX2-5 and, for comparison, TBX5): KO-up responders L5 (bound) vs L4 (unbound): (i) oriented signed score -T, (ii) GB advantage over the 9
comparators (fig4_gene_advantage v2, abs-magnitude fixed). Reported: s_g distribution per class, common support (overlapping s_g range, trimmed to the
5-95% of the pooled L4/L5 s_g), difference L5-L4 within specificity tertiles (weights n5*n4) with 50-module bootstrap 95% CI, and restricted to
non-EC-specific genes (s_g <= 0). Insufficient support (<10 L5 genes retained) -> NO-CALL.
usage (server, repo root): script ADV_GENE_TSV RUN_DIR OUT [--B 2000]"""
import sys, os, json, hashlib
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); sys.path[:0] = [R + 'scratch', R + 'pairspace']
import numpy as np, pandas as pd, anndata as ad, scipy.sparse as sp, gb_substrate as GS
a = sys.argv; ADV, RUN, OUT = a[1], a[2].rstrip('/') + '/', a[3].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True); B = int(a[a.index('--B') + 1]) if '--B' in a else 2000
CM, EC = ['Tnnt2', 'Myl7', 'Myh6', 'Ttn'], ['Pecam1', 'Cdh5', 'Kdr', 'Emcn', 'Tie1']
fx, _, _ = GS.paths('CARDIACV', canonical=True); F = ad.read_h5ad(fx); X = F.X.tocsr() if sp.issparse(F.X) else sp.csr_matrix(F.X)
lib = np.asarray(X.sum(1)).ravel(); Xn = sp.diags(1e4 / np.maximum(lib, 1)) @ X; genes = np.asarray(F.var_names).astype(str); gi = {g: i for i, g in enumerate(genes)}
miss = [g for g in CM + EC if g not in gi]; print('markers missing from fixture:', miss, flush=True)
def score(ms):
    cols = [gi[g] for g in ms if g in gi]; L = np.log1p(Xn[:, cols].toarray()); return ((L - L.mean(0)) / (L.std(0) + 1e-9)).mean(1), len(cols)
sc_cm, ncm = score(CM); sc_ec, nec = score(EC); n = len(lib); k = int(.03 * n)
ec = np.zeros(n, bool); ec[np.argsort(-sc_ec)[:k]] = True; cm = np.zeros(n, bool); cm[np.argsort(-sc_cm)[:k]] = True; both = ec & cm; ec &= ~both; cm &= ~both
mec = np.asarray(Xn[ec].mean(0)).ravel(); mcm = np.asarray(Xn[cm].mean(0)).ravel(); s = np.log2((mec + .01) / (mcm + .01)); SPEC = dict(zip(np.char.upper(genes), s))
pd.DataFrame(dict(gene=genes, s_EC_vs_CM=s, mean_EC=mec, mean_CM=mcm)).to_csv(OUT + 'heart_gene_specificity.tsv', sep='\t', index=False)
AD = pd.read_csv(ADV, sep='\t'); ROWS, CNT = [], []
for tf, u in (('NKX2-5', 'CARDIACV_NKX2-5'), ('TBX5', 'CARDIACV_TBX5')):
    D = pd.read_csv(RUN + f'genes_{u}.tsv', sep='\t'); D['G'] = D.gene.str.upper(); D['s'] = D.G.map(SPEC)
    A = AD[AD.TF == tf][['gene', 'adv']]; A['G'] = A.gene.str.upper(); D = D.merge(A[['G', 'adv']], on='G', how='left', validate='1:1')
    mods = D.module.to_numpy(); um, inv = np.unique(mods, return_inverse=True); rng = np.random.default_rng([20261029, 51, len(um)])
    W = np.vstack([np.ones(len(D))] + [np.bincount(rng.integers(0, len(um), len(um)), minlength=len(um))[inv].astype(float) for _ in range(B)])
    p5 = (D.level == 'L5').to_numpy() & np.isfinite(D.s.to_numpy()); p4 = (D.level == 'L4').to_numpy() & np.isfinite(D.s.to_numpy())
    for cls, m in (('L5 bound', p5), ('L4 unbound', p4)):
        q = D.s[m]; CNT.append(dict(TF=tf, cls=cls, n=int(m.sum()), s_median=q.median(), s_q10=q.quantile(.1), s_q90=q.quantile(.9), frac_EC_specific=float((q > 1).mean()),
                                    frac_CM_specific=float((q < -1).mean())))
    lo, hi = np.nanquantile(D.s[p5 | p4], [.05, .95]); sup = (D.s >= lo).to_numpy() & (D.s <= hi).to_numpy()
    for vname, v in (('oriented T (-T)', -D.T_prod.to_numpy(float)), ('GB advantage over 9 comparators', D.adv.to_numpy(float))):
        for scope, keep in (('all', np.ones(len(D), bool)), ('common support (5-95%)', sup), ('non-EC-specific (s<=0)', (D.s <= 0).to_numpy())):
            a5, a4 = p5 & keep & np.isfinite(v), p4 & keep & np.isfinite(v)
            if a5.sum() < 10: ROWS.append(dict(TF=tf, metric=vname, scope=scope, n_L5=int(a5.sum()), n_L4=int(a4.sum()), verdict='NO-CALL (<10 L5)')); continue
            t = np.digitize(D.s.to_numpy(), np.nanquantile(D.s[a5 | a4], [1 / 3, 2 / 3]))
            def est(w):
                num = den = 0.
                for kk in range(3):
                    x5, x4 = a5 & (t == kk), a4 & (t == kk)
                    if w[x5].sum() > 0 and w[x4].sum() > 0:
                        ww = x5.sum() * x4.sum(); num += ww * ((w[x5] * v[x5]).sum() / w[x5].sum() - (w[x4] * v[x4]).sum() / w[x4].sum()); den += ww
                return num / den if den else np.nan
            bs = np.array([est(w) for w in W]); f = bs[1:][np.isfinite(bs[1:])]
            raw = v[a5].mean() - v[a4].mean()
            ROWS.append(dict(TF=tf, metric=vname, scope=scope, n_L5=int(a5.sum()), n_L4=int(a4.sum()), mods_L5=int(np.unique(mods[a5]).size), raw_diff=raw, strat_diff=bs[0],
                             lo=np.percentile(f, 2.5) if len(f) > .95 * B else np.nan, hi=np.percentile(f, 97.5) if len(f) > .95 * B else np.nan, valid=len(f), verdict=''))
T = pd.DataFrame(ROWS); T.to_csv(OUT + 'endo_sensitivity.tsv', sep='\t', index=False); C = pd.DataFrame(CNT); C.to_csv(OUT + 'endo_specificity_by_class.tsv', sep='\t', index=False)
json.dump(dict(B=B, markers=dict(CM=CM, EC=EC), n_markers_found=dict(CM=ncm, EC=nec), missing=miss, n_EC_like=int(ec.sum()), n_CM_like=int(cm.sum()),
               code_sha256=hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()), open(OUT + 'receipt.json', 'w'), indent=1)
pd.set_option('display.width', 250); print(C.round(3).to_string(index=False)); print(T.round(3).to_string(index=False))
