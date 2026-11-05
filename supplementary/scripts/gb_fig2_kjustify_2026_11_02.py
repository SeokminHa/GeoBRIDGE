"""Is K=8 (depths 0..8) defensible from the reported SERGIO TF32REP benchmark alone? Evaluation-only on kcurve_datasets.tsv (cumulative 0..K).
(1) plateau: regret of K=8 vs the per-(topo,cond) oracle K, and the set of K within .01 of oracle;
(2) robustness: K range where mean Delta vs K=0 > 0 with 6/6 GRNs positive;
(3) nested leave-one-GRN-out: choose K on 5 GRNs (all conds+topos pooled, metric-specific), score held-out GRN; compare to fixed K=8 and K=0.
usage: script KCURVE_DIR OUT"""
import sys, os, numpy as np, pandas as pd
IN, OUT = sys.argv[1].rstrip('/') + '/', sys.argv[2].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
R = pd.read_csv(IN + 'kcurve_datasets.tsv', sep='\t'); R = R[R.kind == 'cumulative']
assert (R.groupby(['topo', 'cond', 'K']).size() == 12).all() and R[['AP', 'AUROC', 'DvI23_AUROC']].notna().all().all() and (R.coverage == 1).all()   # 6 GRN x 2 sims, complete
MET = ['AP', 'DvI23_AUROC', 'AUROC']
G = R.groupby(['topo', 'cond', 'grn', 'K'])[MET].mean().reset_index()      # sims -> GRN
rows = []
for (t, c), d in G.groupby(['topo', 'cond']):
    for m in MET:
        mk = d.groupby('K')[m].mean(); kb = int(mk.idxmax()); w = d.pivot(index='grn', columns='K', values=m); dl = w.sub(w[0], axis=0)
        okK = [K for K in range(1, 31) if (dl[K] > 0).all()]
        rows.append(dict(topo=t, cond=c, metric=m, oracleK=kb, at_oracle=mk[kb], at8=mk[8], at0=mk[0], regret8=mk[kb] - mk[8],
                         gain8_frac_of_oracle_gain=(mk[8] - mk[0]) / (mk[kb] - mk[0]),
                         K_within_01=f'{min(k for k in mk.index if mk[kb]-mk[k] <= .01)}-{max(k for k in mk.index if mk[kb]-mk[k] <= .01)}',
                         K_not_all6pos=','.join(str(K) for K in range(1, 31) if K not in okK) or 'none', n_K_all6pos=len(okK),
                         within01_contiguous=len([k for k in mk.index if mk[kb]-mk[k] <= .01]) == 1 + max(k for k in mk.index if mk[kb]-mk[k] <= .01) - min(k for k in mk.index if mk[kb]-mk[k] <= .01)))
P = pd.DataFrame(rows); P.to_csv(OUT + 'plateau.tsv', sep='\t', index=False)
cv = []
for m in MET:
    for g in sorted(G.grn.unique()):
        tr = G[G.grn != g].groupby('K')[m].mean(); ks = int(tr.idxmax()); te = G[G.grn == g].groupby('K')[m].mean()   # pooled over topo x cond
        cv.append(dict(metric=m, heldout=g, K_selected=ks, heldout_selected=te[ks], heldout_K8=te[8], heldout_K0=te[0]))
C = pd.DataFrame(cv); C.to_csv(OUT + 'logo_cv.tsv', sep='\t', index=False)
print(P.round(3).to_string(index=False)); print(C.round(4).to_string(index=False))
print(C.groupby('metric')[['heldout_selected', 'heldout_K8', 'heldout_K0']].mean().round(4))
