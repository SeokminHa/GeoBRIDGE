"""Fig. S2 label-permutation reference line: mean AUPRC when the distance labels are permuted within each regulator row
(B = 200 per dataset, 48 bifurcation datasets, all operator x channel arms with depth window 0..8).
Reads the score matrices written by gb_fig2_ablation_2026_10_27.py; writes one row to OUT (default supplementary/data/figS2_permutation_null.tsv).
usage: python gb_si_figS2_permutation_null.py [OUT]   (needs GEOBRIDGE_ROOT)"""
import sys, os, numpy as np, pandas as pd
from sklearn.metrics import average_precision_score as APs
R = os.environ["GEOBRIDGE_ROOT"] + '/'; CC = ['C0', 'C1', 'C2', 'C3']
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'figS2_permutation_null.tsv')
A = R + 'results/sergio_tf32rep_2026_10_22/ablation_2026_10_27/'
rng = np.random.default_rng(20261027); PR = []
for g in range(501, 507):
    for c in CC:
        for s_ in [1, 2]:
            z = np.load(f'{A}full/scores_g{g}_bifurcation_{c}_s{s_}.npz'); Dm = z['Dm']
            for b in range(200):
                Dp = np.vstack([r[rng.permutation(len(r))] for r in Dm]); Y = (Dp == 1).ravel()
                for o in ['R1', 'R2h', 'R2s']:
                    for ch in ['U', 'V', 'X']: PR.append(APs(Y, z[f'{o}|{ch}'].ravel()))
perm = float(np.mean(PR))
pd.DataFrame([dict(metric='AP', operator='label_permutation', channel='all', mean=perm, n_draws=len(PR))]).to_csv(OUT, sep='\t', index=False)
print('label permutation AUPRC', perm)
