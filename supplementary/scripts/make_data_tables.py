"""Collect the small tables in supplementary/data/ from the outputs of the compute scripts (selection and merging only, no new numbers).
usage: GEOBRIDGE_ROOT=/path/ws python make_data_tables.py [OUT_DIR]   (default OUT_DIR = supplementary/data)"""
import os, sys, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); REPO = os.path.join(HERE, '..', '..')
R = os.environ["GEOBRIDGE_ROOT"] + '/'; RF = R + 'results/signed/r2fix_2026_11_01/'; RT = R + 'results/sergio_tf32rep_2026_10_22/'
O = (sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'data')).rstrip('/') + '/'; os.makedirs(O, exist_ok=True)
rd = lambda p: pd.read_csv(p, sep='\t')
def w(df, n): df.to_csv(O + n, sep='\t', index=False); print(n, df.shape)
BIF = 'bifurcation'

# Fig. S1a: direct vs all-indirect AUPRC per dataset, 8 methods (gb_si_dviap_panels_2026_11_05.py pairs)
w(rd(RF + 'SI_panels_2026_11_05/full/dviall_ap_per_dataset.tsv'), 'figS1_dvi_auprc.tsv')
# Fig. S1b-e: GeoBRIDGE vs maximum depth K (gb_fig2_kcurve_2026_11_01.py, gb_si_recompute_dviall_2026_11_05.py kc_e1, gb_si_dviap_panels_2026_11_05.py kc)
kd = rd(RF + 'T5_kcurve/kcurve_datasets.tsv').query("kind=='cumulative' and topo==@BIF")[['grn', 'cond', 'sim', 'K', 'AP', 'AUROC']]
kd = kd.merge(rd(RF + 'SI_dviall_2026_11_05/full/kcurve_dviall_datasets.tsv')[['grn', 'cond', 'sim', 'K', 'DvIall_AUROC']], on=['grn', 'cond', 'sim', 'K'], validate='1:1')
kd = kd.merge(rd(RF + 'SI_panels_2026_11_05/full/kcurve_dviall_ap_datasets.tsv')[['grn', 'cond', 'sim', 'K', 'DvIall_AP']], on=['grn', 'cond', 'sim', 'K'], validate='1:1')
assert len(kd) == 6 * 4 * 2 * 31; w(kd, 'figS1_kcurve.tsv')
# Fig. S2: GRN-level AUPRC per operator x channel x depth window (gb_fig2_ablation_aggregate_2026_10_27.py)
GL = rd(RT + 'ablation_2026_10_27/agg/GRN_LEVEL_BY_TOPO.tsv'); w(GL[(GL.kind == 'window') & (GL.topo == BIF)], 'figS2_channel_operator.tsv')
# Fig. S3: pair window and pair cap, GeoBRIDGE depth window 0..8 (gb_e11_pairwindow_2026_11_02.py + gb_e11_summarize_2026_11_02.py)
pw = rd(RF + 'E11_pairwindow/datasets_all.tsv'); w(pw[(pw.topo == BIF) & (pw.arm == 'GB_k08')], 'figS3_pairwindow.tsv')
# Fig. S4: A-only / B-only amplitude series (gb_dose_sep_2026_11_05.py agg)
w(rd(RT + 'dose_sep_2026_11_05/SUMMARY.tsv'), 'figS4_amplitude_summary.tsv')
w(rd(RT + 'dose_sep_2026_11_05/SUMMARY_PAIRED.tsv'), 'figS4_amplitude_paired.tsv')
# Fig. S5/S6: knockout responders by direction (figure3_knockout/ tables)
F3 = os.path.join(REPO, 'figure3_knockout', 'results') + '/'; DR = ['KOup', 'KOdown']
T = rd(F3 + 'fig3_common_v2_UOLD.tsv'); w(T[T.direction.isin(DR)], 'figS5_S6_ko_direction.tsv')
K = rd(F3 + 'fig3d_rebuild/full/fig3d_rebuild_table.tsv'); w(K[(K.kind == 'level') & K.direction.isin(DR)], 'figS5_S6_k0.tsv')
