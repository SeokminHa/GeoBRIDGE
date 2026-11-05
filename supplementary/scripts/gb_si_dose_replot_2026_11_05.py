"""Replot S_dose from SUMMARY.tsv (plot-only): display names, GeoBRIDGE black, k=0 black open dashed. usage: script DOSE_DIR OUT_PNG"""
import sys, pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
S = pd.read_csv(sys.argv[1] + "/SUMMARY.tsv", sep="\t")
NM = dict(GB="GeoBRIDGE", GB_k0="GeoBRIDGE (k = 0)", VEL="Velorama", fasttenet="FastTENET", genie3="GENIE3", grnboost2="GRNBoost2",
          pidc="PIDC", regdiffusion="RegDiffusion", deepsem="DeepSEM", pearson="Pearson |r|")
CO = dict(VEL="#2ca02c", fasttenet="#17becf", genie3="#ff7f0e", grnboost2="#8c564b", pidc="#9467bd", regdiffusion="#e377c2", deepsem="#bcbd22", pearson="#7f7f7f")
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
f, ax = plt.subplots(2, 2, figsize=(7.2, 5.6), sharex=True)
for i, t in enumerate(["bifurcation", "tree"]):
    for j, (k, lab) in enumerate((("AP", "AUPRC"), ("DvI23", "Direct-vs-indirect AUROC"))):
        a = ax[i, j]
        for m in ["VEL", "genie3", "grnboost2", "pidc", "regdiffusion", "deepsem", "fasttenet", "pearson", "GB_k0", "GB"]:
            d = S[(S.topo == t) & (S.method == m)].sort_values("amp")
            if d.empty: continue
            if m == "GB": a.errorbar(d.amp, d[k], yerr=d[k + "_ci95_half"], color="k", lw=2, marker="o", ms=4, capsize=2, label=NM[m], zorder=5)
            elif m == "GB_k0": a.plot(d.amp, d[k], color="k", ls="--", marker="o", mfc="white", ms=4, lw=1, label=NM[m], zorder=4)
            else: a.plot(d.amp, d[k], color=CO[m], marker="o", ms=2.5, lw=.9, label=NM[m])
        a.axhline(.5 if k == "DvI23" else 240 / 3168, color="grey", ls=":", lw=.7)
        a.set_title(t.capitalize(), fontsize=8); a.set_ylabel(lab); a.set_xticks([0, .25, .5, 1.0])
for a in ax[1]: a.set_xlabel("Shared-effect amplitude $a$ ($a_A=a_B$)")
h, l = ax[0, 0].get_legend_handles_labels(); f.legend(h[::-1], l[::-1], loc="lower center", ncol=5, frameon=False, fontsize=7)
f.tight_layout(rect=(0, .08, 1, 1)); f.savefig(sys.argv[2], dpi=300)
