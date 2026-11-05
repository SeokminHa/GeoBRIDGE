"""Fig. S4: strength of each shared effect. Temporal-only (a_B = 0) and branch-only (a_A = 0) amplitude series 0, 0.25, 0.5, 1.0;
AUPRC (a, c) and direct-vs-all-indirect AUROC (b, d) for GeoBRIDGE (mean and 95% t interval over 6 GRNs), GeoBRIDGE k = 0 and 7 methods; one PNG per panel + legend.
Reads data/figS4_amplitude_summary.tsv. usage: python plot_figS4.py --out OUTDIR [--data DATADIR]"""
import os, sys, argparse
import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from PIL import Image
ap = argparse.ArgumentParser(); ap.add_argument("--data", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")); ap.add_argument("--out", required=True)
args = ap.parse_args(); O = args.out.rstrip("/") + "/"; os.makedirs(O, exist_ok=True)
S = pd.read_csv(os.path.join(args.data, "figS4_amplitude_summary.tsv"), sep="\t")
NM = dict(GB="GeoBRIDGE", GB_k0="GeoBRIDGE (k = 0)", VEL="Velorama", fasttenet="FastTENET", genie3="GENIE3", grnboost2="GRNBoost2", pidc="PIDC", regdiffusion="RegDiffusion", deepsem="DeepSEM")
CO = dict(VEL="#2ca02c", fasttenet="#17becf", genie3="#ff7f0e", grnboost2="#8c564b", pidc="#9467bd", regdiffusion="#e377c2", deepsem="#bcbd22")
ORDER = ["VEL", "genie3", "grnboost2", "pidc", "regdiffusion", "deepsem", "fasttenet", "GB_k0", "GB"]
plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
def save(f, n):
    f.savefig(O + n, dpi=300, bbox_inches="tight", facecolor="white"); plt.close(f); Image.open(O + n).convert("RGB").save(O + n); print("wrote", n)
def draw(a, sr, k):
    for m in ORDER:
        d = S[(S.series == sr) & (S.method == m)].sort_values("amp")
        if m == "GB": a.errorbar(d.amp, d[k], yerr=d[k + "_ci95_half"], color="k", lw=2, marker="o", ms=4, capsize=2, label=NM[m], zorder=5)
        elif m == "GB_k0": a.plot(d.amp, d[k], color="k", ls="--", marker="o", mfc="white", ms=4, lw=1, label=NM[m], zorder=4)
        else: a.plot(d.amp, d[k], color=CO[m], marker="o", ms=2.5, lw=.9, label=NM[m])
    a.axhline(.5 if k == "DvIall" else 240 / 3168, color="grey", ls=":", lw=.7)
for p, sr, k, lab, xl in (("a", "A", "AP", "AUPRC", "Temporal amplitude $a_A$ ($a_B=0$)"), ("b", "A", "DvIall", "Direct vs indirect AUROC", "Temporal amplitude $a_A$ ($a_B=0$)"),
                          ("c", "B", "AP", "AUPRC", "Branch amplitude $a_B$ ($a_A=0$)"), ("d", "B", "DvIall", "Direct vs indirect AUROC", "Branch amplitude $a_B$ ($a_A=0$)")):
    f, a = plt.subplots(figsize=(3.4, 2.6)); draw(a, sr, k); a.set_ylabel(lab); a.set_xlabel(xl); a.set_xticks([0, .25, .5, 1.0]); save(f, f"FigS4{p}.png")
f, a = plt.subplots(); draw(a, "A", "AP"); h, l = a.get_legend_handles_labels(); plt.close(f)
f = plt.figure(figsize=(7.0, 0.5)); f.legend(h[::-1], l[::-1], loc="center", ncol=5, frameon=False, fontsize=7); save(f, "FigS4_legend.png")
