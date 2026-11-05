"""Fig 2 bar figure v2 (2026-10-27), readability revision of gb_fig2_draft_bars_2026_10_23.py. Plotting only; same TABLE.tsv, same statistics.
PI feedback + internal review (internal design note (not shipped)): Helvetica-metric font (Nimbus Sans); y labels AUPRC / AUROC / Direct vs indirect AUROC;
methods ordered velocity-derived first (GB, Velorama, FastTENET) then no-velocity, with a gap and legend sub-headers; "Pearson |r|" label;
AUPRC y-max 0.5. Two variants per topology: MAIN = no per-condition stars (internal review Q4b: pooled GRN-level sign-flip is the prespecified test);
STARS = per-condition paired t over 6 GRNs, Holm-8, post hoc, one bracket height per panel, marks aligned on the tick x position."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats
from statsmodels.stats.multitest import multipletests
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/results/sergio_tf32rep_2026_10_22")
OUT = f"{R}/fig2_final"
import os; MODE = os.environ.get("FIG2_MODE", "main")   # main = Fig 2 / SI tree; dviap = SI direct-vs-indirect AUPRC (fig2_final/DVI_AP_TABLE.tsv)
T = pd.read_csv(f"{R}/TABLE.tsv" if MODE == "main" else f"{OUT}/DVI_AP_TABLE.tsv", sep="\t")
TAG = "" if MODE == "main" else "_DVIAP"
COND = [("C0", "No shared\ninput"), ("C1", "Temporal\ninput"), ("C2", "Branch-specific\ninput"), ("C3", "Dual\ninput")]
VELO = [("GB", "GeoBRIDGE (ours)", "#2166ac"), ("VEL", "Velorama", "#2ca02c"), ("fasttenet", "FastTENET (velocity pseudotime)", "#17becf")]
NOVELO = [("genie3", "GENIE3", "#ff7f0e"), ("grnboost2", "GRNBoost2", "#8c564b"), ("pidc", "PIDC", "#9467bd"), ("regdiffusion", "RegDiffusion", "#e377c2"),
          ("deepsem", "DeepSEM", "#bcbd22"), ("pearson", "Pearson |r|", "#7f7f7f")]
M = VELO + NOVELO
MET = [("AP", "AUPRC", .5), ("AUROC", "AUROC", 1.0), ("DvI2", "Direct vs indirect AUROC", 1.0)] if MODE == "main" else [("DvI2_AP", "Direct vs 2-hop AUPRC", 1.0), ("DvI3_AP", "Direct vs 3-hop AUPRC", 1.0)]
RAND = {"AP": 240 / 3168, "AUROC": .5, "DvI2": .5}
G = T.groupby(["grn", "topo", "cond", "method"])[[k for k, *_ in MET] + ([f"{k}_prev".replace("_AP_prev", "_prev") for k, *_ in MET] if MODE != "main" else [])].mean().reset_index()
def stars(p): return "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "ns"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"], "pdf.fonttype": 42,
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
nm = len(M); w = .105; vgap = 0.0; gap = .32
XOFF = np.array([i * w + (vgap if i >= len(VELO) else 0) for i in range(nm)]); span = XOFF[-1] + w
base = np.arange(4) * (span + gap)
ROWS = []
for topo in ["bifurcation", "tree"]:
    for variant in ["main", "stars"]:
        fig, axs = plt.subplots(1, len(MET), figsize=(7 * len(MET), 5.6), gridspec_kw=dict(wspace=.17))
        for j, (k, ylab, ymax) in enumerate(MET):
            ax = axs[j]; cond_ph = []; tops = []
            for ci_, (c, cl) in enumerate(COND):
                S = G[(G.topo == topo) & (G.cond == c)].pivot(index="grn", columns="method", values=k)
                ps = []
                for mi, (m, lab, col) in enumerate(M):
                    x = S[m].values; mu = x.mean(); h = stats.t.ppf(.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
                    ax.bar(base[ci_] + XOFF[mi], mu, w * .9, color=col, edgecolor="none", label=lab if ci_ == 0 else None)
                    ax.errorbar(base[ci_] + XOFF[mi], mu, yerr=h, color="#222", lw=.9, capsize=2); tops.append(mu + h)
                    if m != "GB": ps.append(stats.ttest_rel(S["GB"], S[m]).pvalue)
                ph = multipletests(ps, method="holm")[1]; cond_ph.append(ph)
                if variant == "main":
                    for m_, p_, ph_ in zip([m for m, *_ in M[1:]], ps, ph): ROWS.append(dict(topo=topo, metric=k, cond=c, comparator=m_, p_t=p_, p_holm8=ph_, mark=stars(ph_)))
            if variant == "stars":
                top = max(tops) + .035 * ymax                      # one bracket height per panel (internal review Q1)
                for ci_, ph in enumerate(cond_ph):
                    x0 = base[ci_]
                    ax.plot([x0 + XOFF[0], x0 + XOFF[-1]], [top, top], color="k", lw=.8)
                    for mi, p in enumerate(ph, start=1):
                        xm = x0 + XOFF[mi]
                        ax.plot([xm, xm], [top, top - .015 * ymax], color="k", lw=.8)
                        lab_ = stars(p); txt = "\n".join(lab_) if lab_ != "ns" else "ns"   # stacked, unrotated: each '*' centred on the tick x
                        ax.text(xm, top + .006 * ymax, txt, ha="center", va="bottom", fontsize=10 if lab_ != "ns" else 6, linespacing=.55, clip_on=False)
            if MODE == "main": ax.axhline(RAND[k], ls="--", color="k", lw=1)
            else:                                                          # random AP = per-condition positive fraction (mean over GRNs)
                pv = k.replace("_AP", "_prev")
                for ci_, (c, _) in enumerate(COND):
                    rv = G[(G.topo == topo) & (G.cond == c) & (G.method == "GB")][pv].mean(); ax.plot([base[ci_] - w / 2, base[ci_] + span - w / 2], [rv, rv], ls="--", color="k", lw=1)
            ax.set_xticks(base + (XOFF[-1]) / 2); ax.set_xticklabels([cl for _, cl in COND], fontsize=12)
            ax.set_xlim(base[0] - w, base[-1] + span)
            ax.set_ylim(0, ymax); ax.tick_params(axis="y", labelsize=11)
            ax.set_ylabel(ylab, fontsize=13)
        hd, lb = axs[0].get_legend_handles_labels()
        rnd = plt.Line2D([], [], ls="--", color="k")
        fig.legend(hd + [rnd], lb + ["Random"], loc="lower center", ncol=nm + 1, frameon=False, bbox_to_anchor=(.5, -.08), fontsize=11.5, handlelength=1.4, columnspacing=1.4)
        sfx = "" if variant == "main" else "_stars"
        fig.savefig(f"{OUT}/FIG2_BARS{TAG}_{topo}_2026_10_27{sfx}.png", dpi=300, bbox_inches="tight")
        fig.savefig(f"{OUT}/FIG2_BARS{TAG}_{topo}_2026_10_27{sfx}.pdf", bbox_inches="tight")
        plt.close(fig)
pd.DataFrame(ROWS).to_csv(f"{OUT}/FIG2_BARS{TAG}_STARS_2026_10_27.tsv", sep="\t", index=False)
print("ok", matplotlib.font_manager.findfont("Helvetica"))
