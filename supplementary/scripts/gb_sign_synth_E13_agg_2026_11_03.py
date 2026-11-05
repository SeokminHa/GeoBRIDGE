"""E13 aggregation: sims -> GRN (mean over sims, then equal-weight over topo x cond), mean +- SD and t95 CI over the 6 GRNs; paired diff vs I-k0; AGGREGATE parity; dot plot.
usage: script IN_DIR OUT_DIR"""
import sys, os, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from scipy import stats
IN, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
GRNS = [501, 502, 503, 504, 505, 506]; TOPOS = ["bifurcation", "tree"]; CONDS = ["C0", "C1", "C2", "C3"]; SIMS = [1, 2]
CANDS = ["F-k0", "F-mean0..8", "F-k8", "F-vote", "F-pool", "F-cen", "I-k0", "I-cen", "I-mean0..8", "Pearson", "AGREE"]
MET = ["BA", "recall_act", "recall_rep", "coverage", "BA_called", "root_disagree", "null_BA_mean"]
fs = [f"{IN}/e13_g{g}_{t}_{c}_s{s}.json" for g in GRNS for t in TOPOS for c in CONDS for s in SIMS]
assert not [f for f in fs if not os.path.exists(f)], "INCOMPLETE"
J = [json.load(open(f)) for f in fs]; assert len({j["code_sha"] for j in J}) == 1 and all(j["parity"] == "PASS" for j in J)
R = pd.DataFrame([dict(grn=j["grn"], topo=j["topo"], cond=j["cond"], sim=j["sim"], cand=k, **{m: j[k][m] for m in MET}, null_p=j[k]["null_p"]) for j in J for k in CANDS])
R.to_csv(f"{OUT}/PER_DATASET.tsv", sep="\t", index=False)
Gc = R.groupby(["cand", "grn", "topo", "cond"])[MET].mean().reset_index()      # sims averaged
# AGGREGATE PARITY
ref = pd.read_csv("results/sergio_tf32rep_2026_10_22/signed_2026_10_28/agg/GRN_MEAN_SD_ALL.tsv", sep="\t").sign_BA_mean.iloc[0]
ip = pd.read_csv("results/sergio_tf32rep_2026_10_22/iso_sign_2026_10_28/agg/PER_DATASET.tsv", sep="\t")
mine = Gc[Gc.cand == "F-mean0..8"].groupby("grn").BA.mean().mean(); ik0 = Gc[Gc.cand == "I-k0"].groupby("grn").BA.mean().mean()
print("PARITY F-mean0..8 GRN-mean BA", round(mine, 4), "stored", round(ref, 4)); assert abs(mine - ref) < 1e-9
sub = ip[(ip.get("strata", "one") == "one")] if "strata" in ip else ip
col = "sign_BA"; ipm = sub.groupby("grn")[col].mean().mean() if "grn" in sub else np.nan
print("PARITY I-k0 GRN-mean BA", round(ik0, 4), "stored(iso PER_DATASET)", round(ipm, 4)); assert abs(ik0 - ipm) < 1e-9
tq = stats.t.ppf(.975, 5)
def per_grn(df):   # df: Gc rows for one candidate subset -> per-GRN value (equal weights over remaining topo x cond)
    return df.groupby(["cand", "grn"])[MET].mean().reset_index()
SLICES = [("overall", Gc)] + [(f"cond={c}", Gc[Gc.cond == c]) for c in CONDS] + [(f"topo={t}", Gc[Gc.topo == t]) for t in TOPOS]
rows = []
for name, df in SLICES:
    P = per_grn(df); base = P[P.cand == "I-k0"].set_index("grn").BA
    for k in CANDS:
        p = P[P.cand == k].set_index("grn"); r = dict(slice=name, cand=k)
        for m in MET:
            v = p[m].astype(float); mu, sd = v.mean(), v.std(ddof=1); r[m] = mu; r[m + "_sd"] = sd
            if m == "BA": r["BA_lo"], r["BA_hi"] = mu - tq * sd / 6 ** .5, mu + tq * sd / 6 ** .5
        d = (p.BA - base); mu, sd = d.mean(), d.std(ddof=1)
        r.update(dBA_vs_Ik0=mu, dBA_lo=mu - tq * sd / 6 ** .5, dBA_hi=mu + tq * sd / 6 ** .5, dBA_n_grn_pos=int((d > 0).sum()))
        r["null_p_max"] = float(R[R.cand == k].null_p.max()); rows.append(r)
S = pd.DataFrame(rows); S.to_csv(f"{OUT}/SUMMARY.tsv", sep="\t", index=False)
pd.set_option("display.width", 250)
print(S[S.slice == "overall"][["cand", "BA", "BA_sd", "BA_lo", "BA_hi", "recall_act", "recall_rep", "coverage", "BA_called", "root_disagree", "null_BA_mean", "dBA_vs_Ik0", "dBA_lo", "dBA_hi", "dBA_n_grn_pos"]].round(3).to_string(index=False))
print(S.pivot(index="cand", columns="slice", values="BA").loc[CANDS].round(3).to_string())
# ---- dot plot (SI style of gb_si_figs_2026_11_03.py)
G1, G2 = "#555555", "#888888"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.linewidth": .7, "axes.edgecolor": G2, "xtick.color": G1, "ytick.color": G1, "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": .5, "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False, "legend.frameon": False})
CP = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; CL = ["None", "Temporal (A)", "Branch (B)", "A+B"]
f, ax = plt.subplots(1, 3, figsize=(7.2, 3.3), gridspec_kw=dict(width_ratios=[1.5, 1, 1]), sharey=True); y = np.arange(len(CANDS))[::-1]
O = S[S.slice == "overall"].set_index("cand").loc[CANDS]
for i, c in enumerate(CANDS):
    for j, cc in enumerate(CONDS):
        ax[0].plot(S[(S.slice == f"cond={cc}") & (S.cand == c)].BA, y[i] + (j - 1.5) * .12, "o", ms=2.2, color=CP[j], alpha=.8)
    ax[0].errorbar(O.BA[c], y[i], xerr=[[O.BA[c] - O.BA_lo[c]], [O.BA_hi[c] - O.BA[c]]], fmt="D", color="k", ms=3.5, lw=1, capsize=0, zorder=5)
ax[0].plot(O.null_BA_mean.mean() * np.ones(2), [-.5, len(CANDS) - .5], ":", color=G2); ax[0].set_yticks(y); ax[0].set_yticklabels(CANDS); ax[0].set_xlabel("Sign BA (240 direct edges)")
ax[1].errorbar(O.dBA_vs_Ik0, y, xerr=[O.dBA_vs_Ik0 - O.dBA_lo, O.dBA_hi - O.dBA_vs_Ik0], fmt="o", color="k", ms=3.2, lw=1, capsize=0); ax[1].axvline(0, color=G2, lw=.7); ax[1].set_xlabel("Paired BA difference vs I-k0")
ax[2].plot(O.recall_act, y, "o", color="#2a78d6", ms=3.2, label="activation"); ax[2].plot(O.recall_rep, y, "s", color="#eb6834", ms=3.2, label="repression"); ax[2].plot(O.coverage, y, "^", color=G1, mfc="white", ms=3.2, label="coverage")
ax[2].set_xlabel("Recall / coverage"); ax[2].legend(loc="lower left", fontsize=5.5)
from matplotlib.lines import Line2D
ax[0].legend([Line2D([], [], marker="o", color=CP[j], ls="", ms=3) for j in range(4)] + [Line2D([], [], marker="D", color="k", ls="", ms=3.5)], CL + ["GRN mean, t95 CI"], loc="lower left", fontsize=5.5)
for a_, s_ in zip(ax, "abc"): a_.text(-.02, 1.04, s_, transform=a_.transAxes, fontweight="bold", fontsize=9, ha="right", va="bottom")
f.savefig(f"{OUT}/E13_sign_estimator_dotplot.png", dpi=300, bbox_inches="tight")
