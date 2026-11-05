import sys, json, glob, numpy as np, pandas as pd
from scipy import stats
D = sys.argv[1]; OUT = sys.argv[2]; rows = []; par = []
for f in sorted(glob.glob(D + "/e12_*.json")):
    d = json.load(open(f)); par.append(d["parity"])
    for r in d["rows"]: rows.append(dict(grn=d["grn"], topo=d["topo"], cond=d["cond"], sim=d["sim"], **r))
R = pd.DataFrame(rows); assert len(R) == 96 * 7, len(R)
R["method"] = np.where(R.ref == "iso", "iso_w" + R.window.astype(str), R.ref)
R.to_csv(OUT + "/PER_DATASET.tsv", sep="\t", index=False)
M = ["sign_acc", "sign_BA", "coverage", "n_pairs_per_root_seed"]
G = R.groupby(["method", "cond", "grn"])[M].mean().reset_index()   # sims+topos averaged -> GRN
S = G.groupby(["method", "cond"])[M].agg(["mean", "std"]); S.columns = ["_".join(c) for c in S.columns]; S = S.reset_index()
A = G.groupby(["method", "grn"])[M].mean().reset_index(); A["cond"] = "ALL"; SA = A.groupby(["method", "cond"])[M].agg(["mean", "std"]); SA.columns = ["_".join(c) for c in SA.columns]
S = pd.concat([S, SA.reset_index()]); S.to_csv(OUT + "/SUMMARY_BY_METHOD_COND.tsv", sep="\t", index=False)
GG = pd.concat([G, A]); P = []
for (m, c), g in GG.groupby(["method", "cond"]):
    base = GG[(GG.method == "iso_w0.02") & (GG.cond == c)].set_index("grn")
    for k in ("sign_acc", "sign_BA"):
        d = (g.set_index("grn")[k] - base[k]).to_numpy(); ci = stats.t.interval(.95, 5, loc=d.mean(), scale=d.std(ddof=1) / np.sqrt(6)) if d.std() > 0 else (0, 0)
        P.append(dict(method=m, cond=c, metric=k, mean=g[k].mean(), diff_vs_0p02=d.mean(), ci_lo=ci[0], ci_hi=ci[1], n_grn_better=int((d > 0).sum())))
P = pd.DataFrame(P); P.to_csv(OUT + "/PAIRED_VS_0p02.tsv", sep="\t", index=False)
pd.set_option("display.width", 250); print(P[P.metric == "sign_BA"].round(4).to_string(index=False))
print(S[["method", "cond", "sign_acc_mean", "coverage_mean", "n_pairs_per_root_seed_mean"]].round(3).to_string(index=False))
print("parity: n", len(par), "max_abs", max(p["max_abs"] for p in par), "all pass")
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
W = [0.005, 0.01, 0.02, 0.04, 0.08]; fig, ax = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
for a, k in zip(ax, ("sign_BA", "sign_acc")):
    for c, col in zip(["C0", "C1", "C2", "C3"], ["#1b9e77", "#7570b3", "#d95f02", "#e7298a"]):
        v = [S[(S.method == f"iso_w{w}") & (S.cond == c)][k + "_mean"].iloc[0] for w in W]; e = [S[(S.method == f"iso_w{w}") & (S.cond == c)][k + "_std"].iloc[0] / np.sqrt(6) for w in W]
        a.errorbar(W, v, e, marker="o", color=col, label=c, capsize=2)
        a.axhline(S[(S.method == "fwd_k0_centred") & (S.cond == c)][k + "_mean"].iloc[0], color=col, ls=":", lw=1)
    a.axvline(0.02, color="k", ls="--", lw=.8); a.set_xscale("log"); a.set_xlabel("iso window (|dpt| <=)"); a.set_title(k + " (dotted = forward k=0 centred)")
ax[0].legend(); ax[0].set_ylabel("mean over 6 GRNs (+-SEM)"); plt.tight_layout(); plt.savefig(OUT + "/E12_isowindow.png", dpi=150)
