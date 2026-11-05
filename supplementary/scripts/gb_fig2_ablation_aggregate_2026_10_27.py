"""Aggregate the Fig 2 SI ablation (scratch/gb_fig2_ablation_2026_10_27.py outputs). Evaluation only:
 - complete-grid gate (96 datasets x 558 arm/depth rows) before any aggregation;
 - unit = GRN: equal-weight mean over topo x cond x sim within GRN (also per topology), 6 GRNs for uncertainty;
 - fixed window 0..8: paired GRN differences arm - R1/U, exact sign-flip p (two-sided, 64 patterns), Holm over the 8 non-primary arms
   per metric (resolution floor reported), leave-one-GRN-out range;
 - label-permutation control on the saved 0..8 score matrices: within-source permutation of the distance labels Dm (Y derived from it),
   identical permutation across arms, B=200 per dataset;
 - figures: fixed-window comparison (GRN dots + mean), cumulative-window and single-depth curves k=0..30; only R1/U 0..8 marked PRIMARY;
   no best-K marker.
usage: script ABL_DIR OUT_DIR"""
import sys, os, glob, itertools, json, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score as APs, roc_auc_score as AUC
ABL, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
GRNS = [501, 502, 503, 504, 505, 506]; TOPOS = ["bifurcation", "tree"]; CONDS = ["C0", "C1", "C2", "C3"]; SIMS = [1, 2]
OPS = ["R1", "R2h", "R2s"]; CHS = ["U", "V", "X"]; METRICS = ["AP", "macroAP", "AUROC", "DvI2", "DvI3"]
fs = [f"{ABL}/abl_g{g}_{t}_{c}_s{s}.tsv" for g in GRNS for t in TOPOS for c in CONDS for s in SIMS]
miss = [f for f in fs if not os.path.exists(f)]; assert not miss, f"INCOMPLETE GRID: {len(miss)} missing, e.g. {miss[:3]}"
R = pd.concat([pd.read_csv(f, sep="\t") for f in fs], ignore_index=True)
assert len(R) == 96 * 558 and not R.duplicated(["grn", "topo", "cond", "sim", "op", "ch", "kind", "K"]).any(), len(R)
assert (R.coverage == 1).all() and (R.n_pos == 240).all()
J = [json.load(open(f.replace(".tsv", ".json"))) for f in fs]
pd.DataFrame([dict(file=os.path.basename(f), parity_max=max(j["parity_absS"].values()), **j["velocity"],
                   direct_check_max=max(v for r in j["spectra"].values() for k, v in r.get("direct_check_maxabs", {}).items() if "nan" not in k) if all("direct_check_maxabs" in r for r in j["spectra"].values()) else np.nan)
              for f, j in zip(fs, J)]).to_csv(f"{OUT}/RECEIPTS.tsv", sep="\t", index=False)
R["arm"] = R.op + "/" + R.ch
def grn_mean(df, by=()): return df.groupby(["grn", *by, "arm", "kind", "K"])[METRICS].mean().reset_index()
G_all = grn_mean(R); G_topo = grn_mean(R, ("topo",))
# ---- fixed window 0..8: paired sign-flip vs R1/U
signs = np.array(list(itertools.product([1, -1], repeat=6)))
def signflip(d): obs = abs(d.mean()); return float(np.mean(np.abs((signs * d).mean(1)) >= obs - 1e-15))
def holm(p):
    p = np.asarray(p, float); o = np.argsort(p); out = np.empty_like(p); run = 0
    for i, k in enumerate(o): run = max(run, (len(p) - i) * p[k]); out[k] = min(1, run)
    return out
rows = []
for scope, G in [("all", G_all)] + [(t, G_topo[G_topo.topo == t]) for t in TOPOS]:
    W = G[(G.kind == "window") & (G.K == 8)]
    for m in METRICS:
        P = W.pivot(index="grn", columns="arm", values=m).loc[GRNS]; base = P["R1/U"]; rr = []
        for arm in [a for a in P.columns if a != "R1/U"]:
            d = (P[arm] - base).values
            logo = [np.delete(d, i).mean() for i in range(6)]
            rr.append(dict(scope=scope, metric=m, arm=arm, R1U_mean=base.mean(), arm_mean=P[arm].mean(), diff_mean=d.mean(), diff_sd=d.std(ddof=1),
                           n_grn_arm_better=int((d > 0).sum()), p_signflip=signflip(d), logo_min=min(logo), logo_max=max(logo)))
        h = holm([r["p_signflip"] for r in rr])
        for r, hh in zip(rr, h): r["p_holm8"] = hh; rows.append(r)
FX = pd.DataFrame(rows); FX.to_csv(f"{OUT}/FIXED_WINDOW_0_8_PAIRED.tsv", sep="\t", index=False)
G_all.to_csv(f"{OUT}/GRN_LEVEL_ALL.tsv", sep="\t", index=False); G_topo.to_csv(f"{OUT}/GRN_LEVEL_BY_TOPO.tsv", sep="\t", index=False)
# ---- label-permutation control (evaluation plumbing / random-label check)
rng = np.random.default_rng(20261027); NB = 200; PR = []
for g in GRNS:
    for t in TOPOS:
        for c in CONDS:
            for s in SIMS:
                z = np.load(f"{ABL}/scores_g{g}_{t}_{c}_s{s}.npz"); Dm = z["Dm"]
                for b in range(NB):
                    Dp = np.vstack([r[rng.permutation(len(r))] for r in Dm]); Y = Dp == 1; m2 = (Dp == 1) | (Dp == 2)
                    for op in OPS:
                        for ch in CHS:
                            sc = z[f"{op}|{ch}"]
                            PR.append(dict(grn=g, topo=t, cond=c, sim=s, b=b, arm=f"{op}/{ch}", AP=APs(Y.ravel(), sc.ravel()), DvI2=AUC(Dp[m2] == 1, sc[m2])))
PR = pd.DataFrame(PR); PN = PR.groupby(["arm"])[["AP", "DvI2"]].agg(["mean", "std"]); PN.to_csv(f"{OUT}/LABEL_PERMUTATION_CONTROL.tsv", sep="\t")
# ---- figures
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Nimbus Sans", "Arial", "DejaVu Sans"], "pdf.fonttype": 42,
                     "font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
COL = {"R1": "#2166ac", "R2h": "#d6604d", "R2s": "#f4a582"}; LS = {"U": "-", "V": "--", "X": ":"}
ARMS = [f"{o}/{c}" for o in OPS for c in CHS]
for scope, G in [("all", G_all)] + [(t, G_topo[G_topo.topo == t]) for t in TOPOS]:
    W = G[(G.kind == "window") & (G.K == 8)]
    fig, axs = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw=dict(wspace=.22))
    for ax, m, yl in zip(axs, ["AP", "macroAP", "DvI2"], ["AUPRC (pooled)", "AUPRC (per-TF mean)", "Direct vs indirect AUROC"]):
        for i, a in enumerate(ARMS):
            v = W[W.arm == a].set_index("grn")[m].loc[GRNS].values; op, ch = a.split("/")
            ax.bar(i, v.mean(), .75, color=COL[op], alpha=1 if ch == "U" else .55, hatch=None if ch == "U" else ("//" if ch == "V" else ".."), edgecolor="k" if a == "R1/U" else "none", lw=1.5)
            ax.scatter(np.full(6, i) + np.linspace(-.2, .2, 6), v, s=10, color="k", zorder=3)
        ax.set_xticks(range(len(ARMS))); ax.set_xticklabels(ARMS, rotation=45, ha="right"); ax.set_ylabel(yl, fontsize=12)
        if m in ("DvI2",): ax.axhline(.5, ls="--", color="k", lw=.8)
    fig.suptitle(f"Fixed window k=0..8 ({scope}); black outline = PRIMARY (R1/U); dots = 6 GRNs", fontsize=11)
    fig.savefig(f"{OUT}/ABL_FIXED_{scope}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{OUT}/ABL_FIXED_{scope}.pdf", bbox_inches="tight"); plt.close(fig)
    for kind in ("window", "single"):
        fig, axs = plt.subplots(1, 3, figsize=(18, 4.8), gridspec_kw=dict(wspace=.22))
        for ax, m, yl in zip(axs, ["AP", "macroAP", "DvI2"], ["AUPRC (pooled)", "AUPRC (per-TF mean)", "Direct vs indirect AUROC"]):
            for a in ARMS:
                op, ch = a.split("/"); P = G[(G.kind == kind) & (G.arm == a)].pivot(index="grn", columns="K", values=m)
                mu = P.mean(); se = P.std(ddof=1) / np.sqrt(len(P)) * 2.571
                ax.plot(mu.index, mu.values, ls=LS[ch], color=COL[op], lw=2.2 if a == "R1/U" else 1.3, label=a)
                ax.fill_between(mu.index, mu - se, mu + se, color=COL[op], alpha=.08 if a != "R1/U" else .18, lw=0)
            if kind == "window": ax.axvline(8, color="k", lw=.8, ls="-."); ax.text(8.3, ax.get_ylim()[1] * .98, "PRIMARY 0..8", va="top", fontsize=9)
            ax.set_xlabel("K (window k=0..K)" if kind == "window" else "single depth k"); ax.set_ylabel(yl, fontsize=12)
        axs[0].legend(ncol=3, fontsize=8.5, frameon=False)
        fig.suptitle(f"{'Cumulative-window' if kind == 'window' else 'Single-depth'} profile ({scope}); GRN mean, pointwise 95% t-band over 6 GRNs; root-min z-scores", fontsize=11)
        fig.savefig(f"{OUT}/ABL_CURVE_{kind}_{scope}.png", dpi=300, bbox_inches="tight"); fig.savefig(f"{OUT}/ABL_CURVE_{kind}_{scope}.pdf", bbox_inches="tight"); plt.close(fig)
print(FX[FX.scope == "all"].round(4).to_string(index=False)); print(PN.round(4).to_string())
