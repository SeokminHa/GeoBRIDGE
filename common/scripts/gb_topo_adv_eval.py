#!/usr/bin/env python
"""DIAGNOSTIC (truth used only to EVALUATE stored scores; nothing regenerated, no score tuned).
Task C: topology-dependent advantage of GB PRIMARY vs classics on World-P (confirm_P GRNs 101-109; dev core_P GRNs 0-2 + Velorama/RegVelo).
Scores: PRIMARY = gb_b4_score.zc(k=0..8) (identical to B4_DATASETS), RAWABS_k0 = min over roots of raw |S_U| at k=0 (no within-source z),
GENIE3 / GRNBOOST2 / Pearson_abs from raw/comparators.npz. Universes: REG = 20 true regulators x 99 (B4 primary universe),
ALL39 = 20 regulators + 19 decoy leaves (all candidates).
  python gb_topo_adv_eval.py rows confirm|dev [N]   -> results/sergio_drift/topo_adv/ROWS_{set}.tsv (N = first N datasets, small-N check)
  python gb_topo_adv_eval.py summary               -> per-topology GRN-paired differences (printed + TSVs)"""
import glob, json, os, sys
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
import gb_b4_score as B
import gb_sergio_band_eval as E
import gb_sergio_drift_v5 as V
from sklearn.metrics import average_precision_score as APs, roc_auc_score as AUC

RES = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/results/sergio_drift")
OUT = f"{RES}/topo_adv"
PAT = {"confirm": f"{RES}/confirm_P/*/*_P_q3_G*_g1[0-9][0-9]_s[12]/record.json", "dev": f"{RES}/core_P/*/*_P_q3_G*_g*_s0/record.json"}
METHODS = ("PRIMARY", "RAWABS_k0", "GENIE3", "GRNBOOST2", "Pearson_abs")


def one(rj):
    dd = os.path.dirname(rj)
    rec = json.load(open(rj))
    g = int(rec["grn_seed"])
    if not os.path.exists(f"{dd}/raw/geobridge_raw.npz"):
        return None
    Z, C = np.load(f"{dd}/raw/geobridge_raw.npz"), np.load(f"{dd}/raw/comparators.npz")
    G = V.generate_grn(g)
    src, cols = np.asarray(Z["source_gene_index"], int), np.asarray(Z["cols_PI"], int)
    prim = np.array([s in G["regs"] for s in src])
    direct = {(int(s), int(t)) for s, t in G["gt"]}
    two = E.strict_two_hop(direct, V.N_GENES)
    Y = np.array([[(s, int(t)) in direct for t in cols[i]] for i, s in enumerate(src)])
    L = np.array([[1 if (s, int(t)) in direct else (0 if (s, int(t)) in two else -1) for t in cols[i]] for i, s in enumerate(src)])
    A = {r: np.asarray(Z[f"TRUE|U|{r}|absS"], float) for r in E.ROOTS}                  # (31, nsrc, ngenes)
    per = [{r: A[r][:, i, cols[i]].T for r in E.ROOTS} for i in range(len(src))]
    views = {"PRIMARY": B.zc(per, range(0, 9)),
             "RAWABS_k0": np.minimum(*[np.take_along_axis(A[r][0], cols, 1) for r in E.ROOTS])}
    for m in METHODS[2:]:
        views[m] = np.asarray(C[m], float)
    out = dict(dataset=os.path.basename(dd), obs=rec["obs"], topo=rec["topo"], grn=g, G=round(float(rec["G_target"]), 2), sim=int(rec["sim_seed"]))
    for nm, sc in views.items():
        assert sc.shape == Y.shape, (nm, sc.shape, Y.shape)
        for un, rows in (("REG", prim), ("ALL39", np.ones(len(src), bool))):
            s, y, l = sc[rows], Y[rows], L[rows]
            ok = np.isfinite(s)
            di = ok & (l >= 0)
            out[f"{nm}|{un}|AP"] = APs(y[ok], s[ok])
            out[f"{nm}|{un}|AUROC"] = AUC(y[ok], s[ok])
            out[f"{nm}|{un}|cov"] = float(ok.mean())
            if un == "REG":
                out[f"{nm}|{un}|DvI"] = AUC(l[di] == 1, s[di])
                out[f"{nm}|{un}|macroAP"] = np.mean([APs(y[i][ok[i]], s[i][ok[i]]) for i in range(len(y)) if y[i][ok[i]].any()])
    out["prev_REG"], out["prev_ALL39"] = Y[prim].mean(), Y.mean()
    return out


def rows(which, n=None):
    files = sorted(glob.glob(PAT[which]))
    keep = (lambda g: g < 100) if which == "dev" else (lambda g: 101 <= g <= 109)      # GRN 100 = exposed canary, refused as in B4
    files = [f for f in files if keep(int(json.load(open(f))["grn_seed"]))]
    if n:
        files = files[:int(n)]
    with Pool(min(16, len(files))) as p:
        R = [r for r in p.map(one, files) if r is not None]
    os.makedirs(OUT, exist_ok=True)
    D = pd.DataFrame(R)
    D.to_csv(f"{OUT}/ROWS_{which}{'_n' + str(n) if n else ''}.tsv", sep="\t", index=False)
    print(len(files), "records,", len(D), "scored", flush=True)
    return D


def grn_level(D, cols, by):
    """seeds averaged within (grn,topo,G,obs); then equal weight per G and per obs inside each (grn, *by) stratum"""
    D = D.groupby(["grn", "topo", "G", "obs"], as_index=False)[cols].mean()
    D = D.groupby(["grn", "topo", "G"] + [b for b in by if b not in ("topo", "G")], as_index=False)[cols].mean()   # over obs
    return D.groupby(["grn"] + list(by), as_index=False)[cols].mean()


def ci(d):
    from scipy.stats import t as T
    d = np.asarray(d, float); n = len(d)
    h = T.ppf(.975, n - 1) * d.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return d.mean(), d.mean() - h, d.mean() + h, n


def paired(Gm, by, comps, metrics):
    res = []
    for key, x in Gm.groupby(list(by)):
        key = key if isinstance(key, tuple) else (key,)
        for un, met in metrics:
            for c in comps:
                if f"{c}|{un}|{met}" not in x:
                    continue
                d = (x[f"PRIMARY|{un}|{met}"] - x[f"{c}|{un}|{met}"]).dropna()
                if len(d) < 2:
                    continue
                m, lo, hi, n = ci(d)
                res.append(dict(zip(by, key), univ=un, metric=met, comp=c, PRIMARY=x[f"PRIMARY|{un}|{met}"].mean(),
                                comp_val=x[f"{c}|{un}|{met}"].mean(), diff=m, lo=lo, hi=hi, n_grn=n))
    return pd.DataFrame(res)


def slope(Gm, comps, un="REG", met="AP"):
    """per GRN: OLS slope of (PRIMARY - comp) on topology complexity 1..4; t CI across GRNs"""
    res = []
    for c in comps:
        s = []
        for g, x in Gm.groupby("grn"):
            cx = x.topo.map(E.TOPOLOGY_COMPLEXITY).values
            d = (x[f"PRIMARY|{un}|{met}"] - x[f"{c}|{un}|{met}"]).values
            if len(np.unique(cx)) >= 3:
                s.append(np.polyfit(cx, d, 1)[0])
        m, lo, hi, n = ci(s)
        res.append(dict(comp=c, univ=un, metric=met, slope_per_step=m, lo=lo, hi=hi, n_grn=n))
    return pd.DataFrame(res)


def summary():
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    order = {t: i for i, t in enumerate(E.TOPOLOGIES)}
    C = pd.read_csv(f"{OUT}/ROWS_confirm.tsv", sep="\t")
    num = [c for c in C.columns if "|" in c]
    comps = [m for m in METHODS if m != "PRIMARY"]
    mets = [("REG", "AP"), ("REG", "AUROC"), ("REG", "DvI"), ("ALL39", "AP"), ("ALL39", "AUROC")]
    # cross-check vs frozen B4 table
    B4 = pd.read_csv(f"{RES}/confirm_P/B4_DATASETS.tsv", sep="\t").set_index("dataset")
    j = C.set_index("dataset").join(B4[["PRIMARY|AP", "PRIMARY|DvI"]], how="inner")
    print("B4 parity: n", len(j), "max|dAP|", float((j["PRIMARY|REG|AP"] - j["PRIMARY|AP"]).abs().max()),
          "max|dDvI|", float((j["PRIMARY|REG|DvI"] - j["PRIMARY|DvI"]).abs().max()))
    print("datasets per topo x G:\n", C.groupby(["topo", "G"]).size().unstack())
    out = {}
    for tag, sub in (("allG", C), ("G0_015", C[C.G.isin([0.0, 0.15])])):
        Gm = grn_level(sub, num, ("topo",))
        P = paired(Gm, ("topo",), comps, mets)
        P["o"] = P.topo.map(order); P = P.sort_values(["univ", "metric", "comp", "o"]).drop(columns="o")
        out[tag] = P
        P.to_csv(f"{OUT}/PAIRED_TOPO_confirm_{tag}.tsv", sep="\t", index=False)
        print(f"\n== confirm {tag}: PRIMARY - comp, GRN-level paired (n=9 GRNs), 95% t CI ==")
        print(P.round(3).to_string(index=False))
        for un, met in (("REG", "AP"), ("REG", "DvI"), ("ALL39", "AP")):
            print(f"slope ({tag}) {un} {met}:\n", slope(Gm, comps, un, met).round(4).to_string(index=False))
    # per topo x obs and topo x G (AP REG vs GENIE3/GRNBOOST2/Pearson)
    for by in (("topo", "obs"), ("topo", "G")):
        Gm = grn_level(C, num, by)
        P = paired(Gm, by, comps[1:], [("REG", "AP")])
        P.to_csv(f"{OUT}/PAIRED_{'_'.join(by)}_confirm.tsv", sep="\t", index=False)
        print(f"\n== confirm by {by} (REG AP) ==\n", P.pivot_table(index=list(by), columns="comp", values="diff").round(3).to_string())
    # dev + Velorama/RegVelo (COMPARATORS.tsv, same 34 dev datasets)
    D = pd.read_csv(f"{OUT}/ROWS_dev.tsv", sep="\t")
    K = pd.read_csv(f"{RES.replace('sergio_drift', 'sergio_band')}/COMPARATORS.tsv", sep="\t")
    K = K[K.status == "OK"]
    for m in ("Velorama", "RegVelo", "GENIE3"):
        k = K[K.method == m].set_index("dataset")[["pooled_AP", "AUROC", "direct_vs_indirect_AUROC", "decoy_universe_AP"]]
        k.columns = [f"{m.upper()}_K|REG|AP", f"{m.upper()}_K|REG|AUROC", f"{m.upper()}_K|REG|DvI", f"{m.upper()}_K|ALL39|AP"]
        D = D.set_index("dataset").join(k, how="left").reset_index()
    print("\nCOMPARATORS GENIE3 parity vs recomputed (REG AP):",
          float((D["GENIE3_K|REG|AP"] - D["GENIE3|REG|AP"]).abs().max()), "ALL39:", float((D["GENIE3_K|ALL39|AP"] - D["GENIE3|ALL39|AP"]).abs().max()))
    Dv = D[D["VELORAMA_K|REG|AP"].notna()]
    print("dev datasets with Velorama:", len(Dv), "\n", Dv.groupby(["topo", "G", "obs"]).size().unstack().to_string())
    print("dev GRNs in Velorama subset:", sorted(Dv.grn.unique()))
    # ponytail: one dev GRN only -> no GRN-level CI possible; dataset-level t CI over (G, obs) cells = pseudo-replication within one GRN
    P = paired(Dv, ("topo",), ["VELORAMA_K", "REGVELO_K", "GENIE3", "GRNBOOST2", "Pearson_abs"], [("REG", "AP"), ("REG", "AUROC"), ("REG", "DvI"), ("ALL39", "AP")])
    P["o"] = P.topo.map(order); P = P.sort_values(["univ", "metric", "comp", "o"]).drop(columns="o")
    P.to_csv(f"{OUT}/PAIRED_TOPO_dev_velorama.tsv", sep="\t", index=False)
    print("\n== dev (Velorama/RegVelo subset; n = dev GRNs) ==\n", P.round(3).to_string(index=False))


if __name__ == "__main__":
    {"rows": lambda: rows(*sys.argv[2:4]), "summary": summary}[sys.argv[1]]()
