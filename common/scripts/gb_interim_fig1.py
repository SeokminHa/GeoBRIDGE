#!/usr/bin/env python
# Cross-system benchmark using available comparator outputs.
import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
import gb_killtests as KT  # noqa: F401  (keeps RAWH)
import gb_primary as GP
import gb_real_uniform as RU


def loader(path, key, src_key="sources"):
    if not path.exists():
        return None
    z = np.load(path, allow_pickle=True)
    return {"g": {g: i for i, g in enumerate(z["genes"].astype(str))}, "s": [x.upper() for x in z[src_key].astype(str)],
            "M": np.asarray(z[key], float)}


def row(L, src, genes):
    if L is None or src not in L["s"]:
        return None
    r = L["M"][L["s"].index(src)]
    return np.array([r[L["g"][g]] if g in L["g"] else np.nan for g in genes])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--systems", nargs="+", required=True)
    a = ap.parse_args()
    out = RU.OUT / "interim_fig1"
    out.mkdir(parents=True, exist_ok=True)
    recs = []
    for system in [s.upper() for s in a.systems]:
        d = RU.system_dir(system)
        L = {"|Pearson|": loader(d / "pearson.npz", "score"), "GENIE3": loader(d / "genie3_v2.npz", "score"),
             "GRNBoost2": loader(d / "grnboost2_v2.npz", "score"), "RegVelo": loader(d / "regvelo_v2_full.npz", "score_signed"),
             "Velorama": loader(d / "velorama_v2_native.npz", "score")}
        avail = [m for m, v in L.items() if v is not None]
        print(f"[interim] {system}: available {avail}", flush=True)
        for name, family, _s, src, eps, n, npos in RU.trace_endpoints(systems=[system]):
            src = src.upper()
            E = eps["U"]
            genes = E.obs_genes
            V = {"GeoBRIDGE": GP.zcons(E.RAWH)}
            for m in avail:
                v = row(L[m], src, genes)
                if v is None:
                    continue
                if m == "RegVelo":
                    V["RegVelo |J|"], V["RegVelo +J"], V["RegVelo -J"] = np.abs(v), v, -v
                else:
                    V[m] = v
            common = E.eval_mask.copy()
            for v in V.values():
                common &= np.isfinite(v)
            y = np.zeros(len(genes), bool)
            y[E.eval_mask] = np.asarray(E.y, bool)
            yy = y[common]
            if yy.sum() < 5 or (~yy).sum() < 5:
                continue
            for m in ["GeoBRIDGE", "|Pearson|", "GENIE3", "GRNBoost2", "RegVelo |J|", "RegVelo +J", "RegVelo -J", "Velorama"]:
                if m not in V:
                    recs.append(dict(system=system, family=family, endpoint=name, source=src, method=m, status="MISSING"))
                    continue
                s = V[m][common]
                ap_ = average_precision_score(yy, s)
                recs.append(dict(system=system, family=family, endpoint=name, source=src, method=m, status="OK",
                                 n=int(common.sum()), n_pos=int(yy.sum()), AP=ap_, AP_lift=ap_ / yy.mean(),
                                 AUROC=roc_auc_score(yy, s)))
    D = pd.DataFrame(recs)
    D.to_csv(out / "endpoints.tsv", sep="\t", index=False)
    O = D[D.status == "OK"].copy()
    O["fam"] = np.where(O.family == "F1", "regulon", "functional")
    T = O.groupby(["system", "fam", "method", "source"])[["AP_lift", "AUROC"]].mean().groupby(["system", "fam", "method"]).mean()
    T.to_csv(out / "summary.tsv", sep="\t")
    pd.set_option("display.width", 250)
    for met in ("AUROC", "AP_lift"):
        print(f"== {met} (TF-mean per system)")
        print(T[met].unstack("method").round(3).to_string())
    W = O.pivot_table(index=["system", "fam", "endpoint"], columns="method", values="AUROC")
    others = [c for c in W.columns if c != "GeoBRIDGE" and c not in ("RegVelo +J", "RegVelo -J")]
    W["GB_wins_vs_best_other"] = W["GeoBRIDGE"] > W[others].max(1)
    print("== fraction of endpoints where GB AUROC > best other (|J| RegVelo only)")
    print(W.groupby(level=[0, 1]).GB_wins_vs_best_other.mean().round(3).to_string())


if __name__ == "__main__":
    main()
