#!/usr/bin/env python
# Evaluate KLF1 predictions after checksum verification.
import gzip
import hashlib
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

R = __import__("os").environ["GEOBRIDGE_ROOT"]
OUT = f"{R}/results/klf1_heldout"
PRED_SHA = "bc638abb776db3b218b3f9c42371317e39e80f5afe053c0fb18cfe46c9344ad3"
NB = 2000


def load_pred():
    p = f"{OUT}/predictions.npz"
    if hashlib.sha256(open(p, "rb").read()).hexdigest() != PRED_SHA:
        raise SystemExit("REFUSING: predictions.npz does not match the frozen sha256")
    z = np.load(p, allow_pickle=True)
    return (z["genes"].astype(str), list(z["methods"].astype(str)), z["scores"].astype(float), z["detection_tertile"].astype(str))


def auc_ci(y, s, rng):
    pos, neg = np.flatnonzero(y), np.flatnonzero(~y)
    b = [roc_auc_score(np.r_[np.ones(len(pos)), np.zeros(len(neg))], np.r_[s[rng.choice(pos, len(pos))], s[rng.choice(neg, len(neg))]])
         for _ in range(NB)]
    return roc_auc_score(y, s), np.percentile(b, 2.5), np.percentile(b, 97.5)


def ko_truth(path):
    D = pd.read_csv(path, sep="\t")
    D["sample_1"], D["sample_2"] = D.sample_1.astype(str).str.upper(), D.sample_2.astype(str).str.upper()   # Normalize sample labels to uppercase.
    D = D[((D.sample_1 == "Q2") & (D.sample_2 == "Q3")) | ((D.sample_1 == "Q3") & (D.sample_2 == "Q2"))]
    D = D[(D.status == "OK") & ~D.gene.astype(str).str.contains(",") & (D.gene.astype(str) != "-")].copy()
    l2 = D["log2(fold_change)"].astype(float)
    D["ko_effect"] = np.where(D.sample_1 == "Q3", l2, -l2)            # log2(value_2/value_1) -> orient to KO / het
    D["resp"] = D.significant.astype(str).str.lower() == "yes"
    D["sym"] = D.gene.astype(str).str.upper()
    D = D[~D.sym.duplicated(keep=False)].set_index("sym")
    if len(D) == 0:
        raise SystemExit("REFUSING: no Q2 vs Q3 rows in the cuffdiff table")
    return D


def evaluate(genes, methods, S, y, label, rng, extra=None):
    rows = []
    gb = S[methods.index("GeoBRIDGE")]
    allok = np.isfinite(S).all(0)
    for m in methods:
        v = S[methods.index(m)]
        for scope, ok in (("pairwise", np.isfinite(gb) & np.isfinite(v)), ("all_methods", allok)):
            yy = y[ok]
            if yy.sum() < 5 or (~yy).sum() < 20:
                continue
            a, lo, hi = auc_ci(yy, v[ok], rng)
            rows.append(dict(endpoint=label, scope=scope, method=m, n=int(ok.sum()), n_pos=int(yy.sum()), AUROC=a, lo=lo, hi=hi,
                             AP_lift=average_precision_score(yy, v[ok]) / yy.mean(), **(extra or {})))
    return rows


def main_ko(path):
    rng = np.random.default_rng(20260927)
    genes, methods, S, tert = load_pred()
    T = ko_truth(path)
    gu = np.char.upper(genes)
    tested = np.isin(gu, T.index)
    resp = np.array([bool(T.resp.get(g, False)) for g in gu]) & tested
    eff = np.array([float(T.ko_effect.get(g, np.nan)) for g in gu])
    Sx, gx, tx = S[:, tested], genes[tested], tert[tested]
    rows = evaluate(gx, methods, Sx, resp[tested], "PRIMARY any KO response", rng)
    rows += evaluate(gx, methods, Sx, (resp & (eff < 0))[tested], "SECONDARY KO-down (activation-assoc)", rng)
    rows += evaluate(gx, methods, Sx, (resp & (eff > 0))[tested], "SECONDARY KO-up (repression-assoc)", rng)
    R_ = pd.DataFrame(rows)
    gb = R_[(R_.endpoint.str.startswith("PRIMARY")) & (R_.method == "GeoBRIDGE") & (R_.scope == "all_methods")]
    verdict = "SUCCESS" if len(gb) and gb.lo.iloc[0] > .5 else "NOT_MET"
    # applicability (secondary)
    y = resp[tested]
    app = []
    for comp in ("|Pearson|", "GENIE3", "GRNBoost2"):
        gbv, cv = Sx[methods.index("GeoBRIDGE")], Sx[methods.index(comp)]
        ok = np.isfinite(gbv) & np.isfinite(cv)
        d = {}
        for lev in ("LOW", "HIGH"):
            mk = ok & (tx == lev)
            d[lev] = (mk, roc_auc_score(y[mk], gbv[mk]) - roc_auc_score(y[mk], cv[mk])) if y[mk].sum() >= 5 and (~y[mk]).sum() >= 20 else (mk, np.nan)
        t_obs = d["LOW"][1] - d["HIGH"][1]
        bs = []
        for _ in range(NB):
            vals = []
            for lev in ("LOW", "HIGH"):
                mk = d[lev][0]
                p, n = np.flatnonzero(mk & y), np.flatnonzero(mk & ~y)
                if len(p) < 5:
                    vals.append(np.nan)
                    continue
                ip, ineg = rng.choice(p, len(p)), rng.choice(n, len(n))
                yy = np.r_[np.ones(len(ip)), np.zeros(len(ineg))]
                vals.append(roc_auc_score(yy, np.r_[gbv[ip], gbv[ineg]]) - roc_auc_score(yy, np.r_[cv[ip], cv[ineg]]))
            bs.append(vals[0] - vals[1])
        app.append(dict(comparator=comp, delta_LOW=d["LOW"][1], delta_HIGH=d["HIGH"][1], T=t_obs, lo=np.nanpercentile(bs, 2.5),
                        hi=np.nanpercentile(bs, 97.5), primary=(comp == "|Pearson|")))
    A = pd.DataFrame(app)
    R_.to_csv(f"{OUT}/ko_results.tsv", sep="\t", index=False)
    A.to_csv(f"{OUT}/ko_applicability.tsv", sep="\t", index=False)
    json.dump(dict(truth_file=path, truth_sha256=hashlib.sha256(open(path, "rb").read()).hexdigest(), n_tested=int(tested.sum()),
                   n_resp=int(resp.sum()), primary_verdict=verdict, predictions_sha256=PRED_SHA),
              open(f"{OUT}/ko_receipt.json", "w"), indent=1)
    pd.set_option("display.width", 220)
    print(R_.round(4).to_string(index=False))
    print(A.round(4).to_string(index=False))
    print("PRIMARY VERDICT:", verdict)


def main_chip(path):
    sys.path.insert(0, f"{R}/scratch")
    import gb_gata1_truth_v2 as TV
    rng = np.random.default_rng(20260928)
    genes, methods, S, _ = load_pred()
    rows = []
    with gzip.open(path, "rt") as fh:
        for ln in fh:
            if ln.startswith(("#", "track", "browser")):
                continue
            f = ln.rstrip("\n").split("\t")
            st, en = int(f[1]), int(f[2])
            summit = st + int(f[9]) if len(f) >= 10 and f[9] not in (".", "-1") else (st + en) // 2
            sig = float(f[6]) if len(f) >= 7 else float(f[4]) if len(f) >= 5 else 1.0
            rows.append(dict(chrom=f[0], summit=summit, signal=sig))
    pk = pd.DataFrame(rows)
    F = TV.gene_features(TV.load_tss(), pk).set_index("gene")
    F.index = F.index.str.upper()
    F = F[~F.index.duplicated(keep=False)]
    gu = np.char.upper(genes)
    ok = np.isin(gu, F.index)
    ko = pd.read_csv(f"{OUT}/ko_results.tsv", sep="\t") if __import__("os").path.exists(f"{OUT}/ko_results.tsv") else None
    out = []
    for win in ("BOUND_PROM", "BOUND_10KB"):
        b = np.array([bool(F[win].get(g, False)) for g in gu]) & ok
        out += evaluate(genes[ok], methods, S[:, ok], b[ok], f"ChIP {win} bound vs unbound", rng)
    O = pd.DataFrame(out)
    O.to_csv(f"{OUT}/chip_results.tsv", sep="\t", index=False)
    F.to_csv(f"{OUT}/chip_gene_features.tsv", sep="\t")
    print(O.round(4).to_string(index=False), "\nko results present:", ko is not None)


if __name__ == "__main__":
    {"ko": main_ko, "chip": main_chip}[sys.argv[1]](sys.argv[2])
