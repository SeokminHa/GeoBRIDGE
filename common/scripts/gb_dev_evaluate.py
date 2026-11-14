#!/usr/bin/env python
# Evaluate saved scores against endpoint labels.
from __future__ import annotations
import hashlib, json, os, sys, warnings
import numpy as np, pandas as pd
warnings.filterwarnings("ignore")

R = __import__("os").environ["GEOBRIDGE_ROOT"]


# Threshold-grouped AP of |score| against binary y over supported, evaluable entries
def average_precision(score, y, supported):
    m = supported & np.isfinite(score) & (y >= 0)
    if m.sum() < 10 or y[m].sum() == 0: return np.nan, int(m.sum()), 0
    sc = np.abs(score[m]); t = y[m].astype(bool)
    o = np.argsort(-sc, kind="stable"); sc = sc[o]; t = t[o]
    tp = np.cumsum(t); k = np.arange(1, t.size + 1)
    # end index of each tie group
    last = np.zeros(t.size, int)
    i = 0
    while i < t.size:
        j = i
        while j + 1 < t.size and sc[j + 1] == sc[i]: j += 1
        last[i:j + 1] = j; i = j + 1
    prec = tp[last] / k[last]
    return float((prec * t).sum() / t.sum()), int(m.sum()), int(t.sum())


# Small fixed batch of shared label permutations that preserve class counts
def permutation_reference(score, y, supported, n_perm=20, seed=0):
    m = supported & np.isfinite(score) & (y >= 0)
    if m.sum() < 10 or y[m].sum() == 0: return np.nan, np.nan
    rng = np.random.default_rng(seed)
    yy = y[m].astype(bool); sc = score[m]
    out = []
    for _ in range(n_perm):
        p = rng.permutation(yy.size)
        ap, _, _ = average_precision(sc, yy[p].astype(int), np.ones(sc.size, bool))
        out.append(ap)
    return float(np.nanmean(out)), float(np.nanstd(out))


# Returns y in {1 pos, 0 neg, -1 unknown} plus the polarity vector
def truth_panc(genes):
    de = pd.read_csv(f"{R}/real_1_Pancreas/external_validation/aldous_rfx6/results/aldous_PPs_WT_vs_KO_DE.tsv", sep="\t")
    hom = pd.read_csv(f"{R}/data/orthology/HOM_MouseHumanSequence.rpt", sep="\t", low_memory=False)
    key = hom.columns[0]; org = hom["Common Organism Name"]
    mm = hom[org.str.startswith("mouse")].groupby(key)["Symbol"].apply(list)
    hs = hom[org.str.startswith("human")].groupby(key)["Symbol"].apply(list)
    h2m = {hs[k][0].upper(): mm[k][0] for k in set(mm.index) & set(hs.index)
           if len(mm[k]) == 1 and len(hs[k]) == 1}
    gp = {g: i for i, g in enumerate(genes)}
    y = np.full(len(genes), -1, int); pol = np.zeros(len(genes)); seen = {}
    dupsym = set(de.gene_name.astype(str).str.upper()[de.gene_name.astype(str).str.upper().duplicated(keep=False)])
    for hg, padj, l in zip(de.gene_name.astype(str).str.upper(), de.padj, de.log2FoldChange):
        m = h2m.get(hg)
        if m is None or m not in gp or hg in dupsym: continue
        i = gp[m]
        if not np.isfinite(padj): continue
        if not np.isfinite(l):
            raise SystemExit(f"REFUSING: RFX6 DE row {hg} has finite padj but non-finite log2FC")
        yi = 1 if (padj < 0.05) else 0
        pi = (-1.0 if l < 0 else 1.0) if yi == 1 else 0.0   # KO-down = activation(-1), KO-up = repression(+1)
        if i in seen and seen[i] != (yi, pi):
            raise SystemExit(f"REFUSING: conflicting RFX6 DE rows map to {m}: {seen[i]} vs {(yi, pi)}")
        seen[i] = (yi, pi); y[i] = yi; pol[i] = pi
    for i, g in enumerate(genes):
        if g.upper() == "RFX6":
            y[i] = -1; pol[i] = 0.0     # self-excluded from both the label and the polarity tally.
    return y, pol


def truth_gata1(genes):
    d = pd.read_csv(f"{R}/results/gata1_clean_2026_08_13/chipseq_labels.tsv", sep="\t")
    up = dict(zip(d.gene.astype(str), d.cons_up.astype(int)))
    dn = dict(zip(d.gene.astype(str), d.cons_down.astype(int)))
    y = np.full(len(genes), -1, int); pol = np.zeros(len(genes))
    for i, g in enumerate(genes):
        if g not in up: continue                            # not in the truth resource -> unknown
        u, w = up[g], dn[g]
        y[i] = 1 if (u or w) else 0
        if u: pol[i] = +1.0                                 # cons_up  = REPRESSION
        if w: pol[i] = -1.0                                 # cons_down = ACTIVATION
        if g.upper() == "GATA1": y[i] = -1; pol[i] = 0.0    # self-excluded from both vectors
    return y, pol


TRUTH = {"PANC_RFX6": truth_panc, "GATA1": truth_gata1}


POLARITY_SPLIT = True


def main(score_json, out_json):
    res = json.load(open(score_json))
    binding = {k: res[k] for k in ("scores", "comparators", "raw_stats", "genes", "focal",
                                   "h5ad_sha256", "pair_sha256", "roster_sha256",
                                   "kernel_sha256", "producer_sha256")}
    binding["folds"] = {k: v["fold_sha"] for k, v in res["diagnostics"].items()}
    binding["retained"] = {k: v["retained_pair_sha"] for k, v in res["diagnostics"].items()}
    binding["dropped"] = {k: v["n_dropped"] for k, v in res["diagnostics"].items()}
    binding["lambda"] = {k: v["lambda_hist"] for k, v in res["diagnostics"].items()}
    recomputed = hashlib.sha256(json.dumps(binding, sort_keys=True).encode()).hexdigest()
    if recomputed != res["score_sha256"]:
        raise RuntimeError(f"score_sha256 MISMATCH: stored {res['score_sha256'][:16]} "
                           f"recomputed {recomputed[:16]}; the score file has been altered")
    print(f"score_sha256 VERIFIED: {recomputed[:16]}")
    genes = np.asarray(res["genes"]).astype(str)
    y, pol = TRUTH[res["substrate"]](genes)
    ev = (y >= 0)
    panels = {"BOTH": y}
    if POLARITY_SPLIT:
        for nm, sgn in (("ACTIVATION", -1.0), ("REPRESSION", +1.0)):
            yy = y.copy()
            yy[(y == 1) & (pol != sgn)] = -1
            panels[nm] = yy
    rows = []
    for tag, arms in res["scores"].items():
        for key, e in arms.items():
            sc = np.asarray(e["score"], float); sup = np.asarray(e["supported"], bool)
            for pnm, yy in panels.items():
                ap, n_ev, n_pos = average_precision(sc, yy, sup & (yy >= 0))
                pm, ps = permutation_reference(sc, yy, sup & (yy >= 0))
                prev = n_pos / n_ev if n_ev else np.nan
                rows.append(dict(tag=tag, panel=pnm, kind="arm", name=key, AP=ap, n_eval=n_ev,
                                 n_pos=n_pos, prevalence=prev,
                                 AP_lift=(ap / prev if prev and np.isfinite(ap) else np.nan),
                                 perm_AP_mean=pm, perm_AP_sd=ps))
    for tag, comps in res.get("comparators", {}).items():
        for key, e in comps.items():
            sc = np.asarray(e["value"], float); sup = np.asarray(e["supported"], bool)
            for pnm, yy in panels.items():
                ap, n_ev, n_pos = average_precision(sc, yy, sup & (yy >= 0))
                pm, ps = permutation_reference(sc, yy, sup & (yy >= 0))
                prev = n_pos / n_ev if n_ev else np.nan
                rows.append(dict(tag=tag, panel=pnm, kind="comparator", name=key, AP=ap, n_eval=n_ev,
                                 n_pos=n_pos, prevalence=prev,
                                 AP_lift=(ap / prev if prev and np.isfinite(ap) else np.nan),
                                 perm_AP_mean=pm, perm_AP_sd=ps))
    df = pd.DataFrame(rows)
    out = dict(substrate=res["substrate"], score_sha256=res["score_sha256"],
               endpoint="bound", n_U_deploy=int(len(genes)), n_U_eval=int(ev.sum()),
               n_positives=int((y == 1).sum()), n_unknown=int((y == -1).sum()),
               grade="Development evaluation; certification incomplete.", rows=rows)
    json.dump(out, open(out_json, "w"), indent=1)
    print(f"{res['substrate']}  U_deploy={len(genes)}  U_eval={int(ev.sum())}  "
          f"positives={int((y==1).sum())}  UNKNOWN={int((y==-1).sum())}")
    print(f"score_sha256={res['score_sha256'][:16]}   GRADE: DEVELOPMENT, uncertified\n")
    df["channel"] = df["tag"].str.split("|").str[1]
    df["root"] = df["tag"].str.split("|").str[0]
    piv = (df[df.AP.notna()].groupby(["panel", "kind", "channel", "name"])
             .agg(AP_mean=("AP", "mean"), AP_sd=("AP", "std"), lift=("AP_lift", "mean"),
                  perm=("perm_AP_mean", "mean"), n_eval=("n_eval", "max"),
                  n_pos=("n_pos", "max")).reset_index()
             .sort_values(["panel", "AP_mean"], ascending=[True, False]))
    piv["excess"] = piv["AP_mean"] - piv["perm"]
    for pnm in ("ACTIVATION", "REPRESSION", "BOTH"):
        sub = piv[piv.panel == pnm]
        if not len(sub): continue
        print(f"\n{pnm}  (n_pos={int(sub.n_pos.max())}, prevalence "
              f"{sub.n_pos.max()/sub.n_eval.max():.4f})")
        print(sub.head(12)[["kind", "channel", "name", "AP_mean", "AP_sd", "perm", "excess"]]
              .to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
