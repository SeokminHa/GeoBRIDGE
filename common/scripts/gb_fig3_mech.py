#!/usr/bin/env python
# Four-class, matched and continuous endpoint analyses.
import sys
import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata, spearmanr
from sklearn.metrics import roc_auc_score

sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
import gb_cop_alignment as CA
import gb_primary as GP
import gb_real_uniform as RU
from gb_gata1_fourclass_canon import four_classes
from gb_truth_v2_load import load_verified_gata1_v2

R = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/results")
FINAL = "--final" in sys.argv
OUT = RU.OUT / ("fig3_mech_final" if FINAL else "fig3_mech")
NB, KNN = 1000, 5
SP = "superseded_prefix_2026_09_26"


def mat(path, key, src_key, genes, src):
    z = np.load(path, allow_pickle=True)
    s = [x.upper() for x in z[src_key].astype(str)]
    g = {x: i for i, x in enumerate(z["genes"].astype(str))}
    row = np.asarray(z[key], float)[s.index(src)]
    return np.array([row[g[x]] if x in g else np.nan for x in genes])


def methods(system, src):
    rows, eg = RU.gb_raw_rows(system)
    Rw = rows[src]
    uni, _ = CA.observational_universe(system, Rw, eg)
    genes = np.asarray(eg).astype(str)[uni]
    m = lambda *a: mat(*a, genes, src)
    ru, cn = f"{R}/real_uniform/{system}", f"{R}/canon_{system}/sota"
    V = {"GeoBRIDGE": GP.zcons({h: Rw[(h, "U")][uni] for h in GP.ROOTS}),
         "|Pearson|": m(f"{ru}/pearson.npz", "score", "sources")}
    if FINAL:
        V.update({"GENIE3": m(f"{ru}/genie3_v2.npz", "score", "sources"),
                  "GRNBoost2": m(f"{ru}/grnboost2_v2.npz", "score", "sources"),
                  "Velorama": m(f"{ru}/velorama_v2_native.npz", "score", "sources")})
        rv = m(f"{ru}/regvelo_v2_full.npz", "score_signed", "sources")
    else:
        V.update({"GENIE3 (roster, v1)": m(f"{ru}/genie3.npz", "score", "sources"),
                  "GRNBoost2 (canon)": m(f"{cn}/grnboost2.npz", "score", "tfs"),
                  "Velorama (canon)": m(f"{cn}/velorama.npz", "score", "tfs")})
        rv = m(f"{ru}/{SP}/regvelo_v2_full.npz.oldcode", "score_signed", "sources")
    V.update({"RegVelo |J|": np.abs(rv), "RegVelo -J": -rv, "RegVelo +J": rv})
    return genes, V


def common_rows(lab, V):
    c = lab.copy()
    for v in V.values():
        if np.isfinite(v[lab]).mean() > .9:
            c &= np.isfinite(v)
    return c


def auc_idx(s, ip, ineg, w=None):
    y = np.r_[np.ones(len(ip)), np.zeros(len(ineg))]
    sw = None if w is None else np.r_[np.ones(len(ip)), w]
    return roc_auc_score(y, np.r_[s[ip], s[ineg]], sample_weight=sw)


def auc(s, pos, neg):
    return auc_idx(s, np.flatnonzero(pos), np.flatnonzero(neg)) if pos.any() and neg.any() else np.nan


def boot(masks, rng):
    return [{k: rng.choice(np.flatnonzero(m), m.sum(), replace=True) for k, m in masks.items()} for _ in range(NB)]


def ci(x):
    return tuple(np.nanpercentile(x, [2.5, 97.5]))


CONTRASTS = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D"), ("A", "D")]


# C = dict A/B/C/D masks (already on common rows)
def two_by_two(label, C, V, rng):
    draws = boot(C, rng)
    out = []
    for m, v in V.items():
        if not np.isfinite(v[np.any([C[k] for k in "ABCD"], 0)]).all():
            continue
        pct = pd.Series(v).rank(pct=True).to_numpy()
        row = dict(truth=label, method=m, **{f"n_{k}": int(C[k].sum()) for k in "ABCD"},
                   **{f"med_pct_{k}": float(np.median(pct[C[k]])) for k in "ABCD"})
        E = np.array([2 * auc(v, C[a], C[b]) - 1 for a, b in CONTRASTS])
        EB = np.array([[2 * auc_idx(v, d[a], d[b]) - 1 for a, b in CONTRASTS] for d in draws])
        for j, (a, b) in enumerate(CONTRASTS):
            row[f"AUC_{a}{b}"] = (E[j] + 1) / 2
            row[f"AUC_{a}{b}_lo"], row[f"AUC_{a}{b}_hi"] = [(x + 1) / 2 for x in ci(EB[:, j])]
        row["I_function"], (row["I_function_lo"], row["I_function_hi"]) = E[0] - E[1], ci(EB[:, 0] - EB[:, 1])
        row["I_support"], (row["I_support_lo"], row["I_support_hi"]) = E[2] - E[3], ci(EB[:, 2] - EB[:, 3])
        row["order"] = " > ".join(sorted("ABCD", key=lambda k: -row[f"med_pct_{k}"]))
        out.append(row)
    return out


# truth-free per-target covariates + signed Pearson with the source over control cells
def covariates(genes, system="GATA1", src="GATA1"):
    D = RU.canonical_inputs(system)
    S = D["S"]
    fi = {g: i for i, g in enumerate(D["fixture_genes"])}
    X = np.asarray(S["X"], np.float64)
    gcol = X[:, [fi[g] for g in D["fixture_genes"] if g.upper() == src][0]]
    idx = np.array([fi.get(g, -1) for g in genes])
    Xg = np.where(idx[None, :] >= 0, X[:, np.maximum(idx, 0)], np.nan)
    zc = lambda a: (a - np.nanmean(a, 0)) / np.nanstd(a, 0)
    dyn, sup = [], []
    for root in RU.ROOTS:
        I, J = (np.asarray(a, int) for a in S["pairs"][root])
        dX = Xg[J] - Xg[I]
        dyn.append(np.nanstd(dX, 0) / np.nanstd(Xg, 0))
        sup.append(np.mean(dX != 0, 0))
    return pd.DataFrame({"gene": genes, "pearson_signed": np.nanmean(zc(Xg) * zc(gcol[:, None]), 0),
                         "abundance": np.log1p(np.nanmean(Xg, 0)), "detection": np.nanmean(Xg > 0, 0),
                         "dynamics": np.nanmean(dyn, 0), "pair_support": np.nanmean(sup, 0)})


COV = ["abundance", "detection", "dynamics", "pair_support"]


def matched(label, pos, neg, V, cov, rng, cols=None):
    cols = COV if cols is None else cols
    """5-NN (with replacement) on normal scores of the 4 covariates; weighted AUROC; bootstrap over positives."""
    Z = np.column_stack([norm.ppf((rankdata(cov[c]) - .5) / len(cov)) for c in cols])
    ip, ineg = np.flatnonzero(pos), np.flatnonzero(neg)
    d = ((Z[ip][:, None, :] - Z[ineg][None, :, :]) ** 2).sum(-1)
    nn = np.argsort(d, axis=1, kind="stable")[:, :KNN]                 # positive -> KNN negative positions
    smd = lambda a, b, w=None: (np.average(a, weights=None) - np.average(b, weights=w)) / np.sqrt((a.var() + b.var()) / 2)
    w_all = np.bincount(nn.ravel(), minlength=len(ineg)).astype(float)
    bal = {c: dict(before=smd(cov[c][ip], cov[c][ineg]), after=smd(cov[c][ip], cov[c][ineg], w_all)) for c in COV}
    out = []
    draws = [rng.integers(0, len(ip), len(ip)) for _ in range(NB)]
    for m, v in V.items():
        if not np.isfinite(v[ip]).all() or not np.isfinite(v[ineg]).all():
            continue
        keep = w_all > 0
        a0 = auc(v, pos, neg)
        a1 = auc_idx(v, ip, ineg[keep], w_all[keep])
        b = []
        for dr in draws:
            w = np.bincount(nn[dr].ravel(), minlength=len(ineg)).astype(float)
            k = w > 0
            b.append(auc_idx(v, ip[dr], ineg[k], w[k]))
        out.append(dict(contrast=label, method=m, n_pos=len(ip), n_neg_distinct=int(keep.sum()), AUC_unmatched=a0,
                        AUC_matched=a1, lo=ci(b)[0], hi=ci(b)[1]))
    return out, [dict(contrast=label, covariate=c, **v) for c, v in bal.items()]


def continuous(label, mask, T, V, rng):
    out = []
    sets = {"rho_any |T|": (mask, np.abs(T)), "rho_rep -T|T<=0": (mask & (T <= 0), -T), "rho_act T|T>=0": (mask & (T >= 0), T)}
    for name, (mk, y) in sets.items():
        ii = np.flatnonzero(mk)
        draws = [rng.choice(ii, len(ii)) for _ in range(NB)]
        for m, v in V.items():
            if not np.isfinite(v[ii]).all():
                continue
            b = [spearmanr(v[d], y[d])[0] for d in draws]
            out.append(dict(set=label, metric=name, method=m, n=len(ii), rho=spearmanr(v[ii], y[ii])[0], lo=ci(b)[0], hi=ci(b)[1]))
    return out


def gata1(rng):
    genes, V = methods("GATA1", "GATA1")
    cls, counts = four_classes(list(genes))
    T2 = load_verified_gata1_v2().set_index("gene").reindex(genes)
    cov = covariates(genes)
    pg = pd.DataFrame({"gene": genes}).merge(cov, on="gene")
    for m, v in V.items():
        pg[f"pct|{m}"] = pd.Series(v).rank(pct=True).to_numpy()
    # RA endpoint table
    tested = T2.KO_tested.fillna(False).to_numpy(bool)
    resp = T2.KO_resp.isin(["act", "rep"]).to_numpy()
    eps = [("KO response (chronic)", tested & resp, tested & ~resp)]
    for pol in ("repression", "activation", "any_change"):
        eps.append((f"V1 {pol} A>B", np.asarray(cls[pol]["A"]), np.asarray(cls[pol]["B"])))
    for win, tag in (("BOUND_PROM", "V2 +-1kb (primary)"), ("BOUND_10KB", "V2 +-10kb (sensitivity)")):
        c = T2[f"any_change_{win}"].fillna(".").to_numpy(str)
        eps.append((f"{tag} any A>B", c == "A", c == "B"))
    ra = []
    for name, pos, neg in eps:
        com = common_rows(pos | neg, V)
        p, n = pos & com, neg & com
        draws = boot({"p": p, "n": n}, rng)
        for m, v in V.items():
            if not np.isfinite(v[p | n]).all():
                ra.append(dict(endpoint=name, method=m, AUC=np.nan))
                continue
            b = [auc_idx(v, d["p"], d["n"]) for d in draws]
            ra.append(dict(endpoint=name, method=m, n_pos=int(p.sum()), n_neg=int(n.sum()), AUC=auc(v, p, n),
                           lo=ci(b)[0], hi=ci(b)[1]))
    # R1 2x2 (V1 rep/act; V2 prom/10kb rep/act/any)
    r1, masks = [], {}
    for pol in ("repression", "activation"):
        C = {k: np.asarray(cls[pol][k]) for k in "ABCD"}
        com = common_rows(np.any(list(C.values()), 0), V)
        C = {k: c & com for k, c in C.items()}
        masks[pol] = C
        pg[f"V1_{pol}"] = np.select([C[k] for k in "ABCD"], list("ABCD"), "")
        r1 += two_by_two(f"V1 {pol}", C, V, rng)
    for win in ("BOUND_PROM", "BOUND_10KB"):
        for pol in ("repression", "activation", "any_change"):
            c = T2[f"{pol}_{win}"].fillna(".").to_numpy(str)
            C = {k: c == k for k in "ABCD"}
            com = common_rows(np.any(list(C.values()), 0), V)
            C = {k: m_ & com for k, m_ in C.items()}
            pg[f"V2_{win}_{pol}"] = np.select([C[k] for k in "ABCD"], list("ABCD"), "")
            tag = " UNDERPOWERED_PRIMARY" if (win == "BOUND_PROM" and pol == "repression") else ""
            r1 += two_by_two(f"V2 {win} {pol}{tag}", C, V, rng)
    # R2 Delta_mode
    r2 = []
    Cr, Ca = masks["repression"], masks["activation"]
    Bm = Cr["B"] & Ca["B"]
    ib_, ir_, ia_ = (np.flatnonzero(x) for x in (Bm, Cr["A"], Ca["A"]))
    draws = [(rng.choice(ib_, len(ib_)), rng.choice(ir_, len(ir_)), rng.choice(ia_, len(ia_))) for _ in range(NB)]
    for m, v in V.items():
        if not np.isfinite(v[Bm | Cr["A"] | Ca["A"]]).all():
            continue
        ar, aa = auc(v, Cr["A"], Bm), auc(v, Ca["A"], Bm)
        b = [auc_idx(v, r_, b_) - auc_idx(v, a_, b_) for b_, r_, a_ in draws]
        r2.append(dict(method=m, AUC_rep_AB=ar, AUC_act_AB=aa, Delta_mode=ar - aa, lo=ci(b)[0], hi=ci(b)[1], n_B=len(ib_)))
    desc = []
    for pol in ("repression", "activation"):
        for k in "ABCD":
            sub = pg[pg[f"V1_{pol}"] == k]
            desc.append(dict(polarity=pol, cls=k, n=len(sub), **{c: float(sub[c].median()) for c in
                             ["pearson_signed"] + COV}, pearson_abs=float(sub.pearson_signed.abs().median())))
    # RM matched
    rm, bal = [], []
    for lab, a, b in (("V1 repression A vs B (primary)", "A", "B"), ("V1 repression C vs D", "C", "D"),
                      ("V1 activation A vs B", "A", "B")):
        C = masks["activation" if "activation" in lab else "repression"]
        o, bb = matched(lab, C[a], C[b], V, cov, rng)
        rm += o
        bal += bb
        if "activation" in lab:
            continue
        for cols in (["abundance"], ["detection"], ["dynamics"], ["pair_support"], ["abundance", "detection"]):
            o, bb = matched(f"{lab} | DECOMP match on {'+'.join(cols)}", C[a], C[b], V, cov, rng, cols)
            rm += o
            bal += bb
    # RC continuous
    T = np.array(pd.Series(dict(zip(*_ko_effect(genes)))).reindex(genes), float)
    has = np.isfinite(T)
    sup1 = np.any([masks["repression"][k] | masks["activation"][k] for k in "AB"], 0)
    b10 = T2.BOUND_10KB.fillna(False).to_numpy(bool)
    rc = []
    for lab, mk in (("all tested", has), ("V1-supported", has & sup1), ("V2-bound +-10kb", has & b10)):
        mk = mk & common_rows(mk, V)
        rc += continuous(lab, mk, np.nan_to_num(T), V, rng)
    pg["T_ko"] = T
    return (pd.DataFrame(ra), pd.DataFrame(r1), pd.DataFrame(r2), pd.DataFrame(desc), pd.DataFrame(rm),
            pd.DataFrame(bal), pd.DataFrame(rc), pg, counts)


# frozen V1 KO effect T (same computation as four_classes)
def _ko_effect(genes):
    import anndata as ad
    from gb_gata1_fourclass_canon import G, d_
    W_ = ad.read_h5ad(f"{G}/gata1_control_deterministic.h5ad")
    K_ = ad.read_h5ad(f"{G}/gata1_ko_projected_deterministic.h5ad")
    T_ = -np.log2((np.expm1(d_(K_.X)).mean(0) + 1e-6) / (np.expm1(d_(W_.X)).mean(0) + 1e-6))
    return np.asarray(W_.var_names).astype(str), np.asarray(T_).ravel()


def panc(rng):
    genes, V = methods("PANC", "RFX6")
    T = pd.read_csv(f"{R}/endpoints_v2/rfx6_occupancy.tsv", sep="\t").drop_duplicates("gene", keep=False).set_index("gene")
    k = T.ko_class.reindex(genes).fillna("UNKNOWN").to_numpy(str)
    null, act, rep = k == "TESTED_NULL", k == "ACT_RESPONDER", k == "REP_RESPONDER"
    out = []
    for name, pos, neg in (("PRIMARY any vs null", act | rep, null), ("SECONDARY act vs null", act, null),
                           ("SECONDARY rep vs null", rep, null), ("DIAGNOSTIC act vs rep", act, rep)):
        com = common_rows(pos | neg, V)
        p, n = pos & com, neg & com
        draws = boot({"p": p, "n": n}, rng)
        for m, v in V.items():
            if not np.isfinite(v[p | n]).all():
                out.append(dict(contrast=name, method=m, AUC=np.nan))
                continue
            b = [auc_idx(v, d["p"], d["n"]) for d in draws]
            out.append(dict(contrast=name, method=m, n_pos=int(p.sum()), n_neg=int(n.sum()), AUC=auc(v, p, n),
                            lo=ci(b)[0], hi=ci(b)[1]))
    cov = covariates(genes, "PANC", "RFX6")                     # same frozen matched control as GATA1
    rm, bal = [], []
    for name, pos in (("PANC any vs null (matched)", act | rep), ("PANC act vs null (matched)", act),
                      ("PANC rep vs null (matched)", rep)):
        com = common_rows(pos | null, V)
        o, bb = matched(name, pos & com, null & com, V, cov, rng)
        rm += o
        bal += bb
    return pd.DataFrame(out), pd.DataFrame(rm), pd.DataFrame(bal)


def plots(pg, V):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    sup = pg[pg.V1_repression.isin(list("AB")) | pg.V1_activation.isin(["A"])].copy()
    grp = np.where(sup.V1_repression == "A", "repression-responsive",
                   np.where(sup.V1_activation == "A", "activation-responsive", "nonresponsive"))
    fig, ax = plt.subplots(figsize=(5, 4.2))
    for g_, col in (("nonresponsive", "#bbbbbb"), ("activation-responsive", "#1f77b4"), ("repression-responsive", "#d62728")):
        s = sup[grp == g_]
        ax.scatter(s.pearson_signed, s["pct|GeoBRIDGE"], s=10 if g_ == "nonresponsive" else 18, c=col, label=f"{g_} (n={len(s)})",
                   alpha=.7, lw=0)
    ax.set_xlabel("signed Pearson(GATA1, target), control cells")
    ax.set_ylabel("GeoBRIDGE PRIMARY percentile")
    ax.legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "scatter_static_vs_gb.png", dpi=200)
    A = pg[pg.V1_repression == "A"]
    ms = [m for m in V if m not in ("RegVelo |J|", "RegVelo -J")]
    fig, ax = plt.subplots(figsize=(7, 4))
    for _, r in A.iterrows():
        ax.plot(range(len(ms)), [r[f"pct|{m}"] for m in ms], c="#d62728", alpha=.25, lw=.8)
    ax.boxplot([A[f"pct|{m}"].dropna() for m in ms], positions=range(len(ms)), widths=.4, showfliers=False)
    ax.axhline(.5, ls=":", c="k")
    ax.set_xticks(range(len(ms)))
    ax.set_xticklabels(ms, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("percentile of each V1 repression A gene")
    ax.set_title(f"all {len(A)} V1 repression A genes", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "rankshift_rep_A.png", dpi=200)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    ra, r1, r2, desc, rm, bal, rc, pg, counts = gata1(rng)
    r3, r3m, r3b = panc(rng)
    for n, t in (("RA_endpoints", ra), ("R1_twobytwo", r1), ("R2_deltamode", r2), ("R2_descriptors", desc),
                 ("RM_matched", rm), ("RM_balance", bal), ("RC_continuous", rc), ("R3_panc", r3), ("R3_panc_matched", r3m), ("R3_panc_balance", r3b), ("per_gene_GATA1", pg)):
        t.to_csv(OUT / f"{n}.tsv", sep="\t", index=False)
    plots(pg, methods("GATA1", "GATA1")[1])
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print(counts)
    print(ra.pivot_table(index="method", columns="endpoint", values="AUC").round(3).to_string())
    print(r1[["truth", "method", "n_A", "n_B", "n_C", "n_D", "AUC_AB", "AUC_CD", "AUC_AC", "AUC_BD", "AUC_AD", "order"]]
          .round(3).to_string())
    for t in (r2, desc, rm, bal, rc, r3, r3m, r3b):
        print(t.round(3).to_string())
