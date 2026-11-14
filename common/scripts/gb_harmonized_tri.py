#!/usr/bin/env python
# Harmonized GATA1 / KLF1 functional-vs-occupancy lattice
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
import gb_cop_alignment as CA
import gb_fig3_mech as F3
import gb_klf1_eval as KE
import gb_primary as GP
import gb_real_uniform as RU
from gb_truth_v2_load import load_verified_gata1_v2

R = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/results")
OUT = RU.OUT / "harmonized_tri"
SPECS = {"|Pearson|": ("pearson.npz", "score", "sources"), "GENIE3": ("genie3_v2.npz", "score_all", "tfs"),
         "GRNBoost2": ("grnboost2_v2.npz", "score_all", "tfs"), "RegVelo": ("regvelo_v2_full.npz", "score_all_signed", "regulators"),
         "Velorama": ("velorama_v2_native.npz", "score_all", "regulators")}


# same construction as gb_klf1_freeze (GB primary on the observational universe + catalogue comparator rows)
def scores(system, src):
    rows, eg = RU.gb_raw_rows(system)
    eg = np.asarray(eg).astype(str)
    Rw = rows[src]
    uni, _ = CA.observational_universe(src, Rw, eg)
    genes = eg[uni]
    P = {"GeoBRIDGE": GP.zcons({h: Rw[(h, "U")][uni] for h in GP.ROOTS})}
    d = RU.system_dir(system)
    for m, (f, k, sk) in SPECS.items():
        z = np.load(d / f, allow_pickle=True)
        s = [x.upper() for x in z[sk].astype(str)]
        gi = {g: i for i, g in enumerate(z["genes"].astype(str))}
        row = np.asarray(z[k], float)[s.index(src)]
        v = np.array([row[gi[g]] if g in gi else np.nan for g in genes])
        if m == "RegVelo":
            P["RegVelo |J|"], P["RegVelo +J"], P["RegVelo -J"] = np.abs(v), v, -v
        else:
            P[m] = v
    return genes, P


def truth_gata1(genes):
    T = load_verified_gata1_v2().set_index("gene").reindex(genes)
    tested = T.KO_tested.fillna(False).to_numpy(bool)
    up, down = (T.KO_resp == "rep").to_numpy(), (T.KO_resp == "act").to_numpy()
    lab = T.BOUND_PROM.notna().to_numpy() & T.BOUND_10KB.notna().to_numpy()
    return dict(U=tested & lab, up=up, down=down, prom=(T.BOUND_PROM == 1).to_numpy(), kb10=(T.BOUND_10KB == 1).to_numpy())


def truth_klf1(genes, ko_path):
    T = KE.ko_truth(ko_path)
    Fc = pd.read_csv(f"{KE.OUT}/chip_gene_features.tsv", sep="\t", index_col=0)
    gu = np.char.upper(genes.astype(str))
    tested, lab = np.isin(gu, T.index), np.isin(gu, Fc.index)
    resp = np.array([bool(T.resp.get(g, False)) for g in gu])
    eff = np.array([float(T.ko_effect.get(g, np.nan)) for g in gu])
    b = lambda w: np.array([bool(Fc[w].get(g, False)) for g in gu])
    return dict(U=tested & lab, up=resp & (eff > 0), down=resp & (eff < 0), prom=b("BOUND_PROM"), kb10=b("BOUND_10KB"))


def scoped(V):
    gb = np.isfinite(V["GeoBRIDGE"])
    allok = np.all([np.isfinite(v) for v in V.values()], 0)
    for m, v in V.items():
        yield "pairwise", m, gb & np.isfinite(v)
        yield "all_methods", m, allok


def endpoints(tf, t, V, rng):
    U, up, down = t["U"], t["up"] & t["U"], t["down"] & t["U"]
    resp, null = up | down, t["U"] & ~(t["up"] | t["down"])
    eps = [("F1 any vs null", resp, null), ("F2 KO-up (rep-assoc) vs null", up, null), ("F3 KO-down (act-assoc) vs null", down, null),
           ("O1 bound ±1kb vs unbound", t["prom"] & U, U & ~t["prom"]), ("O2 bound ±10kb vs unbound", t["kb10"] & U, U & ~t["kb10"])]
    out, pol, lat = [], [], []
    for scope, m, ok in scoped(V):
        v = V[m]
        for name, p, n in eps:
            p_, n_ = p & ok, n & ok
            dr = F3.boot({"p": p_, "n": n_}, rng)
            b = [F3.auc_idx(v, d["p"], d["n"]) for d in dr]
            out.append(dict(tf=tf, scope=scope, method=m, endpoint=name, n_pos=int(p_.sum()), n_neg=int(n_.sum()),
                            AUROC=F3.auc(v, p_, n_), lo=F3.ci(b)[0], hi=F3.ci(b)[1]))
        u_, d_, n_ = up & ok, down & ok, null & ok        # Δ_polarity: shared null draw
        dr = F3.boot({"u": u_, "d": d_, "n": n_}, rng)
        b = [F3.auc_idx(v, d["u"], d["n"]) - F3.auc_idx(v, d["d"], d["n"]) for d in dr]
        pol.append(dict(tf=tf, scope=scope, method=m, n_up=int(u_.sum()), n_down=int(d_.sum()), n_null=int(n_.sum()),
                        AUC_up=F3.auc(v, u_, n_), AUC_down=F3.auc(v, d_, n_), delta_polarity=F3.auc(v, u_, n_) - F3.auc(v, d_, n_),
                        lo=F3.ci(b)[0], hi=F3.ci(b)[1]))
        for win, bd in (("±1kb", t["prom"]), ("±10kb", t["kb10"])):
            for lab, r in (("any", up | down), ("KO-up", up), ("KO-down", down)):
                C = {"A": ok & U & bd & r, "B": ok & bd & null, "C": ok & U & ~bd & r, "D": ok & ~bd & null}
                lat += [dict(tf=tf, window=win, scope=scope, **x) for x in F3.two_by_two(lab, C, {m: v}, rng)]
    return out, pol, lat


def panc_polarity(rng):
    genes, V = scores("PANC", "RFX6")
    T = pd.read_csv(f"{R}/endpoints_v2/rfx6_occupancy.tsv", sep="\t").drop_duplicates("gene", keep=False).set_index("gene")
    k = T.ko_class.reindex(genes).fillna("UNKNOWN").to_numpy(str)
    t = dict(U=np.isin(k, ["TESTED_NULL", "ACT_RESPONDER", "REP_RESPONDER"]), up=k == "REP_RESPONDER", down=k == "ACT_RESPONDER")
    pol = []
    for scope, m, ok in scoped(V):
        v = V[m]
        u_, d_, n_ = t["up"] & ok, t["down"] & ok, (k == "TESTED_NULL") & ok
        dr = F3.boot({"u": u_, "d": d_, "n": n_}, rng)
        b = [F3.auc_idx(v, d["u"], d["n"]) - F3.auc_idx(v, d["d"], d["n"]) for d in dr]
        pol.append(dict(tf="RFX6 (PANC)", scope=scope, method=m, n_up=int(u_.sum()), n_down=int(d_.sum()), n_null=int(n_.sum()),
                        AUC_up=F3.auc(v, u_, n_), AUC_down=F3.auc(v, d_, n_), delta_polarity=F3.auc(v, u_, n_) - F3.auc(v, d_, n_),
                        lo=F3.ci(b)[0], hi=F3.ci(b)[1]))
    return pol


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20260930)
    E, P, L = [], [], []
    g1, V1 = scores("GATA1", "GATA1")
    e, p, l = endpoints("GATA1", truth_gata1(g1), V1, rng)
    E += e; P += p; L += l
    gk, mk, Sk, _ = KE.load_pred()                        # frozen KLF1 predictions (sha-checked)
    Vk = dict(zip(mk, Sk))
    e, p, l = endpoints("KLF1", truth_klf1(gk, f"{KE.OUT}/truth/GSE94351_gene_exp.diff.gz"), Vk, rng)
    E += e; P += p; L += l
    P += panc_polarity(rng)
    E, P, L = pd.DataFrame(E), pd.DataFrame(P), pd.DataFrame(L)
    E.to_csv(OUT / "endpoints.tsv", sep="\t", index=False)
    P.to_csv(OUT / "polarity_DESCRIPTIVE.tsv", sep="\t", index=False)
    L.to_csv(OUT / "fourclass.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_rows", 500)
    print(E.round(3).to_string(index=False))
    print(P.round(3).to_string(index=False))
    print(L[["tf", "window", "scope", "truth", "method", "n_A", "n_B", "n_C", "n_D", "AUC_AB", "AUC_CD", "AUC_AC", "AUC_BD", "AUC_AD",
             "I_support", "I_support_lo", "I_support_hi", "order"]].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
