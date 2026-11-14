#!/usr/bin/env python
# GATA1 four-class KO x ChIP analysis on the canonical substrate, across every depth
import argparse, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/pairspace"))
from gb_substrate import load, d_
import anndata as ad
os.environ.setdefault("OMP_NUM_THREADS", "12")
from gb_kprofile_geometry import nap_tg

R = __import__("os").environ["GEOBRIDGE_ROOT"]
G = f"{R}/results/gata1_clean_2026_08_13"


# A/B/C/D masks on the canonical gene universe
def four_classes(genes):
    dfc = pd.read_csv(f"{G}/chipseq_labels.tsv", sep="\t")
    chip = dict(zip(dfc["gene"].astype(str), dfc["ChIP_bound"].astype(float)))
    W_ = ad.read_h5ad(f"{G}/gata1_control_deterministic.h5ad")
    K_ = ad.read_h5ad(f"{G}/gata1_ko_projected_deterministic.h5ad")
    T_ = -np.log2((np.expm1(d_(K_.X)).mean(0) + 1e-6) / (np.expm1(d_(W_.X)).mean(0) + 1e-6))
    tmap = dict(zip(np.asarray(W_.var_names).astype(str), T_))
    cb = np.array([chip.get(g, np.nan) for g in genes])
    ko = np.array([tmap.get(g, np.nan) for g in genes])
    bound = cb == 1.0
    unbound = cb == 0.0                      # explicit 0, never "missing label"
    has_T = np.isfinite(ko)
    act, rep = ko > 0.5, ko < -0.5
    resp_any = act | rep
    cls = {}
    for pol, pmask in (("activation", act), ("repression", rep), ("any_change", resp_any)):
        cls[pol] = dict(A=bound & pmask & has_T,
                        B=bound & has_T & ~resp_any,
                        C=unbound & pmask & has_T,
                        D=unbound & has_T & ~resp_any)
    return cls, dict(n_bound=int(bound.sum()), n_unbound=int(unbound.sum()),
                     n_has_T=int(has_T.sum()), n_resp_any=int((resp_any & has_T).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("--kmax", type=int, default=100)
    a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out_dir, exist_ok=True)
    lad = json.load(open(f"{R}/results/sv2_GATA1/k_decision_v2.json"))
    S = load("GATA1", canonical=True)
    genes, X, V = S["genes"], S["X"], S["V"]
    gu = np.char.upper(genes); ti = int(np.flatnonzero(gu == "GATA1")[0])
    cls, counts = four_classes(genes)
    print(f"GATA1 canonical {X.shape}; ladder={lad.get('ladder')} k_star={lad.get('k_star')}")
    print(f"label coverage: {counts}", flush=True)
    for pol in cls:
        print(f"  {pol}: " + "  ".join(f"{k}={int(v.sum())}" for k, v in cls[pol].items()), flush=True)

    # contrasts: (name, positive class, negative class) all six pairwise contrasts of the 2x2, plus
    # the reversed orientation of the ambiguous ones
    CONTRASTS = [("A_vs_B", "A", "B"),    # function | bound            <- the headline claim
                 ("B_vs_D", "B", "D"),    # binding  | non-responsive   <- specificity control
                 ("A_vs_C", "A", "C"),    # binding  | responsive
                 ("C_vs_D", "C", "D"),    # response | unbound          <- indirect-target detection
                 ("B_vs_C", "B", "C"),    # occupancy axis vs response axis
                 ("C_vs_B", "C", "B"),    # the same contrast, opposite orientation
                 ("A_vs_D", "A", "D")]    # both labels differ; scale reference only
    rows = []
    for root, (I, J) in S["pairs"].items():
        dX = X[J] - X[I]; dX -= dX.mean(0, keepdims=True)
        dV = V[J] - V[I]; dV -= dV.mean(0, keepdims=True)
        U_, sv, _ = np.linalg.svd(dX, full_matrices=False)
        x0 = dX[:, [ti]]
        Px = np.zeros_like(x0); Py = np.zeros_like(dV)
        print(f"  {root}: {dX.shape[0]} pairs, svd {time.time()-t0:.0f}s", flush=True)
        for k in range(0, a.kmax + 1):
            if k > 0:
                u = U_[:, [k - 1]]
                Px += u @ (u.T @ x0); Py += u @ (u.T @ dV)
            Qx, Qy = x0 - Px, dV - Py
            num = (Qx.T @ Qy)[0]; den = (np.abs(Qx).T @ np.abs(Qy))[0]
            s = num / np.maximum(den, 1e-12); s[ti] = 0.0
            supp = (den > 1e-12) & np.isfinite(s)
            for pol in cls:
                for cname, pk, nk in CONTRASTS:
                    pos_m, neg_m = cls[pol][pk], cls[pol][nk]
                    uni = (pos_m | neg_m) & supp; uni[ti] = False
                    pos = pos_m & uni
                    if uni.sum() < 20 or pos.sum() < 5: continue
                    for lab, sc in (("plusS", s), ("minusS", -s), ("absS", np.abs(s))):
                        rows.append(dict(root=root, polarity=pol, contrast=cname, score=lab, k=k,
                                         nAP=float(nap_tg(pos.astype(int), sc, uni)),
                                         n_pos=int(pos.sum()), n_uni=int(uni.sum())))
        del dX, dV, U_, Px, Py
    D = pd.DataFrame(rows)
    D.to_csv(f"{a.out_dir}/gata1_fourclass_canonical.tsv", sep="\t", index=False)
    json.dump(dict(system="GATA1", canonical=True, kmax=a.kmax, ladder=lad.get("ladder"),
                   k_star=lad.get("k_star"), counts=counts,
                   note="A_vs_B duplicates the HIST panel of results/kdenseC_GATA1 and is recomputed here "
                        "only so all four contrasts share one universe-construction code path.",
                   selection_performed=False), open(f"{a.out_dir}/provenance.json", "w"), indent=1)
    pd.set_option("display.width", 250)
    ks = [k for k in sorted(set([0, 6, 13, 20, 25, 39, 54, 66, 71, 80, 100] + (lad.get("ladder") or []))) if k <= a.kmax]
    for pol in ("repression", "activation", "any_change"):
        for sc in ("plusS", "minusS", "absS"):
            sub = D[(D.polarity == pol) & (D.score == sc) & (D.root == "dc1_min")]
            if not len(sub): continue
            P = sub.pivot_table(index="k", columns="contrast", values="nAP")
            cols = [c for c in ("A_vs_B", "B_vs_D", "A_vs_C", "C_vs_D", "B_vs_C", "C_vs_B", "A_vs_D") if c in P.columns]
            print(f"\n{pol.upper()} / {sc} / dc1_min (chance = 1.0)")
            print(P.loc[[k for k in ks if k in P.index], cols].round(3).to_string())
    print(f"\nSUMMARY (dc1_min, best over depth; B_vs_D near 1.0 means the score is not a binding detector)")
    for pol in ("repression", "activation", "any_change"):
        for cname in ("A_vs_B", "B_vs_D", "A_vs_C", "C_vs_D", "B_vs_C", "C_vs_B"):
            for sc in ("plusS", "minusS", "absS"):
                v = D[(D.polarity == pol) & (D.contrast == cname) & (D.score == sc) &
                      (D.root == "dc1_min")].set_index("k").nAP
                if not len(v): continue
                print(f"  {pol:11s} {cname:8s} {sc:7s} k0={v.get(0, np.nan):.3f} "
                      f"max={v.max():.3f}@k={int(v.idxmax())} n_pos={int(D[(D.polarity==pol)&(D.contrast==cname)].n_pos.iloc[0])}")
    print(f"\nwrote {a.out_dir} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
