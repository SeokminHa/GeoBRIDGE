#!/usr/bin/env python
"""GATA1 FOUR-CLASS KO x ChIP, on the CANONICAL substrate, across every depth.

usage: gb_gata1_fourclass_canon.py <out_dir> [--kmax 100]

THE FOUR CLASSES (per polarity d, from `results/gata1_clean_2026_08_13/chipseq_labels.tsv` for binding
and the projected KO contrast T for response, exactly as `gb_endpoint_bridge.gata1_panels` builds them):

    A = ChIP-bound   & KO-responsive in direction d
    B = ChIP-bound   & KO-NOT-responsive in either direction
    C = NOT bound    & KO-responsive in direction d
    D = NOT bound    & KO-NOT-responsive in either direction

WHAT IS ALREADY KNOWN AND NOT RECOMPUTED. The HIST panel used by the canonical dense sweep
(`results/kdenseC_GATA1/`) has positives = A and universe = bound genes, so its negatives ARE class B.
**The canonical A-vs-B curve therefore already exists** and is reported in the results document §15.1a
(repression 3.244 @ k=66 / 3.346 @ k=71; activation at most 1.173/1.204). This file does NOT redo it.

WHAT IS MISSING AND IS COMPUTED HERE:

  B vs D  -- among genes the KO does NOT change, does the score rank the BOUND ones above the UNBOUND?
             This is the decisive specificity control. If B-vs-D is near chance while A-vs-B is strong,
             the score is detecting FUNCTION rather than mere occupancy, which is the claim we want.
             If B-vs-D is also strong, part of the A-vs-B signal is a binding detector, not a
             functional one, and the interpretation must be weakened.

  A vs C  -- among genes the KO DOES change, does the score rank the BOUND ones above the UNBOUND?
             Tests whether the score adds binding information on top of responsiveness.

  A vs D  -- the easy corner, reported only as a scale reference (both labels differ).

B-vs-D and A-vs-C HAVE NO POLARITY in the usual sense: B and D are both non-responders, so there is no
activation/repression direction to align a signed score to. Each is therefore scored THREE ways -- +S,
-S and |S| -- and all three are reported. Nothing is maximised over; the three are printed side by side
so that a one-sided win is visible as such.

  B vs C  -- "bound but INERT" against "responsive but UNBOUND": which AXIS does the score follow?
             C above B  => the score tracks RESPONSE, indirect/cascade targets included -- a weaker
                           claim than functional-target recovery.
             B above C  => the score tracks OCCUPANCY even where there is no function -- also weak.
             ~chance    => neither axis alone carries it, i.e. class A is special, which is the
                           STRONGEST reading and the one the method needs.
  C vs D  -- among UNBOUND genes, does the score find the responsive ones? High => the score detects
             downstream/indirect response without binding.

PRE-REGISTERED INTERPRETATION GRID (written before the numbers are read):
  STRONGEST  A_vs_B high, A_vs_C high, and B_vs_D ~ C_vs_D ~ B_vs_C ~ 1
             => the score marks the CONJUNCTION of binding and function, not either alone.
  WEAKER-1   A_vs_B high AND C_vs_D high, B_vs_C favouring C
             => the score is a KO-RESPONSE detector that happens to include indirect targets.
  WEAKER-2   B_vs_D high, A_vs_C ~ 1, B_vs_C favouring B
             => the score is an OCCUPANCY detector; the A_vs_B result would then be partly binding.

NO SELECTION. Every depth 0..kmax is written out. The frozen outcome-blind ladder is read from
`results/sv2_GATA1/k_decision_v2.json` and only marked.
"""
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


def four_classes(genes):
    """A/B/C/D masks on the CANONICAL gene universe. Same label sources as gata1_panels."""
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
    # 2026-09-20 (PI): a third polarity-blind axis. "KO change in EITHER direction" is the natural
    # positive set for the unsigned |S| endpoint, so the three licensed pairings can be read side by
    # side: activation<->+S, repression<->-S, any_change<->|S|. All three score variants are still
    # emitted for every axis; the licensed pairing is a READING rule, not a filter.
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

    # contrasts: (name, positive class, negative class)
    # ALL SIX pairwise contrasts of the 2x2, plus the reversed orientation of the ambiguous ones.
    # AP is not symmetric under swapping positives and negatives, so B_vs_C and C_vs_B are both reported.
    CONTRASTS = [("A_vs_B", "A", "B"),    # function | BOUND            <- the headline claim
                 ("B_vs_D", "B", "D"),    # binding  | NON-RESPONSIVE   <- specificity control
                 ("A_vs_C", "A", "C"),    # binding  | RESPONSIVE
                 ("C_vs_D", "C", "D"),    # response | UNBOUND          <- indirect-target detection
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
            print(f"\n=== {pol.upper()} / {sc} / dc1_min (chance = 1.0) ===")
            print(P.loc[[k for k in ks if k in P.index], cols].round(3).to_string())
    print(f"\n=== SUMMARY (dc1_min, best over depth; B_vs_D near 1.0 means the score is NOT a binding detector) ===")
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
