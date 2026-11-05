"""E16 mechanism diagnostic: direct-edge BA vs TOTAL-EFFECT BA (indirect d in {2,3}, unambiguous path-sign) + excitation retained.
Reuses E13 verbatim by import (runs its score/candidate computation and ALL its parity asserts). EVALUATION ONLY.
usage: script OUT g topo cond sim"""
import sys, os, json, importlib.util, numpy as np
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
sp = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch/gb_sign_synth_E13_2026_11_03.py")
sys.argv[1] = OUT + "/_e13rerun"            # E13 writes its own outputs here (not touching stored E13 results)
spec = importlib.util.spec_from_file_location("e13", sp); E = importlib.util.module_from_spec(spec); spec.loader.exec_module(E)
g, t, c, s = E.g, E.t, E.c, E.s; src, cols, K = E.src, E.cols, E.G["K"]; ns = E.ns; SG = E.SG; Y = E.Y
# ---- parity vs STORED E13 json (full run)
st = json.load(open(f"{__import__('os').environ['GEOBRIDGE_ROOT']}/results/signed/r2fix_2026_11_01/E13_sign_synth/full/e13_g{g}_{t}_{c}_s{s}.json"))
ts0 = np.sign(E.Ks[Y]); pdiff = {}
for k, sg in SG.items():
    pdiff[k] = abs(E.ba(sg[Y], ts0) - st[k]["BA"]); assert pdiff[k] < 1e-12, ("PARITY stored E13", k, pdiff[k])
# ---- path-sign labels: simple directed paths len<=3 src->target over nonzero K edges
adj = {}
for (a_, b_), v in K.items():
    if v != 0 and a_ != b_: adj.setdefault(int(a_), []).append((int(b_), np.sign(v)))
def paths(u, maxl=3):
    out = []                                           # (target, length, sign)
    def rec(node, seen, l, sg_):
        if l >= maxl: return
        for v, sv in adj.get(node, []):
            if v in seen: continue
            out.append((v, l + 1, sg_ * sv)); rec(v, seen | {v}, l + 1, sg_ * sv)
    rec(u, {u}, 0, 1.0); return out
tot = np.zeros((ns, 99)); grp = np.full((ns, 99), "none", dtype=object); nd = np.full((ns, 99), 0)
for i, u in enumerate(src):
    P = paths(int(u))
    for j, x in enumerate(cols[i]):
        pp = [(l, sg_) for v, l, sg_ in P if v == x]
        if not pp: continue
        sgs = {sg_ for _, sg_ in pp}; dmin = min(l for l, _ in pp); direct = any(l == 1 for l, _ in pp)
        if len(sgs) > 1: grp[i, j] = "direct_disagree" if direct else "indirect_ambig"
        else:
            tot[i, j] = sgs.pop(); grp[i, j] = "direct_agree" if direct else f"indirect_d{dmin}"
        nd[i, j] = dmin
assert (np.where(Y, nd == 1, True)).all()
masks = {"direct_all": Y, "direct_agree": grp == "direct_agree", "direct_disagree": grp == "direct_disagree",
         "indirect_d2": grp == "indirect_d2", "indirect_d3": grp == "indirect_d3"}
masks["indirect_d23"] = masks["indirect_d2"] | masks["indirect_d3"]
cnt = {m: int(v.sum()) for m, v in masks.items()}; cnt["indirect_ambig_excluded"] = int((grp == "indirect_ambig").sum())
def ba2(p, ts):
    a, r_ = ts > 0, ts < 0
    return float(0.5 * ((p[a] > 0).mean() + (p[r_] < 0).mean())) if a.any() and r_.any() else np.nan
rows = []
for k, sg in SG.items():
    for m, mk in masks.items():
        if m == "direct_all": lab = np.sign(E.Ks)       # direct edge sign
        elif m == "direct_agree" or m == "direct_disagree": lab = np.sign(E.Ks)  # direct edge's own sign; total==direct when agree
        else: lab = tot
        ts = lab[mk]; p = sg[mk]
        rows.append(dict(grn=g, topo=t, cond=c, sim=s, cand=k, subset=m, n=int(mk.sum()), n_act=int((ts > 0).sum()), n_rep=int((ts < 0).sum()), BA=ba2(p, ts),
                         BA_vs_total=ba2(p, tot[mk]) if m.startswith("direct") else np.nan, coverage=float((p != 0).mean()) if mk.any() else np.nan))
# direct_all BA must equal E13
for r_ in rows:
    if r_["subset"] == "direct_all": assert abs(r_["BA"] - st[r_["cand"]]["BA"]) < 1e-12
# ---- excitation retained
exc = np.zeros((2, E.KMAX + 1, ns))
for q, root in enumerate(E.ROOTS):
    I, J = np.asarray(E.Z[f"pairs|{root}|I"], int), np.asarray(E.Z[f"pairs|{root}|J"], int)
    X = (E.Ms[J] - E.Ms[I])[:, src]; Q = np.linalg.qr(E.P30[J] - E.P30[I])[0]; x2 = (X ** 2).sum(0)
    for k in range(E.KMAX + 1):
        R_ = X - Q[:, :k] @ (Q[:, :k].T @ X); exc[q, k] = (R_ ** 2).sum(0) / x2
exc = exc.mean(0)                                       # (9, ns)
import pandas as pd
pd.DataFrame(rows).to_csv(f"{OUT}/e16_g{g}_{t}_{c}_s{s}.tsv", sep="\t", index=False)
json.dump(dict(grn=g, topo=t, cond=c, sim=s, counts=cnt, parity_max_absdiff=max(pdiff.values()), exc_median=np.median(exc, 1).tolist(),
               exc_min=exc.min(1).tolist(), exc_per_reg_k8=exc[8].tolist()), open(f"{OUT}/e16_g{g}_{t}_{c}_s{s}.json", "w"))
print(g, t, c, s, "PARITY OK max", max(pdiff.values()), cnt, "exc_med", np.round(np.median(exc, 1), 3))
