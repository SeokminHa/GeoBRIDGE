# Fig 2 SI ablation
import sys, os, json, numpy as np, pandas as pd
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"), (__import__("os").environ["GEOBRIDGE_ROOT"] + "/pairspace")]
import gb_tf32rep as D
import gb_b4_score as B
from gb_target_extension import project_scores
from sklearn.metrics import average_precision_score as APs
import anndata
OUT, g, t, c, s = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]); os.makedirs(OUT, exist_ok=True)
KMAX = 30; ROOTS = B.GP.ROOTS; OPS = ("R1", "R2s", "R2h"); CHS = ("U", "V", "X")
gd = D.gdir(g, t); tag = D.tagof(t, g, s); dd = f"{gd}/{c}/{tag}"
pre = json.load(open(f"{gd}/roster_pre.json")); Z = np.load(f"{dd}/raw/geobridge_raw.npz")
src = np.asarray(Z["source_gene_index"], int); cols = np.asarray(Z["cols_PI"], int); assert list(src) == pre["cands"]
a = anndata.read_h5ad(f"{dd}/{tag}.h5ad")
dn = lambda M: M.toarray() if hasattr(M, "toarray") else np.asarray(M)
Ms = dn(a.layers["Ms"]).astype(np.float64); Mu = dn(a.layers["Mu"]).astype(np.float64); Vr = dn(a.layers["velocity"]).astype(np.float64)
vinfo = dict(V_nonfinite=int((~np.isfinite(Vr)).sum()), V_nonfinite_genes=int((~np.isfinite(Vr)).any(0).sum()), V_constant_genes=int((np.nanstd(Vr, 0) == 0).sum()))
V0 = np.nan_to_num(Vr)                                                   # same replacement as the producer (gb_tf32rep_reference:1027); counts recorded above
P30 = np.asarray(a.obsm["X_pca"], float)[:, :30]
G = D.V_().generate_grn(g); gt = sorted((int(x), int(y)) for x, y in G["gt"]); DM = D.dist_matrix(gt)
Dm = np.array([[DM[s_, int(x)] for x in cols[i]] for i, s_ in enumerate(src)]); Y = Dm == 1; assert Y.sum() == 240
raw = {}; parity = {}; spec = {}
for root in ROOTS:
    I, J = np.asarray(Z[f"pairs|{root}|I"], int), np.asarray(Z[f"pairs|{root}|J"], int)
    dx = (Ms[J] - Ms[I])[:, src].astype(np.float32)
    dys = {"U": Mu[J] - Mu[I], "V": V0[J] - V0[I], "X": Ms[J] - Ms[I]}
    Q1 = np.linalg.qr(P30[J] - P30[I])[0]
    sd = Ms.std(0); gref = sd > 0; Dp = (Ms[J][:, gref] - Ms[I][:, gref]) / sd[gref]; Dp = Dp - Dp.mean(0)
    U2s, S2s, _ = np.linalg.svd(Dp, full_matrices=False)
    DXall = Ms[J] - Ms[I]; DXc = DXall - DXall.mean(0)
    U2h, S2h, _ = np.linalg.svd(DXc, full_matrices=False)
    spec[root] = dict(rank_R1=int(Q1.shape[1]), sv_R2s=S2s[:KMAX + 1].tolist(), sv_R2h=S2h[:KMAX + 1].tolist())
    dxc = (DXc[:, src]).astype(np.float32)
    for op, Q, dxo in (("R1", Q1, dx), ("R2s", U2s[:, :KMAX], dx), ("R2h", U2h[:, :KMAX], dxc)):
        assert Q.shape[1] >= KMAX, (op, Q.shape)
        for ch, dy in dys.items():
            A = np.full((KMAX + 1, len(src), dy.shape[1]), np.nan, np.float32)
            for k, num, den, z2, absS in project_scores(dxo, dy.astype(np.float32), Q): A[k] = absS
            for i, s_ in enumerate(src): A[:, i, s_] = np.nan
            raw[(op, ch, root)] = A
    # independent numerical check : direct historical dual_project_pairspace + score_matrix
    # (float64) vs the project_scores path
    from geobridge_model.scoring import dual_project_pairspace, score_matrix
    U_, S_, Vt_ = np.linalg.svd(DXc, full_matrices=False); chk = {}
    for k in (0, 4, 8, 30):
        for ch in ("U", "V"):
            Xr, Vr_ = dual_project_pairspace(DXc, dys[ch], U_, S_, Vt_, k); Sd = np.abs(score_matrix(Xr, Vr_)[src])
            Rk = raw[("R2h", ch, root)][k]; m = np.isfinite(Rk)
            chk[f"R2h|{ch}|k{k}"] = float(np.max(np.abs(Rk[m] - Sd[m]))); chk[f"R2h|{ch}|k{k}|nan_offdiag"] = int((~m).sum() - len(src))
        Q2 = U2s[:, :k]; Xs = dx.astype(np.float64) - Q2 @ (Q2.T @ dx.astype(np.float64)); Ys = dys["U"] - Q2 @ (Q2.T @ dys["U"])
        Sd = np.abs((Xs.T @ Ys) / np.maximum(np.abs(Xs).T @ np.abs(Ys), 1e-12)); Rk = raw[("R2s", "U", root)][k]; m = np.isfinite(Rk)
        chk[f"R2s|U|k{k}"] = float(np.max(np.abs(Rk[m] - Sd[m])))
    spec[root]["direct_check_maxabs"] = chk
    st = np.asarray(Z[f"TRUE|U|{root}|absS"], np.float32); new = raw[("R1", "U", root)]
    assert new.shape == st.shape and np.array_equal(np.isnan(new), np.isnan(st)), f"PARITY FAIL support {root}"
    parity[root] = float(np.nanmax(np.abs(new - st)))
assert max(parity.values()) <= 1e-6, f"PARITY FAIL vs stored PRIMARY absS {parity}"   # absS stored as float32 (ulp ~6e-8); multithreaded BLAS order gives <=4e-9; value recorded in JSON; metric parity below stays strict
for root in ROOTS:                                                       # k=0: R1 and R2s share the uncentred source -> identical
    assert np.array_equal(raw[("R1", "U", root)][0], raw[("R2s", "U", root)][0], equal_nan=True), "k0 R1 != R2s"
def score(op, ch, ks): return B.zc([{r: raw[(op, ch, r)][:, i, cols[i]].T for r in ROOTS} for i in range(len(src))], ks)
rows = []
for op in OPS:
    for ch in CHS:
        for kind, K in [("window", K) for K in range(KMAX + 1)] + [("single", K) for K in range(KMAX + 1)]:
            ks = range(0, K + 1) if kind == "window" else [K]
            sc = score(op, ch, ks); r = D.metrics(sc, Y, Dm); ok = np.isfinite(sc)
            macro = np.mean([APs(Y[i][ok[i]], sc[i][ok[i]]) for i in range(len(src)) if Y[i][ok[i]].any()])
            rows.append(dict(grn=g, topo=t, cond=c, sim=s, op=op, ch=ch, kind=kind, K=K, coverage=r["coverage"], n_pos=int((Y & ok).sum()),
                             macroAP=macro, **{k_: r[k_] for k_ in ("AP", "AUROC", "DvI2", "DvI3")}))
R = pd.DataFrame(rows)
R.to_csv(f"{OUT}/abl_g{g}_{t}_{c}_s{s}.tsv", sep="\t", index=False)          # written before the gates so a failure leaves a receipt
assert np.array_equal(cols, np.array([[x for x in range(100) if x != q] for q in pre["cands"]])), "cols_PI universe mismatch"
assert (R.coverage == 1.0).all() and (R.n_pos == 240).all(), "coverage < 1 for some arm (reported, run halted)"
np.savez_compressed(f"{OUT}/scores_g{g}_{t}_{c}_s{s}.npz", src=src, cols=cols, Dm=Dm,
                    **{f"{op}|{ch}": score(op, ch, range(0, 9)) for op in OPS for ch in CHS})   # window 0..8 score matrices for the label-permutation control
prim = R[(R.op == "R1") & (R.ch == "U") & (R.kind == "window") & (R.K == 8)].iloc[0]
ref = pd.read_csv(f"{D.MAIN_ROOT}/TABLE.tsv", sep="\t").query("grn==@g and topo==@t and cond==@c and sim==@s and method=='GB'").iloc[0]
assert abs(prim.AP - ref.AP) < 1e-9, ("PRIMARY metric parity FAIL", prim.AP, ref.AP)
assert abs(prim.DvI2 - ref.DvI2) < 1e-9
R.to_csv(f"{OUT}/abl_g{g}_{t}_{c}_s{s}.tsv", sep="\t", index=False)
json.dump(dict(parity_absS=parity, primary_AP=float(prim.AP), table_AP=float(ref.AP), velocity=vinfo, spectra=spec), open(f"{OUT}/abl_g{g}_{t}_{c}_s{s}.json", "w"))
print("OK", g, t, c, s, "parity", parity, "AP", round(prim.AP, 4), vinfo)
