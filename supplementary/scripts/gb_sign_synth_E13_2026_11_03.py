"""E13 sign-estimator comparison on TF32REP (EVALUATION ONLY; spec internal design note (not shipped)).
Reuses forward pipeline (gb_fig2_signed_2026_10_28.py) and iso pipeline (gb_fig2_iso_sign_2026_10_28.py + gb_iso_lib) verbatim; adds candidate sign rules.
Per-dataset PARITY asserts: F-mean0..8 == stored forward smean sign / sign_BA; I-k0 == stored iso sign_BA / iso_C sign.
usage: script OUT g topo cond sim"""
import sys, os, json, hashlib, numpy as np, pandas as pd
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"), (__import__("os").environ["GEOBRIDGE_ROOT"] + "/pairspace")]
import gb_tf32rep_fast_2026_10_22 as D, gb_b4_score as B, gb_iso_lib as L, gb_fig2_iso_sign_2026_10_28 as ISO
from gb_target_extension import project_scores
import anndata
OUT, g, t, c, s = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]); os.makedirs(OUT, exist_ok=True)
ROOTS = L.ROOTS; NB = 200; KMAX = 8
gd = D.gdir(g, t); tag = D.tagof(t, g, s); dd = f"{gd}/{c}/{tag}"
pre = json.load(open(f"{gd}/roster_pre.json")); assert json.load(open(f"{dd}/record.json")).get("status") == "OK"
Z = np.load(f"{dd}/raw/geobridge_raw.npz"); src = np.asarray(Z["source_gene_index"], int); cols = np.asarray(Z["cols_PI"], int)
assert list(src) == pre["cands"] and np.array_equal(cols, np.array([[x for x in range(100) if x != q] for q in pre["cands"]])), "roster"
a = anndata.read_h5ad(f"{dd}/{tag}.h5ad"); dn = lambda M: M.toarray() if hasattr(M, "toarray") else np.asarray(M)
Ms = dn(a.layers["Ms"]).astype(np.float64); Mu = dn(a.layers["Mu"]).astype(np.float64); P30 = np.asarray(a.obsm["X_pca"], float)[:, :30]; n = a.n_obs
ns, GG = len(src), Mu.shape[1]; ri = np.arange(ns)[:, None]
G = D.V_().generate_grn(g); gt = sorted((int(x), int(y)) for x, y in G["gt"]); DM = D.dist_matrix(gt)
Dm = np.array([[DM[s_, int(x)] for x in cols[i]] for i, s_ in enumerate(src)]); Y = Dm == 1; assert Y.sum() == 240
Ks = np.array([[G["K"].get((int(s_), int(x)), 0.0) for x in cols[i]] for i, s_ in enumerate(src)]); assert ((Ks != 0) == Y).all()


def diag_nan(S):                      # S (..., ns, GG): exclude self-target
    for i, s_ in enumerate(src): S[..., i, s_] = np.nan
    return S


def proj(I, J):
    """project_scores on pairs (I,J): returns num, den (9, ns, GG) float64 for k=0..8 (QR of PC30 pair differences, as forward script)."""
    dx = (Ms[J] - Ms[I])[:, src].astype(np.float32); dy = (Mu[J] - Mu[I]).astype(np.float32); Q = np.linalg.qr(P30[J] - P30[I])[0]
    N = np.full((KMAX + 1, ns, GG), np.nan); Dn = N.copy()
    for k, num, den, z2, absS in project_scores(dx, dy, Q):
        if k > KMAX: break
        N[k] = num; Dn[k] = den
    return diag_nan(N), diag_nan(Dn)


def Sk(N, Dn):
    ok = Dn > 1e-12; return np.where(ok, N / np.where(ok, Dn, 1.0), np.nan)


# ---------- forward (stored pairs)
fN, fD = [], []
for root in ROOTS:
    I, J = np.asarray(Z[f"pairs|{root}|I"], int), np.asarray(Z[f"pairs|{root}|J"], int); N_, D_ = proj(I, J); fN.append(N_); fD.append(D_)
fN, fD = np.stack(fN), np.stack(fD)                    # (2, 9, ns, GG)
FS = Sk(fN, fD)[:, :, ri, cols]                         # (2, 9, ns, 99)
assert np.isfinite(FS).all(), "forward support"
# ---------- centred forward k=0 (gb_iso_lib centred=True), as the d4 construction in the iso script
Xs = Ms[:, src].astype(np.float32).astype(np.float64); Yt = Mu.astype(np.float32)
fc = []
for root in ROOTS:
    I, J = np.asarray(Z[f"pairs|{root}|I"], int), np.asarray(Z[f"pairs|{root}|J"], int)
    S_ = L.pair_sums(Xs, Yt, I, J, centred=True); fc.append(L.ratio_C(S_["N"], S_["D"])[ri, cols])
fc = np.stack(fc)                                       # (2, ns, 99)
# ---------- iso pairs: uncentred k0 (adopted), centred k0, conditioned k<=8
pts = {r: np.asarray(Z[f"pairs|{r}|pt"], np.float64) for r in ROOTS}; strata = np.zeros(n, np.int64)
ic, per, st = ISO.iso_C(Xs, Yt, pts, strata); icm = ic[ri, cols]
ci = np.full((2, 3, ns, GG), np.nan); iN, iD = [], []
for q, r in enumerate(ROOTS):
    rN, rD = [], []
    for w, sd in enumerate(L.SEEDS):
        I, J = L.iso_pairs(pts[r], strata, L.WINDOW, L.CAP, sd, q)
        S_ = L.pair_sums(Xs, Yt, I, J, centred=True); ci[q, w] = L.ratio_C(S_["N"], S_["D"])
        N_, D_ = proj(I, J); rN.append(N_); rD.append(D_)
    iN.append(rN); iD.append(rD)
iN, iD = np.array(iN), np.array(iD)                     # (2, 3, 9, ns, GG)
IS = Sk(iN, iD)[..., ri, cols]                          # (2, 3, 9, ns, 99)
ci = ci[..., ri, cols]


def fin_mean(x, axes):                                   # NaN if any element non-finite (no silent re-weighting)
    return np.where(np.isfinite(x).all(axes), x.mean(axes), np.nan)


# ---------- candidate continuous statistics: val (ns,99) used for sign; rt (2,ns,99) per-root version for disagreement
CAND = {}
CAND["F-k0"] = (FS[:, 0].mean(0), FS[:, 0])
CAND["F-mean0..8"] = (FS.mean((0, 1)), FS.mean(1))
CAND["F-k8"] = (FS[:, 8].mean(0), FS[:, 8])
CAND["F-vote"] = (np.sign(FS).sum((0, 1)), np.sign(FS).sum(1))
CAND["F-pool"] = ((fN.sum((0, 1)) / np.where(fD.sum((0, 1)) > 1e-12, fD.sum((0, 1)), np.nan))[ri, cols],
                  (fN.sum(1) / np.where(fD.sum(1) > 1e-12, fD.sum(1), np.nan))[:, ri, cols])
CAND["F-cen"] = (np.where(np.isfinite(fc).all(0), fc.mean(0), np.nan), fc)
CAND["I-k0"] = (icm, np.stack([per[r + "_mean"][ri, cols] for r in ROOTS]))
CAND["I-cen"] = (fin_mean(ci, (0, 1)), np.stack([fin_mean(ci[q], (0,)) for q in range(2)]))
CAND["I-mean0..8"] = (fin_mean(IS, (0, 1, 2)), np.stack([fin_mean(IS[q], (0, 1)) for q in range(2)]))
A_ = Ms[:, src] - Ms[:, src].mean(0); Bm = Mu - Mu.mean(0)
pear = ((A_.T @ Bm) / (np.linalg.norm(A_, axis=0)[:, None] * np.linalg.norm(Bm, axis=0)[None, :] + 1e-300))[ri, cols]
CAND["Pearson"] = (pear, None)
SG = {k: np.sign(np.nan_to_num(v[0], nan=0.0)) for k, v in CAND.items()}
SG["AGREE"] = np.where((SG["F-mean0..8"] == SG["I-k0"]) & (SG["I-k0"] != 0), SG["I-k0"], 0.0)
# ---------- PARITY with stored results
fz = f"{D.MAIN_ROOT}/signed_2026_10_28/full/signed_g{g}_{t}_{c}_s{s}"; iz = f"{D.MAIN_ROOT}/iso_sign_2026_10_28/full/iso_g{g}_{t}_{c}_s{s}_one"
fj, ij = json.load(open(fz + ".json")), json.load(open(iz + ".json"))
assert np.array_equal(np.sign(np.load(fz + ".npz")["smean"]), SG["F-mean0..8"]), "PARITY F-mean sign"
assert np.array_equal(np.sign(np.nan_to_num(np.load(iz + ".npz")["iso_C"])), SG["I-k0"]), "PARITY I-k0 sign"


def ba(sg, ts):                                          # ts truth signs of direct edges; 0 = miss in full denominator
    return 0.5 * ((sg[ts > 0] > 0).mean() + (sg[ts < 0] < 0).mean())


ts0 = np.sign(Ks[Y]); assert abs(ba(SG["F-mean0..8"][Y], ts0) - fj["sign_BA"]) < 1e-12 and abs(ba(SG["I-k0"][Y], ts0) - ij["sign_BA"]) < 1e-12, "PARITY BA"
# ---------- sign-permutation null (same scheme/seed as existing scripts)
rng = np.random.default_rng(1000 + g); TP = []
for b in range(NB):
    Kp = Ks.copy()
    for i in range(ns):
        j = np.flatnonzero(Y[i]); Kp[i, j] = np.abs(Ks[i, j]) * rng.permutation(np.sign(Ks[i, j]))
    TP.append(np.sign(Kp[Y]))
r = dict(grn=g, topo=t, cond=c, sim=s, parity="PASS", n_act=int((ts0 > 0).sum()), n_rep=int((ts0 < 0).sum()))
for k, sg in SG.items():
    p = sg[Y]; o = ba(p, ts0); nul = np.array([ba(p, tp) for tp in TP]); call = p != 0
    r[k] = dict(BA=o, recall_act=float((p[ts0 > 0] > 0).mean()), recall_rep=float((p[ts0 < 0] < 0).mean()), coverage=float(call.mean()),
                BA_called=float(0.5 * ((p[(ts0 > 0) & call] > 0).mean() + (p[(ts0 < 0) & call] < 0).mean())) if call.any() and ((ts0 > 0) & call).any() and ((ts0 < 0) & call).any() else None,
                null_BA_mean=float(nul.mean()), null_p=float((1 + (nul >= o).sum()) / (NB + 1)))
    rt = CAND[k][1] if k in CAND else None
    r[k]["root_disagree"] = None if rt is None else float((np.sign(np.nan_to_num(rt[0])) != np.sign(np.nan_to_num(rt[1])))[Y].mean())
r["code_sha"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
np.savez_compressed(f"{OUT}/e13_g{g}_{t}_{c}_s{s}.npz", K=Ks, Y=Y, **{k.replace('.', '_').replace('-', '_'): v for k, v in SG.items()})
json.dump(r, open(f"{OUT}/e13_g{g}_{t}_{c}_s{s}.json", "w"))
print(g, t, c, s, "PARITY PASS", {k: (round(v["BA"], 3), round(v["coverage"], 2)) for k, v in r.items() if isinstance(v, dict)})
