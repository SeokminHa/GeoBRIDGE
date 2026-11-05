"""E12 iso-window sensitivity on TF32REP (2026-11-01). Reuses gb_fig2_iso_sign_2026_10_28 pipeline verbatim (loading, one stratum, truth=simulator K sign on direct edges,
sign_acc / sign_BA with NaN/0 = miss in full direct-edge denominator). Only the iso window varies; window 0.02 is asserted equal to the stored iso_sign result.
usage: script OUT g topo cond sim"""
import sys, os, json, numpy as np
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"), (__import__("os").environ["GEOBRIDGE_ROOT"] + "/pairspace")]
import gb_iso_lib as L
ROOTS = L.ROOTS; WINS = [0.005, 0.01, 0.02, 0.04, 0.08]

def iso_C_w(Xs, Yt, pts, strata, w):
    per, npairs = {}, []
    for ri, r in enumerate(ROOTS):
        cs = []
        for sd in L.SEEDS:
            I, J = L.iso_pairs(pts[r], strata, w, L.CAP, sd, ri); S = L.pair_sums(Xs, Yt, I, J, centred=False); cs.append(L.ratio_C(S['N'], S['D'])); npairs.append(len(I))
        cs = np.stack(cs); per[r] = np.where(np.isfinite(cs).all(0), cs.mean(0), np.nan)
    a, b = per[ROOTS[0]], per[ROOTS[1]]
    return np.where(np.isfinite(a) & np.isfinite(b), (a + b) / 2, np.nan), float(np.mean(npairs))

def main(OUT, g, t, c, s):
    import gb_tf32rep_fast_2026_10_22 as D, anndata
    os.makedirs(OUT, exist_ok=True); gd = D.gdir(g, t); tag = D.tagof(t, g, s); dd = f"{gd}/{c}/{tag}"
    pre = json.load(open(f"{gd}/roster_pre.json")); assert json.load(open(f"{dd}/record.json")).get("status") == "OK"
    Z = np.load(f"{dd}/raw/geobridge_raw.npz"); src = np.asarray(Z["source_gene_index"], int); cols = np.asarray(Z["cols_PI"], int)
    assert list(src) == pre["cands"] and np.array_equal(cols, np.array([[x for x in range(100) if x != q] for q in pre["cands"]]))
    a = anndata.read_h5ad(f"{dd}/{tag}.h5ad"); n = a.n_obs; dn = lambda M: M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    Xs = dn(a.layers["Ms"])[:, src].astype(np.float32).astype(np.float64); Yt = dn(a.layers["Mu"]).astype(np.float32)
    pts = {r: np.asarray(Z[f"pairs|{r}|pt"], np.float64) for r in ROOTS}; strata = np.zeros(n, np.int64)
    G = D.V_().generate_grn(g); gt = sorted((int(x), int(y)) for x, y in G["gt"]); DM = D.dist_matrix(gt)
    Dm = np.array([[DM[s_, int(x)] for x in cols[i]] for i, s_ in enumerate(src)]); Y = Dm == 1; assert Y.sum() == 240
    Ks = np.array([[G["K"].get((int(s_), int(x)), 0.0) for x in cols[i]] for i, s_ in enumerate(src)]); assert ((Ks != 0) == Y).all()
    ts = np.sign(Ks[Y]); ri = np.arange(len(src))[:, None]
    def mets(icm):
        sg = np.sign(np.nan_to_num(icm, nan=0.0)); ps = sg[Y]; ra = float((ps[ts > 0] > 0).mean()); rr = float((ps[ts < 0] < 0).mean()); called = ps != 0
        return dict(sign_acc=float((ps == ts).mean()), sign_BA=.5 * (ra + rr), coverage=float(called.mean()), acc_called=float((ps[called] == ts[called]).mean()) if called.any() else np.nan,
                    n_nocall_direct=int((~called).sum()), frac_pos_sign_direct=float((ps > 0).mean()))
    R = dict(grn=g, topo=t, cond=c, sim=s, rows=[]); ic02 = None
    for w in WINS:
        ic, npr = iso_C_w(Xs, Yt, pts, strata, w); icm = ic[ri, cols]
        R["rows"].append(dict(ref="iso", window=w, n_pairs_per_root_seed=npr, **mets(icm)))
        if w == 0.02: ic02 = icm
    # forward-pair k=0 reference: stored forward pairs (0.02 < dpt <= 0.25), root mean of ratio_C; uncentred and centred (d4)
    for cen in (False, True):
        d, npf = [], []
        for r_ in ROOTS:
            I_, J_ = np.asarray(Z[f"pairs|{r_}|I"], int), np.asarray(Z[f"pairs|{r_}|J"], int); dp = pts[r_][J_] - pts[r_][I_]; npf.append(len(I_))
            S_ = L.pair_sums(Xs, Yt, I_, J_, centred=cen); d.append(L.ratio_C(S_["N"], S_["D"])[ri, cols])
        d = np.where(np.isfinite(d[0]) & np.isfinite(d[1]), (d[0] + d[1]) / 2, np.nan)
        R["rows"].append(dict(ref="fwd_k0_" + ("centred" if cen else "uncentred"), window=np.nan, n_pairs_per_root_seed=float(np.mean(npf)), dpt_min=float(dp.min()), dpt_max=float(dp.max()), **mets(d)))
    # parity vs stored iso_sign result at 0.02
    st = f"{D.MAIN_ROOT}/iso_sign_2026_10_28/full/iso_g{g}_{t}_{c}_s{s}_one"; sj = json.load(open(st + ".json")); sz = np.load(st + ".npz", allow_pickle=True)["iso_C"]
    m = mets(ic02); both = np.isfinite(ic02) & np.isfinite(sz)
    par = dict(finite_equal=bool(np.array_equal(np.isfinite(ic02), np.isfinite(sz))), max_abs=float(np.abs(ic02[both] - sz[both]).max()),
               BA_diff=abs(m["sign_BA"] - sj["sign_BA"]), acc_diff=abs(m["sign_acc"] - sj["sign_acc"]), nocall_eq=m["n_nocall_direct"] == sj["n_nocall_direct"])
    assert par["finite_equal"] and par["max_abs"] <= 1e-9 and par["BA_diff"] < 1e-12 and par["acc_diff"] < 1e-12 and par["nocall_eq"], f"PARITY FAIL {par}"
    R["parity"] = par; json.dump(R, open(f"{OUT}/e12_g{g}_{t}_{c}_s{s}.json", "w")); print("PARITY PASS", g, t, c, s, par, flush=True)

if __name__ == "__main__": main(sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]))
