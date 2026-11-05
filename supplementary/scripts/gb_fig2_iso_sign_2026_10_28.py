"""Fig 2 ISO-SIGN benchmark (2026-10-28). Transfers the REAL-DATA Fig 4 signed rule to the TF32REP simulation, unchanged (internal review consensus,
internal design note (not shipped)). EVALUATION ONLY.
Estimator (verbatim gb_iso_pilot.py:100-101,142,158-163 via gb_iso_lib): per root r in (dc1_min, dc1_max) with root index (0,1), for seeds (42,43,44):
  iso pairs = L.iso_pairs(pt_r, strata, 0.02, 40, seed, ri); N, D = L.pair_sums(Xs(Ms source, float32->float64), Yt(Mu target, float32), centred=False);
  C = L.ratio_C(N, D); iso_r = seed-mean if all 3 finite else NaN; iso_C = root mean if both finite else NaN; sign = sign(iso_C) (0/NaN = no call = miss).
Magnitude: frozen PRIMARY percentile (= |Z|d4| of Fig 4 when d4 finite & nonzero); T_iso = sign(iso_C) x pct(PRIMARY).
Sim inputs: pt = stored per-cell raw/geobridge_raw.npz pairs|<root>|pt (observational DPT); strata = one stratum (PRIMARY); 'bin' strata = labelled sensitivity.
Feasibility (frozen, gb_iso_pilot.py:241-248): >=100 participating cells per root/seed and >=80% finite-sign coverage per source, else NO_CALL (reported).
modes:  parity SYSTEM FOCAL   recompute iso_C for a real unit from the frozen inputs and compare to the stored features_<unit>.tsv
        sim OUT g topo cond sim [--strata one|bin]"""
import sys, os, json, hashlib, numpy as np, pandas as pd
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"), (__import__("os").environ["GEOBRIDGE_ROOT"] + "/pairspace")]
import gb_iso_lib as L
from scipy.stats import rankdata
ROOTS = L.ROOTS


def iso_C(Xs, Yt, pts, strata):
    """Xs (n, ns) float64, Yt (n, G) float32, pts {root: (n,) pt}. Returns iso_C (ns, G), per-root/seed arrays, pair stats."""
    per, stats = {}, {}
    for ri, r in enumerate(ROOTS):
        cs = []
        for sd in L.SEEDS:
            I, J = L.iso_pairs(pts[r], strata, L.WINDOW, L.CAP, sd, ri); S = L.pair_sums(Xs, Yt, I, J, centred=False)
            cs.append(L.ratio_C(S['N'], S['D']))
            stats[f'{r}|{sd}'] = dict(n_pairs=int(len(I)), n_cells=int(len(np.unique(np.r_[I, J]))), mean_abs_dpt=float(np.abs(pts[r][J] - pts[r][I]).mean()),
                                      sha=L.sha_np(I, J)[:16])
        cs = np.stack(cs); per[r] = cs; per[r + '_mean'] = np.where(np.isfinite(cs).all(0), cs.mean(0), np.nan)
    a, b = per[ROOTS[0] + '_mean'], per[ROOTS[1] + '_mean']
    return np.where(np.isfinite(a) & np.isfinite(b), (a + b) / 2, np.nan), per, stats


def parity(system, focal):
    import gb_iso_pilot as IP, anndata as ad
    FZ, _ = IP.load_freeze(); cfg = FZ['systems'][system]
    zp = np.load(cfg['pairs_npz'], allow_pickle=True); F = ad.read_h5ad(cfg['fixture'])
    pts = {r: np.asarray(zp[f'{r}_pt'], np.float64) for r in ROOTS}; strata = IP.strata_codes(F.obs, cfg['strata_columns_used'])
    ft = pd.read_csv(IP.OUT + f'scores/{system}_{focal}/features_{system}_{focal}.tsv', sep='\t')
    vn = np.char.upper(np.asarray(F.var_names).astype(str)); fi = {}
    for j, g in enumerate(vn): fi.setdefault(g, j)
    genes = ft.gene.astype(str).str.upper().to_numpy(); keep = np.array([g in fi for g in genes]); cols = np.array([fi[g] for g in genes[keep]])
    Xs = IP.dense(F.layers['Ms'][:, [fi[focal.upper()]]], np.float32).astype(np.float64); Yt = IP.dense(F.layers['Mu'][:, cols], np.float32)
    ic, _, st = iso_C(Xs, Yt, pts, strata); new = ic[0]; old = ft.iso_C.to_numpy(float)[keep]
    man = json.load(open(IP.OUT + 'pairs/' + system + '/pairs_manifest.json'))
    sha_ok = {k: man.get(k.replace('|', '|iso'), {}).get('sha256', '')[:16] == v['sha'] for k, v in st.items()}
    fin_eq = np.array_equal(np.isfinite(new), np.isfinite(old)); both = np.isfinite(new) & np.isfinite(old)
    out = dict(unit=f'{system}_{focal}', n_genes=int(keep.sum()), finite_mask_equal=fin_eq, max_abs_diff=float(np.abs(new[both] - old[both]).max()),
               sign_mismatch=int((np.sign(new[both]) != np.sign(old[both])).sum()), pair_sha_match=sha_ok, strata_n=int(len(np.unique(strata))))
    d4 = []   # d4 parity: root mean of centred forward-pair ratio_C vs stored fwd_d4f (gb_iso_pilot.py:139)
    for r in ROOTS:
        S_ = L.pair_sums(Xs, Yt, zp[f'{r}_I'].astype(np.int64), zp[f'{r}_J'].astype(np.int64), centred=True); d4.append(L.ratio_C(S_['N'], S_['D'])[0])
    d4n = np.where(np.isfinite(d4[0]) & np.isfinite(d4[1]), (d4[0] + d4[1]) / 2, np.nan); d4o = ft.fwd_d4f.to_numpy(float)[keep]; b2 = np.isfinite(d4n) & np.isfinite(d4o)
    out.update(d4_finite_equal=bool(np.array_equal(np.isfinite(d4n), np.isfinite(d4o))), d4_max_abs_diff=float(np.abs(d4n[b2] - d4o[b2]).max()), d4_sign_mismatch=int((np.sign(d4n[b2]) != np.sign(d4o[b2])).sum()))
    print(json.dumps(out)); assert out['d4_finite_equal'] and out['d4_max_abs_diff'] <= 1e-6 and out['d4_sign_mismatch'] == 0, 'd4 PARITY FAIL'
    assert fin_eq and out['max_abs_diff'] <= 1e-6 and out['sign_mismatch'] == 0 and all(sha_ok.values()), 'PARITY FAIL'
    print('PARITY PASS')


def sim(OUT, g, t, c, s, strata_mode):
    import gb_tf32rep_fast_2026_10_22 as D, gb_b4_score as B, anndata
    from sklearn.metrics import roc_auc_score as AUC, average_precision_score as APs
    os.makedirs(OUT, exist_ok=True); NB = int(os.environ.get("ISO_NB", "2000"))
    gd = D.gdir(g, t); tag = D.tagof(t, g, s); dd = f"{gd}/{c}/{tag}"
    pre = json.load(open(f"{gd}/roster_pre.json")); assert json.load(open(f"{dd}/record.json")).get("status") == "OK"
    Z = np.load(f"{dd}/raw/geobridge_raw.npz"); src = np.asarray(Z["source_gene_index"], int); cols = np.asarray(Z["cols_PI"], int)
    assert list(src) == pre["cands"] and np.array_equal(cols, np.array([[x for x in range(100) if x != q] for q in pre["cands"]]))
    a = anndata.read_h5ad(f"{dd}/{tag}.h5ad"); n = a.n_obs; dn = lambda M: M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    Xs = dn(a.layers["Ms"])[:, src].astype(np.float32).astype(np.float64); Yt = dn(a.layers["Mu"]).astype(np.float32)
    pts = {}
    for r in ROOTS:
        pt = np.asarray(Z[f"pairs|{r}|pt"], np.float64); I, J = np.asarray(Z[f"pairs|{r}|I"], int), np.asarray(Z[f"pairs|{r}|J"], int)
        assert pt.shape == (n,) and np.isfinite(pt).all() and (pt[J] >= pt[I]).all(), f"pt alignment {r}"; pts[r] = pt
    strata = np.zeros(n, np.int64) if strata_mode == "one" else pd.factorize(a.obs["cell_type"].astype(str), sort=True)[0].astype(np.int64)
    ic, per, st = iso_C(Xs, Yt, pts, strata)
    ri = np.arange(len(src))[:, None]; icm = ic[ri, cols]                      # (ns, 99) targets in cols order
    sg = np.sign(np.nan_to_num(icm, nan=0.0))
    feas = dict(min_cells=min(v["n_cells"] for v in st.values()), coverage_by_source=(np.isfinite(icm) & (icm != 0)).mean(1).round(3).tolist())
    NO_CALL = feas["min_cells"] < 100 or min(feas["coverage_by_source"]) < .80
    G = D.V_().generate_grn(g); gt = sorted((int(x), int(y)) for x, y in G["gt"]); DM = D.dist_matrix(gt)
    Dm = np.array([[DM[s_, int(x)] for x in cols[i]] for i, s_ in enumerate(src)]); Y = Dm == 1; assert Y.sum() == 240
    Ks = np.array([[G["K"].get((int(s_), int(x)), 0.0) for x in cols[i]] for i, s_ in enumerate(src)]); assert ((Ks != 0) == Y).all()
    Kst = {(int(e["regulator"]), int(e["target"])): float(e["K"]) for e in np.load(f"{dd}/raw/geometry_raw.npz")["edges"]}
    assert all(k in Kst and np.sign(Kst[k]) == np.sign(v) for k, v in G["K"].items()), "K sign vs stored edges"
    prim = B.zc([{r: np.asarray(Z[f"TRUE|U|{r}|absS"], float)[:, i, cols[i]].T for r in B.GP.ROOTS} for i in range(len(src))], range(0, 9)); assert np.isfinite(prim).all()
    ref = pd.read_csv(f"{D.MAIN_ROOT}/TABLE.tsv", sep="\t").query("grn==@g and topo==@t and cond==@c and sim==@s and method=='GB'"); assert len(ref) == 1
    assert abs(D.metrics(prim, Y, Dm)["AP"] - ref.AP.iloc[0]) < 1e-9, "TABLE AP parity"
    pct = np.vstack([(rankdata(r) - 0.5) / len(r) for r in prim])
    d4 = []   # Fig 4 literal magnitude: Z_d4 = sign(d4) x pct; d4 = root mean of CENTRED forward-pair ratio_C (float32 storage, as stage-1)
    for r_ in ROOTS:
        I_, J_ = np.asarray(Z[f"pairs|{r_}|I"], int), np.asarray(Z[f"pairs|{r_}|J"], int); S_ = L.pair_sums(Xs, Yt, I_, J_, centred=True)
        d4.append(L.ratio_C(S_["N"], S_["D"]).astype(np.float32).astype(np.float64)[ri, cols])
    d4 = np.where(np.isfinite(d4[0]) & np.isfinite(d4[1]), (d4[0] + d4[1]) / 2, np.nan); Zd4 = np.sign(d4) * pct
    isoS = np.where(np.isfinite(icm), np.sign(icm), np.nan); T = isoS * np.abs(Zd4)          # EXACTLY gb_kochip_audit_lib.py:46
    T_simple = isoS * pct; rank_ok = np.isfinite(T); n_d4_bad = int((~np.isfinite(d4) | (d4 == 0)).sum())
    def mets(Km):   # sign calls: NaN/0 = miss in the full direct-edge denominator; ranking: finite T only (rank_support reported)
        ts = np.sign(Km[Y]); ps = sg[Y]; ra = float((ps[ts > 0] > 0).mean()); rr = float((ps[ts < 0] < 0).mean())
        return dict(sign_acc=float((ps == ts).mean()), sign_BA=.5 * (ra + rr), recall_act=ra, recall_rep=rr, AUROC_act=AUC((Km > 0)[rank_ok], T[rank_ok]),
                    AUROC_rep=AUC((Km < 0)[rank_ok], -T[rank_ok]), AP_act=APs((Km > 0)[rank_ok], T[rank_ok]), AP_rep=APs((Km < 0)[rank_ok], -T[rank_ok]))
    mixed = [i for i in range(len(src)) if len(set(np.sign(Ks[i][Y[i]]))) > 1]
    r = dict(grn=g, topo=t, cond=c, sim=s, strata=strata_mode, NO_CALL=bool(NO_CALL), NB=NB, rank_support=float(rank_ok.mean()), ranking_conditional=bool(not rank_ok.all()),
             n_d4_zero_or_nan=n_d4_bad, max_abs_T_literal_minus_simple=float(np.nanmax(np.abs(T - T_simple))), mixed_sign_sources=len(mixed),
             edges_in_mixed_sources=int(sum(Y[i].sum() for i in mixed)), n_nocall_direct=int((sg[Y] == 0).sum()),
             frac_pos_sign_direct=float((sg[Y] > 0).mean()), root_sign_disagree_direct=float((np.sign(per[ROOTS[0] + '_mean'][ri, cols]) != np.sign(per[ROOTS[1] + '_mean'][ri, cols]))[Y].mean()),
             seed_rep_direct=float(np.nanmean(np.abs(np.concatenate([np.sign(per[rt][:, ri, cols]) for rt in ROOTS]).mean(0))[Y])),
             pair_stats=st, feasibility=feas, **mets(Ks))
    fwd = f"{D.MAIN_ROOT}/signed_2026_10_28/full/signed_g{g}_{t}_{c}_s{s}.npz"
    if os.path.exists(fwd):
        fs = np.sign(np.load(fwd)["smean"]); r["agree_fwd_iso_direct"] = float((fs[Y] == sg[Y]).mean())
        ag = (fs == sg) & (sg != 0); ts = np.sign(Ks[Y]); ps = np.where(ag, sg, 0)[Y]
        r["agreegate_coverage_direct"] = float(ag[Y].mean()); r["agreegate_BA_fullden"] = float(.5 * ((ps[ts > 0] > 0).mean() + (ps[ts < 0] < 0).mean()))
    rng = np.random.default_rng(1000 + g); NUL = []
    for b in range(NB):
        Kp = Ks.copy()
        for i in range(len(src)):
            j = np.flatnonzero(Y[i]); Kp[i, j] = np.abs(Ks[i, j]) * rng.permutation(np.sign(Ks[i, j]))
        NUL.append(mets(Kp))
    NUL = pd.DataFrame(NUL)
    for m in ("sign_BA", "AUROC_act", "AUROC_rep", "AP_act", "AP_rep"):
        r[f"null_{m}_mean"] = float(NUL[m].mean()); r[f"null_{m}_p"] = float((1 + (NUL[m] >= r[m]).sum()) / (NB + 1))
    r["code_sha"] = hashlib.sha256(open(__file__, "rb").read()).hexdigest()[:16]
    stem = f"{OUT}/iso_g{g}_{t}_{c}_s{s}_{strata_mode}"
    np.savez_compressed(stem + ".npz", src=src, cols=cols, Dm=Dm, K=Ks, iso_C=icm, d4=d4, primary=prim, T=T, T_simple=T_simple, null=NUL.to_numpy(), null_cols=np.array(NUL.columns))
    json.dump(r, open(stem + ".json", "w"))
    print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if k not in ("pair_stats",)}))


if __name__ == "__main__":
    if sys.argv[1] == "parity": parity(sys.argv[2].upper(), sys.argv[3].upper())
    else:
        sm = sys.argv[sys.argv.index("--strata") + 1] if "--strata" in sys.argv else "one"
        sim(sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5], int(sys.argv[6]), sm)
