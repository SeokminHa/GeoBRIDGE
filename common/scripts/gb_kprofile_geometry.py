#!/usr/bin/env python
# Full k profile and truth-free spectral geometry for GATA1 and PANC/RFX6
import json, os, sys, time
import numpy as np, pandas as pd
import anndata as ad
from scipy.sparse import issparse
R = __import__("os").environ["GEOBRIDGE_ROOT"]; sys.path.insert(0, R); sys.path.insert(0, f"{R}/scratch")
from geobridge_model.scoring import score_matrix
from geobridge_model.spectral_k import select_k_by_tail_exhaustion
import gb_m1_calib3 as C3
import gb_hist_specificity_family as H
import gb_dev_evaluate as E

CFG = {
    "GATA1": dict(fixture=H.FIXTURE, pairs=H.PAIRS),
    "PANC": dict(fixture=f"{R}/results/pr_arm2_2026_08_16/panc/panc_pr_stochastic.h5ad",
                 pairs=f"{R}/results/panc_pr_arm2_surfaces_2026_08_16/panc_pr_frozen_pairs.npz"),
}
CHANNELS = ("V", "S")
DEPLOY_MAX_K = 40            # spectral_k.select_k_by_tail_exhaustion(max_k=40): the rule cannot choose beyond this
USEFUL_DELTA_F = 0.5         # Historical useful-excess threshold.
SUPPORT_DROP_MAX = 0.02
d_ = lambda M: np.asarray(M.todense()) if issparse(M) else np.asarray(M)


# threshold-grouped nAP on a fixed common support mask
def nap_tg(y, s, mask):
    y2, s2 = y[mask], s[mask]; p = int(y2.sum())
    if p < 3 or p == y2.size: return np.nan
    o = np.argsort(-s2, kind="stable"); yo = y2[o].astype(float); so = s2[o]
    tp = np.cumsum(yo); n = np.arange(1, yo.size + 1)
    end = np.flatnonzero(np.r_[np.diff(so) != 0, True])          # last index of each tie group
    grp = np.searchsorted(end, np.arange(yo.size))               # group id of each position
    prec = (tp[end] / n[end])[grp]                               # precision at the end of the own group
    return float((prec * yo).sum() / p) / (p / y2.size)


# The auto-k rule's own decision surface, re-derived from spectral_k so the plot explains the choice
def selector_surface(sv):
    r = select_k_by_tail_exhaustion(sv)
    ss = np.asarray(r.slope_smoothed, float); m = 3
    stable = np.zeros(len(ss), bool)
    for i in range(len(ss) - m + 1):
        stable[i] = bool(np.all(ss[i:i + m] <= r.threshold))
    return r, ss, stable


# Panels
def truth_cells(dataset, genes, ti):
    if dataset == "GATA1":
        return H.hist_classes(genes, ti)
    y, pol = E.TRUTH["PANC_RFX6"](list(genes)); out = {}
    for cname, ppol, sgn in (("activation", -1.0, +1.0), ("repression", +1.0, -1.0)):
        pos = ((y == 1) & (pol == ppol)).copy()
        uni = ((y != -1) & ~((y == 1) & (pol != ppol))).copy()
        pos[ti] = False; uni[ti] = False
        out[cname] = (pos, uni, sgn)
    return out


def main(dataset, focal, out_dir, n_pseudo=200, n_boot=1000):
    t0 = time.time(); os.makedirs(out_dir, exist_ok=True)
    n_pseudo, n_boot = int(n_pseudo), int(n_boot); cfg = CFG[dataset]
    LAUNCH = dict(code=C3.SEL.file_sha256_16(__file__), fixture=C3.SEL.file_sha256_16(cfg["fixture"]),
                  pairs=C3.SEL.file_sha256_16(cfg["pairs"]),
                  deps={p: C3.SEL.file_sha256_16(f"{R}/{p}") for p in
                        ("geobridge_model/scoring.py", "geobridge_model/spectral_k.py",
                         "scratch/gb_hist_specificity_family.py", "scratch/gb_dev_evaluate.py")})
    A = ad.read_h5ad(cfg["fixture"]); f = np.load(cfg["pairs"]); pg = [str(g) for g in f["genes"]]
    if not set(pg) <= set(str(g) for g in A.var_names): raise SystemExit("pair genes absent from fixture")
    A = A[:, pg].copy()                                          # ordered subset onto the frozen pair axis
    genes = np.asarray(A.var_names).astype(str); gU = np.char.upper(genes)
    ti = {g: i for i, g in enumerate(gU)}[focal.upper()]
    X = d_(A.X).astype(np.float64); Vm = d_(A.layers["velocity"]).astype(np.float64); n = len(genes)
    axis_hash = C3.SEL.raw_hash("|".join(genes).encode())["hex16"]
    mu = X.mean(0); det = (X > 0).mean(0); z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    dist = np.sqrt((z(mu) - z(mu)[ti]) ** 2 + (z(det) - z(det)[ti]) ** 2); dist[ti] = np.inf
    roster = np.argsort(dist)[:n_pseudo]; srcs = np.concatenate([[ti], roster])

    # truth-free pass
    S = {}; geo = []; auto = {}; sel = {}; checks = []
    for root in ("dc1_min", "dc1_max"):
        I, J = f[f"{root}_I"].astype(np.int64), f[f"{root}_J"].astype(np.int64)
        dX_raw = X[J] - X[I]; dXc = dX_raw - dX_raw.mean(0, keepdims=True)
        U_, sv, Vt = np.linalg.svd(dXc, full_matrices=False)
        res, ss, stable = selector_surface(sv); ka = int(res.k_selected); auto[root] = ka
        sel[root] = dict(k_selected=ka, threshold=float(res.threshold), tail_baseline=float(res.tail_baseline),
                         tail_start=int(res.tail_start), slope_smoothed=[float(x) for x in ss],
                         stable_window_k=[int(i) + 1 for i in np.flatnonzero(stable)])
        e2 = sv ** 2; tot_e = float(e2.sum()); eff_rank = float(tot_e ** 2 / float((e2 ** 2).sum()))
        xr = dXc[:, ti]; tot_r = float(xr @ xr); coef = U_.T @ xr
        assert tot_r > 0, f"focal {focal} has zero centred pair variation in {root}"
        KG = list(range(0, min(DEPLOY_MAX_K, len(sv) - 1) + 1))
        dYs = {"V": Vm[J] - Vm[I], "S": X[J] - X[I]}
        for k in KG + ["c0"]:
            kk = 0 if k == "c0" else int(k)
            if k == "c0": Amat = dXc; prj = (lambda Y: Y)
            elif kk == 0: Amat = dX_raw; prj = (lambda Y: Y)     # raw k=0, faithful to the historical producer
            else:
                Uk = U_[:, :kk]; Amat = dXc - (Uk * sv[:kk][None, :]) @ Vt[:kk, :]
                prj = (lambda Y, Uk=Uk: Y - Uk @ (Uk.T @ Y))
                if kk in (1, ka, KG[-1]):
                    checks.append(dict(root=root, k=kk,
                        amat_err=float(np.abs(Amat - (dXc - Uk @ (Uk.T @ dXc))).max()),
                        qr_formula_vs_direct=float(abs(
                            (tot_r - float(coef[:kk] @ coef[:kk])) / tot_r
                            - float((xr - Uk @ (Uk.T @ xr)) @ (xr - Uk @ (Uk.T @ xr))) / tot_r))))
            if k != "c0":
                q = float((tot_r - float(coef[:kk] @ coef[:kk])) / tot_r) if kk else 1.0
                assert -1e-9 <= q <= 1 + 1e-9 and tot_r > 0, f"q_r out of range {q}"
                geo.append(dict(root=root, k=kk, energy_removed=float(e2[:kk].sum() / tot_e) if kk else 0.0,
                                rank_position=kk / eff_rank, effective_rank=eff_rank, q_r=q,
                                cum_energy=float(np.cumsum(e2)[kk - 1] / tot_e) if kk else 0.0,
                                slope_smoothed=float(ss[kk - 1]) if 0 < kk <= len(ss) else np.nan,
                                selector_stable_here=bool(stable[kk - 1]) if 0 < kk <= len(stable) else False,
                                is_auto_k=bool(kk == ka), n_components=int(len(sv)),
                                exploratory=bool(kk > DEPLOY_MAX_K)))
            for ch in CHANNELS:
                S[(root, k, ch)] = score_matrix(Amat[:, srcs], prj(dYs[ch]))
        print(f"  {root} auto-k={ka} all depths scored {time.time()-t0:.0f}s", flush=True)
        del dX_raw, dXc, U_, dYs
    assert all(c["amat_err"] < 1e-8 and c["qr_formula_vs_direct"] < 1e-10 for c in checks), checks
    gdf = pd.DataFrame(geo); gdf.to_csv(f"{out_dir}/geometry.tsv", sep="\t", index=False)
    shash = C3.SEL.raw_hash(b"".join(f"{k}".encode() + np.ascontiguousarray(v).tobytes() for k, v in sorted(S.items(), key=str)))["hex16"]
    pre = dict(dataset=dataset, focal=focal, auto_k=auto, selector=sel, n_genes=int(n), n_pseudo=n_pseudo,
               n_boot=n_boot, deploy_max_k=DEPLOY_MAX_K, k_grid=sorted(set(gdf.k.tolist())),
               identity_checks=checks, score_hash_before_truth=shash, gene_axis_hash=axis_hash,
               roster_genes=[str(genes[i]) for i in roster], launch_hashes=LAUNCH,
               useful_rule=f"Delta_F >= {USEFUL_DELTA_F} AND simultaneous one-sided lower band > 0 in BOTH roots",
               status="Descriptive depth and geometry profiles; these analyses do not define a deployment rule.")
    with open(f"{out_dir}/pre_truth_geometry.json", "x") as fh: json.dump(pre, fh, indent=1)
    print("PRE-TRUTH:", json.dumps(dict(auto_k=auto, score_hash=shash, checks_max=float(
        max(max(c["amat_err"], c["qr_formula_vs_direct"]) for c in checks)))), flush=True)

    # truth opened
    CLS = truth_cells(dataset, genes, ti); rows = []; g2 = np.random.default_rng(145)
    TRUTH_HASHES = {q: C3.SEL.file_sha256_16(q) for q in (
        [f"{H.G}/chipseq_labels.tsv", f"{H.G}/gata1_control_deterministic.h5ad",
         f"{H.G}/gata1_ko_projected_deterministic.h5ad"] if dataset == "GATA1" else [])
        if os.path.exists(q)}
    for cname, (pos, uni, pol) in CLS.items():
        idx = np.flatnonzero(uni); y = pos[idx].astype(int)
        fin = np.ones(len(idx), bool)
        for M in S.values():
            fin &= np.isfinite(M[:, idx]).all(0)
        drop = 1 - fin.mean()
        boot = [g2.integers(0, len(idx), len(idx)) for _ in range(n_boot)]
        for root in ("dc1_min", "dc1_max"):
            for k in sorted(set(list(gdf.k.unique())) | {"c0"}, key=str):
                if drop > SUPPORT_DROP_MAX or y[fin].sum() < 5 or (y[fin] == 0).sum() < 5:
                    rows.append(dict(root=root, k=k, cell=cname, status="UNRESOLVED",
                                     support_drop=float(drop), n_pos=int(y[fin].sum()))); continue
                MV, MS = S[(root, k, "V")], S[(root, k, "S")]
                v = pol * MV[0][idx]; s_ = pol * MS[0][idx]
                FV, FS = nap_tg(y, v, fin), nap_tg(y, s_, fin)
                PV = float(np.nanmedian([nap_tg(y, pol * MV[1 + q][idx], fin) for q in range(n_pseudo)]))
                PS = float(np.nanmedian([nap_tg(y, pol * MS[1 + q][idx], fin) for q in range(n_pseudo)]))
                bd = np.array([nap_tg(y[w], v[w], fin[w]) - nap_tg(y[w], s_[w], fin[w]) for w in boot])
                rows.append(dict(root=root, k=k, cell=cname,
                                 status="OK" if np.isfinite(bd).all() else "UNRESOLVED_BOOTSTRAP",
                                 support_drop=float(drop),
                                 n_pos=int(y[fin].sum()), n_uni=int(fin.sum()), F_V=FV, F_S=FS,
                                 Delta_F=FV - FS, E_V=FV - PV, E_S=FS - PS, Delta_E=(FV - PV) - (FS - PS),
                                 P_V=PV, P_S=PS, boot_sd=float(np.nanstd(bd)), boot_finite=int(np.isfinite(bd).sum()),
                                 _bd=bd))
            print(f"  {cname}/{root} scored {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)

    df["Delta_F_lo_sim"] = np.nan
    for cname in CLS:
        ok = df.index[(df.cell == cname) & (df.status == "OK") & (df.k != "c0")]
        if not len(ok): continue
        D = np.vstack([df.at[i, "_bd"] for i in ok]); est = df.loc[ok, "Delta_F"].to_numpy()
        sd = np.nanstd(D, axis=1); sd = np.where(sd > 1e-12, sd, np.nan)
        t = (D - est[:, None]) / sd[:, None]
        c = float(np.nanquantile(np.nanmax(t, axis=0), .95, method="higher"))
        df.loc[ok, "Delta_F_lo_sim"] = est - c * sd
        df.loc[ok, "maxt_c"] = c
    df = df.drop(columns=["_bd"])
    df["useful"] = (df.Delta_F >= USEFUL_DELTA_F) & (df.Delta_F_lo_sim > 0)
    df.to_csv(f"{out_dir}/kprofile.tsv", sep="\t", index=False)

    # classification: k>=1 integer depths, both roots, contiguous runs
    verdict = {}
    for cname in CLS:
        sub = df[(df.cell == cname) & (df.status == "OK") & (df.k != "c0")].copy()
        sub["k"] = sub["k"].astype(int)
        both = sorted(k for k in sub.k.unique() if k >= 1
                      and bool(sub[(sub.k == k)].useful.all()) and (sub.k == k).sum() == 2)
        runs, cur = [], []
        for k in both:
            cur = cur + [k] if (cur and k == cur[-1] + 1) else [k]
            if len(cur) > len(runs): runs = list(cur)
        shape = ("BROAD USEFUL REGION" if len(runs) >= 3 else
                 "NARROW ISOLATED REGION" if both else "NO USEFUL SVD REGION")
        misses = bool(runs) and not all(auto[r] in runs for r in auto)
        verdict[cname] = dict(shape=shape, passing_depths=both, longest_run=runs, auto_k=auto,
                              auto_k_outside_region=misses)
    now = dict(code=C3.SEL.file_sha256_16(__file__), fixture=C3.SEL.file_sha256_16(cfg["fixture"]),
               pairs=C3.SEL.file_sha256_16(cfg["pairs"]),
               deps={p: C3.SEL.file_sha256_16(f"{R}/{p}") for p in LAUNCH["deps"]})
    assert now == LAUNCH, f"inputs changed during the run -- artifact is not replayable: {LAUNCH} -> {now}"
    with open(f"{out_dir}/kprofile_geometry.json", "x") as fh:
        json.dump(dict(pre_truth=pre, verdict=verdict, n_rows=int(len(df)), runtime_s=time.time() - t0,
                       launch_hashes_reasserted=now,
                       output_hashes={n: C3.SEL.file_sha256_16(f"{out_dir}/{n}")
                                      for n in ("kprofile.tsv", "geometry.tsv")},
                       truth_hashes=TRUTH_HASHES), fh, indent=1)
    pd.set_option("display.width", 260)
    m = df.merge(gdf, on=["root", "k"], how="left")
    for cname in CLS:
        print(f"\n{dataset}/{cname}  Delta_F(k) = F_V - F_S (principal diagnostic)")
        print(m[(m.cell == cname) & (m.status == "OK")].pivot_table(
            index="k", columns="root", values=["Delta_F", "Delta_F_lo_sim", "F_V", "F_S", "q_r"],
            sort=False).round(3).to_string())
        print("VERDICT:", json.dumps(verdict[cname]))


if __name__ == "__main__":
    main(*sys.argv[1:6])
