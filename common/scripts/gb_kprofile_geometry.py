#!/usr/bin/env python
"""FULL k PROFILE + TRUTH-FREE SPECTRAL GEOMETRY, for GATA1 and PANC/RFX6 (PI revised brief §1-§3).

usage: gb_kprofile_geometry.py <dataset> <focal> <out_dir> [n_pseudo=200] [n_boot=1000]

THE QUESTION (§1). PANC/RFX6 failed at auto-k=29. One depth cannot distinguish
    (A) SVD has NO useful regime in PANC   from   (B) useful depths exist and the auto-k RULE missed them.
So every integer depth in the selector's own deployable domain is evaluated and the WHOLE profile is reported.

STATUS. PANC truth and GATA1 truth have both already been inspected. Per the PI correction this profile is a
legitimate RETROSPECTIVE CHARACTERISATION -- the score stays observational (no KO/ChIP label enters the score, the
operator, the roster or the grid) -- but it may NOT validate a deployment rule. A depth rule designed from these
curves must be tested where outcomes are still closed. argmax_k is never a headline.

internal review r145 REQUIRED CHANGES, all implemented here (the previous version was REJECTED):
 (r145-2) grid = EVERY integer k in 0..min(40, n_comp-1), the selector's deployable domain (spectral_k.max_k=40).
          Depths beyond 40 are run only if requested and are flagged `exploratory` -- they cannot support the
          current auto-k rule because the rule can never choose them.
 (r145-3) the selector's ACTUAL internals are serialised (slope, slope_smoothed, tail_baseline, threshold,
          tail_start, and the stable-window predicate evaluated at every k), so the auto-k decision surface is
          visible next to the performance curve. Cumulative spectral energy is kept under its own name and is NOT
          called the tail-exhaustion statistic (the previous version conflated the two).
 (r145-4) direct identity tests run before any score: Amat == (I-P_k) dXc; the closed-form q_r equals the direct
          residual norm; q_r in [0,1]; raw-k0 path; and a clearly DIAGNOSTIC-ONLY centred-k0 row, because raw k=0
          vs centred k=1 changes centring and projection at the same time, so a k0->k1 jump is not attributable
          to the first mode alone.
 (r145-7) ONE common finite support per (cell) across focal V, focal S and every pseudo row at every depth;
          >2% loss vs the panel universe => the row is emitted as UNRESOLVED, never silently skipped. nAP is
          THRESHOLD-GROUPED so tied scores (which deep projection can create) cannot be ordered by gene axis.
 (r145-6) SIMULTANEOUS uncertainty: one shared gene resample, reused across every root x depth, with a
          studentized max-t band, so the profile is not read through 82 independent marginal intervals.

DECISION RULE (frozen before the corrected numbers are read, from PANC_FROZEN_CONTRACT §E3): a depth is USEFUL iff
Delta_F = F_V - F_S >= 0.5 nAP with the SIMULTANEOUS one-sided lower band > 0 in BOTH roots. Classification uses
k>=1 only; raw k=0 is a separate historical baseline, not part of the contiguous profile.
    BROAD USEFUL REGION      >=3 consecutive integer depths pass in both roots
    NARROW ISOLATED REGION   >=1 depth passes but no run of 3
    NO USEFUL SVD REGION     no depth passes
and separately: AUTO-K MISSES IT if a passing run exists but either root's auto-k lies outside it.

GEOMETRY (§2). Integer k is not comparable across datasets (PANC 1,940 genes / ~144k pairs and a far flatter
spectrum than GATA1), so each depth also carries dimensionless truth-free coordinates: cumulative energy removed,
normalised rank position k/effective_rank (participation ratio), and the REGULATOR-SPECIFIC retained variation
    q_r(k) = ||(I - P_k) dX_r||^2 / ||dX_r||^2
which the global energy fraction cannot see: it asks how much of the FOCAL regulator's own pair variation survives
the projection. q_r is monotone non-increasing in k, so a constraint q_r >= q_min is an upper bound on k -- a
dimensionless REPARAMETRISATION of depth, which is the point, but not by itself a new estimator.
"""
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
USEFUL_DELTA_F = 0.5         # PANC_FROZEN_CONTRACT_2026_09_15.md:25
SUPPORT_DROP_MAX = 0.02
d_ = lambda M: np.asarray(M.todense()) if issparse(M) else np.asarray(M)


def nap_tg(y, s, mask):
    """THRESHOLD-GROUPED nAP on a FIXED common support mask (internal review r28 tie rule, gb_dev_evaluate:32).

    Tied scores form one threshold group and every member receives the precision at the END of the group, so the
    arbitrary gene-axis ordering cannot decide the metric. The mask is passed in rather than derived per vector,
    so focal V, focal S and every pseudo are scored on the IDENTICAL gene set (internal review r145-7)."""
    y2, s2 = y[mask], s[mask]; p = int(y2.sum())
    if p < 3 or p == y2.size: return np.nan
    o = np.argsort(-s2, kind="stable"); yo = y2[o].astype(float); so = s2[o]
    tp = np.cumsum(yo); n = np.arange(1, yo.size + 1)
    end = np.flatnonzero(np.r_[np.diff(so) != 0, True])          # last index of each tie group
    grp = np.searchsorted(end, np.arange(yo.size))               # group id of each position
    prec = (tp[end] / n[end])[grp]                               # precision at the END of the own group
    return float((prec * yo).sum() / p) / (p / y2.size)


def selector_surface(sv):
    """The auto-k rule's OWN decision surface, re-derived from spectral_k so the plot explains the choice."""
    r = select_k_by_tail_exhaustion(sv)
    ss = np.asarray(r.slope_smoothed, float); m = 3
    stable = np.zeros(len(ss), bool)
    for i in range(len(ss) - m + 1):
        stable[i] = bool(np.all(ss[i:i + m] <= r.threshold))
    return r, ss, stable


def truth_cells(dataset, genes, ti):
    """Panels. In EVERY panel the opposite polarity's positives are excluded as UNKNOWN, never scored as
    negatives -- the r101 defect that internal review r145-6 found again in the bridge script."""
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
    # internal review r146-6: hash inputs and code AT LAUNCH, not after the expensive score pass. Hashing afterwards lets a
    # process run old bytes while the receipt records whatever is on disk at the end -- which already produced one
    # stale artifact (bridge_PANC_2026_09_15). The launch hashes are re-asserted unchanged at completion.
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

    # ---------------- truth-free pass ----------------
    S = {}; geo = []; auto = {}; sel = {}; checks = []
    for root in ("dc1_min", "dc1_max"):
        I, J = f[f"{root}_I"].astype(np.int64), f[f"{root}_J"].astype(np.int64)
        dX_raw = X[J] - X[I]; dXc = dX_raw - dX_raw.mean(0, keepdims=True)
        U_, sv, Vt = np.linalg.svd(dXc, full_matrices=False)
        res, ss, stable = selector_surface(sv); ka = int(res.k_selected); auto[root] = ka
        sel[root] = dict(k_selected=ka, threshold=float(res.threshold), tail_baseline=float(res.tail_baseline),
                         tail_start=int(res.tail_start), slope_smoothed=[float(x) for x in ss],
                         # r146-5: the selector maps a qualifying slope start i to k = i+1 (spectral_k.py:57),
                         # so serialise the DEPTHS, not the zero-based slope indices.
                         stable_window_k=[int(i) + 1 for i in np.flatnonzero(stable)])
        e2 = sv ** 2; tot_e = float(e2.sum()); eff_rank = float(tot_e ** 2 / float((e2 ** 2).sum()))
        xr = dXc[:, ti]; tot_r = float(xr @ xr); coef = U_.T @ xr
        assert tot_r > 0, f"focal {focal} has zero centred pair variation in {root}"   # r146-5: assert BEFORE use
        KG = list(range(0, min(DEPLOY_MAX_K, len(sv) - 1) + 1))
        dYs = {"V": Vm[J] - Vm[I], "S": X[J] - X[I]}
        for k in KG + ["c0"]:                                    # "c0" = centred k=0, DIAGNOSTIC ONLY (r145-4)
            kk = 0 if k == "c0" else int(k)
            if k == "c0": Amat = dXc; prj = (lambda Y: Y)
            elif kk == 0: Amat = dX_raw; prj = (lambda Y: Y)     # raw k=0, faithful to the historical producer
            else:
                Uk = U_[:, :kk]; Amat = dXc - (Uk * sv[:kk][None, :]) @ Vt[:kk, :]
                prj = (lambda Y, Uk=Uk: Y - Uk @ (Uk.T @ Y))
                if kk in (1, ka, KG[-1]):                        # identity tests (r145-4)
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
               status="RETROSPECTIVE CHARACTERISATION: truth already inspected for both systems; the score is "
                      "observational (no label enters score/operator/roster/grid); NO deployment rule is "
                      "validated here -- a rule designed from these curves must be tested where truth is closed")
    with open(f"{out_dir}/pre_truth_geometry.json", "x") as fh: json.dump(pre, fh, indent=1)
    print("PRE-TRUTH:", json.dumps(dict(auto_k=auto, score_hash=shash, checks_max=float(
        max(max(c["amat_err"], c["qr_formula_vs_direct"]) for c in checks)))), flush=True)

    # ---------------- truth opened ----------------
    CLS = truth_cells(dataset, genes, ti); rows = []; g2 = np.random.default_rng(145)
    TRUTH_HASHES = {q: C3.SEL.file_sha256_16(q) for q in (
        [f"{H.G}/chipseq_labels.tsv", f"{H.G}/gata1_control_deterministic.h5ad",
         f"{H.G}/gata1_ko_projected_deterministic.h5ad"] if dataset == "GATA1" else [])
        if os.path.exists(q)}
    for cname, (pos, uni, pol) in CLS.items():
        idx = np.flatnonzero(uni); y = pos[idx].astype(int)
        # ONE common finite support for this panel across every compared vector at every depth (r145-7)
        fin = np.ones(len(idx), bool)
        for M in S.values():
            fin &= np.isfinite(M[:, idx]).all(0)
        drop = 1 - fin.mean()
        boot = [g2.integers(0, len(idx), len(idx)) for _ in range(n_boot)]   # ONE shared resample (r145-6)
        for root in ("dc1_min", "dc1_max"):
            for k in sorted(set(list(gdf.k.unique())) | {"c0"}, key=str):
                if drop > SUPPORT_DROP_MAX or y[fin].sum() < 5 or (y[fin] == 0).sum() < 5:   # r146-5 negatives too
                    rows.append(dict(root=root, k=k, cell=cname, status="UNRESOLVED",
                                     support_drop=float(drop), n_pos=int(y[fin].sum()))); continue
                MV, MS = S[(root, k, "V")], S[(root, k, "S")]
                v = pol * MV[0][idx]; s_ = pol * MS[0][idx]
                FV, FS = nap_tg(y, v, fin), nap_tg(y, s_, fin)
                PV = float(np.nanmedian([nap_tg(y, pol * MV[1 + q][idx], fin) for q in range(n_pseudo)]))
                PS = float(np.nanmedian([nap_tg(y, pol * MS[1 + q][idx], fin) for q in range(n_pseudo)]))
                bd = np.array([nap_tg(y[w], v[w], fin[w]) - nap_tg(y[w], s_[w], fin[w]) for w in boot])
                rows.append(dict(root=root, k=k, cell=cname,
                                 # internal review r147-1/3: a row is OK only if EVERY declared draw is finite; otherwise
                                 # nanmax would build replicate maxima over a smaller, shifting family.
                                 status="OK" if np.isfinite(bd).all() else "UNRESOLVED_BOOTSTRAP",
                                 support_drop=float(drop),
                                 n_pos=int(y[fin].sum()), n_uni=int(fin.sum()), F_V=FV, F_S=FS,
                                 Delta_F=FV - FS, E_V=FV - PV, E_S=FS - PS, Delta_E=(FV - PV) - (FS - PS),
                                 P_V=PV, P_S=PS, boot_sd=float(np.nanstd(bd)), boot_finite=int(np.isfinite(bd).sum()),
                                 _bd=bd))
            print(f"  {cname}/{root} scored {time.time()-t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)

    # simultaneous one-sided lower band per cell: studentized max-t over (root x depth) (r145-6)
    df["Delta_F_lo_sim"] = np.nan
    for cname in CLS:
        # internal review r146-1 BLOCKER: a simultaneous LOWER band needs the UPPER tail of the estimation error,
        # T_b = max_j (theta*_jb - theta_j)/sd_j, giving L_j = theta_j - c_.95 * sd_j. The previous version used
        # (theta - theta*), the opposite tail; the two agree only under symmetry, which normalised AP with sparse
        # positives and ties does not have. c0 (centred k=0) is DIAGNOSTIC-ONLY and is excluded from the family
        # (r146-5) so it cannot inflate the family and alter the decision.
        ok = df.index[(df.cell == cname) & (df.status == "OK") & (df.k != "c0")]
        if not len(ok): continue
        D = np.vstack([df.at[i, "_bd"] for i in ok]); est = df.loc[ok, "Delta_F"].to_numpy()
        sd = np.nanstd(D, axis=1); sd = np.where(sd > 1e-12, sd, np.nan)
        t = (D - est[:, None]) / sd[:, None]
        # internal review r147-3: finite-B coverage uses method="higher", as the certified max-t band does
        # (gb_hist_specificity_family.py:207). Interpolated quantiles undercover at these B.
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
        print(f"\n=== {dataset}/{cname}  Delta_F(k) = F_V - F_S (principal diagnostic) ===")
        print(m[(m.cell == cname) & (m.status == "OK")].pivot_table(
            index="k", columns="root", values=["Delta_F", "Delta_F_lo_sim", "F_V", "F_S", "q_r"],
            sort=False).round(3).to_string())
        print("VERDICT:", json.dumps(verdict[cname]))


if __name__ == "__main__":
    main(*sys.argv[1:6])
