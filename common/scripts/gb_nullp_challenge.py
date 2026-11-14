#!/usr/bin/env python
# Compare target-prior utility with permutation-null priors.
from __future__ import annotations

import os
import sys

for _i, _a in enumerate(sys.argv):                                    # thread cap before numpy import (shared hosts)
    if _a == "--threads" and _i + 1 < len(sys.argv):
        for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
            os.environ[_v] = sys.argv[_i + 1]

import argparse
import glob
import hashlib
import json
import platform
import time
import warnings
import zlib
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
R = __import__("os").environ["GEOBRIDGE_ROOT"]
D5_DIR = f"{R}/results/real_uniform/defense/D5"
OUT_DEFAULT = f"{R}/results/real_uniform/nullp_challenge"
PIN = {"gb_ptable.py": "5db6f77d7e0bfd6b1f3e666014e74ff9a7b56798af384cad915228ee31719166",
       "gb_defense_recompute_v1_89c4c21b.py": "89c4c21ba6d797f3d91ee509bba8e772034190bd1f77dc4d747affff4ddacc90"}
PREDECL = f"{R}/internal_design_note_not_shipped_NULLP.md"
PREDECL_SHA = "e8afeeedf73274a16e7a43991532ee478538218a87997c410c35450481806b9e"
NPERM, SCHEMES, MINC = 20, ("g", "m50"), 5
MET = ("AUROC", "AUPRC", "APLIFT", "P50")
BMET = {"AUROC": "auc_any", "APLIFT": "aplift_any", "P50": "yield_any@50_stratified", "RHO_BREADTH": "rho_breadth", "RHO_BURDEN": "rho_burden"}
NUL = [f"null_{k:02d}" for k in range(NPERM)]
STRATA = ("all", "det_T1", "det_T2", "det_T3")                      # sm_det low / mid / high tertile of the analysis rows
GROUPS = {"PANC": ["PANC"], "GATA1": ["GATA1", "KLF1", "KLF1_B"], "BM": ["BM_CHIPATLAS", "BM_GATA1_ACUTE"], "K562": ["K562_EMP_ALLNULL"], "BREADTH": []}
LAT2D5 = {"PANC": "PANC", "GATA1": "GATA1", "KLF1": "GATA1", "KLF1_B": "GATA1", "BM_CHIPATLAS": "BONEMARROW", "BM_GATA1_ACUTE": "BONEMARROW", "K562_EMP_ALLNULL": "K562"}
SHUFFLE_LABEL = ("persistence = target-level structure survives destroyed within-stratum alignment; disappearance = alignment dependence. "
                 "NEITHER is an 'artifact fraction' (the shuffle also destroys target-state geometry and smoothness).")


def sha256(p, chunk=1 << 22):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def atomic_tsv(df, p):
    tmp = f"{p}.tmp{os.getpid()}"
    df.to_csv(tmp, sep="\t", index=False)
    os.replace(tmp, p)


def atomic_json(p, obj):
    tmp = f"{p}.tmp{os.getpid()}"
    json.dump(obj, open(tmp, "w"), indent=1, default=str)
    os.replace(tmp, p)


def pins_ok():
    if __import__("os").environ.get("GB_SKIP_PREFLIGHT") == "1":      # opt-in (public release)
        print("preflight skipped (public release)")
        return
    for fn, want in PIN.items():
        got = sha256(HERE / fn)
        if got != want:
            raise SystemExit(f"REFUSING: {fn} sha {got[:12]} != pinned {want[:12]}")
    if sha256(PREDECL) != PREDECL_SHA:
        raise SystemExit("REFUSING: pre-declaration document sha differs from the pinned one")


# D5 vectors
# P_real (G,), P_null (20, G) from the D5 npz
def d5_load(sysname):
    fs = sorted(glob.glob(f"{D5_DIR}/{sysname}_perm[0-9][0-9].npz"))
    if [os.path.basename(f) for f in fs] != [f"{sysname}_perm{k:02d}.npz" for k in range(NPERM)]:
        raise SystemExit(f"REFUSING: D5 {sysname} does not have exactly perm00..perm{NPERM-1:02d}: {len(fs)} files")
    pr, pn, shas, subset = None, [], {}, None
    for f in fs:
        rc = json.load(open(f.replace(".npz", ".receipt.json")))
        shas[os.path.basename(f)] = sha256(f)
        if rc["output_sha256"] != shas[os.path.basename(f)]:
            raise SystemExit(f"REFUSING: {f} differs from its receipt")
        if subset is not None and rc["subset"] != subset:
            raise SystemExit(f"REFUSING: subset differs across D5 perms {f}")
        subset = rc["subset"]
        with np.load(f) as z:
            if pr is None:
                pr = np.array(z["P_real"], float)
            elif not np.array_equal(pr, z["P_real"], equal_nan=True):
                raise SystemExit(f"REFUSING: P_real differs across D5 perms ({f})")
            pn.append(np.array(z["P_null"], float))
    pn = np.vstack(pn)
    return dict(P_real=pr, P_null=pn, shas=shas, subset=subset, script_sha256=rc["script_sha256"], G=len(pr), nT=int(np.isfinite(pr).sum()), nT_null=[int(np.isfinite(v).sum()) for v in pn])


def d5_axis(sysname):
    import gb_real_uniform as RU
    return np.asarray(RU.cached_roster(sysname)[0]).astype(str)


# contrast core
# pts {vec: {metric: float}}, bs {scheme: {vec: {metric: (B,)}}}; vec in {'real', null_00..null_19, ...}
def contrast(pts, bs, metrics, id_cols):
    import gb_ptable as PT
    row, draws = dict(id_cols), {}
    for m in metrics:
        rp = pts["real"][m]
        npts = np.array([pts[n][m] for n in NUL], float)
        ok = bool(np.isfinite(rp) and np.isfinite(npts).all())
        med = float(np.median(npts)) if ok else np.nan
        row.update({f"{m}_real": rp, f"{m}_mednull": med, f"{m}_delta": rp - med if ok else np.nan, f"{m}_nullmin": float(npts.min()) if ok else np.nan,
                    f"{m}_nullmax": float(npts.max()) if ok else np.nan, f"{m}_frac_real_gt_null": float((rp > npts).mean()) if ok else np.nan, f"{m}_n_tie": int((rp == npts).sum()) if ok else -1})
        for sch in SCHEMES:
            rb = np.asarray(bs[sch]["real"][m], float)
            nb = np.vstack([np.asarray(bs[sch][n][m], float) for n in NUL])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                mb = np.nanmedian(nb, 0)
            dd = rb - mb
            sx = "" if sch == "g" else f"_{sch}"
            row[f"{m}_delta{sx}_lo"], row[f"{m}_delta{sx}_hi"] = PT.pct_ci(dd)
            row[f"{m}_delta{sx}_p"] = PT.boot_p(dd)
            row[f"{m}_real{sx}_lo"], row[f"{m}_real{sx}_hi"] = PT.pct_ci(rb)
            row[f"{m}_mednull{sx}_lo"], row[f"{m}_mednull{sx}_hi"] = PT.pct_ci(mb)
            draws[f"{sch}|{m}"] = dd.astype(np.float32)
    return row, draws


# long per-vector point table: real, each permutation, the median-null (median of the per-permutation metrics)
def vector_rows(pts, metrics, id_cols, extra=("elemMedianP", "GBP_fullbank")):
    out = []
    for v in ["real"] + NUL:
        out.append(dict(id_cols, vector=v, **{m: pts[v][m] for m in metrics}))
    out.append(dict(id_cols, vector="median_null", **{m: float(np.median([pts[n][m] for n in NUL])) for m in metrics}))
    for v in extra:
        if v in pts:
            out.append(dict(id_cols, vector=v, **{m: pts[v][m] for m in metrics}))
    return out


# Analysis rows split at the 1/3 and 2/3 quantiles of sm_det
def tertiles(sm, rows):
    q1, q2 = np.quantile(sm[rows], [1 / 3, 2 / 3])
    if not q1 < q2:
        return {"all": rows, "det_T1": None, "det_T2": None, "det_T3": None}
    return {"all": rows, "det_T1": rows & (sm <= q1), "det_T2": rows & (sm > q1) & (sm <= q2), "det_T3": rows & (sm > q2)}


_LOO = {}


# subset-P with the focal TF's own row dropped from the 100+focal subset
def loo_vectors(d5s, tf, d5):
    if d5s not in _LOO:
        import gb_defense_recompute_v1_89c4c21b as M
        D, B, srcs, _present, _axis, au, src_axis, _det = M.common_setup(d5s)
        rr, _used = M.real_raw(d5s, D, B, srcs)
        M.blank_self(rr, src_axis)
        Pr, _U = M.primary_rows(rr, srcs, au)
        Pn = []
        for k in range(NPERM):
            with np.load(f"{D5_DIR}/{d5s}_perm{k:02d}.npz") as z:
                Pn.append(np.asarray(z["Prows_null"], float))
        _LOO[d5s] = (M, srcs, src_axis, Pr.astype(np.float32).astype(float), Pn)
    M, srcs, src_axis, Pr, Pn = _LOO[d5s]
    j = [i for i, x in enumerate(srcs) if x.upper() == tf.upper()]
    if not j:
        return d5["P_real"], d5["P_null"], False
    keep = np.array([i for i in range(len(srcs)) if i != j[0]])
    names = [srcs[i] for i in keep]

    def sp(rows):
        r = M.subset_prior(rows[keep], src_axis[keep], names)
        if r is None:
            raise SystemExit(f"REFUSING: leave-focal-out subset-P undefined for {tf}")
        return r["P"]
    return sp(Pr), np.vstack([sp(x) for x in Pn]), True


# lattice units
def run_lattice_group(a, group, out):
    import gb_audit_decomp as AD
    import gb_layer_lattice as L
    import gb_lattice_extra as LE
    import gb_ptable as PT
    LE.register()
    t0 = time.time()
    crows, vrows, draws, metas = [], [], {}, {}
    jsm = list(L.REFS).index("REF -sm_det")
    ssm = L.REF_SIGNS["REF -sm_det"][1]
    for system in GROUPS[group]:
        d5s = LAT2D5[system]
        d5 = d5_load(d5s)
        axis = d5_axis(d5s)
        if len(axis) != d5["G"]:
            raise SystemExit(f"REFUSING: D5 {d5s} axis {len(axis)} != vector length {d5['G']}")
        dr, _ = PT.lattice_draws(system, a.B, sizes=(PT.PRIMARY_SIZE,))
        if not np.array_equal(np.asarray(dr.gene.names).astype(str), axis):
            raise SystemExit(f"REFUSING: block axis of {system} differs from the D5 {d5s} axis")
        assert dr.schemes == SCHEMES, dr.schemes
        allowed = AD.tf_list_audit(system, L, LE)
        tfs = [t.strip() for t in a.tfs.split(",") if t.strip()] or allowed
        bad = [t for t in tfs if t not in allowed]
        if bad:
            raise SystemExit(f"REFUSING: --tfs {bad} not in the audit roster of {system}")
        if a.nslice > 1:
            tfs = [t for i, t in enumerate(tfs) if i % a.nslice == a.slice]
        for tf in tfs:
            t1 = time.time()
            try:
                u = PT.lattice_unit(system, tf, dr.gene)
            except (L.Blocked, SystemExit) as e:
                metas[f"{system}|{tf}"] = dict(status=f"NOT_RUN: {e}")
                print(f"[nullp] {system}/{tf}: NOT_RUN {e}", flush=True)
                if isinstance(e, SystemExit) and "REFUSING_PARITY" in str(e):
                    raise
                continue
            gi, y, MR = u["gi"], np.asarray(u["y"], bool), np.asarray(u["MR"], bool)
            pr_ax, pn_ax, dropped = loo_vectors(d5s, tf, d5) if a.loo else (d5["P_real"], d5["P_null"], False)
            vr, vn = pr_ax[gi], pn_ax[:, gi]
            fin = np.isfinite(vr) & np.isfinite(vn).all(0)
            rows = MR & fin
            sm = u["Xr"][:, jsm] * ssm                                           # sm_det itself (REF column = sign * feature)
            gbp = np.where(np.isfinite(u["P"]), u["P"], np.nan)
            vecs = {"real": vr, **{n: vn[k] for k, n in enumerate(NUL)}}
            base = dict(kind="lattice", system=system, tf=tf, endpoint="resp", block=PT.MACRO_BLOCK.get(system, ""), d5_system=d5s, gate_MR=bool(u["gate"]["MR"]), n_U=int(u["U"].sum()),
                        n_MR=int(MR.sum()), n_rows=int(rows.sum()), cov_rows_of_MR=float(rows.sum() / max(MR.sum(), 1)), draws_fp=dr.fp, d5_nT=d5["nT"], loo_focal=bool(a.loo), focal_row_dropped=bool(dropped))
            row_all, gbp_au = None, np.nan
            for st, srows in tertiles(sm, rows).items():
                if srows is None:
                    crows.append(dict(base, stratum=st, status="STRATUM_UNDEFINED: sm_det saturated (tertile cuts coincide)", n_rows_stratum=0, n_pos=0, n_neg=0))
                    continue
                yy = y[srows]
                idc = dict(base, stratum=st, n_rows_stratum=int(srows.sum()), n_pos=int(yy.sum()), n_neg=int((~yy).sum()))
                if yy.sum() < MINC or (~yy).sum() < MINC:
                    crows.append(dict(idc, status="TOO_FEW"))
                    continue
                allv = dict(vecs, elemMedianP=np.median(vn, 0))
                res = {n: PT.split_eval(v, y, srows, gi, dr.C, a.B) for n, v in allv.items()}
                if np.isfinite(gbp[srows]).all():
                    res["GBP_fullbank"] = PT.split_eval(gbp, y, srows, gi, dr.C, a.B)
                pts = {n: r[0] for n, r in res.items()}
                bs = {sch: {n: r[1][si] for n, r in res.items()} for si, sch in enumerate(SCHEMES)}
                row, dd = contrast(pts, bs, MET, idc)
                row["status"] = "OK" if base["gate_MR"] else "DESCRIPTIVE_ONLY"
                row["spearman_Preal_vs_fullbankP"] = float(pd.Series(vr[srows]).corr(pd.Series(gbp[srows]), method="spearman")) if np.isfinite(gbp[srows]).all() else np.nan
                crows.append(row)
                if st == "all":
                    row_all, gbp_au = row, pts.get("GBP_fullbank", {}).get("AUROC", np.nan)
                vrows += vector_rows(pts, MET, dict(idc, status=row["status"]))
                uid = f"lattice|{system}|{tf}|resp|{st}"
                draws.update({f"{uid}|{k}": v for k, v in dd.items()})
            metas[f"{system}|{tf}"] = dict(n_U=base["n_U"], n_MR=base["n_MR"], n_rows=base["n_rows"], gate=u["gate"], n_pos=int(y[rows].sum()), files=u["files"], methods_used=u["use"],
                                          seconds=round(time.time() - t1, 1))
            if row_all is not None and "AUROC_real" in row_all:
                r0 = row_all
                print(f"[nullp] {system}/{tf}: MR={base['n_MR']} rows={base['n_rows']} pos={int(y[rows].sum())} gate={base['gate_MR']} AUROC real={r0['AUROC_real']:.4f} medNull={r0['AUROC_mednull']:.4f} "
                      f"d={r0['AUROC_delta']:+.4f} [m {r0['AUROC_delta_m50_lo']:+.3f},{r0['AUROC_delta_m50_hi']:+.3f}] frac={r0['AUROC_frac_real_gt_null']:.2f} fullbankP={gbp_au:.4f} {time.time() - t1:.0f}s", flush=True)
            else:
                print(f"[nullp] {system}/{tf}: MR={base['n_MR']} rows={base['n_rows']} TOO_FEW {time.time() - t1:.0f}s", flush=True)
        write_part(out, f"part_{group}_{system}" + (f"_s{a.slice}of{a.nslice}" if a.nslice > 1 else ""), crows, vrows, draws,
                   dict(group=group, system=system, B=a.B, loo_focal=bool(a.loo), d5=dict(system=d5s, shas=d5["shas"], subset_n=len(d5["subset"]), subset_head=d5["subset"][:5], nT=d5["nT"], nT_null=d5["nT_null"], script_sha256=d5["script_sha256"]),
                        tfs=metas, seconds=round(time.time() - t0, 1)))
        crows, vrows, draws, metas = [], [], {}, {}


def write_part(out, stem, crows, vrows, draws, meta):
    os.makedirs(out, exist_ok=True)
    atomic_tsv(pd.DataFrame(crows), f"{out}/{stem}.contrast.tsv")
    atomic_tsv(pd.DataFrame(vrows), f"{out}/{stem}.vectors.tsv")
    tmp = f"{out}/{stem}.draws.tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, **draws)
    os.replace(tmp, f"{out}/{stem}.draws.npz")
    atomic_json(f"{out}/{stem}.meta.json", dict(meta, script_sha256=sha256(__file__), pins=PIN, predeclaration_sha256=PREDECL_SHA, python=sys.version.split()[0], numpy=np.__version__,
                                                  pandas=pd.__version__, host=platform.node(), created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    print(f"[nullp] wrote {out}/{stem}.*", flush=True)


# BM breadth
def run_breadth(a, out):
    import gb_layer_lattice as L
    import gb_ptable as PT
    t0 = time.time()
    d5 = d5_load("BONEMARROW")
    axis = d5_axis("BONEMARROW")
    U = PT.breadth_unit_bm(a.B, a.n_jobs)
    LD, t7, pf = U["LD"], U["t7"], U["pf"]
    n_all, nP = len(U["genes"]), len(U["perts"])
    pos = {g: i for i, g in enumerate(axis)}
    miss = [g for g in t7 if g not in pos]
    if miss:
        raise SystemExit(f"REFUSING: {len(miss)} breadth genes absent from the D5 BONEMARROW axis, e.g. {miss[:3]}")
    pi = np.array([pos[g] for g in t7])
    vr, vn = d5["P_real"][pi], d5["P_null"][:, pi]
    keep_all = U["mine"] & np.isfinite(vr) & np.isfinite(vn).all(0)
    units_by, _rc = PT.load_modules("BONEMARROW", t7, PT.PRIMARY_SIZE)
    wp = f"{LD.LAD}/final/draws_{PT.BM_SEED_TAG}_{n_all}.npy"
    rc = json.load(open(PT.BREADTH_RCPT))
    rec = [x for x in rc["analyses"] if x["estimand"] == "TOTAL" and x["arm"] == "BM19" and x["variant"] == "full"][0]
    if rec["n_mapped_to_bank"] != n_all or PT._sha(wp) != rec["draws_sha256"]:
        raise SystemExit("REFUSING_PARITY breadth: mapped genes / draws sha differ from the ladder receipt")
    Wg_all = np.asarray(np.load(wp, mmap_mode="r")[:a.B])
    bi_blk = L.BLOCKS.index("BONEMARROW")
    bm = L.BlockAxis(t7, a.B, int(np.random.SeedSequence(PT.mod_seed([L.BOOT_SEED, bi_blk, 20261002, 1], PT.PRIMARY_SIZE)).generate_state(1)[0]), units_by, "module")
    PW = np.random.default_rng(np.random.SeedSequence([LD.SEED, zlib.crc32(PT.BM_SEED_TAG.encode()), 1])).multinomial(nP, np.full(nP, 1.0 / nP), size=6000)[:a.B]
    jsm = list(L.REFS).index("REF -sm_det")
    sm = U["Xr"][:, jsm] * L.REF_SIGNS["REF -sm_det"][1]
    gbp = np.asarray(U["P"], float)
    crows, vrows, draws = [], [], {}
    for st, srows in tertiles(sm, keep_all).items():
        if srows is None:
            raise SystemExit("REFUSING: sm_det tertiles undefined on the breadth universe")
        keep = np.flatnonzero(srows)
        vecs = {"real": vr, **{n: vn[k] for k, n in enumerate(NUL)}, "elemMedianP": np.median(vn, 0)}
        if np.isfinite(gbp[keep]).all():
            vecs["GBP_fullbank"] = gbp
        restricted = {n: np.asarray(v, float)[keep] for n, v in vecs.items()}
        D = dict(models=list(restricted), fitted={}, raw=restricted, keep=keep, gf=U["folds"][keep], pf=pf, truth=tuple(x[:, keep] for x in (U["truth"]["T"], U["truth"]["R"], U["truth"]["E"], U["truth"]["CN"])),
                 perts=list(U["perts"]), tie=None)
        pt, dg = PT._breadth_draw_all(LD, D, Wg_all[:, keep], PW, a.n_jobs)
        _pt2, dm = PT._breadth_draw_all(LD, D, bm.counts[:, keep], PW, a.n_jobs)
        tg = LD.mk_targets(D["truth"][0], D["truth"][1], D["truth"][2], D["truth"][3], np.ones(nP, bool), LD.COVER_FLOOR, min_t=1)
        n_pos, n_neg = int((tg["any"] == 1).sum()), int((tg["any"] == 0).sum())
        idc = dict(kind="breadth", system="BM_BREADTH", tf="ALL", endpoint="any", block="", d5_system="BONEMARROW", gate_MR=True, n_U=n_all, n_MR=int(U["mine"].sum()), n_rows=int(keep_all.sum()),
                   cov_rows_of_MR=float(keep_all.sum() / max(U["mine"].sum(), 1)), draws_fp=f"ladder:{PT._sha(wp)[:12]}|{bm.fingerprint()[:12]}", d5_nT=d5["nT"], stratum=st, n_rows_stratum=int(len(keep)),
                   n_pos=n_pos, n_neg=n_neg, status="OK")
        pts = {n: {m: pt[n][k] for m, k in BMET.items()} for n in vecs}
        bs = {sch: {n: {m: np.array([d[n][k] for d in dd_], float) for m, k in BMET.items()} for n in vecs} for sch, dd_ in (("g", dg), ("m50", dm))}
        row, dd = contrast(pts, bs, tuple(BMET), idc)
        crows.append(row)
        vrows += vector_rows(pts, tuple(BMET), idc)
        draws.update({f"breadth|BM_BREADTH|ALL|any|{st}|{k}": v for k, v in dd.items()})
        print(f"[nullp-brd] {st}: rows={len(keep)} pos={n_pos} neg={n_neg} AUROC real={row['AUROC_real']:.4f} medNull={row['AUROC_mednull']:.4f} d={row['AUROC_delta']:+.4f} "
              f"[{row['AUROC_delta_m50_lo']:+.3f},{row['AUROC_delta_m50_hi']:+.3f}] rho_burden d={row['RHO_BURDEN_delta']:+.4f} {time.time() - t0:.0f}s", flush=True)
    write_part(out, "part_BREADTH_BM", crows, vrows, draws, dict(group="BREADTH", system="BM_BREADTH", B=a.B, d5=dict(system="BONEMARROW", shas=d5["shas"], subset_n=len(d5["subset"]), nT=d5["nT"],
                                                                                                                         script_sha256=d5["script_sha256"]), n_keep_ptable_MR=int(U["mine"].sum()),
                                                                 n_keep=int(keep_all.sum()), tfs={"ALL": dict(status="OK")}, seconds=round(time.time() - t0, 1)))


# selftest
# (1) real subset-P recomputed from the production catbank shards (D5's own functions, unchanged) == D5 P_real
def cmd_selftest(a):
    pins_ok()
    import gb_defense_recompute_v1_89c4c21b as M
    s = a.system.upper()
    d5 = d5_load(s)
    D, B, srcs, present, axis, au, src_axis, det = M.common_setup(s)
    assert srcs == d5["subset"], "subset differs from the D5 receipt"
    rr, _used = M.real_raw(s, D, B, srcs)
    M.blank_self(rr, src_axis)
    Pr, _Ur = M.primary_rows(rr, srcs, au)
    sp = M.subset_prior(Pr, src_axis, srcs)
    d = np.abs(sp["P"] - d5["P_real"])
    same_nan = np.array_equal(np.isfinite(sp["P"]), np.isfinite(d5["P_real"]))
    mx = float(np.nanmax(d))
    print(f"[selftest] {s}: real subset-P recomputed from the production shards vs D5 P_real: finiteness equal={same_nan} max|diff|={mx:.3e} (n_rows={sp['n_rows']}, nT={sp['nT']}, dropped={sp['dropped']})")
    assert same_nan and mx <= 1e-12, "P_real does NOT reproduce"
    res = {"P_real_max_abs_diff": mx, "P_real_n_rows": sp["n_rows"], "P_real_nT": sp["nT"]}
    worst, wrho = 0.0, 1.0
    for k in range(NPERM if a.allperm else 2):
        with np.load(f"{D5_DIR}/{s}_perm{k:02d}.npz") as z:
            Pn_rows, Pn = np.asarray(z["Prows_null"], float), np.asarray(z["P_null"], float)
        spn = M.subset_prior(Pn_rows, src_axis, srcs)
        worst = max(worst, float(np.nanmax(np.abs(spn["P"] - Pn))))
        wrho = min(wrho, M.spear(spn["P"], Pn))
    print(f"[selftest] {s}: null subset-P recomputed from the stored float32 null rows vs D5 P_null: max|diff|={worst:.3e}, min Spearman={wrho:.6f} ({NPERM if a.allperm else 2} perms)")
    res.update(P_null_max_abs_diff=worst, P_null_min_spearman=wrho)
    os.makedirs(a.out, exist_ok=True)
    atomic_json(f"{a.out}/selftest_{s}.json", dict(res, system=s, d5_shas=d5["shas"], script_sha256=sha256(__file__), predeclaration_sha256=PREDECL_SHA))
    print("[selftest] PASS")


# K562 D5 (frozen v1 script, reps subset)
# Run the unchanged frozen D5 script's cmd_d5 for K562 on a subset of permutation indices
def cmd_d5rep(a):
    pins_ok()
    import builtins
    import gb_defense_recompute_v1_89c4c21b as M
    reps = [int(x) for x in a.reps.split(",")]
    tagr = "-".join(str(r) for r in reps)

    class Sel(int):
        pass
    orig_range, orig_json, orig_rcpt = builtins.range, M.RU.atomic_json, M.receipt
    M.range = lambda *x: reps if (len(x) == 1 and isinstance(x[0], Sel)) else orig_range(*x)
    M.RU.atomic_json = lambda p, obj: orig_json(str(p).replace("_summary.json", f"_summary.reps{tagr}.json") if str(p).endswith("_summary.json") else p, obj)
    M.receipt = lambda path, system, extra: None if str(path).endswith("_summary.json") else orig_rcpt(path, system, extra)
    M.cmd_d5(argparse.Namespace(system="K562", nperm=Sel(NPERM), smalln=False, threads=a.threads))


def cmd_d5merge(a):
    import gb_defense_recompute_v1_89c4c21b as M
    fs = sorted(glob.glob(f"{D5_DIR}/K562_summary.reps*.json"))
    parts = [json.load(open(f)) for f in fs]
    perms = sorted((p for x in parts for p in x["perms"]), key=lambda p: p["rep"])
    if [p["rep"] for p in perms] != list(range(NPERM)):
        raise SystemExit(f"REFUSING: reps {[p['rep'] for p in perms]} != 0..{NPERM-1}")
    if any(json.dumps(x["real"], sort_keys=True) != json.dumps(parts[0]["real"], sort_keys=True) for x in parts):
        raise SystemExit("REFUSING: the real arm differs across the per-host summaries")
    rec = {"real": parts[0]["real"], "perms": perms}
    rec["agg"] = {"nullP_vs_realP": M.q3([p["spearman_nullP_vs_realP"] for p in perms if p["spearman_nullP_vs_realP"] is not None]),
                  "disjoint_half": M.q3([p["disjoint_half_P_spearman"] for p in perms if p["disjoint_half_P_spearman"] is not None]),
                  "varfrac_null": M.q3([p["subset_P"]["varfrac"] for p in perms if p["subset_P"]]),
                  "P_vs_detection_null": M.q3([p["subset_P"]["spearman_P_vs_detection"] for p in perms if p["subset_P"]])}
    rec["label"] = parts[0]["label"]
    rec["meta"] = dict(parts[0]["meta"], seconds_total=float(sum(x["meta"]["seconds_total"] for x in parts)),
                       merged_from=[os.path.basename(f) for f in fs], note="merged by gb_nullp_challenge.py d5merge from per-host runs of the frozen v1 script (seconds_total = sum over hosts)")
    M.RU.atomic_json(Path(D5_DIR) / "K562_summary.json", rec)
    print(f"[d5merge] K562_summary.json from {len(fs)} parts; agg nullP~realP median={rec['agg']['nullP_vs_realP']['median']}")


# merge / macro / readings
def label_of(delta, lo, hi, npos, nblk, loso=False):
    if not np.isfinite(delta):
        return "NA"
    if loso:
        return ("REVERSED (null >= real)" if delta <= 0 else "CI excludes 0" if lo > 0 else "CI includes 0") + f" [{npos}/{nblk} substrates positive; no 3/4 rule]"
    if delta <= 0:
        return "REVERSED (null >= real; module CI " + ("excludes" if hi < 0 else "includes") + " 0)"
    if lo > 0 and npos >= int(np.ceil(0.75 * nblk)):
        return "SUPPORTS alignment-driven utility"
    return "NOT SUPPORTED (module CI includes 0)" if lo <= 0 else f"NOT SUPPORTED (module CI excludes 0 but only {npos}/{nblk} substrates positive)"


# C: contrast rows
# V: long vector rows; draws: {part-less key -> (B,)}
# Macro over gated response units (macro block != ''), per metric and stratum
def macro_tables(C, V, draws, B):
    import gb_ptable as PT
    rows = []
    M = C[(C.status == "OK") & (C.kind == "lattice") & C.block.astype(str).ne("")]
    for st in STRATA:
        for m in ("AUROC", "APLIFT", "P50"):
            g = M[(M.stratum == st) & M[f"{m}_delta"].notna()]
            blocks = sorted(g.block.unique())
            if not blocks:
                continue
            bd, bp, bu = {}, {}, {}
            for b in blocks:
                gb_ = g[g.block == b]
                if gb_.draws_fp.nunique() != 1:
                    raise SystemExit(f"REFUSING_MACRO {b}/{st}: units of one block use different draws")
                un = [f"lattice|{r.system}|{r.tf}|resp|{st}" for r in gb_.itertuples()]
                bu[b] = [u.split("|")[1] + "/" + u.split("|")[2] for u in un]
                bp[b] = float(gb_[f"{m}_delta"].mean())
                for sch in SCHEMES:
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore", RuntimeWarning)
                        bd[(b, sch)] = np.nanmean(np.vstack([draws[f"{u}|{sch}|{m}"] for u in un]), 0)
            sets = {"MACRO": blocks, **{f"LOSO-{x}": [b for b in blocks if b != x] for x in blocks}} if len(blocks) > 1 else {"MACRO": blocks}
            for nm, bl in sets.items():
                r = dict(stratum=st, metric=m, set=nm, blocks=",".join(bl), n_blocks=len(bl), units="; ".join(f"{b}:{','.join(bu[b])}" for b in bl),
                         delta=float(np.mean([bp[b] for b in bl])), n_pos_blocks=int(sum(bp[b] > 0 for b in bl)), blocks_delta=";".join(f"{b}={bp[b]:+.4f}" for b in bl))
                for sch in SCHEMES:
                    d = np.nanmean(np.vstack([bd[(b, sch)] for b in bl]), 0)
                    sx = "" if sch == "g" else "_m50"
                    r[f"delta{sx}_lo"], r[f"delta{sx}_hi"] = PT.pct_ci(d)
                    r[f"p{sx}"] = PT.boot_p(d)
                r["label"] = (("[INCOMPLETE: %d/4 substrates] " % len(bl) if nm == "MACRO" and len(bl) < 4 else "") + label_of(r["delta"], r["delta_m50_lo"], r["delta_m50_hi"], r["n_pos_blocks"], len(bl), loso=nm.startswith("LOSO"))) if m == "AUROC" else ""
                rows.append(r)
    # per-permutation macro (secondary): macro of (real - null_k) over the same unit sets, point
    # estimates only
    V = V[(V.status == "OK") & (V.kind == "lattice") & V.block.astype(str).ne("")]
    for st in STRATA:
        for m in ("AUROC",):
            v = V[V.stratum == st]
            if not len(v):
                continue
            dk = []
            for k in NUL:
                blk = []
                for b, gb_ in v.groupby("block"):
                    x = gb_.pivot_table(index=["system", "tf"], columns="vector", values=m)
                    blk.append(float((x["real"] - x[k]).mean()))
                dk.append(float(np.mean(blk)))
            rows.append(dict(stratum=st, metric=m, set="PERMUTATION_MACRO", blocks=",".join(sorted(v.block.unique())), n_blocks=int(v.block.nunique()), delta=float(np.median(dk)),
                             n_pos_blocks=-1, units=f"frac_real_gt_null={np.mean(np.array(dk) > 0):.2f}; per-permutation macro Delta min/median/max={min(dk):+.4f}/{np.median(dk):+.4f}/{max(dk):+.4f}"))
    return pd.DataFrame(rows)


def f3(x, d=3):
    return "  nan " if x is None or not np.isfinite(x) else f"{x:+.{d}f}"


def cmd_merge(a):
    pins_ok()
    out = a.out
    mfs = sorted(glob.glob(f"{out}/part_*.meta.json"))
    if not mfs:
        raise SystemExit("REFUSING: no parts")
    metas = [json.load(open(f)) for f in mfs]
    if len({m["B"] for m in metas}) != 1:
        raise SystemExit("REFUSING_MIXED_PARTS: B differs")
    part_shas = sorted({m["script_sha256"] for m in metas})
    B = int(metas[0]["B"])
    C, V, draws = [], [], {}
    for f in mfs:
        stem = f.replace(".meta.json", "")
        C.append(pd.read_csv(f"{stem}.contrast.tsv", sep="\t", dtype={"tf": str}))
        V.append(pd.read_csv(f"{stem}.vectors.tsv", sep="\t", dtype={"tf": str}))
        with np.load(f"{stem}.draws.npz") as z:
            for k in z.files:
                if k in draws:
                    raise SystemExit(f"REFUSING_DUPLICATE_DRAWS {k}")
                draws[k] = z[k]
    C, V = pd.concat(C, ignore_index=True), pd.concat(V, ignore_index=True)
    for x in (C, V):
        x["block"] = x.block.fillna("").astype(str)                                  # blank block (outside the macro) is read back as NaN
    mac = macro_tables(C, V, draws, B)
    atomic_tsv(C, f"{out}/nullp_contrast.tsv")
    atomic_tsv(V, f"{out}/nullp_vectors_long.tsv")
    atomic_tsv(mac, f"{out}/nullp_macro.tsv")
    # selftest 2: P_real AUROC vs the full-bank GB P AUROC on the same rows, across endpoints (all stratum, lattice)
    L0 = C[(C.stratum == "all") & (C.kind == "lattice") & C.AUROC_real.notna()]
    gb = V[(V.stratum == "all") & (V.vector == "GBP_fullbank") & (V.kind == "lattice")].set_index(["system", "tf"]).AUROC
    L0 = L0.set_index(["system", "tf"]).join(gb.rename("AUROC_GBP"), how="left")
    ok = L0.AUROC_GBP.notna()
    from scipy.stats import spearmanr
    rho = float(spearmanr(L0.AUROC_real[ok], L0.AUROC_GBP[ok])[0]) if ok.sum() > 2 else np.nan
    par = dict(n_endpoints=int(ok.sum()), spearman_AUROC_Preal_vs_fullbankP=rho, max_abs_AUROC_diff=float((L0.AUROC_real[ok] - L0.AUROC_GBP[ok]).abs().max()) if ok.any() else np.nan,
               median_vector_spearman_Preal_vs_fullbankP=float(L0.spearman_Preal_vs_fullbankP.median()))
    # parity of the full-bank GB P AUROC with the ptable's own table, where the analysis rows == MR
    pt = []
    for f in sorted(glob.glob(f"{R}/results/real_uniform/ptable/full_20261002/part_lat_*.metrics.tsv")):
        x = pd.read_csv(f, sep="\t", dtype={"tf": str})
        pt.append(x[(x.predictor == "GB P") & (x["mask"] == "MR")][["system", "tf", "AUROC", "APLIFT", "P50", "n_rows"]])
    if pt:
        pt = pd.concat(pt).set_index(["system", "tf"])
        j = L0[L0.n_rows == L0.n_MR].join(pt, rsuffix="_ptable", how="inner")
        j = j[j.AUROC_GBP.notna()]
        par["parity_fullbankP_vs_ptable_n"] = int(len(j))
        par["parity_fullbankP_vs_ptable_max_abs"] = float((j.AUROC_GBP - j.AUROC).abs().max()) if len(j) else np.nan
    notrun = {f"{m['system']}|{k}": v["status"] for m in metas for k, v in m["tfs"].items() if str(v.get("status", "")).startswith("NOT_RUN")}
    st_self = {os.path.basename(f): json.load(open(f)) for f in sorted(glob.glob(f"{out}/selftest_*.json"))}
    d5info = {m["system"]: m["d5"] for m in metas}
    rc = dict(task="Target-prior utility versus alignment-destroyed null priors", predeclaration_doc=PREDECL, predeclaration_sha256=PREDECL_SHA, script=str(Path(__file__).resolve()), script_sha256=sha256(__file__), part_script_shas=part_shas,
              note=("parts were produced by the script versions in part_script_shas.  Changes between them: merge/report code; the optional --loo path (default off); and the sm_det-tertile guard "
                  "(K562: 74% of MR genes sit at sm_det=1.0 so the tertile cuts coincide -> STRATUM_UNDEFINED; the first K562 parts, which mislabelled the whole universe as 'det_T1', are archived in "
                  "superseded_k562_degenerate_tertile/ and were re-run).  PANC/GATA1/BM/BREADTH tertiles are non-degenerate, so the guard does not change their numbers."),
              k562_d5_provenance=("K562 uses 20 permutations from gb_defense_recompute_v1_89c4c21b.py, distributed with d5rep and combined with d5merge. "
                                  "Permutation 0 matches the reference run for P_null, P_real, Prows_null and perm."), pins=PIN, B=B, parts=[os.path.basename(f) for f in mfs], not_run=notrun, selftest=st_self, consistency_with_ptable=par, d5_inputs=d5info,
              shuffle_label=SHUFFLE_LABEL, readings=dict(SUPPORTS="macro Delta>0, module-block CI excludes 0, Delta>0 in >=3/4 substrates", NOT_SUPPORTED="module CI includes 0",
                                                         REVERSED="null >= real (macro Delta<=0)"), created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), host=platform.node(),
              python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__)
    atomic_json(f"{out}/receipt.json", rc)
    lines = report(C, V, mac, par, notrun)
    open(f"{out}/report.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


def report(C, V, mac, par, notrun):
    L = []
    L.append("### Real-P versus null-P.  Delta = real - median(20 null perms); CIs = 95% percentile, g = gene-identity, m = G/50 module-block.  frac = share of the 20 perms with real > null.\n")
    for st in STRATA:
        c = C[(C.stratum == st) & C.AUROC_real.notna()]
        if not len(c):
            continue
        L.append(f"## endpoints, stratum={st}")
        L.append("unit                          mask n/MR   pos | AUROC real  medNull   Delta  [g CI]            [m CI]            frac | APL real  medNull Delta | P50 real  medNull Delta | status")
        for r in c.sort_values(["system", "tf"]).itertuples():
            L.append(f"{r.system + '/' + r.tf:<29} {r.n_rows_stratum:>4}/{r.n_MR:<4} {r.n_pos:>4} | {r.AUROC_real:.3f}   {r.AUROC_mednull:.3f}  {f3(r.AUROC_delta)} [{f3(r.AUROC_delta_lo)},{f3(r.AUROC_delta_hi)}] [{f3(r.AUROC_delta_m50_lo)},{f3(r.AUROC_delta_m50_hi)}] {r.AUROC_frac_real_gt_null:.2f} | "
                     f"{r.APLIFT_real:.2f}  {r.APLIFT_mednull:.2f} {f3(r.APLIFT_delta, 2)} | {r.P50_real:.3f}  {r.P50_mednull:.3f} {f3(r.P50_delta)} | {r.status}")
        L.append("")
    b = C[(C.kind == "breadth") & C.AUROC_real.notna()]
    if len(b):
        L.append("## BM breadth (any-label AUROC; continuous rho_breadth / rho_burden)")
        for r in b.itertuples():
            L.append(f"{r.stratum:<7} n={r.n_rows_stratum} pos={r.n_pos} | AUROC {r.AUROC_real:.3f} medNull {r.AUROC_mednull:.3f} d={f3(r.AUROC_delta)} [g {f3(r.AUROC_delta_lo)},{f3(r.AUROC_delta_hi)}] [m {f3(r.AUROC_delta_m50_lo)},{f3(r.AUROC_delta_m50_hi)}] frac={r.AUROC_frac_real_gt_null:.2f} | "
                     f"APL d={f3(r.APLIFT_delta, 2)} | P50 d={f3(r.P50_delta)} | rho_breadth {r.RHO_BREADTH_real:+.3f}/{r.RHO_BREADTH_mednull:+.3f} d={f3(r.RHO_BREADTH_delta)} [m {f3(r.RHO_BREADTH_delta_m50_lo)},{f3(r.RHO_BREADTH_delta_m50_hi)}] | "
                     f"rho_burden {r.RHO_BURDEN_real:+.3f}/{r.RHO_BURDEN_mednull:+.3f} d={f3(r.RHO_BURDEN_delta)} [m {f3(r.RHO_BURDEN_delta_m50_lo)},{f3(r.RHO_BURDEN_delta_m50_hi)}]")
        L.append("")
    L.append("## substrate-balanced macro (gated response units; blocks PANC / ERYTHROID / BM / K562) + LOSO")
    for r in mac[mac.metric == "AUROC"].itertuples():
        if r.set == "PERMUTATION_MACRO":
            L.append(f"{r.stratum:<7} {r.set:<18} median per-perm macro Delta {f3(r.delta)}  ({r.units})")
        else:
            L.append(f"{r.stratum:<7} {r.set:<14} blocks={r.n_blocks} Delta {f3(r.delta)} [g {f3(r.delta_lo)},{f3(r.delta_hi)}] [m {f3(r.delta_m50_lo)},{f3(r.delta_m50_hi)}] p_m={r.p_m50:.3f} pos={r.n_pos_blocks}/{r.n_blocks} ({r.blocks_delta}) -> {r.label}")
    L.append("\n## macro, AP-lift and P50 (all rows)")
    for r in mac[(mac.metric != "AUROC") & (mac.stratum == "all") & (mac.set == "MACRO")].itertuples():
        L.append(f"{r.metric:<7} Delta {f3(r.delta, 3)} [g {f3(r.delta_lo)},{f3(r.delta_hi)}] [m {f3(r.delta_m50_lo)},{f3(r.delta_m50_hi)}] pos={r.n_pos_blocks}/{r.n_blocks}")
    L.append(f"\n## selftest 2 / consistency with the ptable: {json.dumps(par)}")
    if notrun:
        L.append(f"## NOT_RUN: {json.dumps(notrun)}")
    return L


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["selftest", "run", "d5rep", "d5merge", "merge"])
    ap.add_argument("--system", default="PANC")
    ap.add_argument("--group", default="PANC", choices=list(GROUPS))
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--tfs", default="")
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--nslice", type=int, default=1)
    ap.add_argument("--reps", default="")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--n-jobs", dest="n_jobs", type=int, default=8)
    ap.add_argument("--loo", action="store_true", help="SENSITIVITY S1: drop the focal TF's own row from the subset before building subset-P (lattice groups only)")
    ap.add_argument("--allperm", action="store_true", help="selftest: check all 20 null vectors (default 2)")
    a = ap.parse_args()
    if a.cmd == "selftest":
        cmd_selftest(a)
    elif a.cmd == "d5rep":
        cmd_d5rep(a)
    elif a.cmd == "d5merge":
        cmd_d5merge(a)
    elif a.cmd == "merge":
        cmd_merge(a)
    else:
        pins_ok()
        os.makedirs(a.out, exist_ok=True)
        run_breadth(a, a.out) if a.group == "BREADTH" else run_lattice_group(a, a.group, a.out)


if __name__ == "__main__":
    main()
