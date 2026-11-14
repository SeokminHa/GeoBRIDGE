#!/usr/bin/env python
"""Propagate calibrated bone-marrow labels through existing evaluations."""
from __future__ import annotations

import os
import sys


def _argv(flag, default):
    for i, a in enumerate(sys.argv):
        if a == flag and i + 1 < len(sys.argv):
            return sys.argv[i + 1]
    return default


_THREADS = _argv("--threads", "1")
if int(_THREADS) > 16:
    raise SystemExit("REFUSING: --threads > 16")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = _THREADS                                                    # before numpy (shared hosts)

import argparse
import glob
import hashlib
import json
import platform
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
R = __import__("os").environ["GEOBRIDGE_ROOT"]
for _p in (str(HERE), f"{R}/scratch"):
    if _p not in sys.path:
        sys.path.insert(0, _p)

PREDECL = f"{R}/internal_design_note_not_shipped_ROUNDC.md"
PREDECL_SHA = "4def53ed4f5dc3c13ae13993a831fbee46582745b1a1a6921cfc878da17dec37"
PINS = {"gb_calbm_reeval.py": "76bd0504", "gb_prior_adjusted.py": "1150425c", "gb_tableBC.py": "6c3a6f76", "gb_ptable.py": "5db6f77d"}
OUT_DEFAULT = f"{R}/results/real_uniform/calbm_propagate"
NULLP_DIR = f"{R}/results/real_uniform/nullp_challenge"                          # A1 parts + nullp_contrast.tsv / nullp_macro.tsv (historical)
PA_FULL = f"{R}/results/real_uniform/prior_adjusted/full_20261002"               # Table B/C inputs (cells, decoys, draws parts)
HISTSYS, CALSYS, D5S = "BM_CHIPATLAS", "BM_CHIPATLAS_CALTOT", "BONEMARROW"
GATED_CAL_EXPECTED = ("GATA1", "RUNX1")                                         # A2: only these two pass the calibrated total gate
MINC_FOURCLASS = 5                                                              # gb_fourclass_cluster_merged --minc default
SCRIPT_SHA = hashlib.sha256(open(__file__, "rb").read()).hexdigest()


# small utilities
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


def read_tsv(p, **kw):
    try:
        return pd.read_csv(p, sep="\t", **kw)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def f3(x, d=3):
    return "  nan " if x is None or not np.isfinite(x) else f"{x:+.{d}f}"


# gb_prior_adjusted.pick_decoys with the eligibility mask intersected with `allowed` (the stored historical decoy sources)
def restrict_pick(pick, allowed):
    allowed = np.asarray(allowed, bool)
    return lambda F, r, elig, n: pick(F, r, np.asarray(elig, bool) & allowed, n)


# draw-dict keys 'lattice|<system>|...' -> new system label (first occurrence only)
def rename_system_keys(d, old, new):
    return {k.replace(f"lattice|{old}|", f"lattice|{new}|", 1): v for k, v in d.items()}


# historical rows of `system` replaced by the calibrated rows Cc (same schema); every other row untouched
def replace_system_rows(C0, Cc, system):
    return pd.concat([C0[C0.system != system], Cc], ignore_index=True)


def receipt_base(task, M, extra=None):
    rc = dict(task=task, script=str(Path(__file__).resolve()), script_sha256=SCRIPT_SHA, predeclaration=PREDECL, predeclaration_sha256=PREDECL_SHA,
              pins={fn: sha256(f"{R}/scratch/{fn}") for fn in PINS}, nullp_script_sha256=sha256(f"{R}/scratch/gb_nullp_challenge.py"), dep_shas=M["PT"].dep_shas(), host=platform.node(),
              python=sys.version.split()[0], numpy=np.__version__, pandas=pd.__version__, threads=_THREADS, created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), argv=sys.argv)
    rc.update(extra or {})
    return rc


# preflight / registration
# add the CAL_TOTAL system (BM_CHIPATLAS_CALTOT) to the lattice registry
def register_cal_arm(CB):
    import gb_layer_lattice as L
    import gb_lattice_extra as LE
    c = CB.ARMS["CAL_TOTAL"]
    assert c["system"] == CALSYS, c
    h = LE.EXTRA[HISTSYS]
    LE.EXTRA[CALSYS] = dict(block=h["block"], axis=h["axis"], tfs=list(h["tfs"]), base=h["base"], win=h["win"], role=c["role"],
                            truth=(lambda tf, g: CB.truth_cal(tf, g, c["est"], c["rule"])))
    LE.register()
    L.SYSTEMS.setdefault(CALSYS, dict(block=h["block"], axis=h["axis"], tfs=list(h["tfs"])))
    L.WIN.setdefault(CALSYS, h["win"])


# Verify analysis-plan and source-file hashes.
def preflight(need_nullp=False):
    if __import__("os").environ.get("GB_SKIP_PREFLIGHT") == "1":      # opt-in (public release): skip only the brief/sha pins
        print("preflight skipped (public release)")
    else:
        if not os.path.exists(PREDECL) or sha256(PREDECL) != PREDECL_SHA:
            raise SystemExit(f"REFUSING: pre-declaration {PREDECL} missing or sha256 != {PREDECL_SHA[:8]}")
        for fn, pre in PINS.items():
            got = sha256(f"{R}/scratch/{fn}")
            if not got.startswith(pre):
                raise SystemExit(f"REFUSING: {fn} sha {got[:8]} != pinned {pre}")
    import gb_calbm_reeval as CB
    PT = CB.preflight()                                                          # Verify the analysis plan and gb_ptable source hash.
    import gb_nullp_challenge as NP
    import gb_prior_adjusted as PA
    import gb_tableBC as TB
    if need_nullp:
        NP.pins_ok()
    register_cal_arm(CB)
    return dict(CB=CB, PT=PT, PA=PA, TB=TB, NP=NP)


# the draws of the historical BM_CHIPATLAS lattice system ([gene; module G/50]); identical for the calibrated system (same block / axis -> same seeds)
def mkdr(PT, B):
    dr, _ = PT.lattice_draws(HISTSYS, B, sizes=(PT.PRIMARY_SIZE,))
    assert dr.schemes == ("g", "m50") and dr.C.shape[0] == 2 * B, (dr.schemes, dr.C.shape)
    return dr


def roster_of(L, LE, AD):
    return list(AD.tf_list_audit(HISTSYS, L, LE))


# (a) A1
# One BM unit of the A1 challenge
def a1_unit(M, system, tf, dr, B, d5, jsm, ssm):
    PT, NP = M["PT"], M["NP"]
    u = PT.lattice_unit(system, tf, dr.gene)
    gi, y, MR = u["gi"], np.asarray(u["y"], bool), np.asarray(u["MR"], bool)
    vr, vn = d5["P_real"][gi], d5["P_null"][:, gi]
    fin = np.isfinite(vr) & np.isfinite(vn).all(0)
    rows = MR & fin
    sm = u["Xr"][:, jsm] * ssm
    gbp = np.where(np.isfinite(u["P"]), u["P"], np.nan)
    vecs = {"real": vr, **{n: vn[k] for k, n in enumerate(NP.NUL)}}
    base = dict(kind="lattice", system=system, tf=tf, endpoint="resp", block=PT.MACRO_BLOCK[HISTSYS], d5_system=D5S, gate_MR=bool(u["gate"]["MR"]), n_U=int(u["U"].sum()), n_MR=int(MR.sum()),
                n_rows=int(rows.sum()), cov_rows_of_MR=float(rows.sum() / max(MR.sum(), 1)), draws_fp=dr.fp, d5_nT=d5["nT"], loo_focal=False, focal_row_dropped=False)
    crows, vrows, draws = [], [], {}
    for st, srows in NP.tertiles(sm, rows).items():
        if srows is None:
            crows.append(dict(base, stratum=st, status="STRATUM_UNDEFINED: sm_det saturated (tertile cuts coincide)", n_rows_stratum=0, n_pos=0, n_neg=0))
            continue
        yy = y[srows]
        idc = dict(base, stratum=st, n_rows_stratum=int(srows.sum()), n_pos=int(yy.sum()), n_neg=int((~yy).sum()))
        if yy.sum() < NP.MINC or (~yy).sum() < NP.MINC:
            crows.append(dict(idc, status="TOO_FEW"))
            continue
        allv = dict(vecs, elemMedianP=np.median(vn, 0))
        res = {n: PT.split_eval(v, y, srows, gi, dr.C, B) for n, v in allv.items()}
        if np.isfinite(gbp[srows]).all():
            res["GBP_fullbank"] = PT.split_eval(gbp, y, srows, gi, dr.C, B)
        pts = {n: r[0] for n, r in res.items()}
        bs = {sch: {n: r[1][si] for n, r in res.items()} for si, sch in enumerate(NP.SCHEMES)}
        row, dd = NP.contrast(pts, bs, NP.MET, idc)
        row["status"] = "OK" if base["gate_MR"] else "DESCRIPTIVE_ONLY"
        row["spearman_Preal_vs_fullbankP"] = float(pd.Series(vr[srows]).corr(pd.Series(gbp[srows]), method="spearman")) if np.isfinite(gbp[srows]).all() else np.nan
        crows.append(row)
        vrows += NP.vector_rows(pts, NP.MET, dict(idc, status=row["status"]))
        draws.update({f"lattice|{system}|{tf}|resp|{st}|{k}": v for k, v in dd.items()})
    return u, crows, vrows, draws


def write_part(out, stem, crows, vrows, draws, meta, M):
    os.makedirs(out, exist_ok=True)
    if crows:
        atomic_tsv(pd.DataFrame(crows), f"{out}/{stem}.contrast.tsv")
        atomic_tsv(pd.DataFrame(vrows), f"{out}/{stem}.vectors.tsv")
        tmp = f"{out}/{stem}.draws.tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, **draws)
        os.replace(tmp, f"{out}/{stem}.draws.npz")
    atomic_json(f"{out}/{stem}.meta.json", dict(meta, script_sha256=SCRIPT_SHA, predeclaration_sha256=PREDECL_SHA, nullp_script_sha256=sha256(f"{R}/scratch/gb_nullp_challenge.py"),
                                                dep_shas=M["PT"].dep_shas(), host=platform.node(), python=sys.version.split()[0], created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
    print(f"[calbm-prop] wrote {out}/{stem}.*", flush=True)


def cmd_a1(a):
    M = preflight(need_nullp=True)
    PT, NP, CB = M["PT"], M["NP"], M["CB"]
    import gb_audit_decomp as AD
    import gb_layer_lattice as L
    import gb_lattice_extra as LE
    out = f"{a.out}/a1"
    os.makedirs(out, exist_ok=True)
    arms = [x.strip().upper() for x in a.arms.split(",") if x.strip()]
    assert set(arms) <= {"CAL", "HIST"}, arms
    d5, axis = NP.d5_load(D5S), NP.d5_axis(D5S)
    dr = mkdr(PT, a.B)
    if len(axis) != d5["G"] or not np.array_equal(np.asarray(dr.gene.names).astype(str), axis):
        raise SystemExit(f"REFUSING: block axis of {HISTSYS} differs from the D5 {D5S} axis")
    jsm, ssm = list(L.REFS).index("REF -sm_det"), L.REF_SIGNS["REF -sm_det"][1]
    allowed = roster_of(L, LE, AD)
    tfs = [t.strip() for t in a.tfs.split(",") if t.strip()] or allowed
    bad = [t for t in tfs if t not in allowed]
    if bad:
        raise SystemExit(f"REFUSING: --tfs {bad} not in the audit roster {allowed} of {HISTSYS}")
    if a.nslice > 1:
        tfs = [t for i, t in enumerate(tfs) if i % a.nslice == a.slice]
    for tf in tfs:
        for arm in arms:
            system = CALSYS if arm == "CAL" else HISTSYS
            stem, t1 = f"part_{arm}_{tf}", time.time()
            try:
                u, crows, vrows, draws = a1_unit(M, system, tf, dr, a.B, d5, jsm, ssm)
            except (L.Blocked, SystemExit) as e:
                if isinstance(e, SystemExit) and "REFUSING_PARITY" in str(e):
                    raise
                write_part(out, stem, [], [], {}, dict(arm=arm, system=system, tf=tf, B=a.B, status=f"NOT_RUN: {type(e).__name__}: {e}"), M)
                print(f"[calbm-prop a1] {arm}/{tf}: NOT_RUN {e}", flush=True)
                continue
            if arm == "CAL":                                                     # truth-free vectors must be bit-identical to the historical unit (only the truth differs)
                uh = PT.lattice_unit(HISTSYS, tf, dr.gene)
                if not CB.truth_free_equal(uh, u):
                    raise SystemExit(f"REFUSING: truth-free vectors of {CALSYS}/{tf} differ from {HISTSYS}")
            r0 = next((r for r in crows if r.get("stratum") == "all" and "AUROC_real" in r), None)
            meta = dict(arm=arm, system=system, tf=tf, B=a.B, status="RUN", gate_MR=bool(u["gate"]["MR"]), n_MR=int(u["MR"].sum()), n_pos_MR=int(np.asarray(u["y"], bool)[u["MR"]].sum()),
                        draws_fp=dr.fp, d5=dict(shas=d5["shas"], subset_n=len(d5["subset"]), nT=d5["nT"], script_sha256=d5["script_sha256"]), seconds=round(time.time() - t1, 1))
            write_part(out, stem, crows, vrows, draws, meta, M)
            if r0 is not None:
                print(f"[calbm-prop a1] {arm}/{tf}: MR={meta['n_MR']} pos={meta['n_pos_MR']} gate={meta['gate_MR']} AUROC real={r0['AUROC_real']:.4f} medNull={r0['AUROC_mednull']:.4f} "
                      f"d={r0['AUROC_delta']:+.4f} frac={r0['AUROC_frac_real_gt_null']:.2f} {time.time() - t1:.0f}s", flush=True)
    assert sha256(__file__) == SCRIPT_SHA, "script changed during the run"


# (metas, contrast, vectors, draws) of every part under d (same loader semantics as gb_nullp_challenge.cmd_merge)
def load_parts(d, prefix="part_"):
    metas, C, V, D = [], [], [], {}
    for f in sorted(glob.glob(f"{d}/{prefix}*.meta.json")):
        m = json.load(open(f))
        metas.append(m)
        stem = f[:-len(".meta.json")]
        if not os.path.exists(f"{stem}.contrast.tsv"):
            continue
        C.append(read_tsv(f"{stem}.contrast.tsv", dtype={"tf": str}))
        V.append(read_tsv(f"{stem}.vectors.tsv", dtype={"tf": str}))
        with np.load(f"{stem}.draws.npz") as z:
            for k in z.files:
                if k in D:
                    raise SystemExit(f"REFUSING_DUPLICATE_DRAWS {k}")
                D[k] = z[k]
    C, V = (pd.concat(x, ignore_index=True) if x else pd.DataFrame() for x in (C, V))
    for x in (C, V):
        if len(x):
            x["block"] = x.block.fillna("").astype(str)
    return metas, C, V, D


# equal-weight mean over `units` [(system, tf)] of the real-minus-null-median Delta, per draw, for AUROC / APLIFT / P50: point, gene CI
def block_summary(PT, C, D, units, label, st="all"):
    rows = []
    for m in ("AUROC", "APLIFT", "P50"):
        g = [C[(C.system == s) & (C.tf == t) & (C.stratum == st)] for s, t in units]
        if not units or any(len(x) != 1 or not np.isfinite(x[f"{m}_delta"].iloc[0]) for x in g):
            rows.append(dict(set=label, metric=m, n_units=len(units), units=";".join(f"{s}:{t}" for s, t in units), status="UNAVAILABLE"))
            continue
        r = dict(set=label, metric=m, n_units=len(units), units=";".join(f"{s}:{t}" for s, t in units), delta=float(np.mean([x[f"{m}_delta"].iloc[0] for x in g])), status="OK")
        for sch, sx in (("g", ""), ("m50", "_m50")):
            d = np.nanmean(np.vstack([D[f"lattice|{s}|{t}|resp|{st}|{sch}|{m}"] for s, t in units]), 0)
            r[f"delta{sx}_lo"], r[f"delta{sx}_hi"] = PT.pct_ci(d)
            r[f"p{sx}"] = PT.boot_p(d)
        rows.append(r)
    return rows


def macro_lines(mac, tag):
    L_ = [f"## {tag}: substrate-balanced macro (AUROC Delta = real - median null) + LOSO"]
    for r in mac[mac.metric == "AUROC"].itertuples():
        if r.set == "PERMUTATION_MACRO":
            L_.append(f"{r.stratum:<7} {r.set:<18} median per-perm macro Delta {f3(r.delta)}  ({r.units})")
        else:
            L_.append(f"{r.stratum:<7} {r.set:<14} blocks={r.n_blocks} Delta {f3(r.delta)} [g {f3(r.delta_lo)},{f3(r.delta_hi)}] [m {f3(r.delta_m50_lo)},{f3(r.delta_m50_hi)}] p_m={r.p_m50:.3f} "
                      f"pos={r.n_pos_blocks}/{r.n_blocks} ({r.blocks_delta}) -> {r.label}")
    for r in mac[(mac.metric != "AUROC") & (mac.stratum == "all") & (mac.set == "MACRO")].itertuples():
        L_.append(f"{r.metric:<7} Delta {f3(r.delta)} [g {f3(r.delta_lo)},{f3(r.delta_hi)}] [m {f3(r.delta_m50_lo)},{f3(r.delta_m50_hi)}] pos={r.n_pos_blocks}/{r.n_blocks}")
    return L_


def cmd_a1merge(a):
    M = preflight(need_nullp=True)
    PT, NP = M["PT"], M["NP"]
    m0, C0, V0, D0 = load_parts(NULLP_DIR)
    mc, Cc, Vc, Dc = load_parts(f"{a.out}/a1", prefix="part_CAL_")
    if not m0 or not mc or not len(Cc):
        raise SystemExit("REFUSING: historical A1 parts or calibrated parts missing")
    B0, Bc = {int(m["B"]) for m in m0}, {int(m["B"]) for m in mc}
    if len(B0) != 1 or B0 != Bc:
        raise SystemExit(f"REFUSING: B differs (historical {B0}, calibrated {Bc}); paired macros need identical draws")
    B = B0.pop()
    if {m["draws_fp"] for m in mc if m.get("status") == "RUN"} != set(Cc.draws_fp.unique()) or len(set(Cc.draws_fp.unique()) & set(C0[C0.system == HISTSYS].draws_fp.unique())) != 1:
        raise SystemExit("REFUSING: calibrated units do not share the historical BM draws fingerprint")
    # historical macro (rebuilt from the stored parts) must equal the stored nullp_macro.tsv (fail closed)
    mac_h = NP.macro_tables(C0, V0, D0, B)
    ref = read_tsv(f"{NULLP_DIR}/nullp_macro.tsv")
    par = dict(status="NO_STORED_nullp_macro.tsv")
    if len(ref):
        j = mac_h.merge(ref, on=["stratum", "metric", "set"], suffixes=("", "_stored"))
        dd = (j.delta - j.delta_stored).abs()
        par = dict(status="PASS" if float(np.nanmax(dd)) <= 1e-9 else "FAIL", n_rows=int(len(j)), max_abs_delta=float(np.nanmax(dd)),
                   max_abs_ci=float(np.nanmax(np.r_[(j.delta_lo - j.delta_lo_stored).abs().to_numpy(float), (j.delta_m50_hi - j.delta_m50_hi_stored).abs().to_numpy(float)])) if "delta_lo_stored" in j else np.nan)
        if par["status"] != "PASS":
            raise SystemExit(f"REFUSING_PARITY A1 historical macro rebuilt from parts != stored nullp_macro.tsv: {par}")
    # calibrated BM replaces historical BM (system BM_CHIPATLAS) in the block macro
    C1, V1, D1 = replace_system_rows(C0, Cc, HISTSYS), replace_system_rows(V0, Vc, HISTSYS), {**D0, **Dc}
    mac_c = NP.macro_tables(C1, V1, D1, B)
    # BM-substrate-only sets (gated units): historical own, historical restricted to the calibrated-gated TFs (like-for-like), calibrated
    gh = sorted(C0[(C0.system == HISTSYS) & (C0.stratum == "all") & (C0.status == "OK")].tf)
    gc = sorted(Cc[(Cc.stratum == "all") & (Cc.status == "OK")].tf)
    both = sorted(set(gh) & set(gc))
    bm = (block_summary(PT, C0, D0, [(HISTSYS, t) for t in gh], "BM_HIST_own_gated") + block_summary(PT, C0, D0, [(HISTSYS, t) for t in both], "BM_HIST_on_calibrated_gated_TFs")
          + block_summary(PT, Cc, Dc, [(CALSYS, t) for t in gc], "BM_CAL_TOTAL_own_gated"))
    cmp_ = pd.concat([C0[C0.system == HISTSYS].assign(arm="HIST"), Cc.assign(arm="CAL_TOTAL")], ignore_index=True)
    keep = [c for c in ("arm", "tf", "stratum", "status", "gate_MR", "n_MR", "n_rows", "n_pos", "n_neg", "AUROC_real", "AUROC_mednull", "AUROC_delta", "AUROC_delta_lo", "AUROC_delta_hi", "AUROC_delta_m50_lo",
                        "AUROC_delta_m50_hi", "AUROC_frac_real_gt_null", "APLIFT_delta", "P50_delta") if c in cmp_]
    os.makedirs(a.out, exist_ok=True)
    atomic_tsv(cmp_[keep], f"{a.out}/calbm_a1_unit_contrast_hist_vs_cal.tsv")
    atomic_tsv(mac_h, f"{a.out}/calbm_a1_macro_HIST.tsv")
    atomic_tsv(mac_c, f"{a.out}/calbm_a1_macro_CALBM.tsv")
    atomic_tsv(pd.DataFrame(bm), f"{a.out}/calbm_a1_bm_substrate.tsv")
    atomic_tsv(C1, f"{a.out}/calbm_a1_contrast_CALBM_all.tsv")
    notrun = {m["tf"]: m["status"] for m in mc if str(m.get("status", "")).startswith("NOT_RUN")}
    L_ = ["### Real-P versus alignment-destroyed null-P with calibrated BM (CAL_TOTAL) replacing historical BM.  Delta = real - median(20 null perms); g = gene-identity CI, m = G/50 module-block CI.",
          f"calibrated-gated BM TFs: {gc} (expected {list(GATED_CAL_EXPECTED)}); historical-gated BM TFs: {gh}; NOT_RUN calibrated: {json.dumps(notrun)}",
          f"parity: historical macro rebuilt from stored parts vs stored nullp_macro.tsv: {json.dumps(par)}", ""]
    if tuple(gc) != GATED_CAL_EXPECTED:
        L_.append(f"Gated TF set differs: expected {list(GATED_CAL_EXPECTED)}, observed {gc}.")
    L_.append("## BM substrate only (equal-weight over the units of the set)")
    for r in bm:
        L_.append(f"{r['set']:<34} {r['metric']:<7} n={r['n_units']} " + (f"Delta {f3(r['delta'])} [g {f3(r['delta_lo'])},{f3(r['delta_hi'])}] [m {f3(r['delta_m50_lo'])},{f3(r['delta_m50_hi'])}] "
                                                                                 f"p_m={r['p_m50']:.3f}  units={r['units']}" if r["status"] == "OK" else "UNAVAILABLE"))
    L_ += [""] + macro_lines(mac_h, "HISTORICAL BM") + [""] + macro_lines(mac_c, "CALIBRATED BM replacing historical BM")
    L_.append("\n## per-unit BM contrast, historical vs calibrated (stratum all)")
    for r in cmp_[cmp_.stratum == "all"].sort_values(["tf", "arm"]).itertuples():
        if hasattr(r, "AUROC_delta") and np.isfinite(r.AUROC_delta):
            L_.append(f"{r.tf:<7} {r.arm:<9} gate={str(r.gate_MR):<5} n={r.n_rows}/{r.n_MR} pos={r.n_pos} Delta {f3(r.AUROC_delta)} [g {f3(r.AUROC_delta_lo)},{f3(r.AUROC_delta_hi)}] "
                      f"[m {f3(r.AUROC_delta_m50_lo)},{f3(r.AUROC_delta_m50_hi)}] frac={r.AUROC_frac_real_gt_null:.2f}")
    open(f"{a.out}/report_a1.txt", "w").write("\n".join(L_) + "\n")
    atomic_json(f"{a.out}/receipt_a1.json", receipt_base("Target-prior comparison with calibrated BM labels", M, dict(B=B, parity_hist_macro=par, gated_cal=gc, gated_hist=gh, not_run_cal=notrun, parts_cal=[m.get("tf") for m in mc],
                                                                                                     n_hist_parts=len(m0), readings="SUPPORTS: macro Delta>0, module CI excludes 0, >=3/4 substrates positive; NOT SUPPORTED: module CI includes 0; REVERSED: macro Delta<=0",
                                                                                                     historical_reported_alongside=True)))
    print("\n".join(L_))


# (b) Table B / C
# {(tf, method): [decoy names in historical match order]} of the historical BM_CHIPATLAS response cells (Table B full run)
def stored_sources(d):
    q = d[(d.kind == "lattice") & (d.system == HISTSYS) & (d.target == "resp")].sort_values("match_rank")
    return {(str(tf), m): list(g.decoy) for (tf, m), g in q.groupby(["tf", "method"], sort=False)}


# Reuse eligible stored decoys per TF/method; fall back to fresh selection when absent
def install_stored_decoys(PA, stored, acct):
    orig_prepare, orig_pick = PA.prepare_decoys, PA.pick_decoys

    def prepare(u, ctx, m, rows, n_decoy=PA.N_DECOY):
        names = stored.get((str(u["tf"]), m))
        if names is None:
            out = orig_prepare(u, ctx, m, rows, n_decoy)
            acct[(str(u["tf"]), m)] = dict(source="FRESH_NO_STORED", n_stored=0, n_used=int(len(out.get("sel", []))), n_dropped=0)
            return out
        idx = {mm: i for i, mm in enumerate(u["bank"].members)}
        miss = [x for x in names if x not in idx]
        if miss:
            raise SystemExit(f"REFUSING: stored decoys {miss[:3]} of {u['tf']}/{m} are not members of the bank")
        allowed = np.zeros(len(u["bank"].members), bool)
        allowed[[idx[x] for x in names]] = True
        PA.pick_decoys = restrict_pick(orig_pick, allowed)
        try:
            out = orig_prepare(u, ctx, m, rows, n_decoy)
        finally:
            PA.pick_decoys = orig_pick
        sel = [u["bank"].members[int(s)] for s in out.get("sel", [])]
        acct[(str(u["tf"]), m)] = dict(source="STORED_HIST", n_stored=len(names), n_used=len(sel), n_dropped=len(names) - len(sel), same_set_as_stored=bool(set(sel) == set(names)),
                                       status=out.get("status"))
        return out

    PA.prepare_decoys = prepare

    def undo():
        PA.prepare_decoys, PA.pick_decoys = orig_prepare, orig_pick
    return undo


def tb_ns(PA, B, n_jobs):
    return types.SimpleNamespace(B=B, secondary=[], no_parity=False, no_decoy=False, n_decoy=PA.N_DECOY, n_jobs=n_jobs, loto=False, loto_only=False)


# frozen gb_prior_adjusted.process_unit on the response target only (LAT_TARGETS rebound for this process), decoys = stored historical sources
def tb_unit(M, system, tf, dr, B, n_jobs, stored=None):
    PA, PT = M["PA"], M["PT"]
    u = PT.lattice_unit(system, tf, dr.gene)
    acct, draws, dec, sec, loto = {}, {}, [], [], {}
    lat0 = PA.LAT_TARGETS
    PA.LAT_TARGETS = ["resp"]
    undo = install_stored_decoys(PA, stored, acct) if stored is not None else (lambda: None)
    try:
        rows_out, par = PA.process_unit(u, dr, tb_ns(PA, B, n_jobs), sec, loto, draws, dec)
    finally:
        undo()
        PA.LAT_TARGETS = lat0
    return u, rows_out, dec, draws, par, acct


def cmd_tb(a):
    M = preflight()
    PT, PA, CB, TB = M["PT"], M["PA"], M["CB"], M["TB"]
    import gb_audit_decomp as AD
    import gb_layer_lattice as L
    import gb_lattice_extra as LE
    out = f"{a.out}/tb"
    os.makedirs(out, exist_ok=True)
    d = read_tsv(f"{PA_FULL}/table2_decoys.tsv", dtype={"tf": str})
    stored = stored_sources(d)
    if not stored:
        raise SystemExit(f"REFUSING: no stored historical {HISTSYS} response decoys in {PA_FULL}/table2_decoys.tsv")
    dr = mkdr(PT, a.B)
    allowed = roster_of(L, LE, AD)
    tfs = [t.strip() for t in a.tfs.split(",") if t.strip()] or allowed
    bad = [t for t in tfs if t not in allowed]
    if bad:
        raise SystemExit(f"REFUSING: --tfs {bad} not in the audit roster {allowed} of {HISTSYS}")
    if a.nslice > 1:
        tfs = [t for i, t in enumerate(tfs) if i % a.nslice == a.slice]
    for tf in tfs:
        t1, stem = time.time(), f"part_CAL_{tf}"
        try:
            u, rows_out, dec, draws, par, acct = tb_unit(M, CALSYS, tf, dr, a.B, a.n_jobs, stored)
        except (L.Blocked, SystemExit) as e:
            if isinstance(e, SystemExit) and "REFUSING_PARITY" in str(e):
                raise
            atomic_json(f"{out}/{stem}.meta.json", dict(stem=stem, tf=tf, B=a.B, status=f"NOT_RUN: {type(e).__name__}: {e}", script_sha256=SCRIPT_SHA, predeclaration_sha256=PREDECL_SHA))
            print(f"[calbm-prop tb] {tf}: NOT_RUN {e}", flush=True)
            continue
        uh = PT.lattice_unit(HISTSYS, tf, dr.gene)
        if not CB.truth_free_equal(uh, u):
            raise SystemExit(f"REFUSING: truth-free vectors of {CALSYS}/{tf} differ from {HISTSYS}")
        for r in rows_out:                                                       # decoy-source accounting on the cells table (never silent)
            x = acct.get((tf, r["method"]))
            if x:
                r.update(decoy_source=x["source"], n_decoy_stored=x["n_stored"], n_decoy_dropped_ineligible=x["n_dropped"])
        atomic_tsv(pd.DataFrame(rows_out), f"{out}/{stem}.cells.tsv")
        if dec:
            atomic_tsv(pd.DataFrame(dec), f"{out}/{stem}.decoys.tsv")
        tmp = f"{out}/{stem}.draws.tmp{os.getpid()}.npz"
        np.savez_compressed(tmp, **draws)
        os.replace(tmp, f"{out}/{stem}.draws.npz")
        gate = bool(rows_out and rows_out[0].get("gate"))
        atomic_json(f"{out}/{stem}.meta.json", dict(stem=stem, tf=tf, B=a.B, status="RUN", system=CALSYS, gate_M1_resp=gate, n_M1=int(u["M1"].sum()), parity=par, draws_fp=dr.fp,
                                                    decoy_accounting={f"{k[0]}|{k[1]}": v for k, v in acct.items()}, script_sha256=SCRIPT_SHA, predeclaration_sha256=PREDECL_SHA,
                                                    pa_sha256=PA.SCRIPT_SHA0, ptable_sha256=PT.SCRIPT_SHA0, tableBC_sha256=sha256(f"{R}/scratch/gb_tableBC.py"), dep_shas=PT.dep_shas(),
                                                    seconds=round(time.time() - t1, 1), host=platform.node(), created=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        nd = [v for k, v in acct.items() if k[0] == tf]
        print(f"[calbm-prop tb] {tf}: M1={int(u['M1'].sum())} gate={gate} cells={len(rows_out)} decoy sources "
              f"{sorted({x['source'] for x in nd})} dropped_ineligible max={max([x['n_dropped'] for x in nd] or [0])} {time.time() - t1:.0f}s", flush=True)
    assert sha256(__file__) == SCRIPT_SHA, "script changed during the run"


def load_hist_tb():
    c = read_tsv(f"{PA_FULL}/table2_cells.tsv", dtype={"tf": str})
    d = read_tsv(f"{PA_FULL}/table2_decoys.tsv", dtype={"tf": str})
    dr = {}
    for f in sorted(glob.glob(f"{PA_FULL}/part_lat_*.draws.npz")):
        with np.load(f) as z:
            for k in z.files:
                if k.startswith("lattice|") and "|resp|" in k and "|AUROC|" in k:     # tableBC.macro_tables reads AUROC draws of the response target only
                    dr[k] = z[k]
    return c, d, dr


# per method: response prior strength AUROC(P)-.5 and calibrated E, 4-substrate macro (gene / module CI) and the BM-substrate-only values
def table_c_like(mac, sub, tag):
    rows = []
    for m in mac.method.unique():
        r = dict(variant=tag, method=m)
        for q, nm in (("P", "prior_strength"), ("E", "E")):
            x = mac[(mac.method == m) & (mac.quantity == q) & (mac.target == "resp")]
            if len(x):
                x = x.iloc[0]
                r.update({f"{nm}_macro": x.point, f"{nm}_macro_gene_lo": x.gene_lo, f"{nm}_macro_gene_hi": x.gene_hi, f"{nm}_macro_module_lo": x.module_lo, f"{nm}_macro_module_hi": x.module_hi,
                          "n_substrates": x.n_substrates, "n_units": x.n_units})
        s = sub[(sub.method == m) & (sub.target == "resp") & (sub.substrate == "BONEMARROW")]
        if len(s):
            s = s.iloc[0]
            r.update(BM_n_units=s.n_units, BM_units=s.units, BM_prior_strength=s.P_point, BM_prior_strength_gene_lo=s.P_gene_lo, BM_prior_strength_gene_hi=s.P_gene_hi, BM_E=s.E_point,
                     BM_E_gene_lo=s.E_gene_lo, BM_E_gene_hi=s.E_gene_hi, BM_E_module_lo=s.E_module_lo, BM_E_module_hi=s.E_module_hi)
        rows.append(r)
    return pd.DataFrame(rows)


def cmd_tbmerge(a):
    M = preflight()
    PT, TB = M["PT"], M["TB"]
    c_h, d_h, dr_h = load_hist_tb()
    c_h = c_h[(c_h.kind == "lattice") & (c_h.target == "resp")].copy()
    ms = [json.load(open(f)) for f in sorted(glob.glob(f"{a.out}/tb/part_CAL_*.meta.json"))]
    if not ms:
        raise SystemExit("REFUSING: no calibrated Table B parts")
    if len({m["B"] for m in ms}) != 1 or len({m["script_sha256"] for m in ms if m.get("status") == "RUN"}) != 1:
        raise SystemExit("REFUSING_MIXED_PARTS: B / script sha differ across calibrated parts")
    cc, dc, drc = [], [], {}
    for m in ms:
        f = f"{a.out}/tb/{m['stem']}"
        if m.get("status") != "RUN":
            continue
        cc.append(read_tsv(f"{f}.cells.tsv", dtype={"tf": str}))
        if os.path.exists(f"{f}.decoys.tsv"):
            dc.append(read_tsv(f"{f}.decoys.tsv", dtype={"tf": str}))
        with np.load(f"{f}.draws.npz") as z:
            for k in z.files:
                if "|AUROC|" in k:
                    drc[k] = z[k]
    cc, dc = pd.concat(cc, ignore_index=True), (pd.concat(dc, ignore_index=True) if dc else pd.DataFrame())
    atomic_tsv(cc, f"{a.out}/tb_cal_cells_native.tsv")
    if len(dc):
        atomic_tsv(dc, f"{a.out}/tb_cal_decoys_native.tsv")
    # the calibrated BM response cells enter the historical Table B machinery under the BM substrate (system label BM_CHIPATLAS)
    cc2 = cc.assign(system=HISTSYS)
    drc2 = rename_system_keys(drc, CALSYS, HISTSYS)
    dr_c = {k: v for k, v in dr_h.items() if not k.startswith(f"lattice|{HISTSYS}|")}
    dr_c.update(drc2)
    c_c = replace_system_rows(c_h, cc2, HISTSYS)
    f_h, f_c = TB.cell_flags(c_h), TB.cell_flags(c_c)
    mac_h, pr_h, sub_h = TB.macro_tables(f_h, dr_h)
    mac_c, pr_c, sub_c = TB.macro_tables(f_c, dr_c)
    tc_h, tc_c = table_c_like(mac_h, sub_h, "HISTORICAL_BM"), table_c_like(mac_c, sub_c, "CALIBRATED_BM")
    # parity of the rebuilt historical macro: GeoBRIDGE P and E must match the reference values
    # (+.110, +.014) within 6e-4, else the run aborts. Stored Table B macro files under any run dir
    # are only reported (max abs difference, tolerance 1e-9); a stale file does not abort.
    par = dict(stored_files=[])
    for stp in sorted(glob.glob(f"{R}/results/real_uniform/**/tableB_macro.tsv", recursive=True)):
        sm = read_tsv(stp)
        if "target" not in sm:
            continue
        j = mac_h.merge(sm[sm.target == "resp"], on=["target", "method", "quantity"], suffixes=("", "_stored"))
        if len(j):
            par["stored_files"].append(dict(file=stp, n_rows=int(len(j)), max_abs_point=float((j.point - j.point_stored).abs().max())))
    par["status"] = "PASS_ANY_STORED_FILE_1e-9" if any(x["max_abs_point"] <= 1e-9 for x in par["stored_files"]) else "NO_STORED_FILE_MATCHES"
    gbp = mac_h[(mac_h.method == "GeoBRIDGE") & (mac_h.quantity == "P")].point
    gbe = mac_h[(mac_h.method == "GeoBRIDGE") & (mac_h.quantity == "E")].point
    par["ledger_s404"] = dict(P=float(gbp.iloc[0]) if len(gbp) else None, E=float(gbe.iloc[0]) if len(gbe) else None,
                              ok=bool(len(gbp) and len(gbe) and abs(gbp.iloc[0] - 0.110) < 6e-4 and abs(gbe.iloc[0] - 0.014) < 6e-4))
    if not par["ledger_s404"]["ok"]:
        raise SystemExit(f"REFUSING_PARITY Table B historical macro rebuild does not reproduce s404 (GB prior strength +.110, E +.014): {par}")
    for nm, df in (("HIST", mac_h), ("CAL", mac_c)):
        atomic_tsv(df, f"{a.out}/tb_macro_{nm}.tsv")
    for nm, df in (("HIST", pr_h), ("CAL", pr_c)):
        atomic_tsv(df, f"{a.out}/tb_macro_paired_E_{nm}.tsv")
    for nm, df in (("HIST", sub_h), ("CAL", sub_c)):
        atomic_tsv(df, f"{a.out}/tb_substrate_values_{nm}.tsv")
    both = pd.concat([tc_h, tc_c], ignore_index=True)
    atomic_tsv(both, f"{a.out}/tableC_resp_hist_vs_calbm.tsv")
    # BM cell side-by-side (every method, every roster TF; ungated / NO_CALL rows kept and flagged)
    keep = [k for k in ("tf", "method", "B_status", "gate", "n_M1", "n_rows", "n_pos", "n_neg", "J", "S_AUROC", "P_AUROC", "M1_AUROC", "d_AUROC", "dec_med_AUROC", "calD_AUROC", "cog_pct_AUROC", "p_pool_AUROC",
                          "decoy_status", "decoy_sha", "decoy_source", "n_decoy_stored", "n_decoy_dropped_ineligible") if k in f_c or k in f_h]
    sbs = f_h[f_h.system == HISTSYS][[k for k in keep if k in f_h]].assign(arm="HIST").merge(
        f_c[(f_c.system == HISTSYS) & f_c.tf.isin(cc.tf)][[k for k in keep if k in f_c]].assign(arm="CAL_TOTAL"), on=["tf", "method"], how="outer", suffixes=("_hist", "_cal"))
    atomic_tsv(sbs, f"{a.out}/tb_bm_side_by_side.tsv")
    notrun = {m["tf"]: m["status"] for m in ms if m.get("status") != "RUN"}
    gates = {m["tf"]: m.get("gate_M1_resp") for m in ms if m.get("status") == "RUN"}
    ok_roster = sorted(f_c[(f_c.system == HISTSYS) & (f_c.B_status == "OK")].tf.unique())
    L_ = ["### Table B/C response evaluation with calibrated BM labels\n"
          "Calibrated BM labels use CAL_TOTAL. Decoy sources are the stored historical sources, with eligibility re-checked on the calibrated rows. Descriptive; no multiplicity correction.",
          f"calibrated BM cells: gate(M1,resp) by TF = {json.dumps(gates)}; Table-B OK roster (>=50 usable decoys) = {ok_roster}; NOT_RUN = {json.dumps(notrun)}",
          f"Historical-reference parity: {json.dumps(par, default=str)}"]
    dropped = cc[cc.get("n_decoy_dropped_ineligible", pd.Series(0, index=cc.index)).fillna(0) > 0] if "n_decoy_dropped_ineligible" in cc else cc.iloc[0:0]
    L_.append(f"stored decoys dropped as ineligible on the calibrated rows: {len(dropped)} (cell x method) rows" + (f"; max {int(dropped.n_decoy_dropped_ineligible.max())}" if len(dropped) else ""))
    for tag, tcx in (("HISTORICAL BM", tc_h), ("CALIBRATED BM", tc_c)):
        L_.append(f"\n## {tag}: 4-substrate macro (equal TF weight within substrate, equal substrate weight) + BM substrate alone; prior strength = AUROC(P)-.5, E = calibrated focal increment")
        L_.append(f"{'method':14s} {'prior_macro':>11s} {'[gene CI]':>18s} {'E_macro':>9s} {'[gene CI]':>18s} | {'BM prior':>9s} {'BM E':>8s} {'[gene CI]':>18s}  BM units")
        for r in tcx.sort_values("prior_strength_macro", ascending=False).itertuples():
            L_.append(f"{r.method:14s} {f3(r.prior_strength_macro):>11s} [{f3(r.prior_strength_macro_gene_lo)},{f3(r.prior_strength_macro_gene_hi)}] {f3(r.E_macro):>9s} "
                      f"[{f3(r.E_macro_gene_lo)},{f3(r.E_macro_gene_hi)}] | {f3(getattr(r, 'BM_prior_strength', np.nan)):>9s} {f3(getattr(r, 'BM_E', np.nan)):>8s} "
                      f"[{f3(getattr(r, 'BM_E_gene_lo', np.nan))},{f3(getattr(r, 'BM_E_gene_hi', np.nan))}]  {getattr(r, 'BM_units', '')}")
    gbh, gbc = (x[x.method == "GeoBRIDGE"].iloc[0] for x in (tc_h, tc_c))
    L_.append(f"\nGeoBRIDGE: macro prior strength {f3(gbh.prior_strength_macro)} -> {f3(gbc.prior_strength_macro)}; macro E {f3(gbh.E_macro)} -> {f3(gbc.E_macro)} (historical -> calibrated BM)")
    L_.append(f"GeoBRIDGE rank by macro E (1 = largest): {int(1 + (tc_h.E_macro > gbh.E_macro).sum())} -> {int(1 + (tc_c.E_macro > gbc.E_macro).sum())} of {len(tc_c)}; "
              f"by prior strength: {int(1 + (tc_h.prior_strength_macro > gbh.prior_strength_macro).sum())} -> {int(1 + (tc_c.prior_strength_macro > gbc.prior_strength_macro).sum())}")
    open(f"{a.out}/report_tb.txt", "w").write("\n".join(L_) + "\n")
    atomic_json(f"{a.out}/receipt_tb.json", receipt_base("Tables B/C with calibrated BM labels", M, dict(B=int(ms[0]["B"]), parity=par, gates=gates, ok_roster=ok_roster, not_run=notrun, inputs=dict(
        prior_adjusted_full=PA_FULL, cells_sha256=sha256(f"{PA_FULL}/table2_cells.tsv"), decoys_sha256=sha256(f"{PA_FULL}/table2_decoys.tsv")), n_dropped_rows=int(len(dropped)),
        note="Table B stays descriptive (Holm family unresolvable, J floor 1/51); nominal gene-bootstrap intervals only; historical reported alongside")))
    print("\n".join(L_))


# (c) merged four-class factorial: feasibility
def cmd_fc(a):
    M = preflight()
    PT = M["PT"]
    import gb_audit_decomp as AD
    import gb_layer_lattice as L
    import gb_lattice_extra as LE
    dr = mkdr(PT, 60)
    rows = []
    for tf in roster_of(L, LE, AD):
        for arm, system in (("HIST", HISTSYS), ("CAL_TOTAL", CALSYS)):
            try:
                u = PT.lattice_unit(system, tf, dr.gene)
            except (L.Blocked, SystemExit) as e:
                if isinstance(e, SystemExit) and "REFUSING_PARITY" in str(e):
                    raise
                rows.append(dict(tf=tf, arm=arm, status=f"NOT_RUN: {type(e).__name__}: {e}"))
                continue
            T = L.targets(u["ep"], u["MR"])
            (yab, rab), (ycd, rcd) = T["AB_prim"], T["CD_prim"]
            yab, rab, ycd, rcd = (np.asarray(x, bool) for x in (yab, rab, ycd, rcd))
            sz = dict(A=int((yab & rab).sum()), B=int((~yab & rab).sum()), C=int((ycd & rcd).sum()), D=int((~ycd & rcd).sum()))
            mn = min(sz.values())
            rows.append(dict(tf=tf, arm=arm, status="OK", gate_MR_resp=bool(u["gate"]["MR"]), n_MR=int(u["MR"].sum()), **{f"size_{k}": v for k, v in sz.items()}, minn=mn,
                             eligible_minc5=bool(mn >= MINC_FOURCLASS), eligible_minc10=bool(mn >= 10)))
    t = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    atomic_tsv(t, f"{a.out}/fourclass_feasibility_BM.tsv")
    ok = t[t.status == "OK"]
    rc = receipt_base("Four-class factorial feasibility with calibrated BM labels", M, dict(
        verdict="CLASS_LABELS_COMBINABLE; FACTORIAL_RERUN_NOT_RUN",
        reason=("the calibrated system BM_CHIPATLAS_CALTOT carries ChIP-Atlas occupancy (unchanged builder gb_lattice_extra.bm_truth) AND calibrated responders in one truth, so the four classes A/B/C/D are defined "
                "(sizes in fourclass_feasibility_BM.tsv).  The frozen factorial (gb_fourclass_cluster_merged -> gb_fourclass_cluster.run -> gb_fourclass_dist.run_unit) consumes saved per-unit "
                "raw/scores_{system}_{tf}.npz and is bound to audited system keys (parity / covariate patches). Required factorial scripts were unavailable; the factorial was not rerun."),
        n_eligible_minc5_hist=int(ok[(ok.arm == "HIST") & ok.eligible_minc5].shape[0]), n_eligible_minc5_cal=int(ok[(ok.arm == "CAL_TOTAL") & ok.eligible_minc5].shape[0])))
    atomic_json(f"{a.out}/receipt_fc.json", rc)
    print(t.to_string(index=False))
    print(f"[fc] eligible (min class >= {MINC_FOURCLASS}): HIST {rc['n_eligible_minc5_hist']} / CAL {rc['n_eligible_minc5_cal']} BM units; factorial re-run NOT performed (see receipt_fc.json)")


# selftest
def selftest(a):
    # 1. restrict_pick: nearest-first among the allowed and eligible sources only, stable ties,
    # inf-distance dropped
    def pick(F, r, elig, n):                                                     # same algebra as gb_prior_adjusted.pick_decoys
        d = np.sqrt(((F - F[r]) ** 2).sum(1))
        d[~np.asarray(elig, bool)] = np.inf
        d[r] = np.inf
        o = np.argsort(d, kind="stable")[:n]
        o = o[np.isfinite(d[o])]
        return o, d[o]
    F = np.array([[0.0], [1.0], [2.0], [3.0], [4.0], [5.0]])
    elig = np.array([1, 1, 1, 1, 0, 1], bool)
    allowed = np.array([0, 1, 0, 1, 1, 1], bool)
    o, d = restrict_pick(pick, allowed)(F, 0, elig, 50)
    assert list(o) == [1, 3, 5] and list(d) == [1.0, 3.0, 5.0], (o, d)          # 2 not allowed; 4 allowed but ineligible; focal 0 never
    o_all, _ = pick(F, 0, elig, 50)
    assert list(o_all) == [1, 2, 3, 5]
    # 2. rename_system_keys / replace_system_rows
    D = {f"lattice|{CALSYS}|GATA1|resp|GeoBRIDGE|AUROC|S|g": np.zeros(2), f"lattice|PANC|X|resp|GeoBRIDGE|AUROC|S|g": np.ones(2)}
    D2 = rename_system_keys(D, CALSYS, HISTSYS)
    assert f"lattice|{HISTSYS}|GATA1|resp|GeoBRIDGE|AUROC|S|g" in D2 and f"lattice|PANC|X|resp|GeoBRIDGE|AUROC|S|g" in D2 and len(D2) == 2
    C0 = pd.DataFrame(dict(system=[HISTSYS, HISTSYS, "PANC"], tf=["A", "B", "X"], v=[1, 2, 3]))
    Cc = pd.DataFrame(dict(system=[HISTSYS], tf=["B"], v=[9]))
    C1 = replace_system_rows(C0, Cc, HISTSYS)
    assert sorted(C1.tf) == ["B", "X"] and int(C1[C1.tf == "B"].v.iloc[0]) == 9
    # 3. stored_sources keeps the historical match order per (tf, method)
    d = pd.DataFrame(dict(kind="lattice", system=HISTSYS, target="resp", tf="GATA1", method="GeoBRIDGE", decoy=["z", "y", "x"], match_rank=[2, 0, 1]))
    assert stored_sources(d)[("GATA1", "GeoBRIDGE")] == ["y", "x", "z"]
    print("[calbm-prop] selftest (synthetic) PASS")
    if getattr(a, "data", False):
        selftest_data(a)


def selftest_data(a):
    M = preflight(need_nullp=True)
    PT, NP, PA, CB = M["PT"], M["NP"], M["PA"], M["CB"]
    import gb_layer_lattice as L
    tf, B = "RUNX1", int(a.B)
    dr = mkdr(PT, B)
    dr2, _ = PT.lattice_draws(CALSYS, B, sizes=(PT.PRIMARY_SIZE,))
    assert dr.fp == dr2.fp, "calibrated-system draws differ from the historical BM draws"
    # (1) A1
    d5, axis = NP.d5_load(D5S), NP.d5_axis(D5S)
    assert np.array_equal(np.asarray(dr.gene.names).astype(str), axis)
    jsm, ssm = list(L.REFS).index("REF -sm_det"), L.REF_SIGNS["REF -sm_det"][1]
    _u, crows, _v, _d = a1_unit(M, HISTSYS, tf, dr, B, d5, jsm, ssm)
    r0 = next(r for r in crows if r["stratum"] == "all")
    st = read_tsv(f"{NULLP_DIR}/nullp_contrast.tsv", dtype={"tf": str})
    st = st[(st.system == HISTSYS) & (st.tf == tf) & (st.stratum == "all")]
    if not len(st):
        raise SystemExit("REFUSING: no stored A1 row for BM_CHIPATLAS/RUNX1")
    worst = max(abs(float(r0[k]) - float(st[k].iloc[0])) for k in ("AUROC_real", "AUROC_mednull", "AUROC_delta", "AUROC_frac_real_gt_null", "APLIFT_real", "P50_real"))
    assert int(r0["n_rows"]) == int(st.n_rows.iloc[0]) and int(r0["n_pos"]) == int(st.n_pos.iloc[0]), "A1 row counts differ from the stored historical row"
    if worst > 1e-9:
        raise SystemExit(f"REFUSING_PARITY A1 {HISTSYS}/{tf}: worst point diff {worst:.3e} vs stored nullp_contrast.tsv")
    print(f"[calbm-prop] selftest A1 historical parity {HISTSYS}/{tf}: worst point |diff| = {worst:.2e}")
    c = read_tsv(f"{PA_FULL}/table2_cells.tsv", dtype={"tf": str})
    c = c[(c.system == HISTSYS) & (c.tf == tf) & (c.target == "resp")].set_index("method")
    d = read_tsv(f"{PA_FULL}/table2_decoys.tsv", dtype={"tf": str})
    stored = stored_sources(d)
    for lab, st_src in (("fresh", None), ("stored", stored)):
        _u, rows_out, _dec, _dr, _par, acct = tb_unit(M, HISTSYS, tf, dr, B, a.n_jobs, st_src)
        got = {r["method"]: r for r in rows_out}
        worst = 0.0
        for m, r in got.items():
            for k in ("S_AUROC", "P_AUROC", "M1_AUROC", "d_AUROC", "dec_med_AUROC", "calD_AUROC", "p_pool_AUROC", "cog_pct_AUROC"):
                if k in r and k in c and np.isfinite(r[k]):
                    worst = max(worst, abs(float(r[k]) - float(c.loc[m, k])))
            if "decoy_sha" in r and pd.notna(c.loc[m, "decoy_sha"]) and int(r["decoy_sha"]) != int(c.loc[m, "decoy_sha"]):
                raise SystemExit(f"REFUSING_PARITY Table B {lab} decoy set of {tf}/{m} differs from the stored historical set")
        if worst > 1e-9:
            raise SystemExit(f"REFUSING_PARITY Table B {lab} {HISTSYS}/{tf}: worst point diff {worst:.3e} vs stored table2_cells.tsv")
        print(f"[calbm-prop] selftest Table B {lab} decoys, historical parity {HISTSYS}/{tf}: worst point |diff| = {worst:.2e}; decoy sets identical")
        if lab == "stored":
            assert all(v["same_set_as_stored"] for v in acct.values() if v["source"] == "STORED_HIST"), "stored-source wrapper changed a historical decoy set on unchanged rows"
    # (3)/(4) calibrated gate + truth-free identity
    for t_ in GATED_CAL_EXPECTED:
        uc, uh = PT.lattice_unit(CALSYS, t_, dr.gene), PT.lattice_unit(HISTSYS, t_, dr.gene)
        assert CB.truth_free_equal(uh, uc), f"truth-free vectors differ {t_}"
        print(f"[calbm-prop] selftest calibrated {t_}: gate MR={uc['gate']['MR']} n_MR={int(uc['MR'].sum())} pos={int(np.asarray(uc['y'], bool)[uc['MR']].sum())}")
    print("[calbm-prop] selftest (data) PASS")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["selftest", "a1", "a1merge", "tb", "tbmerge", "fc"])
    ap.add_argument("--data", action="store_true")
    ap.add_argument("--tfs", default="")
    ap.add_argument("--B", type=int, default=2000)
    ap.add_argument("--out", default=OUT_DEFAULT)
    ap.add_argument("--arms", default="CAL", help="a1: CAL (calibrated, default) | HIST (historical reproduction) | CAL,HIST")
    ap.add_argument("--slice", type=int, default=0)
    ap.add_argument("--nslice", type=int, default=1)
    ap.add_argument("--n-jobs", dest="n_jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1)
    a = ap.parse_args()
    if a.n_jobs * max(a.threads, 1) > 16:
        raise SystemExit("REFUSING: n_jobs x threads > 16")
    {"selftest": selftest, "a1": cmd_a1, "a1merge": cmd_a1merge, "tb": cmd_tb, "tbmerge": cmd_tbmerge, "fc": cmd_fc}[a.cmd](a)


if __name__ == "__main__":
    main()
