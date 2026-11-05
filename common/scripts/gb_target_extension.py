#!/usr/bin/env python
"""E2 — VERSIONED UNIVERSAL TARGET-AXIS EXTENSION (internal review §108/§111 + internal review r7 §110: the decisive acute test is bottlenecked by the
2,000-HVG + catalogue axis, which discards most PRO-seq responders before scoring). NO TRUTH IS READ ANYWHERE IN THIS FILE.
usage: gb_target_extension.py build <system>                      -> results/target_ext/<SYS>_ext.npz + .receipt.json
       gb_target_extension.py score <system> --sources TF[,TF..]  -> results/target_ext/<SYS>_ext_scores.npz + .receipt.json

WHAT IS FROZEN (unchanged from GEOBRIDGE_PREPROCESS_V1): raw counts, scv.pp.filter_genes(min_shared_counts, retain = whole
catalogue), scv.pp.normalize_per_cell, the V1 kNN graph (obsp connectivities/distances copied from the V1 fixture), the V1 pair
artifact (S["pairs"] via the fail-closed canonical loader), the V1 projector (V1 obsm X_pca[:, :30]) and every SOURCE vector (V1 Ms).
WHAT CHANGES: the TARGET axis = every gene passing min_shared_counts (the HVG subset step is skipped for targets only); Ms/Mu from
scv.pp.moments on the frozen graph; velocity from scv.tl.velocity with the frozen parameters. Per-gene stochastic velocity and
moments depend only on that gene's counts and the graph, so V1 genes must be reproduced exactly:
  PARITY GATE 1 (build): Ms, Mu, velocity of every V1 gene == the V1 fixture layers: IDENTICAL NaN / +Inf / -Inf patterns per
  layer (internal review r9) and rel. max-abs error <= 1e-5 over finite entries, else REFUSE.
  PARITY GATE 2 (score): |S| (U and V) and operator-matched pair-static of every V1 gene == the frozen evaluator per_gene arrays
  and the pair-static v3 artifact (<= 1e-4) with IDENTICAL finiteness after self exclusion, over an explicit roster — the first
  (focal) source must carry all 186 cells (2 roots x {U,V from per_gene, M from pair-static} x 31 k) — else REFUSE.
Scores (same algebra as gb_layers.py:717-727 and gb_pairstatic.py): residualise the source pair difference and each target pair
difference against Q_k = QR(dP[:, :k]) (k = 0..30, nested columns of one QR), then E = |sum qx*qy| / sum |qx||qy| with support
den > 1e-12; num, den (float32) and z2 = sum (qx*qy)^2 (FLOAT64, internal review r9) are stored so PR = den^2/z2 is available for the
Amendment-6 support control. The nested projection is applied INCREMENTALLY (q_k <- q_{k-1} - Q_k (Q_k^T y)), algebraically the
same residual as y - Q_{1:k} Q_{1:k}^T y, carried in float64 and cast to float32 per k exactly as the evaluator does
(test_target_extension.py checks it against an independent per-k QR). Channels: U (dMu,
PRIMARY), V (dvelocity, sensitivity), M (dMs = pair-static). Uppercase-colliding symbols are dropped as a group, as the evaluator
does; every statistic (absS, num, den, z2) of a source's own gene is cleared. The scorer REQUIRES the build receipt's script SHA to
equal its own bytes (internal review r9: no scoring on an unaudited builder). This is a disclosed DEVELOPMENT repair applied identically to
every system, not confirmation.
"""
import argparse, hashlib, json, os, sys, time
import numpy as np

R = __import__("os").environ["GEOBRIDGE_ROOT"]
sys.path.insert(0, f"{R}/scratch")
OUT = f"{R}/results/target_ext"
KMAX, NPC, EPS, TOL_BUILD, TOL_SCORE, CHUNK = 30, 30, 1e-12, 1e-5, 1e-4, 1000


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def dense(M):
    return np.asarray(M.todense()) if hasattr(M, "todense") else np.asarray(M)


def parity(a, b):
    """(identical NaN/+Inf/-Inf patterns, rel. max-abs error over entries finite in both) — no nan_to_num masking"""
    a = np.asarray(a, np.float64); b = np.asarray(b, np.float64)
    same = bool((np.isnan(a) == np.isnan(b)).all() and (np.isposinf(a) == np.isposinf(b)).all()
                and (np.isneginf(a) == np.isneginf(b)).all())
    f = np.isfinite(a) & np.isfinite(b)
    err = float(np.abs(a[f] - b[f]).max() / max(np.abs(b[f]).max(), 1e-300)) if f.any() else 0.0
    return same, err


def project_scores(dx, y, Qf, kmax=KMAX):
    """yield (k, num, den, z2, absS) for k = 0..kmax. dx (P, ns) float32, y (P, g) float32, Qf (P, >=kmax) float64 orthonormal.
    The residual is carried in FLOAT64 and updated incrementally, r_k = r_{k-1} - q_k (q_k^T v) with q_k^T v in float64, then cast
    once to float32 per k — the evaluator's rounding (y - Q (Q^T y) formed in float64, cast to float32; gb_layers.py:686,723).
    Elementwise passes run multi-threaded (numexpr) in place; z2 accumulates EXACT float64 squares of the float32 residuals."""
    import numexpr as ne
    Q = np.ascontiguousarray(Qf[:, :kmax], np.float64)
    Cx = Q.T @ dx; Cy = Q.T @ y                                  # float64 (float64 Q)
    rx, ry = dx.astype(np.float64), y.astype(np.float64)
    qx = dx.copy(); qy = y.copy(); w = np.empty(qy.shape, np.float64)
    for k in range(kmax + 1):
        if k:
            q, cx, cy = Q[:, k - 1:k], Cx[k - 1:k], Cy[k - 1:k]
            ne.evaluate("rx - q * cx", out=rx); ne.evaluate("ry - q * cy", out=ry)
            ne.evaluate("rx", out=qx, casting="unsafe"); ne.evaluate("ry", out=qy, casting="unsafe")
        # sums over ~1e5 pairs ACCUMULATE IN FLOAT64 over the float32 residual values: a float32 sgemv over one source
        # was measured 1.4e-5 off the float64 reference while the evaluator's multi-source sgemm is ~1e-6 off
        x64 = qx.astype(np.float64)
        ne.evaluate("qy * 1.0", out=w); num = x64.T @ w
        ne.evaluate("abs(qy) * 1.0", out=w); den = np.abs(x64).T @ w; sup = den > EPS
        ne.evaluate("(qy * 1.0) * (qy * 1.0)", out=w)    # promote to float64 FIRST: exact squares of the float32 residuals
        z2 = (x64 ** 2).T @ w
        with np.errstate(invalid="ignore", divide="ignore"):
            absS = np.where(sup, np.abs(num / np.where(sup, den, 1.0)), np.nan)
        yield k, num, den, z2, absS


def build(system):
    import anndata as ad, scvelo as scv
    import gb_canonical as GC
    t0 = time.time(); os.makedirs(OUT, exist_ok=True)
    cfg = json.load(open(GC.CONFIG)); src, species = GC.RAW[system]
    fix = f"{R}/results/canonical_2026_09_18/{system}_canonical.h5ad"
    F = ad.read_h5ad(fix)
    a = ad.read_h5ad(src)
    cp = cfg["tf_retention"][species]; cp = cp if cp.startswith("/") else f"{R}/{cp}"
    cat = {l.strip().upper() for l in open(cp) if l.strip()}
    fn = cfg["filter_and_normalize"]
    scv.pp.filter_genes(a, min_shared_counts=fn["min_shared_counts"],
                        retain_genes=[g for g in a.var_names if str(g).upper() in cat])
    scv.pp.normalize_per_cell(a)
    if list(a.obs_names) != list(F.obs_names):
        raise SystemExit("REFUSING: extension cells differ from the V1 fixture cells")
    v1 = [str(g) for g in F.var_names]; ext = [str(g) for g in a.var_names]; pos = {g: i for i, g in enumerate(ext)}
    miss = [g for g in v1 if g not in pos]
    if miss:
        raise SystemExit(f"REFUSING: {len(miss)} V1 genes absent from the extension filter, e.g. {miss[:3]}")
    iv = np.array([pos[g] for g in v1])
    # normalisation parity BEFORE the graph is attached: same filter + size factors => identical normalised counts
    for l in ("spliced", "unspliced"):
        same, e = parity(dense(a.layers[l][:, iv]), dense(F.layers[l]))
        if e > TOL_BUILD or not same:
            raise SystemExit(f"REFUSING: normalised {l} of V1 genes differs (rel {e:.2e}, nonfinite pattern equal={same})")
    a.obsp["connectivities"] = F.obsp["connectivities"].copy(); a.obsp["distances"] = F.obsp["distances"].copy()
    a.uns["neighbors"] = dict(F.uns["neighbors"]); a.obsm["X_pca"] = F.obsm["X_pca"].copy()
    scv.pp.moments(a, n_pcs=None, n_neighbors=None)
    if (a.obsp["connectivities"] != F.obsp["connectivities"]).nnz:
        raise SystemExit("REFUSING: moments() rebuilt the graph")
    ve = cfg["velocity"]
    scv.tl.velocity(a, mode=ve["mode"], fit_offset=ve["fit_offset"], perc=ve["perc"])
    pp = {l: parity(dense(a.layers[l][:, iv]), dense(F.layers[l])) for l in ("Ms", "Mu", "velocity")}
    par = {l: v[1] for l, v in pp.items()}; pat = {l: v[0] for l, v in pp.items()}
    print(f"{system}: V1 {len(v1)} -> extension {len(ext)} genes; parity {par}, nonfinite patterns equal {pat}", flush=True)
    if max(par.values()) > TOL_BUILD or not all(pat.values()):
        raise SystemExit(f"REFUSING: PARITY GATE 1 failed {par} nonfinite_patterns={pat}")
    out = f"{OUT}/{system}_ext.npz"
    np.savez(out, genes=np.asarray(ext).astype(str), is_v1=np.isin(ext, v1),
             Ms=dense(a.layers["Ms"]).astype(np.float32), Mu=dense(a.layers["Mu"]).astype(np.float32),
             velocity=dense(a.layers["velocity"]).astype(np.float32))
    json.dump(dict(system=system, out=out, sha256=sha(out), script_sha256=sha(os.path.abspath(__file__)),
                   raw=src, raw_sha256=sha(src), v1_fixture=fix, v1_fixture_sha256=sha(fix), config=GC.CONFIG,
                   config_sha256=sha(GC.CONFIG), catalogue=cp, catalogue_sha256=sha(cp), canonical_builder_sha256=sha(GC.__file__),
                   n_v1=len(v1), n_ext=len(ext), parity_rel_maxabs=par, nonfinite_patterns_equal=pat, tol=TOL_BUILD,
                   wall_s=round(time.time() - t0, 1), truth_read=False),
              open(out.replace(".npz", ".receipt.json"), "w"), indent=1)


def score(system, sources):
    from gb_substrate import load
    import anndata as ad
    t0 = time.time()
    rp = f"{OUT}/{system}_ext.receipt.json"; rec = json.load(open(rp)); ep = rec["out"]
    if sha(ep) != rec["sha256"]:
        raise SystemExit("REFUSING: extension artifact changed since build")
    if rec["script_sha256"] != sha(os.path.abspath(__file__)):         # internal review r9: the builder must be these audited bytes
        raise SystemExit("REFUSING: extension was built by different script bytes; rebuild with the current file")
    S = load(system, canonical=True)                          # every V1 fixture/pair provenance gate
    if sha(S["fixture"]) != rec["v1_fixture_sha256"]:
        raise SystemExit("REFUSING: V1 fixture changed since the extension was built")
    E = np.load(ep); eg = E["genes"].astype(str); gu = np.char.upper(eg)
    u_, c_ = np.unique(gu, return_counts=True); coll = set(u_[c_ > 1])
    keep = np.array([g not in coll for g in gu]); eg, gu = eg[keep], gu[keep]
    Ms, Mu = E["Ms"][:, keep], E["Mu"][:, keep]; Vl = np.nan_to_num(E["velocity"][:, keep], nan=0.0)
    vg = np.asarray(S["genes"]).astype(str); vp = {g.upper(): i for i, g in enumerate(vg)}
    src = []
    for t in sources:
        if t.upper() not in vp or t.upper() in coll:
            raise SystemExit(f"REFUSING: source {t} absent from the V1 axis or colliding")
        src.append(vp[t.upper()])
    Xsrc = np.asarray(S["X"], np.float64)[:, src]            # V1 Ms source vectors, exactly the evaluator's
    P30 = np.asarray(ad.read_h5ad(S["fixture"], backed="r").obsm["X_pca"], float)[:, :NPC]
    out = dict(genes=eg, sources=np.asarray(sources), ks=np.arange(KMAX + 1), is_v1=np.isin(eg, vg))
    G, ns = len(eg), len(sources); CH = ("U", "V", "M")
    for root in S["pairs"]:
        for c in CH:
            for k in range(KMAX + 1):
                for q in ("absS", "num", "den", "z2"):
                    out[f"{root}|{c}|k{k}|{q}"] = np.full((ns, G), np.nan, np.float64 if q == "z2" else np.float32)
    for root, (I, J) in S["pairs"].items():
        dP = P30[J] - P30[I]; Qf, _ = np.linalg.qr(dP)        # nested: QR(dP[:, :k]) spans Qf[:, :k]
        dx = (Xsrc[J] - Xsrc[I]).astype(np.float32)
        for g0 in range(0, G, CHUNK):                          # target chunks bound memory; per-gene algebra is unchanged
            sl = slice(g0, min(g0 + CHUNK, G))
            Y = {"U": (Mu[J, sl] - Mu[I, sl]).astype(np.float32), "V": (Vl[J, sl] - Vl[I, sl]).astype(np.float32),
                 "M": (Ms[J, sl] - Ms[I, sl]).astype(np.float32)}
            for c, y in Y.items():
                for k, num, den, z2, absS in project_scores(dx, y, Qf):
                    out[f"{root}|{c}|k{k}|absS"][:, sl] = absS
                    out[f"{root}|{c}|k{k}|num"][:, sl] = num; out[f"{root}|{c}|k{k}|den"][:, sl] = den
                    # sum_p z_p^2 for the Amendment-6 participation control PR = den^2 / z2 (concentration, not an n_eff)
                    out[f"{root}|{c}|k{k}|z2"][:, sl] = z2
            del Y
        print(f"  {root}: {len(I)} pairs scored {time.time()-t0:.0f}s", flush=True)
    for s_ in range(len(sources)):                            # the source never scores itself: clear EVERY statistic
        own = eg == vg[src[s_]]
        for key in [k_ for k_ in out if k_.count("|") == 3]:
            out[key][s_, own] = np.nan
    # ---------------- PARITY GATE 2 against the frozen evaluator and pair-static (V1 genes only)
    D = f"{R}/results/canon_{system}"; Zn = np.load(f"{D}/per_gene.npz", allow_pickle=True)
    zg = np.asarray(Zn["genes"]).astype(str); ep_ = {g: i for i, g in enumerate(eg)}
    zi = np.array([ep_.get(g, -1) for g in zg])
    if (zi < 0).any():
        raise SystemExit(f"REFUSING: {(zi < 0).sum()} evaluator genes absent from the extension axis")
    PS = np.load(f"{D}/pairstatic_{system}_allk.v3.npz", allow_pickle=True)
    # pair-static keeps exact-case names of uppercase-collision groups (e.g. GATA1 Setd1a/SETD1A) that the extension axis drops as a
    # group; the parity comparison is made on the genes present on BOTH axes (the same identical exclusion on both sides)
    psg_all = np.asarray(PS["genes"]).astype(str); psk = np.array([g in ep_ for g in psg_all])
    psg = psg_all[psk]; pi = np.array([ep_[g] for g in psg])
    # internal review E4 §2: EQUAL finiteness (after identical self/collision exclusion) and an EXPLICIT roster per checked source:
    # U,V from per_gene (2 roots x 2 ch x 31 k) + M from pair-static (2 roots x 31 k) = 186 cells for a panel-bearing focal
    worst = dict(U=0.0, V=0.0, M=0.0); nchk = 0; roster = {}
    roots = list(S["pairs"])

    def cmp(a_, b_, where, ch):
        if not (np.isfinite(a_) == np.isfinite(b_)).all():
            raise SystemExit(f"REFUSING: PARITY GATE 2 finiteness pattern differs at {where}")
        fa = np.isfinite(a_)
        worst[ch] = max(worst[ch], float(np.abs(a_[fa] - b_[fa]).max()) if fa.any() else 0.0)
    for s_, t in enumerate(sources):
        cells = set(); self_z = zg == vg[src[s_]]
        pk = [k_ for k_ in Zn.files if k_.count("|") >= 6 and k_.split("|")[3].upper() == t.upper()
              and k_.endswith("|absS") and "|reference|" not in k_]
        for key in pk:
            root, ch, kk = key.split("|")[:3]
            a_ = np.asarray(Zn[key], float); b_ = out[f"{root}|{ch}|{kk}|absS"][s_, zi].astype(float)
            a_[self_z] = np.nan; b_[self_z] = np.nan
            cmp(a_, b_, key, ch); cells.add((root, ch, kk)); nchk += 1
        psf = [f for f in np.asarray(PS["focals"]).astype(str) if f.upper() == t.upper()]
        for root in (roots if psf else []):
            for k in range(KMAX + 1):
                a_ = np.asarray(PS[f"{root}|k{k}|{psf[0]}"], float)[psk]; b_ = out[f"{root}|M|k{k}|absS"][s_, pi].astype(float)
                self_p = np.char.upper(psg) == t.upper(); a_[self_p] = np.nan; b_[self_p] = np.nan
                cmp(a_, b_, f"pairstatic {root} k{k} {t}", "M"); cells.add((root, "M", f"k{k}")); nchk += 1
        want = {(r, c, f"k{k}") for r in roots for c in (("U", "V") if pk else ()) for k in range(KMAX + 1)} | \
               {(r, "M", f"k{k}") for r in (roots if psf else []) for k in range(KMAX + 1)}
        if cells != want:
            raise SystemExit(f"REFUSING: PARITY GATE 2 roster for {t}: {len(want - cells)} cells missing")
        roster[t] = len(cells)
    print(f"PARITY GATE 2: {nchk} arrays, roster {roster}, worst abs diff {worst}", flush=True)
    if sources[0] not in roster or roster[sources[0]] != 6 * (KMAX + 1) or max(worst.values()) > TOL_SCORE:
        raise SystemExit(f"REFUSING: PARITY GATE 2 failed (first source must carry the full 186-cell roster; {roster}, {worst})")
    op = f"{OUT}/{system}_ext_scores.npz"
    np.savez_compressed(op, **out)
    json.dump(dict(system=system, out=op, sha256=sha(op), script_sha256=sha(os.path.abspath(__file__)), extension=ep,
                   extension_sha256=rec["sha256"], build_script_sha256=rec["script_sha256"], v1_fixture_sha256=sha(S["fixture"]), pairs_sha256=sha(S["pairs_file"]),
                   per_gene_sha256=sha(f"{D}/per_gene.npz"), pairstatic_sha256=sha(f"{D}/pairstatic_{system}_allk.v3.npz"),
                   sources=sources, n_targets=int(len(eg)), n_new=int((~out["is_v1"]).sum()), dropped_collisions=sorted(coll),
                   parity_arrays_checked=nchk, parity_roster=roster, parity_worst_absdiff=worst, tol=TOL_SCORE, truth_read=False,
                   wall_s=round(time.time() - t0, 1)),
              open(op.replace(".npz", ".receipt.json"), "w"), indent=1)
    print(f"wrote {op}: {len(eg)} targets ({int((~out['is_v1']).sum())} new) x {len(sources)} sources", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["build", "score"]); ap.add_argument("system")
    ap.add_argument("--sources", default="")
    a = ap.parse_args()
    build(a.system) if a.cmd == "build" else score(a.system, [s for s in a.sources.split(",") if s])
