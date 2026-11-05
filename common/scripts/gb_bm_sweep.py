#!/usr/bin/env python
"""BONE MARROW FIVE-TF ORACLE SWEEP against ChIP binding truth, with rIgG as negative control.

usage: gb_bm_sweep.py <out_dir> [--kmax 100] [--win 2000]

PROTOCOL. The bone-marrow outcome-blind state was frozen and hashed BEFORE this file was written:
`results/BM_PRETRUTH_FREEZE_2026_09_18.json` (substrate, pairs, spectral ladder, block biology, and the
three source files). This script may therefore read truth. It does not touch k selection: the spectral
ladder is read from the frozen artifact and every k is reported.

TRUTH. ChIP-Atlas hg38 reprocessed peaks for GATA1, GATA2, RUNX1, FLI1, TAL1 and the **rIgG** control,
in `data/mk_chip_gse24674/chipatlas/`. Peak-to-gene uses the ALREADY-FROZEN rule from
`scratch/gb_funcbound.py:36-50` (a gene is BOUND if any peak overlaps TSS +- WIN, WIN = 2000, GRCh38).
No window scanning; the rule predates any performance value. TSS coordinates come from the scPerturb
GRCh38 var table, the same source gb_funcbound used.

THIS IS BINDING, NOT CAUSAL TRUTH. The licensed term is FUNCTIONAL-BOUND TARGET ENRICHMENT. Binding does
not imply regulation, and rIgG is included precisely so that a positive on a nonspecific antibody is
visible rather than assumed away.

UNSIGNED ENDPOINT, REPORTED THREE WAYS. ChIP gives no direction, so ranking by a signed score would
throw away half the information in one direction and all of it in the other. The project has already
measured that polarity-blind statistics lose most of the GATA1 signal, so instead of collapsing to |S|
this reports nAP for +S, -S AND |S| separately at every depth. Nothing is maximised over.

WHAT THE SWEEP IS FOR (PI addendum section 16): to separate
    SCORE FAILURE      no k works for any TF
    SELECTION FAILURE  a useful k exists but the spectral rung misses it
    SUCCESS            the spectral rungs land in a broad useful region
    TF-HETEROGENEITY   different TFs in ONE dataset need different depths -- which would falsify the
                       one-k-per-dataset architecture independently of whether the selector is good.
"""
import argparse, glob, gzip, json, os, sys, time
import numpy as np, pandas as pd, h5py
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
from gb_substrate import load
os.environ.setdefault("OMP_NUM_THREADS", "8")
from gb_kprofile_geometry import nap_tg

R = __import__("os").environ["GEOBRIDGE_ROOT"]
CHIPDIR = f"{R}/data/mk_chip_gse24674/chipatlas"
SCPERTURB = f"{R}/data/scperturb/ReplogleWeissman2022_rpe1.h5ad"
TFS = ["GATA1", "GATA2", "RUNX1", "FLI1", "TAL1"]
CONTROL = "rIgG"


def tss_map(genes):
    """GRCh38 TSS per gene symbol, from the scPerturb var table (same source as gb_funcbound.py:18-30)."""
    gpos = {g.upper(): i for i, g in enumerate(genes)}
    h = h5py.File(SCPERTURB, "r")
    def vcol(n):
        g = h["var"][n]
        if isinstance(g, h5py.Group) and "categories" in g:
            c = np.array(g["categories"]).astype(str); return c[np.array(g["codes"])]
        return np.array(g)
    gn = vcol("gene_name").astype(str); chrom = vcol("chr").astype(str)
    start = np.array(vcol("start")).astype(float); end = np.array(vcol("end")).astype(float)
    strand = vcol("strand").astype(str)
    tss = {}
    for k in range(len(gn)):
        u = gn[k].upper()
        if u in gpos and np.isfinite(start[k]) and np.isfinite(end[k]):
            c = chrom[k] if str(chrom[k]).startswith("chr") else "chr" + str(chrom[k])
            tss[u] = (c, float(start[k]) if str(strand[k]) in ("+", "1", "plus") else float(end[k]))
    h.close()
    return tss


def bound_set(tfname, tss, win):
    hits = sorted(glob.glob(os.path.join(CHIPDIR, f"{tfname}_*.bed")) +
                  glob.glob(os.path.join(CHIPDIR, f"{tfname}_*.bed.gz")))
    if not hits: return None, 0
    peaks = {}
    op = gzip.open if hits[0].endswith(".gz") else open
    with op(hits[0], "rt") as fh:
        for line in fh:
            p = line.split("\t")
            if len(p) < 3 or not p[1].isdigit(): continue
            peaks.setdefault(p[0], []).append((int(p[1]), int(p[2])))
    npk = sum(len(v) for v in peaks.values())
    for c in peaks: peaks[c].sort()
    out = set()
    for g, (c, t) in tss.items():
        arr = peaks.get(c)
        if not arr: continue
        lo, hi = t - win, t + win
        for s, e in arr:
            if e >= lo and s <= hi: out.add(g); break
    return out, npk


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("--kmax", type=int, default=100)
    ap.add_argument("--win", type=int, default=2000)
    a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out_dir, exist_ok=True)
    frz = json.load(open(f"{R}/results/BM_PRETRUTH_FREEZE_2026_09_18.json"))
    ladder = json.load(open(f"{R}/results/sv2_BONEMARROW/k_decision_v2.json"))
    print(f"freeze receipt: {len(frz['files'])} artifacts, frozen {frz['frozen_at']}")
    print(f"spectral ladder (frozen, outcome-blind): k_star={ladder.get('k_star')} ladder={ladder.get('ladder')}", flush=True)

    S = load("BONEMARROW", canonical=True)
    genes, X, V = S["genes"], S["X"], S["V"]
    gu = np.char.upper(genes)
    tss = tss_map(genes)
    print(f"BM canonical {X.shape}; genes with GRCh38 TSS: {len(tss)}", flush=True)

    panels = {}
    for tf in TFS + [CONTROL]:
        bs, npk = bound_set(tf, tss, a.win)
        if bs is None: print(f"  {tf}: NO PEAK FILE -- skipped", flush=True); continue
        pos = np.isin(gu, list({g for g in bs}))
        uni = np.isin(gu, list(tss.keys()))          # scoreable universe = genes with a TSS
        panels[tf] = (pos, uni, npk)
        print(f"  {tf}: {npk} peaks -> {int(pos.sum())} bound of {int(uni.sum())} scoreable "
              f"(prevalence {pos[uni].mean():.3f})", flush=True)

    rows = []
    for root, (I, J) in S["pairs"].items():
        dX = X[J] - X[I]; dX -= dX.mean(0, keepdims=True)
        dV = V[J] - V[I]; dV -= dV.mean(0, keepdims=True)
        U_, sv, _ = np.linalg.svd(dX, full_matrices=False)
        print(f"  {root}: {dX.shape[0]} pairs, svd {time.time()-t0:.0f}s", flush=True)
        for tf, (pos, uni, npk) in panels.items():
            hit = np.flatnonzero(gu == tf.upper())
            if not len(hit):
                print(f"    {tf}: focal gene absent from the universe -- cannot score", flush=True); continue
            ti = int(hit[0])
            x0 = dX[:, [ti]]; n2 = float((x0 ** 2).sum())
            Px = np.zeros_like(x0); Py = np.zeros_like(dV)
            for k in range(0, a.kmax + 1):
                if k > 0:
                    u = U_[:, [k - 1]]
                    Px += u @ (u.T @ x0); Py += u @ (u.T @ dV)
                Qx, Qy = x0 - Px, dV - Py
                nres = (Qx.T @ Qy)[0]; dres = (np.abs(Qx).T @ np.abs(Qy))[0]
                s = nres / np.maximum(dres, 1e-12); s[ti] = 0.0
                ok = uni & np.isfinite(s) & (dres > 1e-12); ok[ti] = False
                p_ = pos.copy(); p_[ti] = False
                if ok.sum() < 20 or p_[ok].sum() < 5: continue
                for lab, sc in (("plusS", s), ("minusS", -s), ("absS", np.abs(s))):
                    rows.append(dict(system="BONEMARROW", root=root, tf=tf, k=k, score=lab,
                                     nAP=float(nap_tg(p_.astype(int), sc, ok)),
                                     q_r=float((Qx ** 2).sum() / max(n2, 1e-12)),
                                     n_pos=int(p_[ok].sum()), n_uni=int(ok.sum())))
            print(f"    {tf} swept {time.time()-t0:.0f}s", flush=True)
        del dX, dV, U_
    D = pd.DataFrame(rows)
    D.to_csv(f"{a.out_dir}/bm_5TF_sweep.tsv", sep="\t", index=False)
    json.dump(dict(system="BONEMARROW", kmax=a.kmax, win=a.win, tfs=list(panels),
                   frozen_before_truth=frz["frozen_at"], spectral_ladder=ladder.get("ladder"),
                   k_star_outcome_blind=ladder.get("k_star"),
                   truth_type="ChIP binding (functional-bound target enrichment), UNSIGNED; rIgG = negative control",
                   peak_to_gene_rule=f"frozen gb_funcbound.py rule: any peak overlapping TSS +-{a.win} bp, GRCh38",
                   selection_performed=False),
              open(f"{a.out_dir}/provenance.json", "w"), indent=1)
    pd.set_option("display.width", 240)
    ks = sorted(set([0] + (ladder.get("ladder") or []) + [5, 10, 20, 40, 60, 100]))
    for sc in ("plusS", "minusS", "absS"):
        sub = D[(D.score == sc) & (D.root == "dc1_min")]
        if not len(sub): continue
        P = sub.pivot_table(index="k", columns="tf", values="nAP")
        print(f"\n=== BM nAP ({sc}, dc1_min), chance = 1.0; rIgG is the NEGATIVE control ===")
        print(P.loc[[k for k in ks if k in P.index]].round(3).to_string())
    print(f"\nORACLE SUMMARY (dc1_min):")
    for sc in ("plusS", "minusS", "absS"):
        for tf in panels:
            v = D[(D.score == sc) & (D.root == "dc1_min") & (D.tf == tf)].set_index("k").nAP
            if not len(v): continue
            print(f"  {sc:7s} {tf:6s} k0={v.get(0, np.nan):.3f} max={v.max():.3f}@k={int(v.idxmax())} "
                  f"| at k_star={ladder.get('k_star')}: {v.get(ladder.get('k_star'), np.nan):.3f} "
                  f"| depths>=0.95max: {int((v >= 0.95 * v.max()).sum())}/{len(v)}")
    print(f"\nwrote {a.out_dir} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
