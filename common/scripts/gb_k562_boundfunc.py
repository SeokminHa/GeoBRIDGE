#!/usr/bin/env python
"""K562 BOUND-FUNCTIONAL: the pre-registered multi-TF decision point.
usage: gb_k562_boundfunc.py <run_dir> <encode_dir> <out_dir>

Implements PREREG_K562_BOUND_FUNCTIONAL_2026_09_22.md (sha c974ec9b0243cbbf) verbatim. Every rule
below was frozen BEFORE any bound-functional value was computed.

THE QUESTION. Within targets a TF is annotated to BIND in K562, does |S| separate CRISPRi-responsive
from adequately-tested-null targets? Binding is controlled BY RESTRICTION; quantitative occupancy
enters only as a covariate, never as a score input.

WHY THIS AND NOT MORE GATA1. GATA1 has answered its questions: raw |S| orders A>B>D consistently,
but adds nothing over static plus observability and does not beat static at the top of the list.
What it has NOT answered is whether that behaviour generalises across many regulators, which is the
difference between a case study and a method. This endpoint has never been examined.

FROZEN RULES APPLIED HERE: summit within +/-2000 bp of the selected TSS; nearest qualifying TSS;
exact ties UNASSIGNED; ChIP+ = at least one assigned peak; occupancy = log1p(max signalValue);
A = within-TF BH q<0.05 either direction; B = TESTED_NULL with NTC detection >=5%; UNKNOWN and
LOW_INFORMATION excluded; >=100 evaluable ChIP+ targets with >=20 A and >=20 B or explicit NO_CALL
with no threshold relaxation.
"""
import argparse, glob, gzip, json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import mannwhitneyu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gb_substrate import load

ap = argparse.ArgumentParser()
ap.add_argument("run_dir"); ap.add_argument("encode_dir"); ap.add_argument("out_dir")
ap.add_argument("--tss-window", type=int, default=2000)
ap.add_argument("--min-targets", type=int, default=100)
ap.add_argument("--min-class", type=int, default=20)
ap.add_argument("--polarity", default="any", choices=["any", "up", "down"])
a = ap.parse_args(); t0 = time.time(); POL = a.polarity
os.makedirs(a.out_dir, exist_ok=True)

R = __import__("os").environ["GEOBRIDGE_ROOT"]
S = load("K562", canonical=True)
genes = S["genes"]
Zn = np.load(f"{a.run_dir}/per_gene.npz", allow_pickle=True)
eg = np.asarray(Zn["genes"]).astype(str)
gu = np.char.upper(eg); gp = {g: i for i, g in enumerate(gu)}
keep = np.array([g in set(eg.tolist()) for g in np.asarray(genes).astype(str)])
Ms = S["X"][:, keep]
Z = Ms - Ms.mean(0); Z = Z / np.maximum(np.linalg.norm(Z, axis=0), 1e-300)
texpr = Ms.mean(0)

# ---- truth: A = significant either direction; B = TESTED_NULL (state 1) ----------------------
zt = np.load(f"{R}/results/k562_truth_v3_2026_09_21/k562_state.npz", allow_pickle=True)
tg = np.char.upper(zt["genes"].astype(str)); tfs = [str(x).upper() for x in zt["tfs"]]
st = zt["state"]
tcol = {g: i for i, g in enumerate(tg)}

# ---- TSS from the GTF the reference was built from -------------------------------------------
tss = {}
gtf = None
for c in (f"{R}/refs/gencode.v44.annotation.gtf.gz", f"{R}/data/refs/gencode.v44.annotation.gtf.gz"):
    if os.path.exists(c): gtf = c; break
if gtf:
    with gzip.open(gtf, "rt") as f:
        for line in f:
            if line[0] == "#": continue
            p = line.split("\t")
            if p[2] != "gene": continue
            nm = [x for x in p[8].split(";") if "gene_name" in x]
            if not nm: continue
            g = nm[0].split('"')[1].upper()
            tss.setdefault(g, (p[0], int(p[3]) if p[6] == "+" else int(p[4])))
print(f"TSS loaded for {len(tss)} genes from {os.path.basename(gtf) if gtf else 'NONE'}", flush=True)
if not tss:
    raise SystemExit("REFUSING[NO_TSS_ANNOTATION]: cannot apply the frozen peak-to-gene rule")

rows, cov = [], []
for bed in sorted(glob.glob(f"{a.encode_dir}/*.bed.gz")):
    TF = os.path.basename(bed).split("_ENC")[0].upper()
    if TF not in tfs or TF not in gp:
        cov.append(dict(tf=TF, status="NOT_IN_TRUTH_OR_UNIVERSE")); continue
    pk = pd.read_csv(bed, sep="\t", header=None, compression="gzip",
                     usecols=[0, 1, 2, 6, 9], names=["chr", "s", "e", "signal", "summit"])
    pk["pos"] = pk.s + pk.summit
    best = {}
    for ch_, pos, sig in zip(pk.chr.astype(str), pk.pos.astype(int), pk.signal.astype(float)):
        for g, (gc, gp_) in tss.items():
            pass
        # indexed lookup instead of an O(n*m) scan
        break
    # build a per-chromosome sorted TSS index once
    if "TSSIDX" not in globals():
        TSSIDX = {}
        for g, (c_, p_) in tss.items():
            TSSIDX.setdefault(c_, []).append((p_, g))
        for c_ in TSSIDX: TSSIDX[c_].sort()
        TSSARR = {c_: (np.array([x[0] for x in v]), [x[1] for x in v]) for c_, v in TSSIDX.items()}
    assign = {}
    for ch_, pos, sig in zip(pk.chr.astype(str), pk.pos.astype(int), pk.signal.astype(float)):
        if ch_ not in TSSARR: continue
        arr, nms = TSSARR[ch_]
        lo, hi = np.searchsorted(arr, [pos - a.tss_window, pos + a.tss_window])
        if hi <= lo: continue
        d = np.abs(arr[lo:hi] - pos)
        m = d.min()
        wins = np.flatnonzero(d == m)
        if len(wins) != 1: continue            # exact tie -> UNASSIGNED, per the prereg
        g = nms[lo + wins[0]]
        assign[g] = max(assign.get(g, 0.0), sig)
    bound = np.array([gu[i] in assign for i in range(len(gu))])
    occ = np.array([np.log1p(assign.get(gu[i], np.nan)) for i in range(len(gu))])
    ti = tfs.index(TF)
    row = st[ti]
    cidx = np.array([tcol.get(g, -1) for g in gu])
    have = cidx >= 0
    stv = np.where(have, row[np.maximum(cidx, 0)], 0)
    # POLARITY SPLIT. GATA1 shows the separation is REPRESSION-SPECIFIC (A>B AUC 0.638, Holm in
    # 77/124 cells) and absent for activation (0.511, 0/124). K562's truth already distinguishes
    # TESTED_UP (2) from TESTED_DOWN (3), so pooling them as one "responsive" class -- which is what
    # I did first -- would dilute exactly the same way `any_change` diluted GATA1.
    POLS = {"any": [2, 3], "up": [2], "down": [3]}
    A = bound & have & np.isin(stv, POLS[POL])
    B = bound & have & (stv == 1)
    cov.append(dict(tf=TF, polarity=POL, n_peaks=len(pk), n_bound_genes=int(bound.sum()),
                    n_A=int(A.sum()), n_B=int(B.sum()),
                    status="OK" if (A.sum() + B.sum() >= a.min_targets and A.sum() >= a.min_class
                                    and B.sum() >= a.min_class) else "NO_CALL_SUPPORT"))
    if cov[-1]["status"] != "OK":
        continue
    for key in Zn.files:
        if not key.endswith("|absS"): continue
        root, chan, kk, rest = key[:-5].split("|", 3)
        if not rest.startswith(TF + "|"): continue
        s = np.asarray(Zn[key], float)
        m = np.asarray(Zn[f"{key[:-5]}|mask_RECOVERY"], bool) & np.isfinite(s) & (A | B)
        y = A[m].astype(int)
        if y.sum() < a.min_class or (len(y) - y.sum()) < a.min_class: continue
        def auc(v):
            try:
                u, p = mannwhitneyu(v[y == 1], v[y == 0], alternative="two-sided")
                return u / (y.sum() * (len(y) - y.sum())), p
            except Exception:
                return np.nan, np.nan
        ad, pd_ = auc(s[m]); asf, _ = auc(np.abs(Z[:, gp[TF]] @ Z)[m]); ae, _ = auc(texpr[m])
        ao, _ = auc(np.nan_to_num(occ[m], nan=0.0))
        rows.append(dict(tf=TF, polarity=POL, root=root, channel=chan, k=int(kk[1:]), n=len(y), n_A=int(y.sum()),
                         prevalence=float(y.mean()), AUC_absS=ad, p_absS=pd_,
                         AUC_static=asf, AUC_targetexpr=ae, AUC_occupancy=ao))
    print(f"  {TF}: {cov[-1]}  {time.time()-t0:.0f}s", flush=True)

C = pd.DataFrame(cov); C.to_csv(f"{a.out_dir}/coverage_{POL}.tsv", sep="\t", index=False)
if not rows:
    print("\nNO EVALUABLE TF — reporting NO_CALL_SUPPORT, thresholds NOT relaxed")
    print(C.to_string(index=False)); sys.exit(0)
D = pd.DataFrame(rows); D.to_csv(f"{a.out_dir}/boundfunc_{POL}.tsv", sep="\t", index=False)
pd.set_option("display.width", 220)
print(f"\n=== K562 BOUND-FUNCTIONAL [polarity={POL}]: {D.tf.nunique()} TFs, {len(D)} cells ===")
print("AUC 0.5 = no separation. MACRO over TFs is primary; pooled edges are descriptive.\n")
g = D.groupby("k").agg(n_tf=("tf", "nunique"), AUC=("AUC_absS", "median"),
                       static=("AUC_static", "median"), texpr=("AUC_targetexpr", "median"),
                       occ=("AUC_occupancy", "median"),
                       pct_gt_half=("AUC_absS", lambda v: 100 * (v > .5).mean())).reset_index()
print(g[g.k.isin([0, 1, 2, 5, 10, 20, 30])].round(4).to_string(index=False))
print(f"\ncoverage: {(C.status=='OK').sum()} OK / {len(C)} TFs")
print(C[C.status != "NOT_IN_TRUTH_OR_UNIVERSE"].to_string(index=False))
