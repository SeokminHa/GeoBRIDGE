#!/usr/bin/env python
# Cardiac_bam_post.py <srr> <name> <kb_dir> <out_root> Per-run post-processing of kb count nac output (de Soysa GSE126128 WT)
import sys, json, os, shutil, hashlib, numpy as np, scipy.sparse as sp, scipy.io as sio
run, name, kb, root = sys.argv[1:5]
EXPECT = int(sys.argv[5]) if len(sys.argv) > 5 else 0      # read pairs bamtofastq wrote; kb must have processed exactly that many (0 = unchecked, never for production)
MIN_UMI = 200
U = f"{kb}/counts_unfiltered"
geo = [l.strip() for l in open(f"{root}/geo_cells.txt")]
gem_of = {}
for c in geo: gem_of.setdefault(c.rsplit("-", 1)[1], set()).add(c.rsplit("-", 1)[0])
bc = np.loadtxt(f"{U}/cells_x_genes.barcodes.txt", dtype=str)
genes = np.loadtxt(f"{U}/cells_x_genes.genes.names.txt", dtype=str)
gid = np.loadtxt(f"{U}/cells_x_genes.genes.txt", dtype=str)
M = {k: sio.mmread(f"{U}/cells_x_genes.{k}.mtx").tocsr() for k in ("mature", "nascent", "ambiguous")}
tot = np.asarray((M["mature"] + M["nascent"] + M["ambiguous"]).sum(1)).ravel()
print(f"kb barcodes {len(bc)}, with >={MIN_UMI} UMI: {(tot>=MIN_UMI).sum()}, genes {len(genes)}", flush=True)
strong = set(bc[tot >= MIN_UMI]); anyb = set(bc)
ov = {g: dict(n_geo=len(s), frac_strong=len(s & strong) / len(s), frac_any=len(s & anyb) / len(s)) for g, s in gem_of.items()}
order = sorted(ov, key=lambda g: -ov[g]["frac_strong"])
best, second = order[0], order[1]
print("gem overlap:", {g: round(ov[g]["frac_strong"], 3) for g in sorted(ov, key=int)}, flush=True)
if not (ov[best]["frac_strong"] >= 0.80 and ov[second]["frac_strong"] <= 0.25):
    raise SystemExit(f"[ABORT] gem verification failed for {run}: best {best} {ov[best]}, second {second} {ov[second]}")
pos = {b: i for i, b in enumerate(bc)}
cells = sorted(gem_of[best])
rows = np.array([pos[c] for c in cells if c in pos]); found = [c for c in cells if c in pos]
S, N, A = (M[k][rows] for k in ("mature", "nascent", "ambiguous"))
for k, X in (("mature", S), ("nascent", N), ("ambiguous", A)):
    assert np.all(X.data == np.round(X.data)) and X.data.min() >= 0, k
tS, tN, tA = (float(X.sum()) for X in (S, N, A))
ci = np.asarray(S.sum(1)).ravel(); ni = np.asarray(N.sum(1)).ravel()
rj = json.load(open(f"{kb}/run_info.json"))
if EXPECT and int(rj["n_processed"]) != EXPECT:
    raise SystemExit(f"[ABORT] kb processed {rj['n_processed']} read pairs but bamtofastq wrote {EXPECT}: FASTQ list incomplete for {run}")
ki_ = None; ki = json.load(open(f"{kb}/kb_info.json")) if os.path.exists(f"{kb}/kb_info.json") else {}
od = f"{root}/kb/{run}"; os.makedirs(od, exist_ok=True)
for k, X in (("mature", S), ("nascent", N), ("ambiguous", A)): sp.save_npz(f"{od}/{run}_{k}.npz", X.tocsr(), compressed=True)
np.savez(f"{od}/{run}_axes.npz", cells=np.array([f"{c}-{best}" for c in found]), genes=genes, gene_ids=gid)
for f in ("run_info.json", "kb_info.json", "inspect.json"):
    if os.path.exists(f"{kb}/{f}"): shutil.copy(f"{kb}/{f}", f"{od}/{f}")
for f in os.listdir(U):
    if f.endswith((".mtx", ".txt")): shutil.copy(f"{U}/{f}", f"{od}/unfiltered_{f}")
rec = dict(run=run, name=name, verified_gem_group=best, gem_overlap=ov,
           n_geo_cells_in_group=len(cells), n_geo_cells_found_in_kb=len(found), n_kb_barcodes=int(len(bc)), n_kb_barcodes_ge_min_umi=int((tot >= MIN_UMI).sum()),
           min_umi_for_verification=MIN_UMI, totals=dict(mature=tS, nascent=tN, ambiguous=tA),
           unspliced_fraction_of_S_plus_U=tN / (tN + tS), ambiguous_fraction_of_all=tA / (tS + tN + tA),
           median_mature_per_cell=float(np.median(ci)), median_nascent_per_cell=float(np.median(ni)),
           kb_n_processed=rj.get("n_processed"), kb_n_pseudoaligned=rj.get("n_pseudoaligned"), kb_p_pseudoaligned=rj.get("p_pseudoaligned"), kb_p_unique=rj.get("p_unique"),
           kb_call=rj.get("call", "")[:400], n_genes=int(len(genes)))
json.dump(rec, open(f"{root}/receipts/{run}.receipt.json", "w"), indent=1)
print(json.dumps({k: v for k, v in rec.items() if k not in ("gem_overlap", "kb_call")}, indent=1))
