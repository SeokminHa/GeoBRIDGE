#!/usr/bin/env python
# Post.py <srr> <name> <gem> <loom[,loom...]> <out_root> -- per-run validation of the velocyto loom(s) (de Soysa WT, GSE126128)
import sys, json, numpy as np, scipy.sparse as sp, h5py, hashlib
run, name, gem, looms, root = sys.argv[1:6]; gem = int(gem); looms = looms.split(",")
geo = {l.strip().rsplit("-", 1)[0] for l in open(f"{root}/../mouse_cardiac/geo_cells.txt") if l.strip().endswith(f"-{gem}")}
cells, S_l, U_l, A_l, ax = [], [], [], [], None
for lm in looms:
    with h5py.File(lm, "r") as f:
        cells += [c.decode() if isinstance(c, bytes) else str(c) for c in f["col_attrs/CellID"][:]]
        gname = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in f["row_attrs/Gene"][:]])
        acc = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in f["row_attrs/Accession"][:]])
        S_l.append(sp.csr_matrix(f["layers/spliced"][:])); U_l.append(sp.csr_matrix(f["layers/unspliced"][:])); A_l.append(sp.csr_matrix(f["layers/ambiguous"][:]))
    if ax is None: ax = (gname, acc)
    else: assert (gname == ax[0]).all() and (acc == ax[1]).all(), f"gene axis differs across chunks: {lm}"
gname, acc = ax
S, U, A = (sp.hstack(x).tocsr() for x in (S_l, U_l, A_l))      # genes x cells
bc = [c.split(":")[-1] for c in cells]
bc = [b[:-1] if b.endswith("x") else b for b in bc]; bc = [b[:-2] if b.endswith("-1") else b for b in bc]
assert all(len(b) == 16 for b in bc), bc[:3]
assert len(set(bc)) == len(bc), "duplicate barcodes (within or across chunks)"
inn = np.array([b in geo for b in bc]); assert inn.all(), f"{(~inn).sum()} loom cells not in GEO gem {gem}"
assert len(set(acc)) == len(acc), "duplicate Ensembl accessions"
tot = {}
for k, M in (("spliced", S), ("unspliced", U), ("ambiguous", A)):
    assert M.data.min() >= 0 and np.all(M.data == np.round(M.data)), k
    tot[k] = float(M.sum())
det = {}
for g in ("Tbx5", "Nkx2-5", "Gata4", "Nppa", "Actc1", "Hand2"):
    i = np.where(gname == g)[0]
    det[g] = float((S[i[0]].toarray().ravel() > 0).mean()) if len(i) else None
rec = dict(run=run, name=name, gem=gem, looms=[dict(path=l, sha256=hashlib.sha256(open(l, "rb").read()).hexdigest()) for l in looms],
           n_cells=len(cells), n_geo_in_group=len(geo), coverage=len(cells) / len(geo), n_genes=int(len(gname)),
           totals=tot, unspliced_fraction_of_S_plus_U=tot["unspliced"] / (tot["unspliced"] + tot["spliced"]), ambiguous_fraction_of_all=tot["ambiguous"] / sum(tot.values()),
           median_spliced_per_cell=float(np.median(np.asarray(S.sum(0)).ravel())), median_unspliced_per_cell=float(np.median(np.asarray(U.sum(0)).ravel())),
           spliced_detection_fraction_of_cells=det, example_cellid=cells[0])
json.dump(rec, open(f"{root}/receipts/{run}.receipt.json", "w"), indent=1)
print(json.dumps(rec, indent=1))
