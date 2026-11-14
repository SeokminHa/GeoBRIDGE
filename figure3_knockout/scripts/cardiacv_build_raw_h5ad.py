#!/usr/bin/env python
# Cardiacv_build_raw_h5ad.py -- merge the 9 per-run velocyto looms into one raw integer h5ad
import glob, json, os, sys, hashlib, h5py
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad
RAW = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/data/raw/mouse_cardiac_velocyto")
PARTIAL = "--partial" in sys.argv
OUT = f"{RAW}/cardiacv_raw_velocyto{'_PARTIAL' if PARTIAL else ''}.h5ad"
ENA = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/data/raw/mouse_cardiac_feasibility/desoysa_ena_runs.tsv")
STAGE = {"E775": "E7.75", "E825": "E8.25", "E925": "E9.25"}
ena = pd.read_csv(ENA, sep="\t").set_index("run_accession")
recs = sorted((json.load(open(p)) for p in glob.glob(f"{RAW}/receipts/SRR*.receipt.json")), key=lambda r: int(r["gem"]))
if not PARTIAL:
    assert len(recs) == 9 and sorted(int(r["gem"]) for r in recs) == list(range(1, 10)), f"{len(recs)} receipts / gem groups not 1..9"
geo = [l.strip() for l in open(f"{RAW}/../mouse_cardiac/geo_cells.txt")]
Ss, Us, obs, axes0 = [], [], [], None
for r in recs:
    run, gem = r["run"], int(r["gem"])
    Sc, Uc, Ac, cells = [], [], [], []
    for lm in r["looms"]:
        assert hashlib.sha256(open(lm["path"], "rb").read()).hexdigest() == lm["sha256"], f"loom changed since receipt: {lm['path']}"
        with h5py.File(lm["path"], "r") as f:
            cells += [c.decode() if isinstance(c, bytes) else str(c) for c in f["col_attrs/CellID"][:]]
            sym = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in f["row_attrs/Gene"][:]]); acc = np.array([c.decode() if isinstance(c, bytes) else str(c) for c in f["row_attrs/Accession"][:]])
            S_, U_, A_ = (sp.csr_matrix(f[f"layers/{k}"][:].T) for k in ("spliced", "unspliced", "ambiguous"))       # -> cells x genes
        Sc.append(S_); Uc.append(U_); Ac.append(A_)
    S, U, A = sp.vstack(Sc).tocsr(), sp.vstack(Uc).tocsr(), sp.vstack(Ac).tocsr()
    if axes0 is None: axes0 = (sym, acc)
    else: assert (sym == axes0[0]).all() and (acc == axes0[1]).all(), f"gene axis differs for {run}"
    bc = [c.split(":")[-1] for c in cells]; bc = [b[:-1] if b.endswith("x") else b for b in bc]; bc = [b[:-2] if b.endswith("-1") else b for b in bc]
    names = np.array([f"{b}-{gem}" for b in bc])
    title = ena.loc[run, "sample_title"]
    o = pd.DataFrame(index=names)
    o["run"], o["gem"], o["embryo"] = run, gem, title
    o["stage"] = STAGE[title.split("_")[0]]; o["platform"] = "HiSeq4000" if "HiSeq" in ena.loc[run, "experiment_title"] else "NextSeq500"
    o["n_ambiguous_umi"] = np.asarray(A.sum(1)).ravel()
    Ss.append(S); Us.append(U); obs.append(o)
    print(run, title, "gem", gem, "cells", len(names), "of", r["n_geo_in_group"], flush=True)
S, U, O = sp.vstack(Ss).tocsr(), sp.vstack(Us).tocsr(), pd.concat(obs)
a = ad.AnnData(X=S.astype(np.float32).copy(), obs=O, var=pd.DataFrame({"gene_id": axes0[1]}, index=axes0[0]))
a.layers["spliced"], a.layers["unspliced"] = S.astype(np.float32), U.astype(np.float32)
a.var_names_make_unique(); assert a.obs_names.is_unique
found = set(a.obs_names); missing = [c for c in geo if c not in found]
a.uns["cardiac_substrate"] = dict(source="GSE126128 WT 10x v2 cellranger mm10 BAMs -> velocyto run (Default logic; Ens102 GRCm38 GTF chr-prefixed; UCSC mm10 rmsk mask; -b GEO barcodes of the verified gem group)",
                                  cells_geo=len(geo), cells_found=int(a.n_obs), cells_missing=len(missing), spliced_definition="velocyto spliced (ambiguous EXCLUDED from both channels)", per_run_receipts=[r["run"] for r in recs])
a.write_h5ad(OUT)
tS, tU = float(S.sum()), float(U.sum())
rep = dict(out=OUT, n_cells=int(a.n_obs), n_genes=int(a.n_vars), n_geo_cells=len(geo), n_missing=len(missing), missing_examples=missing[:5],
           unspliced_fraction_of_S_plus_U=tU / (tS + tU), median_spliced_umi=float(np.median(np.asarray(S.sum(1)).ravel())), median_unspliced_umi=float(np.median(np.asarray(U.sum(1)).ravel())),
           integer=bool(np.all(S.data == np.round(S.data)) and np.all(U.data == np.round(U.data))), per_stage=O.groupby("stage").size().to_dict(), per_run=O.groupby("run").size().to_dict(),
           sha256=hashlib.sha256(open(OUT, "rb").read()).hexdigest())
json.dump(rep, open(OUT.replace(".h5ad", ".receipt.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))
