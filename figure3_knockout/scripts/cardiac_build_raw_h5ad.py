#!/usr/bin/env python
"""cardiac_build_raw_h5ad.py -- merge the 9 per-run kb-nac outputs (GEO-passed cells only, cardiac_bam_post.py) into ONE raw integer h5ad.
UNTREATED SUBSTRATE = all GEO-passed WT cells of GSE126128 (36,753 by the GEO count matrix), NO truth-based / marker-based subsetting.
X = spliced (kb 'mature'); layers spliced = mature, unspliced = nascent; ambiguous is EXCLUDED from both channels (project nac convention, gb_bcl11b_substrate.py) and kept per cell in obs.
  python cardiac_build_raw_h5ad.py            -> data/raw/mouse_cardiac/cardiac_raw_nac.h5ad   (spliced = mature; registered as CARDIAC by scratch/gb_canonical_cardiac.py)
  python cardiac_build_raw_h5ad.py --partial  -> *_PARTIAL.h5ad (smoke only)
NOTE: kb nac assigns every read compatible with BOTH the cDNA and the full-length nascent transcript to `ambiguous` (63 % of all UMIs here); under the project convention (spliced = mature, as for
BCL11B / RPE1 / K562) the primary TFs are therefore almost undetected in the spliced channel (see results/cardiac_diag_*.json).  PI 2026-09-29: convention unchanged, ambiguous NOT added.
"""
import glob, json, os, sys, hashlib
import numpy as np, pandas as pd, scipy.sparse as sp, anndata as ad

RAW = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/data/raw/mouse_cardiac")
VARIANT = "MATURE"
PARTIAL = "--partial" in sys.argv        # SMOKE ONLY: whatever receipts exist, written to a *_PARTIAL path that no production code reads
OUT = f"{RAW}/cardiac_raw_nac{'_PARTIAL' if PARTIAL else ''}.h5ad"
ENA = f"{__import__('os').environ['GEOBRIDGE_ROOT']}/data/raw/mouse_cardiac_feasibility/desoysa_ena_runs.tsv"
STAGE = {"E775": "E7.75", "E825": "E8.25", "E925": "E9.25"}
ena = pd.read_csv(ena_path := ENA, sep="\t").set_index("run_accession")
recs = sorted((json.load(open(p)) for p in glob.glob(f"{RAW}/receipts/SRR*.receipt.json")), key=lambda r: int(r["verified_gem_group"]))
if not PARTIAL:
    assert len(recs) == 9, f"{len(recs)} receipts (need 9)"
    assert sorted(int(r["verified_gem_group"]) for r in recs) == list(range(1, 10)), "gem groups not 1..9 exactly once"
geo = [l.strip() for l in open(f"{RAW}/geo_cells.txt")]
Ss, Us, obs, axes0 = [], [], [], None
for r in recs:
    run, d = r["run"], f"{RAW}/kb/{r['run']}"
    ax = np.load(f"{d}/{run}_axes.npz", allow_pickle=True)
    if axes0 is None:
        axes0 = (ax["genes"].astype(str), ax["gene_ids"].astype(str))
    else:
        assert (ax["genes"].astype(str) == axes0[0]).all() and (ax["gene_ids"].astype(str) == axes0[1]).all(), f"gene axis differs for {run}"
    S, U, A = (sp.load_npz(f"{d}/{run}_{k}.npz").tocsr() for k in ("mature", "nascent", "ambiguous"))
    cells = ax["cells"].astype(str)
    title = ena.loc[run, "sample_title"]
    o = pd.DataFrame(index=cells)
    o["run"], o["gem"], o["embryo"] = run, int(r["verified_gem_group"]), title
    o["stage"] = STAGE[title.split("_")[0]]
    o["platform"] = "HiSeq4000" if "HiSeq" in ena.loc[run, "experiment_title"] else "NextSeq500"
    o["n_ambiguous_umi"] = np.asarray(A.sum(1)).ravel()
    Ss.append(S); Us.append(U); obs.append(o)
    print(run, title, "gem", r["verified_gem_group"], "cells", len(cells), "of", r["n_geo_cells_in_group"], flush=True)
S, U, O = sp.vstack(Ss).tocsr(), sp.vstack(Us).tocsr(), pd.concat(obs)
a = ad.AnnData(X=S.astype(np.float32).copy(), obs=O, var=pd.DataFrame({"gene_id": axes0[1]}, index=axes0[0]))
a.layers["spliced"], a.layers["unspliced"] = S.astype(np.float32), U.astype(np.float32)
a.var_names_make_unique()
assert a.obs_names.is_unique
found = set(a.obs_names); missing = [c for c in geo if c not in found]
a.uns["cardiac_substrate"] = dict(source="GSE126128 WT 10x v2 cellranger BAMs -> bamtofastq -> kb count nac (GRCm38 Ensembl 102)", cells_geo=len(geo), cells_found=int(a.n_obs), cells_missing=len(missing),
                                  spliced_definition="mature (ambiguous EXCLUDED from both channels; project nac convention)", per_run_receipts=[r["run"] for r in recs])
a.write_h5ad(OUT)
tS, tU = float(S.sum()), float(U.sum())
rep = dict(variant=VARIANT, out=OUT, n_cells=int(a.n_obs), n_genes=int(a.n_vars), n_geo_cells=len(geo), n_missing_from_kb=len(missing), missing_examples=missing[:5],
           unspliced_fraction_of_S_plus_U=tU / (tS + tU), median_spliced_umi=float(np.median(np.asarray(S.sum(1)).ravel())), median_unspliced_umi=float(np.median(np.asarray(U.sum(1)).ravel())),
           integer=bool(np.all(S.data == np.round(S.data)) and np.all(U.data == np.round(U.data))), per_stage=O.groupby("stage").size().to_dict(), per_run=O.groupby("run").size().to_dict(),
           per_platform=O.groupby("platform").size().to_dict(), sha256=hashlib.sha256(open(OUT, "rb").read()).hexdigest())
json.dump(rep, open(OUT.replace(".h5ad", ".receipt.json"), "w"), indent=1)
print(json.dumps(rep, indent=1))
