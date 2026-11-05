#!/usr/bin/env python
"""FREEZE THE HUMAN CD34+ BONE-MARROW OBSERVATIONAL SUBSTRATE (NO TRUTH READ).

usage: gb_bonemarrow_substrate.py <out_dir> [--n_top_genes 2000] [--seed 0]

PURPOSE. Build an independent, non-KO real substrate so the RAW/ADJUSTED adjustment principle can be
established WITHOUT GATA1 truth. The matching ground truth is primary human megakaryocyte ChIP for GATA1,
GATA2, RUNX1, FLI1 and SCL/TAL1 (GSE24674) -- BINDING, not functional response -- and it is NOT touched here.
This script emits only observational quantities.

SUBSTRATE. data/BoneMarrow/human_cd34_bone_marrow.h5ad, 5780 cells x 14319 genes, shipping velocyto-derived
spliced/unspliced layers and a precomputed Palantir pseudotime. Clusters: HSC_1 1118, HSC_2 984, Ery_1 742,
Mono_1 724, Precursors 666, CLP 488, Mono_2 429, DCs 303, Ery_2 265, Mega 61.

A DECISION THAT MUST BE FROZEN NOW, BEFORE ANY TRUTH IS READ. The truth is megakaryocyte binding, but the
Mega cluster has only 61 cells -- far too few to rebuild a trajectory, a pair graph and an SVD inside. So the
analysis runs on the FULL CD34+ hierarchy, and the resulting mismatch (MK-specific binding truth scored on a
multi-lineage substrate) is a stated limitation, NOT something to be discovered later and explained away. The
five TFs are all expressed across the HSPC/erythroid/megakaryocytic compartment, so the substrate is not
unreasonable -- but it is not MK-restricted and must never be described as if it were.

FEATURE SELECTION AND THE INPUT LOCK. Highly-variable-gene selection would be free to drop the very TFs the
truth panel names, and hand-adding those five would make the truth panel an INPUT -- the exact violation this
project recorded on 2026-09-08. The remedy used there is applied here: force the WHOLE Lambert 2018 human TF
catalogue into the retained feature set, truth-blind. The five MK TFs enter only as members of that catalogue,
on the same footing as every other human TF.

OUTPUT. A frozen h5ad with velocity, and a pairs .npz carrying genes / dc1_min_I,J / dc1_max_I,J in the same
format as the existing GATA1 and PANC frozen-pair artifacts, so the downstream regime and scoring code runs
unchanged. Nothing here reads a panel, a peak, or a perturbation.
"""
import argparse, json, os, sys, time
import numpy as np
import sys as _sys, types as _types
if "tqdm" not in _sys.modules:
    try:
        import tqdm as _t  # noqa: F401
    except Exception:
        _m = _types.ModuleType("tqdm"); _m.tqdm = lambda x=None, *_a, **_k: x
        _m.trange = lambda *a, **k: range(*a); _sys.modules["tqdm"] = _m
        _au = _types.ModuleType("tqdm.auto"); _au.tqdm = _m.tqdm; _sys.modules["tqdm.auto"] = _au
import anndata as ad
from scipy.sparse import issparse
R = __import__("os").environ["GEOBRIDGE_ROOT"]
for _p in (f"{R}/scratch", f"{R}/pairspace", R):
    if _p not in sys.path: sys.path.insert(0, _p)
os.environ.setdefault("OMP_NUM_THREADS", "8")

SRC = f"{R}/data/BoneMarrow/human_cd34_bone_marrow.h5ad"
TFCAT = f"{R}/GB_real_v2/lambert2018_human_tfs.txt"
d_ = lambda M: np.asarray(M.todense()) if issparse(M) else np.asarray(M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--n_top_genes", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out_dir, exist_ok=True)
    import scanpy as sc, scvelo as scv
    scv.settings.verbosity = 1

    A = ad.read_h5ad(SRC)
    A.var_names_make_unique()
    print(f"loaded {A.shape}  layers={list(A.layers)}", flush=True)

    # ---- truth-blind forced feature set: the ENTIRE Lambert human TF catalogue ----
    tfs = {l.strip().upper() for l in open(TFCAT) if l.strip()}
    keep_tf = np.isin(np.char.upper(np.asarray(A.var_names).astype(str)), list(tfs))
    print(f"Lambert TFs present in substrate: {int(keep_tf.sum())}", flush=True)

    # this scvelo build forwards **kwargs from filter_and_normalize into normalize_per_cell, so
    # n_top_genes/retain_genes must be handed to the dispersion filter explicitly instead.
    # scvelo 0.3.4 removed pp.filter_genes_dispersion and forwards **kwargs into normalize_per_cell,
    # so HVG selection is done with scanpy and the TF catalogue is UNIONED in afterwards -- the
    # catalogue must survive selection, or the feature set silently becomes variance-defined only.
    retain = list(np.asarray(A.var_names)[keep_tf])
    scv.pp.filter_and_normalize(A, min_shared_counts=20, retain_genes=retain)
    # scvelo decides on its own whether to logarithmize and can silently skip it; scanpy's seurat HVG
    # then calls expm1 on unlogged values and overflows to inf. Check the data instead of trusting it.
    _mx = float((A.X.data if issparse(A.X) else A.X).max())
    if _mx > 50:
        print(f"X max={_mx:.1f} => not logarithmized by scvelo; applying log1p", flush=True)
        sc.pp.log1p(A)
    sc.pp.highly_variable_genes(A, n_top_genes=a.n_top_genes)
    hv = np.asarray(A.var["highly_variable"]).astype(bool)
    hv |= np.isin(np.char.upper(np.asarray(A.var_names).astype(str)), list(tfs))
    print(f"retained {int(hv.sum())} genes = {a.n_top_genes} HVG union Lambert catalogue", flush=True)
    A = A[:, hv].copy()
    scv.pp.moments(A, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(A, mode="stochastic")
    scv.tl.velocity_graph(A, n_jobs=8)
    print(f"velocity fitted {A.shape}  {time.time()-t0:.0f}s", flush=True)

    genes = np.asarray(A.var_names).astype(str)
    X = d_(A.X).astype(np.float64)
    V = np.nan_to_num(d_(A.layers["velocity"]).astype(np.float64), nan=0.0)
    # velocity genes are a subset; keep every gene but record which carry a fitted velocity
    vel_ok = np.isfinite(d_(A.layers["velocity"])).all(0) & (np.abs(V).sum(0) > 0)

    from gb_gata1_preflight import canonical_pairs
    ids = np.asarray(A.obs_names).astype(str)
    out = dict(genes=genes, velocity_gene=vel_ok)
    for root in ("dc1_min", "dc1_max"):
        pt, I, J, _C = canonical_pairs(X, V, ids, root, seed=a.seed)
        out[f"{root}_I"] = I.astype(np.int64); out[f"{root}_J"] = J.astype(np.int64)
        out[f"{root}_pt"] = pt
        print(f"  {root}: {len(I)} forward pairs  {time.time()-t0:.0f}s", flush=True)

    np.savez_compressed(f"{a.out_dir}/bonemarrow_frozen_pairs.npz", **out)
    A.write_h5ad(f"{a.out_dir}/bonemarrow_velocity.h5ad")

    five = ["GATA1", "GATA2", "RUNX1", "FLI1", "TAL1"]
    gu = np.char.upper(genes)
    present = {g: bool((gu == g).any()) for g in five}
    det = {g: float((X[:, int(np.where(gu == g)[0][0])] > 0).mean()) for g in five if present[g]}
    json.dump(dict(source=SRC, n_cells=int(A.n_obs), n_genes=int(A.n_vars),
                   n_velocity_genes=int(vel_ok.sum()), seed=a.seed,
                   lambert_tfs_retained=int(np.isin(gu, list(tfs)).sum()),
                   truth_panel_TFs_present=present, truth_panel_detection_frac=det,
                   clusters={k: int(v) for k, v in A.obs["clusters"].value_counts().items()},
                   FROZEN_LIMITATION="MK binding truth evaluated on the FULL CD34+ hierarchy; "
                                     "Mega cluster has only 61 cells so no MK-restricted analysis is possible",
                   truth_read=False),
              open(f"{a.out_dir}/substrate.json", "w"), indent=1)
    print("\ntruth-panel TFs present:", present)
    print("truth-panel detection fraction:", {k: round(v, 3) for k, v in det.items()})
    print(f"\nwrote {a.out_dir} in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
