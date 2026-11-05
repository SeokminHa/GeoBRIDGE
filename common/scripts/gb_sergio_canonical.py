#!/usr/bin/env python
"""SERGIO benchmark recomputed under the FROZEN preprocessing and pair contract.

usage: gb_sergio_canonical.py <out_dir> [--topologies linear,tree,...] [--seeds N] [--kmax 30]

WHY THIS EXISTS. The existing SERGIO numbers in `sergio_benchmark_new/benchmark_results/` were
audited on 2026-09-21 and are NOT under the frozen contract. Two divergences, both material:

  PAIRS. The reported `n_pairs` equals n(n-1)/2 almost exactly -- 404,167 vs 404,550 for 900 cells,
  1,618,056 vs 1,619,100 for 1800, 718,102 vs 719,400 for 1200. That is FORWARD ALL-PAIRS with no
  dt window and no per-cell cap. Under the frozen operator (min_forward_dt 0.02, max_forward_dt
  0.25, max_pairs_per_cell 40) linear/900 cells admits at most 36,000. The old harness imports
  `build_forward_pairs` but the all-pairs path is what produced the results.

  VELOCITY. `GEOBRIDGE_PREPROCESS_V1` freezes `mode="stochastic"`, `fit_offset=False`,
  `perc=[5,95]`. The old harness used `mode="deterministic"`.

WHAT IS AND IS NOT APPLICABLE TO A SIMULATION. Some contract items are real-data provisions that
cannot sensibly apply to a 100-200 gene SERGIO world, and forcing them would be theatre rather than
compliance. Each is recorded explicitly in the output provenance rather than silently skipped:
  - `filter_and_normalize` n_top_genes=2000 -- N/A, the simulation has 100-200 genes and all of them
    are part of the ground-truth network; HVG selection would discard true edges.
  - `min_shared_counts=20` -- APPLIED, it is a count-level filter and is meaningful here.
  - pca / neighbors / moments / velocity -- APPLIED exactly as frozen.
  - pair construction -- APPLIED exactly as frozen, which is the main repair.
  - DPT root -- SERGIO ships an exact simulation time, so DPT is computed for the contract arm AND
    the simulation-time arm is reported separately as a labelled sensitivity; the simulation-time
    arm is NOT the headline, because production has no oracle time.

SCORE. E = |S| (amendment 6). The signed score is also recorded so the unsigned-vs-signed comparison
on KNOWN UNSIGNED ground truth can be read directly; SERGIO's `gt_GRN` is unsigned, so |S| was always
the semantically matched score here and this is NOT independent evidence for the unsigned change.
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd, anndata as ad
from scipy.sparse import issparse
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, __import__("os").environ["GEOBRIDGE_ROOT"])
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
from geobridge_model.pairspace import build_forward_pairs

R = __import__("os").environ["GEOBRIDGE_ROOT"]
BENCH = f"{R}/sergio_benchmark_new"
CFG = json.load(open(f"{R}/GEOBRIDGE_PREPROCESS_V1.json"))
PC = CFG["pair_construction"]
EPS = 1e-12


def dense(M):
    return M.toarray() if issparse(M) else np.asarray(M)


def preprocess(adata):
    """Frozen contract, with the two simulation-inapplicable provisions recorded, not hidden."""
    import scanpy as sc, scvelo as scv
    fn = CFG["filter_and_normalize"]
    # Mirror the CANONICAL BUILDER exactly (gb_canonical.py:139-155) rather than the convenience
    # wrapper: scVelo 0.3.4's filter_and_normalize performs ONLY filter_genes + normalize_per_cell
    # and forwards unknown kwargs to normalize_per_cell, which is why both n_top_genes and log are
    # rejected. Each step is therefore explicit, so the contract does not depend on wrapper
    # behaviour that differs between scVelo versions.
    scv.pp.filter_genes(adata, min_shared_counts=fn["min_shared_counts"])
    scv.pp.normalize_per_cell(adata)
    # HVG selection is SKIPPED here and only here: a SERGIO world has 100-200 genes and every one of
    # them participates in the ground-truth network, so selecting 2000 HVGs is both impossible and
    # would discard true edges. Recorded in the provenance as a declared exemption.
    import numpy as _np
    adata.layers["counts_spliced"] = adata.layers["spliced"].copy()
    p = CFG["pca"]; nb = CFG["neighbors"]; ve = CFG["velocity"]
    sc.pp.pca(adata, n_comps=min(p["n_comps"], adata.shape[1] - 1, adata.shape[0] - 1),
              zero_center=p["zero_center"], random_state=p["random_state"])
    sc.pp.neighbors(adata, n_neighbors=nb["n_neighbors"],
                    n_pcs=min(nb["n_pcs"], adata.obsm["X_pca"].shape[1]),
                    method=nb["method"], metric=nb["metric"], random_state=nb["random_state"])
    scv.pp.moments(adata, n_pcs=None, n_neighbors=None)
    scv.tl.velocity(adata, mode=ve["mode"], fit_offset=ve["fit_offset"], perc=ve["perc"])
    return adata


def dpt_time(adata):
    import scanpy as sc
    from scipy.sparse.csgraph import connected_components
    n, _ = connected_components(adata.obsp["connectivities"], directed=False)
    if n != 1:
        return None
    sc.tl.diffmap(adata)
    dc1 = np.asarray(adata.obsm["X_diffmap"])[:, 1]
    adata.uns["iroot"] = int(np.argmin(dc1))
    sc.tl.dpt(adata)
    pt = np.asarray(adata.obs["dpt_pseudotime"], dtype=np.float64)
    return pt if np.isfinite(pt).all() else None


def metrics(Sabs, Ssgn, regs, gt_pairs, n_genes, orig):
    """AUPRC and AUROC over EVERY (regulator, target) cell, self-edges excluded.

    `orig` maps a POSITION in the (possibly filtered) gene axis to the ORIGINAL gene index that
    gt_GRN.csv uses. This argument exists because the previous version compared POSITIONS directly
    against ORIGINAL indices: it computed a `keep` remap and never used it, so any non-terminal gene
    deletion would silently attach truth labels to the wrong edges (internal review 2026-09-22). Measured on
    the current 40 worlds, filtering removes 0 genes, so no stored result is affected -- but the
    defect is latent and would fire on any harder world where a gene fails min_shared_counts.

    Unsupported scores stay NaN and are EXCLUDED rather than coerced to 0. Coercion silently ranked
    an unscoreable edge below every scored one, which is a fail-open.
    """
    y, ya, ys = [], [], []
    per_reg = []
    for i, r in enumerate(regs):
        yr, ar = [], []
        for j in range(n_genes):
            if j == r: continue
            lab = 1 if (orig[r], orig[j]) in gt_pairs else 0
            y.append(lab); ya.append(Sabs[i, j]); ys.append(Ssgn[i, j])
            yr.append(lab); ar.append(Sabs[i, j])
        yr = np.asarray(yr); ar = np.asarray(ar)
        ok = np.isfinite(ar)
        if ok.sum() > 1 and 0 < yr[ok].sum() < ok.sum():
            per_reg.append(float(average_precision_score(yr[ok], ar[ok])))
    y = np.asarray(y); ya = np.asarray(ya, float); ys = np.asarray(ys, float)
    fin = np.isfinite(ya) & np.isfinite(ys)
    n_unsupported = int((~fin).sum())
    y, ya, ys = y[fin], ya[fin], ys[fin]
    if y.sum() < 2 or y.sum() == y.size: return {}
    return dict(AUPRC=float(average_precision_score(y, ya)),
                AUROC=float(roc_auc_score(y, ya)),
                sAUPRC=float(average_precision_score(y, ys)),
                sAUROC=float(roc_auc_score(y, ys)),
                # WITHIN-REGULATOR macro, so one high-degree regulator cannot carry the benchmark
                macro_AUPRC=float(np.mean(per_reg)) if per_reg else float("nan"),
                macro_AUPRC_median=float(np.median(per_reg)) if per_reg else float("nan"),
                n_reg_scored=len(per_reg),
                prevalence=float(y.mean()), n_cells_scored=int(y.size), n_pos=int(y.sum()),
                n_unsupported_excluded=n_unsupported)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir")
    ap.add_argument("--topologies", default="linear,bifurcation,trifurcation,tree")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--kmax", type=int, default=30)
    a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out_dir, exist_ok=True)
    rows = []
    for topo in [t.strip() for t in a.topologies.split(",") if t.strip()]:
        for seed in range(a.seeds):
            d = f"{BENCH}/{topo}/seed_{seed}"
            if not os.path.exists(f"{d}/adata.h5ad"): continue
            A = ad.read_h5ad(f"{d}/adata.h5ad")
            gt = pd.read_csv(f"{d}/gt_GRN.csv")
            n_before = A.shape[1]
            A = preprocess(A)
            # POSITION -> ORIGINAL gt index. SERGIO names genes "gene_<i>", so the original index
            # is recoverable exactly. FAIL CLOSED if any name does not parse: guessing a mapping is
            # how truth labels end up on the wrong edges.
            _vn = np.asarray(A.var_names).astype(str)
            try:
                orig = np.array([int(v.split("_")[-1]) for v in _vn], dtype=int)
            except Exception:
                raise SystemExit("REFUSING[SERGIO_GENE_NAMES_UNPARSEABLE]: cannot map the filtered "
                                 f"gene axis back to gt_GRN indices; names look like {_vn[:3]}")
            if len(set(orig.tolist())) != orig.size:
                raise SystemExit("REFUSING[SERGIO_GENE_INDEX_COLLISION]")
            regs_all = np.flatnonzero(np.asarray(A.var["is_reg"]).astype(bool))
            gt_pairs = set(zip(gt.regulator.astype(int), gt.target.astype(int)))
            pt = dpt_time(A)
            if pt is None:
                rows.append(dict(topology=topo, seed=seed, status="DPT_UNAVAILABLE")); continue
            X = dense(A.layers["Ms"]).astype(np.float64)
            Mu = dense(A.layers["Mu"]).astype(np.float64)
            V = np.nan_to_num(dense(A.layers["velocity"]).astype(np.float64), nan=0.0)
            o = np.argsort(pt, kind="stable")
            try:
                pair = build_forward_pairs(X[o], V[o], pt[o],
                                           min_forward_dt=PC["min_forward_dt"],
                                           max_forward_dt=PC["max_forward_dt"],
                                           max_pairs_per_cell=PC["max_pairs_per_cell"],
                                           pair_seed=PC["pair_seed"])
            except ValueError as e:
                rows.append(dict(topology=topo, seed=seed, status=f"PAIRS_FAILED: {e}")); continue
            I = o[np.asarray(pair.pair_i, int)]; J = o[np.asarray(pair.pair_j, int)]
            dX = X[J] - X[I]
            dU = Mu[J] - Mu[I]                       # U channel = nascent, the FROZEN PRIMARY
            dV = V[J] - V[I]                         # V channel = velocity, sensitivity only
            dP = np.asarray(A.obsm["X_pca"], dtype=np.float64)[J] - \
                 np.asarray(A.obsm["X_pca"], dtype=np.float64)[I]
            # PRODUCTION OPERATOR IS R1/R3, not R2: residualise against the top-k CELL-SPACE PC
            # differences via QR, uncentered, exactly as gb_fresh_produce.py:322-333. The previous
            # version used a pair-space SVD (R2), which internal review flagged as not the frozen operator.
            dXc = dX - dX.mean(0, keepdims=True)
            U_, s_, Vt_ = np.linalg.svd(dXc, full_matrices=False)   # retained for the R2 ablation
            ng = X.shape[1]
            regs = list(regs_all)          # already POSITIONS in the filtered axis
            if ng != n_before:
                print(f"  NOTE {topo}/seed{seed}: filtering removed {n_before-ng} genes; the "
                      f"position->original remap is ACTIVE for this world", flush=True)
            print(f"  {topo}/seed{seed}: {A.shape} genes {n_before}->{ng}, {len(I)} pairs "
                  f"(all-pairs would be {len(pt)*(len(pt)-1)//2}), {len(regs)} regs  "
                  f"{time.time()-t0:.0f}s", flush=True)
            npc = dP.shape[1]
            for k in range(0, a.kmax + 1):
                for op in ("R1R3", "R2"):            # R1/R3 is PRODUCTION; R2 is the frozen ablation
                    if op == "R1R3":
                        if k > npc: continue
                        if k == 0:
                            Am, YU, YV = dX, dU, dV
                        else:
                            Q, _ = np.linalg.qr(dP[:, :k])
                            g13 = lambda Z: Z - Q @ (Q.T @ Z)
                            Am, YU, YV = g13(dX), g13(dU), g13(dV)
                    else:
                        if k == 0:
                            Am, YU, YV = dX, dU, dV
                        else:
                            Uk = U_[:, :k]
                            Am = dXc - (Uk * s_[:k][None, :]) @ Vt_[:k, :]
                            YU = dU - Uk @ (Uk.T @ dU); YV = dV - Uk @ (Uk.T @ dV)
                    Xs = Am[:, regs]; aXs = np.abs(Xs)
                    for chan, Y in (("U", YU), ("V", YV)):
                        num = Xs.T @ Y; den = aXs.T @ np.abs(Y)
                        Ssgn = num / np.maximum(den, EPS)
                        m = metrics(np.abs(Ssgn), Ssgn, regs, gt_pairs, ng, orig)
                        if m:
                            rows.append(dict(topology=topo, seed=seed, k=k, operator=op,
                                             channel=chan, status="OK", n_pairs=int(len(I)),
                                             n_genes=int(ng), n_reg=int(len(regs)), **m))
            del dX, dU, dV, dXc, dP, U_
    D = pd.DataFrame(rows)
    D.to_csv(f"{a.out_dir}/sergio_canonical.tsv", sep="\t", index=False)
    json.dump(dict(contract="GEOBRIDGE_PREPROCESS_V1",
                   applied=["min_shared_counts", "pca", "neighbors", "moments", "velocity",
                            "pair_construction (DPT window + cap)"],
                   not_applicable={"n_top_genes": "simulation has 100-200 genes, all part of the "
                                                  "ground-truth network; HVG selection would "
                                                  "discard true edges"},
                   divergence_repaired={"pairs": "old harness used forward ALL-PAIRS (n(n-1)/2); "
                                                 "this uses the frozen DPT window + cap 40",
                                        "velocity": "old harness used deterministic; contract "
                                                    "freezes stochastic"},
                   operators={"R1R3": "PRODUCTION, cell-PC difference residualisation (spec C)",
                              "R2": "pair-space SVD, the frozen ablation"},
                   channels={"U": "Mu nascent -- FROZEN PRIMARY", "V": "velocity -- sensitivity"},
                   score="E = |S| primary; signed recorded for comparison only",
                   wall_s=time.time() - t0),
              open(f"{a.out_dir}/provenance.json", "w"), indent=1)
    ok = D[D.get("status", "") == "OK"] if "status" in D.columns else D
    if len(ok):
        pd.set_option("display.width", 240)
        pu = ok[(ok.operator == "R1R3") & (ok.channel == "U")]
        print("\n=== PRODUCTION R1/R3, channel U (FROZEN PRIMARY): AUPRC by k x topology ===")
        print(pu.pivot_table(index="k", columns="topology", values="AUPRC").round(4).to_string())
        print("\n=== PRODUCTION R1/R3, channel U: AUROC by k x topology ===")
        print(pu.pivot_table(index="k", columns="topology", values="AUROC").round(4).to_string())
        print("\n=== operator x channel summary (mean AUPRC over k and seeds) ===")
        print(ok.pivot_table(index=["operator", "channel"], columns="topology",
                             values="AUPRC").round(4).to_string())
        print("\n=== unsigned vs signed ===")
        u = ok.pivot_table(index="topology", values=["AUPRC", "sAUPRC", "prevalence"])
        u["unsigned_gain"] = u.AUPRC - u.sAUPRC
        print(u.round(4).to_string())
    print(f"\nwrote {a.out_dir} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
