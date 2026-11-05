#!/usr/bin/env python
"""GPU comparators (RegDiffusion, DeepSEM[, FastTENET]) on World-P confirm datasets; same input as GENIE3 (h5ad .X, all genes, regulators = roster sources).
Truth-free scoring; truth only in `eval`.  Wrappers reused from gb_sota_ext (run_regdiffusion/run_deepsem, seed 0, package/pinned defaults).
  worker: CUDA_VISIBLE_DEVICES=g python gb_topo_adv_gpu.py run METHOD SHARD NSHARD [N]   -> topo_adv/gpu_scores/METHOD/{tag}.npz (+ .fail on error)
  eval:   python gb_topo_adv_gpu.py eval METHODS [N]  -> topo_adv/ROWS_confirm_gpu.tsv"""
import glob, json, os, sys, time, traceback
import numpy as np
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
RES = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/results/sergio_drift"); OUT = f"{RES}/topo_adv/gpu_scores"
PAT = f"{RES}/confirm_P/*/*_P_q3_G*_g1[0-9][0-9]_s[12]/record.json"

def files():
    fs = sorted(glob.glob(PAT)); return [f for f in fs if 101 <= int(json.load(open(f))["grn_seed"]) <= 109]

def run(method, shard, nsh, n=None):
    import anndata as ad, gb_sota_ext as SE
    fs = files()
    if n:  # small-N: first dataset per topology
        seen, sel = set(), []
        for f in fs:
            t = json.load(open(f))["topo"]
            if t not in seen: seen.add(t); sel.append(f)
        fs = sel
    os.makedirs(f"{OUT}/{method}", exist_ok=True)
    for i, rj in enumerate(fs):
        if i % nsh != shard: continue
        dd = os.path.dirname(rj); tag = os.path.basename(dd); o = f"{OUT}/{method}/{tag}.npz"
        if os.path.exists(o): continue
        try:
            a = ad.read_h5ad(f"{dd}/{tag}.h5ad"); ro = json.load(open(f"{dd}/roster.json")); src = np.array(ro["sources"], int); cols = np.array(ro["cols_PI"], int)
            X = np.asarray(a.X, float); nodes = [str(g) for g in a.var_names]
            if (X.std(0) == 0).any(): raise RuntimeError("zero-variance genes")
            P = dict(X=X, node_idx=np.arange(X.shape[1]), tf_node_idx=src, nodes=nodes, tfs=[nodes[s] for s in src])
            t0 = time.time()
            if method == "regdiffusion": M, meta = SE.run_regdiffusion(P, 0)
            elif method == "deepsem": M, meta = SE.run_deepsem(P, 0)
            elif method == "fasttenet": M, meta = run_tenet(P, a, SE)
            M = np.abs(np.asarray(M, float))
            if not np.isfinite(M).all(): raise RuntimeError("nonfinite scores")
            np.savez_compressed(o, score=np.take_along_axis(M, cols, 1), full=M, wall=time.time() - t0)
            print(i, tag, f"{time.time()-t0:.0f}s", flush=True)
        except Exception as e:
            open(f"{OUT}/{method}/{tag}.fail", "w").write(traceback.format_exc()); print("FAIL", tag, repr(e), flush=True)

def run_tenet(P, a, SE):
    import scvelo as scv, fasttenet as fte
    b = a.copy(); scv.pp.neighbors(b, n_neighbors=30, use_rep="X_pca") if "X_pca" in b.obsm else scv.pp.neighbors(b, n_neighbors=30)
    scv.tl.velocity_graph(b, vkey="velocity", n_jobs=1); scv.tl.velocity_pseudotime(b, vkey="velocity")
    pt = np.asarray(b.obs["velocity_pseudotime"], np.float32)
    al = fte.align_data(data=P["X"].T.astype(np.float32), trj=pt, branch=np.ones(len(pt), np.int32))
    w = fte.FastTENET(aligned_data=al, node_name=np.asarray(P["nodes"]), tfs=np.asarray(P["tfs"]))
    # added in release; shipped numbers were produced without it
    np.random.seed(0)
    try:
        import torch; torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    except ImportError:
        pass
    res = w.run(backend="gpu", device_ids=[0], procs_per_device=1, batch_size=2**14, kp=0.5, binning_method="FSBW-L", dt=1)
    ni = {g: i for i, g in enumerate(P["nodes"])}
    return np.array([res[ni[t]] for t in P["tfs"]]), {}

def evaluate(methods, n=None):
    import pandas as pd
    import gb_topo_adv_eval as T
    from sklearn.metrics import average_precision_score as APs, roc_auc_score as AUC
    import gb_sergio_band_eval as E, gb_sergio_drift_v5 as V
    R = []
    for rj in files():
        dd = os.path.dirname(rj); tag = os.path.basename(dd); rec = json.load(open(rj)); g = int(rec["grn_seed"])
        ro = json.load(open(f"{dd}/roster.json")); src, cols = np.array(ro["sources"], int), np.array(ro["cols_PI"], int)
        views = {}
        for m in methods:
            p = f"{OUT}/{m}/{tag}.npz"
            if os.path.exists(p): views[m] = np.load(p)["score"]
        if n and not views: continue
        if not views and not any(os.path.exists(f"{OUT}/{m}/{tag}.fail") for m in methods): continue
        G = V.generate_grn(g); prim = np.array([s in G["regs"] for s in src])
        direct = {(int(s), int(t)) for s, t in G["gt"]}; two = E.strict_two_hop(direct, V.N_GENES)
        Y = np.array([[(s, int(t)) in direct for t in cols[i]] for i, s in enumerate(src)])
        L = np.array([[1 if (s, int(t)) in direct else (0 if (s, int(t)) in two else -1) for t in cols[i]] for i, s in enumerate(src)])
        out = dict(dataset=tag, obs=rec["obs"], topo=rec["topo"], grn=g, G=round(float(rec["G_target"]), 2), sim=int(rec["sim_seed"]))
        for nm, sc in views.items():
            for un, rows in (("REG", prim), ("ALL39", np.ones(len(src), bool))):
                s, y, l = sc[rows], Y[rows], L[rows]; ok = np.isfinite(s); di = ok & (l >= 0)
                out[f"{nm}|{un}|AP"] = APs(y[ok], s[ok]); out[f"{nm}|{un}|AUROC"] = AUC(y[ok], s[ok]); out[f"{nm}|{un}|cov"] = float(ok.mean())
                if un == "REG":
                    out[f"{nm}|{un}|DvI"] = AUC(l[di] == 1, s[di])
                    out[f"{nm}|{un}|macroAP"] = np.mean([APs(y[i][ok[i]], s[i][ok[i]]) for i in range(len(y)) if y[i][ok[i]].any()])
        R.append(out)
    D = pd.DataFrame(R); fn = f"{RES}/topo_adv/ROWS_confirm_gpu{'_n' if n else ''}.tsv"; D.to_csv(fn, sep="\t", index=False); print(len(D), "rows ->", fn)

if __name__ == "__main__":
    c = sys.argv[1]
    if c == "run": run(sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5] if len(sys.argv) > 5 else None)
    else: evaluate(sys.argv[2].split(","), sys.argv[3] if len(sys.argv) > 3 else None)
