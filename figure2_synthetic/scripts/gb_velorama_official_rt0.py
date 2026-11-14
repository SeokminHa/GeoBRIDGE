#!/usr/bin/env python
# Official Velorama (rs239/velorama cli, unmodified) on World-P SERGIO datasets
import glob, json, os, shutil, subprocess, sys, tempfile
import numpy as np

R = __import__("os").environ["GEOBRIDGE_ROOT"]
RES = f"{R}/results/sergio_drift"
OUT = f"{RES}/topo_adv/velorama_rt0"  # reg_target=0: exclude target history.
VELO = __import__("os").path.join(__import__("os").environ.get("GB_VELORAMA_BIN", __import__("os").path.dirname(__import__("sys").executable)), "velorama")
ARGS = "-s 0 -p H -l 5 -hd 16 -mi 10000 -lr 0.01 -nl 19 -ls -2 -le 1 -nn 15 -nc 50 -rt 0 -xn zscore -pr 1 -vm stochastic".split()


def run(dd, gpu):
    import anndata as ad, pandas as pd
    tag = os.path.basename(dd.rstrip("/"))
    dst = f"{OUT}/{tag}.npz"
    if os.path.exists(dst):
        return
    os.makedirs(OUT, exist_ok=True)
    a = ad.read_h5ad(f"{dd}/{tag}.h5ad")
    cands = json.load(open(f"{dd}/cands.json"))["cands"]
    Z = np.load(f"{dd}/raw/geobridge_raw.npz")
    src, cols = np.asarray(Z["source_gene_index"], int), np.asarray(Z["cols_PI"], int)
    assert list(src) == list(cands) and a.n_vars == 100
    w = ad.AnnData(X=np.asarray(a.X, np.float32), obs=a.obs[[]].copy(),
                   layers={l: np.asarray(a.layers[l].toarray() if hasattr(a.layers[l], "toarray") else a.layers[l], np.float32)
                           for l in ("spliced", "unspliced")})
    w.var_names = [str(i) for i in range(a.n_vars)]
    w.var["is_reg"] = np.isin(np.arange(a.n_vars), src)
    w.var["is_target"] = True
    w.uns["iroot"] = 0                                   # unused by rna_velocity dynamics
    tmp = tempfile.mkdtemp(prefix=f"vo{gpu}_", dir="/tmp")
    try:
        name = "wp"                                      # no 'De-noised' substring: X is already log-normalised
        import scipy.sparse as sp
        w.X = sp.csr_matrix(w.X)                         # Velorama run.py calls .X.toarray()
        w.write_h5ad(f"{tmp}/{name}.h5ad")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), TMPDIR=tmp, RAY_TMPDIR=tmp, PYTHONNOUSERSITE="1", MPLCONFIGDIR=tmp)
        cmd = [VELO, "-ds", name, "-n", name, "-rd", tmp, "-sd", f"{tmp}/res", "-dyn", "rna_velocity", "-dev", os.environ.get("GB_VELO_DEV", "cuda"), "-cp", "2"] + ARGS
        p = subprocess.run(cmd, cwd=tmp, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
        tsv = f"{tmp}/res/{name}.rna_velocity.velorama.interactions.tsv"
        if p.returncode or not os.path.exists(tsv):
            open(f"{OUT}/{tag}.FAILED.log", "w").write(p.stdout[-20000:])
            raise RuntimeError(f"velorama failed: {tag}")
        shutil.copy(tsv, f"{OUT}/{tag}.interactions.tsv")
        t = pd.read_csv(tsv, sep="\t", index_col=0)      # rows = targets (all 100), cols = regs (39 cands)
        t.index = t.index.astype(int); t.columns = t.columns.astype(int)   # rt0: cols = all 100 genes
        assert sorted(t.columns) == list(range(100)) and sorted(t.index) == list(range(100))
        GC = t.reindex(index=range(100), columns=list(src)).values.astype(float)   # keep the 39 candidate sources
        assert np.isfinite(GC).all()
        sc = np.stack([GC[cols[i], i] for i in range(len(src))])          # (39, 99) aligned with geobridge cols_PI
        np.savez_compressed(dst, score=sc, gc=GC, src=src, cols=cols)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def eval_rows(n=None):
    import pandas as pd
    from multiprocessing import Pool
    sys.path.insert(0, f"{R}/scratch")
    import gb_topo_adv_eval as T
    files = sorted(glob.glob(T.PAT["confirm"]))
    files = [f for f in files if 101 <= int(json.load(open(f))["grn_seed"]) <= 109]
    if n:
        files = files[:int(n)]
    with Pool(16) as p:
        rows = [r for r in p.map(_one, files) if r is not None]
    D = pd.DataFrame(rows)
    D.to_csv(f"{RES}/topo_adv/ROWS_confirm_velorama_rt0{'_n' + str(n) if n else ''}.tsv", sep="\t", index=False)
    print(len(files), "records,", len(D), "scored")


# mirror of gb_topo_adv_eval.one restricted to the VELORAMA view (same truth, universes, metrics)
def _one(rj):
    sys.path.insert(0, f"{R}/scratch")
    import gb_topo_adv_eval as T
    dd = os.path.dirname(rj); tag = os.path.basename(dd); rec = json.load(open(rj))
    f = f"{OUT}/{tag}.npz"
    if not os.path.exists(f):
        return None
    S = np.load(f); g = int(rec["grn_seed"])
    G = T.V.generate_grn(g)
    src, cols = np.asarray(S["src"], int), np.asarray(S["cols"], int)
    Zs = np.load(f"{dd}/raw/geobridge_raw.npz")
    assert (src == Zs["source_gene_index"]).all() and (cols == Zs["cols_PI"]).all()
    prim = np.array([s in G["regs"] for s in src])
    direct = {(int(s), int(t)) for s, t in G["gt"]}
    two = T.E.strict_two_hop(direct, T.V.N_GENES)
    Y = np.array([[(s, int(t)) in direct for t in cols[i]] for i, s in enumerate(src)])
    L = np.array([[1 if (s, int(t)) in direct else (0 if (s, int(t)) in two else -1) for t in cols[i]] for i, s in enumerate(src)])
    sc = np.asarray(S["score"], float)
    out = dict(dataset=tag, obs=rec["obs"], topo=rec["topo"], grn=g, G=round(float(rec["G_target"]), 2), sim=int(rec["sim_seed"]))
    nm = "VELORAMA"
    for un, rows in (("REG", prim), ("ALL39", np.ones(len(src), bool))):
        s, y, l = sc[rows], Y[rows], L[rows]
        ok = np.isfinite(s); di = ok & (l >= 0)
        out[f"{nm}|{un}|AP"] = T.APs(y[ok], s[ok])
        out[f"{nm}|{un}|AUROC"] = T.AUC(y[ok], s[ok])
        out[f"{nm}|{un}|cov"] = float(ok.mean())
        if un == "REG":
            out[f"{nm}|{un}|DvI"] = T.AUC(l[di] == 1, s[di])
            out[f"{nm}|{un}|macroAP"] = np.mean([T.APs(y[i][ok[i]], s[i][ok[i]]) for i in range(len(y)) if y[i][ok[i]].any()])
    out["prev_REG"], out["prev_ALL39"] = Y[prim].mean(), Y.mean()
    return out


def summary():
    import pandas as pd
    sys.path.insert(0, f"{R}/scratch")
    import gb_topo_adv_eval as T
    V = pd.read_csv(f"{RES}/topo_adv/ROWS_confirm_velorama_rt0.tsv", sep="\t")
    B = pd.read_csv(f"{RES}/topo_adv/ROWS_confirm.tsv", sep="\t")
    mets = ["REG|AP", "REG|AUROC", "REG|DvI", "REG|macroAP", "ALL39|AP", "ALL39|AUROC"]
    keep = ["dataset"] + [f"{m}|{x}" for m in ("GENIE3", "PRIMARY") for x in mets]
    D = V.merge(B[keep], on="dataset", how="inner")
    cols = [f"{m}|{x}" for m in ("VELORAMA", "GENIE3", "PRIMARY") for x in mets]
    G = T.grn_level(D, cols, ("topo",))
    order = ["linear", "bifurcation", "trifurcation", "tree"]
    M = G.groupby("topo")[cols].mean().reindex(order)
    n = D.groupby("topo").size().reindex(order)
    print(f"scored {len(D)}/624 datasets; per topology:", dict(n))
    for x in mets:
        print(x.ljust(12), " | ".join(f"{m} " + "/".join(f"{v:.3f}" for v in M[f"{m}|{x}"]) for m in ("VELORAMA", "GENIE3", "PRIMARY")))
    d = G.assign(dAP=G["VELORAMA|REG|AP"] - G["GENIE3|REG|AP"]).groupby("topo")["dAP"].agg(["mean", "min", "max", "count"]).reindex(order)
    print("VELORAMA - GENIE3 REG AP, GRN-level (n GRNs):\n", d.round(3).to_string())
    M.to_csv(f"{RES}/topo_adv/VELORAMA_TOPO_MEANS.tsv", sep="\t")


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "eval":
        eval_rows(sys.argv[2] if len(sys.argv) > 2 else None)
    elif sys.argv[1] == "summary":
        summary()
