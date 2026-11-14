#!/usr/bin/env python
# GSE274113 CRISPRi truth panel -- the independent-study endpoint for bone-marrow transfer
import argparse, glob, json, os, re, time
import numpy as np, pandas as pd, h5py
from scipy import sparse, stats

R = __import__("os").environ["GEOBRIDGE_ROOT"]
D = f"{R}/data/GSE274113"
META = f"{D}/GSE274113_annotated_metadata.csv.gz"
CTRL = "NT"


# BH, NaN-safe
# A NaN p (zero-variance gene) is treated as p = 1, never dropped: np.minimum propagates NaN
def bh(p):
    p = np.asarray(p, float); n = p.size
    p = np.where(np.isfinite(p), p, 1.0)
    o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


# Gene-Expression block of every replicate h5, barcodes prefixed as in the metadata index
def load_reps():
    mats, cells, genes = [], [], None
    for path in sorted(glob.glob(f"{D}/GSE274113_rep*_filtered_feature_bc_matrix.h5"),
                       key=lambda s: int(re.search(r"rep(\d+)_", s).group(1))):
        rep = re.search(r"(rep\d+)_", os.path.basename(path)).group(1)
        with h5py.File(path, "r") as f:
            g = f["matrix"]; ft = g["features"]
            typ = np.array(ft["feature_type"]).astype(str)
            gex = np.flatnonzero(typ == "Gene Expression")
            nm = np.array(ft["name"]).astype(str)[gex]
            if genes is None: genes = nm
            assert np.array_equal(genes, nm), f"{rep}: feature order differs"
            nf, nc = np.array(g["shape"])
            M = sparse.csc_matrix((np.array(g["data"]), np.array(g["indices"]),
                                   np.array(g["indptr"])), shape=(nf, nc))
            M = M[gex].T.tocsr().astype(np.float32)          # cells x genes
            bc = np.array(g["barcodes"]).astype(str)
        mats.append(M); cells.append(np.char.add(f"{rep}_", bc))
        print(f"  {rep}: {M.shape[0]} cells, {M.nnz} nnz  {time.time()-T0:.0f}s", flush=True)
    return sparse.vstack(mats, format="csr"), np.concatenate(cells), genes


# cnt/mean/var per group, sparse-only (never densifies)
def group_moments(X, code, K):
    n = X.shape[0]
    Ind = sparse.csr_matrix((np.ones(n, np.float32), (np.arange(n), code)), shape=(n, K))
    cnt = np.asarray(Ind.sum(0)).ravel()
    s1 = np.asarray((Ind.T @ X).todense())
    s2 = np.asarray((Ind.T @ X.multiply(X)).todense())
    mean = s1 / np.maximum(cnt[:, None], 1)
    var = np.maximum(s2 / np.maximum(cnt[:, None], 1) - mean ** 2, 0) * \
        (cnt[:, None] / np.maximum(cnt[:, None] - 1, 1))
    return cnt, mean, var


def welch(mt, vt, nt, mc, vc, nc):
    a, b = np.nan_to_num(vt / max(nt, 1)), np.nan_to_num(vc / max(nc, 1))
    se = np.sqrt(a + b)
    t = (mt - mc) / np.maximum(se, 1e-300)
    df = np.maximum((a + b) ** 2 / (np.maximum(a ** 2 / max(nt - 1, 1), 1e-300) +
                                    np.maximum(b ** 2 / max(nc - 1, 1), 1e-300)), 1)
    return t, 2 * stats.t.sf(np.abs(t), df)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("--min_cells", type=int, default=200)
    ap.add_argument("--min_strat", type=int, default=30); ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--reps", default="", help="comma-separated replicate subset (truth-ceiling halves); default = all")
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)

    X, cells, genes = load_reps()
    md = pd.read_csv(META, index_col=0)
    md = md[md.index.notna() & md.target.notna()]
    keep = pd.Index(cells).isin(md.index)   # hash-based; np.isin falls back to an O(n*m)
                                            # loop when either side is object dtype
    print(f"{X.shape[0]} matrix cells; {int(keep.sum())} carry a guide assignment", flush=True)
    X, cells = X[keep], cells[keep]
    if a.reps:
        sub = np.isin(np.char.partition(cells.astype(str), "_")[:, 0], a.reps.split(","))
        X, cells = X[sub], cells[sub]
        print(f"replicate subset {a.reps}: {int(sub.sum())} cells", flush=True)
    md = md.reindex(cells)

    cp = np.asarray(X.sum(1)).ravel()                       # CP10K + log1p, recorded in the receipt
    X = sparse.diags(1e4 / np.maximum(cp, 1)).astype(np.float32) @ X
    X.data = np.log1p(X.data)

    tgt = md.target.astype(str).values
    ct = md.new_CellType.astype(str).fillna("NA").values
    ok_t = [t for t, n in zip(*np.unique(tgt, return_counts=True)) if n >= a.min_cells and t != CTRL]
    print(f"targets with >= {a.min_cells} cells: {len(ok_t)}  |  NT cells: {int((tgt == CTRL).sum())}", flush=True)

    # per-gene eligibility, so the DOWNSTREAM universe is testable genes only
    # A gene that is simply not expressed in this system comes out "not significant" for every
    # target.
    # Counting such genes as NEGATIVES would reward any score that merely tracks abundance, so the
    # evaluable universe is defined here, from CONTROL cells only, and never from the test result.
    Xnt = X[tgt == CTRL]
    det = np.asarray((Xnt > 0).sum(0)).ravel() / max(Xnt.shape[0], 1)
    pd.DataFrame(dict(gene=genes, nt_detect_frac=det,
                      nt_mean_log=np.asarray(Xnt.mean(0)).ravel())).to_csv(
        f"{a.out_dir}/gene_stats_nt.tsv", sep="\t", index=False)
    print(f"NT detection: {int((det >= 0.05).sum())} genes in >= 5% of the {Xnt.shape[0]} NT cells, "
          f"{int((det >= 0.10).sum())} in >= 10%", flush=True)

    # composition diagnostic
    comp = pd.crosstab(tgt, ct, normalize="index")
    comp.loc[[t for t in [CTRL] + ok_t if t in comp.index]].round(4).to_csv(f"{a.out_dir}/composition.tsv", sep="\t")
    l1 = {t: float(np.abs(comp.loc[t] - comp.loc[CTRL]).sum()) for t in ok_t if t in comp.index}
    print("composition shift vs NT (L1, 0 = identical, 2 = disjoint):")
    for t, v in sorted(l1.items(), key=lambda kv: -kv[1]): print(f"  {t:8s} {v:.3f}")

    # POOLED panel
    lev = [CTRL] + ok_t
    code = np.full(len(tgt), -1, int)
    for i, t in enumerate(lev): code[tgt == t] = i
    m = code >= 0
    cnt, mean, var = group_moments(X[m], code[m], len(lev))
    rows = []
    for i, t in enumerate(lev):
        if i == 0: continue
        tt, pp = welch(mean[i], var[i], cnt[i], mean[0], var[0], cnt[0])
        q = bh(pp); d = mean[i] - mean[0]
        for j in np.flatnonzero((q < a.alpha) & np.isfinite(tt) & (genes != t)):
            rows.append(dict(tf=t, target=genes[j], dlog=float(d[j]), t=float(tt[j]), p=float(pp[j]),
                             padj=float(q[j]), polarity=1.0 if d[j] > 0 else -1.0, n_tf_cells=int(cnt[i]),
                             panel="pooled"))
    P = pd.DataFrame(rows)

    # STRATIFIED panel: Welch within cell type, Stouffer across
    rows, full = [], []
    cts = [c for c, n in zip(*np.unique(ct, return_counts=True)) if n >= a.min_strat * 2]
    for t in ok_t:
        num = np.zeros(len(genes)); den = 0.0; dsum = np.zeros(len(genes)); wsum = 0.0
        for c in cts:
            mt_ = (tgt == t) & (ct == c); mc_ = (tgt == CTRL) & (ct == c)
            n1, n0 = int(mt_.sum()), int(mc_.sum())
            if n1 < a.min_strat or n0 < a.min_strat: continue
            sub = np.flatnonzero(mt_ | mc_)
            cd = (tgt[sub] == t).astype(int)
            cnt_, mean_, var_ = group_moments(X[sub], cd, 2)
            tt, pp = welch(mean_[1], var_[1], cnt_[1], mean_[0], var_[0], cnt_[0])
            z = np.sign(tt) * stats.norm.isf(np.clip(pp / 2, 1e-300, 0.5))
            w = np.sqrt(1.0 / (1.0 / n1 + 1.0 / n0))
            num += w * np.nan_to_num(z); den += w ** 2
            dsum += w * (mean_[1] - mean_[0]); wsum += w
        if den == 0: continue
        Z = num / np.sqrt(den); pp = 2 * stats.norm.sf(np.abs(Z)); q = bh(pp); d = dsum / max(wsum, 1e-12)
        full.append(pd.DataFrame(dict(tf=t, gene=genes, Z=Z, dlog=d, padj=q)))   # every gene (cognate-excess analysis)
        for j in np.flatnonzero((q < a.alpha) & np.isfinite(Z) & (genes != t)):
            rows.append(dict(tf=t, target=genes[j], dlog=float(d[j]), t=float(Z[j]), p=float(pp[j]),
                             padj=float(q[j]), polarity=1.0 if d[j] > 0 else -1.0,
                             n_tf_cells=int((tgt == t).sum()), panel="stratified"))
    Sp = pd.DataFrame(rows)
    pd.concat(full).to_csv(f"{a.out_dir}/gse274113_stratified_full.tsv.gz", sep="\t", index=False)

    for nm, T in (("pooled", P), ("stratified", Sp)):
        T.to_csv(f"{a.out_dir}/gse274113_{nm}_panel.tsv", sep="\t", index=False)
    summ = []
    for nm, T in (("pooled", P), ("stratified", Sp)):
        if not len(T): continue
        g = T.groupby("tf").agg(n_sig=("target", "size"),
                                n_activated=("polarity", lambda s: int((s < 0).sum())),
                                n_repressed=("polarity", lambda s: int((s > 0).sum())),
                                n_cells=("n_tf_cells", "first")).reset_index()
        g["panel"] = nm; g["composition_L1"] = g.tf.map(l1); summ.append(g)
    SU = pd.concat(summ).sort_values(["panel", "n_sig"], ascending=[True, False])
    SU.to_csv(f"{a.out_dir}/gse274113_tf_summary.tsv", sep="\t", index=False)
    json.dump(dict(source="GSE274113 CRISPRi Perturb-multiome, human CD34+ HSPC -> erythroid",
                   n_cells_assigned=int(X.shape[0]), n_control=int((tgt == CTRL).sum()),
                   n_targets=len(ok_t), min_cells=a.min_cells, min_strat=a.min_strat, alpha=a.alpha,
                   normalisation="CP10K then log1p, on raw filtered_feature_bc_matrix Gene Expression",
                   tests="pooled = Welch vs NT; stratified = Welch within new_CellType, Stouffer sqrt(n_eff)",
                   polarity="KO-up = repression target (+1); KO-down = activation target (-1)",
                   caveat="chronic CRISPRi response, days 7-14: indirect effects included; functional "
                          "response, NOT direct binding. Pooled panel is CONFOUNDED by composition shift.",
                   role="VALIDATION TRUTH ONLY -- never an input to score, k, operator or universe",
                   eligibility="gene_stats_nt.tsv gives NT-only detection fraction; downstream "
                               "evaluation must restrict the universe to testable genes so that "
                               "unexpressed genes are not counted as negatives",
                   wall_s=time.time() - T0),
              open(f"{a.out_dir}/provenance.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(f"\nGSE274113 panels\n{SU.to_string(index=False)}")
    print(f"\nwrote {a.out_dir} in {time.time()-T0:.0f}s")


T0 = time.time()
if __name__ == "__main__":
    main()
