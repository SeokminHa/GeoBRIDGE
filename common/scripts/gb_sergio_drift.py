#!/usr/bin/env python
"""SERGIO HIDDEN-DRIVER DRIFT pilot (lane SERGIO-DRIFT; DESCRIPTIVE / PRELIMINARY pending internal review).

One fixed Velorama-style 100-gene cascade GRN (benchmark/sergio_benchmark_new.generate_grn(GRN_SEED)). Exogenous shared
developmental programme = q HIDDEN master regulators whose per-bin basal production follows a smooth profile along the
differentiation graph, with hidden->gene edges of strength lam*K. Hidden genes and their edges are REMOVED from the output and
from the truth, so the observed truth is identical across lam (asserted). Worlds: A = hidden loads only on non-TF targets
(leaves); B = also on TF targets (intermediates); C = high rank (one hidden driver per loaded gene). SERGIO itself is unmodified.
Pipeline = scratch/gb_sergio_canonical.preprocess (frozen contract) + DPT roots dc1_min/dc1_max + frozen forward pairs + R1/R3
QR conditioning (production); production scalar = PROSPECTIVE_SCORING_SPEC_v1 (per source within-universe percentile per
(root,k), mean over K_ADM 8..30, min over roots), pooled over all (regulator, target) edges.
usage: gb_sergio_drift.py <out_dir> --topo bifurcation --world A --q 3 --lams 0,1,3 --seeds 0,1 [--nohidden]"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
from collections import defaultdict
sys.path.insert(0, __import__("os").environ["GEOBRIDGE_ROOT"]); sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/benchmark")); sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/SERGIO"))
import sergio_benchmark_new as SB
from gb_sergio_canonical import preprocess, dense, PC
from geobridge_model.pairspace import build_forward_pairs
from sklearn.metrics import average_precision_score, roc_auc_score

TARGET_PREV = 0.05          # PI: candidate set sized so true edges / candidate TF x target edges ~ .05
GRN_SEED = 0; KA = list(range(8, 31)); KMAX = 30; EPS = 1e-12
ARMS = ["TRUE", "REV", "RIS0", "ROS0"]                         # ponytail: 1 random draw per arm in the pilot; 5 in the full grid
DEPTH = {"linear": [0, 1, 2], "bifurcation": [0, 1, 2, 2], "trifurcation": [0, 1, 1, 1], "tree": [0, 1, 2, 3, 3, 3]}
BRANCH = {"linear": [0, 0, 0], "bifurcation": [0, 0, 1, -1], "trifurcation": [0, 1, -1, 0], "tree": [0, 0, 0, 1, -1, 0]}


def profiles(topo, q):
    """per-bin basal rates for q hidden drivers: monotone up, peaked, branch-specific, monotone down, repeat with shifts"""
    d = np.asarray(DEPTH[topo], float); d = d / max(d.max(), 1); b = np.asarray(BRANCH[topo], float)
    base = [d, np.exp(-((d - .5) ** 2) / .08), .5 + .5 * b, 1 - d]
    out = []
    for m in range(q):
        p = base[m % 4]
        p = (p - p.min()) / max(np.ptp(p), 1e-9)
        out.append(0.3 + 3.5 * (np.roll(p, m // 4) if m >= 4 else p))
    return np.array(out)


def write_inputs(outdir, mrs, edges, n_bins, hidden_rates, hidden_edges, n_obs):
    """Same MR-rate rng as sergio_benchmark_new.write_sergio_inputs; hidden MRs appended; hidden->gene edges appended"""
    os.makedirs(outdir, exist_ok=True); rng = np.random.RandomState(GRN_SEED + 5000)
    lines = []
    for mr in mrs:
        base = rng.uniform(0.5, 4.0)
        lines.append("{},{}".format(float(mr), ",".join(str(max(0.01, base + rng.uniform(-1.5, 1.5))) for _ in range(n_bins))))
    for h, r in enumerate(hidden_rates):
        lines.append("{},{}".format(float(n_obs + h), ",".join(str(x) for x in r)))
    open(f"{outdir}/input_MRs.txt", "w").write("\n".join(lines) + "\n")
    tr = defaultdict(list)
    for tgt, src, K, hill in list(edges) + list(hidden_edges):
        tr[tgt].append((src, K, hill))
    il = ["{},{},{},{},{}".format(float(t), float(len(v)), ",".join(str(float(s)) for s, _, _ in v), ",".join(str(k) for _, k, _ in v),
                                  ",".join(str(h) for _, _, h in v)) for t, v in sorted(tr.items())]
    open(f"{outdir}/input_GRN.txt", "w").write("\n".join(il) + "\n")
    return f"{outdir}/input_MRs.txt", f"{outdir}/input_GRN.txt"


def simulate(topo, world, q, lam, sim_seed, outdir, nohidden=False):
    from SERGIO.sergio import sergio
    mrs, edges, gt = SB.generate_grn(GRN_SEED)
    tgt_of = defaultdict(list)
    for t, s, _, _ in edges:
        tgt_of[s].append(t)
    regs = sorted(tgt_of); n_obs = SB.N_GENES
    leaves = sorted(set(range(n_obs)) - set(regs) - set(mrs)); tfs_tg = sorted(set(regs) - set(mrs))   # MRs cannot be targets
    lrng = np.random.RandomState(777 + GRN_SEED)                                     # loadings: independent of lam and sim seed
    elig = leaves if world == "A" else leaves + tfs_tg
    if world == "C":
        q = len(elig)
    hidden_edges = []
    if not nohidden:
        for h in range(q):
            tg = [elig[h]] if world == "C" else list(lrng.choice(elig, size=int(.4 * len(elig)), replace=False))
            for t in tg:
                hidden_edges.append((t, n_obs + h, lam * lrng.uniform(1.0, 3.0) * lrng.choice([-1, 1]), 2.0))
    cfg = SB.TOPOLOGIES[topo]; hr = profiles(topo, q) if not nohidden else np.zeros((0, cfg["n_bins"]))
    mf, gf = write_inputs(outdir, mrs, edges, cfg["n_bins"], hr, hidden_edges, n_obs)
    np.random.seed(sim_seed)
    P = SB.SERGIO_PARAMS
    sim = sergio(number_genes=n_obs + len(hr), number_bins=cfg["n_bins"], number_sc=SB.CELLS_PER_TYPE, noise_params=P["noise_params"],
                 noise_type=P["noise_type"], decays=P["decays"], sampling_state=P["sampling_state"],
                 noise_params_splice=P["noise_params_splice"], splice_ratio=P["splice_ratio"], dt=P["dt"], dt_splice=P["dt_splice"],
                 dynamics=True, bifurcation_matrix=cfg["bmat"])
    sim.build_graph(input_file_taregts=gf, input_file_regs=mf, shared_coop_state=2)
    sim.simulate_dynamics(); U, S = sim.getExpressions_dynamics()
    S = np.maximum(np.vstack([S[b].T for b in range(cfg["n_bins"])]), 0); U = np.maximum(np.vstack([U[b].T for b in range(cfg["n_bins"])]), 0)
    # candidate sources = true regulators + DECOY genes (fixed rng, independent of lam/seed/results) so prevalence ~ TARGET_PREV
    drng = np.random.RandomState(999 + GRN_SEED); pool = [g for g in range(n_obs) if g not in regs]
    n_dec = max(0, int(round(len(gt) / (TARGET_PREV * (n_obs - 1)))) - len(regs))
    cands = sorted(regs + list(drng.choice(pool, size=min(n_dec, len(pool)), replace=False)))
    return S, U, cands, mrs, gt, n_obs


def score_all(A, regs):
    """per root: forward pairs, R1/R3 arms, |S| U and X for every (reg, gene) at k 0..30. Returns dict[(arm,chan,root)] -> (31, R, G)"""
    import scanpy as sc
    X = dense(A.layers["Ms"]).astype(float); Mu = dense(A.layers["Mu"]).astype(float)
    Vv = np.nan_to_num(dense(A.layers["velocity"]).astype(float)); P30 = np.asarray(A.obsm["X_pca"], float)[:, :30]
    sc.tl.diffmap(A); dc1 = np.asarray(A.obsm["X_diffmap"])[:, 1]
    out, coh, pairs = {}, {}, {}
    for root, ir in (("dc1_min", int(np.argmin(dc1))), ("dc1_max", int(np.argmax(dc1)))):
        A.uns["iroot"] = ir; sc.tl.dpt(A); pt = np.asarray(A.obs["dpt_pseudotime"], float)
        o = np.argsort(pt, kind="stable")
        pr = build_forward_pairs(X[o], Vv[o], pt[o], min_forward_dt=PC["min_forward_dt"], max_forward_dt=PC["max_forward_dt"],
                                 max_pairs_per_cell=PC["max_pairs_per_cell"], pair_seed=PC["pair_seed"])
        I = o[np.asarray(pr.pair_i, int)]; J = o[np.asarray(pr.pair_j, int)]; pairs[root] = (I, J)
        dX, dU, dP = X[J] - X[I], Mu[J] - Mu[I], P30[J] - P30[I]
        c = (dX * Vv[I]).sum(1) / np.maximum(np.linalg.norm(dX, axis=1) * np.linalg.norm(Vv[I], axis=1), EPS); coh[root] = float(np.median(c))
        Qt = np.linalg.qr(dP)[0]
        for arm in ARMS:
            if arm == "TRUE": Q = Qt
            elif arm == "REV": Q = np.linalg.qr(dP[:, ::-1])[0]
            elif arm == "RIS0": Q = np.linalg.qr(dP @ np.linalg.qr(np.random.Generator(np.random.PCG64(0)).standard_normal((dP.shape[1],) * 2))[0])[0]
            else:
                G = np.random.Generator(np.random.PCG64(100)).standard_normal((dP.shape[0], dP.shape[1])); Q = np.linalg.qr(G - Qt @ (Qt.T @ G))[0]
            for chan, Y in (("U", dU), ("X", dX)):
                arr = np.full((KMAX + 1, len(regs), X.shape[1]), np.nan)
                for k in range(KMAX + 1):
                    if k > Q.shape[1]: break
                    Qk = Q[:, :k]; a = dX[:, regs] - Qk @ (Qk.T @ dX[:, regs]) if k else dX[:, regs]
                    y = Y - Qk @ (Qk.T @ Y) if k else Y
                    arr[k] = np.abs(a.T @ y) / np.maximum(np.abs(a).T @ np.abs(y), EPS)
                for i, r in enumerate(regs):
                    arr[:, i, r] = np.nan
                out[(arm, chan, root)] = arr
    return out, coh, pairs, X


def pct_rows(M):
    """within-source percentile over targets (NaN kept)"""
    return pd.DataFrame(M).rank(axis=1, pct=True).to_numpy()


def production_scalar(out, arm, chan):
    per = [np.nanmean(np.stack([pct_rows(out[(arm, chan, r)][k]) for k in KA]), 0) for r in ("dc1_min", "dc1_max")]
    return np.minimum(*per)


def evaluate(score, regs, gt, n_obs, closure):
    y, s, cls = [], [], []
    for i, r in enumerate(regs):
        for j in range(n_obs):
            if j == r or not np.isfinite(score[i, j]): continue
            y.append((r, j) in gt); s.append(score[i, j]); cls.append("direct" if (r, j) in gt else "indirect" if (r, j) in closure else "none")
    y = np.array(y); s = np.array(s); cls = np.array(cls); o = np.argsort(-s, kind="stable")
    res = dict(AUPRC=average_precision_score(y, s), AUROC=roc_auc_score(y, s), prevalence=y.mean(), n_cand=int(y.size), n_true=int(y.sum()))
    res["AUPRC_lift"] = res["AUPRC"] / res["prevalence"]
    di = cls != "none"                                            # direct-over-indirect: among reachable pairs only
    res["AUROC_direct_vs_indirect"] = roc_auc_score(cls[di] == "direct", s[di]) if 0 < (cls[di] == "direct").sum() < di.sum() else np.nan
    for K in (20, 50, 100):
        res[f"P@{K}"] = y[o[:K]].mean(); res[f"indirect@{K}"] = (cls[o[:K]] == "indirect").mean()
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--topo", default="bifurcation"); ap.add_argument("--world", default="A")
    ap.add_argument("--q", type=int, default=3); ap.add_argument("--lams", default="0,1,3"); ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--nohidden", action="store_true"); a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import anndata as ad
    rows, truth_hash = [], set()
    for seed in [int(x) for x in a.seeds.split(",")]:
        for lam in ([0.0] if a.nohidden else [float(x) for x in a.lams.split(",")]):
            t0 = time.time(); tag = f"{a.topo}_{a.world}_q{a.q}_lam{lam:g}_s{seed}" + ("_nohidden" if a.nohidden else "")
            S, U, regs, mrs, gt, n_obs = simulate(a.topo, a.world, a.q, lam, seed, f"{a.out}/{tag}", a.nohidden)
            truth_hash.add(hash(frozenset(gt)))
            cl = SB._transitive_closure(gt, n_obs)
            Hs = S[:, n_obs:]; A = ad.AnnData(X=S[:, :n_obs] + U[:, :n_obs], layers=dict(spliced=S[:, :n_obs].copy(), unspliced=U[:, :n_obs].copy()))
            A.var_names = [f"gene_{i}" for i in range(n_obs)]
            A = preprocess(A)
            assert A.shape[1] == n_obs, "gene filtering removed genes; position->index remap needed"
            out, coh, pairs, X = score_all(A, regs)
            # G: pooled R^2 of observed log-normalised pair differences on hidden-driver pair differences (root dc1_min pairs)
            I, J = pairs["dc1_min"]; lib = (S[:, :n_obs].sum(1, keepdims=True) + 1e-9)
            Lo = np.log1p(S[:, :n_obs] / lib * np.median(lib)); dO = Lo[J] - Lo[I]; dO = dO - dO.mean(0)
            if Hs.shape[1]:
                Lh = np.log1p(Hs / lib * np.median(lib)); dH = Lh[J] - Lh[I]; dH = dH - dH.mean(0)
                B = np.linalg.lstsq(dH, dO, rcond=None)[0]; G = 1 - ((dO - dH @ B) ** 2).sum() / (dO ** 2).sum()
            else:
                G = np.nan
            base = dict(topo=a.topo, world=a.world, q=a.q, lam=lam, seed=seed, G=G, coh_min=coh["dc1_min"], coh_max=coh["dc1_max"],
                        n_pairs=len(I), nohidden=a.nohidden)
            for arm in ARMS:
                for chan in ("U", "X"):
                    rows.append(dict(base, method=f"R_{chan}_{arm}", **evaluate(production_scalar(out, arm, chan), regs, gt, n_obs, cl)))
            for arm in ARMS:                                              # FULL k profile (mechanistic diagnostic; not the headline)
                for chan in ("U", "X"):
                    for k in range(KMAX + 1):
                        sk = np.minimum(pct_rows(out[(arm, chan, "dc1_min")][k]), pct_rows(out[(arm, chan, "dc1_max")][k]))
                        if np.isfinite(sk).any():
                            rows.append(dict(base, method=f"perk_{chan}_{arm}", k=k, **evaluate(sk, regs, gt, n_obs, cl)))
            for chan in ("U", "X"):                                      # k=0 (no conditioning) under the same profile machinery
                k0 = np.minimum(pct_rows(out[("TRUE", chan, "dc1_min")][0]), pct_rows(out[("TRUE", chan, "dc1_max")][0]))
                rows.append(dict(base, method=f"{chan}_k0", **evaluate(k0, regs, gt, n_obs, cl)))
                best = max(evaluate(np.maximum(out[("TRUE", chan, "dc1_min")][k], out[("TRUE", chan, "dc1_max")][k]), regs, gt, n_obs, cl)["AUPRC"]
                           for k in range(KMAX + 1))
                rows.append(dict(base, method=f"ORACLE_bestk_{chan}_DIAGNOSTIC", AUPRC=best))
            import gb_sota_run as SR                                  # GENIE3 / GRNBoost2 via the canonical harness core (public API regressor profiles)
            Lx = np.log1p(dense(A.layers["spliced"]).astype(float)); gn = [f"gene_{i}" for i in range(n_obs)]
            for lr in ("grnboost2", "genie3"):
                L = SR.arboreto_fit(Lx, gn, [gn[r] for r in regs], gn, lr, 0, CLIENT)
                M = np.full((len(regs), n_obs), 0.0); ri = {gn[r]: i for i, r in enumerate(regs)}
                for tf, tg, w in zip(L.TF, L.target, L.importance):
                    M[ri[tf], int(tg.split("_")[1])] = w
                for i, r in enumerate(regs): M[i, r] = np.nan
                rows.append(dict(base, method=lr.upper(), **evaluate(M, regs, gt, n_obs, cl)))
            Z = (X - X.mean(0)) / np.maximum(X.std(0), EPS); C = np.abs(Z.T @ Z / len(Z))[regs]
            for i, r in enumerate(regs): C[i, r] = np.nan
            rows.append(dict(base, method="Pearson_abs", **evaluate(C, regs, gt, n_obs, cl)))
            print(f"{tag}: G={G:.3f} coh={coh['dc1_min']:.3f}/{coh['dc1_max']:.3f} pairs={len(I)} {time.time()-t0:.0f}s", flush=True)
            pd.DataFrame(rows).to_csv(f"{a.out}/pilot_{a.topo}_{a.world}_q{a.q}{'_nohidden' if a.nohidden else ''}.tsv", sep="\t", index=False)
    assert len(truth_hash) == 1, "observed truth differs across lam/seeds"
    print("truth identical across all runs:", len(truth_hash) == 1)


if __name__ == "__main__":
    from distributed import LocalCluster, Client
    CLIENT = Client(LocalCluster(n_workers=8, threads_per_worker=1, memory_limit="0"))
    main()
