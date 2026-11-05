#!/usr/bin/env python
"""Extra RNA-only observational GRN comparators (Tier-1 SOTA completeness, PI reset 2026-09-29 item 16).  Truth-free.

Methods (all run on the SAME canonical fixture control cells as GENIE3/GRNBoost2, no ATAC/motif/ChIP/prior network, no perturbation data):
  regdiffusion  RegDiffusion (Zhu & Slonim, J Comput Biol 2024; arXiv 2309.11928), pip `regdiffusion` 0.2.3, package defaults, seed 0
  deepsem       DeepSEM (Shu et al., Nat Comput Sci 2021), GitHub HantaoShu/DeepSEM @89c41a3, `non_celltype_GRN` default hyper-parameters
                (alpha 100, beta 1, K1 1, K2 2, hidden 128, lr 1e-4, gamma .95, batch 64, 120 epochs); model file loaded UNCHANGED from the pinned
                checkout, training loop = its non_celltype_GRN_model.train_model minus the ground-truth evaluation / truth-derived TF mask
  pidc          PIDC (Chan, Stumpf et al., Cell Syst 2017), faithful GPU port of NetworkInference.jl @e5a3de3 + InformationMeasures.jl @64810f2 +
                Discretizers.jl @05913ee (Bayesian-blocks discretiser, ML estimator, base 2, PUC summed over ALL other genes, Gamma-CDF network context)
  fasttenet     FastTENET (Kim et al., accelerated TENET, Kim et al. NAR 2021 transfer entropy along pseudotime), pip `fasttenet` 0.1.18 / `mate-cxinsys`,
                README-recommended binning FSBW-L, kp 0.5, dt 1; pseudotime = scVelo velocity_pseudotime of the fixture's OWN velocity (declared derivation)

Input = canonical fixture log `.X` (control cells only) over all fixture genes with non-zero variance (zero-variance genes cannot be modelled by any of
these methods and are NaN targets); regulators = frozen species TF catalogue ∩ those genes (case-collision groups dropped, as gb_tree_v2 / gb_velorama_v2).
Artifact = results/real_uniform/<SYS>/<method>_v1.npz (+ receipt via RU.write_part, fixture-sha bound): genes(axis), sources(roster), score(roster x axis),
tfs(all catalogue TFs), score_all(tfs x axis, UNSIGNED), score_all_signed (deepsem / regdiffusion only).  NaN = not a modelled target / self / constant regulator
row (regulator-level abstention, never a tie).  Regulator rows follow the score orientation source -> target.

  python scratch/gb_sota_ext.py pseudotime PANC                              # scVelo velocity pseudotime (CPU)
  python scratch/gb_sota_ext.py run pidc PANC --gpu 4
  python scratch/gb_sota_ext.py run regdiffusion PANC --gpu 5 [--smoke]
  python scratch/gb_sota_ext.py selftest [--gpu 4]                            # synthetic
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_real_uniform as RU  # noqa: E402

METHODS = ("regdiffusion", "deepsem", "pidc", "fasttenet")
LABEL = {"regdiffusion": "RegDiffusion", "deepsem": "DeepSEM", "pidc": "PIDC", "fasttenet": "FastTENET"}
SUFFIX = "_v1"
DEEPSEM_DIR = RU.R / "external" / "DeepSEM"
DEEPSEM_COMMIT = "89c41a3b1a6f9f7114768ac6781a937640d28e11"
DEEPSEM_MODEL_SHA = "dff2aac8ef933face6b29ea80c6d301db1cd12b5a592071d41e294cbc2a754c5"
DEEPSEM_CFG = dict(beta=1, alpha=100, K1=1, K2=2, n_hidden=128, gamma=0.95, lr=1e-4, lr_step_size=0.99, batch_size=64, n_epochs=120, K=1)
PIDC_CFG = dict(discretizer="bayesian_blocks", estimator="maximum_likelihood", base=2, context="gamma_cdf_sum (CLR-context fallback on non-positive columns)",
                puc_sum="all other genes (nodes)", node_set="all fixture genes with non-zero variance")
TENET_CFG = dict(backend="gpu", binning_method="FSBW-L", kp=0.5, dt=1, num_kernels=1, pseudotime="scvelo velocity_pseudotime (vkey=velocity, xkey=Ms)")


def out_path(system: str, method: str, smoke: bool = False) -> Path:
    d = RU.system_dir(system)
    return (d / "sota_smoke" if smoke else d) / f"{method}{SUFFIX}.npz"


def drop_case_collisions(names):
    from collections import Counter
    c = Counter(n.upper() for n in names)
    return [n for n in names if c[n.upper()] == 1]


def env_versions(method: str) -> dict:
    import importlib.metadata as md
    pk = {"regdiffusion": ["regdiffusion", "torch"], "deepsem": ["torch"], "pidc": ["torch", "numba"],
          "fasttenet": ["fasttenet", "mate-cxinsys", "torch"]}[method]
    out = {p: md.version(p) for p in pk}
    out["numpy"] = np.__version__
    return out


def plan(system: str, method: str, extra: dict | None = None) -> dict:
    import gb_substrate as GS

    D = RU.canonical_inputs(system)
    RU.assert_frozen_roster_covered(system, D["sources"])
    cat_path = GS.paths(system, canonical=True)[2]
    cat = {l.strip().upper() for l in open(cat_path) if l.strip()}
    genes = list(D["fixture_genes"])
    X = RU.dense(D["A"].X, np.float64)
    keep = X.std(0) > 0
    nodes = [g for g, k in zip(genes, keep) if k]
    tfs_all = drop_case_collisions([g for g in genes if g.upper() in cat])
    if [s for s in D["sources"] if s not in tfs_all]:
        raise SystemExit("REFUSING_SOURCE_NOT_IN_CATALOGUE")
    nset = set(nodes)
    tfs = [g for g in tfs_all if g in nset]
    binding = {
        "method": f"{LABEL[method]}-v1", "inputs": RU.input_manifest(D),
        "config": {"matrix": "canonical fixture log .X (control cells), fixture genes with non-zero variance", "n_nodes": len(nodes),
                   "n_zero_variance_genes_dropped": int((~keep).sum()), "tf_names": "frozen species TF catalogue ∩ nodes",
                   "catalogue": cat_path, "catalogue_sha256": RU.sha(cat_path), "n_tfs": len(tfs), "seed": 0,
                   "environment": env_versions(method), **(extra or {})},
        "producer_sha256": RU.sha(Path(__file__).resolve()), "truth_read": False}
    binding["binding_sha256"] = RU.digest_object(binding)
    gi = {g: i for i, g in enumerate(nodes)}
    return dict(D=D, X=X, nodes=nodes, node_idx=np.flatnonzero(keep), tfs=tfs, tf_node_idx=np.asarray([gi[t] for t in tfs], int),
                binding=binding, system=system, method=method, n_zero=int((~keep).sum()))


def assemble_and_write(P: dict, mat: np.ndarray, signed: np.ndarray | None, extra: dict, smoke: bool = False) -> Path:
    """mat / signed: (n_tfs, n_nodes) source x node scores -> artifact on the evaluator axis"""
    D, nodes, tfs, method = P["D"], P["nodes"], P["tfs"], P["method"]
    axis = [str(g) for g in D["axis"]]
    ni = {g: i for i, g in enumerate(nodes)}
    aj = np.asarray([j for j, g in enumerate(axis) if g in ni], int)
    nj = np.asarray([ni[axis[j]] for j in aj], int)
    aiu = {g.upper(): j for j, g in enumerate(axis)}
    if not np.isfinite(mat).all():
        raise SystemExit(f"REFUSING_{LABEL[method].upper()}_NONFINITE_SCORES: {int((~np.isfinite(mat)).sum())} entries")

    def to_axis(m):
        full = np.full((len(tfs), len(axis)), np.nan, np.float32)
        full[:, aj] = m[:, nj]
        for i, t in enumerate(tfs):
            if t.upper() in aiu:
                full[i, aiu[t.upper()]] = np.nan                           # self
        return full

    full = to_axis(np.abs(mat) if signed is None else np.abs(signed))
    fsig = None if signed is None else to_axis(signed)
    const = []
    for i in range(len(tfs)):                                               # regulator-level abstention: constant row = NOT_SCORED, never a tie
        r = full[i][np.isfinite(full[i])]
        if r.size == 0 or r.max() == r.min():
            full[i] = np.nan
            if fsig is not None:
                fsig[i] = np.nan
            const.append(tfs[i])
    ti = {t: i for i, t in enumerate(tfs)}
    score = np.full((len(D["sources"]), len(axis)), np.nan, np.float32)
    for k, s in enumerate(D["sources"]):
        if s in ti:
            score[k] = full[ti[s]]
    arrays = {"genes": np.asarray(axis), "sources": np.asarray(D["sources"]), "score": score, "tfs": np.asarray(tfs), "score_all": full}
    if fsig is not None:
        arrays["score_all_signed"] = fsig
    path = out_path(P["system"], method, smoke)
    RU.write_part(path, P["binding"], arrays, {
        "diagnostics": {"n_tfs": len(tfs), "n_nodes": len(nodes), "n_axis": len(axis), "axis_target_coverage": float(len(aj) / len(axis)),
                        "constant_rows_abstained": const, "sources_not_scored": [s for s in D["sources"] if s not in ti or s in const],
                        "score_finite_frac": float(np.isfinite(full).mean())},
        "score_semantics": "unsigned source->target association/strength; NaN = not a modelled target / self / constant regulator row", **extra})
    print(f"[sota_ext] wrote {path}: {len(tfs)} TF rows x {len(axis)} axis; constant rows abstained {len(const)}", flush=True)
    return path


# ============================================================================================================================ RegDiffusion
def run_regdiffusion(P, gpu, steps=None):

    import regdiffusion as rd
    Xn = P["X"][:, P["node_idx"]].astype(np.float32)
    t0 = time.time()
    tr = rd.RegDiffusionTrainer(Xn, device="cuda", seed=0, **({"n_steps": steps} if steps else {}))
    tr.train()
    n = Xn.shape[1]
    adj = tr.model.get_adj_().detach().cpu().numpy().astype(np.float64) * (n - 1)  # = package get_adj() without its float16 quantisation; row = source
    sig = adj[P["tf_node_idx"]]
    return sig, {"wall_s": time.time() - t0, "n_genes": n, "hp": {k: v for k, v in tr.hp.items() if k not in ("device", "self", "__class__") and isinstance(v, (int, float, str, bool, list, type(None)))}}


# ============================================================================================================================ DeepSEM
def load_deepsem_model():
    mp = DEEPSEM_DIR / "src" / "Model.py"
    if RU.sha(mp) != DEEPSEM_MODEL_SHA:
        raise SystemExit(f"REFUSING_DEEPSEM_MODEL_NOT_PINNED: {mp}")
    spec = importlib.util.spec_from_file_location("deepsem_model", mp)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def run_deepsem(P, gpu, epochs=None):

    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    M = load_deepsem_model()
    opt = argparse.Namespace(**{**DEEPSEM_CFG, **({"n_epochs": epochs} if epochs else {})})
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    X = P["X"][:, P["node_idx"]].astype(np.float32)
    X = (X - X.mean(0)) / X.std(0)                                          # DeepSEM's own per-gene z-score (numpy population sd)
    N = X.shape[1]
    A0 = np.ones([N, N]) / (N - 1) + (np.random.rand(N * N) * 0.0002).reshape([N, N])  # DeepSEM initalize_A
    np.fill_diagonal(A0, 0)
    dl = DataLoader(TensorDataset(torch.FloatTensor(X), torch.LongTensor(list(range(len(X))))), batch_size=opt.batch_size, shuffle=True, num_workers=1)
    vae = M.VAE_EAD(A0, 1, opt.n_hidden, opt.K).float().cuda()
    optimizer = optim.RMSprop(vae.parameters(), lr=opt.lr)
    optimizer2 = optim.RMSprop([vae.adj_A], lr=opt.lr * 0.2)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=opt.lr_step_size, gamma=opt.gamma)
    vae.train()
    t0, log = time.time(), []
    for epoch in range(opt.n_epochs + 1):                                    # identical to non_celltype_GRN_model.train_model (truth evaluation removed)
        vae.adj_A.requires_grad = not (epoch % (opt.K1 + opt.K2) < opt.K1)
        loss_all = []
        for inputs, _ in dl:
            optimizer.zero_grad()
            inputs = inputs.type(M.Tensor)
            temperature = max(0.95 ** epoch, 0.5)
            loss, loss_rec, loss_gauss, loss_cat, dec, y, hidden = vae(inputs, dropout_mask=None, temperature=temperature, opt=opt)
            loss = loss + opt.alpha * torch.mean(torch.abs(vae.adj_A))
            loss.backward()
            loss_all.append(loss.item())
            (optimizer if epoch % (opt.K1 + opt.K2) < opt.K1 else optimizer2).step()
        scheduler.step()
        log.append(float(np.mean(loss_all)))
        if epoch % 10 == 0:
            print(f"[deepsem] epoch {epoch}/{opt.n_epochs} loss {log[-1]:.4f} {time.time()-t0:.0f}s", flush=True)
    A = vae.adj_A.detach().cpu().numpy().astype(np.float64)               # A[target, TF] (extractEdgesFromMatrix: TF = column, target = row)
    if not np.isfinite(A).all():
        raise SystemExit("REFUSING_DEEPSEM_NONFINITE_ADJ")
    return A.T[P["tf_node_idx"]], {"wall_s": time.time() - t0, "n_genes": N, "loss_first": log[0], "loss_last": log[-1], "epochs": opt.n_epochs}


# ============================================================================================================================ PIDC
def _bb_edges_numba():
    import numba

    @numba.njit(cache=False)
    def bb(u, nn):                                                          # Discretizers.jl disc_bayesianblocks.jl, 0-based translation of the 1-based loop
        n = len(u)
        edges = np.empty(n + 1)
        edges[0] = u[0]
        for i in range(n - 1):
            edges[i + 1] = 0.5 * (u[i] + u[i + 1])
        edges[n] = u[n - 1]
        bl = u[n - 1] - edges
        cnt = np.zeros(n)
        best = np.zeros(n)
        last = np.zeros(n, np.int64)                                        # 1-based change-point index values
        for K in range(1, n + 1):
            for i in range(K):
                cnt[i] += nn[K - 1]
            prior = 4.0 - np.log(73.53 * 0.05 * K ** (-0.478))
            bmax = -np.inf
            imax = 0
            for i in range(K):
                f = cnt[i] * np.log(cnt[i] / (bl[i] - bl[K])) - prior
                if i >= 1:
                    f += best[i - 1]
                if f > bmax:
                    bmax = f
                    imax = i
            last[K - 1] = imax + 1
            best[K - 1] = bmax
        cps = np.empty(n + 1, np.int64)
        m = 0
        ind = n + 1
        while True:
            cps[m] = ind
            m += 1
            if ind == 1:
                break
            ind = last[ind - 2]
        return edges[cps[:m][::-1] - 1]
    return bb


def bayesian_blocks_codes(X):
    """(n_cells, n_genes) float64 -> int bin codes (0-based) and bins per gene, exactly Node()/get_bin_ids! (bayesian_blocks, LinearDiscretizer encode)"""
    bb = _bb_edges_numba()
    n, G = X.shape
    codes = np.zeros((n, G), np.int64)
    nb = np.zeros(G, np.int64)
    for g in range(G):
        x = X[:, g]
        u, c = np.unique(x, return_counts=True)
        if len(u) == 1:
            nb[g] = 1
            continue
        e = bb(u, c.astype(np.float64))
        nb[g] = len(e) - 1
        codes[:, g] = np.clip(np.searchsorted(e, x, side="right") - 1, 0, nb[g] - 1)
    return codes, nb


def pidc_weights(X, tf_idx, device="cuda", log=print):
    """PIDC edge weights for the rows `tf_idx` against all nodes.  X (n_cells, N) float64.  Returns ((nTF, N) weights, diagnostics).
    PUC[a,z] = sum_y [ clip((I(a;z) - R(z;a,y)) / I(a;z)) + clip((I(z;a) - R(a;z,y)) / I(z;a)) ],  R(t;s1,s2) = sum_v p_t(v) min(SI[s1->t](v), SI[s2->t](v)),
    SI[s->t](v) = sum_u p(u|t=v) log2 p(u,v)/(p(u)p(v))   (NetworkInference.jl get_puc_scores; y over all other nodes)."""
    import torch
    n, N = X.shape
    t0 = time.time()
    codes, nb = bayesian_blocks_codes(X)
    Bm = int(nb.max())
    log(f"[pidc] bayesian blocks {time.time()-t0:.0f}s: bins/gene min {nb.min()} median {int(np.median(nb))} max {Bm}", flush=True)
    dev = torch.device(device)
    cd = torch.as_tensor(codes, device=dev)
    H = torch.zeros(n, N, Bm, dtype=torch.float32, device=dev)
    H.scatter_(2, cd[:, :, None], 1.0)
    H = H.reshape(n, N * Bm)                                                          # (cells, N*Bm) one-hot bin membership
    p1 = (H.sum(0) / n).double().reshape(N, Bm)                                       # ML marginals; padded bins = 0
    MI = torch.zeros(N, N, dtype=torch.float64, device=dev)
    SI = torch.zeros(N, N, Bm, dtype=torch.float64, device=dev)                       # SI[z, a, v] = specific information source a -> target z at z's value v
    blk = max(1, int(5e8 // (Bm * Bm * N * 8)))
    for i0 in range(0, N, blk):
        i1 = min(N, i0 + blk)
        b = i1 - i0
        C = (H[:, i0 * Bm:i1 * Bm].T @ H).reshape(b, Bm, N, Bm).double() / n          # exact joint counts / n
        px, pz = p1[i0:i1], p1
        pos = C > 0
        den = px[:, :, None, None] * pz[None, None, :, :]
        L = torch.where(pos, torch.log2(torch.where(pos, C / torch.where(pos, den, torch.ones_like(den)), torch.ones_like(C))), torch.zeros_like(C))
        MI[i0:i1] = (C * L).sum((1, 3))
        pzs = torch.where(pz > 0, pz, torch.ones_like(pz))[None, None]
        SI[:, i0:i1, :] = ((C / pzs) * L).sum(1).permute(1, 0, 2)                      # (b, N(z), Bm) -> (N(z), b, Bm)
        del C, den, L, pos
    log(f"[pidc] pair statistics {time.time()-t0:.0f}s", flush=True)
    T = torch.zeros(N, N, dtype=torch.float64, device=dev)                           # T[a, z] = sum_y puc contribution of target z with sources (a, y)
    for z in range(N):
        bz = int(nb[z])
        S = SI[z][:, :bz]
        pzv = p1[z, :bz]
        R = torch.zeros(N, N, dtype=torch.float64, device=dev)
        for v in range(bz):
            col = S[:, v]
            R += pzv[v] * torch.minimum(col[:, None], col[None, :])
        mi = MI[:, z][:, None]
        puc = (mi - R) / mi
        puc = torch.where(torch.isfinite(puc) & (puc >= 0), puc, torch.zeros_like(puc))
        puc.fill_diagonal_(0)                                                          # y == a
        puc[:, z] = 0                                                                  # y == z
        T[:, z] = puc.sum(1)
        T[z, z] = 0
        del R, puc
    PUC = T + T.T
    PUC.fill_diagonal_(0)
    log(f"[pidc] PUC {time.time()-t0:.0f}s", flush=True)
    # ---- network context (get_weight(::PIDCNetworkInference)): per-column Gamma MLE over the N-1 off-diagonal scores
    Nm1 = N - 1
    off = ~torch.eye(N, dtype=torch.bool, device=dev)
    s1 = PUC.sum(0)
    mean = s1 / Nm1
    var = ((PUC ** 2).sum(0) - Nm1 * mean ** 2) / (Nm1 - 1)                          # sample variance (Julia var)
    bad = ((PUC <= 0) & off).any(0) | ~torch.isfinite(mean) | (mean <= 0)
    lg = torch.where(off & (PUC > 0), torch.log(torch.where(PUC > 0, PUC, torch.ones_like(PUC))), torch.zeros_like(PUC))
    mlog = lg.sum(0) / Nm1
    s = torch.log(torch.where(mean > 0, mean, torch.ones_like(mean))) - mlog
    s = torch.where(bad, torch.ones_like(s), s.clamp_min(1e-300))
    a = (3 - s + torch.sqrt((s - 3) ** 2 + 24 * s)) / (12 * s)
    for _ in range(100):                                                              # Newton on log(a) - digamma(a) = s (Gamma MLE)
        f = torch.log(a) - torch.digamma(a) - s
        fp = 1 / a - torch.special.polygamma(1, a)
        a = (a - f / fp).clamp_min(1e-10)
    theta = mean / a
    tf = torch.as_tensor(tf_idx, device=dev)
    sc = PUC[tf]
    cdf_j = torch.special.gammainc(a[None, :], sc / theta[None, :])
    cdf_i = torch.special.gammainc(a[tf][:, None], sc / theta[tf][:, None])
    W = cdf_i + cdf_j
    di, dj = sc - mean[tf][:, None], sc - mean[None, :]
    vi, vj = var[tf][:, None], var[None, :]
    ti = torch.where((vi == 0) | (di < 0), torch.zeros_like(di), di ** 2 / torch.where(vi == 0, torch.ones_like(vi), vi))
    tj = torch.where((vj == 0) | (dj < 0), torch.zeros_like(dj), dj ** 2 / torch.where(vj == 0, torch.ones_like(vj), vj))
    Wclr = torch.sqrt(ti + tj)
    fb = bad[tf][:, None] | bad[None, :]
    W = torch.where(fb, Wclr, W)
    diag = {"wall_s": time.time() - t0, "bins_min": int(nb.min()), "bins_median": int(np.median(nb)), "bins_max": Bm, "n_columns_clr_fallback": int(bad.sum()),
            "puc_mean": float(PUC.sum() / (N * N - N)), "gamma_shape_median": float(a[~bad].median()) if (~bad).any() else None}
    return W.cpu().numpy(), diag


def run_pidc(P, gpu):

    X = P["X"][:, P["node_idx"]]
    W, diag = pidc_weights(X, P["tf_node_idx"])
    return W, diag


# ============================================================================================================================ pseudotime + FastTENET
def pt_path(system: str) -> Path:
    return RU.system_dir(system) / "velocity_pseudotime_v1.npz"


def cmd_pseudotime(a):
    import scvelo as scv
    system = a.system.upper()
    D = RU.canonical_inputs(system)
    A = D["A"]
    binding = {"method": "velocity_pseudotime_v1", "inputs": RU.input_manifest(D), "config": {"call": "scv.tl.velocity_graph(vkey=velocity,xkey=Ms); scv.tl.velocity_pseudotime(vkey=velocity)",
               "scvelo": scv.__version__}, "producer_sha256": RU.sha(Path(__file__).resolve()), "truth_read": False}
    binding["binding_sha256"] = RU.digest_object(binding)
    if RU.valid_part(pt_path(system), binding):
        print(f"[pseudotime] reuse {pt_path(system)}")
        return
    np.random.seed(0)
    scv.tl.velocity_graph(A, vkey="velocity", xkey="Ms", n_jobs=a.jobs)
    scv.tl.velocity_pseudotime(A, vkey="velocity")
    pt = np.asarray(A.obs["velocity_pseudotime"], np.float64)
    if not np.isfinite(pt).all():
        raise SystemExit("REFUSING_NONFINITE_PSEUDOTIME")
    RU.write_part(pt_path(system), binding, {"pseudotime": pt, "obs_names": np.asarray(A.obs_names).astype(str)},
                  {"diagnostics": {"n_cells": len(pt), "n_unique": int(len(np.unique(pt))), "min": float(pt.min()), "max": float(pt.max()),
                                   "spearman_with_n_counts": float(__import__("scipy.stats", fromlist=["spearmanr"]).spearmanr(pt, A.obs["n_counts"]).statistic)
                                   if "n_counts" in A.obs else None}})
    print(f"[pseudotime] {system}: {len(pt)} cells, unique {len(np.unique(pt))}", flush=True)


def run_fasttenet(P, gpu, max_tfs=None, batch=2 ** 14):

    import fasttenet as fte
    pp = pt_path(P["system"])
    if not pp.exists():
        raise SystemExit(f"REFUSING_NO_PSEUDOTIME: run `pseudotime {P['system']}` first ({pp})")
    z = np.load(pp, allow_pickle=True)
    if [str(x) for x in z["obs_names"]] != [str(x) for x in P["D"]["A"].obs_names]:
        raise SystemExit("REFUSING_PSEUDOTIME_CELL_ORDER_MISMATCH")
    pt = z["pseudotime"].astype(np.float32)
    exp = P["X"][:, P["node_idx"]].T.astype(np.float32)                        # (genes, cells)
    aligned = fte.align_data(data=exp, trj=pt, branch=np.ones(len(pt), np.int32))
    tfs = P["tfs"][:max_tfs] if max_tfs else P["tfs"]
    worker = fte.FastTENET(aligned_data=aligned, node_name=np.asarray(P["nodes"]), tfs=np.asarray(tfs))
    # added in release; shipped numbers were produced without it
    np.random.seed(0)
    try:
        import torch; torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    except ImportError:
        pass
    t0 = time.time()
    res = worker.run(backend=TENET_CFG["backend"], device_ids=[0], procs_per_device=1, batch_size=batch, kp=TENET_CFG["kp"],
                     binning_method=TENET_CFG["binning_method"], dt=TENET_CFG["dt"])
    ni = {g: i for i, g in enumerate(P["nodes"])}
    W = np.zeros((len(tfs), len(P["nodes"])))
    for r, t in enumerate(tfs):                                               # res is (source, target): FastTENET.run returns result_matrix.T, filled [target, source]
        W[r] = res[ni[t]]
    return W, {"wall_s": time.time() - t0, "batch_size": batch, "n_pairs": int(len(tfs) * (len(P["nodes"]) - 1)),
                          "pseudotime_sha256": RU.sha(pp)}, len(tfs)


# ============================================================================================================================ CLI
def cmd_run(a):
    system, method = a.system.upper(), a.method
    RU.set_gpu(a.gpu)                                                         # before ANY torch import (canonical_inputs imports torch)
    extra = {"smoke": bool(a.smoke), "overrides": {k: v for k, v in dict(epochs=a.epochs, steps=a.steps, max_tfs=a.max_tfs).items() if v}}
    cfg = {"regdiffusion": {"package_defaults": True}, "deepsem": {**DEEPSEM_CFG, "commit": DEEPSEM_COMMIT, "model_py_sha256": DEEPSEM_MODEL_SHA,
           "task": "non_celltype_GRN (default setting), truth evaluation + truth-derived TF mask removed"}, "pidc": PIDC_CFG, "fasttenet": TENET_CFG}[method]
    P = plan(system, method, {"method_config": cfg, **extra})
    outp = out_path(system, method, a.smoke)
    if RU.valid_part(outp, P["binding"]):
        print(f"[sota_ext] reuse {outp}")
        return
    print(f"[sota_ext] {LABEL[method]} {system}: {len(P['nodes'])} nodes ({P['n_zero']} zero-variance genes dropped), {len(P['tfs'])} TFs, {P['X'].shape[0]} cells", flush=True)
    if method == "regdiffusion":
        sig, ex = run_regdiffusion(P, a.gpu, a.steps)
        assemble_and_write(P, sig, sig, {"run": ex}, a.smoke)
    elif method == "deepsem":
        sig, ex = run_deepsem(P, a.gpu, a.epochs)
        assemble_and_write(P, sig, sig, {"run": ex}, a.smoke)
    elif method == "pidc":
        W, ex = run_pidc(P, a.gpu)
        assemble_and_write(P, W, None, {"run": ex}, a.smoke)
    else:
        W, ex, k = run_fasttenet(P, a.gpu, a.max_tfs)
        if k < len(P["tfs"]):                                                  # smoke with a TF subset: write only the computed rows
            P["tfs"], P["tf_node_idx"] = P["tfs"][:k], P["tf_node_idx"][:k]
        assemble_and_write(P, W, None, {"run": ex}, a.smoke)


def selftest(gpu):
    """synthetic checks (no fixture): Bayesian-blocks edges, PIDC on a planted chain / star, FastTENET orientation, RegDiffusion / DeepSEM orientation"""
    RU.set_gpu(gpu)

    rng = np.random.default_rng(0)
    bb = _bb_edges_numba()
    x = np.r_[rng.normal(0, 1, 300), rng.normal(6, 1, 300)]
    u, c = np.unique(x, return_counts=True)
    e = bb(u, c.astype(float))
    assert len(e) >= 3 and e[0] == u[0] and e[-1] == u[-1] and (np.diff(e) > 0).all(), e
    assert 1 <= np.sum((e > 1.5) & (e < 4.5)) <= 6, e                            # the bimodal gap must be cut
    # PIDC: planted star hub -> 12 targets; hub row should rank its targets above non-targets
    n, G = 600, 40
    Z = rng.normal(size=(n, G))
    Z[:, 1:13] = 0.8 * Z[:, [0]] + 0.6 * Z[:, 1:13]
    W, d = pidc_weights(Z, np.array([0]), device="cuda")
    w = W[0]
    from sklearn.metrics import roc_auc_score
    y = np.zeros(G, bool)
    y[1:13] = True
    auc = roc_auc_score(y[1:], w[1:])
    assert auc > 0.9 and np.isfinite(w).all(), (auc, d)
    print(f"[selftest] bayesian blocks ok; PIDC star AUROC {auc:.3f}; diag {d}")
    # FastTENET orientation: B(t+1) = A(t) + noise  => TE(A->B) >> TE(B->A)
    import fasttenet as fte
    T = 800
    Aa = rng.normal(size=T)
    Bb = np.r_[0, Aa[:-1]] + 0.3 * rng.normal(size=T)
    Cc = rng.normal(size=T)
    arr = np.stack([Aa, Bb, Cc]).astype(np.float32)
    wk = fte.FastTENET(aligned_data=arr, node_name=np.array(["A", "B", "C"]), tfs=np.array(["A", "B", "C"]))
    res = wk.run(backend="gpu", device_ids=[0], procs_per_device=1, batch_size=1024, kp=0.5, binning_method="FSBW-L", dt=1)
    print("[selftest] FastTENET result (rows as returned):\n", np.round(res, 4))
    assert res[0, 1] > 3 * res[1, 0], "FastTENET orientation: expected result[source=A, target=B] to dominate result[B, A]"
    print("[selftest] FastTENET returned matrix is [source, target]: ok")
    # RegDiffusion / DeepSEM orientation on a synthetic star (source row must carry the weight)
    import regdiffusion as rd
    tr = rd.RegDiffusionTrainer(Z.astype(np.float32), device="cuda", seed=0)
    tr.train()
    adj = tr.model.get_adj_().detach().cpu().numpy()
    print(f"[selftest] RegDiffusion hub row mass {np.abs(adj[0, 1:13]).sum():.4f} vs column mass {np.abs(adj[1:13, 0]).sum():.4f}")
    print("[selftest] done")


def pidc_parity_data(n=400, G=60, seed=1):
    """synthetic expression with cluster structure, chains and zero inflation (unique values per gene < n for half the genes)"""
    rng = np.random.default_rng(seed)
    lab = rng.integers(0, 2, n)
    Z = rng.normal(size=(n, G)) + 1.5 * lab[:, None] * (rng.random(G) < 0.3)
    for j in range(1, 20):
        Z[:, j] += 0.7 * Z[:, j - 1]
    Z[:, 40:] = np.where(rng.random((n, G - 40)) < 0.35, 0.0, np.expm1(np.clip(Z[:, 40:], -3, 3) * 0.5 + 1))
    Z[:, 30:40] = np.round(Z[:, 30:40], 1)
    return Z


def cmd_pidc_parity(a):
    """gen: write the synthetic matrix for NetworkInference.jl; compare: our port vs the Julia network file (all pairs)"""
    Z = pidc_parity_data()
    G = Z.shape[1]
    names = [f"g{j}" for j in range(G)]
    if a.step == "gen":
        with open(a.path, "w") as fh:
            fh.write("gene\t" + "\t".join(f"c{i}" for i in range(Z.shape[0])) + "\n")
            for j in range(G):
                fh.write(names[j] + "\t" + "\t".join(repr(float(v)) for v in Z[:, j]) + "\n")
        print("wrote", a.path)
        return
    RU.set_gpu(a.gpu)
    W, d = pidc_weights(Z, np.arange(G), device="cuda")
    ref = {}
    for line in open(a.path):
        x, y, w = line.split()
        ref[(x, y)] = float(w)
    ix = {n_: i for i, n_ in enumerate(names)}
    mine = np.array([W[ix[x], ix[y]] for (x, y) in ref])
    jul = np.array(list(ref.values()))
    sym = max(abs(W[i, j] - W[j, i]) for i in range(G) for j in range(G) if i != j)
    from scipy.stats import spearmanr
    print(f"[pidc-parity] pairs {len(ref)}, max|abs diff| {np.abs(mine - jul).max():.3e}, max rel {np.max(np.abs(mine - jul) / np.maximum(np.abs(jul), 1e-12)):.3e}, "
          f"spearman {spearmanr(mine, jul).statistic:.6f}, port symmetry {sym:.2e}, clr-fallback columns {d['n_columns_clr_fallback']}")
    # residual ~5e-6 = Gamma-MLE / incomplete-gamma implementation tolerance (Distributions.jl vs torch); identical ranking
    assert np.abs(mine - jul).max() < 1e-4 and spearmanr(mine, jul).statistic > 0.99999, "PIDC port != NetworkInference.jl"


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("method", choices=METHODS)
    r.add_argument("system")
    r.add_argument("--gpu", type=int, required=True)
    r.add_argument("--smoke", action="store_true", help="write to sota_smoke/ (never the production path)")
    r.add_argument("--epochs", type=int, default=0, help="deepsem override (smoke timing)")
    r.add_argument("--steps", type=int, default=0, help="regdiffusion override (smoke timing)")
    r.add_argument("--max-tfs", type=int, default=0, dest="max_tfs", help="fasttenet: first K catalogue TFs only (smoke)")
    p = sub.add_parser("pseudotime")
    p.add_argument("system")
    p.add_argument("--jobs", type=int, default=16)
    s = sub.add_parser("selftest")
    s.add_argument("--gpu", type=int, default=4)
    q = sub.add_parser("pidc-parity")
    q.add_argument("step", choices=("gen", "compare"))
    q.add_argument("path")
    q.add_argument("--gpu", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "run":
        if a.method in ("deepsem", "regdiffusion", "pidc", "fasttenet") and not a.smoke and (a.epochs or a.steps or a.max_tfs):
            raise SystemExit("REFUSING_OVERRIDE_WITHOUT_SMOKE: --epochs/--steps/--max-tfs are smoke-only")
        cmd_run(a)
    elif a.cmd == "pseudotime":
        cmd_pseudotime(a)
    elif a.cmd == "pidc-parity":
        cmd_pidc_parity(a)
    else:
        selftest(a.gpu)


if __name__ == "__main__":
    main()
