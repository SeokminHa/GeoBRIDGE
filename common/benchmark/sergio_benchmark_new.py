#!/usr/bin/env python3
"""
SERGIO-based benchmark for GeoBRIDGE.

Design (following Velorama paper evaluation):
  - 100 genes, 5 MRs, 15 intermediates, 80 leaves (3-layer cascade)
  - ~180 direct edges, indirect:direct ≈ 3:1 in reachability
  - ONE GRN (fixed per seed) shared across all 4 topologies
  - All-pairs evaluation: 100×99 = 9,900 pairs, baseline AUPRC ≈ 0.018
  - 4 topologies: linear, bifurcation, trifurcation, tree
  - SERGIO dynamics=True for spliced/unspliced layers
  - Then scVelo velocity → GeoBRIDGE (DXc, concordance) + baselines

Usage:
  python sergio_benchmark_new.py --seeds 2          # quick validation
  python sergio_benchmark_new.py --seeds 10         # full benchmark
  python sergio_benchmark_new.py --topos linear bifurcation --seeds 2
"""
import sys
sys.path.insert(0, __import__("os").environ["GEOBRIDGE_ROOT"])
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/SERGIO"))

import numpy as np
import pandas as pd
import scanpy as sc
import scvelo as scv
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import average_precision_score, roc_auc_score
import warnings, time, argparse
warnings.filterwarnings("ignore")

from geobridge_model.pairspace import build_forward_pairs
from geobridge_model.scoring import dual_project_pairspace, score_matrix
from geobridge_model.spectral_k import select_k_by_tail_exhaustion

# ───────────────── Configuration ─────────────────
N_GENES = 100
N_MR = 5             # master regulators (no incoming edges, SERGIO requirement)
N_EDGES_TARGET = 180  # baseline AUPRC ≈ 180/9900 ≈ 0.018
CELLS_PER_TYPE = 300
EPS = 1e-12
K_SWEEP = list(range(0, 31))

OUTBASE = Path((__import__("os").environ["GEOBRIDGE_ROOT"] + "/sergio_benchmark_v2"))

# Topology definitions (bifurcation matrices)
# bMat[i,j] > 0 means cell type i differentiates into type j
TOPOLOGIES = {
    "linear": {
        "n_bins": 3,
        "bmat": np.array([
            [0, 0.1, 0],
            [0, 0, 0.1],
            [0, 0, 0],
        ]),
    },
    "bifurcation": {
        "n_bins": 4,
        "bmat": np.array([
            [0, 0.1, 0, 0],
            [0, 0, 0.1, 0.1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]),
        # A -> B -> C, B -> D
    },
    "trifurcation": {
        "n_bins": 4,
        "bmat": np.array([
            [0, 0.1, 0.1, 0.1],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]),
    },
    "tree": {
        "n_bins": 6,
        "bmat": np.array([
            [0, 0.08, 0, 0, 0, 0],
            [0, 0, 0.09, 0, 0, 0],
            [0, 0, 0, 0.08, 0.08, 0.08],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ]),
    },
}

# SERGIO params (matching Velorama DS6)
SERGIO_PARAMS = dict(
    noise_params=0.3,
    noise_type="dpd",
    decays=0.8,
    sampling_state=1,
    noise_params_splice=0.1,
    splice_ratio=1.5,
    dt=0.01,
    dt_splice=0.01,
)


# ───────────────── GRN Generation ─────────────────

def generate_grn(seed):
    """Generate a cascade GRN (fixed per seed, shared across topologies).

    3-layer hierarchy for rich indirect regulation:
      Layer 0: MRs (5 genes) — no incoming edges
      Layer 1: intermediate regulators (~15 genes) — regulated by MRs
      Layer 2: leaf genes (~80 genes) — regulated by Layer 0/1

    Target: ~180 direct edges, indirect:direct ≈ 3:1 in reachability.
    After generation, we compute and report the actual ratio.

    Returns:
        mrs: sorted list of MR gene IDs
        edges: list of (target, source, K, hill) for SERGIO interaction file
        gt_edges: set of (source, target) for evaluation
    """
    rng = np.random.RandomState(seed)
    all_ids = list(range(N_GENES))
    rng.shuffle(all_ids)

    mrs = sorted(all_ids[:N_MR])
    n_intermediate = 15
    intermediates = sorted(all_ids[N_MR:N_MR + n_intermediate])
    leaves = sorted(all_ids[N_MR + n_intermediate:])

    edges = []
    gt_edges = set()

    def add_edge(src, tgt):
        K = rng.uniform(1.0, 5.0) * rng.choice([-1, 1])
        edges.append((tgt, src, K, 2.0))
        gt_edges.add((src, tgt))

    # ── Layer 0 → 1: Each intermediate gets 1-2 MR inputs ──
    for im in intermediates:
        n_inputs = rng.choice([1, 2], p=[0.5, 0.5])
        srcs = rng.choice(mrs, size=min(n_inputs, len(mrs)), replace=False)
        for s in srcs:
            add_edge(s, im)

    # ── Layer 0/1 → 2: Each leaf gets 1-3 inputs from MRs or intermediates ──
    upper = mrs + intermediates
    for lf in leaves:
        n_inputs = rng.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
        srcs = rng.choice(upper, size=min(n_inputs, len(upper)), replace=False)
        for s in srcs:
            add_edge(s, lf)

    # ── Cross-layer edges: intermediate → intermediate (cascade depth) ──
    n_cross = 10
    for _ in range(n_cross):
        s = intermediates[rng.randint(len(intermediates))]
        t = intermediates[rng.randint(len(intermediates))]
        if s != t and (s, t) not in gt_edges:
            add_edge(s, t)

    # ── Fill to target: extra edges from upper → leaves ──
    attempts = 0
    while len(gt_edges) < N_EDGES_TARGET and attempts < 5000:
        src = upper[rng.randint(len(upper))]
        tgt = leaves[rng.randint(len(leaves))]
        if (src, tgt) not in gt_edges:
            add_edge(src, tgt)
        attempts += 1

    # ── Report indirect:direct ratio ──
    reachable = _transitive_closure(gt_edges, N_GENES)
    n_direct = len(gt_edges)
    n_indirect = len(reachable - gt_edges)
    print("  GRN stats: %d direct edges, %d indirect pairs, ratio=%.1f:1" % (
        n_direct, n_indirect, n_indirect / max(n_direct, 1)), flush=True)

    return mrs, edges, gt_edges


def _transitive_closure(gt_edges, n_genes):
    """Compute all reachable (source, target) pairs via any directed path."""
    adj = defaultdict(set)
    for s, t in gt_edges:
        adj[s].add(t)
    reachable = set()
    for start in range(n_genes):
        visited = set()
        stack = [start]
        while stack:
            node = stack.pop()
            for nb in adj.get(node, []):
                if nb not in visited:
                    visited.add(nb)
                    reachable.add((start, nb))
                    stack.append(nb)
    return reachable


def write_sergio_inputs(outdir, mrs, edges, n_bins, seed):
    """Write SERGIO MR file and interaction file.

    SERGIO constraint: len(MRs) + len(unique_targets) == N_GENES.
    MRs go in MR file; everything else is a target in the interaction file.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.RandomState(seed + 5000)

    # ── MR file ──
    mr_lines = []
    for mr in mrs:
        base = rng.uniform(0.5, 4.0)
        rates = [max(0.01, base + rng.uniform(-1.5, 1.5)) for _ in range(n_bins)]
        mr_lines.append("{},{}".format(float(mr), ",".join(str(r) for r in rates)))

    with open(outdir / "input_MRs.txt", "w") as f:
        f.write("\n".join(mr_lines) + "\n")

    # ── Interaction file ──
    target_regs = defaultdict(list)
    for tgt, src, K, hill in edges:
        target_regs[tgt].append((src, K, hill))

    n_mrs = len(mrs)
    n_tgts = len(target_regs)
    assert n_mrs + n_tgts == N_GENES, \
        "SERGIO gene count: %d MRs + %d targets = %d != %d" % (n_mrs, n_tgts, n_mrs+n_tgts, N_GENES)

    int_lines = []
    for tgt in sorted(target_regs.keys()):
        reg_list = target_regs[tgt]
        n_r = len(reg_list)
        reg_ids = ",".join(str(float(r)) for r, _, _ in reg_list)
        Ks = ",".join(str(k) for _, k, _ in reg_list)
        hills = ",".join(str(h) for _, _, h in reg_list)
        int_lines.append("{},{},{},{},{}".format(float(tgt), float(n_r), reg_ids, Ks, hills))

    with open(outdir / "input_GRN.txt", "w") as f:
        f.write("\n".join(int_lines) + "\n")

    return outdir / "input_MRs.txt", outdir / "input_GRN.txt"


# ───────────────── SERGIO Simulation ─────────────────

def run_sergio_simulation(n_bins, bmat, mr_file, grn_file):
    """Run SERGIO dynamics, return (spliced, unspliced, cell_types)."""
    from SERGIO.sergio import sergio

    sim = sergio(
        number_genes=N_GENES,
        number_bins=n_bins,
        number_sc=CELLS_PER_TYPE,
        noise_params=SERGIO_PARAMS["noise_params"],
        noise_type=SERGIO_PARAMS["noise_type"],
        decays=SERGIO_PARAMS["decays"],
        sampling_state=SERGIO_PARAMS["sampling_state"],
        noise_params_splice=SERGIO_PARAMS["noise_params_splice"],
        splice_ratio=SERGIO_PARAMS["splice_ratio"],
        dt=SERGIO_PARAMS["dt"],
        dt_splice=SERGIO_PARAMS["dt_splice"],
        dynamics=True,
        bifurcation_matrix=bmat,
    )
    sim.build_graph(
        input_file_taregts=str(grn_file),
        input_file_regs=str(mr_file),
        shared_coop_state=2,
    )
    sim.simulate_dynamics()
    exprU, exprS = sim.getExpressions_dynamics()

    spliced_list, unspliced_list, cell_types = [], [], []
    for b in range(n_bins):
        n_cells = exprS[b].shape[1]
        spliced_list.append(exprS[b].T)
        unspliced_list.append(exprU[b].T)
        cell_types.extend([b] * n_cells)

    spliced = np.maximum(np.vstack(spliced_list), 0)
    unspliced = np.maximum(np.vstack(unspliced_list), 0)
    return spliced, unspliced, np.array(cell_types)


def build_adata(spliced, unspliced, cell_types):
    """Build AnnData with spliced/unspliced layers."""
    import anndata as ad
    n_cells, n_genes = spliced.shape
    adata = ad.AnnData(
        X=spliced + unspliced,
        layers={"spliced": spliced.copy(), "unspliced": unspliced.copy()},
    )
    adata.obs["cell_type"] = pd.Categorical(["type_%d" % ct for ct in cell_types])
    adata.var_names = pd.Index(["gene_%d" % i for i in range(n_genes)])
    adata.obs_names = pd.Index(["cell_%d" % i for i in range(n_cells)])
    return adata


# ───────────────── Evaluation ─────────────────

def compute_metrics(scores, gt_edges, n_genes):
    """All-pairs evaluation: AUPRC, sAUPRC, AUROC, EDP@50, EDP@100."""
    labels, pa, ps = [], [], []
    for i in range(n_genes):
        for j in range(n_genes):
            if i == j:
                continue
            labels.append(1 if (i, j) in gt_edges else 0)
            pa.append(abs(scores[i, j]))
            ps.append(scores[i, j])
    labels, pa, ps = np.array(labels), np.array(pa), np.array(ps)
    if labels.sum() == 0:
        return None
    order = np.argsort(-pa)
    nd = labels.sum()
    return {
        "AUPRC": average_precision_score(labels, pa),
        "sAUPRC": average_precision_score(labels, ps),
        "AUROC": roc_auc_score(labels, pa),
        "EDP@50": labels[order[:50]].sum() / min(50, nd) if nd > 0 else 0,
        "EDP@100": labels[order[:100]].sum() / min(100, nd) if nd > 0 else 0,
        "baseline_AUPRC": float(labels.mean()),
    }


def run_gb(adata, gt_edges):
    """Production GeoBRIDGE: DXc + concordance."""
    n_genes = adata.n_vars

    scv.pp.filter_and_normalize(adata, min_shared_counts=5)
    scv.pp.moments(adata, n_pcs=30, n_neighbors=30)
    scv.tl.velocity(adata, mode="deterministic")

    X = np.asarray(adata.layers["Ms"]).astype(np.float64)
    V = np.asarray(adata.layers["velocity"]).astype(np.float64)
    V = np.nan_to_num(V, nan=0.0)

    scv.tl.velocity_graph(adata)
    scv.tl.velocity_pseudotime(adata)
    pt = np.asarray(adata.obs["velocity_pseudotime"]).astype(np.float64)

    order = np.argsort(pt, kind="mergesort")
    X, V, pt = X[order], V[order], pt[order]

    try:
        pair_data = build_forward_pairs(X, V, pt, min_forward_dt=1e-4)
    except ValueError:
        pair_data = build_forward_pairs(X, V, pt, min_forward_dt=1e-6)

    DX, DV = pair_data.delta_x, pair_data.delta_v
    DXc = DX - DX.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(DXc, full_matrices=False)
    spec = select_k_by_tail_exhaustion(S)
    k_auto = spec.k_selected
    n_pairs = DX.shape[0]

    rows = []
    all_ks = sorted(set(K_SWEEP + [k_auto]))
    for k in all_ks:
        if k >= n_genes:
            continue
        DX_res, DV_res = dual_project_pairspace(DXc, DV, U, S, Vt, k)
        sc_mat = np.asarray(score_matrix(DX_res, DV_res, normalize=True,
                                         eps=EPS, method="concordance"))
        m = compute_metrics(sc_mat, gt_edges, n_genes)
        if m is None:
            continue
        m["k"] = k
        m["method"] = "GB_prod"
        m["is_auto_k"] = (k == k_auto)
        m["n_pairs"] = n_pairs
        rows.append(m)

    return rows


def run_baselines(adata, gt_edges):
    """Corr(X,V) and Random baselines."""
    n_genes = adata.n_vars
    X = np.asarray(adata.layers["Ms"]).astype(np.float64)
    V = np.asarray(adata.layers["velocity"]).astype(np.float64)
    V = np.nan_to_num(V, nan=0.0)
    rows = []

    # Corr(X,V)
    from scipy.stats import pearsonr
    corr_mat = np.zeros((n_genes, n_genes))
    for i in range(n_genes):
        for j in range(n_genes):
            if np.std(X[:, i]) > 0 and np.std(V[:, j]) > 0:
                corr_mat[i, j] = pearsonr(X[:, i], V[:, j])[0]
    m = compute_metrics(corr_mat, gt_edges, n_genes)
    if m is not None:
        m.update({"k": -1, "method": "Corr_XV", "is_auto_k": False, "n_pairs": 0})
        rows.append(m)

    # Random (10 reps)
    rng = np.random.RandomState(42)
    for _ in range(10):
        rand_mat = rng.randn(n_genes, n_genes)
        m = compute_metrics(rand_mat, gt_edges, n_genes)
        if m is not None:
            m.update({"k": -1, "method": "Random", "is_auto_k": False, "n_pairs": 0})
            rows.append(m)

    return rows


# ───────────────── Main Pipeline ─────────────────

def run_one(topo_name, seed, outdir, data_only=False):
    """Generate SERGIO data + run GB + baselines for one (topology, seed)."""
    t0 = time.time()
    topo_cfg = TOPOLOGIES[topo_name]
    n_bins = topo_cfg["n_bins"]
    bmat = topo_cfg["bmat"]
    sim_dir = outdir / topo_name / ("seed_%d" % seed)

    print("", flush=True)
    print("=" * 70, flush=True)
    print("  %s  seed=%d  (n_bins=%d)" % (topo_name, seed, n_bins), flush=True)
    print("=" * 70, flush=True)

    # ── 1. Generate GRN (same GRN per seed, shared across topologies) ──
    mrs, edges, gt_edges = generate_grn(seed)

    # Stats
    out_deg = defaultdict(int)
    for _, src, _, _ in edges:
        out_deg[src] += 1
    hub_genes = sorted(out_deg.keys(), key=lambda r: -out_deg[r])[:5]
    n_sources = len(out_deg)
    print("  GRN: %d MRs, %d unique sources, %d edges, %d unique directed" % (
        len(mrs), n_sources, len(edges), len(gt_edges)), flush=True)
    print("  Top hubs: %s" % (
        ", ".join("gene_%d(%d)" % (g, out_deg[g]) for g in hub_genes)), flush=True)

    # ── 2. Write SERGIO inputs (MR rates differ per topology due to n_bins) ──
    mr_file, grn_file = write_sergio_inputs(sim_dir, mrs, edges, n_bins, seed)

    # Save GT
    gt_df = pd.DataFrame(sorted(gt_edges), columns=["source", "target"])
    gt_df.to_csv(sim_dir / "gt_GRN.csv", index=False)
    # Save metadata
    meta = {
        "n_genes": N_GENES, "n_mr": N_MR,
        "n_edges": len(edges), "n_unique_edges": len(gt_edges),
        "master_regulators": mrs, "topology": topo_name,
        "seed": seed, "n_bins": n_bins,
    }
    import json
    with open(sim_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # ── 3. Run SERGIO ──
    print("  Running SERGIO dynamics simulation...", flush=True)
    t_sim = time.time()
    spliced, unspliced, cell_types = run_sergio_simulation(n_bins, bmat, mr_file, grn_file)
    print("  SERGIO: %d cells x %d genes (%.0fs)" % (
        spliced.shape[0], spliced.shape[1], time.time()-t_sim), flush=True)

    # ── 4. Save AnnData ──
    adata = build_adata(spliced, unspliced, cell_types)
    adata.write_h5ad(sim_dir / "adata.h5ad")

    if data_only:
        print("  Data saved (data-only mode, %.0fs)" % (time.time()-t0), flush=True)
        return []

    # ── 5. Run GeoBRIDGE ──
    print("  Running GeoBRIDGE (DXc, concordance)...", flush=True)
    t_gb = time.time()
    gb_rows = run_gb(adata, gt_edges)
    for r in gb_rows:
        r["topology"] = topo_name
        r["seed"] = seed
    if gb_rows:
        best = max(gb_rows, key=lambda r: r["AUPRC"])
        auto_rows = [r for r in gb_rows if r.get("is_auto_k")]
        auto_r = auto_rows[0] if auto_rows else best
        print("  GB best: k=%d AUPRC=%.4f EDP@50=%.2f | auto: k=%d AUPRC=%.4f EDP@50=%.2f (%.0fs)" % (
            best["k"], best["AUPRC"], best["EDP@50"],
            auto_r["k"], auto_r["AUPRC"], auto_r["EDP@50"],
            time.time()-t_gb), flush=True)

    # ── 6. Run baselines ──
    print("  Running baselines...", flush=True)
    bl_rows = run_baselines(adata, gt_edges)
    for r in bl_rows:
        r["topology"] = topo_name
        r["seed"] = seed
    corr_rows = [r for r in bl_rows if r["method"] == "Corr_XV"]
    rand_rows = [r for r in bl_rows if r["method"] == "Random"]
    if corr_rows:
        print("  Corr(X,V): AUPRC=%.4f EDP@50=%.2f" % (
            corr_rows[0]["AUPRC"], corr_rows[0]["EDP@50"]), flush=True)
    if rand_rows:
        print("  Random: AUPRC=%.4f EDP@50=%.2f" % (
            np.mean([r["AUPRC"] for r in rand_rows]),
            np.mean([r["EDP@50"] for r in rand_rows])), flush=True)

    print("  Total: %.0fs" % (time.time()-t0), flush=True)
    return gb_rows + bl_rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SERGIO benchmark for GeoBRIDGE")
    parser.add_argument("--topos", nargs="+", default=None)
    parser.add_argument("--seeds", type=int, default=2,
                        help="Number of seeds (0..seeds-1)")
    parser.add_argument("--seed-start", type=int, default=0,
                        help="Starting seed index (for splitting across servers)")
    parser.add_argument("--seed-end", type=int, default=None,
                        help="Ending seed index (exclusive). Overrides --seeds.")
    parser.add_argument("--data-only", action="store_true",
                        help="Only generate SERGIO data, skip GB/baseline eval")
    parser.add_argument("--outdir", type=str, default=None)
    args = parser.parse_args()

    topos = args.topos if args.topos else list(TOPOLOGIES.keys())
    seed_start = args.seed_start
    seed_end = args.seed_end if args.seed_end is not None else (seed_start + args.seeds)
    seed_list = list(range(seed_start, seed_end))
    outdir = Path(args.outdir) if args.outdir else OUTBASE
    outdir.mkdir(parents=True, exist_ok=True)

    print("SERGIO Benchmark: %s x seeds %s" % (topos, seed_list), flush=True)
    print("Output: %s" % outdir, flush=True)
    print("Genes: %d, MRs: %d, Edges: ~%d, Baseline AUPRC: ~%.3f" % (
        N_GENES, N_MR, N_EDGES_TARGET, N_EDGES_TARGET / (N_GENES * (N_GENES - 1))), flush=True)
    if args.data_only:
        print("MODE: data-only (skip evaluation)", flush=True)
    else:
        print("Score: CONCORDANCE (DXc, LOCKED formula)", flush=True)

    t0 = time.time()
    all_rows = []
    for seed in seed_list:
        for topo in topos:
            rows = run_one(topo, seed, outdir, data_only=args.data_only)
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    out_file = outdir / ("sergio_benchmark_results_s%d_s%d.tsv" % (seed_list[0], seed_list[-1]))
    df.to_csv(out_file, sep="\t", index=False)

    # ── Summary ──
    print("", flush=True)
    print("=" * 70, flush=True)
    print("SUMMARY — %s" % out_file, flush=True)
    print("=" * 70, flush=True)
    bl_auprc = df["baseline_AUPRC"].iloc[0] if len(df) > 0 else 0

    for topo in topos:
        sub = df[df["topology"] == topo]
        gb = sub[sub["method"] == "GB_prod"]
        if len(gb) == 0:
            continue
        best_per_seed = gb.groupby("seed")["AUPRC"].max()
        auto = gb[gb["is_auto_k"]]
        auto_per_seed = auto.groupby("seed")["AUPRC"].first()
        corr = sub[sub["method"] == "Corr_XV"]
        rand = sub[sub["method"] == "Random"]
        print("")
        print("  %s (baseline=%.4f):" % (topo, bl_auprc))
        print("    GB oracle-k:  %.4f +/- %.4f" % (best_per_seed.mean(), best_per_seed.std()))
        if len(auto_per_seed) > 0:
            print("    GB auto-k:    %.4f +/- %.4f" % (auto_per_seed.mean(), auto_per_seed.std()))
        if len(corr) > 0:
            print("    Corr(X,V):    %.4f +/- %.4f" % (corr["AUPRC"].mean(), corr["AUPRC"].std()))
        if len(rand) > 0:
            print("    Random:       %.4f +/- %.4f" % (rand["AUPRC"].mean(), rand["AUPRC"].std()))

    print("")
    print("Total time: %.0fs" % (time.time()-t0), flush=True)
