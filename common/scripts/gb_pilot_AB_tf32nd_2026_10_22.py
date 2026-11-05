#!/usr/bin/env python
"""SERGIO SHARED-GEOMETRY benchmark v5 — continuous within-bin hidden programmes plus the exact pilot World P (PI decisions
§191 and 2026-09-25; implementation briefs SOL_IMPL_SERGIO_V5_CONTINUOUS_DRIVERS and SOL_IMPL_SERGIO_WORLD_P). v4 is retained unchanged.
DESCRIPTIVE / PRELIMINARY; code + fixtures only until internal review PASS (no pilot before PASS).

usage: gb_sergio_drift_v5.py selftest                               fast algebraic fixtures (no SERGIO import)
       gb_sergio_drift_v5.py fixtures <out> core                    simulation fixtures (server only)
       gb_sergio_drift_v5.py feasibility <out> --topo bifurcation --worlds A,B,C,D --q 3 --lams .5,2
       gb_sergio_drift_v5.py tune <out> --topo T --world A|B|C|D|P --q 3 --obs UMI_DEPTH --depth 10
       gb_sergio_drift_v5.py run <out> --topo T --world A|B|C|D|P --q 3 --gtargets 0,.15,.30 --obs UMI_DEPTH --depth 10
       gb_sergio_drift_v5.py run <out> --topo T --world P --q 3 --lams 0,1,3,5 --obs UMI_DEPTH --depth 10
       gb_sergio_drift_v5.py smoke <out> --world A|B|C|D|P --tuning-json <json> [--obs UMI_DEPTH --depth 10]
       gb_sergio_drift_v5.py verify <dataset_dir>                    geometry-only recomputation from raw/geometry_raw.npz
       gb_sergio_drift_v5.py regvelo <dataset_dir> | velorama_to_npz <dataset_dir> <tsv> | gpu_eval <dataset_dir>
       gb_sergio_drift_v5.py report <out> --required-grid <json> [--univ PI]

SEEDS: GRN seed = independent inferential unit (DEV = 0..9, CONFIRMATION = 100..109, refused without --confirm);
simulation seed = technical replicate (SERGIO np.random.seed(1000*grn + sim)); conditioning-draw seed = control distribution
(RIS/ROS draws 0..9 in the pilot).
GRN GENERATOR (matched structure): 100 genes; 5 MRs, 15 intermediates, 80 leaves; EXACT layer-edge counts per GRN: MR→int 22
(int in-degree multiset 8×1 + 7×2), int→int 10 (lower→higher index only: a DAG), MR→leaf 41, int→leaf 120 (leaf in-degree
multiset 19×1 + 41×2 + 20×3); every MR and intermediate has out-degree ≥1 ⇒ exactly 20 true regulators; N_DIRECT = 193.
Candidates = 20 regulators + 19 fixed decoy leaves (39); roster = candidates × all non-self genes (edges into regulator genes
KEPT, PI) = 3,861 edges, prevalence 193/3861 = .04999 IDENTICAL for every GRN. A spec-v1 variant (targets that are candidates
excluded, internal review finding 1) is reported as a second universe on the same scores.
SHARED PROGRAMME: A/B use q hidden SERGIO master regulators (removed from output and truth). C uses q observed master regulators
(retained in the output/candidate axes) and declares their L edges direct truth. D is generator-identical to A, then thins a fixed
3-MR + 3-intermediate source subset in both count layers at render time (p=.03). All profiles are smooth in continuous pseudotime;
programme edges K = λ·L, L fixed per GRN (rng 777+grn; |L| = 2 per edge; λ separately tuned). P is the exact original pilot
programme: native per-bin rates ``profiles(topo,q)``, independently overlapping 40%-of-leaf recipient draws per driver, and
L = U(1,3)·±1 from the same rng 777+grn stream. P is non-regulatory: hidden genes/edges are excluded from output and truth.
G_E (generator-grounded, G_E(λ=0)=0): decompose SERGIO's unspliced production into observed-GRN and hidden-driver components on
the λ-run's frozen forward pairs, then G_E = ||ΔP_H||²/(||ΔP_H||²+||ΔP_GRN||²). λ is tuned per (topology, q, world, GRN) on sim
seed 0 to hit targets in both roots; achieved G_E, the normalized cross-term, rank and alignment are verified per dataset.
PREPROCESSING / PAIRS / SCORING / ARMS / ENDPOINTS: as v2 (gb_canonical.build step for step; canonical_pairs; project_scores;
spec-v1 production scalar; TRUE, REV, RIS0..9, ROS0..9) + EDP@50 = direct/(direct+indirect) among the top 50 (baseline = the same
ratio over the roster) + D-vs-I AP with its baseline.
FIXTURES: candidate-universe identity; direct-edge count identity; prevalence identity across GRNs; GRN identity within paired
conditions; GRN difference across seeds; preprocessing parity (gb_canonical.build on an np.rint copy); G(λ=0)=0; q/strength
separation; direct/indirect truth correctness (BFS closure vs boolean matrix power); score persistence (reload raw → recompute
== stored); spec-v1 scalar == independent reference (1e-12); all arms equal at k=0; TRUE/REV/RIS equal at k=30; ROS ⟂ TRUE."""
import argparse, hashlib, json, os, sys, time
AB_GRN = int(os.environ.get("AB_GRN", 101))
import numpy as np, pandas as pd
from collections import defaultdict
R = __import__("os").environ["GEOBRIDGE_ROOT"]
for p_ in (R, f"{R}/scratch", f"{R}/pairspace", f"{R}/benchmark", f"{R}/SERGIO"):
    sys.path.insert(0, p_)

N_GENES, N_MR, N_INT = 100, 5, 27; N_DECOY = 0; TF40_LAYERS = [5, 7, 7, 7, 6]; TF40_PREV = 0.075
TOPO = os.environ.get('AB_TOPO', 'bifurcation')
INT_INDEG = [1] * 8 + [2] * 7; N_INT_INT = 10; N_MR_LEAF, N_INT_LEAF = 41, 120; LEAF_INDEG = [1] * 19 + [2] * 41 + [3] * 20
DEV_GRNS, CONF_GRNS = set(range(10)), set(range(100, 110))
KA = list(range(8, 31)); KMAX = 30; EPS = 1e-12; BAND = list(range(1, 8)); N_DRAW = 10
BASE_ARMS = ["TRUE", "REV"] + [f"RIS{s}" for s in range(N_DRAW)] + [f"ROS{s}" for s in range(N_DRAW)]
ARMS = BASE_ARMS + ["EQC_hi", "EQC_lo"]
CHANS = ("U", "X", "V"); ROOTS = ("dc1_min", "dc1_max"); VEL_NONFINITE_MAX = 0.0; CELLS_PER_TYPE = 300; S_REF = CELLS_PER_TYPE
TOPOLOGIES = {  # = benchmark/sergio_benchmark_new.TOPOLOGIES (Velorama-style differentiation graphs)
    "linear": (3, [[0, .1, 0], [0, 0, .1], [0, 0, 0]]),
    "bifurcation": (4, [[0, .1, 0, 0], [0, 0, .1, .1], [0, 0, 0, 0], [0, 0, 0, 0]]),
    "trifurcation": (4, [[0, .1, .1, .1], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]),
    "tree": (6, [[0, .08, 0, 0, 0, 0], [0, 0, .09, 0, 0, 0], [0, 0, 0, .08, .08, .08], [0] * 6, [0] * 6, [0] * 6])}
SERGIO_PARAMS = dict(noise_params=0.3, noise_type="dpd", decays=0.8, sampling_state=1, noise_params_splice=0.1, splice_ratio=1.5,
                     dt=0.01, dt_splice=0.01)
DEPTH = {"linear": [0, 1, 2], "bifurcation": [0, 1, 2, 2], "trifurcation": [0, 1, 1, 1], "tree": [0, 1, 2, 3, 3, 3]}
BRANCH = {"linear": [0, 0, 0], "bifurcation": [0, 0, 1, -1], "trifurcation": [0, 1, -1, 0], "tree": [0, 0, 0, 1, -1, 0]}
EDGE_DTYPE = np.dtype([("target", "<i8"), ("regulator", "<i8"), ("K", "<f8"), ("coop", "<f8"), ("half_response", "<f8")])
GRID_KEYS = ("topo", "world", "q", "G_target", "grn_seed", "sim_seed")
WORLDS = ("A", "B", "C", "D", "P", "P2", "H", "AB"); PLIKE = ("P", "P2", "H", "AB"); WORLD_D_P = .03
# PILOT 2026-10-21 (shared-H): world "H" = World-P machinery with q=1 hidden driver H(bin)=profiles(topo,1)[0] (1-D, monotone in depth) whose
# loadings c (dense, all 100 genes, rng 777+grn) enter (i) non-MR genes as hidden-gene hill edges K=lam*c (existing hidden hook) and
# (ii) MRs as an exogenous ADDITIVE term lam*c*Hn(bin) in their per-bin basal rate (MRs cannot carry incoming edges). Truth W untouched.
PILOT = dict(mr_dev_scale=1.0, ab=False, amp=(0.0, 0.0), amp_mod=(0.0, 0.0), mr_off=0.0, frozen_half=None, mr_inj=None, clip_log=[])   # C2 sets 0.2: shrinks the MR-specific per-bin rate deviations (same rng draws, scaled)
P2_N_INT = 6  # World-P2: each hidden driver also drives this many of the 15 intermediates (rng 7777+grn), on top of the unchanged World-P leaf loadings
OBSERVATIONS = ("UMI", "UMI_DEPTH", "CONC")


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha_file(p):
    return sha_bytes(open(p, "rb").read())


def maxdiff_nan(A, B):
    A, B = np.asarray(A), np.asarray(B)
    if A.shape != B.shape or not np.array_equal(np.isnan(A), np.isnan(B)):
        return float("inf")
    finite = np.isfinite(A) & np.isfinite(B)
    return float(np.max(np.abs(A[finite] - B[finite]))) if finite.any() else 0.0


# ───────────────────────── GRN generator (matched structure) ─────────────────────────
def generate_grn(g):
    """TF32ND GRN (2026-10-22, TF32ND_SPEC.md): 100 genes; 32 candidate TFs = 32 active regulators (5 MR + 27 intermediates in layers 7,7,7,6) + 0 decoy TFs
    (no out-edges, random among the 68 non-regulator genes). Edges regulator -> any strictly-higher-layer gene (DAG; SERGIO needs MRs without inputs, so only
    the 5 MRs receive no regulation). Heterogeneous source degrees: per-regulator lognormal weight; density by Bernoulli with scale calibrated to expected
    prevalence TF40_PREV of the 32x99 universe (edge count is RANDOM). Cascades: each intermediate has a parent in the previous layer."""
    rng = np.random.RandomState(g); ids = rng.permutation(N_GENES); layer, groups, pos = {}, [], 0
    for li, sz in enumerate(TF40_LAYERS):
        grp = sorted(int(x) for x in ids[pos:pos + sz]); groups.append(grp); pos += sz
        for v in grp: layer[v] = li
    rest = sorted(int(x) for x in ids[pos:]); top = len(TF40_LAYERS)
    for v in rest: layer[v] = top
    decoys = sorted(int(x) for x in rng.choice(rest, N_DECOY, replace=False)); mrs = groups[0]; ints = sorted(v for gr in groups[1:] for v in gr)
    regs = sorted(mrs + ints); w = {r: float(np.exp(rng.normal(0, 1.0))) for r in regs}; E = set()
    pw = lambda grp: np.array([w[x] for x in grp]) / sum(w[x] for x in grp)
    for li in range(1, len(groups)):
        for v in groups[li]: E.add((int(rng.choice(groups[li - 1], p=pw(groups[li - 1]))), v))
    for r in regs:
        if not any(s_ == r for s_, _ in E):
            E.add((r, int(rng.choice([t for t in range(N_GENES) if layer[t] > layer[r]]))))
    for v in range(N_GENES):
        if layer[v] > 0 and not any(t == v for _, t in E):
            E.add((int(rng.choice([r for r in regs if layer[r] < layer[v]], p=pw([r for r in regs if layer[r] < layer[v]]))), v))
    cand = [(s_, t) for s_ in regs for t in range(N_GENES) if layer[t] > layer[s_] and (s_, t) not in E]
    need = TF40_PREV * len(regs + decoys) * (N_GENES - 1) - len(E); ws = np.array([w[s_] for s_, _ in cand])
    lo, hi = 0.0, 1e3
    for _ in range(60):
        c = (lo + hi) / 2; lo, hi = (c, hi) if np.minimum(1.0, c * ws).sum() < need else (lo, c)
    for k in np.nonzero(rng.uniform(size=len(cand)) < np.minimum(1.0, c * ws))[0]: E.add(cand[k])
    assert not any(s_ == t for s_, t in E) and {s_ for s_, _ in E} == set(regs) and not (set(decoys) & set(regs))
    K = {e: rng.uniform(1.0, 5.0) * rng.choice([-1, 1]) for e in sorted(E)}
    return dict(mrs=mrs, ints=ints, leaves=sorted(set(rest)), gt=set(E), K=K, regs=regs, cands=sorted(regs + decoys), decoys=decoys, layer=layer)


def closure_bfs(gt, n):
    adj = defaultdict(set)
    for s, t in gt:
        adj[s].add(t)
    out = set()
    for a in range(n):
        seen, st = set(), [a]
        while st:
            for b in adj[st.pop()]:
                if b not in seen:
                    seen.add(b); st.append(b); out.add((a, b))
    return out


def closure_matrix(gt, n):
    A = np.zeros((n, n), bool)
    for s, t in gt:
        A[s, t] = True
    Rm = A.copy(); P = A.copy()
    for _ in range(n):
        P = (P.astype(int) @ A.astype(int)) > 0
        if not P.any():
            break
        Rm |= P
    return {(int(a), int(b)) for a, b in zip(*np.nonzero(Rm))}


HIDDEN_BASE, HIDDEN_AMP, HIDDEN_ABS_L = 2.0, 1.8, 2.0
TRUNK_DEGREES, BRANCH_DEGREES = tuple(range(1, 6)), tuple(range(1, 4))


def ab_profiles(topo):
    """2026-10-21 A+B pilot hidden-gene per-bin SERGIO rates. H_A: shared temporal f(depth), rate 0.3+3.5*d (monotone, same as shared-H).
    H_B: branch-dependent: rate 0.3+3.5*(0.5+0.5*b*g), g=0 on the trunk (and at the branch point), 1 on the branches => effective signed
    driver b*g in [-1,1] after centring at the midpoint rate; within-bin continuity comes from the SERGIO ODE relaxation (decay 0.8)."""
    assert topo in ("bifurcation", "tree")
    d = np.asarray(DEPTH[topo], float); d = d / max(d.max(), 1); b = np.asarray(BRANCH[topo], float); g = (b != 0).astype(float)
    return np.array([0.3 + 3.5 * d, 0.3 + 3.5 * (0.5 + 0.5 * b * g)])


def ab_hn(topo):
    """normalised centred drivers used for MR injection: HnA in [0,1] (=(rate-0.3)/3.5), HnB = b*g in [-1,1]"""
    r = ab_profiles(topo); return (r[0] - 0.3) / 3.5, 2 * (r[1] - 0.3) / 3.5 - 1


def ab_loadings(grn_seed):
    """dense heterogeneous mixed-sign loadings, identical distribution U(1,3)*(+-1) for ALL 100 genes (decoys/targets/regulators alike)"""
    cA = H_loadings(grn_seed); rng = np.random.RandomState(31337 + grn_seed)
    cB = np.array([rng.uniform(1.0, 3.0) * rng.choice([-1, 1]) for _ in range(N_GENES)])
    return cA, cB


def profiles(topo, q):
    if PILOT.get("ab"):
        assert q == 2
        return ab_profiles(topo)
    """Exact per-bin pilot programmes (scratch/gb_sergio_drift.py:30-39). Values are native SERGIO basal rates."""
    d = np.asarray(DEPTH[topo], float); d = d / max(d.max(), 1); b = np.asarray(BRANCH[topo], float)
    base = [d, np.exp(-((d - .5) ** 2) / .08), .5 + .5 * b, 1 - d]
    out = []
    for m in range(q):
        p = base[m % 4]
        p = (p - p.min()) / max(np.ptp(p), 1e-9)
        out.append(0.3 + 3.5 * (np.roll(p, m // 4) if m >= 4 else p))
    return np.array(out)


def topology_spec(topo):
    """Parent/child structure and the declared low-frequency basis dimension."""
    nb, mat = TOPOLOGIES[topo]; A = np.asarray(mat, float); parents = np.full(nb, -1, int)
    children = {b: list(np.flatnonzero(A[b] > 0).astype(int)) for b in range(nb)}
    for p, cs in children.items():
        for c in cs:
            if parents[c] >= 0:
                raise ValueError(f"{topo}: bin {c} has multiple parents")
            parents[c] = p
    roots = np.flatnonzero(parents < 0)
    if len(roots) != 1:
        raise ValueError(f"{topo}: expected one root, got {roots.tolist()}")
    descendants = {}
    for b in range(nb):
        seen, stack = set(), [b]
        while stack:
            x = stack.pop(); seen.add(x); stack.extend(children[x])
        descendants[b] = seen
    branch_nodes = [b for b in range(nb) if len(children[b]) > 1]
    dim = len(TRUNK_DEGREES) + sum((len(children[b]) - 1) * len(BRANCH_DEGREES) for b in branch_nodes)
    return dict(parents=parents, children=children, descendants=descendants, branch_nodes=branch_nodes,
                max_tau=float(max(DEPTH[topo]) + 1), dim=dim)


def helmert(n):
    """n × (n-1) orthonormal zero-sum Helmert contrasts."""
    H = np.zeros((n, max(n - 1, 0)))
    for j in range(n - 1):
        H[:j + 1, j] = 1.0 / np.sqrt((j + 1) * (j + 2)); H[j + 1, j] = -(j + 1) / np.sqrt((j + 1) * (j + 2))
    return H


def trajectory_basis(topo, bins, tau):
    """Fixed continuous low-frequency basis: Legendre degrees 1..5 plus branch Helmert contrasts times (tau-tau_bp)^1..3."""
    bins, tau = np.asarray(bins, int), np.asarray(tau, float); sp = topology_spec(topo)
    z = 2.0 * tau / sp["max_tau"] - 1.0
    cols, names = [], []
    for degree in TRUNK_DEGREES:
        coef = np.zeros(degree + 1); coef[-1] = 1.0
        cols.append(np.polynomial.legendre.legval(z, coef)); names.append(f"trunk_legendre_{degree}")
    for bp in sp["branch_nodes"]:
        cs = sp["children"][bp]; H = helmert(len(cs)); tau_bp = DEPTH[topo][bp] + 1.0
        for h in range(H.shape[1]):
            lineage = np.zeros(len(bins))
            for ci, child in enumerate(cs):
                lineage[np.isin(bins, list(sp["descendants"][child]))] = H[ci, h]
            for degree in BRANCH_DEGREES:
                cols.append(lineage * np.maximum(tau - tau_bp, 0.0) ** degree)
                names.append(f"branch{bp}_helmert{h + 1}_degree{degree}")
    return np.column_stack(cols), names


def _orth(A, tol=1e-10):
    if A.size == 0:
        return np.empty((A.shape[0], 0))
    U, s, _ = np.linalg.svd(A, full_matrices=False); keep = s > tol * max(s[0], 1.0)
    return U[:, keep]


def effective_rank(A):
    s = np.linalg.svd(np.asarray(A, float), compute_uv=False)
    return float((s ** 2).sum() ** 2 / (s ** 4).sum()) if np.any(s > 0) else 0.0


def orient_profiles(Bcell, pair_blocks, signal_blocks, q, world):
    """Select the extremal generalized-eigen subspace, then rotate within it so profiles are orthogonal over cells AND pair-space.
    A common final scale makes pair energy equal and max|profile|=1 without changing either orthogonality."""
    Bc = np.asarray(Bcell, float) - np.asarray(Bcell, float).mean(0, keepdims=True)
    C = Bc.T @ Bc; dB = np.vstack(pair_blocks); Ug = _orth(np.vstack(signal_blocks))
    M = dB.T @ dB; Z = Ug.T @ dB; N = Z.T @ Z
    em, Vm = np.linalg.eigh(M); keep = em > 1e-10 * max(em.max(), 1.0)
    if keep.sum() < q:
        return None, dict(status="NO_CALL_RANK", numerical_pair_rank=int(keep.sum()))
    W = Vm[:, keep] / np.sqrt(em[keep])[None, :]
    er, Vr = np.linalg.eigh(W.T @ N @ W); order = np.argsort(er) if world == "A" else np.argsort(er)[::-1]
    A = W @ Vr[:, order[:q]]                                                    # A' M A = I
    ec, Rc = np.linalg.eigh(A.T @ C @ A); A = A @ Rc                            # A' C A diagonal, A' M A still I
    F = Bc @ A; scale = max(float(np.abs(F).max()), EPS); A /= scale; F /= scale
    dF = dB @ A; pair_gram = dF.T @ dF; cell_gram = F.T @ F
    cos = np.linalg.svd(_orth(dF).T @ Ug, compute_uv=False)
    status = "OK"
    if world == "A" and (len(cos) == 0 or cos.max() > .30): status = "NO_CALL_ALIGNMENT"
    if world == "B" and (len(cos) < q or cos.min() < .80): status = "NO_CALL_ALIGNMENT"
    info = dict(status=status, generalized_eigenvalues=er[order[:q]].tolist(), construction_cosines=cos.tolist(),
                construction_cos_max=float(cos.max()) if len(cos) else np.nan,
                construction_cos_min=float(cos.min()) if len(cos) else np.nan,
                cell_rank=effective_rank(F), pair_rank=effective_rank(dF), max_abs_profile=float(np.abs(F).max()),
                cell_orth_max=float(np.abs(cell_gram - np.diag(np.diag(cell_gram))).max()),
                pair_orth_max=float(np.abs(pair_gram - np.diag(np.diag(pair_gram))).max()),
                pair_energy=np.diag(pair_gram).tolist())
    return np.vstack([-(np.asarray(Bcell, float).mean(0) @ A)[None, :], A]), info


def continuous_sergio_class(base_cls=None):
    """Create the SERGIO subclass lazily, so `selftest` needs neither SERGIO nor its dependencies."""
    if base_cls is None:
        from SERGIO.sergio import sergio as base_cls
    class ContinuousSergio(base_cls):
        def configure_hidden_profiles(self, hidden_ids, topo, coefficients, base=HIDDEN_BASE, amp=HIDDEN_AMP, s_ref=S_REF):
            self.gb_hidden_ids_ = {int(g): i for i, g in enumerate(hidden_ids)}; self.gb_topo_ = topo
            self.gb_coefficients_ = np.asarray(coefficients, float); self.gb_base_, self.gb_amp_, self.gb_s_ref_ = float(base), float(amp), int(s_ref)
            nb = TOPOLOGIES[topo][0]
            Bend, _ = trajectory_basis(topo, np.arange(nb), np.asarray(DEPTH[topo], float) + 1.0)
            end_profiles = self.gb_coefficients_[0] + Bend @ self.gb_coefficients_[1:]
            rates = self.gb_base_ + self.gb_amp_ * end_profiles
            if not np.isfinite(rates).all() or np.any(rates <= 0):
                raise ValueError("continuous hidden rates must be finite and strictly positive (no clipping permitted)")
            for g, h in self.gb_hidden_ids_.items():
                self.graph_[g]["rates"] = rates[:, h].tolist()

        def calculate_half_response_(self, level):
            super().calculate_half_response_(level)
            fz = getattr(self, "gb_frozen_half_", None)
            if fz:   # A+B pilot: observed-edge half responses frozen to the C0 values so hidden inputs cannot recalibrate response functions
                for g in self.level2verts_[level]:
                    if g[0].Type == "T":
                        P = self.graph_[g[0].ID]["params"]
                        for c, (r, K, co, h) in enumerate(P):
                            if (g[0].ID, r) in fz:
                                P[c] = (r, K, co, fz[(g[0].ID, r)])

        def _gb_tau(self, binID, gID):
            # dynamics_CLE_simulator_ updates genes sequentially, so use this MR's own history; gene 0 may already be one step ahead.
            s = len(self.binDict[binID][gID].Conc[0]) - 1
            return DEPTH[self.gb_topo_][binID] + min(float(s) / self.gb_s_ref_, 1.0)

        def calculate_prod_rate_U_(self, gID, binID, num_c_to_evolve):
            if not hasattr(self, "gb_hidden_ids_") or gID not in self.gb_hidden_ids_ or not np.any(self.gb_coefficients_[:, self.gb_hidden_ids_[gID]]):
                return super().calculate_prod_rate_U_(gID, binID, num_c_to_evolve)
            B, _ = trajectory_basis(self.gb_topo_, np.array([binID]), np.array([self._gb_tau(binID, gID)]))
            f = (self.gb_coefficients_[0, self.gb_hidden_ids_[gID]] + B @ self.gb_coefficients_[1:, self.gb_hidden_ids_[gID]]).item()
            rate = self.gb_base_ + self.gb_amp_ * f
            if not np.isfinite(rate) or rate <= 0:
                raise FloatingPointError(f"hidden production rate nonpositive: gene={gID} bin={binID} rate={rate}")
            return [rate] * num_c_to_evolve

        def getExpressions_dynamics(self):
            ret = np.zeros((self.nBins_, self.nGenes_, self.nSC_)); ret_S = np.zeros_like(ret)
            meta = {k: np.zeros((self.nBins_, self.nSC_), dtype=(int if k != "tau" else float)) for k in ("bin", "step", "path", "tau")}
            for bi in range(self.nBins_):
                nsteps = len(self.binDict[bi][0].Conc[0]); npaths = len(self.binDict[bi][0].Conc)
                rand = np.random.choice(range(nsteps * npaths), size=self.nSC_, replace=False)
                step, path = rand % nsteps, rand // nsteps
                for gID in range(self.nGenes_):
                    ret[bi, gID] = np.take(np.concatenate(self.binDict[bi][gID].Conc), rand)
                    ret_S[bi, gID] = np.take(np.concatenate(self.binDict[bi][gID].Conc_S), rand)
                meta["bin"][bi], meta["step"][bi], meta["path"][bi] = bi, step, path
                meta["tau"][bi] = DEPTH[self.gb_topo_][bi] + np.minimum(step / float(self.gb_s_ref_), 1.0)
            return ret, ret_S, meta
    return ContinuousSergio


def H_loadings(grn_seed):
    rng = np.random.RandomState(777 + grn_seed)
    return np.array([rng.uniform(1.0, 3.0) * rng.choice([-1, 1]) for _ in range(N_GENES)])


def hidden_loadings(G, grn_seed, world, q):
    if world == "AB":
        assert q == 2
        L = np.zeros((2, N_GENES))
        for h, c in enumerate(ab_loadings(grn_seed)):
            for t in range(N_GENES):
                if t not in G["mrs"]:
                    L[h, t] = c[t]
        return L
    if world == "H":
        assert q == 1
        L = np.zeros((1, N_GENES)); c = H_loadings(grn_seed)
        for t in range(N_GENES):
            if t not in G["mrs"]:
                L[0, t] = c[t]
        return L
    if world in PLIKE:
        # Preserve the pilot's RNG call order exactly: each driver draws its own overlapping 40%-of-leaves set, then edge weights.
        elig = G["leaves"]; rng = np.random.RandomState(777 + grn_seed); L = np.zeros((q, N_GENES))
        for h in range(q):
            tg = list(rng.choice(elig, size=int(.4 * len(elig)), replace=False))
            for t in tg:
                L[h, t] = rng.uniform(1.0, 3.0) * rng.choice([-1, 1])
        if world == "P2":
            rng2 = np.random.RandomState(7777 + grn_seed)
            for h in range(q):
                for t in rng2.choice(G["ints"], size=P2_N_INT, replace=False):
                    L[h, t] = rng2.uniform(1.0, 3.0) * rng2.choice([-1, 1])
        return L
    elig = G["leaves"] if world in ("A", "C", "D") else G["leaves"] + G["ints"]
    rng = np.random.RandomState(777 + grn_seed); L = np.zeros((q, N_GENES)); per = len(elig) // max(q, 1); perm = rng.permutation(elig)
    for h in range(q):
        for t in perm[h * per:(h + 1) * per]:
            L[h, t] = HIDDEN_ABS_L * rng.choice([-1, 1])
    return L


def generator_world(world):
    if world not in WORLDS:
        raise ValueError(f"world must be one of {WORLDS}; got {world!r}")
    return "A" if world == "D" else world


def world_c_drivers(G, q):
    """Observed continuous drivers. Only MRs admit the specified basal-rate intervention without deleting endogenous regulation."""
    if q < 1 or q > len(G["mrs"]):
        return None
    return sorted(G["mrs"])[:q]


def truth_for_world(G, world, L=None, driver_ids=None):
    truth = set(G["gt"])
    if world == "C":
        if L is None or driver_ids is None or len(driver_ids) != len(L):
            raise ValueError("World C truth requires one observed driver per loading row")
        truth |= {(int(driver_ids[h]), int(t)) for h in range(len(L)) for t in np.flatnonzero(L[h])}
    return truth


def world_d_sources(G, grn_seed):
    rng = np.random.RandomState(888 + grn_seed)
    return sorted(rng.choice(sorted(G["mrs"]), 3, replace=False).astype(int).tolist() +
                  rng.choice(sorted(G["ints"]), 3, replace=False).astype(int).tolist())


def thin_world_d_counts(S, U, G, grn_seed, sim_seed, lam_tag, p=WORLD_D_P):
    """Observation-only deterministic binomial thinning; returns new float64 integer-valued count arrays."""
    S0, U0 = np.asarray(S), np.asarray(U); src = world_d_sources(G, grn_seed)
    if not (np.array_equal(S0, np.round(S0)) and np.array_equal(U0, np.round(U0))):
        raise ValueError("World D thinning requires integer-valued UMI counts")
    g = np.random.Generator(np.random.PCG64([20260924, 888 + grn_seed, sim_seed, int(round(lam_tag * 1e6))]))
    St, Ut = S0.copy(), U0.copy()
    St[:, src] = g.binomial(S0[:, src].astype(np.int64), p); Ut[:, src] = g.binomial(U0[:, src].astype(np.int64), p)
    if not (np.array_equal(St, np.round(St)) and np.array_equal(Ut, np.round(Ut))):
        raise AssertionError("World D thinning produced non-integer counts")
    other = np.setdiff1d(np.arange(S0.shape[1]), src)
    if not (np.array_equal(St[:, other], S0[:, other]) and np.array_equal(Ut[:, other], U0[:, other])):
        raise AssertionError("World D thinning changed a non-selected gene")
    return St.astype(np.float64), Ut.astype(np.float64), src


def source_support_receipt(S0, U0, S1, U1, sources, pairs=None):
    """Per-source raw detection plus pair support on a fixed pair set, all measured before nuisance conditioning."""
    out = {}
    for s in sources:
        row = {}
        for label, S, U in (("before", S0, U0), ("after", S1, U1)):
            ps, pu, pe = S[:, s] > 0, U[:, s] > 0, (S[:, s] + U[:, s]) > 0
            row[label] = dict(spliced_total=int(S[:, s].sum()), unspliced_total=int(U[:, s].sum()),
                              spliced_positive_cells=int(ps.sum()), unspliced_positive_cells=int(pu.sum()),
                              either_positive_cells=int(pe.sum()), spliced_detection=float(ps.mean()),
                              unspliced_detection=float(pu.mean()), either_detection=float(pe.mean()))
        if pairs is not None:
            row["pair_support"] = {}
            for root, pair in pairs.items():
                I, J = np.asarray(pair[0], int), np.asarray(pair[1], int); rr = {}
                for label, S, U in (("before", S0, U0), ("after", S1, U1)):
                    observed = ((S[I, s] + U[I, s]) > 0) | ((S[J, s] + U[J, s]) > 0)
                    delta = (S[J, s] != S[I, s]) | (U[J, s] != U[I, s])
                    rr[label] = dict(n_pairs=int(len(I)), either_observed=int(observed.sum()), either_delta_nonzero=int(delta.sum()),
                                     observed_fraction=float(observed.mean()) if len(I) else 0.0,
                                     delta_nonzero_fraction=float(delta.mean()) if len(I) else 0.0)
                row["pair_support"][root] = rr
        out[str(int(s))] = row
    return out


def world_d_receipt_ok(receipt, require_pairs=False):
    for row in receipt.values():
        if not (row["after"]["spliced_total"] <= row["before"]["spliced_total"] and
                row["after"]["unspliced_total"] <= row["before"]["unspliced_total"] and
                row["after"]["either_positive_cells"] < row["before"]["either_positive_cells"]):
            return False
        if require_pairs and any(not (rr["after"]["either_observed"] < rr["before"]["either_observed"] and
                                     rr["after"]["either_delta_nonzero"] < rr["before"]["either_delta_nonzero"])
                                 for rr in row.get("pair_support", {}).values()):
            return False
    return True


def build_sim(topo, G, L, coefficients, lam, grn_seed, sim_seed, workdir, world="A", driver_ids=None):
    Sergio = continuous_sergio_class(); nb, bmat = TOPOLOGIES[topo]; q = L.shape[0]; os.makedirs(workdir, exist_ok=True)
    generator_world(world)
    if world == "C":
        driver_ids = world_c_drivers(G, q) if driver_ids is None else list(driver_ids)
        if driver_ids is None or len(driver_ids) != q or any(x not in G["mrs"] for x in driver_ids):
            raise ValueError("World C requires q observed master-regulator drivers")
    else:
        driver_ids = list(range(N_GENES, N_GENES + q))
    rng = np.random.RandomState(grn_seed + 5000); lines = []
    if world == "AB": PILOT["clip_log"] = []; PILOT["mr_rate_min"] = []
    for mr in G["mrs"]:
        base = rng.uniform(0.5, 4.0)
        if world == "AB":   # A+B: positive common offset (same in all conditions) so no clipping; hidden drivers enter MRs as exogenous basal terms
            hA, hB = ab_hn(topo); cA, cB = ab_loadings(grn_seed); aa, cc = PILOT["amp"]; vals = []
            if mr == G["mrs"][0]:   # common positive offset (identical in all four conditions) so that min rate >= 0.5 over every amplitude combination
                am, cm_ = PILOT["amp_mod"]; r0 = np.random.RandomState(grn_seed + 5000); mn = np.inf
                for m2 in G["mrs"]:
                    b0 = r0.uniform(0.5, 4.0); dv = [r0.uniform(-1.5, 1.5) for _ in range(nb)]
                    for x in (0.0, am):
                        for y in (0.0, cm_):
                            mn = min(mn, min(b0 + dv[b] + x * cA[m2] * hA[b] + y * cB[m2] * hB[b] for b in range(nb)))
                PILOT["mr_off"] = max(0.0, 0.5 - mn)
            for b in range(nb):
                v = PILOT["mr_off"] + base + rng.uniform(-1.5, 1.5) + aa * cA[mr] * hA[b] + cc * cB[mr] * hB[b]
                if v < 0.01: PILOT["clip_log"].append((int(mr), b, float(v)))
                vals.append(max(0.01, v))
            PILOT.setdefault("mr_rate_min", []).append(min(vals))
            lines.append("{},{}".format(float(mr), ",".join(str(x) for x in vals)))
        elif world == "H":   # shared-H additive term (lam=0 -> identical to baseline when mr_dev_scale=1)
            hn = (profiles(topo, 1)[0] - 0.3) / 3.5; cm = H_loadings(grn_seed)[mr]
            lines.append("{},{}".format(float(mr), ",".join(str(max(0.01, base + PILOT["mr_dev_scale"] * rng.uniform(-1.5, 1.5) + lam * cm * hn[b])) for b in range(nb))))
        else:
            lines.append("{},{}".format(float(mr), ",".join(str(max(0.01, base + rng.uniform(-1.5, 1.5))) for _ in range(nb))))
    if world in PLIKE:
        pilot_rates = profiles(topo, q)
        if not np.array_equal(np.asarray(coefficients, float), pilot_rates):
            raise ValueError("World P coefficients must equal the exact pilot per-bin profiles")
        for h, rates in enumerate(pilot_rates):
            lines.append("{},{}".format(float(N_GENES + h), ",".join(str(x) for x in rates)))
    elif world != "C":
        for h in range(q):
            lines.append("{},{}".format(float(N_GENES + h), ",".join([str(HIDDEN_BASE)] * nb)))
    open(f"{workdir}/input_MRs.txt", "w").write("\n".join(lines) + "\n")
    tr = defaultdict(list)
    for (s, t), k in G["K"].items():
        tr[t].append((s, k, 2.0))
    for h in range(q):
        for t in np.flatnonzero(L[h]):
            tr[int(t)].append((driver_ids[h], lam * (PILOT["amp"][h] if world == "AB" else 1.0) * L[h, t], 2.0))
    il = ["{},{},{},{},{}".format(float(t), float(len(v)), ",".join(str(float(s)) for s, _, _ in v), ",".join(str(k) for _, k, _ in v),
                                  ",".join(str(h) for _, _, h in v)) for t, v in sorted(tr.items())]
    open(f"{workdir}/input_GRN.txt", "w").write("\n".join(il) + "\n")
    np.random.seed(1000 * grn_seed + sim_seed)
    sim = Sergio(number_genes=N_GENES + (0 if world == "C" else q), number_bins=nb, number_sc=CELLS_PER_TYPE, dynamics=True,
                 bifurcation_matrix=np.array(bmat), **SERGIO_PARAMS)
    sim.build_graph(input_file_taregts=f"{workdir}/input_GRN.txt", input_file_regs=f"{workdir}/input_MRs.txt", shared_coop_state=2)
    sim.gb_frozen_half_ = PILOT.get("frozen_half")
    if world in PLIKE:
        # The subclass adds sampled-step/tau metadata only; native per-bin rates and production remain the base SERGIO implementation.
        sim.gb_topo_, sim.gb_s_ref_ = topo, S_REF
    else:
        sim.configure_hidden_profiles(driver_ids, topo, coefficients)
    return sim, nb


LAST = {}


def simulate(topo, G, L, coefficients, lam, grn_seed, sim_seed, workdir, world="A", driver_ids=None):
    """→ latent unspliced/spliced CONCENTRATIONS (cells × genes incl. hidden), cell types, the simulator (for the equation fixture)"""
    sim, nb = build_sim(topo, G, L, coefficients, lam, grn_seed, sim_seed, workdir, world, driver_ids)
    sim.simulate_dynamics(); U, S, meta = sim.getExpressions_dynamics()
    ct = np.concatenate([[b] * S[b].shape[1] for b in range(nb)])
    Ulat = np.vstack([U[b].T for b in range(nb)])                              # unclipped latent unspliced (production equation)
    S = np.maximum(np.vstack([S[b].T for b in range(nb)]), 0); U = np.maximum(Ulat, 0)
    meta = {k: np.concatenate(v) for k, v in meta.items()}
    LAST["sha"] = sha_bytes(np.random.get_state()[1].tobytes() + str(np.random.get_state()[2:]).encode())  # post-simulation global RNG state
    return S, U, ct, sim, Ulat, meta


def render_umi(S, U, grn_seed, sim_seed, lam_tag, depth=1.0):
    """PRIMARY observation model (internal review ruling e): Poisson/UMI counts rendered from the latent concentrations BEFORE canonical
    preprocessing (SERGIO convert_to_UMIcounts_dynamics = np.random.poisson; here a dedicated seeded PCG64 stream so the
    rendering is reproducible and independent of the simulator's global stream). Continuous concentrations are a DIAGNOSTIC arm."""
    depth = float(depth)
    if not np.isfinite(depth) or depth <= 0:
        raise ValueError("Poisson observation depth must be finite and positive")
    g = np.random.Generator(np.random.PCG64([20260924, grn_seed, sim_seed, int(round(lam_tag * 1e6))]))
    return g.poisson(np.asarray(S) * depth).astype(np.float64), g.poisson(np.asarray(U) * depth).astype(np.float64)


def render_world_umi(S, U, G, world, grn_seed, sim_seed, lam_tag):
    """World observation layer. A/B/C are exactly the legacy render; only D mutates the rendered counts."""
    S0, U0 = render_umi(S, U, grn_seed, sim_seed, lam_tag)
    if world == "D":
        St, Ut, src = thin_world_d_counts(S0, U0, G, grn_seed, sim_seed, lam_tag)
        return St, Ut, src, S0, U0
    generator_world(world)
    return S0, U0, [], S0, U0


def effective_depth(obs, depth):
    if obs not in OBSERVATIONS:
        raise ValueError(f"observation must be one of {OBSERVATIONS}; got {obs!r}")
    if obs == "CONC":
        return None
    return 1.0 if obs == "UMI" else float(depth)


def observation_tag(obs, depth):
    d = effective_depth(obs, depth)
    return f"obs{obs}_depth{'NA' if d is None else f'{d:g}'}"


def render_observation(S, U, G, world, grn_seed, sim_seed, lam_tag, obs="UMI", depth=10.0):
    """Render the selected observation arm before canonical preprocessing. CONC is diagnostic and never receives the UMI exemption."""
    d = effective_depth(obs, depth)
    if obs == "CONC":
        if world == "D":
            raise ValueError("World D + CONC is undefined: its specified observation intervention is binomial UMI thinning")
        generator_world(world)
        S0, U0 = np.asarray(S, float).copy(), np.asarray(U, float).copy()
        return S0, U0, [], S0, U0
    S0, U0 = render_umi(S, U, grn_seed, sim_seed, lam_tag, d)
    if world == "D":
        St, Ut, src = thin_world_d_counts(S0, U0, G, grn_seed, sim_seed, lam_tag)
        return St, Ut, src, S0, U0
    generator_world(world)
    return S0, U0, [], S0, U0


# ───────────────────────── SERGIO's unspliced production equation (internal review finding 1) ─────────────────────────
def edge_table(sim):
    """(target, regulator, K, coop, half_response) for every interaction of every non-MR gene, read from the simulator itself"""
    rows = []
    for g in range(sim.nGenes_):
        if "params" in sim.graph_[g]:
            for (ri, K, coop, hr) in sim.graph_[g]["params"]:
                rows.append((g, int(ri), float(K), float(coop), float(hr)))
    return rows


def typed_edges(edges):
    out = np.asarray([tuple(x) for x in edges], dtype=EDGE_DTYPE)
    assert out.dtype["target"].kind in "iu" and out.dtype["regulator"].kind in "iu"
    return out


def component_edge_mask(edges, world, L, lam, driver_ids=None):
    """Rows attributable to the shared programme. World C rows are observed-driver L edges, not all edges from those drivers."""
    E = typed_edges(edges) if not (isinstance(edges, np.ndarray) and edges.dtype.fields) else edges
    if world != "C":
        return E["regulator"] >= N_GENES
    if driver_ids is None:
        raise ValueError("World C component classification requires observed driver ids")
    mask = np.zeros(len(E), bool); used = set()
    for h, r in enumerate(driver_ids):
        for t in np.flatnonzero(L[h]):
            hits = [i for i in range(len(E)) if i not in used and int(E["target"][i]) == int(t) and
                    int(E["regulator"][i]) == int(r) and np.isclose(float(E["K"][i]), float(lam * L[h, t]), rtol=0, atol=1e-12)]
            if len(hits) != 1:
                raise RuntimeError(f"World C loading edge match is not unique: driver={r} target={t} hits={hits}")
            mask[hits[0]] = True; used.add(hits[0])
    if int(mask.sum()) != int((L != 0).sum()):
        raise AssertionError("World C component-edge count mismatch")
    return mask


def hill_sergio(x, h, coop, repressive):
    """vectorised SERGIO hill_: x==0 → 0 (act) / 1 (rep); else x^c/(h^c+x^c), repression = 1 − that"""
    x = np.asarray(x, float); v = np.where(x == 0, 0.0, np.power(x, coop) / (np.power(h, coop) + np.power(x, coop) + (x == 0)))
    return np.where(x == 0, 1.0 if repressive else 0.0, 1 - v if repressive else v)


def production_parts(Uconc, edges, n_obs=N_GENES, sign=None, component_mask=None):
    """unspliced production rate per cell split into GRN part and hidden part (regulator concentration = UNSPLICED, SERGIO
    Conc; half response = the simulator's stored steady-state-unspliced mean). sign ∈ {None, 'act', 'rep'} restricts edges."""
    Pg = np.zeros((len(Uconc), n_obs)); Ph = np.zeros((len(Uconc), n_obs))
    edge_iter = list(zip(*(edges[n] for n in EDGE_DTYPE.names))) if isinstance(edges, np.ndarray) and edges.dtype.fields else list(edges)
    if component_mask is not None and len(component_mask) != len(edge_iter):
        raise ValueError("component mask length differs from edge table")
    for z, (t, r, K, c, h) in enumerate(edge_iter):
        if t >= n_obs or (sign == "act" and K < 0) or (sign == "rep" and K >= 0):
            continue
        v = abs(K) * hill_sergio(Uconc[:, r], h, c, K < 0)
        is_component = bool(component_mask[z]) if component_mask is not None else r >= n_obs
        (Ph if is_component else Pg)[:, t] += v
    if PILOT.get("mr_inj") is not None and n_obs == N_GENES and len(PILOT["mr_inj"]) == len(Uconc):
        Ph = Ph + PILOT["mr_inj"]   # A+B: hidden production entering MRs through exogenous basal terms (signed)
    return Pg, Ph


def equation_fixture(sim, max_cells=40):
    """compare production_parts against the pinned simulator's calculate_prod_rate_U_ on its OWN final bin/cell states,
    separately for activating-only and repressing-only edge sets (params temporarily restricted, then restored)"""
    res = {}
    for sign in ("act", "rep"):
        errs_a, errs_r = [], []
        for b in range(sim.nBins_):
            nc = min(max_cells, len(sim.binDict[b][0].Conc))
            Uc = np.array([[sim.binDict[b][gid].Conc[i][-1] for gid in range(sim.nGenes_)] for i in range(nc)])
            edges = edge_table(sim)
            for g in range(sim.nGenes_):
                if "params" not in sim.graph_[g]:
                    continue
                keep = [p for p in sim.graph_[g]["params"] if (p[1] >= 0) == (sign == "act")]
                if not keep:
                    continue
                orig = sim.graph_[g]["params"]; sim.graph_[g]["params"] = keep
                try:
                    ref = np.asarray(sim.calculate_prod_rate_U_(g, b, nc), float)
                finally:
                    sim.graph_[g]["params"] = orig
                mine = sum(abs(K) * hill_sergio(Uc[:, r], h, c, K < 0) for (t, r, K, c, h) in edges if t == g and (K >= 0) == (sign == "act"))
                errs_a.append(float(np.abs(mine - ref).max())); errs_r.append(float((np.abs(mine - ref) / np.maximum(np.abs(ref), 1e-300)).max()))
        res[sign] = dict(max_abs=max(errs_a) if errs_a else 0.0, max_rel=max(errs_r) if errs_r else 0.0, n_genes_tested=len(errs_a))
    res["PASS"] = all(res[s]["max_abs"] <= 1e-10 and res[s]["max_rel"] <= 1e-8 and res[s]["n_genes_tested"] > 0 for s in ("act", "rep"))
    return res


def geometry_stats(Uconc, edges, I, J, q_target=None, component_mask=None):
    """internal review ruling (a): component-energy ratio G_E = |ΔP_H|² / (|ΔP_H|² + |ΔP_GRN|²), the normalised cross-term
    2<ΔP_H, ΔP_GRN> / (|ΔP_H|² + |ΔP_GRN|²), realised hidden effective rank (participation ratio of singular values of ΔP_H)
    and pair-space principal-angle cosines between the retained hidden axes (round(r_eff)) and the GRN-production subspace
    (same number of leading axes of ΔP_GRN)."""
    Pg, Ph = production_parts(Uconc, edges, component_mask=component_mask); dG, dH = Pg[J] - Pg[I], Ph[J] - Ph[I]
    eH, eG = float((dH ** 2).sum()), float((dG ** 2).sum()); cross = float(2 * (dH * dG).sum())
    out = dict(G_E=eH / (eH + eG), cross_norm=cross / (eH + eG) if eH + eG else 0.0)
    if eH > 0:
        Uh, sh, _ = np.linalg.svd(dH, full_matrices=False); Ug, sg, _ = np.linalg.svd(dG, full_matrices=False)
        reff = float((sh ** 2).sum() ** 2 / (sh ** 4).sum()); qh = int(q_target) if q_target is not None else max(1, int(round(reff)))
        qh = max(1, min(qh, Uh.shape[1], Ug.shape[1]))
        cos = np.linalg.svd(Uh[:, :qh].T @ Ug[:, :qh], compute_uv=False)
        out.update(r_eff_hidden=reff, hidden_sv=(sh[:12] / sh[0]).round(4).tolist(), q_used=qh,
                   cos_max=float(cos.max()), cos_min=float(cos.min()))
    return out, dH


# ───────────────────────── preprocessing (gb_canonical.build, step for step) ─────────────────────────
def preprocess(a, cat_names, exemption="SERGIO_UMI"):
    import scanpy as sc, scvelo as scv, gb_canonical as GC
    cfg = json.load(open(GC.CONFIG)); fn = cfg["filter_and_normalize"]
    cat = {c.upper() for c in cat_names}
    retain = [g for g in a.var_names if str(g).upper() in cat]
    scv.pp.filter_genes(a, min_shared_counts=fn["min_shared_counts"], retain_genes=retain)
    scv.pp.normalize_per_cell(a)
    tmp = sc.AnnData(a.X.copy()); tmp.var_names = a.var_names
    sc.pp.log1p(tmp)
    sc.pp.highly_variable_genes(tmp, n_top_genes=min(fn["n_top_genes"], tmp.n_vars), flavor="seurat")
    hv = np.asarray(tmp.var["highly_variable"]).astype(bool); del tmp
    keep = hv | np.isin(np.char.upper(np.asarray(a.var_names).astype(str)), list(cat))
    a = a[:, keep].copy()
    if fn["log"]:
        sc.pp.log1p(a)
    p, nb, ve = cfg["pca"], cfg["neighbors"], cfg["velocity"]
    sc.pp.pca(a, n_comps=p["n_comps"], zero_center=p["zero_center"], random_state=p["random_state"])
    sc.pp.neighbors(a, n_neighbors=nb["n_neighbors"], n_pcs=nb["n_pcs"], random_state=nb["random_state"])
    scv.pp.moments(a, n_pcs=None, n_neighbors=None)
    scv.tl.velocity(a, mode=ve["mode"], fit_offset=ve["fit_offset"], perc=ve["perc"])
    a.uns["gb_contract"] = dict(contract=cfg["contract"], system="SERGIO", **cfg["geobridge_representation"])
    validate_sergio(a, cfg, exemption=exemption)
    return a


def exemptable_failures(a):
    """EXACT messages gb_canonical.validate emits for its magnitude heuristics (checks 3, 5) and the sparsity check 6, built with
    the SAME f-strings from the SAME values — the typed SERGIO_UMI exemption may waive exactly these and nothing else"""
    import gb_canonical as GC
    d_ = GC.d_; Ms, sp, un = d_(a.layers["Ms"]), d_(a.layers["spliced"]), d_(a.layers["unspliced"]); out = set()
    if Ms.max() < 30:
        out.add(f"Ms.max()={Ms.max():.3f} < 30: Ms looks LOG-TRANSFORMED or rescaled; "
                f"velocity is defined in linear Ms units so the representation would be incoherent")
    for nm, M in (("spliced", sp), ("unspliced", un)):
        if M.max() < 30:
            out.add(f"'{nm}'.max()={M.max():.3f} < 30: looks log-transformed; moments() would then build a logged Ms")
    if (sp == 0).mean() <= (Ms == 0).mean():
        out.add(f"Ms zero-fraction {(Ms==0).mean():.3f} is not below spliced {(sp==0).mean():.3f}; "
                f"Ms does not look like a smoothing of spliced")
    return out


def validate_sergio(a, cfg, where="sergio_v5", _orig=None, exemption="SERGIO_UMI"):
    """gb_canonical.validate + ONE typed exemption, SERGIO_UMI: applies only to data whose stored spliced/unspliced layers came
    from integer UMI rendering, including UMI_DEPTH (checked via uns['sergio_obs']=='UMI' and the raw integer check before normalisation), and
    waives only the EXACT failure strings in exemptable_failures(a). Any other failure — or any failure when the exemption is not
    SERGIO_UMI — refuses. The continuous-concentration diagnostic arm is run with exemption=None and is labelled DIAGNOSTIC."""
    import gb_canonical as GC
    try:
        (_orig or GC.validate)(a, cfg, where=where)
    except GC.ContractViolation as e:
        fails = set(str(e).split("\n  - ")[1:])
        if exemption != "SERGIO_UMI" or a.uns.get("sergio_obs") != "UMI" or not fails <= exemptable_failures(a):
            raise
    return True


def make_adata(S, U, ct, obs="UMI"):
    import anndata as ad
    a = ad.AnnData(X=S + U, layers=dict(spliced=S.copy(), unspliced=U.copy()),
                   obs=pd.DataFrame(dict(cell_type=np.asarray(ct).astype(str)), index=[f"cell_{i}" for i in range(len(S))]),
                   var=pd.DataFrame(index=[f"gene_{i}" for i in range(S.shape[1])]))
    if obs in ("UMI", "UMI_DEPTH"):
        assert np.array_equal(S, np.round(S)) and np.array_equal(U, np.round(U)), f"{obs} arm must be integer counts"
    a.uns["sergio_obs"] = "UMI" if obs in ("UMI", "UMI_DEPTH") else "CONC"
    a.uns["sergio_obs_mode"] = obs
    return a


def parity_fixture(S, U, ct, cat_names, workdir, obs="UMI"):
    """certifies the EXACT integer UMI data that are run (no rounding): gb_canonical.build itself (integer-count gate active;
    SERGIO_UMI exemption) vs the v5 preprocess"""
    import anndata as ad, gb_canonical as GC
    os.makedirs(workdir, exist_ok=True)
    if obs not in ("UMI", "UMI_DEPTH"):
        raise ValueError("canonical integer-count parity applies only to UMI observation arms")
    raw = f"{workdir}/parity_raw.h5ad"; make_adata(S, U, ct, obs=obs).write_h5ad(raw)
    catf = f"{workdir}/parity_catalogue.txt"; open(catf, "w").write("\n".join(cat_names) + "\n")
    GC.RAW["SERGIO_PARITY"] = (raw, "sergio_parity"); GC.EXTRA_TF_CATALOGUE["sergio_parity"] = catf
    orig = GC.validate

    def _v(a_, c_, where=""):
        a_.uns["sergio_obs"] = "UMI"; return validate_sergio(a_, c_, where, _orig=orig)
    GC.validate = _v
    try:
        ref = ad.read_h5ad(GC.build(f"{workdir}/parity_build", "SERGIO_PARITY", json.load(open(GC.CONFIG))))
    finally:
        GC.validate = orig
    mine = preprocess(make_adata(S, U, ct, obs=obs), cat_names)
    d = lambda M: M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    res = dict(data_sha256=sha_bytes(np.ascontiguousarray(S).tobytes() + np.ascontiguousarray(U).tobytes()),
               axis=list(ref.var_names) == list(mine.var_names), pca=float(np.abs(ref.obsm["X_pca"] - mine.obsm["X_pca"]).max()),
               graph=float(np.abs(d(ref.obsp["connectivities"]) - d(mine.obsp["connectivities"])).max()))
    for l in ("Ms", "Mu", "velocity"):
        A_, B_ = d(ref.layers[l]), d(mine.layers[l])
        res[l] = float(np.nanmax(np.abs(A_ - B_))) if (np.isnan(A_) == np.isnan(B_)).all() else float("inf")
    res["PASS"] = bool(res["axis"] and max(v for k, v in res.items() if k not in ("axis", "data_sha256")) <= 1e-5)
    return res


# ───────────────────────── frozen baseline orientation ─────────────────────────
def orientation_context(G, topo, world, q, basis_names):
    payload = dict(version="SERGIO_CONTINUOUS_V5", topo=topo, topology=TOPOLOGIES[topo], depth=DEPTH[topo], world=world, q=int(q),
                   n_genes=N_GENES, cells_per_type=CELLS_PER_TYPE, hidden_base=HIDDEN_BASE, hidden_amp=HIDDEN_AMP,
                   trunk_degrees=TRUNK_DEGREES, branch_degrees=BRANCH_DEGREES, basis_names=basis_names,
                   grn_edges=sorted([list(e) + [G["K"][e]] for e in G["K"]]), cands=G["cands"], sergio_params=SERGIO_PARAMS,
                   script_sha256=sha_file(os.path.abspath(__file__)))
    payload["context_sha256"] = sha_bytes(json.dumps(payload, sort_keys=True).encode())
    return payload


def profile_values(topo, meta, coefficients, world="A"):
    if world in PLIKE:
        rates = np.asarray(coefficients, float)
        if rates.shape != (rates.shape[0], TOPOLOGIES[topo][0]):
            raise ValueError(f"World P profile shape mismatch: {rates.shape}")
        return rates[:, np.asarray(meta["bin"], int)].T
    B, _ = trajectory_basis(topo, meta["bin"], meta["tau"])
    A = np.asarray(coefficients, float); return A[0] + B @ A[1:]


def profile_geometry(topo, meta, coefficients, world="A"):
    F = profile_values(topo, meta, coefficients, world); Fc = F - F.mean(0, keepdims=True); Gm = Fc.T @ Fc
    scale = np.sqrt(np.maximum(np.diag(Gm), 0.0)); corr = Gm / np.maximum(np.outer(scale, scale), EPS)
    off = corr - np.diag(np.diag(corr))
    return dict(cell_profile_rank=effective_rank(Fc), cell_profile_sv=np.linalg.svd(Fc, compute_uv=False).tolist(),
                cell_profile_mean_abs_max=float(np.abs(F.mean(0)).max()),
                cell_profile_orth_max=float(np.abs(Gm - np.diag(np.diag(Gm))).max()),
                cell_profile_corr=corr.tolist(), cell_profile_corr_offdiag_max=float(np.abs(off).max()),
                max_abs_profile=float(np.abs(F).max()))


def pilot_design(G, grn_seed, q, topo, design_dir, world="P"):
    """Hash-bound exact pilot programme. There is deliberately no orientation or alignment selection in World P."""
    L, rates = hidden_loadings(G, grn_seed, world, q), profiles(topo, q)
    payload = dict(version=f"SERGIO_PILOT_WORLD_{world}_V1", topo=topo, world=world, q=int(q), grn_seed=int(grn_seed),
                   profile_source="scratch/gb_sergio_drift.py:30-40", loading_source="scratch/gb_sergio_drift.py:68-78",
                   hidden_rates=rates.tolist(), loadings=L.tolist(), recipient_fraction=.4, recipient_axis="leaves" if world == "P" else "leaves+6of15_intermediates(rng 7777+grn)",
                   loading_abs_range=[1.0, 3.0], loading_signs=[-1, 1], hill=2.0, rng_seed=777 + grn_seed,
                   script_sha256=sha_file(os.path.abspath(__file__)))
    payload["context_sha256"] = sha_bytes(json.dumps(payload, sort_keys=True).encode())
    os.makedirs(design_dir, exist_ok=True); p = f"{design_dir}/g{grn_seed}_{topo}_{world}_q{q}.json"
    if os.path.exists(p):
        frozen = json.load(open(p))
        if frozen.get("context_sha256") != payload["context_sha256"]:
            raise RuntimeError(f"REFUSING_PILOT_DESIGN_HASH_MISMATCH: {p}")
    else:
        tmp = p + ".tmp"; json.dump(payload, open(tmp, "w"), indent=1); os.replace(tmp, p)
    info = dict(status="OK", design_kind="pilot_per_bin", orientation_gate_applies=False, alignment_gate_applies=False,
                rank_gate_applies=False, frozen_path=p, frozen_sha256=sha_file(p), context_sha256=payload["context_sha256"],
                driver_ids=list(range(N_GENES, N_GENES + q)), drivers_observed=False)
    return L, rates, q, info


def hidden_design(G, grn_seed, world, q_req, topo, orientation_dir, scratch_dir):
    """Create/load coefficients oriented on the λ=0, sim-seed-0 baseline. The JSON is immutable for a fixed context hash."""
    generator_world(world)
    if world in PLIKE:
        return pilot_design(G, grn_seed, q_req, topo, orientation_dir, world)
    if world == "D":
        L, coef, qr, info0 = hidden_design(G, grn_seed, "A", q_req, topo, orientation_dir, scratch_dir)
        info = dict(info0); info.update(requested_world="D", generator_world="A")
        return L, coef, qr, info
    sp = topology_spec(topo); dummy_B, basis_names = trajectory_basis(topo, np.array([0]), np.array([0.0]))
    info = dict(basis_dim=sp["dim"], basis_names=basis_names, rank_feasible=bool(q_req <= sp["dim"]), q_req=q_req)
    drivers = world_c_drivers(G, q_req) if world == "C" else list(range(N_GENES, N_GENES + q_req))
    if world == "C" and drivers is None:
        info.update(status="NO_CALL_DRIVER_COUNT", observed_driver_capacity=len(G["mrs"])); return np.zeros((q_req, N_GENES)), None, q_req, info
    info["driver_ids"] = drivers; info["drivers_observed"] = world == "C"
    if q_req > sp["dim"]:
        info["status"] = "NO_CALL_RANK"; return hidden_loadings(G, grn_seed, world, q_req), None, q_req, info
    L = hidden_loadings(G, grn_seed, world, q_req); ctx = orientation_context(G, topo, world, q_req, basis_names)
    if world == "C":
        ctx.update(driver_ids=drivers, loading_edges=[[int(drivers[h]), int(t), float(L[h, t])]
                                                      for h in range(q_req) for t in np.flatnonzero(L[h])])
        ctx["context_sha256"] = sha_bytes(json.dumps({k: v for k, v in ctx.items() if k != "context_sha256"}, sort_keys=True).encode())
    os.makedirs(orientation_dir, exist_ok=True); p = f"{orientation_dir}/g{grn_seed}_{topo}_{world}_q{q_req}.json"
    if os.path.exists(p):
        frozen = json.load(open(p))
        if frozen.get("context_sha256") != ctx["context_sha256"]:
            raise RuntimeError(f"REFUSING_ORIENTATION_HASH_MISMATCH: {p}")
        coef = np.asarray(frozen["coefficients"], float); info.update(frozen["construction"]); info.update(frozen_path=p, frozen_sha256=sha_file(p))
        return L, coef, q_req, info

    zero = np.zeros((sp["dim"] + 1, q_req))
    # C is oriented against the untouched baseline GRN, before the observed-MR programme is switched on.
    Sc, Uc, ct, sim, Ulat, meta = simulate(topo, G, L, zero, 0.0, grn_seed, 0, f"{scratch_dir}/orientation_baseline", "A")
    S, U = render_umi(Sc, Uc, grn_seed, 0, 0.0)
    a = preprocess(make_adata(S[:, :N_GENES], U[:, :N_GENES], ct), [f"gene_{c}" for c in G["cands"]])
    from gb_gata1_preflight import canonical_pairs
    import gb_canonical as GC
    pc = json.load(open(GC.CONFIG))["pair_construction"]; V = np.asarray(a.layers["velocity"], float)
    if not np.isfinite(V).all():
        info["status"] = "NO_CALL_VELOCITY"; return L, None, q_req, info
    X = np.asarray(a.X, float); ids = np.asarray(a.obs_names).astype(str); Bcell, _ = trajectory_basis(topo, meta["bin"], meta["tau"])
    Pg, _ = production_parts(Ulat, edge_table(sim)); pair_blocks, dG_blocks, pairs_hash = [], [], {}
    for root in ROOTS:
        _, I, J, _ = canonical_pairs(X, V, ids, root, seed=pc["pair_seed"], min_forward_dt=pc["min_forward_dt"],
                                     max_forward_dt=pc["max_forward_dt"], max_pairs_per_cell=pc["max_pairs_per_cell"])
        I, J = np.asarray(I, int), np.asarray(J, int); pair_blocks.append(Bcell[J] - Bcell[I]); dG_blocks.append(Pg[J] - Pg[I])
        pairs_hash[root] = sha_bytes(np.ascontiguousarray(np.c_[I, J]).tobytes())
    dGcat = np.vstack(dG_blocks); Ug = np.linalg.svd(dGcat, full_matrices=False)[0][:, :q_req]
    coef, construction = orient_profiles(Bcell, pair_blocks, [Ug], q_req, world); info.update(construction)
    if coef is None:
        return L, None, q_req, info
    dense_bin = np.concatenate([np.repeat(b, 1001) for b in range(TOPOLOGIES[topo][0])])
    dense_tau = np.concatenate([DEPTH[topo][b] + np.linspace(0, 1, 1001) for b in range(TOPOLOGIES[topo][0])])
    domain_max = float(np.abs(profile_values(topo, dict(bin=dense_bin, tau=dense_tau), coef)).max())
    if domain_max > 1.0:
        coef /= domain_max
    info["continuous_domain_max_abs_before_scale"] = domain_max
    info["continuous_domain_max_abs"] = float(np.abs(profile_values(topo, dict(bin=dense_bin, tau=dense_tau), coef)).max())
    # The selection used the concatenated roots; record each root separately without changing the frozen coefficients.
    for root, dB, dG in zip(ROOTS, pair_blocks, dG_blocks):
        Uh = _orth(dB @ coef[1:]); Ug_r = np.linalg.svd(dG, full_matrices=False)[0][:, :q_req]
        cs = np.linalg.svd(Uh.T @ Ug_r, compute_uv=False)
        info[f"construction_cos_{root}"] = cs.tolist()
    root_cos = [np.asarray(info[f"construction_cos_{r}"], float) for r in ROOTS]
    if world == "A" and any(len(c) == 0 or c.max() > .30 for c in root_cos): info["status"] = "NO_CALL_ALIGNMENT"
    if world == "B" and any(len(c) < q_req or c.min() < .80 for c in root_cos): info["status"] = "NO_CALL_ALIGNMENT"
    if world == "C": info["alignment_threshold_is_diagnostic_only"] = True
    prof = profile_geometry(topo, meta, coef); info.update(prof)
    end_B, _ = trajectory_basis(topo, np.arange(TOPOLOGIES[topo][0]), np.asarray(DEPTH[topo], float) + 1.0)
    end_rates = HIDDEN_BASE + HIDDEN_AMP * (coef[0] + end_B @ coef[1:])
    if np.any(end_rates <= 0) or np.abs(profile_values(topo, meta, coef)).max() > 1 + 1e-10:
        raise AssertionError("profile/rate invariant failed")
    frozen = dict(**ctx, coefficients=coef.tolist(), construction=info, baseline_sim_seed=0, baseline_pairs_sha256=pairs_hash,
                  baseline_observed_sha256=sha_bytes(np.ascontiguousarray(S[:, :N_GENES]).tobytes() + np.ascontiguousarray(U[:, :N_GENES]).tobytes()),
                  end_rates=end_rates.T.tolist())
    tmp = p + ".tmp"; json.dump(frozen, open(tmp, "w"), indent=1, default=float); os.replace(tmp, p)
    info.update(frozen_path=p, frozen_sha256=sha_file(p)); return L, coef, q_req, info


# ───────────────────────── scoring ─────────────────────────
def haar(n, seed):
    """Haar-uniform O(n): QR of a Gaussian with the sign correction diag(sign(diag R)) (Mezzadri 2007)"""
    Z = np.random.Generator(np.random.PCG64(seed)).standard_normal((n, n)); Qz, Rz = np.linalg.qr(Z)
    return Qz * np.sign(np.diag(Rz))


def _cop_workspace(Qt, D, w):
    Qt, D, w = np.asarray(Qt, float), np.asarray(D, float), np.asarray(w, float)
    return dict(Dcoef=Qt.T @ D, WG=Qt.T @ (w[:, None] * D), Wb=Qt.T @ (w[:, None] * Qt), S0=float((w[:, None] * D * D).sum()))


def _cop_from_coeff(A, ws):
    if A.shape[1] == 0:
        return 0.0
    c, G = A.T @ ws["Dcoef"], A.T @ ws["WG"]; M = A.T @ ws["Wb"] @ A
    residual = ws["S0"] - 2.0 * float((c * G).sum()) + float((M * (c @ c.T)).sum())
    return 1.0 - residual / ws["S0"]


def equal_c_projector(Qt, D, w, Qhidden, mode, tol=.01, n_random=512, seed=20260924):
    """Nested rank-k projectors in span(Qt), matching TRUE C_op(k) while greedily extremising hidden-subspace overlap.
    A prefix with no candidate inside `tol` and every later prefix are invalid; callers must emit NaN for those cells."""
    Qt, Qh = np.asarray(Qt, float), _orth(np.asarray(Qhidden, float)); K = Qt.shape[1]; ws = _cop_workspace(Qt, D, w)
    target = np.array([_cop_from_coeff(np.eye(K)[:, :k], ws) for k in range(K + 1)])
    Hc = Qt.T @ Qh; A = np.empty((K, 0)); valid = np.ones(K + 1, bool); achieved = [0.0]; overlaps = [0.0]
    rng = np.random.Generator(np.random.PCG64(seed)); failed_at = None
    for k in range(1, K + 1):
        bank = [np.eye(K)[:, j] for j in range(K)] + [Hc[:, j] for j in range(Hc.shape[1])]
        bank += list(rng.standard_normal((n_random, K)))
        candidates = []
        for idx, v in enumerate(bank):
            v = np.asarray(v, float) - A @ (A.T @ v)
            nv = np.linalg.norm(v)
            if nv <= 1e-10:
                continue
            v /= nv; At = np.column_stack([A, v]); c = _cop_from_coeff(At, ws); err = abs(c - target[k])
            if err <= tol + 1e-12:
                ov = float(np.linalg.norm(At.T @ Hc, "fro") ** 2 / max(min(k, Qh.shape[1]), 1))
                candidates.append((ov, err, idx, v, c))
        if not candidates:
            failed_at = k; valid[k:] = False; break
        pick = sorted(candidates, key=(lambda x: (-x[0], x[1], x[2])) if mode == "hi" else (lambda x: (x[0], x[1], x[2])))[0]
        A = np.column_stack([A, pick[3]]); achieved.append(pick[4]); overlaps.append(pick[0])
    # Complete Q only to satisfy the scorer's shape contract. Invalid prefixes are explicitly overwritten with NaN downstream.
    for v in np.eye(K).T:
        if A.shape[1] == K: break
        v = v - A @ (A.T @ v); nv = np.linalg.norm(v)
        if nv > 1e-10: A = np.column_stack([A, v / nv])
    while len(achieved) < K + 1:
        achieved.append(_cop_from_coeff(A[:, :len(achieved)], ws)); overlaps.append(float("nan"))
    err = np.abs(np.asarray(achieved) - target)
    info = dict(status="OK" if valid.all() else "NO_CALL_EQUAL_C", mode=mode, tolerance=tol, failed_at=failed_at,
                target_C=target.tolist(), achieved_C=np.asarray(achieved).tolist(), match_error=err.tolist(),
                overlap_hidden=overlaps, valid=valid.tolist(), max_match_error_valid=float(err[valid].max()))
    return Qt @ A, info


def arm_Q(arm, dP, Qt, eqc=None):
    if arm == "TRUE":
        return Qt
    if arm == "REV":
        return np.linalg.qr(dP[:, ::-1])[0]
    if arm.startswith("RIS"):                                       # internal review: Haar rotation of the FROZEN 30-PC basis, nested prefixes
        return Qt @ haar(Qt.shape[1], int(arm[3:]))
    if arm.startswith("ROS"):
        Gm = np.random.Generator(np.random.PCG64(100 + int(arm[3:]))).standard_normal(dP.shape)
        return np.linalg.qr(Gm - Qt @ (Qt.T @ Gm))[0]
    if arm in ("EQC_hi", "EQC_lo"):
        return eqc[arm][0]
    raise ValueError(arm)


def score_dataset(a, src_idx, perm_u=None, hidden_cell=None, exclude_cells=None, fixed_pairs=None):
    from gb_gata1_preflight import canonical_pairs
    from gb_target_extension import project_scores
    import gb_canonical as GC
    pc = json.load(open(GC.CONFIG))["pair_construction"]
    d = lambda M: M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    Xlog = d(a.X).astype(np.float64); Vr = d(a.layers["velocity"]).astype(np.float64)
    Ms = d(a.layers["Ms"]).astype(np.float64); Mu = d(a.layers["Mu"]).astype(np.float64)
    V0 = np.nan_to_num(Vr, nan=0.0); P30 = np.asarray(a.obsm["X_pca"], float)[:, :30]; ids = np.asarray(a.obs_names).astype(str)
    raw, pairs, fx, krule, info_matched = {}, {}, {}, {}, {}
    for root in ROOTS:
        if fixed_pairs is None:
            _pt, I, J, _ = canonical_pairs(Xlog, V0, ids, root, seed=pc["pair_seed"], min_forward_dt=pc["min_forward_dt"],
                                           max_forward_dt=pc["max_forward_dt"], max_pairs_per_cell=pc["max_pairs_per_cell"])
            I, J, pt = np.asarray(I, int), np.asarray(J, int), np.asarray(_pt, float)
        else:
            I, J, pt = fixed_pairs[root]
            I, J, pt = np.asarray(I, int).copy(), np.asarray(J, int).copy(), np.asarray(pt, float).copy()
        if exclude_cells is not None:
            keep = ~(np.asarray(exclude_cells, bool)[I] | np.asarray(exclude_cells, bool)[J])
            fx[f"{root}|root_transient_pairs_retained"] = int(keep.sum())
            fx[f"{root}|root_transient_pairs_total"] = int(len(keep))
            I, J = I[keep], J[keep]                                    # pt is per-CELL (canonical_pairs), not per-pair
        pairs[root] = (I, J, pt)
        dP = P30[J] - P30[I]; Qt = np.linalg.qr(dP)[0]
        sv = np.linalg.svd(dP, compute_uv=False); fx[f"{root}|r_eff_dP"] = float(sv.sum() ** 2 / (sv ** 2).sum())
        dx = (Ms[J] - Ms[I])[:, src_idx].astype(np.float32)
        dys = {"U": Mu[J] - Mu[I], "X": Ms[J] - Ms[I], "V": V0[J] - V0[I]}
        if perm_u is not None:
            dys = {"U": dys["U"][np.random.Generator(np.random.PCG64(perm_u)).permutation(len(I))]}
        else:
            import gb_infomatched as IM
            imo = {nm: np.full((KMAX + 1, len(src_idx), Mu.shape[1]), np.nan, np.float32)
                   for nm in ("linear", "spline", "matched")}
            for k_, lin_, spl_, matched_ in IM.scores(dx, dys["U"], Qt, I, J, range(KMAX + 1), len(Ms), matched=True):
                imo["linear"][k_], imo["spline"][k_], imo["matched"][k_] = lin_, spl_, matched_
            info_matched[root] = imo
        eqc = {}
        if perm_u is None:
            import gb_geometry_audit as GA
            sd = Ms.std(0); gref = sd > 0; Dref = (Ms[J][:, gref] - Ms[I][:, gref]) / sd[gref]
            Qh = _orth(np.asarray(hidden_cell, float)[J] - np.asarray(hidden_cell, float)[I]) if hidden_cell is not None else np.empty((len(I), 0))
            w = GA.weights(I, J, len(Ms))["pair"]
            for mode in ("hi", "lo"):
                if Qh.shape[1] == 0:
                    eqc[f"EQC_{mode}"] = (Qt, dict(status="NO_CALL_EQUAL_C_NO_HIDDEN", mode=mode, tolerance=.01, failed_at=1,
                                                    target_C=GA.cw_curve(Qt, Ms[:, gref], I, J, w, sd[gref]).tolist(),
                                                    achieved_C=[0.0] + [float("nan")] * KMAX, match_error=[0.0] + [float("nan")] * KMAX,
                                                    overlap_hidden=[float("nan")] * (KMAX + 1), valid=[True] + [False] * KMAX,
                                                    max_match_error_valid=0.0))
                else:
                    eqc[f"EQC_{mode}"] = equal_c_projector(Qt, Dref, w, Qh, mode)
                fx[f"{root}|EQC_{mode}|info"] = eqc[f"EQC_{mode}"][1]
        for arm in (["TRUE"] if perm_u is not None else ARMS):
            Q = arm_Q(arm, dP, Qt, eqc)
            if arm.startswith("ROS"):
                fx[f"{root}|{arm}|orth"] = float(np.linalg.norm(Q.T @ Qt, 2))
            if perm_u is None:                                            # internal review (5): k-rule development quantities from the ACTUAL prefix
                cP = np.cumsum((Q.T @ dP) ** 2, 0).sum(1) / (dP ** 2).sum()
                cX = np.cumsum((Q.T @ dx.astype(np.float64)) ** 2, 0) / (dx.astype(np.float64) ** 2).sum(0)
                import gb_geometry_audit as GA                            # C_op harmonised with the REAL-DATA definition (internal review):
                gref = Ms.std(0) > 0; Wt = GA.weights(I, J, len(Ms))      # standardised Ms pair differences of all state genes,
                krule[(arm, root)] = dict(C_dP=np.r_[0.0, cP],             # pair-uniform (primary) and cell-balanced (sensitivity)
                                          C_op=GA.cw_curve(Q, Ms[:, gref], I, J, Wt["pair"], Ms.std(0)[gref]),
                                          C_op_cell=GA.cw_curve(Q, Ms[:, gref], I, J, Wt["cell"], Ms.std(0)[gref]),
                                          Q_r=np.vstack([np.ones(len(src_idx)), 1 - cX]))
            for ch, dy in dys.items():
                out = {nm: np.full((KMAX + 1, len(src_idx), dy.shape[1]), np.nan, np.float32) for nm in ("num", "den", "z2", "absS")}
                for k, num, den, z2, absS in project_scores(dx, dy.astype(np.float32), Q):
                    for nm, v in (("num", num), ("den", den), ("z2", z2), ("absS", absS)):
                        out[nm][k] = v
                if arm.startswith("EQC"):
                    ok = np.asarray(eqc[arm][1]["valid"], bool)
                    for nm in out:
                        out[nm][~ok] = np.nan
                for i, s in enumerate(src_idx):
                    out["absS"][:, i, s] = np.nan
                raw[(arm, ch, root)] = out
    if perm_u is None:
        for root in ROOTS:
            for ch in CHANS:
                b0 = raw[("TRUE", ch, root)]["absS"][0]; b30 = raw[("TRUE", ch, root)]["absS"][KMAX]
                fx[f"{root}|{ch}|k0_all_equal"] = bool(all(np.array_equal(raw[(a_, ch, root)]["absS"][0], b0, equal_nan=True) for a_ in ARMS))
                fx[f"{root}|{ch}|k30_span_maxdiff"] = float(max(np.nanmax(np.abs(raw[(a_, ch, root)]["absS"][KMAX] - b30))
                                                                  for a_ in BASE_ARMS if not a_.startswith("ROS")))
    return (raw, pairs, fx, krule, info_matched) if perm_u is None else (raw, pairs, fx)


def pct_within_source(M):
    from scipy.stats import rankdata
    return np.vstack([rankdata(r, method="average") / r.size for r in M])


def production_scalar(absS_by_root, cols):
    """spec v1 on a fixed per-row column set: cols (S×T index array); within-source percentile per (root,k), mean K_ADM, min roots"""
    per = [np.mean([pct_within_source(np.take_along_axis(absS_by_root[r][k], cols, 1)) for k in KA], 0) for r in ROOTS]
    return np.minimum(*per)


def reference_scalar(absS_by_root, cols):
    S, T = cols.shape; out = np.full((S, T), np.inf)
    for r in ROOTS:
        acc = np.zeros((S, T))
        for k in KA:
            for i in range(S):
                acc[i] += pd.Series(absS_by_root[r][k][i, cols[i]]).rank(method="average").to_numpy() / T
        out = np.minimum(out, acc / len(KA))
    return out


# ───────────────────────── metrics ─────────────────────────
def ap(y, s):
    from sklearn.metrics import average_precision_score
    return float(average_precision_score(y, s))


def endpoints(M, Y, CLS, true_rows):
    from sklearn.metrics import roc_auc_score
    s, y, c = M.ravel(), Y.ravel().astype(bool), CLS.ravel(); pi = y.mean(); o = np.argsort(-s, kind="stable")
    Pm = pct_within_source(M).ravel()
    macro = [((ap(Y[i], M[i]) - Y[i].mean()) / (1 - Y[i].mean())) for i in true_rows if 0 < Y[i].sum() < Y.shape[1]]
    di = c != "none"; dvi = (c[di] == "direct"); top = c[o[:50]]; nr = ((top == "direct") | (top == "indirect")).sum()
    return dict(prevalence=pi, AP_native=ap(y, s), lift_native=ap(y, s) / pi, nAP_native=(ap(y, s) - pi) / (1 - pi),
                AP_pct=ap(y, Pm), nAP_pct=(ap(y, Pm) - pi) / (1 - pi), macro_nAP=float(np.mean(macro)) if macro else np.nan,
                AUROC=float(roc_auc_score(y, s)), P50=float(y[o[:50]].mean()), indirect50=float((top == "indirect").mean()),
                AUROC_dir_vs_ind=float(roc_auc_score(dvi, s[di])) if 0 < dvi.sum() < di.sum() else np.nan,
                AP_dir_vs_ind=ap(dvi, s[di]) if 0 < dvi.sum() < di.sum() else np.nan, AP_dir_vs_ind_baseline=float(dvi.mean()),
                EDP50=float((top == "direct").sum() / nr) if nr else np.nan, EDP_baseline=float(dvi.mean()))


def rewire_double_swap(Y, cols, src, rng, n_swaps=None):
    """directed double-edge swaps restricted to the roster: (a→b),(c→d) → (a→d),(c→b) accepted only if a≠c, b≠d, both new edges are
    roster edges (non-self; the roster is candidates × all non-self genes) and not already present. Preserves source out-degree,
    target in-degree, edge count, self-edge exclusion and uniqueness exactly. Y (S × T) boolean on the roster columns `cols`."""
    E = [(src[i], int(cols[i][j])) for i, j in zip(*np.nonzero(Y))]; Es = set(E); n_swaps = n_swaps or 10 * len(E)
    rowpos = {s_: i for i, s_ in enumerate(src)}
    for _ in range(n_swaps):
        i1, i2 = rng.choice(len(E), 2, replace=False); (a, b), (c, d) = E[i1], E[i2]
        if a == c or b == d or a == d or c == b or (a, d) in Es or (c, b) in Es:
            continue
        Es -= {(a, b), (c, d)}; Es |= {(a, d), (c, b)}; E[i1], E[i2] = (a, d), (c, b)
    Yp = np.zeros_like(Y)
    for a, b in Es:
        i = rowpos[a]; Yp[i, int(np.flatnonzero(cols[i] == b)[0])] = True
    return Yp


def degree_fixture(Y, Yp, cols):
    tin = lambda M: np.bincount(cols[M].ravel() if M.any() else np.array([], int), minlength=N_GENES)
    return dict(out_deg=bool(np.array_equal(Y.sum(1), Yp.sum(1))), in_deg=bool(np.array_equal(tin(Y), tin(Yp))),
                n_edges=bool(Y.sum() == Yp.sum()), changed_frac=float((Y & ~Yp).sum() / max(Y.sum(), 1)))


def nap_pct(M, Y):
    y = Y.ravel().astype(bool); pi = y.mean(); return (ap(y, pct_within_source(M).ravel()) - pi) / (1 - pi)


def macro_nap(M, Y):
    vals = [(ap(Y[i], M[i]) - Y[i].mean()) / (1 - Y[i].mean()) for i in range(len(Y)) if 0 < Y[i].sum() < Y.shape[1]]
    return float(np.mean(vals)) if vals else np.nan


def g_crossfit(dY, dH, cells_I, cells_J, fold):
    sse = sst = 0.0
    for f in (0, 1):
        tr = (fold[cells_I] == f) & (fold[cells_J] == f); te = (fold[cells_I] == 1 - f) & (fold[cells_J] == 1 - f)
        B = np.linalg.lstsq(np.c_[np.ones(tr.sum()), dH[tr]], dY[tr], rcond=None)[0]
        pred = np.c_[np.ones(te.sum()), dH[te]] @ B
        sse += ((dY[te] - pred) ** 2).sum(); sst += ((dY[te] - dY[tr].mean(0)) ** 2).sum()
    return 1 - sse / sst


# ───────────────────────── one dataset ─────────────────────────
def check_grn_seed(g, confirm):
    if g in CONF_GRNS and not confirm:
        raise SystemExit(f"REFUSING: GRN {g} is a CONFIRMATION GRN (sealed until the v2 profile is frozen)")
    if g not in DEV_GRNS | CONF_GRNS:
        raise SystemExit(f"REFUSING: GRN {g} is neither declared DEV nor CONFIRMATION")


def validate_tuning_json(path, topo, world, q, grn_seed, obs="UMI", depth=10.0):
    tune_rec = json.load(open(path)); expected = dict(topo=topo, world=world, q=int(q), grn_seed=int(grn_seed), sim_seed=0,
                                                        obs=obs, depth=effective_depth(obs, depth),
                                                        script_sha256=sha_file(os.path.abspath(__file__)))
    bad = {k: (tune_rec.get(k), v) for k, v in expected.items() if tune_rec.get(k) != v}
    if bad:
        raise RuntimeError(f"REFUSING_TUNING_METADATA_MISMATCH: {path}: {bad}")
    if not isinstance(tune_rec.get("lambda_for_G"), dict):
        raise RuntimeError(f"REFUSING_TUNING_TABLE_MISSING: {path}")
    design_key = "programme" if world in PLIKE else "orientation"; design = tune_rec.get(design_key)
    if not isinstance(design, dict) or not design.get("relative_path") or not design.get("sha256"):
        raise RuntimeError(f"REFUSING_TUNING_{design_key.upper()}_MISSING: {path}")
    op = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), design["relative_path"]))
    if not os.path.isfile(op) or sha_file(op) != design["sha256"]:
        raise RuntimeError(f"REFUSING_TUNING_{design_key.upper()}_HASH_MISMATCH: {op}")
    frozen = json.load(open(op))
    if frozen.get("context_sha256") != design.get("context_sha256"):
        raise RuntimeError(f"REFUSING_TUNING_{design_key.upper()}_CONTEXT_MISMATCH: {op}")
    tune_rec["_orientation_dir"] = os.path.dirname(op)
    tune_rec["_design_key"] = design_key
    tune_rec["_geometry_fixture_paths"] = {}
    fixtures = tune_rec.get("geometry_fixtures")
    if not isinstance(fixtures, dict):
        raise RuntimeError(f"REFUSING_TUNING_GEOMETRY_FIXTURES_MISSING: {path}")
    for target, lam in tune_rec["lambda_for_G"].items():
        if float(target) == 0 or lam is None:
            continue
        fx = fixtures.get(target)
        if not isinstance(fx, dict) or not fx.get("relative_path") or not fx.get("sha256"):
            raise RuntimeError(f"REFUSING_TUNING_GEOMETRY_FIXTURE_MISSING: target={target}")
        fp = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(path)), fx["relative_path"]))
        if not os.path.isfile(fp) or sha_file(fp) != fx["sha256"]:
            raise RuntimeError(f"REFUSING_TUNING_GEOMETRY_FIXTURE_HASH_MISMATCH: {fp}")
        tune_rec["_geometry_fixture_paths"][target] = fp
    return tune_rec


def tuning_json_path(out, topo, world, q, grn_seed, obs="UMI", depth=10.0):
    return f"{out}/tune_{topo}_{world}_q{q}_{observation_tag(obs, depth)}_g{grn_seed}.json"


def root_transient_cells(topo, meta):
    bins, steps = np.asarray(meta["bin"], int), np.asarray(meta["step"], int)
    return (np.asarray(DEPTH[topo], int)[bins] == 0) & (steps < S_REF)


def fixture_statuses(arms_ok, scalar_diff, rewiring_ok, persistence_diff, tidy_diff):
    out = []
    if not arms_ok: out.append("NO_CALL_ARMS")
    if not np.isfinite(scalar_diff) or scalar_diff > 1e-12: out.append("NO_CALL_SCALAR_PARITY")
    if not rewiring_ok: out.append("NO_CALL_REWIRING_DEGREE")
    if not np.isfinite(persistence_diff) or persistence_diff > 1e-12 or not np.isfinite(tidy_diff) or tidy_diff > 1e-12:
        out.append("NO_CALL_PERSISTENCE")
    return out


def canonical_geometry_pairs(a):
    """The single production pair-construction path used by tuning, run_one, and scoring."""
    from gb_gata1_preflight import canonical_pairs
    import gb_canonical as GC
    pc = json.load(open(GC.CONFIG))["pair_construction"]
    d = lambda M: M.toarray() if hasattr(M, "toarray") else np.asarray(M)
    X, V = d(a.X).astype(np.float64), d(a.layers["velocity"]).astype(np.float64)
    ids, out = np.asarray(a.obs_names).astype(str), {}
    for root in ROOTS:
        pt, I, J, _ = canonical_pairs(X, V, ids, root, seed=pc["pair_seed"], min_forward_dt=pc["min_forward_dt"],
                                      max_forward_dt=pc["max_forward_dt"], max_pairs_per_cell=pc["max_pairs_per_cell"])
        out[root] = (np.asarray(I, int), np.asarray(J, int), np.asarray(pt, float))
    return out


def dataset_geometry(a, Ulat, edges, component_mask, q, topo, world, lam, meta, coefficients, fixed_pairs=None):
    """One pair/geometry implementation for both tuning and run_one. World C is a leading-structure falsifier:
    its realised rank is descriptive, not a certification gate; leading-prefix energy is reported instead."""
    pairs = canonical_geometry_pairs(a) if fixed_pairs is None else fixed_pairs
    out = dict(profile_geometry=profile_geometry(topo, meta, coefficients, world))
    P30 = np.asarray(a.obsm["X_pca"], float)[:, :KMAX]
    for root in ROOTS:
        I, J, _ = pairs[root]; out[root], dH = geometry_stats(Ulat, edges, I, J, q, component_mask)
        dP = P30[J] - P30[I]; Qall = np.linalg.qr(dP)[0]
        energy = np.maximum(np.linalg.norm(dH, "fro") ** 2, EPS)
        Qlead, Q3 = Qall[:, :min(q, dP.shape[1])], Qall[:, :min(3, dP.shape[1])]
        out[root]["leading_prefix_energy_fraction"] = float(np.linalg.norm(Qlead.T @ dH, "fro") ** 2 / energy)
        out[root]["driver_pair_energy_prefix3"] = float(np.linalg.norm(Q3.T @ dH, "fro") ** 2 / energy)
    alignment_gate_applies = world not in PLIKE
    bad_align = alignment_gate_applies and lam > 0 and ((world in ("A", "D") and any(out[r].get("cos_max", 1.0) > .30 for r in ROOTS)) or
                             (world == "B" and any(out[r].get("cos_min", 0.0) < .80 for r in ROOTS)))
    rank_gate_applies = world not in ("C",) + PLIKE
    bad_pair_rank = rank_gate_applies and lam > 0 and any(not (.8 * q <= out[r].get("r_eff_hidden", 0.0) <= 1.2 * q) for r in ROOTS)
    bad_cell_rank = rank_gate_applies and not (.8 * q <= out["profile_geometry"]["cell_profile_rank"] <= 1.2 * q)
    out["certification"] = dict(alignment_ok=not bad_align, pair_rank_ok=not bad_pair_rank, cell_rank_ok=not bad_cell_rank)
    out["alignment_gate_applies"] = alignment_gate_applies
    out["rank_gate_applies"] = rank_gate_applies
    if world == "C":
        out["falsifier_geometry"] = dict(
            realised_pair_rank={r: out[r].get("r_eff_hidden", 0.0) for r in ROOTS},
            realised_cell_profile_rank=out["profile_geometry"]["cell_profile_rank"],
            leading_prefix_energy_fraction={r: out[r]["leading_prefix_energy_fraction"] for r in ROOTS})
    return out, pairs


def tuning_trace_geometry(tune_rec, target, lam):
    hits = [x for x in tune_rec.get("trace", []) if "G_E" in x and np.isclose(float(x.get("target", np.nan)), float(target), rtol=0, atol=1e-12)
            and np.isclose(float(x.get("lam", np.nan)), float(lam), rtol=0, atol=1e-12)]
    if len(hits) != 1:
        raise RuntimeError(f"REFUSING_TUNING_TRACE_MATCH: target={target} lambda={lam} matches={len(hits)}")
    return np.asarray(hits[0]["G_E"], float)


def run_one(out_root, topo, world, q, lam, gtarget, grn_seed, sim_seed, client, tune_path=None, obs="UMI", depth=10.0):
    t0 = time.time(); generator_world(world); G = generate_grn(grn_seed)
    obs_depth = effective_depth(obs, depth)
    tune_rec = validate_tuning_json(tune_path, topo, world, q, grn_seed, obs, depth) if tune_path else None
    if tune_rec is not None:
        if gtarget is None:
            raise RuntimeError("raw-lambda runs must not be bound to a G-target tuning file")
        tuned = tune_rec["lambda_for_G"].get(f"{gtarget:.2f}")
        if tuned is None or not np.isclose(float(tuned), float(lam), rtol=0, atol=1e-12):
            raise RuntimeError(f"REFUSING_TUNING_LAMBDA_MISMATCH: target={gtarget} requested={lam} tuned={tuned}")
    strength_tag = f"G{gtarget:g}" if gtarget is not None else f"lam{lam:g}"
    tag = f"{topo}_{world}_q{q}_{strength_tag}_{observation_tag(obs, depth)}_g{grn_seed}_s{sim_seed}"
    dd = f"{out_root}/{tag}"; os.makedirs(f"{dd}/raw", exist_ok=True)
    orientation_dir = tune_rec["_orientation_dir"] if tune_rec is not None else f"{out_root}/orientation"
    L, coefficients, q_real, hinfo = hidden_design(G, grn_seed, world, q, topo, orientation_dir,
                                                   f"{out_root}/orientation_work/g{grn_seed}_{topo}_{world}_q{q}")
    if coefficients is None or hinfo.get("status") != "OK":
        rec = dict(tag=tag, topo=topo, world=world, q_req=q, q_real=q_real, lam=lam, G_target=gtarget, grn_seed=grn_seed,
                   obs=obs, depth=obs_depth,
                   sim_seed=sim_seed, hidden_info=hinfo, status=hinfo.get("status", "NO_CALL_ORIENTATION"),
                   script_sha256=sha_file(os.path.abspath(__file__)))
        json.dump(rec, open(f"{dd}/record.json", "w"), indent=1, default=float); print(tag, rec["status"]); return rec
    driver_ids = world_c_drivers(G, q) if world == "C" else list(range(N_GENES, N_GENES + q))
    Sc, Uc_, ct, sim, Ulat, meta = simulate(topo, G, L, coefficients, lam, grn_seed, sim_seed, f"{dd}/sergio_inputs", world, driver_ids)
    S, U, thinned_sources, S_render, U_render = render_observation(Sc, Uc_, G, world, grn_seed, sim_seed, 0.0 if world in ("H", "AB") else lam, obs, depth)  # DOSE: Poisson render stream must not depend on lam
    if world == "D":
        support_receipt = source_support_receipt(S_render[:, :N_GENES], U_render[:, :N_GENES], S[:, :N_GENES], U[:, :N_GENES], thinned_sources)
    else:
        support_receipt = None
    edges = edge_table(sim); comp_mask = component_edge_mask(edges, world, L, lam, driver_ids if world == "C" else None)
    if world == "AB":
        hA, hB = ab_hn(topo); cA, cB = ab_loadings(grn_seed); inj = np.zeros((len(Ulat), N_GENES)); bb = np.asarray(meta["bin"], int)
        for m_ in G["mrs"]:
            inj[:, m_] = PILOT["amp"][0] * cA[m_] * hA[bb] + PILOT["amp"][1] * cB[m_] * hB[bb]
        PILOT["mr_inj"] = inj
        rec_ab = dict(amp=list(PILOT["amp"]), mr_off=PILOT["mr_off"], n_mr_clipped=len(PILOT["clip_log"]), clip_log=PILOT["clip_log"],
                      mr_rate_min=float(min(PILOT.get("mr_rate_min", [np.nan]))), frozen_half_sha=PILOT.get("frozen_half_sha"))
    n = N_GENES; names = [f"gene_{i}" for i in range(n)]; cat_names = [names[c] for c in G["cands"]]
    truth_set = truth_for_world(G, world, L, driver_ids if world == "C" else None)
    truth = sorted([int(x), int(y)] for x, y in truth_set)
    rec = dict(tag=tag, topo=topo, world=world, q_req=q, q_real=q_real, lam=lam, G_target=gtarget, grn_seed=grn_seed, sim_seed=sim_seed,
               obs=obs, depth=obs_depth, diagnostic_observation=obs == "CONC",
               grn_role="DEV" if grn_seed in DEV_GRNS else "CONFIRMATION", script_sha256=sha_file(os.path.abspath(__file__)),
               tuning_sha256=sha_file(tune_path) if tune_path else None,
               tuning_scope=dict(topo=topo, world=world, q=int(q), grn_seed=int(grn_seed), obs=obs, depth=obs_depth),
               tuning_script_sha256=tune_rec.get("script_sha256") if tune_rec else None,
               config_sha256=sha_file(__import__("gb_canonical").CONFIG),
               truth_sha256=sha_bytes(json.dumps(truth).encode()), n_direct=len(truth), n_regs=len(G["regs"]), n_cands=len(G["cands"]),
               driver_ids=driver_ids, drivers_observed=world == "C", generator_world=generator_world(world),
               loading_fro=float(np.linalg.norm(lam * L)), per_edge_abs_loading_mean=float(np.abs(L[L != 0]).mean()) if L.any() else 0.0,
               loading_sv=np.linalg.svd(lam * L, compute_uv=False)[:10].round(4).tolist() if L.size else [], hidden_info=hinfo,
               profile_geometry=profile_geometry(topo, meta, coefficients, world), hidden_coefficients_sha256=sha_bytes(np.ascontiguousarray(coefficients).tobytes()),
               hidden_end_rates=[sim.graph_[d]["rates"] for d in driver_ids], equation_fixture=None,
               observation=("CONC (continuous SERGIO concentrations; DIAGNOSTIC)" if obs == "CONC" else
                            f"{obs} (Poisson render of SERGIO concentrations x {obs_depth:g})"),
               status="OK")
    if world == "D":
        rec.update(thinning=dict(p=WORLD_D_P, subset_seed=888 + grn_seed, rng="PCG64([20260924,888+grn,sim,lambda_tag])",
                                 sources=thinned_sources, per_source=support_receipt,
                                 integer_counts=bool(np.array_equal(S, np.round(S)) and np.array_equal(U, np.round(U))),
                                 pre_pair_checks_pass=world_d_receipt_ok(support_receipt)))
        if not rec["thinning"]["integer_counts"] or not rec["thinning"]["pre_pair_checks_pass"]:
            rec["status"] = "NO_CALL_D_THINNING"
    if world == "AB": rec["ab"] = rec_ab
    rec["rng_post_sim_sha"] = LAST.get("sha"); rec["cell_sample_sha"] = sha_bytes(np.ascontiguousarray(meta["bin"]).tobytes() + np.ascontiguousarray(meta["step"]).tobytes() + np.ascontiguousarray(meta["path"]).tobytes()); rec["decoys"] = G["decoys"]
    np.savez_compressed(f"{dd}/raw/counts_truth.npz", spliced=S[:, :n], unspliced=U[:, :n], hidden_spliced=S[:, n:], hidden_unspliced=U[:, n:],
                        conc_spliced=Sc, conc_unspliced_latent=Ulat, cell_type=ct, gt=np.array(truth), loadings=L, hidden_coefficients=coefficients,
                        sampled_bin=meta["bin"], sampled_step=meta["step"], sampled_tau=meta["tau"], cands=np.array(G["cands"]),
                        driver_ids=np.asarray(driver_ids, int), thinned_source_ids=np.asarray(thinned_sources, int),
                        component_edge_mask=comp_mask, lam=lam, obs=np.array(obs), depth=np.array(np.nan if obs_depth is None else obs_depth))
    if rec["status"] != "OK":
        json.dump(rec, open(f"{dd}/record.json", "w"), indent=1, default=float); print(tag, rec["status"]); return rec
    a = make_adata(S[:, :n], U[:, :n], ct, obs=obs) if rec["status"] == "OK" else None
    try:
        a = preprocess(a, cat_names, exemption="SERGIO_UMI" if obs in ("UMI", "UMI_DEPTH") else None)
    except Exception as e:
        rec["status"] = f"NO_CALL_PREPROCESS: {str(e)[:200]}"
    if rec["status"] == "OK" and a.n_vars != n:
        rec["status"] = f"NO_CALL_AXIS: {n - a.n_vars} genes filtered (roster identity requires all {n})"
    if rec["status"] == "OK":
        Vr = np.asarray(a.layers["velocity"], float); rec["velocity_nonfinite_frac"] = float((~np.isfinite(Vr)).mean())
        if rec["velocity_nonfinite_frac"] > VEL_NONFINITE_MAX:
            rec["status"] = "NO_CALL_VELOCITY"
    if rec["status"] != "OK":
        json.dump(rec, open(f"{dd}/record.json", "w"), indent=1); print(tag, rec["status"]); return rec
    a.write_h5ad(f"{dd}/{tag}.h5ad")
    src_idx = list(G["cands"])                                               # axis == gene index (asserted above)
    Pg_cell, Ph_cell = production_parts(Ulat, edges, component_mask=comp_mask)
    fixed_geometry_pairs = None
    if tune_rec is not None and sim_seed == 0 and gtarget is not None and gtarget > 0:
        tf = tune_rec["geometry_fixtures"][f"{gtarget:.2f}"]
        replay = dict(latent_sha256=sha_bytes(np.ascontiguousarray(Ulat).tobytes()),
                      observed_sha256=sha_bytes(np.ascontiguousarray(S[:, :N_GENES]).tobytes() +
                                                np.ascontiguousarray(U[:, :N_GENES]).tobytes()),
                      design_sha256=hinfo["frozen_sha256"])
        rec["tuning_geometry_replay"] = dict(expected={k: tf[k] for k in replay}, observed=replay,
                                              PASS=all(replay[k] == tf[k] for k in replay))
        if not rec["tuning_geometry_replay"]["PASS"]:
            rec["status"] = "NO_CALL_TUNING_LATENT_REPLAY_MISMATCH"
            json.dump(rec, open(f"{dd}/record.json", "w"), indent=1, default=float); print(tag, rec["status"]); return rec
        Zt = np.load(tune_rec["_geometry_fixture_paths"][f"{gtarget:.2f}"], allow_pickle=False)
        fixed_geometry_pairs = {r: (Zt[f"pairs_{r}_I"], Zt[f"pairs_{r}_J"], Zt[f"pairs_{r}_pt"]) for r in ROOTS}
    geom, geometry_pairs = dataset_geometry(a, Ulat, edges, comp_mask, q, topo, world, lam, meta, coefficients,
                                            fixed_pairs=fixed_geometry_pairs)
    # Keep the tune/run simulation→preprocess→pair→geometry path side-effect identical. The simulator-equation
    # fixture is intentionally evaluated only after geometry has been frozen.
    eqfx = equation_fixture(sim); rec["equation_fixture"] = eqfx
    if not eqfx["PASS"]:
        rec["status"] = "NO_CALL_EQUATION"
        json.dump(rec, open(f"{dd}/record.json", "w"), indent=1, default=float); print(tag, rec["status"]); return rec
    if world == "AB":   # observed-edge half-responses actually used vs the frozen C0 values
        fz = PILOT.get("frozen_half") or {}
        dh = [abs(float(h) - fz[(int(t), int(r))]) for (t, r, K, c, h) in edges if (int(t), int(r)) in fz]
        rec["ab"]["frozen_half_max_abs_dev"] = max(dh) if dh else None; rec["ab"]["n_frozen_edges_checked"] = len(dh)
    if PILOT.get("gen_only"):   # generator-only calibration: no method scoring
        rec["geom"] = {r: geom[r] for r in ROOTS}; rec["G_E"] = float(np.mean([geom[r]["G_E"] for r in ROOTS])); rec["status"] = "GEN_ONLY_OK"
        np.savez_compressed(f"{dd}/raw/geometry_raw.npz", conc_unspliced_latent=Ulat, edges=typed_edges(edges), component_edge_mask=comp_mask,
                            sampled_bin=meta["bin"], sampled_tau=meta["tau"], **{f"pairs_{r}_{x}": v for r, (I, J, pt) in geometry_pairs.items() for x, v in (("I", I), ("J", J))})
        json.dump(rec, open(f"{dd}/record.json", "w"), indent=1, default=float); print(tag, "GEN_ONLY_OK G_E=%.3f" % rec["G_E"], flush=True); return rec
    raw, pairs, fx, krule, info_scores = score_dataset(a, src_idx, hidden_cell=Ph_cell, fixed_pairs=geometry_pairs)
    transient_cells = root_transient_cells(topo, meta)
    raw_root_excl, pairs_root_excl, fx_root_excl, krule_root_excl, info_root_excl = \
        score_dataset(a, src_idx, hidden_cell=Ph_cell, exclude_cells=transient_cells, fixed_pairs=geometry_pairs)
    rec["fixtures"] = fx
    rec["equal_c"] = {root: {arm: fx[f"{root}|{arm}|info"] for arm in ("EQC_hi", "EQC_lo")} for root in ROOTS}
    rec["root_transient_sensitivity"] = dict(n_cells_excluded=int(transient_cells.sum()), pair_counts={
        r: dict(primary=len(pairs[r][0]), root_transient_excluded=len(pairs_root_excl[r][0])) for r in ROOTS})
    if world == "D":
        support_receipt = source_support_receipt(S_render[:, :N_GENES], U_render[:, :N_GENES], S[:, :N_GENES], U[:, :N_GENES],
                                                 thinned_sources, pairs)
        rec["thinning"]["per_source"] = support_receipt
        rec["thinning"]["pair_support_checks_pass"] = world_d_receipt_ok(support_receipt, require_pairs=True)
        if not rec["thinning"]["pair_support_checks_pass"]:
            rec["status"] = "NO_CALL_D_THINNING"
    np.savez_compressed(f"{dd}/raw/krule.npz", **{f"{arm}|{r}|{nm}": v for (arm, r), o in krule.items() for nm, v in o.items()})
    np.savez_compressed(f"{dd}/raw/krule_root_transient_excluded.npz",
                        **{f"{arm}|{r}|{nm}": v for (arm, r), o in krule_root_excl.items() for nm, v in o.items()})
    rec["fixture_arms_pass"] = bool(all(v for k, v in fx.items() if k.endswith("k0_all_equal"))
                                   and max(v for k, v in fx.items() if k.endswith("k30_span_maxdiff")) <= 1e-4
                                   and max(v for k, v in fx.items() if k.endswith("orth")) <= 1e-10)
    # ROSTER (PI): candidates × all non-self genes; identical for every method; spec-v1 variant = non-candidate targets
    cols_pi = np.array([[t for t in range(n) if t != s] for s in src_idx])
    cols_v1 = np.array([[t for t in range(n) if t not in set(src_idx)] for _ in src_idx])
    roster = [(s, int(t)) for i, s in enumerate(src_idx) for t in cols_pi[i]]
    rec["roster_sha256"] = sha_bytes(json.dumps(roster).encode())
    nonfin = sum(int((~np.isfinite(np.take_along_axis(o["absS"][k], cols_pi, 1))).sum()) for (arm, ch, r), o in raw.items()
                 if ch in ("U", "X") and not arm.startswith("EQC") for k in range(KMAX + 1))
    if nonfin:
        rec["status"] = f"NO_CALL_SUPPORT: {nonfin} non-finite U/X roster scores"; json.dump(rec, open(f"{dd}/record.json", "w"), indent=1); return rec
    cl = closure_bfs(truth_set, n)
    lab = lambda cols: (np.array([[(s, int(t)) in truth_set for t in cols[i]] for i, s in enumerate(src_idx)]),
                        np.array([["direct" if (s, int(t)) in truth_set else "indirect" if (s, int(t)) in cl else "none" for t in cols[i]]
                                  for i, s in enumerate(src_idx)]))
    true_rows = [i for i, s in enumerate(src_idx) if s in G["regs"]]
    # G_E (internal review ruling a) from SERGIO's UNSPLICED production equation on the latent unspliced state, per root; certification
    Ms = np.asarray(a.layers["Ms"], float); lib = Sc[:, :n].sum(1, keepdims=True)
    Hraw = Sc[:, driver_ids] if world == "C" else Sc[:, n:]
    Hn = Hraw / np.maximum(lib, 1e-9) * np.median(lib)
    fold = np.random.RandomState(4242 + grn_seed).randint(0, 2, len(S)); g_ok = []
    rec["profile_geometry"] = geom["profile_geometry"]
    rec["rank_gate_applies"] = geom["rank_gate_applies"]
    rec["alignment_gate_applies"] = geom["alignment_gate_applies"]
    if world == "C":
        rec["falsifier_geometry"] = geom["falsifier_geometry"]
    for r in ROOTS:
        I, J, _ = pairs[r]; gs = geom[r]
        rec[f"geometry_{r}"] = gs
        rec[f"G_E_{r}"] = gs["G_E"]
        rec[f"G_proxy_{r}"] = float(g_crossfit(Ms[J] - Ms[I], Hn[J] - Hn[I], I, J, fold)) if Hn.shape[1] else float("nan")
        cvel = ((Ms[J] - Ms[I]) * Vr[I]).sum(1) / np.maximum(np.linalg.norm(Ms[J] - Ms[I], axis=1) * np.linalg.norm(Vr[I], axis=1), EPS)
        rec[f"vel_coherence_{r}"] = float(np.median(cvel)); rec[f"n_pairs_{r}"] = int(len(I))
        g_ok.append(gtarget is None or abs(gs["G_E"] - gtarget) <= 0.03)
    rec["G_E"] = float(np.mean([rec[f"G_E_{r}"] for r in ROOTS]))
    rec["leading_prefix_energy_fraction"] = {r: rec[f"geometry_{r}"]["leading_prefix_energy_fraction"] for r in ROOTS}
    rec["driver_pair_energy_prefix3"] = {r: rec[f"geometry_{r}"]["driver_pair_energy_prefix3"] for r in ROOTS}
    rec["certification"] = dict(G_E_ok=bool(all(g_ok)), G_E_gate_applies=bool(gtarget is not None and world not in PLIKE),
                                alignment_gate_applies=geom["alignment_gate_applies"], rank_gate_applies=geom["rank_gate_applies"],
                                **geom["certification"],
                                observation_support_ok=(rec.get("thinning", {}).get("pair_support_checks_pass", True)))
    if tune_rec is not None and gtarget is not None and gtarget > 0:
        expected_g = tuning_trace_geometry(tune_rec, gtarget, lam)
        observed_g = np.asarray([rec[f"G_E_{r}"] for r in ROOTS])
        gd = float(np.max(np.abs(expected_g - observed_g)))
        rec["fixture_tune_run_geometry"] = dict(PASS=gd <= 1e-12, max_abs_diff=gd,
                                                 tuning_G_E=expected_g.tolist(), run_one_G_E=observed_g.tolist())
        if gd > 1e-12 and rec["status"] == "OK":
            rec["status"] = "NO_CALL_TUNE_RUN_GEOMETRY_MISMATCH"
    rec["certified"] = bool(rec["certification"]["observation_support_ok"]
                             and (not rec["certification"]["alignment_gate_applies"] or rec["certification"]["alignment_ok"])
                             and (not rec["certification"]["rank_gate_applies"] or
                                  (rec["certification"]["pair_rank_ok"] and rec["certification"]["cell_rank_ok"]))
                             and (not rec["certification"]["G_E_gate_applies"] or rec["certification"]["G_E_ok"]))
    if rec["status"] == "OK" and rec["certification"]["alignment_gate_applies"] and not rec["certification"]["alignment_ok"]:
        rec["status"] = "NO_CALL_ALIGNMENT"
    elif rec["status"] == "OK" and rec["certification"]["rank_gate_applies"] and \
            (not rec["certification"]["pair_rank_ok"] or not rec["certification"]["cell_rank_ok"]):
        rec["status"] = "NO_CALL_RANK"
    elif rec["status"] == "OK" and rec["certification"]["G_E_gate_applies"] and not rec["certification"]["G_E_ok"]:
        rec["status"] = "NO_CALL_G_TARGET"
    by = lambda arm, ch: {r: raw[(arm, ch, r)]["absS"] for r in ROOTS}
    by_sens = lambda arm, ch: {r: raw_root_excl[(arm, ch, r)]["absS"] for r in ROOTS}
    by_info = lambda name: {r: info_scores[r][name] for r in ROOTS}
    by_info_sens = lambda name: {r: info_root_excl[r][name] for r in ROOTS}
    rows, perk, mats = [], [], {}
    for univ, cols in (("PI", cols_pi), ("specv1", cols_v1)):
        Y, CLS = lab(cols)
        if univ == "PI":
            rec["prevalence"] = float(Y.mean()); rec["n_direct_roster"] = int(Y.sum())
            RU = production_scalar(by("TRUE", "U"), cols)
            rec["prod_scalar_ref_maxdiff"] = float(np.abs(RU - reference_scalar(by("TRUE", "U"), cols)).max())
        def add(method, M, analysis="primary", **kw):
            if univ == "PI" and analysis == "primary":
                mats[method] = M
            rows.append(dict(rec_key=tag, univ=univ, analysis=analysis, method=method, **kw, **endpoints(M, Y, CLS, true_rows)))
        for arm in ARMS:
            for ch in CHANS:
                Mprod = production_scalar(by(arm, ch), cols)
                if arm.startswith("EQC") and not np.isfinite(Mprod).all():
                    rows.append(dict(rec_key=tag, univ=univ, analysis="primary", method=f"R_{ch}_{arm}", arm=arm, chan=ch, status="NO_CALL_EQUAL_C",
                                     **{k: np.nan for k in endpoints(np.ones((2, 2)), np.array([[1, 0], [0, 1]], bool),
                                                                    np.array([["direct", "none"], ["none", "direct"]]), [0, 1])}))
                else:
                    add(f"R_{ch}_{arm}", Mprod, arm=arm, chan=ch, status="OK")
        for ch in CHANS:
            add(f"{ch}_k0", np.minimum(*[pct_within_source(np.take_along_axis(raw[("TRUE", ch, r)]["absS"][0], cols, 1)) for r in ROOTS]), chan=ch)
        for nm, label in (("matched", "INFO_MATCHED_U"), ("linear", "INFO_LINEAR"), ("spline", "INFO_SPLINE")):
            add(label, production_scalar(by_info(nm), cols), chan="U", comparator="information_matched")
        if univ == "PI":
            add("R_U_TRUE", production_scalar(by_sens("TRUE", "U"), cols), analysis="root_transient_excluded", arm="TRUE", chan="U", status="OK")
            add("R_X_TRUE", production_scalar(by_sens("TRUE", "X"), cols), analysis="root_transient_excluded", arm="TRUE", chan="X", status="OK")
            add("U_k0", np.minimum(*[pct_within_source(np.take_along_axis(raw_root_excl[("TRUE", "U", r)]["absS"][0], cols, 1)) for r in ROOTS]),
                analysis="root_transient_excluded", chan="U")
            for nm, label in (("matched", "INFO_MATCHED_U"), ("linear", "INFO_LINEAR"), ("spline", "INFO_SPLINE")):
                add(label, production_scalar(by_info_sens(nm), cols), analysis="root_transient_excluded", chan="U",
                    comparator="information_matched")
        for arm in ARMS:
            for k in range(KMAX + 1):
                Mk = np.minimum(*[pct_within_source(np.take_along_axis(raw[(arm, "U", r)]["absS"][k], cols, 1)) for r in ROOTS])
                if arm.startswith("EQC") and not np.isfinite(Mk).all():
                    perk.append(dict(rec_key=tag, univ=univ, analysis="primary", arm=arm, k=k, status="NO_CALL_EQUAL_C"))
                else:
                    perk.append(dict(rec_key=tag, univ=univ, analysis="primary", arm=arm, k=k, status="OK", **endpoints(Mk, Y, CLS, true_rows)))
        Z = (Ms - Ms.mean(0)) / np.maximum(Ms.std(0), EPS)
        add("Pearson_abs", np.take_along_axis(np.abs(Z[:, src_idx].T @ Z / len(Z)), cols, 1))
        arb = {}
        if univ == "PI" and client is not None:   # pilot: client=None skips the arboreto comparators
            import gb_sota_run as SR
            Lx = np.asarray(a.X, float); gn = names; arb = {}
            for lr in ("grnboost2", "genie3"):
                Lk = SR.arboreto_fit(Lx, gn, [gn[s] for s in src_idx], gn, lr, 0, client)
                M = np.zeros((len(src_idx), n)); ri = {gn[s]: i for i, s in enumerate(src_idx)}
                for tf, tg_, w in zip(Lk.TF, Lk.target, Lk.importance):
                    M[ri[tf], int(tg_.split("_")[1])] = w
                arb[lr] = M
        for lr, M in arb.items():
            add(lr.upper(), np.take_along_axis(M, cols, 1))
    perk = pd.DataFrame(perk)
    pkP = perk[perk.univ == "PI"]; tr_ = pkP[pkP.arm == "TRUE"]; rec["oracle_k_TRUE"] = int(tr_.loc[tr_.AP_pct.idxmax(), "k"])
    band = pkP[pkP.k.isin(BAND)].groupby("arm").nAP_pct.mean(); k0 = float(tr_[tr_.k == 0].nAP_pct.iloc[0])
    eqc_band_ok = {arm: all(np.asarray(rec["equal_c"][root][arm]["valid"], bool)[BAND].all() for root in ROOTS)
                   for arm in ("EQC_hi", "EQC_lo")}
    rec["mech"] = dict(TRUE=band["TRUE"], REV=band["REV"], RIS_mean=float(np.mean([band[f"RIS{s}"] for s in range(N_DRAW)])),
                       ROS_mean=float(np.mean([band[f"ROS{s}"] for s in range(N_DRAW)])),
                       EQC_hi=float(band.get("EQC_hi", np.nan)) if eqc_band_ok["EQC_hi"] else np.nan,
                       EQC_lo=float(band.get("EQC_lo", np.nan)) if eqc_band_ok["EQC_lo"] else np.nan, k0=k0)
    rec["mech"].update({f"TRUE_minus_{c}": rec["mech"]["TRUE"] - rec["mech"][c]
                        for c in ("REV", "RIS_mean", "ROS_mean", "EQC_hi", "EQC_lo", "k0")})
    Y, CLS = lab(cols_pi); rw = np.random.Generator(np.random.PCG64(55 + grn_seed)); nulls = []; dfx = []
    if os.environ.get("GB_PILOT_FAST") == "1":  # skip legacy null_rewire / null_Udestroy (unused by pilot; do not feed raw/cands/metrics)
        rec["rewire_degree_fixture"] = dict(PASS=True, skipped=True); rec["null_rewire"] = None; rec["null_Udestroy"] = None; rec["null_Udestroy_prod_nAP"] = None
    else:
        U0 = mats["U_k0"]
        for _ in range(5):
            Yp = rewire_double_swap(Y, cols_pi, src_idx, rw); dfx.append(degree_fixture(Y, Yp, cols_pi))
            tb = np.mean([nap_pct(np.minimum(*[pct_within_source(np.take_along_axis(raw[("TRUE", "U", r)]["absS"][k], cols_pi, 1)) for r in ROOTS]), Yp) for k in BAND])
            nulls.append(dict(prod=nap_pct(RU, Yp), band=tb, k0=nap_pct(U0, Yp),
                              prod_macro=macro_nap(RU, Yp), k0_macro=macro_nap(U0, Yp),
                              prod_minus_k0_macro=macro_nap(RU, Yp) - macro_nap(U0, Yp)))
        rec["rewire_degree_fixture"] = dict(PASS=all(d["out_deg"] and d["in_deg"] and d["n_edges"] for d in dfx), draws=dfx)
        rec["null_rewire"] = dict(draws=nulls, **{f"{k}_mean": float(np.mean([x[k] for x in nulls])) for k in nulls[0]})
        rawd, _, _ = score_dataset(a, src_idx, perm_u=7)
        Udest_by = {r: rawd[("TRUE", "U", r)]["absS"] for r in ROOTS}; Udest_prod = production_scalar(Udest_by, cols_pi)
        Udest_k0 = np.minimum(*[pct_within_source(np.take_along_axis(Udest_by[r][0], cols_pi, 1)) for r in ROOTS])
        Udest_band = float(np.mean([nap_pct(np.minimum(*[pct_within_source(np.take_along_axis(Udest_by[r][k], cols_pi, 1))
                                                          for r in ROOTS]), Y) for k in BAND]))
        rec["null_Udestroy"] = dict(prod_nAP=nap_pct(Udest_prod, Y), band_nAP=Udest_band, k0_nAP=nap_pct(Udest_k0, Y),
                                     prod_macro=macro_nap(Udest_prod, Y), k0_macro=macro_nap(Udest_k0, Y),
                                     prod_minus_k0_macro=macro_nap(Udest_prod, Y) - macro_nap(Udest_k0, Y))
        rec["null_Udestroy_prod_nAP"] = rec["null_Udestroy"]["prod_nAP"]
    # RAW persistence + score-persistence fixture
    np.savez_compressed(f"{dd}/raw/geobridge_raw.npz", source_gene_index=np.array(src_idx), cols_PI=cols_pi, cols_specv1=cols_v1,
                        **{f"{arm}|{ch}|{r}|{nm}": v for (arm, ch, r), o in raw.items() for nm, v in o.items()},
                        **{f"info|{r}|{nm}": v for r, o in info_scores.items() for nm, v in o.items()},
                        **{f"root_excl|{arm}|{ch}|{r}|{nm}": v for (arm, ch, r), o in raw_root_excl.items() for nm, v in o.items()},
                        **{f"root_excl_info|{r}|{nm}": v for r, o in info_root_excl.items() for nm, v in o.items()},
                        **{f"pairs|{r}|{x}": v for r, (I, J, pt) in pairs.items() for x, v in (("I", I), ("J", J), ("pt", pt))},
                        **{f"root_excl_pairs|{r}|{x}": v for r, (I, J, pt) in pairs_root_excl.items() for x, v in (("I", I), ("J", J), ("pt", pt))})
    np.savez_compressed(f"{dd}/raw/geometry_raw.npz", conc_unspliced_latent=Ulat, edges=typed_edges(edges), component_edge_mask=comp_mask,
                        sampled_bin=meta["bin"], sampled_step=meta["step"], sampled_tau=meta["tau"], hidden_coefficients=coefficients,
                        driver_ids=np.asarray(driver_ids, int), production_pca30=np.asarray(a.obsm["X_pca"], float)[:, :KMAX],
                        q=np.array(q), G_target=np.array(np.nan if gtarget is None else gtarget), topo=np.array(topo), world=np.array(world),
                        obs=np.array(obs), depth=np.array(np.nan if obs_depth is None else obs_depth),
                        **{f"pairs_{r}_{x}": v for r, (I, J, pt) in pairs.items() for x, v in (("I", I), ("J", J))})
    rec["geometry_raw_sha256"] = sha_file(f"{dd}/raw/geometry_raw.npz")
    np.savez_compressed(f"{dd}/raw/comparators.npz", **{m: v for m, v in mats.items() if not m.startswith("R_")})
    sv = np.repeat(src_idx, n - 1); tv = cols_pi.ravel(); lb = CLS.ravel(); tid = []
    for (arm, ch, r), o in raw.items():
        for k in range(KMAX + 1):
            tid.append(pd.DataFrame(dict(analysis="primary", method=f"GB_{ch}", arm=arm, root=r, k=k, source=sv, target=tv,
                                         score=np.take_along_axis(o["absS"][k], cols_pi, 1).ravel(), label=lb)))
    for analysis, imo in (("primary", info_scores), ("root_transient_excluded", info_root_excl)):
        for r, o in imo.items():
            for nm, scores_ in o.items():
                for k in range(KMAX + 1):
                    tid.append(pd.DataFrame(dict(analysis=analysis, method=f"INFO_{nm.upper()}", arm="TRUE", root=r, k=k,
                                                 source=sv, target=tv, score=np.take_along_axis(scores_[k], cols_pi, 1).ravel(), label=lb)))
    for m, M in mats.items():
        tid.append(pd.DataFrame(dict(analysis="primary", method=m, arm="", root="", k=-1, source=sv, target=tv, score=M.ravel(), label=lb)))
    T = pd.concat(tid, ignore_index=True); T.insert(0, "dataset", tag); T.to_parquet(f"{dd}/raw/tidy.parquet", index=False)
    Z_ = np.load(f"{dd}/raw/geobridge_raw.npz")
    rel = {r: np.stack([Z_[f"TRUE|U|{r}|absS"][k] for k in range(KMAX + 1)]) for r in ROOTS}
    rec["fixture_persistence_maxdiff"] = float(np.abs(production_scalar(rel, cols_pi) - RU).max())
    rec["fixture_info_persistence_maxdiff"] = max(
        [maxdiff_nan(Z_[f"info|{r}|{nm}"], info_scores[r][nm]) for r in ROOTS for nm in ("linear", "spline", "matched")] +
        [maxdiff_nan(Z_[f"root_excl_info|{r}|{nm}"], info_root_excl[r][nm]) for r in ROOTS for nm in ("linear", "spline", "matched")])
    Tt = pd.read_parquet(f"{dd}/raw/tidy.parquet"); sub = Tt[(Tt.method == "Pearson_abs")]
    rec["fixture_persistence_tidy_AP"] = abs(ap(sub.label.eq("direct"), sub.score) - [r_ for r_ in rows if r_["method"] == "Pearson_abs" and r_["univ"] == "PI"][0]["AP_native"])
    rec["fixture_failures"] = fixture_statuses(rec["fixture_arms_pass"], rec["prod_scalar_ref_maxdiff"],
                                                rec["rewire_degree_fixture"]["PASS"],
                                                max(rec["fixture_persistence_maxdiff"], rec["fixture_info_persistence_maxdiff"]),
                                                rec["fixture_persistence_tidy_AP"])
    if rec["fixture_failures"] and rec["status"] == "OK":
        rec["status"] = rec["fixture_failures"][0]
    rec["tidy_sha256"] = sha_file(f"{dd}/raw/tidy.parquet"); rec["wall_s"] = round(time.time() - t0)
    json.dump(dict(roster=roster, sources=src_idx, cols_PI=cols_pi.tolist(), cols_specv1=cols_v1.tolist()), open(f"{dd}/roster.json", "w"))
    json.dump(dict(cands=src_idx, n=n), open(f"{dd}/cands.json", "w"))           # consumed by gb_sergio_velorama_v2.run (official defaults)
    pd.DataFrame(rows).to_csv(f"{dd}/metrics.tsv", sep="\t", index=False); perk.to_csv(f"{dd}/perk.tsv", sep="\t", index=False)
    json.dump(rec, open(f"{dd}/record.json", "w"), indent=1, default=float)
    print(f"{tag}: λ={lam:.3g} G_E={rec['G_E_dc1_min']:.3f}/{rec['G_E_dc1_max']:.3f} certified={rec['certified']} eq={eqfx['PASS']} proxy={rec['G_proxy_dc1_min']:.3f} arms={rec['fixture_arms_pass']} "
          f"ref={rec['prod_scalar_ref_maxdiff']:.1e} persist={rec['fixture_persistence_maxdiff']:.1e} prev={rec['prevalence']:.4f} {rec['wall_s']}s", flush=True)
    return rec


# ───────────────────────── λ tuning (generator-side only; no scores, no truth) ─────────────────────────
def measure_geometry(topo, world, q, lam, grn_seed, sim_seed, workdir, orientation_dir=None, obs="UMI", depth=10.0):
    """simulate → selected observation → canonical preprocess → production pairs → per-root geometry descriptors."""
    generator_world(world); G = generate_grn(grn_seed)
    orientation_dir = orientation_dir or f"{os.path.dirname(workdir.rstrip('/'))}/orientation"
    L, coefficients, q_real, info = hidden_design(G, grn_seed, world, q, topo, orientation_dir,
                                                   f"{orientation_dir}_work/g{grn_seed}_{topo}_{world}_q{q}")
    if coefficients is None or info.get("status") != "OK":
        return dict(status=info.get("status", "NO_CALL_ORIENTATION"), info=info, q_real=q_real)
    driver_ids = world_c_drivers(G, q) if world == "C" else list(range(N_GENES, N_GENES + q))
    Sc, Uc_, ct, sim, Ulat, meta = simulate(topo, G, L, coefficients, lam, grn_seed, sim_seed, workdir, world, driver_ids)
    S, U, thinned_sources, S0, U0 = render_observation(Sc, Uc_, G, world, grn_seed, sim_seed, lam, obs, depth)
    a = preprocess(make_adata(S[:, :N_GENES], U[:, :N_GENES], ct, obs=obs), [f"gene_{c}" for c in G["cands"]],
                   exemption="SERGIO_UMI" if obs in ("UMI", "UMI_DEPTH") else None)
    V = np.asarray(a.layers["velocity"], float)
    if not np.isfinite(V).all():
        return dict(status="NO_CALL_VELOCITY", info=info)
    edges = edge_table(sim); comp_mask = component_edge_mask(edges, world, L, lam, driver_ids if world == "C" else None)
    out, pairs = dataset_geometry(a, Ulat, edges, comp_mask, q, topo, world, lam, meta, coefficients)
    design = dict(path=info["frozen_path"], sha256=info["frozen_sha256"],
                  context_sha256=json.load(open(info["frozen_path"]))["context_sha256"])
    out.update(status="OK", info=info, q_real=q_real, driver_ids=driver_ids, obs=obs, depth=effective_depth(obs, depth),
               latent_sha256=sha_bytes(np.ascontiguousarray(Ulat).tobytes()),
               observed_sha256=sha_bytes(np.ascontiguousarray(S[:, :N_GENES]).tobytes() + np.ascontiguousarray(U[:, :N_GENES]).tobytes()),
               design=design)
    if world not in PLIKE:
        out["orientation"] = design
    out["_pairs"] = pairs
    if world == "D":
        pair_ij = {r: pairs[r][:2] for r in ROOTS}
        receipt = source_support_receipt(S0[:, :N_GENES], U0[:, :N_GENES], S[:, :N_GENES], U[:, :N_GENES], thinned_sources, pair_ij)
        out["thinning"] = dict(p=WORLD_D_P, sources=thinned_sources, per_source=receipt,
                               certified=world_d_receipt_ok(receipt, require_pairs=True))
        if not out["thinning"]["certified"]: out["status"] = "NO_CALL_D_THINNING"
    out["certification"]["observation_support_ok"] = out.get("thinning", {}).get("certified", True)
    if out["status"] == "OK" and out["alignment_gate_applies"] and not out["certification"]["alignment_ok"]:
        out["status"] = "NO_CALL_ALIGNMENT"
    elif out["status"] == "OK" and out["rank_gate_applies"] and \
            (not out["certification"]["pair_rank_ok"] or not out["certification"]["cell_rank_ok"]):
        out["status"] = "NO_CALL_RANK"
    return out


def verify_dataset(dd, atol=1e-10):
    """Independent geometry verification: reads only raw/geometry_raw.npz plus record.json; no scores and no truth labels."""
    p = f"{dd}/raw/geometry_raw.npz"; Z = np.load(p, allow_pickle=False); rec = json.load(open(f"{dd}/record.json"))
    topo, world, q = str(Z["topo"].item()), str(Z["world"].item()), int(Z["q"].item()); Ulat = Z["conc_unspliced_latent"]; edges = Z["edges"]
    comp_mask = Z["component_edge_mask"]
    edge_dtype_ok = bool(edges.dtype.fields and edges.dtype["target"].kind in "iu" and edges.dtype["regulator"].kind in "iu")
    if not edge_dtype_ok:
        raise RuntimeError(f"REFUSING_GEOMETRY_EDGE_DTYPE: {edges.dtype}")
    meta = dict(bin=Z["sampled_bin"], step=Z["sampled_step"], tau=Z["sampled_tau"]); coef = Z["hidden_coefficients"]
    got = dict(profile_geometry=profile_geometry(topo, meta, coef, world), roots={})
    diffs = []
    for root in ROOTS:
        I, J = Z[f"pairs_{root}_I"], Z[f"pairs_{root}_J"]
        got["roots"][root], dH = geometry_stats(Ulat, edges, I, J, q, comp_mask)
        dP = Z["production_pca30"][J] - Z["production_pca30"][I]; Qp = np.linalg.qr(dP)[0][:, :min(3, dP.shape[1])]
        got["roots"][root]["driver_pair_energy_prefix3"] = float(np.linalg.norm(Qp.T @ dH, "fro") ** 2 /
                                                                   np.maximum(np.linalg.norm(dH, "fro") ** 2, EPS))
        for key in ("G_E", "cross_norm", "r_eff_hidden", "cos_max", "cos_min", "driver_pair_energy_prefix3"):
            a, b = got["roots"][root].get(key, np.nan), rec[f"geometry_{root}"].get(key, np.nan)
            diffs.append(abs(a - b) if np.isfinite(a) and np.isfinite(b) else (0.0 if np.isnan(a) and np.isnan(b) else np.inf))
    for key in ("cell_profile_rank", "cell_profile_mean_abs_max", "cell_profile_orth_max", "cell_profile_corr_offdiag_max", "max_abs_profile"):
        diffs.append(abs(got["profile_geometry"][key] - rec["profile_geometry"][key]))
    transient = root_transient_cells(topo, meta); got["root_transient_sensitivity"] = dict(n_cells_excluded=int(transient.sum()), pair_counts={})
    for root in ROOTS:
        I, J = Z[f"pairs_{root}_I"], Z[f"pairs_{root}_J"]
        got["root_transient_sensitivity"]["pair_counts"][root] = dict(primary=len(I), root_transient_excluded=int((~(transient[I] | transient[J])).sum()))
    diffs.append(0.0 if got["root_transient_sensitivity"] == rec.get("root_transient_sensitivity") else np.inf)
    got.update(raw_sha256=sha_file(p), record_raw_sha256=rec.get("geometry_raw_sha256"), max_record_diff=float(max(diffs)),
               edge_dtype_ok=edge_dtype_ok,
               PASS=bool(edge_dtype_ok and sha_file(p) == rec.get("geometry_raw_sha256") and max(diffs) <= atol))
    print(json.dumps(got, indent=1, default=float)); return got


def tune(out, topo, world, q, grn_seed=0, targets=None, tol=0.03, max_iter=8, obs="UMI", depth=10.0):
    """per (topology, world, q, GRN): bisection on log λ so that G_E is within ±tol of each target in BOTH roots (sim seed 0).
    Generator-side quantity only (no GeoBRIDGE score, no truth labels). Unreached targets are recorded as null."""
    generator_world(world)
    targets = ((.15, .30, .45, .60, .75) if world in PLIKE else (.15, .30)) if targets is None else tuple(targets)
    table, achieved, trace, design, geometry_fixtures = {"0.00": 0.0}, {"0.00": [0.0, 0.0]}, [], None, {}
    for gt_ in targets:
        lo, hi, best = 1e-3, 20.0, None
        for it in range(max_iter):
            lam = float(np.sqrt(lo * hi)); m = measure_geometry(
                topo, world, q, lam, grn_seed, 0,
                f"{out}/tune/{topo}_{world}_q{q}_{observation_tag(obs, depth)}_g{grn_seed}/lam{lam:.5g}",
                f"{out}/orientation", obs, depth)
            # certification gates (alignment/rank) are properties of the FINAL tuned dataset (enforced again in run_one); at
            # intermediate λ with negligible hidden drive they are undefined, so the G_E bisection continues through them
            search_ok = m["status"] in ("OK", "NO_CALL_ALIGNMENT", "NO_CALL_RANK") and all(r in m and "G_E" in m[r] for r in ROOTS)
            if not search_ok:
                trace.append(dict(target=gt_, lam=lam, status=m["status"])); break
            if design is None:
                design = m["design"]
            elif design["sha256"] != m["design"]["sha256"]:
                raise RuntimeError("REFUSING_TUNING_DESIGN_CHANGED_WITHIN_SEARCH")
            g = [m[r]["G_E"] for r in ROOTS]; trace.append(dict(target=gt_, lam=lam, G_E=g, status=m["status"])); print(trace[-1], flush=True)
            if all(abs(x - gt_) <= tol for x in g):
                os.makedirs(f"{out}/tune_geometry", exist_ok=True)
                fp = f"{out}/tune_geometry/{topo}_{world}_q{q}_{observation_tag(obs, depth)}_g{grn_seed}_G{gt_:.2f}.npz"
                np.savez_compressed(fp, latent_sha256=np.array(m["latent_sha256"]), observed_sha256=np.array(m["observed_sha256"]),
                                    design_sha256=np.array(m["design"]["sha256"]), G_E=np.asarray(g, float), lam=np.array(lam),
                                    **{f"pairs_{r}_{x}": v for r, pair in m["_pairs"].items()
                                       for x, v in zip(("I", "J", "pt"), pair)})
                geometry_fixtures[f"{gt_:.2f}"] = dict(relative_path=os.path.relpath(fp, out), sha256=sha_file(fp),
                                                         latent_sha256=m["latent_sha256"], observed_sha256=m["observed_sha256"],
                                                         design_sha256=m["design"]["sha256"], G_E=g, lam=lam)
                best = lam; achieved[f"{gt_:.2f}"] = g; break
            lo, hi = (lam, hi) if np.mean(g) < gt_ else (lo, lam)
        table[f"{gt_:.2f}"] = best
        if best is None:
            achieved[f"{gt_:.2f}"] = None
    if design is None:
        raise RuntimeError("REFUSING_TUNING_WITHOUT_DESIGN_RECEIPT")
    design_rec = dict(relative_path=os.path.relpath(design["path"], out), sha256=design["sha256"],
                      context_sha256=design["context_sha256"])
    rec = dict(topo=topo, world=world, q=q, grn_seed=grn_seed, sim_seed=0, obs=obs, depth=effective_depth(obs, depth), tol=tol,
               lambda_for_G=table, achieved_G_E=achieved, trace=trace, geometry_fixtures=geometry_fixtures,
               script_sha256=sha_file(os.path.abspath(__file__)))
    rec["programme" if world in PLIKE else "orientation"] = design_rec
    p = tuning_json_path(out, topo, world, q, grn_seed, obs, depth); json.dump(rec, open(p, "w"), indent=1)
    print(dict(lambda_for_G=table, achieved_G_E=achieved)); return p


# ───────────────────────── fixtures (no GeoBRIDGE scores) ─────────────────────────
def fixtures(out, part="core", topo=None):
    os.makedirs(out, exist_ok=True); res = {}
    if part == "core":
        Gs = {g: generate_grn(g) for g in (0, 1, 2)}
        res["direct_count_identity"] = all(len(G["gt"]) == N_DIRECT for G in Gs.values())
        res["candidate_count_identity"] = all(len(G["cands"]) == N_MR + N_INT + N_DECOY and len(G["regs"]) == N_MR + N_INT for G in Gs.values())
        res["prevalence_identity"] = len({N_DIRECT / (len(G["cands"]) * (N_GENES - 1)) for G in Gs.values()}) == 1
        lec = {g: dict(mr_int=sum(s_ in G["mrs"] and t in G["ints"] for s_, t in G["gt"]), int_int=sum(s_ in G["ints"] and t in G["ints"] for s_, t in G["gt"]),
                       mr_leaf=sum(s_ in G["mrs"] and t in G["leaves"] for s_, t in G["gt"]), int_leaf=sum(s_ in G["ints"] and t in G["leaves"] for s_, t in G["gt"]))
               for g, G in Gs.items()}
        res["layer_edge_counts"] = lec; res["layer_edge_identity"] = len({json.dumps(v, sort_keys=True) for v in lec.values()}) == 1
        res["grn_identity_within"] = generate_grn(1)["gt"] == Gs[1]["gt"] and generate_grn(1)["K"] == Gs[1]["K"]
        res["grn_difference_across"] = len({frozenset(G["gt"]) for G in Gs.values()}) == 3
        res["closure_bfs_eq_matrix"] = all(closure_bfs(G["gt"], N_GENES) == closure_matrix(G["gt"], N_GENES) for G in Gs.values())
        # degree-preserving rewiring on the real roster of GRN 0
        G = Gs[0]; src = G["cands"]; cols = np.array([[t for t in range(N_GENES) if t != s_] for s_ in src])
        Y = np.array([[(s_, int(t)) in G["gt"] for t in cols[i]] for i, s_ in enumerate(src)])
        rw = np.random.Generator(np.random.PCG64(55)); dfx = [degree_fixture(Y, rewire_double_swap(Y, cols, src, rw), cols) for _ in range(5)]
        res["rewire_degree"] = dict(PASS=all(d["out_deg"] and d["in_deg"] and d["n_edges"] for d in dfx), changed_frac=[d["changed_frac"] for d in dfx])
        # simulator: determinism, equation fixture (act & rep), G_E(λ=0) = 0, UMI integer + parity on the exact UMI data
        wd = f"{out}/sim"; L, coefficients, _, hi = hidden_design(G, 0, "A", 3, "bifurcation", f"{out}/orientation", f"{wd}/orient")  # pilot world (B = NO_CALL_ALIGNMENT, §193)
        res["orientation_status"] = hi.get("status")
        S0, U0, ct, sim0, Ul0, _ = simulate("bifurcation", G, L, coefficients, 0.0, 0, 0, f"{wd}/lam0")
        S0b, *_ = simulate("bifurcation", G, L, coefficients, 0.0, 0, 0, f"{wd}/lam0b"); res["sim_determinism_maxdiff"] = float(np.abs(S0 - S0b).max())
        S1, U1, ct1, sim1, Ul1, _ = simulate("bifurcation", G, L, coefficients, 1.0, 0, 0, f"{wd}/lam1")
        res["equation_fixture_lam1"] = equation_fixture(sim1); res["equation_fixture_lam0"] = equation_fixture(sim0)
        I = np.random.default_rng(0).integers(0, len(S0), 5000); J = np.random.default_rng(1).integers(0, len(S0), 5000)
        res["G_E_lambda0"] = geometry_stats(Ul0, edge_table(sim0), I, J)[0]["G_E"]
        Su, Uu = render_umi(S1, U1, 0, 0, 1.0); res["umi_integer"] = bool(np.array_equal(Su, np.round(Su)) and np.array_equal(Uu, np.round(Uu)))
        res["umi_render_determinism"] = bool(np.array_equal(Su, render_umi(S1, U1, 0, 0, 1.0)[0]))
        res["parity"] = parity_fixture(Su[:, :N_GENES], Uu[:, :N_GENES], ct1, [f"gene_{c}" for c in G["cands"]], f"{out}/parity")
        # World P uses the exact native per-bin pilot programme. Exercise its own simulation and the requested depth-rendered parity.
        Lp, ratesp, _, hip = hidden_design(G, 0, "P", 3, "bifurcation", f"{out}/orientation", f"{wd}/orientP")
        Sp, Up, ctp, simp, Ulp, _ = simulate("bifurcation", G, Lp, ratesp, 1.0, 0, 0, f"{wd}/P_lam1", "P")
        Sdp, Udp, *_ = render_observation(Sp, Up, G, "P", 0, 0, 1.0, "UMI_DEPTH", 10)
        Sdp2, Udp2, *_ = render_observation(Sp, Up, G, "P", 0, 0, 1.0, "UMI_DEPTH", 10)
        res["P_bookkeeping"] = dict(
            truth_excludes_hidden=truth_for_world(G, "P", Lp, list(range(N_GENES, N_GENES + 3))) == G["gt"],
            loading_reproducible=np.array_equal(Lp, hidden_loadings(G, 0, "P", 3)),
            profiles_exact=np.array_equal(ratesp, profiles("bifurcation", 3)),
            native_rates_exact=np.array_equal(np.asarray([simp.graph_[N_GENES + h]["rates"] for h in range(3)]), ratesp),
            depth_render_deterministic=np.array_equal(Sdp, Sdp2) and np.array_equal(Udp, Udp2),
            depth_render_integer=np.array_equal(Sdp, np.round(Sdp)) and np.array_equal(Udp, np.round(Udp)),
            equation_fixture=equation_fixture(simp), design_status=hip.get("status"))
        res["parity_P_UMI_DEPTH"] = parity_fixture(Sdp[:, :N_GENES], Udp[:, :N_GENES], ctp,
                                                       [f"gene_{c}" for c in G["cands"]], f"{out}/parity_P_depth10", "UMI_DEPTH")
        try:                                                                         # the exemption must NOT cover continuous data
            preprocess(make_adata(S1[:, :N_GENES], U1[:, :N_GENES], ct1, obs="CONC"), [f"gene_{c}" for c in G["cands"]]); res["exemption_typed"] = False
        except Exception:
            res["exemption_typed"] = True
        res["PASS"] = bool(res["direct_count_identity"] and res["candidate_count_identity"] and res["prevalence_identity"] and res["layer_edge_identity"]
                           and res["grn_identity_within"] and res["grn_difference_across"] and res["closure_bfs_eq_matrix"] and res["rewire_degree"]["PASS"]
                           and res["orientation_status"] == "OK"
                           and res["sim_determinism_maxdiff"] == 0 and res["equation_fixture_lam1"]["PASS"] and res["G_E_lambda0"] == 0
                           and res["umi_integer"] and res["umi_render_determinism"] and res["parity"]["PASS"] and res["exemption_typed"]
                           and all(bool(res["P_bookkeeping"][k]) for k in ("truth_excludes_hidden", "loading_reproducible", "profiles_exact",
                                                                          "native_rates_exact", "depth_render_deterministic", "depth_render_integer"))
                           and res["P_bookkeeping"]["equation_fixture"]["PASS"] and res["P_bookkeeping"]["design_status"] == "OK"
                           and res["parity_P_UMI_DEPTH"]["PASS"])
    else:
        raise ValueError("fixtures part must be 'core'; use the explicit feasibility subcommand for its frozen q/world/λ grid")
    p = f"{out}/FIXTURES_{part}{'_' + topo if topo else ''}.json"; json.dump(res, open(p, "w"), indent=1, default=str)
    print(json.dumps({k: v for k, v in res.items() if k != "grid"}, indent=1, default=str)); return p


def feasibility(out, topo="bifurcation", worlds=("A", "B"), q=3, lams=(.5, 2.0), obs="UMI", depth=10.0):
    """Only the explicitly requested feasibility grid; q=5/10 are intentionally not implicit defaults."""
    for world in worlds: generator_world(world)
    os.makedirs(out, exist_ok=True); grid = []
    for world in worlds:
        for lam in lams:
            m = measure_geometry(topo, world, q, lam, 0, 0, f"{out}/feas/{topo}_{world}_q{q}_lam{lam:g}", f"{out}/orientation", obs, depth)
            row = dict(topo=topo, world=world, q=q, lam=lam, obs=obs, depth=effective_depth(obs, depth), status=m["status"],
                       basis_dim=m["info"].get("basis_dim"), rank_feasible_by_construction=m["info"].get("rank_feasible"),
                       profile_geometry=m.get("profile_geometry"))
            if world == "D": row["thinning"] = m.get("thinning")
            for root in ROOTS:
                if root in m: row.update({f"{k}_{root}": v for k, v in m[root].items() if k != "hidden_sv"})
            grid.append(row); print(row, flush=True)
    df = pd.DataFrame(grid); cert = []
    for world, d in df.groupby("world"):
        required = [f"r_eff_hidden_{r}" for r in ROOTS]
        if not set(required) <= set(d) or d.status.ne("OK").any():
            cert.append(dict(world=world, q=q, verdict=d.loc[d.status.ne("OK"), "status"].iloc[0] if d.status.ne("OK").any() else "NO_CALL_RANK")); continue
        reff = d[required].to_numpy(float); ok_rank = bool(((reff >= .8 * q) & (reff <= 1.2 * q)).all()
                                                              and (np.ptp(reff, axis=0) / np.maximum(reff.mean(0), EPS) <= .10).all())
        if world in ("A", "D"):
            ok_align = bool((d[[f"cos_max_{r}" for r in ROOTS]].to_numpy(float) <= .30).all())
        elif world == "B":
            ok_align = bool((d[[f"cos_min_{r}" for r in ROOTS]].to_numpy(float) >= .80).all())
        else:                                                                            # C records achieved alignment; no .80 gate
            ok_align = True
        ok_prof = bool(all(.8 * q <= x["cell_profile_rank"] <= 1.2 * q for x in d.profile_geometry))
        rank_gate_applies = world not in ("C",) + PLIKE
        certified = ok_align and (not rank_gate_applies or (ok_rank and ok_prof))
        cert.append(dict(world=world, q=q, rank_gate_applies=rank_gate_applies, realised_pair_rank=d[required].to_dict("records"),
                         realised_cell_profile_rank=[x["cell_profile_rank"] for x in d.profile_geometry],
                         rank_ok=ok_rank if rank_gate_applies else None, cell_profile_rank_ok=ok_prof if rank_gate_applies else None,
                         alignment_ok=ok_align,
                         alignment_diagnostic_only=world in ("C",) + PLIKE,
                         leading_prefix_energy_fraction={r: d[f"leading_prefix_energy_fraction_{r}"].tolist() for r in ROOTS},
                         driver_pair_energy_prefix3={r: d[f"driver_pair_energy_prefix3_{r}"].tolist() for r in ROOTS},
                         verdict="CERTIFIED" if certified else ("NO_CALL_ALIGNMENT" if not ok_align else "NO_CALL_RANK")))
    res = dict(grid=grid, certification=cert); p = f"{out}/FIXTURES_feasibility_{topo}_q{q}.json"
    json.dump(res, open(p, "w"), indent=1, default=str); print(json.dumps(dict(certification=cert), indent=1)); return p


# ───────────────────────── GPU comparators, eval helper, report ─────────────────────────
def regvelo(dd):
    import anndata as ad, torch, regvelo as rgv, scvi
    scvi.settings.seed = 0; torch.manual_seed(0); np.random.seed(0)
    tag = os.path.basename(dd.rstrip("/")); A = ad.read_h5ad(f"{dd}/{tag}.h5ad"); ro = json.load(open(f"{dd}/roster.json"))
    genes = np.asarray(A.var_names).astype(str); regs = [f"gene_{s}" for s in ro["sources"]]
    rgv.REGVELOVI.setup_anndata(A, spliced_layer="Ms", unspliced_layer="Mu")
    REG = rgv.REGVELOVI(A, W=torch.zeros((len(genes), len(genes))), regulators=regs, soft_constraint=True, lam=0.0)
    REG.train(max_epochs=500, batch_size=1024)
    mod = REG.module; fn = getattr(mod.v_encoder, "GRN_Jacobian", None) or getattr(mod.v_encoder, "GRN_Jacobian2")
    with torch.no_grad():
        J = fn(torch.tensor(np.asarray(A.layers["Ms"]), dtype=torch.float32, device=next(mod.parameters()).device).mean(0)).cpu().numpy()
    full = np.abs(J[:, ro["sources"]].T); cols = np.array(ro["cols_PI"]); sc = np.take_along_axis(full, cols, 1)
    fin = np.isfinite(sc); nz = np.abs(sc[fin & (sc != 0)]); _, cnt = np.unique(np.round(sc[fin], 12), return_counts=True)
    diag = dict(n=int(sc.size), n_distinct=int(cnt.size), median_abs_nonzero=float(np.median(nz)) if nz.size else 0.0,
                exact_zero_frac=float((sc == 0).mean()), device=str(next(mod.parameters()).device), arm="ZERO skeleton, lam=0, seed 0, 500 ep, bs 1024")
    diag["OUTPUT_DEGENERATE"] = bool(diag["n_distinct"] < 0.25 * sc.size and diag["median_abs_nonzero"] < 1e-12)
    np.savez_compressed(f"{dd}/raw/regvelo.npz", score=full, jacobian=J, genes=genes); json.dump(diag, open(f"{dd}/raw/regvelo.json", "w")); print(diag)


def velorama_to_npz(dd, tsv):
    ro = json.load(open(f"{dd}/roster.json")); T = pd.read_csv(tsv, sep="\t", index_col=0)
    full = T.loc[[f"gene_{t}" for t in range(N_GENES)], [f"gene_{s}" for s in ro["sources"]]].to_numpy(float).T
    export = f"{dd}/raw/velorama.npz"; np.savez_compressed(export, score=full, source_tsv=tsv, tsv_sha256=sha_file(tsv))
    import gb_sergio_velorama_v2 as VV
    VV.write_export_sidecar(export, VV.cache_key(dd, os.path.basename(dd.rstrip("/")))); print("velorama", full.shape)


def gpu_eval(dd):
    tag = os.path.basename(dd.rstrip("/")); ro = json.load(open(f"{dd}/roster.json")); rec = json.load(open(f"{dd}/record.json"))
    G = generate_grn(rec["grn_seed"]); zt = np.load(f"{dd}/raw/counts_truth.npz", allow_pickle=False)
    truth = truth_for_world(G, rec["world"], zt["loadings"], zt["driver_ids"] if rec["world"] == "C" else None)
    cl = closure_bfs(truth, N_GENES); rows = []
    for univ in ("PI", "specv1"):
        cols = np.array(ro[f"cols_{univ}"]); src = ro["sources"]
        Y = np.array([[(s, int(t)) in truth for t in cols[i]] for i, s in enumerate(src)])
        CLS = np.array([["direct" if (s, int(t)) in truth else "indirect" if (s, int(t)) in cl else "none" for t in cols[i]] for i, s in enumerate(src)])
        true_rows = [i for i, s in enumerate(src) if s in G["regs"]]
        for m in ("velorama", "regvelo"):
            p = f"{dd}/raw/{m}.npz"
            if m == "velorama" and not os.path.exists(p) and os.path.exists(f"{dd}/velorama_v2/{tag}.velorama_v2.npz"):
                p = f"{dd}/velorama_v2/{tag}.velorama_v2.npz"          # gb_sergio_velorama_v2 output: (candidates, all genes)
            if not os.path.exists(p):
                rows.append(dict(rec_key=tag, univ=univ, method=m.upper(), status="MISSING")); continue
            if m == "velorama":
                import gb_sergio_velorama_v2 as VV
                VV.validate_export(p, VV.cache_key(dd, tag))
            M = np.take_along_axis(np.load(p)["score"].astype(float), cols, 1)
            dg = json.load(open(f"{dd}/raw/regvelo.json")) if m == "regvelo" else {}
            if dg.get("OUTPUT_DEGENERATE") or not np.isfinite(M).all():
                rows.append(dict(rec_key=tag, univ=univ, method=m.upper(), status="NOT_SCORED")); continue
            rows.append(dict(rec_key=tag, univ=univ, method=m.upper(), status="OK", **endpoints(M, Y, CLS, true_rows)))
    pd.DataFrame(rows).to_csv(f"{dd}/metrics_gpu.tsv", sep="\t", index=False); print(pd.DataFrame(rows).to_string())


def eval_metric(table, metric_fn, by=("dataset", "method", "arm", "root", "k")):
    """recompute ANY metric from the tidy table: metric_fn(y_direct bool[], score float[], label str[]) -> float"""
    return table.groupby(list(by)).apply(lambda d: metric_fn(d.label.eq("direct").to_numpy(), d.score.to_numpy(), d.label.to_numpy()))


def signflip_wy(D, n_perm=None, seed=0):
    """D: (n_GRN × n_cells) paired GRN-level contrasts (technical seeds already averaged). One-sided (H1: mean > 0) sign-flip test
    with SYNCHRONISED flips across cells (same flip vector per GRN for every cell) and Westfall–Young single-step max-T.
    Exact enumeration when 2^n_GRN ≤ 4096. Returns mean, t, raw p, WY-adjusted p per cell."""
    n = D.shape[0]; se = D.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.full(D.shape[1], EPS); t_obs = D.mean(0) / (se + EPS)
    if 2 ** n <= 4096:
        F = np.array([[1 if (b >> i) & 1 else -1 for i in range(n)] for b in range(2 ** n)], float)
    else:
        F = np.random.Generator(np.random.PCG64(seed)).choice([-1.0, 1.0], size=(n_perm or 10000, n))
    T = np.stack([(D * f[:, None]).mean(0) / ((((D * f[:, None]).std(0, ddof=1) / np.sqrt(n)) if n > 1 else np.full(D.shape[1], EPS)) + EPS)
                  for f in F])
    p_raw = (T >= t_obs[None, :] - 1e-12).mean(0); p_wy = (T.max(1)[:, None] >= t_obs[None, :] - 1e-12).mean(0)
    return dict(mean=D.mean(0), t=t_obs, p_raw=p_raw, p_wy=p_wy, n_grn=n, n_flip=len(F))


def load_required_grid(required_grid):
    if required_grid is None:
        raise SystemExit("REFUSING: --required-grid JSON is mandatory")
    rows = json.load(open(required_grid)) if isinstance(required_grid, (str, os.PathLike)) else required_grid
    if not isinstance(rows, list) or not rows:
        raise SystemExit("REFUSING: required grid must be a non-empty JSON list")
    norm = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != set(GRID_KEYS):
            raise SystemExit(f"REFUSING: each grid row must have exactly {GRID_KEYS}; got {row}")
        norm.append((str(row["topo"]), str(row["world"]), int(row["q"]), float(row["G_target"]),
                     int(row["grn_seed"]), int(row["sim_seed"])))
    if len(norm) != len(set(norm)):
        raise SystemExit("REFUSING: duplicate rows in required grid")
    return norm


def validate_report_records(recs, required_grid):
    if recs.empty:
        raise SystemExit("REFUSING: no dataset records")
    required = load_required_grid(required_grid)
    have = [(str(r.topo), str(r.world), int(r.q_req), float(r.G_target), int(r.grn_seed), int(r.sim_seed)) for r in recs.itertuples()]
    if len(have) != len(set(have)):
        dup = pd.Series(have)[pd.Series(have).duplicated(keep=False)].tolist()
        raise SystemExit(f"REFUSING: duplicate dataset records {dup[:5]}")
    miss, extra = sorted(set(required) - set(have)), sorted(set(have) - set(required))
    if miss or extra:
        raise SystemExit(f"REFUSING: grid is not exact; missing={miss[:5]} extra={extra[:5]}")
    bad = recs[recs.status != "OK"]
    if len(bad):
        raise SystemExit(f"REFUSING: required datasets are NO_CALL: {bad[['tag', 'status']].to_dict('records')}")
    for c in ("script_sha256", "config_sha256"):
        if recs[c].nunique() > 1:
            raise SystemExit(f"REFUSING: mixed {c} across records: {recs[c].unique()}")
    if not (recs.tuning_script_sha256 == recs.script_sha256).all():
        raise SystemExit("REFUSING: tuning metadata script SHA does not match dataset script SHA")
    for scope, d in recs.groupby(["topo", "world", "q_req", "grn_seed"], dropna=False):
        if d.tuning_sha256.isna().any() or d.tuning_sha256.nunique(dropna=False) != 1:
            raise SystemExit(f"REFUSING: mixed/missing tuning SHA within scope {scope}: {d.tuning_sha256.tolist()}")
    if recs.groupby("grn_seed").roster_sha256.nunique().max() > 1:
        raise SystemExit("REFUSING: roster differs within a GRN")
    return True


def report(out_root, univ="PI", required_grid=None):
    """PI three-table format + GRN-level inference. Assertions: every record shares script/config SHA; tuning SHA is constant within
    (topology, world, q, GRN); required grid is exact and duplicate-free;
    roster sha identical within GRN; the required grid (topo × G* × GRN × sim seed) is complete and certified. Technical sim
    seeds are averaged within GRN BEFORE any GRN-level statistic. Oracle rows are DIAGNOSTIC only."""
    import glob, gb_canonical as GC
    recs = pd.DataFrame([json.load(open(f)) for f in glob.glob(f"{out_root}/*/record.json")])
    validate_report_records(recs, required_grid)
    M = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(f"{out_root}/*/metrics.tsv")])
    Pk = pd.concat([pd.read_csv(f, sep="\t") for f in glob.glob(f"{out_root}/*/perk.tsv")])
    Gg = [pd.read_csv(f, sep="\t") for f in glob.glob(f"{out_root}/*/metrics_gpu.tsv")]
    if Gg:
        Ggpu = pd.concat(Gg); Ggpu["analysis"] = "primary"; M = pd.concat([M, Ggpu])
    M = M[(M.univ == univ) & M.rec_key.isin(recs.tag)]; Pk = Pk[(Pk.univ == univ) & Pk.rec_key.isin(recs.tag)]; key = recs.set_index("tag")
    for c in ("topo", "G_target", "grn_seed", "sim_seed"):
        M[c] = M.rec_key.map(key[c]); Pk[c] = Pk.rec_key.map(key[c])
    # technical seeds averaged within GRN FIRST
    Mg = M[M.analysis == "primary"].groupby(["topo", "G_target", "grn_seed", "method"]).mean(numeric_only=True).reset_index()
    Mgs = M[M.analysis == "root_transient_excluded"].groupby(["topo", "G_target", "grn_seed", "method"]).mean(numeric_only=True).reset_index()
    Pg = Pk[Pk.analysis == "primary"].groupby(["topo", "G_target", "grn_seed", "arm", "k"]).mean(numeric_only=True).reset_index()
    out = []
    for topo, R_ in recs.groupby("topo"):
        hdr = {g: f"G*{g:g} (G_E {d[['G_E_dc1_min', 'G_E_dc1_max']].mean().mean():.3f}, n_GRN {d.grn_seed.nunique()})" for g, d in R_.groupby("G_target")}
        m = Mg[Mg.topo == topo]; pk = Pg[Pg.topo == topo]
        sel = ["Pearson_abs", "R_X_TRUE", "GENIE3", "GRNBOOST2", "VELORAMA", "REGVELO", "U_k0", "R_U_TRUE", "R_U_REV",
               "R_U_EQC_hi", "R_U_EQC_lo", "INFO_MATCHED_U", "INFO_LINEAR", "INFO_SPLINE"]
        def tab(col):
            t = m[m.method.isin(sel) | m.method.str.match(r"R_U_(RIS|ROS)\d")].copy()
            t.loc[t.method.str.match(r"R_U_RIS\d"), "method"] = f"R_U_RIS(mean{N_DRAW})"; t.loc[t.method.str.match(r"R_U_ROS\d"), "method"] = f"R_U_ROS(mean{N_DRAW})"
            P = t.pivot_table(index="method", columns="G_target", values=col, aggfunc="mean")
            for k in (0, 3, 5, 8, 10, 15, 20, 30):
                P.loc[f"U k={k} (TRUE)"] = pk[(pk.arm == "TRUE") & (pk.k == k)].groupby("G_target")[col].mean() if col in pk else np.nan
            ok = pk[pk.arm == "TRUE"].groupby(["G_target", "grn_seed"]).apply(lambda d: d.loc[d.AP_pct.idxmax()])
            P.loc["[DIAGNOSTIC] ORACLE best k TRUE"] = ok.groupby("G_target")[col].mean() if col in ok else np.nan
            P.loc["[DIAGNOSTIC]   oracle k (median GRN)"] = ok.groupby("G_target").k.median()
            P.columns = [hdr[c] for c in P.columns]
            return P.round(3).to_string()
        bookkeeping = R_[["world", "n_direct", "n_cands", "roster_sha256", "prevalence"]].drop_duplicates().to_dict("records")
        out.append(f"\n### {topo}  [universe {univ}; prevalence {m.prevalence.mean():.4f}]\nTruth/roster bookkeeping: {json.dumps(bookkeeping, sort_keys=True)}\n"
                   f"Table 1 AUPRC (native)\n{tab('AP_native')}\n"
                   f"Table 1b lift\n{tab('lift_native')}\nTable 1c pooled within-source-percentile AP\n{tab('AP_pct')}\nTable 1d macro nAP\n{tab('macro_nAP')}\n"
                   f"Table 2 AUROC\n{tab('AUROC')}\nTable 3a AUROC direct>indirect\n{tab('AUROC_dir_vs_ind')}\nTable 3b EDP@50\n{tab('EDP50')}\n"
                   f"Table 3c EDP baseline\n{tab('EDP_baseline')}\nTable 3d indirect fraction top-50\n{tab('indirect50')}\n")
        prof = pd.DataFrame([dict(G_target=r.G_target, grn_seed=r.grn_seed, sim_seed=r.sim_seed,
                                  cell_profile_corr_offdiag_max=r.profile_geometry["cell_profile_corr_offdiag_max"])
                             for r in R_.itertuples()])
        out.append("Normalized realized profile correlation (absolute off-diagonal maximum)\n" +
                   prof.groupby("G_target").cell_profile_corr_offdiag_max.agg(["mean", "max"]).round(4).to_string())
        geom_rows = []
        for r in R_.itertuples():
            for root in ROOTS:
                gg = getattr(r, f"geometry_{root}")
                geom_rows.append(dict(world=r.world, G_target=r.G_target, grn_seed=r.grn_seed, sim_seed=r.sim_seed, root=root,
                                      r_eff=gg.get("r_eff_hidden"), cos_min=gg.get("cos_min"), cos_max=gg.get("cos_max"),
                                      driver_energy_prefix3=gg.get("driver_pair_energy_prefix3")))
        out.append("Realized programme geometry (C alignment is diagnostic, never a failure criterion)\n" +
                   pd.DataFrame(geom_rows).groupby(["world", "G_target", "root"]).mean(numeric_only=True).round(4).to_string())
        if (R_.world == "D").any():
            support_rows = []
            for r in R_[R_.world == "D"].itertuples():
                for s, sr in r.thinning["per_source"].items():
                    support_rows.append(dict(G_target=r.G_target, grn_seed=r.grn_seed, sim_seed=r.sim_seed, source=int(s),
                                             detected_before=sr["before"]["either_positive_cells"],
                                             detected_after=sr["after"]["either_positive_cells"],
                                             detection_before=sr["before"]["either_detection"],
                                             detection_after=sr["after"]["either_detection"]))
            out.append("World D per-source detection before/after thinning\n" + pd.DataFrame(support_rows).to_string(index=False))
        sens = Mgs[Mgs.topo == topo]
        if len(sens):
            keep = ["R_U_TRUE", "U_k0", "R_X_TRUE", "INFO_MATCHED_U", "INFO_LINEAR", "INFO_SPLINE"]
            out.append("Root-transient sensitivity (pairs touching a root cell with sampled step < S_REF excluded; macro nAP)\n" +
                       sens[sens.method.isin(keep)].pivot_table(index="method", columns="G_target", values="macro_nAP", aggfunc="mean").round(4).to_string())
    # One predeclared WY family: primary, information-matched, EQC, root-transient, and negative-control contrasts.
    cells, D = [], []
    for (topo, g_), d in Mg.groupby(["topo", "G_target"]):
        piv = d.pivot_table(index="grn_seed", columns="method", values="macro_nAP")
        for ref in ("U_k0", "R_X_TRUE", "INFO_LINEAR", "INFO_SPLINE"):
            if {"R_U_TRUE", ref} <= set(piv.columns):
                cells.append(f"{topo}|G{g_:g}|R_U_TRUE-{ref}"); D.append(piv["R_U_TRUE"] - piv[ref])
        for ref in ("INFO_LINEAR", "INFO_SPLINE"):
            if {"INFO_MATCHED_U", ref} <= set(piv.columns):
                cells.append(f"{topo}|G{g_:g}|INFO_MATCHED_U-{ref}"); D.append(piv["INFO_MATCHED_U"] - piv[ref])
    for (topo, g_), d in Mgs.groupby(["topo", "G_target"]):
        piv = d.pivot_table(index="grn_seed", columns="method", values="macro_nAP")
        for ref in ("U_k0", "R_X_TRUE", "INFO_LINEAR", "INFO_SPLINE"):
            if {"R_U_TRUE", ref} <= set(piv.columns):
                cells.append(f"{topo}|G{g_:g}|ROOT_EXCL R_U_TRUE-{ref}"); D.append(piv["R_U_TRUE"] - piv[ref])
    for (topo, g_), d in recs.groupby(["topo", "G_target"]):
        mech = pd.DataFrame([dict(grn=r.grn_seed, **r.mech) for r in d.itertuples()]).groupby("grn").mean(numeric_only=True)
        for c in ("TRUE_minus_REV", "TRUE_minus_RIS_mean", "TRUE_minus_ROS_mean", "TRUE_minus_EQC_hi", "TRUE_minus_EQC_lo", "TRUE_minus_k0"):
            cells.append(f"{topo}|G{g_:g}|MECH {c}"); D.append(mech[c])
        null_rows = []
        for r in d.itertuples():
            null_rows.append(dict(grn=r.grn_seed,
                                  rewired=r.null_rewire["prod_minus_k0_macro_mean"],
                                  U_destroyed=r.null_Udestroy["prod_minus_k0_macro"]))
        null_df = pd.DataFrame(null_rows).groupby("grn").mean(numeric_only=True)
        for c in ("rewired", "U_destroyed"):
            cells.append(f"{topo}|G{g_:g}|NULL {c} prod-k0 macro_nAP"); D.append(null_df[c])
        out.append(f"Null controls {topo} G={g_:g} (technical seeds averaged; reproduction threshold +.02)\n" +
                   null_df.round(4).to_string())
    if cells:
        Dm = pd.concat(D, axis=1); Dm.columns = cells
        valid = ~Dm.isna().any(axis=0); omitted = Dm.columns[~valid].tolist(); Dm = Dm.loc[:, valid].dropna()
        if Dm.empty or Dm.shape[1] == 0:
            raise SystemExit("REFUSING: no complete GRN-level cells for the WY family")
        inf = signflip_wy(Dm.to_numpy())
        out.append("GRN-level paired sign-flip, synchronised Westfall–Young max-T (one-sided; SESOI .02)\n" +
                   pd.DataFrame(dict(cell=Dm.columns, mean=inf["mean"], t=inf["t"], p_raw=inf["p_raw"], p_WY=inf["p_wy"])).round(4).to_string(index=False) +
                   f"\n n_GRN={inf['n_grn']}, flips={inf['n_flip']} (minimum attainable p = {1 / inf['n_flip']:.3f}); omitted incomplete cells={omitted}")
    txt = "\n".join(out); open(f"{out_root}/REPORT_{univ}.md", "w").write(txt); print(txt)


def smoke(out_root, tuning_json, world="A", obs="UMI", depth=10.0):
    """One-GRN paired end-to-end smoke. This is an implementation check, not the formal three-GRN scientific pilot."""
    generator_world(world); tune_rec = validate_tuning_json(tuning_json, "bifurcation", world, 3, 0, obs, depth)
    for key in ("0.00", "0.15"):
        if tune_rec["lambda_for_G"].get(key) is None:
            raise SystemExit(f"REFUSING_SMOKE: tuning JSON has no attained lambda for G={key}")
    import glob
    existing = glob.glob(f"{out_root}/*/record.json")
    if existing:
        raise SystemExit(f"REFUSING_SMOKE_NONEMPTY_OUTPUT: {existing[:3]}")
    os.makedirs(out_root, exist_ok=True)
    from distributed import LocalCluster, Client
    client = Client(LocalCluster(n_workers=8, threads_per_worker=1, memory_limit="0"))
    try:
        for gt_ in (0.0, 0.15):
            lam = float(tune_rec["lambda_for_G"][f"{gt_:.2f}"])
            rec = run_one(out_root, "bifurcation", world, 3, lam, gt_, 0, 0, client, tuning_json, obs, depth)
            dd = f"{out_root}/{rec['tag']}"; verification = verify_dataset(dd)
            raw_keys = set(np.load(f"{dd}/raw/geobridge_raw.npz").files)
            metrics = pd.read_csv(f"{dd}/metrics.tsv", sep="\t")
            eqc = {k: v for k, v in rec["fixtures"].items() if "|EQC_" in k and k.endswith("|info")}
            checks = dict(status=rec["status"], verify_PASS=verification["PASS"], fixture_failures=rec.get("fixture_failures"),
                          persistence_maxdiff=rec.get("fixture_persistence_maxdiff"),
                          info_persistence_maxdiff=rec.get("fixture_info_persistence_maxdiff"),
                          info_methods=sorted(set(metrics.loc[metrics.method.str.startswith("INFO_"), "method"])),
                          info_raw_bound=any(k.startswith("info|") for k in raw_keys), EQC=eqc,
                          null_rewire=rec.get("null_rewire"), null_Udestroy=rec.get("null_Udestroy"),
                          root_transient_sensitivity=rec.get("root_transient_sensitivity"),
                          profile_corr_offdiag_max=rec["profile_geometry"]["cell_profile_corr_offdiag_max"])
            print("SMOKE_INTERMEDIATE " + json.dumps(checks, default=float), flush=True)
            if not verification["PASS"] or rec.get("fixture_failures") or not checks["info_raw_bound"] or len(checks["info_methods"]) != 3:
                raise RuntimeError(f"SMOKE_CHECK_FAILED: {checks}")
            if gt_ > 0:
                trf = rec.get("fixture_tune_run_geometry", {})
                if not trf.get("PASS"):
                    raise RuntimeError(f"SMOKE_TUNE_RUN_GEOMETRY_FAILED: {trf}")
                # Equal-C is fail-closed per arm, not per dataset: preserve the valid nested prefix and mark every
                # cell from failed_at onward NO_CALL_EQUAL_C. The ±.01 matching tolerance is never relaxed.
                for key, info in eqc.items():
                    valid = np.asarray(info["valid"], bool); failed = info.get("failed_at")
                    prefix_ok = valid[0] and (failed is None or (valid[:failed].all() and not valid[failed:].any()))
                    err = np.asarray(info["match_error"], float)
                    match_ok = bool(np.isfinite(err[valid]).all() and np.max(err[valid]) <= float(info["tolerance"]) + 1e-12)
                    status_ok = info.get("status") == ("OK" if failed is None else "NO_CALL_EQUAL_C")
                    if not (prefix_ok and match_ok and status_ok):
                        raise RuntimeError(f"SMOKE_EQUAL_C_SEMANTICS_FAILED: {key}: {info}")
                    if failed is not None:
                        arm = key.split("|")[1]
                        rows_ = metrics[(metrics.method.str.endswith(arm))]
                        if rows_.empty or not rows_.status.eq("NO_CALL_EQUAL_C").all():
                            raise RuntimeError(f"SMOKE_EQUAL_C_FAIL_OPEN: {key}")
    finally:
        client.close()
    grid = [dict(topo="bifurcation", world=world, q=3, G_target=g, grn_seed=0, sim_seed=0) for g in (0.0, 0.15)]
    report(out_root, "PI", grid)
    print(f"SMOKE PASS world={world} obs={obs} depth={effective_depth(obs, depth)}: paired G=0/.15, tune==run geometry, verify, fail-closed per-arm EQC, persistence, nulls, information-matched scoring, strict-grid report", flush=True)


# ───────────────────────── fast algebraic selftest ─────────────────────────
def selftest():
    rng = np.random.default_rng(0)
    # World P bookkeeping: exact pilot profiles, reproducible overlapping 40%-of-leaf loadings, and hidden-free direct truth.
    def pilot_profile_reference(topo, q):
        d = np.asarray(DEPTH[topo], float); d = d / max(d.max(), 1); b = np.asarray(BRANCH[topo], float)
        base = [d, np.exp(-((d - .5) ** 2) / .08), .5 + .5 * b, 1 - d]; out = []
        for m in range(q):
            p = base[m % 4]; p = (p - p.min()) / max(np.ptp(p), 1e-9)
            out.append(.3 + 3.5 * (np.roll(p, m // 4) if m >= 4 else p))
        return np.array(out)
    for topo in TOPOLOGIES:
        assert np.array_equal(profiles(topo, 9), pilot_profile_reference(topo, 9))
    Gp = generate_grn(0); Lp1 = hidden_loadings(Gp, 0, "P", 3); Lp2 = hidden_loadings(Gp, 0, "P", 3)
    assert np.array_equal(Lp1, Lp2) and all(np.count_nonzero(row) == int(.4 * len(Gp["leaves"])) for row in Lp1)
    assert np.all((np.abs(Lp1[Lp1 != 0]) >= 1) & (np.abs(Lp1[Lp1 != 0]) <= 3))
    assert any(len(set(np.flatnonzero(Lp1[i])) & set(np.flatnonzero(Lp1[j]))) > 0 for i in range(3) for j in range(i))
    assert truth_for_world(Gp, "P", Lp1, list(range(N_GENES, N_GENES + 3))) == Gp["gt"]
    assert all(s < N_GENES and t < N_GENES for s, t in truth_for_world(Gp, "P"))
    Cd = (np.arange(40)[:, None] % 7 + .25) * np.ones((1, 5)); Dd1 = render_umi(Cd, Cd + .5, 0, 0, 1.0, 10)
    Dd2 = render_umi(Cd, Cd + .5, 0, 0, 1.0, 10)
    assert all(np.array_equal(a, b) for a, b in zip(Dd1, Dd2)) and all(np.array_equal(a, np.round(a)) for a in Dd1)
    assert observation_tag("UMI", 10) == "obsUMI_depth1" and observation_tag("UMI_DEPTH", 10) == "obsUMI_DEPTH_depth10"
    # Declared basis is continuous at every bin boundary, including branch points.
    for topo in TOPOLOGIES:
        sp = topology_spec(topo)
        for child, parent in enumerate(sp["parents"]):
            if parent < 0: continue
            tau0 = float(DEPTH[topo][child]); bp, _ = trajectory_basis(topo, [parent], [tau0]); bc, _ = trajectory_basis(topo, [child], [tau0])
            assert np.abs(bp - bc).max() < 1e-12, (topo, parent, child, np.abs(bp - bc).max())
    # Generalized-eigen selection orders alignment and the within-subspace rotation preserves both orthogonalities/equal pair energy.
    Bsyn = rng.normal(size=(300, 7)); Is = rng.integers(0, 150, 800); Js = rng.integers(150, 300, 800); dBsyn = Bsyn[Js] - Bsyn[Is]
    Ugsyn = _orth(dBsyn[:, :2]); Aa, ia = orient_profiles(Bsyn, [dBsyn], [Ugsyn], 2, "A"); Ab, ib = orient_profiles(Bsyn, [dBsyn], [Ugsyn], 2, "B")
    assert ia["construction_cos_max"] < ib["construction_cos_min"], (ia, ib)
    for A_ in (Aa, Ab):
        F_ = A_[0] + Bsyn @ A_[1:]; dF_ = dBsyn @ A_[1:]
        assert np.abs(F_.mean(0)).max() < 1e-12
        assert np.abs(F_.T @ F_ - np.diag(np.diag(F_.T @ F_))).max() < 1e-9
        GP = dF_.T @ dF_; assert np.abs(GP - np.eye(2) * GP[0, 0]).max() < 1e-9
    # All declared profile rates remain positive; no clipping is used.
    bins_ = np.concatenate([np.repeat(b, 101) for b in range(TOPOLOGIES["bifurcation"][0])])
    tau_ = np.concatenate([DEPTH["bifurcation"][b] + np.linspace(0, 1, 101) for b in range(TOPOLOGIES["bifurcation"][0])])
    Ar = np.zeros((topology_spec("bifurcation")["dim"] + 1, 2)); Ar[1, 0] = .3; Ar[2, 1] = .3
    assert np.min(HIDDEN_BASE + HIDDEN_AMP * profile_values("bifurcation", dict(bin=bins_, tau=tau_), Ar)) > 0
    # Zero profile coefficients dispatch exactly to the base SERGIO per-bin MR implementation.
    class FakeBase:
        def calculate_prod_rate_U_(self, gID, binID, n): return [self.graph_[gID]["rates"][binID]] * n
    FakeContinuous = continuous_sergio_class(FakeBase); fs = FakeContinuous.__new__(FakeContinuous)
    fs.graph_ = {100: {"rates": [HIDDEN_BASE] * 4}}; fs.binDict = {b: [type("G", (), {"Conc": [[0.0]]})()] for b in range(4)}
    fs.configure_hidden_profiles([100], "bifurcation", np.zeros((topology_spec("bifurcation")["dim"] + 1, 1)))
    assert fs.calculate_prod_rate_U_(100, 2, 3) == [fs.graph_[100]["rates"][2]] * 3
    nzcoef = np.zeros((topology_spec("bifurcation")["dim"] + 1, 1)); nzcoef[0, 0] = .1
    fs.binDict = {b: {100: type("G", (), {"Conc": [[0.0]]})()} for b in range(4)}; fs.configure_hidden_profiles([100], "bifurcation", nzcoef)
    assert all(isinstance(x, (float, np.floating)) for x in fs.calculate_prod_rate_U_(100, 2, 3))
    # Equal-C toy: isotropic energy in span(Qt) makes every rank-k projector exactly equal-C while hidden overlap can differ.
    Qt_ = np.linalg.qr(rng.normal(size=(200, 6)))[0]; Dtoy = Qt_.copy(); wt = np.full(200, 1 / 200); Qht = Qt_[:, :2]
    Qhi, ehi = equal_c_projector(Qt_, Dtoy, wt, Qht, "hi", n_random=128); Qlo, elo = equal_c_projector(Qt_, Dtoy, wt, Qht, "lo", n_random=128)
    assert ehi["status"] == elo["status"] == "OK" and max(ehi["match_error"]) <= .01 and max(elo["match_error"]) <= .01
    assert ehi["overlap_hidden"][1] > elo["overlap_hidden"][1]
    absS = {r: rng.random((KMAX + 1, 4, 12)) for r in ROOTS}
    absS["dc1_max"][:, 1, 9] = absS["dc1_max"][:, 1, 10]
    cols = np.array([[t for t in range(12) if t != s] for s in range(4)])
    assert np.abs(production_scalar(absS, cols) - reference_scalar(absS, cols)).max() <= 1e-12
    H = rng.standard_normal((400, 3)); I = rng.integers(0, 200, 400); J = rng.integers(0, 200, 400); fold = rng.integers(0, 2, 200)
    B = rng.standard_normal((3, 5)); assert abs(g_crossfit(H @ B, H, I, J, fold) - 1) < 1e-10
    Ud = rng.random((60, 4)) + .1; eds = [(0, 2, 2.0, 2.0, .5), (1, 3, -1.5, 2.0, .6), (1, 2, 1.0, 2.0, .4)]
    g0 = geometry_stats(Ud, eds, I[:30] % 60, J[:30] % 60)[0]["G_E"]
    eds_h = eds + [(0, N_GENES + 0, 0.7, 2.0, .5)]; Uh = np.c_[Ud, np.zeros((60, N_GENES - 4)), rng.random((60, 1)) + .1]
    gh = geometry_stats(Uh, eds_h, I[:30] % 60, J[:30] % 60)[0]["G_E"]; assert g0 == 0.0 and 0 < gh < 1
    assert abs(hill_sergio(np.array([.5]), .5, 2.0, False)[0] - .5) < 1e-15 and hill_sergio(np.array([0.0]), .5, 2.0, True)[0] == 1.0
    Yt = np.zeros((6, 11), bool); Yt[[0, 0, 1, 2, 3, 4, 5, 5], [1, 2, 3, 4, 5, 6, 7, 8]] = True
    ct_ = np.array([[t for t in range(12) if t != s_] for s_ in range(6)])
    Yp_ = rewire_double_swap(Yt, ct_, list(range(6)), np.random.Generator(np.random.PCG64(1)), 500); f_ = degree_fixture(Yt, Yp_, ct_)
    assert f_["out_deg"] and f_["in_deg"] and f_["n_edges"]
    Dt = np.array([[.1, .05], [.2, .0], [.15, -.02], [.12, .01]]); inf = signflip_wy(Dt); assert inf["n_flip"] == 16 and inf["p_wy"][0] >= inf["p_raw"][0]
    dP = rng.standard_normal((500, 30)); Qt = np.linalg.qr(dP)[0]
    Pk = np.mean([(lambda O: O[:, :2] @ O[:, :2].T)(haar(6, s)) for s in range(4000)], 0)   # Haar prefix: E[P_k] = (k/n) I
    assert np.abs(Pk - np.eye(6) * 2 / 6).max() < 0.03, np.abs(Pk - np.eye(6) * 2 / 6).max()
    for s in range(N_DRAW):
        assert np.linalg.norm(arm_Q(f"ROS{s}", dP, Qt).T @ Qt, 2) <= 1e-10
        Qr = arm_Q(f"RIS{s}", dP, Qt); assert np.abs(Qt @ Qt.T - Qr @ Qr.T).max() < 1e-10
    Y = np.zeros((3, 10), bool); Y[0, :2] = Y[1, 3] = True; M = Y + 0.01 * rng.random(Y.shape)
    CLS = np.where(Y, "direct", "none").astype(object); CLS[2, 0] = "indirect"; e = endpoints(M, Y, CLS, [0, 1])
    assert abs(e["AP_native"] - 1) < 1e-12 and abs(e["macro_nAP"] - 1) < 1e-12 and e["EDP50"] == 3 / 4
    for g in (0, 1, 2, 5):
        G = generate_grn(g); assert len(G["gt"]) == N_DIRECT and len(G["cands"]) == 39 and closure_bfs(G["gt"], N_GENES) == closure_matrix(G["gt"], N_GENES)
    # World C truth: observed candidate regulators remain on-axis; every L edge is direct truth; roster is unchanged and prevalence is exact.
    Gc = generate_grn(0); Lc = hidden_loadings(Gc, 0, "C", 3); dc = world_c_drivers(Gc, 3); tc = truth_for_world(Gc, "C", Lc, dc)
    assert dc == sorted(Gc["mrs"])[:3] and set(dc) <= set(Gc["regs"]) <= set(Gc["cands"]) and max(dc) < N_GENES
    assert all((dc[h], int(t)) in tc for h in range(3) for t in np.flatnonzero(Lc[h])) and len(tc) > len(Gc["gt"])
    roster_c = [(s, t) for s in Gc["cands"] for t in range(N_GENES) if t != s]
    assert len(roster_c) == len(Gc["cands"]) * (N_GENES - 1) and sum(e in tc for e in roster_c) == len(tc)
    assert truth_for_world(Gc, "A") == truth_for_world(Gc, "B") == truth_for_world(Gc, "D") == truth_for_world(Gc, "P") == Gc["gt"]
    Lt = np.zeros((1, N_GENES)); Lt[0, 3] = 2.0
    Et = typed_edges([(3, dc[0], 1.25, 2.0, .5), (3, dc[0], 2.0, 2.0, .5)])
    cm = component_edge_mask(Et, "C", Lt, 1.0, [dc[0]]); assert np.array_equal(cm, [False, True])
    Pgt, Pht = production_parts(np.ones((4, N_GENES)), Et, component_mask=cm); assert Pgt[:, 3].min() > 0 and Pht[:, 3].min() > 0
    # World D: exact 3+3 declared sources, deterministic integer thinning, no non-source mutation, truth unchanged.
    sd = world_d_sources(Gc, 0); assert len(sd) == 6 and len(set(sd) & set(Gc["mrs"])) == 3 and len(set(sd) & set(Gc["ints"])) == 3
    C0 = (np.arange(64)[:, None] % 11 + 2) * np.ones((1, N_GENES)); C1 = C0 + 3
    Sd1, Ud1, ss1 = thin_world_d_counts(C0, C1, Gc, 0, 1, .5); Sd2, Ud2, ss2 = thin_world_d_counts(C0, C1, Gc, 0, 1, .5)
    assert ss1 == ss2 == sd and np.array_equal(Sd1, Sd2) and np.array_equal(Ud1, Ud2)
    assert np.array_equal(Sd1, np.round(Sd1)) and np.array_equal(Ud1, np.round(Ud1))
    other = np.setdiff1d(np.arange(N_GENES), sd); assert np.array_equal(Sd1[:, other], C0[:, other]) and np.array_equal(Ud1[:, other], C1[:, other])
    # The common world-render wrapper preserves A/B outputs bit-for-bit on the same fixture.
    legacy_S, legacy_U = render_umi(C0, C1, 0, 1, .5)
    for w in ("A", "B"):
        Sw, Uw, srcw, _, _ = render_world_umi(C0, C1, Gc, w, 0, 1, .5)
        assert not srcw and np.array_equal(Sw, legacy_S) and np.array_equal(Uw, legacy_U)
    import gb_geometry_audit as GA                                          # C_op definition == real-data cw_curve (direct formula)
    Ms_ = rng.random((50, 7)) + .1; I_ = rng.integers(0, 50, 80); J_ = rng.integers(0, 50, 80); Q_ = np.linalg.qr(rng.standard_normal((80, 4)))[0]
    D_ = (Ms_[J_] - Ms_[I_]) / Ms_.std(0); w_ = GA.weights(I_, J_, 50)["pair"]
    direct = [1 - (w_[:, None] * (D_ - Q_[:, :k] @ (Q_[:, :k].T @ D_)) ** 2).sum() / (w_[:, None] * D_ ** 2).sum() for k in range(5)]
    assert np.abs(GA.cw_curve(Q_, Ms_, I_, J_, w_, Ms_.std(0)) - direct).max() < 1e-12
    ws_ = _cop_workspace(Q_, D_, w_); assert np.max(np.abs(np.array([_cop_from_coeff(np.eye(4)[:, :k], ws_) for k in range(5)]) - direct)) < 1e-12
    import gb_infomatched as IM, gb_sergio_velorama_v2 as VV, tempfile
    masks_, _ = IM.fold_masks(rng.integers(0, 400, 10000), rng.integers(0, 400, 10000), 400)
    assert abs(np.mean([tr.mean() for tr, _ in masks_]) - 16 / 25) < .03
    IM.selftest()                                                               # planted-linear, spline-only, shuffled-U, exact leakage
    # Typed geometry edge round-trip remains integer-indexable end to end.
    td = tempfile.mkdtemp(); ep = f"{td}/edges.npz"; e0 = typed_edges([(1, 2, 1.5, 2.0, .4)])
    np.savez_compressed(ep, edges=e0); e1 = np.load(ep, allow_pickle=False)["edges"]
    assert e1.dtype == EDGE_DTYPE and production_parts(rng.random((8, 4)), e1, n_obs=4)[0].shape == (8, 4)
    # Tuning metadata and current script SHA are mandatory.
    op = f"{td}/orientation.json"; json.dump(dict(context_sha256="ctx"), open(op, "w"))
    tp = f"{td}/tune.json"; json.dump(dict(topo="bifurcation", world="A", q=3, grn_seed=0, sim_seed=0, obs="UMI", depth=1.0,
                                             script_sha256=sha_file(os.path.abspath(__file__)), lambda_for_G={"0.00": 0.0},
                                             orientation=dict(relative_path="orientation.json", sha256=sha_file(op), context_sha256="ctx"),
                                             geometry_fixtures={}), open(tp, "w"))
    assert validate_tuning_json(tp, "bifurcation", "A", 3, 0)["lambda_for_G"]["0.00"] == 0.0
    bad_tune = json.load(open(tp)); bad_tune["q"] = 5; json.dump(bad_tune, open(tp, "w"))
    try:
        validate_tuning_json(tp, "bifurcation", "A", 3, 0); raise AssertionError("bad tuning metadata accepted")
    except RuntimeError as e:
        assert "TUNING_METADATA_MISMATCH" in str(e)
    grid_ = [dict(topo="bifurcation", world="A", q=3, G_target=0.0, grn_seed=0, sim_seed=0)]
    assert load_required_grid(grid_) == [("bifurcation", "A", 3, 0.0, 0, 0)]
    try:
        load_required_grid(grid_ * 2); raise AssertionError("duplicate required grid accepted")
    except SystemExit as e:
        assert "duplicate" in str(e)
    report_recs = pd.DataFrame([dict(tag="a", topo="bifurcation", world="A", q_req=3, G_target=0.0, grn_seed=0, sim_seed=0,
                                     status="OK", script_sha256="s", tuning_script_sha256="s", tuning_sha256="t0",
                                     config_sha256="c", roster_sha256="r"),
                                dict(tag="b", topo="bifurcation", world="A", q_req=3, G_target=.15, grn_seed=0, sim_seed=0,
                                     status="OK", script_sha256="s", tuning_script_sha256="s", tuning_sha256="t0",
                                     config_sha256="c", roster_sha256="r")])
    grid2 = grid_ + [dict(topo="bifurcation", world="A", q=3, G_target=.15, grn_seed=0, sim_seed=0)]
    assert validate_report_records(report_recs, grid2)
    mixed = report_recs.copy(); mixed.loc[1, "tuning_sha256"] = "different"
    try:
        validate_report_records(mixed, grid2); raise AssertionError("mixed within-scope tuning SHA accepted")
    except SystemExit as e:
        assert "tuning SHA" in str(e)
    assert fixture_statuses(False, 0, True, 0, 0) == ["NO_CALL_ARMS"]
    assert fixture_statuses(True, 1e-3, False, 1e-3, 1e-3) == ["NO_CALL_SCALAR_PARITY", "NO_CALL_REWIRING_DEGREE", "NO_CALL_PERSISTENCE"]
    tm = dict(bin=np.array([0, 0, 1, 2]), step=np.array([0, CELLS_PER_TYPE, 0, 0]))
    assert np.array_equal(root_transient_cells("bifurcation", tm), [True, False, False, False])
    pg = profile_geometry("bifurcation", dict(bin=bins_, tau=tau_), Ar)
    assert np.isfinite(pg["cell_profile_corr_offdiag_max"]) and pg["cell_profile_corr_offdiag_max"] <= 1 + 1e-12
    VV.selftest()
    print("gb_sergio_drift_v5 selftest PASS")


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "selftest":
        selftest()
    elif cmd == "fixtures":
        selftest(); fixtures(sys.argv[2], *(sys.argv[3:5]))
    elif cmd == "feasibility":
        ap_ = argparse.ArgumentParser(); ap_.add_argument("cmd"); ap_.add_argument("out"); ap_.add_argument("--topo", default="bifurcation")
        wg = ap_.add_mutually_exclusive_group(); wg.add_argument("--world", choices=WORLDS); wg.add_argument("--worlds", default="A,B")
        ap_.add_argument("--q", type=int, default=3); ap_.add_argument("--lams", default=".5,2")
        ap_.add_argument("--obs", choices=OBSERVATIONS, default="UMI"); ap_.add_argument("--depth", type=float, default=10.0)
        a = ap_.parse_args(); worlds = (a.world,) if a.world else tuple(a.worlds.split(","))
        feasibility(a.out, a.topo, worlds, a.q, tuple(float(x) for x in a.lams.split(",")), a.obs, a.depth)
    elif cmd == "tune":
        ap_ = argparse.ArgumentParser(); ap_.add_argument("cmd"); ap_.add_argument("out"); ap_.add_argument("--topo"); ap_.add_argument("--world", choices=WORLDS, default="A")
        ap_.add_argument("--q", type=int, default=3); ap_.add_argument("--grn", type=int, default=0); ap_.add_argument("--targets")
        ap_.add_argument("--tol", type=float, default=.03); ap_.add_argument("--max-iter", type=int, default=14)
        ap_.add_argument("--obs", choices=OBSERVATIONS, default="UMI"); ap_.add_argument("--depth", type=float, default=10.0)
        ap_.add_argument("--confirm", action="store_true")  # prereg v2: confirmation GRNs tune lambda on reserved sim seed 0
        a = ap_.parse_args()
        tg = None if a.targets is None else tuple(float(x) for x in a.targets.split(","))
        check_grn_seed(a.grn, a.confirm); tune(a.out, a.topo, a.world, a.q, a.grn, tg, a.tol, a.max_iter, a.obs, a.depth)
    elif cmd == "run":
        ap_ = argparse.ArgumentParser(); ap_.add_argument("cmd"); ap_.add_argument("out"); ap_.add_argument("--topo", default="bifurcation")
        ap_.add_argument("--world", choices=WORLDS, default="A"); ap_.add_argument("--q", type=int, default=3)
        sg = ap_.add_mutually_exclusive_group(); sg.add_argument("--gtargets"); sg.add_argument("--lams")
        ap_.add_argument("--grn-seeds", default="0,1,2"); ap_.add_argument("--sim-seeds", default="0,1"); ap_.add_argument("--confirm", action="store_true")
        ap_.add_argument("--obs", choices=OBSERVATIONS, default="UMI"); ap_.add_argument("--depth", type=float, default=10.0)
        a = ap_.parse_args(); selftest()
        from distributed import LocalCluster, Client
        client = Client(LocalCluster(n_workers=8, threads_per_worker=1, memory_limit="0"))
        try:
            for g in [int(x) for x in a.grn_seeds.split(",")]:
                check_grn_seed(g, a.confirm)
                if a.lams is not None:
                    for s_ in [int(x) for x in a.sim_seeds.split(",")]:
                        for lam in sorted(float(x) for x in a.lams.split(",")):
                            run_one(a.out, a.topo, a.world, a.q, lam, None, g, s_, client, None, a.obs, a.depth)
                    continue
                gtargets = a.gtargets or ("0,.15,.30,.45,.60,.75" if a.world in PLIKE else "0,.15,.30")
                tp = tuning_json_path(a.out, a.topo, a.world, a.q, g, a.obs, a.depth)
                tb = validate_tuning_json(tp, a.topo, a.world, a.q, g, a.obs, a.depth)["lambda_for_G"]
                for s_ in [int(x) for x in a.sim_seeds.split(",")]:
                    for gt_ in sorted(float(x) for x in gtargets.split(",")):
                        lam = tb.get(f"{gt_:.2f}")
                        if lam is None:
                            print(f"G target {gt_} unreached for GRN {g}: NO_CALL_G_TARGET"); continue
                        run_one(a.out, a.topo, a.world, a.q, lam, gt_, g, s_, client, tp, a.obs, a.depth)
        finally:
            client.close()
    elif cmd == "pilot":   # python gb_pilot_sharedH_2026_10_21.py pilot <out> [lam] [sim_seeds]
        out_, lam_ = sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else 3.0
        sims_ = [int(x) for x in (sys.argv[4] if len(sys.argv) > 4 else "1,2").split(",")]
        for cond, (lm, sc_) in dict(C0=(0.0, 1.0), C1=(lam_, 1.0), C2=(lam_, 0.2)).items():
            if len(sys.argv) > 5 and cond not in sys.argv[5].split(","):
                continue
            PILOT["mr_dev_scale"] = sc_
            for s_ in sims_:
                try:
                    run_one(f"{out_}/{cond}", "bifurcation", "H", 1, lm, None, 101, s_, None, None, "UMI_DEPTH", 10.0)
                except Exception as e:
                    import traceback; traceback.print_exc(); print("FAILED", cond, s_, repr(e), flush=True)
    elif cmd == "dose":   # dose <out> <m> <sim_seeds>   lam = 3*m, MR deviation scale 1
        out_, m_ = sys.argv[2], float(sys.argv[3]); PILOT["mr_dev_scale"] = 1.0
        for s_ in [int(x) for x in sys.argv[4].split(",")]:
            try:
                run_one(f"{out_}/m{m_:g}", "bifurcation", "H", 1, round(3.0 * m_, 6), None, 101, s_, None, None, "UMI_DEPTH", 10.0)
            except Exception as e:
                import traceback; traceback.print_exc(); print("FAILED", m_, s_, repr(e), flush=True)
    elif cmd in ("ab_harvest", "ab_run"):
        # ab_harvest <out> <a_mod> <c_mod> ; ab_run <out> <cond> <a> <c> <a_mod> <c_mod> <sim> [gen_only]
        out_ = sys.argv[2]; a_mod, c_mod = (float(sys.argv[3]), float(sys.argv[4])) if cmd == "ab_harvest" else (float(sys.argv[6]), float(sys.argv[7]))
        PILOT.update(ab=True, amp_mod=(a_mod, c_mod), mr_dev_scale=1.0)
        if cmd == "ab_harvest":
            PILOT.update(amp=(0.0, 0.0), frozen_half=None); G_ = generate_grn(AB_GRN); os.makedirs(out_, exist_ok=True)
            L_, coef_, q_, info_ = hidden_design(G_, AB_GRN, "AB", 2, TOPO, f"{out_}/orientation_harvest", f"{out_}/harvest_work")
            sim_, _ = build_sim(TOPO, G_, L_, coef_, 1.0, AB_GRN, 0, f"{out_}/harvest_work", "AB", [100, 101]); sim_.calculate_ssConc_()
            fz_ = sorted([int(g), int(r), float(h)] for g in range(sim_.nGenes_) if "params" in sim_.graph_[g] for (r, K, c, h) in sim_.graph_[g]["params"] if r < N_GENES)
            json.dump(dict(a_mod=a_mod, c_mod=c_mod, mr_off=PILOT["mr_off"], half=fz_), open(f"{out_}/frozen_half.json", "w")); print("harvested", len(fz_), sha_file(f"{out_}/frozen_half.json"))
        else:
            cond_, aa_, cc_, sim_seed_ = sys.argv[3], float(sys.argv[4]), float(sys.argv[5]), int(sys.argv[8])
            fzj = json.load(open(f"{out_}/frozen_half.json")); assert (fzj["a_mod"], fzj["c_mod"]) == PILOT["amp_mod"]
            PILOT.update(amp=(aa_, cc_), frozen_half={(t, r): h for t, r, h in fzj["half"]}, frozen_half_sha=sha_file(f"{out_}/frozen_half.json"),
                         gen_only=len(sys.argv) > 9 and sys.argv[9] == "gen_only")
            run_one(f"{out_}/{cond_}", TOPO, "AB", 2, 1.0, None, AB_GRN, sim_seed_, None, None, "UMI_DEPTH", 10.0)
    elif cmd == "smoke":
        ap_ = argparse.ArgumentParser(); ap_.add_argument("cmd"); ap_.add_argument("out"); ap_.add_argument("--world", choices=WORLDS, default="A")
        ap_.add_argument("--tuning-json", required=True); ap_.add_argument("--obs", choices=OBSERVATIONS, default="UMI")
        ap_.add_argument("--depth", type=float, default=10.0); a = ap_.parse_args(); selftest(); smoke(a.out, a.tuning_json, a.world, a.obs, a.depth)
    elif cmd == "regvelo":
        regvelo(sys.argv[2])
    elif cmd == "velorama_to_npz":
        velorama_to_npz(sys.argv[2], sys.argv[3])
    elif cmd == "gpu_eval":
        gpu_eval(sys.argv[2])
    elif cmd == "verify":
        if not verify_dataset(sys.argv[2])["PASS"]:
            raise SystemExit("VERIFY FAILED")
    elif cmd == "report":
        ap_ = argparse.ArgumentParser(); ap_.add_argument("cmd"); ap_.add_argument("out"); ap_.add_argument("--univ", default="PI")
        ap_.add_argument("--required-grid", required=True); a = ap_.parse_args(); report(a.out, a.univ, a.required_grid)
