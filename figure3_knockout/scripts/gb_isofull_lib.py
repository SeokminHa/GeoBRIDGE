"""ISOFULL_2026_10_15 library -- full isochronic-pair scoring experiment (spec internal design note (not shipped) + PI amendment 1).
NO TRUTH (no KO / ChIP label, no `cls` column, no vec `y`) IS READ ANYWHERE IN THIS FILE.  New file; imports (never edits) gb_iso_lib, gb_iso_pilot, gb_primary.
Float64 pair-block engine:  for pairs p=(I_p,J_p), dx = Ms_s[J]-Ms_s[I], dy = Mu_g[J]-Mu_g[I], Q = QR(P30[J]-P30[I])[0][:, :KM] (production R1 path),
q_x(k) = dx - Q_k Q_k^T dx, q_y(k) likewise;  N_k = sum_p q_x q_y,  D_k = sum_p |q_x||q_y|,  S_k = N_k/D_k where D_k > EPS (else NaN).
Projection coefficients Q^T dx = W^T Ms with W = B^T Q (B = signed pair-incidence): N_0 = Ms^T (B^T B) Mu is exact (graph Laplacian form); any covariance reading is an approximation.
usage: python -s gb_isofull_lib.py selftest        (synthetic checks only; run on a server)
"""
import hashlib, json, os, sys, time, socket
import numpy as np
R0 = __import__("os").environ["GEOBRIDGE_ROOT"]
sys.path[:0] = [R0 + '/scratch', R0 + '/pairspace', R0]
import gb_iso_lib as L
import gb_primary as GP

SG = R0 + '/results/signed/'; OUT = SG + 'iso_full_2026_10_15/'; PILOT = SG + 'iso_pilot_2026_10_15/'
TDIR = SG + 'grand_v3d_dumpT_2026_10_15/parts/'; VEC = R0 + '/results/real_uniform/meta_nascent_sutil/full/parts/'; CANON = R0 + '/results/canonical_2026_09_18'
ROOTS = L.ROOTS; EPS = L.EPS; SEEDS = L.SEEDS; WINDOW = L.WINDOW; CAP = L.CAP; KMAIN = 8; NPC = 30
# ---- distance-stratified bins (spec E4 E) and near/mid/far summary groups (amendment 1)
BINS = [(0.0, 0.005), (0.005, 0.01), (0.01, 0.02), (0.02, 0.04), (0.04, 0.08), (0.08, 0.16), (0.16, 0.25)]
GROUPS = {'NEAR': [0, 1, 2], 'MID': [3, 4], 'FAR': [5, 6]}
BIN_MIN_CELLS, BIN_MIN_FRAC_OF_MEDIAN = 100, 0.5                      # spec: infeasible if < 50% of median bin count or < 100 participating cells
STRATUM_MIN_CELLS, STRATUM_MIN_PAIRS = 50, 100                       # amendment 1 per-stratum support rule (frozen before any score)
# ---- systems / units
SYS = {'CARDIACV': dict(focals=['TBX5', 'NKX2-5'], units=['CARDIACV_TBX5', 'CARDIACV_NKX2-5'], strata_cols=['embryo'], ab_tag='batch-embryo'),
       'GATA1': dict(focals=['GATA1', 'KLF1'], units=['GATA1_GATA1', 'KLF1_KLF1', 'KLF1_B_KLF1'], strata_cols=['sample'], ab_tag='random0'),
       'PANC': dict(focals=['RFX6'], units=['PANC_RFX6'], strata_cols=['clusters'], ab_tag='random0')}
UNIT = {'CARDIACV_TBX5': ('CARDIACV', 'TBX5', 'CARDIACV'), 'CARDIACV_NKX2-5': ('CARDIACV', 'NKX2-5', 'CARDIACV'), 'GATA1_GATA1': ('GATA1', 'GATA1', 'GATA1'),
        'KLF1_KLF1': ('GATA1', 'KLF1', 'KLF1'), 'KLF1_B_KLF1': ('GATA1', 'KLF1', 'KLF1_B'), 'PANC_RFX6': ('PANC', 'RFX6', 'PANC')}    # unit -> (system, focal, vec endpoint system)
SCORE_UNITS = [('CARDIACV', 'TBX5'), ('CARDIACV', 'NKX2-5'), ('GATA1', 'GATA1'), ('GATA1', 'KLF1'), ('PANC', 'RFX6')]


def log(*a):
    print(f'[{time.strftime("%H:%M:%S")}]', *a, flush=True)


sha_file = L.sha_file; sha_np = L.sha_np
up = lambda a: np.char.upper(np.asarray(a).astype(str))


def dense(M, dtype=np.float32):
    return np.asarray(M.toarray() if hasattr(M, 'toarray') else M, dtype=dtype)


# ======================================================================================= source lists (control data only)
def source_lists(system):
    """focal -> dict(traj=[10 trajectory-matched alts (iso_freeze, sha 8e4b8996)], expr=[10 expression/detection-matched alts (ext_pseudo roster)])  all UPPER"""
    FZ = json.load(open(PILOT + 'iso_freeze.json')); ros = json.load(open(SG + f'ext_pseudo_2026_10_15/roster_{system}.json')); out = {}
    for f in SYS[system]['focals']:
        out[f] = dict(traj=[x.upper() for x in FZ['sources'][f'{system}/{f}']['alts']], expr=[x.upper() for x in ros['alts'][f]])
    return out


def all_sources(system):
    """focals first, then alternatives (traj then expr), deduplicated; plus membership table"""
    sl = source_lists(system); srcs = []
    for f in SYS[system]['focals']:
        if f not in srcs: srcs.append(f)
    for f in SYS[system]['focals']:
        for x in sl[f]['traj'] + sl[f]['expr']:
            if x not in srcs: srcs.append(x)
    return srcs, sl


# ======================================================================================= fixture context (prep once, load mmap)
def strata_codes(obs, cols):
    import pandas as pd
    if not cols: return np.zeros(len(obs), np.int64), np.array(['all'])
    key = obs[cols].astype(str).agg('|'.join, axis=1); codes, uniq = pd.factorize(key, sort=True)
    return codes.astype(np.int64), np.asarray(uniq).astype(str)


def prep_fixture(h5ad, pairs_npz, axis, strata_cols, outdir, aux_cols=('stage',), require_full_axis=True):
    """Materialise the arrays the scorer needs from a fixture (+ its pair artifact) into `outdir` (.npy, mmap-able).  Truth-free."""
    import anndata as ad, pandas as pd
    t0 = time.time(); os.makedirs(outdir, exist_ok=True)
    F = ad.read_h5ad(h5ad); zp = np.load(pairs_npz, allow_pickle=True); n = F.n_obs; obs_names = np.asarray(F.obs_names).astype(str)
    for r in ROOTS:
        assert (zp[f'{r}_cells'].astype(str) == obs_names).all(), f'pairs cells != fixture obs ({r})'
    vn = up(F.var_names); cnt = pd.Series(vn).value_counts(); fi = {}
    for j, g in enumerate(vn): fi.setdefault(g, j)
    axis = up(axis); ok = np.array([(g in fi) and cnt[g] == 1 for g in axis]); dropped = axis[~ok].tolist()
    if require_full_axis: assert not dropped, ('axis genes absent/ambiguous in fixture', dropped[:5])
    axis = axis[ok]; fcol = np.array([fi[g] for g in axis])
    Mu = dense(F.layers['Mu'][:, fcol], np.float32); Ms = dense(F.layers['Ms'][:, fcol], np.float32)
    spdet = (dense(F.layers['spliced'][:, fcol]) > 0).mean(0); undet = (dense(F.layers['unspliced'][:, fcol]) > 0).mean(0)
    st, stnames = strata_codes(F.obs, strata_cols)
    aux = np.array(['' for _ in range(n)], dtype='U32')
    for c in aux_cols:
        if c in F.obs.columns: aux = np.asarray(F.obs[c].astype(str).to_numpy(), dtype='U32')
    np.save(f'{outdir}/axis.npy', axis); np.save(f'{outdir}/obs.npy', obs_names); np.save(f'{outdir}/Mu.npy', Mu); np.save(f'{outdir}/Ms.npy', Ms)
    np.save(f'{outdir}/P30.npy', np.asarray(F.obsm['X_pca'], np.float64)[:, :NPC]); np.save(f'{outdir}/strata.npy', st); np.save(f'{outdir}/strata_names.npy', stnames); np.save(f'{outdir}/aux_label.npy', aux)
    for r in ROOTS:
        np.save(f'{outdir}/pt_{r}.npy', np.asarray(zp[f'{r}_pt'], np.float64)); np.save(f'{outdir}/I_{r}.npy', np.asarray(zp[f'{r}_I'], np.int64)); np.save(f'{outdir}/J_{r}.npy', np.asarray(zp[f'{r}_J'], np.int64))
    np.savez(f'{outdir}/feat.npz', axis=axis, abund=Ms.mean(0, dtype=np.float64), spdet=spdet, undet=undet, mu_var=Mu.var(0, dtype=np.float64))
    meta = dict(h5ad=h5ad, pairs_npz=pairs_npz, n_cells=int(n), n_axis=int(len(axis)), dropped_axis_genes=dropped, strata_cols=list(strata_cols), n_strata=int(len(stnames)), strata_names=stnames.tolist(),
                strata_sizes=np.bincount(st).tolist(), pairs_per_root={r: int(len(zp[f'{r}_I'])) for r in ROOTS}, prep_seconds=time.time() - t0, host=socket.gethostname())
    json.dump(meta, open(f'{outdir}/meta.json', 'w'), indent=1); return meta


class Ctx:
    """mmap view of a prepped fixture"""
    def __init__(self, d):
        self.d = d; self.meta = json.load(open(d + '/meta.json')); self.axis = np.load(d + '/axis.npy'); self.obs = np.load(d + '/obs.npy')
        self.Mu = np.load(d + '/Mu.npy', mmap_mode='r'); self.Ms = np.load(d + '/Ms.npy', mmap_mode='r'); self.P30 = np.load(d + '/P30.npy')
        self.strata = np.load(d + '/strata.npy'); self.strata_names = np.load(d + '/strata_names.npy'); self.aux = np.load(d + '/aux_label.npy')
        self.pt = {r: np.load(d + f'/pt_{r}.npy') for r in ROOTS}; self.fwd = {r: (np.load(d + f'/I_{r}.npy'), np.load(d + f'/J_{r}.npy')) for r in ROOTS}
        self.feat = dict(np.load(d + '/feat.npz')); self.n = len(self.obs); self.G = len(self.axis); self.ai = {g: i for i, g in enumerate(self.axis)}

    def src_idx(self, srcs):
        miss = [s for s in srcs if s not in self.ai]; assert not miss, ('sources absent from axis', miss)
        return np.array([self.ai[s] for s in srcs])

    def Xs(self, srcs):
        return np.asarray(self.Ms[:, self.src_idx(srcs)], np.float64)


# ======================================================================================= pair constructions
def window_pairs(pt, strata, lo, hi, cap=CAP, seed=42, root_idx=0):
    """Within-stratum UNORDERED pairs with lo < |pt_j - pt_i| <= hi (lo == 0: the closed iso rule, i.e. L.iso_pairs(window=hi)).  <= cap partners per cell sampled uniformly without
    replacement from the two-sided candidate list (lower side first), RNG default_rng([seed, root_idx]), cells in ascending index order; edges reduced to unordered (min,max), deduplicated, sorted (I<J)."""
    pt = np.asarray(pt, np.float64); strata = np.asarray(strata); n = len(pt)
    if lo == 0:
        return L.iso_pairs(pt, strata, window=hi, cap=cap, seed=seed, root_idx=root_idx)
    alo = np.zeros(n, np.int64); ahi = np.zeros(n, np.int64); blo = np.zeros(n, np.int64); bhi = np.zeros(n, np.int64); order_of = {}
    for s in np.unique(strata):
        idx = np.flatnonzero(strata == s); o = idx[np.lexsort((idx, pt[idx]))]; p = pt[o]
        alo[o] = np.searchsorted(p, p - hi, 'left'); ahi[o] = np.searchsorted(p, p - lo, 'left'); blo[o] = np.searchsorted(p, p + lo, 'right'); bhi[o] = np.searchsorted(p, p + hi, 'right'); order_of[s] = o
    rng = np.random.default_rng([int(seed), int(root_idx)]); A, B = [], []
    for i in range(n):
        nl = int(ahi[i] - alo[i]); nu = int(bhi[i] - blo[i]); m = nl + nu
        if m <= 0: continue
        offs = np.arange(m) if m <= cap else rng.choice(m, cap, replace=False)
        o = order_of[strata[i]]; pos = np.where(offs < nl, alo[i] + offs, blo[i] + (offs - nl))
        A.append(np.full(len(pos), i, np.int64)); B.append(o[pos])
    if not A: return np.zeros(0, np.int64), np.zeros(0, np.int64)
    a = np.concatenate(A); b = np.concatenate(B); key = np.unique(np.minimum(a, b) * n + np.maximum(a, b))
    return (key // n).astype(np.int64), (key % n).astype(np.int64)


def fwd_pairs_restricted(pt, cell_ids, strata, lo=0.02, hi=0.25, cap=CAP, seed=42):
    """PRODUCTION FORWARD BUILDER LOGIC (gb_gata1_preflight.canonical_pairs -> geobridge_model.pairspace.build_forward_pairs: canonical stable-id order, then rows sorted by pt with canonical-index
    tie-break; for each sorted position i the admissible partners are positions in [searchsorted(pt, pt_i+lo, 'right'), searchsorted(pt, pt_i+hi, 'right')); if more than `cap`, rng.choice(js, cap,
    replace=False) with ONE np.random.default_rng(seed) shared over all i) with the candidate list RESTRICTED to the same stratum as cell i.  One stratum => identical (I,J) to the production artifact.
    Returns (I,J) in ORIGINAL cell indices, I = earlier cell (pt_J > pt_I), sorted lexicographically by (I,J) like the canonical artifact."""
    pt = np.asarray(pt, np.float64); n = len(pt); strata = np.asarray(strata)
    can = np.argsort(np.asarray(cell_ids).astype(str), kind='stable'); ptc = pt[can]            # canonical row c <-> original cell can[c]; pt of canonical row c = pt[can[c]]
    o2 = np.lexsort((np.arange(n), ptc)); ps = ptc[o2]; orig = can[o2]; st = strata[orig]
    start = np.searchsorted(ps, ps + lo, 'right'); start = np.maximum(start, np.arange(n) + 1); end = np.searchsorted(ps, ps + hi, 'right'); end = np.maximum(end, start)
    pos_by = {s: np.flatnonzero(st == s) for s in np.unique(st)}                                  # sorted positions per stratum (ascending)
    rng = np.random.default_rng(int(seed)); A, B = [], []
    for i in range(n - 1):
        s_, e_ = int(start[i]), int(end[i])
        if s_ >= e_: continue
        pa = pos_by[st[i]]; js = pa[np.searchsorted(pa, s_, 'left'):np.searchsorted(pa, e_, 'left')]
        if js.size == 0: continue
        if js.size > cap: js = rng.choice(js, size=cap, replace=False)
        A.append(np.full(js.size, orig[i], np.int64)); B.append(orig[js])
    a = np.concatenate(A); b = np.concatenate(B); o = np.lexsort((b, a))
    return a[o], b[o]


def subsample_pairs(I, J, n_keep, seed=42):
    """uniform subset of exactly n_keep pairs without replacement (sorted to keep the canonical (I,J) order)"""
    if n_keep >= len(I): return I.copy(), J.copy()
    sel = np.sort(np.random.default_rng([int(seed), 99]).choice(len(I), int(n_keep), replace=False)); return I[sel], J[sel]


def thin_match(I, J, n_cells, tgt_deg, n_target, seed=42):
    """Thin a pair set to <= n_target pairs approximating a TARGET per-cell degree distribution.  Per-cell caps: participating cells of (I,J) ranked by their degree (random tie-break); the cell of fractional
    rank u gets cap = the u-quantile of the NONZERO target degrees.  Edges visited in random order; kept iff both endpoints are below their caps; stops at n_target.  Returns (I2,J2,info)."""
    rng = np.random.default_rng([int(seed), 123]); deg = np.bincount(np.concatenate([I, J]), minlength=n_cells); part = np.flatnonzero(deg > 0)
    td = np.sort(tgt_deg[tgt_deg > 0]); m = len(part)
    if m == 0 or len(td) == 0: return I[:0], J[:0], dict(reached=0, n_target=int(n_target), note='empty')
    ordr = part[np.lexsort((rng.random(m), deg[part]))]; u = np.arange(m) / max(m - 1, 1); cap = np.zeros(n_cells, np.int64)
    cap[ordr] = np.maximum(1, np.round(np.interp(u, np.linspace(0, 1, len(td)), td)).astype(np.int64))
    perm = rng.permutation(len(I)).tolist(); Il = I.tolist(); Jl = J.tolist(); n_target = int(min(n_target, len(I))); cap0 = cap.copy()
    for scale in (1.0, 1.25, 1.5, 2.0, 3.0, 5.0, 1e9):                        # relax the quantile caps until the target pair count is reachable
        capl = np.minimum(cap0 * scale, 10 ** 9).astype(np.int64).tolist(); cl = [0] * n_cells; keep = np.zeros(len(I), bool); got = 0
        for p in perm:
            a, b = Il[p], Jl[p]
            if cl[a] < capl[a] and cl[b] < capl[b]:
                cl[a] += 1; cl[b] += 1; keep[p] = True; got += 1
                if got >= n_target: break
        if got >= n_target: break
    I2, J2 = I[keep], J[keep]; o = np.lexsort((J2, I2)); I2, J2 = I2[o], J2[o]
    d2 = np.bincount(np.concatenate([I2, J2]), minlength=n_cells); d2 = d2[d2 > 0]
    qs = [0.05, .25, .5, .75, .95]
    return I2, J2, dict(reached=int(got), cap_scale=float(scale), n_target=int(n_target), shortfall=int(max(0, n_target - got)), deg_q_target=np.quantile(td, qs).tolist(), deg_q_achieved=np.quantile(d2, qs).tolist() if len(d2) else None,
                        ks_degree=float(_ks(td, d2)) if len(d2) else None)


def _ks(a, b):
    a = np.sort(a); b = np.sort(b); allv = np.concatenate([a, b]); return float(np.abs(np.searchsorted(a, allv, 'right') / len(a) - np.searchsorted(b, allv, 'right') / len(b)).max())


def flip_orientation(I, J, frac=0.5, seed=0, permute=True):
    """swap (I_p, J_p) for a random `frac` of the pairs (default 50%, seed 0) and permute the pair order"""
    rng = np.random.default_rng(int(seed)); m = rng.random(len(I)) < frac; I2 = np.where(m, J, I); J2 = np.where(m, I, J)
    if permute: pm = rng.permutation(len(I)); I2, J2 = I2[pm], J2[pm]
    return I2, J2


def build_bins(pt, strata, root_idx, seed=42, cap=CAP, cell_mask=None):
    """E-bin pair sets for one root.  cell_mask (bool over cells) restricts to a subset of cells (e.g. common-support strata).  Returns dict(bins=[(I,J) full cap-40 sets], count=[...], n_cells=[...],
    feasible=[...], nstar, matched=[(I,J) or None], groups={NEAR/MID/FAR: (I,J) or None}, group_n)."""
    n = len(pt); sel = np.arange(n) if cell_mask is None else np.flatnonzero(cell_mask); out = dict(bins=[], count=[], n_cells=[], feasible=[], matched=[], nstar=None, groups={}, group_n={})
    for (lo, hi) in BINS:
        I, J = window_pairs(pt[sel], strata[sel], lo, hi, cap, seed, root_idx); I, J = sel[I], sel[J]
        out['bins'].append((I, J)); out['count'].append(int(len(I))); out['n_cells'].append(int(len(np.unique(np.concatenate([I, J]))) if len(I) else 0))
    med = float(np.median(out['count']))
    out['feasible'] = [bool(c >= BIN_MIN_FRAC_OF_MEDIAN * med and nc >= BIN_MIN_CELLS) for c, nc in zip(out['count'], out['n_cells'])]
    if any(out['feasible']):
        out['nstar'] = int(min(c for c, f in zip(out['count'], out['feasible']) if f))
        for b, f in enumerate(out['feasible']):
            out['matched'].append(subsample_pairs(*out['bins'][b], out['nstar'], seed + b) if f else None)
        sizes = {}
        for g, bl in GROUPS.items():
            parts = [out['matched'][b] for b in bl if out['matched'][b] is not None]
            if len(parts) == len(bl): sizes[g] = (np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts]))
            else: sizes[g] = None
        have = [len(v[0]) for v in sizes.values() if v is not None]
        if have:
            ng = min(have); out['group_n'] = {g: (int(ng) if v is not None else None) for g, v in sizes.items()}
            for gi, (g, v) in enumerate(sizes.items()):
                if v is None: out['groups'][g] = None; continue
                I, J = subsample_pairs(v[0], v[1], ng, seed + 50 + gi); o = np.lexsort((J, I)); out['groups'][g] = (I[o], J[o])
        else: out['groups'] = {g: None for g in GROUPS}
    else: out['matched'] = [None] * len(BINS); out['groups'] = {g: None for g in GROUPS}
    return out


def stratum_support(pt, strata, strata_names, bin_pairs, rootname):
    """per stratum: pt range and, per bin / near-mid-far group, participating cells and pairs; supported iff cells >= STRATUM_MIN_CELLS and pairs >= STRATUM_MIN_PAIRS (else NOT_IDENTIFIABLE)"""
    rows = []
    for s, nm in enumerate(strata_names):
        m = strata == s
        if not m.any(): continue
        r = dict(root=rootname, stratum=str(nm), n_cells=int(m.sum()), pt_min=float(pt[m].min()), pt_max=float(pt[m].max()), pt_range=float(pt[m].max() - pt[m].min()))
        cells_b = []; pairs_b = []
        for b, (I, J) in enumerate(bin_pairs):
            k = strata[I] == s if len(I) else np.zeros(0, bool); pc = int(k.sum()); cc = int(len(np.unique(np.concatenate([I[k], J[k]])))) if pc else 0
            cells_b.append(cc); pairs_b.append(pc); r[f'bin{b + 1}_pairs'] = pc; r[f'bin{b + 1}_cells'] = cc
            r[f'bin{b + 1}_status'] = 'supported' if (cc >= STRATUM_MIN_CELLS and pc >= STRATUM_MIN_PAIRS) else 'NOT_IDENTIFIABLE'
        for g, bl in GROUPS.items():
            r[f'{g}_status'] = 'supported' if all(r[f'bin{b + 1}_status'] == 'supported' for b in bl) else 'NOT_IDENTIFIABLE'
        r['all_bins_supported'] = bool(all(r[f'bin{b + 1}_status'] == 'supported' for b in range(len(BINS)))); r['near_mid_far_supported'] = bool(all(r[f'{g}_status'] == 'supported' for g in GROUPS))
        rows.append(r)
    return rows


# ======================================================================================= engine
def engine(Xs, Y, P30, I, J, km=KMAIN, block=65536, gchunk=256, want_cent=False, self_idx=None, nthreads=None, verbose=False):
    """Float64 scoring engine.  Xs (n,ns) float64 sources; Y (n,G) float32 (mmap ok) targets; returns dict(N,D:(km+1,ns,G), sxx,syy, qxe:(km+1,ns), Cx:(km,ns), Cy:(km,G), sing(30), colnorm(30), Rdiag(30), P, zero_dx(ns), [Nc,Dc:(ns,G) pair-centred k=0]).
    Pair order and orientation do not matter (M1).  Self-target entries (self_idx[j] = target column of source j, -1 if none) are NaN."""
    import numexpr as ne
    if nthreads: ne.set_num_threads(int(nthreads))
    t0 = time.time(); Xs = np.asarray(Xs, np.float64); n, ns = Xs.shape; G = Y.shape[1]; I = np.asarray(I, np.int64); J = np.asarray(J, np.int64); P = len(I)
    assert P >= NPC, ('too few pairs for the 30-column QR', P)
    dP = P30[J] - P30[I]; Qf, Rm = np.linalg.qr(dP); colnorm = np.linalg.norm(dP, axis=0); Rdiag = np.abs(np.diag(Rm)); sing = np.linalg.svd(Rm, compute_uv=False)
    Q = np.ascontiguousarray(Qf[:, :km]); del Qf, dP
    W = np.empty((n, km))
    for j in range(km): W[:, j] = np.bincount(J, weights=Q[:, j], minlength=n) - np.bincount(I, weights=Q[:, j], minlength=n)
    Cx = W.T @ Xs
    qxe = np.zeros((km + 1, ns)); zero_dx = np.zeros(ns)
    for b0 in range(0, P, block):
        bs = slice(b0, min(P, b0 + block)); dx = Xs[J[bs]] - Xs[I[bs]]; Qb = Q[bs]; r = dx; qxe[0] += (r * r).sum(0); zero_dx += (dx == 0).sum(0)
        for k in range(1, km + 1):
            r = r - Qb[:, k - 1:k] * Cx[k - 1]; qxe[k] += (r * r).sum(0)
    N = np.zeros((km + 1, ns, G)); D = np.zeros_like(N); syy = np.zeros(G); Cy = np.zeros((km, G)); dcnt = (np.bincount(J, minlength=n) - np.bincount(I, minlength=n)).astype(np.float64)
    if want_cent: Nc = np.zeros((ns, G)); Dc = np.zeros((ns, G)); mx = (dcnt @ Xs) / P; my_all = np.zeros(G)
    for g0 in range(0, G, gchunk):
        sl = slice(g0, min(G, g0 + gchunk)); Yc = np.ascontiguousarray(Y[:, sl], dtype=np.float32); Yc64 = Yc.astype(np.float64); Cyc = W.T @ Yc64; Cy[:, sl] = Cyc; g = Yc.shape[1]
        if want_cent: my = (dcnt @ Yc64) / P; my_all[sl] = my
        ary = np.empty((min(block, P), g))
        for b0 in range(0, P, block):
            bs = slice(b0, min(P, b0 + block)); Ib, Jb = I[bs], J[bs]; Qb = Q[bs]; B = len(Ib)
            ry = Yc[Jb].astype(np.float64); ry -= Yc[Ib]; rx = Xs[Jb] - Xs[Ib]
            syy[sl] += np.einsum('ij,ij->j', ry, ry)
            if want_cent:
                dxc = rx - mx; dyc = ry - my; Nc[:, sl] += dxc.T @ dyc; Dc[:, sl] += np.abs(dxc).T @ np.abs(dyc); del dxc, dyc
            ab = ary[:B]
            for k in range(km + 1):
                if k:
                    rx = rx - Qb[:, k - 1:k] * Cx[k - 1]; q = Qb[:, k - 1:k]; c = Cyc[k - 1][None, :]
                    ne.evaluate('ry - q * c', local_dict={'ry': ry, 'q': q, 'c': c}, out=ry, casting='same_kind')
                ne.evaluate('abs(ry)', local_dict={'ry': ry}, out=ab, casting='same_kind')
                N[k][:, sl] += rx.T @ ry; D[k][:, sl] += np.abs(rx).T @ ab
        if verbose: log('engine chunk', g0, G, f'{time.time() - t0:.0f}s')
    if self_idx is not None:
        for j, ix in enumerate(self_idx):
            if ix >= 0: N[:, j, ix] = np.nan; D[:, j, ix] = np.nan
    res = dict(N=N, D=D, sxx=qxe[0].copy(), syy=syy, qxe=qxe, Cx=Cx, Cy=Cy, sing=sing, colnorm=colnorm, Rdiag=Rdiag, P=P, zero_dx=zero_dx, runtime=time.time() - t0)
    if want_cent:
        if self_idx is not None:
            for j, ix in enumerate(self_idx):
                if ix >= 0: Nc[j, ix] = np.nan; Dc[j, ix] = np.nan
        res.update(Nc=Nc, Dc=Dc, mx=mx, my=my_all)
    return res


def ratio(N, D):
    ok = D > EPS
    return np.where(ok, N / np.where(ok, D, 1.0), np.nan)


def pair_pearson(N0, sxx, syy):
    den = np.sqrt(sxx)[:, None] * np.sqrt(syy)[None, :]; ok = den > EPS
    return np.where(ok, N0 / np.where(ok, den, 1.0), np.nan)


def laplacian_N0(Xs, Y, I, J, n):
    """independent exact evaluation  N_0 = Ms^T (B^T B) Mu  with the pair-graph Laplacian L = diag(deg) - (A + A^T) (multiplicity weights)"""
    import scipy.sparse as sp
    A = sp.coo_matrix((np.ones(len(I)), (I, J)), shape=(n, n)).tocsr(); Asym = A + A.T; deg = np.asarray(Asym.sum(1)).ravel()
    LY = deg[:, None] * np.asarray(Y, np.float64) - Asym @ np.asarray(Y, np.float64)
    return np.asarray(Xs, np.float64).T @ LY


# ======================================================================================= aggregation
def seed_mean(arrs):
    st = np.stack(arrs); return np.where(np.isfinite(st).all(0), st.mean(0), np.nan)


def root_mean(a, b):
    return np.where(np.isfinite(a) & np.isfinite(b), (a + b) / 2, np.nan)


def zc(Sroot, j, ui, ks=GP.KS):
    """PRIMARY aggregator for source j: Sroot = {root: (km+1, ns, G) signed S}; ks = depths averaged (GP.KS = 0..8; (0,) = single depth) -> (G,) with NaN outside ui"""
    G = Sroot[ROOTS[0]].shape[2]; o = np.full(G, np.nan)
    o[ui] = GP.zcons({h: np.abs(Sroot[h][:, j, :][:, ui].T) for h in ROOTS}, ks); return o


def frac_rank(z, m):
    from scipy.stats import rankdata
    r = np.full(len(z), np.nan); r[m] = (rankdata(z[m]) - 0.5) / m.sum(); return r


def sgn(x):
    x = np.asarray(x, float); return np.where(np.isfinite(x), np.sign(x), np.nan)


# ======================================================================================= pair-set diagnostics (E5)
def graph_diag(I, J, n, pt, strata, strata_names, eig=True, eig_maxiter=3000, eig_k=10):
    import scipy.sparse as sp
    from scipy.sparse.csgraph import connected_components
    d = L.degree_stats(I, J, n); P = len(I); dt = np.abs(pt[J] - pt[I]); q = np.quantile(dt, [0, .05, .25, .5, .75, .95, 1.0]) if P else [np.nan] * 7
    d.update(n_pairs=int(P), mean_abs_dt=float(dt.mean()) if P else np.nan, **{f'abs_dt_q{int(x * 100):03d}': float(v) for x, v in zip([0, .05, .25, .5, .75, .95, 1.0], q)}, frac_cross_strata=float((strata[I] != strata[J]).mean()) if P else np.nan)
    A = sp.coo_matrix((np.ones(P), (I, J)), shape=(n, n)).tocsr(); A = (A + A.T).tocsr(); deg = np.asarray(A.sum(1)).ravel(); part = deg > 0
    ncomp, lab = connected_components(A, directed=False); cs = np.bincount(lab[part], minlength=ncomp); cs = cs[cs > 0]
    d.update(n_components=int(len(cs)), largest_component=int(cs.max()) if len(cs) else 0, component_sizes_top5=sorted(cs.tolist(), reverse=True)[:5], participating_cells=int(part.sum()), frac_cells_participating=float(part.mean()))
    d['participation_by_stratum'] = {str(nm): float(part[strata == s].mean()) for s, nm in enumerate(strata_names) if (strata == s).any()}
    d['eig_smallest_nonzero_normalised_laplacian'] = None; d['eig_note'] = 'not computed'
    if eig and part.sum() > 50:
        try:
            from scipy.sparse.linalg import eigsh, ArpackNoConvergence
            idx = np.flatnonzero(part); Ap = A[idx][:, idx].tocsr(); dp = np.asarray(Ap.sum(1)).ravel(); Dm = sp.diags(1.0 / np.sqrt(dp)); M = (Dm @ Ap @ Dm).tocsr(); k = int(min(len(cs) + eig_k + 2, len(idx) - 2))
            try:
                w = eigsh(M, k=k, which='LA', tol=1e-4, maxiter=eig_maxiter, return_eigenvectors=False); note = 'converged'
            except ArpackNoConvergence as e:
                w = e.eigenvalues; note = f'NOT_CONVERGED(maxiter={eig_maxiter}); {len(w)} eigenvalues returned'
            lam = np.sort(1.0 - np.asarray(w)); nz = lam[lam > 1e-8][:eig_k]; d['eig_smallest_nonzero_normalised_laplacian'] = nz.tolist(); d['eig_note'] = note + f'; zero eigenvalues dropped={int((lam <= 1e-8).sum())} (components={len(cs)}); not temporal frequencies'
        except Exception as e:
            d['eig_note'] = f'eigsh failed: {type(e).__name__}: {e}'
    return d


# ======================================================================================= job I/O
def save_job(path, res, extra=None, meta=None):
    """write engine result (+ named extras) as npz and a sidecar json with the sha256 and meta; returns sha"""
    os.makedirs(os.path.dirname(path), exist_ok=True); arrs = {k: v for k, v in res.items() if isinstance(v, np.ndarray)}; scal = {k: v for k, v in res.items() if not isinstance(v, np.ndarray)}
    if extra: arrs.update(extra)
    np.savez_compressed(path, **arrs); h = sha_file(path)
    json.dump(dict(sha256=h, scalars={k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in scal.items()}, meta=meta or {}, host=socket.gethostname(), script_sha256=sha_file(os.path.abspath(__file__)), written=time.strftime('%Y-%m-%dT%H:%M:%S')),
              open(path + '.json', 'w'), indent=1, default=str)
    return h


def load_job(path):
    z = np.load(path, allow_pickle=True); d = {k: z[k] for k in z.files}
    if os.path.exists(path + '.json'):
        for k, v in json.load(open(path + '.json')).get('scalars', {}).items(): d.setdefault(k, np.array(v))
    return d


# ======================================================================================= selftest
def selftest():
    rng = np.random.default_rng(0)
    # 1. engine vs naive explicit projector, orientation/row-order invariance (M1), Laplacian identity (M2)
    n, ns, G, P = 300, 3, 40, 2500; X = rng.normal(size=(n, ns)); Y = rng.normal(size=(n, G)).astype(np.float32); P30 = rng.normal(size=(n, 30))
    I = rng.integers(0, n, P); J = rng.integers(0, n, P); m = I == J; J[m] = (J[m] + 1) % n
    r = engine(X, Y, P30, I, J, km=8, block=700, gchunk=13, want_cent=True, self_idx=np.array([0, 5, -1]))
    dx = X[J] - X[I]; dy = Y[J].astype(np.float64) - Y[I].astype(np.float64); Q = np.linalg.qr(P30[J] - P30[I])[0]
    for k in range(9):
        H = Q[:, :k] @ Q[:, :k].T; qx = dx - H @ dx; qy = dy - H @ dy; Nn = qx.T @ qy; Dn = np.abs(qx).T @ np.abs(qy)
        a = r['N'][k].copy(); b = r['D'][k].copy(); Nn2 = Nn.copy(); Dn2 = Dn.copy(); a[0, 0] = Nn2[0, 0] = b[0, 0] = Dn2[0, 0] = 0; a[1, 5] = Nn2[1, 5] = b[1, 5] = Dn2[1, 5] = 0
        assert np.isnan(r['N'][k][0, 0]) and np.isnan(r['N'][k][1, 5]) and not np.isnan(r['N'][k][2, 0])
        assert np.allclose(a, Nn2, atol=1e-9, rtol=1e-9) and np.allclose(b, Dn2, atol=1e-9, rtol=1e-9), k
        e = ((qx ** 2).sum(0)); assert np.allclose(r['qxe'][k], e, rtol=1e-10)
    assert np.allclose(r['syy'], (dy ** 2).sum(0)) and np.allclose(r['sxx'], (dx ** 2).sum(0))
    Cyn = Q[:, :8].T @ dy; assert np.allclose(r['Cy'], Cyn, atol=1e-9) and np.allclose(r['Cx'], Q[:, :8].T @ dx, atol=1e-9)
    # centred (d4 implemented) vs naive
    dxc = dx - dx.mean(0); dyc = dy - dy.mean(0); Nc = dxc.T @ dyc; Dc = np.abs(dxc).T @ np.abs(dyc); Nc[0, 0] = Dc[0, 0] = Nc[1, 5] = Dc[1, 5] = 0; a = r['Nc'].copy(); b = r['Dc'].copy(); a[0, 0] = b[0, 0] = a[1, 5] = b[1, 5] = 0
    assert np.allclose(a, Nc, atol=1e-9) and np.allclose(b, Dc, atol=1e-9)
    # flip + permute
    If, Jf = flip_orientation(I, J, 0.5, 0); r2 = engine(X, Y, P30, If, Jf, km=8, block=999, gchunk=17, self_idx=np.array([0, 5, -1]))
    Sa = ratio(r['N'], r['D']); Sb = ratio(r2['N'], r2['D']); assert np.nanmax(np.abs(Sa - Sb)) < 1e-9 and (np.isnan(Sa) == np.isnan(Sb)).all(), np.nanmax(np.abs(Sa - Sb))
    assert (np.abs(Nc - 0) > 0).any()
    # centred is orientation-DEPENDENT for a one-sided oriented set (documented property, not asserted numerically beyond difference > 0 in this random set)
    # Laplacian identity
    NL = laplacian_N0(X, Y, I, J, n); assert np.allclose(NL, dx.T @ dy, atol=1e-8, rtol=1e-9)
    # symmetric (both orientations) pair set: centred == uncentred ratio
    Is = np.concatenate([I, J]); Js = np.concatenate([J, I]); rs = engine(X, Y, P30, Is, Js, km=8, block=1111, gchunk=19, want_cent=True)
    r0 = engine(X, Y, P30, I, J, km=8, want_cent=False)
    assert np.allclose(ratio(rs['Nc'], rs['Dc']), ratio(r0['N'][0], r0['D'][0]), atol=1e-10)
    # 2. window_pairs vs brute force, lo==0 delegates to iso, caps, determinism
    nn = 400; pt = rng.random(nn); pt[:20] = pt[20:40]; st = rng.integers(0, 3, nn); lo_, hi_ = 0.03, 0.12
    I1, J1 = window_pairs(pt, st, lo_, hi_, cap=10 ** 6, seed=1, root_idx=0); bf = set()
    for i in range(nn):
        for j in range(nn):
            if i != j and st[i] == st[j] and ((pt[j] >= pt[i] - hi_ and pt[j] < pt[i] - lo_) or (pt[j] > pt[i] + lo_ and pt[j] <= pt[i] + hi_)): bf.add((min(i, j), max(i, j)))
    assert set(zip(I1.tolist(), J1.tolist())) == bf and (I1 < J1).all(), (len(bf), len(I1))
    Ic, Jc = window_pairs(pt, st, lo_, hi_, cap=4, seed=1, root_idx=0); Id, Jd = window_pairs(pt, st, lo_, hi_, cap=4, seed=1, root_idx=0)
    assert np.array_equal(Ic, Id) and set(zip(Ic.tolist(), Jc.tolist())) <= bf
    Ie, Je = window_pairs(pt, st, 0.0, 0.05, cap=5, seed=3, root_idx=1); Ig, Jg = L.iso_pairs(pt, st, 0.05, 5, 3, 1); assert np.array_equal(Ie, Ig) and np.array_equal(Je, Jg)
    # 3. fwd_pairs_restricted: single stratum == production builder (geobridge_model.pairspace.build_forward_pairs); restricted == brute force with the same RNG draws
    from geobridge_model.pairspace import build_forward_pairs
    nf = 500; ptf = rng.random(nf); ids = np.array([f'c{rng.integers(10 ** 8)}_{i}' for i in range(nf)]); Xd = rng.normal(size=(nf, 2))
    can = np.argsort(ids.astype(str), kind='stable'); o2 = np.lexsort((np.arange(nf), ptf[can])); pr = build_forward_pairs(Xd[can][o2], Xd[can][o2], ptf[can][o2], min_forward_dt=0.02, max_forward_dt=0.25, max_pairs_per_cell=7, pair_seed=42)
    Ip = can[o2[np.asarray(pr.pair_i, int)]]; Jp = can[o2[np.asarray(pr.pair_j, int)]]; oo = np.lexsort((Jp, Ip)); Ip, Jp = Ip[oo], Jp[oo]
    Ir, Jr = fwd_pairs_restricted(ptf, ids, np.zeros(nf, int), 0.02, 0.25, 7, 42); assert np.array_equal(Ip, Ir) and np.array_equal(Jp, Jr), 'single-stratum restricted builder != production builder'
    stf = rng.integers(0, 3, nf); Ir2, Jr2 = fwd_pairs_restricted(ptf, ids, stf, 0.02, 0.25, 6, 42)
    assert (stf[Ir2] == stf[Jr2]).all() and (ptf[Jr2] - ptf[Ir2] > 0.02).all() and (ptf[Jr2] - ptf[Ir2] <= 0.25).all() and np.bincount(Ir2, minlength=nf).max() <= 6
    # 4. subsample / thin / bins invariants
    Is_, Js_ = subsample_pairs(Ir, Jr, 100, 1); assert len(Is_) == 100 and len(set(zip(Is_.tolist(), Js_.tolist()))) == 100
    tgt = np.bincount(np.concatenate([Ie, Je]), minlength=nn); It, Jt, info = thin_match(Ir, Jr, nf if nf > nn else nn, np.pad(tgt, (0, max(0, nf - nn))), 150, 1); assert len(It) <= 150 and (np.bincount(np.concatenate([It, Jt]), minlength=max(nf, nn)) >= 0).all()
    bb = build_bins(rng.random(3000), rng.integers(0, 2, 3000), 0, seed=42, cap=40); assert len(bb['bins']) == 7
    if bb['nstar'] is not None:
        for m_ in bb['matched']:
            if m_ is not None: assert len(m_[0]) == bb['nstar']
    # 5. flip determinism
    a, b = flip_orientation(I, J, 0.5, 0); c, d_ = flip_orientation(I, J, 0.5, 0); assert np.array_equal(a, c) and np.array_equal(b, d_)
    print('gb_isofull_lib selftest OK')


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == 'selftest': selftest()
