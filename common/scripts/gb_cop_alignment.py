#!/usr/bin/env python
# Compare score profiles at matched C_op values.
import glob, hashlib, json, os, sys, time
import numpy as np, pandas as pd

R = __import__("os").environ["GEOBRIDGE_ROOT"]; sys.path.insert(0, f"{R}/scratch")
O = f"{R}/results/cop_alignment"; ROOTS = ("dc1_min", "dc1_max"); KS = np.arange(31)
PREREG_SHA = "1f360e218b285a4106df157f829e60d933887636f8dc1539774e72a08a517215"
BANDS = {"C[.10,.20]": (.10, .20), "C[.20,.30]": (.20, .30), "C[.30,.40]": (.30, .40), "C[.20,.40]": (.20, .40),
         "C[.30,.50]": (.30, .50), "C[.40,.60]": (.40, .60)}
FIXED = {"k1-5": list(range(1, 6)), "v1 k8-30": list(range(8, 31)), "k0": [0]}
CGRID = np.round(np.arange(0, 1.0001, 0.025), 3)
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()


def pct(v):
    o = np.full(len(v), np.nan); ok = np.isfinite(v); o[ok] = pd.Series(v[ok]).rank(method="average").to_numpy() / ok.sum(); return o


def axis(system):
    from gb_substrate import load
    import anndata as ad
    S = load(system, canonical=True); p = f"{R}/results/canon_{system}/per_gene.npz"
    eg = np.asarray(np.load(p, allow_pickle=True)["genes"] if os.path.exists(p) else S["genes"]).astype(str)
    gi = {g: i for i, g in enumerate(np.asarray(S["genes"]).astype(str))}
    return S, eg, np.array([gi[g] for g in eg])


# identical algebra to gb_regulon_panel.score (production project_scores), raw per-k U/X absS for the given sources
def score(system, srcs):
    from gb_target_extension import project_scores, CHUNK, NPC
    import anndata as ad
    t0 = time.time(); S, eg, col = axis(system); gu = np.char.upper(eg)
    F = ad.read_h5ad(S["fixture"]); fm = {g: i for i, g in enumerate(np.asarray(F.var_names).astype(str))}; fc = [fm[g] for g in eg]
    d_ = lambda M: np.asarray(M.todense()) if hasattr(M, "todense") else np.asarray(M)
    Mu = d_(F.layers["Mu"][:, fc]).astype(np.float32); Ms = d_(F.layers["Ms"][:, fc]).astype(np.float32)
    srcs = [s.upper() for s in srcs]; pos = {g: i for i, g in enumerate(gu)}
    miss = [s for s in srcs if s not in pos]
    if miss:
        raise SystemExit(f"REFUSING: sources not on the axis: {miss}")
    si = [pos[s] for s in srcs]; Xsrc = np.asarray(S["X"], np.float64)[:, col[si]]
    assert np.allclose(Xsrc, Ms[:, si], rtol=1e-5, atol=1e-6), "V1 source Ms != fixture Ms"
    P30 = np.asarray(F.obsm["X_pca"], float)[:, :NPC]; G = len(eg)
    raw = {(r, c): np.full((len(srcs), G, 31), np.nan, np.float32) for r in ROOTS for c in ("U", "X")}
    for r in ROOTS:
        I, J = S["pairs"][r]; Qf, _ = np.linalg.qr(P30[J] - P30[I]); dx = (Xsrc[J] - Xsrc[I]).astype(np.float32)
        for g0 in range(0, G, CHUNK):
            sl = slice(g0, min(g0 + CHUNK, G))
            for c, M in (("U", Mu), ("X", Ms)):
                for k, _n, _d, _z, absS in project_scores(dx, (M[J, sl] - M[I, sl]).astype(np.float32), Qf):
                    raw[(r, c)][:, sl, k] = absS
        print(f"  {system} {r}: {len(I)} pairs {time.time()-t0:.0f}s", flush=True)
    for j, s in enumerate(si):
        for a in raw.values():
            a[j, s, :] = np.nan
    os.makedirs(f"{O}/scores", exist_ok=True); out = f"{O}/scores/{system}.npz"
    np.savez_compressed(out, genes=eg, sources=np.asarray(srcs), **{f"{r}|{c}": a for (r, c), a in raw.items()})
    json.dump(dict(system=system, out=out, sha256=sha(out), script_sha256=sha(os.path.abspath(__file__)), truth_read=False,
                   fixture_sha256=sha(S["fixture"]), pairs_sha256=sha(S["pairs_file"]), wall_s=round(time.time() - t0)),
              open(out.replace(".npz", ".receipt.json"), "w"), indent=1)
    print("wrote", out)


# metrics
def auroc(v, y):
    from scipy.stats import rankdata
    p, n = y.sum(), (~y).sum()
    if not p or not n:
        return np.nan
    rk = rankdata(v); return (rk[y].sum() - p * (p + 1) / 2) / (p * n)


# within-expression-decile AUROC (Fig-1 contract): deciles of mean Ms over the universe, positive-count-weighted mean
def strat_auroc(v, y, strat, nbin=10):
    from scipy.stats import rankdata
    q = np.minimum((rankdata(strat, method="ordinal") - 1) * nbin // len(strat), nbin - 1)
    num = w = 0.0
    for b in range(nbin):
        m = q == b; p_ = (m & y).sum()
        if p_ and (m & ~y).sum():
            num += p_ * auroc(v[m], y[m]); w += p_
    return num / w if w else np.nan


# Scores on a truth-free universe; truth selects metric rows afterwards
class Endpoint:
    def __init__(self, name, family, system, source, raw_h, y, metric, strat=None,
                 eval_mask=None, obs_genes=None, obs_sha256=None, legacy_universe=False):
        self.name, self.family, self.system, self.source = name, family, system, source
        self.y, self.metric, self.strat = y, metric, strat
        n = len(next(iter(raw_h.values())))
        self.eval_mask = np.ones(n, bool) if eval_mask is None else np.asarray(eval_mask, bool)
        if self.eval_mask.shape != (n,) or len(y) != int(self.eval_mask.sum()):
            raise ValueError("Endpoint truth/evaluation mask does not match its scoring universe")
        self.obs_genes = (np.arange(n).astype(str) if obs_genes is None
                          else np.asarray(obs_genes).astype(str))
        if len(self.obs_genes) != n:
            raise ValueError("Endpoint observational genes do not match its scoring universe")
        self.genes = self.obs_genes[self.eval_mask]  # compatibility: truth-evaluation axis
        self.obs_sha256 = obs_sha256
        self.legacy_universe = bool(legacy_universe)
        self.P = {h: np.column_stack([pct(raw_h[h][:, k]) for k in KS]) for h in ROOTS}

    def m(self, score):
        score = np.asarray(score)[self.eval_mask]
        return self.metric(score, self.y, self.strat) if self.strat is not None else self.metric(score, self.y)

    # Aggregate on the scoring universe; do not apply the truth mask here
    def combined(self, kh):
        return np.minimum(*[self.P[h][:, kh[h]].mean(1) for h in ROOTS])

    def at(self, kh):                     # kh: {root: list of k}
        return self.m(self.combined(kh))


# C: {root: C_h(k) for k 0..30}
def views(E, C):
    Mk = np.array([E.at({h: [k] for h in ROOTS}) for k in KS])
    cmax = min(C[h][30] for h in ROOTS); grid = CGRID[CGRID <= cmax + 1e-12]
    kc = {h: np.array([int(np.argmin(np.abs(C[h] - c))) for c in grid]) for h in ROOTS}
    Mc = np.array([E.at({h: [kc[h][i]] for h in ROOTS}) for i in range(len(grid))])
    band = {}
    for nm, (lo, hi) in BANDS.items():
        kh = {h: [k for k in range(1, 31) if lo <= C[h][k] <= hi] for h in ROOTS}
        band[nm] = E.at(kh) if all(kh.values()) else np.nan
    for nm, ks in FIXED.items():
        band[nm] = E.at({h: ks for h in ROOTS})
    return Mk, grid, kc, Mc, band


# per-endpoint summary
# C levels unreachable are censored (band = NaN / unresolved), never -1
def summarize(Mk, grid, Mc, band, C):
    mx = np.nanmax(Mk); gain = mx - .5; ko = int(np.nanargmax(Mk))
    r = dict(max=mx, k_oracle=ko, C_h_at_koracle="/".join(f"{C[h][ko]:.3f}" for h in ROOTS), informative=bool(gain >= .02))
    if np.isfinite(Mc).any():
        r["C_star"] = float(grid[int(np.nanargmax(Mc))])
    for q in (90, 95):
        K = [int(k) for k in KS if Mk[k] - .5 >= q / 100 * gain] if gain > 0 else []
        Cs = [float(c) for c, v in zip(grid, Mc) if gain > 0 and v - .5 >= q / 100 * gain]
        r[f"K{q}"] = ",".join(map(str, K))
        r[f"C{q}_lo"], r[f"C{q}_hi"] = (min(Cs), max(Cs)) if Cs else (np.nan, np.nan)
    for nm, v in band.items():
        r[f"M[{nm}]"] = v; r[f"nreg[{nm}]"] = (mx - v) / gain if gain > 0 else np.nan
    return r


def load_C(system, weight):
    Z = np.load(f"{R}/results/geometry_audit/{system}.npz", allow_pickle=True)
    return {h: np.asarray(Z[f"{h}|C_{weight}"], float)[:31] for h in ROOTS}


# endpoints
# {src: {(root, ch): (G, 31)}} from the cop scores or the regulon panel (same algebra); plus axis genes
def raw_rows(system, srcs=None):
    out = {}; eg = None
    for p in (f"{O}/scores/{system}.npz", f"{R}/results/regulon_panel/{system}/rankings.npz"):
        if not os.path.exists(p):
            continue
        Z = np.load(p, allow_pickle=True); zs = [s.upper() for s in np.asarray(Z["sources"]).astype(str)]
        g = np.asarray(Z["genes"]).astype(str)
        if eg is None:
            eg = g
        assert list(g) == list(eg), f"{p}: axis differs"
        key = (lambda r, c: f"{r}|{c}") if "cop_alignment" in p else (lambda r, c: f"{r}|{c}|absS")
        arr = {(r, c): Z[key(r, c)] for r in ROOTS for c in ("U", "X")}
        for s in (zs if srcs is None else srcs):
            if s.upper() in zs and s.upper() not in out:
                j = zs.index(s.upper()); out[s.upper()] = {rc: np.asarray(a[j], float) for rc, a in arr.items()}
    return out, eg


def mean_ms(system, eg):
    from gb_substrate import load
    import anndata as ad
    S = load(system, canonical=True); A = ad.read_h5ad(S["fixture"], backed="r")
    fm = {g: i for i, g in enumerate(np.asarray(A.var_names).astype(str))}
    return np.asarray(A.layers["Ms"][:, [fm[g] for g in eg]].mean(0)).ravel().astype(float)


# Truth-free U_obs and its exact positional hash
def observational_universe(src, rows, eg):
    gu = np.char.upper(np.asarray(eg).astype(str)); uni = gu != src.upper()
    for h in ROOTS:
        for c in ("U", "X"):
            uni &= np.isfinite(rows[(h, c)]).all(1)
    idx = np.flatnonzero(uni)
    return uni, hashlib.sha256(idx.tobytes()).hexdigest()


# Load score rows and record every U_obs before an endpoint truth is opened
def prepare_observational(system, srcs):
    rows, eg = raw_rows(system, srcs); obs = {}
    if eg is None:
        return rows, eg, obs
    srcs = sorted(rows) if srcs is None else srcs
    for src in srcs:
        key = src.upper()
        if key not in rows:
            continue
        mask, digest = observational_universe(key, rows[key], eg)
        obs[key] = (mask, digest)
        print(f"U_OBS system={system} source={key} n={int(mask.sum())} sha256={digest}", flush=True)
    return rows, eg, obs


# Reload score rows after the pre-truth hash pass and verify immutability
def reload_observational(system, srcs, recorded):
    rows, eg = raw_rows(system, srcs)
    if eg is None:
        return rows, eg, recorded
    for src, (mask, digest) in recorded.items():
        if src not in rows:
            continue
        now_mask, now_digest = observational_universe(src, rows[src], eg)
        if now_digest != digest or not np.array_equal(now_mask, mask):
            raise SystemExit(f"REFUSING_U_OBS_CHANGED_AFTER_HASH: {system}/{src}")
    return rows, eg, recorded


# Rank on U_obs (or the old labelled subset), then select truth rows
def build(system, src, rows, eg, y_full, labelled, name, family, metric, strat_full=None,
          observational=None, legacy_universe=False):
    labelled = np.asarray(labelled, bool); y_full = np.asarray(y_full, bool)
    obs, obs_sha = observational if observational is not None else observational_universe(src, rows, eg)
    score_uni = obs & labelled if legacy_universe else obs
    eval_mask = labelled[score_uni]
    eval_y = y_full[score_uni][eval_mask]
    eval_strat = None if strat_full is None else np.asarray(strat_full)[score_uni][eval_mask]
    score_genes = np.asarray(eg).astype(str)[score_uni]
    eps = {}
    for c in ("U", "X"):
        eps[c] = Endpoint(name, family, system, src, {h: rows[(h, c)][score_uni] for h in ROOTS}, eval_y, metric,
                          eval_strat, eval_mask=eval_mask, obs_genes=score_genes, obs_sha256=obs_sha,
                          legacy_universe=legacy_universe)
        eps[c].n_obs = int(obs.sum())
        eps[c].score_universe_sha256 = hashlib.sha256(np.flatnonzero(score_uni).tobytes()).hexdigest()
    return eps, int(eval_mask.sum()), int(eval_y.sum())


# yields (name, family, system, source, eps{U,X}, n, npos); truth is read here only
def endpoints(legacy_universe=False, systems=None):
    selected = set(sp.upper() for sp in systems) if systems is not None else None
    sp = {"GATA1": "mouse", "PANC": "mouse", "BONEMARROW": "human", "DENTATEGYRUS": "mouse", "GASTRULATION": "mouse", "ZEBRAFISH": "zebrafish"}
    k562 = ("ATF4 ATF5 BPTF CDC5L CEBPZ CXXC1 DNMT1 DR1 DRAP1 E4F1 GABPA GATA1 HINFP HSF1 JAZF1 KIN MAX MYBL2 MYC NRF1 SETDB1 SNAPC2 "
            "SON SP2 TBP TERF2 TFDP1 THAP1 THAP11 YY1 ZBTB17 ZNF236 ZNF317 ZNF335 ZNF574 ZNF687").split()
    bm = ["SPI1", "GATA1", "RUNX1", "TAL1", "KLF1", "MYB", "NFE2", "FOSL1"]
    inventory_systems = list(sp) + ["K562"]
    inventory_systems = [s for s in inventory_systems if selected is None or s in selected]
    recorded = {}
    for system in inventory_systems:
        rows, eg, recorded[system] = prepare_observational(system, None)
        # Do not retain every raw system matrix through the truth phase
        del rows, eg

    # No endpoint truth loader is imported or opened above this line.
    EL = json.load(open(f"{R}/results/fig_contracts/fig1_eligibility.json"))
    zfl = json.load(open(f"{O}/zf_sources.json")) if os.path.exists(f"{O}/zf_sources.json") else []
    requests = {}
    for system in sp:
        if selected is not None and system not in selected:
            continue
        requests[system] = list(EL["systems"][system]["eligible"]) if system != "ZEBRAFISH" else list(zfl)
    if selected is None or "GATA1" in selected:
        requests.setdefault("GATA1", []).append("GATA1")
    if selected is None or "PANC" in selected:
        requests.setdefault("PANC", []).append("RFX6")
    if selected is None or "BONEMARROW" in selected:
        requests.setdefault("BONEMARROW", []).extend(bm)
    if selected is None or "K562" in selected:
        requests.setdefault("K562", []).extend(k562)
    for system in requests:
        requests[system] = list(dict.fromkeys(s.upper() for s in requests[system]))
        missing = [s for s in requests[system] if s not in recorded[system]]
        if missing:
            raise SystemExit(f"REFUSING_ENDPOINT_SOURCE_NOT_PREHASHED: {system}: {missing[:10]}")
    from gb_fig1_eligibility import regulons
    regc = {}
    # F1 regulons
    for system in sp:
        if selected is not None and system not in selected:
            continue
        tfs = list(EL["systems"][system]["eligible"]) if system != "ZEBRAFISH" else zfl
        if not tfs:
            continue
        if sp[system] not in regc:
            regc[sp[system]] = regulons(sp[system])[0]
        reg = regc[sp[system]]
        rows, eg, obs = reload_observational(system, requests[system], recorded[system])
        if eg is None:
            continue
        gu = np.char.upper(eg); ms = mean_ms(system, eg)
        for t in tfs:
            if t.upper() not in rows:
                continue
            y = np.isin(gu, list(reg.get(t.upper(), set())))
            eps, n, npos = build(system, t, rows[t.upper()], eg, y, np.ones(len(eg), bool), f"F1|{system}|{t}", "F1", strat_auroc, ms,
                                  obs[t.upper()], legacy_universe)
            if npos >= 10 and n - npos >= 10:
                yield f"F1|{system}|{t}", "F1", system, t, eps, n, npos
    # F2 / F3 GATA1
    if selected is None or "GATA1" in selected:
        from gb_truth_v2_load import load_verified_gata1_v2, bound_mask
        rows, eg, obs = reload_observational("GATA1", requests["GATA1"], recorded["GATA1"])
        gu = np.char.upper(eg); V = load_verified_gata1_v2()
        V = V.set_index("gene").reindex(eg)   # exact-name join (upper-case collides SETD1A/Setd1a; neither labelled)
        tested = V.KO_tested.fillna(False).to_numpy(bool); resp = V.KO_resp.isin(["act", "rep"]).to_numpy()
        eps, n, npos = build("GATA1", "GATA1", rows["GATA1"], eg, resp, tested, "F2|GATA1|chronicKO", "F2", auroc,
                              observational=obs["GATA1"], legacy_universe=legacy_universe)
        yield "F2|GATA1|chronicKO", "F2", "GATA1", "GATA1", eps, n, npos
        for win in ("BOUND_PROM", "BOUND_10KB"):
            cl = V[f"any_change_{win}"].fillna(".").to_numpy(str); lab = np.isin(cl, ["A", "B"])
            eps, n, npos = build("GATA1", "GATA1", rows["GATA1"], eg, cl == "A", lab, f"F3|GATA1|V2_{win}", "F3", auroc,
                                  observational=obs["GATA1"], legacy_universe=legacy_universe)
            yield f"F3|GATA1|V2_{win}", "F3", "GATA1", "GATA1", eps, n, npos
        from gb_gata1_fourclass_canon import four_classes
        cls, _ = four_classes(list(eg)); A_, B_ = np.asarray(cls["any_change"]["A"]), np.asarray(cls["any_change"]["B"])
        eps, n, npos = build("GATA1", "GATA1", rows["GATA1"], eg, A_, A_ | B_, "F3|GATA1|V1_fourclass", "F3", auroc,
                              observational=obs["GATA1"], legacy_universe=legacy_universe)
        yield "F3|GATA1|V1_fourclass", "F3", "GATA1", "GATA1", eps, n, npos
    # PANC RFX6
    if selected is None or "PANC" in selected:
        from gb_dev_evaluate import truth_panc
        rows, eg, obs = reload_observational("PANC", requests["PANC"], recorded["PANC"])
        y, _ = truth_panc(list(eg))
        eps, n, npos = build("PANC", "RFX6", rows["RFX6"], eg, y == 1, y >= 0, "F2|PANC|RFX6", "F2", auroc,
                              observational=obs["RFX6"], legacy_universe=legacy_universe)
        yield "F2|PANC|RFX6", "F2", "PANC", "RFX6", eps, n, npos
    # BM GSE274113 stratified
    if selected is None or "BONEMARROW" in selected:
        tr = f"{R}/results/gse274113_truth_2026_09_19"
        rows, eg, obs = reload_observational("BONEMARROW", requests["BONEMARROW"], recorded["BONEMARROW"])
        gu = np.char.upper(eg)
        nt = pd.read_csv(f"{tr}/gene_stats_nt.tsv", sep="\t"); testable = set(nt.gene[nt.nt_detect_frac >= 0.05].str.upper())
        ST = pd.read_csv(f"{tr}/gse274113_stratified_panel.tsv", sep="\t")
        lab = np.isin(gu, list(testable))
        for t in bm:
            y = np.isin(gu, list(set(ST.target[ST.tf == t].str.upper())))
            nm = f"F2|BONEMARROW|{t}" + ("" if t == "SPI1" else "|sens")
            eps, n, npos = build("BONEMARROW", t, rows[t], eg, y, lab, nm, "F2", auroc,
                                  observational=obs[t], legacy_universe=legacy_universe)
            yield nm, "F2", "BONEMARROW", t, eps, n, npos
    # K562 (36 Fig-2-eligible TFs)
    if selected is None or "K562" in selected:
        rows, eg, obs = reload_observational("K562", requests["K562"], recorded["K562"])
        Z = np.load(f"{R}/results/canon_K562/per_gene.npz", allow_pickle=True); arm = "any_change|tested_sig_vs_tested_null_primary"
        zg = np.asarray(Z["genes"]).astype(str); assert list(zg) == list(eg), "K562 axis differs"
        for t in k562:
            pk = [k for k in Z.files if k.startswith(f"dc1_min|U|k10|") and k.split("|")[3].upper() == t and k.endswith(f"{arm}|pos")]
            if not pk or t not in rows:
                continue
            stem = pk[0][:-len("|pos")]
            y = np.asarray(Z[f"{stem}|pos"], bool); lab = np.asarray(Z[f"{stem}|panel_universe"], bool)
            eps, n, npos = build("K562", t, rows[t], eg, y, lab, f"F2|K562|{t}", "F2", auroc,
                                  observational=obs[t], legacy_universe=legacy_universe)
            yield f"F2|K562|{t}", "F2", "K562", t, eps, n, npos


# ZEBRAFISH Fig-1 roster by coverage only (contract rule: detected catalogue TF >= .05, >= 10 1:1 positives and comparators)
def zf_sources():
    from gb_fig1_eligibility import regulons, axis as fax
    reg, _ = regulons("zebrafish"); gu, det, is_tf, _ = fax("ZEBRAFISH")
    out = [g for g, t, d in zip(gu, is_tf, det) if t and d >= .05 and len(reg.get(g, set()) & (set(gu) - {g})) >= 10
           and len(gu) - 1 - len(reg.get(g, set()) & set(gu)) >= 10]
    os.makedirs(O, exist_ok=True); json.dump(out, open(f"{O}/zf_sources.json", "w")); print(len(out), ",".join(out))


# catalogue-excluded D sensitivity: C(k) with the frozen species TF catalogue removed from D, production projector fixed
def catex(system):
    import gb_geometry_audit as GA
    D = GA.load(system); n = D["X"].shape[0]; cat = np.asarray(D["S"]["is_tf"], bool); gref = D["gref"]
    out = {}
    for h in ROOTS:
        I, J = D["S"]["pairs"][h]; W = GA.weights(I, J, n); Q30, _ = GA.qr_flag(D["P30"][J] - D["P30"][I])
        inc = GA.cw_curve(Q30, D["Ms"][:, gref], I, J, W["pair"], D["sd"][gref])
        ref = np.asarray(np.load(f"{R}/results/geometry_audit/{system}.npz")[f"{h}|C_pair"], float)[:31]
        assert np.abs(inc - ref).max() < 1e-6, f"catex parity {system} {h}: {np.abs(inc-ref).max()}"
        m = gref & ~cat
        out[h] = GA.cw_curve(Q30, D["Ms"][:, m], I, J, W["pair"], D["sd"][m])
        print(system, h, "catex parity ok; C_catex(5,10,30)", out[h][[5, 10, 30]].round(3), "n_genes", int(m.sum()), flush=True)
    np.savez(f"{O}/C_catex_{system}.npz", **out)


def load_Cw(system):
    C = {w: load_C(system, w) for w in ("pair", "cell")}
    Z = np.load(f"{O}/C_catex_{system}.npz"); C["catex"] = {h: np.asarray(Z[h], float) for h in ROOTS}
    return C


UNIT = lambda e: ("F1|" + e.split("|")[1] if e.startswith("F1|") else "F2|K562" if e.startswith("F2|K562|")
                  else "F2|BONEMARROW|sens" if e.endswith("|sens") else e)
VERDICT_CTX = ["F1|GATA1", "F1|PANC", "F1|BONEMARROW", "F1|DENTATEGYRUS", "F1|ZEBRAFISH", "F2|GATA1|chronicKO", "F2|PANC|RFX6",
               "F2|BONEMARROW|SPI1", "F2|BONEMARROW|sens", "F2|K562", "F3|GATA1|V2_BOUND_PROM"]
ALLB = list(BANDS) + list(FIXED)


def band_ks(C):
    kh = {nm: {h: [k for k in range(1, 31) if lo <= C[h][k] <= hi] for h in ROOTS} for nm, (lo, hi) in BANDS.items()}
    kh.update({nm: {h: ks for h in ROOTS} for nm, ks in FIXED.items()}); return kh


def evaluate(legacy_universe=False):
    if __import__("os").environ.get("GB_SKIP_PREFLIGHT") == "1": print("preflight skipped (public release)")
    else: assert sha(f"{O}/PREREG.md") == PREREG_SHA, "PREREG changed"
    cache = list(endpoints(legacy_universe=legacy_universe)); print("endpoints", len(cache), flush=True)
    rows, curves, single = [], [], {}
    for name, fam, system, src, eps, n, npos in cache:
        Cw = load_Cw(system)
        for ch, E in eps.items():
            Mk = np.array([E.at({h: [k] for h in ROOTS}) for k in KS])        # weight-independent raw-k curve
            for weight, C in Cw.items():
                _, grid, kc, Mc, band = views(E, C)
                r = dict(weight=weight, endpoint=name, unit=UNIT(name), family=fam, system=system, source=src, channel=ch, n=n,
                         npos=npos, n_obs=E.n_obs, u_obs_sha256=E.obs_sha256,
                         score_universe_sha256=E.score_universe_sha256, legacy_universe=E.legacy_universe,
                         **summarize(Mk, grid, Mc, band, C))
                cset = [float(c) for c, v in zip(grid, Mc) if r["max"] - .5 > 0 and v - .5 >= .9 * (r["max"] - .5)]
                r["C90_set"] = ",".join(f"{c:.3f}" for c in cset); rows.append(r)
                curves += [dict(weight=weight, endpoint=name, unit=UNIT(name), channel=ch, view="C", x=float(c), M=float(v),
                                k_min=int(kc["dc1_min"][i]), k_max=int(kc["dc1_max"][i])) for i, (c, v) in enumerate(zip(grid, Mc))]
            curves += [dict(weight="any", endpoint=name, unit=UNIT(name), channel=ch, view="k", x=int(k), M=float(Mk[k])) for k in KS]
            if fam != "F1" and not name.startswith("F2|K562|") and not name.endswith("|sens"):
                single[(name, ch)] = (E, Cw)
    EP = pd.DataFrame(rows); CU = pd.DataFrame(curves)
    EP.to_csv(f"{O}/endpoints_per_tf.tsv", sep="\t", index=False); CU.to_csv(f"{O}/curves_per_tf.tsv", sep="\t", index=False)
    return EP, CU, single


# context-level curve, band values, regret per band, coverage; bootstrap replicate regrets (NB x bands) and band-vs-k0 gains
def ctx_stats(ctx, ch, weight, EP, CU, single, rng, NB=200):
    e = EP[(EP.unit == ctx) & (EP.channel == ch) & (EP.weight == weight)]
    ck_tf = CU[(CU.unit == ctx) & (CU.channel == ch) & (CU.view == "k")].pivot_table(index="endpoint", columns="x", values="M")
    ck_tf = ck_tf.loc[e.endpoint]
    bv = e.set_index("endpoint")[[f"M[{b}]" for b in ALLB]]
    med_k = ck_tf.median(0).to_numpy(); med_b = bv.median(0).to_numpy(); mx = np.nanmax(med_k)
    out = dict(context=ctx, channel=ch, weight=weight, n_tf=len(e), max=mx, k_oracle=int(np.nanargmax(med_k)),
               M_k0=float(med_k[0]), informative=bool(mx - .5 >= .02))
    for b, v in zip(ALLB, med_b):
        out[f"M[{b}]"] = float(v); out[f"reg[{b}]"] = float(mx - v)
        out[f"cov[{b}]"] = float(np.isfinite(bv[f"M[{b}]"]).mean())
    reps = np.full((NB, len(ALLB)), np.nan); g0 = np.full((NB, len(ALLB)), np.nan)
    if len(e) > 1:                                                     # resample TFs
        K_ = ck_tf.to_numpy(); Bv = bv.to_numpy()
        for b in range(NB):
            ix = rng.integers(0, len(e), len(e)); mk = np.nanmedian(K_[ix], 0); mb = np.nanmedian(Bv[ix], 0)
            reps[b] = np.nanmax(mk) - mb; g0[b] = mb - mk[0]
    else:                                                              # resample genes, oracle re-maximised
        E, Cw = single[(e.endpoint.iloc[0], ch)]; kh = band_ks(Cw[weight]); n = len(E.y)
        Pe = {h: E.P[h][E.eval_mask] for h in ROOTS}
        for b in range(NB):
            ix = rng.integers(0, n, n); y = E.y[ix]
            if y.all() or not y.any():
                continue
            m_ = lambda sc: E.metric(sc, y, E.strat[ix]) if E.strat is not None else E.metric(sc, y)
            sub = lambda K: np.minimum(*[Pe[h][ix][:, K[h]].mean(1) for h in ROOTS])
            mk = np.array([m_(sub({h: [k] for h in ROOTS})) for k in KS])
            mb = np.array([m_(sub(kh[nm])) if all(kh[nm].values()) else np.nan for nm in ALLB])
            reps[b] = np.nanmax(mk) - mb; g0[b] = mb - mk[0]
    return out, reps, g0


def aggregate(EP, CU, single):
    rng = np.random.default_rng(20260924); A, summ = [], {}
    for weight in ("pair", "cell", "catex"):
        for ch in ("U", "X"):
            ctxs = sorted(EP.unit.unique()); R_, G_ = {}, {}
            for ctx in ctxs:
                o, reps, g0 = ctx_stats(ctx, ch, weight, EP, CU, single, rng); A.append(o); R_[ctx], G_[ctx] = reps, g0
            AA = pd.DataFrame([a for a in A if a["weight"] == weight and a["channel"] == ch]).set_index("context")
            V = [c for c in VERDICT_CTX if c in AA.index]
            macro = {b: float(AA.loc[V, f"reg[{b}]"].mean()) for b in ALLB}
            rep_macro = np.nanmean(np.stack([R_[c] for c in V]), 0)              # NB x bands
            q = .05 / len(BANDS)
            ub = {b: float(np.nanquantile(rep_macro[:, i], 1 - q)) for i, b in enumerate(ALLB)}
            lb = {b: float(np.nanquantile(rep_macro[:, i], q)) for i, b in enumerate(ALLB)}
            cover = {b: bool((AA.loc[V, f"cov[{b}]"] >= .90).all()) for b in ALLB}
            Mb = {b: float(AA.loc[V, f"M[{b}]"].mean()) for b in ALLB}
            noninf = {b: bool(Mb[b] >= Mb["k0"] - .02 and Mb[b] >= Mb["k1-5"] - .02) for b in ALLB}
            nuis = [c for c in V if AA.loc[c, "max"] - AA.loc[c, "M_k0"] >= .02]
            gain, gain_lb = {}, {}
            for i, b in enumerate(ALLB):
                if nuis:
                    gain[b] = float(np.mean([AA.loc[c, f"M[{b}]"] - AA.loc[c, "M_k0"] for c in nuis]))
                    gain_lb[b] = float(np.nanquantile(np.nanmean(np.stack([G_[c][:, i] for c in nuis]), 0), .05))
                else:
                    gain[b] = gain_lb[b] = np.nan
            tf = EP[(EP.unit == "F2|K562") & (EP.channel == ch) & (EP.weight == weight)]
            k562p90 = {b: float(np.nanquantile(tf["max"] - tf[f"M[{b}]"], .9)) if len(tf) else np.nan for b in ALLB}
            ctxpass = {b: cover[b] and ub[b] <= .02 and noninf[b] and gain[b] >= .02 and gain_lb[b] > 0 for b in BANDS}
            A_ = [b for b in BANDS if ctxpass[b] and k562p90[b] <= .05]; B_ = [b for b in BANDS if ctxpass[b] and b not in A_]
            if A_:
                v = "A"
            elif B_:
                v = "B"
            elif all(lb[b] > .02 for b in BANDS):
                v = "C"
            else:
                v = "NO_CALL"
            summ[f"{weight}|{ch}"] = dict(verdict=v, bands_A=A_, bands_B=B_, verdict_contexts=V, nuisance_contexts=nuis,
                                          macro_regret=macro, macro_regret_UB=ub, macro_regret_LB=lb, coverage_ge90_all=cover,
                                          macro_M=Mb, noninferior=noninf, nuisance_gain=gain, nuisance_gain_LB=gain_lb,
                                          k562_p90_regret=k562p90)
            print(weight, ch, "verdict", v, flush=True)
    A = pd.DataFrame(A); A.to_csv(f"{O}/contexts.tsv", sep="\t", index=False)
    json.dump(summ, open(f"{O}/summary.json", "w"), indent=1, default=float)
    return A, summ


def descriptive(EP, CU):
    rows = []
    for (weight, ch), g in CU[(CU.view == "C")].groupby(["weight", "channel"]):
        for ctx in VERDICT_CTX:
            cc = g[g.unit == ctx].groupby("x").M.median(); ck = CU[(CU.view == "k") & (CU.channel == ch) & (CU.unit == ctx)].groupby("x").M.median()
            if not len(cc):
                continue
            mx = ck.max(); gain = mx - .5
            cset = [float(c) for c, v in cc.items() if gain > 0 and v - .5 >= .9 * gain]
            rows.append(dict(weight=weight, channel=ch, context=ctx, informative=bool(gain >= .02), k_oracle=int(ck.idxmax()),
                             C_star=float(cc.idxmax()), C90_set=",".join(f"{c:.3f}" for c in cset),
                             K90=",".join(str(int(k)) for k, v in ck.items() if gain > 0 and v - .5 >= .9 * gain)))
    D = pd.DataFrame(rows)
    out = {}
    for (weight, ch), g in D.groupby(["weight", "channel"]):
        g = g[g.informative]; sets = [set(s.split(",")) - {""} for s in g.C90_set]
        disj = sum(1 for i, s in enumerate(sets) if not any(s & t for j, t in enumerate(sets) if j != i))
        out[f"{weight}|{ch}"] = dict(n=len(g), frac_C90_disjoint=disj / len(g) if len(g) else np.nan,
                                     IQR_koracle_over30=float((g.k_oracle.quantile(.75) - g.k_oracle.quantile(.25)) / 30) if len(g) else np.nan,
                                     IQR_Cstar=float(g.C_star.quantile(.75) - g.C_star.quantile(.25)) if len(g) else np.nan)
    D.to_csv(f"{O}/contexts_C90.tsv", sep="\t", index=False); json.dump(out, open(f"{O}/descriptive.json", "w"), indent=1)
    return D, out


def k562(EP):
    J = json.load(open(f"{R}/results/geometry_audit/K562.json")); Z = np.load(f"{R}/results/geometry_audit/K562.npz", allow_pickle=True)
    el = [s.upper() for s in J["eligible_sources"]]; rows = []
    for _, r in EP[(EP.weight == "pair") & (EP.unit == "F2|K562")].iterrows():
        t = r.source.upper(); k = int(r.k_oracle)
        d = dict(tf=t, channel=r.channel, informative=r.informative, max=r["max"], k_oracle=k, K90=r.K90, C_star=r.get("C_star"),
                 C_h_at_koracle=r.C_h_at_koracle, reg_C2030=r["max"] - r["M[C[.20,.30]]"], reg_k15=r["max"] - r["M[k1-5]"])
        if t in el:
            j = el.index(t)
            for h in ROOTS:
                d[f"Cr_{h}"] = 1 - float(np.asarray(Z[f"{h}|Ret_pair"])[k, j])
        rows.append(d)
    T = pd.DataFrame(rows); T.to_csv(f"{O}/k562_stress.tsv", sep="\t", index=False)
    s = {}
    for ch, g in T.groupby("channel"):
        g = g[g.informative]
        nq = lambda x, rng_: float((x.quantile(.75) - x.quantile(.25)) / rng_) if len(x) else np.nan
        cop = g.C_h_at_koracle.str.split("/").apply(lambda v: np.mean([float(x) for x in v]))
        cr = g[[f"Cr_{h}" for h in ROOTS]].mean(1) if "Cr_dc1_min" in g else pd.Series(dtype=float)
        s[ch] = dict(n_informative=len(g), nIQR_koracle=nq(g.k_oracle, 30), nIQR_Cop=nq(cop, 1.0), nIQR_Cr=nq(cr, 1.0),
                     k_oracle_values=g.k_oracle.tolist(), Cop_at_koracle=[round(x, 3) for x in cop], Cr_at_koracle=[round(x, 3) for x in cr])
    json.dump(s, open(f"{O}/k562_summary.json", "w"), indent=1)
    return T, s


def plots(CU):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    os.makedirs(f"{O}/figs", exist_ok=True); paths = []
    for ch in ("U", "X"):
        fig, ax = plt.subplots(1, 3, figsize=(20, 6))
        for ctx in VERDICT_CTX + ["F1|GASTRULATION"]:
            ck = CU[(CU.view == "k") & (CU.channel == ch) & (CU.unit == ctx)].groupby("x").M.median()
            cc = CU[(CU.view == "C") & (CU.weight == "pair") & (CU.channel == ch) & (CU.unit == ctx)].groupby("x").M.median()
            if not len(ck):
                continue
            l, = ax[0].plot(ck.index, ck.values, label=ctx, lw=1.3, ls=":" if "GASTRULATION" in ctx else "-")
            ax[1].plot(cc.index, cc.values, lw=1.3, color=l.get_color())
            mx = ck.max()
            if mx - .5 > 1e-9:
                ax[2].plot(cc.index, (cc.values - .5) / (mx - .5), lw=1.3, color=l.get_color())
        for a in ax[1:]:
            for nm, (lo, hi) in BANDS.items():
                a.axvspan(lo, hi, alpha=.04, color="k")
        ax[0].set(xlabel="conditioning depth k", ylabel="AUROC (context median)", title=f"{ch}: performance vs k")
        ax[1].set(xlabel="C_op (pair weighting; root-specific k)", title=f"{ch}: performance vs C_op")
        ax[2].set(xlabel="C_op", ylabel="(M-.5)/(max_k M-.5)", title=f"{ch}: standardized overlay"); ax[2].axhline(.9, ls=":", c="grey")
        ax[0].legend(fontsize=7)
        p = f"{O}/figs/cop_alignment_{ch}.png"; fig.tight_layout(); fig.savefig(p, dpi=110); plt.close(fig); paths.append(p)
    return paths


def selftest():
    y = np.array([1, 1, 0, 0, 0], bool); assert auroc(np.array([5., 4, 3, 2, 1]), y) == 1.0
    assert abs(auroc(np.array([1., 1, 1, 1, 1]), y) - .5) < 1e-12
    s = np.arange(40.); yy = np.zeros(40, bool); yy[::4] = True
    assert strat_auroc(s, yy, s) == strat_auroc(s, yy, s)          # runs
    C = {h: np.linspace(0, .6, 31) for h in ROOTS}; raw = {h: np.random.default_rng(0).normal(size=(50, 31)) for h in ROOTS}
    yb = np.zeros(50, bool); yb[:10] = True
    for h in ROOTS:
        raw[h][:10, 10] += 5                                         # signal only at k=10 -> C = .2
    E = Endpoint("t", "F2", "S", "s", raw, yb, auroc); Mk, grid, kc, Mc, band = views(E, C)
    assert int(np.argmax(Mk)) == 10 and abs(grid[int(np.nanargmax(Mc))] - .2) < 1e-9 and band["C[.10,.20]"] > .5

    # Evaluation-universe repair: parity when all observational rows are labelled.
    eg = np.array([f"g{i}" for i in range(6)])
    toy = {}
    for h in ROOTS:
        for c in ("U", "X"):
            a = np.zeros((6, 31), float)
            a[:, 0] = [5, 3, 1, 2, 4, 0]
            a[:, 1] = [0, 1, 3, 5, 2, 4]
            a[:, 2:] = a[:, [1]]
            toy[(h, c)] = a
    yt = np.array([1, 1, 0, 0, 0, 0], bool)
    all_lab = np.ones(6, bool)
    new, _, _ = build("S", "SRC", toy, eg, yt, all_lab, "all", "F2", auroc)
    old, _, _ = build("S", "SRC", toy, eg, yt, all_lab, "all", "F2", auroc, legacy_universe=True)
    for c in ("U", "X"):
        for h in ROOTS:
            assert np.array_equal(new[c].P[h], old[c].P[h])
        assert new[c].at({h: [0, 1] for h in ROOTS}) == old[c].at({h: [0, 1] for h in ROOTS})

    # With a strict truth subset, a multi-k band may change although each single-root/single-k AUROC
    # is rank invariant under the extra rows.
    labelled = np.array([1, 1, 1, 1, 0, 0], bool)
    new, _, _ = build("S", "SRC", toy, eg, yt, labelled, "sub", "F2", auroc)
    old, _, _ = build("S", "SRC", toy, eg, yt, labelled, "sub", "F2", auroc, legacy_universe=True)
    band_new = new["U"].at({h: [0, 1] for h in ROOTS})
    band_old = old["U"].at({h: [0, 1] for h in ROOTS})
    assert band_new != band_old
    for h in ROOTS:
        for k in (0, 1):
            a = auroc(new["U"].P[h][new["U"].eval_mask, k], new["U"].y)
            b = auroc(old["U"].P[h][:, k], old["U"].y)
            assert a == b
    print("gb_cop_alignment selftest PASS")


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "selftest":
        selftest()
    elif c == "score":
        score(sys.argv[2], sys.argv[3].split(","))
    elif c == "zf_sources":
        zf_sources()
    elif c == "catex":
        catex(sys.argv[2])
    elif c == "eval":
        legacy = "--legacy-universe" in sys.argv[2:]
        selftest(); EP, CU, single = evaluate(legacy_universe=legacy); A, summ = aggregate(EP, CU, single); D, dsc = descriptive(EP, CU)
        T, ks = k562(EP); p = plots(CU)
        print(json.dumps(dict(verdict={k: v["verdict"] for k, v in summ.items()}, descriptive=dsc, k562=ks), indent=1, default=float)); print(p)
