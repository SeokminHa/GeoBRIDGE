#!/usr/bin/env python
# Polarity x method comparison on identical masks
import argparse, hashlib, json, os, sys, time
import numpy as np, pandas as pd
from scipy.stats import rankdata, mannwhitneyu, hypergeom

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/pairspace"))
from gb_substrate import load, d_
import anndata as ad
from gb_gata1_fourclass_canon import four_classes, G as FOURCLASS_DIR
from gb_dev_evaluate import truth_panc

R = __import__("os").environ["GEOBRIDGE_ROOT"]
CONTRACT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "SOTA_COMPARISON_CONTRACT_2026_09_22.md")
MIN_POS, MIN_NEG = 10, 10
NS = (20, 50, 100)


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# a case-insensitive collision is refused only where it could change a lookup
def axis_check(names, label, must_be_unique_ci=()):
    names = np.asarray(names).astype(str)
    if len(set(names.tolist())) != len(names):
        raise SystemExit(f"REFUSING: exact duplicate names on the {label} axis")
    up = np.char.upper(names); u, c = np.unique(up, return_counts=True); coll = set(u[c > 1])
    bad = coll & {x.upper() for x in must_be_unique_ci}
    if bad:
        raise SystemExit(f"REFUSING: case-insensitive collision on the {label} axis touches {sorted(bad)[:5]}")
    return names


TRUTH_LABELS = {"GATA1": ("gata1_chipseq_labels", "gata1_control", "gata1_ko_projected"),
                "PANC": ("panc_de", "panc_orthology")}


# truth providers
# the same T used by four_classes (gb_gata1_fourclass_canon.py:73-75), not a re-derivation
def ko_T(genes):
    W_ = ad.read_h5ad(f"{FOURCLASS_DIR}/gata1_control_deterministic.h5ad")
    K_ = ad.read_h5ad(f"{FOURCLASS_DIR}/gata1_ko_projected_deterministic.h5ad")
    T_ = -np.log2((np.expm1(d_(K_.X)).mean(0) + 1e-6) / (np.expm1(d_(W_.X)).mean(0) + 1e-6))
    tmap = dict(zip(np.asarray(W_.var_names).astype(str), T_))
    return np.array([tmap.get(g, np.nan) for g in genes])


# axis -> pos, contrasts {name: (neg_mask, evaluator panel whose recovery mask is used)}
def truth_gata1_classes(genes, keep):
    cls, counts = four_classes(list(genes))
    C = {ax: {k: v[keep] for k, v in cls[ax].items()} for ax in ("repression", "activation")}
    AX = {a: dict(pos=C[a]["A"], contrasts={"A_vs_B": (C[a]["B"], "A_vs_B"),
                                             "A_vs_D": (C[a]["D"], "A_vs_rest")})
          for a in ("repression", "activation")}
    return AX, counts, np.abs(ko_T(list(genes[keep]))), "{root}|{chan}|k{k}|GATA1|{axis}|{panel}", "GATA1"


# PANC RFX6
def truth_panc_classes(genes, keep):
    y, pol = truth_panc(list(genes)); y, pol = y[keep], pol[keep]
    de = pd.read_csv(f"{R}/real_1_Pancreas/external_validation/aldous_rfx6/results/"
                     f"aldous_PPs_WT_vs_KO_DE.tsv", sep="\t")
    hom = pd.read_csv(f"{R}/data/orthology/HOM_MouseHumanSequence.rpt", sep="\t", low_memory=False)
    key = hom.columns[0]; org = hom["Common Organism Name"]
    mm = hom[org.str.startswith("mouse")].groupby(key)["Symbol"].apply(list)
    hs = hom[org.str.startswith("human")].groupby(key)["Symbol"].apply(list)
    h2m = {hs[k][0].upper(): mm[k][0] for k in set(mm.index) & set(hs.index)
           if len(mm[k]) == 1 and len(hs[k]) == 1}
    gp = {g: i for i, g in enumerate(genes[keep])}
    gain = np.full(keep.sum(), np.nan)
    dsym = de.gene_name.astype(str).str.upper(); dupsym = set(dsym[dsym.duplicated(keep=False)])   # ambiguous -> NaN
    for hg, l in zip(dsym, de.log2FoldChange):
        m = h2m.get(hg)
        if m in gp and hg not in dupsym:
            if np.isfinite(gain[gp[m]]) and gain[gp[m]] != abs(l):
                raise SystemExit(f"REFUSING: conflicting RFX6 DE effects map to {m}")
            gain[gp[m]] = abs(l)
    neg = y == 0
    AX = {"repression": dict(pos=(y == 1) & (pol > 0), contrasts={"pos_vs_null": (neg, "resp_vs_null")}),
          "activation": dict(pos=(y == 1) & (pol < 0), contrasts={"pos_vs_null": (neg, "resp_vs_null")})}
    counts = dict(n_act=int(AX["activation"]["pos"].sum()), n_rep=int(AX["repression"]["pos"].sum()),
                  n_null=int(neg.sum()), n_unknown=int((y == -1).sum()))
    return AX, counts, gain, "{root}|{chan}|k{k}|RFX6|any_change|{panel}", "RFX6"


# tie-invariant metrics
# descending score order
def tie_blocks(s):
    o = np.argsort(-s, kind="stable"); ss = s[o]
    edges = np.flatnonzero(np.diff(ss) != 0) + 1
    return o, np.concatenate([[0], edges]), np.concatenate([edges, [len(s)]])


# Mann-Whitney AUC with ties = 1/2; optional non-negative replicate weights (bootstrap multiset)
def auc(s, y, w=None):
    if w is None:
        w = np.ones(len(s))
    wp, wn = w[y].sum(), w[~y].sum()
    if wp == 0 or wn == 0:
        return np.nan
    o = np.argsort(s, kind="stable"); ss = s[o]; yy = y[o]; ww = w[o]
    edges = np.flatnonzero(np.diff(ss) != 0) + 1
    starts = np.concatenate([[0], edges]); stops = np.concatenate([edges, [len(ss)]])
    negw = np.where(~yy, ww, 0.0); posw = np.where(yy, ww, 0.0)
    bn = np.add.reduceat(negw, starts); bp = np.add.reduceat(posw, starts)
    below = np.concatenate([[0.0], np.cumsum(bn)[:-1]])
    return float((bp * (below + 0.5 * bn)).sum() / (wp * wn))


# expected AP when the order inside each tie block is uniformly random (exact block formula)
def ap_tieavg(s, y):
    n_pos = y.sum()
    if n_pos == 0:
        return np.nan
    o, starts, stops = tie_blocks(s); yy = y[o].astype(float)
    ap = 0.0; tp_before = 0.0
    for a, b in zip(starts, stops):
        m = b - a; p = yy[a:b].sum()
        if p == 0:
            continue
        j = np.arange(1, m + 1)
        exp_tp = tp_before + 1 + ((j - 1) * (p - 1) / (m - 1) if m > 1 else 0.0)
        ap += ((p / m) * exp_tp / (a + j)).sum()
        tp_before += p
    return ap / n_pos


# tie-aware precision at N: boundary block contributes its expected share
def prec_at(s, y, N):
    if len(s) < N:
        return np.nan
    thr = np.sort(s)[::-1][N - 1]
    above = s > thr; tied = s == thr
    need = N - above.sum()
    return (y[above].sum() + (need * y[tied].mean() if need > 0 else 0.0)) / N


# expected dcg over tie blocks (partial cutoff block included) / ideal dcg
def ndcg_at(s, gain, N):
    if len(s) < N:
        return np.nan
    disc = 1.0 / np.log2(np.arange(2, len(s) + 2))
    o, starts, stops = tie_blocks(s); g = gain[o]
    dcg = 0.0
    for a, b in zip(starts, stops):
        if a >= N:
            break
        dcg += g[a:b].mean() * disc[a:min(b, N)].sum()
    ideal = (np.sort(gain)[::-1][:N] * disc[:N]).sum()
    return dcg / ideal if ideal > 0 else np.nan


def deciles(s, gain):
    r = rankdata(s, method="average"); q = np.minimum((r - 1) / len(s) * 10, 9).astype(int)
    return {f"dec{d}": float(np.nanmedian(gain[q == d])) if (q == d).any() else np.nan for d in range(10)}


NAMES = ("Delta_rep", "Delta_act", "D_interaction")


# One joint complete-case gene bootstrap over the whole family
def joint_bootstrap(fam, n_boot, seed, t0=None):
    rng = np.random.default_rng(seed)
    U_r = np.unique(np.concatenate([f["pos_r"] for f in fam]))
    U_a = np.unique(np.concatenate([f["pos_a"] for f in fam]))
    U_n = np.unique(np.concatenate([f["neg_r"] for f in fam] + [f["neg_a"] for f in fam]))
    # one draw per replicate for each of the three pools
    W = {"r": rng.multinomial(len(U_r), np.full(len(U_r), 1 / len(U_r)), size=n_boot),
         "a": rng.multinomial(len(U_a), np.full(len(U_a), 1 / len(U_a)), size=n_boot),
         "n": rng.multinomial(len(U_n), np.full(len(U_n), 1 / len(U_n)), size=n_boot)}
    col = {"r": {g: i for i, g in enumerate(U_r)}, "a": {g: i for i, g in enumerate(U_a)},
           "n": {g: i for i, g in enumerate(U_n)}}
    stats = np.full((n_boot, len(fam), 3), np.nan); point = np.empty((len(fam), 3))
    for j, f in enumerate(fam):
        idx_r = np.concatenate([f["pos_r"], f["neg_r"]]); y_r = np.r_[np.ones(len(f["pos_r"]), bool), np.zeros(len(f["neg_r"]), bool)]
        idx_a = np.concatenate([f["pos_a"], f["neg_a"]]); y_a = np.r_[np.ones(len(f["pos_a"]), bool), np.zeros(len(f["neg_a"]), bool)]
        gb_r, m_r = f["gb_r"][idx_r], f["m_r"][idx_r]; gb_a, m_a = f["gb_a"][idx_a], f["m_a"][idx_a]
        cr_ = np.array([col["r"][g] for g in f["pos_r"]]); cn_r = np.array([col["n"][g] for g in f["neg_r"]])
        ca_ = np.array([col["a"][g] for g in f["pos_a"]]); cn_a = np.array([col["n"][g] for g in f["neg_a"]])
        d_r0 = auc(gb_r, y_r) - auc(m_r, y_r); d_a0 = auc(gb_a, y_a) - auc(m_a, y_a)
        point[j] = (d_r0, d_a0, d_r0 - d_a0)
        Wr = np.concatenate([W["r"][:, cr_], W["n"][:, cn_r]], axis=1)
        Wa = np.concatenate([W["a"][:, ca_], W["n"][:, cn_a]], axis=1)
        for b in range(n_boot):
            dr = auc(gb_r, y_r, Wr[b]) - auc(m_r, y_r, Wr[b])
            da = auc(gb_a, y_a, Wa[b]) - auc(m_a, y_a, Wa[b])
            stats[b, j] = (dr, da, dr - da)
        if t0 is not None and j % 100 == 0:
            print(f"  bootstrap member {j}/{len(fam)}  {time.time()-t0:.0f}s", flush=True)
    sd = np.nanstd(stats, axis=0, ddof=1); sd = np.where(sd > 0, sd, np.nan)
    mu = np.nanmean(stats, axis=0)
    crit = {}
    for q in range(3):
        tmax = np.nanmax(np.abs((stats[:, :, q] - mu[:, q]) / sd[:, q]), axis=1)
        crit[q] = float(np.nanpercentile(tmax, 95))
    pools = dict(pos_rep=int(len(U_r)), pos_act=int(len(U_a)), neg=int(len(U_n)))
    return point, stats, sd, crit, pools


# main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir"); ap.add_argument("--pairstatic")
    ap.add_argument("--sota", nargs="*", default=[])
    ap.add_argument("--n-boot", type=int, default=2000); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--system", default="GATA1", choices=["GATA1", "PANC"])
    ap.add_argument("--tag", default="full", choices=["full", "smoke"])
    a = ap.parse_args(); t0 = time.time()
    D = a.run_dir
    if a.n_boot < 2000 and a.tag != "smoke":
        raise SystemExit("REFUSING: n_boot < 2000 requires --tag smoke")
    prov = dict(script_sha=sha(os.path.abspath(__file__)), run_dir=D, system=a.system, seed=a.seed,
                n_boot=a.n_boot, tag=a.tag, min_pos=MIN_POS, min_neg=MIN_NEG,
                contract_sha=sha(CONTRACT) if os.path.exists(CONTRACT) else None,
                per_gene_sha=sha(f"{D}/per_gene.npz"), inputs={})
    Zn = np.load(f"{D}/per_gene.npz", allow_pickle=True)
    eg = np.asarray(Zn["genes"]).astype(str)
    if len(set(np.char.upper(eg).tolist())) != len(eg):
        raise SystemExit("REFUSING: duplicate (case-insensitive) genes on the evaluator axis")
    S = load(a.system, canonical=True)
    genes = np.asarray(S["genes"]).astype(str)
    keep = np.isin(genes, eg)
    assert (genes[keep] == eg).all(), "per_gene gene order != substrate order"
    LP = json.load(open(f"{D}/layers_provenance.json"))
    TA = LP["truth_artifacts"]
    fixture_sha = sha(S["fixture"])
    if TA["substrate_fixture"]["sha256"] != fixture_sha or os.path.realpath(TA["substrate_fixture"]["path"]) != os.path.realpath(S["fixture"]):
        raise SystemExit("REFUSING: substrate fixture differs from the one the evaluator run bound")
    for lbl in TRUTH_LABELS[a.system]:
        rec = TA[lbl]
        if rec["sha256"] == "ABSENT" or sha(rec["path"]) != rec["sha256"]:
            raise SystemExit(f"REFUSING: truth artifact {lbl} changed since the evaluator bound it")
    prov["fixture_sha"] = fixture_sha
    prov["truth_artifacts_verified"] = {l: TA[l] for l in TRUTH_LABELS[a.system]}
    prov["tf_catalogue_sha256"] = LP["tf_catalogue_sha256"]
    AX, counts, T, KEY, tfname = {"GATA1": truth_gata1_classes, "PANC": truth_panc_classes}[a.system](genes, keep)
    Ms = S["X"][:, keep]
    Z = Ms - Ms.mean(0); Z = Z / np.maximum(np.linalg.norm(Z, axis=0), 1e-300)
    gu = np.char.upper(eg); tfi = int(np.where(gu == tfname.upper())[0][0])
    pearson = Z[:, tfi] @ Z
    comp = {"pearson_raw": pearson, "pearson_neg": -pearson, "pearson_abs": np.abs(pearson),
            "target_expr": Ms.mean(0)}
    for nm in comp:
        comp[nm][tfi] = np.nan
    PS = None
    if a.pairstatic:
        PS = np.load(a.pairstatic, allow_pickle=True)
        for fld in ("fixture_sha256", "catalogue_sha256"):
            if fld not in PS.files:
                raise SystemExit(f"REFUSING: pair-static npz lacks {fld} (regenerate with --catalogue)")
        if str(PS["fixture_sha256"]) != fixture_sha:
            raise SystemExit("REFUSING: pair-static was produced on a different fixture")
        if str(PS["catalogue_sha256"]) != LP["tf_catalogue_sha256"]:
            raise SystemExit("REFUSING: pair-static catalogue != the evaluator run's tf_catalogue")
        psg = axis_check(PS["genes"], "pairstatic genes", eg); pos = {g: i for i, g in enumerate(psg)}
        psidx = np.array([pos[g] for g in eg])
        psfo = axis_check(PS["focals"], "pairstatic focals", [tfname])
        psf = [f for f in psfo if f.upper() == tfname.upper()]
        if len(psf) != 1:
            raise SystemExit("REFUSING: focal TF absent from the pair-static roster")
        psf = psf[0]
        prov["inputs"]["pairstatic"] = dict(path=a.pairstatic, sha256=sha(a.pairstatic),
                                            n_focals=int(len(PS["focals"])))
    for spec in a.sota:
        nm, path = spec.split("=", 1)
        M = np.load(path, allow_pickle=True)
        meta = json.loads(str(M["meta"])) if "meta" in M.files else {}
        if meta.get("smoke") and a.tag != "smoke":
            raise SystemExit(f"REFUSING: {nm} is a SMOKE comparator; use --tag smoke")
        if meta.get("fixture_sha256") != fixture_sha:
            raise SystemExit(f"REFUSING: {nm} does not certify the evaluator's fixture")
        sg = axis_check(M["genes"], f"{nm} genes", eg); tfs = axis_check(M["tfs"], f"{nm} tfs", [tfname])
        row = [i for i, t in enumerate(tfs) if t.upper() == tfname.upper()]
        if not row:
            print(f"[sota] {nm}: focal TF not scored -> NOT DIRECTLY COMPARABLE", flush=True); continue
        pos = {g: i for i, g in enumerate(sg)}
        v = np.full(len(eg), np.nan, np.float32)
        hit = np.array([g in pos for g in eg])
        v[hit] = M["score"][row[0]][[pos[g] for g in eg[hit]]]
        if meta.get("signed"):
            v = np.abs(v)
        comp[nm] = v
        prov["inputs"][nm] = dict(path=path, sha256=sha(path), meta=meta)

    # point estimates, every cell
    rows, dec_rows, cells = [], [], {}
    probe = KEY.format(root="", chan="", k="", axis="repression", panel="").split("|", 3)[3].rstrip("|")
    triples = sorted({(r_, c_, int(k_[1:])) for r_, c_, k_, _ in
                      (key[:-5].split("|", 3) for key in Zn.files
                       if key.endswith("|absS") and f"|{probe}|" in key)})
    for root, chan, k in triples:
        cells[(root, chan, k)] = {}
        for ax in ("repression", "activation"):
            cells[(root, chan, k)][ax] = {}
            pos = AX[ax]["pos"]
            for con, (negmask, panel) in AX[ax]["contrasts"].items():
                tag = KEY.format(root=root, chan=chan, k=k, axis=ax, panel=panel)
                if f"{tag}|absS" not in Zn.files:
                    continue
                s_gb = np.asarray(Zn[f"{tag}|absS"], float)
                m = np.asarray(Zn[f"{tag}|mask_RECOVERY"], bool) & np.isfinite(s_gb) & (pos | negmask)
                y = pos[m]
                if y.sum() < MIN_POS or (~y).sum() < MIN_NEG:
                    continue
                cand = {"absS": s_gb}
                if PS is not None and f"{root}|k{k}|{psf}" in PS.files:
                    cand["pairstatic"] = np.asarray(PS[f"{root}|k{k}|{psf}"], float)[psidx]
                cand.update(comp)
                cells[(root, chan, k)][ax][con] = (m, y, cand)
                for nm, v in cand.items():
                    vm = v[m]; ok = np.isfinite(vm)
                    yy = y[ok]; vv = vm[ok]; tt = np.nan_to_num(T[m][ok])
                    r_ = dict(root=root, channel=chan, k=k, axis=ax, contrast=con, method=nm,
                              n_pos=int(yy.sum()), n_neg=int((~yy).sum()), coverage=float(ok.mean()),
                              prevalence=float(yy.mean()) if len(yy) else np.nan)
                    if yy.sum() < MIN_POS or (~yy).sum() < MIN_NEG:
                        r_["status"] = "NO_CALL_LOW_METHOD_COVERAGE"; rows.append(r_); continue
                    r_["status"] = "OK"
                    r_["AUC"] = auc(vv, yy)
                    r_["p_MW"] = float(mannwhitneyu(vv[yy], vv[~yy], alternative="two-sided").pvalue)
                    r_["AP"] = ap_tieavg(vv, yy); r_["nAP"] = r_["AP"] / r_["prevalence"]
                    for N in NS:
                        r_[f"P@{N}"] = prec_at(vv, yy, N)
                        r_[f"p@{N}"] = (float(hypergeom.sf(int(np.floor(r_[f"P@{N}"] * N + 1e-9)) - 1,
                                                            len(yy), int(yy.sum()), N))
                                        if np.isfinite(r_[f"P@{N}"]) else np.nan)
                        r_[f"NDCG@{N}"] = ndcg_at(vv, tt, N)
                    rows.append(r_)
                    dec_rows.append(dict(root=root, channel=chan, k=k, axis=ax, contrast=con, method=nm,
                                         **deciles(vv, tt)))
    Rdf = pd.DataFrame(rows)
    ukey = ["root", "channel", "k", "axis", "contrast", "method"]
    if Rdf.duplicated(ukey).any():
        raise SystemExit("REFUSING: duplicated metric rows")
    Rdf.to_csv(f"{D}/polarity_compare.{a.tag}.tsv", sep="\t", index=False)
    pd.DataFrame(dec_rows).to_csv(f"{D}/polarity_deciles.{a.tag}.tsv", sep="\t", index=False)

    # joint complete-case bootstrap
    inferential = [nm for nm in comp if nm != "target_expr"] + (["pairstatic"] if PS is not None else [])
    fam = []
    for cell, axes in cells.items():
        for con in AX["repression"]["contrasts"]:
            if con not in axes.get("repression", {}) or con not in axes.get("activation", {}):
                continue
            (mr, yr, cr), (ma, ya, ca) = axes["repression"][con], axes["activation"][con]
            for nm in inferential:
                if nm not in cr or nm not in ca:
                    continue
                ok_r = np.isfinite(cr[nm][mr]) & np.isfinite(cr["absS"][mr])
                ok_a = np.isfinite(ca[nm][ma]) & np.isfinite(ca["absS"][ma])
                gr = np.flatnonzero(mr)[ok_r]; ga = np.flatnonzero(ma)[ok_a]      # global gene ids
                pos_r, neg_r = gr[yr[ok_r]], gr[~yr[ok_r]]
                pos_a, neg_a = ga[ya[ok_a]], ga[~ya[ok_a]]
                if min(len(pos_r), len(pos_a)) < MIN_POS or min(len(neg_r), len(neg_a)) < MIN_NEG:
                    continue
                fam.append(dict(cell=cell, contrast=con, comparator=nm, pos_r=pos_r, pos_a=pos_a,
                                neg_r=neg_r, neg_a=neg_a, gb_r=cr["absS"], m_r=cr[nm],
                                gb_a=ca["absS"], m_a=ca[nm]))
    inter = []
    names = NAMES
    if fam:
        point, stats, sd, crit, pools = joint_bootstrap(fam, a.n_boot, a.seed, t0)
        for j, f in enumerate(fam):
            r_ = dict(root=f["cell"][0], channel=f["cell"][1], k=f["cell"][2], contrast=f["contrast"],
                      comparator=f["comparator"], n_pos_rep=len(f["pos_r"]), n_pos_act=len(f["pos_a"]),
                      n_neg_rep=len(f["neg_r"]), n_neg_act=len(f["neg_a"]))
            for q, nm in enumerate(names):
                lo, hi = np.nanpercentile(stats[:, j, q], [2.5, 97.5])
                r_[nm] = point[j, q]; r_[f"{nm}_ci_lo"] = lo; r_[f"{nm}_ci_hi"] = hi
                r_[f"{nm}_simul_lo"] = point[j, q] - crit[q] * sd[j, q]
                r_[f"{nm}_simul_hi"] = point[j, q] + crit[q] * sd[j, q]
                r_[f"{nm}_sig_simul"] = bool(r_[f"{nm}_simul_lo"] > 0 or r_[f"{nm}_simul_hi"] < 0)
            inter.append(r_)
        prov["family_size"] = len(fam); prov["simul_crit_maxt"] = {names[q]: crit[q] for q in range(3)}
        prov["pools"] = pools
    Idf = pd.DataFrame(inter); Idf.to_csv(f"{D}/polarity_interaction.{a.tag}.tsv", sep="\t", index=False)
    prov["counts"] = counts; prov["wall_s"] = round(time.time() - t0, 1)
    json.dump(prov, open(f"{D}/polarity_compare.{a.tag}.provenance.json", "w"), indent=1)

    pd.set_option("display.width", 250)
    print("class counts", counts, "| family size", len(fam), "| tag", a.tag)
    ok = Rdf[Rdf.status == "OK"]
    for con in AX["repression"]["contrasts"]:
        piv = ok[ok.contrast == con].pivot_table(index=["axis", "method"], columns="k", values="AUC", aggfunc="median")
        print(f"\n{con} AUC, median over root x channel cells (every cell is in the TSV):")
        print(piv.loc[:, [c for c in piv.columns if c in (0, 2, 5, 10, 12, 14, 16, 18, 20, 25, 30)]].round(3))
    if len(Idf):
        print("\nfamily-wide simultaneous-band significant cells per comparator:")
        print(Idf.groupby(["contrast", "comparator"]).agg(
            n=("k", "size"), rep_sig=("Delta_rep_sig_simul", "sum"), D_sig=("D_interaction_sig_simul", "sum"),
            rep_med=("Delta_rep", "median"), D_med=("D_interaction", "median")).round(3))


if __name__ == "__main__":
    main()
