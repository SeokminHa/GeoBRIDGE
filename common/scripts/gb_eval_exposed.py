#!/usr/bin/env python
"""Signed evaluation of frozen truth-blind scores against ALREADY-EXPOSED development endpoints (PI fast-lane §10-§14, §21, §27;
internal review r92 Q5). DEVELOPMENT ONLY -- never used to choose D, the source rule, thresholds or any estimator hyperparameter.

usage: gb_eval_exposed.py <scores.npz>[,<scores2.npz>,...] <endpoint> <out.json>     endpoint in {GATA1, PANC_RFX6}
Several score files are evaluated on the INTERSECTION of their gene universes (internal review r94 objection 1: lanes selected targets from each
substrate's own velocity genes, so A/A-prime/B universes differed; any lane comparison must be on common support).

Semantics (internal review r92 Q5):
  activation is scored with T_plus, repression with T_minus; absolute/unsigned scores are never used for a signed panel.
  One truth-evaluable universe per endpoint, identical for every arm: genes present in the score matrix AND labelled by the endpoint
  (y != -1). Unknown genes stay unknown and are excluded from both numerator and denominator; the fraction excluded is reported.
  Local estimates that never resolved are reported as unresolved coverage; they keep their aggregated value (T_plus/T_minus treat a
  non-finite local z as no evidence) and the coverage columns say how often that happened.
  Reported per arm: AP, AP/prevalence lift, precision@50/100, n_pos, n_universe, and Delta = AP(dynamic) - AP(matched static twin).
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gb_dev_evaluate as E

ARMS = ("D1", "D2", "D3")
CHIP = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/results/gata1_clean_2026_08_13/chipseq_labels.tsv")
ISL1_ENDPOINT = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/data/isl1_GSE206093/isl1_endpoint.tsv")


def truth_isl1(genes):
    """frozen GSE206093 E14.5 Isl1 cKO endpoint (gb_isl1_endpoint.py): y in {1 positive, 0 tested negative, -1 unknown},
    polarity -1 = activation (KO-down), +1 = repression (KO-up)"""
    import csv
    lab = {}
    with open(ISL1_ENDPOINT) as fh:
        for r in csv.DictReader(fh, delimiter="\t"): lab[r["gene"]] = (int(r["y"]), float(r["polarity"]))
    y = np.array([lab.get(g, (-1, 0.0))[0] for g in genes])
    pol = np.array([lab.get(g, (-1, 0.0))[1] for g in genes])
    return y, pol


def gata1_four_class(genes):
    """canonical GATA1 KO+ChIP four-class benchmark (FOURCLASS_GATA1_2026_09_04; PI 2026-09-14 correction).
    A = ChIP-bound + KO-responsive (binding-supported functional); response-only = KO-responsive without ChIP annotation;
    bound-nonresponsive = ChIP-bound + not responsive; D = neither. Activation = KO-down, repression = KO-up.
    Panels (each: universe mask, positive mask, polarity) -- A_vs_rest, A_vs_response_only (binding specificity within responders),
    A_vs_bound_nonresponsive (function within bound), and the response-only panel as a separately named sensitivity endpoint."""
    import csv
    col = {}
    with open(CHIP) as fh:
        for r in csv.DictReader(fh, delimiter="\t"): col[r["gene"]] = {k: int(v) for k, v in r.items() if k != "gene"}
    g = np.array(genes); known = np.array([x in col for x in genes])
    def f(key): return np.array([bool(col.get(x, {}).get(key, 0)) for x in genes])
    self_ = np.array([x.upper() == "GATA1" for x in genes])
    bound, up, down = f("ChIP_bound"), f("cons_up"), f("cons_down")
    A_act, A_rep = f("ChIP_cons_down") & ~self_, f("ChIP_cons_up") & ~self_
    only_act, only_rep = f("cons_down_noChIP") & ~self_, f("cons_up_noChIP") & ~self_
    bound_nonresp = bound & ~up & ~down & ~self_
    neither = known & ~bound & ~up & ~down & ~self_
    P = {}
    for pol, A_, only_, in (("activation", A_act, only_act), ("repression", A_rep, only_rep)):
        base = known & ~self_
        P[f"{pol}|A_vs_rest"] = (base & (A_ | bound_nonresp | neither), A_, pol)
        P[f"{pol}|A_vs_response_only"] = (A_ | only_, A_, pol)
        P[f"{pol}|A_vs_bound_nonresponsive"] = (A_ | bound_nonresp, A_, pol)
        P[f"{pol}|response_only_vs_rest"] = (base & (only_ | bound_nonresp | neither), only_, pol)
    return P
BASELINES = ("EXPR_ASSOC",)


def ap(score, pos):
    """average precision with ties grouped at equal score (threshold-grouped, internal review r92 Q5)"""
    order = np.lexsort((np.arange(len(score)), -score))
    s, p = score[order], pos[order]
    groups, i = [], 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]: j += 1
        groups.append((i, j)); i = j + 1
    tp = 0; total = pos.sum(); out = 0.0
    for i, j in groups:
        k = int(p[i:j + 1].sum()); n = j - i + 1
        if k:
            prec = (tp + k) / (j + 1)
            out += prec * k
        tp += k
    return out / total if total else None


def paired_permutation(scores, pos, n_perm=9999, seed=20260914):
    """PI 2026-09-14 §uncertainty: paired label permutation on the SAME universe, identical permutation for every arm.
    Returns per-arm AP and, for each (arm, reference) pair, the two-sided p of the observed AP difference under the
    permutation distribution of that same difference. Arms are evaluated on identical genes, so a gene bootstrap would
    ignore the pairing; permuting the positive labels preserves the number of positives and the score vectors."""
    g = np.random.default_rng(seed); n_pos = int(pos.sum()); n = len(pos)
    obs = {k: ap(v, pos) for k, v in scores.items()}
    diffs = {k: [] for k in scores}
    keys = list(scores)
    perm_ap = {k: np.empty(n_perm) for k in keys}
    for b in range(n_perm):
        idx = g.permutation(n)[:n_pos]; pb = np.zeros(n, bool); pb[idx] = True
        for k in keys: perm_ap[k][b] = ap(scores[k], pb)
    return obs, perm_ap


def two_sided_p(obs_diff, perm_diff):
    return float((1 + np.sum(np.abs(perm_diff) >= abs(obs_diff))) / (len(perm_diff) + 1))


def precision_at(score, pos, k):
    order = np.lexsort((np.arange(len(score)), -score))[:k]
    return float(pos[order].mean()) if k else None


STATIC_REFERENCE = "EXPR_ASSOC"          # PRESPECIFIED before evaluation (internal review r99 obj.4 / r100 obj.4): never chosen by observed AP


def incremental_cv(dyn, stat, pos, n_folds=5, seed=20260915):
    """Cross-fitted incremental-information test (internal review r99/r100, internal review r138): does the dynamic score improve held-out target ranking
    OVER a PRESPECIFIED strong static score? Genes are split into stratified folds; on each training split we fit logistic models
    y ~ static and y ~ static + dynamic (standardised on the TRAINING split only); predictions are made on the held-out split.
    Out-of-fold AP of the two models is compared, and the per-fold paired difference gives the uncertainty. No statistic is chosen
    using the observed labels, and no gene contributes to its own model fit.
    `stat` may be a single static score or a MATRIX of static covariates (n x p) -- internal review r101 required the increment to be
    measured over the COMBINED static information, because on Isl1 the secondary reference D_static (AP .248) beat the
    primary EXPR_ASSOC (AP .178), so a gain over either one alone can be compensation for an incomplete static predictor."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    stat = np.asarray(stat, float)
    if stat.ndim == 1: stat = stat[:, None]
    y = pos.astype(int); n = len(y)
    if y.sum() < n_folds or (n - y.sum()) < n_folds: return None
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    oof_s, oof_d = np.full(n, np.nan), np.full(n, np.nan); per_fold = []
    for tr, te in skf.split(np.zeros(n), y):
        mu, sd = dyn[tr].mean(), dyn[tr].std() + 1e-12
        ms, ss = stat[tr].mean(0), stat[tr].std(0) + 1e-12
        Ztr_s = (stat[tr] - ms) / ss; Zte_s = (stat[te] - ms) / ss
        Ztr_d = np.column_stack([Ztr_s, (dyn[tr] - mu) / sd]); Zte_d = np.column_stack([Zte_s, (dyn[te] - mu) / sd])
        m_s = LogisticRegression(max_iter=1000).fit(Ztr_s, y[tr]); m_d = LogisticRegression(max_iter=1000).fit(Ztr_d, y[tr])
        oof_s[te] = m_s.predict_proba(Zte_s)[:, 1]; oof_d[te] = m_d.predict_proba(Zte_d)[:, 1]
        if y[te].sum() > 0:
            a_s, a_d = ap(oof_s[te], pos[te]), ap(oof_d[te], pos[te])
            if a_s is not None and a_d is not None: per_fold.append(a_d - a_s)
    a_s, a_d = ap(oof_s, pos), ap(oof_d, pos)
    pf = np.array(per_fold)
    return dict(AP_static_only=a_s, AP_static_plus_dynamic=a_d, delta_oof=(None if (a_s is None or a_d is None) else a_d - a_s),
                per_fold_delta=[float(v) for v in pf], mean_fold_delta=float(pf.mean()) if len(pf) else None,
                fold_lower_bound=float(pf.mean() - 1.96 * pf.std(ddof=1) / np.sqrt(len(pf))) if len(pf) > 1 else None,
                n_folds=int(len(pf)), prevalence=float(pos.mean()), static_reference=STATIC_REFERENCE)


def incremental_beyond_static(dyn, stat, pos, n_bins=10):
    """PI 2026-09-14 §V: does the dynamic score carry target information BEYOND strong static information?
    Two truth-blind-construction readouts on the identical universe (no model fitting over cells):
      residual : rank-regress the dynamic score on the static score across genes (isotonic-free: subtract the binned static mean)
                 and score AP of the residual -- information not explained by a monotone function of the static score;
      stratified: AP of the dynamic score computed WITHIN static-score deciles and pooled by positive count -- the static ranking
                 cannot contribute because it is constant within a bin.
    Returns AP(dynamic), AP(static), AP(residual), stratified AP and the prevalence for reference."""
    rd = np.argsort(np.argsort(dyn)) / max(len(dyn) - 1, 1)
    rs = np.argsort(np.argsort(stat)) / max(len(stat) - 1, 1)
    edges = np.quantile(rs, np.linspace(0, 1, n_bins + 1)[1:-1])
    b = np.searchsorted(edges, rs)
    resid = rd - np.array([rd[b == k].mean() if (b == k).any() else 0.0 for k in range(n_bins)])[b]
    strat_num, strat_den = 0.0, 0
    for k in range(n_bins):
        m = b == k
        if m.sum() < 5 or pos[m].sum() == 0: continue
        a_ = ap(dyn[m], pos[m])
        if a_ is not None: strat_num += a_ * pos[m].sum(); strat_den += pos[m].sum()
    return dict(AP_dynamic=ap(dyn, pos), AP_static=ap(stat, pos), AP_residual_of_dynamic_given_static=ap(resid, pos),
                AP_stratified_within_static_deciles=(strat_num / strat_den) if strat_den else None,
                prevalence=float(pos.mean()), n_pos=int(pos.sum()))


def prefixes_of(Z):
    return sorted({k.split("::")[0] + "::" for k in Z if "::" in k}) or [""]


def four_class_main(Z, genes, recs, paths, prefixes, out):
    P = gata1_four_class(genes)
    res = dict(endpoint="GATA1_4CLASS", label="DEVELOPMENT ONLY; canonical GATA1 KO+ChIP four-class benchmark; activation and repression separate",
               sources=[dict(file=os.path.basename(p_), lane=rc.get("lane"), layer=rc.get("layer"), D=rc["D"], npz_hash=rc["npz_hash"]) for p_, rc in zip(paths, recs)],
               n_common_genes=len(genes), chip_labels=CHIP, panels={})
    for name, (uni, pos_, pol) in P.items():
        key = "T_plus" if pol == "activation" else "T_minus"
        pos = pos_[uni]
        if pos.sum() < 5 or (~pos).sum() < 5: res["panels"][name] = f"TOO_FEW (pos {int(pos.sum())}, neg {int((~pos).sum())})"; continue
        row = dict(n_universe=int(uni.sum()), n_pos=int(pos.sum()), prevalence=float(pos.mean()))
        for pre in prefixes:
            for arm in list(ARMS) + [a + "_STATIC" for a in ARMS] + list(BASELINES):
                k_ = f"{pre}{arm}|{key}"
                if k_ not in Z: continue
                sc = np.asarray(Z[k_], float)[uni]
                row[pre + arm] = dict(AP=ap(sc, pos), lift=(ap(sc, pos) / pos.mean()) if pos.mean() else None, p_at_50=precision_at(sc, pos, 50))
            for arm in ARMS:
                if pre + arm in row:
                    stat = [row[k]["AP"] for k in row if isinstance(row[k], dict) and k.startswith(pre) and ("_STATIC" in k or k.endswith(tuple(BASELINES)))]
                    if pre + arm + "_STATIC" in row: row[pre + arm]["delta_vs_matched_static"] = row[pre + arm]["AP"] - row[pre + arm + "_STATIC"]["AP"]
                    if stat: row[pre + arm]["delta_vs_best_static"] = row[pre + arm]["AP"] - max(stat)
        if os.environ.get("GB_PERM"):
            npm = int(os.environ["GB_PERM"])
            sel = {k: np.asarray(Z[f"{k}|{key}"], float)[uni] for k in
                   [c for c in row if isinstance(row[c], dict) and ("::D1" in c or "::D2" in c or "::D3" in c or c.endswith("EXPR_ASSOC") or "_STATIC" in c)]
                   if f"{k}|{key}" in Z}
            if sel:
                obs, perm = paired_permutation(sel, pos, n_perm=npm)
                stat_keys = [k for k in sel if "_STATIC" in k or k.endswith(tuple(BASELINES))]
                for k in sel:
                    if "_STATIC" in k or k.endswith(tuple(BASELINES)): continue
                    best_ref = max(stat_keys, key=lambda r: obs[r]) if stat_keys else None
                    if best_ref:
                        row[k]["perm_p_vs_best_static"] = two_sided_p(obs[k] - obs[best_ref], perm[k] - perm[best_ref])
                        row[k]["perm_reference"] = best_ref
                    row[k]["perm_p_vs_random"] = two_sided_p(obs[k] - float(np.mean(perm[k])), perm[k] - float(np.mean(perm[k])))
                row["n_perm"] = npm
        if os.environ.get("GB_INCREMENTAL"):
            inc = {}
            # PRESPECIFIED reference (never the best-observed one); fall back to the matched static twin only if it is absent
            ref_keys = [k for k in row if isinstance(row[k], dict) and k.endswith(STATIC_REFERENCE)]
            stat_keys = ref_keys or [k for k in row if isinstance(row[k], dict) and "_STATIC" in k]
            if stat_keys:
                bestk = sorted(stat_keys)[0]
                stat_v = np.asarray(Z[f"{bestk}|{key}"], float)[uni]
                for k in [c for c in row if isinstance(row[c], dict) and "_STATIC" not in c and not c.endswith(tuple(BASELINES))]:
                    if f"{k}|{key}" in Z:
                        dv = np.asarray(Z[f"{k}|{key}"], float)[uni]
                        inc[k] = incremental_beyond_static(dv, stat_v, pos)
                        inc[k]["static_reference"] = bestk
                        inc[k]["cross_fitted"] = incremental_cv(dv, stat_v, pos)
            row["incremental"] = inc
        res["panels"][name] = row
    with open(out, "x") as fh: json.dump(res, fh, indent=1)
    for name, row in res["panels"].items():
        if isinstance(row, str): print(f"{name}: {row}"); continue
        print(f"\n{name}: universe {row['n_universe']} pos {row['n_pos']} prevalence {row['prevalence']:.3f}")
        rows = [(k, v) for k, v in row.items() if isinstance(v, dict) and "_STATIC" not in k]
        for k, v in sorted(rows, key=lambda kv: -(kv[1]["AP"] or 0))[:6]:
            d1 = f" d_match {v['delta_vs_matched_static']:+.4f}" if "delta_vs_matched_static" in v else ""
            d2 = f" d_best {v['delta_vs_best_static']:+.4f}" if "delta_vs_best_static" in v else ""
            pp = f" p_vs_static {v['perm_p_vs_best_static']:.4f}" if "perm_p_vs_best_static" in v else ""
            pr = f" p_vs_random {v['perm_p_vs_random']:.4f}" if "perm_p_vs_random" in v else ""
            print(f"   {k:34s} AP {v['AP']:.4f} lift {v['lift']:.2f}{d1}{d2}{pp}{pr}")
    return res


def main(npz_paths, endpoint, out):
    paths = npz_paths.split(",")
    Zs, recs = [], []
    for p_ in paths:
        Z_ = np.load(p_, allow_pickle=True); Zs.append(Z_); recs.append(json.load(open(p_.replace(".npz", ".json"))))
    common = set.intersection(*[{str(g) for g in Z_["genes"]} for Z_ in Zs])
    genes = [g for g in (str(g) for g in Zs[0]["genes"]) if g in common]
    idx = [{str(g): i for i, g in enumerate(Z_["genes"])} for Z_ in Zs]
    Z = {}
    for Z_, ix, rc in zip(Zs, idx, recs):
        tag = f"{rc.get('lane','A')}|{rc.get('layer','L1')}|D{rc['D']}" if len(paths) > 1 else ""
        sel = np.array([ix[g] for g in genes])
        for key in Z_.files:
            if "|" in key and not key.startswith("unresolved"):
                Z[(tag + "::" if tag else "") + key] = np.asarray(Z_[key], float)[sel]
    rec = recs[0]
    if endpoint == "GATA1_4CLASS":
        return four_class_main(Z, genes, recs, paths, prefixes_of(Z), out)
    y, pol = (truth_isl1(genes) if endpoint == "ISL1" else E.TRUTH[endpoint](genes))
    tested = y != -1
    act = (y == 1) & (pol == -1.0); rep = (y == 1) & (pol == +1.0)
    res = dict(scores=[os.path.basename(p_) for p_ in paths], endpoint=endpoint, dataset=rec["dataset"], tf=rec["tf"], D=rec["D"], metric=rec["metric"],
               lane=rec["lane"], label="DEVELOPMENT ONLY; scores frozen before this evaluation (npz " + rec["npz_hash"] + ")",
               n_universe=int(tested.sum()), n_unknown=int((~tested).sum()), n_activation=int(act[tested].sum()), n_repression=int(rep[tested].sum()),
               n_anchor=rec["n_anchor"], anchor_basis=rec["anchor_basis"], finite_frac=rec["finite_frac"],
               sources=[dict(file=os.path.basename(p_), lane=rc.get("lane"), layer=rc.get("layer"), D=rc["D"], npz_hash=rc["npz_hash"],
                             substrate_hash=rc.get("substrate_hash"), n_genes=rc["n_genes"]) for p_, rc in zip(paths, recs)],
               n_common_genes=len(genes), arms={})
    prefixes = prefixes_of(Z)
    for polarity, posmask, key in (("activation", act, "T_plus"), ("repression", rep, "T_minus")):
        pos = posmask[tested]
        if not pos.any(): res["arms"][polarity] = "NO_POSITIVES_IN_UNIVERSE"; continue
        row = {}
        for pre in prefixes:
            for arm in ARMS:
                for sfx in ("", "_STATIC"):
                    k_ = f"{pre}{arm}{sfx}|{key}"
                    if k_ not in Z: continue
                    s = np.asarray(Z[k_], float)[tested]
                    row[pre + arm + sfx] = dict(AP=ap(s, pos), lift=(ap(s, pos) / pos.mean()) if pos.mean() else None,
                                                p_at_50=precision_at(s, pos, 50), p_at_100=precision_at(s, pos, 100))
                if pre + arm in row and pre + arm + "_STATIC" in row:
                    a_d, a_s = row[pre + arm]["AP"], row[pre + arm + "_STATIC"]["AP"]
                    row[pre + arm]["delta_vs_matched_static"] = None if (a_d is None or a_s is None) else a_d - a_s
            for b in BASELINES:
                k_ = f"{pre}{b}|{key}"
                if k_ in Z:
                    s = np.asarray(Z[k_], float)[tested]
                    row[pre + b] = dict(AP=ap(s, pos), lift=(ap(s, pos) / pos.mean()) if pos.mean() else None,
                                        p_at_50=precision_at(s, pos, 50), p_at_100=precision_at(s, pos, 100))
            best_static = max([row[k]["AP"] for k in row if k.startswith(pre) and ("_STATIC" in k or k.endswith(tuple(BASELINES)))] or [None],
                              default=None, key=lambda v: (v is not None, v))
            for arm in ARMS:
                if pre + arm in row and best_static is not None:
                    row[pre + arm]["delta_vs_best_static"] = row[pre + arm]["AP"] - best_static
        row["prevalence"] = float(pos.mean()); row["n_pos"] = int(pos.sum())
        res["arms"][polarity] = row
    with open(out, "x") as fh: json.dump(res, fh, indent=1)
    print(f"{rec['dataset']}|{rec['tf']} D{rec['D']} {endpoint}  universe {res['n_universe']} (unknown {res['n_unknown']}) "
          f"act {res['n_activation']} rep {res['n_repression']} anchors {res['n_anchor']} [{res['anchor_basis']}]")
    for polarity in ("activation", "repression"):
        r = res["arms"][polarity]
        if isinstance(r, str): print(f"  {polarity}: {r}"); continue
        print(f"  {polarity}: prevalence {r['prevalence']:.3f} (n {r['n_pos']})")
        for k_ in sorted(r):
            if not isinstance(r[k_], dict) or "_STATIC" in k_: continue
            d = r[k_]
            extra = "" if "delta_vs_matched_static" not in d else f" | d_matched {d['delta_vs_matched_static']:+.4f}"
            extra += "" if "delta_vs_best_static" not in d else f" d_best {d['delta_vs_best_static']:+.4f}"
            print(f"    {k_:34s} AP {d['AP']:.4f} lift {d['lift']:.2f} P@50 {d['p_at_50']:.2f}{extra}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
