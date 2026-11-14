#!/usr/bin/env python
# Historical GATA1 regulator-specificity evaluation.
import json, os, sys, time
import numpy as np, pandas as pd
import anndata as ad
from scipy.sparse import issparse
R = __import__("os").environ["GEOBRIDGE_ROOT"]; sys.path.insert(0, R); sys.path.insert(0, f"{R}/scratch")
from geobridge_model.scoring import score_matrix
import gb_m1_calib3 as C3
import gb_eval_exposed as EV

FIXTURE = f"{R}/results/published_reference_fixture_2026_08_16/published_reference_out.h5ad"
PAIRS = f"{R}/results/gata1_pr_arm2_surfaces_2026_08_16/gata1_pr_frozen_pairs.npz"
G = f"{R}/results/gata1_clean_2026_08_13"
TF_ROSTER = f"{R}/results/mouse_tf_universe_2026_09_06/mouse_tfs_one2one.txt"
K = 31; SEED = 0; N_HELDOUT = 50; N_SCRAMBLE = 200
USEFUL_EXCESS = 0.5      # Historical useful-excess threshold.

CHANNELS = ("S", "U", "V")
ISVD_CHANNELS = ("U", "V")
N_BOOT = 2000

d_ = lambda M: np.asarray(M.todense()) if issparse(M) else np.asarray(M)


def nap(y, s):
    m = np.isfinite(s); y2, s2 = y[m], s[m]; p = int(y2.sum())
    if p < 3 or p == y2.size: return np.nan
    o = np.argsort(-s2, kind="stable"); yo = y2[o]
    return float((np.cumsum(yo) / np.arange(1, yo.size + 1) * yo).sum() / p) / (p / y2.size)


def hist_classes(genes, ti):
    dfc = pd.read_csv(f"{G}/chipseq_labels.tsv", sep="\t")
    chip = dict(zip(dfc["gene"].astype(str), dfc["ChIP_bound"].astype(float)))
    W_ = ad.read_h5ad(f"{G}/gata1_control_deterministic.h5ad"); K_ = ad.read_h5ad(f"{G}/gata1_ko_projected_deterministic.h5ad")
    T_ = -np.log2((np.expm1(d_(K_.X)).mean(0) + 1e-6) / (np.expm1(d_(W_.X)).mean(0) + 1e-6))
    tmap = dict(zip(np.asarray(W_.var_names).astype(str), T_))
    cb = np.array([chip.get(g, np.nan) for g in genes]); ko = np.array([tmap.get(g, np.nan) for g in genes])
    bound = cb == 1.0; ra = ko > 0.5; rr = ko < -0.5; ran = ra | rr
    out = {"HIST|activation|A_vs_B": (bound & ra, bound & ra | bound & ~ran, +1.0),
           "HIST|repression|A_vs_B": (bound & rr, bound & rr | bound & ~ran, -1.0)}
    for k, (pos, uni, pol) in out.items(): pos[ti] = False; uni[ti] = False
    return out


def canon_classes(genes, ti):
    P = EV.gata1_four_class(list(genes)); out = {}
    for pol_name, pol in (("activation", +1.0), ("repression", -1.0)):
        uni, pos, _ = P[f"{pol_name}|A_vs_bound_nonresponsive"]
        uni = uni.copy(); pos = pos.copy(); uni[ti] = False; pos[ti] = False
        out[f"CANON|{pol_name}|A_vs_B"] = (pos, uni, pol)
    return out


def main(out_dir, n_pseudo=1000):
    t0 = time.time(); os.makedirs(out_dir, exist_ok=True); n_pseudo = int(n_pseudo)
    A = ad.read_h5ad(FIXTURE)
    genes = np.asarray(A.var_names).astype(str); gU = np.char.upper(genes); ti = {g: i for i, g in enumerate(gU)}["GATA1"]
    X = d_(A.X).astype(np.float64); Um = d_(A.layers["Mu"]).astype(np.float64); Vm = d_(A.layers["velocity"]).astype(np.float64)
    f = np.load(PAIRS)
    # truth-blind rosters (frozen before any label is read)
    mu = X.mean(0); det = (X > 0).mean(0); z = lambda v: (v - v.mean()) / (v.std() + 1e-12)
    dist = np.sqrt((z(mu) - z(mu)[ti]) ** 2 + (z(det) - z(det)[ti]) ** 2)
    elig = np.ones(len(genes), bool); elig[ti] = False; elig[~np.isfinite(dist)] = False
    Zx = (X - X.mean(0)) / (X.std(0) + 1e-12); corr_gata1 = (Zx * Zx[:, [ti]]).mean(0)
    tfs = {l.strip().upper() for l in open(TF_ROSTER) if l.strip()}
    is_tf = np.array([g in tfs for g in gU])
    rosters = {"ALL": np.argsort(np.where(elig, dist, np.inf))[:n_pseudo],
               "TF": np.argsort(np.where(elig & is_tf, dist, np.inf))[:min(n_pseudo, int((elig & is_tf).sum()))]}
    bal = {r: dict(n=int(len(c)), mu=[float(mu[c].min()), float(mu[c].max()), float(mu[ti])],
                   det=[float(det[c].min()), float(det[c].max()), float(det[ti])],
                   frac_tf=float(is_tf[c].mean()), abs_corr_with_gata1_median=float(np.median(np.abs(corr_gata1[c]))),
                   abs_corr_with_gata1_q90=float(np.quantile(np.abs(corr_gata1[c]), .9))) for r, c in rosters.items()}
    # scores for focal + every pseudo, full family, before truth
    S = {}      # (root, ch, k) -> dict(real=(n_genes,), pseudo={roster: (n_pseudo, n_genes)})
    MODES = {}  # root -> mode-wise additive numerator contributions (development localisation, truth-free)
    for root in ("dc1_min", "dc1_max"):
        I, J = f[f"{root}_I"].astype(np.int64), f[f"{root}_J"].astype(np.int64)
        dX_raw = X[J] - X[I]; dXc = dX_raw - dX_raw.mean(0, keepdims=True)
        U_, S_, Vt = np.linalg.svd(dXc, full_matrices=False); Uk = U_[:, :K]
        proj = lambda Y: Y - Uk @ (Uk.T @ Y)
        dXc_k = dXc - (Uk * S_[:K][None, :]) @ Vt[:K, :]
        allc = np.unique(np.concatenate([[ti]] + list(rosters.values())))
        pos_of = {c: i for i, c in enumerate(allc)}
        for ch, Y in (("S", X), ("U", Um), ("V", Vm)):
            dY = Y[J] - Y[I]
            for kk, (Amat, resp) in (("k0", (dX_raw, dY)), (f"k{K}", (dXc_k, proj(dY)))):
                Sm = score_matrix(Amat[:, allc], resp)                    # one product for focal + every pseudo
                S[(root, ch, kk)] = dict(real=Sm[pos_of[ti]], pseudo={r: Sm[[pos_of[c] for c in cand]] for r, cand in rosters.items()})
        dY = Vm[J] - Vm[I]
        mode_num = (S_[:K] * Vt[:K, ti])[:, None] * (Uk.T @ dY)                  # (K, n_genes): mode-wise numerator, focal source
        resid_num = (dXc_k[:, ti] @ proj(dY))                                     # residual numerator (what k31 keeps)
        raw_num = dXc[:, ti] @ dY                                                 # centred raw numerator
        MODES[root] = dict(mode_num=mode_num, resid_num=resid_num, raw_num=raw_num,
                           additivity_max_abs_err=float(np.abs(mode_num.sum(0) + resid_num - raw_num).max()), sv=S_[:K].copy())
        print(root, "scored", f"{time.time() - t0:.0f}s", flush=True)
    h = C3.SEL.raw_hash(b"".join(f"{k}|real|{v['real'].shape}".encode() + np.ascontiguousarray(v["real"]).tobytes()
                                 + b"".join(f"{k}|{r}|{v['pseudo'][r].shape}".encode() + np.ascontiguousarray(v["pseudo"][r]).tobytes()
                                            for r in sorted(v["pseudo"])) for k, v in sorted(S.items())))
    score_hash = h["hex16"]
    # truth opened only now
    CLS = {**hist_classes(genes, ti), **canon_classes(genes, ti)}
    FAMILY = {"CONFIRM_HIST": lambda ch, kk, cn: cn.startswith("HIST|"), "SECONDARY_CANON": lambda ch, kk, cn: cn.startswith("CANON|")}
    cells = [(root, ch, kk, cname) for (root, ch, kk) in S for cname in CLS]
    fam_of = np.array([next(fn for fn, pred in FAMILY.items() if pred(ch, kk, cn)) for (root, ch, kk, cn) in cells])

    minp = M_minp

    rng = np.random.default_rng(SEED); rows = []; res = {}; NULLSAVE = {}
    for roster, cand in rosters.items():
        n_p = len(cand)
        real = np.full(len(cells), np.nan); P = np.full((n_p, len(cells)), np.nan); scr = np.full((N_SCRAMBLE, len(cells)), np.nan)
        for ci, (root, ch, kk, cname) in enumerate(cells):
            pos, uni, pol = CLS[cname]; y = pos[uni].astype(int)
            sr = S[(root, ch, kk)]["real"]; sp = S[(root, ch, kk)]["pseudo"][roster]
            real[ci] = nap(y, pol * sr[uni])
            for q in range(n_p): P[q, ci] = nap(y, pol * sp[q][uni])
            for b in range(N_SCRAMBLE): scr[b, ci] = nap(rng.permutation(y), pol * sr[uni])
        res[roster] = dict(balance=bal[roster], role=("INFERENTIAL" if roster == "ALL" else "SENSITIVITY_ONLY"), families={})
        if roster == "ALL":
            head = [(r_, "V", f"k{K}", "HIST|repression|A_vs_B") for r_ in ("dc1_min", "dc1_max")]
            hcols = [cells.index(h) for h in head]
            gb_rng = np.random.default_rng(SEED + 7); boot = np.full((N_BOOT, 2), np.nan)
            pos0, uni0, pol0 = CLS["HIST|repression|A_vs_B"]; idxu = np.flatnonzero(uni0)
            for bi in range(N_BOOT):
                samp = gb_rng.choice(idxu, size=len(idxu), replace=True)     # one multiset, shared by both roots
                yb = pos0[samp].astype(int)
                if yb.sum() < 3 or yb.sum() == yb.size: continue
                for jj, (root, ch, kk, cname) in enumerate(head):
                    sr = S[(root, ch, kk)]["real"]; sp = S[(root, ch, kk)]["pseudo"][roster]
                    rb = nap(yb, pol0 * sr[samp])
                    pb = np.array([nap(yb, pol0 * sp[q][samp]) for q in range(n_p)])
                    if np.isfinite(rb): boot[bi, jj] = rb - np.nanmedian(pb)
            E = np.array([real[ci] - np.nanmedian(P[:, ci]) for ci in hcols])
            sd = np.nanstd(boot, axis=0, ddof=1)
            if np.all(np.isfinite(sd)) and np.all(sd > 0):
                t = (E[None, :] - boot) / sd[None, :]
                crit = float(np.nanquantile(np.nanmax(t, axis=1), .95, method="higher"))
                band = {f"{h[0]}|{h[1]}|{h[2]}|{h[3]}": dict(excess=float(E[jj]), boot_sd=float(sd[jj]),
                        simultaneous_upper=float(E[jj] + crit * sd[jj]), simultaneous_lower=float(E[jj] - crit * sd[jj]))
                        for jj, h in enumerate(head)}
                status = "OK"
            else:
                crit = float("nan"); band = {}; status = "UNRESOLVED"
            res[roster]["simultaneous_band"] = dict(n_boot=int(N_BOOT), unit="target gene", status=status,
                crit_studentized_maxt_95=crit, cells=band, n_boot_finite=int(np.isfinite(boot).all(1).sum()),
                construction="two frozen headline cells, ALL roster, one common gene multiset per replicate shared by both roots and every source row, studentized max-t, quantile method 'higher'",
                rule="FAIL/RETIRE only if BOTH headline-root upper limits < frozen useful excess 0.5; else NO_CALL")
        # cell families (CONFIRM_HIST, SECONDARY_CANON), each corrected on its own
        NULLSAVE[roster] = dict(focal=real.copy(), pseudo=P.copy())
        for fname in FAMILY:
            cols = np.flatnonzero(fam_of == fname)
            pm, pf, rank, rp, mn = minp(real[cols], P[:, cols])
            NULLSAVE[roster][f"rank_p|{fname}"] = rp; NULLSAVE[roster][f"minp|{fname}"] = mn
            NULLSAVE[roster][f"cols|{fname}"] = cols
            held = rng.choice(n_p, size=min(N_HELDOUT, n_p), replace=False); fp = 0
            for hh in held:
                if not np.isfinite(mn[hh]): continue
                fp += int(np.any(np.array([(1 + np.sum(np.delete(mn, hh)[np.isfinite(np.delete(mn, hh))] <= rp[hh, c])) / (1 + np.isfinite(np.delete(mn, hh)).sum())
                                           for c in range(len(cols))]) <= .05))
            res[roster]["families"][fname] = dict(n_cells=int(len(cols)), heldout_pseudofocal_fw_rate_DESCRIPTIVE=fp / len(held), n_heldout=int(len(held)),
                                                  scrambled_target_nAP_median=float(np.nanmedian(scr[:, cols])), scrambled_target_nAP_q95=float(np.nanquantile(scr[:, cols], .95)))
            for j, ci in enumerate(cols):
                root, ch, kk, cname = cells[ci]
                rows.append(dict(roster=roster, family=fname, root=root, channel=ch, k=kk, cell=cname, nAP_real=real[ci],
                                 pseudo_median=float(np.nanmedian(P[:, ci])), pseudo_q95=float(np.nanquantile(P[:, ci], .95)),
                                 excess=float(real[ci] - np.nanmedian(P[:, ci])), p_marginal=float(pm[j]), p_familywise=float(pf[j]),
                                 rank_among_pseudos=int(rank[j]), n_pseudo_finite=int(np.isfinite(P[:, ci]).sum()),
                                 exceeds_frozen_useful=bool(real[ci] - np.nanmedian(P[:, ci]) >= USEFUL_EXCESS)))
        # contrast families, built per pseudo row so the joint null is retained
        contrasts = {"DELTA_DYNAMIC": ([], [], [], False), "I_SVD": ([], [], [], True)}
        for root in ("dc1_min", "dc1_max"):
            for cname in CLS:
                for kk in ("k0", f"k{K}"):
                    iv = cells.index((root, "V", kk, cname)); is_ = cells.index((root, "S", kk, cname))
                    contrasts["DELTA_DYNAMIC"][0].append(real[iv] - real[is_]); contrasts["DELTA_DYNAMIC"][1].append(P[:, iv] - P[:, is_])
                    contrasts["DELTA_DYNAMIC"][2].append((root, "V_minus_S", kk, cname))
            for ch in ISVD_CHANNELS:
                for tdef in ("HIST", "CANON"):    # Evaluate each selected channel against both truth definitions.
                    ra = cells.index((root, ch, f"k{K}", f"{tdef}|repression|A_vs_B")); r0 = cells.index((root, ch, "k0", f"{tdef}|repression|A_vs_B"))
                    aa = cells.index((root, ch, f"k{K}", f"{tdef}|activation|A_vs_B")); a0 = cells.index((root, ch, "k0", f"{tdef}|activation|A_vs_B"))
                    contrasts["I_SVD"][0].append((real[ra] - real[r0]) - (real[aa] - real[a0]))
                    contrasts["I_SVD"][1].append((P[:, ra] - P[:, r0]) - (P[:, aa] - P[:, a0]))
                    contrasts["I_SVD"][2].append((root, ch, f"k{K}_minus_k0", f"{tdef}|interaction"))
        for fname, (obs_l, null_l, lab, two) in contrasts.items():
            obs = np.array(obs_l); NULL = np.column_stack(null_l)
            pm, pf, rank, rp, mn = minp(obs, NULL, two_sided=two)
            NULLSAVE[roster][f"focal|{fname}"] = obs; NULLSAVE[roster][f"pseudo|{fname}"] = NULL
            NULLSAVE[roster][f"rank_p|{fname}"] = rp; NULLSAVE[roster][f"minp|{fname}"] = mn
            res[roster]["families"][fname] = dict(n_cells=int(len(obs)))
            for j, (root, ch, kk, cname) in enumerate(lab):
                rows.append(dict(roster=roster, family=fname, root=root, channel=ch, k=kk, cell=cname, nAP_real=float(obs[j]),
                                 pseudo_median=float(np.nanmedian(NULL[:, j])), pseudo_q95=float(np.nanquantile(NULL[:, j], .95)),
                                 excess=float(obs[j] - np.nanmedian(NULL[:, j])), p_marginal=float(pm[j]), p_familywise=float(pf[j]),
                                 rank_among_pseudos=int(rank[j]), n_pseudo_finite=int(np.isfinite(NULL[:, j]).sum()), exceeds_frozen_useful=False))
    dev = {}
    for root, M in MODES.items():
        pos, uni, pol = CLS["HIST|repression|A_vs_B"]; A_ = pos & uni; B_ = uni & ~pos
        contrast = M["mode_num"][:, A_].mean(1) - M["mode_num"][:, B_].mean(1)          # per mode: mean numerator on A minus on B
        dev[root] = dict(additivity_max_abs_err=M["additivity_max_abs_err"], singular_values=[float(v) for v in M["sv"]],
                         mode_A_minus_B_numerator=[float(v) for v in contrast],
                         residual_A_minus_B_numerator=float(M["resid_num"][A_].mean() - M["resid_num"][B_].mean()),
                         raw_A_minus_B_numerator=float(M["raw_num"][A_].mean() - M["raw_num"][B_].mean()),
                         top_modes_by_abs_contrast=[int(i) for i in np.argsort(-np.abs(contrast))[:5]])
        np.savez(f"{out_dir}/mode_numerators_{root}.npz", mode_num=M["mode_num"], resid_num=M["resid_num"], raw_num=M["raw_num"], sv=M["sv"], genes=genes)
    payload = {"cells": np.array([f"{a}|{b}|{c}|{d}" for a, b, c, d in cells])}
    for r_ in NULLSAVE:
        for kk_, vv_ in NULLSAVE[r_].items(): payload[f"{r_}|{kk_}"] = np.asarray(vv_)
    for (root, ch, kk_), v in S.items():
        # float64, not float32: the persisted arrays must be the exact bytes that
        # score_hash_before_truth was computed over
        payload[f"score|{root}|{ch}|{kk_}|real"] = np.ascontiguousarray(v["real"])
        for r_ in v["pseudo"]: payload[f"score|{root}|{ch}|{kk_}|{r_}"] = np.ascontiguousarray(v["pseudo"][r_])
    np.savez_compressed(f"{out_dir}/null_arrays.npz", **payload)
    df = pd.DataFrame(rows); df.to_csv(f"{out_dir}/specificity_family.tsv", sep="\t", index=False)
    rec = dict(n_pseudo=n_pseudo, K=K, n_cells=len(cells), rosters=res, score_hash_before_truth=score_hash,
               development_mode_localisation=dev, pseudo_ids={r: [str(genes[c]) for c in cand] for r, cand in rosters.items()},
               fixture_hash=C3.SEL.file_sha256_16(FIXTURE), pairs_hash=C3.SEL.file_sha256_16(PAIRS), code=C3.SEL.file_sha256_16(__file__),
               useful_excess_FROZEN=USEFUL_EXCESS, freeze_record="FREEZE_USEFUL_EXCESS_2026_09_15.md", runtime_s=time.time() - t0,
               label="Historical GATA1 regulator-specificity evaluation (development analysis)")
    with open(f"{out_dir}/specificity_family.json", "x") as fh: json.dump(rec, fh, indent=1)
    pd.set_option("display.width", 250)
    print(json.dumps({k: v for k, v in rec.items() if k not in ("rosters", "pseudo_ids", "development_mode_localisation")}, indent=1))
    for r, v in res.items(): print(r, json.dumps(v, indent=1))
    print("Development mode-localisation diagnostic:", json.dumps(dev, indent=1))
    print(df.round(3).to_string(index=False))




# Synthetic checks of family-wise inference.
def fixtures():
    import gb_hist_specificity_family as M
    g = np.random.default_rng(0); B = 200

    # 1. one-cell family: adjusted p == marginal p
    obs = np.array([2.0]); NULL = g.normal(size=(B, 1))
    pm, pf, rank, rp, mn = M_minp(obs, NULL)
    assert abs(pm[0] - pf[0]) < 1e-12, ("one-cell adjusted != marginal", pm[0], pf[0])

    # 2. a family maximum elsewhere must inflate the adjusted p of a rank-1 cell
    obs2 = np.array([5.0, 0.0]); NULL2 = np.column_stack([g.normal(size=B), g.normal(3.0, 1.0, size=B)])
    pm2, pf2, *_ = M_minp(obs2, NULL2)
    assert pm2[0] < 1.5 / (B + 1) + 1e-12, ("focal should be at the floor marginally", pm2[0])
    assert pf2[0] > pm2[0] + 1e-9, ("family maximum elsewhere must inflate the adjusted p", pm2[0], pf2[0])

    # 3. focal-row relabelling preserves inference under an exchangeable null: with all B+1 rows
    # iid, the adjusted p of the true focal is uniform-ish
    rej = 0; R = 400
    for s in range(R):
        gg = np.random.default_rng(1000 + s); A_ = gg.normal(size=(B + 1, 6))
        pm3, pf3, *_ = M_minp(A_[0], A_[1:])
        rej += int(np.nanmin(pf3) <= .05)
    rate = rej / R
    assert rate <= .10, ("exchangeable null rejects too often", rate)

    # 4. non-finite focal or pseudo cells become UNRESOLVED (NaN), never the smallest p
    obs4 = np.array([np.nan, 1.0]); NULL4 = g.normal(size=(B, 2)); NULL4[3, 1] = np.nan
    pm4, pf4, rank4, rp4, _ = M_minp(obs4, NULL4)
    assert np.isnan(pm4[0]) and np.isnan(pf4[0]) and rank4[0] == -1, ("non-finite focal must be UNRESOLVED", pm4, pf4)
    assert np.isnan(rp4[4, 1]), "non-finite pseudo entry must stay NaN"

    # 5. two-sided sidedness is actually two-sided
    obs5 = np.array([-5.0]); NULL5 = g.normal(size=(B, 1))
    assert M_minp(obs5, NULL5, two_sided=True)[0][0] < .05, "two-sided must detect a large NEGATIVE deviation"
    assert M_minp(obs5, NULL5, two_sided=False)[0][0] > .9, "one-sided upper must NOT detect it"

    # 6. declared family sizes
    assert len(CHANNELS) == 3 and len(ISVD_CHANNELS) == 2, (CHANNELS, ISVD_CHANNELS)
    print("FIXTURES PASS: one-cell identity; family-max inflation; exchangeable-null rate %.3f; "
          "non-finite -> UNRESOLVED; sidedness; declared family sizes" % rate)


# Symmetric (B+1)-row Westfall-Young min-p
def M_minp(obs, NULL, two_sided=False):
    f = (lambda a, b: np.abs(a) >= abs(b) - 1e-15) if two_sided else (lambda a, b: a >= b - 1e-15)
    ALL = np.vstack([np.asarray(obs, float)[None, :], np.asarray(NULL, float)])
    B1, n_c = ALL.shape
    if not np.isfinite(ALL).all():
        nan_c = np.full(n_c, np.nan)
        return nan_c, nan_c.copy(), np.full(n_c, -1), np.full((B1, n_c), np.nan), np.full(B1, np.nan)
    rp = np.full((B1, n_c), np.nan)
    for c in range(n_c):
        col = ALL[:, c]
        for r in range(B1):
            oth = np.delete(col, r)
            rp[r, c] = (1 + np.sum(f(oth, col[r]))) / B1
    mn = np.nanmin(rp, axis=1); pm = rp[0]; ok = np.isfinite(mn)
    pf = np.array([((np.sum(mn[ok] <= pm[c] + 1e-15)) / B1) if np.isfinite(pm[c]) else np.nan for c in range(n_c)])
    rank = np.array([int(np.sum(f(ALL[1:, c], ALL[0, c]))) + 1 if np.isfinite(ALL[0, c]) else -1 for c in range(n_c)])
    return pm, pf, rank, rp, mn


# production-path replay FIXTURE: recompute every reported p-value and the pre-truth score hash from the persisted arrays alone
def replay(out_dir):
    import pandas as pd
    z = np.load(f"{out_dir}/null_arrays.npz", allow_pickle=True)
    d = pd.read_csv(f"{out_dir}/specificity_family.tsv", sep="\t")
    j = json.load(open(f"{out_dir}/specificity_family.json"))
    S = {}
    for k in z.files:
        if not k.startswith("score|"): continue
        _, root, ch, kk, who = k.split("|")
        S.setdefault((root, ch, kk), {"pseudo": {}})
        if who == "real": S[(root, ch, kk)]["real"] = z[k]
        else: S[(root, ch, kk)]["pseudo"][who] = z[k]
    h = C3.SEL.raw_hash(b"".join(f"{k}|real|{v['real'].shape}".encode() + np.ascontiguousarray(v["real"]).tobytes()
                                 + b"".join(f"{k}|{r}|{v['pseudo'][r].shape}".encode() + np.ascontiguousarray(v["pseudo"][r]).tobytes()
                                            for r in sorted(v["pseudo"])) for k, v in sorted(S.items())))["hex16"]
    hash_ok = (h == j["score_hash_before_truth"])
    checked = 0; bad = []
    for roster in ("ALL", "TF"):
        for fname in ("CONFIRM_HIST", "SECONDARY_CANON", "DELTA_DYNAMIC", "I_SVD"):
            fk, pk = f"{roster}|focal|{fname}", f"{roster}|pseudo|{fname}"
            if fk in z.files: obs, NULL = z[fk], z[pk]
            elif f"{roster}|cols|{fname}" in z.files:
                cols = z[f"{roster}|cols|{fname}"]; obs = z[f"{roster}|focal"][cols]; NULL = z[f"{roster}|pseudo"][:, cols]
            else: continue
            pm, pf, rank, _, _ = M_minp(obs, NULL, two_sided=(fname == "I_SVD"))
            sub = d[(d.roster == roster) & (d.family == fname)]
            if len(sub) != len(pm): bad.append((roster, fname, "length", len(sub), len(pm))); continue
            for i_, (_, row) in enumerate(sub.iterrows()):
                for nm_, got, want in (("p_marginal", pm[i_], row["p_marginal"]), ("p_familywise", pf[i_], row["p_familywise"])):
                    if not (np.isnan(got) and np.isnan(want)) and abs(got - want) > 1e-9:
                        bad.append((roster, fname, nm_, i_, float(got), float(want)))
                checked += 1
    print(f"REPLAY: score_hash reproduces = {hash_ok}; {checked} cells recomputed; {len(bad)} mismatches")
    for b in bad[:10]: print("   MISMATCH", b)
    assert hash_ok, "pre-truth score hash does not reproduce from the saved arrays"
    assert not bad, f"{len(bad)} reported p-values do not reproduce from the saved arrays"
    print("REPLAY PASS: every reported p-value and the pre-truth score hash reproduce from the persisted arrays")


if __name__ == "__main__":
    if sys.argv[1:2] == ["--fixtures"]: fixtures()
    elif sys.argv[1:2] == ["--replay"]: replay(sys.argv[2])
    else: main(*sys.argv[1:3])
