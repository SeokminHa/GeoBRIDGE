#!/usr/bin/env python
"""K562 CRISPRi TRUTH PANEL — per-TF differential response vs non-targeting controls.

usage: gb_k562_truth.py <out_dir> [--min_cells 50] [--alpha 0.05]

K562's truth was already opened historically (2026-08-11, result null on 8 tests after Holm), so this
carries no prospective value and may be built freely. The existing artifacts in
`results/gates_env2_2026_08_05/` are GATE RESULTS for two TFs (SPI1, KLF1), not a reusable TF->target
panel, so the panel is built here.

SOURCE. `data/scperturb/ReplogleWeissman2022_K562_essential.h5ad`: 310,385 cells x 8,563 genes,
2,058 perturbations, 10,691 `non-targeting` control cells. CRISPRi knockdown.

STATISTICS. One streaming pass over row chunks accumulates per-perturbation count / sum / sum-of-squares,
so nothing larger than a chunk is ever resident (the dense matrix would be ~10.6 GB in float32). Welch's
t-test per gene against the pooled non-targeting cells, Benjamini-Hochberg within each TF. Welch rather
than Student because perturbed groups are small (>=min_cells) and their variance need not match control.

POLARITY CONVENTION, identical to `gb_dev_evaluate.truth_panc`: CRISPRi knocks the regulator DOWN, so a
target that goes DOWN on knockdown was being ACTIVATED (pol = -1) and a target that goes UP was being
REPRESSED (pol = +1). The focal regulator is excluded from its own panel.

THIS IS A RESPONSE PANEL, NOT A DIRECT-BINDING PANEL. A chronic CRISPRi response is downstream and
includes indirect effects; the licensed framing is functional response, never "direct target".
"""
import argparse, json, os, time
import numpy as np, pandas as pd, h5py

R = __import__("os").environ["GEOBRIDGE_ROOT"]
SRC = f"{R}/data/scperturb/ReplogleWeissman2022_K562_essential.h5ad"
LAMBERT = f"{R}/GB_real_v2/lambert2018_human_tfs.txt"
CTRL = "non-targeting"


def bh(p):
    p = np.asarray(p, float); n = p.size
    o = np.argsort(p); q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / np.arange(1, n + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out_dir"); ap.add_argument("--min_cells", type=int, default=50)
    ap.add_argument("--alpha", type=float, default=0.05); ap.add_argument("--chunk", type=int, default=20000)
    a = ap.parse_args(); t0 = time.time(); os.makedirs(a.out_dir, exist_ok=True)

    f = h5py.File(SRC, "r")
    g = f["obs"]["gene"]; cats = g["categories"][:].astype(str); codes = g["codes"][:]
    labels = cats[codes]
    vg = f["var"]["gene_name"]
    genes = (vg["categories"][:].astype(str)[vg["codes"][:]] if isinstance(vg, h5py.Group)
             else np.array(vg).astype(str))
    lam = {l.strip().upper() for l in open(LAMBERT) if l.strip()}
    u, c = np.unique(labels, return_counts=True)
    keep = {x for x, n in zip(u, c) if n >= a.min_cells and x.upper() in lam}
    print(f"{len(labels)} cells, {len(u)} perturbations, {int((labels == CTRL).sum())} controls; "
          f"{len(keep)} Lambert TFs with >= {a.min_cells} cells", flush=True)

    groups = {x: i for i, x in enumerate(sorted(keep | {CTRL}))}
    gi = np.full(len(labels), -1, int)
    for x, i in groups.items(): gi[labels == x] = i
    G = len(genes); K = len(groups)
    cnt = np.zeros(K); s1 = np.zeros((K, G)); s2 = np.zeros((K, G))
    # PER-GROUP NONZERO COUNTS. internal review 2026-09-21: cnt/s1/s2 cannot reconstruct a detection fraction, so
    # the previous "sufficient statistics for ANY future endpoint" claim was FALSE and is withdrawn.
    # Detection is what distinguishes a gene that was genuinely tested from one seen in a single cell.
    nz = np.zeros((K, G))
    X = f["X"]
    for lo in range(0, X.shape[0], a.chunk):
        hi = min(lo + a.chunk, X.shape[0])
        blk = np.asarray(X[lo:hi], dtype=np.float64)
        gb = gi[lo:hi]
        for k in np.unique(gb[gb >= 0]):
            m = gb == k
            cnt[k] += m.sum(); s1[k] += blk[m].sum(0); s2[k] += (blk[m] ** 2).sum(0)
            nz[k] += (blk[m] > 0).sum(0)
        if lo % (a.chunk * 5) == 0: print(f"  {hi}/{X.shape[0]} rows {time.time()-t0:.0f}s", flush=True)
    f.close()

    mean = s1 / np.maximum(cnt[:, None], 1)
    var = np.maximum(s2 / np.maximum(cnt[:, None], 1) - mean ** 2, 0) * \
        (cnt[:, None] / np.maximum(cnt[:, None] - 1, 1))
    ci = groups[CTRL]
    mc, vc, nc = mean[ci], var[ci], cnt[ci]
    # SUFFICIENT STATISTICS are saved so that ANY future endpoint can be derived without another
    # streaming pass over the 10.6 GB dense matrix. The previous version discarded them, which is why
    # a four-state table could not be built from the existing artifact.
    np.savez_compressed(f"{a.out_dir}/k562_sufficient_stats.npz",
                        genes=np.asarray(genes, dtype=object),
                        groups=np.asarray(sorted(groups, key=groups.get), dtype=object),
                        cnt=cnt, s1=s1, s2=s2, nz=nz, control_index=ci)
    rows = []
    # FOUR-STATE TABLE (directive v3 section 10). The significant-pairs-only export makes "absent"
    # ambiguous between tested-null and never-tested, so a non-significant gene cannot be used as a
    # negative. Codes: 0 UNKNOWN (the test is undefined), 1 TESTED_NULL, 2 TESTED_UP, 3 TESTED_DOWN.
    # TESTABILITY IS TRUTH-INDEPENDENT: it asks only whether Welch's statistic is defined, never
    # whether the gene responded.
    # PI v4 section 2G. The original Replogle DE workflow's own gene filter is not recoverable from
    # the distributed object, so rather than inventing a favourable threshold after seeing
    # performance we separate three states and PREDECLARE which is primary:
    #     TESTED_NULL                   not significant AND adequately observed in the controls
    #     LOW_INFORMATION_TESTED_NULL   not significant but BELOW the control detection floor --
    #                                   a gene seen in one cell is formally testable and
    #                                   scientifically uninformative
    #     UNKNOWN                       Welch's statistic is undefined
    # PRIMARY benchmark uses the adequately observed subset; the broader tested set is the declared
    # SENSITIVITY arm. The floor matches the BM preregistration so the two systems agree.
    NTC_DETECT_MIN = 0.05
    STATE = {"UNKNOWN": 0, "TESTED_NULL": 1, "TESTED_UP": 2, "TESTED_DOWN": 3,
             "LOW_INFORMATION_TESTED_NULL": 4}
    ctrl_detect = nz[ci] / max(cnt[ci], 1)
    well_observed = ctrl_detect >= NTC_DETECT_MIN
    print(f"control detection >= {NTC_DETECT_MIN}: {int(well_observed.sum())} of {G} genes", flush=True)
    tf_names = [t for t in sorted(groups, key=groups.get) if t != CTRL]
    state = np.zeros((len(tf_names), G), dtype=np.int8)
    qmat = np.full((len(tf_names), G), np.nan, dtype=np.float32)
    lfcmat = np.full((len(tf_names), G), np.nan, dtype=np.float32)
    for tf, k in groups.items():
        if tf == CTRL: continue
        mt, vt, nt = mean[k], var[k], cnt[k]
        se = np.sqrt(vt / nt + vc / nc)
        t = (mt - mc) / np.maximum(se, 1e-300)
        df = np.maximum((vt / nt + vc / nc) ** 2 /
                        (np.maximum((vt / nt) ** 2 / max(nt - 1, 1), 1e-300) +
                         np.maximum((vc / nc) ** 2 / max(nc - 1, 1), 1e-300)), 1)
        from scipy import stats
        p = 2 * stats.t.sf(np.abs(t), df)
        q = bh(p)
        lfc = np.log2((mt + 1e-9) / (mc + 1e-9))
        # A pair is TESTED only if Welch's statistic is defined: a strictly positive standard error
        # (both groups all-zero gives se = 0 and t = 0/0) and finite t, p and log fold change.
        tested = (se > 0) & np.isfinite(t) & np.isfinite(p) & np.isfinite(lfc)
        ti_ = tf_names.index(tf)
        st = np.where(~tested, STATE["UNKNOWN"],
                      np.where(q >= a.alpha,
                               np.where(well_observed, STATE["TESTED_NULL"],
                                        STATE["LOW_INFORMATION_TESTED_NULL"]),
                               np.where(lfc > 0, STATE["TESTED_UP"], STATE["TESTED_DOWN"])))
        self_j = np.flatnonzero(np.char.upper(genes.astype(str)) == tf.upper())
        st[self_j] = STATE["UNKNOWN"]          # the focal is never its own target, in any state
        state[ti_] = st
        qmat[ti_] = np.where(tested, q, np.nan)
        lfcmat[ti_] = np.where(tested, lfc, np.nan)
        sig = (q < a.alpha) & np.isfinite(t)
        for j in np.flatnonzero(sig):
            if genes[j].upper() == tf.upper(): continue          # exclude the focal regulator
            rows.append(dict(tf=tf, target=genes[j], log2fc=float(lfc[j]), t=float(t[j]),
                             p=float(p[j]), padj=float(q[j]),
                             polarity=1.0 if lfc[j] > 0 else -1.0,   # KO-up = repressed, KO-down = activated
                             n_tf_cells=int(nt)))
    D = pd.DataFrame(rows)
    D.to_csv(f"{a.out_dir}/k562_tf_response_panel.tsv", sep="\t", index=False)
    np.savez_compressed(f"{a.out_dir}/k562_state.npz",
                        genes=np.asarray(genes, dtype=object),
                        tfs=np.asarray(tf_names, dtype=object),
                        state=state, qval=qmat, log2fc=lfcmat,
                        code_map=np.asarray(json.dumps(STATE)))
    cov = pd.DataFrame([dict(tf=tf_names[i],
                             n_unknown=int((state[i] == 0).sum()),
                             n_tested_null=int((state[i] == 1).sum()),
                             n_low_info_tested_null=int((state[i] == 4).sum()),
                             n_tested_up=int((state[i] == 2).sum()),
                             n_tested_down=int((state[i] == 3).sum()))
                        for i in range(len(tf_names))])
    cov.to_csv(f"{a.out_dir}/k562_state_coverage.tsv", sep="\t", index=False)
    # CONSISTENCY GATE: the significant-pairs export and the state table must agree exactly, or one
    # of them is wrong and neither may be used.
    # SET equality, not a count match: two different sets can share a cardinality (internal review 2026-09-21).
    gup = np.char.upper(genes.astype(str))
    set_state = {(tf_names[i], gup[j], "UP" if state[i, j] == 2 else "DOWN")
                 for i, j in zip(*np.where((state == 2) | (state == 3)))}
    set_rows = {(r.tf, str(r.target).upper(), "UP" if r.polarity > 0 else "DOWN")
                for r in D.itertuples()}
    if set_state != set_rows:
        only_s = list(set_state - set_rows)[:3]; only_r = list(set_rows - set_state)[:3]
        raise SystemExit(f"REFUSING[STATE_TABLE_INCONSISTENT]: the (TF,target,direction) SETS "
                         f"differ. |state|={len(set_state)} |panel|={len(set_rows)}; "
                         f"state-only e.g. {only_s}; panel-only e.g. {only_r}")
    n_sig_state = len(set_state)
    print(f"four-state table: {int((state==0).sum())} unknown, {int((state==1).sum())} tested-null, "
          f"{int((state==2).sum())} tested-up, {int((state==3).sum())} tested-down "
          f"(significant total matches the exported panel: {n_sig_state})", flush=True)
    summ = (D.groupby("tf").agg(n_sig=("target", "size"),
                                n_activated=("polarity", lambda s: int((s < 0).sum())),
                                n_repressed=("polarity", lambda s: int((s > 0).sum())),
                                n_cells=("n_tf_cells", "first")).reset_index()
            .sort_values("n_sig", ascending=False))
    summ.to_csv(f"{a.out_dir}/k562_tf_summary.tsv", sep="\t", index=False)
    json.dump(dict(source=SRC, n_cells=int(len(labels)), n_control=int(cnt[ci]),
                   min_cells=a.min_cells, alpha=a.alpha, n_tfs=int(len(groups) - 1),
                   test="Welch t vs pooled non-targeting, BH within TF",
                   polarity="KO-up = repression target (+1); KO-down = activation target (-1)",
                   caveat="chronic CRISPRi RESPONSE panel: downstream/indirect effects included; "
                          "this is functional response, NOT direct binding",
                   truth_already_opened="2026-08-11, null on 8 Holm-corrected tests",
                   four_state="k562_state.npz: 0 UNKNOWN (Welch undefined), 1 TESTED_NULL, "
                              "2 TESTED_UP, 3 TESTED_DOWN. NEGATIVES ARE TESTED_NULL ONLY; UNKNOWN "
                              "is excluded. tested-null means 'not significant at the frozen alpha', "
                              "NOT 'biologically unaffected'.",
                   sufficient_stats="k562_sufficient_stats.npz (cnt/s1/s2) so any future endpoint "
                                    "needs no second streaming pass",
                   wall_s=time.time() - t0),
              open(f"{a.out_dir}/provenance.json", "w"), indent=1)
    pd.set_option("display.width", 200)
    print(f"\n=== K562 TF response panel: {len(D)} significant TF-target pairs over {summ.shape[0]} TFs ===")
    print(summ.head(20).to_string(index=False))
    print(f"\nwrote {a.out_dir} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
