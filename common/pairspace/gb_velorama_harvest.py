#!/usr/bin/env python3
# Harvest the completed official-Velorama runs
import argparse, glob, hashlib, json, os, shutil, sys

import numpy as np


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# Exactly velorama/run.py:211 and :244
def official_lam_list(lam_start, lam_end, num_lambdas):
    return [np.round(l, 4) for l in
            np.logspace(lam_start, lam_end, num=num_lambdas).tolist()]


# Exactly velorama/utils.py:load_gc_interactions and run.py:203
def config_name(name, seed, hidden, penalty, lag, dynamics):
    return f"{name}.seed{seed}.h{hidden}.{penalty}.lag{lag}.{dynamics}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--velorama-root", required=True,
                    help="the official checkout containing velorama/velorama/utils.py")
    ap.add_argument("--search-root", required=True,
                    help="directory tree holding the ray trial outputs")
    ap.add_argument("--name", required=True)
    ap.add_argument("--dynamics", required=True, choices=["rna_velocity", "pseudotime"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--penalty", default="H")
    ap.add_argument("--lag", type=int, default=5)
    ap.add_argument("--lam-start", type=float, default=-2)
    ap.add_argument("--lam-end", type=float, default=1)
    ap.add_argument("--num-lambdas", type=int, default=19)
    ap.add_argument("--reg-genes", default=None,
                    help="one name per line; column labels, as run.py uses reg_genes")
    ap.add_argument("--target-genes", default=None, help="one name per line; row labels")
    ap.add_argument("--stage-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--input-h5ad", default=None,
                    help="the velorama input fixture; hashed into the receipt")
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(a.velorama_root, "velorama"))
    import torch
    import pandas as pd
    from velorama.utils import (load_gc_interactions, estimate_interactions,  # noqa
                                estimate_lags)

    cfg = config_name(a.name, a.seed, a.hidden, a.penalty, a.lag, a.dynamics)
    lam_list = official_lam_list(a.lam_start, a.lam_end, a.num_lambdas)

    staged = os.path.join(a.stage_dir, cfg)
    os.makedirs(staged, exist_ok=True)

    found, missing, sources, digests = {}, [], {}, {}
    for lam in lam_list:
        fn = (f"{a.name}.seed{a.seed}.lam{lam}.h{a.hidden}.{a.penalty}"
              f".lag{a.lag}.{a.dynamics}.pt")
        hits = glob.glob(os.path.join(a.search_root, "**", fn), recursive=True)
        hits = [h for h in hits if os.path.basename(h) == fn]
        if not hits:
            missing.append(float(lam))
            continue
        # deterministic choice when a lambda was written by more than one trial
        src = sorted(hits)[0]
        dst = os.path.join(staged, fn)
        if (not os.path.exists(dst)) or _sha(dst) != _sha(src):
            shutil.copy2(src, dst)
        found[float(lam)] = dst
        sources[float(lam)] = src
        digests[float(lam)] = _sha(dst)

    import velorama as _vel
    _vsrc = os.path.dirname(_vel.__file__)
    report = {"LABEL": "OFFICIAL_VELORAMA_HARVEST", "name": a.name,
              "tensor_sha256": {str(k): v for k, v in sorted(digests.items())},
              "tensor_sources": {str(k): v for k, v in sorted(sources.items())},
              "velorama_utils_sha256": _sha(os.path.join(_vsrc, "utils.py")),
              "velorama_models_sha256": _sha(os.path.join(_vsrc, "models.py")),
              "velorama_run_sha256": _sha(os.path.join(_vsrc, "run.py")),
              "input_h5ad_sha256": (_sha(a.input_h5ad) if a.input_h5ad else None),
              "reg_genes_sha256": (_sha(a.reg_genes) if a.reg_genes else None),
              "target_genes_sha256": (_sha(a.target_genes) if a.target_genes else None),
              "dynamics": a.dynamics, "seed": a.seed, "config_name": cfg,
              "lam_grid": [float(x) for x in lam_list],
              "n_lambda_found": len(found), "missing_lambdas": missing,
              "n_duplicate_sources": sum(
                  1 for lam in found
                  if len(glob.glob(os.path.join(a.search_root, "**",
                                                os.path.basename(found[lam])),
                                   recursive=True)) > 1),
              "aggregation": ("Official velorama.utils.load_gc_interactions, estimate_interactions "
                              "and estimate_lags functions."),
              "diagnosis": ("training completed; the official run.py aggregation failed "
                            "only because move_files walked a CWD-relative save_dir "
                            "while tune.run wrote under root_dir/save_dir")}

    os.makedirs(a.out_dir, exist_ok=True)
    if missing:
        report["VERDICT"] = "REFUSED_INCOMPLETE_LAMBDA_GRID"
        report["reason"] = ("aggregation over a partial grid would silently change the "
                            "estimate; rerun the missing lambdas rather than aggregating")
        with open(os.path.join(a.out_dir, f"{a.name}.{a.dynamics}.harvest.json"),
                  "w") as fh:
            json.dump(report, fh, indent=1)
        print(json.dumps({"VERDICT": report["VERDICT"], "missing": missing}, indent=1))
        return

    all_lags = load_gc_interactions(a.name, a.stage_dir, lam_list, hidden_dim=a.hidden,
                                    lag=a.lag, penalty=a.penalty,
                                    dynamics=a.dynamics, seed=a.seed, ignore_lag=False)
    gc_mat = estimate_interactions(all_lags, lag=a.lag)
    lag_mat = estimate_lags(all_lags, lag=a.lag)

    def names(path, n, kind):
        if path:
            v = [l.strip() for l in open(path) if l.strip()]
            if len(v) != n:
                raise SystemExit(f"REFUSING: {kind} list has {len(v)} entries but the "
                                 f"interaction matrix needs {n}")
            return v
        return [f"{kind}{i}" for i in range(n)]

    gc = np.asarray(gc_mat.cpu().data.numpy())
    lg = np.asarray(lag_mat.cpu().data.numpy())
    tg = names(a.target_genes, gc.shape[0], "target")
    rg = names(a.reg_genes, gc.shape[1], "regulator")

    inter_p = os.path.join(a.out_dir, f"{a.name}.{a.dynamics}.velorama.interactions.tsv")
    lags_p = os.path.join(a.out_dir, f"{a.name}.{a.dynamics}.velorama.lags.tsv")
    pd.DataFrame(gc, index=tg, columns=rg).to_csv(inter_p, sep="\t")
    pd.DataFrame(lg, index=tg, columns=rg).to_csv(lags_p, sep="\t")

    report["VERDICT"] = "HARVESTED"
    report["interactions_tsv"] = inter_p
    report["interactions_tsv_sha256"] = _sha(inter_p)
    report["lags_tsv"] = lags_p
    report["matrix_shape"] = list(gc.shape)
    report["gene_labels_supplied"] = bool(a.target_genes and a.reg_genes)
    with open(os.path.join(a.out_dir, f"{a.name}.{a.dynamics}.harvest.json"), "w") as fh:
        json.dump(report, fh, indent=1)
    print(json.dumps({"VERDICT": "HARVESTED", "shape": list(gc.shape),
                      "n_lambda": len(found), "out": inter_p}, indent=1))


if __name__ == "__main__":
    main()
