#!/usr/bin/env python
"""LONG-TRAINING SENSITIVITY 2026-10-28: as the seed-check copy, but early-stopping patience lookback 50 x check_every 10 (official 5 x 10); names _long{S}. (PI: verify Velorama training): identical to gb_velorama_v2_rt0_alltargets.py except --seed S (init/seeds), separate cache/result names (_seed{S}), verbose=True (logs early stopping). Velorama comparator rerun in the paper's configuration (v2).  Truth-free.

Why v2 (protocol-fidelity audit 2026-09-26, wf_e9144806-347; live log §243): v1 (gb_real_uniform.run_velorama)
passed OUR truth-derived source roster as the candidate regulators (input-lock breach; in a joint group-lasso the
competing regulators shape every score), used every fixture gene as a target, and trained with the CLI defaults
(hidden 32, 1,000 iterations): the proximal group lasso stayed unconverged (weights ~.84-.99 correlated with the
init), 55-87% of regulators got all-zero columns (AUROC .5 by ties), and K562's lambda x lag band was empty.
v2 = Velorama paper STAR 5.2.6/5.3.2 with the official code (external/velorama_official, SHA-bound):
  * regulators = frozen species TF catalogue ∩ fixture genes detected in >=1% of cells, fixture var order;
  * targets: arm `native` (paper) = non-catalogue genes detected in >=1% of cells -> TF targets MISSING, evaluated
    paired on the covered subset; arm `alltargets` (declared deviation, coverage for the main table) = all detected
    genes (a TF is then also a target; its self entry is excluded);
  * hidden 16, max_iter 10,000 with the package's own early stopping (lookback 5 x check_every 10), lr .01, H
    penalty, lag 5, reg_target 1, dynamics rna_velocity, stochastic velocity, proba 1, the official 19-point
    lambda grid 1e-2..10, zscore_clip5 inputs, stored neighbour graph (as the authors' own object);
  * official aggregation (utils.estimate_interactions via pairspace/gb_velorama_harvest.py); an empty (1%, 95%)
    band is reported NOT_SCORED_EMPTY_BAND, never widened; per-source all-zero columns are recorded and are
    NOT_SCORED for that source (regulator-level abstention), never AUROC .5.

  python scratch/gb_velorama_v2.py train K562 --gpu 3 --arm native [--lams 0,9,18]
  python scratch/gb_velorama_v2.py harvest K562 --arm native
    -> results/real_uniform/K562/velorama_v2_native.npz (+ receipt) or velorama_v2_native.status.json
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
SEED = int(sys.argv[sys.argv.index("--seed") + 1]) if "--seed" in sys.argv else 1
if "--seed" in sys.argv: i = sys.argv.index("--seed"); del sys.argv[i:i + 2]

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gb_real_uniform as RU

ARMS = ("alltargets_rt0",)  # 2026-10-26 internal review: all detected genes as targets; same regs + rt0 config as native_rt0 (copy of frozen gb_velorama_v2_rt0.py)
LAMS = RU.VELORAMA_LAMS
HIDDEN, MAX_ITER, LAG, DET = 16, 10000, 5, 0.01
LOOKBACK = 50
UPSTREAM_E315C54 = {  # rs239/velorama @ e315c5434387750250932cd594e4874e1690620c (clean checkout), pinned
    "__init__.py": "8573db0e8c4c79438e802b8af6d28f1e55a116d12e2be95259f8063692dc00a4",
    "models.py": "8b66b79f08764c015a469f383edef4bad0d446b29fadb8aaf8d3efa991b70f89",
    "run.py": "798b164eab0ecccf9e63ab6f78533564b226da2972ef0c3fccf6b57ef3702369",
    "train.py": "63f3a70e0e3f2bd0af468762f5f5f400ec06f346ccc1b0d2323b9fb3592995d2",
    "utils.py": "de14136087bd49347c28a4a2ef5eb7cc177db6ed19fb2c0e10bd339bea07d018",
}
CONFIG = {
    "regulators": "frozen species TF catalogue ∩ fixture genes detected in >=1% cells (var order)",
    "targets": {"native": "non-catalogue genes detected in >=1% cells", "alltargets": "all genes detected in >=1% cells"},
    "hidden": [HIDDEN], "max_iter": MAX_ITER, "lr": 0.01, "lam_ridge": 0.0, "penalty": "H", "lag": LAG,
    "lookback": LOOKBACK, "check_every": 10, "reg_target": 0, "dynamics": "rna_velocity", "velo_mode": "stochastic",
    "proba": 1, "n_neighbors": 30, "n_comps": 50, "use_time": 0, "x_norm": "zscore_clip5", "seed": SEED,
    "lams": [round(x, 10) for x in LAMS], "band": [0.01, 0.95],
}



def drop_case_collisions(names):
    """Regulators whose symbol collides case-insensitively in the fixture (GATA1: Setd1a/SETD1A) are dropped as a whole
    group, matching the evaluator's universe rule (decision 2026-09-26, PI delegated)."""
    from collections import Counter
    c = Counter(n.upper() for n in names)
    return [n for n in names if c[n.upper()] == 1]


def plan(system: str, arm: str) -> dict:
    import anndata as ad
    import gb_substrate as GS

    D = RU.canonical_inputs(system)
    RU.assert_frozen_roster_covered(system, D["sources"])
    cat_path = GS.paths(system, canonical=True)[2]
    cat = {l.strip().upper() for l in open(cat_path) if l.strip()}
    A = ad.read_h5ad(D["fixture"])
    genes = np.asarray(A.var_names).astype(str)
    X = RU.dense(A.X, np.float64)
    det = (X > 0).mean(0) >= DET
    is_tf = np.isin(np.char.upper(genes), list(cat))
    regs = np.asarray(drop_case_collisions(list(genes[is_tf & det])))
    targets = genes[~is_tf & det] if arm.startswith("native") else genes[det]
    upstream = sorted((RU.R / "external" / "velorama_official" / "velorama" / "velorama").glob("*.py"))
    if {p.name: RU.sha(p) for p in upstream} != UPSTREAM_E315C54:
        raise SystemExit("REFUSING_VELORAMA_UPSTREAM_NOT_E315C54")
    binding = {
        "method": f"Velorama-v2-{arm}",
        "inputs": RU.input_manifest(D),
        "config": {**CONFIG, "arm": arm, "catalogue": cat_path, "catalogue_sha256": RU.sha(cat_path),
                   "upstream": {p.name: RU.sha(p) for p in upstream},
                   "harvester_sha256": RU.sha(RU.PAIRSPACE / "gb_velorama_harvest.py")},
        "producer_sha256": RU.sha(Path(__file__).resolve()),
        "truth_read": False,
    }
    binding["binding_sha256"] = RU.digest_object(binding)
    name = f"{system}_velorama_v2_{arm}_long{SEED}"
    return dict(D=D, A=A, X=X, genes=genes, regs=regs, targets=targets, binding=binding, name=name,
                cfg=f"{name}.seed{SEED}.h{HIDDEN}.H.lag{LAG}.rna_velocity",
                out=RU.system_dir(system) / f"velorama_v2_{arm}_long{SEED}_cache",
                result=RU.system_dir(system) / f"velorama_v2_{arm}_long{SEED}.npz")


def model_file(P, lam) -> Path:
    return P["out"] / P["cfg"] / f"{P['name']}.seed{SEED}.lam{np.round(lam, 4)}.h{HIDDEN}.H.lag{LAG}.rna_velocity.pt"


def claim_cache(P) -> None:
    key = P["out"] / "CACHE_KEY.json"
    if key.exists():
        if json.load(open(key)) != P["binding"]:
            raise SystemExit(f"REFUSING_STALE_CACHE: {P['out']} was trained under a different binding; move it aside")
    else:
        P["out"].mkdir(parents=True, exist_ok=True)
        RU.atomic_json(key, P["binding"])


def train(args) -> None:
    RU.set_gpu(args.gpu)  # before torch
    import scanpy as sc
    import torch
    vend = str(RU.R / "external" / "velorama_official" / "velorama")
    sys.path.insert(0, vend)                     # force the vendored, hash-pinned e315c54 checkout
    import velorama.train as _vt
    import velorama.utils as _vu
    for m_ in (_vt, _vu):
        if not str(Path(m_.__file__).resolve()).startswith(str((Path(vend) / "velorama").resolve())) or \
                RU.sha(m_.__file__) != UPSTREAM_E315C54[Path(m_.__file__).name]:
            raise SystemExit(f"REFUSING_VELORAMA_NOT_VENDORED: {m_.__file__}")
    train_model, calculate_diffusion_lags, construct_dag = _vt.train_model, _vu.calculate_diffusion_lags, _vu.construct_dag

    if not torch.cuda.is_available():
        raise SystemExit("REFUSING_NO_CUDA: Velorama must run on GPU")
    P = plan(args.system.upper(), args.arm)
    claim_cache(P)
    idx = [int(i) for i in args.lams.split(",")] if args.lams else range(len(LAMS))
    todo = [LAMS[i] for i in idx if not model_file(P, LAMS[i]).exists()]
    print(f"[velorama_v2] {P['name']}: {len(P['regs'])} regulators x {len(P['targets'])} targets; "
          f"{len(todo)} lambdas to train", flush=True)
    if not todo:
        return
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    A, X, gi = P["A"], P["X"], {g: i for i, g in enumerate(P["genes"])}
    Xr = RU.zscore_torch(X[:, [gi[g] for g in P["regs"]]])
    Yt = RU.zscore_torch(X[:, [gi[g] for g in P["targets"]]])
    sc.pp.scale(A)
    dag = torch.FloatTensor(construct_dag(A, dynamics="rna_velocity", proba=1, n_neighbors=30,
                                          velo_mode="stochastic", use_time=0, n_comps=50))
    AX, AY = calculate_diffusion_lags(dag, Xr, LAG), None  # reg_target=0: no target history
    (P["out"] / P["cfg"]).mkdir(parents=True, exist_ok=True)
    for lam in todo:
        t0 = time.time()
        train_model({"name": P["name"], "AX": AX, "AY": AY, "Y": Yt, "seed": SEED, "lr": 0.01, "lam": lam,
                     "lam_ridge": 0.0, "penalty": "H", "lag": LAG, "hidden": [HIDDEN], "max_iter": MAX_ITER,
                     "device": "cuda", "lookback": LOOKBACK, "check_every": 10, "verbose": True,
                     "dynamics": "rna_velocity", "results_dir": str(P["out"]), "dir_name": P["cfg"], "reg_target": 0})
        g = torch.load(model_file(P, lam), map_location="cpu").detach()
        if not torch.isfinite(g).all():
            raise SystemExit(f"REFUSING_VELORAMA_NONFINITE_TENSOR: lam={lam}")
        RU.atomic_json(model_file(P, lam).with_suffix(".train.json"), {"sha256": RU.sha(model_file(P, lam)),
                                                                     "binding_sha256": P["binding"]["binding_sha256"]})
        nnz = [round(float((g[:, :, j] != 0).float().mean()), 4) for j in range(g.shape[-1])]
        print(f"[velorama_v2] {P['name']} lambda={lam:.5g} {time.time()-t0:.0f}s nnz_by_lag={nnz} shape={tuple(g.shape)} peakGB={torch.cuda.max_memory_allocated()/1e9:.1f}", flush=True)


def harvest(args) -> None:
    import torch

    P = plan(args.system.upper(), args.arm)
    if P["result"].exists() and P["result"].with_suffix(".status.json").exists():
        raise SystemExit("REFUSING_TWO_TERMINAL_OUTCOMES")
    if RU.valid_part(P["result"], P["binding"]):
        print(f"[velorama_v2] reuse {P['result']}", flush=True)
        return
    files = [model_file(P, lam) for lam in LAMS]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"REFUSING_INCOMPLETE_VELORAMA_GRID: {len(missing)} missing, e.g. {missing[:2]}")
    for f in files:  # model bytes must equal the training-time receipt
        tj = f.with_suffix(".train.json")
        if not tj.exists() or json.load(open(tj))["sha256"] != RU.sha(f):
            raise SystemExit(f"REFUSING_VELORAMA_MODEL_CHANGED_SINCE_TRAINING: {f}")
    nnz = {}
    for lam, f in zip(LAMS, files):
        g = torch.load(f, map_location="cpu").detach()
        nnz[f"{lam:.5g}"] = [float((g[:, :, j] != 0).float().mean()) for j in range(g.shape[-1])]
    lo, hi = CONFIG["band"]
    n_band = sum(lo < x < hi for v in nnz.values() for x in v)
    status = P["result"].with_suffix(".status.json")
    models = [{"path": str(f), "sha256": RU.sha(f)} for f in files]

    def retire(path: Path) -> None:  # exactly one terminal outcome: move (never delete) the other one aside
        for q in (path, path.with_suffix(".receipt.json")):
            if q.exists():
                q.rename(q.with_name(q.name + time.strftime(".stale_%Y%m%dT%H%M%S")))

    if n_band == 0:
        retire(P["result"])
        RU.atomic_json(status, {"VERDICT": "NOT_SCORED_EMPTY_BAND", "binding": P["binding"], "models": models,
                                "nnz_by_lambda_lag": nnz, "truth_read": False,
                                "note": "official aggregation has no lambda x lag slice with 1% < nnz < 95%"})
        print(f"[velorama_v2] {P['name']}: NOT_SCORED_EMPTY_BAND", flush=True)
        return
    retire(status)
    stage, hv = P["out"] / "stage", P["out"] / "harvest"
    reg_file, tgt_file = P["out"] / "reg_genes.txt", P["out"] / "target_genes.txt"
    reg_file.write_text("\n".join(P["regs"]) + "\n")
    tgt_file.write_text("\n".join(P["targets"]) + "\n")
    subprocess.run([sys.executable, str(RU.PAIRSPACE / "gb_velorama_harvest.py"),
                    "--velorama-root", str(RU.R / "external" / "velorama_official"),
                    "--search-root", str(P["out"]), "--name", P["name"], "--dynamics", "rna_velocity", "--seed", str(SEED),
                    "--hidden", str(HIDDEN), "--reg-genes", str(reg_file), "--target-genes", str(tgt_file),
                    "--stage-dir", str(stage), "--out-dir", str(hv), "--input-h5ad", str(P["D"]["fixture"])],
                   check=True)
    import pandas as pd

    tsv = hv / f"{P['name']}.rna_velocity.velorama.interactions.tsv"
    hj = hv / f"{P['name']}.rna_velocity.harvest.json"
    H = json.load(open(hj))
    if H.get("VERDICT") != "HARVESTED" or H.get("interactions_tsv_sha256") != RU.sha(tsv) or H.get("n_lambda_found") != 19:
        raise SystemExit("REFUSING_UNBOUND_VELORAMA_HARVEST")
    T = pd.read_csv(tsv, sep="\t", index_col=0)
    if list(T.index.astype(str)) != list(P["targets"]) or list(T.columns.astype(str)) != list(P["regs"]):
        raise SystemExit("REFUSING_VELORAMA_EXACT_NAME_OR_ORDER_MISMATCH")
    D = P["D"]
    score = np.full((len(D["sources"]), len(D["axis"])), np.nan, np.float32)
    rin = [s for s in D["sources"] if s in T.columns]
    tin = np.asarray([g in T.index for g in D["axis"]])
    si = {s: i for i, s in enumerate(D["sources"])}
    score[np.ix_([si[s] for s in rin], np.flatnonzero(tin))] = \
        T.loc[np.asarray(D["axis"])[tin], rin].to_numpy(np.float32).T
    for i, j in enumerate(D["source_axis_idx"]):
        score[i, j] = np.nan
    const = [s for s in rin if np.nanmax(score[si[s]]) == np.nanmin(score[si[s]])]
    for s in const:  # regulator-level abstention: NOT_SCORED for that source, never AUROC .5
        score[si[s]] = np.nan
    if not np.isfinite(T.to_numpy(float)).all():
        raise SystemExit("REFUSING_VELORAMA_NONFINITE_INTERACTIONS")
    full = np.full((len(P["regs"]), len(D["axis"])), np.nan, np.float32)  # ALL regulators (source-excess analysis)
    full[:, tin] = T.loc[np.asarray(D["axis"])[tin], list(P["regs"])].to_numpy(np.float32).T
    aiu = {g.upper(): i for i, g in enumerate(D["axis"])}
    for i, r_ in enumerate(P["regs"]):
        if r_.upper() in aiu:
            full[i, aiu[r_.upper()]] = np.nan                          # self
        row = full[i][np.isfinite(full[i])]
        if row.size and row.max() == row.min():
            full[i] = np.nan                                            # constant source: abstain
    RU.write_part(P["result"], P["binding"], {"genes": np.asarray(D["axis"]), "sources": np.asarray(D["sources"]),
                                               "score": score, "regulators": np.asarray(P["regs"]), "score_all": full}, {
        "harvest_receipt": str(hj), "harvest_receipt_sha256": RU.sha(hj),
        "models": models,
        "diagnostics": {"n_regulators": len(P["regs"]), "n_targets": len(P["targets"]),
                        "axis_target_coverage": float(tin.mean()), "n_band_slices": int(n_band),
                        "sources_not_regulators": [s for s in D["sources"] if s not in T.columns],
                        "sources_all_zero_or_constant": const,
                        "nnz_by_lambda_lag": nnz},
        "score_semantics": "official aggregated Granger interaction strength; unsigned; NaN = not a target / not a "
                           "regulator / constant (abstained) source / self",
    })
    print(f"[velorama_v2] {P['name']}: band slices {n_band}; constant sources {len(const)}/{len(rin)}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("system")
    t.add_argument("--gpu", type=int, required=True)
    t.add_argument("--arm", choices=ARMS, required=True)
    t.add_argument("--lams", default="", help="comma-separated indices into the official 19-point grid")
    h = sub.add_parser("harvest")
    h.add_argument("system")
    h.add_argument("--arm", choices=ARMS, required=True)
    a = ap.parse_args()
    train(a) if a.cmd == "train" else harvest(a)
