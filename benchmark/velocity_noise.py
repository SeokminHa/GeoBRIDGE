"""
T4 — Velocity-noise robustness.

For each dataset and each σ ∈ SIGMAS:
  V_noisy = V_orig + σ · std(V_orig) · N(0, I)
Run GeoBRIDGE, Scribe (and optionally Velorama) on (X, V_noisy) and
measure per-TF AUROC. Plot AUROC vs σ per method.

Default methods: GeoBRIDGE, Scribe (CPU, fast).
Pass --include-velorama to add Velorama (slow, GPU-bound).

Usage:
  python -u -m benchmark.velocity_noise --datasets D --seeds 0 1 2
"""
import sys
import time
import argparse
import subprocess
from pathlib import Path
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from benchmark.utils import (
    load_dataset, load_adata, sample_pair_features,
    score_with_projection, evaluate_scores_per_tf,
    SERGIO_DATASETS, SERGIO_BASE, REAL_BASE,
)

OUTDIR = Path("results/velocity_noise")
SIGMAS_DEFAULT = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]


def method_geobridge(X, V_noisy, pt, gene_names, gt_dict, k=5,
                     n_pairs=100_000, seed=42):
    F_X, F_V, _ = sample_pair_features(X, V_noisy, pt, n_pairs=n_pairs,
                                        seed=seed)
    _, _, Vt = np.linalg.svd(F_V, full_matrices=False)
    P = Vt[: min(k, F_V.shape[1])].T
    scores = score_with_projection(F_X, F_V, P=P, normalize=True)
    return evaluate_scores_per_tf(scores, gt_dict, gene_names)


def method_scribe(adata_copy, V_noisy, gene_names, gt_dict):
    try:
        from Scribe.Scribe import causal_net_dynamics_coupling, CLR
    except ImportError:
        return None

    adata_copy.layers["velocity"] = V_noisy
    if "Ms" not in adata_copy.layers:
        adata_copy.layers["Ms"] = (
            adata_copy.X.toarray() if hasattr(adata_copy.X, "toarray")
            else adata_copy.X
        )

    result = causal_net_dynamics_coupling(
        adata_copy, TFs=None, Targets=None,
        t0_key="Ms", t1_key="velocity",
        normalize=True, drop_zero_cells=True, copy=True,
    )
    rdi_df = result.uns["causal_net"]["RDI"]
    try:
        clr_df = CLR(rdi_df)
    except Exception:
        clr_df = rdi_df

    G = len(gene_names)
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    score_m = np.zeros((G, G))
    for tgt_name in clr_df.index:
        if tgt_name not in name_to_idx:
            continue
        j = name_to_idx[tgt_name]
        for reg_name in clr_df.columns:
            if reg_name not in name_to_idx:
                continue
            i = name_to_idx[reg_name]
            score_m[i, j] = float(clr_df.loc[tgt_name, reg_name])
    return evaluate_scores_per_tf(np.abs(score_m), gt_dict, gene_names)


def method_velorama(ds_name, sigma, seed, adata_copy, V_noisy,
                    gene_names, gt_dict, tmp_dir, device="cuda"):
    """
    Invoke Velorama via CLI. Saves noisy h5ad, runs velorama, reads TSV.
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    name = f"{ds_name}_sigma{sigma}_seed{seed}"

    adata_copy.layers["velocity"] = V_noisy
    if "is_reg" not in adata_copy.var:
        adata_copy.var["is_reg"] = True
    if "is_target" not in adata_copy.var:
        adata_copy.var["is_target"] = True

    h5ad_path = tmp_dir / f"{name}.h5ad"
    adata_copy.write_h5ad(str(h5ad_path))

    sd = tmp_dir / name
    sd.mkdir(parents=True, exist_ok=True)
    cmd = [
        "velorama",
        "-n", name, "-ds", name,
        "-dyn", "rna_velocity", "-dev", device,
        "-l", "5", "-hd", "32", "-mi", "1000", "-cp", "4",
        "-rd", str(tmp_dir),
        "-sd", str(sd),
    ]
    print(f"    [Velorama] {' '.join(cmd[:6])}...")
    try:
        subprocess.run(cmd, check=False, capture_output=True, timeout=3600)
    except Exception as e:
        print(f"    [Velorama] ERROR: {e}")
        return None

    tsv_candidates = list(sd.glob(f"*{name}*interactions*.tsv"))
    if not tsv_candidates:
        return None
    df_tsv = pd.read_csv(tsv_candidates[0], sep="\t", index_col=0)
    df_tsv.index = df_tsv.index.astype(str)
    df_tsv.columns = df_tsv.columns.astype(str)

    G = len(gene_names)
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    score_m = np.zeros((G, G))
    for tgt_name in df_tsv.index:
        if tgt_name not in name_to_idx:
            continue
        j = name_to_idx[tgt_name]
        for reg_name in df_tsv.columns:
            if reg_name not in name_to_idx:
                continue
            i = name_to_idx[reg_name]
            score_m[i, j] = abs(float(df_tsv.loc[tgt_name, reg_name]))
    return evaluate_scores_per_tf(score_m, gt_dict, gene_names)


def run_dataset(ds, sigmas, seeds, methods, include_velorama, velorama_dev):
    print(f"\n=== {ds} ===")
    X, V_orig, pt, gene_names, gt_dict = load_dataset(ds)
    V_std = float(V_orig.std())
    print(f"  V_std = {V_std:.4f}")

    adata_orig = load_adata(ds)

    rows = []
    for sigma in sigmas:
        for seed in seeds:
            rng = np.random.default_rng(int(seed * 10000 + sigma * 1000))
            noise = rng.standard_normal(V_orig.shape) * sigma * V_std
            V_noisy = V_orig + noise

            for method in methods:
                t0 = time.perf_counter()
                try:
                    if method == "GeoBRIDGE":
                        df = method_geobridge(X, V_noisy, pt, gene_names,
                                              gt_dict, seed=42 + seed)
                    elif method == "Scribe":
                        df = method_scribe(adata_orig.copy(), V_noisy,
                                           gene_names, gt_dict)
                    elif method == "Velorama":
                        if not include_velorama:
                            continue
                        df = method_velorama(
                            ds, sigma, seed, adata_orig.copy(), V_noisy,
                            gene_names, gt_dict,
                            tmp_dir=OUTDIR / "velorama_tmp",
                            device=velorama_dev,
                        )
                    else:
                        continue
                    if df is None or len(df) == 0:
                        continue
                    elapsed = time.perf_counter() - t0
                    rows.append({
                        "dataset": ds, "method": method,
                        "sigma": sigma, "seed": seed,
                        "n_tfs": len(df),
                        "auroc_mean": float(df["auroc"].mean()),
                        "auroc_std": float(df["auroc"].std()),
                        "auprc_mean": float(df["auprc"].mean()),
                        "auprc_std": float(df["auprc"].std()),
                        "elapsed_sec": elapsed,
                    })
                    print(f"  {method:<10s} σ={sigma:5.2f} seed={seed}: "
                          f"AUROC={df['auroc'].mean():.4f}  "
                          f"AUPRC={df['auprc'].mean():.4f}  "
                          f"({elapsed:.1f}s)")
                except Exception as e:
                    print(f"  [ERROR] {method} σ={sigma} seed={seed}: {e}")
                    import traceback; traceback.print_exc()
    return pd.DataFrame(rows)


def summarize(df, outdir):
    lines = ["# T4 — Velocity-noise robustness\n"]
    for ds in sorted(df["dataset"].unique()):
        sub = df[df["dataset"] == ds]
        lines.append(f"\n## {ds}\n")
        lines.append("| method | " + " | ".join(
            f"σ={s}" for s in sorted(sub["sigma"].unique())) + " |")
        lines.append("|" + "---|" * (sub["sigma"].nunique() + 1))

        for m in sorted(sub["method"].unique()):
            cells = [m]
            for sigma in sorted(sub["sigma"].unique()):
                s = sub[(sub["method"] == m) & (sub["sigma"] == sigma)]
                if len(s) == 0:
                    cells.append("—")
                else:
                    cells.append(
                        f"{s['auroc_mean'].mean():.3f}±{s['auroc_mean'].std():.3f}")
            lines.append("| " + " | ".join(cells) + " |")
    (outdir / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["D"])
    ap.add_argument("--sigmas", type=float, nargs="+", default=SIGMAS_DEFAULT)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--methods", nargs="+",
                    default=["GeoBRIDGE", "Scribe"])
    ap.add_argument("--include-velorama", action="store_true",
                    help="Also run Velorama (slow)")
    ap.add_argument("--velorama-dev", default="cuda")
    args = ap.parse_args()

    if args.include_velorama and "Velorama" not in args.methods:
        args.methods = list(args.methods) + ["Velorama"]

    OUTDIR.mkdir(parents=True, exist_ok=True)
    frames = []
    for ds in args.datasets:
        try:
            f = run_dataset(ds, args.sigmas, args.seeds, args.methods,
                            args.include_velorama, args.velorama_dev)
            frames.append(f)
            f.to_csv(OUTDIR / f"{ds}_noise_sweep.csv", index=False)
        except Exception as e:
            print(f"[ERROR] {ds}: {e}")
            import traceback; traceback.print_exc()

    if frames:
        full = pd.concat(frames, ignore_index=True)
        full.to_csv(OUTDIR / "noise_sweep.csv", index=False)
        summarize(full, OUTDIR)


if __name__ == "__main__":
    main()
