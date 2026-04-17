"""
EXP 5d. Aggregate stability pilot results.

For each (dataset, TF, method):
  - Collect top-K target sets across all available subsamples
  - Compute pairwise Jaccard similarities across subsamples
  - Report mean Jaccard as the stability score
  - Paired Wilcoxon: is GeoBRIDGE mean Jaccard > Velorama mean Jaccard?

Inputs (read from EXP5_DIR):
  subsamples/{ds}_seed{s}.h5ad       — for gene name reference
  geobridge_out/{ds}_seed{s}_k{K}.npz — GB score matrix
  velorama_out/{ds}_seed{s}/...interactions.tsv

Usage:
  python -u -m benchmark.validation.exp5d_aggregate_stability
"""
import sys
from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
from scipy import stats

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from benchmark.validation.config import (
    EXP5_DIR, DATASETS, DATASET_ORGANISM, STABILITY_SEEDS,
    STABILITY_TOP_K, K_FOCUS, MIN_TARGETS_PER_TF, ensure_dirs,
)
from benchmark.bm_real_data_comparison import (
    _load_collectri_regulons, _build_ground_truth,
)


def jaccard(a, b):
    a, b = set(a), set(b)
    if not a and not b:
        return np.nan
    return len(a & b) / len(a | b)


def top_k_from_gb(npz_path, tf_name, top_k=STABILITY_TOP_K):
    d = np.load(npz_path, allow_pickle=True)
    scores = d["scores"]
    genes = [str(g) for g in d["genes"]]
    name_to_idx = {g: i for i, g in enumerate(genes)}
    # try case variants
    for cand in (tf_name, tf_name.title(), tf_name.upper(), tf_name.lower()):
        if cand in name_to_idx:
            tf = cand
            break
    else:
        return None
    idx = name_to_idx[tf]
    s = np.abs(scores[idx]).copy()
    s[idx] = -np.inf
    order = np.argsort(-s)
    return set(genes[i] for i in order[:top_k])


def top_k_from_velo(velo_dir, seed, ds, tf_name, top_k=STABILITY_TOP_K):
    name = f"{ds}_seed{seed}"
    sub_out = velo_dir / name
    tsvs = list(sub_out.glob(f"*{name}*interactions*.tsv"))
    if not tsvs:
        return None
    df = pd.read_csv(tsvs[0], sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    for cand in (tf_name, tf_name.title(), tf_name.upper(), tf_name.lower()):
        if cand in df.columns:
            return set(df[cand].abs().sort_values(ascending=False).head(top_k).index)
    return None


def collect_tf_list(ds):
    import scanpy as sc
    # Use full (non-subsampled) h5ad to determine universe of TFs
    from benchmark.validation.config import PREPARED_DIR
    h5ad = PREPARED_DIR / f"{ds}.h5ad"
    adata = sc.read_h5ad(str(h5ad))
    gene_names = list(adata.var_names)
    organism = DATASET_ORGANISM[ds]
    uppercase = (organism == "mouse")
    net = _load_collectri_regulons(organism=organism)
    gt = _build_ground_truth(net, gene_names, uppercase_match=uppercase,
                             min_targets=MIN_TARGETS_PER_TF)
    return list(gt.keys())


def run():
    gb_dir = EXP5_DIR / "geobridge_out"
    velo_dir = EXP5_DIR / "velorama_out"

    all_rows = []
    for ds in DATASETS:
        print(f"\n=== {ds} ===")
        tf_list = collect_tf_list(ds)
        print(f"  {len(tf_list)} TFs")

        for tf in tf_list:
            # Collect per-seed top-K sets
            gb_sets = {}
            vr_sets = {}
            for seed in STABILITY_SEEDS:
                gb_path = gb_dir / f"{ds}_seed{seed}_k{K_FOCUS}.npz"
                if gb_path.exists():
                    s = top_k_from_gb(gb_path, tf)
                    if s is not None:
                        gb_sets[seed] = s
                v = top_k_from_velo(velo_dir, seed, ds, tf)
                if v is not None:
                    vr_sets[seed] = v

            if len(gb_sets) < 2 and len(vr_sets) < 2:
                continue

            # Pairwise Jaccard
            gb_jaccards = [jaccard(gb_sets[a], gb_sets[b])
                           for a, b in combinations(gb_sets.keys(), 2)]
            vr_jaccards = [jaccard(vr_sets[a], vr_sets[b])
                           for a, b in combinations(vr_sets.keys(), 2)]

            all_rows.append({
                "dataset": ds, "tf": tf,
                "n_gb_seeds": len(gb_sets), "n_vr_seeds": len(vr_sets),
                "n_gb_pairs": len(gb_jaccards), "n_vr_pairs": len(vr_jaccards),
                "gb_mean_jaccard": np.nanmean(gb_jaccards) if gb_jaccards else np.nan,
                "gb_std_jaccard": np.nanstd(gb_jaccards) if gb_jaccards else np.nan,
                "vr_mean_jaccard": np.nanmean(vr_jaccards) if vr_jaccards else np.nan,
                "vr_std_jaccard": np.nanstd(vr_jaccards) if vr_jaccards else np.nan,
            })

    if not all_rows:
        print("[ERROR] No data collected")
        return

    df = pd.DataFrame(all_rows)
    df.to_csv(EXP5_DIR / "stability_per_tf.csv", index=False)

    # Paired Wilcoxon per dataset
    lines = ["# EXP 5 Stability pilot\n",
             f"- top-K = {STABILITY_TOP_K}",
             f"- subsample fraction = 0.80",
             f"- seeds = {STABILITY_SEEDS}\n"]

    for ds in DATASETS:
        sub = df[df["dataset"] == ds].copy()
        sub = sub.dropna(subset=["gb_mean_jaccard", "vr_mean_jaccard"])
        if len(sub) < 3:
            lines.append(f"\n## {ds}: insufficient paired data")
            continue

        lines.append(f"\n## {ds}")
        lines.append(f"- n paired TFs: {len(sub)}")
        lines.append(f"- GeoBRIDGE mean Jaccard: "
                     f"{sub['gb_mean_jaccard'].mean():.4f} "
                     f"± {sub['gb_mean_jaccard'].std():.3f}")
        lines.append(f"- Velorama  mean Jaccard: "
                     f"{sub['vr_mean_jaccard'].mean():.4f} "
                     f"± {sub['vr_mean_jaccard'].std():.3f}")
        lines.append(f"- Δ (GB − V): "
                     f"{(sub['gb_mean_jaccard'] - sub['vr_mean_jaccard']).mean():.4f}")

        try:
            stat, p = stats.wilcoxon(
                sub["gb_mean_jaccard"], sub["vr_mean_jaccard"],
                alternative="greater")
            lines.append(f"- paired Wilcoxon p(GB > V): **{p:.3g}**")
        except ValueError as e:
            lines.append(f"- Wilcoxon failed: {e}")

        # Pre-specified scale-up rule
        delta = (sub["gb_mean_jaccard"] - sub["vr_mean_jaccard"]).mean()
        try:
            _, p = stats.wilcoxon(sub["gb_mean_jaccard"], sub["vr_mean_jaccard"],
                                  alternative="greater")
        except ValueError:
            p = 1.0
        scale_up = (delta >= 0.05) and (p < 0.1)
        lines.append(f"- **Scale-up criterion** (Δ≥0.05, p<0.1): "
                     f"{'PASS — expand to 20 subsamples' if scale_up else 'FAIL — consider dropping stability claim'}")

    (EXP5_DIR / "REPORT.md").write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"\n[SAVE] {EXP5_DIR}/stability_per_tf.csv")
    print(f"[SAVE] {EXP5_DIR}/REPORT.md")


if __name__ == "__main__":
    ensure_dirs()
    run()
