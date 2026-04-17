"""
EXP 1. Velorama sign-flip sanity check.

Tests:
  (a) AUROC using abs(velorama_score)       — what we reported
  (b) AUROC using signed velorama_score     — original direction
  (c) AUROC using -signed                   — trivially 1 - (b)

If (b) < 0.5 systematically but (c) > 0.5 → Velorama learned signal with
wrong sign and we should flip it for all downstream analysis.

If (a) > 0.5 but (b) ≈ 0.5 → Velorama has magnitude info but no direction.

If (a) ≈ (b) ≈ (c) ≈ 0.5 → Velorama is genuinely random.

Usage:
  python -u -m benchmark.validation.exp1_sign_flip
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from benchmark.validation.config import (
    CSV_DIR, VELO_DIR, PREPARED_DIR, EXP1_DIR,
    DATASETS, DATASET_ORGANISM, ensure_dirs,
)
from benchmark.bm_real_data_comparison import (
    _load_collectri_regulons, _build_ground_truth,
)


def _load_velo_tsv(ds):
    tsvs = list(VELO_DIR.glob(f"*{ds}*interactions*.tsv"))
    if not tsvs:
        return None
    df = pd.read_csv(tsvs[0], sep="\t", index_col=0)
    df.index = df.index.astype(str)
    df.columns = df.columns.astype(str)
    return df


def _eval_one_tf(scores, labels):
    if labels.sum() == 0 or labels.sum() == len(labels):
        return None
    return {
        "auroc": roc_auc_score(labels, scores),
        "auprc": average_precision_score(labels, scores),
        "n_targets": int(labels.sum()),
    }


def run_one_dataset(ds):
    import scanpy as sc

    print(f"\n=== {ds} ===")

    h5ad = PREPARED_DIR / f"{ds}.h5ad"
    if not h5ad.exists():
        print(f"  [SKIP] {h5ad} not found")
        return None

    df_velo = _load_velo_tsv(ds)
    if df_velo is None:
        print(f"  [SKIP] Velorama TSV not found")
        return None

    adata = sc.read_h5ad(str(h5ad))
    gene_names = list(adata.var_names)

    organism = DATASET_ORGANISM[ds]
    uppercase = (organism == "mouse")
    net = _load_collectri_regulons(organism=organism)
    gt_dict = _build_ground_truth(net, gene_names, uppercase_match=uppercase)
    print(f"  GT TFs: {len(gt_dict)}")

    rows = []
    for tf, targets in gt_dict.items():
        # Look up TF column — if uppercase-matched, the column in df_velo
        # uses the ORIGINAL (mouse) case, while tf is the uppercase key.
        # Map: try tf, tf.title(), tf.lower().
        tf_col = None
        for candidate in (tf, tf.title(), tf.lower()):
            if candidate in df_velo.columns:
                tf_col = candidate
                break
        if tf_col is None:
            continue

        scores_signed, labels = [], []
        for gene in gene_names:
            if gene == tf_col or gene not in df_velo.index:
                continue
            scores_signed.append(float(df_velo.loc[gene, tf_col]))
            # _build_ground_truth (uppercase_match=True) stores targets in
            # ORIGINAL case (mouse gene symbols like 'Sox9'), not uppercase.
            # Compare gene directly, no upper() transform.
            labels.append(1 if gene in targets else 0)

        labels = np.array(labels)
        scores_signed = np.array(scores_signed)

        m_abs = _eval_one_tf(np.abs(scores_signed), labels)
        m_signed = _eval_one_tf(scores_signed, labels)
        m_neg = _eval_one_tf(-scores_signed, labels)

        if m_abs is None or m_signed is None:
            continue

        rows.append({
            "dataset": ds,
            "tf": tf,
            "n_targets": m_abs["n_targets"],
            "auroc_abs": m_abs["auroc"],
            "auprc_abs": m_abs["auprc"],
            "auroc_signed": m_signed["auroc"],
            "auprc_signed": m_signed["auprc"],
            "auroc_neg_signed": m_neg["auroc"],
            "auprc_neg_signed": m_neg["auprc"],
            "score_min": float(scores_signed.min()),
            "score_max": float(scores_signed.max()),
            "score_has_negative": bool((scores_signed < 0).any()),
        })

    if not rows:
        return None
    return pd.DataFrame(rows)


def summarize(df):
    if df is None or len(df) == 0:
        return "  (no data)"
    lines = []
    # Baseline AUPRC (mean n_targets / n_candidates-1); n_candidates always
    # equals len(gene_names)-1. Compute from n_targets.
    baseline_auprc = (df["n_targets"] / (2000 - 1)).mean()
    lines.append(f"  Baseline AUPRC (random): ~{baseline_auprc:.4f}")
    for col_auroc, col_auprc, label in [
        ("auroc_abs", "auprc_abs", "abs(score)"),
        ("auroc_signed", "auprc_signed", "signed(score)"),
        ("auroc_neg_signed", "auprc_neg_signed", "-signed(score)"),
    ]:
        mR, sR = df[col_auroc].mean(), df[col_auroc].std()
        mP, sP = df[col_auprc].mean(), df[col_auprc].std()
        lines.append(
            f"  [{label:<18s}] "
            f"AUROC={mR:.4f}±{sR:.3f} (>0.5: {(df[col_auroc] > 0.5).sum()}/{len(df)})  "
            f"AUPRC={mP:.4f}±{sP:.3f} ({mP/max(baseline_auprc, 1e-9):.2f}×baseline)"
        )
    return "\n".join(lines)


def main():
    ensure_dirs()

    all_rows = []
    for ds in DATASETS:
        df = run_one_dataset(ds)
        if df is None:
            continue
        all_rows.append(df)
        df.to_csv(EXP1_DIR / f"{ds}_sign_flip.csv", index=False)

        # Check: any TF negative scores at all?
        has_neg = df["score_has_negative"].any()
        print(f"  Velorama TSV contains negative scores: {has_neg}")

        print(summarize(df))

    if all_rows:
        pd.concat(all_rows).to_csv(EXP1_DIR / "all_sign_flip.csv", index=False)

    # Compact decision summary
    report_lines = ["# EXP 1 Sign-flip check\n"]
    for ds in DATASETS:
        csv_path = EXP1_DIR / f"{ds}_sign_flip.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        baseline_auprc = (df["n_targets"] / 1999).mean()
        report_lines.append(f"\n## {ds}  (n_TFs = {len(df)}, baseline AUPRC ≈ {baseline_auprc:.4f})")
        report_lines.append("\n| score convention | AUROC | AUPRC | AUPRC ÷ baseline | frac(AUROC>0.5) |")
        report_lines.append("|---|---|---|---|---|")
        for col_auroc, col_auprc, label in [
            ("auroc_abs", "auprc_abs", "abs(score)"),
            ("auroc_signed", "auprc_signed", "signed(score)"),
            ("auroc_neg_signed", "auprc_neg_signed", "-signed(score)"),
        ]:
            mR = df[col_auroc].mean()
            mP = df[col_auprc].mean()
            ratio = mP / max(baseline_auprc, 1e-9)
            frac = (df[col_auroc] > 0.5).sum() / len(df)
            report_lines.append(
                f"| {label} | {mR:.4f} | {mP:.4f} | {ratio:.2f}× | "
                f"{frac:.2f} ({(df[col_auroc] > 0.5).sum()}/{len(df)}) |"
            )

        # Decision
        has_neg = df["score_has_negative"].any()
        if not has_neg:
            decision = "Velorama scores are all non-negative → flip is mathematically impossible; only magnitude info is used."
        else:
            diff = df["auroc_signed"].mean() - df["auroc_neg_signed"].mean()
            if abs(diff) < 0.02:
                decision = "Signed vs -signed ~equal → no sign-flip effect."
            elif df["auroc_neg_signed"].mean() > 0.55 and df["auroc_signed"].mean() < 0.45:
                decision = "SIGN FLIP IS REQUIRED — -signed AUROC > 0.55 while signed < 0.45."
            else:
                decision = "No strong sign flip evidence."
        report_lines.append(f"\n**Decision**: {decision}")

    report = "\n".join(report_lines)
    (EXP1_DIR / "REPORT.md").write_text(report)
    print(f"\n[SAVE] {EXP1_DIR}/REPORT.md")
    print(report)


if __name__ == "__main__":
    main()
