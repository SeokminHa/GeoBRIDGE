"""
EXP 3. Proper mouse-human ortholog mapping for pancreas GT.

Replaces the crude `uppercase_match=True` hack with a real ortholog mapping
from human CollecTRI symbols → mouse symbols via mygene.info.

Design:
  1. Load human CollecTRI.
  2. Collect all unique human symbols (sources + targets).
  3. Query mygene.info for mouse orthologs → save mapping JSON.
  4. Build mouse-symbol CollecTRI network.
  5. Filter to genes present in the pancreas h5ad.
  6. Re-evaluate pancreas Velorama and GeoBRIDGE on the new GT.
  7. Compare: new per-TF metrics vs old (uppercase-match) version.

Usage:
  pip install mygene
  python -u -m benchmark.validation.exp3_pancreas_ortholog
"""
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from benchmark.validation.config import (
    CSV_DIR, VELO_DIR, PREPARED_DIR, EXP3_DIR,
    MIN_TARGETS_PER_TF, K_VALUES, ensure_dirs,
)
from benchmark.bm_real_data_comparison import (
    _load_collectri_regulons, _evaluate_tf,
)


def build_ortholog_map(human_symbols, cache_path):
    """
    Query mygene.info for human→mouse ortholog mapping.
    Returns {human_symbol: mouse_symbol}. Unmapped → not in dict.
    """
    if cache_path.exists():
        print(f"  [load cache] {cache_path}")
        return json.loads(cache_path.read_text())

    import mygene
    mg = mygene.MyGeneInfo()

    # Step 1: human symbol → human entrezgene
    print(f"  Querying {len(human_symbols)} human symbols → Entrez IDs...")
    human_syms = sorted(set(human_symbols))
    res1 = mg.querymany(human_syms, scopes="symbol", fields="entrezgene,symbol,homologene",
                        species="human", returnall=False)
    # Map from human symbol → homologene cluster (list of [taxid, entrezgene])
    sym_to_mouse = {}
    for r in res1:
        if "notfound" in r or "homologene" not in r:
            continue
        sym = r.get("query")
        hg = r["homologene"]
        for taxid, egid in hg.get("genes", []):
            if taxid == 10090:  # mouse
                sym_to_mouse[sym] = int(egid)
                break

    mouse_eids = sorted(set(sym_to_mouse.values()))
    print(f"  Found mouse ortholog Entrez IDs for {len(sym_to_mouse)}/"
          f"{len(human_syms)} human genes")

    # Step 2: mouse Entrez → mouse symbol
    print(f"  Querying {len(mouse_eids)} mouse Entrez IDs → symbols...")
    res2 = mg.getgenes(mouse_eids, fields="symbol")
    eid_to_msym = {}
    for r in res2:
        eid = int(r.get("_id", -1))
        sym = r.get("symbol")
        if sym:
            eid_to_msym[eid] = sym

    mapping = {hs: eid_to_msym[eid] for hs, eid in sym_to_mouse.items()
               if eid in eid_to_msym}
    cache_path.write_text(json.dumps(mapping, indent=2))
    print(f"  [save] {cache_path}  ({len(mapping)} symbols mapped)")
    return mapping


def build_mouse_gt(human_net, human2mouse, gene_set_mouse, min_targets=5):
    """
    Convert human CollecTRI → mouse CollecTRI using ortholog map, filter to
    genes present in dataset, return {mouse_tf: set(mouse_targets)}.
    """
    # Vectorized translation
    net = human_net.copy()
    net["source_mouse"] = net["source"].map(human2mouse)
    net["target_mouse"] = net["target"].map(human2mouse)
    before = len(net)
    net = net.dropna(subset=["source_mouse", "target_mouse"])
    print(f"  CollecTRI edges: {before} → {len(net)} after ortholog translation")

    # Filter to genes in dataset
    net = net[net["source_mouse"].isin(gene_set_mouse) &
              net["target_mouse"].isin(gene_set_mouse)]
    print(f"  Edges after filter to pancreas HVGs: {len(net)}")

    tf_counts = net.groupby("source_mouse")["target_mouse"].nunique()
    valid_tfs = tf_counts[tf_counts >= min_targets].index.tolist()

    gt_dict = {}
    for tf in valid_tfs:
        tgts = set(net[net["source_mouse"] == tf]["target_mouse"].unique())
        tgts.discard(tf)
        if len(tgts) >= min_targets:
            gt_dict[tf] = tgts
    return gt_dict, net


def evaluate_gt(score_matrix, gene_names, gt_dict, method_label, k=None):
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    rows = []
    for tf, targets in gt_dict.items():
        tf_idx = name_to_idx.get(tf)
        if tf_idx is None:
            continue
        scores_tf, labels_tf = [], []
        for j, gj in enumerate(gene_names):
            if j == tf_idx:
                continue
            scores_tf.append(abs(score_matrix[tf_idx, j]))
            labels_tf.append(1 if gj in targets else 0)
        scores_tf = np.array(scores_tf)
        labels_tf = np.array(labels_tf)
        if labels_tf.sum() == 0 or labels_tf.sum() == len(labels_tf):
            continue
        m = _evaluate_tf(scores_tf, labels_tf)
        row = {"method": method_label, "tf": tf, **m}
        if k is not None:
            row["k"] = k
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_velorama(df_tsv, gene_names, gt_dict):
    rows = []
    for tf, targets in gt_dict.items():
        if tf not in df_tsv.columns:
            continue
        scores_tf, labels_tf = [], []
        for gj in gene_names:
            if gj == tf or gj not in df_tsv.index:
                continue
            scores_tf.append(abs(df_tsv.loc[gj, tf]))
            labels_tf.append(1 if gj in targets else 0)
        labels_tf = np.array(labels_tf)
        scores_tf = np.array(scores_tf)
        if labels_tf.sum() == 0 or labels_tf.sum() == len(labels_tf):
            continue
        m = _evaluate_tf(scores_tf, labels_tf)
        rows.append({"method": "Velorama", "tf": tf, "k": -1, **m})
    return pd.DataFrame(rows)


def run():
    import scanpy as sc

    ds = "pancreas"
    print(f"\n=== {ds}: proper ortholog mapping ===")

    h5ad = PREPARED_DIR / f"{ds}.h5ad"
    adata = sc.read_h5ad(str(h5ad))
    gene_names = list(adata.var_names)
    gene_set = set(gene_names)

    # Human CollecTRI
    net_h = _load_collectri_regulons(organism="human")
    human_syms = list(set(net_h["source"]).union(set(net_h["target"])))
    print(f"  Unique human symbols in CollecTRI: {len(human_syms)}")

    # Ortholog map
    cache_path = EXP3_DIR / "human_to_mouse_map.json"
    human2mouse = build_ortholog_map(human_syms, cache_path)

    # Build mouse GT
    gt_mouse, net_mouse = build_mouse_gt(net_h, human2mouse, gene_set,
                                         min_targets=MIN_TARGETS_PER_TF)
    print(f"  Mouse GT TFs (>= {MIN_TARGETS_PER_TF} targets): {len(gt_mouse)}")
    net_mouse.to_csv(EXP3_DIR / "collectri_mouse.csv", index=False)
    pd.Series({tf: sorted(ts) for tf, ts in gt_mouse.items()}).to_json(
        EXP3_DIR / "gt_mouse_dict.json", indent=2)

    # Evaluate Velorama
    tsvs = list(VELO_DIR.glob(f"*{ds}*interactions*.tsv"))
    if not tsvs:
        print(f"  [SKIP] Velorama TSV not found")
        return
    df_velo = pd.read_csv(tsvs[0], sep="\t", index_col=0)
    df_velo.index = df_velo.index.astype(str)
    df_velo.columns = df_velo.columns.astype(str)

    vr_df = evaluate_velorama(df_velo, gene_names, gt_mouse)
    print(f"  Velorama (new GT): {len(vr_df)} TFs evaluated")
    print(f"    AUROC={vr_df['auroc'].mean():.4f} ± {vr_df['auroc'].std():.3f}")
    print(f"    AUPRC={vr_df['auprc'].mean():.4f} ± {vr_df['auprc'].std():.3f}")

    # Evaluate GeoBRIDGE at all K values
    print(f"  Computing GeoBRIDGE scores...")
    from geobridge_model import GeoBridgeModel
    X = np.asarray(adata.layers["Ms"] if "Ms" in adata.layers
                   else (adata.X.toarray() if hasattr(adata.X, "toarray")
                         else adata.X), dtype=float)
    V = np.nan_to_num(np.asarray(adata.layers["velocity"], dtype=float), nan=0.0)
    pt = np.nan_to_num(np.asarray(adata.obs["pseudotime"], dtype=float),
                       nan=float(np.nanmedian(adata.obs["pseudotime"])))
    order = np.argsort(pt)
    pre = GeoBridgeModel.precompute_pairspace_stats_streaming(
        X[order], V[order], pt[order], pair_seed=42)

    gb_rows = []
    for k in K_VALUES:
        if k >= len(pre["S"]):
            continue
        S_k = GeoBridgeModel.scores_from_streaming_stats(
            pre, k=k, project_y=True, normalize_score=True, eps=1e-12)
        gb_df = evaluate_gt(S_k, gene_names, gt_mouse,
                            method_label="GeoBRIDGE", k=k)
        gb_rows.append(gb_df)
        print(f"    k={k:3d}: AUROC={gb_df['auroc'].mean():.4f} ± {gb_df['auroc'].std():.3f}  "
              f"AUPRC={gb_df['auprc'].mean():.4f}")

    # Save
    combined = pd.concat([vr_df] + gb_rows, ignore_index=True)
    combined["dataset"] = ds
    combined.to_csv(EXP3_DIR / "pancreas_results_new_gt.csv", index=False)

    # Comparison to old (uppercase-match) GT
    old_csv = CSV_DIR / "real_comparison_full.csv"
    comparison_rows = []
    if old_csv.exists():
        old = pd.read_csv(old_csv)
        old_sub = old[old["dataset"] == "pancreas"]
        for method in ["Velorama", "GeoBRIDGE"]:
            for k in [-1] + K_VALUES:
                if method == "Velorama" and k != -1:
                    continue
                if method == "GeoBRIDGE" and k == -1:
                    continue
                mask_old = (old_sub["method"] == method) & (old_sub["k"] == k)
                mask_new = (combined["method"] == method) & (combined["k"] == k)
                old_vals, new_vals = old_sub[mask_old], combined[mask_new]
                if len(old_vals) == 0 or len(new_vals) == 0:
                    continue
                comparison_rows.append({
                    "method": method, "k": k,
                    "n_tfs_old": len(old_vals), "n_tfs_new": len(new_vals),
                    "auroc_old": old_vals["auroc"].mean(),
                    "auroc_new": new_vals["auroc"].mean(),
                    "auprc_old": old_vals["auprc"].mean(),
                    "auprc_new": new_vals["auprc"].mean(),
                })
    cmp_df = pd.DataFrame(comparison_rows)
    cmp_df.to_csv(EXP3_DIR / "old_vs_new_gt_comparison.csv", index=False)

    # Report
    lines = [
        "# EXP 3 Pancreas ortholog fix\n",
        f"- Human CollecTRI symbols queried: {len(human_syms)}",
        f"- Mouse orthologs resolved via mygene: {len(human2mouse)}",
        f"- Mouse CollecTRI edges: {len(net_mouse)}",
        f"- GT TFs (>={MIN_TARGETS_PER_TF} targets in pancreas HVGs): {len(gt_mouse)}",
        "",
        "## Velorama",
        f"- mean AUROC: {vr_df['auroc'].mean():.4f}",
        f"- mean AUPRC: {vr_df['auprc'].mean():.4f}",
        f"- mean EPR:   {vr_df['epr'].mean():.3f}",
        "",
        "## GeoBRIDGE by k",
    ]
    for k in K_VALUES:
        sub = combined[(combined["method"] == "GeoBRIDGE") & (combined["k"] == k)]
        if len(sub) == 0:
            continue
        lines.append(f"- k={k:3d}: AUROC={sub['auroc'].mean():.4f}  "
                     f"AUPRC={sub['auprc'].mean():.4f}  "
                     f"EPR={sub['epr'].mean():.2f}  (n={len(sub)})")
    if len(cmp_df):
        lines.append("\n## Old (uppercase-match) vs New (ortholog)")
        lines.append(cmp_df.to_markdown(index=False))
    (EXP3_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"\n[SAVE] {EXP3_DIR}/REPORT.md")


if __name__ == "__main__":
    ensure_dirs()
    run()
