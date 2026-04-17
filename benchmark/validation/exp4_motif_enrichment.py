"""
EXP 4. Motif enrichment for featured TFs (PILOT).

Approach: use MSigDB's pre-computed "genes with TF motif in ±2kb promoter"
libraries via gseapy. For each featured TF, take top-50 predicted targets
and test overlap against:
  - TRANSFAC_and_JASPAR_PWMs library (motif-based)
  - GTRD library (experimental ChIP-seq based)
via Fisher's exact.

Background = all genes in the dataset (consistent across methods).

Positive result criterion:
  GeoBRIDGE top-50 shows significant (padj<0.05) enrichment for the TF's
  own motif-target set for >= 3/5 featured TFs per dataset,
  AND GeoBRIDGE odds ratio > Velorama odds ratio for >= 2/5 TFs.

Full-pipeline upgrade path (NOT implemented here):
  - pyjaspar PWMs
  - genomepy genome + pyensembl GTF for TSS coordinates
  - MOODS or Bio.motifs for PWM scanning
  That version is ~500 lines of code and needs ~10 GB of genome data.
  Pilot is sufficient to establish whether signal exists.

Usage:
  pip install gseapy
  python -u -m benchmark.validation.exp4_motif_enrichment
"""
import sys
import re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))

from benchmark.validation.config import (
    PREPARED_DIR, VELO_DIR, EXP4_DIR,
    DATASETS, DATASET_ORGANISM, FEATURED_TFS, K_FOCUS, ensure_dirs,
)

TOP_N = 50
LIBRARIES = ["TRANSFAC_and_JASPAR_PWMs"]  # Enrichr gene-set libraries


def _load_library(lib_name):
    """Load an Enrichr library locally-cached.

    Returns {term_name: set(gene_symbols)} for a given Enrichr library.
    """
    import gseapy as gp
    cache_path = EXP4_DIR / f"lib_{lib_name}.json"
    if cache_path.exists():
        import json
        return {k: set(v) for k, v in json.loads(cache_path.read_text()).items()}

    # Download via gseapy's utility
    print(f"  Downloading library: {lib_name}")
    sets_df = gp.get_library(name=lib_name, organism="Human")
    # Returns dict term → list of genes (newer gseapy) or DataFrame
    if isinstance(sets_df, dict):
        lib = {k: set(v) for k, v in sets_df.items()}
    else:
        lib = {}
        for term, grp in sets_df.groupby(sets_df.columns[0]):
            lib[term] = set(grp.iloc[:, 1].tolist())

    import json
    cache_path.write_text(json.dumps({k: sorted(v) for k, v in lib.items()}))
    return lib


def _find_tf_terms(lib, tf_name):
    """Find library terms that mention the TF (case-insensitive)."""
    pat = re.compile(rf"\b{re.escape(tf_name.upper())}\b")
    hits = [t for t in lib.keys() if pat.search(t.upper())]
    return hits


def _get_top_targets(score_matrix, gene_names, tf, name_to_idx, top_n=TOP_N):
    """Top-N predicted targets for TF (excludes self)."""
    if tf not in name_to_idx:
        # Try case variants
        for cand in (tf.upper(), tf.title(), tf.lower()):
            if cand in name_to_idx:
                tf = cand
                break
        else:
            return []
    tf_idx = name_to_idx[tf]
    s = np.abs(score_matrix[tf_idx]).copy()
    s[tf_idx] = -np.inf
    order = np.argsort(-s)
    return [gene_names[i] for i in order[:top_n]]


def _get_top_targets_velorama(df_tsv, tf, top_n=TOP_N):
    # Try TF column directly; if mouse uppercased, try case variants
    for cand in (tf, tf.title(), tf.upper(), tf.lower()):
        if cand in df_tsv.columns:
            col = df_tsv[cand].abs()
            return col.sort_values(ascending=False).head(top_n).index.tolist()
    return []


def _fisher_enrichment(query_set, term_set, universe_set):
    """Fisher's exact test: query vs universe, intersection with term_set."""
    q = set(query_set) & universe_set
    t = set(term_set) & universe_set
    q_and_t = q & t
    q_not_t = q - t
    not_q_and_t = t - q
    not_q_not_t = universe_set - q - t

    table = [[len(q_and_t), len(q_not_t)],
             [len(not_q_and_t), len(not_q_not_t)]]
    try:
        odds, p = fisher_exact(table, alternative="greater")
    except ValueError:
        odds, p = np.nan, np.nan
    return {
        "n_query": len(q),
        "n_term": len(t),
        "n_universe": len(universe_set),
        "overlap": len(q_and_t),
        "expected_overlap": len(q) * len(t) / max(len(universe_set), 1),
        "odds_ratio": odds,
        "p_fisher": p,
    }


def run_dataset(ds, libs):
    import scanpy as sc

    print(f"\n=== {ds} ===")

    h5ad = PREPARED_DIR / f"{ds}.h5ad"
    adata = sc.read_h5ad(str(h5ad))
    gene_names = list(adata.var_names)
    name_to_idx = {g: i for i, g in enumerate(gene_names)}
    organism = DATASET_ORGANISM[ds]

    # Universe: all genes in dataset, uppercased to match Enrichr convention
    universe = set(g.upper() for g in gene_names)
    print(f"  Universe: {len(universe)} genes")

    # Velorama scores
    tsvs = list(VELO_DIR.glob(f"*{ds}*interactions*.tsv"))
    df_velo = None
    if tsvs:
        df_velo = pd.read_csv(tsvs[0], sep="\t", index_col=0)
        df_velo.index = df_velo.index.astype(str)
        df_velo.columns = df_velo.columns.astype(str)

    # GeoBRIDGE scores at K_FOCUS
    print(f"  Computing GeoBRIDGE scores at k={K_FOCUS}...")
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
    S_gb = GeoBridgeModel.scores_from_streaming_stats(
        pre, k=K_FOCUS, project_y=True, normalize_score=True, eps=1e-12)

    rows = []
    for tf in FEATURED_TFS.get(ds, []):
        top_gb = [g.upper() for g in _get_top_targets(S_gb, gene_names, tf, name_to_idx)]
        top_vr = []
        if df_velo is not None:
            top_vr = [g.upper() for g in _get_top_targets_velorama(df_velo, tf)]

        if not top_gb and not top_vr:
            continue

        for lib_name, lib in libs.items():
            terms = _find_tf_terms(lib, tf)
            if not terms:
                print(f"    [skip] no term for {tf} in {lib_name}")
                continue

            # Take union of all terms matching this TF (different motif variants)
            term_union = set()
            for t in terms:
                term_union |= lib[t]

            if top_gb:
                e_gb = _fisher_enrichment(top_gb, term_union, universe)
                rows.append({"dataset": ds, "tf": tf, "method": "GeoBRIDGE",
                             "library": lib_name, "n_terms": len(terms),
                             **e_gb})
            if top_vr:
                e_vr = _fisher_enrichment(top_vr, term_union, universe)
                rows.append({"dataset": ds, "tf": tf, "method": "Velorama",
                             "library": lib_name, "n_terms": len(terms),
                             **e_vr})

            # Also save the actual top-N lists for auditability
            lists_dir = EXP4_DIR / "top_lists" / ds
            lists_dir.mkdir(parents=True, exist_ok=True)
            pd.Series(top_gb).to_csv(lists_dir / f"{tf}_GeoBRIDGE_top{TOP_N}.txt",
                                     index=False, header=False)
            if top_vr:
                pd.Series(top_vr).to_csv(lists_dir / f"{tf}_Velorama_top{TOP_N}.txt",
                                         index=False, header=False)

    return pd.DataFrame(rows)


def run():
    # Load libraries once
    libs = {}
    for lib_name in LIBRARIES:
        libs[lib_name] = _load_library(lib_name)
        print(f"  Loaded {lib_name}: {len(libs[lib_name])} terms")

    all_rows = []
    for ds in DATASETS:
        try:
            df = run_dataset(ds, libs)
            all_rows.append(df)
        except Exception as e:
            print(f"  [ERROR] {ds}: {e}")
            import traceback; traceback.print_exc()

    if not all_rows:
        return
    full = pd.concat(all_rows, ignore_index=True)

    # Multiple-testing correction within (dataset, library)
    from scipy.stats import false_discovery_control
    full["p_adj"] = np.nan
    for (ds, lib_name), grp in full.groupby(["dataset", "library"]):
        ps = grp["p_fisher"].values
        mask = ~np.isnan(ps)
        if mask.sum() > 0:
            adj = np.full_like(ps, np.nan, dtype=float)
            adj[mask] = false_discovery_control(ps[mask])
            full.loc[grp.index, "p_adj"] = adj

    full.to_csv(EXP4_DIR / "motif_enrichment.csv", index=False)

    # Summary per dataset
    lines = ["# EXP 4 Motif enrichment (pilot, MSigDB TFT)\n"]
    for ds in DATASETS:
        lines.append(f"\n## {ds}\n")
        sub = full[full["dataset"] == ds]
        if len(sub) == 0:
            lines.append("  (no data)")
            continue
        lines.append("| TF | method | library | n_terms | overlap | "
                     "odds_ratio | p_fisher | p_adj |")
        lines.append("|---|---|---|---|---|---|---|---|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['tf']} | {r['method']} | {r['library']} | "
                f"{r['n_terms']} | {r['overlap']}/{r['n_query']} | "
                f"{r['odds_ratio']:.2f} | {r['p_fisher']:.3g} | "
                f"{r['p_adj']:.3g} |"
            )

        # GB > Velorama odds ratio check
        pivoted = sub.pivot_table(index=["tf", "library"], columns="method",
                                  values="odds_ratio").reset_index()
        if {"GeoBRIDGE", "Velorama"}.issubset(pivoted.columns):
            pivoted["gb_gt_velo"] = pivoted["GeoBRIDGE"] > pivoted["Velorama"]
            n_total = pivoted["gb_gt_velo"].notna().sum()
            n_win = pivoted["gb_gt_velo"].sum()
            n_sig_gb = (sub[(sub["method"] == "GeoBRIDGE")]["p_adj"] < 0.05).sum()
            lines.append(f"\n**Summary**: GB odds > Velorama in {n_win}/{n_total} "
                         f"TF×library cells. GB padj<0.05 in {n_sig_gb} cells.")

    (EXP4_DIR / "REPORT.md").write_text("\n".join(lines))
    print(f"\n[SAVE] {EXP4_DIR}/motif_enrichment.csv")
    print(f"[SAVE] {EXP4_DIR}/REPORT.md")


if __name__ == "__main__":
    ensure_dirs()
    run()
