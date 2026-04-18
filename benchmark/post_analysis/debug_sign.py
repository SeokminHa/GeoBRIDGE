"""
Task 1 — Debug CollecTRI sign extraction.

Prints value_counts of every sign-candidate column, then runs the actual
signed-GT builder on bonemarrow gene list to verify sign counts after
extraction. Suggests the minimal fix.

Usage:
  python -u -m benchmark.post_analysis.debug_sign
"""
import sys
from pathlib import Path
import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO))


def inspect(organism):
    from benchmark.bm_real_data_comparison import _load_collectri_regulons
    print(f"\n========== {organism} ==========")
    try:
        net = _load_collectri_regulons(organism=organism)
    except Exception as e:
        print(f"  [ERROR] load failed: {e}")
        return None
    print(f"  shape: {net.shape}")
    print(f"  columns: {net.columns.tolist()}")
    print(f"  dtypes:")
    for c, t in net.dtypes.items():
        print(f"    {c}: {t}")
    for c in ["sign_decision", "sign", "mor", "weight"]:
        if c in net.columns:
            print(f"\n  --- '{c}' ---")
            print(net[c].value_counts(dropna=False).head(15).to_string())
            non_null = net[c].dropna()
            if len(non_null) and np.issubdtype(non_null.dtype, np.number):
                print(f"    numeric range: [{non_null.min()}, {non_null.max()}]")
                print(f"    >0: {(non_null > 0).sum()}  =0: {(non_null == 0).sum()}  <0: {(non_null < 0).sum()}")
    print(f"\n  head(3):")
    print(net.head(3).to_string())
    return net


def verify_signed_build(organism):
    import scanpy as sc
    from benchmark.bm_real_data_comparison import (
        _load_collectri_regulons, _build_signed_ground_truth,
    )
    from benchmark.utils import REAL_BASE

    ds = "bonemarrow" if organism == "human" else "pancreas"
    uppercase = (organism == "mouse")
    h5ad = REAL_BASE / f"{ds}.h5ad"
    if not h5ad.exists():
        print(f"  [SKIP verify] {h5ad} not found")
        return
    adata = sc.read_h5ad(str(h5ad))
    gene_names = list(adata.var_names)

    net = _load_collectri_regulons(organism=organism)
    gt = _build_signed_ground_truth(net, gene_names, uppercase_match=uppercase,
                                     min_targets=5)
    sign_counts = {+1: 0, -1: 0, 0: 0}
    per_tf_neg_counts = []
    for tf, tgts in gt.items():
        per_tf_neg = 0
        for _, s in tgts.items():
            sign_counts[s] += 1
            if s == -1:
                per_tf_neg += 1
        per_tf_neg_counts.append(per_tf_neg)
    print(f"\n  [verify {ds}] gt TFs={len(gt)}")
    print(f"  edge sign counts: +1={sign_counts[+1]}  -1={sign_counts[-1]}  0={sign_counts[0]}")
    if per_tf_neg_counts:
        arr = np.array(per_tf_neg_counts)
        print(f"  per-TF negative-edge count: median={int(np.median(arr))}, "
              f"mean={arr.mean():.1f}, max={int(arr.max())}")
        n_tfs_with_neg = int((arr > 0).sum())
        print(f"  TFs with ≥1 negative edge: {n_tfs_with_neg}/{len(gt)}")


def main():
    for org in ["human", "mouse"]:
        net = inspect(org)
        if net is not None:
            verify_signed_build(org)

    print("\n" + "=" * 40)
    print("Interpretation cheatsheet:")
    print("  - If 'sign_decision' is bool: _to_sign may return 0 for True/False.")
    print("    Fix: add isinstance(v, (bool, np.bool_)) branch, True → +1, False → -1.")
    print("  - If 'sign_decision' is numeric with {+1, -1, 0}: current code handles it.")
    print("  - If all values are +1 or NaN: dataset has no negative edges; signed_neg")
    print("    will be nan across the board — not a bug, just a GT limitation.")
    print("  - Pass if: bonemarrow verify shows ≥10% TFs with negative edges.")


if __name__ == "__main__":
    main()
