# Table 1 rows (LaTeX) from the shipped TSVs
import sys, os, pandas as pd
H = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figure3_knockout", "results", "table1")
rows = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else f"{H}/table1_main_rows.tsv", sep="\t")
whole = pd.read_csv(sys.argv[2] if len(sys.argv) > 2 else f"{H}/table1_wholeset_metrics.tsv", sep="\t")
M = ["GeoBRIDGE", "Velorama", "FastTENET", "GENIE3", "GRNBoost2", "PIDC", "RegDiffusion", "DeepSEM"]   # method order as Fig. 2 (|Pearson| omitted)
def cell(v, bold, dag): s = f"{100 * v:.1f}"; s = f"\\textbf{{{s}}}" if bold else s; return s + ("$^{\\dagger}$" if dag else "")
out = []
for _, r in rows.iterrows():
    v = {m: r[f"pct_{m}"] for m in M}; top = max(round(100 * x, 1) for x in v.values())
    ident = ", ".join(x for x in str(r.subtype).split(";") if x != "Endocrine")
    lfc = f"${r.KO_log2FC_human:+.1f}$"
    out.append(" & ".join([r.gene, ident, lfc] + [cell(v[m], round(100 * v[m], 1) == top, bool(r.get(f"tied_{m}", False))) for m in M]) + r" \\")
w = whole[(whole.markerset == "PRIMARY_CellMarker2024") & (whole.metric == "Q1_median_pct")].set_index("method").value
n = int(whole[(whole.markerset == "PRIMARY_CellMarker2024")].n_markers.iloc[0]); top = max(round(100 * w[m], 1) for m in M)
out.append(r"\midrule")
out.append(r"\multicolumn{3}{l}{Median over all %d markers} & " % n + " & ".join(cell(w[m], round(100 * w[m], 1) == top, False) for m in M) + r" \\")
print("\n".join(out))
