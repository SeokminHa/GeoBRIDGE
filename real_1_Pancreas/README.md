## Pancreas real-data (Fig3) script

This folder is a `.py` implementation of `Fig3_AtoC_Pancreas_paper.ipynb`, using the **current**
repo GeoBRIDGE implementation (`geobridge_model`).

### Run

From the repo root:

```bash
python -m real_1_Pancreas.pancreas_fig3
```

### Install dependencies (if needed)

The original notebook installed these. If you hit import errors, run:

```bash
pip install pandas numpy scipy scikit-learn matplotlib seaborn scanpy scvelo networkx gseapy
```

Or explicitly:

```bash
python real_1_Pancreas/pancreas_fig3.py --outdir real_1_Pancreas
```

### Outputs

- `real_1_Pancreas/figures/`
  - `umap_velocity_stream.png`
  - `branch_specificity_Pax4.png`
  - `branch_specificity_Arx.png`
  - `topn_home_vs_away_boxplots.png`
  - `graph_top_edges_Alpha_branch.png`
  - `graph_top_edges_Beta_branch.png`
  - `neurog3_kgrid_enrichr.png` (if enabled)
- `real_1_Pancreas/tables/`
  - `topn_home_vs_away_summary.csv`
  - `branches.json`
  - `neurog3_kgrid_enrichr.csv` (if enabled)
- `real_1_Pancreas/run_manifest.json`

### Notes

- Enrichr / GO analysis requires `gseapy` and network access. Use `--skip-enrichr` to disable.

### Extra diagnostic runs

Additional `--outdir` runs (parameter sweeps, orientation checks, etc.) live under **`outputs/pancreas_runs/`** in the organized repo layout. Point `build_diag_comparison.py --runs` at those directories when comparing summaries.

