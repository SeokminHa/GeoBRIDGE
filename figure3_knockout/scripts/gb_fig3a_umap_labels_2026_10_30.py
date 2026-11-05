"""Fig 3a with sparse population labels (PI 2026-10-30; internal review FIG34_POLISH Q1). Plotting only; identical pipeline to
scratch/gb_fig3_umap_stream_2026_10_23.py (neighbors on fixture X_pca n=30 seed 0 -> UMAP seed 0 -> Leiden res 1.0 for colour only -> scVelo stream),
plus labels at the median UMAP position of the fixture's own cell annotation (obs 'clusters'; truth-independent metadata). Heart has no annotation -> unlabeled.
Colours remain Leiden clusters (stated in the legend). PNG.
usage: python this.py <SYS> OUTDIR   (PANC | BONEMARROW | CARDIACV)"""
import sys, os, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.patheffects as pe, anndata as ad, scanpy as sc, scvelo as scv
S, O = sys.argv[1], sys.argv[2].rstrip('/') + '/'; os.makedirs(O, exist_ok=True); R = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/results")
LAB = {'PANC': {'Ductal': 'Ductal', 'Ngn3 low EP': 'Ngn3$^{low}$ EP', 'Ngn3 high EP': 'Ngn3$^{high}$ EP', 'Pre-endocrine': 'Pre-endocrine',
                'Alpha': 'Alpha', 'Beta': 'Beta', 'Delta': 'Delta', 'Epsilon': 'Epsilon'},
       'BONEMARROW': {'HSC_1': 'HSC', 'HSC_2': 'HSC', 'Ery_1': 'Ery', 'Ery_2': 'Ery', 'Mono_1': 'Mono', 'Mono_2': 'Mono', 'CLP': 'CLP', 'DCs': 'DC',
                      'Mega': 'Mega', 'Precursors': 'Precursors'}}
A = ad.read_h5ad(f"{R}/canonical_2026_09_18/{S}_canonical.h5ad")
sc.pp.neighbors(A, use_rep="X_pca", n_neighbors=30, random_state=0); sc.tl.umap(A, random_state=0)
sc.tl.leiden(A, resolution=1.0, random_state=0, key_added="leiden_plot"); scv.tl.velocity_graph(A, vkey="velocity", xkey="Ms", n_jobs=16)
pal = plt.get_cmap("tab20").colors; cats = A.obs["leiden_plot"].cat.categories
A.uns["leiden_plot_colors"] = [matplotlib.colors.to_hex(pal[i % 20]) for i in range(len(cats))]
fig, ax = plt.subplots(figsize=(7.2, 6))
scv.pl.velocity_embedding_stream(A, basis="umap", vkey="velocity", color="leiden_plot", ax=ax, show=False, legend_loc="none",
                                 size=10, alpha=.6, density=1.5, linewidth=.9, arrow_size=1.0, title="", frameon=False)
if S in LAB:
    U = A.obsm['X_umap']; g = A.obs['clusters'].astype(str).map(LAB[S]).to_numpy()
    for name in dict.fromkeys(LAB[S].values()):
        m = g == name
        if m.sum() < 20: continue
        x, y = np.median(U[m], 0)
        ax.text(x, y, name, fontsize=13, fontweight='bold', ha='center', va='center', color='#111', path_effects=[pe.withStroke(linewidth=3, foreground='white')])
ax.set_aspect("auto"); ax.set_title("")   # PI 2026-10-30: slightly wider (x stretched 1.2x)
fig.savefig(O + f"{S}_umap_stream_labeled.png", dpi=220, bbox_inches="tight"); print(S, A.n_obs, len(cats), "ok")
