"""Fig 3 system overview: UMAP + RNA-velocity stream per system, ONE uniform style (plotting only; fixtures never modified). JPG.
usage: python this.py <SYS>  (PANC | GATA1 | BONEMARROW | CARDIACV)
Uniform pipeline for every system: neighbors on fixture X_pca (n=30, seed 0) -> UMAP (seed 0) -> Leiden (res 1.0, seed 0) for colour only ->
scVelo velocity_graph on the fixture's own 'velocity' layer (xkey Ms) -> stream. Same palette, figure size, point size, stream density;
no title, no legend."""
import sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, anndata as ad, scanpy as sc, scvelo as scv
S = sys.argv[1]; R = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/results")
A = ad.read_h5ad(f"{R}/canonical_2026_09_18/{S}_canonical.h5ad")
sc.pp.neighbors(A, use_rep="X_pca", n_neighbors=30, random_state=0)
sc.tl.umap(A, random_state=0)
sc.tl.leiden(A, resolution=1.0, random_state=0, key_added="leiden_plot")
scv.tl.velocity_graph(A, vkey="velocity", xkey="Ms", n_jobs=16)
pal = plt.get_cmap("tab20").colors; cats = A.obs["leiden_plot"].cat.categories
A.uns["leiden_plot_colors"] = [matplotlib.colors.to_hex(pal[i % 20]) for i in range(len(cats))]
fig, ax = plt.subplots(figsize=(6, 6))
scv.pl.velocity_embedding_stream(A, basis="umap", vkey="velocity", color="leiden_plot", ax=ax, show=False, legend_loc="none",
                                 size=10, alpha=.6, density=1.5, linewidth=.9, arrow_size=1.0, title="", frameon=False)
ax.set_aspect("equal", adjustable="datalim"); ax.set_title("")
fig.savefig(f"{R}/figures/fig3_umap_stream_2026_10_23/{S}_umap_stream.jpg", dpi=220, format="jpeg", bbox_inches="tight", pil_kwargs={"quality": 92})
print(S, A.n_obs, len(cats), "ok")
