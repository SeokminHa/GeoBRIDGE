"""Download the two scVelo-distributed single-cell datasets used in Figs 2-4 (raw spliced/unspliced integer counts).
  pancreas  : scvelo.datasets.pancreas()   -> Bastidas-Ponce et al. 2019 mouse pancreatic endocrinogenesis E15.5, GEO GSE132188 (3,696 cells)
  bonemarrow: scvelo.datasets.bonemarrow() -> Setty et al. 2019 human CD34+ bone marrow (5,780 cells x 14,319 genes; velocyto loom)
Target file names are the ones the scripts expect (gb_canonical.RAW registry).
usage: GEOBRIDGE_ROOT=... python download_scvelo_datasets.py      (needs internet; scvelo==0.3.4)"""
import os, scvelo as scv
R = os.environ["GEOBRIDGE_ROOT"]
os.makedirs(f"{R}/data/Pancreas", exist_ok=True); os.makedirs(f"{R}/data/BoneMarrow", exist_ok=True)
a = scv.datasets.pancreas(file_path=f"{R}/data/Pancreas/endocrinogenesis_day15.h5ad"); print("pancreas", a.shape)
b = scv.datasets.bonemarrow(file_path=f"{R}/data/BoneMarrow/human_cd34_bone_marrow.h5ad"); print("bonemarrow", b.shape)
