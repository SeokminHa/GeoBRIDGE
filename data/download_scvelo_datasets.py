# Download the two scVelo-distributed single-cell datasets used in Figs 2-4 (raw spliced/unspliced integer counts)
import os, scvelo as scv
R = os.environ["GEOBRIDGE_ROOT"]
os.makedirs(f"{R}/data/Pancreas", exist_ok=True); os.makedirs(f"{R}/data/BoneMarrow", exist_ok=True)
a = scv.datasets.pancreas(file_path=f"{R}/data/Pancreas/endocrinogenesis_day15.h5ad"); print("pancreas", a.shape)
b = scv.datasets.bonemarrow(file_path=f"{R}/data/BoneMarrow/human_cd34_bone_marrow.h5ad"); print("bonemarrow", b.shape)
