#!/usr/bin/env python
# Register cardiac inputs and run gb_canonical.
import sys
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
import gb_canonical as GC

GC.RAW["CARDIAC"] = (f"{GC.R}/data/raw/mouse_cardiac/cardiac_raw_nac.h5ad", "mouse")
GC.RAW["CARDIACV"] = (f"{GC.R}/data/raw/mouse_cardiac_velocyto/cardiacv_raw_velocyto.h5ad", "mouse")   # scratch/cardiacv_build_raw_h5ad.py; spliced = velocyto spliced, ambiguous excluded

if __name__ == "__main__":
    GC.main()
