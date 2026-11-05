#!/usr/bin/env python
"""gb_canonical_retina.py -- RETINA wrapper around the hash-pinned scratch/gb_canonical.py (sha 61374edd, internal design note (not shipped)): registers the RETINA raw source in memory and dispatches to gb_canonical's own main().
gb_canonical.py is NOT modified.  Same CLI as gb_canonical_cardiac.py:
  python scratch/gb_canonical_retina.py build    results/canonical_2026_09_18 --system RETINA
  python scratch/gb_canonical_retina.py validate results/canonical_2026_09_18/RETINA_canonical.h5ad
  python scratch/gb_canonical_retina.py pairs    results/canonical_2026_09_18 --system RETINA
  python scratch/gb_canonical_retina.py reconstruct results/canonical_2026_09_18 --system RETINA
RETINA = Clark 2019 GSE118614 postnatal (P0, P2_rep2, P5, P8_rep1, P8_rep2, P14) 10x v2, velocyto build; spliced = velocyto spliced, ambiguous EXCLUDED   raw: data/raw/mouse_retina_velocyto/retina_raw_velocyto.h5ad"""
import sys
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
import gb_canonical as GC

GC.RAW["RETINA"] = (f"{GC.R}/data/raw/mouse_retina_velocyto/retina_raw_velocyto.h5ad", "mouse")

if __name__ == "__main__":
    GC.main()
