#!/usr/bin/env python
"""gb_canonical_cardiac.py -- CARDIAC wrapper around the hash-pinned scratch/gb_canonical.py (sha 61374edd, internal design note (not shipped)): registers the CARDIAC raw sources in memory
(gb_canonical.RAW.update) and dispatches to gb_canonical's own main().  gb_canonical.py is NOT modified.  Same CLI:
  python scratch/gb_canonical_cardiac.py build    results/canonical_2026_09_18 --system CARDIAC
  python scratch/gb_canonical_cardiac.py validate results/canonical_2026_09_18/CARDIAC_canonical.h5ad
  python scratch/gb_canonical_cardiac.py pairs    results/canonical_2026_09_18 --system CARDIAC       # roots dc1_min, dc1_max (frozen config)
  python scratch/gb_canonical_cardiac.py reconstruct results/canonical_2026_09_18 --system CARDIAC
  (same CLI with --system CARDIACV for the velocyto build; output CARDIACV_canonical.h5ad)
CARDIAC = spliced := kb 'mature' (project nac convention, as BCL11B/RPE1/K562; ambiguous EXCLUDED)   raw: data/raw/mouse_cardiac/cardiac_raw_nac.h5ad
(PI 2026-09-29: the convention is NOT changed and ambiguous reads are NOT added to rescue the primary TFs.)"""
import sys
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
import gb_canonical as GC

GC.RAW["CARDIAC"] = (f"{GC.R}/data/raw/mouse_cardiac/cardiac_raw_nac.h5ad", "mouse")
# 2026-09-29 (PI s363): CARDIACV = PRIMARY cardiac build, velocyto looms (convention-consistent with the other developmental fixtures); CARDIAC (kb nac) stays as the documented sensitivity build.
GC.RAW["CARDIACV"] = (f"{GC.R}/data/raw/mouse_cardiac_velocyto/cardiacv_raw_velocyto.h5ad", "mouse")   # scratch/cardiacv_build_raw_h5ad.py; spliced = velocyto spliced, ambiguous excluded

if __name__ == "__main__":
    GC.main()
