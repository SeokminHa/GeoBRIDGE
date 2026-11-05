"""Raw untreated h5ad (integer layers spliced/unspliced) -> canonical fixture -> canonical pairs -> PRIMARY.
Calls the VENDORED, byte-identical gb_canonical.build / build_pairs / gb_gata1_preflight.canonical_pairs; only the module-level
root path variables (R, GEOBRIDGE_ROOT, _PS_PATH) are re-pointed at the vendored tree (verified by vendor/R/SHA256SUMS)."""
import os, sys, json, hashlib, numpy as np, anndata as ad

VR = os.path.join(os.path.dirname(__file__), "vendor", "R")


def _load():
    for p in (f"{VR}/scratch", f"{VR}/pairspace", VR):
        if p not in sys.path: sys.path.insert(0, p)
    import gb_canonical as GC, gb_gata1_preflight as PF
    GC.R = VR; GC.CONFIG = f"{VR}/GEOBRIDGE_PREPROCESS_V1.json"; PF.GEOBRIDGE_ROOT = VR
    PF._PS_PATH = os.path.join(VR, "geobridge_model", "pairspace.py"); PF._PS_CACHE.clear()
    return GC


def verify_vendor():
    for l in open(f"{VR}/SHA256SUMS"):
        h, p = l.split(); assert hashlib.sha256(open(f"{VR}/{p}", "rb").read()).hexdigest() == h, f"vendored file modified: {p}"


def preprocess(raw_h5ad, out_dir, species, system="USER"):
    """species: mouse|human. Returns (fixture_path, pairs_npz_path)."""
    verify_vendor(); GC = _load(); cfg = json.load(open(GC.CONFIG))
    GC.RAW[system] = (os.path.abspath(raw_h5ad), species)
    fx = GC.build(out_dir, system, cfg); GC.build_pairs(out_dir, system, cfg, GC.CONFIG)
    return fx, f"{out_dir}/{system}_canonical_pairs.npz"


def load_pairs(npz):
    z = np.load(npz, allow_pickle=True)
    return {r: dict({"I": z[f"{r}_I"], "J": z[f"{r}_J"]}, **({"pt": z[f"{r}_pt"]} if f"{r}_pt" in z.files else {})) for r in ("dc1_min", "dc1_max")}
