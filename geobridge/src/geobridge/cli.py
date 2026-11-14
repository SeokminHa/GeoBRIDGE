import argparse, os, numpy as np, pandas as pd, anndata as ad
from . import pipeline
from .score import score_fixture


def main():
    p = argparse.ArgumentParser(prog="geobridge"); sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("score", help="GeoBRIDGE scores from an h5ad of unperturbed cells: raw integer spliced/unspliced counts (preprocessed here), or an already preprocessed dataset with --canonical --pairs")
    s.add_argument("h5ad"); s.add_argument("--regulators", required=True, help="comma-separated regulator gene symbols; each must be a gene kept in the (preprocessed) data")
    s.add_argument("--species", choices=["mouse", "human"], help="required for raw input")
    s.add_argument("--canonical", action="store_true", help="input is already a canonical fixture; needs --pairs")
    s.add_argument("--pairs", help="canonical_pairs.npz (with --canonical)")
    s.add_argument("--workdir", default="gb_work"); s.add_argument("--out", required=True, help="TSV regulator x target unsigned score Z (G_rg in the paper; NaN = unsupported / outside universe)")
    s.add_argument("--profiles", help="optional .npz with the raw |S| values at each depth k=0..30, per pseudotime root and target channel (U = unspliced, X = spliced)")
    s.add_argument("--stratum-key", help="obs column (e.g. embryo, sample, cluster) whose groups define the strata for the near-isochronic pairs used by d_iso (both cells of a pair in the same group); 'NONE' = one stratum. Enables d_iso and T outputs (needs pairs pt)")
    s.add_argument("--iso-window", type=float, default=0.02, help="iso pair window |dpt| <= w (paper: 0.02)")
    s.add_argument("--seed", type=int, default=42, help="first iso-pair sampling seed; seeds s,s+1,s+2 are averaged (paper: 42, 43, 44). Z itself is deterministic")
    a = p.parse_args(); regs = a.regulators.split(",")
    if a.canonical:
        fx, pr = a.h5ad, a.pairs
        if not pr:
            p.error("--canonical needs --pairs PAIRS.npz (keys {root}_I, {root}_J, {root}_pt for root in dc1_min, dc1_max)")
        if not os.path.exists(pr):
            p.error(f"--pairs file not found: {pr}")
        keys = set(np.load(pr).files)
        need = [f"{r}_{k}" for r in ("dc1_min", "dc1_max") for k in ("I", "J") + (("pt",) if a.stratum_key else ())]
        miss = [k for k in need if k not in keys]
        if miss:
            p.error(f"--pairs file {pr} lacks required keys: {', '.join(miss)} (has: {', '.join(sorted(keys))})")
    else:
        assert a.species, "--species required for raw input"
        fx, pr = pipeline.preprocess(a.h5ad, a.workdir, a.species)
    A = ad.read_h5ad(fx)
    have = set(np.char.upper(np.asarray(A.var_names).astype(str)))
    absent = [r for r in regs if r.upper() not in have]
    if absent:
        p.error(f"regulators not in the data ({fx}): {', '.join(absent)}")
    prim, raw, _ = score_fixture(A, regs, pipeline.load_pairs(pr))
    if a.profiles:
        np.savez_compressed(a.profiles, regulators=regs, genes=np.asarray(A.var_names), **{f"{r}|{c}": v for (r, c), v in raw.items()})
    pd.DataFrame(prim, index=regs, columns=list(A.var_names)).to_csv(a.out, sep="\t", float_format="%.12g")
    print(f"wrote {a.out} {prim.shape}")
    if a.stratum_key:
        from .iso import iso_polarity, signed_composite
        pairs = pipeline.load_pairs(pr); assert "pt" in pairs["dc1_min"], "pairs have no pt; d_iso unavailable"
        S0, d, info = iso_polarity(A, regs, pairs, a.stratum_key, a.iso_window, a.seed); T, _ = signed_composite(prim, d)
        stem = os.path.splitext(a.out)[0]
        for nm, M in (("d_iso", d), ("T", T)):
            pd.DataFrame(M, index=regs, columns=list(A.var_names)).to_csv(f"{stem}.{nm}.tsv", sep="\t", float_format="%.12g")
        print(f"wrote {stem}.d_iso.tsv {stem}.T.tsv", info)


if __name__ == "__main__":
    main()
