"""REAL-DATA canonical parity (PANC): package vs stored production arrays. Needs GB_REPO with the stored real-data artifacts."""
import os, sys, numpy as np, anndata as ad, pytest
from geobridge import pipeline
from geobridge.score import channel_absS, score_fixture, observational_universe
import geobridge.primary as P
REPO = os.environ.get("GB_REPO", "")
CAN = f"{REPO}/results/canonical_2026_09_18"; ST = f"{REPO}/results/cop_alignment/scores/PANC.npz"
pytestmark = pytest.mark.skipif(not os.path.exists(ST), reason="requires GB_REPO with the stored real-data artifacts")


def test_vendor_intact():
    pipeline.verify_vendor()


def _setup():
    z = np.load(ST, allow_pickle=True); A = ad.read_h5ad(f"{CAN}/PANC_canonical.h5ad")
    srcs = [str(s) for s in z["sources"][:3]]; eg = np.asarray(z["genes"]).astype(str)
    fi = {g: i for i, g in enumerate(np.asarray(A.var_names).astype(str))}
    return z, A, srcs, eg, np.array([fi[g] for g in eg]), pipeline.load_pairs(f"{CAN}/PANC_canonical_pairs.npz")


def test_scores_exact_vs_stored():
    z, A, srcs, eg, gi, pairs = _setup()
    raw = channel_absS(A, pairs, srcs, gi)
    for r in ("dc1_min", "dc1_max"):
        for c in ("U", "X"):
            assert np.array_equal(raw[(r, c)], z[f"{r}|{c}"][:3], equal_nan=True), (r, c, np.nanmax(np.abs(raw[(r, c)] - z[f"{r}|{c}"][:3])))


def test_primary_and_universe_vs_stored():
    z, A, srcs, eg, gi, pairs = _setup()
    prim, _, unis = score_fixture(A, srcs, pairs, targets=eg)
    for j, s in enumerate(srcs):
        rows = {(r, c): np.asarray(z[f"{r}|{c}"][j], float) for r in P.ROOTS for c in ("U", "X")}
        u = observational_universe(s, rows, eg); assert np.array_equal(u, unis[j])
        ref = np.full(len(eg), np.nan); ref[u] = P.zcons({h: rows[(h, "U")][u] for h in P.ROOTS})
        assert np.array_equal(prim[j], ref, equal_nan=True)


@pytest.mark.skipif(not os.environ.get("GB_E2E"), reason="requires GB_E2E=1 and GB_REPO with the stored real-data artifacts")
@pytest.mark.xfail(reason="DIAGNOSED 2026-10-09: first divergence = sc.pp.pca (X_pca rel 1.8e-5, deterministic across thread counts, not reproduced by any solver/dtype variant); normalisation, HVG set, Ms, Mu, velocity, X bitwise identical; code+config shas identical. Downstream: DPT pt 1.6e-4, pair overlap 97.2%/99.4%, PRIMARY Spearman vs stored 0.99997 (RFX6, n=3212, same support)", strict=False)
def test_end_to_end_from_raw(tmp_path):
    fx, pr = pipeline.preprocess(f"{REPO}/data/Pancreas/endocrinogenesis_day15.h5ad", str(tmp_path), "mouse", "PANC")
    a, b = np.load(pr, allow_pickle=True), np.load(f"{CAN}/PANC_canonical_pairs.npz", allow_pickle=True)
    for r in ("dc1_min", "dc1_max"):
        assert str(a[f"{r}_pairs_sha256"]) == str(b[f"{r}_pairs_sha256"])
