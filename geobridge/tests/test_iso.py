"""d_iso / T: synthetic end-to-end smoke (fixture) + REAL-DATA parity (PANC RFX6) vs stored production pilot/T tables. Real part needs the server repo (GB_REPO)."""
import os, sys, json, hashlib, numpy as np, pandas as pd, anndata as ad, pytest
from scipy.stats import spearmanr
from geobridge import pipeline, iso
from geobridge.score import score_fixture
HERE = os.path.dirname(__file__); FIX = f"{HERE}/../fixture/synthetic_smoke.h5ad"
REPO = os.environ.get("GB_REPO", "")
CAN = f"{REPO}/results/canonical_2026_09_18"; FEAT = f"{REPO}/results/signed/iso_pilot_2026_10_15/scores/PANC_RFX6/features_PANC_RFX6.tsv"


def h(a): return hashlib.sha256(np.nan_to_num(np.round(a, 10), nan=-999.0).tobytes()).hexdigest()


def test_iso_vendor_intact():
    iso.verify_vendor()


def test_fixture_end_to_end_and_deterministic(tmp_path):
    A = ad.read_h5ad(FIX); regs = ["G0", "G1", "G2"]; P = A.uns["gb_pairs"]
    prim, _, _ = score_fixture(A, regs, P)
    S0, d, info = iso.iso_polarity(A, regs, P, "embryo"); T, F = iso.signed_composite(prim, d)
    S0b, db, _ = iso.iso_polarity(A, regs, P, "embryo")
    assert np.array_equal(S0, S0b, equal_nan=True) and np.array_equal(d, db, equal_nan=True)            # deterministic given --seed
    assert set(np.unique(d[np.isfinite(d)])) <= {-1.0, 0.0, 1.0} and np.isfinite(d).sum() > 100
    for j, r in enumerate(regs): assert np.isnan(d[j, list(A.var_names).index(r)]) and np.isnan(T[j, list(A.var_names).index(r)])   # own gene unsupported
    ok = np.isfinite(T); assert ok.sum() > 100 and (np.abs(T[ok]) <= 1).all() and (np.sign(T[ok]) == d[ok]).all()
    assert np.array_equal(np.isfinite(T), np.isfinite(prim) & np.isfinite(d))
    exp = json.load(open(f"{HERE}/../fixture/expected.json"))
    assert h(d) == exp["d_iso_sha256_round10"] and h(T) == exp["T_sha256_round10"]
    # cap (40 partners/cell) only binds with a wide window on this small fixture; then the seed is live
    w = [iso.iso_polarity(A, regs, P, "embryo", window=0.3, seed=s)[0] for s in (42, 42, 7)]
    assert np.array_equal(w[0], w[1], equal_nan=True) and not np.array_equal(w[0], w[2], equal_nan=True)


def test_cli_fixture(tmp_path, monkeypatch):
    from geobridge import cli
    A = ad.read_h5ad(FIX); P = A.uns["gb_pairs"]
    np.savez(tmp_path / "p.npz", **{f"{r}_{k}": P[r][k] for r in P for k in ("I", "J", "pt")})
    out = tmp_path / "z.tsv"
    monkeypatch.setattr(sys, "argv", ["geobridge", "score", FIX, "--canonical", "--pairs", str(tmp_path / "p.npz"), "--regulators", "G0,G1,G2", "--out", str(out), "--stratum-key", "embryo"])
    cli.main()
    T = pd.read_csv(tmp_path / "z.T.tsv", sep="\t", index_col=0); d = pd.read_csv(tmp_path / "z.d_iso.tsv", sep="\t", index_col=0)
    prim, _, _ = score_fixture(A, ["G0", "G1", "G2"], P); _, dd, _ = iso.iso_polarity(A, ["G0", "G1", "G2"], P, "embryo")
    assert np.allclose(d.to_numpy(), dd, equal_nan=True, atol=1e-9) and T.shape == prim.shape


@pytest.mark.skipif(not os.path.exists(FEAT), reason="server repo not available")
def test_panc_rfx6_d_iso_and_T_vs_stored():
    A = ad.read_h5ad(f"{CAN}/PANC_canonical.h5ad"); P = pipeline.load_pairs(f"{CAN}/PANC_canonical_pairs.npz")
    prim, _, _ = score_fixture(A, ["RFX6"], P); S0, d, _ = iso.iso_polarity(A, ["RFX6"], P, "clusters"); T, F = iso.signed_composite(prim, d)
    f = pd.read_csv(FEAT, sep="\t", usecols=["gene", "F", "iso_C"]).assign(gene=lambda x: x.gene.str.upper()).set_index("gene")
    g = pd.Series(np.char.upper(np.asarray(A.var_names).astype(str))); ic = f.iso_C.reindex(g).to_numpy(float); Fp = f.F.reindex(g).to_numpy(float)
    m = np.isfinite(ic) & np.isfinite(S0[0]); assert m.sum() > 1000
    assert np.abs(ic[m] - S0[0][m]).max() < 1e-9                                           # d_iso input S0: identical to stored production aggregate
    assert (np.sign(ic[m]) != np.sign(S0[0][m])).sum() == 0 and (np.isfinite(ic) & ~np.isfinite(S0[0])).sum() == 0   # stored table covers only the evaluation universe (1345 of 3879 genes); package covers a superset
    # T: stored production F ranks over the evaluation mask (MR & ~dup), the package ranks over the full scored universe -> same ordering, not equal values
    Tp = np.sign(ic) * Fp; mm = np.isfinite(Tp) & np.isfinite(T[0]); assert mm.sum() > 1000
    rho = spearmanr(Tp[mm], T[0][mm])[0]; print("T spearman", rho, "n", mm.sum(), "sign mismatches", (np.sign(Tp[mm]) != np.sign(T[0][mm])).sum())
    assert rho >= 0.9999 and (np.sign(Tp[mm]) == np.sign(T[0][mm])).all()
