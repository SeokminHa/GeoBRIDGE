import os, sys, hashlib, json, importlib, numpy as np, anndata as ad, pytest
import geobridge.primary as P
from geobridge.score import score_fixture
HERE = os.path.dirname(__file__); FIX = f"{HERE}/../fixture/synthetic_smoke.h5ad"
REPO = os.environ.get("GB_REPO", "")


def h(a):
    return hashlib.sha256(np.nan_to_num(np.round(a, 10), nan=-999.0).tobytes()).hexdigest()


def test_golden():
    got, _ = P._golden()
    assert got == P.GOLDEN_SHA256 == "27893d7fea1eecee7aff343fe331bc9d7dd3e529301dd07c8e8755851c300374"


@pytest.mark.skipif(not os.path.exists(f"{REPO}/scratch/gb_primary.py"), reason="needs $GB_REPO internal repo (scratch/gb_primary.py)")
def test_vendored_equals_pinned():                    # bitwise vs scratch/gb_primary.py
    sys.path.insert(0, f"{REPO}/scratch"); G = importlib.import_module("gb_primary")
    rng = np.random.default_rng(1); R = {k: rng.standard_t(3, size=(200, 31)) for k in P.ROOTS}; R["dc1_min"][::7, 2] = np.nan
    assert np.array_equal(P.zcons(R), G.zcons(R), equal_nan=True)
    assert P.GOLDEN_SHA256 == G.GOLDEN_SHA256


def test_fixture_expected_hash():
    A = ad.read_h5ad(FIX); prim, _, _ = score_fixture(A, ["G0", "G1", "G2"], A.uns["gb_pairs"])
    exp = json.load(open(f"{HERE}/../fixture/expected.json"))
    assert h(prim) == exp["primary_sha256_round10"]
