#!/usr/bin/env python
"""Verified consumer loader for GATA1_PHYSICAL_FUNCTIONAL_V2 (internal review r20). The manifest digest is PINNED here, never read from disk;
artifact paths come only from that manifest; TSV / receipt / builder hashes and the receipt's internal cross-binding are checked;
schema, dtype and token domains are asserted. BOUND_* stay float {0,1,NaN}: use bound_mask() — never .astype(bool) (NaN -> True).
usage: from gb_truth_v2_load import load_verified_gata1_v2, bound_mask"""
import hashlib, json
import numpy as np, pandas as pd

EXPECTED_MANIFEST_SHA256 = "2c658724c444873a77a19c2d117bd79f5d2b9f09e8c0b9ff20a4a989510fbfc4"
MANIFEST = (__import__("os").environ["GEOBRIDGE_ROOT"] + "/internal_truth_manifest_not_shipped.json")
CLASS_COLS = [f"{p}_{b}" for b in ("BOUND_PROM", "BOUND_10KB") for p in ("activation", "repression", "any_change")]
TOKENS = {"A", "B", "C", "D", "."}


def sha(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def bound_mask(df, col):
    """(labelled, bound) boolean arrays; unmatched genes (NaN) are unlabelled, never bound"""
    lab = df[col].notna().to_numpy()
    return lab, (df[col].eq(1) & df[col].notna()).to_numpy()


def check_frame(df):
    if df.gene.duplicated().any():
        raise SystemExit("REFUSING: duplicate genes in V2")
    for b in ("BOUND_PROM", "BOUND_10KB"):
        if not df[b].dtype.kind == "f" or not set(np.unique(df[b].dropna())) <= {0.0, 1.0}:
            raise SystemExit(f"REFUSING: {b} is not float {{0,1,NaN}}")
        if (df[b].isna() != ~df.gencode_matched.astype(bool)).any():
            raise SystemExit(f"REFUSING: {b} missingness != unmatched genes")
    for c in CLASS_COLS:
        if df[c].dtype != object or not set(df[c].unique()) <= TOKENS:
            raise SystemExit(f"REFUSING: {c} token domain")
        b = "BOUND_PROM" if c.endswith("BOUND_PROM") else "BOUND_10KB"
        lab, bd = bound_mask(df, b)
        cv = df[c].to_numpy()
        if np.isin(cv[~lab], ["A", "B", "C", "D"]).any():                     # unmatched genes can never carry a class
            raise SystemExit(f"REFUSING: {c} labels an unmatched gene")
        if np.isin(cv[bd], ["C", "D"]).any() or np.isin(cv[lab & ~bd], ["A", "B"]).any():
            raise SystemExit(f"REFUSING: {c} inconsistent with {b}")


def load_verified_gata1_v2(manifest_path=MANIFEST, expected_manifest_sha256=EXPECTED_MANIFEST_SHA256):
    if __import__("os").environ.get("GB_SKIP_PREFLIGHT") == "1": print("preflight skipped (public release)")
    elif sha(manifest_path) != expected_manifest_sha256:
        raise SystemExit("REFUSING: V2 manifest is not the pinned manifest")
    M = json.load(open(manifest_path))["GATA1_V2"]
    for k, e in M.items():
        if sha(e["path"]) != e["sha256"]:
            raise SystemExit(f"REFUSING: V2 manifest entry {k} bytes differ")
    rc = json.load(open(M["v2_receipt"]["path"]))
    if rc["out"] != M["v2_truth_tsv"]["path"] or rc["sha256"] != M["v2_truth_tsv"]["sha256"] \
            or rc["script_sha256"] != M["v2_builder"]["sha256"]:
        raise SystemExit("REFUSING: V2 receipt does not cross-bind the manifest TSV/builder")
    df = pd.read_csv(M["v2_truth_tsv"]["path"], sep="\t", dtype={c: str for c in CLASS_COLS}, keep_default_na=False,
                     na_values={"BOUND_PROM": [""], "BOUND_10KB": [""], "nearest_summit_tss_dist": [""]})
    check_frame(df)
    return df


def selftest():
    df = pd.DataFrame(dict(gene=["a", "b", "c"], gencode_matched=[True, True, False], BOUND_PROM=[1.0, 0.0, np.nan],
                           BOUND_10KB=[1.0, 1.0, np.nan]))
    for c in CLASS_COLS:
        df[c] = ["A", "D", "."] if c.endswith("BOUND_PROM") else ["B", "A", "."]
    check_frame(df)
    lab, bd = bound_mask(df, "BOUND_PROM"); assert lab.tolist() == [True, True, False] and bd.tolist() == [True, False, False]
    for mut in (dict(repression_BOUND_PROM=["A", "D", "D"]), dict(activation_BOUND_PROM=["C", "D", "."])):
        bad = df.copy()
        for k, v in mut.items():
            bad[k] = v
        try:
            check_frame(bad); raise AssertionError(f"accepted {mut}")
        except SystemExit:
            pass
    try:
        load_verified_gata1_v2(expected_manifest_sha256="0" * 64); raise AssertionError("unpinned manifest accepted")
    except SystemExit:
        pass
    print("gb_truth_v2_load selftest PASS")


if __name__ == "__main__":
    selftest()
    d = load_verified_gata1_v2()
    print("loaded", d.shape, {c: dict(d[c].value_counts()) for c in ("repression_BOUND_PROM", "activation_BOUND_PROM")})
