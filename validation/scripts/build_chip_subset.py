"""Build TF-target ground truth from a ChIP-derived Enrichr library for a subset of TFs.

Rationale: CollecTRI tier-A is literature-curated and may systematically
favor correlation-style evaluation. ChEA_2022 / ENCODE_TF_ChIP-seq_2015
derive targets directly from ChIP-seq peaks. Using these as an
orthogonal GT tests whether V1 win/loss patterns survive.

We download via Enrichr API (no auth needed). Output matches the tsv
format accepted by run_v1_enrichment.py (tf\ttarget).

Entries in the library often have suffixes like 'KLF1_CHEA' or
'GATA2_ENCODE_HUMAN' — we collapse to the bare TF symbol and
aggregate targets across matching entries.
"""
from __future__ import annotations
import argparse
import json
import re
import sys
from pathlib import Path
from urllib.request import urlopen, Request
import pandas as pd


ENRICHR_DOWNLOAD = "https://maayanlab.cloud/Enrichr/geneSetLibrary"


def fetch_library(name: str) -> dict[str, set[str]]:
    """Download a gmt-format library from Enrichr; return {term: {genes}}."""
    url = f"{ENRICHR_DOWNLOAD}?mode=text&libraryName={name}"
    print(f"[chip] fetching {name} from Enrichr...", file=sys.stderr)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as resp:
        text = resp.read().decode("utf-8", errors="replace")
    out: dict[str, set[str]] = {}
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        term = parts[0].strip()
        # drop optional description column (Enrichr gmt sometimes has blank second col)
        genes = [g.split(",")[0].strip() for g in parts[2:] if g.strip()]
        out[term] = {g for g in genes if g}
    print(f"[chip] {name}: {len(out)} terms", file=sys.stderr)
    return out


def tf_symbol(term: str) -> str:
    """'KLF1_CHEA' -> 'KLF1'; 'GATA2 21682474 ChIP-Seq HUMAN' -> 'GATA2'."""
    s = re.split(r"[\s_]", term.strip())[0]
    return s.upper()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tfs", required=True,
                    help="comma-separated TF symbols (case-insensitive)")
    ap.add_argument("--libraries", default="ChEA_2022,ENCODE_TF_ChIP-seq_2015",
                    help="comma-separated Enrichr library names to union")
    ap.add_argument("--min-targets", type=int, default=10,
                    help="drop TFs with fewer than this many targets after union")
    ap.add_argument("--out-tsv", required=True,
                    help="tsv path for tf<TAB>target rows")
    ap.add_argument("--out-report", default=None,
                    help="optional json summary of coverage per TF/library")
    args = ap.parse_args()

    want = {t.strip().upper() for t in args.tfs.split(",") if t.strip()}
    libs = [L.strip() for L in args.libraries.split(",") if L.strip()]

    # {tf: {library: set_of_targets}}
    coverage: dict[str, dict[str, set[str]]] = {t: {} for t in want}
    for lib in libs:
        try:
            mapping = fetch_library(lib)
        except Exception as e:
            print(f"[chip] WARN {lib}: {e}", file=sys.stderr)
            continue
        # group terms by TF symbol
        per_tf: dict[str, set[str]] = {}
        for term, genes in mapping.items():
            tf = tf_symbol(term)
            if tf not in want:
                continue
            per_tf.setdefault(tf, set()).update(genes)
        for tf in want:
            coverage[tf][lib] = per_tf.get(tf, set())

    # union across libraries per TF
    rows = []
    report = {}
    for tf in sorted(want):
        all_targets: set[str] = set()
        for lib in libs:
            all_targets |= coverage[tf].get(lib, set())
        # keep only if sufficient
        if len(all_targets) < args.min_targets:
            report[tf] = {"n_total": len(all_targets),
                          "per_library": {L: len(coverage[tf].get(L, set())) for L in libs},
                          "kept": False}
            continue
        for g in sorted(all_targets):
            if g.upper() == tf:  # drop self-edges
                continue
            rows.append({"tf": tf, "target": g})
        report[tf] = {"n_total": len(all_targets),
                      "per_library": {L: len(coverage[tf].get(L, set())) for L in libs},
                      "kept": True}

    df = pd.DataFrame(rows, columns=["tf", "target"])
    Path(args.out_tsv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_tsv, sep="\t", index=False)

    print(f"\n[chip] per-TF coverage:")
    print(f"{'TF':12s}  {'total':>6s}  " +
          "  ".join(f"{L:>24s}" for L in libs) + "  kept?")
    for tf in sorted(want):
        r = report[tf]
        per = "  ".join(f"{r['per_library'][L]:>24d}" for L in libs)
        print(f"{tf:12s}  {r['n_total']:>6d}  {per}  {str(r['kept'])}")
    n_kept = sum(1 for r in report.values() if r["kept"])
    print(f"\n[chip] kept {n_kept}/{len(want)} TFs; wrote {args.out_tsv} ({len(df)} edges)")

    if args.out_report:
        # make sets json-serializable
        Path(args.out_report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_report, "w") as f:
            json.dump(report, f, indent=2)
        print(f"[chip] wrote {args.out_report}")


if __name__ == "__main__":
    main()
