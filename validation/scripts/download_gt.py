"""Download DoRothEA tier-A (human + mouse) and the Lambert 2018 TF list.

Uses decoupler's built-in collectri loader; falls back to OmnipathR if needed.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd


LAMBERT_URL = (
    "http://humantfs.ccbr.utoronto.ca/download/v_1.01/"
    "TF_names_v_1.01.txt"
)


def download_lambert(out: Path) -> None:
    import requests
    r = requests.get(LAMBERT_URL, timeout=60)
    r.raise_for_status()
    out.write_text(r.text)
    print(f"[lambert] {out}  ({len(r.text.splitlines())} TFs)")


def _fetch_collectri(organism: str) -> pd.DataFrame:
    """Robust to decoupler 1.x / 2.x / missing."""
    try:
        import decoupler as dc
        if hasattr(dc, "op") and hasattr(dc.op, "collectri"):
            return dc.op.collectri(organism=organism)
        if hasattr(dc, "get_collectri"):
            return dc.get_collectri(organism=organism, split_complexes=False)
    except Exception as e:
        print(f"[collectri] decoupler path failed: {e}")
    try:
        from omnipath.interactions import CollecTRI
        df = CollecTRI.get(genesymbols=True, organism=organism)
        df = df.rename(columns={"source_genesymbol": "source",
                                "target_genesymbol": "target"})
        return df
    except Exception as e:
        print(f"[collectri] omnipath path failed: {e}")
    import requests
    from io import StringIO
    url = "https://github.com/saezlab/CollecTRI/raw/main/CollecTRI_regulons.csv"
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    df = pd.read_csv(StringIO(r.text))
    if organism == "mouse":
        if "source" in df.columns:
            df["source"] = df["source"].astype(str).str.capitalize()
        if "target" in df.columns:
            df["target"] = df["target"].astype(str).str.capitalize()
    return df


def download_collectri(out_human: Path, out_mouse: Path) -> None:
    net_h = _fetch_collectri("human")
    net_m = _fetch_collectri("mouse")

    def _to_tsv(net: pd.DataFrame, path: Path) -> None:
        net = net.rename(columns={"source": "tf"})
        if "weight" not in net.columns:
            net["weight"] = 1.0
        if "confidence" not in net.columns:
            net["confidence"] = "A"
        net[["tf", "target", "weight", "confidence"]].to_csv(path, sep="\t", index=False)
        print(f"[collectri] {path}  ({len(net)} edges)")

    _to_tsv(net_h, out_human)
    _to_tsv(net_m, out_mouse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt-dir", default="data/gt")
    args = ap.parse_args()
    gt = Path(args.gt_dir); gt.mkdir(parents=True, exist_ok=True)

    download_lambert(gt / "lambert2018_tfs.txt")
    # Mouse TF list = capitalize first letter from human (rough but workable)
    (gt / "lambert2018_tfs_mouse.txt").write_text(
        "\n".join(
            t.strip().capitalize() for t in (gt / "lambert2018_tfs.txt").read_text().splitlines() if t.strip()
        )
    )
    download_collectri(gt / "dorothea_tierA.tsv", gt / "dorothea_mouse_tierA.tsv")
    print("done.")


if __name__ == "__main__":
    main()
