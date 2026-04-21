"""Velorama wrapper via subprocess CLI.

Pattern adapted from benchmark/bm_velorama_comparison.py:
  velorama -n {name} -ds {name} -dyn {rna_velocity|pseudotime} -dev cuda
           -l {lags} -hd {hidden} -rd {root_dir} -sd {save_dir}

Output: {save_dir}/{name}.{dynamics}.velorama.interactions.tsv
        rows = targets, cols = regulators, values = interaction strengths.
"""
from __future__ import annotations
import subprocess
import sys
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pandas as pd


def run(
    adata,
    tfs: list[str],
    dynamics: str = "rna_velocity",
    n_lags: int = 5,
    hidden: int = 32,
    device: str = "cuda",
    dataset_name: str = "bm",
    timeout_sec: int = 21600,  # 6h default; BM full 3000 HVGs needs ~3-5h
    **_,
) -> pd.DataFrame:
    if shutil.which("velorama") is None:
        raise RuntimeError(
            "velorama CLI not found in PATH. Install via `pip install velorama`."
        )

    with tempfile.TemporaryDirectory(prefix="velorama_") as td:
        root_dir = Path(td) / "data"
        save_dir = Path(td) / "results"
        root_dir.mkdir(parents=True, exist_ok=True)
        save_dir.mkdir(parents=True, exist_ok=True)

        # Velorama's execute_cmdline reads var['is_reg'] and var['is_target'].
        # Score every gene as a candidate target; mark TFs on the regulator side.
        adata_v = adata.copy()
        tf_set = set(tfs)
        adata_v.var["is_reg"] = adata_v.var_names.isin(tf_set)
        adata_v.var["is_target"] = True
        if not adata_v.var["is_reg"].any():
            raise ValueError("no TFs overlap with var_names; cannot set is_reg mask")

        h5_path = root_dir / f"{dataset_name}.h5ad"
        adata_v.write_h5ad(h5_path)

        cmd = [
            "velorama",
            "-n", dataset_name,
            "-ds", dataset_name,
            "-dyn", dynamics,
            "-dev", device,
            "-l", str(n_lags),
            "-hd", str(hidden),
            "-rd", str(root_dir),
            "-sd", str(save_dir),
        ]
        print(f"[velorama] launching: {' '.join(cmd)}", flush=True)
        # Stream stdout/stderr live so progress is visible; tee into a log file
        # so we can still inspect the tail on failure.
        log_path = save_dir / f"{dataset_name}.velorama.log"
        import time as _time
        t_start = _time.time()
        with open(log_path, "wb") as logf:
            proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                                    stderr=subprocess.STDOUT, bufsize=1)
            try:
                assert proc.stdout is not None
                for line in proc.stdout:
                    sys.stdout.buffer.write(line)
                    sys.stdout.buffer.flush()
                    logf.write(line)
                    if _time.time() - t_start > timeout_sec:
                        proc.kill()
                        raise TimeoutError(
                            f"velorama exceeded {timeout_sec}s; killed. "
                            f"Full log at {log_path}"
                        )
                rc = proc.wait()
            except Exception:
                proc.kill()
                raise
        if rc != 0:
            tail = log_path.read_text(errors="replace")[-2000:]
            raise RuntimeError(
                f"velorama CLI failed (rc={rc})\nLog tail:\n{tail}"
            )

        # Search standard output locations
        candidates = [
            save_dir / f"{dataset_name}.{dynamics}.velorama.interactions.tsv",
            root_dir / f"{dataset_name}.{dynamics}.velorama.interactions.tsv",
        ]
        out_tsv = next((p for p in candidates if p.exists()), None)
        if out_tsv is None:
            # fall back to glob
            matches = list(save_dir.rglob(f"{dataset_name}.{dynamics}.velorama.interactions.tsv"))
            if not matches:
                matches = list(save_dir.rglob("*.velorama.interactions.tsv"))
            if not matches:
                raise FileNotFoundError(
                    f"velorama produced no interactions.tsv under {save_dir}"
                )
            out_tsv = matches[0]

        # rows=targets, cols=regulators
        df = pd.read_csv(out_tsv, sep="\t", index_col=0)
        df.index = df.index.astype(str)
        df.columns = df.columns.astype(str)

    tf_set = set(tfs)
    regs = [c for c in df.columns if c in tf_set]
    if not regs:
        # Velorama wasn't restricted to TF set (it scored all genes); filter here
        regs = list(df.columns)
        regs = [r for r in regs if r in tf_set] or list(df.columns)

    rows = []
    for reg in regs:
        col = df[reg].abs()
        for tgt, score in col.items():
            if tgt == reg:
                continue
            rows.append((reg, tgt, float(score)))
    return pd.DataFrame(rows, columns=["regulator", "target", "score"])
