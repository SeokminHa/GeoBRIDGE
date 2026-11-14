#!/usr/bin/env python
# Run Velorama on SERGIO exports with official z-score preprocessing.
import glob, hashlib, json, os, shutil, subprocess, sys, time
import numpy as np

sys.setrecursionlimit(10000)
R = __import__("os").environ["GEOBRIDGE_ROOT"]


def zscore(M):
    import torch
    sd = M.std(0); sd[sd == 0] = 1
    return torch.clip(torch.FloatTensor((M - M.mean(0)) / sd), -5, 5)


LAMS = np.logspace(-2, 1, num=19).tolist()
CONFIG = dict(x_norm="zscore_clip5", dynamics="rna_velocity", proba=1, n_neighbors=30, velo_mode="stochastic", use_time=0, n_comps=50,
              lag=5, hidden=[32], penalty="H", lr=0.01, lam_ridge=0.0, max_iter=1000, lookback=5, check_every=10, seed=0, reg_target=0,
              lams=[round(l, 10) for l in LAMS])


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def key_valid(key):
    chk = dict(key); digest = chk.pop("binding_sha256", None)
    return digest == hashlib.sha256(json.dumps(chk, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


# Upstream Velorama sources, this harness, the gene roster and the full 19-λ configuration
def cache_key(d, tag):
    up = sorted(glob.glob(f"{R}/external/velorama_official/velorama/velorama/*.py"))
    genes = np.asarray(__import__("anndata").read_h5ad(f"{d}/{tag}.h5ad", backed="r").var_names).astype(str)
    parts = dict(input_h5ad=_sha(f"{d}/{tag}.h5ad"), upstream={os.path.basename(p): _sha(p) for p in up},
                 harness=_sha(os.path.abspath(__file__)), roster=hashlib.sha256("\n".join(genes).encode()).hexdigest(), config=CONFIG)
    parts["binding_sha256"] = hashlib.sha256(json.dumps(parts, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return parts


# reuse cached .pt tensors only if the stored key matches exactly; otherwise move them aside (never silently reuse)
def prepare_cache(out, key):
    kp = f"{out}/CACHE_KEY.json"
    if os.path.exists(kp):
        stored = json.load(open(kp))
        if key_valid(stored) and stored == key:
            return "REUSE"
    if os.path.exists(out) and (os.path.exists(kp) or glob.glob(f"{out}/**/*.pt", recursive=True) or glob.glob(f"{out}/**/*.npz", recursive=True)):
        stale = f"{out}.stale_{time.strftime('%Y%m%dT%H%M%S')}_{time.time_ns()}"; shutil.move(out, stale)
        print(f"[v2] cache key mismatch -> moved to {stale}")
    os.makedirs(out, exist_ok=True); json.dump(key, open(kp, "w"), indent=1)
    return "RETRAIN"


def write_export_sidecar(export, key):
    sidecar = dict(export_file=os.path.basename(export), export_sha256=_sha(export), cache_key=key)
    json.dump(sidecar, open(export + ".CACHE_KEY.json", "w"), indent=1)
    return sidecar


def validate_export(export, expected_key):
    sidecar_path = export + ".CACHE_KEY.json"
    if not os.path.exists(export) or not os.path.exists(sidecar_path):
        raise RuntimeError(f"REFUSING_VELORAMA_EXPORT_MISSING_OR_UNBOUND: {export}")
    sidecar = json.load(open(sidecar_path)); stored_key = sidecar.get("cache_key", {})
    if sidecar.get("export_file") != os.path.basename(export) or sidecar.get("export_sha256") != _sha(export):
        raise RuntimeError(f"REFUSING_VELORAMA_EXPORT_SHA_MISMATCH: {export}")
    if not key_valid(stored_key) or stored_key != expected_key:
        raise RuntimeError(f"REFUSING_VELORAMA_BINDING_KEY_MISMATCH: {export}")
    return sidecar


def run(d):
    import scanpy as sc, torch
    from velorama.utils import construct_dag, calculate_diffusion_lags
    from velorama.train import train_model
    tag = os.path.basename(d.rstrip("/")); out = f"{d}/velorama_v2"
    print(f"[v2] {tag} cache: {prepare_cache(out, cache_key(d, tag))}", flush=True)
    A_ = sc.read_h5ad(f"{d}/{tag}.h5ad")
    Xd = A_.X.toarray() if hasattr(A_.X, "toarray") else np.asarray(A_.X, float)
    genes = np.asarray(A_.var_names).astype(str)
    X = zscore(Xd); Y = zscore(Xd)
    sc.pp.scale(A_)
    A = torch.FloatTensor(construct_dag(A_, dynamics="rna_velocity", proba=1, n_neighbors=30, velo_mode="stochastic",
                                        use_time=0, n_comps=50))
    AX = calculate_diffusion_lags(A, X, 5)
    dn = f"{tag}.seed0.h32.H.lag5.rna_velocity"; os.makedirs(f"{out}/{dn}", exist_ok=True)
    for fn_ in ("reg_genes.txt", "target_genes.txt"):
        open(f"{out}/{fn_}", "w").write("\n".join(genes) + "\n")
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    for lam in LAMS:
        if os.path.exists(f"{out}/{dn}/{tag}.seed0.lam{np.round(lam, 4)}.h32.H.lag5.rna_velocity.pt"):
            continue
        t0 = time.time()
        train_model({"name": tag, "AX": AX, "AY": None, "Y": Y, "seed": 0, "lr": 0.01, "lam": lam, "lam_ridge": 0.0,
                     "penalty": "H", "lag": 5, "hidden": [32], "max_iter": 1000, "device": dev, "lookback": 5,
                     "check_every": 10, "verbose": False, "dynamics": "rna_velocity", "results_dir": out,
                     "dir_name": dn, "reg_target": 0})
        print(f"[v2] {tag} lam={lam:.4g} {time.time()-t0:.0f}s", flush=True)
    # official aggregation via the existing harvester, then (regulator x target) scores on the
    # candidate rows
    subprocess.run([sys.executable, f"{R}/pairspace/gb_velorama_harvest.py", "--velorama-root", f"{R}/external/velorama_official",
                    "--search-root", out, "--name", tag, "--dynamics", "rna_velocity", "--reg-genes", f"{out}/reg_genes.txt",
                    "--target-genes", f"{out}/target_genes.txt", "--stage-dir", f"{out}/stage", "--out-dir", f"{out}/harvest",
                    "--input-h5ad", f"{d}/{tag}.h5ad"], check=True)
    import pandas as pd
    C = json.load(open(f"{d}/cands.json"))
    T = pd.read_csv(f"{out}/harvest/{tag}.rna_velocity.velorama.interactions.tsv", sep="\t", index_col=0)  # rows targets
    names = [f"gene_{i}" for i in range(C["n"])]
    sc_ = T.loc[names, [names[c] for c in C["cands"]]].to_numpy(float).T                              # (cands, targets)
    export = f"{out}/{tag}.velorama_v2.npz"; np.savez_compressed(export, score=sc_)
    write_export_sidecar(export, cache_key(d, tag))
    print(f"[v2] {tag} scores {sc_.shape}", flush=True)


def selftest():
    import tempfile
    t = tempfile.mkdtemp(); out = f"{t}/v"
    def key(a):
        k = dict(a=a); k["binding_sha256"] = hashlib.sha256(json.dumps(k, sort_keys=True, separators=(",", ":")).encode()).hexdigest(); return k
    k1, k2 = key(1), key(2)
    assert prepare_cache(out, k1) == "RETRAIN"; open(f"{out}/x.pt", "w").write("t")
    assert prepare_cache(out, k1) == "REUSE"                                   # identical key -> reuse
    bad = json.load(open(f"{out}/CACHE_KEY.json")); bad["a"] = 999; json.dump(bad, open(f"{out}/CACHE_KEY.json", "w"))
    assert prepare_cache(out, k1) == "RETRAIN"                                 # tampered sidecar fails its own digest
    open(f"{out}/x.pt", "w").write("t")
    assert prepare_cache(out, k2) == "RETRAIN" and not os.path.exists(f"{out}/x.pt")   # changed key -> stale moved aside
    assert glob.glob(f"{out}.stale_*/x.pt")
    export = f"{t}/scores.npz"; np.savez_compressed(export, score=np.eye(2)); write_export_sidecar(export, k2)
    assert validate_export(export, k2)["export_sha256"] == _sha(export)
    with open(export, "ab") as f:
        f.write(b"tamper")
    try:
        validate_export(export, k2)
        raise AssertionError("tampered export was accepted")
    except RuntimeError as e:
        assert "SHA_MISMATCH" in str(e)
    print("gb_sergio_velorama_v2 selftest PASS")


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        selftest()
    else:
        for d in sys.argv[1:]:
            run(d)
