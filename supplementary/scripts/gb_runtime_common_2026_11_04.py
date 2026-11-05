"""Common-host runtime comparison (SI). Reuses gb_tf32rep_fast_2026_10_22 runners unchanged; only redirects outputs to NEW dir and times.
  gb g topo | ext method g topo | velo g topo gpu     -> NEW/rt/<name>.json ; scores into NEW/scores (never touches original results)"""
import json, os, shutil, socket, subprocess, sys, time
sys.path.insert(0, (__import__("os").environ["GEOBRIDGE_ROOT"] + "/scratch"))
import gb_tf32rep_fast_2026_10_22 as T
NEW = f"{T.MAIN_ROOT}/runtime_common_2026_11_04"; os.makedirs(f"{NEW}/rt", exist_ok=True)
C, S = "C3", 1
def hw():
    gpu = subprocess.run("nvidia-smi --query-gpu=name --format=csv,noheader -i %s" % os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0], shell=True, capture_output=True, text=True).stdout.strip()
    return gpu
def save(name, d): json.dump(d, open(f"{NEW}/rt/{name}.json", "w"), indent=1); print(name, d, flush=True)
def gb(g, t):
    od = f"{NEW}/g{g}/{t}"; src = T.gdir(g, t)
    if not os.path.exists(od):
        os.makedirs(od)
        for f in ("frozen_half.json", "roster_pre.json"): shutil.copy(f"{src}/{f}", od)
        for d in ("orientation_harvest", "harvest_work"): shutil.copytree(f"{src}/{d}", f"{od}/{d}")
        os.makedirs(f"{od}/{C}"); shutil.copytree(f"{src}/{C}/orientation", f"{od}/{C}/orientation")
    T.ROOT = NEW      # gdir -> new tree; T.run launches the unchanged ab_run subprocess (OMP/MKL=8, GB_PILOT_FAST/GB_REP_FAST=1)
    t0 = time.perf_counter(); rc = T.run(g, t, C, S); tot = time.perf_counter() - t0
    rec = json.load(open(f"{T.dsdir(g, t, C, S)}/record.json")); tm = rec["timers_s"]
    save(f"GB_{g}_{t}", dict(rc=rc, status=rec.get("status"), total_wall_s=tot, host=socket.gethostname(), OMP=8, **tm,
        gb_score_only_s=tm["geometry_s"] + tm["score_pass1_s"], gb_pre_score_s=tm["preprocess_s"] + tm["geometry_s"] + tm["score_pass1_s"] + tm["rest_s"]))
def ext(m, g, t):
    T.score_path = lambda m_, key: f"{NEW}/scores/{m_}/g{key[1]}_{key[2]}_{key[3]}_s{key[4]}.npz"
    t0 = time.perf_counter(); T.ext_run(m, ("rep", g, t, C, S)); tot = time.perf_counter() - t0
    import numpy as np; z = np.load(T.score_path(m, ("rep", g, t, C, S)))
    save(f"{m}_{g}_{t}", dict(method_s=float(z["wall"]), total_wall_s=tot, host=socket.gethostname(), gpu=hw() if m in T.GPU_EXT else None))
def velo(g, t, gpu):
    import numpy as np, anndata
    import gb_velorama_official_rt0 as M
    h5 = T.h5path(g, T.__dict__.get("x", t), C, S) if False else T.h5path(g, t, C, S)
    pre = json.load(open(f"{T.gdir(g, t)}/roster_pre.json")); cands = pre["cands"]; tag = T.tagof(t, g, S)
    cols = np.array([[x for x in range(100) if x != q] for q in cands]); assert T.sha(cols.tobytes()) == pre["cols_sha"]
    st = f"{NEW}/velo_stage/{tag}"; os.makedirs(f"{st}/raw", exist_ok=True)
    if not os.path.exists(f"{st}/{tag}.h5ad"): os.symlink(h5, f"{st}/{tag}.h5ad")
    json.dump(dict(cands=cands, n=100), open(f"{st}/cands.json", "w")); np.savez(f"{st}/raw/geobridge_raw.npz", source_gene_index=np.array(cands), cols_PI=cols)
    M.OUT = f"{NEW}/velorama_rt0/g{g}_{t}"; vo = f"{M.OUT}/{tag}.npz"
    t0 = time.perf_counter(); M.run(st, int(gpu)); tot = time.perf_counter() - t0
    assert T.score_ok(vo); save(f"velorama_{g}_{t}", dict(method_s=tot, total_wall_s=tot, host=socket.gethostname(), gpu=hw(), note="M.run = staging+CLI subprocess+npz write"))
if __name__ == "__main__":
    a = sys.argv[1:]
    if a[0] == "gb": gb(int(a[1]), a[2])
    elif a[0] == "ext": ext(a[1], int(a[2]), a[3])
    elif a[0] == "velo": velo(int(a[1]), a[2], a[3])
