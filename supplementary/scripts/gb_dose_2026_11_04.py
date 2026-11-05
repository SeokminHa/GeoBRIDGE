"""Dose curve (small-N). usage: gen g cond a | eval g cond   (cond: C0/C3 read stored; others from DOSE root)"""
import sys, os, json, shutil, subprocess, numpy as np
R = __import__("os").environ["GEOBRIDGE_ROOT"]; sys.path.insert(0, f"{R}/scratch"); sys.argv_saved = list(sys.argv)
MAIN = f"{R}/results/sergio_tf32rep_2026_10_22"; DOSE = f"{MAIN}/dose_2026_11_04"; T = "bifurcation"
def gen(g, cond, a):
    gd = f"{DOSE}/g{g}/{T}"; os.makedirs(gd, exist_ok=True)
    for f in ("frozen_half.json", "roster_pre.json"): shutil.copy(f"{MAIN}/g{g}/{T}/{f}", f"{gd}/{f}")
    GEN = f"{R}/scratch/gb_pilot_AB_tf32rep_fast_2026_10_22.py"; PY = __import__("os").path.join(__import__("os").environ.get("GB_JUPYTER_ENV_BIN", __import__("os").path.dirname(__import__("sys").executable)), "python")
    e = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", AB_GRN=str(g), AB_TOPO=T, GB_PILOT_FAST="1", GB_REP_FAST="1")
    sys.exit(subprocess.run([PY, GEN, "ab_run", gd, cond, str(a), str(a), ".5", ".5", "1"], env=e, stdout=open(f"{gd}/log_{cond}_s1.txt", "w"), stderr=subprocess.STDOUT).returncode)
def ev(g, cond):
    import gb_tf32rep_fast_2026_10_22 as M, gb_b4_score as B
    stored = cond in ("C0", "C3"); gd = f"{(MAIN if stored else DOSE)}/g{g}/{T}"; tag = M.tagof(T, g, 1); dd = f"{gd}/{cond}/{tag}"
    V = M.V_(); G = V.generate_grn(g); gt = sorted((int(a), int(b)) for a, b in G["gt"]); D = M.dist_matrix(gt)
    Z = np.load(f"{dd}/raw/geobridge_raw.npz"); src, cols = np.asarray(Z["source_gene_index"], int), np.asarray(Z["cols_PI"], int)
    A = {r: np.asarray(Z[f"TRUE|U|{r}|absS"], float) for r in B.GP.ROOTS}; P = [{r: A[r][:, i, cols[i]].T for r in B.GP.ROOTS} for i in range(len(src))]
    Dm = np.array([[D[s, int(x)] for x in cols[i]] for i, s in enumerate(src)]); Y = Dm == 1; rec = json.load(open(f"{dd}/record.json")); out = dict(grn=g, cond=cond, status=rec["status"], ab=dict(amp=rec["ab"]["amp"], mr_off=rec["ab"]["mr_off"], n_mr_clipped=rec["ab"]["n_mr_clipped"]))
    for nm, ks in (("prod", range(0, 9)), ("k0", range(0, 1))): out[nm] = {k: float(v) for k, v in M.metrics(B.zc(P, ks), Y, Dm).items()}
    out["mr_rate_min"] = rec["ab"].get("mr_rate_min"); json.dump(out, open(f"{DOSE}/eval_g{g}_{cond}.json" if not stored else f"{DOSE}/eval_stored_g{g}_{cond}.json", "w"), indent=1); print(json.dumps(out))
if __name__ == "__main__":
    c = sys.argv[1]
    if c == "gen": gen(int(sys.argv[2]), sys.argv[3], float(sys.argv[4]))
    else: ev(int(sys.argv[2]), sys.argv[3])
