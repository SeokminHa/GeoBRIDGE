# TF32REP replication driver
import glob, hashlib, itertools, json, os, shutil, socket, subprocess, sys, time
import numpy as np
R = __import__("os").environ["GEOBRIDGE_ROOT"]; sys.path.insert(0, f"{R}/scratch")
MAIN_ROOT = f"{R}/results/sergio_tf32rep_2026_10_22"; ND = f"{R}/results/sergio_tf32nd_2026_10_22"
ROOT = MAIN_ROOT + ("/_fastparity" if "--parity-root" in sys.argv else "")      # fast variant: --parity-root isolates run dirs/queue state from the main run
THIS = f"{R}/scratch/gb_tf32rep.py"; GEN = f"{R}/scratch/gb_tf32rep_simulator.py"; GENM = "gb_tf32rep_reference"; GENM_ND = "gb_tf32nd_reference"
PY = __import__("os").path.join(__import__("os").environ.get("GB_JUPYTER_ENV_BIN", __import__("os").path.dirname(__import__("sys").executable)), "python")
GRNS = json.load(open(f"{MAIN_ROOT}/grn_seeds.json"))["grns"] if os.path.exists(f"{MAIN_ROOT}/grn_seeds.json") else []
TOPOS = ("bifurcation", "tree"); COND = dict(C0=(0, 0), C1=(.5, 0), C2=(0, .5), C3=(.5, .5)); SIMS = (1, 2); E_FIXED = 240
EXT_ALL = ("deepsem", "regdiffusion", "fasttenet", "genie3", "grnboost2", "pidc", "pearson"); GPU_EXT = ("deepsem", "regdiffusion", "fasttenet"); CPU_EXT = ("genie3", "grnboost2", "pidc", "pearson")
METHODS = ("GB", "VEL") + EXT_ALL
gdir = lambda g, t: f"{ROOT}/g{g}/{t}"; tagof = lambda t, g, s: f"{t}_AB_q2_lam1_obsUMI_DEPTH_depth10_g{g}_s{s}"
sha = lambda b: hashlib.sha256(b).hexdigest()
def ND_GRN(): return json.load(open(f"{ND}/grn_seeds.json"))
def dsdir(g, t, c, s): return f"{gdir(g, t)}/{c}/{tagof(t, g, s)}"
def h5path(g, t, c, s): d = dsdir(g, t, c, s); return f"{d}/{os.path.basename(d)}.h5ad"
def pilot_h5(t, c): g = ND_GRN()[t]; d = f"{ND}/{t}/{c}/{tagof(t, g, 1)}"; return f"{d}/{os.path.basename(d)}.h5ad"
def score_path(m, key):
    if key[0] == "rep": _, g, t, c, s = key; return f"{ROOT}/scores/{m}/g{g}_{t}_{c}_s{s}.npz"
    _, t, c = key; return f"{ROOT}/pilot_ext/scores/{m}/{t}_{c}_s1.npz"
def dist_matrix(gt, n=100):
    adj = {}; [adj.setdefault(s, []).append(t) for s, t in gt]; D = np.full((n, n), 99, int)
    for a in range(n):
        D[a, a] = 0; fr = [a]; d = 0
        while fr:
            d += 1; nx = []
            for u in fr:
                for v in adj.get(u, []):
                    if D[a, v] > d: D[a, v] = d; nx.append(v)
            fr = nx
    return D
def V_():
    import importlib; return importlib.import_module(GENM)
def prep(g, t):
    V = V_(); G = V.generate_grn(g); gt = sorted((int(s), int(x)) for s, x in G["gt"]); cands = [int(x) for x in G["cands"]]
    assert len(gt) == E_FIXED, f"E={len(gt)} != {E_FIXED}"
    assert cands == sorted(G["regs"] + G["decoys"]) and len(cands) == 32 and len(G["decoys"]) == 0 and all(any(s == r for s, _ in gt) for r in G["regs"]) and not any(s in G["decoys"] for s, _ in gt)
    cols = np.array([[x for x in range(100) if x != s] for s in cands]); D = dist_matrix(gt); rc = np.array(cands)
    c = {k: int((D[rc][:, :] == k).sum()) for k in (1, 2, 3, 4)}; N = len(cands) * 99
    ndir = sum(1 for s, x in gt if x in set(cands)); out = dict(topo=t, grn_seed=g, n_regs=len(cands), n_direct=len(gt), n_universe=N, prevalence=len(gt) / N, n_mrs=len(G["mrs"]), n_ints=len(G["ints"]), n_leaves=len(G["leaves"]),
        reg_to_reg_edges=ndir, strict_hop_pairs_regulator_sources={f"dist{k}": v for k, v in c.items()}, decoys=[int(x) for x in G["decoys"]], min_outdeg=min(sum(1 for s, _ in gt if s == r) for r in G["regs"]),
        pos_neg_sign=[sum(1 for e in gt if G["K"][e] > 0), sum(1 for e in gt if G["K"][e] < 0)], cands=cands, cols_sha=sha(cols.tobytes()), truth_sha=sha(json.dumps(gt).encode()))
    os.makedirs(gdir(g, t), exist_ok=True); json.dump(out, open(f"{gdir(g, t)}/roster_pre.json", "w"), indent=1); print({k: v for k, v in out.items() if k != "cands"})

def graph_stats(V, g, exact):
    G = V.generate_grn(g); gt = sorted((int(s), int(x)) for s, x in G["gt"]); cands = G["cands"]; L = G["layer"]; D = dist_matrix(gt); rc = np.array(cands)
    od = np.array([sum(1 for s, _ in gt if s == r) for r in G["regs"]]); indeg = {v: 0 for v in range(100)}
    for _, x in gt: indeg[x] += 1
    valid = (len(cands) == 32 and len(G["regs"]) == 32 and len(G["decoys"]) == 0 and all(L[s] < L[x] for s, x in gt) and all(indeg[m] == 0 for m in G["mrs"])
             and all(indeg[v] >= 1 for v in range(100) if v not in G["mrs"]) and not any(s == x for s, x in gt) and not any(s in G["decoys"] for s, _ in gt) and od.min() >= 1 and (len(gt) == E_FIXED or not exact))
    N = len(cands) * 99; npos = sum(1 for e in gt if G["K"][e] > 0)
    return dict(seed=g, E=len(gt), prev_TF=len(gt) / N, outdeg_min=int(od.min()), outdeg_med=float(np.median(od)), outdeg_max=int(od.max()), outdeg_sd=float(od.std()),
        reg_to_reg=sum(1 for s, x in gt if x in set(G["regs"])), n2=int((D[rc] == 2).sum()), n3=int((D[rc] == 3).sum()), n4=int((D[rc] == 4).sum()), frac_pos_sign=npos / len(gt), valid=bool(valid))
def audit(s0, n):
    import importlib, pandas as pd
    Vr, Vn = importlib.import_module(GENM), importlib.import_module(GENM_ND); os.makedirs(ROOT, exist_ok=True)
    T = pd.DataFrame([graph_stats(Vr, g, True) for g in range(s0, s0 + n)]); T["pass"] = T.valid & (T.n2 >= 100) & (T.n3 >= 50); Tn = pd.DataFrame([graph_stats(Vn, g, False) for g in range(s0, s0 + n)])
    ok = T[T["pass"]]["seed"].tolist()
    # distortion: exact-240 (rep) vs tf32nd Bernoulli, same seeds, over valid seeds of each
    # generator
    rows = []; a, b = T[T.valid], Tn[Tn.valid]
    for k in ("E", "outdeg_min", "outdeg_med", "outdeg_max", "outdeg_sd", "reg_to_reg", "n2", "n3", "n4", "frac_pos_sign"):
        mr, mn = float(a[k].median()), float(b[k].median()); rows.append(dict(stat=k, nd_bernoulli_median=mn, rep_exact240_median=mr, rel_shift=(mr - mn) / mn if mn else np.nan, nd_mean=float(b[k].mean()), rep_mean=float(a[k].mean()), nd_sd=float(b[k].std()), rep_sd=float(a[k].std())))
    both = T.merge(Tn, on="seed", suffixes=("_rep", "_nd")); rows.append(dict(stat="n2 seed-matched median diff (rep-nd)", nd_bernoulli_median=np.nan, rep_exact240_median=float((both.n2_rep - both.n2_nd).median()), rel_shift=np.nan))
    Dt = pd.DataFrame(rows); shift = {k: float(Dt[Dt.stat == k].rel_shift.iloc[0]) for k in ("n2", "outdeg_sd")}; stop_dist = any(abs(v) > 0.25 for v in shift.values())
    txt = (f"# TF32REP graph-only feasibility audit (no simulation, no method output)\nGenerator: scratch/gb_tf32rep_reference.py generate_grn (EXACT E=240). Audit seeds {s0}..{s0 + n - 1}. valid = DAG layering, MR in-degree 0, every non-MR in-degree>=1, 32 active/0 decoy, E==240, no self-edges, all out-degree>=1.\n"
           "pass = valid & strict2>=100 & strict3>=50; cohort = FIRST SIX passing seeds in numeric order (no other criterion).\n\n" + T.round(4).to_markdown(index=False) + f"\n\nPassing seeds ({len(ok)}): {ok}\n\n"
           "## Distortion report: exact-240 vs TF32ND Bernoulli generator, same seeds, medians over valid seeds\n\n" + Dt.round(4).to_markdown(index=False) + f"\n\nSTOP rule (|rel shift| > 0.25 in median strict2 or outdeg sd): shifts = {shift} -> triggered = {stop_dist}\n")
    T.to_csv(f"{ROOT}/audit_rep.tsv", sep="\t", index=False); Tn.to_csv(f"{ROOT}/audit_nd_bernoulli.tsv", sep="\t", index=False); Dt.to_csv(f"{ROOT}/distortion.tsv", sep="\t", index=False)
    if stop_dist or len(ok) < 6:
        open(f"{ROOT}/STOP", "w").write(f"STOP: distortion_triggered={stop_dist} n_pass={len(ok)} shifts={shift}\n"); open(f"{ROOT}/FEASIBILITY.md", "w").write(txt + "\n**STOP written; no grn_seeds.json.**\n")
        print(T.round(4).to_string()); print(Dt.round(4).to_string()); print("STOP", stop_dist, len(ok), shift); return
    pick = [int(x) for x in ok[:6]]; json.dump(dict(grns=pick), open(f"{ROOT}/grn_seeds.json", "w")); open(f"{ROOT}/FEASIBILITY.md", "w").write(txt + f"\nPICKED (first six passing): {pick}\n")
    print(T.round(4).to_string()); print(Dt.round(4).to_string()); print("PICKED", pick, "stop_rule", stop_dist, shift)

# run primitives (parent logic over (GRN, topo))
def sh(cmd, env=None, log=None):
    e = dict(os.environ, PYTHONNOUSERSITE="1", OMP_NUM_THREADS="8", MKL_NUM_THREADS="8", **(env or {})); f = open(log, "a") if log else None
    return subprocess.run(cmd, env=e, stdout=f, stderr=subprocess.STDOUT).returncode
def harvest(g, t): return sh([PY, GEN, "ab_harvest", gdir(g, t), ".5", ".5"], dict(AB_GRN=str(g), AB_TOPO=t), f"{gdir(g, t)}/harvest.log")
def run(g, t, cond, s):
    a, c = COND[cond]; return sh([PY, GEN, "ab_run", gdir(g, t), cond, str(a), str(c), ".5", ".5", str(s)], dict(AB_GRN=str(g), AB_TOPO=t, GB_PILOT_FAST="1", GB_REP_FAST="1"), f"{gdir(g, t)}/log_{cond}_s{s}.txt")
def velo(g, t, cond, s, gpu):
    import gb_velorama_official_rt0 as M
    tag = tagof(t, g, s); h5 = f"{gdir(g, t)}/{cond}/{tag}/{tag}.h5ad"; pre = json.load(open(f"{gdir(g, t)}/roster_pre.json")); cands = pre["cands"]
    cols = np.array([[x for x in range(100) if x != q] for q in cands]); assert sha(cols.tobytes()) == pre["cols_sha"]
    while True:
        if os.path.exists(h5) and time.time() - os.path.getmtime(h5) > 60:
            try:
                import anndata; anndata.read_h5ad(h5, backed="r").file.close(); break
            except Exception: pass
        time.sleep(20)
    st = f"{gdir(g, t)}/velo_stage/{cond}/{tag}"; os.makedirs(f"{st}/raw", exist_ok=True)
    if not os.path.exists(f"{st}/{tag}.h5ad"): os.symlink(h5, f"{st}/{tag}.h5ad")
    json.dump(dict(cands=cands, n=100), open(f"{st}/cands.json", "w")); np.savez(f"{st}/raw/geobridge_raw.npz", source_gene_index=np.array(cands), cols_PI=cols)
    M.OUT = f"{gdir(g, t)}/velorama_rt0/{cond}"; vo = f"{M.OUT}/{tag}.npz"
    if os.path.exists(vo) and not score_ok(vo): os.remove(vo)        # stale/interrupted output must be recomputed (M.run skips an existing file)
    M.run(st, int(gpu))
    if not score_ok(vo):
        if os.path.exists(vo): os.remove(vo)
        raise SystemExit(f"INVALID velorama output {vo}")
    z = np.load(vo)
    if not (np.array_equal(z["src"], np.array(cands)) and np.array_equal(z["cols"], cols)):
        os.remove(vo); raise SystemExit(f"velorama roster mismatch {vo}")
    print("velo done", g, t, cond, s, flush=True)
def veloq(gpu, jobs):
    for j in jobs.split(","):
        g, t, c, s = j.split(":"); velo(int(g), t, c, int(s), gpu)

# external comparators (frozen wrappers; input = h5ad .X, sources = 32 candidate TFs, targets = cols_PI)
# completion validity of a score npz: readable, (32,99) score, all 3168 finite, src/cols present
def score_ok(p):
    try:
        z = np.load(p); sc = np.asarray(z["score"], float); return sc.shape == (32, 99) and bool(np.isfinite(sc).all()) and np.asarray(z["src"]).shape == (32,) and np.asarray(z["cols"]).shape == (32, 99)
    except Exception: return False
def rec_ok(g, t, c, s):
    try: return json.load(open(f"{dsdir(g, t, c, s)}/record.json")).get("status") == "OK"
    except Exception: return False
def ext_load(h5, pre):
    import anndata; a = None
    for _ in range(4):
        try: a = anndata.read_h5ad(h5); break
        except Exception: time.sleep(30)
    if a is None: a = anndata.read_h5ad(h5)
    X = np.asarray(a.X.toarray() if hasattr(a.X, "toarray") else a.X, float); assert list(a.var_names) == [f"gene_{i}" for i in range(X.shape[1])]
    src = np.array(pre["cands"], int); cols = np.array([[x for x in range(100) if x != q] for q in src]); assert sha(cols.tobytes()) == pre["cols_sha"]
    return a, X, src, cols
def ext_run(m, key):
    if key[0] == "rep": _, g, t, c, s = key; h5 = h5path(g, t, c, s); pre = json.load(open(f"{gdir(g, t)}/roster_pre.json"))
    else: _, t, c = key; h5 = pilot_h5(t, c); pre = json.load(open(f"{ND}/{t}/roster_pre.json"))
    a, X, src, cols = ext_load(h5, pre); nodes = [str(v) for v in a.var_names]; t0 = time.time()
    if m in GPU_EXT:
        import gb_sota_ext as SE, gb_topo_adv_gpu as G
        assert (X.std(0) > 0).all(); P = dict(X=X, node_idx=np.arange(X.shape[1]), tf_node_idx=src, nodes=nodes, tfs=[nodes[s] for s in src])
        M, _ = {"regdiffusion": lambda: SE.run_regdiffusion(P, 0), "deepsem": lambda: SE.run_deepsem(P, 0), "fasttenet": lambda: G.run_tenet(P, a, SE)}[m](); M = np.abs(np.asarray(M, float))
    elif m in ("genie3", "grnboost2"):
        import gb_sota_run as SR
        from dask.distributed import Client, LocalCluster
        client = Client(LocalCluster(n_workers=8, threads_per_worker=1, memory_limit="0"))
        try:
            Lk = SR.arboreto_fit(X, nodes, [nodes[s] for s in src], nodes, m, 0, client)
        finally: client.close()
        M = np.zeros((len(src), X.shape[1])); ri = {nodes[s]: k for k, s in enumerate(src)}
        for tf, tg, w in zip(Lk.TF, Lk.target, Lk.importance): M[ri[tf], int(tg.split("_")[1])] = w
    elif m == "pidc":
        import torch, gb_sota_ext as SE; torch.set_num_threads(4); M, _ = SE.pidc_weights(X, src, device="cpu", log=lambda *a_, **k: None)
    elif m == "pearson":
        # Absolute Pearson correlation on column-centered Ms, as in gb_real_uniform.pearson_scores.
        Ms = np.asarray(a.layers["Ms"].toarray() if hasattr(a.layers["Ms"], "toarray") else a.layers["Ms"], np.float64); Xs = Ms[:, src] - Ms[:, src].mean(0); nx = np.sqrt(np.sum(Xs * Xs, axis=0)); Yc = Ms - Ms.mean(0); ny = np.sqrt(np.sum(Yc * Yc, axis=0))
        with np.errstate(invalid="ignore", divide="ignore"): M = np.abs(Xs.T @ Yc / (nx[:, None] * ny[None, :]))
    else: raise SystemExit(f"unknown method {m}")
    sc = np.take_along_axis(np.asarray(M, float), cols, 1); assert sc.shape == (32, 99) and np.isfinite(sc).all(), f"nonfinite/shape {sc.shape} {int((~np.isfinite(sc)).sum())}"
    p = score_path(m, key); os.makedirs(os.path.dirname(p), exist_ok=True); tmp = p + ".tmp.npz"
    np.savez_compressed(tmp, score=sc, src=src, cols=cols, cols_sha=pre["cols_sha"], wall=time.time() - t0); os.replace(tmp, p); print("ext done", m, key, f"{time.time() - t0:.0f}s", flush=True)

# persistent queue
def D_(sub): os.makedirs(f"{ROOT}/{sub}", exist_ok=True); return f"{ROOT}/{sub}"
def ready_h5(h5): return os.path.exists(h5) and time.time() - os.path.getmtime(h5) > 60
def mk(name, argv, outs, ready=None, deps=(), gate_run=None, env=None):
    return dict(name=name, argv=argv, outs=outs, ready=ready or (lambda: True), deps=list(deps), gate_run=gate_run, env=env or {})
def tasks_cpu():
    T = []
    for g in GRNS:
        for t in TOPOS: T.append(mk(f"ph_g{g}_{t}", ["ph", str(g), t], lambda g=g, t=t: all(os.path.exists(f"{gdir(g, t)}/{f}") for f in ("roster_pre.json", "frozen_half.json"))))
    for g in GRNS:
        for t in TOPOS:
            for s in SIMS:
                for c in COND: T.append(mk(f"run_g{g}_{t}_{c}_s{s}", ["run", str(g), t, c, str(s)], lambda g=g, t=t, c=c, s=s: all(os.path.exists(f"{dsdir(g, t, c, s)}/{f}") for f in ("record.json", "raw/geobridge_raw.npz")) and os.path.exists(h5path(g, t, c, s)) and rec_ok(g, t, c, s), deps=[f"ph_g{g}_{t}"]))
    return T
def tasks_ext(methods, pilot_first=False):
    T = []; PIL = []      # pilot tasks (data already exist): first for cpuext, last for GPU (Velorama on replication data has priority)
    for t in TOPOS:
        for c in COND:
            for m in methods:
                if m in EXT_ALL: PIL.append(mk(f"pext_{m}_{t}_{c}", ["ext", m, "pilot", t, c], lambda m=m, t=t, c=c: score_ok(score_path(m, ("pilot", t, c))), ready=lambda t=t, c=c: ready_h5(pilot_h5(t, c))))
    for m in methods:
        for g in GRNS:
            for t in TOPOS:
                for s in SIMS:
                    for c in COND:
                        if m == "velorama": T.append(mk(f"velo_g{g}_{t}_{c}_s{s}", ["velo1", "{GPU}", str(g), t, c, str(s)], lambda g=g, t=t, c=c, s=s: score_ok(f"{gdir(g, t)}/velorama_rt0/{c}/{tagof(t, g, s)}.npz"), ready=lambda g=g, t=t, c=c, s=s: ready_h5(h5path(g, t, c, s)), gate_run=f"run_g{g}_{t}_{c}_s{s}"))
                        else: T.append(mk(f"ext_{m}_g{g}_{t}_{c}_s{s}", ["ext", m, "rep", str(g), t, c, str(s)], lambda m=m, g=g, t=t, c=c, s=s: score_ok(score_path(m, ("rep", g, t, c, s))), ready=lambda g=g, t=t, c=c, s=s: ready_h5(h5path(g, t, c, s)), gate_run=f"run_g{g}_{t}_{c}_s{s}"))
    return PIL + T if pilot_first else T + PIL
def state(n): return "done" if os.path.exists(f"{ROOT}/_done/{n}") else "fail" if os.path.exists(f"{ROOT}/_fail/{n}") else "locked" if os.path.isdir(f"{ROOT}/_locks/{n}") else "open"
_REG = {}
def reg(): 
    if not _REG: _REG.update({tk["name"]: tk for tk in tasks_cpu() + tasks_ext(["velorama"] + list(EXT_ALL))})
    return _REG
# transitive: first failed prerequisite (deps or gate_run, recursively), else None
def blocked_by(tk):
    for d in tk["deps"] + ([tk["gate_run"]] if tk["gate_run"] else []):
        if state(d) == "fail": return d
        r = reg().get(d); b = blocked_by(r) if r else None
        if b: return b
    return None
# Mark open tasks with a failed prerequisite as blocked, so no worker waits forever
def propagate_blocked(tasks):
    for tk in tasks:
        if state(tk["name"]) == "open":
            b = blocked_by(tk)
            if b:
                try: open(f"{ROOT}/_fail/{tk['name']}", "x").write(f"blocked_by:{b}\n")
                except FileExistsError: pass
def dead(tk): return blocked_by(tk) is not None
def later(tk): return state(tk["name"]) == "open" and not dead(tk)
def claimable(tk): return later(tk) and all(state(d) == "done" for d in tk["deps"]) and tk["ready"]()
def qworker(tasks, slots, label, gpu=None):
    for s_ in ("_locks", "_done", "_fail", "_logs"): D_(s_)
    if "--retry-failed" in sys.argv:
        own = {x["name"] for x in tasks}
        for tk in reg().values():     # own failed tasks and any blocked marker (blocked tasks must be re-evaluated once their prerequisite is retried)
            if state(tk["name"]) == "fail" and (tk["name"] in own or open(f"{ROOT}/_fail/{tk['name']}").read().startswith("blocked_by:")):
                os.remove(f"{ROOT}/_fail/{tk['name']}"); shutil.rmtree(f"{ROOT}/_locks/{tk['name']}", ignore_errors=True)
    host = socket.gethostname(); running = {}; print(f"[{label}] host={host} slots={slots} tasks={len(tasks)}", flush=True)
    while True:
        for n, (p, t0, tk) in list(running.items()):
            rc = p.poll()
            if rc is None: continue
            del running[n]; t1 = time.time(); okout = bool(tk["outs"]())
            if rc == 0 and okout:
                json.dump(dict(host=host, start=t0, end=t1, wall=t1 - t0, rc=rc), open(f"{ROOT}/_done/{n}.tmp", "w")); os.replace(f"{ROOT}/_done/{n}.tmp", f"{ROOT}/_done/{n}")
            else:
                tail = open(f"{ROOT}/_logs/{n}.log", errors="replace").read()[-4000:]; open(f"{ROOT}/_fail/{n}", "w").write(f"host={host} rc={rc} outputs_ok={okout} wall={t1 - t0:.0f}\n{tail}")
            print(f"[{label}] {'DONE' if rc == 0 and okout else 'FAIL'} {n} {t1 - t0:.0f}s", flush=True)
        launched = False; propagate_blocked(tasks)
        while len(running) < slots:
            tk = next((x for x in tasks if claimable(x)), None)
            if tk is None: break
            try: os.mkdir(f"{ROOT}/_locks/{tk['name']}")
            except FileExistsError: continue
            if os.path.exists(f"{ROOT}/_done/{tk['name']}"): continue
            argv = [a_.replace("{GPU}", str(gpu)) for a_ in tk["argv"]]; env = dict(os.environ, PYTHONNOUSERSITE="1", MPLCONFIGDIR="/tmp", **tk["env"])
            if gpu is not None and tk["argv"][0] == "ext" and tk["argv"][1] in GPU_EXT: env["CUDA_VISIBLE_DEVICES"] = str(gpu)
            lf = open(f"{ROOT}/_logs/{tk['name']}.log", "w"); p = subprocess.Popen([PY, "-s", THIS] + argv, stdout=lf, stderr=subprocess.STDOUT, env=env, cwd=R)
            running[tk["name"]] = (p, time.time(), tk); launched = True; print(f"[{label}] START {tk['name']}", flush=True)
        if not running and not any(later(x) for x in tasks): print(f"[{label}] nothing left; exit", flush=True); break
        time.sleep(2 if launched else 20)
def worker(slots): qworker(tasks_cpu(), slots, "worker")
def cpuext(slots, methods): qworker(tasks_ext([m for m in methods.split(",") if m in CPU_EXT], pilot_first=True), slots, "cpuext")
def gpuworker(gpu, methods):
    ms = methods.split(","); ordered = (["velorama"] if "velorama" in ms else []) + [m for m in GPU_EXT if m in ms]; qworker(tasks_ext(ordered), 1, f"gpu{gpu}", gpu=gpu)
def unlock():
    for tk in tasks_cpu() + tasks_ext(["velorama"] + list(EXT_ALL)):
        if state(tk["name"]) == "locked": shutil.rmtree(f"{ROOT}/_locks/{tk['name']}"); print("unlocked", tk["name"])
def status():
    import collections
    C = collections.defaultdict(lambda: collections.Counter())
    for tk in tasks_cpu() + tasks_ext(["velorama"] + list(EXT_ALL)):
        n = tk["name"]; st = state(n); stage = n.split("_g")[0] if n.startswith(("ph_", "run_", "velo_")) else "_".join(n.split("_")[:2]) if n.startswith(("ext_", "pext_")) else n
        C[stage]["running" if st == "locked" else st] += 1
    for k in sorted(C): print(f"{k:22s} done={C[k]['done']:3d} running={C[k]['locked']+C[k]['running']:3d} fail={C[k]['fail']:3d} open={C[k]['open']:3d}")
    for f in sorted(glob.glob(f"{ROOT}/_fail/*"))[:10]: print("FAIL", os.path.basename(f))

# metrics (parent formulas, unchanged)
def branch_r2(M, br, tb):
    bm = br != 0; out = np.zeros(M.shape[1]); w = 0
    for b in np.unique(tb[bm]):
        m = bm & (tb == b); y = br[m]
        if (y > 0).sum() < 10 or (y < 0).sum() < 10: continue
        X = M[m]; v = X.var(0) + 1e-12; mu = np.array([X[y > 0].mean(0), X[y < 0].mean(0)]); pw = np.array([(y > 0).mean(), (y < 0).mean()])
        out += m.sum() * (pw[:, None] * (mu - X.mean(0)) ** 2).sum(0) / v; w += m.sum()
    return out / max(w, 1)
def metrics(sc, Y, Dm):
    from sklearn.metrics import roc_auc_score as AUC, average_precision_score as AP
    ok = np.isfinite(sc); r = dict(AP=AP(Y[ok], sc[ok]), AUROC=AUC(Y[ok], sc[ok]))
    for k in (2, 3):
        m = ok & ((Dm == 1) | (Dm == k)); r[f"DvI{k}"] = AUC(Dm[m] == 1, sc[m])
    r.update(n_finite=int(ok.sum()), coverage=float(ok.mean())); return r
# ctx = dict(V=generator module, g, t, gd=dir of (GRN,topo), ext=lambda m,cond,s -> score path)
# Returns (long rows, diag row)
def one(ctx, cond, s, diag=False):
    V, g, t, gd = ctx["V"], ctx["g"], ctx["t"], ctx["gd"]; G = V.generate_grn(g); gt = {(int(a), int(b)) for a, b in G["gt"]}; D = dist_matrix(sorted(gt)); tag = tagof(t, g, s); dd = f"{gd}/{cond}/{tag}"
    pre = json.load(open(f"{gd}/roster_pre.json")); Z = np.load(f"{dd}/raw/geobridge_raw.npz"); src, cols = np.asarray(Z["source_gene_index"], int), np.asarray(Z["cols_PI"], int)
    pc = np.array([[x for x in range(100) if x != q] for q in pre["cands"]]); assert list(src) == pre["cands"] and np.array_equal(cols, pc), "GB roster != precomputed roster"
    import gb_b4_score as B
    A = {r: np.asarray(Z[f"TRUE|U|{r}|absS"], float) for r in B.GP.ROOTS}; S = {"GB": B.zc([{r: A[r][:, i, cols[i]].T for r in B.GP.ROOTS} for i in range(len(src))], range(0, 9))}; wall = {}
    vf = f"{gd}/velorama_rt0/{cond}/{tag}.npz"
    if os.path.exists(vf): z = np.load(vf); assert np.array_equal(z["src"], src) and np.array_equal(z["cols"], cols); S["VEL"] = np.asarray(z["score"], float)
    for m in EXT_ALL:
        p = ctx["ext"](m, cond, s)
        if os.path.exists(p):
            z = np.load(p); assert np.array_equal(z["src"], src) and np.array_equal(z["cols"], cols) and str(z["cols_sha"]) == pre["cols_sha"]; S[m] = np.asarray(z["score"], float); wall[m] = float(z["wall"])
    Dm = np.array([[D[s_, int(x)] for x in cols[i]] for i, s_ in enumerate(src)]); Y = Dm == 1; rec = json.load(open(f"{dd}/record.json")); assert len(pre["decoys"]) == 0 and Y.sum() == len(gt)
    base = dict(grn=g, topo=t, cond=cond, sim=s, K=int(Y.sum()), N=Y.size, prev=float(Y.mean()), n2=int((Dm == 2).sum()), n3=int((Dm == 3).sum()), status=rec["status"])
    rows = []
    for nm, sc in S.items():
        bad = None if (sc.shape == (32, 99) and np.isfinite(sc).all()) else f"nonfinite_or_shape n_finite={int(np.isfinite(sc).sum())} shape={sc.shape}"
        if nm == "GB" and rec["status"] != "OK": bad = (bad or "") + f" record.status={rec['status']}"
        if bad: rows.append(dict(base, method=nm, invalid=bad)); continue
        r = metrics(sc, Y, Dm); r.update(base); r.update(method=nm, random_AP=base["prev"], lift=r["AP"] / base["prev"], wall=wall.get(nm, np.nan)); rows.append(r)
    dg = None
    if diag:
        import anndata
        dg = dict(base, G_E=rec.get("G_E"), n_mr_clipped=rec["ab"]["n_mr_clipped"])
        try:
            T = np.load(f"{dd}/raw/counts_truth.npz"); Gm = np.load(f"{dd}/raw/geometry_raw.npz"); a = anndata.read_h5ad(f"{dd}/{tag}.h5ad"); d_ = lambda M: np.asarray(M.toarray() if hasattr(M, "toarray") else M, float)
            Ms, Mu = d_(a.layers["Ms"]), d_(a.layers["Mu"]); tau = np.asarray(T["sampled_tau"], float); bn = np.asarray(T["sampled_bin"], int); n = len(tau)
            def cc(A_, B_): A_ = (A_ - A_.mean(0)) / (A_.std(0) + 1e-12); B_ = (B_ - B_.mean(0)) / (B_.std(0) + 1e-12); return A_.T @ B_ / len(A_)
            from sklearn.metrics import roc_auc_score as AUC
            Cu = np.take_along_axis(np.abs(cc(Ms[:, src], Mu)), cols, 1); dg["absr_nonedge_mean"] = float(Cu[~Y].mean()); dg["absr_edge_mean"] = float(Cu[Y].mean()); dg["absr_edge_vs_nonedge_AUROC"] = AUC(Y.ravel(), Cu.ravel())
            br = np.asarray(V.BRANCH[t])[bn]; frac = np.clip(tau - np.asarray(V.DEPTH[t])[bn], 0, .999); tb = (np.asarray(V.DEPTH[t])[bn] * 5 + np.floor(frac * 5)).astype(int)
            dg["branchcov_Mu_R2_mean"] = float(branch_r2(Mu, br, tb).mean()); dg["branchcov_Ms_R2_mean"] = float(branch_r2(Ms, br, tb).mean())
            hA, hB = V.ab_hn(t); cA, cB = V.ab_loadings(g); amp = rec["ab"]["amp"]; iA, iB = np.zeros((n, 100)), np.zeros((n, 100))
            for m in G["mrs"]: iA[:, m] = amp[0] * cA[m] * hA[bn]; iB[:, m] = amp[1] * cB[m] * hB[bn]
            Ed, cm = Gm["edges"], np.asarray(Gm["component_edge_mask"], bool); Pg, _ = V.production_parts(np.asarray(Gm["conc_unspliced_latent"], float), Ed, component_mask=cm); gp = {"A": [], "B": []}
            Ph = {k: V.production_parts(np.asarray(Gm["conc_unspliced_latent"], float), Ed, component_mask=cm & (np.asarray(Ed["regulator"]) == 100 + j))[1] for j, k in enumerate("AB")}
            for rt in V.ROOTS:
                I, J = Gm[f"pairs_{rt}_I"], Gm[f"pairs_{rt}_J"]; eG = ((Pg[J] - Pg[I]) ** 2).sum()
                for k, ii in (("A", iA), ("B", iB)): H_ = Ph[k] + ii; q_ = ((H_[J] - H_[I]) ** 2).sum(); gp[k].append(q_ / (q_ + eG))
            dg["G_A_alone"], dg["G_B_alone"] = float(np.mean(gp["A"])), float(np.mean(gp["B"]))
        except Exception as e: dg["DIAG_ERROR"] = repr(e)[:200]
    return rows, dg
def invariance():
    V = V_(); out = {}
    for g in GRNS:
        G = V.generate_grn(g); gt = {(int(a), int(b)) for a, b in G["gt"]}; tsh = {}
        for t in TOPOS:
            fz = json.load(open(f"{gdir(g, t)}/frozen_half.json")); fk = {(int(a), int(b)) for a, b, _ in fz["half"]}; tsh[t] = json.load(open(f"{gdir(g, t)}/roster_pre.json"))["truth_sha"]
            rec = dict(generator_sha=sha(open(V.__file__, "rb").read())[:12], grn_seed=g, topo=t, truth_sha=sha(json.dumps(sorted(gt)).encode())[:12], frozen_half_file_sha=sha(open(f"{gdir(g, t)}/frozen_half.json", "rb").read())[:12],
                       E=len(gt), frozen_keys_eq_truth_edges=({(b, a) for a, b in fk} == gt), n_frozen=len(fk), n_truth=len(gt)); P = {}
            for c in COND:
                for s in SIMS:
                    d = glob.glob(f"{gdir(g, t)}/{c}/*_s{s}")
                    if not d: continue
                    r = json.load(open(f"{d[0]}/record.json")); rows = {}
                    for l in open(f"{d[0]}/sergio_inputs/input_GRN.txt"):
                        f_ = l.strip().split(",")
                        if not f_ or f_[0] == "": continue
                        n = int(float(f_[1]))
                        for q, K, co in zip(f_[2:2 + n], f_[2 + n:2 + 2 * n], f_[2 + 2 * n:2 + 3 * n]):
                            if int(float(q)) < 100: rows[(int(float(q)), int(float(f_[0])))] = (float(K), float(co))
                    P[(c, s)] = dict(keys=set(rows) == gt, rows=rows, dev=r["ab"].get("frozen_half_max_abs_dev"), nchk=r["ab"].get("n_frozen_edges_checked"), truth=r["truth_sha256"][:12])
            rec["per_run_observed_keys_eq_truth"] = {f"{k[0]}_s{k[1]}": v["keys"] for k, v in P.items()}; rec["frozen_half_max_dev"] = {f"{k[0]}_s{k[1]}": v["dev"] for k, v in P.items()}
            rec["n_frozen_edges_checked_eq_truth"] = all(v["nchk"] == len(gt) for v in P.values()); ref = next(iter(P.values()))["rows"] if P else {}
            rec["observed_edge_K_coop_equal_across_conditions_and_sims"] = all(v["rows"] == ref for v in P.values()); rec["observed_K_equals_generator"] = all(abs(ref[e][0] - G["K"][e]) < 1e-9 for e in ref) if ref else None
            rec["truth_sha_equal"] = len({v["truth"] for v in P.values()}) == 1; rec["n_runs_audited"] = len(P); out[f"g{g}_{t}"] = rec
        out[f"g{g}_W_identical_across_topologies"] = len(set(tsh.values())) == 1
    json.dump(out, open(f"{ROOT}/INVARIANCE_AUDIT.json", "w"), indent=1, default=str); print(json.dumps(out, indent=1, default=str))
# PRIMARY (32x99) exactly as in one(): production GB = mean-z over k=0..8 then root-min, from dd/raw/geobridge_raw.npz TRUE|U|r|absS
def gb_primary(dd):
    Z = np.load(f"{dd}/raw/geobridge_raw.npz"); src, cols = np.asarray(Z["source_gene_index"], int), np.asarray(Z["cols_PI"], int)
    import gb_b4_score as B
    A = {r: np.asarray(Z[f"TRUE|U|{r}|absS"], float) for r in B.GP.ROOTS}
    return B.zc([{r: A[r][:, i, cols[i]].T for r in B.GP.ROOTS} for i in range(len(src))], range(0, 9))
# parity <ref_dataset_dir> <new_dataset_dir>: identity of inputs/pairs/rosters/masks/status + max|PRIMARY_ref - PRIMARY_new| <= 1e-10
def parity(ref, new):
    import anndata
    if not os.path.exists(f"{ref}/record.json"):          # directories of datasets: match by basename
        pairs_ = [(d, f"{new}/{os.path.relpath(d, ref)}") for d in sorted(glob.glob(f"{ref}/**/record.json", recursive=True)) for d in [os.path.dirname(d)]]
        ok = [parity(a_, b_) for a_, b_ in pairs_]; print("ALL", "PASS" if ok and all(ok) else "FAIL", len(ok)); return bool(ok) and all(ok)
    chk = {}; mx = {}
    def eqarr(name, A, B_):
        A, B_ = np.asarray(A), np.asarray(B_); same = A.shape == B_.shape and (np.array_equal(A, B_, equal_nan=True) if A.dtype.kind in "fc" else np.array_equal(A, B_))
        chk[name] = bool(same)
        if A.dtype.kind in "fc" and A.shape == B_.shape and A.size:
            f = np.isfinite(A) & np.isfinite(B_); mx[name] = float(np.abs(A[f] - B_[f]).max()) if f.any() else 0.0
    dn = lambda M: np.asarray(M.toarray() if hasattr(M, "toarray") else M)
    a, b = anndata.read_h5ad(glob.glob(f"{ref}/*.h5ad")[0]), anndata.read_h5ad(glob.glob(f"{new}/*.h5ad")[0])
    eqarr("h5ad.obs_names", a.obs_names.values.astype(str), b.obs_names.values.astype(str)); eqarr("h5ad.var_names", a.var_names.values.astype(str), b.var_names.values.astype(str)); eqarr("h5ad.X", dn(a.X), dn(b.X))
    chk["h5ad.layer_keys"] = sorted(a.layers.keys()) == sorted(b.layers.keys()); chk["h5ad.obsm_keys"] = sorted(a.obsm.keys()) == sorted(b.obsm.keys())
    for k in a.layers.keys():
        if k in b.layers: eqarr(f"h5ad.layer.{k}", dn(a.layers[k]), dn(b.layers[k]))
    for k in a.obsm.keys():
        if k in b.obsm: eqarr(f"h5ad.obsm.{k}", np.asarray(a.obsm[k]), np.asarray(b.obsm[k]))
    for f_ in ("counts_truth", "geometry_raw"):                         # every member the driver may read, plus all others
        za, zb = np.load(f"{ref}/raw/{f_}.npz", allow_pickle=True), np.load(f"{new}/raw/{f_}.npz", allow_pickle=True); chk[f"{f_}.keys"] = sorted(za.files) == sorted(zb.files)
        for k in za.files:
            if k in zb.files: eqarr(f"{f_}.{k}", za[k], zb[k])
    za, zb = np.load(f"{ref}/raw/geobridge_raw.npz"), np.load(f"{new}/raw/geobridge_raw.npz")
    chk["raw.new_keys_subset_of_ref"] = set(zb.files) <= set(za.files)      # fast run writes only TRUE|U + roster + pairs; every key it writes is compared to the reference
    for k in zb.files:
        if k in za.files: eqarr(f"raw.{k}", za[k], zb[k])
    for k in ("source_gene_index", "cols_PI", "cols_specv1") + tuple(f"pairs|{r}|{x}" for r in ("dc1_min", "dc1_max") for x in ("I", "J", "pt")) + tuple(f"TRUE|U|{r}|{m}" for r in ("dc1_min", "dc1_max") for m in ("num", "den", "z2", "absS")):
        chk[f"required.{k}"] = k in za.files and k in zb.files
    ra, rb = json.load(open(f"{ref}/record.json")), json.load(open(f"{new}/record.json"))
    chk["record.status"] = ra["status"] == rb["status"]; chk["record.status_OK_ref"] = ra["status"] == "OK"
    for k in ("truth_sha256", "roster_sha256", "G_E", "cell_sample_sha", "geometry_raw_sha256", "prevalence", "n_pairs_dc1_min", "n_pairs_dc1_max"): chk[f"record.{k}"] = ra.get(k) == rb.get(k)
    chk["record.ab"] = {k: v for k, v in ra["ab"].items()} == {k: v for k, v in rb["ab"].items()}
    Pa, Pb = gb_primary(ref), gb_primary(new); chk["PRIMARY.shape"] = Pa.shape == Pb.shape == (32, 99); chk["PRIMARY.nan_mask"] = bool(np.array_equal(np.isnan(Pa), np.isnan(Pb)))
    f = np.isfinite(Pa) & np.isfinite(Pb); pm = float(np.abs(Pa[f] - Pb[f]).max()) if f.any() else float("nan")
    bad = [k for k, v in chk.items() if not v]; allmax = max(mx.values()) if mx else 0.0
    ok = not bad and np.isfinite(pm) and pm <= 1e-10
    print(f"parity {os.path.basename(ref)}: {'PASS' if ok else 'FAIL'}  max|PRIMARY diff|={pm:.3e}  max other float diff={allmax:.3e}  n_checks={len(chk)}  failed={bad}")
    if ra.get("wall_s") is not None or rb.get("wall_s") is not None: print(f"  wall_s ref={ra.get('wall_s')} new={rb.get('wall_s')}  timers new={rb.get('timers_s')} ref={ra.get('timers_s')}")
    return ok
def evaluate():
    import pandas as pd
    L, DG, err = [], [], []
    for g in GRNS:
        for t in TOPOS:
            ctx = dict(V=V_(), g=g, t=t, gd=gdir(g, t), ext=lambda m, c, s, g=g, t=t: score_path(m, ("rep", g, t, c, s)))
            for c in COND:
                for s in SIMS:
                    try: r, d = one(ctx, c, s, diag=True); L += r; DG.append(d)
                    except Exception as e: err.append(dict(grn=g, topo=t, cond=c, sim=s, ERROR=repr(e)[:200]))
    T0 = pd.DataFrame(L); inv = T0[T0.invalid.notna()] if "invalid" in T0 else T0.iloc[:0]; T = T0[T0.invalid.isna()] if "invalid" in T0 else T0
    inv.to_csv(f"{ROOT}/INVALID.tsv", sep="\t", index=False); T.drop(columns=["invalid"], errors="ignore").to_csv(f"{ROOT}/TABLE.tsv", sep="\t", index=False); pd.DataFrame(DG).to_csv(f"{ROOT}/DIAG.tsv", sep="\t", index=False); pd.DataFrame(err).to_csv(f"{ROOT}/EVAL_ERRORS.tsv", sep="\t", index=False)
    exp = len(GRNS) * len(TOPOS) * len(COND) * len(SIMS); comp = [f"# Completeness (expected {exp} datasets per method; GB errors: {len(err)}; INVALID method-rows excluded from TABLE and listed in INVALID.tsv: {len(inv)})"]
    for m in METHODS:
        n = int((T.method == m).sum()) if len(T) else 0; comp.append(f"- {m}: {n}/{exp}" + ("" if n == exp else f"  MISSING {exp - n} (reported, never imputed)"))
    if len(T): comp.append(f"- GB non-OK status: {int((T[T.method == 'GB'].status != 'OK').sum())}; datasets with all 9 methods: {int((T.groupby(['grn', 'topo', 'cond', 'sim']).method.nunique() == 9).sum())}")
    open(f"{ROOT}/COMPLETENESS.md", "w").write("\n".join(comp) + "\n"); print("\n".join(comp))
    # pilot externals (separate, never pooled)
    PL = []; ng = ND_GRN()
    for t in TOPOS:
        ctx = dict(V=__import__(GENM_ND), g=ng[t], t=t, gd=f"{ND}/{t}", ext=lambda m, c, s, t=t: score_path(m, ("pilot", t, c)))
        for c in COND:
            try: r, _ = one(ctx, c, 1); PL += r
            except Exception as e: PL.append(dict(topo=t, cond=c, method="ERROR", ERR=repr(e)[:200]))
    pd.DataFrame(PL).to_csv(f"{ROOT}/PILOT_EXT_TABLE.tsv", sep="\t", index=False)
def sign_flip_p(d):
    d = np.asarray(d, float); obs = d.mean(); sg = np.array(list(itertools.product([1, -1], repeat=len(d)))); return float((((sg * d).mean(1)) >= obs - 1e-12).mean())
def contrast(d):
    from scipy import stats
    d = np.asarray(d, float); n = len(d); m = d.mean(); se = d.std(ddof=1) / np.sqrt(n); tq = stats.t.ppf(.975, n - 1)
    return dict(n_grn=n, mean=m, ci_lo=m - tq * se, ci_hi=m + tq * se, p_signflip_1sided=sign_flip_p(d), n_pos=int((d > 0).sum()), loo=[float(np.delete(d, i).mean()) for i in range(n)])
def analyze():
    import pandas as pd, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    T = pd.read_csv(f"{ROOT}/TABLE.tsv", sep="\t"); out = ["# TF32REP analysis unit = GRN; datasets are NOT independent replicates", ""]
    cell = T.groupby(["grn", "topo", "cond", "method"])[["AP", "DvI2", "DvI3", "AUROC", "lift", "random_AP"]].mean().reset_index(); cell.to_csv(f"{ROOT}/CELLS.tsv", sep="\t", index=False)
    def glev(df, stratum):
        r = df.groupby(["grn", "method"]).agg(AP=("AP", "mean"), DvI2=("DvI2", "mean"), DvI3=("DvI3", "mean"), AUROC=("AUROC", "mean"), n_ds=("AP", "size")).reset_index(); r["stratum"] = stratum; return r
    GL = [glev(T, "all")] + [glev(T[T.topo == t], f"topo={t}") for t in TOPOS] + [glev(T[T.cond == c], f"cond={c}") for c in COND]; GLd = pd.concat(GL); GLd.to_csv(f"{ROOT}/GRN_LEVEL.tsv", sep="\t", index=False)
    full = GLd[GLd.stratum == "all"]; nds = full.groupby("grn").n_ds.min()
    if (nds < 16).any(): out += [f"WARNING: GRNs with <16 datasets for some method (missing, not imputed): {nds[nds < 16].to_dict()}", ""]
    KEYS = ["grn", "topo", "cond", "sim"]; EXPN = len(TOPOS) * len(COND) * len(SIMS)
    if T.duplicated(KEYS + ["method"]).any(): raise SystemExit("DUPLICATE dataset keys in TABLE.tsv")
    W = {k: T.pivot(index=KEYS, columns="method", values=k) for k in ("AP", "DvI2")}
    def mdiff(a, b, k):      # matched per-dataset differences (only datasets where both methods exist)
        w = W[k]; return (w[a] - w[b]).dropna() if (a in w and b in w) else pd.Series(dtype=float, index=pd.MultiIndex.from_arrays([[]] * 4, names=KEYS))
    def grn_level(d):
        gm = d.groupby(level="grn").mean(); cnt = d.groupby(level="grn").size(); complete = all(g in gm.index and cnt[g] == EXPN for g in GRNS) and len(GRNS) >= 2
        return gm, cnt, complete
    def cstat(label, role, a, b, k):
        d = mdiff(a, b, k); gm, cnt, complete = grn_level(d)
        if complete: return dict(contrast=label, role=role, status="CONFIRMATORY", **contrast(gm.loc[GRNS].values), n_ds_per_grn=EXPN, grns=list(GRNS))
        return dict(contrast=label, role=role, status="NO_CALL", n_grn=int(len(gm)), mean=float(gm.mean()) if len(gm) else np.nan, n_ds_per_grn=str(cnt.to_dict()), note="incomplete matched datasets: descriptive only", grns=list(gm.index))
    rows = [cstat(f"GB-VEL {k}", role, "GB", "VEL", k) for k, role in (("AP", "PRIMARY"), ("DvI2", "SECONDARY"))]
    for k in ("AP", "DvI2"):
        fam = [cstat(f"GB-{m} {k}", f"external Holm-7 ({k})", "GB", m, k) for m in EXT_ALL]      # family size fixed at 7; NO_CALL counts as p=1
        pv = [x["p_signflip_1sided"] if x["status"] == "CONFIRMATORY" else 1.0 for x in fam]; run_max = 0.0
        for rank, i in enumerate(sorted(range(7), key=lambda i: pv[i])):
            run_max = max(run_max, min(1.0, (7 - rank) * pv[i]))
            if fam[i]["status"] == "CONFIRMATORY": fam[i]["p_holm"] = run_max
        rows += fam
    C = pd.DataFrame(rows); C.to_csv(f"{ROOT}/CONTRASTS.tsv", sep="\t", index=False)
    out += ["## GRN-level paired contrasts (per-dataset matched GB-X differences, averaged within GRN, then over GRNs; confirmatory only if all 16 datasets exist for all 6 GRNs, else NO_CALL)", "", C.drop(columns=["grns"]).round(4).to_markdown(index=False), "",
            "Exact one-sided sign-flip permutation p over the GRNs (2^n sign patterns); Holm over the fixed 7-external family (NO_CALL counted as p=1); 95% t-CI with n-1 df; loo = leave-one-GRN-out means.", ""]
    desc = []
    for stratum in GLd.stratum.unique():
        if stratum == "all": continue
        lv, val = stratum.split("="); 
        for k in ("AP", "DvI2"):
            d = mdiff("GB", "VEL", k); d = d[d.index.get_level_values("topo" if lv == "topo" else "cond") == val]
            if len(d): gm = d.groupby(level="grn").mean(); desc.append(dict(stratum=stratum, metric=k, GB_minus_VEL_mean=float(gm.mean()), n_grn_pos=int((gm > 0).sum()), n_grn=len(gm), n_ds=len(d)))
    out += ["## Descriptive stratum-specific GB-VEL GRN-level effects (no tests claimed)", "", pd.DataFrame(desc).round(4).to_markdown(index=False), "",
            "## GRN-level means, all datasets (AP)", "", full.pivot(index="grn", columns="method", values="AP").round(4).to_markdown(), "", "Random AP = 240/3168 = 0.0758 for every dataset. Cell-level table: CELLS.tsv (GRN x topo x cond, mean over sims).", ""]
    open(f"{ROOT}/ANALYSIS.md", "w").write("\n".join(out)); print("\n".join(out))
    col = dict(GB="#1b6ca8", VEL="#d95f02", deepsem="#7570b3", regdiffusion="#66a61e", fasttenet="#e6ab02", genie3="#a6761d", grnboost2="#666666", pidc="#e7298a", pearson="#1b9e77")
    f, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True); cs = list(COND)
    for i, t in enumerate(TOPOS):
        d = T[T.topo == t]
        for m in METHODS:
            dm = d[d.method == m]
            if dm.empty: continue
            gl = dm.groupby(["grn", "cond"]).AP.mean().unstack().reindex(columns=cs)
            if m in ("GB", "VEL"):
                for gg, r in gl.iterrows(): ax[i].plot(np.arange(4) + (.04 if m == "VEL" else -.04), r.values, "o", color=col[m], alpha=.4, ms=4)
            ax[i].plot(range(4), gl.mean().values, "-", color=col[m], lw=2.8 if m in ("GB", "VEL") else 1.2, label=m)
        ax[i].axhline(240 / 3168, color="k", ls=":", lw=.8); ax[i].set_xticks(range(4)); ax[i].set_xticklabels(cs); ax[i].set_title(f"{t}: pooled direct-edge AP (mean over GRNs; dots = GRNs)", fontsize=9)
    ax[0].set_ylabel("AP"); ax[1].legend(fontsize=7, ncol=2); f.tight_layout(); f.savefig(f"{ROOT}/AP_by_condition_topology.jpg", dpi=200, format="jpeg")

if __name__ == "__main__":
    c = sys.argv[1]; a = [x for x in sys.argv[2:] if not x.startswith("--")]
    if c == "prep": prep(int(a[0]), a[1])
    elif c == "harvest": sys.exit(harvest(int(a[0]), a[1]))
    elif c == "ph": prep(int(a[0]), a[1]); sys.exit(harvest(int(a[0]), a[1]))
    elif c == "run": sys.exit(run(int(a[0]), a[1], a[2], int(a[3])))
    elif c == "veloq": veloq(a[0], a[1])
    elif c == "velo1": velo(int(a[1]), a[2], a[3], int(a[4]), a[0])
    elif c == "ext" and a[0] in ("regdiffusion", "fasttenet") and os.path.realpath(sys.executable) != os.path.realpath(__import__("os").path.join(__import__("os").environ.get("GB_GB_SOTA_BIN", __import__("os").path.dirname(__import__("sys").executable)), "python")):
        sys.exit(subprocess.run([__import__("os").path.join(__import__("os").environ.get("GB_GB_SOTA_BIN", __import__("os").path.dirname(__import__("sys").executable)), "python"), "-s", THIS] + sys.argv[1:], env=dict(os.environ, PYTHONNOUSERSITE="1")).returncode)   # these two need the gb_sota env
    elif c == "ext": ext_run(a[0], ("rep", int(a[2]), a[3], a[4], int(a[5])) if a[1] == "rep" else ("pilot", a[2], a[3]))
    elif c == "audit": audit(int(a[0]) if a else 501, int(a[1]) if len(a) > 1 else 60)
    elif c == "worker": worker(int(a[0]))
    elif c == "cpuext": cpuext(int(a[0]), a[1])
    elif c == "gpuworker": gpuworker(a[0], a[1])
    elif c == "unlock": unlock()
    elif c == "status": status()
    elif c == "invariance": invariance()
    elif c == "eval": evaluate()
    elif c == "analyze": analyze()
    elif c == "parity": sys.exit(0 if parity(a[0], a[1]) else 1)
