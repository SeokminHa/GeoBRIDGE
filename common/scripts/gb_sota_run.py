#!/usr/bin/env python
# Comparator runs on canonical inputs.
import argparse, hashlib, json, os, resource, sys, time
import numpy as np

R = __import__("os").environ["GEOBRIDGE_ROOT"]


def fixture(system):
    return f"{R}/results/canonical_2026_09_18/{system}_canonical.h5ad"


def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


# catalogue and gene axis come from the evaluator run's own provenance, never from here
def bind(system):
    D = f"{R}/results/canon_{system}"
    LP = json.load(open(f"{D}/layers_provenance.json"))
    fx = LP["truth_artifacts"]["substrate_fixture"]
    if os.path.realpath(fx["path"]) != os.path.realpath(fixture(system)) or sha(fixture(system)) != fx["sha256"]:
        raise SystemExit("REFUSING: fixture differs from the evaluator's substrate_fixture")
    cat = LP["tf_catalogue"]
    if sha(cat) != LP["tf_catalogue_sha256"]:
        raise SystemExit("REFUSING: TF catalogue changed since the evaluator bound it")
    axis = [str(g) for g in np.load(f"{D}/per_gene.npz", allow_pickle=True)["genes"]]
    if len({g.upper() for g in axis}) != len(axis):
        raise SystemExit("REFUSING: case-insensitive duplicate genes on the evaluator axis")
    return dict(fixture=fixture(system), fixture_sha256=fx["sha256"], catalogue=cat,
                catalogue_sha256=LP["tf_catalogue_sha256"], axis=axis,
                cat={l.strip().upper() for l in open(cat) if l.strip()})


def catalogue_tfs(genes, B):
    return [g for g in genes if g.upper() in B["cat"]]


# per-target arboreto with meta; aborts unless every target fitted exactly once
def arboreto_fit(X, genes, tfs, targets, learner, seed, client):
    from arboreto.core import create_graph, SGBM_KWARGS, RF_KWARGS
    rtype, kw = {"grnboost2": ("GBM", SGBM_KWARGS), "genie3": ("RF", RF_KWARGS)}[learner]   # = public api profiles
    links, meta = create_graph(X, list(genes), list(tfs), regressor_type=rtype, regressor_kwargs=kw, client=client,
                               target_genes=list(targets), include_meta=True, seed=seed)
    links, meta = client.compute([links, meta], sync=True)
    got = meta["target"].astype(str).value_counts()
    bad = sorted(set(targets) - set(got.index)) + sorted(got.index[got != 1])
    if bad or set(got.index) - set(targets):
        raise SystemExit(f"REFUSING: {len(bad)} target regressions failed/duplicated (e.g. {bad[:5]}); no zero-fill")
    check_links(links)
    return links


# Require that the receipt's hash of the output matches the file being collected
def check_output_hash(receipt, key, path):
    if not receipt.get(key) or receipt[key] != sha(path):
        raise SystemExit(f"REFUSING: producer receipt {key} does not match {path} (missing or changed after receipt)")


# nonnegative, one row per (TF, target)
def check_links(links):
    w = links["importance"].to_numpy(float)
    if not np.isfinite(w).all() or (w < 0).any():
        raise SystemExit("REFUSING: non-finite or negative arboreto importances")
    if links.duplicated(["TF", "target"]).any():
        raise SystemExit("REFUSING: duplicated (TF, target) links")


def write(system, method, genes, tfs, score, meta):
    out = f"{R}/results/canon_{system}/sota"
    os.makedirs(out, exist_ok=True)
    tag = method + (".SMOKE" if meta.get("smoke") else "")
    p = f"{out}/{tag}.npz"
    assert score.shape == (len(tfs), len(genes)), score.shape
    if list(genes) != bind(system)["axis"] or len({t.upper() for t in tfs}) != len(tfs):
        raise SystemExit("REFUSING: output gene axis != evaluator axis, or duplicate regulators")
    np.savez_compressed(p, genes=np.asarray(genes).astype(str), tfs=np.asarray(tfs).astype(str),
                        score=score.astype(np.float32), meta=json.dumps(meta))
    print(f"wrote {p}  scored {np.isfinite(score).sum()}/{score.size} (TF,gene) cells", flush=True)


def run_grnboost2(a):
    import anndata as ad
    from distributed import LocalCluster, Client
    t0 = time.time(); B = bind(a.system)
    A = ad.read_h5ad(B["fixture"])
    fg = {g: i for i, g in enumerate(np.asarray(A.var_names).astype(str))}
    genes = B["axis"]
    # conventional GRNBoost2 input = the fixture's log-normalised .X (uns['log1p'] present)
    X = A.X.toarray() if hasattr(A.X, "toarray") else np.asarray(A.X)
    X = np.ascontiguousarray(X[:, [fg[g] for g in genes]], dtype=np.float64)
    tfs = catalogue_tfs(genes, B)
    smoke = bool(a.tfs or a.n_targets)
    if not smoke and a.seed != 0:
        raise SystemExit(f"REFUSING: full {a.cmd} run requires the frozen seed 0 (got {a.seed})")
    if a.tfs:
        want = {t.upper() for t in a.tfs.split(",")}
        tfs = [g for g in genes if g.upper() in want]
    targets = genes
    if a.n_targets:
        rng = np.random.default_rng(a.seed)
        targets = sorted(set(rng.choice(genes, a.n_targets, replace=False).tolist()) | set(tfs))
    print(f"[grnboost2] {a.system} cells={X.shape[0]} genes={len(genes)} tfs={len(tfs)} "
          f"targets={len(targets)} smoke={smoke}", flush=True)
    if not smoke and (targets != genes or tfs != catalogue_tfs(genes, B)):
        raise SystemExit("REFUSING: full run must regress every evaluator-axis target on the whole catalogue")
    gi = {g: i for i, g in enumerate(genes)}
    sub = [gi[g] for g in targets]            # TF columns are always inside `targets`
    cluster = LocalCluster(n_workers=a.workers, threads_per_worker=1, memory_limit="0")
    client = Client(cluster)
    try:
        net = arboreto_fit(X[:, sub], targets, tfs, targets, a.cmd, a.seed, client)
    finally:
        client.close(); cluster.close()
    score = np.full((len(tfs), len(genes)), np.nan, np.float32)
    ti = {t: i for i, t in enumerate(tfs)}
    # every target is now certified fitted (arboreto_fit), so its unlisted links are genuine 0
    # importances
    score[:, [gi[t] for t in targets]] = 0.0
    for tf, tg, w in net[["TF", "target", "importance"]].itertuples(index=False):
        score[ti[tf], gi[tg]] = w
    for t in tfs:
        score[ti[t], gi[t]] = np.nan
    meta = dict(method={"grnboost2": "GRNBoost2", "genie3": "GENIE3"}[a.cmd], package="arboreto", system=a.system, fixture=B["fixture"],
                fixture_sha256=B["fixture_sha256"], layer="X (log1p normalised)",
                catalogue=B["catalogue"], catalogue_sha256=B["catalogue_sha256"], gene_axis="evaluator per_gene.npz",
                targets_certified=len(targets), cells_genotype=sorted(set(A.obs["genotype"].astype(str)))
                if "genotype" in A.obs else None,
                seed=a.seed, n_tfs=len(tfs), n_targets=len(targets), smoke=smoke,
                tfs_override=a.tfs, n_targets_override=a.n_targets,
                wall_s=round(time.time() - t0, 1),
                peak_rss_gb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6, 2),
                self_sha256=sha(os.path.abspath(__file__)))
    write(a.system, a.cmd, genes, tfs, score, meta)


FROZEN_LAM_GRID = [0.01, 0.0147, 0.0215, 0.0316, 0.0464, 0.0681, 0.1, 0.1468, 0.2154, 0.3162, 0.4642, 0.6813, 1.0, 1.4678,
                   2.1544, 3.1623, 4.6416, 6.8129, 10.0]          # np.round(np.logspace(-2, 1, 19), 4): the official Velorama grid


# rna_velocity dynamics, lag 5 and the exact frozen 19-value lambda grid (values, not count)
def velorama_locked(H):
    why = []
    if H.get("seed") != 0:
        why.append(f"seed {H.get('seed')} != 0")
    if H.get("dynamics") != "rna_velocity":
        why.append(f"dynamics {H.get('dynamics')} != rna_velocity")
    if ".lag5." not in str(H.get("config_name", "")) + ".":
        why.append(f"config {H.get('config_name')} is not lag 5")
    g = H.get("lam_grid") or []
    if len(g) != len(FROZEN_LAM_GRID) or not np.allclose(g, FROZEN_LAM_GRID, rtol=0, atol=1e-9):
        why.append("lambda grid differs from the frozen 19-value grid")
    return why


# every saved-model file listed with a sha256 that still matches
def manifest_ok(m):
    return bool(m) and all(os.path.exists(e["path"]) and sha(e["path"]) == e["sha256"] for e in m)


# The full locked configuration (seed 0, 500 epochs, batch 1024, zero skeleton, lam 0), cuda training evidence and a saved model
def regvelo_primary(Rj, prod_sha, cat_sha=None):
    locked = (Rj.get("seed") == 0 and Rj.get("max_epochs") == 500 and Rj.get("batch_size") == 1024
              and Rj.get("skeleton") == "zeros" and Rj.get("lam") == 0 and str(Rj.get("train_device", "")).startswith("cuda")
              and Rj.get("model_saved") is True and manifest_ok(Rj.get("model_manifest"))
              and cat_sha is not None and Rj.get("catalogue_sha256") == cat_sha)
    primary = Rj.get("arm") == "PRIMARY_LOCKED" and locked and Rj.get("script_sha256") == prod_sha
    if Rj.get("arm") == "PRIMARY_LOCKED" and not primary:
        raise SystemExit("REFUSING: receipt claims PRIMARY_LOCKED without current producer bytes / locked config / CUDA evidence")
    return primary


def collect(a):
    import anndata as ad, pandas as pd
    B = bind(a.system)
    A = ad.read_h5ad(B["fixture"], backed="r")
    fgenes = list(np.asarray(A.var_names).astype(str))
    genes = B["axis"]; gi = {g: i for i, g in enumerate(genes)}
    meta = dict(method=a.method, system=a.system, fixture=B["fixture"], fixture_sha256=B["fixture_sha256"],
                catalogue=B["catalogue"], catalogue_sha256=B["catalogue_sha256"], gene_axis="evaluator per_gene.npz",
                self_sha256=sha(os.path.abspath(__file__)))
    fsha = meta["fixture_sha256"]
    if a.method == "velorama":
        hj = a.interactions.replace(".velorama.interactions.tsv", ".harvest.json")
        if not os.path.exists(hj):
            raise SystemExit(f"REFUSING: no harvest receipt {hj}")
        H = json.load(open(hj))
        if H.get("VERDICT") != "HARVESTED":
            raise SystemExit(f"REFUSING: harvest VERDICT={H.get('VERDICT')}")
        if os.path.realpath(H.get("interactions_tsv", "")) != os.path.realpath(a.interactions):
            raise SystemExit("REFUSING: harvest receipt certifies a different interactions TSV")
        check_output_hash(H, "interactions_tsv_sha256", a.interactions)
        if H.get("input_h5ad_sha256") != fsha:
            raise SystemExit("REFUSING: Velorama harvest receipt does not certify the canonical fixture")
        if H.get("missing_lambdas") or not (H.get("n_lambda_found") == len(H.get("lam_grid", [])) == 19):
            raise SystemExit("REFUSING: Velorama lambda grid incomplete per receipt")
        why = velorama_locked(H)
        if why:
            raise SystemExit("REFUSING: Velorama run is not the frozen configuration: " + "; ".join(why))
        if not H.get("gene_labels_supplied"):
            raise SystemExit("REFUSING: harvest matrix has placeholder gene labels")
        meta.update(receipt=hj, receipt_sha256=sha(hj))
        T = pd.read_csv(a.interactions, sep="\t", index_col=0)
        T.index = T.index.astype(str); T.columns = T.columns.astype(str)
        if T.index.duplicated().any() or T.columns.duplicated().any():
            raise SystemExit("REFUSING: duplicate gene labels in the harvest matrix")
        fs = set(fgenes)
        missing = [g for g in T.index if g not in fs] + [g for g in T.columns if g not in fs]
        if missing:
            raise SystemExit(f"REFUSING: {len(missing)} harvest genes not in the fixture universe")
        # truth-blind regulator roster: must be exactly the frozen catalogue on this fixture
        if list(T.columns) != catalogue_tfs(fgenes, B):
            raise SystemExit("REFUSING: Velorama regulator roster != catalogue_tfs(fixture genes)")
        tfs = [t for t in T.columns if t in gi]          # collision-dropped regulators leave with their axis genes
        if tfs != catalogue_tfs(genes, B):
            raise SystemExit("REFUSING: Velorama roster on the evaluator axis != catalogue on that axis")
        rows = [g for g in T.index if g in gi]
        score = np.full((len(tfs), len(genes)), np.nan, np.float32)
        for j, tf in enumerate(tfs):
            score[j, [gi[g] for g in rows]] = T.loc[rows, tf].to_numpy(np.float32)
            score[j, gi[tf]] = np.nan
        meta.update(source=a.interactions, source_sha256=sha(a.interactions), reg_target=1,
                    note="Granger interaction strength, unsigned; lags in the companion TSV")
    elif a.method == "regvelo":
        if not a.npz.endswith(".npz"):
            raise SystemExit("REFUSING: --npz must be the producer's <out>.npz")
        rj = a.npz[:-4] + ".json"
        if not os.path.exists(rj):
            raise SystemExit(f"REFUSING: no RegVelo producer receipt {rj}")
        Rj = json.load(open(rj))
        if Rj.get("script") != "gb_regvelo_noprior_gpu.py":
            raise SystemExit("REFUSING: RegVelo receipt is not from the Jacobian producer")
        if Rj.get("fixture_sha256") != fsha:
            raise SystemExit("REFUSING: RegVelo receipt does not certify the canonical fixture")
        check_output_hash(Rj, "npz_sha256", a.npz)
        if Rj.get("extraction_error") or not str(Rj.get("extraction", "")).startswith("GRN_Jacobian"):
            raise SystemExit("REFUSING: RegVelo extraction was not the Jacobian path")
        if Rj.get("tf") != a.tf:
            raise SystemExit("REFUSING: --tf differs from the producer's focal TF")
        Z = np.load(a.npz, allow_pickle=True)
        zg = list(np.asarray(Z["genes"]).astype(str))
        if zg != fgenes:
            raise SystemExit("REFUSING: regvelo gene axis != fixture universe")
        if a.tf not in gi:
            raise SystemExit("REFUSING: RegVelo focal is not on the evaluator axis")
        primary = regvelo_primary(Rj, sha(f"{R}/scratch/gb_regvelo_noprior_gpu.py"), B["catalogue_sha256"])
        a.method = "regvelo" if primary else f"regvelo_ablation_{Rj.get('skeleton')}_lam{Rj.get('lam'):g}"
        tfs = [a.tf]
        zi = {g: i for i, g in enumerate(zg)}
        score = np.asarray(Z["RegVelo"], np.float32)[None, [zi[g] for g in genes]]
        score[0, gi[a.tf]] = np.nan
        meta.update(method=a.method,
                    source=a.npz, source_sha256=sha(a.npz), receipt=rj, receipt_sha256=sha(rj),
                    signed=True, arm=Rj.get("arm"), skeleton=Rj.get("skeleton"), lam=Rj.get("lam"),
                    seed=Rj.get("seed"), extraction=Rj.get("extraction"),
                    output_diagnostics=Rj.get("output_diagnostics"),
                    output_degenerate=bool((Rj.get("output_diagnostics") or {}).get("OUTPUT_DEGENERATE")),
                    note="SIGNED outgoing Jacobian column GRN_Jacobian(mean Ms)[:, focal]; |score| for "
                         "unsigned ranking")
    else:
        raise SystemExit(a.method)
    write(a.system, a.method, genes, tfs, score, meta)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for _nm in ("grnboost2", "genie3"):
        g = sub.add_parser(_nm); g.add_argument("system")
        g.add_argument("--tfs"); g.add_argument("--n-targets", type=int)
        g.add_argument("--seed", type=int, default=0); g.add_argument("--workers", type=int, default=32)
    c = sub.add_parser("collect"); c.add_argument("system")
    c.add_argument("--method", required=True, choices=["velorama", "regvelo"])
    c.add_argument("--interactions"); c.add_argument("--npz"); c.add_argument("--tf")
    a = ap.parse_args()
    {"grnboost2": run_grnboost2, "genie3": run_grnboost2, "collect": collect}[a.cmd](a)
