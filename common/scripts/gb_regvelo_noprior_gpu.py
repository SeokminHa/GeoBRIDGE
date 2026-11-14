#!/usr/bin/env python
# RegVelo no-prior comparator on GPU
import argparse, json, os, sys, time
import numpy as np, torch, anndata as ad

R = __import__("os").environ["GEOBRIDGE_ROOT"]
sys.path.insert(0, R); sys.path.insert(0, f"{R}/scratch")
import gb_hist_specificity_family as H          # single source of truth for the GATA1 fixture path
CFG = {
    "GATA1": H.FIXTURE,
    "GATA1_DET": f"{R}/results/gata1_clean_2026_08_13/gata1_control_deterministic.h5ad",
    "PANC": f"{R}/results/pr_arm2_2026_08_16/panc/panc_pr_stochastic.h5ad",
    # CANONICAL substrates (GEOBRIDGE_PREPROCESS_V1)
    "GATA1_CANON": f"{R}/results/canonical_2026_09_18/GATA1_canonical.h5ad",
    "PANC_CANON": f"{R}/results/canonical_2026_09_18/PANC_canonical.h5ad",
}
CANONICAL = {"GATA1_CANON", "PANC_CANON"}
REGS = f"{R}/data/LaraAstiaso_raw/ext_ref/external_regulators.txt"


# so the receipt binds the actual trained model
def model_manifest(d, _sha):
    out = []
    for root_, _, files in os.walk(d):
        for f_ in sorted(files):
            p_ = os.path.join(root_, f_); out.append(dict(path=p_, sha256=_sha(p_)))
    return sorted(out, key=lambda x: x["path"])


LOCKED = dict(seed=0, max_epochs=500, batch_size=1024, skeleton="zeros", lam=0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset"); ap.add_argument("tf"); ap.add_argument("seed", type=int)
    ap.add_argument("out"); ap.add_argument("--max_epochs", type=int, default=500)
    ap.add_argument("--batch_size", type=int, default=1024)
    ap.add_argument("--fixture", default=None)
    ap.add_argument("--lam", type=float, required=True)
    ap.add_argument("--skeleton", choices=["zeros", "ones"], required=True)
    a = ap.parse_args()
    t0 = time.time()
    import regvelo as rgv
    import scvi
    scvi.settings.seed = a.seed
    torch.manual_seed(a.seed); np.random.seed(a.seed)

    fx = a.fixture or CFG[a.dataset]
    A = ad.read_h5ad(fx)
    genes = np.asarray(A.var_names).astype(str)
    for L in ("Ms", "Mu"):
        if L not in A.layers: raise SystemExit(f"REFUSING: layer {L} absent from {fx}")
    cat = set(l.strip() for l in open(REGS) if l.strip())
    reg_names = [g for g in genes if g in cat]
    if a.tf not in reg_names: raise SystemExit(f"REFUSING: {a.tf} not in the regulator catalogue")
    print(f"[rgv] {A.shape} cuda={torch.cuda.is_available()} regulators={len(reg_names)} seed={a.seed}", flush=True)

    rgv.REGVELOVI.setup_anndata(A, spliced_layer="Ms", unspliced_layer="Mu")
    W = (torch.zeros if a.skeleton == "zeros" else torch.ones)((len(genes), len(genes)), dtype=torch.float32)
    REG = rgv.REGVELOVI(A, W=W, regulators=reg_names, soft_constraint=True, lam=a.lam)
    REG.train(max_epochs=a.max_epochs, batch_size=a.batch_size)
    print(f"[rgv] trained {time.time()-t0:.0f}s", flush=True)

    gi = int(np.where(genes == a.tf)[0][0])
    mod = REG.module
    score, how, err = None, None, None
    try:
        # preferred: the outgoing signed Jacobian row of the focal regulator
        fn = getattr(mod.v_encoder, "GRN_Jacobian", None) or getattr(mod.v_encoder, "GRN_Jacobian2", None)
        if fn is not None:
            with torch.no_grad():
                Jm = fn(torch.tensor(np.asarray(A.layers["Ms"]), dtype=torch.float32,
                                     device=next(mod.parameters()).device).mean(0))
            Jm = Jm.detach().cpu().numpy()
            score = Jm[:, gi]; how = "GRN_Jacobian[:, focal] (outgoing signed)"
        else:
            raise RuntimeError("no GRN_Jacobian on v_encoder")
    except Exception as e:
        err = str(e)[:300]
        Wm = mod.v_encoder.fc1.weight.detach().cpu().numpy()
        score = Wm[:, gi]; how = "FALLBACK fc1.weight[:, focal] -- NOT the Jacobian arm"
        print(f"[rgv] Jacobian extraction failed ({err}); fell back to {how}", flush=True)

    train_device = str(next(mod.parameters()).device)
    primary = (a.skeleton == "zeros" and a.lam == 0 and a.seed == LOCKED["seed"] and a.max_epochs == LOCKED["max_epochs"]
               and a.batch_size == LOCKED["batch_size"] and train_device.startswith("cuda") and torch.cuda.is_available())
    model_saved = True
    try:
        REG.save(a.out + "_model", overwrite=True, save_anndata=False)
    except Exception as e:
        model_saved = False
        if primary:
            raise SystemExit(f"REFUSING: model save failed for a PRIMARY_LOCKED run: {str(e)[:200]}")
        print(f"[rgv] WARNING model save failed: {str(e)[:200]}", flush=True)

    sc = np.asarray(score, float); fin = np.isfinite(sc); nz = np.abs(sc[fin & (sc != 0)])
    _, counts = np.unique(np.round(sc[fin], 12), return_counts=True)
    scale = float(np.median(nz)) if nz.size else 0.0
    diag = dict(n=int(sc.size), exact_zero_frac=float((sc == 0).mean()),
                nonfinite_frac=float((~fin).mean()),
                near_zero_frac=float((np.abs(sc) < 1e-10).mean()),
                n_distinct=int(counts.size),
                largest_tie_frac=float(counts.max() / sc.size) if counts.size else 1.0,
                effective_distinct_ranks=float((counts.sum() ** 2) / float((counts ** 2).sum())) if counts.size else 1.0,
                max_abs=float(np.abs(sc[fin]).max()) if fin.any() else 0.0,
                min_abs_nonzero=float(nz.min()) if nz.size else 0.0,
                median_abs_nonzero=scale)
    # machine-scale magnitudes with heavy ties = numerical degeneracy, not sparsity
    diag["OUTPUT_DEGENERATE"] = bool(diag["n_distinct"] < 0.25 * sc.size and scale < 1e-12)
    print("[rgv] output_diagnostics:", json.dumps(diag), flush=True)

    np.savez_compressed(a.out + ".npz", genes=genes, RegVelo=np.asarray(score, float))
    import hashlib
    def _sha(p_):
        h = hashlib.sha256()
        with open(p_, "rb") as fh:
            for b in iter(lambda: fh.read(1 << 24), b""): h.update(b)
        return h.hexdigest()
    arm = "PRIMARY_LOCKED" if primary else "ABLATION"
    rep = dict(script="gb_regvelo_noprior_gpu.py", script_sha256=_sha(os.path.abspath(__file__)),
               arm=arm, fixture_sha256=_sha(fx), npz_sha256=_sha(a.out + ".npz"), dataset=a.dataset, fixture=fx, tf=a.tf, seed=a.seed,
               config=f"{a.skeleton.upper()} skeleton, lam={a.lam:g}, soft_constraint=True",
               extraction=how, extraction_error=err, max_epochs=a.max_epochs, batch_size=a.batch_size,
               device="cuda" if torch.cuda.is_available() else "cpu", train_device=train_device,
               locked_config=LOCKED, model_saved=model_saved, catalogue=REGS, catalogue_sha256=_sha(REGS),
               model_manifest=model_manifest(a.out + "_model", _sha) if model_saved else [],
               cuda_device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               n_genes=int(len(genes)), n_regulators=len(reg_names), runtime_s=time.time() - t0,
               output_diagnostics=diag, lam=a.lam, skeleton=a.skeleton,
               regvelo_version=getattr(rgv, "__version__", "unknown"))
    with open(a.out + ".json", "w") as fh: json.dump(rep, fh, indent=1)
    print("[rgv]", json.dumps(rep), flush=True)


if __name__ == "__main__":
    main()
