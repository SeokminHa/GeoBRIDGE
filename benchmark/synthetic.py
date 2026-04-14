import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pandas as pd

TOPOLOGIES = ("LinearChain", "Bifurcation", "Trifurcation", "ManyBifurcations")
DRIFT_TYPES = ("none", "linear", "nonlinear", "branch_specific")

# Two (drift, regime) situations per topology = 8 datasets per rep (quick GeoBRIDGE + baselines).
TOPOLOGY_PROBE_SCENARIOS = (
    ("LinearChain", "none", "baseline"),
    ("LinearChain", "linear", "R3_pseudo_hubs"),
    ("Bifurcation", "none", "baseline"),
    ("Bifurcation", "linear", "R6_combined_real_like"),
    ("Trifurcation", "linear", "baseline"),
    ("Trifurcation", "branch_specific", "R4_noisy_pseudotime"),
    ("ManyBifurcations", "linear", "baseline"),
    ("ManyBifurcations", "nonlinear", "R6_combined_real_like"),
)

# Clip states/velocities so extreme drift + indirect coupling (Hard Mode) stay finite.
SIM_STATE_CLIP = 8_000.0
SIM_VELOCITY_CLIP = 8_000.0

def _stable_int(*parts):
    s = "||".join(parts).encode("utf-8")
    return int(hashlib.sha256(s).hexdigest()[:16], 16) % (2**32)

def ensure_dir(path):
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _module_ids(n_genes, n_modules):
    sz = n_genes // n_modules
    mods = np.repeat(np.arange(n_modules), sz)
    if mods.size < n_genes:
        mods = np.concatenate([mods, np.arange(n_genes - mods.size)])
    return mods[:n_genes]

def make_grn(n_genes, n_modules, edge_prob, weight, rng):
    mods = _module_ids(n_genes, n_modules)
    W = np.zeros((n_genes, n_genes), dtype=float)
    for i in range(n_genes):
        for j in range(n_genes):
            if i == j:
                continue
            if mods[i] == mods[j] and rng.random() < edge_prob:
                sgn = 1.0 if rng.random() < 0.5 else -1.0
                W[i, j] = sgn * weight * (0.5 + 0.5 * rng.random())
    return W

def _normalize(v):
    return v / (np.linalg.norm(v) + 1e-12)

def _branch_offset(name, n_genes, scale=0.15):
    return scale * _normalize(np.random.default_rng(_stable_int("branch_offset", name)).normal(size=n_genes))

def _segments_for_topology(topology):
    if topology == "LinearChain":
        return [("s1", None, 1.0), ("s2", "s1", 1.0), ("s3", "s2", 1.0), ("s4", "s3", 1.0)]
    if topology == "Bifurcation":
        return [("trunk", None, 1.6), ("left", "trunk", 1.2), ("right", "trunk", 1.2)]
    if topology == "Trifurcation":
        return [("trunk", None, 1.5), ("b1", "trunk", 1.1), ("b2", "trunk", 1.1), ("b3", "trunk", 1.1)]
    if topology == "ManyBifurcations":
        return [("a1", None, 0.9), ("a2", "a1", 0.9), ("a3", "a2", 0.9), ("a4", "a3", 0.9), ("a5", "a4", 0.9),
                ("b1", "a2", 0.8), ("b2", "b1", 0.8), ("b3", "b2", 0.8),
                ("c1", "b2", 0.7), ("c2", "c1", 0.7), ("c3", "c2", 0.7)]
    raise ValueError(topology)

def generate_base(
    topology,
    seed,
    n_genes=100,
    n_modules=5,
    edge_prob=0.05,
    weight=4.0,
    decay=2.0,
    dt=0.05,
    n_cells=1200,
    velocity_noise_std=0.6,
    indirect_strength=0.35,
    indirect_hops=2,
    *,
    state_noise_scale=0.10,
    init_scale=0.2,
):
    """
    Simulate baseline dynamics for one topology.

    ``seed`` (typically the replicate index ``rep``) seeds the GRN and other
    base draws. It is intentionally **not** derived from ``topology``: the same
    integer seed yields the same random GRN parameters for every topology,
    enabling controlled cross-topology comparisons on matched base networks
    (recorded as ``same_rep_shared_base_grn_across_topologies`` in manifests
    and per-dataset scenario metadata).
    """
    rng = np.random.default_rng(seed)
    W = make_grn(n_genes, n_modules, edge_prob, weight, rng)
    global_vec = _normalize(rng.normal(size=n_genes))
    nonlinear_vec = _normalize(rng.normal(size=n_genes))
    segments = _segments_for_topology(topology)
    names = [s[0] for s in segments]
    branch_offsets = {name: _branch_offset(name, n_genes, scale=0.15) for name in names}
    branch_vecs = {name: _normalize(np.random.default_rng(_stable_int("branch_vec", name)).normal(size=n_genes)) for name in names}

    # indirect operator
    A = W.copy()
    indirect_ops = []
    current = A.copy()
    for hop in range(2, indirect_hops + 1):
        current = current @ W
        indirect_ops.append(current.copy())

    init_by_seg, state_noise_by_seg, vel_noise_by_seg = {}, {}, {}
    total_steps = 0
    for name, _, duration in segments:
        steps = int(round(duration / dt)) + 1
        total_steps += steps
        init_by_seg[name] = rng.normal(scale=init_scale, size=n_genes)
        state_noise_by_seg[name] = rng.normal(scale=state_noise_scale, size=(steps - 1, n_genes))
        vel_noise_by_seg[name] = rng.normal(scale=velocity_noise_std, size=(steps, n_genes))
    sample_indices = rng.choice(total_steps, size=n_cells, replace=True)
    return {
        "W": W,
        "global_vec": global_vec,
        "nonlinear_vec": nonlinear_vec,
        "segments": segments,
        "branch_offsets": branch_offsets,
        "branch_vecs": branch_vecs,
        "init_by_seg": init_by_seg,
        "state_noise_by_seg": state_noise_by_seg,
        "vel_noise_by_seg": vel_noise_by_seg,
        "sample_indices": sample_indices,
        "dt": dt,
        "decay": decay,
        "indirect_strength": indirect_strength,
        "indirect_ops": indirect_ops,
    }

def _drift_and_dev(drift_type, drift_amp, alpha, global_vec, nonlinear_vec, branch_name, branch_vecs):
    if drift_type == "none":
        drift = np.zeros_like(global_vec)
    elif drift_type == "linear":
        drift = drift_amp * alpha * global_vec
    elif drift_type == "nonlinear":
        drift = drift_amp * (alpha**2 + 0.5 * np.sin(np.pi * alpha)) * nonlinear_vec
    elif drift_type == "branch_specific":
        drift = drift_amp * alpha * (0.55 * global_vec + 0.85 * branch_vecs[branch_name])
    else:
        raise ValueError(drift_type)
    return drift

def simulate_from_base(base, drift_type, drift_amp):
    W = base["W"]
    global_vec = base["global_vec"]
    nonlinear_vec = base["nonlinear_vec"]
    segments = base["segments"]
    branch_offsets = base["branch_offsets"]
    branch_vecs = base["branch_vecs"]
    dt = base["dt"]
    decay = base["decay"]
    indirect_strength = base["indirect_strength"]
    indirect_ops = base["indirect_ops"]

    X_parts, V_parts, D_parts, T_parts, P_parts = [], [], [], [], []
    seg_data = {}
    for name, parent, duration in segments:
        steps = int(round(duration / dt)) + 1
        alpha = np.linspace(0.0, 1.0, steps)
        x = np.zeros((steps, W.shape[0]), dtype=float)
        x[0] = base["init_by_seg"][name].copy() if parent is None else seg_data[parent]["x"][-1] + branch_offsets[name]
        v = np.zeros_like(x)
        d = np.zeros_like(x)
        state_noise = base["state_noise_by_seg"][name]
        vel_noise = base["vel_noise_by_seg"][name]
        branch_bias = branch_offsets[name]
        for t in range(steps - 1):
            drift = _drift_and_dev(drift_type, drift_amp, alpha[t], global_vec, nonlinear_vec, name, branch_vecs)
            d[t] = drift
            direct_reg = x[t] @ W
            indirect_reg = np.zeros_like(direct_reg)
            for op in indirect_ops:
                indirect_reg += x[t] @ op
            reg_for_state = direct_reg - decay * x[t]
            # state update
            x[t + 1] = x[t] + dt * (reg_for_state + drift) + branch_bias + state_noise[t]
            np.clip(x[t + 1], -SIM_STATE_CLIP, SIM_STATE_CLIP, out=x[t + 1])
            # dev(Y): direct + indirect + drift + noise
            v[t] = direct_reg + indirect_strength * indirect_reg + drift - decay * x[t] + vel_noise[t]
            np.clip(v[t], -SIM_VELOCITY_CLIP, SIM_VELOCITY_CLIP, out=v[t])
        drift_last = _drift_and_dev(drift_type, drift_amp, alpha[-1], global_vec, nonlinear_vec, name, branch_vecs)
        d[-1] = drift_last
        direct_last = x[-1] @ W
        indirect_last = np.zeros_like(direct_last)
        for op in indirect_ops:
            indirect_last += x[-1] @ op
        v[-1] = direct_last + indirect_strength * indirect_last + drift_last - decay * x[-1] + vel_noise[-1]
        np.clip(x[-1], -SIM_STATE_CLIP, SIM_STATE_CLIP, out=x[-1])
        np.clip(v[-1], -SIM_VELOCITY_CLIP, SIM_VELOCITY_CLIP, out=v[-1])

        seg_data[name] = {"x": x}
        X_parts.append(x); V_parts.append(v); D_parts.append(d); T_parts.append(alpha); P_parts.append(np.array([name] * steps, dtype=object))

    X_full, V_full, D_full = np.vstack(X_parts), np.vstack(V_parts), np.vstack(D_parts)
    T_full, P_full = np.concatenate(T_parts), np.concatenate(P_parts)
    idx = base["sample_indices"]
    X, V, D, time, path = X_full[idx], V_full[idx], D_full[idx], T_full[idx], P_full[idx]
    order = np.argsort(time)
    return {"X": X[order], "V": V[order], "D": D[order], "time": time[order], "path": path[order], "pseudotime": time[order], "W": W}

REALISM_REGIMES = (
    "baseline",
    "R1_weak_velocity",
    "R2_high_overlap",
    "R3_pseudo_hubs",
    "R4_noisy_pseudotime",
    "R5_nonstationary_noise",
    "R6_combined_real_like",
    "R7_active_tf_overlap",
    "R8_strong_multirank_confound",
    "R9_asymmetric_confound",
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", required=True)
    p.add_argument("--topologies", nargs="+", default=list(TOPOLOGIES))
    p.add_argument("--drift-types", nargs="+", default=list(DRIFT_TYPES))
    p.add_argument(
        "--regimes",
        nargs="+",
        default=["baseline"],
        help=(
            f"Realism / failure-mode regimes applied after base simulation. "
            f"Choices include: {', '.join(REALISM_REGIMES)}. "
            "Filenames become topology__drift__repK__regime.npz (+ _scenario_meta.json)."
        ),
    )
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--n-genes", type=int, default=100)
    p.add_argument("--n-cells", type=int, default=1200)
    p.add_argument("--edge-prob", type=float, default=0.05)
    p.add_argument("--weight", type=float, default=4.0)
    p.add_argument("--decay", type=float, default=2.0)
    p.add_argument("--dt", type=float, default=0.05)
    p.add_argument("--velocity-noise-std", type=float, default=0.6)
    p.add_argument("--drift-amp", type=float, default=10.0)
    p.add_argument("--indirect-strength", type=float, default=0.35)
    p.add_argument("--indirect-hops", type=int, default=2)
    p.add_argument(
        "--preset-realistic-benchmark",
        action="store_true",
        help=f"Shortcut: set --regimes to all non-baseline regimes ({len(REALISM_REGIMES)-1} regimes).",
    )
    p.add_argument(
        "--preset-probe-topologies",
        action="store_true",
        help=(
            "Generate only two (drift, regime) combos per topology (8 datasets per rep). "
            "Ignores --topologies, --drift-types, and --regimes. Intended for fast GeoBRIDGE checks."
        ),
    )
    return p.parse_args()

def main():
    args = parse_args()
    outdir = ensure_dir(args.outdir)
    if args.preset_probe_topologies and args.preset_realistic_benchmark:
        raise SystemExit("Choose only one of --preset-probe-topologies or --preset-realistic-benchmark.")
    if args.preset_realistic_benchmark:
        args.regimes = [r for r in REALISM_REGIMES if r != "baseline"]
    if not args.preset_probe_topologies:
        for r in args.regimes:
            if r not in REALISM_REGIMES:
                raise SystemExit(f"Unknown regime {r!r}. Choose from {REALISM_REGIMES}")
    else:
        for topo, drift, reg in TOPOLOGY_PROBE_SCENARIOS:
            if topo not in TOPOLOGIES:
                raise SystemExit(f"Invalid probe topology {topo!r}")
            if drift not in DRIFT_TYPES:
                raise SystemExit(f"Invalid probe drift {drift!r}")
            if reg not in REALISM_REGIMES:
                raise SystemExit(f"Invalid probe regime {reg!r}")

    from collections import defaultdict

    from . import realistic_regimes as rr
    from . import scenario_meta as sm

    if args.preset_probe_topologies:
        by_topo: dict[str, list[tuple[str, str]]] = defaultdict(list)
        for topo, drift, regime in TOPOLOGY_PROBE_SCENARIOS:
            by_topo[topo].append((drift, regime))
        n_topo = len(by_topo)
        n_probe_scenarios = len(TOPOLOGY_PROBE_SCENARIOS)
        total_datasets = n_probe_scenarios * args.reps
        t_gen0 = time.perf_counter()
        print(
            f"[GEN] Starting synthetic generation (preset-probe-topologies) -> {outdir.resolve()} | "
            f"topologies={n_topo} probe_scenarios={n_probe_scenarios} reps={args.reps} "
            f"total_datasets={total_datasets}"
        )
    else:
        n_topo = len(args.topologies)
        n_drift = len(args.drift_types)
        n_reg = len(args.regimes)
        total_datasets = n_topo * args.reps * n_drift * n_reg
        t_gen0 = time.perf_counter()
        print(
            f"[GEN] Starting synthetic generation -> {outdir.resolve()} | "
            f"topologies={n_topo} drift_types={n_drift} regimes={n_reg} reps={args.reps} "
            f"total_datasets={total_datasets}"
        )

    manifest = []
    scenario_rows = []
    regime_emit_i = 0

    def emit_from_sim(topo: str, drift: str, regime: str, rep: int, sim: dict) -> None:
        nonlocal regime_emit_i
        regime_emit_i += 1
        t_ds0 = time.perf_counter()
        stem_short = f"{topo}__{drift}__rep{rep}"
        rng = np.random.default_rng(_stable_int("regime", topo, drift, regime, str(rep)))
        sim_r, meta = rr.apply_regime(
            sim,
            regime,
            rng,
            drift_amp_proxy=float(args.drift_amp),
        )
        meta["topology"] = topo
        meta["drift_type"] = drift
        meta["rep"] = int(rep)
        meta["same_rep_shared_base_grn_across_topologies"] = True
        stem = f"{topo}__{drift}__rep{rep}__{regime}"
        fname = f"{stem}.npz"
        path = outdir / fname
        np.savez(path, **{k: sim_r[k] for k in sim_r if isinstance(sim_r[k], np.ndarray)})
        sm.save_scenario_meta(path, meta)
        manifest.append(
            {
                "file": fname,
                "topology": topo,
                "drift_type": drift,
                "rep": rep,
                "scenario_regime": regime,
            }
        )
        scenario_rows.append({"dataset_stem": stem, **sm.meta_to_flat_row(meta)})
        dt = time.perf_counter() - t_ds0
        print(
            f"[REGIME] {regime_emit_i}/{total_datasets} {stem_short} regime={regime} -> {fname} ({dt:.2f}s)"
        )

    if args.preset_probe_topologies:
        for topo in TOPOLOGIES:
            if topo not in by_topo:
                continue
            for rep in range(args.reps):
                t_base0 = time.perf_counter()
                base = generate_base(
                    topo, rep, args.n_genes, 5, args.edge_prob, args.weight, args.decay, args.dt,
                    args.n_cells, args.velocity_noise_std, args.indirect_strength, args.indirect_hops
                )
                print(
                    f"[SIM] Topology={topo} rep={rep} base simulation complete "
                    f"({time.perf_counter() - t_base0:.2f}s)"
                )
                for drift, regime in by_topo[topo]:
                    sim = simulate_from_base(base, drift, args.drift_amp)
                    emit_from_sim(topo, drift, regime, rep, sim)
    else:
        for topo in args.topologies:
            for rep in range(args.reps):
                t_base0 = time.perf_counter()
                base = generate_base(
                    topo, rep, args.n_genes, 5, args.edge_prob, args.weight, args.decay, args.dt,
                    args.n_cells, args.velocity_noise_std, args.indirect_strength, args.indirect_hops
                )
                print(
                    f"[SIM] Topology={topo} rep={rep} base simulation complete "
                    f"({time.perf_counter() - t_base0:.2f}s)"
                )
                for drift in args.drift_types:
                    sim = simulate_from_base(base, drift, args.drift_amp)
                    for regime in args.regimes:
                        emit_from_sim(topo, drift, regime, rep, sim)

    generation_wall_s = time.perf_counter() - t_gen0
    generation_config = {
        "same_rep_shared_base_grn_across_topologies": True,
        "same_rep_shared_base_grn_note": (
            "Integer rep is passed as generate_base(..., seed=rep, ...); seed does not depend on "
            "topology so the same rep index yields matched base GRN draws across topologies."
        ),
        "preset_probe_topologies": bool(args.preset_probe_topologies),
        "n_topologies": n_topo,
        "reps": int(args.reps),
        "total_datasets_written": len(manifest),
        "wall_seconds_total": round(generation_wall_s, 4),
    }
    if args.preset_probe_topologies:
        generation_config["probe_scenarios"] = [list(t) for t in TOPOLOGY_PROBE_SCENARIOS]
        generation_config["n_probe_scenarios"] = len(TOPOLOGY_PROBE_SCENARIOS)
        generation_config["topologies"] = [t for t in TOPOLOGIES if t in by_topo]
        generation_config["drift_types"] = "probe (see probe_scenarios)"
        generation_config["regimes"] = "probe (see probe_scenarios)"
    else:
        generation_config["n_drift_types"] = len(args.drift_types)
        generation_config["n_regimes"] = len(args.regimes)
        generation_config["topologies"] = list(args.topologies)
        generation_config["drift_types"] = list(args.drift_types)
        generation_config["regimes"] = list(args.regimes)
    manifest_doc = {"generation_config": generation_config, "datasets": manifest}
    (outdir / "manifest.json").write_text(json.dumps(manifest_doc, indent=2), encoding="utf-8")
    pd.DataFrame(scenario_rows).to_csv(outdir / "scenario_metadata.csv", index=False)

    summary_lines = [
        "Synthetic benchmark — generation summary",
        "========================================",
        "",
        "These datasets feed the regime-based **realism-stress** benchmark "
        "(`python -m benchmark.run_benchmark`), not the Fig2 **structural X→V misspecification** "
        "track (`benchmark.fig2_benchmark_paper`). After a run, see `tables/benchmark_roles.md` under `--outdir`.",
        "",
        f"Output directory: {outdir.resolve()}",
        f"Total datasets (.npz): {len(manifest)}",
        f"Number of topologies: {n_topo}",
    ]
    if args.preset_probe_topologies:
        summary_lines.append(
            f"Mode: preset-probe-topologies ({n_probe_scenarios} drift×regime scenarios, 2 per topology)"
        )
    else:
        summary_lines.extend(
            [
                f"Number of drift types: {len(args.drift_types)}",
                f"Number of regimes: {len(args.regimes)}",
            ]
        )
    summary_lines.extend(
        [
            f"Reps: {args.reps}",
            "",
            "Files written:",
            "  - manifest.json (includes generation_config + datasets list)",
            "  - scenario_metadata.csv",
            "  - generation_summary.txt (this file)",
            f"  - {len(manifest)} × *.npz",
            f"  - {len(manifest)} × *_scenario_meta.json",
            "",
            f"same_rep_shared_base_grn_across_topologies: {generation_config['same_rep_shared_base_grn_across_topologies']}",
            generation_config["same_rep_shared_base_grn_note"],
            "",
            f"Total wall time (generation): {generation_wall_s:.2f}s",
        ]
    )
    summary_path = outdir / "generation_summary.txt"
    summary_path.write_text("\n".join(summary_lines) + "\n", encoding="utf-8")

    print(f"[GEN] Wrote manifest.json, scenario_metadata.csv, {summary_path.name}")
    if args.preset_probe_topologies:
        print(
            f"[GEN] Summary: datasets={len(manifest)} preset-probe-topologies "
            f"scenarios={n_probe_scenarios} reps={args.reps} | per-dataset: *.npz + *_scenario_meta.json"
        )
    else:
        print(
            f"[GEN] Summary: datasets={len(manifest)} topologies={n_topo} "
            f"drift_types={len(args.drift_types)} regimes={len(args.regimes)} reps={args.reps} | "
            f"per-dataset: *.npz + *_scenario_meta.json"
        )
    print(
        f"[GEN] Done. Output dir {outdir.resolve()} "
        f"(total wall {generation_wall_s:.2f}s)"
    )

if __name__ == "__main__":
    main()
