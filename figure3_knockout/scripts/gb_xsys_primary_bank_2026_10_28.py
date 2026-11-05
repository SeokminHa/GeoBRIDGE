"""Own-universe PRIMARY bank (fixes internal review FIX_AND_REVIEW on gb_ery_phase1: alternatives were z-scored on the focal's U_obs).
For every source (focal + the SAME seeded-random alternatives as results/signed/<SYS>_bank.npz; CARDIACV: same selection rule as
gb_signed_bank.py, rng=default_rng(0)), raw |S| is computed with the frozen production algebra RU.score_gb_sources (U and X channels,
k=0..30), U_obs = CA.observational_universe (own source), PRIMARY = GP.zcons on own U_obs. No truth is read.
Parity gate: for every source also present in the production caches, raw |S| must equal the cached rows (else STOP).
usage: script SYSTEM OUTDIR [--nsub N] [--shard i --nshard n]   (N limits alternatives for the small-N check; shards = contiguous source blocks,
       written as primary_bank_<SYS>.shard<i>of<n>.npz and merged by scratch/gb_xsys_bank_merge_2026_10_28.py in the original roster order)"""
import sys, os, json, time
import numpy as np
R = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/')
sys.path[:0] = [R + 'scratch', R + 'pairspace']
import gb_real_uniform as RU, gb_cop_alignment as CA, gb_primary as GP
a = sys.argv; SYS = a[1].upper(); OUT = a[2].rstrip('/') + '/'; os.makedirs(OUT, exist_ok=True)
NSUB = int(a[a.index('--nsub') + 1]) if '--nsub' in a else None
FOCAL = {'CARDIACV': ['NKX2-5', 'TBX5']}
t0 = time.time()
bp = f'{R}results/signed/{SYS}_bank.npz'
if os.path.exists(bp):
    z = np.load(bp, allow_pickle=True); srcs = [s.upper() for s in z['sources'].astype(str)]; nfocal = int(z['nfocal'])
    _, eg = RU.gb_raw_rows(SYS, srcs=[srcs[0]])
else:   # replicate gb_signed_bank.py:14-17 selection exactly
    focal = FOCAL[SYS]; mem = json.load(open(R + 'results/signed/members.json'))[SYS]['members']
    _, eg = RU.gb_raw_rows(SYS, srcs=[focal[0]]); au = {str(g).upper() for g in eg}
    rng = np.random.default_rng(0); cand = [m for m in mem if m.upper() in au and m.upper() not in focal]
    sub = [cand[i] for i in rng.choice(len(cand), size=min(150, len(cand)), replace=False)]
    srcs = list(dict.fromkeys(focal + [m.upper() for m in sub])); nfocal = len(focal)
axis = [str(g) for g in eg]
if NSUB is not None: srcs = srcs[:nfocal + NSUB]
ROSTER = list(srcs); SH = (int(a[a.index('--shard') + 1]), int(a[a.index('--nshard') + 1])) if '--shard' in a else None
if SH is not None: srcs = [ROSTER[i] for i in np.array_split(np.arange(len(ROSTER)), SH[1])[SH[0]]]
RG = (int(a[a.index('--range') + 1]), int(a[a.index('--range') + 2])) if '--range' in a else None   # roster slice [lo, hi) for refilling dead shards
if RG is not None: srcs = ROSTER[RG[0]:RG[1]]
# production-cache sources for the parity gate
cached = set()
for p in RU.cache_paths(SYS):
    if p.exists():
        with np.load(p, allow_pickle=True) as c: cached |= set(np.char.upper(c['sources'].astype(str)))
G = len(axis); ns = len(srcs); Z = np.full((ns, G), np.nan); UOBS = np.zeros((ns, G), bool); par = []
BATCH = int(a[a.index("--batch") + 1]) if "--batch" in a else 30
for b0 in range(0, ns, BATCH):
    bs = srcs[b0:b0 + BATCH]
    raw, meta = RU.score_gb_sources(SYS, axis, bs)
    pc = [s for s in bs if s in cached]
    if pc:
        rows, eg2 = RU.gb_raw_rows(SYS, srcs=pc); assert [str(g) for g in eg2] == axis, 'axis mismatch vs cache'
        for s in pc:
            j = bs.index(s); mx = 0.0
            for key, arr in raw.items():
                c = np.asarray(rows[s][key], float); v = arr[j].astype(float)
                assert np.array_equal(np.isfinite(c), np.isfinite(v)), f'PARITY finite pattern {s} {key}'
                f = np.isfinite(c); mx = max(mx, float(np.abs(c[f] - v[f]).max()) if f.any() else 0.0)
            assert mx < 1e-5, f'PARITY FAIL {s} max {mx}'; par.append((s, mx))
    for j, s in enumerate(bs):
        rw = {key: arr[j] for key, arr in raw.items()}
        uni, _ = CA.observational_universe(s, rw, np.array(axis))
        i = b0 + j; UOBS[i] = uni
        Z[i, uni] = GP.zcons({h: rw[(h, 'U')][uni].astype(float) for h in GP.ROOTS})
    print(SYS, f'{b0 + len(bs)}/{ns}', f'{time.time() - t0:.0f}s', 'parity', par[-len(pc):] if pc else '-', flush=True)
np.savez_compressed(f'{OUT}primary_bank_{SYS}' + (f'.shard{SH[0]}of{SH[1]}' if SH else '') + (f'.range{RG[0]}_{RG[1]}' if RG else '') + '.npz', genes=np.array(axis), sources=np.array(srcs), roster=np.array(ROSTER), nfocal=nfocal, Z=Z.astype(np.float32), Uobs=UOBS,
                    parity=np.array([f'{s}:{m:.2e}' for s, m in par]))
print('DONE', SYS, ns, 'sources; parity-checked', len(par), [s for s, _ in par][:10], f'{time.time() - t0:.0f}s')
