"""GRAND SIGNED BENCH v3-extsign (copy of v2 + unsigned externals x shared d; nulls skipped). v2: (EVALUATION ONLY; truth is used only to evaluate saved score vectors; no best-row selection in code).  Spec: internal design note (not shipped).
Derived from gb_grand_signed_bench.py (pinned, untouched).  Changes vs v1: system-aware FAIL-CLOSED d4/d1/Z/Pearson/T_A parity; sha256 ledger enforced at every load; bootstrap CIs over resamples only (point estimate excluded);
ONE common-support universe per endpoint for ALL rows (GB + unsigned/signed externals); self gene NaN in every row; signed DeepSEM/RegDiffusion (if signsem.json says include);
scrambled-target null; matched non-TF pseudo-source check; system-level shared-gene-bootstrap arrays.
usage: gb_grand_v2.py ko   OUTDIR B ENDPOINT[,..] [--smoke] [--nnull=200] [--pseudo=PSDIR] [--drop-ext=NAME;NAME]
       gb_grand_v2.py chip OUTDIR B UNIT[,..]     [--smoke] [--drop-ext=...]"""
import sys, os, json, hashlib, zlib, numpy as np, pandas as pd
from pathlib import Path
from scipy.stats import rankdata, spearmanr
sys.path[:0] = [(__import__("os").environ["GEOBRIDGE_ROOT"] + '/scratch'), (__import__("os").environ["GEOBRIDGE_ROOT"] + '/pairspace')]
import gb_fig3d_magnitude as F, gb_primary as GP, gb_real_uniform as RU
from gb_fig3d_magnitude import up, BIOSYS, VEC, effect, METH
from gb_fig3d_magnitude_v2 import inv_of, wrank
R0 = (__import__("os").environ["GEOBRIDGE_ROOT"] + '/'); SG = R0 + 'results/signed/'; MIN = SG + 'minimal_2026_10_14/'; S1 = SG + 'grand_signed_bench_2026_10_15/stage1/'
TA = SG + 'TA_mainfig_2026_10_14/'; FLIP = {'GATA1': -1.0}; ROOTS = GP.ROOTS
MODE, OUT, B = sys.argv[1], Path(sys.argv[2]).resolve(), int(sys.argv[3]); ITEMS = sys.argv[4].split(','); FL = sys.argv[5:]
SMOKE = '--smoke' in FL; NNULL = int(([f for f in FL if f.startswith('--nnull=')] or ['--nnull=200'])[0].split('=')[1])
PSD = ([f.split('=', 1)[1] for f in FL if f.startswith('--pseudo=')] or [None])[0]; DROPX = set(([f.split('=', 1)[1] for f in FL if f.startswith('--drop-ext=')] or [''])[0].split(';')) - {''}
PARTS = OUT / 'parts'; PARTS.mkdir(parents=True, exist_ok=True)
sg = np.sign
MROWS = ['Z', 'Z_M', 'Z_V', 'Z_rawU', 'Z_R2U', 'R0_absS0', 'absPear', 'absPearMu']
DROWS = ['d1', 'd4', 'd_M', 'd_V', 'd_rawU', 'd_R2k012', 'd_Pear', 'd_PearMu']
UNS_EXT = [m for m in METH if m not in ('GeoBRIDGE',)]
SMOKE_ROWS = [('Z', 'd1'), ('Z', 'd4'), ('Z_V', 'd1')]
PRE = [('Z', 'd1'), ('Z', 'd4'), ('Z', 'd_Pear'), ('absPear', 'd_Pear'), ('absPearMu', 'd_PearMu'), ('R0_absS0', 'd1')]      # PRE-NAMED control rows (+ every signed external row)
PSEUDO_EP = {'GATA1:GATA1', 'KLF1:KLF1', 'CARDIACV:TBX5', 'CARDIACV:NKX2-5', 'PANC:RFX6'}
TAG = ITEMS[0].replace(':', '_')
LOG = open(PARTS / f'parity_{MODE}_{TAG}.log', 'w')
def P(*a):
    s = ' '.join(str(x) for x in a); print(s, flush=True); LOG.write(s + '\n'); LOG.flush()
def need(c, msg):
    if not c: P('PARITY_FAIL', msg); raise SystemExit('PARITY_FAIL ' + msg)
# ---------------------------------------------------------------- provenance ledger: no file is loaded unless its sha256 equals the pinned ledger entry
LED = dict(pd.read_csv(OUT / 'provenance.tsv', sep='\t')[['path', 'sha256']].values); _OK = {}
def chk(p):
    p = str(p)
    if p not in _OK:
        need(p in LED, f'PROVENANCE_MISSING {p}'); h = hashlib.sha256()
        with open(p, 'rb') as f:
            for b in iter(lambda: f.read(1 << 24), b''): h.update(b)
        need(h.hexdigest() == LED[p], f'PROVENANCE_MISMATCH {p}'); _OK[p] = h.hexdigest()
    return p
def ld(p): return np.load(chk(p), allow_pickle=True)
def rtsv(p, **k): return pd.read_csv(chk(p), sep='\t', **k)
SIGN = json.load(open(OUT / 'signsem.json')); P('signsem decisions', SIGN)

# ---------------------------------------------------------------- metrics (copied algebra: gb_TA_mainfig_report.auc/apl/jt_z/cliff)
def auc(iv, pos, neg, w):
    we = w * (pos | neg); Wp, Wn = (w * pos).sum(), (w * neg).sum()
    if Wp < 1 or Wn < 1: return np.nan
    r = wrank(iv, we); return float(((w * pos * r).sum() - Wp * (Wp + 1) / 2) / (Wp * Wn))
def apl(iv, pos, neg, w):
    i, ng = iv; we = w * (pos | neg); Wp = (w * pos).sum()
    if Wp < 1 or (w * neg).sum() < 1: return np.nan, np.nan
    gp = np.bincount(i, w * pos, ng)[::-1]; gt = np.bincount(i, we, ng)[::-1]; cp, ct = np.cumsum(gp), np.cumsum(gt)
    ok = gt > 0; ap = float((gp[ok] * cp[ok] / ct[ok]).sum() / Wp); return ap, ap / float(Wp / we.sum())
def jt_z(groups):
    ns = [len(x) for x in groups]; N = sum(ns); allv = np.concatenate(groups); J = 0.0
    for i in range(len(groups)):
        for j in range(i + 1, len(groups)):
            r = rankdata(np.concatenate([groups[i], groups[j]])); J += r[:ns[i]].sum() - ns[i] * (ns[i] + 1) / 2
    mu = (N * N - sum(n * n for n in ns)) / 4.0
    t = np.unique(allv, return_counts=True)[1].astype(float); n_ = np.array(ns, float)
    v = (N * (N - 1) * (2 * N + 5) - (n_ * (n_ - 1) * (2 * n_ + 5)).sum() - (t * (t - 1) * (2 * t + 5)).sum()) / 72.0
    v += (n_ * (n_ - 1) * (n_ - 2)).sum() * (t * (t - 1) * (t - 2)).sum() / (36.0 * N * (N - 1) * (N - 2)) + (n_ * (n_ - 1)).sum() * (t * (t - 1)).sum() / (8.0 * N * (N - 1))
    return (J - mu) / np.sqrt(v)
def cliff(x, y, mn=5):
    x, y = x[np.isfinite(x)], y[np.isfinite(y)]
    if len(x) < mn or len(y) < mn: return np.nan
    r = rankdata(np.concatenate([x, y])); return 2 * ((r[:len(x)].sum() - len(x) * (len(x) + 1) / 2) / (len(x) * len(y))) - 1
def ci(v):
    """percentile CI over RESAMPLES ONLY (fix 3): callers pass the full (point + B resamples) array of a KO metric; index 0 is the point estimate and is dropped here.  chip callers pass resamples only (use ci_r)."""
    return ci_r(np.asarray(v, float)[1:])
def ci_r(v): v = np.asarray(v, float); return tuple(np.nanpercentile(v, [2.5, 97.5])) if np.isfinite(v).sum() > 20 else (np.nan, np.nan)
def frac_rank(z, m):
    r = np.full(len(z), np.nan); r[m] = (rankdata(z[m]) - 0.5) / m.sum(); return r

# ---------------------------------------------------------------- stage-1 loading with shape / roster assertions
def get_uni(bsy, tf):
    sc = ld(f'{SG}{bsy}.npz')
    if tf.upper() not in [x.upper() for x in sc['sources'].astype(str)]: sc = ld(f'{SG}{bsy}_bank.npz')
    j = [s.upper() for s in sc['sources'].astype(str)].index(tf.upper()); uni = np.asarray(sc['uni'][j]); axis = np.char.upper(sc['genes'].astype(str))
    if not (uni.dtype == bool or (uni.dtype == object and isinstance(uni.flat[0], (bool, np.bool_)))):
        m = np.zeros(len(axis), bool); m[uni.astype(int)] = True; uni = m
    return uni.astype(bool), axis, sc, j
def load_s1(bsy, tf):
    out = {}; ref = None
    for r in ROOTS:
        z1 = ld(f'{S1}{bsy}__{r}__R1.npz'); z2 = ld(f'{S1}{bsy}__{r}__R2.npz')
        srcs = [s.upper() for s in z1['sources'].astype(str)]; need([s.upper() for s in z2['sources'].astype(str)] == srcs, f'stage1 R1/R2 source roster differ {bsy} {r}'); need(tf.upper() in srcs, f'{tf} not in stage1 roster {bsy}')
        G = len(z1['genes']); n = len(srcs)
        for c in ('U', 'M', 'rawU', 'V'): need(z1[c].shape == (n, G, 9), f'stage1 {c} shape {z1[c].shape} != {(n, G, 9)}')
        need(z2['U'].shape == (n, G, 9) and z1['Sc0'].shape == (n, G) and z1['pear_MsMs'].shape == (n, G) and list(z1['ks']) == list(range(9)), 'stage1 shapes/ks')
        if ref is None: ref = (srcs, [g for g in z1['genes'].astype(str)])
        need(srcs == ref[0] and list(z1['genes'].astype(str)) == ref[1], 'stage1 roster/axis differ between roots')
        j = srcs.index(tf.upper())
        out[r] = dict(U=z1['U'][j], M=z1['M'][j], rawU=z1['rawU'][j], V=z1['V'][j], Sc0=z1['Sc0'][j], R2U=z2['U'][j], pear=z1['pear_MsMs'][j], pearMu=z1['pear_MsMu'][j], axis=np.char.upper(z1['genes'].astype(str)), j=j)
    P(bsy, 'stage1 roster', ref[0], 'axis', len(ref[1]))
    return out

def axis_rows(bsy, tf):
    """GB magnitude/polarity rows over the system gene AXIS (self = NaN in EVERY row)."""
    uni, axis, sc, jc = get_uni(bsy, tf); s1 = load_s1(bsy, tf); need((s1[ROOTS[0]]['axis'] == axis).all(), 'stage1 axis != production cache axis')
    ai = {x: i for i, x in enumerate(axis)}; tfi = ai[tf.upper()]; G = len(axis); ui = np.flatnonzero(uni)
    def zc(chan, ks):
        o = np.full(G, np.nan); o[ui] = GP.zcons({h: np.abs(s1[h][chan][ui].astype(float)) for h in ROOTS}, ks); o[tfi] = np.nan; return o
    Mr = {'Z': zc('U', GP.KS), 'Z_M': zc('M', GP.KS), 'Z_V': zc('V', GP.KS), 'Z_rawU': zc('rawU', GP.KS), 'Z_R2U': zc('R2U', GP.KS), 'R0_absS0': zc('U', (0,))}
    pr = s1[ROOTS[0]]['pear'].astype(float); pm = s1[ROOTS[0]]['pearMu'].astype(float); pr[tfi] = np.nan; pm[tfi] = np.nan
    Mr['absPear'] = np.abs(np.where(uni, pr, np.nan)); Mr['absPearMu'] = np.abs(np.where(uni, pm, np.nan))
    def k0(chan):
        a = [np.where(uni, s1[h][chan][:, 0].astype(float), np.nan) for h in ROOTS]; m = np.isfinite(a[0]) & np.isfinite(a[1]); o = np.where(m, (a[0] + a[1]) / 2, np.nan); o[tfi] = np.nan; return o
    mean_roots = lambda aa: np.where(np.isfinite(aa[0]) & np.isfinite(aa[1]), (aa[0] + aa[1]) / 2, np.nan)
    sc0 = [np.where(uni, s1[h]['Sc0'].astype(float), np.nan) for h in ROOTS]; r2 = []
    for h in ROOTS:
        x = s1[h]['R2U'][:, :3].astype(float); r2.append(np.where(uni & np.isfinite(x).all(1), x.mean(1), np.nan))
    cont = {'d1': k0('U'), 'd4': mean_roots(sc0), 'd_M': k0('M'), 'd_V': k0('V'), 'd_rawU': k0('rawU'), 'd_R2k012': mean_roots(r2), 'd_Pear': np.where(uni, pr, np.nan), 'd_PearMu': np.where(uni, pm, np.nan)}
    for k in ('d4', 'd_R2k012'): cont[k][tfi] = np.nan
    D = {k: np.where(np.isfinite(v), sg(v), np.nan) for k, v in cont.items()}
    return Mr, D, cont, dict(axis=axis, uni=uni, tfi=tfi, s1=s1, sc=sc, jc=jc)

def mapper(axis, genes):
    ai = {x: i for i, x in enumerate(axis)}; idx = np.array([ai.get(x, -1) for x in genes]); has = idx >= 0
    def mp(full): o = np.full(len(genes), np.nan); o[has] = np.asarray(full, float)[idx[has]]; return o
    return mp

# ---------------------------------------------------------------- system-aware d4 parity file (fix 1)
D4G1 = {('GATA1', 'GATA1'), ('GATA1', 'KLF1'), ('CARDIACV', 'TBX5'), ('CARDIACV', 'NKX2-5'), ('PANC', 'RFX6')}      # gb_d1d4 EPS_ (system, source) pairs scored in geom_phase1_2026_10_14
def d4_parity(bsy, tf, axis, cont_d4, name):
    if (bsy, tf.upper()) in D4G1: f = f'{SG}geom_phase1_2026_10_14/d1_d4_scores_{tf.upper()}.tsv'; src = 'geom_phase1_2026_10_14'
    else: f = f'{(OUT / "d4par").resolve()}/{bsy}_{tf.upper()}/d1_d4_scores_{tf.upper()}.tsv'; src = 'v2 independent gb_d1d4.py run'
    need(os.path.exists(f), f'NO independent d4 file for system={bsy} source={tf}: {f}')
    T4 = rtsv(f); a4 = up(T4.gene); need(len(a4) == len(axis) and (a4 == axis).all(), f'd4 file axis != stage1 axis for {bsy}:{tf}')
    x4 = T4.Sc0_mean.to_numpy(float); m = np.isfinite(x4) & np.isfinite(cont_d4); pat = int((np.isfinite(x4) != np.isfinite(cont_d4)).sum()); sm = int((sg(x4[m]) != sg(cont_d4[m])).sum()); md = float(np.abs(x4[m] - cont_d4[m]).max()) if m.any() else np.nan
    P(name, 'PARITY d4 (system-aware: independent', src, bsy, tf, ') stage1 Sc0 mean vs gb_d1d4 Sc0_mean over AXIS: n', int(m.sum()), 'max|diff|', md, 'sign mismatches', sm, 'finite-pattern mismatches', pat)
    need(m.sum() > 0 and md <= 1e-6 and sm == 0 and pat == 0, f'd4 parity failed {bsy}:{tf} max|diff|={md} sign={sm} pattern={pat}')

# ---------------------------------------------------------------- signed external vectors for DeepSEM / RegDiffusion (fix 5)
MISSING = []
def sig_ext(bsy, tf, genes):
    out = {}
    for lab, fn in (('DeepSEM', 'deepsem_v1.npz'), ('RegDiffusion', 'regdiffusion_v1.npz')):
        if not SIGN.get(lab, '').startswith('include'): continue
        z = ld(RU.system_dir(bsy) / fn); tfs = [x.upper() for x in z['tfs'].astype(str)]
        if tf.upper() not in tfs: MISSING.append((bsy, tf, lab + '(signed)', 'TF not in method regulator roster')); continue
        row = np.asarray(z['score_all_signed'], float)[tfs.index(tf.upper())]; gi = {x.upper(): i for i, x in enumerate(z['genes'].astype(str))}
        out[lab + '(signed)'] = np.array([row[gi[x]] if x in gi else np.nan for x in genes])
    return out

# ---------------------------------------------------------------- KO
def bootstrap_weights(classes, B, seed=0):
    rng = np.random.default_rng(seed); n = len(classes); W = np.zeros((B + 1, n)); W[0] = 1
    idxs = [np.flatnonzero(classes == c) for c in np.unique(classes)]
    for b in range(1, B + 1):
        for ix in idxs: W[b] += np.bincount(rng.choice(ix, len(ix)), minlength=n)
    return W
def shared_weights(bsy, axis, gidx, B):
    """system-shared gene bootstrap: counts drawn ONCE per (system, b) over the system AXIS and read by gene identity, so every endpoint of a system sees the same resampled genes"""
    G = len(axis); W = np.ones((B + 1, len(gidx)))
    for b in range(1, B + 1): W[b] = np.random.default_rng([zlib.crc32(bsy.encode()), b]).multinomial(G, np.full(G, 1.0 / G))[gidx]
    return W

def ko_eval_row(Fv, dv, s, down, up_, nul, yy, W, resp_only=False, with_resp=True):
    o = {k: [] for k in ('resp_AUROC', 'resp_APlift', 'act_AUPRC', 'act_APlift', 'rep_AUPRC', 'rep_APlift', 'BA', 'rec_down', 'rec_up')}
    ivm = inv_of(Fv) if with_resp else None
    if not resp_only: ivp, ivn = inv_of(s), inv_of(-s); ad, au_ = ~down, ~up_
    for w in W:
        if with_resp: o['resp_AUROC'].append(auc(ivm, yy, nul, w)); o['resp_APlift'].append(apl(ivm, yy, nul, w)[1])
        if resp_only: continue
        a, l = apl(ivp, down, ad, w); o['act_AUPRC'].append(a); o['act_APlift'].append(l)
        a, l = apl(ivn, up_, au_, w); o['rep_AUPRC'].append(a); o['rep_APlift'].append(l)
        called = dv != 0; Wd, Wu = (w * down).sum(), (w * up_).sum()
        rd = (w * down * called * (s > 0)).sum() / Wd if Wd > 0 else np.nan; ru = (w * up_ * called * (s < 0)).sum() / Wu if Wu > 0 else np.nan
        o['rec_down'].append(rd); o['rec_up'].append(ru); o['BA'].append(np.mean([rd, ru]) if Wd > 0 and Wu > 0 else np.nan)
    return {k: np.array(v, float) for k, v in o.items() if len(v)}

def run_ko(name):
    sysn, tf = name.split(':'); sub, tr = [(e[1], e[3]) for e in F.EPS if e[0] == sysn and e[2] == tf][0]; bsy = BIOSYS[sysn]
    z = ld(f'{VEC}ep_{sysn}__{tf}__{tr}.vec.npz'); g = up(z['genes']); Mm = z['MR'].astype(bool); y = z['y'].astype(bool); eff = FLIP.get(sysn, 1.0) * effect(sysn, tf)
    d0 = ld(f'{MIN}scores_{sysn}__{tf}.npz'); need((up(d0['genes']) == g).all(), 'minimal genes != vec genes')
    bs = rtsv(F.BS, usecols=['system', 'method', 'mode', 'gene', 'raw_det', 'abundance']); bsg = bs[(bs.method == 'GeoBRIDGE') & (bs['mode'] == 'P')]
    b = bsg[bsg.system == bsy]; b = b.assign(G=up(b.gene)).drop_duplicates('G', keep=False).set_index('G')
    Mr_ax, D_ax, cont_ax, info = axis_rows(bsy, tf); mp = mapper(info['axis'], g)
    Mr = {k: mp(v) for k, v in Mr_ax.items()}; Dd = {k: mp(v) for k, v in D_ax.items()}; Cd = {k: mp(v) for k, v in cont_ax.items()}
    dup = pd.Series(g).duplicated(keep=False).to_numpy(); Zst = z['cog::GeoBRIDGE'].astype(float); base_prod = Mm & ~dup; selfm = (g == tf.upper())
    # ---------- fail-closed parity
    mz = np.isfinite(Zst) & np.isfinite(Mr['Z']); zd = float(np.abs(Zst[mz] - Mr['Z'][mz]).max()); zp = int((np.isfinite(Zst) != np.isfinite(Mr['Z']))[base_prod].sum())
    P(name, 'PARITY Z recomputed(stage1,zcons) vs stored vec Z: n', int(mz.sum()), 'max|diff|', zd, 'nan-pattern mismatch', zp); need(zd <= 1e-6 and zp == 0, 'Z parity')
    S0m = (d0['S0a'] + d0['S0b']) / 2; m1 = np.isfinite(S0m) & np.isfinite(Cd['d1']); d1d = float(np.abs(S0m[m1] - Cd['d1'][m1]).max()); d1s = int((sg(S0m[m1]) != sg(Cd['d1'][m1])).sum()); d1p = int((np.isfinite(S0m) != np.isfinite(Cd['d1']))[base_prod].sum())
    P(name, 'PARITY d1 stage1 S0 vs minimal S0(mean roots): n', int(m1.sum()), 'max|diff|', d1d, 'sign mismatches', d1s, 'finite-pattern mismatch', d1p); need(d1d <= 1e-6 and d1s == 0 and d1p == 0, 'd1 parity')
    mp_ = np.isfinite(d0['pear']) & np.isfinite(Cd['d_PearMu']); pd_ = float(np.abs(d0['pear'][mp_] - Cd['d_PearMu'][mp_]).max()); P(name, 'PARITY pear(Ms-Mu) minimal vs stage1: n', int(mp_.sum()), 'max|diff|', pd_); need(pd_ <= 1e-6, 'pearMu parity')
    pe = ld(f'{SG}{bsy}_pearson.npz'); pg = np.char.upper(pe['genes'].astype(str)); pi = {x: i for i, x in enumerate(pg)}
    r_ = pe['r'][[s.upper() for s in pe['sources'].astype(str)].index(tf.upper())]; Pr = np.array([r_[pi[x]] if x in pi else np.nan for x in g])
    mq = np.isfinite(Pr) & np.isfinite(Cd['d_Pear']); pq = float(np.abs(Pr[mq] - Cd['d_Pear'][mq]).max()); P(name, 'PARITY pearson npz r vs stage1 pear_MsMs: n', int(mq.sum()), 'max|diff|', pq); need(pq <= 1e-6, 'pearson parity')
    mq = np.isfinite(z['cog::|Pearson|'].astype(float)) & np.isfinite(Mr['absPear']); sp_ = float(spearmanr(z['cog::|Pearson|'].astype(float)[mq], Mr['absPear'][mq])[0]); P(name, 'PARITY vec |Pearson| vs |stage1 pear_MsMs| spearman', sp_); need(sp_ > 1 - 1e-9, '|Pearson| spearman')
    d4_parity(bsy, tf, info['axis'], cont_ax['d4'], name)
    # ---------- external scores (self NaN everywhere)
    nanself = lambda v: np.where(selfm, np.nan, np.asarray(v, float))
    ext_un = {m: nanself(z[f'cog::{m}'].astype(float)) for m in UNS_EXT}; ext_sg = {'Pearson(signed)': nanself(Pr)}
    zr = ld(RU.system_dir(bsy) / 'regvelo_v2_full.npz'); rg = {x.upper(): i for i, x in enumerate(zr['genes'].astype(str))}
    rr = np.asarray(zr['score_all_signed'], float)[[s.upper() for s in zr['regulators'].astype(str)].index(tf.upper())]; ext_sg['RegVelo(signed J)'] = nanself([rr[rg[x]] if x in rg else np.nan for x in g])
    for k, v in sig_ext(bsy, tf, g).items(): ext_sg[k] = nanself(v)
    for m_ in list(MISSING): P(name, 'MISSING row (reported missing, not zero):', m_)
    # ---------- ONE common-support universe for ALL rows (fix 4)
    e_ = pd.Series(g).map(eff).to_numpy(float); det = pd.Series(g).map(b['raw_det']).to_numpy(float); ab = pd.Series(g).map(b['abundance']).to_numpy(float)
    prodset = d0['prodset']; S0a, S0b = d0['S0a'], d0['S0b']
    req = [d0['Z'], d0['F2'], S0m, (d0['C2na'] + d0['C2nb']) / 2, (d0['C8na'] + d0['C8nb']) / 2, (d0['C2a'] + d0['C2b']) / 2, (d0['C8a'] + d0['C8b']) / 2, d0['pear'], d0['dpt_sign'], (S0a + S0b) / (np.abs(S0a) + np.abs(S0b))]
    if bool(np.isfinite(d0['tcommon']).any()): req.append(d0['tcommon'])
    req += [d0[k] for k in ('S0a', 'S0b', 'C2na', 'C2nb', 'C8na', 'C8nb')]
    okb = Mm & ~dup & prodset & np.isfinite(det) & np.isfinite(ab) & (~y | np.isfinite(e_)) & ~selfm
    for v in req: okb &= np.isfinite(v)
    Mr['Z'] = Zst.copy(); Mr = {k: nanself(v) for k, v in Mr.items()}
    Fm = {k: frac_rank(v, Mm & ~dup & np.isfinite(v)) for k, v in Mr.items()}
    def point(okmask, Fv, dv):
        yy = y[okmask]; ee = np.where(np.isfinite(e_[okmask]), e_[okmask], 0.0); dn, upm, nl = yy & (ee < 0), yy & (ee > 0), ~yy
        s = dv[okmask] * Fv[okmask]; o = ko_eval_row(Fv[okmask], dv[okmask], s, dn, upm, nl, yy, np.ones((1, int(okmask.sum()))))
        return {k: float(v[0]) for k, v in o.items()}, int(okmask.sum())
    pt, n_b = point(okb & np.isfinite(Fm['Z']) & np.isfinite(Dd['d1']), Fm['Z'], Dd['d1']); refa = rtsv(TA + 'ko.tsv'); ref = refa[(refa.endpoint == name) & (refa.score == 'T_A')].set_index('metric').est; nref = int(refa.query('endpoint==@name').n_genes.iloc[0])
    pairs = [('act_APlift', 'act_APlift'), ('rep_APlift', 'rep_APlift'), ('BA', 'dir_BA_zero_thr_amb_miss'), ('act_AUPRC', 'act_AUPRC'), ('rep_AUPRC', 'rep_AUPRC')]
    P(name, 'PARITY T_A (Z,d1) vs TA_mainfig ko.tsv: n_mine', n_b, 'n_ref', nref, '|', ' '.join(f'{a} {pt[a]:.6f}/{float(ref[c]):.6f}' for a, c in pairs)); need(n_b == nref and all(abs(pt[a] - float(ref[c])) <= 1e-6 for a, c in pairs), 'T_A parity')
    rows = SMOKE_ROWS if SMOKE else [(m, d) for m in MROWS for d in DROWS]; mneed = sorted({m for m, _ in rows}); dneed = sorted({d for _, d in rows})
    ok = okb.copy(); cover = {}
    for m_ in mneed: cover[m_] = int((okb & np.isfinite(Fm[m_])).sum()); ok &= np.isfinite(Fm[m_])
    for d_ in dneed: cover[d_] = int((okb & np.isfinite(Dd[d_])).sum()); ok &= np.isfinite(Dd[d_])
    extall = {} if SMOKE else {**{'ext|' + k: v for k, v in ext_un.items()}, **{'ext|' + k: v for k, v in ext_sg.items()}}
    for k, v in extall.items():
        cover[k] = int((okb & np.isfinite(v)).sum())
        if k[4:] in DROPX: P(name, 'universe: EXCLUDED from common-support definition (--drop-ext):', k); continue
        ok &= np.isfinite(v)
    n = int(ok.sum()); yy = y[ok]; ee = np.where(np.isfinite(e_[ok]), e_[ok], 0.0); down, up_, nul = yy & (ee < 0), yy & (ee > 0), ~yy
    cls = np.where(down, 0, np.where(up_, 1, 2)); W = bootstrap_weights(cls, B, 0)
    P(name, 'n_base(T_A universe)', int(okb.sum()), 'n_common(ALL rows incl externals)', n, 'resp', int(yy.sum()), 'down', int(down.sum()), 'up', int(up_.sum()), 'nul', int(nul.sum()), 'coverage_before_intersection', cover)
    meta = dict(endpoint=name, n_common=n, n_base=int(okb.sum()), n_resp=int(yy.sum()), n_KOdown=int(down.sum()), n_KOup=int(up_.sum()), n_null=int(nul.sum()))
    axis = info['axis']; aix = {x: i for i, x in enumerate(axis)}; gidx = np.array([aix[x] for x in g[ok]]); Wsys = shared_weights(bsy, axis, gidx, B)
    OUTR = []; base = None; SYSARR = {}; NULLSPEC = {}
    def add(rowid, M_, d_, kind, o, cov, namb):
        nonlocal base
        r = dict(**meta, row=rowid, M=M_, d=d_, kind=kind, n_coverage_before=cov, n_ambiguous=namb)
        for k, v in o.items(): r[k] = v[0]; r[k + '_lo'], r[k + '_hi'] = ci(v)
        if base is not None and kind in ('GB', 'ext_signed'):
            for k in ('act_APlift', 'rep_APlift', 'act_AUPRC', 'rep_AUPRC', 'BA'):
                if k in o and k in base: dv = o[k] - base[k]; r['d_' + k] = dv[0]; r['d_' + k + '_lo'], r['d_' + k + '_hi'] = ci(dv)
        if base is not None and 'resp_AUROC' in o and 'resp_AUROC' in base:
            for k in ('resp_AUROC', 'resp_APlift'): dv = o[k] - base[k]; r['d_' + k] = dv[0]; r['d_' + k + '_lo'], r['d_' + k + '_hi'] = ci(dv)
        OUTR.append(r)
    for M_, d_ in [('Z', 'd1')] + [x for x in rows if x != ('Z', 'd1')]:
        Fv = Fm[M_][ok]; dv = Dd[d_][ok]; s = dv * Fv; o = ko_eval_row(Fv, dv, s, down, up_, nul, yy, W)
        if (M_, d_) == ('Z', 'd1'): base = o
        if (M_, d_) == ('Z', 'd4'): base4 = o
        add(f'{M_}|{d_}', M_, d_, 'GB', o, min(cover[M_], cover[d_]), int((dv == 0).sum()))
        if (M_, d_) in PRE or SMOKE: SYSARR[f'{M_}|{d_}'] = ko_eval_row(Fv, dv, s, down, up_, nul, yy, Wsys, with_resp=False); NULLSPEC[f'{M_}|{d_}'] = (Fv, dv, s)
    if not SMOKE:
        for k, v in ext_un.items():
            Fv = frac_rank(v, Mm & ~dup & np.isfinite(v))[ok]; o = ko_eval_row(Fv, None, None, None, None, nul, yy, W, resp_only=True); add(f'ext|{k}', k, '-', 'ext_unsigned', o, int((okb & np.isfinite(v)).sum()), 0)
        for k, v in ext_sg.items():
            s = v[ok]; Fv = frac_rank(np.abs(v), Mm & ~dup & np.isfinite(v))[ok]; dv = sg(s); o = ko_eval_row(Fv, dv, s, down, up_, nul, yy, W)
            add(f'ext|{k}', k, 'own sign', 'ext_signed', o, int((okb & np.isfinite(v)).sum()), int((s == 0).sum())); OUTR[-1]['n_used'] = n
            SYSARR[f'ext|{k}'] = ko_eval_row(Fv, dv, s, down, up_, nul, yy, Wsys, with_resp=False); NULLSPEC[f'ext|{k}'] = (Fv, dv, s)
    # v3: every unsigned external magnitude x a SHARED polarity rule (d1, d4, d_Pear) -> act/rep AP differ only by magnitude
    for k, v in ext_un.items():
        Fv = frac_rank(v, Mm & ~dup & np.isfinite(v))[ok]
        for d_ in ('d1', 'd4', 'd_Pear'):
            dv = Dd[d_][ok]; o = ko_eval_row(Fv, dv, dv * Fv, down, up_, nul, yy, W); add(f'extmag|{k}|{d_}', k, d_, 'ext_signed', o, int((okb & np.isfinite(v)).sum()), int((dv == 0).sum()))
            if d_ == 'd4':  # v3b: paired (same W draws) Z|d4 minus this external, resamples only
                for kk in ('act_APlift', 'rep_APlift', 'resp_APlift', 'act_AUPRC', 'rep_AUPRC'):
                    if kk in o and kk in base4: dd = base4[kk] - o[kk]; OUTR[-1]['GBd4_minus_' + kk] = dd[0]; OUTR[-1]['GBd4_minus_' + kk + '_lo'], OUTR[-1]['GBd4_minus_' + kk + '_hi'] = ci(dd)
    pd.DataFrame(OUTR).to_csv(PARTS / f'ko_{name.replace(":", "_")}.tsv', sep='\t', index=False)
    return  # v3: skip nulls/pseudo (already in v2)
    np.savez(PARTS / f'sys_{name.replace(":", "_")}.npz', bsy=bsy, **{f'{r}|{k}': v for r, o in SYSARR.items() for k, v in o.items()}); P(name, 'wrote', len(OUTR), 'rows + system-shared arrays', len(SYSARR), 'rows')
    # ---------- item 7: scrambled-target null (labels permuted within abundance deciles; class counts preserved per decile; unchanged evaluator)
    dec = pd.qcut(ab[ok], 10, labels=False, duplicates='drop'); rng = np.random.default_rng(zlib.crc32(name.encode())); NR = []
    real = {r: ko_eval_row(Fv, dv, s, down, up_, nul, yy, np.ones((1, n)), with_resp=False) for r, (Fv, dv, s) in NULLSPEC.items()}
    nulls = {r: {k: [] for k in ('act_APlift', 'rep_APlift', 'BA')} for r in NULLSPEC}
    for _ in range(NNULL):
        cp = cls.copy()
        for q in np.unique(dec): ix = np.flatnonzero(dec == q); cp[ix] = rng.permutation(cls[ix])
        dn_, upn, nl_ = cp == 0, cp == 1, cp == 2; yn = dn_ | upn
        for r, (Fv, dv, s) in NULLSPEC.items():
            o = ko_eval_row(Fv, dv, s, dn_, upn, nl_, yn, np.ones((1, n)), with_resp=False)
            for k in nulls[r]: nulls[r][k].append(o[k][0])
    for r in NULLSPEC:
        for k in ('act_APlift', 'rep_APlift', 'BA'):
            v = np.array(nulls[r][k]); rv = float(real[r][k][0]); lo, hi = np.nanpercentile(v, [2.5, 97.5])
            NR.append(dict(endpoint=name, row=r, metric=k, real=rv, null_mean=float(np.nanmean(v)), null_lo=float(lo), null_hi=float(hi), p_ge=(1 + np.nansum(v >= rv)) / (1 + np.isfinite(v).sum()), n_draws=int(np.isfinite(v).sum()), n_common=n, n_down=int(down.sum()), n_up=int(up_.sum())))
    pd.DataFrame(NR).to_csv(PARTS / f'nulls_{name.replace(":", "_")}.tsv', sep='\t', index=False); P(name, 'null done', NNULL, 'draws x', len(NULLSPEC), 'rows')
    # ---------- item 8: matched non-TF pseudo-sources
    if PSD and name in PSEUDO_EP:
        ros = json.load(open(chk(f'{PSD}/roster_{bsy}.json'))); pseu = ros['pseudo'][tf.upper()]; uni_f = info['uni']; PR = []
        def ps_axis(src):
            A = {}
            for r in ROOTS:
                zz = ld(f'{PSD}/stage1/{bsy}__{r}__R1.npz'); ss = [s.upper() for s in zz['sources'].astype(str)]; need((np.char.upper(zz['genes'].astype(str)) == axis).all() and zz['U'].shape[1:] == (len(axis), 9), 'pseudo stage1 axis/shape')
                j = ss.index(src.upper()); A[r] = (zz['U'][j].astype(float), zz['Sc0'][j].astype(float))
            si = aix[src.upper()]; ui = np.flatnonzero(uni_f); Z = np.full(len(axis), np.nan); Z[ui] = GP.zcons({h: np.abs(A[h][0][ui]) for h in ROOTS}, GP.KS)
            a = [np.where(uni_f, A[h][0][:, 0], np.nan) for h in ROOTS]; d1 = np.where(np.isfinite(a[0]) & np.isfinite(a[1]), (a[0] + a[1]) / 2, np.nan)
            c = [np.where(uni_f, A[h][1], np.nan) for h in ROOTS]; d4 = np.where(np.isfinite(c[0]) & np.isfinite(c[1]), (c[0] + c[1]) / 2, np.nan)
            Z[si] = np.nan; d1[si] = np.nan; d4[si] = np.nan; return Z, sg(d1), sg(d4)
        def ev(src):
            Za, d1a, d4a = ps_axis(src); Fz = frac_rank(mp(Za), base_prod & np.isfinite(mp(Za))); out = []
            for dn_, da in (('d1', d1a), ('d4', d4a)):
                dv_ = mp(da); m = ok & np.isfinite(Fz) & np.isfinite(dv_); yy_ = y[m]; ee_ = np.where(np.isfinite(e_[m]), e_[m], 0.0); dn, upm, nl = yy_ & (ee_ < 0), yy_ & (ee_ > 0), ~yy_
                s = dv_[m] * Fz[m]; o = ko_eval_row(Fz[m], dv_[m], s, dn, upm, nl, yy_, np.ones((1, int(m.sum()))), with_resp=False); out.append(dict(endpoint=name, source=src, is_real=src.upper() == tf.upper(), row=f'Z|{dn_}', n_used=int(m.sum()), **{k: float(v[0]) for k, v in o.items() if k in ('act_APlift', 'rep_APlift', 'act_AUPRC', 'rep_AUPRC', 'BA')}))
            return out, mp(Za)
        rl, Zr = ev(tf); mz2 = np.isfinite(Zr) & np.isfinite(Zst); zdd = float(np.abs(Zr[mz2] - Zst[mz2]).max()); P(name, 'PARITY pseudo-stage1 real-source Z vs stored production Z: max|diff|', zdd); need(zdd <= 1e-6, 'pseudo-stage1 real Z parity')
        for rrw in rl:
            main = [r_ for r_ in OUTR if r_['row'] == rrw['row']][0]; P(name, 'PARITY pseudo-path real', rrw['row'], 'act_APlift', rrw['act_APlift'], 'main', main['act_APlift'], 'BA', rrw['BA'], 'main', main['BA'])
        PR += rl
        for p_ in pseu: PR += ev(p_)[0]
        pdet = pd.DataFrame(PR); pdet.to_csv(PARTS / f'pseudodetail_{name.replace(":", "_")}.tsv', sep='\t', index=False); SM = []
        for rw in ('Z|d1', 'Z|d4'):
            for k in ('act_APlift', 'rep_APlift', 'BA', 'act_AUPRC', 'rep_AUPRC'):
                dd = pdet[pdet.row == rw]; rv = float(dd[dd.is_real][k].iloc[0]); pv = dd[~dd.is_real][k].to_numpy(float); pv = pv[np.isfinite(pv)]
                SM.append(dict(endpoint=name, row=rw, metric=k, real=rv, n_pseudo=len(pv), pseudo_mean=float(pv.mean()), pseudo_sd=float(pv.std(ddof=1)), pseudo_min=float(pv.min()), pseudo_max=float(pv.max()), real_rank_of_n=1 + int((pv > rv).sum()), n_total=len(pv) + 1, p_ge=(1 + int((pv >= rv).sum())) / (1 + len(pv))))
        pd.DataFrame(SM).to_csv(PARTS / f'pseudo_{name.replace(":", "_")}.tsv', sep='\t', index=False); P(name, 'pseudo done', len(pseu), 'pseudo sources')
    if MISSING: pd.DataFrame(MISSING, columns=['system', 'tf', 'row', 'reason']).to_csv(PARTS / f'missing_{TAG}.tsv', sep='\t', index=False)

# ---------------------------------------------------------------- chip5
CHIP_UNITS = {'PANC_RFX6': ('PANC', 'RFX6'), 'GATA1_GATA1': ('GATA1', 'GATA1'), 'KLF1_KLF1': ('KLF1', 'KLF1'), 'KLF1_B_KLF1': ('KLF1_B', 'KLF1'), 'BONEMARROW_RUNX1': ('BM_CHIPATLAS_CALTOT', 'RUNX1'),
              'BM_GATA1_ACUTE_GATA1': ('BM_GATA1_ACUTE', 'GATA1'), 'CARDIACV_TBX5': ('CARDIACV', 'TBX5'), 'CARDIACV_NKX2-5': ('CARDIACV', 'NKX2-5'), 'BM_CHIPATLAS_KLF1': (None, 'KLF1'), 'BM_CHIPATLAS_NFE2': (None, 'NFE2')}
LEV = ['A_act', 'C_act', 'B', 'D', 'C_rep', 'A_rep']
def groups_of(T, L):
    return [x for x in [T[L['A_act']], T[L['C_act']], np.concatenate([T[L['B']], T[L['D']]]), T[L['C_rep']], T[L['A_rep']]] if len(x)]

def run_chip(unit):
    import gb_fourclass_signed_k0 as GF
    sysn, tf = CHIP_UNITS[unit]; chk(GF.U[unit][0] % unit); dc = GF.build(unit); gg, c6 = dc['gg'], dc['c6']; _, _, _, bsy, _ = GF.U[unit]
    Mr_ax, D_ax, cont_ax, info = axis_rows(bsy, tf); selfg = (gg == tf.upper())
    if sysn is not None:
        sub, tr = [(e[1], e[3]) for e in F.EPS if e[0] == sysn and e[2] == tf][0]
        z = ld(f'{VEC}ep_{sysn}__{tf}__{tr}.vec.npz'); g = up(z['genes']); Mm = z['MR'].astype(bool); dup = pd.Series(g).duplicated(keep=False).to_numpy(); selfe = (g == tf.upper())
        mp = mapper(info['axis'], g); Mr = {k: mp(v) for k, v in Mr_ax.items()}; Dd = {k: mp(v) for k, v in D_ax.items()}; base_prod = Mm & ~dup
        d4_parity(bsy, tf, info['axis'], cont_ax['d4'], unit)
        pe = ld(f'{SG}{bsy}_pearson.npz'); pg = np.char.upper(pe['genes'].astype(str)); pi = {x: i for i, x in enumerate(pg)}
        r_ = pe['r'][[s.upper() for s in pe['sources'].astype(str)].index(tf.upper())]; Pr = np.array([r_[pi[x]] if x in pi else np.nan for x in g])
        zr = ld(RU.system_dir(bsy) / 'regvelo_v2_full.npz'); rg = {x.upper(): i for i, x in enumerate(zr['genes'].astype(str))}
        rr = np.asarray(zr['score_all_signed'], float)[[s.upper() for s in zr['regulators'].astype(str)].index(tf.upper())]
        ext_sg = {'Pearson(signed)': Pr, 'RegVelo(signed J)': np.array([rr[rg[x]] if x in rg else np.nan for x in g]), **sig_ext(bsy, tf, g)}; src = 'endpoint-vector route (npz-equivalent)'
        Mr['Z'] = z['cog::GeoBRIDGE'].astype(float); genes = g
        ext_sg = {k: np.where(selfe, np.nan, v) for k, v in ext_sg.items()}; Mr = {k: np.where(selfe, np.nan, v) for k, v in Mr.items()}
    else:
        genes = gg; mp = mapper(info['axis'], gg); Mr = {k: mp(v) for k, v in Mr_ax.items()}; Dd = {k: mp(v) for k, v in D_ax.items()}
        Mr['Z'] = np.asarray(dc['uns'], float); base_prod = np.ones(len(gg), bool); src = 'four-class-gene route (bank; Z = fourclass-file production)'
        ext_sg = {k: np.where(selfg, np.nan, v) for k, v in sig_ext(bsy, tf, gg).items()}; Mr = {k: np.where(selfg, np.nan, v) for k, v in Mr.items()}
        s0 = mp(cont_ax['d1']); m = np.isfinite(s0) & np.isfinite(dc['s0']); dd_ = float(np.abs(s0[m] - dc['s0'][m]).max()); ss_ = int((sg(s0[m]) != sg(dc['s0'][m])).sum()); P(unit, 'PARITY d1 stage1 S0 vs fourclass bank s0: n', int(m.sum()), 'max|diff|', dd_, 'sign mismatches', ss_); need(dd_ <= 1e-6 and ss_ == 0, 'bank d1 parity')
    for m_ in list(MISSING): P(unit, 'MISSING row (reported missing, not zero):', m_)
    Fm = {k: frac_rank(v, base_prod & np.isfinite(v)) for k, v in Mr.items()}
    rowsl = SMOKE_ROWS if SMOKE else [(m, d) for m in MROWS for d in DROWS]
    if sysn is None: rowsl = [x for x in rowsl if x[1] != 'd4']; P(unit, 'NOTE: bank unit has no independent gb_d1d4 d4 file -> all d4 rows NOT RUN (reported missing, not zero)')
    def to_gg(T):
        if sysn is None: return T
        f = np.isfinite(T); return pd.Series(gg).map(dict(zip(genes[f], T[f]))).to_numpy(float)
    rng = np.random.default_rng(0); idx_b = [{c: (rng.choice(np.flatnonzero(c6 == c), (c6 == c).sum()) if (c6 == c).any() else np.array([], int)) for c in LEV} for _ in range(B)]
    Tsets = {(m, d): to_gg(Dd[d] * Fm[m]) for m, d in rowsl}
    for k, v in ext_sg.items(): Tsets[('ext|' + k, 'own sign')] = to_gg(np.where(base_prod & np.isfinite(v), v, np.nan))
    if sysn is not None:  # v3: unsigned external magnitudes x the SAME GB polarity (d4, d1) -> only the magnitude differs
        for m_ in UNS_EXT:
            Fx = frac_rank(np.where(selfe, np.nan, z[f'cog::{m_}'].astype(float)), base_prod & np.isfinite(z[f'cog::{m_}'].astype(float)))
            for d_ in ('d4', 'd1'): Tsets[('extmag|' + m_, d_)] = to_gg(Dd[d_] * Fx)
    for k in Tsets: Tsets[k] = np.where(selfg, np.nan, Tsets[k])
    # v3d: dump per-gene signed scores + 6-class label for selected rows, then stop (no bootstrap)
    keep = [k for k in Tsets if k in (('Z', 'd1'), ('Z', 'd4'), ('absPear', 'd_Pear'), ('absPear', 'd4'), ('Z_M', 'd4')) or (k[0].startswith('extmag|') and k[1] == 'd4')]
    pd.DataFrame({'gene': gg, 'cls': c6, **{f'{m}|{d}': Tsets[(m, d)] for m, d in keep}}).to_csv(PARTS / f'T_{unit}.tsv', sep='\t', index=False); P(unit, 'dumped T', len(keep)); return
    # parity on the ORIGINAL (pre-common-support) T_A (Z,d1): JT z must equal the T_A reference
    L0 = {c: np.flatnonzero((c6 == c) & np.isfinite(Tsets[('Z', 'd1')])) for c in LEV}; zpar = jt_z(groups_of(Tsets[('Z', 'd1')], L0)); ref = rtsv(TA + 'chip5.tsv').set_index('unit')
    P(unit, 'PARITY T_A (Z,d1) JT_z mine', zpar, 'ref', float(ref.loc[unit, 'JT_z']), '| n_T_finite', int(np.isfinite(Tsets[('Z', 'd1')]).sum()), int(ref.loc[unit, 'n_T_finite']), '| ref T_source:', ref.loc[unit, 'T_source']); need(abs(zpar - float(ref.loc[unit, 'JT_z'])) <= 1e-6 and int(np.isfinite(Tsets[('Z', 'd1')]).sum()) == int(ref.loc[unit, 'n_T_finite']), 'chip T_A parity')
    # ONE common-support universe over ALL rows (fix 4)
    fin = np.ones(len(gg), bool); cov = {}
    for (m, d), T in Tsets.items():
        cov[f'{m}|{d}'] = int(np.isfinite(T).sum())
        if m[4:] in DROPX: continue
        fin &= np.isfinite(T)
    P(unit, 'common-support universe n', int(fin.sum()), 'of', len(gg), 'row coverage', {k: v for k, v in cov.items() if v < len(gg) - 0 and v != int(fin.sum())})
    Tsets = {k: np.where(fin, T, np.nan) for k, T in Tsets.items()}; OUTR = []
    for (m, d), T in Tsets.items():
        L = {c: np.flatnonzero((c6 == c) & np.isfinite(T)) for c in LEV}; row = dict(unit=unit, row=f'{m}|{d}', M=m, d=d, T_source=src, n_genes=len(gg), n_common=int(fin.sum()), n_T_finite=int(np.isfinite(T).sum()), n_coverage_before=cov[f'{m}|{d}'], n_ambiguous_d0=int((T == 0).sum()))
        for c in LEV: row[f'n_{c}'] = len(L[c]); row[f'median_{c}'] = float(np.median(T[L[c]])) if len(L[c]) else np.nan
        grp = groups_of(T, L); row['JT_levels_used'] = len(grp); zobs = jt_z(grp) if len(grp) >= 2 else np.nan; row['JT_z'] = zobs
        allidx = np.concatenate([L[c] for c in LEV]); sizes = [len(x) for x in grp]; zp = []
        if len(grp) >= 2:
            prng = np.random.default_rng(1); cut = np.cumsum([0] + sizes)
            for _ in range(500):
                tp = T[allidx] * prng.choice([-1.0, 1.0], len(allidx)); zp.append(jt_z([tp[cut[i]:cut[i + 1]] for i in range(len(sizes))]))
        zp = np.array(zp); row['JT_perm_p'] = (1 + np.nansum(zp >= zobs)) / (1 + np.isfinite(zp).sum()) if np.isfinite(zobs) and len(zp) else np.nan
        def contrasts(ix=None):
            gc = (lambda c: L[c]) if ix is None else (lambda c: ix[c][np.isfinite(T[ix[c]])])
            L3 = np.concatenate([gc('B'), gc('D')]); res = {}
            res['L1_vs_L2'] = cliff(T[gc('A_act')], T[gc('C_act')]); res['L1_vs_L3'] = cliff(T[gc('A_act')], T[L3])
            res['L5_vs_L4'] = cliff(-T[gc('A_rep')], -T[gc('C_rep')]); res['L5_vs_L3'] = cliff(-T[gc('A_rep')], -T[L3]); return res
        c0 = contrasts(); bb = [contrasts(ix) for ix in idx_b]
        for k, v in c0.items(): row[k] = v; row[k + '_lo'], row[k + '_hi'] = ci_r([x[k] for x in bb])
        OUTR.append(row)
    D = pd.DataFrame(OUTR); D.to_csv(PARTS / f'chip_{unit}.tsv', sep='\t', index=False); P(unit, 'wrote', len(D), 'rows')
    if MISSING: pd.DataFrame(MISSING, columns=['system', 'tf', 'row', 'reason']).to_csv(PARTS / f'missing_chip_{TAG}.tsv', sep='\t', index=False)

for it in ITEMS:
    (run_ko if MODE == 'ko' else run_chip)(it)
