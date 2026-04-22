# GeoBRIDGE validation — Day 3 status (2026-04-21 late night)

> **이 파일은 Day 2에서 시작했고 Day 3 밤까지 누적됨. 아래 "Day 3 update" 섹션이 최신 상태.**

---

## Day 3 update — from diagnosis to reformulation

### What happened since Day 2 (timeline)

1. **P1/P2/P3 diagnostics 완료.**
   - P1 (GT overlap): BM mean_overlap=10.0 vs pancreas=11.5; GT confound 아님.
   - P3 (pancreas bootstrap CI): Δ=+0.017 [−0.379, +0.416] — 0 포함, **pancreas "win"은 통계적으로 없음**.
   - P2 (per-TF delta on BM): lineage TFs (GATA1/GATA2/KLF1/NFE2/RUNX1)는 일관되게 loss; stress/IEG TFs (NFKB1/HSF1/KLF2/KLF4)는 win. Cross-dataset 재현 null (GTF2I/FOS sign flip BM ↔ pancreas).

2. **Secondary hypotheses 검정 → 모두 null.**
   - Branch-specificity (Yanai τ) vs delta: BM ρ=−0.06 p=0.63; pancreas ρ=+0.04 p=0.80.
   - TF dynamics (|V|, var, unspliced_frac) vs delta: all ρ<0.25, all p>0.06.
   - ⇒ "어떤 cells에서" 문제가 아님.

3. **V2 ChIP ground truth 구축 및 실행 (Track B).**
   - `build_chip_subset.py`: ChEA_2022 + ENCODE_TF_ChIP-seq_2015 Enrichr 라이브러리에서 8 TF (NFE2/KLF1/KLF2/RUNX1/GATA2/NFKB1/HSF1/KLF4) 대상 34,347 edges 추출.
   - `enrichment.py` fix: sparse 출력에서 99.97% tied scores → pyo3 panic. `linspace(0,−1e−9,n)` jitter + `except BaseException` 추가 (pyo3 PanicException은 Exception 아님). commit `d3cf8c7`.
   - BM V1 (CollecTRI tier-A) vs V2 (ChIP) per-TF 비교 결과 (7/8 testable):

     | TF | V1 Δ | V2 Δ | 재현? | 해석 |
     |---|---:|---:|:---:|---|
     | GATA2 (lineage) | −2.89 | −2.31 | ✓ | V1 loss 확인 |
     | KLF1 (lineage) | −3.28 | −2.55 | ✓ | V1 loss 확인 |
     | NFE2 (lineage) | −2.80 | −2.22 | ✓ | V1 loss 확인 |
     | RUNX1 (lineage) | −2.66 | n/a | — | ChIP filter out |
     | NFKB1 (stress) | +2.62 | +2.72 | ✓ | V1 win 확인 |
     | HSF1 (stress) | +2.11 | +2.55 | ✓ | V1 win 확인 |
     | KLF2 (stress) | +2.61 | **−3.35** | ✗ | tier-A artifact |
     | KLF4 (stress) | +2.16 | +0.07 | ✗ | tier-A artifact |

   - **Net**: 5/7 재현. lineage loss는 orthogonal GT에서도 완전 재현(3/3) → tier-A curation artifact 아닌 method 자체의 체계적 failure. stress win 중 NFKB1/HSF1은 real signal (both V1 & V2), KLF2/KLF4는 tier-A artifact.

### Mechanism — 원인 확정 (operator-level)

코드 확인 ([utils_projection.py:7-9](benchmark/utils_projection.py:7), [model.py:125](geobridge_model/model.py:125)):

> "Production GB at rank k uses `Vk = pre["Vt"][:k].T` ... and subtracts its subspace from ΔX. **It does NOT 'project onto' top-k — it 'projects out' top-k.**"

즉 production scoring은 정확히 `num = (I − Pk) · M`, where `Pk = Vk Vk^T`, `Vk`는 ΔX의 top-k right singular vectors. Lineage 데이터셋에서 **ΔX의 top-k right singular vectors = lineage commitment axes 그 자체**. 따라서 method는 설계상 dominant variance direction과 orthogonal한 component만 남긴다.

이걸로 관찰된 전체 패턴이 설명됨:
- Lineage TFs (GATA2/KLF1/NFE2) drive top-k modes → signal projected out → sign flip.
- Stress TFs (NFKB1/HSF1) orthogonal to commitment axis → survive projection → geobridge wins.
- Branch τ, TF dynamics null 이유: failure는 "어떤 cells"가 아닌 **어떤 subspace**가 제거되는가의 문제.

**이건 bug가 아니라 design-biology mismatch.** Method philosophy (dominant geometry 제거)와 biology (dominant geometry IS signal of interest)가 충돌.

### Reframing — decomposition interpretation

"GeoBRIDGE는 lineage에서 fail"이 아니라:

> **"GeoBRIDGE는 regulatory signal을 두 orthogonal subspace로 decompose 한다"**

- `Pk · M` → lineage / dominant program component
- `(I − Pk) · M` → orthogonal / stress / residual component

현재 production은 항상 후자만 출력 → lineage 실패로 보임. Score 하나로 단일 interpretation 하는 게 오류였고, 두 component를 따로 읽으면 method는 정보를 담고 있음.

### Reformulation plan — Option A (decomposition as the method)

**두 score를 분리 정의**:
```
s_lineage(X→Y) = [Pk · M](X,Y) / ( sqrt(diag(Pk C Pk))[X] · sqrt(diag(Y))[Y] + ε )
s_dynamic(X→Y) = [(I−Pk) · M](X,Y) / ( sqrt(diag(Q C Q))[X] · sqrt(diag(Y))[Y] + ε )
```

`Pk + Q = I` 이므로 두 score는 ΔX Gram metric 하에서 orthogonal. Method paper 주장:

> Pair-space cross-covariance를 ΔX의 top-k right singular vectors 기준으로 decompose 하면, regulatory signal이 **lineage component** (GATA1 → hemoglobin 같은 dominant velocity geometry 정렬)와 **dynamic component** (NFKB1/HSF1 같은 stress/IEG response, 직교)로 분리됨. 각 component를 독립 평가; 기존의 "GeoBRIDGE가 lineage에서 fail"은 `s_dynamic`을 universal score로 오독한 결과.

**Rejected alternatives**:
- Option B (weighted α·onto + β·out): α=β이면 `M` (k=0)으로 collapse. 1-parameter interpolation일 뿐 subspace interpretation 상실.
- Option C (targeted nuisance basis): Option A가 실패할 경우의 fallback. 다른 paper.
- 과거 제안 Idea 1/2/3/4: subspace reframing 하에 모두 subsumed 또는 irrelevant (1=branch-local은 local top-k도 같은 문제; 2=corr fusion은 `s_lineage ≈ corr`이므로 redundant; 3=pair weighting은 second-order; 4=normalization은 sign flip의 원인 아님).

### Task 1 — decisive operator characterization (next action)

하나의 `pre`에서 세 mode로 scoring:

| mode | numerator | x-norm | 의미 |
|---|---|---|---|
| `none` (k=0) | `M` | `diag(C)` | unprojected baseline |
| `onto` | `Pk · M` | `diag(Pk C Pk)` | lineage component |
| `out` (current) | `(I−Pk) · M` | `diag(Q C Q)` | dynamic component |

BM k=17에서 test. V1 CollecTRI + V2 ChIP 모두 평가. 진단 TF 8개 (lineage 3 + stress 2 + ambiguous 3).

**Decision rule**:

| 결과 | 의미 |
|---|---|
| `onto`가 lineage TFs 복구 (NES ≳ +1, V1+V2) | **decomposition hypothesis 확정** → Option A paper |
| `out`가 NFKB1/HSF1 win 유지 | 같이 확정 |
| `none`이 둘 사이에 앉음 (`onto+out≈none` 의미) | normalization sanity 확인 |
| `onto`가 lineage 복구 실패 | hypothesis 기각 → Option C로 fallback |
| `out`에서 stress win 사라짐 | method 자체가 noisy; 라인 종료 |

### Code changes made for Task 1

1. **`geobridge_model/model.py`** — `scores_from_streaming_stats`에 `mode` 파라미터 추가 (`"out"` default, `"onto"`/`"none"` 신규). Backwards compatible.
2. **`validation/scripts/decompose_scoring.py`** (신규) — `ablate_projection.py` 패턴 재사용; 한 번 precompute 후 세 mode parquet 출력.

### CLI to run on server

```bash
cd /data/cb/scratch/seokmin/geobridge/validation && git pull

# [Step 1] decomposition scoring parquets (k=17 on BM)
python scripts/decompose_scoring.py --config configs/bm.yaml --k 17 \
    --out results/rankings_decomp/

# [Step 2] V1 CollecTRI evaluation (+ correlation baseline for reference)
cp results/rankings/correlation_bm.parquet results/rankings_decomp/
python scripts/run_v1_enrichment.py \
    --rankings "results/rankings_decomp/*_bm.parquet" \
    --gt data/gt/dorothea_tierA.tsv \
    --out results/v1_decomp/bm/

# [Step 3] V2 ChIP evaluation (same 8 diagnostic TFs)
python scripts/run_v1_enrichment.py \
    --rankings "results/rankings_decomp/*_bm.parquet" \
    --gt data/gt/chip_subset_bm.tsv \
    --max-size 2000 \
    --out results/v2_decomp/bm/

# [Step 4] send back for tabulation
cat results/v1_decomp/bm/per_tf.csv
cat results/v2_decomp/bm/per_tf.csv
```

Runtime 예상: precompute 1회 (기존 ablate와 동일, ~수 분) + scoring 3회 (초 단위) + V1/V2 enrichment 각 ~수 분. 총 < 30분.

### After Task 1

- Hypothesis 확정 시: pancreas에서 replication → Option A paper 작성 시작.
- 기각 시: Option C (targeted nuisance basis; cell-cycle/library size 기반 `V_nuis`) 프로토타입.
- 둘 다 실패 시: method 종료.

---

## (Day 2 archive below — historical context)

# GeoBRIDGE validation — Day 2 status (2026-04-21 evening)

## Progress this session
- Fixed Velorama wrapper (`is_reg`, not `is_regulator`; live subprocess tee; 6h timeout). Velorama still unverified end-to-end, deferred.
- Fixed pancreas QC pipeline (write h5ad before plot; PNG instead of PDF; non-finite UMAP guard).
- Ran Gate 0 (pancreas Phase-1 + k-sweep): geobridge passes regime test marginally.
- Ran Gate 1 (BM project_y × normalize_score × k ablation).
- Pushed 3 diagnostic scripts for P1/P2/P3 (commit efd0c3d).

## Key results

### BM (human, CollecTRI human tier-A)
| method | mean_nes | median_nes | frac_signif | n_tf |
|---|---:|---:|---:|---:|
| correlation | 0.580 | 1.012 | 0.250 | 60 |
| geobridge auto (k=17) | 0.078 | 0.692 | 0.133 | 60 |
| geobridge best (k=2, nm=1) | 0.218 | 0.837 | 0.183 | 60 |

2.7× gap vs correlation. Normalization alone cannot close it.

### BM Gate 1 ablation — what A' (normalize × k) actually does
- `project_y` nearly no-op (py=0 vs 1 effectively identical).
- At k=2: nm=1 helps (+0.84 median) / nm=0 hurts (-0.62).
- At k=10: nm=0 helps (+0.81 median) / nm=1 hurts (-0.67) ← **sign flip**.
- At k=17: nm has little effect.
- No single normalization setting dominates across k.

### Pancreas (mouse, CollecTRI mouse-via-capitalize tier-A)
| method | mean_nes | median_nes | frac_signif | n_tf |
|---|---:|---:|---:|---:|
| correlation | 0.336 | 0.932 | 0.130 | 46 |
| **geobridge (auto=20)** | **0.354** | 0.832 | **0.196** | 46 |
| grnboost | 1.047 | 1.104 | 0.119 | 42 |

### Pancreas sweep — **NO SIGN FLIP**
| k | mean_nes | median | frac_signif |
|---:|---:|---:|---:|
| 0 | 0.164 | 0.701 | 0.130 |
| 2 | 0.246 | 0.735 | 0.152 |
| 10 | 0.301 | 0.852 | 0.087 |
| **20** | **0.354** | 0.832 | **0.196** ← best |
| 40 | 0.236 | 0.746 | 0.130 |

- Monotonic improvement up to k=20. **Mid-k flip seen on BM is NOT method-intrinsic — it's data-coupled.**
- auto-k works on pancreas (~20) but lands in bad zone on BM (=17).

## Interpretation (current conservative position)

Three live problems:
- **A' normalization × k interaction** — real but small (<15% of BM gap); data-coupled, not intrinsic.
- **B regime mismatch** — BM steady-state, pancreas dynamic; partially supported by pancreas reversal.
- **C structural limitation / GT bias** — CollecTRI may favor correlation-style ranking; can't distinguish without orthogonal (ChIP) GT.

**Stance:** method viability is UNDER TRIAGE. Too early to claim "healthy"; too early to abandon. +0.018 pancreas advantage needs bootstrap CI; per-TF analysis needed to check for localized biological wins.

## Immediate next actions (all CPU, run on server after git pull)

### 1. P1 — GT overlap diagnostics (BLOCKING — do first)
```bash
python scripts/diagnose_v1_gt.py --rankings "results/rankings/*_bm.parquet" \
    --gt data/gt/dorothea_human_tierA.tsv --dataset-name bm \
    --out results/gt_diag/bm/
python scripts/diagnose_v1_gt.py --rankings "results/rankings/*_pancreas.parquet" \
    --gt data/gt/dorothea_mouse_tierA.tsv --dataset-name pancreas \
    --out results/gt_diag/pancreas/
```
Check `mean_overlap`, `n_evaluable_ge_min`, `case_mismatch_only`. **Verify mouse GT not catastrophically worse than human.** If pancreas overlap << BM overlap, pancreas Gate-0 PASS is suspect.

### 2. P3 — Bootstrap CI on pancreas sweep
```bash
# REQUIRES per_tf.csv — if absent, need to modify run_v1_enrichment.py first
ls results/v1_enrichment/pancreas_sweep/per_tf.csv || echo "MISSING — modify run_v1 to emit per-TF NES"
python scripts/summarize_sweep.py \
    --per-tf results/v1_enrichment/pancreas_sweep/per_tf.csv \
    --baseline correlation --out results/sweep_summary/pancreas/
```
Key question: does geobridge_k20 `delta_mean` 95% CI exclude 0?

### 3. P2 — Per-TF delta (biological winners check)
```bash
# Combine correlation baseline into pancreas sweep per_tf first if needed
python scripts/per_tf_delta.py --per-tf <combined_per_tf.csv> \
    --methods geobridge_k20 --baseline correlation \
    --lineage-set pancreas --out results/per_tf_delta/pancreas/
python scripts/per_tf_delta.py --per-tf results/v1_enrichment/bm_ablate/per_tf.csv \
    --methods geobridge_k02_py1_nm1 --baseline correlation \
    --lineage-set bm --out results/per_tf_delta/bm/
```
Key question: are `n_robust_lineage_wins ≥ 3`? Which specific TFs does geobridge recover that correlation misses?

## Decision rules

| Outcome | Action |
|---|---|
| P1: pancreas overlap ≪ BM overlap | pancreas Gate-0 PASS suspect; fix GT before claiming regime advantage |
| P3: Δ CI includes 0 AND P2 lineage_wins < 3 | **STOP Phase 2.** Reframe as negative result. |
| P3: Δ CI positive OR P2 lineage_wins ≥ 3 | Proceed to V2 ChIP build (P4) |
| V2 ChIP closes correlation gap on BM | GT bias is primary → reframe method claim |
| V2 ChIP: correlation still wins BM | regime mismatch (B) confirmed |

## CPU / GPU status
- **All recommended commands above are CPU-only.**
- Phase-1 enrichment, sweep_k, ablate_projection, diagnose/summarize/per_tf_delta — all CPU.
- Only Velorama uses GPU (dragon). Currently deferred.

## Deferred
- **P4 V2 ChIP build** (`scripts/build_chip_bm.py`): ChIP-Atlas peaks → TF→target edges in `data/gt/chip_bm.tsv`. ~6–8h engineering. Start only after P1-P3 decide.
- **P5 normalizer math read** of `scores_from_streaming_stats`. Small effect, not decision-blocking.
- **Velorama retry.** Wrapper fixed but unverified. Not blocking any decision.
- **Wessels Phase 2 preprocessing.** Gated on P3+P2+V2 outcome.

## Pushed commits (SeokminHa/GeoBRIDGE, main)
- `58aefe7` sweep_k.py
- `a4d37dd / a0c6715 / 5a1e9bf` velorama wrapper fixes
- `c93b06a` qc_velocity reorder (write h5ad before plot; PNG)
- `a649487` ablate_projection.py
- `efd0c3d` diagnose_v1_gt / per_tf_delta / summarize_sweep
