# GeoBRIDGE validation — Day 5 status (2026-04-22)

> **Day 2 → Day 3 → Day 4 → Day 5 누적. 아래 "Day 5" 섹션이 최신. 세션 리셋 후 이 섹션부터 읽으면 됨. Day 4 이하는 배경으로만.**

---

## Day 5 — from "nuisance projection" to direct pseudotime-drift test

**핵심 전환**: Day 4의 "nuisance projection" 실험이 겉보기로는 lineage TF 2개를 복구했지만, nuisance feature가 processed X에서 fallback된 상태였고 V_nuis의 top gene loadings를 확인해 보니 그것이 *technical confounder* 가 아니라 *lineage commitment axis* 였다. 그래서 "nuisance 제거" 라는 해석이 무너졌고, 대신 바로 그 lineage axis = **pseudotime drift** 를 projecting out 하는 clean mechanistic test로 넘어간다.

### Day 4 실행 결과 (settled facts)

STEP 1 (nuisance projection) + STEP 2 (early-PT restriction) 둘 다 서버에서 실행 완료.

**STEP 2 — early-PT restriction** (strict decision rule fail):
- `--pt-quantile 0.25`, `--ks 0,2,auto` 3개 모두 BM V2 ChIP 에서 lineage TFs GATA2/KLF1/NFE2 NES 여전히 음수 (모두 < 0). stress TFs NFKB1/HSF1 은 양수 유지.
- **해석 (교정된 버전)**: 단순히 "late-cell plateau 가 원인이 아니다" 라고만 말할 수 있음. 같은 subset 위에서 k=0/k=2/k=auto 를 모두 test 했기 때문에 "auto-k 가 바뀌어서 결론이 달라졌다" 는 confound 는 제거됨. 하지만 cell subset + GxG subset 이 동시에 줄어든 상황이라 "early regime 자체 에서도 lineage negative" 라는 제한된 statement 만 유효.
- **Decision**: late plateau 가설 **기각**.

**STEP 1 — nuisance projection** (provenance 분리 후 재해석):
- 원래 설계 (cell cycle S/G2M, log_total_counts, pct_mito, pct_ribo, n_genes) → BM adata 에는 `adata.raw = None` + obs 에 QC 컬럼 전무 → 5 feature 전부 processed X (Ms) 에서 fallback 계산됨.
- Script 를 provenance-aware 로 patch (commit 97289a3 계열): tag 가 `gb_nuisQC_*` (all QC-genuine) vs `gb_nuisPROC_*` (any processed fallback) 로 분화됨. BM 은 전부 `gb_nuisPROC_*` 만 나옴.
- `gb_nuisPROC_out_d5` (d=5, 모든 feature 를 processed X 기반으로) V2 ChIP per_tf 결과:
  | TF | class | NES |
  |---|---|---:|
  | GATA2 | lineage | **−1.17** (unchanged) |
  | KLF1  | lineage | **+1.01** (flipped ✓) |
  | NFE2  | lineage | **+1.06** (flipped ✓) |
  | NFKB1 | stress  | +1.21 (kept) |
  | HSF1  | stress  | +1.53 (kept) |
  | KLF4  | stress  | +1.05 (kept) |
- Strict rule (lineage 2/3 NES ≥ +0.5 AND stress 유지) = **FIRST STRONG SUCCESS** on BM. 그러나 이게 "nuisance 가 lineage TF 를 오염시키고 있었다" 는 증거인지는 별개 문제.

### V_nuis gene loading 검사 — 결정적 반전

`vnuis_bm.npz` 의 V_nuis (G × 5) 각 component 에서 |loading| top gene 들을 확인한 결과 (값은 approx):

| component | top genes |
|---|---|
| v1 (S-score analog)     | MPO, LYZ, ELANE, AZU1 (granulocyte/myeloid lineage markers) |
| v2 (G2M-score analog)   | TOP2A, MKI67 (실제 cycling 신호) + MPO 혼합 |
| v3 (log_total_counts)   | HBB, HBA1, HBA2, GATA1 타깃 (erythroid lineage markers) |
| v4 (pct_mito proxy)     | MT-코드 안나오고 대신 HLA / Ig germline 단편 |
| v5 (pct_ribo proxy)     | CD79B, VPREB1/3, MS4A1 (B-lineage markers) |

즉, "processed X 에서 뽑은 nuisance proxy" 5개가 사실상 **myeloid / cycling / erythroid / lymphoid / B-cell** 축을 span 하는 lineage commitment basis 와 거의 같다. `(I − V_nuis V_nuis^T) M` 이 KLF1/NFE2 를 복구한 이유는 "nuisance 제거" 가 아니라 **lineage commitment direction 자체를 제거해 보니 sign 이 뒤집혔다** 는 뜻.

### Sign-inversion framing (corrected interpretation)

Day 3 / Day 4 에서 "method fails on lineage" 라고 했지만, magnitudes 를 보면 |NES| ≈ 1.0–1.5 로 correlation baseline 과 비슷. 단지 **부호** 가 반대. Score 가 `[−1, 1]` 이고 −1 이 strong negative regulation, +1 이 strong positive regulation 이라는 semantics 하에서:

> GeoBRIDGE 는 KLF1 → hemoglobin targets 같은 **positive** lineage regulation 을 **negative** 로 예측 중. Targets 은 맞게 고르고 있지만 direction 을 체계적으로 뒤집음.

"fails to rank" 가 아니라 **"systematically sign-inverts on lineage programs"**. 이건 근본적으로 다른 진단이고, 해결 가능성이 더 있을 수도 있는 failure mode.

### Core hypothesis — sign-flip mechanism

Sign inversion 의 후보 2 개 (상호 배타 아님):

**(A) Pseudotime drift dominance** — forward pair sampling 이 pseudotime 증가 방향 pair 만 뽑기 때문에, committed regime 에서 ΔX ≈ Δpt · (lineage axis) 가 되고 ΔV 의 미분 방향과 **기하적으로** 둔각을 이룸. 결과: M = ΔXc^T ΔV 의 lineage row 가 음수.

**(B) Velocity phase mismatch** — TF 의 transcription (ΔV_TF) 은 미래 변화를 예측하므로 target 의 현재 ΔV 와 positive 하게 상관되어야 하지만, mature commitment 단계에서 TF 는 이미 peak plateau 에 들어가 있고 target 은 여전히 overshooting → ΔX_TF ≈ 0 이지만 ΔV_target 은 계속 양수. 이건 spliced/unspliced RNA velocity 의 time-lag artifact.

가설 (A) 를 test 하는 direct operator 가 Day 5 의 STEP 5.

### STEP 5 (implemented, not yet run) — direct pseudotime-drift projection

파일: [validation/scripts/run_pt_drift_projection.py](validation/scripts/run_pt_drift_projection.py)

핵심 차이 vs Day 4 nuisance projection:
- Nuisance feature 를 돌아가지 않음. **Δpt 자체** (polynomial basis `[Δpt, Δpt², ...]`) 로 `ΔX ≈ poly(Δpt) @ B` OLS.
- B.T 의 QR → `V_drift ∈ R^{G × d}` (d = order).
- `(I − V_drift V_drift^T) · M` 으로 scoring.
- `--orders 1,2,3` 다 비교 (order=1 은 순수 linear drift, order≥2 는 accelerating/decelerating commitment).
- `--max-pairs-per-cell` override (BM full precompute 75분 → 5분 수준으로 감축 가능).

출력 tag:
- `gb_drift_none` — raw M baseline (한 번만 emit)
- `gb_drift_out_o{order}d{d}` — main test (order 별로 하나)

Decision rule (STEP 3 과 동일 strict):
- lineage TFs GATA2 / KLF1 / NFE2 중 **2/3 이상 NES ≥ +0.5** AND stress TFs (NFKB1/HSF1) 유지 → **가설 (A) 확정** — sign-flip 은 pseudotime drift 가 M 을 dominate 했기 때문. Method 는 drift-residual regime 에서 올바른 방향 찾음.
- lineage 복구 실패 → 가설 (A) 기각, (B) velocity phase mismatch 로 넘어감 (STEP 6).

### STEP 6 (planned, only if STEP 5 fails) — ODE-degradation correction

spliced/unspliced velocity 의 time-lag 가 원인이면 ΔV 를 `ΔV_corrected = ΔV + γ · ΔX_target` 로 조금씩 "앞으로 당겨서" scoring 해보는 1-parameter sweep (γ ∈ {0, 0.1, 0.5, 1.0}). 그러나 이건 STEP 5 결과 본 후 결정.

### Data caveats (다음 session 시작 시 반드시 확인)

1. `adata.raw is None` for BM h5ad — raw spliced count 기반 QC feature 재생성 불가.
2. `adata.obs` 에 `total_counts`, `n_genes`, `pct_counts_mt`, `pct_counts_ribo` 전무 — scanpy 기본 QC 도 돌아있지 않음.
3. 따라서 "genuine QC nuisance" 실험을 정식으로 하려면 raw h5ad 에서 다시 preprocess 해야 함 (Wessels pipeline rerun). 지금 session 에서는 하지 않았고, STEP 5 결과 에 따라 필요할 수도 있고 불필요할 수도 있음.
4. BM GxG = 3000, n_cells = 5780, forward pair pool ≈ 16.7M. `max_pairs_per_cell=200` 이면 ~1.2M pair 로 GxG accumulation ~5분; default (`None` = 모두) 면 75분.

### Server CLI — STEP 5 execution order (CPU 전부)

```bash
cd /data/cb/scratch/seokmin/geobridge/validation && git pull

# ---- STEP 5: pseudotime-drift projection on BM ----
# max_pairs_per_cell 200 으로 빠르게 돌림. 정확도 더 필요하면 300~500.
python scripts/run_pt_drift_projection.py --config configs/bm.yaml \
    --orders 1,2,3 --modes none,out \
    --max-pairs-per-cell 200 \
    --out results/rankings_drift/

# ---- STEP 5 eval: V2 ChIP 만 돌림 (lineage/stress TFs 8개) ----
mkdir -p results/rankings_day5/bm
cp results/rankings/correlation_bm.parquet          results/rankings_day5/bm/
cp results/rankings_drift/bm/gb_drift_none_*.parquet results/rankings_day5/bm/
cp results/rankings_drift/bm/gb_drift_out_*.parquet  results/rankings_day5/bm/

python scripts/run_v1_enrichment.py \
    --rankings "results/rankings_day5/bm/*_bm.parquet" \
    --gt data/gt/chip_subset_bm.tsv --max-size 2000 \
    --out results/v2_day5/bm/

cat results/v2_day5/bm/per_tf.csv
```

Runtime 예상: drift script ~7분 (streaming precompute 5분 + 3 orders scoring ~2분), V2 enrichment ~5분. 총 < 15분.

### Session-reset handoff checklist

Next session 시작 시:
1. 이 Day 5 섹션 먼저. Day 4 Task 1 결과 (decomposition falsified) 와 Day 4 nuisance 성공의 **재해석** (lineage axis 제거였음) 만 기억하면 됨.
2. 서버에서 STEP 5 결과 `results/v2_day5/bm/per_tf.csv` 받아옴.
3. **Decision**:
   - lineage TFs 2/3 이상 NES ≥ +0.5 on `gb_drift_out_o{1|2|3}` AND NFKB1/HSF1 > 0 유지 → **가설 (A) 확정** → "GeoBRIDGE scores are reliable after removing pseudotime-drift axis" 라는 scoped claim 으로 writeup 시작.
   - 실패 → 가설 (B) velocity phase mismatch 로 넘어가서 STEP 6 (ODE γ sweep) 구현.
   - Marginal (NES 0 ~ +0.3 범위) → 가설 (A) 일부만 기여, drift 제거 + phase correction 합쳐야 할 수도 있음. 이 경우 STEP 5 + STEP 6 결합 실험.
4. **중요 negative constraints (금지 사항)**: GNN / IRM / contrastive / learned nuisance NN 전부 여전히 금지. STEP 5/6 전부 mechanistic minimal test. Method paper redesign 금지.

### Artifacts committed this session

- [validation/scripts/run_pt_drift_projection.py](validation/scripts/run_pt_drift_projection.py) (STEP 5)
- 이 `real_day2.md` 의 Day 5 섹션

---

# GeoBRIDGE validation — Day 4 status (2026-04-21 deep night)

> Day 4 섹션 이하는 Day 5 의 전제/배경. Day 5 읽고 필요시 참조.

---

## Day 4 FINAL plan — three decisive steps, no branching

**Session context reset OK from this section.** 이 섹션만 읽으면 다음 세션에서 이어서 돌릴 수 있음. 아래는 결정된 실행 계획 전체.

### What we know now (settled)

1. 원인은 raw `M = ΔX^T ΔV` 자체의 sign flip. Lineage TFs (GATA2/KLF1/NFE2) V2 ChIP negative가 실재 (tier-A curation artifact 아님).
2. Projection trick (top-k SVD 유무, onto/out/none)은 해결 못함 — Day 3 Task 1 실험에서 clean falsification (Task 1 결과는 아래 archive 섹션).
3. GeoBRIDGE는 orthogonal component (NFKB1, HSF1 등 stress/IEG)는 잡음. 전체 GRN은 못 잡음.
4. 따라서 **ML heavy idea (GNN/IRM/contrastive) 전부 premature**. 남은 건 mechanistic minimal test 뿐.

### STEP 1 — Nuisance-only basis projection (final falsification)

목적: blind SVD 대신 explicit confounder basis (cell cycle S/G2M + library size + n_genes + mito + ribo) 를 projecting out 했을 때 lineage TFs 복구되는지 확인. 실패 예상이지만 projection-space 접근 닫기 위해 반드시 한 번 돌림.

구현: `validation/scripts/run_nuisance_projection.py` (이전 commit의 `nuisance_scoring.py`를 이 이름으로 rename + ribo/n_genes feature 추가). Pair-space `ΔX ≈ ΔN @ B` OLS 로 `V_nuis` 만들고 `(I − V_nuis V_nuis^T) M` 로 scoring. Mode `out`이 main test; `onto`/`none`은 진단용.

### STEP 2 — Early-pseudotime restriction (mechanism test)

목적: late/committed cell 에서 TF plateau + target overshoot 가 발생해서 `ΔX_TF · ΔV_target < 0` 을 만든다는 가설 검증. Pseudotime 첫 4분위수 (`pt ≤ Q1`) cells 만 남긴 뒤 production operator 그대로 재계산.

구현: `validation/scripts/run_early_pt_geobridge.py`. `--pt-quantile` (default 0.25) 로 restriction 조절. Filtered subset 위에서 `precompute_pairspace_stats_streaming` → `select_k_by_tail_exhaustion` auto-k → `scores_from_streaming_stats(mode="out")`.

### STEP 3 — Decision (strict, unambiguous)

V2 ChIP lineage TFs **GATA2 / KLF1 / NFE2** NES 기준:

| 결과 | 판정 |
|---|---|
| 위 3개 중 **2 이상에서 NES ≥ +0.5** AND stress TFs (NFKB1/HSF1) 가 부정 안됨 | **STRONG SUCCESS** → regime/formulation 확보 → writeup 진행 |
| lineage TFs 여전히 negative, 또는 미세 개선만 (< +0.5) | **FAILURE** — projection-space 전체 종료 |

"marginal improvement"는 failure 로 간주. Step 1, 2 **둘 다 strong success 조건 미달** 이면 STEP 4로 넘어가기 전에 현재 formulation은 중단.

### STEP 4 — Single dataset regime test (only if STEP 1–2 완료 후)

**정확히 1개 데이터셋**: Datlinger 2017 T-cell activation (CROP-seq). Non-commitment / activation regime 이라 dynamic/stress TF가 주도하는 best-case.

Dentategyrus, PBMC, 기타 다중 데이터셋은 **run 안함**.

Decision:
- Datlinger 에서 correlation 대비 clear advantage → regime-conditional method 로 reframe.
- 여전히 fail → GRN estimator로서 fundamentally invalid.

### STEP 5 — Ensemble (optional, STEP 4 에서 signal 있을 때만)

`correlation + geobridge` (+ optionally grnboost) rank fusion. Simple weighted average 수준. Main method 아니고 fallback publication path.

### DO NOT RUN (record for future self)

- GNN / message passing reformulation
- IRM, environment-based causal ML
- contrastive learning losses
- learned nuisance basis via NN
- velorama + geobridge joint fusion — Datlinger signal 확인 전까지 금지

### CLI — 오늘 밤 서버 실행 순서 (전부 CPU, 순차적)

```bash
cd /data/cb/scratch/seokmin/geobridge/validation && git pull

# ---- STEP 1: nuisance projection on BM ----
python scripts/run_nuisance_projection.py --config configs/bm.yaml \
    --out results/rankings_nuis/

# ---- STEP 2: early-pt restriction on BM (first quartile) ----
python scripts/run_early_pt_geobridge.py --config configs/bm.yaml \
    --pt-quantile 0.25 --out results/rankings_earlyPT/

# ---- STEP 3: V1 CollecTRI + V2 ChIP evaluation (same 8 diagnostic TFs) ----
# bundle correlation baseline + both new rankings into one eval dir
mkdir -p results/rankings_day4/bm
cp results/rankings/correlation_bm.parquet             results/rankings_day4/bm/
cp results/rankings_nuis/bm/gb_nuis_out_*.parquet      results/rankings_day4/bm/
cp results/rankings_nuis/bm/gb_nuis_none_*.parquet     results/rankings_day4/bm/
cp results/rankings_earlyPT/bm/gb_earlyPT_q25_*.parquet results/rankings_day4/bm/

python scripts/run_v1_enrichment.py \
    --rankings "results/rankings_day4/bm/*_bm.parquet" \
    --gt data/gt/dorothea_tierA.tsv \
    --out results/v1_day4/bm/

python scripts/run_v1_enrichment.py \
    --rankings "results/rankings_day4/bm/*_bm.parquet" \
    --gt data/gt/chip_subset_bm.tsv --max-size 2000 \
    --out results/v2_day4/bm/

# ---- Step 3 readout: send these two files back ----
cat results/v2_day4/bm/per_tf.csv
cat results/v1_day4/bm/per_tf.csv
```

### What Seokmin checks in the output (decision SOP)

1. `results/v2_day4/bm/per_tf.csv` 열어서 lineage TFs 3개 (`GATA2`, `KLF1`, `NFE2`) 의 NES 컬럼 확인.
2. 각 TF 행에서 methods `gb_nuis_out_d{d}` 과 `gb_earlyPT_q25_k{K}` 가 둘 다 존재하는지 체크.
3. **Strong success**: 위 두 methods 중 적어도 한쪽에서 `GATA2/KLF1/NFE2` 세 TF 중 **2 이상 NES ≥ +0.5** AND `NFKB1/HSF1` row에서 NES > 0 유지.
4. 아니면 failure → STEP 4 (Datlinger) 로 이동.

### Artifacts committed this session

- `validation/scripts/run_nuisance_projection.py` (STEP 1)
- `validation/scripts/run_early_pt_geobridge.py` (STEP 2)
- `validation/scripts/gsp_diagnostic.py` (reference; Day 4 interpretive, STEP 1-3 decision에는 사용 안 함)
- 이 `real_day2.md` Day 4 섹션

---

## Day 4 archive — decomposition falsified (Task 1 결과 기록)

### Task 1 결과 (decomposition 실험, k=2 / k=17)

decompose_scoring.py 로 `gb_none`(unprojected M), `gb_onto`(`Pk M`), `gb_out`(`(I−Pk) M`) 세 버전을 같은 `pre`에서 뽑아서 BM V1 CollecTRI + V2 ChIP 모두 평가.

**k=17 (auto), V2 ChIP 8 diagnostic TFs**: lineage TFs 복구 실패. `gb_onto` NES GATA2/KLF1/NFE2 ≈ `gb_out` ≈ `gb_none` (전부 음수 또는 약한 양수). `gb_none` global mean_nes=0.20 > `gb_onto`=0.14 > `gb_out`=0.08, 그러나 lineage TFs 기준으로는 셋 다 fail.

**k=2, V2 ChIP 재검 (cleaner regime)**: 7/7 test 가능한 TF 중 6개에서 `gb_none ≈ gb_onto ≈ gb_out` (±0.15 NES 이내). 남은 1개도 Pk·M과 (I−Pk)·M이 서로 sign 반대로 갈라지는 게 아니라 같은 방향으로 함께 약해짐.

**결론**: decomposition 가설 **clean falsification**.
- Sign flip이 projection operator에서 발생하는 게 아니라 이미 **raw `M = ΔXc^T ΔV` 자체**에 들어있음.
- Pk와 I−Pk 어느 subspace에서 보든 M의 lineage TF row는 lineage target 방향으로 음수임.
- Day 3의 "production operator가 lineage를 projecting out해서 flip된다"는 mechanistic statement는 **부정확**. 실제 mechanism은 velocity ΔV가 lineage TF 자신의 ΔX와 pair-space 상에서 어긋나 있다는 더 근본적인 현상.

### 남은 projection-space 변종 — Option C (nuisance-only) 왜 그래도 돌리는가

decomposition이 falsified 된 이상 "더 좋은 subspace를 고르면 해결된다" 는 희망은 약하다. 그러나 한 variant가 아직 안 돌았음: **data-driven top-k SVD** 대신 **biologically-motivated low-d nuisance basis** (cell cycle + library size + mito) 를 projecting out.

Prior prediction (honest): 실패 확률 높음. 이유 — `gb_none`에 이미 sign flip이 있고, small-rank projection(d≈3–4)은 M을 약간만 흔들기 때문에 sign을 뒤집을 만한 힘이 없다. **그럼에도 돌리는 이유**:
1. 싸다 (수 분).
2. "projection operator space" 자체를 닫는 falsification. 실패하면 "operator 문제가 아니라 raw M 문제" 라는 진단이 완성됨.
3. 작은 확률로, 만약 sign-flip 원인이 cell-cycle confounding 이었으면 성공할 수 있음 (hematopoiesis에서 G2M/S activity는 분화 축과 크게 어긋나지 않지만 완전히 같지도 않음).

**Decision rule (strict)**: lineage TFs 중 GATA2/KLF1/NFE2 가 **V2 ChIP에서 NES ≥ +0.5** 셋 중 둘 이상에 도달해야 success. 그보다 약하면 falsification 완료 → projection-space 접근 전부 종료.

### Option C 구현 (`validation/scripts/nuisance_scoring.py`)

핵심 steps:
1. **Nuisance features** (adata.obs level): Tirosh 2016 S_score / G2M_score (case-insensitive + mouse capitalize fallback), log_total_counts, pct_mito. Mito prefix MT-/Mt- 자동 탐지.
2. **Pair-space regression**: pseudotime-ordered 에서 forward pair 100k 샘플, `ΔX ≈ ΔN @ B` OLS. `B.T` 를 QR 해서 gene-space orthonormal basis `V_nuis ∈ R^{G×d}` (d≤4).
3. **Scoring** (`_scores_with_vnuis`): `scores_from_streaming_stats` 구조를 그대로 따라가되 SVD Vk 대신 V_nuis 삽입. Mode `out`(`(I−P_nuis) M`), `onto`(`P_nuis M`), `none`(M, sanity).
4. **project_y 끔**: y-side residualization 공식은 SVD spectrum Σ 가정이 있어 V_nuis 에 그대로 적용 불가. 단순하게 `diag(Y)` 사용.
5. 저장: parquet + `vnuis_<dataset>.npz` (V_nuis + column names, 재현용).

### GSP diagnostic (`validation/scripts/gsp_diagnostic.py`) — interpretation only

목적: TF별 expression이 velocity graph 상에서 어떤 주파수 band에 앉는지 확인. **알고리즘 변경 아님**, 진단/시각화 용.

Pipeline:
1. `W = 0.5*(A + A^T)`, A = `adata.uns["velocity_graph"]` (cosine sim transition). 자기루프 제거.
2. `L = I − D^{-1/2} W D^{-1/2}`, sparse sym.
3. `scipy.sparse.linalg.eigsh(L, k=50, which="SM")` → 작은 50개 eigenpair.
4. 각 TF j 에 대해 expression vector x_j ∈ R^{n_cells} 를 center, basis 에 project → power spectrum `p_k = |⟨u_k, x_j⟩|^2 / ||x_j||^2`. low(bottom 20%) / mid(20–60%) / high(top 40%) band fraction per TF.
5. 출력: `spectral_summary.csv` (tf, low/mid/high_frac, peak_idx) + `cumulative_power.png` + `spectra.npz`.

기대: lineage TFs (GATA2/KLF1/NFE2) 가 low-frequency 쪽에 집중. Production operator 가 top-k SVD basis (≈ low-freq graph modes 와 상관) 를 projecting out 하므로 그들의 signal 이 denominator 에 비해 작아진다는 시각적 확인. NFKB1/HSF1 은 mid/high 쪽으로 흩어질 것. — 단 이건 raw expression 의 spectral localization 이지 M 자체의 localization 은 아님. 상관 관계 level 의 증거만 제공.

### 검증받을 요소들 (Seokmin 말한 내용 검토)

원문 claim → 내 review:
- "Stress/IEG TFs succeed because orthogonal to dominant geometry" — **부분적으로만 맞음**. Day 3 mechanism 은 decomposition 실험으로 부정되었다. 정확히는 "stress TFs 는 M 에서 애초에 부호가 양수고 lineage TFs 는 음수임". geometry orthogonality 는 해석 프레임이지 증명된 mechanism 아님.
- "data-driven top-k SVD subspace inevitably aligns with lineage axis and removes signal" — **일부 맞지만 필요조건은 아님**. k=0 (projection 없음) 인 `gb_none` 에서도 이미 lineage fail. "projection 이 없으면 signal 복구" 가 틀렸음.
- "nuisance-only projection 이 structural confounders 만 제거해서 causal signal 보존" — **too optimistic**. 위 prediction 참고. 돌려보되 기대치 낮게.
- Decision rule (ΔNES ≥ 0 이면 success) — **너무 느슨함**. `gb_none` 이 이미 ΔNES ≈ 0 인 lineage TF 가 여러 개. 더 strict 기준 (NES ≥ +0.5 absolute 가 lineage TFs 2/3) 으로 상향.

### 추가 아이디어 (run 안 할 거지만 기록)

- **Early-pseudotime pair restriction**: committed 세포 (pt > 0.5) 만 뺀 progenitor-only pair subset 에서 재계산. Commitment axis 가 안 형성된 구간에서는 ΔV 가 lineage gene 과 어긋나지 않을 가능성. 30 분 정도면 돌릴 수 있고 negative 면 "method 는 mature bifurcation 이후 실패" 라는 clean scope statement 가 됨. Nuisance + GSP 결과 보고 판단.

### 실행 CLI (server, 전부 CPU)

```bash
cd /data/cb/scratch/seokmin/geobridge/validation && git pull

# [Step 1] Option C nuisance scoring on BM (main test)
python scripts/nuisance_scoring.py --config configs/bm.yaml \
    --out results/rankings_nuis/

# [Step 2] V1 CollecTRI + V2 ChIP eval (동일 8 TFs)
cp results/rankings/correlation_bm.parquet results/rankings_nuis/bm/
python scripts/run_v1_enrichment.py \
    --rankings "results/rankings_nuis/bm/*_bm.parquet" \
    --gt data/gt/dorothea_tierA.tsv \
    --out results/v1_nuis/bm/

python scripts/run_v1_enrichment.py \
    --rankings "results/rankings_nuis/bm/*_bm.parquet" \
    --gt data/gt/chip_subset_bm.tsv --max-size 2000 \
    --out results/v2_nuis/bm/

# [Step 3] GSP diagnostic — 8 diagnostic TFs
python scripts/gsp_diagnostic.py --config configs/bm.yaml --K 50 \
    --tfs GATA2 KLF1 NFE2 RUNX1 NFKB1 HSF1 KLF2 KLF4 \
    --out results/gsp_diag/

# [Step 4] 결과 송출
cat results/v1_nuis/bm/per_tf.csv
cat results/v2_nuis/bm/per_tf.csv
cat results/gsp_diag/bm/spectral_summary.csv
```

런타임 예상: nuisance scoring 수 분, enrichment 2×수 분, gsp eigsh 수 분. 총 < 30분.

### Session-reset handoff checklist

Next session 시작 시:
1. 이 Day 4 섹션 먼저 읽고, Day 3/Day 2 는 배경으로만 참고.
2. Server 결과 per_tf.csv × 2 + spectral_summary.csv + cumulative_power.png 수합 후 공유.
3. **Decision rule 적용**:
   - lineage TFs 2/3 이상에서 `gb_nuis_out` V2 ChIP NES ≥ +0.5 → Option C 성공 → Option A/C 비교 writeup 으로 진행.
   - 미달 → projection-space 접근 전부 종료. GSP 결과 + nuisance fail 을 합쳐 "sign flip은 raw M 자체에 있음" 이라는 완결된 진단을 Day 5 에 기록. Early-pseudotime idea 로 넘어가거나, 아예 다른 데이터셋 (pancreas 이외) 에서 sign flip 재현 여부 확인.
4. Pushed commits: nuisance_scoring.py, gsp_diagnostic.py, 이 Day 4 섹션.

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
