# 🎵 AI 개발자 가이드 — Tutti 편곡 엔진

> **대상 독자**: AI/ML 개발자 (@sonicwarp)
> **마지막 업데이트**: 2026-04-14

---

## 1. 프로젝트 구조

```text
tutti-backend-ai/
├── ai_core/              ← 🎯 AI 개발자 작업 영역
│   ├── constants.py      # 악기 그룹, PROGRAM_TO_REP, DROP_SET
│   ├── vocab.py          # 토큰 사전 (build_v5_vocab)
│   ├── model_loader.py   # 체크포인트 로드 (load_model)
│   ├── tokenizer.py      # MIDI → 토큰 변환
│   ├── generator.py      # 토큰 생성 (슬라이딩 윈도우 추론)
│   ├── decoder.py        # 토큰 → 노트 이벤트 디코딩
│   ├── postprocess.py    # 노트 후처리 (피치 클리핑, 모노포닉 등)
│   ├── arrangement.py    # 오케스트레이션 (run_arrangement 진입점)
│   ├── midi_writer.py    # MIDI 파일 저장 (save_midi)
│   ├── metrics.py        # 품질 메트릭 (기본 통계 + 음악적 평가)
│   ├── evaluate.py       # 모델 평가 CLI + MLflow 기록
│   └── tune.py           # Optuna 하이퍼파라미터 자동 튜닝
│
├── contracts/            ← ⚠️ 양쪽 합의 필요 (시그니처 변경 시)
│   └── interfaces.py     # Protocol 정의 (ArrangementEngine 등)
│
├── app/                  ← 🔒 인프라 영역 (수정 불필요)
│   └── services/
│       └── inference.py  # Facade — ai_core를 re-export
│
├── worker.py             ← 🔒 Redis 워커 (인프라)
├── tests/                ← 테스트
│   ├── test_ai_core.py   # AI Core 단위 테스트
│   └── test_contracts.py # 계약 준수 테스트
└── pyproject.toml        ← 의존성 관리 (uv)
```

### 핵심 규칙

| 영역 | 수정 가능? | 조건 |
|------|:---:|------|
| `ai_core/` | ✅ | `contracts/interfaces.py` 시그니처 유지 |
| `contracts/` | ⚠️ | 인프라 담당자와 합의 후 변경 |
| `app/`, `worker.py` | ❌ | 수정 불필요, 건드리지 마세요 |

---

## 2. 로컬 개발 환경 설정

### 2-1. uv 설치 (패키지 매니저)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# 설치 확인
uv --version
```

### 2-2. 의존성 설치

```bash
cd ~/tutti-backend-ai

# GPU 서버 (전체 — 추론 + MLflow + Optuna)
uv sync --extra gpu --extra mlflow --group test

# GPU 서버 (기본 — 추론만)
uv sync --extra gpu --group test

# CPU만 있는 로컬 (torch 설치 안 함 — 테스트/린트용)
uv sync --group test
```

### 2-3. IDE 설정 (PyCharm)

1. **Remote Interpreter** 설정:
   - SSH로 GPU 서버 연결
   - Python 인터프리터: `/path/to/.venv/bin/python`

2. **Source Root** 설정:
   - 프로젝트 루트(`tutti-backend-ai/`)를 Source Root로 지정
   - `ai_core`와 `contracts`가 import 가능하도록 설정

---

## 3. 코드 수정 가이드

### 3-1. 추론 로직 변경

`ai_core/` 내부 모듈을 자유롭게 수정하세요. 예시:

```python
# ai_core/generator.py — 샘플링 전략 변경
def sample_next_token(logits, temperature=1.0, top_k=50):
    # 기존 top-k 대신 top-p (nucleus) 샘플링으로 변경
    ...
```

### 3-2. 시그니처 변경이 필요한 경우

`run_arrangement()`에 새 파라미터를 추가하고 싶다면:

1. `contracts/interfaces.py`의 Protocol 업데이트
2. `ai_core/arrangement.py` 구현 수정
3. **PR 생성** → 인프라 담당자(@eddy81848)가 워커 호출부도 함께 수정

```python
# contracts/interfaces.py
class ArrangementEngine(Protocol):
    def run_arrangement(
        self,
        ...,
        top_p: float = 0.9,  # ← 새 파라미터 (기본값 필수!)
    ) -> str: ...
```

> 💡 **기본값이 있으면** 기존 워커가 수정 없이도 동작합니다.

---

## 4. 테스트

### 4-1. 단위 테스트 실행 (GPU 불필요)

```bash
# ai_core 테스트 (상수, vocab, postprocess 등)
uv run pytest tests/test_ai_core.py -v

# 계약 준수 테스트 (시그니처 검증)
uv run pytest tests/test_contracts.py -v

# 전체
uv run pytest -v
```

### 4-2. 통합 테스트 (GPU 서버에서)

```python
# test_inference_local.py — GPU 서버에서만 실행
from ai_core.arrangement import run_arrangement
from ai_core.model_loader import load_model
from ai_core.vocab import build_v5_vocab

# 모델 로드
vocab = build_v5_vocab()
vocab_r = {v: k for k, v in vocab.items()}
model, device = load_model(
    ckpt_path="/path/to/checkpoint.safetensors",
    vocab_size=len(vocab),
    vocab=vocab,
    device="cuda:0",
)

# 편곡 실행
result = run_arrangement(
    song_path="test_input.mid",
    target="keyboard",
    genre="POP",
    temperature=1.0,
    pitch_min=21,
    pitch_max=108,
    output_path="test_output.mid",
    model=model,
    vocab=vocab,
    vocab_r=vocab_r,
    device=device,
)
print(f"결과 파일: {result}")
```

### 4-3. CI 테스트 게이트

PR을 올리면 GitHub Actions에서 자동으로:

1. `test_ai_core.py` 실행 (GPU 없이)
2. `test_contracts.py` 실행 (시그니처 검증)
3. 통과해야 빌드 진행

---

## 5. 모델 평가 (evaluate)

편곡 결과의 품질을 자동 측정합니다. 결과는 터미널 + MLflow에 기록됩니다.

### 5-1. 단일 파일 평가

```bash
python -m ai_core.evaluate \
    --source ~/midi_data/original_song.mid \
    --generated ~/results/song_violin.mid
```

### 5-2. 디렉토리 배치 평가

같은 이름의 원본-결과 MIDI 쌍을 자동 매칭합니다:

```bash
python -m ai_core.evaluate \
    --source ~/midi_data/originals/ \
    --generated ~/midi_data/results/ \
    --experiment "violin-model-v2"
```

### 5-3. MLflow 기록 끄기

```bash
python -m ai_core.evaluate \
    --source song.mid --generated song_violin.mid \
    --no-mlflow
```

### 5-4. 측정되는 메트릭

**기본 통계 (Stage 1)** — 생성 MIDI만으로 측정

| 메트릭 | 의미 |
| ------ | ---- |
| `note_count` | 생성된 총 노트 수 |
| `pitch_range` | 최고음 - 최저음 (반음 단위) |
| `pitch_mean` | 평균 음정 |
| `avg_velocity` | 평균 벨로시티 |
| `density_per_sec` | 초당 노트 밀도 |

**음악적 평가 (Stage 2)** — 원본과 비교 (논문 기반)

| 메트릭 | 의미 | 좋은 방향 | 출처 |
| ------ | ---- | :-------: | ---- |
| `chord_accuracy` | 코드 구성음 일치율 | ↑ | NeurIPS 2024 |
| `pch_similarity` | 조성 분포 코사인 유사도 | ↑ | AccoMontage |
| `doa` | 편곡 창의성 (음고 다양성) | ↑ | NeurIPS 2024 |
| `dissonance_rate` | 동시 발음 불협화 비율 | ↓ | AccoMontage2 |

### 5-5. MLflow에서 결과 확인

1. `https://mlflow.tutti.asia` 접속 (또는 `http://localhost:5000`)
2. 좌측에서 실험 선택 (기본: `arrangement-eval`)
3. 실행 목록에서 메트릭 비교, 차트 생성 가능
4. 결과 MIDI 파일도 아티팩트에 첨부됨

---

## 6. 하이퍼파라미터 자동 튜닝 (Optuna)

Optuna가 편곡 파라미터를 자동 탐색하고, **모든 트라이얼을 MLflow에 기록**합니다.

### 6-1. 기본 실행

```bash
# keyboard 타겟, chord_accuracy 최대화, 30회 탐색
python -m ai_core.tune \
    --source ~/midi_data/test_songs/ \
    --target keyboard \
    --n-trials 30
```

### 6-2. 최적화 대상 변경

```bash
# 불협화도 최소화
python -m ai_core.tune \
    --source ~/midi_data/ \
    --target violin \
    --optimize dissonance_rate

# 편곡 창의성 최대화
python -m ai_core.tune \
    --source ~/midi_data/ \
    --target keyboard \
    --optimize doa
```

### 6-3. 기존 탐색 이어서 실행

```bash
# 이전 스터디에 50회 추가
python -m ai_core.tune \
    --source ~/midi_data/ \
    --target keyboard \
    --study-name keyboard-tuning-v1 \
    --n-trials 50
```

### 6-4. ⚡ 빠른 탐색 전략

전체 데이터로 튜닝하면 시간이 오래 걸립니다. **3가지 전략**을 독립적으로 또는 조합하여 **5~10배 빠르게** 경향성을 파악할 수 있습니다.

```bash
# 🚀 최고 속도: 모든 최적화 ON
python -m ai_core.tune \
    --source ~/midi_data/ \
    --target keyboard \
    --n-trials 100 \
    --pruning \
    --sample-ratio 0.1 \
    --max-files 5
```

#### A. 조기 종료 (Pruning) — `--pruning`

곡 1~2개 처리 후 중간 점수가 다른 트라이얼보다 나쁘면 **즉시 나머지 곡을 건너뛰고** 다음 파라미터 조합으로 넘어갑니다.

| 항목 | 내용 |
| ---- | ---- |
| 작동 방식 | MedianPruner가 매 곡 처리 후 누적 평균을 비교 |
| 절약 효과 | 50~70% 시간 절약 (최악의 조합을 1~2곡에서 배제) |
| 제약 조건 | 소스 곡 **5개 이상**에서 효과적 |
| 주의점 | 처음 3트라이얼은 프루닝하지 않음 (기준 축적 warmup) |

```text
동작 예시:
  Trial A:  곡1=0.7  곡2=0.6  곡3=0.5  → 평균 0.60 (완료)
  Trial B:  곡1=0.3  → 중간값(0.7)보다 나쁨 → ✂ 프루닝! (30초 절약)
  Trial C:  곡1=0.8  곡2=0.7  → 계속...
```

```bash
python -m ai_core.tune \
    --source ~/midi_data/ \
    --target violin \
    --n-trials 50 \
    --pruning
```

> ⚠️ 프루닝된 트라이얼은 '불완전' 상태로 기록됩니다. 최종 best_trial에는 포함되지 않습니다.

#### B. 데이터 샘플링 — `--sample-ratio`

전체 소스 MIDI 중 **N%만 랜덤 선택**하여 사용합니다. 트라이얼마다 다른 서브셋을 뽑아 과적합을 방지합니다.

| 항목 | 내용 |
| ---- | ---- |
| 작동 방식 | 매 트라이얼마다 전체의 N%를 랜덤 셔플 추출 |
| 절약 효과 | 0.2 = 5배, 0.1 = 10배 빨라짐 |
| 제약 조건 | 샘플이 편향될 수 있음 (트라이얼마다 다른 셔플로 보완) |
| 추천 비율 | 탐색: 0.1~0.3 / 검증: 1.0 (전체) |

```bash
# 전체 곡의 10%만 사용 (100곡 → ~10곡)
python -m ai_core.tune \
    --source ~/midi_data/ \
    --target keyboard \
    --n-trials 50 \
    --sample-ratio 0.1
```

#### C. 파일 수 제한 — `--max-files`

트라이얼당 **최대 N개 파일만** 평가합니다. 실행 시간을 예측 가능하게 만듭니다.

| 항목 | 내용 |
| ---- | ---- |
| 작동 방식 | 평가 파일 수를 하드 리밋으로 제한 |
| 절약 효과 | 5곡 제한 시 → N곡 × ~30초 × 트라이얼 수 = 예측 가능한 시간 |
| 제약 조건 | 평가 곡 수가 적으면 일반화 약함 |
| 추천 값 | 빠른 탐색: 5곡 / 정밀 검증: 제한 없음 |

```bash
# 최대 5곡만 평가 (예측 시간: 5곡 × 30초 × 30회 ≈ 75분)
python -m ai_core.tune \
    --source ~/midi_data/ \
    --target keyboard \
    --n-trials 30 \
    --max-files 5
```

### 6-5. 추천 워크플로우

```text
┌──────────────────────────────────────────────────┐
│  1단계: 빠른 탐색 (30분~1시간)                       │
│    --pruning --sample-ratio 0.1 --max-files 5    │
│    --n-trials 100                                │
│    → 넓은 범위에서 유망한 파라미터 영역 발견             │
│                                                  │
│  2단계: 좁은 범위 탐색 (선택)                         │
│    1단계 결과를 보고 tune.py의 TUNABLE_PARAMS        │
│    범위를 좁힌 뒤 다시 실행                           │
│                                                  │
│  3단계: 전체 데이터 검증                              │
│    --pruning --n-trials 20 (샘플링/파일 제한 없이)    │
│    → 최종 파라미터를 arrangement.py에 적용            │
└──────────────────────────────────────────────────┘
```

### 6-6. 탐색 파라미터

| 파라미터 | 탐색 범위 | 설명 |
| -------- | --------- | ---- |
| `temperature` | 0.5 ~ 1.5 | 샘플링 랜덤성 (높을수록 다양) |
| `top_p` | 0.8 ~ 0.99 | Nucleus 샘플링 확률 커트라인 |
| `rest_penalty` | 0.5 ~ 3.0 | 쉼표 토큰 생성 억제 강도 |
| `window_bars` | 4 ~ 16 | 슬라이딩 윈도우 크기 (마디) |
| `context_bars` | 4 ~ 16 | 이전 컨텍스트 마디 수 |
| `fade_bars` | 4 ~ 16 | 윈도우 겹침 페이드 길이 |

### 6-7. MLflow에서 결과 확인

```text
📂 optuna-tuning (실험)
└── 🏃 tuning-tune-keyboard-chord_accuracy (상위 런)
    ├── params: target=keyboard, n_trials=30, pruning=true
    ├── metrics: best_chord_accuracy=0.82
    │
    ├── 🏃 trial-0-song1.mid (개별 런)
    │   ├── params: temperature=0.85, top_p=0.92, ...
    │   └── metrics: chord_accuracy=0.71, dissonance_rate=0.18
    │
    ├── 🏃 trial-1-song1.mid
    └── ...
```

실행 완료 후 **최적 파라미터와 프루닝 통계**가 터미널에 출력됩니다:

```text
═══════════════════════════════════════════
  🏆 최적 하이퍼파라미터
═══════════════════════════════════════════
  📊 트라이얼 통계:
     완료: 18  |  프루닝: 12  |  실패: 0
     → 프루닝으로 ~40% 시간 절약!

  Trial #17
  chord_accuracy: 0.8234

      temperature = 0.85
            top_p = 0.93
     rest_penalty = 1.75
      window_bars = 8
     context_bars = 8
        fade_bars = 8

  → arrangement.py에 아래 값을 적용하세요
═══════════════════════════════════════════
```

### 6-8. 전체 CLI 옵션

```bash
python -m ai_core.tune --help

필수:
  --source           소스 MIDI (파일 또는 디렉토리)
  --target           타겟 악기 (keyboard, violin 등)

기본 옵션:
  --genre            장르 (기본: CLASSICAL)
  --model-type       모델 타입 (기본: registry 기본값)
  --n-trials         탐색 횟수 (기본: 30)
  --optimize         최적화 메트릭 (기본: chord_accuracy)
  --direction        최적화 방향 (maximize|minimize, 기본: 자동)

최적화 전략:
  --pruning          조기 종료 ON (소스 5곡 이상에서 효과적)
  --sample-ratio     데이터 샘플링 비율 (0.1 = 10%, 기본: 1.0)
  --max-files        트라이얼당 최대 평가 파일 수 (기본: 제한 없음)

저장 및 기록:
  --study-name       Optuna 스터디 이름 (기존 이어서 탐색)
  --experiment       MLflow 실험 이름 (기본: optuna-tuning)
  --db-path          SQLite DB 경로 (기본: optuna_studies.db)
  --no-mlflow        MLflow 기록 비활성화
```

### 6-9. Optuna 대시보드

튜닝 결과를 웹 UI로 시각화할 수 있습니다. **별도 터미널**에서 실행:

```bash
# 대시보드 시작 (포트 8080)
optuna-dashboard sqlite:///optuna_studies.db
```

`http://localhost:8080` 에서 확인 가능:

- **Optimization History**: 트라이얼별 점수 추이
- **Parallel Coordinate**: 파라미터 간 관계 시각화
- **Parameter Importance**: 어떤 파라미터가 가장 영향 큰지
- **Slice Plot**: 개별 파라미터가 점수에 미치는 영향
- **Pruning History**: 프루닝된 트라이얼 비율과 시점

> 💡 `tune.py` 실행 중에도 대시보드를 열어두면 **실시간으로 결과가 업데이트**됩니다.

---


## 7. 실험 추적 (MLflow)

### 7-1. MLflow 서버

- **로컬 접속**: `http://localhost:5000` (GPU 서버)
- **외부 접속**: `https://mlflow.tutti.asia` (Cloudflare 인증 필요)

### 7-2. 학습 코드에 직접 연동

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("arrangement-v5")

with mlflow.start_run(run_name="top-p-sampling"):
    mlflow.log_param("temperature", 1.0)
    mlflow.log_param("learning_rate", 0.001)

    for epoch in range(100):
        loss = train_one_epoch(model, dataloader)
        mlflow.log_metric("train_loss", loss, step=epoch)

    mlflow.log_artifact("checkpoints/best.safetensors")
```

> 💡 `evaluate.py`와 `tune.py`는 자동으로 MLflow에 연동되므로, 직접 `mlflow.log_*`를 호출할 필요 없습니다.

---

## 8. 데이터 관리 (DVC)

### 8-1. 학습 데이터 추가

```bash
cp -r ~/new_midi_files/ data/midi_dataset/
dvc add data/midi_dataset/
git add data/midi_dataset.dvc .gitignore
git commit -m "data: MIDI 데이터셋 v2 (50곡 추가)"
```

### 8-2. 데이터 복원

```bash
git pull
dvc pull    # ~/dvc_store에서 실제 데이터 복원
```

---

## 9. 워크플로우 요약

```text
┌──────────────────────────────────────────────────┐
│  일반적인 개발 사이클                                 │
│                                                  │
│  1. ai_core/ 코드 수정                             │
│  2. pytest 단위 테스트 실행                          │
│  3. GPU 서버에서 통합 추론 테스트                      │
│  4. python -m ai_core.evaluate 로 품질 평가          │
│  5. python -m ai_core.tune 으로 파라미터 최적화       │
│  6. MLflow (mlflow.tutti.asia) 에서 결과 비교        │
│  7. PR 생성 → CI 테스트 통과 → 코드 리뷰              │
│  8. main 머지 → 자동 Docker 빌드 → 서버 배포          │
└──────────────────────────────────────────────────┘
```

---

## 10. 자주 묻는 질문

### Q: `app/` 폴더의 코드를 수정해야 하나요?

**아닙니다.** `app/services/inference.py`는 `ai_core/`를 re-export하는 facade입니다. AI 로직은 전부 `ai_core/`에 있습니다.

### Q: 새 Python 패키지가 필요하면?

`pyproject.toml`의 `dependencies`에 추가하고 PR을 보내세요. 인프라 담당자가 Docker 빌드에 반영합니다.

### Q: 모델 체크포인트는 어디에?

GPU 서버의 `~/tutti-backend-ai/models/` 디렉토리. `registry.json`에 등록해야 워커가 로드합니다. 자세한 내용은 `docs/AI_ML_GUIDE.md`를 참조하세요.

### Q: worker.py가 뭐하는 건가요?

Redis에서 편곡 요청을 받아서 `ai_core.arrangement.run_arrangement()`를 호출하고, 결과 MIDI를 백엔드에 콜백으로 보냅니다. 건드릴 필요 없습니다.

### Q: 프로덕션 편곡에서도 품질 메트릭이 기록되나요?

**네.** 워커가 매 편곡 완료 시 자동으로 `qualityMetrics`를 콜백에 포함합니다 (chord_accuracy, dissonance_rate 등). 메인 서버 DB에 저장됩니다.

### Q: Optuna 튜닝은 얼마나 걸리나요?

곡 길이와 트라이얼 수에 따라 다릅니다. 짧은 곡 1개 × 30 트라이얼이면 대략 수십 분, 긴 곡 10개 × 50 트라이얼이면 수 시간 이상 소요될 수 있습니다. 먼저 `--n-trials 3`으로 테스트해보세요.
