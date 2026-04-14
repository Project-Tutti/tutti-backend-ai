# 🎵 AI 개발자 가이드 — Tutti 편곡 엔진

> **대상 독자**: AI/ML 개발자 (@sonicwarp)
> **마지막 업데이트**: 2026-04-14

---

## 1. 프로젝트 구조

```
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
│   └── midi_writer.py    # MIDI 파일 저장 (save_midi)
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

| 영역 | 자유롭게 수정 가능? | 조건 |
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

# GPU 서버 (CUDA 있는 환경)
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

## 5. 실험 추적 (MLflow)

### 5-1. MLflow 서버

- **URL**: `http://localhost:5000` (GPU 서버 로컬)
- **외부 접속**: `https://mlflow.tutti.asia` (Cloudflare 인증 필요)

### 5-2. 학습 코드에 MLflow 연동

```python
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("arrangement-v5")

with mlflow.start_run(run_name="top-p-sampling"):
    # 파라미터 기록
    mlflow.log_param("temperature", 1.0)
    mlflow.log_param("top_p", 0.9)
    mlflow.log_param("learning_rate", 0.001)

    # 학습 루프
    for epoch in range(100):
        loss = train_one_epoch(model, dataloader)
        mlflow.log_metric("train_loss", loss, step=epoch)

    # 체크포인트 저장
    mlflow.log_artifact("checkpoints/best.safetensors")

    # 추론 품질 메트릭
    mlflow.log_metric("avg_note_density", 4.2)
    mlflow.log_metric("pitch_range_coverage", 0.85)
```

### 5-3. 추론 테스트에 MLflow 연동

```python
with mlflow.start_run(run_name="inference-test-pop"):
    mlflow.log_param("test_midi", "pop_song_01.mid")
    mlflow.log_param("target", "keyboard")
    mlflow.log_param("temperature", 1.0)

    result = run_arrangement(...)

    mlflow.log_artifact(result)  # 결과 MIDI 파일 기록
    mlflow.log_metric("inference_time_sec", 45.3)
```

---

## 6. 데이터 관리 (DVC)

### 6-1. 학습 데이터 추가

```bash
# 새 MIDI 데이터셋 추가
cp -r ~/new_midi_files/ data/midi_dataset/

# DVC 추적
dvc add data/midi_dataset/

# Git 커밋 (메타데이터만)
git add data/midi_dataset.dvc .gitignore
git commit -m "data: MIDI 데이터셋 v2 (50곡 추가)"
```

### 6-2. 데이터 복원

```bash
git pull
dvc pull    # ~/dvc_store에서 실제 데이터 복원
```

---

## 7. 워크플로우 요약

```
┌─────────────────────────────────────────────────────┐
│  일반적인 개발 사이클                                    │
│                                                     │
│  1. ai_core/ 코드 수정                                │
│  2. pytest 테스트 실행                                 │
│  3. GPU 서버에서 통합 추론 테스트                         │
│  4. MLflow에 실험 기록                                 │
│  5. PR 생성 → CI 테스트 통과 → 코드 리뷰                 │
│  6. main 머지 → 자동 Docker 빌드 → 서버 배포             │
└─────────────────────────────────────────────────────┘
```

---

## 8. 자주 묻는 질문

### Q: `app/` 폴더의 코드를 수정해야 하나요?
**아닙니다.** `app/services/inference.py`는 `ai_core/`를 re-export하는 facade입니다. AI 로직은 전부 `ai_core/`에 있습니다.

### Q: 새 Python 패키지가 필요하면?
`pyproject.toml`의 `dependencies`에 추가하고 PR을 보내세요. 인프라 담당자가 Docker 빌드에 반영합니다.

### Q: 모델 체크포인트는 어디에?
GPU 서버의 `~/tutti-backend-ai/models/` 디렉토리. `registry.json`에 등록해야 워커가 로드합니다. 자세한 내용은 `docs/AI_ML_GUIDE.md`를 참조하세요.

### Q: worker.py가 뭐하는 건가요?
Redis에서 편곡 요청을 받아서 `ai_core.arrangement.run_arrangement()`를 호출하고, 결과 MIDI를 백엔드에 콜백으로 보냅니다. 건드릴 필요 없습니다.
