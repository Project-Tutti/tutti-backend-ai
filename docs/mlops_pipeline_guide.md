# MLOps 파이프라인 종합 가이드 — Tutti AI 편곡 서비스

> **작성일**: 2026-04-14
> **대상 독자**: 이 프로젝트에 처음 참여하는 사람, MLOps 개념이 생소한 사람
> **전제 지식**: Python 기초, Git 사용 경험

---

## 0. 이 문서는 무엇인가?

이 문서는 Tutti AI 편곡 서비스에 **MLOps 파이프라인**을 왜, 어떻게 도입했는지를 처음부터 설명합니다. 코드를 어떻게 구조화했고, 어떤 도구를 왜 선택했으며, 실제로 무엇이 달라졌는지를 "개념을 모르는 사람"도 이해할 수 있도록 작성했습니다.

---

## 1. 배경: 우리가 풀고 있는 문제

### Tutti는 뭘 하는 서비스인가?

사용자가 MIDI 파일(악보)을 업로드하면, **AI가 새로운 악기 파트를 자동으로 작곡(편곡)**합니다. 예를 들어 피아노 악보를 주면, AI가 어울리는 바이올린 파트를 만들어주는 것입니다.

### 왜 MLOps가 필요했나?

MLOps 도입 전에는 이런 문제가 있었습니다:

```text
문제 1: "AI가 만든 편곡이 좋은 건지 나쁜 건지 모른다"
   → 사람이 직접 들어봐야만 판단 가능
   → 밀린 곡이 100곡이면 100곡을 다 들어야 함

문제 2: "어떤 설정이 좋은 결과를 만드는지 모른다"
   → temperature를 0.8로 설정하면 더 나을까? 1.2가 나을까?
   → 직감에 의존, 기록도 없음

문제 3: "코드가 한 파일에 800줄"
   → AI 개발자가 수정하면 서버 인프라가 깨질 수 있음
   → 인프라 담당자가 수정하면 AI 로직이 깨질 수 있음
```

**MLOps는 이 세 가지를 시스템으로 해결합니다:**

| 문제 | 해결책 | 도구 |
|------|--------|------|
| 품질 측정 불가 | 자동 메트릭 수집 | `ai_core/metrics.py` |
| 설정 최적화 불가 | 자동 하이퍼파라미터 탐색 | Optuna + `ai_core/tune.py` |
| 코드 얽힘 | AI 코드와 인프라 코드 분리 | `ai_core/` vs `app/` |

---

## 2. 전체 구조: 새 위에서 보기

### 2-1. 시스템 아키텍처

```text
사용자
  │
  ▼
메인 서버 (GKE)
  │  Redis Streams
  ▼
┌────────────────────────────── GPU 서버 (온프레미스, RTX 4090) ──┐
│                                                                │
│  worker.py ──→ ai_core/ (AI 편곡) ──→ 콜백으로 결과 반환       │
│                    │                                            │
│                    ├─→ metrics.py → 품질 점수 자동 계산          │
│                    ├─→ evaluate.py → 배치 평가 + MLflow 기록    │
│                    └─→ tune.py → Optuna 자동 튜닝 + MLflow 기록 │
│                                                                │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │ Docker 서비스들                                          │   │
│  │  MLflow Server     (mlflow.tutti.asia)    → 실험 기록    │   │
│  │  Optuna Dashboard  (optuna.tutti.asia)    → 튜닝 시각화  │   │
│  │  Cloudflared       → 외부 접속 터널                       │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  optuna_studies.db  → Optuna 결과 저장 (SQLite 파일)           │
│  ~/dvc_store        → 학습 데이터 버전 관리 (DVC)               │
└────────────────────────────────────────────────────────────────┘
```

### 2-2. 코드 구조

프로젝트의 코드는 크게 **세 영역**으로 나뉩니다:

```text
tutti-backend-ai/
│
├── ai_core/              🧠 AI 개발자 영역 (자유롭게 수정 가능)
│   ├── constants.py         악기 정의, 매핑 테이블
│   ├── vocab.py             토큰 사전 구축
│   ├── model_loader.py      모델 체크포인트 로딩
│   ├── tokenizer.py         MIDI → 토큰 변환
│   ├── generator.py         AI 추론 (슬라이딩 윈도우)
│   ├── decoder.py           토큰 → 노트 변환
│   ├── postprocess.py       후처리 (피치 클리핑 등)
│   ├── arrangement.py       편곡 오케스트레이션 (진입점)
│   ├── midi_writer.py       MIDI 파일 저장
│   ├── metrics.py           품질 메트릭 계산
│   ├── evaluate.py          모델 평가 CLI
│   └── tune.py              Optuna 하이퍼파라미터 튜닝
│
├── contracts/            📋 AI ↔ 인프라 인터페이스 계약
│   └── interfaces.py       양쪽이 지켜야 할 함수 시그니처
│
├── app/                  🔧 인프라 영역 (AI 개발자는 수정 불필요)
│   ├── core/
│   │   ├── config.py        환경 설정
│   │   └── model_registry.py 모델 레지스트리
│   └── services/
│       ├── inference.py     Facade (ai_core를 re-export)
│       └── midi_processor.py 트랙 재매핑
│
├── worker.py             Redis Streams 워커 (프로덕션 진입점)
│
├── mlflow/
│   └── docker-compose.mlflow.yml   MLflow + Optuna Dashboard 서비스
│
├── docs/
│   └── AI_DEVELOPER_GUIDE.md       AI 개발자용 상세 가이드
│
└── pyproject.toml        의존성 관리 (uv 패키지 매니저)
```

> **왜 코드를 분리했나?**
> 이전에는 `app/services/inference.py` 한 파일에 815줄이 몰려 있었습니다. AI 개발자가 추론 로직을 수정하면 서버 인프라가 깨지고, 반대도 마찬가지. **코드 분리(디커플링)**를 통해 각자의 영역을 안전하게 수정할 수 있게 했습니다.

---

## 3. 도입된 도구들: 각각 무엇이고, 왜 쓰는가?

### 3-1. MLflow — "실험 기록장"

**한 줄 요약**: AI 실험의 모든 것을 자동 기록하고 웹에서 비교하는 도구.

```text
비유: 연구실 실험 노트를 디지털로 옮긴 것

  "temperature=0.85에서 chord_accuracy가 0.72였고,
   temperature=1.1에서 0.68이었다"

  이런 기록을 수천 건 자동으로 쌓아서 표와 그래프로 볼 수 있게 해줍니다.
```

**우리 프로젝트에서 하는 일:**
- `evaluate.py`가 모델 품질 평가 결과를 MLflow에 기록
- `tune.py`가 하이퍼파라미터 튜닝 과정을 MLflow에 기록
- 웹 UI(`mlflow.tutti.asia`)에서 실험 비교, 차트 생성

**접속 방법:**
- GPU 서버 로컬: `http://localhost:5000`
- 외부 접속: `https://mlflow.tutti.asia` (Cloudflare Access 인증 필요)

---

### 3-2. Optuna — "자동 탐색기"

**한 줄 요약**: "어떤 파라미터 조합이 가장 좋은 결과를 내는지" 자동으로 찾아주는 도구.

```text
비유: 요리 레시피 최적화

  소금을 얼마나 넣고 (temperature)
  불 세기를 어떻게 하고 (top_p)
  몇 분 볶을지 (rest_penalty)

  수십~수백 가지 조합을 직접 시도하는 대신,
  Optuna가 "이전 시도 결과를 보고" 다음에 뭘 시도할지 알아서 결정합니다.
```

**우리 프로젝트에서 탐색하는 파라미터:**

| 파라미터 | 의미 | 탐색 범위 |
|----------|------|-----------|
| `temperature` | AI의 창의성 정도 (높을수록 다양) | 0.5 ~ 1.5 |
| `top_p` | 상위 확률 토큰만 고려 | 0.8 ~ 0.99 |
| `rest_penalty` | 쉼표 억제 강도 | 0.5 ~ 3.0 |
| `window_bars` | 한 번에 처리하는 마디 수 | 4 ~ 16 |
| `context_bars` | 참고하는 이전 마디 수 | 4 ~ 16 |
| `fade_bars` | 윈도우 겹침 길이 | 4 ~ 16 |

---

### 3-3. Optuna 대시보드 — "탐색 과정 시각화"

**한 줄 요약**: Optuna가 탐색 중인/완료된 결과를 웹 그래프로 보여주는 도구.

**제공하는 차트:**
- **Optimization History**: 트라이얼 번호 vs 점수 그래프 (점수가 올라가는지 추세 확인)
- **Parallel Coordinate**: 여러 파라미터를 동시에 보며 좋은 조합 패턴 발견
- **Parameter Importance**: "temperature가 점수에 40% 영향, top_p는 15% 영향" 같은 분석
- **Slice Plot**: 파라미터 하나를 X축에, 점수를 Y축에 놓고 관계 확인

**접속 방법:**
- `https://optuna.tutti.asia` (Cloudflare Access 인증 필요)

---

### 3-4. DVC — "데이터 버전 관리"

**한 줄 요약**: 대용량 학습 데이터(MIDI 파일들)를 Git처럼 버전 관리하는 도구.

```text
비유: Git이 코드 버전을 관리하듯, DVC는 데이터 버전을 관리

  "이번 학습에 쓴 데이터셋 v3 (500곡)"
  "다음 학습에 쓸 데이터셋 v4 (700곡, 재즈 200곡 추가)"

  데이터 자체는 Git에 올리지 않고(용량이 크니까),
  "어떤 데이터를 썼는지"만 Git에 기록합니다.
```

---

### 3-5. uv — "패키지 매니저"

**한 줄 요약**: Python 패키지를 관리하는 도구 (pip + venv를 대체, 10~100배 빠름).

```bash
# 기존 (pip)
pip install -r requirements.txt    # 느리고, 의존성 충돌 위험

# 현재 (uv)
uv sync --extra gpu --extra mlflow  # 빠르고, 정확한 lock 파일로 재현 가능
```

---

## 4. 품질 메트릭: AI가 만든 편곡의 "점수"

### 4-1. 왜 점수가 필요한가?

AI가 만든 편곡이 "좋은지 나쁜지"를 사람이 일일이 듣지 않고 **자동으로 수치화**하기 위함입니다. 이 점수가 있어야 Optuna가 "어떤 설정이 더 좋은지" 비교할 수 있습니다.

### 4-2. 측정되는 메트릭

#### Stage 1: 기본 통계 (AI 결과물만으로 측정)

| 메트릭 | 의미 | 예시 |
|--------|------|------|
| `note_count` | 생성된 노트 수 | 342개 |
| `pitch_range` | 최고음 - 최저음 | 36 (3옥타브) |
| `density_per_sec` | 초당 노트 밀도 | 2.5개/초 |

#### Stage 2: 음악적 평가 (원본 악보와 비교)

| 메트릭 | 의미 | 좋은 방향 | 직관적 설명 |
|--------|------|:---------:|-------------|
| `chord_accuracy` | 코드 일치율 | ↑ 높을수록 좋음 | "원곡의 화음을 잘 따라가는가?" |
| `pch_similarity` | 조성 유사도 | ↑ 높을수록 좋음 | "원곡과 같은 키(조성)인가?" |
| `doa` | 편곡 다양성 | ↑ 높을수록 좋음 | "너무 단조롭지 않고 창의적인가?" |
| `dissonance_rate` | 불협화 비율 | ↓ 낮을수록 좋음 | "귀에 거슬리는 소리가 얼마나 나는가?" |

### 4-3. 메트릭이 기록되는 곳

```text
1. 프로덕션 편곡 (사용자 요청)
   worker.py → 편곡 완료 → metrics.py가 자동 측정 → 콜백에 포함
   → 메인 서버 DB에 저장

2. 모델 평가 (개발자가 수동 실행)
   python -m ai_core.evaluate → MLflow에 기록

3. 하이퍼파라미터 튜닝 (개발자가 수동 실행)
   python -m ai_core.tune → MLflow에 기록 + Optuna DB에 저장
```

---

## 5. 하이퍼파라미터 튜닝: 자세히 알아보기

### 5-1. 기본 흐름

```text
┌─────────────────────────────────────────────────┐
│ Optuna가 파라미터 조합을 제안                       │
│   예: temperature=0.85, top_p=0.92               │
│                     │                            │
│                     ▼                            │
│ 해당 설정으로 MIDI 편곡 실행                        │
│                     │                            │
│                     ▼                            │
│ 결과 MIDI의 품질 점수 계산 (metrics.py)             │
│   예: chord_accuracy = 0.72                      │
│                     │                            │
│                     ▼                            │
│ Optuna에게 점수 보고                               │
│ → Optuna가 "다음에는 이 조합을 시도해보자" 결정       │
│                     │                            │
│                     ▼                            │
│ N번 반복 후 최적 조합 출력                          │
└─────────────────────────────────────────────────┘
```

### 5-2. 빠른 탐색 전략 3가지

튜닝은 **매우 오래 걸리는 작업**입니다. 곡 10개 × 30초/곡 × 30회 = **150분**. 이를 빠르게 만드는 세 가지 전략이 있습니다:

#### A. 조기 종료 (Pruning) — `--pruning`

```text
"이 설정은 분명 안 되겠다" 싶으면 끝까지 안 하고 바로 다음으로 넘어가기

예시 (10곡 평가 중):
  Trial A:  곡1=0.7  곡2=0.6  곡3=0.5  → 전체 평균 0.60 (정상 완료)
  Trial B:  곡1=0.2  → 이미 다른 것들보다 한참 나쁨 → ✂ 중단!
            나머지 9곡을 건너뛰고 → 약 4.5분 절약

┌──────────────────────────────────────────────┐
│ 장점: 명백히 나쁜 조합에 시간 낭비 안 함           │
│ 단점: 첫 곡만 운 나쁘게 낮으면 좋은 조합도 잘릴 수 │
│       있음 (최소 1곡은 보고 판단하도록 보호)        │
│ 추천: 평가 파일이 5개 이상일 때 사용               │
└──────────────────────────────────────────────┘
```

#### B. 데이터 샘플링 — `--sample-ratio 0.1`

```text
"100곡 전체를 다 쓰지 말고, 랜덤으로 10곡만 골라 빠르게 경향 파악"

예시:
  전체 데이터: 100곡
  --sample-ratio 0.1 → 매 트라이얼마다 랜덤 10곡 선택
  매번 다른 10곡을 뽑으므로 특정 곡에 편향되지 않음

┌──────────────────────────────────────────────┐
│ 장점: 10배 빨라짐 (100곡 → 10곡)                │
│ 단점: 샘플이 전체를 대표하지 못할 수 있음          │
│ 추천: 첫 탐색은 0.1, 검증은 1.0 (전체)           │
└──────────────────────────────────────────────┘
```

#### C. 파일 수 제한 — `--max-files 5`

```text
"무조건 최대 5곡만 평가" (실행 시간 예측 가능)

예시:
  100곡 데이터 + --max-files 5 + --n-trials 30
  = 5곡 × 30초 × 30회 = 약 75분 (전체 사용 시 약 25시간)

┌──────────────────────────────────────────────┐
│ 장점: "이거 몇 시간 걸려?" → 정확히 계산 가능      │
│ 단점: 5곡의 결과가 전체 100곡과 다를 수 있음        │
│ 추천: 빠른 탐색에 5곡, 정밀 검증에 제한 없이        │
└──────────────────────────────────────────────┘
```

### 5-3. 추천 워크플로우

```text
┌──────────────────────────────────────────────────────┐
│                                                      │
│  1단계: 빠른 탐색 (30분~1시간)                          │
│    → 세 전략을 다 켜고, 100번 시도 (넓은 범위)           │
│    python -m ai_core.tune \                           │
│        --source ~/midi/ --target keyboard \            │
│        --n-trials 100 \                               │
│        --pruning --sample-ratio 0.1 --max-files 5     │
│                                                      │
│  2단계: 범위 좁히기                                     │
│    → 1단계에서 좋았던 파라미터 범위를 코드에서 조정        │
│    → 다시 탐색                                         │
│                                                      │
│  3단계: 전체 데이터 검증 (수 시간)                        │
│    → 샘플링/제한 없이 전체 데이터로 20번 시도             │
│    python -m ai_core.tune \                           │
│        --source ~/midi/ --target keyboard \            │
│        --n-trials 20 --pruning                        │
│                                                      │
│  4단계: 결과 적용                                       │
│    → 최적 파라미터를 arrangement.py에 반영               │
│                                                      │
└──────────────────────────────────────────────────────┘
```

---

## 6. 인프라 구성

### 6-1. Docker 서비스 구성

GPU 서버에서 Docker Compose로 관리되는 서비스:

| 서비스 | 포트 | 외부 접속 | 역할 |
|--------|------|-----------|------|
| MLflow | 5000 | mlflow.tutti.asia | 실험 기록 서버 |
| Optuna Dashboard | 8080 | optuna.tutti.asia | 튜닝 시각화 |
| Cloudflared | — | — | 외부 접속 터널 |

모든 포트는 `127.0.0.1`에만 바인딩되어 **외부에서 직접 접근 불가**. Cloudflare Tunnel이 안전한 아웃바운드 연결을 만들고, Cloudflare Access가 인증을 처리합니다.

### 6-2. 데이터 저장소

| 파일 | 위치 | 내용 | Git에 올라감? |
|------|------|------|:------------:|
| `optuna_studies.db` | 프로젝트 루트 | Optuna 튜닝 결과 (SQLite) | ❌ |
| `mlflow.db` | Docker 볼륨 | MLflow 실험 기록 (SQLite) | ❌ |
| `~/dvc_store/` | 홈 디렉토리 | 학습 데이터 실체 (DVC 관리) | ❌ |

> **SQLite란?** 별도 서버 설치 없이 **파일 하나**로 동작하는 초경량 데이터베이스. Python에 내장되어 있어서 추가 설치 불필요.

---

## 7. 구축 과정: Phase별 진행 내역

### Phase 0: 레거시 정리 + 브랜치 전략

```
v3.0-pre-mlops 태그 → feat/mlops-pipeline 브랜치 생성
사용하지 않는 HTTP 레거시 코드를 _legacy/ 디렉토리로 이동
docker-compose.yml에서 불필요한 서비스 블록 제거
```

**커밋**: `390343c feat: ai_core 패키지 분리 및 inference.py 디커플링 (Phase 0+1)`

### Phase 1: 코드 디커플링

```
inference.py (815줄) → ai_core/ 패키지로 분해 (8개 모듈)
contracts/interfaces.py → AI ↔ 인프라 인터페이스 계약 정의
app/services/inference.py → 얇은 re-export facade로 전환
CODEOWNERS → AI 개발자 / 인프라 담당자 영역 명시
```

**커밋**: `edeab61 refactor: save_midi를 ai_core/로 이동하여 역방향 의존 제거`

### Phase 2: 의존성 관리 + MLflow + DVC

```
requirements.txt → pyproject.toml + uv로 전환
CI/CD에 테스트 게이트 추가 (테스트 통과해야 빌드)
MLflow Docker 서비스 추가 + Cloudflare 터널 연결
DVC 초기화 (로컬 저장소)
```

**커밋**: `291e1ec feat(phase2): uv 전환, CI 테스트 게이트, DVC/MLflow 세팅`

### Phase 3: 모니터링 + AI 개발자 온보딩

```
worker.py에 /metrics Prometheus 엔드포인트 추가
docs/AI_DEVELOPER_GUIDE.md 작성
```

**커밋**: `6fd3304 feat(phase3): worker /metrics 엔드포인트 + AI 개발자 가이드`

### Phase 4: 품질 메트릭 수집

```
ai_core/metrics.py → 기본 통계 + 음악적 평가(논문 기반) 메트릭
worker.py → 인퍼런스 완료 후 자동으로 품질 점수 콜백에 포함
ai_core/evaluate.py → 배치 평가 CLI + MLflow 자동 기록
```

**커밋**: `1b05ee2 feat(phase4): 편곡 품질 메트릭 자동 수집`

### Phase 5: Optuna 자동 튜닝

```
ai_core/tune.py → Optuna 하이퍼파라미터 자동 탐색
  + SQLite 영속 저장 (optuna-dashboard 연동)
  + Pruning (조기 종료)
  + Data Sampling (데이터 일부만 사용)
  + Max Files (파일 수 제한)
  + MLflow 자동 기록 (모든 트라이얼 Nested Run)
docker-compose.mlflow.yml → optuna-dashboard 서비스 추가
pyproject.toml → optuna, optuna-dashboard, numpy 의존성 추가
```

**커밋**:
- `ad7448e feat(phase3-4): 메트릭 엔드포인트 + 품질 평가 + 개발자 가이드`
- `a83af27 feat: Optuna 튜닝 고도화 + 대시보드 인프라`

---

## 8. 실행 명령어 모음

### 의존성 설치

```bash
# GPU 서버에서
uv sync --extra gpu --extra mlflow
```

### 모델 평가

```bash
# 단일 파일 평가
python -m ai_core.evaluate \
    --source 원본.mid \
    --generated AI결과.mid

# 디렉토리 배치 평가 (MLflow 자동 기록)
python -m ai_core.evaluate \
    --source ~/originals/ \
    --generated ~/results/ \
    --experiment "violin-model-v3"
```

### 하이퍼파라미터 튜닝

```bash
# 빠른 탐색 (모든 최적화 ON)
python -m ai_core.tune \
    --source ~/midi_data/ \
    --target keyboard \
    --n-trials 100 \
    --pruning --sample-ratio 0.1 --max-files 5

# 정밀 검증 (전체 데이터)
python -m ai_core.tune \
    --source ~/midi_data/ \
    --target keyboard \
    --n-trials 20 --pruning
```

### Docker 서비스

```bash
# MLflow + Optuna Dashboard 시작
docker compose -f mlflow/docker-compose.mlflow.yml up -d

# 상태 확인
docker compose -f mlflow/docker-compose.mlflow.yml ps
```

### 데이터 관리 (DVC)

```bash
# 학습 데이터 추가
cp -r ~/new_midi/ data/midi_dataset/
dvc add data/midi_dataset/
git add data/midi_dataset.dvc .gitignore
git commit -m "data: MIDI 데이터셋 v2"
```

---

## 9. 용어 사전

처음 보는 용어가 있다면 여기서 찾아보세요:

| 용어 | 의미 |
|------|------|
| **MLOps** | Machine Learning + Operations. AI 모델 개발에 DevOps 사례를 적용하는 것 |
| **하이퍼파라미터** | AI 모델의 "설정값". 학습 데이터가 아니라 사람이 정하는 값 (예: temperature) |
| **트라이얼 (Trial)** | Optuna에서 하나의 파라미터 조합으로 실행한 한 번의 시도 |
| **프루닝 (Pruning)** | "가지치기". 유망하지 않은 트라이얼을 중간에 중단하는 것 |
| **메트릭 (Metric)** | 측정 지표. 모델의 성능을 숫자로 나타낸 것 |
| **Facade** | 복잡한 시스템 앞에 놓은 간단한 인터페이스 (디자인 패턴) |
| **디커플링** | 서로 얽힌 코드를 독립적으로 분리하는 것 |
| **SQLite** | 파일 하나로 동작하는 경량 데이터베이스. 별도 서버 불필요 |
| **Cloudflare Tunnel** | 서버 포트를 열지 않고도 외부에서 접속할 수 있게 해주는 보안 터널 |
| **DVC** | Data Version Control. 대용량 파일을 Git처럼 버전 관리 |
| **uv** | Rust로 만든 초고속 Python 패키지 매니저 (pip 대체) |

---

## 10. 관련 문서

| 문서 | 위치 | 설명 |
|------|------|------|
| AI 개발자 상세 가이드 | `docs/AI_DEVELOPER_GUIDE.md` | 실제 명령어, CLI 옵션, 워크플로우 |
| Implementation Plan | `local_docs/implementation_plan.md` | Phase별 기술 구현 상세 계획 |
| Redis 아키텍처 | `docs/REDIS_ARCHITECTURE.md` | 워커 시스템 상세 구조 |
| 인프라 가이드 | `docs/BACKEND_INFRA_GUIDE.md` | 서버 배포, CI/CD 파이프라인 |
| 모델 관리 | `docs/AI_ML_GUIDE.md` | 체크포인트 등록, 모델 교체 방법 |
