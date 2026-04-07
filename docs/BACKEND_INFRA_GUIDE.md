# Tutti AI Server — 백엔드 / 인프라 엔지니어 가이드

> **대상 독자**: 백엔드 API, DevOps, 인프라 엔지니어  
> **최종 수정**: 2026-04-07  
> **버전**: v2 (Qwen2.5 통합 모델 전환 이후)

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [프로젝트 구조](#2-프로젝트-구조)
3. [요청 흐름 (Request Lifecycle)](#3-요청-흐름-request-lifecycle)
4. [API 레퍼런스](#4-api-레퍼런스)
5. [설정 및 환경 변수](#5-설정-및-환경-변수)
6. [배포 아키텍처 (K8s / GKE)](#6-배포-아키텍처-k8s--gke)
7. [CI/CD 파이프라인](#7-cicd-파이프라인)
8. [파일별 종속성 맵](#8-파일별-종속성-맵)
9. [수정 가이드 — "이런 걸 바꾸고 싶을 때"](#9-수정-가이드--이런-걸-바꾸고-싶을-때)
10. [모니터링 및 트러블슈팅](#10-모니터링-및-트러블슈팅)
11. [테스트](#11-테스트)

---

## 1. 시스템 개요

Tutti AI Server는 **사용자가 업로드한 MIDI 파일에 새 악기 파트를 AI로 생성·추가**하는 추론(Inference) 전용 마이크로서비스입니다.

```
┌────────────┐     POST /api/v1/arrange      ┌─────────────────┐
│ Main Server│ ──────────────────────────────→ │  AI Server      │
│ (Spring)   │ ←── 콜백 (진행률 + MIDI 파일) ─ │  (FastAPI+GPU)  │
└────────────┘                                └─────────────────┘
```

**핵심 특성**:
- **비동기 처리**: 요청 즉시 `202 Accepted` 반환, 백그라운드에서 추론 후 콜백으로 결과 전달
- **GPU 의존**: NVIDIA GPU (T4 권장)에서 Qwen2.5-0.5B 모델로 추론 수행
- **Zero-Scaling**: KEDA HTTP Add-on으로 요청이 없을 때 레플리카 0 → GPU 비용 $0
- **Stateless**: 모델 체크포인트는 Init Container가 GCS에서 매번 다운로드

---

## 2. 프로젝트 구조

```
tutti-backend-ai/
├── app/
│   ├── main.py                 # FastAPI 앱 진입점, Lifespan(모델 워밍업)
│   ├── api/
│   │   ├── health.py           # GET /health (헬스체크 + 로드된 모델 정보)
│   │   └── v1/
│   │       ├── arrange.py      # POST /api/v1/arrange (편곡 요청 수신)
│   │       └── download.py     # GET /api/v1/download/{job_id} (결과 다운로드)
│   ├── core/
│   │   ├── config.py           # 환경 변수 설정 (Settings)
│   │   └── model_registry.py   # 모델 로드·관리 레지스트리
│   ├── schemas/
│   │   ├── request.py          # ArrangeRequest, Mapping 스키마
│   │   └── response.py         # ArrangeResponse, HealthResponse 스키마
│   └── services/
│       ├── arrangement.py      # 편곡 라이프사이클 오케스트레이터
│       ├── callback.py         # Main Server 콜백 전송 (재시도 포함)
│       ├── inference.py        # [AI 영역] 추론 엔진 (수정 비권장)
│       └── midi_processor.py   # MIDI 다운로드 + 트랙 재매핑
├── k8s/
│   └── base/
│       ├── deployment.yaml     # GKE Deployment (Init Container + GPU)
│       ├── service.yaml        # ClusterIP Service
│       ├── keda-http.yaml      # KEDA Zero-Scaling 설정
│       └── kustomization.yaml  # Kustomize 이미지 오버라이드
├── .github/workflows/
│   ├── ci-ai.yml               # 메인 CI/CD (Build → Push → Deploy)
│   └── deploy-registry.yml     # registry.json 변경 시 GCS 업로드 + Pod 재시작
├── Dockerfile                  # 멀티스테이지 빌드 (Builder + Runtime)
├── registry.json               # 모델 레지스트리 설정 파일
├── requirements.txt            # Python 의존성
├── setup_gcs.sh                # GCS 버킷 + Workload Identity 초기 설정
├── setup_gpu_pool.sh           # GKE GPU Node Pool 생성
└── setup_keda.sh               # KEDA + HTTP Add-on 설치
```

---

## 3. 요청 흐름 (Request Lifecycle)

```
Main Server                     AI Server
    │                               │
    │  POST /api/v1/arrange         │
    │  (ArrangeRequest JSON)        │
    │──────────────────────────────→│
    │                               │
    │  200 {"status":"accepted"}    │  ← 즉시 반환
    │←──────────────────────────────│
    │                               │
    │                               │  ┌─ Background Task ─────────────┐
    │  callback (progress: 10%)     │  │ 1. MIDI 다운로드 (httpx)      │
    │←──────────────────────────────│  │    → /tmp/tutti_midi_downloads │
    │                               │  │                               │
    │  callback (progress: 20%)     │  │ 2. 트랙 재매핑 (mido)         │
    │←──────────────────────────────│  │    → 악기 변경 / 트랙 삭제    │
    │                               │  │                               │
    │  callback (progress: 80%)     │  │ 3. AI 추론 (torch)            │
    │←──────────────────────────────│  │    → 새 파트 생성             │
    │                               │  │                               │
    │  callback (progress: 100%)    │  │ 4. 결과 MIDI 콜백 전송        │
    │  + multipart MIDI file        │  │    → multipart/form-data      │
    │←──────────────────────────────│  └───────────────────────────────┘
    │                               │
```

### 단계별 상세

| 단계 | 진행률 | 파일 | 핵심 동작 |
|------|--------|------|-----------|
| MIDI 다운로드 | 10% | `midi_processor.py` → `download_midi()` | Main Server가 제공한 Supabase Signed URL에서 MIDI 스트리밍 다운로드 |
| 트랙 재매핑 | 20% | `midi_processor.py` → `remap_original_tracks()` | `mappings`에 따라 악기 변경(`program_change`) 또는 트랙 삭제(ID=129) |
| AI 추론 | 80% | `inference.py` → `run_arrangement()` | 모델이 입력 MIDI를 컨텍스트로 받아 새 악기 파트 생성 (**AI 영역**) |
| 결과 전송 | 100% | `callback.py` → `send_callback_with_file()` | multipart/form-data로 MIDI 파일 + 메타데이터 전송 |

---

## 4. API 레퍼런스

### `POST /api/v1/arrange`

편곡 요청을 수신하고 즉시 반환합니다. 추론은 백그라운드에서 실행됩니다.

**Request Body** (`application/json`):

```json
{
  "projectId": 1,
  "versionId": 1,
  "midiFilePath": "https://storage.supabase.co/.../file.mid",
  "mappings": [
    { "trackIndex": 0, "targetInstrumentId": 0 },
    { "trackIndex": 1, "targetInstrumentId": 129 }
  ],
  "targetInstrumentId": 40,
  "minNote": null,
  "maxNote": null,
  "modelType": null,
  "genre": "CLASSICAL",
  "temperature": 1.0,
  "callbackUrl": "https://api.tutti.com/internal/ai/callback",
  "callbackSecret": "secret-token-123"
}
```

| 필드 | 타입 | 필수 | 설명 |
|------|------|------|------|
| `projectId` | int | ✅ | 프로젝트 ID |
| `versionId` | int | ✅ | 버전 ID |
| `midiFilePath` | URL | ✅ | 원본 MIDI 다운로드 URL (Supabase Signed URL) |
| `mappings` | List[Mapping] | ✅ | 트랙별 악기 매핑. `targetInstrumentId=129`이면 트랙 삭제 |
| `targetInstrumentId` | int | ✅ | AI가 새로 생성할 악기의 MIDI Program 번호 (0~128) |
| `minNote` | int? | ❌ | 음역 최솟값 (미입력 시 악기 기본값 자동 적용) |
| `maxNote` | int? | ❌ | 음역 최댓값 (미입력 시 악기 기본값 자동 적용) |
| `modelType` | str? | ❌ | 모델 선택 키 (미입력 시 `registry.json`의 `default` 사용) |
| `genre` | Literal | ❌ | `CLASSICAL`·`JAZZ`·`POP`·`ROCK`·`ELECTRONIC`·`FOLK`·`UNKNOWN` |
| `temperature` | float | ❌ | 생성 다양성 (0.1~2.0, 기본값 1.0) |
| `callbackUrl` | URL | ✅ | 진행 상황 및 결과 수신 콜백 URL |
| `callbackSecret` | str | ✅ | 콜백 인증 토큰 (`X-Callback-Secret` 헤더로 전송) |

**Response** (`200 OK`):

```json
{ "status": "accepted", "message": "편곡 요청을 수신했습니다." }
```

### 콜백 전송 형식

**진행률 콜백** (`application/json`, `send_callback`):

```json
{
  "projectId": 1,
  "versionId": 1,
  "status": "processing",
  "progress": 10
}
```

**완료 콜백** (`multipart/form-data`, `send_callback_with_file`):

| Part | Content-Type | 내용 |
|------|-------------|------|
| `metadata` | `application/json` | `{"projectId":1, "versionId":1, "status":"complete", "progress":100}` |
| `file` | `audio/midi` | 생성된 MIDI 바이너리 |

**실패 콜백** (`application/json`):

```json
{
  "projectId": 1,
  "versionId": 1,
  "status": "failed",
  "progress": 0,
  "errorMessage": "에러 상세 메시지"
}
```

**재시도 정책**:
- 진행률 콜백: 최대 3회, 지수 백오프 (1s → 2s → 4s)
- 파일 콜백: 최대 5회, 지수 백오프 (1s → 2s → 4s → 8s → 16s), 타임아웃 60초

### `GET /health`

서버 상태 및 로드된 모델 정보를 반환합니다. K8s Probe에서 사용됩니다.

### `GET /api/v1/download/{job_id}`

결과 MIDI 파일 직접 다운로드 (현재 콜백 방식이 기본이므로 보조 엔드포인트).

---

## 5. 설정 및 환경 변수

> 설정 파일: `app/core/config.py` — Pydantic `BaseSettings` 기반, `.env` 파일 또는 환경 변수에서 로드

| 변수 | 기본값 | K8s 설정 위치 | 설명 |
|------|--------|--------------|------|
| `HOST` | `0.0.0.0` | Dockerfile CMD | 바인드 주소 |
| `PORT` | `8000` | Dockerfile CMD | 리스닝 포트 |
| `LOG_LEVEL` | `info` | `deployment.yaml` env | 로그 레벨 (`debug`, `info`, `warning`, `error`) |
| `MODEL_DIR` | `/models` | `deployment.yaml` env | 모델 체크포인트 디렉토리 (볼륨 마운트 타겟) |
| `RESULTS_DIR` | `/tmp/results` | `deployment.yaml` env | 생성된 MIDI 임시 저장 디렉토리 |
| `GPU_ID` | `0` | 미사용 (향후 멀티GPU) | 사용할 GPU 인덱스 |

### 설정 수정이 필요한 경우

| 시나리오 | 수정 파일 | 방법 |
|---------|-----------|------|
| 새 환경 변수 추가 | `app/core/config.py` | `Settings` 클래스에 필드 추가 |
| 배포 환경 값 변경 | `k8s/base/deployment.yaml` | `containers[0].env` 수정 |
| 로컬 개발 | `.env` 파일 생성 | `.env.example` 참조 |

---

## 6. 배포 아키텍처 (K8s / GKE)

### 전체 구성

```
┌─── GKE Cluster (us-central1-a) ───────────────────────────────────────┐
│                                                                        │
│  ┌── Default Node Pool (CPU) ──┐   ┌── GPU Node Pool (gpu-pool-t4) ─┐ │
│  │  Main Server, Other Pods    │   │  n1-standard-4 + T4 GPU        │ │
│  │                             │   │  min: 0 / max: 1 (Auto-scale)  │ │
│  └─────────────────────────────┘   │  Taint: nvidia.com/gpu=present │ │
│                                    │                                 │ │
│                                    │  ┌──────────────────────────┐   │ │
│  ┌── KEDA (keda namespace) ────┐   │  │ ai-server Pod            │   │ │
│  │ HTTP Interceptor Proxy      │──→│  │ ┌── Init Container ───┐  │   │ │
│  │ 요청 감지 → Pod 스케일링     │   │  │ │ GCS → /models 복사  │  │   │ │
│  └─────────────────────────────┘   │  │ └─────────────────────┘  │   │ │
│                                    │  │ ┌── App Container ─────┐ │   │ │
│  ┌── GCS Bucket ───────────────┐   │  │ │ FastAPI + Torch      │ │   │ │
│  │ tutti-ai-models/v1/         │───│  │ │ GPU 추론             │ │   │ │
│  │   registry.json             │   │  │ └─────────────────────┘ │   │ │
│  │   best/model.safetensors    │   │  └──────────────────────────┘   │ │
│  └─────────────────────────────┘   └─────────────────────────────────┘ │
└────────────────────────────────────────────────────────────────────────┘
```

### 핵심 K8s 리소스

| 리소스 | 파일 | 역할 |
|--------|------|------|
| `Deployment` | `k8s/base/deployment.yaml` | Init Container(GCS 다운로드) + App Container(추론) |
| `Service` | `k8s/base/service.yaml` | ClusterIP (`ai-server:8000`) |
| `HTTPScaledObject` | `k8s/base/keda-http.yaml` | KEDA Zero-Scaling (min:0 → max:3) |

### 모델 로딩 플로우

```
Pod 시작
  ↓
Init Container (google/cloud-sdk:slim)
  ├── gsutil cp gs://tutti-ai-models/v1/* /models/
  └── emptyDir 볼륨에 저장
  ↓
App Container (FastAPI)
  ├── Lifespan → ModelRegistry(model_dir=/models)
  ├── registry.json 읽기 → 모델 목록 파악
  ├── load_all_models() → 체크포인트 로드 → GPU 메모리 적재
  └── 준비 완료 → /health 200 OK → readinessProbe 통과
```

### Probe 설정

| Probe | 경로 | 지연 | 주기 | 목적 |
|-------|------|------|------|------|
| `startupProbe` | `/health` | 60s | 10s (최대 3분) | 모델 로딩 완료 대기 |
| `readinessProbe` | `/health` | 15s | 10s | 트래픽 수신 준비 확인 |
| `livenessProbe` | `/health` | 30s | 30s | 데드락/OOM 감지 |

### 초기 인프라 구축 스크립트

| 스크립트 | 실행 시점 | 내용 |
|---------|-----------|------|
| `setup_gcs.sh` | 최초 1회 | GCS 버킷 생성, Workload Identity 바인딩, IAM 권한 |
| `setup_gpu_pool.sh` | 최초 1회 | GPU Node Pool 생성 (T4, min:0/max:1, Taint) |
| `setup_keda.sh` | 최초 1회 | KEDA Core + HTTP Add-on Helm 설치 |

---

## 7. CI/CD 파이프라인

### 메인 파이프라인: `ci-ai.yml`

**트리거**: `main` 브랜치에 `app/`, `k8s/`, `Dockerfile`, `requirements.txt` 변경 시

```
Push to main
  ↓
Job 1: Build & Push
  ├── GCP 인증 (Workload Identity Federation)
  ├── Docker Buildx 멀티스테이지 빌드
  ├── Artifact Registry 푸시 (태그: commit SHA + latest)
  └── GHA 캐시 활용
  ↓
Job 2: Deploy to GKE
  ├── GKE Credentials 취득
  ├── Kustomize 이미지 태그 업데이트
  ├── kubectl apply -k .
  └── rollout status 확인 (300s 타임아웃)
```

### 레지스트리 파이프라인: `deploy-registry.yml`

**트리거**: `registry.json` 변경 시

```
Push registry.json
  ↓
  ├── GCS에 registry.json 업로드
  └── kubectl rollout restart → Pod 재시작 → Init Container가 새 registry 반영
```

### 필요한 GitHub Secrets

| Secret | 설명 |
|--------|------|
| `GCP_PROJECT_ID` | GCP 프로젝트 ID |
| `GCP_WORKLOAD_IDENTITY_PROVIDER` | WIF Provider 리소스 이름 |
| `GCP_SERVICE_ACCOUNT` | GCP Service Account 이메일 |

---

## 8. 파일별 종속성 맵

### `app/main.py` — 앱 진입점

| 종속 모듈 | 사용 방식 |
|-----------|-----------|
| `core/config.py` | `settings` 싱글톤에서 `MODEL_DIR`, `LOG_LEVEL` 읽기 |
| `core/model_registry.py` | `ModelRegistry` 인스턴스 생성 → `app.state.registry`에 저장 |
| `api/v1/arrange.py` | 라우터 등록 (`/api/v1` 프리픽스) |
| `api/v1/download.py` | 라우터 등록 (`/api/v1` 프리픽스) |
| `api/health.py` | 라우터 등록 (프리픽스 없음) |

**수정 시 주의**: Lifespan 함수에서 `registry.load_all_models()`가 호출되므로, 서버 시작 시 모델 로드 순서 변경 시 이 파일을 수정합니다.

---

### `app/api/v1/arrange.py` — 편곡 요청 엔드포인트

| 종속 모듈 | 사용 방식 |
|-----------|-----------|
| `schemas/request.py` | `ArrangeRequest` 스키마로 요청 바디 검증 |
| `schemas/response.py` | `ArrangeResponse` 스키마로 응답 구성 |
| `services/arrangement.py` | `process_arrangement()`을 `BackgroundTasks`로 호출 |

**동작**: 요청 수신 → `app.state.registry` 참조 → `process_arrangement`를 백그라운드 태스크로 등록 → 즉시 `202 Accepted` 반환

---

### `app/services/arrangement.py` — 편곡 오케스트레이터

**역할**: 전체 편곡 라이프사이클을 4단계로 관리하는 **중앙 조율자**.

| 종속 모듈 | 사용하는 함수 | 호출 시점 |
|-----------|-------------|-----------|
| `midi_processor.py` | `download_midi()` | Step 1 (10%) |
| `midi_processor.py` | `remap_original_tracks()` | Step 2 (20%) |
| `inference.py` | `resolve_target()` | Step 3 준비 |
| `inference.py` | `run_arrangement()` | Step 3 (80%) — `run_in_executor`로 동기 함수 비동기 래핑 |
| `callback.py` | `send_callback()` | 각 단계 진행률 전송 |
| `callback.py` | `send_callback_with_file()` | Step 4 (100%) — 결과 MIDI 전송 |
| `core/config.py` | `settings.RESULTS_DIR` | 결과 파일 저장 경로 |

**에러 처리**: `try/except`로 전체 감싸며, 실패 시 `failed` 콜백 전송. `finally`에서 임시 파일(다운로드 MIDI + 결과 MIDI) 정리.

**수정 시 주의**: 이 파일은 Step 순서, 진행률 비율, 콜백 payload 구조를 관리합니다. 새 전처리 단계를 추가하려면 이 파일의 Step 구조를 변경하세요.

---

### `app/services/callback.py` — 콜백 전송 모듈

**역할**: Main Server에 진행률과 결과를 HTTP로 전송.

| 함수 | 용도 | 재시도 | 타임아웃 |
|------|------|--------|---------|
| `send_callback()` | JSON 진행률 전송 | 3회, 지수 백오프 | 10초 |
| `send_callback_with_file()` | MIDI 파일 + 메타데이터 전송 | 5회, 지수 백오프 | 60초 |

**인증**: `X-Callback-Secret` 헤더에 `callbackSecret` 전달.

**수정 시 주의**: 재시도 횟수, 타임아웃 변경 시 이 파일의 `max_retries`, `timeout` 파라미터를 조정하세요.

---

### `app/services/midi_processor.py` — MIDI 다운로드 + 재매핑

**역할**: 원본 MIDI 파일을 다운로드하고, `mappings`에 따라 트랙/채널의 악기를 변경하거나 삭제합니다.

| 함수 | 역할 | 호출자 |
|------|------|--------|
| `download_midi(url)` | URL에서 MIDI 스트리밍 다운로드 → 임시 파일 저장 | `arrangement.py` |
| `remap_original_tracks(path, mappings)` | 트랙 재매핑 (Type 0/1 자동 판별) | `arrangement.py` |
| `_remap_type1(mid, mappings)` | Type 1: `trackIndex`로 트랙별 처리 | 내부 |
| `_remap_type0(mid, mappings)` | Type 0: `trackIndex`를 채널로 해석하여 처리 | 내부 |

**`mappings` 처리 규칙**:
- `targetInstrumentId == 129` (DROP_INSTRUMENT_ID) → 해당 트랙/채널 삭제
- 그 외 → `program_change`를 해당 값으로 변경 (없으면 자동 삽입)

**안전장치**:
- Type 0 채널 삭제 시 **delta time 누적**으로 타이밍 보존
- `program_change` 없는 트랙에 **자동 삽입**
- Type 1 **메타 전용 트랙** (tempo, time_signature) 삭제 방지

**수정 시 주의**: `DROP_INSTRUMENT_ID` 상수 변경 시 Main Server 측의 Drop 카테고리 ID와 동기화 필요.

---

### `app/services/inference.py` — AI 추론 엔진

> ⚠️ **이 파일은 AI/ML 영역입니다.** 백엔드 관점에서 알아야 할 것만 기술합니다.

| 공개 함수 | 백엔드에서의 사용 |
|-----------|-----------------|
| `resolve_target(instrument_id)` | MIDI Program 번호(0~128) → 내부 악기 그룹 이름 변환 |
| `run_arrangement(song_path, target, genre, ...)` | **동기 함수** — CPU/GPU 블로킹이므로 반드시 `run_in_executor`로 호출 |

**`run_arrangement` 파라미터**: `arrangement.py`에서 모든 인자를 주입합니다. 백엔드에서 직접 호출하지 마세요.

**기본 모델 설정**: `app/core/model_registry.py`의 `registry.json` 파일에서 `default` 필드로 기본 모델을 지정합니다.

---

### `app/core/model_registry.py` — 모델 레지스트리

**역할**: `registry.json`을 읽어 모델을 로드하고, `modelType`으로 선택할 수 있게 합니다.

| 클래스/메서드 | 역할 |
|-------------|------|
| `LoadedModel` | 로드된 모델 + vocab + device를 묶는 데이터 클래스 |
| `ModelRegistry.__init__(model_dir)` | 모델 디렉토리 경로 설정 |
| `ModelRegistry.load_all_models()` | `registry.json`의 모든 모델 로드 (앱 시작 시 1회) |
| `ModelRegistry.get_model(model_type)` | `modelType`으로 모델 선택 (None이면 기본 모델) |
| `ModelRegistry.list_models()` | 로드된 모델 타입 목록 반환 |

**`registry.json` 형식**:

```json
{
  "version": "v2",
  "default": "qwen2.5",
  "models": [
    {
      "type": "qwen2.5",
      "name": "Tutti Unified v1",
      "path": "best",
      "description": "Qwen2.5-0.5B 기반 13그룹 통합 모델"
    }
  ]
}
```

- `type`: API의 `modelType` 필드와 매칭되는 키
- `path`: `MODEL_DIR` 하위의 체크포인트 디렉토리 이름
- `default`: 기본 모델 타입 (요청에 `modelType`이 없을 때 사용)

**새 모델 추가 시**: `registry.json`에 엔트리 추가 → `_load_single_model()`에 새 model_type 분기 추가 → GCS에 체크포인트 업로드

---

### `app/schemas/request.py` — 요청 스키마

| 클래스 | 사용처 |
|--------|--------|
| `Mapping` | `trackIndex` + `targetInstrumentId` 쌍 |
| `GenreType` | `Literal` 장르 검증 (7종) |
| `ArrangeRequest` | 편곡 요청 전체 바디 모델 |

**수정 시 주의**: 필드 추가/변경 시 Main Server의 요청 생성 로직과 동기화 필요.

---

### `app/schemas/response.py` — 응답 스키마

| 클래스 | 사용처 |
|--------|--------|
| `ArrangeResponse` | `POST /api/v1/arrange` 즉시 응답 |
| `HealthResponse` | `GET /health` 응답 |
| `LoadedInstrument` | 헬스체크에서 로드된 악기 정보 |

---

## 9. 수정 가이드 — "이런 걸 바꾸고 싶을 때"

### 📌 콜백 URL/헤더 변경

| 수정 파일 | 위치 |
|-----------|------|
| `app/services/callback.py` | `headers` 딕셔너리, `send_callback()` / `send_callback_with_file()` |

### 📌 콜백 payload에 필드 추가

| 수정 파일 | 위치 |
|-----------|------|
| `app/services/arrangement.py` | `send_callback()` 호출부의 payload 딕셔너리 |

### 📌 API에 새 필드 추가

| 수정 파일 | 위치 |
|-----------|------|
| `app/schemas/request.py` | `ArrangeRequest` 클래스에 필드 추가 |
| `app/services/arrangement.py` | 새 필드를 사용하는 로직 추가 |

### 📌 새 API 엔드포인트 추가

| 수정 파일 | 위치 |
|-----------|------|
| `app/api/v1/` | 새 라우터 파일 생성 |
| `app/main.py` | `app.include_router()` 추가 |

### 📌 MIDI 다운로드 경로/방식 변경

| 수정 파일 | 위치 |
|-----------|------|
| `app/services/midi_processor.py` | `download_midi()` 함수 |

### 📌 환경 변수 추가

| 수정 파일 | 위치 |
|-----------|------|
| `app/core/config.py` | `Settings` 클래스에 필드 추가 |
| `k8s/base/deployment.yaml` | `containers[0].env` 에 값 추가 |
| `.env.example` | 예시 업데이트 |

### 📌 새 모델 추가

| 수정 파일 | 위치 |
|-----------|------|
| `registry.json` | 새 모델 엔트리 추가 |
| `app/core/model_registry.py` | `_load_single_model()`에 새 `model_type` 분기 추가 |
| GCS 버킷 | 체크포인트 파일 업로드 |

### 📌 K8s 리소스 변경 (CPU/메모리/GPU)

| 수정 파일 | 위치 |
|-----------|------|
| `k8s/base/deployment.yaml` | `resources.requests` / `resources.limits` |

### 📌 스케일링 정책 변경

| 수정 파일 | 위치 |
|-----------|------|
| `k8s/base/keda-http.yaml` | `replicas.min` / `replicas.max` |

### 📌 Drop 악기 ID 변경

| 수정 파일 | 위치 | 동기화 대상 |
|-----------|------|-----------|
| `app/services/midi_processor.py` | `DROP_INSTRUMENT_ID` 상수 | Main Server의 Drop 카테고리 ID |

---

## 10. 모니터링 및 트러블슈팅

### 일반 로그 확인

```bash
# 실시간 로그
kubectl logs -f deployment/ai-server -n tutti

# 최근 100줄
kubectl logs deployment/ai-server -n tutti --tail=100
```

### 자주 발생하는 문제

| 증상 | 원인 | 해결 |
|------|------|------|
| Pod가 `Pending` 상태 | GPU 노드 프로비저닝 중 (1~2분 소요) | 대기. `kubectl describe pod`로 이벤트 확인 |
| `startupProbe` 실패 | 모델 로드 시간 초과 (체크포인트 크기 과대) | `failureThreshold` 증가 또는 모델 경량화 |
| 콜백 실패 로그 | Main Server 다운 또는 네트워크 이슈 | 재시도 로직이 내장되어 있으므로 확인만 |
| `FileNotFoundError: 체크포인트 없음` | GCS 경로 불일치 또는 Init Container 실패 | `registry.json`의 `path`와 GCS 구조 확인 |
| OOM (메모리 초과) | 매우 큰 MIDI 파일 + 모델 메모리 부족 | `resources.limits.memory` 증가 |

### KEDA Zero-Scaling 확인

```bash
# KEDA HTTP 인터셉터 상태
kubectl get httpscaledobject -n tutti
kubectl get pods -n keda

# 현재 레플리카 수
kubectl get deployment ai-server -n tutti
```

---

## 11. 테스트

### 테스트 실행

```bash
# MIDI 프로세서 테스트 (GPU 불필요)
python -m pytest tests/test_midi_processor.py -v

# MIDI 오염 분석 테스트
python -m pytest tests/test_midi_corruption_analysis.py -v

# 전체
python -m pytest tests/ -v
```

### 테스트 커버리지

| 테스트 파일 | 테스트 대상 | 테스트 수 |
|------------|-----------|----------|
| `test_midi_processor.py` | `midi_processor.py`의 Type 0/1 재매핑, 삭제, 에지케이스 | 18개 |
| `test_midi_corruption_analysis.py` | delta time 보존, 메타 데이터 보호, 참조 오염 방지 등 | 16개 |

> **주의**: 추론 통합 테스트는 GPU 환경에서만 가능합니다. 로컬 CPU 환경에서는 MIDI 프로세서 테스트만 실행하세요.
