# Tutti AI Server — 백엔드 연동 및 인프라 가이드

> **대상 독자**: 백엔드 API, DevOps, 인프라 엔지니어  
> **초점 영역**: 백엔드 ↔ AI 간 통신 규약, 콜백 및 환경 셋업  
> **업데이트 내역**: v3 (Redis Streams 기반 비동기 워커 구조 도입)

---

> [!WARNING]
> 기존에 사용하던 **FastAPI, KEDA, Cloudflared Tunnel 인프라는 전면 폐기**되었습니다. 새로운 아키텍처의 상세 원리는 [REDIS_ARCHITECTURE.md](./REDIS_ARCHITECTURE.md) 문서를 참조해 주십시오. 

---

## 목차

1. [시스템 개요 (Redis Streams 통신)](#1-시스템-개요-redis-streams-통신)
2. [백엔드 요청 스키마 (`XADD`)](#2-백엔드-요청-스키마-xadd)
3. [결과 수신 (HTTP Callback)](#3-결과-수신-http-callback)
4. [환경 변수 및 배포 구성](#4-환경-변수-및-배포-구성)
5. [장애 복구 (Recovery) 매커니즘](#5-장애-복구-recovery-매커니즘)

---

## 1. 시스템 개요 (Redis Streams 통신)

Tutti AI Server는 **Redis Streams를 매개체로 하는 100% 비동기 워커(Worker) 시스템**입니다.

```
┌────────────┐     XADD (JSON Payload)       ┌───────────────────┐
│ Main Server│ ────────────────────────────→ │ Upstash Redis     │
│ (Spring)   │ ←── HTTP Callback (Result) ─  │                   │
└────────────┘                               └─────────┬─────────┘
                                                       │ XREADGROUP
                                                       ▼
                                             ┌───────────────────┐
                                             │ GPU Server (4090) │
                                             │ [ ai-worker ]     │
                                             └───────────────────┘
```

**핵심 변경점**:
- **OOM 원천 차단**: GPU(ai-worker)가 오직 한 번에 1개의 작업만 Redis에서 꺼내(Pull)갑니다. 트래픽은 Redis가 흡수합니다.
- **포트 포워딩 불필요**: 온프레미스 장비는 큐를 읽고 외부(Main Server)로 POST만 쏘므로, 인바운드 방화벽을 열거나 Nginx를 둘 필요가 없습니다.

---

## 2. 백엔드 요청 스키마 (`XADD`)

백엔드는 `ai:arrange:queue` (또는 지정된 stream-key) 채널에 아래 구조의 JSON을 문자열 직렬화하여 적재합니다.

**JSON Payload 예시**:
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

- **`mappings.targetInstrumentId`가 `129`인 경우**: 트랙 드롭(Drop).
- **`callbackUrl`**: 폴링 없이 AI가 작업 완료/실패 시 능동적으로 찔러줄 주소.

---

## 3. 결과 수신 (HTTP Callback)

작업이 끝나면 워커가 백엔드의 `callbackUrl`로 HTTP 요청을 전송합니다. 

### 성공 (파일 반환)
- **Content-Type**: `multipart/form-data`
- **Payload**:
  - `metadata`: JSON `{"projectId": 1, "versionId": 1, "status": "complete", "progress": 100}`
  - `file`: 생성 및 믹싱이 완료된 `audio/midi` 바이너리 파일

### 실패 (에러 반환)
- **Content-Type**: `application/json`
- **Payload**: 
  ```json
  {
    "projectId": 1,
    "versionId": 1,
    "status": "failed",
    "progress": 0,
    "errorMessage": "GPU OutOfMemory 혹은 다운로드 실패 에러"
  }
  ```

---

## 4. 환경 변수 및 배포 구성

서버 배포는 `docker-compose.yml`을 통하며 필수 환경변수는 `.env` 스크립트로 CI에서 주입됩니다.

| 환경 변수 | 의미 |
|-----------|------|
| `REDIS_HOST` | Upstash 등 Redis 엔드포인트 도메인 |
| `REDIS_PASSWORD` | 인가용 패스워드 |
| `REDIS_TLS` | SSL 연결 강제 (`true` 필수 권장) |
| `AI_SERVER_API_KEY` | 모델 헬스체크 및 내부 보안 통신용 여분 키 |
| `LOG_LEVEL` | 기본값 `info` |

배포 및 설치에 관련한 물리적 가이드는 [ONPREM_SETUP_GUIDE.md](./ONPREM_SETUP_GUIDE.md)를 확인하십시오.

---

## 5. 장애 복구 (Recovery) 매커니즘

- 백엔드가 요청을 보냈는데 수 분이 지나도 응답이 없는 경우 2가지 상황이 있습니다:
  1. **큐 대기열이 길다**: Worker가 바쁜 경우. Redis에는 안전하게 보관 중입니다.
  2. **처리 중 Worker가 죽었다**: 해당 작업은 `XPENDING`(처리 중 장애) 상태로 빠집니다.
- `worker.py` 내부 로직에 의해, 처리시간이 **10분을 초과**한 비정상 작업은 현재 살아있는 워커가 주도권(`XCLAIM`)을 빼앗아 처음부터 다시 처리하도록 자가 치유(Self-Healing) 로직이 삽입되어 있습니다.
- 백엔드 개발자는 별도의 복구 로직을 짜지 않아도 시스템이 스스로 고아 메시지(Orphaned Message)를 처리합니다.
