# 🏗️ Tutti AI: Redis Streams 아키텍처

> **도입 배경**: 기존 HTTP 기반(FastAPI) 인퍼런스 서버가 겪던 GPU OOM(메모리 초과), 로드밸런싱 한계, Nginx 타임아웃 문제를 해결하기 위해 도입된 **비동기 큐 기반의 단일 워커(Single-Worker) 시스템**입니다.

---

## 1. 코어 아키텍처 (Message Queue)

```mermaid
graph TD
    subgraph Spring Boot (Main Backend)
        API[Client API] -->|1. LPUSH / XADD| Redis[(GKE Redis Streams)]
        CallbackEndpoint[HTTP Callback Receiver]
    end

    subgraph GPU Server (On-Premise)
        Worker[worker.py (Single-Thread)]
        Redis -->|2. XREADGROUP (Polling)| Worker
        Worker -->|3. run_arrangement()| GPU[RTX 4090 (24GB)]
        Worker -->|4. POST Results| CallbackEndpoint
        Worker -.->|5. XACK| Redis
    end
```

### 아키텍처 설계의 이유
1. **GPU 독점 방지 (OOM 완벽 해결)**
   - GPU 추론은 파이썬 GIL을 점유하고 VRAM을 대량 소모합니다.
   - 단일 스레드로 무한 루프를 도는 `worker.py`가 **한 번에 딱 1개의 작업만** 가져오도록 강제하여, 외부에서 수만 건의 요청이 오더라도 서버가 터지지 않는 완벽한 Rate Limiting을 구현합니다.
2. **비동기 격리**
   - 백엔드는 Redis에 던져두기만 하면 되므로 HTTP 트래픽(Nginx, Cloudflare 등) 설정이 전면 제거되었습니다.

---

## 2. 작업 처리 흐름 (Stream Lifecycle)

Redis Streams의 **소비자 그룹 (Consumer Group)** 기능을 활용해 신뢰성을 보장합니다.

1. **작업 생성 (`XADD`)**: 메인 백엔드가 `ai:arrange:stream` 큐에 JSON 작업을 푸시.
2. **작업 할당 (`XREADGROUP`)**: `ai-worker`가 큐에서 Block 대기(또는 Polling)하며 작업을 1개 가져옵니다. (이때 메시지는 `Pending` 상태가 됨)
3. **추론 (Inference)**: GPU 연산 (3~5분 소요).
4. **결과 전송**: HTTP POST로 메인 백엔드에 완성된 MIDI를 전송. (지수 백오프 기반 재시도 로직 포함)
5. **작업 완료 (`XACK`)**: 처리가 끝난 메시지를 큐에서 영구 삭제.

---

## 3. 고가용성 (크래시 자동 복구) 메커니즘

AI 서버의 전원이 꺼지거나 Docker 데몬이 크래시 나는 "최악의 상황"을 막기 위한 방어 로직입니다.

### 3.1 XPENDING & XCLAIM
GPU OOM 등으로 `worker.py`가 중간에 죽으면, 해당 작업은 Redis에 `Pending` (진행 중이지만 완료 안 됨) 상태로 남습니다.
- 워커는 1분 주기로 `recover_pending()` 함수를 실행합니다.
- `Pending`된 지 **10분이 초과**한 메시지(이전 워커가 죽었다고 판단)를 발견하면, 해당 작업을 훔쳐와서(`XCLAIM`) 다시 처리합니다.

### 3.2 중복 처리 방지 (10분 임계값)
추론에는 обычно 3~5분이 소요되므로, 10분 임계값을 설정하여 멀쩡히 처리 중인 남의 작업을 가져오는 참사를 방지했습니다. (백엔드 GC가 15분에 FAILED 처리하는 것과 보조를 맞춤)

---

## 4. 모니터링 및 헬스체크

기존 FastAPI 헬스체크 방식이었던 `HTTP /health`는 내부 프로세스로 내장되었습니다.

1. `worker.py` 구동 시, 내부적으로 포트 8000번에 경량 `http.server`를 백그라운드 스레드로 오픈.
2. 메인 스레드에 있는 `ModelRegistry` 메모리 로딩이 끝나기 전까지는 헬스체크가 블로킹됨 (실제 GPU 준비 완료 전까지 Healthy가 뜨지 않음).
3. Docker Compose의 `healthcheck` 설정이 이 포트를 찔러서 컨테이너의 상태를 확인.
4. **CI 배포 스크립트(`ci-ai.yml`)** 에서 `docker compose ps` 출력값을 감지하여 무중단 롤아웃 로직이 연동됨.

---

## 5. 로컬 테스트 방법

로컬에서 디버그 시, 큐를 통하지 않고 다이렉트로 HTTP로 찔러보고 싶을 수 있습니다.
이 경우 `docker-compose.yml`에는 과거 FastAPI 기반 서버(`ai-server`) 코드가 잠들어 있습니다.

```bash
# Redis Worker 모드 (운영 환경 기본값)
docker compose up -d

# HTTP 강제 구동 모드 (디버깅)
# Warning: GPU 겹침 방지를 위해 ai-worker를 꺼두는 것이 유리
docker compose up -d --scale ai-server=1 --scale nginx=1
```
