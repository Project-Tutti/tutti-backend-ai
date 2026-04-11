# 🏗️ 온프레미스 AI 서버 셋업 가이드 (Redis Streams 기반)

> **대상 서버**: Ubuntu + RTX 4090 (GPU 드라이버 설치 완료 상태)
> **목표**: GPU 자원 보호를 위한 단일 Redis Worker 컨테이너 무중단 동작

---

## 목차

1. [사전 확인](#1-사전-확인)
2. [Docker Engine 설치](#2-docker-engine-설치)
3. [NVIDIA Container Toolkit 설치](#3-nvidia-container-toolkit-설치)
4. [GCP Artifact Registry 인증](#4-gcp-artifact-registry-인증)
5. [모델 파일 배치](#5-모델-파일-배치)
6. [서비스 실행](#6-서비스-실행)
7. [서비스 관리 명령어](#7-서비스-관리-명령어)
8. [GitHub Actions Self-Hosted Runner 셋업 (권장)](#8-github-actions-self-hosted-runner-셋업-필수-권장)
9. [트러블슈팅](#9-트러블슈팅)

*(비고: 기존 HTTP 통신 기반의 아키텍처는 `REDIS_ARCHITECTURE.md`를 참고하여 전면 개편되었습니다.)*

---

## 1. 사전 확인

```bash
# GPU 드라이버 설치 확인
nvidia-smi

# Ubuntu 버전 확인
lsb_release -a
```

> [!IMPORTANT]
> `nvidia-smi`에서 GPU가 정상 인식되어야 합니다. 드라이버가 없으면 먼저 설치하세요:
> ```bash
> sudo apt install -y nvidia-driver-550
> sudo reboot
> ```

---

## 2. Docker Engine 설치

### 2.1 기존 Docker 잔여물 제거 (있을 경우)

```bash
sudo apt-get remove docker docker-engine docker.io containerd runc 2>/dev/null || true
```

### 2.2 Docker 공식 GPG 키 + 저장소 추가

```bash
sudo apt-get update
sudo apt-get install -y ca-certificates curl gnupg

sudo install -m 0755 -d /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
sudo chmod a+r /etc/apt/keyrings/docker.gpg

echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

### 2.3 Docker 설치

```bash
sudo apt-get update
sudo apt-get install -y \
  docker-ce \
  docker-ce-cli \
  containerd.io \
  docker-buildx-plugin \
  docker-compose-plugin
```

### 2.4 현재 유저를 docker 그룹에 추가 (sudo 없이 사용)

```bash
sudo usermod -aG docker $USER
newgrp docker
```

---

## 3. NVIDIA Container Toolkit 설치

> GPU 드라이버가 이미 있는 상태에서 Docker 컨테이너가 GPU에 접근할 수 있도록 합니다.

### 3.1 저장소 추가

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
```

### 3.2 설치

```bash
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```

### 3.3 Docker 런타임에 GPU 연동

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### 3.4 GPU 접근 테스트

```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

---

## 4. GCP Artifact Registry 인증

메인 서버와 동일한 GCP 프로젝트의 Artifact Registry에서 Docker 이미지를 pull합니다.

### 4.1 gcloud CLI 설치 (없는 경우)

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

### 4.2 Docker 인증 구성

```bash
# 대화형 인증
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
```

---

## 5. 모델 파일 배치

```bash
# 프로젝트 디렉토리 생성
mkdir -p ~/tutti-backend-ai/models/best

# 다운로드 (GCS 등에서)
gsutil cp gs://tutti-ai-models/v1/registry.json ~/tutti-backend-ai/registry.json
gsutil cp -r gs://tutti-ai-models/v1/best/ ~/tutti-backend-ai/models/best/
```

---

## 6. 서비스 실행

### 6.1 프로젝트 클론 (최초 1회)

```bash
cd ~
git clone https://github.com/Project-Tutti/tutti-backend-ai.git
cd tutti-backend-ai
```

> **[권장] Git Sparse-Checkout 설정 (서버 디렉토리 최적화)**
> 온프레미스 서버는 도커를 통해 실행되므로 소스코드가 물리 파일 시스템에 남아있을 필요가 없습니다. 다음 명령어를 실행하면 서버 구동에 필요한 핵심 파일만 남겨 최적화됩니다.
> ```bash
> git sparse-checkout init --cone
> git sparse-checkout set docker-compose.yml nginx models registry.json
> ```

### 6.2 환경 변수 설정 (.env)

```bash
nano .env
```

`.env` 파일에 아래 Redis 및 연결 정보를 설정합니다:

```env
GCP_PROJECT_ID=여기에값입력
AI_SERVER_API_KEY=콜백인증키
REDIS_HOST=xxx.upstash.io
REDIS_PORT=6379
REDIS_PASSWORD=xxx...
REDIS_TLS=true
LOG_LEVEL=info
```

### 6.3 서비스 시작

```bash
# AI Worker 단독 백그라운드 구동 (FastAPI 라우팅 없음)
docker compose up -d

# 상태 확인
docker compose ps
```

---

## 7. 서비스 관리 명령어

### 로그 확인

```bash
# Redis Worker 루프 모니터링
docker compose logs -f ai-worker
```

### 서비스 재시작 및 업데이트

```bash
# 이미지 업데이트 및 롤아웃 (Zero-downtime grace period 10min)
docker compose pull
docker compose up -d --remove-orphans
```

---

## 8. GitHub Actions Self-Hosted Runner 셋업 (필수 권장)

보안과 배포 자동화를 위해 GitHub Actions 클라우드와 터널링되어 안전하게 무중단 배포를 실행하는 Runner 설치를 권장합니다.
GitHub Repository의 `Settings` -> `Actions` -> `Runners` 메뉴에서 지침을 따르세요.

---

## 9. 트러블슈팅

### GPU가 컨테이너에서 인식되지 않는 경우

```bash
# 직접 GPU 테스트
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 안되면 NVIDIA Container Toolkit 재설정
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Worker 큐 수신 실패 (Redis 연결 에러)

```bash
# 워커 로그 확인 (Redis 연결 실패 시 Traceback 출력됨)
docker compose logs ai-worker | grep "Redis"

# REDIS_TLS=true 여부 점검 (Upstash 사용 시 필수)
cat .env | grep REDIS_TLS
```

### 메모리 부족 (OOM)

본 아키텍처는 단일 Worker 동기 처리(Synchronous Queue Pull) 방식이므로 **일반적인 상황에서는 OOM이 발생하지 않습니다.**
단, 모델 가중치 파일 크기가 24GB를 근접하여 초과할 경우 `registry.json`의 active 모델 개수를 조절해 주세요.
