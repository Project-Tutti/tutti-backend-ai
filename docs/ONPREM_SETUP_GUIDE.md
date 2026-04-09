# 🏗️ 온프레미스 AI 서버 셋업 가이드

> **대상 서버**: Ubuntu + RTX 4090 (GPU 드라이버 설치 완료 상태)
> **목표**: Docker Compose 기반 AI 편곡 서버 2대 + Nginx + Cloudflare Tunnel 구동

---

## 목차

1. [사전 확인](#1-사전-확인)
2. [Docker Engine 설치](#2-docker-engine-설치)
3. [NVIDIA Container Toolkit 설치](#3-nvidia-container-toolkit-설치)
4. [Cloudflare Tunnel 설치 및 구성](#4-cloudflare-tunnel-설치-및-구성)
5. [GCP Artifact Registry 인증](#5-gcp-artifact-registry-인증)
6. [모델 파일 배치](#6-모델-파일-배치)
7. [서비스 실행](#7-서비스-실행)
8. [서비스 관리 명령어](#8-서비스-관리-명령어)
9. [트러블슈팅](#9-트러블슈팅)
10. [GPU 시분할 및 MPS 설정 (선택)](#10-gpu-시분할-및-mps-설정-선택)

---

## 1. 사전 확인

```bash
# GPU 드라이버 설치 확인
nvidia-smi

# Ubuntu 버전 확인
lsb_release -a

# 예상 출력:
# NVIDIA-SMI 5xx.xx   Driver Version: 5xx.xx   CUDA Version: 12.x
# Ubuntu 22.04 LTS (또는 24.04 LTS)
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

### 2.5 설치 확인

```bash
docker --version
# Docker version 27.x.x

docker compose version
# Docker Compose version v2.x.x

# 테스트 컨테이너 실행
docker run --rm hello-world
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

> [!TIP]
> 위 명령이 호스트의 `nvidia-smi`와 동일한 GPU 정보를 출력하면 성공입니다.

---

## 4. Cloudflare Tunnel 설치 및 구성

### 4.1 cloudflared 설치

```bash
# 바이너리 직접 설치
curl -L https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 \
  -o /usr/local/bin/cloudflared
chmod +x /usr/local/bin/cloudflared

# 설치 확인
cloudflared --version
```

### 4.2 Cloudflare 로그인

```bash
cloudflared tunnel login
# → 브라우저가 열립니다. 도메인이 등록된 Cloudflare 계정으로 로그인
```

### 4.3 터널 생성

```bash
cloudflared tunnel create tutti-ai
# → Tunnel ID가 출력됩니다. 기록해두세요.
# → ~/.cloudflared/에 인증서 파일이 생성됩니다.
```

### 4.4 DNS 레코드 연결

```bash
cloudflared tunnel route dns tutti-ai <AI_SERVER_HOST>
# → <AI_SERVER_HOST> CNAME 레코드가 자동 생성됩니다.
```

### 4.5 터널 토큰 발급 (Docker Compose용)

Cloudflare Dashboard에서 터널 토큰을 발급받습니다:

1. [Cloudflare Zero Trust](https://one.dash.cloudflare.com/) 대시보드 접속
2. **Networks → Tunnels** → `tutti-ai` 터널 선택
3. **Configure** → **Install and run a connector** 섹션에서 토큰 복사
4. 토큰을 `.env` 파일의 `TUNNEL_TOKEN`에 설정

> [!IMPORTANT]
> 터널 설정 시 **Public Hostname**을 추가해야 합니다:
> - **Subdomain**: `<서브도메인>`
> - **Domain**: `<도메인>`
> - **Service**: `http://nginx:80`

---

## 5. GCP Artifact Registry 인증

메인 서버와 동일한 GCP 프로젝트의 Artifact Registry에서 Docker 이미지를 pull합니다.

### 5.1 gcloud CLI 설치 (없는 경우)

```bash
curl https://sdk.cloud.google.com | bash
exec -l $SHELL
gcloud init
```

### 5.2 Docker 인증 구성

```bash
# 방법 1: gcloud 통한 인증 (대화형)
gcloud auth configure-docker us-central1-docker.pkg.dev --quiet

# 방법 2: Service Account 키 파일 사용 (자동화용)
# GitHub Actions에서 사용하는 것과 같은 SA 키 파일을 복사 후:
cat sa-key.json | docker login -u _json_key --password-stdin https://us-central1-docker.pkg.dev
```

### 5.3 pull 테스트

```bash
docker pull us-central1-docker.pkg.dev/<PROJECT_ID>/tutti/ai-server:latest
```

---

## 6. 모델 파일 배치

```bash
# 프로젝트 디렉토리 생성
mkdir -p ~/tutti-backend-ai/models/best

# 모델 파일 다운로드 (GCS에서)
gsutil cp gs://tutti-ai-models/v1/registry.json ~/tutti-backend-ai/models/registry.json
gsutil cp -r gs://tutti-ai-models/v1/best/ ~/tutti-backend-ai/models/best/

# 또는 직접 복사 (USB, SCP 등)
# scp user@source-server:/path/to/model.safetensors ~/tutti-backend-ai/models/best/
```

### 디렉토리 구조 확인

```bash
tree ~/tutti-backend-ai/models/
# models/
# ├── registry.json
# └── best/
#     └── model.safetensors
```

---

## 7. 서비스 실행

### 7.1 프로젝트 클론 (최초 1회)

```bash
cd ~
git clone https://github.com/Project-Tutti/tutti-backend-ai.git
cd tutti-backend-ai
```

### 7.2 환경 변수 설정

```bash
cp .env.production .env

# .env 파일 편집
nano .env
```

`.env` 파일에서 반드시 설정할 값:

```env
TUNNEL_TOKEN=<Cloudflare 대시보드에서 복사한 토큰>
```

### 7.3 서비스 시작

```bash
# AI Worker 2대 + Nginx + Cloudflare Tunnel 시작
docker compose up -d --scale ai-server=2

# 상태 확인
docker compose ps
```

### 7.4 동작 확인

```bash
# 로컬에서 health check
curl http://localhost:8080/health

# 외부에서 확인 (다른 컴퓨터에서)
curl https://<AI_SERVER_HOST>/health
```

---

## 8. 서비스 관리 명령어

### 로그 확인

```bash
# 전체 로그
docker compose logs -f

# 특정 서비스 로그
docker compose logs -f ai-server
docker compose logs -f nginx
docker compose logs -f cloudflared
```

### 서비스 재시작

```bash
# 전체 재시작
docker compose restart

# AI 서버만 재시작
docker compose restart ai-server
```

### 이미지 업데이트 (새 배포)

```bash
cd ~/tutti-backend-ai
git pull origin main

# 새 이미지 pull + 재시작
docker compose pull ai-server
docker compose up -d --scale ai-server=2
```

### 서비스 중지

```bash
docker compose down
```

### Worker 수 조정

```bash
# 3대로 확장
docker compose up -d --scale ai-server=3

# 1대로 축소
docker compose up -d --scale ai-server=1
```

---

## 9. 트러블슈팅

### GPU가 컨테이너에서 인식되지 않는 경우

```bash
# Docker 런타임 확인
docker info | grep -i runtime
# → nvidia 런타임이 목록에 있어야 함

# 직접 GPU 테스트
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 안되면 NVIDIA Container Toolkit 재설정
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### Cloudflare Tunnel 연결이 안 되는 경우

```bash
# 터널 상태 확인
docker compose logs cloudflared

# 로컬에서 Nginx 직접 접근 테스트
curl http://localhost:8080/health

# TUNNEL_TOKEN이 올바른지 확인
cat .env | grep TUNNEL_TOKEN
```

### 모델 로드 실패

```bash
# 모델 파일 존재 확인
ls -la ~/tutti-backend-ai/models/best/

# registry.json 확인
cat ~/tutti-backend-ai/models/registry.json

# AI 서버 로그에서 에러 확인
docker compose logs ai-server | grep -i error
```

### 메모리 부족 (OOM)

```bash
# GPU 메모리 사용량 확인
nvidia-smi

# Worker가 너무 많으면 축소
docker compose up -d --scale ai-server=1
```

---

## 10. GPU 시분할 및 MPS 설정 (선택)

### 기본 동작: 시분할 (Time-Slicing)

2대의 AI 컨테이너가 동일 GPU를 사용할 때, 기본적으로 **시분할 방식**으로 GPU를 공유합니다.
별도 설정 없이 동작하며, Qwen2.5-0.5B 2대 정도는 충분히 처리 가능합니다.

- 단독 실행 대비 ~10-20% 오버헤드 발생 가능
- 대부분의 워크로드에서 체감 차이 없음

### CUDA MPS (Multi-Process Service) — 선택 사항

동시 요청이 빈번하고 개별 추론 시간이 중요한 경우, MPS를 활성화하면 GPU 활용률이 향상됩니다.

```bash
# MPS 제어 데몬 시작
sudo nvidia-cuda-mps-control -d

# MPS 상태 확인
echo get_server_list | nvidia-cuda-mps-control

# MPS 종료
echo quit | nvidia-cuda-mps-control
```

> [!NOTE]
> - MPS는 **동일 GPU에서 다수 프로세스의 GPU 커널을 동시 실행**할 수 있게 해줍니다.
> - 0.5B 모델 2대에서는 기본 시분할로 충분하므로-**초기에는 설정하지 않아도 됩니다.**
> - 성능 병목이 확인된 후에 MPS 적용을 검토하세요.

---

## 아키텍처 요약

```
외부 요청 (Main Server, GKE)
    │
    │  HTTPS
    ▼
Cloudflare Edge (<AI_SERVER_HOST>)
    │
    │  Cloudflare Tunnel (암호화)
    ▼
┌─────────────────────────────────────────┐
│  학교 서버 (RTX 4090)                      │
│                                         │
│  cloudflared ─── Nginx (:80)            │
│                   │                     │
│            ┌──────┴──────┐              │
│            ▼             ▼              │
│      ai-server-1   ai-server-2          │
│       (:8000)       (:8000)             │
│            │             │              │
│            └──────┬──────┘              │
│                   ▼                     │
│              RTX 4090 GPU               │
│           (시분할 공유)                   │
│                                         │
│         ./models/ (로컬 볼륨)             │
└─────────────────────────────────────────┘
```
