# ═══════════════════════════════════════════════════════════
# Stage 1: Builder — uv로 의존성 설치 (CUDA 빌드 도구 포함)
# ═══════════════════════════════════════════════════════════
FROM nvidia/cuda:12.1.0-devel-ubuntu22.04 AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

WORKDIR /build

# Python 3.11 + 빌드 도구 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# uv 설치
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

# 의존성 캐시: 소스 없이 pyproject.toml만 먼저 복사
COPY pyproject.toml ./
# uv.lock이 있으면 복사 (없으면 무시)
COPY uv.loc[k] ./

# venv 생성 + 의존성 설치 (소스 변경에 영향받지 않는 Docker 캐시 레이어)
RUN uv venv --python 3.11 /build/.venv \
    && uv sync --no-dev --extra gpu --no-install-project

# ═══════════════════════════════════════════════════════════
# Stage 2: Runtime — 최소 CUDA 런타임 이미지
# ═══════════════════════════════════════════════════════════
FROM nvidia/cuda:12.1.0-base-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

WORKDIR /app

# Python 3.11 + 런타임 시스템 라이브러리 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    build-essential \
    curl \
    libsndfile1 \
    fluid-soundfont-gm \
    fluidsynth \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Builder에서 설치된 venv 복사
COPY --from=builder /build/.venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

# 애플리케이션 소스 복사
COPY ai_core/ ./ai_core/
COPY contracts/ ./contracts/
COPY app/ ./app/
COPY worker.py ./worker.py

# Environment defaults
ENV HOST=0.0.0.0
ENV PORT=8000
ENV LOG_LEVEL=info
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000

CMD ["python3.11", "worker.py"]
