# ═══════════════════════════════════════════════════════════
# Stage 1: Builder — pip install 전용 (CUDA 빌드 도구 포함)
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
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# pip를 Python 3.11로 연결
RUN python3.11 -m ensurepip && \
    python3.11 -m pip install --upgrade pip

COPY requirements.txt .
RUN python3.11 -m pip install --no-cache-dir -r requirements.txt

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
    build-essential \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    curl \
    libsndfile1 \
    fluid-soundfont-gm \
    fluidsynth \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3.11 /usr/bin/python3 \
    && ln -sf /usr/bin/python3.11 /usr/bin/python

# Builder에서 설치된 Python 패키지만 복사
COPY --from=builder /usr/local/lib/python3.11/dist-packages /usr/local/lib/python3.11/dist-packages
COPY --from=builder /usr/lib/python3/dist-packages /usr/lib/python3/dist-packages
COPY --from=builder /usr/lib/python3.11 /usr/lib/python3.11
COPY --from=builder /usr/local/bin /usr/local/bin

# 애플리케이션 소스 복사
COPY app/ ./app/
COPY ai_core/ ./ai_core/
COPY contracts/ ./contracts/
COPY worker.py ./worker.py

# Environment defaults
ENV HOST=0.0.0.0
ENV PORT=8000
ENV LOG_LEVEL=info
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

EXPOSE 8000

CMD ["python3.11", "worker.py"]
