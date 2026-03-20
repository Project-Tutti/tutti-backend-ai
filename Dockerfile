# ═══════════════════════════════════════════════════════════
# Stage 1: Builder — pip install 전용 (git, 빌드 도구 포함)
# ═══════════════════════════════════════════════════════════
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ═══════════════════════════════════════════════════════════
# Stage 2: Runtime — 최소 런타임 이미지 (git, pip 캐시 제외)
# ═══════════════════════════════════════════════════════════
FROM python:3.11-slim

WORKDIR /app

# 런타임에 필요한 시스템 라이브러리만 설치
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    fluid-soundfont-gm \
    fluidsynth \
    && rm -rf /var/lib/apt/lists/*

# Builder에서 설치된 Python 패키지만 복사 (git, 빌드 도구, 캐시 제외)
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# 애플리케이션 소스 복사
COPY app/ ./app/

# Environment defaults
ENV HOST=0.0.0.0
ENV PORT=8000
ENV LOG_LEVEL=info

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
