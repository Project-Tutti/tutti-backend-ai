FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for MIDI processing and ML binaries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    fluid-soundfont-gm \
    fluidsynth \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source
COPY app/ ./app/

# Environment defaults
ENV HOST=0.0.0.0
ENV PORT=8000
ENV LOG_LEVEL=info

# Exposed port
EXPOSE 8000

# Start server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
