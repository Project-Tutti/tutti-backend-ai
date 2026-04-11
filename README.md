# Tutti AI Inference Server

FastAPI based Python inference server for the Tutti Music Platform.
Takes multi-track MIDI input, generates complementary instrumental tracks using PyTorch and `anticipation` models, and returns a unified MIDI file.

## Architecture

- **Message Broker**: Upstash Serverless Redis (Streams)
- **Worker Execution**: Pure Python single-threaded event loop (No web framework overhead)
- **Model Registry**: Dynamic loading based on `registry.json` from Google Cloud Storage
- **Orchestration**: Asynchronous polling (`XREADGROUP`) and HTTP webhooks for progress/completion
- **Containerization**: Lean Docker container meant to be deployed via Docker Compose on-premise
- **Scaling**: Controlled vertically by GPU VRAM, single worker execution block to avoid OOM

## Environment Setup

Copy `.env.example` to `.env`:

```env
GCP_PROJECT_ID=tutti-production
AI_SERVER_API_KEY=xxx...
LOG_LEVEL=info
REDIS_HOST=xxx.upstash.io
REDIS_PORT=6379
REDIS_PASSWORD=xxx...
REDIS_TLS=true
```

## Running Locally

1. Create a `models/` directory locally with a synthetic `registry.json`
2. Define models per the schema format:

```json
{
  "version": "v1",
  "instruments": [
    {
      "midi_program": 40,
      "name": "Violin",
      "category": "Strings",
      "model_file": "violin.pt",
      "model_type": "pytorch"
    }
  ]
}
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Run worker (using docker-compose is recommended):

```bash
docker compose up -d
```
(Alternatively, to run natively: `python3.11 worker.py`)

## Adding New Instruments

Hot-loading is supported:

3. Restart the `ai-worker` container: `docker compose restart ai-worker`

> _Note: Model cache architecture (`app/core/model_cache.py`) is stubbed pending Google Colab definitions porting._
