# Tutti AI Inference Server

FastAPI based Python inference server for the Tutti Music Platform.
Takes multi-track MIDI input, generates complementary instrumental tracks using PyTorch and `anticipation` models, and returns a unified MIDI file.

## Architecture

- **Web Framework**: FastAPI (Uvicorn backend)
- **Model Registry**: Dynamic loading based on `registry.json` from Google Cloud Storage
- **Orchestration**: Asynchronous execution tracking job progress and firing HTTP webhooks
- **Containerization**: Lean Docker container meant to be deployed as a GKE Deployment
- **Scaling**: CPU utilized HPA scaling

## Environment Setup

Copy `.env.example` to `.env`:

```
HOST=0.0.0.0
PORT=8000
LOG_LEVEL=info
MODEL_DIR=/models
RESULTS_DIR=/tmp/results
GCS_MODEL_BUCKET=tutti-ai-models
MODEL_VERSION=v1
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

4. Run server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Adding New Instruments

Hot-loading is supported:

1. Upload the `.pt` or `.onnx` file to GCS
2. Add corresponding JSON block in `registry.json` tracking the `midi_program` target
3. Perform a rollout restart on the `ai-server` deployment `kubectl rollout restart deployment/ai-server -n tutti`

> _Note: Model cache architecture (`app/core/model_cache.py`) is stubbed pending Google Colab definitions porting._
