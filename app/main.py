from contextlib import asynccontextmanager
from fastapi import FastAPI
from pathlib import Path

from app.core.config import settings
from app.core.model_registry import ModelRegistry
from app.api.v1 import arrange, download
from app.api import health

import logging

logging.basicConfig(level=settings.LOG_LEVEL.upper())
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Setup Logic
    logger.info("Initializing Model Registry and warming up models.")
    model_dir = Path(settings.MODEL_DIR)
    registry = ModelRegistry(model_dir)
    registry.load_all_models()

    app.state.registry = registry
    yield
    # Cleanup logic
    logger.info("Shutting down Application...")
    app.state.registry = None


app = FastAPI(title="Tutti AI Inference Server", lifespan=lifespan)

# Setup Routers
app.include_router(health.router)
app.include_router(arrange.router, prefix="/api/v1")
app.include_router(download.router, prefix="/api/v1")
