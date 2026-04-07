from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    LOG_LEVEL: str = "info"
    MODEL_DIR: str = "/models"
    RESULTS_DIR: str = "/tmp/results"
    GPU_ID: int = 0                     # 사용할 GPU 번호

    class Config:
        env_file = ".env"


settings = Settings()

# Ensure results dir exists
Path(settings.RESULTS_DIR).mkdir(parents=True, exist_ok=True)
