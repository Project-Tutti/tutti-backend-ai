import json
import logging
from typing import Any
from pathlib import Path
from pydantic import BaseModel

logger = logging.getLogger(__name__)


class InstrumentConfig(BaseModel):
    midi_program: int
    name: str
    category: str
    model_file: str
    model_type: str


class ModelRegistry:
    """JSON 설정 파일 기반 모델 레지스트리."""

    def __init__(self, model_dir: Path):
        self._model_dir = model_dir
        self._instruments: dict[int, InstrumentConfig] = {}
        self._loaded_models: dict[int, Any] = {}
        self._load_registry()

    def _load_registry(self):
        registry_path = self._model_dir / "registry.json"
        if not registry_path.exists():
            logger.warning(
                f"registry.json not found in {self._model_dir}. Proceeding with empty registry."
            )
            return

        with open(registry_path) as f:
            data = json.load(f)

        for item in data.get("instruments", []):
            config = InstrumentConfig(**item)
            self._instruments[config.midi_program] = config

        logger.info(f"레지스트리 로드 완료: {len(self._instruments)}개 악기")

    def load_all_models(self):
        """앱 시작 시 모든 모델을 메모리에 로드"""
        total = len(self._instruments)
        loaded = 0
        for midi_program, config in self._instruments.items():
            model_path = self._model_dir / config.model_file
            if model_path.exists():
                try:
                    self._loaded_models[midi_program] = self._load_model(
                        model_path, config.model_type
                    )
                    logger.info(f"모델 로드 성공: {config.name} ({config.model_file})")
                    loaded += 1
                except Exception as e:
                    logger.error(
                        f"모델 로드 실패: {config.model_file} - Exception: {e}"
                    )
            else:
                logger.warning(f"모델 파일 없음: {model_path} ({config.name})")
        logger.info(f"전체 모델 로드 상태: {loaded}/{total}")

    def _load_model(self, path: Path, model_type: str) -> Any:
        # Pytorch
        if model_type == "pytorch":
            import torch

            # NOTE: Custom classes like GPT, ModelConfig must be available in context.
            # Referencing 'app.core.model_cache' stub implementation.
            from app.core.model_cache import GPT, ModelConfig

            ckpt = torch.load(path, map_location="cpu")
            c = ModelConfig()
            if "config" in ckpt:
                for k, v in ckpt["config"].items():
                    if hasattr(c, k):
                        setattr(c, k, v)

            # Note: The colab had NUM_LAYERS logic here
            model = GPT(c)
            # if cuda is available, push it
            if torch.cuda.is_available():
                model = model.cuda()
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()
            return model

        elif model_type == "onnx":
            import onnxruntime

            return onnxruntime.InferenceSession(str(path))
        else:
            raise ValueError(f"지원하지 않는 model_type: {model_type}")

    def get_model(self, midi_program: int) -> Any:
        if midi_program not in self._loaded_models:
            raise ValueError(
                f"지원하지 않는 악기 혹은 모델 로드 실패: midi_program={midi_program}"
            )
        return self._loaded_models[midi_program]

    def get_config(self, midi_program: int) -> InstrumentConfig:
        return self._instruments.get(midi_program)

    def list_instruments(self) -> list[InstrumentConfig]:
        return [
            config
            for midi_program, config in self._instruments.items()
            if midi_program in self._loaded_models
        ]
