"""
model_registry.py
다중 모델 지원 레지스트리.

현재는 통합 Qwen2.5 모델 1개만 사용하지만,
향후 모델 추가 시 registry.json + modelType으로 선택할 수 있는 구조를 유지합니다.
"""

import json
import logging
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class LoadedModel:
    """로드된 모델과 관련 리소스를 묶는 컨테이너"""
    name: str
    model_type: str           # "qwen2.5", 향후 추가 가능
    model: Any = None         # 실제 모델 인스턴스
    vocab: dict = field(default_factory=dict)
    vocab_r: dict = field(default_factory=dict)
    device: str = "cuda"


class ModelRegistry:
    """다중 모델 지원 레지스트리.

    현재는 통합 모델 1개만 로드하지만,
    메인 서버가 modelType으로 모델을 선택할 수 있는 구조를 유지합니다.

    향후 모델 추가 시:
    1. registry.json에 새 모델 엔트리 추가
    2. load_all_models()에서 자동 로드
    3. 메인 서버가 request.modelType으로 선택
    """

    def __init__(self, model_dir: Path):
        self._model_dir = model_dir
        self._models: dict[str, LoadedModel] = {}  # key = model_type
        self._default_model_type: str | None = None
        self._registry_config: dict = {}

    def _load_registry_config(self):
        registry_path = self._model_dir / "registry.json"
        if not registry_path.exists():
            logger.warning(f"registry.json not found in {self._model_dir}")
            return
        with open(registry_path) as f:
            self._registry_config = json.load(f)

    def load_all_models(self):
        """앱 시작 시 registry.json의 모든 모델을 로드"""
        self._load_registry_config()

        # 기본 모델 설정: registry.json의 "default" 필드 (model_type 기준)
        self._default_model_type = self._registry_config.get("default")

        models = self._registry_config.get("models", [])

        for idx, model_cfg in enumerate(models):
            # 활성화 여부 확인 (디폴트는 true)
            if not model_cfg.get("active", True):
                continue
                
            model_type = model_cfg["type"]        # "qwen2.5"
            ckpt_path  = self._model_dir / model_cfg["path"]  # "best"
            name       = model_cfg.get("name", model_type)

            try:
                loaded = self._load_single_model(model_type, ckpt_path, name)
                self._models[model_type] = loaded
                logger.info(f"모델 로드 성공: {name} ({model_type})")
            except Exception as e:
                logger.error(f"모델 로드 실패: {name} - {e}", exc_info=True)

        # default가 설정 안 됐으면 첫 번째 모델을 기본으로
        if not self._default_model_type and self._models:
            self._default_model_type = next(iter(self._models))
            logger.info(f"기본 모델 자동 설정: {self._default_model_type}")

        logger.info(
            f"전체 모델 로드: {len(self._models)}개, "
            f"기본 모델: {self._default_model_type}"
        )

    def _load_single_model(
        self, model_type: str, ckpt_path: Path, name: str
    ) -> LoadedModel:
        if model_type == "qwen2.5":
            from app.services.inference import build_v5_vocab, load_model
            import torch

            device = f"cuda" if torch.cuda.is_available() else "cpu"
            vocab = build_v5_vocab()
            vocab_r = {v: k for k, v in vocab.items()}
            model = load_model(str(ckpt_path), len(vocab), vocab, device)

            return LoadedModel(
                name=name,
                model_type=model_type,
                model=model,
                vocab=vocab,
                vocab_r=vocab_r,
                device=device,
            )
        else:
            raise ValueError(f"지원하지 않는 model_type: {model_type}")

    def get_model(self, model_type: str = None) -> LoadedModel:
        """model_type으로 모델 선택. None이면 기본 모델 반환.
        등록되지 않은 model_type이면 기본 모델로 폴백."""
        key = model_type or self._default_model_type
        if key not in self._models:
            if self._default_model_type and self._default_model_type in self._models:
                logger.warning(
                    f"모델 '{key}' 없음, 기본 모델 '{self._default_model_type}'로 폴백"
                )
                key = self._default_model_type
            else:
                raise ValueError(
                    f"모델을 찾을 수 없음: {key}. "
                    f"사용 가능: {list(self._models.keys())}"
                )
        return self._models[key]

    def list_models(self) -> list[str]:
        """로드된 모델 타입 목록 반환"""
        return list(self._models.keys())
