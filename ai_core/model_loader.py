"""AI Core — 모델 로딩 모듈.

Qwen2.5-0.5B 기반 체크포인트 로드 로직.
"""

import os
import logging

import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)


def load_model(ckpt_path: str, vocab_size: int, vocab: dict, device: str):
    """모델 로드. model_registry.py의 _load_single_model()에서 호출.

    Args:
        ckpt_path:  체크포인트 디렉토리 경로 (model.safetensors 또는 pytorch_model.bin 포함)
        vocab_size: build_v5_vocab()의 len(vocab)
        vocab:      build_v5_vocab() 결과
        device:     "cuda" 또는 "cpu"

    Returns:
        eval 모드로 설정된 모델 (device로 이동 완료)

    Raises:
        FileNotFoundError: 체크포인트 파일 없음
    """
    _device = torch.device(device)

    MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.vocab_size              = vocab_size
    config.pad_token_id            = vocab["PAD"]
    config.max_position_embeddings = 2048
    config.sliding_window          = None

    model = AutoModelForCausalLM.from_config(config)
    model = model.to(torch.bfloat16)
    model.model.embed_tokens = nn.Embedding(vocab_size, config.hidden_size).to(torch.bfloat16)
    model.lm_head            = nn.Linear(config.hidden_size, vocab_size, bias=False).to(torch.bfloat16)

    sf   = os.path.join(ckpt_path, "model.safetensors")
    bin_ = os.path.join(ckpt_path, "pytorch_model.bin")
    if os.path.exists(sf):
        from safetensors.torch import load_file
        state = load_file(sf, device="cpu")
        model.load_state_dict(state, strict=True)
        logger.info(f"체크포인트 로드 (safetensors): {sf}")
    elif os.path.exists(bin_):
        state = torch.load(bin_, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        logger.info(f"체크포인트 로드 (.bin): {bin_}")
    else:
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")

    model.eval()
    model.to(_device)

    # RTX 4090 / T4 TF32 가속
    torch.set_float32_matmul_precision("high")
    logger.info("TF32 하드웨어 가속 활성화 (torch.compile 비활성화)")

    return model
