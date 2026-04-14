"""test_contracts.py — contracts/interfaces.py 계약 준수 테스트.

ai_core의 public API가 contracts에 정의된 Protocol을 준수하는지 확인합니다.
"""
import sys
import os
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestContracts:
    """Protocol 시그니처 호환성 테스트.

    실제 typing.runtime_checkable 대신, 함수 시그니처의 파라미터 이름을
    검사하여 계약 위반을 감지합니다.
    """

    def test_run_arrangement_signature(self):
        import inspect
        from ai_core.arrangement import run_arrangement
        sig = inspect.signature(run_arrangement)
        params = list(sig.parameters.keys())

        required = [
            "song_path", "target", "genre", "temperature",
            "pitch_min", "pitch_max", "output_path",
            "model", "vocab", "vocab_r", "device",
        ]
        for p in required:
            assert p in params, f"run_arrangement missing required param: {p}"

    def test_resolve_target_signature(self):
        import inspect
        from ai_core.arrangement import resolve_target
        sig = inspect.signature(resolve_target)
        params = list(sig.parameters.keys())
        assert "instrument_id" in params

    def test_load_model_signature(self):
        import inspect
        from ai_core.model_loader import load_model
        sig = inspect.signature(load_model)
        params = list(sig.parameters.keys())
        required = ["ckpt_path", "vocab_size", "vocab", "device"]
        for p in required:
            assert p in params, f"load_model missing required param: {p}"

    def test_build_v5_vocab_signature(self):
        import inspect
        from ai_core.vocab import build_v5_vocab
        sig = inspect.signature(build_v5_vocab)
        params = list(sig.parameters.keys())
        assert "actual_vocab_size" in params
