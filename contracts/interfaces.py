"""contracts/interfaces.py — AI ↔ 인프라 인터페이스 계약.

이 파일은 양 팀이 함께 관리합니다 (CODEOWNERS: @eddy81848, @sonicwarp).
시그니처를 변경하면 CI의 mypy/pyright가 실패합니다.
"""

from typing import Protocol, Optional, Callable, Dict, Any


class ArrangementRunner(Protocol):
    """run_arrangement()의 시그니처 계약.

    ai_core/arrangement.py의 run_arrangement()은 이 프로토콜을 준수해야 합니다.
    """
    def __call__(
        self,
        song_path: str,
        target: str,
        genre: str,
        temperature: float,
        pitch_min: Optional[int],
        pitch_max: Optional[int],
        output_path: str,
        model: Any,
        vocab: Dict[str, int],
        vocab_r: Dict[int, str],
        device: Any,
        progress_hook: Optional[Callable[[int], None]] = None,
        original_song_path: Optional[str] = None,
        actual_instrument_name: Optional[str] = None,
        actual_midi_program: Optional[int] = None,
    ) -> str: ...


class TargetResolver(Protocol):
    """resolve_target()의 시그니처 계약."""
    def __call__(self, instrument_id: int) -> str: ...


class ModelLoader(Protocol):
    """load_model()의 시그니처 계약."""
    def __call__(
        self,
        ckpt_path: str,
        vocab_size: int,
        vocab: Dict[str, int],
        device: str,
    ) -> Any: ...


class VocabBuilder(Protocol):
    """build_v5_vocab()의 시그니처 계약."""
    def __call__(self, actual_vocab_size: int = 682) -> Dict[str, int]: ...
