"""
inference.py — Facade (하위 호환성 유지용)

⚠️  이 파일은 ai_core/ 패키지로 분리된 후에도 기존 import 경로를 유지하기 위한
     얇은 re-export 레이어입니다. 새로운 코드에서는 ai_core를 직접 import 하세요.

기존 호출자 (worker.py, model_registry.py):
    from app.services.inference import run_arrangement, resolve_target
    from app.services.inference import load_model, build_v5_vocab
→ 모두 정상 동작합니다.
"""

# 공개 API — ai_core에서 re-export
from ai_core.arrangement import run_arrangement, resolve_target
from ai_core.vocab import build_v5_vocab
from ai_core.model_loader import load_model

# 상수 — worker.py / model_registry.py에서 참조할 수 있으므로 re-export
from ai_core.constants import (
    INSTRUMENT_GROUPS,
    ALL_TARGET_NAMES,
    PROGRAM_TO_REP,
    DROP_SET,
    FLAT_TO_SHARP,
    MONOPHONIC_INSTRUMENTS,
)

# save_midi — midi_writer에서 re-export
from app.services.midi_writer import save_midi

__all__ = [
    "run_arrangement",
    "resolve_target",
    "build_v5_vocab",
    "load_model",
    "save_midi",
    "INSTRUMENT_GROUPS",
    "ALL_TARGET_NAMES",
    "PROGRAM_TO_REP",
    "DROP_SET",
    "FLAT_TO_SHARP",
    "MONOPHONIC_INSTRUMENTS",
]