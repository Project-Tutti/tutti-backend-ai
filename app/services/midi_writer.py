"""MIDI 저장 모듈 — re-export facade.

실제 구현은 ai_core/midi_writer.py에 있습니다.
기존 import 경로 호환을 위해 유지합니다.
"""

from ai_core.midi_writer import save_midi

__all__ = ["save_midi"]
