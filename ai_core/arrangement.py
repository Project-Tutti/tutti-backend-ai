"""AI Core — 편곡 오케스트레이션 모듈.

resolve_target()과 run_arrangement()을 제공합니다.
이 두 함수의 시그니처는 contracts/interfaces.py에서 정의된 계약을 준수해야 합니다.
"""

import os
import random
import logging

import torch

from ai_core.constants import INSTRUMENT_GROUPS, PROGRAM_TO_REP, _REP_TO_GROUP
from ai_core.tokenizer import midi_to_bar_tokens
from ai_core.generator import generate_for_target
from ai_core.postprocess import postprocess
from ai_core.midi_writer import save_midi

logger = logging.getLogger(__name__)


def resolve_target(instrument_id: int) -> str:
    """MIDI program 번호(0~128) → INSTRUMENT_GROUPS 키.

    Args:
        instrument_id: MIDI program 번호 (0~128)

    Returns:
        INSTRUMENT_GROUPS 딕셔너리의 키 (예: "violin", "woodwind")

    Raises:
        ValueError: 지원하지 않는 악기 ID
    """
    rep = PROGRAM_TO_REP.get(instrument_id, instrument_id)
    group_name = _REP_TO_GROUP.get(rep)
    if group_name is not None:
        return group_name
    raise ValueError(f"지원하지 않는 악기 ID: {instrument_id}")


def run_arrangement(
    song_path:     str,
    target:        str,
    genre:         str,
    temperature:   float,
    pitch_min,     pitch_max,
    output_path:   str,
    model, vocab, vocab_r, device,
    progress_hook=None,
    original_song_path: str = None,
    actual_instrument_name: str = None,
    actual_midi_program: int = None,
) -> str:
    """편곡 추론 실행. 결과 MIDI 경로 반환."""
    # 고정 하이퍼파라미터
    window_bars    = 8
    context_bars   = 8
    top_p          = 0.95
    rest_penalty   = 1.5
    fade_bars      = 8
    seed           = 42

    # 입력 검증
    if target not in INSTRUMENT_GROUPS:
        raise ValueError(f"지원하지 않는 target: '{target}'. "
                         f"가능한 값: {list(INSTRUMENT_GROUPS.keys())}")
    if not os.path.exists(song_path):
        raise FileNotFoundError(f"입력 파일 없음: {song_path}")

    _device = torch.device(device) if isinstance(device, str) else device

    random.seed(seed)
    torch.manual_seed(seed)

    cfg         = INSTRUMENT_GROUPS[target]
    target_prog = cfg["representative"]
    pitch_min   = pitch_min if pitch_min is not None else cfg["pitch_min"]
    pitch_max   = pitch_max if pitch_max is not None else cfg["pitch_max"]

    logger.info(f"입력 MIDI 토크나이징: {song_path}")
    header, bar_tokens, max_bar, source_pm = midi_to_bar_tokens(song_path, genre, vocab)
    logger.info(f"총 마디 수: {max_bar + 1}")

    logger.info(f"생성 시작 (target={target}, genre={genre}, temp={temperature})")
    all_notes = generate_for_target(
        model, header, bar_tokens, max_bar,
        target_prog, pitch_min, pitch_max,
        window_bars, context_bars,
        temperature, top_p,
        vocab, vocab_r, source_pm, _device,
        rest_penalty=rest_penalty,
        fade_bars=fade_bars,
        progress_hook=progress_hook,
    )

    logger.info(f"디코딩 노트: {len(all_notes)}")
    all_notes = postprocess(all_notes, pitch_min, pitch_max, target_name=target)
    logger.info(f"후처리 후: {len(all_notes)}")

    if len(all_notes) == 0:
        raise RuntimeError("No notes generated.")

    save_midi(all_notes, source_pm, output_path, target_prog, target,
              original_song_path,
              actual_instrument_name=actual_instrument_name,
              actual_midi_program=actual_midi_program)

    return output_path
