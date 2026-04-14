"""ai_core/metrics.py — 생성 MIDI 품질 메트릭.

인퍼런스 후 결과 MIDI 파일의 기본 통계를 계산합니다.
mido로 노트 이벤트만 순회하므로 1~2ms 이내 완료됩니다.
"""

import logging
from pathlib import Path

import mido

logger = logging.getLogger(__name__)


def compute_basic_quality_metrics(midi_path: str) -> dict:
    """생성된 MIDI 파일의 기본 품질 통계를 반환합니다.

    Args:
        midi_path: 결과 MIDI 파일 경로.

    Returns:
        dict with keys:
            note_count        — 생성된 총 노트 수
            pitch_min         — 최저 음정 (MIDI number)
            pitch_max         — 최고 음정 (MIDI number)
            pitch_range       — 최고음 - 최저음
            pitch_mean        — 평균 음정
            pitch_std         — 음정 표준편차
            avg_velocity      — 평균 벨로시티
            avg_duration_sec  — 평균 음표 길이 (초)
            total_duration_sec— 전체 MIDI 길이 (초)
            density_per_sec   — 초당 평균 노트 수

        파일을 읽을 수 없거나 노트가 없으면 빈 dict 반환.
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        logger.warning(f"품질 메트릭 계산 실패 (MIDI 읽기 오류): {e}")
        return {}

    # 모든 트랙에서 note_on 이벤트 수집
    notes = []  # (pitch, velocity, start_sec, end_sec)
    for track in mid.tracks:
        abs_time = 0  # tick 단위 누적 시간
        pending = {}  # pitch → (velocity, start_tick)

        for msg in track:
            abs_time += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                pending[msg.note] = (msg.velocity, abs_time)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in pending:
                    vel, start_tick = pending.pop(msg.note)
                    start_sec = mido.tick2second(start_tick, mid.ticks_per_beat, _get_tempo(mid))
                    end_sec = mido.tick2second(abs_time, mid.ticks_per_beat, _get_tempo(mid))
                    notes.append((msg.note, vel, start_sec, end_sec))

    if not notes:
        logger.warning(f"품질 메트릭: 노트 0개 — {midi_path}")
        return {"note_count": 0}

    pitches = [n[0] for n in notes]
    velocities = [n[1] for n in notes]
    durations = [n[3] - n[2] for n in notes]

    pitch_mean = sum(pitches) / len(pitches)
    pitch_std = (sum((p - pitch_mean) ** 2 for p in pitches) / len(pitches)) ** 0.5

    total_duration = max(n[3] for n in notes)

    return {
        "note_count": len(notes),
        "pitch_min": min(pitches),
        "pitch_max": max(pitches),
        "pitch_range": max(pitches) - min(pitches),
        "pitch_mean": round(pitch_mean, 1),
        "pitch_std": round(pitch_std, 1),
        "avg_velocity": round(sum(velocities) / len(velocities), 1),
        "avg_duration_sec": round(sum(durations) / len(durations), 3),
        "total_duration_sec": round(total_duration, 1),
        "density_per_sec": round(len(notes) / max(total_duration, 0.1), 2),
    }


def _get_tempo(mid: mido.MidiFile) -> int:
    """MIDI 파일에서 첫 번째 set_tempo 메시지의 tempo를 반환합니다.

    없으면 기본값 500000 (120 BPM).
    """
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return msg.tempo
    return 500000
