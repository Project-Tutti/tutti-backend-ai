"""ai_core/metrics.py — 생성 MIDI 품질 메트릭.

Stage 1: 기본 통계 (mido, 1~2ms) — note_count, pitch_range, density 등
Stage 2: 음악적 평가 (pretty_midi + numpy, ~50ms) — chord_accuracy, pch_similarity, doa, dissonance_rate

논문 기반 지표:
  1. Chord Accuracy       — 코드 구성음 일치율 (NeurIPS 2024 Structured Arrangement)
  2. PCH Similarity       — 조성 분포 코사인 유사도 (AccoMontage)
  3. DOA                  — 트랙 간 음고 다양성/창의성 (NeurIPS 2024)
  4. Dissonance Rate      — 동시 발음 불협화 비율 (AccoMontage2)
"""

import logging
from pathlib import Path

import mido

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# 협화음 테이블 (반음 간격 → 불협화도 0~1)
# 참고: AccoMontage2 dissonance 정의
# ══════════════════════════════════════════════════════════════
DISSONANCE_TABLE = {
    0:  0.0,   # 동음 (완전협화)
    1:  1.0,   # 단2도 (불협화)
    2:  0.5,   # 장2도 (약협화)
    3:  0.2,   # 단3도 (협화)
    4:  0.2,   # 장3도 (협화)
    5:  0.1,   # 완전4도 (협화)
    6:  1.0,   # 삼전음 (강불협화)
    7:  0.0,   # 완전5도 (완전협화)
    8:  0.2,   # 단6도 (협화)
    9:  0.2,   # 장6도 (협화)
    10: 0.5,   # 단7도 (약불협화)
    11: 0.8,   # 장7도 (불협화)
}

# 코드 구성음 템플릿 (루트 기준 반음 간격)
CHORD_TEMPLATES = {
    "maj":  [0, 4, 7],
    "min":  [0, 3, 7],
    "dom7": [0, 4, 7, 10],
    "maj7": [0, 4, 7, 11],
    "min7": [0, 3, 7, 10],
    "dim":  [0, 3, 6],
    "aug":  [0, 4, 8],
}


# ══════════════════════════════════════════════════════════════
# Stage 1: 기본 품질 통계 (mido 기반, 1~2ms)
# ══════════════════════════════════════════════════════════════
def compute_basic_quality_metrics(midi_path: str) -> dict:
    """생성된 MIDI 파일의 기본 품질 통계를 반환합니다.

    Args:
        midi_path: 결과 MIDI 파일 경로.

    Returns:
        dict with keys:
            note_count, pitch_min, pitch_max, pitch_range,
            pitch_mean, pitch_std, avg_velocity,
            avg_duration_sec, total_duration_sec, density_per_sec

        파일을 읽을 수 없거나 노트가 없으면 빈 dict 반환.
    """
    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        logger.warning(f"품질 메트릭 계산 실패 (MIDI 읽기 오류): {e}")
        return {}

    notes = []  # (pitch, velocity, start_sec, end_sec)
    for track in mid.tracks:
        abs_time = 0
        pending = {}

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


# ══════════════════════════════════════════════════════════════
# Stage 2: 음악적 평가 (pretty_midi + numpy 기반, ~50ms)
# ══════════════════════════════════════════════════════════════
def compute_musical_quality(source_path: str, generated_path: str,
                            target_program: int | None = None,
                            resolution: float = 0.125) -> dict:
    """원본과 생성 MIDI를 비교하여 음악적 품질 지표를 반환합니다.

    논문 기반 4개 지표:
      - chord_accuracy:  코드 구성음 일치율 (↑ 좋음, 0~1)
      - pch_similarity:  조성 분포 유사도  (↑ 좋음, 0~1)
      - doa:             편곡 창의성       (↑ 다양, 0~1)
      - dissonance_rate: 불협화 비율       (↓ 좋음, 0~1)

    Args:
        source_path: 원본 소스 MIDI 경로.
        generated_path: 생성된 MIDI 경로.
        target_program: 타겟 악기 프로그램 번호 (None이면 자동 감지).
        resolution: 분석 시간 해상도 (초). 기본 0.125초 (8분음표).

    Returns:
        dict with 4 metrics, or empty dict on failure.
    """
    try:
        import numpy as np
        import pretty_midi
    except ImportError as e:
        logger.warning(f"음악적 평가 건너뜀 (의존성 없음): {e}")
        return {}

    try:
        source_pm = pretty_midi.PrettyMIDI(source_path)
        generated_pm = pretty_midi.PrettyMIDI(generated_path)
    except Exception as e:
        logger.warning(f"음악적 평가 실패 (MIDI 읽기 오류): {e}")
        return {}

    # 타겟 프로그램 자동 감지
    if target_program is not None:
        target_programs = {target_program}
    else:
        source_progs = {inst.program for inst in source_pm.instruments if not inst.is_drum}
        gen_progs = {inst.program for inst in generated_pm.instruments if not inst.is_drum}
        target_programs = gen_progs - source_progs
        if not target_programs:
            target_programs = gen_progs

    chord_acc = _chord_accuracy(source_pm, generated_pm, target_programs, resolution, np)
    pch_sim = _pch_similarity(source_pm, generated_pm, target_programs, np)
    doa = round(1.0 - pch_sim, 4)
    dissonance = _dissonance_rate(source_pm, generated_pm, target_programs, resolution, np)

    return {
        "chord_accuracy": round(chord_acc, 4),
        "pch_similarity": round(pch_sim, 4),
        "doa": doa,
        "dissonance_rate": round(dissonance, 4),
    }


# ──────────────────────────────────────────────
# 내부 함수들
# ──────────────────────────────────────────────
def _get_tempo(mid: mido.MidiFile) -> int:
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                return msg.tempo
    return 500000


def _get_active_pitches_at(pm, time: float, exclude_programs=None) -> set:
    """특정 시간에 울리는 음들의 pitch class 집합."""
    pitches = set()
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        if exclude_programs and inst.program in exclude_programs:
            continue
        for note in inst.notes:
            if note.start <= time < note.end:
                pitches.add(note.pitch % 12)
    return pitches


def _get_pch(pm, programs=None, np=None) -> "np.ndarray":
    """Pitch Class Histogram (12차원 벡터)."""
    pch = np.zeros(12)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        if programs is not None and inst.program not in programs:
            continue
        for note in inst.notes:
            duration = note.end - note.start
            pch[note.pitch % 12] += duration
    total = pch.sum()
    if total > 0:
        pch /= total
    return pch


def _detect_chord(pitches: set) -> tuple | None:
    """현재 울리는 음들로 가장 유사한 코드 추정."""
    if len(pitches) < 2:
        return None
    best_match = None
    best_score = -1
    for root in range(12):
        for chord_type, intervals in CHORD_TEMPLATES.items():
            chord_tones = {(root + i) % 12 for i in intervals}
            overlap = len(pitches & chord_tones)
            score = overlap / max(len(chord_tones), len(pitches))
            if score > best_score:
                best_score = score
                best_match = (root, chord_type, chord_tones)
    return best_match if best_score > 0.5 else None


def _chord_accuracy(source_pm, generated_pm, target_programs, resolution, np) -> float:
    """코드 구성음 일치율."""
    end_time = max(source_pm.get_end_time(), generated_pm.get_end_time())
    times = np.arange(0, end_time, resolution)
    total = 0
    in_chord = 0

    for t in times:
        source_pitches = _get_active_pitches_at(source_pm, t, exclude_programs=target_programs)
        chord_info = _detect_chord(source_pitches)
        if chord_info is None:
            continue
        _, _, chord_tones = chord_info

        gen_pitches = set()
        for inst in generated_pm.instruments:
            if inst.is_drum:
                continue
            if inst.program in target_programs:
                for note in inst.notes:
                    if note.start <= t < note.end:
                        gen_pitches.add(note.pitch % 12)
        if not gen_pitches:
            continue

        for p in gen_pitches:
            total += 1
            if p in chord_tones:
                in_chord += 1

    return in_chord / total if total > 0 else 0.0


def _pch_similarity(source_pm, generated_pm, target_programs, np) -> float:
    """Pitch Class Histogram 코사인 유사도."""
    source_pch = _get_pch(source_pm, np=np)
    gen_pch = _get_pch(generated_pm, programs=target_programs, np=np)

    dot = np.dot(source_pch, gen_pch)
    norm_s = np.linalg.norm(source_pch)
    norm_g = np.linalg.norm(gen_pch)

    if norm_s == 0 or norm_g == 0:
        return 0.0
    return float(dot / (norm_s * norm_g))


def _dissonance_rate(source_pm, generated_pm, target_programs, resolution, np) -> float:
    """소스-생성 파트 간 음정 불협화도 평균."""
    end_time = max(source_pm.get_end_time(), generated_pm.get_end_time())
    times = np.arange(0, end_time, resolution)
    total_dissonance = 0.0
    count = 0

    for t in times:
        source_pitches = _get_active_pitches_at(source_pm, t, exclude_programs=target_programs)
        gen_pitches = set()
        for inst in generated_pm.instruments:
            if inst.is_drum:
                continue
            if inst.program in target_programs:
                for note in inst.notes:
                    if note.start <= t < note.end:
                        gen_pitches.add(note.pitch % 12)

        if not source_pitches or not gen_pitches:
            continue

        pair_dissonances = []
        for sp in source_pitches:
            for gp in gen_pitches:
                interval = abs(sp - gp) % 12
                pair_dissonances.append(DISSONANCE_TABLE[interval])

        total_dissonance += np.mean(pair_dissonances)
        count += 1

    return total_dissonance / count if count > 0 else 0.0
