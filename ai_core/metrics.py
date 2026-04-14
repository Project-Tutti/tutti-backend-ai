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
    default_metrics = {
        "note_count": 0,
        "pitch_min": 0,
        "pitch_max": 0,
        "pitch_range": 0,
        "pitch_mean": 0.0,
        "pitch_std": 0.0,
        "avg_velocity": 0.0,
        "avg_duration_sec": 0.0,
        "total_duration_sec": 0.0,
        "density_per_sec": 0.0,
    }

    try:
        mid = mido.MidiFile(midi_path)
    except Exception as e:
        logger.warning(f"품질 메트릭 계산 실패 (MIDI 읽기 오류): {e}")
        return default_metrics

    # 전체 트랙에서 템포 이벤트를 시간순으로 수집 (동적 템포 대응)
    tempo_map = _build_tempo_map(mid)

    notes = []  # (pitch, velocity, start_sec, end_sec)
    for track in mid.tracks:
        abs_tick = 0
        pending = {}

        for msg in track:
            abs_tick += msg.time

            if msg.type == "note_on" and msg.velocity > 0:
                pending[msg.note] = (msg.velocity, abs_tick)
            elif msg.type == "note_off" or (msg.type == "note_on" and msg.velocity == 0):
                if msg.note in pending:
                    vel, start_tick = pending.pop(msg.note)
                    start_sec = _tick_to_second(start_tick, mid.ticks_per_beat, tempo_map)
                    end_sec = _tick_to_second(abs_tick, mid.ticks_per_beat, tempo_map)
                    notes.append((msg.note, vel, start_sec, end_sec))

    if not notes:
        logger.warning(f"품질 메트릭: 노트 0개 — {midi_path}")
        return default_metrics

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
    default_musical_metrics = {
        "chord_accuracy": 0.0,
        "pch_similarity": 0.0,
        "doa": 0.0,
        "dissonance_rate": 0.0,
    }

    try:
        import numpy as np
        import pretty_midi
    except ImportError as e:
        logger.warning(f"음악적 평가 건너뜀 (의존성 없음): {e}")
        return default_musical_metrics

    try:
        source_pm = pretty_midi.PrettyMIDI(source_path)
        generated_pm = pretty_midi.PrettyMIDI(generated_path)
    except Exception as e:
        logger.warning(f"음악적 평가 실패 (MIDI 읽기 오류): {e}")
        return default_musical_metrics

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
    doa = _degree_of_arrangement(generated_pm, target_programs, np)
    dissonance = _dissonance_rate(source_pm, generated_pm, target_programs, resolution, np)

    return {
        "chord_accuracy": round(chord_acc, 4),
        "pch_similarity": round(pch_sim, 4),
        "doa": round(doa, 4),
        "dissonance_rate": round(dissonance, 4),
    }


# ──────────────────────────────────────────────
# 내부 함수들
# ──────────────────────────────────────────────
def _build_tempo_map(mid: mido.MidiFile) -> list:
    """MIDI 파일에서 (tick, tempo) 리스트를 시간순으로 구축.

    템포 변화가 있는 곡에서 정확한 tick→second 변환을 위해
    모든 set_tempo 이벤트를 추적합니다.
    """
    tempo_events = []
    for track in mid.tracks:
        abs_tick = 0
        for msg in track:
            abs_tick += msg.time
            if msg.type == "set_tempo":
                tempo_events.append((abs_tick, msg.tempo))

    # tick 기준 정렬
    tempo_events.sort(key=lambda x: x[0])

    if not tempo_events:
        return [(0, 500000)]  # 기본 120 BPM

    # tick 0에 이벤트가 없으면 첫 번째 템포를 tick 0부터 적용
    if tempo_events[0][0] != 0:
        tempo_events.insert(0, (0, tempo_events[0][1]))

    return tempo_events


def _tick_to_second(tick: int, ticks_per_beat: int, tempo_map: list) -> float:
    """템포 변화를 고려하여 tick을 초로 변환합니다.

    각 템포 구간별로 경과 시간을 누적하여 정확한 시간을 계산합니다.
    단일 템포만 사용하던 이전 방식의 타이밍 왜곡 문제를 해결합니다.
    """
    elapsed_sec = 0.0
    prev_tick = 0
    prev_tempo = tempo_map[0][1]

    for i in range(1, len(tempo_map)):
        t_tick, t_tempo = tempo_map[i]
        if t_tick >= tick:
            break
        # 이전 구간의 시간 누적
        delta_ticks = t_tick - prev_tick
        elapsed_sec += mido.tick2second(delta_ticks, ticks_per_beat, prev_tempo)
        prev_tick = t_tick
        prev_tempo = t_tempo

    # 남은 구간
    remaining_ticks = tick - prev_tick
    elapsed_sec += mido.tick2second(remaining_ticks, ticks_per_beat, prev_tempo)
    return elapsed_sec


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


def _degree_of_arrangement(generated_pm, target_programs, np) -> float:
    """편곡 창의성 — 생성 파트의 음정(pitch class) 다양성.

    PCH의 엔트로피(정보량)를 정규화하여 0~1 범위로 환산합니다.
    값이 높을수록 다양한 음정을 고르게 사용했다는 뜻입니다.
    기존 '1 - pch_similarity'는 PCH와 완전 역상관이라 독립 정보가 없었으므로,
    생성 파트 자체의 분포 엔트로피로 대체합니다.
    """
    gen_pch = _get_pch(generated_pm, programs=target_programs, np=np)
    # 이미 정규화된 분포이므로 그대로 엔트로피 계산
    # Shannon entropy, log2(12) ≈ 3.585 로 정규화 → 0~1 범위
    eps = 1e-12
    entropy = -np.sum(gen_pch * np.log2(gen_pch + eps))
    max_entropy = np.log2(12)  # 12개 pitch class 균등 분포의 최대 엔트로피
    return float(np.clip(entropy / max_entropy, 0.0, 1.0))


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
