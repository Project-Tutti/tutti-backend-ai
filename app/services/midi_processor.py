import httpx
import tempfile
import logging
import mido
from pathlib import Path
from typing import List, Dict, Any, Tuple

from anticipation.convert import midi_to_events, events_to_midi
from anticipation.tokenize import extract_instruments
from anticipation.config import *
from anticipation.vocab import *
from anticipation import ops

from app.core.config import settings

logger = logging.getLogger(__name__)


async def download_midi(midi_url: str) -> Path:
    """Download MIDI from the main server's Supabase storage."""
    temp_dir = Path("/tmp/tutti_midi_downloads")
    temp_dir.mkdir(parents=True, exist_ok=True)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mid", dir=temp_dir)
    file_path = Path(temp_file.name)
    temp_file.close()

    logger.info(f"Downloading MIDI from {midi_url} to {file_path}")
    async with httpx.AsyncClient(follow_redirects=True, timeout=60) as client:
        try:
            async with client.stream("GET", str(midi_url)) as response:
                response.raise_for_status()
                with open(file_path, "wb") as f:
                    async for chunk in response.aiter_bytes():
                        f.write(chunk)
            logger.info("MIDI download completed.")
            return file_path
        except Exception as e:
            logger.error(f"Failed to download MIDI: {e}")
            if file_path.exists():
                file_path.unlink()
            raise


def get_song_length(midi_path: Path) -> float:
    """MIDI 파일의 실제 음표 기반 곡 길이(초)를 계산합니다."""
    mid = mido.MidiFile(str(midi_path))
    midi_tempo = 500000  # 기본 120 BPM
    for track in mid.tracks:
        for msg in track:
            if msg.type == "set_tempo":
                midi_tempo = msg.tempo
                break

    tpb = mid.ticks_per_beat
    tps = tpb * 1000000.0 / midi_tempo

    last_note_end = 0.0
    for track in mid.tracks:
        abs_time = 0
        active = {}
        for msg in track:
            abs_time += msg.time
            if msg.type == "note_on" and msg.velocity > 0:
                active[msg.note] = abs_time
            elif msg.type == "note_off" or (
                msg.type == "note_on" and msg.velocity == 0
            ):
                if msg.note in active:
                    end_sec = abs_time / tps
                    if end_sec > last_note_end:
                        last_note_end = end_sec
                    del active[msg.note]

    return last_note_end


def parse_midi(midi_path: Path) -> Tuple[List[int], List[int], float]:
    """
    MIDI를 anticipation 이벤트로 변환하고, 추론용 controls를 생성합니다.

    반환:
        Tuple[events, controls, song_length]:
            - events: anticipation 포맷의 전체 이벤트 리스트
            - controls: 모든 악기를 포함한 컨트롤 이벤트 (inference에 전달)
            - song_length: 곡 길이(초)
    """
    logger.info(f"Parsing MIDI: {midi_path}")

    # 1. anticipation의 midi_to_events로 MIDI → 이벤트 시퀀스 변환
    events = midi_to_events(str(midi_path))
    logger.info(f"MIDI parsed: {len(events) // 3} events")

    # 2. 곡 길이 계산 (음표 기반 vs anticipation 기반 중 큰 값)
    amt_length = ops.max_time(events)
    note_length = get_song_length(midi_path)
    song_length = max(note_length, amt_length) if note_length > 0 else amt_length
    logger.info(f"Song length: {song_length:.1f}s")

    # 3. 원곡의 모든 악기를 controls로 추출 (anticipation의 extract_instruments 사용)
    #    controls는 inference 시 모델이 원곡의 맥락을 참조할 수 있게 해줍니다.
    instruments = ops.get_instruments(events)
    all_instrs = list(instruments.keys())
    logger.info(f"Instruments found: {all_instrs} ({len(all_instrs)} instruments)")

    if all_instrs:
        _, controls = extract_instruments(events, all_instrs)
    else:
        controls = []
        logger.warning("No instruments found in MIDI — controls will be empty.")

    return events, controls, song_length


def inject_instrument_track(
    original_midi_path: Path,
    instrument_events: List[int],
    target_instrument_id: int,
    output_path: Path,
) -> bool:
    """
    생성된 악기 이벤트를 원본 MIDI에 새 트랙으로 주입합니다.
    generate_colab.ipynb의 inject_violin_track 로직을 일반화한 버전입니다.

    Args:
        original_midi_path: 원본 MIDI 파일 경로
        instrument_events: anticipation 포맷의 생성된 이벤트 리스트 [time, dur, note, ...]
        target_instrument_id: MIDI program number (악기 ID)
        output_path: 결과 MIDI 저장 경로

    Returns:
        성공 여부
    """
    if len(instrument_events) < 3:
        logger.warning(f"No events to inject for instrument {target_instrument_id}")
        return False

    try:
        mid = mido.MidiFile(str(original_midi_path))
        if mid.type == 0:
            mid.type = 1

        # 원곡 템포 추출
        tempo = 500000  # 기본 120 BPM
        for track in mid.tracks:
            for msg in track:
                if msg.type == "set_tempo":
                    tempo = msg.tempo
                    break

        tpb = mid.ticks_per_beat
        ticks_per_second = tpb * 1000000.0 / tempo
        grid = tpb // 2  # 8분음표 그리드
        min_length = grid

        # 이벤트 → 노트 변환
        raw_notes = []
        for i in range(0, len(instrument_events), 3):
            if i + 2 >= len(instrument_events):
                break
            t_sec = (instrument_events[i] - TIME_OFFSET) / TIME_RESOLUTION
            d_sec = (instrument_events[i + 1] - DUR_OFFSET) / TIME_RESOLUTION
            note_val = instrument_events[i + 2] - NOTE_OFFSET
            if 0 <= note_val < MAX_NOTE:
                pitch = note_val % 128
                start_tick = int(t_sec * ticks_per_second)
                end_tick = int((t_sec + d_sec) * ticks_per_second)
                raw_notes.append((pitch, start_tick, end_tick))

        if not raw_notes:
            logger.warning(f"No valid notes for instrument {target_instrument_id}")
            return False

        # 그리드 양자화 + 모노포닉 정리
        notes = []
        for pitch, start, end in raw_notes:
            s = round(start / grid) * grid
            e = round(end / grid) * grid
            if e <= s:
                e = s + grid
            notes.append((pitch, s, e))

        notes = sorted(notes, key=lambda n: n[1])

        # 겹치는 노트 정리 (모노포닉)
        mono = []
        for pitch, start, end in notes:
            if mono:
                pp, ps, pe = mono[-1]
                if start < pe:
                    pe = start
                    if pe <= ps:
                        mono.pop()
                    else:
                        mono[-1] = (pp, ps, pe)
            mono.append((pitch, start, end))

        mono = [(p, s, e) for p, s, e in mono if e - s >= min_length]

        # 짧은 간격 연결
        sixteenth = tpb // 4
        for i in range(len(mono) - 1):
            p, s, e = mono[i]
            _, ns, _ = mono[i + 1]
            gap = ns - e
            if 0 < gap < sixteenth:
                mono[i] = (p, s, ns)

        logger.info(
            f"Instrument {target_instrument_id}: {len(raw_notes)} raw → {len(mono)} quantized notes"
        )

        # MIDI 트랙 생성 (channel 15 사용, 원곡과 충돌 방지)
        new_track = mido.MidiTrack()
        new_track.append(
            mido.Message(
                "program_change",
                channel=15,
                program=target_instrument_id,
                time=0,
            )
        )

        midi_events = []
        for pitch, start, end in mono:
            midi_events.append((start, "note_on", pitch, 80))
            midi_events.append((end, "note_off", pitch, 0))
        midi_events.sort(key=lambda x: (x[0], x[1] == "note_on"))

        prev_tick = 0
        for tick, msg_type, pitch, vel in midi_events:
            delta = max(0, tick - prev_tick)
            new_track.append(
                mido.Message(msg_type, channel=15, note=pitch, velocity=vel, time=delta)
            )
            prev_tick = tick

        mid.tracks.append(new_track)
        mid.save(str(output_path))
        return True

    except Exception as e:
        logger.error(f"Failed to inject instrument track: {e}")
        return False


def merge_tracks(
    results: List[List[int]],
    original_midi_path: Path,
    target_instrument_ids: List[int],
    output_path: Path,
) -> Path:
    """
    여러 트랙의 추론 결과 이벤트를 원본 MIDI에 합쳐 하나의 최종 MIDI 파일로 생성합니다.

    generate_colab.ipynb의 inject_violin_track 로직을 일반화하여,
    여러 악기의 결과를 순차적으로 원본 MIDI에 주입합니다.

    Args:
        results: 각 트랙의 추론 결과 이벤트 리스트들
        original_midi_path: 원본 MIDI 파일 경로
        target_instrument_ids: 각 결과에 대응하는 악기 ID 리스트
        output_path: 결과 MIDI 저장 경로

    Returns:
        최종 MIDI 파일 경로
    """
    logger.info(f"Merging {len(results)} tracks into {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not results:
        logger.warning("No results to merge, copying original MIDI")
        import shutil

        shutil.copy2(str(original_midi_path), str(output_path))
        return output_path

    # 첫 번째 트랙: 원본 MIDI에 주입
    current_source = original_midi_path
    for i, (events, instrument_id) in enumerate(zip(results, target_instrument_ids)):
        # 마지막 트랙이면 최종 출력 경로, 아니면 임시 경로
        if i == len(results) - 1:
            dest = output_path
        else:
            dest = output_path.parent / f"_temp_merge_{i}.mid"

        success = inject_instrument_track(current_source, events, instrument_id, dest)

        if not success:
            logger.warning(
                f"Failed to inject track {i} (instrument {instrument_id}), skipping"
            )
            # 실패 시 이전 결과를 그대로 사용
            if i == len(results) - 1:
                import shutil

                shutil.copy2(str(current_source), str(dest))

        # 다음 반복에서는 방금 생성한 파일을 원본으로 사용
        if i > 0 and current_source != original_midi_path:
            # 이전 임시 파일 정리
            try:
                current_source.unlink()
            except Exception:
                pass

        current_source = dest

    logger.info(f"Merge completed: {output_path}")
    return output_path
