"""
midi_processor.py
MIDI 파일 다운로드 및 트랙 재매핑 모듈.

anticipation 라이브러리 의존성을 제거하고,
mido 기반으로 Type 0/Type 1 MIDI 파일의 트랙 재매핑을 처리합니다.
"""

import httpx
import tempfile
import logging
import mido
from pathlib import Path
from typing import List

from app.core.config import settings

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
# MIDI 다운로드
# ──────────────────────────────────────────────
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


# ──────────────────────────────────────────────
# 트랙 재매핑 (Type 0 / Type 1 지원)
# ──────────────────────────────────────────────
DROP_INSTRUMENT_ID = 129  # 트랙/채널 삭제 시그널


def remap_original_tracks(midi_path: Path, mappings: list) -> None:
    """
    원본 MIDI의 트랙 악기를 mappings에 따라 재매핑합니다.

    - targetInstrumentId == 129: 해당 트랙/채널의 이벤트 삭제
    - 그 외: 해당 트랙/채널의 program_change를 targetInstrumentId로 변경

    Type 0 (단일 트랙, 채널로 구분)과 Type 1 (멀티 트랙) 모두 지원.
    결과는 원본 파일에 덮어씁니다.

    Args:
        midi_path: MIDI 파일 경로
        mappings: Mapping 객체 리스트 (trackIndex, targetInstrumentId)
    """
    if not mappings:
        logger.info("재매핑할 mappings가 비어있어 건너뜁니다.")
        return

    mid = mido.MidiFile(str(midi_path))

    if mid.type == 0:
        _remap_type0(mid, mappings)
    else:
        _remap_type1(mid, mappings)

    mid.save(str(midi_path))
    logger.info(f"트랙 재매핑 완료: {midi_path}")


def _remap_type1(mid: mido.MidiFile, mappings: list) -> None:
    """Type 1: trackIndex로 트랙을 찾아 처리"""
    tracks_to_delete = set()

    # 프론트엔드의 @tonejs/midi 파싱 로직과 동기화
    # (노트가 있는 트랙만 필터링한 인덱스 사용)
    valid_track_indices = []
    for i, t in enumerate(mid.tracks):
        if any(msg.type in ('note_on', 'note_off') for msg in t):
            valid_track_indices.append(i)

    for mapping in mappings:
        frontend_idx = mapping.trackIndex
        if frontend_idx >= len(valid_track_indices):
            logger.warning(
                f"trackIndex {frontend_idx} out of range "
                f"(총 {len(valid_track_indices)} 유효 노트 트랙), 건너뜁니다."
            )
            continue

        idx = valid_track_indices[frontend_idx]

        if mapping.targetInstrumentId == DROP_INSTRUMENT_ID:
            tracks_to_delete.add(idx)
        else:
            is_drum = mapping.targetInstrumentId >= 128
            target_prog = 0 if is_drum else mapping.targetInstrumentId

            # program_change 존재 여부 확인 후 변경 또는 삽입
            has_program_change = False
            for msg in mid.tracks[idx]:
                if hasattr(msg, 'channel'):
                    if is_drum:
                        msg.channel = 9
                    elif msg.channel == 9:
                        msg.channel = 0
                if msg.type == "program_change":
                    msg.program = target_prog
                    has_program_change = True

            # program_change가 없으면 트랙 맨 앞에 삽입
            if not has_program_change:
                # 첫 번째 채널 메시지에서 channel 추출, 없으면 0
                track_channel = 9 if is_drum else 0
                for msg in mid.tracks[idx]:
                    if hasattr(msg, 'channel'):
                        track_channel = msg.channel
                        break
                pc = mido.Message(
                    'program_change',
                    channel=track_channel,
                    program=target_prog,
                    time=0,
                )
                mid.tracks[idx].insert(0, pc)
                logger.info(
                    f"트랙 {idx}에 program_change 삽입: "
                    f"ch={track_channel}, prog={target_prog}"
                )

    # 역순 처리 (인덱스 밀림 방지 용도였으나 필터링 방식으로 변경되어 역순 유지는 안전을 위해 그대로 둠)
    for idx in sorted(tracks_to_delete, reverse=True):
        # 메타 전용 트랙 보호: set_tempo, time_signature 등 글로벌 정보 소실 방지
        is_meta_only = all(msg.is_meta for msg in mid.tracks[idx])
        if is_meta_only:
            logger.warning(
                f"trackIndex {idx}는 메타 전용 트랙(tempo/time_signature 등)이므로 "
                f"삭제 처리에서 제외합니다."
            )
            continue
            
        logger.info(f"트랙 {idx} 채널 이벤트 안전 삭제 (메타/SysEx 보존)")
        new_track = []
        accumulated_time = 0
        for msg in mid.tracks[idx]:
            # 메타 및 SysEx 보존, 채널 이벤트(음표 등)만 필터링하여 시간 이월
            if hasattr(msg, 'channel'):
                accumulated_time += msg.time
            else:
                msg = msg.copy(time=msg.time + accumulated_time)
                accumulated_time = 0
                new_track.append(msg)
                
        # 트랙 객체를 통째로 파괴하지 않고 알맹이만 비워진 상태로 갈아끼움
        mid.tracks[idx] = new_track


def _remap_type0(mid: mido.MidiFile, mappings: list) -> None:
    """Type 0: channel로 이벤트를 찾아 처리

    Type 0에서는 모든 이벤트가 단일 트랙(tracks[0])에 있고,
    채널(channel)로 악기를 구분합니다.
    따라서 노트가 있는 채널 목록을 정렬하여 trackIndex와 매핑합니다.
    """
    track = mid.tracks[0]

    channels_with_notes = set()
    for msg in track:
        if msg.type in ('note_on', 'note_off') and hasattr(msg, 'channel'):
            channels_with_notes.add(msg.channel)
    valid_channels = sorted(list(channels_with_notes))

    channels_to_delete = set()
    channel_remap = {}

    for mapping in mappings:
        frontend_idx = mapping.trackIndex
        if frontend_idx >= len(valid_channels):
            continue
        
        ch = valid_channels[frontend_idx]
        if mapping.targetInstrumentId == DROP_INSTRUMENT_ID:
            channels_to_delete.add(ch)
        else:
            channel_remap[ch] = mapping.targetInstrumentId

    # 1. 채널 삭제 먼저 수행 (이후 channel 변경 시 충돌 방지)
    # 주의: meta 메시지(tempo, time_signature 등)는 channel이 없으므로 유지됨
    # 중요: mido는 delta time 기반이므로, 삭제된 메시지의 time을
    #        다음 생존 메시지에 누적해야 타이밍이 보존됨
    if channels_to_delete:
        new_track = []
        accumulated_time = 0
        for msg in track:
            should_delete = hasattr(msg, 'channel') and msg.channel in channels_to_delete
            if should_delete:
                accumulated_time += msg.time
            else:
                msg = msg.copy(time=msg.time + accumulated_time)
                accumulated_time = 0
                new_track.append(msg)
        mid.tracks[0] = new_track
        track = mid.tracks[0]
        logger.info(f"Type 0 채널 삭제: {channels_to_delete}")

    # 2. program_change 및 channel 변경
    channels_with_pc = set()
    for msg in track:
        if hasattr(msg, 'channel') and msg.channel in channel_remap:
            orig_ch = msg.channel
            target_prog = channel_remap[orig_ch]
            is_drum = target_prog >= 128
            safe_prog = 0 if is_drum else target_prog

            if is_drum:
                msg.channel = 9
            elif msg.channel == 9:
                msg.channel = 0

            if msg.type == "program_change":
                msg.program = safe_prog
                channels_with_pc.add(orig_ch)

    # 3. program_change가 없는 채널에 트랙 맨 앞 삽입
    for ch, target_prog in channel_remap.items():
        if ch not in channels_with_pc:
            is_drum = target_prog >= 128
            safe_prog = 0 if is_drum else target_prog
            new_ch = 9 if is_drum else (0 if ch == 9 else ch)
            
            pc = mido.Message(
                'program_change',
                channel=new_ch,
                program=safe_prog,
                time=0,
            )
            track.insert(0, pc)
            logger.info(
                f"Type 0 채널 {ch}에 program_change 삽입: prog={safe_prog} (ch={new_ch})"
            )
