"""MIDI 저장 모듈 — 원본 보존 기반 Mido Append 전략.

원본 MIDI 파일에 AI 생성 노트를 별도 트랙으로 덧붙여 저장합니다.
인프라(파일 I/O) 영역이지만, ai_core/arrangement.py에서 import하여
AI 개발자의 로컬 테스트에서도 자연스럽게 사용됩니다.
"""

import logging

logger = logging.getLogger(__name__)


def save_midi(notes, source_pm, output_path, target_prog, target_name,
              original_song_path=None,
              actual_instrument_name=None, actual_midi_program=None):
    """
    원본 미디 파일(`original_song_path`)에 AI가 생성한 노트들만 새로운 전용 트랙으로
    덧붙여 저장(Append)합니다. 원본 트랙들의 Jitter 발생이나 SysEx 메타데이터
    손실을 100% 방지합니다.

    Args:
        actual_instrument_name: 실제 악기 이름 오버라이드 (예: "Viola"). None이면 target_name 사용.
        actual_midi_program:    실제 MIDI program 번호 오버라이드 (예: 41). None이면 target_prog 사용.
    """
    if not original_song_path:
        raise ValueError("original_song_path is required for Mido Append strategy")
        
    import mido
    mid = mido.MidiFile(original_song_path)

    # 엣지 1: 원본이 Type-0일 때 다중 트랙 저장을 위해 형식을 Type-1로 강제 전환
    if mid.type == 0:
        mid.type = 1

    # 트랙 이름: actual_instrument_name이 있으면 우선 사용 (예: "Viola")
    display_name = actual_instrument_name or target_name
        
    new_track = mido.MidiTrack()
    # 첫 메타 메시지로 트랙 명찰 부여 (악보/DAW 인식용)
    new_track.append(mido.MetaMessage('track_name', name=f"AI_{display_name}", time=0))
    
    # 엣지 2: 드럼 전용 채널 충돌 및 오버플로우 방어
    is_drum = (target_prog == 128)
    # MIDI program: 드럼이 아니고 actual_midi_program이 있으면 사용
    if is_drum:
        msg_prog = 0
    else:
        msg_prog = actual_midi_program if actual_midi_program is not None else target_prog

    # Mido 예외 방지: program은 무조건 0~127 범위 안에 있어야 함
    msg_prog = max(0, min(127, msg_prog))

    if is_drum:
        msg_chan = 9  # 드럼 채널은 무조건 9 (MIDI 10번 채널)
    else:
        # 사용 중인 전체 채널(0~15) 스캔
        used_channels = set()
        for track in mid.tracks:
            for msg in track:
                if hasattr(msg, 'channel'):
                    used_channels.add(msg.channel)
        
        # 9번 채널은 악몽의 근원이므로 검색 영역에서 아예 제외(Hard Exclusion)
        free_channels = [c for c in range(16) if c != 9 and c not in used_channels]
        if free_channels:
            msg_chan = free_channels[0]
        else:
            # 16개 채널을 모조리 사용한 최악의 곡: 최대한 피해 안 가는 채널 공유 (9번만큼은 제외)
            fallback = [c for c in range(16) if c != 9]
            msg_chan = fallback[0] if fallback else 0
            logger.warning(f"MIDI 16채널 한계 포화 상태 도달! AI 악기가 채널 {msg_chan}을 일부 공유합니다.")
            
    # 할당된 채널에 프로그램(악기 톤) 체인지 신호 기록
    new_track.append(mido.Message('program_change', program=msg_prog, channel=msg_chan, time=0))
    
    # 생성된 노트들(absolute seconds)을 절대 틱(Absolute Ticks)으로 재전환
    events = []
    for n in notes:
        st_tick = int(round(source_pm.time_to_tick(n["start"])))
        en_tick = int(round(source_pm.time_to_tick(n["end"])))
        pitch = max(0, min(127, int(n["pitch"])))
        vel = max(1, min(127, int(n["velocity"])))

        events.append((st_tick, 'note_on', pitch, vel))
        events.append((en_tick, 'note_off', pitch, 0))

    # 절대 틱 기준 시간 순 정렬. 
    # 동시간대라면 켜기 전 먼저 끄도록(note_off) 우선순위를 부여해 겹친 노트들의 틱 꼬임 방지
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'note_off' else 1))
    
    # 델타 시간(Delta Ticks)으로 전환하면서 트랙에 붙임
    last_tick = 0
    for tick, msg_type, pitch, vel in events:
        delta = tick - last_tick
        if delta < 0:
            delta = 0  # 수학적 역전의 여지 차단
        new_track.append(mido.Message(msg_type, note=pitch, velocity=vel, time=delta, channel=msg_chan))
        last_tick = tick
        
    # 트랙 종단 마커
    new_track.append(mido.MetaMessage('end_of_track', time=0))
    
    # 새로 빚은 순수 AI 트랙 한 줄을 원본 파일에 조심스레 첨부
    mid.tracks.append(new_track)
    mid.save(output_path)
    logger.info(f"원본 100% 보존 기반 병합 저장 완료: {output_path} (생성된 노트 {len(notes)}개)")
