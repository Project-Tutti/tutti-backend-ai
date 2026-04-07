"""
test_midi_processor.py
MIDI 프로세서 철저 검증 테스트 스위트

검증 항목:
  1. Type 1 MIDI: 트랙 program_change 변경
  2. Type 1 MIDI: 트랙 삭제 (인덱스 정합성)
  3. Type 0 MIDI: 채널 program_change 변경
  4. Type 0 MIDI: 채널 삭제 + delta time 보존
  5. 빈 mappings 처리
  6. 범위 초과 trackIndex 처리
  7. 복합 시나리오: 삭제 + 재매핑 동시
  8. 전체 라운드트립: 생성 → 재매핑 → 저장 → 재로드 → 검증
  9. pretty_midi 호환성: mido 재매핑 결과를 pretty_midi가 올바르게 읽는지
  10. 실제 복잡한 MIDI: 다수 트랙/채널, 컨트롤체인지, 피치벤드 등
"""

import os
import sys
import copy
import tempfile
import traceback

import mido
import pretty_midi

# ──────────────────────────────────────────────
# 프로젝트 모듈 임포트를 위한 경로 추가
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.services.midi_processor import (
    remap_original_tracks,
    _remap_type0,
    _remap_type1,
    DROP_INSTRUMENT_ID,
)


# ──────────────────────────────────────────────
# 테스트 유틸리티
# ──────────────────────────────────────────────
class FakeMapping:
    """Pydantic Mapping 모델을 흉내내는 간단한 객체"""
    def __init__(self, trackIndex, targetInstrumentId):
        self.trackIndex = trackIndex
        self.targetInstrumentId = targetInstrumentId


def absolute_times(track):
    """mido 트랙의 delta time → absolute time 목록 반환"""
    result = []
    abs_t = 0
    for msg in track:
        abs_t += msg.time
        result.append(abs_t)
    return result


def total_duration_ticks(track):
    """트랙의 총 duration (tick)"""
    return sum(msg.time for msg in track)


def save_and_reload(mid, suffix=".mid"):
    """임시 파일에 저장 후 다시 로드하여 반환"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        path = f.name
    mid.save(path)
    reloaded = mido.MidiFile(path)
    os.unlink(path)
    return reloaded


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  ✅ {name}")

    def fail(self, name, detail):
        self.failed += 1
        self.errors.append((name, detail))
        print(f"  🔴 {name}: {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 60}")
        print(f"총 {total}개 테스트: ✅ {self.passed} 통과, 🔴 {self.failed} 실패")
        if self.errors:
            print("\n실패한 테스트:")
            for name, detail in self.errors:
                print(f"  - {name}: {detail}")
        print(f"{'=' * 60}")
        return self.failed == 0


results = TestResult()


# ══════════════════════════════════════════════
# 테스트 MIDI 생성 헬퍼
# ══════════════════════════════════════════════

def create_type1_midi():
    """
    Type 1 MIDI 생성:
    - Track 0: 메타 트랙 (tempo, time_sig)
    - Track 1: Piano (ch0, prog=0), 3개 노트
    - Track 2: Violin (ch1, prog=40), 3개 노트
    - Track 3: Flute (ch2, prog=73), 3개 노트
    """
    mid = mido.MidiFile(type=1, ticks_per_beat=480)

    # Track 0: 메타
    meta_track = mido.MidiTrack()
    meta_track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    meta_track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    meta_track.append(mido.MetaMessage('end_of_track', time=1920))
    mid.tracks.append(meta_track)

    # Track 1: Piano
    t1 = mido.MidiTrack()
    t1.append(mido.MetaMessage('track_name', name='Piano', time=0))
    t1.append(mido.Message('program_change', channel=0, program=0, time=0))
    for i, note in enumerate([60, 64, 67]):
        t1.append(mido.Message('note_on', channel=0, note=note, velocity=100, time=i*480))
        t1.append(mido.Message('note_off', channel=0, note=note, velocity=0, time=240))
    t1.append(mido.MetaMessage('end_of_track', time=0))
    mid.tracks.append(t1)

    # Track 2: Violin
    t2 = mido.MidiTrack()
    t2.append(mido.MetaMessage('track_name', name='Violin', time=0))
    t2.append(mido.Message('program_change', channel=1, program=40, time=0))
    for i, note in enumerate([55, 59, 62]):
        t2.append(mido.Message('note_on', channel=1, note=note, velocity=90, time=i*480))
        t2.append(mido.Message('note_off', channel=1, note=note, velocity=0, time=240))
    t2.append(mido.MetaMessage('end_of_track', time=0))
    mid.tracks.append(t2)

    # Track 3: Flute
    t3 = mido.MidiTrack()
    t3.append(mido.MetaMessage('track_name', name='Flute', time=0))
    t3.append(mido.Message('program_change', channel=2, program=73, time=0))
    for i, note in enumerate([72, 76, 79]):
        t3.append(mido.Message('note_on', channel=2, note=note, velocity=80, time=i*480))
        t3.append(mido.Message('note_off', channel=2, note=note, velocity=0, time=240))
    t3.append(mido.MetaMessage('end_of_track', time=0))
    mid.tracks.append(t3)

    return mid


def create_type0_midi():
    """
    Type 0 MIDI 생성:
    단일 트랙에 3개 채널의 이벤트 혼합.
    채널별 배치가 일정한 간격으로 인터리빙됨.
    """
    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()

    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))

    # 채널별 program_change
    track.append(mido.Message('program_change', channel=0, program=0, time=0))    # Piano
    track.append(mido.Message('program_change', channel=1, program=40, time=0))   # Violin
    track.append(mido.Message('program_change', channel=2, program=73, time=0))   # Flute

    # 인터리빙 노트: ch0, ch1, ch2가 번갈아 등장
    # time=0: ch0 note_on 60
    track.append(mido.Message('note_on', channel=0, note=60, velocity=100, time=0))
    # time=120: ch1 note_on 55
    track.append(mido.Message('note_on', channel=1, note=55, velocity=90, time=120))
    # time=120: ch2 note_on 72
    track.append(mido.Message('note_on', channel=2, note=72, velocity=80, time=120))
    # time=120: ch0 note_off 60
    track.append(mido.Message('note_off', channel=0, note=60, velocity=0, time=120))
    # time=120: ch1 note_off 55
    track.append(mido.Message('note_off', channel=1, note=55, velocity=0, time=120))
    # time=120: ch2 note_off 72
    track.append(mido.Message('note_off', channel=2, note=72, velocity=0, time=120))

    # 두 번째 비트
    track.append(mido.Message('note_on', channel=0, note=64, velocity=100, time=0))
    track.append(mido.Message('note_on', channel=1, note=59, velocity=90, time=120))
    track.append(mido.Message('note_on', channel=2, note=76, velocity=80, time=120))
    track.append(mido.Message('note_off', channel=0, note=64, velocity=0, time=120))
    track.append(mido.Message('note_off', channel=1, note=59, velocity=0, time=120))
    track.append(mido.Message('note_off', channel=2, note=76, velocity=0, time=120))

    track.append(mido.MetaMessage('end_of_track', time=0))
    mid.tracks.append(track)

    return mid


def create_complex_type0_midi():
    """
    복잡한 Type 0 MIDI: 컨트롤 체인지, 피치벤드, 다수 메타 메시지 포함.
    실제 DAW 출력과 유사한 구조.
    """
    mid = mido.MidiFile(type=0, ticks_per_beat=480)
    track = mido.MidiTrack()

    # 메타 헤더
    track.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4, time=0))
    track.append(mido.MetaMessage('key_signature', key='C', time=0))
    track.append(mido.MetaMessage('track_name', name='Complex Track', time=0))

    # Program changes
    track.append(mido.Message('program_change', channel=0, program=0, time=0))
    track.append(mido.Message('program_change', channel=1, program=40, time=0))
    track.append(mido.Message('program_change', channel=9, program=0, time=0))   # 드럼
    track.append(mido.Message('program_change', channel=3, program=33, time=0))  # Bass

    # 채널 0: CC + 노트
    track.append(mido.Message('control_change', channel=0, control=7, value=100, time=0))
    track.append(mido.Message('control_change', channel=0, control=10, value=64, time=0))
    track.append(mido.Message('note_on', channel=0, note=60, velocity=100, time=0))
    track.append(mido.Message('note_on', channel=9, note=36, velocity=110, time=0))    # 드럼 킥
    track.append(mido.Message('note_on', channel=1, note=55, velocity=90, time=0))
    track.append(mido.Message('note_on', channel=3, note=36, velocity=80, time=0))     # 베이스

    # 120틱 후 피치벤드
    track.append(mido.Message('pitchwheel', channel=1, pitch=2000, time=120))

    # 마커 (meta - 채널 없음)
    track.append(mido.MetaMessage('marker', text='Chorus', time=60))

    # 모든 채널 note_off
    track.append(mido.Message('note_off', channel=0, note=60, velocity=0, time=60))
    track.append(mido.Message('note_off', channel=9, note=36, velocity=0, time=0))
    track.append(mido.Message('note_off', channel=1, note=55, velocity=0, time=0))
    track.append(mido.Message('note_off', channel=3, note=36, velocity=0, time=0))

    # 다음 비트: 컨트롤체인지 + 노트
    track.append(mido.Message('control_change', channel=1, control=64, value=127, time=240))  # sustain
    track.append(mido.Message('note_on', channel=0, note=64, velocity=100, time=0))
    track.append(mido.Message('note_on', channel=1, note=59, velocity=90, time=0))
    track.append(mido.Message('note_off', channel=0, note=64, velocity=0, time=240))
    track.append(mido.Message('note_off', channel=1, note=59, velocity=0, time=0))
    track.append(mido.Message('control_change', channel=1, control=64, value=0, time=0))     # sustain off

    track.append(mido.MetaMessage('end_of_track', time=0))
    mid.tracks.append(track)

    return mid


# ══════════════════════════════════════════════
# 테스트 1: Type 1 program_change 변경
# ══════════════════════════════════════════════
def test_type1_program_change():
    name = "T1-01: Type 1 program_change 변경"
    try:
        mid = create_type1_midi()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        # Track 1 (Piano, prog=0) → prog=25 (Guitar)
        mappings = [FakeMapping(trackIndex=1, targetInstrumentId=25)]
        remap_original_tracks(path, mappings)

        reloaded = mido.MidiFile(path)
        os.unlink(path)

        progs = [msg.program for msg in reloaded.tracks[1] if msg.type == 'program_change']
        if progs == [25]:
            results.ok(name)
        else:
            results.fail(name, f"expected [25], got {progs}")
    except Exception as e:
        results.fail(name, f"예외: {e}")


# ══════════════════════════════════════════════
# 테스트 2: Type 1 트랙 삭제
# ══════════════════════════════════════════════
def test_type1_track_delete():
    name = "T1-02: Type 1 트랙 삭제 (역순 인덱스 정합성)"
    try:
        mid = create_type1_midi()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        original_track_count = len(mid.tracks)  # 4

        # Track 2 (Violin) 삭제
        mappings = [FakeMapping(trackIndex=2, targetInstrumentId=DROP_INSTRUMENT_ID)]
        remap_original_tracks(path, mappings)

        reloaded = mido.MidiFile(path)
        os.unlink(path)

        if len(reloaded.tracks) == original_track_count - 1:
            # 남은 트랙은: meta(0), Piano(1), Flute(2→이전3)
            # Flute 트랙에 prog=73이 있어야 함
            flute_progs = [msg.program for msg in reloaded.tracks[2]
                           if msg.type == 'program_change']
            if flute_progs == [73]:
                results.ok(name)
            else:
                results.fail(name, f"Flute 트랙 prog expected [73], got {flute_progs}")
        else:
            results.fail(name, f"트랙 수 expected {original_track_count-1}, got {len(reloaded.tracks)}")
    except Exception as e:
        results.fail(name, f"예외: {e}")


# ══════════════════════════════════════════════
# 테스트 3: Type 1 다중 삭제 (인접 인덱스)
# ══════════════════════════════════════════════
def test_type1_multi_delete():
    name = "T1-03: Type 1 다중 인접 트랙 삭제"
    try:
        mid = create_type1_midi()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        # Track 1 (Piano) + Track 3 (Flute) 삭제
        mappings = [
            FakeMapping(trackIndex=1, targetInstrumentId=DROP_INSTRUMENT_ID),
            FakeMapping(trackIndex=3, targetInstrumentId=DROP_INSTRUMENT_ID),
        ]
        remap_original_tracks(path, mappings)

        reloaded = mido.MidiFile(path)
        os.unlink(path)

        # 남은: meta(0), Violin(이전2→1)
        if len(reloaded.tracks) != 2:
            results.fail(name, f"트랙 수 expected 2, got {len(reloaded.tracks)}")
            return

        violin_progs = [msg.program for msg in reloaded.tracks[1]
                        if msg.type == 'program_change']
        if violin_progs == [40]:
            results.ok(name)
        else:
            results.fail(name, f"Violin prog expected [40], got {violin_progs}")
    except Exception as e:
        results.fail(name, f"예외: {e}")


# ══════════════════════════════════════════════
# 테스트 4: Type 1 삭제+변경 복합
# ══════════════════════════════════════════════
def test_type1_delete_and_remap():
    name = "T1-04: Type 1 삭제 + 재매핑 복합"
    try:
        mid = create_type1_midi()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        mappings = [
            FakeMapping(trackIndex=1, targetInstrumentId=25),   # Piano→Guitar
            FakeMapping(trackIndex=2, targetInstrumentId=DROP_INSTRUMENT_ID),  # Violin 삭제
            FakeMapping(trackIndex=3, targetInstrumentId=65),   # Flute→Sax
        ]
        remap_original_tracks(path, mappings)

        reloaded = mido.MidiFile(path)
        os.unlink(path)

        # 남은: meta(0), Guitar(1), Sax(2→이전3)
        if len(reloaded.tracks) != 3:
            results.fail(name, f"트랙 수 expected 3, got {len(reloaded.tracks)}")
            return

        guitar_progs = [msg.program for msg in reloaded.tracks[1] if msg.type == 'program_change']
        sax_progs = [msg.program for msg in reloaded.tracks[2] if msg.type == 'program_change']

        if guitar_progs == [25] and sax_progs == [65]:
            results.ok(name)
        else:
            results.fail(name, f"Guitar={guitar_progs}, Sax={sax_progs}")
    except Exception as e:
        results.fail(name, f"예외: {e}")


# ══════════════════════════════════════════════
# 테스트 5: Type 0 program_change 변경
# ══════════════════════════════════════════════
def test_type0_program_change():
    name = "T0-01: Type 0 채널 program_change 변경"
    try:
        mid = create_type0_midi()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        # ch0 (Piano→Guitar), ch2 (Flute→Sax)
        mappings = [
            FakeMapping(trackIndex=0, targetInstrumentId=25),
            FakeMapping(trackIndex=2, targetInstrumentId=65),
        ]
        remap_original_tracks(path, mappings)

        reloaded = mido.MidiFile(path)
        os.unlink(path)

        progs = {}
        for msg in reloaded.tracks[0]:
            if msg.type == 'program_change':
                progs[msg.channel] = msg.program

        if progs.get(0) == 25 and progs.get(1) == 40 and progs.get(2) == 65:
            results.ok(name)
        else:
            results.fail(name, f"expected ch0=25,ch1=40,ch2=65, got {progs}")
    except Exception as e:
        results.fail(name, f"예외: {e}")


# ══════════════════════════════════════════════
# 테스트 6: Type 0 채널 삭제 + delta time 보존
# ══════════════════════════════════════════════
def test_type0_channel_delete_timing():
    name = "T0-02: Type 0 채널 삭제 — delta time 보존"
    try:
        mid = create_type0_midi()

        # 삭제 전 총 duration 기록
        original_duration = total_duration_ticks(mid.tracks[0])
        original_abs_times = absolute_times(mid.tracks[0])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        # ch1 (Violin) 삭제
        mappings = [FakeMapping(trackIndex=1, targetInstrumentId=DROP_INSTRUMENT_ID)]
        remap_original_tracks(path, mappings)

        reloaded = mido.MidiFile(path)
        os.unlink(path)

        new_track = reloaded.tracks[0]

        # 검증 1: ch1 메시지가 완전히 제거되었는지
        ch1_msgs = [msg for msg in new_track if hasattr(msg, 'channel') and msg.channel == 1]
        if ch1_msgs:
            results.fail(name, f"ch1 메시지가 {len(ch1_msgs)}개 남아있음")
            return

        # 검증 2: 총 duration이 보존되었는지
        new_duration = total_duration_ticks(new_track)
        if new_duration != original_duration:
            results.fail(name, f"총 duration 변경: {original_duration} → {new_duration}")
            return

        # 검증 3: 생존한 메시지들의 absolute time이 원본과 동일한지
        new_abs = absolute_times(new_track)
        original_map = {}
        for msg, abs_t in zip(mid.tracks[0], original_abs_times):
            key = (msg.type, getattr(msg, 'channel', None), getattr(msg, 'note', None))
            if key not in original_map:
                original_map[key] = []
            original_map[key].append(abs_t)

        ok = True
        for msg, abs_t in zip(new_track, new_abs):
            key = (msg.type, getattr(msg, 'channel', None), getattr(msg, 'note', None))
            if key in original_map and original_map[key]:
                expected = original_map[key].pop(0)
                if abs_t != expected:
                    results.fail(name, f"{msg.type} ch={getattr(msg, 'channel', '?')}: abs expected {expected}, got {abs_t}")
                    ok = False
                    break

        if ok:
            results.ok(name)
    except Exception as e:
        results.fail(name, f"예외: {e}")


# ══════════════════════════════════════════════
# 테스트 7: Type 0 다중 채널 삭제
# ══════════════════════════════════════════════
def test_type0_multi_channel_delete():
    name = "T0-03: Type 0 다중 채널 삭제"
    try:
        mid = create_type0_midi()
        original_duration = total_duration_ticks(mid.tracks[0])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        # ch0, ch2 삭제 → ch1만 남김
        mappings = [
            FakeMapping(trackIndex=0, targetInstrumentId=DROP_INSTRUMENT_ID),
            FakeMapping(trackIndex=2, targetInstrumentId=DROP_INSTRUMENT_ID),
        ]
        remap_original_tracks(path, mappings)

        reloaded = mido.MidiFile(path)
        os.unlink(path)

        new_track = reloaded.tracks[0]

        # ch0, ch2 메시지 없어야 함
        bad_channels = set()
        for msg in new_track:
            if hasattr(msg, 'channel') and msg.channel in {0, 2}:
                bad_channels.add(msg.channel)
        if bad_channels:
            results.fail(name, f"삭제되어야 할 채널 메시지 존재: {bad_channels}")
            return

        # 총 duration 보존
        new_duration = total_duration_ticks(new_track)
        if new_duration != original_duration:
            results.fail(name, f"총 duration 변경: {original_duration} → {new_duration}")
            return

        # ch1 메시지가 존재하는지
        ch1_notes = [msg for msg in new_track
                     if msg.type == 'note_on' and hasattr(msg, 'channel') and msg.channel == 1]
        if len(ch1_notes) >= 2:
            results.ok(name)
        else:
            results.fail(name, f"ch1 note_on 부족: {len(ch1_notes)}개")
    except Exception as e:
        results.fail(name, f"예외: {e}")


# ══════════════════════════════════════════════
# 테스트 8: 빈 mappings
# ══════════════════════════════════════════════
def test_empty_mappings():
    name = "EDGE-01: 빈 mappings → 변경 없음"
    try:
        mid = create_type1_midi()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        original_data = open(path, 'rb').read()
        remap_original_tracks(path, [])
        new_data = open(path, 'rb').read()
        os.unlink(path)

        if original_data == new_data:
            results.ok(name)
        else:
            results.fail(name, "파일이 변경됨 (변경되지 않아야 함)")
    except Exception as e:
        results.fail(name, f"예외: {e}")


# ══════════════════════════════════════════════
# 테스트 9: 범위 초과 trackIndex
# ══════════════════════════════════════════════
def test_out_of_range_track():
    name = "EDGE-02: 범위 초과 trackIndex → 무시"
    try:
        mid = create_type1_midi()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        mappings = [FakeMapping(trackIndex=99, targetInstrumentId=25)]
        remap_original_tracks(path, mappings)

        reloaded = mido.MidiFile(path)
        os.unlink(path)

        # 트랙 수 변경 없음
        if len(reloaded.tracks) == 4:
            results.ok(name)
        else:
            results.fail(name, f"트랙 수 expected 4, got {len(reloaded.tracks)}")
    except Exception as e:
        results.fail(name, f"예외: {e}")


# ══════════════════════════════════════════════
# 테스트 10: 복잡한 Type 0 — CC + 피치벤드 + 메타 보존
# ══════════════════════════════════════════════
def test_complex_type0_channel_delete():
    name = "T0-04: 복잡한 Type 0 — CC/피치벤드/메타 보존"
    try:
        mid = create_complex_type0_midi()
        original_duration = total_duration_ticks(mid.tracks[0])
        original_msg_count = len(mid.tracks[0])

        # 원본 메타 메시지 수
        original_meta_count = sum(1 for msg in mid.tracks[0] if msg.is_meta)

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        # ch1 삭제 → note_on/off, CC, pitchwheel 모두 제거
        mappings = [FakeMapping(trackIndex=1, targetInstrumentId=DROP_INSTRUMENT_ID)]
        remap_original_tracks(path, mappings)

        reloaded = mido.MidiFile(path)
        os.unlink(path)

        new_track = reloaded.tracks[0]

        # 검증 1: ch1 메시지 완전 제거
        ch1_msgs = [msg for msg in new_track if hasattr(msg, 'channel') and msg.channel == 1]
        if ch1_msgs:
            results.fail(name, f"ch1 메시지 {len(ch1_msgs)}개 잔존 — 타입: {set(m.type for m in ch1_msgs)}")
            return

        # 검증 2: 메타 메시지 전부 보존
        new_meta_count = sum(1 for msg in new_track if msg.is_meta)
        if new_meta_count != original_meta_count:
            results.fail(name, f"메타 메시지 수: {original_meta_count} → {new_meta_count}")
            return

        # 검증 3: 마커 text 보존
        markers = [msg for msg in new_track if msg.type == 'marker']
        if not markers or markers[0].text != 'Chorus':
            results.fail(name, f"마커 누락 또는 변형: {markers}")
            return

        # 검증 4: 총 duration 보존
        new_duration = total_duration_ticks(new_track)
        if new_duration != original_duration:
            results.fail(name, f"총 duration: {original_duration} → {new_duration}")
            return

        # 검증 5: ch0, ch9, ch3 노트 보존
        surviving_channels = set()
        for msg in new_track:
            if hasattr(msg, 'channel'):
                surviving_channels.add(msg.channel)
        if surviving_channels == {0, 9, 3}:
            results.ok(name)
        else:
            results.fail(name, f"생존 채널 expected {{0,9,3}}, got {surviving_channels}")
    except Exception as e:
        results.fail(name, f"예외: {e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════
# 테스트 11: 복잡한 Type 0 삭제+변경 동시
# ══════════════════════════════════════════════
def test_complex_type0_delete_and_remap():
    name = "T0-05: 복잡한 Type 0 삭제+변경 복합"
    try:
        mid = create_complex_type0_midi()
        original_duration = total_duration_ticks(mid.tracks[0])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        # ch0을 Guitar로 변경, ch1 삭제, ch3을 Violin으로 변경
        mappings = [
            FakeMapping(trackIndex=0, targetInstrumentId=25),    # Piano→Guitar
            FakeMapping(trackIndex=1, targetInstrumentId=DROP_INSTRUMENT_ID),  # Violin 삭제
            FakeMapping(trackIndex=3, targetInstrumentId=40),    # Bass→Violin
        ]
        remap_original_tracks(path, mappings)

        reloaded = mido.MidiFile(path)
        os.unlink(path)

        new_track = reloaded.tracks[0]

        # 검증 1: ch1 없음
        ch1_msgs = [msg for msg in new_track if hasattr(msg, 'channel') and msg.channel == 1]
        if ch1_msgs:
            results.fail(name, f"ch1 잔존: {len(ch1_msgs)}개")
            return

        # 검증 2: program_change 확인
        progs = {}
        for msg in new_track:
            if msg.type == 'program_change':
                progs[msg.channel] = msg.program

        if progs.get(0) != 25:
            results.fail(name, f"ch0 prog expected 25, got {progs.get(0)}")
            return
        if progs.get(3) != 40:
            results.fail(name, f"ch3 prog expected 40, got {progs.get(3)}")
            return

        # 검증 3: 총 duration 보존
        new_duration = total_duration_ticks(new_track)
        if new_duration != original_duration:
            results.fail(name, f"총 duration: {original_duration} → {new_duration}")
            return

        results.ok(name)
    except Exception as e:
        results.fail(name, f"예외: {e}")


# ══════════════════════════════════════════════
# 테스트 12: pretty_midi 호환성
# ══════════════════════════════════════════════
def test_pretty_midi_compatibility_type1():
    name = "COMPAT-01: Type 1 재매핑 후 pretty_midi 로드"
    try:
        mid = create_type1_midi()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        # Guitar(25)로 변경, Violin 삭제
        mappings = [
            FakeMapping(trackIndex=1, targetInstrumentId=25),
            FakeMapping(trackIndex=2, targetInstrumentId=DROP_INSTRUMENT_ID),
        ]
        remap_original_tracks(path, mappings)

        # pretty_midi로 읽기
        pm = pretty_midi.PrettyMIDI(path)
        os.unlink(path)

        # 악기 확인: Guitar(25), Flute(73)
        prog_set = set()
        for inst in pm.instruments:
            if not inst.is_drum:
                prog_set.add(inst.program)

        if 25 in prog_set and 73 in prog_set and 40 not in prog_set:
            # 노트 수 확인
            total_notes = sum(len(inst.notes) for inst in pm.instruments)
            if total_notes >= 6:  # Guitar 3 + Flute 3
                results.ok(name)
            else:
                results.fail(name, f"노트 수 부족: {total_notes}")
        else:
            results.fail(name, f"악기 prog set expected {{25,73}}, got {prog_set}")
    except Exception as e:
        results.fail(name, f"예외: {e}\n{traceback.format_exc()}")


def test_pretty_midi_compatibility_type0():
    name = "COMPAT-02: Type 0 채널삭제 후 pretty_midi 로드 + timing 검증"
    try:
        mid = create_type0_midi()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        # 삭제 전 pretty_midi timing 기록
        pm_before = pretty_midi.PrettyMIDI(path)
        original_end_time = pm_before.get_end_time()

        # ch1 삭제
        mappings = [FakeMapping(trackIndex=1, targetInstrumentId=DROP_INSTRUMENT_ID)]
        remap_original_tracks(path, mappings)

        pm_after = pretty_midi.PrettyMIDI(path)
        os.unlink(path)

        new_end_time = pm_after.get_end_time()

        # ch1 악기가 없어야 함 (prog=40)
        remaining_progs = set()
        for inst in pm_after.instruments:
            if not inst.is_drum:
                remaining_progs.add(inst.program)

        if 40 in remaining_progs:
            results.fail(name, f"삭제된 ch1 (prog=40)이 여전히 존재")
            return

        # end_time이 보존 또는 적절하게 조정되었는지
        # (ch1 노트가 제거되면 end_time이 같거나 짧아질 수 있음)
        if new_end_time <= original_end_time + 0.001:
            results.ok(name)
        else:
            results.fail(name, f"end_time 비정상: {original_end_time:.4f} → {new_end_time:.4f}")
    except Exception as e:
        results.fail(name, f"예외: {e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════
# 테스트 14: Type 0 전체 채널 삭제 → 메타만 남음
# ══════════════════════════════════════════════
def test_type0_delete_all_channels():
    name = "EDGE-03: Type 0 전체 채널 삭제 → 메타만 남음"
    try:
        mid = create_type0_midi()  # ch0, ch1, ch2
        original_duration = total_duration_ticks(mid.tracks[0])

        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        mappings = [
            FakeMapping(trackIndex=0, targetInstrumentId=DROP_INSTRUMENT_ID),
            FakeMapping(trackIndex=1, targetInstrumentId=DROP_INSTRUMENT_ID),
            FakeMapping(trackIndex=2, targetInstrumentId=DROP_INSTRUMENT_ID),
        ]
        remap_original_tracks(path, mappings)

        reloaded = mido.MidiFile(path)
        os.unlink(path)

        new_track = reloaded.tracks[0]

        # 채널 메시지 없어야 함
        channel_msgs = [msg for msg in new_track if hasattr(msg, 'channel')]
        if channel_msgs:
            results.fail(name, f"채널 메시지 {len(channel_msgs)}개 잔존")
            return

        # 메타 메시지만 있어야 함
        for msg in new_track:
            if not msg.is_meta:
                results.fail(name, f"비-메타 메시지 발견: {msg}")
                return

        # delta time 보존
        new_duration = total_duration_ticks(new_track)
        if new_duration != original_duration:
            results.fail(name, f"총 duration: {original_duration} → {new_duration}")
            return

        results.ok(name)
    except Exception as e:
        results.fail(name, f"예외: {e}")


# ══════════════════════════════════════════════
# 테스트 15: 라운드트립 — save_midi(inference) 후 재로드
# ══════════════════════════════════════════════
def test_full_roundtrip_mido_then_pretty_midi():
    name = "RT-01: mido 재매핑 → pretty_midi read → 추론 저장 시뮬레이션"
    try:
        mid = create_type1_midi()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        # Step 1: mido로 재매핑 (arrangement 파이프라인 시뮬)
        mappings = [
            FakeMapping(trackIndex=1, targetInstrumentId=25),   # Piano→Guitar
            FakeMapping(trackIndex=2, targetInstrumentId=DROP_INSTRUMENT_ID),  # Violin 삭제
        ]
        remap_original_tracks(path, mappings)

        # Step 2: pretty_midi로 로드 (inference 파이프라인)
        pm = pretty_midi.PrettyMIDI(path)

        # Step 3: 추론 결과 시뮬 — 새 악기 추가
        import copy
        out_pm = copy.deepcopy(pm)
        new_inst = pretty_midi.Instrument(program=40, is_drum=False, name="violin_gen")
        new_inst.notes.append(pretty_midi.Note(velocity=100, pitch=60, start=0.0, end=0.5))
        new_inst.notes.append(pretty_midi.Note(velocity=100, pitch=64, start=0.5, end=1.0))
        out_pm.instruments.append(new_inst)

        # Step 4: 최종 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            out_path = f.name
        out_pm.write(out_path)

        # Step 5: 재로드 검증
        final_pm = pretty_midi.PrettyMIDI(out_path)
        os.unlink(path)
        os.unlink(out_path)

        # 악기 수 확인: Guitar(25) + Flute(73) + Violin_gen(40) = 3
        if len(final_pm.instruments) < 3:
            results.fail(name, f"악기 수 expected ≥3, got {len(final_pm.instruments)}")
            return

        prog_set = {i.program for i in final_pm.instruments if not i.is_drum}
        if 25 in prog_set and 73 in prog_set and 40 in prog_set:
            total_notes = sum(len(i.notes) for i in final_pm.instruments)
            if total_notes >= 8:  # Guitar 3 + Flute 3 + Generated 2
                results.ok(name)
            else:
                results.fail(name, f"노트 수 부족: {total_notes}")
        else:
            results.fail(name, f"프로그램 세트: {prog_set}")
    except Exception as e:
        results.fail(name, f"예외: {e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════
# 테스트 16: Type 1 타이밍 보존
# ══════════════════════════════════════════════
def test_type1_timing_preservation():
    name = "T1-05: Type 1 재매핑 후 노트 타이밍 보존"
    try:
        mid = create_type1_midi()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        pm_before = pretty_midi.PrettyMIDI(path)
        before_timing = {}
        for inst in pm_before.instruments:
            before_timing[inst.program] = [(n.pitch, round(n.start, 4), round(n.end, 4))
                                           for n in inst.notes]

        # program_change만 변경 (타이밍 영향 없어야 함)
        mappings = [FakeMapping(trackIndex=1, targetInstrumentId=25)]
        remap_original_tracks(path, mappings)

        pm_after = pretty_midi.PrettyMIDI(path)
        os.unlink(path)

        # Guitar(25)의 노트 타이밍 == 이전 Piano(0)의 타이밍
        after_timing = {}
        for inst in pm_after.instruments:
            after_timing[inst.program] = [(n.pitch, round(n.start, 4), round(n.end, 4))
                                          for n in inst.notes]

        if before_timing.get(0) == after_timing.get(25):
            results.ok(name)
        else:
            results.fail(name, f"Piano→Guitar 타이밍 불일치")
    except Exception as e:
        results.fail(name, f"예외: {e}")


# ══════════════════════════════════════════════
# 테스트 17: Type 0 채널 삭제 시 note timing (pretty_midi)
# ══════════════════════════════════════════════
def test_type0_delete_precise_timing():
    name = "T0-06: Type 0 채널 삭제 — pretty_midi 노트별 정밀 timing"
    try:
        mid = create_type0_midi()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        # 삭제 전 ch0, ch2 노트 타이밍 기록
        pm_before = pretty_midi.PrettyMIDI(path)
        before_notes = {}
        for inst in pm_before.instruments:
            before_notes[inst.program] = [
                (n.pitch, round(n.start, 4), round(n.end, 4)) for n in inst.notes
            ]

        # ch1 (Violin, prog=40) 삭제
        mappings = [FakeMapping(trackIndex=1, targetInstrumentId=DROP_INSTRUMENT_ID)]
        remap_original_tracks(path, mappings)

        pm_after = pretty_midi.PrettyMIDI(path)
        os.unlink(path)

        after_notes = {}
        for inst in pm_after.instruments:
            after_notes[inst.program] = [
                (n.pitch, round(n.start, 4), round(n.end, 4)) for n in inst.notes
            ]

        # ch0 (Piano, prog=0) 타이밍 보존 확인
        if before_notes.get(0) != after_notes.get(0):
            results.fail(name, f"ch0 타이밍 변경!\n  before: {before_notes.get(0)}\n  after:  {after_notes.get(0)}")
            return

        # ch2 (Flute, prog=73) 타이밍 보존 확인
        if before_notes.get(73) != after_notes.get(73):
            results.fail(name, f"ch2 타이밍 변경!\n  before: {before_notes.get(73)}\n  after:  {after_notes.get(73)}")
            return

        # Violin(40)이 제거되었는지
        if 40 in after_notes:
            results.fail(name, f"삭제된 Violin(40) 잔존")
            return

        results.ok(name)
    except Exception as e:
        results.fail(name, f"예외: {e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════
# 테스트 18: Type 0 재매핑 결과 program_change 값이 정확히 pretty_midi에 반영되는지
# ══════════════════════════════════════════════
def test_type0_remap_pretty_midi_programs():
    name = "COMPAT-03: Type 0 재매핑 → pretty_midi 악기 판별"
    try:
        mid = create_complex_type0_midi()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mid") as f:
            path = f.name
        mid.save(path)

        # ch0: Piano(0) → Guitar(25), ch3: Bass(33) → Violin(40)
        mappings = [
            FakeMapping(trackIndex=0, targetInstrumentId=25),
            FakeMapping(trackIndex=3, targetInstrumentId=40),
        ]
        remap_original_tracks(path, mappings)

        pm = pretty_midi.PrettyMIDI(path)
        os.unlink(path)

        prog_set = set()
        for inst in pm.instruments:
            if not inst.is_drum:
                prog_set.add(inst.program)

        # Guitar(25), Violin_orig(40→unchanged? no, ch1 is 40), Violin_remp(ch3→40)
        # ch1은 원래 prog=40, ch3이 40으로 변경 → pretty_midi가 같은 프로그램 합칠 수 있음
        # 핵심은 25가 존재하고 33(원래 bass)이 없는 것
        if 25 in prog_set and 33 not in prog_set:
            results.ok(name)
        else:
            results.fail(name, f"prog set: {prog_set}")
    except Exception as e:
        results.fail(name, f"예외: {e}")


# ══════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 60)
    print("🔬 MIDI 프로세서 철저 검증 테스트")
    print("=" * 60)

    print("\n── Type 1 MIDI 테스트 ──")
    test_type1_program_change()
    test_type1_track_delete()
    test_type1_multi_delete()
    test_type1_delete_and_remap()
    test_type1_timing_preservation()

    print("\n── Type 0 MIDI 테스트 ──")
    test_type0_program_change()
    test_type0_channel_delete_timing()
    test_type0_multi_channel_delete()
    test_type0_delete_precise_timing()

    print("\n── 복잡한 Type 0 (CC/피치벤드/메타) ──")
    test_complex_type0_channel_delete()
    test_complex_type0_delete_and_remap()

    print("\n── 에지케이스 ──")
    test_empty_mappings()
    test_out_of_range_track()
    test_type0_delete_all_channels()

    print("\n── pretty_midi 호환성 ──")
    test_pretty_midi_compatibility_type1()
    test_pretty_midi_compatibility_type0()
    test_type0_remap_pretty_midi_programs()

    print("\n── 전체 라운드트립 ──")
    test_full_roundtrip_mido_then_pretty_midi()

    all_pass = results.summary()
    sys.exit(0 if all_pass else 1)
