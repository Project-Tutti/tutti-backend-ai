"""
test_midi_corruption_analysis.py
delta time 외의 모든 데이터 결손/오염 벡터 분석 및 검증

분석 항목:
  ┌─────────────────────────────────────────────────────────────┐
  │ 1. 참조 오염 (Reference Mutation)                          │
  │    - _remap_type0에서 program_change 변경(L129-131)이       │
  │      채널 삭제 전에 발생 → 삭제 대상 채널의 prog도 변경?    │
  │    - msg.copy()가 모든 속성을 올바르게 복사하는지            │
  │                                                             │
  │ 2. 메시지 속성 보존                                         │
  │    - velocity, note, channel 등 비-time 속성이 copy 후 유지 │
  │    - MetaMessage 속성 (tempo, text 등) 보존                 │
  │                                                             │
  │ 3. 순서 보존                                                │
  │    - 삭제/변경 후 메시지 순서가 바뀌지 않는지               │
  │                                                             │
  │ 4. 삭제+변경 중복 매핑                                      │
  │    - 같은 채널에 변경과 삭제가 동시에 매핑되면?             │
  │                                                             │
  │ 5. Type 0 program_change 누락 케이스                        │
  │    - program_change 없는 채널(기본 prog=0) 처리             │
  │                                                             │
  │ 6. Type 1 메타 트랙(track 0) 삭제                           │
  │    - trackIndex=0이 메타 트랙을 삭제                         │
  │                                                             │
  │ 7. SysEx 메시지 처리                                        │
  │    - sysex는 channel 속성이 없음 → 보존되어야 함            │
  │                                                             │
  │ 8. mido.MidiFile.save →  reload 바이트 수준 무결성          │
  │                                                             │
  │ 9. Type 2 MIDI 처리                                         │
  │    - Type 2 (비표준)가 들어오면 어떻게 되는지               │
  │                                                             │
  │ 10. 동일 채널 중복 program_change                           │
  │    - 한 채널에 여러 program_change → 전부 바뀌는지          │
  └─────────────────────────────────────────────────────────────┘
"""

import os
import sys
import copy
import tempfile
import traceback

import mido
import pretty_midi

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from app.services.midi_processor import (
    remap_original_tracks,
    _remap_type0,
    _remap_type1,
    DROP_INSTRUMENT_ID,
)


class FakeMapping:
    def __init__(self, trackIndex, targetInstrumentId):
        self.trackIndex = trackIndex
        self.targetInstrumentId = targetInstrumentId


class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.warnings = 0
        self.errors = []

    def ok(self, name):
        self.passed += 1
        print(f"  ✅ {name}")

    def fail(self, name, detail):
        self.failed += 1
        self.errors.append((name, detail))
        print(f"  🔴 {name}")
        for line in detail.split('\n'):
            print(f"     {line}")

    def warn(self, name, detail):
        self.warnings += 1
        print(f"  ⚠️  {name}: {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'=' * 65}")
        print(f"총 {total}개 | ✅ {self.passed} 통과 | 🔴 {self.failed} 실패 | ⚠️  {self.warnings} 경고")
        if self.errors:
            print("\n실패:")
            for n, d in self.errors:
                print(f"  - {n}: {d}")
        print('=' * 65)
        return self.failed == 0


R = TestResult()


def tmppath():
    f = tempfile.NamedTemporaryFile(delete=False, suffix=".mid")
    p = f.name
    f.close()
    return p


# ══════════════════════════════════════════════════════════════
# 1. 참조 오염 — Type 0 program_change→삭제 순서
# ══════════════════════════════════════════════════════════════
def test_reference_mutation_order():
    """
    _remap_type0 순서:
      1) program_change 변경  (L129-131: 전체 track 순회, 직접 msg.program = 값)
      2) 채널 삭제           (L137-149: msg.copy() 사용)

    만약 같은 채널에 변경+삭제가 동시에 오면?
    → channel_remap과 channels_to_delete에 동시에 같은 ch가 들어갈 수 있는지 확인.
    코드 L121-126: if DROP → channels_to_delete, else → channel_remap
    → 같은 ch는 한쪽에만 들어감. 하지만 같은 trackIndex가 두 번 매핑에 오면?
    """
    name = "REF-01: 같은 채널 삭제+변경 중복 매핑"
    try:
        mid = mido.MidiFile(type=0, ticks_per_beat=480)
        t = mido.MidiTrack()
        t.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        t.append(mido.Message('program_change', channel=0, program=0, time=0))
        t.append(mido.Message('note_on', channel=0, note=60, velocity=100, time=0))
        t.append(mido.Message('note_off', channel=0, note=60, velocity=0, time=480))
        t.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(t)

        p = tmppath()
        mid.save(p)

        # 같은 채널에 변경과 삭제 동시 → 마지막 매핑이 우선해야 하나?
        # 현재 코드: 두 번째 mapping이 channels_to_delete에 ch=0을 추가
        mappings = [
            FakeMapping(trackIndex=0, targetInstrumentId=25),    # 변경
            FakeMapping(trackIndex=0, targetInstrumentId=129),   # 삭제
        ]
        remap_original_tracks(p, mappings)

        reloaded = mido.MidiFile(p)
        os.unlink(p)

        # 마지막 매핑이 삭제 → ch0 메시지 없어야 함
        ch0_msgs = [msg for msg in reloaded.tracks[0]
                    if hasattr(msg, 'channel') and msg.channel == 0]
        if ch0_msgs:
            # 변경이 먼저 적용되고 삭제도 적용 → 삭제가 우선
            # 근데 코드 순서: channel_remap에 ch0=25 추가됨, 그 다음 channels_to_delete에 ch0 추가됨
            # channel_remap={0:25}, channels_to_delete={0}
            # prog_change 변경 실행 → 삭제 실행 → 삭제가 이겨야 함
            results_text = f"ch0 채널 메시지 {len(ch0_msgs)}개 잔존 — 삭제가 우선되어야 함"
            R.fail(name, results_text)
        else:
            R.ok(name)
    except Exception as e:
        R.fail(name, str(e))


# ══════════════════════════════════════════════════════════════
# 2. msg.copy() 속성 완전성 — 모든 속성이 복사되는지
# ══════════════════════════════════════════════════════════════
def test_msg_copy_completeness():
    """
    L145: msg = msg.copy(time=msg.time + accumulated_time)
    mido의 Message.copy()가 velocity, note, channel 등을 모두 보존하는지
    """
    name = "COPY-01: msg.copy() 속성 완전 보존"
    try:
        mid = mido.MidiFile(type=0, ticks_per_beat=480)
        t = mido.MidiTrack()
        t.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        t.append(mido.Message('program_change', channel=0, program=42, time=0))
        t.append(mido.Message('control_change', channel=0, control=7, value=100, time=0))
        t.append(mido.Message('control_change', channel=0, control=10, value=32, time=0))
        t.append(mido.Message('note_on', channel=0, note=72, velocity=110, time=0))
        t.append(mido.Message('pitchwheel', channel=0, pitch=4000, time=60))
        # ch1 삭제 대상
        t.append(mido.Message('note_on', channel=1, note=55, velocity=90, time=60))
        t.append(mido.Message('note_off', channel=1, note=55, velocity=0, time=120))
        # ch0 계속
        t.append(mido.Message('note_off', channel=0, note=72, velocity=0, time=120))
        t.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(t)

        p = tmppath()
        mid.save(p)

        mappings = [FakeMapping(trackIndex=1, targetInstrumentId=129)]
        remap_original_tracks(p, mappings)

        reloaded = mido.MidiFile(p)
        os.unlink(p)

        track = reloaded.tracks[0]

        # 검증: 각 메시지의 모든 속성
        errors = []
        for msg in track:
            if msg.type == 'program_change':
                if msg.program != 42:
                    errors.append(f"program_change.program: expected 42, got {msg.program}")
                if msg.channel != 0:
                    errors.append(f"program_change.channel: expected 0, got {msg.channel}")
            elif msg.type == 'control_change' and msg.control == 7:
                if msg.value != 100:
                    errors.append(f"CC7.value: expected 100, got {msg.value}")
            elif msg.type == 'control_change' and msg.control == 10:
                if msg.value != 32:
                    errors.append(f"CC10.value: expected 32, got {msg.value}")
            elif msg.type == 'note_on':
                if msg.note != 72:
                    errors.append(f"note_on.note: expected 72, got {msg.note}")
                if msg.velocity != 110:
                    errors.append(f"note_on.velocity: expected 110, got {msg.velocity}")
            elif msg.type == 'pitchwheel':
                if msg.pitch != 4000:
                    errors.append(f"pitchwheel.pitch: expected 4000, got {msg.pitch}")
            elif msg.type == 'note_off':
                if msg.note != 72:
                    errors.append(f"note_off.note: expected 72, got {msg.note}")

        if errors:
            R.fail(name, '\n'.join(errors))
        else:
            R.ok(name)
    except Exception as e:
        R.fail(name, str(e))


# ══════════════════════════════════════════════════════════════
# 3. MetaMessage 속성 보존 (tempo, text, key_signature 등)
# ══════════════════════════════════════════════════════════════
def test_meta_message_preservation():
    name = "META-01: MetaMessage 속성 완전 보존"
    try:
        mid = mido.MidiFile(type=0, ticks_per_beat=480)
        t = mido.MidiTrack()
        t.append(mido.MetaMessage('set_tempo', tempo=600000, time=0))
        t.append(mido.MetaMessage('time_signature', numerator=3, denominator=4, time=0))
        t.append(mido.MetaMessage('key_signature', key='Dm', time=0))
        t.append(mido.MetaMessage('track_name', name='My Song', time=0))
        t.append(mido.MetaMessage('text', text='Copyright 2024', time=0))
        t.append(mido.MetaMessage('marker', text='Verse', time=0))
        t.append(mido.Message('note_on', channel=0, note=60, velocity=100, time=0))
        t.append(mido.Message('note_on', channel=1, note=55, velocity=90, time=120))  # 삭제 대상
        t.append(mido.MetaMessage('marker', text='Chorus', time=120))  # 삭제 메시지 사이의 메타
        t.append(mido.Message('note_off', channel=1, note=55, velocity=0, time=120))  # 삭제 대상
        t.append(mido.Message('note_off', channel=0, note=60, velocity=0, time=120))
        t.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(t)

        p = tmppath()
        mid.save(p)

        mappings = [FakeMapping(trackIndex=1, targetInstrumentId=129)]
        remap_original_tracks(p, mappings)

        reloaded = mido.MidiFile(p)
        os.unlink(p)

        track = reloaded.tracks[0]
        errors = []

        # 메타 메시지 수집
        meta_msgs = [msg for msg in track if msg.is_meta]
        meta_types = [msg.type for msg in meta_msgs]

        expected_types = ['set_tempo', 'time_signature', 'key_signature',
                          'track_name', 'text', 'marker', 'marker', 'end_of_track']
        if meta_types != expected_types:
            errors.append(f"meta types: expected {expected_types}, got {meta_types}")

        # 각 속성 검증
        for msg in meta_msgs:
            if msg.type == 'set_tempo' and msg.tempo != 600000:
                errors.append(f"tempo: expected 600000, got {msg.tempo}")
            elif msg.type == 'time_signature':
                if msg.numerator != 3:
                    errors.append(f"time_sig numerator: expected 3, got {msg.numerator}")
                if msg.denominator != 4:
                    errors.append(f"time_sig denominator: expected 4, got {msg.denominator}")
            elif msg.type == 'key_signature' and msg.key != 'Dm':
                errors.append(f"key_sig: expected Dm, got {msg.key}")
            elif msg.type == 'track_name' and msg.name != 'My Song':
                errors.append(f"track_name: expected 'My Song', got '{msg.name}'")
            elif msg.type == 'text' and msg.text != 'Copyright 2024':
                errors.append(f"text: expected 'Copyright 2024', got '{msg.text}'")

        # 마커 텍스트 검증
        markers = [msg for msg in meta_msgs if msg.type == 'marker']
        marker_texts = [m.text for m in markers]
        if marker_texts != ['Verse', 'Chorus']:
            errors.append(f"markers: expected ['Verse','Chorus'], got {marker_texts}")

        if errors:
            R.fail(name, '\n'.join(errors))
        else:
            R.ok(name)
    except Exception as e:
        R.fail(name, str(e))


# ══════════════════════════════════════════════════════════════
# 4. 메시지 순서 보존
# ══════════════════════════════════════════════════════════════
def test_message_order_preserved():
    name = "ORDER-01: 채널 삭제 후 생존 메시지 상대 순서 보존"
    try:
        mid = mido.MidiFile(type=0, ticks_per_beat=480)
        t = mido.MidiTrack()
        t.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        # 번호가 매겨진 순서대로 추가
        t.append(mido.Message('note_on',  channel=0, note=60, velocity=100, time=0))    # idx 0
        t.append(mido.Message('note_on',  channel=1, note=55, velocity=90, time=10))    # 삭제
        t.append(mido.Message('note_on',  channel=0, note=64, velocity=100, time=20))   # idx 1
        t.append(mido.Message('note_on',  channel=1, note=59, velocity=90, time=30))    # 삭제
        t.append(mido.Message('note_off', channel=0, note=60, velocity=0, time=40))     # idx 2
        t.append(mido.Message('note_on',  channel=1, note=62, velocity=90, time=50))    # 삭제
        t.append(mido.Message('note_off', channel=0, note=64, velocity=0, time=60))     # idx 3
        t.append(mido.Message('note_off', channel=1, note=55, velocity=0, time=10))     # 삭제
        t.append(mido.Message('note_off', channel=1, note=59, velocity=0, time=10))     # 삭제
        t.append(mido.Message('note_off', channel=1, note=62, velocity=0, time=10))     # 삭제
        t.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(t)

        # ch0 메시지의 원래 순서 기록
        original_ch0_sequence = []
        for msg in t:
            if hasattr(msg, 'channel') and msg.channel == 0:
                original_ch0_sequence.append((msg.type, msg.note))

        p = tmppath()
        mid.save(p)

        mappings = [FakeMapping(trackIndex=1, targetInstrumentId=129)]
        remap_original_tracks(p, mappings)

        reloaded = mido.MidiFile(p)
        os.unlink(p)

        new_ch0_sequence = []
        for msg in reloaded.tracks[0]:
            if hasattr(msg, 'channel') and msg.channel == 0:
                new_ch0_sequence.append((msg.type, msg.note))

        if original_ch0_sequence == new_ch0_sequence:
            R.ok(name)
        else:
            R.fail(name, f"순서 변경!\n  before: {original_ch0_sequence}\n  after:  {new_ch0_sequence}")
    except Exception as e:
        R.fail(name, str(e))


# ══════════════════════════════════════════════════════════════
# 5. program_change 없는 채널 (기본 prog=0)
# ══════════════════════════════════════════════════════════════
def test_no_program_change_channel():
    name = "EDGE-01: program_change 없는 채널 재매핑"
    try:
        mid = mido.MidiFile(type=0, ticks_per_beat=480)
        t = mido.MidiTrack()
        t.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        # ch0: program_change 없이 바로 노트
        t.append(mido.Message('note_on', channel=0, note=60, velocity=100, time=0))
        t.append(mido.Message('note_off', channel=0, note=60, velocity=0, time=480))
        t.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(t)

        p = tmppath()
        mid.save(p)

        # ch0을 prog=25로 변경 → program_change가 없으므로 아무 변경도 없어야 함
        mappings = [FakeMapping(trackIndex=0, targetInstrumentId=25)]
        remap_original_tracks(p, mappings)

        reloaded = mido.MidiFile(p)
        os.unlink(p)

        # program_change가 추가되지 않았는지 확인
        prog_msgs = [msg for msg in reloaded.tracks[0] if msg.type == 'program_change']
        if len(prog_msgs) == 0:
            R.warn(name, "program_change 미삽입 — 코드는 기존 prog_change만 수정하므로 정상이지만, "
                        "pretty_midi는 기본 prog=0을 사용하게 됨")
        else:
            R.ok(name)
    except Exception as e:
        R.fail(name, str(e))


# ══════════════════════════════════════════════════════════════
# 6. Type 1 메타 트랙 삭제
# ══════════════════════════════════════════════════════════════
def test_type1_meta_track_deletion():
    name = "EDGE-02: Type 1 메타 트랙(track 0) 삭제"
    try:
        mid = mido.MidiFile(type=1, ticks_per_beat=480)
        # Track 0: meta
        meta = mido.MidiTrack()
        meta.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        meta.append(mido.MetaMessage('end_of_track', time=1920))
        mid.tracks.append(meta)
        # Track 1: Piano
        t1 = mido.MidiTrack()
        t1.append(mido.Message('program_change', channel=0, program=0, time=0))
        t1.append(mido.Message('note_on', channel=0, note=60, velocity=100, time=0))
        t1.append(mido.Message('note_off', channel=0, note=60, velocity=0, time=480))
        t1.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(t1)

        p = tmppath()
        mid.save(p)

        # track 0 (메타) 삭제 → 코드가 허용함 (방어 없음)
        mappings = [FakeMapping(trackIndex=0, targetInstrumentId=129)]
        remap_original_tracks(p, mappings)

        reloaded = mido.MidiFile(p)
        os.unlink(p)

        # 메타 트랙이 삭제되면 tempo 정보 손실 → pretty_midi 해석 오류 가능
        has_tempo = False
        for track in reloaded.tracks:
            for msg in track:
                if msg.type == 'set_tempo':
                    has_tempo = True
                    break

        if not has_tempo:
            R.warn(name, "메타 트랙 삭제됨 → set_tempo 손실. "
                        "메인 서버가 track 0을 삭제 대상으로 보내면 tempo 정보 소실 가능. "
                        "방어 코드 권장.")
        else:
            R.ok(name)
    except Exception as e:
        R.fail(name, str(e))


# ══════════════════════════════════════════════════════════════
# 7. SysEx 메시지 보존
# ══════════════════════════════════════════════════════════════
def test_sysex_preservation():
    name = "SYSEX-01: SysEx 메시지 보존 (채널 삭제 시)"
    try:
        mid = mido.MidiFile(type=0, ticks_per_beat=480)
        t = mido.MidiTrack()
        t.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        t.append(mido.Message('program_change', channel=0, program=0, time=0))
        # SysEx GM Reset
        t.append(mido.Message('sysex', data=[0x7E, 0x7F, 0x09, 0x01], time=0))
        t.append(mido.Message('note_on', channel=0, note=60, velocity=100, time=0))
        t.append(mido.Message('note_on', channel=1, note=55, velocity=90, time=120))
        t.append(mido.Message('note_off', channel=0, note=60, velocity=0, time=120))
        t.append(mido.Message('note_off', channel=1, note=55, velocity=0, time=120))
        t.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(t)

        p = tmppath()
        mid.save(p)

        mappings = [FakeMapping(trackIndex=1, targetInstrumentId=129)]
        remap_original_tracks(p, mappings)

        reloaded = mido.MidiFile(p)
        os.unlink(p)

        # SysEx가 보존되었는지
        sysex_msgs = [msg for msg in reloaded.tracks[0] if msg.type == 'sysex']
        if len(sysex_msgs) == 1:
            if tuple(sysex_msgs[0].data) == (0x7E, 0x7F, 0x09, 0x01):
                R.ok(name)
            else:
                R.fail(name, f"SysEx data 변경: {sysex_msgs[0].data}")
        else:
            R.fail(name, f"SysEx 수: expected 1, got {len(sysex_msgs)}")
    except Exception as e:
        R.fail(name, str(e))


# ══════════════════════════════════════════════════════════════
# 8. ticks_per_beat 보존
# ══════════════════════════════════════════════════════════════
def test_ticks_per_beat_preserved():
    name = "HEADER-01: ticks_per_beat 보존"
    try:
        for tpb in [96, 120, 240, 480, 960]:
            mid = mido.MidiFile(type=1, ticks_per_beat=tpb)
            meta = mido.MidiTrack()
            meta.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
            meta.append(mido.MetaMessage('end_of_track', time=0))
            mid.tracks.append(meta)
            t1 = mido.MidiTrack()
            t1.append(mido.Message('program_change', channel=0, program=0, time=0))
            t1.append(mido.Message('note_on', channel=0, note=60, velocity=100, time=0))
            t1.append(mido.Message('note_off', channel=0, note=60, velocity=0, time=tpb))
            t1.append(mido.MetaMessage('end_of_track', time=0))
            mid.tracks.append(t1)

            p = tmppath()
            mid.save(p)

            mappings = [FakeMapping(trackIndex=1, targetInstrumentId=25)]
            remap_original_tracks(p, mappings)

            reloaded = mido.MidiFile(p)
            os.unlink(p)

            if reloaded.ticks_per_beat != tpb:
                R.fail(name, f"tpb={tpb} → {reloaded.ticks_per_beat}")
                return

        R.ok(name)
    except Exception as e:
        R.fail(name, str(e))


# ══════════════════════════════════════════════════════════════
# 9. MIDI type 보존 (Type 0 → 0, Type 1 → 1)
# ══════════════════════════════════════════════════════════════
def test_midi_type_preserved():
    name = "HEADER-02: MIDI type 보존"
    try:
        for midi_type in [0, 1]:
            mid = mido.MidiFile(type=midi_type, ticks_per_beat=480)
            t = mido.MidiTrack()
            t.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
            t.append(mido.Message('program_change', channel=0, program=0, time=0))
            t.append(mido.Message('note_on', channel=0, note=60, velocity=100, time=0))
            t.append(mido.Message('note_off', channel=0, note=60, velocity=0, time=480))
            t.append(mido.MetaMessage('end_of_track', time=0))
            mid.tracks.append(t)

            p = tmppath()
            mid.save(p)

            mappings = [FakeMapping(trackIndex=0, targetInstrumentId=25)]
            remap_original_tracks(p, mappings)

            reloaded = mido.MidiFile(p)
            os.unlink(p)

            if reloaded.type != midi_type:
                R.fail(name, f"type {midi_type} → {reloaded.type}")
                return

        R.ok(name)
    except Exception as e:
        R.fail(name, str(e))


# ══════════════════════════════════════════════════════════════
# 10. 동일 채널 다중 program_change
# ══════════════════════════════════════════════════════════════
def test_multiple_program_change_same_channel():
    name = "PROG-01: 동일 채널 다중 program_change 전부 변경"
    try:
        mid = mido.MidiFile(type=0, ticks_per_beat=480)
        t = mido.MidiTrack()
        t.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        # ch0에 두 번 program_change (악기 전환)
        t.append(mido.Message('program_change', channel=0, program=0, time=0))
        t.append(mido.Message('note_on', channel=0, note=60, velocity=100, time=0))
        t.append(mido.Message('note_off', channel=0, note=60, velocity=0, time=480))
        t.append(mido.Message('program_change', channel=0, program=48, time=0))  # String 전환
        t.append(mido.Message('note_on', channel=0, note=64, velocity=100, time=0))
        t.append(mido.Message('note_off', channel=0, note=64, velocity=0, time=480))
        t.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(t)

        p = tmppath()
        mid.save(p)

        # ch0 → prog=25
        mappings = [FakeMapping(trackIndex=0, targetInstrumentId=25)]
        remap_original_tracks(p, mappings)

        reloaded = mido.MidiFile(p)
        os.unlink(p)

        progs = [msg.program for msg in reloaded.tracks[0] if msg.type == 'program_change']
        # 모든 program_change가 25로 변경되어야 함
        if progs == [25, 25]:
            R.ok(name)
        else:
            R.fail(name, f"expected [25, 25], got {progs}")
    except Exception as e:
        R.fail(name, str(e))


# ══════════════════════════════════════════════════════════════
# 11. Type 1 — 재매핑 대상이 아닌 트랙의 무결성
# ══════════════════════════════════════════════════════════════
def test_type1_untouched_tracks_integrity():
    name = "T1-INTEG-01: 재매핑 대상이 아닌 트랙 완전 무결성"
    try:
        mid = mido.MidiFile(type=1, ticks_per_beat=480)

        meta = mido.MidiTrack()
        meta.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        meta.append(mido.MetaMessage('end_of_track', time=1920))
        mid.tracks.append(meta)

        t1 = mido.MidiTrack()
        t1.append(mido.Message('program_change', channel=0, program=0, time=0))
        t1.append(mido.Message('note_on', channel=0, note=60, velocity=100, time=0))
        t1.append(mido.Message('note_off', channel=0, note=60, velocity=0, time=480))
        t1.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(t1)

        t2 = mido.MidiTrack()
        t2.append(mido.Message('program_change', channel=1, program=40, time=0))
        t2.append(mido.Message('control_change', channel=1, control=7, value=80, time=0))
        t2.append(mido.Message('note_on', channel=1, note=55, velocity=90, time=0))
        t2.append(mido.Message('note_off', channel=1, note=55, velocity=0, time=480))
        t2.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(t2)

        # Track 2 직렬화 (변경 전)
        p = tmppath()
        mid.save(p)

        p2 = tmppath()
        mid.save(p2)
        before_t2_bytes = open(p2, 'rb').read()

        # Track 1만 변경
        mappings = [FakeMapping(trackIndex=1, targetInstrumentId=25)]
        remap_original_tracks(p, mappings)

        reloaded = mido.MidiFile(p)
        os.unlink(p)

        # Track 2는 변경되지 않아야 함
        track2 = reloaded.tracks[2]
        errors = []
        expected = [
            ('program_change', {'channel': 1, 'program': 40}),
            ('control_change', {'channel': 1, 'control': 7, 'value': 80}),
            ('note_on',        {'channel': 1, 'note': 55, 'velocity': 90}),
            ('note_off',       {'channel': 1, 'note': 55, 'velocity': 0}),
            ('end_of_track',   {}),
        ]

        for i, (exp_type, exp_attrs) in enumerate(expected):
            msg = track2[i]
            if msg.type != exp_type:
                errors.append(f"idx {i}: type expected {exp_type}, got {msg.type}")
            for k, v in exp_attrs.items():
                actual = getattr(msg, k, None)
                if actual != v:
                    errors.append(f"idx {i}: {k} expected {v}, got {actual}")

        os.unlink(p2)

        if errors:
            R.fail(name, '\n'.join(errors))
        else:
            R.ok(name)
    except Exception as e:
        R.fail(name, str(e))


# ══════════════════════════════════════════════════════════════
# 12. Type 0 — 삭제 대상 채널의 program_change가 먼저 변경되는 문제
# ══════════════════════════════════════════════════════════════
def test_type0_delete_channel_prog_change_mutation():
    """
    _remap_type0 코드 순서:
      L128-131: program_change 변경 (전체 track 순회)
      L137-149: 채널 삭제

    만약 mappings에 ch=1 삭제가 있고, ch=1의 program_change도 변경하는
    별도 매핑은 없지만, 코드가 channel_remap에 없어도 만진다면?
    → L130: msg.channel in channel_remap 체크로 안전.

    하지만 program_change 변경 후 삭제가 실행되므로,
    삭제 대상 채널의 program은 이미 (불필요하게) 변경될 수 있다.
    실질적으로는 삭제되므로 문제없지만, 원본 track 객체가 오염됨.
    """
    name = "MUTATION-01: 삭제 대상 채널 program_change 무의미한 변경 여부"
    try:
        mid = mido.MidiFile(type=0, ticks_per_beat=480)
        t = mido.MidiTrack()
        t.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        t.append(mido.Message('program_change', channel=0, program=0, time=0))
        t.append(mido.Message('program_change', channel=1, program=40, time=0))
        t.append(mido.Message('note_on', channel=0, note=60, velocity=100, time=0))
        t.append(mido.Message('note_on', channel=1, note=55, velocity=90, time=120))
        t.append(mido.Message('note_off', channel=0, note=60, velocity=0, time=120))
        t.append(mido.Message('note_off', channel=1, note=55, velocity=0, time=120))
        t.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(t)

        p = tmppath()
        mid.save(p)

        # ch0 변경, ch1 삭제 → ch1의 program은 변경되지 않아야 함(channel_remap에 없으니까)
        mappings = [
            FakeMapping(trackIndex=0, targetInstrumentId=25),
            FakeMapping(trackIndex=1, targetInstrumentId=129),
        ]
        remap_original_tracks(p, mappings)

        reloaded = mido.MidiFile(p)
        os.unlink(p)

        # ch1 완전 삭제 확인
        ch1_msgs = [msg for msg in reloaded.tracks[0]
                    if hasattr(msg, 'channel') and msg.channel == 1]
        if ch1_msgs:
            R.fail(name, f"ch1 메시지 {len(ch1_msgs)}개 잔존")
            return

        # ch0 program = 25 확인
        ch0_progs = [msg.program for msg in reloaded.tracks[0]
                     if msg.type == 'program_change' and msg.channel == 0]
        if ch0_progs == [25]:
            R.ok(name)
        else:
            R.fail(name, f"ch0 program expected [25], got {ch0_progs}")
    except Exception as e:
        R.fail(name, str(e))


# ══════════════════════════════════════════════════════════════
# 13. Type 0 — end_of_track 보존
# ══════════════════════════════════════════════════════════════
def test_end_of_track_preserved():
    name = "META-02: end_of_track 메시지 보존"
    try:
        mid = mido.MidiFile(type=0, ticks_per_beat=480)
        t = mido.MidiTrack()
        t.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        t.append(mido.Message('note_on', channel=0, note=60, velocity=100, time=0))
        t.append(mido.Message('note_on', channel=1, note=55, velocity=90, time=120))
        t.append(mido.Message('note_off', channel=0, note=60, velocity=0, time=120))
        t.append(mido.Message('note_off', channel=1, note=55, velocity=0, time=120))
        t.append(mido.MetaMessage('end_of_track', time=480))  # 끝에 여유 있는 eot
        mid.tracks.append(t)

        original_eot_time = 480

        p = tmppath()
        mid.save(p)

        mappings = [FakeMapping(trackIndex=1, targetInstrumentId=129)]
        remap_original_tracks(p, mappings)

        reloaded = mido.MidiFile(p)
        os.unlink(p)

        eot_msgs = [msg for msg in reloaded.tracks[0] if msg.type == 'end_of_track']
        if len(eot_msgs) != 1:
            R.fail(name, f"end_of_track 수: expected 1, got {len(eot_msgs)}")
            return

        # eot time이 삭제된 메시지의 delta를 흡수했는지 확인
        # 원본: ...ch1 note_off(120)...eot(480)
        # ch1 삭제 후: ch0 note_off가 120+120=240 흡수, eot가 120+480=600 흡수?
        # 확인 필요
        total_dur = sum(msg.time for msg in reloaded.tracks[0])
        original_dur = sum(msg.time for msg in mid.tracks[0])
        if total_dur == original_dur:
            R.ok(name)
        else:
            R.fail(name, f"총 duration: {original_dur} → {total_dur}")
    except Exception as e:
        R.fail(name, str(e))


# ══════════════════════════════════════════════════════════════
# 14. Type 2 MIDI 처리
# ══════════════════════════════════════════════════════════════
def test_type2_midi():
    name = "EDGE-03: Type 2 MIDI 처리"
    try:
        mid = mido.MidiFile(type=2, ticks_per_beat=480)
        t = mido.MidiTrack()
        t.append(mido.Message('note_on', channel=0, note=60, velocity=100, time=0))
        t.append(mido.Message('note_off', channel=0, note=60, velocity=0, time=480))
        t.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(t)

        p = tmppath()
        mid.save(p)

        # Type 2는 _remap_type1로 처리됨 (else 분기)
        mappings = [FakeMapping(trackIndex=0, targetInstrumentId=25)]
        remap_original_tracks(p, mappings)

        reloaded = mido.MidiFile(p)
        os.unlink(p)

        R.warn(name, f"Type 2 MIDI는 Type 1로 처리됨 (mid.type={reloaded.type}). "
                      "실제 Type 2 사용은 극히 드물어 영향 없음.")
    except Exception as e:
        R.fail(name, str(e))


# ══════════════════════════════════════════════════════════════
# 15. 바이트 수준 라운드트립 — 변경 없는 save/load
# ══════════════════════════════════════════════════════════════
def test_roundtrip_no_change():
    name = "RT-01: mappings 비어있을 때 파일 바이트 동일"
    try:
        mid = mido.MidiFile(type=1, ticks_per_beat=480)
        meta = mido.MidiTrack()
        meta.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        meta.append(mido.MetaMessage('end_of_track', time=1920))
        mid.tracks.append(meta)

        t1 = mido.MidiTrack()
        t1.append(mido.Message('program_change', channel=0, program=0, time=0))
        t1.append(mido.Message('note_on', channel=0, note=60, velocity=100, time=0))
        t1.append(mido.Message('note_off', channel=0, note=60, velocity=0, time=480))
        t1.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(t1)

        p = tmppath()
        mid.save(p)
        original = open(p, 'rb').read()

        remap_original_tracks(p, [])  # 빈 mappings → return 즉시
        after = open(p, 'rb').read()
        os.unlink(p)

        if original == after:
            R.ok(name)
        else:
            R.fail(name, f"바이트 변경: {len(original)} → {len(after)}")
    except Exception as e:
        R.fail(name, str(e))


# ══════════════════════════════════════════════════════════════
# 16. velocity 0 note_on vs note_off 구분
# ══════════════════════════════════════════════════════════════
def test_velocity_zero_note_on():
    """일부 MIDI에서는 note_off 대신 velocity=0 note_on을 사용"""
    name = "COMPAT-01: velocity=0 note_on (note_off 대체) 보존"
    try:
        mid = mido.MidiFile(type=0, ticks_per_beat=480)
        t = mido.MidiTrack()
        t.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        t.append(mido.Message('note_on', channel=0, note=60, velocity=100, time=0))
        t.append(mido.Message('note_on', channel=1, note=55, velocity=90, time=120))
        # velocity=0 note_on = note_off
        t.append(mido.Message('note_on', channel=0, note=60, velocity=0, time=120))
        t.append(mido.Message('note_on', channel=1, note=55, velocity=0, time=120))
        t.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(t)

        p = tmppath()
        mid.save(p)

        mappings = [FakeMapping(trackIndex=1, targetInstrumentId=129)]
        remap_original_tracks(p, mappings)

        reloaded = mido.MidiFile(p)
        os.unlink(p)

        # ch0의 velocity=0 note_on이 보존되어야 함
        ch0_notes = [msg for msg in reloaded.tracks[0]
                     if msg.type == 'note_on' and hasattr(msg, 'channel') and msg.channel == 0]
        if len(ch0_notes) != 2:
            R.fail(name, f"ch0 note_on 수: expected 2, got {len(ch0_notes)}")
            return

        # 첫 번째: vel=100, 두 번째: vel=0
        if ch0_notes[0].velocity == 100 and ch0_notes[1].velocity == 0:
            R.ok(name)
        else:
            R.fail(name, f"velocity: {ch0_notes[0].velocity}, {ch0_notes[1].velocity}")
    except Exception as e:
        R.fail(name, str(e))


# ══════════════════════════════════════════════════════════════
# 17. pretty_midi 전후 비교 — 노트별 정밀 검증
# ══════════════════════════════════════════════════════════════
def test_pretty_midi_note_by_note_integrity():
    name = "COMPAT-02: pretty_midi 노트별 start/end/velocity/pitch 정밀 비교"
    try:
        mid = mido.MidiFile(type=1, ticks_per_beat=480)

        meta = mido.MidiTrack()
        meta.append(mido.MetaMessage('set_tempo', tempo=500000, time=0))
        meta.append(mido.MetaMessage('end_of_track', time=3840))
        mid.tracks.append(meta)

        t1 = mido.MidiTrack()
        t1.append(mido.Message('program_change', channel=0, program=0, time=0))
        notes_ch0 = [(60, 100, 0, 240), (62, 95, 240, 240), (64, 110, 0, 480)]
        for note, vel, offset, dur in notes_ch0:
            t1.append(mido.Message('note_on', channel=0, note=note, velocity=vel, time=offset))
            t1.append(mido.Message('note_off', channel=0, note=note, velocity=0, time=dur))
        t1.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(t1)

        t2 = mido.MidiTrack()
        t2.append(mido.Message('program_change', channel=1, program=40, time=0))
        notes_ch1 = [(55, 85, 0, 480), (57, 90, 0, 240)]
        for note, vel, offset, dur in notes_ch1:
            t2.append(mido.Message('note_on', channel=1, note=note, velocity=vel, time=offset))
            t2.append(mido.Message('note_off', channel=1, note=note, velocity=0, time=dur))
        t2.append(mido.MetaMessage('end_of_track', time=0))
        mid.tracks.append(t2)

        p = tmppath()
        mid.save(p)

        # 변경 전 pretty_midi 노트 기록
        pm_before = pretty_midi.PrettyMIDI(p)
        before_notes = {}
        for inst in pm_before.instruments:
            before_notes[inst.program] = [
                (n.pitch, n.velocity, round(n.start, 6), round(n.end, 6))
                for n in sorted(inst.notes, key=lambda x: x.start)
            ]

        # Track 1 변경, Track 2 삭제
        mappings = [
            FakeMapping(trackIndex=1, targetInstrumentId=25),
            FakeMapping(trackIndex=2, targetInstrumentId=129),
        ]
        remap_original_tracks(p, mappings)

        pm_after = pretty_midi.PrettyMIDI(p)
        os.unlink(p)

        after_notes = {}
        for inst in pm_after.instruments:
            after_notes[inst.program] = [
                (n.pitch, n.velocity, round(n.start, 6), round(n.end, 6))
                for n in sorted(inst.notes, key=lambda x: x.start)
            ]

        errors = []

        # Piano(0) → Guitar(25): 노트 동일해야 함
        if before_notes.get(0) != after_notes.get(25):
            errors.append(f"Piano→Guitar 노트 불일치\n"
                          f"  before(prog=0):  {before_notes.get(0)}\n"
                          f"  after(prog=25):  {after_notes.get(25)}")

        # Violin(40) 완전 삭제
        if 40 in after_notes:
            errors.append(f"Violin(40) 잔존: {after_notes[40]}")

        if errors:
            R.fail(name, '\n'.join(errors))
        else:
            R.ok(name)
    except Exception as e:
        R.fail(name, f"{e}\n{traceback.format_exc()}")


# ══════════════════════════════════════════════════════════════
# 실행
# ══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("=" * 65)
    print("🔬 MIDI 프로세서 데이터 결손/오염 벡터 종합 분석")
    print("=" * 65)

    print("\n── 참조 오염 / 실행 순서 ──")
    test_reference_mutation_order()
    test_type0_delete_channel_prog_change_mutation()

    print("\n── msg.copy() 속성 완전성 ──")
    test_msg_copy_completeness()
    test_velocity_zero_note_on()

    print("\n── MetaMessage 보존 ──")
    test_meta_message_preservation()
    test_end_of_track_preserved()

    print("\n── 메시지 순서 보존 ──")
    test_message_order_preserved()

    print("\n── 파일 헤더 보존 ──")
    test_ticks_per_beat_preserved()
    test_midi_type_preserved()

    print("\n── 특수 메시지 타입 ──")
    test_sysex_preservation()
    test_multiple_program_change_same_channel()

    print("\n── 무결성 ──")
    test_type1_untouched_tracks_integrity()
    test_roundtrip_no_change()

    print("\n── pretty_midi 호환성 ──")
    test_pretty_midi_note_by_note_integrity()

    print("\n── 에지케이스 ──")
    test_no_program_change_channel()
    test_type1_meta_track_deletion()
    test_type2_midi()

    all_pass = R.summary()
    sys.exit(0 if all_pass else 1)
