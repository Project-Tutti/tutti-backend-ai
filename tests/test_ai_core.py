"""test_ai_core.py — ai_core 패키지 단위 테스트.

GPU/모델 없이 실행 가능한 테스트만 포함합니다.
"""
import pytest
import sys
import os

# 프로젝트 루트를 path에 추가
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import torch  # noqa: F401
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ──────────────────────────────────────────────
# 1. constants 모듈 테스트
# ──────────────────────────────────────────────
class TestConstants:
    def test_instrument_groups_all_have_required_fields(self):
        from ai_core.constants import INSTRUMENT_GROUPS
        for name, cfg in INSTRUMENT_GROUPS.items():
            assert "representative" in cfg, f"{name} missing 'representative'"
            assert "is_drum" in cfg, f"{name} missing 'is_drum'"
            assert "pitch_min" in cfg, f"{name} missing 'pitch_min'"
            assert "pitch_max" in cfg, f"{name} missing 'pitch_max'"
            assert cfg["pitch_min"] < cfg["pitch_max"], f"{name}: pitch_min >= pitch_max"

    def test_program_to_rep_covers_0_to_128(self):
        from ai_core.constants import PROGRAM_TO_REP, DROP_SET
        for p in range(129):
            if p not in DROP_SET:
                assert p in PROGRAM_TO_REP, f"program {p} not mapped (and not in DROP_SET)"

    def test_all_target_names_length(self):
        from ai_core.constants import ALL_TARGET_NAMES, INSTRUMENT_GROUPS
        assert len(ALL_TARGET_NAMES) == len(INSTRUMENT_GROUPS)


# ──────────────────────────────────────────────
# 2. vocab 모듈 테스트
# ──────────────────────────────────────────────
class TestVocab:
    def test_build_v5_vocab_default_size(self):
        from ai_core.vocab import build_v5_vocab
        vocab = build_v5_vocab()
        assert len(vocab) == 682, f"Expected 682 tokens, got {len(vocab)}"

    def test_vocab_has_essential_tokens(self):
        from ai_core.vocab import build_v5_vocab
        vocab = build_v5_vocab()
        essential = ["PAD", "BOS", "EOS", "PIECE_START", "PIECE_END",
                     "BAR_START", "BAR_END", "GENRE_POP", "KEY_NONE"]
        for tok in essential:
            assert tok in vocab, f"Missing essential token: {tok}"

    def test_vocab_no_duplicate_ids(self):
        from ai_core.vocab import build_v5_vocab
        vocab = build_v5_vocab()
        ids = list(vocab.values())
        assert len(ids) == len(set(ids)), "Duplicate token IDs found"


# ──────────────────────────────────────────────
# 3. arrangement 모듈 테스트
# ──────────────────────────────────────────────
@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestArrangement:
    def test_resolve_target_valid(self):
        from ai_core.arrangement import resolve_target
        assert resolve_target(0) == "keyboard"
        assert resolve_target(33) == "bass"
        assert resolve_target(128) == "drum"
        assert resolve_target(40) == "violin"

    def test_resolve_target_invalid(self):
        from ai_core.arrangement import resolve_target
        with pytest.raises(ValueError):
            resolve_target(999)

    def test_resolve_target_drop_set_members(self):
        from ai_core.constants import DROP_SET
        from ai_core.arrangement import resolve_target
        for p in DROP_SET:
            # DROP_SET 멤버는 PROGRAM_TO_REP에 없으므로 ValueError 발생
            with pytest.raises(ValueError):
                resolve_target(p)


# ──────────────────────────────────────────────
# 4. postprocess 모듈 테스트
# ──────────────────────────────────────────────
class TestPostprocess:
    def test_pitch_clipping(self):
        from ai_core.postprocess import postprocess
        notes = [
            {"start": 0.0, "end": 1.0, "pitch": 10, "velocity": 80},
            {"start": 1.0, "end": 2.0, "pitch": 60, "velocity": 80},
            {"start": 2.0, "end": 3.0, "pitch": 120, "velocity": 80},
        ]
        result = postprocess(notes, pitch_min=50, pitch_max=100)
        assert len(result) == 1
        assert result[0]["pitch"] == 60

    def test_short_note_removal(self):
        from ai_core.postprocess import postprocess
        notes = [
            {"start": 0.0, "end": 0.02, "pitch": 60, "velocity": 80},  # too short
            {"start": 1.0, "end": 2.0, "pitch": 60, "velocity": 80},
        ]
        result = postprocess(notes, pitch_min=0, pitch_max=127)
        assert len(result) == 1

    def test_monophonic_enforcement(self):
        from ai_core.postprocess import postprocess
        notes = [
            {"start": 0.0, "end": 2.0, "pitch": 60, "velocity": 80},
            {"start": 1.0, "end": 3.0, "pitch": 65, "velocity": 80},
        ]
        result = postprocess(notes, pitch_min=0, pitch_max=127, target_name="bass")
        # bass is monophonic, so overlap should be resolved
        assert result[0]["end"] <= result[1]["start"]


# ──────────────────────────────────────────────
# 5. facade (inference.py) 하위 호환성 테스트
# ──────────────────────────────────────────────
@pytest.mark.skipif(not HAS_TORCH, reason="torch not installed")
class TestFacade:
    def test_facade_exports_run_arrangement(self):
        from app.services.inference import run_arrangement
        assert callable(run_arrangement)

    def test_facade_exports_resolve_target(self):
        from app.services.inference import resolve_target
        assert callable(resolve_target)

    def test_facade_exports_build_v5_vocab(self):
        from app.services.inference import build_v5_vocab
        assert callable(build_v5_vocab)

    def test_facade_exports_load_model(self):
        from app.services.inference import load_model
        assert callable(load_model)

    def test_facade_exports_constants(self):
        from app.services.inference import INSTRUMENT_GROUPS, PROGRAM_TO_REP
        assert isinstance(INSTRUMENT_GROUPS, dict)
        assert isinstance(PROGRAM_TO_REP, dict)

    def test_facade_exports_save_midi(self):
        from app.services.inference import save_midi
        assert callable(save_midi)


# ──────────────────────────────────────────────
# 6. metrics 모듈 테스트
# ──────────────────────────────────────────────
class TestMetrics:
    def test_compute_basic_quality_metrics_with_notes(self, tmp_path):
        import mido
        from ai_core.metrics import compute_basic_quality_metrics

        # 테스트용 MIDI 파일 생성
        mid = mido.MidiFile()
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=500000, time=0))
        # C4(60), E4(64), G4(67) 각각 480tick 길이
        for pitch in [60, 64, 67]:
            track.append(mido.Message("note_on", note=pitch, velocity=80, time=0))
            track.append(mido.Message("note_off", note=pitch, velocity=0, time=480))

        midi_path = str(tmp_path / "test.mid")
        mid.save(midi_path)

        result = compute_basic_quality_metrics(midi_path)
        assert result["note_count"] == 3
        assert result["pitch_min"] == 60
        assert result["pitch_max"] == 67
        assert result["pitch_range"] == 7

    def test_compute_basic_quality_metrics_empty_file(self, tmp_path):
        import mido
        from ai_core.metrics import compute_basic_quality_metrics

        # 노트 없는 MIDI
        mid = mido.MidiFile()
        mid.tracks.append(mido.MidiTrack())
        midi_path = str(tmp_path / "empty.mid")
        mid.save(midi_path)

        result = compute_basic_quality_metrics(midi_path)
        assert result.get("note_count") == 0

    def test_compute_basic_quality_metrics_invalid_path(self):
        from ai_core.metrics import compute_basic_quality_metrics

        result = compute_basic_quality_metrics("/nonexistent/file.mid")
        # 에러 시 zero-filled dummy dict 반환 (downstream JSON 스키마 호환)
        assert isinstance(result, dict)
        assert result.get("note_count") == 0
        assert result.get("avg_velocity") == 0.0
