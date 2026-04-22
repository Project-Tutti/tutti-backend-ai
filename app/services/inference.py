"""
inference.py
Qwen2.5-0.5B 기반 악기 파트 생성 추론 파이프라인 (확장판)

사용법:
    python inference.py --song KissTheRain.mid
    python inference.py --song KissTheRain.mid --target flute
    python inference.py --song KissTheRain.mid --target trumpet
    python inference.py --song KissTheRain.mid --target piano --polyphonic
"""

import os
import bisect
import random
import copy
import logging

import torch
import torch.nn as nn
import pretty_midi
from collections import defaultdict
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 악기 설정 테이블
# monophonic=True  → 단선율 강제 (violin/flute/oboe/trumpet/clarinet 등)
# monophonic=False → 폴리포닉 허용 (piano/guitar/harp 등)
# ──────────────────────────────────────────────
TARGET_CONFIG = {
    # ── Piano (0-7) ──
    "acoustic_grand_piano":    {"program": 0,   "pitch_min": 21,  "pitch_max": 108, "monophonic": False},
    "bright_acoustic_piano":   {"program": 1,   "pitch_min": 21,  "pitch_max": 108, "monophonic": False},
    "electric_grand_piano":    {"program": 2,   "pitch_min": 21,  "pitch_max": 108, "monophonic": False},
    "honky_tonk_piano":        {"program": 3,   "pitch_min": 21,  "pitch_max": 108, "monophonic": False},
    "electric_piano_1":        {"program": 4,   "pitch_min": 28,  "pitch_max": 103, "monophonic": False},
    "electric_piano_2":        {"program": 5,   "pitch_min": 28,  "pitch_max": 103, "monophonic": False},
    "harpsichord":             {"program": 6,   "pitch_min": 29,  "pitch_max": 89,  "monophonic": False},
    "clavinet":                {"program": 7,   "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},

    # ── Chromatic Perc (8-15) ──
    "celesta":                 {"program": 8,   "pitch_min": 60,  "pitch_max": 108, "monophonic": False},
    "glockenspiel":            {"program": 9,   "pitch_min": 72,  "pitch_max": 108, "monophonic": True},
    "music_box":               {"program": 10,  "pitch_min": 60,  "pitch_max": 96,  "monophonic": True},
    "vibraphone":              {"program": 11,  "pitch_min": 53,  "pitch_max": 89,  "monophonic": False},
    "marimba":                 {"program": 12,  "pitch_min": 45,  "pitch_max": 96,  "monophonic": False},
    "xylophone":               {"program": 13,  "pitch_min": 65,  "pitch_max": 108, "monophonic": False},
    "tubular_bells":           {"program": 14,  "pitch_min": 60,  "pitch_max": 77,  "monophonic": True},
    "dulcimer":                {"program": 15,  "pitch_min": 48,  "pitch_max": 84,  "monophonic": False},

    # ── Organ (16-23) ──
    "drawbar_organ":           {"program": 16,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "percussive_organ":        {"program": 17,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "rock_organ":              {"program": 18,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "church_organ":            {"program": 19,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "reed_organ":              {"program": 20,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "accordion":               {"program": 21,  "pitch_min": 53,  "pitch_max": 89,  "monophonic": True},
    "harmonica":               {"program": 22,  "pitch_min": 60,  "pitch_max": 84,  "monophonic": True},
    "tango_accordion":         {"program": 23,  "pitch_min": 53,  "pitch_max": 89,  "monophonic": True},

    # ── Guitar (24-31) ──
    "nylon_guitar":            {"program": 24,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "steel_guitar":            {"program": 25,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "jazz_guitar":             {"program": 26,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "clean_guitar":            {"program": 27,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "muted_guitar":            {"program": 28,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "overdriven_guitar":       {"program": 29,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "distortion_guitar":       {"program": 30,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "guitar_harmonics":        {"program": 31,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": True},

    # ── Bass (32-39) ──
    "acoustic_bass":           {"program": 32,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},
    "electric_bass_finger":    {"program": 33,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},
    "electric_bass_pick":      {"program": 34,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},
    "fretless_bass":           {"program": 35,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},
    "slap_bass_1":             {"program": 36,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},
    "slap_bass_2":             {"program": 37,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},
    "synth_bass_1":            {"program": 38,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},
    "synth_bass_2":            {"program": 39,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},

    # ── Strings (40-47) ──
    "violin":                  {"program": 40,  "pitch_min": 55,  "pitch_max": 100, "monophonic": True},
    "viola":                   {"program": 41,  "pitch_min": 48,  "pitch_max": 91,  "monophonic": True},
    "cello":                   {"program": 42,  "pitch_min": 36,  "pitch_max": 76,  "monophonic": True},
    "contrabass":              {"program": 43,  "pitch_min": 28,  "pitch_max": 60,  "monophonic": True},
    "tremolo_strings":         {"program": 44,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "pizzicato_strings":       {"program": 45,  "pitch_min": 40,  "pitch_max": 84,  "monophonic": False},
    "orchestral_harp":         {"program": 46,  "pitch_min": 23,  "pitch_max": 103, "monophonic": False},
    "timpani":                 {"program": 47,  "pitch_min": 36,  "pitch_max": 57,  "monophonic": True},

    # ── Ensemble (48-55) ──
    "string_ensemble_1":       {"program": 48,  "pitch_min": 28,  "pitch_max": 96,  "monophonic": False},
    "string_ensemble_2":       {"program": 49,  "pitch_min": 28,  "pitch_max": 96,  "monophonic": False},
    "synth_strings_1":         {"program": 50,  "pitch_min": 28,  "pitch_max": 96,  "monophonic": False},
    "synth_strings_2":         {"program": 51,  "pitch_min": 28,  "pitch_max": 96,  "monophonic": False},
    "choir_aahs":              {"program": 52,  "pitch_min": 48,  "pitch_max": 79,  "monophonic": False},
    "voice_oohs":              {"program": 53,  "pitch_min": 48,  "pitch_max": 79,  "monophonic": True},
    "synth_voice":             {"program": 54,  "pitch_min": 48,  "pitch_max": 84,  "monophonic": True},
    "orchestra_hit":           {"program": 55,  "pitch_min": 48,  "pitch_max": 72,  "monophonic": False},

    # ── Brass (56-63) ──
    "trumpet":                 {"program": 56,  "pitch_min": 52,  "pitch_max": 82,  "monophonic": True},
    "trombone":                {"program": 57,  "pitch_min": 40,  "pitch_max": 72,  "monophonic": True},
    "tuba":                    {"program": 58,  "pitch_min": 28,  "pitch_max": 58,  "monophonic": True},
    "muted_trumpet":           {"program": 59,  "pitch_min": 52,  "pitch_max": 82,  "monophonic": True},
    "french_horn":             {"program": 60,  "pitch_min": 34,  "pitch_max": 77,  "monophonic": True},
    "brass_section":           {"program": 61,  "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "synth_brass_1":           {"program": 62,  "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "synth_brass_2":           {"program": 63,  "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},

    # ── Reed (64-71) ──
    "soprano_sax":             {"program": 64,  "pitch_min": 54,  "pitch_max": 84,  "monophonic": True},
    "alto_sax":                {"program": 65,  "pitch_min": 49,  "pitch_max": 80,  "monophonic": True},
    "tenor_sax":               {"program": 66,  "pitch_min": 44,  "pitch_max": 75,  "monophonic": True},
    "baritone_sax":            {"program": 67,  "pitch_min": 36,  "pitch_max": 69,  "monophonic": True},
    "oboe":                    {"program": 68,  "pitch_min": 58,  "pitch_max": 91,  "monophonic": True},
    "english_horn":            {"program": 69,  "pitch_min": 52,  "pitch_max": 81,  "monophonic": True},
    "bassoon":                 {"program": 70,  "pitch_min": 34,  "pitch_max": 75,  "monophonic": True},
    "clarinet":                {"program": 71,  "pitch_min": 50,  "pitch_max": 94,  "monophonic": True},

    # ── Pipe (72-79) ──
    "piccolo":                 {"program": 72,  "pitch_min": 74,  "pitch_max": 108, "monophonic": True},
    "flute":                   {"program": 73,  "pitch_min": 60,  "pitch_max": 96,  "monophonic": True},
    "recorder":                {"program": 74,  "pitch_min": 60,  "pitch_max": 91,  "monophonic": True},
    "pan_flute":               {"program": 75,  "pitch_min": 60,  "pitch_max": 96,  "monophonic": True},
    "blown_bottle":            {"program": 76,  "pitch_min": 60,  "pitch_max": 96,  "monophonic": True},
    "shakuhachi":              {"program": 77,  "pitch_min": 55,  "pitch_max": 84,  "monophonic": True},
    "whistle":                 {"program": 78,  "pitch_min": 60,  "pitch_max": 96,  "monophonic": True},
    "ocarina":                 {"program": 79,  "pitch_min": 60,  "pitch_max": 84,  "monophonic": True},

    # ── Synth Lead (80-87) ──
    "lead_square":             {"program": 80,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": True},
    "lead_sawtooth":           {"program": 81,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": True},
    "lead_calliope":           {"program": 82,  "pitch_min": 48,  "pitch_max": 96,  "monophonic": True},
    "lead_chiff":              {"program": 83,  "pitch_min": 48,  "pitch_max": 96,  "monophonic": True},
    "lead_charang":            {"program": 84,  "pitch_min": 48,  "pitch_max": 96,  "monophonic": True},
    "lead_voice":              {"program": 85,  "pitch_min": 48,  "pitch_max": 84,  "monophonic": True},
    "lead_fifths":             {"program": 86,  "pitch_min": 36,  "pitch_max": 84,  "monophonic": True},
    "lead_bass_lead":          {"program": 87,  "pitch_min": 28,  "pitch_max": 72,  "monophonic": False},

    # ── Synth Pad (88-95) ──
    "pad_new_age":             {"program": 88,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "pad_warm":                {"program": 89,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "pad_polysynth":           {"program": 90,  "pitch_min": 36,  "pitch_max": 96,  "monophonic": False},
    "pad_choir":               {"program": 91,  "pitch_min": 48,  "pitch_max": 84,  "monophonic": False},
    "pad_bowed":               {"program": 92,  "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "pad_metallic":            {"program": 93,  "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "pad_halo":                {"program": 94,  "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "pad_sweep":               {"program": 95,  "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},

    # ── Synth Effects (96-103) ──
    "fx_rain":                 {"program": 96,  "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "fx_soundtrack":           {"program": 97,  "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "fx_crystal":              {"program": 98,  "pitch_min": 48,  "pitch_max": 96,  "monophonic": False},
    "fx_atmosphere":           {"program": 99,  "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "fx_brightness":           {"program": 100, "pitch_min": 48,  "pitch_max": 96,  "monophonic": False},
    "fx_goblins":              {"program": 101, "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "fx_echoes":               {"program": 102, "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "fx_sci_fi":               {"program": 103, "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},

    # ── Ethnic (104-111) ──
    "sitar":                   {"program": 104, "pitch_min": 48,  "pitch_max": 77,  "monophonic": True},
    "banjo":                   {"program": 105, "pitch_min": 48,  "pitch_max": 84,  "monophonic": False},
    "shamisen":                {"program": 106, "pitch_min": 50,  "pitch_max": 79,  "monophonic": True},
    "koto":                    {"program": 107, "pitch_min": 55,  "pitch_max": 84,  "monophonic": True},
    "kalimba":                 {"program": 108, "pitch_min": 60,  "pitch_max": 91,  "monophonic": True},
    "bag_pipe":                {"program": 109, "pitch_min": 55,  "pitch_max": 79,  "monophonic": True},
    "fiddle":                  {"program": 110, "pitch_min": 55,  "pitch_max": 96,  "monophonic": True},
    "shanai":                  {"program": 111, "pitch_min": 48,  "pitch_max": 79,  "monophonic": True},

    # ── Percussive (112-119) ──
    "tinkle_bell":             {"program": 112, "pitch_min": 72,  "pitch_max": 108, "monophonic": True},
    "agogo":                   {"program": 113, "pitch_min": 60,  "pitch_max": 84,  "monophonic": True},
    "steel_drums":             {"program": 114, "pitch_min": 52,  "pitch_max": 84,  "monophonic": False},
    "woodblock":               {"program": 115, "pitch_min": 60,  "pitch_max": 72,  "monophonic": True},
    "taiko_drum":              {"program": 116, "pitch_min": 36,  "pitch_max": 57,  "monophonic": True},
    "melodic_tom":             {"program": 117, "pitch_min": 36,  "pitch_max": 57,  "monophonic": True},
    "synth_drum":              {"program": 118, "pitch_min": 36,  "pitch_max": 57,  "monophonic": True},
    "reverse_cymbal":          {"program": 119, "pitch_min": 36,  "pitch_max": 57,  "monophonic": False},

    # ── Sound Effects (120-127) ──
    "guitar_fret_noise":       {"program": 120, "pitch_min": 36,  "pitch_max": 84,  "monophonic": True},
    "breath_noise":            {"program": 121, "pitch_min": 36,  "pitch_max": 84,  "monophonic": True},
    "seashore":                {"program": 122, "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "bird_tweet":              {"program": 123, "pitch_min": 60,  "pitch_max": 96,  "monophonic": True},
    "telephone_ring":          {"program": 124, "pitch_min": 60,  "pitch_max": 84,  "monophonic": True},
    "helicopter":              {"program": 125, "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "applause":                {"program": 126, "pitch_min": 36,  "pitch_max": 84,  "monophonic": False},
    "gunshot":                 {"program": 127, "pitch_min": 36,  "pitch_max": 84,  "monophonic": True},
}

NOTE_TOKEN_LEN = 7
VEL_OFFSET     = 6



# ──────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────
def build_v5_vocab(actual_vocab_size: int = 682):
    vocab = {}

    def add(prefix, r):
        for i in r: vocab[f"{prefix}{i}"] = len(vocab)

    for t in ["PAD", "BOS", "EOS", "SEP", "PIECE_START", "PIECE_END",
              "BAR_START", "BAR_END", "PHRASE_END", "<PRE>", "<SUF>", "<MID>"]:
        vocab[t] = len(vocab)
    for g in ["CLASSICAL", "JAZZ", "POP", "ROCK", "ELECTRONIC", "FOLK", "UNKNOWN"]:
        vocab[f"GENRE_{g}"] = len(vocab)
    roots = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    for r in roots:
        for m in [":maj", ":min"]: vocab[f"KEY_{r}{m}"] = len(vocab)
    vocab["KEY_NONE"] = len(vocab)
    
    # vocab_size가 682 이상일 때만 기존 TARGET_ 토큰 3종 포함
    if actual_vocab_size >= 682:
        for p in [40, 68, 73]: vocab[f"TARGET_{p}"] = len(vocab)
    for m in ["4:4", "3:4", "2:4", "6:8", "12:8", "OTHER"]:
        vocab[f"METER_{m}"] = len(vocab)
    add("DENSITY_", range(1, 6))
    add("INST=", range(129))
    for a in ["ART_NORMAL", "ART_LEGATO", "ART_VIBRATO", "ART_STACCATO"]:
        vocab[a] = len(vocab)
    add("EXPR_", range(32))
    add("TIME=", range(96))
    add("PITCH=", range(128))
    add("DUR=", range(1, 193))
    add("VEL=", range(32))
    for w in ["melodic", "epic", "calm", "fast", "slow", "sad", "happy",
              "piano", "strings", "orchestra", "cinematic"]:
        vocab[f"TEXT_{w}"] = len(vocab)
    return vocab


FLAT_TO_SHARP = {
    "Db": "C#", "Eb": "D#", "Fb": "E", "Gb": "F#",
    "Ab": "G#", "Bb": "A#", "Cb": "B"
}

NOTE_TOKEN_LEN = 7
VEL_OFFSET = 6

# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────
def load_model(ckpt_path, vocab_size, vocab, device):
    _device = torch.device(device)

    MODEL_NAME = "Qwen/Qwen2.5-0.5B"
    config = AutoConfig.from_pretrained(MODEL_NAME)
    config.vocab_size              = vocab_size
    config.pad_token_id            = vocab["PAD"]
    config.max_position_embeddings = 2048
    config.sliding_window          = None

    model = AutoModelForCausalLM.from_config(config)
    model = model.to(torch.bfloat16)
    model.model.embed_tokens = nn.Embedding(vocab_size, config.hidden_size).to(torch.bfloat16)
    model.lm_head            = nn.Linear(config.hidden_size, vocab_size, bias=False).to(torch.bfloat16)

    sf   = os.path.join(ckpt_path, "model.safetensors")
    bin_ = os.path.join(ckpt_path, "pytorch_model.bin")
    if os.path.exists(sf):
        from safetensors.torch import load_file
        state = load_file(sf, device="cpu")
        model.load_state_dict(state, strict=True)
        logger.info(f"체크포인트 로드 (safetensors): {sf}")
    elif os.path.exists(bin_):
        state = torch.load(bin_, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        logger.info(f"체크포인트 로드 (.bin): {bin_}")
    else:
        raise FileNotFoundError(f"❌ 체크포인트 없음: {ckpt_path}")

    model.config.use_cache = False
    model.eval()
    model.to(_device)

    # RTX 4090 / T4 TF32 가속
    # (transformers Dynamo 충돌 이슈로 인해 torch.compile 대신 네이티브 TF32 사용)
    torch.set_float32_matmul_precision("high")
    logger.info("TF32 하드웨어 가속 활성화 (torch.compile 비활성화)")

    return model


# ──────────────────────────────────────────────
# MIDI → 마디 토큰
# ──────────────────────────────────────────────
def midi_to_bar_tokens(midi_path, genre, VOCAB):
    pm          = pretty_midi.PrettyMIDI(midi_path)
    res         = pm.resolution
    timeline    = defaultdict(lambda: defaultdict(list))
    key_changes = sorted(pm.key_signature_changes,  key=lambda x: x.time)
    key_times   = [k.time for k in key_changes]
    ts_changes  = sorted(pm.time_signature_changes, key=lambda x: x.time)
    ts_times    = [t.time for t in ts_changes]
    tempo_times, tempos = pm.get_tempo_changes()

    for inst in pm.instruments:
        p     = 128 if inst.is_drum else inst.program
        notes = sorted(inst.notes, key=lambda x: (x.start, x.pitch))
        last_end_tick = -1

        for i, n in enumerate(notes):
            note_dur   = n.end - n.start
            tempo_idx  = max(0, bisect.bisect_right(tempo_times, n.start) - 1)
            bpm        = tempos[tempo_idx] if len(tempos) > 0 else 120.0
            s_per_beat = 60.0 / bpm
            dur_tick   = max(1, min(192, round((note_dur / s_per_beat) * 24)))

            ts_idx = max(0, bisect.bisect_right(ts_times, n.start) - 1)
            if ts_idx < len(ts_changes):
                ts            = ts_changes[ts_idx]
                mkey          = f"{ts.numerator}:{ts.denominator}"
                meter_tok     = VOCAB.get(f"METER_{mkey}", VOCAB["METER_OTHER"])
                beats_per_bar = ts.numerator
            else:
                meter_tok     = VOCAB["METER_OTHER"]
                beats_per_bar = 4

            k_idx = bisect.bisect_right(key_times, n.start) - 1
            if 0 <= k_idx < len(key_changes):
                ks    = key_changes[k_idx]
                root  = ks.key_number % 12
                mode  = "maj" if ks.key_number < 12 else "min"
                roots = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
                rname = FLAT_TO_SHARP.get(roots[root], roots[root])
                key_tok = VOCAB.get(f"KEY_{rname}:{mode}", VOCAB["KEY_NONE"])
            else:
                key_tok = VOCAB["KEY_NONE"]

            bar_idx        = int(pm.time_to_tick(n.start) // (res * beats_per_bar))
            bar_start_tick = bar_idx * res * beats_per_bar
            rel_tick       = pm.time_to_tick(n.start) - bar_start_tick
            time_tok       = VOCAB[f"TIME={min(95, rel_tick * 96 // (res * beats_per_bar))}"]

            n_start_tick  = pm.time_to_tick(n.start)
            is_phrase_end = (last_end_tick > 0 and
                             (n_start_tick - last_end_tick) >= res)

            nxt_start    = notes[i+1].start if i+1 < len(notes) else n.end + 1
            legato_ratio = note_dur / max(nxt_start - n.start, 1e-6)
            if dur_tick <= 2:          art_tok = VOCAB["ART_STACCATO"]
            elif nxt_start == n.start: art_tok = VOCAB["ART_NORMAL"]
            elif legato_ratio > 0.95:  art_tok = VOCAB["ART_LEGATO"]
            else:                      art_tok = VOCAB["ART_NORMAL"]

            expr_tok = VOCAB[f"EXPR_{min(31, n.velocity * 32 // 128)}"]
            vel_tok  = VOCAB[f"VEL={min(31, n.velocity * 32 // 128)}"]
            inst_tok = VOCAB[f"INST={p}"]

            timeline[bar_idx][p].append(
                (time_tok, inst_tok, art_tok, expr_tok,
                 n.pitch, dur_tick, vel_tok,
                 meter_tok, key_tok, is_phrase_end))
            last_end_tick = pm.time_to_tick(n.end)

    header      = [VOCAB["PIECE_START"], VOCAB[f"GENRE_{genre}"]]
    final_beats = ts_changes[-1].numerator if ts_changes else 4
    max_bar     = int(pm.time_to_tick(pm.get_end_time()) // (res * final_beats))

    bar_tokens        = {}
    accumulated_ticks = 0
    for bar_idx in range(max_bar + 1):
        bar_time  = pm.tick_to_time(accumulated_ticks)
        ts_idx    = max(0, bisect.bisect_right(ts_times, bar_time) - 1)
        beats     = ts_changes[ts_idx].numerator if ts_idx < len(ts_changes) else 4
        mkey      = (f"{ts_changes[ts_idx].numerator}:{ts_changes[ts_idx].denominator}"
                     if ts_idx < len(ts_changes) else "OTHER")
        meter_tok = VOCAB.get(f"METER_{mkey}", VOCAB["METER_OTHER"])

        k_idx = bisect.bisect_right(key_times, bar_time) - 1
        if 0 <= k_idx < len(key_changes):
            ks    = key_changes[k_idx]
            root  = ks.key_number % 12
            mode  = "maj" if ks.key_number < 12 else "min"
            roots = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            rname = FLAT_TO_SHARP.get(roots[root], roots[root])
            key_tok = VOCAB.get(f"KEY_{rname}:{mode}", VOCAB["KEY_NONE"])
        else:
            key_tok = VOCAB["KEY_NONE"]

        if bar_idx in timeline:
            bar_data    = timeline[bar_idx]
            total_notes = sum(len(v) for v in bar_data.values())
            density     = min(5, max(1, total_notes // 4))
            btoks = [VOCAB["BAR_START"], key_tok, meter_tok, VOCAB[f"DENSITY_{density}"]]
            all_notes = []
            for p in bar_data.keys():
                for (tt, it, at, et, pitch, dt, vt, _, _, phrase_end) in bar_data[p]:
                    all_notes.append((tt - VOCAB["TIME=0"], it, phrase_end,
                                      it, at, et, tt, pitch, dt, vt))
            all_notes.sort(key=lambda x: (x[0], x[1]))
            for (_, _, phrase_end, it, at, et, tt, pitch, dt, vt) in all_notes:
                if phrase_end: btoks.append(VOCAB["PHRASE_END"])
                btoks += [it, at, et, tt, VOCAB[f"PITCH={pitch}"],
                          VOCAB[f"DUR={dt}"], vt]
            btoks.append(VOCAB["BAR_END"])
        else:
            btoks = [VOCAB["BAR_START"], key_tok, meter_tok,
                     VOCAB["DENSITY_1"], VOCAB["BAR_END"]]

        bar_tokens[bar_idx]    = btoks
        accumulated_ticks     += res * beats

    return header, bar_tokens, max_bar, pm


# ──────────────────────────────────────────────
# 컨텍스트 트리밍 (마디 단위, 비율 보존)
# ──────────────────────────────────────────────
def trim_bars_preserving_ratio(past_bars, future_bars,
                               header_len, current_len, max_ctx,
                               target_past_ratio):
    """마디 단위로 제거하며 목표 past:future 비율을 최대한 유지."""
    p_list = list(past_bars)
    f_list = list(future_bars)

    def total():
        return (header_len + current_len
                + sum(len(t) for _, t in p_list)
                + sum(len(t) for _, t in f_list))

    while total() > max_ctx and (p_list or f_list):
        p_count = len(p_list)
        f_count = len(f_list)

        if not p_list:
            removed = f_list.pop()
            logger.info(f"    ✂️  Future Bar {removed[0]} 제거 ({len(removed[1])} tok)")
            continue
        if not f_list:
            removed = p_list.pop(0)
            logger.info(f"    ✂️  Past Bar {removed[0]} 제거 ({len(removed[1])} tok)")
            continue

        ratio_if_remove_past = (p_count - 1) / (p_count - 1 + f_count) if (p_count - 1 + f_count) > 0 else 0
        ratio_if_remove_future = p_count / (p_count + f_count - 1) if (p_count + f_count - 1) > 0 else 1

        diff_past   = abs(ratio_if_remove_past   - target_past_ratio)
        diff_future = abs(ratio_if_remove_future - target_past_ratio)

        if diff_future <= diff_past:
            removed = f_list.pop()
            logger.info(f"    ✂️  Future Bar {removed[0]} 제거 ({len(removed[1])} tok)")
        else:
            removed = p_list.pop(0)
            logger.info(f"    ✂️  Past Bar {removed[0]} 제거 ({len(removed[1])} tok)")

    return p_list, f_list


# ──────────────────────────────────────────────
# 마디 토큰 파싱 / 인터리빙 병합
# ──────────────────────────────────────────────
def parse_bar_notes(bar_toks, VOCAB_R):
    """마디 토큰을 header와 노트 리스트로 분리.
    Returns: (header_tokens, [(time_val, inst_val, note_token_list), ...])
    """
    header = []
    body   = []
    in_header = True
    for tok in bar_toks:
        name = VOCAB_R.get(tok, "")
        if in_header:
            if name.startswith("INST=") or name == "PHRASE_END":
                in_header = False
                body.append(tok)
            elif name == "BAR_END":
                break
            else:
                header.append(tok)
        else:
            if name != "BAR_END":
                body.append(tok)

    notes = []
    i = 0
    while i < len(body):
        name = VOCAB_R.get(body[i], "")
        prefix = []
        if name == "PHRASE_END":
            prefix = [body[i]]
            i += 1
            if i >= len(body):
                break
            name = VOCAB_R.get(body[i], "")
        if name.startswith("INST=") and i + 7 <= len(body):
            note_toks = prefix + body[i:i+7]
            time_name = VOCAB_R.get(body[i+3], "")
            time_val  = int(time_name.split("=")[1]) if time_name.startswith("TIME=") else 0
            inst_val  = int(name.split("=")[1])
            notes.append((time_val, inst_val, note_toks))
            i += 7
        else:
            i += 1
    return header, notes


def merge_bars(source_bar_toks, gen_bar_toks, VOCAB, VOCAB_R):
    """원곡 마디와 생성 마디를 학습 포맷대로 인터리빙 병합."""
    if not gen_bar_toks:
        return list(source_bar_toks)
    if not source_bar_toks:
        return list(gen_bar_toks)

    src_header, src_notes = parse_bar_notes(source_bar_toks, VOCAB_R)
    _,          gen_notes = parse_bar_notes(gen_bar_toks,    VOCAB_R)

    all_notes = src_notes + gen_notes
    all_notes.sort(key=lambda x: (x[0], x[1]))

    # density 업데이트
    density = min(5, max(1, len(all_notes) // 4))
    updated_header = []
    for tok in src_header:
        name = VOCAB_R.get(tok, "")
        if name.startswith("DENSITY_"):
            updated_header.append(VOCAB[f"DENSITY_{density}"])
        else:
            updated_header.append(tok)

    merged = updated_header
    for _, _, toks in all_notes:
        merged.extend(toks)
    merged.append(VOCAB["BAR_END"])
    return merged


# ──────────────────────────────────────────────
# 슬라이딩 윈도우 생성
# monophonic=True  → INST 재발음을 VEL 완결 전까지 억제 (단선율)
# monophonic=False → 억제 없음 (폴리포닉)
# ──────────────────────────────────────────────
@torch.no_grad()
def generate_sliding_window(model, header, bar_tokens, max_bar,
                             target_prog, pitch_min, pitch_max,
                             window_bars, context_bars, future_bars,
                             temperature, top_p,
                             VOCAB, VOCAB_R, source_pm, device,
                             monophonic=True, progress_hook=None):
    SEQ_LEN  = 2048
    MAX_CTX  = SEQ_LEN - 300
    all_notes         = []
    gen_bar_tokens    = {}

    VEL_IDS        = {VOCAB[f"VEL={i}"] for i in range(32)}
    INST_TARGET_ID = VOCAB[f"INST={target_prog}"]

    target_past_ratio = context_bars / (context_bars + future_bars) if (context_bars + future_bars) > 0 else 0.5
    total_windows = (max_bar // window_bars) + 1
    
    logger.info(f"   총 {max_bar+1}마디 / 윈도우 {window_bars}마디 → {total_windows}번 생성")
    logger.info(f"   단선율 강제: {'ON' if monophonic else 'OFF (폴리포닉)'}")

    for win_idx in range(total_windows):
        win_start = win_idx * window_bars
        win_end   = min(win_start + window_bars - 1, max_bar)
        
        if progress_hook is not None:
            pct = int(20 + (win_idx / total_windows) * 60)
            progress_hook(pct)

        if win_start > max_bar:
            break

        ctx_start = max(0, win_start - context_bars)
        fut_end   = min(max_bar, win_end + future_bars)

        # 1. Header (고정)
        header_toks = list(header)

        # 2. Past (과거 원본 + 과거 생성 결과)
        past_bar_list = []
        for b in range(ctx_start, win_start):
            bar_toks = list(bar_tokens.get(b, []))
            if b in gen_bar_tokens:
                bar_toks += gen_bar_tokens[b]
            if bar_toks:
                past_bar_list.append((b, bar_toks))

        # 3. Current Window (현재 작곡해야 할 구간의 원본)
        current_toks = []
        for b in range(win_start, win_end + 1):
            current_toks += bar_tokens.get(b, [])

        # 4. Future (미래 가이드라인 원본)
        future_bar_list = []
        for b in range(win_end + 1, fut_end + 1):
            bar_toks = list(bar_tokens.get(b, []))
            if bar_toks:
                future_bar_list.append((b, bar_toks))

        h_len = len(header_toks)
        p_len = sum(len(t) for _, t in past_bar_list)
        c_len = len(current_toks)
        f_len = sum(len(t) for _, t in future_bar_list)
        total_len = h_len + p_len + c_len + f_len

        logger.info(f"   --- WINDOW #{win_idx} (Bar {win_start}~{win_end}) ---")
        logger.info(f"   ● [Context] Past: {p_len} | Current: {c_len} | Future: {f_len} -> Total: {total_len}/{SEQ_LEN}")

        if total_len > MAX_CTX:
            logger.warning(f"   ⚠️ OVERFLOW! {total_len - MAX_CTX} tokens over limit. Trimming...")
            past_bar_list, future_bar_list = trim_bars_preserving_ratio(
                past_bar_list, future_bar_list,
                h_len, c_len, MAX_CTX, target_past_ratio)

        past_toks = []
        for _, toks in past_bar_list:
            past_toks += toks
        future_toks = []
        for _, toks in future_bar_list:
            future_toks += toks

        context = header_toks + past_toks + current_toks + future_toks
        input_ids = torch.tensor([context], dtype=torch.long, device=device)
        gen_toks  = []

        with torch.no_grad():
            out = model(input_ids=input_ids, use_cache=True)
            pkv = out.past_key_values

            for tok in [VOCAB["BAR_START"], INST_TARGET_ID]:
                t_in = torch.tensor([[tok]], dtype=torch.long, device=device)
                out  = model(input_ids=t_in, past_key_values=pkv, use_cache=True)
                pkv  = out.past_key_values
                gen_toks.append(tok)

            bar_count      = 1
            target_playing = True   # 첫 INST 강제 주입 후 발음 중
            cur_in = torch.tensor([[gen_toks[-1]]], dtype=torch.long, device=device)

            for _ in range(1024):
                out    = model(input_ids=cur_in, past_key_values=pkv, use_cache=True)
                pkv    = out.past_key_values
                logits = out.logits[0, -1, :].float()

                # 1. pitch 음역 마스킹
                for pitch in range(128):
                    if pitch < pitch_min or pitch > pitch_max:
                        logits[VOCAB[f"PITCH={pitch}"]] = -1e9

                # 2. 단선율 강제 (monophonic=True일 때만)
                if monophonic and target_playing:
                    logits[INST_TARGET_ID] = -1e9

                logits  = logits / temperature
                probs   = torch.softmax(logits, dim=-1)

                # Top-p Sampling
                s_probs, s_idx = torch.sort(probs, descending=True)
                cumsum = torch.cumsum(s_probs, dim=0)
                mask = cumsum - s_probs > top_p
                s_probs[mask] = 0
                s_probs /= s_probs.sum()

                next_tok = s_idx[torch.multinomial(s_probs, 1)].item()
                gen_toks.append(next_tok)

                # 3. 단선율 상태 업데이트
                if next_tok == INST_TARGET_ID:
                    target_playing = True
                elif monophonic and target_playing and next_tok in VEL_IDS:
                    target_playing = False

                # 4. BAR/종료 처리
                if next_tok == VOCAB["BAR_START"]:
                    bar_count += 1
                    target_playing = False
                    if bar_count > window_bars:
                        break
                if next_tok in (VOCAB["PIECE_END"], VOCAB["EOS"]):
                    break

                cur_in = torch.tensor([[next_tok]], dtype=torch.long, device=device)

        # 생성 토큰 마디별 분리
        cur_bar_toks = []
        cur_bar_num  = win_start
        for tok in gen_toks:
            if VOCAB_R.get(tok, "") == "BAR_START":
                if cur_bar_toks:
                    gen_bar_tokens[cur_bar_num] = cur_bar_toks
                    cur_bar_num += 1
                cur_bar_toks = [tok]
            else:
                cur_bar_toks.append(tok)
        if cur_bar_toks:
            gen_bar_tokens[cur_bar_num] = cur_bar_toks

        win_notes = decode_tokens(gen_toks, source_pm, target_prog,
                                  bar_offset=win_start,
                                  win_start=win_start,
                                  win_end=win_end,
                                  VOCAB=VOCAB, VOCAB_R=VOCAB_R,
                                  pitch_min=pitch_min,  # ← 추가
                                  pitch_max=pitch_max)  # ← 추가
        all_notes.extend(win_notes)
        logger.info(f"   윈도우 {win_idx+1}/{total_windows} "
                    f"(bar {win_start}~{win_end}): {len(win_notes)}노트")

    return all_notes


# ──────────────────────────────────────────────
# 토큰 디코딩
# ──────────────────────────────────────────────
def decode_tokens(tokens, source_pm, target_prog,
                  bar_offset=0, win_start=0, win_end=9999,
                  VOCAB=None, VOCAB_R=None,
                  pitch_min=0, pitch_max=127):   # ← 추가
    res         = source_pm.resolution
    ts_changes  = sorted(source_pm.time_signature_changes, key=lambda x: x.time)
    ts_times    = [t.time for t in ts_changes]
    tempo_times, tempos = source_pm.get_tempo_changes()

    bar_tick_map = {}
    acc = 0
    for b in range(2000):
        bar_tick_map[b] = acc
        bt  = source_pm.tick_to_time(acc)
        idx = max(0, bisect.bisect_right(ts_times, bt) - 1)
        bpb = ts_changes[idx].numerator if idx < len(ts_changes) else 4
        acc += res * bpb

    notes_out = []
    bar_idx   = bar_offset - 1
    cur_inst  = cur_time_tok = cur_pitch = cur_dur = cur_vel = None

    for tok in tokens:
        name = VOCAB_R.get(tok, "?")
        if   name == "BAR_START":       bar_idx += 1
        elif name.startswith("INST="):  cur_inst     = int(name.split("=")[1])
        elif name.startswith("TIME="):  cur_time_tok = int(name.split("=")[1])
        elif name.startswith("PITCH="): cur_pitch    = int(name.split("=")[1])
        elif name.startswith("DUR="):   cur_dur      = int(name.split("=")[1])
        elif name.startswith("VEL="):
            cur_vel = int(name.split("=")[1])
            if (cur_inst == target_prog and
                    cur_pitch    is not None and
                    cur_dur      is not None and
                    cur_time_tok is not None and
                    win_start <= bar_idx <= win_end):

                # ── 디버그 추가 ──
                logger.debug(f"  [BAR {bar_idx}] INST={cur_inst} PITCH={cur_pitch} "
                             f"DUR={cur_dur} VEL={cur_vel} "
                             f"{'⚠️ OUT OF RANGE' if cur_pitch < pitch_min or cur_pitch > pitch_max else '✅'}")

                b_tick    = bar_tick_map.get(bar_idx, 0)
                b_time    = source_pm.tick_to_time(b_tick)
                ts_idx    = max(0, bisect.bisect_right(ts_times, b_time) - 1)
                bpb       = ts_changes[ts_idx].numerator if ts_idx < len(ts_changes) else 4
                bar_ticks = res * bpb
                abs_tick  = b_tick + cur_time_tok * bar_ticks // 96
                start_sec = source_pm.tick_to_time(abs_tick)

                t_idx   = max(0, bisect.bisect_right(tempo_times, start_sec) - 1)
                bpm     = tempos[t_idx] if len(tempos) > 0 else 120.0
                dur_sec = (cur_dur / 24.0) * (60.0 / bpm)

                notes_out.append({
                    "start":    start_sec,
                    "end":      start_sec + dur_sec,
                    "pitch":    cur_pitch,
                    "velocity": max(1, min(127, (cur_vel + 1) * 4)),
                })
            cur_pitch = cur_dur = cur_vel = None

    return notes_out


# ──────────────────────────────────────────────
# 후처리
# monophonic=True  → 겹침 제거 + 도약 완화 적용
# monophonic=False → 겹침 제거만 생략 (폴리포닉 허용)
# ──────────────────────────────────────────────
def postprocess(notes, target_name, monophonic=True):
    cfg = TARGET_CONFIG[target_name]

    # 1. 음역 클리핑
    notes = [n for n in notes
             if cfg["pitch_min"] <= n["pitch"] <= cfg["pitch_max"]]

    # 2. 비정상적으로 긴 음표 클리핑
    MAX_DUR = 4.0
    for n in notes:
        if n["end"] - n["start"] > MAX_DUR:
            n["end"] = n["start"] + MAX_DUR

    # 3. 너무 짧은 음표 제거
    notes = [n for n in notes if (n["end"] - n["start"]) >= 0.05]

    # 4. 단선율 악기만: 폴리포닉 제거
    if monophonic:
        notes = sorted(notes, key=lambda x: x["start"])
        mono  = []
        for n in notes:
            if mono and n["start"] < mono[-1]["end"]:
                mono[-1]["end"] = n["start"]
                if mono[-1]["end"] - mono[-1]["start"] < 0.05:
                    mono.pop()
            mono.append(n)
        notes = mono

    # 5. 슬라이딩 윈도우 경계 연결 보정
    LEGATO_GAP = 0.03
    notes = sorted(notes, key=lambda x: x["start"])
    for i in range(len(notes) - 1):
        gap = notes[i+1]["start"] - notes[i]["end"]
        if 0 < gap < LEGATO_GAP:
            notes[i]["end"] = notes[i+1]["start"]

    # 6. 단선율 악기만: 큰 도약 완화
    if monophonic:
        MAX_INTERVAL = 12
        notes = sorted(notes, key=lambda x: x["start"])
        for i in range(1, len(notes)):
            interval = notes[i]["pitch"] - notes[i-1]["pitch"]
            if abs(interval) > MAX_INTERVAL:
                if interval > 0: notes[i]["pitch"] -= 12
                else:            notes[i]["pitch"] += 12
                if not (cfg["pitch_min"] <= notes[i]["pitch"] <= cfg["pitch_max"]):
                    if interval > 0: notes[i]["pitch"] += 12
                    else:            notes[i]["pitch"] -= 12

    # 7. 최종 음역 재확인
    notes = [n for n in notes
             if cfg["pitch_min"] <= n["pitch"] <= cfg["pitch_max"]]

    # 8. 최종 길이 재확인
    notes = [n for n in notes if (n["end"] - n["start"]) >= 0.05]

    return notes


# ──────────────────────────────────────────────
# MIDI 저장
# ──────────────────────────────────────────────
def save_midi(notes, source_pm, output_path, target_prog, target_name,
              original_song_path=None,
              actual_instrument_name=None, actual_midi_program=None):
    """
    원본 미디 파일(`original_song_path`)에 AI가 생성한 노트들만 새로운 트랙으로
    덧붙여 저장(Append)합니다. (Jitter 및 SysEx 손실 방지)
    """
    if not original_song_path:
        raise ValueError("original_song_path is required for Mido Append strategy")
        
    import mido
    mid = mido.MidiFile(original_song_path)
    
    if mid.type == 0:
        mid.type = 1

    display_name = actual_instrument_name or target_name
    new_track = mido.MidiTrack()
    new_track.append(mido.MetaMessage('track_name', name=f"AI_{display_name}", time=0))
    
    is_drum = (target_prog == 128)
    if is_drum:
        msg_prog = 0
    else:
        msg_prog = actual_midi_program if actual_midi_program is not None else target_prog
    
    msg_prog = max(0, min(127, msg_prog))
    
    if is_drum:
        msg_chan = 9  # 드럼 채널 설정
    else:
        used_channels = set()
        for track in mid.tracks:
            for msg in track:
                if hasattr(msg, 'channel'):
                    used_channels.add(msg.channel)
        
        free_channels = [c for c in range(16) if c != 9 and c not in used_channels]
        if free_channels:
            msg_chan = free_channels[0]
        else:
            fallback = [c for c in range(16) if c != 9]
            msg_chan = fallback[0] if fallback else 0
            logger.warning(f"MIDI 채널 포화 상태. AI 악기가 채널 {msg_chan}을 일부 공유합니다.")
            
    new_track.append(mido.Message('program_change', program=msg_prog, channel=msg_chan, time=0))
    
    events = []
    for n in notes:
        st_tick = int(round(source_pm.time_to_tick(n["start"])))
        en_tick = int(round(source_pm.time_to_tick(n["end"])))
        pitch = max(0, min(127, int(n["pitch"])))
        vel = max(1, min(127, int(n["velocity"])))
        
        events.append((st_tick, 'note_on', pitch, vel))
        events.append((en_tick, 'note_off', pitch, 0))
        
    events.sort(key=lambda x: (x[0], 0 if x[1] == 'note_off' else 1))
    
    last_tick = 0
    for tick, msg_type, pitch, vel in events:
        delta = max(0, tick - last_tick)
        new_track.append(mido.Message(msg_type, note=pitch, velocity=vel, time=delta, channel=msg_chan))
        last_tick = tick
        
    new_track.append(mido.MetaMessage('end_of_track', time=0))
    mid.tracks.append(new_track)
    
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    mid.save(output_path)
    logger.info(f"✅ 원본 보존 기반 병합 저장 완료: {output_path} (생성된 노트 {len(notes)}개)")

# ──────────────────────────────────────────────
# 공개 API: run_arrangement
# ──────────────────────────────────────────────
def run_arrangement(
    song_path: str,
    target_prog: int,
    genre: str,
    temperature: float,
    pitch_min: int,
    pitch_max: int,
    output_path: str,
    model, vocab, vocab_r, device,
    progress_hook=None,
    original_song_path: str = None,
    actual_instrument_name: str = None,
    actual_midi_program: int = None,
) -> str:
    """편곡 추론 실행 - 외부에서 호출되는 API 엔트리포인트."""
    # 고정 하이퍼파라미터 (2-2-2 셋업 반영)
    context_bars = 10
    window_bars = 4
    future_bars = 0
    top_p = 0.95
    seed = 42

    if not os.path.exists(song_path):
        raise FileNotFoundError(f"입력 파일 없음: {song_path}")

    _device = torch.device(device) if isinstance(device, str) else device

    random.seed(seed)
    torch.manual_seed(seed)

    # TARGET_CONFIG에서 타겟 정보 수집
    target_name = "acoustic_grand_piano"
    monophonic = False
    for k, v in TARGET_CONFIG.items():
        if v["program"] == target_prog:
            target_name = k
            monophonic = v["monophonic"]
            if pitch_min is None: pitch_min = v["pitch_min"]
            if pitch_max is None: pitch_max = v["pitch_max"]
            break
    else:
        # 매칭 실패 처리 (예: 128번 드럼)
        if target_prog == 128:
            target_name = "drum"
            monophonic = False
            if pitch_min is None: pitch_min = 35
            if pitch_max is None: pitch_max = 81
        else:
            if pitch_min is None: pitch_min = 0
            if pitch_max is None: pitch_max = 127

    logger.info(f"입력 MIDI 토크나이징: {song_path}")
    header, bar_tokens, max_bar, source_pm = midi_to_bar_tokens(song_path, genre, vocab)
    logger.info(f"총 마디 수: {max_bar + 1}")

    logger.info(f"생성 시작 (target_prog={target_prog}, name={target_name}, "
                f"genre={genre}, temp={temperature}, monophonic={monophonic})")
                
    all_notes = generate_sliding_window(
        model, header, bar_tokens, max_bar,
        target_prog, pitch_min, pitch_max,
        window_bars, context_bars, future_bars,
        temperature, top_p,
        vocab, vocab_r, source_pm, _device,
        monophonic=monophonic,
        progress_hook=progress_hook
    )

    logger.info(f"디코딩 노트: {len(all_notes)}")
    
    if target_prog != 128:
        all_notes = postprocess(all_notes, target_name, monophonic=monophonic)
    else:
        # 드럼 전용 간단 후처리
        all_notes = [n for n in all_notes if pitch_min <= n["pitch"] <= pitch_max]
        all_notes = [n for n in all_notes if (n["end"] - n["start"]) >= 0.05]
        
    logger.info(f"후처리 후: {len(all_notes)}")

    if len(all_notes) == 0:
        raise RuntimeError("No notes generated.")

    save_midi(all_notes, source_pm, output_path, target_prog, target_name,
              original_song_path=original_song_path,
              actual_instrument_name=actual_instrument_name,
              actual_midi_program=actual_midi_program)

    return output_path
