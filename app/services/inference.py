"""
inference.py  —  Qwen2.5-0.5B 기반 악기 파트 생성 추론 파이프라인 (서버용)

[공개 API — arrangement.py에서만 호출할 것]
    resolve_target(instrument_id)  → str
    run_arrangement(song_path, target, genre, temperature,
                    pitch_min, pitch_max, output_path,
                    model, vocab, vocab_r, device)  → str

[모델 초기화 — model_registry.py에서만 호출할 것]
    load_model(ckpt_path, vocab_size, vocab, device)  → model
    build_v5_vocab()                                  → dict

⚠️  run_arrangement() 및 resolve_target()의 시그니처를 변경하면
    arrangement.py가 깨집니다. 반드시 백엔드 엔지니어와 협의하세요.
"""

import os
import bisect
import random
import copy
import logging
from collections import defaultdict

import torch
import torch.nn as nn
import pretty_midi
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 1. 악기 그룹 정의 (INSTRUMENT_GROUPS)
# ──────────────────────────────────────────────
INSTRUMENT_GROUPS = {
    "drum":       {"representative": 128, "is_drum": True,  "pitch_min": 35,  "pitch_max": 81},
    "keyboard":   {"representative": 0,   "is_drum": False, "pitch_min": 21,  "pitch_max": 108},
    "organ":      {"representative": 16,  "is_drum": False, "pitch_min": 36,  "pitch_max": 96},
    "mallet":     {"representative": 12,  "is_drum": False, "pitch_min": 48,  "pitch_max": 96},
    "guitar":     {"representative": 25,  "is_drum": False, "pitch_min": 40,  "pitch_max": 88},
    "dist_guitar":{"representative": 30,  "is_drum": False, "pitch_min": 40,  "pitch_max": 88},
    "bass":       {"representative": 33,  "is_drum": False, "pitch_min": 28,  "pitch_max": 67},
    "violin":     {"representative": 40,  "is_drum": False, "pitch_min": 55,  "pitch_max": 103},
    "woodwind":   {"representative": 73,  "is_drum": False, "pitch_min": 60,  "pitch_max": 96},
    "saxophone":  {"representative": 65,  "is_drum": False, "pitch_min": 49,  "pitch_max": 80},
    "synth":      {"representative": 81,  "is_drum": False, "pitch_min": 36,  "pitch_max": 96},
    "brass":      {"representative": 56,  "is_drum": False, "pitch_min": 52,  "pitch_max": 82},
    "ensemble":   {"representative": 48,  "is_drum": False, "pitch_min": 36,  "pitch_max": 96},
}

ALL_TARGET_NAMES = list(INSTRUMENT_GROUPS.keys())

# program → representative 룩업
_GROUPING_PROGRAMS = {
    128: [128],
    0:   [0,1,2,3,4,5,6,7],
    16:  [16,17,18,19,20,21,22,23],
    12:  [8,9,10,11,12,13,14,15,112,114],
    25:  [24,25,26,27,28,31,45,46,104,105,106,107,108,110],
    30:  [29,30],
    33:  [32,33,34,35,36,37,38,39],
    40:  [40,41,42,43],
    73:  [68,69,70,71,72,73,74,75,77,78,79,111],
    65:  [64,65,66,67],
    81:  [80,81,82,83,84,85,86,87],
    56:  [56,57,58,59,60],
    48:  [44,48,49,50,51,52,53,54,61,62,63,76,88,89,90,
          91,92,93,94,95,96,97,98,99,100,101,102,103],
}
PROGRAM_TO_REP = {}
for _rep, _programs in _GROUPING_PROGRAMS.items():
    for _p in _programs:
        PROGRAM_TO_REP[_p] = _rep

# representative → group name 역매핑
_REP_TO_GROUP = {cfg["representative"]: name for name, cfg in INSTRUMENT_GROUPS.items()}

DROP_SET = {47, 55, 109, 113, 115, 116, 117, 118, 119, 120,
            121, 122, 123, 124, 125, 126, 127}

FLAT_TO_SHARP = {
    "Db":"C#","Eb":"D#","Fb":"E","Gb":"F#",
    "Ab":"G#","Bb":"A#","Cb":"B"
}


# ──────────────────────────────────────────────
# 2. resolve_target  (arrangement.py에서 호출)
# ──────────────────────────────────────────────
def resolve_target(instrument_id: int) -> str:
    """MIDI program 번호(0~128) → INSTRUMENT_GROUPS 키.

    Args:
        instrument_id: MIDI program 번호 (0~128)

    Returns:
        INSTRUMENT_GROUPS 딕셔너리의 키 (예: "violin", "woodwind")

    Raises:
        ValueError: 지원하지 않는 악기 ID
    """
    rep = PROGRAM_TO_REP.get(instrument_id, instrument_id)
    group_name = _REP_TO_GROUP.get(rep)
    if group_name is not None:
        return group_name
    raise ValueError(f"지원하지 않는 악기 ID: {instrument_id}")


# ──────────────────────────────────────────────
# 3. Vocabulary  (682 토큰)
# ──────────────────────────────────────────────
def build_v5_vocab(actual_vocab_size: int = 682):
    vocab = {}
    def add(prefix, r):
        for i in r: vocab[f"{prefix}{i}"] = len(vocab)
    for t in ["PAD","BOS","EOS","SEP","PIECE_START","PIECE_END",
              "BAR_START","BAR_END","PHRASE_END","<PRE>","<SUF>","<MID>"]:
        vocab[t] = len(vocab)
    for g in ["CLASSICAL","JAZZ","POP","ROCK","ELECTRONIC","FOLK","UNKNOWN"]:
        vocab[f"GENRE_{g}"] = len(vocab)
    roots = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
    for r in roots:
        for m in [":maj",":min"]: vocab[f"KEY_{r}{m}"] = len(vocab)
    vocab["KEY_NONE"] = len(vocab)
    # vocab_size가 682 이상일 때만 기존 TARGET_ 토큰 3종 포함
    if actual_vocab_size >= 682:
        for p in [40, 68, 73]: vocab[f"TARGET_{p}"] = len(vocab)
    for m in ["4:4","3:4","2:4","6:8","12:8","OTHER"]:
        vocab[f"METER_{m}"] = len(vocab)
    add("DENSITY_", range(1,6))
    add("INST=",    range(129))
    for a in ["ART_NORMAL","ART_LEGATO","ART_VIBRATO","ART_STACCATO"]:
        vocab[a] = len(vocab)
    add("EXPR_",  range(32))
    add("TIME=",  range(96))
    add("PITCH=", range(128))
    add("DUR=",   range(1,193))
    add("VEL=",   range(32))
    for w in ["melodic","epic","calm","fast","slow","sad","happy",
              "piano","strings","orchestra","cinematic"]:
        vocab[f"TEXT_{w}"] = len(vocab)
    return vocab


# ──────────────────────────────────────────────
# 4. load_model  (model_registry.py에서 호출)
# ──────────────────────────────────────────────
def load_model(ckpt_path: str, vocab_size: int, vocab: dict, device: str):
    """모델 로드. model_registry.py의 _load_single_model()에서 호출.

    Args:
        ckpt_path:  체크포인트 디렉토리 경로 (model.safetensors 또는 pytorch_model.bin 포함)
        vocab_size: build_v5_vocab()의 len(vocab)
        vocab:      build_v5_vocab() 결과
        device:     "cuda" 또는 "cpu"

    Returns:
        eval 모드로 설정된 모델 (device로 이동 완료)

    Raises:
        FileNotFoundError: 체크포인트 파일 없음
    """
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
        raise FileNotFoundError(f"체크포인트 없음: {ckpt_path}")

    model.config.use_cache = False
    model.eval()
    model.to(_device)

    # RTX 4090 / T4 TF32 가속
    # (transformers Dynamo 충돌 이슈로 인해 torch.compile 대신 네이티브 TF32 사용)
    torch.set_float32_matmul_precision("high")
    logger.info("TF32 하드웨어 가속 활성화 (torch.compile 비활성화)")

    return model


# ──────────────────────────────────────────────
# 5. MIDI → 마디 토큰
# ──────────────────────────────────────────────
def midi_to_bar_tokens(midi_path, genre, vocab):
    pm          = pretty_midi.PrettyMIDI(midi_path)
    res         = pm.resolution
    timeline    = defaultdict(lambda: defaultdict(list))
    key_changes = sorted(pm.key_signature_changes,  key=lambda x: x.time)
    key_times   = [k.time for k in key_changes]
    ts_changes  = sorted(pm.time_signature_changes, key=lambda x: x.time)
    ts_times    = [t.time for t in ts_changes]
    tempo_times, tempos = pm.get_tempo_changes()

    for inst in pm.instruments:
        rep = 128 if inst.is_drum else PROGRAM_TO_REP.get(inst.program)
        if rep is None or inst.program in DROP_SET:
            continue

        notes = sorted(inst.notes, key=lambda x: (x.start, x.pitch))
        last_end_tick = -1

        for i, n in enumerate(notes):
            note_dur  = n.end - n.start
            tempo_idx = max(0, bisect.bisect_right(tempo_times, n.start) - 1)
            # bpm=0 방어
            bpm       = tempos[tempo_idx] if len(tempos) > 0 and tempos[tempo_idx] > 0 else 120.0
            s_per_beat = 60.0 / bpm
            dur_tick  = max(1, min(192, round((note_dur / s_per_beat) * 24)))

            ts_idx = max(0, bisect.bisect_right(ts_times, n.start) - 1)
            if ts_idx < len(ts_changes):
                ts            = ts_changes[ts_idx]
                mkey          = f"{ts.numerator}:{ts.denominator}"
                meter_tok     = vocab.get(f"METER_{mkey}", vocab["METER_OTHER"])
                beats_per_bar = ts.numerator
            else:
                meter_tok     = vocab["METER_OTHER"]
                beats_per_bar = 4

            k_idx = bisect.bisect_right(key_times, n.start) - 1
            if 0 <= k_idx < len(key_changes):
                ks    = key_changes[k_idx]
                root  = ks.key_number % 12
                mode  = "maj" if ks.key_number < 12 else "min"
                roots = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
                rname = FLAT_TO_SHARP.get(roots[root], roots[root])
                key_tok = vocab.get(f"KEY_{rname}:{mode}", vocab["KEY_NONE"])
            else:
                key_tok = vocab["KEY_NONE"]

            bar_idx        = int(pm.time_to_tick(n.start) // (res * beats_per_bar))
            bar_start_tick = bar_idx * res * beats_per_bar
            rel_tick       = pm.time_to_tick(n.start) - bar_start_tick
            time_tok       = vocab[f"TIME={min(95, rel_tick * 96 // (res * beats_per_bar))}"]

            n_start_tick  = pm.time_to_tick(n.start)
            is_phrase_end = (last_end_tick > 0 and
                             (n_start_tick - last_end_tick) >= res)

            nxt_start    = notes[i+1].start if i+1 < len(notes) else n.end + 1
            legato_ratio = note_dur / max(nxt_start - n.start, 1e-6)
            if dur_tick <= 2:          art_tok = vocab["ART_STACCATO"]
            elif nxt_start == n.start: art_tok = vocab["ART_NORMAL"]
            elif legato_ratio > 0.95:  art_tok = vocab["ART_LEGATO"]
            else:                      art_tok = vocab["ART_NORMAL"]

            expr_tok = vocab[f"EXPR_{min(31, n.velocity * 32 // 128)}"]
            vel_tok  = vocab[f"VEL={min(31, n.velocity * 32 // 128)}"]
            inst_tok = vocab[f"INST={rep}"]

            timeline[bar_idx][rep].append(
                (time_tok, inst_tok, art_tok, expr_tok,
                 n.pitch, dur_tick, vel_tok,
                 meter_tok, key_tok, is_phrase_end))
            last_end_tick = pm.time_to_tick(n.end)

    header      = [vocab["PIECE_START"], vocab[f"GENRE_{genre}"]]
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
        meter_tok = vocab.get(f"METER_{mkey}", vocab["METER_OTHER"])

        k_idx = bisect.bisect_right(key_times, bar_time) - 1
        if 0 <= k_idx < len(key_changes):
            ks    = key_changes[k_idx]
            root  = ks.key_number % 12
            mode  = "maj" if ks.key_number < 12 else "min"
            roots = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]
            rname = FLAT_TO_SHARP.get(roots[root], roots[root])
            key_tok = vocab.get(f"KEY_{rname}:{mode}", vocab["KEY_NONE"])
        else:
            key_tok = vocab["KEY_NONE"]

        if bar_idx in timeline:
            bar_data    = timeline[bar_idx]
            total_notes = sum(len(v) for v in bar_data.values())
            density     = min(5, max(1, total_notes // 4))
            btoks = [vocab["BAR_START"], key_tok, meter_tok, vocab[f"DENSITY_{density}"]]
            bar_notes = []
            for p in bar_data.keys():
                for (tt, it, at, et, pitch, dt, vt, _, _, phrase_end) in bar_data[p]:
                    bar_notes.append((tt - vocab["TIME=0"], it, phrase_end,
                                      it, at, et, tt, pitch, dt, vt))
            bar_notes.sort(key=lambda x: (x[0], x[1]))
            for (_, _, phrase_end, it, at, et, tt, pitch, dt, vt) in bar_notes:
                if phrase_end:
                    btoks.append(vocab["PHRASE_END"])
                btoks += [it, at, et, tt, vocab[f"PITCH={pitch}"],
                          vocab[f"DUR={dt}"], vt]
            btoks.append(vocab["BAR_END"])
        else:
            btoks = [vocab["BAR_START"], key_tok, meter_tok,
                     vocab["DENSITY_1"], vocab["BAR_END"]]

        bar_tokens[bar_idx]  = btoks
        accumulated_ticks   += res * beats

    return header, bar_tokens, max_bar, pm


# ──────────────────────────────────────────────
# 6. 컨텍스트 트리밍
# ──────────────────────────────────────────────
def trim_context(context, header, vocab, max_tokens=1748):
    if len(context) <= max_tokens:
        return context
    overflow     = context[-max_tokens:]
    bar_start_id = vocab["BAR_START"]
    first_bar    = next((i for i, t in enumerate(overflow) if t == bar_start_id), 0)
    # header는 앞에 한 번만, overflow에서 첫 BAR_START 이전을 잘라냄
    return list(header) + overflow[first_bar:]


# ──────────────────────────────────────────────
# 7. 시작 무음 구간 감지
# ──────────────────────────────────────────────

# 빈 마디 토큰 수: BAR_START + key_tok + meter_tok + DENSITY_1 + BAR_END = 5
EMPTY_BAR_TOKEN_COUNT = 5

def _get_first_note_bar(bar_tokens):
    for bar_idx in sorted(bar_tokens.keys()):
        if len(bar_tokens[bar_idx]) > EMPTY_BAR_TOKEN_COUNT:
            return bar_idx
    return 0


# ──────────────────────────────────────────────
# 8. 끝부분 페이드아웃 패널티
# ──────────────────────────────────────────────
def _get_rest_penalty(win_start, max_bar, base_penalty, fade_bars):
    bars_remaining = max_bar - win_start
    if bars_remaining < fade_bars:
        ratio  = 1.0 - (bars_remaining / fade_bars)
        factor = 1.0 + ratio * 3.0
        return base_penalty * factor
    return base_penalty


# ──────────────────────────────────────────────
# 9. 슬라이딩 윈도우 생성 (generate_for_target)
# ──────────────────────────────────────────────
@torch.no_grad()
def generate_for_target(
    model, header, bar_tokens, max_bar,
    target_prog, pitch_min, pitch_max,
    window_bars, context_bars,
    temperature, top_p,
    vocab, vocab_r, source_pm, device,
    rest_penalty=1.5, fade_bars=8,
    progress_hook=None,
):
    SEQ_LEN = 2048
    MAX_CTX = 1748   # SEQ_LEN - 300

    all_notes      = []
    gen_bar_tokens = {}

    VEL_IDS        = {vocab[f"VEL={i}"] for i in range(32)}
    INST_TARGET_ID = vocab[f"INST={target_prog}"]
    BAR_START_ID   = vocab["BAR_START"]
    PIECE_END_ID   = vocab["PIECE_END"]
    EOS_ID         = vocab["EOS"]
    # CR-06: TIME 토큰 단조증가 강제를 위한 ID 배열
    TIME_IDS       = [vocab[f"TIME={i}"] for i in range(96)]

    # pitch 마스크: 루프 밖에서 한 번만 계산
    p0         = vocab["PITCH=0"]
    pitch_mask = torch.zeros(len(vocab), device=device)
    for pitch in range(128):
        if pitch < pitch_min or pitch > pitch_max:
            pitch_mask[p0 + pitch] = -1e9

    first_note_bar = _get_first_note_bar(bar_tokens)
    total_windows  = (max_bar // window_bars) + 1

    logger.info(f"[{target_prog}] 총 {max_bar+1}마디 / "
                f"윈도우 {window_bars}마디 → {total_windows}번 생성")

    for win_idx in range(total_windows):
        win_start = win_idx * window_bars
        win_end   = min(win_start + window_bars - 1, max_bar)
        if win_start > max_bar: break
        if win_end < first_note_bar: continue

        if progress_hook is not None:
            pct = int(20 + (win_idx / total_windows) * 60)
            progress_hook(pct)

        cur_penalty = _get_rest_penalty(win_start, max_bar, rest_penalty, fade_bars)
        ctx_start   = max(0, win_start - context_bars)

        # 컨텍스트 조립: 원본 + 생성 토큰을 마디 단위로 interleave
        context = list(header)
        for b in range(ctx_start, win_start):
            context += bar_tokens.get(b, [])
            if b in gen_bar_tokens:
                context += gen_bar_tokens[b]
        for b in range(win_start, win_end + 1):
            context += bar_tokens.get(b, [])

        context   = trim_context(context, header, vocab, MAX_CTX)
        input_ids = torch.tensor([context], dtype=torch.long, device=device)

        out = model(input_ids=input_ids, use_cache=True)
        pkv = out.past_key_values

        # 시작 프롬프트 주입 (BAR_START + INST=target)
        gen_toks = []
        for tok in [BAR_START_ID, INST_TARGET_ID]:
            t_in = torch.tensor([[tok]], dtype=torch.long, device=device)
            out  = model(input_ids=t_in, past_key_values=pkv, use_cache=True)
            pkv  = out.past_key_values
            gen_toks.append(tok)

        bar_count      = 1
        target_playing = True
        last_time_val  = -1   # CR-06: 마디 내 마지막 TIME 토큰 값 추적
        cur_in = torch.tensor([[gen_toks[-1]]], dtype=torch.long, device=device)

        for _ in range(1024):
            out    = model(input_ids=cur_in, past_key_values=pkv, use_cache=True)
            pkv    = out.past_key_values
            logits = out.logits[0, -1, :].float()

            logits += pitch_mask

            # CR-06: 과거 TIME 토큰 차단 — 모델 환각으로 시간이 역행하는 것을 방지
            if last_time_val >= 0:
                for tid in TIME_IDS[:last_time_val + 1]:
                    logits[tid] = -1e9

            if target_playing:
                logits[INST_TARGET_ID] = -1e9
            else:
                logits[INST_TARGET_ID] -= cur_penalty

            logits = logits / max(temperature, 1e-8)
            probs  = torch.softmax(logits, dim=-1)

            s_probs, s_idx = torch.sort(probs, descending=True)
            cumsum = torch.cumsum(s_probs, dim=0)
            cutoff = (cumsum - s_probs > top_p).nonzero()
            if len(cutoff): s_probs[cutoff[0].item():] = 0
            s_probs /= s_probs.sum().clamp(min=1e-8)

            next_tok = s_idx[torch.multinomial(s_probs, 1)].item()
            gen_toks.append(next_tok)

            # 상태 업데이트
            if next_tok == INST_TARGET_ID:
                target_playing = True
            elif target_playing and next_tok in VEL_IDS:
                target_playing = False

            if next_tok == BAR_START_ID:
                bar_count += 1
                target_playing = False
                last_time_val  = -1   # 마디 전환 시 TIME 추적 리셋
                if bar_count > window_bars: break
            elif next_tok in (PIECE_END_ID, EOS_ID):
                break

            # CR-06: TIME 토큰이면 last_time_val 갱신
            if next_tok in TIME_IDS:
                last_time_val = TIME_IDS.index(next_tok)

            cur_in = torch.tensor([[next_tok]], dtype=torch.long, device=device)

        # 생성 토큰 → 마디별 분리 (다음 윈도우 히스토리)
        cur_bar_toks, cur_bar_num = [], win_start
        for tok in gen_toks:
            if vocab_r.get(tok, "") == "BAR_START":
                if cur_bar_toks:
                    gen_bar_tokens[cur_bar_num] = cur_bar_toks
                    cur_bar_num += 1
                cur_bar_toks = [tok]
            else:
                cur_bar_toks.append(tok)
        if cur_bar_toks:
            gen_bar_tokens[cur_bar_num] = cur_bar_toks

        win_notes = decode_tokens(
            gen_toks, source_pm, target_prog,
            bar_offset=win_start, win_start=win_start, win_end=win_end,
            vocab_r=vocab_r)
        all_notes.extend(win_notes)
        logger.info(f"윈도우 {win_idx+1}/{total_windows} "
                    f"(bar {win_start}~{win_end}): {len(win_notes)}노트")

    return all_notes


# ──────────────────────────────────────────────
# 10. 토큰 → 노트 디코딩 (decode_tokens)
# ──────────────────────────────────────────────
def decode_tokens(tokens, source_pm, target_prog,
                  bar_offset, win_start, win_end, vocab_r):
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
    # 첫 BAR_START에서 bar_offset으로 올바르게 증가하도록 -1에서 시작
    bar_idx  = bar_offset - 1
    cur_inst = cur_time_tok = cur_pitch = cur_dur = cur_vel = None

    for tok in tokens:
        name = vocab_r.get(tok, "?")
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

                b_tick    = bar_tick_map.get(bar_idx, 0)
                b_time    = source_pm.tick_to_time(b_tick)
                ts_idx    = max(0, bisect.bisect_right(ts_times, b_time) - 1)
                bpb       = ts_changes[ts_idx].numerator if ts_idx < len(ts_changes) else 4
                bar_ticks = res * bpb
                abs_tick  = b_tick + cur_time_tok * bar_ticks // 96
                start_sec = source_pm.tick_to_time(abs_tick)

                t_idx   = max(0, bisect.bisect_right(tempo_times, start_sec) - 1)
                # bpm=0 방어
                bpm     = tempos[t_idx] if len(tempos) > 0 and tempos[t_idx] > 0 else 120.0
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
# 11. 후처리 (postprocess)
# ──────────────────────────────────────────────

# 단선율 악기 목록 — 이 악기에만 모노포니 강제 적용
# keyboard, guitar, ensemble, organ, synth 등 화음/패드 악기에는 적용하지 않음
MONOPHONIC_INSTRUMENTS = {"bass", "saxophone", "woodwind", "violin", "brass"}

def postprocess(notes, pitch_min, pitch_max, target_name=None):
    # 1. 음역 클리핑
    notes = [n for n in notes if pitch_min <= n["pitch"] <= pitch_max]

    # 2. 비정상적으로 긴 음표 제한
    for n in notes:
        if n["end"] - n["start"] > 4.0:
            n["end"] = n["start"] + 4.0

    # 3. 초단음 제거
    notes = [n for n in notes if (n["end"] - n["start"]) >= 0.05]

    # 4. 단선율 악기에만 모노포니 강제
    #    keyboard, guitar, ensemble 등 화음 악기는 건드리지 않음
    if target_name in MONOPHONIC_INSTRUMENTS:
        notes = sorted(notes, key=lambda x: x["start"])
        mono  = []
        for n in notes:
            if mono and n["start"] < mono[-1]["end"]:
                mono[-1]["end"] = n["start"]
                if mono[-1]["end"] - mono[-1]["start"] < 0.05:
                    mono.pop()
            mono.append(n)
        notes = mono

    # 5. 윈도우 경계 갭 채우기 (30ms 미만)
    LEGATO_GAP = 0.03
    for i in range(len(notes) - 1):
        gap = notes[i+1]["start"] - notes[i]["end"]
        if 0 < gap < LEGATO_GAP:
            notes[i]["end"] = notes[i+1]["start"]

    # 6. 큰 도약 완화 (옥타브 초과)
    MAX_INTERVAL = 12
    for i in range(1, len(notes)):
        interval = notes[i]["pitch"] - notes[i-1]["pitch"]
        if abs(interval) > MAX_INTERVAL:
            if interval > 0: notes[i]["pitch"] -= 12
            else:            notes[i]["pitch"] += 12
            if not (pitch_min <= notes[i]["pitch"] <= pitch_max):
                if interval > 0: notes[i]["pitch"] += 12
                else:            notes[i]["pitch"] -= 12

    # 7. 최종 재확인
    notes = [n for n in notes if pitch_min <= n["pitch"] <= pitch_max]
    notes = [n for n in notes if (n["end"] - n["start"]) >= 0.05]
    return notes


# ──────────────────────────────────────────────
# 12. MIDI 저장 (save_midi)
# ──────────────────────────────────────────────
def save_midi(notes, source_pm, output_path, target_prog, target_name,
              original_song_path=None,
              actual_instrument_name=None, actual_midi_program=None):
    """
    원본 미디 파일(`original_song_path`)에 AI가 생성한 노트들만 새로운 전용 트랙으로
    덧붙여 저장(Append)합니다. 이렇게 하면 원본 트랙들의 Jitter 발생이나 SysEx 메타데이터
    손실을 100% 방지할 수 있습니다.

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
        # source_pm.time_to_tick은 매핑본(복사본)이 가진 템포 맵대로 초를 틱으로 반환시킴
        # 이 매핑본은 원본과 템포가 동일하므로 원본 미디의 틱으로 완벽하게 일치됨.
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


# ──────────────────────────────────────────────
# 13. 공개 API: run_arrangement
# ──────────────────────────────────────────────
def run_arrangement(
    song_path:     str,
    target:        str,
    genre:         str,
    temperature:   float,
    pitch_min,     pitch_max,
    output_path:   str,
    model, vocab, vocab_r, device,
    progress_hook=None,
    original_song_path: str = None,
    actual_instrument_name: str = None,
    actual_midi_program: int = None,
) -> str:
    """편곡 추론 실행. 결과 MIDI 경로 반환."""
    # 고정 하이퍼파라미터
    window_bars    = 8
    context_bars   = 8
    top_p          = 0.95
    rest_penalty   = 1.5
    fade_bars      = 8
    seed           = 42

    # 입력 검증
    if target not in INSTRUMENT_GROUPS:
        raise ValueError(f"지원하지 않는 target: '{target}'. "
                         f"가능한 값: {list(INSTRUMENT_GROUPS.keys())}")
    if not os.path.exists(song_path):
        raise FileNotFoundError(f"입력 파일 없음: {song_path}")

    _device = torch.device(device) if isinstance(device, str) else device

    random.seed(seed)
    torch.manual_seed(seed)

    cfg         = INSTRUMENT_GROUPS[target]
    target_prog = cfg["representative"]
    pitch_min   = pitch_min if pitch_min is not None else cfg["pitch_min"]
    pitch_max   = pitch_max if pitch_max is not None else cfg["pitch_max"]

    logger.info(f"입력 MIDI 토크나이징: {song_path}")
    header, bar_tokens, max_bar, source_pm = midi_to_bar_tokens(song_path, genre, vocab)
    logger.info(f"총 마디 수: {max_bar + 1}")

    logger.info(f"생성 시작 (target={target}, genre={genre}, temp={temperature})")
    all_notes = generate_for_target(
        model, header, bar_tokens, max_bar,
        target_prog, pitch_min, pitch_max,
        window_bars, context_bars,
        temperature, top_p,
        vocab, vocab_r, source_pm, _device,
        rest_penalty=rest_penalty,
        fade_bars=fade_bars,
        progress_hook=progress_hook,
    )

    logger.info(f"디코딩 노트: {len(all_notes)}")
    all_notes = postprocess(all_notes, pitch_min, pitch_max, target_name=target)
    logger.info(f"후처리 후: {len(all_notes)}")

    if len(all_notes) == 0:
        raise RuntimeError("No notes generated.")

    save_midi(all_notes, source_pm, output_path, target_prog, target,
              original_song_path,
              actual_instrument_name=actual_instrument_name,
              actual_midi_program=actual_midi_program)

    return output_path