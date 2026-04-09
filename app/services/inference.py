"""
inference.py
Qwen2.5-0.5B 기반 악기 파트 생성 추론 파이프라인 (서버용)

새 매핑 기준 13개 악기 그룹 지원.
입력 MIDI 전체를 컨텍스트로 받아 새 악기 파트를 추가 생성.

원본: local_docs/inference_new.py (CLI 기반) → 서버 서비스로 변환
"""

import os
import copy
import bisect
import random
import logging

import torch
import torch.nn as nn
import pretty_midi
from collections import defaultdict
from transformers import AutoConfig, AutoModelForCausalLM

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# 악기 그룹 정의 (새 매핑 13그룹)
# ──────────────────────────────────────────────
INSTRUMENT_GROUPS = {
    "drum":       {"representative": 128, "is_drum": True,
                   "pitch_min": 35,  "pitch_max": 81},
    "keyboard":   {"representative": 0,   "is_drum": False,
                   "pitch_min": 21,  "pitch_max": 108},
    "organ":      {"representative": 16,  "is_drum": False,
                   "pitch_min": 36,  "pitch_max": 96},
    "mallet":     {"representative": 12,  "is_drum": False,
                   "pitch_min": 48,  "pitch_max": 96},
    "guitar":     {"representative": 25,  "is_drum": False,
                   "pitch_min": 40,  "pitch_max": 88},
    "dist_guitar":{"representative": 30,  "is_drum": False,
                   "pitch_min": 40,  "pitch_max": 88},
    "bass":       {"representative": 33,  "is_drum": False,
                   "pitch_min": 28,  "pitch_max": 67},
    "violin":     {"representative": 40,  "is_drum": False,
                   "pitch_min": 55,  "pitch_max": 103},
    "woodwind":   {"representative": 73,  "is_drum": False,
                   "pitch_min": 60,  "pitch_max": 96},
    "saxophone":  {"representative": 65,  "is_drum": False,
                   "pitch_min": 49,  "pitch_max": 80},
    "synth":      {"representative": 81,  "is_drum": False,
                   "pitch_min": 36,  "pitch_max": 96},
    "brass":      {"representative": 56,  "is_drum": False,
                   "pitch_min": 52,  "pitch_max": 82},
    "ensemble":   {"representative": 48,  "is_drum": False,
                   "pitch_min": 36,  "pitch_max": 96},
}

ALL_TARGET_NAMES = list(INSTRUMENT_GROUPS.keys())

# drop_list (토크나이징 시 제외)
DROP_SET = {47, 55, 109, 113, 115, 116, 117, 118, 119, 120,
            121, 122, 123, 124, 125, 126, 127}

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

FLAT_TO_SHARP = {
    "Db":"C#","Eb":"D#","Fb":"E",
    "Gb":"F#","Ab":"G#","Bb":"A#","Cb":"B"
}


# ──────────────────────────────────────────────
# resolve_target: program number → group name
# ──────────────────────────────────────────────
def resolve_target(instrument_id: int) -> str:
    """MIDI program number → INSTRUMENT_GROUPS 키.

    Args:
        instrument_id: MIDI program 번호 (0~128)

    Returns:
        INSTRUMENT_GROUPS 딕셔너리의 키 (예: "violin", "brass")

    Raises:
        ValueError: 지원하지 않는 악기 ID
    """
    rep = PROGRAM_TO_REP.get(instrument_id, instrument_id)
    group_name = _REP_TO_GROUP.get(rep)
    if group_name is not None:
        return group_name
    raise ValueError(f"지원하지 않는 악기 ID: {instrument_id}")


# ──────────────────────────────────────────────
# Vocabulary
# ──────────────────────────────────────────────
def build_v5_vocab():
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
    for p in [40, 68, 73]: vocab[f"TARGET_{p}"] = len(vocab)
    for m in ["4:4","3:4","2:4","6:8","12:8","OTHER"]:
        vocab[f"METER_{m}"] = len(vocab)
    add("DENSITY_", range(1, 6))
    add("INST=",    range(129))
    for a in ["ART_NORMAL","ART_LEGATO","ART_VIBRATO","ART_STACCATO"]:
        vocab[a] = len(vocab)
    add("EXPR_",  range(32))
    add("TIME=",  range(96))
    add("PITCH=", range(128))
    add("DUR=",   range(1, 193))
    add("VEL=",   range(32))
    for w in ["melodic","epic","calm","fast","slow","sad","happy",
              "piano","strings","orchestra","cinematic"]:
        vocab[f"TEXT_{w}"] = len(vocab)
    return vocab


# ──────────────────────────────────────────────
# 모델 로드
# ──────────────────────────────────────────────
def load_model(ckpt_path, vocab_size, vocab, device):
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
    model.to(device)
    
    # RTX 4090 하드웨어 가속(TF32) 활성화
    # (transformers Dynamo 충돌 이슈로 인해 torch.compile 대신 네이티브 TF32 사용)
    torch.set_float32_matmul_precision('high')
    logger.info("⚡ TF32 하드웨어 코어 가속 활성화 (torch.compile 비활성화)")
    
    return model


# ──────────────────────────────────────────────
# MIDI → bar_tokens
# ──────────────────────────────────────────────
def midi_to_bar_tokens(midi_path, genre, vocab, vocab_r):
    pm  = pretty_midi.PrettyMIDI(midi_path)
    res = pm.resolution
    timeline    = defaultdict(lambda: defaultdict(list))
    key_changes = sorted(pm.key_signature_changes,  key=lambda x: x.time)
    key_times   = [k.time for k in key_changes]
    ts_changes  = sorted(pm.time_signature_changes, key=lambda x: x.time)
    ts_times    = [t.time for t in ts_changes]
    tempo_times, tempos = pm.get_tempo_changes()

    for inst in pm.instruments:
        if inst.is_drum:
            rep = 128
        else:
            if inst.program in DROP_SET:
                continue
            rep = PROGRAM_TO_REP.get(inst.program, None)
            if rep is None:
                continue

        notes = sorted(inst.notes, key=lambda x: (x.start, x.pitch))
        last_end_tick = -1

        for i, n in enumerate(notes):
            note_dur  = n.end - n.start
            tempo_idx = max(0, bisect.bisect_right(tempo_times, n.start) - 1)
            bpm       = tempos[tempo_idx] if len(tempos) > 0 else 120.0
            s_per_beat = 60.0 / bpm
            dur_tick  = max(1, min(192, round((note_dur / s_per_beat) * 24)))

            ts_idx = max(0, bisect.bisect_right(ts_times, n.start) - 1)
            if ts_idx < len(ts_changes):
                ts            = ts_changes[ts_idx]
                mkey          = f"{ts.numerator}:{ts.denominator}"
                meter_tok     = vocab.get(f"METER_{mkey}", vocab["METER_OTHER"])
                beats_per_bar = ts.numerator
            else:
                meter_tok, beats_per_bar = vocab["METER_OTHER"], 4

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
            is_phrase_end = last_end_tick > 0 and (n_start_tick - last_end_tick) >= res

            nxt_start    = notes[i+1].start if i+1 < len(notes) else n.end + 1
            legato_ratio = note_dur / max(nxt_start - n.start, 1e-6)
            if dur_tick <= 2:           art_tok = vocab["ART_STACCATO"]
            elif nxt_start == n.start:  art_tok = vocab["ART_NORMAL"]
            elif legato_ratio > 0.95:   art_tok = vocab["ART_LEGATO"]
            else:                       art_tok = vocab["ART_NORMAL"]

            expr_tok = vocab[f"EXPR_{min(31, n.velocity * 32 // 128)}"]
            vel_tok  = vocab[f"VEL={min(31, n.velocity * 32 // 128)}"]
            inst_tok = vocab[f"INST={rep}"]   # ← representative 사용

            timeline[bar_idx][rep].append(
                (time_tok, inst_tok, art_tok, expr_tok,
                 n.pitch, dur_tick, vel_tok,
                 meter_tok, key_tok, is_phrase_end))
            last_end_tick = pm.time_to_tick(n.end)

    header      = [vocab["PIECE_START"], vocab[f"GENRE_{genre}"]]
    final_beats = ts_changes[-1].numerator if ts_changes else 4
    max_bar     = int(pm.time_to_tick(pm.get_end_time()) // (res * final_beats))

    bar_tokens = {}
    accum_ticks = 0
    for bar_idx in range(max_bar + 1):
        bar_time = pm.tick_to_time(accum_ticks)
        ts_idx   = max(0, bisect.bisect_right(ts_times, bar_time) - 1)
        beats    = ts_changes[ts_idx].numerator if ts_idx < len(ts_changes) else 4
        mkey     = (f"{ts_changes[ts_idx].numerator}:{ts_changes[ts_idx].denominator}"
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
            all_notes = []
            for p in bar_data.keys():
                for (tt, it, at, et, pitch, dt, vt, _, _, phrase_end) in bar_data[p]:
                    all_notes.append((tt - vocab["TIME=0"], it, phrase_end,
                                      it, at, et, tt, pitch, dt, vt))
            all_notes.sort(key=lambda x: (x[0], x[1]))
            for (_, _, phrase_end, it, at, et, tt, pitch, dt, vt) in all_notes:
                if phrase_end: btoks.append(vocab["PHRASE_END"])
                btoks += [it, at, et, tt, vocab[f"PITCH={pitch}"],
                          vocab[f"DUR={dt}"], vt]
            btoks.append(vocab["BAR_END"])
        else:
            btoks = [vocab["BAR_START"], key_tok, meter_tok,
                     vocab["DENSITY_1"], vocab["BAR_END"]]

        bar_tokens[bar_idx] = btoks
        accum_ticks += res * beats

    return header, bar_tokens, max_bar, pm


# ──────────────────────────────────────────────
# 컨텍스트 트리밍
# ──────────────────────────────────────────────
def trim_context(context, header, max_tokens, vocab):
    if len(context) <= max_tokens:
        return context
    overflow  = context[-max_tokens:]
    first_bar = next((i for i, t in enumerate(overflow)
                      if t == vocab["BAR_START"]), 0)
    return list(header) + overflow[first_bar:]


# ──────────────────────────────────────────────
# 단일 타겟 슬라이딩 윈도우 생성
# ──────────────────────────────────────────────
@torch.no_grad()
def generate_for_target(
    model, header, bar_tokens, max_bar,
    target_name, pitch_min, pitch_max,
    window_bars, context_bars,
    temperature, top_p, max_new_tokens,
    vocab, vocab_r, source_pm, device
):
    cfg         = INSTRUMENT_GROUPS[target_name]
    target_prog = cfg["representative"]

    SEQ_LEN  = 2048
    MAX_CTX  = SEQ_LEN - 300
    all_notes         = []
    gen_bar_tokens    = {}   # 생성된 타겟 마디 토큰 (다음 윈도우 컨텍스트용)

    total_windows = (max_bar // window_bars) + 1
    logger.info(f"[{target_name}] 총 {max_bar+1}마디 / "
                f"윈도우 {window_bars}마디 → {total_windows}번 생성")

    TIME_IDS = torch.tensor([vocab[f"TIME={i}"] for i in range(96)], device=device)
    PITCH_IDS = torch.tensor([vocab[f"PITCH={i}"] for i in range(128)], device=device)
    VEL_IDS = torch.tensor([vocab[f"VEL={i}"] for i in range(32)], device=device)
    INST_TARGET_ID = vocab[f"INST={target_prog}"]
    BAR_START_ID = vocab["BAR_START"]
    PIECE_END_ID = vocab["PIECE_END"]
    EOS_ID = vocab["EOS"]

    pitch_mask = torch.full((len(vocab),), -1e9, device=device)
    pitch_mask[PITCH_IDS[pitch_min: pitch_max + 1]] = 0.0

    for win_idx in range(total_windows):
        win_start = win_idx * window_bars
        win_end   = min(win_start + window_bars - 1, max_bar)
        if win_start > max_bar:
            break

        ctx_start = max(0, win_start - context_bars)

        # 컨텍스트 조립 (optimized)
        context = list(header)
        for b in range(ctx_start, win_start):
            context += bar_tokens.get(b, []) + gen_bar_tokens.get(b, [])
        for b in range(win_start, win_end + 1):
            context += bar_tokens.get(b, [])

        context   = trim_context(context, header, MAX_CTX, vocab)
        input_ids = torch.tensor([context], dtype=torch.long, device=device)

        # KV cache로 컨텍스트 사전 인코딩
        out = model(input_ids=input_ids, use_cache=True)
        pkv = out.past_key_values
        gen_toks = [BAR_START_ID, INST_TARGET_ID]

        # 프롬프트 토큰들을 하나씩 넣어 pkv 갱신 처리
        for tok in gen_toks:
            t_in = torch.tensor([[tok]], dtype=torch.long, device=device)
            out  = model(input_ids=t_in, past_key_values=pkv, use_cache=True)
            pkv  = out.past_key_values

        bar_count = 1
        target_playing = True
        last_time_val = -1

        cur_in = torch.tensor([[gen_toks[-1]]], dtype=torch.long, device=device)

        for _ in range(max_new_tokens):
            out    = model(input_ids=cur_in, past_key_values=pkv, use_cache=True)
            pkv    = out.past_key_values
            logits = out.logits[0, -1, :].float()

            logits[PITCH_IDS] += pitch_mask[PITCH_IDS]
            if last_time_val >= 0: logits[TIME_IDS[:last_time_val + 1]] = -1e9
            if target_playing: logits[INST_TARGET_ID] = -1e9

            # nucleus sampling
            logits  = logits / max(temperature, 1e-8)
            probs   = torch.softmax(logits, dim=-1)
            s_probs, s_idx = torch.sort(probs, descending=True)
            cumsum  = torch.cumsum(s_probs, dim=0)
            cutoff  = (cumsum - s_probs > top_p).nonzero()
            if len(cutoff):
                s_probs[cutoff[0].item():] = 0
            s_probs /= s_probs.sum()
            next_tok = s_idx[torch.multinomial(s_probs, 1)].item()
            gen_toks.append(next_tok)

            if next_tok == INST_TARGET_ID:
                target_playing = True
                last_time_val = -1
            elif target_playing and (next_tok in VEL_IDS):
                target_playing = False
            elif next_tok == BAR_START_ID:
                bar_count += 1
                last_time_val = -1
                target_playing = False
                if bar_count > window_bars: break
            elif next_tok in [PIECE_END_ID, EOS_ID]:
                break
            
            if (next_tok >= TIME_IDS[0]) and (next_tok <= TIME_IDS[-1]):
                last_time_val = (next_tok - TIME_IDS[0]).item()

            cur_in = torch.tensor([[next_tok]], dtype=torch.long, device=device)

        # 생성 토큰 → 마디별 분리 (다음 윈도우 히스토리)
        cur_bar_toks = []
        cur_bar_num  = win_start
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

        # 토큰 → 노트 디코딩
        win_notes = decode_tokens(
            gen_toks, source_pm, target_prog,
            bar_offset=win_start, win_start=win_start, win_end=win_end,
            vocab_r=vocab_r)
        all_notes.extend(win_notes)
        logger.info(f"[{target_name}] 윈도우 {win_idx+1}/{total_windows} "
                    f"(bar {win_start}~{win_end}): {len(win_notes)}노트")

    return all_notes


# ──────────────────────────────────────────────
# 토큰 → 노트 디코딩
# ──────────────────────────────────────────────
def decode_tokens(tokens, source_pm, target_prog,
                  bar_offset, win_start, win_end, vocab_r):
    res         = source_pm.resolution
    ts_changes  = sorted(source_pm.time_signature_changes, key=lambda x: x.time)
    ts_times    = [t.time for t in ts_changes]
    tempo_times, tempos = source_pm.get_tempo_changes()

    # 마디 시작 tick 맵
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
# ──────────────────────────────────────────────
def postprocess(notes, pitch_min, pitch_max):
    # 1. 음역 클리핑
    notes = [n for n in notes if pitch_min <= n["pitch"] <= pitch_max]

    # 2. 비정상적으로 긴 음표 클리핑
    for n in notes:
        if n["end"] - n["start"] > 4.0:
            n["end"] = n["start"] + 4.0

    # 3. 너무 짧은 음표 제거
    notes = [n for n in notes if (n["end"] - n["start"]) >= 0.05]

    # 4. 슬라이딩 윈도우 경계 연결 보정
    notes = sorted(notes, key=lambda x: x["start"])
    for i in range(len(notes) - 1):
        gap = notes[i+1]["start"] - notes[i]["end"]
        if 0 < gap < 0.03:
            notes[i]["end"] = notes[i+1]["start"]

    # 5. 큰 도약 완화 (옥타브 초과 시 한 옥타브 조정)
    for i in range(1, len(notes)):
        interval = notes[i]["pitch"] - notes[i-1]["pitch"]
        if abs(interval) > 12:
            notes[i]["pitch"] += (-12 if interval > 0 else 12)
            if not (pitch_min <= notes[i]["pitch"] <= pitch_max):
                notes[i]["pitch"] += (12 if interval > 0 else -12)

    # 6. 최종 음역 재확인
    notes = [n for n in notes if pitch_min <= n["pitch"] <= pitch_max]
    notes = [n for n in notes if (n["end"] - n["start"]) >= 0.05]

    return notes


# ──────────────────────────────────────────────
# MIDI 저장
# ──────────────────────────────────────────────
def save_midi(notes, source_pm, output_path, target_prog, target_name):
    out_pm   = copy.deepcopy(source_pm)
    new_inst = pretty_midi.Instrument(
        program=target_prog if target_prog < 128 else 0,
        is_drum=(target_prog == 128),
        name=target_name)
    for n in notes:
        new_inst.notes.append(pretty_midi.Note(
            velocity=n["velocity"], pitch=n["pitch"],
            start=n["start"], end=n["end"]))
    out_pm.instruments.append(new_inst)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    out_pm.write(output_path)
    logger.info(f"저장: {output_path}  ({len(notes)} 노트)")


# ──────────────────────────────────────────────
# 공개 API: run_arrangement
# ──────────────────────────────────────────────
def run_arrangement(
    song_path: str,
    target: str,
    genre: str = "CLASSICAL",
    temperature: float = 1.0,
    pitch_min: int = None,
    pitch_max: int = None,
    output_path: str = None,
    model=None,
    vocab: dict = None,
    vocab_r: dict = None,
    device: str = "cuda",
) -> str:
    """편곡 추론 실행, 결과 MIDI 경로 반환.

    Args:
        song_path: 입력 MIDI 파일 경로
        target: INSTRUMENT_GROUPS 키 (예: "violin", "brass")
        genre: 장르 토큰 (CLASSICAL, JAZZ, POP, ROCK, ELECTRONIC, FOLK, UNKNOWN)
        temperature: 생성 다양성 (0.1~2.0)
        pitch_min: 음역 최솟값 (None이면 악기 기본값)
        pitch_max: 음역 최댓값 (None이면 악기 기본값)
        output_path: 결과 MIDI 저장 경로
        model: 사전 로드된 모델 (ModelRegistry에서 가져옴)
        vocab: 보캡 딕셔너리
        vocab_r: 역방향 보캡 (id → token name)
        device: 디바이스 ("cuda" or "cpu")

    Returns:
        결과 MIDI 파일 경로
    """
    # 고정 하이퍼파라미터
    window_bars    = 8
    context_bars   = 8
    top_p          = 0.95
    max_new_tokens = 1024
    seed           = 42

    random.seed(seed)
    torch.manual_seed(seed)

    # 악기 그룹 설정
    cfg = INSTRUMENT_GROUPS[target]
    pitch_min = pitch_min if pitch_min is not None else cfg["pitch_min"]
    pitch_max = pitch_max if pitch_max is not None else cfg["pitch_max"]
    logger.info(f"타겟: {target} (rep={cfg['representative']}, "
                f"pitch {pitch_min}~{pitch_max})")

    # MIDI 토큰화
    logger.info(f"입력 MIDI 토큰화: {song_path}")
    header, bar_tokens, max_bar, source_pm = midi_to_bar_tokens(
        song_path, genre, vocab, vocab_r)
    logger.info(f"총 마디 수: {max_bar + 1}")

    # 생성
    logger.info(f"생성 시작 [{target}] (genre={genre}, temp={temperature})")
    all_notes = generate_for_target(
        model, header, bar_tokens, max_bar,
        target, pitch_min, pitch_max,
        window_bars, context_bars,
        temperature, top_p, max_new_tokens,
        vocab, vocab_r, source_pm, device)

    logger.info(f"디코딩 노트: {len(all_notes)}")
    all_notes = postprocess(all_notes, pitch_min, pitch_max)
    logger.info(f"후처리 후: {len(all_notes)}")

    if not all_notes:
        raise RuntimeError(
            f"노트 없음 — temperature를 높이거나 입력 MIDI를 확인하세요. "
            f"(target={target}, genre={genre})"
        )

    # 저장
    save_midi(all_notes, source_pm, output_path, cfg["representative"], target)
    return output_path
