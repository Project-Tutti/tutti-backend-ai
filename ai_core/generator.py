"""AI Core — 슬라이딩 윈도우 토큰 생성 모듈.

모델을 사용하여 마디별 토큰을 자동완성 합니다.
"""

import logging

import torch

from ai_core.tokenizer import trim_context, _get_first_note_bar
from ai_core.decoder import decode_tokens

logger = logging.getLogger(__name__)


def _get_rest_penalty(win_start, max_bar, base_penalty, fade_bars):
    bars_remaining = max_bar - win_start
    if bars_remaining < fade_bars:
        ratio  = 1.0 - (bars_remaining / fade_bars)
        factor = 1.0 + ratio * 3.0
        return base_penalty * factor
    return base_penalty


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

            # CR-06: 과거 TIME 토큰 차단 — 시간 역행 방지
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
