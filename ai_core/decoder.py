"""AI Core — 토큰 → 노트 디코딩 모듈.

생성된 토큰 시퀀스를 노트 딕셔너리 리스트로 변환합니다.
"""

import bisect


def decode_tokens(tokens, source_pm, target_prog,
                  bar_offset, win_start, win_end, vocab_r,
                  max_bar=None):
    res         = source_pm.resolution
    ts_changes  = sorted(source_pm.time_signature_changes, key=lambda x: x.time)
    ts_times    = [t.time for t in ts_changes]
    tempo_times, tempos = source_pm.get_tempo_changes()

    # 동적 범위: max_bar가 주어지면 필요한 만큼만 계산 (+ 여유분)
    bar_limit = (max_bar + 50) if max_bar is not None else 2000
    bar_tick_map = {}
    acc = 0
    for b in range(bar_limit):
        bar_tick_map[b] = acc
        bt  = source_pm.tick_to_time(acc)
        idx = max(0, bisect.bisect_right(ts_times, bt) - 1)
        bpb = ts_changes[idx].numerator if ts_changes and idx < len(ts_changes) else 4
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

                b_tick = bar_tick_map.get(bar_idx)
                if b_tick is None:
                    # bar_limit을 넘어간 모델 길이 폭주의 경우, 마지막 바 기준으로 대략 누적
                    last_b_tick = bar_tick_map.get(bar_limit - 1, 0) if bar_limit > 0 else 0
                    b_tick = last_b_tick + res * 4 * max(0, bar_idx - (bar_limit - 1))

                b_time    = source_pm.tick_to_time(b_tick)
                ts_idx    = max(0, bisect.bisect_right(ts_times, b_time) - 1)
                bpb       = ts_changes[ts_idx].numerator if ts_changes and ts_idx < len(ts_changes) else 4
                bar_ticks = res * bpb
                abs_tick  = b_tick + cur_time_tok * bar_ticks // 96
                start_sec = source_pm.tick_to_time(abs_tick)

                t_idx   = max(0, bisect.bisect_right(tempo_times, start_sec) - 1)
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
