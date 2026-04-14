"""AI Core — MIDI → 마디 토큰 변환 모듈.

원본 MIDI 파일을 마디(Bar) 단위 토큰 시퀀스로 변환합니다.
"""

import bisect
import logging
from collections import defaultdict

import pretty_midi

from ai_core.constants import PROGRAM_TO_REP, DROP_SET, FLAT_TO_SHARP

logger = logging.getLogger(__name__)

# 빈 마디 토큰 수: BAR_START + key_tok + meter_tok + DENSITY_1 + BAR_END = 5
EMPTY_BAR_TOKEN_COUNT = 5


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


def trim_context(context, header, vocab, max_tokens=1748):
    if len(context) <= max_tokens:
        return context
    overflow     = context[-max_tokens:]
    bar_start_id = vocab["BAR_START"]
    first_bar    = next((i for i, t in enumerate(overflow) if t == bar_start_id), 0)
    return list(header) + overflow[first_bar:]


def _get_first_note_bar(bar_tokens):
    for bar_idx in sorted(bar_tokens.keys()):
        if len(bar_tokens[bar_idx]) > EMPTY_BAR_TOKEN_COUNT:
            return bar_idx
    return 0
