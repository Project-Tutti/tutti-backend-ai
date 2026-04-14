"""AI Core — 생성 후처리 모듈.

음역 클리핑, 모노포니 강제, 레가토 갭 채우기 등 후처리 로직.
"""

from ai_core.constants import MONOPHONIC_INSTRUMENTS


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
