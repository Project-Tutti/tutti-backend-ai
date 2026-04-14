"""AI Core — 악기 그룹 및 매핑 상수 정의.

음악 도메인에 속하는 상수들을 모아놓은 모듈입니다.
인프라 변경 없이 AI 개발자가 자유롭게 수정할 수 있습니다.
"""

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

# 단선율 악기 목록 — 이 악기에만 모노포니 강제 적용
MONOPHONIC_INSTRUMENTS = {"bass", "saxophone", "woodwind", "violin", "brass"}
