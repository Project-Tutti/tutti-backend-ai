"""AI Core — Vocabulary 구축 모듈.

682-토큰 어휘를 구축하는 함수를 제공합니다.
"""


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
