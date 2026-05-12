# inference.py 생성 품질 이슈 보고서

> 대상 파일: `app/services/inference.py`  
> 작성일: 2026-05-08  
> 목적: 생성 결과물 품질에 악영향을 줄 수 있는 로직에 대한 문제 분석 및 수정 제안

---

## 요약

| # | 위치 | 심각도 | 제목 | 조치 |
|---|------|--------|------|------|
| 1 | L818-828 | 🔴 높음 | 도약 완화 연쇄 왜곡 | 제거 또는 수정 필요 |
| 2 | L619-627 | 🟡 중간 | 헤더 토큰 누락으로 초반부 어색 | 수정 필요 |
| 3 | L335-338 | 🟡 중간 | Time Quantization 학습 일관성 확인 필요 | 확인 필요 |
| 4 | L680-692 | 🟢 낮음 | 윈도우 경계 바 토큰 범위 초과 | 가드 추가 권장 |
| 5 | L335, L362-363 | 🟡 중간 | 박자 변경 시 bar_idx 및 max_bar 불일치 | 수정 필요 (박자 변경 곡) |
| 6 | L352 vs L769 | 🟡 중간 | Velocity 인코딩/디코딩 비대칭 | 학습 코드 확인 필요 |

> 모든 이슈는 **모델 재학습 없이** `inference.py`만 수정하면 해결됩니다.

---

## Issue #1: 도약 완화 로직의 연쇄 왜곡 (L818-828)

**심각도: 🔴 높음**

### 현재 코드

```python
# postprocess() 내부, Step 6
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
```

### 의도

단선율 악기에서 1옥타브(12반음) 초과 도약이 발생하면, 한 옥타브 접어서 도약을 줄여 자연스러운 멜로디를 만든다.

### 문제점

#### 1-A. 연쇄 왜곡 (Cascading Distortion) — 가장 심각

`notes[i]`의 피치를 수정하면, 다음 반복(`i+1`)에서 **수정된** `notes[i]`가 기준점이 됩니다.
한 음이 옥타브 이동하면 그 다음 음과의 관계도 변하여, 원래는 정상이었던 음까지 연쇄적으로 옥타브 이동합니다.

**구체적 예시:**

```
원본 시퀀스:  C3(48) → C5(72) → B4(71) → C3(48)
의도된 멜로디: 낮은 음 → 높은 음 → 높은 음 유지 → 다시 낮은 음으로 복귀

Step 1: notes[1] - notes[0] = 72 - 48 = +24 (> 12)
        → C5(72) → C4(60)으로 이동
Step 2: notes[2] - notes[1] = 71 - 60 = +11 (≤ 12)
        → 변경 없음 (B4=71 유지)
Step 3: notes[3] - notes[2] = 48 - 71 = -23 (> 12)
        → C3(48) → C4(60)으로 이동

결과:  C3(48) → C4(60) → B4(71) → C4(60)
원본:  C3(48) → C5(72) → B4(71) → C3(48)
```

원래 `C3 → C5 → B4 → C3`이라는 **의도된 옥타브 전환**이 `C4 → C4 → B4 → C4`로 납작하게 눌려서 멜로디 윤곽이 완전히 파괴됩니다.

#### 1-B. 2옥타브 이상 도약 미처리

interval이 24 이상이면 12를 빼도 여전히 >12이므로 보정 효과 없음.

#### 1-C. AI 의도 파괴

모델은 이미 추론 시점에 `pitch_min ~ pitch_max` 마스킹(L641-644)을 적용받아 음역 내에서만 생성합니다.
모델이 의도적으로 생성한 큰 도약(옥타브 점프 등)을 후처리에서 강제 변경하면 음악적으로 어색한 음이 만들어집니다.

### 권장 수정

**Option A: 로직 제거 (강력 권장)**

```python
# Step 6 전체를 삭제 또는 주석 처리
# 모델이 생성한 피치 시퀀스를 그대로 존중
```

근거:
- 모델이 추론 시점에 이미 음역 마스킹(L641-644)을 받고 있어 극단적 도약이 드뭄
- 후처리 Step 7(L830-832)에서 음역 밖 음을 최종 필터링하므로 이중 안전장치 존재
- 학습 데이터에 없는 피치 패턴을 후처리로 만들어내면 오히려 품질 저하

**Option B: 연쇄 왜곡 방지 (유지해야 한다면)**

```python
if monophonic:
    MAX_INTERVAL = 12
    notes = sorted(notes, key=lambda x: x["start"])
    original_pitches = [n["pitch"] for n in notes]  # 원본 보존
    for i in range(1, len(notes)):
        interval = notes[i]["pitch"] - original_pitches[i-1]  # 수정된 값이 아닌 원본 기준
        if abs(interval) > MAX_INTERVAL:
            if interval > 0: notes[i]["pitch"] -= 12
            else:            notes[i]["pitch"] += 12
            if not (cfg["pitch_min"] <= notes[i]["pitch"] <= cfg["pitch_max"]):
                if interval > 0: notes[i]["pitch"] += 12
                else:            notes[i]["pitch"] -= 12
```

차이점: `notes[i-1]["pitch"]` 대신 `original_pitches[i-1]`을 사용하여, 이전 음의 수정이 다음 음의 판단에 영향을 주지 않도록 합니다.

### 검증 방법

1. 현재 코드로 monophonic 악기(flute, violin 등) 생성 → 결과 MIDI의 피치 시퀀스 기록
2. 도약 완화 로직 제거 후 동일 조건(seed=42)으로 생성 → 비교
3. 특히 옥타브 전환이 빈번한 곡에서 A/B 청취 비교

---

## Issue #2: 윈도우 시작 시 헤더 토큰 누락 (L619-627)

**심각도: 🟡 중간 — 초반부 어색함의 직접적 원인으로 추정**

### 현재 코드

```python
# generate_sliding_window() 내부
for tok in [VOCAB["BAR_START"], INST_TARGET_ID]:
    t_in = torch.tensor([[tok]], dtype=torch.long, device=device)
    out  = model(input_ids=t_in, past_key_values=pkv, use_cache=True)
    pkv  = out.past_key_values
    gen_toks.append(tok)

bar_count      = 1
target_playing = True
```

### 의도

각 윈도우의 생성을 시작할 때, `BAR_START`와 타겟 악기의 `INST` 토큰을 강제 주입하여 모델이 올바른 악기로 생성을 시작하도록 유도.

### 문제점

#### 2-A. 학습 분포와의 불일치

학습 데이터의 토크나이징(`midi_to_bar_tokens`, L390-401)에서 각 마디는 아래 순서로 토큰화됩니다:

```
BAR_START → KEY_C:maj → METER_4:4 → DENSITY_3 → INST=73 → ART → EXPR → TIME → PITCH → DUR → VEL → ...
```

하지만 추론 시 모델에게 주입되는 시퀀스는:

```
BAR_START → INST=73 → (자유 생성)
         ↑
   KEY, METER, DENSITY 누락
```

모델은 학습 중 `BAR_START` 뒤에 항상 `KEY → METER → DENSITY` 패턴을 경험했는데, 추론에서는 이를 건너뛰고 바로 `INST`를 받습니다. 이로 인해 **첫 번째 음의 피치/리듬 선택이 학습 분포에서 벗어날 수 있습니다.**

#### 2-B. 첫 윈도우의 이중 핸디캡

첫 윈도우(`win_idx=0`)는 추가적으로 불리한 조건이 겹칩니다:

```python
win_start = 0
ctx_start = max(0, 0 - 8) = 0   # → past_bar_list 비어있음
future_bars = 0                   # → future_bar_list 비어있음
```

결과적으로 모델에게 주어지는 전체 컨텍스트:

```
PIECE_START → GENRE_POP → [Bar 0~3 원곡] → BAR_START → INST=73 → ???
```

- 과거 생성 결과: 없음 (첫 윈도우)
- 미래 가이드: 없음 (future_bars=0)
- 마디 헤더(KEY/METER/DENSITY): 없음 (강제 주입에서 생략)

**가장 열악한 조건에서 곡의 첫 부분을 생성**하게 되므로, 초반부가 어색해지는 현상과 직접적으로 연결됩니다.

### 현상

- 생성 결과물의 **초반부(특히 첫 1~2 윈도우)**가 어색한 경향이 관찰됨
- 이후 윈도우로 갈수록 과거 생성 결과가 컨텍스트에 쌓이면서 품질이 안정화됨

### 권장 수정

현재 윈도우의 첫 번째 마디에서 KEY/METER/DENSITY 토큰을 추출하여 함께 주입합니다:

```python
# 변경 전
for tok in [VOCAB["BAR_START"], INST_TARGET_ID]:
    t_in = torch.tensor([[tok]], dtype=torch.long, device=device)
    out  = model(input_ids=t_in, past_key_values=pkv, use_cache=True)
    pkv  = out.past_key_values
    gen_toks.append(tok)

# 변경 후
first_bar = bar_tokens.get(win_start, [])
bar_header = [VOCAB["BAR_START"]]
for t in first_bar:
    name = VOCAB_R.get(t, "")
    if name.startswith("KEY_") or name.startswith("METER_") or name.startswith("DENSITY_"):
        bar_header.append(t)
bar_header.append(INST_TARGET_ID)

for tok in bar_header:
    t_in = torch.tensor([[tok]], dtype=torch.long, device=device)
    out  = model(input_ids=t_in, past_key_values=pkv, use_cache=True)
    pkv  = out.past_key_values
    gen_toks.append(tok)
```

이렇게 하면 모델이 보는 시퀀스가 학습 분포와 일치합니다:

```
... context ... → BAR_START → KEY_C:maj → METER_4:4 → DENSITY_3 → INST=73 → (자유 생성)
```

**참고:** `gen_toks`에 헤더 토큰들이 추가로 들어가지만, `decode_tokens()`는 `INST=`로 시작하는 토큰부터 노트로 파싱하므로 KEY/METER/DENSITY는 자연스럽게 무시됩니다. 부작용 없음.

### 검증 방법

1. 수정 전후 동일 조건(같은 MIDI, seed=42)으로 생성
2. 첫 윈도우(Bar 0~3)의 생성 결과를 A/B 청취 비교
3. 특히 첫 2~4마디의 피치/리듬 자연스러움에 초점

---

## Issue #3: Time Quantization 학습 일관성 확인 필요 (L335-338)

**심각도: 🟡 중간 — 확인 필요**

### 현재 코드

```python
# midi_to_bar_tokens() 내부
time_tok = VOCAB[f"TIME={min(95, rel_tick * 96 // (res * beats_per_bar))}"]
```

### 의도

마디 내 음표의 상대적 위치를 0~95 사이의 정수로 양자화.

### 확인 필요 사항

`//` (정수 나눗셈, floor division)을 사용하므로 **항상 아래로 반올림**됩니다. `round()`를 사용하면 오차가 ±0.5 step으로 줄지만, `//`는 0~1 step의 편향된 오차를 만듭니다.

```
예시 (res=480, 4/4박자, beats_per_bar=4):
실제 위치 1919 ticks → 1919 * 96 // 1920 = 95 (정확)
실제 위치 100 ticks  → 100 * 96 // 1920 = 5   (실제는 5.0, 정확)
실제 위치 110 ticks  → 110 * 96 // 1920 = 5   (실제는 5.5, 0.5 손실)
```

**이론상 모든 음이 약간씩 앞당겨지는 경향**이 있습니다.

### 핵심 질문

> **학습 전처리 코드의 토크나이저도 동일한 `//` (floor division)을 사용하는가?**

- **동일하다면**: 학습/추론 일관성이 유지되므로 **무해**. 수정 불필요.
- **학습 코드가 `round()`를 쓴다면**: 추론에서 미세한 타이밍 오프셋이 발생하여 음악의 그루브/리듬감에 영향을 줄 수 있음. 이 경우 추론 코드도 `round()`로 맞춰야 함.

### 확인 방법

1. 학습 전처리 코드의 time quantization 부분을 확인
2. `//` vs `round()` 비교
3. 불일치 시 추론 코드를 학습 코드에 맞춰 수정

---

## Issue #4: 윈도우 경계 바 토큰 범위 초과 (L680-692)

**심각도: 🟢 낮음**

### 현재 코드

```python
# generate_sliding_window() 내부 — 생성 토큰 마디별 분리
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
```

### 문제점

모델이 `window_bars`보다 많은 수의 `BAR_START`를 생성하면(L673-674의 break로 최대 1개 초과까지 가능), `cur_bar_num`이 다음 윈도우 영역을 침범하여 **다음 윈도우의 생성 결과를 미리 덮어쓸 수 있습니다.**

또한 break 직후 L691-692에서 마지막 미완성 바가 저장되므로, **윈도우 경계의 마지막 마디가 불완전한 토큰으로 저장**될 수 있습니다.

### 권장 수정

`cur_bar_num`이 현재 윈도우 범위를 초과하지 않도록 가드를 추가합니다:

```python
# 변경 전
if cur_bar_toks:
    gen_bar_tokens[cur_bar_num] = cur_bar_toks

# 변경 후
if cur_bar_toks and cur_bar_num <= win_end:
    gen_bar_tokens[cur_bar_num] = cur_bar_toks
```

**부작용 없음.** 다음 윈도우 영역에 실수로 쓰는 것만 방지하는 순수한 안전장치입니다.

---

## Issue #5: 박자 변경 시 bar_idx 계산 불일치 (L335 vs L366-407)

**심각도: 🟡 중간 — 박자 변경이 있는 곡에서만 발생**

### 현재 코드

토크나이징에서 bar_idx를 계산하는 방식이 **두 단계에서 서로 다릅니다:**

**1단계 — 음표 → timeline 배정 (L335):**

```python
# 나눗셈 기반: 현재 음표 시점의 beats_per_bar로 나눔
bar_idx = int(pm.time_to_tick(n.start) // (res * beats_per_bar))
```

**2단계 — timeline → bar_tokens 조립 (L366-407):**

```python
# 축적 기반: 각 마디의 실제 길이를 순차 누적
accumulated_ticks = 0
for bar_idx in range(max_bar + 1):
    ...
    beats = ts_changes[ts_idx].numerator  # 해당 시점 박자
    accumulated_ticks += res * beats       # 마디 길이 누적
```

### 문제점

**단일 박자 곡에서는 두 방식이 동일한 결과**를 냅니다. 하지만 **박자가 바뀌는 곡**에서는 불일치가 발생합니다.

#### 구체적 예시: 4/4 → 3/4 전환

```
resolution = 480 (ticks per beat)

곡 구조:
  Bar 0: 4/4 → 4 * 480 = 1920 ticks (tick 0~1919)
  Bar 1: 4/4 → 4 * 480 = 1920 ticks (tick 1920~3839)
  Bar 2: 3/4 → 3 * 480 = 1440 ticks (tick 3840~5279)  ← 박자 변경
  Bar 3: 3/4 → 3 * 480 = 1440 ticks (tick 5280~6719)
```

**2단계 (축적 기반) — 정확:**

| bar_idx | accumulated_ticks (시작) | 길이 |
|---------|------------------------|------|
| 0 | 0 | 1920 |
| 1 | 1920 | 1920 |
| 2 | 3840 | 1440 |
| 3 | 5280 | 1440 |

**1단계 (나눗셈 기반) — Bar 2 이후 tick 5280의 음표:**

```python
# 이 시점에서 beats_per_bar = 3 (3/4 박자)
bar_idx = 5280 // (480 * 3) = 5280 // 1440 = 3  ← ✅ 맞음
```

하지만 **Bar 1의 tick 2000의 음표 (아직 4/4 구간):**

```python
# beats_per_bar = 4 (4/4 박자)
bar_idx = 2000 // (480 * 4) = 2000 // 1920 = 1  ← ✅ 맞음
```

**문제가 되는 경계 케이스 — 3/4 구간에서 tick 4320의 음표:**

```python
# beats_per_bar = 3 (3/4 박자)
bar_idx = 4320 // (480 * 3) = 4320 // 1440 = 3  ← ❌ 틀림! 실제는 Bar 2
```

축적 기반으로는 Bar 2 (tick 3840~5279) 안에 있는 음표인데, 나눗셈 기반은 **곡 시작부터 3/4가 계속 유지되었다고 가정**하여 bar_idx=3으로 계산합니다.

결과: **1단계에서 timeline[3]에 넣은 음표가, 2단계의 bar_tokens[3]이 아닌 bar_tokens[2]에 들어가야 할 음표**입니다. timeline에는 있지만 bar_tokens 조립에서 빠지거나, 잘못된 마디에 배정됩니다.

### 영향

- **단일 박자 곡**: 완전히 무해 (대부분의 팝/클래식)
- **박자 변경 곡**: 변경 지점 이후의 음표들이 잘못된 마디에 배정되어, 모델 컨텍스트에 포함되는 원곡 정보가 왜곡됨 → 생성 품질 저하

### 권장 수정

1단계의 bar_idx 계산을 2단계와 동일한 축적 방식으로 변경합니다.

**Option A: 사전에 bar_tick_map을 만들어 재사용**

```python
def midi_to_bar_tokens(midi_path, genre, VOCAB):
    pm  = pretty_midi.PrettyMIDI(midi_path)
    res = pm.resolution
    # ... (기존 ts_changes, key_changes 등 파싱)

    # ── 마디별 시작 tick 사전 계산 (축적 기반) ──
    bar_tick_starts = {}  # bar_idx → start_tick
    acc = 0
    for b in range(5000):  # 충분히 큰 범위
        bar_tick_starts[b] = acc
        bt = pm.tick_to_time(acc)
        idx = max(0, bisect.bisect_right(ts_times, bt) - 1)
        bpb = ts_changes[idx].numerator if idx < len(ts_changes) else 4
        acc += res * bpb
        if acc > pm.time_to_tick(pm.get_end_time()) + res * 8:
            break

    def get_bar_idx(tick):
        """tick이 속한 bar_idx를 반환 (축적 기반)"""
        for b in sorted(bar_tick_starts.keys(), reverse=True):
            if tick >= bar_tick_starts[b]:
                return b
        return 0

    # 기존 L335 대체:
    # bar_idx = int(pm.time_to_tick(n.start) // (res * beats_per_bar))
    bar_idx = get_bar_idx(pm.time_to_tick(n.start))
```

이렇게 하면 1단계와 2단계의 bar_idx가 항상 일치합니다.

### 검증 방법

1. 박자 변경이 포함된 MIDI 파일(예: 4/4 → 3/4 → 4/4)을 준비
2. 수정 전후 `midi_to_bar_tokens()` 결과의 `timeline` 딕셔너리 키(bar_idx)를 비교
3. 2단계의 `bar_tokens` 딕셔너리와 대조하여 모든 음표가 올바른 마디에 배정되었는지 확인

### 5-B. 연관 문제: max_bar 계산도 동일한 근본 원인 (L362-363)

`max_bar` 계산 역시 동일한 문제를 가지고 있습니다:

```python
final_beats = ts_changes[-1].numerator if ts_changes else 4
max_bar     = int(pm.time_to_tick(pm.get_end_time()) // (res * final_beats))
```

**마지막 박자로 전체 곡의 총 마디 수를 나누고 있습니다.** 이전 박자가 다르면 마디 수가 잘못 계산됩니다.

#### 구체적 예시: 3/4(6마디) → 4/4(2마디)

```
3/4 구간: 6 * 1440 = 8640 ticks
4/4 구간: 2 * 1920 = 3840 ticks
총: 12480 ticks

final_beats = 4
max_bar = 12480 // (480 * 4) = 6  ← 실제 8마디인데 6으로 계산
```

**곡 끝부분 2마디에 AI 파트가 아예 생성되지 않습니다.**

반대로 4/4 → 3/4인 경우에는 max_bar가 과대 계산되어 빈 윈도우가 추가 생성됩니다 (데이터 손상은 없지만 불필요한 연산).

#### 권장 수정

Issue #5의 `bar_tick_starts` 사전을 재사용하면 간단합니다:

```python
# 변경 전
final_beats = ts_changes[-1].numerator if ts_changes else 4
max_bar     = int(pm.time_to_tick(pm.get_end_time()) // (res * final_beats))

# 변경 후 (bar_tick_starts를 먼저 계산한 뒤)
end_tick = pm.time_to_tick(pm.get_end_time())
max_bar = 0
for b in sorted(bar_tick_starts.keys()):
    if bar_tick_starts[b] <= end_tick:
        max_bar = b
    else:
        break
```

---

## Issue #6: Velocity 인코딩/디코딩 비대칭 (L352 vs L769)

**심각도: 🟡 중간 — 학습 코드 확인 필요**

### 현재 코드

**인코딩 (L352) — 원곡 MIDI → 토큰:**

```python
vel_tok = VOCAB[f"VEL={min(31, n.velocity * 32 // 128)}"]
```

**디코딩 (L769) — 생성 토큰 → MIDI:**

```python
velocity = max(1, min(127, (cur_vel + 1) * 4))
```

### 문제점

인코딩과 디코딩의 수식이 역함수 관계가 아닙니다.

| 원본 velocity | 인코딩 (VEL) | 디코딩 결과 | 오차 |
|:---:|:---:|:---:|:---:|
| 1 | 0 | 4 | +3 |
| 32 | 8 | 36 | +4 |
| 64 | 16 | 68 | +4 |
| 100 | 25 | 104 | +4 |
| 120 | 30 | 124 | +4 |
| 127 | 31 | 127 | 0 |

**인코딩:**
```
vel → vel * 32 // 128 = vel // 4  (floor division)
```

**디코딩:**
```
VEL → (VEL + 1) * 4
```

정확한 역변환이라면 `VEL * 4` 또는 `VEL * 4 + 2` (반올림 중앙값)이어야 하는데, `(VEL + 1) * 4`를 사용하여 **모든 velocity에 약 +4의 상향 편향**이 있습니다.

### 음악적 영향

이 비대칭은 두 가지 경로로 영향을 줍니다:

1. **원곡 컨텍스트**: 인코딩만 관여 → 모델에 전달되는 원곡의 VEL 토큰 자체는 정확
2. **생성된 파트의 MIDI 출력**: 디코딩만 관여 → 생성된 모든 음표의 velocity가 약간 높게 출력

결과적으로 **AI가 생성한 파트가 원곡 대비 약간 더 크게(louder) 재생**됩니다.

### 핵심 질문

> **학습 전처리 코드에서 디코딩을 사용하는 곳이 있는가? 동일한 공식인가?**

- **학습 코드가 동일한 공식**: 모델이 이 편향을 학습했으므로 무해. 단, 최종 MIDI의 다이나믹스가 전체적으로 밝음
- **학습 코드가 다른 공식**: 학습/추론 불일치 → 수정 필요

### 수정이 필요한 경우의 권장 코드

```python
# 변경 전 (L769)
velocity = max(1, min(127, (cur_vel + 1) * 4))

# 변경 후 — 인코딩의 정확한 역변환 (구간 중앙값)
velocity = max(1, min(127, cur_vel * 4 + 2))
```

---

## 우선순위 요약

| 순위 | 이슈 | 기대 효과 | 수정 난이도 |
|------|------|----------|------------|
| 1 | Issue #1 — 도약 완화 제거/수정 | 멜로디 윤곽 보존, 연쇄 왜곡 방지 | 매우 쉬움 (삭제 or 1줄 수정) |
| 2 | Issue #2 — 헤더 토큰 주입 | 초반부 생성 품질 개선 | 쉬움 (~10줄 수정) |
| 3 | Issue #5 — bar_idx + max_bar 통일 | 박자 변경 곡의 토크나이징 정확성 + 마디 절단 방지 | 중간 (~20줄 수정) |
| 4 | Issue #3 — Time Quantization 확인 | 학습/추론 일관성 보장 | 확인만 필요 |
| 5 | Issue #6 — Velocity 비대칭 확인 | 다이나믹스 정확성 | 확인 후 1줄 수정 |
| 6 | Issue #4 — 바 토큰 범위 가드 | 경계 마디 안정성 | 매우 쉬움 (1줄 추가) |
