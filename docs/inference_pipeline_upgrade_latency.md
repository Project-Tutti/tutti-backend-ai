# 🚀 Tutti AI 추론 파이프라인 최적화: 추론 속도(Latency) 단축

**문서 목적**: `inference.py`에 적용된 추론 속도 최적화 기법들을 실제 코드 라인 기준으로 정확히 명세하고, 각 기법이 없을 경우 어떤 성능 저하가 발생하는지를 수치적 근거와 함께 분석합니다.

---

## 📊 1. 추론 속도 기대치 (Qwen2.5-0.5B 기준)


| **예상 총 소요 시간 (현재 파이프라인)** | **약 3분 정도까지 축소 / 78마디(태연_I.mid) 기준** |


---

## 🔍 2. 상세 최적화 항목 및 코드 분석

### 2.1. KV Cache 활성화 — O(n) 생성 루프 유지

- **최적화 전 문제점**: `past_key_values`를 넘겨주지 않고 매 스텝마다 전체 시퀀스를 처음부터 다시 계산하면, 생성 스텝이 n번째일 때 계산량이 O(n)이 되어 전체 생성은 O(n²)의 복잡도를 가집니다. 1,500번째 토큰 생성 시 처음보다 1,500배의 연산이 필요하게 됩니다.

- **적용된 로직** (`inference.py`, Line 653~685):
  ```python
  # [Line 653] Prefill: 전체 컨텍스트를 딱 한 번에 처리 후 KV Cache 확보
  out = model(input_ids=combined_input, use_cache=True)
  pkv = out.past_key_values  # ← KV Cache를 변수에 저장

  # [Line 684~685] 이후 루프: 마지막 토큰 1개만 넣고 pkv를 재활용
  out = model(input_ids=cur_in, past_key_values=pkv, use_cache=True)
  pkv = out.past_key_values   # ← 매 스텝마다 갱신하여 다음 스텝에 전달
  ```

- **효과**: 512번의 생성 루프 전체를 통틀어 모델의 Self-Attention 계산량이 일정하게(O(1) per step) 유지됩니다. 78마디 곡 후반부에서도 초반과 동일한 토큰당 생성 속도를 보장합니다.

---

### 2.2. Prefill 단일 호출로 통합 (윈도우 전환 지연 제거)

- **최적화 전 문제점**: 윈도우 시작 시점에 강제 주입 토큰(`BAR_START`, `KEY_`, `INST=`)을 단계적으로 나누어 모델에 넣으면, 윈도우 전환마다 모델을 여러 번 호출하여 초기 로딩 딜레이가 누적되었습니다.

- **적용된 로직** (`inference.py`, Line 641~654):
  ```python
  # [Line 641~647] 현재 마디의 KEY_, METER_, DENSITY_ 토큰 추출
  first_bar = bar_tokens.get(win_start, [])
  injected = [VOCAB["BAR_START"]]
  for t in first_bar:
      name = VOCAB_R.get(t, "")
      if any(name.startswith(p) for p in ["KEY_", "METER_", "DENSITY_"]):
          injected.append(t)
  injected.append(INST_TARGET_ID)

  # [Line 649~653] 컨텍스트 + 강제 주입 토큰을 하나의 텐서로 합쳐 단 1회 호출
  forced_tokens  = torch.tensor([injected], dtype=torch.long, device=device)
  combined_input = torch.cat([input_ids, forced_tokens], dim=1)
  out = model(input_ids=combined_input, use_cache=True)  # ← 단 1회 호출
  pkv = out.past_key_values
  ```

- **효과**: 78마디 곡 기준 78번의 윈도우 전환마다 발생하던 추가 모델 호출을 완전히 제거했습니다. 동시에 모델이 마디의 조성(`KEY_`), 박자(`METER_`), 밀도(`DENSITY_`)를 인지한 상태에서 첫 음을 생성하여 음악적 정합성도 향상시켰습니다.

---

### 2.3. 피치 마스킹 벡터화 (CPU 병목 제거)

- **최적화 전 문제점**: 매 토큰 생성 스텝마다 128번의 Python 루프를 통해 음역대를 검사하고 logit을 마스킹했습니다. GPU 연산이 완료되어도 CPU(Python 인터프리터)가 루프를 처리하는 동안 다음 토큰 생성이 대기하는 **CPU-Bound 병목**이 발생했습니다.
  ```python
  # [삭제된 코드 - Line 689~691 주석 처리됨]
  # for pitch in range(128):           ← 128번 Python 루프 (병목)
  #     if pitch < pitch_min or pitch > pitch_max:
  #         logits[VOCAB[f"PITCH={pitch}"]] = -1e9
  ```

- **적용된 로직** (`inference.py`, Line 672~694):
  ```python
  # [Line 672~674] 루프 시작 전 한 번만 마스크 텐서를 사전 계산 (GPU 상주)
  pitch_token_ids = torch.tensor([VOCAB[f"PITCH={p}"] for p in range(128)], device=device)
  invalid_pitch_mask = (torch.arange(128, device=device) < pitch_min) | \
                       (torch.arange(128, device=device) > pitch_max)
  invalid_pitch_token_ids = pitch_token_ids[invalid_pitch_mask]

  # [Line 694] 루프 내부: 단 1줄의 벡터 연산으로 마스킹 완료
  logits[invalid_pitch_token_ids] = -1e9
  ```

- **효과**: 512번의 생성 루프 × 128번의 Python 루프 = 총 65,536회의 Python 인터프리터 호출이 **단 512번의 GPU 벡터 연산**으로 대체됩니다. GPU의 병렬 연산 성능을 샘플링 단계에서 온전히 활용할 수 있게 되었습니다.

---

### 2.4. 텐서 사전 할당 및 값만 교체 (GC 부하 제거)

- **최적화 전 문제점**: 생성 루프 내부에서 매 스텝마다 새로운 `torch.tensor` 객체를 생성하면, GPU 메모리 할당과 Python 가비지 컬렉터(GC)의 부하가 매 스텝 누적됩니다.
  ```python
  # [삭제된 코드 - Line 729 주석 처리됨]
  # cur_in = torch.tensor([[next_tok]], dtype=torch.long, device=device)  ← 매번 새 객체 생성
  ```

- **적용된 로직** (`inference.py`, Line 666~732):
  ```python
  # [Line 666] 루프 진입 전 단 한 번만 1×1 텐서를 GPU에 할당
  cur_in = torch.zeros((1, 1), dtype=torch.long, device=device)

  # [Line 732] 루프 내부: 새 객체 생성 없이 이미 할당된 텐서의 값만 교체
  cur_in[0, 0] = next_tok
  ```

- **효과**: 512번의 생성 루프 전체에서 텐서 메모리 할당이 단 1회로 감소합니다. GC 호출 빈도와 GPU↔CPU 메모리 전송 오버헤드가 제거되어 반복적인 생성 스텝에서의 시간 누적 손실을 차단합니다.

---

### 2.5. 컨텍스트 길이 제한 및 동적 트리밍 (VRAM 보호)

- **적용 배경**: 긴 곡에서 과거 컨텍스트가 무한정 누적되면 GPU VRAM이 부족해져 메모리 스와핑이 발생하고, 이는 KV Cache 병목과 구분하기 어려운 급격한 속도 저하로 나타납니다.

- **적용된 로직** (`inference.py`, Line 563~635):
  ```python
  # [Line 563~564] 최대 시퀀스 길이 128K, 실질 컨텍스트 상한선 8,192 토큰
  SEQ_LEN = 128000
  MAX_CTX = SEQ_LEN - 119808  # 실질 컨텍스트 = 8,192 토큰

  # [Line 621~624] 컨텍스트 초과 시 과거/미래 마디를 비율 보존 방식으로 트리밍
  if total_len > MAX_CTX:
      past_bar_list, future_bar_list = trim_bars_preserving_ratio(
          past_bar_list, future_bar_list,
          h_len, c_len, MAX_CTX, target_past_ratio)
  ```

- **효과**: VRAM 내 KV Cache 메모리 맵의 크기를 상한선 이내로 유지하여 메모리 스와핑을 방지합니다. GPU 연산이 항상 VRAM 내에서만 이루어지도록 보장하여 속도 저하 없는 78마디 이상의 장편 곡 생성을 가능하게 합니다.

---

## 📝 3. 진단 체크리스트

| 점검 항목 | 현재 상태 |
|---|---|
| `past_key_values` 전달 여부 | ✅ Line 684: `model(input_ids=cur_in, past_key_values=pkv, ...)` |
| `use_cache=True` 설정 여부 | ✅ Line 653, 684 |
| 피치 마스킹 벡터화 여부 | ✅ Line 672~694 (루프 방식 주석 처리됨) |
| 텐서 사전 할당 여부 | ✅ Line 666, 732 |
| 모델 정밀도 | ✅ `torch.bfloat16` (`load_model` 참조) |
| 컨텍스트 길이 상한 보호 | ✅ Line 621~624 (`trim_bars_preserving_ratio`) |