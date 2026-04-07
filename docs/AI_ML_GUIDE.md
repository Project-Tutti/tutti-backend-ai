# Tutti AI Server — AI / ML 엔지니어 가이드

> **대상 독자**: AI/ML 연구자, 모델 학습/배포 엔지니어  
> **최종 수정**: 2026-04-07  
> **버전**: v2 (Qwen2.5-0.5B 통합 모델)

---

## 목차

1. [시스템 개요](#1-시스템-개요)
2. [추론 파이프라인 전체 흐름](#2-추론-파이프라인-전체-흐름)
3. [프로젝트 구조 (AI 관점)](#3-프로젝트-구조-ai-관점)
4. [악기 그룹 정의 (INSTRUMENT_GROUPS)](#4-악기-그룹-정의-instrument_groups)
5. [Vocabulary 구조](#5-vocabulary-구조)
6. [모델 아키텍처 및 로딩](#6-모델-아키텍처-및-로딩)
7. [MIDI 토큰화 (midi_to_bar_tokens)](#7-midi-토큰화-midi_to_bar_tokens)
8. [생성 알고리즘 (generate_for_target)](#8-생성-알고리즘-generate_for_target)
9. [토큰 → 노트 디코딩 (decode_tokens)](#9-토큰--노트-디코딩-decode_tokens)
10. [후처리 (postprocess)](#10-후처리-postprocess)
11. [MIDI 저장 (save_midi)](#11-midi-저장-save_midi)
12. [MIDI 전처리 — 트랙 재매핑](#12-midi-전처리--트랙-재매핑)
13. [모델 레지스트리와 배포](#13-모델-레지스트리와-배포)
14. [파일별 함수 종속성 맵](#14-파일별-함수-종속성-맵)
15. [수정 가이드 — "이런 걸 바꾸고 싶을 때"](#15-수정-가이드--이런-걸-바꾸고-싶을-때)
16. [하이퍼파라미터 참조](#16-하이퍼파라미터-참조)
17. [테스트 및 검증](#17-테스트-및-검증)

---

## 1. 시스템 개요

Tutti AI Server는 **입력 MIDI를 컨텍스트로 받아 새로운 악기 파트를 자동 생성**하는 추론 서비스입니다.

```
사용자 요청 (Main Server에서 전달)
  ↓
[전처리] 원본 MIDI 트랙 재매핑 (midi_processor.py)
  ↓
[토큰화] MIDI → 마디별 토큰 시퀀스 변환 (inference.py)
  ↓
[추론] Qwen2.5-0.5B 모델로 새 파트 자동회귀 생성 (inference.py)
  ↓
[디코딩] 토큰 → 노트 좌표 변환 (inference.py)
  ↓
[후처리] 음역 클리핑, 도약 완화, 윈도우 경계 보정 (inference.py)
  ↓
[저장] 원본 MIDI에 새 트랙 추가 후 저장 (inference.py)
  ↓
[전송] 결과 MIDI를 Main Server에 콜백 (arrangement.py → callback.py)
```

**핵심 설계 결정**:
- **단일 패스 추론**: 트랙별 반복 추론이 아닌, 한 번에 하나의 타겟 악기 파트만 생성
- **슬라이딩 윈도우**: 긴 곡을 `window_bars`(8마디) 단위로 분할하여 순차 생성
- **Constrained decoding**: 노트 구조(INST→ART→EXPR→TIME→PITCH→DUR→VEL)에 따라 허용 토큰을 마스킹

---

## 2. 추론 파이프라인 전체 흐름

```
run_arrangement()
│
├── 1. 입력 검증 및 악기 설정
│   ├── INSTRUMENT_GROUPS[target]에서 pitch_min, pitch_max, representative 조회
│   └── pitch_min/max가 None이면 악기 기본값 자동 적용
│
├── 2. MIDI 토큰화: midi_to_bar_tokens()
│   ├── pretty_midi로 MIDI 파싱
│   ├── 각 노트를 마디(bar) 단위로 분류
│   ├── 노트별 토큰 생성: [INST, ART, EXPR, TIME, PITCH, DUR, VEL]
│   ├── 마디별 메타 토큰: [BAR_START, KEY, METER, DENSITY, ..., BAR_END]
│   └── 헤더 토큰: [PIECE_START, GENRE_{genre}]
│
├── 3. 생성: generate_for_target()
│   ├── 슬라이딩 윈도우 순회 (8마디씩)
│   ├── 컨텍스트 구성 = 헤더 + 앞 밴드 + 앞 타겟 히스토리 + 현재 밴드
│   ├── KV Cache로 컨텍스트 인코딩
│   ├── constrained autoregressive 생성 (note_pos 상태 머신)
│   └── nucleus sampling (top_p=0.95)
│
├── 4. 디코딩: decode_tokens()
│   ├── 토큰 시퀀스 → (start, end, pitch, velocity) 좌표 변환
│   └── 소스 MIDI의 tempo/time_signature로 절대 시간 복원
│
├── 5. 후처리: postprocess()
│   ├── 음역 클리핑
│   ├── 최대 4초 초과 음표 제한
│   ├── 50ms 미만 초단음 제거
│   ├── 윈도우 경계 갭(30ms 미만) 연결
│   └── 옥타브 초과 도약 완화
│
└── 6. 저장: save_midi()
    ├── 소스 MIDI 딥카피
    ├── 새 Instrument 트랙 추가
    └── 출력 파일 저장
```

---

## 3. 프로젝트 구조 (AI 관점)

```
tutti-backend-ai/
├── app/
│   ├── services/
│   │   ├── inference.py        # ★ 핵심: 추론 엔진 전체
│   │   ├── midi_processor.py   # MIDI 전처리 (재매핑)
│   │   ├── arrangement.py      # [서버 코드] 편곡 라이프사이클 (수정 비권장)
│   │   └── callback.py         # [서버 코드] 결과 전송 (수정 비권장)
│   ├── core/
│   │   ├── model_registry.py   # 모델 로드/관리 (새 모델 추가 시 수정)
│   │   └── config.py           # [서버 코드] 환경 변수
│   ├── schemas/
│   │   └── request.py          # API 스키마 (GenreType 등 AI 관련 검증)
│   ├── api/                    # [서버 코드] 엔드포인트 (수정 비권장)
│   └── main.py                 # [서버 코드] 앱 진입점
├── registry.json               # 모델 레지스트리 설정
├── tests/
│   ├── test_midi_processor.py  # MIDI 전처리 테스트
│   └── test_midi_corruption_analysis.py  # MIDI 오염 방지 테스트
└── local_docs/
    └── inference_new.py        # 원본 CLI 추론 코드 (레퍼런스)
```

> **💡 주의**: `[서버 코드]`로 표시된 파일들은 백엔드 엔지니어가 관리합니다. AI/ML 수정 시 해당 파일을 직접 수정하지 말고, `inference.py`의 공개 API 인터페이스를 유지하세요.

---

## 4. 악기 그룹 정의 (INSTRUMENT_GROUPS)

**위치**: `app/services/inference.py` L28-55

13개 악기 그룹으로 128개 MIDI 프로그램을 분류합니다. 학습과 추론 모두 이 그룹핑을 사용합니다.

| 그룹 | representative | is_drum | pitch_min | pitch_max | 포함 프로그램 |
|------|---------------|---------|-----------|-----------|-------------|
| `drum` | 128 | ✅ | 35 | 81 | 128 (드럼) |
| `keyboard` | 0 | ❌ | 21 | 108 | 0-7 (피아노, 일렉트릭 피아노 등) |
| `organ` | 16 | ❌ | 36 | 96 | 16-23 (오르간류) |
| `mallet` | 12 | ❌ | 48 | 96 | 8-15, 112, 114 (타악기, 벨 등) |
| `guitar` | 25 | ❌ | 40 | 88 | 24-28, 31, 45-46, 104-108, 110 (어쿠스틱/나일론) |
| `dist_guitar` | 30 | ❌ | 40 | 88 | 29-30 (디스토션 기타) |
| `bass` | 33 | ❌ | 28 | 67 | 32-39 (베이스류) |
| `violin` | 40 | ❌ | 55 | 103 | 40-43 (바이올린, 비올라, 첼로, 콘트라베이스) |
| `woodwind` | 73 | ❌ | 60 | 96 | 68-75, 77-79, 111 (플룻, 클라리넷 등) |
| `saxophone` | 65 | ❌ | 49 | 80 | 64-67 (색소폰류) |
| `synth` | 81 | ❌ | 36 | 96 | 80-87 (신스류) |
| `brass` | 56 | ❌ | 52 | 82 | 56-60 (트럼펫, 트롬본 등) |
| `ensemble` | 48 | ❌ | 36 | 96 | 44, 48-54, 61-63, 76, 88-103 (앙상블, 학습되지 않는 악기 Catch-all) |

### 관련 룩업 테이블

| 변수 | 타입 | 설명 | 위치 |
|------|------|------|------|
| `INSTRUMENT_GROUPS` | `dict[str, dict]` | 그룹 이름 → 설정 (representative, pitch 범위) | L28-55 |
| `_GROUPING_PROGRAMS` | `dict[int, list[int]]` | representative → 포함 프로그램 목록 | L64-79 |
| `PROGRAM_TO_REP` | `dict[int, int]` | MIDI 프로그램 → representative 역매핑 | L80-83 |
| `_REP_TO_GROUP` | `dict[int, str]` | representative → 그룹 이름 역매핑 | L86 |
| `DROP_SET` | `set[int]` | 토큰화 시 무시할 프로그램 목록 | L60-61 |
| `ALL_TARGET_NAMES` | `list[str]` | 생성 가능한 13개 그룹 이름 리스트 | L57 |

### `resolve_target(instrument_id: int) → str`

**위치**: L97-113

API에서 받은 MIDI program 번호(0~128)를 `INSTRUMENT_GROUPS` 키로 변환합니다.

```python
resolve_target(0)   → "keyboard"
resolve_target(40)  → "violin"
resolve_target(128) → "drum"
resolve_target(200) → ValueError
```

### DROP_SET (토크나이징 제외 목록)

아래 MIDI 프로그램은 학습 데이터에서 제외되었으며, 토큰화 시에도 무시됩니다:

```python
DROP_SET = {47, 55, 109, 113, 115, 116, 117, 118, 119, 120,
            121, 122, 123, 124, 125, 126, 127}
```

이 프로그램의 악기는 트레이닝 데이터에 노이즈를 주거나 효과음(Sound Effects) 계열이므로 제외됩니다.

---

## 5. Vocabulary 구조

**위치**: `app/services/inference.py` → `build_v5_vocab()` (L119-147)

총 토큰 수: ~700개 (가변)

### 토큰 구성

| 카테고리 | 토큰 형식 | 개수 | 설명 |
|---------|----------|------|------|
| **특수 토큰** | `PAD`, `BOS`, `EOS`, `SEP`, `PIECE_START`, `PIECE_END`, `BAR_START`, `BAR_END`, `PHRASE_END`, `<PRE>`, `<SUF>`, `<MID>` | 12 | 시퀀스 구조 마커 |
| **장르** | `GENRE_CLASSICAL`, `GENRE_JAZZ`, ... | 7 | 조건부 생성용 장르 |
| **조성** | `KEY_C:maj`, `KEY_C#:min`, ..., `KEY_NONE` | 25 | 12음 × 2모드 + NONE |
| **타겟** | `TARGET_40`, `TARGET_68`, `TARGET_73` | 3 | (레거시, 현재 미사용) |
| **박자** | `METER_4:4`, `METER_3:4`, ..., `METER_OTHER` | 6 | 박자 표기 |
| **밀도** | `DENSITY_1` ~ `DENSITY_5` | 5 | 마디당 노트 밀도 |
| **악기** | `INST=0` ~ `INST=128` | 129 | representative 프로그램 번호 |
| **아티큘레이션** | `ART_NORMAL`, `ART_LEGATO`, `ART_VIBRATO`, `ART_STACCATO` | 4 | 연주 스타일 |
| **표현** | `EXPR_0` ~ `EXPR_31` | 32 | 세밀한 다이내믹 제어 |
| **시간** | `TIME=0` ~ `TIME=95` | 96 | 마디 내 상대 위치 (96분음표 해상도) |
| **음높이** | `PITCH=0` ~ `PITCH=127` | 128 | MIDI 피치 |
| **듀레이션** | `DUR=1` ~ `DUR=192` | 192 | 틱 단위 음표 길이 |
| **벨로시티** | `VEL=0` ~ `VEL=31` | 32 | 32단계 벨로시티 양자화 |
| **텍스트** | `TEXT_melodic`, `TEXT_epic`, ... | 11 | (향후 텍스트 조건부 생성용) |

### 토큰 시퀀스 구조

하나의 곡은 다음과 같은 토큰 시퀀스로 표현됩니다:

```
PIECE_START → GENRE_{g}
  → BAR_START → KEY_{k} → METER_{m} → DENSITY_{d}
    → [PHRASE_END]
    → INST={i} → ART_{a} → EXPR_{e} → TIME={t} → PITCH={p} → DUR={d} → VEL={v}
    → INST={i} → ART_{a} → EXPR_{e} → TIME={t} → PITCH={p} → DUR={d} → VEL={v}
    → ...
  → BAR_END
  → BAR_START → ...
  → ...
→ PIECE_END
```

**노트 하나 = 7개 토큰**: `INST → ART → EXPR → TIME → PITCH → DUR → VEL`

---

## 6. 모델 아키텍처 및 로딩

### 아키텍처

**베이스 모델**: `Qwen/Qwen2.5-0.5B` (Autoregressive Causal LM)

| 설정 | 값 | 비고 |
|------|-----|------|
| `vocab_size` | ~700 (동적) | `build_v5_vocab()` 결과에 따라 결정 |
| `max_position_embeddings` | 2048 | 최대 시퀀스 길이 |
| `sliding_window` | `None` (비활성) | 전체 어텐션 사용 |
| `precision` | `bfloat16` | 메모리 효율 + 속도 |
| `use_cache` | `False` (학습) / `True` (추론 KV cache) | |

### 커스텀 레이어

```python
model.model.embed_tokens = nn.Embedding(vocab_size, hidden_size)   # bfloat16
model.lm_head            = nn.Linear(hidden_size, vocab_size)      # bfloat16, bias=False
```

`embed_tokens`와 `lm_head`는 MIDI vocabulary에 맞게 재초기화됩니다.

### 체크포인트 로딩 (`load_model`)

**위치**: L153-183

```python
def load_model(ckpt_path, vocab_size, vocab, device):
    # 1. Qwen2.5 config 기반 모델 생성 (vocab_size 오버라이드)
    # 2. bfloat16 변환
    # 3. embed_tokens, lm_head 재초기화
    # 4. safetensors 또는 pytorch_model.bin 로드 (strict=True)
    # 5. eval 모드 + device 이동
```

**체크포인트 경로 탐색 순서**:
1. `{ckpt_path}/model.safetensors` (우선)
2. `{ckpt_path}/pytorch_model.bin` (폴백)

### 모델 레지스트리

**위치**: `app/core/model_registry.py`

서버 시작 시 `ModelRegistry.load_all_models()`가 `registry.json`의 모든 모델을 로드합니다.

```json
// registry.json
{
  "version": "v2",
  "default": "qwen2.5",
  "models": [
    {
      "type": "qwen2.5",       // ← get_model("qwen2.5")으로 선택
      "name": "Tutti Unified v1",
      "path": "best",          // ← MODEL_DIR/best/ 디렉토리
      "description": "Qwen2.5-0.5B 기반 13그룹 통합 모델"
    }
  ]
}
```

**모델 교체 시**: 체크포인트 파일만 교체하면 됩니다. vocab이나 모델 구조가 바뀌지 않는 한 코드 변경 불필요.

**새 아키텍처 모델 추가 시**:
1. `registry.json`에 새 엔트리 추가
2. `model_registry.py`의 `_load_single_model()`에 새 `model_type` 분기 추가
3. GCS에 체크포인트 업로드

---

## 7. MIDI 토큰화 (midi_to_bar_tokens)

**위치**: L189-312  
**의존성**: `pretty_midi`  
**호출자**: `run_arrangement()` → 내부 호출

### 입력/출력

| | 타입 | 설명 |
|--|------|------|
| **입력** | `midi_path: str`, `genre: str`, `vocab`, `vocab_r` | MIDI 파일 경로, 장르, 보캡 |
| **출력** | `(header, bar_tokens, max_bar, source_pm)` | 헤더 토큰, 마디별 토큰 맵, 총 마디 수, PrettyMIDI 객체 |

### 처리 과정

```
1. PrettyMIDI로 MIDI 파싱
     ↓
2. 각 악기(Instrument)의 노트 순회
   ├── DROP_SET에 포함된 프로그램 → 건너뜀
   ├── 드럼 → rep=128
   └── 일반 → PROGRAM_TO_REP으로 representative 매핑
     ↓
3. 노트별 토큰 계산
   ├── tempo 기반 duration → DUR= 토큰 (tick 단위, 24tick/beat)
   ├── time_signature 기반 bar 위치 → TIME= 토큰 (96분음표 해상도)
   ├── key_signature → KEY_ 토큰
   ├── 아티큘레이션 판별 (staccato, legato, normal)
   ├── 표현 = velocity 기반 → EXPR_ 토큰
   └── PHRASE_END 판별 (이전 노트와 1beat 이상 갭)
     ↓
4. 마디별 정렬 (TIME 순, 동일 TIME일 때 INST 순)
     ↓
5. 마디별 토큰 조립
   [BAR_START, KEY, METER, DENSITY, {노트들...}, BAR_END]
```

### 주요 변환 로직

| 항목 | 계산식 | 비고 |
|------|--------|------|
| `dur_tick` (DUR 토큰) | `round((note_dur / s_per_beat) * 24)` → clamp [1, 192] | 24tick/beat 해상도 |
| `time_tok` (TIME 토큰) | `rel_tick * 96 // (res * beats_per_bar)` → clamp [0, 95] | 마디 내 96분음표 해상도 |
| `density` | `min(5, max(1, total_notes // 4))` | 마디당 노트 수 / 4 |
| `vel_tok` | `min(31, velocity * 32 // 128)` | 128단계 → 32단계 양자화 |
| `art_tok` | dur ≤ 2 → staccato, legato_ratio > 0.95 → legato, else normal | |

### FLAT_TO_SHARP 매핑

```python
FLAT_TO_SHARP = {"Db":"C#", "Eb":"D#", "Fb":"E", "Gb":"F#", "Ab":"G#", "Bb":"A#", "Cb":"B"}
```

key_signature 이름이 플랫 표기일 경우 샤프 표기로 정규화합니다 (vocab에는 샤프만 존재).

---

## 8. 생성 알고리즘 (generate_for_target)

**위치**: L330-502  
**호출자**: `run_arrangement()` → 내부 호출

### 슬라이딩 윈도우 구조

```
곡 전체: 0 ─────────────────────────────────── max_bar
  
window 0: [0 ──── 7]
           ctx: (없음)
  
window 1:  ← ctx → [8 ─── 15]
           [0 ── 7]
  
window 2:          ← ctx → [16 ── 23]
                   [8 ── 15]
```

- `window_bars = 8`: 한 번에 8마디 생성
- `context_bars = 8`: 앞 8마디를 컨텍스트로 사용
- 컨텍스트에는 **앞 밴드(원본)** + **앞 타겟(이전에 생성된 것)** + **현재 밴드(원본)** 포함

### Constrained Decoding (note_pos 상태 머신)

생성 시 각 토큰 위치마다 **허용되는 토큰만 마스킹 해제**합니다.

```
note_pos = 0: 자유 상태 → INST=target, BAR_START, PIECE_END, EOS 만 허용
  ↓ (INST= 토큰 생성 시)
note_pos = 1: ART 대기 → ART_NORMAL/LEGATO/VIBRATO/STACCATO 만 허용
  ↓
note_pos = 2: EXPR 대기 → EXPR_0~31 만 허용
  ↓
note_pos = 3: TIME 대기 → TIME={t} (t > last_time_val) 만 허용  ★ 시간 단조증가 강제
  ↓
note_pos = 4: PITCH 대기 → PITCH={p} (pitch_min ≤ p ≤ pitch_max) 만 허용
  ↓
note_pos = 5: DUR 대기 → DUR=1~192 만 허용
  ↓
note_pos = 6: VEL 대기 → VEL=0~31 만 허용
  ↓ (→ note_pos = 0 으로 리셋)
```

**마디 전환**: `BAR_START` 토큰 생성 시 `bar_count` 증가, `last_time_val` 리셋.  
**종료 조건**: `bar_count > window_bars` 또는 `PIECE_END`/`EOS` 생성.

### Nucleus Sampling

```python
temperature → logits / temperature
softmax → cumulative probability
top_p = 0.95 → 누적 확률 95% 초과 토큰 차단
multinomial sampling
```

### KV Cache 활용

```
1. 컨텍스트 전체를 한 번에 인코딩 → past_key_values 획득
2. 프롬프트 (BAR_START + INST=target) 토큰 투입
3. 이후 한 토큰씩 자동회귀 생성 (KV cache 사용)
```

### 생성 결과 저장

각 윈도우에서 생성된 토큰은 `gen_bar_tokens[bar_idx]`에 마디별로 분리 저장됩니다.  
다음 윈도우의 타겟 히스토리 컨텍스트로 사용됩니다.

---

## 9. 토큰 → 노트 디코딩 (decode_tokens)

**위치**: L508-564  
**호출자**: `generate_for_target()` → 각 윈도우별 호출

### 디코딩 로직

```
토큰 시퀀스 순회:
  BAR_START  → bar_idx += 1
  INST={n}   → cur_inst = n
  TIME={t}   → cur_time_tok = t
  PITCH={p}  → cur_pitch = p
  DUR={d}    → cur_dur = d
  VEL={v}    → cur_vel = v
               └── if cur_inst == target_prog and (모든 필드 존재) and (마디 범위 내):
                   └── 절대 시간 계산 → notes_out에 추가
```

### 절대 시간 복원

```python
# 마디 시작 tick (누적)
b_tick = bar_tick_map[bar_idx]

# 마디 내 상대 tick
abs_tick = b_tick + cur_time_tok * bar_ticks // 96

# tick → 초
start_sec = source_pm.tick_to_time(abs_tick)

# duration: DUR 토큰 → 초
dur_sec = (cur_dur / 24.0) * (60.0 / bpm)
```

### 벨로시티 역양자화

```python
velocity = max(1, min(127, (cur_vel + 1) * 4))
```

32단계 (0~31) → 128단계 (4~128)

---

## 10. 후처리 (postprocess)

**위치**: L570-601  
**호출자**: `run_arrangement()` → 생성 후 호출

| 단계 | 처리 내용 | 목적 |
|------|----------|------|
| 1 | `pitch_min ≤ pitch ≤ pitch_max` 필터 | 음역 클리핑 |
| 2 | `end - start > 4.0s` → 4.0s로 제한 | 비정상 긴 음표 방지 |
| 3 | `end - start < 0.05s` → 제거 | 초단음 제거 |
| 4 | 연속 노트 간 `0 < gap < 0.03s` → 갭 채움 | 윈도우 경계 연결 |
| 5 | `|interval| > 12` → 옥타브 조정 | 큰 도약 완화 |
| 6 | 음역 + 최소 길이 재확인 | 안전 보장 |

---

## 11. MIDI 저장 (save_midi)

**위치**: L607-620  
**호출자**: `run_arrangement()` → 최종 단계

```python
def save_midi(notes, source_pm, output_path, target_prog, target_name):
    out_pm = copy.deepcopy(source_pm)                      # 원본 MIDI 딥카피
    new_inst = pretty_midi.Instrument(
        program=target_prog if target_prog < 128 else 0,   # 드럼은 program=0
        is_drum=(target_prog == 128),
        name=target_name)
    for n in notes:
        new_inst.notes.append(pretty_midi.Note(...))
    out_pm.instruments.append(new_inst)                    # 트랙 추가
    out_pm.write(output_path)
```

**핵심**: 원본 MIDI의 모든 기존 트랙을 보존하고, **새 트랙만 추가**합니다.

---

## 12. MIDI 전처리 — 트랙 재매핑

**위치**: `app/services/midi_processor.py`  
**의존성**: `mido`  
**호출 시점**: 추론 전에 `arrangement.py`에서 호출 (AI 추론 코드와 독립)

### 목적

Main Server의 `mappings` 지시에 따라 **추론 전에** 원본 MIDI의 트랙 악기를 변경하거나 불필요한 트랙을 삭제합니다.

### 동작 원리

| mapping.targetInstrumentId | 동작 |
|---------------------------|------|
| `129` (DROP) | 해당 트랙/채널의 모든 이벤트 삭제 |
| `0~128` | 해당 트랙/채널의 `program_change`를 지정 값으로 변경 |

### Type 0 vs Type 1

| | Type 0 | Type 1 |
|--|--------|--------|
| 트랙 수 | 1개 | 여러 개 |
| 악기 구분 | 채널 (0~15) | 트랙 인덱스 |
| `trackIndex` 해석 | **채널 번호**로 사용 | 트랙 인덱스 그대로 |
| 삭제 방식 | 단일 트랙 내 채널 메시지 필터링 | 트랙 자체 삭제 |

### 안전장치

| 안전장치 | 설명 | 위치 |
|---------|------|------|
| Delta time 누적 | Type 0 채널 삭제 시 삭제된 이벤트의 delay를 다음 이벤트에 누적 | `_remap_type0` L186-195 |
| program_change 자동 삽입 | 트랙에 program_change가 없으면 자동 삽입 | `_remap_type1` L108-125 |
| 메타 트랙 보호 | tempo/time_signature만 있는 트랙은 삭제 거부 | `_remap_type1` L130-136 |

> **AI/ML 관점에서의 중요성**: 재매핑은 추론 **이전**에 실행되므로, `midi_to_bar_tokens()`가 읽는 MIDI에 직접 영향을 줍니다. 잘못된 재매핑은 잘못된 토큰화 → 잘못된 생성으로 이어집니다.

---

## 13. 모델 레지스트리와 배포

### 모델 파일 구조

```
GCS: gs://tutti-ai-models/v1/
├── registry.json
└── best/                     ← registry.json의 models[0].path
    ├── model.safetensors     ← 체크포인트 (우선)
    └── (또는 pytorch_model.bin)
```

### 모델 배포 프로세스

```
1. 학습 완료 → 체크포인트 파일 저장
     ↓
2. GCS 업로드
   $ gsutil cp model.safetensors gs://tutti-ai-models/v1/best/model.safetensors
     ↓
3. (필요 시) registry.json 수정 후 Git push
   → CI가 자동으로 GCS 업로드 + Pod 재시작
     ↓
4. Pod 재시작 시 Init Container가 새 체크포인트를 다운로드하여 자동 반영
```

### 기본 모델 설정 방법

`registry.json`의 `"default"` 필드를 변경합니다:

```json
{
  "default": "qwen2.5",  ← 이 값을 models[].type 중 하나로 설정
  "models": [...]
}
```

API 요청에서 `modelType`을 명시하지 않으면 이 기본 모델이 사용됩니다.

---

## 14. 파일별 함수 종속성 맵

### `inference.py` — 내부 호출 그래프

```
run_arrangement()                    ← 유일한 공개 API (arrangement.py에서 호출)
├── INSTRUMENT_GROUPS[target]        ← 악기 설정 조회
├── midi_to_bar_tokens()             ← MIDI 토큰화
│   ├── pretty_midi.PrettyMIDI()
│   ├── PROGRAM_TO_REP               ← 프로그램 → representative 변환
│   ├── DROP_SET                     ← 제외 프로그램 필터
│   ├── FLAT_TO_SHARP                ← 조성 정규화
│   └── vocab[...]                   ← 토큰 ID 조회
│
├── generate_for_target()            ← 슬라이딩 윈도우 생성
│   ├── trim_context()               ← 컨텍스트 길이 제한 (MAX_CTX = 1748)
│   ├── model() (forward)            ← Qwen2.5 추론 (KV cache)
│   ├── decode_tokens()              ← 토큰 → 노트 디코딩
│   │   └── source_pm.tick_to_time() ← 절대 시간 복원
│   └── constrained sampling        ← note_pos 기반 마스킹
│
├── postprocess()                    ← 후처리 (음역, 도약, 갭 보정)
│
└── save_midi()                      ← MIDI 파일 저장
    ├── copy.deepcopy(source_pm)
    └── pretty_midi.Instrument()
```

### `inference.py` — 외부 의존성

| 외부 의존성 | 사용처 |
|------------|--------|
| `torch` | 모델 로드, 텐서 연산, 생성 |
| `torch.nn` | `Embedding`, `Linear` 커스텀 레이어 |
| `pretty_midi` | MIDI 파싱, 시간 변환, 출력 저장 |
| `transformers` | `AutoConfig`, `AutoModelForCausalLM` 모델 아키텍처 |
| `safetensors` | 체크포인트 로드 (선택적) |
| `bisect` | 정렬된 리스트에서 이진 검색 (tempo/key/ts 변경점) |

### `midi_processor.py` — 외부 의존성

| 외부 의존성 | 사용처 |
|------------|--------|
| `mido` | MIDI 파싱, 트랙/메시지 조작, 저장 |
| `httpx` | MIDI 파일 HTTP 다운로드 |

### `arrangement.py` → `inference.py` 호출 인터페이스

```python
# arrangement.py에서의 호출
target_name = resolve_target(request.targetInstrumentId)   # int → str
loaded = registry.get_model(request.modelType)              # LoadedModel

result_path = await loop.run_in_executor(
    None,
    run_arrangement,
    str(midi_path),         # song_path: str
    target_name,            # target: str (INSTRUMENT_GROUPS 키)
    request.genre,          # genre: str (CLASSICAL, JAZZ, ...)
    request.temperature,    # temperature: float (0.1~2.0)
    request.minNote,        # pitch_min: int | None
    request.maxNote,        # pitch_max: int | None
    str(output_path),       # output_path: str
    loaded.model,           # model: torch.nn.Module
    loaded.vocab,           # vocab: dict[str, int]
    loaded.vocab_r,         # vocab_r: dict[int, str]
    loaded.device,          # device: str ("cuda" | "cpu")
)
```

> ⚠️ **인터페이스 계약**: `run_arrangement()`와 `resolve_target()`의 시그니처를 변경하면 `arrangement.py`가 깨집니다. 반드시 백엔드 엔지니어와 협의하세요.

---

## 15. 수정 가이드 — "이런 걸 바꾸고 싶을 때"

### 📌 악기 그룹 추가/변경

| 수정 파일 | 위치 | 내용 |
|-----------|------|------|
| `inference.py` | `INSTRUMENT_GROUPS` (L28-55) | 그룹 추가/수정 |
| `inference.py` | `_GROUPING_PROGRAMS` (L64-79) | 프로그램 매핑 수정 |
| `inference.py` | `DROP_SET` (L60-61) | 제외 프로그램 수정 |

> **경고**: 학습 시 사용한 그룹핑과 일치해야 합니다. 변경 시 **재학습 필요**.

### 📌 Vocabulary 수정

| 수정 파일 | 위치 | 내용 |
|-----------|------|------|
| `inference.py` | `build_v5_vocab()` (L119-147) | 토큰 추가/변경 |

> **경고**: vocab 변경 = vocab_size 변경 → **체크포인트와 호환 깨짐** → 재학습 필요.

### 📌 새 장르 추가

| 수정 파일 | 위치 |
|-----------|------|
| `inference.py` | `build_v5_vocab()`의 장르 리스트 (L126-127) |
| `schemas/request.py` | `GenreType` Literal 타입 (L11-13) |

> **주의**: 두 파일의 장르 목록이 **반드시 동기화**되어야 합니다.

### 📌 하이퍼파라미터 튜닝

| 수정 파일 | 위치 | 파라미터 |
|-----------|------|---------|
| `inference.py` | `run_arrangement()` (L658-662) | `window_bars`, `context_bars`, `top_p`, `max_new_tokens`, `seed` |

> 이 값들은 현재 하드코딩되어 있습니다. API 파라미터로 노출하려면 `run_arrangement()` 시그니처와 `ArrangeRequest` 스키마를 함께 수정해야 합니다.

### 📌 후처리 규칙 변경

| 수정 파일 | 위치 | 내용 |
|-----------|------|------|
| `inference.py` | `postprocess()` (L570-601) | 클리핑 임계값, 도약 완화 규칙 등 |

### 📌 모델 아키텍처 변경

| 수정 파일 | 위치 | 내용 |
|-----------|------|------|
| `inference.py` | `load_model()` (L153-183) | 모델 구성 변경 |
| `model_registry.py` | `_load_single_model()` (L86-107) | 새 model_type 분기 추가 |

### 📌 피치 범위 기본값 변경

| 수정 파일 | 위치 |
|-----------|------|
| `inference.py` | `INSTRUMENT_GROUPS`의 `pitch_min`, `pitch_max` |

### 📌 DROP_SET (토큰화 제외 프로그램) 변경

| 수정 파일 | 위치 |
|-----------|------|
| `inference.py` | `DROP_SET` (L60-61) |

> **주의**: 학습 데이터에서 동일한 프로그램이 제외되었는지 확인하세요.

### 📌 Constrained Decoding 규칙 변경

| 수정 파일 | 위치 | 내용 |
|-----------|------|------|
| `inference.py` | `generate_for_target()` (L410-474) | `note_pos` 상태 머신, 마스킹 규칙 |

> `note_pos`의 상태 전이를 변경하면 생성되는 토큰 구조가 바뀌므로, `decode_tokens()`도 함께 수정해야 합니다.

---

## 16. 하이퍼파라미터 참조

### 고정 하이퍼파라미터 (`run_arrangement` 내부)

| 파라미터 | 값 | 설명 |
|---------|-----|------|
| `window_bars` | 8 | 한 번에 생성하는 마디 수 |
| `context_bars` | 8 | 앞 컨텍스트 마디 수 |
| `top_p` | 0.95 | Nucleus sampling 임계값 |
| `max_new_tokens` | 1024 | 윈도우당 최대 생성 토큰 수 |
| `seed` | 42 | 재현성을 위한 랜덤 시드 |

### API에서 조절 가능한 파라미터

| 파라미터 | 범위 | 기본값 | 효과 |
|---------|------|--------|------|
| `temperature` | 0.1 ~ 2.0 | 1.0 | 낮으면 보수적, 높으면 다양한 생성 |
| `genre` | 7종 Literal | `CLASSICAL` | 조건부 생성 스타일 제어 |
| `minNote` / `maxNote` | 0~127 | 악기별 기본값 | 생성 음역 제한 |

### 모델 설정

| 설정 | 값 | 위치 |
|------|-----|------|
| `SEQ_LEN` | 2048 | `generate_for_target()` L341 |
| `MAX_CTX` | 1748 (= 2048 - 300) | `generate_for_target()` L342 |

---

## 17. 테스트 및 검증

### MIDI 프로세서 테스트

```bash
python -m pytest tests/test_midi_processor.py -v
```

18개 테스트: Type 0/1 재매핑, 삭제, program_change 삽입, 에지케이스

### MIDI 오염 분석 테스트

```bash
python -m pytest tests/test_midi_corruption_analysis.py -v
```

16개 테스트: delta time 보존, 참조 오염 방지, 메타 데이터 보호 등

### 추론 통합 테스트 (GPU 필요)

현재 자동화된 추론 테스트는 없습니다. GPU 환경에서 수동으로 검증하세요:

```python
from app.services.inference import build_v5_vocab, load_model, run_arrangement

vocab   = build_v5_vocab()
vocab_r = {v: k for k, v in vocab.items()}
model   = load_model("/models/best", len(vocab), vocab, "cuda")

result = run_arrangement(
    song_path="test.mid",
    target="violin",
    genre="CLASSICAL",
    temperature=1.0,
    output_path="output.mid",
    model=model, vocab=vocab, vocab_r=vocab_r, device="cuda"
)
```

### 레퍼런스 코드

`local_docs/inference_new.py`: 원본 CLI 기반 추론 코드. 서버 버전(`app/services/inference.py`)의 원본이며, 코드 오염 여부 비교 시 이 파일을 기준으로 사용합니다.
