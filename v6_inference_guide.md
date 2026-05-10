# Tutti V6 Inference & Training Guide

## 1. 개요 (Overview)
이 문서는 Qwen2.5-0.5B 기반 Tutti V6 모델의 추론(Inference) 파이프라인과 데이터 전처리(Preprocessing) 동기화 내역을 정리한 가이드입니다. 모델이 학습한 데이터 구조와 정확히 동일하게 추론 환경을 구성하여 텐서 오류를 방지하고 편곡 품질을 극대화하는 데 목적이 있습니다.

## 2. 주요 아키텍처 변경 사항 (Core Architecture Changes)
### 2.1. 시퀀스 길이 (Sequence Length) 확장
- 기존 2048 토큰 제한에서 모델의 최대 수용량인 **8192** 토큰으로 확장 (`SEQ_LEN = 8192`, `max_position_embeddings = 8192`).
- 이로써 한 번에 넓은 문맥(최대 8~16마디 이상)의 악보를 읽고 일관성 있는 편곡이 가능해졌습니다.

### 2.2. Vocab 최적화 (645 토큰)
- `vocab_size`를 645로 고정 (`build_v6_vocab()`).
- 기존 추론 코드에 남아있어 에러를 유발하던 미학습 구조 토큰(`PHRASE_END` 등)을 전면 제거하여 토큰 인덱스(Token Index) 불일치 문제를 해결했습니다.

## 3. FIM (Fill-In-the-Middle) 동기화
기존의 단순 이어붙이기 프롬프트를 버리고, 학습 스크립트(`preprocess_lmd.py`)와 100% 동일한 **FIM 구조**를 도입했습니다.

### 3.1. 프롬프트 구조 (Prompt Structure)
- **Header**: `[PIECE_START, GENRE_*, TARGET_*]` (타겟 악기를 헤더에 명시하여 정확도 향상)
- **Past (과거 8마디)**: `<PRE>` 토큰과 함께 제공
- **Future (미래 8마디)**: `<SUF>` 토큰과 함께 제공
- **Current (현재 8마디)**: `<MID>` 토큰 뒤에 모델이 생성 (`window_bars=8`)

### 3.2. 반주(Guide Track) 인식 개선
- **기존 문제**: 전처리 시 `suf_start = mid_end`로 설정되어 모델이 "현재 작곡해야 할 마디의 다른 악기(반주)"를 듣지 못했습니다.
- **해결 방안**: 전처리 코드에서 **`suf_start = mid_start`**로 수정하여, 모델이 현재 마디의 피아노, 베이스 등 다른 악기들의 흐름을 읽고 화성에 맞춰 편곡하도록 재학습 환경을 구축했습니다.
- **구체화**: 현재 학습 인코딩 전처리 다시 하고, resume 학습 예정(큰 문제는 아니라서, 지금 학습된 모델에서 이어서 학습 가능함 - 시간문제 없음)

## 4. CLI 래퍼(Wrapper) 사용법
별도의 API 서버 없이 커맨드라인에서 곧바로 추론 로직을 테스트할 수 있는 `run_inference.py`를 제공합니다.

### 실행 방법
```bash
source ~/music-env/bin/activate
python ~/Qwen2.5_0.5B/run_inference.py \
    --input Havana.mid \
    --target 42 \
    --genre POP \
    --pitch_min 36 \
    --pitch_max 72
```

### 주요 파라미터
- `--input`: 입력 MIDI 파일명 (기본적으로 `~/INPUT/` 폴더 참조)
- `--target`: 타겟 악기의 MIDI Program 번호 (예: 42=Cello, 40=Violin)
- `--genre`: 곡의 장르 (POP, CLASSICAL 등)
- `--pitch_min` / `--pitch_max`: 악기의 실제 음역대에 맞추어 생성될 음표의 한계를 설정 (예: 첼로 36~72)
- `--ckpt`: 체크포인트 경로 (기본값: `/data2/tutti/Qwen_Checkpoints/checkpoint-65000`)

## 5. 파인튜닝 (Fine-tuning) 가이드
1. `preprocess_lmd.py`를 통해 데이터를 생성하면 `tmux` 세션에서 자동으로 `jsonl_to_memmap.py`가 이어져 `.bin` 파일이 생성됩니다.
2. 생성이 완료되면 `train.py`를 실행하여 기존 체크포인트(`checkpoint-65000`)부터 이어서 학습을 진행하세요.
3. 새로운 모델은 "반주의 흐름"을 듣는 능력이 추가되어 기존 대비 월등한 오케스트레이션 성능을 보여줄 것입니다.
