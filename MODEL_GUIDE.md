# Tutti AI 모델 등록 및 업로드 가이드

이 문서는 GCS 버킷에 AI 모델을 업로드하고, AI 서버가 이를 인식할 수 있도록 `registry.json`을 작성하는 방법을 설명합니다.

## 1. 모델 파일 준비 및 이름 규칙

AI 서버는 PyTorch 모델(`.pt` 또는 `.pth`)을 사용합니다. 모델 파일의 이름은 **직관적이고 영어 소문자 및 언더스코어(`_`)** 로 작성하는 것을 권장합니다.
특정 악기나 버전을 명시하면 관리가 편해집니다.

- **권장 이름 규칙**: `[악기명]_[버전].pt`
- **예시**:
  - `piano_v1.pt`
  - `violin_v2.pt`
  - `acoustic_guitar_v1.pt`

## 2. registry.json 작성 방법

`registry.json`은 AI 서버가 켜질 때 어떤 모델 파일을 불러와서 어떤 MIDI Program ID(악기 번호)와 매핑할지 알려주는 "명세서"입니다.

### 파일 구조 (예시)

```json
{
  "version": "v1",
  "instruments": [
    {
      "midi_program": 0,
      "name": "Acoustic Grand Piano",
      "category": "Piano",
      "model_file": "piano_v1.pt",
      "model_type": "pytorch"
    },
    {
      "midi_program": 40,
      "name": "Violin",
      "category": "Strings",
      "model_file": "violin_v2.pt",
      "model_type": "pytorch"
    }
  ]
}
```

### 필수 필드 설명

`instruments` 배열 안의 각 객체는 하나의 악기 모델을 정의합니다.

- `midi_program` (정수): **가장 중요한 필드입니다**. MIDI 표준 악기 번호 (0~127)를 지정합니다. 사용자가 프론트엔드에서 이 악기를 선택하면, 서버는 이 ID와 일치하는 모델을 찾습니다. (예: 피아노는 `0`, 어쿠스틱 기타는 `25`, 바이올린은 `40` 등)
- `name` (문자열): 악기의 이름입니다. 사람이 읽기 위한 목적입니다.
- `category` (문자열): 악기가 속한 분류입니다. (예: `Piano`, `Strings`, `Brass` 등)
- `model_file` (문자열): 해당 악기를 연주(추론)할 실제 모델 파일의 **정확한 파일명**입니다. `.pt`까지 모두 적어야 합니다.
- `model_type` (문자열): 항상 `"pytorch"` 로 설정합니다.

## 3. GCS에 업로드 방법

모델 파일들과 `registry.json` 파일이 준비되었다면, Google Cloud Storage(GCS) 버킷의 `v1/` 경로에 업로드해야 합니다.

### 설정 전제 조건

- 앞서 `setup_gcs.sh`를 통해 `tutti-ai-models` 버킷이 생성되어 있어야 합니다.
- 아래 명령어들은 모델 파일들이 있는 로컬 디렉토리에서 실행합니다.

### 터미널 명령어

**1) 파일 업로드하기**

```bash
# 레지스트리 설정 파일 업로드
gsutil cp registry.json gs://tutti-ai-models/v1/registry.json

# 모델 1번 (피아노) 업로드
gsutil cp piano_v1.pt gs://tutti-ai-models/v1/piano_v1.pt

# 모델 2번 (바이올린) 업로드
gsutil cp violin_v2.pt gs://tutti-ai-models/v1/violin_v2.pt
```

---

> 💡 **참고 (Init Container)**
> 파드가 다시 시작될 때마다, `deployment.yaml`에 정의된 `Init Container`가 작동하여 위 GCS 버킷(`gs://tutti-ai-models/v1/*`)에 있는 모든 파일을 컨테이너 내부의 `/models/` 디렉토리로 다운로드합니다. 이 과정이 무사히 끝나야 FastAPI 메인 서버가 `registry.json`을 읽고 모델을 메모리에 로드합니다.
