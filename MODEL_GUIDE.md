# Tutti AI 모델 반영 및 레지스트리(registry.json) 관리 가이드 (On-Premise)

이 문서는 온프레미스 서버(물리 서버) 환경에서 새로운 AI 모델 가중치(Weight) 파일을 서버에 넣고, 이를 서버가 인식하도록 `registry.json`을 수정하여 자동 배포하는 방법을 설명합니다.

---

## 1. 개요: 2-Track 배포 방식

온프레미스 아키텍처에서는 용량이 큰 모델 파일과 가벼운 설정 파일의 배포 방식이 분리되어 있습니다.

| 항목 | 설명 | 배포 방법 |
| --- | --- | --- |
| **모델 가중치 (Weight)** | 수 기가바이트의 실제 모델 파일 (`.safetensors`, `.bin`, `.pt`) | 온프레미스 서버에 **직접 복사 (USB, SCP 등)** |
| **registry.json** | 어떤 폴더의 어떤 모델을 쓸지 정의하는 설정 파일 | GitHub에 커밋 후 **Push (CI/CD 자동 반영)** |

---

## 2. 1단계: 모델 파일 서버에 직접 넣기

모델의 크기가 매우 크므로 GitHub을 통하지 않고 서버에 직접 넣습니다.

1. 새로운 모델이 포함된 폴더(예: `new_best_model/`)를 준비합니다.
2. 온프레미스 서버의 `~/tutti-backend-ai/models/` 경로 안으로 해당 폴더를 복사합니다.

**서버 접속 후 파일 복사 예시:**
```bash
# SCP 등을 통해 로컬에서 서버로 전송
scp -r ./new_best_model globaltutti@<온프레미스_IP>:~/tutti-backend-ai/models/
```

**최종 디렉토리 구조 예시:**
```text
~/tutti-backend-ai/models/
├── best/               # (기존에 쓰던 모델)
├── new_best_model/     # (새롭게 추가한 무거운 모델 폴더)
└── registry.json       # (Git 자동 동기화됨. 직접 건드리지 마세요!)
```

---

## 3. 2단계: registry.json 수정 및 배포 (GitHub)

실제 모델 파일이 서버에 들어갔다면, 이제 서버가 새 모델을 로드하도록 설정 파일을 수정해야 합니다. 
이 작업은 **작업용 로컬 PC에서 깃허브 프로젝트를 열고 진행**합니다.

### 3.1. 루트에 있는 `registry.json` 수정
에디터를 열어 프로젝트 최상위 루트에 있는 `registry.json`을 수정합니다.

```json
{
  "version": "v2",
  "default": "qwen2.5",
  "models": [
    {
      "type": "qwen2.5",
      "name": "Tutti Unified v2 (신규 버전)",
      "path": "new_best_model", 
      "description": "새롭게 학습된 대규모 모델 적용",
      "active": true
    }
  ]
}
```
- `path`: 1단계에서 서버에 복사해 둔 폴더명(예: `new_best_model`)을 적습니다.
- `active`: (선택) `false`로 설정하면 서버 메모리에 해당 모델을 로드하지 않고 완전히 건너뜁니다. 과거 모델의 정보만 백업용으로 남겨둘 때 유용합니다.

> [!CAUTION]
> **동일한 `type`을 가진 모델을 두 개 이상 동시에 활성화할 수 없습니다!**
> 서버는 메모리에 모델을 로드할 때 `"type"` 값을 고유 식별자(Key)로 사용합니다. 따라서 `"active": true` 상태인 배열 내의 모델들 중 `"type": "qwen2.5"`처럼 똑같은 타입 값이 중복으로 존재해서는 안 됩니다 (나중에 작성된 모델로 덮어씌워짐).  
> 여러 개의 모델 정보를 적어두려면 반드시 사용하지 않는 예전 버전을 `"active": false`로 명시하세요.

### 3.2. GitHub 커밋 및 푸시 (자동 배포)
파일 저장이 끝났다면, Git에 커밋하고 `main` 브랜치에 Push합니다.

```bash
git add registry.json
git commit -m "feat: 모델 버전을 new_best_model로 교체"
git push origin main
```

---

## 4. CI/CD 자동화 메커니즘 🤖 (FAQ)

**Q. `registry.json` 파일 하나만 수정해서 푸시해도 알아서 전체가 배포되나요?**
네, 완벽하게 통째로 돌아갑니다! 
`main` 브랜치에 푸시가 발생하면 GitHub Actions(`ci-ai.yml`)가 즉시 실행되어 다음 과정을 자동으로 수행합니다.

1. **GitHub Action 시작**: `registry.json` 변경 사항 감지
2. **Docker Build/Push 생략**: 앱 소스는 바뀌지 않았으므로 엄청나게 빠른 속도로 캐시를 통과합니다.
3. **온프레미스 서버 원격 접속**: 
   - 서버 내부에서 자동으로 `git pull`을 실행하여 최신 `registry.json`을 받아옵니다.
   - 받아온 최신 `.json` 파일을 컨테이너가 볼 수 있는 `models/` 내부로 즉시 강제 덮어쓰기(`cp -f`) 합니다.
   - AI 서버 컨테이너들을 **순환 재시작(Rolling Update 느낌의 재시작)** 시켜 새로운 모델을 즉시 메모리에 로드하게 만듭니다.

모든 과정이 끝나면 몇 초 후 외부에서 API를 찔러봤을 때 곧바로 새로운 버전에 등록된 텍스트와 모델 아키텍처로 응답하게 됩니다!
