"""AI Core — AI 개발자 전용 영역.

이 패키지 내부의 코드는 AI 개발자가 자유롭게 수정할 수 있습니다.
단, contracts/interfaces.py에 정의된 시그니처를 반드시 준수해야 합니다.

Note: 무거운 의존성(torch, transformers)을 가진 모듈은
      사용 시점에 import하여 로컬 개발환경에서의 부분 테스트를 지원합니다.
"""


def __getattr__(name):
    """Lazy imports to avoid importing torch at package load time.

    이렇게 하면 `from ai_core.constants import ...` 처럼
    torch 없는 모듈만 사용할 때도 에러가 발생하지 않습니다.
    """
    if name == "run_arrangement":
        from ai_core.arrangement import run_arrangement
        return run_arrangement
    elif name == "resolve_target":
        from ai_core.arrangement import resolve_target
        return resolve_target
    elif name == "build_v5_vocab":
        from ai_core.vocab import build_v5_vocab
        return build_v5_vocab
    elif name == "load_model":
        from ai_core.model_loader import load_model
        return load_model
    raise AttributeError(f"module 'ai_core' has no attribute {name!r}")


__all__ = ["run_arrangement", "resolve_target", "build_v5_vocab", "load_model"]
