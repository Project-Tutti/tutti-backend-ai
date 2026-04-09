import os
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader

_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

_EXPECTED_KEY = os.getenv("AI_SERVER_API_KEY", "")


async def verify_api_key(
    api_key: str | None = Security(_api_key_header),
) -> str:
    """
    X-API-Key 헤더를 검증합니다.
    AI_SERVER_API_KEY 환경변수가 비어있으면 인증을 건너뜁니다 (로컬 개발용).
    """
    if not _EXPECTED_KEY:
        # 환경변수 미설정 시 인증 스킵 (로컬 개발)
        return "dev"

    if not api_key or api_key != _EXPECTED_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key
