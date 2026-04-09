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
    """
    if not _EXPECTED_KEY:
        # 서버 환경변수 자체가 누락된 치명적 에러 상황
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: AI_SERVER_API_KEY is missing.",
        )

    if not api_key or api_key != _EXPECTED_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key",
        )
    return api_key
