import httpx
import logging
import asyncio
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


async def send_callback(
    callback_url: str, callback_secret: str, payload: dict, max_retries: int = 3
):
    """
    main-server에 콜백 전송 (JSON만). 실패 시 지수 백오프로 재시도.
    진행률(processing) 콜백에 사용됩니다.
    """
    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    str(callback_url),
                    json=payload,
                    headers={"X-Callback-Secret": callback_secret},
                )
                response.raise_for_status()
                logger.info(
                    f"콜백 전송 성공: {payload.get('status')} (progress: {payload.get('progress')})"
                )
                return
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(
                    f"콜백 재시도 {attempt + 1}/{max_retries}: {e}. {wait_time}초 대기"
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"콜백 최종 실패: {payload}. 에러: {e}")


async def send_callback_with_file(
    callback_url: str,
    callback_secret: str,
    payload: dict,
    file_path: Path,
    max_retries: int = 5,
):
    """
    main-server에 콜백 전송 + MIDI 파일 첨부 (multipart/form-data).
    완료(complete) 콜백에 사용되며, KEDA Zero-Scaling 환경에서도 안전합니다.

    전송 형식:
      - Part 1 "metadata": JSON (projectId, versionId, status, progress 등)
      - Part 2 "file": MIDI 바이너리 파일
    """
    if not file_path.exists():
        logger.error(f"전송할 파일이 존재하지 않습니다: {file_path}")
        return

    for attempt in range(max_retries):
        try:
            async with httpx.AsyncClient(timeout=60) as client:
                with open(file_path, "rb") as f:
                    files = {
                        "file": (file_path.name, f, "audio/midi"),
                    }
                    data = {
                        "metadata": _to_json_str(payload),
                    }
                    response = await client.post(
                        str(callback_url),
                        data=data,
                        files=files,
                        headers={"X-Callback-Secret": callback_secret},
                    )
                    response.raise_for_status()
                    logger.info(
                        f"파일 첨부 콜백 전송 성공: {file_path.name} ({file_path.stat().st_size} bytes)"
                    )
                    return
        except Exception as e:
            if attempt < max_retries - 1:
                wait_time = 2**attempt
                logger.warning(
                    f"파일 콜백 재시도 {attempt + 1}/{max_retries}: {e}. {wait_time}초 대기"
                )
                await asyncio.sleep(wait_time)
            else:
                logger.error(f"파일 콜백 최종 실패: {file_path.name}. 에러: {e}")


def _to_json_str(payload: dict) -> str:
    """dict를 JSON 문자열로 변환"""
    import json

    return json.dumps(payload)
