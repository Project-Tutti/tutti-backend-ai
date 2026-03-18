import httpx
import logging
import asyncio

logger = logging.getLogger(__name__)


async def send_callback(
    callback_url: str, callback_secret: str, payload: dict, max_retries: int = 3
):
    """
    main-server에 콜백 전송. 실패 시 지수 백오프로 재시도.
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
                # Don't strictly raise. main app handles progress callback failure.
                # Just log and continue if it's not fatal, but raising might be better
                # if main-server must receive it to unblock downstream.
                pass
