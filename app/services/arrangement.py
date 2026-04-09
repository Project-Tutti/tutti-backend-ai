"""
arrangement.py
편곡 전체 라이프사이클 — 단일 타겟 1회 추론.

기존: mappings 루프 → 트랙별 개별 모델 + 추론
신규: mappings → 원본 트랙 재매핑만, targetInstrumentId 기준 1회 추론
"""

import asyncio
import uuid
import logging
from pathlib import Path

from app.schemas.request import ArrangeRequest
from app.services.callback import send_callback, send_callback_with_file
from app.services.midi_processor import download_midi, remap_original_tracks
from app.services.inference import run_arrangement, resolve_target
from app.core.config import settings

logger = logging.getLogger(__name__)


async def process_arrangement(request: ArrangeRequest, registry):
    """편곡 전체 라이프사이클 — 진행률을 콜백으로 main-server에 전송"""
    job_id = str(uuid.uuid4())
    midi_path = None
    cb = str(request.callbackUrl)
    secret = request.callbackSecret
    project_id = request.projectId
    version_id = request.versionId

    try:
        # Step 1: MIDI 다운로드 (10%)
        logger.info(f"[{job_id}] Step 1: Downloading MIDI")
        midi_path = await download_midi(request.midiFilePath)
        await send_callback(cb, secret, {
            "projectId": project_id,
            "versionId": version_id,
            "status": "processing",
            "progress": 10,
        })

        # Step 2: 원본 트랙 재매핑 (20%)
        #   mappings의 targetInstrumentId에 따라:
        #   - 129 → 해당 트랙/채널 삭제
        #   - 그 외 → program_change 변경
        logger.info(f"[{job_id}] Step 2: Remapping original tracks")
        remap_original_tracks(midi_path, request.mappings)
        await send_callback(cb, secret, {
            "projectId": project_id,
            "versionId": version_id,
            "status": "processing",
            "progress": 20,
        })

        # Step 3: 추론 — 단일 타겟 1회 생성 (80%)
        #   targetInstrumentId → INSTRUMENT_GROUPS 키로 변환
        #   모델은 modelType으로 선택 (None이면 기본 모델)
        target_name = resolve_target(request.targetInstrumentId)
        loaded = registry.get_model(request.modelType)  # LoadedModel

        logger.info(
            f"[{job_id}] Step 3: Inference "
            f"(target={target_name}, model={loaded.name}, "
            f"genre={request.genre}, temp={request.temperature})"
        )

        output_path = Path(settings.RESULTS_DIR) / f"{job_id}.mid"

        loop = asyncio.get_running_loop()
        
        def inference_progress_hook(pct: int):
            # 동기 스레드인 run_arrangement 내부에서 호출되어 비동기 콜백 전송
            asyncio.run_coroutine_threadsafe(
                send_callback(cb, secret, {
                    "projectId": project_id,
                    "versionId": version_id,
                    "status": "processing",
                    "progress": pct,
                }),
                loop
            )

        result_path = await loop.run_in_executor(
            None,
            run_arrangement,
            str(midi_path),         # song_path
            target_name,            # target
            request.genre,          # genre
            request.temperature,    # temperature
            request.minNote,        # pitch_min (None이면 기본값)
            request.maxNote,        # pitch_max (None이면 기본값)
            str(output_path),       # output_path
            loaded.model,           # 사전 로드된 모델
            loaded.vocab,           # vocab
            loaded.vocab_r,         # vocab_r
            loaded.device,          # device
            inference_progress_hook # progress_hook
        )

        await send_callback(cb, secret, {
            "projectId": project_id,
            "versionId": version_id,
            "status": "processing",
            "progress": 80,
        })

        # Step 4: 콜백 전송 (100%)
        logger.info(f"[{job_id}] Step 4: Sending result file")
        await send_callback_with_file(cb, secret, {
            "projectId": project_id,
            "versionId": version_id,
            "status": "complete",
            "progress": 100,
        }, file_path=Path(result_path))

        logger.info(f"[{job_id}] Arrangement completed successfully")

    except Exception as e:
        logger.error(f"[{job_id}] Arrangement Failed: {e}", exc_info=True)
        await send_callback(cb, secret, {
            "projectId": project_id,
            "versionId": version_id,
            "status": "failed",
            "progress": 0,
            "errorMessage": str(e),
        })
    finally:
        # 임시 파일 정리
        try:
            result_file = Path(settings.RESULTS_DIR) / f"{job_id}.mid"
            if result_file.exists():
                result_file.unlink()
                logger.debug(f"[{job_id}] 임시 결과 파일 삭제: {result_file}")
        except Exception:
            pass
        # 다운로드된 원본 MIDI 정리
        try:
            if midi_path and midi_path.exists():
                midi_path.unlink()
                logger.debug(f"[{job_id}] 다운로드 MIDI 삭제: {midi_path}")
        except Exception:
            pass
