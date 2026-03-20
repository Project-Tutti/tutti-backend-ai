import asyncio
import uuid
import logging
from pathlib import Path
from typing import Optional

from app.schemas.request import ArrangeRequest
from app.services.callback import send_callback, send_callback_with_file
from app.services.midi_processor import download_midi, parse_midi, merge_tracks
from app.services.inference import run_inference
from app.core.config import settings

logger = logging.getLogger(__name__)


async def process_arrangement(request: ArrangeRequest, registry):
    """편곡 전체 라이프사이클 — 진행률을 콜백으로 main-server에 전송"""
    job_id = str(uuid.uuid4())
    cb = str(request.callbackUrl)
    secret = request.callbackSecret
    project_id = request.projectId
    version_id = request.versionId
    total_tracks = len(request.mappings)

    try:
        # Step 1: MIDI 다운로드 (10%)
        logger.info(f"[{job_id}] Step 1: Downloading MIDI")
        midi_path = await download_midi(request.midiFilePath)
        await send_callback(
            cb,
            secret,
            {
                "projectId": project_id,
                "versionId": version_id,
                "status": "processing",
                "progress": 10,
            },
        )

        # Step 2: MIDI 파싱 (20%)
        logger.info(f"[{job_id}] Step 2: Parsing MIDI")
        tracks = parse_midi(midi_path)
        await send_callback(
            cb,
            secret,
            {
                "projectId": project_id,
                "versionId": version_id,
                "status": "processing",
                "progress": 20,
            },
        )

        # Step 3: 트랙별 추론 (20%~80%, 균등 분배)
        logger.info(f"[{job_id}] Step 3: Inference Loop")
        results = []
        for i, mapping in enumerate(request.mappings):
            try:
                model = registry.get_model(mapping.targetInstrumentId)
                track_data = (
                    tracks[mapping.trackIndex]
                    if mapping.trackIndex < len(tracks)
                    else []
                )

                song_length_seconds = 180.0

                logger.info(
                    f"[{job_id}] Inferencing track {mapping.trackIndex} for target {mapping.targetInstrumentId}"
                )
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    run_inference,
                    model,
                    mapping.targetInstrumentId,
                    track_data,
                    song_length_seconds,
                )
                results.append(result)
            except Exception as e:
                logger.error(
                    f"[{job_id}] Failed to infer track {mapping.trackIndex}: {e}"
                )
                raise e

            # 진행률 균등 분배: 20% + (60% * (i+1) / total_tracks)
            progress = 20 + int(60 * (i + 1) / total_tracks)
            await send_callback(
                cb,
                secret,
                {
                    "projectId": project_id,
                    "versionId": version_id,
                    "status": "processing",
                    "progress": progress,
                },
            )

        # Step 4: 결과 병합 (90%)
        logger.info(f"[{job_id}] Step 4: Merging results")
        merged_midi_path = Path(settings.RESULTS_DIR) / f"{job_id}.mid"
        merged_midi_path.parent.mkdir(parents=True, exist_ok=True)

        merge_tracks(results, midi_path)

        # TODO: 실제 merge_tracks 결과를 파일로 저장하도록 수정 필요
        with open(merged_midi_path, "wb") as f:
            f.write(b"MOCK_MIDI_DATA")

        await send_callback(
            cb,
            secret,
            {
                "projectId": project_id,
                "versionId": version_id,
                "status": "processing",
                "progress": 90,
            },
        )

        # Step 5: 완료 — MIDI 파일을 직접 첨부하여 콜백 전송 (100%)
        # KEDA Zero-Scaling 환경에서도 안전: 파일이 메인 서버에 직접 도착하므로
        # AI 파드가 삭제되어도 결과물이 유실되지 않습니다.
        logger.info(f"[{job_id}] Step 5: Sending result file to main-server")
        await send_callback_with_file(
            cb,
            secret,
            {
                "projectId": project_id,
                "versionId": version_id,
                "status": "complete",
                "progress": 100,
            },
            file_path=merged_midi_path,
        )

        logger.info(f"[{job_id}] Arrangement completed successfully")

    except Exception as e:
        logger.error(f"[{job_id}] Arrangement Failed: {e}", exc_info=True)
        await send_callback(
            cb,
            secret,
            {
                "projectId": project_id,
                "versionId": version_id,
                "status": "failed",
                "progress": 0,
                "errorMessage": str(e),
            },
        )
    finally:
        # 결과 파일 정리 — 메인 서버에 전송 완료 후 로컬 파일 삭제
        try:
            result_file = Path(settings.RESULTS_DIR) / f"{job_id}.mid"
            if result_file.exists():
                result_file.unlink()
                logger.debug(f"[{job_id}] 임시 결과 파일 삭제: {result_file}")
        except Exception:
            pass
