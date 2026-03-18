import asyncio
import uuid
import logging
from pathlib import Path
from typing import Optional

from app.schemas.request import ArrangeRequest
from app.services.callback import send_callback
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
        # In a real scenario, use asyncio.gather for parallelization if models are distinct or batchable
        for i, mapping in enumerate(request.mappings):
            try:
                model = registry.get_model(mapping.targetInstrumentId)
                # Mocking the tracks logic for orchestration integration
                track_data = (
                    tracks[mapping.trackIndex]
                    if mapping.trackIndex < len(tracks)
                    else []
                )

                # We need the song length to cap generation
                # In robust implementation, this would come from parsing MIDI
                song_length_seconds = 180.0

                logger.info(
                    f"[{job_id}] Inferencing track {mapping.trackIndex} for target {mapping.targetInstrumentId}"
                )
                # Blocking CPU bound task. Offload to an executor.
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
                # Decide if one failure fails the whole job or skips it. For now, continue but maybe should fail.
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

        # Call merger
        merge_tracks(results, midi_path)

        # Emulating saving the result
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

        # Step 5: 완료 (100%)
        logger.info(f"[{job_id}] Step 5: Completed successfully")
        await send_callback(
            cb,
            secret,
            {
                "projectId": project_id,
                "versionId": version_id,
                "status": "complete",
                "progress": 100,
                "resultMidiPath": f"/api/v1/download/{job_id}",
            },
        )

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
        # Cleanup original download mapping if it's there
        pass
