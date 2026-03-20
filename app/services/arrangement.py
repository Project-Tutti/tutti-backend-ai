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

        # Step 2: MIDI 파싱 — anticipation 이벤트 + controls 추출 (20%)
        logger.info(f"[{job_id}] Step 2: Parsing MIDI")
        events, controls, song_length = parse_midi(midi_path)
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
        #   controls: 원곡의 모든 악기 이벤트를 컨텍스트로 제공
        #   song_length: 곡 길이(초) — 생성 종료 시점 결정
        logger.info(f"[{job_id}] Step 3: Inference Loop ({total_tracks} tracks)")
        results = []
        target_instrument_ids = []
        for i, mapping in enumerate(request.mappings):
            try:
                model = registry.get_model(mapping.targetInstrumentId)

                logger.info(
                    f"[{job_id}] Inferencing track {mapping.trackIndex} "
                    f"for target instrument {mapping.targetInstrumentId}"
                )

                # controls를 source_midi_events로 전달하여
                # 모델이 원곡의 맥락을 참조하며 생성하도록 함
                loop = asyncio.get_running_loop()
                result = await loop.run_in_executor(
                    None,
                    run_inference,
                    model,
                    mapping.targetInstrumentId,
                    controls,
                    song_length,
                )
                results.append(result)
                target_instrument_ids.append(mapping.targetInstrumentId)

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

        # Step 4: 결과 병합 — 생성된 이벤트들을 원본 MIDI에 주입 (90%)
        logger.info(f"[{job_id}] Step 4: Merging {len(results)} tracks")
        output_path = Path(settings.RESULTS_DIR) / f"{job_id}.mid"
        merged_midi_path = merge_tracks(
            results, midi_path, target_instrument_ids, output_path
        )

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
