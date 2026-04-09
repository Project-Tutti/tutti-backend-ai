from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pathlib import Path
from app.core.config import settings
from app.core.auth import verify_api_key

router = APIRouter()


@router.get("/download/{job_id}")
async def download_result(job_id: str, _: str = Depends(verify_api_key)):
    """main-server가 결과 MIDI를 다운로드하는 엔드포인트"""
    result_path = Path(settings.RESULTS_DIR) / f"{job_id}.mid"
    if not result_path.exists():
        raise HTTPException(404, "결과 파일을 찾을 수 없습니다.")
    return FileResponse(result_path, media_type="audio/midi", filename=f"{job_id}.mid")
