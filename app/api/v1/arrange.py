from fastapi import APIRouter, BackgroundTasks, Request
from app.schemas.request import ArrangeRequest
from app.schemas.response import ArrangeResponse
from app.services.arrangement import process_arrangement

router = APIRouter()


@router.post("/arrange", response_model=ArrangeResponse)
async def arrange(
    request_data: ArrangeRequest, background_tasks: BackgroundTasks, request: Request
):
    """
    편곡 요청 수신 - 바로 반환하고 백그라운드에서 추론 진행.
    """
    registry = request.app.state.registry
    background_tasks.add_task(process_arrangement, request_data, registry)
    return ArrangeResponse(status="accepted", message="편곡 요청을 수신했습니다.")
