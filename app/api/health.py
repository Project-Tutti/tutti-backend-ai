from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from app.schemas.response import HealthResponse

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    registry = request.app.state.registry
    models = registry.list_models()
    if not models:
        return JSONResponse(
            {"status": "unhealthy", "loaded_models": []},
            status_code=503,
        )
    return HealthResponse(
        status="ok",
        loaded_models=models,
    )
