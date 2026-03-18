from fastapi import APIRouter, Request
from app.schemas.response import HealthResponse, LoadedInstrument

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    registry = request.app.state.registry
    instruments = [
        LoadedInstrument(
            midi_program=inst.midi_program, name=inst.name, category=inst.category
        )
        for inst in registry.list_instruments()
    ]
    return HealthResponse(status="ok", loaded_instruments=instruments)
