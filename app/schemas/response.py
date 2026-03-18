from pydantic import BaseModel
from typing import List, Any


class ArrangeResponse(BaseModel):
    status: str
    message: str


class LoadedInstrument(BaseModel):
    midi_program: int
    name: str
    category: str


class HealthResponse(BaseModel):
    status: str
    loaded_instruments: List[LoadedInstrument]
