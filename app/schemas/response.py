from pydantic import BaseModel
from typing import List


class ArrangeResponse(BaseModel):
    status: str
    message: str


class HealthResponse(BaseModel):
    status: str
    loaded_models: List[str]
