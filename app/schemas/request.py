from pydantic import BaseModel, HttpUrl
from typing import List


class Mapping(BaseModel):
    trackIndex: int
    targetInstrumentId: int


class ArrangeRequest(BaseModel):
    projectId: int
    versionId: int
    midiFilePath: HttpUrl
    mappings: List[Mapping]
    callbackUrl: HttpUrl
    callbackSecret: str
