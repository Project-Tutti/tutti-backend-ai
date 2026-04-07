from pydantic import BaseModel, HttpUrl, Field
from typing import Optional, List, Literal


class Mapping(BaseModel):
    trackIndex: int
    targetInstrumentId: int


# Vocabulary에 정의된 유효 장르 목록
GenreType = Literal[
    "CLASSICAL", "JAZZ", "POP", "ROCK", "ELECTRONIC", "FOLK", "UNKNOWN"
]


class ArrangeRequest(BaseModel):
    projectId: int
    versionId: int
    midiFilePath: HttpUrl
    mappings: List[Mapping]
    # ── 신규 필수 필드 ──
    targetInstrumentId: int                    # AI 추론 대상 악기 (필수)
    minNote: Optional[int] = None              # 음역 최솟값 (None이면 악기 기본값)
    maxNote: Optional[int] = None              # 음역 최댓값 (None이면 악기 기본값)
    modelType: Optional[str] = None            # 모델 선택 (None이면 기본 모델)
    genre: GenreType = Field(default="CLASSICAL")    # 장르 (유효값만 허용)
    temperature: float = Field(default=1.0, ge=0.1, le=2.0)  # 다양성
    callbackUrl: HttpUrl
    callbackSecret: str

