import uuid
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel
from backend_main.schemas.enums import DurationEnum, AspectRatioEnum, StyleEnum
from backend_main.schemas.media import MediaResponse

class ProjectCreate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    target_duration: DurationEnum
    aspect_ratio: AspectRatioEnum = AspectRatioEnum.nine_sixteen
    style: Optional[StyleEnum] = None
    caption_style: Optional[str] = "hormozi"

class ProjectResponse(BaseModel):
    id: uuid.UUID
    title: Optional[str]
    description: Optional[str]
    target_duration: int
    aspect_ratio: str
    style: Optional[str]
    caption_style: Optional[str]
    music_id: Optional[uuid.UUID]
    created_at: datetime

class ProjectDetailResponse(ProjectResponse):
    media: List[MediaResponse]

class ProjectListItem(BaseModel):
    id: uuid.UUID
    title: Optional[str]
    description: Optional[str]
    target_duration: int
    aspect_ratio: str
    style: Optional[str]
    caption_style: Optional[str]
    created_at: Optional[datetime]
    render_status: str  # not_started | running | done | error
