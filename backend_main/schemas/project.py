import uuid
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel
from backend_main.schemas.enums import DurationEnum, AspectRatioEnum, StyleEnum
from backend_main.schemas.media import MediaResponse

class ProjectCreate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None

class ProjectResponse(BaseModel):
    id: uuid.UUID
    title: Optional[str]
    description: Optional[str]
    target_duration: Optional[int] = None
    aspect_ratio: Optional[str] = None
    style: Optional[str] = None
    caption_style: Optional[str] = None
    music_id: Optional[uuid.UUID]
    created_at: datetime

class ProjectDetailResponse(ProjectResponse):
    media: List[MediaResponse]

class ProjectListItem(BaseModel):
    id: uuid.UUID
    title: Optional[str]
    description: Optional[str]
    target_duration: Optional[int] = None
    aspect_ratio: Optional[str] = None
    style: Optional[str] = None
    caption_style: Optional[str] = None
    created_at: Optional[datetime]
    render_status: str  # not_started | running | done | error
