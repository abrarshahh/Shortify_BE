from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime
from enum import Enum

class DurationEnum(int, Enum):
    fifteen = 15
    thirty = 30
    sixty = 60

class AspectRatioEnum(str, Enum):
    nine_sixteen = "9:16"
    one_one = "1:1"
    sixteen_nine = "16:9"

class StyleEnum(str, Enum):
    travel = "travel"
    cinematic = "cinematic"
    fast_cut = "fast_cut"
    birthday = "birthday"
    adventure = "adventure"
    romantic = "romantic"
    funny = "funny"
    dramatic = "dramatic"

class ProjectCreate(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    target_duration: DurationEnum
    aspect_ratio: AspectRatioEnum = AspectRatioEnum.nine_sixteen
    style: Optional[StyleEnum] = None

class ProjectResponse(BaseModel):
    id: str
    title: Optional[str]
    description: Optional[str]
    target_duration: int
    aspect_ratio: str
    style: Optional[str]
    music_id: Optional[str]
    created_at: datetime

class MediaResponse(BaseModel):
    id: str
    original_filename: str
    storage_path: str
    mime_type: str
    file_size: int
    duration: Optional[int] = None
    thumbnail_path: Optional[str] = None
    uploaded_at: datetime
    project_ids: List[str] = []

class UploadItem(BaseModel):
    id: str
    path: str

class UploadResponse(BaseModel):
    uploaded: List[UploadItem]

class MediaLinkRequest(BaseModel):
    media_ids: Optional[List[str]] = None

class AudioLinkRequest(BaseModel):
    audio_id: str

class ProjectDetailResponse(ProjectResponse):
    media: List[MediaResponse]

class OutputVideoResponse(BaseModel):
    project_id: str
    output_video: Optional[str] = None

class ProjectListItem(BaseModel):
    id: str
    title: Optional[str]
    description: Optional[str]
    target_duration: int
    aspect_ratio: str
    style: Optional[str]
    created_at: Optional[datetime]
    render_status: str  # not_started | running | done | error

class RenderRequest(BaseModel):
    prompt: str
    output_filename: Optional[str] = "final_output.mp4"

class RenderResponse(BaseModel):
    project_id: str
    status: str
    message: str
    final_video_path: Optional[str] = None
    safe_zone_verdict: Optional[str] = None
