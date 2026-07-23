import uuid
from typing import Optional, List
from pydantic import BaseModel

class OutputVideoResponse(BaseModel):
    project_id: uuid.UUID
    output_video: Optional[str] = None

from backend_main.schemas.enums import DurationEnum, AspectRatioEnum, StyleEnum

class RenderRequest(BaseModel):
    prompt: str
    output_filename: Optional[str] = "final_output.mp4"
    target_duration: DurationEnum
    aspect_ratio: AspectRatioEnum = AspectRatioEnum.nine_sixteen
    style: Optional[StyleEnum] = None
    caption_style: Optional[str] = "none"
    add_subtitle: bool
    add_stickers: bool
    add_textoverlay: bool
    audio_ducking: bool = True

class RenderResponse(BaseModel):
    project_id: uuid.UUID
    status: str
    message: str
    final_video_path: Optional[str] = None
    safe_zone_verdict: Optional[str] = None
    progress_percentage: Optional[int] = 0
    current_step: Optional[str] = "Initializing..."
    skipped_clips: Optional[List[str]] = []
