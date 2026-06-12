import uuid
from typing import Optional, List
from pydantic import BaseModel

class OutputVideoResponse(BaseModel):
    project_id: uuid.UUID
    output_video: Optional[str] = None

class RenderRequest(BaseModel):
    prompt: str
    output_filename: Optional[str] = "final_output.mp4"

class RenderResponse(BaseModel):
    project_id: uuid.UUID
    status: str
    message: str
    final_video_path: Optional[str] = None
    safe_zone_verdict: Optional[str] = None
    progress_percentage: Optional[int] = 0
    current_step: Optional[str] = "Initializing..."
    skipped_clips: Optional[List[str]] = []
