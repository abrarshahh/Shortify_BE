import uuid
from typing import Optional, List
from datetime import datetime
from pydantic import BaseModel

class MediaResponse(BaseModel):
    id: uuid.UUID
    original_filename: str
    storage_path: str
    mime_type: str
    file_size: int
    duration: Optional[int] = None
    thumbnail_path: Optional[str] = None
    uploaded_at: datetime
    project_ids: List[uuid.UUID] = []

class UploadItem(BaseModel):
    id: uuid.UUID
    path: str

class UploadResponse(BaseModel):
    uploaded: List[UploadItem]

class MediaLinkRequest(BaseModel):
    media_ids: Optional[List[uuid.UUID]] = None

class AudioLinkRequest(BaseModel):
    audio_id: uuid.UUID
