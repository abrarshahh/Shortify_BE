import os, uuid
from pathlib import Path
from fastapi import UploadFile, HTTPException
from backend_main.config import STORAGE_ROOT

# Extensions that browsers / curl may not label with a proper MIME type
_ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",   # video
    ".jpg", ".jpeg", ".png", ".gif", ".webp",    # image
    ".heic", ".heif",                             # Apple image formats
    ".wav", ".mp3", ".aac", ".m4a", ".ogg",      # audio
}

def save_upload_file(user_id: str, upload_file: UploadFile) -> str:
    ext = Path(upload_file.filename).suffix
    media_id = uuid.uuid4()
    # Store in user's library
    rel_dir = Path(f"users/{user_id}/media")
    full_dir = STORAGE_ROOT / rel_dir
    full_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{media_id}{ext}"
    full_path = full_dir / filename
    with full_path.open("wb") as f:
        f.write(upload_file.file.read())
    return str((rel_dir / filename))  # relative path to store in DB

def validate_file(upload_file: UploadFile) -> None:
    mime = upload_file.content_type or ""
    filename = upload_file.filename or ""
    ext = Path(filename).suffix.lower()

    # Accept if MIME type is a known media type
    is_valid_mime = (
        mime.startswith("video/")
        or mime.startswith("image/")
        or mime.startswith("audio/")
    )
    # Fall back to extension check for files curl sends as application/octet-stream
    is_valid_ext = ext in _ALLOWED_EXTENSIONS

    if not is_valid_mime and not is_valid_ext:
        raise HTTPException(400, f"Invalid file type: '{mime}' / '{ext}'. Allowed: video, image, audio.")

    # Limit max file size (200MB)
    max_size = 200 * 1024 * 1024
    file_size = len(upload_file.file.read())
    upload_file.file.seek(0)  # Reset file pointer
    if file_size > max_size:
        raise HTTPException(400, "File too large (max 200MB)")
