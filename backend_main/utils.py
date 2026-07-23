import os, uuid
import logging
import subprocess
from pathlib import Path
from fastapi import UploadFile, HTTPException
from backend_main.config import STORAGE_ROOT
from backend_main.media_metadata import extract_media_metadata
from backend_main.supabase_storage import upload_to_supabase

logger = logging.getLogger("backend_main.utils")

# Extensions that browsers / curl may not label with a proper MIME type
_ALLOWED_EXTENSIONS = {
    ".mp4", ".mov", ".avi", ".mkv", ".webm",   # video
    ".jpg", ".jpeg", ".png", ".gif", ".webp",    # image
    ".heic", ".heif",                             # Apple image formats
    ".wav", ".mp3", ".aac", ".m4a", ".ogg",      # audio
}

def compress_video_to_mp4(input_path: str, output_path: str) -> bool:
    """
    Compresses input video to a standard web-optimized H.264 MP4.
    Limits resolution to max 1080p height and crf=28 to get massive size reduction.
    """
    from backend_ai.core.config import FFMPEG_PATH
    
    # Scale to max height of 1080p preserving aspect ratio. 
    # Use standard FFmpeg filter scale=-2:min(ih,1080)
    cmd = [
        FFMPEG_PATH,
        "-y",
        "-i", input_path,
        "-vcodec", "libx264",
        "-crf", "28",
        "-preset", "veryfast",
        "-vf", "scale=-2:min(ih\\,1080)",
        "-acodec", "aac",
        "-b:a", "128k",
        output_path
    ]
    try:
        logger.info(f"[Video Compressor] Running FFmpeg command: {' '.join(cmd)}")
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode == 0:
            logger.info("[Video Compressor] Video compressed successfully.")
            return True
        else:
            logger.error(f"[Video Compressor] FFmpeg compression failed: {res.stderr}")
            return False
    except Exception as e:
        logger.error(f"[Video Compressor] Exception during video compression: {e}")
        return False

def save_upload_file(user_id: str, upload_file: UploadFile) -> str:
    ext = Path(upload_file.filename).suffix.lower()
    media_id = uuid.uuid4()
    # Store in user's library
    rel_dir = Path(f"users/{user_id}/media")
    full_dir = STORAGE_ROOT / rel_dir
    full_dir.mkdir(parents=True, exist_ok=True)
    
    is_video = ext in {".mp4", ".mov", ".avi", ".mkv", ".webm"}
    use_supabase = os.getenv("USE_SUPABASE", "false").strip().lower() == "true"
    
    if is_video and use_supabase:
        target_filename = f"{media_id}.mp4"
        raw_filename = f"raw_{media_id}{ext}"
        
        raw_path = full_dir / raw_filename
        compressed_path = full_dir / target_filename
        
        # Save raw uploaded file first
        upload_file.file.seek(0)
        with raw_path.open("wb") as f:
            f.write(upload_file.file.read())
            
        logger.info(f"[Video Compressor] Compressing uploaded video {upload_file.filename}...")
        success = compress_video_to_mp4(str(raw_path), str(compressed_path))
        
        if success and os.path.exists(compressed_path):
            # Clean up the raw file
            try:
                os.unlink(raw_path)
            except Exception as e:
                logger.warning(f"[Video Compressor] Failed to delete raw video file: {e}")
            filename = target_filename
            mime_type = "video/mp4"
            final_path = compressed_path
        else:
            # Fall back to using the raw file
            logger.warning("[Video Compressor] Compression failed, falling back to raw file.")
            filename = f"{media_id}{ext}"
            fallback_path = full_dir / filename
            os.rename(raw_path, fallback_path)
            mime_type = upload_file.content_type
            final_path = fallback_path
    else:
        # For non-video files (images, audio), save normally
        filename = f"{media_id}{ext}"
        final_path = full_dir / filename
        upload_file.file.seek(0)
        with final_path.open("wb") as f:
            f.write(upload_file.file.read())
        mime_type = upload_file.content_type

    # Upload to Supabase
    storage_path = f"users/{user_id}/media/{filename}"
    upload_to_supabase(str(final_path), storage_path, mime_type=mime_type)
    
    return storage_path

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
