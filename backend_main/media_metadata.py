import json
import os
import re
import subprocess
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, Optional


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".gif", ".webp", ".heic", ".heif"}


def _to_float(value: Any) -> Optional[float]:
    if value in (None, "", "0/0", "N/A"):
        return None
    try:
        return float(Fraction(str(value)))
    except Exception:
        try:
            return float(value)
        except Exception:
            return None


def _probe_video(file_path: str) -> Optional[Dict[str, Any]]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v", "error",
                "-print_format", "json",
                "-show_format",
                "-show_streams",
                file_path,
            ],
            capture_output=True,
            text=True,
            check=True,
            stdin=subprocess.DEVNULL,
        )
        return json.loads(result.stdout)
    except (FileNotFoundError, subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def _stream_of_type(probe: Dict[str, Any], stream_type: str) -> Dict[str, Any]:
    for stream in probe.get("streams", []):
        if stream.get("codec_type") == stream_type:
            return stream
    return {}


def _normalize_tags(tags: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not isinstance(tags, dict):
        return {}
    return {str(k).lower(): v for k, v in tags.items() if v not in (None, "")}


def _camera_info_from_tags(tags: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    camera_info: Dict[str, Any] = {}
    aliases = {
        "make": ["make", "com.apple.quicktime.make"],
        "model": ["model", "com.apple.quicktime.model"],
        "software": ["software", "encoder", "com.apple.quicktime.software"],
        "creation_time": ["creation_time"],
    }

    for field, keys in aliases.items():
        for key in keys:
            if tags.get(key):
                camera_info[field] = tags[key]
                break

    return camera_info or None


def _gps_from_tags(tags: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    gps_value = None
    for key in (
        "location",
        "location-eng",
        "com.apple.quicktime.location.iso6709",
        "com.apple.quicktime.location.name",
        "gpscoordinates",
        "gps_coordinates",
    ):
        if tags.get(key):
            gps_value = tags[key]
            break

    if not gps_value:
        return None

    gps: Dict[str, Any] = {"raw": gps_value}
    if isinstance(gps_value, str):
        match = re.match(r"^([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)(?:[+-]\d+(?:\.\d+)?)?/?$", gps_value)
        if match:
            gps["latitude"] = float(match.group(1))
            gps["longitude"] = float(match.group(2))
    return gps


def _image_metadata(file_path: str) -> Dict[str, Any]:
    from PIL import ExifTags, Image as PILImage

    with PILImage.open(file_path) as img:
        width, height = img.size
        exif = img.getexif()

    tags: Dict[str, Any] = {}
    gps_tags: Dict[str, Any] = {}
    if exif:
        for key, value in exif.items():
            tag_name = ExifTags.TAGS.get(key, str(key))
            if tag_name == "GPSInfo" and isinstance(value, dict):
                gps_tags = {
                    ExifTags.GPSTAGS.get(gps_key, str(gps_key)).lower(): gps_value
                    for gps_key, gps_value in value.items()
                    if gps_value not in (None, "")
                }
            elif value not in (None, ""):
                tags[tag_name.lower()] = value

    camera_info = _camera_info_from_tags(tags)
    gps = _gps_from_tags({**tags, **gps_tags})

    return {
        "filename": os.path.basename(file_path),
        "extension": Path(file_path).suffix.lower(),
        "file_size_bytes": os.path.getsize(file_path),
        "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
        "media_type": "photo",
        "duration_seconds": None,
        "resolution": {"width": width, "height": height},
        "aspect_ratio": round(width / height, 2) if height else None,
        "fps": None,
        "has_audio": False,
        "codec": None,
        "audio_codec": None,
        "bitrate_bps": None,
        "camera_info": camera_info,
        "gps": gps,
    }


def _video_metadata(file_path: str) -> Dict[str, Any]:
    probe = _probe_video(file_path)
    if probe:
        format_data = probe.get("format", {})
        video_stream = _stream_of_type(probe, "video")
        audio_stream = _stream_of_type(probe, "audio")

        tags = _normalize_tags(format_data.get("tags"))
        tags.update(_normalize_tags(video_stream.get("tags")))

        width = video_stream.get("width")
        height = video_stream.get("height")
        duration = _to_float(format_data.get("duration") or video_stream.get("duration"))
        bitrate = format_data.get("bit_rate") or video_stream.get("bit_rate")
        fps = _to_float(video_stream.get("avg_frame_rate")) or _to_float(video_stream.get("r_frame_rate"))

        return {
            "filename": os.path.basename(file_path),
            "extension": Path(file_path).suffix.lower(),
            "file_size_bytes": os.path.getsize(file_path),
            "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
            "media_type": "video",
            "duration_seconds": round(duration, 2) if duration is not None else None,
            "resolution": {
                "width": int(width) if width is not None else None,
                "height": int(height) if height is not None else None,
            },
            "aspect_ratio": round(float(width) / float(height), 2) if width and height else None,
            "fps": round(fps, 2) if fps is not None else None,
            "has_audio": bool(audio_stream),
            "codec": video_stream.get("codec_name") or video_stream.get("codec_long_name"),
            "audio_codec": audio_stream.get("codec_name") if audio_stream else None,
            "bitrate_bps": int(bitrate) if bitrate else None,
            "camera_info": _camera_info_from_tags(tags),
            "gps": _gps_from_tags(tags),
        }

    try:
        import cv2

        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise RuntimeError("Could not open video file via cv2")
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps > 0 else 0.0
        cap.release()
        has_audio = False
    except Exception:
        return {
            "filename": os.path.basename(file_path),
            "extension": Path(file_path).suffix.lower(),
            "file_size_bytes": os.path.getsize(file_path),
            "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
            "media_type": "video",
            "duration_seconds": None,
            "resolution": {"width": None, "height": None},
            "aspect_ratio": None,
            "fps": None,
            "has_audio": False,
            "codec": None,
            "audio_codec": None,
            "bitrate_bps": None,
            "camera_info": None,
            "gps": None,
        }

    return {
        "filename": os.path.basename(file_path),
        "extension": Path(file_path).suffix.lower(),
        "file_size_bytes": os.path.getsize(file_path),
        "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
        "media_type": "video",
        "duration_seconds": round(duration, 2) if duration is not None else None,
        "resolution": {"width": width, "height": height},
        "aspect_ratio": round(width / height, 2) if height else None,
        "fps": round(fps, 2) if fps is not None else None,
        "has_audio": has_audio,
        "codec": None,
        "audio_codec": None,
        "bitrate_bps": None,
        "camera_info": None,
        "gps": None,
    }


def extract_media_metadata(file_path: str) -> Dict[str, Any]:
    """Extracts technical metadata without raising on parser failures."""
    try:
        ext = Path(file_path).suffix.lower()
        if ext in IMAGE_EXTENSIONS:
            return _image_metadata(file_path)
        return _video_metadata(file_path)
    except Exception as exc:
        return {
            "filename": os.path.basename(file_path),
            "extension": Path(file_path).suffix.lower(),
            "file_size_bytes": os.path.getsize(file_path) if os.path.exists(file_path) else None,
            "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2) if os.path.exists(file_path) else None,
            "media_type": "unknown",
            "duration_seconds": None,
            "resolution": {"width": None, "height": None},
            "aspect_ratio": None,
            "fps": None,
            "has_audio": False,
            "codec": None,
            "audio_codec": None,
            "bitrate_bps": None,
            "camera_info": None,
            "gps": None,
            "error": str(exc),
        }
