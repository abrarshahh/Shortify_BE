# Configuration settings for the backend AI application.

# Target Video Tolerance for Duration (in percentage)
TARGET_DURATION_TOLERANCE = 0.24

# Video editor settings
VIDEO_EDITOR_DEFAULT_FADE_DURATION = 0.3
VIDEO_EDITOR_ORIGINAL_AUDIO_VOLUME = 0.3
VIDEO_EDITOR_MUSIC_VOLUME = 0.9
VIDEO_EDITOR_PACING_SPEEDS = {
    "speed-ramp": 1.5,
    "jump-cut": 1.0,
    "cinematic-slow": 0.75,
}

# FFMPEG settings
import os
import shutil

COLOR_GRADING_ENABLED = True

# Detect local static ffmpeg binaries for production PaaS environments (like Render free tier)
_local_bin = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "bin"))
_local_ffmpeg = os.path.join(_local_bin, "ffmpeg")

if os.path.exists(_local_ffmpeg):
    FFMPEG_PATH = _local_ffmpeg
else:
    FFMPEG_PATH = shutil.which("ffmpeg") or "ffmpeg"
