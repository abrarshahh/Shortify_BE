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
COLOR_GRADING_ENABLED = True
FFMPEG_PATH = "ffmpeg"
