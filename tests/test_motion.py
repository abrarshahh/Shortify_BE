import pytest
import math
from pydantic import ValidationError
from backend_ai.schemas.edl import EDLTimelineItem
from backend_ai.effects.motion import get_average_speed_linear, build_ffmpeg_speed_filter, SPEED_PRESETS

def test_get_average_speed_linear():
    # Constant speed
    assert math.isclose(get_average_speed_linear([(0.0, 2.0), (1.0, 2.0)]), 2.0)
    assert math.isclose(get_average_speed_linear([(0.0, 0.5), (1.0, 0.5)]), 0.5)

    # Piecewise constant linear transition
    # From 0.0 to 0.5 at speed 1.0 (dx = 0.5, output duration = 0.5/1.0 = 0.5)
    # From 0.5 to 1.0 at speed 2.0 (dx = 0.5, output duration = 0.5/2.0 = 0.25)
    # Total output duration = 0.75. Avg speed = 1.0 / 0.75 = 1.33333333
    avg = get_average_speed_linear([(0.0, 1.0), (0.5, 1.0), (0.5, 2.0), (1.0, 2.0)])
    assert math.isclose(avg, 4.0 / 3.0)

    # Pure linear ramping from 1.0 to 2.0 over full clip
    # dx = 1.0. output_fraction = (1.0 / (2.0 - 1.0)) * ln(2.0/1.0) = ln(2.0) = 0.693147
    # Avg speed = 1.0 / ln(2.0) = 1.442695
    avg_linear = get_average_speed_linear([(0.0, 1.0), (1.0, 2.0)])
    assert math.isclose(avg_linear, 1.0 / math.log(2.0))

def test_build_ffmpeg_speed_filter():
    # Test neutral / fallback filter (no keyframes)
    v_str, a_str = build_ffmpeg_speed_filter(duration=5.0)
    assert "setpts=PTS-STARTPTS" in v_str
    assert a_str == "copy"

    # Test reverse only
    v_str, a_str = build_ffmpeg_speed_filter(duration=5.0, reverse=True)
    assert "setpts=PTS-STARTPTS" in v_str
    assert "reverse" in v_str
    assert "areverse" in a_str

    # Test constant preset (constant_fast = 2x)
    v_str, a_str = build_ffmpeg_speed_filter(duration=5.0, preset="constant_fast")
    assert "setpts" in v_str
    assert "atempo=2.0000" in a_str

    # Test custom curve with speed spike
    v_str, a_str = build_ffmpeg_speed_filter(
        duration=10.0,
        keyframes=[(0.0, 1.0), (0.5, 3.0), (1.0, 1.0)]
    )
    # check that we have log expressions or linear segments inside setpts
    assert "setpts=" in v_str
    assert "log" in v_str or "if" in v_str
    assert "atempo" in a_str

def test_edl_item_motion_validation():
    # Base valid EDL timeline item details
    details = {
        "visual_cue": "Cue",
        "sound_design": "Sound",
        "pacing_style": "jump-cut",
        "is_hook": False,
        "keep_original_audio": True
    }

    # Valid preset
    item = EDLTimelineItem(
        clip_name="clip.mp4",
        start_in_clip=0.0,
        end_in_clip=5.0,
        timeline_start=0.0,
        timeline_end=5.0,
        transition="none",
        speed_preset="ramp_up",
        details=details
    )
    assert item.speed_preset == "ramp_up"

    # Valid custom keyframes
    item2 = EDLTimelineItem(
        clip_name="clip.mp4",
        start_in_clip=0.0,
        end_in_clip=5.0,
        timeline_start=0.0,
        timeline_end=5.0,
        transition="none",
        speed_keyframes=[(0.0, 1.0), (0.5, 2.0), (1.0, 1.0)],
        details=details
    )
    assert len(item2.speed_keyframes) == 3

    # Reject both preset and keyframes
    with pytest.raises(ValidationError):
        EDLTimelineItem(
            clip_name="clip.mp4",
            start_in_clip=0.0,
            end_in_clip=5.0,
            timeline_start=0.0,
            timeline_end=5.0,
            transition="none",
            speed_preset="ramp_up",
            speed_keyframes=[(0.0, 1.0), (1.0, 1.0)],
            details=details
        )

    # Reject invalid preset name
    with pytest.raises(ValidationError):
        EDLTimelineItem(
            clip_name="clip.mp4",
            start_in_clip=0.0,
            end_in_clip=5.0,
            timeline_start=0.0,
            timeline_end=5.0,
            transition="none",
            speed_preset="invalid_preset_name",
            details=details
        )

    # Reject unsorted keyframes
    with pytest.raises(ValidationError):
        EDLTimelineItem(
            clip_name="clip.mp4",
            start_in_clip=0.0,
            end_in_clip=5.0,
            timeline_start=0.0,
            timeline_end=5.0,
            transition="none",
            speed_keyframes=[(0.0, 1.0), (0.7, 2.0), (0.5, 1.0), (1.0, 1.0)],
            details=details
        )

    # Reject negative speed multiplier
    with pytest.raises(ValidationError):
        EDLTimelineItem(
            clip_name="clip.mp4",
            start_in_clip=0.0,
            end_in_clip=5.0,
            timeline_start=0.0,
            timeline_end=5.0,
            transition="none",
            speed_keyframes=[(0.0, 1.0), (1.0, -2.0)],
            details=details
        )

    # Reject keyframes that don't start at 0.0 or end at 1.0
    with pytest.raises(ValidationError):
        EDLTimelineItem(
            clip_name="clip.mp4",
            start_in_clip=0.0,
            end_in_clip=5.0,
            timeline_start=0.0,
            timeline_end=5.0,
            transition="none",
            speed_keyframes=[(0.1, 1.0), (1.0, 1.0)],
            details=details
        )
