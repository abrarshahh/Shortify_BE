import pytest
from pydantic import ValidationError
from backend_ai.schemas.edl import EDLTimelineItem
from backend_ai.effects.text_animation import build_ffmpeg_overlay_filters
from backend_ai.effects.text_presets import TEXT_PRESETS

def test_edl_item_text_graphics_validation():
    # Base valid EDL timeline item details
    details = {
        "visual_cue": "Cue",
        "sound_design": "Sound",
        "pacing_style": "jump-cut",
        "is_hook": False,
        "keep_original_audio": True
    }

    # Valid presets and animations
    item = EDLTimelineItem(
        clip_name="clip.mp4",
        start_in_clip=0.0,
        end_in_clip=5.0,
        timeline_start=0.0,
        timeline_end=5.0,
        transition="none",
        text_preset="bold_hype",
        text_animation="slide_up",
        sticker_animation="fade",
        sticker_path="path/to/sticker.png",
        sticker_position="top-right",
        effect_path="path/to/effect.mp4",
        details=details
    )
    assert item.text_preset == "bold_hype"
    assert item.text_animation == "slide_up"
    assert item.sticker_animation == "fade"
    assert item.sticker_path == "path/to/sticker.png"
    assert item.sticker_position == "top-right"
    assert item.effect_path == "path/to/effect.mp4"

    # Reject invalid text preset
    with pytest.raises(ValidationError):
        EDLTimelineItem(
            clip_name="clip.mp4",
            start_in_clip=0.0,
            end_in_clip=5.0,
            timeline_start=0.0,
            timeline_end=5.0,
            transition="none",
            text_preset="non_existent_preset",
            details=details
        )

    # Reject invalid text animation
    with pytest.raises(ValidationError):
        EDLTimelineItem(
            clip_name="clip.mp4",
            start_in_clip=0.0,
            end_in_clip=5.0,
            timeline_start=0.0,
            timeline_end=5.0,
            transition="none",
            text_animation="fly_in",
            details=details
        )

    # Reject invalid sticker animation
    with pytest.raises(ValidationError):
        EDLTimelineItem(
            clip_name="clip.mp4",
            start_in_clip=0.0,
            end_in_clip=5.0,
            timeline_start=0.0,
            timeline_end=5.0,
            transition="none",
            sticker_animation="spin_zoom",
            details=details
        )

def test_build_ffmpeg_overlay_filters():
    # Test neutral / fallback (none or copy)
    video_filter, x_expr, y_expr = build_ffmpeg_overlay_filters(
        duration=4.0,
        anim_type="none",
        x_resting="X_0",
        y_resting="Y_0"
    )
    assert video_filter == "copy"
    assert x_expr == "X_0"
    assert y_expr == "Y_0"

    # Test fade animation
    video_filter, x_expr, y_expr = build_ffmpeg_overlay_filters(
        duration=4.0,
        anim_type="fade",
        x_resting="X_0",
        y_resting="Y_0",
        anim_duration=0.5
    )
    assert "fade=in:st=0:d=0.500" in video_filter
    assert "fade=out:st=3.500:d=0.500" in video_filter
    assert x_expr == "X_0"
    assert y_expr == "Y_0"

    # Test slide_up animation
    video_filter, x_expr, y_expr = build_ffmpeg_overlay_filters(
        duration=6.0,
        anim_type="slide_up",
        x_resting="X_0",
        y_resting="Y_0",
        anim_duration=0.5
    )
    assert video_filter == "copy"
    assert x_expr == "X_0"
    assert "if(lt(t,0.500),H-(H-(Y_0))*(t/0.500)" in y_expr
    assert "if(gt(t,5.500),(Y_0)+(H-(Y_0))*((t-5.500)/0.500)" in y_expr

    # Test slide_down animation
    video_filter, x_expr, y_expr = build_ffmpeg_overlay_filters(
        duration=6.0,
        anim_type="slide_down",
        x_resting="X_0",
        y_resting="Y_0",
        anim_duration=0.5
    )
    assert "if(lt(t,0.500),-h+((Y_0)+h)*(t/0.500)" in y_expr
    assert "if(gt(t,5.500),(Y_0)-((Y_0)+h)*((t-5.500)/0.500)" in y_expr

    # Test slide_left animation
    video_filter, x_expr, y_expr = build_ffmpeg_overlay_filters(
        duration=6.0,
        anim_type="slide_left",
        x_resting="X_0",
        y_resting="Y_0",
        anim_duration=0.5
    )
    assert y_expr == "Y_0"
    assert "if(lt(t,0.500),-w+((X_0)+w)*(t/0.500)" in x_expr
    assert "if(gt(t,5.500),(X_0)-((X_0)+w)*((t-5.500)/0.500)" in x_expr

    # Test slide_right animation
    video_filter, x_expr, y_expr = build_ffmpeg_overlay_filters(
        duration=6.0,
        anim_type="slide_right",
        x_resting="X_0",
        y_resting="Y_0",
        anim_duration=0.5
    )
    assert y_expr == "Y_0"
    assert "if(lt(t,0.500),W-(W-(X_0))*(t/0.500)" in x_expr
    assert "if(gt(t,5.500),(X_0)+(W-(X_0))*((t-5.500)/0.500)" in x_expr
