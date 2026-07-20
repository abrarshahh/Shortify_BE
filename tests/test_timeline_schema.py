import pytest
from pydantic import ValidationError
from backend_ai.schemas.edl import (
    TimelineIR, TimelineClip, TimelineAudio, TimelineText, TimelineSticker,
    VisualProperties, EDLDocument, convert_edl_to_timeline_ir, TransitionType
)

def test_visual_properties():
    vp = VisualProperties(x=0.1, y=-0.2, scale=1.5, opacity=0.8)
    assert vp.x == 0.1
    assert vp.opacity == 0.8

def test_timeline_clip_valid():
    clip = TimelineClip(
        id="c1",
        source="test.mp4",
        start_in_clip=0.0,
        end_in_clip=5.0,
        timeline_start=0.0,
        timeline_end=5.0,
        layer=1
    )
    assert clip.id == "c1"
    assert clip.speed == 1.0

def test_timeline_clip_invalid():
    with pytest.raises(ValidationError):
        # end_in_clip <= start_in_clip
        TimelineClip(
            id="c1",
            source="test.mp4",
            start_in_clip=5.0,
            end_in_clip=5.0,
            timeline_start=0.0,
            timeline_end=5.0
        )

def test_timeline_ir_validation():
    ir = TimelineIR(
        title="Test Reel",
        storyline="A simple test",
        total_duration=10.0,
        video_clips=[
            TimelineClip(
                id="c1",
                source="test.mp4",
                start_in_clip=0.0,
                end_in_clip=10.0,
                timeline_start=0.0,
                timeline_end=10.0
            )
        ]
    )
    assert ir.title == "Test Reel"
    assert len(ir.video_clips) == 1

def test_timeline_ir_invalid_empty():
    with pytest.raises(ValidationError):
        # Empty clips
        TimelineIR(
            title="Empty",
            storyline="No clips",
            total_duration=5.0
        )

def test_convert_edl_to_timeline_ir():
    # Construct a legacy EDL document dict
    legacy_edl_data = {
        "title": "Legacy Reel",
        "storyline": "Hiking trip",
        "total_duration": 10.0,
        "music_start_offset": 2.5,
        "timeline": [
            {
                "clip_name": "vlog.mp4",
                "start_in_clip": 1.0,
                "end_in_clip": 11.0,
                "timeline_start": 0.0,
                "timeline_end": 10.0,
                "transition": "crossfade",
                "text_overlay": "Awesome Hike!",
                "text_preset": "bold_hype",
                "text_animation": "slide_up",
                "sticker_path": "subscribe_icon.gif",
                "sticker_animation": "fade",
                "color_grade": {
                    "brightness": 1.1,
                    "contrast": 1.0,
                    "gamma": 1.0,
                    "saturation": 1.0,
                    "vibrance": 1.0,
                    "hue": 0.0,
                    "temperature": 10.0,
                    "vignette_strength": 0.0,
                    "vignette_radius": 0.75
                },
                "details": {
                    "visual_cue": "Hiking shot",
                    "sound_design": "whoosh",
                    "pacing_style": "jump-cut",
                    "is_hook": True,
                    "keep_original_audio": True
                }
            }
        ]
    }
    
    edl = EDLDocument.model_validate(legacy_edl_data)
    ir = convert_edl_to_timeline_ir(edl)
    
    assert ir.title == "Legacy Reel"
    assert len(ir.video_clips) == 1
    assert ir.video_clips[0].source == "vlog.mp4"
    assert ir.video_clips[0].transition_in.value == "crossfade"
    
    assert len(ir.text_overlays) == 1
    assert ir.text_overlays[0].text == "Awesome Hike!"
    assert ir.text_overlays[0].font_size == 52 # bold_hype preset mapped
    
    assert len(ir.stickers) == 1
    assert ir.stickers[0].sticker_asset_id == "sticker_subscribe"
    
    assert len(ir.audio_clips) == 1
    assert ir.audio_clips[0].source == "background_music"
    assert ir.audio_clips[0].start_in_audio == 2.5
