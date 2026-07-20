import os
import pytest
from backend_ai.schemas.edl import TimelineIR, TimelineClip, TimelineAudio, TimelineText, TimelineSticker
from backend_ai.services.render_planner import RenderPlanner

def test_render_planner_compilation():
    # Setup folders
    clips_dir = "tests/mock_clips"
    output_dir = "tests/mock_exports"
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Write small dummy input files so the resolver checks find them
    dummy_video = os.path.join(clips_dir, "vlog.mp4")
    with open(dummy_video, "w") as f:
        f.write("dummy")

    dummy_music = os.path.join(clips_dir, "music.mp3")
    with open(dummy_music, "w") as f:
        f.write("dummy")

    timeline = TimelineIR(
        title="Test Reel",
        storyline="Hype reel",
        total_duration=10.0,
        video_clips=[
            TimelineClip(
                id="c1",
                source="vlog.mp4",
                start_in_clip=0.0,
                end_in_clip=5.0,
                timeline_start=0.0,
                timeline_end=5.0,
                speed=1.0,
                effect_asset_id="overlay_light_leak"
            ),
            TimelineClip(
                id="c2",
                source="vlog.mp4",
                start_in_clip=0.0,
                end_in_clip=5.0,
                timeline_start=5.0,
                timeline_end=10.0,
                speed=2.0
            )
        ],
        audio_clips=[
            TimelineAudio(
                id="a1",
                source="music.mp3",
                start_in_audio=0.0,
                end_in_audio=10.0,
                timeline_start=0.0,
                timeline_end=10.0,
                volume=0.25
            )
        ],
        text_overlays=[
            TimelineText(
                id="t1",
                text="Hello World",
                timeline_start=1.0,
                timeline_end=4.0,
                x=0.0,
                y=0.2
            )
        ],
        stickers=[
            TimelineSticker(
                id="s1",
                sticker_asset_id="sticker_subscribe",
                timeline_start=2.0,
                timeline_end=5.0,
                x=0.5,
                y=0.5
            )
        ]
    )

    planner = RenderPlanner(clips_dir=clips_dir, output_dir=output_dir)
    cmd = planner.compile_timeline_to_ffmpeg_cmd(
        timeline=timeline,
        output_filename="output.mp4",
        aspect_ratio="9:16"
    )

    assert isinstance(cmd, list)
    assert cmd[0] == "ffmpeg"
    assert "-filter_complex" in cmd
    
    # Check that output file is mapped
    assert cmd[-1].replace("\\", "/").endswith("tests/mock_exports/output.mp4")

    # Cleanup mock directories/files
    try:
        os.remove(dummy_video)
        os.remove(dummy_music)
        os.rmdir(clips_dir)
        os.rmdir(output_dir)
    except Exception:
        pass
