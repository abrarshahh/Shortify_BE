import os
import pytest
from unittest import mock
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


def test_render_planner_compilation_with_filters():
    from backend_ai.schemas.edl import ColorGradeParams, ClipEffectParams
    clips_dir = "tests/mock_clips"
    output_dir = "tests/mock_exports"
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    dummy_video = os.path.join(clips_dir, "vlog.mp4")
    with open(dummy_video, "w") as f:
        f.write("dummy")

    timeline = TimelineIR(
        title="Test Filters",
        storyline="Filters reel",
        total_duration=5.0,
        video_clips=[
            TimelineClip(
                id="c1",
                source="vlog.mp4",
                start_in_clip=0.0,
                end_in_clip=5.0,
                timeline_start=0.0,
                timeline_end=5.0,
                speed=1.0,
                color_grade=ColorGradeParams(
                    brightness=1.2,
                    contrast=1.1,
                    gamma=1.0,
                    saturation=1.0,
                    vibrance=1.0,
                    hue=0.0,
                    temperature=10.0,
                    vignette_strength=0.3,
                    vignette_radius=0.75
                ),
                clip_effect=ClipEffectParams(
                    effect_type="blur",
                    parameters={"max_blur_size": 25}
                )
            )
        ]
    )

    planner = RenderPlanner(clips_dir=clips_dir, output_dir=output_dir)
    cmd = planner.compile_timeline_to_ffmpeg_cmd(
        timeline=timeline,
        output_filename="output_filtered.mp4",
        aspect_ratio="9:16"
    )

    assert isinstance(cmd, list)
    filter_complex = cmd[cmd.index("-filter_complex") + 1]

    # Verify that eq and vignette filters (from color grading) are present
    assert "eq=" in filter_complex
    assert "vignette=" in filter_complex

    # Verify that boxblur filter (from clip effect) is present
    assert "boxblur=" in filter_complex

    # Cleanup mock directories/files
    try:
        os.remove(dummy_video)
        os.rmdir(clips_dir)
        os.rmdir(output_dir)
    except Exception:
        pass


def test_render_planner_compilation_audio_ducking_disabled():
    clips_dir = "tests/mock_clips"
    output_dir = "tests/mock_exports"
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    dummy_video = os.path.join(clips_dir, "vlog.mp4")
    with open(dummy_video, "w") as f:
        f.write("dummy")

    timeline = TimelineIR(
        title="Test Ducking Disabled",
        storyline="Ducking disabled reel",
        total_duration=5.0,
        video_clips=[
            TimelineClip(
                id="c1",
                source="vlog.mp4",
                start_in_clip=0.0,
                end_in_clip=5.0,
                timeline_start=0.0,
                timeline_end=5.0,
                speed=1.0
            )
        ]
    )

    planner = RenderPlanner(clips_dir=clips_dir, output_dir=output_dir)
    
    with mock.patch.object(RenderPlanner, "_check_has_audio", return_value=True):
        # 1. Test audio_ducking=True (ignores video audio, only music used)
        cmd_mute = planner.compile_timeline_to_ffmpeg_cmd(
            timeline=timeline,
            output_filename="output_ducking_enabled.mp4",
            aspect_ratio="9:16",
            audio_ducking=True
        )
        assert isinstance(cmd_mute, list)
        filter_complex_mute = cmd_mute[cmd_mute.index("-filter_complex") + 1]
        assert "anullsrc=" in filter_complex_mute
        assert "[0:a]" not in filter_complex_mute

        # 2. Test audio_ducking=False (dynamic ducking active, retains original video audio)
        cmd_keep = planner.compile_timeline_to_ffmpeg_cmd(
            timeline=timeline,
            output_filename="output_ducking_disabled.mp4",
            aspect_ratio="9:16",
            audio_ducking=False
        )
        assert isinstance(cmd_keep, list)
        filter_complex_keep = cmd_keep[cmd_keep.index("-filter_complex") + 1]
        assert "volume=1.0" in filter_complex_keep
        assert "[0:a]" in filter_complex_keep

    # Cleanup mock directories/files
    try:
        os.remove(dummy_video)
        os.rmdir(clips_dir)
        os.rmdir(output_dir)
    except Exception:
        pass


def test_render_planner_compilation_with_background_music():
    clips_dir = "tests/mock_clips"
    output_dir = "tests/mock_exports"
    os.makedirs(clips_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    dummy_video = os.path.join(clips_dir, "vlog.mp4")
    with open(dummy_video, "w") as f:
        f.write("dummy")

    dummy_music = os.path.join(clips_dir, "music.mp3")
    with open(dummy_music, "w") as f:
        f.write("dummy")

    timeline = TimelineIR(
        title="Test Bg Music",
        storyline="Music integration reel",
        total_duration=5.0,
        video_clips=[
            TimelineClip(
                id="c1",
                source="vlog.mp4",
                start_in_clip=0.0,
                end_in_clip=5.0,
                timeline_start=0.0,
                timeline_end=5.0,
                speed=1.0
            )
        ],
        audio_clips=[
            TimelineAudio(
                id="bg_music_id",
                source="background_music",
                start_in_audio=0.0,
                end_in_audio=5.0,
                timeline_start=0.0,
                timeline_end=5.0,
                volume=0.22
            )
        ]
    )

    planner = RenderPlanner(clips_dir=clips_dir, output_dir=output_dir)
    cmd = planner.compile_timeline_to_ffmpeg_cmd(
        timeline=timeline,
        output_filename="output_bg_music.mp4",
        aspect_ratio="9:16",
        music_path=dummy_music
    )

    assert isinstance(cmd, list)
    # Check that dummy_music is in the input list
    assert os.path.abspath(dummy_music).replace("\\", "/") in [os.path.abspath(ip).replace("\\", "/") for ip in cmd]

    filter_complex = cmd[cmd.index("-filter_complex") + 1]
    # Check that it uses the input index of dummy_music in the filter complex
    # Input 0: vlog.mp4, Input 1: music.mp3
    assert "[1:a]" in filter_complex
    assert "adelay=" in filter_complex
    assert "volume=0.22" in filter_complex

    # Cleanup mock directories/files
    try:
        os.remove(dummy_video)
        os.remove(dummy_music)
        os.rmdir(clips_dir)
        os.rmdir(output_dir)
    except Exception:
        pass



