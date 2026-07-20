import pytest
from backend_ai.schemas.edl import TimelineIR, TimelineClip, TimelineAudio, TimelineText
from backend_ai.services.validation_resolver import ValidationConflictResolver

def test_speed_clamping():
    resolver = ValidationConflictResolver()
    
    timeline = TimelineIR(
        title="Test",
        storyline="Test",
        total_duration=10.0,
        video_clips=[
            TimelineClip(
                id="c1",
                source="test.mp4",
                start_in_clip=0.0,
                end_in_clip=10.0,
                timeline_start=0.0,
                timeline_end=10.0,
                speed=0.1  # below min
            ),
            TimelineClip(
                id="c2",
                source="test.mp4",
                start_in_clip=0.0,
                end_in_clip=10.0,
                timeline_start=0.0,
                timeline_end=10.0,
                speed=12.0  # above max
            )
        ]
    )
    
    resolutions = resolver.resolve_conflicts(timeline)
    
    # Assert speed was clamped
    assert timeline.video_clips[0].speed == 0.25
    assert timeline.video_clips[1].speed == 8.0
    assert len(resolutions) >= 2

def test_transition_durations_clamping():
    resolver = ValidationConflictResolver()
    
    timeline = TimelineIR(
        title="Test",
        storyline="Test",
        total_duration=10.0,
        video_clips=[
            TimelineClip(
                id="c1",
                source="test.mp4",
                start_in_clip=0.0,
                end_in_clip=10.0,  # eff_dur = 10s
                timeline_start=0.0,
                timeline_end=10.0,
                transition_in_duration=6.0,  # above 50% (5.0s)
                transition_out_duration=7.0  # above 50% (5.0s)
            )
        ]
    )
    
    resolver.resolve_conflicts(timeline)
    
    assert timeline.video_clips[0].transition_in_duration == 5.0
    assert timeline.video_clips[0].transition_out_duration == 5.0

def test_reverse_stabilization_conflict():
    resolver = ValidationConflictResolver()
    
    timeline = TimelineIR(
        title="Test",
        storyline="Test",
        total_duration=10.0,
        video_clips=[
            TimelineClip(
                id="c1",
                source="test.mp4",
                start_in_clip=0.0,
                end_in_clip=10.0,
                timeline_start=0.0,
                timeline_end=10.0,
                reverse=True,
                stabilize=True
            )
        ]
    )
    
    resolver.resolve_conflicts(timeline)
    
    assert timeline.video_clips[0].stabilize is False

def test_text_safe_zone_clamping():
    resolver = ValidationConflictResolver()
    
    timeline = TimelineIR(
        title="Test",
        storyline="Test",
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
        ],
        text_overlays=[
            TimelineText(
                id="t1",
                text="Too high",
                timeline_start=0.0,
                timeline_end=5.0,
                x=-1.5,
                y=-0.95
            ),
            TimelineText(
                id="t2",
                text="Too low",
                timeline_start=0.0,
                timeline_end=5.0,
                x=1.2,
                y=0.85
            )
        ]
    )
    
    resolver.resolve_conflicts(timeline)
    
    # Check text safe zone limits
    assert timeline.text_overlays[0].x == -0.8
    assert timeline.text_overlays[0].y == -0.7
    
    assert timeline.text_overlays[1].x == 0.8
    assert timeline.text_overlays[1].y == 0.7
