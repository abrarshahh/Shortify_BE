import os
import sys

import pytest
from pydantic import ValidationError

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend_ai.orchestrator import ShortifyOrchestrator
from backend_ai.schemas.edl import EDLDocument, EDLGenerationError, EDLValidationError
from backend_ai.services.edl_validation_service import validate_edl, validate_timeline_continuity


def _base_edl():
    return {
        "title": "Test Reel",
        "storyline": "A short story",
        "total_duration": 10.0,
        "music_start_offset": 0.0,
        "timeline": [
            {
                "clip_name": "clip-a.mp4",
                "start_in_clip": 0.0,
                "end_in_clip": 5.0,
                "timeline_start": 0.0,
                "timeline_end": 5.0,
                "transition": "none",
                "text_overlay": "",
                "details": {
                    "visual_cue": "Opening shot",
                    "sound_design": "whoosh",
                    "pacing_style": "jump-cut",
                },
            },
            {
                "clip_name": "clip-b.mp4",
                "start_in_clip": 0.0,
                "end_in_clip": 5.0,
                "timeline_start": 5.0,
                "timeline_end": 10.0,
                "transition": "jump_cut",
                "text_overlay": "",
                "details": {
                    "visual_cue": "Closing shot",
                    "sound_design": "hit",
                    "pacing_style": "jump-cut",
                },
            },
        ],
    }


def test_edl_schema_rejects_bad_duration():
    edl = _base_edl()
    edl["timeline"][0]["timeline_end"] = 0.0

    with pytest.raises(ValidationError):
        EDLDocument.model_validate(edl)


def test_validate_timeline_continuity_detects_overlap():
    edl = EDLDocument.model_validate(_base_edl())
    edl.timeline[1].timeline_start = 4.5

    issues = validate_timeline_continuity(edl)

    assert any(issue["type"] == "timeline_overlap" for issue in issues)


def test_validate_edl_checks_virtual_segments(tmp_path, monkeypatch):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "vlog.mp4").write_bytes(b"fake")
    (clips_dir / "clip-b.mp4").write_bytes(b"fake")

    class FakeVideoCapture:
        def __init__(self, path):
            pass
        def isOpened(self):
            return True
        def get(self, prop):
            import cv2
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 600.0
            return 0.0
        def release(self):
            pass

    monkeypatch.setattr("cv2.VideoCapture", FakeVideoCapture)

    edl = _base_edl()
    edl["timeline"][0]["clip_name"] = "vlog.mp4:5.0:15.0"

    validated = validate_edl(edl, str(clips_dir))
    assert validated.timeline[0].clip_name == "vlog.mp4:5.0:15.0"


def test_validate_edl_rejects_out_of_bounds_virtual_segment(tmp_path, monkeypatch):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "vlog.mp4").write_bytes(b"fake")
    (clips_dir / "clip-b.mp4").write_bytes(b"fake")

    class FakeVideoCapture:
        def __init__(self, path):
            pass
        def isOpened(self):
            return True
        def get(self, prop):
            import cv2
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 600.0
            return 0.0
        def release(self):
            pass

    monkeypatch.setattr("cv2.VideoCapture", FakeVideoCapture)

    edl = _base_edl()
    edl["timeline"][0]["clip_name"] = "vlog.mp4:5.0:25.0"

    with pytest.raises(EDLValidationError) as exc_info:
        validate_edl(edl, str(clips_dir))

    assert any(issue["type"] == "virtual_segment_out_of_bounds" for issue in exc_info.value.issues)


def test_validate_edl_rejects_target_duration_mismatch(tmp_path, monkeypatch):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "clip-a.mp4").write_bytes(b"fake")
    (clips_dir / "clip-b.mp4").write_bytes(b"fake")

    class FakeVideoCapture:
        def __init__(self, path):
            pass
        def isOpened(self):
            return True
        def get(self, prop):
            import cv2
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 600.0
            return 0.0
        def release(self):
            pass

    monkeypatch.setattr("cv2.VideoCapture", FakeVideoCapture)

    edl = _base_edl()
    with pytest.raises(EDLValidationError) as exc_info:
        validate_edl(edl, str(clips_dir), target_duration=30.0)

    assert any(issue["type"] == "target_duration_mismatch" for issue in exc_info.value.issues)
    feedback = exc_info.value.to_feedback()
    assert "You only made" in feedback
    assert "make it 30.0s" in feedback


def test_validate_edl_rejects_render_duration_mismatch(tmp_path, monkeypatch):
    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "clip-a.mp4").write_bytes(b"fake")
    (clips_dir / "clip-b.mp4").write_bytes(b"fake")

    class FakeVideoCapture:
        def __init__(self, path):
            pass
        def isOpened(self):
            return True
        def get(self, prop):
            import cv2
            if prop == cv2.CAP_PROP_FPS:
                return 30.0
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return 600.0
            return 0.0
        def release(self):
            pass

    monkeypatch.setattr("cv2.VideoCapture", FakeVideoCapture)

    edl = _base_edl()
    edl["timeline"][0]["end_in_clip"] = 10.0
    edl["timeline"][0]["timeline_end"] = 10.0
    edl["timeline"][1]["start_in_clip"] = 0.0
    edl["timeline"][1]["end_in_clip"] = 10.0
    edl["timeline"][1]["timeline_start"] = 10.0
    edl["timeline"][1]["timeline_end"] = 20.0
    edl["total_duration"] = 20.0
    for item in edl["timeline"]:
        item["details"]["pacing_style"] = "speed-ramp"

    with pytest.raises(EDLValidationError) as exc_info:
        validate_edl(edl, str(clips_dir), target_duration=30.0)

    assert any(issue["type"] == "render_duration_mismatch" for issue in exc_info.value.issues)
    assert "You only made" in exc_info.value.to_feedback()


def test_generate_edl_caps_retries(monkeypatch, tmp_path):
    orchestrator = ShortifyOrchestrator.__new__(ShortifyOrchestrator)

    class FakeDirector:
        def __init__(self):
            self.calls = []

        def generate_edl(self, **kwargs):
            self.calls.append(kwargs.get("feedback"))
            return {"invalid": True}

    fake_director = FakeDirector()
    orchestrator.director_agent = fake_director

    clips_dir = tmp_path / "clips"
    clips_dir.mkdir()
    (clips_dir / "clip-a.mp4").write_bytes(b"fake")

    state = {
        "video_paths": [str(clips_dir / "clip-a.mp4")],
        "project_title": "Test Prompt",
        "visual_data": [],
        "rhythm_data": {},
        "target_duration": 10,
        "aspect_ratio": "9:16",
        "style": "cinematic",
        "edl_feedback": "",
        "max_edl_retries": 0,
    }

    def always_fail(_edl, _clips_dir, **kwargs):
        raise EDLValidationError([
            {"field": "timeline", "message": "bad timeline", "type": "validation_error"}
        ])

    monkeypatch.setattr("backend_ai.orchestrator.validate_edl", always_fail)

    with pytest.raises(EDLGenerationError) as exc_info:
        ShortifyOrchestrator.node_generate_edl(orchestrator, state)

    assert exc_info.value.retry_count == 3
    assert len(fake_director.calls) == 3
