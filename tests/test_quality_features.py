import os
import sys
import pytest
import subprocess
from typing import Dict, Any, List, Optional
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend_ai.agents.project_analyst_agent import ProjectAnalystAgent, parse_virtual_segment
from backend_ai.services.edl_validation_service import _parse_virtual_clip_name
from backend_ai.services.editor_service import VideoEditor


def test_parse_virtual_segment():
    # Simple file path
    assert parse_virtual_segment("clip.mp4") is None
    # Absolute path without virtual notation
    assert parse_virtual_segment("C:\\path\\to\\clip.mp4") is None
    # Virtual segment with simple path
    assert parse_virtual_segment("clip.mp4:10:20") == ("clip.mp4", 10.0, 20.0)
    # Virtual segment with Windows absolute path (colons)
    assert parse_virtual_segment("C:\\path\\to\\clip.mp4:15.5:30") == ("C:\\path\\to\\clip.mp4", 15.5, 30.0)
    # Invalid floats
    assert parse_virtual_segment("clip.mp4:start:end") is None


def test_pre_flight_virtual_segments(tmp_path, monkeypatch):
    # Setup mock file structure
    clips_dir = tmp_path / "testing_clips"
    clips_dir.mkdir()
    
    # Create fake files
    video_file = clips_dir / "test_video.mp4"
    video_file.write_bytes(b"fake video data")
    
    # Mock cv2.VideoCapture in analyst_service
    class MockVideoCapture:
        def __init__(self, path):
            self.opened = True
        def isOpened(self):
            return self.opened
        def get(self, prop):
            if prop == 5:  # cv2.CAP_PROP_FPS
                return 30.0
            if prop == 7:  # cv2.CAP_PROP_FRAME_COUNT
                return 1350  # 45 seconds at 30 fps
            return 0
        def release(self):
            self.opened = False

    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", MockVideoCapture)
    
    # Mock Laplacian variance and mean for scoring
    monkeypatch.setattr(ProjectAnalystAgent, "_score_video", lambda self, p, s=None, e=None: (0.8, 120.0, 128.0))

    agent = ProjectAnalystAgent()
    
    # Test valid virtual segment
    valid_virtual_path = f"{str(video_file)}:10.0:30.0"
    report = agent.analyze_inputs([valid_virtual_path])
    assert report["valid_files"] == 1
    assert len(report["rejected_files"]) == 0
    assert report["media"][0]["duration"] == 20.0  # 30.0 - 10.0
    assert report["media"][0]["path"] == valid_virtual_path

    # Test invalid virtual segment boundaries (start >= end)
    invalid_bounds_path = f"{str(video_file)}:25.0:10.0"
    report = agent.analyze_inputs([invalid_bounds_path])
    assert report["valid_files"] == 0
    assert report["rejected_files"][0]["reason"] == "invalid_virtual_segment"

    # Test virtual segment out of bounds (end exceeds video duration)
    out_of_bounds_path = f"{str(video_file)}:10.0:60.0"
    report = agent.analyze_inputs([out_of_bounds_path])
    assert report["valid_files"] == 0
    assert report["rejected_files"][0]["reason"] == "virtual_segment_out_of_bounds"


def test_edl_validation_windows_paths():
    # Verify that splitting parses Windows drive letters correctly
    parsed = _parse_virtual_clip_name("C:\\path\\to\\vlog.mp4:5.0:15.0")
    assert parsed is not None
    assert parsed[0] == "C:\\path\\to\\vlog.mp4"
    assert parsed[1] == 5.0
    assert parsed[2] == 15.0


def test_editor_graceful_fallback(tmp_path, monkeypatch):
    clips_dir = tmp_path / "testing_clips"
    clips_dir.mkdir()
    
    # Create one valid clip and one that doesn't exist
    (clips_dir / "good_clip.mp4").write_bytes(b"good")

    # Mock cv2.VideoCapture in editor_service
    class MockVideoCapture:
        def __init__(self, path):
            self.opened = True
        def isOpened(self):
            return self.opened
        def get(self, prop):
            if prop == 3:  # cv2.CAP_PROP_FRAME_WIDTH
                return 1920
            if prop == 4:  # cv2.CAP_PROP_FRAME_HEIGHT
                return 1080
            if prop == 5:  # cv2.CAP_PROP_FPS
                return 30.0
            if prop == 7:  # cv2.CAP_PROP_FRAME_COUNT
                return 300  # 10s
            return 0
        def release(self):
            self.opened = False

    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", MockVideoCapture)
    
    from backend_ai.schemas.edl import EDLDocument
    monkeypatch.setattr("backend_ai.services.editor_service.validate_edl", lambda edl, clips_dir: EDLDocument.model_validate(edl))

    # Mock single clip processing, concat assembly and duration enforcement
    monkeypatch.setattr(VideoEditor, "_process_single_clip", lambda *args, **kwargs: 5.0)
    monkeypatch.setattr(VideoEditor, "_build_ffmpeg_concat", lambda *args, **kwargs: None)
    monkeypatch.setattr(VideoEditor, "_enforce_final_duration", lambda *args, **kwargs: None)

    editor = VideoEditor(clips_dir=str(clips_dir), output_dir=str(tmp_path / "exports"))
    
    edl = {
        "title": "Test Fallback",
        "storyline": "A test storyline",
        "total_duration": 10.0,
        "music_start_offset": 0.0,
        "timeline": [
            {
                "clip_name": "good_clip.mp4",
                "start_in_clip": 0.0,
                "end_in_clip": 5.0,
                "timeline_start": 0.0,
                "timeline_end": 5.0,
                "transition": "none",
                "text_overlay": "",
                "details": {
                    "visual_cue": "Cue 1",
                    "sound_design": "whoosh",
                    "pacing_style": "jump-cut"
                }
            },
            {
                "clip_name": "non_existent.mp4",  # Missing clip that will be skipped
                "start_in_clip": 0.0,
                "end_in_clip": 5.0,
                "timeline_start": 5.0,
                "timeline_end": 10.0,
                "transition": "none",
                "text_overlay": "",
                "details": {
                    "visual_cue": "Cue 2",
                    "sound_design": "whoosh",
                    "pacing_style": "jump-cut"
                }
            }
        ]
    }

    output = editor.render(edl)
    
    # Check that skipped clip was tracked
    assert "non_existent.mp4" in editor.skipped_clips
    assert len(editor.skipped_clips) == 1
    # Check that output path was returned and rendering succeeded
    assert output is not None


def test_editor_exact_target_duration(tmp_path, monkeypatch):
    clips_dir = tmp_path / "testing_clips"
    clips_dir.mkdir()
    (clips_dir / "good_clip.mp4").write_bytes(b"good")

    edl = {
        "title": "Test Padding",
        "storyline": "A test storyline",
        "total_duration": 5.0,
        "music_start_offset": 0.0,
        "timeline": [
            {
                "clip_name": "good_clip.mp4",
                "start_in_clip": 0.0,
                "end_in_clip": 5.0,
                "timeline_start": 0.0,
                "timeline_end": 5.0,
                "transition": "none",
                "text_overlay": "",
                "details": {
                    "visual_cue": "Opening",
                    "sound_design": "beat",
                    "pacing_style": "jump-cut"
                }
            }
        ]
    }

    # Mock cv2.VideoCapture in editor_service
    class MockVideoCapture:
        def __init__(self, path):
            self.opened = True
        def isOpened(self):
            return self.opened
        def get(self, prop):
            if prop == 3:  # cv2.CAP_PROP_FRAME_WIDTH
                return 1080
            if prop == 4:  # cv2.CAP_PROP_FRAME_HEIGHT
                return 1920
            if prop == 5:  # cv2.CAP_PROP_FPS
                return 30.0
            if prop == 7:  # cv2.CAP_PROP_FRAME_COUNT
                return 90  # 3.0s
            return 0
        def release(self):
            self.opened = False

    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", MockVideoCapture)
    
    from backend_ai.schemas.edl import EDLDocument
    monkeypatch.setattr("backend_ai.services.editor_service.validate_edl", lambda edl, clips_dir: EDLDocument.model_validate(edl))

    monkeypatch.setattr(VideoEditor, "_process_single_clip", lambda *args, **kwargs: 3.0)
    monkeypatch.setattr(VideoEditor, "_build_ffmpeg_concat", lambda *args, **kwargs: None)
    
    enforced_dur = 0.0
    def mock_enforce(self, file_path, target_duration):
        nonlocal enforced_dur
        enforced_dur = target_duration
    monkeypatch.setattr(VideoEditor, "_enforce_final_duration", mock_enforce)

    editor = VideoEditor(clips_dir=str(clips_dir), output_dir=str(tmp_path / "exports"))
    editor.render(edl)

    # Duration should be exactly 5.0 seconds due to padding enforcement call!
    assert enforced_dur == 5.0


def test_remove_silence(tmp_path, monkeypatch):
    editor = VideoEditor(clips_dir=str(tmp_path), output_dir=str(tmp_path))
    
    def mock_load(file_path, sr=None):
        # 2 seconds sound, 2 seconds silence, 2 seconds sound
        sr_rate = 22050
        part1 = np.ones(2 * sr_rate) * 0.5
        part2 = np.zeros(2 * sr_rate)
        part3 = np.ones(2 * sr_rate) * 0.5
        return np.concatenate([part1, part2, part3]), sr_rate

    def mock_split(y, top_db=None):
        sr_rate = 22050
        return np.array([[0, 2 * sr_rate], [4 * sr_rate, 6 * sr_rate]])

    import librosa
    monkeypatch.setattr(librosa, "load", mock_load)
    monkeypatch.setattr(librosa.effects, "split", mock_split)
    
    intervals, duration = editor._detect_non_silent_intervals("fake_path")
    # Silence is stripped out into two intervals
    assert duration == 6.0
    assert len(intervals) == 2
    assert intervals[0] == (0.0, 2.0)
    assert intervals[1] == (4.0, 6.0)


def test_normalize_audio(tmp_path, monkeypatch):
    def mock_load(file_path, sr=None):
        # average RMS should be 0.05
        return np.ones(100) * 0.05, 22050

    import librosa
    monkeypatch.setattr(librosa, "load", mock_load)

    editor = VideoEditor(clips_dir=str(tmp_path), output_dir=str(tmp_path))
    gain = editor._get_audio_normalization_gain("fake_path", target_rms=0.15)
    
    # Gain should be 3.0x to hit target RMS of 0.15
    assert gain == pytest.approx(3.0, abs=0.1)


def test_music_looping(tmp_path, monkeypatch):
    clips_dir = tmp_path / "testing_clips"
    clips_dir.mkdir()
    music_file = clips_dir / "short_music.wav"
    music_file.write_bytes(b"fake music")
    (clips_dir / "good_clip.mp4").write_bytes(b"good")
    
    edl = {
        "title": "Test Music Loop",
        "storyline": "A test storyline",
        "total_duration": 10.0,
        "music_start_offset": 0.0,
        "timeline": [
            {
                "clip_name": "good_clip.mp4",
                "start_in_clip": 0.0,
                "end_in_clip": 10.0,
                "timeline_start": 0.0,
                "timeline_end": 10.0,
                "transition": "none",
                "text_overlay": "",
                "details": {
                    "visual_cue": "Opening",
                    "sound_design": "beat",
                    "pacing_style": "jump-cut"
                }
            }
        ]
    }

    # Mock cv2.VideoCapture in editor_service
    class MockVideoCapture:
        def __init__(self, path):
            self.opened = True
        def isOpened(self):
            return self.opened
        def get(self, prop):
            if prop == 3:  # cv2.CAP_PROP_FRAME_WIDTH
                return 1080
            if prop == 4:  # cv2.CAP_PROP_FRAME_HEIGHT
                return 1920
            if prop == 5:  # cv2.CAP_PROP_FPS
                return 30.0
            if prop == 7:  # cv2.CAP_PROP_FRAME_COUNT
                return 300  # 10s
            return 0
        def release(self):
            self.opened = False

    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", MockVideoCapture)
    
    from backend_ai.schemas.edl import EDLDocument
    monkeypatch.setattr("backend_ai.services.editor_service.validate_edl", lambda edl, clips_dir: EDLDocument.model_validate(edl))
    
    monkeypatch.setattr(VideoEditor, "_process_single_clip", lambda *args, **kwargs: 10.0)
    monkeypatch.setattr(VideoEditor, "_enforce_final_duration", lambda *args, **kwargs: None)

    captured_music = None
    def mock_build_ffmpeg_concat(self, *args, **kwargs):
        nonlocal captured_music
        captured_music = kwargs.get("music_path")
        if not captured_music:
            for arg in args:
                if isinstance(arg, str) and (arg.endswith(".wav") or arg.endswith(".mp3")):
                    captured_music = arg
                    break

    monkeypatch.setattr(VideoEditor, "_build_ffmpeg_concat", mock_build_ffmpeg_concat)

    editor = VideoEditor(clips_dir=str(clips_dir), output_dir=str(tmp_path / "exports"))
    editor.render(edl, music_path=str(music_file))
    
    # Music loops successfully to cover full video length (10s)
    assert captured_music == str(music_file)


def test_configurable_beat_snap_tolerance(tmp_path, monkeypatch):
    clips_dir = tmp_path / "testing_clips"
    clips_dir.mkdir()
    (clips_dir / "good_clip.mp4").write_bytes(b"good")

    # Mock cv2.VideoCapture in editor_service
    class MockVideoCapture:
        def __init__(self, path):
            self.opened = True
        def isOpened(self):
            return self.opened
        def get(self, prop):
            if prop == 3:  # cv2.CAP_PROP_FRAME_WIDTH
                return 1080
            if prop == 4:  # cv2.CAP_PROP_FRAME_HEIGHT
                return 1920
            if prop == 5:  # cv2.CAP_PROP_FPS
                return 30.0
            if prop == 7:  # cv2.CAP_PROP_FRAME_COUNT
                return 300  # 10s
            return 0
        def release(self):
            self.opened = False

    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", MockVideoCapture)
    
    from backend_ai.schemas.edl import EDLDocument
    monkeypatch.setattr("backend_ai.services.editor_service.validate_edl", lambda edl, clips_dir: EDLDocument.model_validate(edl))
    
    # Overwrite PACING_SPEED in test to avoid speed scaling dur adjustments
    monkeypatch.setattr(VideoEditor, "PACING_SPEED", {"speed-ramp": 1.0, "cinematic-slow": 1.0})

    # Test "speed-ramp" style which has tolerance 0.1 in config
    editor = VideoEditor(clips_dir=str(clips_dir), output_dir=str(tmp_path / "exports"))
    editor.default_beat_snap_tolerance = 0.2
    editor.beat_snap_tolerances = {"speed-ramp": 0.1, "cinematic-slow": 0.3}

    edl_speed_ramp = {
        "title": "Speed Ramp Test",
        "storyline": "Test tolerance limits",
        "total_duration": 4.8,
        "music_start_offset": 0.0,
        "timeline": [
            {
                "clip_name": "good_clip.mp4",
                "start_in_clip": 0.0,
                "end_in_clip": 6.0,
                "timeline_start": 0.0,
                "timeline_end": 4.8,
                "transition": "none",
                "text_overlay": "",
                "details": {
                    "visual_cue": "Opening",
                    "sound_design": "beat",
                    "pacing_style": "speed-ramp"
                }
            }
        ]
    }

    # Beat is at 5.0. Diff is 0.2.
    # Speed-ramp tolerance is 0.1. Diff 0.2 > 0.1 => Should NOT snap!
    rhythm_data = {"beat_times": [5.0]}
    
    captured_durations = []
    def mock_process(self, **kwargs):
        captured_durations.append(kwargs.get("target_duration"))
        return kwargs.get("target_duration")
    monkeypatch.setattr(VideoEditor, "_process_single_clip", mock_process)
    monkeypatch.setattr(VideoEditor, "_build_ffmpeg_concat", lambda *args, **kwargs: None)
    monkeypatch.setattr(VideoEditor, "_enforce_final_duration", lambda *args, **kwargs: None)

    editor.render(edl_speed_ramp, rhythm_data=rhythm_data)
    assert len(editor.skipped_clips) == 0
    assert len(captured_durations) == 1
    assert captured_durations[0] == 4.8
    
    # Now let's test "cinematic-slow" style which has tolerance 0.3 in config
    edl_cinematic = {
        "title": "Cinematic Test",
        "storyline": "Test tolerance limits",
        "total_duration": 5.0,
        "music_start_offset": 0.0,
        "timeline": [
            {
                "clip_name": "good_clip.mp4",
                "start_in_clip": 0.0,
                "end_in_clip": 6.0,
                "timeline_start": 0.0,
                "timeline_end": 4.8,
                "transition": "none",
                "text_overlay": "",
                "details": {
                    "visual_cue": "Opening",
                    "sound_design": "beat",
                    "pacing_style": "cinematic-slow"
                }
            }
        ]
    }
    
    captured_durations.clear()
    editor.render(edl_cinematic, rhythm_data=rhythm_data)
    assert len(captured_durations) == 1
    assert captured_durations[0] == 5.0


def test_creative_director_drops_context(monkeypatch):
    import json
    from backend_ai.agents.director_agent import CreativeDirector
    
    # Mock GEMINI_API_KEY env var
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    
    captured_messages = []
    
    class MockResponse:
        def __init__(self, text):
            self.text = text

    class MockModels:
        def generate_content(self, model, contents, config=None):
            nonlocal captured_messages
            captured_messages = [
                {"role": "system", "content": config.system_instruction},
                {"role": "user", "content": contents}
            ]
            return MockResponse(json.dumps({
                "title": "Mock Video",
                "storyline": "Storyline",
                "total_duration": 15.0,
                "music_start_offset": 0.0,
                "timeline": []
            }))

    class MockClient:
        models = MockModels()

    import backend_ai.agents.director_agent as da
    monkeypatch.setattr(da, "get_gemini_client", lambda: MockClient())

    director = CreativeDirector()
    
    audio_analysis = {
        "tempo": 120.0,
        "beat_times": [0.5, 1.0, 1.5],
        "peak_times": [2.0, 4.0, 6.0],  # Peak drops
        "energy_segments": [],
        "sentiment": {"label": "energetic"}
    }
    
    director.generate_edl(
        user_prompt="Make a cool reel",
        audio_analysis=audio_analysis,
        media_analyses=[],
        target_duration=15,
        style="fast_cut"
    )
    
    assert len(captured_messages) > 0
    # Context should contain "drops" key with the peak times
    user_msg_content = captured_messages[1]["content"]
    assert "Context Data: " in user_msg_content
    context_str = user_msg_content.split("Context Data: ")[1]
    context_data = json.loads(context_str)
    
    assert context_data["audio_rhythm"]["drops"] == [2.0, 4.0, 6.0]


def test_media_analyst_cache_ttl(tmp_path, monkeypatch):
    import time
    import json
    from backend_ai.agents.media_agent import MediaAnalyst
    
    # Mock GEMINI_API_KEY
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    
    # Create temp source file
    src_file = tmp_path / "test_video.mp4"
    src_file.write_bytes(b"some raw data")
    
    # Initialize analyst and override cache_dir
    analyst = MediaAnalyst()
    analyst.cache_dir = str(tmp_path / "cache")
    os.makedirs(analyst.cache_dir, exist_ok=True)
    
    # Compute cache path
    cache_path = analyst._get_cache_path(str(src_file))
    
    # Write a fake cached analysis
    fake_result = {
        "file_metadata": {"media_type": "video", "duration_seconds": 10.0},
        "summary": "This is a cached summary",
        "mood": "Calm"
    }
    with open(cache_path, "w") as f:
        json.dump(fake_result, f)
        
    # 1. First test: cache is fresh (less than 7 days old)
    called_upload = False
    class MockFiles:
        def upload(self, file):
            nonlocal called_upload
            called_upload = True
            class MockState:
                name = "SUCCESS"
            class MockFile:
                state = MockState()
                name = "files/mock-name"
            return MockFile()
            
        def delete(self, name):
            pass
            
    analyst.client = type("Client", (), {
        "files": MockFiles(),
        "models": type("Models", (), {"generate_content": lambda *a, **k: type("Response", (), {"text": "{}"})()})()
    })()
    
    res = analyst.analyze_video(str(src_file))
    assert res["summary"] == "This is a cached summary"
    assert not called_upload  # Cache hit, no upload called!

    # 2. Second test: cache is stale (8 days old) but stored forever
    eight_days_ago = time.time() - 8 * 24 * 3600
    os.utime(cache_path, (eight_days_ago, eight_days_ago))
    
    res_stale = analyst.analyze_video(str(src_file))
    assert res_stale["summary"] == "This is a cached summary"
    assert not called_upload  # Expired cache TTL is disabled, so still no upload called!


def test_media_analyst_upload_retry(tmp_path, monkeypatch):
    import time
    from backend_ai.agents.media_agent import MediaAnalyst
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    
    src_file = tmp_path / "test_video.mp4"
    src_file.write_bytes(b"data")
    
    analyst = MediaAnalyst()
    analyst.cache_dir = str(tmp_path / "cache")
    
    # Mock sleep to prevent waiting and capture sleep durations
    sleep_calls = []
    monkeypatch.setattr("time.sleep", lambda seconds: sleep_calls.append(seconds))
    
    # Mock upload: fail twice, succeed on third attempt
    upload_attempts = 0
    class MockFiles:
        def upload(self, file):
            nonlocal upload_attempts
            upload_attempts += 1
            if upload_attempts < 3:
                raise Exception("ResourceExhausted: 429 Quota exceeded")
            class MockState:
                name = "SUCCESS"
            class MockFile:
                state = MockState()
                name = "files/mock-success"
            return MockFile()
            
        def delete(self, name):
            pass
            
    analyst.client = type("Client", (), {
        "files": MockFiles(),
        "models": type("Models", (), {"generate_content": lambda *a, **k: type("Response", (), {"text": "{}"})()})()
    })()
    
    analyst.analyze_video(str(src_file))
    
    assert upload_attempts == 3
    assert len(sleep_calls) >= 2  # slept at least twice during upload retries


def test_media_analyst_finally_cleanup(tmp_path, monkeypatch):
    import pytest
    from backend_ai.agents.media_agent import MediaAnalyst
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    
    src_file = tmp_path / "test_video.mp4"
    src_file.write_bytes(b"data")
    
    analyst = MediaAnalyst()
    analyst.cache_dir = str(tmp_path / "cache")
    
    deleted_filename = None
    class MockFiles:
        def upload(self, file):
            class MockState:
                name = "SUCCESS"
            class MockFile:
                state = MockState()
                name = "files/test-cleanup-file"
            return MockFile()
            
        def delete(self, name):
            nonlocal deleted_filename
            deleted_filename = name
            
    # Mock generate_content to raise an error so that the analysis itself fails
    def mock_generate(*args, **kwargs):
        raise RuntimeError("Generation failed")
        
    analyst.client = type("Client", (), {
        "files": MockFiles(),
        "models": type("Models", (), {"generate_content": mock_generate})()
    })()
    
    with pytest.raises(RuntimeError, match="Generation failed"):
        analyst.analyze_video(str(src_file))
        
    # The file should be deleted in the finally block despite the exception
    assert deleted_filename == "files/test-cleanup-file"


def test_clip_scoring_agent(tmp_path, monkeypatch):
    from backend_ai.agents.clip_scoring_agent import ClipScoringAgent
    
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"data")
    
    class MockVideoCapture:
        def __init__(self, path):
            self.opened = True
        def isOpened(self):
            return self.opened
        def get(self, prop):
            if prop == 5:  # cv2.CAP_PROP_FPS
                return 30.0
            if prop == 7:  # cv2.CAP_PROP_FRAME_COUNT
                return 300  # 10 seconds at 30 fps
            return 0
        def set(self, prop, val):
            pass
        def read(self):
            return True, np.zeros((100, 100, 3), dtype=np.uint8)
        def release(self):
            self.opened = False
            
    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", MockVideoCapture)
    
    # Mock MediaPipe face detection to avoid dependencies in test run
    monkeypatch.setattr(ClipScoringAgent, "_detect_faces_mediapipe", lambda self, frame: (False, 0.5))
    
    agent = ClipScoringAgent(cache_dir=str(tmp_path / "cache"))
    metrics = agent.score_file(str(video_file), style="cinematic")
    
    assert "sharpness" in metrics
    assert "motion_score" in metrics
    assert metrics["motion_tier"] == "static"
    assert metrics["face_detected"] is False
    assert "composite_score" in metrics


def test_smart_face_cropping(tmp_path, monkeypatch):
    clips_dir = tmp_path / "testing_clips"
    clips_dir.mkdir()
    (clips_dir / "good_clip.mp4").write_bytes(b"good")

    # Mock cv2.VideoCapture in editor_service
    class MockVideoCapture:
        def __init__(self, path):
            self.opened = True
        def isOpened(self):
            return self.opened
        def get(self, prop):
            if prop == 3:  # cv2.CAP_PROP_FRAME_WIDTH
                return 1600
            if prop == 4:  # cv2.CAP_PROP_FRAME_HEIGHT
                return 900
            if prop == 5:  # cv2.CAP_PROP_FPS
                return 30.0
            if prop == 7:  # cv2.CAP_PROP_FRAME_COUNT
                return 300
            return 0
        def set(self, prop, val):
            pass
        def read(self):
            return True, np.zeros((900, 1600, 3), dtype=np.uint8)
        def release(self):
            self.opened = False

    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", MockVideoCapture)
    
    from backend_ai.schemas.edl import EDLDocument
    monkeypatch.setattr("backend_ai.services.editor_service.validate_edl", lambda edl, clips_dir: EDLDocument.model_validate(edl))
    
    monkeypatch.setattr(VideoEditor, "_build_ffmpeg_concat", lambda *args, **kwargs: None)
    monkeypatch.setattr(VideoEditor, "_enforce_final_duration", lambda *args, **kwargs: None)

    # Mock detect_face_center to return a face located on the far right (x=0.8, y=0.5)
    monkeypatch.setattr("backend_ai.services.editor_service.detect_face_center", lambda frame: (0.8, 0.5))

    # Mock subprocess.run to intercept the FFmpeg command
    captured_cmd = []
    def mock_run(cmd, **kwargs):
        nonlocal captured_cmd
        if len(cmd) > 0 and "ffmpeg" in cmd[0]:
            captured_cmd = cmd
        return subprocess.CompletedProcess(cmd, 0, "", "")
    monkeypatch.setattr(subprocess, "run", mock_run)

    editor = VideoEditor(clips_dir=str(clips_dir), output_dir=str(tmp_path / "exports"))
    editor.target_w = 1080
    editor.target_h = 1920

    edl = {
        "title": "Smart Crop Test",
        "storyline": "Test crop shifting",
        "total_duration": 5.0,
        "music_start_offset": 0.0,
        "timeline": [
            {
                "clip_name": "good_clip.mp4",
                "start_in_clip": 0.0,
                "end_in_clip": 5.0,
                "timeline_start": 0.0,
                "timeline_end": 5.0,
                "transition": "none",
                "text_overlay": "",
                "details": {
                    "visual_cue": "Face segment",
                    "sound_design": "whoosh",
                    "pacing_style": "jump-cut"
                }
            }
        ]
    }

    editor.render(edl)
    
    assert len(captured_cmd) > 0
    # Search the filter complex for the crop parameter
    filter_complex = ""
    for idx, arg in enumerate(captured_cmd):
        if arg == "-filter_complex":
            filter_complex = captured_cmd[idx + 1]
            break
            
    assert "crop=" in filter_complex
    # Format of crop: crop=out_w:out_h:x:y. Center crop for 1600x900 scaled to height of 1920
    # (scaled width = 1600 * 1920 / 900 = 3413). Default center x is (3413-1080)//2 = 1166.
    # Face center is at x=0.8, crop center = 0.8 * 3413 = 2730. Crop x is 2730 - 540 = 2190.
    # The assert checks that crop shifting occurred (x > 1500)
    crop_part = [p for p in filter_complex.split(";") if "crop=" in p][0]
    crop_coords = crop_part.split("crop=")[1].split("[")[0].split(":")
    crop_x = int(crop_coords[2])
    assert crop_x > 1500


def test_orchestrator_clip_scoring(monkeypatch):
    from backend_ai.orchestrator import ShortifyOrchestrator
    
    # Initialize orchestrator
    orchestrator = ShortifyOrchestrator()
    
    # Assert that score_clips node is present in the graph nodes list
    assert "score_clips" in orchestrator.app.nodes


def test_director_json_schema(monkeypatch):
    import json
    from backend_ai.agents.director_agent import CreativeDirector
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    
    captured_config = None
    class MockResponse:
        text = json.dumps({
            "title": "A", "storyline": "B", "total_duration": 10.0, "music_start_offset": 0.0,
            "timeline": [{"clip_name": "x.mp4", "start_in_clip": 0.0, "end_in_clip": 10.0,
                          "timeline_start": 0.0, "timeline_end": 10.0, "transition": "none",
                          "details": {"visual_cue": "v", "sound_design": "s", "pacing_style": "jump-cut", "is_hook": True}}]
        })

    class MockModels:
        def generate_content(self, model, contents, config=None):
            nonlocal captured_config
            captured_config = config
            return MockResponse()

    class MockClient:
        models = MockModels()

    import backend_ai.agents.director_agent as da
    monkeypatch.setattr(da, "get_gemini_client", lambda: MockClient())
    
    director = CreativeDirector()
    director.generate_edl(user_prompt="intent", audio_analysis={}, media_analyses=[], target_duration=10)
    
    assert captured_config is not None
    assert captured_config.response_mime_type == "application/json"



def test_director_quality_scores_context(monkeypatch):
    import json
    from backend_ai.agents.director_agent import CreativeDirector
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    
    captured_messages = []
    class MockResponse:
        def __init__(self, text):
            self.text = text

    class MockModels:
        def generate_content(self, model, contents, config=None):
            nonlocal captured_messages
            captured_messages = [
                {"role": "system", "content": config.system_instruction},
                {"role": "user", "content": contents}
            ]
            return MockResponse(json.dumps({
                "title": "A", "storyline": "B", "total_duration": 10.0, "music_start_offset": 0.0,
                "timeline": [{"clip_name": "x.mp4", "start_in_clip": 0.0, "end_in_clip": 10.0,
                              "timeline_start": 0.0, "timeline_end": 10.0, "transition": "none",
                              "details": {"visual_cue": "v", "sound_design": "s", "pacing_style": "jump-cut", "is_hook": True}}]
            }))

    class MockClient:
        models = MockModels()

    import backend_ai.agents.director_agent as da
    monkeypatch.setattr(da, "get_gemini_client", lambda: MockClient())
    
    director = CreativeDirector()
    
    pre_flight_report = {
        "media": [
            {
                "path": "C:\\path\\to\\clip_a.mp4:10:20",
                "quality_score": 0.85,
                "avg_sharpness": 150.5,
                "avg_brightness": 110.2
            }
        ]
    }
    
    media_analyses = [
        {
            "file_metadata": {"filename": "clip_a.mp4", "duration_seconds": 10.0},
            "summary": "Clip summary",
            "interesting_segments": [],
            "all_segments": []
        }
    ]
    
    director.generate_edl(
        user_prompt="intent",
        audio_analysis={},
        media_analyses=media_analyses,
        target_duration=10,
        pre_flight_report=pre_flight_report
    )
    
    user_msg_content = captured_messages[1]["content"]
    context_str = user_msg_content.split("Context Data: ")[1]
    context_data = json.loads(context_str)
    
    clip_info = context_data["available_clips"][0]
    assert clip_info["filename"] == "clip_a.mp4"
    assert clip_info["quality_score"] == 0.85
    assert clip_info["avg_sharpness"] == 150.5
    assert clip_info["avg_brightness"] == 110.2


def test_director_hook_enforcement(monkeypatch):
    import json
    from backend_ai.agents.director_agent import CreativeDirector
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    
    class MockResponse:
        def __init__(self, text):
            self.text = text

    class MockModels:
        def generate_content(self, model, contents, config=None):
            return MockResponse(json.dumps({
                "title": "A",
                "storyline": "B",
                "total_duration": 10.0,
                "music_start_offset": 0.0,
                "timeline": [
                    {
                        "clip_name": "x.mp4",
                        "start_in_clip": 0.0,
                        "end_in_clip": 10.0,
                        "timeline_start": 0.0,
                        "timeline_end": 10.0,
                        "transition": "none",
                        "details": {
                            "visual_cue": "v",
                            "sound_design": "s",
                            "pacing_style": "jump-cut",
                            "is_hook": False  # Hook is false!
                        }
                    }
                ]
            }))

    class MockClient:
        models = MockModels()

    import backend_ai.agents.director_agent as da
    monkeypatch.setattr(da, "get_gemini_client", lambda: MockClient())
    
    director = CreativeDirector()
    edl = director.generate_edl(user_prompt="intent", audio_analysis={}, media_analyses=[], target_duration=10)
    
    # Assert that details.is_hook was programmatically corrected to True
    assert edl["timeline"][0]["details"]["is_hook"] is True


def test_subtitle_styles_drawtext(tmp_path, monkeypatch):
    from backend_ai.agents.subtitle_agent import SubtitleAgent
    
    written_filter_content = ""
    original_open = open
    
    def mock_open(file, *args, **kwargs):
        nonlocal written_filter_content
        filename = str(file)
        if "temp_filter_" in filename and filename.endswith(".txt"):
            class MockFile:
                def __init__(self):
                    self.content = []
                def write(self, data):
                    self.content.append(data)
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    nonlocal written_filter_content
                    written_filter_content = "".join(self.content)
            return MockFile()
        return original_open(file, *args, **kwargs)
        
    monkeypatch.setattr("builtins.open", mock_open)
    monkeypatch.setattr("os.path.exists", lambda path: True)
    
    captured_cmd = []
    class MockSubprocessResult:
        returncode = 0
        stdout = ""
        stderr = ""
        
    def mock_run(cmd, **kwargs):
        nonlocal captured_cmd
        captured_cmd = cmd
        return MockSubprocessResult()
        
    monkeypatch.setattr("subprocess.run", mock_run)
    
    agent = SubtitleAgent(caption_style="hormozi")
    
    captions = [
        {
            "start": 0.5,
            "end": 2.5,
            "text": "Hello world caption",
            "words": [
                {"word": "Hello", "start": 0.5, "end": 1.0},
                {"word": "world", "start": 1.0, "end": 1.5},
                {"word": "caption", "start": 1.5, "end": 2.5}
            ]
        }
    ]
    
    # 1. Test minimal style
    agent.burn_subtitles("in.mp4", captions, "out.mp4", style="minimal")
    assert "drawtext" in written_filter_content
    assert "fontsize=32" in written_filter_content
    assert "fontcolor=white" in written_filter_content
    assert "between(t,0.500,2.500)" in written_filter_content
    assert "x='(w-" in written_filter_content
    assert "y='" in written_filter_content
    assert "Hello world caption" in written_filter_content
    
    # 2. Test bold style
    agent.burn_subtitles("in.mp4", captions, "out.mp4", style="bold")
    assert "drawtext" in written_filter_content
    assert "fontsize=52" in written_filter_content
    assert "borderw=4" in written_filter_content
    assert "bordercolor=black" in written_filter_content
    assert "HELLO WORLD CAPTION" in written_filter_content
    
    # 3. Test outline style
    agent.burn_subtitles("in.mp4", captions, "out.mp4", style="outline")
    assert "drawtext" in written_filter_content
    assert "fontsize=64" in written_filter_content
    assert "borderw=4" in written_filter_content
    assert "-filter_script:v" in captured_cmd
    
    # 4. Test hormozi style (word-level highlight animations)
    agent.burn_subtitles("in.mp4", captions, "out.mp4", style="hormozi")
    assert "drawtext" in written_filter_content
    assert "fontsize=44" in written_filter_content
    assert "fontcolor=white" in written_filter_content      # inactive text color
    assert "fontcolor=yellow" in written_filter_content     # active text highlight color
    assert "shadowx=2" in written_filter_content
    assert "shadowy=2" in written_filter_content
    assert "shadowcolor=black" in written_filter_content
    assert "box=1" in written_filter_content
    assert "boxcolor=black@0.5" in written_filter_content
    assert "HELLO" in written_filter_content
    assert "WORLD" in written_filter_content
    assert "CAPTION" in written_filter_content
    assert "between(t,0.500,1.000)" in written_filter_content
    assert "between(t,1.000,1.500)" in written_filter_content
    assert "between(t,1.500,2.500)" in written_filter_content



def test_media_analyst_normalize_analysis_durations(monkeypatch):
    from backend_ai.agents.media_agent import MediaAnalyst
    monkeypatch.setenv("GEMINI_API_KEY", "fake_key")
    analyst = MediaAnalyst()

    # Case 1: result is in MM.SS float format and needs normalization
    result = {
        "file_metadata": {"duration_seconds": 63.9},
        "interesting_segments": [
            {"start": 0.05, "end": 0.17},
            {"start": 0.54, "end": 1.04}
        ],
        "all_segments": [
            {"start": 0.0, "end": 0.05},
            {"start": 0.54, "end": 1.04}
        ]
    }
    normalized = analyst._normalize_analysis_durations(result)
    assert normalized["interesting_segments"][0]["start"] == 5.0
    assert normalized["interesting_segments"][0]["end"] == 17.0
    assert normalized["interesting_segments"][1]["start"] == 54.0
    assert normalized["interesting_segments"][1]["end"] == 64.0
    assert normalized["all_segments"][0]["start"] == 0.0
    assert normalized["all_segments"][0]["end"] == 5.0
    assert normalized["all_segments"][1]["start"] == 54.0
    assert normalized["all_segments"][1]["end"] == 64.0

    # Case 2: result is already in seconds, should NOT be normalized
    result2 = {
        "file_metadata": {"duration_seconds": 63.9},
        "interesting_segments": [
            {"start": 5.0, "end": 17.0},
            {"start": 54.0, "end": 63.9}
        ]
    }
    normalized2 = analyst._normalize_analysis_durations(result2)
    assert normalized2["interesting_segments"][0]["start"] == 5.0
    assert normalized2["interesting_segments"][0]["end"] == 17.0
    assert normalized2["interesting_segments"][1]["start"] == 54.0
    assert normalized2["interesting_segments"][1]["end"] == 63.9

    # Case 3: result with values that would go out of bounds if converted (should NOT be normalized)
    result3 = {
        "file_metadata": {"duration_seconds": 10.0},
        "interesting_segments": [
            {"start": 1.5, "end": 3.0}
        ]
    }
    normalized3 = analyst._normalize_analysis_durations(result3)
    assert normalized3["interesting_segments"][0]["start"] == 1.5
    assert normalized3["interesting_segments"][0]["end"] == 3.0
