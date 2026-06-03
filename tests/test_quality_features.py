import os
import sys
import pytest
from typing import Dict, Any, List
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend_ai.services.analyst_service import ProjectAnalystAgent, parse_virtual_segment
from backend_ai.services.edl_validation_service import _parse_virtual_clip_name
from backend_ai.services.editor_service import VideoEditor
from moviepy import ColorClip


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
    
    # Mock VideoFileClip duration
    class MockVideoFileClip(ColorClip):
        def __init__(self, path):
            super().__init__(size=(1080, 1920), color=(0,0,0), duration=45.0)

    monkeypatch.setattr("backend_ai.services.analyst_service.VideoFileClip", MockVideoFileClip)
    
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

    class MockVideoFileClip(ColorClip):
        def __init__(self, path):
            super().__init__(size=(1920, 1080), color=(0, 0, 0), duration=10.0)
            self.audio = None
            
        def subclipped(self, start, end):
            self.duration = end - start
            return self
            
        def cropped(self, **kwargs):
            return self
            
        def with_effects(self, effects):
            return self

    monkeypatch.setattr("backend_ai.services.editor_service.VideoFileClip", MockVideoFileClip)
    
    from backend_ai.schemas.edl import EDLDocument
    monkeypatch.setattr("backend_ai.services.editor_service.validate_edl", lambda edl, clips_dir: EDLDocument.model_validate(edl))

    # Mock assemble timeline and video writing
    monkeypatch.setattr(VideoEditor, "_assemble_timeline", lambda self, processed_clips, transitions: processed_clips[0])
    
    # Mock write_videofile to do nothing
    def mock_write_videofile(self, *args, **kwargs):
        pass
    monkeypatch.setattr("moviepy.video.VideoClip.VideoClip.write_videofile", mock_write_videofile)

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

    # Target duration = 5.0s, sum of clip durations in EDL is 5.0s
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

    # Mock clip returns duration of 3.0s (less than the EDL target of 5.0s)
    class MockVideoFileClip(ColorClip):
        def __init__(self, path):
            super().__init__(size=(1080, 1920), color=(0, 0, 0), duration=3.0)
            self.audio = None
            
        def subclipped(self, start, end):
            # Mock subclipped to return 3.0s
            self.duration = min(3.0, end - start)
            return self
            
        def cropped(self, **kwargs):
            return self
            
        def with_effects(self, effects):
            return self

    monkeypatch.setattr("backend_ai.services.editor_service.VideoFileClip", MockVideoFileClip)
    from backend_ai.schemas.edl import EDLDocument
    monkeypatch.setattr("backend_ai.services.editor_service.validate_edl", lambda edl, clips_dir: EDLDocument.model_validate(edl))

    # Mock assemble timeline to return MockVideoFileClip
    monkeypatch.setattr(VideoEditor, "_assemble_timeline", lambda self, processed_clips, transitions: processed_clips[0])

    # Let's inspect final_video duration in write_videofile mock
    written_duration = 0.0

    def mock_write_videofile(self, filename, *args, **kwargs):
        nonlocal written_duration
        written_duration = self.duration

    monkeypatch.setattr("moviepy.video.VideoClip.VideoClip.write_videofile", mock_write_videofile)

    editor = VideoEditor(clips_dir=str(clips_dir), output_dir=str(tmp_path / "exports"))
    editor.render(edl)

    # Duration should be exactly 5.0 seconds due to padding!
    assert written_duration == 5.0


def test_remove_silence(tmp_path, monkeypatch):
    class MockAudio(ColorClip):
        def __init__(self):
            super().__init__(size=(10, 10), color=(0, 0, 0), duration=6.0)
            self.nchannels = 2

        def to_soundarray(self, fps):
            # 2 seconds sound, 2 seconds silence, 2 seconds sound
            sr = 22050
            part1 = np.ones(2 * sr) * 0.5
            part2 = np.zeros(2 * sr)
            part3 = np.ones(2 * sr) * 0.5
            return np.concatenate([part1, part2, part3])

        def subclipped(self, start, end):
            sub = MockAudio()
            sub.duration = end - start
            return sub

    class MockClip(ColorClip):
        def __init__(self):
            super().__init__(size=(100, 100), color=(0, 0, 0), duration=6.0)
            self.audio = MockAudio()
            
        def subclipped(self, start, end):
            sub = MockClip()
            sub.audio = self.audio.subclipped(start, end)
            sub.duration = end - start
            return sub

    editor = VideoEditor(clips_dir=str(tmp_path), output_dir=str(tmp_path))
    clip = MockClip()
    
    cleaned = editor._remove_silence(clip, top_db=30)
    # Silence is stripped out, duration should be shortened
    assert cleaned.duration < 5.0


def test_normalize_audio(tmp_path, monkeypatch):
    class MockAudio:
        def __init__(self, amplitude=0.05):
            self.amplitude = amplitude
            self.volume_gain = 1.0
            
        def to_soundarray(self, fps):
            return np.ones(100) * self.amplitude
            
        def with_effects(self, effects):
            for fx in effects:
                self.volume_gain *= fx.factor
            return self

    class MockClip(ColorClip):
        def __init__(self):
            super().__init__(size=(100, 100), color=(0, 0, 0), duration=5.0)
            self.audio = MockAudio(amplitude=0.05)
            
        def with_audio(self, audio):
            self.audio = audio
            return self

    gains = []
    class FakeMultiplyVolume:
        def __init__(self, factor):
            self.factor = factor
            gains.append(factor)

    monkeypatch.setattr("backend_ai.services.editor_service.MultiplyVolume", FakeMultiplyVolume)

    editor = VideoEditor(clips_dir=str(tmp_path), output_dir=str(tmp_path))
    clip = MockClip()
    
    normalized = editor._normalize_audio(clip, target_rms=0.15)
    
    # Gain should be 3.0x to hit target RMS of 0.15
    assert normalized.audio.volume_gain == pytest.approx(3.0, abs=0.1)


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

    class MockVideoFileClip(ColorClip):
        def __init__(self, path):
            super().__init__(size=(1080, 1920), color=(0, 0, 0), duration=10.0)
            self.audio = None
            
        def subclipped(self, start, end):
            self.duration = end - start
            return self
            
        def cropped(self, **kwargs):
            return self
            
        def with_effects(self, effects):
            return self

    class MockAudioFileClip(ColorClip):
        def __init__(self, path):
            super().__init__(size=(10, 10), color=(0, 0, 0), duration=2.0)
            self.nchannels = 2
            
        def subclipped(self, start, end):
            self.duration = end - start
            return self
            
        def with_effects(self, effects):
            return self

    monkeypatch.setattr("backend_ai.services.editor_service.VideoFileClip", MockVideoFileClip)
    monkeypatch.setattr("backend_ai.services.editor_service.AudioFileClip", MockAudioFileClip)
    from backend_ai.schemas.edl import EDLDocument
    monkeypatch.setattr("backend_ai.services.editor_service.validate_edl", lambda edl, clips_dir: EDLDocument.model_validate(edl))
    monkeypatch.setattr(VideoEditor, "_assemble_timeline", lambda self, processed_clips, transitions: processed_clips[0])

    mixed_audio_duration = 0.0
    def mock_write_videofile(self, filename, *args, **kwargs):
        nonlocal mixed_audio_duration
        if self.audio:
            mixed_audio_duration = self.audio.duration

    monkeypatch.setattr("moviepy.video.VideoClip.VideoClip.write_videofile", mock_write_videofile)

    editor = VideoEditor(clips_dir=str(clips_dir), output_dir=str(tmp_path / "exports"))
    editor.render(edl, music_path=str(music_file))
    
    # Music loops successfully to cover full video length (10s)
    assert mixed_audio_duration == 10.0


def test_configurable_beat_snap_tolerance(tmp_path, monkeypatch):
    clips_dir = tmp_path / "testing_clips"
    clips_dir.mkdir()
    (clips_dir / "good_clip.mp4").write_bytes(b"good")

    class MockVideoFileClip(ColorClip):
        def __init__(self, path):
            super().__init__(size=(1080, 1920), color=(0, 0, 0), duration=10.0)
            self.audio = None
            
        def subclipped(self, start, end):
            self.duration = end - start
            return self
            
        def cropped(self, **kwargs):
            return self
            
        def with_effects(self, effects):
            return self

    monkeypatch.setattr("backend_ai.services.editor_service.VideoFileClip", MockVideoFileClip)
    from backend_ai.schemas.edl import EDLDocument
    monkeypatch.setattr("backend_ai.services.editor_service.validate_edl", lambda edl, clips_dir: EDLDocument.model_validate(edl))
    monkeypatch.setattr("moviepy.video.VideoClip.VideoClip.write_videofile", lambda *args, **kwargs: None)
    
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
    
    captured_clips = []
    def mock_assemble(self, processed_clips, transitions):
        captured_clips.extend(processed_clips)
        return processed_clips[0]
    monkeypatch.setattr(VideoEditor, "_assemble_timeline", mock_assemble)

    editor.render(edl_speed_ramp, rhythm_data=rhythm_data)
    assert len(editor.skipped_clips) == 0
    assert len(captured_clips) == 1
    assert captured_clips[0].duration == 4.8
    
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
    
    captured_clips.clear()
    editor.render(edl_cinematic, rhythm_data=rhythm_data)
    assert len(captured_clips) == 1
    assert captured_clips[0].duration == 5.0


def test_creative_director_drops_context(monkeypatch):
    import json
    from backend_ai.services.director_service import CreativeDirector
    
    # Mock GROQ_API_KEY env var
    monkeypatch.setenv("GROQ_API_KEY", "fake_key")
    
    captured_messages = []
    
    class MockCompletions:
        def create(self, model, messages, response_format=None):
            nonlocal captured_messages
            captured_messages = messages
            
            # Return a mock response with valid EDL JSON structure
            class MockMessage:
                content = json.dumps({
                    "title": "Mock Video",
                    "storyline": "Storyline",
                    "total_duration": 15.0,
                    "music_start_offset": 0.0,
                    "timeline": []
                })
            class MockChoice:
                message = MockMessage()
            class MockResponse:
                choices = [MockChoice()]
                
            return MockResponse()

    class MockGroq:
        def __init__(self, api_key):
            self.chat = type("Chat", (), {"completions": MockCompletions()})()

    monkeypatch.setattr("backend_ai.services.director_service.Groq", MockGroq)

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
    from backend_ai.services.media_service import MediaAnalyst
    
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

    # 2. Second test: cache is stale (8 days old)
    eight_days_ago = time.time() - 8 * 24 * 3600
    os.utime(cache_path, (eight_days_ago, eight_days_ago))
    
    res_stale = analyst.analyze_video(str(src_file))
    assert called_upload  # Expired cache, so upload was called!
    assert os.path.exists(cache_path)
    # The cache file should now be freshly written (modification time close to current time, not 8 days ago)
    assert os.path.getmtime(cache_path) > time.time() - 10


def test_media_analyst_upload_retry(tmp_path, monkeypatch):
    import time
    from backend_ai.services.media_service import MediaAnalyst
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
    from backend_ai.services.media_service import MediaAnalyst
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
    from backend_ai.services.clip_scoring_service import ClipScoringAgent
    
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"data")
    
    class MockVideoFileClip:
        def __init__(self, path):
            self.duration = 10.0
            
        def get_frame(self, t):
            return np.zeros((100, 100, 3), dtype=np.uint8)
            
        def __enter__(self):
            return self
            
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass
            
    monkeypatch.setattr("backend_ai.services.clip_scoring_service.VideoFileClip", MockVideoFileClip)
    
    agent = ClipScoringAgent()
    metrics = agent.score_video_segment(str(video_file), start=1.0, end=5.0)
    
    assert "sharpness" in metrics
    assert "motion_score" in metrics
    assert metrics["motion_type"] == "static"
    assert metrics["face_present"] is False
    assert "local_score" in metrics


def test_smart_face_cropping(tmp_path, monkeypatch):
    clips_dir = tmp_path / "testing_clips"
    clips_dir.mkdir()
    (clips_dir / "good_clip.mp4").write_bytes(b"good")

    class MockVideoFileClip(ColorClip):
        def __init__(self, path):
            super().__init__(size=(1600, 900), color=(0, 0, 0), duration=10.0)
            self.audio = None
            
        def subclipped(self, start, end):
            self.duration = end - start
            return self
            
        def cropped(self, x1, y1, width, height):
            self.cropped_x1 = x1
            self.cropped_y1 = y1
            return self
            
        def with_effects(self, effects):
            return self

    monkeypatch.setattr("backend_ai.services.editor_service.VideoFileClip", MockVideoFileClip)
    from backend_ai.schemas.edl import EDLDocument
    monkeypatch.setattr("backend_ai.services.editor_service.validate_edl", lambda edl, clips_dir: EDLDocument.model_validate(edl))
    
    captured_clip = None
    def mock_assemble(self, processed_clips, transitions):
        nonlocal captured_clip
        captured_clip = processed_clips[0]
        return processed_clips[0]
    monkeypatch.setattr(VideoEditor, "_assemble_timeline", mock_assemble)
    monkeypatch.setattr("moviepy.video.VideoClip.VideoClip.write_videofile", lambda *args, **kwargs: None)

    # Mock detect_face_center to return a face located on the far right (x=0.8, y=0.5)
    monkeypatch.setattr("backend_ai.services.editor_service.detect_face_center", lambda frame: (0.8, 0.5))

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
    
    assert captured_clip is not None
    # x1 should be shifted towards the right side of the frame (around 2190) rather than 1166 (default center)
    assert captured_clip.cropped_x1 > 1500


def test_orchestrator_clip_scoring(monkeypatch):
    from backend_ai.orchestrator import ShortifyOrchestrator
    
    # Initialize orchestrator
    orchestrator = ShortifyOrchestrator()
    
    # Assert that score_clips node is present in the graph nodes list
    assert "score_clips" in orchestrator.app.nodes



def test_director_json_schema(monkeypatch):
    import json
    from backend_ai.services.director_service import CreativeDirector
    monkeypatch.setenv("GROQ_API_KEY", "fake_key")
    
    captured_kwargs = {}
    class MockCompletions:
        def create(self, **kwargs):
            nonlocal captured_kwargs
            captured_kwargs = kwargs
            class MockMessage:
                content = json.dumps({
                    "title": "A", "storyline": "B", "total_duration": 10.0, "music_start_offset": 0.0,
                    "timeline": [{"clip_name": "x.mp4", "start_in_clip": 0.0, "end_in_clip": 10.0,
                                  "timeline_start": 0.0, "timeline_end": 10.0, "transition": "none",
                                  "details": {"visual_cue": "v", "sound_design": "s", "pacing_style": "jump-cut", "is_hook": True}}]
                })
            class MockChoice:
                message = MockMessage()
            class MockResponse:
                choices = [MockChoice()]
            return MockResponse()

    class MockGroq:
        def __init__(self, api_key):
            self.chat = type("Chat", (), {"completions": MockCompletions()})()

    monkeypatch.setattr("backend_ai.services.director_service.Groq", MockGroq)
    
    director = CreativeDirector()
    director.generate_edl(user_prompt="intent", audio_analysis={}, media_analyses=[], target_duration=10)
    
    assert "response_format" in captured_kwargs
    rf = captured_kwargs["response_format"]
    assert rf["type"] == "json_schema"
    assert rf["json_schema"]["name"] == "EDLDocument"
    assert "properties" in rf["json_schema"]["schema"]


def test_director_quality_scores_context(monkeypatch):
    import json
    from backend_ai.services.director_service import CreativeDirector
    monkeypatch.setenv("GROQ_API_KEY", "fake_key")
    
    captured_messages = []
    class MockCompletions:
        def create(self, **kwargs):
            nonlocal captured_messages
            captured_messages = kwargs["messages"]
            class MockMessage:
                content = json.dumps({
                    "title": "A", "storyline": "B", "total_duration": 10.0, "music_start_offset": 0.0,
                    "timeline": [{"clip_name": "x.mp4", "start_in_clip": 0.0, "end_in_clip": 10.0,
                                  "timeline_start": 0.0, "timeline_end": 10.0, "transition": "none",
                                  "details": {"visual_cue": "v", "sound_design": "s", "pacing_style": "jump-cut", "is_hook": True}}]
                })
            class MockChoice:
                message = MockMessage()
            class MockResponse:
                choices = [MockChoice()]
            return MockResponse()

    class MockGroq:
        def __init__(self, api_key):
            self.chat = type("Chat", (), {"completions": MockCompletions()})()

    monkeypatch.setattr("backend_ai.services.director_service.Groq", MockGroq)
    
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
    from backend_ai.services.director_service import CreativeDirector
    monkeypatch.setenv("GROQ_API_KEY", "fake_key")
    
    class MockCompletions:
        def create(self, **kwargs):
            # Return an EDL where the first clip details have is_hook = False
            class MockMessage:
                content = json.dumps({
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
                })
            class MockChoice:
                message = MockMessage()
            class MockResponse:
                choices = [MockChoice()]
            return MockResponse()

    class MockGroq:
        def __init__(self, api_key):
            self.chat = type("Chat", (), {"completions": MockCompletions()})()

    monkeypatch.setattr("backend_ai.services.director_service.Groq", MockGroq)
    
    director = CreativeDirector()
    edl = director.generate_edl(user_prompt="intent", audio_analysis={}, media_analyses=[], target_duration=10)
    
    # Assert that details.is_hook was programmatically corrected to True
    assert edl["timeline"][0]["details"]["is_hook"] is True


def test_subtitle_styles_ass(tmp_path, monkeypatch):
    from backend_ai.services.subtitle_service import SubtitleAgent
    
    # Mock subprocess.run to verify the generated ASS contents without running FFmpeg
    written_ass_content = ""
    original_open = open
    
    def mock_open(file, *args, **kwargs):
        nonlocal written_ass_content
        if str(file).endswith(".ass"):
            class MockFile:
                def __init__(self):
                    self.content = []
                def write(self, data):
                    self.content.append(data)
                def __enter__(self):
                    return self
                def __exit__(self, exc_type, exc_val, exc_tb):
                    nonlocal written_ass_content
                    written_ass_content = "".join(self.content)
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
    assert "Style: Default" in written_ass_content
    # minimal has size 48, outline 1.5, no shadow, Arial
    assert ",48," in written_ass_content
    assert ",1.5,0.0," in written_ass_content
    assert "Dialogue: 0," in written_ass_content
    # standard case, not uppercase for minimal
    assert "Hello world caption" in written_ass_content
    
    # 2. Test bold style
    agent.burn_subtitles("in.mp4", captions, "out.mp4", style="bold")
    # bold has size 72, outline 3.0, shadow 2.0, bold=1, uppercase
    assert ",72," in written_ass_content
    assert ",3.0,2.0," in written_ass_content
    assert "HELLO WORLD CAPTION" in written_ass_content
    
    # 3. Test outline style
    agent.burn_subtitles("in.mp4", captions, "out.mp4", style="outline")
    # outline has size 64, outline 4.0, shadow 0.0, uppercase
    assert ",64," in written_ass_content
    assert ",4.0,0.0," in written_ass_content
    assert "subtitles" in captured_cmd[5]
    
    # 4. Test hormozi style (word-level highlight animations)
    agent.burn_subtitles("in.mp4", captions, "out.mp4", style="hormozi")
    # hormozi has size 76, outline 5.0, shadow 0.0, uppercase
    assert ",76," in written_ass_content
    assert ",5.0,0.0," in written_ass_content
    
    # Dialogue events in hormozi should be split into individual words
    # Event 1: {\c&H0000FFFF&}HELLO{\r} WORLD CAPTION
    # Event 2: HELLO {\c&H0000FFFF&}WORLD{\r} CAPTION
    # Event 3: HELLO WORLD {\c&H0000FFFF&}CAPTION{\r}
    assert "{\\c&H0000FFFF&}HELLO{\\r} WORLD CAPTION" in written_ass_content
    assert "HELLO {\\c&H0000FFFF&}WORLD{\\r} CAPTION" in written_ass_content
    assert "HELLO WORLD {\\c&H0000FFFF&}CAPTION{\\r}" in written_ass_content
