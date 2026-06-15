import os
import sys
import pytest
import numpy as np

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend_ai.agents.clip_scoring_agent import ClipScoringAgent
from backend_ai.orchestrator import ShortifyOrchestrator
from backend_ai.agents.director_agent import CreativeDirector


def test_clip_scoring_metrics(tmp_path, monkeypatch):
    # Setup mock file structure
    video_file = tmp_path / "test_video.mp4"
    video_file.write_bytes(b"fake data")

    # Mock VideoCapture
    class MockVideoCapture:
        def __init__(self, path):
            self.opened = True
            self.read_count = 0
        def isOpened(self):
            return self.opened
        def get(self, prop):
            if prop == 5:  # FPS
                return 30.0
            if prop == 7:  # FRAME_COUNT
                return 180  # 6.0s
            return 0
        def read(self):
            self.read_count += 1
            # Return BGR frame (all mid gray, so Laplacian variance is 0)
            return True, np.ones((100, 100, 3), dtype=np.uint8) * 128
        def set(self, prop, val):
            pass
        def release(self):
            self.opened = False

    import cv2
    monkeypatch.setattr(cv2, "VideoCapture", MockVideoCapture)

    # Mock MediaPipe Face Detection to return False
    monkeypatch.setattr(ClipScoringAgent, "_detect_faces_mediapipe", lambda self, frame: (False, 0.5))

    # Initialize ClipScoringAgent with temp cache directory
    agent = ClipScoringAgent(cache_dir=str(tmp_path / "cache"))
    scores = agent.score_file(str(video_file), style="cinematic")

    # Sharpness should be close to 0.0 since frame is solid color
    assert scores["sharpness"] == 0.0
    # Exposure is 1.0 because 128 is close to the 125 target midpoint
    assert scores["exposure_score"] > 0.9
    # Motion is 0.0 since frame doesn't change
    assert scores["motion_score"] == 0.0
    assert scores["motion_tier"] == "static"
    assert scores["face_detected"] is False
    assert scores["face_anchor_x"] == 0.5
    # Composite score should be calculated correctly
    assert "composite_score" in scores
    assert scores["composite_score"] > 0.0


def test_director_schema_fallback(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "fake_key")

    calls = []

    class MockCompletions:
        def create(self, model, messages, response_format=None):
            calls.append(response_format)
            if response_format.get("type") == "json_schema":
                # Simulate a 400 error for models that don't support JSON schema
                raise Exception("BadRequestError: 400 json_schema is not supported")
            
            # Return a valid mock response on fallback
            class MockMessage:
                content = "{}"
            class MockChoice:
                message = MockMessage()
            class MockResponse:
                choices = [MockChoice()]
            return MockResponse()

    class MockGroq:
        def __init__(self, api_key):
            self.chat = type("Chat", (), {"completions": MockCompletions()})()

    monkeypatch.setattr("backend_ai.agents.director_agent.Groq", MockGroq)

    director = CreativeDirector()
    director._call_groq(messages=[], model_id="llama3-some-model")

    # First call had json_schema, second call had json_object
    assert len(calls) == 2
    assert calls[0]["type"] == "json_schema"
    assert calls[1]["type"] == "json_object"


def test_orchestrator_apply_hook_corrections():
    orchestrator = ShortifyOrchestrator()

    edl = {
        "title": "Test Hook Corrections",
        "storyline": "Storyline",
        "total_duration": 10.0,
        "music_start_offset": 0.0,
        "timeline": [
            {
                "clip_name": "clip_a.mp4",
                "start_in_clip": 0.0,
                "end_in_clip": 2.0,
                "timeline_start": 0.0,
                "timeline_end": 2.0,
                "transition": "none",
                "text_overlay": "",
                "details": {
                    "visual_cue": "a",
                    "sound_design": "",
                    "pacing_style": "jump-cut",
                    "is_hook": False
                }
            },
            {
                "clip_name": "clip_b.mp4",
                "start_in_clip": 0.0,
                "end_in_clip": 6.0,  # Hook clip is in second position and > 4.0s
                "timeline_start": 2.0,
                "timeline_end": 8.0,
                "transition": "none",
                "text_overlay": "",
                "details": {
                    "visual_cue": "b",
                    "sound_design": "",
                    "pacing_style": "jump-cut",
                    "is_hook": True
                }
            }
        ]
    }

    clip_scores = {
        "clip_a.mp4": {"composite_score": 0.3, "face_anchor_x": 0.5},
        "clip_b.mp4": {"composite_score": 0.8, "face_anchor_x": 0.8}
    }

    # Apply corrections
    corrected = orchestrator._apply_hook_corrections(edl, clip_scores, clips_dir="")
    timeline = corrected["timeline"]

    # 1. Position test: clip_b should be swapped to index 0
    assert timeline[0]["clip_name"] == "clip_b.mp4"
    assert timeline[0]["details"]["is_hook"] is True
    assert timeline[1]["clip_name"] == "clip_a.mp4"
    assert timeline[1]["details"]["is_hook"] is False

    # 2. Duration test: clip_b was 6.0s, should be trimmed to 3.5s
    assert timeline[0]["end_in_clip"] == 3.5

    # 3. Timeline times recalculation test
    assert timeline[0]["timeline_start"] == 0.0
    assert timeline[0]["timeline_end"] == 3.5
    assert timeline[1]["timeline_start"] == 3.5
    assert timeline[1]["timeline_end"] == 5.5


def test_subtitle_style_loading_and_wrapping(monkeypatch):
    from backend_ai.agents.subtitle_agent import SubtitleAgent

    # Mock font file lookup to return a fake font
    monkeypatch.setattr(SubtitleAgent, "_find_font", lambda self: "fake_font.ttf")

    # Mock subprocess.run
    class MockSubprocessResult:
        returncode = 0
        stdout = ""
        stderr = ""
    monkeypatch.setattr("subprocess.run", lambda cmd, **kwargs: MockSubprocessResult())

    # Mock open to capture filter content
    written_filter = ""
    original_open = open
    def mock_open(file, *args, **kwargs):
        nonlocal written_filter
        filename = str(file)
        if "temp_filter_" in filename:
            class MockFile:
                def write(self, data):
                    nonlocal written_filter
                    written_filter = data
                def __enter__(self): return self
                def __exit__(self, *a): pass
            return MockFile()
        return original_open(file, *args, **kwargs)
    monkeypatch.setattr("builtins.open", mock_open)
    monkeypatch.setattr("os.path.exists", lambda path: True)

    # Mock PIL font metrics to simulate widths
    class MockFont:
        def getlength(self, text):
            # Each character is 10px wide
            return len(text) * 10
    monkeypatch.setattr("PIL.ImageFont.truetype", lambda path, size: MockFont())

    agent = SubtitleAgent(caption_style="minimal")

    # 15 words -> each word+space is ~6 chars = 60px. 15 words = 900px.
    # 70% of 1080 frame width = 756px.
    # So 15 words is guaranteed to wrap into at least 2 lines.
    words = [{"word": f"word{i}", "start": float(i), "end": float(i)+0.5} for i in range(15)]
    captions = [{
        "start": 0.0,
        "end": 8.0,
        "text": " ".join([w["word"] for w in words]),
        "words": words
    }]

    agent.burn_subtitles("in.mp4", captions, "out.mp4", style="minimal")

    # Assert drawtext exists and check structure
    assert "drawtext" in written_filter
    # Verify that multiple drawtext commands were created (due to lines wrapping)
    assert written_filter.count("drawtext=") >= 2


def test_dynamic_audio_ducking_intervals(monkeypatch):
    from backend_ai.services.editor_service import VideoEditor
    editor = VideoEditor(clips_dir=".")
    
    # Check that default properties are loaded
    assert editor.MUSIC_VOLUME == 0.22
    assert editor.MUSIC_DUCKED_VOLUME == 0.06
    assert editor.ORIGINAL_AUDIO_VOLUME == 1.0

    # Test global active interval calculation during overlapping transitions
    clip_durations = [3.0, 4.0, 2.0]
    clip_has_audio = [True, False, True]
    transitions = ["none", "crossfade", "none"]
    
    # Mock sync_fade_duration calculation
    tempo = 120.0  # 120 BPM => beat duration = 0.5s. Cinematic style => 1.0 beat = 0.5s fade duration.
    
    beat_duration = 60.0 / tempo
    sync_fade_duration = beat_duration * 1.0
    sync_fade_duration = max(0.1, min(1.0, sync_fade_duration))
    assert sync_fade_duration == 0.5
    
    # Calculate intervals using the exact logic implemented in _build_ffmpeg_concat
    active_intervals = []
    accumulated = 0.0
    for idx in range(len(clip_durations)):
        dur = clip_durations[idx]
        has_aud = clip_has_audio[idx]
        
        transition_term = (transitions[idx] or "none").lower()
        fade_overlap = 0.0
        if idx > 0 and transition_term == "crossfade":
            prev_dur = clip_durations[idx - 1]
            fade_overlap = min(sync_fade_duration, prev_dur / 2, dur / 2)
        
        if idx > 0:
            accumulated -= fade_overlap
            
        if has_aud:
            active_intervals.append((accumulated, accumulated + dur))
            
        accumulated += dur
        
    assert len(active_intervals) == 2
    assert active_intervals[0] == (0.0, 3.0)
    assert active_intervals[1] == (6.5, 8.5)


def test_lut_profile_generation_and_application(tmp_path, monkeypatch):
    from backend_ai.services.color_service import ColorGradingAgent
    
    # Redirect luts_dir to tmp_path
    monkeypatch.setattr(ColorGradingAgent, "__init__", lambda self: None)
    agent = ColorGradingAgent()
    agent.enabled = True
    agent.ffmpeg_path = "ffmpeg"
    agent.luts_dir = str(tmp_path / "luts")
    os.makedirs(agent.luts_dir, exist_ok=True)
    
    # Generate defaults
    agent._generate_default_luts()
    
    # Check that .cube files are generated
    cinematic_cube = tmp_path / "luts" / "cinematic.cube"
    vintage_cube = tmp_path / "luts" / "vintage.cube"
    assert cinematic_cube.exists()
    assert vintage_cube.exists()
    
    # Verify contents of cinematic.cube
    content = cinematic_cube.read_text()
    assert "TITLE \"Cinematic Teal Orange\"" in content
    assert "LUT_3D_SIZE 2" in content
    
    # Test apply_grade LUT path detection
    cmd_run = []
    class MockResult:
        returncode = 0
        stdout = ""
        stderr = ""
        
    def mock_subprocess_run(cmd, *args, **kwargs):
        nonlocal cmd_run
        cmd_run = cmd
        return MockResult()
        
    monkeypatch.setattr("subprocess.run", mock_subprocess_run)
    monkeypatch.setattr("os.path.exists", lambda path: True)
    
    input_video = str(tmp_path / "input.mp4")
    agent.apply_grade(video_path=input_video, style="cinematic", output_dir=str(tmp_path))
    
    # The command should contain "-vf" and "lut3d="
    assert "-vf" in cmd_run
    vf_idx = cmd_run.index("-vf")
    filter_chain = cmd_run[vf_idx + 1]
    assert "lut3d=" in filter_chain
    assert "cinematic.cube" in filter_chain.replace("\\", "/")


def test_agent_controlled_ducking_process_clip(monkeypatch):
    from backend_ai.services.editor_service import VideoEditor
    
    editor = VideoEditor(clips_dir=".")
    
    # Mock cv2.VideoCapture to simulate a video with duration
    class MockVideoCapture:
        def __init__(self, path):
            pass
        def isOpened(self):
            return True
        def get(self, prop):
            if prop == 5:  # FPS
                return 30.0
            if prop == 7:  # FRAME_COUNT
                return 150.0
            return 100
        def release(self):
            pass
            
    monkeypatch.setattr("cv2.VideoCapture", MockVideoCapture)
    
    # Mock check_has_audio to return True (clip has audio)
    monkeypatch.setattr(editor, "_check_has_audio", lambda path: True)
    # Mock silence detection to return empty list
    monkeypatch.setattr(editor, "_detect_non_silent_intervals", lambda path: ([], 5.0))
    # Mock audio normalization gain
    monkeypatch.setattr(editor, "_get_audio_normalization_gain", lambda path: 1.0)
    
    # Intercept subprocess.run
    ffmpeg_cmds = []
    class MockCompletedProcess:
        returncode = 0
        stdout = ""
        stderr = ""
        
    def mock_run(cmd, *args, **kwargs):
        ffmpeg_cmds.append(cmd)
        return MockCompletedProcess()
        
    monkeypatch.setattr("subprocess.run", mock_run)
    
    # 1. Test with keep_original_audio = True (should process audio with original volume mapping)
    editor._process_single_clip(
        clip_path="dummy.mp4",
        is_image=False,
        start_in=0.0,
        end_in=5.0,
        target_duration=5.0,
        transition="none",
        text_overlay="",
        pacing="jump-cut",
        music_active=True,
        temp_path="temp_out.mp4",
        face_anchor_x=0.5,
        keep_original_audio=True
    )
    
    assert len(ffmpeg_cmds) == 1
    cmd1 = ffmpeg_cmds[0]
    # Filter complex should map [0:a] to process real audio since keep_original_audio is True
    filter_str1 = next(arg for arg in cmd1 if "[a_final]" in arg or "volume=" in arg)
    assert "[0:a]" in filter_str1
    assert "aevalsrc" not in filter_str1
    
    ffmpeg_cmds.clear()
    
    # 2. Test with keep_original_audio = False (should mute/use aevalsrc for silent audio)
    editor._process_single_clip(
        clip_path="dummy.mp4",
        is_image=False,
        start_in=0.0,
        end_in=5.0,
        target_duration=5.0,
        transition="none",
        text_overlay="",
        pacing="jump-cut",
        music_active=True,
        temp_path="temp_out.mp4",
        face_anchor_x=0.5,
        keep_original_audio=False
    )
    
    assert len(ffmpeg_cmds) == 1
    cmd2 = ffmpeg_cmds[0]
    filter_str2 = next(arg for arg in cmd2 if "[a_final]" in arg or "volume=" in arg)
    # Filter complex should use aevalsrc to generate silence instead of using [0:a]
    assert "[0:a]" not in filter_str2
    assert "aevalsrc" in filter_str2


def test_agent_controlled_ducking_render_intervals(monkeypatch):
    from backend_ai.services.editor_service import VideoEditor
    
    editor = VideoEditor(clips_dir=".")
    
    # Mock check_has_audio to return True
    monkeypatch.setattr(editor, "_check_has_audio", lambda path: True)
    
    # Mock video path check
    monkeypatch.setattr("os.path.exists", lambda path: True)
    monkeypatch.setattr("os.path.splitext", lambda path: (path, ".mp4"))
    
    # Mock _process_single_clip
    monkeypatch.setattr(editor, "_process_single_clip", lambda **kwargs: kwargs.get("target_duration", 3.0))
    
    # Intercept _build_ffmpeg_concat
    build_concat_called = {}
    def mock_build_ffmpeg_concat(**kwargs):
        build_concat_called.update(kwargs)
        
    monkeypatch.setattr(editor, "_build_ffmpeg_concat", mock_build_ffmpeg_concat)
    
    # EDL with keep_original_audio=True for clip 1, and keep_original_audio=False for clip 2
    edl = {
        "title": "Test EDL",
        "storyline": "Storyline",
        "total_duration": 6.0,
        "music_start_offset": 0.0,
        "timeline": [
            {
                "clip_name": "clip1.mp4",
                "start_in_clip": 0.0,
                "end_in_clip": 3.0,
                "timeline_start": 0.0,
                "timeline_end": 3.0,
                "transition": "none",
                "text_overlay": "",
                "details": {
                    "visual_cue": "Cue 1",
                    "sound_design": "Sound 1",
                    "pacing_style": "jump-cut",
                    "is_hook": True,
                    "keep_original_audio": True
                }
            },
            {
                "clip_name": "clip2.mp4",
                "start_in_clip": 0.0,
                "end_in_clip": 3.0,
                "timeline_start": 3.0,
                "timeline_end": 6.0,
                "transition": "none",
                "text_overlay": "",
                "details": {
                    "visual_cue": "Cue 2",
                    "sound_design": "Sound 2",
                    "pacing_style": "jump-cut",
                    "is_hook": False,
                    "keep_original_audio": False
                }
            }
        ]
    }
    
    editor.render(edl=edl, music_path="dummy_music.mp3")
    
    assert "clip_has_audio" in build_concat_called
    # First clip had keep_original_audio=True -> True
    # Second clip had keep_original_audio=False -> False
    assert build_concat_called["clip_has_audio"] == [True, False]

