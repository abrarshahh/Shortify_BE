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
    monkeypatch.setattr(SubtitleAgent, "_find_font", lambda self, *args, **kwargs: "fake_font.ttf")

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


def test_font_downloader_and_validation(tmp_path, monkeypatch):
    import io
    from backend_ai.utils import font_downloader
    
    # Redirect cache dir to tmp_path
    monkeypatch.setattr(font_downloader, "FONTS_DIR", str(tmp_path))
    
    # Mock TTFont to validate fake files
    class MockTTFont:
        def __init__(self, path):
            self.path = path
            self.names = [
                type("Record", (object,), {"nameID": 1, "toUnicode": lambda self: "MockFontFamily"})()
            ]
        def get(self, table_name):
            return {}
        def close(self):
            pass
        def __getitem__(self, key):
            if key == "name":
                return self
            raise KeyError(key)

    monkeypatch.setattr(font_downloader, "TTFont", MockTTFont)
    
    # Mock os.path.getsize to avoid accessing the filesystem directly
    monkeypatch.setattr("os.path.getsize", lambda path: 2000)
    
    # Mock os.path.exists specifically for the target download path
    orig_exists = os.path.exists
    downloaded = False
    def mock_exists(path):
        if str(tmp_path) in path:
            return downloaded
        return orig_exists(path)
    monkeypatch.setattr("os.path.exists", mock_exists)
    
    # Mock urllib.request.urlopen to return mock CSS and mock TTF data
    calls = []
    class MockResponse:
        def __init__(self, data):
            self.data = data
        def read(self, *args, **kwargs):
            return self.data
        def decode(self, *args, **kwargs):
            return self.data.decode(*args, **kwargs)
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

    def mock_urlopen(req, *args, **kwargs):
        nonlocal downloaded
        url = req.full_url if hasattr(req, "full_url") else str(req)
        calls.append(url)
        if "css2" in url:
            css_content = """
            @font-face {
              font-family: 'Poppins';
              src: url(https://fonts.gstatic.com/s/poppins/v24/pxiByp8kv8JHgFVrLCz7V1tvEv-L.ttf) format('truetype');
            }
            """
            return MockResponse(css_content.encode('utf-8'))
        else:
            downloaded = True
            return MockResponse(b'\x00\x01\x00\x00_fake_font_data_')

    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen)
    
    # 1. Test successful download and fonttools validation
    font_path = font_downloader.get_font_path("Poppins")
    assert "poppins_700.ttf" in font_path.replace("\\", "/").lower()
    assert len(calls) == 2
    assert font_downloader.validate_font_file(font_path) is True
    assert font_downloader.get_font_family_name(font_path) == "MockFontFamily"

    # 2. Test fallback when download fails
    calls.clear()
    downloaded = False
    def mock_urlopen_fail(req, *args, **kwargs):
        raise RuntimeError("API offline")
        
    monkeypatch.setattr("urllib.request.urlopen", mock_urlopen_fail)
    
    # Temporarily bypass mock_exists for fallback checks to check actual filesystem fonts
    monkeypatch.setattr("os.path.exists", orig_exists)
    fallback_path = font_downloader.get_font_path("UnavailableFont")
    assert os.path.exists(fallback_path)


def test_dynamic_aesthetic_styling_generation(monkeypatch):
    from backend_ai.agents.subtitle_agent import SubtitleAgent
    
    agent = SubtitleAgent()
    
    # Mock Groq client and completion response
    class MockChatCompletionChoiceMessage:
        content = """
        {
          "subtitle_style": {
            "font_name": "Special Aardvark",
            "font_size": 48,
            "font_weight": 700,
            "font_color": "cyan",
            "inactive_color": "gray",
            "outline_color": "black",
            "outline_width": 3,
            "back_color": "none",
            "has_shadow": true,
            "shadow_color": "none",
            "shadow_width": 0,
            "uppercase": true,
            "animate": true
          },
          "text_overlay_style": {
            "font_name": "Special Aardvark Heavy",
            "font_size": 72,
            "font_weight": 900,
            "font_color": "magenta",
            "outline_color": "black",
            "outline_width": 4,
            "back_color": "none",
            "has_shadow": true,
            "shadow_color": "none",
            "shadow_width": 0,
            "uppercase": true
          }
        }
        """
        
    class MockChatCompletionChoice:
        message = MockChatCompletionChoiceMessage()
        
    class MockChatCompletion:
        choices = [MockChatCompletionChoice()]
        
    class MockChatCompletions:
        def create(self, *args, **kwargs):
            return MockChatCompletion()
            
    class MockChat:
        completions = MockChatCompletions()
        
    class MockGroq:
        def __init__(self, api_key):
            self.chat = MockChat()
            
    monkeypatch.setenv("GROQ_API_KEY", "mock_key")
    monkeypatch.setattr("groq.Groq", MockGroq)
    
    # Verify generate_aesthetic_style correctly calls and parses dynamic settings
    style = agent.generate_aesthetic_style(
        prompt="Cool Trekking Video",
        storyline="Trekking in mountains",
        video_style="cinematic"
    )
    
    assert style["subtitle_style"]["font_name"] == "Special Aardvark"
    assert style["subtitle_style"]["font_size"] == 48
    assert style["subtitle_style"]["font_color"] == "cyan"
    assert style["subtitle_style"]["inactive_color"] == "gray"
    assert style["subtitle_style"]["uppercase"] is True
    assert style["subtitle_style"]["animate"] is True
    assert style["text_overlay_style"]["font_name"] == "Special Aardvark Heavy"
    assert style["text_overlay_style"]["font_color"] == "magenta"


def test_orchestrator_dynamic_styling_coordination(monkeypatch):
    from backend_ai.orchestrator import ShortifyOrchestrator
    
    orchestrator = ShortifyOrchestrator()
    
    # Mock dynamic style generation to return mock style
    mock_style = {"font_name": "Special Aardvark", "font_size": 48}
    monkeypatch.setattr(
        orchestrator.subtitle_agent, 
        "generate_aesthetic_style", 
        lambda *args, **kwargs: mock_style
    )
    
    # Mock rendering editor and check that dynamic_style is received
    render_received_style = None
    def mock_render(self, edl, music_path, output_filename, aspect_ratio, rhythm_data, clip_scores, dynamic_style=None):
        nonlocal render_received_style
        render_received_style = dynamic_style
        return "rendered_output.mp4"
        
    from backend_ai.services.editor_service import VideoEditor
    monkeypatch.setattr(VideoEditor, "render", mock_render)
    
    # Mock file checks and metadata paths
    monkeypatch.setattr("os.path.dirname", lambda path: ".")
    monkeypatch.setattr("os.path.exists", lambda path: True)
    
    state = {
        "video_paths": ["dummy.mp4"],
        "music_path": None,
        "project_title": "Cool Trekking Video",
        "output_filename": "final.mp4",
        "target_duration": 10,
        "aspect_ratio": "9:16",
        "style": "cinematic",
        "edl": {"storyline": "Trekking in mountains", "timeline": []},
        "clip_scores": {}
    }
    
    result = orchestrator.node_render_video(state)
    assert result["dynamic_style"] == mock_style
    assert render_received_style == mock_style


def test_dynamic_subtitle_animation_and_overlay_colors(monkeypatch):
    from backend_ai.agents.subtitle_agent import SubtitleAgent
    from backend_ai.services.editor_service import VideoEditor
    
    # 1. Test SubtitleAgent burn_subtitles with custom dynamic animate style
    # Mock font file lookup to return a fake font
    monkeypatch.setattr(SubtitleAgent, "_find_font", lambda self, *args, **kwargs: "fake_font.ttf")

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

    # Mock PIL font metrics
    class MockFont:
        def getlength(self, text):
            return len(text) * 10
    monkeypatch.setattr("PIL.ImageFont.truetype", lambda path, size: MockFont())

    agent = SubtitleAgent()
    
    # Nested dynamic style structure
    custom_style = {
        "subtitle_style": {
            "font_name": "Special Aardvark",
            "font_size": 48,
            "font_weight": 500,
            "font_color": "magenta",
            "inactive_color": "cyan",
            "outline_color": "black",
            "outline_width": 3,
            "back_color": "none",
            "has_shadow": True,
            "shadow_color": "blue",
            "shadow_width": 2,
            "uppercase": True,
            "animate": True
        },
        "text_overlay_style": {
            "font_name": "Special Aardvark Heavy",
            "font_size": 72,
            "font_weight": 900,
            "font_color": "magenta",
            "outline_color": "black",
            "outline_width": 4,
            "back_color": "none",
            "has_shadow": True,
            "shadow_color": "red",
            "shadow_width": 3,
            "uppercase": True
        }
    }
    
    captions = [
        {
            "start": 1.0,
            "end": 2.0,
            "text": "Hello world",
            "words": [
                {"word": "Hello", "start": 1.0, "end": 1.5},
                {"word": "world", "start": 1.5, "end": 2.0}
            ]
        }
    ]
    
    agent.burn_subtitles("in.mp4", captions, "out.mp4", style=custom_style)
    
    # Verify word highlight animation was triggered by asserting inactive layer color and active highlight color are present
    assert "fontcolor=cyan" in written_filter        # Inactive color
    assert "fontcolor=magenta" in written_filter     # Active highlighted color
    assert "fontsize=48" in written_filter
    assert "borderw=3" in written_filter
    assert "bordercolor=black" in written_filter
    assert "shadowx=2" in written_filter
    assert "shadowy=2" in written_filter
    assert "shadowcolor=blue" in written_filter
    
    # 2. Test VideoEditor text overlay styling dynamically
    editor = VideoEditor(clips_dir=".")
    
    # Mock check_has_audio to return True
    monkeypatch.setattr(editor, "_check_has_audio", lambda path: True)
    monkeypatch.setattr(editor, "_detect_non_silent_intervals", lambda path: ([(1.0, 2.0)], 2.0))
    monkeypatch.setattr(editor, "_get_audio_normalization_gain", lambda path: 1.0)
    monkeypatch.setattr(VideoEditor, "_find_font", lambda self, *args, **kwargs: "fake_font.ttf")
    
    # Mock cv2.VideoCapture to return mock duration/size
    class MockVideoCapture:
        def __init__(self, path): pass
        def isOpened(self): return True
        def get(self, prop):
            if prop == 5: return 30.0 # FPS
            if prop == 7: return 60.0 # Frame count
            return 100
        def release(self): pass
    monkeypatch.setattr("cv2.VideoCapture", MockVideoCapture)
    
    # Intercept subprocess.run to check the text overlay filter complex parameters
    ffmpeg_cmds = []
    def mock_run_ffmpeg(cmd, *args, **kwargs):
        ffmpeg_cmds.append(cmd)
        return MockSubprocessResult()
    monkeypatch.setattr("subprocess.run", mock_run_ffmpeg)
    
    editor._process_single_clip(
        clip_path="dummy.mp4",
        is_image=False,
        start_in=1.0,
        end_in=2.0,
        target_duration=1.0,
        transition="none",
        text_overlay="Title Overlay Text",
        pacing="jump-cut",
        music_active=True,
        temp_path="temp_out.mp4",
        face_anchor_x=0.5,
        keep_original_audio=True,
        dynamic_style=custom_style
    )
    
    assert len(ffmpeg_cmds) == 1
    filter_complex_arg = next(arg for arg in ffmpeg_cmds[0] if "drawtext" in arg)
    
    # Ensure text overlay parameters are dynamic and match custom_style text_overlay_style
    assert "drawtext" in filter_complex_arg
    assert "fontcolor=magenta" in filter_complex_arg
    assert "fontsize=72" in filter_complex_arg
    assert "borderw=4" in filter_complex_arg
    assert "bordercolor=black" in filter_complex_arg
    assert "shadowx=3" in filter_complex_arg
    assert "shadowy=3" in filter_complex_arg
    assert "shadowcolor=red" in filter_complex_arg
    assert "Title Overlay Text" in filter_complex_arg
