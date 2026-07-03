import os
import sys
import pytest
from unittest.mock import MagicMock, patch
from pydantic import ValidationError

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend_ai.schemas.edl import EDLTimelineItem, AudioDuckingParams
from backend_ai.services.editor_service import VideoEditor

def test_audio_ducking_params_validation():
    # Valid params
    valid = AudioDuckingParams(original_audio_volume=0.5, music_volume_during_segment=0.2)
    assert valid.original_audio_volume == 0.5
    assert valid.music_volume_during_segment == 0.2

    # Out of bounds original_audio_volume
    with pytest.raises(ValidationError):
        AudioDuckingParams(original_audio_volume=-0.1, music_volume_during_segment=0.5)
    with pytest.raises(ValidationError):
        AudioDuckingParams(original_audio_volume=1.1, music_volume_during_segment=0.5)

    # Out of bounds music_volume_during_segment
    with pytest.raises(ValidationError):
        AudioDuckingParams(original_audio_volume=0.5, music_volume_during_segment=-0.5)
    with pytest.raises(ValidationError):
        AudioDuckingParams(original_audio_volume=0.5, music_volume_during_segment=1.5)

def test_edl_timeline_item_with_audio_ducking():
    item_dict = {
        "clip_name": "test.mp4",
        "start_in_clip": 0.0,
        "end_in_clip": 5.0,
        "timeline_start": 0.0,
        "timeline_end": 5.0,
        "transition": "none",
        "text_overlay": "Hello",
        "audio_ducking": {
            "original_audio_volume": 0.8,
            "music_volume_during_segment": 0.1
        },
        "details": {
            "visual_cue": "Cue",
            "sound_design": "whoosh",
            "pacing_style": "jump-cut",
            "is_hook": False,
            "keep_original_audio": True
        }
    }
    
    item = EDLTimelineItem.model_validate(item_dict)
    assert item.audio_ducking is not None
    assert item.audio_ducking.original_audio_volume == 0.8
    assert item.audio_ducking.music_volume_during_segment == 0.1

@patch("backend_ai.services.editor_service.ffmpeg")
@patch("backend_ai.services.editor_service.os.path.exists")
def test_build_ffmpeg_concat_volume_envelope(mock_exists, mock_ffmpeg):
    mock_exists.return_value = True
    
    editor = VideoEditor(clips_dir="fake_dir")
    editor.MUSIC_VOLUME = 0.9
    
    mock_input = MagicMock()
    mock_ffmpeg.input.return_value = mock_input
    
    mock_audio = MagicMock()
    mock_input.audio = mock_audio
    
    # Setup clip inputs
    clip_paths = ["clip1.mp4", "clip2.mp4"]
    clip_durations = [3.0, 4.0]
    clip_has_audio = [True, False]
    transitions = ["none", "none"]
    music_volumes = [0.15, 0.85]
    
    with patch("backend_ai.services.editor_service.subprocess.run") as mock_run:
        mock_run.return_value = MagicMock(return_value=0)
        
        editor._build_ffmpeg_concat(
            clip_paths=clip_paths,
            clip_durations=clip_durations,
            clip_has_audio=clip_has_audio,
            transitions=transitions,
            output_path="output.mp4",
            music_path="music.mp3",
            music_volumes=music_volumes
        )
        
    # Check if mock_audio.filter was called with 'volume' and a time-varying expression
    volume_filter_called = False
    volume_expr = ""
    for call in mock_audio.filter.mock_calls:
        args, kwargs = call[1], call[2]
        if args and args[0] == "volume":
            volume_filter_called = True
            volume_expr = args[1]
            break
            
    assert volume_filter_called
    assert "between(t,0.000,3.000)" in volume_expr
    assert "between(t,3.000,7.000)" in volume_expr
    assert "0.150" in volume_expr
    assert "0.850" in volume_expr

@patch("backend_ai.services.editor_service.detect_face_center")
@patch("backend_ai.services.editor_service.subprocess.Popen")
def test_process_single_clip_custom_original_volume(mock_popen, mock_detect_face):
    mock_detect_face.return_value = (0.5, 0.5)
    
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = MagicMock()
    mock_proc.stderr = MagicMock()
    mock_popen.return_value = mock_proc
    
    editor = VideoEditor(clips_dir="fake_dir")
    
    with patch.object(editor, "_check_has_audio", return_value=True), \
         patch.object(editor, "_get_audio_normalization_gain", return_value=1.5), \
         patch.object(editor, "_detect_non_silent_intervals", return_value=([], 10.0)), \
         patch("backend_ai.services.editor_service.cv2.VideoCapture") as mock_capture:
         
         mock_cap_instance = MagicMock()
         mock_cap_instance.isOpened.return_value = True
         
         def mock_get(prop):
             import cv2
             if prop == cv2.CAP_PROP_FRAME_WIDTH:
                 return 1080
             if prop == cv2.CAP_PROP_FRAME_HEIGHT:
                 return 1920
             if prop == cv2.CAP_PROP_FPS:
                 return 30.0
             if prop == cv2.CAP_PROP_FRAME_COUNT:
                 return 300
             return 0.0
             
         mock_cap_instance.get.side_effect = mock_get
         mock_capture.return_value = mock_cap_instance
         
         with patch("backend_ai.services.editor_service.subprocess.run") as mock_sub_run:
             mock_res = MagicMock()
             mock_res.returncode = 0
             mock_sub_run.return_value = mock_res
             
             editor._process_single_clip(
                 clip_path="clip.mp4",
                 is_image=False,
                 start_in=0.0,
                 end_in=5.0,
                 target_duration=5.0,
                 transition="none",
                 text_overlay="",
                 pacing="jump-cut",
                 music_active=True,
                 temp_path="temp_output.mp4",
                 original_audio_volume=0.75
             )
             
             cmd_args = mock_sub_run.call_args[0][0]
             filter_str = ""
             for idx, arg in enumerate(cmd_args):
                 if arg == "-filter_complex":
                     filter_str = cmd_args[idx + 1]
                     break
                     
             assert "volume=1.125" in filter_str
