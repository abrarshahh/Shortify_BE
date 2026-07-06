import os
import json
import shutil
import pytest
from backend_ai.orchestrator import ShortifyOrchestrator, AgentState

def test_project_cache_creation_and_bypass():
    # Setup test identifiers
    test_user = "test_user_cache"
    test_project = "test_project_123"
    
    # Instantiate orchestrator
    orchestrator = ShortifyOrchestrator(
        exports_dir="data/exports/test_run",
        project_id=test_project,
        user=test_user
    )
    
    # Verify cache directory property
    assert orchestrator.cache_dir == os.path.join("cache", test_user, test_project)
    
    # Ensure cache is cleaned before starting
    if os.path.exists(orchestrator.cache_dir):
        shutil.rmtree(orchestrator.cache_dir)
        
    # Setup mock agent state
    mock_state: AgentState = {
        "video_paths": ["clip1.mp4"],
        "music_path": "music.mp3",
        "project_title": "Test Title Prompt",
        "output_filename": "final_output.mp4",
        "target_duration": 15,
        "aspect_ratio": "9:16",
        "style": "travel",
        "caption_style": "minimal_pop",
        "add_subtitle": True,
        "add_stickers": False,
        "add_textoverlay": True,
        "rhythm_data": {"bpm": 120, "beat_times": [1.0, 2.0, 3.0]},
        "visual_data": [{"clip_name": "clip1.mp4", "score": 0.95}],
        "edl": {"title": "Test EDL storyboard", "timeline": []},
        "edl_feedback": "",
        "rendered_video_path": "",
        "color_graded_path": "",
        "safe_zone_report": {},
        "transcription": {},
        "final_video_path": "",
        "retry_count": 0,
        "max_edl_retries": 0,
        "pre_flight_report": {},
        "progress_callback": None,
        "clip_scores": {"clip1.mp4": {"composite_score": 0.88}},
        "dynamic_style": {},
        "has_cached_director": False
    }
    
    # 1. Test init_pipeline without cache
    init_res = orchestrator.node_init_pipeline(mock_state)
    assert init_res["has_cached_director"] is False
    assert orchestrator.route_init_pipeline(init_res) == "default_procedure"
    
    # 2. Test saving cache
    orchestrator._save_cache(mock_state)
    
    assert os.path.exists(orchestrator.cache_dir)
    assert orchestrator.media_agent.cache_dir == os.path.join(orchestrator.cache_dir, "media_analysis")
    assert orchestrator.clip_scoring_agent.cache_dir == os.path.join(orchestrator.cache_dir, "clip_scores")
    assert orchestrator.rhythm_agent.cache_dir == os.path.join(orchestrator.cache_dir, "music_analysis")
    assert os.path.isdir(orchestrator.media_agent.cache_dir)
    assert os.path.isdir(orchestrator.clip_scoring_agent.cache_dir)
    assert os.path.isdir(orchestrator.rhythm_agent.cache_dir)
    
    assert os.path.exists(os.path.join(orchestrator.cache_dir, "director_analysis", "director_analysis.json"))
    assert os.path.exists(os.path.join(orchestrator.cache_dir, "clip_scores", "clip_scores.json"))
    assert os.path.exists(os.path.join(orchestrator.cache_dir, "media_analysis", "media_analysis.json"))
    assert os.path.exists(os.path.join(orchestrator.cache_dir, "music_analysis", "music_analysis.json"))
    assert os.path.exists(os.path.join(orchestrator.cache_dir, "metadata", "metadata.json"))
    
    # Verify cache file content
    with open(os.path.join(orchestrator.cache_dir, "metadata", "metadata.json"), "r") as f:
        meta = json.load(f)
    assert meta["project_title"] == "Test Title Prompt"
    assert meta["target_duration"] == 15
    assert meta["aspect_ratio"] == "9:16"
    assert meta["style"] == "travel"
    
    # 3. Test init_pipeline with cache populated
    loaded_state = orchestrator.node_init_pipeline(mock_state)
    assert loaded_state["has_cached_director"] is True
    assert loaded_state["edl"]["title"] == "Test EDL storyboard"
    assert loaded_state["clip_scores"]["clip1.mp4"]["composite_score"] == 0.88
    assert loaded_state["visual_data"][0]["clip_name"] == "clip1.mp4"
    assert loaded_state["rhythm_data"]["bpm"] == 120
    
    # Verify routing condition works
    assert orchestrator.route_init_pipeline(loaded_state) == "direct_edit"
    
    # Cleanup cache
    shutil.rmtree(orchestrator.cache_dir)
    # Also remove root level cache directory if empty
    user_cache_dir = os.path.dirname(orchestrator.cache_dir)
    if os.path.exists(user_cache_dir) and not os.listdir(user_cache_dir):
        os.rmdir(user_cache_dir)
    cache_root = os.path.dirname(user_cache_dir)
    if os.path.exists(cache_root) and not os.listdir(cache_root):
        os.rmdir(cache_root)
