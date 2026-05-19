import os
import sys

# Add project root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend_ai.orchestrator import ShortifyOrchestrator, AgentState

def test_orchestrator():
    print("=== STARTING PHASE 7 END-TO-END TEST ===")
    
    # 1. Define Input Paths
    videos_dir = "path/to/Videos"
    music_dir = "path/to/Music"
    
    # Let's use the hiking video
    target_video = os.path.join(videos_dir, "video.mp4")
    # And a sample music track
    target_music = os.path.join(music_dir, "music.mp3")
    
    if not os.path.exists(target_video):
        print(f"Error: Target video not found at {target_video}")
        return
        
    music_path = target_music if os.path.exists(target_music) else None

    # 2. Setup Initial State
    initial_state: AgentState = {
        "video_paths": [target_video],
        "music_path": music_path,
        "project_title": "A short, motivational TikTok about a hiker struggling in deep snow. Make it highly engaging with fast pacing.",
        "output_filename": "test_output.mp4",
        "target_duration": 15,
        "aspect_ratio": "9:16",
        "style": "cinematic",
        
        # Initialize empty outputs
        "rhythm_data": {},
        "visual_data": [],
        "edl": {},
        "edl_feedback": "",
        "rendered_video_path": "",
        "color_graded_path": "",
        "safe_zone_report": {},
        "transcription": {},
        "final_video_path": "",
        "retry_count": 0,
        "pre_flight_report": {}
    }
    
    # 3. Run the Orchestrator
    orchestrator = ShortifyOrchestrator()
    final_state = orchestrator.run(initial_state)
    
    print("\n--- FINAL STATE OUTPUTS ---")
    print(f"Rendered Video Path: {final_state.get('rendered_video_path')}")
    print(f"Final Video Path:    {final_state.get('final_video_path')}")
    print(f"Safe Zone Verdict:   {final_state.get('safe_zone_report', {}).get('verdict')}")
    print("Test Complete.")

if __name__ == "__main__":
    test_orchestrator()
