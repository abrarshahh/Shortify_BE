import json
import os
import sys

# Add project root to sys.path to allow imports from backend_ai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend_ai.agents.media_agent import MediaAnalyst

def run_video_analysis(video_path):
    if not os.path.exists(video_path):
        print(f"Error: File not found at {video_path}")
        return

    print(f"--- Starting Gemini Analysis for: {os.path.basename(video_path)} ---")
    
    try:
        analyst = MediaAnalyst()
        result = analyst.analyze_video(video_path)
        
        # Save to a JSON file for inspection
        output_file = "media_analysis_output.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)
            
        print(f"Analysis Complete!")
        if "error" in result:
            print(f"Warning: {result['error']}")
        else:
            print(f"Mood: {result.get('mood')}")
            print(f"Segments Found: {len(result.get('interesting_segments', []))}")
        
        print(f"Full JSON output saved to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    # Pointing to the test file
    video_file = r"path/to/video.mp4"
    run_video_analysis(video_file)
