import json
import os
import sys

# Add project root to sys.path to allow imports from backend_ai
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from backend_ai.agents.director_agent import CreativeDirector

def test_director():
    print("--- Starting Creative Director (Groq) Test ---")
    
    # 1. Load Audio Analysis
    audio_path = "rhythm_analysis_output.json"
    if not os.path.exists(audio_path):
        print(f"Error: {audio_path} not found. Run test_rhythm.py first.")
        return
    with open(audio_path, "r") as f:
        audio_analysis = json.load(f)

    # 2. Load Media Analysis (Assuming one for now, but director handles list)
    media_path = "media_analysis_output.json"
    if not os.path.exists(media_path):
        print(f"Error: {media_path} not found. Run test_media.py first.")
        return
    with open(media_path, "r") as f:
        media_analysis = json.load(f)

    # 3. Define User Intent
    user_prompt = "Create a fast-paced, cinematic hiking reel with high-energy cuts matching the beat."

    # 4. Initialize and Run Director
    try:
        director = CreativeDirector()
        edl = director.generate_edl(
            user_prompt=user_prompt,
            audio_analysis=audio_analysis,
            media_analyses=[media_analysis] # Passing as a list
        )
        
        # Save output
        output_file = "edl_output.json"
        with open(output_file, "w") as f:
            json.dump(edl, f, indent=4)
            
        print(f"EDL Generation Complete!")
        print(f"Generated Title: {edl.get('title')}")
        print(f"Number of segments: {len(edl.get('timeline', []))}")
        print(f"Full EDL saved to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred during EDL generation: {e}")

if __name__ == "__main__":
    test_director()
