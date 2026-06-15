import json
import os
import sys
from backend_ai.services.rhythm_service import RhythmEngineer

def run_audio_analysis(audio_path):
    if not os.path.exists(audio_path):
        print(f"Error: File not found at {audio_path}")
        return

    print(f"--- Starting Analysis for: {os.path.basename(audio_path)} ---")
    
    try:
        engineer = RhythmEngineer()
        result = engineer.analyze_music(audio_path)
        
        # Save to a JSON file for inspection
        output_file = "rhythm_analysis_output.json"
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)
            
        print(f"Analysis Complete!")
        print(f"Tempo: {result['tempo']:.2f} BPM")
        print(f"Beats Detected: {result['beat_count']}")
        print(f"Energy Segments: {len(result['energy_segments'])}")
        print(f"Sentiment: {result['sentiment']['label']} (Score: {result['sentiment']['score']})")
        print(f"Full JSON output saved to: {output_file}")
        
    except Exception as e:
        print(f"An error occurred during analysis: {e}")

if __name__ == "__main__":
    # Pointing to the test file
    audio_file = "path/to/music.mp3"
    run_audio_analysis(audio_file)
