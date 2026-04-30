import os
import json
from typing import List, Dict, Any
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

class CreativeDirector:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        self.model_id = "llama-3.3-70b-versatile" # Using the latest Llama 3.3 for high reasoning

    def generate_edl(self, user_prompt: str, audio_analysis: Dict[str, Any], media_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generates an Edit Decision List (EDL) by reasoning over audio beats and visual content.
        """
        
        # Prepare the context for the LLM
        context = {
            "user_intent": user_prompt,
            "audio_rhythm": {
                "tempo": audio_analysis.get("tempo"),
                "beats": audio_analysis.get("beat_times", [])[:50], # Limit for token budget if needed
                "energy_segments": audio_analysis.get("energy_segments", []),
                "audio_mood": audio_analysis.get("sentiment", {}).get("label")
            },
            "available_clips": []
        }

        for analysis in media_analyses:
            clip_info = {
                "filename": analysis.get("file_metadata", {}).get("filename"),
                "duration": analysis.get("file_metadata", {}).get("duration_seconds"),
                "summary": analysis.get("summary"),
                "hooks": [s for s in analysis.get("interesting_segments", []) if s.get("is_hook")],
                "segments": analysis.get("all_segments", [])
            }
            context["available_clips"].append(clip_info)

        system_prompt = """
        You are a top-tier Social Media Influencer and Viral Content Director. 
        Your goal is to create a high-retention Edit Decision List (EDL) for a TikTok/Reel that tells a compelling story.
        
        GOALS:
        1. INFLUENCER MINDSET: Think about "scroll-stopping" moments. The first 1.5 - 2 seconds MUST be a high-energy "hook".
        2. STORYLINE: Create a clear narrative arc (e.g., The Struggle -> The Process -> The Victory).
        3. STRATEGY: Identify what the viewer should focus on and what common editing mistakes to avoid for this specific content.
        4. DETAIL: Every timeline item should have 'details' that help the editor understand the vibe, sound design, and specific visual cues.
        
        OUTPUT FORMAT:
        You must return a raw JSON object with this structure:
        {
          "title": "Viral-worthy title",
          "storyline": "A 1-2 sentence narrative arc for the video",
          "influencer_strategy": {
            "main_focuses": ["Key visual/emotional elements to highlight"],
            "things_to_avoid": ["Clutter, slow starts, or irrelevant segments to skip"]
          },
          "total_duration": float,
          "timeline": [
            {
              "clip_name": "filename.mp4",
              "start_in_clip": float,
              "end_in_clip": float,
              "timeline_start": float,
              "timeline_end": float,
              "transition": "fade | crossfade | none | zoom_in | glitch",
              "text_overlay": "On-screen text",
              "details": {
                "visual_cue": "Specific action or object to focus on in this cut",
                "sound_design": "Suggested SFX (e.g., 'whoosh', 'camera click', 'bass drop')",
                "pacing_style": "speed-ramp | jump-cut | cinematic-slow"
              }
            }
          ]
        }
        
        RULES:
        - Transitions MUST align with 'audio_rhythm' beats.
        - Only use 'clip_name' from 'available_clips'.
        - Only return the JSON object.
        """

        user_message = f"User Intent: {user_prompt}\n\nContext Data: {json.dumps(context, indent=2)}"

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message}
            ],
            response_format={"type": "json_object"}
        )

        try:
            edl = json.loads(response.choices[0].message.content)
            return edl
        except Exception as e:
            print(f"Error parsing Groq response: {e}")
            return {
                "error": "Failed to generate structured EDL",
                "raw_response": response.choices[0].message.content
            }

if __name__ == "__main__":
    # Small test case if run directly
    director = CreativeDirector()
    print("Director initialized.")
