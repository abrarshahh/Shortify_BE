import os
import json
from typing import List, Dict, Any
from groq import Groq
from dotenv import load_dotenv
from backend_ai.core.config_loader import AGENTS_CONFIG

load_dotenv()

class CreativeDirector:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        config = AGENTS_CONFIG.get("creative_director", {})
        self.model_id = config.get("model", "llama-3.3-70b-versatile")

    def generate_edl(
        self, 
        user_prompt: str, 
        audio_analysis: Dict[str, Any], 
        media_analyses: List[Dict[str, Any]],
        target_duration: int = 30,
        aspect_ratio: str = "9:16",
        style: str = "cinematic"
    ) -> Dict[str, Any]:
        """
        Generates an Edit Decision List (EDL) by reasoning over audio beats and visual content.
        """
        
        # Prepare the context for the LLM
        context = {
            "user_intent": user_prompt,
            "target_duration": target_duration,
            "aspect_ratio": aspect_ratio,
            "style": style,
            "audio_rhythm": {
                "tempo": audio_analysis.get("tempo"),
                "beats": audio_analysis.get("beat_times", [])[:100], 
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

        system_prompt = f"""
        You are a top-tier Social Media Influencer and Viral Content Director. 
        Your goal is to create a high-retention Edit Decision List (EDL) for a TikTok/Reel that tells a compelling story.
        
        GOALS:
        1. INFLUENCER MINDSET: The first 1.5 - 2 seconds MUST be a high-energy "hook".
        2. STORYLINE: Create a clear, sequential narrative arc (e.g., Setup -> Conflict -> Resolution).
        3. DURATION: The total video duration MUST be approximately {target_duration} seconds.
        4. STYLE: Follow the '{style}' style. 
           - 'cinematic': slow fades, dramatic zooms.
           - 'fast_cut': rapid transitions, high energy.
           - 'travel': smooth slides, upbeat pacing.
        
        OUTPUT FORMAT:
        You must return a raw JSON object with this structure:
        {{
          "title": "Viral-worthy title",
          "storyline": "A 1-2 sentence narrative arc",
          "total_duration": float,
          "music_start_offset": float,
          "timeline": [
            {{
              "clip_name": "filename.mp4",
              "start_in_clip": float,
              "end_in_clip": float,
              "timeline_start": float,
              "timeline_end": float,
              "transition": "none | fade | crossfade | slide_left | slide_right | zoom_in | zoom_out | glitch",
              "text_overlay": "On-screen text",
              "details": {{
                "visual_cue": "Specific action to focus on",
                "sound_design": "SFX (e.g., 'whoosh', 'bass drop')",
                "pacing_style": "speed-ramp | jump-cut | cinematic-slow"
              }}
            }}
          ]
        }}
        
        RULES:
        - SEQUENTIAL ORDER: 'timeline_start' must strictly increase. No overlapping clips.
        - NO REPETITION: Avoid using the same clip twice in a row. Use different clips to maintain visual interest.
        - DURATION MATCH: Ensure 'timeline_end - timeline_start' is roughly equal to 'end_in_clip - start_in_clip'.
        - MUSIC SELECTION: Use 'music_start_offset' to pick a good starting point from the audio track (e.g., an energy segment).
        - TRANSITIONS: Only add transitions if it improves the flow. Use 'none' for most cuts.
        - BEAT SYNC: Try to align 'timeline_end' of clips with 'audio_rhythm' beats if possible.
        - Only use 'clip_name' from 'available_clips'.
        - Only return the JSON object.
        """

        user_message = f"User Intent: {user_prompt}\n\nContext Data: {json.dumps(context, indent=2)}"

        import time
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message}
                    ],
                    response_format={"type": "json_object"}
                )
                break
            except Exception as e:
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise e

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
