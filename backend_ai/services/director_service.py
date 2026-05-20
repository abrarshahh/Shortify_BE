import os
import json
from typing import List, Dict, Any
from groq import Groq
from dotenv import load_dotenv
from backend_ai.core.config_loader import AGENTS_CONFIG
from backend_ai.core.api_utils import rate_limit_guard

load_dotenv()

class CreativeDirector:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment variables")
        
        self.client = Groq(api_key=api_key)
        config = AGENTS_CONFIG.get("creative_director", {})
        self.model_id = config.get("model", "llama-3.3-70b-versatile")

    @rate_limit_guard(max_retries=5)
    def generate_edl(
        self, 
        user_prompt: str, 
        audio_analysis: Dict[str, Any], 
        media_analyses: List[Dict[str, Any]],
        target_duration: int = 30,
        aspect_ratio: str = "9:16",
        style: str = "cinematic",
        feedback: str = None
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
                "highlights": analysis.get("interesting_segments", []),
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
              "transition": "none | jump_cut | crossfade | dip_to_black | slide_left | slide_right | zoom_in | zoom_out | glitch",
              "text_overlay": "On-screen text (leave empty unless explicitly requested)",
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
        - USER INTENT IS SUPREME: You MUST follow the User Intent perfectly. If the user asks for a specific scene, action, or chronological order, you must provide it exactly as requested, even if the priority_score is lower.
        - SMART CLIP SELECTION (STRICT CASCADE LOGIC): You MUST select clips using this exact priority sequence:
          1. FIRST, only look at segments from the 'highlights' array.
          2. FILTER by 'should_be_used': prioritize segments where 'should_be_used' is true.
          3. MATCH FOCUS: Ensure that the 'segment_focus' is consistent across the selected clips (e.g., if the main theme is 'mountain', try to pick other 'mountain' clips).
          4. If you still need more duration to hit the target, fall back to the 'segments' array and apply the exact same logic (highest priority_score, should_be_used: true, matching focus).
        - EXACT DURATION REQUIRED: You MUST calculate the total video length. The sum of all clip durations ('end_in_clip' - 'start_in_clip') MUST EXACTLY equal {target_duration} seconds.
        - DURATION MATCH: Ensure 'timeline_end - timeline_start' is exactly equal to 'end_in_clip - start_in_clip' for each clip.
        - MUSIC SELECTION: Use 'music_start_offset' to pick a good starting point from the audio track (e.g., an energy segment).
        - TEXT OVERLAYS: Only provide a 'text_overlay' if the user explicitly requests on-screen text, captions, or subtitles. Otherwise, it MUST be an empty string ("").
        - TRANSITIONS: Choose transitions deliberately based on style and narrative moment:
          - 'none' or 'jump_cut': default for fast_cut style, high energy sequences, beat-sync cuts. Never use crossfade on beat-sync cuts.
          - 'crossfade': smooth scene changes, travel style, when two clips share similar mood or color.
          - 'dip_to_black': dramatic scene breaks, time jumps, emotional pauses, cinematic style chapter markers.
          - 'zoom_in': push into an action moment, build tension before a reveal.
          - 'zoom_out': pull back after a climax, reveal scale or context.
          - 'slide_left' or 'slide_right': travel style, geographic transitions, before/after comparisons.
          - 'glitch': sparingly for fast_cut or dramatic style, maximum one or two times per reel.
        - BEAT SYNC: Try to align 'timeline_end' of clips with 'audio_rhythm' beats if possible.
        - PHOTO HANDLING: For clips where 'duration_seconds' is null in file_metadata, the clip is a photo.
          Photos have no inherent duration. You MUST assign a 'timeline_end - timeline_start' between 2.0
          and 5.0 seconds for photos. Photos pair well with 'dip_to_black' or 'crossfade' transitions.
          Never assign 'speed-ramp' or 'cinematic-slow' pacing to photos since they have no video to ramp.
          Use 'jump-cut' as the pacing_style for all photos.
        - Only use 'clip_name' from 'available_clips' or its sub-segment virtual forms.
        - SINGLE-SOURCE MULTI-CLIP MAPPING:
          - If the user uploaded a single long video and you want to use multiple non-contiguous segments from it, you MUST reference specific virtual segments using the format: "filename:start_time:end_time".
          - For example, if "vlog.mp4" has segments from 10.0 to 15.0 and from 25.0 to 30.0, you can specify two timeline items with "clip_name": "vlog.mp4:10.0:15.0" and "clip_name": "vlog.mp4:25.0:30.0" respectively.
          - When using this notation:
            - "start_in_clip" must be relative to the virtual segment boundaries (e.g. 0.0 is the beginning of the virtual segment).
            - "end_in_clip" must be also relative to the virtual segment (e.g. 5.0).
        - Only return the JSON object.
        """

        user_message = f"User Intent: {user_prompt}\n\nContext Data: {json.dumps(context, indent=2)}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        if feedback:
            messages.append({
                "role": "system",
                "content": f"CRITICAL REVISION DIRECTIVE:\nYour previous EDL generation failed rendering safety checks. Please correct this issue in your new EDL output:\n{feedback}"
            })

        import time
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
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
