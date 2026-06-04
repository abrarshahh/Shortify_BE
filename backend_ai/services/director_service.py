import os
import json
import logging
from typing import List, Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv
from backend_ai.core.config_loader import AGENTS_CONFIG
from backend_ai.core.api_utils import rate_limit_guard
from backend_ai.schemas.edl import EDLDocument

load_dotenv()
logger = logging.getLogger("agents.director")

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
        feedback: str = None,
        pre_flight_report: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Generates an Edit Decision List (EDL) by reasoning over audio beats and visual content.
        """
        logger.info(f"CreativeDirector: Starting EDL generation. Prompt: '{user_prompt}'")
        logger.debug(f"CreativeDirector: Aspect Ratio: {aspect_ratio}, Style: {style}, Target Duration: {target_duration}s")
        if feedback:
            logger.warning(f"CreativeDirector: Re-generating EDL due to validation failure. Feedback: {feedback}")
        
        # Build a lookup mapping base filename to pre-flight scores
        quality_lookup = {}
        if pre_flight_report and "media" in pre_flight_report:
            for m in pre_flight_report["media"]:
                path_str = m["path"]
                # Windows safe split for virtual segment notation (filename:start:end)
                if ":" in path_str:
                    parts = path_str.rsplit(":", 2)
                    # Check if the split ends represent numeric start and end times
                    if len(parts) == 3 and parts[1].replace(".", "").isdigit() and parts[2].replace(".", "").isdigit():
                        base_path = parts[0]
                    else:
                        base_path = path_str
                else:
                    base_path = path_str
                filename = os.path.basename(base_path)
                quality_lookup[filename] = {
                    "quality_score": m.get("quality_score", 0.5),
                    "avg_sharpness": m.get("avg_sharpness", 100.0),
                    "avg_brightness": m.get("avg_brightness", 128.0)
                }

        # Prepare the context for the LLM
        context = {
            "user_intent": user_prompt,
            "target_duration": target_duration,
            "aspect_ratio": aspect_ratio,
            "style": style,
            "audio_rhythm": {
                "tempo": audio_analysis.get("tempo"),
                "beats": audio_analysis.get("beat_times", [])[:100], 
                "drops": audio_analysis.get("peak_times", [])[:100],
                "energy_segments": audio_analysis.get("energy_segments", []),
                "audio_mood": audio_analysis.get("sentiment", {}).get("label")
            },
            "available_clips": []
        }

        for analysis in media_analyses:
            filename = analysis.get("file_metadata", {}).get("filename")
            q_info = quality_lookup.get(filename, {
                "quality_score": 0.5,
                "avg_sharpness": 100.0,
                "avg_brightness": 128.0
            })
            clip_info = {
                "filename": filename,
                "duration": analysis.get("file_metadata", {}).get("duration_seconds"),
                "quality_score": q_info["quality_score"],
                "avg_sharpness": q_info["avg_sharpness"],
                "avg_brightness": q_info["avg_brightness"],
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
                "pacing_style": "speed-ramp | jump-cut | cinematic-slow",
                "is_hook": boolean
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
          2. FILTER by 'should_be_used': prioritize segments where 'should_be_used' is true, and further prioritize segments with higher 'local_score' (scored by our local ClipScoringAgent).
          3. MATCH PACING STYLE: You MUST match the 'pacing_style' under 'details' based on the clip segment's 'motion_type' metric. High-motion segments ('motion_type' == 'high-motion') MUST be assigned to 'speed-ramp' pacing; static segments ('motion_type' == 'static') MUST be assigned to 'cinematic-slow' pacing.
          4. MATCH FOCUS: Ensure that the 'segment_focus' is consistent across the selected clips (e.g., if the main theme is 'mountain', try to pick other 'mountain' clips).
          5. If you still need more duration to hit the target, fall back to the 'segments' array and apply the exact same logic (highest priority_score, should_be_used: true, matching focus).
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
        - MUSIC DROPS: Align major high-energy transitions, sound design triggers (like bass drops or whooshes), and important visual actions with the music 'drops' timestamps in 'audio_rhythm' for maximum impact.
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

        logger.info(f"CreativeDirector: Calling Groq model: {self.model_id}")
        logger.debug(f"CreativeDirector (System Prompt):\n{system_prompt}")
        logger.debug(f"CreativeDirector (User Message):\n{user_message}")

        import time
        schema_dict = EDLDocument.model_json_schema()
        
        for attempt in range(3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    response_format={"type": "json_object"}
                )
                break
            except Exception as e:
                logger.warning(f"CreativeDirector: Groq API call failed (attempt {attempt + 1}/3): {e}")
                if attempt < 2:
                    time.sleep(2)
                    continue
                raise e
        # Parse LLM response
        try:
            raw_content = response.choices[0].message.content
            logger.debug(f"CreativeDirector (Raw Output):\n{raw_content}")
            edl = json.loads(raw_content)
            timeline = edl.get("timeline", [])
            # Programmatic Hook Enforcement
            if timeline:
                first_item = timeline[0]
                if not isinstance(first_item.get("details"), dict):
                    first_item["details"] = {
                        "visual_cue": "Hook segment",
                        "sound_design": "beat",
                        "pacing_style": "jump-cut",
                        "is_hook": True,
                    }
                else:
                    first_item["details"]["is_hook"] = True
                    logger.info("Hook enforcement: Programmatically forced first timeline item details.is_hook = True.")
        except Exception as e:
            logger.error(f"CreativeDirector: Error parsing response from Groq: {e}")
            return {
                "error": "Failed to generate structured EDL",
                "raw_response": response.choices[0].message.content if response else "No response",
            }

        # Post-process timeline items to ensure virtual subclip ranges are within clip duration
        # Clip format: "filename:start:end" where start/end are seconds relative to original clip
        # If end exceeds the clip's total duration, clamp it to the clip duration
        if isinstance(timeline, list):
            # Build a lookup of clip durations from context
            clip_duration_lookup = {}
            for clip in context.get("available_clips", []):
                clip_name = clip.get("filename")
                duration = clip.get("duration")
                if clip_name and isinstance(duration, (int, float)):
                    clip_duration_lookup[clip_name] = duration
            for item in timeline:
                clip_name = item.get("clip_name", "")
                parts = clip_name.split(":")
                if len(parts) == 3:
                    filename, start_str, end_str = parts
                    try:
                        start = float(start_str)
                        end = float(end_str)
                        max_duration = clip_duration_lookup.get(filename)
                        if max_duration is not None:
                            # Clamp start to clip duration
                            if start > max_duration:
                                start = max_duration
                            # Available duration from start
                            available = max_duration - start
                            if available < 0:
                                available = 0.0
                            # Desired segment length
                            requested_len = end - start
                            # Clip segment length to available duration
                            segment_len = min(requested_len, available)
                            # Compute absolute end timestamp
                            end_abs = start + segment_len
                            # Update clip_name with absolute timestamps
                            item["clip_name"] = f"{filename}:{start:.3f}:{end_abs:.3f}"
                            # Use relative timestamps within the virtual segment
                            item["start_in_clip"] = 0.0
                            item["end_in_clip"] = segment_len
                        else:
                            # No duration info; keep original values
                            item["clip_name"] = f"{filename}:{start:.3f}:{end:.3f}"
                    except ValueError:
                        continue
        # Ensure start is within clip duration and adjust if needed
        for item in timeline:
            clip_name = item.get("clip_name", "")
            parts = clip_name.split(":")
            if len(parts) == 3:
                filename, start_str, end_str = parts
                try:
                    start = float(start_str)
                    end = float(end_str)
                    max_duration = clip_duration_lookup.get(filename)
                    if max_duration is not None:
                        # Clamp start to max_duration
                        if start > max_duration:
                            start = max_duration
                        # Ensure end is not less than start
                        if end < start:
                            end = start
                        # Clamp end to max_duration
                        if end > max_duration:
                            end = max_duration
                        # Update clip_name with clamped values
                        item["clip_name"] = f"{filename}:{start:.3f}:{end:.3f}"
                except ValueError:
                    continue
        # Adjust total timeline duration to meet target_duration if short
        try:
            target_duration = float(context.get("target_duration", 30))
        except Exception:
            target_duration = 30.0
        rendered_duration = sum(
            item.get("timeline_end", 0) - item.get("timeline_start", 0) for item in timeline
        )
        if rendered_duration < target_duration:
            diff = target_duration - rendered_duration
            if timeline:
                last_item = timeline[-1]
                # Extend timeline_end and end_in_clip by diff, respecting clip max duration
                last_item["timeline_end"] = last_item.get("timeline_end", 0) + diff
                # Adjust end_in_clip if possible
                clip_name = last_item.get("clip_name", "")
                parts = clip_name.split(":")
                if len(parts) == 3:
                    filename, start_str, end_str = parts
                    try:
                        end = float(end_str) + diff
                        max_duration = clip_duration_lookup.get(filename)
                        if max_duration is not None and end > max_duration:
                            end = max_duration
                        last_item["end_in_clip"] = end
                        # Update clip_name with new end if changed
                        last_item["clip_name"] = f"{filename}:{start_str}:{end:.3f}"
                    except ValueError:
                        pass
                logger.warning(f"Adjusted timeline to meet target duration: added {diff:.3f}s to last clip.")
        rendered_duration = sum(
            item.get("timeline_end", 0) - item.get("timeline_start", 0) for item in timeline
        )
        if rendered_duration < target_duration:
            remaining = target_duration - rendered_duration
            # Find a clip that can accommodate the remaining duration (max 5s for filler)
            filler_duration = min(remaining, 5.0)
            filler_clip = None
            for fname, dur in clip_duration_lookup.items():
                if dur >= filler_duration:
                    filler_clip = fname
                    break
            if filler_clip is None:
                # fallback to any clip (use first available)
                filler_clip = next(iter(clip_duration_lookup), None)
            if filler_clip:
                # Determine start time within the clip (use 0)
                start_in_clip = 0.0
                end_in_clip = start_in_clip + filler_duration
                # Compute timeline positions
                last_timeline_end = timeline[-1].get("timeline_end", 0) if timeline else 0.0
                filler_item = {
                    "clip_name": f"{filler_clip}:0.000:{end_in_clip:.3f}",
                    "start_in_clip": start_in_clip,
                    "end_in_clip": end_in_clip,
                    "timeline_start": last_timeline_end,
                    "timeline_end": last_timeline_end + filler_duration,
                    "transition": "none",
                    "text_overlay": "",
                    "details": {
                        "visual_cue": "Filler",
                        "sound_design": "",
                        "pacing_style": "jump-cut",
                        "is_hook": False,
                    },
                }
                timeline.append(filler_item)
                logger.warning(
                    f"Added filler clip '{filler_clip}' of {filler_duration:.3f}s to meet target duration."
                )
            else:
                logger.error("Unable to find any clip to use as filler for duration mismatch.")
        # Log final adjustment if any
        if rendered_duration < target_duration:
            logger.warning(
                f"Final rendered duration {rendered_duration:.3f}s still below target {target_duration:.3f}s after filler attempt."
            )


        logger.info(
            f"CreativeDirector: EDL successfully generated! "
            f"Title: '{edl.get('title')}', "
            f"Storyline: '{edl.get('storyline')}', "
            f"Total Duration: {edl.get('total_duration')}s, "
            f"Timeline Clips: {len(timeline)}"
        )
        return edl
if __name__ == "__main__":
    # Small test case if run directly
    director = CreativeDirector()
    logger.info("Director initialized.")

