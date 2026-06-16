import os
import json
import logging
import urllib.request
import urllib.error
import socket
from typing import List, Dict, Any, Optional
from groq import Groq
from dotenv import load_dotenv
from backend_ai.core.config_loader import AGENTS_CONFIG
from backend_ai.core.api_utils import rate_limit_guard
from backend_ai.schemas.edl import EDLDocument

load_dotenv()
logger = logging.getLogger("agents.director")

EDL_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "title": {"type": "string"},
        "storyline": {"type": "string"},
        "total_duration": {"type": "number", "minimum": 0.1},
        "music_start_offset": {"type": "number", "minimum": 0.0},
        "timeline": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "clip_name": {"type": "string"},
                    "start_in_clip": {"type": "number", "minimum": 0.0},
                    "end_in_clip": {"type": "number", "minimum": 0.0},
                    "timeline_start": {"type": "number", "minimum": 0.0},
                    "timeline_end": {"type": "number", "minimum": 0.0},
                    "transition": {
                        "type": "string",
                        "enum": [
                            "none", "jump_cut", "crossfade", "dip_to_black",
                            "slide_left", "slide_right", "zoom_in", "zoom_out", "glitch"
                        ]
                    },
                    "text_overlay": {"type": "string"},
                    "details": {
                        "type": "object",
                        "properties": {
                            "visual_cue": {"type": "string"},
                            "sound_design": {"type": "string"},
                            "pacing_style": {
                                "type": "string",
                                "enum": ["speed-ramp", "jump-cut", "cinematic-slow"]
                            },
                            "is_hook": {"type": "boolean"},
                            "keep_original_audio": {"type": "boolean"},
                            "effect_type": {
                                "type": "string",
                                "enum": ["none", "particles", "overlay_blend", "light_leak", "smoke"]
                            },
                            "effect_query": {"type": "string"},
                            "sticker_query": {"type": "string"},
                            "sticker_position": {
                                "type": "string",
                                "enum": ["center", "top-left", "top-right", "bottom-left", "bottom-right", "bottom-center"]
                            }
                        },
                        "required": ["visual_cue", "sound_design", "pacing_style", "is_hook", "keep_original_audio"],
                        "additionalProperties": False
                    }
                },
                "required": [
                    "clip_name", "start_in_clip", "end_in_clip",
                    "timeline_start", "timeline_end", "transition", "text_overlay", "details"
                ],
                "additionalProperties": False
            }
        }
    },
    "required": ["title", "storyline", "total_duration", "music_start_offset", "timeline"],
    "additionalProperties": False
}

class CreativeDirector:
    def __init__(self):
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            logger.warning("GROQ_API_KEY not found; Groq primary model will be skipped")
        
        config = AGENTS_CONFIG.get("creative_director", {})
        self.client = Groq(api_key=api_key) if api_key else None
        self.model_id = config.get("model", "llama-3.3-70b-versatile")
        self.fallback_models = config.get("fallback_models", ["deepseek-r1:8b", "llama3.2-vision:latest"])
        self.ollama_timeout_seconds = int(config.get("ollama_timeout_seconds", 60))

    def _normalize_pacing_style(self, pacing_style: Any) -> str:
        value = str(pacing_style or "jump-cut").strip().lower().replace("_", "-").replace(" ", "-")
        aliases = {
            "fast-cut": "jump-cut",
            "fastcut": "jump-cut",
            "speed-ramp": "speed-ramp",
            "speedramp": "speed-ramp",
            "cinematic-slow": "cinematic-slow",
            "cinematicslow": "cinematic-slow",
            "jump-cut": "jump-cut",
            "jumpcut": "jump-cut",
        }
        return aliases.get(value, "jump-cut")

    def _sanitize_edl(self, edl: Dict[str, Any]) -> Dict[str, Any]:
        timeline = edl.get("timeline", [])
        for item in timeline:
            details = item.get("details") or {}
            details["pacing_style"] = self._normalize_pacing_style(details.get("pacing_style"))
            item["details"] = details
        edl["timeline"] = timeline
        return edl

    def _build_heuristic_edl(
        self,
        user_prompt: str,
        media_analyses: List[Dict[str, Any]],
        target_duration: int,
        style: str,
    ) -> Dict[str, Any]:
        """Deterministic fallback when all LLM calls fail or time out."""
        clips: List[Dict[str, Any]] = []
        for analysis in media_analyses:
            metadata = analysis.get("file_metadata", {})
            filename = metadata.get("filename")
            if not filename:
                continue
            duration = metadata.get("duration_seconds")
            clips.append({
                "filename": filename,
                "duration": float(duration) if duration else 3.0,
                "summary": analysis.get("summary", ""),
            })

        if not clips:
            raise RuntimeError("No usable clips available for heuristic EDL fallback")

        total = float(target_duration)
        timeline = []
        cursor = 0.0
        idx = 0
        while cursor < total - 0.001:
            clip = clips[idx % len(clips)]
            remaining = total - cursor
            segment_duration = min(max(0.5, min(4.0, clip["duration"])), remaining)

            if segment_duration <= 0.5:
                idx += 1
                if idx >= len(clips) and cursor == 0.0:
                    break
                continue

            timeline.append({
                "clip_name": clip["filename"],
                "start_in_clip": 0.0,
                "end_in_clip": round(float(segment_duration), 3),
                "timeline_start": round(float(cursor), 3),
                "timeline_end": round(float(cursor + segment_duration), 3),
                "transition": "jump_cut" if idx > 0 else "none",
                "text_overlay": "",
                "details": {
                    "visual_cue": clip.get("summary") or "Key visual moment",
                    "sound_design": "",
                    "pacing_style": "jump-cut",
                    "is_hook": False,
                    "keep_original_audio": True,
                    "effect_type": "none",
                    "effect_query": "",
                    "sticker_query": "",
                    "sticker_position": "bottom-center"
                },
            })
            cursor += segment_duration
            idx += 1

        if not timeline:
            raise RuntimeError("Heuristic EDL fallback could not build any timeline items")

        return self._sanitize_edl({
            "title": f"{style.title()} Reel",
            "storyline": user_prompt[:120] or "A fast-cut highlight reel",
            "total_duration": round(total, 3),
            "music_start_offset": 0.0,
            "timeline": timeline,
        })

    def _call_groq(self, messages: List[Dict[str, str]], model_id: str, use_schema: bool = True):
        if not self.client:
            raise RuntimeError("Groq client unavailable")
        if use_schema:
            try:
                return self.client.chat.completions.create(
                    model=model_id,
                    messages=messages,
                    response_format={
                        "type": "json_schema",
                        "json_schema": {
                            "name": "edl_document",
                            "schema": EDL_JSON_SCHEMA
                        }
                    }
                )
            except Exception as e:
                logger.warning(f"CreativeDirector: Groq model '{model_id}' failed with json_schema: {e}. Falling back to json_object.")
        
        return self.client.chat.completions.create(
            model=model_id,
            messages=messages,
            response_format={"type": "json_object"}
        )

    def _call_ollama(self, model_id: str, messages: List[Dict[str, str]]):
        payload = {
            "model": model_id,
            "messages": messages,
            "stream": False,
            "format": "json",
        }
        req = urllib.request.Request(
            "http://localhost:11434/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.ollama_timeout_seconds) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, socket.timeout) as e:
            raise RuntimeError(f"Ollama unavailable for model '{model_id}': {e}") from e

        if "message" not in body or "content" not in body["message"]:
            raise RuntimeError(f"Unexpected Ollama response for model '{model_id}': {body}")

        class _Response:
            def __init__(self, content: str):
                self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]

        return _Response(body["message"]["content"])

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
        pre_flight_report: Optional[Dict[str, Any]] = None,
        clip_scores: Optional[Dict[str, Any]] = None
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
            
            cs_info = {}
            if clip_scores and filename in clip_scores:
                cs_info = clip_scores[filename]
                
            clip_info = {
                "filename": filename,
                "duration": analysis.get("file_metadata", {}).get("duration_seconds"),
                "quality_score": cs_info.get("composite_score", q_info["quality_score"]),
                "avg_sharpness": cs_info.get("sharpness", q_info["avg_sharpness"]),
                "avg_brightness": cs_info.get("exposure_score", q_info["avg_brightness"]),
                "motion_score": cs_info.get("motion_score", 0.0),
                "motion_tier": cs_info.get("motion_tier", "medium"),
                "face_detected": cs_info.get("face_detected", False),
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
                "text_overlay": "On-screen text (decide dynamically based on user prompt/style; leave empty if not appropriate)",
              "details": {{
                "visual_cue": "Specific action to focus on",
                "sound_design": "SFX (e.g., 'whoosh', 'bass drop')",
                "pacing_style": "MUST be exactly one of: speed-ramp, jump-cut, cinematic-slow",
                "is_hook": boolean,
                "keep_original_audio": boolean,
                "effect_type": "none | particles | overlay_blend | light_leak | smoke",
                "effect_query": "specific search query (decide dynamically based on user prompt/style, like 'lens flare', 'bokeh', 'smoke', 'fog', or empty string)",
                "sticker_query": "specific search query for Giphy (decide dynamically based on user prompt/style, like 'subscribe', 'fire', 'arrow', or empty string)",
                "sticker_position": "center | top-left | top-right | bottom-left | bottom-right | bottom-center"
              }}
            }}
          ]
        }}
        
        RULES:
        - QUALITY PREFERENCE: Quality score is objective. Prefer clips with quality_score above 0.70 unless narrative requires otherwise. Never use clips with quality_score below 0.40 as the hook.
        - PACING CONSTRAINTS: Do not assign cinematic-slow pacing to high-motion clips (motion_tier == 'high'). Do not assign jump-cut pacing to static clips (motion_tier == 'static') unless the total reel style is speed-ramp or fast_cut.
        - ORIGINAL AUDIO & DUCKING: Set 'keep_original_audio' to true if the clip contains dialogue, voices, or speaking parts that should be heard. This will play the clip's audio and duck the background music. Set it to false for clips that only contain background ambient noise, wind, silence, or where the raw audio is not useful. This will mute the clip's raw audio and keep the background music playing at full volume.
        - SEQUENTIAL ORDER: 'timeline_start' must strictly increase. No overlapping clips.
        - NO REPETITION: Avoid using the same clip twice in a row. Use different clips to maintain visual interest.
        - USER INTENT IS SUPREME: You MUST follow the User Intent perfectly. If the user asks for a specific scene, action, or chronological order, you must provide it exactly as requested, even if the priority_score is lower.
        - SMART CLIP SELECTION (STRICT CASCADE LOGIC): You MUST select clips using this exact priority sequence:
          1. FIRST, only look at segments from the 'highlights' array.
          2. FILTER by 'should_be_used': prioritize segments where 'should_be_used' is true, and further prioritize segments with higher 'local_score' (scored by our local ClipScoringAgent).
          3. MATCH PACING STYLE: You MUST match the 'pacing_style' under 'details' based on the clip segment's 'motion_type' metric. High-motion segments ('motion_type' == 'high-motion' or motion_tier == 'high') MUST be assigned to 'speed-ramp' pacing; static segments ('motion_type' == 'static' or motion_tier == 'static') MUST be assigned to 'cinematic-slow' pacing.
          4. MATCH FOCUS: Ensure that the 'segment_focus' is consistent across the selected clips (e.g., if the main theme is 'mountain', try to pick other 'mountain' clips).
          5. If you still need more duration to hit the target, fall back to the 'segments' array and apply the exact same logic (highest priority_score, should_be_used: true, matching focus).
        - EXACT DURATION REQUIRED: The total expected render duration (sum of effective clip durations) MUST equal {target_duration} seconds. For each clip, its effective duration is: (end_in_clip - start_in_clip) / 1.5 if details.pacing_style is 'speed-ramp', and (end_in_clip - start_in_clip) otherwise. The sum of these effective durations must be exactly {target_duration} seconds.
        - DURATION MATCH: Ensure 'timeline_end - timeline_start' is exactly equal to 'end_in_clip - start_in_clip' for each clip.
        - MUSIC SELECTION: Use 'music_start_offset' to pick a good starting point from the audio track (e.g., an energy segment).
        - DYNAMIC ELEMENTS & STYLES SELECTION: You MUST select and tailor transitions, pacing_style, text_overlay, effect_type, effect_query, sticker_query, and sticker_position dynamically based on the user's prompt and intention. Do not use generic defaults if the prompt indicates a specific vibe (e.g. if they ask for a 'scary video', use 'smoke'/'fog' effects, 'ghost'/'scared' stickers, and glitch/dip_to_black transitions; if a 'hype/gym video', use 'particles'/'sparks' effects, 'fire'/'subscribe' stickers, and fast_cut/speed-ramp pacing).
        - TEXT OVERLAYS: You, as the viral video analyst, must decide which key moments or scene transitions would benefit from on-screen text overlays (title cards, main hooks, call-to-actions, or section headers) based on the user's prompt theme. For each timeline clip, determine if a short, punchy text overlay (1-4 words) fits the scene, and if so, write it in the 'text_overlay' field. Do not include subtitles of dialogue here (subtitles are processed separately). Use text overlays strategically to capture and keep viewer attention.
        - STICKERS & VISUAL EFFECTS: Determine if a clip would benefit from a visual effect loop or an animated transparent sticker based on the user's prompt and theme:
          - If a sticker is useful (e.g. arrows to draw focus, subscribe button, expressions like 'fire', 'wow', 'laugh'), set 'sticker_query' to a 1-2 word query reflecting the user's prompt theme and pick its spatial 'sticker_position'.
          - If a clip needs an aesthetic layer (like floating dust, sparks, light leaks, bokeh, fog, smoke), set 'effect_type' and provide a search term in 'effect_query' reflecting the user's prompt theme.
          - Set 'effect_type' to 'none' and leave queries empty ("") if they are not needed for a clip.
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

        all_models = [self.model_id] + list(self.fallback_models)
        logger.info(f"CreativeDirector: Model order: {all_models}")
        logger.debug(f"CreativeDirector (System Prompt):\n{system_prompt}")
        logger.debug(f"CreativeDirector (User Message):\n{user_message}")

        import time

        schema_dict = EDLDocument.model_json_schema()

        response = None
        last_error = None

        for attempt in range(3):
            for model_id in all_models:
                try:
                    if model_id == self.model_id:
                        logger.info(f"CreativeDirector: Calling Groq model: {model_id}")
                        response = self._call_groq(messages, model_id)
                    else:
                        logger.info(
                            f"CreativeDirector: Falling back to Ollama model: {model_id} "
                            f"(timeout={self.ollama_timeout_seconds}s)"
                        )
                        response = self._call_ollama(model_id, messages)
                    break
                except Exception as e:
                    last_error = e
                    if "timed out" in str(e).lower() or isinstance(e, TimeoutError):
                        logger.warning(f"CreativeDirector: Model '{model_id}' timed out after {self.ollama_timeout_seconds}s")
                else:
                    logger.warning(f"CreativeDirector: Model '{model_id}' failed: {e}")
                continue

        if response is None:
            logger.warning(f"CreativeDirector: All models failed, using heuristic fallback EDL: {last_error}")
            return self._build_heuristic_edl(user_prompt, media_analyses, target_duration, style)

        try:
            raw_content = response.choices[0].message.content
            logger.debug(f"CreativeDirector (Raw Output):\n{raw_content}")
            edl = self._sanitize_edl(json.loads(raw_content))
            
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
                        "keep_original_audio": True,
                    }
                else:
                    first_item["details"]["is_hook"] = True
                    if "keep_original_audio" not in first_item["details"]:
                        first_item["details"]["keep_original_audio"] = True
                    logger.info("Hook enforcement: Programmatically forced first timeline item details.is_hook = True and keep_original_audio = True.")

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

