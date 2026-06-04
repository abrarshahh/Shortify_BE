import os
import json
import logging
import urllib.request
import urllib.error
import socket
from typing import List, Dict, Any
from groq import Groq
from dotenv import load_dotenv
from backend_ai.core.config_loader import AGENTS_CONFIG
from backend_ai.core.api_utils import rate_limit_guard

load_dotenv()
logger = logging.getLogger("agents.director")

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

    def _call_groq(self, messages: List[Dict[str, str]], model_id: str):
        if not self.client:
            raise RuntimeError("Groq client unavailable")
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
        feedback: str = None
    ) -> Dict[str, Any]:
        """
        Generates an Edit Decision List (EDL) by reasoning over audio beats and visual content.
        """
        logger.info(f"CreativeDirector: Starting EDL generation. Prompt: '{user_prompt}'")
        logger.debug(f"CreativeDirector: Aspect Ratio: {aspect_ratio}, Style: {style}, Target Duration: {target_duration}s")
        if feedback:
            logger.warning(f"CreativeDirector: Re-generating EDL due to validation failure. Feedback: {feedback}")
        
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
                "pacing_style": "MUST be exactly one of: speed-ramp, jump-cut, cinematic-slow"
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

        all_models = [self.model_id] + list(self.fallback_models)
        logger.info(f"CreativeDirector: Model order: {all_models}")
        logger.debug(f"CreativeDirector (System Prompt):\n{system_prompt}")
        logger.debug(f"CreativeDirector (User Message):\n{user_message}")

        response = None
        last_error = None

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
            
            logger.info(
                f"CreativeDirector: EDL successfully generated! "
                f"Title: '{edl.get('title')}', "
                f"Storyline: '{edl.get('storyline')}', "
                f"Total Duration: {edl.get('total_duration')}s, "
                f"Timeline Clips: {len(edl.get('timeline', []))}"
            )
            return edl
        except Exception as e:
            logger.error(f"CreativeDirector: Error parsing response from Groq: {e}")
            return {
                "error": "Failed to generate structured EDL",
                "raw_response": response.choices[0].message.content
            }

if __name__ == "__main__":
    # Small test case if run directly
    director = CreativeDirector()
    logger.info("Director initialized.")

