import os
import json
import re
import logging
import urllib.request
import urllib.error
import socket
from typing import List, Dict, Any, Optional
from backend_ai.core.api_utils import get_gemini_client
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
        "global_color_grade": {
            "type": "object",
            "properties": {
                "brightness": {"type": "number", "minimum": 0.5, "maximum": 1.8},
                "contrast": {"type": "number", "minimum": 0.5, "maximum": 2.0},
                "gamma": {"type": "number", "minimum": 0.1, "maximum": 10.0},
                "saturation": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                "vibrance": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                "hue": {"type": "number", "minimum": -180.0, "maximum": 180.0},
                "temperature": {"type": "number", "minimum": -50.0, "maximum": 50.0},
                "vignette_strength": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                "vignette_radius": {"type": "number", "minimum": 0.0, "maximum": 2.0}
            },
            "required": ["brightness", "contrast", "gamma", "saturation", "vibrance", "hue", "temperature", "vignette_strength", "vignette_radius"],
            "additionalProperties": False
        },
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
                            "none", "jump_cut", "fade", "crossfade", "dip_to_black", "fade_to_white",
                            "slide_left", "slide_right", "slide_up", "slide_down", "slide_push",
                            "wipe_left", "wipe_right", "wipe_up", "wipe_down",
                            "wipe_diagonal_tl", "wipe_diagonal_tr", "wipe_diagonal_bl", "wipe_diagonal_br",
                            "split_horizontal", "split_vertical", "iris", "iris_circle",
                            "diamond", "heart", "blinds_horizontal", "blinds_vertical",
                            "checkerboard", "clock_wipe", "zoom_in", "zoom_out", "glitch",
                            "pixelate", "spin", "ripple", "blur", "light_leak"
                        ]
                    },
                    "transition_params": {
                        "type": "object",
                        "additionalProperties": True
                    },
                    "color_grade": {
                        "type": "object",
                        "properties": {
                            "brightness": {"type": "number", "minimum": 0.5, "maximum": 1.8},
                            "contrast": {"type": "number", "minimum": 0.5, "maximum": 2.0},
                            "gamma": {"type": "number", "minimum": 0.1, "maximum": 10.0},
                            "saturation": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                            "vibrance": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                            "hue": {"type": "number", "minimum": -180.0, "maximum": 180.0},
                            "temperature": {"type": "number", "minimum": -50.0, "maximum": 50.0},
                            "vignette_strength": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "vignette_radius": {"type": "number", "minimum": 0.0, "maximum": 2.0}
                        },
                        "required": ["brightness", "contrast", "gamma", "saturation", "vibrance", "hue", "temperature", "vignette_strength", "vignette_radius"],
                        "additionalProperties": False
                    },
                    "audio_ducking": {
                        "type": "object",
                        "properties": {
                            "original_audio_volume": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                            "music_volume_during_segment": {"type": "number", "minimum": 0.0, "maximum": 1.0}
                        },
                        "required": ["original_audio_volume", "music_volume_during_segment"],
                        "additionalProperties": False
                    },
                    "speed_preset": {
                        "type": "string",
                        "enum": ["constant_fast", "constant_slow", "ramp_up", "ramp_down", "speed_bump", "freeze_frame"]
                    },
                    "speed_keyframes": {
                        "type": "array",
                        "items": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 2,
                            "maxItems": 2
                        }
                    },
                    "reverse": {"type": "boolean"},
                    "stabilize": {"type": "boolean"},
                    "stabilize_strength": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "text_preset": {
                        "type": "string",
                        "enum": ["bold_hype", "classic_clean", "neon_glow", "minimal_pop", "none"]
                    },
                    "text_animation": {
                        "type": "string",
                        "enum": ["none", "fade", "slide_up", "slide_down", "slide_left", "slide_right"]
                    },
                    "sticker_animation": {
                        "type": "string",
                        "enum": ["none", "fade", "slide_up", "slide_down", "slide_left", "slide_right"]
                    },
                    "text_overlay": {"type": "string"},
                    "clip_effect": {
                        "type": "object",
                        "properties": {
                            "effect_type": {
                                "type": "string",
                                "enum": ["none", "blur", "pixelate", "vignette", "glitch", "mirror"]
                            },
                            "parameters": {
                                "type": "object",
                                "additionalProperties": True
                            }
                        },
                        "required": ["effect_type"],
                        "additionalProperties": False
                    },
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
                            },
                            "effect_asset_id": {
                                "type": "string",
                                "enum": ["", "overlay_film_grain", "overlay_light_leak", "overlay_particles", "overlay_smoke"]
                            },
                            "sticker_asset_id": {
                                "type": "string",
                                "enum": ["", "sticker_subscribe", "sticker_arrow", "sticker_fire"]
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
        config = AGENTS_CONFIG.get("creative_director", {})
        self.client = None
        self.gemini_client = get_gemini_client()
        self.model_id = config.get("model", "gemini-2.5-flash")
        self.fallback_models = config.get("fallback_models", ["gemini-2.5-pro", "gemini-1.5-pro", "gemini-1.5-flash"])
        self.gemini_model = self.model_id
        self.ollama_timeout_seconds = int(config.get("ollama_timeout_seconds", 60))

    def _clean_profanity(self, text: str) -> str:
        if not isinstance(text, str):
            return text
        profanities = {
            r"\bfuck(ing|er|ed|s)?\b": "[expletive]",
            r"\bshit(s|ted|ting|head)?\b": "[expletive]",
            r"\bass(hole)?s?\b": "[expletive]",
            r"\bbitch(es)?\b": "[expletive]",
            r"\bcrap\b": "[expletive]",
            r"\bdamn\b": "[expletive]",
        }
        cleaned = text
        for pattern, replacement in profanities.items():
            cleaned = re.sub(pattern, replacement, cleaned, flags=re.IGNORECASE)
        return cleaned

    def _clean_nested_profanity(self, data: Any) -> Any:
        if isinstance(data, str):
            return self._clean_profanity(data)
        elif isinstance(data, dict):
            return {k: self._clean_nested_profanity(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._clean_nested_profanity(x) for x in data]
        return data

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

    def _sanitize_edl(self, edl: Dict[str, Any], media_analyses: List[Dict[str, Any]]) -> Dict[str, Any]:
        durations = {}
        for analysis in media_analyses:
            filename = analysis.get("file_metadata", {}).get("filename")
            dur = analysis.get("file_metadata", {}).get("duration_seconds")
            if filename and dur is not None:
                durations[filename] = float(dur)

        timeline = edl.get("timeline", [])
        for item in timeline:
            details = item.get("details") or {}
            details["pacing_style"] = self._normalize_pacing_style(details.get("pacing_style"))
            item["details"] = details
            
            # Virtual segment duration & bounds auto-correction
            clip_name = item.get("clip_name", "")
            if ":" in clip_name:
                parts = clip_name.split(":")
                if len(parts) == 3:
                    filename = parts[0]
                    try:
                        v_start = float(parts[1])
                        v_end = float(parts[2])
                        
                        actual_dur = durations.get(filename)
                        if actual_dur is not None:
                            if v_end > actual_dur:
                                v_end = actual_dur
                                item["clip_name"] = f"{filename}:{v_start:.3f}:{v_end:.3f}"
                        
                        v_len = v_end - v_start
                        
                        start_in = float(item.get("start_in_clip", 0.0))
                        end_in = float(item.get("end_in_clip", 0.0))
                        
                        if start_in >= v_start:
                            start_in = start_in - v_start
                        
                        if end_in > v_len:
                            end_in = end_in - v_start
                            
                        start_in = max(0.0, min(start_in, v_len))
                        end_in = max(start_in + 0.1, min(end_in, v_len))
                        
                        item["start_in_clip"] = round(start_in, 3)
                        item["end_in_clip"] = round(end_in, 3)
                    except ValueError:
                        pass

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
        }, media_analyses)

    def _call_groq(self, messages: List[Dict[str, str]], model_id: str, use_schema: bool = True):
        if not self.client:
            raise RuntimeError("Groq client unavailable")
        
        # Certain models on Groq (like llama-3.3, mixtral) do not support json_schema
        if any(m in model_id.lower() for m in ["llama-3.3", "mixtral"]):
            use_schema = False

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

    def _call_gemini(self, model_id: str, messages: List[Dict[str, str]]):
        if not self.gemini_client:
            raise RuntimeError("Gemini client unavailable")
        
        system_instructions = ""
        user_content = ""
        for msg in messages:
            if msg["role"] == "system":
                system_instructions += msg["content"] + "\n"
            elif msg["role"] == "user":
                user_content += msg["content"] + "\n"

        from google.genai import types
        response = self.gemini_client.models.generate_content(
            model=model_id,
            contents=user_content,
            config=types.GenerateContentConfig(
                system_instruction=system_instructions,
                response_mime_type="application/json"
            )
        )
        
        class _Response:
            def __init__(self, content: str):
                self.choices = [type("Choice", (), {"message": type("Message", (), {"content": content})()})()]

        return _Response(response.text)

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

        # Compute total input duration across all available assets
        total_input_duration = 0.0
        for analysis in media_analyses:
            dur = analysis.get("file_metadata", {}).get("duration_seconds")
            if isinstance(dur, (int, float)):
                total_input_duration += dur

        # Filter beats and drops to only those within the target duration + 5s.
        # This keeps the rhythm context relevant to the edit and saves hundreds of tokens.
        beats_filtered = [b for b in audio_analysis.get("beat_times", []) if b <= target_duration + 5.0]
        drops_filtered = [d for d in audio_analysis.get("peak_times", []) if d <= target_duration + 5.0]

        # Prepare the context for the LLM
        context = {
            "user_intent": user_prompt,
            "target_duration": target_duration,
            "total_input_duration": round(total_input_duration, 2),
            "aspect_ratio": aspect_ratio,
            "style": style,
            "audio_rhythm": {
                "tempo": audio_analysis.get("tempo"),
                "beats": beats_filtered, 
                "drops": drops_filtered,
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

            # Sort highlights and segments by priority_score DESC
            highlights_sorted = sorted(
                analysis.get("interesting_segments", []),
                key=lambda s: s.get("priority_score", 0),
                reverse=True
            )
            segments_sorted = sorted(
                analysis.get("all_segments", []),
                key=lambda s: s.get("priority_score", 0),
                reverse=True
            )

            # Truncate summary to save tokens
            summary_raw = analysis.get("summary", "")
            summary_clean = summary_raw[:120] + "..." if len(summary_raw) > 120 else summary_raw

            # Simplify and limit highlights (top 4 DESC) to avoid rate limits
            simplified_highlights = []
            for seg in highlights_sorted[:4]:
                desc = seg.get("description", "")
                desc_clean = desc[:100] + "..." if len(desc) > 100 else desc
                simplified_highlights.append({
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "description": desc_clean,
                    "priority_score": seg.get("priority_score"),
                    "relevance_score": seg.get("relevance_score"),
                    "segment_focus": seg.get("segment_focus")
                })

            # Simplify and limit chronological segments (top 3 DESC) to avoid rate limits
            simplified_segments = []
            for seg in segments_sorted[:3]:
                desc = seg.get("description", "")
                desc_clean = desc[:100] + "..." if len(desc) > 100 else desc
                simplified_segments.append({
                    "start": seg.get("start"),
                    "end": seg.get("end"),
                    "description": desc_clean,
                    "priority_score": seg.get("priority_score"),
                    "relevance_score": seg.get("relevance_score"),
                    "segment_focus": seg.get("segment_focus")
                })

            # Simplify audio context - remove huge raw captions arrays
            audio_raw = analysis.get("audio", {})
            captions_list = audio_raw.get("captions", [])
            audio_info = {
                "has_dialogue": len(captions_list) > 0,
                "audio_mood": audio_raw.get("audio_mood", "Neutral")
            }

            clip_info = {
                "filename": filename,
                "duration": analysis.get("file_metadata", {}).get("duration_seconds"),
                "quality_score": cs_info.get("composite_score", q_info["quality_score"]),
                "avg_sharpness": cs_info.get("sharpness", q_info["avg_sharpness"]),
                "avg_brightness": cs_info.get("exposure_score", q_info["avg_brightness"]),
                "motion_score": cs_info.get("motion_score", 0.0),
                "motion_tier": cs_info.get("motion_tier", "medium"),
                "face_detected": cs_info.get("face_detected", False),
                "summary": summary_clean,
                "highlights": simplified_highlights,
                "segments": simplified_segments,
                "audio": audio_info
            }
            context["available_clips"].append(clip_info)

        system_prompt = f"""
        You are a top-tier Social Media Influencer and Viral Content Director. 
        Your goal is to create a high-retention Edit Decision List (EDL) for a TikTok/Reel that tells a compelling story.
        
        GOALS:
        1. INFLUENCER MINDSET: The first 1.5 - 2 seconds MUST be a high-energy "hook" (is_hook: true).
        2. STORYLINE: Create a clear, sequential narrative arc (e.g., Setup -> Conflict -> Resolution).
        3. DURATION: The total video duration MUST be approximately {target_duration} seconds.
        4. STYLE: Strictly follow the '{style}' style rules and editing grammar:
           - 'mrbeast': Ultra-high-intensity pacing. Snappy jump-cuts every 1.5s–2s. Uses zoom-ins/zooms on peak moments. Triggers SFX ('whoosh' / 'bass_drop') and stickers ('sticker_fire' / 'sticker_subscribe') for visual pop.
           - 'ali_abdaal': Minimalistic and clean. Pacing is slow (3s–5s clips). Uses smooth crossfade transitions. Subtitles/text overlays should be classic_clean. Uses warm cinematic grading and subtle overlays ('overlay_film_grain').
           - 'alex_hormozi': High-energy narrative. Snappy jump-cuts with bold neon_glow text overlays. High frequency of highlights, arrows ('sticker_arrow') pointing at elements, and sharp audio ducking.
           - 'travel': Cinematic pacing. Uses smooth slide transitions ('slide_left'/'slide_right'). Warm, highly-saturated color grades, and warm light leaks ('overlay_light_leak') or ambient dust ('overlay_particles').
        
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
                "transition": "none | jump_cut | fade | crossfade | dip_to_black | fade_to_white | slide_left | slide_right | slide_up | slide_down | slide_push | wipe_left | wipe_right | wipe_up | wipe_down | wipe_diagonal_tl | wipe_diagonal_tr | wipe_diagonal_bl | wipe_diagonal_br | split_horizontal | split_vertical | iris | iris_circle | diamond | heart | blinds_horizontal | blinds_vertical | checkerboard | clock_wipe | zoom_in | zoom_out | glitch | pixelate | spin | ripple | blur | light_leak",
              "transition_params": {{
                "num_bars": 15,
                "intensity": 1.5
              }},
               "color_grade": {{
                 "brightness": 1.1,
                 "contrast": 1.05,
                 "saturation": 0.9,
                 "temperature": 10.0,
                 "vignette_strength": 0.3
               }},
               "audio_ducking": {{
                 "original_audio_volume": 0.8,
                 "music_volume_during_segment": 0.1
               }},
              "speed_preset": "ramp_up",
              "reverse": false,
              "stabilize": true,
              "text_preset": "bold_hype",
              "text_animation": "slide_up",
              "sticker_animation": "fade",
              "text_overlay": "On-screen text (decide dynamically based on user prompt/style; leave empty if not appropriate)",
              "details": {{
                "visual_cue": "Specific action to focus on",
                "sound_design": "SFX (e.g., 'whoosh', 'bass drop')",
                "pacing_style": "MUST be exactly one of: speed-ramp, jump-cut, cinematic-slow",
                "is_hook": boolean,
                "keep_original_audio": boolean,
                "effect_type": "none | particles | overlay_blend | light_leak | smoke",
                "effect_query": "legacy field (always set to empty string \"\")",
                "sticker_query": "legacy field (always set to empty string \"\")",
                "sticker_position": "center | top-left | top-right | bottom-left | bottom-right | bottom-center"
              }}
            }}
          ]
        }}
        
        RULES:
        - QUALITY PREFERENCE: Quality score is objective. Prefer clips with quality_score above 0.70 unless narrative requires otherwise. Never use clips with quality_score below 0.40 as the hook.
        - FACE/PEOPLE PRIORITY: If user_intent contains first-person words ('me', 'us', 'we', 'I', 'myself', 'ourselves') or people references ('person', 'people', 'creator', 'athlete', 'speaker', 'friend', 'team', or any person's name):
          - Clips where face_detected=True MUST be selected FIRST for every timeline slot.
          - Clips where face_detected=False (pure B-roll, landscapes, objects) may ONLY be used when no face clip remains unused, or as a brief (max 2s) establishing/cutaway shot between face clips.
          - NEVER open the video (hook slot, is_hook=true) with a clip where face_detected=False if ANY face-detected clip is available.
          - When multiple face clips exist, rank them by relevance_score first, then by quality_score.
        - INTELLIGENT REUSE POLICY: Decide whether to reuse clip segments dynamically based on available input footage (`total_input_duration`) versus the requested `target_duration`, the prompt, and overall video vibe.
          - If the total combined duration of all available unique clips is close to or less than the target duration, you should reuse or loop high-quality clips/segments to meet the time requirement.
          - Even when you have plenty of unique footage (e.g., 60s of footage for a 30s target video), you should STILL choose to reuse highly interesting, highly relevant segments (e.g., priority_score >= 8.5) if the remaining unused segments are uninteresting, low-scoring (priority_score < 7.0), or irrelevant. Never force yourself to use boring, low-scoring B-roll just for the sake of uniqueness.
          - Minimize reuse of mediocre clips. If the prompt suggests a transition flow that repeats (e.g., flashing back to the hook or matching a repeated beat drop), reuse is encouraged.
          - Avoid reusing identical segments consecutively unless aiming for a specific visual repeat/loop effect.
        - ALWAYS USE HIGHEST PRIORITY FIRST: Within each clip's 'highlights' list, segments are sorted by priority_score from highest to lowest. Always start from the top of this list. Never pick a segment with a lower priority_score if a higher-priority segment from the same clip has not been used yet.
        - PACING CONSTRAINTS: Do not assign cinematic-slow pacing to high-motion clips (motion_tier == 'high'). Do not assign jump-cut pacing to static clips (motion_tier == 'static') unless the total reel style is speed-ramp or fast_cut.
        - ORIGINAL AUDIO & DUCKING (INTELLIGENT): You MUST determine the audio levels for each clip segment. Provide an "audio_ducking" object containing "original_audio_volume" (0.0 to 1.0) and "music_volume_during_segment" (0.0 to 1.0) for each timeline item:
          - If a segment contains dialogue, voices, or spoken words (which you can detect from the clip's "audio" captions context), set 'keep_original_audio' to true, set "original_audio_volume" high (0.8 to 1.0), and duck the background music volume "music_volume_during_segment" low (0.05 to 0.10).
          - If a segment is non-dialogue B-roll or contains only atmospheric noise (wind, rain, etc.) that you want to mix with the music, set 'keep_original_audio' to true, set "original_audio_volume" high (0.8 to 1.0) so it's clearly heard, and set "music_volume_during_segment" to a normal level (0.20 to 0.25).
          - If the segment has no useful audio, set 'keep_original_audio' to false, set "original_audio_volume" to 0.0, and set "music_volume_during_segment" to a normal level (0.20 to 0.25).
        - RELEVANCE PREFERENCE (STRICT): Each segment in 'highlights' and 'segments' has a 'relevance_score' between 0.0 and 1.0 indicating how well it matches the user's prompt. You MUST weight relevance_score over energy_score and priority_score. A clip segment that is highly relevant to the creative prompt (relevance_score >= 0.70) MUST be prioritized over an irrelevant segment, even if the irrelevant segment has a higher energy_score.
        - SEQUENTIAL ORDER: 'timeline_start' must strictly increase. No overlapping clips.
        - NO REPETITION: Avoid using the same clip twice in a row. Use different clips to maintain visual interest.
        - USER INTENT IS SUPREME: You MUST follow the User Intent perfectly. If the user asks for a specific scene, action, or chronological order, you must provide it exactly as requested, prioritizing matching segments (highest relevance_score).
        - CHRONOLOGICAL NARRATIVE SEQUENCING (CRITICAL): If the user's prompt or brief specifies a sequence of events (e.g., ascending/climbing -> struggling -> reaching the destination/lake -> joy), you MUST sequence the selected timeline clips to form this exact chronological progression. Do not mix up the order of narrative stages.
        - SMART CLIP SELECTION (STRICT CASCADE LOGIC): You MUST select clips using this exact priority sequence:
          1. FIRST, only look at segments from the 'highlights' array.
          2. FILTER by prompt relevance: prioritize segments where 'relevance_score' is high (>= 0.70), and further prioritize segments with higher 'local_score' (scored by our local ClipScoringAgent).
          3. MATCH PACING STYLE: You MUST match the 'pacing_style' under 'details' based on the clip segment's 'motion_type' metric. High-motion segments ('motion_type' == 'high-motion' or motion_tier == 'high') MUST be assigned to 'speed-ramp' pacing; static segments ('motion_type' == 'static' or motion_tier == 'static') MUST be assigned to 'cinematic-slow' pacing.
          4. MATCH FOCUS: Ensure that the 'segment_focus' is consistent across the selected clips (e.g., if the main theme is 'mountain', try to pick other 'mountain' clips).
          5. If you still need more duration to hit the target, fall back to the 'segments' array and apply the exact same logic (highest relevance_score, then highest priority_score/local_score, matching focus).
        - EXACT DURATION REQUIRED: The total expected render duration (sum of effective clip durations) MUST equal {target_duration} seconds.
          For each clip, its speed is determined as:
          - If `speed_preset` is set:
            - "constant_fast" -> speed is 2.0 (effective duration = (end_in_clip - start_in_clip) / 2.0)
            - "ramp_up" -> speed is 1.13 (effective duration = (end_in_clip - start_in_clip) / 1.13)
            - "ramp_down" -> speed is 1.13 (effective duration = (end_in_clip - start_in_clip) / 1.13)
            - "speed_bump" -> speed is 1.2 (effective duration = (end_in_clip - start_in_clip) / 1.2)
            - "constant_slow" -> speed is 1.0 (effective duration = end_in_clip - start_in_clip)
            - "freeze_frame" -> speed is 1.0 (effective duration = end_in_clip - start_in_clip)
          - If `speed_preset` is NOT set:
            - If details.pacing_style is 'speed-ramp' -> speed is 1.5 (effective duration = (end_in_clip - start_in_clip) / 1.5)
            - Otherwise -> speed is 1.0 (effective duration = end_in_clip - start_in_clip)
          The sum of these effective durations must be exactly {target_duration} seconds.
          PREFER PACING STYLE OVER PRESETS: Unless the user prompt explicitly requests custom speed presets like constant fast/slow or ramping, do NOT set `speed_preset` or `speed_keyframes` (leave them empty or null). Just set details.pacing_style to 'speed-ramp' or 'cinematic-slow' and let the pacing speeds apply automatically. This keeps the duration calculations simple.
        - DURATION MATCH: Ensure 'timeline_end - timeline_start' is exactly equal to the effective duration: '(end_in_clip - start_in_clip) / speed' for each clip.
        - MUSIC SELECTION: Use 'music_start_offset' to pick a good starting point from the audio track (e.g., an energy segment).
        - DYNAMIC ELEMENTS & STYLES SELECTION: You MUST select and tailor transitions, pacing_style, text_overlay, effect_type, effect_query, sticker_query, and sticker_position dynamically based on the user's prompt and intention. Do not use generic defaults if the prompt indicates a specific vibe (e.g. if they ask for a 'scary video', use 'smoke'/'fog' effects, 'ghost'/'scared' stickers, and glitch/dip_to_black transitions; if a 'hype/gym video', use 'particles'/'sparks' effects, 'fire'/'subscribe' stickers, and fast_cut/speed-ramp pacing).
        - TEXT OVERLAYS: You, as the viral video analyst, must decide which key moments or scene transitions would benefit from on-screen text overlays (title cards, main hooks, call-to-actions, or section headers) based on the user's prompt theme. For each timeline clip, determine if a short, punchy text overlay (1-4 words) fits the scene, and if so, write it in the 'text_overlay' field. Do not include subtitles of dialogue here (subtitles are processed separately). Use text overlays strategically to capture and keep viewer attention.
        - STICKERS & VISUAL EFFECTS: Determine if a clip would benefit from a visual effect loop or an animated transparent sticker based on the user's prompt and theme:
          - If a sticker is useful (e.g. arrows to draw focus, subscribe button, expressions like 'fire', 'wow'), select the most appropriate 'sticker_asset_id' from: 'sticker_subscribe' (for calls to action), 'sticker_arrow' (to highlight/point), or 'sticker_fire' (for high-energy/hype). If none matches, leave it empty (""). Set its spatial 'sticker_position' accordingly.
          - If a clip needs an aesthetic layer, select the most appropriate 'effect_asset_id' from: 'overlay_film_grain' (for cinematic retro feel), 'overlay_light_leak' (for warm flares), 'overlay_particles' (for dust/sparks), or 'overlay_smoke' (for foggy/moody vibe). If none matches, leave it empty ("").
          - Always set both 'sticker_query' and 'effect_query' to empty strings (""), and set 'effect_type' to 'none', as they are legacy fields replaced by local Asset IDs.
        - TRANSITIONS: Choose transitions deliberately based on style and narrative moment:
          - 'none' or 'jump_cut': default for fast_cut style, high energy sequences, beat-sync cuts. Never use crossfade on beat-sync cuts.
          - 'crossfade' or 'fade': smooth scene changes, travel style, when two clips share similar mood or color.
          - 'dip_to_black' or 'fade_to_white': dramatic scene breaks, time jumps, emotional pauses, cinematic style chapter markers.
          - 'zoom_in' or 'zoom_out': push into or pull back from an action moment.
          - 'slide_left', 'slide_right', 'slide_up', 'slide_down', or 'slide_push': dynamic panning slides.
          - 'wipe_left', 'wipe_right', 'wipe_up', 'wipe_down', 'wipe_diagonal_tl', 'wipe_diagonal_tr', 'wipe_diagonal_bl', or 'wipe_diagonal_br': screen wipes.
          - 'split_horizontal' or 'split_vertical': split-screen reveals opening from the edges.
          - 'iris', 'iris_circle', 'diamond', or 'heart': shaped reveals.
          - 'blinds_horizontal', 'blinds_vertical', 'checkerboard', or 'clock_wipe': grid-based/radial reveals.
          - 'glitch', 'pixelate', 'spin', 'ripple', 'blur', or 'light_leak': distortion/blur effects.
          - PARAMETERS: You can dynamically tune transition variants using `"transition_params"`. For example:
            - blinds: `{{"num_bars": int}}`
            - checkerboard: `{{"grid_size": int}}`
            - glitch: `{{"intensity": float}}`
            - pixelate: `{{"max_cell_size": int}}`
            - blur: `{{"max_blur_size": int}}`
            - light_leak: `{{"color": "white | orange | red | blue", "intensity": float}}`
            - spin: `{{"angle_delta": float, "zoom_scale": float}}`
            - ripple: `{{"wave_frequency": float, "wave_amplitude": float}}`
          - MASKING & CUSTOM TRANSITIONS:
            To apply custom shape-mask transitions (like circle, heart, star, diamond reveals):
            1. Set the `"transition"` to the shape name, one of: `"circleopen"`, `"heart"`, `"star"`, `"diamond"`.
            2. Define `"transition_params"` with:
               - `"mask_mode"`: Set to `"custom"` to invoke the custom Mask Agent.
               - `"custom_mask_name"`: One of `"circle"`, `"heart"`, `"star"`, `"diamond"`.
            If you want standard native transitions, set `"mask_mode"` to `"native"` (which is the default).
        - COLOR GRADING (GLOBAL & LOCAL): You can define a global color grade aesthetic at the root level using `global_color_grade`. If you want a specific clip to have its own color correction/grading override, define the `color_grade` object inside that timeline item. Otherwise, leave it out to inherit the global grading. Reason over each clip's `avg_brightness` (exposure score), `face_detected`, and the overall project style/vibe to set optimal parameters:
          - Brightness & Exposure Correction:
            - If `avg_brightness` is low (e.g. < 90), the clip is dark/underexposed: you MUST boost `brightness` (1.1 to 1.3), increase `contrast` (1.1 to 1.25), and lift `gamma` (1.05 to 1.3) to recover shadow detail.
            - If `avg_brightness` is high (e.g. > 165), the clip is bright/overexposed: you MUST reduce `brightness` (0.85 to 0.95) and decrease `gamma` (0.8 to 0.95) to preserve highlight details.
          - Style & Aesthetic Enhancements:
            - Cinematic: Set `temperature` (+5.0 to +15.0) for warmth, `contrast` (1.1 to 1.25) for depth, `vibrance` (0.9 to 0.95) for rich tones, and a subtle vignette (vignette_strength: 0.1 to 0.25).
            - Travel / Vibrant: Set `saturation` (1.15 to 1.35) and `vibrance` (1.2 to 1.4) high, with warm `temperature` (+10.0 to +20.0).
            - Tech / Cool / Clean: Set `temperature` (-5.0 to -15.0) for cool/blue tones, and sharp contrast (1.15 to 1.35).
            - Moody / Noir: Set `contrast` (1.3 to 1.6) high, `saturation` (0.1 to 0.4) very low, `temperature` (-10.0 to -25.0) cool, and apply a heavy vignette (vignette_strength: 0.35 to 0.6).
          - Face Preservation: If `face_detected` is true, avoid extreme temperature/hue shifts that make skin look unnatural. Prefer boosting `vibrance` instead of raw `saturation` to preserve skin realism.
          - Aesthetic Continuity: Ensure that sequential clips of the same setting share similar or smoothly transitioning grading settings for continuity.
          - Allowed ranges: brightness (0.5 to 1.8), contrast (0.5 to 2.0), gamma (0.1 to 10.0), saturation (0.0 to 2.0), vibrance (0.0 to 2.0), hue (-180.0 to 180.0), temperature (-50.0 to 50.0), vignette_strength (0.0 to 1.0), vignette_radius (0.0 to 2.0). All fields must be explicitly populated.
        - CLIP EFFECTS: You can dynamically apply visual filters to individual clips by defining `"clip_effect"`. Specify `"effect_type"` (one of: `none`, `blur`, `pixelate`, `vignette`, `glitch`, `mirror`) and a `"parameters"` object to control it (e.g. `max_blur_size` for blur, `cell_size` for pixelate, `intensity` for glitch).
        - SPEED RAMPING & MOTION: You can alter the time mapping of a clip.
          - Option A: Set `"speed_preset"` to one of:
            - `"constant_fast"` (flat 2.0x speed)
            - `"constant_slow"` (flat 0.5x slow-mo)
            - `"ramp_up"` (starts normal 1.0x, accelerates to 2.5x at the end)
            - `"ramp_down"` (starts fast 2.5x, decelerates to 1.0x)
            - `"speed_bump"` (normal -> fast 3.0x mid-clip spike -> normal)
            - `"freeze_frame"` (starts normal, holds/slows down to 0.05x in the middle, then resumes)
          - Option B: Construct custom `"speed_keyframes"` as a list of `[time_fraction, speed_multiplier]` pairs. For example, `[[0.0, 1.0], [0.5, 3.0], [1.0, 1.0]]` speeds up the middle of the clip by 3x. The array must start at `0.0` and end at `1.0`, with time fractions sorted in ascending order.
          - Mutual Exclusion: Never set both `speed_preset` and `speed_keyframes` on the same item.
        - REVERSE PLAYBACK: Set `"reverse"` to `true` to play the clip backwards.
        - STABILIZATION: Set `"stabilize"` to `true` (and optional `"stabilize_strength"` between 0.0 and 1.0, default 0.5) for clips where the visual analysis notes handheld or shaky camera movement.
        - TEXT STYLE PRESETS: Set `"text_preset"` to style the visual text overlay:
          - `"bold_hype"` (Impact font, yellow, heavy black outline)
          - `"classic_clean"` (Arial font, white, thin outline)
          - `"neon_glow"` (Courier, bright neon cyan/green)
          - `"minimal_pop"` (Arial font, white, clean/borderless)
        - TEXT & STICKER ANIMATIONS: Set `"text_animation"` or `"sticker_animation"` to animate the entry/exit motion:
          - `"fade"` (opacity fade in/out)
          - `"slide_up"` (enters sliding up from bottom, exits sliding down off bottom)
          - `"slide_down"` (enters sliding down from top, exits sliding up off top)
          - `"slide_left"` (enters sliding left from right, exits sliding left off left)
          - `"slide_right"` (enters sliding right from left, exits sliding right off right)
          - `"none"` (default, static)
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

        cleaned_context = self._clean_nested_profanity(context)
        cleaned_prompt = self._clean_profanity(user_prompt)
        user_message = f"User Intent: {cleaned_prompt}\n\nContext Data: {json.dumps(cleaned_context, indent=2)}"

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        if feedback:
            cleaned_feedback = self._clean_profanity(feedback)
            messages.append({
                "role": "system",
                "content": f"CRITICAL REVISION DIRECTIVE:\nYour previous EDL generation failed rendering safety checks. Please correct this issue in your new EDL output:\n{cleaned_feedback}"
            })

        all_models = [self.model_id]
        for m in self.fallback_models:
            if m not in all_models:
                all_models.append(m)

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
                    logger.info(f"CreativeDirector: Calling Gemini model: {model_id}")
                    response = self._call_gemini(model_id, messages)
                    break
                except Exception as e:
                    last_error = e
                    logger.warning(f"CreativeDirector: Gemini Model '{model_id}' failed: {e}")
            if response is not None:
                break

        if response is None:
            logger.warning(f"CreativeDirector: All models failed, using heuristic fallback EDL: {last_error}")
            return self._build_heuristic_edl(user_prompt, media_analyses, target_duration, style)

        try:
            raw_content = response.choices[0].message.content
            logger.debug(f"CreativeDirector (Raw Output):\n{raw_content}")
            edl = self._sanitize_edl(json.loads(raw_content), media_analyses)
            
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
            logger.error(f"CreativeDirector: Error parsing response from Gemini: {e}")
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
                            safe_max_duration = max(0.0, max_duration - 0.15)
                            # Clamp start to safe_max_duration
                            if start > safe_max_duration:
                                start = safe_max_duration
                            # Available duration from start
                            available = safe_max_duration - start
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
                        safe_max_duration = max(0.0, max_duration - 0.15)
                        # Clamp start to safe_max_duration
                        if start > safe_max_duration:
                            start = safe_max_duration
                        # Ensure end is not less than start
                        if end < start:
                            end = start
                        # Clamp end to safe_max_duration
                        if end > safe_max_duration:
                            end = safe_max_duration
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
                clip_name = last_item.get("clip_name", "")
                parts = clip_name.split(":")
                
                # Determine speed multiplier to convert screen seconds to source clip seconds
                speed = 1.0
                speed_preset = last_item.get("speed_preset")
                pacing_style = last_item.get("details", {}).get("pacing_style", "jump-cut")
                if speed_preset:
                    if speed_preset == "constant_fast":
                        speed = 2.0
                    elif speed_preset in ("ramp_up", "ramp_down"):
                        speed = 1.13
                    elif speed_preset == "speed_bump":
                        speed = 1.2
                else:
                    if pacing_style == "speed-ramp":
                        speed = 1.5
                
                actual_diff = 0.0
                if len(parts) == 3:
                    filename, start_str, end_str = parts
                    try:
                        start_val = float(start_str)
                        end_val = float(end_str)
                        max_duration = clip_duration_lookup.get(filename)
                        if max_duration is not None:
                            safe_max_duration = max(0.0, max_duration - 0.15)
                            allowed_extension_screen = max(0.0, safe_max_duration - end_val) / speed
                            actual_diff = min(diff, allowed_extension_screen)
                            source_diff = actual_diff * speed
                            
                            last_item["timeline_end"] = last_item.get("timeline_end", 0) + actual_diff
                            new_end = end_val + source_diff
                            last_item["end_in_clip"] = new_end - start_val
                            last_item["clip_name"] = f"{filename}:{start_val:.3f}:{new_end:.3f}"
                    except ValueError:
                        pass
                else:
                    filename = clip_name
                    try:
                        max_duration = clip_duration_lookup.get(filename)
                        if max_duration is not None:
                            safe_max_duration = max(0.0, max_duration - 0.15)
                            start_val = float(last_item.get("start_in_clip", 0.0))
                            end_val = float(last_item.get("end_in_clip", 0.0))
                            allowed_extension_screen = max(0.0, safe_max_duration - end_val) / speed
                            actual_diff = min(diff, allowed_extension_screen)
                            source_diff = actual_diff * speed
                            
                            last_item["timeline_end"] = last_item.get("timeline_end", 0) + actual_diff
                            last_item["end_in_clip"] = end_val + source_diff
                    except ValueError:
                        pass
                
                if actual_diff > 0:
                    logger.warning(f"Adjusted timeline to meet target duration: added {actual_diff:.3f}s to last clip.")
        rendered_duration = sum(
            item.get("timeline_end", 0) - item.get("timeline_start", 0) for item in timeline
        )
        while rendered_duration < target_duration:
            remaining = target_duration - rendered_duration
            if remaining < 0.1:
                break
            filler_duration = min(remaining, 5.0)

            # Find the best unused highlight segment to use as filler.
            used_ranges: Dict[str, List] = {}
            for item in timeline:
                cn = item.get("clip_name", "")
                parts = cn.split(":")
                if len(parts) == 3:
                    fn, s, e = parts
                    used_ranges.setdefault(fn, []).append((float(s), float(e)))

            best_filler_clip = None
            best_filler_start = 0.0
            best_filler_score = -1.0

            for clip in context.get("available_clips", []):
                fname = clip.get("filename")
                if not fname:
                    continue
                # Check highlights (already sorted by priority DESC)
                for seg in clip.get("highlights", []) + clip.get("segments", []):
                    seg_start = float(seg.get("start", 0.0))
                    seg_end = float(seg.get("end", seg_start + filler_duration))
                    seg_duration = seg_end - seg_start
                    if seg_duration < filler_duration:
                        continue  # too short
                    score = float(seg.get("priority_score", 0.0))
                    if score <= best_filler_score:
                        continue
                    # Check it doesn't overlap with already used ranges
                    already_used = False
                    for (us, ue) in used_ranges.get(fname, []):
                        if not (seg_end <= us or seg_start >= ue):  # overlap
                            already_used = True
                            break
                    if already_used:
                        continue
                    best_filler_clip = fname
                    best_filler_start = seg_start
                    best_filler_score = score

            # Pass 2: If no completely unused range was found (e.g. short input footage),
            # find the best segment overall (allowing reuse/overlaps)
            if best_filler_clip is None:
                for clip in context.get("available_clips", []):
                    fname = clip.get("filename")
                    if not fname:
                        continue
                    for seg in clip.get("highlights", []) + clip.get("segments", []):
                        seg_start = float(seg.get("start", 0.0))
                        seg_end = float(seg.get("end", seg_start + filler_duration))
                        seg_duration = seg_end - seg_start
                        if seg_duration < filler_duration:
                            continue
                        score = float(seg.get("priority_score", 0.0))
                        if score > best_filler_score:
                            best_filler_clip = fname
                            best_filler_start = seg_start
                            best_filler_score = score

            # Hard fallback: first clip at time 0 if nothing better found
            if best_filler_clip is None:
                best_filler_clip = next(iter(clip_duration_lookup), None)
                best_filler_start = 0.0

            if best_filler_clip:
                end_in_clip = best_filler_start + filler_duration
                max_dur = clip_duration_lookup.get(best_filler_clip)
                if max_dur is not None:
                    safe_max_dur = max(0.0, max_dur - 0.15)
                    end_in_clip = min(end_in_clip, safe_max_dur)
                    filler_duration = end_in_clip - best_filler_start
                
                if filler_duration <= 0.05:
                    break
                    
                last_timeline_end = timeline[-1].get("timeline_end", 0) if timeline else 0.0
                filler_item = {
                    "clip_name": f"{best_filler_clip}:{best_filler_start:.3f}:{end_in_clip:.3f}",
                    "start_in_clip": 0.0,
                    "end_in_clip": filler_duration,
                    "timeline_start": last_timeline_end,
                    "timeline_end": last_timeline_end + filler_duration,
                    "transition": "crossfade",
                    "text_overlay": "",
                    "details": {
                        "visual_cue": "Continuation",
                        "sound_design": "",
                        "pacing_style": "jump-cut",
                        "is_hook": False,
                        "keep_original_audio": True,
                        "effect_type": "none",
                        "effect_query": "",
                        "sticker_query": "",
                        "sticker_position": "bottom-center",
                    },
                }
                timeline.append(filler_item)
                logger.warning(
                    f"Added filler clip '{best_filler_clip}' [{best_filler_start:.1f}s] "
                    f"(priority={best_filler_score:.1f}) of {filler_duration:.3f}s to meet target duration."
                )
            else:
                logger.error("Unable to find any clip to use as filler for duration mismatch.")
                break
            
            # Recalculate rendered_duration for next loop check
            rendered_duration = sum(
                item.get("timeline_end", 0) - item.get("timeline_start", 0) for item in timeline
            )
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

