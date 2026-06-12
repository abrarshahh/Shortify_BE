import os
import time
import json
import logging
from google import genai
from google.genai import types
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from backend_ai.core.config_loader import AGENTS_CONFIG
from backend_ai.core.api_utils import rate_limit_guard
from backend_main.media_metadata import extract_media_metadata

load_dotenv()
logger = logging.getLogger("agents.media")

class MediaAnalyst:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # New Google GenAI SDK Client
        self.client = genai.Client(api_key=api_key)
        # Configuration from agents_config.yaml
        config = AGENTS_CONFIG.get("media_analyst", {})
        self.primary_model = config.get("primary_model", "gemini-1.5-flash")
        self.fallback_models = config.get("fallback_models", ["gemini-1.5-flash-8b"])
        
        # Cache configuration
        self.cache_dir = "data/cache/media_analysis"
        os.makedirs(self.cache_dir, exist_ok=True)

    def _get_cache_path(self, file_path: str) -> str:
        """Generates a cache file path based on file metadata."""
        stats = os.stat(file_path)
        # Using a simple fingerprint: filename + size + mtime
        fingerprint = f"{os.path.basename(file_path)}_{stats.st_size}_{stats.st_mtime}"
        import hashlib
        cache_key = hashlib.md5(fingerprint.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{cache_key}.json")

    def _get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extracts technical metadata from the file locally."""
        try:
            return extract_media_metadata(file_path)
        except Exception as e:
            logger.error(f"MediaAnalyst: Error extracting file metadata: {e}")
            return {"error": f"Could not extract technical metadata: {str(e)}"}

    def _normalize_analysis_durations(self, result: Dict[str, Any], cache_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Robustly converts segment start/end timestamps from MM.SS format to seconds
        if a unit mismatch (MM.SS float) is detected. Saves back to cache if updated.
        """
        metadata = result.get("file_metadata", {})
        duration_seconds = metadata.get("duration_seconds")
        if not duration_seconds or not isinstance(duration_seconds, (int, float)):
            return result

        interesting = result.get("interesting_segments", [])
        all_segs = result.get("all_segments", [])
        
        # Collect all start/end values
        vals = []
        for seg in interesting:
            if "start" in seg:
                vals.append(float(seg["start"]))
            if "end" in seg:
                vals.append(float(seg["end"]))
        for seg in all_segs:
            if "start" in seg:
                vals.append(float(seg["start"]))
            if "end" in seg:
                vals.append(float(seg["end"]))

        if not vals:
            return result

        # Check if they are MM.SS format
        # 1. Seconds part of MM.SS must be < 60
        for v in vals:
            frac = round((v - int(v)) * 100)
            if frac >= 60:
                return result

        # 2. Convert and check bounds
        converted_vals = []
        for v in vals:
            minutes = int(v)
            seconds = round((v - minutes) * 100)
            conv = minutes * 60 + seconds
            converted_vals.append(conv)

        max_original = max(vals)
        max_converted = max(converted_vals)

        # If converted values exceed duration + 5s, it's not MM.SS
        if max_converted > duration_seconds + 5.0:
            return result

        # If max converted is significantly scaled and closer to duration
        if max_converted > max_original * 1.5:
            logger.info(f"MediaAnalyst: Converting segment timestamps from MM.SS float to seconds (max original={max_original}, max converted={max_converted}, duration={duration_seconds})")
            
            # Helper to convert a single timestamp
            def convert_val(v):
                minutes = int(v)
                seconds = round((v - minutes) * 100)
                return float(minutes * 60 + seconds)

            for seg in interesting:
                if "start" in seg:
                    seg["start"] = convert_val(seg["start"])
                if "end" in seg:
                    seg["end"] = convert_val(seg["end"])
            for seg in all_segs:
                if "start" in seg:
                    seg["start"] = convert_val(seg["start"])
                if "end" in seg:
                    seg["end"] = convert_val(seg["end"])

            # Save updated result back to cache if cache_path is provided
            if cache_path:
                try:
                    with open(cache_path, "w") as f:
                        json.dump(result, f, indent=2)
                    logger.info(f"MediaAnalyst: Cache updated with normalized durations: {cache_path}")
                except Exception as e:
                    logger.warning(f"MediaAnalyst: Failed to update cache with normalized durations: {e}")

        return result

    @rate_limit_guard(max_retries=5)
    def analyze_video(self, file_path: str) -> Dict[str, Any]:
        """
        Uploads a video to Gemini and analyzes it for key segments, hooks, and mood.
        Using the new google-genai SDK.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")

        # Check Cache with 7-day TTL check
        cache_path = self._get_cache_path(file_path)
        if os.path.exists(cache_path):
            cache_mtime = os.path.getmtime(cache_path)
            age_seconds = time.time() - cache_mtime
            seven_days_seconds = 7 * 24 * 3600
            
            if age_seconds > seven_days_seconds:
                logger.info(f"MediaAnalyst: Cache has expired (age: {age_seconds / 3600:.1f} hours > 7 days). Invalidating cache.")
                try:
                    os.remove(cache_path)
                except Exception as ex:
                    logger.warning(f"MediaAnalyst: Could not delete expired cache file: {ex}")
            else:
                logger.info(f"MediaAnalyst: Found cached analysis for: {file_path}")
                try:
                    with open(cache_path, "r") as f:
                        cached_data = json.load(f)
                    return self._normalize_analysis_durations(cached_data, cache_path)
                except Exception as e:
                    logger.error(f"MediaAnalyst: Error reading cache: {e}")

        file_metadata = self._get_file_metadata(file_path)
        is_image = file_metadata.get("media_type") == "photo"
        logger.info(f"MediaAnalyst: Uploading {'image' if is_image else 'video'} to Gemini. Path: {file_path}")
        
        # Upload using the new SDK with exponential backoff + jitter
        max_upload_retries = 5
        upload_backoff = 2.0
        video_file = None
        
        for attempt in range(max_upload_retries):
            try:
                video_file = self.client.files.upload(file=file_path)
                break
            except Exception as e:
                err_msg = str(e).lower()
                is_transient = any(phrase in err_msg for phrase in [
                    "429", "rate limit", "quota exceeded", "resource exhausted",
                    "timeout", "too many requests", "service unavailable", "503"
                ])
                if not is_transient or attempt >= max_upload_retries - 1:
                    logger.error(f"MediaAnalyst: Non-retryable error during upload or retries exhausted: {e}")
                    raise e
                import random
                jitter = random.uniform(0.8, 1.2)
                sleep_time = upload_backoff * jitter
                logger.warning(
                    f"MediaAnalyst: Upload failed due to transient error: {e}. "
                    f"Retrying in {sleep_time:.2f}s... (Attempt {attempt + 1}/{max_upload_retries})"
                )
                time.sleep(sleep_time)
                upload_backoff *= 2.0

        if video_file is None:
            raise Exception("Failed to upload video file to Gemini.")

        try:
            # Wait for the file to be processed
            while video_file.state.name == "PROCESSING":
                logger.info("MediaAnalyst: Waiting for video processing on Gemini...")
                time.sleep(5)
                video_file = self.client.files.get(name=video_file.name)

            if video_file.state.name == "FAILED":
                raise Exception(f"Video processing failed: {video_file.state.name}")

            logger.info(f"MediaAnalyst: Video processed successfully. Name on Gemini: {video_file.name}")

            if is_image:
                prompt = """
                Analyze this photo for a short-form content editor.

                Output MUST be a valid JSON object with the following structure:
                {
                  "summary": "Detailed description of the photo content",
                  "mood": "Visual mood (e.g., Energetic, Calm, Dark, Vibrant)",
                  "lighting": "Description of lighting conditions",
                  "subjects": ["List of main subjects or objects visible"],
                  "inferred_metadata": {
                     "inferred_location": "Inferred location if applicable",
                     "time_of_day": "Inferred time of day",
                     "camera_movement": "static"
                  },
                  "audio": {
                     "captions": [],
                     "audio_mood": "none",
                     "audio_features": "none"
                  },
                  "interesting_segments": [
                    {
                      "start": 0.0,
                      "end": 3.0,
                      "description": "The entire photo — describe the most visually compelling aspect",
                      "priority_score": 8.0,
                      "energy_score": 0.7,
                      "is_hook": true,
                      "should_be_used": true,
                      "segment_focus": "one word describing the main subject"
                    }
                  ],
                  "all_segments": [
                    {
                      "start": 0.0,
                      "end": 3.0,
                      "description": "Full description of the photo",
                      "audio_description": "none",
                      "priority_score": 8.0,
                      "should_be_used": true,
                      "segment_focus": "one word describing the main subject"
                    }
                  ]
                }

                Important: Only return the raw JSON object. No markdown.
                """
            else:
                prompt = """
                Analyze this video for a short-form content editor. Listen to the audio track and watch the visual track carefully.
                
                Output MUST be a valid JSON object with the following structure:
                {
                  "summary": "Complete and detailed summary of the entire video content",
                  "mood": "Overall visual and thematic mood (e.g., Energetic, Calm, Dark, Vibrant)",
                  "lighting": "Detailed description of lighting conditions",
                  "subjects": ["Detailed list of main subjects, people, or objects in the video"],
                  "inferred_metadata": {
                     "inferred_location": "Inferred location if applicable",
                     "time_of_day": "Inferred time of day",
                     "camera_movement": "Description of camera movement (e.g., static, handheld, panning)"
                  },
                  "audio": {
                     "captions": ["List of transcribed spoken sentences/captions extracted from the audio, if any. Keep chronological."],
                     "audio_mood": "Overall mood of the audio/music/speech",
                     "audio_features": "Description of audio elements (e.g., background noise, music genre, sound effects)"
                  },
                  "interesting_segments": [
                    {
                      "start": float (start time of the segment in seconds, e.g., 12.5),
                      "end": float (end time of the segment in seconds, e.g., 18.0),
                      "description": "Why this segment is visually or audibly interesting",
                      "priority_score": float (1-10, rate the overall value/importance of this segment),
                      "energy_score": float (0-1),
                      "is_hook": boolean,
                      "should_be_used": boolean,
                      "segment_focus": "string (STRICTLY ONE SINGLE WORD describing the main focus, e.g., mountain, person, river, snow)"
                    }
                  ],
                  "all_segments": [
                    {
                      "start": float (start time of the segment in seconds, e.g., 0.0),
                      "end": float (end time of the segment in seconds, e.g., 5.5),
                      "description": "Detailed visual description of what is happening in this segment",
                      "audio_description": "What is heard in this segment",
                      "priority_score": float (1-10, rate the overall aesthetic and narrative value of this segment),
                      "should_be_used": boolean (True if this segment is highly recommended for the final video),
                      "segment_focus": "string (STRICTLY ONE SINGLE WORD describing the main subject or theme)"
                    }
                  ]
                }
                
                Important Instructions:
                1. "all_segments": Break the ENTIRE video down into chronological, sequential segments. Let the natural action dictate the duration of each segment. Segment the video at natural boundaries such as camera cuts, changes in scene, or major shifts in action/subject. A segment can be short or long depending on the action. Give each segment a priority_score based on how useful it would be for a highlight reel. All start and end values MUST be in raw seconds, not minutes or MM.SS format.
                2. "captions": Accurately transcribe any speech heard in the video into the captions list.
                3. Do not include any markdown formatting or extra text. Only return the raw JSON object.
                """

            # Generate content using the new SDK with model fallback and exponential backoff
            all_models = [self.primary_model] + self.fallback_models
            last_error = None
            response = None
            
            for model_id in all_models:
                logger.info(f"MediaAnalyst: Attempting analysis with model: {model_id}")
                max_retries = 3
                base_delay = 3
                
                success = False
                for attempt in range(max_retries):
                    try:
                        response = self.client.models.generate_content(
                            model=model_id,
                            contents=[video_file, prompt],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json"
                            )
                        )
                        success = True
                        break # Success with this model!
                    except Exception as e:
                        last_error = e
                        if "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e):
                            if attempt < max_retries - 1:
                                delay = base_delay * (2 ** attempt)
                                logger.warning(f"MediaAnalyst: Quota exceeded for {model_id}. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                                time.sleep(delay)
                                continue
                            else:
                                logger.warning(f"MediaAnalyst: Quota exhausted for {model_id}. Switching to fallback model if available...")
                                break # Try next model
                        raise e # Re-raise if not a rate limit
                
                if success:
                    break
            else:
                # If we exhausted all models
                raise last_error if last_error else Exception("All models failed analysis")

            try:
                # The new SDK might return a parsed object if response_mime_type is set,
                # but let's handle it safely as text just in case.
                text = response.text.strip()
                # Remove markdown if it somehow snuck in (though response_mime_type should prevent it)
                if text.startswith("```"):
                    text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
                
                analysis = json.loads(text)
                
                # Extract technical metadata locally
                # Combine results
                final_result = {
                    "file_metadata": file_metadata,
                    **analysis
                }
                final_result = self._normalize_analysis_durations(final_result)
                
                # Save to Cache
                try:
                    with open(cache_path, "w") as f:
                        json.dump(final_result, f, indent=2)
                    logger.info(f"MediaAnalyst: Analysis cached successfully: {cache_path}")
                except Exception as e:
                    logger.error(f"MediaAnalyst: Error writing cache: {e}")
                    
                logger.info(
                    f"MediaAnalyst: Analysis completed. Visual Mood: {final_result.get('mood')}, "
                    f"Lighting: {final_result.get('lighting')}, "
                    f"Main Subjects: {final_result.get('subjects')}, "
                    f"Highlights detected: {len(final_result.get('interesting_segments', []))}"
                )
                return final_result
            except Exception as e:
                logger.error(f"MediaAnalyst: Error parsing Gemini response: {e}")
                return {
                    "raw_response": response.text if response else "No response",
                    "error": f"Failed to parse structured JSON: {str(e)}"
                }
        finally:
            if video_file is not None:
                try:
                    logger.info(f"MediaAnalyst: Deleting remote Gemini file {video_file.name} to conserve storage quota...")
                    self.client.files.delete(name=video_file.name)
                    logger.info("MediaAnalyst: Remote Gemini file deleted successfully.")
                except Exception as e:
                    logger.warning(f"MediaAnalyst: Failed to delete remote Gemini file {video_file.name}: {e}")

if __name__ == "__main__":
    # Test script
    import sys
    if len(sys.argv) > 1:
        analyst = MediaAnalyst()
        result = analyst.analyze_video(sys.argv[1])
        logger.info(json.dumps(result, indent=2))

