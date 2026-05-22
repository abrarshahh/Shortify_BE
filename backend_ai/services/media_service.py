import os
import time
import json
from google import genai
from google.genai import types
from typing import List, Dict, Any
from dotenv import load_dotenv
from moviepy import VideoFileClip
from backend_ai.core.config_loader import AGENTS_CONFIG
from backend_ai.core.api_utils import rate_limit_guard
from backend_main.media_metadata import extract_media_metadata

load_dotenv()

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
            print(f"Error extracting file metadata: {e}")
            return {"error": f"Could not extract technical metadata: {str(e)}"}

    @rate_limit_guard(max_retries=5)
    def analyze_video(self, file_path: str) -> Dict[str, Any]:
        """
        Uploads a video to Gemini and analyzes it for key segments, hooks, and mood.
        Using the new google-genai SDK.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")

        # Check Cache
        cache_path = self._get_cache_path(file_path)
        if os.path.exists(cache_path):
            print(f"Found cached analysis for: {file_path}")
            try:
                with open(cache_path, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading cache: {e}")

        file_metadata = self._get_file_metadata(file_path)
        is_image = file_metadata.get("media_type") == "photo"
        print(f"Uploading {'image' if is_image else 'video'} to Gemini: {file_path}")
        
        # Upload using the new SDK (file is the correct argument)
        video_file = self.client.files.upload(file=file_path)
        
        # Wait for the file to be processed
        while video_file.state.name == "PROCESSING":
            print("Waiting for video to process...")
            time.sleep(5)
            video_file = self.client.files.get(name=video_file.name)

        if video_file.state.name == "FAILED":
            raise Exception(f"Video processing failed: {video_file.state.name}")

        print(f"Video processed successfully: {video_file.name}")

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
                  "start": float,
                  "end": float,
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
                  "start": float,
                  "end": float,
                  "description": "Detailed visual description of what is happening in this segment",
                  "audio_description": "What is heard in this segment",
                  "priority_score": float (1-10, rate the overall aesthetic and narrative value of this segment),
                  "should_be_used": boolean (True if this segment is highly recommended for the final video),
                  "segment_focus": "string (STRICTLY ONE SINGLE WORD describing the main subject or theme)"
                }
              ]
            }
            
            Important Instructions:
            1. "all_segments": Break the ENTIRE video down into chronological, sequential segments. Let the natural action dictate the duration of each segment. Segment the video at natural boundaries such as camera cuts, changes in scene, or major shifts in action/subject. A segment can be short or long depending on the action. Give each segment a priority_score based on how useful it would be for a highlight reel.
            2. "captions": Accurately transcribe any speech heard in the video into the captions list.
            3. Do not include any markdown formatting or extra text. Only return the raw JSON object.
            """

        # Generate content using the new SDK with model fallback and exponential backoff
        all_models = [self.primary_model] + self.fallback_models
        last_error = None
        
        for model_id in all_models:
            print(f"Attempting analysis with model: {model_id}")
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
                            print(f"Quota exceeded for {model_id}. Retrying in {delay}s... (Attempt {attempt + 1}/{max_retries})")
                            time.sleep(delay)
                            continue
                        else:
                            print(f"Quota exhausted for {model_id}. Switching to fallback model if available...")
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
            
            # Save to Cache
            try:
                with open(cache_path, "w") as f:
                    json.dump(final_result, f, indent=2)
                print(f"Analysis cached successfully: {cache_path}")
            except Exception as e:
                print(f"Error writing cache: {e}")
                
            return final_result
        except Exception as e:
            print(f"Error parsing Gemini response: {e}")
            return {
                "raw_response": response.text if response else "No response",
                "error": f"Failed to parse structured JSON: {str(e)}"
            }

if __name__ == "__main__":
    # Test script
    import sys
    if len(sys.argv) > 1:
        analyst = MediaAnalyst()
        result = analyst.analyze_video(sys.argv[1])
        print(json.dumps(result, indent=2))
