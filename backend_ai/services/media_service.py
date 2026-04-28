import os
import time
import json
from google import genai
from google.genai import types
from typing import List, Dict, Any
from dotenv import load_dotenv
from moviepy import VideoFileClip

load_dotenv()

class MediaAnalyst:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # New Google GenAI SDK Client
        self.client = genai.Client(api_key=api_key)
        # Using gemini-flash-latest to avoid quota issues with experimental/new models
        self.model_id = "gemini-flash-latest"

    def _get_file_metadata(self, file_path: str) -> Dict[str, Any]:
        """
        Extracts technical metadata from the file locally using MoviePy.
        """
        try:
            with VideoFileClip(file_path) as clip:
                return {
                    "filename": os.path.basename(file_path),
                    "file_size_mb": round(os.path.getsize(file_path) / (1024 * 1024), 2),
                    "duration_seconds": round(clip.duration, 2),
                    "resolution": {
                        "width": clip.size[0],
                        "height": clip.size[1]
                    },
                    "aspect_ratio": round(clip.size[0] / clip.size[1], 2),
                    "fps": round(clip.fps, 2),
                    "has_audio": clip.audio is not None,
                    "extension": os.path.splitext(file_path)[1].lower()
                }
        except Exception as e:
            print(f"Error extracting file metadata: {e}")
            return {"error": f"Could not extract technical metadata: {str(e)}"}

    def analyze_video(self, file_path: str) -> Dict[str, Any]:
        """
        Uploads a video to Gemini and analyzes it for key segments, hooks, and mood.
        Using the new google-genai SDK.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Video file not found: {file_path}")

        print(f"Uploading video to Gemini: {file_path}")
        
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
              "energy_score": float (0-1),
              "is_hook": boolean
            }
          ],
          "all_segments": [
            {
              "start": float,
              "end": float,
              "description": "Detailed visual description of what is happening in this segment",
              "audio_description": "What is heard in this segment"
            }
          ]
        }
        
        Important Instructions:
        1. "all_segments": Break the ENTIRE video down into chronological, sequential segments. Each segment MUST be approximately 5 to 8 seconds long.
        2. "captions": Accurately transcribe any speech heard in the video into the captions list.
        3. Do not include any markdown formatting or extra text. Only return the raw JSON object.
        """

        # Generate content using the new SDK
        response = self.client.models.generate_content(
            model=self.model_id,
            contents=[video_file, prompt],
            config=types.GenerateContentConfig(
                response_mime_type="application/json"
            )
        )

        try:
            # The new SDK might return a parsed object if response_mime_type is set,
            # but let's handle it safely as text just in case.
            text = response.text.strip()
            # Remove markdown if it somehow snuck in (though response_mime_type should prevent it)
            if text.startswith("```"):
                text = text.split("\n", 1)[1].rsplit("\n", 1)[0]
            
            analysis = json.loads(text)
            
            # Extract technical metadata locally
            file_metadata = self._get_file_metadata(file_path)
            
            # Combine results
            final_result = {
                "file_metadata": file_metadata,
                **analysis
            }
            return final_result
        except Exception as e:
            print(f"Error parsing Gemini response: {e}")
            return {
                "raw_response": response.text,
                "error": f"Failed to parse structured JSON: {str(e)}"
            }

if __name__ == "__main__":
    # Test script
    import sys
    if len(sys.argv) > 1:
        analyst = MediaAnalyst()
        result = analyst.analyze_video(sys.argv[1])
        print(json.dumps(result, indent=2))
