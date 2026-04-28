import os
import time
import json
from google import genai
from google.genai import types
from typing import List, Dict, Any
from dotenv import load_dotenv

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
        Analyze this video for a short-form content editor. 
        Identify the most 'interesting' segments, high-energy moments, and potential 'hooks' (the first 2 seconds that grab attention).
        
        Output MUST be a valid JSON object with the following structure:
        {
          "summary": "Brief description of the video content",
          "mood": "Overall mood (e.g., Energetic, Calm, Dark, Vibrant)",
          "lighting": "Description of lighting conditions",
          "subjects": ["List of main subjects/objects in the video"],
          "interesting_segments": [
            {
              "start": float,
              "end": float,
              "description": "Why this segment is interesting",
              "energy_score": float (0-1),
              "is_hook": boolean
            }
          ]
        }
        Do not include any markdown formatting or extra text. Only return the JSON.
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
            return analysis
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
