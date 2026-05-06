import os
from google import genai
from dotenv import load_dotenv

load_dotenv()

def check_models():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Error: GEMINI_API_KEY not found in .env")
        return

    client = genai.Client(api_key=api_key)
    
    print("--- Available Gemini Models for your API Key ---")
    try:
        # The new SDK list method
        for model in client.models.list():
            print(f"Model: {model.name}")
            print(f"  Supported Actions: {model.supported_actions}")
            print("-" * 30)
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    check_models()
