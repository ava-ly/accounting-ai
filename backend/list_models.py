import os
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    print("No API Key found")
else:
    genai.configure(api_key=api_key)
    print("Listing available models for your API Key...\n")
    try:
        # List all models that support 'generateContent' (Chat)
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                print(f"Available: {m.name}")
    except Exception as e:
        print(f"Error listing models: {e}")