import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    print("Error: Missing GEMINI_API_KEY in .env file")
    exit(1)

MODEL_NAME = "gemini-2.5-flash" 

print(f"Initializing Brain ({MODEL_NAME})...")

try:
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        temperature=0, # 0 = Precise for Accounting, 1 = Creative
        google_api_key=api_key
    )

    print("Sending request...")
    response = llm.invoke("Hello! Identify yourself and tell me the VAT rate in Vietnam.")
    
    print("-" * 30)
    print(response.content)
    print("-" * 30)
    print("System Operational.")

except Exception as e:
    print(f"Connection Failed: {e}")