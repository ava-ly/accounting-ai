import os
from supabase import create_client
from dotenv import load_dotenv

# Load env locally, or use GitHub Secrets in Action
load_dotenv()

url = os.environ.get("SUPABASE_URL")
key = os.environ.get("SUPABASE_KEY")

if not url or not key:
    print("Error: Missing credentials")
    exit(1)

supabase = create_client(url, key)

print("Sending Heartbeat (Write Operation)...")

# UPSERT: This forces a Write to the disk
data = {"id": 1, "last_ping": "now()"}
response = supabase.table("heartbeat").upsert(data).execute()

print("Success! Database is active.")
print(response.data)