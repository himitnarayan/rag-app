import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get values
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "meta-llama/llama-3-8b-instruct")

# Optional: validation
if not OPENROUTER_API_KEY:
    print("⚠️ WARNING: OPENROUTER_API_KEY not found in .env")