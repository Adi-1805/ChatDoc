from dotenv import load_dotenv
import os

load_dotenv()

key = os.getenv("OPENAI_API_KEY")
ls_key = os.getenv("LANGCHAIN_API_KEY")

if key and key.startswith("sk-") and ls_key:
    print("✅ Environment is ready! API keys detected.")
else:
    print("❌ Check your .env file.")