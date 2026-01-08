import os
from dotenv import load_dotenv

load_dotenv()

# Exporting the keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_ACCESS_TOKEN") 
GOOGLE_API_KEY = os.getenv("GEMINI_API_KEY")
