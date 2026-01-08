import time
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")

def initialize_database():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    existing_indexes = [i.name for i in pc.list_indexes()]
    
    print(f"Current Indexes: {existing_indexes}")
    
    # Check if our specific free index exists
    if PINECONE_INDEX_NAME not in existing_indexes:
        print(f"Index '{PINECONE_INDEX_NAME}' not found. Creating...")
        try:
            pc.create_index(
                name=PINECONE_INDEX_NAME,
                dimension=384,  
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            while not pc.describe_index(PINECONE_INDEX_NAME).status['ready']:
                time.sleep(1)
            print("New 384-dimension index is ready!")
        except Exception as e:
            print(f"Error: {e}")
    else:
        print(f"Index '{PINECONE_INDEX_NAME}' already exists.")

if __name__ == "__main__":
    initialize_database()