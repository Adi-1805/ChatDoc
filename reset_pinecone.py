import time
import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = os.getenv("PINECONE_INDEX_NAME") 

def hard_reset():
    pc = Pinecone(api_key=PINECONE_API_KEY)
    
    # Listing all indexes
    indexes = [i.name for i in pc.list_indexes()]
    print(f"Current Cloud Indexes: {indexes}")
    
    # Deleting everything 
    for name in indexes:
        print(f"Deleting old index: '{name}'...")
        pc.delete_index(name)
        time.sleep(5) 
        
    print("All old indexes deleted.")
    
    # Creating the correct 384-dimension index
    print(f"Creating new index '{INDEX_NAME}' with Dimension: 384...")
    try:
        pc.create_index(
            name=INDEX_NAME,
            dimension=384,  
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        
        # Waiting for initialization
        print("Waiting for Pinecone to initialize...")
        while not pc.describe_index(INDEX_NAME).status['ready']:
            time.sleep(2)
            
        print(f"Success! Index '{INDEX_NAME}' is ready for free models.")
        
    except Exception as e:
        print(f"Error creating index: {e}")

if __name__ == "__main__":
    hard_reset()