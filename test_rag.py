import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.config import PINECONE_INDEX_NAME, PINECONE_API_KEY

# 1. Initialize Embedding Function
embedding_function = OpenAIEmbeddings(model="text-embedding-3-small")

def get_vectorstore():
    """
    Returns the Pinecone Vector Store connected to your cloud index.
    """
    # This automatically looks for PINECONE_API_KEY in env vars
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedding_function
    )

def ingest_pdf(file_path: str):
    """
    Loads PDF, splits it, and UPLOADS vectors to Pinecone.
    """
    print(f"📄 Processing {file_path}...")
    
    # A. Load
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    # B. Split
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    # C. Upload to Pinecone
    print(f"🚀 Uploading {len(splits)} chunks to Pinecone...")
    vector_store = get_vectorstore()
    vector_store.add_documents(documents=splits)
    
    print(f"✅ Success! Data is now live in the cloud.")
    return True

def get_retriever():
    """
    Returns a retriever that fetches from Pinecone.
    """
    vector_store = get_vectorstore()
    return vector_store.as_retriever(search_kwargs={"k": 3})