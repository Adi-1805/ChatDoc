import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from src.config import PINECONE_INDEX_NAME

# Initialize Embeddings
embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore():
    """Returns the base vector store object."""
    return PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embedding_function
    )

def ingest_pdf(file_path: str, namespace: str):  # Added namespace argument
    """
    Ingests a PDF into a SPECIFIC namespace (Session ID).
    """
    print(f"Processing {file_path} into namespace: {namespace}...")
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=300)
    splits = text_splitter.split_documents(docs)
    
    print(f"Uploading {len(splits)} chunks...")
    
    # We add the namespace argument here to isolate data
    vector_store = get_vectorstore()
    vector_store.add_documents(documents=splits, namespace=namespace)
    
    print("Success!")
    return True

def get_retriever(namespace: str):
    """
    Returns a retriever that ONLY looks in the given namespace.
    """
    vector_store = get_vectorstore()
    return vector_store.as_retriever(
        search_kwargs={
            "k": 50,
            "namespace": namespace  # Restricts search to this session
        }
    )

def delete_namespace(namespace: str):
    """
    Deletes all vectors in a specific namespace (Session ID).
    """
    from pinecone import Pinecone
    from src.config import PINECONE_API_KEY, PINECONE_INDEX_NAME
    
    # We connect directly to the index for this operation
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(PINECONE_INDEX_NAME)
    
    try:
        print(f"🧹 Deleting namespace: {namespace}")
        index.delete(delete_all=True, namespace=namespace)
        print("Namespace deleted.")
        return True
    except Exception as e:
        print(f"Error deleting namespace: {e}")
        return False
