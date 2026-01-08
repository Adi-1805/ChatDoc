import streamlit as st
import os
import tempfile
import uuid
from datetime import datetime
from langchain_core.messages import HumanMessage
from src.graph import app as graph_app
from src.rag import ingest_pdf, delete_namespace 

# Page configuration
st.set_page_config(
    page_title="AI Document Analyzer",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Header styling */
    h1 {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] > div:first-child {
        padding-top: 2rem;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: rgba(255, 255, 255, 0.9);
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        border-radius: 0.5rem;
        border: none;
        padding: 0.75rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* File uploader styling */
    .uploadedFile {
        border-radius: 0.5rem;
        border: 2px dashed #667eea;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Chat message styling */
    .stChatMessage {
        padding: 1rem;
        border-radius: 0.75rem;
        margin: 1rem 0;
    }
    
    /* Success/Error message styling */
    .stSuccess {
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    .stError {
        border-radius: 0.5rem;
        padding: 1rem;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea15 0%, #764ba215 100%);
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    /* Session info badge */
    .session-badge {
        display: inline-block;
        background: rgba(255, 255, 255, 0.2);
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        margin-top: 0.5rem;
    }
    
    /* Divider styling */
    hr {
        margin: 2rem 0;
        border: none;
        border-top: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    /* Chat input styling */
    [data-testid="stChatInput"] {
        border-radius: 1rem;
        border: 2px solid #667eea;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
    }
    ::-webkit-scrollbar-track {
        background: #f1f1f1;
    }
    ::-webkit-scrollbar-thumb {
        background: #667eea;
        border-radius: 4px;
    }
    ::-webkit-scrollbar-thumb:hover {
        background: #764ba2;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
    <div style='text-align: center; margin-bottom: 2rem;'>
        <h1>AI Document Analyzer</h1>
        <p style='font-size: 1.2rem; color: #666; margin-top: -1rem;'>
            Upload documents, get intelligent insights instantly
        </p>
    </div>
""", unsafe_allow_html=True)

# 1. Session Management
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.files_processed = False
    st.session_state.start_time = datetime.now()

session_id = st.session_state.session_id

# 2. Sidebar
with st.sidebar:
    st.header("Upload Material")
    st.caption(f"Session ID: {session_id[:8]}...")
    
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if uploaded_file and not st.session_state.files_processed:
        if st.button("Process PDF"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                tmp_path = tmp_file.name
            
            with st.spinner("Indexing document..."):
                try:
                    ingest_pdf(tmp_path, namespace=session_id)
                    st.session_state.files_processed = True
                    st.success("Indexed!")
                    st.warning(
                        "IMPORTANT: When you are done, please click 'End Session' below. "
                        "This frees up the database for your next use!"
                    )
                except Exception as e:
                    st.error(f"Error: {e}")
                finally:
                    os.remove(tmp_path)
    
    st.divider()
    
    # 3. END SESSION BUTTON
    if st.button("End Session & Clear Data"):
        with st.spinner("Cleaning up your data from the cloud..."):
            # A. Delete vectors from Pinecone
            delete_namespace(session_id)
            
            # B. Clear Streamlit History
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            
            # C. Refresh to generate a new Session ID
            st.rerun()

# 4. Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display Chat History
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User Input
if prompt := st.chat_input("Ask a question about your notes..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        # Retry logic for the Brain is handled inside graph.py, but we catch UI errors here
        try:
            inputs = {
                "messages": [HumanMessage(content=prompt)],
                "namespace": session_id
            }
            result = graph_app.invoke(inputs)
            bot_response = result["messages"][-1].content
            
            message_placeholder.markdown(bot_response)
            st.session_state.messages.append({"role": "assistant", "content": bot_response})
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
