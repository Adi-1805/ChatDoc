import time
from typing import Annotated, List, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from src.rag import get_retriever
from src.config import GOOGLE_API_KEY

# 1. Configuration the LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash-lite",
    temperature=0,
    max_retries=2,
    api_key=GOOGLE_API_KEY
)

# Initialize Search Tool
search_tool = DuckDuckGoSearchRun()

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    context: str
    namespace: str

# 2. Nodes
def retrieve_node(state: AgentState):
    latest_question = state["messages"][-1].content
    namespace = state["namespace"]
    
    print(f"Retrieving for: {latest_question} in namespace: {namespace}")
    
    retriever = get_retriever(namespace=namespace)
    docs = retriever.invoke(latest_question)
    
    context_text = "\n\n".join([d.page_content for d in docs])
    return {"context": context_text}

def generate_node(state: AgentState):
    context = state["context"]
    question = state["messages"][-1].content
    
    if not context:
        return {"messages": [BaseMessage(content="I checked your documents but couldn't find an answer.", type="ai")]}

    template = ChatPromptTemplate.from_messages([
        ("system", """You are a helpful chat assistant for analyzing user-uploaded documents and answering related queries. 
Prioritize document content; use external sources only if needed, and clearly disclose them.

For each query:
1. Check documents first using the context.
2. If info is missing, output EXACTLY the word: "NO_ANSWER".
3. In response:
   - If from documents: Respond directly, cite sources.
   - If from web: Prefix with "This part is from external sources (internet), not your document: [info]."
   - Combine if mixed, separating clearly.
4. Keep responses clear, structured. No hallucinations.

Start by noting source.
        
        Context Snippets:
        {context}"""),
        ("human", "{question}"),
    ])

    doc_chain = template | llm
    
    # Try 1: Ask the Document
    try:
        response = doc_chain.invoke({"context": context, "question": question})
        content = response.content.strip()
        
        # If the document had the answer, return it immediately
        if "NO_ANSWER" not in content:
            return {"messages": [response]}
            
        # --- FALLBACK: WEB SEARCH MODE ---
        print("Answer not in doc. Switching to Web Search...")
        
        # 1. Perform Search
        web_results = search_tool.invoke(question)
        
        # 2. Generate Answer with Warning
        web_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful assistant.
            The user's document did not contain the answer, so we searched the internet.
            
            Instructions:
            1. Answer the question using the Web Search Results below.
            2. You MUST start your response with this exact phrase: 
               "**Note:** This information comes from the internet, not your uploaded document."
            
            Web Search Results:
            {web_results}"""),
            ("human", "{question}"),
        ])
        
        web_chain = web_template | llm
        web_response = web_chain.invoke({"web_results": web_results, "question": question})
        
        return {"messages": [web_response]}

    except Exception as e:
        # Simple Error Handling
        return {"messages": [BaseMessage(content=f"An error occurred: {str(e)}", type="ai")]}

    # # Retry loop logic
    # max_retries = 3
    # for attempt in range(max_retries):
    #     try:
    #         response = chain.invoke({"context": context, "question": question})
    #         return {"messages": [response]}
    #     except Exception as e:
    #         if "503" in str(e) or "overloaded" in str(e).lower():
    #             if attempt < max_retries - 1:
    #                 wait_time = 2 * (attempt + 1)
    #                 print(f"Google Server 503. Retrying in {wait_time}s...")
    #                 time.sleep(wait_time)
    #                 continue
    #         raise e
    
    # chain = template | llm
    # response = chain.invoke({"context": context, "question": question})
    
    # return {"messages": [response]}

# 3. Graph
workflow = StateGraph(AgentState)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_edge(START, "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)
app = workflow.compile()
