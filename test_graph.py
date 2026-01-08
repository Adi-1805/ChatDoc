from langchain_core.messages import HumanMessage
from src.graph import app

def test_chatbot():
    print("Initializing Chatbot Test...\n")
    
    # 1. Simulating a user asking a question
    # (Ensure you have uploaded a PDF in the previous step so Pinecone isn't empty!)
    user_input = "What is the main topic of the uploaded document?"
    
    inputs = {
        "messages": [HumanMessage(content=user_input)]
    }
    
    # 2. Run the graph
    print(f"User: {user_input}")
    print("Thinking...", end="", flush=True)
    
    # app.invoke runs the whole flow from START to END
    result = app.invoke(inputs)
    
    # 3. Extract output
    bot_response = result["messages"][-1].content
    
    print("\n\n" + "-"*30)
    print("BOT RESPONSE:")
    print("-"*30)
    print(bot_response)

if __name__ == "__main__":
    test_chatbot()