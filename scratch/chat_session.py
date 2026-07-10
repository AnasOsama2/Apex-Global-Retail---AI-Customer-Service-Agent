import sys
import os

# Reconfigure stdout to support unicode emojis on Windows console
sys.stdout.reconfigure(encoding='utf-8')

# Add workspace folder to sys.path so we can import app
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/.."))

from app.graph.builder import graph_instance
from langchain_core.messages import HumanMessage
import app.tools.db as db

def main():
    print("====================================================")
    # Ensure database and tables exist
    db.init_db()
    
    # Check if vector index exists
    if not os.path.exists("./vector_index/index.faiss"):
        print("WARNING: Vector index not found at ./vector_index/index.faiss")
        print("Please run the ingestion script first:")
        print("  py app/rag/ingest.py")
        print("====================================================")
        return

    print("Apex Global Retail Customer Service AI Agent CLI")
    print("Type 'exit' or 'quit' to end the session.")
    print("====================================================")
    
    thread_id = input("Enter Thread ID (default: test_thread): ").strip() or "test_thread"
    customer_id = input("Enter Customer ID (default: cust_101): ").strip() or "cust_101"
    
    config = {"configurable": {"thread_id": thread_id}}
    
    # Check if thread is already initialized in LangGraph checkpoint memory
    state = graph_instance.get_state(config)
    if not state.values:
        print(f"\n[System] Initializing new session for customer {customer_id} in thread {thread_id}...")
        # Supply initial inputs
        input_data = {
            "messages": [],
            "customer_id": customer_id,
            "current_order_id": None,
            "human_handoff": False
        }
        graph_instance.update_state(config, input_data)
    else:
        print(f"\n[System] Resuming existing session for customer {state.values.get('customer_id')}...")
        
    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
                
            input_data = {
                "messages": [HumanMessage(content=user_input)],
                "human_handoff": False  # reset handoff flag for new query
            }
            
            # Run graph execution
            result = graph_instance.invoke(input_data, config=config)
            
            # Fetch latest state details
            final_state = graph_instance.get_state(config)
            agent_response = result["messages"][-1].content
            
            print(f"\nAgent: {agent_response}")
            
            # Print state info helper
            order_id = final_state.values.get("current_order_id")
            handoff = final_state.values.get("human_handoff", False)
            print(f"[State] Current Order ID: {order_id} | Human Handoff: {handoff}")
            
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"\nError: {str(e)}")

if __name__ == "__main__":
    main()
