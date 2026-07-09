from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from app.graph.state import AgentState
from app.graph.router_node import supervisor_node, should_continue
from app.graph.tools_node import execute_tools_node
from app.graph.response_node import response_generator_node

def create_graph():
    # Initialize the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("supervisor", supervisor_node)
    workflow.add_node("tools", execute_tools_node)
    workflow.add_node("response_generator", response_generator_node)
    
    # Set entry point
    workflow.set_entry_point("supervisor")
    
    workflow.add_conditional_edges(
        "supervisor",
        should_continue,
        {
            "tools": "tools",
            "response_generator": "response_generator",
            "end": END
        }
    )
    
    # Add edges from tools and response_generator
    workflow.add_edge("tools", "response_generator")
    workflow.add_edge("response_generator", END)
    
    # Compile graph with memory saver checkpointer for session state persistence
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)

# Precompiled graph instance
graph_instance = create_graph()
