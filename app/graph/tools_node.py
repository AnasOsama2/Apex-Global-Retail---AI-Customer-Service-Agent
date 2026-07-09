import re
from langchain_core.messages import ToolMessage, AIMessage
from app.graph.state import AgentState
from app.rag.retriever import retrieve_and_grade
from app.tools.order_tools import create_order, update_order, get_order_status

# Map of tool names to their implementations
ORDER_TOOLS = {
    "create_order": create_order,
    "update_order": update_order,
    "get_order_status": get_order_status
}

def execute_tools_node(state: AgentState):
    print("--- EXECUTE TOOLS NODE ---")
    messages = state.get("messages", [])
    if not messages:
        return {}
        
    last_message = messages[-1]
    if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
        print("No tool calls found on the last message.")
        return {}
        
    tool_messages = []
    state_updates = {}
    
    for tool_call in last_message.tool_calls:
        name = tool_call["name"]
        args = tool_call["args"]
        tool_id = tool_call["id"]
        
        print(f"Executing tool {name} with args {args}...")
        
        if name == "policy_rag":
            # Call our retriever with grading logic
            query = args.get("query", "")
            context = retrieve_and_grade(query)
            
            if context == "[RELEVANCE_FAILED]":
                print("Tool policy_rag: Relevance grading failed. Triggering human handoff.")
                content = "No relevant policy found in company database. (Relevance score below threshold)"
                state_updates["human_handoff"] = True
            else:
                content = context
                
            tool_messages.append(ToolMessage(content=content, name=name, tool_call_id=tool_id))
            
        elif name in ORDER_TOOLS:
            tool_obj = ORDER_TOOLS[name]
            try:
                res_str = tool_obj.invoke(args)
            except Exception as e:
                res_str = f"Error: Tool execution failed. {str(e)}"
                
            print(f"Tool {name} result: {res_str}")
            
            # Extract and update current_order_id if present
            if "order_id" in args:
                state_updates["current_order_id"] = str(args["order_id"])
            elif "Order #" in res_str:
                match = re.search(r"Order #(\d+)", res_str)
                if match:
                    state_updates["current_order_id"] = match.group(1)
                    print(f"Extracted current_order_id: {state_updates['current_order_id']}")
                    
            tool_messages.append(ToolMessage(content=res_str, name=name, tool_call_id=tool_id))
            
        else:
            error_msg = f"Error: Tool '{name}' is not recognized."
            print(error_msg)
            tool_messages.append(ToolMessage(content=error_msg, name=name, tool_call_id=tool_id))
            
    # Combine state updates with new messages
    state_updates["messages"] = tool_messages
    return state_updates
