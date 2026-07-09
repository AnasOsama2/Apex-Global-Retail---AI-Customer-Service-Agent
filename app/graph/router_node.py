import re
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from app.graph.state import AgentState
from app.llm.groq_client import get_llm
from app.memory.thread_memory import trim_to_last_n_user_messages
from app.tools.order_tools import create_order, update_order, get_order_status
from pydantic import BaseModel, Field
from langchain_core.tools import tool

class PolicyRagInput(BaseModel):
    query: str = Field(description="The user's query about policies, shipping, returns, refunds, terms, etc. Be specific.")

@tool("policy_rag", args_schema=PolicyRagInput)
def policy_rag(query: str) -> str:
    """Queries the company policy retriever to answer informational questions about company policies, shipping, returns, refunds, terms, etc.

    NOTE: This is a *binding-only* tool definition so the LLM can request it via
    tool_calls. The actual retrieval logic lives in tools_node.execute_tools_node(),
    which intercepts calls to 'policy_rag' and delegates to app.rag.retriever.
    This function body is never executed at runtime.
    """
    return ""

def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks that some models (e.g. Qwen3) emit."""
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

def supervisor_node(state: AgentState):
    print("--- SUPERVISOR NODE ---")
    messages = state.get("messages", [])
    customer_id = state.get("customer_id", "")
    current_order_id = state.get("current_order_id", None)
    
    # Format the system prompt dynamically
    system_content = f"""You are a professional, helpful Customer Service AI Agent for Apex Global Retail.
Your primary workloads:
1. Informational Queries: Answer company policy questions (refunds, shipping, terms). You MUST use the `policy_rag` tool to search the policies before answering.
2. Operational Queries: Execute transactional actions on orders (creating, updating, checking status). Use the appropriate order management tools.

Current Customer ID: {customer_id}
Current Order ID: {current_order_id or 'None'}

Rules:
- You must always invoke the `policy_rag` tool to search policy documents when the customer asks informational questions. Never guess or make up policies.
- Do not make promises or state policy exceptions to the customer (e.g. extending refund windows) unless the policy text explicitly states they are allowed.
- If the customer wants to buy/place an order, use `create_order`.
- If the customer wants to change an address or quantity of an order, use `update_order`.
- If the customer wants to check the status, use `get_order_status`.
"""
    
    # Build message list: system message at top, then conversation (without stale system messages)
    full_messages = [SystemMessage(content=system_content)] + [
        msg for msg in messages if not isinstance(msg, SystemMessage)
    ]
    
    # Trim to last 5 user messages
    trimmed_messages = trim_to_last_n_user_messages(full_messages, n=5)
    
    # Bind tools to LLM
    llm = get_llm()
    tools = [policy_rag, create_order, update_order, get_order_status]
    llm_with_tools = llm.bind_tools(tools)
    
    # Invoke LLM
    response = llm_with_tools.invoke(trimmed_messages)
    
    # P5 fix: When the supervisor responds directly (no tool calls), its raw output
    # goes to END without passing through response_generator. Strip <think> tags
    # so they don't leak into the user-facing response.
    if not (hasattr(response, "tool_calls") and response.tool_calls):
        response.content = _strip_think_tags(response.content)
    
    return {"messages": [response]}

def should_continue(state: AgentState):
    messages = state.get("messages", [])
    if not messages:
        return "end"
        
    last_message = messages[-1]
    
    # Check if human handoff was already set
    if state.get("human_handoff", False):
        return "response_generator"
        
    # If the supervisor generated tool calls, execute them
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        print(f"Supervisor requested tool calls: {[tc['name'] for tc in last_message.tool_calls]}")
        return "tools"
        
    print("Supervisor responded directly with no tool calls. Routing to end.")
    return "end"
