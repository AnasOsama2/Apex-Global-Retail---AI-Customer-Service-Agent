from langchain_core.messages import SystemMessage, AIMessage
from app.graph.state import AgentState
from app.llm.groq_client import get_llm

def response_generator_node(state: AgentState):
    print("--- RESPONSE GENERATOR NODE ---")
    messages = state.get("messages", [])
    human_handoff = state.get("human_handoff", False)
    
    # Layer 1 Handoff (or previously set handoff)
    if human_handoff:
        handoff_text = "I'm sorry, I could not find a clear answer to your request in our policy documents. I am routing your query to a human supervisor."
        print("Layer 1 Handoff active. Returning standard handoff response.")
        return {
            "messages": [AIMessage(content=handoff_text)],
            "human_handoff": True
        }
        
    # Standard synthesis prompt
    system_instruction = """You are the Response Generator for Apex Global Retail Customer Service.
Your task is to write a final, customer-facing response based on the conversation history and the results of any tools executed.

Rules:
- If a policy was retrieved via policy_rag, synthesize your response grounded STRICTLY in that context. Do not make promises or state policies that are not explicitly written in the context.
- If the retrieved policy does not contain a clear answer to the customer's question, do not speculate or make up information. You must explicitly reply that you are routing them to a human supervisor, and append the tag [HANDOFF] at the end of your message.
- If an order action was executed, summarize the results clearly and professionally.
- Keep the response friendly, helpful, and concise.
"""
    
    # Prepare messages for LLM
    llm_messages = [SystemMessage(content=system_instruction)] + [
        msg for msg in messages if not isinstance(msg, SystemMessage)
    ]
    
    # Call LLM (without tools) to synthesize final response
    llm = get_llm()
    response = llm.invoke(llm_messages)
    
    content = response.content
    state_updates = {}
    
    # Layer 2 Handoff: check if the LLM output requests a handoff
    if "[HANDOFF]" in content or "human supervisor" in content.lower():
        print("Layer 2 Handoff detected in LLM response.")
        state_updates["human_handoff"] = True
        # Clean up any technical tags
        content = content.replace("[HANDOFF]", "").strip()
        response.content = content
        
    state_updates["messages"] = [response]
    return state_updates
