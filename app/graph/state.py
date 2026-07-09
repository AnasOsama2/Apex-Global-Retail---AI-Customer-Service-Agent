from typing import Annotated, TypedDict, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    # Appends new messages to the existing list automatically
    messages: Annotated[list[AnyMessage], add_messages]
    customer_id: str
    current_order_id: Optional[str]
    human_handoff: bool
