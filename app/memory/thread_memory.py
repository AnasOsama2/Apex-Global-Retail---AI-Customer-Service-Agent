from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage

def trim_to_last_n_user_messages(messages: list[AnyMessage], n: int = 5) -> list[AnyMessage]:
    """
    Trims the conversation history to remember only the last `n` user messages (HumanMessage)
    and any intermediate responses or tool messages, preserving the leading SystemMessage.
    """
    # Find indices of all HumanMessages in the list
    human_indices = [i for i, msg in enumerate(messages) if isinstance(msg, HumanMessage)]
    
    # If there are fewer or equal to n user messages, keep all of them
    if len(human_indices) <= n:
        return messages
        
    # Slicing index starts at the n-th most recent user message
    start_index = human_indices[-n]
    
    # Find any SystemMessages that occurred before the start_index to preserve them at the very top
    system_messages = [msg for msg in messages[:start_index] if isinstance(msg, SystemMessage)]
    
    # Return system messages followed by the trimmed subset of messages
    return system_messages + messages[start_index:]
