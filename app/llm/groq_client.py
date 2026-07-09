import os
from langchain_groq import ChatGroq
from app.config import GROQ_API_KEY, GROQ_MODEL

def get_llm():
    # Make sure GROQ_API_KEY is in environment if needed, ChatGroq reads it from groq_api_key argument or GROQ_API_KEY env var
    return ChatGroq(
        model_name=GROQ_MODEL,
        groq_api_key=GROQ_API_KEY,
        temperature=0.3
    )
