import os
from dotenv import load_dotenv

# Load env variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "qwen/qwen3-32b")
VECTOR_INDEX_PATH = os.getenv("VECTOR_INDEX_PATH", "./vector_index")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "2"))
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "5"))
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.75"))
