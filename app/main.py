import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from app.graph.builder import graph_instance
from app.rag.ingest import ingest_pdf
from app.config import VECTOR_INDEX_PATH
from app.tools.db import get_db_connection
from typing import Optional

app = FastAPI(title="Apex Global Retail AI Customer Service API")

class ChatRequest(BaseModel):
    thread_id: str
    customer_id: str
    message: str

class ChatResponse(BaseModel):
    response: str
    current_order_id: Optional[str]
    human_handoff: bool

class IngestResponse(BaseModel):
    message: str
    chunks_created: int

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """
    Exposes the conversation agent endpoint. It processes messages within a persisted thread state.
    """
    config = {"configurable": {"thread_id": request.thread_id}}
    
    # Check if the thread is already initialized. If not, initialize metadata in state
    state = graph_instance.get_state(config)
    
    input_data = {
        "messages": [HumanMessage(content=request.message)],
        # P3 fix: always reset human_handoff so the thread isn't permanently stuck
        # after a previous handoff. Each new user message starts fresh.
        "human_handoff": False
    }
    
    # If the state has not been initialized with values, we supply the customer_id
    if not state.values:
        print(f"Initializing new conversation thread: {request.thread_id}")
        input_data["customer_id"] = request.customer_id
        input_data["current_order_id"] = None
    else:
        # If customer_id is changing, we update it
        if state.values.get("customer_id") != request.customer_id:
            input_data["customer_id"] = request.customer_id
            print(f"Updating customer_id in thread {request.thread_id} to {request.customer_id}")
            
    try:
        # Run graph execution
        result = graph_instance.invoke(input_data, config=config)
        
        # Extract response and updates
        messages = result.get("messages", [])
        if not messages:
            raise HTTPException(status_code=500, detail="Agent did not return any messages.")
            
        final_message = messages[-1]
        
        # Get latest state values (in case tool nodes mutated state)
        final_state = graph_instance.get_state(config)
        
        return ChatResponse(
            response=final_message.content,
            current_order_id=final_state.values.get("current_order_id"),
            human_handoff=final_state.values.get("human_handoff", False)
        )
    except Exception as e:
        print(f"API chat error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to process chat. {str(e)}")

@app.post("/ingest", response_model=IngestResponse)
async def ingest():
    """
    Triggers text extraction and FAISS indexing of the policies PDF.
    """
    pdf_path = "Apex Global Retail.pdf"
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="Source PDF file 'Apex Global Retail.pdf' not found in workspace.")
        
    try:
        chunks_count = ingest_pdf(pdf_path, VECTOR_INDEX_PATH)
        return IngestResponse(
            message="PDF ingested and index created successfully.",
            chunks_created=chunks_count
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingestion failed. {str(e)}")

# --- Debug / inspection endpoints (not part of core spec) ---

def _fetch_all_rows(table_name: str) -> list[dict]:
    """Helper to avoid duplicating raw sqlite3 connection logic (S5 fix)."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(f"SELECT * FROM {table_name}")  # table_name is hardcoded below, not user input
    rows = cursor.fetchall()
    conn.close()
    return [dict(r) for r in rows]

@app.get("/db/orders")
async def get_orders():
    """Debug endpoint to inspect database orders."""
    try:
        return _fetch_all_rows("orders")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
@app.get("/db/customers")
async def get_customers():
    """Debug endpoint to inspect database customers."""
    try:
        return _fetch_all_rows("customers")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
