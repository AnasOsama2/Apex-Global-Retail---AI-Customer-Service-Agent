# Apex Global Retail - AI Customer Service Agent

An intelligent, stateful customer support chatbot system built with **FastAPI**, **LangGraph**, and **LangChain**. The agent helps retail customers track/update their orders in a database and answers policy questions (refunds, returns, delivery timelines, etc.) using Retrieval-Augmented Generation (RAG) over official corporate policy documents.

It comes equipped with an interactive **React & Tailwind CSS Visual Sandbox** that allows users to test the agent in real time, inspect the internal sqlite database, and visualize the tool calls made by the agent.

---

## 🚀 Key Features

* **LangGraph State Orchestration:** Uses a structured state graph to manage multi-turn conversations, tracking state values like `customer_id`, `current_order_id`, and `human_handoff` flags across message threads.
* **Intent-Based Routing:** A supervisor routing node dynamically evaluates user messages to route them either to RAG-based policy retrieval or database order tools.
* **Retrieval-Augmented Generation (RAG):** Extracts and indexes policies from the `Apex Global Retail.pdf` document using PyPDF2, generates sentence-transformer embeddings, builds a local FAISS vector index, and retrieves relevant chunks with a relevance grader.
* **Order Management Tools:** Executes SQL transactions against an SQLite database (`customer_service_agent.db`) via custom LangChain tools:
  * `get_order_status`: Retrieve active order delivery updates.
  * `create_order`: Generate a new retail order.
  * `update_order`: Update the shipping address on an existing order.
* **Context Window Memory:** Trims the conversation thread dynamically to retain only the last 5 messages to ensure efficient LLM context-window utilization.
* **Modern Web Sandbox:** Includes a dual-mode web interface (`demo.html`) supporting:
  * **Mock Mode:** Client-side mock response simulator.
  * **API Mode:** Connects to the live FastAPI backend with real-time logging of tool executions, database updates, and human-handoff triggers.

---

## 🛠️ Project Structure

The project has been organized under the `the project` directory:

```text
customer_service_agent/
├── .env                         # Environment variables (API keys, models, parameters)
├── .env.example                 # Template for environment configuration
├── .gitignore                   # Files and directories ignored by Git
├── requirements.txt             # Python packages and project dependencies
├── Apex Global Retail.pdf       # Source policies document used for RAG
├── customer_service_agent.db    # SQLite database containing customers & orders
├── demo.html                    # Visual Sandbox interface (React / Tailwind)
├── CustomerServiceChat.jsx      # Standalone React chat component
├── app/                         # Backend FastAPI source code
│   ├── main.py                  # FastAPI entry point & API endpoints
│   ├── config.py                # Environment variable loader
│   ├── graph/                   # LangGraph agent definitions
│   │   ├── state.py             # AgentState TypedDict
│   │   ├── builder.py           # Orchestration state graph assembly
│   │   ├── router_node.py       # Intent router/supervisor node
│   │   └── response_node.py     # Final response generator node
│   ├── tools/                   # SQL/Database interaction tools
│   │   ├── db.py                # Database connection utility
│   │   └── order_tools.py       # create_order, update_order, and get_order_status
│   ├── rag/                     # Retrieval-Augmented Generation pipeline
│   │   ├── ingest.py            # PDF chunking and FAISS indexing pipeline
│   │   ├── retriever.py         # Similarity search & relevance evaluator
│   │   └── vector_store.py      # FAISS loading and saving utility
│   └── memory/                  # Memory trim and conversation trimming logic
│       └── thread_memory.py     # Last-5-message window trimmer
└── vector_index/                # Persisted FAISS vector index & metadata (generated)
```

---

## ⚡ Setup & Installation

### 1. Prerequisites
Ensure you have **Python 3.10+** installed.

### 2. Clone the Repository
```bash
git clone https://github.com/AnasOsama2/Mid_project-digital_hub.git
cd Mid_project-digital_hub/the project
```

### 3. Create a Virtual Environment & Install Dependencies
```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root of `the project` (or duplicate and rename `.env.example`):
```ini
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=qwen/qwen3-32b
VECTOR_INDEX_PATH=./vector_index
CHUNK_SIZE=600
CHUNK_OVERLAP=120
RETRIEVAL_TOP_K=2
MEMORY_WINDOW=5
RELEVANCE_THRESHOLD=0.50
```

---

## 🚀 Running the Project

### 1. Run Text Ingestion (RAG Setup)
Before starting the chatbot, you must parse the corporate policies PDF and build the FAISS vector index:
```bash
# Set your working directory to the project folder containing app/
# Run uvicorn server first
uvicorn app.main:app --reload
```
Once the server is running, trigger the ingestion endpoint via `curl` (or use the ingest action in the frontend sandbox):
```bash
curl -X POST http://127.0.0.1:8000/ingest
```
This generates the FAISS index folder inside `vector_index/`.

### 2. Run the FastAPI Server
If you did not start the server in the step above, launch the FastAPI service:
```bash
uvicorn app.main:app --reload
```
The API documentation will be available at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

### 3. Open the Frontend Visual Sandbox
To run the interactive chat widget:
1. Open the [demo.html](file:///d:/gits/Mid_project-digital_hub/the%20project/demo.html) file directly in your web browser.
2. Enter a **Customer ID** (e.g., `1001` or `1002`) to load their details and order history.
3. Toggle the connection switch from **Mock Mode** to **API Mode** to connect directly to the running FastAPI server.
4. Interact with the bot! Ask about return policies, track orders, or create new ones.

---

## 📡 API Endpoints Reference

### Chat Endpoint
* **Route:** `/chat`
* **Method:** `POST`
* **Request Body:**
  ```json
  {
    "thread_id": "unique-session-id",
    "customer_id": "1001",
    "message": "Where is my order?"
  }
  ```
* **Response Body:**
  ```json
  {
    "response": "Your order #APX-9821 is currently in transit. It is scheduled to arrive on July 12, 2026.",
    "current_order_id": "APX-9821",
    "human_handoff": false
  }
  ```

### Ingestion Endpoint
* **Route:** `/ingest`
* **Method:** `POST`
* **Description:** Extracts text from `Apex Global Retail.pdf`, chunks it, creates embeddings, and updates the local FAISS store.

### Debug / Inspection Endpoints (GET)
* `/db/orders`: Return all rows in the orders SQLite table.
* `/db/customers`: Return all rows in the customers SQLite table.

---

## 🛠️ Built With

* **[LangGraph](https://github.com/langchain-ai/langgraph):** Agent state machine & cyclic workflows.
* **[LangChain](https://github.com/langchain-ai/langchain):** Tool creation, prompt engineering, and LLM orchestration wrappers.
* **[FastAPI](https://fastapi.tiangolo.com/):** High-performance Python backend server.
* **[FAISS (CPU)](https://github.com/facebookresearch/faiss):** High-performance vector database for local semantic search.
* **[SentenceTransformers](https://sbert.net/):** Generates local embeddings for the RAG policy library.
* **[React](https://react.dev/) & [Tailwind CSS](https://tailwindcss.com/):** Clean, modern, responsive frontend chat interface widget.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
