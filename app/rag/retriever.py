import os
from sentence_transformers import SentenceTransformer
from app.config import VECTOR_INDEX_PATH, RELEVANCE_THRESHOLD, RETRIEVAL_TOP_K
from app.rag.vector_store import FAISSVectorStore

# Global singletons
_model = None
_store = None

def get_retriever():
    global _model, _store
    if _model is None:
        _model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    if _store is None:
        _store = FAISSVectorStore()
        _store.load(VECTOR_INDEX_PATH)
    return _model, _store

def retrieve_and_grade(query: str) -> str:
    """
    Retrieves the top K chunks for the query, grades their relevance, 
    and returns either the context or a '[RELEVANCE_FAILED]' message.
    """
    try:
        model, store = get_retriever()
    except Exception as e:
        return f"Error: Vector store failed to load. {str(e)}"
        
    if store.index is None:
        return "Error: FAISS vector index is not initialized. Please run ingestion first."
        
    # Embed the query with normalization
    query_emb = model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
    
    # Search top_k chunks
    results = store.search(query_emb, k=RETRIEVAL_TOP_K)
    
    if not results:
        return "[RELEVANCE_FAILED]: No matching document chunks found."
        
    # Grade the best chunk similarity
    best_match = results[0]
    best_dist = best_match["distance"]
    
    # Map L2 distance to similarity score: s = 1.0 - dist / 2.0
    best_score = 1.0 - (best_dist / 2.0)
    
    print(f"RAG search query: '{query}' -> Best chunk distance: {best_dist:.4f}, similarity score: {best_score:.4f}")
    
    if best_score < RELEVANCE_THRESHOLD:
        print(f"RAG score {best_score:.4f} is below threshold {RELEVANCE_THRESHOLD}.")
        return "[RELEVANCE_FAILED]"
        
    # Combine retrieved chunks
    context_parts = []
    for i, res in enumerate(results):
        score = 1.0 - (res["distance"] / 2.0)
        context_parts.append(f"--- Context Source {i+1} (Relevance Score: {score:.4f}) ---\n{res['chunk']}")
        
    return "\n\n".join(context_parts)
