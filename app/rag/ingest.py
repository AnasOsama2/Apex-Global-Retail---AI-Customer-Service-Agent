import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import faiss
import PyPDF2
from sentence_transformers import SentenceTransformer
from app.config import VECTOR_INDEX_PATH, CHUNK_SIZE, CHUNK_OVERLAP
from app.rag.vector_store import FAISSVectorStore

def split_text_recursive(text: str, chunk_size: int = 600, chunk_overlap: int = 120, separators=None) -> list[str]:
    if separators is None:
        separators = ["\n\n", "\n", " ", ""]
        
    if len(text) <= chunk_size:
        return [text.strip()]
        
    for sep in separators:
        if sep == "":
            # Character fallback if no other separators match or we are down to empty string
            chunks = []
            start = 0
            while start < len(text):
                chunks.append(text[start:start + chunk_size].strip())
                start += chunk_size - chunk_overlap
            return [c for c in chunks if c]
            
        if sep in text:
            parts = text.split(sep)
            chunks = []
            current_chunk = []
            current_len = 0
            
            for part in parts:
                part_len = len(part) + (len(sep) if current_chunk else 0)
                if current_len + part_len > chunk_size:
                    if current_chunk:
                        chunk_text = sep.join(current_chunk)
                        chunks.append(chunk_text.strip())
                        
                        # Build overlap by taking items from end of current_chunk
                        overlap_chunk = []
                        overlap_len = 0
                        for p in reversed(current_chunk):
                            p_len = len(p) + (len(sep) if overlap_chunk else 0)
                            if overlap_len + p_len <= chunk_overlap:
                                overlap_chunk.insert(0, p)
                                overlap_len += p_len
                            else:
                                break
                        current_chunk = overlap_chunk
                        current_len = overlap_len
                    else:
                        # Single part is larger than chunk_size, split recursively
                        sub_chunks = split_text_recursive(part, chunk_size, chunk_overlap, separators)
                        if sub_chunks:
                            chunks.extend(sub_chunks[:-1])
                            current_chunk = [sub_chunks[-1]]
                            current_len = len(sub_chunks[-1])
                        continue
                
                current_chunk.append(part)
                current_len += part_len
                
            if current_chunk:
                chunks.append(sep.join(current_chunk).strip())
            return [c for c in chunks if c]
            
    return [text.strip()]

def embed_chunks(chunks, model_name='sentence-transformers/all-MiniLM-L6-v2'):
    model = SentenceTransformer(model_name)
    # Use normalize_embeddings=True to get normalized vectors for cosine similarity
    embeddings = model.encode(chunks, convert_to_numpy=True, normalize_embeddings=True)
    return model, embeddings

def create_faiss_index(embeddings):
    dim = embeddings.shape[1] 
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def ingest_pdf(pdf_path: str, vector_db_path: str):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found at: {pdf_path}")
        
    print(f"Reading PDF from: {pdf_path}...")
    reader = PyPDF2.PdfReader(pdf_path)
    full_text = ""
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            full_text += text + "\n"
            
    print("Splitting text into chunks...")
    chunks = split_text_recursive(full_text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    print(f"Created {len(chunks)} chunks.")
    
    print("Embedding chunks and creating FAISS index...")
    model, embeddings = embed_chunks(chunks)
    index = create_faiss_index(embeddings)
    
    print(f"Saving vector database to {vector_db_path}...")
    store = FAISSVectorStore()
    store.set_index(index, chunks)
    store.save(vector_db_path)
    print("Ingestion completed successfully.")
    return len(chunks)

if __name__ == "__main__":
    import sys
    pdf_file = "Apex Global Retail.pdf"
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
    ingest_pdf(pdf_file, VECTOR_INDEX_PATH)
