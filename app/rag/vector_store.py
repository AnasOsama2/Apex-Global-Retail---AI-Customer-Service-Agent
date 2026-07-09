import os
import faiss
import json

class FAISSVectorStore:
    def __init__(self):
        self.index = None
        self.chunks = []

    def save(self, directory: str):
        if not os.path.exists(directory):
            os.makedirs(directory)
            
        index_path = os.path.join(directory, "index.faiss")
        chunks_path = os.path.join(directory, "chunks.json")
        
        if self.index is not None:
            faiss.write_index(self.index, index_path)
            
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(self.chunks, f, ensure_ascii=False, indent=2)

    def load(self, directory: str):
        index_path = os.path.join(directory, "index.faiss")
        chunks_path = os.path.join(directory, "chunks.json")
        
        if os.path.exists(index_path):
            self.index = faiss.read_index(index_path)
        else:
            self.index = None
            
        if os.path.exists(chunks_path):
            with open(chunks_path, "r", encoding="utf-8") as f:
                self.chunks = json.load(f)
        else:
            self.chunks = []

    def set_index(self, index, chunks):
        self.index = index
        self.chunks = chunks

    def search(self, query_embedding, k=2):
        if self.index is None:
            return []
            
        # FAISS search returns (distances, indices)
        # distances is a 2D numpy array: [[dist1, dist2, ...]]
        # indices is a 2D numpy array: [[idx1, idx2, ...]]
        distances, indices = self.index.search(query_embedding, k)
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1 and idx < len(self.chunks):
                results.append({
                    "chunk": self.chunks[idx],
                    "distance": float(dist)
                })
        return results
