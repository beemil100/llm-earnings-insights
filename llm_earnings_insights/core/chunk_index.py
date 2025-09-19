from typing import List, Tuple
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np


class ChunkIndex:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def build(self, pages: List[Tuple[int, str]], chunk_size: int = 500):
        self.chunks = []
        texts = []
        for page_num, text in pages:
            words = text.split()
            for i in range(0, len(words), chunk_size):
                chunk = " ".join(words[i:i+chunk_size])
                if chunk.strip():
                    self.chunks.append((page_num, chunk))
                    texts.append(chunk)
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def query(self, q: str, top_k: int = 3):
        q_emb = self.model.encode([q], convert_to_numpy=True)
        distances, indices = self.index.search(q_emb, top_k)
        results = []
        for idx in indices[0]:
            page, chunk = self.chunks[idx]
            results.append((page, chunk))
        return results
