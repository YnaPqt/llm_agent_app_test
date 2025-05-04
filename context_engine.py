from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

class SimpleVectorDB:
    def __init__(self):
        self.texts = []
        self.index = faiss.IndexFlatL2(384)
        self.embeddings = []

    def add_text(self, text):
        embedding = model.encode([text])
        self.embeddings.append(embedding)
        self.index.add(embedding)
        self.texts.append(text)

    def query(self, query_text, top_k=1):
        query_embedding = model.encode([query_text])
        D, I = self.index.search(query_embedding, top_k)
        return [self.texts[i] for i in I[0] if i < len(self.texts)]
