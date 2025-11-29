from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class CVVectorStore:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2")
        self.embedder = SentenceTransformer(model_name)
        self.index = None
        self.chunks = []

    def chunk_text(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size = 500,
            chunk_overlap = 50
        )
        self.chunks = splitter.split_text(text)
        return self.chunks
    
    def create_index(self):
        embeddings = self.embedder.encode(self.chunks)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def search(self, query, top_k=5):
        q_emb = self.embedder.encode([query]).astype("float32")
        distances, indices = self.index.search(q_emb, top_k)
        return [self.chunks[i] for i in indices[0]]