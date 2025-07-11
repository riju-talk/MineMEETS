# embedder.py
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

class MeetingEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)  # Dimension of MiniLM embeddings
        self.text_chunks = []
        
    def chunk_text(self, text, max_length=256, overlap=50):
        words = text.split()
        chunks = []
        for i in range(0, len(words), max_length - overlap):
            chunks.append(" ".join(words[i:i+max_length]))
        return chunks
        
    def add_transcript(self, segments):
        for segment in segments:
            chunks = self.chunk_text(segment["text"])
            for chunk in chunks:
                self.text_chunks.append(chunk)
                embedding = self.model.encode([chunk])
                self.index.add(embedding)