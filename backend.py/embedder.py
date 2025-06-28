from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

model = SentenceTransformer("all-MiniLM-L6-v2")
index = faiss.IndexFlatL2(384)

def embed_transcript(text):
    chunks = text.split(". ")
    embeddings = model.encode(chunks)
    return list(zip(chunks, embeddings))

def store_to_faiss(chunk_embeddings):
    vectors = np.array([v for _, v in chunk_embeddings])
    index.add(vectors)
    # store chunk mapping somewhere if needed