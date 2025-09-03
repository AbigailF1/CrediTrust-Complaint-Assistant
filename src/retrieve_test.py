# src/retrieve_test.py
# Retrieval test for CrediTrust Complaint Chatbot

import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import pandas as pd

# -----------------------------
# 1. Load vector store
# -----------------------------
index = faiss.read_index("vector_store/complaints_faiss.index")

with open("vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f"Loaded FAISS index with {index.ntotal} vectors and metadata entries: {len(metadata)}")

# -----------------------------
# 2. Load embedding model
# -----------------------------
model_name = "sentence-transformers/all-MiniLM-L6-v2"
embedder = SentenceTransformer(model_name)

# -----------------------------
# 3. Define a test query
# -----------------------------
query = "I was charged extra interest on my credit card even after I paid the balance. What should I do?"
query_embedding = embedder.encode([query])

# -----------------------------
# 4. Retrieve top-k similar chunks
# -----------------------------
k = 5
distances, indices = index.search(np.array(query_embedding, dtype=np.float32), k)

print("\n🔍 Top Retrieved Complaints:")
for i, idx in enumerate(indices[0]):
    meta = metadata[idx]
    print(f"\nRank {i+1}")
    print(f"Product: {meta['product']}")
    print(f"Complaint ID: {meta['complaint_id']}")
