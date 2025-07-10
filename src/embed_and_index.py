import os
os.environ["USE_TF"] = "0"
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split
import faiss
import os
import numpy as np
from tqdm import tqdm

df = pd.read_csv("../data/filtered_complaints.csv")
texts = df['cleaned_narrative'].tolist()
products = df['Product'].tolist()

def chunk_text(text, max_words=100, overlap=20):
    words = text.split()
    chunks = []
    i = 0
    while i < len(words):
        chunk = words[i:i+max_words]
        chunks.append(" ".join(chunk))
        i += max_words - overlap
    return chunks

chunked_texts = []
metadata = []

for idx, (text, product) in enumerate(zip(texts, products)):
    for chunk in chunk_text(text):
        chunked_texts.append(chunk)
        metadata.append({"original_index": idx, "product": product})

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

embeddings = model.encode(chunked_texts, show_progress_bar=True)

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 = Euclidean distance
index.add(np.array(embeddings))

import pickle

# Save vector index
faiss.write_index(index, "../vector_store/complaints.index")

# Save metadata
with open("../vector_store/metadata.pkl", "wb") as f:
    pickle.dump(metadata, f)
