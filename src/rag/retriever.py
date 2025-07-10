# src/rag/retriever.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Load vector store and embeddings
def load_vector_store():
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    db = FAISS.load_local("vector_store/complaints_index", embedding_model, allow_dangerous_deserialization=True)
    return db

def retrieve_top_k_chunks(question: str, k=5):
    db = load_vector_store()
    return db.similarity_search(question, k=k)
