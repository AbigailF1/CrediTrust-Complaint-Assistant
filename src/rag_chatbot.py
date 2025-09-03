import faiss
import pickle
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from sentence_transformers import SentenceTransformer

# -------------------------
# Load FAISS + metadata
# -------------------------
index = faiss.read_index("vector_store/complaints_faiss.index")

with open("vector_store/metadata.pkl", "rb") as f:
    metadata = pickle.load(f)

print(f"✅ Loaded FAISS index with {index.ntotal} vectors")

# -------------------------
# Load embedding model
# -------------------------
embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# -------------------------
# Load local LLM (free)
# -------------------------
model_name = "microsoft/phi-2" 
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# -------------------------
# Ask user query
# -------------------------
query = input("\nAsk a question about consumer complaints: ")

# Embed query and search FAISS
query_emb = embedder.encode([query])
k = 5
D, I = index.search(query_emb, k)

# Retrieve top results
context_docs = []
for idx in I[0]:
    doc_meta = metadata[idx]
    context_docs.append(f"Product: {doc_meta['product']}, Complaint ID: {doc_meta['complaint_id']}")

context = "\n".join(context_docs)

# -------------------------
# Build prompt for LLM
# -------------------------
prompt = (
    f"You are a helpful assistant summarizing consumer complaints.\n"
    f"Use the following context to answer the user's question.\n\n"
    f"Context:\n{context}\n\n"
    f"Question: {query}\nAnswer:"
)

# -------------------------
# Generate response
# -------------------------
response = generator(prompt, max_new_tokens=150, temperature=0.7, do_sample=True)
print("\n AI Response:\n", response[0]['generated_text'])
