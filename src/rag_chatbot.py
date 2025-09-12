import faiss
import pickle
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch

class RAGChatbot:
    def __init__(self):
        # -------------------------
        # Load FAISS index + metadata
        # -------------------------
        self.index = faiss.read_index("vector_store/complaints_faiss.index")
        with open("vector_store/metadata.pkl", "rb") as f:
            self.metadata = pickle.load(f)
        print(f"✅ Loaded FAISS index with {self.index.ntotal} vectors")

        # -------------------------
        # Load embedding model (CPU only)
        # -------------------------
        self.embedder = SentenceTransformer(
            "sentence-transformers/all-MiniLM-L6-v2",
            device="cpu"  # <- Fix for meta tensor error
        )

        # -------------------------
        # Load local LLM (free)
        # -------------------------
        model_name = "microsoft/phi-2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float32)
        self.generator = pipeline("text-generation", model=self.model, tokenizer=self.tokenizer)

    def answer(self, query, k=5, max_new_tokens=150):
        # Embed query
        query_emb = self.embedder.encode([query])

        # Search FAISS
        D, I = self.index.search(query_emb, k)

        # Build context
        context_docs = []
        for idx in I[0]:
            doc_meta = self.metadata[idx]
            context_docs.append(f"Product: {doc_meta['product']}, Complaint ID: {doc_meta['complaint_id']}")
        context = "\n".join(context_docs)

        # Build prompt
        prompt = (
            f"You are a helpful assistant summarizing consumer complaints.\n"
            f"Use the following context to answer the user's question.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {query}\nAnswer:"
        )

        # Generate response
        output = self.generator(prompt, max_new_tokens=max_new_tokens, temperature=0.7, do_sample=True)
        answer = output[0]['generated_text']

        # Return only AI response and retrieved context
        return answer, context_docs
