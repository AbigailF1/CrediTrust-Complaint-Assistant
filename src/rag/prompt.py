# src/rag/prompt.py

def format_prompt(context: str, question: str) -> str:
    template = """You are a financial analyst assistant for CrediTrust.
Your task is to answer questions about customer complaints.
Use only the context below to answer the question. 
If the context does not contain the answer, say: "I don't have enough information."

Context:
{context}

Question:
{question}

Answer:"""
    return template.format(context=context, question=question)
