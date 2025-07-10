# src/rag/generator.py

from transformers import pipeline
from .retriever import retrieve_top_k_chunks
from .prompt import format_prompt

qa_pipeline = pipeline("text-generation", model="google/flan-t5-base", device=-1)


def generate_answer(question: str):
    # Retrieve top chunks
    docs = retrieve_top_k_chunks(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    # Format prompt
    full_prompt = format_prompt(context, question)
    
    # Generate answer
    output = qa_pipeline(full_prompt, max_new_tokens=300, do_sample=True, temperature=0.7)
    return output[0]['generated_text'].split("Answer:")[-1].strip(), docs
