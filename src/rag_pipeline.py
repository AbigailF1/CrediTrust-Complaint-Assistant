import torch
from langchain.prompts import PromptTemplate
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# --- 1. Load the Vector Store and Embedding Function ---
# Ensure you are using the same embedding model as in Task 2
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vector_store_dir = "vector_store"
db = Chroma(
    persist_directory=vector_store_dir,
    embedding_function=embedding_function
)
print("Vector store loaded successfully.")


# --- 2. Set up the Retriever ---
# The retriever's job is to fetch the most relevant documents from the vector store
retriever = db.as_retriever(search_kwargs={'k': 5})
print("Retriever configured to fetch top 5 documents.")


# --- 3. Design the Prompt Template ---
prompt_template = """
You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints based ONLY on the provided context.
Provide a concise, professional summary of the findings. If the context does not contain the answer, state that you don't have enough information from the complaints provided.

Context:
{context}

Question:
{question}

Answer:
"""

PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
)


# --- 4. Configure the LLM (Generator) ---
# We use a quantized version of Mistral-7B for efficiency
# NOTE: This requires a GPU. If you don't have one, this step will be very slow or fail.
# A Hugging Face account and token may be required to download the model.

model_id = "mistralai/Mistral-7B-Instruct-v0.2"

# Use 4-bit quantization to reduce memory usage
# For more info: https://huggingface.co/blog/4bit-transformers-bitsandbytes
quantization_config = {
    "load_in_4bit": True,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4",
    "bnb_4bit_compute_dtype": torch.bfloat16
}

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=quantization_config,
    device_map="auto" # Automatically use GPU if available
)

# Create a text generation pipeline
text_generation_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=512, # Limit the length of the generated answer
    do_sample=True,
    temperature=0.3,
    top_p=0.95
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)
print("LLM pipeline created successfully.")


# --- 5. Assemble the RAG Chain ---
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff", # "stuff" puts all retrieved chunks into the context
    retriever=retriever,
    return_source_documents=True,
    chain_type_kwargs={"prompt": PROMPT}
)

print("RAG chain assembled. Ready for questions.")


# --- Example Usage for Testing ---
if __name__ == "__main__":
    question = "Why are customers unhappy with the BNPL service?"
    print(f"\n> Asking question: {question}")
    result = qa_chain(question)

    print("\n--- Answer ---")
    print(result["result"].strip())

    print("\n--- Sources ---")
    for doc in result["source_documents"]:
        print(f"- Source from Product: {doc.metadata.get('product', 'N/A')}")
        print(f"  Content: {doc.page_content[:200]}...") # Print first 200 chars