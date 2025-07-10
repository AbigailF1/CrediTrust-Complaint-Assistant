import gradio as gr
import torch
from threading import Thread
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, TextIteratorStreamer
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_huggingface import HuggingFacePipeline

# --- 1. Configuration and Model Loading ---

# Global variables to hold the model and retriever
qa_chain = None

def load_model_and_retriever():
    """Loads the RAG pipeline components into global variables."""
    global qa_chain

    # Load the vector store we created in Task 2
    embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    db = Chroma(persist_directory="vector_store", embedding_function=embedding_function)

    # Configure the retriever
    retriever = db.as_retriever(search_kwargs={'k': 5})

    # Define the prompt template
    prompt_template = """
    You are a financial analyst assistant for CrediTrust. Your task is to answer questions about customer complaints based ONLY on the provided context.
    Provide a concise, professional summary of the findings. If the context does not contain the answer, state that you don't have enough information from the complaints provided.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Configure and load the LLM (Mistral-7B Instruct)
    model_id = "mistralai/Mistral-7B-Instruct-v0.2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        # Use 4-bit quantization for memory efficiency
        quantization_config={
            "load_in_4bit": True,
            "bnb_4bit_use_double_quant": True,
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_compute_dtype": torch.bfloat16
        },
        device_map="auto"
    )

    # Use a streamer for token-by-token output
    streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    text_pipeline = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.3,
        top_p=0.95,
        streamer=streamer
    )
    
    # Use HuggingFacePipeline for LangChain integration
    from langchain_community.llms import HuggingFacePipeline
    llm = HuggingFacePipeline(pipeline=text_pipeline)

    # Assemble the RAG chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    print("✅ RAG pipeline and model loaded successfully!")

# --- 2. Gradio Application Core Logic (REPLACE the old get_response function with this one) ---

def format_sources(docs):
    """Formats the source documents into a markdown string."""
    if not docs:
        return "*No sources found.*"
    
    source_list = []
    for i, doc in enumerate(docs):
        product = doc.metadata.get('product', 'N/A')
        source_str = f"**Source {i+1} (Product: {product})**\n> {doc.page_content.strip()}"
        source_list.append(source_str)
        
    return "\n\n---\n\n".join(source_list)


def get_response(question):
    """
    Main function to handle user query using the .stream() method.
    """
    if not qa_chain:
        yield "Model not loaded yet. Please wait.", "*Loading...*"
        return

    # 1. Get the source documents first, so we can display them immediately.
    # The retriever is still accessible from our setup.
    source_docs = qa_chain.retriever.invoke(question)
    formatted_sources = format_sources(source_docs)
    
    # 2. Use the .stream() method which is the new, correct way to handle streaming.
    # It takes a dictionary as input.
    answer_stream = qa_chain.stream({"query": question})
    
    # 3. Stream the answer tokens and yield the formatted sources.
    # The answer will build up token by token, while the sources are shown all at once.
    full_answer = ""
    for chunk in answer_stream:
        # The stream yields dictionaries, we need the 'result' key
        if "result" in chunk:
            full_answer += chunk["result"]
            yield full_answer, formatted_sources


    
# --- 3. Gradio Interface Definition ---

with gr.Blocks(theme=gr.themes.Soft(), title="CrediTrust Complaint Analysis") as app:
    gr.Markdown("# 🤖 CrediTrust: Intelligent Complaint Analysis Chatbot")
    gr.Markdown(
        "Ask questions about customer complaints to get synthesized, evidence-backed answers in seconds. "
        "This tool helps product, support, and compliance teams understand customer pain points in real-time."
    )
    
    with gr.Row():
        question_box = gr.Textbox(
            label="Your Question",
            placeholder="e.g., Why are people unhappy with BNPL?",
            lines=2,
            scale=4
        )
        submit_btn = gr.Button("Ask", variant="primary", scale=1)
    
    gr.Markdown("---")
    
    with gr.Accordion("🔎 Synthesized Answer", open=True):
        answer_box = gr.Markdown(label="Answer", value="Your answer will appear here...")
    
    with gr.Accordion("📚 Retrieved Sources", open=False):
        source_box = gr.Markdown(label="Sources", value="The evidence used to generate the answer will appear here...")

    # Define interactions
    submit_btn.click(
        fn=get_response,
        inputs=[question_box],
        outputs=[answer_box, source_box]
    )
    
    clear_btn = gr.ClearButton([question_box, answer_box, source_box])

# --- 4. Application Startup ---

if __name__ == "__main__":
    print("Loading model... This may take a few minutes.")
    load_model_and_retriever()
    print("Starting Gradio application...")
    app.launch(share=True) # Set share=True to get a public link