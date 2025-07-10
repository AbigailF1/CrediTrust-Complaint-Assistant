import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Chroma
import os

# --- 1. Load the Cleaned Data ---
df = pd.read_csv('../data/filtered_complaints.csv')
print("Cleaned dataset loaded successfully.")
print(f"Dataset shape: {df.shape}")

# Drop any rows that might have become empty after cleaning
df.dropna(subset=['cleaned_narrative'], inplace=True)
if 'Product' not in df.columns or 'cleaned_narrative' not in df.columns:
    print("Error: The CSV must contain 'Product' and 'cleaned_narrative' columns.")
    exit()

# --- 2. Text Chunking ---
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=750,
    chunk_overlap=100,
    length_function=len
)

# Process each narrative and create chunks with metadata
all_chunks = []
all_metadatas = []
for index, row in df.iterrows():
    # Split the narrative into chunks
    chunks = text_splitter.split_text(row['cleaned_narrative'])
    
    # Create metadata for each chunk
    for chunk in chunks:
        all_chunks.append(chunk)
        all_metadatas.append({'product': row['Product'], 'source_id': index})

print(f"Total number of text chunks created: {len(all_chunks)}")

# --- 3. Embedding and Vector Store Indexing ---

# Initialize the embedding model
# This will download the model from Hugging Face on its first run
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
print("Embedding model loaded.")

# Define the directory to save the vector store
vector_store_dir = "vector_store"
if not os.path.exists(vector_store_dir):
    os.makedirs(vector_store_dir)
    
print(f"Creating and persisting vector store in '{vector_store_dir}'...")

# Create the Chroma vector store from the chunks and persist it
# This process will take some time as it generates embeddings for all chunks
db = Chroma.from_texts(
    texts=all_chunks,
    embedding=embedding_function,
    metadatas=all_metadatas,
    persist_directory=vector_store_dir
)

print("\nVector store created successfully!")
print(f"Number of vectors in the store: {db._collection.count()}")
print(f"\nTask 2 is complete. The interim submission deliverables are ready.")