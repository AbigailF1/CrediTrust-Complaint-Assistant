# 🧠 CrediTrust Consumer Complaints Chatbot

A **Retrieval-Augmented Generation (RAG)** chatbot that summarizes consumer complaints about financial products such as credit cards, personal loans, and BNPL (Buy Now, Pay Later).  
It leverages **FAISS** for semantic retrieval and a **free local LLM** for generating natural language responses.

---

## 📘 Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Example Questions](#example-questions)
- [Project Structure](#project-structure)
- [Requirements](#requirements)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)

---

## 🏗️ Project Overview

This project centralizes and summarizes consumer complaints for **CrediTrust**, enabling faster insights for stakeholders.  
Using embeddings and a local LLM, the chatbot provides concise, context-aware answers **without relying on paid APIs**.

---

## ✨ Features

- Summarizes complaints for:
  - Credit Cards
  - Personal Loans
  - BNPL (Buy Now, Pay Later)
  - Savings Accounts
  - Money Transfers  
- Uses **FAISS** vector store for fast semantic search  
- Generates human-readable answers using a **local free LLM (`microsoft/phi-2`)**  
- Interactive **Streamlit UI**  
- Toggle sections for “How It Works” and “Show Retrieved Complaints”

---

## ⚙️ How It Works

1. **Preprocessing** – Raw CSV is filtered, cleaned, and tokenized into text chunks.  
2. **Embedding & Indexing** – Each chunk is embedded using `sentence-transformers/all-MiniLM-L6-v2` and indexed in **FAISS**.  
3. **Query Handling** – User query is embedded and top relevant chunks are retrieved from FAISS.  
4. **Response Generation** – Retrieved context is fed into a local LLM to generate a concise answer.

**Example:**

> **Question:** “Why are people complaining about credit cards?”  
> **AI Response:** “People are complaining about credit cards due to high interest rates, billing errors, and poor customer service.”

**You can ask about:**
- Issues with Credit Cards, Personal Loans, BNPL, Savings Accounts, or Money Transfers.  
- General complaint trends, common problems, or product-specific issues.

---

## 🚀 Getting Started

### 1. Clone the repository
```bash
git clone https://github.com/AbigailF1/CrediTrust-Complaint-Assistant
cd CrediTrust-Complaint-Assistant
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

pip install -r requirements.txt

```

### 3. Run preprocessing to generate filtered complaints CSV

```bash
python src/preprocessing.py
```

### 4. Build FAISS index

```bash
python src/chunking.py
```

### 5. Launch the chatbot UI

```bash
streamlit run src/ui_app.py
```

## 💬 Usage

1. Type your question in the input box.  
2. Click **Submit** to get the AI-generated answer.  
3. Click **Show Retrieved Complaints** to see which complaints contributed to the answer.  

> **Note:** Currently, only complaint IDs and product types are displayed.

---

## 📂 Project Structure
| Path | Description |
|------|--------------|
| `src/rag_chatbot.py` | Core RAG chatbot logic |
| `src/ui_app.py` | Streamlit interface for user interaction |
| `src/preprocessing.py` | Cleans and filters raw complaint data |
| `src/chunking.py` | Handles text chunking and FAISS indexing |
| `vector_store/` | Stores FAISS index and metadata |
| `data/` | Contains raw and processed CSV complaint data |
| `requirements.txt` | List of Python dependencies |
| `README.md` | Project documentation |

---

## 🧩 Requirements

- Python 3.12  
- torch==2.2.2  
- sentence-transformers  
- faiss-cpu  
- transformers  
- streamlit  
- tqdm  

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## 🖼️ Screenshots
Chatbot Interface
<img width="1499" height="945" alt="image_2025-09-12_11-27-03" src="https://github.com/user-attachments/assets/98c7d8bb-779f-46ec-ac33-8d8e4359ae4d" />

How It Works / Example
<img width="1054" height="908" alt="image_2025-09-12_11-27-27" src="https://github.com/user-attachments/assets/73573c1b-8842-4ff5-a8e0-13b7ae5bf060" />

## 🔮 Future Improvements

- Show actual complaint text instead of just IDs  
- Improve UI/UX design for better readability  
- Provide more detailed context summaries automatically  
- Optimize LLM usage for lower CPU/memory requirements 
