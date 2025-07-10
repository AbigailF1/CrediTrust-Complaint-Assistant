# CrediTrust Complaint Assistant (Week 6 – 10 Academy AI Mastery)

## 📌 Project Overview

CrediTrust Financial is a fast-growing digital finance company serving East Africa. To support its internal product, support, and compliance teams, this project builds a **Retrieval-Augmented Generation (RAG)** system to answer product-specific questions based on **real customer complaint narratives**.

The system allows internal users (like Product Managers) to ask plain English questions such as:

> “Why are people unhappy with Buy Now, Pay Later?”

It then:

- Searches complaint narratives using **semantic search**
- Uses a **language model** to generate grounded answers
- Supports multi-product querying across five financial product lines.

---

## 🎯 Objective

Build a RAG-based chatbot system that:

- Allows querying of complaint narratives using plain-English questions
- Retrieves relevant documents using **vector similarity**
- Generates concise and insightful answers using an **LLM**
- Serves insights across 5 product categories:
  - Credit Cards
  - Personal Loans
  - Buy Now, Pay Later (BNPL)
  - Savings Accounts
  - Money Transfers

---

## 🗃️ Project Structure

project-root/
│
├── data/ # Raw and cleaned data (ignored by git)
│ └── complaints.csv
│
├── notebooks/ # EDA, prototyping, visualization
│ └── 1.0-eda.ipynb
│
├── src/ # Source scripts for RAG pipeline
│ ├── preprocess.py
│ ├── embed_and_index.py
│ └── query_engine.py
│
├── vector_store/ # FAISS/ChromaDB persistent index
│
├── requirements.txt # Python dependencies
├── .gitignore
└── README.md # Project documentation

---

## ✅ Setup Instructions

1. **Clone the repository**

```bash
git clone https://github.com/your-username/project-name.git
cd project-name
```

2.**Install dependencies**

```bash
pip install -r requirements.txt

```

3. **Add your data**
   Download the complaint dataset and place it under data/.

4. **Run the pipline**

```bash
python src/embed_and_index.py
python src/query_engine.py
```

**📎Notes**
Large files (data/, vector_store/, **pycache**/) are excluded from version control via .gitignore.

You can switch between FAISS and ChromaDB in the vector store module.

Embedding model: sentence-transformers/all-MiniLM-L6-v2 used for balance between speed and accuracy.
