# src/ui_app.py

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
from rag_chatbot import RAGChatbot

# -------------------------
# Initialize chatbot
# -------------------------
chatbot = RAGChatbot()

st.set_page_config(page_title="CrediTrust Consumer Complaints Chatbot", layout="centered")

st.title("📄 CrediTrust Consumer Complaints Chatbot")
st.write("Ask questions about consumer complaints and get summarized answers.")

# Collapsible "How it works" section

with st.expander("How it works"):
    st.markdown(
        """
1. **Ask a question** about consumer complaints.
2. The chatbot **retrieves similar complaints** from the database.
3. The AI **summarizes these complaints** into a clear answer.
4. Retrieved complaint IDs are shown separately for reference.

**Example Question:**  
- "Why are people complaining about credit cards?"  
- "What issues do customers face with personal loans?"  
- "What are the common complaints about savings accounts?"

**Topics you can ask about include:**  
- Credit card issues  
- Personal loans  
- Buy Now, Pay Later (BNPL)  
- Savings accounts  
- Money transfers  

This helps you get quick insights from thousands of consumer complaints in one summarized answer.
"""
    )
# -------------------------
# User query input
# -------------------------
query = st.text_area("Your question:")

if st.button("Submit"):
    if query.strip() == "":
        st.warning("Please enter a question first.")
    else:
        with st.spinner("Generating AI response..."):
            answer, context_docs = chatbot.answer(query)

        st.markdown("### 🤖 AI Response")
        st.success(answer)

        with st.expander("Show Retrieved Complaints / Context"):
            st.write("These complaints are related to the type of issues in your query:")
            for doc in context_docs:
                st.write(f"- {doc}")
