# 📄 File QA RAG Chatbot

An end-to-end RAG (Retrieval-Augmented Generation) chatbot that answers questions from uploaded PDFs using LangChain, Groq, FAISS, and Streamlit.

## 🚀 Features
- Upload PDF documents and ask questions about them
- Answers grounded in document content with source citations
- Real-time streaming responses
- Chat history support

## 🛠️ Tech Stack
| Component | Tool |
|---|---|
| LLM | Groq (llama-3.3-70b-versatile) |
| Embeddings | HuggingFace (all-MiniLM-L6-v2) |
| Vector Store | FAISS |
| Framework | LangChain |
| UI | Streamlit |

## ⚙️ Setup

### 1. Install dependencies
pip install langchain-groq langchain-huggingface langchain-community faiss-cpu pypdf sentence-transformers streamlit

### 2. Set environment variables
export GROQ_API_KEY=your_groq_api_key
export HF_TOKEN=your_huggingface_token

### 3. Run the app
streamlit run app.py

## 📁 Project Structure
- app.py — Main Streamlit application
- README.md — Project documentation

## 👤 Author
Varun F Kshatriya
