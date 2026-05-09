# paste the full app.py content here
"""
File QA RAG Chatbot — Fixed Version
Fixes:
  1. Keys loaded from env vars (not hardcoded)
  2. Larger chunks for better context
  3. Similarity search instead of MMR (more relevant results)
  4. Less restrictive prompt — LLM can synthesize across chunks
  5. Stronger model: llama-3.3-70b-versatile
"""

import os
import tempfile
import re
import pandas as pd
import streamlit as st

from langchain_community.vectorstores import FAISS
from langchain_community.chat_message_histories import StreamlitChatMessageHistory
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from operator import itemgetter

# ── Keys: read from env (set via Colab secrets or os.environ before running) ──
HF_TOKEN    = os.environ.get("HF_TOKEN", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

if not GROQ_API_KEY:
    st.error("❌ GROQ_API_KEY not set. Add it to Colab Secrets or os.environ.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="File QA RAG Chatbot", page_icon="🤖", layout="wide")
st.title("📄 File QA RAG Chatbot 🤖")
st.caption("Upload PDFs → Ask questions → Get answers with sources")


def clean_text(text: str) -> str:
    text = re.sub(r"\[span_\d+\]\((start|end)_span\)", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@st.cache_resource
def load_llm():
    return ChatGroq(
        # FIX 1: Use a stronger model that synthesises better across chunks
        model="llama-3.3-70b-versatile",
        api_key=GROQ_API_KEY,
        temperature=0.2,
        streaming=True,
    )


@st.cache_resource(ttl="1h")
def configure_retriever(uploaded_files):
    docs = []
    temp_dir = tempfile.TemporaryDirectory()
    for file in uploaded_files:
        temp_filepath = os.path.join(temp_dir.name, file.name)
        with open(temp_filepath, "wb") as f:
            f.write(file.getvalue())
        loader = PyPDFLoader(temp_filepath)
        raw_docs = loader.load()
        for doc in raw_docs:
            doc.page_content = clean_text(doc.page_content)
        docs.extend(raw_docs)

    if not docs:
        st.error("No content extracted from PDFs.")
        st.stop()

    # FIX 2: Larger chunks so each chunk has more complete context
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,    # was 600 — too small for technical docs
        chunk_overlap=200,  # was 100
    )
    doc_chunks = text_splitter.split_documents(docs)

    embeddings_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"token": HF_TOKEN} if HF_TOKEN else {},
    )
    vectordb = FAISS.from_documents(doc_chunks, embeddings_model)

    # FIX 3: Use plain similarity search (not MMR) for top-k most relevant chunks
    return vectordb.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5},
    )


class PostMessageHandler(BaseCallbackHandler):
    def __init__(self, msg):
        super().__init__()
        self.msg = msg
        self.sources = []

    def on_retriever_end(self, documents, *, run_id, parent_run_id, **kwargs):
        source_ids = []
        for d in documents:
            metadata = {
                "source": d.metadata.get("source", "unknown"),
                "page": d.metadata.get("page", "?"),
                "content": d.page_content[:120],
            }
            if metadata not in self.sources:
                self.sources.append(metadata)
                source_ids.append(metadata)

    def on_llm_end(self, response, *, run_id, parent_run_id, **kwargs):
        if self.sources:
            df = pd.DataFrame(self.sources[:5])
            with self.msg.container():
                st.markdown("**📚 Sources Used:**")
                st.dataframe(df, use_container_width=True)


# ── Sidebar: file upload ──────────────────────────────────────────────────────
with st.sidebar:
    st.header("📂 Upload Documents")
    uploaded_files = st.file_uploader(
        "Choose PDF files", type=["pdf"], accept_multiple_files=True
    )
    st.divider()
    if st.button("🗑️ Clear chat history"):
        st.session_state["langchain_messages"] = []
        st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
if not uploaded_files:
    st.info("👈 Upload one or more PDFs from the sidebar to get started.")
    st.stop()

retriever = configure_retriever(uploaded_files)
llm = load_llm()
msgs = StreamlitChatMessageHistory(key="langchain_messages")

# FIX 4: Prompt is helpful but not over-restrictive
#   Old prompts often say "ONLY use the context, say I don't know otherwise"
#   which causes the LLM to refuse even when chunks are partially relevant.
#   This version asks the LLM to use the context and be transparent when unsure.
SYSTEM_PROMPT = """You are a helpful assistant answering questions about uploaded documents.

Use the retrieved context below to answer the question thoroughly and clearly.
If the context contains relevant information, use it — even if it is partial or spread across multiple chunks.
If you are unsure or the context doesn't cover the question, say so briefly and share what you do know.
Never fabricate facts. Cite the page number when referring to specific content.

Context:
{context}"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("placeholder", "{chat_history}"),
    ("human", "{question}"),
])

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def format_docs(docs):
    return "\n\n---\n\n".join(
        f"[Page {d.metadata.get('page', '?')}]\n{d.page_content}" for d in docs
    )

chain = (
    {
        "context": itemgetter("question") | retriever | format_docs,
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | prompt
    | llm
    | StrOutputParser()
)

# ── Chat UI ───────────────────────────────────────────────────────────────────
for msg in msgs.messages:
    st.chat_message(msg.type).write(msg.content)

if question := st.chat_input("Ask a question about your documents…"):
    st.chat_message("human").write(question)
    msgs.add_user_message(question)

    with st.chat_message("ai"):
        sources_container = st.empty()
        handler = PostMessageHandler(sources_container)

        response = ""
        with st.spinner("Thinking…"):
            for chunk in chain.stream(
                {"question": question, "chat_history": msgs.messages[:-1]},
                config={"callbacks": [handler]},
            ):
                response += chunk

        st.write(response)
        msgs.add_ai_message(response)
